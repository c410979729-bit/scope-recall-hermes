"""Adoption follows witnessed conversation order across durable ingress."""

import itertools
import json
from dataclasses import replace

import pytest
from scope_recall.contracts import ImportProvenance, TrustedSourcePrincipal, import_source_fingerprint
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.core.adoption_storage import adoption_context
from scope_recall.core.capture_inbox import enqueue, replay_inbox
from scope_recall.core.worker_adoption import prepare_adoptions, publish_adoption
from v11_support import source_event

from tests.contract.test_adoption import PLAN, capture, model
from tests.contract.test_claims import app
from tests.contract.test_shared_store import shared, shared_context


class Model:
    """Synthetic consolidation and adoption responses without external calls."""

    def __init__(self):
        self.judge, self.calls = model()

    def propose(self, sources, **kwargs):
        return json.dumps(
            dict(
                protocol_version="1.1",
                source_refs=[f"{s.ref}@{s.revision}" for s in sources],
                claim_proposals=[],
                resume_proposals=[],
                reference_proposals=[],
            )
        )


def reply_event(**changes):
    """A reply witnessed before its owner's later confirmation."""
    return source_event(
        **dict(
            source_event_key="TEST-late-reply",
            origin="assistant_visible",
            role="assistant",
            content=PLAN,
            occurred_at="2026-09-01T12:00:01Z",
            recorded_at="2026-09-01T12:00:01Z",
        )
        | changes
    )


def replay(core, ctx):
    """Use the real inbox commit with synthetic host authorization."""
    return replay_inbox(core.storage, core.clock, ctx, authorize=lambda _: ctx.allowed_scope_ids, remaining_seconds=10)


def active_adoptions(core, ctx):
    with core.storage.read(ctx) as tx:
        return [
            tx.claims.version(row[0], row[1])
            for row in tx._check().execute(
                "SELECT claim_ref,claim_revision FROM adoption_relations WHERE state='bound'"
            )
        ]


@pytest.mark.parametrize("queued", [False, True])
def test_confirmation_consolidated_before_late_reply_is_adopted(app, queued):
    core, ctx = app
    capture(core, ctx, "给我两个配置方案", when="2026-09-01T12:00:00Z")
    event = reply_event()
    actor = replace(ctx, actor_origin="assistant_visible")
    if queued:
        enqueue(core.storage, core.clock, actor, event, scope_id="TEST-scope", host_scope=None)
    owner = capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:02Z")
    judge = Model()
    core.drain_worker(ctx, consolidation=judge, max_items=16, remaining_seconds=20)
    if queued:
        assert replay(core, ctx)[0].durability == "persisted"
    else:
        assert core.record_event(actor, event, scope_id="TEST-scope").durability == "persisted"
    core.drain_worker(ctx, consolidation=judge, max_items=16, remaining_seconds=20)
    rows = active_adoptions(core, ctx)
    assert len(rows) == 1
    assert rows[0].state == "active"
    spans = rows[0].payload["evidence_spans"]
    assert len(spans) == 2 and owner.ref in {s["source_ref"] for s in spans}


def store_reply(core, ctx, **changes):
    """Persist one synthetic assistant occurrence or revision."""
    event = reply_event(**changes)
    receipt = core.record_event(replace(ctx, actor_origin=event["origin"]), event, scope_id="TEST-scope")
    assert receipt.durability == "persisted"
    return core.source(ctx, receipt.event_refs[0].ref, event["source_revision"])


def confirm(core, ctx, **changes):
    return capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:02Z", **changes)


def drain(core, ctx, judge=None):
    return core.drain_worker(ctx, consolidation=judge or Model(), max_items=16, remaining_seconds=20)


def test_unknown_host_time_does_not_retract_existing_adoption(app):
    core, ctx = app
    store_reply(core, ctx)
    confirm(core, ctx)
    drain(core, ctx)
    original = active_adoptions(core, ctx)[0]
    capture(
        core, ctx, "TEST 继续处理重定向消息", key="TEST-undated-redirect", when=None, recorded_at="2026-09-06T12:00:01Z"
    )
    current = active_adoptions(core, ctx)
    assert len(current) == 1 and current[0].state == "active"
    assert (current[0].ref, current[0].revision) == (original.ref, original.revision)


def test_replaying_same_reply_twice_preserves_completed_checkpoint(app):
    core, ctx = app
    owner = confirm(core, ctx)
    drain(core, ctx)
    actor = replace(ctx, actor_origin="assistant_visible")
    judge = Model()
    for _ in range(2):
        enqueue(core.storage, core.clock, actor, reply_event(), scope_id="TEST-scope", host_scope=None)
        assert replay(core, ctx)[0].durability == "persisted"
        drain(core, ctx, judge)
    assert len(active_adoptions(core, ctx)) == 1
    assert sum(call.get("task") == "locate_adoption" for call in judge.calls) == 1
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT count(*) FROM claim_versions").fetchone()[0] == 1
        assert (
            tx._check()
            .execute("SELECT count(*) FROM adoption_processing WHERE confirmation_ref=?", (owner.ref,))
            .fetchone()[0]
            == 1
        )


@pytest.mark.parametrize(
    "reply_changes,owner_time",
    [
        ({}, "2026-09-01T12:00:01Z"),
        ({"occurred_at": None, "time_precision": "unknown"}, "2026-09-01T12:00:02Z"),
        ({}, None),
        ({"time_precision": "day"}, "2026-09-01T12:00:02Z"),
        ({"occurred_at": "2026-09-01T12:00:01.0001Z"}, "2026-09-01T12:00:01.0002Z"),
    ],
)
def test_host_time_does_not_override_recorded_order(app, reply_changes, owner_time):
    core, ctx = app
    store_reply(core, ctx, **reply_changes)
    owner = capture(core, ctx, "好，就用方案二", when=owner_time, recorded_at="2026-09-01T12:00:02Z")
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant is not None
    drain(core, ctx)
    assert len(active_adoptions(core, ctx)) == 1


@pytest.mark.parametrize("origin", ["imported", "memory_reinjection", "origin_unknown", "host_generated"])
@pytest.mark.parametrize("revised", [False, True])
def test_nonconversational_source_cannot_supply_predecessor(app, origin, revised):
    core, ctx = app
    event = reply_event(origin=origin)
    if revised:
        store_reply(core, ctx)
        event["source_revision"] = 2
    provenance = None
    if origin == "imported":
        event.update(source_original_origin="assistant_visible", dataset_id="TEST-import")
        provenance = ImportProvenance("assistant_visible", "a" * 64, frozenset({import_source_fingerprint(event)}))
    actor = replace(ctx, actor_origin=origin, import_provenance=provenance)
    assert core.record_event(actor, event, scope_id="TEST-scope").durability == "persisted"
    owner = confirm(core, ctx)
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner) is None
    assert not prepare_adoptions(core.storage, ctx, (owner,), Model().judge, clock=core.clock)


def test_edit_cannot_move_old_occurrence_in_front_of_confirmation(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-old", occurred_at="2026-09-01T12:00:00Z")
    expected = store_reply(core, ctx)
    store_reply(
        core,
        ctx,
        source_event_key="TEST-old",
        source_revision=2,
        occurred_at="2026-09-01T12:00:01.5Z",
        recorded_at="2026-09-01T12:00:01.5Z",
    )
    owner = confirm(core, ctx)
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant_ref == expected.ref


def test_waiting_second_reply_blocks_binding_to_first(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-old", occurred_at="2026-09-01T12:00:00Z", content="更早的建议")
    actor = replace(ctx, actor_origin="assistant_visible")
    enqueue(core.storage, core.clock, actor, reply_event(), scope_id="TEST-scope", host_scope=None)
    owner = confirm(core, ctx)
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner) is None
    drain(core, ctx)
    assert not active_adoptions(core, ctx)
    replay(core, ctx)
    drain(core, ctx)
    assert len(active_adoptions(core, ctx)) == 1


def test_unqueued_second_reply_preserves_published_relationship(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-old", occurred_at="2026-09-01T12:00:00Z")
    owner = confirm(core, ctx)
    drain(core, ctx)
    previous = active_adoptions(core, ctx)[0]
    store_reply(core, ctx, content="方案甲：热更新。\n\n方案乙：先验证再发布。")
    with core.storage.read(ctx) as tx:
        head = tx.claims.version(previous.ref, previous.revision)
        current = tx.claims.version(previous.ref, head.current_revision)
        assert current.state == "active" and current.revision == previous.revision
        assert (
            tx._check()
            .execute("SELECT state FROM work_items WHERE subject_ref=? AND work_type='consolidate'", (owner.ref,))
            .fetchone()[0]
            == "done"
        )
    assert len(active_adoptions(core, ctx)) == 1
    assert previous.ref in core.session_brief(replace(ctx, session_id="TEST-next"))["text"]
    judge = Model()
    drain(core, ctx, judge)
    assert not judge.calls
    assert len(active_adoptions(core, ctx)) == 1
    assert core.current_claim(ctx, previous.ref).state == "active"


def test_prepared_wrong_binding_cannot_publish_after_late_reply(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-old", occurred_at="2026-09-01T12:00:00Z")
    owner = confirm(core, ctx)
    prepared = prepare_adoptions(core.storage, ctx, (owner,), Model().judge, clock=core.clock, remaining_seconds=10)[0]
    store_reply(core, ctx)
    with core.storage.write(ctx) as tx:
        assert publish_adoption(tx, prepared, now=core.clock.utc_now()) is None
        assert tx._check().execute("SELECT count(*) FROM adoption_relations").fetchone()[0] == 0


@pytest.mark.parametrize("mode", ["delete", "suppress"])
def test_unavailable_confirmation_does_not_resume_on_late_reply(app, mode):
    from tests.contract.test_deletion import authorize, request

    core, ctx = app
    owner = confirm(core, ctx)
    drain(core, ctx)
    authorize(core, ctx, owner, mode=mode)
    core.forget(ctx, request(owner, mode=mode), remaining_seconds=10)
    store_reply(core, ctx)
    judge = Model()
    drain(core, ctx, judge)
    assert not active_adoptions(core, ctx) and not judge.calls


@pytest.mark.parametrize("state", ["paused", "resumed", "private"])
def test_late_inbox_capture_respects_entry_and_private_fences(app, state):
    from scope_recall.core.management import MemoryManagement
    from scope_recall.core.restore import InstallationMaintenance

    core, ctx = app
    actor = replace(ctx, actor_origin="assistant_visible")
    enqueue(core.storage, core.clock, actor, reply_event(), scope_id="TEST-scope", host_scope=None)
    confirm(core, ctx)
    drain(core, ctx)
    management = MemoryManagement(core, InstallationMaintenance(ctx))
    if state != "private":
        management.set_entry_state("local", "paused", actor="TEST-owner")
        if state == "resumed":
            management.set_entry_state("local", "normal", actor="TEST-owner")
    else:
        management.private_on(target="session", session=ctx.session_id)
    receipts = replay(core, ctx)
    assert (receipts[0].durability == "persisted") == (state == "private")
    judge = Model()
    drain(core, ctx, judge)
    assert not judge.calls
    assert not active_adoptions(core, ctx)


def test_old_worker_checkpoint_cannot_publish_after_parent_changes(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-old", occurred_at="2026-09-01T12:00:00Z")
    confirm(core, ctx)
    judge = Model()
    original = judge.judge
    late = []

    def interrupted(messages, **kwargs):
        if not late and json.loads(messages[-1]["content"]).get("task") == "describe_adoption":
            late.append(store_reply(core, ctx))
        return original(messages, **kwargs)

    judge.judge = interrupted
    drain(core, ctx, judge)
    drain(core, ctx, judge)
    assert late and not active_adoptions(core, ctx)
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT assistant_ref FROM adoption_processing").fetchone()[0] != late[0].ref


@pytest.fixture
def accounts(shared):
    """Two synthetic speakers share one entry, with separate ownership proofs."""
    _storage, binding = shared
    grants = {"TEST-human-A": "TEST-owner-A", "TEST-human-B": "TEST-owner-B", "TEST-assistant-A": "TEST-owner-A"}
    ctx = replace(
        shared_context(binding, "agent3"),
        source_principal=TrustedSourcePrincipal("human", "verified", "TEST-human-A"),
        owner_resolver=lambda entry, principal, memo: grants.get(principal.get("principal_ref")) if principal else None,
    )
    core = MemoryCore(CoreConfig(binding))
    from tests.contract.test_claims import Clock

    core.clock = Clock()
    core.test_sequence = itertools.count(1)
    return core, ctx, grants


@pytest.mark.parametrize("account", ["A", "B"])
def test_shared_entry_late_reply_keeps_speaker_ownership(accounts, account):
    core, ctx, _grants = accounts
    human = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", f"TEST-human-{account}"))
    confirm(core, human)
    drain(core, ctx)
    actor = replace(
        ctx,
        actor_origin="assistant_visible",
        source_principal=TrustedSourcePrincipal("assistant", "verified", "TEST-assistant-A"),
    )
    enqueue(core.storage, core.clock, actor, reply_event(), scope_id="TEST-scope", host_scope=None)
    replay(core, ctx)
    drain(core, ctx)
    assert len(active_adoptions(core, ctx)) == (1 if account == "A" else 0)


def test_owner_change_during_preparation_revokes_adoption(accounts):
    core, ctx, grants = accounts
    actor = replace(
        ctx,
        actor_origin="assistant_visible",
        source_principal=TrustedSourcePrincipal("assistant", "verified", "TEST-assistant-A"),
    )
    core.record_event(actor, reply_event(), scope_id="TEST-scope")
    owner = confirm(core, ctx)
    prepared = prepare_adoptions(core.storage, ctx, (owner,), Model().judge, clock=core.clock, remaining_seconds=10)[0]
    grants["TEST-human-A"] = "TEST-owner-B"
    with core.storage.write(ctx) as tx:
        assert publish_adoption(tx, prepared, now=core.clock.utc_now()) is None
    assert not active_adoptions(core, ctx)


@pytest.mark.parametrize("edit_first", [False, True])
def test_revision_tied_with_confirmation_uses_insertion_order(app, edit_first):
    core, ctx = app
    plan = store_reply(core, ctx)
    if edit_first:
        store_reply(core, ctx, source_revision=2, recorded_at="2026-09-01T12:00:02Z")
    owner = capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:02Z", recorded_at="2026-09-01T12:00:02Z")
    if not edit_first:
        store_reply(core, ctx, source_revision=2, recorded_at="2026-09-01T12:00:02Z")
    with core.storage.read(ctx) as tx:
        snapshot = adoption_context(tx, owner)
        assert snapshot.assistant_ref == plan.ref and snapshot.assistant_revision == (2 if edit_first else 1)
    drain(core, ctx)
    assert bool(active_adoptions(core, ctx)) == edit_first


def test_revision_selection_uses_recorded_confirmation_time(app):
    core, ctx = app
    plan = store_reply(core, ctx)
    store_reply(core, ctx, source_revision=2, occurred_at="2026-09-01T12:00:03Z", recorded_at="2026-09-01T12:00:03Z")
    owner = capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:02Z", recorded_at="2026-09-01T12:00:04Z")
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant_revision == plan.revision + 1
    drain(core, ctx)
    assert len(active_adoptions(core, ctx)) == 1


def test_two_replies_at_same_recorded_instant_use_insertion_order(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-first")
    expected = store_reply(core, ctx)
    owner = confirm(core, ctx)
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant_ref == expected.ref


def test_late_reply_cannot_replace_published_fact(app):
    core, ctx = app
    old = "旧提案甲：覆盖。\n\n旧提案乙：先备份配置，再平滑重启，失败时恢复备份。"
    store_reply(core, ctx, source_event_key="TEST-old", occurred_at="2026-09-01T12:00:00Z", content=old)
    owner = confirm(core, ctx)
    judge = Model()
    original = judge.judge

    def choose_old(messages, **kwargs):
        body = json.loads(messages[-1]["content"])
        if body.get("task") == "locate_adoption":
            return json.dumps(
                dict(adopted=True, fragment_ids=[2], scope="whole", conditions=[], reference_label="旧提案乙")
            )
        return original(messages, **kwargs)

    judge.judge = choose_old
    drain(core, ctx, judge)
    old_claim = active_adoptions(core, ctx)[0]
    new_parent = store_reply(core, ctx)
    drain(core, ctx)
    claims = active_adoptions(core, ctx)
    assert len(claims) == 1 and claims[0].ref == old_claim.ref
    assert core.current_claim(ctx, old_claim.ref).state == "active"
    with core.storage.read(ctx) as tx:
        history = tx.claims.versions(old_claim.ref)
        assert len(history) == 1 and history[0].state == "active"
        row = (
            tx._check().execute("SELECT state,reason,assistant_ref,confirmation_ref FROM adoption_relations").fetchone()
        )
        assert (row["state"], row["reason"]) == ("bound", None)
        assert row["confirmation_ref"] == owner.ref
        assert row["assistant_ref"] != new_parent.ref
