"""Late captures fill omissions without changing any existing adoption relationship."""

import json
from dataclasses import replace
from io import StringIO
from types import SimpleNamespace

import pytest
from scope_recall.adapters.clients.remote_server import client_times_and_lead
from scope_recall.adapters.hermes.boundary import SourceObservationLedger, sync_assistant_key, sync_turn_source_events
from scope_recall.contracts import ContractError
from scope_recall.core.adoption_schedule import schedule_adoptions
from scope_recall.core.adoption_storage import PreparedAdoption, adoption_context, binding_reason
from scope_recall.core.review_sources import decide_adoption
from scope_recall.core.worker_adoption import advance_adoption, prepare_adoptions

from tests.contract.test_adoption import PLAN, capture, prepared, publish
from tests.contract.test_adoption_late_reply import Model, drain, store_reply
from tests.contract.test_claims import app
from tests.contract.test_review_list import owner as management


def relation(core, ctx, owner):
    with core.storage.read(ctx) as tx:
        return dict(
            tx._check()
            .execute(
                "SELECT * FROM adoption_relations WHERE confirmation_ref=? AND confirmation_revision=?",
                (owner.ref, owner.revision),
            )
            .fetchone()
        )


def completed(core, ctx, owner, *, checkpoint=False, state="done"):
    """Represent a completed ordinary extraction, including older checkpoint-free stores."""
    with core.storage.write(ctx) as tx:
        conn = tx._check(write=True)
        conn.execute(
            "UPDATE work_items SET state=? WHERE work_type='consolidate' AND subject_ref=?", (state, owner.ref)
        )
        if checkpoint:
            row = relation(core, ctx, owner)
            work_id = conn.execute(
                "SELECT work_id FROM work_items WHERE work_type='consolidate' AND subject_ref=?", (owner.ref,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO adoption_processing VALUES(?,?,?,?,?,?)",
                (
                    work_id,
                    owner.ref,
                    owner.revision,
                    row["assistant_ref"],
                    row["assistant_revision"],
                    json.dumps(dict(stage="done")),
                ),
            )


def accepted(core, ctx, owner):
    rows, _ = prepared(core, ctx, owner)
    result = publish(core, ctx, rows[0])
    assert result.state == "active"
    completed(core, ctx, owner)
    return result


def pending_review(core, ctx, owner):
    rows, _ = prepared(core, ctx, owner)
    publish(core, ctx, replace(rows[0], reason="ambiguous_reference"))


def adoption_state(core, ctx, owner):
    """Capture every persisted adoption receipt and its owner-visible review items."""
    with core.storage.read(ctx) as tx:
        conn = tx._check()
        return dict(
            relation=relation(core, ctx, owner),
            # Entity resolution is a derived worker cache, independent of fact version history.
            versions=[
                {key: row[key] for key in row.keys() if key != "entity_resolved"}
                for row in conn.execute("SELECT * FROM claim_versions ORDER BY claim_id,revision")
            ],
            checkpoints=[tuple(row) for row in conn.execute("SELECT * FROM adoption_processing ORDER BY work_id")],
            work=[
                tuple(row)
                for row in conn.execute(
                    "SELECT * FROM work_items WHERE work_type='consolidate' AND subject_ref=?", (owner.ref,)
                )
            ],
            reviews=management((core, ctx)).reviews(source="adoption")["items"],
        )


@pytest.mark.parametrize("state", ["published", "pending", "rejected", "approved", "undo", "retracted"])
def test_late_capture_never_changes_existing_relationship(app, state):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-A")
    owner = capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:04Z")
    api = management(app)
    if state in {"published", "retracted"}:
        claim = accepted(core, ctx, owner)
        if state == "retracted":
            api.revise(claim.ref, expected_revision=claim.revision, value=None)
    else:
        pending_review(core, ctx, owner)
        completed(core, ctx, owner)
        if state != "pending":
            item = api.reviews(source="adoption")["items"][0]
            receipt = api.review_decide(
                item["key"],
                "reject" if state == "rejected" else "approve",
                expected_version=item["version"],
                selection=dict(fragment_ids=[2]),
            )
            if state == "undo":
                api.review_undo(receipt["action_id"], expected_version=receipt["version"])
            elif state == "approved":
                drain(core, ctx)
    before = adoption_state(core, ctx, owner)
    store_reply(core, ctx, source_event_key="TEST-B", recorded_at="2026-09-01T12:00:03Z")
    assert adoption_state(core, ctx, owner) == before
    judge = Model()
    drain(core, ctx, judge)
    assert not judge.calls
    assert adoption_state(core, ctx, owner) == before


def assert_preserved_after_late(core, ctx, owner, claim, before):
    current = core.current_claim(ctx, claim.ref)
    assert current is not None and current.state == "active" and current.revision == claim.revision
    assert adoption_state(core, ctx, owner) == before
    judge = Model()
    drain(core, ctx, judge)
    assert not judge.calls
    assert adoption_state(core, ctx, owner) == before


def test_relation_without_checkpoint_is_authority_for_unchanged_parent(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-A", recorded_at="2026-09-01T12:00:01Z")
    plan = store_reply(core, ctx, source_event_key="TEST-A", source_revision=3, recorded_at="2026-09-01T12:00:03Z")
    owner = capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:04Z")
    claim = accepted(core, ctx, owner)
    before = relation(core, ctx, owner)
    store_reply(core, ctx, source_event_key="TEST-A", source_revision=2, recorded_at="2026-09-01T12:00:02Z")
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant_revision == plan.revision
        assert tx._check().execute("SELECT count(*) FROM adoption_processing").fetchone()[0] == 0
    assert core.current_claim(ctx, claim.ref).state == "active"
    assert core.current_claim(ctx, claim.ref).revision == claim.revision
    assert relation(core, ctx, owner) == before


@pytest.mark.parametrize("state", ["done", "obsolete"])
def test_late_predecessor_cannot_override_explicit_rejection(app, state):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-A")
    owner = capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:04Z")
    pending_review(core, ctx, owner)
    with core.storage.write(ctx) as tx:
        decide_adoption(tx, core.clock, ["adoption", owner.ref, owner.revision], "reject", None)
    completed(core, ctx, owner, checkpoint=True, state=state)
    before = relation(core, ctx, owner)
    store_reply(core, ctx, source_event_key="TEST-B", recorded_at="2026-09-01T12:00:03Z")
    assert relation(core, ctx, owner) == before
    judge = Model()
    drain(core, ctx, judge)
    assert not judge.calls
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT count(*) FROM claim_versions").fetchone()[0] == 0


def test_evicted_unpersisted_witness_does_not_change_existing_relationship(app):
    core, ctx = app
    ledger = SourceObservationLedger()
    key = sync_assistant_key(ctx, ctx.session_id, "TEST-A")
    ledger.witness((key, 1), "2026-09-01T12:00:01Z", "2026-09-01T12:00:01Z")
    store_reply(core, ctx, source_event_key="TEST-B", recorded_at="2026-09-01T12:00:02Z")
    owner = capture(core, ctx, "好，就用方案二", when="2026-09-01T12:00:04Z")
    claim = accepted(core, ctx, owner)
    before = adoption_state(core, ctx, owner)
    for index in range(4096):
        ledger.witness((f"TEST-other-{index}", 1), "2026-09-01T12:00:02Z", None)
    events, _ = sync_turn_source_events(
        ledger,
        ctx,
        session_id=ctx.session_id,
        turn_id="TEST-A",
        user_content="",
        assistant_content=PLAN,
        recorded_at="2026-09-01T12:00:03Z",
        outcome="success",
        include_user=False,
    )
    assert events[0][0]["recorded_at"] == "2026-09-01T12:00:03Z"
    core.record_event(replace(ctx, actor_origin="assistant_visible"), events[0][0], scope_id="TEST-scope")
    assert_preserved_after_late(core, ctx, owner, claim, before)


def test_remote_recalibration_does_not_change_existing_relationship(app):
    core, ctx = app

    def adjusted(observed, server):
        return client_times_and_lead(
            dict(
                payload=dict(session_id="TEST-remote"),
                observed_at=f"2026-09-01T12:00:{observed}Z",
                clock=dict(client_at=f"2026-09-01T12:00:{observed}Z", server_at=f"2026-09-01T12:00:{server}Z"),
            )
        )[0]

    a_time, b_time, c_time = adjusted("01", "03"), adjusted("02", "02.1"), adjusted("04", "04.1")
    assert b_time < a_time < c_time
    store_reply(core, ctx, source_event_key="TEST-B", recorded_at=b_time)
    owner = capture(core, ctx, "好，就用方案二", when=c_time)
    claim = accepted(core, ctx, owner)
    before = adoption_state(core, ctx, owner)
    store_reply(core, ctx, source_event_key="TEST-A", recorded_at=a_time)
    assert_preserved_after_late(core, ctx, owner, claim, before)


def test_remote_recalibration_can_invert_the_first_binding(app):
    core, ctx = app
    moments = [
        client_times_and_lead(
            dict(
                payload=dict(session_id="TEST-remote"),
                observed_at=f"2026-09-01T12:00:{observed}Z",
                clock=dict(client_at=f"2026-09-01T12:00:{observed}Z", server_at=f"2026-09-01T12:00:{server}Z"),
            )
        )[0]
        for observed, server in (("01", "03"), ("02", "02.1"), ("04", "04.1"))
    ]
    first = store_reply(core, ctx, source_event_key="TEST-A", content="Plan A: keep red.", recorded_at=moments[0])
    store_reply(core, ctx, source_event_key="TEST-B", content="Plan B: keep blue.", recorded_at=moments[1])
    owner = capture(core, ctx, "Yes, use that plan.", when=moments[2])
    # Recalibrated times do not carry the true host sequence in their envelopes.
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant_ref == first.ref
    claim = english_decision(core, ctx, owner, "Plan A", "keep red.")
    assert claim.state == "active"
    assert first.ref in {span["source_ref"] for span in core.current_claim(ctx, claim.ref).payload["evidence_spans"]}


def english_decision(core, ctx, owner, label, decision, topic="TEST palette"):
    """Run the normal three preparation stages with fixed synthetic responses."""

    def judge(messages, **kwargs):
        body = json.loads(messages[-1]["content"])
        if "proposal" in body:
            result = dict(support="supported", action="new", target_ref=None, target_revision=None)
        elif body["task"] == "locate_adoption":
            result = dict(adopted=True, fragment_ids=[1], scope="whole", conditions=[], reference_label=label)
        else:
            result = dict(topic=topic, decision=decision)
        return json.dumps(result)

    rows = prepare_adoptions(core.storage, ctx, (owner,), judge, clock=core.clock, remaining_seconds=10)
    assert len(rows) == 1 and rows[0].reason is None
    return publish(core, ctx, rows[0])


def test_late_different_topic_cannot_offer_rebinding_or_publish_a_second_fact(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-A", content="Plan A: keep red.")
    owner = capture(core, ctx, "Yes, use that plan.", when="2026-09-01T12:00:04Z")
    claim = english_decision(core, ctx, owner, "Plan A", "keep red.")
    completed(core, ctx, owner)
    before = adoption_state(core, ctx, owner)
    store_reply(core, ctx, content="Plan B: keep blue.", recorded_at="2026-09-01T12:00:03Z")
    assert adoption_state(core, ctx, owner) == before
    api = management(app)
    assert api.reviews(source="adoption")["items"] == []
    judge = Model()
    drain(core, ctx, judge)
    assert not judge.calls
    assert core.current_claim(ctx, claim.ref).revision == 1
    assert adoption_state(core, ctx, owner) == before


def test_historical_fact_approval_is_irrelevant_to_late_capture(app):
    core, ctx = app
    api = management(app)
    api.set_review_policy("TEST-scope", True, expected_version=0)
    store_reply(core, ctx, source_event_key="TEST-A0", content="Plan A: keep red.")
    first = capture(core, ctx, "Yes, use that plan.", when="2026-09-01T12:00:02Z")
    old = english_decision(core, ctx, first, "Plan A", "keep red.")
    item = api.reviews(source="fact")["items"][0]
    api.review_decide(item["key"], "approve", expected_version=item["version"])
    completed(core, ctx, first)
    store_reply(
        core, ctx, source_event_key="TEST-A1", content="Plan B: keep green.", recorded_at="2026-09-01T12:00:03Z"
    )
    later = capture(core, ctx, "Yes, use that plan.", when="2026-09-01T12:00:06Z")
    held = english_decision(core, ctx, later, "Plan B", "keep green.")
    assert held.ref == old.ref and held.state == "proposed"
    completed(core, ctx, later)
    before = adoption_state(core, ctx, later)
    store_reply(core, ctx, content="Plan C: keep blue.", recorded_at="2026-09-01T12:00:05Z")
    assert adoption_state(core, ctx, later) == before
    judge = Model()
    drain(core, ctx, judge)
    assert not judge.calls
    assert adoption_state(core, ctx, later) == before
    item = api.reviews(source="fact")["items"][0]
    with pytest.raises(ContractError, match="VERSION_CONFLICT: invalid adoption_dependencies"):
        api.review_decide(item["key"], "approve", expected_version=item["version"])
    assert adoption_state(core, ctx, later) == before


def test_old_parent_edit_uses_existing_source_invalidation_without_rebinding(app):
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-A", content="Plan A: keep red.")
    owner = capture(core, ctx, "Yes, use that plan.", when="2026-09-01T12:00:04Z")
    claim = english_decision(core, ctx, owner, "Plan A", "keep red.")
    completed(core, ctx, owner)
    before = adoption_state(core, ctx, owner)
    store_reply(core, ctx, content="Plan B: keep blue.", recorded_at="2026-09-01T12:00:03Z")
    assert adoption_state(core, ctx, owner) == before
    store_reply(
        core,
        ctx,
        source_event_key="TEST-A",
        source_revision=2,
        content="Plan A: keep yellow.",
        recorded_at="2026-09-01T12:00:05Z",
    )
    row = relation(core, ctx, owner)
    assert (row["state"], row["reason"]) == ("pending_review", "source_changed")
    assert row["assistant_ref"] == before["relation"]["assistant_ref"]
    assert core.current_claim(ctx, claim.ref).revision == claim.revision
    assert management(app).reviews(source="adoption")["items"] == []


@pytest.mark.parametrize("index", ["source_session", "source_adoption_order"])
@pytest.mark.parametrize("budget", [None, 1.0])
def test_missing_order_index_runs_explicit_maintenance_before_model_work(app, monkeypatch, index, budget):
    from scope_recall.core import storage
    from scope_recall.runtime import worker_entry
    from scope_recall.runtime.store_maintenance import store_upgrade_pending

    core, ctx = app
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute(f"DROP INDEX {index}")
    monkeypatch.setattr(storage, "HEAVY_UPGRADE_BYTES", 0)
    with core.storage.write(ctx, remaining_seconds=budget) as tx:
        assert tx._check().execute("SELECT 1 FROM sqlite_master WHERE name=?", (index,)).fetchone() is None
    assert store_upgrade_pending(ctx.binding)
    monkeypatch.setattr(
        worker_entry, "load_config", lambda *args: SimpleNamespace(binding=ctx.binding, owner_id="TEST-owner")
    )
    monkeypatch.setattr(
        worker_entry, "build_runtime_instance", lambda *args: pytest.fail("model work before maintenance")
    )
    output = StringIO()
    assert worker_entry.run_worker("TEST-unused", output=output) == 0
    assert json.loads(output.getvalue())["job_kind"] == "upgrade"
    assert not store_upgrade_pending(ctx.binding)
    with core.storage.read(ctx) as tx:
        assert (
            tx._check()
            .execute(
                "SELECT count(*) FROM sqlite_master WHERE type='index' "
                "AND name IN ('source_session','source_adoption_order')"
            )
            .fetchone()[0]
            == 2
        )


def test_an_edited_confirmation_is_scheduled_again_after_its_first_version_was_adopted(app):
    """A relationship is final for its own confirmation version; an edit is a new version, as before."""
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-A")
    owner = capture(core, ctx, "好，就用方案二", key="TEST-C", when="2026-09-01T12:00:04Z")
    accepted(core, ctx, owner)
    edited = capture(core, ctx, "好，就用方案二吧", key="TEST-C", revision=2, when="2026-09-01T12:00:04Z")
    assert edited.ref == owner.ref and edited.revision == 2
    completed(core, ctx, edited)
    with core.storage.write(ctx) as tx:
        schedule_adoptions(tx, (f"{edited.ref}@{edited.revision}",), now=core.clock.utc_now())
    with core.storage.read(ctx) as tx:
        rows = (
            tx._check()
            .execute("SELECT confirmation_revision FROM adoption_processing WHERE confirmation_ref=?", (edited.ref,))
            .fetchall()
        )
    assert [row[0] for row in rows] == [2]


@pytest.mark.parametrize("owner_reviewed", [False, True])
def test_prepared_adoption_cannot_revive_a_fact_after_owner_undo(app, owner_reviewed):
    """An in-flight paraphrase cannot bypass a fact Undo when review policy is disabled."""
    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-A", content="Plan A: keep red.")
    owner = capture(core, ctx, "Yes, use that plan.", when="2026-09-01T12:00:04Z")
    api = management(app)
    with core.storage.read(ctx) as tx:
        snapshot = adoption_context(tx, owner)
    selection = dict(adopted=True, fragment_ids=[1], scope="whole", conditions=[], reference_label="Plan A")
    if owner_reviewed:
        publish(core, ctx, PreparedAdoption(snapshot, selection, reason="ambiguous_reference"))
        item = api.reviews(source="adoption")["items"][0]
        api.review_decide(item["key"], "approve", expected_version=item["version"], selection=dict(fragment_ids=[1]))
        selection = json.loads(relation(core, ctx, owner)["selection_json"])

    def prepare(decision):
        def judge(messages, **kwargs):
            body = json.loads(messages[-1]["content"])
            return json.dumps(
                dict(support="supported", action="new", target_ref=None, target_revision=None)
                if "proposal" in body
                else dict(topic="TEST palette", decision=decision)
            )

        state = dict(stage="describe", selection=selection)
        while state["stage"] != "done":
            state, result = advance_adoption(
                core.storage, ctx, snapshot, state, judge, clock=core.clock, remaining_seconds=10
            )
        assert result is not None and result.support.state == "supported"
        return result

    first, stale = prepare("keep red."), prepare("use red for this plan.")
    api.set_review_policy("TEST-scope", True, expected_version=0)
    claim = publish(core, ctx, first)
    assert claim.state == "proposed"
    item = api.reviews(source="fact")["items"][0]
    receipt = api.review_decide(item["key"], "approve", expected_version=item["version"])
    assert core.current_claim(ctx, claim.ref).state == "active"
    api.review_undo(receipt["action_id"], expected_version=receipt["version"])
    assert core.current_claim(ctx, claim.ref) is None
    api.set_review_policy("TEST-scope", False, expected_version=1)
    before = adoption_state(core, ctx, owner)

    assert publish(core, ctx, stale) is None
    with core.storage.read(ctx) as tx:
        assert binding_reason(tx, snapshot, stale.selection) == "owner_review_undo"
    assert core.current_claim(ctx, claim.ref) is None
    assert adoption_state(core, ctx, owner) == before

    store_reply(core, ctx, content="Plan B: keep blue.", recorded_at="2026-09-01T12:00:05Z")
    later = capture(core, ctx, "Yes, use Plan B.", when="2026-09-01T12:00:06Z")
    fresh = english_decision(core, ctx, later, "Plan B", "keep blue.")
    assert fresh.ref == claim.ref and fresh.state == "active"
