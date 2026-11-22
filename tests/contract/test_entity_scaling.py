"""Name-level work, indexed nomination and the default runtime identity boundary."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import Context
from dataclasses import replace
from difflib import SequenceMatcher
from functools import partial
from types import SimpleNamespace

import pytest
from scope_recall.contracts import ContractError
from scope_recall.core.claims import Qualification
from scope_recall.core.entity_resolution import queue_entity_backfill
from scope_recall.core.read_views import resolve_subject
from scope_recall.core.worker import WorkerConfig, drain_worker
from scope_recall.core.worker_projection import process_rebuild_projection
from scope_recall.runtime.instance import build_runtime_instance
from scope_recall.runtime.instance_config import RuntimeInstanceConfig
from scope_recall.runtime.worker_entry import _drain_once

from tests.contract.test_aliases import _alias, _identity, app
from tests.contract.test_claims import accept, capture, draft
from tests.contract.test_entity_resolution import RecordedModel, drain_entities, fact, merge, withdraw


def named_fact(core, ctx, name, predicate):
    source = capture(core, ctx, f"{name} {predicate} TEST-value.")
    receipt = accept(
        core,
        ctx,
        draft(
            source,
            "TEST-value",
            kind="fact",
            subject=name,
            predicate=predicate,
            statement_kind="assertion",
            entity_names=[name],
        ),
    )
    assert receipt.items[0].state == "active"
    return core.current_claim(ctx, receipt.items[0].ref)


def test_candidate_scoring_is_bounded_before_python_similarity(app, monkeypatch):
    import difflib

    core, ctx = app
    for number in range(180):
        fact(core, ctx, f"TEST-person-{number:03d}", "category", "person")
    calls = []

    def score(*args, **kwargs):
        calls.append(1)
        return SequenceMatcher(*args, **kwargs)

    monkeypatch.setattr(difflib, "SequenceMatcher", score)
    monkeypatch.setattr("scope_recall.core.entity_names.SequenceMatcher", score)
    with core.storage.read(ctx) as tx:
        assert not tx.claims._entity_lookup("TEST-scope", "𓀀𓀁𓀂𓀃", (), "")
        assert tx.claims._entity_lookup("TEST-scope", "TEST-person-999", (), "")
    assert 1 <= len(calls) <= 128


def test_candidates_rank_nearer_names_before_early_repeated_subjects(app):
    core, ctx = app
    for number in range(24):
        fact(core, ctx, "TEST-Alexander", f"tag{number}", "person")
    target = fact(core, ctx, "TEST-Alexandria", "category", "person")
    with core.storage.read(ctx) as tx:
        refs = tx.claims._entity_lookup("TEST-scope", "TEST-Alexandriya", (), "")
        subjects = [tx.claims.version(ref, tx.claims.current_revision(ref)).payload["subject"] for ref in refs]
    assert refs[0] == target.ref
    assert len(subjects) == len(set(subjects))


def test_followup_facts_with_checked_names_do_not_call_the_model(app):
    core, ctx = app
    named_fact(core, ctx, "TEST-Ricecake", "home")
    model = RecordedModel(False)
    drain_entities(core, ctx, model)
    assert len(model.calls) == 1
    named_fact(core, ctx, "TEST-Ricecake", "color")
    drain_entities(core, ctx, model)
    assert len(model.calls) == 1


def test_one_name_keeps_one_alias_when_another_fact_supplies_the_evidence(app):
    core, ctx = app
    target = fact(core, ctx, "TEST-Nian", "category", "cat")
    first = fact(core, ctx, "TEST-Ricecake", "home", "TEST-Harbor")
    second = fact(core, ctx, "TEST-Ricecake", "color", "TEST-white")
    left, right = merge(core, ctx, first, target), merge(core, ctx, second, target)
    assert right is None or right.ref == left.ref
    with core.storage.read(ctx) as tx:
        assert len(tx.claims.list_refs(alias_name="TEST-Ricecake")) == 1


def revise_endpoint(core, ctx, target):
    with core.storage.write(ctx) as tx:
        return tx.claims.append(
            target.scope_id,
            target.payload,
            Qualification("active", "direct_report", "supported"),
            recorded_at=core.clock.utc_now(),
            previous=target,
        )


def test_invalidated_name_reenters_bounded_work_but_withdrawn_relation_does_not(app):
    core, ctx = app
    target = fact(core, ctx, "TEST-Nian", "category", "cat")
    source = fact(
        core, ctx, "TEST-Ricecake", "home", "TEST-Harbor", "TEST-Ricecake is TEST-Nian. TEST-Ricecake home TEST-Harbor."
    )
    model = RecordedModel()
    drain_entities(core, ctx, model)
    target = revise_endpoint(core, ctx, target)
    with core.storage.write(ctx) as tx:
        queue_entity_backfill(tx, now=core.clock.utc_now())
        state = (
            tx._check()
            .execute(
                "SELECT state FROM work_items WHERE work_type='rebuild_projection' "
                "AND subject_ref=? AND subject_revision=?",
                (source.ref, source.revision),
            )
            .fetchone()[0]
        )
    assert state == "pending"
    drain_entities(core, ctx, model)
    with core.storage.read(ctx) as tx:
        resolution = resolve_subject(tx, "TEST-Ricecake", core.clock.utc_now())
        assert resolution.subject == "TEST-Nian"
        alias = next(
            tx.claims.version(item.ref, item.revision)
            for item in resolution.fence
            if tx.claims.version(item.ref, item.revision).payload["kind"] == "alias"
        )
    withdraw(core, ctx, alias)
    revise_endpoint(core, ctx, target)
    with core.storage.write(ctx) as tx:
        queue_entity_backfill(tx, now=core.clock.utc_now())
        state = (
            tx._check()
            .execute(
                "SELECT state FROM work_items WHERE work_type='rebuild_projection' "
                "AND subject_ref=? AND subject_revision=?",
                (source.ref, source.revision),
            )
            .fetchone()[0]
        )
    assert state == "done"
    drain_entities(core, ctx, model)
    with core.storage.read(ctx) as tx:
        assert resolve_subject(tx, "TEST-Ricecake", core.clock.utc_now()).status == "literal"


def test_query_name_lookup_does_not_apply_instr_to_stored_rows(app):
    core, ctx = app
    target = _identity(core, ctx)
    source = capture(core, ctx, "TEST-mist 项目改名为 TEST暮光，保留旧称。")
    assert accept(core, ctx, _alias(source, target, "TEST暮光")).items[0].state == "active"
    with core.storage.read(ctx) as tx:

        def reject_scan(*_args):
            raise AssertionError("stored-row substring scan")

        tx._check().create_function("instr", 2, reject_scan)
        assert tx.claims.aliases_for_query("TEST暮光 配色")
        assert tx.claims.entity_names_for_query("TEST-mist 配色", matches=("TEST-mist",))


def test_default_runtime_drains_entity_judgements_through_the_clamped_port(app):
    core, ctx = app
    fact(core, ctx, "TEST-Nian", "category", "cat")
    fact(
        core, ctx, "TEST-Ricecake", "home", "TEST-Harbor", "TEST-Ricecake is TEST-Nian. TEST-Ricecake home TEST-Harbor."
    )
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute("UPDATE work_items SET state='done'")

    class RecordingPort:
        calls = []

        def propose(self, messages, *, remaining_seconds):
            self.calls.append(remaining_seconds)
            body = json.loads(messages[-1]["content"])
            target = next(item for item in body["candidates"] if item["payload"]["subject"] == "TEST-Nian")
            return json.dumps(
                dict(
                    support="supported",
                    action="duplicate",
                    target_ref=target["ref"],
                    target_revision=target["revision"],
                )
            )

    port = RecordingPort()
    config = RuntimeInstanceConfig(
        ctx.binding,
        ctx.session_id,
        ctx.allowed_scope_ids,
        project_id=ctx.project_id,
        branch_id=ctx.branch_id,
        request_seconds=0.25,
        drain_seconds=20,
        supervisor_enabled=False,
    )
    instance = build_runtime_instance(config, consolidation=port)
    try:
        receipt = instance.drain(remaining_seconds=20)
        assert receipt.failed == 0
        assert port.calls and max(port.calls) <= 0.25
        with core.storage.read(ctx) as tx:
            assert tx._check().execute("SELECT count(*) FROM entity_alias_proofs").fetchone()[0] == 1
    finally:
        instance.close()


def test_two_fact_leases_cannot_judge_the_same_pending_name(app):
    core, ctx = app
    named_fact(core, ctx, "TEST-Ricecake", "home")
    named_fact(core, ctx, "TEST-Ricecake", "color")
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute("UPDATE work_items SET state='done'")
        queue_entity_backfill(tx, now=core.clock.utc_now())
        first = tx.work.claim_next(
            "TEST-name-owner",
            core.clock.utc_now(),
            lease_seconds=60,
            allowed_work_types=frozenset({"rebuild_projection"}),
        )[0]
    config = RuntimeInstanceConfig(
        ctx.binding,
        ctx.session_id,
        ctx.allowed_scope_ids,
        project_id=ctx.project_id,
        branch_id=ctx.branch_id,
        daily_work_limit=1,
    )
    nested = []

    class ReentrantModel(RecordedModel):
        def judge(self, messages, **kwargs):
            answer = super().judge(messages, **kwargs)
            if len(self.calls) == 1:
                instance = SimpleNamespace(
                    auxiliary=SimpleNamespace(),
                    background_gaps=(),
                    ingress_receipts=(),
                    status=partial(core.status, ctx),
                    drain=lambda *, max_items, purge_only, remaining_seconds: drain_worker(
                        core.storage,
                        core.clock,
                        ctx,
                        config=WorkerConfig("TEST-concurrent", max_items=max_items, purge_only=purge_only),
                        consolidation=self,
                        remaining_seconds=remaining_seconds,
                    ),
                )
                with ThreadPoolExecutor(max_workers=1) as workers:
                    nested.append(workers.submit(_drain_once, config, instance, time.monotonic() + 10).result())
                    core.clock.now = "2026-09-06T12:00:02Z"
                    nested.append(workers.submit(_drain_once, config, instance, time.monotonic() + 10).result())
            return answer

    model = ReentrantModel(False)
    result = process_rebuild_projection(
        core.storage, core.clock, ctx, first, model=model, started=core.clock.monotonic(), budget=10
    )
    assert result[2] == "done"
    assert nested[0]["deferred"] == 1 and nested[0]["items"][0]["error_code"] == "entity_name_leased"
    assert nested[1]["processed"] == 0 and "daily_queue_budget" not in nested[1]["capability_gaps"]
    assert [receipt["daily_queue_used"] for receipt in nested] == [0, 0]
    assert len(model.calls) == 1
    with core.storage.read(ctx) as tx:
        assert tuple(
            tx._check()
            .execute("SELECT attempt,available_at FROM work_items WHERE work_id=?", (nested[0]["items"][0]["work_id"],))
            .fetchone()
        ) == (
            0,
            "2026-09-06T12:01:00Z",
        )
    core.clock.now = "2026-09-06T12:01:01Z"
    receipt = drain_worker(
        core.storage, core.clock, ctx, config=WorkerConfig("TEST-concurrent"), consolidation=model, remaining_seconds=20
    )
    assert receipt.completed == 1 and len(model.calls) == 1


@pytest.mark.parametrize("owner_state", ["failed", "pending", "leased"])
def test_worker_reuses_abandoned_name_reservation(app, owner_state):
    core, ctx = app
    previous = named_fact(core, ctx, "TEST-Ricecake", "home")
    ready = named_fact(core, ctx, "TEST-Ricecake", "color")
    core.clock.now = "2026-09-06T12:00:01.125000Z"
    with core.storage.write(ctx) as tx:
        conn = tx._check(write=True)
        conn.execute("UPDATE work_items SET state='done'")
        conn.execute(
            "UPDATE work_items SET state=?,available_at='2026-09-07T12:00:00Z',last_error_code='http_400',"
            "lease_owner='TEST-older-worker',lease_token=3,lease_until=? "
            "WHERE work_type='rebuild_projection' AND subject_ref=?",
            (owner_state, core.clock.utc_now(), previous.ref),
        )
        conn.execute(
            "UPDATE entity_names SET work_id=(SELECT work_id FROM work_items "
            "WHERE work_type='rebuild_projection' AND subject_ref=?) WHERE name='TEST-Ricecake'",
            (previous.ref,),
        )
        conn.execute(
            "UPDATE work_items SET state='pending' WHERE work_type='rebuild_projection' AND subject_ref=?",
            (ready.ref,),
        )
    model = RecordedModel(False)
    receipt = drain_worker(
        core.storage, core.clock, ctx, config=WorkerConfig("TEST-ready"), consolidation=model, remaining_seconds=20
    )
    assert receipt.completed == 1 and receipt.deferred == 0
    assert len(model.calls) == 1
    with core.storage.read(ctx) as tx:
        assert tuple(tx._check().execute("SELECT status,source_ref FROM entity_names").fetchone()) == (
            "checked",
            ready.ref,
        )


def test_reclaimed_work_cannot_reuse_the_previous_leases_name_reservation(app):
    core, ctx = app
    named_fact(core, ctx, "TEST-Ricecake", "home")
    named_fact(core, ctx, "TEST-Ricecake", "color")
    projection = frozenset({"rebuild_projection"})
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute("UPDATE work_items SET state='done'")
        queue_entity_backfill(tx, now=core.clock.utc_now())
        original = tx.work.claim_next(
            "TEST-original", core.clock.utc_now(), lease_seconds=1, allowed_work_types=projection
        )[0]

    class ReentrantModel(RecordedModel):
        nested = None

        def judge(self, messages, **kwargs):
            answer = super().judge(messages, **kwargs)
            if len(self.calls) == 1:
                core.clock.now = "2026-09-06T12:00:02Z"
                with core.storage.write(ctx) as tx:
                    reclaimed = tx.work.claim_next(
                        "TEST-reclaim", core.clock.utc_now(), lease_seconds=60, limit=2, allowed_work_types=projection
                    )
                assert reclaimed[0].work_id == original.work_id
                assert reclaimed[0].lease_token > original.lease_token
                self.nested = Context().run(
                    process_rebuild_projection,
                    core.storage,
                    core.clock,
                    ctx,
                    reclaimed[1],
                    model=self,
                    started=core.clock.monotonic(),
                    budget=10,
                )
            return answer

    model = ReentrantModel(False)
    result = process_rebuild_projection(
        core.storage, core.clock, ctx, original, model=model, started=core.clock.monotonic(), budget=10
    )
    assert result[0] == "stale"
    assert model.nested == ("completed", None, "done")
    assert len(model.calls) == 2
    with core.storage.read(ctx) as tx:
        assert tuple(tx._check().execute("SELECT status,source_ref FROM entity_names").fetchone()) == (
            "checked",
            model.calls[1]["candidates"][0]["ref"],
        )


def test_expired_vector_lookup_does_not_start_a_late_entity_judgement(app):
    core, ctx = app
    named_fact(core, ctx, "TEST-Ricecake", "home")
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute("UPDATE work_items SET state='done'")
        queue_entity_backfill(tx, now=core.clock.utc_now())
        item = tx.work.claim_next(
            "TEST-expired", core.clock.utc_now(), lease_seconds=1, allowed_work_types=frozenset({"rebuild_projection"})
        )[0]

    class ExpiringVector:
        def search(self, *args, **kwargs):
            core.clock.now = "2026-09-06T12:00:02Z"
            return ()

    core.storage.entity_vectors = ExpiringVector
    model = RecordedModel(False)
    result = process_rebuild_projection(
        core.storage, core.clock, ctx, item, model=model, started=core.clock.monotonic(), budget=10
    )
    assert result[0] == "stale"
    assert not model.calls


@pytest.mark.parametrize("expires", ["lease", "budget"])
def test_entity_provider_rechecks_expiry_after_evidence_reads(app, monkeypatch, expires):
    from scope_recall.core import private, worker_entities

    core, ctx = app
    named_fact(core, ctx, "TEST-Ricecake", "home")
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute("UPDATE work_items SET state='done'")
        queue_entity_backfill(tx, now=core.clock.utc_now())
        item = tx.work.claim_next(
            "TEST-late", core.clock.utc_now(), lease_seconds=1, allowed_work_types=frozenset({"rebuild_projection"})
        )[0]

    checking_provider = False
    check_evidence = worker_entities._publication_blocked
    monotonic = core.clock.monotonic

    def slow_evidence_check(*args):
        blocked = check_evidence(*args)
        if checking_provider:
            if expires == "lease":
                core.clock.now = "2026-09-06T12:00:01.001000Z"
            else:
                monkeypatch.setattr(core.clock, "monotonic", lambda: monotonic() + 20)
        return blocked

    class GuardedModel(RecordedModel):
        def judge(self, messages, **kwargs):
            nonlocal checking_provider
            checking_provider = True
            private.require_outbound()
            return super().judge(messages, **kwargs)

    monkeypatch.setattr(worker_entities, "_publication_blocked", slow_evidence_check)
    model = GuardedModel(False)
    process_rebuild_projection(core.storage, core.clock, ctx, item, model=model, started=monotonic(), budget=10)
    assert checking_provider and not model.calls


def test_expired_worker_cannot_mark_its_reclaimed_mention_ineligible(app):
    core, ctx = app
    source = named_fact(core, ctx, "TEST-Ricecake", "home")
    projection = frozenset({"rebuild_projection"})
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute("UPDATE work_items SET state='done'")
        queue_entity_backfill(tx, now=core.clock.utc_now())
        original = tx.work.claim_next(
            "TEST-original", core.clock.utc_now(), lease_seconds=1, allowed_work_types=projection
        )[0]

    class LateRefusal:
        def judge(self, messages, **kwargs):
            core.clock.now = "2026-09-06T12:00:02Z"
            with core.storage.write(ctx) as tx:
                renewed = tx.work.claim_next(
                    "TEST-renewed", core.clock.utc_now(), lease_seconds=60, allowed_work_types=projection
                )[0]
            assert renewed.work_id == original.work_id and renewed.lease_token > original.lease_token
            raise ContractError("ACCESS_DENIED", "authority_revoked")

    result = process_rebuild_projection(
        core.storage, core.clock, ctx, original, model=LateRefusal(), started=core.clock.monotonic(), budget=10
    )
    assert result == ("stale", None, "leased")
    with core.storage.read(ctx) as tx:
        row = (
            tx._check()
            .execute(
                "SELECT m.usable,v.entity_resolved FROM entity_name_mentions m JOIN claim_versions v "
                "ON v.claim_id=m.claim_ref AND v.revision=m.claim_revision WHERE m.claim_ref=?",
                (source.ref,),
            )
            .fetchone()
        )
        assert tuple(row) == (1, 0)


def test_candidate_evidence_matches_a_name_before_sentence_punctuation(app):
    core, ctx = app
    target = fact(core, ctx, "TEST-mapzjirkrhee", "category", "person")
    with core.storage.read(ctx) as tx:
        assert target.ref in tx.claims._entity_lookup("TEST-scope", "TEST-称呼", (), "TEST-称呼 is TEST-mapzjirkrhee.")


def test_deleted_alias_endpoint_returns_the_surviving_name_to_pending(app):
    core, ctx = app
    target = fact(core, ctx, "TEST-Nian", "category", "cat")
    source = named_fact(core, ctx, "TEST-Ricecake", "home")
    merge(core, ctx, source, target)
    capture(core, ctx, f"删除 {target.ref}。")
    core.forget(
        ctx,
        dict(
            protocol_version="1.1",
            target_refs=[target.ref],
            mode="delete",
            expected_revisions={target.ref: target.revision},
        ),
        remaining_seconds=10,
    )
    with core.storage.write(ctx) as tx:
        assert (
            tx._check().execute("SELECT status FROM entity_names WHERE name='TEST-Ricecake'").fetchone()[0] == "pending"
        )
        queue_entity_backfill(tx, now=core.clock.utc_now())
        assert (
            tx._check()
            .execute(
                "SELECT entity_resolved FROM claim_versions WHERE claim_id=? AND revision=?",
                (source.ref, source.revision),
            )
            .fetchone()[0]
            == 0
        )
    model = RecordedModel(False)
    drain_entities(core, ctx, model)
    assert len(model.calls) == 1


def test_alias_graph_joins_use_identity_keys_with_stale_namespace_statistics(app):
    core, ctx = app
    target = _identity(core, ctx)
    source = capture(core, ctx, "TEST-mist 项目改名为 TEST暮光，保留旧称。")
    assert accept(core, ctx, _alias(source, target, "TEST暮光")).items[0].state == "active"
    with core.storage.write(ctx) as tx:
        conn = tx._check(write=True)
        conn.execute("ANALYZE")
        conn.execute("UPDATE sqlite_stat1 SET stat='100000 1 1 1 1' WHERE idx='entity_alias_scope'")
        conn.execute("ANALYZE sqlite_schema")
        statements = []
        conn.set_trace_callback(statements.append)
        assert tx.claims.list_refs(alias_name="TEST暮光")
        conn.set_trace_callback(None)
        plans = [
            tuple(row)
            for sql in statements
            if sql.startswith("SELECT DISTINCT c.claim_id")
            for row in conn.execute("EXPLAIN QUERY PLAN " + sql)
        ]
        assert any("c USING INDEX" in row[3] and "(claim_id=?)" in row[3] for row in plans), plans
