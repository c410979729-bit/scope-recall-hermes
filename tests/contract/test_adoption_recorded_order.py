"""First entrance order survives host timestamp gaps, replay and edits."""

import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
from scope_recall.adapters.hermes.boundary import SourceObservationLedger, sync_assistant_key, sync_turn_source_events
from scope_recall.core import adoption_schedule
from scope_recall.core.adoption_order import adjacent_occurrence, first_occurrence
from scope_recall.core.adoption_storage import adoption_context
from scope_recall.core.capture_inbox import enqueue
from scope_recall.core.schema import SCHEMA_VERSION
from v11_support import downgrade_store

from tests.contract.test_adoption import PLAN, capture
from tests.contract.test_adoption_late_reply import active_adoptions, confirm, drain, replay, reply_event, store_reply
from tests.contract.test_claims import app


def test_reply_witness_precedes_next_turn_even_when_sync_is_late(app):
    core, ctx = app
    ledger = SourceObservationLedger()
    key = sync_assistant_key(ctx, ctx.session_id, "TEST-first-turn")
    ledger.witness((key, 1), "2026-09-01T12:00:01Z", "2026-09-01T12:00:01Z")
    owner = confirm(core, ctx)
    drain(core, ctx)
    events, _ = sync_turn_source_events(
        ledger,
        ctx,
        session_id=ctx.session_id,
        turn_id="TEST-first-turn",
        user_content="",
        assistant_content=PLAN,
        recorded_at="2026-09-01T12:00:03Z",
        outcome="success",
        include_user=False,
    )
    event = events[0][0]
    assert event["recorded_at"] == "2026-09-01T12:00:01Z"
    receipt = core.record_event(replace(ctx, actor_origin="assistant_visible"), event, scope_id="TEST-scope")
    with core.storage.read(ctx) as tx:
        assert (
            first_occurrence(tx._check(), receipt.event_refs[0].ref)["source_id"]
            > first_occurrence(tx._check(), owner.ref)["source_id"]
        )
    drain(core, ctx)
    claims = active_adoptions(core, ctx)
    assert len(claims) == 1 and claims[0].state == "active"
    assert {span["source_ref"] for span in claims[0].payload["evidence_spans"]} == {
        owner.ref,
        receipt.event_refs[0].ref,
    }


def test_same_handoff_orders_equivalent_utc_spellings_by_source_id(app):
    core, ctx = app
    actor = replace(ctx, actor_origin="assistant_visible")
    with core.storage.write(actor) as tx:
        earlier = tx.sources.put_source(
            reply_event(source_event_key="TEST-first", recorded_at="2026-09-01T12:00:01.000000+00:00"),
            scope_id="TEST-scope",
            persisted_at=core.clock.utc_now(),
        )
        later = tx.sources.put_source(
            reply_event(recorded_at="2026-09-01T12:00:01Z"), scope_id="TEST-scope", persisted_at=core.clock.utc_now()
        )
    owner = capture(core, ctx, "好，就用方案二", when=None, recorded_at="2026-09-01T12:00:01.0Z")
    with core.storage.read(ctx) as tx:
        assert (
            first_occurrence(tx._check(), earlier.ref)["source_id"]
            < first_occurrence(tx._check(), later.ref)["source_id"]
        )
        assert adoption_context(tx, owner).assistant_ref == later.ref


def test_recorded_order_preserves_submillisecond_distinctions(app):
    core, ctx = app
    expected = store_reply(core, ctx, source_event_key="TEST-later", recorded_at="2026-09-01T12:00:01.000002Z")
    store_reply(core, ctx, recorded_at="2026-09-01T12:00:01.000001Z")
    owner = capture(core, ctx, "好，就用方案二", when=None, recorded_at="2026-09-01T12:00:01.000003Z")
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant_ref == expected.ref


@pytest.mark.parametrize("kind", ["user", "tool", "waiting", "hidden"])
def test_empty_new_context_preserves_binding_checkpoint_and_fact(app, kind):
    core, ctx = app
    plan = store_reply(core, ctx)
    owner = confirm(core, ctx)
    drain(core, ctx)
    with core.storage.read(ctx) as tx:
        before = tuple(tx._check().execute("SELECT * FROM adoption_relations").fetchone())
        checkpoint = tuple(tx._check().execute("SELECT * FROM adoption_processing").fetchone())
    if kind in {"user", "tool"}:
        capture(
            core,
            ctx,
            "TEST intervening message",
            key="TEST-intervening",
            when=None,
            recorded_at="2026-09-01T12:00:01.5Z",
            origin="human_direct" if kind == "user" else "tool_observation",
        )
    elif kind == "waiting":
        enqueue(
            core.storage,
            core.clock,
            replace(ctx, actor_origin="assistant_visible"),
            reply_event(source_event_key="TEST-waiting", recorded_at="2026-09-01T12:00:01.5Z"),
            scope_id="TEST-scope",
            host_scope=None,
        )
    else:
        with core.storage.write(ctx) as tx:
            tx._check(write=True).execute("UPDATE source_events SET read_blocked=1 WHERE event_id=?", (plan.ref,))
    with core.storage.write(ctx) as tx:
        adoption_schedule.schedule_adoptions(tx, (f"{owner.ref}@{owner.revision}",), now=core.clock.utc_now())
        assert tuple(tx._check().execute("SELECT * FROM adoption_relations").fetchone()) == before
        assert tuple(tx._check().execute("SELECT * FROM adoption_processing").fetchone()) == checkpoint
        assert (
            tx._check().execute("SELECT state FROM claim_versions ORDER BY revision DESC LIMIT 1").fetchone()[0]
            == "active"
        )


def test_pending_tie_follows_persisted_reply_and_blocks_earlier_binding(app):
    core, ctx = app
    store_reply(core, ctx)
    actor = replace(ctx, actor_origin="assistant_visible")
    enqueue(
        core.storage,
        core.clock,
        actor,
        reply_event(source_event_key="TEST-pending"),
        scope_id="TEST-scope",
        host_scope=None,
    )
    owner = confirm(core, ctx)
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner) is None
    replay(core, ctx)
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant.event["source_event_key"] == "TEST-pending"


def test_existing_checkpoint_is_not_rebound_to_late_predecessor(app):
    from tests.contract.test_adoption_late_reply import Model

    core, ctx = app
    store_reply(core, ctx, source_event_key="TEST-old", content="TEST earlier suggestion")
    owner = confirm(core, ctx)
    core.drain_worker(ctx, consolidation=Model(), max_items=1, remaining_seconds=20)
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT 1 FROM adoption_processing").fetchone()
    enqueue(
        core.storage,
        core.clock,
        replace(ctx, actor_origin="assistant_visible"),
        reply_event(recorded_at="2026-09-01T12:00:01.5Z"),
        scope_id="TEST-scope",
        host_scope=None,
    )
    drain(core, ctx)
    with core.storage.read(ctx) as tx:
        assert (
            tx._check()
            .execute("SELECT state FROM work_items WHERE work_type='consolidate' AND subject_ref=?", (owner.ref,))
            .fetchone()[0]
            == "obsolete"
        )
    replay(core, ctx)
    drain(core, ctx)
    assert not active_adoptions(core, ctx)
    with core.storage.read(ctx) as tx:
        row = tx._check().execute("SELECT assistant_ref FROM adoption_processing").fetchone()
        assert tx.source(row[0], 1).event["source_event_key"] == "TEST-old"


def test_retry_recorded_at_cannot_move_first_envelope(app):
    core, ctx = app
    plan = store_reply(core, ctx)
    store_reply(core, ctx, recorded_at="2026-09-01T12:00:10Z")
    owner = confirm(core, ctx)
    with core.storage.read(ctx) as tx:
        assert adoption_context(tx, owner).assistant_ref == plan.ref
        assert first_occurrence(tx._check(), plan.ref)["recorded_at"] == "2026-09-01T12:00:01Z"


def test_in_order_capture_skips_adoption_contexts(app, monkeypatch):
    core, ctx = app
    store_reply(core, ctx)
    confirm(core, ctx)
    drain(core, ctx)
    monkeypatch.setattr(adoption_schedule, "adoption_context", lambda *args: pytest.fail("in-order adoption lookup"))
    capture(core, ctx, "TEST next message", when=None, recorded_at="2026-09-01T12:00:03Z")


@pytest.mark.parametrize("schema", [1133, 1134, SCHEMA_VERSION])
def test_initialization_supplies_order_index_on_existing_store(app, schema):
    core, ctx = app
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute("DROP INDEX source_adoption_order")
    if schema != SCHEMA_VERSION:
        downgrade_store(core.storage.path, schema)
    with closing(sqlite3.connect(core.storage.path.as_uri() + "?mode=ro", uri=True)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema
    core.storage.initialize()
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT 1 FROM sqlite_master WHERE name='source_adoption_order'").fetchone()
        assert tx._check().execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_neighbor_queries_seek_the_order_index(app):
    core, ctx = app
    plan = store_reply(core, ctx)
    owner = confirm(core, ctx)
    statements = []
    with core.storage.read(ctx) as tx:
        conn = tx._check()
        conn.set_trace_callback(statements.append)
        adjacent_occurrence(conn, plan, first_occurrence(conn, plan.ref), following=True)
        adjacent_occurrence(conn, owner, first_occurrence(conn, owner.ref))
        conn.set_trace_callback(None)
        neighbors = [sql for sql in statements if "AS moment FROM source_events e WHERE e.session_id" in sql]
        assert len(neighbors) == 2
        for sql in neighbors:
            plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql)]
            assert any("USING INDEX source_adoption_order" in step and "<expr>" in step for step in plan), plan
