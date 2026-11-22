"""Focused offline checks for long-source progress, transaction and authority fences."""

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from scope_recall.contracts import ContractError
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.core.consolidate import consolidation_messages
from scope_recall.core.schema import SCHEMA_VERSION
from scope_recall.core.storage import SQLiteStorage
from scope_recall.core.work_storage import WorkItems
from test_deletion import authorize, request
from test_worker import FakeConsolidation, _mark_embed_done, app, capture, consolidation_payload, draft, worker_app
from v11_support import downgrade_store

FIRST = "TEST-project 配色 蓝色。"
LAST = "TEST-project 主题 深色。"


def long_source(core, ctx, content=None):
    source = capture(core, ctx, content or (FIRST + "\n" + "这是一段归档资料；" * 5000 + "\n" + LAST))
    _mark_embed_done(core)
    return source


def row(core, source):
    with sqlite3.connect(core.storage.path) as db:
        return db.execute(
            "SELECT state,consolidation_offset,attempt FROM work_items WHERE work_type='consolidate' AND subject_ref=?",
            (source.ref,),
        ).fetchone()


def claims(core):
    with sqlite3.connect(core.storage.path) as db:
        return [
            (json.loads(payload), state)
            for payload, state in db.execute("SELECT payload_json,state FROM claim_versions")
        ]


def proposal(page, quote, *, value="蓝色", predicate="配色"):
    return draft(
        page,
        value,
        predicate=predicate,
        evidence_spans=[dict(source_ref=page.ref, source_revision=page.revision, quote=quote)],
    )


def test_a_page_that_cannot_end_at_a_sentence_ends_between_runs():
    """On a long line with no sentence end, a page cut inside a word could carry the first letters of a value the
    request guard judges whole (``credential_env: nul`` of ``null``, ``<your api`` of ``<your api key>``), and be
    refused every time it is asked.  The offsets put the cut at every position within the repeated words, whatever
    the input budget."""
    from scope_recall.core.consolidation_chunks import source_chunk
    from scope_recall.runtime.consolidation_models import validate_chat_messages
    from test_candidate_evidence_window import _source

    words = " ".join(f"TEST{index} api_key: <your api key> credential_env: null" for index in range(1500))
    for offset in range(60):
        page, window = source_chunk(_source("a" * offset + " " + words), 0, formatter=consolidation_messages)
        assert window.end < window.total and page.event["content"][-1] == " ", offset
        validate_chat_messages(consolidation_messages([page]))


def test_existing_1105_source_upgrades_explicitly_and_resumes_to_exact_end(worker_app):
    core, ctx, clock = worker_app
    padding = ("这是一段归档资料；" * 8000)[: 65536 - len(FIRST) - len(LAST) - 2]
    source = long_source(core, ctx, FIRST + "\n" + padding + "\n" + LAST)
    assert len(source.event["content"]) == 65536
    downgrade_store(core.storage.path, 1105)
    # A foreign binding is refused before anything is touched...
    foreign = SQLiteStorage(replace(ctx.binding, installation_id="TEST-unrelated-installation"))
    with pytest.raises(ContractError, match="IDENTITY_UNBOUND"):
        foreign.initialize()
    with sqlite3.connect(core.storage.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1105
        assert "consolidation_offset" not in {r[1] for r in db.execute("PRAGMA table_info(work_items)")}
    # ...and the bound store is brought forward through every step by its first open.
    assert core.status(ctx).schema_version == SCHEMA_VERSION
    assert core.initialize().schema_version == SCHEMA_VERSION
    windows, contents = [], []

    def build(sources, **kwargs):
        assert len(sources) == 1
        page = sources[0]
        window = page.consolidation_window
        windows.append(window)
        contents.append(page.event["content"])
        messages = consolidation_messages(sources, episode_ref=kwargs["episode_ref"])
        assert sum(len(m["content"].encode("utf-8")) for m in messages) <= 16000
        assert (page.ref, page.revision) == (source.ref, source.revision)
        found = []
        if FIRST in page.event["content"]:
            found.append(proposal(page, FIRST))
        if LAST in page.event["content"]:
            found.append(proposal(page, LAST, value="深色", predicate="主题"))
        return consolidation_payload(page, claims=found)

    model = FakeConsolidation(build)
    first = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=5)
    assert first.deferred == 1 and first.completed == 0
    assert row(core, source)[0] == "pending" and 0 < row(core, source)[1] < len(source.event["content"])
    # A new Core reads the durable cursor; no in-memory model/session state is required.
    core = MemoryCore(CoreConfig(ctx.binding), clock=clock)
    # Schema, instructions and source metadata share the byte limit with each page.
    # Keep the call bound fixed so growing metadata cannot silently shrink progress.
    for _ in range(120):
        if row(core, source)[0] == "done":
            break
        result = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=5)
        assert result.failed == result.retried == 0
    assert row(core, source) == ("done", len(source.event["content"]), 1)
    assert "".join(contents) == source.event["content"]
    assert all(a.end == b.start for a, b in zip(windows, windows[1:]))
    assert {(p["value_text"], state) for p, state in claims(core)} == {("蓝色", "active"), ("深色", "active")}
    assert core.source(ctx, source.ref, source.revision).event["content"] == source.event["content"]
    with sqlite3.connect(core.storage.path) as db:
        assert db.execute("SELECT MAX(processed_sequence) FROM episode_versions").fetchone()[0] in (None, 0)


def test_page_checkpoint_does_not_stand_down_other_consolidation(worker_app):
    core, ctx, clock = worker_app
    source = long_source(core, ctx)
    others = [
        capture(core, replace(ctx, session_id=f"TEST-other-{index}"), f"TEST 另一段短资料 {index}。")
        for index in range(3)
    ]
    _mark_embed_done(core)

    def build(sources, **_):
        if sources[0].ref == source.ref:
            clock.advance(iso="2026-09-06T12:00:01Z")  # the page ends after the others were queued
        return consolidation_payload(*sources)

    receipt = core.drain_worker(ctx, consolidation=FakeConsolidation(build), max_items=4, remaining_seconds=5)
    assert [(item.work_type, item.disposition) for item in receipt.items] == (
        [("consolidate", "deferred")] + [("consolidate", "completed")] * 3
    )
    assert row(core, source)[0] == "pending" and 0 < row(core, source)[1] < len(source.event["content"])
    assert [row(core, other)[0] for other in others] == ["done"] * 3


def _advance_worker_clock(clock, seconds):
    """Move both clocks so the simulated provider and the SQLite lease agree."""
    now = datetime.fromisoformat(clock.utc_now()) + timedelta(seconds=seconds)
    clock.advance(seconds=seconds, iso=now.isoformat().replace("+00:00", "Z"))


@pytest.mark.parametrize(("timeout", "claim_delay"), [(False, 0), (True, 0), (False, 7)])
def test_resumed_page_leaves_time_to_commit_or_record_a_timeout(worker_app, monkeypatch, timeout, claim_delay):
    from scope_recall.core import worker
    from scope_recall.runtime.models import AuxiliaryModelError

    core, ctx, clock = worker_app
    source = long_source(core, ctx)
    quick = FakeConsolidation(lambda sources, **_: consolidation_payload(*sources))
    assert core.drain_worker(ctx, consolidation=quick, max_items=1, remaining_seconds=25).deferred == 1
    _, checkpoint, _ = row(core, source)
    assert checkpoint > 0
    budgets = []
    claim = worker._claim_idle

    def claim_after_waiting_for_writer(*args, **kwargs):
        _advance_worker_clock(clock, claim_delay)
        return claim(*args, **kwargs)

    monkeypatch.setattr(worker, "_claim_idle", claim_after_waiting_for_writer)

    class SlowPage:
        def propose(self, sources, *, episode_ref, remaining_seconds):
            assert sources[0].consolidation_window.start == checkpoint
            budgets.append(remaining_seconds)
            _advance_worker_clock(clock, remaining_seconds)
            if timeout:
                raise AuxiliaryModelError("timeout")
            return json.dumps(consolidation_payload(*sources))

    receipt = core.drain_worker(ctx, consolidation=SlowPage(), max_items=1, remaining_seconds=25, lease_seconds=8)
    state, offset, attempt = row(core, source)
    assert len(budgets) == 1
    assert state == "pending"
    assert 3 < budgets[0] < 8
    if timeout:
        assert (offset, attempt) == (checkpoint, 1)
        assert [(item.disposition, item.error_code) for item in receipt.items] == [("retry", "timeout")]
    else:
        assert offset > checkpoint and attempt == 0 and receipt.deferred == 1


@pytest.mark.parametrize("proposal_uses_request_window", [False, True])
def test_publication_judge_uses_lease_time_left_after_the_proposal(worker_app, proposal_uses_request_window):
    from scope_recall.runtime.instance import RuntimeInstance
    from scope_recall.runtime.instance_config import RuntimeInstanceConfig

    core, ctx, clock = worker_app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    _mark_embed_done(core)
    proposal = draft(source, subject="TEST-project", predicate="database", value="PostgreSQL")
    budgets = []

    class JudgedPage:
        def propose(self, sources, *, episode_ref, remaining_seconds):
            _advance_worker_clock(clock, remaining_seconds if proposal_uses_request_window else 2)
            return json.dumps(consolidation_payload(*sources, claims=[proposal]), ensure_ascii=False)

        def judge(self, messages, *, remaining_seconds):
            budgets.append(remaining_seconds)
            _advance_worker_clock(clock, remaining_seconds)
            return json.dumps(dict(support="supported", action="new", target_ref=None, target_revision=None))

    config = RuntimeInstanceConfig(
        binding=ctx.binding,
        session_id=ctx.session_id,
        allowed_scope_ids=ctx.allowed_scope_ids,
        project_id=ctx.project_id,
        branch_id=ctx.branch_id,
        request_seconds=8,
        lease_seconds=8,
        drain_seconds=25,
        max_items=1,
    )
    receipt = RuntimeInstance(config=config, core=core, auxiliary=None).drain(consolidation=JudgedPage())
    if proposal_uses_request_window:
        assert budgets == []
        assert [(item.disposition, item.error_code) for item in receipt.items] == [("retry", "DEADLINE_EXCEEDED")]
        assert row(core, source) == ("pending", 0, 1) and claims(core) == []
        return
    assert len(budgets) == 1
    assert receipt.completed == 1
    assert 0 < budgets[0] < 6
    assert [(payload["value_text"], state) for payload, state in claims(core)] == [("PostgreSQL", "active")]


def test_adoption_continuation_keeps_its_pass_budget(worker_app):
    from test_adoption import assistant, model

    core, ctx, _clock = worker_app
    assistant(core, ctx)
    owner = capture(core, ctx, "好，就用方案二")
    _mark_embed_done(core)
    judge, budgets = model()[0], []

    class AdoptingModel(FakeConsolidation):
        def judge(self, messages, *, remaining_seconds):
            budgets.append(remaining_seconds)
            return judge(messages, remaining_seconds=remaining_seconds)

    port = AdoptingModel(lambda sources, **_: consolidation_payload(*sources))
    for _ in range(2):
        receipt = core.drain_worker(ctx, consolidation=port, max_items=1, remaining_seconds=25, lease_seconds=8)
        assert receipt.deferred == 1
    assert row(core, owner)[0] == "pending"
    assert budgets == [25]


def test_port_refusal_still_stands_consolidation_down_for_the_pass(worker_app):
    core, ctx, _clock = worker_app
    for index in range(3):
        capture(core, replace(ctx, session_id=f"TEST-refused-{index}"), f"TEST 预算暂停资料 {index}。")
    _mark_embed_done(core)

    class BudgetRefusal(RuntimeError):
        error_type = "budget_exhausted"

    class RefusingPort:
        calls = 0

        def propose(self, *args, **kwargs):
            self.calls += 1
            raise BudgetRefusal("budget_exhausted")

    port = RefusingPort()
    receipt = core.drain_worker(ctx, consolidation=port, max_items=4, remaining_seconds=5)
    assert [(item.disposition, item.error_code) for item in receipt.items] == [("deferred", "budget_exhausted")]
    assert port.calls == 1
    with sqlite3.connect(core.storage.path) as db:
        rows = db.execute("""SELECT state,attempt,lease_token,last_error_code FROM work_items
                             WHERE work_type='consolidate' ORDER BY work_id""").fetchall()
    assert rows == [("pending", 0, 1, "budget_exhausted"), ("pending", 0, 0, None), ("pending", 0, 0, None)]


def test_chunk_claim_and_cursor_roll_back_together(worker_app, monkeypatch):
    core, ctx, _clock = worker_app
    source = long_source(core, ctx)
    real_advance = WorkItems.advance_consolidation

    def interrupted_commit(self, *args, **kwargs):
        real_advance(self, *args, **kwargs)
        raise ContractError("STORAGE_UNAVAILABLE", "synthetic_commit_failure")

    monkeypatch.setattr(WorkItems, "advance_consolidation", interrupted_commit)
    model = FakeConsolidation(
        lambda sources, **_: consolidation_payload(sources[0], claims=[proposal(sources[0], FIRST)])
    )
    result = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=5)
    assert result.retried == 1 and result.completed == 0
    assert row(core, source) == ("pending", 0, 1)
    assert claims(core) == []


def test_fragment_evidence_cannot_escape_window_or_omit_original_negation(worker_app):
    core, ctx, clock = worker_app
    # The negative qualifier lies outside the final fragment, in the same
    # original sentence. The complete-source qualification must still see it.
    source = long_source(core, ctx, "不允许" + "长" * 9000 + FIRST)
    bad = FakeConsolidation(
        lambda sources, **_: consolidation_payload(sources[0], claims=[proposal(sources[0], FIRST)])
    )
    result = core.drain_worker(ctx, consolidation=bad, max_items=1, remaining_seconds=5)
    assert result.retried == 0 and row(core, source)[1] > 0 and claims(core) == []
    with sqlite3.connect(core.storage.path) as conn:
        assert (
            conn.execute("SELECT error_field FROM work_error_details WHERE stage='claim'").fetchone()[0]
            == "claim_proposals[0].evidence_spans[0].quote"
        )
    clock.advance(iso="2026-09-06T12:00:03Z")
    model = FakeConsolidation(
        lambda sources, **_: consolidation_payload(
            sources[0], claims=[proposal(sources[0], FIRST)] if FIRST in sources[0].event["content"] else []
        )
    )
    for _ in range(30):
        if row(core, source)[0] == "done":
            break
        result = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=5)
        assert result.failed == result.retried == 0
    assert row(core, source)[0] == "done"
    assert claims(core) and all(state == "proposed" for _p, state in claims(core))
    assert model.feedbacks and all(value is None for value in model.feedbacks)


def test_deletion_during_fragment_model_call_cannot_commit_progress(worker_app):
    core, ctx, _clock = worker_app
    source = long_source(core, ctx)

    def remove(sources, **_):
        authorize(core, ctx, source)
        core.forget(ctx, request(source), remaining_seconds=5)
        return consolidation_payload(sources[0], claims=[proposal(sources[0], FIRST)])

    result = core.drain_worker(ctx, consolidation=FakeConsolidation(remove), max_items=1, remaining_seconds=5)
    assert result.completed == 0 and row(core, source)[0] == "obsolete" and row(core, source)[1] == 0
    assert claims(core) == [] and core.source(ctx, source.ref, source.revision) is None


def test_oversized_repair_is_exact_once_and_short_resume_cannot_skip_long_source(worker_app):
    from scope_recall.core.episodes import source_watermark

    core, ctx, clock = worker_app
    source = long_source(core, ctx)
    short = capture(core, ctx, "请帮我完成 TEST 报告整理。")
    ordinary_invalid = capture(core, replace(ctx, session_id="TEST-other-session"), "TEST 普通无效结果。")
    _mark_embed_done(core)
    with sqlite3.connect(core.storage.path) as db:
        db.execute(
            "UPDATE work_items SET state='failed',attempt=3,last_error_code='INPUT_INVALID' WHERE work_type='consolidate' AND subject_ref IN (?,?)",
            (source.ref, ordinary_invalid.ref),
        )
    with core.storage.write(ctx) as tx:
        assert tx.work.recover_oversized_consolidations(now=clock.utc_now(), formatter=consolidation_messages) == 1
    assert row(core, source) == ("pending", 0, 3)
    assert row(core, ordinary_invalid) == ("failed", 0, 3)
    # Make the short source the next lease while the repaired long source is
    # pending. It is excluded by the prompt budget, not falsely covered.
    with sqlite3.connect(core.storage.path) as db:
        db.execute(
            "UPDATE work_items SET available_at='2026-09-06T11:59:59Z' WHERE work_type='consolidate' AND subject_ref=?",
            (short.ref,),
        )

    def short_resume(sources, episode_ref=None):
        assert [s.ref for s in sources] == [short.ref]
        refs = [f"{short.ref}@{short.revision}"]
        resume = dict(
            episode_ref=episode_ref,
            goal=dict(text=short.event["content"], evidence_refs=refs),
            decisions=[],
            verified_progress=[],
            open_items=[],
            blockers=[],
            next_step=None,
            next_step_basis="unknown",
            artifact_refs=[],
            source_watermark=source_watermark(refs),
            evidence_refs=refs,
        )
        return consolidation_payload(short, resume_proposals=[resume])

    assert (
        core.drain_worker(
            ctx, consolidation=FakeConsolidation(short_resume), max_items=1, remaining_seconds=5
        ).completed
        == 1
    )
    assert row(core, source) == ("pending", 0, 3)
    with sqlite3.connect(core.storage.path) as db:
        processed = db.execute("SELECT MAX(processed_sequence) FROM episode_versions").fetchone()[0]
    assert processed == 0
    clock.advance(iso="2026-09-06T12:00:10Z")
    empty = FakeConsolidation(lambda sources, **_: consolidation_payload(*sources))
    assert core.drain_worker(ctx, consolidation=empty, max_items=1, remaining_seconds=5).deferred == 1
    assert row(core, source)[1] > 0

    # Force a deterministic input failure on the next page: the old repair
    # must never become an unbounded retry, even with an exhausted attempt count.
    class Invalid:
        def propose(self, *args, **kwargs):
            raise ContractError("INPUT_INVALID", "synthetic_invalid_model_input")

    assert core.drain_worker(ctx, consolidation=Invalid(), max_items=1, remaining_seconds=5).failed == 1
    with sqlite3.connect(core.storage.path) as db:
        # Also verify the durable marker at offset zero, the repair predicate.
        db.execute("UPDATE work_items SET consolidation_offset=0 WHERE subject_ref=?", (source.ref,))
    with core.storage.write(ctx) as tx:
        assert tx.work.recover_oversized_consolidations(now=clock.utc_now(), formatter=consolidation_messages) == 0
    assert row(core, source)[0] == "failed"


def test_embed_and_projection_leases_start_after_writer_lock_wait(worker_app, monkeypatch):
    """Writer-lock wait before ``_claim_idle`` must not eat into a fresh embed or projection lease."""
    from scope_recall.core import worker, worker_projection

    core, ctx, clock = worker_app
    source = capture(core, ctx, "TEST 嵌入与投影租约从拿到写锁后起算。")
    with sqlite3.connect(core.storage.path) as db:
        db.execute("UPDATE work_items SET state='done' WHERE work_type='consolidate'")
    claim = worker._claim_idle
    waited = False

    def claim_after_waiting_for_writer(*args, **kwargs):
        nonlocal waited
        if not waited:
            waited = True
            _advance_worker_clock(clock, 7)
        return claim(*args, **kwargs)

    monkeypatch.setattr(worker, "_claim_idle", claim_after_waiting_for_writer)
    embed_windows = []

    class RecordingEmbed:
        def prepare_source(self, item_source, *, remaining_seconds):
            embed_windows.append(remaining_seconds)
            return item_source.ref

        def publish_source(self, prepared, **kwargs):
            return prepared

    receipt = core.drain_worker(ctx, max_items=1, remaining_seconds=25, lease_seconds=8, embed=RecordingEmbed())
    assert receipt.completed == 1
    assert embed_windows == [4.0]

    with core.storage.write(ctx) as tx:
        tx.work.enqueue("rebuild_projection", source.ref, source.revision, available_at=clock.utc_now())
    waited = False
    projection_windows = []
    real_projection = worker_projection.process_rebuild_projection

    def recording_projection(storage, item_clock, item_ctx, item, *, started, budget, **kwargs):
        projection_windows.append(budget - (item_clock.monotonic() - started))
        return real_projection(storage, item_clock, item_ctx, item, started=started, budget=budget, **kwargs)

    monkeypatch.setattr(worker, "process_rebuild_projection", recording_projection)
    proj_receipt = core.drain_worker(ctx, max_items=1, remaining_seconds=25, lease_seconds=8)
    assert proj_receipt.completed == 1
    assert projection_windows == [8.0]
