"""A worker's summary that does not qualify does not take the claims beside it down.

A resume that names as the user's goal something the user had not asked for breaks the resume rules
(``goal_authority``). A paged consolidation drops such a summary and keeps its claims; a single page must do the
same, not roll the whole result back, pay for a second model call and, when that answer fails too, lose every
claim in the page.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest
from scope_recall.contracts import ContractError
from scope_recall.core.episodes import source_watermark

from tests.contract.test_claims import app, capture, draft  # noqa: F401  (fixture)
from tests.contract.test_worker import (  # noqa: F401  (fixture)
    FakeConsolidation,
    _mark_embed_done,
    consolidation_payload,
    worker_app,
)


def _goal_from(source):
    refs = [f"{source.ref}@{source.revision}"]
    return dict(
        episode_ref=None,
        goal=dict(text=source.event["content"], evidence_refs=refs),
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


def _stored_values(core):
    with sqlite3.connect(core.storage.path) as db:
        return {
            json.loads(payload)["value_text"] for (payload,) in db.execute("SELECT payload_json FROM claim_versions")
        }


def test_an_unqualified_goal_is_dropped_and_the_claims_beside_it_are_kept(worker_app):
    core, ctx, _clock = worker_app
    said = capture(core, ctx, "TEST-project 配色 蓝色。", key="TEST-rc36/said")
    # A question is never a goal (goal_authority), and both messages share one page.
    asked = capture(core, ctx, "TEST 要不要先把海报整理好？", key="TEST-rc36/asked")
    _mark_embed_done(core)

    def builder(sources, episode_ref=None):
        by_ref = {source.ref: source for source in sources}
        return consolidation_payload(
            *sources, claims=[draft(by_ref[said.ref], "蓝色")], resume_proposals=[_goal_from(by_ref[asked.ref])]
        )

    model = FakeConsolidation(builder)
    receipt = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=10)

    assert receipt.completed == 1 and model.calls == 1
    assert "蓝色" in _stored_values(core)
    with sqlite3.connect(core.storage.path) as db:
        errors = db.execute("SELECT stage,error_code,error_field FROM work_error_details").fetchall()
        outcomes = db.execute("SELECT disposition,detail FROM consolidation_outcomes").fetchall()
    assert errors == [("summary", "DERIVATION_INVALID", "goal_authority")]
    assert outcomes == [("partial", "resume_qualification_failed")]


def test_a_result_submitted_outside_the_worker_is_still_rejected_whole(app):
    core, ctx = app
    said = capture(core, ctx, "TEST-project 配色 蓝色。", key="TEST-rc36/direct-said")
    asked = capture(core, ctx, "TEST 要不要先把海报整理好？", key="TEST-rc36/direct-asked")
    value = consolidation_payload(said, asked, claims=[draft(said, "蓝色")], resume_proposals=[_goal_from(asked)])

    with pytest.raises(ContractError, match="goal_authority"):
        core.records.accept_consolidation(ctx, value, scope_id="TEST-scope", remaining_seconds=10)

    assert "蓝色" not in _stored_values(core)


@pytest.mark.parametrize("missing_field", ["goal", "next_step"])
def test_malformed_resume_leaves_its_valid_summary_and_claim_siblings(worker_app, missing_field):
    core, context, _ = worker_app
    ctx = replace(context, task_anchor="TEST-report")
    text = "TEST 整理验收报告。\nTEST-project 配色  蓝色。"
    source = capture(core, ctx, text)
    goal = _goal_from(source)
    goal["goal"]["text"] = text.splitlines()[0]
    malformed = dict(goal)
    del malformed[missing_field]
    good = draft(source, quote_override="TEST-project 配色 蓝色 .")
    bad_shape = dict(good, value_text=[])
    bad_quote = draft(source, value="绿色", quote_override="TEST-project 配色绿色。")
    unsupported_value = dict(good, value_text="红色")
    model = FakeConsolidation(
        lambda sources, **kw: consolidation_payload(
            *sources, claims=[bad_shape, bad_quote, unsupported_value, good], resume_proposals=[malformed, goal]
        )
    )

    receipt = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=10)

    assert receipt.completed == 1 and receipt.retried == receipt.failed == 0 and model.calls == 1
    assert core.episodes(ctx)[0].resume["goal"] == goal["goal"]
    with core.storage.read(ctx) as tx:
        db = tx._check()
        (row,) = db.execute("SELECT payload_json,state FROM claim_versions").fetchall()
        stored = json.loads(row[0])
        errors = db.execute("SELECT stage,error_code,error_field FROM work_error_details ORDER BY detail_id").fetchall()
        (outcome,) = db.execute("SELECT disposition,detail FROM consolidation_outcomes").fetchall()
    assert stored["value_text"] == "蓝色"
    assert row["state"] == "active"
    (span,) = stored["evidence_spans"]
    assert text[span["start"] : span["end"]] == span["quote"] == text.splitlines()[1]
    assert [tuple(error) for error in errors] == [
        ("claim", "INPUT_INVALID", "claim_proposals[0].value_text"),
        ("claim", "DERIVATION_INVALID", "claim_proposals[1].evidence_spans[0].quote"),
        ("summary", "INPUT_INVALID", f"resume_proposals[0].{missing_field}"),
        ("claim", "DERIVATION_INVALID", "evidence_value"),
    ]
    assert outcome["disposition"] == "partial"


def test_unconfirmed_work_goal_leaves_the_fact_and_records_the_refusal(worker_app):
    core, ctx, _ = worker_app
    source = capture(core, ctx, "TEST-project 配色 蓝色。")
    model = FakeConsolidation(
        lambda sources, **kw: consolidation_payload(
            *sources, claims=[draft(source)], resume_proposals=[_goal_from(source)]
        )
    )
    receipt = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=10)
    assert receipt.completed == 1 and receipt.retried == receipt.failed == 0
    assert _stored_values(core) == {"蓝色"}
    assert core.episodes(ctx)[0].resume is None
    with core.storage.read(ctx) as tx:
        (error,) = tx._check().execute("SELECT error_field FROM work_error_details").fetchall()
        assert error["error_field"] == "work_goal_unconfirmed"


def test_unauthorized_resume_artifact_rolls_back_siblings_even_without_work_goal(worker_app):
    core, ctx, _ = worker_app
    source = capture(core, ctx, "TEST-project 配色 蓝色。")
    proposal = dict(_goal_from(source), artifact_refs=["TEST-unavailable-artifact@1"])
    model = FakeConsolidation(
        lambda sources, **kw: consolidation_payload(*sources, claims=[draft(source)], resume_proposals=[proposal])
    )
    receipt = core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=10)
    assert receipt.completed == 0 and receipt.obsolete == 1
    assert _stored_values(core) == set()
    assert core.episodes(ctx)[0].resume is None


def test_sqlite_summary_failure_rolls_back_claim_and_rejection_rows(worker_app):
    core, context, _ = worker_app
    ctx = replace(context, task_anchor="TEST-report")
    source = capture(core, ctx, "TEST-project 配色 蓝色。")
    good = _goal_from(source)
    with core.storage.write(ctx) as tx:
        tx._check(write=True).execute(
            """CREATE TRIGGER test_resume_failure BEFORE INSERT ON episode_versions
               WHEN new.resume_json IS NOT NULL BEGIN SELECT RAISE(ABORT, 'synthetic resume write failure'); END"""
        )
    model = FakeConsolidation(
        lambda sources, **kw: consolidation_payload(
            *sources, claims=[draft(source)], resume_proposals=[dict(good, goal=None), good]
        )
    )

    with pytest.raises(sqlite3.IntegrityError, match="synthetic resume write failure"):
        core.drain_worker(ctx, consolidation=model, max_items=1, remaining_seconds=10)

    assert _stored_values(core) == set()
    assert core.episodes(ctx)[0].resume is None
    with core.storage.read(ctx) as tx:
        db = tx._check()
        assert db.execute("SELECT count(*) FROM work_error_details").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM consolidation_outcomes").fetchone()[0] == 0
