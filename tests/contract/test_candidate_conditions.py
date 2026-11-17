"""Candidate evaluations preserve the evidence boundary of claim conditions."""

from scope_recall.core.claims import Qualification
from test_candidate_lifecycle import Evaluator, _candidate_rows, _finish_source_work
from test_claims import app, capture, draft


def _evaluate(core, ctx, text, quote, original, evaluated, *, value="浅蓝"):
    """Evaluate a synthetic candidate and return its versions and evaluation row."""
    source = capture(core, ctx, text)
    proposal = draft(
        source,
        value,
        quote_override=quote,
        kind="fact",
        subject="已开启状态",
        predicate="背景色为",
        statement_kind="assertion",
        conditions=original,
    )
    with core.storage.write(ctx) as tx:
        saved = tx.claims.append(
            "TEST-scope",
            proposal,
            Qualification("proposed", "inferred_suggestion", "test_candidate"),
            recorded_at=core.clock.utc_now(),
        )
        tx.candidates.register(saved.ref, saved.revision, observed_at=core.clock.utc_now())
    _finish_source_work(core)
    evaluator = Evaluator({**proposal, "conditions": evaluated})
    receipt = core.drain_worker(ctx, max_items=8, remaining_seconds=10, consolidation=evaluator)
    assert evaluator.calls == 1 and receipt.failed == 0
    _, evaluations, _ = _candidate_rows(core)
    with core.storage.read(ctx) as tx:
        versions = tx.claims.slot("TEST-scope", {**proposal, "conditions": evaluated})
        original_versions = tx.claims.slot("TEST-scope", proposal)
    return saved, versions, original_versions, evaluations[0]


def test_unsupported_added_condition_is_dropped_and_recorded(app):
    core, ctx = app
    saved, _, versions, evaluation = _evaluate(
        core, ctx, "状态卡标题增加着色背景\n全部映射 蓝\n已开启 浅蓝", "已开启 浅蓝", [], ["全部映射 蓝"]
    )
    assert versions[-1].payload["conditions"] == []
    assert versions[-1].ref == saved.ref
    assert (versions[-1].state, versions[-1].reason) == ("proposed", "subject_not_bound")
    assert evaluation["reason"] == "unsupported_added_condition_dropped"


def test_supported_candidate_with_unsupported_added_condition_becomes_active(app):
    core, ctx = app
    saved, _, versions, evaluation = _evaluate(
        core, ctx, "全部映射 蓝\n已开启状态背景色为浅蓝。", "已开启状态背景色为浅蓝", [], ["全部映射 蓝"]
    )
    assert versions[-1].ref == saved.ref and versions[-1].state == "active"
    assert versions[-1].payload["conditions"] == []
    assert evaluation["reason"] == "unsupported_added_condition_dropped"


def test_supported_added_condition_is_kept(app):
    core, ctx = app
    _, versions, _, evaluation = _evaluate(
        core, ctx, "要是值班表调整，已开启状态背景色为浅蓝。", "已开启状态背景色为浅蓝", [], ["值班表调整"]
    )
    assert versions[-1].payload["conditions"] == ["值班表调整"]
    assert versions[-1].state == "active"
    assert evaluation["reason"] == "fact_active"


def test_only_unsupported_additions_are_dropped(app):
    core, ctx = app
    _, _, versions, evaluation = _evaluate(
        core,
        ctx,
        "其他设置 深蓝\n要是值班表调整，已开启状态背景色为浅蓝。",
        "已开启状态背景色为浅蓝",
        [],
        ["其他设置 深蓝", "值班表调整"],
    )
    with core.storage.read(ctx) as tx:
        retained = tx.claims.slot("TEST-scope", {**versions[0].payload, "conditions": ["值班表调整"]})
    assert retained[-1].payload["conditions"] == ["值班表调整"]
    assert retained[-1].state == "active"
    assert evaluation["reason"] == "unsupported_added_condition_dropped"


def test_unsupported_original_condition_still_fails_qualification(app):
    core, ctx = app
    _, versions, _, _ = _evaluate(
        core, ctx, "已开启状态背景色为浅蓝。", "已开启状态背景色为浅蓝", ["全部映射 蓝"], ["全部映射 蓝"]
    )
    assert versions[-1].state == "proposed"
    assert versions[-1].reason == "condition_not_supported"


def test_removed_original_condition_remains_removed(app):
    core, ctx = app
    _, versions, _, _ = _evaluate(
        core, ctx, "要是值班表调整，已开启状态背景色为浅蓝。", "已开启状态背景色为浅蓝", ["值班表调整"], []
    )
    assert versions[-1].payload["conditions"] == []
    assert (versions[-1].state, versions[-1].reason) == ("proposed", "qualified_statement")


def test_rewritten_original_condition_remains_subject_to_qualification(app):
    core, ctx = app
    _, versions, _, _ = _evaluate(
        core, ctx, "要是值班表调整，已开启状态背景色为浅蓝。", "已开启状态背景色为浅蓝", ["值班表调整"], ["值班表变更"]
    )
    assert versions[-1].payload["conditions"] == ["值班表变更"]
    assert (versions[-1].state, versions[-1].reason) == ("proposed", "qualified_statement")
