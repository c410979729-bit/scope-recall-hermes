"""Corrected facts and inventory rows retain the information needed to read them."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from scope_recall.core import injection_format, session_injection
from scope_recall.core.recall_budget import estimate_tokens
from scope_recall.maintenance import memory

from tests.contract.test_claims import app


def make_notice(monkeypatch, **overrides):
    payload = dict(subject="DSH", predicate="API 调用可用性", value_text="能用", conditions=["登录后"])
    payload.update(overrides)
    claim = SimpleNamespace(payload=payload, state="active")
    tx = SimpleNamespace(claims=SimpleNamespace(version=lambda *_: claim))
    replacement = SimpleNamespace(kind="claim", ref="claim-new", revision=2, temporal_status="current")
    row = dict(object_kind="claim", object_ref="claim-old", revision=1, notice_marker=None)
    monkeypatch.setattr(session_injection, "check_claim_release", lambda *args, **kwargs: None)
    return session_injection._make_notice(tx, row, "claim:claim-old@1", replacement)


@pytest.mark.parametrize("style", ["json", "compact", "outline"])
def test_corrected_notice_includes_statement_and_conditions(monkeypatch, style):
    notice = make_notice(monkeypatch)
    text = injection_format.notice(notice, style)
    assert all(part in text for part in ("DSH", "API 调用可用性", "能用", "登录后"))


@pytest.mark.parametrize("field", ["subject", "predicate", "conditions"])
def test_corrected_notice_with_secret_in_statement_withholds_value(monkeypatch, field):
    secret = "password=" + "SYNTHETIC9example"
    notice = make_notice(monkeypatch, **{field: [secret] if field == "conditions" else secret})
    assert not notice["corrected"]
    assert "能用" not in notice["text"] and secret not in notice["text"]


@pytest.mark.parametrize("style", ["json", "compact", "outline"])
def test_notice_budget_preserves_short_value(monkeypatch, style):
    notice = make_notice(monkeypatch, subject="Long subject " * 100, conditions=["Condition " * 100])
    notice["text"] = injection_format.notice(notice, style)
    notice["_text_prefix"] = injection_format.notice_prefix(notice, style)
    notice["_format"] = style
    selected = session_injection._select_notices([notice], 80)
    assert selected and "能用" in selected[0]["text"]
    assert "subject:" in selected[0]["text"] and "predicate:" in selected[0]["text"]
    assert "conditions:" in selected[0]["text"]
    assert estimate_tokens("Memory updates:\n" + selected[0]["text"]) <= 80


@pytest.mark.parametrize("style", ["json", "compact", "outline"])
@pytest.mark.parametrize("limit", [0, 80])
def test_long_notice_shrinks_strictly_and_terminates(monkeypatch, style, limit):
    notice = make_notice(
        monkeypatch,
        subject="Long subject " * 100,
        predicate="Long predicate " * 100,
        conditions=["Long condition " * 100],
        value_text="Long value " * 100,
    )
    notice["text"] = injection_format.notice(notice, style)
    notice["_text_prefix"] = injection_format.notice_prefix(notice, style)
    notice["_format"] = style
    shrink = session_injection._shrink_statement_notice
    steps = 0

    def checked_shrink(candidate):
        nonlocal steps
        steps += 1
        assert steps < 100
        before = len(candidate["text"])
        changed = shrink(candidate)
        assert not changed or len(candidate["text"]) < before
        return changed

    monkeypatch.setattr(session_injection, "_shrink_statement_notice", checked_shrink)
    selected = session_injection._select_notices([notice], limit)
    assert steps and (not selected or estimate_tokens("Memory updates:\n" + selected[0]["text"]) <= limit)
    assert not selected if limit == 0 else selected


def test_notice_shrink_does_not_expand_short_fields(monkeypatch):
    notice = make_notice(monkeypatch, subject="AB", predicate="CD", conditions=["X"])
    notice["_format"] = "json"
    before = len(notice["text"])
    changed = session_injection._shrink_statement_notice(notice)
    assert not changed or len(notice["text"]) < before


@pytest.mark.parametrize("action", ["list", "search", "history"])
def test_inventory_rows_keep_a_long_fact_whole(capsys, action):
    fact = dict(subject="Long subject " * 5, predicate="Long predicate " * 5, value_text="complete value")
    item = dict(kind="claim", ref="claim-test", revision=1, value=fact)
    memory.render(dict(status="ok", action=action, data=dict(items=[item])), as_json=False)
    output = capsys.readouterr().out
    assert all(part in output for part in (fact["subject"], fact["predicate"], fact["value_text"]))
    assert json.dumps(fact, ensure_ascii=False) in output


def test_inventory_rows_keep_other_kinds_at_the_overview_width(capsys):
    said = "word " * 40
    fact = dict(subject="Long subject " * 5, predicate="Long predicate " * 5, value_text="complete value")
    items = [
        dict(kind="event", ref="event-test", revision=1, content=said),
        dict(kind="claim", ref="claim-test", revision=1, value=fact),
    ]
    memory.render(dict(status="ok", action="list", data=dict(items=items)), as_json=False)
    header, event_row, fact_row = capsys.readouterr().out.splitlines()
    assert event_row.endswith(said[:72]) and said[:73] not in event_row
    assert fact_row.endswith(json.dumps(fact, ensure_ascii=False))
    # The whole fact lengthens its own line only: the header and the shorter row carry no padding up to it.
    assert header == header.rstrip() and len(event_row) < len(fact_row)


def test_real_correction_and_memory_list(app, monkeypatch, capsys):
    from scope_recall.core.management import MemoryManagement
    from scope_recall.core.restore import InstallationMaintenance
    from scope_recall.core.session_brief import mark_delivered
    from scope_recall.maintenance import cli

    from tests.contract.test_memory_health import separate_fact

    core, ctx = app
    subject = "TEST long subject " * 6
    item = separate_fact(core, ctx, subject, "blue")
    with core.storage.write(ctx) as tx:
        mark_delivered(tx, ctx, [dict(kind="claim", ref=item.ref, revision=1)], core.clock.utc_now(), 0, "question")
    owner = MemoryManagement(core, InstallationMaintenance(replace(ctx, allowed_scope_ids=ctx.binding.scope_ids)))
    owner.revise(item.ref, expected_revision=1, value="green")
    text = core.prepare_session_injection(ctx, "")["_injection"]["text"]
    assert subject in text and "color" in text and "green" in text
    monkeypatch.setattr(memory, "open_management", lambda args, **kwargs: owner)
    assert cli.main(["memory", "list"]) == 0
    output = capsys.readouterr().out
    assert subject in output and "color" in output and "green" in output
