"""Host event sequences preserve turn identity and baseline fallback safety."""

from __future__ import annotations

import time
from contextvars import Context

import pytest

from tests.host.hermes.test_capture_boundaries import _rows
from tests.host.hermes.test_hermes_private_mode import _owner
from tests.host.hermes.test_steers import _row, _steer
from tests.host.hermes.test_turn_binding_recovery import _busy_hook, _hook, _post
from tests.store_clock import StoreMeteredClock

RAW = "TEST skill scaffold\nTEST question\nTEST trailing scaffold"
CLEAN = "TEST question"
REVIEW = "TEST review prompt"
MISSING_START = "capture_gap:turn_start_missing"


def test_unbound_cleaned_private_delivery_keeps_original_witnesses(adapter):
    """Cleaning the sync payload cannot replace a private turn's original source identities or dates."""
    provider, clock = adapter
    host = Context()
    owner, ctx = _owner(provider)
    owner.private_on(target="session", session=ctx.session_id)
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message=RAW)
    _hook(provider, "post_llm_call", host, turn_id="TEST-A", user_message=RAW, assistant_response="TEST answer")
    clock.now = "2026-09-06T13:00:00Z"
    owner.private_off(target="session", session=ctx.session_id)
    host.run(provider.sync_turn, CLEAN, "TEST answer")
    assert _rows(provider) == _rows(provider, "capture_inbox") == []
    assert not provider._retry.captures


def test_unbound_cleaned_delivery_does_not_recreate_deleted_opener(adapter):
    """A cleaned completion uses the original key and handled marker after owner deletion."""
    provider, _clock = adapter
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message=RAW)
    source = _rows(provider)[0]
    owner, _ctx = _owner(provider)
    owner.apply(owner.preview([source["event_id"]]))
    _hook(provider, "post_llm_call", host, turn_id="TEST-A", user_message=RAW, assistant_response="TEST answer")
    before = [row for row in _rows(provider) if row["role"] == "user"]
    assert before and all(row["read_blocked"] for row in before)
    host.run(provider.sync_turn, CLEAN, "TEST answer")
    after = [row for row in _rows(provider) if row["role"] == "user"]
    assert after == before
    assert all(row["read_blocked"] for row in after)


def test_evicted_uuid_does_not_authorize_a_late_request_to_rebind_current(adapter):
    """The carried old object remains proof even after its UUID leaves the optimization window."""
    provider, _clock = adapter
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST input A")
    host.run(provider.on_turn_start, 1, "TEST input A")
    _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST input A")
    old = host.copy()
    host.run(provider.sync_turn, "TEST input A", "TEST answer A")
    for number in range(2, 66):
        turn = f"TEST-{number}"
        host.run(provider.on_turn_start, number, "TEST input", turn_id=turn)
    _hook(provider, "pre_llm_call", host, turn_id="TEST-B", user_message="TEST input B")
    host.run(provider.on_turn_start, 66, "TEST input B")
    _hook(provider, "pre_api_request", host, turn_id="TEST-B", user_message="TEST input B")
    _post(provider, "TEST-B", host)
    before = provider.diagnostics.current_source_refs
    assert before
    binding = host.run(provider._turns._sync_binding.get)
    _hook(provider, "pre_api_request", old, turn_id="TEST-A", user_message="TEST input A")
    assert binding.turn_id == provider._active_turn_id == "TEST-B"
    assert provider.diagnostics.current_source_refs == before
    host.run(provider.sync_turn, "TEST input B", "TEST answer TEST-B")
    assert any(
        row["content"] == "TEST answer TEST-B" and ":sync_assistant:TEST-B@1" in row["source_event_key"]
        for row in _rows(provider)
    )


@pytest.mark.parametrize("mode", ["persistent", "fresh", "pooled"])
def test_missing_next_start_cannot_change_an_already_queued_binding(adapter, mode):
    """A confirmed old binding is not a writable substitute for a missing next start."""
    provider, _clock = adapter
    host = Context()
    following = host if mode == "persistent" else Context()
    if mode == "pooled":
        following.run(provider.on_turn_start, 0, "TEST old worker", turn_id="TEST-old-worker")
        following.run(provider.sync_turn, "TEST old worker", "TEST old answer")
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST input A")
    host.run(provider.on_turn_start, 1, "TEST input A")
    _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST input A")
    _post(provider, "TEST-A", host)
    queued = host.copy()
    binding = host.run(provider._turns._sync_binding.get)
    _hook(provider, "pre_llm_call", following, turn_id="TEST-B", user_message="TEST input B")
    _hook(provider, "pre_api_request", following, turn_id="TEST-B", user_message="TEST input B")
    _post(provider, "TEST-B", following)
    assert binding.turn_id == "TEST-A"
    assert MISSING_START in provider.diagnostics.pending_outcome_gaps
    queued.run(provider.sync_turn, "TEST input A", "TEST answer TEST-A")
    following.run(provider.sync_turn, "TEST input B", "TEST answer TEST-B")
    for turn in ("TEST-A", "TEST-B"):
        assert any(
            row["content"] == f"TEST answer {turn}" and f":sync_assistant:{turn}@1" in row["source_event_key"]
            for row in _rows(provider)
        )


def test_late_private_post_still_dates_reply_before_private_off(adapter):
    """Discarding late body caches must not discard the privacy witness for its queued sync."""
    provider, clock = adapter
    owner, ctx = _owner(provider)
    owner.private_on(target="session", session=ctx.session_id)
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST private A")
    host.run(provider.on_turn_start, 1, "TEST private A")
    queued = host.copy()
    host.run(provider.on_turn_start, 2, "TEST private B", turn_id="TEST-B")
    _hook(
        provider,
        "post_llm_call",
        queued,
        turn_id="TEST-A",
        user_message="TEST private A",
        assistant_response="TEST private answer A",
    )
    clock.now = "2026-09-06T13:00:00Z"
    owner.private_off(target="session", session=ctx.session_id)
    queued.run(provider.sync_turn, "TEST private A", "TEST private answer A")
    assert _rows(provider) == _rows(provider, "capture_inbox") == []


@pytest.mark.parametrize("event", ["on_session_end", "on_pre_compress"])
def test_boundary_budget_is_reserved_for_unanswered_user_input(adapter, monkeypatch, event):
    """Completed assistant caches cannot consume a boundary's one remaining chance to observe U."""
    provider, _clock = adapter
    host = Context()
    host.run(provider.on_turn_start, 1, "TEST A input", turn_id="TEST-A")
    history = [dict(role="user", content="TEST A input")]
    for number in range(3):
        history.extend(
            [
                dict(role="assistant", content=f"TEST interim {number}", tool_calls=[dict(id=f"TEST-tool-{number}")]),
                dict(role="tool", content="TEST tool output"),
            ]
        )
    history.append(dict(role="assistant", content="TEST answer"))
    _hook(
        provider,
        "post_llm_call",
        host,
        turn_id="TEST-A",
        user_message="TEST A input",
        assistant_response="TEST answer",
        conversation_history=history,
    )
    semantic = StoreMeteredClock(0.0)
    monkeypatch.setattr(time, "monotonic", semantic)
    record = provider._core.record_host_event

    def slow(context, source, **kwargs):
        if ":interim:" in source["source_event_key"]:
            semantic.advance(0.8)
        return record(context, source, **kwargs)

    monkeypatch.setattr(provider._core, "record_host_event", slow)
    incoming = [dict(role="user", content="TEST unanswered U", timestamp=1788695980), *history]
    getattr(provider, event)(incoming)
    assert any(row["content"] == "TEST unanswered U" for row in _rows(provider))
    assert "TEST-A" in provider._interim_said


@pytest.mark.parametrize("first_busy", [False, True])
def test_latest_opening_replaces_vetoed_record_in_both_directions(adapter, monkeypatch, first_busy):
    """Newer successful or skipped input replaces a vetoed notice's candidate."""
    provider, _clock = adapter
    host = Context()
    first = dict(
        turn_id="TEST-A",
        user_message="TEST notice A",
        conversation_history=[dict(role="user", content="TEST notice A", display_kind="internal_notification")],
    )
    second = dict(
        turn_id="TEST-B", user_message="TEST human B", conversation_history=[dict(role="user", content="TEST human B")]
    )
    if first_busy:
        _busy_hook(provider, monkeypatch, host, **first)
        _hook(provider, "pre_llm_call", host, **second)
    else:
        _hook(provider, "pre_llm_call", host, **first)
        _busy_hook(provider, monkeypatch, host, **second)
    host.run(provider.on_turn_start, 2, "TEST human B")
    _hook(provider, "pre_api_request", host, turn_id="TEST-B", user_message="TEST human B")
    host.run(provider.sync_turn, "TEST human B", "TEST answer B")
    rows = [row for row in _rows(provider) if row["content"] == "TEST human B"]
    assert len(rows) == 1 and rows[0]["origin"] == "human_direct"
    assert ":user:TEST-B@1" in rows[0]["source_event_key"]


def test_missing_opening_is_detected_by_start_ordinal_gap(adapter, monkeypatch):
    """A missed B hook cannot claim a vetoed skipped A when only one of two inputs was observed."""
    provider, _clock = adapter
    host = Context()
    host.run(provider.on_turn_start, 1, "TEST seed", turn_id="TEST-seed")
    _busy_hook(
        provider,
        monkeypatch,
        host,
        turn_id="TEST-A",
        user_message="TEST notice A",
        conversation_history=[dict(role="user", content="TEST notice A", display_kind="internal_notification")],
    )
    host.run(provider.on_turn_start, 3, "TEST human B")
    assert not any(row["content"] == "TEST human B" for row in _rows(provider))
    _hook(provider, "pre_api_request", host, turn_id="TEST-B", user_message="TEST human B")
    host.run(provider.sync_turn, "TEST human B", "TEST answer B")
    rows = [row for row in _rows(provider) if row["content"] == "TEST human B"]
    assert len(rows) == 1 and rows[0]["origin"] == "human_direct"
    assert ":user:TEST-B@1" in rows[0]["source_event_key"]


def test_session_delimiter_does_not_hide_tool_and_reply_receipts(adapter):
    """Source keys are opaque: a session's user delimiter is not an opening-event tag."""
    provider, _clock = adapter
    provider.on_session_switch("TEST-session:user:tail", reset=True)
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST input A")
    host.run(provider.on_turn_start, 1, "TEST input A")
    _hook(
        provider,
        "post_tool_call",
        host,
        turn_id="TEST-A",
        tool_call_id="TEST-tool",
        tool_name="terminal",
        result="TEST tool result",
    )
    host.run(provider.sync_turn, "TEST input A", "TEST answer A")
    assert len(provider.diagnostics.current_source_refs) == 3


def test_opening_receipt_uses_explicit_turn_even_with_user_delimiter(adapter):
    """The caller's full turn ID owns its private-span receipt and current source reference."""
    provider, _clock = adapter
    host = Context()
    turn = "TEST-A:user:tail"
    _hook(provider, "pre_llm_call", host, turn_id=turn, user_message="TEST public <private>TEST hidden</private>")
    stored = provider._identity.stored_session_id()
    assert provider._private_span_receipts[stored, turn] == 1
    assert provider.diagnostics.current_source_refs


@pytest.mark.parametrize("mode", ["persistent", "fresh", "pooled"])
def test_normal_two_turns_keep_every_message_in_their_own_turn(adapter, mode):
    """All supported context lifetimes carry a separately confirmed object for each normal start."""
    provider, _clock = adapter
    contexts = [Context(), Context()]
    if mode == "persistent":
        contexts[1] = contexts[0]
    elif mode == "pooled":
        contexts[1].run(provider.on_turn_start, 0, "TEST older worker", turn_id="TEST-worker-old")
    for number, host in enumerate(contexts, 1):
        turn = f"TEST-{number}"
        _hook(provider, "pre_llm_call", host, turn_id=turn, user_message=f"TEST input {turn}")
        host.run(provider.on_turn_start, number, f"TEST input {turn}")
        history = [
            dict(role="user", content=f"TEST input {turn}"),
            _row(_steer(f"TEST correction {turn}", message_id=f"TEST-steer-{number}", origin=False)),
        ]
        _hook(
            provider,
            "pre_api_request",
            host,
            turn_id=turn,
            user_message=f"TEST input {turn}",
            conversation_history=history,
        )
        _post(provider, turn, host)
        host.run(provider.sync_turn, f"TEST input {turn}", f"TEST answer {turn}")
    rows = _rows(provider)
    assert len(rows) == 8
    assert all(
        f"TEST-{row['content'][-1]}" in row["source_event_key"]
        for row in rows
        if not row["content"].startswith("TEST correction")
    )
    steers = [row for row in rows if row["content"].startswith("TEST correction")]
    assert len({row["source_event_key"] for row in steers}) == 2
    assert all(row["origin"] == "human_direct" for row in steers)
    assert MISSING_START not in provider.diagnostics.pending_outcome_gaps


def test_dispatch_keeps_host_payload_separate_from_admission_keywords(adapter):
    """Future host fields named like internal admission options cannot shadow the dispatcher policy."""
    provider, _clock = adapter
    _hook(
        provider,
        "pre_llm_call",
        turn_id="TEST-host-fields",
        user_message="TEST input",
        _wait="TEST data",
        _selected="TEST data",
    )
    assert any(row["content"] == "TEST input" for row in _rows(provider))


def test_explicit_private_refusal_is_not_retried_by_each_request(adapter, monkeypatch):
    """A final private refusal remains handled for this UUID while its first witness is retained."""
    provider, _clock = adapter
    owner, ctx = _owner(provider)
    owner.private_on(target="session", session=ctx.session_id)
    record, calls = provider._core.record_host_event, []

    def counted(*args, **kwargs):
        calls.append(1)
        return record(*args, **kwargs)

    monkeypatch.setattr(provider._core, "record_host_event", counted)
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST private input")
    host.run(provider.on_turn_start, 1, "TEST private input")
    for _retry in range(3):
        _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST private input")
    assert len(calls) == 1
    assert _rows(provider) == _rows(provider, "capture_inbox") == []


def test_transient_opening_failure_retries_with_its_original_time(adapter, monkeypatch):
    """Temporary store failure can be retried by a request without replacing the first observation time."""
    from scope_recall.contracts import ContractError

    provider, clock = adapter
    host = Context()
    record = provider._core.record_host_event
    first_seen = clock.now

    def busy(*args, **kwargs):
        raise ContractError("STORAGE_UNAVAILABLE")

    monkeypatch.setattr(provider._retry, "start", lambda: None)
    monkeypatch.setattr(provider._core, "record_host_event", busy)
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST retry input")
    host.run(provider.on_turn_start, 1, "TEST retry input")
    clock.now = "2026-09-06T13:00:00Z"
    monkeypatch.setattr(provider._core, "record_host_event", record)
    _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST retry input")
    rows = _rows(provider)
    assert len(rows) == 1 and rows[0]["occurred_at"] == first_seen
    assert not provider._retry.captures


def test_first_resumed_start_defers_uncertain_skipped_input_until_request(adapter, monkeypatch):
    """A resumed ordinal cannot prove that the only skipped hook was not a vetoed earlier input."""
    provider, _clock = adapter
    host = Context()
    _busy_hook(
        provider,
        monkeypatch,
        host,
        turn_id="TEST-A",
        user_message="TEST notice A",
        conversation_history=[dict(role="user", content="TEST notice A", display_kind="internal_notification")],
    )
    host.run(provider.on_turn_start, 11, "TEST human B")
    assert _rows(provider) == []
    _hook(provider, "pre_api_request", host, turn_id="TEST-B", user_message="TEST human B")
    host.run(provider.sync_turn, "TEST human B", "TEST answer B")
    rows = [row for row in _rows(provider) if row["role"] == "user"]
    assert len(rows) == 1 and rows[0]["origin"] == "human_direct"
    assert ":user:TEST-B@1" in rows[0]["source_event_key"]


@pytest.mark.parametrize("evict", [False, True])
def test_late_opening_window_and_current_request_correction(adapter, evict):
    """An aged-out opening has baseline ambiguity; the carried current request still confirms its own UUID."""
    provider, _clock = adapter
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST input A")
    host.run(provider.on_turn_start, 1, "TEST input A")
    _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST input A")
    late = host.copy()
    if evict:
        for number in range(2, 66):
            host.run(provider.on_turn_start, number, "TEST other input", turn_id=f"TEST-other-{number}")
    _hook(provider, "pre_llm_call", host, turn_id="TEST-B", user_message="TEST input B")
    host.run(provider.on_turn_start, 66 if evict else 2, "TEST input B")
    _hook(provider, "pre_llm_call", late, turn_id="TEST-A", user_message="TEST input A")
    assert provider._active_turn_id == ("TEST-A" if evict else "TEST-B")
    _hook(provider, "pre_api_request", host, turn_id="TEST-B", user_message="TEST input B")
    host.run(provider.sync_turn, "TEST input B", "TEST answer B")
    assert any(
        row["content"] == "TEST answer B" and ":sync_assistant:TEST-B@1" in row["source_event_key"]
        for row in _rows(provider)
    )


@pytest.mark.parametrize("same_payload", [False, True])
def test_missing_start_with_stale_context_has_no_delivery_token(adapter, same_payload):
    """Without a new start, an unused old object or identical replay cannot prove the next delivery's owner."""
    provider, _clock = adapter
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST identical input")
    host.run(provider.on_turn_start, 1, "TEST identical input")
    _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST identical input")
    if same_payload:
        host.run(provider.sync_turn, "TEST identical input", "TEST identical answer")
    _hook(provider, "pre_llm_call", host, turn_id="TEST-B", user_message="TEST identical input")
    _hook(provider, "pre_api_request", host, turn_id="TEST-B", user_message="TEST identical input")
    host.run(provider.sync_turn, "TEST identical input", "TEST identical answer")
    replies = [row for row in _rows(provider) if row["role"] == "assistant"]
    assert len(replies) == 1 and ":sync_assistant:TEST-A@1" in replies[0]["source_event_key"]
    assert MISSING_START in provider.diagnostics.pending_outcome_gaps


def _review_request(provider, context):
    """The inference request of a detached review: the session and context of the finished turn, a turn id of
    its own, the review's prompt as the message, and neither an opening hook nor a start before it."""
    _hook(
        provider,
        "pre_api_request",
        context,
        turn_id="TEST-review",
        user_message=REVIEW,
        conversation_history=[
            dict(role="user", content="TEST input A"),
            dict(role="assistant", content="TEST answer TEST-A"),
            dict(role="user", content=REVIEW),
        ],
    )


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("next_started", [False, True])
def test_detached_review_request_is_not_the_persons_turn(adapter, confirmed, next_started):
    """A review Hermes runs after a reply stores nothing and leaves the finished turn's completion its own."""
    provider, _clock = adapter
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST input A")
    host.run(provider.on_turn_start, 1, "TEST input A")
    if confirmed:
        _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST input A")
        _post(provider, "TEST-A", host)
    finished = host.copy()
    if next_started:
        _hook(provider, "pre_llm_call", host, turn_id="TEST-B", user_message="TEST input B")
        host.run(provider.on_turn_start, 2, "TEST input B")
    active, refs = provider._active_turn_id, provider.diagnostics.current_source_refs
    _review_request(provider, finished)
    assert (provider._active_turn_id, provider.diagnostics.current_source_refs) == (active, refs)
    finished.run(provider.sync_turn, "TEST input A", "TEST answer TEST-A")
    rows = _rows(provider)
    assert REVIEW not in {row["content"] for row in rows}
    assert any(
        row["content"] == "TEST answer TEST-A" and ":sync_assistant:TEST-A@1" in row["source_event_key"] for row in rows
    )
    assert MISSING_START not in provider.diagnostics.pending_outcome_gaps


def test_only_the_started_turns_own_request_names_it(adapter):
    """A start with no opening hook takes its UUID from the request that carries the text it started with."""
    provider, _clock = adapter
    host = Context()
    host.run(provider.on_turn_start, 1, "TEST input A")
    _review_request(provider, host)
    assert _rows(provider) == []
    _hook(provider, "pre_api_request", host, turn_id="TEST-A", user_message="TEST input A")
    host.run(provider.sync_turn, "TEST input A", "TEST answer A")
    rows = _rows(provider)
    assert sorted(row["content"] for row in rows) == ["TEST answer A", "TEST input A"]
    assert all("TEST-A@1" in row["source_event_key"] for row in rows)


@pytest.mark.parametrize("entry", ["dispatcher", "public"])
def test_input_entries_ignore_explicitly_unbound_session(adapter, entry):
    """Neither timed dispatch nor blocking public admission can observe input for an unbound session."""
    provider, _clock = adapter
    provider._unbound_session = True
    if entry == "dispatcher":
        _hook(provider, "pre_llm_call", turn_id="TEST-A", user_message="TEST unbound input")
    else:
        provider.observe_pre_llm(turn_id="TEST-A", user_message="TEST unbound input")
    assert not provider._turns._opening
    assert _rows(provider) == []
