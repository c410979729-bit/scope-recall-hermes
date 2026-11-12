"""Copied host contexts cover vetoes, late callbacks and completed-turn recovery."""

from __future__ import annotations

import sqlite3
import threading
from contextvars import Context, copy_context

import pytest
from scope_recall.adapters.hermes import turn_capture
from scope_recall.adapters.hermes.hooks import _global_callback, _register_adapter_instance

from tests.host.hermes.test_capture_boundaries import _human_words, _rows
from tests.host.hermes.test_hermes_private_mode import _owner
from tests.host.hermes.test_steers import _row, _steer, telegram  # noqa: F401


def _hook(provider, event, context=None, **kwargs):
    """Run a registered callback on a helper thread in the host caller's copied context."""
    _register_adapter_instance(provider)
    copied = context.copy() if context is not None else copy_context()
    payload = dict(session_id=provider._identity.session_id, **kwargs)
    errors = []

    def run():
        try:
            copied.run(_global_callback(event), **payload)
        except BaseException as exc:  # noqa: BLE001 - surface helper-thread failures in the test
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(10)
    assert not thread.is_alive()
    if errors:
        raise errors[0]


def _post(provider, turn, context=None):
    """Observe one synthetic completed turn containing a visible intermediate reply."""
    _hook(
        provider,
        "post_llm_call",
        context,
        turn_id=turn,
        user_message=f"TEST input {turn}",
        assistant_response=f"TEST answer {turn}",
        conversation_history=[
            dict(role="user", content=f"TEST input {turn}"),
            dict(role="assistant", content=f"TEST interim {turn}", tool_calls=[dict(id=f"TEST-tool-{turn}")]),
            dict(role="tool", content="TEST tool result"),
            dict(role="assistant", content=f"TEST answer {turn}"),
        ],
    )


def _busy_hook(provider, monkeypatch, context=None, **kwargs):
    """Exercise actual timed admission on a copied helper thread while this thread owns the adapter lock."""
    monkeypatch.setattr("scope_recall.adapters.hermes.hooks._SESSION_WAIT_CAP_S", 0.01)
    with provider._lock:
        _hook(provider, "pre_llm_call", context, **kwargs)


@pytest.mark.parametrize("event", ["pre_llm_call", "pre_api_request"])
@pytest.mark.parametrize("hidden", [False, True])
def test_request_hooks_do_not_restore_untyped_history(telegram, event, hidden):
    """Old untyped input has no stable link to its original source or deletion state."""
    history = [dict(role="user", content="TEST old untyped input", timestamp=1788695800)]
    if hidden:
        history.append(dict(role="assistant", content="TEST interruption", display_kind="hidden"))
    history.append(dict(role="user", content="TEST current input"))
    if event == "pre_api_request":
        # A request writes for the turn the host started, and for no other.
        telegram.on_turn_start(1, "TEST current input")
    _hook(telegram, event, turn_id="TEST-current", user_message="TEST current input", conversation_history=history)
    assert _human_words(telegram) == ["TEST current input"]


def test_unanswered_lookup_failure_waits_for_another_observation(telegram, monkeypatch):
    """An unknown deletion/deduplication result must not create a differently keyed copy."""
    lookup = telegram._core.said_in_session
    history = [
        dict(role="user", content="TEST restored input", timestamp=1788695800),
        dict(role="user", content="TEST current input"),
    ]

    def busy(*args, **kwargs):
        raise sqlite3.OperationalError("TEST locked")

    monkeypatch.setattr(telegram._core, "said_in_session", busy)
    telegram.on_session_end(history)
    assert _human_words(telegram) == []
    assert not telegram._retry.captures
    monkeypatch.setattr(telegram._core, "said_in_session", lookup)
    telegram.on_session_end(history)
    assert [(row["content"], row["origin"]) for row in _rows(telegram)] == [("TEST restored input", "origin_unknown")]


def test_untimed_unanswered_history_stays_blocked_after_private_off(adapter):
    """Discovery after an interval closes cannot date its old message outside that interval."""
    provider, clock = adapter
    owner, ctx = _owner(provider)
    owner.private_on(target="session", session=ctx.session_id)
    clock.now = "2026-09-06T13:00:00Z"
    owner.private_off(target="session", session=ctx.session_id)
    provider.on_session_end(
        [dict(role="user", content="TEST private old input"), dict(role="user", content="TEST next input")]
    )
    assert _rows(provider) == _rows(provider, "capture_inbox") == []
    assert not provider._retry.captures


@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("mode", ["same_thread", "fresh_context", "pooled_thread"])
def test_vetoed_opening_does_not_own_the_next_turn(adapter, monkeypatch, busy, mode):
    """The next request corrects the exact object the host will later copy into sync."""
    provider, _clock = adapter
    a_context, b_context = Context(), Context()
    if mode == "same_thread":
        b_context = a_context
    if mode == "pooled_thread":
        b_context.run(provider.on_turn_start, 0, "TEST seed input", turn_id="TEST-seed")
    _hook(provider, "pre_llm_call", a_context, turn_id="TEST-A", user_message="TEST input A")
    # A is vetoed: neither start nor sync is delivered.
    if busy:
        monkeypatch.setattr("scope_recall.adapters.hermes.hooks._SESSION_WAIT_CAP_S", 0.01)
        with provider._lock:
            _hook(provider, "pre_llm_call", b_context, turn_id="TEST-B", user_message="TEST input B")
    b_context.run(provider.on_turn_start, 2, "TEST input B")
    bound = b_context.run(provider._turns._sync_binding.get)
    _hook(provider, "pre_api_request", b_context, turn_id="TEST-B", user_message="TEST input B")
    assert b_context.run(provider._turns._sync_binding.get) is bound
    _post(provider, "TEST-B", b_context)
    b_context.run(provider.sync_turn, "TEST input B", "TEST answer TEST-B")
    rows = _rows(provider)
    assert len(rows) == 4
    assert [row["content"] for row in rows if ":user:TEST-A@1" in row["source_event_key"]] == ["TEST input A"]
    assert all("TEST-B" in row["source_event_key"] for row in rows if row["content"] != "TEST input A")


@pytest.mark.parametrize("event", ["pre_llm_call", "pre_api_request"])
@pytest.mark.parametrize("inherited", [False, True])
def test_late_request_hook_keeps_new_turn_state(adapter, event, inherited):
    """A known older UUID cannot mutate the active input, pending UUID or recall fence."""
    provider, _clock = adapter
    host = Context()
    host.run(provider.on_turn_start, 1, "TEST old input", turn_id="TEST-A")
    late = host.copy() if inherited else Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-B", user_message="TEST new input")
    host.run(provider.on_turn_start, 2, "TEST new input")
    before = (provider._active_turn_id, provider._turns._opening, provider.diagnostics.current_source_refs)
    assert before[2]
    bound = host.run(provider._turns._sync_binding.get)
    _hook(provider, event, late, turn_id="TEST-A", user_message="TEST old input")
    assert (provider._active_turn_id, provider._turns._opening, provider.diagnostics.current_source_refs) == before
    assert host.run(provider._turns._sync_binding.get) is bound
    host.run(provider.sync_turn, "TEST new input", "TEST new answer")
    assert any(row["content"] == "TEST new answer" and "TEST-B" in row["source_event_key"] for row in _rows(provider))


def test_pre_hook_does_not_create_a_host_binding(adapter):
    """Only the inline start may establish the binding propagated to the memory worker."""
    provider, _clock = adapter
    host = Context()
    _hook(provider, "pre_llm_call", host, turn_id="TEST-A", user_message="TEST input A")
    assert provider._turns._current_binding is None
    assert host.run(provider._turns._sync_binding.get) is None
    host.run(provider.on_turn_start, 1, "TEST input A")
    assert host.run(provider._turns._sync_binding.get) is not None
    host.run(provider.shutdown)
    assert host.run(provider._turns._sync_binding.get) is None


def test_unbound_sync_uses_each_delivered_uuid_and_replays_without_duplicates(adapter):
    """Without a start, ordinary sequential pre-hooks still supply the baseline UUID and handled marker."""
    provider, _clock = adapter
    host = Context()
    for turn in ("TEST-A", "TEST-B"):
        _hook(provider, "pre_llm_call", host, turn_id=turn, user_message=f"TEST input {turn}")
        host.run(provider.sync_turn, f"TEST input {turn}", f"TEST answer {turn}")
        host.run(provider.sync_turn, f"TEST input {turn}", f"TEST answer {turn}")
    assert len(_rows(provider)) == 4
    assert len({row["source_event_key"] for row in _rows(provider)}) == 4
    assert {row["content"] for row in _rows(provider)} == {
        "TEST input TEST-A",
        "TEST input TEST-B",
        "TEST answer TEST-A",
        "TEST answer TEST-B",
    }
    assert all(row["content"].rsplit(" ", 1)[-1] in row["source_event_key"] for row in _rows(provider))


def test_next_sync_writes_earlier_completed_snapshots_with_their_own_session(adapter):
    """A skipped sync leaves an earlier completed snapshot for the next ordered memory task."""
    provider, _clock = adapter
    host = Context()
    host.run(provider.on_turn_start, 1, "TEST input TEST-A", turn_id="TEST-A")
    original = provider._identity.stored_session_id()
    _post(provider, "TEST-A", host)
    provider.on_session_switch("TEST-second-session", reason="compression")
    host.run(provider.on_turn_start, 2, "TEST input TEST-B", turn_id="TEST-B")
    _post(provider, "TEST-B", host)
    host.run(provider.sync_turn, "TEST input TEST-B", "TEST answer TEST-B")
    rows = {row["content"]: row for row in _rows(provider)}
    assert rows["TEST interim TEST-A"]["session_id"] == original
    assert ":interim:TEST-A:1@1" in rows["TEST interim TEST-A"]["source_event_key"]
    assert ":interim:TEST-B:1@1" in rows["TEST interim TEST-B"]["source_event_key"]


def test_unbound_sync_flushes_completed_snapshots_with_original_ids(adapter):
    """The baseline fallback drains all cached messages under each one's preserved metadata."""
    provider, _clock = adapter
    host = Context()
    host.run(provider.on_turn_start, 1, "TEST input TEST-A", turn_id="TEST-A")
    _post(provider, "TEST-A", host)
    Context().run(provider.sync_turn, "TEST input TEST-A", "TEST answer TEST-A")
    assert "TEST-A" not in provider._interim_said
    assert any(row["content"] == "TEST interim TEST-A" for row in _rows(provider))


def test_new_uuid_is_accepted_from_an_older_worker_context(adapter):
    """A pooled worker's old ContextVar value alone cannot make a new opener late."""
    provider, _clock = adapter
    worker, other = Context(), Context()
    worker.run(provider.on_turn_start, 1, "TEST old worker input", turn_id="TEST-old")
    other.run(provider.on_turn_start, 2, "TEST other input", turn_id="TEST-other")
    _hook(provider, "pre_llm_call", worker, turn_id="TEST-new", user_message="TEST new input")
    worker.run(provider.on_turn_start, 3, "TEST new input")
    _hook(provider, "pre_api_request", worker, turn_id="TEST-new", user_message="TEST new input")
    worker.run(provider.sync_turn, "TEST new input", "TEST new answer")
    assert len(_rows(provider)) == 2
    assert all("TEST-new" in row["source_event_key"] for row in _rows(provider))


@pytest.mark.parametrize("event", ["on_pre_compress", "on_session_end"])
def test_session_hooks_leave_completed_snapshots_for_sync(adapter, event):
    """Session boundaries preserve the baseline historical-input budget instead of draining assistant caches."""
    provider, _clock = adapter
    provider.on_turn_start(1, "TEST input TEST-A", turn_id="TEST-A")
    _post(provider, "TEST-A")
    getattr(provider, event)([])
    assert not any(row["content"] == "TEST interim TEST-A" for row in _rows(provider))
    provider.sync_turn("TEST input TEST-A", "TEST answer TEST-A")
    assert any(row["content"] == "TEST interim TEST-A" for row in _rows(provider))


def test_completed_snapshot_capacity_reports_eviction(adapter, monkeypatch):
    """Bounded memory discards the earliest completed turn with an explicit capture gap."""
    provider, _clock = adapter
    monkeypatch.setattr(turn_capture, "_CACHED_TURNS", 2, raising=False)
    for number in range(3):
        turn = f"TEST-{number}"
        provider.on_turn_start(number, "TEST input", turn_id=turn)
        _post(provider, turn)
    assert len(provider._interim_said) == 2
    assert "TEST-0" not in provider._interim_said
    assert "capture_gap:completed_turn_cache_limit" in provider.diagnostics.pending_outcome_gaps


@pytest.mark.parametrize("notice", [False, True])
def test_busy_start_preserves_typed_origin_after_host_redaction(adapter, monkeypatch, notice):
    """Source classification belongs to the delivered UUID, independent of the host's text transformation."""
    provider, _clock = adapter
    raw, clean = "TEST notice Bearer synthetic-value", "TEST notice Bearer [redacted]"
    row = dict(role="user", content=raw)
    if notice:
        row["display_kind"] = "internal_notification"
    _busy_hook(provider, monkeypatch, turn_id="TEST-notice", user_message=raw, conversation_history=[row])
    provider.on_turn_start(1, clean)
    provider.sync_turn(clean, "TEST answer")
    rows = [row for row in _rows(provider) if row["role"] == "user"]
    assert len(rows) == 1 and rows[0]["content"] == clean
    assert rows[0]["origin"] == ("host_generated" if notice else "human_direct")


@pytest.mark.parametrize("event", ["pre_llm_call", "pre_api_request"])
def test_real_hook_and_public_observer_use_the_same_input_pipeline(adapter, monkeypatch, event):
    """The dispatcher and public observer share one internal input implementation with separate payload data."""
    provider, _clock = adapter
    method = "observe_pre_llm" if event == "pre_llm_call" else "observe_pre_api_request"
    observed = []
    original = provider._observe_input

    def watched(kind, payload, **policy):
        observed.append((kind, payload["turn_id"]))
        return original(kind, payload, **policy)

    monkeypatch.setattr(provider, "_observe_input", watched)
    _hook(provider, event, turn_id="TEST-shared", user_message="TEST shared pipeline")
    getattr(provider, method)(turn_id="TEST-public", user_message="TEST public method")
    assert observed == [(event, "TEST-shared"), (event, "TEST-public")]
