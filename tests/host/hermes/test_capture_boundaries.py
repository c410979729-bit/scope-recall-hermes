"""Synthetic host deliveries distinguish omitted, delayed and retained human sources."""

from __future__ import annotations

import json
import sqlite3
from contextvars import copy_context
from dataclasses import replace

import pytest
from scope_recall.adapters.hermes import HermesIdentityError, ScopeRecallHermesAdapter, kept_captures
from scope_recall.adapters.hermes.capture import RETRY_EVENT_BYTES
from scope_recall.contracts import ContractError
from scope_recall.core import capture_inbox, injection_guard
from scope_recall.core.injection_policy import ScanResult

from tests.host.hermes.test_hermes_private_mode import _owner
from tests.host.hermes.test_steers import _row, _steer, telegram  # noqa: F401


def _hook(*args, **kwargs):
    """Use the registered helper-thread dispatcher without a module import cycle."""
    from tests.host.hermes.test_turn_binding_recovery import _hook as dispatch

    dispatch(*args, **kwargs)


def _rows(provider, table="source_events"):
    """Read only the synthetic store, including sources withheld from public retrieval."""
    assert table in {"source_events", "capture_inbox", "source_safety"}
    with sqlite3.connect(provider._core.storage.path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(f"SELECT * FROM {table}")]


def _human_words(provider):
    return [row["content"] for row in _rows(provider) if row["origin"] == "human_direct"]


def _pre(provider, text, turn="TEST-turn", **kwargs):
    _hook(provider, "pre_llm_call", turn_id=turn, user_message=text, **kwargs)


def test_recording_controls_release_observation_slots(telegram):
    """Excluded control text must not consume the finite ledger needed by later messages."""
    for number in range(64):
        _pre(telegram, "scope-recall resume", f"TEST-control-{number}")
    _pre(telegram, "TEST ordinary message after controls", "TEST-ordinary")
    assert _human_words(telegram) == ["TEST ordinary message after controls"]
    assert telegram._ledger.pending_identities() == ()


@pytest.mark.parametrize("finish", ["cancelled", "api_error", "timeout"])
@pytest.mark.parametrize("pre_called", [False, True])
def test_skipped_pre_capture_is_recovered_at_turn_start(adapter, monkeypatch, finish, pre_called):
    """A delivered but busy pre-hook leaves enough identity for the inline start to recover its input."""
    provider, _clock = adapter
    text = "TEST input whose turn has no reply"
    if pre_called:
        _pre(provider, text)
    else:
        from tests.host.hermes.test_turn_binding_recovery import _busy_hook

        _busy_hook(provider, monkeypatch, turn_id="TEST-turn", user_message=text)
    provider.on_turn_start(1, text)
    if finish != "cancelled":
        provider.observe_api_request_error(turn_id="TEST-turn", status=finish)
    provider.on_session_end([dict(role="user", content=text, timestamp=1788696000)])
    assert _human_words(provider) == [text]


def test_sync_recovers_opener_when_pre_hook_is_missing(adapter):
    provider, _clock = adapter
    provider.on_turn_start(1, "TEST skipped opener")
    provider.sync_turn("TEST skipped opener", "TEST reply")
    assert _human_words(provider) == ["TEST skipped opener"]


@pytest.mark.parametrize("pre_called", [False, True])
def test_delayed_sync_keeps_each_turns_opener_and_reply(adapter, pre_called):
    """The memory worker runs a queued context after a later turn has already started."""
    provider, _clock = adapter
    if pre_called:
        _pre(provider, "TEST older input without pre capture", "TEST-older")
    provider.on_turn_start(1, "TEST older input without pre capture")
    older_id = provider._active_turn_id
    queued = copy_context()
    _pre(provider, "TEST newer input", "TEST-newer")
    provider.on_turn_start(2, "TEST newer input")
    queued.run(provider.sync_turn, "TEST older input without pre capture", "TEST older reply")
    provider.sync_turn("TEST newer input", "TEST newer reply")
    queued.run(provider.sync_turn, "TEST older input without pre capture", "TEST older reply")
    rows = _rows(provider)
    assert len(rows) == 4
    by_content = {row["content"]: row["source_event_key"] for row in rows}
    assert by_content["TEST older input without pre capture"].endswith(f":user:{older_id}@1")
    assert by_content["TEST older reply"].endswith(f":sync_assistant:{older_id}@1")
    assert by_content["TEST newer input"].endswith(":user:TEST-newer@1")
    assert by_content["TEST newer reply"].endswith(":sync_assistant:TEST-newer@1")


def test_pre_api_request_captures_steers_before_the_turn_finishes(telegram):
    """The per-request hook sees the steer after insertion and before a model can choose tools."""
    _pre(telegram, "TEST opener")
    telegram.on_turn_start(1, "TEST opener")
    history = [dict(role="user", content="TEST opener"), _row(_steer("TEST correction", message_id="TEST-early"))]
    for retry in range(2):
        _hook(
            telegram,
            "pre_api_request",
            turn_id="TEST-turn",
            user_message="TEST opener",
            conversation_history=history,
            retry_count=retry,
        )
    assert _human_words(telegram) == ["TEST opener", "TEST correction"]
    assert len({row["source_event_key"] for row in _rows(telegram)}) == 2
    telegram.sync_turn("TEST opener", "TEST reply", messages=history)
    assert _human_words(telegram) == ["TEST opener", "TEST correction"]


def test_next_pre_hook_does_not_restore_untyped_interrupted_input(telegram):
    """Request hooks cannot link untyped interrupted history to its original source or deletion state."""
    telegram.on_turn_start(1, "TEST missed input")
    history = [
        dict(role="user", content="TEST missed input", timestamp=1788696000),
        dict(role="assistant", content="TEST interruption placeholder", display_kind="hidden"),
        dict(role="user", content="TEST next input", timestamp=1788696060),
    ]
    _pre(telegram, "TEST next input", "TEST-next", conversation_history=history)
    assert _human_words(telegram) == ["TEST next input"]
    telegram.on_session_end(history)
    assert _human_words(telegram) == ["TEST next input"]


def test_delayed_sync_leaves_the_next_turns_interim_intact(adapter):
    """Each queued sync drains only its own post-model snapshot."""
    provider, _clock = adapter
    queued = []
    for number in (1, 2):
        turn = f"TEST-turn-{number}"
        opener, answer, interim = f"TEST opener {number}", f"TEST answer {number}", f"TEST interim {number}"
        _pre(provider, opener, turn)
        provider.on_turn_start(number, opener)
        _hook(
            provider,
            "post_llm_call",
            turn_id=turn,
            assistant_response=answer,
            conversation_history=[
                dict(role="user", content=opener),
                dict(role="assistant", content=interim, tool_calls=[dict(id=f"TEST-tool-{number}")]),
                dict(role="tool", content="TEST tool result"),
                dict(role="assistant", content=answer),
            ],
        )
        queued.append((copy_context(), opener, answer))
    context, opener, answer = queued[0]
    context.run(provider.sync_turn, opener, answer)
    assert "TEST-turn-2" in provider._interim_said
    context, opener, answer = queued[1]
    context.run(provider.sync_turn, opener, answer)
    assert len(_rows(provider)) == 6
    for row in _rows(provider):
        number = row["content"][-1]
        assert f"TEST-turn-{number}" in row["source_event_key"]


def test_first_api_request_promotes_a_missing_pre_hooks_ordinal(adapter):
    """The first request supplies the UUID even when the pre-hook was suppressed entirely."""
    provider, _clock = adapter
    provider.on_turn_start(11, "TEST resumed input")
    queued = copy_context()
    # Hermes invokes plugin callbacks in copied contexts on helper threads.
    _hook(
        provider,
        "pre_api_request",
        turn_id="TEST-resumed-uuid",
        user_message="TEST resumed input",
        conversation_history=[dict(role="user", content="TEST resumed input")],
    )
    _pre(provider, "TEST later input", "TEST-later")
    provider.on_turn_start(12, "TEST later input")
    queued.run(provider.sync_turn, "TEST resumed input", "TEST resumed answer")
    rows = [row for row in _rows(provider) if row["content"].startswith("TEST resumed")]
    assert len(rows) == 2
    assert all("TEST-resumed-uuid" in row["source_event_key"] for row in rows)


def test_a_late_pre_api_hook_cannot_promote_the_next_turns_binding(adapter):
    """An abandoned hook retains its own copied context when the next turn has already begun."""
    provider, _clock = adapter
    _pre(provider, "TEST first input", "TEST-first")
    provider.on_turn_start(1, "TEST first input")
    late = copy_context()
    provider.on_turn_start(2, "TEST second input")
    _hook(provider, "pre_api_request", late, turn_id="TEST-first", user_message="TEST first input")
    _hook(
        provider,
        "pre_api_request",
        turn_id="TEST-second",
        user_message="TEST second input",
        conversation_history=[dict(role="user", content="TEST second input")],
    )
    provider.sync_turn("TEST second input", "TEST second answer")
    rows = [row for row in _rows(provider) if row["content"].startswith("TEST second")]
    assert len(rows) == 2
    assert all("TEST-second" in row["source_event_key"] for row in rows)


@pytest.mark.parametrize("hook", ["start", "api"])
def test_opener_recovery_respects_private_mode_and_controls(adapter, monkeypatch, hook):
    """Fallback capture uses the same privacy and recording-command boundary as the pre-hook."""
    provider, _clock = adapter
    owner, ctx = _owner(provider)
    owner.private_on(target="session", session=ctx.session_id)
    from tests.host.hermes.test_turn_binding_recovery import _busy_hook

    _busy_hook(provider, monkeypatch, turn_id="TEST-private", user_message="TEST private recovered input")
    provider.on_turn_start(1, "TEST private recovered input")
    if hook == "api":
        _hook(
            provider,
            "pre_api_request",
            turn_id="TEST-private",
            user_message="TEST private recovered input",
            conversation_history=[],
        )
    assert _rows(provider) == _rows(provider, "capture_inbox") == []
    assert not provider._retry.captures
    owner.private_off(target="session", session=ctx.session_id)
    _busy_hook(provider, monkeypatch, turn_id="TEST-control", user_message="scope-recall resume")
    provider.on_turn_start(2, "scope-recall resume")
    assert _rows(provider) == []
    assert provider._ledger.pending_identities() == ()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  ",
        [{"type": "image_url", "image_url": {"url": "https://example.invalid/TEST.png"}}],
        "[screenshot]",
        "data:image/png;base64,QUJDRA==",
        "<private>TEST private text</private>",
        "ghp_" + "TESTONLY" * 4,
    ],
    ids=["empty", "whitespace", "image-part", "image-placeholder", "data-url", "all-private", "all-secret"],
)
def test_input_without_retained_text_or_authorized_artifact_has_no_source(adapter, text):
    provider, _clock = adapter
    _pre(provider, text)
    assert _rows(provider) == _rows(provider, "capture_inbox") == []


def test_unauthorized_attachment_only_leaves_a_diagnostic(adapter):
    provider, _clock = adapter
    _pre(provider, "", attachments=[dict(filename="TEST.png", mediaType="image/png")])
    assert _rows(provider) == []
    assert any("attachment_gap" in gap for gap in provider.diagnostics.pending_outcome_gaps)


def test_authorized_artifact_only_has_a_human_source(adapter):
    provider, _clock = adapter
    _pre(provider, "", attachments=[dict(filename="TEST.svg", svg="<svg xmlns='http://www.w3.org/2000/svg'></svg>")])
    assert _human_words(provider) == [""]
    assert json.loads(_rows(provider)[0]["extra_json"])["artifact_refs"]


@pytest.mark.parametrize(
    "text",
    [
        "/TEST-command argument",
        "> TEST quoted line\nTEST correction",
        "[Tue 2026-11-10 12:00:00 UTC] TEST wrapped text",
    ],
)
def test_delivered_commands_and_wrappers_are_retained(adapter, text):
    provider, _clock = adapter
    _pre(provider, text)
    assert _human_words(provider) == [text]


def test_equal_text_in_distinct_turns_is_two_sources(adapter):
    provider, _clock = adapter
    _pre(provider, "TEST same words", "TEST-first")
    _pre(provider, "TEST same words", "TEST-second")
    assert _human_words(provider) == ["TEST same words", "TEST same words"]


def test_image_only_completed_turn_can_store_the_hosts_image_count(adapter):
    provider, _clock = adapter
    _pre(provider, [{"type": "image_url", "image_url": {"url": "https://example.invalid/TEST.png"}}])
    provider.on_turn_start(1, "")
    provider.sync_turn("[1 image]", "TEST image response")
    assert _human_words(provider) == ["[1 image]"]


def test_host_notice_has_a_source_without_human_attribution(adapter):
    provider, _clock = adapter
    text = "TEST scheduled notification"
    _pre(provider, text, conversation_history=[dict(role="user", content=text, display_kind="process_complete")])
    assert _human_words(provider) == []
    assert [row["origin"] for row in _rows(provider)] == ["host_generated"]


@pytest.mark.parametrize("context", ["cron", "subagent", "flush"])
def test_non_primary_session_has_no_capture(adapter, initialize_kwargs, context):
    provider, _clock = adapter
    provider.initialize("TEST-non-primary", **dict(initialize_kwargs, agent_context=context))
    _pre(provider, "TEST scheduled or delegated input")
    provider.sync_turn("TEST scheduled or delegated input", "TEST response")
    assert provider._identity.read_only
    assert _rows(provider) == []


def test_explicitly_unbound_session_has_no_capture(adapter):
    provider, _clock = adapter
    provider._unbound_session = True
    _pre(provider, "TEST unbound input")
    provider.on_turn_start(1, "TEST unbound input")
    provider.sync_turn("TEST unbound input", "TEST response")
    assert _rows(provider) == []


def test_userless_cron_session_is_explicitly_unbound(adapter, initialize_kwargs):
    provider, _clock = adapter
    provider.initialize("TEST-userless-cron", **dict(initialize_kwargs, platform="cron", user_id=""))
    assert provider._unbound_session and not provider._initialized
    provider.observe_pre_llm(session_id="TEST-userless-cron", turn_id="TEST-cron", user_message="TEST scheduled input")
    assert _rows(provider) == []


def test_userless_gateway_cannot_bind(adapter, initialize_kwargs):
    provider, _clock = adapter
    with pytest.raises(HermesIdentityError, match="user principal required"):
        provider.initialize("TEST-userless-gateway", **dict(initialize_kwargs, platform="telegram", user_id=""))
    assert _rows(provider) == []


def test_audience_without_capture_scope_has_no_source(adapter):
    provider, _clock = adapter
    bound = provider._identity
    provider._identity = replace(bound, runtime_audience=replace(bound.runtime_audience, allowed_scope_ids=frozenset()))
    _pre(provider, "TEST unmapped audience")
    assert _rows(provider) == []


@pytest.mark.parametrize("mode", ["full", "read-only"])
def test_private_session_retains_neither_source_nor_ingress(adapter, mode):
    provider, _clock = adapter
    owner, ctx = _owner(provider)
    owner.private_on(target="session", session=ctx.session_id, mode=mode)
    _pre(provider, "TEST private interval input")
    assert _rows(provider) == _rows(provider, "capture_inbox") == []
    assert not provider._retry.captures


def test_paused_entry_has_no_capture(adapter):
    provider, _clock = adapter
    owner, _ctx = _owner(provider)
    owner.set_entry_state("local", "paused", actor="TEST-owner")
    _pre(provider, "TEST paused entry input")
    assert _rows(provider) == _rows(provider, "capture_inbox") == []


@pytest.mark.parametrize("state", ["pending_review", "quarantined"])
def test_safety_hold_keeps_the_human_source_row(adapter, monkeypatch, state):
    provider, _clock = adapter
    monkeypatch.setattr(injection_guard, "_scan_events", lambda *args, **kwargs: ScanResult(state, ("TEST-rule",)))
    _pre(provider, "TEST held source text")
    assert _human_words(provider) == ["TEST held source text"]
    assert [row["state"] for row in _rows(provider, "source_safety")] == [state]


def test_durable_inbox_waits_for_a_replay_hook_without_a_configured_worker(adapter, monkeypatch):
    provider, _clock = adapter
    record = capture_inbox.record_event

    def busy(*args, **kwargs):
        raise ContractError("STORAGE_UNAVAILABLE")

    monkeypatch.setattr(capture_inbox, "record_event", busy)
    _pre(provider, "TEST durable pending input")
    assert _rows(provider) == []
    assert len(_rows(provider, "capture_inbox")) == 1
    assert not provider._retry.captures
    monkeypatch.setattr(capture_inbox, "record_event", record)
    provider.on_turn_start(1, "TEST durable pending input")
    provider.sync_turn("TEST durable pending input", "TEST response")
    assert _human_words(provider) == []
    assert len(_rows(provider, "capture_inbox")) == 1
    provider.on_pre_compress([])
    assert _human_words(provider) == ["TEST durable pending input"]
    assert _rows(provider, "capture_inbox") == []


def test_terminal_inbox_error_is_inspectable_without_automatic_replay(adapter, monkeypatch):
    provider, _clock = adapter
    record = capture_inbox.record_event

    def invalid(*args, **kwargs):
        raise ContractError("INPUT_INVALID", "TEST-terminal")

    monkeypatch.setattr(capture_inbox, "record_event", invalid)
    _pre(provider, "TEST input retained for inspection")
    monkeypatch.setattr(capture_inbox, "record_event", record)
    provider.on_pre_compress([])
    assert _human_words(provider) == []
    assert [row["last_error_code"] for row in _rows(provider, "capture_inbox")] == ["INPUT_INVALID"]


@pytest.mark.parametrize("oversize", [False, True])
def test_failure_before_ingress_is_not_restart_visible(adapter, initialize_kwargs, monkeypatch, oversize):
    """A pre-ingress failure has only bounded process memory until an explicit checkpoint."""
    provider, clock = adapter
    text = "TEST before ingress " + ("x" * RETRY_EVENT_BYTES if oversize else "small input")
    lookup = provider._core._source_replay_metadata

    def busy(*args, **kwargs):
        raise ContractError("DEADLINE_EXCEEDED")

    monkeypatch.setattr(provider._retry, "start", lambda: None)
    monkeypatch.setattr(provider._core, "_source_replay_metadata", busy)
    _pre(provider, text)
    assert _rows(provider) == _rows(provider, "capture_inbox") == []
    assert bool(provider._retry.captures) is not oversize
    assert not list(kept_captures.folder(provider._identity.binding).glob("*.json"))
    monkeypatch.setattr(provider._core, "_source_replay_metadata", lookup)
    replacement = ScopeRecallHermesAdapter(core=provider._core, clock=clock)
    replacement.initialize("TEST-session-1", **initialize_kwargs)
    try:
        assert replacement._retry.captures == {}
        assert _rows(replacement) == []
    finally:
        replacement.shutdown()


def test_unknown_privacy_state_does_not_keep_a_failed_capture(adapter, monkeypatch):
    provider, _clock = adapter

    def busy(*args, **kwargs):
        raise ContractError("STORAGE_UNAVAILABLE")

    monkeypatch.setattr(provider._core, "record_host_event", busy)
    monkeypatch.setattr(provider._writer, "private_state", lambda *args, **kwargs: "unknown")
    _pre(provider, "TEST input with unreadable privacy policy")
    assert _rows(provider) == _rows(provider, "capture_inbox") == []
    assert not provider._retry.captures
    assert "capture_gap:private_mode_unverified" in provider.diagnostics.pending_outcome_gaps


@pytest.mark.parametrize("origin", [False, True])
def test_gateway_steer_needs_the_sessions_sender(telegram, origin):
    text = _steer("TEST unverified steer", origin=origin, user_id="TEST-someone-else")
    telegram.on_session_end([_row(text)])
    assert _rows(telegram) == []


def test_unconsumed_steer_is_invisible_until_the_host_delivers_history(telegram):
    row = _row(_steer("TEST queued steer", message_id="TEST-1"))
    assert _rows(telegram) == []
    telegram.on_session_end([row])
    assert _human_words(telegram) == ["TEST queued steer"]


def test_active_redirect_without_a_steer_kind_has_no_separate_source(telegram):
    """The host's redirect row is untyped and its combined sync input reuses the captured opener."""
    opener, correction, answer = "TEST opening input", "TEST active correction", "TEST final response"
    _pre(telegram, opener)
    telegram.on_turn_start(1, opener)
    origin_text = _steer(correction, message_id="TEST-redirect").split("\n", 1)[1].rsplit("\n", 1)[0]
    history = [
        dict(role="user", content=opener),
        dict(role="assistant", content="", display_kind="hidden"),
        dict(role="user", content=origin_text, timestamp=1788696000),
        dict(role="assistant", content=answer),
    ]
    _hook(
        telegram,
        "post_llm_call",
        turn_id="TEST-turn",
        assistant_response=answer,
        conversation_history=history,
    )
    telegram.sync_turn(f"{opener}\n\nUser correction during the turn: {origin_text}", answer, messages=history)
    telegram.on_session_end(history)
    assert _human_words(telegram) == [opener]


@pytest.mark.parametrize("identified", [False, True])
def test_equal_steers_need_distinct_message_ids_or_times(telegram, identified):
    rows = [
        _row(_steer("TEST repeated steer", message_id=f"TEST-{number}" if identified else None)) for number in range(2)
    ]
    telegram.on_session_end(rows)
    assert _human_words(telegram) == ["TEST repeated steer"] * (2 if identified else 1)


@pytest.mark.parametrize("group", [False, True])
def test_recovered_unanswered_message_needs_a_private_chat(telegram, group):
    if group:
        bound = telegram._identity
        telegram._identity = replace(bound, scope=replace(bound.scope, chat_type="group"))
    rows = [
        dict(role="user", content="TEST recovered input", timestamp=1788696000),
        dict(role="user", content="TEST current input", timestamp=1788696060),
    ]
    telegram.on_session_end(rows)
    assert _human_words(telegram) == ([] if group else ["TEST recovered input"])


def test_untimed_recovered_equal_messages_share_a_fallback_identity(telegram):
    rows = [
        dict(role="user", content="TEST repeated recovered input"),
        dict(role="user", content="TEST repeated recovered input"),
        dict(role="user", content="TEST current input"),
    ]
    telegram.on_session_end(rows)
    assert _human_words(telegram) == ["TEST repeated recovered input"]


@pytest.mark.parametrize(
    "text", ["[TEST NOTICE] human text", "Gateway message origin TEST human text", "TEST Hermes is shutting down quote"]
)
def test_recovered_notice_shaped_text_is_not_promoted(telegram, text):
    telegram.on_session_end([dict(role="user", content=text), dict(role="user", content="TEST current input")])
    assert _rows(telegram) == []
