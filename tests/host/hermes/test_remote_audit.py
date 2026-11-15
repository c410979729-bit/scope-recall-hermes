"""Regression witnesses for remote transport privacy, bounded snapshots and host routing."""

from __future__ import annotations

import copy
import json
import threading
import time
from contextvars import copy_context

from scope_recall.adapters.clients import remote_client
from scope_recall.adapters.hermes import hooks
from scope_recall.adapters.hermes.boundary import _shown_text
from scope_recall.adapters.hermes.provider import ScopeRecallHermesAdapter
from scope_recall.adapters.hermes.remote_wire import visible
from scope_recall.core.file_lock import advisory_file_lock

from .test_remote_entry import AGENT, WORDS, WORKSPACE, entry, say, sources


def test_prefetch_filters_private_query_before_transport(entry, monkeypatch):
    _root, _home, _client, _config, _service, provider = entry
    sent = []
    monkeypatch.setattr(remote_client, "hermes_hook", lambda _config, payload, **kwargs: sent.append(payload) or {})
    provider().prefetch("TEST query <private>TEST QUERY SECRET</private>")
    assert "TEST QUERY SECRET" not in json.dumps(sent)


def test_mcp_metadata_has_no_conversation_body(entry, monkeypatch):
    _root, _home, _client, _config, _service, provider = entry
    remote = provider()
    remote.remote_hook("pre_llm_call", {"turn_id": "TEST-meta", "user_message": "TEST BODY SECRET"})
    sent = []
    monkeypatch.setattr(remote_client, "hermes_tool", lambda _config, name, args, meta: sent.append(meta) or "{}")
    remote.handle_tool_call("status", {})
    assert "TEST BODY SECRET" not in json.dumps(sent)
    assert "conversation_history" not in json.dumps(sent)


def test_all_content_exits_stop_when_privacy_check_cannot_verify(entry, monkeypatch):
    _root, _home, _client, config, _service, provider = entry
    remote_client._remember_private_rules(config, {"private_rules": True})
    sent = []

    def post(_config, body, timeout, *, endpoint="/hook"):
        sent.append((endpoint, body))
        return None

    monkeypatch.setattr(remote_client, "_post", post)
    remote = provider()
    remote.prefetch("TEST BLOCKED QUERY")
    remote.handle_tool_call("status", {})
    assert all(endpoint == "/private/check" for endpoint, _body in sent)
    assert "TEST BLOCKED QUERY" not in json.dumps(sent)


def test_history_is_checked_at_its_original_time_after_resume(entry, monkeypatch):
    _root, _home, _client, _config, service, provider = entry
    service.start()
    remote = provider()
    say(remote, "scope-recall pause", turn="TEST-pause")
    hidden_at = time.time()
    say(remote, "scope-recall resume", turn="TEST-resume")
    sent = []
    original = remote_client._post

    def post(config, body, timeout, **kwargs):
        sent.append((kwargs.get("endpoint", "/hook"), copy.deepcopy(body)))
        return original(config, body, timeout, **kwargs)

    monkeypatch.setattr(remote_client, "_post", post)
    history = [
        {"role": "user", "content": "TEST PAST PRIVATE SECRET", "timestamp": hidden_at},
        {"role": "assistant", "content": "TEST past reply", "timestamp": hidden_at},
        {"role": "user", "content": WORDS, "timestamp": time.time()},
    ]
    say(remote, messages=history, turn="TEST-after-resume")
    assert "TEST PAST PRIVATE SECRET" not in json.dumps(sent)


def test_private_span_cannot_turn_input_into_resume_command(entry):
    root, _home, _client, _config, service, provider = entry
    service.start()
    remote = provider()
    say(remote, "scope-recall pause", turn="TEST-pause")
    say(remote, "scope-recall resume<private>TEST suffix</private>", turn="TEST-not-command")
    say(remote, "TEST remains private", turn="TEST-hidden")
    assert "TEST remains private" not in json.dumps(sources(root))


def test_ungranted_chat_does_not_block_authorized_queue(entry):
    root, _home, _client, config, service, provider = entry
    service.start()
    bad = provider("TEST-ungranted", platform="telegram", chat_type="group", chat_id="TEST-no-grant")
    say(bad, "TEST ungranted text", session="TEST-ungranted")
    good = provider("TEST-allowed")
    say(good, session="TEST-allowed")
    remote_client.flush_spool(config, seconds=20)
    assert any(row["content"] == WORDS for row in sources(root))


def test_inline_start_isolates_delayed_sync_when_pre_hook_is_missing(entry):
    root, _home, _client, _config, service, provider = entry
    service.start()
    remote = provider()
    remote.remote_hook("pre_llm_call", {"session_id": "TEST-session", "turn_id": "TEST-a", "user_message": "TEST A"})
    remote.on_turn_start(1, "TEST A")
    a = copy_context()
    remote.on_turn_start(2, "TEST B")
    remote.remote_hook("pre_api_request", {"session_id": "TEST-session", "turn_id": "TEST-b", "user_message": "TEST B"})
    b = copy_context()
    a.run(remote.sync_turn, "TEST A", "TEST A answer", session_id="TEST-session")
    b.run(remote.sync_turn, "TEST B", "TEST B answer", session_id="TEST-session")
    answers = {row["content"]: row["source_event_key"] for row in sources(root) if row["role"] == "assistant"}
    assert "TEST-a" in answers["TEST A answer"]
    assert "TEST-b" in answers["TEST B answer"]


def test_profile_context_selects_local_even_with_remote_candidate(entry, monkeypatch):
    _root, home, _client, _config, _service, provider = entry
    remote = provider()
    local = ScopeRecallHermesAdapter()
    local.initialize(
        "TEST-session",
        hermes_home=str(home),
        platform="cli",
        agent_identity=AGENT,
        agent_workspace=WORKSPACE,
        user_id="TEST-owner",
    )
    calls = []
    monkeypatch.setattr(local, "_observe_input", lambda *args, **kwargs: calls.append("local"))
    monkeypatch.setattr(remote, "remote_hook", lambda *args: calls.append("remote"))
    try:
        hooks._dispatch("pre_llm_call", {"session_id": "TEST-session", "hermes_home": str(home)}, wait=1)
        assert calls == ["local"]
    finally:
        local.shutdown()


def test_wire_preserves_only_visible_codex_commentary():
    row = {
        "role": "assistant",
        "content": "",
        "codex_message_items": [
            {
                "type": "message",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "TEST visible commentary"}],
            },
            {"type": "reasoning", "content": [{"type": "output_text", "text": "TEST hidden reasoning"}]},
        ],
    }
    assert _shown_text(visible(row)) == _shown_text(row) == "TEST visible commentary"
    assert "TEST hidden reasoning" not in json.dumps(visible(row))


def test_snapshot_size_is_independent_of_old_history_and_api_attempts(entry, monkeypatch):
    _root, _home, _client, _config, _service, provider = entry
    sent = []
    monkeypatch.setattr(
        remote_client, "hermes_hook", lambda _config, payload, **kwargs: sent.append(copy.deepcopy(payload)) or {}
    )
    remote = provider()
    history = [{"role": "user", "content": "TEST old " + str(i) + "x" * 16384} for i in range(16)]
    history += [{"role": "user", "content": WORDS, "timestamp": time.time()}]
    remote.remote_hook(
        "pre_llm_call", {"turn_id": "TEST-bounded", "user_message": WORDS, "conversation_history": history}
    )
    remote.on_turn_start(1, WORDS)
    for _ in range(30):
        remote.remote_hook(
            "pre_api_request", {"turn_id": "TEST-bounded", "user_message": WORDS, "conversation_history": history}
        )
    assert max(len(json.dumps(payload).encode()) for payload in sent) < 96 * 1024
    assert len(sent[-1]["events"]) < 10


def test_offline_snapshots_of_one_turn_coalesce(entry):
    _root, _home, _client, config, _service, provider = entry
    say(provider())
    assert len(list((config["state_dir"] / "spool").glob("*.json"))) == 1


def test_claimed_hermes_head_is_not_overtaken(entry, monkeypatch):
    _root, _home, _client, config, _service, provider = entry
    say(provider("TEST-one"), session="TEST-one")
    say(provider("TEST-two"), session="TEST-two")
    head = sorted((config["state_dir"] / "spool").glob("*.json"))[0]
    sent = []
    monkeypatch.setattr(remote_client, "_post", lambda *args, **kwargs: sent.append(args) or {"retry": False})
    ready, release = threading.Event(), threading.Event()

    def hold():
        with advisory_file_lock(head.with_suffix(".claim"), timeout_seconds=0):
            ready.set()
            release.wait(10)

    worker = threading.Thread(target=hold)
    worker.start()
    try:
        assert ready.wait(5)
        assert remote_client.flush_spool(config, seconds=2) == 0
        assert not sent
    finally:
        release.set()
        worker.join(5)
        assert not worker.is_alive()


def test_byte_capacity_and_durable_loss_are_reported_by_status(tmp_path, monkeypatch, capsys):
    from scope_recall.adapters.clients import hermes_queue

    config = {"host": "hermes", "state_dir": tmp_path / "state"}
    monkeypatch.setattr(hermes_queue, "MAX_BYTES", 650)
    for number in range(5):
        hermes_queue.put(config, {"queue_key": f"{number:064x}", "revision": 1, "body": "x" * 190}, f"TEST-{number}")
    state = hermes_queue.status(config)
    assert state["queued_bytes"] <= 650
    assert state["dropped_count"] >= 3
    assert state["first_dropped_at"] == "TEST-1"
    monkeypatch.setattr(remote_client, "load_client_config", lambda path: config)
    assert remote_client.main(["status", "--config", str(tmp_path / "client.json")]) == 0
    assert json.loads(capsys.readouterr().out) == state


def test_refused_route_does_not_erase_known_private_history(entry):
    _root, _home, _client, config, service, provider = entry
    service.start()
    remote_client._remember_private_rules(config, {"private_rules": True})
    bad = provider("TEST-ungranted", platform="telegram", chat_type="group", chat_id="TEST-no-grant")
    say(bad, "TEST rejected", session="TEST-ungranted")
    assert remote_client._private_rules(config)
    assert not list((config["state_dir"] / "spool").glob("*.json"))


def test_original_times_filter_private_rows_inside_current_turn(entry, monkeypatch):
    _root, _home, _client, _config, service, provider = entry
    service.start()
    remote = provider()
    say(remote, "scope-recall pause", turn="TEST-pause")
    hidden_at = time.time()
    say(remote, "scope-recall resume", turn="TEST-resume")
    sent = []
    original = remote_client._post

    def post(config, body, timeout, **kwargs):
        sent.append(copy.deepcopy(body))
        return original(config, body, timeout, **kwargs)

    monkeypatch.setattr(remote_client, "_post", post)
    say(
        remote,
        messages=[
            {"role": "user", "content": WORDS, "timestamp": time.time()},
            {
                "role": "assistant",
                "content": "TEST ORIGINAL TIME SECRET",
                "timestamp": hidden_at,
                "metadata": {"text": "TEST NESTED SECRET"},
            },
        ],
        turn="TEST-resumed",
    )
    assert "TEST ORIGINAL TIME SECRET" not in json.dumps(sent)
    assert "TEST NESTED SECRET" not in json.dumps(sent)


def test_visible_codex_commentary_is_a_remote_assistant_source(entry):
    root, _home, _client, _config, service, provider = entry
    service.start()
    remote = provider()
    remote.remote_hook("pre_llm_call", {"turn_id": "TEST-commentary", "user_message": WORDS})
    remote.on_turn_start(1, WORDS)
    row = {
        "role": "assistant",
        "content": "",
        "timestamp": time.time(),
        "codex_message_items": [
            {
                "type": "message",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "TEST commentary answer"}],
            },
            {"type": "reasoning", "content": [{"type": "output_text", "text": "TEST reasoning secret"}]},
        ],
    }
    history = [{"role": "user", "content": WORDS}, row, {"role": "assistant", "content": "TEST final answer"}]
    remote.remote_hook(
        "post_llm_call",
        {"turn_id": "TEST-commentary", "assistant_response": "TEST final answer", "conversation_history": history},
    )
    remote.sync_turn(WORDS, "TEST final answer", session_id="TEST-session", messages=history)
    rows = sources(root)
    assert "TEST commentary answer" in [row["content"] for row in rows]
    assert "TEST reasoning secret" not in json.dumps(rows)


def test_gzip_large_current_turn_stays_deliverable_and_413_retains_it(entry, monkeypatch):
    import gzip

    from scope_recall.adapters.hermes.remote_wire import MAX_WIRE_BYTES

    _root, _home, _client, config, _service, provider = entry
    remote = provider()
    monkeypatch.setattr(remote_client, "hermes_hook", lambda *args, **kwargs: {})
    remote.on_turn_start(1, WORDS)
    remote.sync_turn(WORDS, "x" * (9 * 1024 * 1024), session_id="TEST-session")
    body = {"payload": remote._payload()}
    raw, headers = remote_client._encoded_request(config, body, "/hook")
    assert len(raw) < MAX_WIRE_BYTES and headers["Content-Encoding"] == "gzip"
    assert json.loads(gzip.decompress(raw)) == body

    assert remote_client._refused_response(config, "Hermes", 413) == {"refused": 413}
    assert remote_client._refused_response({**config, "host": "codex"}, "Stop", 413) == {"refused": 413}


def test_denied_history_rows_and_unverifiable_decisions_never_keep_bodies():
    from scope_recall.adapters.clients.hermes_privacy import eligible, metadata

    payload = {
        "binding": {"session_id": "TEST-session"},
        "events": [],
        "history": [
            {"role": "assistant", "timestamp": 1, "metadata": {"content": "TEST hidden"}},
            "TEST other body",
        ],
    }
    assert "TEST hidden" not in json.dumps(metadata(payload))
    answer = {"decisions": [], "history_decisions": {"history": [True, True]}}
    assert eligible(payload, answer)["history"] == [{"role": "assistant", "timestamp": 1}, {}]
    assert eligible(payload, {**answer, "history_decisions": {"history": [None, False]}}) is None


def test_acknowledging_old_snapshot_keeps_concurrent_new_revision(tmp_path):
    from scope_recall.adapters.clients import hermes_queue

    config = {"state_dir": tmp_path / "state"}
    payload = {"queue_key": "a" * 64, "revision": 1}
    path = hermes_queue.put(config, payload, "TEST-first")
    assert hermes_queue.put(config, {**payload, "revision": 2}, "TEST-second") == path
    hermes_queue.acknowledge(config, path, payload)
    assert json.loads(path.read_text(encoding="utf-8"))["payload"]["revision"] == 2
