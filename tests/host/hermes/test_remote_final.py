"""Regressions for delayed capture, checkpoints, bounded metadata and loss accounting."""

from __future__ import annotations

import copy
import gzip
import http.client
import json
import logging
import random
import sys
import time
from contextvars import copy_context
from types import ModuleType

import pytest
from scope_recall.adapters.clients import hermes_queue, remote_client, remote_server
from scope_recall.adapters.hermes import hooks

from .test_remote_entry import TOKEN, WORDS, entry, say, sources
from .test_steers import _row, _steer


def test_delayed_private_sync_never_retimes_any_copy(entry, monkeypatch):
    root, _home, _client, config, service, provider = entry
    service.start()
    remote = provider()
    sent, original = [], remote_client._post
    written, original_write = [], hermes_queue._write

    def write(path, value):
        written.append(copy.deepcopy(value))
        return original_write(path, value)

    monkeypatch.setattr(hermes_queue, "_write", write)

    def post(config, body, timeout, **kwargs):
        sent.append(copy.deepcopy(body))
        return original(config, body, timeout, **kwargs)

    monkeypatch.setattr(remote_client, "_post", post)
    say(remote, "scope-recall pause", turn="TEST-pause")
    secret, answer = "TEST PAUSED OPENER", "TEST PAUSED ANSWER"
    remote.remote_hook("pre_llm_call", {"turn_id": "TEST-private", "user_message": secret})
    remote.on_turn_start(2, secret)
    remote.remote_hook("post_llm_call", {"turn_id": "TEST-private", "assistant_response": answer})
    old = copy_context()
    say(remote, "scope-recall resume", turn="TEST-resume")
    old.run(remote.sync_turn, secret, answer, session_id="TEST-session")
    history = [{"role": "user", "content": secret}, {"role": "assistant", "content": answer}]
    old.run(remote.on_pre_compress, history)
    old.run(remote.on_session_end, history)
    kept = [p.read_text(encoding="utf-8") for p in (config["state_dir"] / "spool").glob("*.json")]
    for body in (json.dumps(sent), json.dumps(kept), json.dumps(written), json.dumps(sources(root))):
        assert secret not in body and answer not in body


@pytest.mark.parametrize(
    "opening,started",
    [
        ([{"type": "text", "text": WORDS}], WORDS),
        ("TEST raw bearer-like value", "TEST [redacted] value"),
    ],
)
def test_start_claims_pre_hook_in_order_despite_host_text_changes(entry, opening, started):
    root, _home, _client, _config, service, provider = entry
    service.start()
    remote = provider()
    remote.remote_hook("pre_llm_call", {"turn_id": "TEST-native", "user_message": opening})
    remote.on_turn_start(1, started)
    remote.remote_hook("post_llm_call", {"turn_id": "TEST-native", "assistant_response": "TEST answer"})
    remote.sync_turn(started, "TEST answer", session_id="TEST-session")
    rows = sources(root)
    assert len([r for r in rows if r["role"] == "user"]) == 1
    assert all("TEST-native" in r["source_event_key"] for r in rows)


def test_offline_checkpoint_survives_shorter_post_compression_history(entry):
    root, _home, _client, config, service, provider = entry
    remote = provider(platform="telegram", chat_type="private", chat_id="TEST-owner", thread_id="main")
    remote.remote_hook("pre_llm_call", {"turn_id": "TEST-turn", "user_message": WORDS})
    remote.on_turn_start(1, WORDS)
    history = [
        {"role": "user", "content": WORDS},
        {"role": "assistant", "content": "TEST interim", "tool_calls": [{}]},
        _row(_steer("TEST checkpoint correction", message_id="TEST-checkpoint", user_id="TEST-owner")),
    ]
    remote.on_pre_compress(history)
    remote.on_session_switch("TEST-compressed", reason="compression", parent_session_id="TEST-session")
    remote.sync_turn(WORDS, "TEST answer", session_id="TEST-compressed", messages=[{"role": "user", "content": WORDS}])
    service.start()
    remote_client.flush_spool(config, seconds=20)
    assert [r["content"] for r in sources(root)].count("TEST checkpoint correction") == 1


def test_privacy_preflight_batches_by_encoded_bytes(entry, monkeypatch):
    _root, _home, _client, config, service, provider = entry
    service.start()
    remote = provider()
    remote.on_turn_start(1, WORDS)
    payload = remote._payload()
    payload["history"] = [{"role": "assistant", "content": "TEST row", "timestamp": time.time()} for _ in range(3000)]
    sizes, original = [], remote_client._post

    def post(config, body, timeout, *, endpoint="/hook"):
        if endpoint == "/private/check":
            sizes.append(len(hermes_queue.encode(body)))
            if sizes[-1] > 65536:
                return None
        return original(config, body, timeout, endpoint=endpoint)

    monkeypatch.setattr(remote_client, "_post", post)
    checked = remote_client._hermes_checked(config, payload, 10)
    assert checked is not None and checked["history"] == payload["history"]
    assert len(sizes) > 1 and max(sizes) <= 65536


def test_switch_binding_stays_bounded_for_long_lived_gateway(entry, monkeypatch):
    _root, _home, _client, _config, _service, provider = entry
    monkeypatch.setattr(remote_client, "hermes_hook", lambda *args, **kwargs: {})
    remote = provider()
    for n in range(500):
        remote.on_session_switch(f"TEST-session-{n}", reason="reset", reset=True)
    assert len(json.dumps(remote._read_payload()).encode()) < 8192
    assert len(remote._switches) <= 1


def test_decoded_size_limit_is_http_413(entry, monkeypatch):
    _root, _home, _client, config, service, _provider = entry
    monkeypatch.setattr(remote_server, "HERMES_MAX_FACT_BYTES", 1024)
    service.start()
    from urllib.parse import urlsplit

    url = urlsplit(config["url"])
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    try:
        connection.request(
            "POST",
            "/hook",
            gzip.compress(json.dumps({"padding": "x" * 2048}).encode()),
            {"Authorization": f"Bearer {TOKEN}", "Content-Encoding": "gzip"},
        )
        response = connection.getresponse()
        assert response.status == 413
        response.read()
        connection.request(
            "POST", "/hook", b"TEST invalid gzip", {"Authorization": f"Bearer {TOKEN}", "Content-Encoding": "gzip"}
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
    finally:
        connection.close()


def test_unsendable_head_counts_loss_and_does_not_block_next_turn(entry, monkeypatch):
    root, _home, _client, config, service, provider = entry
    say(provider("TEST-huge"), session="TEST-huge")
    first = next((config["state_dir"] / "spool").glob("*.json"))
    kept = json.loads(first.read_text(encoding="utf-8"))
    kept["payload"]["padding"] = random.Random(7).randbytes(400000).hex()
    first.write_text(json.dumps(kept), encoding="utf-8")
    say(provider("TEST-small"), "TEST small after huge", session="TEST-small")
    monkeypatch.setattr(remote_client, "HERMES_MAX_WIRE_BYTES", 8192)
    service.start()
    remote_client.flush_spool(config, seconds=20)
    assert not first.exists()
    assert hermes_queue.status(config)["dropped_count"] == 1
    assert "TEST small after huge" in [r["content"] for r in sources(root)]


def test_loss_record_survives_exit_before_eviction_unlink(tmp_path, monkeypatch):
    from pathlib import Path

    config = {"state_dir": tmp_path / "state"}
    hermes_queue.put(config, {"queue_key": "a" * 64, "revision": 1, "body": "x" * 150}, "TEST-first")
    monkeypatch.setattr(hermes_queue, "MAX_BYTES", 500)
    original = Path.unlink

    def crash(path, *args, **kwargs):
        if path.suffix == ".json" and path.parent.name == "spool":
            raise OSError("TEST interrupted eviction")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", crash)
        with pytest.raises(OSError, match="interrupted"):
            hermes_queue.put(config, {"queue_key": "b" * 64, "revision": 1, "body": "x" * 150}, "TEST-second")
    state = hermes_queue.status(config)
    assert state["dropped_count"] == 1 and state["first_dropped_at"] == "TEST-second"
    assert state["queued_turns"] == 1


def test_profile_selection_does_not_invoke_refusal_logger(entry, monkeypatch, caplog):
    _root, _home, client, _config, _service, provider = entry
    remote = provider()
    other = provider()
    other._home = client / "TEST-other-profile"
    agent, scope, constants, secrets = (
        ModuleType(n) for n in ("agent", "agent.memory_caller_scope", "hermes_constants", "agent.secret_scope")
    )
    scope.bound_scope = lambda home: home

    def refusal(home, label):
        if home != client:
            logging.getLogger("TEST.hermes").warning("refused")
            return None
        return {"hermes_home": str(client)}

    scope.caller_scope = refusal
    constants.get_hermes_home_override = lambda: str(client)
    constants.get_hermes_home = lambda: client
    constants.hermes_home_key = lambda home: str(home)
    secrets.current_secret_scope_home = lambda: str(client)
    secrets.is_multiplex_active = lambda: True
    for name, module in (
        ("agent", agent),
        ("agent.memory_caller_scope", scope),
        ("hermes_constants", constants),
        ("agent.secret_scope", secrets),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    calls = []
    monkeypatch.setattr(remote, "remote_hook", lambda *args: calls.append("right"))
    with caplog.at_level(logging.WARNING):
        hooks._dispatch("pre_llm_call", {"session_id": "TEST-session"}, wait=1)
    assert calls == ["right"]
    assert "refused" not in caplog.text


def test_history_without_a_witness_cannot_acquire_a_later_timestamp():
    from scope_recall.adapters.hermes.remote_turn import RemoteTurn

    turn = RemoteTurn({}, [], "TEST-session", WORDS, "a" * 64)
    turn.observe("observe_pre_llm", {"user_message": WORDS}, "2026-01-01T00:00:00Z")
    rows = [{"role": "user", "content": WORDS}, {"role": "assistant", "content": "TEST unknown time"}]
    turn.observe("observe_pre_api_request", {"conversation_history": rows}, "2026-01-01T00:00:01Z")
    rows[-1]["timestamp"] = 2000000000
    turn.observe("on_session_end", {"messages": rows}, "2026-01-02T00:00:00Z")
    assert turn.payload()["history"][-1].get("timestamp") is None


def test_queue_and_http_use_identical_decoded_envelope_bytes(entry):
    _root, _home, _client, config, service, _provider = entry
    payload = {"queue_key": "a" * 64, "revision": 1, "text": "TEST 中文"}
    path = hermes_queue.put(config, payload, "TEST-instant")
    kept = json.loads(path.read_text(encoding="utf-8"))
    raw, headers = remote_client._encoded_request(config, kept, "/hook")
    assert not headers and raw == path.read_bytes()
    assert remote_server._hook_body(service.config, None, raw) == kept
