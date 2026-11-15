"""Synthetic Hermes providers and real loopback HTTP prove capture, recall and outage replay."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import socket
import sqlite3
import sys
import threading
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
import uvicorn
from scope_recall.adapters.clients import remote_client, remote_server
from scope_recall.adapters.codex.remote_server import build_app
from scope_recall.adapters.hermes import hooks
from scope_recall.adapters.hermes.installation import build_installation_manifest
from scope_recall.adapters.hermes.remote import RemoteHermesProvider
from scope_recall.adapters.hermes.remote_handler import checked_payload
from scope_recall.adapters.hermes.remote_wire import facts
from scope_recall.adapters.shared_entries import attach_shared_entry, new_shared_payload, write_shared_payload
from scope_recall.maintenance import install_remote

TOKEN = "TEST-hermes-entry-token"
WORDS = "TEST zircon lighthouse uses violet paint"
AGENT = "TEST-agent"
WORKSPACE = "TEST-workspace"


class LoopbackService:
    """Own the single loopback server thread and join it before leaving a test."""

    def __init__(self, config):
        self.config = config
        self.server = None
        self.thread = None

    def start(self):
        """Start a fresh app on the same registered port, including after a simulated outage."""
        self.server = uvicorn.Server(
            uvicorn.Config(
                build_app(self.config),
                host="127.0.0.1",
                port=self.config.port,
                log_level="warning",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert self.server.started

    def stop(self):
        """Stop only the thread this fixture created; no host process lookup is used."""
        if self.server is not None:
            self.server.should_exit = True
            self.thread.join(10)
            assert not self.thread.is_alive()
            self.server = None


@pytest.fixture
def entry(tmp_path, monkeypatch):
    """An isolated shared store, a Hermes entry, and its unrelated client profile."""
    root = tmp_path / "TEST-store"
    write_shared_payload(root, new_shared_payload(root, agent_id=AGENT))
    home = tmp_path / "TEST-server-home"
    home.mkdir()
    manifest = build_installation_manifest(
        home, agent_id=AGENT, user_id="TEST-owner", agent_workspace=WORKSPACE, local_platforms=("desktop",)
    )
    routes = [
        {
            "platform": "telegram",
            "user_id": "TEST-owner",
            "chat_type": kind,
            "chat_id": chat,
            "thread_id": "main",
            "gateway_session_key": "",
            "agent_workspace": WORKSPACE,
            "allowed_scope_ids": [scope],
            "writable_scope_ids": [scope],
            "capture_scope_id": scope,
            "kind": "conversation",
        }
        for kind, chat, scope in (
            ("private", "TEST-owner", "scope:TEST-private"),
            ("group", "TEST-a", "scope:TEST-a"),
            ("group", "TEST-b", "scope:TEST-b"),
        )
    ]
    manifest = replace(
        manifest,
        audiences=(*manifest.audiences, *routes),
        scope_ids=manifest.scope_ids | frozenset(row["capture_scope_id"] for row in routes),
    )
    attach_shared_entry(root, manifest, entry_id="cloud-hermes", display_name="Cloud Hermes", now=remote_client._now())
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    remote_server.write_server_config(
        home, "hermes", listen="127.0.0.1", port=port, token_sha256=hashlib.sha256(TOKEN.encode()).hexdigest()
    )
    client_home = tmp_path / "TEST-client-home"
    client_home.mkdir()
    token = client_home / "token"
    token.write_text(TOKEN, encoding="utf-8")
    client_path = client_home / "client.json"
    client_path.write_text(
        json.dumps(
            {
                "host": "hermes",
                "url": f"http://127.0.0.1:{port}",
                "token_file": str(token),
                "state_dir": str(client_home / "state"),
            }
        ),
        encoding="utf-8",
    )
    config = remote_client.load_client_config(client_path)
    service = LoopbackService(remote_server.load_server_config(home, "hermes"))
    providers = []

    def provider(session="TEST-session", **given):
        result = RemoteHermesProvider(config)
        options = dict(
            hermes_home=str(client_home),
            platform="cli",
            agent_identity=AGENT,
            agent_workspace=WORKSPACE,
            user_id="TEST-owner",
        )
        options.update(given)
        result.initialize(session, **options)
        providers.append(result)
        return result

    monkeypatch.setattr(remote_client, "AWAY_SECONDS", 0)
    yield root, home, client_home, config, service, provider
    for item in providers:
        item.shutdown()
    service.stop()


def say(provider, text=WORDS, *, session="TEST-session", turn="TEST-turn", messages=None):
    """Use the order and signatures of Hermes' public provider and global observation hooks."""
    history = messages or [{"role": "user", "content": text, "platform_message_id": "TEST-platform-1"}]
    hooks._dispatch(
        "pre_llm_call",
        {"session_id": session, "turn_id": turn, "user_message": text, "conversation_history": history},
        wait=1,
    )
    provider.on_turn_start(1, text)
    provider.remote_hook(
        "post_llm_call",
        {
            "session_id": session,
            "turn_id": turn,
            "assistant_response": "TEST acknowledged",
            "conversation_history": history,
        },
    )
    provider.sync_turn(text, "TEST acknowledged", session_id=session, messages=history)


def sources(root):
    with closing(sqlite3.connect(root / "memory.sqlite3")) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM source_events ORDER BY rowid")]


def test_turn_reaches_remote_entry_and_new_session_recalls_it(entry):
    root, _home, client_home, _config, service, provider = entry
    service.start()
    first = provider()
    say(first)
    rows = sources(root)
    assert [row["content"] for row in rows] == [WORDS, "TEST acknowledged"]
    assert {row["entry_id"] for row in rows} == {"cloud-hermes"}
    assert {row["origin"] for row in rows} == {"human_direct", "assistant_visible"}
    second = provider("TEST-new-session")
    recalled = second.prefetch("What paint does the zircon lighthouse use?", session_id="TEST-new-session")
    assert "violet" in recalled and "Cloud Hermes" in recalled
    assert not list(client_home.rglob("*.sqlite3"))
    status = json.loads(second.handle_tool_call("status", {"protocol_version": "1.1"}))
    assert status["result"]["session"].endswith("TEST-new-session")
    handoffs = json.loads(second.handle_tool_call("handoff_list", {"protocol_version": "1.1"}))
    assert "result" in handoffs


def test_outage_retains_hooks_and_restart_replays_each_source_once(entry):
    root, _home, client_home, config, service, provider = entry
    first = provider()
    say(first)
    queued = list((config["state_dir"] / "spool").glob("*.json"))
    assert queued and not sources(root)
    retained = [path.read_bytes() for path in queued]
    service.start()
    assert remote_client.flush_spool(config, seconds=20) == len(queued)
    assert not list((config["state_dir"] / "spool").glob("*.json"))
    assert len(sources(root)) == 2
    service.stop()
    service.start()
    # A lost response is the same envelope after restart, not a new source or changed occurrence time.
    for raw in retained:
        answer = remote_client._post(config, json.loads(raw), 10)
        assert answer is not None and not answer["retry"]
    assert len(sources(root)) == 2
    assert not list(client_home.rglob("*.sqlite3"))


def test_wire_preserves_origin_and_steer_markers_without_tools_or_paths():
    value = facts(
        {
            "hermes_home": "/private/home",
            "conversation_history": [
                {
                    "role": "user",
                    "content": "TEST notice",
                    "display_kind": "hidden",
                    "_compressed_summary": True,
                    "platform_message_id": "TEST-platform",
                    "api_content": "TEST redirect",
                },
                {
                    "role": "assistant",
                    "content": "TEST interim",
                    "tool_calls": [{"function": {"arguments": "TEST secret"}}],
                },
                {"role": "tool", "content": "TEST tool output", "path": "/private/path"},
            ],
        }
    )
    encoded = json.dumps(value)
    assert "TEST secret" not in encoded and "TEST tool output" not in encoded and "/private" not in encoded
    assert value["conversation_history"][0]["_compressed_summary"] is True
    assert value["conversation_history"][1]["tool_calls"]
    assert value["conversation_history"][2] == {"role": "tool"}


def test_untrusted_method_is_rejected_before_store_access():
    with pytest.raises(ValueError, match="observation"):
        checked_payload(
            {
                "binding": {"session_id": "TEST-session"},
                "events": [
                    {"method": "handle_tool_call", "values": {}, "at": remote_client._now()},
                ],
            }
        )


def test_remote_installer_writes_profile_plugin_without_database(entry):
    _root, _home, client_home, config, _service, provider = entry
    result = install_remote.install(config, client_home)
    assert result["written"]
    plugin = client_home / "plugins" / "scope-recall"
    assert "register_memory_provider" in (plugin / "__init__.py").read_text(encoding="utf-8")
    assert not list(client_home.rglob("*.sqlite3"))
    assert {row["name"] for row in provider().get_tool_schemas()} == {
        "recall",
        "inspect",
        "trace",
        "profile",
        "entity",
        "status",
        "handoff_list",
    }


def test_server_refuses_mutation_and_unbound_mcp_reads(entry):
    _root, _home, _client_home, config, service, provider = entry
    service.start()
    assert "ACCESS_DENIED" in provider().handle_tool_call("forget", {})
    for name in ("forget", "status"):
        answer = remote_client.hermes_tool(config, name, {}, {})
        assert "ACCESS_DENIED" in answer


def test_host_notice_stays_host_generated(entry):
    root, _home, _client_home, _config, service, provider = entry
    service.start()
    say(
        provider(),
        "TEST background completed",
        messages=[
            {"role": "user", "content": "TEST background completed", "display_kind": "background_result"},
        ],
    )
    assert next(row for row in sources(root) if row["role"] == "user")["origin"] == "host_generated"


def test_recording_controls_survive_replayed_pause_after_resume(entry):
    root, _home, _client_home, config, service, provider = entry
    service.start()
    remote = provider()
    say(remote, "scope-recall pause", turn="TEST-pause")
    paused = remote._payload()
    say(remote, "TEST hidden while paused", turn="TEST-private")
    assert not sources(root)
    say(remote, "scope-recall resume", turn="TEST-resume")
    # A late duplicated pause must not reopen an interval after the newer resume.
    reply = remote_client._post(config, {"payload": paused}, 10)
    assert reply is not None and not reply.get("retry")
    say(remote, "TEST visible after resume", turn="TEST-visible")
    assert "TEST visible after resume" in [row["content"] for row in sources(root)]
    assert "TEST hidden while paused" not in json.dumps(sources(root))


def test_compression_keeps_opener_in_original_session(entry):
    root, _home, _client_home, _config, service, provider = entry
    service.start()
    remote = provider()
    remote.remote_hook("pre_llm_call", {"session_id": "TEST-session", "turn_id": "TEST-turn", "user_message": WORDS})
    remote.on_turn_start(1, WORDS)
    remote.on_session_switch("TEST-compressed", reason="compression", parent_session_id="TEST-session")
    remote.sync_turn(WORDS, "TEST answer", session_id="TEST-compressed")
    users = [row for row in sources(root) if row["role"] == "user"]
    assert len(users) == 1 and users[0]["session_id"].endswith("TEST-session")
    say(remote, "TEST next turn", session="TEST-compressed", turn="TEST-next")
    assert len([row for row in sources(root) if row["content"] == "TEST next turn"]) == 1


def test_private_spans_and_tool_content_never_reach_queue(entry):
    root, _home, _client_home, config, _service, provider = entry
    remote = provider()
    say(
        remote,
        "TEST public <private>TEST hidden span</private>",
        messages=[
            {"role": "user", "content": "TEST public <private>TEST hidden span</private>"},
            {"role": "tool", "content": "TEST tool output"},
        ],
    )
    remote.remote_hook("post_tool_call", {"session_id": "TEST-session", "result": "TEST secret output"})
    kept = "".join(path.read_text(encoding="utf-8") for path in (config["state_dir"] / "spool").glob("*.json"))
    assert "TEST hidden span" not in kept and "TEST tool output" not in kept and "TEST secret output" not in kept
    assert not sources(root)


def test_remote_profile_cannot_rebind_or_dispatch_another_profile(entry, tmp_path):
    _root, _home, _client_home, _config, _service, provider = entry
    remote = provider()
    with pytest.raises(ValueError, match="another profile"):
        remote.initialize("TEST-foreign", hermes_home=str(tmp_path / "TEST-other"))
    assert "ACCESS_DENIED" in remote.handle_tool_call("status", {}, hermes_home=str(tmp_path / "TEST-other"))


def test_remote_install_refuses_a_local_store_binding(entry):
    _root, _home, client_home, config, _service, _provider = entry
    target = client_home / "scope-recall"
    target.mkdir()
    (target / "attachment.json").write_text("{}", encoding="utf-8")
    with pytest.raises(remote_client.RemoteClientError, match="local store"):
        install_remote.install(config, client_home)
    assert not (client_home / "plugins").exists()


def test_private_and_group_audiences_remain_separate(entry):
    root, _home, _client_home, _config, service, provider = entry
    service.start()
    private = provider("TEST-private", platform="telegram", chat_type="private", chat_id="TEST-owner", thread_id="main")
    group_a = provider("TEST-group-a", platform="telegram", chat_type="group", chat_id="TEST-a", thread_id="main")
    group_b = provider("TEST-group-b", platform="telegram", chat_type="group", chat_id="TEST-b", thread_id="main")
    say(private, "TEST quartz private sentinel", session="TEST-private")
    say(group_a, "TEST zircon group sentinel", session="TEST-group-a")
    assert "quartz" not in group_a.prefetch("quartz private sentinel", session_id="TEST-group-a")
    assert "zircon" not in group_b.prefetch("zircon group sentinel", session_id="TEST-group-b")
    assert {row["scope_id"] for row in sources(root) if row["role"] == "user"} == {"scope:TEST-private", "scope:TEST-a"}


def test_desktop_transport_uses_client_account_instead_of_server_account(entry, monkeypatch):
    from scope_recall.adapters.hermes import identity

    root, _home, _client_home, _config, service, provider = entry
    monkeypatch.setattr(identity, "host_account_principal", lambda: "uid:TEST-client")
    remote = provider(
        platform="desktop",
        user_id="uid:TEST-client",
        chat_type="dm",
        chat_id="TEST-window",
        thread_id="",
        gateway_session_key="agent:default:local:dm:TEST-window",
    )
    monkeypatch.setattr(identity, "host_account_principal", lambda: "sid:TEST-server")
    service.start()
    say(remote)
    assert any(row["content"] == WORDS for row in sources(root))


def test_registered_plugin_selects_remote_mode_without_store_access(entry, monkeypatch):
    _root, _home, client_home, config, _service, _provider = entry
    install_remote.install(config, client_home)
    monkeypatch.setenv("HERMES_HOME", str(client_home))
    # Load the installed directory as Hermes loads a user plugin: a package under its own name whose register
    # attribute it calls. A submodule of the same name would replace that function once imported.
    plugin = client_home / "plugins" / "scope-recall"
    spec = importlib.util.spec_from_file_location(
        "TEST_hermes_user_plugin", plugin / "__init__.py", submodule_search_locations=[str(plugin)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    for name in (path.stem for path in plugin.glob("*.py") if path.stem != "__init__"):
        importlib.import_module(f"{spec.name}.{name}")
    assert callable(module.register)

    class Context:
        def register_memory_provider(self, provider):
            self.provider = provider

        def register_hook(self, event, callback):
            pass

    context = Context()
    module.register(context)
    remote = context.provider
    try:
        assert isinstance(remote, RemoteHermesProvider)
        assert remote.is_available()
        assert not list(client_home.rglob("*.sqlite3"))
    finally:
        remote.shutdown()


def test_typed_steer_keeps_platform_identity_and_tools_are_not_sources(entry):
    from .test_steers import _row, _steer

    root, _home, _client_home, _config, service, provider = entry
    service.start()
    remote = provider()
    history = [{"role": "user", "content": WORDS}]
    remote.remote_hook(
        "pre_llm_call",
        {"session_id": "TEST-session", "turn_id": "TEST-turn", "user_message": WORDS, "conversation_history": history},
    )
    remote.on_turn_start(1, WORDS)
    history += [
        {"role": "assistant", "content": "TEST interim", "tool_calls": [{"function": {"arguments": "TEST args"}}]},
        {"role": "tool", "content": "TEST output"},
        _row(_steer("TEST steering correction", message_id="TEST-platform-steer", user_id="TEST-owner")),
    ]
    remote.remote_hook(
        "pre_api_request",
        {"session_id": "TEST-session", "turn_id": "TEST-turn", "user_message": WORDS, "conversation_history": history},
    )
    remote.sync_turn(WORDS, "TEST answer", session_id="TEST-session", messages=history)
    rows = sources(root)
    steers = [row for row in rows if row["content"] == "TEST steering correction"]
    assert len(steers) == 1 and "TEST-platform-steer" in steers[0]["source_event_key"]
    assert "TEST output" not in json.dumps(rows) and "TEST args" not in json.dumps(rows)


def test_read_tool_uses_hook_source_fence_without_replaying_captures(entry):
    root, _home, _client_home, _config, service, provider = entry
    service.start()
    remote = provider()
    say(remote)
    before = sources(root)
    result = json.loads(
        remote.handle_tool_call(
            "recall",
            {
                "protocol_version": "1.1",
                "query": WORDS,
                "mode": "auto",
                "max_items": 8,
                "budget_tokens": 4096,
            },
        )
    )
    assert not result["result"]["items"]
    assert sources(root) == before


def test_global_hook_refuses_ambiguous_remote_audiences(entry, monkeypatch):
    _root, _home, _client_home, _config, _service, provider = entry
    first = provider(platform="telegram", chat_type="private", chat_id="TEST-owner")
    second = provider(platform="telegram", chat_type="group", chat_id="TEST-a")
    calls = []
    monkeypatch.setattr(first, "remote_hook", lambda *args: calls.append("private"))
    monkeypatch.setattr(second, "remote_hook", lambda *args: calls.append("group"))
    assert hooks._dispatch("pre_llm_call", {"session_id": "TEST-session", "platform": "telegram"}, wait=1) is None
    assert not calls


def test_server_uses_registered_home_and_ignores_paths_in_facts(entry, tmp_path):
    root, _home, _client_home, config, service, provider = entry
    remote = provider()
    payload = remote._payload()
    foreign = tmp_path / "TEST-not-a-store"
    payload["binding"]["hermes_home"] = str(foreign)
    payload["binding"]["trusted_runtime_config_path"] = str(foreign / "runtime-config.json")
    payload["events"] = [
        {
            "method": "observe_pre_llm",
            "at": remote_client._now(),
            "values": {
                "session_id": "TEST-session",
                "turn_id": "TEST-path",
                "user_message": WORDS,
                "transcript_path": str(foreign / "record.jsonl"),
                "hermes_home": str(foreign),
            },
        }
    ]
    service.start()
    answer = remote_client._post(config, {"payload": payload}, 10)
    assert answer is not None and not answer.get("retry")
    assert not foreign.exists()
    assert [row["entry_id"] for row in sources(root)] == ["cloud-hermes"]


def test_path_presence_preserves_the_existing_attachment_refusal():
    from scope_recall.adapters.hermes.attachments import authorize_attachment_metadata

    attachment = {
        "filename": "TEST.txt",
        "media_type": "text/plain",
        "sha256": "a" * 64,
        "byte_length": 10,
        "path": "/TEST/client-only.txt",
    }
    forwarded = facts({"attachments": [attachment]})["attachments"][0]
    assert forwarded["path"] is True and "/TEST/client-only.txt" not in json.dumps(forwarded)
    assert authorize_attachment_metadata(forwarded).gap == "attachment_gap:untrusted_path"


def test_remote_session_preserves_the_existing_hermes_identity_length_limit(entry):
    root, _home, _client_home, _config, service, provider = entry
    session = "TEST-" + "s" * 180
    service.start()
    say(provider(session), session=session)
    assert any(row["content"] == WORDS and row["session_id"].endswith(session) for row in sources(root))
