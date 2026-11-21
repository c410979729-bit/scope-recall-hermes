"""Antigravity hooks, installations and real transports on synthetic shared entries."""

import asyncio
import hashlib
import json
import os
import socket
import sqlite3
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from scope_recall.adapters.clients import antigravity, remote_client, remote_server
from scope_recall.adapters.clients.config import CodexConfigError, load_remote_client, load_shared_client
from scope_recall.adapters.clients.handler import CodexHookHandler
from scope_recall.adapters.hermes.installation import build_installation_manifest
from scope_recall.adapters.shared_entries import (
    attach_shared_entry,
    attach_shared_record,
    client_entry_record,
    new_shared_payload,
    read_shared_payload,
    write_shared_payload,
)
from scope_recall.maintenance.install import apply_install, apply_uninstall, plan_install, plan_uninstall

from tests.unit.test_antigravity_record import step

NOW = "2026-11-17T08:00:00Z"
TEXT = "TEST atlas launch code is BLUEBIRD-17."


@pytest.fixture
def entries(tmp_path, monkeypatch):
    """Two locally authorized entries share one synthetic scope; no model runtime is attached."""
    monkeypatch.setattr(CodexHookHandler, "ensure_runtime", lambda *a, **k: None)
    root = tmp_path / "store"
    write_shared_payload(root, new_shared_payload(root, agent_id="TEST-agent"))
    seed = tmp_path / "seed"
    seed.mkdir()
    attach_shared_entry(
        root,
        build_installation_manifest(
            seed, agent_id="TEST-agent", user_id="TEST-owner", agent_workspace="TEST-workspace"
        ),
        entry_id="seed",
        display_name="TEST seed",
        now=NOW,
    )
    scope = next(row for row in read_shared_payload(root)["entries"][0]["audiences"] if row["kind"] == "owner_private")
    homes = [tmp_path / name for name in ("one", "two")]
    for index, home in enumerate(homes):
        attach_shared_record(
            root,
            client_entry_record(
                host="antigravity",
                home=home,
                entry_id=f"test-ag-{index}",
                display_name="TEST Antigravity",
                attached_at=NOW,
                allowed_scope_ids=scope["allowed_scope_ids"],
                writable_scope_ids=scope["writable_scope_ids"],
                capture_scope_id=scope["capture_scope_id"],
            ),
            now=NOW,
        )
    return root, homes


def native(home, text=TEXT, *, session=None, index=0):
    """Create only synthetic native steps in an isolated entry home."""
    session = session or str(uuid4())
    path = home / "antigravity" / "brain" / session / ".system_generated" / "logs" / "transcript_full.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        content = f"<USER_REQUEST>\n{text}\n</USER_REQUEST>\n<ADDITIONAL_METADATA>TEST host context"
        stream.write(json.dumps(step(index, content=content)) + "\n")
    return dict(
        conversationId=session,
        workspacePaths=[str(home)],
        transcriptPath=str(path),
        lastUserInput=text,
        invocationNum=0,
        initialNumSteps=index + 1,
    )


def rows(root):
    with closing(sqlite3.connect(root / "memory.sqlite3")) as db:
        return db.execute("SELECT role, content, origin FROM source_events ORDER BY rowid").fetchall()


def test_native_prompt_repeats_and_identical_occurrences(entries):
    root, homes = entries
    raw = native(homes[0])
    with closing(CodexHookHandler.from_home(str(homes[0]), "antigravity")) as hook:
        first = antigravity.normalize(raw, "PreInvocation")
        assert first["prompt"] == TEXT
        hook.handle_payload(first)
        for invocation in (1, 2):
            assert antigravity.normalize(dict(raw, invocationNum=invocation), "PreInvocation") == {}
        again = native(homes[0], session=raw["conversationId"], index=2)
        hook.handle_payload(antigravity.normalize(again, "PreInvocation"))
        path = Path(raw["transcriptPath"])
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    step(
                        3,
                        content="TEST visible reply",
                        type="PLANNER_RESPONSE",
                        source="MODEL",
                        thinking="TEST hidden reasoning",
                    )
                )
                + "\n"
            )
            stream.write(json.dumps(step(4, content="TEST generated", type="GENERIC", source="MODEL")) + "\n")
        hook.handle_payload(antigravity.normalize(again, "PostInvocation"))
        hook.handle_payload(antigravity.normalize(again, "Stop"))
    stored = rows(root)
    assert stored == [
        ("user", TEXT, "human_direct"),
        ("user", TEXT, "human_direct"),
        ("assistant", "TEST visible reply", "assistant_visible"),
    ]
    assert not list(homes[0].rglob("*.sqlite3"))
    assert load_shared_client(homes[0], "antigravity").local_account_attested
    assert not load_remote_client(homes[0], "antigravity").local_account_attested


def test_local_account_pin_cannot_be_borrowed_by_another_process(entries, monkeypatch):
    from scope_recall.adapters.clients.identity import capture_owner_principal
    from scope_recall.core import entry_control

    from tests.host.test_local_account_pin import other_account

    _root, homes = entries
    remote = load_remote_client(homes[0], "antigravity")
    monkeypatch.setattr(entry_control, "process_account_id", other_account)
    with pytest.raises(CodexConfigError):
        load_shared_client(homes[0], "antigravity")
    assert capture_owner_principal(remote).resolution == "unresolved"


def test_missing_or_new_incomplete_boundary_never_reuses_user_input(entries):
    root, homes = entries
    payload = native(homes[0])
    assert "prompt" not in antigravity.normalize(dict(payload, initialNumSteps=2), "PreInvocation")
    with Path(payload["transcriptPath"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(step(1, status="RUNNING")) + "\n")
    normalized = antigravity.normalize(dict(payload, initialNumSteps=2), "PreInvocation")
    assert "prompt" not in normalized
    with closing(CodexHookHandler.from_home(str(homes[0]), "antigravity")) as hook:
        hook.handle_payload(normalized)
    assert rows(root) == []
    assert antigravity.normalize_bytes(b"bad", "PreInvocation") == b"{}"
    assert antigravity.answer({"hookSpecificOutput": {"additionalContext": "TEST memory"}}, "PreInvocation") == {
        "injectSteps": [{"ephemeralMessage": "TEST memory"}]
    }
    assert antigravity.answer({"decision": "continue"}, "Stop") == {}


@pytest.mark.parametrize("original", ["2026-01-02T09:00:00Z", None])
@pytest.mark.parametrize("private_mode", ["full", "read-only"])
def test_ingress_checks_the_native_prompt_time(monkeypatch, original, private_mode):
    """An old forwarder supplies a new hook clock for an input it observed earlier."""
    from types import SimpleNamespace

    observed = "2026-01-02T10:00:00Z"
    payload = {"hook_event_name": "UserPromptSubmit", "session_id": "TEST-session", "prompt": "TEST question"}
    if original is not None:
        payload["occurred_at"] = original
    checked = {}

    def decision(_config, request):
        checked.update(zip(request["times"], request["clocks"], strict=True))
        blocked = [moment != observed for moment in request["times"]]
        return {
            "mode": "none",
            "private_rules": True,
            "capture_blocked": False,
            "recall_blocked": False,
            "decisions": blocked,
            "full_decisions": [value and private_mode == "full" for value in blocked],
        }

    monkeypatch.setattr(remote_server, "private_check", decision)
    _, refused, _ = remote_server._checked_body(
        SimpleNamespace(host="antigravity"), {"payload": payload, "observed_at": observed, "clock": {"fresh": True}}
    )
    assert refused
    assert checked[original] is None


@pytest.mark.parametrize("key", ["parentConversationId", "rootConversationId"])
def test_delegated_runs_are_not_human_claims(entries, key):
    root, homes = entries
    raw = native(homes[0]) | {key: str(uuid4())}
    for event in antigravity.EVENTS:
        assert antigravity.normalize(raw, event) == {}
    assert rows(root) == []


def test_two_entries_recall_before_invocation(entries):
    root, homes = entries
    with closing(CodexHookHandler.from_home(str(homes[0]), "antigravity")) as hook:
        hook.handle_payload(antigravity.normalize(native(homes[0]), "PreInvocation"))
    question = native(homes[1], "What is the TEST atlas launch code?")
    with closing(CodexHookHandler.from_home(str(homes[1]), "antigravity")) as hook:
        result = hook.handle_payload(antigravity.normalize(question, "PreInvocation"))
    assert "BLUEBIRD-17" in json.dumps(antigravity.answer(result, "PreInvocation"))
    assert len(rows(root)) == 2


def test_install_preserves_foreign_settings_and_disabled_then_uninstalls(entries):
    from scope_recall.maintenance.doctor import check_host_registration

    _root, homes = entries
    home = homes[0]
    config = home / "config"
    config.mkdir()
    other = {"other": {"PreInvocation": [{"command": "TEST other"}], "enabled": False}}
    (config / "hooks.json").write_text(json.dumps(other), encoding="utf-8")
    (config / "mcp_config.json").write_text(
        json.dumps({"mcpServers": {"other": {"command": "TEST-other"}}}), encoding="utf-8"
    )
    options = dict(
        host="antigravity",
        target_plugin_dir=home,
        instance_root=home,
        python_executable=Path(sys.executable),
        agent_id="TEST-agent",
        test_mode=False,
        project_root=None,
    )
    plan = plan_install(**options)
    assert not plan.conflicts
    apply_install(plan)
    assert check_host_registration("antigravity", home) == "registered"
    hooks = json.loads((config / "hooks.json").read_text())
    assert hooks["other"] == other["other"]
    hooks["scope-recall"]["enabled"] = False
    (config / "hooks.json").write_text(json.dumps(hooks), encoding="utf-8")
    mcp = json.loads((config / "mcp_config.json").read_text())
    mcp["mcpServers"]["scope-recall"]["disabled"] = True
    (config / "mcp_config.json").write_text(json.dumps(mcp), encoding="utf-8")
    apply_install(plan_install(**options))
    assert json.loads((config / "hooks.json").read_text())["scope-recall"]["enabled"] is False
    assert json.loads((config / "mcp_config.json").read_text())["mcpServers"]["scope-recall"]["disabled"] is True
    assert check_host_registration("antigravity", home) == "not_selected"
    uninstall = plan_uninstall(target_plugin_dir=home, instance_root=home)
    assert not uninstall.conflicts
    apply_uninstall(uninstall)
    assert json.loads((config / "hooks.json").read_text()) == other
    assert json.loads((config / "mcp_config.json").read_text()) == {"mcpServers": {"other": {"command": "TEST-other"}}}


async def protocol(args):
    """Exercise initialize, tools/list and actual tool calls over an owned stdio pipe."""
    env = dict(os.environ)
    repo = Path(__file__).parents[3]
    env["PYTHONPATH"] = os.pathsep.join((str(repo), env.get("PYTHONPATH", "")))
    async with stdio_client(StdioServerParameters(command=sys.executable, args=args, env=env, cwd=str(repo))) as (
        read,
        write,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"recall", "status"} <= names
            assert not {"revise", "forget", "propose_memory"} & names
            result = await session.call_tool(
                "recall",
                {
                    "protocol_version": "1.1",
                    "request_id": "TEST-recall",
                    "query": "TEST atlas launch code",
                    "mode": "current",
                    "max_items": 6,
                    "budget_tokens": 1200,
                },
            )
            assert not result.is_error and "BLUEBIRD-17" in str(result)
            for metadata in (
                {},
                {"conversationId": str(uuid4()), "threadId": str(uuid4())},
                {"conversationId": str(uuid4()), "witness": {"issued_at": "2000-01-01T00:00:00Z"}},
            ):
                # The host has no native request-binding contract. Neither an
                # unknown conversation nor a claimed witness unlocks mutations.
                refused = await session.call_tool(
                    "forget", {"protocol_version": "1.1", "targets": ["unknown"]}, meta=metadata
                )
                assert refused.is_error


def test_remote_install_uninstall_has_no_embedded_token(served):
    from scope_recall.maintenance import install_remote

    _root, homes, config = served
    home = config["state_dir"] / "host"
    home.mkdir()
    result = install_remote.install(config, home)
    assert len(result["written"]) == 2
    token = config["token_file"].read_text()
    assert all(token not in Path(path).read_text() for path in result["written"])
    hooks = home / "config" / "hooks.json"
    value = json.loads(hooks.read_text())
    value["other"] = {"enabled": False}
    hooks.write_text(json.dumps(value), encoding="utf-8")
    config["token_file"].unlink()
    removed = install_remote.install(config, home, remove=True)
    assert len(removed["backups"]) == 2
    assert json.loads(hooks.read_text()) == {"other": {"enabled": False}}
    assert not json.loads((home / "config" / "mcp_config.json").read_text())["mcpServers"]
    assert not list(home.rglob("*.sqlite3"))


def test_remote_filters_secrets_before_transport_and_checks_private(served, monkeypatch):
    _root, _homes, config = served
    secret = "sk-" + "TESTsyntheticcredential" * 3
    text = f"TEST before <private>TEST hidden</private> after api_key={secret}"
    payload = native(config["state_dir"], text)
    observed, checks = [], []
    post = remote_client._post
    check = remote_client._private_check

    def checked(*args, **kwargs):
        checks.append(True)
        return check(*args, **kwargs)

    def sent(cfg, body, *args, **kwargs):
        observed.append(json.dumps(body))
        return post(cfg, body, *args, **kwargs)

    monkeypatch.setattr(remote_client, "_private_check", checked)
    monkeypatch.setattr(remote_client, "_post", sent)
    remote_client.run_hook(config, json.dumps(payload).encode(), event="PreInvocation")
    remote_client.run_hook(config, json.dumps(payload).encode(), event="Stop")
    assert checks and observed
    assert all(secret not in body and "TEST hidden" not in body for body in observed)
    assert not list((config["state_dir"] / "spool").glob("*.json"))
    observed.clear()
    monkeypatch.setattr(remote_client, "_private_check", lambda *a, **k: None)
    remote_client.run_hook(config, json.dumps(payload).encode(), event="PreInvocation")
    assert observed == []


def test_remote_notification_accepted_without_logging_an_error(served, monkeypatch):
    from scope_recall.adapters.clients import remote_mcp

    _root, _homes, cfg = served
    logs = []
    monkeypatch.setattr(remote_client, "_log", lambda _config, message: logs.append(message))
    request = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    assert remote_mcp.forward(cfg, request, remote_client._post) is None
    assert logs == []


def test_native_hook_entry_subprocess_injects_ephemeral_context(entries):
    import subprocess

    _root, homes = entries
    with closing(CodexHookHandler.from_home(str(homes[0]), "antigravity")) as hook:
        hook.handle_payload(antigravity.normalize(native(homes[0]), "PreInvocation"))
    payload = native(homes[1], "What is the TEST atlas launch code?")
    repo = Path(__file__).parents[3]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo), os.environ.get("PYTHONPATH", ""))))
    done = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "scope_recall.adapters.clients.hook_entry",
            "--host",
            "antigravity",
            "--home",
            str(homes[1]),
            "--event",
            "PreInvocation",
        ],
        input=json.dumps(payload).encode(),
        capture_output=True,
        timeout=20,
        env=env,
        cwd=repo,
    )
    assert done.returncode == 0, done.stderr.decode(errors="replace")
    answer = json.loads(done.stdout)
    assert set(answer) == {"injectSteps"}
    assert "BLUEBIRD-17" in answer["injectSteps"][0]["ephemeralMessage"]


@pytest.mark.parametrize("conflict", ["foreign_hook", "malformed_hooks", "foreign_mcp"])
def test_install_refuses_foreign_owned_names_without_writes(entries, conflict):
    from scope_recall.maintenance.install_antigravity import files
    from scope_recall.maintenance.install_common import InstallError

    _root, homes = entries
    home = homes[0]
    folder = home / "config"
    folder.mkdir()
    if conflict == "foreign_mcp":
        path, value = folder / "mcp_config.json", {"mcpServers": {"scope-recall": {"command": "TEST-other"}}}
    else:
        path = folder / "hooks.json"
        handlers = None if conflict == "malformed_hooks" else [{"command": "TEST-other"}]
        value = {"scope-recall": {"PreInvocation": handlers}}
    original = json.dumps(value).encode()
    path.write_bytes(original)
    with pytest.raises(InstallError):
        files(home, home, Path(sys.executable))
    assert path.read_bytes() == original


def test_actual_local_stdio_protocol(entries):
    _root, homes = entries
    with closing(CodexHookHandler.from_home(str(homes[0]), "antigravity")) as hook:
        hook.handle_payload(antigravity.normalize(native(homes[0]), "PreInvocation"))
    asyncio.run(
        protocol(["-m", "scope_recall.adapters.clients.mcp_entry", "--host", "antigravity", "--home", str(homes[1])])
    )


@pytest.fixture
def served(entries, tmp_path):
    """One actual authenticated HTTP server, closed and joined at fixture exit."""
    import uvicorn

    root, homes = entries
    with closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    token = "TEST-transport-" + uuid4().hex
    remote_server.write_server_config(
        homes[1], "antigravity", listen="127.0.0.1", port=port, token_sha256=hashlib.sha256(token.encode()).hexdigest()
    )
    config = remote_server.load_server_config(homes[1], "antigravity")
    server = uvicorn.Server(
        uvicorn.Config(remote_server.build_app(config), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started
    state = tmp_path / "remote"
    state.mkdir()
    (state / "token").write_text(token, encoding="utf-8")
    path = state / "client.json"
    path.write_text(
        json.dumps(
            dict(
                host="antigravity",
                url=f"http://127.0.0.1:{port}",
                token_file=str(state / "token"),
                state_dir=str(state),
            )
        ),
        encoding="utf-8",
    )
    try:
        yield root, homes, remote_client.load_client_config(path)
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


def test_loopback_refuses_missing_wrong_capability_and_foreign_host(served):
    import http.client
    from urllib.parse import urlsplit

    _root, homes, config = served
    url = urlsplit(config["url"])
    for headers in ({}, {"Authorization": "Bearer TEST-wrong-entry-capability"}):
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        try:
            conn.request(
                "POST",
                "/mcp",
                body='{"jsonrpc":"2.0","id":1,"method":"tools/list"}',
                headers={"Content-Type": "application/json", **headers},
            )
            response = conn.getresponse()
            assert response.status == 401
            response.read()
        finally:
            conn.close()
    with pytest.raises(CodexConfigError, match="host|entry|attachment"):
        load_remote_client(homes[0], "codex")
    missing = dict(config, token_file=config["state_dir"] / "absent-token")
    assert remote_client._post(missing, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, 2, endpoint="/mcp") is None


def test_loopback_capture_shared_recall_and_stdio_shim(served):
    root, homes, config = served
    payload = native(config["state_dir"])
    remote_client.run_hook(config, json.dumps(payload).encode(), event="PreInvocation")
    remote_client.run_hook(config, json.dumps(payload).encode(), event="Stop")
    with closing(CodexHookHandler.from_home(str(homes[0]), "antigravity")) as hook:
        result = hook.handle_payload(
            antigravity.normalize(native(homes[0], "What is TEST atlas launch code?"), "PreInvocation")
        )
    assert "BLUEBIRD-17" in json.dumps(result)
    assert sum(row[1] == TEXT for row in rows(root)) == 1
    assert not list(config["state_dir"].rglob("*.sqlite3"))
    asyncio.run(protocol(["-m", "scope_recall.adapters.codex.remote_client", "mcp", "--config", str(config["config"])]))


def test_hook_cli_rejects_antigravity_with_a_codex_manifest(tmp_path, capsys):
    from scope_recall.adapters.clients import hook_entry

    with pytest.raises(SystemExit) as rejected:
        hook_entry.main(["--host", "antigravity", "--config", str(tmp_path / "codex-installation.json")])
    assert rejected.value.code == 2
    assert "--config is for a local Codex installation" in capsys.readouterr().err
