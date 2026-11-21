"""The documented Cherry Studio stdio connection against an isolated shared store."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from scope_recall.adapters.clients import CodexHookHandler, local_endpoint, mcp_entry
from scope_recall.adapters.clients.config import CodexConfigError, load_shared_client
from scope_recall.adapters.hermes.installation import build_installation_manifest
from scope_recall.adapters.shared_entries import attach_shared_entry, entry_grants
from scope_recall.contracts import TrustedContext
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.maintenance import cli, shared
from scope_recall.maintenance.setup_bindings import scope_choice
from scope_recall.runtime.worker_entry import load_config

from tests.v11_support import source_event


def _source(store, tmp_path, name, text):
    """Give each synthetic writer a distinct private scope and one identifiable source."""
    home = tmp_path / name
    manifest = build_installation_manifest(home, agent_id="TEST-agent", agent_workspace=name, test_mode=True)
    view = attach_shared_entry(store, manifest, entry_id=name, display_name=name, now="2026-09-05T12:00:00Z")
    scope = view.audience_scopes["owner_private"]
    binding = view.to_binding()
    core = MemoryCore(CoreConfig(binding))
    context = TrustedContext(binding, name + "-session", frozenset({scope}), "human_direct", entry_id=name)
    event = source_event(content=text, source_event_key=name + "-message")
    ref = core.record_event(context, event, scope_id=scope).event_refs[0]
    return scope, ref.ref


def _documented_server(home):
    """Exercise the published JSON with the test interpreter and entry substituted for the two paths."""
    guide = Path(__file__).resolve().parents[3] / "docs/install/cherry-studio.md"
    block = guide.read_text("utf-8").split("```json\n")[1].split("\n```")[0]
    server = json.loads(block)["mcpServers"]["scope-recall"]
    assert server.pop("type") == "stdio" and server["command"] == "<absolute-python-executable>"
    server["command"] = sys.executable
    server["args"][server["args"].index("<absolute-cherry-entry>")] = str(home)
    return StdioServerParameters(**server, env=dict(os.environ))


async def _read_through_stdio(home, visible_ref, hidden_ref):
    async with stdio_client(_documented_server(home)) as (read, write), ClientSession(read, write) as session:
        info = await session.initialize()
        assert info.server_info.name == "scope-recall-cherry-studio"
        tools = (await session.list_tools()).tools
        assert {tool.name for tool in tools} == {
            "recall",
            "inspect",
            "profile",
            "entity",
            "trace",
            "status",
            "handoff_list",
        }
        assert all(tool.annotations.read_only_hint is True for tool in tools)
        status = await session.call_tool("status", {})
        assert status.structured_content["result"]["entry"] == {"id": "reader", "name": "Cherry Studio"}
        args = {"protocol_version": "1.1", "query": "TEST compass", "mode": "history", "max_items": 5}
        recall = await session.call_tool("recall", args)
        contents = [item["content"] for item in recall.structured_content["result"]["items"]]
        assert any("violet" in item for item in contents) and all("copper" not in item for item in contents)
        visible = await session.call_tool("inspect", {"protocol_version": "1.1", "ref": visible_ref + "@1"})
        assert not visible.is_error and "violet" in json.dumps(visible.structured_content)
        hidden = await session.call_tool("inspect", {"protocol_version": "1.1", "ref": hidden_ref + "@1"})
        assert "copper" not in str(hidden)
        assert (await session.call_tool("handoff_write", {"protocol_version": "1.1", "report": {}})).is_error
        assert (await session.call_tool("recall", {**args, "scope_id": "TEST-other-scope"})).is_error


def test_cherry_studio_reads_only_the_selected_entry_scopes(tmp_path, capsys):
    store, home = tmp_path / "TEST-store", tmp_path / "TEST-cherry home"
    shared.init_shared(root=store, agent_id="TEST-agent", test_mode=True)
    scope, visible_ref = _source(store, tmp_path, "allowed", "TEST compass marker violet.")
    _, hidden_ref = _source(store, tmp_path, "separate", "TEST compass marker copper.")
    attach = [
        "attach",
        "--host",
        "cherry-studio",
        "--root",
        str(store),
        "--instance-root",
        str(home),
        "--entry",
        "reader",
        "--display-name",
        "Cherry Studio",
    ]
    assert cli.main(attach) == 2
    assert cli.main([*attach, "--grants-like", "allowed", "--capture-like", "allowed"]) == 2
    assert cli.main([*attach, "--grants-like", "allowed"]) == 0
    capsys.readouterr()
    config = load_shared_client(home, "cherry-studio")
    assert config.audience.allowed_scope_ids == {scope}
    assert not config.audience.writable_scope_ids and config.audience.capture_scope_id is None
    with pytest.raises(CodexConfigError, match="no lifecycle hooks"):
        CodexHookHandler.from_home(str(home), "cherry-studio")
    asyncio.run(_read_through_stdio(home, visible_ref, hidden_ref))

    grants = entry_grants(home)
    grants["audiences"][0].update(writable_scope_ids=[scope], capture_scope_id=scope)
    proposed = tmp_path / "TEST-writable-grants.json"
    proposed.write_text(json.dumps(grants), encoding="utf-8")
    assert cli.main(["grants", "--instance-root", str(home), "--grants-from", str(proposed), "--apply"]) == 2
    assert "read-only MCP entry" in capsys.readouterr().out
    assert cli.main(["detach", "--instance-root", str(home)]) == 0
    with pytest.raises(CodexConfigError, match="not attached"):
        load_shared_client(home, "cherry-studio")


def test_cherry_studio_first_in_an_empty_store_keeps_a_later_writer_independent(tmp_path):
    store, home = tmp_path / "TEST-store", tmp_path / "TEST-reader"
    shared.init_shared(root=store, agent_id="TEST-agent", test_mode=True)
    route = tmp_path / "TEST-routes.json"
    route.write_text(
        json.dumps(
            {
                "binding": {
                    "agent_id": "TEST-agent",
                    "installation_id": "TEST-routes",
                    "data_directory": str(tmp_path),
                    "scope_ids": ["TEST-source"],
                },
                "allowed_scope_ids": ["TEST-source"],
                "session_id": "TEST-routes",
            }
        ),
        encoding="utf-8",
    )
    shared.attach(
        host="cherry-studio",
        instance_root=home,
        root=store,
        entry_id="reader",
        display_name="Reader",
        runtime_config_from=route,
    )
    client = load_shared_client(home, "cherry-studio")
    assert client.audience.allowed_scope_ids and not client.audience.writable_scope_ids
    assert load_config(client.runtime_config_path).host_adapter == "cherry-studio"
    next_home = tmp_path / "TEST-writer"
    choice = scope_choice(next_home, "codex", store, share_with=None, fresh=True)
    assert choice["source_entry"] is None
    shared.attach(host="codex", instance_root=next_home, root=store, entry_id="writer", display_name="Writer")
    writer = load_shared_client(next_home, "codex")
    assert writer.audience.capture_scope_id in client.audience.allowed_scope_ids


def test_doctor_reads_cherry_attachment_without_falling_back_to_codex(tmp_path, capsys):
    store, home = tmp_path / "TEST-store", tmp_path / "TEST-cherry"
    shared.init_shared(root=store, agent_id="TEST-agent", test_mode=True)
    shared.attach(host="cherry-studio", instance_root=home, root=store, entry_id="reader", display_name="Reader")
    command = ["doctor", "--host", "cherry-studio", "--instance-root", str(home)]
    assert cli.main(command) in (0, 1)
    report = json.loads(capsys.readouterr().out)
    assert report["binding_ok"]
    assert report["shared_store"]["entry_id"] == "reader"
    assert not any(check["name"] in {"hook_trust", "capture_authorization"} for check in report["checks"])

    shared.detach(instance_root=home)
    (home / "codex-installation.json").write_text("{}", encoding="utf-8")
    assert cli.main(command) == 1
    report = json.loads(capsys.readouterr().out)
    assert not report["binding_ok"]
    assert "installation_config_missing" in report["capability_gaps"]


def test_cherry_mcp_starts_with_or_without_declared_embedding_credentials(tmp_path, monkeypatch, capsys):
    store, home = tmp_path / "TEST-store", tmp_path / "TEST-cherry"
    shared.init_shared(root=store, agent_id="TEST-agent", test_mode=True)
    key = "SCOPE_RECALL_TEST_CHERRY_EMBED"
    route = tmp_path / "TEST-embedding-route.json"
    route.write_text(
        json.dumps(
            {
                "binding": {
                    "agent_id": "TEST-agent",
                    "installation_id": "TEST-routes",
                    "data_directory": str(tmp_path),
                    "scope_ids": ["TEST-source"],
                },
                "allowed_scope_ids": ["TEST-source"],
                "session_id": "TEST-routes",
                "auxiliary": {
                    "external_embedding": True,
                    "external_consolidation": False,
                    "embedding": {
                        "model": "TEST-embedding",
                        "endpoint": "https://example.test/v1/embeddings",
                        "dimensions": 64,
                        "dialect": "openai",
                        "credential_env": key,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    shared.attach(
        host="cherry-studio",
        instance_root=home,
        root=store,
        entry_id="reader",
        display_name="Reader",
        runtime_config_from=route,
    )
    env_file = tmp_path / "TEST-embedding.env"
    tool_sessions = []

    def opened(server):
        tool_sessions.append((server.name, os.environ.get(key)))

    def no_resident(*args):
        pytest.fail("A read-only MCP connector must not start a hook resident")

    monkeypatch.setattr(mcp_entry, "run_stdio", opened)
    monkeypatch.setattr(local_endpoint, "request_recall", no_resident)
    monkeypatch.setenv(key, "TEST-previous")
    monkeypatch.delenv(key)
    args = ["--host", "cherry-studio", "--home", str(home), "--env-file", str(env_file)]
    assert mcp_entry.main(args) == 0
    assert "credential environment unavailable" in capsys.readouterr().err
    env_file.write_text(f"{key}=TEST-placeholder\nTEST_UNDECLARED_CHERRY=TEST-other\n", encoding="utf-8")
    assert mcp_entry.main(args) == 0
    assert tool_sessions == [("scope-recall-cherry-studio", None), ("scope-recall-cherry-studio", "TEST-placeholder")]
    assert "TEST_UNDECLARED_CHERRY" not in os.environ
    assert "TEST-placeholder" not in capsys.readouterr().err
