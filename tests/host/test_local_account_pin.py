"""Synthetic local routes must prove the installation's approved OS account."""

import json
from dataclasses import replace

import pytest
from scope_recall.adapters.clients.config import (
    CodexConfigError,
    install_codex_scope_recall,
    load_codex_config,
    load_remote_client,
    load_shared_client,
)
from scope_recall.adapters.clients.identity import capture_owner_principal, trusted_context
from scope_recall.adapters.clients.identity import resolve_runtime_audience as client_audience
from scope_recall.adapters.host_identity import (
    HermesRuntimeScope,
    entry_owner,
    resolve_granted_audience,
    resolve_runtime_audience,
)
from scope_recall.adapters.installation import (
    approve_local_platforms,
    build_installation_manifest,
    load_installation_manifest,
    write_installation_manifest,
)
from scope_recall.adapters.shared_entries import (
    ATTACHMENT_SCHEMA,
    attachment_path,
    client_entry_record,
    new_shared_payload,
    read_shared_payload,
    shared_entry_manifest,
    write_shared_payload,
)
from scope_recall.contracts import ContractError
from scope_recall.core import entry_control
from scope_recall.core.storage import SQLiteStorage
from scope_recall.maintenance.local_account import approve, main

from tests.host.hermes.test_remote_entry import entry as remote_entry
from tests.host.hermes.test_remote_entry import say as remote_say
from tests.host.hermes.test_remote_entry import sources as remote_sources


def route(platform="cli", entry_id=""):
    """Make the exact local route declared by the synthetic installation."""
    return HermesRuntimeScope(
        platform,
        "local",
        "cli" if platform == "cli" else "private",
        "local",
        "main",
        "TEST-agent",
        "default",
        "primary",
        entry_id=entry_id,
    )


def other_account():
    """Return a valid synthetic OS identity different from the current account."""
    return "S-1-5-21-999" if entry_control.os.name == "nt" else "999999"


def unavailable():
    """Represent an OS identity read failure, rather than a caller-supplied account."""
    raise OSError("TEST account read failed")


def test_another_os_account_cannot_use_local_owner_grant(tmp_path, monkeypatch):
    """File access alone does not establish the approved local person."""
    manifest = build_installation_manifest(tmp_path / "home", agent_id="TEST-agent")
    assert resolve_runtime_audience(manifest, route()).includes_owner_private
    monkeypatch.setattr(entry_control, "process_account_id", other_account)
    assert not resolve_runtime_audience(manifest, route()).allowed_scope_ids


@pytest.mark.parametrize("reader", [unavailable, lambda: "", lambda: None])
def test_unreadable_account_fails_closed(tmp_path, monkeypatch, reader):
    manifest = build_installation_manifest(tmp_path / "home", agent_id="TEST-agent")
    monkeypatch.setattr(entry_control, "process_account_id", reader)
    assert not resolve_runtime_audience(manifest, route()).allowed_scope_ids
    with pytest.raises(ContractError):
        build_installation_manifest(tmp_path / "new-home", agent_id="TEST-agent")


def test_environment_cannot_replace_os_identity(tmp_path, monkeypatch):
    manifest = build_installation_manifest(tmp_path / "home", agent_id="TEST-agent")
    for name in ("USER", "USERNAME", "UID", "USERDOMAIN"):
        monkeypatch.setenv(name, "TEST impostor")
    assert resolve_runtime_audience(manifest, route()).includes_owner_private


def test_old_manifest_stays_unbound_until_explicit_approval(tmp_path):
    home = tmp_path / "home"
    manifest = build_installation_manifest(home, agent_id="TEST-agent")
    path = write_installation_manifest(manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["local_account_id"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    original = path.read_bytes()
    legacy = load_installation_manifest(home)
    assert not resolve_runtime_audience(legacy, route()).allowed_scope_ids
    assert path.read_bytes() == original
    assert approve("hermes", home)["account_bound"]
    loaded = load_installation_manifest(home)
    assert resolve_runtime_audience(loaded, route()).includes_owner_private
    assert loaded.audiences == manifest.audiences
    assert loaded.to_binding() == manifest.to_binding()


def test_explicit_surface_approval_pins_legacy_without_changing_existing_grants(tmp_path):
    legacy = replace(build_installation_manifest(tmp_path / "home", agent_id="TEST-agent"), local_account_id=None)
    approved = approve_local_platforms(legacy, ("desktop",), agent_workspace="default")
    assert approved.local_account_id == entry_control.current_account_pin()
    assert all(row in approved.audiences for row in legacy.audiences)


def test_different_account_cannot_overwrite_approved_pin(tmp_path, monkeypatch):
    home = tmp_path / "home"
    path = write_installation_manifest(build_installation_manifest(home, agent_id="TEST-agent"))
    original = path.read_bytes()
    monkeypatch.setattr(entry_control, "process_account_id", other_account)
    with pytest.raises(ContractError, match="local_account_binding"):
        approve("hermes", home)
    assert path.read_bytes() == original


def test_local_codex_config_checks_account_at_context_boundary(tmp_path, monkeypatch):
    project = tmp_path / "project"
    config, _ = install_codex_scope_recall(tmp_path / "client", project_root=project)
    loaded = load_codex_config(config.config_path)
    audience = client_audience(loaded, str(project))
    assert audience.allowed_scope_ids
    monkeypatch.setattr(entry_control, "process_account_id", other_account)
    assert not client_audience(loaded, str(project)).allowed_scope_ids
    with pytest.raises(ContractError, match="local_account_binding"):
        trusted_context(loaded, audience, session_id="TEST-session")


def test_old_codex_config_requires_explicit_approval(tmp_path):
    project = tmp_path / "project"
    config, _ = install_codex_scope_recall(tmp_path / "client", project_root=project)
    payload = json.loads(config.config_path.read_text(encoding="utf-8"))
    del payload["local_account_id"]
    config.config_path.write_text(json.dumps(payload), encoding="utf-8")
    legacy = load_codex_config(config.config_path)
    assert not client_audience(legacy, str(project)).allowed_scope_ids
    approve("codex", config.config_path.parent)
    assert client_audience(load_codex_config(config.config_path), str(project)).allowed_scope_ids


def shared_world(tmp_path):
    """Build synthetic Core and entry documents without invoking host installation discovery."""
    root = tmp_path / "store"
    payload = new_shared_payload(root, agent_id="TEST-agent", test_mode=True)
    records = []
    for entry in ("first", "second"):
        home = tmp_path / entry
        record = client_entry_record(
            host="codex",
            home=home,
            entry_id=entry,
            display_name=entry,
            attached_at="2026-09-05T12:00:00Z",
            allowed_scope_ids=["TEST-" + entry],
            writable_scope_ids=["TEST-" + entry],
            capture_scope_id="TEST-" + entry,
        )
        records.append(record)
        path = attachment_path(home)
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                dict(schema=ATTACHMENT_SCHEMA, root=str(root), entry_id=entry, display_name=entry, host="codex")
            ),
            encoding="utf-8",
        )
    payload.update(entries=records, scope_ids=["TEST-first", "TEST-second"])
    write_shared_payload(root, payload)
    manifest = shared_entry_manifest(root, "first")
    SQLiteStorage(replace(manifest.to_binding(), scope_ids=frozenset(payload["scope_ids"]))).initialize()
    return root, payload


def test_shared_account_approval_is_per_entry(tmp_path):
    root, payload = shared_world(tmp_path)
    payload["entries"][0].pop("local_account_id")
    payload["entries"][1]["local_account_id"] = "sid:S-1-5-21-999" if entry_control.os.name == "nt" else "uid:999999"
    write_shared_payload(root, payload)
    with pytest.raises(CodexConfigError):
        load_shared_client(tmp_path / "first", "codex")
    before_second = dict(payload["entries"][1])
    approve("codex", tmp_path / "first")
    config = load_shared_client(tmp_path / "first", "codex")
    assert capture_owner_principal(config).resolution == "verified"
    with pytest.raises(CodexConfigError):
        load_shared_client(tmp_path / "second", "codex")
    assert read_shared_payload(root)["entries"][1] == before_second


def test_remote_token_does_not_borrow_server_account(tmp_path, monkeypatch):
    shared_world(tmp_path)
    config = load_remote_client(tmp_path / "first", "codex")
    assert config.audience.allowed_scope_ids == frozenset({"TEST-first"})
    assert capture_owner_principal(config).resolution == "unresolved"
    monkeypatch.setattr(entry_control, "process_account_id", unavailable)
    assert capture_owner_principal(config).resolution == "unresolved"


def test_approval_cli_requires_confirmation_before_any_write(tmp_path):
    with pytest.raises(SystemExit):
        main(["--host", "hermes", "--instance-root", str(tmp_path)])


def test_unbound_shared_entry_does_not_supply_owner_equivalences(tmp_path):
    root, payload = shared_world(tmp_path)
    payload["entries"][0].pop("local_account_id")
    write_shared_payload(root, payload)
    manifest = shared_entry_manifest(root, "first")
    assert entry_owner(manifest.to_binding(), "first", None, {}) is None
    assert entry_owner(manifest.to_binding(), "second", None, {}) == manifest.installation_id


def test_replay_checks_entry_grants_without_claiming_worker_account_is_the_speaker(tmp_path, monkeypatch):
    root, _ = shared_world(tmp_path)
    manifest = shared_entry_manifest(root, "first")
    monkeypatch.setattr(entry_control, "process_account_id", unavailable)
    assert not resolve_runtime_audience(manifest, route("codex", "first")).allowed_scope_ids
    assert resolve_granted_audience(manifest, route("codex", "first")).writable_scope_ids == frozenset({"TEST-first"})


def test_remote_hermes_local_route_keeps_token_grants_without_owner_person(tmp_path, monkeypatch):
    from scope_recall.adapters.hermes.identity import bind_hermes_identity
    from scope_recall.adapters.installation import install_hermes_scope_recall

    home = tmp_path / "hermes"
    install_hermes_scope_recall(home, agent_id="TEST-agent")
    monkeypatch.setattr(entry_control, "process_account_id", unavailable)
    remote = bind_hermes_identity(
        "TEST-remote",
        hermes_home=str(home),
        agent_identity="TEST-agent",
        agent_workspace="default",
        platform="cli",
        trusted_local_account_principal="",
    )
    assert remote.runtime_audience.allowed_scope_ids
    assert remote.source_principal("human_direct").resolution == "unresolved"
    local = bind_hermes_identity(
        "TEST-local", hermes_home=str(home), agent_identity="TEST-agent", agent_workspace="default", platform="cli"
    )
    assert not local.runtime_audience.allowed_scope_ids


def test_remote_capture_retains_entry_grants_with_unresolved_local_person(remote_entry, monkeypatch):
    from scope_recall.adapters.hermes import remote_handler

    root, _home, _client_home, _config, service, provider = remote_entry
    observed = []
    shutdown = remote_handler.StoredHermesAdapter.shutdown

    def inspect_shutdown(adapter):
        observed.append((adapter._initialized, adapter._diagnostics.capability_gaps))
        shutdown(adapter)

    monkeypatch.setattr(remote_handler.StoredHermesAdapter, "shutdown", inspect_shutdown)
    service.start()
    remote_say(provider())
    rows = remote_sources(root)
    assert rows, observed
    people = [json.loads(row["extra_json"])["source_principal"] for row in rows if row["origin"] == "human_direct"]
    assert people and all(person["resolution"] == "unresolved" for person in people)


def test_top_level_account_approval_command_binds_only_explicit_home(tmp_path, capsys):
    from scope_recall.maintenance.cli import main as maintenance_main

    home = tmp_path / "home"
    manifest = replace(build_installation_manifest(home, agent_id="TEST-agent"), local_account_id=None)
    write_installation_manifest(manifest)
    assert maintenance_main(["approve-local-account", "--host", "hermes", "--instance-root", str(home), "--yes"]) == 0
    assert json.loads(capsys.readouterr().out)["account_bound"]
    assert resolve_runtime_audience(load_installation_manifest(home), route()).includes_owner_private
