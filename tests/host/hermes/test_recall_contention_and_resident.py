"""Regression coverage for Hermes hook capture bounds, partial recall under deadline, and resident recall."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from scope_recall.adapters.clients import local_endpoint, resident_entry, store_service
from scope_recall.adapters.hermes import ScopeRecallHermesAdapter
from scope_recall.adapters.hermes import hooks as hermes_hooks
from scope_recall.adapters.hermes.installation import build_installation_manifest, install_hermes_scope_recall
from scope_recall.adapters.hermes.shared_entries import attach_shared_entry, new_shared_payload, write_shared_payload
from scope_recall.contracts import validate_payload
from scope_recall.core import CoreConfig, MemoryCore, retrieval_storage
from scope_recall.maintenance import resident as resident_cli

from tests.v11_support import source_event


def test_pre_api_request_hook_stops_within_deadline_instead_of_overrunning_host_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When lock contention slows steer processing, pre_api_request stops inside the host timeout."""
    home = tmp_path / "TEST-hermes-home"
    install_hermes_scope_recall(home, agent_id="TEST-agent")
    provider = ScopeRecallHermesAdapter()
    provider.initialize(
        "TEST-session",
        hermes_home=str(home),
        agent_identity="TEST-agent",
        agent_workspace="default",
        platform="cli",
    )

    class FakePluginContext:
        def __init__(self) -> None:
            self.callbacks: dict[str, object] = {}

        def register_hook(self, event: str, callback: object) -> None:
            self.callbacks[event] = callback

    ctx = FakePluginContext()
    monkeypatch.setattr(hermes_hooks, "_iter_hook_callbacks", lambda: lambda *args: ())
    hermes_hooks.register_capture_hooks(ctx, provider)
    callback = ctx.callbacks["pre_api_request"]
    assert callable(callback)

    monkeypatch.setattr(hermes_hooks, "host_hook_timeout", lambda: 0.9)
    provider.on_turn_start(1, "TEST opener under contention", session_id="TEST-session")

    original_pre_api = provider._turns.pre_api
    release_contender = threading.Event()
    contender_held = threading.Event()

    def contender() -> None:
        with provider._lock:
            contender_held.set()
            release_contender.wait(timeout=2.0)

    def pre_api_then_contend(**kwargs):
        result = original_pre_api(**kwargs)

        # Immediately after pre_api releases _lock, another session callback holds _lock for 1.5s,
        # reproducing unbounded blocking inside steers() past the host's 0.9s hook timeout.
        def grab_after_release() -> None:
            time.sleep(0.01)
            contender()

        threading.Thread(target=grab_after_release, daemon=True).start()
        time.sleep(0.05)
        return result

    monkeypatch.setattr(provider._turns, "pre_api", pre_api_then_contend)

    history = [
        {
            "role": "user",
            "display_kind": "steer",
            "content": f"TEST steer message {idx}",
            "timestamp": f"2026-10-10T10:00:{idx:02d}Z",
        }
        for idx in range(1, 10)
    ]

    caplog.set_level(logging.WARNING, logger="scope_recall")
    try:
        started = time.monotonic()
        callback(
            session_id="TEST-session",
            turn_id="TEST-turn-1",
            user_message="TEST opener under contention",
            conversation_history=history,
            hermes_home=str(home),
        )
        elapsed = time.monotonic() - started
    finally:
        release_contender.set()
        provider.shutdown()

    assert elapsed < 0.9, f"hook callback took {elapsed:.2f}s, exceeding 0.9s host timeout"
    assert not [rec for rec in caplog.records if "past the host's" in rec.getMessage()]
    assert (provider._calls.counts().get("pre_api_request_overran") or 0) == 0


def test_recall_packet_delivers_collected_items_when_collection_hits_deadline(tmp_path: Path, monkeypatch) -> None:
    """Candidates collected before a deadline interruption still hydrate and ship within the release reserve."""
    home = tmp_path / "TEST-partial-home"
    binding, core = install_hermes_scope_recall(home, agent_id="TEST-agent")
    scope = next(iter(binding.scope_ids))
    from scope_recall.contracts import TrustedContext

    ctx = TrustedContext(binding, "TEST-partial-session", frozenset({scope}), "human_direct")
    for idx in range(4):
        core.record_event(
            ctx,
            source_event(
                content=f"TEST CobaltFalcon deployment port is 94{idx}0.", source_event_key=f"TEST-partial-{idx}"
            ),
            scope_id=scope,
            remaining_seconds=5.0,
        )

    original_claims = retrieval_storage.RetrievalStorage.claims

    def slow_claims(self, tx, context, *, limit):
        # Simulate GIL contention pushing wall time past working.deadline right after lexical succeeded,
        # while leaving part of the release reserve before search_context.deadline.
        clock._mono = context.deadline + 0.08
        return original_claims(self, tx, context, limit=limit)

    class FakeClock:
        def __init__(self) -> None:
            self._mono = 100.0

        def monotonic(self) -> float:
            return self._mono

        def utc_now(self) -> str:
            return "2026-10-10T10:05:00Z"

    clock = FakeClock()
    timed_core = MemoryCore(CoreConfig(binding), clock=clock)
    monkeypatch.setattr(retrieval_storage.RetrievalStorage, "claims", slow_claims)

    packet = timed_core.recall_packet(
        ctx,
        {
            "protocol_version": "1.1",
            "request_id": "TEST-partial-deadline",
            "query": "CobaltFalcon deployment port",
            "mode": "current",
            "max_items": 3,
            "budget_tokens": 2048,
        },
        deadline_seconds=0.5,
        background_without_evidence=False,
    )
    validate_payload("recall_packet", packet)
    assert "deadline_exceeded_collect" in packet["gaps"]
    assert len(packet["items"]) >= 1
    assert packet["status"] == "partial"
    assert any("CobaltFalcon" in item["content"] for item in packet["items"])


def _seed_shared_hermes_store(tmp_path: Path):
    """Create a synthetic shared store with owner_private and group-only scopes and seed one event in each."""
    from scope_recall.contracts import TrustedContext

    store_root = tmp_path / "TEST-shared-store"
    write_shared_payload(store_root, new_shared_payload(store_root, agent_id="TEST-agent", test_mode=True))
    hermes_home = tmp_path / "TEST-hermes-entry"
    hermes_home.mkdir()
    manifest = build_installation_manifest(
        hermes_home, agent_id="TEST-agent", agent_workspace="default", test_mode=True
    )
    shared_scope = "scope:TEST-group-shared"
    owner_scope = manifest.audience_scopes["owner_private"]
    updated_audiences = [
        {**row, "allowed_scope_ids": sorted({*row["allowed_scope_ids"], shared_scope})} for row in manifest.audiences
    ]
    group_row = {
        "platform": "telegram",
        "user_id": "TEST-visitor",
        "chat_type": "group",
        "chat_id": "TEST-group-1",
        "thread_id": "",
        "gateway_session_key": "",
        "agent_workspace": "default",
        "kind": "conversation",
        "allowed_scope_ids": [shared_scope],
        "writable_scope_ids": [shared_scope],
        "capture_scope_id": shared_scope,
    }
    manifest = type(manifest)(
        **{
            **manifest.__dict__,
            "scope_ids": frozenset({*manifest.scope_ids, shared_scope}),
            "audiences": (*updated_audiences, group_row),
        }
    )
    attached = attach_shared_entry(
        store_root,
        manifest,
        entry_id="desk",
        display_name="Desk Hermes",
        now="2026-10-10T10:00:00Z",
    )
    runtime_cfg = {
        "binding": {
            "agent_id": attached.agent_id,
            "installation_id": attached.installation_id,
            "data_directory": str(store_root),
            "scope_ids": sorted(attached.scope_ids),
            "test_mode": True,
            "installation_kind": "shared",
        },
        "session_id": "TEST-worker-session",
        "allowed_scope_ids": sorted(attached.scope_ids),
        "actor_origin": "human_direct",
        "owner_id": "TEST-owner",
        "request_seconds": 5.0,
        "drain_seconds": 5.0,
        "max_items": 4,
        "lease_seconds": 10.0,
        "auxiliary": {"external_embedding": False, "external_consolidation": False},
    }
    (hermes_home / "scope-recall" / "runtime-config.json").write_text(json.dumps(runtime_cfg), encoding="utf-8")
    seed_core = MemoryCore(CoreConfig(attached.to_binding()))
    seed_ctx = TrustedContext(attached.to_binding(), "TEST-seed", attached.scope_ids, "human_direct", entry_id="desk")
    seed_core.record_event(
        seed_ctx,
        source_event(content="TEST OrchidVault private passphrase policy is strict.", source_event_key="TEST-priv-1"),
        scope_id=owner_scope,
        remaining_seconds=5.0,
    )
    seed_core.record_event(
        seed_ctx,
        source_event(content="TEST OrchidVault public channel schedule is Tuesday.", source_event_key="TEST-shared-1"),
        scope_id=shared_scope,
        remaining_seconds=5.0,
    )
    return hermes_home, attached


def _exercise_resident_providers(hermes_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify owner and group sessions delegate prefetch and explicit recall while preserving scope isolation."""
    owner_provider = ScopeRecallHermesAdapter()
    owner_provider.initialize(
        "TEST-owner-cli",
        hermes_home=str(hermes_home),
        agent_identity="TEST-agent",
        agent_workspace="default",
        platform="cli",
    )
    group_provider = ScopeRecallHermesAdapter()
    group_provider.initialize(
        "TEST-group-chat",
        hermes_home=str(hermes_home),
        agent_identity="TEST-agent",
        agent_workspace="default",
        platform="telegram",
        user_id="TEST-visitor",
        chat_type="group",
        chat_id="TEST-group-1",
        thread_id="",
    )
    try:

        def forbidden_in_process(*args, **kwargs):
            raise AssertionError("TEST gateway should have used resident recall endpoint")

        monkeypatch.setattr(owner_provider._require_core(), "recall_packet", forbidden_in_process)
        monkeypatch.setattr(group_provider._require_core(), "recall_packet", forbidden_in_process)

        stop_hog = threading.Event()

        def gil_hog() -> None:
            acc = 0
            while not stop_hog.is_set():
                for i in range(100_000):
                    acc = (acc + i) ^ 0x5A5A
                time.sleep(0.002)
            assert acc >= 0

        hog = threading.Thread(target=gil_hog, daemon=True)
        hog.start()
        try:
            owner_provider.on_turn_start(1, "OrchidVault schedule policy", session_id="TEST-owner-cli")
            owner_text = owner_provider.prefetch(
                "What is the OrchidVault policy and schedule?", session_id="TEST-owner-cli"
            )
            group_provider.on_turn_start(1, "OrchidVault schedule policy", session_id="TEST-group-chat")
            group_text = group_provider.prefetch(
                "What is the OrchidVault policy and schedule?", session_id="TEST-group-chat"
            )
        finally:
            stop_hog.set()
            hog.join(timeout=2.0)

        assert "private passphrase policy" in owner_text
        assert "public channel schedule" in owner_text
        assert "public channel schedule" in group_text
        assert "private passphrase policy" not in group_text

        explicit = json.loads(
            owner_provider.handle_tool_call(
                "recall",
                {
                    "protocol_version": "1.1",
                    "query": "OrchidVault private passphrase policy",
                    "mode": "current",
                    "max_items": 4,
                    "budget_tokens": 2048,
                },
            )
        )
        assert any("private passphrase policy" in item["content"] for item in explicit["result"]["items"])
    finally:
        owner_provider.shutdown()
        group_provider.shutdown()


def test_hermes_resident_recall_serves_prefetch_and_preserves_scope_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hermes delegates recall to an out-of-process resident endpoint while enforcing each session's audience."""
    os_home = tmp_path / "TEST-os-home"
    os_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(os_home))
    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: os_home))

    hermes_home, attached = _seed_shared_hermes_store(tmp_path)

    assert local_endpoint.resident_minutes(hermes_home, "hermes") == 120
    local_endpoint.request_recall(hermes_home, "hermes")
    assert local_endpoint.resident_alive(hermes_home, "hermes").is_file()
    assert store_service.library_minutes(attached.to_binding()) == 120

    resident_code = (
        "import sys\n"
        "from scope_recall.adapters.clients import resident_entry\n"
        "raise SystemExit(resident_entry.main(sys.argv[1:]))\n"
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    resident_proc = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            resident_code,
            "--home",
            str(hermes_home),
            "--host",
            "hermes",
            "--idle-seconds",
            "60",
        ],
        creationflags=creationflags,
    )
    try:
        folder = local_endpoint.endpoints(hermes_home)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not list(folder.glob("*.json")):
            time.sleep(0.05)
        assert list(folder.glob("*.json")), "resident server did not advertise its endpoint"

        assert resident_cli.main(["status", "--home", str(hermes_home), "--host", "hermes"]) == 0
        status_out = json.loads(capsys.readouterr().out)
        assert status_out["status"] == "ok"
        assert status_out["running"] is True
        assert status_out["resident_recall_minutes"] == 120

        _exercise_resident_providers(hermes_home, monkeypatch)

        assert resident_cli.main(["stop", "--home", str(hermes_home), "--host", "hermes"]) == 0
        stop_out = json.loads(capsys.readouterr().out)
        assert stop_out["running"] is False
    finally:
        if resident_proc.poll() is None:
            resident_proc.kill()
        resident_proc.wait(timeout=10)
    assert resident_entry is not None
    assert hermes_hooks._input_hook_deadline(100.0, 0.2) < 100.2


def test_hermes_resident_ignores_worker_only_config_and_cross_account_entries(tmp_path: Path) -> None:
    """Entries without their own runtime-config or pinned to another OS account stay at 0 resident minutes."""
    from scope_recall.adapters.clients.config import CodexConfigError
    from scope_recall.adapters.clients.hermes_resident import load_hermes_resident_target, shared_resident_entries
    from scope_recall.adapters.hermes.shared_entries import read_shared_payload

    store_root = tmp_path / "TEST-worker-only-store"
    write_shared_payload(store_root, new_shared_payload(store_root, agent_id="TEST-agent", test_mode=True))
    hermes_home = tmp_path / "TEST-lexical-entry"
    hermes_home.mkdir()
    manifest = build_installation_manifest(
        hermes_home, agent_id="TEST-agent", agent_workspace="default", test_mode=True
    )
    attached = attach_shared_entry(
        store_root, manifest, entry_id="lexical", display_name="Lexical Hermes", now="2026-10-10T10:00:00Z"
    )
    MemoryCore(CoreConfig(attached.to_binding())).initialize()
    (store_root / "runtime-config.json").write_text("{}", encoding="utf-8")

    # Worker's <store>/runtime-config.json alone does not enable resident recall on a lexical-only entry.
    assert local_endpoint.resident_minutes(hermes_home, "hermes") == 0

    # Pinning the entry to a different OS account excludes it from local resident target loading and enumeration.
    payload = read_shared_payload(store_root)
    payload["entries"][0]["local_account_id"] = "uid:999999999"
    write_shared_payload(store_root, payload)
    with pytest.raises(CodexConfigError, match="another local OS account"):
        load_hermes_resident_target(hermes_home)
    assert list(shared_resident_entries(attached.to_binding())) == []

    # A home holding both a shared-store attachment and its own installation.json is refused fail-closed.
    payload["entries"][0].pop("local_account_id", None)
    write_shared_payload(store_root, payload)
    (hermes_home / "scope-recall" / "installation.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CodexConfigError, match="both its own installation and a shared store attachment"):
        load_hermes_resident_target(hermes_home)


def _seed_standalone_hermes_store(tmp_path: Path):
    """Create a standalone installation_kind=='local' Hermes home with owner_private and group scopes."""
    from scope_recall.adapters.hermes.installation import write_installation_manifest
    from scope_recall.contracts import TrustedContext

    hermes_home = tmp_path / "TEST-standalone-hermes"
    hermes_home.mkdir()
    shared_scope = "scope:TEST-local-group"
    manifest = build_installation_manifest(
        hermes_home, agent_id="TEST-agent", agent_workspace="default", test_mode=True
    )
    owner_scope = manifest.audience_scopes["owner_private"]
    updated_audiences = [
        {**row, "allowed_scope_ids": sorted({*row["allowed_scope_ids"], shared_scope})} for row in manifest.audiences
    ]
    group_row = {
        "platform": "telegram",
        "user_id": "TEST-visitor",
        "chat_type": "group",
        "chat_id": "TEST-group-1",
        "thread_id": "",
        "gateway_session_key": "",
        "agent_workspace": "default",
        "kind": "conversation",
        "allowed_scope_ids": [shared_scope],
        "writable_scope_ids": [shared_scope],
        "capture_scope_id": shared_scope,
    }
    manifest = type(manifest)(
        **{
            **manifest.__dict__,
            "scope_ids": frozenset({*manifest.scope_ids, shared_scope}),
            "audiences": (*updated_audiences, group_row),
        }
    )
    write_installation_manifest(manifest)
    seed_core = MemoryCore(CoreConfig(manifest.to_binding()))
    seed_core.initialize()
    seed_ctx = TrustedContext(manifest.to_binding(), "TEST-seed", manifest.scope_ids, "human_direct")
    seed_core.record_event(
        seed_ctx,
        source_event(content="TEST OrchidVault private passphrase policy is strict.", source_event_key="TEST-lpriv-1"),
        scope_id=owner_scope,
        remaining_seconds=5.0,
    )
    seed_core.record_event(
        seed_ctx,
        source_event(content="TEST OrchidVault public channel schedule is Tuesday.", source_event_key="TEST-lpub-1"),
        scope_id=shared_scope,
        remaining_seconds=5.0,
    )
    return hermes_home, manifest


def test_standalone_local_hermes_store_serves_resident_recall_outside_gateway_gil(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unattached installation_kind=='local' Hermes homes run resident recall and preserve audience isolation."""
    os_home = tmp_path / "TEST-local-os-home"
    os_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USERPROFILE", str(os_home))
    monkeypatch.setenv("HOME", str(os_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: os_home))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "TEST-venv"))

    hermes_home, manifest = _seed_standalone_hermes_store(tmp_path)

    # Without runtime-config.json, standalone local Hermes stays at 0 minutes and starts no resident process.
    assert manifest.installation_kind == "local"
    assert local_endpoint.resident_minutes(hermes_home, "hermes") == 0
    assert store_service.manage(manifest.to_binding()) == "disabled"

    runtime_cfg = {
        "binding": {
            "agent_id": manifest.agent_id,
            "installation_id": manifest.installation_id,
            "data_directory": str(manifest.data_directory),
            "scope_ids": sorted(manifest.scope_ids),
            "test_mode": True,
            "installation_kind": "local",
        },
        "session_id": "TEST-local-worker",
        "allowed_scope_ids": sorted(manifest.scope_ids),
        "actor_origin": "human_direct",
        "owner_id": "TEST-owner",
        "request_seconds": 5.0,
        "drain_seconds": 5.0,
        "max_items": 4,
        "lease_seconds": 10.0,
        "auxiliary": {"external_embedding": False, "external_consolidation": False},
    }
    (manifest.data_directory / "runtime-config.json").write_text(json.dumps(runtime_cfg), encoding="utf-8")

    assert local_endpoint.resident_minutes(hermes_home, "hermes") == 120
    assert store_service.library_minutes(manifest.to_binding()) == 120
    assert store_service.manage(manifest.to_binding()) == "idle"

    started_commands: list[list[str]] = []
    monkeypatch.setattr(
        local_endpoint,
        "start_detached",
        lambda command, cwd: (started_commands.append(command), True)[1],
    )
    probe_provider = ScopeRecallHermesAdapter()
    probe_provider.initialize(
        "TEST-init-mark",
        hermes_home=str(hermes_home),
        agent_identity="TEST-agent",
        agent_workspace="default",
        platform="cli",
    )
    probe_provider.shutdown()
    assert local_endpoint.resident_alive(hermes_home, "hermes").is_file()
    assert store_service.manage(manifest.to_binding()) == "started"
    assert store_service.manage(manifest.to_binding()) == "recent"
    assert len(started_commands) == 1

    resident_code = (
        "import sys\n"
        "from scope_recall.adapters.clients import resident_entry\n"
        "raise SystemExit(resident_entry.main(sys.argv[1:]))\n"
    )
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    resident_proc = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            resident_code,
            "--home",
            str(hermes_home),
            "--host",
            "hermes",
            "--idle-seconds",
            "60",
        ],
        creationflags=creationflags,
    )
    try:
        folder = local_endpoint.endpoints(hermes_home)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not list(folder.glob("*.json")):
            time.sleep(0.05)
        assert list(folder.glob("*.json")), "standalone local resident server did not advertise its endpoint"

        assert resident_cli.main(["status", "--home", str(hermes_home), "--host", "hermes"]) == 0
        status_out = json.loads(capsys.readouterr().out)
        assert status_out["status"] == "ok"
        assert status_out["running"] is True
        assert status_out["resident_recall_minutes"] == 120

        _exercise_resident_providers(hermes_home, monkeypatch)

        assert resident_cli.main(["stop", "--home", str(hermes_home), "--host", "hermes"]) == 0
        stop_out = json.loads(capsys.readouterr().out)
        assert stop_out["running"] is False
    finally:
        if resident_proc.poll() is None:
            resident_proc.kill()
        resident_proc.wait(timeout=10)


def test_hermes_resident_handler_reads_profile_dotenv_without_host_secret_scope(tmp_path, monkeypatch):
    """Resident handler resolves credentials from <hermes_home>/.env without calling agent.secret_scope."""
    from types import SimpleNamespace

    from scope_recall.adapters.clients.hermes_resident import HermesResidentHandler

    hermes_home = tmp_path / "TEST-hermes-dotenv"
    manifest, _core = install_hermes_scope_recall(
        hermes_home,
        agent_id="TEST-agent",
        platform="cli",
        user_id="local",
        agent_workspace="hermes",
        test_mode=False,
    )
    scope_id = next(iter(manifest.scope_ids))
    embed_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents"
    config_payload = {
        "binding": {
            "agent_id": manifest.agent_id,
            "installation_id": manifest.installation_id,
            "data_directory": str(manifest.data_directory),
            "scope_ids": sorted(manifest.scope_ids),
            "test_mode": manifest.test_mode,
        },
        "session_id": "TEST-resident-dotenv",
        "allowed_scope_ids": [scope_id],
        "actor_origin": "human_direct",
        "owner_id": "TEST-worker",
        "request_seconds": 10.0,
        "drain_seconds": 30.0,
        "max_items": 16,
        "lease_seconds": 30.0,
        "auxiliary": {
            "external_embedding": True,
            "external_consolidation": False,
            "installation_dir": str(manifest.data_directory),
            "budget": {
                "batch": "memory",
                "approved_models": ["gemini-embedding-2"],
                "pricing": {"gemini-embedding-2": {"input_usd_per_million": "0", "output_usd_per_million": "0"}},
                "cap_micro_usd": 0,
                "total_call_cap": 100,
                "total_input_cap": 100000,
                "total_output_cap": 0,
            },
            "embedding": {
                "model": "gemini-embedding-2",
                "endpoint": embed_endpoint,
                "dimensions": 3072,
                "dialect": "gemini",
                "credential_env": "SCOPE_RECALL_TEST_RESIDENT_EMBED_KEY",
            },
        },
    }
    (manifest.data_directory / "runtime-config.json").write_text(json.dumps(config_payload), encoding="utf-8")
    (hermes_home / ".env").write_text(
        "SCOPE_RECALL_TEST_RESIDENT_EMBED_KEY=TEST-from-hermes-dotenv\n", encoding="utf-8"
    )
    monkeypatch.delenv("SCOPE_RECALL_TEST_RESIDENT_EMBED_KEY", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "agent.secret_scope",
        SimpleNamespace(get_secret=lambda _name: (_ for _ in ()).throw(AssertionError("must not call secret_scope"))),
    )

    handler = HermesResidentHandler.from_home(hermes_home)
    try:
        handler.ensure_runtime()
        assert handler.runtime_ready
        assert handler._host_runtime._host_adapter == "hermes"
        assert (
            handler._host_runtime.runtime.auxiliary.query_embedding._credential_getter(
                "SCOPE_RECALL_TEST_RESIDENT_EMBED_KEY"
            )
            == "TEST-from-hermes-dotenv"
        )
    finally:
        handler.close()
