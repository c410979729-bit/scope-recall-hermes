"""Public CLI creation and exact configuration edits for a remote Hermes profile."""

from __future__ import annotations

import importlib.util
import json
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from scope_recall._version import __version__
from scope_recall.adapters.clients.remote_client import RemoteClientError
from scope_recall.adapters.hermes import bind_hermes_identity
from scope_recall.adapters.hermes.shared_entries import load_binding_for_home
from scope_recall.adapters.shared_entries import read_shared_payload, write_shared_payload
from scope_recall.maintenance import install_remote, shared
from scope_recall.maintenance.cli import main as maintenance_main
from scope_recall.maintenance.install_common import manifest_version
from scope_recall.maintenance.install_hermes import DIST_HERMES, wrapper_manifest, wrapper_pyproject


def test_attach_remote_profile_shares_owner_grants_and_needs_no_hermes(tmp_path, capsys):
    root, local, remote = (tmp_path / name for name in ("TEST-store", "TEST-local", "TEST-remote"))
    assert shared.main(["init-shared", "--root", str(root), "--test-mode"]) == 0
    assert (
        shared.main(
            [
                "attach",
                "--host",
                "hermes",
                "--instance-root",
                str(local),
                "--root",
                str(root),
                "--entry",
                "local-hermes",
                "--display-name",
                "Local Hermes",
            ]
        )
        == 0
    )
    argv = [
        "attach",
        "--host",
        "hermes",
        "--instance-root",
        str(remote),
        "--root",
        str(root),
        "--entry",
        "remote-hermes",
        "--display-name",
        "Remote Hermes",
        "--remote-profile",
        "TEST-profile",
        "--grants-like",
        "all",
        "--capture-like",
        "local-hermes",
    ]
    assert shared.main(argv) == 0
    original = load_binding_for_home(local)
    added = load_binding_for_home(remote)
    assert added.agent_identity == "TEST-profile"
    assert {row["platform"] for row in added.audiences} == {"cli", "desktop"}
    for row in added.audiences:
        assert set(row["allowed_scope_ids"]) == original.scope_ids
        assert row["capture_scope_id"] in original.scope_ids
    assert shared.main(argv) == 0
    assert load_binding_for_home(remote).audiences == added.audiences
    assert not (remote / "plugins").exists()
    assert not list(remote.glob("*.sqlite3"))


def test_catalog_candidate_preparation_explicitly_approves_the_legacy_local_entry(tmp_path, capsys):
    """The released catalog entry stays inaccessible until the candidate's operator approval runs."""
    runner, store = tmp_path / "runner", tmp_path / "TEST-store"
    home = runner / "hermes-catalog-install/home/.hermes"
    assert shared.main(["init-shared", "--root", str(store), "--test-mode"]) == 0
    assert (
        shared.main(
            [
                "attach",
                "--host",
                "hermes",
                "--instance-root",
                str(home),
                "--root",
                str(store),
                "--entry",
                "hermes",
                "--display-name",
                "TEST catalog",
            ]
        )
        == 0
    )
    legacy = read_shared_payload(store)
    legacy["entries"][0].pop("local_account_id")
    write_shared_payload(store, legacy)
    sent = dict(hermes_home=str(home), platform="cli", agent_identity="default", agent_workspace="hermes")
    before = bind_hermes_identity("TEST-before-approval", **sent)
    assert before.read_only and not before.runtime_audience.allowed_scope_ids
    plugin = home / "plugins/scope-recall"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text('pip_dependencies: ["hermes-scope-recall[vectors]==4.0.4"]\n', encoding="utf-8")
    launcher = home / "bin/hermes"
    launcher.parent.mkdir()
    launcher.write_text("# TEST launcher placeholder; not executed", encoding="utf-8")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    with zipfile.ZipFile(wheels / "hermes_scope_recall-TEST.whl", "w") as wheel:
        wheel.writestr("hermes_scope_recall-TEST.dist-info/METADATA", f"Version: {__version__}\n")
    script = Path(__file__).resolve().parents[2] / ".github/scripts/hermes_remote_install.py"
    spec = importlib.util.spec_from_file_location("TEST_hermes_catalog_exercise", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def run(argv, env, **kwargs):
        # Environment creation is outside this fixture; the explicit management operation is real.
        calls.append(argv)
        if argv[1:3] == ["-m", module.CLI]:
            assert maintenance_main(argv[3:]) == 0
        else:
            assert argv[1:3] in (["-m", "venv"], ["-m", "pip"])
        return ""

    work = tmp_path / "exercise"
    work.mkdir()
    module.prepare_catalog(
        SimpleNamespace(work=work, run=run),
        {
            "RUNNER_TEMP": str(runner),
            "HERMES_HOME": str(home),
            "WHEELS": str(wheels),
        },
    )
    after = bind_hermes_identity("TEST-after-approval", **sent)
    assert after.runtime_audience.includes_owner_private and not after.read_only
    expected = legacy["entries"][0]
    actual = read_shared_payload(store)["entries"][0]
    assert actual.pop("local_account_id")
    assert actual == expected, "approval must preserve identity and existing grants"
    assert any("approve-local-account" in argv for argv in calls)


def test_remote_install_selects_provider_preserves_bytes_and_backs_up(tmp_path):
    home = tmp_path / "TEST-home"
    home.mkdir()
    original = (
        b'\xef\xbb\xbf# TEST keep\r\nmodel: "TEST-model" # keep\r\n'
        b'memory:\r\n  provider: "builtin" # keep\r\n  other: 42\r\n'
    )
    path = home / "config.yaml"
    path.write_bytes(original)
    token = tmp_path / "TEST-token"
    token.write_text("TEST-token", encoding="utf-8")
    config = {
        "host": "hermes",
        "url": "http://127.0.0.1:18765",
        "token_file": token,
        "state_dir": tmp_path / "state",
        "config": tmp_path / "client.json",
    }
    result = install_remote.install(config, home)
    assert path.read_bytes() == original.replace(b'"builtin"', b'"scope-recall"')
    assert any(Path(backup).read_bytes() == original for backup in result["backups"])
    plugin = home / "plugins/scope-recall"
    # A fresh directory takes the catalog's manifest-only form: the pin is in plugin.yaml, nothing else declares it.
    manifest = yaml.safe_load((plugin / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["pip_dependencies"] == [f"hermes-scope-recall[vectors]=={__version__}"]
    assert manifest["version"] == manifest_version()
    assert (plugin / "__init__.py").read_text(encoding="utf-8") == (DIST_HERMES / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert sorted(item.name for item in plugin.iterdir()) == ["__init__.py", "plugin.yaml"]
    assert not install_remote.install(config, home)["backups"]
    assert not list(home.rglob("*.sqlite3"))


def test_remote_install_keeps_an_authored_declaration_form(tmp_path):
    # Hermes' package manager does not rebuild a plugin whose declaration changes form, so a home plugin the local
    # installer wrote (an authored pyproject.toml) keeps that form, pinned to this build.
    home = tmp_path / "TEST-home"
    plugin = home / "plugins/scope-recall"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text(wrapper_pyproject("4.0.0"), encoding="utf-8")
    token = tmp_path / "TEST-token"
    token.write_text("TEST-token", encoding="utf-8")
    config = {
        "host": "hermes",
        "url": "http://127.0.0.1:18765",
        "token_file": token,
        "state_dir": tmp_path / "state",
        "config": tmp_path / "client.json",
    }
    install_remote.install(config, home)
    declaration = tomllib.loads((plugin / "pyproject.toml").read_text(encoding="utf-8"))
    assert declaration["project"]["dependencies"] == [f"hermes-scope-recall[vectors]=={__version__}"]
    assert (plugin / "plugin.yaml").read_text(encoding="utf-8") == wrapper_manifest(
        (DIST_HERMES / "plugin.yaml").read_text(encoding="utf-8")
    )
    assert not (plugin / "register.py").exists()


def test_ambiguous_yaml_is_refused_before_any_remote_install_writes(tmp_path):
    home = tmp_path / "TEST-home"
    home.mkdir()
    path = home / "config.yaml"
    path.write_text("memory: {provider: builtin}\n", encoding="utf-8")
    token = tmp_path / "TEST-token"
    token.write_text("TEST-token", encoding="utf-8")
    config = {
        "host": "hermes",
        "url": "http://127.0.0.1:18765",
        "token_file": token,
        "state_dir": tmp_path / "state",
        "config": tmp_path / "client.json",
    }
    with pytest.raises(RemoteClientError, match="block memory"):
        install_remote.install(config, home)
    assert sorted(item.name for item in home.iterdir()) == ["config.yaml"]
    assert path.read_text(encoding="utf-8") == "memory: {provider: builtin}\n"


@pytest.mark.parametrize("host", ["codex", "hermes"])
def test_grants_like_never_activates_unverified_owner_rows(tmp_path, capsys, host):
    from scope_recall.adapters.shared_entries import read_shared_payload, write_shared_payload
    from scope_recall.maintenance import shared

    root, local, target = (tmp_path / name for name in ("TEST-store", "TEST-local", "TEST-new"))
    assert shared.main(["init-shared", "--root", str(root), "--test-mode"]) == 0
    from dataclasses import replace

    hidden = "scope:TEST-not-owned"
    manifest = shared.fresh_installation(local, read_shared_payload(root))
    scope_ids = manifest.scope_ids | {hidden}
    guest = {
        **manifest.audiences[0],
        "platform": "desktop",
        "user_id": "basic:TEST-guest",
        "allowed_scope_ids": sorted(scope_ids),
        "writable_scope_ids": sorted(scope_ids),
    }
    shared.attach(
        host="hermes",
        instance_root=local,
        root=root,
        entry_id="desk",
        display_name="Desk",
        grants_from=replace(manifest, scope_ids=scope_ids, audiences=(*manifest.audiences, guest)),
    )
    argv = [
        "attach",
        "--host",
        host,
        "--instance-root",
        str(target),
        "--root",
        str(root),
        "--entry",
        "target",
        "--display-name",
        "Target",
        "--grants-like",
        "all",
        "--capture-like",
        "desk",
    ]
    if host == "hermes":
        argv += ["--remote-profile", "TEST-profile"]
    assert shared.main(argv) == 0
    added = next(item for item in read_shared_payload(root)["entries"] if item["entry_id"] == "target")
    assert hidden not in added["scope_ids"]
