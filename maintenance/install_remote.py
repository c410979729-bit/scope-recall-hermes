"""The remote client's installer: the plugin that sends a client's hooks and MCP calls to its entry's server, or for
WorkBuddy/dsh those hooks and that server merged into the host's own files (``remote_client install``).  The installed
entry ``scope_recall.adapters.codex.remote_client`` sends ``install`` here and every other command to the client
(``adapters/clients/remote_client.py``)."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

from scope_recall.adapters.clients.remote_client import RemoteClientError, absolute_path, load_client_config
from scope_recall.host_registry import HOOK_TIMEOUTS, HOSTS

from . import install_antigravity as antigravity
from . import install_dsh as dsh
from . import install_workbuddy as workbuddy
from .install_claude_code import SHELL_WORD
from .install_common import (
    CLAUDE_CODE_DESCRIPTION,
    CODEX_LONG_DESCRIPTION,
    CODEX_SHORT_DESCRIPTION,
    REMOTE_STORE_NOTE,
    SKILLS,
    InstallError,
    manifest_version,
    plugin_metadata,
)


def _hook_argv(config: dict[str, Any]) -> list[str]:
    return [
        Path(sys.executable).as_posix(),
        "-I",
        "-B",
        "-m",
        "scope_recall.adapters.codex.remote_client",
        "--config",
        config["config"].as_posix(),
    ]


def hermes_files(config: dict[str, Any], home: Path) -> dict[Path, str]:
    """Install the catalog's wrapper beside a transport binding, refusing to replace a local store binding.

    The wrapper is the catalog's own entry module: its ``register`` selects the remote provider when the profile
    holds a transport binding, so no second registration module exists to diverge (a ``register`` submodule beside
    it would also replace the package's ``register`` function once imported). The core declaration keeps the form
    the plugin directory already has: Hermes' package manager does not rebuild a plugin whose declaration moves
    between ``plugin.yaml``'s ``pip_dependencies`` and an authored ``pyproject.toml``, and disables it instead. A
    directory the catalog installed keeps its pin in ``plugin.yaml``; an existing authored project keeps its
    ``pyproject.toml``. Fresh wrappers use the manifest form in both local and remote installation.
    """
    from scope_recall.adapters.hermes.remote import config_path

    from .hermes_config import provider_configuration
    from .install_hermes import home_plugin_dir, wrapper_files

    if any((home / "scope-recall" / name).exists() for name in ("attachment.json", "installation.json")):
        raise RemoteClientError("Hermes profile already has a local store binding; use a separate profile")
    wrapper = home_plugin_dir(home)
    remote = {key: str(config[key]) for key in ("url", "host", "token_file", "state_dir")}
    try:
        configuration = provider_configuration(home / "config.yaml").decode("utf-8")
        files = wrapper_files(wrapper)
    except InstallError as exc:
        raise RemoteClientError(str(exc)) from exc
    files[home / "config.yaml"] = configuration
    files[config_path(home)] = json.dumps(remote, indent=2) + "\n"
    return files


def plugin_files(config: dict[str, Any], plugin_dir: Path) -> dict[Path, str]:
    """The plugin that sends this client's hooks and MCP calls to its entry's server."""
    host = config["host"]
    profile = HOSTS[host]
    if profile.remote_install == "provider":
        return hermes_files(config, plugin_dir)
    if profile.remote_install != "plugin":
        raise RemoteClientError(f"{host} uses native settings, not a plugin directory")
    author, description = plugin_metadata()
    token = config["token_file"].read_text(encoding="utf-8").strip()
    argv = _hook_argv(config)
    mcp_url = f"{config['url']}/mcp"
    auth = {"Authorization": f"Bearer {token}"}
    skill = SKILLS["scope-recall-memory"].read_text(encoding="utf-8")
    dump = lambda value: json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"  # noqa: E731
    if host == "claude-code":
        unsafe = [part for part in argv if not SHELL_WORD.fullmatch(part)]
        if unsafe:
            raise RemoteClientError(
                "Claude Code runs a hook through a shell: keep the interpreter and client.json "
                f"on paths of ASCII letters, digits and ._-/: only (not {unsafe[0]!r})"
            )
        command = " ".join(argv)
        hooks = {
            "hooks": {
                event: [{"hooks": [{"type": "command", "command": command, "timeout": timeout}]}]
                for event, timeout in sorted(HOOK_TIMEOUTS[host].items())
            }
        }
        return {
            plugin_dir / ".claude-plugin" / "plugin.json": dump(
                {
                    "name": plugin_dir.name,
                    "version": manifest_version(),
                    "author": {"name": author},
                    "description": f"{description}. {CLAUDE_CODE_DESCRIPTION}" + REMOTE_STORE_NOTE,
                    "hooks": "./hooks/hooks.json",
                    "mcpServers": "./.mcp.json",
                }
            ),
            plugin_dir / "hooks" / "hooks.json": dump(hooks),
            plugin_dir / ".mcp.json": dump(
                {"mcpServers": {"scope-recall": {"type": "http", "url": mcp_url, "headers": auth}}}
            ),
            plugin_dir / "skills" / "scope-recall-memory" / "SKILL.md": skill,
        }
    cmd = plugin_dir / "hooks" / "scope-recall-hook.cmd"
    windows = (
        "@echo off\r\nchcp 65001 >nul\r\n" + " ".join(f'"{part}"' for part in argv) + "\r\nexit /b %ERRORLEVEL%\r\n"
    )
    hooks = {
        "hooks": {
            event: [
                {
                    "hooks": [
                        {"type": "command", "command": shlex.join(argv), "commandWindows": str(cmd), "timeout": timeout}
                    ]
                }
            ]
            for event, timeout in sorted(HOOK_TIMEOUTS[host].items())
        }
    }
    return {
        plugin_dir / ".codex-plugin" / "plugin.json": dump(
            {
                "name": plugin_dir.name,
                "version": manifest_version(),
                "author": {"name": author},
                "mcpServers": "./.mcp.json",
                "description": description,
                "interface": {
                    "displayName": "Scope Recall",
                    "shortDescription": CODEX_SHORT_DESCRIPTION,
                    "longDescription": CODEX_LONG_DESCRIPTION + REMOTE_STORE_NOTE,
                    "category": "Productivity",
                    "capabilities": [],
                    "developerName": author,
                },
            }
        ),
        plugin_dir / "hooks" / "hooks.json": dump(hooks),
        cmd: windows,
        plugin_dir / ".mcp.json": dump({"mcpServers": {"scope-recall": {"url": mcp_url, "http_headers": auth}}}),
        plugin_dir / "skills" / "scope-recall-memory" / "SKILL.md": skill,
    }


def workbuddy_files(config: dict[str, Any], home: Path) -> dict[Path, bytes]:
    """WorkBuddy's own settings.json and mcp.json in ``home``, with this client's hooks and MCP server merged in by the
    local installer's rules (``maintenance/install_workbuddy.py``): the files that change, as they are to be written.

    The hooks are this client's when they run it with this ``client.json``; another Scope Recall hook (a local entry's,
    or another client's) is refused, since WorkBuddy would run both.  The server ``scope-recall`` is this client's when
    it names this client's server.
    """
    argv = _hook_argv(config)
    token = config["token_file"].read_text(encoding="utf-8").strip()
    server = {
        "type": "http",
        "url": f"{config['url']}/mcp",
        "headers": {"Authorization": f"Bearer {token}"},
        "description": workbuddy.SERVER_DESCRIPTION,
    }

    def this_client(parts: list[str]) -> bool:
        return "scope_recall.adapters.codex.remote_client" in parts and workbuddy.same_path(
            workbuddy.option(parts, "--config"), config["config"]
        )

    def this_server(value: object) -> bool:
        return isinstance(value, dict) and value.get("url") == server["url"]

    changed = {}
    try:
        command = (
            " ".join(
                [
                    workbuddy.quoted(Path(argv[0]), "interpreter"),
                    *argv[1:-1],
                    workbuddy.quoted(config["config"], "client.json"),
                ]
            )
            + workbuddy.FAIL_OPEN
        )
        for name in (workbuddy.SETTINGS_FILENAME, workbuddy.MCP_FILENAME):
            value, raw = workbuddy.read_config(home / name)
            merged = (
                workbuddy.with_hooks(value, command, HOOK_TIMEOUTS["workbuddy"], this_client)
                if name == workbuddy.SETTINGS_FILENAME
                else workbuddy.with_server(value, server, this_server)
            )
            if raw is None or merged != value:
                changed[home / name] = workbuddy.encode_config(merged, raw)
    except InstallError as exc:
        raise RemoteClientError(str(exc)) from None
    return changed


def _write_private(path: Path, content: str | bytes) -> None:
    """Write ``path`` through a temporary file only this account can read, so the replacement is as private.  dsh's
    patch and WorkBuddy's MCP file carry the entry's token: written under the default umask, a reinstall handed it
    to every local account (POSIX modes; on Windows the profile's ACLs keep the files the account's)."""
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    pending = path.with_name(path.name + ".tmp")
    pending.unlink(missing_ok=True)  # one an interrupted run left behind may be readable: never write into it
    handle = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(handle, "wb") as stream:
        stream.write(data)
    os.replace(pending, path)


def _keep_private(config: dict[str, Any], home: Path) -> None:
    """The native file that carries the entry's token, and every backup the installer kept, readable by this account
    alone whether or not this install changed them, and no temporary copy left beside it: a reinstall repairs an
    installation an older one left open."""
    token_file = home / (dsh.PATCH_FILENAME if config["host"] == "dsh" else workbuddy.MCP_FILENAME)
    if token_file.is_file():
        token_file.chmod(0o600)
    # A temporary file an interrupted install left behind may hold a token that is still valid.
    token_file.with_name(token_file.name + ".tmp").unlink(missing_ok=True)
    backups = config["state_dir"] / "backups"
    if backups.is_dir():
        backups.chmod(0o700)
        for path in backups.rglob("*"):
            path.chmod(0o700 if path.is_dir() else 0o600)


def _hermes_backups(config: dict, home: Path, files: dict[Path, str] | dict[Path, bytes]) -> list[str]:
    """Keep changed profile files in their relative locations so the wrapper and binding can be restored together."""
    kept = config["state_dir"] / "backups" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backups = []
    for path, content in files.items():
        data = content.encode("utf-8") if isinstance(content, str) else content
        if path.is_file() and path.read_bytes() != data:
            target = kept / path.relative_to(home)
            target.parent.mkdir(parents=True, exist_ok=True)
            backups.append(shutil.copy2(path, target))
    return backups


def _antigravity_files(config: dict[str, Any], home: Path, remove: bool) -> dict[Path, bytes]:
    return antigravity.files(home, config["config"], Path(sys.executable), remote=True, remove=remove)


_MERGED_INSTALLERS = {
    "antigravity-settings": _antigravity_files,
    "dsh-patch": lambda config, home, remove: dsh.remote_files(config, home, Path(sys.executable)),
    "workbuddy-settings": lambda config, home, remove: workbuddy_files(config, home),
}


def install(config: dict[str, Any], plugin_dir: Path, *, remove: bool = False) -> dict[str, list[str]]:
    """Write the plugin, or merge WorkBuddy/dsh's native files after copying changed files to ``backups``."""
    layout = HOSTS[config["host"]].remote_install
    merge = layout in _MERGED_INSTALLERS
    if remove and layout != "antigravity-settings":
        raise RemoteClientError("native remote uninstall is supported for Antigravity only")
    if not remove and not config["token_file"].exists():
        raise RemoteClientError("no token yet: run remote_client token first")
    files: dict[Path, str] | dict[Path, bytes]
    backups = []
    if layout is not None and merge:
        if not plugin_dir.is_dir():
            raise RemoteClientError(
                f"{plugin_dir} does not exist: name {config['host']}'s home, or start that host once"
            )
        try:
            files = _MERGED_INSTALLERS[layout](config, plugin_dir, remove)
        except InstallError as exc:
            raise RemoteClientError(str(exc)) from None
        kept = config["state_dir"] / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for path in files:
            if path.is_file():
                kept.mkdir(parents=True, exist_ok=True)
                kept.chmod(0o700)
                backups.append(shutil.copy2(path, kept / path.name))
                Path(backups[-1]).chmod(0o600)
    else:
        files = plugin_files(config, plugin_dir)
    if layout == "provider":
        backups = _hermes_backups(config, plugin_dir, files)
    written = []
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if merge:
            _write_private(path, content)
        else:
            pending = path.with_name(path.name + ".tmp")
            if isinstance(content, bytes):
                pending.write_bytes(content)
            else:
                pending.write_text(content, encoding="utf-8", newline="")
            os.replace(pending, path)
        written.append(str(path))
    if layout in {"workbuddy-settings", "dsh-patch"}:
        _keep_private(config, plugin_dir)
    config["state_dir"].mkdir(parents=True, exist_ok=True)
    return {"written": written, "backups": [str(path) for path in backups]}


def main(argv: list[str]) -> int:
    """``remote_client install --config <client.json> --plugin-dir <dir>``: write the plugin, print what was written."""
    parser = argparse.ArgumentParser(prog="scope-recall-remote-client")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--remove", action="store_true", help="remove only this remote Antigravity entry's settings")
    parsed = parser.parse_args(argv)
    try:
        config = load_client_config(absolute_path(parsed.config, "config"))
        print(
            json.dumps(
                install(config, absolute_path(parsed.plugin_dir, "plugin_dir"), remove=parsed.remove),
                ensure_ascii=False,
            )
        )
        return 0
    except RemoteClientError as exc:
        raise SystemExit(str(exc)) from None
