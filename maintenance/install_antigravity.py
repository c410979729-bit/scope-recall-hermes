"""Antigravity settings merged through the shared attached-client installer.

The target is the Antigravity home (.gemini), never a model credential or rules
file. Only the named Scope Recall definition and server are owned here.
"""

from __future__ import annotations

import base64
import copy
import os
import re
import shlex
from pathlib import Path

from . import install_client
from .install_client import (  # noqa: F401 -- the common installer calls this interface
    approve_local_platforms,
    config_path,
    data_dir,
    home_plugin_dir,
    instance_wrapper_files,
    purge_identity,
    unapproved_local_platforms,
    unapproved_owner_logins,
    validate_local_platforms,
    validate_owner_logins,
)
from .install_common import InstallError, InstallPlan
from .install_workbuddy import encode_config, option, read_config, same_path, with_server, without_server, words

HOST = "antigravity"
HOOK_MODULE = "scope_recall.adapters.clients.hook_entry"
MCP_MODULE = "scope_recall.adapters.clients.mcp_entry"
REMOTE_MODULE = "scope_recall.adapters.codex.remote_client"
HOOK_TIMEOUTS = {"SessionStart": 10, "PreInvocation": 15, "PostInvocation": 10, "Stop": 10}
RESTART_NOTE = "reload Antigravity customizations after installation; a disabled hook or MCP server stays disabled"
_ENTRY = install_client.AttachedEntry(HOST, "Antigravity")
_POWERSHELL_FLAGS = "-NoLogo -NoProfile -NonInteractive -EncodedCommand"
_PS_ARG = re.compile(r"'((?:[^'\r\n]|'')*)'")
_PS_EXIT = "; exit $LASTEXITCODE"
validate_options = _ENTRY.validate_options
foreign_instance_entries = _ENTRY.foreign_instance_entries
initialize_instance = _ENTRY.initialize_instance
installation_id = _ENTRY.installation_id
validate_reuse = _ENTRY.validate_reuse


def default_home() -> Path:
    """The global Antigravity configuration home, without inspecting account data."""
    return Path.home() / ".gemini"


def host_config_files(target_plugin_dir: Path) -> tuple[Path, ...]:
    """The two host JSON files covered by the common backup and receipt boundary."""
    return tuple(target_plugin_dir / "config" / name for name in ("hooks.json", "mcp_config.json"))


def planned_files(plan: InstallPlan) -> dict[Path, str | bytes]:
    """No extra rules, credentials, native database or copied plugin are installed."""
    return {}


def command(argv: list[str]) -> str:
    """Transport fixed argv through Antigravity's Windows cmd /c launcher."""
    if os.name != "nt":
        return shlex.join(argv)
    if any(any(char in part for char in '\x00\r\n"‘’‚‛') for part in argv):
        raise InstallError("Antigravity hook paths contain unsupported shell expansion characters")
    shell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    if not shell.is_absolute() or re.search(r'[\s"%!&|<>^()]', str(shell)):
        raise InstallError("Antigravity's Windows system shell path cannot be used without cmd quoting")
    # Go escapes embedded double quotes when passing a command string to cmd.exe;
    # cmd does not decode those escapes. The outer command has no embedded quotes.
    script = "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in argv) + _PS_EXIT
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    launch = f"{shell} {_POWERSHELL_FLAGS} {encoded}"
    if len(launch) > 8000:
        raise InstallError("Antigravity's Windows hook exceeds the cmd command length limit")
    return launch


def _hook_arguments(value: object) -> list[str]:
    """Read only our launcher grammar for ownership; never execute a config to identify it."""
    if not isinstance(value, str) or len(value) > 32768:
        return []
    launcher, separator, encoded = value.rpartition(f" {_POWERSHELL_FLAGS} ")
    if not separator:
        return words(value)  # previous installs remain upgradeable and removable
    if Path(launcher).name.lower() != "powershell.exe":
        return []
    try:
        script = base64.b64decode(encoded, validate=True).decode("utf-16-le")
        args = [item.replace("''", "'") for item in _PS_ARG.findall(script)]
        return args if args and command(args) == value else []
    except (ValueError, InstallError):
        return []


def recognizes(parts: list[str], identity: Path, *, remote: bool = False, mcp: bool = False) -> bool:
    """Identify only this entry's installed command, regardless of interpreter changes."""
    module = REMOTE_MODULE if remote else MCP_MODULE if mcp else HOOK_MODULE
    return (
        option(parts, "-m") == module
        and (remote or option(parts, "--host") == HOST)
        and same_path(option(parts, "--config" if remote else "--home"), identity)
        and (not remote or ("mcp" in parts) == mcp)
    )


def _owned_definition(value, identity, remote):
    if not isinstance(value, dict):
        return False
    if set(value) - {*HOOK_TIMEOUTS, "enabled", "description"}:
        return False
    groups = [value[event] for event in HOOK_TIMEOUTS if event in value]
    if not groups or any(not isinstance(group, list) for group in groups):
        return False
    handlers = [item for group in groups for item in group]
    return bool(handlers) and all(
        isinstance(item, dict) and recognizes(_hook_arguments(item.get("command")), identity, remote=remote)
        for item in handlers
    )


def registration(home: Path, identity: Path) -> str:
    """Observe the enabled local settings, not whether a running App loaded them."""
    hooks_path, mcp_path = host_config_files(home)
    hooks, hooks_raw = read_config(hooks_path)
    mcp, mcp_raw = read_config(mcp_path)
    if hooks_raw is None or mcp_raw is None:
        return "host_config_missing"
    definition = hooks.get("scope-recall")
    servers = mcp.get("mcpServers")
    server = servers.get("scope-recall") if isinstance(servers, dict) else None
    if not isinstance(definition, dict) or not all(definition.get(event) for event in HOOK_TIMEOUTS):
        return "not_selected"
    if not _owned_definition(definition, identity, False) or not isinstance(server, dict):
        return "not_selected"
    args = server.get("args")
    if not isinstance(args, list) or not recognizes(args, identity, mcp=True):
        return "not_selected"
    if definition.get("enabled") is False or server.get("disabled") is True:
        return "not_selected"
    return "registered"


def merged_hooks(value: dict, argv: list[str], identity: Path, *, remote: bool = False, remove: bool = False) -> dict:
    """Preserve all foreign definitions and user switches, refusing a foreign owner of our name."""
    result = copy.deepcopy(value)
    current = value.get("scope-recall")
    if current is not None and not _owned_definition(current, identity, remote):
        raise InstallError("Antigravity hook scope-recall belongs to another installation")
    if remove:
        result.pop("scope-recall", None)
        return result
    own = {key: item for key, item in (current or {}).items() if key not in HOOK_TIMEOUTS}
    own.update(
        {
            event: [{"type": "command", "command": command([*argv, "--event", event]), "timeout": timeout}]
            for event, timeout in HOOK_TIMEOUTS.items()
        }
    )
    result["scope-recall"] = own
    return result


def files(
    home: Path,
    identity: Path,
    python: Path,
    *,
    remote: bool = False,
    env_file: Path | None = None,
    remove: bool = False,
) -> dict[Path, bytes]:
    """Compute only changed settings for local installation or the remote transport shim."""
    where = ["--config", identity.as_posix()] if remote else ["--home", identity.as_posix(), "--host", HOST]
    extra = ["--env-file", env_file.as_posix()] if env_file is not None and not remote else []
    hook = [python.as_posix(), "-I", "-B", "-m", REMOTE_MODULE if remote else HOOK_MODULE, *where, *extra]
    server_args = [
        "-I",
        "-B",
        "-m",
        REMOTE_MODULE if remote else MCP_MODULE,
        *(["mcp"] if remote else []),
        *where,
        *extra,
    ]
    server = {"command": python.as_posix(), "args": server_args}
    changed = {}
    for path in host_config_files(home):
        value, raw = read_config(path)
        if path.name == "hooks.json":
            merged = merged_hooks(value, hook, identity, remote=remote, remove=remove)
        else:

            def ours(candidate):
                return (
                    isinstance(candidate, dict)
                    and isinstance(candidate.get("args"), list)
                    and recognizes(candidate["args"], identity, remote=remote, mcp=True)
                )

            merged = without_server(value, ours) if remove else with_server(value, server, ours)
        if merged != value or (raw is None and not remove):
            changed[path] = encode_config(merged, raw)
    return changed


def merged_file(plan: InstallPlan, path: Path) -> bytes | None:
    """Provide the common installer's merge and backup operation."""
    return files(plan.target_plugin_dir, plan.instance_root, plan.python_executable, env_file=plan.env_file).get(path)


def unmerged_file(instance_root: Path, path: Path) -> bytes | None:
    """Uninstall only this entry's settings, without reverting other host edits."""
    return files(path.parent.parent, instance_root, Path("unused"), remove=True).get(path)
