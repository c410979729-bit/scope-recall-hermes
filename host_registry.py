"""Static host registry shared by protocol validation, entry points and maintenance.

A profile describes what a host can do. It never resolves identity, grants access, imports host software
or performs I/O. Installed names and relative paths remain part of their existing adapter contracts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

HostChoice = str


@dataclass(frozen=True)
class HostProfile:
    """Static host capabilities. Adapters own the execution and file formats."""

    integration: Literal["provider", "hook", "mcp"]
    installer: str | None = None
    remote_install: Literal["provider", "plugin", "workbuddy-settings", "dsh-patch", "antigravity-settings"] | None = (
        None
    )
    container_remote: bool = False
    home: tuple[str, str] | None = None
    discovery_priority: int = 0
    setup_target: tuple[str, ...] = ()
    setup_memory_skill: bool = True
    existing_registration: Literal["catalog", "mcp", "patch"] | None = None
    existing_mcp_file: str = ".mcp.json"
    local_manifest: str | None = None
    local_data: str | None = None
    doctor_registration: Literal["provider", "receipt"] | None = None
    hook_trust: bool = False
    workspace_mapping: bool = False
    native_write_witness: bool = False
    remote_hook_wire: Literal["antigravity"] | None = None
    remote_hook_timeouts: tuple[tuple[str, int], ...] = ()
    local_hook_timeouts: tuple[tuple[str, int], ...] = ()

    session_field: str = "session_id"
    turn_field: str = "turn_id"
    events: frozenset[str] = frozenset()
    empty_answer: str = "{}"
    record_format: str | None = None
    record_inline: bool = False
    record_directory: tuple[str, ...] = ("cwd",)
    record_directory_events: frozenset[str] | None = None
    record_uses_hook_directory: bool = False
    thread_meta: str | None = None
    local_surfaces: frozenset[str] = frozenset()
    hook_observers: tuple[tuple[str, str], ...] = ()
    prompt_prefixes: tuple[str, ...] = ()
    reminder_open: str | None = None
    reminder_close: str = "</system-reminder>"
    reminder_trailing: str = r"(?:[ \t]*\r?\n)*"
    query_block: str | None = None
    suggestion_frame: str | None = None
    suggestion_topic: str | None = None
    suggestion_headings: str | None = None
    subagent_prefix: str | None = None
    subagent_path: str | None = None
    generated_reply_shapes: tuple[tuple[str, type], ...] = ()


HOSTS: Mapping[str, HostProfile] = MappingProxyType(
    {
        "hermes": HostProfile(
            integration="provider",
            installer="install_hermes",
            remote_install="provider",
            container_remote=True,
            home=("HERMES_HOME", ".hermes"),
            discovery_priority=2,
            setup_target=("plugins", "scope-recall"),
            local_manifest="scope-recall/installation.json",
            local_data="scope-recall",
            doctor_registration="provider",
            events=frozenset(
                {
                    "on_turn_start",
                    "pre_llm_call",
                    "pre_api_request",
                    "post_tool_call",
                    "api_request_error",
                    "post_llm_call",
                    "sync_turn",
                    "prefetch",
                    "on_session_end",
                    "on_session_switch",
                    "on_pre_compress",
                }
            ),
            hook_observers=(
                ("pre_llm_call", "observe_pre_llm"),
                ("pre_api_request", "observe_pre_api_request"),
                ("api_request_error", "observe_api_request_error"),
                ("post_tool_call", "observe_post_tool_call"),
                ("post_llm_call", "observe_post_llm_call"),
            ),
            local_surfaces=frozenset({"desktop", "tui"}),
        ),
        "codex": HostProfile(
            integration="hook",
            installer="install_codex",
            remote_install="plugin",
            container_remote=True,
            home=("CODEX_HOME", ".codex"),
            discovery_priority=1,
            setup_target=("skills", "scope-recall"),
            existing_registration="catalog",
            local_manifest="codex-installation.json",
            local_data="data",
            hook_trust=True,
            workspace_mapping=True,
            native_write_witness=True,
            local_hook_timeouts=(("PostToolUse", 2), ("PreModelRequest", 5), ("PreToolUse", 2)),
            remote_hook_timeouts=(
                ("SessionStart", 5),
                ("UserPromptSubmit", 15),
                ("Stop", 10),
                ("Interrupt", 3),
                ("SessionEnd", 3),
            ),
            events=frozenset(
                {
                    "SessionStart",
                    "UserPromptSubmit",
                    "PreModelRequest",
                    "PreToolUse",
                    "Stop",
                    "PostToolUse",
                    "Interrupt",
                    "SessionEnd",
                }
            ),
            generated_reply_shapes=(("suggestions", list), ("suggestion", str)),
            record_format="codex",
            record_directory=("payload", "cwd"),
            record_directory_events=frozenset({"session_meta", "turn_context"}),
            thread_meta="threadId",
            suggestion_frame=(
                r"This is an app-generated request for a suggested next user message\.\s+"
                r"The user did not write this message\."
            ),
            suggestion_topic=r"\bhyperpersonali[sz]ed\s+suggestions?\b",
            suggestion_headings=r"^#{1,2}[ \t]*(?:overview|rules|examples|bad examples|response format)[ \t]*$",
        ),
        "claude-code": HostProfile(
            integration="hook",
            installer="install_claude_code",
            remote_install="plugin",
            container_remote=True,
            home=("CLAUDE_CONFIG_DIR", ".claude"),
            discovery_priority=0,
            setup_target=("skills", "scope-recall"),
            existing_registration="mcp",
            remote_hook_timeouts=(("SessionStart", 5), ("UserPromptSubmit", 15), ("Stop", 10), ("SessionEnd", 10)),
            turn_field="prompt_id",
            events=frozenset({"SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"}),
            record_format="claude-code",
            prompt_prefixes=("<task-notification>",),
            reminder_open=r"^[ \t]*<system-reminder\b[^>\n]*>",
        ),
        "workbuddy": HostProfile(
            integration="hook",
            installer="install_workbuddy",
            remote_install="workbuddy-settings",
            container_remote=True,
            home=("CODEBUDDY_CONFIG_DIR", ".workbuddy"),
            discovery_priority=3,
            existing_registration="mcp",
            existing_mcp_file="mcp.json",
            remote_hook_timeouts=(("UserPromptSubmit", 15), ("Stop", 10), ("SessionEnd", 10)),
            turn_field="generation_id",
            events=frozenset({"UserPromptSubmit", "Stop", "SessionEnd"}),
            empty_answer="",
            record_format="workbuddy",
            prompt_prefixes=("<task-notification>", "Stop hook feedback:"),
            reminder_open=r"<system-reminder\b[^>]*>",
            reminder_trailing=r"\s*",
            query_block=r"<user_query>(.*?)</user_query>",
            subagent_prefix="agent-",
            subagent_path=r"[\\/]subagents[\\/]",
        ),
        "dsh": HostProfile(
            integration="hook",
            installer="install_dsh",
            remote_install="dsh-patch",
            container_remote=True,
            home=("DSH_HOME", ".dsh"),
            discovery_priority=4,
            existing_registration="patch",
            remote_hook_timeouts=(("UserPromptSubmit", 9), ("Stop", 20)),
            events=frozenset({"UserPromptSubmit", "Stop"}),
            record_format="dsh",
            record_inline=True,
            record_uses_hook_directory=True,
        ),
        "antigravity": HostProfile(
            integration="hook",
            installer="install_antigravity",
            remote_install="antigravity-settings",
            home=("SCOPE_RECALL_ANTIGRAVITY_HOME", ".gemini"),
            discovery_priority=5,
            setup_memory_skill=False,
            existing_registration="mcp",
            existing_mcp_file="config/mcp_config.json",
            doctor_registration="receipt",
            remote_hook_wire="antigravity",
            remote_hook_timeouts=(("SessionStart", 10), ("UserPromptSubmit", 15), ("Stop", 10)),
            events=frozenset({"SessionStart", "UserPromptSubmit", "Stop"}),
            record_format="antigravity",
            record_uses_hook_directory=True,
        ),
        "cherry-studio": HostProfile(integration="mcp"),
    }
)

# These views keep the installed command, manifest and runtime-config vocabulary in sync.
ENTRY_HOSTS = tuple(HOSTS)
HOOK_CLIENT_HOSTS = tuple(host for host, profile in HOSTS.items() if profile.integration == "hook")
RESIDENT_HOSTS = ("hermes", *HOOK_CLIENT_HOSTS)
READ_ONLY_CLIENT_HOSTS = tuple(host for host, profile in HOSTS.items() if profile.integration == "mcp")
CLIENT_HOSTS = (*HOOK_CLIENT_HOSTS, *READ_ONLY_CLIENT_HOSTS)
INSTALL_HOSTS = tuple(host for host, profile in HOSTS.items() if profile.installer is not None)
LOCAL_STORE_HOSTS = tuple(host for host, profile in HOSTS.items() if profile.local_manifest is not None)
REMOTE_HOSTS = tuple(host for host, profile in HOSTS.items() if profile.remote_install is not None)
CONTAINER_REMOTE_HOSTS = tuple(host for host, profile in HOSTS.items() if profile.container_remote)
REMOTE_RECORD_HOSTS = frozenset(
    host for host in REMOTE_HOSTS if HOSTS[host].record_format is not None and not HOSTS[host].record_inline
)
HOOK_TIMEOUTS = {
    host: dict(profile.remote_hook_timeouts) for host, profile in HOSTS.items() if profile.remote_hook_timeouts
}


def remote_hook_wire(host: str | None) -> str | None:
    """Return the protocol for a known remote hook. A missing host uses the standard envelope."""
    profile = HOSTS.get(host) if host is not None else None
    return profile.remote_hook_wire if profile is not None else None
