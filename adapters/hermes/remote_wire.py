"""Path-free Hermes host facts shared by the forwarder and the store-side adapter."""

from __future__ import annotations

from typing import Any

from ...core.capture_filters import without_private
from ..clients.boundary import without_lone_surrogates
from ..host_profiles import HERMES_BINDING_FIELDS as BINDING_FIELDS
from ..host_profiles import HERMES_MAX_FACT_BYTES, HERMES_MAX_WIRE_BYTES
from .boundary import _shown_text

EVENT_FIELDS = BINDING_FIELDS | frozenset(
    {
        "session_id",
        "turn_id",
        "turn_number",
        "message",
        "user_message",
        "conversation_history",
        "assistant_response",
        "user_content",
        "assistant_content",
        "messages",
        "query",
        "sender_id",
        "author_id",
        "author_name",
        "author_is_bot",
        "parent_session_id",
        "new_session_id",
        "reset",
        "rewound",
        "reason",
        "status",
        "status_code",
        "outcome",
        "attachments",
    }
)
ROW_FIELDS = frozenset(
    {
        "role",
        "content",
        "type",
        "text",
        "origin",
        "timestamp",
        "created_at",
        "message_id",
        "platform_message_id",
        "author_id",
        "author_name",
        "author_is_bot",
        "sender_id",
        "sender_name",
        "user_id",
        "name",
        "is_bot",
        "metadata",
        "source",
        "kind",
        "id",
        "turn_id",
        "_hermes_meta",
        "codex_message_items",
        "phase",
        "filename",
        "mediaType",
        "media_type",
        "sha256",
        "byte_length",
        "byteLength",
        "reference",
        "artifact_ref",
        "display_kind",
        "api_content",
        "_compressed_summary",
        "_todo_snapshot_synthetic",
        "_empty_recovery_synthetic",
        "_verification_stop_synthetic",
        "_pre_verify_synthetic",
        "_dropped_toolcall_nudge",
        "_kanban_stop_synthetic",
        "_merged_turn_prefix",
        "_absorbed_message_uids",
        "_absorbed_row_ids",
    }
)
METHODS = frozenset(
    {
        "on_turn_start",
        "observe_pre_llm",
        "observe_pre_api_request",
        "observe_post_llm_call",
        "observe_api_request_error",
        "sync_turn",
        "on_session_end",
        "on_session_switch",
        "on_pre_compress",
    }
)
READ_TOOLS = frozenset({"recall", "inspect", "trace", "profile", "entity", "status", "handoff_list"})
META_KEY = "scope-recall/hermes"
MAX_FACT_BYTES = HERMES_MAX_FACT_BYTES
MAX_WIRE_BYTES = HERMES_MAX_WIRE_BYTES


def visible(value: Any) -> Any:
    """Keep only spoken text and host provenance, excluding tool rows and filesystem references."""
    if isinstance(value, str):
        return without_lone_surrogates(without_private(value)[0])
    if isinstance(value, list):
        return [visible(item) for item in value]
    if isinstance(value, dict):
        if value.get("role") in {"tool", "function"}:
            return {"role": "tool"}
        result = {
            key: visible(item) for key, item in value.items() if key in ROW_FIELDS and key != "codex_message_items"
        }
        if isinstance(value.get("codex_message_items"), list):
            result["codex_message_items"] = [
                visible(item)
                for item in value["codex_message_items"]
                if isinstance(item, dict) and item.get("type") == "message" and item.get("phase") == "commentary"
            ]
        if value.get("role") == "assistant":
            result["content"] = scrub_text(_shown_text(value))
        if value.get("path") or value.get("url") or value.get("raw_path") is not None:
            # The existing attachment validator must still refuse a path-bearing observation. Only the presence
            # travels, so the store machine receives no client filesystem reference it could mistake for its own.
            result["path"] = True
        if value.get("tool_calls"):
            # A content-free batch marker preserves the host's steer boundary without sending any invocation.
            result["tool_calls"] = [{}]
        return result
    if value is None or type(value) in (int, float, bool):
        return value
    return None


def facts(values: dict, *, binding: bool = False) -> dict:
    """Select supported facts; workspace paths are never passed as server-side project paths."""
    fields = BINDING_FIELDS if binding else EVENT_FIELDS
    result = {key: visible(value) for key, value in values.items() if key in fields}
    workspace = result.get("agent_workspace")
    if isinstance(workspace, str) and ("/" in workspace or "\\" in workspace or ":" in workspace):
        result.pop("agent_workspace")
    return result


def scrub_text(value: Any) -> Any:
    """Filter private spans at every content exit without changing tool argument field names."""
    if isinstance(value, str):
        return without_lone_surrogates(without_private(value)[0])
    if isinstance(value, list):
        return [scrub_text(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_text(item) for key, item in value.items()}
    return value
