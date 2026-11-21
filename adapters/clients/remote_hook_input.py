"""Prepare local hook payloads and session records before sending them to a remote store."""

from __future__ import annotations

import json
from typing import Any, cast

from scope_recall.core.capture_filters import without_private
from scope_recall.core.private import normalize_project_path

from ...contracts import ContractError
from . import transcript
from .boundary import IN_CONTEXT_FIELD, filter_capture_reply, without_lone_surrogates
from .session_marks import clear_compaction, compaction_marker, mark_compaction, rebase_compaction

#: A Stop sends at most this much of the record; a long backlog goes over several turns.
RECORD_READ_BYTES = 2 * 1024 * 1024


def _record_part(
    config: dict[str, Any], payload: dict[str, Any]
) -> tuple[dict[str, Any], transcript.Cursor, transcript.RecordPage] | None:
    """What this Stop sends of the session record, and the cursor to move on success."""
    session_id = payload.get("session_id")
    if type(session_id) is not str or not session_id.strip():
        return None
    record = transcript.host_record_path(config["host"], payload, session_id.strip())
    if record is None:
        return None
    cursor = transcript.Cursor(config["state_dir"], session_id.strip(), record)
    start = cursor.load()
    try:
        lines = transcript.read(
            record,
            start,
            limit=RECORD_READ_BYTES,
            rows=transcript.PARSERS[config["host"]],
            directory=cursor.directory(),
        )
    except OSError:
        return None
    if not lines:
        return None
    wire = [[end, transcript.said_to_wire(said)] for end, said in lines if said is not None]
    if not wire or wire[-1][0] != lines[-1][0]:
        wire.append([lines[-1][0], None])
    return {"start": start, "lines": wire}, cursor, lines


def _error_reply(payload: dict[str, Any]) -> bool:
    """Whether a WorkBuddy Stop's reply is an error its record here marks (``transcript.workbuddy_error_reply``)."""
    reply, session_id = payload.get("last_assistant_message"), payload.get("session_id")
    if type(reply) is not str or type(session_id) is not str or not session_id.strip():
        return False
    try:
        record = transcript.workbuddy_record_path(
            payload.get("transcript_path"), session_id.strip(), record_id=payload.get("agent_id")
        )
    except OSError:
        return False
    return record is not None and transcript.workbuddy_error_reply(record, reply)


def _with_context_since(payload: dict[str, Any]) -> dict[str, Any]:
    """A Claude Code prompt's payload with since when the model here still holds the session's own turns, read from
    the session record on this machine (``transcript.context_since``): the server opens no path a request names, and
    without it recalls them again."""
    kept = {key: value for key, value in payload.items() if key != IN_CONTEXT_FIELD}
    session_id = payload.get("session_id")
    if type(session_id) is not str or not session_id.strip():
        return kept
    record = transcript.record_path(payload.get("transcript_path"), session_id.strip())
    since = transcript.context_since(record) if record is not None else None
    return kept if since is None else {**kept, IN_CONTEXT_FIELD: since}


def _visible_value(value: Any) -> Any:
    """Apply the shared inline filter to nested capturable text, never to identity fields."""
    if isinstance(value, str):
        return without_private(value)[0]
    if isinstance(value, list):
        return [_visible_value(item) for item in value]
    if isinstance(value, dict):
        return {_visible_value(key): _visible_value(item) for key, item in value.items()}
    return value


def _private_span_count(value: Any) -> int:
    """Count the shared filter's removals across a wire field before it is copied."""
    if isinstance(value, str):
        return without_private(value)[1]
    if isinstance(value, list):
        return sum(_private_span_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_private_span_count(key) + _private_span_count(item) for key, item in value.items())
    return 0


def _visible_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip inline private text before the same envelope reaches transport or durable spool."""
    result = dict(payload)
    event_name = result.get("hook_event_name")
    event_name = event_name if isinstance(event_name, str) else ""
    fields = {
        "UserPromptSubmit": ("prompt",),
        "Stop": ("last_assistant_message",),
        "PostToolUse": ("tool_input", "tool_response"),
    }.get(event_name, ("prompt",))
    count = sum(_private_span_count(result.get(field)) for field in fields)
    result.pop("scope_recall_private_spans_removed", None)
    if count:
        result["scope_recall_private_spans_removed"] = count
    for field in ("prompt", "last_assistant_message", "tool_input", "tool_response"):
        if field in result:
            result[field] = _visible_value(result[field])
    if isinstance(result.get("record"), list):
        rows = []
        for row in result["record"]:
            if isinstance(row, dict) and isinstance(row.get("text"), str):
                visible, removed = without_private(row["text"])
                previous = row.get("private_spans_removed", 0)
                previous = previous if type(previous) is int and 0 <= previous <= 1000000 else 0
                row = cast(dict[str, Any], dict(row, text=visible))
                if removed + previous:
                    row["private_spans_removed"] = removed + previous
            rows.append(row)
        result["record"] = rows
    return result


def _hook_input(
    raw: bytes, *, host: str | None = None, event: str | None = None
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    """Decode one event and resolve local text filtering before any transport access."""
    if host == "antigravity" and event is not None:
        from .antigravity import normalize_bytes

        raw = normalize_bytes(raw, event)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError):
        return None, {}, None
    if not isinstance(payload, dict):
        return None, {}, None
    original = payload
    # JavaScript can emit lone surrogates; replacement keeps JSON text valid UTF-8.
    payload = cast(dict[str, Any], without_lone_surrogates(payload))
    filtered = filter_capture_reply(payload)
    if filtered is not None:
        return None, filtered, None
    return _visible_payload(payload), {}, original


def _pending_compaction(
    config: dict[str, Any], payload: dict[str, Any], observed_at: str
) -> tuple[str | None, dict | None]:
    """Record this client's compaction before transport and return its pending watermark."""
    session = payload.get("session_id")
    if config["host"] not in {"claude-code", "codex"} or type(session) is not str or not session.strip():
        return None, None
    if payload.get("hook_event_name") == "SessionStart" and payload.get("source") == "compact":
        mark_compaction(config, session, observed_at)
    return session, compaction_marker(config, session)


def _ack_compaction(config, session: str | None, compact: dict | None, answer: dict) -> None:
    """Retire only the client mark the server confirmed."""
    if compact is not None and session is not None:
        watermark = answer.get("compaction_watermark")
        if type(watermark) is int:
            rebase_compaction(config, session, compact["epoch"], compact["ordinal"], watermark)
        if answer.get("compaction_ack") == compact["ordinal"]:
            clear_compaction(config, session, compact["ordinal"], compact["epoch"], watermark)


def _normalize_hook_project(payload):
    """Normalize the client's project path while preserving an unverified identity marker."""
    cwd = payload.get("cwd")
    if isinstance(cwd, str):
        try:
            payload["cwd"] = normalize_project_path(cwd, require_identity=True)
        except ContractError:
            payload["cwd"] = normalize_project_path(cwd, resolve_links=False)
            payload["project_unverified"] = True
