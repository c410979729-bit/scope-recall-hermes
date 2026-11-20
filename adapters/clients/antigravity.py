"""Antigravity's native wire format translated to the shared hook contract.

Only a complete native user occurrence supplies prompt identity. The latest
input string alone is not evidence, and none of these fields bind a model tool
call to a human authorization. Unknown layouts therefore grant nothing.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from ...core.capture_filters import redact_secret_text
from . import transcript

EVENTS = {"SessionStart": "SessionStart", "PreInvocation": "UserPromptSubmit", "PostInvocation": "Stop", "Stop": "Stop"}


def _prompt(payload: dict[str, Any], session: str) -> transcript.Said | None:
    """Read a bounded complete prefix and match the latest native input, without guessing past a new boundary."""
    path = transcript.antigravity_record_path(payload.get("transcriptPath"), session)
    count, invocation = payload.get("initialNumSteps"), payload.get("invocationNum")
    if path is None or type(count) is not int or count < 1 or type(invocation) is not int or invocation != 0:
        return None
    try:
        if path.stat().st_size > transcript.READ_BYTES:
            return None
        latest = None
        boundary_complete = False
        with path.open("rb") as stream:
            for raw in stream:
                if not raw.endswith(b"\n"):
                    return None
                try:
                    row = json.loads(raw)
                except (ValueError, RecursionError):
                    return None
                if isinstance(row, dict) and type(row.get("step_index")) is int and row["step_index"] == count - 1:
                    # The hook can arrive before the transcript has flushed the new step.
                    boundary_complete = row.get("status") == "DONE"
                if isinstance(row, dict) and row.get("type") == "USER_INPUT":
                    latest = transcript.antigravity_said(row)
                    if latest is not None and row["step_index"] >= count:
                        return None
        return (
            latest if boundary_complete and latest is not None and latest.text == payload.get("lastUserInput") else None
        )
    except OSError:
        return None


def normalize(payload: object, event: str) -> dict[str, Any]:
    """Translate allowlisted fields only; host metadata and tool bodies never enter a remote spool."""
    if not isinstance(payload, dict) or event not in EVENTS:
        return {}
    session = payload.get("conversationId")
    try:
        if type(session) is not str or str(UUID(session)) != session:
            return {}
    except ValueError:
        return {}
    # Delegated tasks are instructions produced by another agent, not a human turn.
    if payload.get("parentConversationId") not in (None, "") or payload.get("rootConversationId") not in (
        None,
        "",
        session,
    ):
        return {}
    if event == "PreInvocation" and (type(payload.get("invocationNum")) is not int or payload["invocationNum"] != 0):
        return {}  # Tool continuations are captured at PostInvocation/Stop without repeating recall.
    result: dict[str, Any] = {"hook_event_name": EVENTS[event], "session_id": session}
    paths = payload.get("workspacePaths")
    if isinstance(paths, list) and len(paths) == 1 and type(paths[0]) is str:
        result["cwd"] = paths[0]
    else:
        result["project_unverified"] = True
    path = transcript.antigravity_record_path(payload.get("transcriptPath"), session)
    if path is not None:
        result["transcript_path"] = str(path)
    if event == "PreInvocation":
        prompt = _prompt(payload, session)
        if prompt is not None:
            result.update(prompt=prompt.text, turn_id=prompt.entry_id, occurred_at=prompt.occurred_at)
    return result


def normalize_bytes(raw: bytes, event: str) -> bytes:
    """Decode only the hook's bounded input, returning an inert object on invalid JSON."""
    if len(raw) > 65536:
        return b"{}"
    try:
        return json.dumps(normalize(json.loads(raw), event), ensure_ascii=True).encode("ascii")
    except (ValueError, UnicodeError, RecursionError):
        return b"{}"


def scrub_remote(body: dict[str, Any]) -> None:
    """Apply Core's exact-text secret filter before transport; never keep a payload spool."""
    payload = body["payload"]
    payload.pop("transcript_path", None)
    if isinstance(payload.get("prompt"), str):
        payload["prompt"] = redact_secret_text(payload["prompt"])[0]
    record = body.get("record")
    if isinstance(record, dict):
        for _offset, entry in record.get("lines", []):
            if entry is not None:
                entry["text"] = redact_secret_text(entry["text"])[0]


def answer(result: dict[str, Any], event: str) -> dict[str, Any]:
    """Deliver memory only as transient system context, never user input or execution control."""
    if event not in {"SessionStart", "PreInvocation"}:
        return {}
    output = result.get("hookSpecificOutput")
    text = output.get("additionalContext") if isinstance(output, dict) else None
    return {"injectSteps": [{"ephemeralMessage": text}]} if isinstance(text, str) and text else {}
