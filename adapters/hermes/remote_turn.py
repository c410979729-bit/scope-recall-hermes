"""One independent Hermes turn and its compact, reconstructible host observations."""

from __future__ import annotations

import copy
import hashlib
from datetime import UTC, datetime

from scope_recall.core.private import parse_recording_command

from .boundary import _own_text, _shown_text, _staged_row, extract_user_text
from .remote_wire import facts


def digest(message: object) -> str:
    """Normalize host text blocks for observation identity and repeated body timestamps."""
    return hashlib.sha256(" ".join(extract_user_text(message).split()).encode("utf-8", "replace")).hexdigest()


def _tail(history: object, anchor: dict | None, message: str) -> list:
    """Keep the host rows from the current opener, never the completed conversation prefix."""
    if not isinstance(history, list):
        return []
    if anchor is not None:
        for field in ("platform_message_id", "message_id", "timestamp"):
            if anchor.get(field) is not None:
                found = [
                    i
                    for i, row in enumerate(history)
                    if isinstance(row, dict) and row.get("role") == "user" and row.get(field) == anchor[field]
                ]
                if found:
                    return history[found[-1] :]
    found = [
        i
        for i, row in enumerate(history)
        if isinstance(row, dict)
        and row.get("role") == "user"
        and row.get("display_kind") != "steer"
        and digest(_own_text(row)) == digest(message)
    ]
    return history[found[-1] :] if found else history[-1:]


class RemoteTurn:
    """A start owns this record permanently; later turns never append to its event slots."""

    def __init__(self, binding: dict, switches: list, session: str, message: str, key: str) -> None:
        self.binding = copy.deepcopy(binding)
        self.switches = copy.deepcopy(switches)
        self.session = session
        self.message = extract_user_text(message)
        self.user_at: str | None = None
        self.answer_at: str | None = None
        self.observed: dict[tuple[str, str], str | None] = {}
        self.message_digest = digest(message)
        self.key = key
        self.turn_id = ""
        self.events: list[dict] = []
        self.history: list = []
        self.anchor: dict | None = None
        self.revision = 0
        self.acknowledged = -1
        self.source_fence: dict | None = None

    def _content_times(self, values: dict, at: str) -> dict:
        times = {}
        for field in ("user_message", "message", "user_content"):
            if field in values:
                self.user_at = self.user_at or at
                times[field] = self.user_at
                self.observed.setdefault(("user", digest(values[field])), self.user_at)
        for field in ("assistant_response", "assistant_content"):
            if field in values:
                if field == "assistant_response":
                    self.answer_at = self.answer_at or at
                instant = self.answer_at if field == "assistant_content" and self.answer_at else at
                times[field] = self.observed.setdefault(("assistant", digest(values[field])), instant)
        return times

    def _timed_rows(self, history: object, at: str) -> list:
        rows = facts({"messages": _tail(history, self.anchor, self.message)})["messages"]
        for row in rows:
            if not isinstance(row, dict) or row.get("role") not in {"user", "assistant"}:
                continue
            role = row["role"]
            text = _shown_text(row) if role == "assistant" else _own_text(row)
            key = (role, digest(text))
            moment = row.get("timestamp")
            if key not in self.observed:
                # Retrospective rows need their host timestamp. An undated row cannot acquire a later callback time.
                self.observed[key] = None
                if isinstance(moment, (int, float)) and not isinstance(moment, bool) and moment > 0:
                    try:
                        self.observed[key] = datetime.fromtimestamp(moment, UTC).isoformat()
                    except (OverflowError, OSError, ValueError):
                        row.pop("timestamp", None)
            witnessed = self.observed.get(key)
            if witnessed is not None:
                row["timestamp"] = datetime.fromisoformat(witnessed.replace("Z", "+00:00")).timestamp()
            else:
                row.pop("timestamp", None)
        return rows

    def observe(self, method: str, values: dict, at: str) -> None:
        """Keep original body witnesses, replace API observations, and retain each compression checkpoint."""
        previous_history = self.history
        values = dict(values)
        history = values.pop("conversation_history", values.pop("messages", None))
        if method == "observe_pre_llm":
            self.anchor = _staged_row(history)
        history_field = (
            "messages" if method in {"sync_turn", "on_pre_compress", "on_session_end"} else "conversation_history"
        )
        times = self._content_times(values, at)
        event = {"method": method, "at": at, "values": facts(values), "content_times": times}
        if method in {"observe_pre_llm", "observe_pre_api_request"}:
            event["command"] = (
                parse_recording_command(values.get("user_message")) if not values.get("attachments") else None
            )
        if isinstance(history, list):
            rows = self._timed_rows(history, at)
            if method in {"observe_pre_llm", "on_pre_compress"}:
                event["values"][history_field] = rows
            else:
                self.history = rows
                event["history_field"] = history_field
        old = next((item for item in reversed(self.events) if item["method"] == method), None)
        if old is not None and method in {"on_turn_start", "observe_pre_llm"}:
            return
        if old is not None:
            if {**event, "at": old["at"]} == old and previous_history == self.history:
                return
            if method not in {"on_session_switch", "on_pre_compress"}:
                self.events.remove(old)
        self.events.append(event)
        self.revision += 1
        self.turn_id = str(values.get("turn_id") or self.turn_id)

    def payload(self) -> dict:
        """Serialize constant event slots plus one current history, ready for queue coalescing."""
        ordered = sorted(
            self.events, key=lambda event: {"observe_pre_llm": 0, "on_turn_start": 1}.get(event["method"], 2)
        )
        return copy.deepcopy(
            {
                "hook_event_name": "Hermes",
                "session_id": self.session,
                "binding": self.binding,
                "switches": self.switches,
                "queue_key": self.key,
                "revision": self.revision,
                "events": ordered,
                "history": self.history,
                "source_fence": self.source_fence,
            }
        )
