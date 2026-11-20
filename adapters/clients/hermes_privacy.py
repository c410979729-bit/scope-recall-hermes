"""Content-free Hermes privacy metadata and filtering by original observation times."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta
from time import get_clock_info

from ...contracts import ContractError, utc_timestamp
from ..host_profiles import HERMES_BINDING_FIELDS
from .hermes_queue import encode

_CONTENT = {
    "user_message",
    "message",
    "user_content",
    "assistant_content",
    "assistant_response",
    "attachments",
    "query",
}
_ROUTE = HERMES_BINDING_FIELDS | {
    "session_id",
    "new_session_id",
    "reset",
    "rewound",
    "reason",
    "turn_id",
    "turn_number",
}
_HISTORY_TICK = timedelta(seconds=get_clock_info("time").resolution)


def _observation_time(at: str | None, *, historical: bool) -> str | dict[str, str] | None:
    """Compare coarse history with privacy switches conservatively on both edges of an interval."""
    if at is None:
        return None
    try:
        point = utc_timestamp(at)
        moment = datetime.fromisoformat(point)
        # A copied whole-second observation has no finer precision witness,
        # even if the replay clock is newer or the callback has the same timestamp.
        tick = max(_HISTORY_TICK, timedelta(seconds=1)) if historical and not moment.microsecond else _HISTORY_TICK
        if tick <= timedelta(microseconds=1):
            return point
        return {
            "earliest": (moment - tick).isoformat(timespec="microseconds"),
            "latest": (moment + tick).isoformat(timespec="microseconds"),
        }
    except (ContractError, OverflowError):
        return None


def _times(rows: list) -> list:
    moments = []
    for row in rows:
        stamp = row.get("timestamp") if isinstance(row, dict) else None
        try:
            moment = (
                datetime.fromtimestamp(stamp, UTC).isoformat()
                if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and stamp > 0
                else None
            )
        except (OverflowError, OSError, ValueError):
            moment = None
        moments.append(_observation_time(moment, historical=True))
    return moments


def metadata(payload: dict) -> dict:
    """Describe routes and original row instants without copying any spoken content or query."""
    value = {key: copy.deepcopy(payload[key]) for key in ("binding", "session_id") if key in payload}
    value["switches"] = [
        {
            "method": event["method"],
            "at": event["at"],
            "values": {key: item for key, item in event["values"].items() if key in _ROUTE},
        }
        for event in payload.get("switches", [])
    ]
    value["events"] = [
        {
            "method": event["method"],
            "at": event["at"],
            "values": {key: item for key, item in event["values"].items() if key in _ROUTE},
        }
        for event in payload.get("events", [])
    ]
    histories = {"history": _times(payload.get("history", []))}
    for i, event in enumerate(payload.get("events", [])):
        fields = sorted(_CONTENT & event["values"].keys())
        original = event.get("content_times", {})
        histories[f"{i}:content"] = [
            _observation_time(original.get(field, event["at"]), historical=field in original) for field in fields
        ]
        for field in ("messages", "conversation_history"):
            if isinstance(event["values"].get(field), list):
                moments = _times(event["values"][field])
                # The staged opener is a current delivery witnessed by this hook, unlike historical undated rows.
                rows = event["values"][field]
                if (
                    rows
                    and isinstance(rows[-1], dict)
                    and rows[-1].get("role") == "user"
                    and rows[-1].get("content") == event["values"].get("user_message")
                ):
                    if moments[-1] is None:
                        moments[-1] = _observation_time(event["at"], historical=False)
                histories[f"{i}:{field}"] = moments
    value["history_times"] = histories
    return {"hermes": value}


def _masked(row: dict) -> dict:
    # A denied row carries only its location and role, never nested metadata or alternative text fields.
    return {
        key: row[key]
        for key in ("role", "timestamp", "platform_message_id", "message_id", "display_kind")
        if key in row
    }


def eligible(payload: dict, answer: dict) -> dict | None:
    """Retain structural markers but exclude every body that policy could not verify."""
    if answer.get("terminal"):
        return {**payload, "_refused": True}
    decisions = answer.get("decisions")
    histories = answer.get("history_decisions")
    if (
        not isinstance(decisions, list)
        or len(decisions) != len(payload.get("events", []))
        or any(type(item) is not bool for item in decisions)
        or not isinstance(histories, dict)
    ):
        return None
    if ("read" in payload or payload.get("tool_read")) and answer.get("recall_blocked"):
        return None
    result = copy.deepcopy(payload)
    result["history"] = _filter_rows(result.get("history", []), histories.get("history"))
    if result["history"] is None:
        return None
    for i, (event, blocked) in enumerate(zip(result.get("events", []), decisions, strict=True)):
        values = event["values"]
        for field in ("messages", "conversation_history"):
            if field in values:
                rows = _filter_rows(values[field], histories.get(f"{i}:{field}"))
                if rows is None:
                    return None
                values[field] = rows
        fields = sorted(_CONTENT & values.keys())
        denied = histories.get(f"{i}:content", [blocked] * len(fields))
        if not isinstance(denied, list) or len(denied) != len(fields) or any(type(item) is not bool for item in denied):
            return None
        for key, refuse in zip(fields, denied, strict=True):
            if refuse:
                values[key] = [] if isinstance(values[key], list) else ""
        if blocked and event.get("command") in {"pause", "resume"}:
            event["values"] = {key: value for key, value in values.items() if key in _ROUTE}
            event["values"]["user_message"] = "scope-recall " + event["command"]
    return result


def _filter_rows(rows: list, decisions: object) -> list | None:
    if (
        not isinstance(decisions, list)
        or len(rows) != len(decisions)
        or any(type(item) is not bool for item in decisions)
    ):
        return None
    return [
        (_masked(row) if isinstance(row, dict) else {}) if blocked else row
        for row, blocked in zip(rows, decisions, strict=True)
    ]


class MetadataTooLarge(ValueError):
    """The fixed routing facts alone cannot fit the existing privacy endpoint."""


def batches(payload: dict, limit: int):
    """Split time arrays by actual UTF-8 envelope bytes, retaining routes in every request."""
    value = metadata(payload)["hermes"]
    times = value.pop("history_times")
    empty = {key: [] for key, moments in times.items() if not moments}
    common = {"hermes": {**value, "history_times": empty}}
    if len(encode(common)) > limit:
        raise MetadataTooLarge("Hermes routing metadata exceeds HTTP bound")
    pending = dict(empty)
    for key, moments in times.items():
        start = 0
        while start < len(moments):
            low, high = 0, len(moments) - start
            while low < high:
                size = (low + high + 1) // 2
                candidate = {"hermes": {**value, "history_times": {**pending, key: moments[start : start + size]}}}
                if len(encode(candidate)) <= limit:
                    low = size
                else:
                    high = size - 1
            if low == 0:
                if pending == empty:
                    raise MetadataTooLarge("Hermes time metadata exceeds HTTP bound")
                yield {"hermes": {**value, "history_times": pending}}
                pending = dict(empty)
                continue
            pending[key] = moments[start : start + low]
            start += low
            if start < len(moments):
                yield {"hermes": {**value, "history_times": pending}}
                pending = dict(empty)
    yield {"hermes": {**value, "history_times": pending}}
