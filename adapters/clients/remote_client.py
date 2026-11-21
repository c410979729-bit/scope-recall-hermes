"""Forward authenticated host hooks and resume eligible queued captures.

The server checks each message against privacy rules at its original time. After a client learns that a store
has private intervals, it checks event metadata before saving or sending further captures. Existing captures
wait until the checks succeed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.client
import itertools
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.parse
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scope_recall.core.file_lock import advisory_file_lock
from scope_recall.core.private import notice_for_mode, parse_recording_command

from ...host_registry import HOOK_TIMEOUTS as HOOK_TIMEOUTS
from ...host_registry import REMOTE_HOSTS as HOSTS
from ...host_registry import REMOTE_RECORD_HOSTS as _RECORD_HOSTS
from ...host_registry import remote_hook_wire
from ...runtime.remote_clock import event_times as _private_times
from ...runtime.remote_clock import private_metadata, prompt_times
from ..host_profiles import HERMES_MAX_FACT_BYTES, HERMES_MAX_WIRE_BYTES, agent_run
from . import hermes_privacy, hermes_queue, transcript
from .boundary import (
    EMPTY_ANSWER,
    IN_CONTEXT_FIELD,
    is_workbuddy_agent_run,
)
from .hook_answer import command_block
from .remote_hook_input import RECORD_READ_BYTES as RECORD_READ_BYTES
from .remote_hook_input import (
    _ack_compaction,
    _error_reply,
    _hook_input,
    _normalize_hook_project,
    _pending_compaction,
    _record_part,
    _with_context_since,
)
from .remote_hook_input import _visible_payload as _visible_payload
from .session_marks import compaction_marker

#: The part of each wait the request may use; the interpreter's start and the answer take the rest.
_REQUEST_SHARE = 0.8

#: Codex hooks kept for later, oldest first; past this many the oldest is dropped.
SPOOL_LIMIT = 256
#: How long a flush of the spool may run, in its own process after a hook that got through.
FLUSH_SECONDS = 60.0
#: A connection to the server not open after this long is given up (over a tailnet relay one opens in about
#: a second), so a prompt is not held for the hook's whole wait while the server is away.
CONNECT_SECONDS = 3.0
#: After a connection could not be opened, hooks do not try for this long: a client whose server is away is
#: held up once in this time, not at every prompt.
AWAY_SECONDS = 60.0
#: The log of requests that did not get through is kept to about this size (one older copy is kept).
LOG_BYTES = 256 * 1024
_SPOOLED = frozenset({"UserPromptSubmit", "Stop", "SessionEnd", "Interrupt"})
#: The server refused the request itself (malformed, too large): sent again it is refused again, so it is
#: dropped and said in the log rather than kept at the head of the spool, where it would stop every later flush.
_REFUSED_FOR_GOOD = frozenset({400, 413})
_MAX_STDIN = 65536


class RemoteClientError(ValueError):
    pass


def absolute_path(value: object, field: str) -> Path:
    if type(value) is not str or not value.strip():
        raise RemoteClientError(f"{field} must be an absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RemoteClientError(f"{field} must be an absolute path")
    return path


def load_client_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RemoteClientError(f"unreadable client config {path}") from exc
    if not isinstance(raw, dict) or raw.get("host") not in HOSTS:
        raise RemoteClientError(f"client config needs host: {', '.join(HOSTS)}")
    url = raw.get("url")
    if (
        type(url) is not str
        or urllib.parse.urlsplit(url).scheme not in ("http", "https")
        or not urllib.parse.urlsplit(url).hostname
    ):
        raise RemoteClientError("client config needs the server's url")
    return {
        "url": url.rstrip("/"),
        "host": raw["host"],
        "token_file": absolute_path(raw.get("token_file"), "token_file"),
        "state_dir": absolute_path(raw.get("state_dir"), "state_dir"),
        "config": path,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _log(config: dict[str, Any], line: str) -> None:
    """One line in the state folder's log: a request that did not get through, or what the spool did."""
    path = config["state_dir"] / "remote-client.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > LOG_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_now()} {line}\n")
    except OSError:
        pass


def _away_marker(config: dict[str, Any]) -> Path:
    return config["state_dir"] / "server-away"


def _server_away(config: dict[str, Any]) -> bool:
    """Whether a connection failed less than ``AWAY_SECONDS`` ago."""
    try:
        return time.time() - _away_marker(config).stat().st_mtime < AWAY_SECONDS
    except OSError:
        return False


def _connection_failed(config: dict[str, Any], event: object, exc: OSError) -> None:
    """Record the transport cooldown without retaining any hook content."""
    _log(config, f"{event}: no connection ({type(exc).__name__}); not trying for {AWAY_SECONDS:.0f} s")
    try:
        _away_marker(config).parent.mkdir(parents=True, exist_ok=True)
        _away_marker(config).write_text(_now(), encoding="utf-8")
    except OSError:
        pass


class HermesSizeError(ValueError):
    """A snapshot exceeds an encoded or decoded limit and cannot be retried unchanged."""


def _encoded_request(config: dict, body: dict, endpoint: str) -> tuple[bytes, dict]:
    raw = (
        hermes_queue.encode(body)
        if config["host"] == "hermes"
        else json.dumps(body, ensure_ascii=False).encode("utf-8")
    )
    headers = {}
    if config["host"] == "hermes" and endpoint == "/hook":
        if len(raw) > HERMES_MAX_FACT_BYTES:
            raise HermesSizeError("Hermes decoded envelope exceeds HTTP bound")
        if len(raw) > 512 * 1024:
            raw = gzip.compress(raw, mtime=0)
            headers["Content-Encoding"] = "gzip"
        if len(raw) > HERMES_MAX_WIRE_BYTES:
            raise HermesSizeError("Hermes encoded envelope exceeds HTTP bound")
    return raw, headers


def _refused_response(config: dict, event, status: int):
    _log(config, f"{event}: HTTP {status}, refused for good: not kept")
    return {"refused": status}


def _post(
    config: dict[str, Any], body: dict[str, Any], timeout: float, *, endpoint: str = "/hook"
) -> dict[str, Any] | None:
    """The server's answer, or None when it could not be reached in time or refused.

    The connection is opened to the server itself, never through a proxy: HTTP_PROXY and the system proxy
    are for the internet, and one on 127.0.0.1 would take the request away from the private network.
    """
    event = "mcp" if endpoint == "/mcp" else (body.get("payload") or {}).get("hook_event_name")
    try:
        token = config["token_file"].read_text(encoding="utf-8").strip()
    except OSError as exc:
        _log(config, f"{event}: no token to send ({type(exc).__name__})")
        return None
    url = urllib.parse.urlsplit(config["url"])
    kind = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    timeout = max(0.2, timeout)
    until = time.monotonic() + timeout
    connection = kind(url.hostname, url.port, timeout=min(CONNECT_SECONDS, timeout))
    try:
        try:
            connection.connect()
        except OSError as exc:
            _connection_failed(config, event, exc)
            return None
        connection.sock.settimeout(max(0.2, until - time.monotonic()))
        if (
            endpoint == "/hook"
            and config["host"] != "hermes"
            and _private_rules(config)
            and not direct_owner_command(body, config["host"])
        ):
            state = _private_check(config, body, until - time.monotonic())
            if state is None:
                return None
            body, state = eligible_body(body, state, config.get("host"))
            if _private_drop(state, event):
                return None
            if state["capture_blocked"]:
                body = _recall_only(body, config.get("host"))
        encoded, encoding_headers = _encoded_request(config, body, endpoint)
        connection.request(
            "POST",
            f"{url.path}{endpoint}",
            body=encoded,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "Accept": "application/json, text/event-stream",
                **encoding_headers,
            },
        )
        response = connection.getresponse()
        raw = response.read()
        if response.status in _REFUSED_FOR_GOOD:
            return _refused_response(config, event, response.status)
        if response.status != 200:
            # Streamable HTTP acknowledges notifications with an empty 202 response.
            acknowledged = endpoint == "/mcp" and "id" not in body and response.status == 202 and not raw
            if not acknowledged:
                _log(config, f"{event}: HTTP {response.status}")
            return None
        decoded = raw.decode("utf-8")
        data = [line[5:].strip() for line in decoded.splitlines() if line.startswith("data:")]
        answer = json.loads(data[-1] if data else decoded)
    except HermesSizeError:
        _log(config, f"{event}: Hermes envelope cannot fit HTTP bounds")
        answer = {"refused": 413}
    except (OSError, http.client.HTTPException, ValueError) as exc:
        _log(config, f"{event}: no answer in {timeout:.1f} s ({type(exc).__name__})")
        return None
    finally:
        connection.close()
    try:
        _away_marker(config).unlink(missing_ok=True)
    except OSError:
        pass
    if isinstance(answer, dict):
        _remember_private_rules(config, answer)
    return answer if isinstance(answer, dict) else None


def _hermes_checked(config: dict, payload: dict, seconds: float) -> dict | None:
    """Verify original event and row times before any content request, including reads and MCP."""
    until = time.monotonic() + seconds
    combined: dict[str, Any] = {"decisions": [False] * len(payload.get("events", [])), "history_decisions": {}}
    try:
        for body in hermes_privacy.batches(payload, 65536):
            if time.monotonic() >= until:
                return None
            answer = _post(config, body, until - time.monotonic(), endpoint="/private/check")
            if answer is None:
                return None
            if answer.get("refused") == 413:
                return {**payload, "_oversized": True}
            if answer.get("terminal"):
                return hermes_privacy.eligible(payload, answer)
            if not isinstance(answer.get("decisions"), list) or not isinstance(answer.get("history_decisions"), dict):
                return None
            combined["decisions"] = [a or b for a, b in zip(combined["decisions"], answer["decisions"], strict=True)]
            for key, decisions in answer["history_decisions"].items():
                combined["history_decisions"].setdefault(key, []).extend(decisions)
            combined["recall_blocked"] = combined.get("recall_blocked", False) or answer.get("recall_blocked", False)
        return hermes_privacy.eligible(payload, combined)
    except hermes_privacy.MetadataTooLarge:
        return {**payload, "_oversized": True}


def _send_hermes_item(config: dict, item: Path, until: float) -> tuple[int | None, dict]:
    """Settle one head snapshot; never transmit content after an unsuccessful metadata check."""
    kept = json.loads(item.read_text(encoding="utf-8"))
    payload = kept["payload"]
    checked = _hermes_checked(config, payload, min(5.0, max(0.2, until - time.monotonic())))
    if checked is None:
        return None, {}
    if checked.get("_oversized"):
        hermes_queue.discard(config, item, payload, _now())
        return 0, {"terminal": True, "retry": False}
    if checked.get("_refused"):
        hermes_queue.acknowledge(config, item, payload)
        return 0, {"terminal": True, "retry": False}
    if time.monotonic() >= until:
        return None, {}
    answer = _post(config, {**kept, "payload": checked}, min(30.0, until - time.monotonic()))
    if answer is None or answer.get("retry"):
        return None, answer or {}
    if answer.get("refused") == 413:
        hermes_queue.discard(config, item, payload, _now())
        return 0, {"terminal": True, "retry": False}
    hermes_queue.acknowledge(config, item, payload)
    return 1, {**answer, "delivered_revision": payload.get("revision")}


def _drain_hermes(config: dict, until: float, target: str | None = None) -> tuple[int, dict]:
    """Drain under the common send mutex; a claimed head stops this sender rather than being overtaken."""
    sent, result = 0, {}
    while time.monotonic() < until - 0.2:
        items = sorted(_spool_dir(config).glob("*.json"))
        if not items:
            break
        item = items[0]
        try:
            with advisory_file_lock(item.with_suffix(".claim"), timeout_seconds=0):
                outcome, answer = _send_hermes_item(config, item, until)
        except TimeoutError:
            break
        except (OSError, ValueError) as exc:
            _log(config, f"Hermes queue: {type(exc).__name__}; head retained")
            break
        _remove_settled_claim(item)
        if outcome is None:
            break
        sent += outcome
        if target is not None and item.name.endswith(f"-{target}.json"):
            result = answer
            break
    return sent, result


def _flush_hermes(config: dict, seconds: float) -> int:
    try:
        with hermes_queue.sending(config):
            return _drain_hermes(config, time.monotonic() + seconds)[0]
    except TimeoutError:
        return 0


def hermes_hook(config: dict[str, Any], payload: dict, *, keep: bool = True) -> dict:
    """Coalesce one turn, then serialize transmissions from the shared Hermes queue."""
    until = time.monotonic() + 7.0
    checked = None if _server_away(config) else _hermes_checked(config, payload, min(5.0, until - time.monotonic()))
    if checked is not None and (checked.get("_oversized") or checked.get("_refused")):
        if keep and checked.get("_oversized"):
            hermes_queue.discard(config, None, payload, _now())
        return {"terminal": True, "retry": False}
    if checked is None and _private_rules(config):
        return {}
    if keep:
        try:
            hermes_queue.put(config, checked or payload, _now())
        except (OSError, TimeoutError, ValueError) as exc:
            _log(config, f"Hermes queue: could not retain ({type(exc).__name__})")
            return {}
    if checked is None:
        return {}
    try:
        with hermes_queue.sending(config):
            _sent, answer = _drain_hermes(config, until, payload.get("queue_key") if keep else None)
            if keep:
                return answer
            if any(_spool_dir(config).glob("*.json")) or time.monotonic() >= until:
                return {}
            return _post(config, {"payload": checked}, until - time.monotonic()) or {}
    except TimeoutError:
        return {}


def hermes_tool(config: dict[str, Any], name: str, arguments: dict, meta: dict) -> str | None:
    """Preflight read privacy with content-free metadata before sending a tool's filtered arguments."""
    if _server_away(config):
        return None
    payload = meta.get("scope-recall/hermes")
    if not isinstance(payload, dict):
        return '{"error":"ACCESS_DENIED"}'
    checked = _hermes_checked(config, {**payload, "tool_read": True}, 1.0) if isinstance(payload, dict) else None
    if checked is None or checked.get("_refused") or checked.get("_oversized"):
        return None
    try:
        with hermes_queue.sending(config):
            answer = _post(
                config,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments, "_meta": meta},
                },
                6.0,
                endpoint="/mcp",
            )
    except TimeoutError:
        return None
    result = (answer or {}).get("result", {})
    return next((row["text"] for row in result.get("content", []) if row.get("type") == "text"), None)


def owner_prompt(host: str, payload: dict[str, Any]) -> bool:
    """Accept a host-attested direct user prompt, retaining every source field."""
    if payload.get("hook_event_name") != "UserPromptSubmit" or agent_run(host, payload):
        return False
    if payload.get("attachments") not in (None, []) or payload.get("record") is not None:
        return False
    if remote_hook_wire(host) == "antigravity":
        return False
    if host == "dsh" and payload.get("scope_recall_command_eligible") is False:
        return False
    role = payload.get("role")
    origin = payload.get("origin")
    source = payload.get("source")
    if isinstance(source, dict):
        source = source.get("kind")
    return role in (None, "user") and origin in (None, "human_direct") and source in (None, "user", "human_direct")


def direct_owner_command(body: dict[str, Any], host: str) -> bool:
    """Check the entire fresh owner-command envelope before accepting it."""
    payload = body.get("payload")
    return bool(
        isinstance(payload, dict)
        and body.get("direct_owner_command") is True
        and body.get("fresh_observation") is True
        and "record" not in body
        and owner_prompt(host, payload)
        and parse_recording_command(payload.get("prompt"))
    )


# -- private state before anything is sent -----------------------------------


def _private_rules(config: dict[str, Any]) -> bool:
    """Use ordinary offline queues until the server reports private history."""
    try:
        answer = json.loads((config["state_dir"] / "private-rules.json").read_text(encoding="utf-8"))
        return isinstance(answer, dict) and answer.get("private_rules") is True
    except (OSError, ValueError):
        return False


def _remember_private_rules(config: dict[str, Any], answer: dict[str, Any]) -> None:
    """Atomically retain only the latest boolean; readers must never see a partial file."""
    value = answer.get("private_rules")
    if type(value) is not bool:
        return
    folder = config["state_dir"]
    try:
        folder.mkdir(parents=True, exist_ok=True)
        pending = folder / f"private-rules.{os.getpid()}.tmp"
        pending.write_text(json.dumps({"private_rules": value}), encoding="utf-8", newline="\n")
        os.replace(pending, folder / "private-rules.json")
    except OSError as exc:
        _log(config, f"private rule metadata not kept ({type(exc).__name__})")


def _private_drop(state: dict[str, Any], event: object) -> bool:
    """Whether this hook must not be sent: full blocks everything; read-only keeps only the prompt's recall."""
    if state.get("recall_blocked"):
        return True
    if not state.get("capture_blocked"):
        return False
    return not (state.get("mode") == "read-only" and event == "UserPromptSubmit")


def _private_check(config: dict[str, Any], body: dict[str, Any], timeout: float) -> dict[str, Any] | None:
    """Check all record instants in bounded metadata requests before sending any content."""
    times = _private_times(body, body.get("observed_at"), config.get("host"))
    until = time.monotonic() + timeout
    fresh = body.get("fresh_observation") is True or body["payload"].get("fresh_observation") is True
    if fresh and not isinstance(body.get("clock"), dict):
        calibration = _private_batch(config, body, [], timeout)
        if calibration is None:
            return None
        body["clock"] = calibration.get("clock")
        if type(calibration.get("rule_marker")) is int:
            body.setdefault("rule_marker", calibration["rule_marker"])
        if calibration.get("recall_blocked"):
            return calibration
    state = None
    decisions = {}
    full_decisions = {}
    for start in range(0, max(1, len(times)), 64):
        left = until - time.monotonic()
        if left <= 0:
            return None
        answer = _private_batch(config, body, times[start : start + 64], left)
        if answer is None:
            return None
        if type(answer.get("rule_marker")) is int and "rule_marker" not in body:
            body["rule_marker"] = answer["rule_marker"]
        verdicts = answer.get("decisions")
        if isinstance(verdicts, list) and len(verdicts) == len(times[start : start + 64]):
            decisions.update(zip(times[start : start + 64], verdicts, strict=True))
        full = answer.get("full_decisions")
        if not isinstance(full, list) or len(full) != len(times[start : start + 64]):
            full = [answer["capture_blocked"]] * len(times[start : start + 64])
        full_decisions.update(zip(times[start : start + 64], full, strict=True))
        if state is None:
            state = dict(answer)
        else:
            state["capture_blocked"] |= answer["capture_blocked"]
            state["recall_blocked"] |= answer["recall_blocked"]
        if type(answer.get("private_rules")) is bool:
            state["private_rules"] = answer["private_rules"]
    if state is not None:
        state["event_decisions"] = decisions
        state["full_event_decisions"] = full_decisions
        state["clocks"] = private_metadata(body, times, host=config.get("host"))["clocks"]
    return state


def eligible_body(
    body: dict[str, Any], state: dict[str, Any], host: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate per-event Core decisions to content-free record lines, preserving cursor progress."""
    decisions = state.get("event_decisions") or {}
    body, state = dict(body), dict(state)
    record = body.get("record")
    if isinstance(record, dict):
        body["record"] = {
            **record,
            "lines": [
                [end, None if isinstance(said, dict) and decisions.get(said.get("occurred_at"), True) else said]
                for end, said in record.get("lines", [])
            ],
        }
    payload = dict(body["payload"])
    refused_replies = []
    if isinstance(record, dict):
        refused_replies.extend(
            said
            for _end, said in record.get("lines", [])
            if isinstance(said, dict) and decisions.get(said.get("occurred_at"), True)
        )
    if isinstance(payload.get("record"), list):
        lines = dict(transcript.dsh_lines(payload["record"]))
        refused_replies.extend(
            dict(id=said.entry_id, text=said.text, role=said.role)
            for said in lines.values()
            if said is not None and decisions.get(said.occurred_at, True)
        )
        payload["record"] = [
            None if (said := lines.get(index)) is not None and decisions.get(said.occurred_at, True) else row
            for index, row in enumerate(payload["record"], start=1)
        ]
    if any(
        said.get("role") == "assistant"
        and (
            (payload.get("reply_id") is not None and said.get("id", said.get("entry_id")) == payload.get("reply_id"))
            or said.get("text") == payload.get("last_assistant_message")
        )
        for said in refused_replies
    ):
        payload.pop("last_assistant_message", None)
        payload.pop("reply_id", None)
    body["payload"] = payload
    moments = prompt_times(body, body.get("observed_at"), host) or [None]
    if decisions:
        state["capture_blocked"] = state.get("mode") != "none" or any(decisions.get(t, True) for t in moments)
    full = state.get("full_event_decisions") or {}
    if any(full.get(t, False) for t in moments):
        # Read-only permits ordinary questions, never the replay of words from
        # a full interval or an undated question that could belong to one.
        state["recall_blocked"] = True
    return body, state


def _private_batch(
    config: dict[str, Any], body: dict[str, Any], times: list[str], timeout: float
) -> dict[str, Any] | None:
    """Ask the entry's server whether this request or a recall is private, before sending any content.

    Only a session id, a project path and event instants travel here.  A missing
    answer means the state could not be verified: the caller must not send the
    hook's content.
    """
    payload = body.get("payload")
    if not isinstance(payload, dict):
        return None
    session = payload.get("session_id")
    if type(session) is not str or not session.strip():
        return None
    request = {
        **private_metadata(body, times, host=config.get("host")),
        "sent_at": body.get("observed_at") or payload.get("calibration_started_at") or _now(),
    }
    try:
        token = config["token_file"].read_text(encoding="utf-8").strip()
    except OSError as exc:
        _log(config, f"{payload.get('hook_event_name')}: no token to send ({type(exc).__name__})")
        return None
    url = urllib.parse.urlsplit(config["url"])
    kind = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    timeout = max(0.2, min(1.0, timeout))
    connection = kind(url.hostname, url.port, timeout=timeout)
    try:
        try:
            connection.connect()
        except OSError as exc:
            _connection_failed(config, payload.get("hook_event_name"), exc)
            return None
        connection.request(
            "POST",
            f"{url.path}/private/check",
            body=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            _log(config, f"{payload.get('hook_event_name')}: HTTP {response.status}; private state unverified")
            return None
        answer = json.loads(raw.decode("utf-8"))
        if isinstance(answer, dict) and isinstance(answer.get("clock"), dict) and "received_at" not in answer["clock"]:
            answer["clock"]["received_at"] = _now()
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        connection.close()
    if (
        not isinstance(answer, dict)
        or type(answer.get("capture_blocked")) is not bool
        or type(answer.get("recall_blocked")) is not bool
    ):
        return None
    _remember_private_rules(config, answer)
    return answer


# -- the spool (Codex) -------------------------------------------------------


def _spool_dir(config: dict[str, Any]) -> Path:
    return config["state_dir"] / "spool"


_SPOOL_SEQUENCE = itertools.count()


def _spool(
    config: dict[str, Any], payload: dict[str, Any], observed_at: str, *, clock=None, claims=None
) -> Path | None:
    folder = _spool_dir(config)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        kept = sorted(folder.glob("*.json"))
        dropped = kept[: max(0, len(kept) - SPOOL_LIMIT + 1)]
        for stale in dropped:
            stale.unlink(missing_ok=True)
            _remove_settled_claim(stale)
        if dropped:
            _log(config, f"spool full: dropped the {len(dropped)} oldest")
        # The clock can read the same twice in a row, and a name taken twice would replace the hook kept under it.
        name = f"{time.time_ns():020d}-{os.getpid()}-{next(_SPOOL_SEQUENCE):06d}.json"
        pending = folder / f"{name}.tmp"
        pending.write_text(
            json.dumps({"payload": payload, "observed_at": observed_at, "clock": clock}, ensure_ascii=False),
            encoding="utf-8",
        )
        if claims is not None:
            claims.callback(_remove_settled_claim, folder / name)
            claims.enter_context(advisory_file_lock((folder / name).with_suffix(".claim"), timeout_seconds=0))
        os.replace(pending, folder / name)
        return folder / name
    except OSError:
        return None


def flush_spool(config: dict[str, Any], seconds: float = FLUSH_SECONDS) -> int:
    """Send what earlier hooks could not, oldest first, within ``seconds``; stop at the first failure.

    One flusher at a time (a lock file beside the spool).  Returns how many were sent.
    """
    if config["host"] == "hermes":
        return _flush_hermes(config, seconds)
    folder = _spool_dir(config)
    if not folder.is_dir():
        return 0
    lock = folder / "flush.lock"
    try:
        if lock.exists() and time.time() - lock.stat().st_mtime > 2 * seconds:
            lock.unlink(missing_ok=True)
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except OSError:
        return 0
    sent = 0
    until = time.monotonic() + seconds
    try:
        for item in sorted(folder.glob("*.json")):
            left = until - time.monotonic()
            if left < 1.0:
                break
            try:
                with advisory_file_lock(item.with_suffix(".claim"), timeout_seconds=0):
                    outcome = _flush_item(config, item, left)
            except TimeoutError:
                continue
            _remove_settled_claim(item)
            if outcome is None:
                break
            sent += outcome
    finally:
        lock.unlink(missing_ok=True)
    left = sum(1 for _item in folder.glob("*.json"))
    _log(config, f"flush: sent {sent}, {left} still kept")
    return sent


def _remove_settled_claim(item: Path) -> None:
    """Remove an unlocked sidecar only after its uniquely named payload has gone."""
    try:
        if not item.exists():
            item.with_suffix(".claim").unlink(missing_ok=True)
    except OSError:
        # Another sender may still have opened this settled sidecar on Windows.
        # Its cleanup cannot turn a delivered hook into a failed request.
        pass


def _flush_item(config, item: Path, left: float) -> int | None:
    """Send one claimed item; None stops this pass, zero skips it, one counts delivery."""
    if not item.exists():
        return 0
    try:
        kept = json.loads(item.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _log(config, f"flush: dropped an unreadable spool file ({type(exc).__name__})")
        item.unlink(missing_ok=True)
        return 0
    payload = kept.get("payload")
    session = payload.get("session_id") if isinstance(payload, dict) else None
    session = session if type(session) is str else None
    compact = compaction_marker(config, session) if session is not None else None
    if compact is not None:
        kept.update(compacted_at=compact["at"], compaction_seq=compact["ordinal"], compaction_epoch=compact["epoch"])
    if config["host"] != "hermes" and _private_rules(config):
        state = _private_check(config, kept, min(left, 1.0))
        if state is None:
            return None
        if state["capture_blocked"] or state["recall_blocked"]:
            item.unlink(missing_ok=True)
            _log(config, f"flush: dropped one kept hook under private mode {state.get('mode')}")
            return 0
    answer = _post(config, kept, min(left, 30.0))
    if answer is None or answer.get("retry"):
        return None
    _ack_compaction(config, session, compact, answer)
    item.unlink(missing_ok=True)
    return int(not answer.get("refused"))


def _start_flush(config: dict[str, Any]) -> None:
    """Flush the spool in a process of its own, so the hook itself answers inside its short wait.

    On Windows with a console of its own that has no window, not detached: the interpreter is often a launcher
    that starts python.exe as its child, and a console program started by a process without a console gets a
    new console, which Windows Terminal shows as a window on the desktop.
    """
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-m",
                "scope_recall.adapters.codex.remote_client",
                "flush",
                "--config",
                str(config["config"]),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=flags,
            start_new_session=os.name != "nt",
        )
    except OSError:
        pass


# -- the hook ----------------------------------------------------------------


def run_hook(
    config: dict[str, Any], raw: bytes, *, started: float | None = None, event: str | None = None
) -> dict[str, Any]:
    started = time.monotonic() if started is None else started
    payload, local, original = _hook_input(raw, host=config.get("host"), event=event)
    if payload is None:
        return local
    event = payload.get("hook_event_name")
    _normalize_hook_project(payload)
    host = config["host"]
    if event == "PrivateCheck":
        return _private_check(config, {"payload": payload}, 2.0) or {}
    wait = HOOK_TIMEOUTS[host].get(event)
    if wait is None:
        return {}
    until = started + wait * _REQUEST_SHARE
    # Every request carries the moment the hook ran here.  A request that timed out here may still have been
    # stored there, and its replay must then be the same event, not a second one with a later time.
    observed_at = _now()
    direct_command = bool(original and owner_prompt(host, original) and parse_recording_command(original.get("prompt")))
    session, compact = (None, None) if direct_command else _pending_compaction(config, payload, observed_at)
    if host == "claude-code" and event == "UserPromptSubmit":
        payload = _with_context_since(payload)
    body: dict[str, Any] = {"payload": payload, "observed_at": observed_at, "fresh_observation": True}
    if direct_command:
        body = {
            "payload": payload,
            "observed_at": observed_at,
            "fresh_observation": True,
            "direct_owner_command": True,
            "command_id": secrets.token_hex(16),
        }
    if compact is not None:
        body.update(compacted_at=compact["at"], compaction_seq=compact["ordinal"], compaction_epoch=compact["epoch"])
    cursor = None
    record_page = None
    if (
        host in _RECORD_HOSTS
        and event in ("Stop", "SessionEnd")
        and not (host == "workbuddy" and is_workbuddy_agent_run(payload))
    ):
        part = _record_part(config, payload)
        if part is not None:
            body["record"], cursor, record_page = part
        if host == "workbuddy" and event == "Stop" and _error_reply(payload):
            # The record is here, not on the server: this side says whether the reply is an error WorkBuddy showed.
            body["error_reply"] = True

    if remote_hook_wire(host) == "antigravity":
        from .antigravity import scrub_remote

        scrub_remote(body)
    answer = _post(config, body, until - time.monotonic()) if direct_command else _send_checked(config, body, until)
    if answer is None:
        return command_block(host, "Scope Recall could not change recording mode.") if direct_command else {}
    if answer.get("refused"):
        return command_block(host, "Scope Recall could not change recording mode.") if direct_command else {}
    _ack_compaction(config, session, compact, answer)
    through = answer.get("through")
    if cursor is not None and type(through) is int and through > body["record"]["start"]:
        cursor.save(through, directory=record_page.directory_at(through) if record_page is not None else None)
    if host == "codex" and any(_spool_dir(config).glob("*.json")):
        _start_flush(config)
    result = answer.get("result")
    return result if isinstance(result, dict) else {}


def _send_ordinary(config: dict[str, Any], body: dict[str, Any], until: float, claims) -> dict[str, Any] | None:
    """Keep ordinary offline hooks before connecting, including during the server cooldown."""
    payload = body["payload"]
    event = payload.get("hook_event_name")
    spooled = (
        _spool(config, payload, body["observed_at"], clock=body.get("clock"), claims=claims)
        if config["host"] == "codex" and event in _SPOOLED
        else None
    )
    if _server_away(config):
        _log(config, f"{event}: not sent, the server was away less than {AWAY_SECONDS:.0f} s ago")
        return None
    answer = _post(config, body, until - time.monotonic())
    if spooled is not None and answer is not None and not answer.get("retry"):
        spooled.unlink(missing_ok=True)
    return answer


def _send_checked(config: dict[str, Any], body: dict[str, Any], until: float) -> dict[str, Any] | None:
    """Send ordinary hooks directly; verify before keeping or sending once private history is known."""
    with ExitStack() as claims:
        if config["host"] != "antigravity" and not _private_rules(config):
            return _send_ordinary(config, body, until, claims)
        return _send_private(config, body, until, claims)


def _send_private(config, body, until, claims):
    """Verify and send a hook while retaining its foreground spool claim."""
    payload = body["payload"]
    event = payload.get("hook_event_name")
    if _server_away(config):
        _log(config, f"{event}: not sent, the server was away less than {AWAY_SECONDS:.0f} s ago")
        return None
    # The server answers whether these moments are private before any content
    # leaves this machine. If it cannot answer, no content is kept or sent.
    state = _private_check(config, body, until - time.monotonic())
    if state is None:
        _log(config, f"{event}: private state unverified, not sent")
        return None
    body, state = eligible_body(body, state, config.get("host"))
    payload = body["payload"]
    if _private_drop(state, event):
        _log(config, f"{event}: private mode {state.get('mode')}, not sent")
        if event == "UserPromptSubmit" and state.get("mode") == "full":
            notice = notice_for_mode("full") or ""
            return {
                "result": {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": notice}}
            }
        return None
    if state["capture_blocked"]:
        # Read-only prompts travel only as recall questions, without attachments or a durable spool copy.
        body = _recall_only(body, config.get("host"))
        spooled = None
    else:
        spooled = (
            _spool(config, payload, body["observed_at"], clock=body.get("clock"), claims=claims)
            if config["host"] == "codex" and event in _SPOOLED
            else None
        )
    # Disk I/O and queue preparation can outlive the first decision. Recheck at
    # the last boundary before content is transmitted, using the original clock.
    fresh = _private_check(config, body, until - time.monotonic())
    if fresh is not None:
        body, fresh = eligible_body(body, fresh, config.get("host"))
    if fresh is None or _private_drop(fresh, event):
        if spooled is not None:
            spooled.unlink(missing_ok=True)
        return None
    if fresh["capture_blocked"]:
        body = _recall_only(body, config.get("host"))
        if spooled is not None:
            spooled.unlink(missing_ok=True)
            spooled = None
    answer = _post(config, body, until - time.monotonic())
    if answer is None:
        return None
    if spooled is not None and not answer.get("retry"):
        spooled.unlink(missing_ok=True)
    return answer


def _recall_only(body, host=None):
    """Read-only prompts carry only the question and its routing metadata."""
    fields = {
        "hook_event_name",
        "session_id",
        "turn_id",
        "prompt_id",
        "cwd",
        "project_unverified",
        "prompt",
        IN_CONTEXT_FIELD,
    }
    if remote_hook_wire(host) == "antigravity":
        fields.add("occurred_at")
    return {**body, "payload": {key: value for key, value in body["payload"].items() if key in fields}}


# -- setup on this machine ---------------------------------------------------


def make_token(config: dict[str, Any]) -> str:
    """Create the entry's token here if there is none, readable by this user only; return its SHA-256."""
    path = config["token_file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.name + ".tmp")
        pending.write_text(secrets.token_urlsafe(32), encoding="utf-8")
        if os.name == "nt":
            user = os.environ.get("USERNAME", "")
            subprocess.run(
                ["icacls", str(pending), "/inheritance:r", "/grant:r", f"{user}:F", "*S-1-5-18:F"],
                check=True,
                capture_output=True,
            )
        else:
            pending.chmod(0o600)
        os.replace(pending, path)
    return hashlib.sha256(path.read_text(encoding="utf-8").strip().encode("utf-8")).hexdigest()


def _empty_answer(path: object) -> str:
    """What a hook whose config did not load writes: nothing to add, as the host its config still names takes it
    (``EMPTY_ANSWER``), else "{}"."""
    try:
        host = json.loads(Path(str(path)).read_text(encoding="utf-8")).get("host")
    except (OSError, ValueError, AttributeError):
        host = None
    answer = EMPTY_ANSWER.get(host, "{}") if type(host) is str else "{}"
    return answer + "\n" if answer else ""


def _queue_report(config: dict, command: str) -> dict:
    result = {"sent": flush_spool(config)} if command == "flush" else {}
    if config["host"] == "hermes":
        result.update(hermes_queue.status(config))
    return result


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = list(sys.argv[1:] if argv is None else argv)
    command = args.pop(0) if args and args[0] in ("token", "flush", "status", "mcp") else "hook"
    parser = argparse.ArgumentParser(prog="scope-recall-remote-client")
    parser.add_argument("--config", required=True)
    parser.add_argument("--event", choices=("SessionStart", "PreInvocation", "PostInvocation", "Stop"))
    parsed = parser.parse_args(args)
    try:
        config = load_client_config(absolute_path(parsed.config, "config"))
        if command == "mcp":
            from .remote_mcp import serve

            return serve(config, _post)
        if command == "token":
            print(json.dumps({"token_sha256": make_token(config)}))
            return 0
        if command in {"flush", "status"}:
            print(json.dumps(_queue_report(config, command)))
            return 0
    except RemoteClientError as exc:
        if command == "hook":
            sys.stdout.write(_empty_answer(parsed.config))
            sys.stderr.write(f"SCOPE_RECALL_REMOTE:{exc}\n")
            return 0
        raise SystemExit(str(exc)) from None
    raw = sys.stdin.buffer.read(_MAX_STDIN + 1)
    result = run_hook(config, raw, started=started, event=parsed.event) if len(raw) <= _MAX_STDIN else {}
    if config["host"] == "antigravity":
        from .antigravity import answer

        result = answer(result, parsed.event or "")
    # ASCII only, as the local hook writes it: a pipe on Windows carries the system code page (GBK on a Chinese
    # Windows, under -I whatever PYTHONUTF8 says), and the host reads UTF-8, so recalled Chinese text arrived
    # garbled or not at all.  Nothing to add is written as the local hook writes it for this client (EMPTY_ANSWER).
    if result or EMPTY_ANSWER[config["host"]]:
        sys.stdout.write(json.dumps(result, ensure_ascii=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
