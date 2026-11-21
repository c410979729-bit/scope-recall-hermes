"""Local Codex pre-tool witnesses joined to retained rollout occurrences.

The disposable files contain locators, occurrence ids and argument digests,
never source bodies or scope grants. SQLite remains the source authority.
Unknown native layouts and clients without both public hooks refuse binding.
"""

import json
from contextlib import contextmanager
from hashlib import sha256
from typing import Any
from uuid import uuid4

from scope_recall.core.entry_control import refuse_unbound_mutation
from scope_recall.core.file_lock import advisory_file_lock

from ...host_registry import HOSTS
from ..request_binding import bind_source
from . import transcript
from .boundary import host_source_key
from .config import SharedClientConfig


def _path(config, session):
    home = config.home if isinstance(config, SharedClientConfig) else config.data_directory
    return home / "scope-recall" / "native-requests" / (sha256(session.encode()).hexdigest() + ".json")


def _load(path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def _digest(args):
    # Null optionals and request ids have no authorization semantics. All
    # effect-bearing fields still match the native tool invocation exactly.
    if not isinstance(args, dict):
        refuse_unbound_mutation()
    normalized = {key: value for key, value in args.items() if value is not None and key != "request_id"}
    return sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _rows(config, record, session, *, through=None):
    path = transcript.codex_record_path(record, session)
    if path is None or (
        isinstance(config, SharedClientConfig)
        and not path.resolve().is_relative_to((config.home / "sessions").resolve())
    ):
        refuse_unbound_mutation()
    size = path.stat().st_size if through is None else through
    if type(size) is not int or not 0 < size <= transcript.READ_BYTES or size > path.stat().st_size:
        refuse_unbound_mutation()
    with path.open("rb") as stream:
        raw = stream.read(size)
    if not raw.endswith(b"\n"):
        refuse_unbound_mutation()
    rows = [json.loads(line) for line in raw.splitlines()]
    if any(
        not isinstance(row, dict)
        or (
            row.get("type") == "response_item"
            and (not isinstance(row.get("payload"), dict) or not isinstance(row.get("metadata"), dict))
        )
        for row in rows
    ):
        refuse_unbound_mutation()
    return rows, size


def _latest(rows):
    """Select the last user boundary, refusing ambiguous ordering or provenance."""
    latest = None
    sequences, occurrences, orders = set(), set(), set()
    previous = 0
    for row in rows:
        if row.get("type") != "response_item":
            continue
        payload, meta = row.get("payload", {}), row.get("metadata", {})
        sequence = meta.get("history_sequence")
        if type(sequence) is not int or sequence <= previous or sequence in sequences:
            refuse_unbound_mutation()
        previous = sequence
        sequences.add(sequence)
        if payload.get("role") != "user":
            continue
        latest = transcript.codex_user_occurrence(row)
        if latest is None:
            continue
        if latest[0] in occurrences or (latest[2], latest[4]) in orders:
            refuse_unbound_mutation()
        occurrences.add(latest[0])
        orders.add((latest[2], latest[4]))
    if latest is None:
        refuse_unbound_mutation()
    return latest


def _source_key(config, session, occurrence, rows):
    opener = transcript.codex_openers(rows).get(occurrence[0]) == occurrence[2]
    return host_source_key(
        host="codex",
        installation_id=config.installation_id,
        session_id=session,
        event_kind="user" if opener else "record",
        event_id=occurrence[2] if opener else occurrence[0],
    )


def observe(hook, event, session, audience, payload, deadline):
    """Observe public native hooks without emitting model context or blocking output."""
    path = _path(hook.config, session)
    with advisory_file_lock(path.with_suffix(".lock"), timeout_seconds=0.2):
        state: dict[str, Any] = _load(path)
        if event == "PreModelRequest":
            state = dict(generation=uuid4().hex, calls={})
            _save(path, state)
        elif event in {"SessionStart", "UserPromptSubmit", "Stop", "Interrupt", "SessionEnd"}:
            _save(path, dict(generation=uuid4().hex, calls={}))
            return
        elif event != "PreToolUse":
            return
        if event == "PreToolUse" and payload.get("tool_name") not in {
            "mcp__" + server + "__" + tool
            for server in ("scope-recall", "scope_recall")
            for tool in ("revise", "forget", "propose_memory")
        }:
            calls = state.get("calls")
            if isinstance(calls, dict) and payload.get("tool_use_id") in calls:
                calls[payload["tool_use_id"]] = None
                _save(path, state)
            return
        rows, through = _rows(hook.config, payload.get("transcript_path"), session)
        latest = _latest(rows)
        if latest[2] != payload.get("turn_id"):
            refuse_unbound_mutation()
        if event == "PreModelRequest":
            # Use the normal record cursor, source filters and deletion ledger.
            # Its exact native keys distinguish repeated text and preserve blocks.
            hook.record_reader.read(session, audience, payload, deadline)
            state.update(record=payload["transcript_path"], occurrence=list(latest[:5]), turn=latest[2])
        else:
            call, name = payload.get("tool_use_id"), payload.get("tool_name")
            calls = state.get("calls", {})
            if (
                state.get("occurrence") != list(latest[:5])
                or state.get("turn") != latest[2]
                or state.get("record") != payload.get("transcript_path")
                or type(call) is not str
                or not call
                or type(name) is not str
                or not name
                or not isinstance(calls, dict)
                or len(calls) >= 128
            ):
                refuse_unbound_mutation()
            if call in calls:
                calls[call] = None
            else:
                calls[call] = dict(name=name, digest=_digest(payload.get("tool_input")), through=through)
            state["calls"] = calls
        _save(path, state)


def _parent(rows, item, turn, latest):
    found = [row for row in rows if row.get("type") == "response_item" and row.get("payload", {}).get("id") == item]
    if len(found) != 1:
        refuse_unbound_mutation()
    row = found[0]
    payload, meta = row["payload"], row.get("metadata", {})
    watermark, sequence = meta.get("sampled_through_sequence"), meta.get("history_sequence")
    if (
        payload.get("type") not in {"custom_tool_call", "function_call"}
        or payload.get("internal_chat_message_metadata_passthrough", {}).get("turn_id") != turn
        or meta.get("client_authored") is not False
        or type(watermark) is not int
        or type(sequence) is not int
        or not latest[3] <= watermark < sequence
        or not any(row.get("metadata", {}).get("history_sequence") == watermark for row in rows)
    ):
        refuse_unbound_mutation()


@contextmanager
def request_context(server, ctx, tool, body, context):
    """Consume one actual pre-tool witness before entering the public Core call."""
    try:
        with _request_context(server, ctx, tool, body, context) as bound:
            yield bound
    except (OSError, ValueError, TypeError, KeyError, AttributeError, TimeoutError):
        refuse_unbound_mutation()


@contextmanager
def _request_context(server, ctx, tool, body, context):
    if (
        not HOSTS[server.host].native_write_witness
        or not isinstance(server.config, SharedClientConfig)
        or not server.config.local_account_attested
    ):
        refuse_unbound_mutation()
    meta = getattr(ctx.request_context, "meta", None)
    if not isinstance(meta, dict):
        refuse_unbound_mutation()
    session, call, item = meta.get("threadId"), meta.get("callId"), meta.get("itemId")
    turn_meta = meta.get("x-codex-turn-metadata")
    if (
        any(type(value) is not str or not value for value in (session, call, item))
        or not isinstance(turn_meta, dict)
        or turn_meta.get("thread_id") != session
        or turn_meta.get("session_id") != session
        or meta.get("sessionId") != session
    ):
        refuse_unbound_mutation()
    path = _path(server.config, session)
    with advisory_file_lock(path.with_suffix(".lock"), timeout_seconds=0.2):
        state = _load(path)
        calls = state.get("calls")
        witness = calls.get(call) if isinstance(calls, dict) else None
        if (
            not isinstance(witness, dict)
            or state.get("turn") != turn_meta.get("turn_id")
            or witness.get("name") not in {"mcp__scope-recall__" + tool, "mcp__scope_recall__" + tool}
            or witness.get("digest") != _digest(body)
        ):
            refuse_unbound_mutation()
        state["calls"][call] = None
        _save(path, state)
        rows, _through = _rows(server.config, state.get("record"), session, through=witness.get("through"))
        latest = _latest(rows)
        if state.get("occurrence") != list(latest[:5]):
            refuse_unbound_mutation()
        _parent(rows, item, state["turn"], latest)
        generation = state["generation"]
        active = [True]

        def live():
            current = _load(path)
            now, _size = _rows(server.config, state["record"], session)
            return active[0] and current.get("generation") == generation and _latest(now)[:5] == latest[:5]

        bound = bind_source(
            server.core,
            context,
            _source_key(server.config, session, latest, rows),
            latest[5],
            latest[0] + ":" + latest[1],
            live,
        )
        try:
            yield bound
        finally:
            active[0] = False
