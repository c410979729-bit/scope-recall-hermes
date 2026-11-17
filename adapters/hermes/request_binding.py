"""Native Hermes request witnesses, scoped to one routed provider instance."""

import json
import math
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock

from scope_recall.core.entry_control import refuse_unbound_mutation

from ..request_binding import bind_source
from .boundary import host_source_key, person_steers, steer_source_key


def request_ids(fields):
    """Use execution-owned occurrence identifiers, never tool arguments."""
    keys = ("session_id", "turn_id", "api_request_id", "task_id")
    values = tuple(fields.get(key) for key in keys)
    return values if all(type(value) is str and value.strip() for value in values) else None


def fingerprint(messages) -> str:
    """Require an exact immutable copy of the actual provider input."""
    if not isinstance(messages, list):
        return ""
    return sha256(json.dumps(messages, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def words(content):
    """Accept text-only provider messages; multimodal or transformed input refuses."""
    if type(content) is str:
        return content
    if (
        isinstance(content, list)
        and content
        and all(
            isinstance(part, dict) and part.get("type") in {"text", "input_text"} and type(part.get("text")) is str
            for part in content
        )
    ):
        return "\n".join(part["text"] for part in content)
    return None


@dataclass(frozen=True)
class Snapshot:
    """One pre-request observation; a helper thread cannot grant tool authority."""

    ids: tuple
    started: float
    retry: int
    request_digest: str
    source_key: str
    text: str
    uid: str
    identity: object


def snapshot(adapter, fields):
    """Select the latest user row only, refusing merges and ambiguous inclusion."""
    ids = request_ids(fields)
    started, retry = fields.get("started_at"), fields.get("retry_count")
    if (
        ids is None
        or type(started) not in (int, float)
        or not math.isfinite(started)
        or type(retry) is not int
        or retry < 0
    ):
        return None
    history, request = fields.get("conversation_history"), fields.get("request_messages")
    if not isinstance(history, list) or not isinstance(request, list):
        return None
    users = [row for row in history if isinstance(row, dict) and row.get("role") == "user"]
    if not users:
        return None
    row = users[-1]
    uid, text = row.get("message_uid"), words(row.get("content"))
    if (
        type(uid) is not str
        or not uid.strip()
        or len(uid) > 240
        or not text
        or len(text) > 8192
        or any(
            row.get(key)
            for key in ("_compressed_summary", "_merged_turn_prefix", "_absorbed_message_uids", "_absorbed_row_ids")
        )
        or sum(words(item.get("content")) == text for item in users) != 1
        or sum(item.get("message_uid") == uid for item in users) != 1
    ):
        return None
    wire = [item for item in request if isinstance(item, dict) and item.get("role") == "user"]
    if (
        not wire
        or words(wire[-1].get("content")) != text
        or sum(words(item.get("content")) == text for item in wire) != 1
    ):
        return None
    identity = adapter._require_identity()
    context = identity.trusted_context(mutation=True)
    if row.get("display_kind") in (None, "") and text == fields.get("user_message"):
        key = host_source_key(
            installation_id=context.binding.installation_id,
            entry_id=context.entry_id,
            session_id=ids[0],
            event_kind="user",
            event_id=ids[1],
        )
    elif row.get("display_kind") == "steer":
        selected = steer_occurrence(identity, context, row, users)
        if selected is None:
            return None
        key, text = selected
    else:
        return None
    return Snapshot(ids, float(started), retry, fingerprint(request), key, text, uid, identity)


def steer_occurrence(identity, context, row, users):
    """Unwrap only the existing native steer grammar, with a unique stored key."""
    stamp = row.get("timestamp")
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0:
        return None
    steers = person_steers([row], platform=identity.scope.platform, user_id=identity.scope.user_id, local_surface=True)
    if len(steers) != 1:
        return None
    selected = steers[0]
    matching = sum(
        steer.key == selected.key and steer.kind == selected.kind
        for item in users
        for steer in person_steers(
            [item], platform=identity.scope.platform, user_id=identity.scope.user_id, local_surface=True
        )
    )
    return (steer_source_key(context, selected), selected.words) if matching == 1 else None


def _field(value, name):
    """Read native SDK objects and equivalent decoded wire mappings without coercion."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def tool_calls(response):
    """Seal only known successful Chat Completions or Responses tool occurrences."""
    choices = _field(response, "choices")
    if isinstance(choices, list) and len(choices) == 1:
        calls = _field(_field(choices[0], "message"), "tool_calls") or []
        pairs = [(_field(call, "id"), _field(_field(call, "function"), "name")) for call in calls]
    elif _field(response, "status") == "completed" and isinstance(output := _field(response, "output"), list):
        pairs = []
        for item in output:
            if _field(item, "type") != "function_call":
                continue
            if _field(item, "status") not in (None, "completed"):
                return {}
            pairs.append((_field(item, "call_id"), _field(item, "name")))
    else:
        return {}
    result = {}
    for call_id, name in pairs:
        if type(call_id) is not str or not call_id or type(name) is not str or not name or call_id in result:
            return {}
        result[call_id] = name
    return result


class NativeRequests:
    """A provider's latest attempt and synchronously executing tool credential.

    Every execution start revokes the previous seal, including a retry reusing
    identifiers. A late observer cannot replace a consumed attempt.
    """

    def __init__(self):
        self.lock = RLock()
        self.current: Snapshot | None = None
        self.ready = False
        self.consumed = False
        self.calls = {}
        self.active: ContextVar[tuple[Snapshot, list[bool]] | None] = ContextVar("hermes_native_request", default=None)
        self.highwater = -math.inf
        self.occurrences = {}
        self.executed = set()

    def observe(self, adapter, fields):
        """Invalidate before capture and retain only an ordered immutable observation."""
        with self.lock:
            self.current = None
            self.calls = {}
            self.ready = False
            candidate = snapshot(adapter, fields)
            if candidate is None or candidate.started <= self.highwater or candidate.ids in self.executed:
                self.current = None
                return None
            if len(self.occurrences) >= 4096 and candidate.source_key not in self.occurrences:
                return None
            if self.occurrences.setdefault(candidate.source_key, candidate.uid) != candidate.uid:
                return None
            self.highwater = candidate.started
            self.current, self.consumed = candidate, False
            return candidate

    def finish(self, candidate):
        """Only an observation that finished before execution can be consumed."""
        with self.lock:
            if candidate is not None and self.current is candidate and not self.consumed:
                self.ready = True

    def llm(self, adapter, fields, next_call, request):
        """Seal only a successful execution with its own pre-request observation.

        Execution metadata omits the retry counter. An API id is consumed even
        on failure, so a late observer cannot authorize a retry reusing that id.
        """
        with self.lock:
            candidate = self.current
            ids = request_ids(fields)
            self.calls = {}
            eligible = (
                candidate is not None
                and self.ready
                and not self.consumed
                and candidate.ids == request_ids(fields)
                and candidate.request_digest == fingerprint(request.get("messages", request.get("input")))
                and candidate.identity is adapter._identity
                and ids not in self.executed
                and len(self.executed) < 4096
            )
            if ids is not None and len(self.executed) < 4096:
                self.executed.add(ids)
            self.consumed = True
            if not eligible:
                self.current = None
        result = next_call(request)
        with self.lock:
            if eligible and self.current is candidate:
                self.calls = tool_calls(result)
        return result

    def tool(self, adapter, fields, next_call, args):
        """Bind around the synchronous final dispatcher and always reset its token."""
        with self.lock:
            candidate = self.current
            eligible = (
                candidate is not None
                and candidate.ids == request_ids(fields)
                and self.calls.get(fields.get("tool_call_id")) == fields.get("tool_name")
                and candidate.identity is adapter._identity
            )
            self.calls.pop(fields.get("tool_call_id"), None)
            lease = [True]
            handle = (candidate, lease) if eligible and candidate is not None else None
            token = self.active.set(handle)
            try:
                return next_call(args)
            finally:
                lease[0] = False
                self.active.reset(token)

    def context(self, adapter, context):
        """Resolve the sealed occurrence through Core without repairing model refs."""
        handle = self.active.get()
        if handle is None:
            refuse_unbound_mutation()
        candidate = handle[0]
        if self.current is not candidate or candidate.identity is not adapter._identity:
            refuse_unbound_mutation()
        if self.occurrences.get(candidate.source_key) != candidate.uid:
            refuse_unbound_mutation()
        return bind_source(
            adapter._require_core(),
            context,
            candidate.source_key,
            candidate.text,
            candidate.uid,
            lambda: (
                handle[1][0]
                and self.current is candidate
                and self.active.get() is handle
                and adapter._identity is candidate.identity
            ),
        )
