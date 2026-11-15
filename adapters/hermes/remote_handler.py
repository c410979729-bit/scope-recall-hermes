"""Reconstruct compact host facts through the ordinary store-side Hermes adapter."""

from __future__ import annotations

import hashlib
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent, Tool, ToolAnnotations
from scope_recall.contracts import ContractError
from scope_recall.core import CoreConfig, MemoryCore, private
from scope_recall.core.retrieval import MAX_CURRENT_SOURCE_REFS
from starlette.concurrency import run_in_threadpool

from ..clients.config import load_server_entry
from ..clients.hermes_queue import encode
from .identity import HermesIdentityError, SessionUnbound, bind_hermes_identity, switch_hermes_identity
from .provider import ScopeRecallHermesAdapter
from .remote_wire import MAX_FACT_BYTES, META_KEY, METHODS, READ_TOOLS, facts
from .tool_surface import tool_schemas


class HermesRemoteError(ValueError):
    """Malformed host facts, separate from retryable storage failures."""


class EventClock:
    """Original host observation time for capture; Core admission uses its own server clock."""

    def __init__(self) -> None:
        self.at = datetime.now(UTC).isoformat()

    def utc_now(self) -> str:
        """Return the observation's original instant."""
        return self.at

    def monotonic(self) -> float:
        """Keep admission deadlines on this process's monotonic clock."""
        return time.monotonic()


class StoredHermesAdapter(ScopeRecallHermesAdapter):
    """Use native Hermes capture rules with the existing remote command replay fence."""

    def _recording_command(self, context, command: str, turn_id: str) -> None:
        core = self._require_core()
        identifier = hashlib.sha256(f"{context.session_id}:{turn_id}".encode()).hexdigest()
        try:
            private.apply_remote_command(
                core.storage, self._clock, context, command=command, command_id=identifier, observed_at=self._utc_now()
            )
        except ContractError as exc:
            if (exc.code, exc.field) != ("VERSION_CONFLICT", "command_replay"):
                raise


def checked_payload(payload: object) -> dict:
    """Validate fact sizes and method names before any lifecycle side effect."""
    if not isinstance(payload, dict) or len(encode(payload)) > MAX_FACT_BYTES:
        raise HermesRemoteError("invalid Hermes facts")
    binding = payload.get("binding")
    if not isinstance(binding, dict):
        raise HermesRemoteError("Hermes binding is required")
    session = binding.get("session_id")
    if not isinstance(session, str) or not session.strip() or len(session) > 240:
        raise HermesRemoteError("invalid Hermes session")
    for field in ("switches", "events"):
        events = payload.get(field, [])
        if not isinstance(events, list):
            raise HermesRemoteError("invalid Hermes observations")
        for event in events:
            if (
                not isinstance(event, dict)
                or event.get("method") not in METHODS
                or not isinstance(event.get("values"), dict)
            ):
                raise HermesRemoteError("invalid Hermes observation")
            try:
                if datetime.fromisoformat(event["at"].replace("Z", "+00:00")).tzinfo is None:
                    raise ValueError("missing timestamp offset")
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                raise HermesRemoteError("invalid Hermes observation time") from exc
    return payload


def binding_values(binding: dict) -> dict:
    """Translate declared client account facts; the token authenticates the entry, not an OS account."""
    values = facts(binding, binding=True)
    principal = values.pop("local_account_principal", None)
    if principal is not None and (not isinstance(principal, str) or not principal.startswith(("uid:", "sid:"))):
        raise HermesRemoteError("invalid local account principal")
    values["trusted_local_account_principal"] = principal if principal is not None else ""
    return values


def _identity(home: Path, payload: dict):
    binding = payload["binding"]
    identity = bind_hermes_identity(binding["session_id"], hermes_home=str(home), **binding_values(binding))
    for event in payload.get("switches", []):
        values = event["values"]
        identity = switch_hermes_identity(identity, values["new_session_id"], **facts(values, binding=True))
    return identity


def _refusal() -> dict:
    # No private_rules=False: a refused route has no authority to clear the client's known rule history.
    return {"result": {}, "retry": False, "reason": "ungranted_route", "through": None, "terminal": True}


def _source_fence(adapter, payload):
    fence = payload.get("source_fence")
    refs = fence.get("refs") if isinstance(fence, dict) else None
    if (
        isinstance(fence, dict)
        and isinstance(refs, list)
        and len(refs) <= MAX_CURRENT_SOURCE_REFS
        and all(isinstance(ref, str) for ref in refs)
    ):
        adapter._current_source_refs = list(refs)
        adapter._current_source_refs_overflow = fence.get("overflow") is not False
    elif payload.get("active_turn"):
        adapter._current_source_refs_overflow = True
    adapter._active_turn_id = str(payload.get("turn_id") or "")


@contextmanager
def bound_adapter(home: Path, payload: dict, *, replay: bool = True):
    """Rebuild from a bounded turn snapshot, supplying only the registered server home to native code."""
    payload = checked_payload(payload)
    manifest = load_server_entry(home, "hermes")
    core = MemoryCore(CoreConfig(manifest.to_binding()))
    clock = EventClock()
    adapter = StoredHermesAdapter(core=core, clock=clock)
    binding = payload["binding"]
    try:
        adapter.initialize(binding["session_id"], hermes_home=str(home), **binding_values(binding))
        for event in payload.get("switches", []):
            adapter.on_session_switch(**facts(event["values"]))
        for event in payload.get("events", []) if replay else []:
            clock.at = event["at"]
            values = facts(event["values"])
            if event.get("history_field") in {"messages", "conversation_history"}:
                values[event["history_field"]] = payload.get("history", [])
            adapter._recording_command_allowed = event.get("command") in {"pause", "resume"} and event[
                "command"
            ] == private.parse_recording_command(values.get("user_message"))
            getattr(adapter, event["method"])(**values)
        if not replay:
            _source_fence(adapter, payload)
        yield adapter
    finally:
        adapter.shutdown()


def handle_hook(home: Path, body: dict) -> dict:
    """Capture or recall with native Hermes rules; ungranted chats settle without blocking the spool."""
    payload = checked_payload(body.get("payload"))
    try:
        identity = _identity(home, payload)
    except SessionUnbound:
        return _refusal()
    if not identity.runtime_audience.allowed_scope_ids:
        return _refusal()
    read = payload.get("read")
    with bound_adapter(home, payload, replay=not isinstance(read, dict)) as adapter:
        if adapter._unbound_session or not adapter._require_identity().runtime_audience.allowed_scope_ids:
            return _refusal()
        context = adapter.prefetch(**facts(read)) if isinstance(read, dict) else ""
        return {
            "context": context,
            "result": {},
            "retry": bool(adapter._retry.pending_identities()),
            "reason": "hermes",
            "through": None,
            "source_fence": {
                "refs": list(adapter._current_source_refs),
                "overflow": adapter._current_source_refs_overflow,
            },
            "private_rules": core_private(adapter, [])["private_rules"],
        }


def core_private(adapter, times: list) -> dict:
    """Evaluate policy through the native identity's trusted context."""
    return adapter._require_core().private_check(adapter._require_identity().trusted_context(), times)


def _decisions(core, identity, times: list) -> tuple[list, dict]:
    answers, latest = [], {}
    for start in range(0, len(times), 64):
        latest = core.private_check(identity.trusted_context(), times[start : start + 64])
        answers.extend(latest["decisions"])
    return answers, latest


def check_private(home: Path, body: dict) -> dict:
    """Check event times and each historical row's original time without receiving its content."""
    payload = checked_payload(body.get("hermes"))
    try:
        identity = _identity(home, payload)
    except SessionUnbound:
        return _refusal()
    if not identity.runtime_audience.allowed_scope_ids:
        return _refusal()
    core = MemoryCore(CoreConfig(identity.binding))
    original = identity
    decisions = []
    for event in payload.get("events", []):
        if event["method"] == "on_session_switch":
            values = event["values"]
            identity = switch_hermes_identity(identity, values["new_session_id"], **facts(values, binding=True))
        if not identity.runtime_audience.allowed_scope_ids:
            return _refusal()
        context = identity.trusted_context(session_id=event["values"].get("session_id") or identity.session_id)
        decisions.append(core.private_check(context, [event["at"]])["decisions"][0])
    histories = payload.get("history_times", {})
    if not isinstance(histories, dict) or any(not isinstance(times, list) for times in histories.values()):
        raise HermesRemoteError("invalid history times")
    history_decisions = {key: _decisions(core, original, times)[0] for key, times in histories.items()}
    latest = core.private_check(identity.trusted_context(), [])
    return {**latest, "decisions": decisions, "history_decisions": history_decisions}


class HermesMCPServer(MCPServer):
    """Read tools use token-bound grants and declared route metadata, never a conversation body."""

    def __init__(self, home: Path) -> None:
        super().__init__(name="scope-recall-hermes")
        self.home = home

    async def list_tools(self) -> list[Tool]:
        """Advertise the existing Hermes read schemas."""
        return [
            Tool(
                name=row["name"],
                description=row["description"],
                input_schema=row["parameters"],
                annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False),
            )
            for row in tool_schemas()
            if row["name"] in READ_TOOLS
        ]

    async def call_tool(self, name: str, arguments: dict, context=None) -> CallToolResult:
        """Reject mutations and unbound reads before dispatching through the existing tool surface."""
        if name not in READ_TOOLS:
            return CallToolResult(content=[TextContent(type="text", text='{"error":"ACCESS_DENIED"}')], is_error=True)
        metadata = getattr(context.request_context, "meta", None) if context is not None else None
        payload = metadata.get(META_KEY) if isinstance(metadata, dict) else None
        try:
            result = await run_in_threadpool(self._read_tool, checked_payload(payload), name, arguments)
        except (ValueError, HermesIdentityError):
            result = '{"error":"ACCESS_DENIED"}'
        return CallToolResult(content=[TextContent(type="text", text=result)])

    def _read_tool(self, payload: dict, name: str, arguments: dict) -> str:
        with bound_adapter(self.home, payload, replay=False) as adapter:
            return adapter.handle_tool_call(name, arguments)
