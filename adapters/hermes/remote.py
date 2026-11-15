"""Hermes MemoryProvider forwarding bounded turn facts without a local memory database."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from ..clients import remote_client as transport
from ..host_identity import switched_scope
from .boundary import extract_user_text
from .provider import _MemoryProviderBase
from .remote_turn import RemoteTurn, digest
from .remote_wire import META_KEY, READ_TOOLS, facts, scrub_text
from .tool_surface import tool_schemas

CONFIG_NAME = "remote-client.json"


def config_path(home: Path) -> Path:
    """Return the profile-local transport binding, separate from any local store attachment."""
    return home / "scope-recall" / CONFIG_NAME


class RemoteHermesProvider(_MemoryProviderBase):  # pyright: ignore[reportGeneralTypeIssues]
    """Keep independent turn records and forward compact snapshots through the existing remote client."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._binding: dict = {}
        self._route: dict = {}
        self._session = ""
        self._switches: list[dict] = []
        self._turns: dict[str, RemoteTurn] = {}
        self._current: RemoteTurn | None = None
        self._pending: RemoteTurn | None = None
        self._context: ContextVar[RemoteTurn | None] = ContextVar("hermes_remote_turn", default=None)
        self._initialized = False
        self._home: Path | None = None

    @property
    def name(self) -> str:
        """The provider identifier Hermes selects in memory.provider."""
        return "scope-recall"

    def is_available(self) -> bool:
        """Check local configuration only, without opening a database or contacting the service."""
        return self.config["host"] == "hermes" and self.config["token_file"].is_file()

    def initialize(self, session_id: str, **kwargs) -> None:
        """Bind one profile; only its transportable identity facts leave this machine."""
        from .hooks import update_adapter_binding
        from .identity import host_account_principal

        with self._lock:
            home = Path(kwargs["hermes_home"]).resolve()
            if self._home is not None and self._home != home:
                raise ValueError("remote provider belongs to another profile")
            self._home, self._session = home, session_id
            self._binding = {
                "session_id": session_id,
                **facts(kwargs, binding=True),
                "local_account_principal": host_account_principal(),
            }
            self._route = facts(kwargs, binding=True)
            self._switches.clear()
            self._turns.clear()
            self._current = self._pending = None
            self._context.set(None)
            self._initialized = True
        update_adapter_binding(self)

    def matches_hook(self, kwargs: dict) -> bool:
        """Match host-supplied session, platform and sender; profile filtering belongs to the common dispatcher."""
        return (
            self._initialized
            and kwargs.get("session_id") == self._session
            and (not kwargs.get("platform") or kwargs["platform"] == self._route.get("platform", "cli"))
            and (not kwargs.get("sender_id") or kwargs["sender_id"] == self._route.get("user_id"))
        )

    @property
    def hook_route(self) -> tuple:
        """Return a content-free route fingerprint for ambiguity checks."""
        fields = (
            "platform",
            "user_id",
            "user_id_alt",
            "chat_type",
            "chat_id",
            "thread_id",
            "gateway_session_key",
            "agent_identity",
            "agent_workspace",
            "agent_context",
        )
        return str(self.config["config"]), tuple(str(self._route.get(key, "")) for key in fields)

    def _new_turn(self, message: str, key: str | None = None) -> RemoteTurn:
        return RemoteTurn(
            self._binding, self._switches, self._session, message, key or hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        )

    def _remember(self, turn: RemoteTurn) -> None:
        if turn.turn_id:
            self._turns[turn.turn_id] = turn
        while len(self._turns) > 64:
            self._turns.pop(next(iter(self._turns)))

    def remote_hook(self, event: str, kwargs: dict) -> None:
        """Forward observations; tool arguments and tool output are excluded."""
        method = {
            "pre_llm_call": "observe_pre_llm",
            "pre_api_request": "observe_pre_api_request",
            "post_llm_call": "observe_post_llm_call",
            "api_request_error": "observe_api_request_error",
        }.get(event)
        if method:
            self._event(method, kwargs)

    def _event(self, method: str, values: dict) -> dict:
        with self._lock:
            if not self._initialized:
                return {}
            native = str(values.get("turn_id") or "")
            turn = self._turns.get(native) if native else None
            if method == "observe_pre_llm" and turn is None:
                message = extract_user_text(values.get("user_message"))
                if self._current and not self._current.turn_id and self._current.message_digest == digest(message):
                    turn = self._current
                else:
                    turn = self._pending = self._new_turn(message)
            if method in {"sync_turn", "on_pre_compress", "on_session_end"}:
                turn = self._context.get() or turn
            turn = turn or self._current or self._new_turn("")
            turn.observe(method, values, transport._now())
            self._remember(turn)
            if turn.acknowledged == turn.revision:
                return {}
            payload = turn.payload()
        answer = transport.hermes_hook(self.config, payload)
        with self._lock:
            if isinstance(answer.get("source_fence"), dict):
                turn.source_fence = answer["source_fence"]
            if answer.get("terminal"):
                turn.acknowledged = payload["revision"]
            elif type(answer.get("delivered_revision")) is int:
                turn.acknowledged = max(turn.acknowledged, answer["delivered_revision"])
        return answer

    def _payload(self) -> dict:
        """Return the current capture snapshot for explicit replay diagnostics."""
        return self._current.payload() if self._current else {**self._read_payload(), "events": [], "history": []}

    def _read_payload(self) -> dict:
        turn = self._context.get() or self._current
        return copy.deepcopy(
            {
                "hook_event_name": "Hermes",
                "session_id": self._session,
                "binding": self._binding,
                "switches": self._switches,
                "source_fence": turn.source_fence if turn else None,
                "active_turn": turn is not None,
                "turn_id": turn.turn_id if turn else "",
            }
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Allocate a fresh record even if the global pre-hook was skipped; copied contexts retain it."""
        with self._lock:
            pending = self._pending
            # The host calls start after its unclaimed pre-hook. Its redactor may change the text in between.
            turn = (
                pending
                if pending is not None and pending.session == self._session
                else self._new_turn(extract_user_text(message))
            )
            self._pending = None
            self._current = turn
            self._context.set(turn)
            self._remember(turn)
        self._event("on_turn_start", dict(kwargs, turn_number=turn_number, message=message))

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Send only a filtered query and content-free read context, subject to the transport privacy check."""
        with self._lock:
            payload = self._read_payload()
            payload["read"] = {"query": scrub_text(query), "session_id": session_id}
        return str(transport.hermes_hook(self.config, payload, keep=False).get("context") or "")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """The next prefetch performs its bounded request without adding a background task."""

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None, **kwargs
    ) -> None:
        """Forward the completed turn through the record captured in the host's copied context."""
        self._event(
            "sync_turn",
            dict(
                user_content=user_content, assistant_content=assistant_content, session_id=session_id, messages=messages
            ),
        )

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        """Keep switch facts both in this turn's replay order and in subsequent bindings."""
        self._event("on_session_switch", dict(kwargs, new_session_id=new_session_id))
        with self._lock:
            self._route = switched_scope(self._route, facts(kwargs, binding=True))
            previous = self._switches[-1]["values"] if self._switches else self._binding
            self._switches[:] = [
                {
                    "method": "on_session_switch",
                    "at": transport._now(),
                    "values": {
                        **self._route,
                        "new_session_id": new_session_id,
                        "parent_session_id": kwargs.get("parent_session_id") or previous.get("parent_session_id", ""),
                        **{key: kwargs[key] for key in ("reset", "rewound", "reason") if key in kwargs},
                    },
                }
            ]
            self._session = new_session_id

    def on_session_end(self, messages: list[dict]) -> None:
        """Forward remaining host observations at an actual session boundary."""
        self._event("on_session_end", {"messages": messages})

    def on_pre_compress(self, messages: list[dict], **kwargs) -> str:
        """Checkpoint current-turn observations through the ordinary Hermes compression hook."""
        self._event("on_pre_compress", {"messages": messages})
        return ""

    def get_tool_schemas(self) -> list[dict]:
        """Offer existing read tools only; profile context does not attest mutation authorization."""
        return [row for row in tool_schemas() if row["name"] in READ_TOOLS]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Send no conversation body in MCP metadata; the transport verifies read privacy first."""
        if tool_name not in READ_TOOLS or (
            kwargs.get("hermes_home") and Path(kwargs["hermes_home"]).resolve() != self._home
        ):
            return json.dumps({"error": "ACCESS_DENIED"})
        with self._lock:
            meta = {META_KEY: self._read_payload()}
        return transport.hermes_tool(self.config, tool_name, scrub_text(args), meta) or json.dumps(
            {"error": "STORAGE_UNAVAILABLE"}
        )

    def shutdown(self) -> None:
        """Unregister the provider while leaving unacknowledged snapshots in the shared spool."""
        from .hooks import unregister_adapter

        self._initialized = False
        unregister_adapter(self)
        self._context.set(None)
