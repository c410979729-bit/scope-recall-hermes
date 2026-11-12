"""What the Hermes adapter does with a turn: its start, the person's message before the model call (``pre_llm``),
each tool's result, a failed request, what the model showed on the way (``post_llm``), the finished turn
(``sync``), and what the person sent while it ran (``steers``).  It works on the adapter's state under the adapter's
locks (``self._adapter``) and writes through the adapter's ``CaptureWriter``."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scope_recall.contracts import ContractError
from scope_recall.core import private
from scope_recall.core.capture_filters import without_private
from scope_recall.core.deadline import fits

from .audiences import LOCAL_PLATFORMS
from .boundary import (
    UNANSWERED_KIND,
    Steer,
    extract_user_text,
    interim_messages,
    interim_source_event,
    interim_source_key,
    opener_origin,
    person_steers,
    pre_llm_source_event,
    steer_source_event,
    steer_source_key,
    sync_assistant_key,
    sync_opener_origin,
    sync_turn_source_events,
    tool_call_source_event,
    unanswered_messages,
)
from .capture import CAPTURE_TIMEOUT_S, GAP_CURRENT_SOURCE_REFS_LIMIT
from .tool_surface import TOOL_NAMES

if TYPE_CHECKING:
    from .identity import HermesIdentity
    from .provider import ScopeRecallHermesAdapter

#: The adapter's log name, the provider module's, which this module logs under as well: a host writes it into each
#: line, and a logging configuration may name it.
_log = logging.getLogger("scope_recall.adapters.hermes.provider")

#: Recent openers captured, excluded as controls or terminally refused for privacy; retained across compression.
_USER_CAPTURED_TURNS = 64
#: Completed snapshots cover the same recent-turn window as opener deduplication. A skipped memory task must
#: not retain an unbounded conversation; eviction reports a gap rather than silently losing an observation.
_CACHED_TURNS = _USER_CAPTURED_TURNS
#: What one turn showed between tool calls, written message by message after the reply: the first ones are
#: kept, the answer is always written, and a turn past this says so (``capture_gap:interim_limit``).  With no
#: bound a turn of three hundred tool steps would hold the adapter lock for minutes while each waited for the store.
_INTERIM_PER_TURN = 64
#: Steers this process wrote or found stored, by their key: read again at every turn's end, compression and session
#: end, each is looked up and written once.
_STEERS_WRITTEN = 4096
#: Soft store-admission budget for a steer pass. Parsing and lock reacquisition can extend the whole callback;
#: captures without admission time are deferred to the existing retry path.
_STEERS_SECONDS = 2.0


def direct_owner_cli(identity) -> bool:
    """A CLI-shaped gateway route is not proof of the owner's local input."""
    return (
        identity.scope.platform == "cli"
        and not identity.scope.gateway_session_key
        and identity.runtime_audience.includes_owner_private
    )


def steer_deadline() -> float:
    """End of a new steer pass's soft store-admission budget (``_STEERS_SECONDS``)."""
    return time.monotonic() + _STEERS_SECONDS


def _budget(deadline: float) -> float:
    """What one store call may have of a hook's steer budget: a capture's own at most, and nothing once it is spent,
    which the store refuses."""
    return min(CAPTURE_TIMEOUT_S, deadline - time.monotonic())


def _steers(identity, messages: object) -> tuple[Steer, ...]:
    """The steers in ``messages`` that the session's person sent (``person_steers``)."""
    scope = identity.scope
    return person_steers(
        messages,
        platform=scope.platform,
        user_id=scope.user_id,
        local_surface=scope.platform == "cli" or scope.platform in LOCAL_PLATFORMS,
    )


def _opener_origin(identity, history: object, message: object, *, sync: bool = False, staged: bool = False) -> str:
    """Classify the opener, applying the resolved platform's ``person_origin`` to a human result.

    ``staged``: ``history`` is the conversation Hermes hands ``pre_llm_call``, its turn row just appended.
    """
    local = identity.scope.platform == "cli" or identity.scope.platform in LOCAL_PLATFORMS
    if sync:
        origin = sync_opener_origin(history, message, local_surface=local)
    else:
        origin = opener_origin(history, message, local_surface=local, staged=staged)
    return identity.person_origin if origin == "human_direct" else origin


def _shows_rows(history: object) -> bool:
    """Whether a hook's conversation shows any user row: a classification made without one rests on fallbacks alone."""
    return isinstance(history, list) and any(isinstance(row, dict) and row.get("role") == "user" for row in history)


#: Only private chats recover unmarked rows that no turn answered; their authors remain unknown.
_PRIVATE_CHATS = frozenset({"dm", "private", "cli"})


def _said_aside(identity, messages: object) -> tuple[Steer, ...]:
    """The person's steers and, in an attested private chat, unmarked rows no turn answered.

    A restarted gateway restores queued messages as plain rows, but Hermes also writes rows of that shape. Their
    authors are unknown; recognizable Hermes rows are filtered out by ``unanswered_messages``.
    """
    steers = _steers(identity, messages)
    if identity.scope.chat_type not in _PRIVATE_CHATS or identity.person_origin != "human_direct":
        return steers
    return (*steers, *unanswered_messages(messages))


def _is_scope_recall_tool_name(tool_name: object) -> bool:
    """Recognize only names routed by Hermes' registered memory provider.

    Hermes' frozen memory manager builds ``_tool_to_provider`` from each
    provider's returned schemas, rejects duplicate names, and dispatches an
    exact name to that provider.  The post-tool hook supplies no provider
    object, so this exact frozen registry surface is the strongest available
    host identity.  The result body is deliberately never inspected.
    """

    return type(tool_name) is str and tool_name in TOOL_NAMES


#: Hermes' own tools that hand back what was already said or remembered: its search over past
#: sessions and its built-in memory notes.  Their output is recall, not a new observation.  Captured
#: as one, a session search comes back as a page of old conversation, and consolidation would turn
#: it into new facts (six from one search in one store) that then fill the next automatic recall.
#: Like Scope Recall's own output it is kept as a source only.
_HOST_MEMORY_TOOL_NAMES = frozenset({"session_search", "memory"})


def _is_memory_tool_name(tool_name: object) -> bool:
    return _is_scope_recall_tool_name(tool_name) or (type(tool_name) is str and tool_name in _HOST_MEMORY_TOOL_NAMES)


#: Said when Hermes sends an inference request for a turn it opened (``pre_llm_call``) and never started
#: (``on_turn_start``).  A context that still carries the turn before it then hands that turn's binding to the
#: completion that follows, which is stored under the earlier turn or taken for a repeat of it.
TURN_START_MISSING_GAP = "capture_gap:turn_start_missing"


def _text_digest(message: object) -> str:
    """A message's words, whitespace aside: the start receives flattened text, the hooks the content parts."""
    return hashlib.sha256(" ".join(extract_user_text(message).split()).encode()).hexdigest()


@dataclass
class _TurnBinding:
    """Content-free turn identity carried by Hermes' copied background context.

    Only the inline host start creates this object. The turn's own request corrects its UUID in place so copied
    contexts share the correction; ``started_with`` (``_text_digest``) is how that request is known.  Starting
    another turn leaves queued work's object intact.
    """

    identity: HermesIdentity
    turn_id: str
    sequence: int
    started_with: str
    confirmed: bool = False
    delivery: str | None = None


@dataclass(frozen=True)
class _Opening:
    """The latest accepted opening hook and its classification when admission was skipped."""

    turn_id: str
    skipped: bool
    origin: str


@dataclass(frozen=True)
class _CompletedTurn:
    """The original identity, session and start order of both completed-message caches."""

    identity: HermesIdentity
    session_id: str
    turn_id: str
    sequence: int


class TurnCapture:
    """A Hermes adapter's turns; one per adapter.

    Each adapter owns its ContextVar so nested providers in one host context cannot replace each other's turn.
    A long-lived context retains that variable and its last binding, including the immutable identity. Shutdown
    clears the calling context; copies already queued by the host retain their bindings until those copies end.
    """

    def __init__(self, adapter: ScopeRecallHermesAdapter) -> None:
        self._adapter = adapter
        self._sync_binding: ContextVar[_TurnBinding | None] = ContextVar("hermes_capture_turn", default=None)
        self._current_binding: _TurnBinding | None = None
        self._sequence = 0
        self._known_turns: dict[str, _TurnBinding] = {}
        self._completed: dict[str, _CompletedTurn] = {}
        self._opening: _Opening | None = None
        self._opening_count = 0
        self._last_start: int | None = None
        self.capture_gaps: tuple[str, ...] = ()

    # All turn identity state uses _said_lock. A delivered or skipped opening replaces the pending record;
    # replay of the latest or already started UUID contributes no additional input to the host ordinal count.
    # Only delivered openings change active state. Known older openings cannot change either. Start always clears
    # the record and creates the sole context binding; claiming requires ordinals to account for every opener.
    # The first start may claim without an ordinal baseline, but skipped text needs ordinal or explicit UUID proof.
    # A request can confirm only the current, unconfirmed object carried by its host context, and give it another
    # UUID only with the text the turn started with; any other request writes nothing and changes no state, and one
    # for an opening no start claimed is reported. Post always witnesses time before applying that proof and
    # caches bodies only for the active UUID. Sync snapshots its carried binding, accepting one completion
    # payload per object; a missing or reused binding uses the active baseline identity. Sync alone drains
    # completed bodies with their original metadata. Reset clears caller state, leaving queued copies intact.
    def bind_turn(self, turn_id: str, message: object, *, confirmed: bool = False) -> None:
        """Bind the inline start's caller under the observation lock; only the host start creates bindings."""
        self._sequence += 1
        self._current_binding = _TurnBinding(
            self._adapter._require_identity(), turn_id, self._sequence, _text_digest(message), confirmed
        )
        self._remember_binding(turn_id, self._current_binding)
        self._sync_binding.set(self._current_binding)

    def reset_binding(self) -> None:
        """Clear the caller's binding on a fresh session without altering queued contexts."""
        with self._adapter._said_lock:
            self._current_binding = None
            self._sync_binding.set(None)
            self._known_turns.clear()
            self._opening = None
            self._opening_count = 0
            self._last_start = None

    def _remember_binding(self, turn_id: str, binding: _TurnBinding) -> None:
        """Keep a bounded witness of UUIDs already claimed by a started turn, including corrected aliases."""
        self._known_turns[turn_id] = binding
        while len(self._known_turns) > _USER_CAPTURED_TURNS:
            self._known_turns.pop(next(iter(self._known_turns)))

    def _late_opening(self, turn_id: str) -> bool:
        """Recognize recent older opening UUIDs; worker contexts cannot establish a new opener's order."""
        known = self._known_turns.get(turn_id)
        return known is not None and (known is not self._current_binding or known.turn_id != turn_id)

    def note_skipped(self, kwargs: dict) -> None:
        """Keep the skipped UUID and origin atomically without retaining another user body."""
        turn_id = str(kwargs.get("turn_id") or "").strip()
        history = kwargs.get("conversation_history")
        origin = _opener_origin(self._adapter._require_identity(), history, kwargs.get("user_message"), staged=True)
        with self._adapter._said_lock:
            if not turn_id or self._late_opening(turn_id):
                return
            self._note_opening(turn_id, skipped=True, origin=origin)
            if _shows_rows(history):
                self._note_origin(turn_id, origin, kwargs.get("user_message"))

    def _note_opening(self, turn_id: str, *, skipped: bool, origin: str) -> None:
        """Replace the candidate under the observation lock; latest or already started openings are replays."""
        binding = self._current_binding
        if (binding is None or binding.turn_id != turn_id) and (
            self._opening is None or self._opening.turn_id != turn_id
        ):
            self._opening_count += 1
        self._opening = _Opening(turn_id, skipped, origin)

    def _request_binding(self, turn_id: str, message: object) -> _TurnBinding | None:
        """The started turn's binding when a request hook is that turn's own, settling its UUID; else None.

        Hermes sends the request hooks for every inference made under the session.  A review it runs after a
        reply (``agent/background_review.py``) shares the session, copies the finished turn's context and has a
        turn id of its own, with no opening hook and no start.  So the carried object says nothing about an unknown
        UUID: a turn not yet confirmed takes one only from a request with the text it started with.  Hermes scrubs
        secrets from the start's text alone; such a turn keeps the id its start chose.
        """
        with self._adapter._said_lock:
            binding = self._current_binding
            if not turn_id or binding is None or self._sync_binding.get() is not binding:
                return None
            if binding.turn_id != turn_id:
                if binding.confirmed or binding.started_with != _text_digest(message):
                    return None
                self.reset_source_refs()
                binding.turn_id = turn_id
            if not binding.confirmed:
                binding.confirmed = True
                self._remember_binding(turn_id, binding)
            self._adapter._active_turn_id = turn_id
            return binding

    def note_opener(
        self, turn_id: str, history: object, user_message: object, identity, *, staged: bool = False
    ) -> str:
        """Classify the opener and remember it for later callbacks when the conversation showed the turn's rows."""
        origin = _opener_origin(identity, history, user_message, staged=staged)
        if _shows_rows(history):
            with self._adapter._said_lock:
                self._note_origin(turn_id, origin, user_message)
        return origin

    def _note_origin(self, turn_id: str, origin: str, message: object) -> None:
        """Retain the classification and up to four text variants under the observation lock.

        ``pre_llm_call`` sees the turn's row just staged, before Hermes joins rows and compacts the conversation, when
        its evidence is clearest; later callbacks see redacted text, joined rows, steers and scaffolding.  A callback
        whose text is one of the turn's remembered texts reuses the classification.  Another classification of the
        same turn replaces it; a turn pushed out by newer ones, or forgotten at a session switch, is classified again
        from what its callback sees.  Only classifications made with the conversation's rows in view are kept: one
        made without them rests on the fallback rules and must not outweigh what a later callback sees.
        """
        previous = self._adapter._opener_origins.pop(turn_id, None)
        text = extract_user_text(message).strip()
        texts = previous[0] if previous and previous[1] == origin else ()
        if text not in texts:
            texts = (*texts, text)[-4:]
        self._adapter._opener_origins[turn_id] = (texts, origin)
        while len(self._adapter._opener_origins) > _USER_CAPTURED_TURNS:
            self._adapter._opener_origins.pop(next(iter(self._adapter._opener_origins)))

    def remembered_origin(self, turn_id: str, user_content: str) -> str | None:
        """Return a remembered origin only for one of its observed opening texts."""
        with self._adapter._said_lock:
            opener = self._adapter._opener_origins.get(turn_id)
        return opener[1] if opener is not None and user_content.strip() in opener[0] else None

    def start(self, turn_number: int, message: str, **kwargs) -> None:
        """Associate the inline host start with its pre-hook UUID and recover a busy hook's input."""
        with self._adapter._said_lock:
            number = int(turn_number)
            opening, count = self._opening, self._opening_count
            accounted = count == number - (self._last_start or 0)
            claimed = opening if self._last_start is None or accounted else None
            self._opening, self._opening_count, self._last_start = None, 0, number
            self._adapter._turn_counter = number
            explicit = str(kwargs.get("turn_id") or "")
            turn_id = explicit or (claimed.turn_id if claimed else "") or str(number)
            if turn_id != self._adapter._active_turn_id:
                self.reset_source_refs()
            self._adapter._active_turn_id = turn_id
            self.bind_turn(turn_id, message, confirmed=bool(explicit))
            skipped = bool(claimed and claimed.skipped and claimed.turn_id == turn_id)
            recover = skipped and (accounted or explicit == turn_id)
            origin = claimed.origin if claimed and skipped else "human_direct"
            if skipped and (recover or origin != "human_direct"):
                # Host redaction may change the body between the hook and the inline callback, so the start's text is
                # remembered as the turn's.  A person's origin needs the start proven to be that turn: another text
                # would otherwise take it.
                self._note_origin(turn_id, origin, message)
        session_id = self._adapter._effective_session_id(str(kwargs.get("session_id") or ""))
        with self._adapter._said_lock:
            pending = self._adapter._pending_private_receipt
            stored_session = self._adapter._require_identity().stored_session_id(session_id)
            if pending is not None and pending[0] == stored_session:
                removed = self._adapter._private_span_receipts.pop(pending, 0)
                if removed:
                    self._adapter._private_span_receipts[(stored_session, turn_id)] = removed
                self._adapter._pending_private_receipt = None
        if type(message) is str and message:
            self._adapter._current_task_message = without_private(message)[0][:8192]
        self._adapter._outcomes.open_turn(session_id, turn_id)
        if recover:
            self._capture_opener(
                dict(session_id=session_id, turn_id=turn_id, user_message=message), origin=origin, current=True
            )

    def pre_api(self, **kwargs) -> bool:
        """Recover the started turn's missing opener before inference, under the request's UUID; True for that turn.

        Any other request writes nothing (``_request_binding``): a review's prompt would be stored as the person's
        words.  An earlier turn's late request loses nothing by it, its completion writes the message.
        """
        turn_id = str(kwargs.get("turn_id") or "").strip()
        binding = self._request_binding(turn_id, kwargs.get("user_message"))
        if binding is None:
            with self._adapter._said_lock:
                if turn_id and self._opening is not None and self._opening.turn_id == turn_id:
                    self._note_gap(TURN_START_MISSING_GAP)
            return False
        origin = self.remembered_origin(turn_id, extract_user_text(kwargs.get("user_message"))) or self.note_opener(
            turn_id, kwargs.get("conversation_history"), kwargs.get("user_message"), binding.identity
        )
        self._capture_opener(kwargs, origin=origin, current=True, bound=binding.identity)
        return True

    def pre_llm(self, **kwargs) -> bool:
        """Observe an opener UUID without binding the helper thread's context to a new turn."""
        with self._adapter._said_lock:
            supplied = str(kwargs.get("turn_id") or "").strip()
            turn_id = supplied or self._adapter._active_turn_id or str(self._adapter._turn_counter or "turn")
            current = not self._late_opening(turn_id)
            if current:
                if turn_id != self._adapter._active_turn_id:
                    self.reset_source_refs()
                self._adapter._active_turn_id = turn_id
                self._note_opening(supplied, skipped=False, origin="human_direct")
            carried = self._sync_binding.get()
            identity = carried.identity if not current and carried else self._adapter._require_identity()
        if current:
            session = self._adapter._effective_session_id(str(kwargs.get("session_id") or ""))
            self._adapter._outcomes.open_turn(session, turn_id)
            message = kwargs.get("user_message")
            if type(message) is str and message:
                self._adapter._current_task_message = without_private(message)[0][:8192]
            origin = self.remembered_origin(turn_id, extract_user_text(message)) or self.note_opener(
                turn_id, kwargs.get("conversation_history"), message, identity, staged=True
            )
        else:
            origin = _opener_origin(
                identity, kwargs.get("conversation_history"), kwargs.get("user_message"), staged=True
            )
        self._capture_opener(dict(kwargs, turn_id=turn_id), origin=origin, current=current, bound=identity)
        return current

    def _capture_opener(self, kwargs: dict, *, origin: str, current: bool, bound=None) -> None:
        """Store a delivered opener under its UUID; late delivery cannot alter active bookkeeping."""
        identity = bound or self._adapter._require_identity()
        turn_id = str(kwargs.get("turn_id") or "turn")
        with self._adapter._said_lock:
            if turn_id in self._adapter._user_captured_turns:
                return
        if identity.read_only or not identity.runtime_audience.allowed_scope_ids:
            self._adapter._diagnostics.capability_gaps = identity.runtime_audience.capability_gaps
            return
        session_id = str(kwargs.get("session_id") or identity.session_id)
        current_message = kwargs.get("user_message")
        context = identity.trusted_context(session_id=session_id, actor_origin=origin, mutation=True)
        if (
            current
            and direct_owner_cli(identity)
            and context.actor_origin == "human_direct"
            and private.parse_recording_command(current_message)
            and kwargs.get("attachments") in (None, [])
        ):
            core = self._adapter._require_core()
            if private.parse_recording_command(current_message) == "pause":
                private.open_interval(core.storage, core.clock, context, target="session", session=context.session_id)
            else:
                private.close_interval(core.storage, core.clock, context, target="session", session=context.session_id)
            with self._adapter._said_lock:
                self._adapter._user_captured_turns[turn_id] = None
            return
        event, gaps, ledger_identity = pre_llm_source_event(
            self._adapter._ledger,
            context,
            session_id=session_id,
            turn_id=turn_id,
            user_message=kwargs.get("user_message"),
            recorded_at=self._adapter._utc_now(),
            attachments=kwargs.get("attachments") if isinstance(kwargs.get("attachments"), list) else None,
        )
        if event is None and not gaps:
            return
        receipt = self._adapter._writer.write(
            context,
            event,
            identity=ledger_identity,
            gaps=gaps,
            scope_id=identity.local_scope_id,
            bound=identity,
            opener_turn_id=turn_id,
        )
        with self._adapter._said_lock:
            if (
                current
                and receipt is not None
                and any(gap.startswith("private_spans_removed:") for gap in receipt.gaps)
            ):
                self._adapter._pending_private_receipt = (context.session_id, turn_id)
            if receipt is not None and (
                receipt.durability in ("persisted", "queued") or receipt.error_code == "PRIVATE_MODE"
            ):
                # An explicit privacy refusal settles this opener just as a durable capture does. Its original
                # ledger witness remains, while temporary failures may be retried by the next request.
                self._adapter._user_captured_turns.pop(turn_id, None)
                self._adapter._user_captured_turns[turn_id] = None
                while len(self._adapter._user_captured_turns) > _USER_CAPTURED_TURNS:
                    self._adapter._user_captured_turns.pop(next(iter(self._adapter._user_captured_turns)))

    def tool_result(self, **kwargs) -> None:
        """Capture one tool result; the caller holds ``_lock`` exactly once, and the store I/O runs without it.

        Hermes calls the hook for each of a step's parallel tool calls at once.  Held across its write (1.4-4.4 s on
        a busy shared store), one capture would keep the others waiting, and those past the hook's bound would not
        be taken.

        A call that failed is kept as well, as Codex's are.  Hermes calls a result failed for a non-zero exit code or
        an error field, and what such a call printed (a traceback, a failing test) is what the agent saw and acted
        on; dropped as having no scope, such results were about 6% of the tool results of five installations, each
        logged as a failed capture.  It is stored ``partial``, which also keeps it from ending its task
        (``core/episodes.py``).
        """
        identity = self._adapter._require_identity()
        if identity.read_only or not identity.runtime_audience.allowed_scope_ids:
            self._adapter._diagnostics.capability_gaps = identity.runtime_audience.capability_gaps
            return
        session_id = self._adapter._effective_session_id(str(kwargs.get("session_id") or ""))
        turn_id = str(kwargs.get("turn_id") or self._adapter._active_turn_id or "turn")
        tool_call_id = str(kwargs.get("tool_call_id") or kwargs.get("id") or turn_id)
        tool_name = str(kwargs.get("tool_name") or kwargs.get("name") or "tool")
        result = kwargs.get("result") if "result" in kwargs else kwargs.get("content")
        status = str(kwargs.get("status") or kwargs.get("outcome") or "success").lower()
        outcome = "success"
        if status in {"error", "failed", "failure"}:
            outcome = "failure"
            self._adapter._outcomes.mark_failure(session_id, turn_id, reason=status)
        elif status in {"cancelled", "canceled"}:
            outcome = "cancelled"
            self._adapter._outcomes.mark_cancelled(session_id, turn_id)
        elif status in {"interrupted"}:
            outcome = "interrupted"
            self._adapter._outcomes.mark_interrupted(session_id, turn_id)
        elif result is None and "result" not in kwargs and "content" not in kwargs:
            outcome = "truncated"
            self._adapter._outcomes.mark_truncated(session_id, turn_id)
        is_memory_tool = _is_memory_tool_name(tool_name)
        captured_origin = "memory_reinjection" if is_memory_tool else "tool_observation"
        context = identity.trusted_context(session_id=session_id, actor_origin=captured_origin, mutation=True)
        event, gaps, ledger_identity = tool_call_source_event(
            self._adapter._ledger,
            context,
            session_id=session_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
            recorded_at=self._adapter._utc_now(),
            outcome=outcome,
            origin=captured_origin,
        )
        self._adapter._writer.write(
            context,
            event,
            identity=ledger_identity,
            gaps=gaps,
            scope_id=identity.local_scope_id,
            bound=identity,
            release=True,
        )
        self._adapter._diagnostics.pending_outcome_gaps = self._adapter._outcomes.pending_gaps()

    def request_error(self, **kwargs) -> None:
        self._adapter._require_identity()
        session_id = self._adapter._effective_session_id(str(kwargs.get("session_id") or ""))
        turn_id = str(kwargs.get("turn_id") or self._adapter._active_turn_id or "turn")
        status = str(kwargs.get("status") or kwargs.get("status_code") or "error")
        self._adapter._outcomes.mark_failure(session_id, turn_id, reason=status)
        self._adapter._diagnostics.pending_outcome_gaps = self._adapter._outcomes.pending_gaps()

    def post_llm(self, **kwargs) -> None:
        """Keep what the assistant showed on the way through this turn for ``sync_turn`` to record.

        Hermes calls this once, when a turn that has an answer ends, with a copy of the conversation, and
        before it sends the reply.  It reads that copy and writes nothing, so it takes only ``_said_lock``: under
        the adapter lock the reply would wait behind whatever held it, a capture on a busy store or a recall, and
        a callback Hermes gives up on is then skipped for a minute, for every session (Hermes 0.21.5).
        """
        turn_id = str(kwargs.get("turn_id") or "").strip()
        if not turn_id:
            return
        with self._adapter._said_lock:
            carried = self._sync_binding.get()
            identity = carried.identity if carried else self._adapter._require_identity()
            sequence = carried.sequence if carried else self._sequence
        if identity.read_only or not identity.runtime_audience.allowed_scope_ids:
            return
        answer = kwargs.get("assistant_response")
        history = kwargs.get("conversation_history")
        observed_at = self._adapter._utc_now()
        session = str(kwargs.get("session_id") or identity.session_id)
        context = identity.trusted_context(session_id=session)
        # The ledger keeps only identity and first-observation times. Even a
        # refused reply needs that witness so delayed sync cannot date it anew.
        if isinstance(answer, str) and answer.strip():
            key = sync_assistant_key(context, session, turn_id)
            self._adapter._ledger.witness((key, 1), observed_at, observed_at)
        self._request_binding(turn_id, kwargs.get("user_message"))
        # Conversation history is retrospective. Only a host timestamp or an
        # earlier witness establishes when its message occurred.
        said = tuple(
            (
                text,
                self._adapter._ledger.witness(
                    (interim_source_key(context, session, turn_id, ordinal), 1), observed_at, moment
                )[1],
            )
            for ordinal, (text, moment) in enumerate(
                interim_messages(history, answer=answer if isinstance(answer, str) else ""), start=1
            )
        )
        steered = tuple(
            steer._replace(
                occurred_at=self._adapter._ledger.witness(
                    (steer_source_key(context, steer), 1), observed_at, steer.occurred_at
                )[1]
            )
            for steer in _steers(identity, history)
        )
        try:
            check = self._adapter._require_core().private_check
            moments = [moment for _text, moment in said] + [steer.occurred_at for steer in steered]
            # Keep no body if any of this observation is private or unverified.
            refused = any(
                check(context, moments[start : start + 64])["capture_blocked"]
                for start in range(0, max(1, len(moments)), 64)
            )
        except (ContractError, OSError, RuntimeError, sqlite3.Error):
            refused = True
        if refused:
            return
        with self._adapter._said_lock:
            if turn_id != self._adapter._active_turn_id:
                return
            if said:
                self._adapter._interim_said[turn_id] = said
            if steered:
                self._adapter._steer_said[turn_id] = steered
            if said or steered:
                self._completed[turn_id] = _CompletedTurn(identity, session, turn_id, sequence)
                while len(self._completed) > _CACHED_TURNS:
                    oldest = min(self._completed, key=lambda key: self._completed[key].sequence)
                    self._completed.pop(oldest)
                    self._adapter._interim_said.pop(oldest, None)
                    self._adapter._steer_said.pop(oldest, None)
                    self._note_gap("capture_gap:completed_turn_cache_limit")

    def _note_gap(self, gap: str) -> None:
        """Keep a turn diagnostic across outcome updates; the caller holds the observation lock."""
        self.capture_gaps = tuple(dict.fromkeys((*self.capture_gaps, gap)))
        self._adapter._merge_gaps((gap,))

    def clear_completed(self) -> None:
        """Clear both message caches and their owners; the caller holds the small observation lock."""
        self._completed.clear()
        self._adapter._interim_said.clear()
        self._adapter._steer_said.clear()

    def flush_completed(self, *, through: int | None = None) -> None:
        """Write completed snapshots in start order under their original identities.

        A bound sync passes its start sequence so later turns remain untouched; an unbound sync drains all. The caller
        holds the sync lock, without an outer adapter-lock hold, so per-message writes may release that lock.
        """
        adapter = self._adapter
        with adapter._said_lock:
            ready = sorted(
                (item for item in self._completed.values() if through is None or item.sequence <= through),
                key=lambda item: item.sequence,
            )
        for item in ready:
            with adapter._said_lock:
                if self._completed.get(item.turn_id) is not item:
                    continue
                self._completed.pop(item.turn_id)
                said = adapter._interim_said.pop(item.turn_id, ())
                steers = adapter._steer_said.pop(item.turn_id, ())
            identity = item.identity
            shown = identity.trusted_context(
                session_id=item.session_id, actor_origin="assistant_visible", mutation=True
            )
            limited = self._write_messages(
                interim_source_event, {item.turn_id: said}, identity, shown, item.session_id, limit=_INTERIM_PER_TURN
            )
            # Steers Hermes marked as such, each behind an origin it wrote naming the session's person
            # (``person_steers``): their own words.
            context = identity.trusted_context(
                session_id=item.session_id, actor_origin=identity.person_origin, mutation=True
            )
            self._write_steers(steers, identity, context, "completed_turn")
            if limited:
                with adapter._lock, adapter._said_lock:
                    self._note_gap("capture_gap:interim_limit")

    def sync(self, user_content: str, assistant_content: str, *, session_id: str, messages: object = None) -> None:
        """Write the finished turn one capture per hold of the lock, each one's store I/O without it.

        Hermes runs this on its memory worker after the reply, with no time limit, while the next turn may already
        start.  Held for the whole turn (up to 64 interim messages, the steers, the reply and the retries, each
        waiting up to 1 s for a busy store) it would keep the next turn's hooks, its start and its prefetch waiting.
        The captures keep the binding the turn was said under (``bound``).  One turn is written at a time, and a
        shutdown waits for it (the adapter's ``_sync_lock``): one that came between two captures would close the
        runtime under the rest of the turn, the reply included.
        """
        adapter = self._adapter
        with adapter._lock:
            with adapter._said_lock:
                binding = self._sync_binding.get()
                # This digest only detects reuse of a stale context object. It never becomes an event key or
                # a time witness: missing starts retain the host adapter's baseline fallback semantics.
                delivery = hashlib.sha256(
                    json.dumps([session_id, user_content, assistant_content]).encode()
                ).hexdigest()
                if binding is not None and binding.delivery not in (None, delivery):
                    binding = None
                if binding is not None:
                    binding.delivery = delivery
                identity = binding.identity if binding else adapter._require_identity()
                turn_id = (
                    binding.turn_id if binding else adapter._active_turn_id or str(adapter._turn_counter or "turn")
                )
                through = binding.sequence if binding else None
                user_captured = turn_id in adapter._user_captured_turns
            if identity.read_only:
                return
            # The turn's message and reply are dated when its writing begins: dated as each is reached, the reply
            # could come after the next turn's message, written between this turn's captures.
            said_at = adapter._utc_now()
            effective_session = session_id or identity.session_id
            if not identity.runtime_audience.allowed_scope_ids:
                adapter._diagnostics.capability_gaps = identity.runtime_audience.capability_gaps
                return
        # The context of the person's steers and redirects (``_write_aside``): their own words.
        context = identity.trusted_context(
            session_id=effective_session, actor_origin=identity.person_origin, mutation=True
        )
        shown = identity.trusted_context(session_id=effective_session, actor_origin="assistant_visible", mutation=True)
        origin = self.remembered_origin(turn_id, user_content)
        if origin is None:
            # A captured opener is not written again; this context only matters for an uncaptured opener.
            origin = (
                identity.person_origin if user_captured else _opener_origin(identity, messages, user_content, sync=True)
            )
        opened = identity.trusted_context(session_id=effective_session, actor_origin=origin, mutation=True)
        self.flush_completed(through=through)
        self._write_aside(_said_aside(identity, messages), identity, context, effective_session, "sync_turn")
        self._write_turn(identity, opened, shown, effective_session, turn_id, user_content, assistant_content, said_at)
        # Hermes runs this on its memory worker, after the reply: the place to write again what a busy store kept in
        # memory, once the turn's own sources are written.  Before them, a pass over a store that cannot take the
        # buffer would hold the turn's message and reply back by its whole time.
        adapter._retry.write_buffered(release=True)
        with adapter._lock, adapter._said_lock:
            adapter._diagnostics.pending_outcome_gaps = adapter._outcomes.pending_gaps()
            adapter._merge_gaps(self.capture_gaps)

    def _write_messages(self, make_event, messages: dict, identity, context, session_id: str, *, limit=None) -> bool:
        """Each message of each turn (what the model showed between tool calls, or what the person sent while the turn
        ran), one capture per hold of the lock; with ``limit``, a turn's first messages only.  Whether a turn had
        more than ``limit``."""
        adapter = self._adapter
        limited = False
        for said_turn, said in messages.items():
            if limit is not None and len(said) > limit:
                _log.warning(
                    "scope-recall: turn %s showed %d messages between tool calls; the first %d are kept",
                    said_turn,
                    len(said),
                    limit,
                )
                limited, said = True, said[:limit]
            for ordinal, (text, occurred_at) in enumerate(said, 1):
                with adapter._lock, adapter._calls.holding("sync_turn"):
                    event, gaps, ledger_identity = make_event(
                        adapter._ledger,
                        context,
                        session_id=session_id,
                        turn_id=said_turn,
                        ordinal=ordinal,
                        content=text,
                        recorded_at=adapter._utc_now(),
                        occurred_at=occurred_at,
                    )
                    if event is not None or gaps:
                        adapter._writer.write(
                            context,
                            event,
                            identity=ledger_identity,
                            gaps=gaps,
                            scope_id=identity.local_scope_id,
                            bound=identity,
                            release=True,
                        )
        return limited

    def steers(
        self,
        messages: object,
        *,
        session_id: str = "",
        hook: str,
        deadline: float | None = None,
        bound=None,
        typed_only: bool = False,
    ) -> None:
        """Write now what the person sent while turns ran (``person_steers``) in ``messages``, and the unmarked rows
        no turn answered (``_said_aside``), whose authors are unknown: a compression is
        about to take them out of the conversation, or the session ends, and no turn's end may come to read them.
        ``deadline`` is the hook's, when its other captures share the budget (``steer_deadline``).  The caller must
        not hold the adapter's lock: the store I/O runs without it. ``typed_only`` excludes untyped historical
        recovery from request hooks. ``bound`` keeps the identity selected before releasing the admission lock.
        """
        adapter = self._adapter
        with adapter._lock:
            identity = bound or adapter._identity
            if identity is None or identity.read_only or not identity.runtime_audience.allowed_scope_ids:
                return
            effective_session = session_id or identity.session_id
        found = _steers(identity, messages) if typed_only else _said_aside(identity, messages)
        if found:
            # The context of the person's steers and redirects; unanswered rows get one of their own (``_write_aside``).
            context = identity.trusted_context(
                session_id=effective_session, actor_origin=identity.person_origin, mutation=True
            )
            self._write_aside(found, identity, context, effective_session, hook, deadline)

    def _write_aside(self, found, identity, context, session_id: str, hook: str, deadline: float | None = None) -> None:
        """Write attested steers and unattributed history with separate contexts and one admission budget."""
        deadline = steer_deadline() if deadline is None else deadline
        delivered = tuple(steer for steer in found if steer.kind != UNANSWERED_KIND)
        unanswered = tuple(steer for steer in found if steer.kind == UNANSWERED_KIND)
        if delivered:
            self._write_steers(delivered, identity, context, hook, deadline)
        if unanswered:
            unknown = identity.trusted_context(session_id=session_id, actor_origin="origin_unknown", mutation=True)
            self._write_steers(unanswered, identity, unknown, hook, deadline)

    def _write_steers(self, steers, identity, context, hook: str, deadline: float | None = None) -> None:
        """Each steer the store does not hold yet (``_held``), one capture per hold of the lock, its store I/O
        without it, all by the hook's ``deadline``: one it has no time left for is kept to retry with no store call
        (``CaptureWriter.defer``).  Only a write the store took marks one written: a write still in
        flight in another hook may fail, and one the retry buffer kept may be given up.  The next hook reads either
        again; meanwhile the observation ledger and the retry buffer keep it from being written twice."""
        adapter = self._adapter
        deadline = steer_deadline() if deadline is None else deadline
        # One message read by two routes (what post_llm_call kept, the conversation sync_turn is handed) is one,
        # with what either read said alike: the two may hold different stretches of the conversation.
        reads: dict[str, tuple[Steer, str]] = {}
        for steer in steers:
            key = steer_source_key(context, steer)
            if key in adapter._steers_written:
                continue
            if key in reads:
                alike = set(reads[key][0].alike) | set(steer.alike)
                steer = reads[key][0]._replace(
                    alike=tuple(sorted(alike, key=lambda moment: (moment is None, moment or "")))
                )
            reads[key] = (steer, key)
        unread = list(reads.values())
        if not unread:
            return
        began = time.monotonic()
        guard = adapter._writer.capture_guard(
            context,
            _budget(deadline),
            observations=[
                (
                    context,
                    [{"occurred_at": steer.occurred_at, "recorded_at": adapter._utc_now()} for steer, _key in unread],
                )
            ],
        )
        took = time.monotonic() - began
        with adapter._writer.using_capture_guard(guard):
            for (steer, key), held in zip(unread, self._held(unread, identity, context, deadline), strict=True):
                if held is None:
                    # Without a stable original key, an unreadable lookup cannot
                    # distinguish new input from a source already held or deleted.
                    continue
                with adapter._lock, adapter._calls.holding(hook):
                    taken = held
                    # Kept to retry: the buffer writes it, or gives it up and the next hook reads it again.
                    if not held and (key, 1) not in adapter._retry.captures:
                        event, gaps, ledger_identity = steer_source_event(
                            adapter._ledger, context, steer=steer, recorded_at=adapter._utc_now()
                        )
                        if event is None:
                            # Being stored by another hook, or the ledger is full for now.
                            if gaps:
                                adapter._merge_gaps(gaps)
                        elif not fits(deadline, took):
                            # The hook's time is spent, or what is left would not hold another write as long as the last
                            # (``fits``): kept to retry with no store call.
                            adapter._writer.defer(
                                context,
                                event,
                                identity=ledger_identity,
                                gaps=gaps,
                                scope_id=identity.local_scope_id,
                                bound=identity,
                                guard=guard,
                            )
                        else:
                            began = time.monotonic()
                            receipt = adapter._writer.write(
                                context,
                                event,
                                identity=ledger_identity,
                                gaps=gaps,
                                scope_id=identity.local_scope_id,
                                remaining_seconds=_budget(deadline),
                                bound=identity,
                                release=True,
                            )
                            took = time.monotonic() - began
                            taken = receipt is not None
                    if taken:
                        adapter._steers_written[key] = None
                        while len(adapter._steers_written) > _STEERS_WRITTEN:
                            adapter._steers_written.pop(next(iter(adapter._steers_written)))

    def _held(self, unread, identity, context, deadline: float) -> tuple[bool | None, ...]:
        """Which steers or unattributed historical rows the store holds already or holds deleted: a compression
        carries a steer into the session it continues, and a process that starts again remembers none it wrote.
        When the store cannot say, typed steers may retry under their stable keys. Untyped historical input
        returns None and waits for a later observation, since its original key and deletion state are unknown.

        A message no turn answered may still be a turn's own, stored by ``pre_llm_call`` under the turn's key when the
        gateway stopped before any reply: one of those is looked up by its words and moment in this session as well.
        Every message of the conversation said with the same words (``Steer.alike``) is looked up with it, once, and
        the store pairs the closest message and copy first: the copy of one a turn answered, or of one held already,
        is not taken for another."""
        if not unread:
            return ()
        core = self._adapter._require_core()
        try:
            held = list(
                core.said_in_session(
                    context,
                    identity.local_scope_id,
                    [("user", steer.words, steer.occurred_at or "", key) for steer, key in unread],
                    remaining_seconds=_budget(deadline),
                    across_sessions=True,
                )
            )
            loose = [index for index, (steer, _key) in enumerate(unread) if steer.kind == UNANSWERED_KIND]
            loose = [index for index in loose if index < len(held) and not held[index]]
            if loose:
                items, owners = [], []
                for words in dict.fromkeys(unread[index][0].words for index in loose):
                    asked = {unread[index][0].occurred_at: index for index in loose if unread[index][0].words == words}
                    moments = set(asked).union(*(unread[index][0].alike for index in asked.values()))
                    for moment in sorted(moments, key=lambda moment: (moment is None, moment or "")):
                        items.append(("user", words, moment or "", None))
                        owners.append(asked.get(moment))
                worded = core.said_in_session(
                    context, identity.local_scope_id, items, remaining_seconds=_budget(deadline)
                )
                for owner, said in zip(owners, worded, strict=True):
                    if owner is not None:
                        held[owner] = said
        except (ContractError, OSError, RuntimeError, sqlite3.Error, ValueError):
            return tuple(None if steer.kind == UNANSWERED_KIND else False for steer, _key in unread)
        return (
            tuple(held)
            if len(held) == len(unread)
            else tuple(None if steer.kind == UNANSWERED_KIND else False for steer, _key in unread)
        )

    def _write_turn(
        self, identity, opened, shown, session_id: str, turn_id: str, user_content: str, assistant_content: str, said_at
    ) -> None:
        """The turn's message (unless ``pre_llm`` stored it) and its reply, with the turn's outcome."""
        adapter = self._adapter
        outcome = "success"
        with adapter._lock, adapter._said_lock:
            if not assistant_content.strip():
                outcome = "truncated"
                adapter._outcomes.mark_truncated(session_id, turn_id)
            else:
                adapter._outcomes.mark_success(session_id, turn_id)
            event_pairs, gaps = sync_turn_source_events(
                adapter._ledger,
                opened,
                session_id=session_id,
                turn_id=turn_id,
                user_content=user_content,
                assistant_content=assistant_content,
                recorded_at=said_at,
                outcome=outcome,
                include_user=turn_id not in adapter._user_captured_turns,
            )
        for event, ledger_identity in event_pairs:
            event_context = shown if event["role"] == "assistant" else opened
            with adapter._lock, adapter._calls.holding("sync_turn"):
                adapter._writer.write(
                    event_context,
                    event,
                    identity=ledger_identity,
                    gaps=gaps,
                    scope_id=identity.local_scope_id,
                    bound=identity,
                    release=True,
                    opener_turn_id=turn_id if event["role"] == "user" else None,
                )

    def reset_source_refs(self) -> None:
        """Open a new current-turn fence: no refs, no overflow, no overflow gap."""
        with self._adapter._said_lock:
            self._adapter._current_source_refs.clear()
            self._adapter._current_source_refs_overflow = False
            self._adapter._diagnostics.capability_gaps = tuple(
                gap for gap in self._adapter._diagnostics.capability_gaps if gap != GAP_CURRENT_SOURCE_REFS_LIMIT
            )
