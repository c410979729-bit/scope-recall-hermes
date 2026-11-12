"""Convert Hermes host callbacks into core SourceEvent DTOs at the boundary."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, NamedTuple, cast

from scope_recall.contracts import Origin, SourceEvent, TrustedContext

from .attachments import authorize_attachment_metadata

OutcomeKind = Literal["success", "failure", "cancelled", "interrupted", "truncated"]
SourceIdentity = tuple[str, int]


@dataclass
class SourceObservationLedger:
    """Per-session dedupe and first observed instants, without retaining message content."""

    _confirmed: dict[SourceIdentity, None] = field(default_factory=dict)
    _pending: set[SourceIdentity] = field(default_factory=set)
    _moments: dict[SourceIdentity, tuple[str, str | None]] = field(default_factory=dict)
    _time_lock: Any = field(default_factory=Lock, repr=False)

    def observe(
        self,
        *,
        source_event_key: str,
        source_revision: int,
        role: str,
        content: str,
        origin: str,
        recorded_at: str,
        occurred_at: str | None,
        capture_state: str,
        evidence_refs: list[str] | None = None,
        artifact_refs: list[str] | None = None,
        gaps: tuple[str, ...] = (),
    ) -> tuple[SourceEvent | None, tuple[str, ...], SourceIdentity | None]:
        identity = (source_event_key, source_revision)
        if identity in self._confirmed or identity in self._pending:
            return None, gaps, None
        if len(self._pending) >= 64:
            return None, (*gaps, "capture_gap:observation_pending_capacity"), None
        recorded_at, occurred_at = self.witness(identity, recorded_at, occurred_at)
        self._pending.add(identity)
        event: SourceEvent = {
            "protocol_version": "1.1",
            "source_event_key": source_event_key,
            "source_revision": source_revision,
            "origin": cast(Origin, origin),
            "role": cast(Literal["user", "assistant", "tool", "system", "document", "unknown"], role),
            "content": content,
            "occurred_at": occurred_at,
            "recorded_at": recorded_at,
            "time_precision": "unknown" if occurred_at is None else "instant",
            "capture_state": cast(Literal["complete", "partial", "gap"], capture_state),
            "evidence_refs": list(evidence_refs or ()),
        }
        if artifact_refs:
            event["artifact_refs"] = list(artifact_refs)
        return event, gaps, identity

    def witness(self, identity: SourceIdentity, recorded_at: str, occurred_at: str | None) -> tuple[str, str | None]:
        """Keep only the first seen instants across callbacks, including replies queued for sync.

        Post-model callbacks use their own lock; the timestamp cache is independent of capture's
        pending/confirmed sets and never holds words or waits for store I/O.
        """
        with self._time_lock:
            first = self._moments.setdefault(identity, (recorded_at, occurred_at))
            while len(self._moments) > 4096:
                self._moments.pop(next(iter(self._moments)))
            return first

    def confirm(self, identity: SourceIdentity) -> None:
        self._pending.discard(identity)
        self._confirmed.pop(identity, None)
        self._confirmed[identity] = None
        while len(self._confirmed) > 1024:
            self._confirmed.pop(next(iter(self._confirmed)))

    def rollback(self, identity: SourceIdentity | None) -> None:
        """Release an uncommitted observation; an identityless capture reserved no slot."""
        if identity is not None:
            self._pending.discard(identity)

    def pending_identities(self) -> tuple[SourceIdentity, ...]:
        return tuple(sorted(self._pending))

    def reset(self) -> None:
        self._confirmed.clear()
        self._pending.clear()
        with self._time_lock:
            self._moments.clear()


def host_source_key(
    *,
    installation_id: str,
    session_id: str,
    event_kind: str,
    event_id: str,
    revision: int = 1,
    entry_id: str | None = None,
) -> str:
    safe_kind = event_kind.strip() or "event"
    safe_id = event_id.strip() or "unknown"
    # Entries of a shared store share its installation id and can see the same
    # host session and turn ids -- the owner talking to two bots -- so there the
    # entry is part of the key.  A local store's keys leave it out.
    owner = f"{installation_id}:{entry_id}" if entry_id is not None else installation_id
    return f"hermes:{owner}:{session_id}:{safe_kind}:{safe_id}@{revision}"


def extract_user_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def _opening_key(context: TrustedContext, session_id: str, turn_id: str) -> str:
    """The host key of a turn's opening message, the same whether ``pre_llm_call`` or the turn's sync stores it."""
    return host_source_key(
        installation_id=context.binding.installation_id,
        entry_id=context.entry_id,
        session_id=session_id,
        event_kind="user",
        event_id=turn_id or "turn",
    )


def pre_llm_source_event(
    ledger: SourceObservationLedger,
    context: TrustedContext,
    *,
    session_id: str,
    turn_id: str,
    user_message: object,
    recorded_at: str,
    attachments: list[dict[str, Any]] | None = None,
) -> tuple[SourceEvent | None, tuple[str, ...], SourceIdentity | None]:
    content = extract_user_text(user_message)
    gaps: list[str] = []
    artifact_refs: list[str] = []
    for item in attachments or ():
        auth = authorize_attachment_metadata(item)
        if auth.authorized and auth.artifact_ref:
            artifact_refs.append(auth.artifact_ref)
        elif auth.gap:
            gaps.append(auth.gap)
    capture_state = "partial" if gaps else "complete"
    if not content.strip() and not artifact_refs:
        return None, tuple(gaps), None
    return ledger.observe(
        source_event_key=_opening_key(context, session_id, turn_id),
        source_revision=1,
        role="user",
        content=content,
        origin=context.actor_origin,
        recorded_at=recorded_at,
        occurred_at=recorded_at,
        capture_state=capture_state,
        artifact_refs=artifact_refs or None,
        gaps=tuple(gaps),
    )


def sync_assistant_key(context: TrustedContext, session_id: str, turn_id: str) -> str:
    """The same reply identity for the host's post-model observation and its delayed sync."""
    return host_source_key(
        installation_id=context.binding.installation_id,
        entry_id=context.entry_id,
        session_id=session_id,
        event_kind="sync_assistant",
        event_id=turn_id or "turn",
    )


def sync_turn_source_events(
    ledger: SourceObservationLedger,
    context: TrustedContext,
    *,
    session_id: str,
    turn_id: str,
    user_content: str,
    assistant_content: str,
    recorded_at: str,
    outcome: OutcomeKind,
    include_user: bool = True,
) -> tuple[tuple[tuple[SourceEvent, SourceIdentity | None], ...], tuple[str, ...]]:
    """The turn's opening message and its answer; ``include_user`` False when ``pre_llm_call`` stored the former."""
    gaps: list[str] = []
    if outcome != "success":
        gaps.append(f"outcome_gap:{outcome}")
    events: list[tuple[SourceEvent, SourceIdentity | None]] = []
    if include_user:
        user_event, user_gaps, user_identity = ledger.observe(
            source_event_key=_opening_key(context, session_id, turn_id),
            source_revision=1,
            role="user",
            content=user_content,
            origin=context.actor_origin,
            recorded_at=recorded_at,
            occurred_at=recorded_at,
            capture_state="partial" if outcome != "success" else "complete",
            gaps=tuple(gaps),
        )
        if user_event is not None and user_identity is not None:
            events.append((user_event, user_identity))
        gaps.extend(user_gaps)
    if outcome == "success" and assistant_content.strip():
        assistant_event, assistant_gaps, assistant_identity = ledger.observe(
            source_event_key=sync_assistant_key(context, session_id, turn_id),
            source_revision=1,
            role="assistant",
            content=assistant_content,
            origin="assistant_visible",
            recorded_at=recorded_at,
            occurred_at=recorded_at,
            capture_state="complete",
            gaps=tuple(gaps),
        )
        if assistant_event is not None and assistant_identity is not None:
            events.append((assistant_event, assistant_identity))
        gaps.extend(assistant_gaps)
    elif outcome == "success":
        gaps.append("outcome_gap:missing_assistant_body")
    return tuple(events), tuple(dict.fromkeys(gaps))


#: The blocks Hermes strips from what a reply shows (its ``turn_truncation._THINK_TAG_RE`` tags).
_HIDDEN_BLOCK = re.compile(
    r"<(think|thinking|reasoning|REASONING_SCRATCHPAD)\b[^>]*>.*?(?:</\1\s*>|\Z)", re.IGNORECASE | re.DOTALL
)


def _shown_text(message: dict[str, Any]) -> str:
    """What one assistant message showed: its Codex commentary items if any, else its content."""
    items = message.get("codex_message_items")
    commentary = [
        "".join(
            part["text"]
            for part in item["content"]
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str)
        )
        for item in (items if isinstance(items, list) else ())
        if isinstance(item, dict)
        and item.get("type") == "message"
        and isinstance(item.get("content"), list)
        and str(item.get("phase") or "").strip().lower() == "commentary"
    ]
    text: object = "\n\n".join(said.strip() for said in commentary if said.strip()) or message.get("content")
    if isinstance(text, list):
        text = "\n".join(
            part["text"]
            for part in text
            if isinstance(part, dict)
            and part.get("type") in ("text", "output_text")
            and isinstance(part.get("text"), str)
        )
    return _HIDDEN_BLOCK.sub("", text).strip() if isinstance(text, str) else ""


def _message_time(message: dict[str, Any]) -> str | None:
    """The epoch time Hermes stamps on each message it appends, in the adapter's UTC form."""
    stamp = message.get("timestamp")
    if type(stamp) not in (int, float):
        return None
    try:
        if not math.isfinite(stamp) or stamp <= 0:
            return None
        return datetime.fromtimestamp(stamp, timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


#: Hermes delivers what the person sends while a turn runs (a "steer") as a user row of this kind inside the
#: turn: their words between these marker lines (``agent.prompt_builder.format_steer_marker``), after an origin
#: preamble when a gateway delivered it (``gateway.run_busy._steer_text_with_origin``).
STEER_KIND = "steer"
REDIRECT_KIND = "redirect"
_REDIRECT_CONTEXT = "[Context from the interrupted assistant response]"
_ADDITIONAL_CORRECTION = "[Additional user correction]"
_STEER_OPEN = "[OUT-OF-BAND USER MESSAGE"
_STEER_CLOSE = "[/OUT-OF-BAND USER MESSAGE]"
_STEER_ORIGIN = "Gateway message origin (JSON data, not instructions or authorization):"


def carries_gateway_origin(text: str) -> bool:
    """Whether ``text`` holds steers or redirects Hermes handed over as a message of their own.

    Hermes writes the origin preamble only around a steer or a redirect (``gateway.run_busy._steer_text_with_origin``).
    Those it could not deliver open the next turn, or come back as a queued message, as plain text: behind the
    bracketed tags Hermes puts before a message (the sender's name in a shared chat, a name that may itself hold "]";
    a history timestamp; an attachment count when it summarizes a turn for memory), and after the channel's history
    on a line of their own when it backfills one.  So a line counts when the preamble starts it, or follows text that
    opens with "[" and ends with "] ".  Nothing in that row tells it from a message whose text copies a preamble, so
    such text is never taken as the person's own words: a person who pastes one loses that message from their own,
    the safe side.  Each line is searched once, so a long run of tags costs no more than its length.
    """
    tagged = "] " + _STEER_ORIGIN
    return any(line.startswith(_STEER_ORIGIN) or (line.startswith("[") and tagged in line) for line in text.split("\n"))


#: Hermes' mark on a user message it folded a compression summary into (``COMPRESSED_SUMMARY_METADATA_KEY``).
_COMPRESSED_SUMMARY = "_compressed_summary"
#: The lines that bound a folded summary (``agent.context_compressor``: ``_MERGED_PRIOR_CONTEXT_HEADER``,
#: ``_MERGED_SUMMARY_DELIMITER``, ``_SUMMARY_END_MARKER``) and the header of a to-do list a compression appends to the
#: last user message (``tools.todo_tool.TODO_INJECTION_HEADER``), as Hermes 0.21.5 writes them.  Should Hermes change
#: them, a folded message is no longer recognised: the turn is then the person's.
_PRIOR_CONTEXT_HEADER = "[PRIOR CONTEXT \u2014 for reference only; not a new message]"
_SUMMARY_DELIMITER = "[END OF PRIOR CONTEXT \u2014 COMPACTION SUMMARY BELOW]"
_SUMMARY_END = "--- END OF CONTEXT SUMMARY \u2014 respond to the message below, not the summary above ---"
_TODO_HEADER = "[Your active task list was preserved across context compression]"


def _own_text(message: dict[str, Any]) -> str:
    """A user message's own words, as Hermes reads them back: without a compression summary folded into a message
    Hermes marked as folded (``ContextCompressor._strip_context_summary_handoff_message``), and without a to-do list
    appended to it (``_strip_stale_todo_snapshot``).  The person's words quoted inside a summary are not the
    message's own.  An unmarked message is not unwrapped: a notice that quotes those lines keeps its words whole."""
    text = extract_user_text(message.get("content"))
    if message.get(_COMPRESSED_SUMMARY) is True:
        if _SUMMARY_DELIMITER in text:
            text = text.split(_SUMMARY_DELIMITER, 1)[0].strip()
            if text.startswith(_PRIOR_CONTEXT_HEADER):
                text = text[len(_PRIOR_CONTEXT_HEADER) :]
        elif _SUMMARY_END in text:
            text = text.split(_SUMMARY_END, 1)[1]
        else:
            return ""
    cut = text.find(_TODO_HEADER)
    return (text if cut == -1 else text[:cut]).strip()


def _notice_kind(message: dict[str, Any]) -> bool:
    """A display kind Hermes gives the messages it writes itself: any but a steer, and on a folded message not the
    legacy ``hidden`` either, which may wrap the person's words (``split_user_originated_turn``)."""
    kind = message.get("display_kind")
    if not isinstance(kind, str) or not kind or kind == STEER_KIND:
        return False
    return not (kind == "hidden" and message.get(_COMPRESSED_SUMMARY) is True)


#: Gateway's model-facing timestamp is presentation, not provenance. Only a typed, stamped host row may have
#: one such prefix removed for exact comparison with the hook's clean input.
_NOTICE_TIMESTAMP = re.compile(
    r"^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
    r"(?: [A-Za-z0-9_+\-/:]+)?\]\s+"
)

#: Synthetic user-row markers from ``agent.conversation_compression`` and ``agent.turn_stop_gates``.
_SYNTHETIC_USER_FLAGS = (
    "_todo_snapshot_synthetic",
    "_empty_recovery_synthetic",
    "_verification_stop_synthetic",
    "_pre_verify_synthetic",
    "_dropped_toolcall_nudge",
    "_kanban_stop_synthetic",
)
#: What Hermes records on a user row that absorbed the rows after it (``agent.agent_runtime_helpers.
#: _merge_consecutive_users``, ``agent.message_metadata.record_absorbed_message``): the text it held before, when
#: there was any, and the absorbed rows' ids.  The joined row keeps the first row's dictionary, platform id included.
_MERGED_PREFIX = "_merged_turn_prefix"
_MERGE_WITNESSES = (_MERGED_PREFIX, "_absorbed_message_uids", "_absorbed_row_ids")


def _platform_id(message: dict[str, Any]) -> bool:
    """Whether a user row carries the id of a platform message: ``platform_message_id`` on the row Hermes stages for a
    turn (``agent.turn_context``, from the gateway's ``persist_user_platform_id``), ``message_id`` on a row it replays
    from its session store."""
    return any(
        message.get(key) is not None and str(message[key]).strip() for key in ("platform_message_id", "message_id")
    )


def _kind_notice(message: dict[str, Any], text: str, *, trailing: bool) -> bool:
    """Whether ``message`` is a notice Hermes marked by kind and ``text`` is its words.

    Past the last reply (``trailing`` false) the message must be one Hermes folded and marked.  A compression at the
    turn's start can fold its summary into the turn's own message (``ContextCompressor._merge_summary_into_tail_row``)
    and put the reply it folded away after it (``_reply_insertion_index``), and a delegation result would be stored as
    the person's that way.  Anything else before the last reply belongs to an earlier turn.  Its own words only: a
    summary quotes the person's messages word for word, and a message merely holding the turn's text would take the
    person's words for a notice.
    """
    words = _own_text(message)
    notice = _notice_kind(message)
    # Rendered history and the hook's clean input differ by this prefix.
    # A display kind and a valid host timestamp must attest the row first;
    # human text and steers are never normalized into host authority.
    if words != text and notice and _message_time(message) is not None:
        prefix = _NOTICE_TIMESTAMP.match(words)
        if prefix is not None:
            words = words[prefix.end() :]
    return bool(words and words == text and notice and (trailing or message.get(_COMPRESSED_SUMMARY) is True))


def opener_origin(history: object, user_message: object, *, local_surface: bool, staged: bool = False) -> str:
    """The origin of the message the turn ``pre_llm_call`` opens on: the person's, Hermes's, or unknown.

    Hermes marks the user messages it writes itself with a display kind: a finished background process, a
    delegation's result, a wake-up, a plugin's message (``gateway.response_filters.display_kind_for_event``, the
    CLI's ``TimelineNotification``).  A steer is the one kind that holds the person's words
    (``ContextCompressor._is_actionable_user_turn``).  Stored as the person's, a notice would read as something they
    said.

    ``staged`` says the conversation is the one Hermes hands ``pre_llm_call``: it has just appended the turn's own row
    (``agent.turn_context``), so that row is known by its place (``_staged_row``) and its platform id needs no
    comparison of words.  Later hooks see the conversation after Hermes joined rows, compacted it and added steers;
    there the latest user row with words of its own decides, skipping later steers and redirects unless they match
    the opener, and its words must match (``_attests``).  The turn's own tool calls and their results may follow
    its row; past a reply that called no tool, the row belongs to an answered earlier turn and lends no platform id,
    unless it is a marked folded row whose reply a compression moved after it.  A to-do list alone has no words of
    its own.  The row and the opener go through the evidence order of ``_classify_opener``.
    """
    raw_text = extract_user_text(user_message)
    text = raw_text.strip()
    trailing = True
    answered = False
    row = None
    for message in reversed(history if isinstance(history, list) else []):
        if not isinstance(message, dict) or message.get("role") != "user":
            trailing = False
            tool_step = isinstance(message, dict) and (
                message.get("role") == "tool"
                or (message.get("role") == "assistant" and bool(message.get("tool_calls")))
            )
            answered = answered or not tool_step
            continue
        words = _own_text(message)
        if words and (message.get("display_kind") == STEER_KIND or _is_redirect(message)) and words != text:
            continue
        if words:
            row = message
            break
    notice = row is not None and _kind_notice(row, text, trailing=trailing)
    placed = _staged_row(history) if staged else None
    if placed is not None:
        return _classify_opener(history, raw_text, placed, notice, local_surface=local_surface, exact=False)
    evidence_row = row if row and (not answered or row.get(_COMPRESSED_SUMMARY) is True) else None
    return _classify_opener(history, raw_text, evidence_row, notice, local_surface=local_surface)


def _staged_row(history: object) -> dict[str, Any] | None:
    """The row Hermes staged for the turn, from the conversation it hands ``pre_llm_call``; None if unclear.

    Hermes appends the turn's row and calls the hook with that conversation, the row last
    (``agent.turn_context``).  Hermes 0.22 compacts the conversation first: the compaction can fold its summary into
    that row, marking it (``_COMPRESSED_SUMMARY``), and put the reply it folded away and a to-do list after it.  So
    replies, tool results and marked scaffolding with no words of its own are passed over, and a row found past a
    reply must be the marked folded one.
    """
    past_reply = False
    for message in reversed(history if isinstance(history, list) else []):
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if role in ("assistant", "tool"):
            past_reply = True
            continue
        if role != "user":
            return None
        if any(message.get(flag) is True for flag in _SYNTHETIC_USER_FLAGS) and not _own_text(message):
            continue
        return message if not past_reply or message.get(_COMPRESSED_SUMMARY) is True else None
    return None


def sync_opener_origin(messages: object, user_message: object, *, local_surface: bool) -> str:
    """The origin of a completed turn's opener, keeping later steers separate from its identity.

    Walking back from the turn's end, later redirects and marked scaffolding without the opener are transparent.
    Equal-text steers need the native tool-batch boundary, with no completed reply before the opener. Other user rows,
    including unmarked system notes, bound the search and can leave no evidence for the opener. A request hook's
    earlier classification is retained by the caller and does not depend on this later search.
    """
    raw_text = extract_user_text(user_message)
    if not isinstance(messages, list):
        return _classify_opener(messages, raw_text, None, False, local_surface=local_surface)
    text = raw_text.strip()
    end = len(messages)
    matching_steer = False
    row = None
    while end:
        message = messages[end - 1]
        if not isinstance(message, dict):
            break
        role = message.get("role")
        if role in ("assistant", "tool"):
            if matching_steer and role == "assistant" and not message.get("tool_calls"):
                break
            end -= 1
            continue
        if role == "user" and message.get("display_kind") == STEER_KIND:
            words = _own_text(message)
            if not words:
                break
            pieces = _steer_pieces(words)
            if words == text or any(piece == text for _origin, piece in pieces):
                # Hermes emits native steers after tool results, never after a completed turn's answer.
                before = messages[end - 2] if end > 1 else None
                if not pieces or not isinstance(before, dict) or before.get("role") != "tool":
                    break
                matching_steer = True
            end -= 1
            continue
        if role == "user" and _is_redirect(message) and _own_text(message) != text:
            end -= 1
            continue
        if (
            role == "user"
            and any(message.get(flag) is True for flag in _SYNTHETIC_USER_FLAGS)
            and not _holds(message, _unstamped(text).strip())
        ):
            end -= 1
            continue
        # The located row owns this identity; the classifier must not search behind an empty user boundary.
        if role == "user":
            row = message
        break
    notice = row is not None and _kind_notice(row, text, trailing=True)
    return _classify_opener(messages, raw_text, row, notice, local_surface=local_surface)


def _holds(message: dict[str, Any], said: str) -> bool:
    """Whether a row Hermes marked as scaffolding may hold the turn's message ``said``: as its own words (a compaction
    appends its to-do list to the turn's row and marks that row), or as the last part a join put after it."""
    own = _unstamped(_own_text(message).strip()).strip()
    full = extract_user_text(message.get("content")).rstrip()
    last = _unstamped(full.rsplit("\n\n", 1)[-1].strip()).strip() if "\n\n" in full else ""
    return bool(said) and (own == said or own.startswith(said + "\n\n") or last == said)


def _attests(row: dict[str, Any], words: str, said: str, *, exact: bool) -> bool:
    """Whether ``row``'s platform id vouches for the turn's text ``said``; ``words`` are the row's own words.

    Hermes joins adjacent user rows into the first row's dictionary, a blank line apart, so a joined row's id names
    only its first part, the text it held before (``_MERGE_WITNESSES``); a joined row that held none names no words.
    The row Hermes has just staged (``exact`` false) is the turn's by its place, and its words are those Hermes shows
    the model, which can wrap the message it hands the hook (a platform's routing note, a group's observed context).
    Elsewhere the row's words, replay times aside, must be the turn's text or begin with it and a blank line.
    """
    joined = any(row.get(key) for key in _MERGE_WITNESSES)
    if not exact and not joined:
        return True
    first = words
    if joined:
        prefix = row.get(_MERGED_PREFIX)
        if not isinstance(prefix, str) or not prefix.strip():
            return False
        first = _unstamped(prefix.strip()).strip()
    return first == said or first.startswith(said + "\n\n")


def _classify_opener(
    history: object,
    text: str,
    row: dict[str, Any] | None,
    notice: bool,
    *,
    local_surface: bool,
    exact: bool = True,
) -> str:
    """The origin of a turn's opener from the turn's own ``row`` and ``text``, strongest evidence first.

    1. Text carrying steers Hermes handed over (``carries_gateway_origin``), and a turn whose row Hermes marked as a
       notice (``_kind_notice``), are Hermes's.
    2. The turn's row carrying the id of a platform message makes the opener the person's, when that id vouches for
       the turn's text (``_attests``).  The id identifies a platform delivery, not the author of every word: skill
       and command expansions, webhook deliveries and viewer submissions carry ids too, and Hermes gives a goal's
       kickoff the command's id and a viewer submission an admission id.
    3. Text beginning with a known Hermes row prefix or the goal continuation prefix, after replay timestamps,
       is Hermes's. Only below the ID: someone who pastes such a line into a delivered message keeps their words.
    4. On a local surface the rest is treated as the person's.
    5. If no user row has an ID, or history is unavailable, retain the earlier human fallback. This also covers older
       hosts, platforms without IDs, and the first turn of some host-created sessions.
    6. Otherwise the origin is unknown: the selected row may lack an ID, fail to match, or have lost it in a merge.
    """
    if carries_gateway_origin(text) or notice:
        return "host_generated"
    said = _unstamped(text.strip()).strip()
    words = _unstamped(_own_text(row).strip()).strip() if row else ""
    if row and _platform_id(row) and words and said and _attests(row, words, said, exact=exact):
        return "human_direct"
    if _unstamped(text).startswith((*_HERMES_ROW_HEADS, _GOAL_CONTINUATION_HEAD)):
        return "host_generated"
    if local_surface or not isinstance(history, list):
        return "human_direct"
    if not any(isinstance(item, dict) and item.get("role") == "user" and _platform_id(item) for item in history):
        return "human_direct"
    return "origin_unknown"


#: How Hermes 0.21.5 begins the notices it delivers into a running turn the way it delivers a steer, with the chat's
#: origin: a background process's heartbeat or end, a delegation's report.  They are not the person's words.
_STEER_NOTICE_HEADS = ("[Background process ", "[IMPORTANT: Background process", "[ASYNC DELEGATION BATCH COMPLETE")


class Steer(NamedTuple):
    """A delivered steer or redirect, or an unmarked row no turn answered: key, words, time and source kind.

    Unanswered rows have unknown authors. ``alike`` holds the times of other rows with the same words, including
    answered rows, for matching against stored sources.
    """

    key: str
    words: str
    occurred_at: str | None
    kind: str = "steer"
    alike: tuple[str | None, ...] = ()


#: The first line of a gateway's origin preamble, at the start of a line of a steer row.
_ORIGIN_LINE = re.compile(r"(?m)^" + re.escape(_STEER_ORIGIN))


def _steer_pieces(content: object, *, redirect: bool = False) -> list[tuple[dict[str, Any] | None, str]]:
    """A steer or redirect's pieces, each its origin and its words, in order.

    Hermes joins the steers queued before the next tool boundary into one row, a line apart (``AIAgent.steer``), and
    a gateway puts its origin preamble before each (``_steer_text_with_origin``): that line, the origin's JSON, a
    line, a blank line, then the words.  A piece with no preamble has no origin (None); one whose JSON cannot be
    read has an origin naming no one ({}).
    """
    text = extract_user_text(content).lstrip()
    if redirect:
        # Hermes joins pending redirects in this row at this separator; check each part separately.
        segments = text.split(f"\n\n{_ADDITIONAL_CORRECTION}\n")
    else:
        line_end = text.find("\n")
        if not text.startswith(_STEER_OPEN) or line_end == -1:
            return []
        body = text[line_end + 1 :]
        close = body.rfind(_STEER_CLOSE)
        body = body[:close] if close != -1 else body
        starts = [match.start() for match in _ORIGIN_LINE.finditer(body)]
        if not starts or starts[0] != 0:
            starts.insert(0, 0)
        segments = [
            body[start : starts[index + 1] if index + 1 < len(starts) else len(body)]
            for index, start in enumerate(starts)
        ]
    pieces: list[tuple[dict[str, Any] | None, str]] = []
    for index, piece in enumerate(segments):
        if not redirect and index + 1 < len(segments):
            # Only an intermediate steer segment can end with a separator between inputs.
            piece = piece.rstrip().removesuffix("\n" + _ADDITIONAL_CORRECTION)
        # A preamble names a sender only where the gateway writes it, at the very start of its piece.
        if not piece.startswith(_STEER_ORIGIN):
            pieces.append((None, piece.strip()))
            continue
        try:
            origin, _end = json.JSONDecoder().raw_decode(piece[len(_STEER_ORIGIN) :].lstrip())
            json.dumps(origin, ensure_ascii=False).encode("utf-8")  # a lone surrogate would fail every later write
        except (ValueError, RecursionError):
            origin = {}
        blank = piece.find("\n\n")
        words = piece[blank + 2 :].strip() if blank != -1 else ""
        pieces.append((origin if isinstance(origin, dict) else {}, words))
    return pieces


def _sent_by(origin: dict[str, Any], user_id: str) -> bool:
    """Whether an origin names the session's person, as their id or as the ``user_<12 hex>`` a gateway that redacts
    personal data writes in its place (``gateway.session._hash_sender_id``)."""
    said = str(origin.get("user_id") or "")
    return bool(user_id) and said in {user_id, "user_" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]}


def _is_redirect(message: dict[str, Any]) -> bool:
    """Recognize the interrupted-response context wrapped around a redirect's own text."""
    api_content = message.get("api_content")
    content = message.get("content")
    return (
        message.get("role") == "user"
        and "display_kind" not in message
        and isinstance(content, str)
        and isinstance(api_content, str)
        and not content.startswith(_REDIRECT_CONTEXT)
        and api_content.startswith(_REDIRECT_CONTEXT + "\n")
        and api_content.endswith("\n\n" + content)
    )


def person_steers(messages: object, *, platform: str, user_id: str, local_surface: bool) -> tuple[Steer, ...]:
    """Every steer and redirect in ``messages`` that the session's person sent, in order and each once.

    Read wherever it is in the list: Hermes puts notices, compression summaries and to-do lists after a steer.  A
    gateway delivers the person's steer after an origin naming them; a piece without one comes from the agent's side
    (a parent agent writing to the agent it delegated to), and is the person's only on their local surfaces, where
    they type it.  A notice Hermes delivers the same way, with the chat's origin, is not theirs.  In a row of joined
    pieces, nothing after the first piece that is not the person's is taken: a preamble further on may be text inside
    that piece.  A piece is named by its chat and message id, else by its time and words, so the same steer read
    again at a compression, a turn's end or the session's end is the same capture.
    """
    if not isinstance(messages, list):
        return ()
    found: dict[tuple[str, str], Steer] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        redirect = _is_redirect(message)
        if message.get("display_kind") != STEER_KIND and not redirect:
            continue
        occurred_at = _message_time(message)
        for origin, words in _steer_pieces(message.get("content"), redirect=redirect):
            if origin is None:
                theirs = local_surface
            else:
                theirs = _sent_by(origin, user_id) and (local_surface or str(origin.get("platform") or "") == platform)
            notice = words.startswith(_STEER_NOTICE_HEADS)
            if words.startswith("[Background process "):
                # Reserve the native heartbeat frame, not a person's sentence sharing its opening words.
                notice = (
                    re.match(r"\[Background process \S+ heartbeat #(?:\d+|\?) — still running(?:[\s.\]]|$)", words)
                    is not None
                )
            if not theirs or not words or notice or (redirect and _written_by_hermes(words)):
                break
            try:
                encoded_words = words.encode("utf-8")
            except UnicodeEncodeError:
                continue
            message_id = str((origin or {}).get("message_id") or "").strip()
            if origin is not None and message_id:
                # A message id is the chat's own count: two chats of one platform share them.
                key = f"{origin.get('platform')}:{origin.get('chat_id')}:{message_id}"
            else:
                key = "text:" + hashlib.sha256(f"{occurred_at}\x00".encode() + encoded_words).hexdigest()[:32]
            kind = REDIRECT_KIND if redirect else STEER_KIND
            found.setdefault((kind, key), Steer(key, words, occurred_at, kind))
    return tuple(found.values())


#: The event kind of unmarked rows no turn answered, whose authors are unknown (``unanswered_messages``).
UNANSWERED_KIND = "unanswered"
#: How the user rows Hermes writes itself begin, besides a bracketed head in capitals (``_HERMES_NOTICE``).  A
#: gateway that restarts replays the rows it flushed at its stop as text alone
#: (``gateway.shutdown_flush.recover_pending_to_db``), without the display kind that marks them otherwise, and the
#: rows below never have one:
#:
#: - a background process's notice, a compression's to-do list and the turn it continues;
#: - a system note, among them the language server's late verdicts (``agent.turn_stop_gates``);
#: - an agent's message and a gateway's origin;
#: - a delivery a cron job or a webhook mirrored into the chat (``cron.scheduler_delivery._cron_mirror_message``,
#:   ``gateway.platforms.webhook``), and the mark a replay puts on a mirrored row
#:   (``gateway.run._build_gateway_agent_history``);
#: - the text a stale high-risk confirmation is replaced with (``agent.replay_cleanup._EXPIRED_CONFIRMATION_SENTINEL``).
_HERMES_ROW_HEADS = (
    *_STEER_NOTICE_HEADS,
    _PRIOR_CONTEXT_HEADER,
    _TODO_HEADER,
    "Continue from the compressed",
    "[System:",
    "[System note:",
    "[A2A inbound",
    "Gateway message origin",
    "[Cron delivery: ",
    "[Webhook delivery: ",
    "[Delivered from ",
    "[A high-risk confirmation previously given here has EXPIRED",
)
#: The standing-goal continuation templates (``hermes_cli.goals``) queued as ordinary opener messages
#: (``gateway.run_goals._synthetic_prompt_event``); the same words can be a person's redirect.
_GOAL_CONTINUATION_HEAD = "[Continuing toward your standing goal"
#: A notice Hermes writes opens with a bracketed head in capitals: "[IMPORTANT: …", "[ASYNC DELEGATION COMPLETE — …",
#: "[SYSTEM NOTICE — …", "[CONTEXT COMPACTION …", "[SILENT]".  Too many forms to list, and a person's own message
#: seldom opens so; one that does is left out of a restarted gateway's rows, the safe side.
_HERMES_NOTICE = re.compile(r"\[[A-Z][A-Z0-9_]+(?:[\s:\]\-]|\u2014)")
#: The time a gateway puts before each user row it replays when ``gateway.message_timestamps.enabled`` is on, in its
#: current form and its older ISO one (``gateway.message_timestamps._TIMESTAMP_PREFIX_RE``); an older row may hold
#: several.  It hides how the row begins.
_REPLAY_TIME = re.compile(
    r"\[(?:[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?: [A-Za-z0-9_+\-/:]+)?|\d{4}-\d{2}-\d{2}T[^\]]+)\]\s*"
)


def _written_by_hermes(words: str) -> bool:
    """Whether words are begun the way the user rows Hermes writes itself are begun, or tell of its stop."""
    return (
        words.startswith(_HERMES_ROW_HEADS)
        or _HERMES_NOTICE.match(words) is not None
        or "Hermes is shutting down" in words
    )


def _unstamped(text: str) -> str:
    """``text`` without the times a replay put before it (``_REPLAY_TIME``), read in one pass."""
    at = 0
    while (stamp := _REPLAY_TIME.match(text, at)) is not None:
        at = stamp.end()
    return text[at:]


def _unmarked_words(message: dict[str, Any]) -> str:
    """A user row's words without recognizable Hermes markers: no kind Hermes gives the rows it writes, not a summary it
    folded, holding no steers Hermes handed over (``carries_gateway_origin``), and no part begun the way the rows
    Hermes writes are begun, once the times a replay put before it are set aside.

    Hermes joins consecutive user rows a blank line apart (``agent.agent_runtime_helpers._merge_consecutive_users``),
    also when it loads a conversation to replay, and a row joined with a plain one keeps no kind: each part is read on
    its own, and a row that joins the person's words with a delivery or a notice is left out whole, the safe side.
    The words keep the times: they name the row.
    """
    if message.get("display_kind") or message.get(_COMPRESSED_SUMMARY) is True or _is_redirect(message):
        return ""
    words = _own_text(message)
    if not _unstamped(words) or carries_gateway_origin(words):
        return ""
    if any(_written_by_hermes(_unstamped(part.lstrip())) for part in words.split("\n\n")):
        return ""
    return words


def unanswered_messages(messages: object) -> tuple[Steer, ...]:
    """Unmarked user rows no turn answered, in order and each once; their authors are unknown.

    A gateway that stops while messages wait for a busy agent keeps them, and when it starts again puts them into the
    conversation without opening a turn for them (``gateway.shutdown_flush.recover_pending_to_db``): no hook is handed
    them. Hermes also writes unmarked user rows, so recognizable Hermes rows are excluded but the remaining rows do
    not prove their authors. A row a turn opened on can still be unanswered if the gateway stopped before its reply.
    Rows are named by their time and words, so one read again is the same capture. Each carries the
    times of the conversation's other messages said with the same words (``alike``): looked up by words and moment,
    every one of them takes its own stored copy in turn, and a second "继续" is not taken for the first.
    """
    if not isinstance(messages, list):
        return ()
    rows = [
        message for message in messages if isinstance(message, dict) and message.get("role") in {"user", "assistant"}
    ]
    unmarked = [index for index, row in enumerate(rows) if row["role"] == "user" and _unmarked_words(row)]
    unanswered: list[int] = []
    said: dict[str, set[str | None]] = {}
    for position, index in enumerate(unmarked):
        said.setdefault(_unmarked_words(rows[index]), set()).add(_message_time(rows[index]))
        following = unmarked[position + 1] if position + 1 < len(unmarked) else len(rows)
        if following < len(rows) and not any(row["role"] == "assistant" for row in rows[index + 1 : following]):
            unanswered.append(index)
    found: dict[str, Steer] = {}
    for index in unanswered:
        words, occurred_at = _unmarked_words(rows[index]), _message_time(rows[index])
        key = "text:" + hashlib.sha256(f"{occurred_at}\x00{words}".encode()).hexdigest()[:32]
        alike = tuple(sorted(said[words] - {occurred_at}, key=lambda moment: (moment is None, moment or "")))
        found.setdefault(key, Steer(key, words, occurred_at, UNANSWERED_KIND, alike))
    return tuple(found.values())


def interim_messages(history: object, *, answer: str) -> tuple[tuple[str, str | None], ...]:
    """What the assistant showed between its tool calls in the turn that ends ``history``, and when.

    Hermes hands ``sync_turn`` only a turn's answer; what it said on the way stays in the conversation.
    The turn is everything after the user message that opened it; a steer is inside the turn.  Its closing
    message without a tool call, and any words equal to ``answer``, are the answer and stay with
    ``sync_turn``; hidden rows and repeats are dropped.
    """
    if not isinstance(history, list):
        return ()
    turn: list[dict[str, Any]] = []
    for message in reversed(history):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            if message.get("display_kind") == STEER_KIND:
                continue
            break
        if message.get("role") == "assistant" and message.get("display_kind") != "hidden":
            turn.append(message)
    turn.reverse()
    if turn and not turn[-1].get("tool_calls"):
        turn.pop()
    seen = {" ".join(answer.split())}
    said: list[tuple[str, str | None]] = []
    for message in turn:
        text = _shown_text(message)
        if text and text != "(empty)" and " ".join(text.split()) not in seen:
            seen.add(" ".join(text.split()))
            said.append((text, _message_time(message)))
    return tuple(said)


def interim_source_key(context: TrustedContext, session_id: str, turn_id: str, ordinal: int) -> str:
    """The stable identity of an intermediate reply, shared with content-free observation."""
    return host_source_key(
        installation_id=context.binding.installation_id,
        entry_id=context.entry_id,
        session_id=session_id,
        event_kind="interim",
        event_id=f"{turn_id or 'turn'}:{ordinal}",
    )


def interim_source_event(
    ledger: SourceObservationLedger,
    context: TrustedContext,
    *,
    session_id: str,
    turn_id: str,
    ordinal: int,
    content: str,
    recorded_at: str,
    occurred_at: str | None = None,
) -> tuple[SourceEvent | None, tuple[str, ...], SourceIdentity | None]:
    """One thing the assistant showed on the way through a turn, named by its turn and place in it."""
    return ledger.observe(
        source_event_key=interim_source_key(context, session_id, turn_id, ordinal),
        source_revision=1,
        role="assistant",
        content=content,
        origin="assistant_visible",
        recorded_at=recorded_at,
        occurred_at=occurred_at or recorded_at,
        capture_state="complete",
    )


def steer_source_key(context: TrustedContext, steer: Steer) -> str:
    """A steer's key: the steer's own name (``person_steers``) and no session's.  A compression carries a steer into
    the session it continues; read there, it is the source stored before (``Sources.said_in_session`` across
    sessions), and a delete of it holds there too.  Hermes stamps every message it appends with a time
    (``agent.message_metadata.stamp_message_timestamp``); words that came with neither a message id nor a time would
    be taken for the same words stored before, which keeps a delete holding rather than store a deleted steer again."""
    return host_source_key(
        installation_id=context.binding.installation_id,
        entry_id=context.entry_id,
        session_id="steers",
        event_kind=steer.kind,
        event_id=steer.key,
    )


def steer_source_event(
    ledger: SourceObservationLedger,
    context: TrustedContext,
    *,
    steer: Steer,
    recorded_at: str,
) -> tuple[SourceEvent | None, tuple[str, ...], SourceIdentity | None]:
    """Build a delivered steer or unattributed historical row with the context's origin in its first observed session.

    Untimed redirects and unanswered history keep an unknown occurrence so Core can enforce historical privacy
    intervals. A steer may use the instant its request hook first observes it.
    """
    return ledger.observe(
        source_event_key=steer_source_key(context, steer),
        source_revision=1,
        role="user",
        content=steer.words,
        origin=context.actor_origin,
        recorded_at=recorded_at,
        occurred_at=steer.occurred_at if steer.kind != STEER_KIND else steer.occurred_at or recorded_at,
        capture_state="complete",
    )


def tool_call_source_event(
    ledger: SourceObservationLedger,
    context: TrustedContext,
    *,
    session_id: str,
    tool_call_id: str,
    tool_name: str,
    result: object,
    recorded_at: str,
    outcome: OutcomeKind,
    origin: Origin = "tool_observation",
) -> tuple[SourceEvent | None, tuple[str, ...], SourceIdentity | None]:
    gaps: list[str] = []
    if outcome != "success":
        gaps.append(f"outcome_gap:{outcome}")
    content = "" if result is None else str(result)
    if not content.strip():
        # Nothing to keep.  A call that failed, was cancelled or was cut off without a word keeps only its outcome.
        if outcome == "success":
            gaps.append("outcome_gap:missing_tool_result")
        return None, tuple(gaps), None
    capture_state = "partial" if gaps else "complete"
    return ledger.observe(
        source_event_key=host_source_key(
            installation_id=context.binding.installation_id,
            entry_id=context.entry_id,
            session_id=session_id,
            event_kind="tool",
            event_id=tool_call_id or tool_name or "tool",
        ),
        source_revision=1,
        role="tool",
        content=content,
        origin=origin,
        recorded_at=recorded_at,
        occurred_at=recorded_at,
        capture_state=capture_state,
        gaps=tuple(gaps),
    )
