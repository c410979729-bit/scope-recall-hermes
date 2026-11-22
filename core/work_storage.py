"""SQLite work_items lease and lifecycle inside the owning transaction."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..contracts import ContractError
from .embedding_queue import RESPACE_CURSOR as RESPACE_CURSOR
from .embedding_queue import RESPACE_MARKER as RESPACE_MARKER
from .embedding_queue import RESPACE_PAGE as RESPACE_PAGE
from .embedding_queue import RESPACE_SCAN as RESPACE_SCAN
from .embedding_queue import EmbeddingQueue
from .embedding_queue import respace_refusal as respace_refusal
from .failure_policy import (
    AUTO_RECOVERABLE_ERRORS,
    CAPACITY_REFUSALS,
    WORK_RETRY,
    capacity_backoff_seconds,
)
from .failure_retry import (
    DERIVATION_RETRY_MARKER,
    RETRY_GENERATION,
    RETRY_MARKER,
    already_retried,
    marked,
    validate_page,
    validation_feedback,
    wanted,
)
from .inbox_rules import deferred_path
from .schema import SCHEMA_VERSION
from .work_subjects import _RETRY_SUBJECT_REASON, _SUBJECT_CONTEXT, ALLOWED_WORK_TYPES, _marks
from .work_subjects import AUTO_RECOVERABLE_WORK_TYPES as AUTO_RECOVERABLE_WORK_TYPES

MAX_RECOVERABLE_ATTEMPTS = WORK_RETRY.attempts or 0
MAX_OPERATOR_RETRIES = 2
#: Work items one explicit operator retry may name.
MAX_OPERATOR_PAGE = 8
#: Failed rows ``retry_failed`` reads at a time (``WorkItems._failed_rows``).
RETRY_SCAN_PAGE = 512
#: Failed rows one pass may reopen.  Reopening a failure is not the same kind of
#: work as claiming a ready item: each reopened row becomes a model call later,
#: so it keeps its own page however large a pass of cheap work may be.  It was
#: ``max_items`` while the two were the same size.
MAX_RECOVERY_PAGE = 200
#: Work items one claim may lease.  Matches what one pass may hold
#: (``core/worker.py``'s ``max_items``), because a pass that claims its embedding
#: group together claims as much of itself as it is allowed to process.
MAX_CLAIM_PAGE = 1000
INTERRUPTED_RETRY_MARKER = "interrupted_retry:1"
#: What ``recover_oversized_consolidations`` writes on the row it repairs, its one repair.
CHUNK_UPGRADE_MARKER = "chunk_upgrade:1106"
OPERATOR_ORIGINS = frozenset({"human_direct", "host_generated"})
_OPERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
#: The longest error code a row keeps, and the longest failure kind a shortened one keeps beside its marks.
CODE_BOUND = 1024
_KIND_BOUND = 128
# A mark is read wherever a later step left it: at the start, after another segment, or behind an explicit retry's
# ``prior:``, which keeps the code it found.
_AT_A_MARK = r"(?:^|\|)(?:prior:)?"
#: Each explicit retry's token; together they are the item's explicit budget.
_OPERATOR_TOKEN = re.compile(rf"{_AT_A_MARK}operator_retry:([A-Za-z0-9][A-Za-z0-9_.-]{{0,63}})(?=\||$)")
_AUTO_TOKEN = re.compile(rf"{_AT_A_MARK}auto_retry:([0-9]+)(?=\||$)")
#: The marks a retry budget is read from, but the automatic count: a release's re-look, the extra derivation
#: attempt, the interrupted attempt's, the one repair of an old oversized consolidation, an explicit retry.
_BUDGET_MARK = re.compile(
    rf"{_AT_A_MARK}({RETRY_MARKER}:[^|]+|{DERIVATION_RETRY_MARKER}|{INTERRUPTED_RETRY_MARKER}|{CHUNK_UPGRADE_MARKER}"
    rf"|operator_retry:[A-Za-z0-9][A-Za-z0-9_.-]{{0,63}})(?=\||$)"
)

#: Token recording how many times in a row a provider refused for capacity.
#: Kept in the error code beside ``auto_retry:`` rather than a new column, so
#: no migration is needed and an operator reading the row sees the history.
_CAPACITY_TOKEN = re.compile(r"(?:^|\|)capacity:([0-9]+)(?=\||$)")

#: Purge rows are claimed before anything else: a delete must land before
#: derived work can rebuild what it removes.  Candidate evaluation is claimed
#: last: it re-judges what memory already holds, while consolidation and
#: embedding are what make a new conversation recallable at all, so a busy
#: candidate queue must never hold captured conversation back.
_CLAIM_ORDER = "CASE work_type WHEN 'purge' THEN 0 WHEN 'evaluate_candidate' THEN 2 ELSE 1 END"
#: How long a ready evaluation waits at the back before it joins the FIFO.  Projection work does not run dry on a
#: store upgraded with tens of thousands of claims (an entity backfill page, retries and recoveries every pass), so
#: strict last place would never reach an evaluation.  Fresh conversation keeps its own lane (``_fresh_first``)
#: whatever the backlog holds, and fifteen minutes lets the consolidation an evaluation depends on run first.
EVALUATION_AGING_SECONDS = 900
_AGED_CLAIM_ORDER = (
    "CASE WHEN work_type='purge' THEN 0 WHEN work_type='evaluate_candidate' AND available_at>? THEN 2 ELSE 1 END"
)
_LEASED_ROW = "work_id=? AND state='leased' AND lease_token=? AND lease_owner=?"

# Recent conversations receive at most every other claim so the existing backlog continues to drain.
FRESH_LANE_SECONDS = 2 * 3600.0
#: The conversation roots a person or a tool produced in the conversation.
#: The other roots are ingested content: external_document and imported record
#: when content was stored, not when anyone said it, and one bulk import would
#: otherwise fill the lane.
FRESH_CONVERSATION_ORIGINS = frozenset({"human_direct", "tool_observation"})
#: assistant_visible is not a consolidation root, but recall searches it
#: semantically, so its embedding is fresh work too.
FRESH_LANE_ORIGINS = {
    "consolidate": FRESH_CONVERSATION_ORIGINS,
    "embed": FRESH_CONVERSATION_ORIGINS | {"assistant_visible"},
}
#: Recently available rows one lane claim examines, newest first.  The bound
#: keeps the query off the backlog: a reverse range scan of ``work_ready``, one
#: primary-key probe into ``source_events`` per row and a sort of at most this
#: many rows, a few milliseconds with tens of thousands pending.  It covers 16
#: passes of the largest refill (16 sources x 2 types), which is more than a
#: realistic window holds.  Fresh work beyond it is still claimed in FIFO order.
FRESH_LANE_SCAN_ROWS = 512


def fresh_since(now: str) -> str:
    """The oldest ``persisted_at`` that still counts as fresh conversation."""
    return _after(now, -FRESH_LANE_SECONDS)


def _auto_count(error: str) -> int:
    return max((int(value) for value in _AUTO_TOKEN.findall(error)), default=0)


def _failure_kind(error: str) -> str:
    return error.rsplit("|", 1)[-1]


def _budget_marks(error: str, *, counted: bool = True) -> str:
    """Every mark in ``error`` a retry budget is read from, once each in the order met, and the automatic count
    unless ``counted`` is off: what a code that is rebuilt or shortened must keep, or a spent budget begins again."""
    marks = "".join(f"{mark}|" for mark in dict.fromkeys(_BUDGET_MARK.findall(error)))
    count = _auto_count(error) if counted else 0
    return marks + (f"auto_retry:{count}|" if count else "")


def _within(code: str) -> str:
    """``code``, a history ending in its failure kind, within ``CODE_BOUND``.  Past it, what lies between its budget
    marks and its kind gives way: the history is detail, the marks and the kind decide what is done next.  A kind
    longer than any the worker writes is cut, so the marks always fit."""
    if len(code) <= CODE_BOUND:
        return code
    return (_budget_marks(code) + _failure_kind(code)[-_KIND_BOUND:])[:CODE_BOUND]


def _preserve_retry_history(error) -> bool:
    """Whether a new failure is added to ``error`` rather than replacing it: it holds a mark a budget is read from."""
    return bool(error) and bool(_budget_marks(str(error)))


def _auto_retry_stamp(error: str) -> str:
    """The error code of a row granted one automatic recovery: its marks, its new count and its failure kind."""
    return _within(f"{_budget_marks(error, counted=False)}auto_retry:{_auto_count(error) + 1}|{_failure_kind(error)}")


def _capacity_count(error: object) -> int:
    return max((int(value) for value in _CAPACITY_TOKEN.findall(str(error or ""))), default=0)


def _operator_retry_ids(value: str) -> tuple[str, ...]:
    return tuple(_OPERATOR_TOKEN.findall(value))


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ContractError("INPUT_INVALID", "timestamp")
    return parsed.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _after(now: str, seconds: float) -> str:
    return _format_time(_parse_time(now) + timedelta(seconds=seconds))


def _lease_remaining(lease_until: str, now: str) -> float:
    """Compare lease instants independently of the UTC offset and fraction spelling."""
    return (_parse_time(lease_until) - _parse_time(now)).total_seconds()


@dataclass(frozen=True)
class LeasedWork:
    work_id: int
    work_type: str
    subject_ref: str
    subject_revision: int
    scope_id: str
    project_id: str | None
    branch_id: str | None
    lease_token: int
    attempt: int
    lease_owner: str

    @property
    def lease(self) -> tuple[int, int, str]:
        """``(work_id, lease_token, lease_owner)``: the fence every mutation takes."""
        return self.work_id, self.lease_token, self.lease_owner


@dataclass(frozen=True)
class WorkMutation:
    work_id: int
    disposition: str
    state: str
    lease_token: int | None = None


@dataclass(frozen=True)
class OperatorRetryMutation:
    """Auditable result of one explicit, operator-authorized retry request.

    This is intentionally separate from ``WorkMutation``: an operator request
    may be rejected without owning a lease, and must never impersonate a
    worker completion/failure transition.
    """

    work_id: int
    disposition: str
    state: str
    reason: str | None = None
    operation_id: str | None = None


def _retry_disposition(work_type: str, state: str) -> tuple[str, str, str] | None:
    """Return a decision for work that cannot enter the failed-item retry path."""
    if work_type == "evaluate_candidate":
        return "rejected", state, "new_evidence_required"
    if state == "failed":
        return None
    if state == "pending":
        return "already_pending", state, "idempotent"
    return "rejected", state, "leased" if state == "leased" else "state_not_failed"


class WorkItems(EmbeddingQueue):
    """Schedule, lease and complete work in the current transaction."""

    def derivation_feedback(self, work_id: int) -> dict[str, str] | None:
        """Read repair metadata only for a work item granted the bounded retry."""
        conn = self._tx._check()
        row = conn.execute("SELECT last_error_code FROM work_items WHERE work_id=?", (work_id,)).fetchone()
        if row is None or f"{DERIVATION_RETRY_MARKER}|" not in str(row[0] or ""):
            return None
        # A successful fragment consumes this feedback, not the durable retry
        # allowance. Keep the marker/history, but never replay an old page's
        # validation error after its checkpoint (including later timeouts).
        outstanding = str(row[0]).lower().rsplit("consolidation_chunk_pending", 1)[-1]
        if not {"derivation_invalid", "input_invalid"}.intersection(outstanding.split("|")):
            return None
        # A summary or a claim dropped beside an accepted result (stage ``summary`` or ``claim``) failed nothing; the
        # feedback is the refusal's.
        detail = conn.execute(
            """SELECT error_code,error_field FROM work_error_details WHERE work_id=?
               AND lower(error_code) IN ('derivation_invalid','input_invalid') AND stage NOT IN ('summary','claim')
               ORDER BY detail_id DESC LIMIT 1""",
            (work_id,),
        ).fetchone()
        return validation_feedback(*(detail if detail is not None else (None, None)))

    def _visible_filter(self) -> tuple[str, tuple]:
        """SQL restricting work_items to the context's scopes, project and branch."""
        context = self._tx.context
        scopes = sorted(context.allowed_scope_ids)
        return (
            f"scope_id IN ({_marks(scopes)}) AND (project_id IS NULL OR project_id=?) AND (branch_id IS NULL OR branch_id=?)",
            (*scopes, context.project_id, context.branch_id),
        )

    def pending_depth(self, work_type: str) -> int:
        """How many rows of one type are pending, whether or not they are ready yet."""
        visible, params = self._visible_filter()
        return int(
            self._tx._check()
            .execute(
                f"SELECT count(*) FROM work_items WHERE state='pending' AND work_type=? AND {visible}",
                (work_type, *params),
            )
            .fetchone()[0]
        )

    def other_work_ready(self, *, now: str, kinds: frozenset[str]) -> bool:
        """Whether any of these work types has a row ready to claim now.

        The caller passes what this pass can still do, so a queue it cannot
        touch -- an instance with no embedding credential holds thousands --
        does not count as work waiting behind anything.
        """
        if not kinds:
            return False
        visible, params = self._visible_filter()
        marks = _marks(sorted(kinds))
        return (
            self._tx._check()
            .execute(
                f"""SELECT 1 FROM work_items WHERE state='pending' AND work_type IN ({marks})
                AND available_at<=? AND {visible} LIMIT 1""",
                (*sorted(kinds), now, *params),
            )
            .fetchone()
            is not None
        )

    def _context_denial(self, row) -> str | None:
        """Why this context may not act on the row, or None when it may."""
        context = self._tx.context
        if row["scope_id"] not in context.allowed_scope_ids:
            return "scope_denied"
        if row["project_id"] is not None and row["project_id"] != context.project_id:
            return "project_denied"
        if row["branch_id"] is not None and row["branch_id"] != context.branch_id:
            return "branch_denied"
        return None

    def _retry_subject_reason(self, row, *, current_epoch: int | None = None) -> str | None:
        return _RETRY_SUBJECT_REASON[row["work_type"]](
            self._tx, row["subject_ref"], row["subject_revision"], current_epoch=current_epoch
        )

    def enqueue(self, work_type: str, subject_ref: str, subject_revision: int, *, available_at: str) -> bool:
        conn = self._tx._check(write=True)
        if work_type not in ALLOWED_WORK_TYPES:
            raise ContractError("INPUT_INVALID", "work_type")
        if type(subject_ref) is not str or not subject_ref or type(subject_revision) is not int or subject_revision < 1:
            raise ContractError("INPUT_INVALID", "work_subject")
        scope_id, project_id, branch_id = _SUBJECT_CONTEXT[work_type](self._tx, subject_ref, subject_revision)
        before = conn.total_changes
        conn.execute(
            """INSERT INTO work_items(work_type,subject_ref,subject_revision,scope_id,project_id,branch_id,available_at)
               VALUES (?,?,?,?,?,?,?) ON CONFLICT(work_type,subject_ref,subject_revision) DO NOTHING""",
            (work_type, subject_ref, subject_revision, scope_id, project_id, branch_id, available_at),
        )
        return conn.total_changes > before

    def release_stale(self, now: str) -> int:
        conn = self._tx._check(write=True)
        visible, params = self._visible_filter()
        base = f"state='leased' AND {visible} AND lease_until IS NOT NULL AND lease_until < ?"
        params = (*params, now)
        # A lost lease is a failure like another: a history a budget is read from is added to, anything else replaced.
        spent = conn.execute(
            f"SELECT work_id,last_error_code FROM work_items WHERE {base} AND attempt >= ?",
            (*params, MAX_RECOVERABLE_ATTEMPTS),
        ).fetchall()
        for work_id, prior in spent:
            code = _within(f"{prior}|lease_exhausted") if _preserve_retry_history(prior) else "lease_exhausted"
            conn.execute(
                """UPDATE work_items SET state='failed',lease_owner=NULL,lease_until=NULL,last_error_code=?,
                   available_at=? WHERE work_id=? AND state='leased'""",
                (code, now, work_id),
            )
        failed = len(spent)
        released = conn.execute(
            f"""UPDATE work_items SET state='pending',lease_owner=NULL,lease_until=NULL
                WHERE {base} AND attempt < ?""",
            (*params, MAX_RECOVERABLE_ATTEMPTS),
        ).rowcount
        return failed + released

    def claim_next(
        self,
        owner: str,
        now: str,
        *,
        lease_seconds: float,
        limit: int = 1,
        allowed_work_types: frozenset[str] = ALLOWED_WORK_TYPES,
        fresh_lane: bool = False,
    ) -> tuple[LeasedWork, ...]:
        """Lease ready work: purge first, candidate evaluation last until it has waited
        ``EVALUATION_AGING_SECONDS``, FIFO by availability within each.

        ``fresh_lane`` places ready fresh conversation work (see
        ``FRESH_LANE_ORIGINS``) after purge and before the rest of the FIFO
        page.  Without fresh work the result is exactly the FIFO claim.
        """
        conn = self._tx._check(write=True)
        # The page is bounded by what one pass may hold, which is the worker's
        # own ``max_items`` (``core/worker.py``: up to 200).  It was 32 while a
        # pass was, and a pass that claims its embed group in one page then
        # asked for more than it was allowed to.
        if type(owner) is not str or not owner or type(limit) is not int or not 1 <= limit <= MAX_CLAIM_PAGE:
            raise ContractError("INPUT_INVALID", "work_claim")
        if type(lease_seconds) not in (int, float) or not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ContractError("INPUT_INVALID", "lease_seconds")
        if type(allowed_work_types) is not frozenset or not allowed_work_types <= ALLOWED_WORK_TYPES:
            raise ContractError("INPUT_INVALID", "work_types")
        if type(fresh_lane) is not bool:
            raise ContractError("INPUT_INVALID", "work_claim")
        if not allowed_work_types:
            return ()
        visible, params = self._visible_filter()
        kinds = sorted(allowed_work_types)
        self.release_stale(now)
        rows = conn.execute(
            f"""SELECT work_id,work_type FROM work_items
                WHERE state='pending' AND available_at<=? AND {visible} AND work_type IN ({_marks(kinds)})
                ORDER BY {_AGED_CLAIM_ORDER}, available_at, work_id LIMIT ?""",
            (now, *params, *kinds, _after(now, -EVALUATION_AGING_SECONDS), limit),
        ).fetchall()
        if fresh_lane:
            rows = self._fresh_first(rows, now=now, limit=limit, kinds=allowed_work_types)
        until = _after(now, float(lease_seconds))
        claimed = []
        for row in rows:
            # The write transaction is IMMEDIATE, so the pending rows selected
            # above cannot change under this loop; the new token is the fence
            # every later mutation must present.
            conn.execute(
                """UPDATE work_items SET state='leased',lease_token=lease_token+1,lease_owner=?,lease_until=?,attempt=attempt+1
                   WHERE work_id=?""",
                (owner, until, row["work_id"]),
            )
            item = conn.execute("SELECT * FROM work_items WHERE work_id=?", (row["work_id"],)).fetchone()
            if item["work_type"] == "rebuild_projection":
                # The same work id can be leased again. Drop its previous name reservations
                # before another writer can mistake them for this lease's reservations.
                conn.execute(
                    "UPDATE entity_names SET work_id=NULL WHERE work_id=? AND name_id IN "
                    "(SELECT name_id FROM entity_name_mentions WHERE claim_ref=? AND claim_revision=?)",
                    (item["work_id"], item["subject_ref"], item["subject_revision"]),
                )
            claimed.append(
                LeasedWork(
                    item["work_id"],
                    item["work_type"],
                    item["subject_ref"],
                    item["subject_revision"],
                    item["scope_id"],
                    item["project_id"],
                    item["branch_id"],
                    item["lease_token"],
                    item["attempt"],
                    owner,
                )
            )
        return tuple(claimed)

    def _fresh_first(self, fifo, *, now: str, limit: int, kinds: frozenset[str]) -> list:
        """Purge rows, then ready fresh conversation work, then the rest of the FIFO page."""
        purges = [row for row in fifo if row["work_type"] == "purge"]
        lane = sorted(kinds & FRESH_LANE_ORIGINS.keys())
        if len(purges) == limit or not lane:
            return fifo
        since = fresh_since(now)
        visible, params = self._visible_filter()
        origins = " OR ".join(f"(w.work_type=? AND e.origin IN ({_marks(FRESH_LANE_ORIGINS[kind])}))" for kind in lane)
        origin_params = tuple(value for kind in lane for value in (kind, *sorted(FRESH_LANE_ORIGINS[kind])))
        # Work never becomes available before its source was persisted, so the
        # window bounds the index range as well as the source it joins.
        fresh = (
            self._tx._check()
            .execute(
                f"""SELECT w.work_id,w.work_type FROM (
                    SELECT work_id,work_type,subject_ref,subject_revision,available_at FROM work_items
                    WHERE state='pending' AND available_at>=? AND available_at<=? AND {visible}
                      AND work_type IN ({_marks(lane)})
                    ORDER BY available_at DESC,work_id DESC LIMIT ?) w
                JOIN source_events e ON e.event_id=w.subject_ref AND e.source_revision=w.subject_revision
                WHERE e.persisted_at>=? AND ({origins})
                ORDER BY w.available_at,w.work_id LIMIT ?""",
                (since, now, *params, *lane, FRESH_LANE_SCAN_ROWS, since, *origin_params, limit - len(purges)),
            )
            .fetchall()
        )
        chosen = {row["work_id"] for row in (*purges, *fresh)}
        return [*purges, *fresh, *(row for row in fifo if row["work_id"] not in chosen)][:limit]

    def recover_transient_failures(
        self,
        *,
        now: str,
        allowed_work_types: frozenset[str],
        cooldown_seconds: float = 3600,
        max_recoveries: int = 2,
        limit: int = 32,
    ) -> int:
        """Grant bounded, cooled-down attempts without resetting lifetime attempts.

        Scope/project/branch and source/delete authority are rechecked in this
        transaction; a later worker still owns its lease and epoch checks.
        The existing error field records the finite automatic recovery budget.
        """
        conn = self._tx._check(write=True)
        if not allowed_work_types <= ALLOWED_WORK_TYPES or not 0 <= max_recoveries <= 4:
            raise ContractError("INPUT_INVALID", "auto_recovery")
        if (
            not math.isfinite(cooldown_seconds)
            or not 60 <= cooldown_seconds <= 86400
            or not 1 <= limit <= MAX_RECOVERY_PAGE
        ):
            raise ContractError("INPUT_INVALID", "auto_recovery_budget")
        cutoff = _after(now, -cooldown_seconds)
        kinds = sorted(allowed_work_types & frozenset(_RETRY_SUBJECT_REASON))
        if not kinds or max_recoveries == 0:
            return 0
        visible, params = self._visible_filter()
        error_filter = " OR ".join("(last_error_code=? OR last_error_code LIKE ?)" for _ in AUTO_RECOVERABLE_ERRORS)
        error_params = tuple(value for code in sorted(AUTO_RECOVERABLE_ERRORS) for value in (code, f"%|{code}"))
        # Rows that already spent their automatic budget are excluded in SQL so
        # they cannot fill the page; ``_auto_count`` below remains the authority.
        exhausted_filter = " AND ".join("last_error_code NOT LIKE ?" for _ in range(max_recoveries, 5))
        exhausted_params = tuple(f"%auto_retry:{count}|%" for count in range(max_recoveries, 5))
        rows = conn.execute(
            f"""SELECT * FROM work_items WHERE state='failed' AND available_at<=?
                AND {visible} AND ({error_filter}) AND {exhausted_filter}
                AND work_type IN ({_marks(kinds)})
                ORDER BY {_CLAIM_ORDER},available_at,work_id LIMIT ?""",
            (cutoff, *params, *error_params, *exhausted_params, *kinds, limit),
        ).fetchall()
        recovered = 0
        for row in rows:
            error = str(row["last_error_code"] or "")
            if _auto_count(error) >= max_recoveries or _failure_kind(error) not in AUTO_RECOVERABLE_ERRORS:
                continue
            try:
                revoked = self._retry_subject_reason(row) is not None
            except ContractError:
                revoked = True
            if revoked:
                conn.execute(
                    "UPDATE work_items SET state='obsolete',last_error_code='authority_revoked' WHERE work_id=? AND state='failed'",
                    (row["work_id"],),
                )
                continue
            recovered += conn.execute(
                """UPDATE work_items SET state='pending',available_at=?,last_error_code=?,
                   lease_owner=NULL,lease_until=NULL WHERE work_id=? AND state='failed'""",
                (now, _auto_retry_stamp(error), row["work_id"]),
            ).rowcount
        return recovered

    def recover_oversized_consolidations(self, *, now: str, formatter, limit: int = 8) -> int:
        """One repair of old input-budget failures, never a general model retry.

        The old worker retained INPUT_INVALID but lost its field. Therefore
        re-open the live original source and reproduce the exact formatter
        failure before changing its state. Retain the old attempt count and a
        durable repair marker so an invalid model result cannot loop here.
        """
        conn = self._tx._check(write=True)
        if type(limit) is not int or not 1 <= limit <= 16:
            raise ContractError("INPUT_INVALID", "repair_limit")
        visible, params = self._visible_filter()
        rows = conn.execute(
            f"""SELECT * FROM work_items
                WHERE work_type='consolidate' AND state='failed'
                  AND last_error_code='INPUT_INVALID' AND consolidation_offset=0 AND {visible}
                ORDER BY work_id LIMIT ?""",
            (*params, limit),
        ).fetchall()
        recovered = 0
        for row in rows:
            oversized = False
            try:
                self._tx.claims.require_live_source(row["subject_ref"], row["subject_revision"])
                source = self._tx.source(row["subject_ref"], row["subject_revision"])
                if source is not None and not source.capture_gaps and source.event["capture_state"] == "complete":
                    episode = self._tx.episodes.source_episode(source.ref, source.revision)
                    formatter((source,), episode_ref=episode.ref if episode else None)
            except ContractError as exc:
                if exc.code in {"STORAGE_UNAVAILABLE", "DEADLINE_EXCEEDED"}:
                    raise
                oversized = exc.code == "INPUT_INVALID" and exc.field == "consolidation_input_budget"
            if not oversized:
                # Keep the failure and its attempt count, but do not let a
                # non-budget failure repeatedly occupy this bounded repair
                # page. No original content or claim is changed.
                conn.execute(
                    """UPDATE work_items SET last_error_code='chunk_checked:1106|INPUT_INVALID'
                    WHERE work_id=? AND state='failed' AND last_error_code='INPUT_INVALID'
                    AND consolidation_offset=0""",
                    (row["work_id"],),
                )
                continue
            recovered += conn.execute(
                """UPDATE work_items SET state='pending',available_at=?,lease_owner=NULL,lease_until=NULL,
                   last_error_code=?
                   WHERE work_id=? AND state='failed' AND last_error_code='INPUT_INVALID'
                   AND consolidation_offset=0""",
                (now, f"{CHUNK_UPGRADE_MARKER}|INPUT_INVALID", row["work_id"]),
            ).rowcount
        return recovered

    def reopen_oversized_candidate_evaluation(self, work_id: int, *, now: str, fits: bool) -> int:
        """Re-queue (or durably mark) one work item whose evaluation was oversized.

        ``recover_oversized_consolidations`` cannot reach these rows: it filters
        on ``work_type='consolidate'`` and reproduces the failure by re-formatting
        a single source, while a candidate evaluation formats a candidate against
        its whole evidence set. The candidate worker also lowercases its error
        codes, so that sibling's exact-case ``INPUT_INVALID`` filter would miss
        them even without the work-type gate -- matching here is case-insensitive.

        The caller owns deciding ``fits`` and moving the candidate tables with it;
        this only touches ``work_items``. Either way the row gets a durable
        marker, so a still-oversized evaluation cannot loop through the bounded
        repair page.
        """
        conn = self._tx._check(write=True)
        marker = "budget_upgrade" if fits else "budget_checked"
        return conn.execute(
            """UPDATE work_items SET state=?,available_at=COALESCE(?,available_at),
               lease_owner=NULL,lease_until=NULL,last_error_code=?
               WHERE work_id=? AND state='failed' AND UPPER(last_error_code)='INPUT_INVALID'""",
            (
                "pending" if fits else "failed",
                now if fits else None,
                f"{marker}:{SCHEMA_VERSION}|input_invalid",
                work_id,
            ),
        ).rowcount

    def recover_interrupted_attempts(self, *, now: str, allowed_work_types: frozenset[str], limit: int = 8) -> int:
        """Grant one fresh attempt to an evaluation whose attempt no process finished.

        The at-most-once fence is committed before the model call, so a worker
        killed after it -- a gateway stop, a lost lease, a machine restart --
        leaves the candidate failed with nothing recorded and nothing to look at,
        and waiting for new evidence never comes for a candidate whose evidence is
        already in.

        One attempt: the marker says it was given, and ``attempt`` bounds it even
        when a later failure overwrites the marker -- an interruption observed as
        a lost lease reports ``lease_exhausted`` and keeps no history.  So a row
        interrupted again and again is offered to the model at most
        ``MAX_RECOVERABLE_ATTEMPTS`` times in its life, whatever it is called
        each time.  A retry revalidates source and epoch authority first, exactly
        as the first attempt did.
        """
        conn = self._tx._check(write=True)
        if type(limit) is not int or not 1 <= limit <= MAX_RECOVERY_PAGE:
            raise ContractError("INPUT_INVALID", "retry_limit")
        if "evaluate_candidate" not in allowed_work_types:
            return 0
        visible, params = self._visible_filter()
        rows = conn.execute(
            f"""SELECT work_id,last_error_code FROM work_items
                WHERE state='failed' AND work_type='evaluate_candidate' AND {visible}
                AND (lower(last_error_code)='candidate_attempt_interrupted'
                     OR lower(last_error_code) LIKE '%|candidate_attempt_interrupted')
                AND last_error_code NOT LIKE ? AND attempt < ? ORDER BY work_id LIMIT ?""",
            (*params, f"%{INTERRUPTED_RETRY_MARKER}|%", MAX_RECOVERABLE_ATTEMPTS, limit),
        ).fetchall()
        recovered = 0
        for row in rows:
            if not self._tx.candidates.reopen_evaluation(row["work_id"], now=now, reason="interrupted_retry"):
                continue
            recovered += conn.execute(
                """UPDATE work_items SET state='pending',available_at=?,lease_owner=NULL,
                   lease_until=NULL,last_error_code=? WHERE work_id=? AND state='failed'""",
                (now, _within(f"{INTERRUPTED_RETRY_MARKER}|{row['last_error_code']}"), row["work_id"]),
            ).rowcount
        return recovered

    def recover_invalid_derivations(self, *, now: str, allowed_work_types: frozenset[str], limit: int = 32) -> int:
        """Grant legacy invalid results the same single extra attempt as new work.

        Reuse the operator retry's three-table transition for candidates. The
        next worker revalidates source/epoch authority before any model call.
        The durable marker is independent of schema generation and attempts.
        """
        conn = self._tx._check(write=True)
        if type(limit) is not int or not 1 <= limit <= MAX_RECOVERY_PAGE:
            raise ContractError("INPUT_INVALID", "retry_limit")
        kinds = sorted(allowed_work_types & {"consolidate", "evaluate_candidate"})
        if not kinds:
            return 0
        visible, params = self._visible_filter()
        rows = conn.execute(
            f"""SELECT work_id,work_type,last_error_code FROM work_items
                WHERE state='failed' AND {visible} AND work_type IN ({_marks(kinds)})
                AND (lower(last_error_code)='derivation_invalid'
                     OR lower(last_error_code) LIKE '%|derivation_invalid')
                AND last_error_code NOT LIKE ? ORDER BY work_id LIMIT ?""",
            (*params, *kinds, f"%{DERIVATION_RETRY_MARKER}|%", limit),
        ).fetchall()
        return sum(
            self._reopen_failed(row["work_id"], row["work_type"], row["last_error_code"], now=now, automatic=True)
            for row in rows
        )

    def retry_failed(self, *, now: str, include_terminal: bool = False, limit: int = 64, dry_run: bool = True) -> dict:
        """Grant one bounded re-look to failures a shipped fix may have cured.

        Distinct from ``recover_transient_failures``, which is automatic and
        governed by a per-item budget that these rows have already spent, and
        from the two oversized-evaluation repairs, which reproduce a specific
        failure before deciding.  This one reproduces nothing: it records that
        an operator granted the attempt and lets the next run decide.

        A candidate evaluation needs three tables to move together -- evaluation,
        lifecycle and work item -- or ``begin_model_attempt`` refuses the next
        attempt as ``candidate_attempt_interrupted``.  Other work types need
        only the work item.

        Idempotent: every row is stamped with the release that granted it, and
        a row already carrying this release's stamp is skipped.  Skipped rows the
        request covers are counted (``already_retried``, the first ids in
        ``already_retried_work_ids``): each failed again after this release
        re-opened it, and a run that re-opens nothing says so rather than
        nothing.  However many failed rows a run passes over, by design or
        already re-opened, it reads on to the rows it may re-open
        (``_failed_rows``), and each row it re-opens starts its attempts afresh
        (``_reopen_failed``).

        Preview and apply share the candidate retry fence. Declined evaluations are counted with the first
        ``MAX_RECOVERY_PAGE`` work ids and reasons; neither mode retires them or consumes a retry slot for them.
        """
        validate_page(limit)
        if type(include_terminal) is not bool or type(dry_run) is not bool:
            raise ContractError("INPUT_INVALID", "retry_flags")
        # A preview must be able to run inside a read transaction, which is what
        # makes "look before you touch production" the cheap option.
        conn = self._tx._check(write=not dry_run)
        report = {
            "examined": 0,
            "retried": 0,
            "by_kind": {},
            "applied": not dry_run,
            "already_retried": 0,
            "already_retried_work_ids": [],
            "declined": 0,
            "declined_work_items": [],
            "retried_work_ids": [],
            "reconciled_evaluations": 0,
        }
        # Reconciliation and a retry grant can change the same work item; both share one page slot.
        handled: set[int] = set()
        for row in self._failed_rows(conn):
            if len(handled) >= limit:
                break
            report["examined"] += 1
            code = self._reconcile_retry_row(row, report, now=now, dry_run=dry_run, handled=handled)
            if not wanted(code, include_terminal=include_terminal):
                continue
            if already_retried(code, generation=RETRY_GENERATION):
                report["already_retried"] += 1
                # As many as one explicit retry by work id takes (``worker_entry --retry-failed``).
                if len(report["already_retried_work_ids"]) < MAX_OPERATOR_PAGE:
                    report["already_retried_work_ids"].append(int(row["work_id"]))
                continue
            if self._decline_candidate_retry(row, report):
                continue
            if not dry_run and not self._reopen_failed(row["work_id"], row["work_type"], code, now=now):
                continue
            kind = str(code or "").rsplit("|", 1)[-1]
            report["by_kind"][kind] = report["by_kind"].get(kind, 0) + 1
            report["retried"] += 1
            report["retried_work_ids"].append(int(row["work_id"]))
            handled.add(row["work_id"])
        # The automatic recovery read every embed's subject as a source, so a claim embed a provider failed was made
        # obsolete instead of reopened (``_embed_retry_reason``).  A readable head it left without a vector is worked
        # on again, with the attempts of new work, and one an earlier conversion never queued is queued.
        heads = self._claim_heads_without_vectors(limit=limit)
        report["claim_embeds_reopened"] = sum(1 for work_id, _ref, _revision in heads if work_id is not None)
        report["claim_embeds_queued"] = sum(1 for work_id, _ref, _revision in heads if work_id is None)
        if not dry_run:
            for work_id, ref, revision in heads:
                if work_id is None:
                    self.enqueue("embed", ref, revision, available_at=now)
                else:
                    conn.execute(
                        """UPDATE work_items SET state='pending',attempt=0,available_at=?,lease_owner=NULL,
                                    lease_until=NULL,last_error_code=? WHERE work_id=? AND state='obsolete'""",
                        (now, marked("authority_revoked", generation=RETRY_GENERATION), work_id),
                    )
        # Captures a replay gave up after its tries (``inbox_rules.GAVE_UP``) go back to it, their tries counted
        # anew: whatever kept them out has been fixed, or they are given up again, visibly.  Only the partition this
        # config's replay takes (``replay_inbox``): returned by what the config could see, a row of another would go
        # back to a replay that never takes it.
        context = self._tx.context
        scopes = sorted(context.allowed_scope_ids)
        abandoned = conn.execute(
            f"""SELECT token,last_error_code FROM capture_inbox WHERE last_error_code LIKE 'GAVE_UP|%'
                AND scope_id IN ({_marks(scopes)}) AND project_id IS ? AND branch_id IS ?""",
            (*scopes, context.project_id, context.branch_id),
        ).fetchall()
        report["inbox_given_up"] = len(abandoned)
        report["inbox_by_kind"] = {}
        for _token, code in abandoned:
            kind = str(code).rsplit("|", 1)[-1]
            report["inbox_by_kind"][kind] = report["inbox_by_kind"].get(kind, 0) + 1
        # A capture an earlier release refused as ACCESS_DENIED, most often one under a deleted message's key, stays
        # in the inbox for good with doctor's capture_ingress_blocked up, and nothing but a hand removes it.  The
        # replay cancels a copy of the deleted message and stores another message under a key of its own
        # (``Sources.refuse_under_a_deleted_key``), so such rows go back to it once asked.
        refused = conn.execute(
            f"""SELECT token FROM capture_inbox WHERE last_error_code='ACCESS_DENIED'
                AND scope_id IN ({_marks(scopes)}) AND project_id IS ? AND branch_id IS ?""",
            (*scopes, context.project_id, context.branch_id),
        ).fetchall()
        report["inbox_refused"] = len(refused)
        if not dry_run:
            # One the rekey path gave up goes back to it: returned as never tried, the plain replay would meet the old
            # collision, and a row it put off would be matched by a delete through the key it had taken.
            conn.executemany(
                "UPDATE capture_inbox SET last_error_code=? WHERE token=?",
                [("VERSION_CONFLICT" if deferred_path(code) == "rekey" else None, token) for token, code in abandoned],
            )
            conn.executemany(
                "UPDATE capture_inbox SET last_error_code=NULL WHERE token=?", [(row[0],) for row in refused]
            )
        return report

    def _reconcile_retry_row(self, row, report, *, now, dry_run, handled: set[int]):
        """Make stopped question bookkeeping inspectable even when its release grant was already spent."""
        if row["work_type"] == "evaluate_candidate":
            reconciled = self._tx.candidates.reconcile_failed_evaluation(row["work_id"], now=now, dry_run=dry_run)
            report["reconciled_evaluations"] += reconciled
            if reconciled:
                handled.add(row["work_id"])
        return row["last_error_code"]

    def _decline_candidate_retry(self, row, report: dict) -> bool:
        """Report a candidate's read-only retry refusal with bounded detail; other work stays eligible."""
        if row["work_type"] != "evaluate_candidate":
            return False
        rejection = self._tx.candidates.evaluation_retry_rejection(row["work_id"], include_unfinished=True)
        if rejection is None:
            return False
        report["declined"] += 1
        if len(report["declined_work_items"]) < MAX_RECOVERY_PAGE:
            report["declined_work_items"].append({"work_id": int(row["work_id"]), "reason": rejection})
        return True

    def _failed_rows(self, conn) -> Iterator:
        """Every failed row of this context, ``RETRY_SCAN_PAGE`` at a time, in the order ``work_ready`` keeps them (by
        when each failed): a run reaches the rows it may re-open however many come before them.

        A page is read whole before the caller changes a row, and the next one starts after the last row read: a row
        the caller re-opened has left the failed rows, and no row is read twice or passed over."""
        visible, params = self._visible_filter()
        last = ("", 0)
        while True:
            page = conn.execute(
                f"""SELECT work_id, work_type, last_error_code, available_at FROM work_items
                    WHERE state='failed' AND (available_at, work_id)>(?,?) AND {visible}
                    ORDER BY available_at, work_id LIMIT ?""",
                (*last, *params, RETRY_SCAN_PAGE),
            ).fetchall()
            yield from page
            if len(page) < RETRY_SCAN_PAGE:
                return
            last = (page[-1]["available_at"], page[-1]["work_id"])

    def due_unreached(self, *, before: str) -> list[dict]:
        """Work of every partition due since before ``before`` and still waiting, by partition and type: pending, or
        leased by a worker whose lease ran out then (only a pass of the same partition releases it).  For the doctor;
        counts and times only, read through ``work_ready``."""
        rows = (
            self._tx._check()
            .execute(
                """SELECT scope_id,project_id,branch_id,work_type,count(*) AS n,
                      min(CASE WHEN state='leased' THEN lease_until ELSE available_at END) AS oldest FROM work_items
               WHERE (state='pending' AND available_at<?) OR (state='leased' AND lease_until<?)
               GROUP BY scope_id,project_id,branch_id,work_type""",
                (before, before),
            )
            .fetchall()
        )
        return [
            {
                "scope_id": row["scope_id"],
                "project_id": row["project_id"],
                "branch_id": row["branch_id"],
                "work_type": row["work_type"],
                "work": int(row["n"]),
                "oldest": row["oldest"],
            }
            for row in rows
        ]

    def _reopen_failed(self, work_id: int, work_type: str, code: object, *, now: str, automatic: bool = False) -> bool:
        """Move one failed row, and its sibling tables when it has any.

        An operator's re-look starts the row's attempts afresh, as new work's: left at the count that failed it, the
        row would fail again at its first transient fault.  The derivation retry keeps the count; it is one extra
        attempt (``recover_invalid_derivations``)."""
        conn = self._tx._check(write=True)
        reason = "derivation_retry" if automatic else "operator_retry"
        if work_type == "evaluate_candidate" and not self._tx.candidates.reopen_evaluation(
            work_id, now=now, reason=reason
        ):
            return False
        attempts = "attempt" if automatic else "0"
        return (
            conn.execute(
                f"""UPDATE work_items SET state='pending',attempt={attempts},available_at=?,lease_owner=NULL,
               lease_until=NULL,last_error_code=? WHERE work_id=? AND state='failed'""",
                (
                    now,
                    _within(
                        f"{DERIVATION_RETRY_MARKER}|{code}" if automatic else marked(code, generation=RETRY_GENERATION)
                    ),
                    work_id,
                ),
            ).rowcount
            == 1
        )

    def defer_without_attempt(
        self, work_id: int, lease_token: int, owner: str, *, now: str, error_code: str, seconds: float = 3600
    ) -> WorkMutation:
        """A pre-network capability/budget refusal is not a model attempt."""
        conn = self._tx._check(write=True)
        if not self._verify_lease(work_id, lease_token, owner, now=now):
            return WorkMutation(work_id, "stale", self.read_state(work_id) or "stale", lease_token)
        prior = conn.execute("SELECT last_error_code FROM work_items WHERE work_id=?", (work_id,)).fetchone()[0]
        if _preserve_retry_history(prior):
            error_code = _within(f"{prior}|{error_code}")
        conn.execute(
            f"""UPDATE work_items SET state='pending',attempt=MAX(0,attempt-1),
                       lease_owner=NULL,lease_until=NULL,last_error_code=?,available_at=?
                       WHERE {_LEASED_ROW}""",
            (error_code, _after(now, seconds), work_id, lease_token, owner),
        )
        return WorkMutation(work_id, "deferred", "pending", lease_token)

    def read_state(self, work_id: int) -> str | None:
        row = self._tx._check().execute("SELECT state FROM work_items WHERE work_id=?", (work_id,)).fetchone()
        return None if row is None else row["state"]

    def operator_retry_failed(
        self,
        work_ids,
        *,
        now: str,
        operation_id: str,
        expected_memory_epoch: int | None = None,
        max_items: int = 8,
    ) -> tuple[OperatorRetryMutation, ...]:
        """Requeue explicitly named failed work under the current fence.

        This is the sole storage mutation for the maintenance retry entry
        point.  It never resets the automatic attempt counter (the normal
        three-attempt ceiling therefore remains intact), and it does not
        lease or execute work.  A subsequent bounded worker drain performs the
        actual operation.  Calling the same operation again while its item is
        pending is idempotent.
        """
        conn = self._tx._check(write=True)
        self._check_operator_request(work_ids, operation_id, expected_memory_epoch, max_items)
        # Validate the timestamp through the strict UTC parser before any row
        # is changed.  The worker's clock remains the authority.
        _parse_time(now)
        current_epoch = self._tx.status().memory_epoch
        results = []
        for work_id in work_ids:
            row = conn.execute("SELECT * FROM work_items WHERE work_id=?", (work_id,)).fetchone()
            verdict = self._operator_retry_rejection(row, operation_id, current_epoch)
            if verdict is None:
                verdict = self._operator_requeue(conn, row, operation_id, now)
            results.append(OperatorRetryMutation(work_id, *verdict, operation_id))
        # Re-read the epoch in the same write transaction.  A concurrent
        # deletion/update cannot commit between the eligibility check and the
        # requeue commit; a later worker fence still handles a post-commit
        # epoch change conservatively.
        if expected_memory_epoch is not None and self._tx.status().memory_epoch != expected_memory_epoch:
            raise ContractError("VERSION_CONFLICT", "memory_epoch")
        return tuple(results)

    def _check_operator_request(self, work_ids, operation_id, expected_memory_epoch, max_items) -> None:
        if self._tx.context.actor_origin not in OPERATOR_ORIGINS:
            raise ContractError("ACCESS_DENIED", "operator_origin")
        if not isinstance(work_ids, (tuple, list)) or not work_ids:
            raise ContractError("INPUT_INVALID", "retry_work_ids")
        if type(max_items) is not int or not 1 <= max_items <= MAX_OPERATOR_PAGE or len(work_ids) > max_items:
            raise ContractError("INPUT_INVALID", "retry_budget")
        if any(type(value) is not int or value < 1 for value in work_ids) or len(set(work_ids)) != len(work_ids):
            raise ContractError("INPUT_INVALID", "retry_work_ids")
        if type(operation_id) is not str or not _OPERATION_ID.fullmatch(operation_id):
            raise ContractError("INPUT_INVALID", "retry_operation")
        if expected_memory_epoch is not None:
            if type(expected_memory_epoch) is not int or expected_memory_epoch < 0:
                raise ContractError("INPUT_INVALID", "memory_epoch")
            if self._tx.status().memory_epoch != expected_memory_epoch:
                raise ContractError("VERSION_CONFLICT", "memory_epoch")

    def _operator_retry_rejection(self, row, operation_id: str, current_epoch: int) -> tuple[str, str, str] | None:
        """``(disposition, state, reason)`` refusing the retry, or None to requeue."""
        if row is None:
            return "rejected", "missing", "work_missing"
        denied = self._context_denial(row)
        if denied is not None:
            return "rejected", "unavailable", denied
        state = row["state"]
        disposition = _retry_disposition(row["work_type"], state)
        if disposition is not None:
            return disposition
        prior_operations = _operator_retry_ids(row["last_error_code"] or "")
        if operation_id in prior_operations:
            return "already_retried", state, "idempotent"
        if len(prior_operations) >= MAX_OPERATOR_RETRIES:
            return "rejected", state, "operator_retry_budget"
        reason = self._retry_subject_reason(row, current_epoch=current_epoch)
        if reason is not None:
            return "rejected", state, reason
        return None

    @staticmethod
    def _operator_requeue(conn, row, operation_id: str, now: str) -> tuple[str, str, str]:
        # Keep attempt as-is: explicit maintenance retry grants exactly one
        # new worker execution and cannot silently restart the automatic
        # three-attempt budget.  The operation marker is durable in the
        # existing bounded error field for audit/idempotence.
        marker = f"operator_retry:{operation_id}"
        prior_error = row["last_error_code"] or ""
        stamped = marker if not prior_error else _within(f"{marker}|prior:{prior_error}")
        conn.execute(
            """UPDATE work_items SET state='pending',available_at=?,lease_owner=NULL,
               lease_until=NULL,last_error_code=?
               WHERE work_id=? AND state='failed'""",
            (now, stamped, row["work_id"]),
        )
        return "requeued", "pending", "operator_retry_requested"

    def _verify_lease(self, work_id: int, lease_token: int, owner: str, *, now: str) -> bool:
        # Verification is a read operation.  Mutations call it from a write
        # transaction, while staged vector publication uses the same fence
        # from a read-only guard immediately before external commit.
        row = (
            self._tx._check()
            .execute(
                """SELECT state,lease_token,lease_owner,lease_until,scope_id,project_id,branch_id
               FROM work_items WHERE work_id=?""",
                (work_id,),
            )
            .fetchone()
        )
        if row is None or row["state"] != "leased" or row["lease_token"] != lease_token or row["lease_owner"] != owner:
            return False
        if self._context_denial(row) is not None:
            return False
        return row["lease_until"] is None or _lease_remaining(row["lease_until"], now) > 0

    def _stale_mutation(self, conn, work_id: int) -> WorkMutation:
        row = conn.execute("SELECT state,lease_token FROM work_items WHERE work_id=?", (work_id,)).fetchone()
        if row is None:
            raise ContractError("SOURCE_MISSING", "work_item")
        return WorkMutation(work_id, "stale", row["state"], row["lease_token"])

    def complete(self, work_id: int, lease_token: int, owner: str, *, now: str) -> WorkMutation:
        conn = self._tx._check(write=True)
        if not self._verify_lease(work_id, lease_token, owner, now=now):
            return self._stale_mutation(conn, work_id)
        conn.execute(
            f"UPDATE work_items SET state='done',lease_owner=NULL,lease_until=NULL,last_error_code=NULL WHERE {_LEASED_ROW}",
            (work_id, lease_token, owner),
        )
        return WorkMutation(work_id, "completed", "done", lease_token)

    def complete_consolidation(
        self,
        work_id: int,
        lease_token: int,
        owner: str,
        *,
        now: str,
        covered_source_refs: frozenset[str],
        pending_sources: tuple[tuple[str, int, int], ...] = (),
    ) -> WorkMutation:
        conn = self._tx._check(write=True)
        if not self._verify_lease(work_id, lease_token, owner, now=now):
            return self.complete(work_id, lease_token, owner, now=now)
        current = conn.execute("SELECT * FROM work_items WHERE work_id=?", (work_id,)).fetchone()
        if (
            current["work_type"] != "consolidate"
            or f"{current['subject_ref']}@{current['subject_revision']}" not in covered_source_refs
        ):
            raise ContractError("DERIVATION_INVALID", "consolidation_subject_uncovered")
        result = self.complete(work_id, lease_token, owner, now=now)
        for ref, revision, token in pending_sources:
            if f"{ref}@{revision}" not in covered_source_refs:
                continue
            conn.execute(
                """UPDATE work_items SET state='done',lease_owner=NULL,lease_until=NULL,last_error_code=NULL
                   WHERE work_type='consolidate' AND subject_ref=? AND subject_revision=?
                   AND state='pending' AND lease_token=? AND scope_id=?
                   AND project_id IS ? AND branch_id IS ?""",
                (ref, revision, token, current["scope_id"], current["project_id"], current["branch_id"]),
            )
        return result

    def consolidation_offset(self, work_id: int, lease_token: int, owner: str, *, now: str) -> int:
        if not self._verify_lease(work_id, lease_token, owner, now=now):
            raise ContractError("ACCESS_DENIED", "lease_stale")
        row = (
            self._tx._check()
            .execute(
                "SELECT work_type,consolidation_offset FROM work_items WHERE work_id=?",
                (work_id,),
            )
            .fetchone()
        )
        if row["work_type"] != "consolidate":
            raise ContractError("INPUT_INVALID", "work_type")
        return row["consolidation_offset"]

    def advance_consolidation(
        self,
        work_id: int,
        lease_token: int,
        owner: str,
        *,
        now: str,
        expected_offset: int,
        next_offset: int,
        final: bool,
    ) -> WorkMutation:
        """Commit accepted claims and their exact window checkpoint together.

        This method is called in the same transaction as claim application.
        Successful pages do not spend the finite failed-attempt allowance.
        """
        conn = self._tx._check(write=True)
        current = self.consolidation_offset(work_id, lease_token, owner, now=now)
        if type(next_offset) is not int or next_offset <= expected_offset or current != expected_offset:
            raise ContractError("DERIVATION_INVALID", "consolidation_offset")
        conn.execute("UPDATE work_items SET consolidation_offset=? WHERE work_id=?", (next_offset, work_id))
        if final:
            return self.complete(work_id, lease_token, owner, now=now)
        return self.defer_without_attempt(
            work_id, lease_token, owner, now=now, error_code="consolidation_chunk_pending", seconds=0
        )

    def fail(
        self, work_id: int, lease_token: int, owner: str, *, error_code: str, now: str, recoverable: bool
    ) -> WorkMutation:
        conn = self._tx._check(write=True)
        if not self._verify_lease(work_id, lease_token, owner, now=now):
            return self._stale_mutation(conn, work_id)
        row = conn.execute(
            "SELECT attempt,last_error_code,work_type FROM work_items WHERE work_id=?", (work_id,)
        ).fetchone()
        attempt, prior = row["attempt"], row["last_error_code"]
        # Invalid output gets exactly one extra execution, even when earlier
        # infrastructure failures have consumed the ordinary attempt counter.
        invalid = _failure_kind(error_code).lower() == "derivation_invalid" and row["work_type"] in {
            "consolidate",
            "evaluate_candidate",
        }
        if invalid:
            recoverable = f"{DERIVATION_RETRY_MARKER}|" not in str(prior or "")
            if recoverable:
                prior = f"{DERIVATION_RETRY_MARKER}|{prior or ''}".rstrip("|")
        if recoverable and _failure_kind(error_code) in CAPACITY_REFUSALS:
            # The provider refused everyone; this lease never got an attempt, so
            # the attempt is refunded and the item is never given up on.  What
            # *is* bounded is how often it asks again: refunding alone left the
            # ordinary 60s ceiling in place and 197 items retried without pause
            # for an entire provider outage, which is the silent loop this
            # whole design exists to prevent.
            # Rebuilt from its count, the code keeps every budget mark it had, so an outage spends no budget.
            refusals = _capacity_count(prior) + 1
            retry_error = _within(f"{_budget_marks(str(prior or ''))}capacity:{refusals}|{error_code}")
            conn.execute(
                f"""UPDATE work_items SET state='pending',lease_owner=NULL,lease_until=NULL,
                   last_error_code=?,available_at=?,attempt=MAX(attempt-1,0) WHERE {_LEASED_ROW}""",
                (retry_error, _after(now, capacity_backoff_seconds(refusals)), work_id, lease_token, owner),
            )
            return WorkMutation(work_id, "retry", "pending", lease_token)
        history = _within(f"{prior}|{error_code}") if _preserve_retry_history(prior) else error_code
        if recoverable and (invalid or attempt < MAX_RECOVERABLE_ATTEMPTS):
            conn.execute(
                f"""UPDATE work_items SET state='pending',lease_owner=NULL,lease_until=NULL,last_error_code=?,available_at=?
                   WHERE {_LEASED_ROW}""",
                (history, _after(now, WORK_RETRY.delay(attempt)), work_id, lease_token, owner),
            )
            return WorkMutation(work_id, "retry", "pending", lease_token)
        conn.execute(
            f"""UPDATE work_items SET state='failed',lease_owner=NULL,lease_until=NULL,last_error_code=?,available_at=?
               WHERE {_LEASED_ROW}""",
            (history, now, work_id, lease_token, owner),
        )
        return WorkMutation(work_id, "failed", "failed", lease_token)

    def mark_obsolete(self, work_id: int, lease_token: int, owner: str, *, now: str) -> WorkMutation:
        conn = self._tx._check(write=True)
        if not self._verify_lease(work_id, lease_token, owner, now=now):
            return self._stale_mutation(conn, work_id)
        conn.execute(
            f"""UPDATE work_items SET state='obsolete',lease_owner=NULL,lease_until=NULL,last_error_code='authority_revoked'
               WHERE {_LEASED_ROW}""",
            (work_id, lease_token, owner),
        )
        return WorkMutation(work_id, "obsolete", "obsolete", lease_token)
