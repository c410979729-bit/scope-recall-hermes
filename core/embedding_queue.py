"""Embedding queue inspection, recovery and migration to another embedding space."""

from __future__ import annotations

import re
from uuid import uuid4

from ..contracts import ContractError
from .work_subjects import _embed_retry_reason, _marks

# A named cursor tracks the target space and the last embedding work item inspected.
# Only an explicit re-embedding run can reopen previously completed work.
RESPACE_CURSOR = "embed_respace"
#: What a reopened row says until it is embedded again.
RESPACE_MARKER = "embedding_space_changed"
#: Rows one page reopens at most; the embed queue's room bounds it further.
RESPACE_PAGE = 64
#: Work ids one page looks through at most.
RESPACE_SCAN = 4096
_SPACE_ID = re.compile(r"[0-9a-f]{64}")
#: A tool output whose vector the retention window expired keeps none (``runtime/vector_retention.py``).
_NOT_EXPIRED = """NOT EXISTS(SELECT 1 FROM expired_vectors x
                       WHERE x.source_ref=w.subject_ref AND x.source_revision=w.subject_revision)"""


def respace_refusal(run: dict | None, space_id: str, *, restart: bool) -> str | None:
    """Return the reason a migration needs an explicit restart, or None if it can proceed."""
    if run is None or restart:
        return None
    if not run["completed"]:
        return "respace_running"
    return "respace_finished" if run["embedding_space"] == space_id else None


class EmbeddingQueue:
    """Inspect and reopen embedding work within the caller's transaction."""

    def __init__(self, transaction) -> None:
        self._tx = transaction

    def _claim_heads_without_vectors(self, *, limit: int) -> list[tuple[int | None, str, int]]:
        """Return eligible (work ID or None, claim ID, head revision) tuples, bounded by limit.

        Match the retry policy on scope, project and branch before filling the page.
        """
        conn = self._tx._check()
        context = self._tx.context
        scopes = sorted(context.allowed_scope_ids)
        rows = conn.execute(
            f"""SELECT w.work_id, c.claim_id, c.current_revision FROM claims c
                LEFT JOIN work_items w ON w.work_type='embed' AND w.subject_ref=c.claim_id
                     AND w.subject_revision=c.current_revision
                WHERE c.read_blocked=0 AND c.scope_id IN ({_marks(scopes)})
                  AND c.project_id IS ? AND c.branch_id IS ?
                  AND NOT EXISTS (SELECT 1 FROM restored_absence_blocks a
                                  WHERE a.object_kind='claim' AND a.object_ref=c.claim_id)
                  AND NOT EXISTS (SELECT 1 FROM object_blocks b WHERE b.object_kind='claim' AND b.object_ref=c.claim_id
                                  AND (b.read_blocked<>0 OR b.scope_id NOT IN ({_marks(scopes)})))
                  AND (w.work_id IS NULL OR (w.state='obsolete' AND w.last_error_code='authority_revoked'))
                ORDER BY c.claim_id LIMIT ?""",
            (*scopes, context.project_id, context.branch_id, *scopes, limit * 8),
        ).fetchall()
        heads: list[tuple[int | None, str, int]] = []
        for work_id, ref, revision in rows:
            if len(heads) >= limit:
                break
            if _embed_retry_reason(self._tx, ref, revision) is None:
                heads.append((work_id, ref, revision))
        return heads

    # --- a re-embed run after the embedding space changed (``respace-embeddings``) -------------------------------

    def respace_run(self) -> dict | None:
        """The store's re-embed run (``RESPACE_CURSOR``), or None when none was started."""
        row = (
            self._tx._check()
            .execute(
                """SELECT position_ref,position_revision,processed_count,completed,updated_at
                   FROM candidate_scan_cursors
               WHERE cursor_name=?""",
                (RESPACE_CURSOR,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return {
            "embedding_space": row[0],
            "next_work_id": int(row[1] or 0),
            "reopened": int(row[2]),
            "completed": bool(row[3]),
            "updated_at": row[4],
        }

    def embed_queue(self) -> dict:
        """Count pending and failed embedding jobs across the store and report the oldest pending time.

        The state-first index keeps this check proportional to the outstanding queue.
        """
        pending, failed, oldest = (
            self._tx._check()
            .execute(
                """SELECT sum(state='pending'),sum(state='failed'),min(CASE WHEN state='pending' THEN available_at END)
               FROM work_items WHERE state IN ('pending','failed') AND +work_type='embed'"""
            )
            .fetchone()
        )
        return {"pending": int(pending or 0), "failed": int(failed or 0), "oldest_pending_at": oldest}

    def respace_remaining(self, *, at_most: int | None = None) -> int:
        """Count completed embeddings through the optional work ID limit, excluding expired tool vectors."""
        bound = (2**63 - 1) if at_most is None else int(at_most)
        return int(
            self._tx._check()
            .execute(
                f"""SELECT count(*) FROM work_items w WHERE w.work_type='embed' AND w.state='done' AND w.work_id<=?
                AND {_NOT_EXPIRED}""",
                (bound,),
            )
            .fetchone()[0]
        )

    def rebuild_lost_vectors(self, *, now: str) -> int:
        """Requeue prior embeddings after restoring a store without its vectors.

        The restore owns an unopened destination. Previously completed or leased
        embeddings become pending with fresh leases; expired outputs stay expired
        and failed work keeps its retry history. Already pending work also counts
        toward the restore receipt.
        """
        self._tx._check(write=True).execute(
            f"""UPDATE work_items AS w SET state='pending',attempt=0,available_at=?,
                lease_token=lease_token+1,lease_owner=NULL,lease_until=NULL,last_error_code=NULL
                WHERE w.work_type='embed' AND w.state IN ('done','leased') AND {_NOT_EXPIRED}""",
            (now,),
        )
        return self.embed_queue()["pending"]

    def respace_authorization(self, space_id: str) -> str | None:
        """Read the durable identity of an explicitly authorized re-embedding operation for this space."""
        row = (
            self._tx._check()
            .execute(
                "SELECT position_ref FROM candidate_scan_cursors WHERE cursor_name=?",
                ("embedding-authorization:" + space_id,),
            )
            .fetchone()
        )
        return None if row is None else str(row[0])

    def _authorize_respace(self, space_id: str, now: str) -> None:
        """Record authorization independently of queue progress so companion recovery survives interruption."""
        self._tx._check(write=True).execute(
            "INSERT INTO candidate_scan_cursors(cursor_name,position_ref,completed,updated_at) VALUES(?,?,1,?) "
            "ON CONFLICT(cursor_name) DO UPDATE SET position_ref=excluded.position_ref,updated_at=excluded.updated_at",
            ("embedding-authorization:" + space_id, uuid4().hex, now),
        )

    def start_respace(self, space_id: str, *, now: str, restart: bool = False) -> dict:
        """Start a run into ``space_id`` over every embedding done so far, from the newest down (``respace_refusal``
        says when it may).  Work queued after the start is embedded into the new space as it comes and lies above the
        run, so it is never reopened; an embedding still waiting at the start lies below it, is embedded into the new
        space when its turn comes, and is reopened again when the run reaches it."""
        if type(space_id) is not str or not _SPACE_ID.fullmatch(space_id) or type(restart) is not bool:
            raise ContractError("INPUT_INVALID", "respace_space")
        conn = self._tx._check(write=True)
        refusal = respace_refusal(self.respace_run(), space_id, restart=restart)
        if refusal is not None:
            raise ContractError("VERSION_CONFLICT", refusal)
        top = int(conn.execute("SELECT coalesce(max(work_id),0) FROM work_items").fetchone()[0])
        conn.execute(
            """INSERT INTO candidate_scan_cursors(cursor_name,position_ref,position_revision,processed_count,completed,
                   updated_at) VALUES (?,?,?,0,?,?)
               ON CONFLICT(cursor_name) DO UPDATE SET position_ref=excluded.position_ref,
                   position_revision=excluded.position_revision,processed_count=0,completed=excluded.completed,
                   updated_at=excluded.updated_at""",
            (RESPACE_CURSOR, space_id, top, int(top == 0), now),
        )
        self._authorize_respace(space_id, now)
        return self.respace_run()

    def cancel_respace(self) -> bool:
        """Remove the migration cursor and return whether it existed. Already queued jobs continue."""
        return (
            self._tx._check(write=True)
            .execute("DELETE FROM candidate_scan_cursors WHERE cursor_name=?", (RESPACE_CURSOR,))
            .rowcount
            == 1
        )

    def respace_page(self, space_id: str, *, now: str, room: int) -> dict:
        """Queue a bounded page of completed embeddings for the selected space, newest first.

        Scan at most RESPACE_SCAN work IDs per transaction. Reset attempts and fence old leases
        when returning a job to the queue. A worker in another space leaves the cursor alone.
        """
        if type(room) is not int:
            raise ContractError("INPUT_INVALID", "respace_room")
        run = self.respace_run()
        if run is None or run["completed"]:
            return {"outcome": "none"}
        if run["embedding_space"] != space_id:
            return {"outcome": "space_mismatch", "embedding_space": run["embedding_space"]}
        if self.respace_authorization(space_id) is None:
            self._authorize_respace(space_id, now)
        top = run["next_work_id"]
        if room <= 0:
            return {"outcome": "held", "reopened": 0, "next_work_id": top}
        conn = self._tx._check(write=True)
        bottom = max(1, top - RESPACE_SCAN + 1)
        page = min(room, RESPACE_PAGE)
        ids = [
            row[0]
            for row in conn.execute(
                f"""SELECT w.work_id FROM work_items AS w NOT INDEXED
                WHERE w.work_id BETWEEN ? AND ? AND w.work_type='embed' AND w.state='done' AND {_NOT_EXPIRED}
                ORDER BY w.work_id DESC LIMIT ?""",
                (bottom, top, page),
            )
        ]
        if ids:
            conn.execute(
                f"""UPDATE work_items SET state='pending',attempt=0,available_at=?,lease_token=lease_token+1,
                    lease_owner=NULL,lease_until=NULL,last_error_code=?
                    WHERE work_id IN ({_marks(ids)}) AND state='done'""",
                (now, RESPACE_MARKER, *ids),
            )
        # A full page stops below its last row; a short one has looked through the whole window.
        below = (ids[-1] if len(ids) == page else bottom) - 1
        conn.execute(
            """UPDATE candidate_scan_cursors SET position_revision=?,processed_count=processed_count+?,completed=?,
               updated_at=? WHERE cursor_name=?""",
            (below, len(ids), int(below < 1), now, RESPACE_CURSOR),
        )
        return {"outcome": "complete" if below < 1 else "progress", "reopened": len(ids), "next_work_id": below}
