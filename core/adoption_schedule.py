"""Reconcile source-bound adoption checkpoints after consolidation or late capture."""

import json

from .adoption_order import adjacent_occurrence, first_occurrence
from .adoption_storage import adoption_context, invalidate_adoptions, source_reason
from .entry_control import entry_fence


def schedule_adoptions(tx, source_refs, *, now):
    """Start adoption only for completed extraction with no relationship or checkpoint.

    Reuse the ordinary work row, preserving its ownership and deletion fences.
    A relationship is final for automatic scheduling of its own confirmation version, regardless of its state;
    an edited confirmation is a new version and is scheduled as before. Completed checkpoints also remain final,
    so replay never repeats paid stages.
    """
    conn = tx._check(write=True)
    for key in source_refs:
        ref, revision = key.rsplit("@", 1)
        work = conn.execute(
            "SELECT w.work_id FROM work_items w WHERE w.work_type='consolidate' "
            "AND w.subject_ref=? AND w.subject_revision=? AND w.state='done' "
            "AND NOT EXISTS(SELECT 1 FROM adoption_relations r "
            "WHERE r.confirmation_ref=w.subject_ref AND r.confirmation_revision=w.subject_revision) "
            "AND NOT EXISTS(SELECT 1 FROM adoption_processing p WHERE p.work_id=w.work_id)",
            (ref, int(revision)),
        ).fetchone()
        if work is None:
            continue
        owner = tx.source(ref, int(revision))
        snapshot = adoption_context(tx, owner) if owner is not None else None
        if snapshot is None or source_reason(tx, snapshot) is not None:
            continue
        payload = json.dumps(dict(stage="locate", entry_fence=entry_fence(tx, (owner, snapshot.assistant))))
        conn.execute(
            "INSERT INTO adoption_processing VALUES (?,?,?,?,?,?)",
            (work[0], ref, int(revision), snapshot.assistant_ref, snapshot.assistant_revision, payload),
        )
        conn.execute(
            "UPDATE work_items SET state='pending',attempt=0,available_at=?,last_error_code=NULL WHERE work_id=?",
            (now, work[0]),
        )


def reconcile_after_capture(tx, ref, revision, *, now):
    """Seek only the immediate successor; in-order capture never computes adoption contexts."""
    conn = tx._check(write=True)
    invalidate_adoptions(conn, ref, revision)
    source = tx.source(ref, revision)
    if source is None:
        return
    first = first_occurrence(conn, ref)
    successor = adjacent_occurrence(conn, source, first, following=True)
    if successor is None or (successor["origin"], successor["role"]) != ("human_direct", "user"):
        return
    latest = conn.execute(
        "SELECT max(source_revision) FROM source_events WHERE event_id=?", (successor["event_id"],)
    ).fetchone()[0]
    schedule_adoptions(tx, (f"{successor['event_id']}@{latest}",), now=now)
