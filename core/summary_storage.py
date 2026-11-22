"""Bounded summary discovery, durable coverage and scoped raw-message substitution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

from ..contracts import ContractError
from . import private
from .claims import canonical_time
from .failure_policy import failure_kind
from .handoff import close_drafts
from .language_resources import language_rules
from .recall_scope import QueryScope
from .summary_parts import queue_summary_part, reuse_completed_parts
from .summary_validity import (
    group_members,
    group_signature,
    members_in_scope,
    open_task,
    summary_live,
    summary_source_failure,
)

IDLE_SECONDS = 1800
# Each group has few pages. Keep the page first so SQLite uses the unique work identity
# instead of scanning every failed work item in the store for each group.
_PAGE_WORK = """CROSS JOIN work_items w ON w.subject_ref=p.claim_ref AND w.subject_revision=p.claim_revision
    AND w.work_type='rebuild_projection' AND w.scope_id=g.scope_id
    AND w.project_id IS g.project_id AND w.branch_id IS g.branch_id"""
_NO_RESUMABLE_PAGE = f"""NOT EXISTS(SELECT 1 FROM summary_parts p {_PAGE_WORK}
    WHERE p.group_key=g.group_key AND p.generation=g.generation AND p.coverage_json='[]'
    AND w.state IN ('pending','leased','failed'))"""
_RECOVERABLE_GROUP = f"""(g.state='pending' OR (g.state='partial' AND g.failure IN (
    'source_coverage','source_unavailable','summary_source','summary_sources','source_revision')))
    AND {_NO_RESUMABLE_PAGE}"""
# Only the current empty page can make a legacy group eligible. Old generations
# may be retired during replacement if their own last error proves the cause.
_ERROR_DETAIL = """LEFT JOIN work_error_details d ON d.detail_id=(
    SELECT detail_id FROM work_error_details WHERE work_id=w.work_id AND lease_token=w.lease_token
    AND stage='process' ORDER BY detail_id DESC LIMIT 1)"""
_ERROR_FIELD = "COALESCE(d.error_field, CASE WHEN p.generation=g.generation THEN g.failure END)"
_LEGACY_SOURCE_FAILURE = f"""w.state='failed' AND p.coverage_json='[]' AND (
    ((lower(w.last_error_code)='source_missing' OR lower(w.last_error_code) LIKE '%|source_missing')
        AND {_ERROR_FIELD} IN ('summary_source','source_unavailable','source_coverage'))
    OR ((lower(w.last_error_code)='version_conflict' OR lower(w.last_error_code) LIKE '%|version_conflict')
        AND {_ERROR_FIELD} IN ('summary_sources','source_revision')))"""
_LEGACY_ERROR = f"""(SELECT w.last_error_code FROM summary_parts p {_PAGE_WORK} {_ERROR_DETAIL}
    WHERE p.group_key=g.group_key AND p.generation=g.generation AND {_LEGACY_SOURCE_FAILURE}
    LIMIT 1)"""


def previous_conversation_scope(tx, query, *, now):
    """Resolve a previous-conversation question to one visible source session, preserving raw fallback."""
    if not language_rules("query_time")["previous_conversation"].fullmatch(query.strip()):
        return None
    ctx = tx.context
    rows = (
        tx._check()
        .execute(
            "SELECT event_id,source_revision,session_id,entry_id FROM source_events WHERE scope_id IN ("
            + ",".join("?" for _ in ctx.allowed_scope_ids)
            + ") AND project_id IS ? AND branch_id IS ? AND session_id!=? AND occurred_at<=? "
            "AND role='user' AND read_blocked=0 AND suppressed=0 ORDER BY occurred_at DESC,rowid DESC LIMIT 64",
            (*sorted(ctx.allowed_scope_ids), ctx.project_id, ctx.branch_id, ctx.session_id, now),
        )
        .fetchall()
    )
    for row in rows:
        source = tx.source(row[0], row[1])
        if source is None or source.suppressed:
            continue
        stamps = (
            tx._check()
            .execute(
                "SELECT min(occurred_at),max(occurred_at) FROM source_events WHERE session_id=? "
                "AND scope_id=? AND entry_id=? AND project_id IS ? AND branch_id IS ?",
                (row[2], source.scope_id, row[3], ctx.project_id, ctx.branch_id),
            )
            .fetchone()
        )
        if stamps[0] and stamps[1]:
            end = (datetime.fromisoformat(stamps[1]) + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
            return QueryScope(((stamps[0], end),), (row[3],), query, (row[2],))
    return None


def note_source(tx, source, *, now, historical=False):
    """Register conversation and task partitions without copying text or requesting a model."""
    if source is None or source.event["role"] not in {"user", "assistant", "tool"}:
        return
    groups = [("session", source.session_id)] if source.event["role"] != "tool" else []
    # Capture already attached this source; only the task identity and terminal state are needed here.
    episode = (
        tx._check()
        .execute(
            "SELECT e.episode_id,e.anchor_kind,v.state FROM episode_events ee "
            "JOIN episodes e ON e.episode_id=ee.episode_id "
            "JOIN episode_versions v ON v.episode_id=e.episode_id AND v.revision=e.current_revision "
            "WHERE ee.source_ref=? AND ee.source_revision=?",
            (source.ref, source.revision),
        )
        .fetchone()
    )
    if episode is not None and episode["anchor_kind"] == "task":
        groups.append(("task", episode["episode_id"]))
    if source.event["role"] == "tool" and episode is not None and episode["state"] == "completed":
        tx._check(write=True).execute(
            "UPDATE summary_groups SET state='waiting' WHERE kind='session' AND subject_ref=? "
            "AND scope_id=? AND project_id IS ? AND branch_id IS ? AND state='open_task'",
            (source.session_id, source.scope_id, source.project_id, source.branch_id),
        )
    for kind, subject in groups:
        values = [kind, subject, source.scope_id, source.project_id, source.branch_id, source.entry_id]
        key = hashlib.sha256(json.dumps(values).encode()).hexdigest()
        tx._check(write=True).execute(
            "INSERT INTO "
            "summary_groups(group_key,kind,subject_ref,scope_id,project_id,branch_id,entry_id,last_activity) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(group_key) DO UPDATE SET "
            "last_activity=max(last_activity,excluded.last_activity),closed=CASE WHEN ? THEN closed ELSE 0 END,"
            "state=CASE WHEN ? THEN state ELSE 'waiting' END",
            (key, *values, now, int(historical), int(historical)),
        )


def close_session(tx, session_id, *, now):
    """Record an explicit end signal for this bound session; new capture reopens it."""
    if session_id != tx.context.session_id:
        raise ContractError("ACCESS_DENIED", "session")
    close_drafts(tx, session_id, now)
    tx._check(write=True).execute(
        "UPDATE summary_groups SET closed=1 WHERE kind='session' AND subject_ref=? AND scope_id IN ("
        + ",".join("?" for _ in tx.context.allowed_scope_ids)
        + ") AND project_id IS ? AND branch_id IS ?",
        (session_id, *sorted(tx.context.allowed_scope_ids), tx.context.project_id, tx.context.branch_id),
    )


def _discover(tx, limit):
    ctx = tx.context
    audience = hashlib.sha256(
        json.dumps([sorted(ctx.allowed_scope_ids), ctx.project_id, ctx.branch_id]).encode()
    ).hexdigest()
    conn = tx._check(write=True)
    conn.execute("INSERT OR IGNORE INTO summary_backfill(audience) VALUES(?)", (audience,))
    position = conn.execute("SELECT position FROM summary_backfill WHERE audience=?", (audience,)).fetchone()[0]
    rows = conn.execute(
        "SELECT rowid,event_id,source_revision FROM source_events WHERE rowid>? AND scope_id IN ("
        + ",".join("?" for _ in ctx.allowed_scope_ids)
        + ") AND project_id IS ? AND branch_id IS ? ORDER BY rowid LIMIT ?",
        (position, *sorted(ctx.allowed_scope_ids), ctx.project_id, ctx.branch_id, limit),
    ).fetchall()
    for row in rows:
        source = tx.source(row[1], row[2])
        if source is not None:
            # Historical sources keep their recorded activity; importing them does not make an old conversation active.
            note_source(tx, source, now=source.event["recorded_at"], historical=True)
    if rows:
        conn.execute("UPDATE summary_backfill SET position=? WHERE audience=?", (rows[-1][0], audience))


def summary_wait_state(tx, group, now):
    """Return the state a group must wait in, or None when it has ended."""
    if group["kind"] == "task":
        episode = tx.episodes.get(group["subject_ref"])
        return None if episode is not None and episode.state == "completed" and not episode.suppressed else "open_task"
    if open_task(tx, group):
        return "open_task"
    if group["closed"]:
        return None
    last, current = canonical_time(group["last_activity"]), canonical_time(now)
    if last is None or current is None:
        return "waiting"
    idle = datetime.fromisoformat(current) - datetime.fromisoformat(last)
    return None if idle >= timedelta(seconds=IDLE_SECONDS) else "waiting"


def set_summary_state(tx, group, state, failure=None):
    """Update the current generation's scheduling state without discarding its coverage."""
    tx._check(write=True).execute(
        "UPDATE summary_groups SET state=?,failure=? WHERE group_key=? AND generation=?",
        (state, failure, group["group_key"], group["generation"]),
    )


def _retire_failed_source_pages(tx, group, *, replacing=False):
    """Settle only identified failed pages in this group, leaving live leases to their owners."""
    generation = "p.generation<=g.generation" if replacing else "p.generation=g.generation"
    tx._check(write=True).execute(
        "UPDATE work_items SET state='obsolete',last_error_code='authority_revoked',lease_owner=NULL,lease_until=NULL "
        "WHERE state='failed' AND work_id IN ("
        f"SELECT w.work_id FROM summary_groups g JOIN summary_parts p ON p.group_key=g.group_key {_PAGE_WORK} "
        f"{_ERROR_DETAIL} WHERE g.group_key=? AND g.generation=? AND {generation} AND {_LEGACY_SOURCE_FAILURE})",
        (group["group_key"], group["generation"]),
    )


def schedule_summary_generation(tx, previous, members, signature, *, now):
    """Start one generation for validated inputs and reuse the previous generation's safe coverage."""
    group = dict(
        previous,
        generation=previous["generation"] + 1,
        manifest_json=json.dumps(members),
        state="pending",
        total_units=sum(m["total"] for m in members),
        covered_units=0,
        next_offset=0,
        signature=signature,
    )
    conn = tx._check(write=True)
    _retire_failed_source_pages(tx, previous, replacing=True)
    conn.execute(
        "UPDATE summary_groups SET generation=?,manifest_json=?,state='pending',total_units=?,covered_units=0,"
        "next_offset=0,signature=?,failure=NULL WHERE group_key=?",
        (group["generation"], group["manifest_json"], group["total_units"], signature, group["group_key"]),
    )
    try:
        group = reuse_completed_parts(tx, previous, group, now=now)
        if group["state"] != "complete":
            ordinal = conn.execute(
                "SELECT COALESCE(max(ordinal),-1)+1 FROM summary_parts WHERE group_key=? AND generation=?",
                (group["group_key"], group["generation"]),
            ).fetchone()[0]
            queue_summary_part(tx, group, ordinal, now)
    except ContractError as exc:
        if exc.code not in {"SOURCE_MISSING", "ACCESS_DENIED", "VERSION_CONFLICT"}:
            raise
        set_summary_state(tx, group, "partial", "summary_part_unavailable")
        return False
    return True


def _scheduling_page(tx, position, *, include_complete):
    """Return one audience-scoped page of groups that can make scheduling progress."""
    ctx = tx.context
    normal_states = "('waiting','complete')" if include_complete else "('waiting')"
    return (
        tx._check()
        .execute(
            f"SELECT g.rowid AS position,{_NO_RESUMABLE_PAGE} AS no_work,{_LEGACY_ERROR} AS legacy_error,g.* "
            "FROM summary_groups g WHERE g.rowid>? AND g.scope_id IN ("
            + ",".join("?" for _ in ctx.allowed_scope_ids)
            + f") AND g.project_id IS ? AND g.branch_id IS ? AND (g.state IN {normal_states} OR ("
            + _RECOVERABLE_GROUP
            + ") OR (g.state IN ('pending','partial','open_task') AND legacy_error IS NOT NULL)) "
            "ORDER BY g.rowid LIMIT 64",
            (position, *sorted(ctx.allowed_scope_ids), ctx.project_id, ctx.branch_id),
        )
        .fetchall()
    )


def _legacy_changed(group, signature, failure):
    """Find proven authority changes without consuming the legacy page's retry budget."""
    return group["legacy_error"] is not None and (
        group["state"] == "open_task"
        or signature != group["signature"]
        or (failure is None and failure_kind(group["legacy_error"]) == "source_missing")
    )


def queue_summaries(tx, *, now, limit=128, enabled=True):
    """Discover one old-source page and queue at most eight ended groups for the budgeted worker."""
    if not enabled:
        return 0
    _discover(tx, limit)
    ctx = tx.context
    audience = (
        "groups:"
        + hashlib.sha256(
            json.dumps([sorted(ctx.allowed_scope_ids), ctx.project_id, ctx.branch_id]).encode()
        ).hexdigest()
    )
    conn = tx._check(write=True)
    conn.execute("INSERT OR IGNORE INTO summary_backfill(audience) VALUES(?)", (audience,))
    position = conn.execute("SELECT position FROM summary_backfill WHERE audience=?", (audience,)).fetchone()[0]
    rows = _scheduling_page(tx, position, include_complete=True)
    if position and not rows:
        rows = _scheduling_page(tx, 0, include_complete=True)
    queued, position = 0, 0
    for row in rows:
        if queued == 8:
            break
        position = row["position"]
        queued += _queue_group(tx, row, now)
    conn.execute("UPDATE summary_backfill SET position=? WHERE audience=?", (position, audience))
    return queued


def _queue_group(tx, group, now):
    """Reconcile one group, then defer it or prepare one new generation."""
    wait_state = summary_wait_state(tx, group, now)
    if wait_state == "open_task":
        if group["legacy_error"] is not None:
            _retire_failed_source_pages(tx, group)
        set_summary_state(tx, group, wait_state)
        return False
    if wait_state is not None and group["legacy_error"] is None:
        return False
    members = group_members(tx, group)
    signature = group_signature(tx, members)
    if group["state"] == "complete" and group["signature"] == signature:
        return False
    failure = summary_source_failure(tx, members)
    changed = _legacy_changed(group, signature, failure)
    if group["legacy_error"] is not None and not changed:
        return False
    if wait_state is not None or failure is not None:
        if changed:
            _retire_failed_source_pages(tx, group)
        state = wait_state or "partial"
        if (group["state"], group["failure"]) != (state, failure):
            set_summary_state(tx, group, state, failure)
    elif group["signature"] == signature and 0 < group["covered_units"] == group["total_units"]:
        if changed:
            _retire_failed_source_pages(tx, group)
        set_summary_state(tx, group, "complete")
    elif changed or group["signature"] != signature or group["no_work"]:
        return schedule_summary_generation(tx, group, members, signature, now=now)
    return False


def summarized_rows(tx, context, rows, *, limit):
    """Offer valid summaries first; remove only raw messages literally covered in full."""
    if context.mode in {"history", "as_of"}:
        return rows
    raw = {(r["event_id"], r["source_revision"]) for r in rows}
    if not raw:
        return rows
    raw_json = json.dumps(sorted(raw))
    groups = (
        tx._check()
        .execute(
            "SELECT g.* FROM summary_groups g WHERE g.state='complete' AND g.scope_id IN ("
            + ",".join("?" for _ in tx.context.allowed_scope_ids)
            + ") AND g.project_id IS ? AND g.branch_id IS ? AND EXISTS("
            "SELECT 1 FROM json_each(?) r JOIN source_events s ON s.event_id=json_extract(r.value,'$[0]') "
            "AND s.source_revision=json_extract(r.value,'$[1]') WHERE s.scope_id=g.scope_id "
            "AND s.entry_id=g.entry_id AND ((g.kind='session' AND g.subject_ref=s.session_id) OR "
            "(g.kind='task' AND EXISTS(SELECT 1 FROM episode_events ee WHERE ee.episode_id=g.subject_ref "
            "AND ee.source_ref=s.event_id AND ee.source_revision=s.source_revision)))) "
            "ORDER BY g.last_activity DESC LIMIT 64",
            (*sorted(tx.context.allowed_scope_ids), tx.context.project_id, tx.context.branch_id, raw_json),
        )
        .fetchall()
    )
    replacements, covered = [], set()
    for group in groups:
        manifest = json.loads(group["manifest_json"])
        members = {(m["ref"], m["revision"]) for m in manifest if m.get("role") in {"user", "assistant"}}
        if (
            not members
            or not members <= raw
            or members & covered
            or not members_in_scope(manifest, context.scope)
            or any(f"{m['ref']}@{m['revision']}" in context.current_source_refs for m in manifest)
        ):
            continue
        parts = (
            tx._check()
            .execute(
                "SELECT * FROM summary_parts WHERE group_key=? AND generation=? ORDER BY ordinal",
                (group["group_key"], group["generation"]),
            )
            .fetchall()
        )
        versions = [(p, tx.claims.version(p["claim_ref"], p["claim_revision"])) for p in parts]
        active = [(p, v) for p, v in versions if v is not None and v.state == "active"]
        if (
            not active
            or len(active) + len(replacements) > limit
            or not all(summary_live(tx, p["claim_ref"], p["claim_revision"]) for p, _v in active)
        ):
            continue
        if all(not version.suppressed for _p, version in active):
            replacements.extend(
                dict(
                    event_id=p["claim_ref"],
                    source_revision=p["claim_revision"],
                    role="summary",
                    size=0,
                    short=None,
                    candidate_kind="claim",
                )
                for p, _v in active
            )
            covered.update(fully_quoted_sources(tx, [version for _p, version in active], members))
    return [*replacements, *(r for r in rows if (r["event_id"], r["source_revision"]) not in covered)]


def fully_quoted_sources(tx, versions, members):
    """Identify raw entries whose every character is present in positioned exact excerpts."""
    spans = {}
    for version in versions:
        for span in version.payload["evidence_spans"]:
            if "start" in span and "end" in span:
                spans.setdefault((span["source_ref"], span["source_revision"]), []).append(span)
    covered = set()
    for key in members:
        source = tx.source(*key)
        if source is None:
            continue
        content, position = source.event["content"], 0
        for span in sorted(spans.get(key, ()), key=lambda span: span["start"]):
            if span["start"] > position or content[span["start"] : span["end"]] != span["quote"]:
                break
            position = max(position, span["end"])
        if position == len(content):
            covered.add(key)
    return covered


def next_summary_due(tx, *, now):
    """Advise the existing supervisor about old-source discovery and idle closure; grant no write authority."""
    if private.source_blocked(private.active_intervals(tx, now), None):
        return None
    ctx = tx.context
    audience = hashlib.sha256(
        json.dumps([sorted(ctx.allowed_scope_ids), ctx.project_id, ctx.branch_id]).encode()
    ).hexdigest()
    cursor = tx._check().execute("SELECT position FROM summary_backfill WHERE audience=?", (audience,)).fetchone()
    position = cursor[0] if cursor else 0
    scopes = ",".join("?" for _ in ctx.allowed_scope_ids)
    params = (*sorted(ctx.allowed_scope_ids), ctx.project_id, ctx.branch_id)
    if (
        tx._check()
        .execute(
            "SELECT 1 FROM source_events WHERE rowid>? AND scope_id IN ("
            + scopes
            + ") AND project_id IS ? AND branch_id IS ? LIMIT 1",
            (position, *params),
        )
        .fetchone()
    ):
        return now
    # Stable gaps cannot hide a later actionable group or schedule another
    # drain by themselves. Read the partition in bounded pages and stop on
    # the first group that could advance now.
    position, due = 0, None
    while rows := _scheduling_page(tx, position, include_complete=False):
        for group in rows:
            group_due = _group_due(tx, group, now)
            if group_due == now:
                return now
            if group_due is not None:
                due = group_due if due is None else min(due, group_due)
        position = rows[-1]["position"]
    return due


def _group_due(tx, group, now):
    wait_state = summary_wait_state(tx, group, now)
    if group["legacy_error"] is not None:
        if wait_state == "open_task":
            return now
        members = group_members(tx, group)
        signature = group_signature(tx, members)
        failure = summary_source_failure(tx, members)
        return now if _legacy_changed(group, signature, failure) else None
    if wait_state is None:
        if group["state"] == "waiting" or summary_source_failure(tx, group_members(tx, group)) is None:
            return now
    elif wait_state == "waiting":
        stamp = canonical_time(group["last_activity"])
        if stamp:
            return (datetime.fromisoformat(stamp) + timedelta(seconds=IDLE_SECONDS)).isoformat().replace("+00:00", "Z")
    return None
