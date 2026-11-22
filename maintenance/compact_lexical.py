"""Measure and rebuild old tool-output postings without changing source evidence."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from types import SimpleNamespace

from ..core import lexical_index
from ..core.backup_primitives import BackupError
from ..core.delete_storage import group_digest
from ..core.events import tool_lexical_projection
from ..core.injection_guard import held_roots_sql
from ..core.secret_patterns import SECRET_RULES_VERSION

_PAGE_SIZE = 512
_POSTING_BYTES = 16  # Two logical 64-bit identifiers, counted once per posting.
_TERM_ID_BYTES = 8

# Walk primary storage in order and only inspect indexed tool outputs. The body
# receipt identifies text already scanned under the current credential rules.
_PAGE = """
    SELECT e.rowid,e.source_id,e.event_id,e.origin,e.source_original_origin,e.role,e.content,
           e.scope_id,e.project_id,e.branch_id,e.source_group_key,
           (e.secret_scan_version=? AND e.secret_scan_names='[]' AND e.secret_scan_input_sha256=e.content_sha256)
    FROM source_events e
    WHERE e.rowid>? AND e.read_blocked=0 AND e.suppressed=0
      AND (e.origin='tool_observation' OR (e.origin='imported' AND e.source_original_origin='tool_observation'
          AND e.import_provenance_sha256<>''))
      AND EXISTS (SELECT 1 FROM lexical_postings p WHERE p.source_id=e.source_id)
      AND NOT EXISTS (SELECT 1 FROM object_blocks b WHERE b.object_kind='event' AND b.object_ref=e.event_id
                      AND (b.read_blocked=1 OR b.suppressed=1 OR b.scope_id<>e.scope_id))
      AND NOT EXISTS (SELECT 1 FROM restored_absence_blocks a WHERE a.object_kind='event' AND a.object_ref=e.event_id)
    ORDER BY e.rowid LIMIT ?
"""

_NO_FACT_POSTINGS = """
    NOT EXISTS (SELECT 1 FROM claim_lexical_postings p WHERE p.term_id=t.term_id)
    AND NOT EXISTS (SELECT 1 FROM claim_version_postings p WHERE p.term_id=t.term_id)
"""
_ORPHAN_TERMS = f"""
    SELECT t.term_id,t.term FROM lexical_terms t
    WHERE NOT EXISTS (SELECT 1 FROM lexical_postings p WHERE p.term_id=t.term_id) AND {_NO_FACT_POSTINGS}
"""


def _blocked_groups(conn: sqlite3.Connection, binding, page) -> set[str]:
    """Check the existing group tombstone identity even if only the ledger has it yet."""
    group_by_event = {row[2]: group_digest(binding, *row[7:11]) for row in page}
    blocked = {
        row[0]
        for row in conn.execute(
            "SELECT b.group_sha256 FROM json_each(?) j CROSS JOIN source_group_blocks b ON b.group_sha256=j.value "
            "WHERE b.read_blocked=1 OR b.suppressed=1",
            (json.dumps(list(set(group_by_event.values()))),),
        )
    }
    return {event_id for event_id, group in group_by_event.items() if group in blocked}


def _tool_source_pages(conn: sqlite3.Connection) -> Iterator[dict[int, tuple[tuple[str, ...], bool]]]:
    """Yield eligible versions and their projections, outside group and lineage holds."""
    binding = None
    if conn.execute("SELECT 1 FROM source_group_blocks WHERE read_blocked=1 OR suppressed=1 LIMIT 1").fetchone():
        installation = conn.execute("SELECT installation_id FROM instance_meta WHERE singleton=1").fetchone()[0]
        binding = SimpleNamespace(installation_id=installation)
    holds = "SELECT 1 FROM source_safety WHERE state IN ('pending_review','quarantined') LIMIT 1"
    has_holds = conn.execute(holds).fetchone() is not None
    after = 0
    while page := conn.execute(_PAGE, (SECRET_RULES_VERSION, after, _PAGE_SIZE)).fetchall():
        after = page[-1][0]
        blocked = _blocked_groups(conn, binding, page) if binding is not None else set()
        if has_holds:
            refs = list({row[2] for row in page} - blocked)
            blocked.update(
                row[0]
                for row in conn.execute(
                    held_roots_sql("SELECT value,'event',value FROM json_each(?)"), (json.dumps(refs),)
                )
            )
        projected = {}
        for _, source, ref, origin, original, role, content, *_, secret_free in page:
            if ref in blocked:
                continue
            event = dict(origin=origin, source_original_origin=original, role=role, content=content)
            projection = tool_lexical_projection(event, secret_free=secret_free, skip_clean_short=True)
            if projection is not None:
                projected[source] = projection
        if projected:
            yield projected


def _refresh_postings(conn: sqlite3.Connection, *, apply: bool) -> tuple[dict[str, int], Counter[str]]:
    """Compare postings in source order and optionally update a disposable copy."""
    stats = dict(
        sources_current=0, sources_rebuilt=0, postings_before=0, postings_after=0, postings_removed=0, postings_added=0
    )
    term_delta: Counter[str] = Counter()
    for page in _tool_source_pages(conn):
        existing_by_source = lexical_index.terms_for_sources(conn, page)
        for source, (keep, may_add) in page.items():
            existing = existing_by_source[source]
            if not may_add:
                # A historically unfiltered body can only surrender old postings.
                allowed = set(keep)
                keep = tuple(term for term in existing if term in allowed)
            stats["postings_before"] += len(existing)
            stats["postings_after"] += len(keep)
            if existing == keep:
                stats["sources_current"] += 1
                continue
            removed = set(existing).difference(keep)
            added = set(keep).difference(existing)
            stats["sources_rebuilt"] += 1
            stats["postings_removed"] += len(removed)
            stats["postings_added"] += len(added)
            if apply:
                lexical_index.index_terms(conn, source, sorted(added))
                if lexical_index.unindex_beyond(conn, source, keep) != len(removed):
                    raise BackupError("tool posting selection changed on the staging copy")
            else:
                term_delta.subtract(removed)
                term_delta.update(added)
    return stats, term_delta


def _measure_terms(conn: sqlite3.Connection, delta: Counter[str]) -> tuple[int, int, int]:
    """Count lost and new words once across the shared dictionary and all posting tables."""
    removed = added = net_bytes = 0
    for _, term in conn.execute(_ORPHAN_TERMS):
        if delta[term] <= 0:
            removed += 1
            net_bytes += _TERM_ID_BYTES + len(term.encode("utf-8"))

    # Ask about the original usage once per changed term. A new posting on
    # another source balances a removal; current and historical facts also
    # keep their shared words. Each count reads only that term's PK range.
    changes = ((term, net) for term, net in delta.items() if net)
    while batch := dict(islice(changes, _PAGE_SIZE)):
        rows = conn.execute(
            f"""SELECT j.key,t.term_id,CASE WHEN j.value<0 AND {_NO_FACT_POSTINGS} THEN
                  (SELECT count(*) FROM lexical_postings p WHERE p.term_id=t.term_id)=-j.value END
                FROM json_each(?) j LEFT JOIN lexical_terms t ON t.term=j.key""",
            (json.dumps(batch, ensure_ascii=False),),
        )
        for term, term_id, orphaned in rows:
            if term_id is None:
                added += 1
                net_bytes -= _TERM_ID_BYTES + len(term.encode("utf-8"))
            elif orphaned:
                removed += 1
                net_bytes += _TERM_ID_BYTES + len(term.encode("utf-8"))
    return removed, added, net_bytes


def measure_tool_lexical(conn: sqlite3.Connection) -> dict[str, int]:
    """Return projection changes and net logical bytes from the caller's read snapshot."""
    stats, delta = _refresh_postings(conn, apply=False)
    removed, added, net_bytes = _measure_terms(conn, delta)
    return dict(
        stats,
        terms_removed=removed,
        terms_added=added,
        estimated_bytes=net_bytes + _POSTING_BYTES * (stats["postings_removed"] - stats["postings_added"]),
    )


def compact_tool_lexical(conn: sqlite3.Connection, measured: dict[str, int]) -> None:
    """Repair only the staging copy and verify net posting and dictionary counts."""
    term_count = "SELECT count(*) FROM lexical_terms"
    before = conn.execute(term_count).fetchone()[0]
    stats, _ = _refresh_postings(conn, apply=True)
    after = 0
    query = _ORPHAN_TERMS + " AND t.term_id>? ORDER BY t.term_id LIMIT ?"
    while page := conn.execute(query, (after, _PAGE_SIZE)).fetchall():
        after = page[-1][0]
        lexical_index.prune_terms(conn, (term_id for term_id, _ in page))
    if (
        any(measured[key] != value for key, value in stats.items())
        or before - conn.execute(term_count).fetchone()[0] != measured["terms_removed"] - measured["terms_added"]
    ):
        raise BackupError("tool index selection changed on the staging copy")
