"""Read-only inventory of legacy Hermes provenance candidates; bodies are opt-in."""

from __future__ import annotations

import json

from scope_recall.adapters.hermes.boundary import stored_origin_rule
from scope_recall.contracts import ContractError
from scope_recall.core.management_query import partition
from scope_recall.core.source_origin import dependent_claim_count
from scope_recall.core.source_records import source_origin


def _finding(tx, source, message, rule, snippet):
    count = dependent_claim_count(tx, source.ref, source.revision)
    item = dict(
        ref=source.ref,
        revision=source.revision,
        kind="event",
        entry=source.entry_id,
        time=source.event["occurred_at"] or source.event["recorded_at"],
        rule=rule,
        affected_claims=count,
        platform_id_available=any(message.get(k) for k in ("platform_message_id", "message_id")),
    )
    if snippet:
        item["snippet"] = source.event["content"][:120]
    return item


def _candidate(tx, context, row, entries, snippet):
    view = partition(tx, context, row["project_id"], row["branch_id"])
    source = view.source(row["event_id"], row["source_revision"])
    if source is None or source_origin(source) != "human_direct":
        return None
    key = source.event["source_event_key"]
    imported = key.startswith(("import:", "memory-import:")) or source.event["origin"] == "imported"
    if not (imported or key.startswith("hermes:") or entries.get(source.entry_id) == "hermes"):
        return None
    # Only preserved host metadata can attest a platform delivery. The admitted source always owns the body.
    extra = json.loads(row["extra_json"])
    message = dict(extra, role="user", content=source.event["content"])
    rule = stored_origin_rule(message, imported=imported)
    return None if rule is None else _finding(tx, source, message, rule, snippet)


def candidates(api, *, limit=100, after=0, snippet=False):
    """Scan a bounded source-ID page without writes; missing historical IDs require owner review."""
    if type(limit) is not int or not 1 <= limit <= 1000 or type(after) is not int or after < 0:
        raise ContractError("INPUT_INVALID", "limit_or_after")
    context = api._context()
    with api.core.storage.read(context) as tx:
        entries = {r["entry_id"]: r["host"] for r in tx._check().execute("SELECT entry_id,host FROM entries")}
        rows = (
            tx._check()
            .execute(
                "SELECT event_id,source_revision,source_id,scope_id,project_id,branch_id,extra_json "
                "FROM source_events WHERE read_blocked=0 AND role='user' AND source_id>? "
                "AND (origin='human_direct' OR (origin='imported' AND source_original_origin='human_direct' "
                "AND import_provenance_sha256 IS NOT NULL)) "
                "AND scope_id IN (SELECT value FROM json_each(?)) ORDER BY source_id LIMIT ?",
                (after, json.dumps(sorted(context.allowed_scope_ids)), limit + 1),
            )
            .fetchall()
        )
        items = [item for row in rows[:limit] if (item := _candidate(tx, context, row, entries, snippet)) is not None]
    return dict(
        items=items,
        examined=min(len(rows), limit),
        next_after=rows[limit - 1]["source_id"] if len(rows) > limit else None,
    )
