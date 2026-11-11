"""Owner batch decisions over exact pending source versions, using native review writers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import replace
from datetime import datetime

from ..contracts import ContractError
from .injection_guard import notice
from .quarantine_management import finish_review, release_source
from .restore import maintenance_context
from .source_records import source_origin, stored_source
from .visibility import deletion_allows
from .writer_lease import TruthWriterBusyError

FORMAT = "scope-recall-source-review/1"
MAX_ITEMS = 1000
EXCERPT_CHARS = 240


def _time(value):
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed
    except (AttributeError, TypeError, ValueError):
        pass
    raise ContractError("INPUT_INVALID", "review_time")


def _filters(state, rules, origins, after, before, limit):
    if state != "pending_review" or type(limit) is not int or not 1 <= limit <= MAX_ITEMS:
        raise ContractError("INPUT_INVALID", "pending_review_selection")
    rules, origins = tuple(rules or ()), tuple(origins or ())
    if any(type(value) is not str or not 1 <= len(value) <= 240 for value in (*rules, *origins)):
        raise ContractError("INPUT_INVALID", "review_filters")
    lower, upper = _time(after), _time(before)
    if lower is not None and upper is not None and lower >= upper:
        raise ContractError("INPUT_INVALID", "review_time")
    return rules, origins, lower, upper


def preview_sources(
    core, authority, *, rules=(), origins=(), after=None, before=None, limit=100, decision="release", **selection
):
    """Preview at most limit authorized pending versions; time filters use scanned_at.

    Rules and origins each match any supplied value. Time bounds are exclusive.
    The returned database identity and hashes bind confirmation to these bytes.
    """
    context = maintenance_context(core.storage, authority)
    state = selection.pop("state", "pending_review")
    if selection:
        raise ContractError("INPUT_INVALID", "review_filters")
    rules, origins, lower, upper = _filters(state, rules, origins, after, before, limit)
    if decision not in {"release", "retain"}:
        raise ContractError("INPUT_INVALID", "review_decision")
    items = []
    with core.storage.read(context) as tx:
        rows = tx._check().execute(
            "SELECT e.*,s.rules_json,s.scanned_at FROM source_safety s JOIN source_events e "
            "ON e.event_id=s.source_ref AND e.source_revision=s.source_revision "
            "WHERE s.state='pending_review' AND e.read_blocked=0 "
            "AND (?='release' OR NOT EXISTS(SELECT 1 FROM source_events n "
            "WHERE n.event_id=e.event_id AND n.source_revision>e.source_revision)) "
            "AND e.scope_id IN (SELECT value FROM json_each(?)) ORDER BY julianday(s.scanned_at),e.source_id",
            (decision, json.dumps(sorted(context.allowed_scope_ids))),
        )
        for row in rows:
            matched = set(json.loads(row["rules_json"]))
            origin = source_origin(stored_source(row, None))
            scanned = _time(row["scanned_at"])
            if scanned is None:
                raise ContractError("INPUT_INVALID", "review_time")
            if (rules and not matched.intersection(rules)) or (origins and origin not in origins):
                continue
            if (lower is not None and scanned <= lower) or (upper is not None and scanned >= upper):
                continue
            view = tx.with_context(replace(context, project_id=row["project_id"], branch_id=row["branch_id"]))
            if notice(view, row["event_id"], row["source_revision"]) is None:
                continue
            items.append(
                dict(
                    ref=row["event_id"],
                    revision=row["source_revision"],
                    origin=origin,
                    rules=sorted(matched),
                    scanned_at=row["scanned_at"],
                    excerpt=row["content"][:EXCERPT_CHARS],
                    content_sha256=hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
                )
            )
            if len(items) > limit:
                break
        identity = tx.database_identity()[0]
    more = len(items) > limit
    items = items[:limit]
    return dict(
        format=FORMAT,
        database_id=identity,
        decision=decision,
        state=state,
        count=len(items),
        more=more,
        by_rule=dict(sorted(Counter(rule for item in items for rule in item["rules"]).items())),
        by_origin=dict(sorted(Counter(item["origin"] for item in items).items())),
        items=items,
    )


def _validate_preview(tx, preview, decision):
    if not isinstance(preview, dict) or preview.get("format") != FORMAT or preview.get("state") != "pending_review":
        raise ContractError("INPUT_INVALID", "source_preview")
    if decision not in {"release", "retain"} or decision != preview.get("decision"):
        raise ContractError("INPUT_INVALID", "review_decision")
    if preview.get("database_id") != tx.database_identity()[0]:
        raise ContractError("VERSION_CONFLICT", "database_id")
    items = preview.get("items")
    if not isinstance(items, list) or len(items) > MAX_ITEMS:
        raise ContractError("INPUT_INVALID", "source_preview")
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            raise ContractError("INPUT_INVALID", "source_preview")
        ref, revision, digest = (item.get(name) for name in ("ref", "revision", "content_sha256"))
        if (
            type(ref) is not str
            or not 1 <= len(ref) <= 240
            or type(revision) is not int
            or revision < 1
            or type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or (ref, revision) in seen
        ):
            raise ContractError("INPUT_INVALID", "source_preview")
        seen.add((ref, revision))
    return items


def _already_decided(core, context, item, decision):
    """Recognize an unchanged completed decision without exposing inaccessible rows."""
    with core.storage.read(context) as tx:
        row = (
            tx._check()
            .execute(
                "SELECT e.project_id,e.branch_id,s.* FROM source_events e JOIN source_safety s "
                "ON s.source_ref=e.event_id AND s.source_revision=e.source_revision "
                "WHERE e.event_id=? AND e.source_revision=?",
                (item["ref"], item["revision"]),
            )
            .fetchone()
        )
        if row is None or (
            row["state"] != "released"
            if decision == "release"
            else row["state"] != "quarantined" or "review_rejected" not in json.loads(row["rules_json"])
        ):
            return False
        view = tx.with_context(replace(context, project_id=row["project_id"], branch_id=row["branch_id"]))
        loaded = view._source_row(view._check(), item["ref"], item["revision"])
        return (
            loaded is not None
            and deletion_allows(view, "event", item["ref"])
            and hashlib.sha256(loaded[0]["content"].encode("utf-8")).hexdigest() == item["content_sha256"]
        )


def apply_sources(core, authority, preview, *, decision):
    """Apply previewed pending decisions independently through existing review writers.

    An intervening quarantine or changed body fails that item. Other items proceed;
    unchanged completed decisions are skipped when a command is repeated.
    """
    context = maintenance_context(core.storage, authority)
    with core.storage.read(context) as tx:
        items = _validate_preview(tx, preview, decision)
    result: dict = dict(succeeded=0, failed=0, skipped=0, errors=[])
    for item in items:
        try:
            if decision == "release":
                release_source(
                    core,
                    authority,
                    item["ref"],
                    item["revision"],
                    content_sha256=item["content_sha256"],
                    expected_state="pending_review",
                )
            else:
                finish_review(
                    core,
                    authority,
                    item["ref"],
                    item["revision"],
                    content_sha256=item["content_sha256"],
                    approved=False,
                )
            result["succeeded"] += 1
        except (ContractError, sqlite3.Error, OSError, TruthWriterBusyError) as exc:
            if isinstance(exc, ContractError) and _already_decided(core, context, item, decision):
                result["skipped"] += 1
            else:
                result["failed"] += 1
                result["errors"].append(
                    dict(
                        ref=item["ref"],
                        revision=item["revision"],
                        code=exc.code
                        if isinstance(exc, ContractError)
                        else "TRUTH_WRITER_BUSY"
                        if isinstance(exc, TruthWriterBusyError)
                        else type(exc).__name__,
                    )
                )
    return result
