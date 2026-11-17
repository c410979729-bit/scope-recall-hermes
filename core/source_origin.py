"""Owner-only provenance revisions, independent of content revisions and their evidence identities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..contracts import ContractError
from . import lineage
from .claim_storage import parse_source_ref
from .claims import Qualification, canonical_time, evidence_refs
from .delete_storage import canonical
from .management_query import partition
from .requalify import _denial_standing
from .restore import maintenance_context
from .secret_patterns import contains_secret_like_text
from .source_records import source_origin
from .summary_validity import is_conversation_summary

ACTION = "source_origin"
REASON = "source_origin_corrected"


def history(conn, ref, revision):
    """Read content-free provenance revisions for one exact content version, oldest first."""
    rows = conn.execute(
        "SELECT action_id,actor,recorded_at,targets_json FROM management_actions WHERE action=? "
        "AND EXISTS(SELECT 1 FROM json_each(targets_json,'$.sources') "
        "WHERE json_extract(value,'$.ref')=? AND json_extract(value,'$.revision')=?) ORDER BY rowid",
        (ACTION, ref, revision),
    )
    items = []
    for row in rows:
        action = json.loads(row["targets_json"])
        source = next(s for s in action["sources"] if (s["ref"], s["revision"]) == (ref, revision))
        items.append(
            dict(
                action_id=row["action_id"],
                actor=row["actor"],
                recorded_at=row["recorded_at"],
                reason=action["reason"],
                after=action["origin"],
                **source,
            )
        )
    return items


def _selection(api, tx, refs, origin):
    context = api._context()
    sources, seen = [], set()
    for value in sorted(set(refs)):
        ref, revision = parse_source_ref(value) if "@" in value else (value, None)
        kind, location = api._location(tx, ref)
        if kind != "event":
            raise ContractError("INPUT_INVALID", "event_required")
        revision = revision or location["revision"]
        if (ref, revision) in seen:
            continue
        seen.add((ref, revision))
        view = partition(tx, context, location["project_id"], location["branch_id"])
        source = view.source(ref, revision)
        if source is None:
            raise ContractError("SOURCE_MISSING", "source_ref")
        prior = history(tx._check(), ref, revision)
        if source_origin(source) != "human_direct" and not prior:
            raise ContractError("ACCESS_DENIED", "origin_correction_requires_human_source")
        before = dict(
            origin=source.event["origin"],
            original_origin=source.event.get("source_original_origin"),
            principal=source.event.get("source_principal"),
        )
        if origin == "human_direct" and not any(
            s["before"]["origin"] == "human_direct" or s["before"]["original_origin"] == "human_direct" for s in prior
        ):
            raise ContractError("ACCESS_DENIED", "origin_restore_requires_history")
        if source_origin(source) == origin:
            raise ContractError("VERSION_CONFLICT", "origin_unchanged")
        sources.append(
            dict(
                ref=ref,
                revision=revision,
                scope_id=source.scope_id,
                project_id=source.project_id,
                branch_id=source.branch_id,
                before=before,
                origin_revision=len(prior) + 1,
            )
        )
    return sources


def _descendants(conn, pairs):
    """Follow source-to-source lineage without expanding through an episode's unrelated messages."""
    return {
        tuple(row)
        for row in conn.execute(
            "WITH RECURSIVE changed(ref,revision) AS ("
            "SELECT json_extract(value,'$[0]'),json_extract(value,'$[1]') FROM json_each(?) UNION "
            "SELECT e.object_ref,e.object_revision FROM evidence_links e JOIN changed c "
            "ON e.source_ref=c.ref AND e.source_revision=c.revision WHERE e.object_kind='event') "
            "SELECT ref,revision FROM changed",
            (canonical(sorted(pairs)),),
        )
    }


def dependent_claim_count(tx, ref, revision):
    """Count claim identities with any version backed by this source lineage."""
    changed = _descendants(tx._check(), {(ref, revision)})
    return sum(1 for _view, _versions in _dependent_claims(tx, changed))


def _dependent_claims(tx, changed):
    """Read complete claim histories; even a fully shadowed assertion can matter after a withdrawal."""
    rows = (
        tx._check()
        .execute(
            "SELECT c.claim_id,c.project_id,c.branch_id FROM claims c "
            "WHERE c.read_blocked=0 AND c.scope_id IN (SELECT value FROM json_each(?)) "
            "AND EXISTS(SELECT 1 FROM evidence_links e JOIN json_each(?) s "
            "ON e.source_ref=json_extract(s.value,'$[0]') AND e.source_revision=json_extract(s.value,'$[1]') "
            "WHERE e.object_kind='claim' AND e.object_ref=c.claim_id)",
            (canonical(sorted(tx.context.allowed_scope_ids)), canonical(sorted(changed))),
        )
        .fetchall()
    )
    for row in rows:
        view = partition(tx, tx.context, row["project_id"], row["branch_id"])
        yield view, view.claims.versions(row["claim_id"])


def _leaves(conn, pairs, selected):
    """Resolve derived source evidence to its roots; selected roots stop traversal even after demotion."""
    pending, seen, leaves = list(pairs), set(), set()
    while pending:
        pair = pending.pop()
        if pair in seen:
            continue
        seen.add(pair)
        if len(seen) > 2000:
            raise ContractError("INPUT_INVALID", "evidence_depth")
        parents = (
            []
            if pair in selected
            else list(
                conn.execute(
                    "SELECT source_ref,source_revision FROM evidence_links WHERE object_kind='event' "
                    "AND object_ref=? AND object_revision=?",
                    pair,
                )
            )
        )
        if parents:
            pending.extend(tuple(row) for row in parents)
        else:
            leaves.add(pair)
    return leaves


def _version_evidence(conn, version, selected):
    """Describe source roots without disclosing evidence text or dropping auxiliary references.

    A version's evidence is its payload's references together with its stored evidence links: a fact migrated
    from a legacy store keeps its evidence only as links, with no spans in its payload.
    """
    pairs = {parse_source_ref(ref) for ref in evidence_refs(version.payload)}
    pairs.update(lineage.evidence(conn, "claim", version.ref, version.revision))
    leaves = _leaves(conn, pairs, selected)
    chosen, other = leaves & selected, leaves - selected
    distribution = "mixed" if chosen and other else "selected_only" if chosen else "elsewhere" if other else "none"
    return dict(
        revision=version.revision,
        state=version.state,
        valid_from=version.valid_from,
        valid_to=version.valid_to,
        evidence_distribution=distribution,
        evidence_refs=sorted(f"{r}@{v}" for r, v in leaves),
        selected_evidence_refs=sorted(f"{r}@{v}" for r, v in chosen),
        other_evidence_refs=sorted(f"{r}@{v}" for r, v in other),
    )


def _claim_effect(view, head, versions, details):
    """Use ordinary head withdrawal only when no earlier version can regain independent authority."""
    live = [item for item in details if item["state"] != "retracted"]
    affected = [item for item in live if item["selected_evidence_refs"]]
    if head.state == "retracted" or not affected:
        return "historical_unchanged"
    if is_conversation_summary(head) or head.payload["kind"] == "observation":
        return "source_view_unchanged"
    # Legacy denial heads have a separate repair path that may republish an earlier payload.
    if head.state in {"active", "disputed"} and _denial_standing(view, head, versions)[1] is not None:
        return "review"
    if all(item["evidence_distribution"] == "selected_only" for item in live):
        # Compared as the selector reads them, so equal instants written differently count as the same window.
        window = (canonical_time(head.valid_from), canonical_time(head.valid_to))
        same = all((canonical_time(item["valid_from"]), canonical_time(item["valid_to"])) == window for item in live)
        # A migrated fact keeps its evidence only in links; withdrawing it would append a payload without spans,
        # so the owner decides instead.
        migrated = any(not evidence_refs(v.payload) for v in versions if v.state != "retracted")
        return "retract" if same and not migrated else "review"
    if any(item["evidence_distribution"] == "selected_only" for item in affected):
        return "review"
    return "mixed_unchanged"


def _claims(tx, selected, changed):
    """Preview one decision per claim, with every version's validity and evidence distribution."""
    conn, items = tx._check(), []
    for view, versions in _dependent_claims(tx, changed):
        head = next(v for v in versions if v.revision == v.current_revision)
        details = [_version_evidence(conn, version, selected) for version in versions]
        if not any(item["selected_evidence_refs"] for item in details):
            continue
        items.append(
            dict(
                ref=head.ref,
                revision=head.revision,
                head_revision=head.current_revision,
                scope_id=head.scope_id,
                project_id=head.project_id,
                branch_id=head.branch_id,
                state=head.state,
                effect=_claim_effect(view, head, versions, details),
                evidence_refs=sorted({ref for item in details for ref in item["evidence_refs"]}),
                versions=details,
            )
        )
    return sorted(items, key=lambda s: s["ref"])


def _pending(conn, changed, claims):
    refs = {f"{ref}@{revision}" for ref, revision in changed}
    withdrawn = {(s["ref"], s["revision"]) for s in claims if s["effect"] == "retract"}
    evaluations = []
    for row in conn.execute("SELECT * FROM candidate_evaluations WHERE state IN ('queued','failed')"):
        if (row["candidate_ref"], row["candidate_revision"]) in withdrawn or set(
            json.loads(row["evidence_refs_json"])
        ) & refs:
            evaluations.append(
                dict(
                    evaluation_id=row["evaluation_id"],
                    work_id=row["work_id"],
                    ref=row["candidate_ref"],
                    revision=row["candidate_revision"],
                    state=row["state"],
                )
            )
    evaluation_work = {s["work_id"] for s in evaluations}
    for row in conn.execute(
        "SELECT work_id,confirmation_ref,confirmation_revision,assistant_ref,assistant_revision "
        "FROM adoption_processing"
    ):
        if any((row[f"{side}_ref"], row[f"{side}_revision"]) in changed for side in ("confirmation", "assistant")):
            evaluation_work.add(row["work_id"])
    work = []
    for row in conn.execute(
        "SELECT work_id,work_type,subject_ref,subject_revision,state,lease_token FROM work_items "
        "WHERE state IN ('pending','leased','failed') AND work_type IN ('consolidate','evaluate_candidate')"
    ):
        if (row["subject_ref"], row["subject_revision"]) in changed or row["work_id"] in evaluation_work:
            work.append(dict(row))
    return evaluations, work


def _related(tx, selected, changed):
    """Find stored authority decisions separately from claim evidence and episode membership."""
    conn, items = tx._check(), []
    for row in conn.execute(
        "SELECT update_id,source_ref,source_revision FROM unresolved_updates WHERE state='unresolved'"
    ):
        if (row["source_ref"], row["source_revision"]) in changed:
            items.append(dict(kind="update", ref=row["update_id"], effect="obsolete"))
    for row in conn.execute(
        "SELECT confirmation_ref,confirmation_revision,assistant_ref,assistant_revision "
        "FROM adoption_relations WHERE state='bound'"
    ):
        if any((row[f"{side}_ref"], row[f"{side}_revision"]) in changed for side in ("confirmation", "assistant")):
            items.append(
                dict(
                    kind="adoption",
                    ref=row["confirmation_ref"],
                    revision=row["confirmation_revision"],
                    effect="pending_review",
                )
            )
    for row in conn.execute(
        "SELECT r.reference_id,r.current_revision,v.payload_json FROM reference_bindings r "
        "JOIN reference_versions v ON v.reference_id=r.reference_id "
        "AND v.revision=r.current_revision WHERE r.read_blocked=0"
    ):
        if not json.loads(row["payload_json"]).get("resolved_ref"):
            continue
        pairs = lineage.evidence(conn, "reference", row["reference_id"], row["current_revision"])
        leaves = _leaves(conn, pairs, selected)
        if leaves & selected:
            items.append(
                dict(
                    kind="reference",
                    ref=row["reference_id"],
                    revision=row["current_revision"],
                    effect="unresolved" if leaves <= selected else "mixed_unchanged",
                )
            )
    for row in conn.execute("SELECT source_ref,source_revision FROM candidate_source_triggers WHERE truncated=1"):
        if (row["source_ref"], row["source_revision"]) in changed:
            items.append(dict(kind="trigger", ref=row["source_ref"], revision=row["source_revision"], effect="stop"))
    return sorted(items, key=lambda s: (s["kind"], s["ref"]))


def _settle_related(conn, items, now):
    for item in items:
        if item["kind"] == "update":
            conn.execute(
                "UPDATE unresolved_updates SET state='obsolete',resolved_at=? WHERE update_id=?", (now, item["ref"])
            )
        elif item["kind"] == "adoption":
            conn.execute(
                "UPDATE adoption_relations SET state='pending_review',reason=?,proposal_sha256=NULL "
                "WHERE confirmation_ref=? AND confirmation_revision=?",
                (REASON, item["ref"], item["revision"]),
            )
        elif item["kind"] == "trigger":
            conn.execute(
                "UPDATE candidate_source_triggers SET truncated=0 WHERE source_ref=? AND source_revision=?",
                (item["ref"], item["revision"]),
            )
        elif item["effect"] == "unresolved":
            _unresolve_reference(conn, item, now)


def _unresolve_reference(conn, item, now):
    """Append an unresolved binding while retaining its candidates, evidence and earlier decisions."""
    ref, old = item["ref"], item["revision"]
    revision = conn.execute("SELECT max(revision)+1 FROM reference_versions WHERE reference_id=?", (ref,)).fetchone()[0]
    conn.execute(
        "INSERT INTO reference_versions SELECT reference_id,?,"
        "json_set(payload_json,'$.resolution','unresolved','$.resolved_ref',NULL),?,? "
        "FROM reference_versions WHERE reference_id=? AND revision=?",
        (revision, REASON, now, ref, old),
    )
    conn.execute("UPDATE reference_bindings SET current_revision=? WHERE reference_id=?", (revision, ref))
    for source, version in lineage.evidence(conn, "reference", ref, old):
        lineage.link(conn, "reference", ref, revision, source, version)
    for kind, target, version in conn.execute(
        "SELECT dependency_kind,dependency_ref,dependency_revision "
        "FROM object_dependencies WHERE object_kind='reference' "
        "AND object_ref=? AND object_revision=?",
        (ref, old),
    ).fetchall():
        lineage.depend(conn, "reference", ref, revision, kind, target, version)


def plan(api, tx, refs, origin, reason):
    """Build an exact, content-free snapshot that can be compared under the writer lock."""
    if (
        type(origin) is not str
        or origin not in {"host_generated", "origin_unknown", "human_direct"}
        or type(reason) is not str
        or not reason.strip()
        or len(reason) > 500
        or contains_secret_like_text(reason)
    ):
        raise ContractError("INPUT_INVALID", "origin_or_reason")
    if not isinstance(refs, (list, tuple)) or not 1 <= len(refs) <= 2000 or any(type(r) is not str for r in refs):
        raise ContractError("INPUT_INVALID", "source_refs")
    sources = _selection(api, tx, refs, origin)
    selected = {(s["ref"], s["revision"]) for s in sources}
    changed = _descendants(tx._check(), selected)
    claims = _claims(tx, selected, changed) if origin != "human_direct" else []
    evaluations, work = _pending(tx._check(), changed, claims)
    related = _related(tx, selected, changed) if origin != "human_direct" else []
    result: dict[str, Any] = dict(
        origin=origin,
        reason=reason,
        memory_epoch=tx.memory_epoch(),
        sources=sources,
        claims=claims,
        evaluations=evaluations,
        work=work,
        related=related,
    )
    if len(claims) + len(evaluations) + len(work) + len(related) > 10000:
        raise ContractError("INPUT_INVALID", "selection_size")
    result["preview_id"] = hashlib.sha256(canonical(result).encode()).hexdigest()
    return result


def preview(api, refs, *, origin, reason):
    """Preview a local owner's provenance correction without writing any rows."""
    with api.core.storage.read(maintenance_context(api.core.storage, api.authority)) as tx:
        return plan(api, tx, refs, origin, reason)


def _write_sources(conn, current):
    # The ingress fingerprint remains a retry receipt. Replacing it would let an old human-labelled envelope
    # collide, acquire another key and restore its former authority as a new source.
    for source in current["sources"]:
        before, origin = source["before"], current["origin"]
        imported = before["origin"] == "imported"
        principal = None
        if origin == "human_direct":
            prior = history(conn, source["ref"], source["revision"])
            principal = prior[0]["before"]["principal"]
        conn.execute(
            "UPDATE source_events SET origin=?,source_original_origin=?, "
            "extra_json=json_remove(extra_json,'$.source_principal') WHERE event_id=? AND source_revision=?",
            ("imported" if imported else origin, origin if imported else None, source["ref"], source["revision"]),
        )
        if principal is not None:
            conn.execute(
                "UPDATE source_events SET extra_json=json_set(extra_json,'$.source_principal',json(?)) "
                "WHERE event_id=? AND source_revision=?",
                (canonical(principal), source["ref"], source["revision"]),
            )


def _settle_work(conn, current, now):
    for item in current["work"]:
        conn.execute(
            "UPDATE work_items SET state='obsolete',lease_token=lease_token+1,lease_owner=NULL,lease_until=NULL,"
            "last_error_code=? WHERE work_id=?",
            (REASON, item["work_id"]),
        )
    for item in current["evaluations"]:
        conn.execute(
            "UPDATE candidate_evaluations SET state='obsolete',reason=?,completed_at=? WHERE evaluation_id=?",
            (REASON, now, item["evaluation_id"]),
        )
        conn.execute(
            "UPDATE candidate_lifecycle SET processing_state='waiting_evidence',reason=?,updated_at=? "
            "WHERE candidate_ref=? AND candidate_revision=?",
            (REASON, now, item["ref"], item["revision"]),
        )
    for item in current["claims"]:
        if item["effect"] == "retract":
            conn.execute(
                "UPDATE candidate_lifecycle SET processing_state='blocked',reason=?,updated_at=? "
                "WHERE candidate_ref=? AND candidate_revision=?",
                (REASON, now, item["ref"], item["revision"]),
            )


def apply(api, proposed):
    """Recheck and apply all previewed consequences in the sole SQLite write transaction."""
    context = maintenance_context(api.core.storage, api.authority)
    try:
        refs = [f"{s['ref']}@{s['revision']}" for s in proposed["sources"]]
        origin, reason, identity = proposed["origin"], proposed["reason"], proposed["preview_id"]
    except (KeyError, TypeError):
        raise ContractError("INPUT_INVALID", "preview") from None
    with api.core.storage.write(context, remaining_seconds=10) as tx:
        current = plan(api, tx, refs, origin, reason)
        if current["preview_id"] != identity:
            raise ContractError("VERSION_CONFLICT", "preview")
        now, conn = api.core.clock.utc_now(), tx._check(write=True)
        for item in current["claims"]:
            if item["effect"] != "retract":
                continue
            view = partition(tx, context, item["project_id"], item["branch_id"], writable=True)
            head = view.claims.version(item["ref"], item["revision"])
            view.claims.append(
                head.scope_id,
                head.payload,
                Qualification("retracted", head.basis, REASON),
                recorded_at=now,
                previous=head,
                advance_head=True,
                expected_revision=head.current_revision,
            )
        _settle_work(conn, current, now)
        _settle_related(conn, current["related"], now)
        _write_sources(conn, current)
        conn.execute("UPDATE instance_meta SET memory_epoch=memory_epoch+1 WHERE singleton=1")
        api._audit(tx, identity, ACTION, current)
        return dict(preview_id=identity, actor="local_user_management", memory_epoch=tx.memory_epoch())
