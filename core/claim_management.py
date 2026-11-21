"""Owner-authorized fact review, merge and restoration within an existing transaction."""

from __future__ import annotations

from copy import deepcopy

from ..contracts import ContractError
from .claims import Qualification, canonical_time, evidence_refs
from .review_policy import owner_publication


def management_claims_private(tx, clock, versions):
    """Check whether a management write would copy any private claim evidence."""
    from . import private

    # Read-only mode permits reads but not the new records made by these actions.
    intervals = [dict(rule, mode="full") for rule in private.active_intervals(tx, clock.utc_now())]
    return any(private.claim_blocked(tx, intervals, version) for version in versions)


def merge_entity_from_management(tx, clock, source_ref, source_revision, target_ref, target_revision, *, name=None):
    """Record an owner's identity decision as the existing source-backed alias."""
    from .entity_resolution import publish_entity_merge
    from .evidence_support import SupportVerdict

    versions = []
    for ref, revision in ((source_ref, source_revision), (target_ref, target_revision)):
        if type(revision) is not int or revision < 1:
            raise ContractError("INPUT_INVALID", "expected_revision")
        version = tx.claims.version(ref, revision)
        if version is None or version.current_revision != revision:
            raise ContractError("VERSION_CONFLICT", "expected_revision")
        versions.append(version)
    source, target = versions
    if management_claims_private(tx, clock, versions):
        raise ContractError("ACCESS_DENIED", "private_mode")
    verdict = SupportVerdict("", "supported", target.ref, target.revision, "duplicate")
    with owner_publication():
        result = publish_entity_merge(tx, source, target, verdict, now=clock.utc_now(), name=name, only_unmerged=True)
    if result is None:
        raise ContractError("VERSION_CONFLICT", "entity_alias")
    return result


def health_fact_pair(tx, clock, first_ref, first_revision, second_ref, second_revision):
    """Load two reviewed current facts in one audience and reject stale or private input."""
    from .claims import valid_at

    versions = []
    for ref, revision in ((first_ref, first_revision), (second_ref, second_revision)):
        if type(revision) is not int or revision < 1:
            raise ContractError("INPUT_INVALID", "expected_revision")
        version = tx.claims.version(ref, revision)
        if version is None or version.current_revision != revision:
            raise ContractError("VERSION_CONFLICT", "expected_revision")
        if version.state != "active" or version.suppressed or not valid_at(version, clock.utc_now(), historical=False):
            raise ContractError("VERSION_CONFLICT", "fact_state")
        tx.claims.live_sources(evidence_refs(version.payload), version.scope_id)
        if version.payload["kind"] == "observation":
            from .reflection_validity import reflection_live

            if not reflection_live(tx, version):
                raise ContractError("VERSION_CONFLICT", "observation_dependencies")
        versions.append(version)
    if first_ref == second_ref or management_claims_private(tx, clock, versions):
        raise ContractError("ACCESS_DENIED", "fact_pair")
    return tuple(versions)


def health_pair_kind(first, second):
    """Classify only strict normalized slots; a lossy candidate reading proves no fact relation."""
    from .claims import claim_slot, valid_at
    from .fact_identity import normalize_fact_component

    if (first.scope_id, first.project_id, first.branch_id) != (second.scope_id, second.project_id, second.branch_id):
        return None
    proposals = []
    for item in (first, second):
        payload = dict(item.payload)
        payload.update((key, normalize_fact_component(payload[key])) for key in ("subject", "predicate"))
        proposals.append(payload)
    if claim_slot(first.scope_id, first.project_id, first.branch_id, proposals[0]) != claim_slot(
        second.scope_id, second.project_id, second.branch_id, proposals[1]
    ):
        return None
    values = [normalize_fact_component(item.payload["value_text"]) for item in (first, second)]
    if values[0] == values[1]:
        periods = [(canonical_time(v.valid_from), canonical_time(v.valid_to)) for v in (first, second)]
        return "duplicate" if periods[0] == periods[1] else None
    start = max((v.valid_from for v in (first, second) if v.valid_from), default="0001-01-01T00:00:00Z")
    return "contradiction" if all(valid_at(v, start, historical=False) for v in (first, second)) else None


def health_apply_facts(tx, clock, first, second, *, merge):
    """Append owner-reviewed fact heads; callers record the cross-reference atomically."""
    if merge and (canonical_time(first.valid_from), canonical_time(first.valid_to)) != (
        canonical_time(second.valid_from),
        canonical_time(second.valid_to),
    ):
        raise ContractError("DERIVATION_INVALID", "different_validity")
    kind = health_pair_kind(first, second)
    if kind != ("duplicate" if merge else "contradiction"):
        raise ContractError("VERSION_CONFLICT", "fact_pair")
    kept = first
    if merge:
        payload = deepcopy(first.payload)
        for span in second.payload["evidence_spans"]:
            if span not in payload["evidence_spans"]:
                payload["evidence_spans"].append(deepcopy(span))
        kept = tx.claims.append(
            first.scope_id,
            payload,
            Qualification(first.state, first.basis, first.reason),
            recorded_at=clock.utc_now(),
            previous=first,
            expected_revision=first.revision,
        )
    replaced = tx.claims.append(
        second.scope_id,
        second.payload,
        Qualification("superseded", second.basis, "health_superseded"),
        recorded_at=clock.utc_now(),
        previous=second,
        expected_revision=second.revision,
    )
    if first.payload["kind"] == "observation":
        if merge:
            health_observation_dependencies(tx, kept, (first, second))
        health_observation_dependencies(tx, replaced, (second,))
    return kept, replaced


def health_restore_fact(tx, clock, original, head):
    """Append the pre-decision state without rewriting either reference's history."""
    with owner_publication():
        saved = tx.claims.append(
            original.scope_id,
            original.payload,
            Qualification(original.state, original.basis, original.reason),
            recorded_at=clock.utc_now(),
            previous=head,
            expected_revision=head.revision,
        )
    if original.payload["kind"] == "alias":
        from .entity_resolution import copy_entity_proof

        copy_entity_proof(tx, original, saved)
    if original.payload["kind"] == "observation":
        row = (
            tx._check()
            .execute(
                "SELECT manifest_json,signature,confirmed FROM observations WHERE claim_ref=? AND claim_revision=?",
                (original.ref, original.revision),
            )
            .fetchone()
        )
        if row is None:
            raise ContractError("SOURCE_MISSING", "observation_dependencies")
        tx._check(write=True).execute("INSERT INTO observations VALUES(?,?,?,?,?)", (saved.ref, saved.revision, *row))
    return saved


def health_observation_dependencies(tx, saved, originals):
    """Carry the full induction manifests onto the new version with a fresh dependency fence."""
    import json

    from . import private
    from .reflection_validity import dependency_signature, live_members

    members = {}
    confirmed = 0
    for original in originals:
        row = (
            tx._check()
            .execute(
                "SELECT manifest_json,confirmed FROM observations WHERE claim_ref=? AND claim_revision=?",
                (original.ref, original.revision),
            )
            .fetchone()
        )
        if row is None:
            raise ContractError("SOURCE_MISSING", "observation_dependencies")
        confirmed = max(confirmed, row[1])
        for item in json.loads(row[0]):
            members[(item["ref"], item["revision"])] = item
    manifest = sorted(members.values(), key=lambda item: item["position"])
    sources = live_members(tx, manifest)
    intervals = [dict(rule, mode="full") for rule in private.active_intervals(tx, saved.recorded_from)]
    if any(private.source_blocked(intervals, source) for source in sources):
        raise ContractError("ACCESS_DENIED", "private_mode")
    tx._check(write=True).execute(
        "INSERT INTO observations VALUES(?,?,?,?,?)",
        (saved.ref, saved.revision, json.dumps(manifest), dependency_signature(tx, manifest), confirmed),
    )
