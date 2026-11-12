"""Bounded read-only profile and one-hop entity views over admitted claims.

Both views follow one shape: resolve the subject, select and clip candidate
versions in a first read, release them through the authority boundary, recheck
them in a second read, then render.  They differ only in which refs are
enumerated and how the released statements are grouped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Callable

from ..contracts import (
    ContractError,
    EntryLabel,
    SourceContext,
    TrustedContext,
    validate_model_request,
    validate_payload,
)
from .aliases import entity_alias_proof, validate_alias_source, validate_alias_target
from .claim_storage import parse_source_ref
from .claims import ClaimVersion, effective_origin, evidence_refs, select_effective
from .model_content import WITHHELD_PREFIX, withhold_secret_items
from .recall_budget import canonical_render_json
from .retrieval_hydration import evidence_entries, evidence_source_contexts
from .visibility import CLOSED_INTENTION_STATES, ObjectRef, allowed, epoch_retracted, release_objects

DEFAULT_MAX_ITEMS = 16
DEFAULT_BUDGET_TOKENS = 4096
CANDIDATE_CAP = 200
_STATEMENT_KINDS = frozenset({"fact", "preference", "constraint", "decision", "intention"})
_SECTION_FOR_KIND = {
    "fact": "facts",
    "preference": "preferences",
    "constraint": "constraints",
    "decision": "decisions",
}
_SECTION_ORDER = ("facts", "preferences", "constraints", "decisions", "pending_intentions")
_RELEASE_ERRORS = frozenset({"VERSION_CONFLICT", "SOURCE_MISSING"})
_CURRENT_SUBJECT_ALIASES = frozenset({"user", "current_user", "用户", "我", "me", "myself", "i"})


# -- request and subject ------------------------------------------------------


def _verified_current_principal(context: TrustedContext) -> str | None:
    principal = context.source_principal
    if (
        principal is None
        or principal.kind != "human"
        or principal.resolution != "verified"
        or principal.principal_ref is None
    ):
        return None
    return principal.principal_ref


def _profile_subject(context: TrustedContext, requested: str) -> str:
    """Bind explicit current-person aliases to an attested speaker or local entry owner."""
    if requested.casefold() not in _CURRENT_SUBJECT_ALIASES:
        return requested
    principal_ref = _verified_current_principal(context)
    if principal_ref is None and context.subject_resolver is not None:
        subjects = context.subject_resolver(None)
        principal_ref = subjects[0] if subjects else None
    if principal_ref is None:
        raise ContractError("IDENTITY_UNBOUND", "subject")
    return principal_ref


def _budget_bytes(value: object) -> int:
    return len(canonical_render_json(value).encode("utf-8"))


def _strict_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ContractError("INPUT_INVALID", field)
    return value


def _normalize_read_request(name: str, request, context: TrustedContext, *, entity: bool) -> dict:
    if type(request) is not dict:
        raise ContractError("INPUT_INVALID", "object")
    body = dict(request)
    body.setdefault("max_items", DEFAULT_MAX_ITEMS)
    body.setdefault("budget_tokens", DEFAULT_BUDGET_TOKENS)
    if entity and "direction" not in body:
        body["direction"] = "outgoing" if body.get("action") == "probe" else "both"
    _strict_int(body["max_items"], "max_items")
    _strict_int(body["budget_tokens"], "budget_tokens")
    return validate_model_request(name, body, context)


def _public_markers(values: list[str], *, limit: int) -> list[str]:
    return list(dict.fromkeys(value for value in values if type(value) is str and value))[:limit]


# -- claim versions -----------------------------------------------------------


def _claim_evidence_keys(version: ClaimVersion) -> tuple[str, ...]:
    return tuple(dict.fromkeys(evidence_refs(version.payload)))[:32]


def _claim_evidence_live(tx, version: ClaimVersion) -> bool:
    if version.suppressed or not allowed(tx, "claim", version.ref, automatic=False):
        return False
    keys = _claim_evidence_keys(version)
    if not keys:
        return False
    for key in keys:
        try:
            source_ref, source_revision = parse_source_ref(key)
        except ContractError:
            return False
        if not allowed(tx, "event", source_ref, automatic=False):
            return False
        try:
            tx.claims.require_live_source(source_ref, source_revision)
        except ContractError:
            return False
        source = tx.source(source_ref, source_revision)
        if source is None or source.suppressed:
            return False
    return True


def _intention_state(version: ClaimVersion) -> str | None:
    intention = version.payload.get("intention")
    if not isinstance(intention, dict):
        return None
    state = intention.get("state")
    return state if type(state) is str else None


def _is_current_statement(version: ClaimVersion) -> bool:
    if version.suppressed or version.state not in {"active", "disputed"}:
        return False
    kind = version.payload["kind"]
    if kind not in _STATEMENT_KINDS:
        return False
    return not (kind == "intention" and _intention_state(version) in CLOSED_INTENTION_STATES)


def _sort_key(version: ClaimVersion) -> tuple[str, str, str, int]:
    payload = version.payload
    return (payload["predicate"], payload["value_text"], version.ref, version.revision)


def _claim_item(
    version: ClaimVersion,
    contexts: list[SourceContext] | None,
    *,
    entries: list[EntryLabel] | None = None,
    direction: str | None = None,
) -> dict:
    """One rendered statement; entity statements lead with their direction."""
    payload = version.payload
    identity = {"ref": version.ref, "revision": version.revision, "kind": payload["kind"]}
    statement = {"subject": payload["subject"], "predicate": payload["predicate"], "value_text": payload["value_text"]}
    item = {**identity, **statement} if direction is None else {"direction": direction, **statement, **identity}
    item.update(
        conditions=list(payload.get("conditions") or []),
        temporal_status="disputed"
        if version.state == "disputed"
        else "current"
        if version.state == "active"
        else "unknown",
        claim_state="disputed" if version.state == "disputed" else "active",
        valid_from=version.valid_from,
        valid_to=version.valid_to,
        evidence_refs=list(_claim_evidence_keys(version)),
        basis=version.basis,
    )
    if contexts:
        item["source_contexts"] = [dict(context) for context in contexts]
    if entries:
        item["entries"] = [dict(entry) for entry in entries]
    return item


# -- alias resolution ---------------------------------------------------------


@dataclass(frozen=True)
class _AliasResolution:
    """A subject proof and its bounded read names; ambiguity never chooses a canonical subject."""

    status: str
    subject: str | None
    fence: tuple[ObjectRef, ...]
    scan_capped: bool
    names: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()


def _alias_source_bound(tx, alias: ClaimVersion, target: ClaimVersion) -> bool:
    alias_payload = alias.payload.get("alias")
    if not isinstance(alias_payload, dict) or type(alias_payload.get("name")) is not str:
        return False
    try:
        roots = tx.claims.roots(_claim_evidence_keys(alias))
    except ContractError:
        return False
    for root in roots:
        if effective_origin(root) != "human_direct" or root.capture_state != "complete" or root.capture_gaps:
            continue
        for span in alias.payload.get("evidence_spans") or ():
            if (span.get("source_ref"), span.get("source_revision")) != (root.ref, root.revision):
                continue
            try:
                validate_alias_source(span["quote"], alias_payload["name"], target, source_text=root.content)
                return True
            except ContractError:
                continue
    return False


def _load_effective(tx, ref: str, now: str) -> ClaimVersion | None:
    versions = tx.claims.versions(ref)
    return select_effective(versions, now) if versions else None


def _accepted_alias(tx, ref, now):
    """Read one current relation through its evidence and exact-audience fences."""
    alias = _load_effective(tx, ref, now)
    if alias is None or alias.state != "active" or not _claim_evidence_live(tx, alias):
        return None
    relation = alias.payload.get("alias", {})
    target = _load_effective(tx, relation.get("target_ref", ""), now)
    if target is None or not _claim_evidence_live(tx, target):
        return None
    semantic = entity_alias_proof(tx, alias, target, now)
    try:
        validate_alias_target(
            target,
            scope_id=alias.scope_id,
            project_id=tx.context.project_id,
            branch_id=tx.context.branch_id,
            entity=semantic,
        )
    except ContractError:
        return None
    if alias.payload["subject"] != target.payload["subject"]:
        return None
    if not semantic and not _alias_source_bound(tx, alias, target):
        return None
    return alias, target, semantic


def _alias_component(tx, queried, now):
    """Walk only verified name links, with a shared cap for reads and admitted relations."""
    names, pending, seen, edges = {queried}, [queried], set(), []
    while pending and len(seen) < CANDIDATE_CAP:
        name = pending.pop()
        refs = tuple(
            dict.fromkeys(
                (
                    *tx.claims.list_refs(alias_name=name, limit=CANDIDATE_CAP),
                    *tx.claims.list_refs(subject=name, kind="alias", limit=CANDIDATE_CAP),
                )
            )
        )
        if len(refs) >= CANDIDATE_CAP:
            return names, edges, True
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            accepted = _accepted_alias(tx, ref, now)
            if accepted is None:
                continue
            alias, target, semantic = accepted
            edges.append((alias, target, semantic))
            for linked in (alias.payload.get("alias", {})["name"], target.payload["subject"]):
                if linked not in names:
                    names.add(linked)
                    pending.append(linked)
    return names, edges, bool(pending)


def _literal_audiences(tx, names, now):
    """Use recorded labels only as a conservative collision guard, never as identity evidence."""
    return tx.claims.subject_audiences(names)


def resolve_subject(tx, queried: str, now: str, *, owner_view: bool = False) -> _AliasResolution:
    """Resolve admitted aliases; read views also require a complete proof within the release bound."""
    resolver = tx.context.subject_resolver if owner_view else None
    owners = resolver(queried) if resolver is not None else ()
    if owners or _verified_current_principal(tx.context) == queried:
        return _AliasResolution("literal", queried, (), False, owners or (queried,), owners)
    names, edges, capped = _alias_component(tx, queried, now)
    if capped:
        return _AliasResolution("ambiguous", None, (), True)
    if not edges:
        if len(_literal_audiences(tx, (queried,), now)) > 1:
            return _AliasResolution("ambiguous", None, (), False, (queried,))
        return _AliasResolution("literal", queried, (), False, (queried,))
    alias_audiences = {(a.scope_id, a.project_id, a.branch_id) for a, _t, _s in edges}
    audiences = set(alias_audiences)
    audiences.update(_literal_audiences(tx, names, now))
    destinations = {t.payload["subject"] for _a, t, _s in edges}
    origins = {a.payload["alias"]["name"] for a, t, _s in edges if a.payload["alias"]["name"] != t.payload["subject"]}
    canonical = destinations - origins
    canonical_subject = min(canonical) if canonical else None
    owner_subjects = resolver(canonical_subject) if resolver is not None and canonical_subject is not None else ()
    if owner_subjects and not canonical.issubset(owner_subjects):
        owner_subjects = ()
    # Ownership permits facts under approved principals across audiences. A literal
    # name elsewhere still belongs to that audience and cannot inherit this alias.
    owner_conflict = owner_subjects and bool(_literal_audiences(tx, names - set(owner_subjects), now) - alias_audiences)
    literal_conflict = bool(tx.claims.subject_audiences((queried,), all_kinds=True)) and queried not in canonical
    if (
        (len(audiences) != 1 and (not owner_subjects or owner_conflict))
        or (len(canonical) != 1 and not owner_subjects)
        or (literal_conflict and not all(semantic for _a, _t, semantic in edges))
    ):
        return _AliasResolution("ambiguous", None, (), False, (queried,))
    fence = tuple(dict.fromkeys(ObjectRef("claim", v.ref, v.revision) for a, t, _s in edges for v in (a, t)))
    proofs = tx._check().execute(
        "SELECT p.source_ref,p.source_revision FROM json_each(?) accepted "
        "JOIN entity_alias_proofs p ON p.alias_ref=json_extract(accepted.value,'$[0]') "
        "AND p.alias_revision=json_extract(accepted.value,'$[1]')",
        (json.dumps([(a.ref, a.revision) for a, _t, _s in edges]),),
    )
    fence = tuple(dict.fromkeys((*fence, *(ObjectRef("claim", row[0], row[1]) for row in proofs))))
    # Read views release the complete proof, so a prefix cannot establish identity.
    # Query-name lookups do not publish this fence and keep their own resolution contract.
    if owner_view and len(fence) > CANDIDATE_CAP:
        return _AliasResolution("ambiguous", None, (), True)
    return _AliasResolution(
        "resolved", canonical_subject, fence, False, tuple(sorted(names | set(owner_subjects))), owner_subjects
    )


# -- select, release, recheck -------------------------------------------------


@dataclass(frozen=True)
class _Selection:
    """What the first read chose for release, and what bounded it."""

    versions: tuple[ClaimVersion, ...]
    scan_capped: bool
    truncated: bool
    outgoing: frozenset[str] = frozenset()
    incoming: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _Loaded:
    """Only released and rechecked versions, with provenance from the final snapshot."""

    epoch: int
    resolution: _AliasResolution
    selection: _Selection
    versions: dict[str, ClaimVersion]
    contexts: dict[str, list[SourceContext]]
    entries: dict[str, list[EntryLabel]]


@dataclass(frozen=True)
class _Unavailable:
    """A failed release fence without any statement or subject body."""

    epoch: int | None
    gaps: tuple[str, ...] = ()


Selector = Callable[[object, str, str, _AliasResolution], _Selection]


def _select_versions(tx, refs: tuple[str, ...], now: str) -> tuple[ClaimVersion, ...]:
    selected = []
    for ref in refs:
        version = _load_effective(tx, ref, now)
        if version is not None and _is_current_statement(version) and _claim_evidence_live(tx, version):
            selected.append(version)
    return tuple(selected)


def _clip_candidate_refs(
    *groups: tuple[str, ...], reserved: tuple[ObjectRef, ...] = ()
) -> tuple[tuple[str, ...], bool]:
    """Bound the union with the identity proof; selected targets already in it cost no extra release objects."""
    merged = sorted({ref for group in groups for ref in group})
    proof_refs = {item.ref for item in reserved}
    room = max(0, CANDIDATE_CAP - len(reserved))
    additional = [ref for ref in merged if ref not in proof_refs]
    kept = proof_refs | set(additional[:room])
    overflow = len(additional) > room or any(len(group) >= CANDIDATE_CAP for group in groups)
    return tuple(ref for ref in merged if ref in kept), overflow


def _clip_versions_for_view(
    versions: tuple[ClaimVersion, ...], *, max_items: int
) -> tuple[tuple[ClaimVersion, ...], bool]:
    ordered = tuple(sorted(versions, key=_sort_key))
    return ordered[:max_items], len(ordered) > max_items


def _release_selected(
    storage, clock, context, versions: tuple[ClaimVersion, ...], fence: tuple[ObjectRef, ...], epoch: int
):
    refs = tuple(dict.fromkeys((*(ObjectRef("claim", version.ref, version.revision) for version in versions), *fence)))
    if not refs:
        with storage.read(context) as tx:
            if epoch_retracted(tx, context, epoch):
                raise ContractError("VERSION_CONFLICT", "memory_epoch")
        return ()
    return release_objects(storage, clock, context, refs, expected_epoch=epoch, automatic=False, history=False)


def _recheck_released(tx, released, expected: tuple[ClaimVersion, ...]) -> tuple[ClaimVersion, ...] | None:
    by_ref = {item.ref: item for item in released if isinstance(item, ClaimVersion)}
    checked = []
    for version in expected:
        current = by_ref.get(version.ref)
        if current is None or current.revision != version.revision:
            return None
        if current.suppressed or not _is_current_statement(current) or not _claim_evidence_live(tx, current):
            return None
        checked.append(current)
    return tuple(checked)


def _load_view(
    storage, clock, context: TrustedContext, lookup_subject: str, select: Selector, *, requested_subject: str
) -> _Loaded | _Unavailable:
    """Resolve, select, release and recheck; the epoch must hold across all three reads."""
    now = clock.utc_now()
    try:
        with storage.read(context) as tx:
            epoch = tx.status().memory_epoch
            resolution = resolve_subject(tx, lookup_subject, now, owner_view=True)
            if resolution.status == "ambiguous" and resolution.scan_capped:
                selection = _Selection((), resolution.scan_capped, False)
            else:
                selection = select(tx, now, resolution.subject or lookup_subject, resolution)
        released = _release_selected(storage, clock, context, selection.versions, resolution.fence, epoch)
        with storage.read(context) as tx:
            if epoch_retracted(tx, context, epoch):
                return _Unavailable(tx.status().memory_epoch)
            # Manifest approvals are outside SQLite's epoch. Revalidate both the
            # current-person fallback and owner equivalences before exposing any body.
            try:
                subject = _profile_subject(context, requested_subject)
            except ContractError:
                return _Unavailable(epoch)
            if subject != lookup_subject or resolve_subject(tx, lookup_subject, now, owner_view=True) != resolution:
                return _Unavailable(epoch)
            checked = _recheck_released(tx, released, selection.versions)
            if checked is None and selection.versions:
                return _Unavailable(epoch)
            versions = {version.ref: version for version in checked or ()}
            contexts = {
                ref: evidence_source_contexts(tx, _claim_evidence_keys(version)) for ref, version in versions.items()
            }
            entries = {ref: evidence_entries(tx, _claim_evidence_keys(version)) for ref, version in versions.items()}
            current = tx.status().memory_epoch
            if current != epoch:
                return _Unavailable(current)
    except ContractError as exc:
        if exc.code not in _RELEASE_ERRORS:
            raise
        try:
            with storage.read(context) as tx:
                current = tx.status().memory_epoch
        except ContractError:
            current = None
        return _Unavailable(current, (exc.field,))
    return _Loaded(epoch, resolution, selection, versions, contexts, entries)


# -- rendering ----------------------------------------------------------------


def _empty_sections() -> dict[str, list]:
    return {name: [] for name in _SECTION_ORDER}


def _profile_item_lists(result: dict) -> list[list]:
    bodies = [result, *result.get("candidate_groups", [])]
    return [
        items for body in bodies for items in [*(body["sections"][name] for name in _SECTION_ORDER), body["disputed"]]
    ]


def _entity_item_lists(result: dict) -> list[list]:
    return [body["statements"] for body in [result, *result.get("candidate_groups", [])]]


def _candidate_body(body: dict, loaded: _Loaded, *, entity: bool) -> dict:
    """Move ambiguous literal facts into exact-audience groups after their normal release.

    Group labels come only from released versions. Empty groups, including ones
    emptied by secret withholding or budgets, disclose no namespace metadata.
    """
    if loaded.resolution.status != "ambiguous":
        return body
    groups: dict[tuple, dict] = {}
    lists = (
        [("statements", body["statements"])] if entity else [*body["sections"].items(), ("disputed", body["disputed"])]
    )
    for name, items in lists:
        for item in items:
            version = loaded.versions[item["ref"]]
            audience = (version.scope_id, version.project_id, version.branch_id)
            if audience not in groups:
                group: dict = {"audience": dict(zip(("scope_id", "project_id", "branch_id"), audience, strict=True))}
                if entity:
                    group["statements"] = []
                else:
                    group.update(sections=_empty_sections(), disputed=[])
                groups[audience] = group
            group = groups[audience]
            target = group[name] if entity or name == "disputed" else group["sections"][name]
            target.append(item)
        items.clear()
    if groups:
        body["candidate_groups"] = list(groups.values())
    return body


def _mark_gap(result: dict, gap: str, *, unmet: str | None = None) -> None:
    result["gaps"] = _public_markers([*result.get("gaps", []), gap], limit=16)
    if unmet:
        result["unmet_needs"] = _public_markers([*result.get("unmet_needs", []), unmet], limit=8)


def _apply_max_items(result: dict, item_lists: list[list], max_items: int) -> None:
    total = sum(len(items) for items in item_lists)
    if total <= max_items:
        return
    remaining = total - max_items
    for items in reversed(item_lists):
        while remaining and items:
            items.pop()
            remaining -= 1
    result["truncated"] = True
    _mark_gap(result, "max_items_cap", unmet="retry_with_higher_max_items")


def _drop_one_item(item_lists: list[list]) -> bool:
    for items in reversed(item_lists):
        if items:
            items.pop()
            return True
    return False


def _clear_terminal_body(result: dict, item_lists: list[list]) -> None:
    if result["status"] not in {"no_match", "unavailable"}:
        return
    for items in item_lists:
        items.clear()
    if result["status"] == "unavailable":
        result["resolved_subject"] = None
        result["alias_resolution"] = "none"


def _finalize_status(result: dict, *, item_count: int, alias_status: str, scan_capped: bool, disputed: bool) -> None:
    """The one place that turns what happened into status, coverage and answerability."""
    if result["status"] == "unavailable":
        result["coverage"] = "unknown"
        result["answerability"] = "unknown"
        return
    if alias_status == "ambiguous":
        result["status"] = "partial"
        result["coverage"] = "partial" if scan_capped else "unknown"
        result["answerability"] = "ambiguous"
        return
    truncated = result["truncated"]
    withheld = any(gap.startswith(WITHHELD_PREFIX) for gap in result["gaps"])
    budget_capped = "budget_token_cap" in result["gaps"]
    if item_count == 0:
        capped = truncated or budget_capped or withheld or "max_items_cap" in result["gaps"]
        result["status"] = "partial" if capped else "no_match"
        result["coverage"] = "partial" if capped else "unknown"
        result["answerability"] = "partial" if capped else "unknown"
        return
    partial = truncated or scan_capped or withheld
    result["status"] = "partial" if partial else "ok"
    result["coverage"] = "partial" if partial or budget_capped else "complete_for_query"
    result["answerability"] = "partial" if partial or disputed else "supported"


def _finish_view(
    result: dict,
    *,
    schema: str,
    max_items: int,
    budget_tokens: int,
    item_lists: list[list],
    alias_status: str,
    disputed_from,
    ambiguity_fallback: dict | None = None,
) -> dict:
    """Fit the canonical envelope; an optional expansion cannot displace a fitting empty ambiguity response."""
    _apply_max_items(result, item_lists, max_items)
    budget_clipped = False
    while True:
        if "candidate_groups" in result:
            result["candidate_groups"] = [
                group
                for group in result["candidate_groups"]
                if any(_entity_item_lists(group) if "statements" in group else _profile_item_lists(group))
            ]
        _finalize_status(
            result,
            item_count=sum(len(items) for items in item_lists),
            alias_status=alias_status,
            scan_capped=result["scan_capped"],
            disputed=bool(disputed_from()),
        )
        _clear_terminal_body(result, item_lists)
        fits = _budget_bytes(result) <= budget_tokens
        if ambiguity_fallback is not None and not any(item_lists) and (budget_clipped or not fits):
            # Discard every expansion-only marker with the last budget-clipped
            # item. The fallback is finalized normally and is attempted only once.
            result, ambiguity_fallback = ambiguity_fallback, None
            item_lists = []
            continue
        if fits:
            return validate_payload(schema, result)
        if not _drop_one_item(item_lists):
            raise ContractError("INPUT_INVALID", "budget_tokens")
        result["truncated"] = True
        budget_clipped = True
        _mark_gap(result, "budget_token_cap")
        if budget_tokens < DEFAULT_BUDGET_TOKENS:
            _mark_gap(result, "budget_token_cap", unmet="retry_once_with_budget_tokens_4096")


def _unavailable(request: dict, *, epoch: int | None, extra_gaps: tuple[str, ...] = (), entity: bool = False) -> dict:
    result = {
        "protocol_version": "1.1",
        "request_id": request["request_id"],
        "status": "unavailable",
        "memory_epoch": epoch,
        "subject": request["subject"],
        "resolved_subject": None,
        "alias_resolution": "none",
        "gaps": _public_markers([*extra_gaps, "memory_epoch_or_authority_changed"], limit=16),
        "coverage": "unknown",
        "answerability": "unknown",
        "truncated": False,
        "scan_capped": False,
        "unmet_needs": ["retry_against_current_epoch"],
    }
    if entity:
        result.update(action=request["action"], direction=request["direction"], statements=[])
        item_lists = [result["statements"]]
    else:
        result.update(sections=_empty_sections(), disputed=[])
        item_lists = _profile_item_lists(result)
    return _finish_view(
        result,
        schema="entity_view" if entity else "profile_view",
        max_items=request["max_items"],
        budget_tokens=request["budget_tokens"],
        item_lists=item_lists,
        alias_status="none",
        disputed_from=lambda: False,
    )


def paused_view(context, request, *, entity=False):
    """Return the ordinary empty view contract for a paused entry."""
    name = "entity_request" if entity else "profile_request"
    payload = _normalize_read_request(name, request, context, entity=entity)
    result = _unavailable(payload, epoch=None, entity=entity)
    result["gaps"] = ["entry_state:paused"]
    result["unmet_needs"] = []
    return result


def _view_result(payload: dict, loaded: _Loaded, lookup_subject: str, body: dict, gaps: list[str]) -> dict:
    """The shared envelope around a view body, with the bounds that applied."""
    ambiguous = loaded.resolution.status == "ambiguous"
    result = {
        "protocol_version": "1.1",
        "request_id": payload["request_id"],
        "status": "ok",
        "memory_epoch": loaded.epoch,
        "subject": payload["subject"],
        "resolved_subject": None if ambiguous else (loaded.resolution.subject or lookup_subject),
        "alias_resolution": loaded.resolution.status,
        **body,
        "gaps": gaps,
        "coverage": "partial",
        "answerability": "unknown",
        "truncated": False,
        "scan_capped": loaded.selection.scan_capped,
        "unmet_needs": [],
    }
    # Alias resolution can copy stored text into the envelope independently of its statements.
    subjects = withhold_secret_items([result["resolved_subject"]], gaps)
    result["resolved_subject"] = subjects[0] if subjects else None
    if loaded.selection.truncated:
        result["truncated"] = True
        result["gaps"].append("max_items_cap")
        result["unmet_needs"].append("retry_with_higher_max_items")
    if loaded.selection.scan_capped:
        result["gaps"].append("scan_capped")
        result["unmet_needs"].append("bounded_candidate_enumeration")
    return result


# -- the two views ------------------------------------------------------------


def _empty_ambiguity(payload: dict, loaded: _Loaded, lookup_subject: str, body: dict) -> dict | None:
    """Preserve the unexpanded ambiguity envelope, including only its identity-resolution bounds."""
    if loaded.resolution.status != "ambiguous":
        return None
    empty = replace(loaded, selection=_Selection((), loaded.resolution.scan_capped, False))
    return _view_result(payload, empty, lookup_subject, body, ["alias_ambiguous"])


def read_profile(storage, clock, context: TrustedContext, request) -> dict:
    """Read admitted current statements for a subject, within the caller's existing publication fences."""
    payload = _normalize_read_request("profile_request", request, context, entity=False)
    lookup_subject = _profile_subject(context, payload["subject"])

    def select(tx, now: str, subject: str, resolution: _AliasResolution) -> _Selection:
        groups = tuple(
            tx.claims.list_refs(subject=name, limit=CANDIDATE_CAP) for name in resolution.subjects or (subject,)
        )
        clipped, overflow = _clip_candidate_refs(*groups, reserved=resolution.fence)
        versions, truncated = _clip_versions_for_view(
            _select_versions(tx, clipped, now), max_items=payload["max_items"]
        )
        return _Selection(versions, resolution.scan_capped or overflow, truncated)

    loaded = _load_view(storage, clock, context, lookup_subject, select, requested_subject=payload["subject"])
    if isinstance(loaded, _Unavailable):
        return _unavailable(payload, epoch=loaded.epoch, extra_gaps=loaded.gaps)
    sections = _empty_sections()
    disputed: list[dict] = []
    for version in sorted(loaded.versions.values(), key=_sort_key):
        item = _claim_item(version, loaded.contexts.get(version.ref), entries=loaded.entries.get(version.ref))
        kind = version.payload["kind"]
        if version.state == "disputed":
            disputed.append(item)
        elif kind == "intention":
            if _intention_state(version) == "pending":
                sections["pending_intentions"].append(item)
        elif kind in _SECTION_FOR_KIND:
            sections[_SECTION_FOR_KIND[kind]].append(item)
    ambiguous = loaded.resolution.status == "ambiguous"
    gaps: list[str] = ["alias_ambiguous"] if ambiguous else []
    if disputed:
        gaps.append("disputed_facts_separated")
    for name, items in sections.items():
        sections[name] = withhold_secret_items(items, gaps)
    disputed = withhold_secret_items(disputed, gaps)
    if not ambiguous and not any(sections.values()) and not disputed:
        gaps.append("consolidation_required" if not loaded.versions else "no_current_qualified_facts")
    body = _candidate_body({"sections": sections, "disputed": disputed}, loaded, entity=False)
    if "candidate_groups" in body:
        body["candidate_groups"] = withhold_secret_items(body["candidate_groups"], gaps)
    result = _view_result(payload, loaded, lookup_subject, body, gaps)
    return _finish_view(
        result,
        schema="profile_view",
        max_items=payload["max_items"],
        budget_tokens=payload["budget_tokens"],
        item_lists=_profile_item_lists(result),
        alias_status=loaded.resolution.status,
        disputed_from=lambda: any(body["disputed"] for body in [result, *result.get("candidate_groups", [])]),
        ambiguity_fallback=_empty_ambiguity(
            payload, loaded, lookup_subject, {"sections": _empty_sections(), "disputed": []}
        ),
    )


def read_entity(storage, clock, context: TrustedContext, request) -> dict:
    """Read exact one-hop statements, keeping ambiguous audiences separate and all bounds shared."""
    payload = _normalize_read_request("entity_request", request, context, entity=True)
    lookup_subject = _profile_subject(context, payload["subject"])
    predicate = payload.get("predicate")
    probe = payload["action"] == "probe"
    direction = "outgoing" if probe else payload["direction"]
    want_out = probe or direction in {"outgoing", "both"}
    want_in = not probe and direction in {"incoming", "both"}

    def select(tx, now: str, subject: str, resolution: _AliasResolution) -> _Selection:
        outgoing_refs: tuple[str, ...] = ()
        incoming_refs: tuple[str, ...] = ()
        incoming_names: tuple[str, ...] = ()
        scan_hit = False
        if want_out:
            outgoing_refs = tuple(
                dict.fromkeys(
                    ref
                    for name in (resolution.names or (subject,))
                    for ref in tx.claims.list_refs(
                        subject=name, predicate=predicate, kind="fact" if probe else None, limit=CANDIDATE_CAP
                    )
                )
            )
            scan_hit = len(outgoing_refs) >= CANDIDATE_CAP
        if want_in:
            incoming_names = resolution.names or (lookup_subject,)
            collected: list[str] = []
            for name in incoming_names:
                refs = tx.claims.list_refs(value_text=name, predicate=predicate, limit=CANDIDATE_CAP)
                scan_hit = scan_hit or len(refs) >= CANDIDATE_CAP
                collected.extend(refs)
            incoming_refs = tuple(dict.fromkeys(collected))
        clipped, overflow = _clip_candidate_refs(outgoing_refs, incoming_refs, reserved=resolution.fence)
        loaded = _select_versions(tx, clipped, now)
        outgoing = {v.ref for v in loaded if v.ref in outgoing_refs and (not probe or v.payload["kind"] == "fact")}
        incoming = {v.ref for v in loaded if v.ref in incoming_refs and v.payload.get("value_text") in incoming_names}
        versions, truncated = _clip_versions_for_view(
            tuple(v for v in loaded if v.ref in outgoing or v.ref in incoming),
            max_items=payload["max_items"],
        )
        kept = {v.ref for v in versions}
        return _Selection(
            versions,
            resolution.scan_capped or scan_hit or overflow,
            truncated,
            outgoing=frozenset(outgoing & kept),
            incoming=frozenset(incoming & kept),
        )

    loaded = _load_view(storage, clock, context, lookup_subject, select, requested_subject=payload["subject"])
    if isinstance(loaded, _Unavailable):
        return _unavailable(payload, epoch=loaded.epoch, extra_gaps=loaded.gaps, entity=True)
    ambiguous = loaded.resolution.status == "ambiguous"
    statements: list[dict] = []
    for group, side in ((loaded.selection.outgoing, "outgoing"), (loaded.selection.incoming, "incoming")):
        members = [loaded.versions[ref] for ref in group if ref in loaded.versions]
        statements.extend(
            _claim_item(v, loaded.contexts.get(v.ref), entries=loaded.entries.get(v.ref), direction=side)
            for v in sorted(members, key=_sort_key)
        )
    gaps: list[str] = ["alias_ambiguous"] if ambiguous else []
    statements = withhold_secret_items(statements, gaps)
    if not statements and not ambiguous:
        gaps.append("consolidation_required")
    body = _candidate_body(
        {"action": payload["action"], "direction": direction, "statements": statements}, loaded, entity=True
    )
    if "candidate_groups" in body:
        body["candidate_groups"] = withhold_secret_items(body["candidate_groups"], gaps)
    result = _view_result(payload, loaded, lookup_subject, body, gaps)
    return _finish_view(
        result,
        schema="entity_view",
        max_items=payload["max_items"],
        budget_tokens=payload["budget_tokens"],
        item_lists=_entity_item_lists(result),
        alias_status=loaded.resolution.status,
        disputed_from=lambda: any(
            item["temporal_status"] == "disputed" for items in _entity_item_lists(result) for item in items
        ),
        ambiguity_fallback=_empty_ambiguity(
            payload, loaded, lookup_subject, {"action": payload["action"], "direction": direction, "statements": []}
        ),
    )
