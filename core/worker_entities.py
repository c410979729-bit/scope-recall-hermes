"""Entity projection processor using the existing work lease and model budget port."""

from __future__ import annotations

import json
from datetime import datetime
from functools import partial

from ..contracts import ContractError, decode_payload
from . import private
from .aliases import ENTITY_KINDS
from .claims import canonical_time, evidence_refs, valid_at
from .entity_names import (
    backfill_names,
    checked,
    index_claim,
    ineligible,
    owns_pending_names,
    pending_names,
    release_names,
    reserve_names,
)
from .entity_resolution import entity_messages, entity_name_evidenced, live_entity, publish_entity_merge
from .entry_control import entry_fence, paused_since
from .evidence_support import SupportVerdict
from .failure_retry import validation_feedback
from .injection_guard import blocked_refs
from .retrieval import SearchContext, SearchLimits
from .semantic_proposals import _publication_blocked, merge_candidates, validate_judgement
from .worker_outcomes import (
    Outcome,
    budget_left,
    deadline_result,
    finalize_work,
    hold_for_entry_state,
    hold_for_private_mode,
    model_failure,
    record_error_detail,
    stale_result,
    work_result,
)

ENTITY_NAME_LEASED = "entity_name_leased"


def defer_held_entity(tx, item, now):
    """Retain the existing projection job while a live fact's evidence is withheld.

    Safety holds are reversible. They must not withdraw names or acknowledge
    entity resolution; deletion fences continue to retire truly absent facts.
    """
    from .visibility import deletion_allows

    if deletion_allows(tx, "claim", item.subject_ref) and blocked_refs(tx, "claim", (item.subject_ref,)):
        return work_result(
            tx.work.defer_without_attempt(*item.lease, now=now, seconds=1, error_code="source_safety_hold"),
            error_code="source_safety_hold",
        )
    return None


def _vector_hits(storage, clock, context, source, seconds):
    """Ask only the existing scoped search interface; no vector storage is owned here."""
    port = storage.entity_vectors() if storage.entity_vectors is not None else None
    if port is None:
        return (), "entity_vector_unavailable"
    search = SearchContext(
        query=source.payload["subject"],
        mode="current",
        as_of=None,
        focus_refs=(),
        limits=SearchLimits(),
        deadline=clock.monotonic() + seconds,
        now=clock.utc_now(),
        trusted_context=context,
    )
    private.require_outbound()
    try:
        return tuple(port.search(search, limit=16, remaining_seconds=seconds)), None
    except (RuntimeError, OSError, TimeoutError):
        return (), "entity_vector_unavailable"


def _entity_outbound_blocked(storage, clock, context, item, names, deadline, evidence, entries):
    """Fence the reservation and evidence before asking an external port."""
    sources, targets = evidence
    try:
        blocked = _publication_blocked(storage, context, sources, targets, clock)
    except ContractError as exc:
        if len(targets) == 1 or exc.code not in {"SOURCE_MISSING", "ACCESS_DENIED"}:
            raise
        # A candidate disappearing after nomination invalidates this attempt, not the source fact.
        with storage.read(context, remaining_seconds=1.0) as tx:
            live_entity(tx, item.subject_ref, item.subject_revision, clock.utc_now())
        raise ContractError("VERSION_CONFLICT", "entity_candidate") from exc
    if blocked:
        return True
    # Evidence reads may consume the last of either clock. Check them at handoff.
    with storage.read(context, remaining_seconds=1.0) as tx:
        if not owns_pending_names(tx, item, names):
            raise ContractError("VERSION_CONFLICT", "entity_names")
        if paused_since(tx, entries):
            raise ContractError("ACCESS_DENIED", "entry_state:paused")
        if not tx.work._verify_lease(*item.lease, now=clock.utc_now()):
            raise ContractError("VERSION_CONFLICT", "entity_lease")
    if deadline <= clock.monotonic():
        raise ContractError("DEADLINE_EXCEEDED")
    return False


def _decode_decision(raw, candidates):
    """Accept exactly one valid entity verdict; transport and refusals are handled by the model port."""
    if type(raw) is not str:
        raise ContractError("DERIVATION_INVALID", "model_output_content")
    try:
        parsed = decode_payload(raw)
    except ContractError as exc:
        raise ContractError("DERIVATION_INVALID", exc.field) from None
    answer = validate_judgement(parsed, candidates)
    support, action = answer["support"], answer["action"]
    if action == "update" or (action == "duplicate" and support != "supported"):
        raise ContractError("DERIVATION_INVALID", "entity_decision")
    return SupportVerdict("", support, answer["target_ref"], answer["target_revision"], action)


def _judge(storage, clock, context, item, model, prepared, deadline, *, evidenced_names):
    """Nominate and judge reserved names outside the writer transaction."""
    source, sources, names, entries = prepared
    targets = ((source.ref, source.revision),)
    check = partial(_entity_outbound_blocked, storage, clock, context, item, names, deadline)
    with private.using_outbound_check(partial(check, (sources, targets), entries)):
        hits, gap = _vector_hits(storage, clock, context, source, max(0.001, deadline - clock.monotonic()))
    with storage.read(context, remaining_seconds=max(0.001, deadline - clock.monotonic())) as tx:
        candidates = merge_candidates(
            tx, source.payload, source.scope_id, now=clock.utc_now(), entity=source, vector_hits=hits
        )
        if source.payload.get("entity_names"):
            candidates = (dict(ref=source.ref, revision=source.revision, payload=source.payload), *candidates)[:16]
        eligible = []
        for entry in candidates:
            try:
                _, evidence = live_entity(tx, entry["ref"], entry["revision"], clock.utc_now())
            except ContractError as exc:
                if entry["ref"] != source.ref and exc.code in {"SOURCE_MISSING", "ACCESS_DENIED", "VERSION_CONFLICT"}:
                    continue
                raise
            eligible.append(entry)
            sources += evidence
        candidates = tuple(eligible)
        for key, value in entry_fence(tx, sources).items():
            entries.setdefault(key, value)
    if not candidates:
        return None, gap
    targets += tuple((entry["ref"], entry["revision"]) for entry in candidates)
    if deadline <= clock.monotonic():
        raise ContractError("DEADLINE_EXCEEDED")
    with private.using_outbound_check(partial(check, (sources, targets), entries)):
        private.require_outbound()
        raw = model.judge(
            entity_messages(
                source, tuple({(s.ref, s.revision): s for s in sources}.values()), candidates, names=evidenced_names
            ),
            remaining_seconds=max(0.001, deadline - clock.monotonic()),
        )
    return _decode_decision(raw, candidates), gap


def _entity_time_outcome(tx, item, source, now):
    """Defer a future fact until its start and retire only the projection of an expired fact."""
    if source.current_revision != item.subject_revision or valid_at(source, now, historical=False):
        return None
    start, instant = canonical_time(source.valid_from), canonical_time(now)
    if start is not None and instant is not None and start > instant:
        delay = (datetime.fromisoformat(start) - datetime.fromisoformat(instant)).total_seconds()
        code = "entity_not_yet_valid"
        return work_result(
            tx.work.defer_without_attempt(*item.lease, now=now, error_code=code, seconds=delay), error_code=code
        )
    raise ContractError("ACCESS_DENIED", "entity_interval_ended")


def _prepare(storage, clock, context, item, seconds):
    """Prepare the names under the writer lease, or return an early queue outcome."""
    with storage.write(context, remaining_seconds=seconds) as tx:
        if not tx.work._verify_lease(*item.lease, now=clock.utc_now()):
            return None, stale_result(tx, item)
        source = tx.claims.version(item.subject_ref, item.subject_revision)
        if source is None:
            raise ContractError("SOURCE_MISSING", "entity")
        sources = tuple(tx.claims.live_sources(evidence_refs(source.payload), source.scope_id))
        entries = entry_fence(tx, sources)
        if paused_since(tx, entries):
            raise ContractError("ACCESS_DENIED", "entry_state:paused")
        interval = _entity_time_outcome(tx, item, source, clock.utc_now())
        if interval is not None:
            return None, interval
        index_claim(tx, source)
        if not backfill_names(tx, source.scope_id):
            code = "entity_index_pending"
            mutation = tx.work.defer_without_attempt(*item.lease, now=clock.utc_now(), error_code=code, seconds=0)
            return None, work_result(mutation, error_code=code)
        names = pending_names(tx, source)
        if names:
            source, sources = live_entity(tx, item.subject_ref, item.subject_revision, clock.utc_now())
            now = clock.utc_now()
            wait = reserve_names(tx, item, now)
            if wait:
                # The same pass can spend its time on other work until this lease expires.
                mutation = tx.work.defer_without_attempt(
                    *item.lease, now=now, error_code=ENTITY_NAME_LEASED, seconds=wait
                )
                return None, work_result(mutation, error_code=ENTITY_NAME_LEASED)
        return (source, sources, names, entries), None


def _retire_ineligible(storage, clock, context, item, seconds):
    """Exclude the current mention only while this worker can still act for it."""

    def retire(tx, now):
        deferred = defer_held_entity(tx, item, now)
        if deferred is not None:
            return deferred
        ineligible(tx, item.subject_ref, item.subject_revision)
        tx._check(write=True).execute(
            "UPDATE claim_versions SET entity_resolved=1 WHERE claim_id=? AND revision=?",
            (item.subject_ref, item.subject_revision),
        )
        return _entity_write_result(
            tx.work.mark_obsolete(*item.lease, now=clock.utc_now()), error_code="authority_revoked"
        )

    return _write_entity_outcome(storage, clock, context, item, seconds, retire)


class _StaleEntityWrite(Exception):
    """Abort an entity write when its final work transition loses the lease."""

    def __init__(self, mutation):
        self.outcome = work_result(mutation)
        super().__init__(self.outcome)


def _entity_write_result(mutation, *, error_code=None):
    """A stale transition must escape the storage context to roll back earlier writes."""
    if mutation.disposition == "stale":
        raise _StaleEntityWrite(mutation)
    return work_result(mutation, error_code=error_code)


def _write_entity_outcome(storage, clock, context, item, seconds, apply):
    """Run the derived writes and the work transition within one lease-fenced transaction."""
    if seconds <= 0:
        return deadline_result(storage, context, item)
    try:
        with storage.write(context, remaining_seconds=seconds) as tx:
            now = clock.utc_now()
            if not tx.work._verify_lease(*item.lease, now=now):
                return stale_result(tx, item)
            return apply(tx, now)
    except _StaleEntityWrite as exc:
        return exc.outcome


def _fail_entity(tx, now, *, clock, item, exc, stage):
    """Keep an actionable field and free this lease's pending names in one terminal outcome."""
    code = exc.code.lower()
    field = validation_feedback(exc.code, exc.field)["field"]
    record_error_detail(
        tx._check(write=True),
        item.work_id,
        item.lease_token,
        now=now,
        stage=stage,
        code=exc.code,
        field=field,
        retain=True,
    )
    release_names(tx, item)
    mutation = tx.work.fail(*item.lease, error_code=code, now=clock.utc_now(), recoverable=False)
    return Outcome(*_entity_write_result(mutation, error_code=code), detail=field)


def _fail_invalid_projection(storage, clock, context, item, exc, stage, *, started, budget):
    """Record invalid work only if the failed attempt still owns the live lease."""
    apply = partial(_fail_entity, clock=clock, item=item, exc=exc, stage=stage)
    return _write_entity_outcome(storage, clock, context, item, budget_left(started, clock, budget), apply)


def process_entity_projection(storage, clock, context, item, *, model, started: float, budget: float):
    """Finish one fact's identity check atomically with its projection acknowledgement."""
    finish = partial(finalize_work, storage, clock, context, item, started=started, budget=budget)
    stage = "entity_prepare"
    try:
        prepared, early = _prepare(storage, clock, context, item, budget_left(started, clock, budget))
        if early is not None:
            return early
        assert prepared is not None
        source, _sources, names, entries = prepared
        evidenced_names = tuple(name for name in names if entity_name_evidenced(source, name))
        stage = "entity_judge"
        decision, gap = (
            _judge(storage, clock, context, item, model, prepared, started + budget, evidenced_names=evidenced_names)
            if evidenced_names
            else (None, None)
        )
        stage = "entity_publish"
        publication = (source, decision, names, evidenced_names, gap, entries)
        return _publish(storage, clock, context, item, publication, started, budget)
    except ContractError as exc:
        if exc.field == "entry_state:paused":
            return hold_for_entry_state(storage, clock, context, item, started=started, budget=budget)
        if exc.field == "private_mode":
            return hold_for_private_mode(storage, clock, context, item, started=started, budget=budget)
        if exc.code in {"SOURCE_MISSING", "ACCESS_DENIED"}:
            return _retire_ineligible(storage, clock, context, item, budget_left(started, clock, budget))
        if exc.code in {"DERIVATION_INVALID", "INPUT_INVALID"}:
            return _fail_invalid_projection(storage, clock, context, item, exc, stage, started=started, budget=budget)
        code = "memory_epoch_changed" if exc.code == "VERSION_CONFLICT" else exc.code.lower()
        return finish("retry", code, stage=stage, error_detail=exc.field)
    except (RuntimeError, OSError, ValueError, TypeError) as exc:
        return finish(*model_failure(exc), stage=stage)


def _publish(storage, clock, context, item, prepared, started, budget):
    """Commit a prepared judgement only while this projection owns its work lease."""
    apply = partial(_apply_judgement, clock=clock, item=item, prepared=prepared)
    return _write_entity_outcome(storage, clock, context, item, budget_left(started, clock, budget), apply)


def _apply_judgement(tx, now, *, clock, item, prepared):
    """Recheck the lease and every endpoint before writing any alias or completion bit."""
    source, decision, names, evidenced_names, gap, entries = prepared
    if paused_since(tx, entries):
        return work_result(
            tx.work.defer_without_attempt(*item.lease, now=clock.utc_now(), error_code="entry_paused", seconds=60),
            error_code="entry_paused",
        )
    if names:
        if not owns_pending_names(tx, item, names):
            raise ContractError("VERSION_CONFLICT", "entity_names")
        live_entity(tx, source.ref, source.revision, now)
    else:
        current = tx.claims.version(source.ref, source.revision)
        if current is None or current.current_revision != source.revision or current.state != "active":
            raise ContractError("VERSION_CONFLICT", "entity")
        if pending_names(tx, current):
            raise ContractError("VERSION_CONFLICT", "entity_names")
    if decision is not None and decision.action == "duplicate":
        target = _decided_target(tx, decision, now)
        for name in evidenced_names:
            publish_entity_merge(tx, source, target, decision, now=now, name=name)
    else:
        for name in evidenced_names:
            checked(tx, source, name)
    if evidenced_names != names:
        # Unsupported extracted names stay unresolved; they must not roll back a valid judgement of other names.
        exc = ContractError("DERIVATION_INVALID", "entity_name")
        return _fail_entity(tx, now, clock=clock, item=item, exc=exc, stage="entity_prepare")
    tx._check(write=True).execute(
        "UPDATE claim_versions SET entity_resolved=1 WHERE claim_id=? AND revision=?", (source.ref, source.revision)
    )
    if gap:
        record_error_detail(tx._check(write=True), item.work_id, item.lease_token, now=now, stage="entity", code=gap)
    return _entity_write_result(tx.work.complete(*item.lease, now=clock.utc_now()))


def _decided_target(tx, decision, now):
    """A selected target changing after judgement makes the verdict stale, with the source still eligible."""
    try:
        return live_entity(tx, decision.target_ref, decision.target_revision, now)[0]
    except ContractError as exc:
        if exc.code in {"SOURCE_MISSING", "ACCESS_DENIED"} and exc.field != "private_mode":
            raise ContractError("VERSION_CONFLICT", "entity_candidate") from exc
        raise


def needs_entity_projection(tx, item) -> bool:
    """Identify unresolved live fact heads without loading conversation bodies."""
    row = (
        tx._check()
        .execute(
            "SELECT 1 FROM claims c JOIN claim_versions v ON v.claim_id=c.claim_id AND v.revision=c.current_revision "
            "WHERE c.claim_id=? AND v.revision=? AND c.kind IN (SELECT value FROM json_each(?)) AND "
            "v.state='active' AND v.entity_resolved=0",
            (item.subject_ref, item.subject_revision, json.dumps(ENTITY_KINDS)),
        )
        .fetchone()
    )
    return row is not None
