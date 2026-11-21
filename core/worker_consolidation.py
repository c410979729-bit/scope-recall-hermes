"""Consolidation batching, model envelopes and acceptance fences.

Owned by the worker drain; model calls stay outside SQLite transactions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Protocol

from ..contracts import ContractError, decode_payload, partition_consolidation_items, utc_instant, validate_payload
from . import private
from .adoption_work import adoption_checkpoint, process_adoption
from .candidate_lifecycle import candidate_model_subject
from .consolidate import (
    ConsolidationWorkFence,
    RejectedProposal,
    accept_consolidation,
    candidate_evaluation_messages,
    consolidation_messages,
    using_memory_instruction,
)
from .consolidation_chunks import source_chunk
from .consolidation_summary import resume_seed
from .entry_control import sources_paused, sources_paused_now
from .episodes import source_origin, source_watermark
from .evidence_question import DERIVATION_ROOT_ORIGINS, NO_DERIVATION_ROOT_REASON
from .evidence_quote import QuoteAnchors, align_claim_evidence
from .evidence_support import using_support
from .failure_policy import failure_category
from .semantic_proposals import prepare_publication
from .source_records import StoredSource
from .worker_outcomes import (
    Outcome,
    budget_left,
    deadline_result,
    derivation_changed,
    epoch_changed,
    finalize_work,
    hold_for_entry_state,
    hold_for_private_mode,
    mark_obsolete,
    model_failure,
    read_derivation_fence,
    work_result,
)


class ConsolidationModel(Protocol):
    def propose(
        self,
        sources: tuple[StoredSource, ...],
        *,
        episode_ref: str | None,
        remaining_seconds: float,
        validation_feedback: dict[str, str] | None = None,
    ) -> str: ...


def _fit_prompt_budget(
    episode_ref: str | None,
    batch: list[StoredSource],
    subject: tuple[str, int],
) -> tuple[list[StoredSource], list[StoredSource]]:
    """Keep the consolidation batch inside the serialized model-input budget.

    The subject source is always kept because the acceptance fence requires
    its coverage.  Other sources are added in episode order only while the
    exact prompt built by ``consolidation_messages`` still fits; deferred
    sources stay pending for a later bounded pass.  Without this the whole
    episode stalls: an over-budget batch raises INPUT_INVALID on every
    attempt and no work item ever completes.
    """
    kept = [stored for stored in batch if (stored.ref, stored.revision) == subject]
    if not kept:
        return list(batch), []
    deferred: list[StoredSource] = []
    for stored in batch:
        if (stored.ref, stored.revision) == subject:
            continue
        candidate = kept + [stored]
        try:
            consolidation_messages(tuple(candidate), episode_ref=episode_ref)
        except ContractError as exc:
            if exc.code == "INPUT_INVALID" and exc.field == "consolidation_input_budget":
                deferred.append(stored)
                continue
            raise
        kept.append(stored)
    order = {id(stored): index for index, stored in enumerate(batch)}
    kept.sort(key=lambda stored: order[id(stored)])
    return kept, deferred


def _episode_batch(tx, source: StoredSource, item, *, now: str):
    episode = tx.episodes.source_episode(source.ref, source.revision)
    if episode is None:
        return None, (source,), ()
    prior = (
        tx._check()
        .execute(
            """SELECT v.processed_sequence,v.resume_json,ee.sequence FROM episode_versions v
           JOIN episode_events ee ON ee.episode_id=v.episode_id
           WHERE v.episode_id=? AND v.revision=? AND ee.source_ref=? AND ee.source_revision=?""",
            (episode.ref, episode.revision, source.ref, source.revision),
        )
        .fetchone()
    )
    processed_sequence = prior["processed_sequence"] if prior and prior["resume_json"] else 0
    if prior and prior["sequence"] <= processed_sequence:
        return episode.ref, (), ()
    batch_rows = (
        tx._check()
        .execute(
            """SELECT ee.sequence,ee.source_ref,ee.source_revision,w.lease_token,w.work_id
           FROM episode_events ee JOIN work_items w ON w.work_type='consolidate'
             AND w.subject_ref=ee.source_ref AND w.subject_revision=ee.source_revision
           WHERE ee.episode_id=? AND ee.sequence>? AND w.scope_id=?
             AND w.project_id IS ? AND w.branch_id IS ?
             AND (w.work_id=? OR (w.state='pending' AND w.available_at<=? AND w.consolidation_offset=0))
           ORDER BY (w.work_id=?) DESC,ee.sequence LIMIT 32""",
            (
                episode.ref,
                processed_sequence,
                source.scope_id,
                source.project_id,
                source.branch_id,
                item.work_id,
                now,
                item.work_id,
            ),
        )
        .fetchall()
    )
    batch, pending = [], []
    lease_tokens: dict[tuple[str, int], int] = {}
    for row in sorted(batch_rows, key=lambda row: row["sequence"]):
        stored = tx.source(row["source_ref"], row["source_revision"])
        if stored is None:
            continue
        try:
            tx.claims.require_live_source(stored.ref, stored.revision)
        except ContractError:
            continue
        batch.append(stored)
        lease_tokens[(stored.ref, stored.revision)] = row["lease_token"]
        if row["work_id"] != item.work_id:
            pending.append((stored.ref, stored.revision, row["lease_token"]))
    batch, deferred = _fit_prompt_budget(episode.ref, batch, (source.ref, source.revision))
    if deferred:
        pending_keys = {(ref, revision) for ref, revision, _token in pending}
        for stored in deferred:
            key = (stored.ref, stored.revision)
            if key not in pending_keys:
                pending.append((stored.ref, stored.revision, lease_tokens[key]))
    return episode.ref, tuple(batch), tuple(pending)


def _root_only_sources(tx, sources: tuple[StoredSource, ...]) -> tuple[StoredSource, ...]:
    roots: list[StoredSource] = []
    seen: set[tuple[str, int]] = set()
    for source in sources:
        origin = source_origin(source)
        if origin not in DERIVATION_ROOT_ORIGINS:
            continue
        key = (source.ref, source.revision)
        if key in seen:
            continue
        # A partial/interrupted capture remains durable evidence with gaps,
        # but it is not a root from which the model may derive durable claims.
        if source.capture_gaps or source.event.get("capture_state") != "complete":
            continue
        seen.add(key)
        try:
            tx.claims.require_live_source(source.ref, source.revision)
        except ContractError:
            continue
        roots.append(source)
    return tuple(roots)


def _requote(claim: dict, span: dict, content: str) -> bool:
    """A quote of a serialized JSON value, re-escaped once (with the claim's fields that held it) when exactly that
    escaped span is stored."""
    quote = span["quote"]
    changed = False
    if quote not in content:
        # The model may quote the decoded value of serialized JSON.
        # Re-escape once only when that exact, unique stored span
        # exists. No fuzzy matching, truncation or text generation.
        encoded = json.dumps(quote, ensure_ascii=False)[1:-1]
        if encoded != quote and content.count(encoded) == 1 and len(encoded) <= 4096:
            span["quote"] = encoded
            for field in ("subject", "predicate", "value_text"):
                fragment = claim.get(field)
                if isinstance(fragment, str) and fragment in quote and fragment not in encoded:
                    claim[field] = json.dumps(fragment, ensure_ascii=False)[1:-1]
                    changed = True
    return changed


def _undecorate_claim(claim: dict, source_text: dict) -> None:
    """Normalize optional transport decoration before checking item shape."""
    for kind_field in ("procedure", "intention", "alias"):
        if claim.get("kind") != kind_field and claim.get(kind_field) in (None, {}):
            claim.pop(kind_field, None)
    for time_field in ("valid_from", "valid_to", "occurred_at"):
        if time_field in claim:
            claim[time_field] = utc_instant(claim[time_field])
    spans = claim.get("evidence_spans")
    if not isinstance(spans, list):
        return
    for span in spans:
        if not isinstance(span, dict):
            continue
        if span.get("location") is None:
            span.pop("location", None)
        _bare_source_ref(span, source_text)
        start, end = span.get("start"), span.get("end")
        if source_text and (type(start) is not int or type(end) is not int or not 0 <= start < end):
            # Unverified positions are hints. Normalize only complete,
            # plausible pairs before validating shape and finding the anchor.
            span.pop("start", None)
            span.pop("end", None)


def _bare_source_ref(span: dict, source_text: dict) -> None:
    """A span's ``source_ref`` written in the combined ``ref@revision`` form, with that revision beside it in
    ``source_revision`` and that version among the sources the model was given, names nothing but that version: the
    suffix is taken off.  Left on, the pair would read ``ref@1@1`` and the result be refused as citing an undeclared
    source.  Any other ``@`` is left for the schema to refuse by name."""
    ref, revision = span.get("source_ref"), span.get("source_revision")
    if type(ref) is not str or type(revision) is not int:
        return
    bare, separator, suffix = ref.rpartition("@")
    if separator and suffix == str(revision) and (bare, revision) in source_text:
        span["source_ref"] = bare


def _restore_request_fields(value, sources, episode_ref, candidate):
    """Rebuild request constants; selected evidence and values remain model choices."""
    # Missing/null arrays express no proposals; other types must reach validation.
    for field in ("claim_proposals", "resume_proposals", "reference_proposals"):
        if value.get(field) is None:
            value[field] = []
    value["protocol_version"] = "1.1"
    if not sources:
        return
    refs = [f"{s.ref}@{s.revision}" for s in sources]
    value["source_refs"] = refs
    if candidate is not None:
        claims = value.get("claim_proposals")
        for claim in claims if isinstance(claims, list) else ():
            if isinstance(claim, dict):
                claim.update(
                    kind=candidate.payload["kind"],
                    subject=candidate_model_subject(candidate, sources),
                    predicate=candidate.payload["predicate"],
                )
    resumes = value.get("resume_proposals")
    watermark = source_watermark(refs)
    for resume in resumes if isinstance(resumes, list) else ():
        if isinstance(resume, dict):
            resume.update(episode_ref=episode_ref, evidence_refs=list(refs), source_watermark=watermark)


def decode_consolidation_result(
    raw: str, sources=(), *, rejected_items: list[RejectedProposal] | None = None, episode_ref=None, candidate=None
) -> dict:
    """Decode bounded model JSON and restore request-owned metadata.

    Ordinary worker batches may omit malformed claims and resumes, reporting
    each original index. Candidate evaluation keeps its single-verdict
    contract. Callers asking only for an envelope still get strict shape
    validation. Publication rechecks all accepted evidence against live data.
    """
    if type(raw) is not str:
        raise ValueError("consolidation_result_not_text")
    encoded = raw.strip()
    if encoded.startswith("```"):
        lines = encoded.splitlines()
        if len(lines) < 3 or lines[0].strip() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise ValueError("consolidation_result_fence")
        encoded = "\n".join(lines[1:-1]).strip()
    value = decode_payload(encoded)
    _restore_request_fields(value, sources, episode_ref, candidate)
    source_text = {(s.ref, s.revision): s.event["content"] for s in sources}
    available = {(s.ref, s.revision): s for s in sources}
    claims = value.get("claim_proposals")
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict):
                _undecorate_claim(claim, source_text)
    anchors = {key: QuoteAnchors(content) for key, content in source_text.items()}
    align = (
        partial(_align_model_claim, available=available, source_text=source_text, anchors=anchors)
        if available
        else None
    )
    if rejected_items is None or candidate is not None:
        value = validate_payload("consolidation_result", value, indexed_errors=True)
        if candidate is not None:
            _check_candidate_shape(value)
        if align is not None:
            changed = False
            for index, claim in enumerate(value["claim_proposals"]):
                if align(claim, index):
                    changed = True
            if changed:
                return validate_payload("consolidation_result", value, indexed_errors=True)
        return value
    malformed = partition_consolidation_items(value, align_claim=align)
    value["claim_proposals"] = _keep_model_items(value, "claim_proposals", "claim", malformed, rejected_items)
    value["resume_proposals"] = _keep_model_items(value, "resume_proposals", "summary", malformed, rejected_items)
    return value


def _check_candidate_shape(value: dict) -> None:
    """Limit a candidate verdict to one claim before it reaches publication."""
    if len(value["claim_proposals"]) > 1:
        raise ContractError("DERIVATION_INVALID", "candidate_proposal_count")
    for field in ("resume_proposals", "reference_proposals"):
        if value[field]:
            raise ContractError("INPUT_INVALID", field)


def _keep_model_items(value, collection, stage, malformed, rejected_items):
    """Keep valid items in input order and count each rejected item once."""
    kept = []
    for index, proposal in enumerate(value[collection]):
        error = malformed.get((collection, index))
        if error is None:
            kept.append(proposal)
        else:
            rejected_items.append(RejectedProposal(stage, error.code, error.field))
    return kept


def _align_model_claim(claim, index, *, available, source_text, anchors) -> bool:
    """Locate quotes in the source versions the model actually received."""
    changed = False
    for span in claim["evidence_spans"]:
        content = source_text.get((span["source_ref"], span["source_revision"]))
        if content is not None and _requote(claim, span, content):
            changed = True
    align_claim_evidence(claim, available, index=index, anchors=anchors)
    return changed


@dataclass(frozen=True)
class _ConsolidationBatch:
    """What one consolidation attempt works on, read in one snapshot: the work item's source (None when it is gone or
    no longer live), its episode, the batch and the sources left for a later one, the root-only sources the model
    reads, a long source's resume point, and the dependencies the result is fenced on."""

    source: Any = None
    episode_ref: str | None = None
    batch: tuple = ()
    pending_sources: tuple = ()
    roots: tuple = ()
    offset: int = 0
    seed: tuple = ()
    dependencies: Any = None


def _read_batch(tx, item, clock) -> _ConsolidationBatch:
    """The batch a consolidation attempt works on, in the read transaction that verified its lease."""
    source = tx.source(item.subject_ref, item.subject_revision)
    if source is not None:
        try:
            tx.claims.require_live_source(source.ref, source.revision)
        except ContractError:
            source = None
    if source is None:
        return _ConsolidationBatch()
    intervals = private.active_intervals(tx, clock.utc_now())
    seed = ()
    offset = tx.work.consolidation_offset(*item.lease, now=clock.utc_now())
    if offset:
        seed = resume_seed(tx, item.work_id)
        episode = tx.episodes.source_episode(source.ref, source.revision)
        episode_ref, batch, pending_sources = episode.ref if episode else None, (source,), ()
    else:
        episode_ref, batch, pending_sources = _episode_batch(tx, source, item, now=clock.utc_now())
    # A full interval may cover only some sources of an episode.  Leave those out
    # of this request; their own work items stay pending and will run when it ends.
    batch = private.filter_sources(intervals, batch)
    # Excluded entries keep their own queued work; another entry's source can
    # proceed without being coupled to a paused member of the same episode.
    batch = tuple(source for source in batch if not sources_paused(tx, (source,)))
    roots = _root_only_sources(tx, batch)
    # The dependencies come from the same snapshot as the batch itself.
    dependencies = read_derivation_fence(tx, scope_id=item.scope_id, sources=(source, *batch), episode_ref=episode_ref)
    return _ConsolidationBatch(source, episode_ref, batch, pending_sources, roots, offset, seed, dependencies)


def _complete_without_model(storage, clock, context, item, read: _ConsolidationBatch, started: float, budget: float):
    """Complete a batch with nothing for the model to read (no root-only source) without a model call, while the
    lease, every source and the dependencies still hold."""
    if budget_left(started, clock, budget) <= 0:
        return deadline_result(storage, context, item)
    with storage.write(context, remaining_seconds=budget_left(started, clock, budget)) as tx:
        now = clock.utc_now()
        if not tx.work._verify_lease(*item.lease, now=now):
            return work_result(tx.work.complete(*item.lease, now=now))
        try:
            tx.claims.require_live_source(item.subject_ref, item.subject_revision)
            for stored in read.batch:
                tx.claims.require_live_source(stored.ref, stored.revision)
        except ContractError:
            return mark_obsolete(tx, item, now)
        if derivation_changed(tx, read.dependencies) is not None:
            return epoch_changed(tx, item, now)
        return work_result(
            tx.work.complete_consolidation(
                *item.lease,
                now=now,
                covered_source_refs=frozenset(f"{s.ref}@{s.revision}" for s in read.batch)
                | {f"{read.source.ref}@{read.source.revision}"},
                pending_sources=read.pending_sources,
            )
        )


def _needs_chunk(roots, episode_ref, feedback, offset) -> bool:
    """Whether the source goes to the model a page at a time: it is already part-way through, or its messages exceed
    the input budget."""
    if offset:
        return True
    try:
        consolidation_messages(roots, episode_ref=episode_ref, validation_feedback=feedback)
    except ContractError as exc:
        if exc.code != "INPUT_INVALID" or exc.field != "consolidation_input_budget":
            raise
        return True
    return False


def _decode_refused(finish, exc: Exception, *, private_hold=None):
    """A model answer that does not decode: a bounded fresh attempt, with the clause or envelope that failed."""
    if isinstance(exc, ContractError) and exc.field == "private_mode" and private_hold is not None:
        return private_hold()
    if isinstance(exc, ContractError) and exc.code in {"SOURCE_MISSING", "VERSION_CONFLICT", "ACCESS_DENIED"}:
        return _accept_refused(finish, exc)
    detail = exc.field if isinstance(exc, ContractError) else "json_envelope"
    return Outcome(
        *finish(
            "retry",
            "derivation_invalid",
            error_detail=detail,
            stage="decode",
            validation_code=exc.code if isinstance(exc, ContractError) else "INPUT_INVALID",
        ),
        detail=detail,
    )


def _accept_refused(finish, exc: ContractError):
    """A decoded result the store refused to accept."""
    code = exc.code or "derivation_invalid"
    # Reject this result, but retain live work for a fresh bounded attempt:
    # something it was derived from changed during the call.
    if code == "VERSION_CONFLICT" and exc.field == "memory_epoch":
        return finish("retry", "memory_epoch_changed")
    if code in {"SOURCE_MISSING", "VERSION_CONFLICT", "ACCESS_DENIED"}:
        return finish("obsolete", "authority_revoked")
    recoverable = code in {"STORAGE_UNAVAILABLE", "DEADLINE_EXCEEDED", "DERIVATION_INVALID"}
    # Keep the clause that rejected the result. Without it a terminal
    # DERIVATION_INVALID cannot be told apart from any other, and diagnosing
    # one costs a full reproduction against live work.
    return Outcome(
        *finish("retry" if recoverable else "failed", code, error_detail=exc.field, stage="accept"),
        detail=exc.field,
    )


def _without_model(storage, clock, context, item, read, model, *, started, budget):
    """Settle unavailable or empty input before preparing a provider request."""
    if read.source is None:
        return finalize_work(
            storage, clock, context, item, "obsolete", "authority_revoked", started=started, budget=budget
        )
    if not read.roots:
        return _complete_without_model(storage, clock, context, item, read, started, budget)
    if model is None:
        return finalize_work(
            storage, clock, context, item, "retry", "model_unavailable", started=started, budget=budget
        )
    return None


def process_consolidate(
    storage,
    clock,
    context,
    item,
    *,
    model: ConsolidationModel | None,
    started: float,
    budget: float,
) -> tuple[str, str | None, str]:
    with storage.read(context, remaining_seconds=budget_left(started, clock, budget)) as tx:
        feedback = tx.work.derivation_feedback(item.work_id)
        if not tx.work._verify_lease(*item.lease, now=clock.utc_now()):
            state = tx.work.read_state(item.work_id) or "stale"
            return "stale", "authority_revoked" if state == "obsolete" else None, state
        checkpoint = adoption_checkpoint(tx, item.work_id)
        source = tx.source(item.subject_ref, item.subject_revision)
        # Old queues may hold work that admission no longer creates. Do not let
        # such a subject consolidate neighbouring roots on its behalf. A started
        # adoption has its own durable fences and must retain its continuation.
        rootless_source = (
            checkpoint is None and source is not None and source_origin(source) not in DERIVATION_ROOT_ORIGINS
        )
        instruction = _memory_instruction(tx, source.scope_id) if source is not None else None
        # The batch is sized with the instruction it will be sent with.
        with using_memory_instruction(instruction):
            read = _read_batch(tx, item, clock) if checkpoint is None and not rootless_source else None
        entry_paused = source is not None and sources_paused(tx, (source,))
    if entry_paused:
        return hold_for_entry_state(storage, clock, context, item, started=started, budget=budget)
    if checkpoint is not None:
        return process_adoption(storage, clock, context, item, checkpoint, model, started=started, budget=budget)
    if rootless_source:
        return finalize_work(
            storage, clock, context, item, "obsolete", NO_DERIVATION_ROOT_REASON, started=started, budget=budget
        )
    with using_memory_instruction(instruction):
        return _process_ordinary(
            storage, clock, context, item, (read, feedback), model=model, started=started, budget=budget
        )


def _memory_instruction(tx, scope_id: str) -> str | None:
    """The scope's latest owner memory instruction, read in the snapshot its batch is read in."""
    row = (
        tx._check()
        .execute(
            "SELECT text FROM scope_memory_instructions WHERE scope_id=? ORDER BY version DESC LIMIT 1", (scope_id,)
        )
        .fetchone()
    )
    return row[0] if row else None


def _process_ordinary(storage, clock, context, item, prepared, *, model, started, budget):
    """Run ordinary consolidation independently of any pending adoption stage.

    The caller binds the scope's memory instruction, so a long source's pages
    are sized with it as well as the batch."""
    read, feedback = prepared
    finish = partial(finalize_work, storage, clock, context, item, started=started, budget=budget)
    # Do not open a write transaction while the read transaction above is
    # still active.  SQLite's reader lock otherwise turns an obsolete source
    # into a spurious "database is locked" failure.
    source, episode_ref = read.source, read.episode_ref
    refused = _without_model(storage, clock, context, item, read, model, started=started, budget=budget)
    if refused is not None:
        return refused
    assert model is not None
    roots, batch, pending_sources, chunk = read.roots, read.batch, read.pending_sources, None
    try:
        if _needs_chunk(roots, episode_ref, feedback, read.offset):
            page, chunk = source_chunk(
                source,
                read.offset,
                formatter=partial(consolidation_messages, validation_feedback=feedback),
                episode_ref=episode_ref,
                resume_seed=read.seed,
            )
            roots, batch, pending_sources = (page,), (source,), ()
        allowed_refs = frozenset(f"{stored.ref}@{stored.revision}" for stored in roots)
        repair = {"validation_feedback": feedback} if feedback is not None else {}
        if private.outbound_blocked(storage, clock, context, sources=roots):
            return hold_for_private_mode(storage, clock, context, item, started=started, budget=budget)
        if sources_paused_now(storage, context, roots, remaining_seconds=budget_left(started, clock, budget)):
            return hold_for_entry_state(storage, clock, context, item, started=started, budget=budget)
        with private.using_outbound_check(lambda: private.outbound_blocked(storage, clock, context, sources=roots)):
            raw = model.propose(
                roots, episode_ref=episode_ref, remaining_seconds=budget_left(started, clock, budget), **repair
            )
    except ContractError as exc:
        # A probabilistic model occasionally emits an invalid derivation; give
        # the work item a bounded fresh attempt instead of killing it on the
        # first bad generation (attempts stay capped by MAX_RECOVERABLE_ATTEMPTS).
        return finish("retry", exc.code or "model_unavailable", error_detail=exc.field, stage="prepare_or_model")
    except Exception as exc:
        return finish(*model_failure(exc))
    try:
        rejected_items: list[RejectedProposal] = []
        value = decode_consolidation_result(raw, roots, rejected_items=rejected_items, episode_ref=episode_ref)
        verdicts = prepare_publication(
            storage,
            context,
            value,
            roots,
            getattr(model, "judge", None),
            remaining_seconds=budget_left(started, clock, budget),
            clock=clock,
        )
    except (ContractError, ValueError, TypeError, json.JSONDecodeError) as exc:
        # A suppression replay can block support preparation before publication's
        # dependency check. Missing sources remain an obsolete authority fence.
        if isinstance(exc, ContractError) and exc.code == "ACCESS_DENIED":
            with storage.read(context, remaining_seconds=budget_left(started, clock, budget)) as tx:
                if derivation_changed(tx, read.dependencies) is not None:
                    exc = ContractError("VERSION_CONFLICT", "memory_epoch")
        return _decode_refused(
            finish,
            exc,
            private_hold=partial(hold_for_private_mode, storage, clock, context, item, started=started, budget=budget),
        )
    fence = ConsolidationWorkFence(
        item.work_id,
        item.lease_token,
        item.lease_owner,
        item.subject_ref,
        item.subject_revision,
        read.dependencies.only_sources((source, *batch)),
        allowed_refs,
        skipped_source_refs=frozenset(f"{s.ref}@{s.revision}" for s in batch) - allowed_refs,
        pending_sources=pending_sources,
        chunk=chunk,
        rejected_items=tuple(rejected_items),
    )
    try:
        with using_support(verdicts):
            accept_consolidation(
                storage,
                clock,
                # A durable work item runs outside the originating host session.
                # Use the session from its freshly read source, never from model
                # output, so current_human can validate that session's actual
                # latest evidence. Scope, project and the work/dependency fence
                # remain unchanged; stale or unrelated human evidence still fails.
                replace(context, session_id=source.session_id, recent_messages=()),
                value,
                scope_id=item.scope_id,
                remaining_seconds=budget_left(started, clock, budget),
                work_fence=fence,
            )
    except ContractError as exc:
        return _accept_refused(finish, exc)
    return _consolidation_outcome(
        storage, context, item, verdicts, chunk, remaining_seconds=budget_left(started, clock, budget)
    )


def _consolidation_outcome(storage, context, item, verdicts, chunk, *, remaining_seconds):
    """Reflect the durable row when ordinary work handed off to adoption stages."""
    support_failure = next((v.failure_code for v in verdicts if v.failure_code), None)
    if chunk is not None and not chunk.final:
        return "deferred", support_failure, "pending"
    with storage.read(context, remaining_seconds=remaining_seconds) as tx:
        if tx.work.read_state(item.work_id) == "pending":
            return "deferred", support_failure, "pending"
    disposition = "deferred" if support_failure and failure_category(support_failure) == "capability" else "completed"
    return disposition, support_failure, "done"


def build_consolidation_model(port) -> ConsolidationModel | None:
    if port is None:
        return None

    class Adapter:
        def __init__(self):
            self._used = 0

        @property
        def remaining_calls(self):
            """Expose a pass allowance without renewing or replacing the model port."""
            if hasattr(port, "remaining_calls"):
                return port.remaining_calls
            limit = getattr(port, "max_calls_per_pass", None)
            return max(0, limit - self._used) if limit is not None else None

        def _call(self, messages, remaining_seconds):
            self._used += 1
            return port.propose(messages, remaining_seconds=remaining_seconds)

        def judge(self, messages, *, remaining_seconds=1.0):
            return self._call(messages, remaining_seconds)

        def propose(self, sources, *, episode_ref=None, remaining_seconds=1.0, validation_feedback=None):
            messages = consolidation_messages(sources, episode_ref=episode_ref, validation_feedback=validation_feedback)
            return self._call(messages, remaining_seconds)

        def evaluate_candidate(self, candidate, sources, *, remaining_seconds=1.0, validation_feedback=None):
            messages = candidate_evaluation_messages(candidate, sources, validation_feedback=validation_feedback)
            return self._call(messages, remaining_seconds)

    return Adapter()
