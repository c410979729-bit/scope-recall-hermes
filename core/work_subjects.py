"""Resolve work subjects and check whether their current revisions can be retried."""

from __future__ import annotations

from ..contracts import ContractError
from .delete_storage import purge_work_parts


def _marks(values) -> str:
    return ",".join("?" for _ in values)


# --- where each work type's subject lives ----------------------------------
#
# Enqueue resolves the subject to the scope, project and branch a new row
# inherits; the retry paths ask whether that subject may still be worked on.


def _source_context(tx, ref: str, revision: int) -> tuple[str, str | None, str | None]:
    source = tx.source(ref, revision)
    if source is None:
        raise ContractError("SOURCE_MISSING")
    return source.scope_id, source.project_id, source.branch_id


def _claim_context(tx, ref: str, revision: int) -> tuple[str, str | None, str | None]:
    version = tx.claims.version(ref, revision)
    if version is None:
        raise ContractError("SOURCE_MISSING")
    tx.claims.require_target(version)
    return version.scope_id, version.project_id, version.branch_id


def _source_or_claim_context(tx, ref: str, revision: int) -> tuple[str, str | None, str | None]:
    # A claim is embeddable and projectable too; without this fallback derived
    # objects never reach the vector index and enter recall only by relation
    # expansion out of some event retrieved first.
    source = tx.source(ref, revision)
    if source is not None:
        return source.scope_id, source.project_id, source.branch_id
    return _claim_context(tx, ref, revision)


def _purge_context(tx, ref: str, revision: int) -> tuple[str, str | None, str | None]:
    _operation_id, scope_id = purge_work_parts(ref)
    tx._scope(scope_id)
    return scope_id, tx.context.project_id, tx.context.branch_id


def _candidate_context(tx, ref: str, revision: int) -> tuple[str, str | None, str | None]:
    row = (
        tx._check()
        .execute(
            """SELECT candidate_ref,scope_id,project_id,branch_id,state FROM candidate_evaluations e
           JOIN candidate_lifecycle l USING(candidate_ref,candidate_revision)
           WHERE evaluation_id=?""",
            (revision,),
        )
        .fetchone()
    )
    if row is None or "candidate:" + row["candidate_ref"] != ref or row["state"] != "queued":
        raise ContractError("SOURCE_MISSING", "candidate_evaluation")
    tx._scope(row["scope_id"])
    if (row["project_id"], row["branch_id"]) != (tx.context.project_id, tx.context.branch_id):
        raise ContractError("ACCESS_DENIED", "candidate_context")
    return row["scope_id"], row["project_id"], row["branch_id"]


_SUBJECT_CONTEXT = {
    "consolidate": _source_context,
    "embed": _source_or_claim_context,
    "rebuild_projection": _source_or_claim_context,
    "purge": _purge_context,
    "evaluate_candidate": _candidate_context,
}
ALLOWED_WORK_TYPES = frozenset(_SUBJECT_CONTEXT)


def _current_source_reason(tx, source) -> str | None:
    current = tx.sources.source_current(source.ref)
    if current is None or current.revision != source.revision:
        return "source_revision_stale"
    try:
        tx.claims.require_live_source(source.ref, source.revision)
    except ContractError:
        return "source_revision_stale"
    return None


def _source_retry_reason(tx, ref: str, revision: int, *, current_epoch: int | None = None) -> str | None:
    source = tx.source(ref, revision)
    if source is None:
        return "source_deleted_or_inaccessible"
    return _current_source_reason(tx, source)


def _projection_retry_reason(tx, ref: str, revision: int, *, current_epoch: int | None = None) -> str | None:
    source = tx.source(ref, revision)
    if source is not None:
        return _current_source_reason(tx, source)
    version = tx.claims.version(ref, revision)
    if version is None or tx.claims.current_revision(ref) != revision:
        return "claim_revision_stale_or_deleted"
    try:
        tx.claims.require_target(version)
    except ContractError:
        return "claim_denied"
    return None


def _embed_retry_reason(tx, ref: str, revision: int, *, current_epoch: int | None = None) -> str | None:
    """Return why the source or claim head for an embedding is no longer eligible."""
    if ref.startswith("claim-"):
        return _projection_retry_reason(tx, ref, revision, current_epoch=current_epoch)
    return _source_retry_reason(tx, ref, revision, current_epoch=current_epoch)


def _purge_retry_reason(tx, ref: str, revision: int, *, current_epoch: int | None = None) -> str | None:
    try:
        receipt = tx.deletions.receipt(purge_work_parts(ref)[0])
    except ContractError:
        receipt = None
    if receipt is None or receipt.get("mode") != "delete":
        return "delete_fence_missing"
    if current_epoch is not None and int(receipt.get("memory_epoch", current_epoch)) > current_epoch:
        return "memory_epoch_stale"
    return None


#: Why a failed row's subject may no longer be worked on.  Candidate
#: evaluations are absent on purpose: an evidence combination is evaluated at
#: most once, and new evidence makes a new row rather than a retry.
_RETRY_SUBJECT_REASON = {
    "consolidate": _source_retry_reason,
    "embed": _embed_retry_reason,
    "rebuild_projection": _projection_retry_reason,
    "purge": _purge_retry_reason,
}
#: The work types ``recover_transient_failures`` can reopen.  The planner wakes
#: only for these: a candidate evaluation failed for good was counted as due, and
#: one instance ran a pass every twenty seconds that recovered nothing.
AUTO_RECOVERABLE_WORK_TYPES = frozenset(_RETRY_SUBJECT_REASON)
