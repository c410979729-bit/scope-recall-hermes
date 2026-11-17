"""Model revision receipts under the existing scoped SQLite transaction owner."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import TYPE_CHECKING

from ..contracts import ContractError
from . import private
from .claim_storage import parse_source_ref
from .claims import evidence_refs
from .delete_storage import canonical

if TYPE_CHECKING:
    from .storage import Transaction


def _digest(value) -> str:
    """Hash canonical structured metadata without storing its human-readable content."""
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _bare_ref(value: str) -> None:
    """Reject malformed tokens before hashing; never turn them into usable references."""
    if not value or any(char.isspace() for char in value) or "@" in value:
        raise ContractError("INPUT_INVALID", "ref")


class ModelRevisionReceipts:
    """Recover one native occurrence's revision result after checking its live authority.

    The caller owns commit and rollback. No message text, replacement value,
    runtime callback or per-attempt API/tool identifier is persisted here.
    """

    def __init__(self, tx: Transaction, clock, request: dict, request_id: str) -> None:
        """Validate the explicit native request and derive its immutable lookup digests."""
        self.tx, self.clock, self.request = tx, clock, request
        if type(request_id) is not str or not request_id.strip() or len(request_id) > 100:
            raise ContractError("INPUT_INVALID", "request_id")
        _bare_ref(request["target_ref"])
        for token in request["source_evidence_refs"]:
            _bare_ref(parse_source_ref(token)[0])
        ctx = tx.context
        proof = ctx.request_message
        if proof is None or not proof.occurrence:
            raise ContractError("ACCESS_DENIED", "request_message_binding")
        # The native witness is checked independently of payload equality.
        # Ordinary revision authority enforces evidence membership on first execution.
        tx.claims.require_request_message()
        self.proof = proof
        self.key = _digest(
            [
                ctx.binding.installation_id,
                ctx.binding.agent_id,
                ctx.entry_id,
                ctx.session_id,
                proof.human_principal.kind,
                proof.human_principal.resolution,
                proof.human_principal.principal_ref,
                proof.scope_id,
                ctx.project_id,
                ctx.branch_id,
                proof.occurrence,
                proof.source_ref,
                proof.source_revision,
                request_id,
            ]
        )
        self.payload_digest = _digest(request)

    def _version(self, ref: str, revision: int | None = None):
        """Require a visible result version or target head in this exact write audience."""
        versions = self.tx.claims.versions(ref)
        head = next((v for v in versions if v.revision == (revision or v.current_revision)), None)
        if head is None or head.suppressed:
            raise ContractError("SOURCE_MISSING")
        self.tx.claims.require_target(head)
        return head

    def _visible(self, versions) -> None:
        """Apply live source, deletion and private-write rules before releasing a result."""
        tx, ctx = self.tx, self.tx.context
        intervals = [dict(rule, mode="full") for rule in private.active_intervals(tx, self.clock.utc_now())]
        if any(private.claim_blocked(tx, intervals, head) for head in versions):
            raise ContractError("ACCESS_DENIED", "private_mode")
        sources = tx.claims.live_sources(self.request["source_evidence_refs"], self.proof.scope_id)
        if private.matching_intervals(tx, ctx, now=self.clock.utc_now()) or any(
            source.suppressed or private.source_blocked(intervals, source) for source in sources
        ):
            raise ContractError("ACCESS_DENIED", "private_mode")
        for head in versions:
            tx.claims.live_sources(evidence_refs(head.payload), head.scope_id)

    def replay(self):
        """Return persisted result columns only while both target and result remain unchanged."""
        row = (
            self.tx._check()
            .execute("SELECT * FROM model_revision_receipts WHERE request_key=?", (self.key,))
            .fetchone()
        )
        if row is None:
            return None
        if row["payload_sha256"] != self.payload_digest:
            raise ContractError("VERSION_CONFLICT", "request_id")
        heads = [self._version(row["target_ref"]), self._version(row["result_ref"], row["result_revision"])]
        for name, head in zip(("target", "result"), heads, strict=True):
            if head.revision != row[name + "_revision"] or _digest(asdict(head)) != row[name + "_sha256"]:
                raise ContractError("VERSION_CONFLICT", "expected_revision")
        self._visible(heads)
        return row

    def record(self, item, epoch: int) -> None:
        """Persist the mutation's exact response in its caller's uncommitted write transaction."""
        target, result = self._version(self.request["target_ref"]), self._version(item.ref, item.revision)
        self._visible((target, result))
        self.tx._check(write=True).execute(
            """INSERT INTO model_revision_receipts
            (request_key,payload_sha256,scope_id,source_ref,source_revision,target_ref,target_revision,target_sha256,
             result_ref,result_revision,result_sha256,disposition,state,memory_epoch)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.key,
                self.payload_digest,
                self.proof.scope_id,
                self.proof.source_ref,
                self.proof.source_revision,
                target.ref,
                target.revision,
                _digest(asdict(target)),
                result.ref,
                item.revision,
                _digest(asdict(result)),
                item.disposition,
                item.state,
                epoch,
            ),
        )
