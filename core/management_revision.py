"""Create truthful local-management provenance for an explicit owner correction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from uuid import uuid4

from ..contracts import ContractError, validate_capture
from .delete_storage import canonical
from .entry_control import current_account_pin
from .mutate import management_claims_private, revise_from_management


def correct(tx, clock, ref, expected_revision, value, *, action_id=None):
    """Write one correction and its source atomically through existing repositories."""
    if type(expected_revision) is not int or expected_revision < 1:
        raise ContractError("INPUT_INVALID", "expected_revision")
    if value is not None and (type(value) is not str or not value.strip() or len(value) > 4096):
        raise ContractError("INPUT_INVALID", "value")
    head = tx.claims.version(ref, expected_revision)
    if head is None or head.current_revision != expected_revision:
        raise ContractError("VERSION_CONFLICT", "expected_revision")
    tx.claims.require_target(head)
    now, action_id = clock.utc_now(), action_id or "management-" + uuid4().hex
    shared = tx.context.binding.installation_kind == "shared"
    if shared:
        tx.registry.register_entry("local-management", "Local management", "management", now=now)
    context = replace(
        tx.context,
        session_id=action_id,
        actor_origin="host_generated",
        source_principal=None,
        entry_id="local-management" if shared else None,
    )
    view = tx.with_context(context, writable=True)
    event = dict(
        protocol_version="1.1",
        source_event_key=action_id,
        source_revision=1,
        origin="host_generated",
        role="system",
        content=value if value is not None else f"Withdraw {ref}: {head.payload['value_text']}",
        occurred_at=now,
        recorded_at=now,
        time_precision="instant",
        capture_state="complete",
        evidence_refs=[],
        source_context=dict(platform="local_management", chat_type="management"),
    )
    saved = view.sources.put_source(validate_capture(event, context), scope_id=head.scope_id, persisted_at=now)
    view.sources.index_source(saved.ref, saved.revision)
    source = view.source(saved.ref, saved.revision)
    if source is None:
        raise ContractError("SOURCE_MISSING", "management_source")
    request = dict(
        target_ref=ref,
        expected_revision=expected_revision,
        new_value=value,
        conditions=head.payload["conditions"],
        valid_from=now,
    )
    result = revise_from_management(view, clock, request, source)
    return action_id, result, source.ref


def revision_request(context, ref, expected_revision, value, request_id):
    """Bind an optional retry key to the local capability and the exact command.

    The audit keeps only the command digest, never another copy of its value.
    Without an explicit occurrence key, a fresh command retains normal version checks.
    """
    if request_id is None:
        return None, None
    if type(request_id) is not str or not request_id.strip() or len(request_id) > 240:
        raise ContractError("INPUT_INVALID", "request_id")
    authority = [
        context.binding.installation_id,
        context.binding.agent_id,
        current_account_pin(),
        context.session_id,
        context.entry_id,
        sorted(context.allowed_scope_ids),
        context.project_id,
        context.branch_id,
        context.actor_origin,
        context.task_anchor,
        asdict(context.source_principal) if context.source_principal is not None else None,
    ]
    key = "management-revise-" + hashlib.sha256(canonical([authority, request_id]).encode()).hexdigest()
    digest = hashlib.sha256(canonical([ref, expected_revision, value]).encode()).hexdigest()
    return key, digest


def replay_revision(tx, clock, identity, digest):
    """Recover a durable response only while its target and authority remain usable."""
    if identity is None:
        return None
    row = (
        tx._check()
        .execute("SELECT targets_json FROM management_actions WHERE action_id=? AND action='revise'", (identity,))
        .fetchone()
    )
    if row is None:
        return None
    receipt = json.loads(row[0])[0]["receipt"]
    if receipt["request_sha256"] != digest:
        raise ContractError("VERSION_CONFLICT", "request_id")
    result = receipt["result"]
    item = result["item"]
    head = tx.claims.version(item["ref"], item["revision"])
    if head is None:
        raise ContractError("SOURCE_MISSING", "claim")
    tx.claims.require_target(head)
    if head.current_revision != item["revision"]:
        raise ContractError("VERSION_CONFLICT", "expected_revision")
    if management_claims_private(tx, clock, (head,)):
        raise ContractError("ACCESS_DENIED", "private_mode")
    # The original provenance must still be readable; deletion cannot be undone by replay.
    tx.claims.require_live_source(result["source_ref"], 1)
    return result
