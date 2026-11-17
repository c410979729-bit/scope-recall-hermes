"""Resolve native occurrence witnesses through the existing source authority."""

from dataclasses import replace
from hashlib import sha256

from scope_recall.contracts import RequestMessageBinding
from scope_recall.core.entry_control import refuse_unbound_mutation


def bind_source(core, context, source_key, text, occurrence, live):
    """Bind one exact persisted source; missing or filtered capture refuses."""
    with core.storage.read(context) as tx:
        source = tx.sources.source_by_event_key(source_key)
        if source is None:
            refuse_unbound_mutation()
        proof = RequestMessageBinding(
            context.binding,
            context.entry_id,
            context.session_id,
            context.project_id,
            context.branch_id,
            source.scope_id,
            source.ref,
            source.revision,
            sha256(text.encode("utf-8")).hexdigest(),
            occurrence,
            context.source_principal,
            live,
        )
        bound = replace(context, request_message=proof)
        proof.validate(bound, source)
        tx.claims.require_live_source(source.ref, source.revision)
    return bound
