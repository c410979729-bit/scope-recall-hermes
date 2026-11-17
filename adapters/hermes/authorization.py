"""What a worker asks of the host's installation: authorization for durable ingress replay, and which principals
are a shared store's owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

from scope_recall.contracts import ContractError, InstanceBinding, SourcePrincipal

from ..host_identity import HermesRuntimeScope, entry_owner
from ..host_identity import resolve_granted_audience as resolve_runtime_audience
from .installation import (
    HermesIdentityError,
    InstallationManifest,
    assert_binding_matches_manifest,
    load_installation_manifest,
)
from .shared_entries import shared_entry_manifest


def _entry_grants(binding: InstanceBinding, entry_id: str) -> InstallationManifest:
    """The grants of the shared store entry that made a capture.

    Entries' audiences differ, so a capture is checked against its own entry's,
    whoever replays it: the shared worker, which binds every scope, or another
    entry whose scopes are not these.  What that replayer may write still bounds
    the result (``replay_inbox`` intersects both).
    """
    if not entry_id:
        raise ContractError("IDENTITY_UNBOUND", "ingress_host_scope")
    manifest = shared_entry_manifest(binding.data_directory, entry_id)
    if (manifest.installation_id, manifest.agent_id, manifest.test_mode) != (
        binding.installation_id,
        binding.agent_id,
        binding.test_mode,
    ):
        raise HermesIdentityError("core binding does not match installation manifest")
    return manifest


def build_ingress_authorizer(binding: InstanceBinding) -> Callable[[object], frozenset[str]]:
    """Re-resolve the captured Hermes audience against the current manifest."""

    if not isinstance(binding, InstanceBinding):
        raise TypeError("binding must be InstanceBinding")
    shared = binding.installation_kind == "shared"

    def authorize(raw: object) -> frozenset[str]:
        if raw is None and binding.test_mode:
            return binding.scope_ids
        if not isinstance(raw, Mapping):
            raise ContractError("IDENTITY_UNBOUND", "ingress_host_scope")
        if not shared:
            manifest = load_installation_manifest(binding.data_directory.parent)
            assert_binding_matches_manifest(binding, manifest)
        try:
            scope = HermesRuntimeScope(**dict(raw))
        except (TypeError, ValueError) as exc:
            raise ContractError("IDENTITY_UNBOUND", "ingress_host_scope") from exc
        if shared:
            manifest = _entry_grants(binding, scope.entry_id)
        return resolve_runtime_audience(manifest, scope).writable_scope_ids

    return authorize


def build_owner_proof(binding: InstanceBinding) -> Callable[[str, SourcePrincipal | None, dict], str | None] | None:
    """A shared store's ownership proof for work that runs outside any host session.

    It is what a host's context carries as ``owner_resolver``: for an entry's verified speaker it names the store's
    owner when that entry's grants approve the principal as the owner's, read from the store's current manifest at
    each request (``host_identity.entry_owner``).  ``None`` for a store of one installation, whose sources name
    no entry.
    """
    if not isinstance(binding, InstanceBinding):
        raise TypeError("binding must be InstanceBinding")
    return partial(entry_owner, binding) if binding.installation_kind == "shared" else None


__all__ = ["build_ingress_authorizer", "build_owner_proof"]
