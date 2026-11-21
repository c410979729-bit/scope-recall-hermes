"""Verified owner grants for local shared-store attachment previews."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from ..adapters.audiences import LOCAL_PLATFORMS, LOCAL_USER_ID, HermesIdentityError
from ..adapters.clients.config import CodexConfigError, load_shared_client
from ..adapters.hermes.identity import bind_hermes_identity
from ..adapters.host_identity import matches_owner_principal
from ..adapters.shared_entries import read_attachment, read_shared_payload
from ..contracts import CLIENT_HOSTS, READ_ONLY_CLIENT_HOSTS, ContractError, TrustedContext
from ..core.entry_control import entry_fence_for_keys, local_account_matches
from ..core.restore import restore_pending
from ..core.storage import SQLiteStorage, Transaction
from .install_hermes import DEFAULT_AGENT_WORKSPACE
from .legacy_catalog import offline_source_path
from .migration_records import MigrationError, open_immutable


def _local_owner(row: dict) -> bool:
    return row["user_id"] == LOCAL_USER_ID and row["platform"] in {"cli", *LOCAL_PLATFORMS, *CLIENT_HOSTS}


def owner_rows(entry: dict) -> list[dict]:
    """Only grants of an explicitly approved owner can pass to a new local entry."""
    return [
        row
        for row in entry["audiences"]
        if row.get("kind") == "owner_private"
        and matches_owner_principal(entry["owner_principals"], platform=row["platform"], user_id=row["user_id"])
    ]


def matching_donor(entries: list[dict], *, read_only: bool) -> str | None:
    """Select a donor only when all active entries identify the same locally approved owner and grants."""
    signatures = set()
    for entry in entries:
        if not local_account_matches(entry.get("local_account_id")):
            return None
        rows = owner_rows(entry)
        if not any(_local_owner(r) for r in rows) or (not read_only and entry["host"] in READ_ONLY_CLIENT_HOSTS):
            return None
        owner = entry["audience_scopes"].get("owner_private")
        # An external login remains a separate identity even when two entries run under the same OS account.
        identities = frozenset(
            (principal["platform"], principal["user_id"])
            for principal in entry["owner_principals"]
            if not _local_owner(principal)
        )
        for row in rows:
            allowed = frozenset(row["allowed_scope_ids"])
            writable = frozenset(row["writable_scope_ids"])
            capture = row["capture_scope_id"]
            if owner not in allowed or (not read_only and (capture != owner or capture not in writable)):
                return None
            signatures.add((owner, identities, allowed, writable, capture))
    if len(signatures) == 1:
        # For a writer, this spelling expands to every donor in the explicit command.
        return min((entry["entry_id"] for entry in entries if read_only or entry["entry_id"] != "all"), default=None)
    return None


def _runtime_owner_matches(root: Path, entry: dict, binding) -> bool:
    """Compare the installed entry's real local route with the owner grants to be copied."""
    home = Path(entry["home"])
    entry_id, host = entry["entry_id"], entry["host"]
    pointer = read_attachment(home)
    # Avoid opening a store named by a pointer to a different installation.
    if pointer is None or (pointer.root, pointer.entry_id, pointer.host) != (root, entry_id, host):
        return False
    if host == "hermes":
        identity = bind_hermes_identity(
            "attach-preview",
            hermes_home=home,
            platform="cli",
            user_id=LOCAL_USER_ID,
            agent_identity=entry.get("agent_identity") or binding.agent_id,
            agent_workspace=DEFAULT_AGENT_WORKSPACE,
        )
        manifest, audience = identity.manifest, identity.runtime_audience
    else:
        client = load_shared_client(home, host)
        manifest, audience = client.manifest, client.audience
    owner = entry["audience_scopes"]["owner_private"]
    row = owner_rows(entry)[0]
    return (
        manifest.entry_id == entry_id
        and manifest.entry_host == host
        and manifest.installation_id == binding.installation_id
        and manifest.agent_id == binding.agent_id
        and manifest.data_directory == root
        and manifest.test_mode == binding.test_mode
        and manifest.audience_scopes.get("owner_private") == owner
        and audience.includes_owner_private
        and audience.allowed_scope_ids == frozenset(row["allowed_scope_ids"])
        and audience.writable_scope_ids == frozenset(row["writable_scope_ids"])
        and audience.capture_scope_id == (row["capture_scope_id"] or None)
    )


def _checkpoint_stamp(storage: SQLiteStorage):
    """Require an offline view before using SQLite's side-effect-free reader."""
    if restore_pending(storage.binding.data_directory):
        return None
    offline_source_path(storage.path)
    return storage._file_stamp()


def _registered_owners_match(active, binding) -> bool:
    """Read registry and entry state only from an unchanged, journal-free database."""
    storage = SQLiteStorage(binding, upgrade_on_open=False)
    stamp = _checkpoint_stamp(storage)
    if stamp is None:
        return False
    context = TrustedContext(binding, "attach-preview", binding.scope_ids, "host_generated")
    # A normal read-only SQLite connection may still create WAL and SHM files.
    with closing(open_immutable(storage.path)) as conn:
        storage.verify(conn)
        tx = Transaction(conn, context, writable=False)
        registered = tx.registry.entries()
        states = entry_fence_for_keys(tx, (entry["entry_id"] for entry in active))
    return (
        _checkpoint_stamp(storage) == stamp
        and all(registered.get(e["entry_id"], {}).get("host") == e["host"] for e in active)
        and all(state == "normal" for state, _ in states.values())
    )


def verified_donor(root: Path, payload: dict, binding, *, read_only: bool) -> str | None:
    """Return the matching donor only if its peer attachments and current store controls agree."""
    active = [entry for entry in payload["entries"] if not entry.get("detached_at")]
    try:
        donor = matching_donor(active, read_only=read_only)
        if (
            donor is not None
            and all(_runtime_owner_matches(root, entry, binding) for entry in active)
            and _registered_owners_match(active, binding)
            and read_shared_payload(root) == payload
        ):
            return donor
    except (
        CodexConfigError,
        ContractError,
        HermesIdentityError,
        KeyError,
        MigrationError,
        OSError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ):
        return None
    return None
