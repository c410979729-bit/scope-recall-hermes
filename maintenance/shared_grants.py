"""Verified owner grants for local shared-store attachment previews."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..adapters.audiences import LOCAL_PLATFORMS, LOCAL_USER_ID, HermesIdentityError
from ..adapters.host_identity import matches_owner_principal
from ..adapters.shared_entries import read_attachment
from ..contracts import CLIENT_HOSTS, READ_ONLY_CLIENT_HOSTS, ContractError, TrustedContext
from ..core.entry_control import entry_fence_for_keys, local_account_matches
from ..core.storage import SQLiteStorage


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
            if owner not in allowed or (not read_only and (capture not in writable or capture not in allowed)):
                return None
            signatures.add((owner, identities, allowed, writable, capture))
    if len(signatures) == 1:
        return min(entry["entry_id"] for entry in entries)
    return None


def verified_donor(root: Path, payload: dict, binding, *, read_only: bool) -> str | None:
    """Return the matching donor only if its peer attachments and current store controls agree."""
    active = [entry for entry in payload["entries"] if not entry.get("detached_at")]
    donor = matching_donor(active, read_only=read_only)
    if donor is None:
        return None
    try:
        for entry in active:
            pointer = read_attachment(entry["home"])
            if pointer is None or (pointer.root, pointer.entry_id, pointer.host) != (
                root,
                entry["entry_id"],
                entry["host"],
            ):
                return None
        storage = SQLiteStorage(binding, upgrade_on_open=False)
        context = TrustedContext(binding, "attach-preview", binding.scope_ids, "host_generated")
        with storage.read(context, remaining_seconds=1.0) as tx:
            registered = tx.registry.entries()
            if any(registered.get(e["entry_id"], {}).get("host") != e["host"] for e in active):
                return None
            states = entry_fence_for_keys(tx, (entry["entry_id"] for entry in active))
            if any(state != "normal" for state, _ in states.values()):
                return None
    except (ContractError, HermesIdentityError, OSError, sqlite3.Error):
        return None
    return donor
