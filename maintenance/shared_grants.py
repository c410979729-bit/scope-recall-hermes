"""Resolve shared-store attachment grants and verify when a client can reuse them."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..adapters.audiences import LOCAL_PLATFORMS, LOCAL_USER_ID, HermesIdentityError
from ..adapters.clients.config import CodexConfigError, load_shared_client
from ..adapters.hermes.identity import bind_hermes_identity
from ..adapters.hermes.installation import InstallationManifest, build_installation_manifest
from ..adapters.host_identity import matches_owner_principal
from ..adapters.shared_entries import read_attachment, read_shared_payload
from ..contracts import CLIENT_HOSTS, READ_ONLY_CLIENT_HOSTS, ContractError, TrustedContext
from ..core.entry_control import entry_fence_for_keys, local_account_matches
from ..core.restore import restore_pending
from ..core.storage import SQLiteStorage, Transaction
from .install_hermes import DEFAULT_AGENT_WORKSPACE
from .legacy_catalog import offline_source_path
from .migration_records import MigrationError, open_immutable
from .shared_run import SharedStoreError


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


def fresh_installation(
    home: Path,
    payload: dict[str, Any],
    agent_identity: str | None = None,
    *,
    local_platforms: Sequence[str] = (),
) -> InstallationManifest:
    """Build a new entry's owner grants from the store identity and selected local platforms."""
    manifest = build_installation_manifest(
        home,
        agent_id=payload["agent_id"],
        agent_workspace=DEFAULT_AGENT_WORKSPACE,
        test_mode=payload["test_mode"],
        local_platforms=local_platforms,
    )
    return replace(manifest, agent_identity=agent_identity)


def client_grants(
    payload: dict[str, Any],
    home: Path,
    entry_id: str,
    like: tuple[str, ...],
    capture_like: str | None,
    *,
    read_only: bool = False,
) -> tuple[set[str], set[str], str]:
    """Resolve a client's readable and writable scopes and capture destination.

    The first entry gets the default owner's scope. Later entries must reference
    attached donors. A write target must be readable by every other owner entry.
    """
    entries = {
        entry["entry_id"]: entry
        for entry in payload["entries"]
        if not entry.get("detached_at")
        and entry["entry_id"] != entry_id
        and (read_only or entry["host"] not in READ_ONLY_CLIENT_HOSTS)
    }
    if not entries:
        if like or capture_like:
            raise SharedStoreError(
                "no other entry is attached to this store to take grants like: attach its first client without "
                "--grants-like and --capture-like, and it brings the owner's scope"
            )
        owner = fresh_installation(home, payload).audience_scopes["owner_private"]
        return {owner}, set() if read_only else {owner}, owner
    others = {name: owner_rows(entry) for name, entry in entries.items()}
    if read_only:
        if len(like) != 1 or like[0] not in entries:
            raise SharedStoreError("a read-only MCP entry needs --grants-like naming one attached source entry")
        donor = entries[like[0]]
        allowed = {scope for row in others[like[0]] for scope in row["allowed_scope_ids"]}
        owner = donor["audience_scopes"]["owner_private"]
        if owner not in allowed:
            raise SharedStoreError("the selected entry has no verified owner row that reads its owner scope")
        return allowed, set(), owner
    if not like or not capture_like:
        raise SharedStoreError(
            "a client's entry needs --grants-like and --capture-like: the store has entries attached"
        )
    names = tuple(others) if like == ("all",) else like
    unknown = sorted(set(names) - set(others)) + ([capture_like] if capture_like not in others else [])
    if unknown:
        raise SharedStoreError(f"not other entries attached to this store: {', '.join(unknown)}")
    allowed = {scope for name in names for row in others[name] for scope in row["allowed_scope_ids"]}
    writable = {scope for name in names for row in others[name] for scope in row["writable_scope_ids"]}
    captures = {row["capture_scope_id"] for row in others[capture_like]}
    if len(captures) != 1:
        raise SharedStoreError(
            f"{capture_like}'s owner rows capture into {len(captures)} scopes; name an entry with one"
        )
    capture = next(iter(captures))
    unread = sorted(
        f"{name}:{row['platform']}"
        for name, rows in others.items()
        for row in rows
        if capture not in row["allowed_scope_ids"]
    )
    if unread:
        raise SharedStoreError(
            f"{capture_like}'s capture scope is not read by these owner rows: {', '.join(unread)}; "
            "what the owner says in the client would not reach them"
        )
    return allowed | {capture}, writable | {capture}, capture


def matching_donor(entries: list[dict], *, read_only: bool) -> str | None:
    """Choose a verified donor when peer grants agree on the access the new entry can use."""
    read_grants = set()
    write_grants = set()
    donors = set()
    for entry in entries:
        if not local_account_matches(entry.get("local_account_id")):
            return None
        rows = owner_rows(entry)
        if not any(_local_owner(r) for r in rows):
            return None
        can_donate_writes = entry["host"] not in READ_ONLY_CLIENT_HOSTS
        # The explicit write command reserves "all" for a group of donors.
        if read_only or (can_donate_writes and entry["entry_id"] != "all"):
            donors.add(entry["entry_id"])
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
            if owner not in allowed:
                return None
            read_grants.add((owner, identities, allowed))
            if not read_only and can_donate_writes:
                if capture != owner or capture not in writable:
                    return None
                write_grants.add((writable, capture))
    if len(read_grants) != 1 or len(write_grants) > 1:
        return None
    return min(donors, default=None)


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
