"""Resolve shared-store attachment grants and verify when a client can reuse them."""

from __future__ import annotations

import sqlite3
import stat
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..adapters.attachment_paths import UnverifiedAttachmentPath, local_attachment_stat, same_local_path
from ..adapters.audiences import LOCAL_PLATFORMS, LOCAL_USER_ID, HermesIdentityError
from ..adapters.clients.config import CodexConfigError, load_shared_client, remote_server_config_path
from ..adapters.hermes.installation import InstallationManifest, build_installation_manifest
from ..adapters.host_identity import matches_owner_principal
from ..adapters.shared_entries import read_attachment, read_shared_payload
from ..contracts import CLIENT_HOSTS, READ_ONLY_CLIENT_HOSTS, ContractError, InstanceBinding, TrustedContext
from ..core.entry_control import entry_fence_for_keys, local_account_matches
from ..core.restore import restore_pending
from ..core.storage import SQLiteStorage, Transaction
from ..core.store_upgrade import upgrade_in_progress
from ..core.truth_connection import TruthDatabaseConnectionError
from .install_common import RUNTIME_CONFIG_LIMIT
from .install_hermes import DEFAULT_AGENT_WORKSPACE
from .shared_run import SharedStoreError
from .store_snapshot import read_store_snapshot


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


def _other_entries(payload: dict[str, Any], entry_id: str | None, *, read_only: bool) -> dict[str, dict]:
    """Return the attached entries relevant to read or capture grants for the recipient."""
    return {
        entry["entry_id"]: entry
        for entry in payload["entries"]
        if not entry.get("detached_at")
        and entry["entry_id"] != entry_id
        and (read_only or entry["host"] not in READ_ONLY_CLIENT_HOSTS)
    }


def _capture_scope(owners: dict[str, list[dict]], capture_like: str) -> str:
    """Return the source's single capture scope if every relevant owner row can read it."""
    captures = {row["capture_scope_id"] for row in owners[capture_like]}
    if len(captures) != 1:
        raise SharedStoreError(
            f"{capture_like}'s owner rows capture into {len(captures)} scopes; name an entry with one"
        )
    capture = next(iter(captures))
    unread = sorted(
        f"{name}:{row['platform']}"
        for name, rows in owners.items()
        for row in rows
        if capture not in row["allowed_scope_ids"]
    )
    if unread:
        raise SharedStoreError(
            f"{capture_like}'s capture scope is not read by these owner rows: {', '.join(unread)}; "
            "what the owner says in the client would not reach them"
        )
    return capture


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
    entries = _other_entries(payload, entry_id, read_only=read_only)
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
    capture = _capture_scope(others, capture_like)
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


def _local_clients(root: Path, entries: list[dict]) -> list[dict]:
    """Find local client peers; a broken pointer on a known local home is a conflict."""
    clients = []
    for entry in entries:
        if entry["host"] not in CLIENT_HOSTS or not local_account_matches(entry.get("local_account_id")):
            continue
        home = Path(entry["home"])
        try:
            info = local_attachment_stat(home)
        except UnverifiedAttachmentPath:
            continue
        if info is None:
            continue
        pointer = read_attachment(home, local_only=True)
        if (
            pointer is None
            or (pointer.entry_id, pointer.host) != (entry["entry_id"], entry["host"])
            or not same_local_path(pointer.root, root)
        ):
            raise HermesIdentityError("local client attachment no longer matches the shared store")
        remote = local_attachment_stat(remote_server_config_path(home))
        if remote is None:
            clients.append(entry)
        else:
            from ..adapters.clients.remote_server import load_server_config

            if not stat.S_ISREG(remote.st_mode) or remote.st_size > RUNTIME_CONFIG_LIMIT:
                raise HermesIdentityError("remote entry configuration is invalid")
            load_server_config(home, entry["host"])
    return clients


def _runtime_owner_matches(root: Path, entry: dict, binding: InstanceBinding) -> bool:
    """Compare the installed entry's real local route with the owner grants to be copied."""
    entry_id, host = entry["entry_id"], entry["host"]
    client = load_shared_client(Path(entry["home"]), host, expected_store=root)
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


def _registered_owners_match(clients: list[dict], binding: InstanceBinding) -> bool:
    """Verify local registrations and entry state in one current read snapshot."""
    storage = SQLiteStorage(binding, upgrade_on_open=False)
    root = binding.data_directory
    if restore_pending(root) or upgrade_in_progress(root):
        return False
    context = TrustedContext(binding, "attach-preview", binding.scope_ids, "host_generated")
    with read_store_snapshot(storage.path) as conn:
        storage.verify(conn)
        tx = Transaction(conn, context, writable=False)
        try:
            registered = tx.registry.entries()
            states = entry_fence_for_keys(tx, (entry["entry_id"] for entry in clients))
        finally:
            tx._finish()
    return (
        not (restore_pending(root) or upgrade_in_progress(root))
        and all(registered.get(e["entry_id"], {}).get("host") == e["host"] for e in clients)
        and all(state == "normal" for state, _ in states.values())
    )


def verified_donor(
    root: Path,
    payload: dict,
    binding: InstanceBinding,
    *,
    read_only: bool,
    target_entry_id: str | None = None,
    explain_capture_conflict: bool = False,
) -> str | None:
    """Return a local donor usable by the recipient, optionally reporting conflicting owner rows."""
    active = _other_entries(payload, target_entry_id, read_only=True)
    try:
        clients = _local_clients(root, list(active.values()))
        donor = matching_donor(clients, read_only=read_only)
        if (
            donor is not None
            and all(_runtime_owner_matches(root, entry, binding) for entry in clients)
            and _registered_owners_match(clients, binding)
            and read_shared_payload(root) == payload
        ):
            if not read_only:
                entries = _other_entries(payload, target_entry_id, read_only=False)
                _capture_scope({name: owner_rows(entry) for name, entry in entries.items()}, donor)
            return donor
    except SharedStoreError:
        if explain_capture_conflict:
            raise
    except (
        CodexConfigError,
        ContractError,
        HermesIdentityError,
        KeyError,
        OSError,
        TruthDatabaseConnectionError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ):
        pass
    return None
