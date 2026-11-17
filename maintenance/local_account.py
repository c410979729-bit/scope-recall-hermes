"""Explicit local OS-account approval for existing installations and shared entries."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from scope_recall.adapters.clients.config import CONFIG_FILENAME, load_codex_config, write_codex_config
from scope_recall.adapters.installation import MANIFEST_FILENAME, write_installation_manifest
from scope_recall.adapters.shared_entries import (
    ENTRY_HOSTS,
    load_binding_for_home,
    read_attachment,
    read_shared_payload,
    shared_entry_manifest,
    write_shared_payload,
)
from scope_recall.contracts import ContractError
from scope_recall.core.entry_control import current_account_pin, require_local_account
from scope_recall.core.retained_artifacts import extended_path

from .shared_run import Run, home_lock, store_lock


def _approve_existing(pin):
    """Enroll an unbound installation explicitly; a different account cannot replace a pin."""
    if pin is not None:
        require_local_account(pin)
    return current_account_pin()


def approve(host, home):
    """Approve the invoking OS account, preserving identity, data and every audience grant.

    This management operation never runs from capture, MCP, installation loading or
    remote forwarding. An existing pin is accepted only for that same account.
    """
    home = Path(home).expanduser()
    if not home.is_absolute() or not home.is_dir() or host not in ENTRY_HOSTS:
        raise ContractError("INPUT_INVALID", "installation_home")
    now = datetime.now(UTC).isoformat()
    with home_lock(home, "approve-local-account", now):
        attachment = read_attachment(home)
        if attachment is not None:
            if attachment.host != host:
                raise ContractError("ACCESS_DENIED", "entry_host")
            with store_lock(attachment.root, "approve-local-account", now):
                manifest = shared_entry_manifest(attachment.root, attachment.entry_id, hermes_home=home)
                pin = _approve_existing(manifest.local_account_id)
                payload = read_shared_payload(attachment.root)
                record = next(row for row in payload["entries"] if row["entry_id"] == attachment.entry_id)
                run = Run(extended_path(attachment.root), "approve-local-account", now)
                run.keep(extended_path(attachment.root / MANIFEST_FILENAME), "before-account-approval")
                record["local_account_id"] = pin
                write_shared_payload(attachment.root, payload)
                receipt = run.receipt(dict(entry_id=attachment.entry_id, account_bound=True))
        else:
            manifest = load_binding_for_home(home) if host == "hermes" else None
            config = load_codex_config(home / CONFIG_FILENAME) if host == "codex" else None
            if manifest is not None:
                pin = _approve_existing(manifest.local_account_id)
                path = manifest.data_directory / MANIFEST_FILENAME
            elif config is not None:
                pin = _approve_existing(config.local_account_id)
                path = config.config_path
            else:
                raise ContractError("ACCESS_DENIED", "shared_entry_required")
            run = Run(extended_path(path.parent), "approve-local-account", now)
            run.keep(extended_path(path), "before-account-approval")
            if manifest is not None:
                write_installation_manifest(replace(manifest, local_account_id=pin))
            else:
                assert config is not None
                write_codex_config(replace(config, local_account_id=pin))
            receipt = run.receipt(dict(account_bound=True))
    return dict(account_bound=True, receipt=receipt)


def main(argv=None):
    """Approve only an explicitly selected local installation after operator confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, choices=ENTRY_HOSTS)
    parser.add_argument("--instance-root", required=True)
    parser.add_argument("--yes", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(approve(args.host, args.instance_root), sort_keys=True))
        return 0
    except (ContractError, OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                dict(
                    status="error",
                    code=getattr(exc, "code", "CONFIG_INVALID"),
                    field=getattr(exc, "field", "installation_home"),
                )
            )
        )
        return 2
