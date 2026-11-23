"""Explicit local-owner commands for source review and the store's quote exception."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from scope_recall.contracts import ContractError
from scope_recall.core.quarantine_management import injection_settings, set_injection_settings
from scope_recall.core.source_review import apply_sources, preview_sources

from .memory import _common, open_management
from .restore import RestoreError


def parser(command):
    """Build a bounded owner command without opening a store."""
    root = argparse.ArgumentParser(prog=f"scope-recall {command}", description=__doc__)
    _common(root)
    root.add_argument("--apply", action="store_true", help="confirm the previewed action")
    if command == "injection-policy":
        root.add_argument("--auto-release-owner-quotes", choices=("on", "off"))
    else:
        root.add_argument("--state", choices=("pending_review",), default="pending_review")
        root.add_argument("--rule", action="append", default=[])
        root.add_argument("--origin", action="append", default=[])
        root.add_argument("--after", help="exclusive scanned_at lower bound, ISO 8601 with timezone")
        root.add_argument("--before", help="exclusive scanned_at upper bound, ISO 8601 with timezone")
        root.add_argument("--limit", type=int, default=100, help="maximum selected versions, 1 to 1000")
        root.add_argument("--decision", choices=("release", "retain"))
        root.add_argument("--preview", help="saved JSON preview required with --apply")
    return root


def _policy(api, args):
    current = injection_settings(api.core, api.authority)
    selected = args.auto_release_owner_quotes
    if args.apply:
        if selected is None:
            raise ContractError("INPUT_INVALID", "auto_release_owner_quotes")
        return dict(
            applied=True, **set_injection_settings(api.core, api.authority, auto_release_owner_quotes=selected == "on")
        )
    return dict(applied=False, **current, requested=selected)


def _review(api, args):
    if args.apply:
        if not args.preview:
            raise ContractError("INPUT_INVALID", "preview_required")
        if args.rule or args.origin or args.after or args.before:
            raise ContractError("INPUT_INVALID", "preview_filters")
        path = Path(args.preview)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ContractError("INPUT_INVALID", "source_preview")
        plan = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            raise ContractError("INPUT_INVALID", "source_preview")
        return apply_sources(api.core, api.authority, plan, decision=args.decision or plan.get("decision"))
    if args.preview:
        raise ContractError("INPUT_INVALID", "confirmation_required")
    return preview_sources(
        api.core,
        api.authority,
        state=args.state,
        rules=args.rule,
        origins=args.origin,
        after=args.after,
        before=args.before,
        limit=args.limit,
        decision=args.decision or "release",
    )


def main(argv):
    """Preview without initializing the store, or apply an explicitly confirmed owner action."""
    command, *arguments = argv
    args = parser(command).parse_args(arguments)
    try:
        api = open_management(args, initialize=False)
        result = _policy(api, args) if command == "injection-policy" else _review(api, args)
    except ContractError as exc:
        result = dict(status="error", code=exc.code, field=exc.field)
    except (OSError, ValueError, TypeError, sqlite3.Error, RestoreError):
        result = dict(status="error", code="REVIEW_INPUT_INVALID")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 2 if result.get("status") == "error" else 1 if result.get("failed") else 0
