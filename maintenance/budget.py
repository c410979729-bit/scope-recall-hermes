"""Operator preview and explicit write-off of uncertain auxiliary-model usage."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from ..runtime.budget_reservations import release_unknown_reservations
from ..runtime.model_budget import AuxiliaryBudgetLedger
from ..runtime.subscription_budget import SubscriptionBudgetLedger
from ..runtime.worker_entry import load_config


def _cutoff(value: str) -> int:
    """Parse an absolute ISO timestamp with an explicit UTC offset, without float rounding."""
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.utcoffset() is None:
        raise ValueError("before_requires_time_zone")
    delta = moment.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def main(argv: list[str] | None = None) -> int:
    """Use the installation's config; preview by default, write only with --apply."""
    parser = argparse.ArgumentParser(prog="scope-recall release-budget-reservations", description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--before", required=True, help="exclusive request start cutoff, ISO timestamp with UTC offset")
    parser.add_argument(
        "--apply", action="store_true", help="accept possible provider charges and release listed usage"
    )
    parser.add_argument("--subscription", action="store_true", help="use the CLI subscription's current UTC day")
    args = parser.parse_args(argv)
    ledger = None
    try:
        before = _cutoff(args.before)
        path = Path(args.config)
        if not path.is_absolute():
            raise ValueError("config_must_be_absolute")
        auxiliary = load_config(path).auxiliary
        if auxiliary is None or auxiliary.ledger_path is None:
            raise ValueError("ledger_not_configured")
        if args.subscription:
            route = getattr(auxiliary, "consolidation", None)
            if route is None or route.kind != "codex_cli":
                raise ValueError("subscription_not_configured")
            account = SubscriptionBudgetLedger(auxiliary.ledger_path, route.subscription_budget)
            receipt = account.release_unknown_reservations(before_ns=before, apply=args.apply)
        else:
            ledger = AuxiliaryBudgetLedger(auxiliary.ledger_path, auxiliary.budget)
            receipt = release_unknown_reservations(ledger, before_ns=before, apply=args.apply)
    except (ValueError, OSError, sqlite3.Error):
        # Neither a provider body nor a configuration value belongs in an error receipt.
        receipt = dict(status="error", code="BUDGET_RECONCILIATION_INVALID")
    finally:
        if ledger is not None:
            ledger.close()
    sys.stdout.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 2 if receipt.get("status") == "error" else 0
