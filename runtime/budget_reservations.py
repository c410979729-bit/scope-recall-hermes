"""Inspection and explicit release of uncertain auxiliary-model reservations."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing

from .model_budget import (
    UNKNOWN_RESERVED_SUFFIX,
    AuxiliaryBudgetLedger,
    BudgetPolicy,
    _adjust_budget_totals,
    _readonly_uri,
    budget_totals,
    budget_window,
    prepare_budget_totals,
    reservation_cap_checks,
)


def retained_records(
    db: sqlite3.Connection, window: tuple[int, int] | None, *, before_ns: int | None = None
) -> list[dict]:
    """Read settled uncertain records, excluding in-flight and already released calls.

    Selection uses the start time, with an exclusive cutoff. Older ledgers that
    predate status tracking have no identifiable retained reservations.
    """
    columns = {row[1] for row in db.execute("PRAGMA table_info(requests)")}
    if not {"status", "started_ns", "model"} <= columns:
        return []
    clauses = ["status GLOB ?"]
    values: list[object] = ["*" + UNKNOWN_RESERVED_SUFFIX]
    if window is not None:
        clauses.append("started_ns >= ? AND started_ns < ?")
        values.extend(window)
    if before_ns is not None:
        clauses.append("started_ns < ?")
        values.append(before_ns)
    keys = ("id", "model", "status", "started_ns", "input_tokens", "output_tokens", "charge_micro_usd")
    return [
        dict(zip(keys, row, strict=True))
        for row in db.execute(
            "SELECT id,model,status,started_ns,COALESCE(actual_input,reserved_input,0),"
            "COALESCE(actual_output,reserved_output,0),COALESCE(charge_micro_usd,0) FROM requests WHERE "
            + " AND ".join(clauses)
            + " ORDER BY id",
            values,
        )
    ]


def retained_totals(records: list[dict]) -> dict[str, int]:
    """Count selected calls and their currently counted token and money amounts."""
    return {
        "requests": len(records),
        **{key: sum(row[key] for row in records) for key in ("input_tokens", "output_tokens", "charge_micro_usd")},
    }


def reservation_headroom(db: sqlite3.Connection, policy: BudgetPolicy, window, totals: dict, *, auxiliary=None) -> dict:
    """Retained amounts and caps that cannot admit one configured reservation.

    Route input floors are a lower bound: a larger serialized prompt can need
    more. Without route metadata, use the policy's default generation reserve.
    """
    records = retained_records(db, window)
    retained = dict(retained_totals(records))
    caps = dict(
        calls=getattr(policy, "total_call_cap", None),
        input_tokens=getattr(policy, "total_input_cap", None),
        output_tokens=getattr(policy, "total_output_cap", None),
        charge_micro_usd=getattr(policy, "cap_micro_usd", None),
    )
    ratios = {key: retained["requests" if key == "calls" else key] / cap if cap else None for key, cap in caps.items()}
    retained_info = {**retained, "cap_ratios": ratios}
    retained_info["by_model"] = {
        model: {
            **retained_totals([row for row in records if row["model"] == model]),
            "cap_ratios": {
                key: sum(row[key] for row in records if row["model"] == model) / cap if cap else None
                for key, cap in zip(("input_tokens", "output_tokens"), pair, strict=True)
            },
        }
        for model, pair in getattr(policy, "model_token_caps", {}).items()
    }
    blocked = []
    for model, inputs, outputs in _route_reservations(policy, auxiliary):
        model_totals = budget_totals(db, window, model=model)
        required = dict(
            calls=1,
            input_tokens=inputs,
            output_tokens=outputs,
            charge_micro_usd=policy.pricing[model].charge_micro_usd(inputs, outputs),
        )
        checks = reservation_cap_checks(
            policy,
            model,
            totals,
            model_totals,
            amount=required["charge_micro_usd"],
            reserved_input=inputs,
            reserved_output=outputs,
        )
        exceeded = [key for key, cap, used, need in checks if cap is not None and used + need > cap]
        if exceeded:
            held = retained_totals([row for row in records if row["model"] == model])
            blocked.append(dict(model=model, caps=exceeded, required=required, retained_unknown=held))
    return dict(retained_unknown=retained_info, blocked_reservations=blocked)


def _route_reservations(policy, auxiliary) -> list[tuple[str, int, int]]:
    """Configured API route floors, using the same output policy as the adapters."""
    from .consolidation_models import model_output_reserve
    from .embedding_models import EMBED_RESERVE_FLOOR

    if auxiliary is None:
        return [
            (model, policy.default_reserve_input, model_output_reserve(policy, model, 0))
            for model in sorted(getattr(policy, "approved_models", ()))
        ]
    reservations = []
    route = getattr(auxiliary, "consolidation", None)
    if getattr(auxiliary, "external_consolidation", False) and route is not None and route.kind != "codex_cli":
        reservations.append(
            (
                route.model,
                auxiliary.consolidation_reserve_input,
                model_output_reserve(policy, route.model, route.max_output_tokens),
            )
        )
    route = getattr(auxiliary, "embedding", None)
    if getattr(auxiliary, "external_embedding", False) and route is not None:
        reservations.append((route.space()["model"], EMBED_RESERVE_FLOOR, 0))
    return [entry for entry in reservations if entry[0] in policy.approved_models]


def release_unknown_reservations(
    ledger: AuxiliaryBudgetLedger, *, before_ns: int, apply: bool = False, now_ns: int | None = None
) -> dict:
    """Preview or atomically zero old retained charges in the current policy period.

    Every request row and its original reservation survive. Call counts stay
    charged; releasing uncertain billing does not authorize extra API attempts.
    The operator accepts the risk that the provider billed these requests.
    """
    if type(before_ns) is not int or before_ns < 0:
        raise ValueError("invalid_before")
    # Capture the current period inside the write transaction, after lock wait.
    if apply:
        with ledger._write_transaction(deadline=None) as db:
            zones = prepare_budget_totals(db, ledger.policy)
            window = budget_window(ledger.policy, time.time_ns() if now_ns is None else now_ns)
            records = retained_records(db, window, before_ns=before_ns)
            for row in records:
                status = row["status"][: -len(UNKNOWN_RESERVED_SUFFIX)] + "_usage_unknown_released_by_operator"
                db.execute(
                    "UPDATE requests SET status=?,actual_input=0,actual_output=0,charge_micro_usd=0 WHERE id=?",
                    (status, row["id"]),
                )
                _adjust_budget_totals(
                    db,
                    zones,
                    row["started_ns"],
                    row["model"],
                    (0, -row["charge_micro_usd"], -row["input_tokens"], -row["output_tokens"]),
                )
    else:
        with closing(sqlite3.connect(_readonly_uri(ledger.path), uri=True, timeout=5)) as db:
            db.execute("BEGIN")
            window = budget_window(ledger.policy, time.time_ns() if now_ns is None else now_ns)
            records = retained_records(db, window, before_ns=before_ns)
    return dict(
        applied=apply,
        period=ledger.policy.period,
        time_zone=ledger.policy.time_zone,
        window_start_ns=window[0] if window else None,
        window_end_ns=window[1] if window else None,
        before_ns=before_ns,
        records=records,
        totals=retained_totals(records),
    )
