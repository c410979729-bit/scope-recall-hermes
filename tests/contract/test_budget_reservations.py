"""Synthetic refusals and explicit reconciliation of uncertain API charges."""

from __future__ import annotations

import importlib.util
import json
import socket
import sqlite3
import ssl
import time
from contextlib import closing
from dataclasses import replace
from unittest.mock import patch

import pytest
from scope_recall.maintenance.cli import main
from scope_recall.maintenance.doctor import _ledger_headroom
from scope_recall.maintenance.doctor_report import DoctorReport
from scope_recall.maintenance.doctor_store import check_ledger
from scope_recall.runtime import model_budget as budget
from scope_recall.runtime import models
from test_monthly_auxiliary_budget import instant, policy
from test_runtime_auxiliary import FakeTransport

NOW = instant("2026-11-10T12:00:00")
CUTOFF = instant("2026-11-09T12:00:00")
OLD = instant("2026-11-02T17:21:00")
SUFFIX = "_usage_unknown_reserved_charge_retained"
REFUSALS = (400, 401, 402, 403, 404, 405, 413, 415, 422, 429)


@pytest.fixture
def ledger(tmp_path):
    configured = policy(
        cap_micro_usd=10_000_000,
        total_input_cap=20_000_000,
        total_output_cap=1_500_000,
        total_call_cap=10_000,
        default_reserve_input=32768,
        default_reserve_output=8192,
    )
    path = tmp_path / "budget.sqlite3"
    budget.initialize_auxiliary_budget_ledger(path, configured)
    result = budget.AuxiliaryBudgetLedger(path, configured)
    yield result
    result.close()


def reserve(ledger, when=OLD):
    with patch.object(budget.time, "time_ns", return_value=when):
        return ledger.reserve("model", b"TEST", reserved_input=32768, reserved_output=8192)


def totals(ledger):
    with closing(sqlite3.connect(ledger.path)) as db:
        return budget.recompute_budget_totals(db, budget.budget_window(ledger.policy, NOW))


def post(ledger, handler, settle=None):
    return models.metered_post(
        ledger=ledger,
        settle=settle or ledger.finish,
        model="model",
        body=b"TEST",
        reserved_input=32768,
        reserved_output=8192,
        deadline=time.monotonic() + 10,
        transport=FakeTransport(handler),
        endpoint="https://synthetic.invalid/model",
        headers={},
        max_response_bytes=1024,
        read_usage=lambda payload: payload.get("usage"),
        read_result=lambda payload: "answer",
    )


@pytest.mark.parametrize("code", REFUSALS)
@pytest.mark.parametrize("embedding", (False, True))
def test_burst_of_explicit_refusals_releases_tokens_and_charge(ledger, code, embedding):
    ledger.policy = replace(ledger.policy, total_output_cap=8192)
    finish = ledger.finish_embedding if embedding else ledger.finish
    with patch.object(budget.time, "time_ns", return_value=NOW):
        for _ in range(4):
            with pytest.raises(models.AuxiliaryModelError, match="http_status"):
                post(ledger, lambda **_: (code, b'{"error":{"type":"Refused"}}'), finish)
        assert totals(ledger) == dict(requests=4, charge_micro_usd=0, input_tokens=0, output_tokens=0)
        assert reserve(ledger, NOW) > 0


@pytest.mark.parametrize("phase", ("connect", "read", None))
@pytest.mark.parametrize("error", ("network_error", "timeout", "http_protocol"))
def test_completed_worker_reply_distinguishes_unsent_from_uncertain(ledger, phase, error):
    reply = json.dumps(dict(ok=False, status=None, body_b64="", error=error, phase=phase, elapsed_ms=1)).encode()

    def fail(**_):
        return models._worker_reply(reply, 1024)

    with patch.object(budget.time, "time_ns", return_value=NOW):
        with pytest.raises(models.AuxiliaryModelError):
            post(ledger, fail)
    expected = 0 if phase == "connect" else 8192
    assert totals(ledger)["output_tokens"] == expected


def test_parent_connect_progress_timeout_is_still_uncertain(ledger):
    def fail(**_):
        raise models.AuxiliaryModelError("timeout", phase="connect")

    with patch.object(budget.time, "time_ns", return_value=NOW):
        with pytest.raises(models.AuxiliaryModelError):
            post(ledger, fail)
    assert totals(ledger)["output_tokens"] == 8192


@pytest.mark.parametrize("code", (429, 503))
def test_http_headers_remain_authoritative_when_error_body_read_fails(ledger, code):
    reply = json.dumps(dict(ok=False, status=code, body_b64="", error="timeout", phase="read", elapsed_ms=1)).encode()
    with patch.object(budget.time, "time_ns", return_value=NOW):
        with pytest.raises(models.AuxiliaryModelError):
            post(ledger, lambda **_: models._worker_reply(reply, 1024))
    assert totals(ledger)["output_tokens"] == (0 if code == 429 else 8192)


@pytest.mark.parametrize("status", ("network_error", "timeout", "http_500", "http_503", "http_200", "http_408"))
@pytest.mark.parametrize("usage", (None, {"prompt_tokens": -1, "completion_tokens": 0}))
def test_uncertain_or_invalid_usage_keeps_reservation_and_status(ledger, status, usage):
    request = reserve(ledger)
    assert ledger.finish(request, status, usage) == status + SUFFIX
    assert totals(ledger)["output_tokens"] == 8192


def test_doctor_shows_retained_amounts_ratios_and_refused_reservations(ledger):
    ledger.policy = replace(ledger.policy, total_output_cap=8192)
    ledger.finish(reserve(ledger), "network_error", None)
    with patch.object(budget.time, "time_ns", return_value=NOW):
        headroom = _ledger_headroom(ledger.path, ledger.policy)
    retained = headroom["retained_unknown"]
    assert retained["requests"] == 1
    assert retained["input_tokens"] == 32768
    assert retained["output_tokens"] == 8192
    assert retained["charge_micro_usd"] == 40960
    assert retained["cap_ratios"] == dict(
        calls=0.0001, input_tokens=32768 / 20_000_000, output_tokens=1.0, charge_micro_usd=40960 / 10_000_000
    )
    assert headroom["blocked_reservations"][0]["model"] == "model"
    report = DoctorReport(host="hermes", status="ok")
    report.ledger_headroom = headroom
    check_ledger(report)
    assert "auxiliary_budget_exhausted" in report.capability_gaps
    text = str(report.checks)
    assert "requests are being refused" in text
    assert "release-budget-reservations" in text
    assert "8192" in text and "40960" in text and "100.0%" in text


def reconcile(ledger, *, apply=False):
    from scope_recall.runtime.budget_reservations import release_unknown_reservations

    return release_unknown_reservations(ledger, before_ns=CUTOFF, apply=apply, now_ns=NOW)


def test_preview_apply_scope_and_idempotency(ledger):
    old = reserve(ledger)
    ledger.finish(old, "network_error", None)
    for when in (instant("2026-10-31T23:59:59"), CUTOFF, NOW):
        ledger.finish(reserve(ledger, when), "network_error", None)
    ledger.finish(reserve(ledger), "http_200", {"prompt_tokens": 1, "completion_tokens": 1})
    reserve(ledger)  # An in-flight call is never an operator release candidate.
    with closing(sqlite3.connect(ledger.path)) as db:
        original = db.execute("SELECT * FROM requests").fetchall()
        schema = db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    preview = reconcile(ledger)
    assert preview["totals"] == dict(requests=1, input_tokens=32768, output_tokens=8192, charge_micro_usd=40960)
    assert [row["id"] for row in preview["records"]] == [old]
    with closing(sqlite3.connect(ledger.path)) as db:
        assert db.execute("SELECT * FROM requests").fetchall() == original
    assert reconcile(ledger, apply=True)["totals"] == preview["totals"]
    assert reconcile(ledger, apply=True)["totals"]["requests"] == 0
    with closing(sqlite3.connect(ledger.path)) as db:
        assert db.execute("SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema
        rows = db.execute("SELECT * FROM requests").fetchall()
        assert rows[1:] == original[1:]
        status, inputs, outputs, charge = db.execute(
            "SELECT status,actual_input,actual_output,charge_micro_usd FROM requests WHERE id=?", (old,)
        ).fetchone()
        assert status == "network_error_usage_unknown_released_by_operator"
        assert (inputs, outputs, charge) == (0, 0, 0)
        for window in (None, budget.budget_window(ledger.policy, NOW)):
            for model in (None, "model"):
                assert budget.budget_totals(db, window, model=model) == budget.recompute_budget_totals(
                    db, window, model=model
                )


def test_106_uncertain_failures_lock_out_then_operator_release_restores(ledger):
    with patch.object(budget.time, "time_ns", return_value=OLD):
        successful = ledger.reserve("model", b"TEST", reserved_input=32768, reserved_output=631004)
    ledger.finish(successful, "http_200", {"prompt_tokens": 32768, "completion_tokens": 631004})
    for _ in range(106):
        ledger.finish(reserve(ledger), "network_error", None)
    assert totals(ledger)["output_tokens"] == 1_499_356
    with pytest.raises(ValueError, match="budget_exhausted"):
        reserve(ledger)
    assert reconcile(ledger)["totals"]["output_tokens"] == 868352
    reconcile(ledger, apply=True)
    assert totals(ledger)["output_tokens"] == 631004
    assert reserve(ledger) > 0


def test_cli_preview_requires_apply_and_aware_cutoff(ledger, capsys, monkeypatch):
    from types import SimpleNamespace

    ledger.finish(reserve(ledger), "network_error", None)
    monkeypatch.setattr(
        "scope_recall.maintenance.budget.load_config",
        lambda _: SimpleNamespace(auxiliary=SimpleNamespace(ledger_path=ledger.path, budget=ledger.policy)),
    )
    args = [
        "release-budget-reservations",
        "--config",
        str(ledger.path.parent / "runtime.json"),
        "--before",
        "2026-11-09T12:00:00Z",
    ]
    with patch.object(budget.time, "time_ns", return_value=NOW):
        assert main(args) == 0
        assert json.loads(capsys.readouterr().out)["totals"]["requests"] == 1
        assert totals(ledger)["output_tokens"] == 8192
        assert main([*args, "--apply"]) == 0
        assert json.loads(capsys.readouterr().out)["applied"] is True
    assert totals(ledger)["output_tokens"] == 0
    assert main([*args[:-1], "2026-11-09"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_subscription_unknown_usage_can_be_inspected_and_explicitly_released(ledger, monkeypatch):
    from scope_recall.runtime import subscription_budget as sub

    monkeypatch.setattr(sub, "_day", lambda: "2026-11-10")
    account = sub.SubscriptionBudgetLedger(ledger.path, sub.SubscriptionBudgetPolicy(daily_calls=20))
    with patch.object(budget.time, "time_ns", return_value=OLD):
        refused = account.reserve("subscription-model", timeout_seconds=1)
    account.finish(refused, "request_not_sent", None, timeout_seconds=1)
    assert account.status()["input_tokens"] == 0
    for _ in range(2):
        with patch.object(budget.time, "time_ns", return_value=OLD):
            request = account.reserve("subscription-model", timeout_seconds=1)
        account.finish(request, "timeout", None, timeout_seconds=1)
    assert account.status()["retained_unknown"]["requests"] == 2
    preview = account.release_unknown_reservations(before_ns=CUTOFF)
    assert preview["totals"]["requests"] == 2
    assert account.status()["output_tokens"] == 16384
    account.release_unknown_reservations(before_ns=CUTOFF, apply=True)
    assert account.status()["output_tokens"] == 0
    assert account.release_unknown_reservations(before_ns=CUTOFF, apply=True)["totals"]["requests"] == 0
    assert account.status()["daily_calls"] == 3


def test_subscription_spawn_failure_proves_no_request_was_sent(tmp_path, monkeypatch):
    from scope_recall.runtime import codex_cli

    def fail(*args, **kwargs):
        raise FileNotFoundError("synthetic executable")

    monkeypatch.setattr(codex_cli.subprocess, "Popen", fail)
    with pytest.raises(models.AuxiliaryModelError) as caught:
        codex_cli._run(["synthetic"], cwd=tmp_path, prompt=b"TEST", seconds=1)
    assert caught.value.request_not_sent is True


@pytest.mark.parametrize("stage", ("connect", "read"))
@pytest.mark.parametrize("failure", (socket.gaierror, ConnectionRefusedError, ssl.SSLError, TimeoutError))
def test_real_worker_marks_only_completed_connection_failures_as_unsent(ledger, monkeypatch, stage, failure):
    spec = importlib.util.spec_from_file_location("synthetic_http_worker", models._HTTP_WORKER_PATH)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    sent = []

    class Connection:
        sock = None
        timeout = None

        def connect(self):
            if stage == "connect":
                raise failure("synthetic")

        def request(self, *args, **kwargs):
            sent.append(True)

        def getresponse(self):
            raise failure("synthetic")

        def close(self):
            pass

    monkeypatch.setattr(worker, "_take_connection", lambda *args, **kwargs: ("TEST", Connection()))
    request = models._worker_request(
        "https://synthetic.invalid/model", body=b"TEST", headers={}, budget=1, max_response_bytes=1024
    )
    reply = worker._request(request)
    assert bool(sent) == (stage == "read")
    with patch.object(budget.time, "time_ns", return_value=NOW):
        with pytest.raises(models.AuxiliaryModelError):
            post(ledger, lambda **_: models._worker_reply(reply, 1024))
    assert totals(ledger)["output_tokens"] == (0 if stage == "connect" else 8192)


def test_subscription_release_keeps_other_days_newer_and_inflight_calls(ledger, monkeypatch):
    from scope_recall.runtime import subscription_budget as sub

    account = sub.SubscriptionBudgetLedger(ledger.path, sub.SubscriptionBudgetPolicy(daily_calls=20))
    for day, when in (("2026-11-09", OLD), ("2026-11-10", CUTOFF), ("2026-11-10", NOW)):
        monkeypatch.setattr(sub, "_day", lambda day=day: day)
        with patch.object(budget.time, "time_ns", return_value=when):
            request = account.reserve("model", timeout_seconds=1)
        account.finish(request, "timeout", None, timeout_seconds=1)
    with patch.object(budget.time, "time_ns", return_value=OLD):
        account.reserve("model", timeout_seconds=1)
    assert account.release_unknown_reservations(before_ns=CUTOFF, apply=True)["totals"]["requests"] == 0
    assert account.status()["daily_calls"] == 3


def test_unbilled_outcome_cannot_charge_reported_extras(ledger):
    request = reserve(ledger)
    ledger.finish(request, "http_429", {"prompt_tokens": 5, "completion_tokens": 2, "unreported_output_tokens": 100})
    assert totals(ledger)["charge_micro_usd"] == 0


def test_model_specific_cap_and_lifetime_amount_blocking_are_visible(ledger):
    ledger.policy = replace(
        ledger.policy, period="lifetime", model_token_caps={"model": (None, 8192)}, cap_micro_usd=40960
    )
    ledger.finish(reserve(ledger), "http_503", None)
    headroom = _ledger_headroom(ledger.path, ledger.policy)
    blocked = headroom["blocked_reservations"][0]
    assert set(blocked["caps"]) == {"model_output_tokens", "charge_micro_usd"}
    assert headroom["retained_unknown"]["by_model"]["model"]["cap_ratios"]["output_tokens"] == 1.0


def test_embedding_route_ignores_generation_output_headroom(ledger):
    from types import SimpleNamespace

    from scope_recall.runtime.embedding_models import EmbeddingRouteConfig

    ledger.policy = replace(ledger.policy, total_output_cap=0)
    auxiliary = SimpleNamespace(
        external_consolidation=False,
        external_embedding=True,
        embedding=EmbeddingRouteConfig(
            credential_env="SCOPE_RECALL_TEST",
            model="model",
            endpoint="https://synthetic.invalid/embeddings",
            dimensions=8,
            dialect="openai",
        ),
    )
    assert _ledger_headroom(ledger.path, ledger.policy, auxiliary=auxiliary)["blocked_reservations"] == []


def test_subscription_cli_selector_and_doctor_report(ledger, monkeypatch, capsys):
    from types import SimpleNamespace

    from scope_recall.runtime import subscription_budget as sub

    monkeypatch.setattr(sub, "_day", lambda: "2026-11-10")
    configured = sub.SubscriptionBudgetPolicy(daily_calls=20, daily_input_tokens=65536, daily_output_tokens=16384)
    account = sub.SubscriptionBudgetLedger(ledger.path, configured)
    for _ in range(2):
        with patch.object(budget.time, "time_ns", return_value=OLD):
            request = account.reserve("model", timeout_seconds=1)
        account.finish(request, "timeout", None, timeout_seconds=1)
    ledger.finish(reserve(ledger), "network_error", None)
    auxiliary = SimpleNamespace(
        ledger_path=ledger.path,
        budget=ledger.policy,
        external_consolidation=True,
        external_embedding=False,
        consolidation=SimpleNamespace(kind="codex_cli", subscription_budget=configured),
    )
    monkeypatch.setattr("scope_recall.maintenance.budget.load_config", lambda _: SimpleNamespace(auxiliary=auxiliary))
    headroom = _ledger_headroom(ledger.path, ledger.policy, auxiliary=auxiliary)
    assert headroom["subscription"]["retained_unknown"]["cap_ratios"]["output_tokens"] == 1.0
    report = DoctorReport(host="hermes", status="ok")
    report.ledger_headroom = headroom
    check_ledger(report)
    assert "subscription_budget_exhausted" in report.capability_gaps
    assert "--subscription" in str(report.checks)
    args = [
        "release-budget-reservations",
        "--subscription",
        "--config",
        str(ledger.path.parent / "runtime.json"),
        "--before",
        "2026-11-09T12:00:00Z",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["totals"]["charge_micro_usd"] is None
    assert account.status()["output_tokens"] == 16384
    assert main([*args, "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is True
    assert account.status()["output_tokens"] == 0
    assert totals(ledger)["output_tokens"] == 8192


def assert_all_summaries_match_details(path):
    """Check every persisted calendar/model projection against authoritative request rows."""
    with closing(sqlite3.connect(path)) as db:
        rows = db.execute("SELECT * FROM request_totals ORDER BY window_start_ns,window_end_ns,model_key").fetchall()
        assert rows
        for start, end, model_key, calls, charge, inputs, outputs in rows:
            window = None if (start, end) == (0, 0) else (start, end)
            model = json.loads(model_key) if model_key else None
            assert budget.recompute_budget_totals(db, window, model=model) == dict(
                requests=calls, charge_micro_usd=charge, input_tokens=inputs, output_tokens=outputs
            )
    return rows


@pytest.mark.parametrize("release_first", (False, True))
def test_release_and_settlement_serialize_without_double_adjustment(ledger, release_first):
    from scope_recall.runtime.budget_reservations import release_unknown_reservations

    request = reserve(ledger)
    maintenance = budget.AuxiliaryBudgetLedger(ledger.path, ledger.policy)
    try:
        if release_first:
            # A maintenance transaction cannot take a reservation still in flight.
            assert reconcile(maintenance, apply=True)["totals"]["requests"] == 0
            assert_all_summaries_match_details(ledger.path)
        assert ledger.finish(request, "network_error", None) == "network_error" + SUFFIX
        assert_all_summaries_match_details(ledger.path)
        released = release_unknown_reservations(maintenance, before_ns=CUTOFF, apply=True, now_ns=NOW)
        assert released["totals"]["requests"] == 1
        after_release = assert_all_summaries_match_details(ledger.path)
        # A delayed completion on the other handle cannot settle a released row again.
        with pytest.raises(ValueError, match="reservation_state"):
            ledger.finish(request, "http_200", {"prompt_tokens": 12, "completion_tokens": 3})
        assert reconcile(maintenance, apply=True)["totals"]["requests"] == 0
        assert assert_all_summaries_match_details(ledger.path) == after_release
        assert totals(ledger) == dict(requests=1, charge_micro_usd=0, input_tokens=0, output_tokens=0)
    finally:
        maintenance.close()


def test_released_reservation_survives_dirty_summary_rebuild(ledger):
    ledger.finish(reserve(ledger), "network_error", None)
    ledger.finish(reserve(ledger), "http_200", {"prompt_tokens": 12, "completion_tokens": 3})
    assert reconcile(ledger, apply=True)["totals"]["requests"] == 1
    incremental = assert_all_summaries_match_details(ledger.path)
    with closing(sqlite3.connect(ledger.path)) as db, db:
        db.execute("UPDATE budget_summary_state SET dirty=1 WHERE id=1")
    statements = []
    with ledger._write_transaction(deadline=None) as db:
        db.set_trace_callback(statements.append)
        try:
            budget.prepare_budget_totals(db, ledger.policy)
        finally:
            db.set_trace_callback(None)
    assert any(sql.startswith("DELETE FROM request_totals") for sql in statements)
    assert assert_all_summaries_match_details(ledger.path) == incremental
    with closing(sqlite3.connect(ledger.path)) as db:
        assert db.execute("SELECT dirty FROM budget_summary_state WHERE id=1").fetchone() == (0,)


def test_release_uses_local_month_and_offset_cutoff_across_utc_days(ledger):
    from scope_recall.maintenance.budget import _cutoff
    from scope_recall.runtime.budget_reservations import release_unknown_reservations

    ledger.policy = replace(ledger.policy, time_zone="Asia/Tokyo")
    budget.initialize_auxiliary_budget_ledger(ledger.path, ledger.policy)
    cutoff = _cutoff("2026-11-01T09:00:00+09:00")
    assert cutoff == instant("2026-11-01T00:00:00")
    times = (
        "2026-10-31T14:59:59",
        "2026-10-31T15:00:00",
        "2026-10-31T23:59:59",
        "2026-11-01T00:00:00",
        "2026-11-01T00:00:01",
        "2026-11-30T15:00:00",
    )
    requests = []
    for when in times:
        request = reserve(ledger, instant(when))
        ledger.finish(request, "network_error", None)
        requests.append(request)
    with closing(sqlite3.connect(ledger.path)) as db:
        original = {row[0]: row for row in db.execute("SELECT * FROM requests")}
        assert {row[0] for row in db.execute("SELECT zone FROM budget_summary_zones")} == {"UTC", "Asia/Tokyo"}
    receipt = release_unknown_reservations(ledger, before_ns=cutoff, apply=True, now_ns=NOW)
    assert receipt["window_start_ns"] == instant("2026-10-31T15:00:00")
    assert receipt["window_end_ns"] == instant("2026-11-30T15:00:00")
    assert [row["id"] for row in receipt["records"]] == requests[1:3]
    assert receipt["totals"] == dict(requests=2, input_tokens=65536, output_tokens=16384, charge_micro_usd=81920)
    with closing(sqlite3.connect(ledger.path)) as db:
        for request in (requests[0], *requests[3:]):
            assert db.execute("SELECT * FROM requests WHERE id=?", (request,)).fetchone() == original[request]
    assert_all_summaries_match_details(ledger.path)
