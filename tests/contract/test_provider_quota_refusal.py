"""Provider refusals in successful HTTP envelopes follow the same policy as transport-level refusals."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
from scope_recall.core.failure_policy import (
    AUTO_RECOVERABLE_ERRORS,
    CAPACITY_REFUSALS,
    RATE_LIMITED_ERRORS,
    retryable,
)
from scope_recall.core.work_storage import MAX_RECOVERABLE_ATTEMPTS
from scope_recall.core.worker_outcomes import model_exception_outcome
from scope_recall.runtime.auxiliary import build_auxiliary_runtime
from scope_recall.runtime.models import AuxiliaryModelError
from test_candidate_lifecycle import _candidate, _candidate_rows, _finish_source_work
from test_claims import app
from test_runtime_auxiliary import FakeTransport, _runtime_config

QUOTA_CODE = "provider_quota_exhausted"
ERROR_CODE = "provider_error"


def _envelope(status_msg: str, status_code: int) -> bytes:
    """A 200 answer in which the provider declines and says why."""
    return json.dumps(
        {"choices": None, "base_resp": {"status_code": status_code, "status_msg": status_msg}},
        ensure_ascii=False,
    ).encode("utf-8")


def _refusal(tmp_path, monkeypatch, status_msg: str, status_code: int = 9999) -> AuxiliaryModelError:
    config, _ledger, _ = _runtime_config(tmp_path)
    monkeypatch.setenv("SCOPE_RECALL_TEST_CHAT_KEY", "TEST-key")
    transport = FakeTransport(lambda **kwargs: (200, _envelope(status_msg, status_code)))
    runtime = build_auxiliary_runtime(config, transport=transport)
    with pytest.raises(AuxiliaryModelError) as exc:
        runtime.consolidation.propose([{"role": "user", "content": "TEST bounded input"}], remaining_seconds=2.0)
    assert transport.calls == 1
    return exc.value


@pytest.mark.parametrize(
    "status_msg,expected",
    [
        pytest.param("额度已用尽", QUOTA_CODE, id="chinese-quota"),
        pytest.param("配额不足，请稍后重试", QUOTA_CODE, id="chinese-quota-words"),
        pytest.param("insufficient_quota", QUOTA_CODE, id="english-quota"),
        pytest.param("QUOTA_EXCEEDED", QUOTA_CODE, id="english-quota-joined"),
        pytest.param("Token Plan limit reached", QUOTA_CODE, id="plan-limit"),
    ],
)
def test_a_provider_refusal_in_a_200_answer_is_named(tmp_path, monkeypatch, status_msg, expected):
    """The provider's own refusal is read as a refusal, not as a shape this layer cannot read."""
    assert _refusal(tmp_path, monkeypatch, status_msg).error_type == expected


def test_a_provider_saying_it_has_no_limit_is_not_read_as_one(tmp_path, monkeypatch):
    """A provider saying it is unlimited has said there is no limit: it has not hit one."""
    config, _ledger, _ = _runtime_config(tmp_path)
    monkeypatch.setenv("SCOPE_RECALL_TEST_CHAT_KEY", "TEST-key")
    body = json.dumps({"choices": None, "base_resp": {"status_msg": "unlimited plan"}}).encode("utf-8")
    transport = FakeTransport(lambda **kwargs: (200, body))
    runtime = build_auxiliary_runtime(config, transport=transport)
    with pytest.raises(AuxiliaryModelError) as exc:
        runtime.consolidation.propose([{"role": "user", "content": "TEST bounded input"}], remaining_seconds=2.0)
    assert exc.value.error_type == "provider_error"


def test_a_refusal_that_states_its_status_is_read_as_that_status(tmp_path, monkeypatch):
    """A provider that names its status joins the codes this layer already knows."""
    refusal = _refusal(tmp_path, monkeypatch, "service temporarily unavailable", 503)
    assert refusal.error_type == "http_status"
    assert refusal.detail == "503"
    assert model_exception_outcome(refusal) == ("retry", "http_503")


def test_an_account_refusal_keeps_its_own_meaning(tmp_path, monkeypatch):
    """A credential refusal is not a capacity refusal: retrying it would ask the same key again."""
    refusal = _refusal(tmp_path, monkeypatch, "invalid api key", 401)
    assert refusal.error_type == "http_status"
    assert model_exception_outcome(refusal) == ("failed", "http_401")


@pytest.mark.parametrize(
    "status,message,outcome",
    [
        (402, "quota exceeded", ("failed", "http_402")),
        (400, "context limit exceeded", ("failed", "http_400")),
        (1004, "access denied", ("failed", "http_401")),
        (1008, "insufficient balance", ("failed", "http_402")),
        (9999, "insufficient funds for this plan", ("failed", "http_402")),
        (1027, "output content flagged", ("failed", "model_refused")),
        (2013, "invalid parameters", ("failed", "input_invalid")),
        (9999, "input exceeds context length limit", ("failed", "request_too_large")),
        (1002, "busy", ("retry", QUOTA_CODE)),
        (2056, "usage limit exceeded", ("retry", QUOTA_CODE)),
        (9999, "You exceeded your current quota, please check your plan and billing details.", ("retry", QUOTA_CODE)),
        (9999, "今日额度已用完，请明天再试", ("retry", QUOTA_CODE)),
        (9999, "API rate limit exceeded for this organization", ("retry", "rate_limited")),
        (9999, "RPM rate limit exceeded", ("retry", "rate_limited")),
        (9999, "请求过于频繁，触发限流", ("retry", "rate_limited")),
        (9999, "Your account balance is insufficient", ("failed", "http_402")),
        (9999, "The API key provided is invalid", ("failed", "http_401")),
        (9999, "unexpected provider refusal", ("retry", ERROR_CODE)),
        (9999, "quota metadata unavailable", ("retry", ERROR_CODE)),
        (9999, "unsupported rate limit option", ("retry", ERROR_CODE)),
        (9999, "invalid parameter; the account is not rate limited", ("retry", ERROR_CODE)),
        (9999, "rate limited: false", ("retry", ERROR_CODE)),
        (9999, "No quota exceeded for this account", ("retry", ERROR_CODE)),
        (9999, "Quota exhausted = false", ("retry", ERROR_CODE)),
        (9999, "Quota exceeded is incorrect", ("retry", ERROR_CODE)),
        (9999, "已排除额度不足", ("retry", ERROR_CODE)),
        (9999, "今日额度未用完", ("retry", ERROR_CODE)),
        (9999, "不是额度不足，是认证失败", ("failed", "http_401")),
        (9999, "不是额度不足而是认证失败", ("failed", "http_401")),
        (9999, "Not rate limited but the API key provided is invalid", ("failed", "http_401")),
        (9999, "Not rate limited and the API key provided is invalid", ("failed", "http_401")),
        (9999, "Authentication failed was incorrect; quota exhausted", ("retry", QUOTA_CODE)),
        (9999, "Neither rate limited nor quota exhausted", ("retry", ERROR_CODE)),
        (9999, "Neither rate limited, nor quota exhausted; authentication failed", ("failed", "http_401")),
        (9999, "并非触发限流，但是今日额度已用完", ("retry", QUOTA_CODE)),
    ],
)
def test_provider_refusals_keep_their_queue_policy(tmp_path, monkeypatch, status, message, outcome):
    assert model_exception_outcome(_refusal(tmp_path, monkeypatch, message, status)) == outcome


def test_a_declining_provider_that_names_no_status_is_a_provider_error(tmp_path, monkeypatch):
    """The provider declined and said nothing this layer can name."""
    config, _ledger, _ = _runtime_config(tmp_path)
    monkeypatch.setenv("SCOPE_RECALL_TEST_CHAT_KEY", "TEST-key")
    body = json.dumps({"choices": None, "base_resp": {"status_msg": "cannot serve"}}).encode("utf-8")
    transport = FakeTransport(lambda **kwargs: (200, body))
    runtime = build_auxiliary_runtime(config, transport=transport)
    with pytest.raises(AuxiliaryModelError) as exc:
        runtime.consolidation.propose([{"role": "user", "content": "TEST bounded input"}], remaining_seconds=2.0)
    assert exc.value.error_type == ERROR_CODE


def test_an_envelope_without_a_refusal_still_reads_as_an_unsupported_shape(tmp_path, monkeypatch):
    """``choices: null`` with nothing said about it stays a shape error: no refusal was stated."""
    config, _ledger, _ = _runtime_config(tmp_path)
    monkeypatch.setenv("SCOPE_RECALL_TEST_CHAT_KEY", "TEST-key")
    transport = FakeTransport(lambda **kwargs: (200, json.dumps({"choices": None}).encode("utf-8")))
    runtime = build_auxiliary_runtime(config, transport=transport)
    with pytest.raises(AuxiliaryModelError) as exc:
        runtime.consolidation.propose([{"role": "user", "content": "TEST bounded input"}], remaining_seconds=2.0)
    assert exc.value.error_type == "unsupported_response_shape"


def test_a_provider_refusal_is_a_capacity_refusal():
    """A spent quota clears by itself, so it recovers the way the other capacity refusals do."""
    assert QUOTA_CODE in CAPACITY_REFUSALS
    assert QUOTA_CODE in AUTO_RECOVERABLE_ERRORS
    assert QUOTA_CODE in RATE_LIMITED_ERRORS
    assert retryable(QUOTA_CODE)
    assert retryable(QUOTA_CODE, candidate_attempted=True)


def test_an_unreadable_shape_still_ends_a_candidate_for_good():
    """The change must not make every unfamiliar shape retryable: an unknown shape is unknown."""
    assert "unsupported_response_shape" not in CAPACITY_REFUSALS
    assert not retryable("unsupported_response_shape", candidate_attempted=True)


def test_a_quota_refusal_clears_the_candidate_attempt_fence(app):
    """The refusal is a known outcome, so the candidate may be asked again."""
    core, ctx = app
    _candidate(core, ctx)
    _finish_source_work(core)

    class QuotaEvaluation:
        calls = 0

        def evaluate_candidate(self, candidate, sources, *, remaining_seconds):
            self.calls += 1
            raise AuxiliaryModelError(QUOTA_CODE)

    port = QuotaEvaluation()
    core.drain_worker(ctx, max_items=8, remaining_seconds=10, consolidation=port)
    _, evaluations, work = _candidate_rows(core)
    assert port.calls == 1 and work[0]["state"] != "failed"
    moment = datetime.fromisoformat(core.clock.now.replace("Z", "+00:00")) + timedelta(hours=1)
    core.clock.now = moment.isoformat().replace("+00:00", "Z")
    core.drain_worker(ctx, max_items=8, remaining_seconds=10, consolidation=port)
    assert port.calls == 2
    _, evaluations, _work = _candidate_rows(core)
    assert evaluations[0]["failure_code"] == QUOTA_CODE


@pytest.mark.parametrize(
    "message,code,expected_work",
    [
        ("cannot serve", ERROR_CODE, (MAX_RECOVERABLE_ATTEMPTS, "failed", MAX_RECOVERABLE_ATTEMPTS)),
        ("已排除额度不足，实际是认证失败", "http_401", (1, "pending", 0)),
        ("Quota exceeded is incorrect; authentication failed", "http_401", (1, "pending", 0)),
        ("Quota exceeded; the API key provided is invalid", "http_401", (1, "pending", 0)),
    ],
)
def test_unknown_and_account_refusals_stay_out_of_capacity_retries(
    app, tmp_path, monkeypatch, message, code, expected_work
):
    """Account refusals park for account recovery; unknown refusals spend the ordinary retry budget."""
    core, ctx = app
    _candidate(core, ctx)
    _finish_source_work(core)
    refusal = _refusal(tmp_path, monkeypatch, message)

    class Declined:
        calls = 0

        def evaluate_candidate(self, candidate, sources, *, remaining_seconds):
            self.calls += 1
            raise refusal

    port = Declined()
    for _ in range(MAX_RECOVERABLE_ATTEMPTS + 2):
        core.drain_worker(ctx, max_items=8, remaining_seconds=10, consolidation=port)
        moment = datetime.fromisoformat(core.clock.now.replace("Z", "+00:00")) + timedelta(minutes=10)
        core.clock.now = moment.isoformat().replace("+00:00", "Z")
    _, evaluations, work = _candidate_rows(core)
    assert evaluations[0]["failure_code"] == code
    assert (port.calls, work[0]["state"], work[0]["attempt"]) == expected_work
    assert "capacity:" not in work[0]["last_error_code"]
    assert code not in RATE_LIMITED_ERRORS
