"""A provider that refuses on its own account is a capacity refusal, not an unfamiliar shape."""

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
from scope_recall.core.worker_outcomes import model_exception_outcome
from scope_recall.runtime.auxiliary import build_auxiliary_runtime
from scope_recall.runtime.models import AuxiliaryModelError
from test_candidate_lifecycle import _candidate, _candidate_rows, _finish_source_work
from test_claims import app
from test_runtime_auxiliary import FakeTransport, _runtime_config

QUOTA_CODE = "provider_quota_exhausted"
ERROR_CODE = "provider_error"


def _envelope(status_msg: str, status_code: int = 429) -> bytes:
    """A 200 answer in which the provider declines and says why."""
    return json.dumps(
        {"choices": None, "base_resp": {"status_code": status_code, "status_msg": status_msg}},
        ensure_ascii=False,
    ).encode("utf-8")


def _refusal(tmp_path, monkeypatch, status_msg: str, status_code: int = 429) -> AuxiliaryModelError:
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
        pytest.param("Token Plan limit reached", QUOTA_CODE, id="plan-limit"),
        pytest.param("insufficient funds for this plan", QUOTA_CODE, id="quota-by-other-word"),
    ],
)
def test_a_provider_refusal_in_a_200_answer_is_named(tmp_path, monkeypatch, status_msg, expected):
    """The provider's own refusal is read as a refusal, not as a shape this layer cannot read."""
    assert _refusal(tmp_path, monkeypatch, status_msg).error_type == expected


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
