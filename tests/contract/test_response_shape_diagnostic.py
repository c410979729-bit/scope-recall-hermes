"""An envelope this layer does not understand leaves its shape behind, and nothing else."""

from __future__ import annotations

import json
import logging

import pytest
from scope_recall.runtime import consolidation_models, models
from scope_recall.runtime.models import AuxiliaryModelError
from test_runtime_auxiliary import FakeTransport


class _Ledger:
    """A ledger that reserves nothing; the diagnostic is about the reply, not the charge."""

    def reserve(self, *_args, **_kwargs):
        return 1

    def finish(self, *_args, **_kwargs):
        return "ok"


def post(payload, *, read_result=None, model="synthetic-model"):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return models.metered_post(
        ledger=_Ledger(),
        settle=lambda *_args, **_kwargs: "ok",
        model=model,
        body=b"TEST",
        reserved_input=0,
        reserved_output=0,
        deadline=__import__("time").monotonic() + 10,
        transport=FakeTransport(lambda **_kwargs: (200, body)),
        endpoint="https://synthetic.invalid/model",
        headers={},
        max_response_bytes=1024 * 1024,
        read_usage=lambda value: value.get("usage"),
        read_result=read_result or consolidation_models._extract_chat_content,
    )


def _shape_warning(caplog):
    return [record for record in caplog.records if "scope_unsupported_response_shape" in record.getMessage()]


def test_an_unreadable_envelope_leaves_its_shape_in_the_log(caplog):
    """The shape is what tells "the gateway cut the answer off" apart from "the model wrote something else"."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        with pytest.raises(AuxiliaryModelError, match="unsupported_response_shape"):
            post({"choices": [{"message": {"role": "assistant", "content": 7}}], "base_resp": {"status_code": 2056}})
    warnings = _shape_warning(caplog)
    assert len(warnings) == 1
    described = warnings[0].getMessage()
    assert "choices" in described and "base_resp" in described
    assert "status_code" in described and "2056" in described


def test_the_shape_never_carries_the_providers_own_text(caplog):
    """A description is a shape, not a copy: the answer's text is not written down."""
    secret_ish = "TEST 这是一段模型生成的内容，不应被写进日志"
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        with pytest.raises(AuxiliaryModelError, match="unsupported_response_shape"):
            post({"choices": None, "detail": secret_ish})
    described = _shape_warning(caplog)[0].getMessage()
    assert secret_ish not in described
    assert "len" in described


def test_the_shape_describes_sizes_rather_than_texts(caplog):
    """A string is described by its length, so its content cannot leak through it."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        with pytest.raises(AuxiliaryModelError, match="unsupported_response_shape"):
            post({"choices": None, "note": "abcdefghij"})
    described = json.loads(_shape_warning(caplog)[0].getMessage().split(":", 2)[2])
    assert described["note"] == {"type": "str", "len": 10}


def test_a_readable_envelope_leaves_no_shape_behind(caplog):
    """An answer that was read says nothing about its own shape."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        assert post({"choices": [{"message": {"role": "assistant", "content": "TEST answer"}}]}) == "TEST answer"
    assert _shape_warning(caplog) == []


def test_a_refusal_the_layer_names_leaves_no_shape_behind(caplog):
    """A refusal this layer can name is named, not described: it needs no diagnosis."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        with pytest.raises(AuxiliaryModelError, match="http_status"):
            post({}, read_result=lambda _value: (_ for _ in ()).throw(AuxiliaryModelError("http_status", detail="503")))
    assert _shape_warning(caplog) == []


def test_a_credential_like_member_name_is_not_written_down(caplog):
    """A member name came from the provider, so a name that reads as a credential is not described."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        with pytest.raises(AuxiliaryModelError, match="unsupported_response_shape"):
            post({"choices": None, "api_key": "hunter2"})
    described = _shape_warning(caplog)[0].getMessage()
    assert "hunter2" not in described
    assert "api_key" not in described
