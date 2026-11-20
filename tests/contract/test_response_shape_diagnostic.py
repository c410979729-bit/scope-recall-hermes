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
            post({"choices": None, "error": {"message": "abcdefghij"}})
    described = json.loads(_shape_warning(caplog)[0].getMessage().split(":", 2)[2])
    assert described["error"]["message"] == {"type": "str", "len": 10}


def test_only_numeric_protocol_status_fields_keep_their_values(caplog):
    payload = {
        "choices": [None],
        "base_resp": {"status_code": 2056, "account_id": 781093},
        "code": 429,
        "status": 503.0,
        "status_code": [7352],
        "amounts": [9837.125, 821903],
        "counts": {"code": True},
    }
    with pytest.raises(AuxiliaryModelError, match="unsupported_response_shape"):
        post(payload)
    described = json.loads(_shape_warning(caplog)[0].getMessage().split(":", 2)[2])
    assert described == {
        "choices": [{"type": "NoneType"}],
        "base_resp": {"status_code": {"type": "int", "value": 2056}, "field_2": {"type": "int"}},
        "code": {"type": "int", "value": 429},
        "status": {"type": "float", "value": 503.0},
        "status_code": [{"type": "int"}],
        "field_6": [{"type": "float"}, {"type": "int"}],
        "field_7": {"field_1": {"type": "bool"}},
    }


def test_a_readable_envelope_leaves_no_shape_behind(caplog):
    """An answer that was read says nothing about its own shape."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        assert post({"choices": [{"message": {"role": "assistant", "content": "TEST answer"}}]}) == "TEST answer"
    assert _shape_warning(caplog) == []


def test_business_codes_and_arbitrary_member_names_stay_out_of_the_warning(caplog):
    payload = {
        "choices": [None],
        "error": {"code": 503, "status": 429},
        "invoice": {"code": 678207, "status_code": 563890, "status": 501983},
        "base_resp": {"code": 907358, "status": 837156},
        "TEST customer wrote this heading": {"435879": "TEST text"},
        "metadata": [{"error": {"code": 981358}}],
    }
    with pytest.raises(AuxiliaryModelError, match="unsupported_response_shape"):
        post(payload)
    warning = _shape_warning(caplog)[0].getMessage()
    private_names = ("invoice", "TEST customer", "435879", "TEST text")
    business_values = (678207, 563890, 501983, 907358, 837156, 981358)
    for private in (*private_names, *map(str, business_values)):
        assert private not in warning
    shape = json.loads(warning.split(":", 2)[2])
    assert shape["error"] == {"code": {"type": "int", "value": 503}, "status": {"type": "int", "value": 429}}


def test_a_refusal_the_layer_names_leaves_no_shape_behind(caplog):
    """A refusal this layer can name is named, not described: it needs no diagnosis."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        with pytest.raises(AuxiliaryModelError, match="http_status"):
            post({}, read_result=lambda _value: (_ for _ in ()).throw(AuxiliaryModelError("http_status", detail="503")))
    assert _shape_warning(caplog) == []


@pytest.mark.parametrize("member", [{"api_key": "hunter2"}, {"details": {"api_key": "hunter2"}}])
def test_a_credential_like_member_name_is_not_written_down(caplog, member):
    """A member name came from the provider, so a name that reads as a credential is not described."""
    with caplog.at_level(logging.WARNING, logger="scope_recall.runtime.models"):
        with pytest.raises(AuxiliaryModelError, match="unsupported_response_shape"):
            post({"choices": None, **member})
    described = _shape_warning(caplog)[0].getMessage()
    assert "hunter2" not in described
    assert "api_key" not in described
    assert json.loads(described.split(":", 2)[2]) == {"type": "dict", "members": 2}
