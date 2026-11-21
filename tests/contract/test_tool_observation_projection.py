"""Lexical and semantic projection of a long tool source retain its evidence."""

from dataclasses import replace

import pytest
from scope_recall.core import lexical_index
from scope_recall.core.events import (
    TOOL_OBSERVATION_INDEX_CHARS,
    indexed_terms,
    lexical_terms,
    tool_observation_excerpt,
)
from scope_recall.core.recall_policy import encode_embedding_text
from scope_recall.runtime.auxiliary import build_auxiliary_runtime
from scope_recall.runtime.models import AuxiliaryModelError
from test_claims import app, capture
from test_runtime_auxiliary import FakeTransport, _recorded_embeddings, _runtime_config, _source

LONG_OUTPUT = "headsignal\n" + " ".join(f"diagnostictoken{i:04d}" for i in range(300)) + "\ntailsignal"


def test_capture_and_projection_repair_share_the_search_excerpt(app):
    core, ctx = app
    source = capture(core, ctx, LONG_OUTPUT, origin="tool_observation")
    assert source.event["content"] == LONG_OUTPUT
    assert {"headsignal", "tailsignal"} <= set(indexed_terms(source.event))
    assert "diagnostictoken0150" not in indexed_terms(source.event)
    assert [item.ref for item in core.search_sources(ctx, "headsignal tailsignal")] == [source.ref]
    assert not core.search_sources(ctx, "diagnostictoken0150")
    with core.storage.write(ctx) as tx:
        conn = tx._check(write=True)
        key = lexical_index.source_id(conn, source.ref, source.revision)
        assert lexical_index.terms_of(conn, key) == indexed_terms(source.event)
        # Simulate an older index before normal source projection repair runs.
        lexical_index.index_terms(conn, key, lexical_terms(LONG_OUTPUT))
        assert tx.sources.source_projection_status(source.ref, source.revision)[0] == "not_ready"
        tx.sources.index_source(source.ref, source.revision)
        assert lexical_index.terms_of(conn, key) == indexed_terms(source.event)
        assert not conn.execute("SELECT term FROM lexical_terms WHERE term='diagnostictoken0150'").fetchone()
        assert tx.sources.source_projection_status(source.ref, source.revision)[0] == "ready"
    assert core.source(ctx, source.ref, source.revision).event["content"] == LONG_OUTPUT


def test_single_and_batch_vector_requests_share_the_lexical_source_excerpt(tmp_path, monkeypatch):
    config, _, _ = _runtime_config(tmp_path)
    monkeypatch.setenv("SCOPE_RECALL_TEST_EMBED_KEY", "test-key")
    sent = []
    runtime = build_auxiliary_runtime(config, transport=FakeTransport(_recorded_embeddings(sent)))
    human = _source(LONG_OUTPUT)
    observed = replace(human, event={**human.event, "origin": "tool_observation", "role": "tool"})
    excerpt = tool_observation_excerpt(observed.event)
    assert len(excerpt) == TOOL_OBSERVATION_INDEX_CHARS
    expected = encode_embedding_text(excerpt, kind="document")
    runtime.source_embedding.embed_source(observed, remaining_seconds=2.0)
    runtime.source_embedding.embed_sources([human, observed], remaining_seconds=2.0)
    assert sent == [[expected], [encode_embedding_text(LONG_OUTPUT, kind="document"), expected]]


def test_vector_projection_does_not_expose_a_partial_credential_across_the_tail_cut(tmp_path):
    config, _, _ = _runtime_config(tmp_path)
    transport = FakeTransport(lambda **_kwargs: (200, b"{}"))
    runtime = build_auxiliary_runtime(config, transport=transport)
    content = "TEST log\n" * 160 + "api_key=TESTsecretvalue1234567890 " + "x " * 250
    source = _source(content)
    source = replace(source, event={**source.event, "origin": "tool_observation", "role": "tool"})
    assert "api_key=" not in tool_observation_excerpt(source.event)
    with pytest.raises(AuxiliaryModelError, match="sensitive_request"):
        runtime.source_embedding.embed_sources([_source("TEST"), source], remaining_seconds=2.0)
    assert transport.calls == 0
