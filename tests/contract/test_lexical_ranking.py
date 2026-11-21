"""Lexical density changes ordering without changing the evidence admission score."""

import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest
from scope_recall.core import retrieval_storage
from scope_recall.core.retrieval import SearchContext
from scope_recall.core.retrieval_storage import RetrievalStorage

from tests.contract.test_claims import app, capture
from tests.v11_support import recall_request


@pytest.fixture(scope="module")
def has_metadata_length():
    """Ask the linked SQLite engine whether it can measure a body without reading it."""
    with closing(sqlite3.connect(":memory:")) as db:
        row = db.execute("SELECT 1 FROM pragma_function_list WHERE name = 'octet_length' AND narg = 1").fetchone()
        return row is not None


@pytest.mark.parametrize("origin", ["tool_observation", "external_document"])
@pytest.mark.parametrize(
    ("query", "answer", "long_text"),
    [
        (
            "orchard deployment database port",
            "The orchard database port is 6101.",
            "orchard deployment database port. " + "Unrelated inventory details. " * 400,
        ),
        (
            "果园部署数据库端口",
            "果园数据库端口是六一零一。",
            "果园部署数据库端口。" + "这里列出各种无关的库存信息。" * 400,
        ),
    ],
    ids=["english", "chinese"],
)
def test_body_length_changes_rank_only_when_supported(app, has_metadata_length, origin, query, answer, long_text):
    """Concise evidence wins with length metadata, while older engines rank by query hits."""
    core, ctx = app
    short = capture(core, ctx, answer, when="2026-09-01T12:00:00Z")
    long = capture(core, ctx, long_text, origin=origin, when="2026-09-02T12:00:00Z")
    reader = replace(ctx, session_id="TEST-lexical-density-reader")
    search = SearchContext.from_request(
        recall_request(query=query, mode="current"),
        reader,
        now=core.clock.utc_now(),
        deadline=core.clock.monotonic() + 10,
    )
    with core.storage.read(reader) as tx:
        found = RetrievalStorage().lexical(tx, search, limit=2)
    expected = [short.ref, long.ref] if has_metadata_length else [long.ref, short.ref]
    assert [candidate.ref for candidate in found] == expected
    # Match counts still describe admission evidence; length changes only the channel rank.
    scores = {candidate.ref: candidate.lexical_score for candidate in found}
    short_score, long_score = scores[short.ref], scores[long.ref]
    assert short_score is not None and long_score is not None
    assert short_score < long_score
    assert all(candidate.lexical_score == len(candidate.matched_query_terms or ()) for candidate in found)


def test_ordinary_messages_keep_recency_when_their_query_hits_are_equal(app):
    """Length normalization must not reorder normal messages on the same topic by tiny length differences."""
    core, ctx = app
    older = capture(core, ctx, "Orchard database port 6101.", when="2026-09-01T12:00:00Z")
    newer = capture(
        core,
        ctx,
        "Orchard database port 6201. " + "A note about the current deployment status. " * 8,
        when="2026-09-02T12:00:00Z",
    )
    search = SearchContext.from_request(
        recall_request(query="orchard database port", mode="current"),
        ctx,
        now=core.clock.utc_now(),
        deadline=core.clock.monotonic() + 10,
    )
    with core.storage.read(ctx) as tx:
        found = RetrievalStorage().lexical(tx, search, limit=2)
    assert [candidate.ref for candidate in found] == [newer.ref, older.ref]
    assert found[0].lexical_score == found[1].lexical_score


def test_multibyte_body_rank_uses_encoded_bytes_when_available(app, has_metadata_length):
    """Equal hit counts use encoded length when available, then recency when length is unavailable."""
    core, ctx = app
    ascii_source = capture(core, ctx, "Orchard database port. " + "a" * 3000, when="2026-09-01T12:00:00Z")
    multibyte_source = capture(core, ctx, "Orchard database port. " + "汉" * 3000, when="2026-09-02T12:00:00Z")
    search = SearchContext.from_request(
        recall_request(query="orchard database port", mode="current"),
        ctx,
        now=core.clock.utc_now(),
        deadline=core.clock.monotonic() + 10,
    )
    with core.storage.read(ctx) as tx:
        found = RetrievalStorage().lexical(tx, search, limit=2)
    expected = (
        [ascii_source.ref, multibyte_source.ref] if has_metadata_length else [multibyte_source.ref, ascii_source.ref]
    )
    assert [candidate.ref for candidate in found] == expected
    assert found[0].lexical_score == found[1].lexical_score


def test_engine_without_metadata_length_keeps_hit_and_recency_order_without_reading_body(app, monkeypatch):
    """An engine without octet_length uses the baseline rank, never a body-reading compatibility expression."""
    import runpy

    # A separate namespace exercises initialization without replacing classes already imported by Core.
    version = retrieval_storage.sqlite3.sqlite_version_info
    monkeypatch.setattr(retrieval_storage.sqlite3, "sqlite_version_info", (3, 42, 0))
    legacy = runpy.run_path(retrieval_storage.__file__, run_name="scope_recall.core._legacy_lexical")
    monkeypatch.setattr(retrieval_storage, "_LEXICAL_RANK", legacy["_LEXICAL_RANK"])
    monkeypatch.setattr(retrieval_storage.sqlite3, "sqlite_version_info", version)
    core, ctx = app
    older = capture(core, ctx, "Orchard database port.", when="2026-09-01T12:00:00Z")
    newer = capture(core, ctx, "Orchard database port. " + "x" * 12000, when="2026-09-02T12:00:00Z")
    more_hits = capture(core, ctx, "Orchard deployment database port. " + "x" * 16000, when="2026-08-01T12:00:00Z")
    search = SearchContext.from_request(
        recall_request(query="orchard deployment database port", mode="current"),
        ctx,
        now=core.clock.utc_now(),
        deadline=core.clock.monotonic() + 10,
    )
    statements = []
    with core.storage.read(ctx) as tx:
        tx._check().set_trace_callback(statements.append)
        found = RetrievalStorage().lexical(tx, search, limit=3)
    assert [candidate.ref for candidate in found] == [more_hits.ref, newer.ref, older.ref]
    ranked = next(statement for statement in statements if "ORDER BY CASE" in statement and "hits" in statement)
    assert "length(" not in ranked.lower()
    assert "cast(e.content" not in ranked.lower()
