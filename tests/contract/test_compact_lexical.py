"""Compaction repairs only existing, visible full-text tool projections."""

import sqlite3
from contextlib import closing

import pytest
from scope_recall.core import lexical_index
from scope_recall.core.delete_storage import group_digest
from scope_recall.core.events import indexed_terms, lexical_terms
from scope_recall.core.storage import Transaction
from scope_recall.maintenance import compact_lexical
from scope_recall.maintenance.compact_store import compact_store
from test_claims import app, capture, initial
from test_deletion import authorize, request

LONG_OUTPUT = "headsignal\n" + "trace " * 120 + "staleunique sharedkeep " + "trace " * 120 + "\ntailsignal"


def test_compaction_rebuilds_existing_historical_tool_postings_without_changing_evidence(app, tmp_path):
    core, ctx = app
    old = capture(core, ctx, LONG_OUTPUT, origin="tool_observation", key="TEST-old")
    capture(core, ctx, "TEST newer revision", origin="tool_observation", key="TEST-old", revision=2)
    capture(core, ctx, "sharedkeep in an ordinary source")
    claim, _ = initial(core, ctx)
    active = capture(core, ctx, LONG_OUTPUT, origin="tool_observation")
    duplicate = capture(core, ctx, LONG_OUTPUT, origin="tool_observation")
    target = tmp_path / "compacted.sqlite3"
    path = core.storage.path
    with closing(sqlite3.connect(path)) as db, db:
        identity = lexical_index.source_id(db, old.ref, old.revision)
        lexical_index.index_terms(db, identity, lexical_terms(old.event["content"]))
        active_id = lexical_index.source_id(db, active.ref, active.revision)
        active_terms = lexical_index.terms_of(db, active_id)
        duplicate_id = lexical_index.source_id(db, duplicate.ref, duplicate.revision)
        lexical_index.index_terms(db, duplicate_id, lexical_terms(duplicate.event["content"]))
        projections = (
            ("claim_lexical_postings", "subject_terms", "headfact"),
            ("claim_version_postings", "subject_term", "versionfact"),
        )
        for table, marker, term in projections:
            term_id = db.execute("INSERT INTO lexical_terms(term) VALUES (?) RETURNING term_id", (term,)).fetchone()[0]
            db.execute(f"INSERT INTO {table}(term_id,claim_id,{marker}) VALUES (?,?,0)", (term_id, claim.ref))
            lexical_index.index_terms(db, identity, (term,))
        db.execute("INSERT INTO lexical_terms(term) VALUES ('preexistingorphan')")
        assert "staleunique" in lexical_index.terms_of(db, identity)

    preview = compact_store(path)
    receipt = compact_store(path, output=target)
    with closing(sqlite3.connect(target)) as db:
        assert lexical_index.terms_of(db, identity) == indexed_terms(old.event)
        assert lexical_index.terms_of(db, active_id) == active_terms
        assert lexical_index.terms_of(db, duplicate_id) == active_terms
        saved = db.execute("SELECT content FROM source_events WHERE source_id=?", (identity,)).fetchone()
        assert saved == (LONG_OUTPUT,)
        retained = db.execute(
            "SELECT term FROM lexical_terms WHERE term IN "
            "('staleunique','preexistingorphan','sharedkeep','headfact','versionfact') ORDER BY term"
        ).fetchall()
        assert retained == [("headfact",), ("sharedkeep",), ("versionfact",)]
    assert preview["tool_lexical"]["sources_rebuilt"] == 2
    assert preview["tool_lexical"]["sources_current"] == 1
    assert preview["tool_lexical"]["estimated_bytes"] == 6 * 16 + 2 * 8 + len("staleuniquepreexistingorphan")
    assert receipt["tool_lexical"] == preview["tool_lexical"]
    assert compact_store(target)["tool_lexical"]["estimated_bytes"] == 0


def test_compaction_never_adds_postings_to_ineligible_sources(app, tmp_path):
    core, ctx = app
    deleted = capture(core, ctx, "TEST delete generates a group block operation")
    authorize(core, ctx, deleted)
    assert core.forget(ctx, request(deleted), remaining_seconds=10)["read_blocked"]
    sources = {
        kind: capture(core, ctx, LONG_OUTPUT, origin="tool_observation")
        for kind in ("absent", "blocked", "group", "held", "object", "suppressed", "short")
    }
    path = core.storage.path
    with closing(sqlite3.connect(path)) as db, db:
        ids = {kind: lexical_index.source_id(db, item.ref, item.revision) for kind, item in sources.items()}
        for source_id in ids.values():
            lexical_index.unindex(db, (source_id,))
            lexical_index.index_terms(db, source_id, ("onlylegacy",))
        lexical_index.unindex(db, (ids["absent"],))
        db.execute("UPDATE source_events SET read_blocked=1 WHERE source_id=?", (ids["blocked"],))
        db.execute("UPDATE source_events SET suppressed=1 WHERE source_id=?", (ids["suppressed"],))
        db.execute(
            "INSERT INTO source_safety VALUES (?,?,'pending_review','[]','2000-01-01',NULL)",
            (sources["held"].ref, sources["held"].revision),
        )
        db.execute(
            "INSERT INTO object_blocks SELECT 'event',?,scope_id,project_id,branch_id,1,1,"
            "(SELECT operation_id FROM deletion_operations LIMIT 1) FROM source_events WHERE source_id=?",
            (sources["object"].ref, ids["object"]),
        )
        db.execute(
            "UPDATE source_events SET content=? WHERE source_id=?",
            ("ﬃ " * 200, ids["short"]),  # The text still fits the excerpt after normalization.
        )
        key, scope, project, branch = db.execute(
            "SELECT source_group_key,scope_id,project_id,branch_id FROM source_events WHERE source_id=?",
            (ids["group"],),
        ).fetchone()
        digest = group_digest(ctx.binding, scope, project, branch, key)
        db.execute(
            "INSERT INTO source_group_blocks SELECT ?,?,?,?,1,1,operation_id FROM deletion_operations LIMIT 1",
            (digest, scope, project, branch),
        )
    target = tmp_path / "compacted.sqlite3"
    result = compact_store(path, output=target)
    assert result["tool_lexical"]["sources_rebuilt"] == 0
    with closing(sqlite3.connect(target)) as db:
        for kind, source_id in ids.items():
            assert lexical_index.terms_of(db, source_id) == (() if kind == "absent" else ("onlylegacy",))


@pytest.mark.parametrize("output", ["shortoutput tail", LONG_OUTPUT], ids=["short", "long"])
def test_compaction_withdraws_unsafe_legacy_tool_matches(app, tmp_path, output):
    core, ctx = app
    private = "left<private>syntheticprivatecanary</private>right\npublicprivate " + output
    payload = "c3ludGhldGljVHJhbnNwb3J0Q2FuYXJ5"
    transport = f"publictransport data:image/png;base64,{payload}\n" + output
    credential = "api_key=SyntheticScopeRecallTestCanary123456789\n" + output
    legacy = [
        (capture(core, ctx, "TEST old private", origin="tool_observation"), private, "syntheticprivatecanary"),
        (capture(core, ctx, "TEST old transport", origin="tool_observation"), transport, payload.lower()),
        (
            capture(core, ctx, "TEST old credential", origin="tool_observation"),
            credential,
            "syntheticscoperecalltestcanary123456789",
        ),
    ]
    shared = capture(core, ctx, "syntheticprivatecanary belongs to a separate source")
    safe = capture(core, ctx, output, origin="tool_observation")
    path = core.storage.path
    with closing(sqlite3.connect(path)) as db, db:
        ids = [lexical_index.source_id(db, item.ref, item.revision) for item, _, _ in legacy]
        for source_id, (_, raw, _) in zip(ids, legacy, strict=True):
            db.execute("UPDATE source_events SET content=? WHERE source_id=?", (raw, source_id))
            lexical_index.unindex(db, (source_id,))
            lexical_index.index_terms(db, source_id, lexical_terms(raw))
        originals = db.execute("SELECT * FROM source_events ORDER BY source_id").fetchall()
        db.row_factory = sqlite3.Row
        tx = Transaction(db, ctx, writable=False)
        for source, _, canary in legacy:
            assert source.ref in {item.ref for item in tx.sources.search_sources(canary)}

    preview = compact_store(path)
    target = tmp_path / "safety-compacted.sqlite3"
    receipt = compact_store(path, output=target)
    assert preview["tool_lexical"]["sources_rebuilt"] == len(legacy)
    assert preview["tool_lexical"]["postings_added"] == 0
    assert receipt["tool_lexical"] == preview["tool_lexical"]
    with closing(sqlite3.connect(target)) as db:
        assert db.execute("SELECT * FROM source_events ORDER BY source_id").fetchall() == originals
        assert lexical_index.terms_of(db, ids[2]) == ()
        assert not set(lexical_index.terms_of(db, ids[0])).intersection({"leftright", "private"})
        db.row_factory = sqlite3.Row
        tx = Transaction(db, ctx, writable=False)
        assert [item.ref for item in tx.sources.search_sources("syntheticprivatecanary")] == [shared.ref]
        for _, _, canary in legacy[1:]:
            assert not tx.sources.search_sources(canary)
        assert [item.ref for item in tx.sources.search_sources("publicprivate")] == [legacy[0][0].ref]
        assert [item.ref for item in tx.sources.search_sources("publictransport")] == [legacy[1][0].ref]
        signal = "headsignal" if output == LONG_OUTPUT else "shortoutput"
        assert safe.ref in {item.ref for item in tx.sources.search_sources(signal)}

    # A queued projection can still run after the offline copy is put back into service.
    with closing(sqlite3.connect(target)) as db, db:
        db.row_factory = sqlite3.Row
        tx = Transaction(db, ctx, writable=True)
        for source, _, canary in legacy:
            tx.sources.index_source(source.ref, source.revision)
            assert source.ref not in {item.ref for item in tx.sources.search_sources(canary)}
            assert tx.sources.source_projection_status(source.ref, source.revision)[0] == "ready"
        assert not tx.sources.search_sources("leftright")
    # Both modes leave the original index in place for explicit rollback.
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        tx = Transaction(db, ctx, writable=False)
        for source, _, canary in legacy:
            assert source.ref in {item.ref for item in tx.sources.search_sources(canary)}


def test_compaction_reads_postings_once_per_page(app, monkeypatch):
    core, ctx = app
    for _ in range(5):
        capture(core, ctx, LONG_OUTPUT, origin="tool_observation")
    monkeypatch.setattr(compact_lexical, "_PAGE_SIZE", 2)
    statements = []
    with closing(sqlite3.connect(core.storage.path)) as db:
        db.set_trace_callback(statements.append)
        assert compact_lexical.measure_tool_lexical(db)["sources_current"] == 5
    posting_reads = [sql for sql in statements if "json_group_array" in sql and "lexical_postings" in sql]
    assert len(posting_reads) == 3


def test_compaction_preserves_only_safe_error_terms_from_eligible_omission_summaries(app, tmp_path):
    core, ctx = app
    envelope = "Tool execution summary (terminal): protocolmarker="
    marker = "; output_preview=omitted"
    summaries = [
        (envelope + "wire " * 240 + "; error=visiblelongerror" + marker, ("visiblelongerror",)),
        (envelope + "wire; error=<private>hiddenerror</private> visibleshorterror" + marker, ("visibleshorterror",)),
        (envelope + "wire; error=untrusted; <private>output_preview=omitted", ()),
    ]
    short_clean = envelope + "wire; error=ordinaryerror" + marker
    summaries.append((short_clean, lexical_terms(short_clean)))
    sources = [capture(core, ctx, "TEST legacy omission summary", origin="tool_observation") for _ in summaries]
    path = core.storage.path
    with closing(sqlite3.connect(path)) as db, db:
        ids = [lexical_index.source_id(db, item.ref, item.revision) for item in sources]
        for source_id, (summary, _) in zip(ids, summaries, strict=True):
            db.execute("UPDATE source_events SET content=? WHERE source_id=?", (summary, source_id))
            lexical_index.unindex(db, (source_id,))
            lexical_index.index_terms(db, source_id, lexical_terms(summary))
    target = tmp_path / "placeholder-compacted.sqlite3"
    preview = compact_store(path)
    assert preview["tool_lexical"]["sources_rebuilt"] == 3
    assert compact_store(path, output=target)["tool_lexical"] == preview["tool_lexical"]
    with closing(sqlite3.connect(target)) as db:
        assert [lexical_index.terms_of(db, source_id) for source_id in ids] == [keep for _, keep in summaries]


@pytest.mark.parametrize(
    "raw",
    [LONG_OUTPUT.replace("\n", "\x00", 1), "ﬃ " * 175 + " staleuniquemiddle " + "ﬃ " * 175],
    ids=["embedded-nul", "normalized-length"],
)
def test_compaction_uses_the_runtime_excerpt_boundary(app, tmp_path, raw):
    core, ctx = app
    source = capture(core, ctx, raw, origin="tool_observation")
    path = core.storage.path
    with closing(sqlite3.connect(path)) as db, db:
        source_id = lexical_index.source_id(db, source.ref, source.revision)
        lexical_index.unindex(db, (source_id,))
        lexical_index.index_terms(db, source_id, lexical_terms(raw))
    target = tmp_path / "excerpt-compacted.sqlite3"
    receipt = compact_store(path, output=target)
    assert receipt["tool_lexical"]["sources_rebuilt"] == 1
    with closing(sqlite3.connect(target)) as db:
        assert lexical_index.terms_of(db, source_id) == indexed_terms(source.event)
