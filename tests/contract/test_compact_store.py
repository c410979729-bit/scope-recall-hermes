"""Offline compaction preserves SQLite authority and refuses unsafe publication."""

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime

import pytest
from scope_recall.core.backup_primitives import BackupError, sha256
from scope_recall.core.storage import Transaction
from scope_recall.core.writer_lease import TruthWriterLease
from scope_recall.maintenance import compact_store
from scope_recall.maintenance.cli import main
from test_claims import app, capture, initial
from test_deletion import authorize, request

OLD = "2000-01-01T00:00:00Z"
LARGE = json.dumps({"process": "SYNTHETIC-RETIRED-PROCESS " * 2000})


def _fragment(db, source, *, state, outcome=None, when=OLD, payload=LARGE):
    key = db.execute(
        """INSERT INTO work_items(work_type,subject_ref,subject_revision,scope_id,project_id,branch_id,
        state,available_at)
        VALUES('consolidate',?,?,?,?,?,?,?) RETURNING work_id""",
        (source.ref, source.revision, source.scope_id, source.project_id, source.branch_id, state, when),
    ).fetchone()[0]
    db.execute("INSERT INTO consolidation_fragments VALUES (?,?,?,?,?)", (key, 0, 32, 64, payload))
    if outcome is not None:
        db.execute("INSERT INTO consolidation_outcomes VALUES (?,?,?,?)", (key, outcome, "scope kept", when))
    return key


def _evaluation(db, claim, fingerprint, refs, *, reason="superseded_by_new_evidence", attempted=None):
    return db.execute(
        """INSERT INTO candidate_evaluations(candidate_ref,candidate_revision,evidence_fingerprint,evidence_refs_json,
        rule_version,memory_epoch,state,reason,created_at,completed_at,model_attempted_at)
        VALUES (?,?,?,?,'test',0,'obsolete',?,?,?,?) RETURNING evaluation_id""",
        (claim.ref, claim.revision, fingerprint, json.dumps(refs), reason, OLD, OLD, attempted),
    ).fetchone()[0]


def _fixture(app):
    core, ctx = app
    claim, evidence = initial(core, ctx)
    deleted = capture(core, ctx, "TEST original withdrawn")
    authorize(core, ctx, deleted)
    assert core.forget(ctx, request(deleted), remaining_seconds=10)["read_blocked"]
    sources = [capture(core, ctx, f"TEST background task {i}", origin="tool_observation") for i in range(6)]
    now = datetime.now(UTC).isoformat()
    path = core.storage.path
    with closing(sqlite3.connect(path)) as db, db:
        retired = _fragment(db, sources[0], state="done", outcome="complete")
        _fragment(db, sources[1], state="done", outcome="partial", when=now)
        _fragment(db, sources[2], state="pending", outcome="complete")
        _fragment(db, sources[3], state="failed", outcome="partial")
        _fragment(db, sources[4], state="done")
        _fragment(db, sources[5], state="done", outcome="partial", payload="{}")
        db.execute("INSERT INTO instance_scopes VALUES ('TEST-other-scope')")
        db.execute("UPDATE source_events SET scope_id='TEST-other-scope' WHERE event_id=?", (sources[0].ref,))
        db.execute("UPDATE work_items SET scope_id='TEST-other-scope' WHERE subject_ref=?", (sources[0].ref,))
        db.execute(
            """INSERT OR REPLACE INTO candidate_lifecycle(candidate_ref,candidate_revision,scope_id,project_id,
            branch_id,processing_state,reason,rule_version,evidence_fingerprint,created_at,updated_at)
            VALUES(?,?,?,?,?,'waiting_evidence','unknown','test','current',?,?)""",
            (claim.ref, claim.revision, evidence.scope_id, evidence.project_id, evidence.branch_id, OLD, OLD),
        )
        db.execute(
            "INSERT OR IGNORE INTO candidate_evidence VALUES (?,?,?,?,?)",
            (claim.ref, claim.revision, evidence.ref, evidence.revision, OLD),
        )
        refs = [f"{evidence.ref}@{evidence.revision}"]
        obsolete = _evaluation(db, claim, "old-question", refs)
        _evaluation(db, claim, "old-verdict", refs, attempted=OLD)
        _evaluation(db, claim, "old-block", refs, reason="authority_revoked")
        _evaluation(db, claim, "old-missing", [f"{deleted.ref}@{deleted.revision}"])
        _evaluation(db, claim, "current", refs)
    return path, retired, obsolete, evidence, deleted, sources[0]


def _tables(db):
    return {
        row[0] for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    }


def test_dry_run_and_apply_trim_just_proven_retired_payloads(app, tmp_path, capsys):
    source, retired, obsolete, evidence, deleted, other_scope = _fixture(app)
    target = tmp_path / "compacted.sqlite3"
    before = sha256(source)
    assert main(["compact-store", "--source", str(source), "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["consolidation_fragments"] == {"rows": 1, "payload_bytes": len(LARGE.encode())}
    assert preview["obsolete_evaluations"]["rows"] == 1
    assert sha256(source) == before and not target.exists()
    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(target)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["consolidation_fragments"] == preview["consolidation_fragments"]
    assert receipt["obsolete_evaluations"] == preview["obsolete_evaluations"]
    assert receipt["output_bytes"] == target.stat().st_size < preview["source_bytes"]
    assert sha256(source) == before

    with closing(sqlite3.connect(source)) as original, closing(sqlite3.connect(target)) as copy:
        assert copy.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert not copy.execute("PRAGMA foreign_key_check").fetchall()
        assert not copy.execute("SELECT 1 FROM consolidation_fragments WHERE work_id=?", (retired,)).fetchone()
        assert copy.execute("SELECT count(*) FROM consolidation_fragments").fetchone() == (5,)
        saved = "SELECT * FROM candidate_evaluations WHERE evaluation_id=?"
        old = list(original.execute(saved, (obsolete,)).fetchone())
        old[4] = "[]"
        assert copy.execute(saved, (obsolete,)).fetchone() == tuple(old)
        trial = "SELECT * FROM candidate_evaluations WHERE evaluation_id<>? ORDER BY evaluation_id"
        assert copy.execute(trial, (obsolete,)).fetchall() == original.execute(trial, (obsolete,)).fetchall()
        assert _tables(copy) == _tables(original)
        for table in _tables(original) - {"candidate_evaluations", "consolidation_fragments"}:
            query = f'SELECT * FROM "{table}"'
            assert set(original.execute(query)) == set(copy.execute(query)), table
        assert list(original.execute("SELECT rowid,event_id FROM source_events")) == list(
            copy.execute("SELECT rowid,event_id FROM source_events")
        )
        copy.row_factory = sqlite3.Row
        tx = Transaction(copy, app[1], writable=False)
        assert tx.source(evidence.ref, evidence.revision).event["content"] == evidence.event["content"]
        assert tx.source(deleted.ref, deleted.revision) is None
        assert tx.source(other_scope.ref, other_scope.revision) is None


def test_existing_output_or_source_alias_is_refused_without_changes(app, tmp_path, capsys):
    source = app[0].storage.path
    output = tmp_path / "existing.sqlite3"
    output.write_bytes(b"TEST existing output")
    before = sha256(source)
    for target in (source, output):
        assert main(["compact-store", "--source", str(source), "--apply", "--output", str(target)]) == 2
        assert "already exists" in capsys.readouterr().out
    assert sha256(source) == before
    assert output.read_bytes() == b"TEST existing output"


def test_live_writer_and_sqlite_sidecar_are_refused(app, tmp_path, capsys):
    source = app[0].storage.path
    output = tmp_path / "live-source.sqlite3"
    apply = ["compact-store", "--source", str(source), "--apply", "--output", str(output)]
    lease = TruthWriterLease(source.parent, role="provider")
    assert lease.acquire()["status"] == "acquired"
    try:
        assert main(apply) == 2
        assert "store is busy" in capsys.readouterr().out
    finally:
        lease.release()
    before = sha256(source)
    with closing(sqlite3.connect(source)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE instance_meta SET storage_revision=storage_revision+1")
        assert main(apply) == 2
        assert "SQLite sidecars" in capsys.readouterr().out
    assert not output.exists() and sha256(source) == before


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("mode", ["IMMEDIATE", "EXCLUSIVE"])
def test_rollback_journal_transactions_refuse_compaction(app, tmp_path, capsys, apply, mode):
    source = app[0].storage.path
    output = tmp_path / "locked-source.sqlite3"
    args = ["compact-store", "--source", str(source)]
    args += ["--apply", "--output", str(output)] if apply else ["--dry-run"]
    with closing(sqlite3.connect(source)) as writer:
        assert writer.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
        before = sha256(source)
        writer.execute(f"BEGIN {mode}")
        assert not list(source.parent.glob("memory.sqlite3-*"))
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=0)) as reader:
            if mode == "EXCLUSIVE":
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    reader.execute("SELECT * FROM instance_meta").fetchall()
            else:
                assert reader.execute("SELECT * FROM instance_meta").fetchall()
        assert main(args) == 2
        assert "store is busy" in capsys.readouterr().out
        assert not output.exists() and not list(tmp_path.glob(".scope-recall-compact-*.sqlite3*"))
        assert not list(source.parent.glob("memory.sqlite3-*")) and sha256(source) == before


@pytest.mark.parametrize("mode", ["DELETE", "WAL"])
def test_offline_pager_lock_blocks_competing_read_and_write(app, tmp_path, capsys, monkeypatch, mode):
    source = app[0].storage.path
    output = tmp_path / "locked-through-publish.sqlite3"
    with closing(sqlite3.connect(source)) as setup:
        assert setup.execute(f"PRAGMA journal_mode={mode}").fetchone() == (mode.lower(),)
    before = sha256(source)
    measure = compact_store._measure

    def check_gate(reader, params):
        for statement in ("SELECT * FROM instance_meta", "BEGIN IMMEDIATE"):
            with closing(sqlite3.connect(source, timeout=0)) as peer:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    peer.execute(statement).fetchall()
        assert not list(source.parent.glob("memory.sqlite3-*"))
        return measure(reader, params)

    monkeypatch.setattr(compact_store, "_measure", check_gate)
    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "compacted"
    assert output.exists() and sha256(source) == before
    assert not list(source.parent.glob("memory.sqlite3-*"))
    with closing(sqlite3.connect(source)) as peer:
        assert peer.execute("SELECT * FROM instance_meta").fetchall()


def test_compaction_drops_the_staged_copy_if_the_pager_lock_is_lost(app, tmp_path, capsys, monkeypatch):
    source = app[0].storage.path
    output = tmp_path / "lost-lock.sqlite3"
    before = sha256(source)
    exclusive = compact_store._exclusive_pager
    trim = compact_store._trim_and_vacuum

    @contextmanager
    def exit_after_trim(path, info):
        with exclusive(path, info) as owner:

            def trim_then_exit(conn, params, measured):
                trim(conn, params, measured)
                owner.terminate()
                owner.wait(timeout=5)

            monkeypatch.setattr(compact_store, "_trim_and_vacuum", trim_then_exit)
            yield owner

    monkeypatch.setattr(compact_store, "_exclusive_pager", exit_after_trim)
    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(output)]) == 2
    assert "exclusive SQLite file lock was lost" in capsys.readouterr().out
    assert not output.exists() and not list(tmp_path.glob(".scope-recall-compact-*.sqlite3*"))
    assert sha256(source) == before


def test_verification_failure_leaves_no_published_or_staging_file(app, tmp_path, capsys, monkeypatch):
    source = app[0].storage.path
    output = tmp_path / "failed.sqlite3"
    before = sha256(source)

    def fail(_conn):
        raise BackupError("injected verification failure")

    monkeypatch.setattr(compact_store, "_check_store", fail)
    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(output)]) == 2
    assert "verification failure" in capsys.readouterr().out
    assert not output.exists() and not list(tmp_path.glob(".scope-recall-compact-*.sqlite3*"))
    assert sha256(source) == before
