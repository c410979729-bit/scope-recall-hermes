"""Offline compaction preserves SQLite authority and refuses unsafe publication."""

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
        _evaluation(db, claim, "old-question", refs)
        _evaluation(db, claim, "old-verdict", refs, attempted=OLD)
        _evaluation(db, claim, "old-block", refs, reason="authority_revoked")
        _evaluation(db, claim, "old-missing", [f"{deleted.ref}@{deleted.revision}"])
        _evaluation(db, claim, "current", refs)
    return path, retired, evidence, deleted, sources[0]


def _tables(db):
    return {
        row[0] for row in db.execute("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    }


def test_dry_run_and_apply_trim_just_proven_retired_payloads(app, tmp_path, capsys):
    source, retired, evidence, deleted, other_scope = _fixture(app)
    target = tmp_path / "compacted.sqlite3"
    before = sha256(source)
    assert main(["compact-store", "--source", str(source), "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["consolidation_fragments"] == {"rows": 1, "estimated_bytes": len(LARGE.encode())}
    assert preview["candidate_evaluations"]["rows"] == 0
    assert sha256(source) == before and not target.exists()
    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(target)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["consolidation_fragments"] == preview["consolidation_fragments"]
    assert receipt["candidate_evaluations"] == preview["candidate_evaluations"]
    assert receipt["output_bytes"] == target.stat().st_size < preview["source_bytes"]
    assert sha256(source) == before

    with closing(sqlite3.connect(source)) as original, closing(sqlite3.connect(target)) as copy:
        assert copy.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert not copy.execute("PRAGMA foreign_key_check").fetchall()
        assert not copy.execute("SELECT 1 FROM consolidation_fragments WHERE work_id=?", (retired,)).fetchone()
        assert copy.execute("SELECT count(*) FROM consolidation_fragments").fetchone() == (5,)
        assert _tables(copy) == _tables(original)
        for table in _tables(original) - {"consolidation_fragments"}:
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


def test_online_dry_run_reads_committed_wal_without_store_locks(app, capsys):
    core, ctx = app
    source = capture(core, ctx, "TEST committed WAL fragment", origin="tool_observation")
    path = core.storage.path
    wal = path.with_name(path.name + "-wal")
    lease = TruthWriterLease(path.parent, role="provider")
    assert lease.acquire()["status"] == "acquired"
    try:
        with closing(sqlite3.connect(path)) as writer:
            assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
            writer.execute("PRAGMA wal_autocheckpoint=0")
            with writer:
                _fragment(writer, source, state="done", outcome="complete")
            before = (sha256(path), sha256(wal))
            assert main(["compact-store", "--source", str(path), "--dry-run"]) == 0
            preview = json.loads(capsys.readouterr().out)
            assert preview["retention_days"] == 30
            assert preview["consolidation_fragments"]["rows"] == 1
            assert (sha256(path), sha256(wal)) == before
    finally:
        lease.release()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_preview_leaves_an_incomplete_wal_pair_untouched(app, suffix):
    path = app[0].storage.path
    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    journal = path.with_name(path.name + suffix)
    journal.write_bytes(b"TEST incomplete journal")
    before = {file.name: sha256(file) for file in (path, journal)}
    with pytest.raises(sqlite3.OperationalError, match="incomplete"):
        compact_store.compact_store(path)
    after = {file.name: sha256(file) for file in path.parent.glob(path.name + "*")}
    assert after == before


def test_last_wal_writer_can_close_while_preview_opens(app, monkeypatch):
    path = app[0].storage.path
    native_connect = sqlite3.connect
    writer = native_connect(path)
    assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    with writer:
        writer.execute("UPDATE instance_meta SET storage_revision=storage_revision+1")
    seen = {}

    def close_writer_then_open(*args, **kwargs):
        writer.close()
        seen.update({f: f.exists() for f in (path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm"))})
        return native_connect(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(sqlite3, "connect", close_writer_then_open)
        compact_store.compact_store(path)
    assert seen and all(was_present or not f.exists() for f, was_present in seen.items())


@pytest.mark.parametrize("suffixes", [("-journal",), ("-wal", "-shm")])
def test_read_preview_does_not_initialize_unrecovered_wal_sidecars(app, suffixes):
    path = app[0].storage.path
    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    for suffix in suffixes:
        path.with_name(path.name + suffix).touch()
    before = {f.name: sha256(f) for f in path.parent.glob(path.name + "*")}
    try:
        compact_store.compact_store(path)
    except sqlite3.OperationalError:
        pass  # SQLite can decline an unrecovered WAL without initializing it.
    assert {f.name: sha256(f) for f in path.parent.glob(path.name + "*")} == before


def test_online_dry_run_uses_one_snapshot_while_wal_writer_commits(app, monkeypatch):
    core, ctx = app
    source = capture(core, ctx, "TEST commit while measuring", origin="tool_observation")
    path = core.storage.path
    measure = compact_store._measure
    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("PRAGMA busy_timeout=0")
        with writer:
            writer.execute("UPDATE instance_meta SET storage_revision=storage_revision+1")

        def measure_after_commit(reader, params):
            assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
            with writer:
                _fragment(writer, source, state="done", outcome="complete")
            assert writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() == (0, 0, 0)
            return measure(reader, params)

        with monkeypatch.context() as context:
            context.setattr(compact_store, "_measure", measure_after_commit)
            preview = compact_store.compact_store(path)
        next_preview = compact_store.compact_store(path)
    assert preview["consolidation_fragments"]["rows"] == 0
    assert next_preview["consolidation_fragments"]["rows"] == 1


def test_compact_cli_writes_a_utf8_receipt_on_a_legacy_pipe(app, tmp_path):
    source = app[0].storage.path
    target = tmp_path / "项目-🧭.sqlite3"
    root = Path(__file__).resolve().parents[2]
    boot = (
        "import sys,types; package=types.ModuleType('scope_recall'); "
        "package.__path__=[sys.argv.pop(1)]; sys.modules['scope_recall']=package; "
        "from scope_recall.maintenance.cli import main; raise SystemExit(main())"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            boot,
            str(root),
            "compact-store",
            "--source",
            str(source),
            "--apply",
            "--output",
            str(target),
        ],
        env={**os.environ, "PYTHONUTF8": "0", "PYTHONIOENCODING": "cp936"},
        capture_output=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.decode("utf-8"))
    assert receipt["output"] == str(target)
    assert receipt["output_bytes"] == target.stat().st_size


@pytest.mark.parametrize("fail", [False, True])
def test_disk_preview_releases_its_private_copy(app, tmp_path, monkeypatch, fail):
    from scope_recall.maintenance import store_snapshot

    source = app[0].storage.path
    before = sha256(source)
    monkeypatch.setattr(store_snapshot, "_MEMORY_SNAPSHOT_BYTES", 0)
    monkeypatch.setattr(store_snapshot.tempfile, "tempdir", str(tmp_path))
    error = None
    try:
        with store_snapshot.detached_store_snapshot(source, compact_store._verify_schema) as copy:
            assert copy.execute("PRAGMA query_only").fetchone()[0] == 1
            assert len(list(tmp_path.glob("scope-recall-preview-*/snapshot.sqlite3"))) == 1
            if fail:
                raise ValueError("synthetic measurement interrupted")
    except ValueError as exc:
        error = exc
    assert bool(error) == fail
    assert list(tmp_path.glob("scope-recall-preview-*")) == []
    assert sha256(source) == before


def test_disk_preview_accepts_the_system_temp_directory_alias(app, tmp_path, monkeypatch):
    from scope_recall.maintenance import store_snapshot

    real = tmp_path / "physical"
    real.mkdir()
    alias = tmp_path / "system-temp"
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(real), str(alias))
    else:
        alias.symlink_to(real, target_is_directory=True)
    try:
        monkeypatch.setattr(store_snapshot.tempfile, "tempdir", str(alias))
        monkeypatch.setattr(store_snapshot, "_MEMORY_SNAPSHOT_BYTES", 0)
        assert compact_store.compact_store(app[0].storage.path)["status"] == "dry_run"
        assert list(real.iterdir()) == []
    finally:
        if os.name == "nt":
            alias.rmdir()
        else:
            alias.unlink()


def test_retention_cutoff_defaults_to_30_days_and_zero_includes_recent_retired_work(app, capsys):
    core, ctx = app
    path = core.storage.path
    now = datetime.now(UTC)
    sources = [capture(core, ctx, f"TEST timestamp {days}", origin="tool_observation") for days in (31, 29, -1)]
    unfinished = capture(core, ctx, "TEST unfinished", origin="tool_observation")
    with closing(sqlite3.connect(path)) as db, db:
        for source, days in zip(sources, (31, 29, -1), strict=True):
            _fragment(db, source, state="done", outcome="complete", when=(now - timedelta(days=days)).isoformat())
        _fragment(db, unfinished, state="pending", outcome="complete")
    args = ["compact-store", "--source", str(path), "--dry-run"]
    for flags, retained, rows in (([], 30, 1), (["--retention-days", "0"], 0, 2)):
        assert main([*args, *flags]) == 0
        receipt = json.loads(capsys.readouterr().out)
        assert receipt["retention_days"] == retained
        assert receipt["consolidation_fragments"]["rows"] == rows
    assert main([*args, "--retention-days", "-1"]) == 2
    assert "between 0 and 36500" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("apply", [False, True])
@pytest.mark.parametrize("mode", ["IMMEDIATE", "EXCLUSIVE"])
def test_rollback_journal_only_allows_dry_run_with_a_reserved_lock(app, tmp_path, capsys, apply, mode):
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
        assert main(args) == (0 if not apply and mode == "IMMEDIATE" else 2)
        status = json.loads(capsys.readouterr().out)["status"]
        assert status == ("dry_run" if not apply and mode == "IMMEDIATE" else "error")
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


def test_compaction_never_publishes_if_the_pager_dies_after_the_parent_source_check(app, tmp_path, capsys, monkeypatch):
    source = app[0].storage.path
    output = tmp_path / "late-writer.sqlite3"
    with closing(sqlite3.connect(source)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
        revision = db.execute("SELECT storage_revision FROM instance_meta").fetchone()[0]
    exclusive = compact_store._exclusive_pager
    unchanged = compact_store._unchanged
    writer_committed = False

    @contextmanager
    def stop_at_publish(path, info):
        with exclusive(path, info) as owner:

            def check_then_stop(source_path, before):
                nonlocal writer_committed
                unchanged(source_path, before)
                if not list(tmp_path.glob(".scope-recall-compact-*.sqlite3")):
                    return
                owner.terminate()
                owner.wait(timeout=5)
                with closing(sqlite3.connect(source, timeout=0)) as writer, writer:
                    writer.execute("BEGIN IMMEDIATE")
                    writer.execute("UPDATE instance_meta SET storage_revision=storage_revision+1")
                writer_committed = True

            monkeypatch.setattr(compact_store, "_unchanged", check_then_stop)
            yield owner

    monkeypatch.setattr(compact_store, "_exclusive_pager", stop_at_publish)
    result = main(["compact-store", "--source", str(source), "--apply", "--output", str(output)])
    assert writer_committed
    with closing(sqlite3.connect(source)) as db:
        assert db.execute("SELECT storage_revision FROM instance_meta").fetchone() == (revision + 1,)
    assert result == 2
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert not output.exists() and not list(tmp_path.glob(".scope-recall-compact-*.sqlite3*"))


def test_compaction_verifies_the_output_if_the_pager_exits_after_linking(app, tmp_path, capsys, monkeypatch):
    source = app[0].storage.path
    output = tmp_path / "linked-before-exit.sqlite3"
    before = sha256(source)
    shim = tmp_path / "pager-exit-after-link.py"
    shim.write_text(
        "import os, runpy, sys\n"
        f"pager = runpy.run_path({str(compact_store.SCRIPT_PATH)!r})\n"
        "link = os.link\n"
        "def link_and_exit(*args, **kwargs):\n"
        "    link(*args, **kwargs)\n"
        "    os._exit(79)\n"
        "os.link = link_and_exit\n"
        "raise SystemExit(pager['_main'](sys.argv[1:]))\n",
        encoding="utf-8",
    )
    exclusive = compact_store._exclusive_pager
    exit_codes = []

    @contextmanager
    def record_exit(path, info):
        with exclusive(path, info) as owner:
            yield owner
            exit_codes.append(owner.returncode)

    monkeypatch.setattr(compact_store, "SCRIPT_PATH", shim)
    monkeypatch.setattr(compact_store, "_exclusive_pager", record_exit)
    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(output)]) == 0
    assert exit_codes == [79]
    assert json.loads(capsys.readouterr().out)["status"] == "compacted"
    with closing(sqlite3.connect(output)) as copy:
        assert copy.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    assert not list(tmp_path.glob(".scope-recall-compact-*.sqlite3*")) and sha256(source) == before


def test_pager_ignores_a_publication_frame_interrupted_by_parent_exit(app, tmp_path):
    source = app[0].storage.path
    before = sha256(source)
    stage = tmp_path / f"{compact_store.STAGE_PREFIX}interrupted.sqlite3"
    stage.write_bytes(b"TEST staged bytes")
    output = tmp_path / "partial-request.sqlite3"
    request = json.dumps((str(stage), str(output), compact_store.file_revision(stage.stat()))).encode("ascii")
    with compact_store._exclusive_pager(source, source.stat()) as owner:
        assert owner.stdin is not None
        with owner.stdin as pipe:
            pipe.write(request)
        assert owner.wait(timeout=5) == 2
    assert not output.exists() and sha256(source) == before


def test_compaction_preserves_an_output_created_during_publication(app, tmp_path, capsys, monkeypatch):
    source = app[0].storage.path
    output = tmp_path / "late-output.sqlite3"
    before = sha256(source)
    unchanged = compact_store._unchanged

    def check_and_create_output(path, info):
        unchanged(path, info)
        if list(tmp_path.glob(".scope-recall-compact-*.sqlite3")):
            output.write_bytes(b"TEST independently published data")

    monkeypatch.setattr(compact_store, "_unchanged", check_and_create_output)
    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(output)]) == 2
    assert "output path is occupied" in capsys.readouterr().out
    assert output.read_bytes() == b"TEST independently published data"
    assert not list(tmp_path.glob(".scope-recall-compact-*.sqlite3*")) and sha256(source) == before


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


@pytest.mark.parametrize("host", ["hermes", "shared"])
def test_documented_backup_and_compaction_commands_complete_end_to_end(tmp_path, host, capsys):
    import shlex

    from scope_recall.adapters.hermes.installation import build_installation_manifest, install_hermes_scope_recall
    from scope_recall.adapters.hermes.shared_entries import (
        attach_shared_record,
        read_shared_payload,
        shared_entry_record,
    )
    from scope_recall.maintenance.shared import init_shared, store_binding

    documentation = Path(__file__).resolve().parents[2] / "docs" / "backup-restore.md"
    commands = [
        line for line in documentation.read_text(encoding="utf-8").splitlines() if line.startswith("scope-recall ")
    ]
    home = tmp_path / "original"
    if host == "shared":
        init_shared(root=home, agent_id="agent1", test_mode=True)
        local = build_installation_manifest(tmp_path / "agent1", agent_id="agent1", test_mode=True)
        moment = "2026-10-23T00:00:00Z"
        record = shared_entry_record(local, entry_id="agent1", display_name="agent1", attached_at=moment)
        attach_shared_record(home, record, now=moment)
        payload = read_shared_payload(home)
        binding = store_binding(payload, home, frozenset(payload["scope_ids"]))
    else:
        binding, _core = install_hermes_scope_recall(home, agent_id="agent1", test_mode=True)
    archive = tmp_path / "archive"
    archive.mkdir()
    selected = [commands[0 if host == "shared" else 1], *commands[2:]]
    replacements = {
        "/path/to/shared-store": str(home),
        "/path/to/agent1": str(home),
        "/path/to/backups/memory.zip": str(tmp_path / "backups" / "memory.zip"),
        "/path/to/recovered": str(tmp_path / "recovered"),
        "/path/to/memory.sqlite3": str(binding.data_directory / "memory.sqlite3"),
        "/path/to/archive/compacted.sqlite3": str(archive / "compacted.sqlite3"),
    }
    receipts = []
    for command in selected:
        arguments = [replacements.get(value, value) for value in shlex.split(command)[1:]]
        assert main(arguments) == 0
        receipts.append(json.loads(capsys.readouterr().out))
    assert [item["status"] for item in receipts] == [
        "created",
        "verified",
        "verified",
        "planned",
        "restored",
        "dry_run",
        "compacted",
    ]
    assert (archive / "compacted.sqlite3").is_file()


def test_dry_run_and_apply_check_disk_space_and_record_source_sha256(app, tmp_path, capsys, monkeypatch):
    from collections import namedtuple

    from scope_recall.maintenance import store_snapshot

    Usage = namedtuple("Usage", ["total", "used", "free"])
    source = app[0].storage.path
    custom_temp = tmp_path / "custom-preview-temp"
    custom_temp.mkdir()
    monkeypatch.setattr(store_snapshot, "_MEMORY_SNAPSHOT_BYTES", 0)

    seen_dirs = []
    real_mkdtemp = store_snapshot.tempfile.mkdtemp

    def track_mkdtemp(*args, **kwargs):
        seen_dirs.append(Path(kwargs["dir"]))
        return real_mkdtemp(*args, **kwargs)

    monkeypatch.setattr(store_snapshot.tempfile, "mkdtemp", track_mkdtemp)
    assert main(["compact-store", "--source", str(source), "--dry-run", "--temp-dir", str(custom_temp)]) == 0
    assert seen_dirs == [custom_temp.resolve()]
    capsys.readouterr()

    missing_temp = tmp_path / "does-not-exist"
    assert main(["compact-store", "--source", str(source), "--dry-run", "--temp-dir", str(missing_temp)]) == 2
    assert "existing directory" in json.loads(capsys.readouterr().out)["error"]

    output = tmp_path / "compacted-out.sqlite3"
    assert (
        main(
            [
                "compact-store",
                "--source",
                str(source),
                "--apply",
                "--output",
                str(output),
                "--temp-dir",
                str(custom_temp),
            ]
        )
        == 2
    )
    assert "--temp-dir is only available with --dry-run" in json.loads(capsys.readouterr().out)["error"]

    with monkeypatch.context() as low_disk:
        low_disk.setattr(store_snapshot.shutil, "disk_usage", lambda _path: Usage(1024, 1000, 24))
        assert main(["compact-store", "--source", str(source), "--dry-run", "--temp-dir", str(custom_temp)]) == 2
        assert "insufficient free disk space" in json.loads(capsys.readouterr().out)["error"]
        assert main(["compact-store", "--source", str(source), "--apply", "--output", str(output)]) == 2
        assert "insufficient free disk space" in json.loads(capsys.readouterr().out)["error"]
        assert not output.exists()
        assert not list(tmp_path.glob(".scope-recall-compact-*.sqlite3*"))

    assert main(["compact-store", "--source", str(source), "--apply", "--output", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["source_sha256"] == sha256(source)
    assert not list(tmp_path.glob(".scope-recall-work-*"))
    assert not list(tmp_path.glob(".scope-recall-compact-*"))


def test_sidecar_refusal_warns_never_to_delete_wal_and_cli_surfaces_snapshot_diagnostics(app, tmp_path, capsys):
    path = app[0].storage.path
    output = tmp_path / "blocked-by-wal.sqlite3"
    with closing(sqlite3.connect(path)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    wal = path.with_name(path.name + "-wal")
    wal.write_bytes(b"TEST uncheckpointed wal")
    try:
        assert main(["compact-store", "--source", str(path), "--apply", "--output", str(output)]) == 2
        apply_err = json.loads(capsys.readouterr().out)["error"]
        assert "SQLite sidecars" in apply_err
        assert "never delete -wal or -journal files" in apply_err

        assert main(["compact-store", "--source", str(path), "--dry-run"]) == 2
        preview_err = json.loads(capsys.readouterr().out)["error"]
        assert "SQLite WAL and shared memory are incomplete; retry the preview" in preview_err
    finally:
        wal.unlink(missing_ok=True)
