"""Copy an offline store and trim only superseded processing payloads."""

from __future__ import annotations

import argparse
import os
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from ..core.backup_primitives import BackupError, create_output, exclusive_store, first_link, safe_path
from ..core.file_access import preserve_file_access
from ..core.schema import APPLICATION_ID, SCHEMA_VERSION
from ..core.writer_lease import TRUTH_WRITER_LEASE_FILENAME, TRUTH_WRITER_LEASE_INFO_FILENAME

RETENTION_DAYS = 180
_LARGE_FRAGMENT_BYTES = 4096
_STORE_LOCKS = (
    TRUTH_WRITER_LEASE_FILENAME,
    TRUTH_WRITER_LEASE_INFO_FILENAME,
    "scope-recall-retained.lock",
    "runtime-worker.lock",
)

# Keep the pager lock in a separate process: POSIX record locks belong to a
# process and would otherwise merge with SQLite's own locks in that process.
# The Windows VFS locks the reserved 512-byte range at the 1 GiB boundary.
# A POSIX whole-file record lock covers that range even beyond the file's EOF.
# POSIX needs a writable descriptor for that lock. The holder never writes.
_PAGER_LOCK_PROCESS = """
import os
import sys

try:
    with open(sys.argv[1], 'rb' if os.name == 'nt' else 'r+b', buffering=0) as source:
        info = os.fstat(source.fileno())
        if (info.st_dev, info.st_ino) != tuple(map(int, sys.argv[2:])):
            sys.exit(2)
        if os.name == 'nt':
            import msvcrt
            source.seek(1073741824)
            msvcrt.locking(source.fileno(), msvcrt.LK_NBLCK, 512)
        elif os.name == 'posix':
            import fcntl
            fcntl.lockf(source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            sys.exit(2)
        sys.stdout.buffer.write(b'1')
        sys.stdout.buffer.flush()
        sys.stdin.buffer.read(1)
except OSError:
    sys.exit(2)
"""

# A partially processed or failed item can resume from its staged fragments.
# Require both a final outcome and a done, unleased work item before trimming.
_ENDED_FRAGMENTS = """
    SELECT f.work_id,f.start_offset,length(CAST(f.proposals_json AS BLOB)) AS bytes
    FROM consolidation_fragments f JOIN consolidation_outcomes o USING(work_id) JOIN work_items w USING(work_id)
    WHERE w.work_type='consolidate' AND w.state='done' AND w.lease_owner IS NULL AND w.lease_until IS NULL
      AND o.disposition IN ('complete','partial')
      AND julianday(o.recorded_at)<julianday(:cutoff) AND julianday(w.available_at)<julianday(:cutoff)
      AND length(CAST(f.proposals_json AS BLOB))>=:large_fragment_bytes
"""

# Keep attempts, the unique fingerprint and work receipt. A question made
# obsolete by suppression can be requeued, and one judged by a model still
# governs the candidate's retry budget and whether new evidence is unjudged.
_SUPERSEDED_QUESTIONS = """
    SELECT e.evaluation_id,length(CAST(e.evidence_refs_json AS BLOB))-2 AS bytes
    FROM candidate_evaluations e
    JOIN candidate_lifecycle l USING(candidate_ref,candidate_revision)
    LEFT JOIN work_items w ON w.work_id=e.work_id
    WHERE e.state='obsolete' AND e.reason='superseded_by_new_evidence'
      AND e.model_attempted_at IS NULL AND e.result_digest IS NULL AND e.failure_code IS NULL
      AND julianday(e.completed_at)<julianday(:cutoff) AND julianday(e.created_at)<julianday(:cutoff)
      AND (e.work_id IS NULL OR (
        w.work_type='evaluate_candidate' AND w.state='obsolete' AND w.last_error_code IS NULL
        AND w.lease_owner IS NULL AND w.lease_until IS NULL AND w.subject_ref='candidate:'||e.candidate_ref
        AND w.subject_revision=e.evaluation_id AND w.scope_id=l.scope_id
        AND w.project_id IS l.project_id AND w.branch_id IS l.branch_id))
      AND e.evidence_fingerprint<>l.evidence_fingerprint
      AND json_type(e.evidence_refs_json)='array' AND json_array_length(e.evidence_refs_json)>0
      AND EXISTS (SELECT 1 FROM candidate_evaluations newer
        WHERE newer.candidate_ref=e.candidate_ref AND newer.candidate_revision=e.candidate_revision
          AND newer.evaluation_id>e.evaluation_id AND newer.evidence_fingerprint<>e.evidence_fingerprint)
      AND NOT EXISTS (
        SELECT 1 FROM json_each(e.evidence_refs_json) ref
        WHERE ref.type<>'text' OR NOT EXISTS (
          SELECT 1 FROM candidate_evidence kept
          WHERE kept.candidate_ref=e.candidate_ref AND kept.candidate_revision=e.candidate_revision
            AND kept.source_ref=substr(ref.value,1,instr(ref.value,'@')-1)
            AND kept.source_revision=CAST(substr(ref.value,instr(ref.value,'@')+1) AS INTEGER)
            AND ref.value=kept.source_ref||'@'||kept.source_revision))
"""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare an explicit source, a mode, and a fresh output for apply."""
    parser.add_argument("--source", required=True, help="path to the offline Scope Recall SQLite database")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="count eligible payloads without changing the database")
    mode.add_argument("--apply", action="store_true", help="write a separate verified, vacuumed SQLite file")
    parser.add_argument("--output", help="new SQLite file to create in an existing folder; required for --apply")
    parser.add_argument("--retention-days", type=int, default=RETENTION_DAYS, help="minimum payload age (default: 180)")


def _sidecars(path: Path) -> tuple[Path, ...]:
    return tuple(path.with_name(path.name + suffix) for suffix in ("-wal", "-shm", "-journal"))


def _offline_file(path: Path) -> os.stat_result:
    """Refuse live or unrecovered journals and aliases that evade the directory lease."""
    if any(os.path.lexists(candidate) for candidate in _sidecars(path)):
        raise BackupError("store is busy or needs recovery: close it and clear its SQLite sidecars normally")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise BackupError("source must be a regular SQLite file with no other hard links")
    return info


@contextmanager
def _exclusive_pager(source: Path, before: os.stat_result):
    """Hold SQLite's native file lock from before inspection through publication."""
    argv = [sys.executable, "-IBS", "-c", _PAGER_LOCK_PROCESS, str(source), str(before.st_dev), str(before.st_ino)]
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as owner:
        if owner.stdout is None or owner.stdout.read(1) != b"1":
            raise BackupError("store is busy or the SQLite file could not be locked exclusively")
        yield owner


@contextmanager
def _read_store(source: Path):
    """Hold the runtime locks through the final source check and output publication."""
    if any(first_link(source.parent / name) is not None for name in _STORE_LOCKS):
        raise BackupError("store lock files must not be symlinks or reparse points")
    with exclusive_store(source.parent, busy=lambda _reason: BackupError("store is busy: stop all store processes")):
        before = _offline_file(source)
        with _exclusive_pager(source, before) as owner:

            def check_source() -> None:
                if owner.poll() is not None:
                    raise BackupError("exclusive SQLite file lock was lost")
                _unchanged(source, before)

            check_source()
            # The native pager lock excludes independent SQLite readers and
            # writers. This handle can read without opening source journals.
            with closing(sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)) as reader:
                _verify_schema(reader)
                yield reader, before, check_source


def _unchanged(source: Path, before: os.stat_result) -> None:
    after = _offline_file(source)
    same_data = (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)
    if not os.path.samestat(before, after) or not same_data:
        raise BackupError("source changed during compaction")


def _verify_schema(conn: sqlite3.Connection) -> None:
    header = (
        conn.execute("PRAGMA application_id").fetchone()[0],
        conn.execute("PRAGMA user_version").fetchone()[0],
    )
    if header != (APPLICATION_ID, SCHEMA_VERSION) or conn.execute(
        "SELECT schema_version FROM instance_meta WHERE singleton=1"
    ).fetchone() != (SCHEMA_VERSION,):
        raise BackupError("compact-store requires a current Scope Recall schema")


def _check_store(conn: sqlite3.Connection) -> None:
    """Refuse to publish a copy with a bad header, structural damage or broken foreign keys."""
    _verify_schema(conn)
    if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise BackupError("SQLite integrity check failed")
    if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise BackupError("SQLite foreign key check failed")


def _measure(conn: sqlite3.Connection, params: dict) -> dict:
    fragments = conn.execute(f"SELECT count(*),coalesce(sum(bytes),0) FROM ({_ENDED_FRAGMENTS})", params).fetchone()
    questions = conn.execute(
        f"SELECT count(*),coalesce(sum(bytes),0) FROM ({_SUPERSEDED_QUESTIONS})", params
    ).fetchone()
    return {
        "consolidation_fragments": {"rows": fragments[0], "payload_bytes": fragments[1]},
        "obsolete_evaluations": {"rows": questions[0], "payload_bytes": questions[1]},
    }


def _trim_authorizer(action, table, column, database, trigger) -> int:
    """Deny trigger or cascade writes beyond the two intended payload changes."""
    if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
        allowed = (action, table, column) in {
            (sqlite3.SQLITE_DELETE, "consolidation_fragments", None),
            (sqlite3.SQLITE_UPDATE, "candidate_evaluations", "evidence_refs_json"),
        }
        return sqlite3.SQLITE_OK if allowed and database == "main" and trigger is None else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _trim_and_vacuum(conn: sqlite3.Connection, params: dict, measured: dict) -> None:
    # All writes are on an unpublished, disposable copy. Do not emit journal
    # payloads beside it under the output directory's inherited permissions.
    if conn.execute("PRAGMA journal_mode=OFF").fetchone()[0] != "off":
        raise BackupError("unable to disable journaling on the staging copy")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.set_authorizer(_trim_authorizer)
    try:
        trimmed = conn.execute(
            "DELETE FROM consolidation_fragments WHERE (work_id,start_offset) IN "
            f"(SELECT work_id,start_offset FROM ({_ENDED_FRAGMENTS}))",
            params,
        ).rowcount
        cleared = conn.execute(
            "UPDATE candidate_evaluations SET evidence_refs_json='[]' WHERE evaluation_id IN "
            f"(SELECT evaluation_id FROM ({_SUPERSEDED_QUESTIONS}))",
            params,
        ).rowcount
    finally:
        conn.set_authorizer(None)
    if trimmed != measured["consolidation_fragments"]["rows"] or cleared != measured["obsolete_evaluations"]["rows"]:
        raise BackupError("compaction selection changed on the staging copy")
    conn.commit()
    conn.execute("VACUUM")
    _check_store(conn)


def _copy_store(
    reader: sqlite3.Connection,
    source: Path,
    output: Path,
    params: dict,
    measured: dict,
    check_source: Callable[[], None],
) -> int:
    """Transform a protected staging file; publish atomically without overwriting."""
    stage = output.with_name(f".scope-recall-compact-{uuid4().hex}.sqlite3")
    create_output(stage)
    owned = stage.stat()
    try:
        preserve_file_access(source, stage, protect_acl=True)
        with closing(sqlite3.connect(stage)) as copy:
            reader.backup(copy)
            _trim_and_vacuum(copy, params, measured)
        if not os.path.samestat(owned, stage.lstat()) or any(os.path.lexists(p) for p in _sidecars(stage)):
            raise BackupError("staging copy changed or has unexpected SQLite sidecars")
        if any(os.path.lexists(p) for p in (output, *_sidecars(output))):
            raise BackupError("output or an output sidecar already exists")
        with stage.open("rb+") as file:
            os.fsync(file.fileno())
            size = os.fstat(file.fileno()).st_size
        check_source()
        os.link(stage, output)
        return size
    finally:
        if os.path.lexists(stage) and os.path.samestat(owned, stage.lstat()):
            stage.unlink()


def compact_store(
    source: str | Path, *, output: str | Path | None = None, retention_days: int = RETENTION_DAYS
) -> dict:
    """Plan or create a compacted SQLite copy, reading the source under its offline locks."""
    if type(retention_days) is not int or not 1 <= retention_days <= 36500:
        raise BackupError("retention days must be between 1 and 36500")
    source = safe_path(source, must_exist=True)
    if output is not None:
        output = safe_path(output)
        if not output.parent.is_dir() or output in _sidecars(source) or output.name in _STORE_LOCKS:
            raise BackupError("output must be a new SQLite file in an existing directory")
        if any(os.path.lexists(p) for p in (output, *_sidecars(output))):
            raise BackupError("output or an output sidecar already exists")
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    params = {"cutoff": cutoff, "large_fragment_bytes": _LARGE_FRAGMENT_BYTES}
    with _read_store(source) as (reader, before, check_source):
        measured = _measure(reader, params)
        result = {
            "status": "dry_run",
            "source_bytes": before.st_size,
            "schema_version": SCHEMA_VERSION,
            "retention_days": retention_days,
            "cutoff_utc": cutoff,
            **measured,
        }
        if output is None:
            check_source()
        else:
            output_bytes = _copy_store(reader, source, output, params, measured, check_source)
            result.update(status="compacted", output=str(output), output_bytes=output_bytes)
    return result
