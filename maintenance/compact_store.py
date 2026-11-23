"""Preview storage reclamation and publish a verified offline SQLite copy."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import stat
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing, contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from scope_recall.core.backup_primitives import (
    BackupError,
    create_output,
    exclusive_store,
    first_link,
    safe_path,
    sha256,
)
from scope_recall.core.file_access import preserve_file_access
from scope_recall.core.schema import APPLICATION_ID, SCHEMA_VERSION
from scope_recall.core.writer_lease import TRUTH_WRITER_LEASE_FILENAME, TRUTH_WRITER_LEASE_INFO_FILENAME

from ._compact_pager import SCRIPT_PATH, STAGE_PREFIX, file_revision, sidecars
from .compact_candidates import measure_candidates, trim_candidates
from .compact_lexical import compact_tool_lexical, measure_tool_lexical
from .store_snapshot import detached_store_snapshot, require_disk_space

RETENTION_DAYS = 30
_LARGE_FRAGMENT_BYTES = 4096
_STORE_LOCKS = (
    TRUTH_WRITER_LEASE_FILENAME,
    TRUTH_WRITER_LEASE_INFO_FILENAME,
    "scope-recall-retained.lock",
    "runtime-worker.lock",
)

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

_COPY_WRITES = frozenset(
    {
        (sqlite3.SQLITE_DELETE, "consolidation_fragments", None),
        (sqlite3.SQLITE_UPDATE, "candidate_lifecycle", "processing_state"),
        (sqlite3.SQLITE_UPDATE, "candidate_lifecycle", "reason"),
        (sqlite3.SQLITE_UPDATE, "candidate_lifecycle", "dormant_at"),
        (sqlite3.SQLITE_UPDATE, "candidate_evaluations", "evidence_refs_json"),
        (sqlite3.SQLITE_DELETE, "candidate_evaluations", None),
        (sqlite3.SQLITE_DELETE, "candidate_evidence", None),
        (sqlite3.SQLITE_DELETE, "candidate_trigger_terms", None),
        (sqlite3.SQLITE_DELETE, "candidate_source_triggers", None),
        (sqlite3.SQLITE_INSERT, "lexical_terms", None),
        (sqlite3.SQLITE_DELETE, "lexical_terms", None),
        (sqlite3.SQLITE_INSERT, "lexical_postings", None),
        (sqlite3.SQLITE_DELETE, "lexical_postings", None),
    }
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare an explicit source, a mode, and a fresh output for apply."""
    parser.add_argument("--source", required=True, help="path to the Scope Recall SQLite database")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="estimate reclaimable data from the live database")
    mode.add_argument("--apply", action="store_true", help="write a verified, vacuumed copy of an offline database")
    parser.add_argument("--output", help="new SQLite file to create in an existing folder; required for --apply")
    parser.add_argument(
        "--temp-dir",
        help="existing directory for large dry-run snapshots (defaults to the OS temporary directory)",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=RETENTION_DAYS,
        help=f"minimum processing payload age (default: {RETENTION_DAYS}; 0 includes payloads older than now)",
    )


def _offline_file(path: Path) -> os.stat_result:
    """Refuse live or unrecovered journals and aliases that evade the directory lease."""
    if any(os.path.lexists(candidate) for candidate in sidecars(path)):
        raise BackupError(
            "store is busy or has uncheckpointed SQLite sidecars: "
            "stop all store processes and open and close the store normally to checkpoint; "
            "never delete -wal or -journal files"
        )
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise BackupError("source must be a regular SQLite file with no other hard links")
    return info


@contextmanager
def _exclusive_pager(source: Path, before: os.stat_result):
    """Hold SQLite's native file lock from before inspection through publication."""
    # POSIX record locks belong to a process and must conflict with every
    # SQLite connection even when the command has been called in-process.
    argv = [sys.executable, "-IBS", str(SCRIPT_PATH), str(source), *map(str, file_revision(before))]
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as owner:
        if owner.stdout is None or owner.stdout.read(1) != b"1":
            raise BackupError("store is busy or the SQLite file could not be locked exclusively")
        yield owner


def _publish_with_pager(owner: subprocess.Popen[bytes], stage: Path, output: Path, verified: os.stat_result) -> None:
    """Request a link from the lock owner and verify the outcome after it has stopped."""
    if owner.stdin is None:
        raise BackupError("missing SQLite publication pipe")
    request = json.dumps((str(stage), str(output), file_revision(verified))).encode("ascii") + b"\n"
    try:
        with suppress(OSError), owner.stdin as pipe:
            pipe.write(request)
    finally:
        owner.wait()
    # The link may have succeeded immediately before the child exited abnormally.
    # The output inode, rather than the exit status, records the commit point.
    try:
        published = output.lstat()
    except FileNotFoundError:
        raise BackupError("SQLite lock was lost or the compacted copy could not be published") from None
    if file_revision(published) != file_revision(verified) or not stat.S_ISREG(published.st_mode):
        raise BackupError("output path is occupied by another file")


@contextmanager
def _read_offline_store(source: Path):
    """Hold the runtime locks through the final source check and output publication."""
    if any(first_link(source.parent / name) is not None for name in _STORE_LOCKS):
        raise BackupError("store lock files must not be symlinks or reparse points")
    with exclusive_store(source.parent, busy=lambda _reason: BackupError("store is busy: stop all store processes")):
        before = _offline_file(source)
        # On Windows, hash before the 1 GiB mandatory byte lock so multi-gigabyte files read cleanly.
        # On POSIX, never open/close a descriptor for ``source`` in this process before the child acquires
        # ``LOCK_EX``, because closing any descriptor to the inode would release an in-process SQLite fcntl lock.
        source_digest = sha256(source) if os.name == "nt" else None
        if source_digest is not None:
            _unchanged(source, before)
        with _exclusive_pager(source, before) as owner:
            if source_digest is None:
                source_digest = sha256(source)

            def check_source() -> None:
                if owner.poll() is not None:
                    raise BackupError("exclusive SQLite file lock was lost")
                _unchanged(source, before)

            def publish(stage: Path, output: Path, verified: os.stat_result) -> None:
                check_source()
                _publish_with_pager(owner, stage, output, verified)

            check_source()
            # The native pager lock excludes independent SQLite readers and
            # writers. This handle can read without opening source journals.
            with closing(sqlite3.connect(source.as_uri() + "?mode=ro&immutable=1", uri=True)) as reader:
                _verify_schema(reader)
                yield reader, before, source_digest, publish


def _unchanged(source: Path, before: os.stat_result) -> None:
    if file_revision(_offline_file(source)) != file_revision(before):
        raise BackupError("source changed during compaction")


def _verify_schema(conn: sqlite3.Connection) -> None:
    header = (
        conn.execute("PRAGMA application_id").fetchone()[0],
        conn.execute("PRAGMA user_version").fetchone()[0],
    )
    if header != (APPLICATION_ID, SCHEMA_VERSION):
        raise BackupError("compact-store requires a current Scope Recall schema")
    meta = conn.execute("SELECT schema_version FROM instance_meta WHERE singleton=1").fetchone()
    if meta is None or meta[0] != SCHEMA_VERSION:
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
    return {
        "consolidation_fragments": {"rows": fragments[0], "estimated_bytes": fragments[1]},
        **measure_candidates(conn, params["cutoff"]),
        "tool_lexical": measure_tool_lexical(conn),
    }


def _trim_authorizer(action, table, column, database, trigger) -> int:
    """Allow only expected staging writes and refuse writes from triggers."""
    if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE}:
        allowed = (action, table, column) in _COPY_WRITES
        return sqlite3.SQLITE_OK if allowed and database == "main" and trigger is None else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _trim_and_vacuum(conn: sqlite3.Connection, params: dict, measured: dict) -> None:
    # All writes are on an unpublished, disposable copy. Do not emit journal
    # payloads beside it under the output directory's inherited permissions.
    if conn.execute("PRAGMA journal_mode=OFF").fetchone()[0] != "off":
        raise BackupError("unable to disable journaling on the staging copy")
    conn.execute("PRAGMA temp_store=MEMORY")
    # Rebuilding postings touches the reverse index in scattered page order.
    # Give this disposable connection up to 64 MiB of cache without changing the source.
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.set_authorizer(_trim_authorizer)
    try:
        conn.execute("BEGIN")
        trimmed = conn.execute(
            "DELETE FROM consolidation_fragments WHERE (work_id,start_offset) IN "
            f"(SELECT work_id,start_offset FROM ({_ENDED_FRAGMENTS}))",
            params,
        ).rowcount
        if trimmed != measured["consolidation_fragments"]["rows"]:
            raise BackupError("consolidation fragment selection changed on the staging copy")
        trim_candidates(conn, params["cutoff"], measured)
        compact_tool_lexical(conn, measured["tool_lexical"])
    finally:
        conn.set_authorizer(None)
    conn.commit()


def _copy_store(
    reader: sqlite3.Connection,
    source: Path,
    output: Path,
    params: dict,
    measured: dict,
    publish: Callable[[Path, Path, os.stat_result], None],
) -> int:
    """Transform a protected work copy and VACUUM directly into a protected staging file in output.parent."""
    # Staging copies the source and runs VACUUM INTO inside output.parent, transiently holding both copies there
    # without writing any temporary database bytes to the OS temporary directory.
    require_disk_space(output.parent, source.stat().st_size * 2, label="compaction output directory")
    work = output.with_name(f".scope-recall-work-{uuid4().hex}.sqlite3")
    stage = output.with_name(f"{STAGE_PREFIX}{uuid4().hex}.sqlite3")
    create_output(work)
    owned_work = work.stat()
    owned = None
    try:
        preserve_file_access(source, work, protect_acl=True)
        create_output(stage)
        owned = stage.stat()
        preserve_file_access(source, stage, protect_acl=True)
        with closing(sqlite3.connect(work)) as copy:
            reader.backup(copy)
            _trim_and_vacuum(copy, params, measured)
            copy.execute("VACUUM main INTO ?", (str(stage),))
        if os.path.lexists(work) and os.path.samestat(owned_work, work.lstat()):
            work.unlink()
        with closing(sqlite3.connect(stage)) as compacted:
            _check_store(compacted)
        if not os.path.samestat(owned, stage.lstat()) or any(os.path.lexists(p) for p in sidecars(stage)):
            raise BackupError("staging copy changed or has unexpected SQLite sidecars")
        if any(os.path.lexists(p) for p in (output, *sidecars(output))):
            raise BackupError("output or an output sidecar already exists")
        with stage.open("rb+") as file:
            os.fsync(file.fileno())
            verified = os.fstat(file.fileno())
        if not os.path.samestat(owned, verified):
            raise BackupError("staging copy changed before publication")
        publish(stage, output, verified)
        return verified.st_size
    finally:
        if os.path.lexists(work) and os.path.samestat(owned_work, work.lstat()):
            work.unlink()
        if owned is not None and os.path.lexists(stage) and os.path.samestat(owned, stage.lstat()):
            stage.unlink()


def compact_store(
    source: str | Path,
    *,
    output: str | Path | None = None,
    retention_days: int = RETENTION_DAYS,
    temp_dir: str | Path | None = None,
) -> dict:
    """Estimate from a read snapshot or create a compacted copy under offline locks."""
    if type(retention_days) is not int or not 0 <= retention_days <= 36500:
        raise BackupError("retention days must be between 0 and 36500")
    if output is not None and temp_dir is not None:
        raise BackupError("--temp-dir is only available with --dry-run")
    source = safe_path(source, must_exist=True)
    if output is not None:
        output = safe_path(output)
        if not output.parent.is_dir() or output in sidecars(source) or output.name in _STORE_LOCKS:
            raise BackupError("output must be a new SQLite file in an existing directory")
        if any(os.path.lexists(p) for p in (output, *sidecars(output))):
            raise BackupError("output or an output sidecar already exists")
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
    params = {"cutoff": cutoff, "large_fragment_bytes": _LARGE_FRAGMENT_BYTES}
    result = {"schema_version": SCHEMA_VERSION, "retention_days": retention_days, "cutoff_utc": cutoff}
    if output is None:
        preview_temp = Path(temp_dir) if temp_dir is not None else None
        with detached_store_snapshot(source, _verify_schema, temp_dir=preview_temp) as reader:
            pages = reader.execute("PRAGMA page_count").fetchone()[0]
            page_bytes = reader.execute("PRAGMA page_size").fetchone()[0]
            source_bytes = pages * page_bytes
            measured = _measure(reader, params)
        result.update(status="dry_run", source_bytes=source_bytes)
    else:
        with _read_offline_store(source) as (reader, before, source_digest, publish):
            measured = _measure(reader, params)
            output_bytes = _copy_store(reader, source, output, params, measured, publish)
        result.update(
            status="compacted",
            source_bytes=before.st_size,
            source_sha256=source_digest,
            output=str(output),
            output_bytes=output_bytes,
        )
    estimated = sum(item["estimated_bytes"] for item in measured.values())
    return {**result, **measured, "estimated_reclaimable_bytes": estimated}
