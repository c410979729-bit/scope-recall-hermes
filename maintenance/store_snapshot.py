"""Read a consistent SQLite snapshot without changing a store or its journal files."""

from __future__ import annotations

import sqlite3
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from scope_recall.core.backup_primitives import create_output, safe_path
from scope_recall.core.file_access import preserve_file_access
from scope_recall.core.truth_connection import require_query_only

from ._compact_pager import SCRIPT_PATH

_MEMORY_SNAPSHOT_BYTES = 16 * 1024 * 1024


def _revision(database: Path) -> tuple[int, int, int, int, int]:
    info = database.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _has_journal(database: Path, *, wal_format: bool) -> bool:
    journals = set()
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            info = database.with_name(database.name + suffix).lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise sqlite3.OperationalError("unexpected SQLite journal file")
        journals.add(suffix)
    # Even a mode=ro connection can create shared memory for an unpaired WAL.
    if ("-wal" in journals) != ("-shm" in journals):
        raise sqlite3.OperationalError("SQLite WAL and shared memory are incomplete; retry the preview")
    if (wal_format and "-journal" in journals) or (not wal_format and "-wal" in journals):
        raise sqlite3.OperationalError("SQLite journal format does not match the database; recovery is required")
    return bool(journals)


@contextmanager
def read_store_snapshot(database: Path) -> Iterator[sqlite3.Connection]:
    """Read committed data, including WAL frames. A closed WAL store keeps its directory unchanged."""
    path = safe_path(database, must_exist=True, error_type=sqlite3.OperationalError)
    # A separate process owns the shared pager lock. Closing any other descriptor
    # for this inode in the parent would release a process-owned POSIX record lock.
    argv = [sys.executable, "-IBS", str(SCRIPT_PATH), "--read-guard", str(path)]
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as guard:
        header = guard.stdout.read(1) if guard.stdout is not None else b""
        if header not in (b"R", b"W"):
            raise sqlite3.OperationalError("database is locked or cannot be read")
        journals = _has_journal(path, wal_format=header == b"W")
        # A closed WAL-format store must not create sidecars. If a writer opens it
        # during an immutable read, discard that preview and read its WAL on retry.
        stamp = _revision(path) if header == b"W" and not journals else None
        uri = path.as_uri() + ("?mode=ro&immutable=1" if stamp is not None else "?mode=ro&readonly_shm=1")
        with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
            conn.row_factory = sqlite3.Row
            require_query_only(conn)
            conn.execute("BEGIN")
            conn.execute("SELECT rootpage FROM sqlite_schema LIMIT 1").fetchone()
            yield conn
            if guard.poll() is not None or (
                stamp is not None and (_has_journal(path, wal_format=True) or _revision(path) != stamp)
            ):
                raise sqlite3.OperationalError("the checkpoint changed during the read; retry the preview")


@contextmanager
def detached_store_snapshot(
    database: Path, validate: Callable[[sqlite3.Connection], None]
) -> Iterator[sqlite3.Connection]:
    """Release the source pager after copying committed pages, before expensive analysis starts."""
    with ExitStack() as cleanup:
        with read_store_snapshot(database) as source:
            validate(source)
            size = source.execute("PRAGMA page_size").fetchone()[0] * source.execute("PRAGMA page_count").fetchone()[0]
            target: Path | str = ":memory:"
            if size > _MEMORY_SNAPSHOT_BYTES:
                # The OS temporary directory may have a system alias such as /var on macOS.
                # Resolve that trusted root before enforcing link checks on our own new file.
                temporary_root = Path(tempfile.gettempdir()).resolve(strict=True)
                directory = Path(tempfile.mkdtemp(prefix="scope-recall-preview-", dir=temporary_root))
                cleanup.callback(directory.rmdir)
                target = directory / "snapshot.sqlite3"
                create_output(target)
                cleanup.callback(target.unlink)
                # This private copy can contain the entire source. Protect the empty file before filling it.
                preserve_file_access(database, target, protect_acl=True)
            copy = cleanup.enter_context(closing(sqlite3.connect(target)))
            copy.execute("PRAGMA journal_mode=OFF")
            source.backup(copy)
        require_query_only(copy)
        copy.row_factory = sqlite3.Row
        yield copy
