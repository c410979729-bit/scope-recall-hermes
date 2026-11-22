"""Coordinate native SQLite read locks and exclusive offline publication."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import BinaryIO

SCRIPT_PATH = Path(__file__).resolve()

_WINDOWS_LOCK_OFFSET = 1 << 30
_WINDOWS_LOCK_BYTES = 512
_SHARED_FIRST = _WINDOWS_LOCK_OFFSET + 2
_SHARED_SIZE = _WINDOWS_LOCK_BYTES - 2
STAGE_PREFIX = ".scope-recall-compact-"


def sidecars(path: Path) -> tuple[Path, ...]:
    """Return SQLite's possible journal and shared-memory siblings."""
    return tuple(path.with_name(path.name + suffix) for suffix in ("-wal", "-shm", "-journal"))


def file_revision(info: os.stat_result) -> tuple[int, int, int, int]:
    """Use the physical file identity and the read or staged byte version."""
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _acquire(source: BinaryIO) -> bool:
    """Exclude SQLite's standard native pager without changing the file."""
    if os.name == "nt":
        import msvcrt

        # The SQLite file format reserves this range for the Windows VFS locks.
        source.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(source.fileno(), msvcrt.LK_NBLCK, _WINDOWS_LOCK_BYTES)
        return True
    if os.name == "posix":
        import fcntl

        # A whole-file record lock covers SQLite's lock range even beyond EOF.
        fcntl.lockf(source.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    return False


def _acquire_shared(source: BinaryIO) -> bool:
    """Keep the last SQLite connection from unlinking its WAL while a preview opens."""
    if sys.platform == "win32":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        class Overlapped(ctypes.Structure):
            _fields_ = [
                ("internal", ctypes.c_size_t),
                ("internal_high", ctypes.c_size_t),
                ("offset", wintypes.DWORD),
                ("offset_high", wintypes.DWORD),
                ("event", wintypes.HANDLE),
            ]

        lock = ctypes.WinDLL("kernel32", use_last_error=True).LockFileEx
        lock.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(Overlapped),
        )
        lock.restype = wintypes.BOOL
        # FAIL_IMMEDIATELY without EXCLUSIVE_LOCK lets ordinary WAL writers commit.
        position = Overlapped(offset=_SHARED_FIRST)
        return bool(lock(msvcrt.get_osfhandle(source.fileno()), 1, 0, _SHARED_SIZE, 0, ctypes.byref(position)))
    if os.name == "posix":
        import fcntl

        fcntl.lockf(source.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB, _SHARED_SIZE, _SHARED_FIRST)
        return True
    return False


def _read_guard(path: Path) -> int:
    """Hold a shared lock on a read-only descriptor until the parent closes its pipe."""
    with path.open("rb", buffering=0) as source:
        if not _acquire_shared(source):
            return 2
        header = source.read(20)
        sys.stdout.buffer.write(
            b"W" if header.startswith(b"SQLite format 3\0") and header[18:] == b"\x02\x02" else b"R"
        )
        sys.stdout.buffer.flush()
        sys.stdin.buffer.read(1)
        return 0


def _publish(source: Path, expected: tuple[int, ...], request: str) -> int:
    """Link one validated staging file while this process still owns the source lock."""
    if not request.endswith("\n"):
        return 2
    stage_name, output_name, verified = json.loads(request)
    stage, output = Path(stage_name), Path(output_name)
    if (
        not stage.is_absolute()
        or stage.parent != output.parent
        or not stage.name.startswith(STAGE_PREFIX)
        or output in (source, stage, *sidecars(source))
    ):
        return 2
    current, staged = source.lstat(), stage.lstat()
    if (
        file_revision(current) != expected
        or current.st_nlink != 1
        or not stat.S_ISREG(current.st_mode)
        or not stat.S_ISREG(staged.st_mode)
        or file_revision(staged) != tuple(verified)
        or any(os.path.lexists(p) for p in (*sidecars(source), *sidecars(stage), *sidecars(output)))
    ):
        return 2
    os.link(stage, output, follow_symlinks=False)
    return 0


def _main(argv: list[str]) -> int:
    """Lock one physical file and publish at most one fully received request."""
    if len(argv) == 2 and argv[0] == "--read-guard":
        try:
            return _read_guard(Path(argv[1]))
        except OSError:
            return 2
    if len(argv) != 5:
        return 2
    path = Path(argv[0])
    try:
        expected = tuple(map(int, argv[1:]))
        # POSIX requires a writable descriptor for its exclusive lock. This
        # process never writes to that descriptor or opens a SQLite connection.
        with path.open("rb" if os.name == "nt" else "r+b", buffering=0) as source:
            if file_revision(os.fstat(source.fileno())) != expected or not _acquire(source):
                return 2
            sys.stdout.write("1")
            sys.stdout.flush()
            request = sys.stdin.readline()
            return _publish(path, expected, request) if request else 0
    except (OSError, TypeError, ValueError):
        return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
