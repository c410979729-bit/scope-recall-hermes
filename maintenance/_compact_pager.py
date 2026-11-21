"""Hold the native SQLite file lock for an offline compaction parent process."""

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
