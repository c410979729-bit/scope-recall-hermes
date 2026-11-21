"""Hold the native SQLite file lock for an offline compaction parent process."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import BinaryIO

SCRIPT_PATH = Path(__file__).resolve()

_WINDOWS_LOCK_OFFSET = 1 << 30
_WINDOWS_LOCK_BYTES = 512


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


def _main(argv: list[str]) -> int:
    """Lock the expected physical file and hold until the parent closes stdin."""
    if len(argv) != 3:
        return 2
    path, device, inode = argv
    try:
        # POSIX requires a writable descriptor for its exclusive lock. This
        # process never writes to that descriptor or opens a SQLite connection.
        with Path(path).open("rb" if os.name == "nt" else "r+b", buffering=0) as source:
            info = os.fstat(source.fileno())
            if (info.st_dev, info.st_ino) != (int(device), int(inode)) or not _acquire(source):
                return 2
            sys.stdout.write("1")
            sys.stdout.flush()
            sys.stdin.read(1)
    except (OSError, ValueError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
