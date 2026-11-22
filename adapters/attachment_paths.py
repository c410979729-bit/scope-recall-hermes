"""Inspect attachment files without following a recorded path onto a remote filesystem."""

from __future__ import annotations

import os
import re
import stat
import sys
import unicodedata
from pathlib import Path

from .audiences import HermesIdentityError

_LINUX_REMOTE_FILESYSTEMS = frozenset(
    {b"9p", b"afs", b"autofs", b"ceph", b"cifs", b"davfs", b"glusterfs", b"nfs", b"nfs4", b"smb3"}
)


class UnverifiedAttachmentPath(HermesIdentityError):
    """The path cannot be inspected on this computer."""


def _local_drive(path: Path) -> bool:
    """Reject Windows shares and network drives before probing their paths."""
    if sys.platform != "win32" or not re.fullmatch(r"[a-zA-Z]:\\", path.anchor):
        return False
    import ctypes
    from ctypes import wintypes

    drive_type = ctypes.WinDLL("kernel32").GetDriveTypeW
    drive_type.argtypes = (wintypes.LPCWSTR,)
    drive_type.restype = wintypes.UINT
    return drive_type(path.anchor) in {2, 3, 5, 6}  # removable, fixed, CD-ROM, RAM disk


def _linux_remote_mount(path: Path) -> bool:
    """Consult mount metadata without looking up any name inside a network mount."""
    for line in Path("/proc/self/mountinfo").read_bytes().splitlines():
        local, _, filesystem = line.partition(b" - ")
        kind = filesystem.partition(b" ")[0]
        if kind not in _LINUX_REMOTE_FILESYSTEMS and not kind.startswith(b"fuse"):
            continue
        # mountinfo encodes whitespace and backslashes as octal escapes.
        point = re.sub(rb"\\([0-7]{3})", lambda match: bytes((int(match[1], 8),)), local.split()[4])
        if path.is_relative_to(os.fsdecode(point)):
            return True
    return False


def _darwin_remote_mount(path: Path) -> bool:
    """Use cached mount flags instead of calling statfs on the candidate path."""
    import ctypes

    class StatFS(ctypes.Structure):
        """Darwin's 64-bit statfs layout on Intel and Apple silicon."""

        _fields_ = [
            ("block_size", ctypes.c_uint32),
            ("io_size", ctypes.c_int32),
            ("capacity_counts", ctypes.c_uint64 * 5),
            ("filesystem_id", ctypes.c_int32 * 2),
            ("owner", ctypes.c_uint32),
            ("kind", ctypes.c_uint32),
            ("flags", ctypes.c_uint32),
            ("subtype", ctypes.c_uint32),
            ("filesystem", ctypes.c_char * 16),
            ("mount_path", ctypes.c_char * 1024),
            ("source", ctypes.c_char * 1024),
            ("extended_flags_and_reserved", ctypes.c_uint32 * 8),
        ]

    mnt_nowait, mnt_local = 2, 0x00001000
    libc = ctypes.CDLL(None, use_errno=True)
    getfsstat = getattr(libc, "getfsstat$INODE64", None) or getattr(libc, "getfsstat", None)
    if getfsstat is None:
        raise OSError("could not read current mount information")
    getfsstat.argtypes = (ctypes.POINTER(StatFS), ctypes.c_int, ctypes.c_int)
    getfsstat.restype = ctypes.c_int
    size = getfsstat(None, 0, mnt_nowait)
    if size <= 0:
        raise OSError("could not read current mount information")
    entries = (StatFS * (size + 1))()
    count = getfsstat(entries, ctypes.sizeof(entries), mnt_nowait)
    if not 0 < count < len(entries):
        raise OSError("mount information changed during discovery")
    candidate = Path(unicodedata.normalize("NFD", str(path)).casefold())
    return any(
        not entry.flags & mnt_local
        and candidate.is_relative_to(unicodedata.normalize("NFD", os.fsdecode(entry.mount_path)).casefold())
        for entry in entries[:count]
    )


def local_attachment_stat(path: Path) -> os.stat_result | None:
    """Return a local file's metadata or confirmed absence. Refuse unverified paths before any lookup."""
    if (
        not path.is_absolute()
        or path.anchor.startswith("//")
        or (os.name == "nt" and not _local_drive(path))
        or (sys.platform == "linux" and _linux_remote_mount(path))
        or (sys.platform == "darwin" and _darwin_remote_mount(path))
    ):
        raise UnverifiedAttachmentPath("attachment path cannot be verified on this computer")
    if ".." in path.parts:
        raise HermesIdentityError("shared entry path must be canonical")
    try:
        # Inspect ancestors first: even lstat on a child would cross its parent's junction.
        for component in (*reversed(path.parents), path):
            info = component.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise HermesIdentityError("shared entry path contains a link or reparse point")
    except FileNotFoundError:
        return None
    return info


def same_local_path(recorded: Path, expected: Path) -> bool:
    """Verify possible casing or Unicode aliases using local filesystem identities."""
    if str(recorded) == str(expected):
        return True
    # A manifest can also name homes on other machines. Only probe two plausible
    # spellings of this path, using the same local-mount and link checks as attachment discovery.
    if unicodedata.normalize("NFD", str(recorded)).casefold() != unicodedata.normalize("NFD", str(expected)).casefold():
        return False
    left = local_attachment_stat(recorded)
    right = local_attachment_stat(expected)
    return left is not None and right is not None and os.path.samestat(left, right)
