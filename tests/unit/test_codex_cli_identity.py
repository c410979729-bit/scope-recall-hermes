"""Synthetic PE identity and CLI admission contracts; no executable is launched."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import pytest
from scope_recall.runtime import codex_cli as cli
from scope_recall.runtime.models import AuxiliaryModelError

IMAGE_SHA256 = "583ed1f42412daf9ac61a74788f077bfe8d178a1f4acd6a39bdd4e171fa6ba4f"
LEGACY_SHA256 = "960c111d47afd61669954b9df9e56083e302edbfa3ef6962d81dcc14a30051dc"


def pe_bytes(*, magic=0x20B, certificate=b"TESTCERT", checksum=0):
    """Build two contiguous raw sections and a terminal certificate table."""
    optional_size = 240 if magic == 0x20B else 224
    data = bytearray(0x400)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", data, 0x86, 2)
    struct.pack_into("<H", data, 0x94, optional_size)
    struct.pack_into("<H", data, 0x98, magic)
    struct.pack_into("<I", data, 0x98 + 60, 0x200)
    struct.pack_into("<I", data, 0x98 + 64, checksum)
    directories = 112 if magic == 0x20B else 96
    struct.pack_into("<I", data, 0x98 + directories - 4, 16)
    struct.pack_into("<II", data, 0x98 + directories + 32, 0x400, len(certificate))
    table = 0x98 + optional_size
    for index, offset in enumerate((0x200, 0x300)):
        struct.pack_into("<II", data, table + index * 40 + 16, 0x100, offset)
    data[0x200:0x400] = b"A" * 0x100 + b"B" * 0x100
    return data + certificate


def image_file(tmp_path, data):
    """Write a synthetic candidate, returning only its isolated path."""
    path = tmp_path / "candidate.exe"
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("magic", [0x10B, 0x20B])
def test_image_digest_excludes_signature_and_checksum(tmp_path, magic):
    original = pe_bytes(magic=magic)
    path = image_file(tmp_path, original)
    directories = 112 if magic == 0x20B else 96
    checksum, security = 0x98 + 64, 0x98 + directories + 32
    expected = hashlib.sha256(original[:checksum] + original[checksum + 4 : security] + original[security + 8 : 0x400])
    digest = cli.pe_image_sha256(path)
    assert digest == expected.hexdigest()
    path.write_bytes(pe_bytes(magic=magic, certificate=b"OTHER_CERT_TABLE"))
    assert cli.pe_image_sha256(path) == digest
    path.write_bytes(pe_bytes(magic=magic, certificate=b"\0" * 8))
    assert cli.pe_image_sha256(path) == digest
    path.write_bytes(pe_bytes(magic=magic, checksum=0x12345678))
    assert cli.pe_image_sha256(path) == digest


@pytest.mark.parametrize("offset", [0x200, 0x300])
def test_image_digest_covers_every_section(tmp_path, offset):
    data = pe_bytes()
    path = image_file(tmp_path, data)
    digest = cli.pe_image_sha256(path)
    data[offset] ^= 1
    path.write_bytes(data)
    assert cli.pe_image_sha256(path) not in (None, digest)


@pytest.mark.parametrize(
    ("offset", "fmt", "value"),
    [
        (0, "2s", b"NO"),
        (0x3C, "I", 0x20),
        (0x3C, "I", 0xFFFFFFF0),
        (0x80, "4s", b"NOPE"),
        (0x86, "H", 0),
        (0x86, "H", 97),
        (0x94, "H", 64),
        (0x94, "H", 0),
        (0x94, "H", 1),
        (0x94, "H", 0xFFFF),
        (0x98, "H", 0x107),
        (0x98 + 108, "I", 4),
        (0x98 + 108, "I", 17),
        (0x98 + 60, "I", 0x100),
        (0x98 + 60, "I", 0x500),
        (0x98 + 144, "I", 0),
        (0x98 + 148, "I", 0),
        (0x98 + 148, "I", 16),
        (0x98 + 240 + 20, "I", 0x100),
        (0x98 + 240 + 20, "I", 0x208),
        (0x98 + 240 + 40 + 20, "I", 0x200),
        (0x98 + 240 + 40 + 16, "I", 0x108),
    ],
    ids=[
        "not-mz",
        "pe-inside-dos",
        "pe-outside-file",
        "not-pe",
        "no-sections",
        "too-many-sections",
        "short-optional-header",
        "empty-optional-header",
        "truncated-optional-magic",
        "optional-header-outside-file",
        "unknown-magic",
        "no-security-directory",
        "directories-outside-header",
        "headers-before-section-table",
        "headers-outside-file",
        "no-certificate-offset",
        "no-certificate-size",
        "certificate-past-eof",
        "section-overlaps-headers",
        "gap-before-section",
        "overlapping-sections",
        "section-overlaps-certificate",
    ],
)
def test_image_digest_rejects_malformed_headers(tmp_path, offset, fmt, value):
    data = pe_bytes()
    struct.pack_into("<" + fmt, data, offset, value)
    assert cli.pe_image_sha256(image_file(tmp_path, data)) is None


@pytest.mark.parametrize(
    ("certificate", "size"), [(0x400, 7), (0x401, 8)], ids=["short-certificate", "unaligned-certificate"]
)
def test_image_digest_rejects_invalid_certificate_extent(tmp_path, certificate, size):
    data = pe_bytes()
    struct.pack_into("<II", data, 0x98 + 144, certificate, size)
    # Preserve the terminal range and section bounds so only length or alignment is invalid.
    data = data[: certificate + size]
    data.extend(b"\0" * (certificate + size - len(data)))
    assert certificate + size == len(data)
    assert certificate >= 0x400
    assert (certificate % 8 == 0) != (size >= 8)
    assert cli.pe_image_sha256(image_file(tmp_path, data)) is None


@pytest.mark.parametrize("length", [0, 1, 63, 0x83, 0x97, 0x98 + 239, 0x98 + 240 + 79])
def test_image_digest_rejects_truncated_headers(tmp_path, length):
    assert cli.pe_image_sha256(image_file(tmp_path, pe_bytes()[:length])) is None


def test_image_digest_rejects_nonterminal_certificate(tmp_path):
    assert cli.pe_image_sha256(image_file(tmp_path, pe_bytes() + b"overlay")) is None


def test_image_digest_rejects_missing_certificate(tmp_path):
    data = pe_bytes(certificate=b"")
    struct.pack_into("<I", data, 0x98 + 144, 0)
    assert cli.pe_image_sha256(image_file(tmp_path, data)) is None


@pytest.mark.parametrize("name", ["missing.exe", "\0.exe"])
def test_image_digest_handles_io_error(tmp_path, name):
    assert cli.pe_image_sha256(tmp_path / name) is None


def test_image_digest_rejects_short_read(tmp_path, monkeypatch):
    path = image_file(tmp_path, pe_bytes())
    real_open = Path.open

    def truncated_open(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        original_read = stream.read
        stream.read = lambda size: original_read(size)[:-1]
        return stream

    monkeypatch.setattr(Path, "open", truncated_open)
    assert cli.pe_image_sha256(path) is None


def test_image_digest_rejects_truncated_hash_range(tmp_path, monkeypatch):
    path = image_file(tmp_path, pe_bytes())
    real_open = Path.open

    def truncated_open(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        original_read = stream.read
        stream.read = lambda size: b"" if stream.tell() == 0 and size != 64 else original_read(size)
        return stream

    monkeypatch.setattr(Path, "open", truncated_open)
    assert cli.pe_image_sha256(path) is None


def test_image_digest_bounds_reads(tmp_path, monkeypatch):
    data = pe_bytes()
    data[0x400:0x400] = b"X" * (3 * 1048576)
    struct.pack_into("<I", data, 0x98 + 144, len(data) - 8)
    path = image_file(tmp_path, data)
    real_open = Path.open
    sizes = []

    def bounded_open(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        original_read = stream.read

        def read(size):
            sizes.append(size)
            return original_read(size)

        stream.read = read
        return stream

    monkeypatch.setattr(Path, "open", bounded_open)
    assert cli.pe_image_sha256(path) is not None
    assert max(sizes) == 1048576
    assert sizes.count(1048576) == 3


@pytest.mark.parametrize("mutation", ["headers", "overlay"])
def test_image_digest_covers_other_noncertificate_bytes(tmp_path, mutation):
    data = pe_bytes()
    path = image_file(tmp_path, data)
    digest = cli.pe_image_sha256(path)
    if mutation == "headers":
        data[0x70] ^= 1
    else:
        data[0x400:0x400] = b"OVERLAY!"
        struct.pack_into("<I", data, 0x98 + 144, 0x408)
    path.write_bytes(data)
    assert cli.pe_image_sha256(path) not in (None, digest)


@pytest.mark.parametrize(
    ("whole_allowed", "image_allowed", "matches_config", "admitted"),
    [
        (True, False, True, True),
        (False, True, True, True),
        (False, True, False, False),
        (True, False, False, False),
        (False, False, True, False),
    ],
)
def test_catalog_admission_requires_file_pin_and_verified_identity(
    tmp_path, monkeypatch, whole_allowed, image_allowed, matches_config, admitted
):
    path = image_file(tmp_path, pe_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    image = cli.pe_image_sha256(path)
    assert image is not None
    monkeypatch.setattr(cli, "VERIFIED_BINARIES", frozenset({digest} if whole_allowed else ()))
    monkeypatch.setattr(cli, "VERIFIED_PE_IMAGES", frozenset({image} if image_allowed else ()))
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return 0, json.dumps({"models": [{"slug": cli.MODEL}]}).encode()

    monkeypatch.setattr(cli, "_run", run)
    route = cli.CodexCliRouteConfig(path, digest if matches_config else "0" * 64)
    if admitted:
        assert cli._prepare_catalog(route, tmp_path, seconds=1).is_file()
        assert len(calls) == 1
    else:
        with pytest.raises(AuxiliaryModelError, match="codex_unverified_binary"):
            cli._prepare_catalog(route, tmp_path, seconds=1)
        assert not calls


def test_legacy_non_pe_binary_can_still_be_admitted(tmp_path, monkeypatch):
    path = image_file(tmp_path, b"TEST non-PE executable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(cli, "VERIFIED_BINARIES", frozenset({digest}))
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: (0, json.dumps({"models": [{"slug": cli.MODEL}]}).encode()))
    assert cli._prepare_catalog(cli.CodexCliRouteConfig(path, digest), tmp_path, seconds=1).is_file()


def test_catalog_identity_checks_share_one_open_file(tmp_path, monkeypatch):
    original = pe_bytes()
    path = image_file(tmp_path, original)
    replacement = tmp_path / "replacement.exe"
    changed = pe_bytes()
    changed[0x300] ^= 1
    replacement.write_bytes(changed)
    replacement_image = cli.pe_image_sha256(replacement)
    assert replacement_image is not None
    monkeypatch.setattr(cli, "VERIFIED_BINARIES", frozenset())
    monkeypatch.setattr(cli, "VERIFIED_PE_IMAGES", frozenset({replacement_image}))
    real_open = Path.open
    opened = []

    def changing_path_open(self, *args, **kwargs):
        if self == path:
            selected = replacement if opened else path
            opened.append(selected)
            return real_open(selected, *args, **kwargs)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changing_path_open)
    calls = []
    monkeypatch.setattr(
        cli, "_run", lambda *a, **kw: (calls.append(a) or 0, json.dumps({"models": [{"slug": cli.MODEL}]}).encode())
    )
    route = cli.CodexCliRouteConfig(path, hashlib.sha256(original).hexdigest())
    with pytest.raises(AuxiliaryModelError, match="codex_unverified_binary"):
        cli._prepare_catalog(route, tmp_path, seconds=5)
    assert opened == [path]
    assert not calls


def test_catalog_legacy_admission_skips_image_digest(tmp_path, monkeypatch):
    path = image_file(tmp_path, b"TEST non-PE executable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(cli, "VERIFIED_BINARIES", frozenset({digest}))
    monkeypatch.setattr(cli, "_pe_image_ranges", lambda *a: pytest.fail("legacy admission must skip PE parsing"))
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: (0, json.dumps({"models": [{"slug": cli.MODEL}]}).encode()))
    assert cli._prepare_catalog(cli.CodexCliRouteConfig(path, digest), tmp_path, seconds=1).is_file()


@pytest.mark.parametrize(
    ("whole_allowed", "seconds", "expected_remaining"),
    [(False, 5, 3), (True, 5, 3.75), (False, 2, None), (False, 1.5, None), (True, 1, None)],
    ids=["both-digests", "whole-file-only", "deadline-reached", "deadline-passed", "legacy-deadline-passed"],
)
def test_catalog_digest_time_reduces_process_deadline(
    tmp_path, monkeypatch, whole_allowed, seconds, expected_remaining
):
    path = image_file(tmp_path, pe_bytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    image = cli.pe_image_sha256(path)
    assert image is not None
    monkeypatch.setattr(cli, "VERIFIED_BINARIES", frozenset({digest} if whole_allowed else ()))
    monkeypatch.setattr(cli, "VERIFIED_PE_IMAGES", frozenset({image}))
    clock = [100.0]
    monkeypatch.setattr(cli.time, "monotonic", lambda: clock[0])
    real_digest = cli.hashlib.file_digest
    real_ranges = cli._pe_image_ranges

    def whole_digest(*args, **kwargs):
        result = real_digest(*args, **kwargs)
        clock[0] += 1.25
        return result

    def image_ranges(*args, **kwargs):
        result = real_ranges(*args, **kwargs)
        clock[0] += 0.75
        return result

    monkeypatch.setattr(cli.hashlib, "file_digest", whole_digest)
    monkeypatch.setattr(cli, "_pe_image_ranges", image_ranges)
    remaining = []

    def run(*args, **kwargs):
        remaining.append(kwargs["seconds"])
        return 0, json.dumps({"models": [{"slug": cli.MODEL}]}).encode()

    monkeypatch.setattr(cli, "_run", run)
    route = cli.CodexCliRouteConfig(path, digest)
    if expected_remaining is None:
        with pytest.raises(AuxiliaryModelError, match="timeout"):
            cli._prepare_catalog(route, tmp_path, seconds=seconds)
        assert not remaining
    else:
        assert cli._prepare_catalog(route, tmp_path, seconds=seconds).is_file()
        assert remaining == [expected_remaining]


@pytest.mark.parametrize("image_allowlist_nonempty", [False, True])
def test_catalog_rejects_missing_image_identity(tmp_path, monkeypatch, image_allowlist_nonempty):
    path = image_file(tmp_path, b"TEST non-PE executable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(cli, "VERIFIED_BINARIES", frozenset())
    monkeypatch.setattr(cli, "VERIFIED_PE_IMAGES", frozenset({IMAGE_SHA256} if image_allowlist_nonempty else ()))
    monkeypatch.setattr(cli, "_run", lambda *a, **kw: pytest.fail("unverified candidates must not start"))
    with pytest.raises(AuxiliaryModelError, match="codex_unverified_binary"):
        cli._prepare_catalog(cli.CodexCliRouteConfig(path, digest), tmp_path, seconds=1)


def test_verified_identities_retain_legacy_and_admit_verified_build():
    assert cli.VERIFIED_BINARIES == frozenset({LEGACY_SHA256})
    assert cli.VERIFIED_PE_IMAGES == frozenset({IMAGE_SHA256})


def test_native_pe_image_digest():
    executable = os.environ.get("SCOPE_RECALL_CODEX_CLI")
    if not executable:
        pytest.skip("set SCOPE_RECALL_CODEX_CLI for the native PE image digest proof")
    assert cli.pe_image_sha256(Path(executable)) == IMAGE_SHA256
