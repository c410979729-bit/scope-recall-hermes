#!/usr/bin/env python3
"""Verify offline CI evidence for an exact tagged source and its two distributions."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tomllib
import unicodedata
import zipfile
from contextlib import ExitStack, contextmanager
from email.parser import BytesHeaderParser
from pathlib import Path
from typing import Any, BinaryIO

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

POLICY_SCHEMA = "scope-recall/local-ci-policy/v1"
BUNDLE_SCHEMA = "scope-recall/local-ci-bundle/v1"
STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"
PREDICATE = "https://github.com/c410979729-bit/scope-recall-hermes/attestations/local-ci/"
SOURCE_FIELDS = "repo_id tag version commit tree source_sha256 lock_sha256 rules_sha256"
LANE_FIELDS = "key platform python kind selection_sha256 cases_sha256 case_count allowed_skips"
RESULT_FIELDS = (
    "source policy_sha256 lane platform python_version selection_sha256 cases_sha256 passed skips exit_codes log_sha256"
)
MAX_JSON = 16 * 1024 * 1024
MAX_METADATA = 256 * 1024
MAX_CORE_SOURCE = 4 * 1024 * 1024
MAX_EXPANDED_PACKAGE = 256 * 1024 * 1024
CORE_ENTRY_POINTS = {"core/capture.py": "record_event", "core/recall.py": "recall", "maintenance/cli.py": "main"}
CORE_FILES = ("__init__.py", "_version.py", *CORE_ENTRY_POINTS)
EMPTY_LOG = hashlib.sha256(b"").hexdigest()
EMPTY_MANIFEST = hashlib.sha256(b"[]").hexdigest()


def require(ok: bool, message: str) -> None:
    """Reject incomplete evidence without logging untrusted payloads or file contents."""
    if not ok:
        raise ValueError(message)


def canonical(value: object) -> bytes:
    """Serialize receipt references independently of the envelope's outer JSON formatting."""
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(_number: str) -> None:
    raise ValueError("non-finite JSON number")


def decode(data: bytes) -> dict:
    """Require one UTF-8 JSON object, with exactly one interpretation of each key."""
    require(0 < len(data) <= MAX_JSON, "invalid evidence size")
    value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
    require(type(value) is dict, "JSON object required")
    return value


def fields(value: Any, names: str) -> None:
    require(type(value) is dict and set(value) == set(names.split()), "unexpected or missing evidence fields")


def matches(value: object, pattern: str) -> bool:
    return isinstance(value, str) and re.fullmatch(pattern, value, re.ASCII) is not None


def sha256(value: object) -> bool:
    return matches(value, "[0-9a-f]{64}")


def manifest_sha256(value: object) -> bool:
    return sha256(value) and value not in (EMPTY_LOG, EMPTY_MANIFEST)


def b64(value: object, length: int | None = None) -> bytes:
    require(isinstance(value, str), "base64 string required")
    raw = base64.b64decode(value, validate=True)
    require(base64.b64encode(raw).decode("ascii") == value, "noncanonical base64")
    require(length is None or len(raw) == length, "incorrect key or signature length")
    return raw


def _link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def regular(path: Path) -> BinaryIO:
    """Open only a regular file, refusing links and a substituted directory entry."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and not _link(before), "regular file required")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    stream = os.fdopen(os.open(path, flags), "rb")
    if not os.path.samestat(before, os.fstat(stream.fileno())):
        stream.close()
        raise ValueError("input changed while opening")
    return stream


def document(path: Path) -> tuple[dict, str]:
    with regular(path) as stream:
        data = stream.read(MAX_JSON + 1)
    return decode(data), digest(data)


def hashes(values: Any) -> set:
    require(isinstance(values, list) and all(sha256(v) for v in values), "expected a list of SHA-256 digests")
    require(len(set(values)) == len(values), "duplicate digest in list")
    return set(values)


def validate_policy(
    policy: dict, repo_id: str, tag: str, commit: str, tree: str, tag_object: str | None = None
) -> None:
    """Validate the out-of-band authority and its source, keys, and complete approved matrix."""
    fields(policy, "schema id source release_key lanes")
    require(
        policy["schema"] == POLICY_SCHEMA and matches(policy["id"], r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"),
        "invalid policy",
    )
    source = policy["source"]
    fields(source, SOURCE_FIELDS)
    require(
        matches(repo_id, r"[1-9][0-9]*") and matches(tag, r"v(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}"),
        "invalid release identity",
    )
    require(all(matches(v, "[0-9a-f]{40}") for v in (commit, tree)), "invalid Git object ID")
    if tag_object is not None:
        require(matches(tag_object, "[0-9a-f]{40}"), "invalid annotated tag object ID")
    require(
        [source[n] for n in ("repo_id", "tag", "version", "commit", "tree")] == [repo_id, tag, tag[1:], commit, tree],
        "release differs from policy",
    )
    require(all(sha256(source[n]) for n in ("source_sha256", "lock_sha256", "rules_sha256")), "invalid source digest")
    roles = {b64(policy["release_key"], 32): "release"}
    lanes = policy["lanes"]
    require(isinstance(lanes, dict) and 0 < len(lanes) <= 256, "empty or excessive lane policy")
    for name, lane in lanes.items():
        require(matches(name, r"[a-z0-9][a-z0-9._-]{0,127}"), "invalid lane name")
        fields(lane, LANE_FIELDS)
        platform = lane["platform"]
        require(
            platform in ("linux", "macos", "windows") and lane["kind"] in ("pytest", "command"),
            "invalid lane platform or kind",
        )
        require(matches(lane["python"], r"[1-9][0-9]*\.(?:0|[1-9][0-9]*)"), "invalid Python policy")
        require(roles.setdefault(b64(lane["key"], 32), platform) == platform, "public key reused across roles")
        require(
            all(manifest_sha256(lane[k]) for k in ("selection_sha256", "cases_sha256")), "empty or invalid selection"
        )
        require(type(lane["case_count"]) is int and lane["case_count"] > 0, "empty case policy")
        hashes(lane["allowed_skips"])


def pae(payload: bytes) -> bytes:
    """Construct the exact bytes signed by both the platform and the release identity."""
    encoded_type = PAYLOAD_TYPE.encode("ascii")
    return b" ".join([b"DSSEv1", str(len(encoded_type)).encode(), encoded_type, str(len(payload)).encode(), payload])


def signed(envelope: dict, key: str, role: str, subject: list) -> dict:
    """Verify DSSE pre-authentication encoding before interpreting the signed in-toto statement."""
    fields(envelope, "payloadType payload signatures")
    require(envelope["payloadType"] == PAYLOAD_TYPE, "unexpected DSSE payload type")
    signatures = envelope["signatures"]
    require(isinstance(signatures, list) and len(signatures) == 1, "exactly one role signature required")
    signature = signatures[0]
    fields(signature, "keyid sig")
    public = b64(key, 32)
    require(signature["keyid"] == digest(public), "unexpected signature key")
    payload = b64(envelope["payload"])
    try:
        Ed25519PublicKey.from_public_bytes(public).verify(b64(signature["sig"], 64), pae(payload))
    except InvalidSignature as exc:
        raise ValueError("invalid evidence signature") from exc
    statement = decode(payload)
    fields(statement, "_type subject predicateType predicate")
    require(
        statement["_type"] == STATEMENT_TYPE and statement["predicateType"] == PREDICATE + role + "/v1",
        "unexpected statement type",
    )
    # Comparing encoded JSON distinguishes a byte count of 1 from the boolean true.
    require(canonical(statement["subject"]) == canonical(subject), "distribution bytes or identities differ")
    return statement["predicate"]


def lane_result(result: dict, name: str, lane: dict) -> None:
    """Require a complete successful run of exactly the cases authorized for this platform."""
    fields(result, RESULT_FIELDS)
    require(result["lane"] == name and result["platform"] == lane["platform"], "lane identity mismatch")
    require(
        matches(result["python_version"], re.escape(lane["python"]) + r"\.(?:0|[1-9][0-9]*)"), "wrong Python runtime"
    )
    require(
        all(manifest_sha256(result[k]) and result[k] == lane[k] for k in ("selection_sha256", "cases_sha256")),
        "empty or unapproved test selection",
    )
    require(sha256(result["log_sha256"]) and result["log_sha256"] != EMPTY_LOG, "missing execution log")
    exits = result["exit_codes"]
    fields(exits, f"runner {lane['kind']} cleanup")
    require(all(type(v) is int and v == 0 for v in exits.values()), "unsuccessful execution or cleanup")
    skips = hashes(result["skips"])
    require(skips <= set(lane["allowed_skips"]), "unapproved skip")
    passed = result["passed"]
    require(type(passed) is int and passed > 0 and passed + len(skips) == lane["case_count"], "incomplete execution")


def verify_lanes(receipts: dict, policy: dict, policy_sha: str, subject: list) -> dict[str, str]:
    """Return the receipt hashes a release signer may endorse, after validating every required lane."""
    require(type(receipts) is dict and set(receipts) == set(policy["lanes"]), "incomplete lane set")
    for name, lane in policy["lanes"].items():
        result = signed(receipts[name], lane["key"], "lane", subject)
        lane_result(result, name, lane)
        require(
            result["source"] == policy["source"] and result["policy_sha256"] == policy_sha,
            "lane source or policy mismatch",
        )
    return {name: digest(canonical(receipt)) for name, receipt in receipts.items()}


def verify_statements(bundle: dict, policy: dict, policy_sha: str, tag_object: str, subject: list) -> None:
    """Check the role-separated signatures, annotated tag, and exact links from release to platform receipts."""
    fields(bundle, "schema release lanes")
    require(bundle["schema"] == BUNDLE_SCHEMA, "unknown bundle schema")
    aggregate = signed(bundle["release"], policy["release_key"], "release", subject)
    fields(aggregate, "source policy_sha256 tag_object lanes")
    require(
        matches(tag_object, "[0-9a-f]{40}") and aggregate["tag_object"] == tag_object, "annotated tag object mismatch"
    )
    require(
        aggregate["source"] == policy["source"] and aggregate["policy_sha256"] == policy_sha,
        "aggregate source or policy mismatch",
    )
    linked = verify_lanes(bundle["lanes"], policy, policy_sha, subject)
    require(aggregate["lanes"] == linked, "aggregate does not bind exactly these receipts")


def _member(name: str, seen: set[str]) -> str:
    """Reject archive aliases that different extractors could resolve to the same installed file."""
    name = name.removesuffix("/")
    parts = name.split("/")
    require(
        all(p and p == p.rstrip(" .") and not any(c in p for c in "\\\x00:") for p in parts), "ambiguous archive path"
    )
    normalized = unicodedata.normalize("NFC", name.casefold())
    require(normalized not in seen, "duplicate archive path")
    seen.add(normalized)
    return name


def _metadata(raw: bytes, version: str) -> None:
    headers = BytesHeaderParser().parsebytes(raw)
    require(
        not headers.defects
        and headers.get_all("Name") == ["hermes-scope-recall"]
        and headers.get_all("Version") == [version],
        "metadata name or version mismatch",
    )


def _wheel_record(archive: zipfile.ZipFile, files: dict, record_name: str, raw: bytes) -> None:
    """Match RECORD to every member, including the mandatory unhashed self-reference."""
    seen = set()
    try:
        for row in csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True):
            require(len(row) == 3, "invalid wheel RECORD")
            name, encoded, size = row
            require(name in files and name not in seen, "unexpected wheel RECORD entry")
            seen.add(name)
            if name == record_name:
                require(not encoded and not size, "invalid wheel RECORD self-reference")
                continue
            with archive.open(files[name]) as member:
                actual = base64.urlsafe_b64encode(hashlib.file_digest(member, "sha256").digest()).decode().rstrip("=")
            require(encoded == "sha256=" + actual and size == str(files[name].file_size), "wheel RECORD mismatch")
    except csv.Error as exc:
        raise ValueError("invalid wheel RECORD") from exc
    require(seen == files.keys(), "incomplete wheel RECORD")


def _wheel(stream: BinaryIO, stem: str, version: str) -> dict[str, bytes]:
    info = stem + ".dist-info/"
    seen: set[str] = set()
    with zipfile.ZipFile(stream) as archive:
        files = {}
        for item in archive.infolist():
            name = _member(item.orig_filename, seen)
            require(not stat.S_ISLNK(item.external_attr >> 16), "link in wheel")
            if not item.is_dir():
                files[name] = item
        require(sum(item.file_size for item in files.values()) <= MAX_EXPANDED_PACKAGE, "excessive wheel size")
        require([n for n in files if n.endswith(".dist-info/METADATA")] == [info + "METADATA"], "unexpected metadata")

        def read(name: str, limit: int) -> bytes:
            require(name in files and 0 < files[name].file_size <= limit, "missing or invalid wheel member")
            return archive.read(files[name])

        _metadata(read(info + "METADATA", MAX_METADATA), version)
        wheel = BytesHeaderParser().parsebytes(read(info + "WHEEL", MAX_METADATA))
        require(
            not wheel.defects
            and wheel.get_all("Wheel-Version") == ["1.0"]
            and wheel.get_all("Root-Is-Purelib") == ["true"]
            and wheel.get_all("Tag") == ["py3-none-any"],
            "invalid wheel build metadata",
        )
        _wheel_record(archive, files, info + "RECORD", read(info + "RECORD", MAX_METADATA))
        return {n: read("scope_recall/" + n, MAX_CORE_SOURCE) for n in CORE_FILES}


def _sdist_project(raw: bytes, version: str) -> None:
    config = tomllib.loads(raw.decode("utf-8"))
    project = config.get("project")
    require(isinstance(project, dict) and project.get("name") == "hermes-scope-recall", "sdist project mismatch")
    dynamic = project.get("dynamic", [])
    require(isinstance(dynamic, list), "invalid project metadata")
    if "version" not in dynamic:
        require(project.get("version") == version, "sdist project version mismatch")
        return
    binding = config
    for key in ("tool", "setuptools", "dynamic", "version"):
        binding = binding.get(key) if isinstance(binding, dict) else None
    require(
        "version" not in project and binding == {"attr": "scope_recall._version.__version__"},
        "sdist version source mismatch",
    )


def _sdist(stream: BinaryIO, stem: str, version: str) -> dict[str, bytes]:
    limits = dict.fromkeys(CORE_FILES, MAX_CORE_SOURCE) | {"PKG-INFO": MAX_METADATA, "pyproject.toml": MAX_METADATA}
    seen: set[str] = set()
    contents = {}
    total = 0
    # Read the tar in order, keeping only the metadata and public source files.
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for item in archive:
            name = _member(item.name, seen)
            require(
                (name == stem or name.startswith(stem + "/")) and (item.isreg() or item.isdir()), "invalid sdist member"
            )
            total += item.size
            require(total <= MAX_EXPANDED_PACKAGE, "excessive sdist size")
            relative = name.removeprefix(stem + "/")
            if relative in limits:
                require(item.isreg() and 0 < item.size <= limits[relative], "invalid sdist project file")
                member = archive.extractfile(item)
                require(member is not None, "missing sdist member")
                contents[relative] = member.read()
    require(contents.keys() == limits.keys(), "missing sdist project file")
    _metadata(contents["PKG-INFO"], version)
    _sdist_project(contents["pyproject.toml"], version)
    return {name: contents[name] for name in CORE_FILES}


def _core_sources(contents: dict[str, bytes], version: str) -> None:
    """Check the static version and public capture, recall and CLI declarations without importing the candidate."""
    try:
        module = ast.parse(contents["_version.py"])
        versions = [
            node.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        ]
        require(
            len(versions) == 1 and isinstance(versions[0], ast.Constant) and versions[0].value == version,
            "package source version mismatch",
        )
        for name, entry in CORE_ENTRY_POINTS.items():
            body = ast.parse(contents[name]).body
            require(any(isinstance(n, ast.FunctionDef) and n.name == entry for n in body), "missing core entry point")
    except SyntaxError as exc:
        raise ValueError("invalid package source") from exc


@contextmanager
def _products(dist_dir: Path, version: str):
    """Hash both input descriptors, then check their static metadata after any caller signature checks."""
    require(matches(version, r"(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*)){2}"), "invalid product version")
    stem = f"hermes_scope_recall-{version}"
    wheel, sdist = stem + "-py3-none-any.whl", stem + ".tar.gz"
    info = dist_dir.lstat()
    require(stat.S_ISDIR(info.st_mode) and not _link(info), "regular distribution directory required")
    require({p.name for p in dist_dir.iterdir()} == {wheel, sdist}, "distribution inventory mismatch")
    with ExitStack() as stack:
        subject = []
        streams = {}
        for name in sorted((wheel, sdist)):
            stream = stack.enter_context(regular(dist_dir / name))
            size = os.fstat(stream.fileno()).st_size
            require(size > 0, "empty distribution")
            subject.append(
                {
                    "name": name,
                    "digest": {"sha256": hashlib.file_digest(stream, "sha256").hexdigest()},
                    "annotations": {"size": size},
                }
            )
            stream.seek(0)
            streams[name] = stream
        yield subject
        wheel_core = _wheel(streams[wheel], stem, version)
        require(wheel_core == _sdist(streams[sdist], stem, version), "core package files differ")
        _core_sources(wheel_core, version)


def subjects(dist_dir: Path, version: str) -> list[dict]:
    """Read the exact in-toto subjects and validate both package identities for the trusted collector."""
    with _products(dist_dir, version) as subject:
        return subject


def verify(
    policy: Path, bundle: Path, dist_dir: Path, repo_id: str, tag: str, commit: str, tree: str, tag_object: str
) -> dict:
    """Verify one authorized release without evaluating any candidate code or writing state."""
    rules, policy_sha = document(policy)
    evidence, bundle_sha = document(bundle)
    validate_policy(rules, repo_id, tag, commit, tree, tag_object)
    with _products(dist_dir, rules["source"]["version"]) as subject:
        verify_statements(evidence, rules, policy_sha, tag_object, subject)
    return {
        "policy_id": rules["id"],
        "policy_sha256": policy_sha,
        "bundle_sha256": bundle_sha,
        "lanes": len(rules["lanes"]),
    }


def main() -> int:
    """Print only a receipt for verified inputs, and return failure for malformed evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("policy", "bundle", "dist-dir"):
        parser.add_argument("--" + arg, type=Path, required=True)
    for arg in ("repo-id", "tag", "commit", "tree", "tag-object"):
        parser.add_argument("--" + arg, required=True)
    args = parser.parse_args()
    try:
        summary = verify(**vars(args))
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        OverflowError,
        RecursionError,
        zipfile.BadZipFile,
        tarfile.TarError,
        EOFError,
    ):
        parser.exit(1, "Local CI evidence or distributions failed verification.\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
