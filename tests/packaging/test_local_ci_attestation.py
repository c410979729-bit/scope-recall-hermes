"""Offline tests for verifying the release signature and each platform's CI receipts."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

SCRIPT = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "verify_local_ci.py"
SPEC = importlib.util.spec_from_file_location("verify_local_ci", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
PREDICATE = "https://github.com/c410979729-bit/scope-recall-hermes/attestations/local-ci/"
VERSION = "1.2.3"
TAG = "v" + VERSION
REPO = "807"
COMMIT, TREE, TAG_OBJECT = "a" * 40, "b" * 40, "c" * 40


def json_bytes(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def b64(data):
    return base64.b64encode(data).decode("ascii")


def signed(statement, key):
    # Build DSSE from its wire specification, independently of the release verifier.
    payload = json_bytes(statement)
    media = b"application/vnd.in-toto+json"
    pae = b"DSSEv1 %d %s %d %s" % (len(media), media, len(payload), payload)
    return {
        "payloadType": media.decode(),
        "payload": b64(payload),
        "signatures": [{"keyid": sha(public(key)), "sig": b64(key.sign(pae))}],
    }


def distributions(folder, *, version=VERSION, project="hermes-scope-recall", wheel_changes=None, sdist_changes=None):
    """Build two matching anonymous archives, with independently computed wheel RECORD data."""
    stem = "hermes_scope_recall-" + VERSION
    headers = f"Metadata-Version: 2.3\nName: {project}\nVersion: {version}\n\n".encode()
    common = {
        "__init__.py": b'"""Synthetic package."""\n',
        "_version.py": f"__version__ = {VERSION!r}\n".encode(),
        "core/capture.py": b"def record_event():\n    return 'synthetic'\n",
        "core/recall.py": b"def recall():\n    return []\n",
        "maintenance/cli.py": b"def main():\n    return 0\n",
    }
    dist_info = stem + ".dist-info/"
    wheel = {"scope_recall/" + n: data for n, data in common.items()}
    wheel[dist_info + "METADATA"] = headers
    wheel[dist_info + "WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
    wheel.update(wheel_changes or {})
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in wheel.items():
        if data is not None and name != dist_info + "RECORD":
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            writer.writerow((name, "sha256=" + encoded, len(data)))
    writer.writerow((dist_info + "RECORD", "", ""))
    wheel.setdefault(dist_info + "RECORD", record.getvalue().encode())
    with zipfile.ZipFile(folder / (stem + "-py3-none-any.whl"), "w") as f:
        for name, data in wheel.items():
            if data is not None:
                f.writestr(name, data)
    sdist = {
        **common,
        "PKG-INFO": headers,
        "pyproject.toml": f"[project]\nname='hermes-scope-recall'\nversion='{VERSION}'\n".encode(),
        **(sdist_changes or {}),
    }
    with tarfile.open(folder / (stem + ".tar.gz"), "w:gz") as f:
        for name, data in sdist.items():
            if data is not None:
                info = tarfile.TarInfo(stem + "/" + name)
                info.size = len(data)
                f.addfile(info, io.BytesIO(data))


class Candidate:
    """Keep synthetic signing identities in process and separate from the files under review."""

    def __init__(self, root):
        self.dist = root / "dist"
        self.dist.mkdir()
        distributions(self.dist)
        self.policy_file = root / "policy.json"
        self.bundle_file = root / "bundle.json"
        self.keys = {lane: Ed25519PrivateKey.generate() for lane in ("release", "linux-unit", "windows-install")}
        self.skipped = sha(json_bytes(["tests.synthetic::test_optional", "feature absent"]))
        self.source = {
            "repo_id": REPO,
            "tag": TAG,
            "version": VERSION,
            "commit": COMMIT,
            "tree": TREE,
            "source_sha256": sha(b"tracked inputs"),
            "lock_sha256": sha(b"lock"),
            "rules_sha256": sha(b"rules"),
        }
        self.policy = {
            "schema": "scope-recall/local-ci-policy/v1",
            "id": "test-policy",
            "source": self.source,
            "release_key": b64(public(self.keys["release"])),
            "lanes": {},
        }
        for lane, platform, kind in (("linux-unit", "linux", "pytest"), ("windows-install", "windows", "command")):
            self.policy["lanes"][lane] = {
                "key": b64(public(self.keys[lane])),
                "platform": platform,
                "kind": kind,
                "python": "3.12",
                "selection_sha256": sha(json_bytes(["synthetic/file"])),
                "cases_sha256": sha(json_bytes(["tests.synthetic::test_optional", "tests.synthetic::test_required"])),
                "case_count": 2,
                "allowed_skips": [self.skipped],
            }

    def evidence(self, *, edit_lane=None, edit_release=None):
        self.policy_file.write_bytes(json_bytes(self.policy))
        subjects = [
            {"name": p.name, "digest": {"sha256": sha(p.read_bytes())}, "annotations": {"size": p.stat().st_size}}
            for p in sorted(self.dist.iterdir(), key=lambda p: p.name)
        ]
        common = {"source": dict(self.source), "policy_sha256": sha(self.policy_file.read_bytes())}

        def statement(role, predicate):
            return {
                "_type": "https://in-toto.io/Statement/v1",
                "predicateType": PREDICATE + role + "/v1",
                "subject": subjects,
                "predicate": predicate,
            }

        envelopes = {}
        for lane, rule in self.policy["lanes"].items():
            result = {
                **common,
                "lane": lane,
                "platform": rule["platform"],
                "python_version": "3.12.9",
                "selection_sha256": rule["selection_sha256"],
                "cases_sha256": rule["cases_sha256"],
                "passed": 1,
                "skips": [self.skipped],
                "exit_codes": {"runner": 0, rule["kind"]: 0, "cleanup": 0},
                "log_sha256": sha(b"synthetic captured output"),
            }
            msg = statement("lane", result)
            if edit_lane and lane == "linux-unit":
                edit_lane(msg)
            envelopes[lane] = signed(msg, self.keys[lane])
        final = statement(
            "release",
            {**common, "tag_object": TAG_OBJECT, "lanes": {k: sha(json_bytes(v)) for k, v in envelopes.items()}},
        )
        if edit_release:
            edit_release(final)
        bundle = {
            "schema": "scope-recall/local-ci-bundle/v1",
            "release": signed(final, self.keys["release"]),
            "lanes": envelopes,
        }
        self.bundle_file.write_bytes(json_bytes(bundle))
        return bundle

    def verify(self, **overrides):
        kwargs = {
            "policy": self.policy_file,
            "bundle": self.bundle_file,
            "dist_dir": self.dist,
            "repo_id": REPO,
            "tag": TAG,
            "commit": COMMIT,
            "tree": TREE,
            "tag_object": TAG_OBJECT,
        }
        return verifier.verify(**(kwargs | overrides))


@pytest.fixture
def candidate(tmp_path):
    return Candidate(tmp_path)


def test_release_verifies_original_bytes_and_role_scoped_lanes(candidate):
    candidate.evidence()
    wanted = {
        "policy_id": "test-policy",
        "policy_sha256": sha(candidate.policy_file.read_bytes()),
        "bundle_sha256": sha(candidate.bundle_file.read_bytes()),
        "lanes": 2,
    }
    assert candidate.verify() == wanted
    cli = [
        sys.executable,
        "-I",
        "-B",
        str(SCRIPT),
        "--policy",
        str(candidate.policy_file),
        "--bundle",
        str(candidate.bundle_file),
        "--dist-dir",
        str(candidate.dist),
        "--repo-id",
        REPO,
        "--tag",
        TAG,
        "--commit",
        COMMIT,
        "--tree",
        TREE,
        "--tag-object",
        TAG_OBJECT,
    ]
    out = subprocess.run(cli, capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(out.stdout) == wanted


@pytest.mark.parametrize(
    "field,value",
    [
        ("repo_id", "808"),
        ("tag", "v1.2.4"),
        ("commit", "d" * 40),
        ("tree", "e" * 40),
        ("tag_object", "f" * 40),
        ("repo_id", "0807"),
        ("tag", "v01.2.3"),
    ],
)
def test_external_source_and_tag_must_match(candidate, field, value):
    candidate.evidence()
    with pytest.raises(ValueError):
        candidate.verify(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", 0),
        ("passed", True),
        ("passed", 2),
        ("skips", []),
        ("skips", ["0" * 64]),
        ("platform", "macos"),
        ("python_version", "3.11.9"),
        ("lane", "windows-install"),
        ("selection_sha256", "1" * 64),
        ("cases_sha256", "2" * 64),
        ("policy_sha256", "3" * 64),
        ("log_sha256", ""),
        ("log_sha256", sha(b"")),
    ],
)
def test_even_signed_evidence_must_meet_the_actual_policy(candidate, field, value):
    candidate.evidence(edit_lane=lambda s: s["predicate"].update({field: value}))
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize("field", ["selection_sha256", "cases_sha256"])
@pytest.mark.parametrize("empty", [b"", b"[]"])
def test_signed_nonzero_count_cannot_promote_an_empty_manifest(candidate, field, empty):
    candidate.policy["lanes"]["linux-unit"][field] = sha(empty)
    candidate.evidence()
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize("field", ["runner", "pytest", "cleanup"])
@pytest.mark.parametrize("value", [1, False])
def test_all_exit_codes_are_genuine_integer_zero(candidate, field, value):
    candidate.evidence(edit_lane=lambda s: s["predicate"]["exit_codes"].update({field: value}))
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(predicateType=PREDICATE + "release/v1"),
        lambda s: s["predicate"].update(source=dict(s["predicate"]["source"], source_sha256="4" * 64)),
        lambda s: s["predicate"]["skips"].append(s["predicate"]["skips"][0]),
        lambda s: s["predicate"].update(private_data="unrecognized"),
        lambda s: s["subject"][0].update(name="../distribution.whl"),
    ],
)
def test_lane_cannot_change_subject_source_authority_or_schema(candidate, change):
    candidate.evidence(edit_lane=change)
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["predicate"]["lanes"].pop("windows-install"),
        lambda s: s["predicate"]["lanes"].update({"linux-unit": "0" * 64}),
        lambda s: s["predicate"].update(tag_object="0" * 40),
        lambda s: s["predicate"].update(source=dict(s["predicate"]["source"], lock_sha256="0" * 64)),
    ],
)
def test_aggregate_must_endorse_every_lane_on_this_tag(candidate, change):
    candidate.evidence(edit_release=change)
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize("part", ["lane", "release", "unsigned", "duplicate", "substituted-key"])
def test_signers_and_envelopes_are_not_interchangeable(candidate, part):
    bundle = candidate.evidence()
    lane, final = bundle["lanes"]["linux-unit"], bundle["release"]
    if part == "lane":
        lane["signatures"] = final["signatures"]
    elif part == "release":
        final["signatures"] = lane["signatures"]
    elif part == "unsigned":
        final["signatures"] = []
    elif part == "duplicate":
        final["signatures"] *= 2
    else:
        # The key ID alone grants no authority to a fresh key.
        other = signed(json.loads(base64.b64decode(final["payload"])), Ed25519PrivateKey.generate())
        final["signatures"][0]["sig"] = other["signatures"][0]["sig"]
    candidate.bundle_file.write_bytes(json_bytes(bundle))
    with pytest.raises(ValueError):
        candidate.verify()


def test_policy_cannot_assign_one_key_to_distinct_roles(candidate):
    candidate.policy["lanes"]["linux-unit"]["key"] = candidate.policy["lanes"]["windows-install"]["key"]
    candidate.evidence()
    with pytest.raises(ValueError, match="roles"):
        candidate.verify()
    candidate.policy["lanes"]["linux-unit"]["key"] = candidate.policy["release_key"]
    candidate.evidence()
    with pytest.raises(ValueError, match="roles"):
        candidate.verify()


def test_receipts_are_not_reusable_with_updated_policy_bytes(candidate):
    candidate.evidence()
    candidate.policy_file.write_bytes(candidate.policy_file.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="policy"):
        candidate.verify()


@pytest.mark.parametrize("bad", [b'{"schema":0,"schema":1}', b'{"x":NaN}', b'{"x":Infinity}', b"[]", b"\xff"])
def test_decoder_refuses_ambiguous_evidence(candidate, bad):
    candidate.evidence()
    candidate.bundle_file.write_bytes(bad)
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize("change", ["missing", "extra", "changed"])
@pytest.mark.parametrize("suffix", ["*.whl", "*.tar.gz"])
def test_hashes_and_exact_package_inventory(candidate, change, suffix):
    candidate.evidence()
    package = next(candidate.dist.glob(suffix))
    if change == "missing":
        package.rename(candidate.dist.parent / package.name)
    elif change == "extra":
        (candidate.dist / "additional.txt").write_bytes(b"synthetic")
    else:
        with package.open("ab") as f:
            f.write(b"changed")
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize("version,project", [("1.2.4", "hermes-scope-recall"), (VERSION, "other-project")])
def test_signed_packages_must_describe_the_authorized_project(candidate, version, project):
    distributions(candidate.dist, version=version, project=project)
    candidate.evidence()
    with pytest.raises(ValueError, match="metadata"):
        candidate.verify()


def test_shared_collector_apis_validate_before_aggregate_signing(candidate):
    bundle = candidate.evidence()
    rules, policy_sha = verifier.document(candidate.policy_file)
    verifier.validate_policy(rules, REPO, TAG, COMMIT, TREE)
    subjects = verifier.subjects(candidate.dist, VERSION)
    aggregate = json.loads(base64.b64decode(bundle["release"]["payload"]))["predicate"]
    assert verifier.verify_lanes(bundle["lanes"], rules, policy_sha, subjects) == aggregate["lanes"]
    assert verifier.pae(b"{}") == b"DSSEv1 28 application/vnd.in-toto+json 2 {}"
    bundle["lanes"].pop("linux-unit")
    with pytest.raises(ValueError, match="lane set"):
        verifier.verify_lanes(bundle["lanes"], rules, policy_sha, subjects)


@pytest.mark.parametrize(
    "member",
    [
        "unrelated/../METADATA",
        "./extra",
        "pkg//entry",
        "extra.",
        "z:member",
        "HERMES_SCOPE_RECALL-1.2.3.DIST-INFO/metadata",
    ],
)
def test_signed_distribution_cannot_expose_different_archive_paths(candidate, member):
    with zipfile.ZipFile(next(candidate.dist.glob("*.whl")), "a") as wheel:
        wheel.writestr(member, b"synthetic fixture")
    candidate.evidence()
    with pytest.raises(ValueError, match="archive path"):
        candidate.verify()


def test_static_check_rejects_sdist_with_a_different_version(candidate):
    headers = b"Name: hermes-scope-recall\nVersion: 2.0.0\n\n"
    distributions(candidate.dist, sdist_changes={"PKG-INFO": headers})
    candidate.evidence()
    with pytest.raises(ValueError, match="metadata name or version"):
        candidate.verify()


@pytest.mark.parametrize("product", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "member", ["__init__.py", "_version.py", "core/capture.py", "core/recall.py", "maintenance/cli.py"]
)
def test_both_archives_must_contain_the_release_code(candidate, product, member):
    key = "scope_recall/" + member if product == "wheel" else member
    distributions(candidate.dist, **{product + "_changes": {key: None}})
    candidate.evidence()
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize(
    "product,member,replacement",
    [
        ("wheel", "hermes_scope_recall-1.2.3.dist-info/WHEEL", None),
        ("wheel", "hermes_scope_recall-1.2.3.dist-info/RECORD", None),
        ("wheel", "hermes_scope_recall-1.2.3.dist-info/WHEEL", b"Tag: py2-none-any\n"),
        ("wheel", "hermes_scope_recall-1.2.3.dist-info/RECORD", b"empty,broken,1\n"),
        ("wheel", "scope_recall/_version.py", b"__version__='2.0.0'\n"),
        ("sdist", "_version.py", b"__version__='2.0.0'\n"),
        ("sdist", "pyproject.toml", None),
        ("sdist", "pyproject.toml", b"[project]\nname='other'\nversion='1.2.3'\n"),
        ("sdist", "core/capture.py", b"\n"),
        ("sdist", "core/recall.py", b"def recall():\n    return 'different bytes'\n"),
    ],
)
def test_even_a_signed_package_must_have_consistent_release_contents(candidate, product, member, replacement):
    distributions(candidate.dist, **{product + "_changes": {member: replacement}})
    candidate.evidence()
    with pytest.raises(ValueError):
        candidate.verify()


@pytest.mark.parametrize(
    "name,data",
    [
        ("_version.py", b"__version__='2.0.0'\n"),
        ("_version.py", b"__version__='1.2.3'\n__version__=read_version()\n"),
        ("core/recall.py", b'"""Only a package marker."""\n'),
        ("core/capture.py", b"def other():\n    pass\n"),
    ],
)
def test_matching_archives_still_need_the_actual_version_and_public_entry_points(candidate, name, data):
    distributions(candidate.dist, wheel_changes={"scope_recall/" + name: data}, sdist_changes={name: data})
    candidate.evidence()
    with pytest.raises(ValueError):
        candidate.verify()


def test_record_checks_the_hash_of_each_file_inside_the_signed_wheel(candidate):
    name = "hermes_scope_recall-1.2.3.dist-info/RECORD"
    with zipfile.ZipFile(next(candidate.dist.glob("*.whl"))) as wheel:
        record = wheel.read(name).replace(b"sha256=", b"sha256=wrong", 1)
    distributions(candidate.dist, wheel_changes={name: record})
    candidate.evidence()
    with pytest.raises(ValueError, match="RECORD mismatch"):
        candidate.verify()


@pytest.mark.parametrize("name", ["directory\\entry", "prefix\x00suffix"])
def test_archive_paths_reject_platform_separators_and_nul(name):
    # Windows' ZIP writer normalizes backslashes when creating a fixture.
    with pytest.raises(ValueError, match="archive path"):
        verifier._member(name, set())


def test_bundle_cannot_refer_to_file_through_link(candidate, monkeypatch):
    candidate.evidence()
    # Unprivileged Windows users cannot create symlinks. Model the file type
    # returned by lstat so the refusal is exercised on every supported host.
    original = Path.lstat

    def lstat(path):
        info = original(path)
        if path == candidate.bundle_file:
            return os.stat_result((stat.S_IFLNK, *tuple(info)[1:]))
        return info

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="regular file"):
        candidate.verify()
