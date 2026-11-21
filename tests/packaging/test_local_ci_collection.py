"""Synthetic qualification records, detached signing and exact release-bundle verification."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))
import collect_local_ci as collector  # noqa: E402


def _public(key):
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _b64(data):
    return base64.b64encode(data).decode("ascii")


def _seal(directory, key):
    raw = (directory / "statement.json").read_bytes()
    return {
        "payloadType": collector.v.PAYLOAD_TYPE,
        "payload": _b64(raw),
        "signatures": [
            {"keyid": collector.v.digest(_public(key)), "sig": _b64(key.sign((directory / "request.pae").read_bytes()))}
        ],
    }


def _package(root):
    dist = root / "dist"
    dist.mkdir()
    site = root / "site"
    modules = {
        "__init__.py": b'"""Synthetic fixture."""\n',
        "_version.py": b'__version__ = "1.0.0"\n',
        "core/capture.py": b"def record_event(): pass\n",
        "core/recall.py": b"def recall(): pass\n",
        "maintenance/cli.py": b"def main(): pass\n",
    }
    stem = "hermes_scope_recall-1.0.0"
    info = stem + ".dist-info/"
    headers = b"Metadata-Version: 2.3\nName: hermes-scope-recall\nVersion: 1.0.0\n\n"
    files = {"scope_recall/" + name: raw for name, raw in modules.items()}
    for name, raw in files.items():
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    files[info + "METADATA"] = headers
    files[info + "WHEEL"] = b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, raw in files.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + encoded, len(raw)])
    writer.writerow([info + "RECORD", "", ""])
    files[info + "RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(dist / (stem + "-py3-none-any.whl"), "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    source = {
        **modules,
        "PKG-INFO": headers,
        "pyproject.toml": b'[project]\nname = "hermes-scope-recall"\nversion = "1.0.0"\n',
    }
    with tarfile.open(dist / (stem + ".tar.gz"), "w:gz") as archive:
        for name, raw in source.items():
            entry = tarfile.TarInfo(stem + "/" + name)
            entry.size = len(raw)
            archive.addfile(entry, io.BytesIO(raw))
    return dist, site


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    """Use an isolated source fixture, real package bytes and ephemeral role-specific signing keys."""
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "quality.py").write_text("print('synthetic quality probe executed')\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    dist, site = _package(tmp_path)
    observed = {
        "commit": "a" * 40,
        "tree": "b" * 40,
        "source_sha256": "1" * 64,
        "lock_sha256": "2" * 64,
        "rules_sha256": "3" * 64,
    }
    monkeypatch.setattr(collector, "snapshot", lambda _: observed.copy())
    monkeypatch.setattr(collector, "tag_object", lambda *_: "c" * 40)
    monkeypatch.setattr(
        collector.importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(version="1.0.0", locate_file=lambda p: site / p),
    )
    gate = f"{collector.PLATFORMS[sys.platform]}-py{sys.version_info.major}.{sys.version_info.minor}-quality"
    worker, release = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    selection = collector.selection_digest(["scripts/quality.py"])
    source = {"repo_id": "807", "tag": "v1.0.0", "version": "1.0.0", **observed}
    policy = {
        "schema": collector.v.POLICY_SCHEMA,
        "id": "fixture",
        "release_key": _b64(_public(release)),
        "source": source,
        "lanes": {
            gate: {
                "key": _b64(_public(worker)),
                "platform": collector.PLATFORMS[sys.platform],
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "kind": "command",
                "selection_sha256": selection,
                "cases_sha256": selection,
                "case_count": 1,
                "allowed_skips": [],
            }
        },
    }
    policy_file = tmp_path / "policy.json"
    collector.write_json(policy_file, policy)
    return SimpleNamespace(
        root=root,
        scratch=scratch,
        dist=dist,
        site=site,
        source=source,
        gate=gate,
        worker=worker,
        release=release,
        policy=policy,
        policy_file=policy_file,
    )


def _args(candidate, output, mode, **extra):
    return SimpleNamespace(
        command=mode,
        checkout=candidate.root,
        dist_dir=candidate.dist,
        repo_id="807",
        tag="v1.0.0",
        scratch=candidate.scratch,
        output=output,
        lane=candidate.gate,
        policy=candidate.policy_file,
        **extra,
    )


def _detached(candidate, directory, key, destination):
    signature = directory / "detached.bin"
    signature.write_bytes(key.sign((directory / "request.pae").read_bytes()))
    collector.execute(
        SimpleNamespace(
            command="envelope",
            policy=candidate.policy_file,
            dist_dir=candidate.dist,
            statement=directory / "statement.json",
            signature=signature,
            output=destination,
        )
    )


@pytest.mark.parametrize("signature", [bytes(64), bytes(63), bytes(65)])
def test_detached_signature_must_be_valid_and_exact_size(candidate, tmp_path, signature):
    lane = tmp_path / "job"
    collector.execute(_args(candidate, lane, "collect"))
    raw = tmp_path / "detached.bin"
    raw.write_bytes(signature)
    target = tmp_path / "lane.json"
    flags = SimpleNamespace(
        command="envelope",
        policy=candidate.policy_file,
        dist_dir=candidate.dist,
        statement=lane / "statement.json",
        signature=raw,
        output=target,
    )
    with pytest.raises(ValueError):
        collector.execute(flags)
    assert not target.exists()


def test_independently_signed_observations_are_verified_end_to_end(candidate, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_RELEASE_CANARY", "private-controller-canary")
    gate_program = candidate.root / "scripts" / "quality.py"
    gate_program.write_text(
        "import os\nassert 'TEST_RELEASE_CANARY' not in os.environ\nprint('synthetic quality probe executed')\n",
        encoding="utf-8",
    )
    lane = tmp_path / "lane-job"
    collector.execute(_args(candidate, lane, "collect"))
    assert "synthetic quality probe executed" in (lane / "run.log").read_text("utf-8")
    assert not list(candidate.scratch.iterdir())
    lanes = tmp_path / "signed-lanes"
    lanes.mkdir()
    _detached(candidate, lane, candidate.worker, lanes / (candidate.gate + ".json"))
    aggregate = tmp_path / "aggregate"
    collector.execute(_args(candidate, aggregate, "aggregate", lanes_dir=lanes))
    release_file = tmp_path / "release.json"
    _detached(candidate, aggregate, candidate.release, release_file)
    output = tmp_path / "published-bundle"
    collector.execute(_args(candidate, output, "bundle", lanes_dir=lanes, release=release_file))
    verified = collector.v.verify(
        candidate.policy_file,
        output / "LOCAL-CI.dsse.json",
        candidate.dist,
        "807",
        "v1.0.0",
        candidate.source["commit"],
        candidate.source["tree"],
        "c" * 40,
    )
    assert verified["lanes"] == 1


def test_unapproved_probe_never_creates_signing_material(candidate, tmp_path):
    output = tmp_path / "unsigned"
    collector.execute(_args(candidate, output, "observe"))
    observation = json.loads((output / "observation.json").read_bytes())
    assert observation["result"]["passed"] == 1 and not (output / "request.pae").exists()


@pytest.mark.parametrize(
    "alter", ["changed-commit", "wrong-inventory", "different-installed-bytes", "unmeasured-platform"]
)
def test_altered_lane_yields_no_signing_request(candidate, tmp_path, alter):
    flags = _args(candidate, tmp_path / "rejected", "collect")
    if alter == "changed-commit":
        candidate.policy["source"]["commit"] = "c" * 40
    if alter == "wrong-inventory":
        candidate.policy["lanes"][candidate.gate]["case_count"] = 2
    if alter == "different-installed-bytes":
        (candidate.site / "scope_recall" / "__init__.py").write_text("different = True\n", encoding="utf-8")
    if alter == "unmeasured-platform":
        flags.lane = "unknown-platform"
    candidate.policy_file.write_bytes(collector.v.canonical(candidate.policy))
    with pytest.raises((ValueError, KeyError)):
        collector.execute(flags)
    assert not (flags.output / "request.pae").exists()


def test_state_change_during_measurement_is_rejected(candidate, tmp_path, monkeypatch):
    reads = iter((candidate.source, {**candidate.source, "commit": "e" * 40}))
    monkeypatch.setattr(collector, "snapshot", lambda _: next(reads))
    flags = _args(candidate, tmp_path / "changed", "observe")
    with pytest.raises(ValueError, match="source changed"):
        collector.execute(flags)
    assert not (flags.output / "observation.json").exists()


def test_skip_identity_and_reason_are_bound_to_the_actual_case(tmp_path):
    xml = tmp_path / "results.xml"
    xml.write_text(
        '<testsuite><testcase classname="tests.host" name="test_one"/>'
        '<testcase classname="tests.host" name="test_two[a]"><skipped message="optional device"/></testcase>'
        "</testsuite>",
        encoding="utf-8",
    )
    ids, skips = collector.test_cases(xml)
    assert ids == ["tests.host::test_one", "tests.host::test_two[a]"]
    assert skips == [collector.v.digest(collector.v.canonical(["tests.host::test_two[a]", "optional device"]))]
    collector.recorded_modules(["tests/host.py"], ids)
    with pytest.raises(ValueError, match="produced no cases"):
        collector.recorded_modules(["tests/missing.py"], ids)
    xml.write_text('<testsuite><testcase name="a"/><testcase name="a"/></testsuite>', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        collector.test_cases(xml)


def test_a_missing_or_stale_runner_receipt_is_refused(tmp_path, monkeypatch):
    receipts = tmp_path / "checkout" / "verification" / "P17"
    receipts.mkdir(parents=True)
    (receipts / "unit-114.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(collector, "run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="unique fresh"):
        collector.pytest_evidence(tmp_path / "checkout", "unit", tmp_path, io.BytesIO(), {}, {})


def test_quiet_command_still_records_the_action_and_exit(candidate, tmp_path):
    (candidate.root / "scripts" / "quality.py").write_text("pass\n", encoding="utf-8")
    out = tmp_path / "quiet"
    collector.execute(_args(candidate, out, "collect"))
    lines = (out / "run.log").read_bytes().splitlines()
    assert json.loads(lines[0])["argv"][-1] == "scripts/quality.py"
    assert json.loads(lines[-1]) == {"exit": 0}


def test_disagreement_between_captured_stdout_and_the_receipt_is_rejected(tmp_path):
    receipt = {"command": ["pytest"], "source_inputs_sha256": "1" * 64, "passed": 3}
    with (tmp_path / "run.log").open("xb") as log:
        log.write(json.dumps(receipt).encode() + b'\n{"exit":0}\n')
        collector.reported_run(log, receipt)
        with pytest.raises(ValueError, match="stdout and receipt differ"):
            collector.reported_run(log, {**receipt, "passed": 4})


def test_an_unapproved_signer_cannot_promote_a_lane(candidate, tmp_path):
    lane = tmp_path / "worker"
    collector.execute(_args(candidate, lane, "collect"))
    lanes = tmp_path / "signed"
    lanes.mkdir()
    collector.write_json(lanes / (candidate.gate + ".json"), _seal(lane, candidate.release))
    flags = _args(candidate, tmp_path / "aggregate", "aggregate", lanes_dir=lanes)
    with pytest.raises(ValueError):
        collector.execute(flags)
    assert not (flags.output / "request.pae").exists()
