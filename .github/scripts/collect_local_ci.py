#!/usr/bin/env python3
"""Measure a release lane on a credential-free worker and prepare its DSSE signing request.

Run this controller from a separately approved read-only copy with the lane's locked Python and installed wheel.
`inspect` reports source identities, `observe` runs a gate for qualification, and `collect` additionally requires
an independently approved policy. Lane names take the form windows-py3.12-unit or linux-py3.12-hermes-catalog.
Signers run in a separate security domain. They accept PAE bytes only from their authenticated controller job,
after checking its trusted execution records. Files emitted by a worker alone confer no signing authority.
`aggregate` verifies those signed lanes against the actual annotated tag and prepares the release request.
`envelope` checks a detached 64-byte signature, its approved public key and the original signed payload.
`bundle` verifies the externally signed aggregate together with its lanes before emitting publishable bytes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

# Isolated Python omits the script directory; load the verifier from the operator's approved controller copy.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_local_ci as v  # noqa: E402

TIERS = frozenset(
    (
        "unit contract storage capture claims deletion episodes retrieval packaging migration host native integration"
    ).split()
)
LOCKS = ("constraints/release-hashed.txt", "uv.lock")
REPO_SLUG = "c410979729-bit/scope-recall-hermes"
CATALOG = (
    ".github/scripts/hermes_catalog_install.sh",
    ".github/scripts/hermes_remote_install.py",
    "scripts/hermes_surface.py",
)
PLATFORMS = {"win32": "windows", "linux": "linux", "darwin": "macos"}


def git(checkout: Path, *args: str) -> bytes:
    """Read local Git state using an explicit directory exception for the isolated worker."""
    return subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-c",
            f"safe.directory={checkout}",
            "-C",
            str(checkout),
            *args,
        ],
        stdin=subprocess.DEVNULL,
    )


def file_digest(path: Path) -> str:
    with v.regular(path) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(checkout: Path) -> dict:
    """Bind a clean checkout, normalized Git blobs, dependencies and independently selected control programs."""
    v.require(
        Path(os.fsdecode(git(checkout, "rev-parse", "--show-toplevel")).strip()).resolve() == checkout,
        "checkout root required",
    )
    v.require(not git(checkout, "status", "--porcelain", "--untracked-files=all"), "dirty candidate checkout")
    commit, tree = git(checkout, "rev-parse", "HEAD", "HEAD^{tree}").decode("ascii").splitlines()
    tracked = {}
    for row in git(checkout, "ls-tree", "-rz", commit).split(b"\0"):
        if not row:
            continue
        header, path = row.split(b"\t", 1)
        mode, _, blob = header.split()
        v.require(mode in (b"100644", b"100755"), "non-regular source file")
        name = path.decode("utf-8")
        v.require(not any(char in name for char in "\r\n"), "unsupported source path")
        tracked[name] = blob
    # A status check alone can trust the index's stat or assume-unchanged flags. Hash the actual working files.
    rows = subprocess.check_output(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            f"safe.directory={checkout}",
            "-c",
            "core.autocrlf=input",
            "hash-object",
            "--stdin-paths",
        ],
        input="".join(name + "\n" for name in tracked).encode(),
    ).splitlines()
    v.require(rows == list(tracked.values()), "working files differ from the approved tree")
    entries = {
        p: "git-blob:" + blob.decode("ascii")
        for p, blob in tracked.items()
        if not p.startswith(("docs/", "verification/"))
    }
    # Match the runner's existing manifest encoding, which predates the DSSE canonicalization.
    source_sha = v.digest(json.dumps(entries, sort_keys=True).encode())
    locks = {name: v.digest(git(checkout, "show", f"{commit}:{name}")) for name in LOCKS}
    controls = {
        "collector": file_digest(Path(__file__)),
        "verifier": file_digest(Path(v.__file__)),
        "harness": v.digest(git(checkout, "show", f"{commit}:scripts/check.py")),
    }
    return {
        "commit": commit,
        "tree": tree,
        "source_sha256": source_sha,
        "lock_sha256": v.digest(v.canonical(locks)),
        "rules_sha256": v.digest(v.canonical(controls)),
    }


def installed_wheel(dist_dir: Path, version: str) -> None:
    """Verify package data and modules used by the worker against the released wheel's actual bytes."""
    wheel = dist_dir / f"hermes_scope_recall-{version}-py3-none-any.whl"
    installed = importlib.metadata.distribution("hermes-scope-recall")
    v.require(installed.version == version, "wrong installed Scope Recall version")
    root = Path(installed.locate_file("scope_recall"))
    files = {
        p.relative_to(root.parent).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    with v.regular(wheel) as stream, zipfile.ZipFile(stream) as archive:
        members = [name for name in archive.namelist() if name.startswith("scope_recall/") and not name.endswith("/")]
        v.require(
            files and len(members) == len(files) and set(members) == files, "installed package differs from wheel"
        )
        for name in members:
            with archive.open(name) as packed:
                v.require(
                    file_digest(root.parent / name) == hashlib.file_digest(packed, "sha256").hexdigest(),
                    "installed file differs from wheel",
                )


def test_cases(junit: Path) -> tuple[list[str], list[str]]:
    """Return stable xunit2 case IDs and digests of each allowed-to-be-reviewed skip reason."""
    with v.regular(junit) as stream:
        cases = ET.parse(stream).findall(".//testcase")
    identities, skips = [], []
    for case in cases:
        name = case.get("name")
        v.require(bool(name) and case.find("failure") is None and case.find("error") is None, "failed or unnamed case")
        identity = f"{case.get('classname', '')}::{name}"
        identities.append(identity)
        skip = case.find("skipped")
        if skip is not None:
            reason = skip.get("message")
            v.require(bool(reason), "skip has no reviewable reason")
            skips.append(v.digest(v.canonical([identity, reason])))
    v.require(bool(identities) and len(identities) == len(set(identities)), "empty or duplicate case inventory")
    return sorted(identities), sorted(skips)


def selection_digest(names: list[str]) -> str:
    v.require(
        bool(names) and all(isinstance(n, str) and n for n in names) and len(names) == len(set(names)),
        "empty, invalid or duplicate selection",
    )
    return v.digest(v.canonical(sorted(names)))


def write_json(path: Path, value: object) -> None:
    with path.open("xb") as output:
        output.write(v.canonical(value))


def recorded_modules(selected: list[str], identities: list[str]) -> None:
    """A reported selection is evidence only when its file or directory has cases in JUnit."""
    classes = {identity.split("::", 1)[0] for identity in identities}
    # Pytest emits module-level collection skips with an empty class and the module path as the name.
    modules = {identity[2:] for identity in identities if identity.startswith("::")}
    for selection in selected:
        path = selection.split("::", 1)[0].replace("\\", "/").rstrip("/")
        prefix = path.removesuffix(".py").replace("/", ".")
        present = any(name == prefix or name.startswith(prefix + ".") for name in classes)
        present |= any(name == path or name.startswith(path + "/") for name in modules)
        v.require(present, "a selected test module produced no cases")


def environment(home: Path, checkout: Path) -> dict[str, str]:
    """Expose only process essentials, an isolated home and task-owned temporary storage to candidate code."""
    env = {
        k: value
        for k, value in os.environ.items()
        if k.upper()
        in {"SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "PATH"}
    }
    for key in (
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "HERMES_HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
    ):
        path = home / key.lower()
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    env.update(
        PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        SCOPE_RECALL_TEST_BOUNDARY_PARENT=str(home),
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="safe.directory",
        GIT_CONFIG_VALUE_0=str(checkout),
    )
    return env


def run(argv: list[str], checkout: Path, env: dict, output, timeout: int = 7500) -> None:
    output.write(v.canonical({"argv": argv}) + b"\n")
    output.flush()
    result = subprocess.run(
        argv,
        cwd=checkout,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output.write(v.canonical({"exit": result.returncode}) + b"\n")
    output.flush()
    result.check_returncode()


def reported_run(log, expected: dict) -> None:
    """Pair the isolated receipt file with the result printed by this exact child."""
    log.flush()
    with v.regular(Path(log.name)) as capture:
        capture.seek(max(0, os.fstat(capture.fileno()).st_size - v.MAX_JSON - 8192))
        tail = capture.read().splitlines()
    for line in reversed(tail):
        if line.startswith(b"{") and b'"source_inputs_sha256":' in line:
            v.require(v.decode(line) == expected, "runner stdout and receipt differ")
            return
    raise ValueError("runner did not emit its receipt")


def pytest_evidence(checkout: Path, gate: str, files: Path, log, env: dict, state: dict) -> tuple:
    """Only accept the fresh receipt from this invocation, plus its raw JUnit file."""
    receipts = checkout / "verification" / "P17"
    pattern = re.compile(re.escape(gate) + r"-[0-9]+\.json")
    before = {p.name for p in receipts.glob("*.json")}
    run([sys.executable, "-B", "scripts/check.py", "--tier", gate, "--task", "P17", "--keep-junit"], checkout, env, log)
    added = [p for p in receipts.glob("*.json") if pattern.fullmatch(p.name) and p.name not in before]
    v.require(len(added) == 1, "no unique fresh pytest receipt")
    receipt, _ = v.document(added[0])
    reported_run(log, receipt)
    expected = f"verification/P17/{added[0].stem}"
    v.require(
        receipt["log"] == expected + ".log" and receipt["source_manifest"] == expected + "-inputs.json",
        "unexpected local receipt paths",
    )
    v.require(
        receipt["source_base"] == state["commit"] and receipt["source_inputs_sha256"] == state["source_sha256"],
        "pytest measured a different source",
    )
    exits = {field: receipt[field + "_exit_code"] for field in ("pytest", "wrapper", "cleanup")}
    v.require(
        all(type(code) is int and code == 0 for code in [*exits.values(), receipt["overall_exit_code"]])
        and not receipt["missing_gates"]
        and receipt["model_calls"] is False,
        "incomplete pytest/runner/cleanup",
    )
    raw_xml = receipts / (added[0].stem + "-junit.xml")
    ids, skips = test_cases(raw_xml)
    recorded_modules(receipt["selected"], ids)
    manifest, _ = v.document(checkout / receipt["source_manifest"])
    v.require(
        v.digest(json.dumps(manifest, sort_keys=True).encode()) == state["source_sha256"], "incorrect source manifest"
    )
    passed = len(ids) - len(skips)
    v.require(
        (receipt["passed"], receipt["skipped"], receipt["failed"]) == (passed, len(skips), 0), "JUnit count mismatch"
    )
    with v.regular(raw_xml) as source, (files / "junit.xml").open("xb") as dest:
        shutil.copyfileobj(source, dest)
    write_json(files / "runner.json", receipt)
    return receipt["selected"], ids, skips


PACKAGE_STEPS = (
    "build-wheel-from-sdist",
    "compare-wheel-members",
    "install-and-load-wheel",
    "install-and-load-rebuilt-wheel",
)
PACKAGE_PROBE = """
import importlib, importlib.metadata as metadata, sys
from pathlib import Path
from packaging.requirements import Requirement
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
distributions = [d for d in metadata.distributions(path=[str(target)]) if d.metadata["Name"] == "hermes-scope-recall"]
if len(distributions) != 1 or distributions[0].version != sys.argv[2]:
    raise RuntimeError("fresh install version mismatch")
package = distributions[0]
for raw in package.requires or ():
    requirement = Requirement(raw)
    if not requirement.marker or any(requirement.marker.evaluate({"extra": e}) for e in ("", "codex")):
        if metadata.version(requirement.name) not in requirement.specifier:
            raise RuntimeError("declared runtime requirement not satisfied")
for module in ("scope_recall", "scope_recall.adapters.clients.antigravity", "scope_recall.adapters.clients.remote_mcp"):
    if not Path(importlib.import_module(module).__file__).resolve().is_relative_to(target):
        raise RuntimeError("import did not come from the fresh install")
entry, = package.entry_points.select(group="console_scripts", name="scope-recall")
sys.argv[:] = ["scope-recall", "--help"]
raise SystemExit(entry.load()())
"""


def wheel_members(path: Path) -> dict[str, str]:
    """Hash decompressed members, including RECORD, independently of ZIP timestamps."""
    result = {}
    with v.regular(path) as stream, zipfile.ZipFile(stream) as archive:
        for member in archive.infolist():
            if not member.is_dir():
                v.require(member.filename not in result, "duplicate wheel member")
                with archive.open(member) as content:
                    result[member.filename] = hashlib.file_digest(content, "sha256").hexdigest()
    return result


def package_smoke(dist_dir: Path, home: Path, env: dict, log, version: str) -> tuple:
    """Rebuild and install the artifacts offline with the worker's locked build tools and dependencies."""
    uv = shutil.which("uv", path=env["PATH"])
    v.require(bool(uv), "the packaging worker needs uv")
    original = next(dist_dir.glob("*.whl"))
    sdist = next(dist_dir.glob("*.tar.gz"))
    rebuilt = home / "rebuilt" / original.name
    env.update(UV_OFFLINE="1", PIP_NO_INDEX="1", UV_CACHE_DIR=str(home / "uv-cache"))
    run(
        [
            uv,
            "build",
            "--offline",
            "--no-config",
            "--no-index",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--wheel",
            "--out-dir",
            str(rebuilt.parent),
            str(sdist),
        ],
        home,
        env,
        log,
    )
    v.require(wheel_members(original) == wheel_members(rebuilt), "wheel from sdist has different contents")
    for name, wheel in (("original-install", original), ("rebuilt-install", rebuilt)):
        target = home / name
        run(
            [
                uv,
                "pip",
                "install",
                "--offline",
                "--no-config",
                "--no-index",
                "--no-deps",
                "--python",
                sys.executable,
                "--target",
                str(target),
                str(wheel),
            ],
            home,
            env,
            log,
        )
        run([sys.executable, "-I", "-B", "-c", PACKAGE_PROBE, str(target), version], home, env, log)
    return list(PACKAGE_STEPS), list(PACKAGE_STEPS), []


def command_evidence(checkout: Path, gate: str, dist_dir: Path, home: Path, env: dict, log, state: dict) -> tuple:
    """Use commands and per-step case IDs fixed by the controller for this gate."""
    if gate == "quality":
        run([sys.executable, "-B", "scripts/quality.py"], checkout, env, log)
        return ["scripts/quality.py"], ["scripts/quality.py"], []
    if gate == "package-smoke":
        return package_smoke(dist_dir, home, env, log, state["version"])
    v.require(gate == "hermes-catalog" and sys.platform in ("linux", "darwin"), "unsupported command gate")
    # The installer asks Git for the release commit. Resolve that specific repository to the frozen local clone.
    url = "https://github.com/" + REPO_SLUG
    wheel = next(dist_dir.glob("*.whl"))
    constraint = home / "tested-wheel.txt"
    constraint.write_text(f"hermes-scope-recall @ {wheel.resolve().as_uri()}\n", encoding="utf-8")
    env.update(
        REPOSITORY=REPO_SLUG,
        COMMIT=state["commit"],
        WHEELS=str(dist_dir),
        UV_CONSTRAINT=str(constraint),
        PIP_CONSTRAINT=str(constraint),
        RUNNER_TEMP=str(home),
        GIT_CONFIG_COUNT="3",
        GIT_CONFIG_KEY_1="url." + checkout.as_uri() + ".insteadOf",
        GIT_CONFIG_VALUE_1=url + ".git",
        GIT_CONFIG_KEY_2="url." + checkout.as_uri() + ".insteadOf",
        GIT_CONFIG_VALUE_2=url,
    )
    # Use the same installation home layout as the catalog job, including the later independent inventory check.
    env.pop("HERMES_HOME")
    run(["bash", CATALOG[0]], checkout, env, log, 2400)
    run([sys.executable, "-B", CATALOG[1]], checkout, env, log, 2400)
    source = home / "hermes-catalog-install" / "home" / ".hermes" / "hermes-agent"
    run(
        [sys.executable, "-B", CATALOG[2], "diff", "fixtures/hermes/surface-0.23.0.json", str(source)],
        checkout,
        env,
        log,
    )
    return list(CATALOG), list(CATALOG), []


def observed(checkout: Path, dist_dir: Path, output: Path, scratch: Path, source: dict, lane: str) -> dict:
    """Launch the approved gate and preserve unsigned measurements after all child environments exit."""
    platform_name = PLATFORMS[sys.platform]
    prefix = f"{platform_name}-py{sys.version_info.major}.{sys.version_info.minor}-"
    v.require(lane.startswith(prefix), "lane belongs to a different OS or interpreter")
    gate = lane[len(prefix) :]
    v.require(gate in TIERS or gate in {"quality", "hermes-catalog", "package-smoke"}, "unregistered local CI gate")
    subject = v.subjects(dist_dir, source["version"])
    installed_wheel(dist_dir, source["version"])
    kind = "pytest" if gate in TIERS else "command"
    log_path = output / "run.log"
    with tempfile.TemporaryDirectory(prefix="TEST-release-", dir=scratch) as directory, log_path.open("xb") as log:
        home = Path(directory)
        env = environment(home, checkout)
        if kind == "pytest":
            selected, cases, skips = pytest_evidence(checkout, gate, output, log, env, source)
        else:
            selected, cases, skips = command_evidence(checkout, gate, dist_dir, home, env, log, source)
    v.require(all(source[k] == val for k, val in snapshot(checkout).items()), "source changed during the run")
    v.require(subject == v.subjects(dist_dir, source["version"]), "distributions changed during the run")
    installed_wheel(dist_dir, source["version"])
    write_json(output / "selection.json", {"selected": sorted(selected), "cases": cases, "skips": skips})
    passed = len(cases) - len(skips)
    v.require(passed > 0, "no cases passed")
    return {
        "subject": subject,
        "result": {
            "source": source,
            "lane": lane,
            "platform": platform_name,
            "python_version": platform.python_version(),
            "selection_sha256": selection_digest(selected),
            "cases_sha256": selection_digest(cases),
            "passed": passed,
            "skips": skips,
            "exit_codes": {"runner": 0, kind: 0, "cleanup": 0},
            "log_sha256": file_digest(log_path),
        },
    }


def request(directory: Path, role: str, subject: list, predicate: dict) -> None:
    """Export public statement and exact signing input, without material capable of signing it."""
    statement = {
        "_type": v.STATEMENT_TYPE,
        "predicateType": v.PREDICATE + role + "/v1",
        "subject": subject,
        "predicate": predicate,
    }
    payload = v.canonical(statement)
    with (directory / "statement.json").open("xb") as output:
        output.write(payload)
    with (directory / "request.pae").open("xb") as output:
        output.write(v.pae(payload))


def outside(checkout: Path, path: Path) -> None:
    v.require(
        not path.resolve().is_relative_to(checkout), "policy, receipts and artifacts must be outside the candidate"
    )


def release_source(checkout: Path, flags) -> dict:
    return {"repo_id": flags.repo_id, "tag": flags.tag, "version": flags.tag.removeprefix("v"), **snapshot(checkout)}


def tag_object(checkout: Path, source: dict) -> str:
    ref = "refs/tags/" + source["tag"]
    v.require(git(checkout, "cat-file", "-t", ref).strip() == b"tag", "annotated release tag required")
    tag, commit = git(checkout, "rev-parse", ref, ref + "^{commit}").decode().splitlines()
    v.require(commit == source["commit"], "wrong tagged commit")
    header = git(checkout, "cat-file", "tag", tag).split(b"\n\n", 1)[0].splitlines()
    expected = [f"object {commit}".encode(), b"type commit", f"tag {source['tag']}".encode()]
    v.require(header[:3] == expected, "tag object does not directly name the approved tag and commit")
    return tag


def approved(flags, checkout: Path, source: dict) -> tuple[dict, str]:
    outside(checkout, flags.policy)
    policy, sha = v.document(flags.policy)
    v.validate_policy(policy, flags.repo_id, flags.tag, source["commit"], source["tree"])
    v.require(policy["source"] == source, "approved source/lock/rule digests differ from the measured snapshot")
    return policy, sha


def signing_inputs(flags, checkout: Path, source: dict, policy: dict, policy_sha: str, output: Path) -> None:
    tag = tag_object(checkout, source)
    outside(checkout, flags.lanes_dir)
    paths = {p.name: p for p in flags.lanes_dir.iterdir()}
    v.require(set(paths) == {name + ".json" for name in policy["lanes"]}, "unexpected or missing signed lane")
    envelopes = {name: v.document(paths[name + ".json"])[0] for name in policy["lanes"]}
    subject = v.subjects(flags.dist_dir, source["version"])
    linked = v.verify_lanes(envelopes, policy, policy_sha, subject)
    if flags.command == "aggregate":
        request(
            output,
            "release",
            subject,
            {"source": source, "policy_sha256": policy_sha, "tag_object": tag, "lanes": linked},
        )
        return
    outside(checkout, flags.release)
    bundle = {"schema": v.BUNDLE_SCHEMA, "release": v.document(flags.release)[0], "lanes": envelopes}
    v.verify_statements(bundle, policy, policy_sha, tag, subject)
    write_json(output / "LOCAL-CI.dsse.json", bundle)


def envelope(flags) -> None:
    """Wrap one previously issued raw Ed25519 signature after verifying the exact statement and its authorized key."""
    policy, policy_sha = v.document(flags.policy)
    source = policy["source"]
    v.validate_policy(policy, source["repo_id"], source["tag"], source["commit"], source["tree"])
    with v.regular(flags.statement) as stream:
        raw = stream.read(v.MAX_JSON + 1)
    statement = v.decode(raw)
    result = statement["predicate"]
    role = "release" if statement["predicateType"] == v.PREDICATE + "release/v1" else "lane"
    key = policy["release_key"] if role == "release" else policy["lanes"][result["lane"]]["key"]
    with v.regular(flags.signature) as stream:
        sig = stream.read(65)
    v.require(len(sig) == 64, "raw 64-byte Ed25519 signature required")
    wrapped = {
        "payloadType": v.PAYLOAD_TYPE,
        "payload": base64.b64encode(raw).decode("ascii"),
        "signatures": [{"keyid": v.digest(v.b64(key, 32)), "sig": base64.b64encode(sig).decode("ascii")}],
    }
    result = v.signed(wrapped, key, role, v.subjects(flags.dist_dir, source["version"]))
    v.require(
        result["source"] == source and result["policy_sha256"] == policy_sha, "statement source or policy mismatch"
    )
    if role == "lane":
        v.lane_result(result, result["lane"], policy["lanes"][result["lane"]])
    write_json(flags.output, wrapped)


def execute(flags) -> None:
    if flags.command == "envelope":
        envelope(flags)
        return
    checkout = flags.checkout.resolve(strict=True)
    v.require(
        v.matches(flags.repo_id, r"[1-9][0-9]*") and v.matches(flags.tag, r"v[0-9]+\.[0-9]+\.[0-9]+"),
        "invalid release identity",
    )
    if flags.command in {"collect", "aggregate", "bundle"}:
        outside(checkout, Path(__file__))
        outside(checkout, Path(v.__file__))
    source = release_source(checkout, flags)
    if flags.command == "inspect":
        print(json.dumps(source, sort_keys=True))
        return
    outside(checkout, flags.output)
    outside(checkout, flags.dist_dir)
    flags.dist_dir = flags.dist_dir.resolve(strict=True)
    output = flags.output.resolve()
    output.mkdir(mode=0o700, exist_ok=False)
    if flags.command == "observe":
        outside(checkout, flags.scratch)
        write_json(
            output / "observation.json", observed(checkout, flags.dist_dir, output, flags.scratch, source, flags.lane)
        )
        return
    policy, policy_sha = approved(flags, checkout, source)
    if flags.command == "collect":
        outside(checkout, flags.scratch)
        rule = policy["lanes"][flags.lane]
        measured = observed(checkout, flags.dist_dir, output, flags.scratch, source, flags.lane)
        result = {**measured["result"], "policy_sha256": policy_sha}
        v.lane_result(result, flags.lane, rule)
        request(output, "lane", measured["subject"], result)
        return
    signing_inputs(flags, checkout, source, policy, policy_sha, output)


def main() -> int:
    """Measure one fixed gate or assemble evidence for a separately authorized signer."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    wrap = subs.add_parser("envelope", help="verify and wrap a detached signature from the independent signer")
    for field in ("policy", "statement", "signature", "dist-dir", "output"):
        wrap.add_argument("--" + field, type=Path, required=True)
    for action in ("inspect", "observe", "collect", "aggregate", "bundle"):
        sub = subs.add_parser(action)
        sub.add_argument("--checkout", type=Path, required=True)
        sub.add_argument("--repo-id", required=True)
        sub.add_argument("--tag", required=True)
        if action != "inspect":
            sub.add_argument("--dist-dir", type=Path, required=True)
            sub.add_argument(
                "--output", type=Path, required=True, help="new controller-owned directory outside the checkout"
            )
        if action in ("observe", "collect"):
            sub.add_argument("--lane", required=True)
            sub.add_argument("--scratch", type=Path, required=True, help="existing private worker temp root")
        if action in ("collect", "aggregate", "bundle"):
            sub.add_argument("--policy", type=Path, required=True)
        if action in ("aggregate", "bundle"):
            sub.add_argument(
                "--lanes-dir", type=Path, required=True, help="directory with one signed <lane>.json per policy lane"
            )
        if action == "bundle":
            sub.add_argument("--release", type=Path, required=True, help="externally signed release envelope")
    flags = parser.parse_args()
    try:
        execute(flags)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, ET.ParseError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Local CI {flags.command} failed ({type(error).__name__}). Inspect local run evidence.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
