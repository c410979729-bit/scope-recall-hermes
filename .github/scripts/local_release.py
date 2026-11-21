"""Verify signed local distributions and promote an identified draft without rebuilding it.

Run this file and its sibling verifier from the externally pinned control commit.
Only the promotion command mutates GitHub. All other commands read existing objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Isolated Python omits the script directory. Only this approved checkout contains helpers.
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from release_provenance import build_provenance, save_receipt, verify_provenance  # noqa: E402
from stage_release_assets import expected_package_names, stage_release_assets  # noqa: E402

BUNDLE = "LOCAL-CI.dsse.json"
RECEIPT = "RELEASE-PROVENANCE.json"
SUMS = "SHA256SUMS"


def require(condition: bool, message: str) -> None:
    """Refuse a release whose GitHub objects or verified bytes do not match."""
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def api(path: str, *args: str) -> dict:
    """Use only paths built from platform repository identity and validated IDs."""
    result = subprocess.check_output(["gh", "api", path, *args])
    return json.loads(result)


def repo_path(path: str) -> str:
    return f"repos/{os.environ['GITHUB_REPOSITORY']}/{path}"


def object_id(value: object) -> int:
    """Release and asset IDs are positive JSON integers or their plain decimal strings."""
    require(type(value) in (int, str) and re.fullmatch(r"[1-9][0-9]{0,19}", str(value)) is not None, "invalid ID")
    return int(str(value))


def git_source(tag: str, commit: str, *, at_main: bool) -> dict:
    """Require the actual annotated tag, its direct commit target, and that commit's tree."""
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag) is not None, "invalid stable tag")
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid run commit")
    ref = api(repo_path(f"git/ref/tags/{tag}"))
    require(ref["ref"] == f"refs/tags/{tag}" and ref["object"]["type"] == "tag", "annotated tag required")
    tag_object = ref["object"]["sha"]
    require(re.fullmatch(r"[0-9a-f]{40}", tag_object) is not None, "invalid tag object")
    annotation = api(repo_path(f"git/tags/{tag_object}"))
    require(annotation["sha"] == tag_object and annotation["tag"] == tag, "tag object mismatch")
    require(annotation["object"]["type"] == "commit" and annotation["object"]["sha"] == commit, "run/tag mismatch")
    if at_main:
        require(api(repo_path("git/ref/heads/main"))["object"]["sha"] == commit, "main moved beyond the candidate")
    source = api(repo_path(f"git/commits/{commit}"))
    tree = source["tree"]["sha"]
    require(source["sha"] == commit and re.fullmatch(r"[0-9a-f]{40}", tree) is not None, "invalid source tree")
    return {"commit": commit, "tree": tree, "tag_object": tag_object}


def inventory(release: dict, release_id: int, tag: str, *, draft: bool) -> dict:
    """Require each listed upload to belong to this release and have a unique name and ID."""
    require(release["id"] == release_id and release["tag_name"] == tag, "release identity mismatch")
    require(release["draft"] is draft and release["prerelease"] is False, "wrong publication state")
    rows = release["assets"]
    assets = {row["name"]: row for row in rows}
    require(len(rows) == len(assets) == len({object_id(row["id"]) for row in rows}), "duplicate asset")
    require(all(row["state"] == "uploaded" for row in rows), "incomplete asset")
    return assets


def draft_ids(payload: dict, tag: str) -> dict:
    """Resolve the ready event's three explicitly named asset IDs before reading their bytes."""
    require(payload["tag"] == tag, "no local approval for this tag")
    wheel, sdist = expected_package_names(tag[1:])
    return {
        wheel: object_id(payload["wheel_id"]),
        sdist: object_id(payload["sdist_id"]),
        BUNDLE: object_id(payload["bundle_id"]),
    }


def exact_assets(rows: dict, expected: dict) -> None:
    require({name: row["id"] for name, row in rows.items()} == expected, "release asset inventory changed")


def verified_assets(rows: dict, ids: dict, digests: dict) -> None:
    """Both sides of draft promotion must contain the exact verified assets, including derived metadata."""
    exact_assets(rows, ids)
    require({name: row.get("digest") for name, row in rows.items()} == digests, "release asset bytes changed")


def draft_input_assets(rows: dict, ids: dict) -> None:
    """On retry, only our two derived files may already accompany the identified inputs."""
    exact_assets({name: row for name, row in rows.items() if name not in (SUMS, RECEIPT)}, ids)


def existing_metadata(rows: dict, proof: Path) -> None:
    """A completed upload is reusable only when its exact bytes belong to this proof."""
    for name in (SUMS, RECEIPT):
        if name in rows:
            require(rows[name].get("digest") == "sha256:" + sha256(proof / name), "conflicting draft metadata")


def download(row: dict, destination: Path) -> None:
    """Fetch a specific asset. Independently compare its stored size and GitHub digest."""
    path = repo_path(f"releases/assets/{object_id(row['id'])}")
    with destination.open("xb") as stream:
        subprocess.run(["gh", "api", path, "-H", "Accept: application/octet-stream"], stdout=stream, check=True)
    require(destination.stat().st_size == row["size"], "asset length mismatch")
    require("sha256:" + sha256(destination) == row.get("digest"), "asset digest mismatch")


def verify_signature(policy: Path, bundle: Path, packages: Path, tag: str, source: dict) -> dict:
    """Pass only locally approved policy bytes and the two downloaded distributions to DSSE."""
    policy_bytes = os.environ["LOCAL_CI_POLICY_JSON"].encode()
    require(0 < len(policy_bytes) <= 48 * 1024, "local approval is missing or too large")
    policy.write_bytes(policy_bytes)
    result = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                "-I",
                str(SCRIPTS / "verify_local_ci.py"),
                "--policy",
                str(policy),
                "--bundle",
                str(bundle),
                "--dist-dir",
                str(packages),
                "--repo-id",
                os.environ["GITHUB_REPOSITORY_ID"],
                "--tag",
                tag,
                "--commit",
                source["commit"],
                "--tree",
                source["tree"],
                "--tag-object",
                source["tag_object"],
            ]
        )
    )
    require(
        result["bundle_sha256"] == sha256(bundle) and result["policy_sha256"] == sha256(policy),
        "verifier digest mismatch",
    )
    return {key: result[key] for key in ("policy_id", "policy_sha256", "bundle_sha256")}


def verify_draft(event: dict, tag: str) -> None:
    """Produce run-bound evidence from a ready draft without executing any candidate files."""
    payload = event["client_payload"]
    ids = draft_ids(payload, tag)
    release_id = object_id(payload["release_id"])
    source = git_source(tag, os.environ["GITHUB_SHA"], at_main=True)
    rows = inventory(api(repo_path(f"releases/{release_id}")), release_id, tag, draft=True)
    draft_input_assets(rows, ids)
    packages, proof = Path("packages"), Path("proof")
    packages.mkdir()
    proof.mkdir()
    bundle = Path(BUNDLE)
    for name in ids:
        download(rows[name], bundle if name == BUNDLE else packages / name)
    decision = verify_signature(Path("policy.json"), bundle, packages, tag, source)
    local_ci = dict(
        decision,
        control_sha=os.environ["LOCAL_CI_CONTROL_SHA"],
        tag_object=source["tag_object"],
        release_id=release_id,
        asset_ids=ids,
    )
    receipt = build_provenance(
        repository=os.environ["GITHUB_REPOSITORY"],
        source_sha=source["commit"],
        source_tree=source["tree"],
        release_tag=tag,
        workflow_run_id=os.environ["GITHUB_RUN_ID"],
        workflow_run_attempt=os.environ["GITHUB_RUN_ATTEMPT"],
        packages_dir=packages,
    )
    receipt["local_ci"] = local_ci
    save_receipt(proof / RECEIPT, receipt)
    (proof / SUMS).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in receipt["artifacts"].items()), encoding="utf-8"
    )
    existing_metadata(rows, proof)


def verify_receipt_identity(receipt: dict, tag: str, run_id: int, attempt: int, source: dict) -> None:
    """The immutable run artifact identifies the source and the specific approval used in that run."""
    require(receipt["format"] == "scope-recall-release-provenance-v1", "wrong provenance format")
    require(
        receipt["release_tag"] == tag and receipt["repository"] == os.environ["GITHUB_REPOSITORY"],
        "wrong repository/tag",
    )
    require(receipt["source_sha"] == source["commit"] and receipt["source_tree"] == source["tree"], "source changed")
    require(
        receipt["workflow"] == dict(path=".github/workflows/release.yml", run_id=str(run_id), run_attempt=str(attempt)),
        "wrong release run or attempt",
    )
    local = receipt["local_ci"]
    require(
        local["tag_object"] == source["tag_object"] and local["control_sha"] == os.environ["LOCAL_CI_CONTROL_SHA"],
        "approval identity changed",
    )
    require(
        local["policy_sha256"] == hashlib.sha256(os.environ["LOCAL_CI_POLICY_JSON"].encode()).hexdigest(),
        "approval policy changed",
    )


def promote(event: dict, tag: str, receipt: dict) -> None:
    """Add only two verified metadata files, then promote the same draft and check its immutable state."""
    source = git_source(tag, os.environ["GITHUB_SHA"], at_main=True)
    verify_receipt_identity(
        receipt, tag, int(os.environ["GITHUB_RUN_ID"]), int(os.environ["GITHUB_RUN_ATTEMPT"]), source
    )
    local, payload = receipt["local_ci"], event["client_payload"]
    ids = draft_ids(payload, tag)
    release_id = object_id(payload["release_id"])
    require(ids == local["asset_ids"] and release_id == local["release_id"], "ready event changed")
    endpoint = repo_path(f"releases/{release_id}")
    digests = {**receipt["artifacts"], BUNDLE: local["bundle_sha256"]}
    rows = inventory(api(endpoint), release_id, tag, draft=True)
    draft_input_assets(rows, ids)
    existing_metadata(rows, Path("proof"))
    for name in ids:
        require(rows[name].get("digest") == "sha256:" + digests[name], "verified asset bytes changed")
    for name in (SUMS, RECEIPT):
        if name in rows:
            ids[name] = object_id(rows[name]["id"])
            continue
        file = Path("proof") / name
        upload = api(
            f"https://uploads.github.com/{endpoint}/assets?name={name}",
            "--method",
            "POST",
            "-H",
            "Content-Type: application/octet-stream",
            "--input",
            str(file),
        )
        require(upload["name"] == name and upload["digest"] == "sha256:" + sha256(file), "metadata upload mismatch")
        ids[name] = object_id(upload["id"])
    require(git_source(tag, source["commit"], at_main=True) == source, "tag changed during promotion")
    hashes = {name: "sha256:" + value for name, value in digests.items()}
    hashes.update({name: "sha256:" + sha256(Path("proof") / name) for name in (SUMS, RECEIPT)})
    verified_assets(inventory(api(endpoint), release_id, tag, draft=True), ids, hashes)
    published = api(endpoint, "--method", "PATCH", "-F", "draft=false")
    verified_assets(inventory(published, release_id, tag, draft=False), ids, hashes)


def successful_origin(run: dict, repo_id: int) -> None:
    """Only the approved repository's successful release workflow may hand off to OIDC publication."""
    require(
        run["repository"]["id"] == repo_id and run["head_repository"]["id"] == repo_id, "foreign upstream repository"
    )
    require(
        run["path"] == ".github/workflows/release.yml" and run["event"] == "repository_dispatch",
        "wrong upstream workflow",
    )
    require(run["status"] == "completed" and run["conclusion"] == "success", "upstream release run did not succeed")


def prepare_pypi(event: dict, tag: str, receipt: dict) -> None:
    """Read back public bytes by their recorded asset IDs and independently verify the signed evidence."""
    origin = event["workflow_run"]
    run_id, attempt = object_id(origin["id"]), object_id(origin["run_attempt"])
    run = api(repo_path(f"actions/runs/{run_id}/attempts/{attempt}"))
    successful_origin(run, int(os.environ["GITHUB_REPOSITORY_ID"]))
    require(
        run["id"] == run_id and run["run_attempt"] == attempt and run["head_sha"] == origin["head_sha"],
        "upstream run mismatch",
    )
    source = git_source(tag, run["head_sha"], at_main=False)
    verify_receipt_identity(receipt, tag, run_id, attempt, source)
    local = receipt["local_ci"]
    release_id = object_id(local["release_id"])
    rows = inventory(api(repo_path(f"releases/{release_id}")), release_id, tag, draft=False)
    wheel, sdist = expected_package_names(tag[1:])
    require(set(rows) == {wheel, sdist, BUNDLE, RECEIPT, SUMS}, "unexpected public inventory")
    exact_assets({name: rows[name] for name in (wheel, sdist, BUNDLE)}, local["asset_ids"])
    source_dir, staged = Path("public"), Path("release-staging")
    source_dir.mkdir()
    for name in rows:
        download(rows[name], source_dir / name)
    for name in (RECEIPT, SUMS):
        require((source_dir / name).read_bytes() == (Path("proof") / name).read_bytes(), "run/public metadata mismatch")
    packages = staged / "packages"
    stage_release_assets(
        source_dir, packages_dir=packages, metadata_dir=staged / "metadata", version=tag[1:], local_ci=True
    )
    decision = verify_signature(Path("policy.json"), source_dir / BUNDLE, packages, tag, source)
    require(all(local[key] == value for key, value in decision.items()), "local approval changed")
    verify_provenance(
        source_dir / RECEIPT,
        expected_repository=os.environ["GITHUB_REPOSITORY"],
        expected_source_sha=source["commit"],
        expected_source_tree=source["tree"],
        expected_release_tag=tag,
        expected_workflow_run_id=str(run_id),
        workflow_run_status=run["status"],
        workflow_run_conclusion=run["conclusion"],
        packages_dir=packages,
    )


def main() -> None:
    """Dispatch a fixed workflow operation after verifying the independent control revision."""
    control = os.environ["LOCAL_CI_CONTROL_SHA"]
    actual = subprocess.check_output(["git", "-C", str(SCRIPTS), "rev-parse", "HEAD"], text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", control) is not None and control == actual, "unapproved control checkout")
    tag = os.environ["LOCAL_CI_RELEASE_TAG"]
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag) is not None, "invalid approved stable tag")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_bytes())
    if sys.argv[1] == "verify":
        verify_draft(event, tag)
    else:
        action = {"promote": promote, "pypi": prepare_pypi}[sys.argv[1]]
        action(event, tag, json.loads((Path("proof") / RECEIPT).read_bytes()))


if __name__ == "__main__":
    main()
