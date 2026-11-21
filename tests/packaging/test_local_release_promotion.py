"""Offline release routing and promotion checks with synthetic GitHub objects and package bytes."""

import hashlib
import importlib
import io
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
TAG = "v8.0.1"
COMMIT, TREE, ANNOTATION, CONTROL = (letter * 40 for letter in "abcd")


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    """An identified draft, a pinned control revision and an isolated synthetic run."""
    monkeypatch.syspath_prepend(str(ROOT / ".github/scripts"))
    release = importlib.import_module("local_release")
    monkeypatch.chdir(tmp_path)
    for key, value in {
        "GITHUB_SHA": COMMIT,
        "GITHUB_REPOSITORY": "test/repository",
        "GITHUB_REPOSITORY_ID": "7",
        "GITHUB_RUN_ID": "13",
        "GITHUB_RUN_ATTEMPT": "2",
        "LOCAL_CI_CONTROL_SHA": CONTROL,
        "LOCAL_CI_POLICY_JSON": "{}",
    }.items():
        monkeypatch.setenv(key, value)
    wheel, sdist = release.expected_package_names(TAG[1:])
    payload = dict(tag=TAG, release_id=21, wheel_id=51, sdist_id=52, bundle_id=53)
    content = {wheel: b"test wheel", sdist: b"test sdist", release.BUNDLE: b"test envelope"}
    ids = release.draft_ids(payload, TAG)

    def asset(name, index):
        return dict(
            id=index, name=name, state="uploaded", size=len(content[name]), digest="sha256:" + digest(content[name])
        )

    draft = dict(
        id=21, tag_name=TAG, draft=True, prerelease=False, assets=[asset(key, val) for key, val in ids.items()]
    )
    run = dict(
        id=13,
        run_attempt=2,
        repository=dict(id=7),
        head_repository=dict(id=7),
        head_sha=COMMIT,
        path=".github/workflows/release.yml",
        event="repository_dispatch",
        status="completed",
        conclusion="success",
    )
    objects = {
        "git/ref/heads/main": dict(object=dict(sha=COMMIT)),
        "git/ref/tags/" + TAG: dict(ref="refs/tags/" + TAG, object=dict(type="tag", sha=ANNOTATION)),
        "git/tags/" + ANNOTATION: dict(sha=ANNOTATION, tag=TAG, object=dict(type="commit", sha=COMMIT)),
        "git/commits/" + COMMIT: dict(sha=COMMIT, tree=dict(sha=TREE)),
        "releases/21": draft,
        "actions/runs/13/attempts/2": run,
    }

    def api(endpoint, *args):
        path = endpoint.split("repos/test/repository/", 1)[1]
        if endpoint.startswith("https://uploads.github.com/"):
            name = path.split("?name=")[1]
            content[name] = Path(args[-1]).read_bytes()
            row = asset(name, 100 + len(content))
            draft["assets"].append(row)
            return row
        if args:
            assert path == "releases/21" and args == ("--method", "PATCH", "-F", "draft=false")
            draft.update(draft=False, immutable=draft.get("immutable", True))
        return objects[path]

    def verifier(argv):
        # This suite exercises the release handoff. Signature fixtures exercise the verifier separately.
        assert argv[1] == "-I" and Path(argv[2]).name == "verify_local_ci.py"
        for flag, expected in (
            ("--repo-id", "7"),
            ("--tag", TAG),
            ("--commit", COMMIT),
            ("--tree", TREE),
            ("--tag-object", ANNOTATION),
        ):
            assert argv[argv.index(flag) + 1] == expected
        policy = Path(argv[argv.index("--policy") + 1])
        bundle = Path(argv[argv.index("--bundle") + 1])
        assert set(path.name for path in Path(argv[argv.index("--dist-dir") + 1]).iterdir()) == {wheel, sdist}
        assert bundle.read_bytes() == b"test envelope"
        return json.dumps(
            dict(policy_id="test", policy_sha256=release.sha256(policy), bundle_sha256=release.sha256(bundle))
        ).encode()

    def download(argv, *, stdout, check):
        assert check and argv[-1] == "Accept: application/octet-stream"
        index = int(argv[2].rsplit("/", 1)[1])
        row = next(row for row in draft["assets"] if row["id"] == index)
        stdout.write(content[row["name"]])

    monkeypatch.setattr(release, "api", api)
    monkeypatch.setattr(release.subprocess, "run", download)
    monkeypatch.setattr(release.subprocess, "check_output", verifier)
    return release, dict(client_payload=payload, workflow_run=run.copy()), objects, content


@pytest.mark.parametrize("immutable", [True, False])
def test_signed_draft_handoff_keeps_release_and_package_bytes(candidate, immutable):
    release, event, objects, content = candidate
    objects["releases/21"]["immutable"] = immutable
    release.verify_draft(event, TAG)
    receipt = json.loads((Path("proof") / release.RECEIPT).read_text())
    release.promote(event, TAG, receipt)
    release.prepare_pypi(event, TAG, receipt)
    assert objects["releases/21"]["immutable"] is immutable
    packages = Path("release-staging/packages")
    expected = {name: content[name] for name in release.expected_package_names(TAG[1:])}
    assert {file.name: file.read_bytes() for file in packages.iterdir()} == expected
    assert (Path("release-staging/metadata") / release.BUNDLE).read_bytes() == content[release.BUNDLE]


@pytest.mark.parametrize(
    "change", ["foreign-tag", "lightweight", "moved-main", "wrong-id", "extra", "published", "changed-bytes"]
)
def test_draft_refuses_wrong_material(candidate, change):
    release, event, objects, content = candidate
    draft = objects["releases/21"]
    if change == "foreign-tag":
        draft["tag_name"] = "v1.0.0"
    elif change == "lightweight":
        objects["git/ref/tags/" + TAG]["object"]["type"] = "commit"
    elif change == "moved-main":
        objects["git/ref/heads/main"]["object"]["sha"] = "e" * 40
    elif change == "wrong-id":
        event["client_payload"]["wheel_id"] += 100
    elif change == "extra":
        draft["assets"].append(dict(draft["assets"][0], name="extra", id=90))
    elif change == "published":
        draft["draft"] = False
    else:
        content[release.BUNDLE] += b"x"
    with pytest.raises(ValueError):
        release.verify_draft(event, TAG)
    assert not (Path("proof") / release.RECEIPT).exists()


@pytest.mark.parametrize("change", ["id", "attempt", "failed", "other-repo", "forged-receipt"])
def test_changed_run_or_public_inventory_cannot_reach_pypi(candidate, change):
    release, event, objects, content = candidate
    release.verify_draft(event, TAG)
    receipt = json.loads((Path("proof") / release.RECEIPT).read_text())
    release.promote(event, TAG, receipt)
    run = objects["actions/runs/13/attempts/2"]
    if change == "id":
        objects["releases/21"]["assets"][0]["id"] += 300
    elif change == "attempt":
        run["run_attempt"] = 3
    elif change == "failed":
        run["conclusion"] = "failure"
    elif change == "other-repo":
        run["head_repository"] = {"id": 8}
    else:
        content[release.RECEIPT] += b" "
        row = next(row for row in objects["releases/21"]["assets"] if row["name"] == release.RECEIPT)
        row.update(size=len(content[release.RECEIPT]), digest="sha256:" + digest(content[release.RECEIPT]))
    with pytest.raises(ValueError):
        release.prepare_pypi(event, TAG, receipt)
    assert not Path("release-staging/packages").exists()


def test_cloud_route_and_oidc_permissions_are_separate():
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())["jobs"]
    pypi = yaml.safe_load((ROOT / ".github/workflows/pypi.yml").read_text())["jobs"]
    assert "vars.LOCAL_CI_RELEASE_TAG == ''" in release["verify_release_policy"]["if"]
    assert "!= vars.LOCAL_CI_RELEASE_TAG" in release["verify_release_policy"]["if"]
    assert release["build"]["needs"] == "verify_release_policy"
    assert release["publish"]["needs"] == "build"
    assert release["verify_local"]["permissions"] == {"contents": "read"}
    assert release["promote_local"]["needs"] == "verify_local"
    assert pypi["prepare_local"]["permissions"] == {"actions": "read", "contents": "read"}
    assert set(pypi["publish"]["needs"]) == {"prepare", "prepare_local"}
    assert pypi["publish"]["environment"] == "pypi"
    assert pypi["publish"]["permissions"]["id-token"] == "write"
    assert all("run" not in step for step in pypi["publish"]["steps"])


def test_pypi_reuses_only_identical_files_and_checks_actual_downloads(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / ".github/scripts"))
    files = importlib.import_module("local_pypi_files")
    expected = {name: tmp_path / name for name in files.expected_package_names(TAG[1:])}
    rows = []
    for name, path in expected.items():
        path.write_bytes(name.encode())
        rows.append(
            dict(
                filename=name,
                size=len(name),
                digests={"sha256": digest(name.encode())},
                url="https://files.pythonhosted.org/" + name,
                yanked=False,
            )
        )
    assert not files.check_files(expected, rows[:1], read_back=False)
    with pytest.raises(ValueError):
        files.check_files(expected, [dict(rows[0], size=0)], read_back=False)
    monkeypatch.setattr(files, "urlopen", lambda url, timeout: io.BytesIO(url.rsplit("/", 1)[-1].encode()))
    assert files.check_files(expected, rows, read_back=True)
    monkeypatch.setattr(files, "urlopen", lambda url, timeout: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="download digest mismatch"):
        files.check_files(expected, rows, read_back=True)


def test_existing_draft_metadata_must_match_the_verified_run(candidate):
    release, event, objects, content = candidate
    release.verify_draft(event, TAG)
    receipt = json.loads((Path("proof") / release.RECEIPT).read_text())
    prior = (Path("proof") / release.SUMS).read_bytes()
    row = dict(name=release.SUMS, id=68, state="uploaded", digest="sha256:" + digest(b"other"), size=len(prior))
    draft = objects["releases/21"]
    draft["assets"].append(row)
    with pytest.raises(ValueError, match="conflicting draft metadata"):
        release.promote(event, TAG, receipt)
    assert draft["draft"]
    content[release.SUMS] = prior
    row["digest"] = "sha256:" + digest(prior)
    release.promote(event, TAG, receipt)
    assert release.inventory(draft, 21, TAG, draft=False)[release.SUMS]["id"] == 68
