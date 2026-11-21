"""Check the changelog sections that the release workflow publishes."""

from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "extract_changelog_release_notes.py"
CHANGELOG = ROOT / "CHANGELOG.md"


def _module():
    spec = importlib.util.spec_from_file_location("extract_changelog_release_notes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: a dataclass in the module resolves its own
    # module through sys.modules while the class body runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)  # the import the release workflow's final step makes
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _version() -> str:
    return runpy.run_path(str(ROOT / "_version.py"))["__version__"]


def test_the_extractor_imports_without_a_retired_module():
    assert callable(_module().extract_version_section)


def test_it_returns_one_released_section_from_the_real_changelog():
    notes = _module().extract_version_section(CHANGELOG.read_text(encoding="utf-8"), "2.0.1")
    assert notes.strip() and notes.endswith("\n")
    assert "## [" not in notes, "a section stops at the next release heading"


def test_it_fails_closed_rather_than_publishing_the_wrong_body():
    module = _module()
    changelog = CHANGELOG.read_text(encoding="utf-8")
    for version, reason in (("9.9.9", "missing"), ("3.1.0rc39", "not a release version"), ("", "empty")):
        with pytest.raises(ValueError):
            module.extract_version_section(changelog, version)


def test_a_release_version_has_a_section_to_publish():
    """Current development and final releases each need a complete changelog section."""
    version = _version()
    if "rc" in version:
        assert f"### Scope Recall {version} " in CHANGELOG.read_text(encoding="utf-8"), (
            f"{version} has no changelog entry"
        )
        return
    notes = _module().extract_version_section(CHANGELOG.read_text(encoding="utf-8"), version)
    assert notes.strip()


def test_the_script_runs_as_the_workflow_runs_it(tmp_path):
    output = tmp_path / "notes.md"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--version", "2.0.1", "--changelog", str(CHANGELOG), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert output.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("version", ["1.2.3", "1.2.3.dev0"])
def test_sections_keep_categories_and_end_before_the_next_version(version):
    body = "### Added\n\n- Shared entries.\n- Offline capture.\n\n### Fixed\n\n- Restore queued work."
    changelog = f"## [{version}]\n\n{body}\n\n## [1.2.2] - 2026-01-01\n\nOlder notes.\n"
    assert _module().extract_version_section(changelog, version) == body + "\n"


def test_only_final_versions_can_be_published():
    with pytest.raises(ValueError, match="major.minor.patch"):
        _module().extract_release_notes("## [1.2.3.dev0]\nNotes.", "1.2.3.dev0")


@pytest.mark.parametrize("body", ["", "### Fixed\n\n- TBD", "- TODO: finish.", "- FIXME details"])
def test_empty_or_unfinished_notes_are_rejected(body):
    with pytest.raises(ValueError):
        _module().extract_version_section(f"## [1.2.3]\n{body}\n## [1.2.2]\nDone.", "1.2.3")


def test_fenced_example_does_not_end_a_section_and_duplicate_sections_are_rejected():
    module = _module()
    body = "Example:\n\n```md\n## [1.2.2]\n```"
    changelog = f"## [1.2.3]\n{body}\n## [1.2.2]\nOlder notes."
    assert module.extract_version_section(changelog, "1.2.3") == body + "\n"
    with pytest.raises(ValueError):
        module.extract_version_section(changelog + "\n## [1.2.3]\nOther notes.", "1.2.3")
