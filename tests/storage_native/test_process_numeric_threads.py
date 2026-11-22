"""Owned entry processes choose numerical threads before NumPy initializes."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OWNED_ENTRIES = (
    "adapters/clients/hook_entry.py",
    "adapters/clients/mcp_entry.py",
    "adapters/clients/remote_server.py",
    "adapters/clients/resident_entry.py",
    "adapters/codex/hook_entry.py",
    "adapters/codex/mcp_entry.py",
    "adapters/codex/remote_client.py",
    "adapters/codex/remote_server.py",
    "adapters/codex/resident_entry.py",
    "distribution/docker/entry.py",
    "maintenance/cli.py",
    "maintenance/migrate.py",
    "maintenance/upgrade_cli.py",
    "runtime/_worker_bootstrap.py",
    "runtime/resume_entry.py",
    "runtime/worker_entry.py",
    "runtime/worker_watchdog.py",
)
EXCEPTIONS = {
    "__init__.py": "The package can be imported by a host process.",
    "adapters/hermes/register.py": "Hermes owns the importing process.",
    "distribution/hermes/__init__.py": "Hermes loads this plugin wrapper in its own process.",
    "runtime/_http_worker.py": "The isolated HTTP transport uses only the standard library.",
    "maintenance/_compact_pager.py": "The isolated SQLite lock holder uses only the standard library.",
    "adapters/clients/remote_client.py": "The HTTP forwarding client does not perform vector computations.",
    "maintenance/autostart.py": "Scheduler administration launches workers in separate owned processes.",
    "maintenance/package_health.py": "Installed-artifact diagnostics do not import numerical libraries.",
    "maintenance/package_upgrade.py": "Offline package replacement does not perform vector computations.",
    "maintenance/private.py": "Private-mode management does not execute vector searches.",
    "maintenance/resident.py": "Resident process administration does not execute vector searches.",
}

PROBE = """
import atexit, importlib.abc, importlib.util, json, os, sys
from pathlib import Path
observed = []
class ObserveNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy':
            observed.append(os.environ.get('OPENBLAS_NUM_THREADS'))
sys.meta_path.insert(0, ObserveNumpy())
def finish():
    import numpy
    Path(OUTPUT).write_text(json.dumps(dict(observed=observed,
        final=os.environ.get('OPENBLAS_NUM_THREADS'))), encoding='utf-8')
atexit.register(finish)
spec = importlib.util.spec_from_file_location('scope_recall',
    str(Path(ROOT) / '__init__.py'), submodule_search_locations=[ROOT])
package = importlib.util.module_from_spec(spec)
sys.modules['scope_recall'] = package
spec.loader.exec_module(package)
"""


def _child(tmp_path: Path, arguments: list[str], value: str | None, *, release: bool = False) -> dict:
    """Observe the first NumPy import in a fresh interpreter with only synthetic paths."""
    probe = tmp_path / "probe"
    probe.mkdir()
    output = tmp_path / "observed.json"
    source = PROBE.replace("ROOT", repr(str(ROOT))).replace("OUTPUT", repr(str(output)))
    (probe / "sitecustomize.py").write_text(source, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(probe), env.get("PYTHONPATH", "")))
    env.pop("OPENBLAS_NUM_THREADS", None)
    if value is not None:
        env["OPENBLAS_NUM_THREADS"] = value
    done = subprocess.run(
        [sys.executable, "-B", *arguments],
        env=env,
        input=b"\x01" if release else b"",
        capture_output=True,
        timeout=60,
        check=False,
    )
    # The released bootstrap reaches a deliberately absent configuration after imports.
    assert done.returncode == (1 if release else 0), done.stderr.decode(errors="replace")
    assert output.is_file(), done.stderr.decode(errors="replace")
    return json.loads(output.read_text(encoding="utf-8"))


def _process_entries(root: Path) -> set[str]:
    """Discover startup guards in product sources independently of the wheel inventory."""
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    packages = config["tool"]["setuptools"]["packages"]
    directories = {root / package.split(".")[1] for package in packages if "." in package}
    sources = {path for directory in directories for path in directory.rglob("*.py")}
    sources.update(root.glob("*.py"))
    entries = set()
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
                continue
            test = node.test
            if len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                values = (test.left, test.comparators[0])
                if any(isinstance(v, ast.Name) and v.id == "__name__" for v in values) and any(
                    isinstance(v, ast.Constant) and v.value == "__main__" for v in values
                ):
                    entries.add(path.relative_to(root).as_posix())
    return entries


def test_every_entry_has_a_thread_policy_or_an_explained_exception():
    from packaging_hooks.module_inventory import ENTRY_POINTS

    discovered = _process_entries(ROOT)
    classified = set(OWNED_ENTRIES) | set(EXCEPTIONS)
    assert discovered <= classified, f"unclassified process entries: {sorted(discovered - classified)}"
    assert set(OWNED_ENTRIES) <= discovered
    assert set(ENTRY_POINTS) <= classified
    assert set(OWNED_ENTRIES) <= set(ENTRY_POINTS)
    assert not set(OWNED_ENTRIES) & set(EXCEPTIONS)
    for entry in OWNED_ENTRIES:
        tree = ast.parse((ROOT / entry).read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "configure_numeric_threads"
        ]
        assert calls, f"{entry} must configure its process before importing numerical code"


def test_process_discovery_finds_an_entry_missing_from_the_manifest(tmp_path: Path):
    (tmp_path / "runtime").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npackages = ["scope_recall", "scope_recall.runtime"]\n', encoding="utf-8"
    )
    (tmp_path / "runtime" / "new_entry.py").write_text('if __name__ == "__main__":\n    pass\n', encoding="utf-8")
    assert _process_entries(tmp_path) == {"runtime/new_entry.py"}


@pytest.mark.parametrize("entry", OWNED_ENTRIES)
@pytest.mark.parametrize("value", [None, "7"])
def test_owned_entry_sets_threads_before_numpy(tmp_path: Path, entry: str, value: str | None):
    if entry == "runtime/_worker_bootstrap.py":
        arguments = [str(ROOT / entry), str(tmp_path / "missing-config.json")]
        observed = _child(tmp_path, arguments, value, release=True)
    elif entry == "distribution/docker/entry.py":
        # Import the container composition root without acquiring its store lease.
        observed = _child(tmp_path, ["-c", "import scope_recall.distribution.docker.entry"], value)
    else:
        module = "scope_recall." + entry.removesuffix(".py").replace("/", ".")
        observed = _child(tmp_path, ["-m", module, "--help"], value)
    assert observed == {"observed": [value if value is not None else "1"], "final": value or "1"}


@pytest.mark.parametrize("value", [None, "7"])
def test_console_script_sets_threads_before_numpy(tmp_path: Path, value: str | None):
    code = "from scope_recall.maintenance.cli import main; raise SystemExit(main(['--help']))"
    observed = _child(tmp_path, ["-c", code], value)
    assert observed == {"observed": [value or "1"], "final": value or "1"}


@pytest.mark.parametrize("module", ["maintenance.migrate", "maintenance.cli"])
@pytest.mark.parametrize("value", [None, "7"])
def test_unprefixed_maintenance_entry_preserves_its_startup_contract(tmp_path: Path, module: str, value: str | None):
    observed = _child(tmp_path, ["-m", module, "--help"], value)
    assert observed == {"observed": [value or "1"], "final": value or "1"}


def test_real_package_initialization_does_not_preload_numpy(tmp_path: Path):
    code = (
        "import scope_recall, sys; from pathlib import Path; "
        f"assert Path(scope_recall.__file__).resolve() == Path({str(ROOT / '__init__.py')!r}).resolve(); "
        "assert 'numpy' not in sys.modules"
    )
    assert _child(tmp_path, ["-c", code], None) == {"observed": [None], "final": None}


@pytest.mark.parametrize("value", [None, "7"])
def test_hermes_imports_preserve_the_hosts_thread_policy(tmp_path: Path, value: str | None):
    # Setup verification shares worker helpers with an owned worker process.
    # Importing those helpers in Hermes must not configure the host's BLAS.
    code = (
        "import importlib, runpy; "
        f"runpy.run_path({str(ROOT / '__init__.py')!r}); "
        "importlib.import_module('scope_recall.adapters.hermes.register'); "
        "importlib.import_module('scope_recall.distribution.hermes'); "
        "importlib.import_module('scope_recall.maintenance.setup_verify')"
    )
    observed = _child(tmp_path, ["-c", code], value)
    assert observed == {"observed": [value], "final": value}


def test_thread_helper_import_does_not_load_numpy(tmp_path: Path):
    code = (
        "import sys; from scope_recall.core.process_environment import configure_numeric_threads; "
        "configure_numeric_threads(); assert 'numpy' not in sys.modules"
    )
    assert _child(tmp_path, ["-c", code], None)["final"] == "1"
