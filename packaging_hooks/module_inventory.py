"""Derive the wheel's module list and version stamps from the source tree.

The v3 wheel ships a deliberate subset of this repository: the 2.x modules
(``provider.py``, ``cli.py``, ``config_schema.py`` and friends) still live here
and must stay out.  Kept as a hand-copied list of paths, that subset would miss
every new module nobody remembered to add, found only when the packaging tier
runs.

The subset is not arbitrary, though: it is exactly what the v3 entry points can
reach by import.  So it can be computed, and the committed allowlist becomes a
reviewed artifact that a test compares against the computation rather than a
list somebody retypes.  The safety property holds -- a stray module still
cannot enter the wheel silently, because it can only enter by being imported
from an entry point, and the diff is shown to a human either way.

``_version.py`` supplies the version stamps. During development the catalog
manifest keeps a released core so default-branch installs can resolve it.

Not responsible for: building the wheel (``packaging_hooks/v11_build.py``) or
deciding what the entry points are -- that list is right here, in the open,
because it *is* the product's public surface.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from _version import manifest_version, stamp_manifest_version

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = REPO_ROOT / "packaging" / "v11-module-allowlist.json"

#: Every way the outside world enters this package: the package itself, the
#: host adapters' registration surfaces, the console/worker entry points, and
#: the module-form CLIs.  Adding a genuinely new entry point is a deliberate
#: act and belongs in this list; adding a module that an existing entry point
#: imports needs no edit at all.
ENTRY_POINTS = (
    "__init__.py",
    "adapters/clients/hook_entry.py",
    "adapters/clients/mcp_entry.py",
    "adapters/clients/remote_server.py",
    "adapters/clients/resident_entry.py",
    "adapters/codex/hook_entry.py",
    "adapters/codex/mcp_entry.py",
    "adapters/codex/remote_client.py",
    "adapters/codex/remote_server.py",
    "adapters/codex/resident_entry.py",
    "adapters/hermes/register.py",
    "distribution/hermes/__init__.py",
    "distribution/docker/entry.py",
    "maintenance/cli.py",
    "maintenance/migrate.py",
    "maintenance/upgrade_cli.py",
    "runtime/_http_worker.py",
    "runtime/_worker_bootstrap.py",
    "runtime/resume_entry.py",
    "runtime/worker_entry.py",
    "runtime/worker_watchdog.py",
)

#: Manifests whose version line is stamped in the spelling of ``manifest_version``: the adapter's always from
#: ``_version.py``, the catalog's from it too except during development (``expected_manifest``).
PLUGIN_MANIFESTS = ("adapters/hermes/plugin.yaml", "distribution/hermes/plugin.yaml")


def source_version(repo_root: Path | None = None) -> str:
    """The one true version, read as a literal so nothing has to be imported."""
    root = repo_root or REPO_ROOT
    tree = ast.parse((root / "_version.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            return node.value.value
    raise RuntimeError("_version.py does not define a literal __version__")


def _relative_paths(module: str, root: Path) -> list[str]:
    """The file or files that provide a dotted ``scope_recall.*`` name."""
    tail = module[len("scope_recall") :].lstrip(".")
    if not tail:
        return ["__init__.py"]
    base = tail.replace(".", "/")
    found = []
    if (root / f"{base}.py").is_file():
        found.append(f"{base}.py")
    if (root / base / "__init__.py").is_file():
        found.append(f"{base}/__init__.py")
    return found


def _imported_modules(relative: str, root: Path) -> set[str]:
    """Every ``scope_recall.*`` name one file imports, however deeply nested.

    Walks the whole tree rather than the module body because this code base
    imports inside functions on purpose -- to keep construction cheap and to
    break cycles -- and those imports are just as real at run time.
    """
    try:
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()
    parts = relative[:-3].split("/")
    package = ["scope_recall"] + parts[:-1]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - node.level + 1]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target.startswith("scope_recall"):
                found.add(target)
                # ``from x import y`` may name a submodule rather than a value.
                found.update(f"{target}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("scope_recall"))
    return found


def reachable_modules(repo_root: Path | None = None) -> list[str]:
    """Sorted module paths the v3 entry points can reach, as the wheel needs them."""
    root = repo_root or REPO_ROOT
    seen: set[str] = set()
    queue = [entry for entry in ENTRY_POINTS if (root / entry).is_file()]
    while queue:
        relative = queue.pop()
        if relative in seen or not (root / relative).is_file():
            continue
        seen.add(relative)
        for module in _imported_modules(relative, root):
            queue.extend(path for path in _relative_paths(module, root) if path not in seen)
    # A module is only importable if every package above it is too, and a
    # package __init__ that imports nothing would otherwise never be reached.
    for relative in list(seen):
        parts = relative.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            init = "/".join(parts[:depth]) + "/__init__.py"
            if (root / init).is_file():
                seen.add(init)
    return sorted(seen)


def reachable_packages(modules: list[str]) -> list[str]:
    """The distribution package names implied by a module list."""
    packages = {"scope_recall"}
    for relative in modules:
        parts = relative.split("/")[:-1]
        for depth in range(1, len(parts) + 1):
            packages.add("scope_recall." + ".".join(parts[:depth]))
    return sorted(packages)


def expected_allowlist(repo_root: Path | None = None) -> dict:
    """The allowlist this source tree implies, with curated fields preserved.

    ``package_data`` and ``forbidden_wheel_substrings`` stay hand-maintained:
    they are judgements about what must ship and what must never ship, and
    neither is derivable from imports.
    """
    root = repo_root or REPO_ROOT
    current = json.loads((root / "packaging" / "v11-module-allowlist.json").read_text(encoding="utf-8"))
    modules = reachable_modules(root)
    return {
        "schema": current["schema"],
        "package_version": source_version(root),
        "packages": reachable_packages(modules),
        "python_modules": modules,
        "package_data": list(current["package_data"]),
        "forbidden_wheel_substrings": list(current["forbidden_wheel_substrings"]),
    }


def stamped_manifest(text: str, version: str, *, core_dependency: bool = False) -> str:
    """Stamp the host SemVer derived from the distribution version, preserving other fields."""
    stamped = stamp_manifest_version(text, version)
    pattern = r"(?m)^pip_dependencies:\r?\n(?:[ \t]+[^\n]*\n?)*"
    if core_dependency or re.search(pattern, stamped):
        dependency = f'pip_dependencies:\n  - "hermes-scope-recall[vectors]=={version}"\n'
        stamped = (
            re.sub(pattern, lambda _: dependency, stamped)
            if re.search(pattern, stamped)
            else stamped.rstrip("\n") + "\n" + dependency
        )
    return stamped


def expected_manifest(text: str, version: str, relative: str) -> str | None:
    """Return the required manifest, or None when the catalog release cannot be determined.

    Hermes installs the catalog directory from the default branch, so its core
    must remain a published release while the source version is in development.
    During development only the manifest's own consistency is checked: its version
    line names a release and its pin equals it. Whether that release is the newest
    one published is not something this check can see.
    """
    catalog = relative == "distribution/hermes/plugin.yaml"
    if catalog and ".dev" in version:
        match = re.search(r"(?m)^version: (\d+\.\d+\.\d+)\r?$", text)
        if not match:
            return None
        version = match.group(1)
    return stamped_manifest(text, version, core_dependency=catalog)


def stale_version_files(repo_root: Path | None = None) -> dict[str, str]:
    """Report stamps that do not follow the source and released catalog rules."""
    root = repo_root or REPO_ROOT
    version = source_version(root)
    stale: dict[str, str] = {}
    for relative in PLUGIN_MANIFESTS:
        text = (root / relative).read_text(encoding="utf-8")
        expected = expected_manifest(text, version, relative)
        if expected is None:
            stale[relative] = "during development the catalog manifest must keep the latest release version"
        elif text != expected:
            stale[relative] = "version or core dependency pin differs"
    listed = json.loads((root / "packaging" / "v11-module-allowlist.json").read_text(encoding="utf-8"))
    if listed.get("package_version") != version:
        stale["packaging/v11-module-allowlist.json"] = str(listed.get("package_version"))
    return stale


def write_all(repo_root: Path | None = None) -> list[str]:
    """Bring every generated file in line with the source.  Returns what changed."""
    root = repo_root or REPO_ROOT
    version = source_version(root)
    changed: list[str] = []
    for relative in PLUGIN_MANIFESTS:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        stamped = expected_manifest(text, version, relative)
        if stamped is not None and stamped != text:
            path.write_text(stamped, encoding="utf-8", newline="\n")
            changed.append(relative)
    allowlist_path = root / "packaging" / "v11-module-allowlist.json"
    expected = expected_allowlist(root)
    current = json.loads(allowlist_path.read_text(encoding="utf-8"))
    if current != expected:
        allowlist_path.write_text(
            json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        changed.append("packaging/v11-module-allowlist.json")
    return changed


__all__ = [
    "ALLOWLIST_PATH",
    "ENTRY_POINTS",
    "PLUGIN_MANIFESTS",
    "REPO_ROOT",
    "expected_allowlist",
    "expected_manifest",
    "manifest_version",
    "reachable_modules",
    "reachable_packages",
    "source_version",
    "stale_version_files",
    "stamped_manifest",
    "write_all",
]
