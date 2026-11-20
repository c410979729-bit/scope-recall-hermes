"""Single source for the Scope Recall distribution version.

``scripts/build.package_manifest.py`` stamps the adapter manifest and packaging
allowlist from here. During development the distribution manifest keeps the
latest release, because Hermes installs that directory from the default branch.
"""

import re

__version__ = "4.0.5"


def manifest_version(version: str) -> str:
    """Return the host SemVer spelling of a PEP 440 distribution version."""
    if ".dev" in version:
        return version.replace(".dev", "-dev.", 1)
    # Preserve trailing segments while converting the release-candidate prefix.
    return re.sub(r"(\d+\.\d+\.\d+)rc(\d+)", r"\1-rc.\2", version)


def stamp_manifest_version(text: str, version: str) -> str:
    """Replace a manifest's version with host spelling, preserving its line ending."""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("version:"):
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            lines[index] = f"version: {manifest_version(version)}{ending}"
            return "".join(lines)
    raise RuntimeError("plugin manifest has no version line")
