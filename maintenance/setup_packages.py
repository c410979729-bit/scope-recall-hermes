"""Choose a usable hook interpreter and plan installation into a fresh host venv."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .._version import __version__
from ..adapters.audiences import HermesIdentityError
from ..adapters.shared_entries import Attachment, read_shared_payload, shared_entry_for_home
from .doctor import probe_python_package
from .install_common import InstallError


def receipt_interpreter(receipt: dict, home: Path, host: str, attachment: Attachment | None) -> Path:
    """Use a receipt's launcher, recovering an old resolved path only from the same attached installation."""
    recorded = Path(receipt["python_executable"])
    if (
        receipt.get("python_executable_as_invoked") is True
        or attachment is None
        or attachment.host != host
        or receipt["host"] != host
        or Path(receipt["instance_root"]) != home
    ):
        return recorded
    payload = read_shared_payload(attachment.root)
    if payload["installation_id"] != receipt["installation_id"]:
        return recorded
    try:
        entry = shared_entry_for_home(payload, attachment.entry_id, hermes_home=home)
    except HermesIdentityError:
        return recorded
    value = entry.get("python_executable")
    if entry.get("host") == host and isinstance(value, str):
        launcher = Path(value)
        if launcher.is_absolute() and launcher.is_file() and launcher.resolve() == recorded:
            return launcher
    return recorded


def _source() -> str:
    """Use the installed development checkout locally; published releases use their exact package version."""
    if ".dev" in __version__:
        # A source checkout can expose egg-info before the installed dist-info;
        # its metadata then has no direct_url, and the dev build is not on PyPI.
        checkout = Path(__file__).resolve().parent.parent
        if (checkout / "pyproject.toml").is_file():
            return str(checkout)
        raw = distribution("hermes-scope-recall").read_text("direct_url.json")
        if raw:
            url = urlsplit(json.loads(raw).get("url", ""))
            if url.scheme == "file" and not url.netloc:
                path = unquote(url.path)
                if os.name == "nt" and path.startswith("/"):
                    path = path[1:]
                if Path(path).is_dir():
                    return path
    return "hermes-scope-recall==" + __version__


def choose_python(host: str, candidate: Path, *, explicit: bool) -> tuple[Path, list[str], str | None]:
    """Probe the chosen launcher; only an unpinned client environment may fall back to setup's interpreter."""
    if candidate == Path(sys.executable):
        return candidate, [], None
    probe = probe_python_package(candidate)
    if probe.get("version") == __version__:
        return candidate, [], None
    if not explicit and host != "hermes":
        return Path(sys.executable), [], None
    if probe.get("version"):
        return (
            candidate,
            [],
            "the host environment has another release. Coordinate its upgrade or select the correct venv with --python",
        )
    venv = candidate.parent.parent
    uv = shutil.which("uv")
    if not (venv / "pyvenv.cfg").is_file() or not uv:
        return (
            candidate,
            [],
            "install the package in the selected venv or use --python to select its launcher. "
            "uv is needed for a pip-less venv",
        )
    return candidate, [uv, "pip", "install", "--python", str(candidate), _source()], None


def install_package(command: list[str], python: Path) -> None:
    """Run only the approved venv package step, with bounded output and no credential-bearing error text."""
    if command:
        result = subprocess.run(command, capture_output=True, timeout=300)
        if result.returncode or probe_python_package(python).get("version") != __version__:
            raise InstallError("host venv package installation failed; preserve the environment and rerun setup")
