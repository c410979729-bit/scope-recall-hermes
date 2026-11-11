"""Launch one bounded worker process without creating a second queue."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path

from .validation import absolute_path, strict_bool, strict_float, strict_int

#: A per-pass copy of a runtime config is written beside it as ``<stem>-worker-<8 random>.json``
#: (``adapters.runtime_wiring.write_ephemeral_worker_config``).  ``--cleanup-config`` deletes a
#: file of that name and nothing else: given an operator's real ``runtime-config.json`` it
#: would delete that, silently, and every host would drop to basic mode.
EPHEMERAL_CONFIG_INFIX = "-worker-"
#: A runtime config's file name, beside the binding's data or an entry's attachment.
RUNTIME_CONFIG_FILENAME = "runtime-config.json"

#: System executables, user locations, temporary files, locale and timezone must resolve as they do in the host.
_WORKER_ENV_NAMES = (
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "SYSTEMDRIVE",
    "HOME",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "APPDATA",
    "LOCALAPPDATA",
    "USER",
    "LOGNAME",
    "USERNAME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LC_COLLATE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "TZ",
    "TZDIR",
    "PYTHONTZPATH",
    # The same interpreter keeps its package policy, encoding and operator-selected execution behavior.
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "PYTHONWARNINGS",
    "PYTHONSAFEPATH",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "PYTHONCOERCECLOCALE",
    "PYTHONLEGACYWINDOWSSTDIO",
    "PYTHONLEGACYWINDOWSFSENCODING",
    # Interpreter and native dependencies must remain discoverable without enabling preload injection.
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    # SQLite scratch files and TLS configuration retain deployment-specific locations.
    "SQLITE_TMPDIR",
    "OPENSSL_CONF",
    # HTTP model routes and CLI subscriptions retain the operator's proxy and certificate locations.
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    # A subscription route locates its existing CLI login without inheriting ambient provider overrides.
    "CODEX_HOME",
    # NumPy's native vector math retains explicit CPU limits, affinity and Linux huge-page compatibility settings.
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_MAIN_FREE",
    "NUMPY_MADVISE_HUGEPAGE",
)


def _worker_environment(config_path: Path, environment: dict[str, str] | None) -> dict[str, str]:
    """Select child settings and declared route credentials, with caller values taking precedence.

    Overrides pass through the same selection as the parent, including explicitly requested deployment names.
    Windows aliases merge case-insensitively; the last explicit item in mapping iteration order wins when aliases
    collide. POSIX names remain distinct. Values may contain secrets, and the watchdog's descendants inherit the
    selected environment without a separate filtering policy.
    """
    from .instance_config import WORKER_ENVIRONMENT_FORBIDDEN
    from .worker_entry import load_config

    config = load_config(config_path)
    windows = os.name == "nt"
    overrides = environment or {}
    if windows:
        overrides = {name.upper(): value for name, value in overrides.items()}
    source = ChainMap(overrides, os.environ)
    names: set[str] = set(_WORKER_ENV_NAMES) | set(config.worker_environment_names)
    # Plugin-owned settings include runtime authorization paths and the isolated test process guard.
    names.update(name for name in source if name.startswith("SCOPE_RECALL_"))
    auxiliary = config.auxiliary
    # Only credentials named by the trusted runtime configuration are eligible; absent names stay absent.
    if auxiliary is not None:
        names.update(
            name
            for route in (auxiliary.embedding, auxiliary.consolidation)
            if (name := getattr(route, "credential_env", None))
        )
    if windows:
        names = {name.upper() for name in names}
    names.difference_update(WORKER_ENVIRONMENT_FORBIDDEN)
    return {name: source[name] for name in names if name in source}


def is_ephemeral_worker_config(path: Path) -> bool:
    """Whether ``path`` is named as a per-pass config copy, the only kind ``--cleanup-config`` removes."""
    stem, infix, tail = path.name.rpartition(EPHEMERAL_CONFIG_INFIX)
    random, _, suffix = tail.partition(".")
    return (
        bool(stem)
        and bool(infix)
        and suffix == "json"
        and len(random) == 8
        and all(character in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in random)
    )


def validate_wake_arguments(after_pid: int | None, delay_seconds: float) -> None:
    """The follow-up pid and delay a host may attach to a wake; shared with the watchdog."""
    if after_pid is not None:
        strict_int("worker_after_pid", after_pid, minimum=1, maximum=0xFFFFFFFF)
    strict_float("worker_delay_seconds", delay_seconds, minimum=0, maximum=3600)


def detached_creationflags() -> int:
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if os.name == "nt":
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    return flags


def taskkill_tree(process: subprocess.Popen) -> None:
    """Windows: kill the whole Popen-owned tree at once; terminating the parent
    first can orphan native children.  ``taskkill`` is a console program started
    from a process without a console (the task's ``pythonw``, a windowless
    worker), so without ``CREATE_NO_WINDOW`` Windows gives it a console window of
    its own, which flashed on the desktop at every worker teardown."""
    try:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5.0,
                creationflags=detached_creationflags(),
            )
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()


def reap_process(process: subprocess.Popen) -> None:
    """Wait briefly, then force the process (and its POSIX group) down."""
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                process.kill()
        process.wait(timeout=5.0)


@dataclass
class WorkerProcess:
    process: subprocess.Popen[str]
    config_path: Path
    upgrade_only: bool = False

    @property
    def pid(self) -> int:
        return int(self.process.pid)

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float = 120.0) -> int:
        strict_float("worker_wait_timeout", timeout, minimum=1e-9, maximum=120.0)
        try:
            return self.process.wait(timeout=float(timeout))
        except subprocess.TimeoutExpired:
            self._stop_owned()
            raise

    def communicate(self, timeout: float = 120.0) -> tuple[str, str]:
        strict_float("worker_communicate_timeout", timeout, minimum=1e-9, maximum=120.0)
        try:
            stdout, stderr = self.process.communicate(timeout=float(timeout))
        except subprocess.TimeoutExpired:
            self._stop_owned()
            # Reap pipes after the owned process tree has been stopped.  A
            # short bounded wait keeps a stuck child from becoming a leaked
            # worker while preserving the original timeout signal.
            try:
                self.process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.communicate(timeout=5.0)
            raise
        return stdout or "", stderr or ""

    def terminate(self) -> None:
        self._stop_owned()

    def _stop_owned(self) -> None:
        if self.process.poll() is not None:
            return
        if os.name == "nt":
            taskkill_tree(self.process)
        else:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                self.process.terminate()
        reap_process(self.process)


def launch_worker(
    config_path: str | Path,
    *,
    python_executable: str | Path | None = None,
    cleanup_config: bool = False,
    after_pid: int | None = None,
    delay_seconds: float = 0.0,
    detach_output: bool = False,
    environment: dict[str, str] | None = None,
    upgrade_only: bool = False,
) -> WorkerProcess:
    strict_bool("worker_detach_output", detach_output)
    strict_bool("worker_upgrade_only", upgrade_only)
    validate_wake_arguments(after_pid, delay_seconds)
    path = absolute_path("config", os.fspath(config_path))
    if not path.is_file():
        raise ValueError("config_not_found")
    path = path.resolve()
    executable = Path(python_executable) if python_executable is not None else Path(sys.executable)
    if not executable.is_absolute() or not executable.exists():
        raise ValueError("python_executable")
    command = [
        str(executable),
        "-m",
        "scope_recall.runtime.worker_watchdog",
        "--config",
        str(path),
        "--python",
        str(executable),
    ]
    if cleanup_config:
        command.append("--cleanup-config")
    if upgrade_only:
        command.append("--upgrade-only")
    if after_pid is not None:
        command.extend(("--after-pid", str(after_pid)))
    if delay_seconds:
        command.extend(("--delay-seconds", str(delay_seconds)))
    child_env = _worker_environment(path, environment)
    package_root = Path(__file__).resolve().parents[1]
    inherited_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = str(package_root) + (os.pathsep + inherited_pythonpath if inherited_pythonpath else "")
    process = subprocess.Popen(
        command,
        cwd=str(package_root),
        env=child_env,
        stdout=subprocess.DEVNULL if detach_output else subprocess.PIPE,
        stderr=subprocess.DEVNULL if detach_output else subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=detached_creationflags(),
        start_new_session=(os.name != "nt"),
    )
    return WorkerProcess(process=process, config_path=path, upgrade_only=upgrade_only)


#: ``ExceptionClass: message`` -- the last line of a traceback, and nothing else.
_TRACEBACK_TAIL = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt)\b.*")


def failure_reason(stderr: str) -> str | None:
    """The one line of a child's stderr that names why it died, or None.

    Only the last line of a Python traceback qualifies (``ModuleNotFoundError:
    No module named 'scope_recall'``): bounded, no paths, no model text.  A
    line that looks like a credential is dropped rather than recorded.
    """
    from ..core.secret_patterns import contains_secret_like_text

    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if _TRACEBACK_TAIL.match(line) and not contains_secret_like_text(line):
            return line[:200]
    return None


__all__ = ["WorkerProcess", "launch_worker", "validate_wake_arguments"]
