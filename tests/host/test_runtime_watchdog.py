"""Real process-tree lifecycle checks for the owned runtime watchdog."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from scope_recall.adapters.clients import install_codex_scope_recall
from scope_recall.adapters.runtime_wiring import write_ephemeral_worker_config
from scope_recall.runtime import worker_launch, worker_watchdog
from scope_recall.runtime.worker_entry import FINALIZE_MARGIN_SECONDS
from scope_recall.runtime.worker_launch import launch_worker
from scope_recall.runtime.worker_watchdog import KILL_GRACE_SECONDS, _kill_tree, _OwnedWindowsJob


def _alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if handle:
        kernel.CloseHandle(handle)
        return True
    return False


def test_owned_watchdog_kills_real_child_tree_after_abrupt_parent_exit(tmp_path: Path):
    pid_file = tmp_path / "child.pid"
    script = (
        "import pathlib,subprocess,sys,time; "
        "assert sys.stdin.buffer.read(1)==b'\\x01'; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid), encoding='ascii'); "
        "__import__('os')._exit(17)"
    )
    flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if os.name == "nt":
        flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    parent = subprocess.Popen(
        [sys.executable, "-c", script, str(pid_file)],
        stdin=subprocess.PIPE,
        creationflags=flags,
        start_new_session=(os.name != "nt"),
    )
    job = _OwnedWindowsJob()
    try:
        assert job.assign(parent)
        assert parent.stdin is not None
        parent.stdin.write(b"\x01")
        parent.stdin.close()
        deadline = time.monotonic() + 5.0
        while parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert parent.poll() == 17
        # The parent wrote the child's pid before it exited, so the file holds all of it now.
        child_pid = int(pid_file.read_text(encoding="ascii"))
        _kill_tree(parent, job)
        assert parent.poll() is not None
        deadline = time.monotonic() + 5.0
        while _alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _alive(child_pid)
    finally:
        _kill_tree(parent, job)


def test_bootstrap_exits_without_running_worker_when_owner_pipe_closes(tmp_path: Path):
    bootstrap = Path(worker_watchdog.__file__).with_name("_worker_bootstrap.py")
    worker = subprocess.Popen(
        [sys.executable, "-I", "-B", str(bootstrap), str(tmp_path / "missing.json")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    stdout, stderr = worker.communicate(input=b"", timeout=5)
    assert worker.returncode == 125
    assert stdout == stderr == b""


def test_bootstrap_with_a_handed_deadline_still_waits_for_the_release_token(tmp_path: Path):
    bootstrap = Path(worker_watchdog.__file__).with_name("_worker_bootstrap.py")
    for extra in ([repr(time.time() + 60)], [repr(time.time() + 60), "TEST-unexpected"]):
        worker = subprocess.Popen(
            [sys.executable, "-I", "-B", str(bootstrap), str(tmp_path / "missing.json"), *extra],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
        )
        stdout, stderr = worker.communicate(input=b"", timeout=5)
        assert worker.returncode == 125
        assert stdout == stderr == b""


def test_backup_child_publishes_with_a_budget_independent_of_the_model_drain(tmp_path, capsys):
    from scope_recall.runtime.backup_bundle import verify_backup_bundle

    config_path, _ = _runtime_payload(tmp_path, drain_seconds=0.1)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = Path(config["binding"]["data_directory"])
    (data / "backup-config.json").write_text(json.dumps({"backup_dir": str(tmp_path / "backups")}))
    result = {}
    assert (
        worker_watchdog._run_once(
            config_path,
            Path(sys.executable),
            cleanup_config=False,
            timeout_seconds=30,
            result_sink=result,
            job_kind="backup",
        )
        == 0
    )
    capsys.readouterr()
    assert result["backup_outcome"] == "backed_up"
    state = json.loads((data / "backup-state.json").read_text())
    assert verify_backup_bundle(Path(state["last_bundle"]))["status"] == "verified"
    assert not (data / "runtime-worker-day.json").exists()
    assert not (data / "runtime-worker-status.json").exists()


# The real bootstrap and worker entry, with only the Core drain replaced: "busy"
# uses the whole window it is given, "hung" never returns.
_OWNED_CHILD = """import json, runpy, sys, time
from pathlib import Path
import scope_recall.core.worker as worker

record, mode = Path(sys.argv[1]), sys.argv[2]
sys.argv = sys.argv[3:]


def drain(*args, remaining_seconds, **kwargs):
    now = time.time()
    record.write_text(json.dumps({"handed": float(sys.argv[2]), "drain_ends": now + remaining_seconds}))
    time.sleep(remaining_seconds if mode == "busy" else 600)
    return worker.WorkerReceipt(0, 0, 0, 0, 0, 0, 0, True, ())


worker.drain_worker = drain
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _owned_children(monkeypatch, tmp_path: Path, mode: str) -> tuple[Path, list]:
    wrapper = tmp_path / "TEST-owned-child.py"
    wrapper.write_text(_OWNED_CHILD, encoding="utf-8")
    record = tmp_path / f"TEST-{mode}-drain.json"
    children = []
    real_popen = subprocess.Popen

    def spawn(command, **kwargs):
        if len(command) > 2 and str(command[2]).endswith("_worker_bootstrap.py"):
            command = [*command[:2], str(wrapper), str(record), mode, *command[2:]]
        children.append(real_popen(command, **kwargs))
        return children[-1]

    monkeypatch.setattr(worker_watchdog.subprocess, "Popen", spawn)
    return record, children


def test_busy_child_finishes_inside_the_window_it_was_handed(tmp_path: Path, monkeypatch, capsys):
    """The watchdog's deadline starts before the wait for a predecessor and the
    spawn; the child restarted a full drain budget after its own start-up.  A
    busy child was killed with 124 before its receipt, and the page it had
    reserved for the day was never refunded."""
    config_path, _ = _runtime_payload(tmp_path, drain_seconds=120.0)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    window = 8.0  # The supervisor's, which is tighter than drain_seconds here.
    payload.update(supervisor_seconds=window, supervisor_max_drains=1)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    data = Path(payload["binding"]["data_directory"])
    predecessor = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(2.5)"],
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    record, children = _owned_children(monkeypatch, tmp_path, "busy")
    try:
        started = time.time()
        code = worker_watchdog.run(config_path, Path(sys.executable), cleanup_config=False, after_pid=predecessor.pid)
        finished = time.time()
    finally:
        if predecessor.poll() is None:
            predecessor.kill()
        predecessor.wait(timeout=5)
    assert code == 0
    drained = json.loads(record.read_text(encoding="utf-8"))
    # The supervisor's remaining window, not restarted after the 2.5 s wait.
    # The slack covers the supervisor's own synced state writes before the pass.
    assert drained["handed"] <= started + window + 1.0
    clock_reads = 0.1  # Epoch/monotonic conversions, a few 15.6 ms Windows ticks.
    assert drained["drain_ends"] <= drained["handed"] - FINALIZE_MARGIN_SECONDS + clock_reads
    assert finished < drained["handed"] + KILL_GRACE_SECONDS  # It exited; nothing was killed.
    receipt = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert receipt["status"] == "idle" and receipt["daily_queue_used"] == 0
    status = json.loads((data / "runtime-worker-status.json").read_text(encoding="utf-8"))
    assert status["exit_code"] == 0 and status["status"] == "idle"
    # The whole page reserved before the drain was refunded after it.
    assert json.loads((data / "runtime-worker-day.json").read_text(encoding="utf-8"))["used"] == 0
    assert children and all(child.poll() is not None for child in children)


def test_hung_child_is_still_killed_with_124_after_the_grace(tmp_path: Path, monkeypatch, capsys):
    drain_seconds = 5.0
    config_path, _ = _runtime_payload(tmp_path, drain_seconds=drain_seconds)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["supervisor_enabled"] = False
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    data = Path(payload["binding"]["data_directory"])
    record, children = _owned_children(monkeypatch, tmp_path, "hung")
    started = time.monotonic()
    assert worker_watchdog.run(config_path, Path(sys.executable), cleanup_config=False) == 124
    assert time.monotonic() - started >= drain_seconds + KILL_GRACE_SECONDS
    assert record.exists()  # It hung inside the drain, not before it.
    assert json.loads(capsys.readouterr().out)["capability_gaps"] == ["worker_watchdog_timeout"]
    status = json.loads((data / "runtime-worker-status.json").read_text(encoding="utf-8"))
    assert status["exit_code"] == 124 and status["capability_gaps"] == ["worker_watchdog_timeout"]
    assert children and all(child.poll() is not None for child in children)
    # A killed pass keeps the page it reserved: a hung worker gets no free retries.
    assert json.loads((data / "runtime-worker-day.json").read_text(encoding="utf-8"))["used"] == 32


def test_assignment_failure_aborts_real_blocked_worker(tmp_path: Path, monkeypatch, capsys):
    config_path, _ = _runtime_payload(tmp_path, drain_seconds=30.0)
    children = []
    real_popen = subprocess.Popen

    def record_child(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        children.append(process)
        return process

    class FailedOwnership:
        def assign(self, _process):
            raise OSError("TEST_job_assignment_failed")

        def close(self):
            pass

    monkeypatch.setattr(worker_watchdog, "_OwnedWindowsJob", FailedOwnership)
    monkeypatch.setattr(worker_watchdog.subprocess, "Popen", record_child)
    assert worker_watchdog.run(config_path, Path(sys.executable), cleanup_config=False) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["capability_gaps"] == ["watchdog_error:OSError"]
    assert config_path.exists()
    assert children and all(child.poll() is not None for child in children)


def test_detached_trailing_wake_survives_short_host_shutdown(tmp_path: Path):
    config_path, _ = _runtime_payload(tmp_path, drain_seconds=5.0)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["worker_min_interval_seconds"] = 1.0
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    marker = tmp_path / "host-receipt.json"
    script = """import json,os,sys,time
from pathlib import Path
from scope_recall.runtime.worker_entry import load_config
from scope_recall.adapters.hermes.runtime_wiring import attach_trusted_host_runtime
path=Path(sys.argv[1]); config=load_config(path)
host=attach_trusted_host_runtime(config_path=path,expected_binding=config.binding,
    session_id=config.session_id,allowed_scope_ids=config.allowed_scope_ids)
host._last_worker_launch=time.monotonic()
host.maybe_launch_bounded_worker(session_id=config.session_id,allowed_scope_ids=config.allowed_scope_ids)
host.maybe_launch_bounded_worker(session_id=config.session_id,allowed_scope_ids=config.allowed_scope_ids)
active,tail=host._owned_worker,host._trailing_worker
Path(sys.argv[2]).write_text(json.dumps({'pids':[active.pid,tail.pid],
    'configs':[str(active.config_path),str(tail.config_path)]}),encoding='utf-8')
host.close()
os._exit(0)
"""
    env = os.environ.copy()
    root = str(Path(worker_watchdog.__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    parent = subprocess.Popen(
        [sys.executable, "-B", "-c", script, str(config_path), str(marker)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
    )
    out, err = parent.communicate(timeout=5)
    assert parent.returncode == 0, (out, err)
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 8
    while any(_alive(pid) for pid in receipt["pids"]) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any(_alive(pid) for pid in receipt["pids"])
    assert all(not Path(path).exists() for path in receipt["configs"])
    status = json.loads((Path(payload["binding"]["data_directory"]) / "runtime-worker-status.json").read_text())
    assert status["exit_code"] == 0 and status["status"] == "idle"


def _runtime_payload(tmp_path: Path, *, drain_seconds: float) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir(parents=True)
    config, _core = install_codex_scope_recall(tmp_path / "install", project_root=project, test_mode=True)
    path = tmp_path / "runtime.json"
    binding = config.to_binding()
    path.write_text(
        json.dumps(
            {
                "binding": {
                    "agent_id": binding.agent_id,
                    "installation_id": binding.installation_id,
                    "data_directory": str(binding.data_directory),
                    "scope_ids": sorted(binding.scope_ids),
                    "test_mode": binding.test_mode,
                },
                "session_id": "TEST-watchdog",
                "allowed_scope_ids": sorted(binding.scope_ids),
                "auxiliary": {"external_embedding": False, "external_consolidation": False},
                "drain_seconds": drain_seconds,
            }
        ),
        encoding="utf-8",
    )
    return path, config.config_path


@pytest.mark.parametrize("override", [False, True])
def test_launch_worker_limits_environment_through_watchdog_and_child(tmp_path, monkeypatch, override):
    """Probe real interpreters at both spawn boundaries without opening a model route."""
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["auxiliary"]["embedding"] = {"credential_env": "TEST_DECLARED_KEY"}
    payload["auxiliary"]["consolidation"] = {
        "model": "deepseek-v4-flash",
        "endpoint": "https://example.test/v1/chat/completions",
        "credential_env": "TEST_MISSING_KEY",
        "output_limit_field": "max_tokens",
        "max_output_tokens": 512,
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.delenv("TEST_MISSING_KEY", raising=False)
    values = {
        "TEST_UNRELATED_KEY": "synthetic-unrelated",
        "TEST_DECLARED_KEY": "synthetic-parent-route",
        "HTTP_PROXY": "http://proxy.example.test:8080",
        "HTTPS_PROXY": "http://proxy.example.test:8080",
        "ALL_PROXY": "http://proxy.example.test:8080",
        "NO_PROXY": "localhost,example.test",
        "SSL_CERT_FILE": str(tmp_path / "TEST-ca.pem"),
        "SSL_CERT_DIR": str(tmp_path / "TEST-certificates"),
        "REQUESTS_CA_BUNDLE": str(tmp_path / "TEST-requests.pem"),
        "CURL_CA_BUNDLE": str(tmp_path / "TEST-curl.pem"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "NUMPY_MADVISE_HUGEPAGE": "0",
        "OPENBLAS_MAIN_FREE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CODEX_HOME": str(tmp_path / "TEST-cli-home"),
        "SCOPE_RECALL_TEST_ENV_MARKER": "synthetic-plugin-setting",
    }
    if os.name != "nt":
        values.update({name.lower(): value for name, value in values.items() if name.endswith("_PROXY")})
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    required = ("SYSTEMROOT", "WINDIR", "COMSPEC", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP")
    if os.name != "nt":
        required = ("HOME", "TMPDIR", "LANG", "LC_ALL", "TZ")
        for name in required:
            if name not in os.environ:
                monkeypatch.setenv(
                    name, {"TMPDIR": str(tmp_path), "LANG": "C", "LC_ALL": "C", "TZ": "UTC"}.get(name, str(tmp_path))
                )
    expected = {name: os.environ[name] for name in (*values, *required, "PATH")}
    del expected["TEST_UNRELATED_KEY"]
    inherited = os.environ.get("PYTHONPATH")
    expected["PYTHONPATH"] = str(Path(worker_launch.__file__).resolve().parents[1]) + (
        os.pathsep + inherited if inherited else ""
    )
    environment = None
    if override:
        environment = {
            "TEST_DECLARED_KEY": "synthetic-profile-route",
            "TEST_UNRELATED_OVERRIDE": "synthetic",
            "PYTHONPATH": (inherited + os.pathsep if inherited else "") + str(tmp_path),
        }
        expected["TEST_DECLARED_KEY"] = environment["TEST_DECLARED_KEY"]
        expected["PYTHONPATH"] = (
            str(Path(worker_launch.__file__).resolve().parents[1]) + os.pathsep + environment["PYTHONPATH"]
        )
    names = [*expected, "TEST_UNRELATED_KEY", "TEST_UNRELATED_OVERRIDE", "TEST_MISSING_KEY"]
    probe = (
        "import json,os,sys; print(json.dumps({n:os.environ[n] for n in json.loads(sys.argv[1]) if n in os.environ}))"
    )
    wrapper = tmp_path / "TEST-worker-environment.py"
    wrapper.write_text(
        "import json,os,subprocess,sys,time\nfrom pathlib import Path\n"
        "from scope_recall.runtime import worker_watchdog\n"
        f"names = {names!r}\nprobe = {probe!r}\n"
        "owner = {n:os.environ[n] for n in names if n in os.environ}\n"
        "real_popen = subprocess.Popen\n"
        "def spawn(command, **kwargs):\n"
        "    return real_popen([command[0], '-c', probe, json.dumps(names)], **kwargs)\n"
        "worker_watchdog.subprocess.Popen = spawn\n"
        "child = worker_watchdog._spawn_worker(Path(sys.argv[1]), Path(sys.executable), time.monotonic()+30)\n"
        "out, err = child.communicate(timeout=15)\n"
        "assert child.returncode == 0, err\n"
        "print(json.dumps({'watchdog': owner, 'worker': json.loads(out)}))\n",
        encoding="utf-8",
    )
    real_popen = subprocess.Popen

    def spawn(command, **kwargs):
        return real_popen([command[0], str(wrapper), str(config)], **kwargs)

    monkeypatch.setattr(worker_launch.subprocess, "Popen", spawn)
    child = launch_worker(config, environment=environment)
    out, err = child.communicate(timeout=30)
    assert child.process.returncode == 0, err
    observed = json.loads(out)
    assert observed["watchdog"] == expected
    assert observed["worker"] == expected
    assert os.environ["TEST_DECLARED_KEY"] == "synthetic-parent-route"


@pytest.mark.parametrize(
    "locations",
    [
        {
            name: "TEST-windows-location"
            for name in (
                "SYSTEMROOT",
                "WINDIR",
                "COMSPEC",
                "PATHEXT",
                "USERPROFILE",
                "APPDATA",
                "LOCALAPPDATA",
                "TEMP",
                "TMP",
            )
        },
        {name: "TEST-posix-location" for name in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "TZ")},
    ],
)
def test_worker_environment_preserves_platform_locations(tmp_path, locations):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    selected = worker_launch._worker_environment(config, locations)
    assert {name: selected[name] for name in locations} == locations


def _environment_probe(config, monkeypatch, environment=None):
    """Read interpreter flags and environment from a real launcher's child."""
    real_popen = subprocess.Popen
    selected = {}
    probe = (
        "import json,os,sys; print(json.dumps({'environment':dict(os.environ),"
        "'no_user_site':sys.flags.no_user_site,'safe_path':sys.flags.safe_path,"
        "'write_bytecode':sys.dont_write_bytecode,'warnings':sys.warnoptions,"
        "'https_proxy_lower':os.environ.get('https_proxy')}))"
    )

    def spawn(command, **kwargs):
        selected.update(kwargs["env"])
        return real_popen([command[0], "-c", probe], **kwargs)

    monkeypatch.setattr(worker_launch.subprocess, "Popen", spawn)
    child = launch_worker(config, environment=environment)
    out, err = child.communicate(timeout=30)
    assert child.process.returncode == 0, err
    return selected, json.loads(out)


@pytest.mark.parametrize("value", ["", "TEST-timezone-directory"])
def test_worker_preserves_python_timezone_path(tmp_path, monkeypatch, value):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    monkeypatch.setenv("PYTHONTZPATH", value)
    _selected, observed = _environment_probe(config, monkeypatch)
    assert observed["environment"].get("PYTHONTZPATH") == value


def test_worker_preserves_disabled_user_site(tmp_path, monkeypatch):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    _selected, observed = _environment_probe(config, monkeypatch)
    assert observed["no_user_site"] == 1
    assert observed["environment"].get("PYTHONNOUSERSITE") == "1"


@pytest.mark.skipif(os.name != "nt", reason="Windows environment names are case-insensitive")
@pytest.mark.parametrize("name", ["HTTPS_PROXY", "https_proxy"])
def test_windows_worker_proxy_override_is_unique_and_reaches_child(tmp_path, monkeypatch, name):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    monkeypatch.setenv("HTTPS_PROXY", "http://parent.example.test:8080")
    value = "http://override.example.test:9090"
    selected, observed = _environment_probe(config, monkeypatch, {name: value})
    assert observed["environment"]["HTTPS_PROXY"] == value
    assert [key for key in selected if key.upper() == "HTTPS_PROXY"] == ["HTTPS_PROXY"]


@pytest.mark.skipif(os.name != "nt", reason="Windows environment names are case-insensitive")
@pytest.mark.parametrize("names", [("HTTPS_PROXY", "https_proxy"), ("https_proxy", "HTTPS_PROXY")])
def test_windows_worker_explicit_alias_order_and_lowercase_lookup(tmp_path, monkeypatch, names):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    monkeypatch.setenv("HTTPS_PROXY", "http://parent.example.test:8080")
    environment = {names[0]: "http://first.example.test:8081", names[1]: "http://last.example.test:8082"}
    selected, observed = _environment_probe(config, monkeypatch, environment)
    assert [key for key in selected if key.upper() == "HTTPS_PROXY"] == ["HTTPS_PROXY"]
    assert observed["environment"]["HTTPS_PROXY"] == environment[names[1]]
    assert observed["https_proxy_lower"] == environment[names[1]]


@pytest.mark.parametrize(
    "name",
    [
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONBREAKPOINT",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "LD_AUDIT",
        "PYTHON_PRESITE",
    ],
)
def test_declared_credentials_cannot_reintroduce_code_loading_overrides(tmp_path, monkeypatch, name):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["auxiliary"]["embedding"] = {"credential_env": name}
    config.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(name, "TEST-parent-code-reference")
    selected = worker_launch._worker_environment(config, {name: "TEST-explicit-code-reference"})
    assert name not in selected


def test_posix_worker_keeps_environment_name_case_distinct(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    monkeypatch.setattr(
        worker_launch, "os", SimpleNamespace(name="posix", environ={"HTTPS_PROXY": "upper", "https_proxy": "lower"})
    )
    selected = worker_launch._worker_environment(config, {"HTTPS_PROXY": "override"})
    assert selected["HTTPS_PROXY"] == "override"
    assert selected["https_proxy"] == "lower"


def test_worker_preserves_python_behavior_and_deployment_locations(tmp_path, monkeypatch):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    values = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONWARNINGS": "ignore::UserWarning",
        "PYTHONSAFEPATH": "1",
        "LD_LIBRARY_PATH": str(tmp_path),
        "DYLD_LIBRARY_PATH": str(tmp_path),
        "DYLD_FALLBACK_LIBRARY_PATH": str(tmp_path),
        "SQLITE_TMPDIR": str(tmp_path),
        "OPENSSL_CONF": str(tmp_path / "TEST-openssl.cnf"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    for name in (
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONBREAKPOINT",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
    ):
        monkeypatch.setenv(name, "TEST-excluded")
    selected, observed = _environment_probe(config, monkeypatch)
    assert {name: observed["environment"].get(name) for name in values} == values
    assert observed["write_bytecode"] and observed["safe_path"]
    assert "ignore::UserWarning" in observed["warnings"]
    assert (
        not {"PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONBREAKPOINT", "LD_PRELOAD", "DYLD_INSERT_LIBRARIES"}
        & selected.keys()
    )


def test_worker_extra_environment_names_are_selected_with_overrides(tmp_path, monkeypatch):
    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["worker_environment_names"] = ["TEST_DEPLOYMENT_LOCATION", "TEST_ABSENT_LOCATION"]
    config.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("TEST_DEPLOYMENT_LOCATION", "parent-location")
    monkeypatch.delenv("TEST_ABSENT_LOCATION", raising=False)
    selected, observed = _environment_probe(config, monkeypatch, {"TEST_DEPLOYMENT_LOCATION": "override-location"})
    assert observed["environment"]["TEST_DEPLOYMENT_LOCATION"] == "override-location"
    assert "TEST_ABSENT_LOCATION" not in selected


def test_filtered_environment_runs_real_bootstrap_with_monthly_timezone_and_native_dependencies(tmp_path, monkeypatch):
    """Use a synthetic TZif file absent from system data, then execute the unmodified worker bootstrap."""
    import struct
    import zoneinfo

    config, _ = _runtime_payload(tmp_path, drain_seconds=30)
    zone_root = tmp_path / "TEST-zoneinfo"
    zone_file = zone_root / "TEST" / "Worker"
    zone_file.parent.mkdir(parents=True)
    zone_file.write_bytes(
        b"TZif\x00"
        + b"\x00" * 15
        + struct.pack(">6l", 0, 0, 0, 0, 1, 4)
        + struct.pack(">lbb", -18000, 0, 0)
        + b"TST\x00"
    )
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["auxiliary"]["budget"] = {
        "period": "monthly",
        "time_zone": "TEST/Worker",
        "approved_models": [],
        "pricing": {},
    }
    payload["supervisor_enabled"] = False
    config.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("PYTHONTZPATH", str(zone_root))
    original_tzpath = zoneinfo.TZPATH
    try:
        zoneinfo.reset_tzpath()
        zoneinfo.ZoneInfo.clear_cache()
        environment = worker_launch._worker_environment(config, None)
    finally:
        zoneinfo.reset_tzpath(original_tzpath)
        zoneinfo.ZoneInfo.clear_cache()
    environment["PYTHONPATH"] = (
        str(Path(worker_launch.__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    bootstrap = Path(worker_watchdog.__file__).with_name("_worker_bootstrap.py")
    probe = (
        "import json,runpy,sqlite3,sys,zoneinfo; import numpy as np; from datetime import datetime; "
        "from scope_recall.runtime.worker_entry import load_config; "
        "cfg=load_config(sys.argv[2]); zone=zoneinfo.ZoneInfo(cfg.auxiliary.budget.time_zone); "
        "connection=sqlite3.connect(':memory:'); "
        "print(json.dumps({'offset':datetime(2026,1,1,tzinfo=zone).utcoffset().total_seconds(),"
        "'tzpath':list(zoneinfo.TZPATH),'numpy':int(np.array([1,2,3]).sum()),"
        "'sqlite':connection.execute('select 42').fetchone()[0]}),flush=True); connection.close(); "
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", probe, str(bootstrap), str(config)],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=worker_launch.detached_creationflags(),
    )
    try:
        out, err = child.communicate(input=b"\x01", timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
    assert child.returncode == 0, err.decode("utf-8", errors="replace")
    native, receipt = [json.loads(line) for line in out.splitlines()]
    assert native == {"offset": -18000, "tzpath": [str(zone_root)], "numpy": 6, "sqlite": 42}
    assert receipt["status"] == "idle" and receipt["processed"] == 0


def test_schema_child_observes_pause_after_delayed_wake(tmp_path, monkeypatch, capsys):
    import hashlib
    import sqlite3
    from contextlib import closing

    from v11_support import downgrade_store

    path, _ = _runtime_payload(tmp_path, drain_seconds=30)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["supervisor_enabled"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    data = Path(payload["binding"]["data_directory"])
    downgrade_store(data / "memory.sqlite3", 1111)
    identity = payload["binding"]["installation_id"]
    control = {
        "installation_id": identity,
        "enabled": True,
        "task_name": "ScopeRecall-" + hashlib.sha256(identity.encode()).hexdigest()[:20],
    }
    switch = data / "runtime-autostart.json"
    switch.write_text(json.dumps(control), encoding="utf-8")
    spawn = worker_watchdog._spawn_worker

    def pause_before_child(*args, **kwargs):
        switch.write_text(json.dumps({**control, "enabled": False}), encoding="utf-8")
        return spawn(*args, **kwargs)

    monkeypatch.setattr(worker_watchdog, "_spawn_worker", pause_before_child)
    assert worker_watchdog.run(path, Path(sys.executable), cleanup_config=False, delay_seconds=0.03) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "paused"
    with closing(sqlite3.connect(data / "memory.sqlite3")) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1111
    assert not (data / "store-upgrade.json").exists()


def test_worker_config_cleanup_is_explicit_and_bounded(tmp_path: Path):
    persistent, _installation = _runtime_payload(tmp_path / "persistent", drain_seconds=30.0)
    normal = launch_worker(persistent, python_executable=sys.executable)
    assert normal.wait(timeout=30.0) == 0
    normal.communicate(timeout=5.0)
    assert persistent.exists()

    timed, _installation = _runtime_payload(tmp_path / "timeout", drain_seconds=0.001)
    timeout_worker = launch_worker(timed, python_executable=sys.executable)
    assert timeout_worker.wait(timeout=30.0) == 124
    timeout_worker.communicate(timeout=5.0)
    data_dir = Path(json.loads(timed.read_text(encoding="utf-8"))["binding"]["data_directory"])
    status = json.loads((data_dir / "runtime-worker-status.json").read_text(encoding="utf-8"))
    assert status["exit_code"] == 124
    assert status["capability_gaps"] == ["worker_watchdog_timeout"]
    assert status["worker_pid"] > 0 and status["finished_at"] >= status["started_at"]
    assert timed.exists()

    ephemeral = write_ephemeral_worker_config(
        persistent,
        session_id="TEST-ephemeral",
        allowed_scope_ids=frozenset(json.loads(persistent.read_text(encoding="utf-8"))["allowed_scope_ids"]),
    )
    owned = launch_worker(ephemeral, python_executable=sys.executable, cleanup_config=True)
    assert owned.wait(timeout=30.0) == 0
    owned.communicate(timeout=5.0)
    assert persistent.exists()
    assert not ephemeral.exists()


def test_cleanup_config_never_deletes_a_config_it_did_not_write(tmp_path: Path):
    """--cleanup-config deletes only a config the run wrote.  Given an operator's real
    runtime-config.json, deleting it would leave no trace, and every host would drop to basic mode."""
    from scope_recall.runtime.worker_launch import is_ephemeral_worker_config

    real, _installation = _runtime_payload(tmp_path / "real", drain_seconds=30.0)
    worker = launch_worker(real, python_executable=sys.executable, cleanup_config=True)
    assert worker.wait(timeout=30.0) == 0
    _out, err = worker.communicate(timeout=5.0)
    assert real.exists(), "a config the run did not write outlives the run"
    assert "not_an_ephemeral_worker_config" in err, "the refusal is said, not silent"
    ephemeral = write_ephemeral_worker_config(
        real,
        session_id="TEST-naming",
        allowed_scope_ids=frozenset(json.loads(real.read_text(encoding="utf-8"))["allowed_scope_ids"]),
    )
    assert is_ephemeral_worker_config(ephemeral), "the writer and the watchdog agree on the name"
    for name in ("runtime-config.json", "runtime.json", "my-worker-config.json", "x-worker-abcdefgh.yaml"):
        assert not is_ephemeral_worker_config(tmp_path / name), name
    ephemeral.unlink()


# --- the interpreter is executed as given --------------------------------------


def _linked_interpreter(tmp_path: Path) -> Path:
    """The interpreter reached through a link, as a venv's bin/python is on POSIX.

    A Windows account without symlink privilege gets a directory junction to
    the interpreter's folder instead; ``Path.resolve`` follows both the same way.
    """
    from plugin_source import linked_interpreter

    link = linked_interpreter(tmp_path)
    if link is None:
        pytest.skip("no link to an interpreter can be created here")
    return link


def test_linked_interpreter_is_executable_as_given(tmp_path: Path):
    """An alias must launch before autostart can prove it preserves that path."""
    link = _linked_interpreter(tmp_path)
    probe = subprocess.run(
        [str(link), "-I", "-B", "-c", "import json,sys; print(json.dumps(sys.executable))"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert Path(json.loads(probe.stdout)).resolve() == Path(sys.executable).resolve()


def test_main_hands_run_the_interpreter_path_as_given(tmp_path: Path, monkeypatch):
    """A venv's bin/python is a symlink to the base interpreter.  Resolving it
    started the child outside the venv, where this package is not importable,
    and every wake died in 60 ms with nothing but ``worker_process_failed``."""
    link = _linked_interpreter(tmp_path)
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    seen: dict[str, Path] = {}

    def fake_run(
        config_path, python_executable, *, cleanup_config, after_pid=None, delay_seconds=0.0, upgrade_only=False
    ):
        seen["python"] = python_executable
        return 0

    monkeypatch.setattr(worker_watchdog, "run", fake_run)
    assert worker_watchdog.main(["--config", str(config), "--python", str(link)]) == 0
    assert seen["python"] == link
    assert seen["python"] != link.resolve()


def test_autostart_records_the_interpreter_path_as_given(tmp_path: Path):
    from scope_recall.maintenance import autostart

    link = _linked_interpreter(tmp_path)
    config_path, _ = _runtime_payload(tmp_path, drain_seconds=5.0)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    bound = Path(payload["binding"]["data_directory"]) / "runtime-config.json"
    bound.write_text(json.dumps(payload), encoding="utf-8")
    prepared = autostart.plan(bound, link, user_id="TEST-user")
    assert prepared["python_executable"] == str(link)
    assert str(link.resolve()) not in prepared["python_executable"]


def test_a_child_that_dies_before_its_receipt_names_the_reason(tmp_path: Path, monkeypatch, capsys):
    """The failure above left ``exit_code: 1`` and no cause anywhere: the child's
    stderr was read and dropped.  The last line of its traceback is the one
    line worth keeping, bounded, and it reaches the status file and the doctor."""
    from scope_recall.maintenance.doctor import run_doctor

    config_path, _ = _runtime_payload(tmp_path, drain_seconds=5.0)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["supervisor_enabled"] = False
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    data = Path(payload["binding"]["data_directory"])
    real_popen = subprocess.Popen

    def spawn(command, **kwargs):
        if len(command) > 2 and str(command[2]).endswith("_worker_bootstrap.py"):
            command = [
                command[0],
                "-c",
                "import sys; sys.stdin.read(1); raise ModuleNotFoundError(\"No module named 'scope_recall'\")",
            ]
        return real_popen(command, **kwargs)

    monkeypatch.setattr(worker_watchdog.subprocess, "Popen", spawn)
    assert worker_watchdog.run(config_path, Path(sys.executable), cleanup_config=False) == 1
    capsys.readouterr()
    status = json.loads((data / "runtime-worker-status.json").read_text(encoding="utf-8"))
    assert status["exit_code"] == 1 and status["capability_gaps"] == ["worker_process_failed"]
    assert status["worker_error"] == "ModuleNotFoundError: No module named 'scope_recall'"
    report = run_doctor(host="codex", instance_root=tmp_path / "install")
    assert report.worker_status["worker_error"] == status["worker_error"]
    assert "worker_last_exit_failed" in report.capability_gaps


def test_a_pass_that_yielded_to_another_writer_is_not_a_failed_exit(tmp_path: Path):
    """A pass that met another writer, or a held worker lock, exits 75 with status busy and the supervisor tries
    again after a pause: nothing failed, so the doctor does not call it ``worker_last_exit_failed``; any other
    non-zero exit still is one."""
    from scope_recall.maintenance.doctor import run_doctor
    from scope_recall.runtime.worker_entry import load_config, persist_worker_status

    config_path, _ = _runtime_payload(tmp_path, drain_seconds=5.0)
    config = load_config(config_path)
    for payload, exit_code, failed in (
        ({"status": "busy", "capability_gaps": ["worker_writer_busy"]}, 75, False),
        ({"status": "busy", "capability_gaps": ["worker_wait_timeout"]}, 75, False),
        ({"status": "degraded", "capability_gaps": ["worker_process_failed"]}, 1, True),
    ):
        persist_worker_status(
            config,
            {**payload, "installation_id": config.binding.installation_id},
            started_at="2026-09-30T08:00:00Z",
            exit_code=exit_code,
        )
        report = run_doctor(host="codex", instance_root=tmp_path / "install")
        assert report.worker_status["exit_code"] == exit_code
        assert ("worker_last_exit_failed" in report.capability_gaps) is failed, payload


def test_only_a_traceback_tail_is_kept_as_the_reason():
    assert (
        worker_launch.failure_reason("Traceback (most recent call last):\n  File x\nKeyError: 'k'\n") == "KeyError: 'k'"
    )
    assert worker_launch.failure_reason("just some chatter\n/some/path: not a reason\n") is None
    assert worker_launch.failure_reason("") is None
    assert worker_launch.failure_reason("RuntimeError: token sk-ant-api03-" + "A" * 40 + "\n") is None


class _Running:
    pid = 4321

    @staticmethod
    def poll():
        return None


def test_a_teardown_starts_taskkill_without_a_console_window(monkeypatch):
    """The worker's teardown runs ``taskkill`` from a process with no console; started without ``CREATE_NO_WINDOW``
    it would get a console window of its own, which would flash on the desktop at every pass."""
    calls = []
    monkeypatch.setattr(worker_launch.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    worker_launch.taskkill_tree(_Running())
    ((args, kwargs),) = calls
    assert args[0][:2] == ["taskkill", "/PID"]
    assert kwargs["creationflags"] == worker_launch.detached_creationflags()
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    assert kwargs["creationflags"] & no_window == no_window
