"""Exercise two real Hermes homes over loopback, using the wheel built by this job.

Run after hermes_catalog_install.sh. The catalog home keeps its local setup; a second
home uses the public remote installer. All model requests use loopback_chat_model.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

CLIENT = "scope_recall.adapters.codex.remote_client"
SERVER = "scope_recall.adapters.codex.remote_server"
CLI = "scope_recall.maintenance.cli"


class Exercise:
    """Own private process groups and diagnostics for commands started by this exercise."""

    def __init__(self, work: Path) -> None:
        self.work = work
        self.children: list[subprocess.Popen] = []
        self.serial = 0
        self.groups: set[int] = set()

    def start(self, argv: list[str], env: dict) -> subprocess.Popen:
        """Start one owned child with durable output and no inherited interactive input."""
        self.serial += 1
        path = self.work / f"command-{self.serial:02d}.log"
        print(f"command {self.serial}: {argv}", flush=True)
        with path.open("wb") as output:
            child = subprocess.Popen(
                argv, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
        self.children.append(child)
        self.groups.add(child.pid)
        return child

    def run(self, argv: list[str], env: dict, *, timeout: int = 180, check: bool = True) -> str:
        """Run a bounded command and retain its complete output on failure."""
        child = self.start(argv, env)
        path = self.work / f"command-{self.serial:02d}.log"
        try:
            code = child.wait(timeout)
        except subprocess.TimeoutExpired:
            self.stop(child)
            raise AssertionError(f"command timed out after {timeout}s: {argv}; see {path}") from None
        self.stop(child)
        output = path.read_text(encoding="utf-8", errors="replace")
        if code and check:
            raise AssertionError(f"command exited {code}: {argv}\n{output[-12000:]}")
        return output

    def stop(self, child: subprocess.Popen) -> None:
        """Reap the direct child and terminate only its recorded private process group."""
        if child.pid not in self.groups:
            return
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(15)
        except subprocess.TimeoutExpired:
            pass
        # A launcher can exit before its package-manager children. Its process group remains ours.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(10)
        self.groups.remove(child.pid)

    def close(self) -> None:
        """Stop every remaining owned child, including after a failed assertion."""
        for child in reversed(self.children):
            self.stop(child)


def free_port() -> int:
    """Choose an unused loopback port; readiness also checks that our child stayed alive."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def ready(child: subprocess.Popen, url: str, *, token: str = "") -> None:
    """Probe an owned loopback service with bounded attempts and per-request timeouts."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(100):
        assert child.poll() is None, f"child {child.pid} exited before readiness: {url}"
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with opener.open(request, timeout=0.3) as answer:
                if answer.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise AssertionError(f"service not ready: {url}")


def recalled(log: Path, question: str, fact: str) -> None:
    """Prove a fresh session sent the earlier fact to the model inside memory-context."""
    contexts = []
    for line in log.read_text(encoding="utf-8").splitlines():
        body = json.loads(line)["body"]
        users = [row for row in body.get("messages", []) if row.get("role") == "user"]
        if users and question in json.dumps(users[-1]):
            contexts.extend(re.findall(r"<memory-context>[\s\S]*?</memory-context>", json.dumps(body)))
    assert any(fact in context for context in contexts), f"missing recalled fact {fact!r}; see {log}"


def count(root: Path, fact: str) -> int:
    """Count the exact synthetic human source through a read-only SQLite connection."""
    with sqlite3.connect(f"file:{(root / 'memory.sqlite3').as_posix()}?mode=ro", uri=True) as connection:
        return connection.execute(
            "SELECT count(*) FROM source_events WHERE role='user' AND content=?", (fact,)
        ).fetchone()[0]


def wheel_version(wheel: Path) -> str:
    """The version this job built, read from the wheel's own metadata."""
    with zipfile.ZipFile(wheel) as archive:
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        return next(
            line.removeprefix("Version: ")
            for line in archive.read(metadata).decode().splitlines()
            if line.startswith("Version: ")
        )


def verify_core(job: Exercise, command: list[str], env: dict, home: Path, wheel: Path) -> None:
    """Assert the version and implementation bytes imported by the real host, before chatting."""
    expected = wheel_version(wheel)
    with zipfile.ZipFile(wheel) as archive:
        expected_hash = hashlib.sha256(archive.read("scope_recall/adapters/hermes/remote.py")).hexdigest()
    marker = job.work / (home.name + "-runtime.json")
    probe = (
        "\nfrom importlib.metadata import version as _core_version\n"
        "from pathlib import Path as _ProbePath\nimport hashlib as _probe_hash\nimport json as _probe_json\n"
        "from scope_recall.adapters.hermes import remote as _remote_impl\n"
        f"assert _core_version('hermes-scope-recall') == {expected!r}, 'wrong Hermes core version'\n"
        "assert _probe_hash.sha256(_ProbePath(_remote_impl.__file__).read_bytes()).hexdigest() == "
        f"{expected_hash!r}, 'wrong Hermes core build'\n"
        f"_ProbePath({str(marker)!r}).write_text(_probe_json.dumps("
        "{'version': _core_version('hermes-scope-recall'), 'module': _remote_impl.__file__}), encoding='utf-8')\n"
    )
    wrapper = home / "plugins/scope-recall/__init__.py"
    wrapper.write_text(wrapper.read_text(encoding="utf-8") + probe, encoding="utf-8")
    job.run([*command, "memory", "status"], env, timeout=300)
    assert marker.is_file(), "Hermes did not load the selected provider"
    assert json.loads(marker.read_text(encoding="utf-8"))["version"] == expected


def prepare_catalog(job: Exercise, env: dict) -> tuple:
    """Install this wheel's control environment and update the isolated catalog declaration."""
    runner = Path(env["RUNNER_TEMP"])
    catalog = runner / "hermes-catalog-install/home"
    local = Path(env.get("HERMES_HOME") or catalog / ".hermes")
    launcher = next(path for path in (catalog / ".local/bin/hermes", local / "bin/hermes") if path.is_file())
    wheels = Path(env["WHEELS"]).resolve()
    wheel = next(wheels.glob("hermes_scope_recall-*.whl"))
    control = job.work / "control"
    job.run([sys.executable, "-m", "venv", str(control)], env)
    python = str(control / "bin/python")
    job.run([python, "-m", "pip", "install", f"{wheel}[codex]"], env, timeout=600)
    local_env = {**env, "HERMES_HOME": str(local), "UV_FIND_LINKS": str(wheels), "LOOPBACK_KEY": "TEST-loopback-key"}
    # The catalog deliberately pins the released core. This isolated fixture pins the just-built wheel in the same
    # manifest, so the real host's package manager tests this change rather than the release. The declaration keeps
    # the catalog's form: Hermes does not rebuild a plugin whose declaration moves to an authored pyproject.toml.
    manifest = local / "plugins/scope-recall/plugin.yaml"
    pinned, pins = re.subn(
        r'"hermes-scope-recall\[vectors\]==[^"]+"',
        f'"hermes-scope-recall[vectors]=={wheel_version(wheel)}"',
        manifest.read_text(encoding="utf-8"),
    )
    assert pins == 1, f"the catalog manifest does not pin one core package: {manifest}"
    manifest.write_text(pinned, encoding="utf-8")
    return python, local, launcher, wheel, local_env


def configure_model(job: Exercise, command: list[str], env: dict, port: int) -> None:
    """Point one Hermes profile at this run's loopback model."""
    provider = [
        {
            "name": "loopback",
            "base_url": f"http://127.0.0.1:{port}/v1",
            "key_env": "LOOPBACK_KEY",
            "api_mode": "chat_completions",
            "model": "loopback-model",
            "models": {"loopback-model": {"context_length": 128000}},
        }
    ]
    for key, value in (
        ("custom_providers", json.dumps(provider)),
        ("model.default", "loopback-model"),
        ("model.provider", "custom:loopback"),
        ("model.base_url", f"http://127.0.0.1:{port}/v1"),
    ):
        job.run([*command, "config", "set", key, value], env, timeout=600)


def exercise(job: Exercise, env: dict) -> None:
    """Install via public commands and prove bidirectional recall plus durable outage replay."""
    python, local, launcher, wheel, local_env = prepare_catalog(job, env)
    pointer = json.loads((local / "scope-recall/attachment.json").read_text(encoding="utf-8"))
    root = Path(pointer["root"])
    local_entry = pointer["entry_id"]
    profile = "remote-check"
    local_command = [str(launcher)]
    remote_command = [str(launcher), "-p", profile]
    # A one-shot `hermes chat` leaves an idle backend that keeps the configuration and dependency environment it
    # started with, and the catalog script's chats leave one. Stopping it when none runs is not an error.
    job.run([*local_command, "gateway", "stop"], local_env, timeout=120, check=False)
    # Enabling an enabled plugin changes nothing; re-admitting it makes the package manager build the new pin.
    job.run([*local_command, "plugins", "disable", "scope-recall"], local_env, timeout=600)
    job.run([*local_command, "plugins", "enable", "scope-recall"], local_env, timeout=900)
    requests = job.work / "requests.jsonl"
    model_port, server_port = free_port(), free_port()
    assert model_port != server_port
    model = job.start(
        [sys.executable, str(Path(__file__).with_name("loopback_chat_model.py")), str(model_port), str(requests)], env
    )
    ready(model, f"http://127.0.0.1:{model_port}/v1/models")
    configure_model(job, local_command, local_env, model_port)
    # A new profile starts from the current model settings, so it is created once they name this run's model.
    job.run([*local_command, "profile", "create", profile, "--no-alias", "--no-skills"], local_env, timeout=600)
    remote = local / "profiles" / profile
    assert remote.is_dir()
    configure_model(job, remote_command, local_env, model_port)
    server_home = job.work / "server-entry"
    job.run(
        [
            python,
            "-m",
            CLI,
            "attach",
            "--host",
            "hermes",
            "--remote-profile",
            profile,
            "--instance-root",
            str(server_home),
            "--root",
            str(root),
            "--entry",
            "remote-hermes",
            "--display-name",
            "Remote Hermes",
            "--grants-like",
            "all",
            "--capture-like",
            local_entry,
        ],
        env,
    )
    config = job.work / "client.json"
    config.write_text(
        json.dumps(
            {
                "host": "hermes",
                "url": f"http://127.0.0.1:{server_port}",
                "token_file": str(job.work / "token"),
                "state_dir": str(job.work / "state"),
            }
        ),
        encoding="utf-8",
    )
    digest = json.loads(job.run([python, "-m", CLIENT, "token", "--config", str(config)], env))["token_sha256"]
    job.run(
        [
            python,
            "-m",
            SERVER,
            "configure",
            "--host",
            "hermes",
            "--home",
            str(server_home),
            "--listen",
            "127.0.0.1",
            "--port",
            str(server_port),
            "--token-sha256",
            digest,
        ],
        env,
    )
    job.run([python, "-m", CLIENT, "install", "--config", str(config), "--plugin-dir", str(remote)], env)
    for command in (local_command, remote_command):
        job.run([*command, "gateway", "stop"], local_env, timeout=120, check=False)
    verify_core(job, local_command, local_env, local, wheel)
    verify_core(job, remote_command, local_env, remote, wheel)
    server_args = [python, "-m", SERVER, "serve", "--host", "hermes", "--home", str(server_home)]
    server = job.start(server_args, env)
    token = (job.work / "token").read_text(encoding="utf-8").strip()
    ready(server, f"http://127.0.0.1:{server_port}/health", token=token)
    remote_fact = "The zircon lighthouse uses violet paint."
    local_fact = "The copper observatory opens on Tuesday."
    for sender, receiver, fact, question in (
        (remote_command, local_command, remote_fact, "What paint does the zircon lighthouse use?"),
        (local_command, remote_command, local_fact, "When does the copper observatory open?"),
    ):
        job.run([*sender, "chat", "-q", fact, "-Q"], local_env)
        job.run([*receiver, "chat", "-q", question, "-Q"], local_env)
        recalled(requests, question, fact)
    job.stop(server)
    offline_fact = "The amber harbor signal is seven green flashes."
    started = time.monotonic()
    job.run([*remote_command, "chat", "-q", offline_fact, "-Q"], local_env, timeout=90)
    assert time.monotonic() - started < 90
    assert any(
        offline_fact in json.dumps(json.loads(line)["body"])
        for line in requests.read_text(encoding="utf-8").splitlines()
    ), "offline turn never reached the model"
    assert list((job.work / "state/spool").glob("*.json")), "offline capture was not retained"
    assert count(root, offline_fact) == 0
    server = job.start(server_args, env)
    ready(server, f"http://127.0.0.1:{server_port}/health", token=token)
    for _ in range(2):
        status = json.loads(job.run([python, "-m", CLIENT, "flush", "--config", str(config)], env))
        assert status["queued_turns"] == 0, status
    assert count(root, offline_fact) == 1, "outage replay lost or duplicated the source"
    question = "What is the amber harbor signal?"
    job.run([*local_command, "chat", "-q", question, "-Q"], local_env)
    recalled(requests, question, offline_fact)
    print("real Hermes: bidirectional recall, offline chat and exactly-once source replay passed", flush=True)


def main() -> None:
    """Keep runner diagnostics and always reap owned services, without changing the user's HOME."""
    work = Path(os.environ["RUNNER_TEMP"]) / "hermes-remote-install"
    work.mkdir(exist_ok=False)
    job = Exercise(work)
    try:
        exercise(job, dict(os.environ))
    except BaseException:
        logs = [
            *sorted(work.glob("command-*.log")),
            work / "server-entry/scope-recall/remote-server.log",
            work / "state/remote-client.log",
        ]
        for path in (path for path in logs if path.is_file()):
            print(f"\n{path.name}:\n{path.read_text(encoding='utf-8', errors='replace')[-8000:]}", file=sys.stderr)
        raise
    finally:
        job.close()
        pids = [child.pid for child in job.children]
        print(f"diagnostics: {work}; private process groups and child PIDs reaped: {pids}", flush=True)


if __name__ == "__main__":
    main()
