"""Exercise capture, processing and recall without adding test messages to the owner's store."""

from __future__ import annotations

import copy
import io
import json
import os
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from ..adapters.clients import CodexHookHandler
from ..adapters.hermes.audiences import LOCAL_USER_ID
from ..adapters.hermes.provider import ScopeRecallHermesAdapter
from ..contracts import ContractError
from ..core.recall_log_writer import close_writers
from ..core.writer_lease import TruthWriterBusyError
from ..runtime.instance import build_runtime_instance
from ..runtime.instance_config import RuntimeInstanceConfig
from ..runtime.worker_entry import credential_environment, run_worker
from .doctor import load_binding
from .install_common import HostChoice
from .install_hermes import profile_identity
from .shared import attach, fresh_installation, init_shared

MARKER = "amber-orchid"
STATEMENT = "The synthetic bluebird project marker is amber-orchid."
QUESTION = "What is the synthetic bluebird project marker?"
# Setup already allows ten seconds for shared-store locks. Give cold diagnostic
# files the same headroom instead of imposing the short runtime shutdown budget.
DIAGNOSTIC_CLOSE_SECONDS = 10.0
# A setup probe may confirm storage outside a live turn's short delivery window.
# Keep the same lock headroom as setup, without changing any host runtime budget.
READBACK_SECONDS = 10.0
# Cold delivery checks can exhaust their first short window. Two warmed host
# reads allow recovery while bounding work when the injection path is broken.
HOST_READ_ATTEMPTS = 3
_UNCONFIRMED = frozenset({"brief_safety_unconfirmed", "brief_delivery_unconfirmed", "injection_delivery_unconfirmed"})
# Unknown gap codes remain failures even when their names resemble deadline or
# confirmation codes. These are the explicit deadline exits of Core recall.
_READBACK_WAITING_GAPS = frozenset(
    {
        "deadline_exceeded",
        "deadline_exceeded_collect",
        "deadline_exceeded_compile",
        "deadline_exceeded_followup",
        "deadline_exceeded_hydrate",
        "deadline_exceeded_relation",
        "deadline_exceeded_release",
        "deadline_exceeded_release_fence",
        "deadline_exceeded_select",
        "deadline_exceeded_vector",
    }
)


def _host_answer(answer, reason):
    """Carry only fixed diagnostics when a host leaves delivery unconfirmed."""
    if MARKER not in json.dumps(answer, ensure_ascii=False) and reason in _UNCONFIRMED:
        return dict(unconfirmed=reason)
    return answer


def _client(home: Path, host: str, session: str, prompt: str) -> dict:
    with closing(CodexHookHandler.from_home(str(home), host)) as handler:
        answer = handler.handle_payload(
            dict(
                hook_event_name="UserPromptSubmit",
                session_id=session,
                turn_id=session + "-turn",
                prompt_id=session + "-prompt",
                generation_id=session,
                cwd=str(home.parent),
                prompt=prompt,
            )
        )
        return _host_answer(answer, handler.diagnostics.delivery_reason)


def _hermes(home: Path, session: str, prompt: str, *, capture: bool, agent_identity: str):
    provider = ScopeRecallHermesAdapter()
    try:
        provider.initialize(
            session,
            hermes_home=str(home),
            platform="cli",
            agent_context="primary",
            agent_identity=agent_identity,
            agent_workspace="hermes",
            user_id=LOCAL_USER_ID,
            parent_session_id="",
        )
        if capture:
            provider.on_turn_start(1, prompt, turn_id=session + "-turn", session_id=session)
            provider.observe_pre_llm(session_id=session, turn_id=session + "-turn", user_message=prompt)
            provider.sync_turn(prompt, "Recorded.", session_id=session)
            return ""
        answer = provider.prefetch(prompt)
        return _host_answer(answer, provider.diagnostics.delivery_reason)
    finally:
        provider.shutdown()


def _clean_owned_check(root: Path) -> dict:
    """Retire this probe's writers and report removal or retention without changing readiness."""
    # Host shutdown can detach its runtime. Retire only this probe's diagnostics
    # before enumerating files so a late WAL or health receipt cannot race removal.
    retained = dict(status="retained", directory=str(root), reason="verification diagnostics still active")
    try:
        if not close_writers(root, timeout=DIAGNOSTIC_CLOSE_SECONDS):
            return retained
        paths = list(root.rglob("*"))
        if any(path.is_symlink() for path in paths):
            return dict(retained, reason="verification directory contains a link; retain it for inspection")
        for path in paths:
            if path.is_file():
                path.unlink()
        for path in sorted((path for path in paths if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
            path.rmdir()
        root.rmdir()
    except (OSError, ValueError) as exc:
        # Filesystem exception text can contain caller data; keep a fixed reason and type.
        return dict(retained, reason="verification directory cleanup failed", error_type=type(exc).__name__)
    return dict(status="removed", directory=str(root))


@contextmanager
def temporary_credentials(config: RuntimeInstanceConfig, env_file: str | None):
    """Apply only declared credential references during verification and restore the caller's environment."""
    values = credential_environment(config, env_file) if env_file else {}
    original = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            os.environ.setdefault(name, value)
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _readback(config: RuntimeInstanceConfig, read_host) -> dict:
    """Require the host to deliver evidence; independent lookup never proves readiness.

    Every attempt uses the host's own deadlines and safety checks. Only explicit
    unconfirmed results permit another attempt; ordinary empty responses fail.
    """
    reasons = []
    for _ in range(HOST_READ_ATTEMPTS):
        try:
            answer = read_host()
        except (ContractError, TruthWriterBusyError, OSError, sqlite3.Error) as exc:
            failure = _probe_error(exc, "cross_session_recall")
            reasons.append(failure["reason"])
            return dict(failure, method="host_adapter", host_attempts=len(reasons), host_reasons=reasons)
        if MARKER in json.dumps(answer, ensure_ascii=False):
            reasons.append("marker_recalled")
            return dict(status="passed", method="host_adapter", host_attempts=len(reasons), host_reasons=reasons)
        reason = answer.get("unconfirmed") if isinstance(answer, dict) else None
        reasons.append(reason if reason in _UNCONFIRMED else "marker_not_recalled")
        if reason not in _UNCONFIRMED:
            return dict(
                status="failed",
                method="host_adapter",
                check="cross_session_recall",
                reason="marker_not_recalled",
                host_attempts=len(reasons),
                host_reasons=reasons,
            )
    detail = dict(
        method="independent_readback", host_delivery=reasons[-1], host_attempts=len(reasons), host_reasons=reasons
    )
    try:
        with closing(build_runtime_instance(config)) as runtime:
            with runtime.core.storage.read(config.context(), remaining_seconds=READBACK_SECONDS) as tx:
                stored = tx._check().execute("SELECT 1 FROM source_events WHERE content=?", (STATEMENT,)).fetchone()
                pending = tx._check().execute("SELECT 1 FROM capture_inbox LIMIT 1").fetchone()
            if not stored:
                return dict(
                    detail,
                    status="incomplete" if pending else "failed",
                    check="capture",
                    reason="capture_pending" if pending else "capture_missing",
                )
            packet = runtime.core.recall(
                config.context(),
                dict(
                    protocol_version="1.1",
                    request_id="setup-readback",
                    query=QUESTION,
                    mode="current",
                    max_items=6,
                    budget_tokens=1200,
                ),
                deadline_seconds=READBACK_SECONDS,
            )
            if MARKER in str(packet.items):
                return dict(
                    detail,
                    status="incomplete",
                    check="cross_session_recall",
                    reason="host_delivery_unconfirmed",
                    independent_recall=True,
                )
            waiting = any(gap in _READBACK_WAITING_GAPS for gap in packet.gaps)
            return dict(
                detail,
                status="incomplete" if waiting else "failed",
                check="cross_session_recall",
                reason="readback_unconfirmed" if waiting else "marker_not_recalled",
            )
    except (ContractError, TruthWriterBusyError, OSError, sqlite3.Error) as exc:
        return dict(detail, **_probe_error(exc, "cross_session_recall"))


def _probe_error(exc, check):
    """Separate an unfinished operation from a concrete refusal without exposing exception text."""
    waiting = (
        isinstance(exc, (TruthWriterBusyError, TimeoutError))
        or isinstance(exc, ContractError)
        and exc.code == "DEADLINE_EXCEEDED"
        or isinstance(exc, sqlite3.Error)
        and getattr(exc, "sqlite_errorcode", 0) & 255
        in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_INTERRUPT}
    )
    return dict(
        status="incomplete" if waiting else "failed",
        check=check,
        reason="operation_unconfirmed" if waiting else "operation_error",
        error_type=type(exc).__name__,
        **{
            name: getattr(exc, name)
            for name in ("errno", "winerror")
            if isinstance(exc, OSError) and type(getattr(exc, name, None)) is int
        },
    )


def verify_memory(parent: Path, host: HostChoice, source: dict, *, env_file: str | None = None) -> dict:
    """Probe an independent synthetic store and report its readiness and cleanup separately."""
    from .setup_configuration import runtime_payload

    if host == "hermes":
        from ..adapters.hermes.identity import bind_hermes_identity

        # Check the selected entry with the real host identity before probing a disposable store.
        bind_hermes_identity(
            "setup-check",
            hermes_home=str(parent),
            platform="cli",
            user_id=LOCAL_USER_ID,
            agent_identity=profile_identity(parent),
            agent_workspace="hermes",
        )

    probe_parent = parent.parent
    # Named profiles sit inside another home; disposable stores must be outside every host home.
    for ancestor in parent.parents:
        if (ancestor / "scope-recall").is_dir() or (ancestor / "codex-installation.json").is_file():
            probe_parent = ancestor.parent
    root = probe_parent / ("setup-check-" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    result: dict = {}
    stage = "probe_store"
    try:
        store, home = root / "store", root / "host"
        init_shared(root=store, agent_id="setup-check", test_mode=True)
        identity = profile_identity(parent)
        grants = (
            fresh_installation(home, {"agent_id": "setup-check", "test_mode": True}, identity)
            if host == "hermes"
            else None
        )
        attach(
            host=host,
            instance_root=home,
            root=store,
            entry_id="setup-check",
            display_name="Setup check",
            grants_from=grants,
        )
        stage = "capture"
        if host == "hermes":
            _hermes(home, "setup-first", STATEMENT, capture=True, agent_identity=identity)
        else:
            _client(home, host, "setup-first", STATEMENT)
        binding, _directory = load_binding(host, home)
        models = {
            key: copy.deepcopy(source[key]) for key in ("auxiliary", "vector", "vector_threshold") if key in source
        }
        raw = runtime_payload(binding, models=models, worker="manual")
        # A non-default name keeps host hooks from launching another worker during the probe.
        selected = RuntimeInstanceConfig.from_mapping(source)
        if selected.auxiliary and selected.auxiliary.ledger_path:
            # Verification spends against the selected ledger, including its existing usage.
            # Only the synthetic truth and vector stores are disposable.
            raw["auxiliary"]["ledger_path"] = str(selected.auxiliary.ledger_path)
        config = RuntimeInstanceConfig.from_mapping(raw)
        config_path = root / "check-worker.json"
        config_path.write_text(json.dumps(raw), encoding="utf-8")
        with temporary_credentials(config, env_file):
            stream = io.StringIO()
            stage = "worker"
            code = run_worker(str(config_path), output=stream)
            worker = json.loads(stream.getvalue())
            stage = "cross_session_recall"
            readback = _readback(
                config,
                lambda: (
                    _hermes(home, "setup-second", QUESTION, capture=False, agent_identity=identity)
                    if host == "hermes"
                    else _client(home, host, "setup-second", QUESTION)
                ),
            )
            recalled = readback["status"] == "passed"
            auxiliary = raw.get("auxiliary", {})
            configured = auxiliary.get("external_embedding") and auxiliary.get("external_consolidation")
            model_recall = False
            if configured:
                stage = "models"
                with closing(build_runtime_instance(config)) as runtime:
                    packet = runtime.recall(
                        dict(
                            protocol_version="1.1",
                            request_id="setup-second",
                            query=QUESTION,
                            mode="current",
                            max_items=6,
                            budget_tokens=1200,
                        )
                    )
                    model_recall = MARKER in str(packet.items) and not packet.gaps
            full = bool(
                configured
                and model_recall
                and code == 0
                and not worker.get("capability_gaps")
                and worker.get("processed", 0)
            )
            result.update(
                capture=readback.get("check") != "capture",
                worker_exit=code,
                worker=worker,
                cross_session_recall=recalled,
                readback=readback,
                models_verified=full,
                isolation="independent synthetic store; no test messages enter the selected memory store",
            )
            return result
    except (ContractError, TruthWriterBusyError, OSError, sqlite3.Error) as exc:
        result.update(
            capture=False, cross_session_recall=False, models_verified=False, readback=_probe_error(exc, stage)
        )
        return result
    finally:
        result["cleanup"] = _clean_owned_check(root)
