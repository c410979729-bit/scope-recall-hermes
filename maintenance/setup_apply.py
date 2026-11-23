"""Execute an approved setup plan through existing binding and install boundaries."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from scope_recall.adapters.hermes.shared_entries import attachment_path, read_shared_payload
from scope_recall.contracts import ENTRY_ID
from scope_recall.core.schema import SCHEMA_VERSION
from scope_recall.runtime.instance_config import RuntimeInstanceConfig
from scope_recall.runtime.model_budget import initialize_auxiliary_budget_ledger

from .backup import atomic_write, sha256
from .doctor import load_binding, run_doctor
from .hermes_config import provider_configuration
from .install import apply_install
from .install_common import InstallError
from .install_hermes import profile_identity
from .install_receipt import setup_receipts_directory
from .setup_activation import register_codex
from .setup_bindings import repair_profile
from .setup_configuration import runtime_payload, write_configuration
from .setup_packages import install_package
from .setup_preflight import check_plan, installation_plan
from .setup_projects import register_project_copy
from .shared import attach, fresh_installation, init_shared, store_binding
from .shared_run import SharedStoreError


def _runtime(row: dict, plan, backups: list[str]) -> Path:
    home = Path(row["home"])
    binding, directory = load_binding(row["host"], home)
    path = (
        attachment_path(home).parent / "runtime-config.json"
        if binding.installation_kind == "shared"
        else directory / "runtime-config.json"
    )
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    raw = runtime_payload(
        binding,
        models=plan.models,
        host=row["host"],
        existing=existing,
        worker=row["worker"] if plan.worker_explicit or existing is None else None,
    )
    config = RuntimeInstanceConfig.from_mapping(raw)
    write_configuration(path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n", backups)
    if config.auxiliary is not None and config.auxiliary.ledger_path is not None:
        initialize_auxiliary_budget_ledger(config.auxiliary.ledger_path, config.auxiliary.budget)
    return path


def hermes_entry_name(home: Path) -> str:
    """The entry name a Hermes home attaches with: ``hermes-<profile>`` for a named profile, else ``hermes``.

    Recalled items carry the name of the entry they came in through, so a named profile's name is what lets another
    agent say which one heard something. The default profile, a custom home and a profile name that cannot form an
    entry id (lowercase letters, digits and hyphens, at most 32 characters) keep the plain host name.
    """
    identity = profile_identity(home)
    candidate = "hermes-" + identity.replace("_", "-")
    return candidate if identity not in {"default", "custom"} and ENTRY_ID.fullmatch(candidate) else "hermes"


def _attach(row: dict, plan) -> None:
    if row["state"] != "fresh":
        if row["host"] == "hermes":
            row["identity_repair"] = repair_profile(Path(row["home"]))
        return
    root = plan.store
    if not (root / "installation.json").is_file():
        init_shared(root=root, agent_id="scope-recall", test_mode=plan.test_mode)
    payload = read_shared_payload(root)
    entry = hermes_entry_name(Path(row["home"])) if row["host"] == "hermes" else row["host"]
    taken = {item["entry_id"] for item in payload["entries"]}
    if row["host"] == "hermes" and entry in taken:
        import hashlib

        entry = "hermes-" + hashlib.sha256(str(Path(row["home"]).resolve()).encode()).hexdigest()[:16]
    if entry in taken:
        raise InstallError("entry name already belongs to another home; choose that home or detach explicitly")
    source = row["scopes"]["source_entry"]
    kwargs: dict[str, Any] = (
        dict(grants_like=(source,), capture_like=source) if source and row["host"] != "hermes" else {}
    )
    if row["host"] == "hermes":
        kwargs["grants_from"] = fresh_installation(
            Path(row["home"]),
            payload,
            profile_identity(Path(row["home"])),
            local_platforms=row["scopes"]["local_platforms"],
        )
    routes = root / "runtime-config.json"
    attach(
        host=row["host"],
        instance_root=Path(row["home"]),
        root=root,
        entry_id=entry,
        display_name=entry,
        runtime_config_from=routes if routes.is_file() else None,
        python_executable=Path(row["python"]),
        **kwargs,
    )


def _activate_hermes(home: Path, backups: list[str]) -> None:
    path = home / "config.yaml"
    write_configuration(path, provider_configuration(path), backups)


def _worker(plan, config_path: Path, python: str, backups: list[str], env_file: str | None = None) -> dict:
    command = [python, "-I", "-B", "-m", "scope_recall.runtime.worker_entry", "--config", str(config_path)]
    if env_file:
        command += ["--env-file", env_file]
    mode = (
        "on-demand" if json.loads(config_path.read_text(encoding="utf-8")).get("supervisor_enabled", True) else "manual"
    )
    value = dict(
        mode=mode,
        command=command,
        system_registration=False,
        explanation="Host hooks wake bounded worker passes."
        if mode == "on-demand"
        else "Run this command to process queued memory.",
    )
    write_configuration(config_path.parent / "worker-command.json", json.dumps(value, indent=2) + "\n", backups)
    return value


def _install(row: dict, plan, backups: list[str]) -> dict:
    home = Path(row["home"])
    install = installation_plan(row, plan)
    result = apply_install(install)
    backups.extend(result.backups)
    if row["host"] == "hermes":
        _activate_hermes(home, backups)
    elif row["host"] == "codex":
        register_codex(home, Path(row["target"]), plan.user_root, backups)
    return result.to_dict()


def _store_usable(report: dict) -> bool:
    """Require the selected truth store to pass its own checks before considering the isolated probe."""
    return bool(
        report["binding_ok"]
        and report["package_ok"]
        and report["database_present"]
        and report["sources"] is not None
        and report["schema_version"] == SCHEMA_VERSION
        and not {
            "capture_authorization_invalid",
            "hermes_runtime_package_missing",
            "hermes_runtime_package_mismatch",
        }.intersection(report["capability_gaps"])
    )


def _full_store_usable(report: dict) -> bool:
    """A successful disposable probe cannot repair errors in the selected store's derived surfaces."""
    return _store_usable(report) and not any(
        name.startswith(("vector_", "embedding_", "hermes_runtime_"))
        or name in {"storage_budget_exceeded", "dependency_drift"}
        for name in report["capability_gaps"]
    )


def _register_copy(item: dict, backups: list[str]) -> None:
    """Track this registration's exact prior file so a later registration has an independent rollback."""
    backup = register_project_copy(Path(item["home"]), {key: value for key, value in item.items() if key != "home"})
    backups.append(str(backup))


def _write_setup_receipt(path: Path, value: dict[str, Any], backups: list[str]) -> dict[str, Any]:
    """Journal paths and digests without including any original configuration bytes."""
    value["backup_hashes"] = {name: sha256(Path(name)) for name in backups if Path(name).is_file()}
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value


def _verification_result(results: list[dict], ready: bool) -> dict:
    """Name unsuccessful checks with fixed codes, never host text or credentials."""
    findings = []
    for item in results:
        check = item["verification"]
        if not _store_usable(item["doctor"]):
            findings.append(
                dict(
                    host=item["host"],
                    check="selected_installation",
                    status="failed",
                    reason="doctor_failed",
                    capability_gaps=item["doctor"]["capability_gaps"],
                )
            )
        elif not check.get("cross_session_recall"):
            readback = check.get("readback", {})
            findings.append(
                dict(
                    host=item["host"],
                    check=readback.get("check", "cross_session_recall"),
                    status=readback.get("status", "failed"),
                    reason=readback.get("reason", "marker_not_recalled"),
                )
            )
    incomplete = bool(findings) and all(item["status"] == "incomplete" for item in findings)
    return dict(
        status="ready" if ready else "verification_incomplete" if incomplete else "verification_failed",
        verification_findings=findings,
        reason="Installation applied; verification is not yet confirmed. Rerun setup to confirm readiness."
        if incomplete
        else None,
    )


def apply_setup(plan, *, register_copies: bool = False) -> dict[str, Any]:
    """Apply one approved plan and preserve completed stages and rollback paths on failure."""
    from .setup_verify import verify_memory

    if plan.blockers:
        raise InstallError("setup plan is blocked; resolve the listed findings before applying")
    conflicts = check_plan(plan)
    if conflicts:
        return dict(status="blocked", failed_step="preflight", completed=[], conflicts=conflicts)
    completed, backups, results = [], [], []
    receipt_home = setup_receipts_directory(Path(plan.hosts[0]["home"]))
    receipt = receipt_home / (uuid.uuid4().hex + ".json")
    stage = "binding"
    value: dict[str, Any]
    try:
        if register_copies:
            for item in plan.project_copies:
                stage = "project-registration"
                _register_copy(item, backups)
                completed.append(stage)
        for row in plan.hosts:
            stage = row["host"] + ":package"
            install_package(row.get("package_command", []), Path(row["python"]))
            if row.get("package_command"):
                completed.append(stage)
            stage = row["host"] + ":binding"
            _attach(row, plan)
            completed.append(stage)
            stage = row["host"] + ":configuration"
            config_path = _runtime(row, plan, backups)
            completed.append(stage)
            stage = row["host"] + ":install"
            installed = _install(row, plan, backups)
            completed.append(stage)
            worker_path = config_path
            binding, _directory = load_binding(row["host"], Path(row["home"]))
            if binding.installation_kind == "shared":
                root = binding.data_directory
                worker_path = root / "runtime-config.json"
                payload = read_shared_payload(root)
                shared_binding = store_binding(payload, root, frozenset(payload["scope_ids"]))
                if not worker_path.is_file() or plan.models is not None:
                    existing = json.loads(worker_path.read_text("utf-8")) if worker_path.is_file() else None
                    raw = runtime_payload(
                        shared_binding,
                        models=plan.models,
                        existing=existing,
                        worker=plan.worker if plan.worker_explicit or existing is None else None,
                    )
                    write_configuration(worker_path, json.dumps(raw, indent=2) + "\n", backups)
                elif plan.worker_explicit:
                    raw = json.loads(worker_path.read_text(encoding="utf-8"))
                    raw["supervisor_enabled"] = plan.worker == "on-demand"
                    RuntimeInstanceConfig.from_mapping(raw)
                    write_configuration(worker_path, json.dumps(raw, indent=2) + "\n", backups)
            worker = _worker(plan, worker_path, row["python"], backups, row.get("env_file"))
            stage = row["host"] + ":doctor"
            doctor = run_doctor(
                host=row["host"], instance_root=Path(row["home"]), python_executable=Path(row["python"])
            )
            completed.append(stage)
            stage = row["host"] + ":verification"
            configured = json.loads(worker_path.read_text(encoding="utf-8"))
            check = (
                verify_memory(Path(row["home"]), row["host"], configured, env_file=row.get("env_file"))
                if _store_usable(doctor.to_dict())
                else dict(
                    cross_session_recall=False, models_verified=False, skipped="selected installation failed doctor"
                )
            )
            completed.append(stage)
            results.append(
                dict(
                    host=row["host"],
                    installation=installed,
                    worker=worker,
                    doctor=doctor.to_dict(),
                    verification=check,
                    identity_repair=row.get("identity_repair"),
                )
            )
        full = bool(results) and all(
            item["verification"].get("models_verified") and _full_store_usable(item["doctor"]) for item in results
        )
        ready = all(
            item["verification"].get("cross_session_recall") and _store_usable(item["doctor"]) for item in results
        )
        value = dict(
            **_verification_result(results, ready),
            state=("完整记忆功能可用" if full else "原文检索可用") if ready else None,
            missing=[]
            if full
            else sorted({gap for item in results for gap in item["doctor"]["capability_gaps"]})
            + [
                (
                    "Configure embedding and extraction routes, credentials and a nonzero budget; "
                    "rerun setup to verify them."
                )
            ],
            completed=completed,
            hosts=results,
            backups=backups,
            receipt=str(receipt),
            host_activation="Start a new host session; approve plugin hooks or MCP when the host requests it.",
        )
    except Exception as exc:  # noqa: BLE001 -- external failures must produce a sanitized recovery receipt.
        # External exception text may carry credentials. Only installer refusals,
        # the known stage and numeric operating-system error codes belong in receipts.
        value = dict(
            status="failed",
            failed_step=stage,
            error_type=type(exc).__name__,
            reason=str(exc) if isinstance(exc, (InstallError, SharedStoreError)) else None,
            **{
                name: getattr(exc, name)
                for name in ("errno", "winerror")
                if isinstance(exc, OSError) and type(getattr(exc, name, None)) is int
            },
            completed=completed,
            backups=backups,
            receipt=str(receipt),
            retry="Rerun the original setup command; completed bindings and truth databases are reused.",
            rollback=(
                "Use the install receipt with apply-uninstall to remove wrappers while retaining memory; "
                "restore the listed configuration backups if needed."
            ),
        )
    return _write_setup_receipt(receipt, value, backups)
