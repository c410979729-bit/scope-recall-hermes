"""Setup readiness survives disposable diagnostics cleanup without hiding retained data."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Timer

import pytest
from scope_recall.core.recall_log_storage import RecallLogStore
from scope_recall.core.recall_log_writer import close_writers
from scope_recall.maintenance import setup_apply, setup_verify
from scope_recall.maintenance.install_common import InstallError
from scope_recall.maintenance.setup import build_plan, parser

from tests.packaging.test_setup import fake_user  # noqa: F401


def _plan(user):
    """Select one synthetic client without model routes or an automatic worker."""
    (user / ".codex").mkdir()
    return build_plan(parser().parse_args(["--host", "codex", "--test-mode", "--worker", "manual"]))


def test_slow_probe_diagnostics_finish_before_directory_removal(fake_user, monkeypatch):
    """Release the real writer after the short shutdown budget has already expired."""
    entered, release = Event(), Event()
    write = RecallLogStore.write
    closing = setup_verify.close_writers
    roots, budgets, timers = [], [], []

    def delayed(store, record):
        if any(part.startswith("setup-check-") for part in store.path.parts):
            entered.set()
            assert release.wait(30)
        return write(store, record)

    def finish(root, *, timeout=1.0):
        roots.append(root)
        budgets.append(timeout)
        assert entered.wait(5)
        timer = Timer(1.5, release.set)
        timer.start()
        timers.append(timer)
        return closing(root, timeout=timeout)

    monkeypatch.setattr(RecallLogStore, "write", delayed)
    monkeypatch.setattr(setup_verify, "close_writers", finish)
    try:
        result = setup_apply.apply_setup(_plan(fake_user))
        print("SLOW_PROBE_RECEIPT", json.dumps(result))
        assert result["status"] == "ready", result
        check = result["hosts"][0]["verification"]
        assert check["capture"] and check["cross_session_recall"] and not check["models_verified"]
        assert check["cleanup"]["status"] == "removed"
        assert budgets == [setup_verify.DIAGNOSTIC_CLOSE_SECONDS]
        assert budgets[0] > 1.5
        assert roots and all(not root.exists() for root in roots)
    finally:
        release.set()
        for timer in timers:
            timer.join(5)
        for root in roots:
            assert close_writers(root, timeout=5)


@pytest.mark.parametrize("recalled", [True, False])
def test_unclosed_probe_is_retained_without_changing_verification(fake_user, monkeypatch, recalled):
    """Cleanup metadata must preserve both a positive and a negative recall conclusion."""
    client = setup_verify._client

    def recall(*args):
        answer = client(*args)
        return answer if recalled or args[2] != "setup-second" else {}

    with monkeypatch.context() as patch:
        patch.setattr(setup_verify, "close_writers", lambda *args, **kwargs: False)
        patch.setattr(setup_verify, "_client", recall)
        result = setup_apply.apply_setup(_plan(fake_user))
    print("RETAINED_PROBE_RECEIPT", json.dumps(result))
    roots = list(fake_user.glob("setup-check-*"))
    try:
        assert result["status"] == ("ready" if recalled else "verification_failed"), result
        check = result["hosts"][0]["verification"]
        assert check["cross_session_recall"] is recalled and not check["models_verified"]
        cleanup = check["cleanup"]
        assert cleanup == dict(
            status="retained", directory=str(roots[0]), reason="verification diagnostics still active"
        )
        assert len(roots) == 1 and (roots[0] / "store" / "memory.sqlite3").is_file()
        assert json.loads(Path(result["receipt"]).read_text("utf-8"))["hosts"][0]["verification"] == check
    finally:
        for root in roots:
            setup_verify._clean_owned_check(root)


def test_probe_cleanup_io_failure_is_reported_without_exception_text(fake_user, monkeypatch):
    """A filesystem refusal retains the probe without persisting arbitrary exception text."""
    unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if any(part.startswith("setup-check-") for part in path.parts):
            raise PermissionError(13, "TEST-sensitive-cleanup-detail", str(path))
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        result = setup_apply.apply_setup(_plan(fake_user))
    roots = list(fake_user.glob("setup-check-*"))
    try:
        assert result["status"] == "ready", result
        cleanup = result["hosts"][0]["verification"]["cleanup"]
        assert cleanup["status"] == "retained" and cleanup["directory"] == str(roots[0])
        assert cleanup["reason"] == "verification directory cleanup failed"
        assert cleanup["error_type"] == "PermissionError"
        assert "TEST-sensitive-cleanup-detail" not in Path(result["receipt"]).read_text("utf-8")
    finally:
        for root in roots:
            setup_verify._clean_owned_check(root)


def test_setup_os_failure_receipt_contains_only_numeric_error_details(fake_user, monkeypatch):
    """OS codes locate an installation failure while its text and filename remain private."""
    failure = OSError(13, "TEST-sensitive-install-detail", "TEST-sensitive-file")
    failure.winerror = 5

    def fail(*args):
        raise failure

    monkeypatch.setattr(setup_apply, "_install", fail)
    result = setup_apply.apply_setup(_plan(fake_user))
    assert result["status"] == "failed" and result["failed_step"] == "codex:install"
    assert result["error_type"] == "PermissionError" and result["reason"] is None
    assert result["errno"] == 13 and result["winerror"] == 5
    assert "TEST-sensitive" not in Path(result["receipt"]).read_text("utf-8")


def test_entry_collision_uses_the_installers_safe_refusal(fake_user, monkeypatch):
    """An installer-owned fixed refusal remains visible in the setup receipt."""
    plan = _plan(fake_user)
    monkeypatch.setattr(setup_apply, "read_shared_payload", lambda root: dict(entries=[dict(entry_id="codex")]))
    result = setup_apply.apply_setup(plan)
    assert result["status"] == "failed" and result["failed_step"] == "codex:binding"
    assert result["error_type"] == InstallError.__name__
    assert result["reason"] == "entry name already belongs to another home; choose that home or detach explicitly"


def test_changed_project_suggestion_uses_the_installers_safe_refusal(fake_user, monkeypatch):
    """A stale registration suggestion reports the installer's fixed recovery instruction."""
    from scope_recall.maintenance import setup_projects

    plan = _plan(fake_user)
    plan.project_copies.append(dict(home=str(fake_user / ".codex"), directory=str(fake_user / "TEST-copy")))
    monkeypatch.setattr(setup_projects, "load_codex_config", lambda path: None)
    monkeypatch.setattr(setup_projects, "project_suggestions", lambda *args: [])
    result = setup_apply.apply_setup(plan, register_copies=True)
    assert result["status"] == "failed" and result["failed_step"] == "project-registration"
    assert result["error_type"] == InstallError.__name__
    assert result["reason"] == "project suggestion changed; rerun setup"
