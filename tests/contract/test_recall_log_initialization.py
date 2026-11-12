"""Reads of a recall log whose initial schema has not been committed."""

import sqlite3

import pytest
from scope_recall.contracts import ContractError
from scope_recall.core.management import MemoryManagement
from scope_recall.core.recall_log_storage import RecallLogStore
from scope_recall.core.restore import InstallationMaintenance

from tests.contract.test_claims import app


@pytest.mark.parametrize("open_wal", [False, True])
@pytest.mark.parametrize("now", [None, "2026-09-06T12:00:00Z"])
def test_uninitialized_log_is_an_empty_history(tmp_path, open_wal, now):
    store = RecallLogStore(tmp_path / "recall.sqlite3")
    connection = store._open() if open_wal else sqlite3.connect(store.path)
    connection.close()
    assert store.list(now=now) == []


def test_inspection_of_uninitialized_log_reports_missing(app, monkeypatch):
    core, ctx = app
    sqlite3.connect(core.recall_diagnostics.path).close()
    # Keep the file in its initial state while the reader checks it.
    monkeypatch.setattr(core.recall_diagnostics, "expire", lambda _now: None)
    with pytest.raises(ContractError, match="SOURCE_MISSING"):
        core.inspect_object(ctx, "recall-diag:TEST-missing")


def test_uninitialized_history_is_an_empty_page(app, monkeypatch):
    core, ctx = app
    sqlite3.connect(core.recall_diagnostics.path).close()
    # Keep the file in its initial state while the reader checks it.
    monkeypatch.setattr(core.recall_diagnostics, "expire", lambda _now: None)
    owner = MemoryManagement(core, InstallationMaintenance(ctx))
    assert owner.recalls() == {"items": [], "next_before": None}


def test_first_write_can_be_viewed_before_schema_commit(tmp_path, monkeypatch):
    from threading import Event

    from scope_recall.core import recall_log_writer
    from scope_recall.core.recall_diagnostics import RecallDiagnostics

    diagnostics = RecallDiagnostics(tmp_path / "recall.sqlite3")
    entered, release = Event(), Event()
    original_schema = diagnostics.store._schema

    def blocked_schema(conn):
        entered.set()
        assert release.wait(60)
        original_schema(conn)

    monkeypatch.setattr(diagnostics.store, "_schema", blocked_schema)
    try:
        ref = diagnostics.record(
            installation_id="TEST-installation",
            session_id="TEST-session",
            request_id="TEST-request",
            memory_epoch=1,
            phase="compile",
            status="ok",
            retrieval_gaps=(),
            compile_gaps=(),
            items_delivered=0,
            items_dropped_stale=0,
            items_dropped_budget=0,
            elapsed_ms=0,
            deadline_remaining_ms=None,
        )
        assert entered.wait(10)
        assert diagnostics.path.is_file()
        assert not diagnostics.flush(0.05)
        monkeypatch.setattr(recall_log_writer, "VIEW_WAIT_SECONDS", 0.05)
        assert diagnostics.get(ref) is None
        assert diagnostics.store.list(ref=ref, limit=1) == []
        release.set()
        monkeypatch.setattr(recall_log_writer, "VIEW_WAIT_SECONDS", 60.0)
        assert diagnostics.flush()
        assert diagnostics.get(ref).ref == ref
    finally:
        release.set()
        assert diagnostics.close(timeout=5)
