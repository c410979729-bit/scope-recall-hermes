"""Model receipt schema upgrades preserve synthetic shared truth and its rollback snapshot."""

import json
import sqlite3
from contextlib import closing

import pytest
from scope_recall.adapters.clients.config import load_shared_client
from scope_recall.adapters.clients.identity import resolve_runtime_audience, trusted_context
from scope_recall.core import CoreConfig, MemoryCore, storage
from scope_recall.core.schema import SCHEMA_VERSION
from scope_recall.core.store_upgrade import read_upgrade_state

from tests.host.test_local_account_pin import shared_world
from tests.v11_support import downgrade_store, source_event


def inventory(path):
    """Read every existing user table except the schema marker and new receipt table."""
    with closing(sqlite3.connect(path)) as conn:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "AND name NOT IN ('instance_meta','model_revision_receipts')"
            )
        ]
        return {table: sorted(conn.execute('SELECT * FROM "' + table + '"').fetchall(), key=repr) for table in tables}


def markers(path):
    """Read both schema stamps and the installation identity from a closed synthetic store."""
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute("PRAGMA user_version").fetchone()[0], dict(
            conn.execute("SELECT * FROM instance_meta").fetchone()
        )


@pytest.fixture
def old_shared(tmp_path):
    """Make a populated shared store with two distinct entry grants, then its synthetic old format."""
    shared_world(tmp_path)
    config = load_shared_client(tmp_path / "first", "codex")
    core = MemoryCore(CoreConfig(config.to_binding()))
    audience = resolve_runtime_audience(config, str(config.home))
    ctx = trusted_context(config, audience, session_id="TEST-migration", actor_origin="human_direct", mutation=True)
    with core.storage.write(ctx) as tx:
        tx.registry.register_entry(config.entry_id, config.entry_name, "codex", now=core.clock.utc_now())
    event = source_event(content="TEST migration source", occurred_at=None, time_precision="unknown")
    core.record_event(ctx, event, scope_id=audience.capture_scope_id)
    downgrade_store(core.storage.path, 1134)
    return core, ctx, config


@pytest.mark.parametrize("journal", ["delete", "wal"])
def test_upgrade_preserves_shared_data_identity_and_grants(old_shared, journal):
    core, ctx, config = old_shared
    with closing(sqlite3.connect(core.storage.path)) as conn:
        conn.execute("PRAGMA journal_mode=" + journal)
    before, (_, identity) = inventory(core.storage.path), markers(core.storage.path)
    assert identity.pop("schema_version") == 1134
    assert core.storage.initialize(remaining_seconds=30).schema_version == SCHEMA_VERSION == 1135
    assert inventory(core.storage.path) == before
    header, after = markers(core.storage.path)
    assert header == after.pop("schema_version") == 1135
    assert after == identity
    fresh_config = load_shared_client(config.home, "codex")
    assert resolve_runtime_audience(fresh_config, str(config.home)).allowed_scope_ids == ctx.allowed_scope_ids
    assert fresh_config.to_binding() == config.to_binding()
    assert "TEST-second" not in ctx.allowed_scope_ids
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT count(*) FROM model_revision_receipts").fetchone()[0] == 0
        assert not tx._check().execute("PRAGMA foreign_key_check").fetchall()
    state = read_upgrade_state(core.storage.path.parent)
    assert state["status"] == "complete" and state["snapshot_verified"]
    assert markers(state["backup"])[0] == 1134
    assert inventory(state["backup"]) == before


@pytest.mark.parametrize("stage", ["before_ddl", "after_ddl"])
def test_failed_upgrade_rolls_back_ddl_and_preserves_backup(old_shared, monkeypatch, stage):
    core, ctx, config = old_shared
    before, identity = inventory(core.storage.path), markers(core.storage.path)
    upgrade = storage.SCHEMA_MIGRATIONS[1134]

    def fail(conn):
        if stage == "after_ddl":
            upgrade(conn)
        raise RuntimeError("TEST migration failure")

    with monkeypatch.context() as patch:
        patch.setitem(storage.SCHEMA_MIGRATIONS, 1134, fail)
        with pytest.raises(RuntimeError, match="TEST migration failure"):
            core.storage.initialize(remaining_seconds=30)
    assert markers(core.storage.path) == identity
    assert inventory(core.storage.path) == before
    with closing(sqlite3.connect(core.storage.path)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='model_revision_receipts'").fetchone() is None
    state = read_upgrade_state(core.storage.path.parent)
    assert state["status"] == "failed" and state["snapshot_verified"]
    assert markers(state["backup"]) == identity
    assert inventory(state["backup"]) == before
    assert core.storage.initialize(remaining_seconds=30).schema_version == 1135
