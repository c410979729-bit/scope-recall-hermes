"""The sole core SQLite transaction boundary. No host or Provider dependencies."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import stat
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from ..contracts import (
    ContractError,
    InstanceBinding,
    TrustedContext,
)
from . import lexical_index
from .deadline import remaining_seconds as request_remaining_seconds
from .delete_storage import watch_mirrored_tombstones
from .file_lock import advisory_file_lock
from .restore import restore_pending
from .schema import (
    APPLICATION_ID,
    PRIVATE_INTERVALS_STATEMENT,
    PRIVATE_TARGET_INDEX,
    SCHEMA_VERSION,
    STATEMENTS,
    UPGRADE_CHAIN,
    ensure_source_session_index,
    source_session_index_present,
    stale_header_schema,
    upgrade_1105,
    upgrade_1106,
    upgrade_1107,
    upgrade_1108,
    upgrade_1109,
    upgrade_1110,
    upgrade_1111,
    upgrade_1112,
    upgrade_1113,
    upgrade_1114,
    upgrade_1115,
    upgrade_1116,
    upgrade_1117,
    upgrade_1118,
    upgrade_1119,
    upgrade_1120,
    upgrade_1121,
    upgrade_1122,
    upgrade_1123,
    upgrade_1124,
    upgrade_1125,
    upgrade_1126,
    upgrade_1127,
    upgrade_1128,
    upgrade_1129,
    upgrade_1130,
    upgrade_1131,
    upgrade_1132,
    upgrade_1133,
    upgrade_1134,
)
from .source_records import clear_body_scan_receipts
from .store_upgrade import UPGRADE_LOCK, recover_upgrade_snapshot, upgrade_in_progress, upgrade_store
from .transaction import StoreStatus as StoreStatus
from .transaction import Transaction as Transaction
from .transaction import _cleanup_error
from .truth_connection import TruthDatabaseMode, connect_truth_database
from .writer_lease import TruthWriterBusyError

#: How often a writer looks again for another process's lease while it waits.
_LEASE_POLL_SECONDS = 0.01

# Bounded process-local structural results only. Identity, scopes and live
# restore/privacy fences never use this cache. A fork gets different keys.
_STRUCTURAL_STATE: dict[tuple, tuple[bool, bool]] = {}
_STRUCTURAL_LOCK = threading.Lock()

# A migration consumes its key's format and advances it to the next format.
SCHEMA_MIGRATIONS = dict(
    zip(
        UPGRADE_CHAIN,
        (
            upgrade_1105,
            upgrade_1106,
            upgrade_1107,
            upgrade_1108,
            upgrade_1109,
            upgrade_1110,
            upgrade_1111,
            upgrade_1112,
            upgrade_1113,
            upgrade_1114,
            upgrade_1115,
            upgrade_1116,
            upgrade_1117,
            upgrade_1118,
            upgrade_1119,
            upgrade_1120,
            upgrade_1121,
            upgrade_1122,
            upgrade_1123,
            upgrade_1124,
            upgrade_1125,
            upgrade_1126,
            upgrade_1127,
            upgrade_1128,
            upgrade_1129,
            upgrade_1130,
            upgrade_1131,
            upgrade_1132,
            upgrade_1133,
            upgrade_1134,
        ),
        strict=True,
    )
)


def _privacy_read_authorizer(action: int, table: str | None, *_rest: object) -> int:
    """Restrict a privacy snapshot to interval metadata without exposing source rows."""
    return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_READ and table != "private_intervals" else sqlite3.SQLITE_OK


def _check_privacy_contract(conn: sqlite3.Connection) -> None:
    """Require the canonical privacy schema, including case-sensitive stored literal values."""
    for name, expected in (
        ("private_intervals", PRIVATE_INTERVALS_STATEMENT),
        ("private_target", PRIVATE_TARGET_INDEX),
    ):
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if row is None or " ".join(row[0].split()) != " ".join(expected.split()):
            raise ContractError("SCHEMA_UNSUPPORTED", "privacy_contract")


def _directory(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


#: A store this large is brought forward only by a caller with this much
#: budget: the 1109 step rebuilds the lexical index, 95 s for 5.2 million
#: rows on a 1.4 GB store, which no hook can carry and every worker pass can.
HEAVY_UPGRADE_BYTES = 100_000_000
HEAVY_UPGRADE_SECONDS = 60.0


def upgrade_fits(store_bytes: int, remaining_seconds: float | None) -> bool:
    """Whether an open with this budget may bring a store of this size forward."""
    return store_bytes < HEAVY_UPGRADE_BYTES or (
        remaining_seconds is not None and remaining_seconds >= HEAVY_UPGRADE_SECONDS
    )


def _store_bytes(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute("PRAGMA page_size").fetchone()[0]


def _initialization_left(deadline: float | None) -> float | None:
    """Return the unspent part of the single initialization deadline."""
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _finish_claim_index(conn: sqlite3.Connection, deadline: float | None) -> None:
    """Commit bounded index pages under initialization admission; an interrupted page is repeatable."""
    while lexical_index.claim_backfill_pending(conn):
        if deadline is not None and time.monotonic() >= deadline:
            raise ContractError("SCHEMA_UNSUPPORTED", "upgrade_deadline_exceeded")
        conn.execute("BEGIN IMMEDIATE")
        try:
            lexical_index.backfill_claim_page(conn)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise


#: Requested memory-map size.  Operations open separate connections, so they do not share SQLite's page cache.
#: Mapped reads use the system's file cache.  SQLite maps no more than the file holds or its build supports;
#: remaining pages use ordinary reads, and writes are unchanged.  An I/O error on a mapped page ends the process,
#: and a mapped file cannot shrink.
STORE_MMAP_BYTES = 8 << 30


def _ensure_wal(conn: sqlite3.Connection) -> None:
    """Keep the store in WAL mode, where readers and the writer coexist.

    Under the rollback journal a two-second read left a writer "database is
    locked" after its whole timeout; under WAL it commits in 20 ms.  The mode
    is persistent in the file, so this is one pragma read almost always.  It
    runs only on a store this code has verified as its own, outside any
    transaction (where SQLite allows the switch); a concurrent connection can
    make SQLite decline the switch, and the next writable open tries again.
    Backups (maintenance/backup.py) already write their snapshot in rollback
    mode, and every read-only open reads a WAL store in every file state.
    """
    if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
        conn.execute("PRAGMA journal_mode=WAL")


class SQLiteStorage:
    def __init__(
        self,
        binding: InstanceBinding,
        *,
        timeout_seconds: float = 1.0,
        upgrade_on_open: bool = True,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(binding, InstanceBinding):
            raise ContractError("IDENTITY_UNBOUND")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 <= timeout_seconds <= 30
        ):
            raise ContractError("INPUT_INVALID", "storage_timeout")
        if type(upgrade_on_open) is not bool:
            raise ContractError("INPUT_INVALID", "upgrade_on_open")
        self.__binding = binding
        self.timeout_seconds = float(timeout_seconds)
        #: A store left at an older known schema by a package upgrade is brought
        #: forward by the first transaction that opens it (``initialize``: one
        #: transaction, identity-verified, rolled back whole on failure), so
        #: ``pip install -U`` alone is enough.  The doctor turns this off: it
        #: reports a pending upgrade and never applies one.
        self.upgrade_on_open = upgrade_on_open
        #: The monotonic clock a writer's wait for the lease is timed on: the one its ``MemoryCore`` runs on, so that
        #: the store and the recalls and worker passes over it keep one time.
        self.monotonic = clock if clock is not None else time.monotonic
        self.__pending_close: list[sqlite3.Connection] = []
        #: Optional runtime-owned scheduling edge; core never imports a worker launcher.
        self.on_upgrade_pending: Callable[[], object] | None = None
        #: Late-bound vector read port owned by the composition root.
        self.entity_vectors: Callable[[], object] | None = None

    def _defer_upgrade(self) -> NoReturn:
        """Notify the scheduling owner without replacing the authoritative schema gap."""
        if self.on_upgrade_pending is not None:
            with suppress(Exception):
                self.on_upgrade_pending()
        raise ContractError("SCHEMA_UNSUPPORTED", "upgrade_pending")

    @property
    def binding(self) -> InstanceBinding:
        return self.__binding

    @property
    def path(self) -> Path:
        return self.__binding.data_directory / "memory.sqlite3"

    def _path_check(self) -> None:
        # Reject any symlink/junction ancestor, including read-only opens.
        for path in (self.path, *self.path.parents):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            # Both flags come from one native record. Re-read it on every open:
            # path safety is live, even when the stored schema is unchanged.
            junction = getattr(info, "st_reparse_tag", None) == getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", -1)
            if stat.S_ISLNK(info.st_mode) or junction:
                raise ContractError("IDENTITY_UNBOUND", "data_directory")
        if _directory(self.binding.data_directory.resolve()) != _directory(self.binding.data_directory):
            raise ContractError("IDENTITY_UNBOUND", "data_directory")

    def _context_check(self, context: TrustedContext) -> None:
        if not isinstance(context, TrustedContext) or context.binding != self.binding:
            raise ContractError("IDENTITY_UNBOUND")
        if not context.allowed_scope_ids:
            raise ContractError("ACCESS_DENIED")

    def _timeout(self, remaining_seconds: float | None) -> float:
        """Refuse expired callers, including a request whose positive lock wait outlives its deadline."""
        timeout = self.timeout_seconds
        for remaining in (remaining_seconds, request_remaining_seconds()):
            if remaining is None:
                continue
            if type(remaining) not in (int, float) or not math.isfinite(remaining) or remaining <= 0:
                raise ContractError("DEADLINE_EXCEEDED")
            timeout = min(timeout, remaining)
        return timeout

    def _remaining_budget(self, remaining_seconds: float | None, started: float) -> float | None:
        """Share one caller budget across connection setup, initialization and reopen."""
        explicit = None if remaining_seconds is None else remaining_seconds - (self.monotonic() - started)
        return min(
            (value for value in (explicit, request_remaining_seconds()) if value is not None),
            default=None,
        )

    def _open(
        self,
        mode: TruthDatabaseMode,
        remaining_seconds: float | None = None,
        *,
        restoring: bool = False,
        coordinating: bool = False,
    ) -> sqlite3.Connection:
        # Commit, rollback and connection/lease cleanup finish outside this entry check.
        timeout = self._timeout(remaining_seconds)
        # A failed close cannot silently abandon an acquired writer lease. No
        # new connection is opened until the prior close succeeds.
        while self.__pending_close:
            pending = self.__pending_close[-1]
            try:
                pending.close()
            except BaseException as exc:
                raise ContractError("STORAGE_UNAVAILABLE", "connection_cleanup") from exc
            self.__pending_close.pop()
        self._path_check()
        if not restoring and not coordinating and upgrade_in_progress(self.binding.data_directory):
            raise ContractError("SCHEMA_UNSUPPORTED", "upgrade_in_progress")
        if not restoring and restore_pending(self.binding.data_directory):
            raise ContractError("RESTORE_UNVERIFIED")
        # The writer lease is taken without blocking and held for one
        # transaction, so writers in separate processes -- a host and its
        # worker, the entries of a shared store -- take turns.  A turn ends in
        # milliseconds; failing at once on one sent captures to the memory-only
        # retry (one in ten with three entries writing).  A writer waits for the
        # lease as SQLite waits for its own lock: within this timeout, counted from the first refusal, so that an
        # open that finds the lease free reads no clock.  The polls are counted as well, since the clock a caller
        # gives need not move while this sleeps: a poll is slept only while the polls so far and it fit in the
        # timeout, so no wait sleeps past it on any clock.  On one that keeps time, the polls so far have taken at
        # least their own length, so the count gives up only where the clock does too.
        deadline: float | None = None
        polls = 0
        while True:
            try:
                left = timeout if deadline is None else max(0.0, deadline - self.monotonic())
                # A zero lock wait still gets its first nonblocking attempt;
                # request budgets have already been checked by _timeout.
                if deadline is not None and left <= 0:
                    raise TruthWriterBusyError()
                conn = connect_truth_database(self.path, mode=mode, timeout=left, isolation_level=None)
            except TruthWriterBusyError:
                now = self.monotonic()
                deadline = now + timeout if deadline is None else deadline
                if now + _LEASE_POLL_SECONDS >= deadline or (polls + 1) * _LEASE_POLL_SECONDS >= timeout:
                    raise
                polls += 1
                time.sleep(_LEASE_POLL_SECONDS)
                continue
            try:
                conn.execute(f"PRAGMA mmap_size={STORE_MMAP_BYTES}")
            except BaseException as exc:
                self._close(conn, exc)
                raise
            return conn

    def verify(self, conn: sqlite3.Connection, *, expected_schema: int | None = None) -> sqlite3.Row:
        """Refuse the database ``conn`` reads unless it is this binding's store at ``expected_schema``: its header,
        its recorded schema, its kind, its identity and its scopes.  Its file need not be the store's own: a
        snapshot of the store is checked the same way before it is copied over it (``maintenance/restore.py``).

        Return the freshly verified metadata row, never a reusable authorization verdict.
        """
        expected_schema = SCHEMA_VERSION if expected_schema is None else expected_schema
        if (
            conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
            or conn.execute("PRAGMA user_version").fetchone()[0] != expected_schema
        ):
            if stale_header_schema(conn) is not None:
                # The store is intact and records its schema; only the header was overwritten.
                raise ContractError("SCHEMA_UNSUPPORTED", "header_stale:run_upgrade_store")
            raise ContractError("SCHEMA_UNSUPPORTED")
        row = conn.execute("SELECT * FROM instance_meta WHERE singleton=1").fetchone()
        if row is None or row["schema_version"] != expected_schema:
            raise ContractError("SCHEMA_UNSUPPORTED")
        # A store from before 1110 has no kind column; it was a host's own store.
        kind = row["installation_kind"] if "installation_kind" in row.keys() else "local"
        if kind != self.binding.installation_kind:
            raise ContractError("IDENTITY_UNBOUND", "installation_kind")
        # Counted in one row, every transaction: read row by row, the shared store's 760 scopes cost a busy Hermes
        # gateway 3 s per transaction (``lexical_index.index_terms``).  A binding's scopes are a set.
        stored, held = conn.execute(
            "SELECT (SELECT count(*) FROM instance_scopes),"
            " (SELECT count(*) FROM instance_scopes WHERE scope_id IN (SELECT value FROM json_each(?)))",
            (json.dumps(sorted(self.binding.scope_ids), ensure_ascii=False),),
        ).fetchone()
        if kind == "local":
            if (row["agent_id"], row["installation_id"], row["data_directory"], row["test_mode"]) != (
                self.binding.agent_id,
                self.binding.installation_id,
                _directory(self.binding.data_directory),
                int(self.binding.test_mode),
            ):
                raise ContractError("IDENTITY_UNBOUND")
            if stored != held or held != len(self.binding.scope_ids):
                raise ContractError("IDENTITY_UNBOUND", "scope_binding")
            return row
        # A shared store is its fixed id, not its directory: a copied store opens
        # nowhere until ``adopt`` records the new place.  Entries each bind a
        # subset of its scopes; the store grows as they attach.
        if (row["agent_id"], row["installation_id"], row["test_mode"]) != (
            self.binding.agent_id,
            self.binding.installation_id,
            int(self.binding.test_mode),
        ):
            raise ContractError("IDENTITY_UNBOUND")
        if row["data_directory"] != _directory(self.binding.data_directory):
            raise ContractError("IDENTITY_UNBOUND", "store_moved:run_adopt")
        if held != len(self.binding.scope_ids):
            raise ContractError("IDENTITY_UNBOUND", "scope_binding")
        return row

    def _close(self, conn: sqlite3.Connection, original: BaseException | None) -> None:
        try:
            conn.close()
        except BaseException as cleanup:
            self.__pending_close.append(conn)
            if original is None:
                raise
            _cleanup_error(original, cleanup, "close")

    def _upgrade_schema(self, conn: sqlite3.Connection) -> None:
        """Apply each supported migration in order after checking its input identity."""
        for version, migrate in SCHEMA_MIGRATIONS.items():
            if conn.execute("PRAGMA user_version").fetchone()[0] == version:
                self.verify(conn, expected_schema=version)
                migrate(conn)

    def initialize(
        self,
        *,
        remaining_seconds: float | None = None,
        backup_dir: Path | None = None,
        repair_header: bool = False,
        receipt: dict | None = None,
        unbounded_upgrade: bool = False,
        created_at: str | None = None,
    ) -> StoreStatus:
        """Create or upgrade this store and return its verified status within the caller's lock-wait budget."""
        self._timeout(remaining_seconds)
        budget = min((v for v in (remaining_seconds, request_remaining_seconds()) if v is not None), default=None)
        deadline = None if budget is None else time.monotonic() + budget
        conn = self._open("rwc", _initialization_left(deadline), coordinating=True)
        original = None
        try:
            with advisory_file_lock(
                self.binding.data_directory / UPGRADE_LOCK,
                timeout_seconds=self._timeout(_initialization_left(deadline)),
            ):
                return self._initialize(
                    conn,
                    deadline=deadline,
                    backup_dir=backup_dir,
                    repair_header=repair_header,
                    receipt=receipt,
                    unbounded_upgrade=unbounded_upgrade,
                    created_at=created_at or datetime.now(UTC).isoformat(),
                )
        except BaseException as exc:
            original = exc
            raise
        finally:
            self._close(conn, original)

    def _initialize(
        self,
        conn: sqlite3.Connection,
        *,
        deadline: float | None,
        backup_dir: Path | None,
        repair_header: bool,
        receipt: dict | None,
        unbounded_upgrade: bool,
        created_at: str,
    ) -> StoreStatus:
        status: StoreStatus | None = None

        def migrate(conn):
            nonlocal status
            self._migrate_store(conn)
            status = self._initial_status(conn)

        try:
            conn.execute(f"PRAGMA busy_timeout={int(self._timeout(_initialization_left(deadline)) * 1000)}")
            conn.execute("BEGIN IMMEDIATE")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            recorded = stale_header_schema(conn) if repair_header else None
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1").fetchone()
            if exists:
                if version in UPGRADE_CHAIN or recorded is not None:
                    if not unbounded_upgrade and not upgrade_fits(_store_bytes(conn), _initialization_left(deadline)):
                        self._defer_upgrade()
                    outcome = upgrade_store(
                        conn,
                        self.path,
                        target=SCHEMA_VERSION,
                        verify=self.verify,
                        migrate=migrate,
                        backup_dir=backup_dir,
                        recorded_schema=recorded,
                        deadline=deadline,
                    )
                    if receipt is not None:
                        receipt.update(outcome)
                    if recorded is not None and receipt is not None:
                        receipt["header_restamped"] = {"from": version, "to": recorded}
                else:
                    self.verify(conn)
                    recover_upgrade_snapshot(self.binding.data_directory, current_schema=version)
                    ensure_source_session_index(conn)
            else:
                if (
                    conn.execute("PRAGMA user_version").fetchone()[0] != 0
                    or conn.execute("PRAGMA application_id").fetchone()[0] != 0
                ):
                    raise ContractError("SCHEMA_UNSUPPORTED")
                for statement in STATEMENTS:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO instance_meta(singleton,agent_id,installation_id,data_directory,schema_version,"
                    "test_mode,installation_kind,database_id,database_created_at) VALUES (1,?,?,?,?,?,?,?,?)",
                    (
                        self.binding.agent_id,
                        self.binding.installation_id,
                        _directory(self.binding.data_directory),
                        SCHEMA_VERSION,
                        int(self.binding.test_mode),
                        self.binding.installation_kind,
                        uuid.uuid4().hex,
                        created_at,
                    ),
                )
                conn.executemany(
                    "INSERT INTO instance_scopes(scope_id) VALUES (?)", [(s,) for s in sorted(self.binding.scope_ids)]
                )
                conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            if status is None:
                status = self._initial_status(conn)
            conn.commit()
            _finish_claim_index(conn, deadline)
            with suppress(sqlite3.Error):
                conn.execute("PRAGMA busy_timeout=0")
                _ensure_wal(conn)
        except BaseException as exc:
            try:
                if conn.in_transaction:
                    conn.rollback()
            except BaseException as cleanup:
                _cleanup_error(exc, cleanup, "rollback")
            raise
        return status

    def _initial_status(self, conn: sqlite3.Connection) -> StoreStatus:
        """Read the verified result inside its owning transaction before commit."""
        context = TrustedContext(self.binding, "initialization", self.binding.scope_ids, "host_generated")
        tx = Transaction(conn, context, writable=False)
        try:
            return tx.status()
        finally:
            tx._finish()

    def _migrate_store(self, conn: sqlite3.Connection) -> None:
        """Apply the ordered format changes and their required lookup index."""
        self._upgrade_schema(conn)
        ensure_source_session_index(conn)

    def adopt(self) -> str:
        """Record this binding's directory as where a copied shared store now lives.

        A shared store is its fixed id, so a copy opens nowhere until this runs:
        ``verify`` refuses it with ``store_moved:run_adopt``.  Everything
        ``verify`` checks is checked here except the directory, which is then
        written.  Returns the directory the store recorded before.  A local store
        is its directory and is never adopted.
        """
        if self.binding.installation_kind != "shared":
            raise ContractError("ACCESS_DENIED", "local_store")
        conn = self._open("rw")
        original = None
        try:
            if (
                conn.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
                or conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION
            ):
                raise ContractError("SCHEMA_UNSUPPORTED")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM instance_meta WHERE singleton=1").fetchone()
            if row is None or row["schema_version"] != SCHEMA_VERSION:
                raise ContractError("SCHEMA_UNSUPPORTED")
            if row["installation_kind"] != "shared":
                raise ContractError("IDENTITY_UNBOUND", "installation_kind")
            if (row["agent_id"], row["installation_id"], row["test_mode"]) != (
                self.binding.agent_id,
                self.binding.installation_id,
                int(self.binding.test_mode),
            ):
                raise ContractError("IDENTITY_UNBOUND")
            if not self.binding.scope_ids <= frozenset(
                r[0] for r in conn.execute("SELECT scope_id FROM instance_scopes")
            ):
                raise ContractError("IDENTITY_UNBOUND", "scope_binding")
            previous = row["data_directory"]
            clear_body_scan_receipts(conn)
            conn.execute(
                "UPDATE instance_meta SET data_directory=? WHERE singleton=1",
                (_directory(self.binding.data_directory),),
            )
            conn.commit()
            return previous
        except BaseException as exc:
            original = exc
            try:
                if conn.in_transaction:
                    conn.rollback()
            except BaseException as cleanup:
                _cleanup_error(exc, cleanup, "rollback")
            raise
        finally:
            self._close(conn, original)

    def _file_stamp(self):
        """Observe file identity and content without opening a descriptor that could disturb POSIX locks.

        The first three fields name the file; modification time and size stand for its content. The inode change
        time is left out: macOS advances it on every write transaction, even one that writes nothing, which would
        make every transaction recheck the structural state.
        """
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        return (
            info.st_dev,
            info.st_ino,
            getattr(info, "st_birthtime_ns", 0),
            info.st_mtime_ns,
            info.st_size,
        )

    def _structural_state(self, conn, row, opened_stamp):
        """Reuse index readiness only for an unchanged file, schema and metadata snapshot."""
        stamp = self._file_stamp()
        if stamp is None or opened_stamp is None or stamp[:3] != opened_stamp[:3]:
            raise ContractError("IDENTITY_UNBOUND", "store_replaced_during_open")
        key = (
            _directory(self.path),
            os.getpid(),
            stamp,
            conn.execute("PRAGMA schema_version").fetchone()[0],
            tuple(row),
        )
        with _STRUCTURAL_LOCK:
            cached = _STRUCTURAL_STATE.get(key)
        if cached is not None:
            return cached
        state = (self._claim_index_pending(conn, False, SCHEMA_VERSION), source_session_index_present(conn))
        # Pending work and missing indexes must be retried under their current
        # budget. Only complete structural validation is reusable.
        if state == (False, True):
            with _STRUCTURAL_LOCK:
                if len(_STRUCTURAL_STATE) >= 128:
                    _STRUCTURAL_STATE.clear()
                _STRUCTURAL_STATE[key] = state
        return state

    def _begin_transaction(self, context, writable, remaining_seconds, restoring):
        """Open and verify a fresh authority snapshot, initializing only outside it."""
        started = self.monotonic() if remaining_seconds is not None else 0.0
        opened_stamp = self._file_stamp()
        conn = self._open("rw" if writable else "ro", remaining_seconds, restoring=restoring)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            upgrade = self.upgrade_on_open and not restoring and version in UPGRADE_CHAIN
            budget = self._remaining_budget(remaining_seconds, started)
            fits = upgrade_fits(_store_bytes(conn), budget) if upgrade else True
            if upgrade and not fits:
                # Refusing a migration reads no truth rows. Keep that budget
                # refusal ahead of opening an authority snapshot.
                self._defer_upgrade()
            if writable and version == SCHEMA_VERSION:
                _ensure_wal(conn)
            conn.execute("BEGIN IMMEDIATE" if writable else "BEGIN")
            row = self.verify(
                conn,
                expected_schema=version
                if self.upgrade_on_open and not restoring and version in UPGRADE_CHAIN
                else None,
            )
            if version == SCHEMA_VERSION and not restoring:
                pending, indexed = self._structural_state(conn, row, opened_stamp)
                fits = upgrade_fits(_store_bytes(conn), budget) if pending else fits
                upgrade = (
                    upgrade
                    or pending
                    or (writable and self.upgrade_on_open and not indexed and upgrade_fits(_store_bytes(conn), budget))
                )
            if upgrade:
                conn.rollback()
                self._close(conn, None)
                conn = None
                if not fits:
                    self._defer_upgrade()
                self.initialize(remaining_seconds=self._remaining_budget(remaining_seconds, started))
                conn = self._open(
                    "rw" if writable else "ro", self._remaining_budget(remaining_seconds, started), restoring=restoring
                )
                conn.execute("BEGIN IMMEDIATE" if writable else "BEGIN")
                self.verify(conn)
            return conn
        except BaseException as exc:
            if conn is not None:
                try:
                    if conn.in_transaction:
                        conn.rollback()
                except BaseException as cleanup:  # noqa: BLE001 - preserve the original failure while reporting cleanup
                    _cleanup_error(exc, cleanup, "rollback")
                self._close(conn, exc)
            raise

    @contextmanager
    def _transaction(
        self, context: TrustedContext, *, writable: bool, remaining_seconds: float | None, restoring: bool = False
    ) -> Iterator[Transaction]:
        self._context_check(context)
        conn = self._begin_transaction(context, writable, remaining_seconds, restoring)
        tx = Transaction(conn, context, writable=writable)
        original = None
        mirror = False
        try:
            # A restore can establish its fence while this connection waits
            # for the writer lease. Recheck after acquiring the transaction.
            if not restoring and restore_pending(self.binding.data_directory):
                raise ContractError("RESTORE_UNVERIFIED")
            with watch_mirrored_tombstones(conn) as touched:
                yield tx
            if writable:
                tx._assert_committable()
                conn.commit()
                mirror = bool(touched)
            else:
                conn.rollback()
        except BaseException as exc:
            original = exc
            try:
                if conn.in_transaction:
                    conn.rollback()
            except BaseException as cleanup:
                _cleanup_error(exc, cleanup, "rollback")
            raise
        finally:
            tx._finish()
            self._close(conn, original)
        if mirror:
            self._after_commit_deletions()

    def _claim_index_pending(self, conn, restoring: bool, version: int) -> bool:
        """Keep partially backfilled stores behind initialization admission, including after interruption."""
        pending = not restoring and version == SCHEMA_VERSION and lexical_index.claim_backfill_pending(conn)
        if pending and not self.upgrade_on_open:
            self._defer_upgrade()
        return pending

    def _after_commit_deletions(self) -> None:
        """Try bounded mirroring after lease cleanup; I/O errors leave durable catch-up."""
        try:
            from .deletion_mirror import mirror_committed_deletions

            mirror_committed_deletions(self.binding.data_directory, self.path)
        except Exception:  # noqa: BLE001
            pass

    def write(self, context: TrustedContext, *, remaining_seconds: float | None = None):
        return self._transaction(context, writable=True, remaining_seconds=remaining_seconds)

    @contextmanager
    def capture_privacy(self, context: TrustedContext, *, remaining_seconds: float = 1.0):
        """Read only the recognized privacy contract while ordinary storage is refusing an upgrade.

        The header and recorded schema must agree, as must identity, location, scopes and installation kind. An
        exact privacy table and index contract are required; an unknown format never authorizes a local copy.
        Restore fences remain active. No general source read or write is exposed by this transaction.
        """
        deadline = time.monotonic() + remaining_seconds
        with ExitStack() as opened:
            try:
                ordinary = opened.enter_context(self.read(context, remaining_seconds=remaining_seconds))
            except ContractError as exc:
                if exc.code != "SCHEMA_UNSUPPORTED":
                    raise
            else:
                _check_privacy_contract(ordinary._check())
                ordinary._check().set_authorizer(_privacy_read_authorizer)
                yield ordinary
                return
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            raise ContractError("DEADLINE_EXCEEDED")
        self._context_check(context)
        conn = self._open("ro", remaining_seconds, coordinating=True)
        tx = Transaction(conn, context, writable=False)
        try:
            conn.execute("BEGIN")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.verify(conn, expected_schema=version)
            _check_privacy_contract(conn)
            if restore_pending(self.binding.data_directory):
                raise ContractError("RESTORE_UNVERIFIED")
            conn.set_authorizer(_privacy_read_authorizer)
            yield tx
        finally:
            original = sys.exception()
            tx._finish()
            try:
                if conn.in_transaction:
                    conn.rollback()
            finally:
                self._close(conn, original)

    def read(self, context: TrustedContext, *, remaining_seconds: float | None = None):
        return self._transaction(context, writable=False, remaining_seconds=remaining_seconds)

    def truncate_wal(self, context: TrustedContext, *, remaining_seconds: float, restoring: bool = False) -> bool:
        """Copy the write-ahead log into the store and truncate it; whether that was done.

        A commit's page images go to ``memory.sqlite3-wal`` and the store's pages keep their earlier images until a
        checkpoint copies the new ones over them; a frame stays in the log until it is written over or the log is
        truncated.  Done, no frame from before this call is left, and the commits after it start the log again.  A
        reader still on an older snapshot holds the truncation back: SQLite waits for it within this connection's
        timeout, then reports it busy.  The writer lease is held meanwhile, as for a write.  ``restoring`` is the
        replay's (``core/restore.py``), which truncates the log while admission is still closed.
        """
        self._context_check(context)
        conn = self._open("rw", remaining_seconds, restoring=restoring)
        try:
            busy = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        except BaseException as exc:
            self._close(conn, exc)
            raise
        self._close(conn, None)
        return busy == 0
