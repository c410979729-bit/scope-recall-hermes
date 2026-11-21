"""Read and write access within one SQLite transaction, including scope checks and per-read caches."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..contracts import ContractError, TrustedContext
from .delete_storage import Deletions
from .failure_policy import failure_kind
from .failure_retry import retry_class
from .registry_storage import Registry
from .source_records import SOURCE_COLUMNS, StoredSource, source_size, stored_source
from .source_storage import Sources
from .visibility import allowed, allowed_refs
from .work_storage import WorkItems


def _most_frequent(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    """The sixteen largest counts, largest first."""
    return tuple(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:16])


def _cleanup_error(original: BaseException, cleanup: BaseException, stage: str) -> None:
    original.add_note(f"SQLite {stage} cleanup failed: {type(cleanup).__name__}; connection discarded")
    errors = getattr(original, "cleanup_errors", ())
    original.__dict__["cleanup_errors"] = (*errors, cleanup)
    if original.__cause__ is None:
        original.__cause__ = cleanup


@dataclass(frozen=True)
class StoreStatus:
    schema_version: int
    memory_epoch: int
    config_version: int
    sources: int
    pending_work: int
    failed_work: int = 0
    leased_work: int = 0
    oldest_pending_at: str | None = None
    #: Why the failed rows failed, one count per failure kind (``failure_policy.failure_kind``).
    work_error_counts: tuple[tuple[str, int], ...] = ()
    source_only_sources: int | None = None
    deferred_sources: int | None = None
    oldest_deferred_at: str | None = None
    #: The failed rows ``failure_retry.retry_class`` calls terminal: by design, cleared by no retry.
    terminal_failed_work: int = 0
    #: The error a queued row kept from its last attempt.  It is waiting, not failed, so it is told apart from the
    #: failures and counted in none of them.
    pending_error_counts: tuple[tuple[str, int], ...] = ()


#: Source versions ``Transaction.prefetch_sources`` loads per statement.
_PREFETCH_PAGE = 400
#: Read-transaction memo limits (``Transaction.remember``): entry count and total text size, counted in characters
#: rather than encoded bytes.  Values beyond either bound are used without being cached.
_MEMO_ENTRIES = 16384
_MEMO_BYTES = 32 << 20


class Transaction:
    """Scoped repository operations; no public connection or SQL execution surface."""

    def __init__(self, connection: sqlite3.Connection, context: TrustedContext, *, writable: bool) -> None:
        self.__connection = connection
        self.context = context
        self.__writable = writable
        self.__active = True
        self.__poisoned = False
        self.__savepoint_sequence = 0
        self.__entry_labels: dict[str, dict[str, str]] | None = None
        #: What a read transaction loaded (``remembered``), and the size of the text it keeps; ``None`` in a write
        #: transaction.
        self.__memo: dict | None = None if writable else {}
        self.__memo_bytes = 0
        self.__parent: Transaction | None = None
        self.__authorized_sources: dict = {}
        # Release diagnostics are semantic state, independent of the bounded
        # read memo: disabling or filling that cache must not erase omissions.
        self.withheld_evidence: set[tuple[str, int]] = set()

    def with_context(self, context: TrustedContext, *, writable: bool = False) -> Transaction:
        """Borrow this transaction for an authorized partition, without widening its scopes or lifetime."""
        connection = self._check(write=writable)
        if context.binding != self.context.binding or not context.allowed_scope_ids <= self.context.allowed_scope_ids:
            raise ContractError("ACCESS_DENIED", "transaction_scope")
        view = Transaction(connection, context, writable=writable)
        view.__parent = self
        if self.__writable:
            view.__memo = None
        return view

    def entry_label(self, entry_id: str) -> dict[str, str] | None:
        """The ``{id, name}`` a reader is shown for a source's entry, or None.

        None in a local store, whose rows all say ``local`` and whose recall
        output is exactly what it was.  Read once per transaction.
        """
        if self.context.binding.installation_kind != "shared":
            return None
        if self.__entry_labels is None:
            self.__entry_labels = {
                key: {"id": key, "name": value["name"]} for key, value in self.registry.entries().items()
            }
        label = self.__entry_labels.get(entry_id)
        return dict(label) if label is not None else None

    def _check(self, *, write: bool = False) -> sqlite3.Connection:
        if self.__parent is not None:
            self.__parent._check(write=write)
        if not self.__active or self.__poisoned:
            raise ContractError("STORAGE_UNAVAILABLE", "transaction_closed")
        if write and not self.__writable:
            raise ContractError("ACCESS_DENIED", "read_only")
        return self.__connection

    @property
    def deletions(self):
        return Deletions(self)

    @property
    def episodes(self):
        from .episode_storage import Episodes  # on first use: most of a hook's processes never need it

        return Episodes(self)

    @property
    def artifacts(self):
        from .artifact_storage import Artifacts  # on first use: most of a hook's processes never need it

        return Artifacts(self)

    @property
    def sources(self):
        return Sources(self)

    @property
    def registry(self):
        return Registry(self)

    def _entries_changed(self) -> None:
        """An entry was registered or renamed in this transaction: ``entry_label`` reads them again."""
        self.__entry_labels = None

    @property
    def references(self):
        from .reference_storage import References  # on first use: most of a hook's processes never need it

        return References(self)

    def _finish(self) -> None:
        self.__active = False
        # What it loaded goes with it, also when a traceback keeps the transaction.
        if self.__memo is not None:
            self.__memo = {}
            self.__memo_bytes = 0

    def _assert_committable(self) -> None:
        self._check(write=True)

    @property
    def claims(self):
        from .claim_storage import Claims  # on first use: most of a hook's processes never need it

        self._check()
        return Claims(self)

    @property
    def work(self):
        self._check()
        return WorkItems(self)

    @property
    def candidates(self):
        from .candidate_storage import CandidateLifecycle  # on first use: most of a hook's processes never need it

        self._check()
        return CandidateLifecycle(self)

    @contextmanager
    def savepoint(self) -> Iterator[Transaction]:
        """Borrow the owning transaction; only this boundary manages savepoint SQL."""
        conn = self._check(write=True)
        self.__savepoint_sequence += 1
        name = f"core_{self.__savepoint_sequence}"
        active = False
        try:
            conn.execute(f"SAVEPOINT {name}")
            active = True
            yield self
            conn.execute(f"RELEASE SAVEPOINT {name}")
            active = False
        except BaseException as original:
            if active and conn.in_transaction:
                try:
                    conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
                    conn.execute(f"RELEASE SAVEPOINT {name}")
                    active = False
                except BaseException as cleanup:
                    self.__poisoned = True
                    _cleanup_error(original, cleanup, "savepoint")
            raise

    def _scope(self, scope_id: str) -> None:
        if scope_id not in self.context.allowed_scope_ids:
            raise ContractError("ACCESS_DENIED")

    def memory_epoch(self) -> int:
        """Read the authority fence in this transaction without queue diagnostics."""
        return int(self._check().execute("SELECT memory_epoch FROM instance_meta WHERE singleton=1").fetchone()[0])

    def database_identity(self) -> tuple[str, str | None]:
        """Return this database's immutable identity and known creation time from its verified transaction."""
        row = (
            self._check()
            .execute("SELECT database_id,database_created_at FROM instance_meta WHERE singleton=1")
            .fetchone()
        )
        identity = row[0]
        if type(identity) is not str or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ContractError("IDENTITY_UNBOUND", "database_id")
        return identity, row[1]

    def status(
        self, *, include_all_projects: bool = False, include_admission: bool = False, include_queue_age: bool = True
    ) -> StoreStatus:
        conn = self._check()
        meta = conn.execute(
            "SELECT schema_version,memory_epoch,config_version FROM instance_meta WHERE singleton=1"
        ).fetchone()
        scopes = sorted(self.context.allowed_scope_ids)
        marks = ",".join("?" for _ in scopes)
        context_filter = "AND (project_id IS NULL OR project_id=?) AND (branch_id IS NULL OR branch_id=?)"
        params = (*scopes, self.context.project_id, self.context.branch_id)
        if include_all_projects:
            # Metadata-only installation diagnostics; scope isolation remains.
            context_filter, params = "", tuple(scopes)
        # The store's own size and the age of the oldest queued item are what an
        # operator reads; a pass reports its queue depth and its own items.  Both
        # walk every row, so a pass that asked for them paid for them once per
        # pass and more the fuller the queue was -- the wrong way round for a
        # report whose job is to say the queue is deep.
        source_count = 0
        if include_queue_age:
            source_count = conn.execute(
                f"SELECT count(*) FROM source_events WHERE read_blocked=0 AND scope_id IN ({marks}) {context_filter}",
                params,
            ).fetchone()[0]
        work_count = conn.execute(
            "SELECT count(*) FROM work_items WHERE state IN ('pending','leased') "
            f"AND scope_id IN ({marks}) {context_filter}",
            params,
        ).fetchone()[0]
        failed_count = conn.execute(
            f"SELECT count(*) FROM work_items WHERE state='failed' AND scope_id IN ({marks}) {context_filter}", params
        ).fetchone()[0]
        leased_count = conn.execute(
            f"SELECT count(*) FROM work_items WHERE state='leased' AND scope_id IN ({marks}) {context_filter}", params
        ).fetchone()[0]
        oldest = None
        if include_queue_age:
            oldest = conn.execute(
                f"""SELECT MIN(COALESCE((SELECT e.persisted_at FROM source_events e
                WHERE e.event_id=work_items.subject_ref
                      AND e.source_revision=work_items.subject_revision),available_at))
                FROM work_items WHERE state IN ('pending','leased') AND scope_id IN ({marks}) {context_filter}""",
                params,
            ).fetchone()[0]
        # Detailed source processing counts are diagnostic-only; avoid a JSON
        # scan of all sources on every internal epoch/queue status read.
        admission = (None, None, None)
        if include_admission:
            admission = conn.execute(
                f"""SELECT
            COALESCE(SUM(json_extract(extra_json,'$._scope_recall_admission.disposition')='source_only'),0),
            COALESCE(SUM(json_extract(extra_json,'$._scope_recall_admission.disposition')='deferred'),0),
            MIN(CASE WHEN json_extract(extra_json,'$._scope_recall_admission.disposition')='deferred'
                     THEN persisted_at END)
            FROM source_events WHERE read_blocked=0 AND suppressed=0 AND scope_id IN ({marks}) {context_filter}
            AND NOT EXISTS(SELECT 1 FROM source_events newer WHERE newer.source_group_key=source_events.source_group_key
                AND newer.source_revision>source_events.source_revision)
            AND NOT EXISTS(SELECT 1 FROM object_blocks b WHERE b.object_kind='event'
                AND b.object_ref=source_events.event_id AND (b.read_blocked=1 OR b.suppressed=1))""",
                params,
            ).fetchone()
        errors = conn.execute(
            f"""SELECT state,last_error_code,COUNT(*) AS n FROM work_items
            WHERE state IN ('pending','failed') AND last_error_code IS NOT NULL
            AND scope_id IN ({marks}) {context_filter}
            GROUP BY state,last_error_code""",
            params,
        ).fetchall()
        # A code carries its retry history (``auto_retry:1|derivation_invalid``) and its writer's case; each kind is
        # counted once, as ``failure_policy.failure_kind`` reads it, and a failure is terminal as ``retry_class`` and so
        # ``retry-failures`` read it.
        kinds: dict[str, dict[str, int]] = {"failed": {}, "pending": {}}
        terminal = 0
        for state, code, count in errors:
            kind = failure_kind(code)[:80]
            kinds[state][kind] = kinds[state].get(kind, 0) + int(count)
            if state == "failed" and retry_class(code) == "terminal":
                terminal += int(count)
        return StoreStatus(
            int(meta["schema_version"]),
            int(meta["memory_epoch"]),
            int(meta["config_version"]),
            int(source_count),
            int(work_count),
            int(failed_count),
            int(leased_count),
            oldest,
            _most_frequent(kinds["failed"]),
            admission[0],
            admission[1],
            admission[2],
            terminal_failed_work=terminal,
            pending_error_counts=_most_frequent(kinds["pending"]),
        )

    @contextmanager
    def _reuse_source(self, source):
        """Reuse an authorized immutable source during one admission operation only.

        The caller must already have read this version in this transaction.
        The narrow lifetime excludes later visibility-changing mutations.
        """
        self._check(write=True)
        key = (source.ref, source.revision)
        previous = self.__authorized_sources.get(key)
        self.__authorized_sources[key] = source
        try:
            yield source
        finally:
            if previous is None:
                self.__authorized_sources.pop(key, None)
            else:
                self.__authorized_sources[key] = previous

    def source(self, ref: str, revision: int) -> StoredSource | None:
        conn = self._check()
        if type(ref) is not str or not ref or len(ref) > 240 or type(revision) is not int or revision < 1:
            raise ContractError("INPUT_INVALID", "source_ref")
        if (ref, revision) in self.__authorized_sources:
            return self.__authorized_sources[(ref, revision)]
        if not allowed(self, "event", ref):
            return None
        loaded = self.remembered(
            ("source", ref, revision), lambda: self._source_row(conn, ref, revision), size=source_size
        )
        return None if loaded is None else stored_source(*loaded)

    def _source_row(self, conn, ref: str, revision: int):
        """The visible row of one source version and, for a part of a long message, how many parts of it are readable;
        ``None`` when it is not visible."""
        scopes = sorted(self.context.allowed_scope_ids)
        marks = ",".join("?" for _ in scopes)
        row = conn.execute(
            f"""SELECT {",".join(SOURCE_COLUMNS)} FROM source_events
                   WHERE event_id=? AND source_revision=? AND read_blocked=0 AND scope_id IN ({marks})
            AND (project_id IS NULL OR project_id=?) AND (branch_id IS NULL OR branch_id=?)""",
            (ref, revision, *scopes, self.context.project_id, self.context.branch_id),
        ).fetchone()
        if row is None:
            return None
        count = None
        if "segment" in json.loads(row["extra_json"]):
            count = conn.execute(
                "SELECT count(*) FROM source_events WHERE source_group_key=? AND source_revision=? AND read_blocked=0",
                (row["source_group_key"], row["source_revision"]),
            ).fetchone()[0]
        return row, count

    def prefetch_sources(self, pairs) -> None:
        """Load these source versions into a read transaction's memory (``remembered``), with whether each is its
        group's newest version and whether ``visibility.allowed`` admits it: two statements and two rows a page,
        however many there are.  A write transaction loads nothing ahead."""
        if self.__memo is None:
            return
        conn = self._check()
        wanted = [
            (ref, revision)
            for ref, revision in dict.fromkeys(pairs)
            if type(ref) is str
            and ref
            and len(ref) <= 240
            and type(revision) is int
            and revision >= 1
            and ("source", ref, revision) not in self.__memo
        ]

        scopes = sorted(self.context.allowed_scope_ids)
        # Unary ``+`` prevents the scope predicates from selecting a scope-index scan instead of primary-key lookups.
        for start in range(0, len(wanted), _PREFETCH_PAGE):
            page = wanted[start : start + _PREFETCH_PAGE]
            admitted = allowed_refs(self, "event", (ref for ref, _revision in page))
            fields = ",".join(f"'{column}',s.{column}" for column in SOURCE_COLUMNS)
            row = conn.execute(
                f"""SELECT json_group_array(json_object({fields},
                       'segment_count',CASE WHEN json_type(s.extra_json,'$.segment') IS NOT NULL THEN
                           (SELECT count(*) FROM source_events g WHERE g.source_group_key=s.source_group_key
                            AND g.source_revision=s.source_revision AND g.read_blocked=0) END,
                       'head',NOT EXISTS(SELECT 1 FROM source_events newer
                                         WHERE newer.source_group_key=s.source_group_key
                            AND newer.source_revision>s.source_revision)))
                    FROM source_events s WHERE (s.event_id,s.source_revision) IN ({",".join("(?,?)" for _ in page)})
                    AND +s.read_blocked=0 AND +s.scope_id IN ({",".join("?" for _ in scopes)})
                    AND (s.project_id IS NULL OR s.project_id=?) AND (s.branch_id IS NULL OR s.branch_id=?)""",
                (*(value for pair in page for value in pair), *scopes, self.context.project_id, self.context.branch_id),
            ).fetchone()
            found = {(item["event_id"], item["source_revision"]): item for item in json.loads(row[0])}
            for ref, revision in page:
                item = found.get((ref, revision))
                if ref not in admitted:
                    continue
                loaded = None if item is None else (item, item["segment_count"])
                self.remember(("source", ref, revision), loaded, size=source_size(loaded))
                if item is not None:
                    self.remember(("head", ref, revision), bool(item["head"]))

    def remembered(self, key: tuple, load, *, size=None):
        """``load()``'s answer for ``key``, memoized within a read transaction.

        A read transaction reads a fixed snapshot, so its loaded values remain valid
        until it ends.  Cached values can be reused within the memo's size bounds.
        Write transactions bypass memoization because their reads can change.
        """
        memo = self.__memo
        if memo is None:
            return load()
        if key in memo:
            return memo[key]
        value = load()
        self.remember(key, value, size=0 if size is None else size(value))
        return value

    def remember(self, key: tuple, value, *, size: int = 0) -> None:
        """Keep ``value`` for ``key`` in a read transaction's memory (``remembered``); nothing in a write transaction.

        A transaction that reads a whole store keeps no more than ``_MEMO_ENTRIES`` answers and ``_MEMO_BYTES`` of
        source text (``size``): past either, what it loads is used and not kept.
        """
        memo = self.__memo
        if memo is None or key in memo:
            return
        if len(memo) >= _MEMO_ENTRIES or self.__memo_bytes + size > _MEMO_BYTES:
            return
        memo[key] = value
        self.__memo_bytes += size

    @property
    def remembers(self) -> bool:
        """Whether this transaction keeps what it loads (``remembered``): a read transaction does."""
        return self.__memo is not None

    def knows(self, key: tuple) -> bool:
        """Whether a read transaction has already loaded ``key`` (``remembered``)."""
        return self.__memo is not None and key in self.__memo
