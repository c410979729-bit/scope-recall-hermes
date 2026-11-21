"""Collect and combine exact, lexical, dated and vector candidates under one deadline."""

from __future__ import annotations

import inspect
import sqlite3
import time
from dataclasses import dataclass, field, replace
from itertools import islice
from typing import Protocol, cast

from ..contracts import ContractError
from .deadline import RequestDeadline, using_request_deadline
from .events import lexical_terms
from .owner_view import OwnerView, owner_names, owner_view
from .recall_needs import statement_text
from .recall_policy import RecallPolicy, hard_identifiers, meaningful_query_terms, rrf_score
from .recall_trace import note_candidate
from .retrieval import CandidateRef, SearchContext, SearchLimits
from .retrieval_storage import RetrievalStorage
from .vector_failure import note_embedding_failure, vector_failure_label

#: Claims offered per round.  A packet holds at most 30 items and a default one
#: six; a claim that answers competes for those slots on rank, so a deep pool
#: of claims would only spend hydration on statements that cannot be shown.
CLAIM_CANDIDATES = 16


#: SQLite virtual-machine steps between two looks at the recall's deadline while a candidate statement runs: one
#: every 10-60 ms of statement time.  Each look takes the GIL back from the statement, and beside a busy thread (the
#: MCP server's hook threads, a gateway's) waits for it: one every 10,000 steps made a 25 ms statement take 1.4 s.
_INTERRUPT_STEPS = 1_000_000


#: ``vector_owner`` is the vector channel's search for a question that names the owner, asked in their first person
#: (``core/owner_view.py``); it has slots of its own, so the question as asked keeps all of the channel's.
_CHANNELS = ("scoped", "exact", "lexical", "dated", "claim", "recent", "vector", "vector_owner")


#: Messages the scoped channel offers: a day of one entry, for a question that asks what was said then
#: (``recall_scope``).
SCOPED_CANDIDATES = 12


#: Sources the dated channel offers, as many as the scoped channel: said within the time a question names, holding
#: what else it asks (``RetrievalStorage.dated``).
DATED_CANDIDATES = 12


#: What a scoped candidate's fused score is multiplied by.  The question asked only what was said on its day: that
#: day's messages go before a candidate one channel found elsewhere by the question's words or the entry's name,
#: which on a shared store would fill the packet ("Claude Code" is in thousands of other messages).  Three puts a
#: first rank above any candidate two channels agree on at their first rank.
SCOPED_WEIGHT = 3.0


#: Vector admission reasons that are reported, and how; the rest are silent.
_VECTOR_REJECTION_GAPS = {
    "embedding_space_mismatch": "vector_old_or_mismatched_space",
    "vector_threshold_unconfigured": "vector_threshold_unconfigured",
}


_VECTOR_SILENT_REJECTIONS = frozenset({"vector_below_threshold", "vector_id_missing", "vector_score_invalid"})


def _note_below_floor(gaps: list[str], channel: str) -> None:
    """Accumulate content-free counts of channel candidates below automatic admission."""
    prefix = f"auto_relevance_below_floor:{channel}:"
    for index, gap in enumerate(gaps):
        if gap.startswith(prefix):
            gaps[index] = prefix + str(int(gap[len(prefix) :]) + 1)
            return
    gaps.append(prefix + "1")


def _takes_also(prefetch) -> bool:
    """Whether a vector port's ``prefetch_query`` asks for other readings of the question in the same request."""
    try:
        parameters = inspect.signature(prefetch).parameters
    except (TypeError, ValueError):
        return False
    return "also" in parameters or any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values())


def _asked_early(prefetched) -> tuple[str, ...]:
    """The questions an embedding asked for as the recall started holds: one, or several asked together."""
    if prefetched is None:
        return ()
    return tuple(getattr(prefetched, "queries", None) or (prefetched.query,))


class RetrievalClock(Protocol):
    def monotonic(self) -> float: ...


class ChannelBudget:
    """Candidate slots left per channel across follow-up rounds.

    A round may not spend what later rounds need: ``hold`` reserves one slot
    per remaining round in every channel so a directed follow-up always has
    room, even after rejected-by-dedup work consumed the first round.
    """

    def __init__(self, limits: SearchLimits) -> None:
        self.left = {
            "scoped": SCOPED_CANDIDATES,
            "exact": limits.candidate_pool,
            "lexical": limits.candidate_pool,
            "dated": DATED_CANDIDATES,
            "claim": min(CLAIM_CANDIDATES, limits.candidate_pool),
            "recent": limits.recent_items,
            "vector": limits.vector_limit,
            "vector_owner": limits.vector_limit,
        }
        self.total = sum(self.left.values())
        self.reserve = dict.fromkeys(_CHANNELS, 0)

    def hold(self, future_rounds: int) -> None:
        # A follow-up round never reads the day or the time a question names (``RetrievalPipeline._collect_rounds``).
        self.reserve = {
            channel: 0 if channel in {"scoped", "dated"} else min(future_rounds, self.left[channel])
            for channel in _CHANNELS
        }

    def allowance(self, channel: str, limit: int) -> int:
        free_total = self.total - sum(self.reserve.values())
        return min(limit, max(0, self.left[channel] - self.reserve[channel]), max(0, free_total))

    def spend(self, channel: str, count: int) -> None:
        self.left[channel] -= count
        self.total -= count


@dataclass
class _VectorPool:
    """What one collection round's vector search ranked, admitted or not (``RetrievalPipeline._vector_candidates``)."""

    searched_scopes: set[str] = field(default_factory=set)
    keys: set[tuple[str, str, int]] = field(default_factory=set)

    def note(self, context, candidates, searched_scopes, gaps):
        """Record returned hits and fully checked scopes; report incomplete coverage without dropping partial hits."""
        # Ordinary ports promise a complete search. Runtime ports additionally
        # report the scopes whose physical partitions all completed.
        allowed = context.trusted_context.allowed_scope_ids
        checked = allowed if searched_scopes is None else searched_scopes
        self.searched_scopes.update(checked)
        self.keys.update(candidate.key for candidate in candidates)
        if checked != allowed:
            gaps.append("vector_search_incomplete")


class RecallChannels:
    """Collect and qualify candidates using the configured storage and vector channels."""

    def __init__(
        self,
        storage,
        *,
        vector_port=None,
        policy: RecallPolicy | None = None,
        clock: RetrievalClock | None = None,
        storage_reader: RetrievalStorage | None = None,
    ):
        self.storage = storage
        self.vector_port = vector_port
        self.policy = policy if policy is not None else RecallPolicy(vector_threshold=None)
        self.clock = clock if clock is not None else time
        self.storage_reader = storage_reader if storage_reader is not None else RetrievalStorage(clock=self.clock)
        if hasattr(self.storage_reader, "clock"):
            self.storage_reader.clock = self.clock

    def _remaining(self, context: SearchContext) -> float:
        return context.deadline - self.clock.monotonic()

    def _deadline(self, context: SearchContext) -> RequestDeadline:
        """Bind the absolute cutoff to the injected clock, so storage and channel waits never read wall time."""
        return RequestDeadline.from_absolute(context.deadline, clock=self.clock.monotonic)

    # -- candidate channels ---------------------------------------------------

    def _prefetch_query(self, context: SearchContext):
        """Start query embeddings before SQLite collection, including the owner variant when supported."""
        if self.vector_port is None or context.limits.vector_limit == 0:
            return None
        prefetch = getattr(self.vector_port, "prefetch_query", None)
        if not callable(prefetch):
            return None
        view = self._owner_view(context)
        try:
            if view is not None and _takes_also(prefetch):
                return prefetch(context, also=(view.first,))
            return prefetch(context)
        except Exception:
            return None

    def _owner_view(self, context: SearchContext) -> OwnerView | None:
        """Return a first-person query when the owner is named and the rest of the query has content."""
        names = owner_names(self.policy.owner_names, context.trusted_context.source_principal)
        view = owner_view(context.query, names) if names else None
        return view if view is not None and meaningful_query_terms(view.bare) else None

    def _admission_query(self, context: SearchContext) -> str:
        """The question coverage reads: without the owner's name, which their own words never carry, so that it is no
        word an answer in them is short of."""
        view = self._owner_view(context)
        return view.bare if view is not None else context.query

    def _lexically_admitted(self, candidate: CandidateRef, context: SearchContext) -> bool:
        """Admitted by the question as asked, which a reply naming the owner beside its other words still meets, or
        without the owner's name: their own words never carry it, and in a short question it would be one of the two
        words a candidate has to match.  The name alone admits nothing either way."""
        if self.policy.lexical_admission(candidate, context.query, exact=False)[0]:
            return True
        view = self._owner_view(context)
        return view is not None and self.policy.lexical_admission(candidate, view.bare, exact=False)[0]

    def _vector_candidates(
        self,
        context: SearchContext,
        gaps: list[str],
        *,
        budget: ChannelBudget | None = None,
        prefetched=None,
        channel: str = "vector",
        pool: _VectorPool | None = None,
    ) -> tuple[CandidateRef, ...]:
        """The vector hits admitted against the recall's best hit; ``pool`` also learns every hit ranked."""
        if (
            self.vector_port is None
            or not getattr(self.vector_port, "available", True)
            or context.limits.vector_limit == 0
        ):
            gaps.append("vector_unavailable")
            return ()
        limit = context.limits.vector_limit
        if budget is not None:
            limit = budget.allowance(channel, limit)
        if limit <= 0:
            return ()
        remaining = self._remaining(context)
        if remaining <= 0:
            gaps.append("deadline_exceeded_vector")
            return ()
        # Keep part of the caller's deadline for SQLite hydration and the
        # packet's final authority checks if the optional vector call times out.
        vector_context = replace(context, deadline=context.deadline - remaining * 0.25)
        # The embedding asked for when the recall started is this round's only if the round kept its query.
        early = {"prefetched": prefetched} if context.query in _asked_early(prefetched) else {}
        searched_scopes: set[str] = set()
        tracked = getattr(self.vector_port, "reports_searched_scopes", False)
        coverage = {"searched_scopes": searched_scopes} if tracked else {}
        try:
            raw = tuple(
                islice(
                    iter(
                        self.vector_port.search(
                            vector_context, limit=limit, remaining_seconds=remaining * 0.75, **early, **coverage
                        )
                        or ()
                    ),
                    limit,
                )
            )
        except Exception as exc:
            note_embedding_failure(exc)
            # The class alone does not say what went wrong; see core/vector_failure.py.
            gaps.append("vector_unavailable")
            gaps.append(f"vector_error:{vector_failure_label(exc)}")
            if self._remaining(context) <= 0:
                gaps.append("deadline_exceeded_vector")
            return ()
        if budget is not None:
            budget.spend(channel, len(raw))
        if pool is not None:
            pool.note(context, raw, searched_scopes if tracked else None, gaps)
        admitted = []
        # The current message is never recalled, so it neither sets the best hit nor takes a place in the ranking.
        current = context.current_source_refs
        ranked = [
            candidate
            for candidate in raw
            if f"{candidate.ref}@{candidate.revision}" not in current
            and candidate.embedding_space == self.policy.embedding_space_id
        ]
        best = self.policy.vector_best(ranked)
        # Ranked hits below the threshold join automatic injection only: an explicit search asks for ten items, and
        # the hits they added there pushed answers other channels found further down.
        place = (
            {candidate.key: index for index, candidate in enumerate(ranked, 1)} if context.automatic_injection else {}
        )
        for rank, candidate in enumerate(raw, 1):
            if candidate.rank != rank:
                candidate = replace(candidate, rank=rank)
            accepted, reason = self.policy.vector_admission(candidate, best=best, rank=place.get(candidate.key))
            note_candidate(candidate, None if accepted else "threshold")
            if accepted:
                admitted.append(candidate)
                continue
            if context.automatic_injection and reason == "vector_below_threshold":
                _note_below_floor(gaps, "vector")
                continue
            gap = _VECTOR_REJECTION_GAPS.get(reason)
            if gap is None and reason not in _VECTOR_SILENT_REJECTIONS:
                gap = f"vector_rejected:{reason}"
            if gap is not None:
                gaps.append(gap)
        return tuple(admitted)

    def _admit(self, candidate: CandidateRef, context: SearchContext) -> bool:
        if candidate.kind == "event" and f"{candidate.ref}@{candidate.revision}" in context.current_source_refs:
            return False
        if candidate.source == "vector":
            # Its channel admitted it against the recall's best hit and its rank there (``_vector_candidates``).
            return True
        if candidate.source in {"exact_ref", "relation", "claim_lexical", "scoped", "dated"}:
            # A claim candidate already passed its own statement rule in storage; a scoped one is a message of
            # the day and entry the question names, and a dated one a source of the time it names that holds what
            # else it asks.
            return True
        return self._lexically_admitted(candidate, context)

    def _fuse_candidates(
        self, admitted: list[CandidateRef], seen: set[tuple[str, str, int]] | None
    ) -> tuple[CandidateRef, ...]:
        by_key: dict[tuple[str, str, int], list[CandidateRef]] = {}
        for candidate in admitted:
            if seen is not None and candidate.key in seen:
                continue
            by_key.setdefault(candidate.key, []).append(candidate)
        seeds: list[CandidateRef] = []
        for _key, signals in sorted(by_key.items()):
            representative = min(signals, key=lambda item: (item.rank, item.source))
            fusion = rrf_score((item.rank for item in signals), k=self.policy.rrf_k)
            scoped = [item for item in signals if item.source == "scoped"]
            if scoped:
                fusion *= SCOPED_WEIGHT
            # A message of the day or the time the question named stays one, whichever channel ranked it higher; a
            # ref the caller named keeps its place first.
            named = scoped or [item for item in signals if item.source == "dated"]
            if named and representative.source != "exact_ref":
                representative = named[0]
            seeds.append(replace(representative, fusion_score=fusion))
        return tuple(seeds)

    def _automatic_floor(self, tx, candidate: CandidateRef, context: SearchContext, gaps: list[str]) -> bool:
        """Apply the channel floor before fusion and relation expansion.

        Storage reports lexical matches, including credited synonyms. Ports
        without those matches are checked against their hydrated statement.
        Rejections count channel candidates, once per collection round.
        """
        question = self._admission_query(context)
        if candidate.source == "dated" and context.time is not None:
            question = context.time.rest
        if candidate.source not in {"vector", "exact_ref", "scoped", "relation"}:
            if candidate.matched_query_terms is None:
                obj = self.storage_reader.hydrate(tx, candidate, context)
                hits = meaningful_query_terms(question)
                matched = set(hits).intersection(lexical_terms(statement_text(obj))) if obj is not None else set()
                candidate = replace(candidate, matched_query_terms=tuple(sorted(matched)))
        if self.policy.automatic_admission(candidate, question):
            return True
        note_candidate(candidate, "threshold")
        _note_below_floor(gaps, candidate.source)
        return False

    def _qualified_candidates(
        self, tx, raw: list[CandidateRef], context: SearchContext, gaps: list[str], pool: _VectorPool | None = None
    ):
        """Require a passing channel per object, retaining its admitted ranking signals.

        A strong semantic hit can also have a weak lexical signal. Its position
        still uses both ranks; that weak signal cannot qualify another object.
        Each object with no legacy-admitted channel uses its passing signals
        directly, so an unrelated candidate cannot suppress that fallback.
        For a named time window, the dated channel owns relaxed lexical
        admission; ordinary fallback would bypass that window.  Lexical hits
        a completed scope's vector search passed over are dropped (``_outside_vector_pool``).
        """
        for candidate in raw:
            note_candidate(
                candidate,
                "current_prompt" if f"{candidate.ref}@{candidate.revision}" in context.current_source_refs else None,
            )
        admitted = [candidate for candidate in raw if self._admit(candidate, context)]
        if not context.automatic_injection:
            return admitted
        passing = [
            candidate
            for candidate in raw
            if f"{candidate.ref}@{candidate.revision}" not in context.current_source_refs
            and self._automatic_floor(tx, candidate, context, gaps)
        ]
        keys = {candidate.key for candidate in passing}
        qualified = [candidate for candidate in admitted if candidate.key in keys]
        if context.time is None:
            qualified_keys = {candidate.key for candidate in qualified}
            qualified += [candidate for candidate in passing if candidate.key not in qualified_keys]
        outside = self._outside_vector_pool(tx, qualified, context, pool)
        return [candidate for candidate in qualified if candidate.source != "lexical" or candidate.key not in outside]

    def _outside_vector_pool(
        self, tx, candidates: list[CandidateRef], context: SearchContext, pool: _VectorPool | None
    ) -> frozenset[tuple[str, str, int]]:
        """Embedded assistant lexical hits absent from the finite vector pool in fully searched scopes.

        Absence is a ranking signal, not proof of irrelevance: a question about an earlier assistant reply can lose
        its answer here. Explicit lookup retains it. Unchecked scopes cannot supply this negative signal.

        The owner's own messages stay: an aggregate question ("how many times did I ...") is answered by many of
        them that its words find and its meaning ranks too low.  So does a reply not embedded yet (a message said a
        moment ago is found by its words), and every hit when the vector search did not run or the question names an
        identifier, which words find and meaning does not.
        """
        reader = getattr(self.storage_reader, "embedded_replies", None)
        if pool is None or not pool.searched_scopes or not callable(reader):
            return frozenset()
        loose = [
            candidate for candidate in candidates if candidate.source == "lexical" and candidate.key not in pool.keys
        ]
        if not loose or hard_identifiers(self._admission_query(context)):
            return frozenset()
        outside = cast(frozenset[tuple[str, str, int]], reader(tx, loose, searched_scopes=pool.searched_scopes))
        for candidate in loose:
            if candidate.key in outside:
                note_candidate(candidate, "outside_vector_pool")
        return outside

    def _collect(
        self,
        tx,
        context: SearchContext,
        gaps: list[str],
        *,
        seen: set[tuple[str, str, int]],
        budget: ChannelBudget,
        prefetched=None,
        on_ready=None,
    ) -> tuple[CandidateRef, ...]:
        """One round of exact, lexical, recent and vector candidates, fused by identity."""
        if self._remaining(context) <= 0:
            gaps.append("deadline_exceeded_collect")
            return ()
        raw: list[CandidateRef] = []
        pool = _VectorPool()

        def admitted(*, quiet: bool = False) -> tuple[CandidateRef, ...]:
            # Deadline exits still pass every collected candidate through the
            # same current-source and channel qualification gate.  A quiet pass
            # re-qualifies what an earlier pass already counted the rejections of.
            accepted = self._qualified_candidates(tx, raw, context, [] if quiet else gaps, pool)
            return self._fuse_candidates(accepted, seen)

        channels = (
            # First, so a slow statement after it cannot cost the one channel that reads what the question names.
            (
                "scoped",
                getattr(self.storage_reader, "scoped", None) if context.scope is not None else None,
                SCOPED_CANDIDATES,
            ),
            ("exact", self.storage_reader.exact, context.limits.candidate_pool),
            ("lexical", self.storage_reader.lexical, context.limits.candidate_pool),
            (
                "dated",
                getattr(self.storage_reader, "dated", None) if context.time is not None else None,
                DATED_CANDIDATES,
            ),
            # A reader predating the claim channel simply offers no claims.
            ("claim", getattr(self.storage_reader, "claims", None), CLAIM_CANDIDATES),
            ("recent", self.storage_reader.recent, context.limits.recent_items),
        )
        # A statement cannot see the recall's deadline: with no planner statistics the lexical one ran 18-21 s for a
        # long Telegram message, and the recall came back empty long after its deadline.  Past the
        # deadline a channel's statement is interrupted: that channel gives nothing, the gap says so, and what the
        # channels before it found still answers.  Only here: interrupted while hydrating, a recall would lose what
        # it had already found.
        check = getattr(tx, "_check", None)
        conn = check() if callable(check) else None
        interruptible = callable(getattr(conn, "set_progress_handler", None))
        if interruptible:
            conn.set_progress_handler(lambda: 1 if self._remaining(context) <= 0 else 0, _INTERRUPT_STEPS)
        try:
            for channel, loader, limit in channels:
                if loader is None:
                    continue
                if self._remaining(context) <= 0:
                    gaps.append("deadline_exceeded_collect")
                    return admitted()
                allowance = budget.allowance(channel, limit)
                if allowance <= 0:
                    continue
                values = tuple(loader(tx, context, limit=allowance))
                budget.spend(channel, len(values))
                raw.extend(values)
        except ContractError:
            raise
        except sqlite3.OperationalError as exc:
            # Only the interrupt is the deadline; a locked store past it is still reported as the error it is.
            if interruptible and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                gaps.append("deadline_exceeded_collect")
                return admitted()
            gaps.append(f"sqlite_candidate_error:{type(exc).__name__}")
        except Exception as exc:
            gaps.append(f"sqlite_candidate_error:{type(exc).__name__}")
        finally:
            if interruptible:
                conn.set_progress_handler(None, 0)
        if self._remaining(context) <= 0:
            gaps.append("deadline_exceeded_collect")
            return admitted()
        return self._collect_optional(
            context, gaps, raw, budget, prefetched=prefetched, admitted=admitted, on_ready=on_ready, pool=pool
        )

    def _collect_optional(self, context, gaps, raw, budget, *, prefetched, admitted, on_ready, pool=None):
        """Hydrate ready channels before an optional wait, then merge any added signals."""
        ready = admitted()
        if on_ready is not None:
            cutoff = on_ready(ready)
            if cutoff is not None:
                context = replace(context, deadline=min(context.deadline, cutoff))
        if self._remaining(context) <= 0:
            gaps.append("deadline_exceeded_collect")
            return ready
        with using_request_deadline(self._deadline(context)):
            optional = list(self._vector_candidates(context, gaps, budget=budget, prefetched=prefetched, pool=pool))
            view = self._owner_view(context)
            searchable = self.vector_port is not None and context.limits.vector_limit > 0
            if view is not None and searchable and self._remaining(context) > 0:
                # The question as the owner would ask it, beside the question as asked: their own words answer it
                # and never
                # carry their name, while the question as asked finds what names them.  One candidate the two searches
                # agree on ranks by both.
                owned = replace(context, query=view.first)
                optional.extend(
                    self._vector_candidates(
                        owned, gaps, budget=budget, prefetched=prefetched, channel="vector_owner", pool=pool
                    )
                )
        raw.extend(optional)
        if optional:
            return admitted()
        # A search that ran re-qualifies even with nothing admitted: what it ranked decides which lexical hits stay.
        return admitted(quiet=True) if pool is not None and (pool.searched_scopes or pool.keys) else ready
