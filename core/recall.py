"""The one production retrieval pipeline used by automatic and tool recall."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..contracts import ContractError, utc_timestamp
from . import private
from .adoption_storage import adopted_source
from .auto_recall import automatic_recall_route
from .background_context import background_candidates, current_task_candidate
from .claim_storage import parse_source_ref
from .claims import canonical_time
from .coverage import note_truncation
from .deadline import current_request_deadline, using_request_deadline
from .duplicate_collapse import DistinctContent, note_duplicates
from .events import lexical_terms
from .model_content import withhold_secret_items
from .recall_budget import estimate_tokens, event_admission_order, release_reserve_seconds
from .recall_channels import CLAIM_CANDIDATES as CLAIM_CANDIDATES
from .recall_channels import DATED_CANDIDATES as DATED_CANDIDATES
from .recall_channels import SCOPED_CANDIDATES as SCOPED_CANDIDATES
from .recall_channels import SCOPED_WEIGHT as SCOPED_WEIGHT
from .recall_channels import ChannelBudget, RecallChannels
from .recall_channels import RetrievalClock as RetrievalClock
from .recall_diagnostics import RecallTimings
from .recall_needs import (
    CHOICE_MARKERS,
    Answers,
    answering,
    directed_followup_query,
    evidence_roots,
    mentions,
    statement_text,
    unmet_needs,
)
from .recall_overflow import visible_query_pairs
from .recall_policy import (
    RecallPolicy,
    asks_without_answering,
    identifiers_compatible,
    meaningful_query_terms,
    rrf_score,
    same_message,
)
from .recall_restatement import has_conversation_summaries, prefer_originals, repeated_report_claims
from .recall_scope import asks_what_held_before, asks_what_was_said, query_scope, query_time
from .recall_trace import note_candidate, note_removed, snapshot
from .retrieval import (
    CandidateRef,
    CollectionQuery,
    RetrievalResult,
    RetrievedObject,
    SearchContext,
    SearchLimits,
    current_prompt_refs,
    effective_limits,
    optional_json,
)
from .retrieval_hydration import conflicting_record, note_superseded_evidence, replacements
from .retrieval_storage import CollectionPage, RetrievalStorage


def _excluded_relation_source(tx, context, seed, candidate):
    """Keep held prompts and unselected assistant alternatives out of automatic evidence expansion.

    The adopted decision already contains its selected content and references.
    Explicit source reads retain the complete assistant reply for inspection.
    """
    if candidate.kind != "event":
        return False
    return f"{candidate.ref}@{candidate.revision}" in context.current_source_refs or (
        context.automatic_injection
        and seed.kind == "claim"
        and adopted_source(tx, seed.ref, seed.revision) == (candidate.ref, candidate.revision)
    )


#: Share of its fusion score context-only evidence keeps in live modes: a bare
#: question, or a reply restating what a recall tool returned in its turn.
#: Asked again, a question can return five earlier questions like it ahead of the
#: reply that answered one, and a report of a recall test can come back ahead of
#: the evidence it quoted.  At 0.6 a first-ranked one falls below a reply that
#: ranked in the mid-teens, and still ranks as context when nothing answers.
CONTEXT_ONLY_WEIGHT = 0.6
#: Meaningful terms a query needs before an older copy of it leads to what that copy was told
#: (``RetrievalPipeline._expand``).  A short command sent again asks nothing an old turn answered.  Chinese terms are
#: overlapping character pairs, so "继续执行" holds three and "按你说的做" four, and three would let both bring back
#: every old turn they opened; five is about six characters.  Of the owner's prompts over two weeks, 33 of the 115 with
#: fewer had an exact older copy; of the owner's real questions, 167 of 173 have five.
ECHO_TURN_MIN_TERMS = 5


#: What the fused score of a candidate said within the time its question names (``recall_scope.query_time``) is
#: multiplied by.  Notes on one subject from other days hold the question's words as well, and the newest ranked
#: first: 2.5 puts what one channel ranked within its first sixteen above any candidate of another time that two
#: channels agree on at their first rank.  Nothing is left out for its time, which may be the answer's.
IN_TIME_WEIGHT = 2.5


def _empty_result(context: SearchContext, gap: str) -> RetrievalResult:
    return RetrievalResult((), (), None, (gap,), "unknown", "unknown", 0, 0, request_id=context.request_id)


def _in_context(tx, obj: RetrievedObject, context: SearchContext) -> bool:
    """Whether everything ``obj`` stands on is this session's own and was said since ``context.in_context_since``,
    so that the asking host's model still holds it word for word.  Anything else is not: an item without evidence,
    one with any evidence of another session, from before that moment, or that cannot be read."""
    if not obj.evidence_refs or context.in_context_since is None:
        return False
    session = context.trusted_context.session_id
    for ref in obj.evidence_refs:
        try:
            source = tx.source(*parse_source_ref(ref))
            said = utc_timestamp(source.event.get("occurred_at")) if source is not None else None
        except ContractError:
            return False
        if source is None or source.session_id != session or said is None or said < context.in_context_since:
            return False
    return True


def _says_only_its_state(content: str) -> bool:
    """Whether an episode body is the placeholder a never-consolidated resume leaves."""
    payload = optional_json(content)
    return isinstance(payload, dict) and set(payload) <= {"state"}


def _said_after(obj: RetrievedObject, moment: str) -> bool:
    """Whether ``obj`` was said after ``moment`` (``canonical_time``), as far as its time tells: an event's own, a
    claim's newest evidence's.  Without one it is taken as said before."""
    stamp = dict(obj.metadata).get("occurred_at")
    try:
        return (canonical_time(stamp) or "") > moment
    except ContractError:
        return False


def _identifiers_match(
    tx, candidate: CandidateRef, context: SearchContext, obj: RetrievedObject, question: str
) -> bool:
    """A scoped hit need not repeat the entry or date. Narrower follow-ups still target the original link."""
    query = context.query
    if candidate.source == "scoped" and context.scope is not None:
        query = context.scope.rest
    elif candidate.source == "dated" and context.time is not None:
        query = context.time.rest
    return identifiers_compatible(
        query, obj.content, numbered_urls=context.automatic_injection and not context.focus_refs
    ) and not conflicting_record(tx, context, obj, query=question)


def _said_within(obj: RetrievedObject, windows: tuple[tuple[str, str], ...]) -> bool:
    """Whether ``obj`` was said within one of ``windows`` (``canonical_time`` bounds, ``[start, end)``), as far as its
    time tells (``_said_after``).  Without one it is not."""
    try:
        moment = canonical_time(dict(obj.metadata).get("occurred_at"))
    except ContractError:
        return False
    return moment is not None and any(start <= moment < end for start, end in windows)


def _weighed_by_time(
    hydrated: list[tuple[CandidateRef, RetrievedObject]], context: SearchContext
) -> list[tuple[CandidateRef, RetrievedObject]]:
    """``hydrated`` with each candidate said within the time the question names (``SearchContext.time``) weighed by
    ``IN_TIME_WEIGHT``.  The others keep their scores, and nothing changes for a question that names no time or a read
    that leaves it unread (``RetrievalPipeline.search``)."""
    if context.time is None:
        return hydrated
    windows = tuple((utc_timestamp(start), utc_timestamp(end)) for start, end in context.time.windows)
    return [
        (replace(candidate, fusion_score=candidate.fusion_score * IN_TIME_WEIGHT), obj)
        if _said_within(obj, windows)
        else (candidate, obj)
        for candidate, obj in hydrated
    ]


#: See ``RetrievalPipeline.RELATION_WEIGHT``.
RELATION_WEIGHT = 0.5


@dataclass
class _Aside:
    """What hydration sets aside rather than delivers (``RetrievalPipeline._hydrate_admit``): older copies of the
    current message, which lead to their turn's replies, and what the asking model still holds, which leads on to
    what is related to it."""

    echoes: list[CandidateRef] = field(default_factory=list)
    held: list[CandidateRef] = field(default_factory=list)


class RetrievalPipeline(RecallChannels):
    """Collect, hydrate, admit, and rank candidates in one read-only pass."""

    def _hydrate_admit(
        self,
        tx,
        candidate: CandidateRef,
        context: SearchContext,
        *,
        original_query: str | None = None,
        aside: _Aside | None = None,
    ) -> RetrievedObject | None:
        empty_episode = getattr(self.storage_reader, "empty_episode", None)
        if candidate.kind == "episode" and candidate.source != "exact_ref" and callable(empty_episode):
            if empty_episode(tx, candidate):
                # Preserve relation inspection/budget accounting while avoiding
                # the evidence walk for a state-only object admission discards.
                note_candidate(candidate, "empty_resume")
                return None
        obj = self.storage_reader.hydrate(tx, candidate, context)
        note_candidate(candidate, obj=obj)
        if obj is None:
            from .recall_log_view import rejection_reason

            note_candidate(candidate, rejection_reason(tx, candidate, context))
            return None
        if context.automatic_injection:
            from .session_brief import already_injected

            if already_injected(tx, candidate.kind, candidate.ref, candidate.revision):
                return None
        # The current message already supplies this text. Older copies add no
        # information to automatic context and can crowd out its actual evidence.
        # What such a copy was told in its turn still answers it: the copy is set
        # aside in ``aside.echoes`` to lead to its turn's replies, never delivered.
        question = context.query if original_query is None else original_query
        if context.mode == "auto" and obj.kind == "event":
            if same_message(obj.content, question) or (
                context.original_prompt is not None and same_message(obj.content, context.original_prompt)
            ):
                if aside is not None:
                    aside.echoes.append(candidate)
                note_candidate(candidate, "current_prompt")
                return None
        if candidate.source != "exact_ref" and not _identifiers_match(tx, candidate, context, obj, question):
            note_candidate(candidate, "identifier_mismatch")
            return None
        # An episode whose resume was never consolidated carries nothing but its
        # own state: `{"state": "unknown"}` fills a packet slot and answers
        # nothing.  Every plain chat turn opens one.  Named directly it is still
        # delivered, because then the caller asked for that object.
        if candidate.source != "exact_ref" and obj.kind == "episode" and _says_only_its_state(obj.content):
            note_candidate(candidate, "empty_resume")
            return None
        if context.mode == "auto" and context.in_context_since is not None and _in_context(tx, obj, context):
            # The asking model still holds everything it stands on, word for word: delivered, it would be a second copy
            # in that model's context, and take a slot from what the model does not have.  Set aside in ``aside``, it
            # still leads to what is related to it, as it would if delivered: left out of that, a session's own messages
            # would stop leading to the claims and procedures drawn from it before its last compaction.
            if aside is not None:
                aside.held.append(candidate)
            note_candidate(candidate, "already_in_context")
            return None
        return obj

    def _hydrate_all(
        self,
        tx,
        hydrated: list[tuple[CandidateRef, RetrievedObject]],
        candidates,
        context: SearchContext,
        gaps: list[str],
        *,
        original_query: str | None = None,
        aside: _Aside | None = None,
        attempted: set[tuple[str, str, int]] | None = None,
    ) -> None:
        """Hydrate in rank order within the retrieval portion of the request."""
        known = {candidate.key for candidate, _obj in hydrated}
        candidates = tuple(candidates)
        prefetch = getattr(self.storage_reader, "prefetch", None)
        if prefetch is not None and self._remaining(context) > 0:
            # What hydration reads, loaded together (``RetrievalStorage.prefetch``).
            prefetch(tx, tuple(candidate for candidate in candidates if candidate.key not in known), context)
        for candidate in candidates:
            if self._remaining(context) <= 0:
                gaps.append("deadline_exceeded_hydrate")
                break
            if attempted is not None:
                attempted.add(candidate.key)
            if candidate.key in known:
                continue
            obj = self._hydrate_admit(tx, candidate, context, original_query=original_query, aside=aside)
            if obj is None:
                continue
            hydrated.append((candidate, obj))
            known.add(candidate.key)

    #: A relation hit receives half the smaller of its own reciprocal-rank score and its seed's fusion score.
    #: With the default ``rrf_k=60`` and 48-candidate pool, its score is at most 1/124, below a direct hit at rank 48.
    #: Related hits can differ in their own ranks, so their final order need not match seed order.  Turn replies are
    #: collected before relation hops and capped at their seed's score. Echoes
    #: retain the reply's own score to reach an earlier answer to the same question.
    RELATION_WEIGHT = RELATION_WEIGHT

    def _expand_turns(self, tx, context, seeds, *, echoes, candidates, seen) -> tuple[int, bool]:
        """Append turn replies with their strongest seed path; return inspections and deadline exhaustion."""
        turn_replies = getattr(self.storage_reader, "turn_replies", None)
        if not callable(turn_replies):
            return 0, False
        # Replies precede other relations so a low-ranked question can reach
        # its answer. Half the bound leaves room for claims and episodes.
        # Echoes lead only to their turn: over repeated real questions, 42 of
        # 46 otherwise missed answers were reached only through such a copy.
        turn_bound = max(1, context.limits.relation_objects // 2)
        echo_keys = {candidate.key for candidate in echoes}
        turn_at = {}
        inspected = 0
        for seed in sorted((*seeds, *echoes), key=lambda item: (-item.fusion_score, item.key)):
            if inspected >= turn_bound:
                break
            # A seed without replies still costs its bounded turn lookups.
            if self._remaining(context) <= 0:
                return inspected, True
            for candidate in turn_replies(tx, seed):
                if inspected >= turn_bound:
                    break
                if self._remaining(context) <= 0:
                    return inspected, True
                inspected += 1
                own = rrf_score((candidate.rank + 1,), k=self.policy.rrf_k)
                score = own if seed.key in echo_keys else min(own, seed.fusion_score)
                if candidate.key in turn_at:
                    # Several seeds can share a reply. An echo's path retains
                    # its own score even when an ordinary seed reached it first.
                    index = turn_at[candidate.key]
                    prior = candidates[index]
                    candidates[index] = replace(prior, fusion_score=max(prior.fusion_score, score))
                    continue
                if candidate.key in seen:
                    continue
                seen.add(candidate.key)
                if f"{candidate.ref}@{candidate.revision}" in context.current_source_refs:
                    continue
                if self._hydrate_admit(tx, candidate, context) is None:
                    continue
                turn_at[candidate.key] = len(candidates)
                candidates.append(replace(candidate, fusion_score=score))
        return inspected, False

    def _related_candidate(self, tx, context, seed, candidate):
        """Weight an accessible neighbor. Held evidence can lead to another hop without being displayed."""
        aside = _Aside()
        visible = self._hydrate_admit(tx, candidate, context, aside=aside) is not None
        if not visible and not aside.held:
            return None
        score = rrf_score((candidate.rank + 1,), k=self.policy.rrf_k)
        if seed.fusion_score > 0:
            score = min(score, seed.fusion_score)
        return replace(candidate, fusion_score=score * self.RELATION_WEIGHT), visible

    def _expand(
        self,
        tx,
        context: SearchContext,
        seeds: tuple[CandidateRef, ...],
        gaps: list[str],
        *,
        echoes: tuple[CandidateRef, ...] = (),
    ) -> tuple[CandidateRef, ...]:
        """Bounded relation hops out of the seeds; every inspected object counts.

        ``echoes`` are older copies of the current message (``_hydrate_admit``): each leads to the replies of its turn
        like a seed, and to nothing else.  What the latest copy was told is raised once hydrated (``_raise_echo_turn``).
        """
        limits = context.limits
        if limits.relation_hops == 0 or limits.relation_objects == 0:
            return seeds
        all_candidates = list(seeds)
        seen = {candidate.key for candidate in (*seeds, *echoes)}
        frontier = list(seeds)

        def stopped(gap: str) -> tuple[CandidateRef, ...]:
            gaps.append(gap)
            return tuple(all_candidates)

        inspected, expired = self._expand_turns(tx, context, seeds, echoes=echoes, candidates=all_candidates, seen=seen)
        if expired:
            return stopped("deadline_exceeded_relation")

        for _hop in range(limits.relation_hops):
            next_frontier: list[CandidateRef] = []
            # The shared object bound is spent best seed first.  In key order it
            # would go to whichever refs sort lowest, often a weak match's claims.
            for seed in sorted(frontier, key=lambda item: (-item.fusion_score, item.key)):
                if inspected >= limits.relation_objects:
                    return stopped("relation_bound_reached")
                if self._remaining(context) <= 0:
                    return stopped("deadline_exceeded_relation")
                for candidate in self.storage_reader.related(tx, seed, limit=limits.relation_objects - inspected):
                    inspected += 1
                    bound_reached = inspected >= limits.relation_objects
                    if self._remaining(context) <= 0:
                        return stopped("deadline_exceeded_relation")
                    if candidate.key in seen:
                        if bound_reached:
                            return stopped("relation_bound_reached")
                        continue
                    seen.add(candidate.key)
                    if _excluded_relation_source(tx, context, seed, candidate):
                        continue
                    related = self._related_candidate(tx, context, seed, candidate)
                    if related is not None:
                        weighted, visible = related
                        next_frontier.append(weighted)
                        if visible:
                            all_candidates.append(weighted)
                    if bound_reached:
                        return stopped("relation_bound_reached")
            frontier = next_frontier
            if not frontier:
                break
        return tuple(all_candidates)

    # -- ranking and budget ---------------------------------------------------

    def _rank_hydrated(
        self, hydrated: list[tuple[CandidateRef, RetrievedObject]], context: SearchContext | None = None, *, tx=None
    ) -> list[tuple[CandidateRef, RetrievedObject]]:
        context_only: set[tuple[str, str, int]] = set()
        if context is not None and context.mode in {"auto", "current"}:
            # A bare question, and a reply restating what a recall tool returned
            # in its turn, are context, not evidence.  The weight goes on the
            # candidate itself: budget admission re-sorts on fusion score, and a
            # short question would otherwise win its place back there.
            context_only = {
                candidate.key
                for candidate, obj in hydrated
                if obj.kind == "event"
                and candidate.source != "exact_ref"
                and (asks_without_answering(obj.content) or dict(obj.metadata).get("recall_echo") == "true")
            }
            hydrated = [
                (replace(candidate, fusion_score=candidate.fusion_score * CONTEXT_ONLY_WEIGHT), obj)
                if candidate.key in context_only
                else (candidate, obj)
                for candidate, obj in hydrated
            ]
        if context is not None:
            hydrated = _weighed_by_time(hydrated, context)
        ranked = sorted(
            hydrated,
            key=lambda pair: (
                -pair[0].fusion_score,
                -(pair[0].vector_score if pair[0].vector_score is not None else -1.0),
                pair[0].kind,
                pair[0].ref,
                pair[0].revision,
            ),
        )
        # Only candidates that already passed hydration, scope, time and source
        # admission reach this point; scores never create permission or facts.
        # Explicit historical lookups keep their original ordering.
        if context is None or context.mode in {"history", "as_of"}:
            return ranked
        query_terms = set(meaningful_query_terms(self._admission_query(context)))
        if not query_terms:
            return ranked
        indexed = (
            self.storage_reader.source_term_matches(tx, ranked, tuple(sorted(query_terms)))
            if tx is not None and isinstance(self.storage_reader, RetrievalStorage)
            else {}
        )
        matched = {
            candidate.key: (
                indexed[candidate.key]
                if candidate.key in indexed
                else query_terms.intersection(lexical_terms(statement_text(obj)))
            )
            for candidate, obj in ranked
        }
        # Diversity rescoring visits a candidate repeatedly; its immutable
        # evidence roots need parsing only once for this admitted pool.
        roots_by_key = {candidate.key: evidence_roots(obj) for candidate, obj in ranked}
        copies = repeated_report_claims(tx, ranked, context, matched)
        # Multiple channels can nominate the same report under several labels. A
        # later copy keeps its best channel rank, without counting their votes again.
        fusion_caps = {
            c.key: min(c.fusion_score, rrf_score((c.rank,), k=self.policy.rrf_k)) for c, _ in ranked if c.key in copies
        }
        selected: list[tuple[CandidateRef, RetrievedObject]] = []
        covered: set[str] = set()
        selected_roots: set[str] = set()

        def score(pair: tuple[CandidateRef, RetrievedObject]) -> tuple[bool, float]:
            candidate, obj = pair
            hits = matched[candidate.key]
            roots = roots_by_key[candidate.key]
            # A small, bounded diversity bonus favours another part of the
            # question over repeats from one source. RRF remains the base.
            relevance = len(hits) / len(query_terms)
            additional = len(hits - covered) / len(query_terms)
            repeated = bool(roots and roots <= selected_roots and not (hits - covered))
            # An earlier question or a restating reply covers the query's words
            # by construction, so its coverage bonus is weighted like its fusion.
            weight = CONTEXT_ONLY_WEIGHT if candidate.key in context_only else 1.0
            return (
                candidate.source == "exact_ref",
                fusion_caps.get(candidate.key, candidate.fusion_score)
                + weight * (0.008 * relevance + 0.008 * additional)
                - 0.006 * repeated,
            )

        while ranked:
            if self._remaining(context) <= 0:
                selected.extend(ranked)
                break
            pair = ranked.pop(max(range(len(ranked)), key=lambda index: score(ranked[index])))
            selected.append(pair)
            covered.update(matched[pair[0].key])
            selected_roots.update(roots_by_key[pair[0].key])
        return selected

    def _apply_budget(
        self,
        ranked: list[tuple[CandidateRef, RetrievedObject]],
        limits: SearchLimits,
        gaps: list[str] | None = None,
        *,
        context: SearchContext | None = None,
    ) -> list[tuple[CandidateRef, RetrievedObject]]:
        """Scan only candidates that can take a slot; withheld items consume no budget.

        Admitted ordinary hits survive a collection deadline. The optional oversized fallback uses only time left.
        """
        gaps = [] if gaps is None else gaps
        kept: list[tuple[CandidateRef, RetrievedObject]] = []
        oversized: list[tuple[CandidateRef, RetrievedObject]] = []
        total_tokens = 0
        for candidate, item in event_admission_order(ranked, limits):
            if len(kept) >= limits.max_items:
                break
            tokens = estimate_tokens(getattr(item, "content", ""))
            if tokens > limits.budget_tokens:
                # Never charge an undeliverable item against all later hits.
                oversized.append((candidate, item))
                continue
            if kept and total_tokens + tokens > limits.budget_tokens:
                # One large candidate must not starve smaller useful evidence
                # later in the ranking. The compiler budgets the whole packet.
                continue
            if not withhold_secret_items([item], gaps):
                continue
            total_tokens += tokens
            kept.append((candidate, item))
        # An item that alone exceeds the whole budget cannot fit in what is
        # left of it, so it is worth returning only when there is nothing else
        # to show: the expandable hint is a fallback, not a suffix.
        if kept:
            return kept
        for pair in oversized:
            if len(kept) >= limits.max_items:
                break
            if context is not None and self._remaining(context) <= 0:
                gaps.append("deadline_exceeded")
                break
            if withhold_secret_items([pair[1]], gaps):
                kept.append(pair)
        return kept

    # -- the search itself ----------------------------------------------------

    def search(self, context: SearchContext) -> RetrievalResult:
        if not isinstance(context, SearchContext):
            raise ContractError("INPUT_INVALID", "search_context")
        limits = effective_limits(context)
        # A historical or as-of read is left as it was: the time a question names is read for a recall of what holds
        # now, and no channel searches it or weighs by it otherwise.
        reads_time = context.mode not in {"history", "as_of"}
        time = query_time(context.query, now=context.now, zone=context.zone) if reads_time else None
        started = self.clock.monotonic()
        deadline = context.deadline
        outer = current_request_deadline()
        if outer is not None:
            deadline = min(deadline, started + outer.remaining())
        remaining = deadline - started
        reserve = release_reserve_seconds(limits.max_items, remaining_seconds=remaining)
        working = replace(context, limits=limits, time=time, deadline=deadline - reserve)
        timings = RecallTimings(
            self.clock.monotonic,
            dict.fromkeys(("prefetch", "collect", "hydrate", "followup", "relation", "select"), 0.0),
        )
        with (
            timings.measure("retrieval"),
            using_request_deadline(self._deadline(working)),
        ):
            result = self._search(working, timings, release_deadline=deadline)
        return replace(result, diagnostics=timings.snapshot(), release_deadline=deadline)

    def _search(self, working: SearchContext, timings: RecallTimings, *, release_deadline: float) -> RetrievalResult:
        """Read optional channels within the reserved retrieval window."""
        if self._remaining(working) <= 0:
            return _empty_result(working, "deadline_exceeded")
        gaps: list[str] = []
        try:
            with self.storage.read(
                working.trusted_context, remaining_seconds=max(self._remaining(working), 0.001)
            ) as tx:
                working = replace(working, current_source_refs=current_prompt_refs(tx, working.current_source_refs))
                snapshot(tx, working)
                epoch = self.storage_reader.epoch(tx)
                if working.automatic_injection and not working.focus_refs:
                    route = automatic_recall_route(
                        working.query,
                        tx=tx,
                        context=working.trusted_context,
                        current_source_refs=working.current_source_refs,
                        deadline=working.deadline,
                        clock=self.clock,
                    )
                    if route == "current_task_only":
                        task = current_task_candidate(tx, working, self.storage_reader, self.clock, project_only=True)
                        if task is None:
                            # The host can ask which task to resume without sending a broad vector query.
                            empty = _empty_result(working, "auto_recall_skipped:unanchored_followup")
                            return replace(empty, unmet_needs=unmet_needs(working.query, ()))
                        # A unique grounded task is the only evidence the prompt identifies.
                        # It still passes ordinary privacy, budget and final release checks.
                        candidate, obj = task
                        ranked, items, _ = self._select(
                            tx, working, [(replace(candidate, source="relation"), obj)], gaps, include_background=False
                        )
                        return self._result(working, epoch, ranked, items, gaps, 1, 1, working.query)
                    if route is not None:
                        return _empty_result(working, f"auto_recall_skipped:{route}")
                with timings.measure("prefetch"):
                    prefetched = self._prefetch_query(working)
                working = self._scoped(tx, working, gaps)
                aside = _Aside()
                hydrated, seed_count, working = self._collect_rounds(
                    tx,
                    working,
                    gaps,
                    prefetched=prefetched,
                    aside=aside,
                    timings=timings,
                    release_deadline=release_deadline,
                )
                if len(meaningful_query_terms(working.query)) < ECHO_TURN_MIN_TERMS:
                    aside.echoes.clear()
                with (
                    timings.measure("relation"),
                    using_request_deadline(self._deadline(working)),
                ):
                    self._hydrate_related(
                        tx, working, hydrated, gaps, echoes=tuple(aside.echoes), held=tuple(aside.held)
                    )
                with timings.measure("select"):
                    ranked, query_items, overflow = self._select(tx, working, hydrated, gaps)
                question = self._admission_query(working)
                withheld = note_superseded_evidence(tx, gaps)
                return replace(
                    self._result(working, epoch, ranked, query_items, gaps, seed_count, len(hydrated), question),
                    overflow_candidates=overflow,
                    overflow_policy_revision=private.rule_revision(tx) if overflow else None,
                    withheld_evidence=tuple(sorted(withheld)),
                )
        except ContractError as exc:
            if exc.code == "DEADLINE_EXCEEDED":
                return _empty_result(working, "deadline_exceeded")
            return _empty_result(working, f"sqlite_unavailable:{exc.code}")
        except Exception as exc:
            return _empty_result(working, f"sqlite_unavailable:{type(exc).__name__}")

    @staticmethod
    def _scoped(tx, working: SearchContext, gaps: list[str]) -> SearchContext:
        """The context with the days and entries its question names (``recall_scope``) when it names a day and asks
        only what was said or done then; any other question has no scope.

        Reading the scope never fails a recall: a question it cannot read is recalled as if it named no day, and the
        gap says so.  A prompt grown past the search's length by normalisation would empty the whole recall.
        """
        scope = None
        try:
            from .summary_storage import previous_conversation_scope

            if query_scope(working.query, now=working.now, zone=working.zone, entries={}) is not None:
                entries = getattr(getattr(tx, "registry", None), "entries", None)
                names = (
                    {entry_id: str(value.get("name") or entry_id) for entry_id, value in entries().items()}
                    if callable(entries)
                    else {}
                )
                scope = query_scope(working.query, now=working.now, zone=working.zone, entries=names)
                if scope is not None and not asks_what_was_said(scope.rest):
                    scope = None
            if scope is None:
                scope = previous_conversation_scope(tx, working.query, now=working.now)
        except Exception as exc:
            gaps.append(f"scope_unreadable:{type(exc).__name__}")
            scope = None
        return working if working.scope == scope else replace(working, scope=scope)

    def _collect_rounds(
        self,
        tx,
        working: SearchContext,
        gaps: list[str],
        *,
        prefetched=None,
        aside: _Aside | None = None,
        timings: RecallTimings,
        release_deadline: float,
    ) -> tuple[list[tuple[CandidateRef, RetrievedObject]], int, SearchContext]:
        """The first round plus at most one directed follow-up for an open need; what hydration sets aside on the
        way goes to ``aside``."""
        hydrated: list[tuple[CandidateRef, RetrievedObject]] = []
        seen: set[tuple[str, str, int]] = set()
        seed_count = 0
        max_rounds = min(2, 1 + working.limits.followups)
        budget = ChannelBudget(working.limits)
        follow_query: str | None = None
        for round_index in range(max_rounds):
            if self._remaining(working) <= 0:
                gaps.append("deadline_exceeded_followup" if round_index else "deadline_exceeded")
                break
            # The day and the time a question names are read once: a follow-up asks for something else.
            round_context = replace(working, query=follow_query, scope=None, time=None) if follow_query else working
            budget.hold(max_rounds - round_index - 1)
            # Collection leaves one hydration pass before retrieval itself stops.
            remaining_work = self._remaining(working)
            hydration = (
                min(remaining_work * 0.25, 0.5)
                if remaining_work >= 1.0
                else min(remaining_work / 2, release_reserve_seconds(working.limits.max_items) / 4)
            )
            collecting = replace(round_context, deadline=round_context.deadline - hydration)
            attempted: set[tuple[str, str, int]] = set()

            def hydrate_ready(
                candidates,
                *,
                attempted=attempted,
                round_context=round_context,
                original_query=working.query,
            ):
                fresh = tuple(candidate for candidate in candidates if candidate.key not in attempted)
                if not fresh:
                    return
                with timings.measure("hydrate"):
                    self._hydrate_all(
                        tx,
                        hydrated,
                        fresh,
                        round_context,
                        gaps,
                        original_query=original_query,
                        aside=aside,
                        attempted=attempted,
                    )
                cutoff = self._release_cutoff(round_context, release_deadline, timings, hydrated)
                return cutoff

            prior_hydration = timings.stages["hydrate"]
            with (
                timings.measure("collect"),
                using_request_deadline(self._deadline(collecting)),
            ):
                seeds = self._collect(
                    tx, collecting, gaps, seen=seen, budget=budget, prefetched=prefetched, on_ready=hydrate_ready
                )
            timings.stages["collect"] -= timings.stages["hydrate"] - prior_hydration
            seed_count += len(seeds)
            seen.update(candidate.key for candidate in seeds)
            if not hydrated and seeds and self._remaining(round_context) <= 0:
                # When GIL contention or a slow channel pushes past the retrieval cutoff, spend at most half of
                # the remaining release reserve to hydrate the top already-collected seeds rather than returning 0.
                reserve_left = release_deadline - self.clock.monotonic()
                if reserve_left > 0.002:
                    salvage = replace(round_context, deadline=self.clock.monotonic() + reserve_left / 2)
                    hydrate_ready(seeds[: working.limits.max_items], round_context=salvage)
                else:
                    hydrate_ready(seeds)
            else:
                hydrate_ready(seeds)
            # Optional signals can improve a ready candidate's fused rank.
            # Refresh its rank without repeating hydration or echo side effects.
            # A ready candidate this round hydrated early that the final qualification dropped leaves with it.
            fused = {candidate.key: candidate for candidate in seeds}
            hydrated[:] = [
                (fused.get(candidate.key, candidate), obj)
                for candidate, obj in hydrated
                if candidate.key in fused or candidate.key not in attempted
            ]
            cutoff = self._release_cutoff(working, release_deadline, timings, hydrated)
            working = replace(working, deadline=cutoff)
            if round_index + 1 >= max_rounds:
                break
            follow_query = self._followup_query(tx, working, hydrated, gaps, timings)
            if follow_query is None:
                break
        return hydrated, seed_count, working

    def _release_cutoff(self, context, deadline, timings, hydrated) -> float:
        """Keep two fresh reads of the ready prefix before further optional work."""
        reserve = release_reserve_seconds(
            context.limits.max_items,
            remaining_seconds=deadline - self.clock.monotonic(),
            hydration_seconds=timings.stages["hydrate"] / 1000,
            hydrated_count=len(hydrated),
        )
        return deadline - reserve

    def _followup_query(self, tx, working, hydrated, gaps, timings) -> str | None:
        """Plan the next query and admit a grounded task only while retrieval has time."""
        items = tuple(obj for _candidate, obj in hydrated)
        needs = unmet_needs(working.query, items)
        if not needs:
            return None
        if self._remaining(working) <= 0:
            gaps.append("deadline_exceeded_followup")
            return None
        with timings.measure("followup"):
            query = directed_followup_query(working, needs, items)
        if query is not None and "resume_state" in needs:
            task = current_task_candidate(tx, working, self.storage_reader, self.clock)
            if task is not None and task[0].key not in {candidate.key for candidate, _obj in hydrated}:
                hydrated.append((replace(task[0], source="relation", lexical_score=1.0), task[1]))
        return query

    def _hydrate_related(
        self,
        tx,
        working: SearchContext,
        hydrated: list[tuple[CandidateRef, RetrievedObject]],
        gaps: list[str],
        *,
        echoes: tuple[CandidateRef, ...] = (),
        held: tuple[CandidateRef, ...] = (),
    ) -> None:
        # What the asking model holds is a seed as well, and never delivered (``_hydrate_admit``).
        seeds = (*(candidate for candidate, _obj in hydrated), *held)
        known = {candidate.key for candidate in (*seeds, *echoes)}
        expanded = [
            candidate
            for candidate in self._expand(tx, working, seeds, gaps, echoes=echoes)
            if candidate.key not in known
        ]
        self._hydrate_all(tx, hydrated, expanded, working, gaps)
        self._raise_echo_turn(tx, working, hydrated, echoes, gaps)

    def _raise_echo_turn(
        self,
        tx,
        working: SearchContext,
        hydrated: list[tuple[CandidateRef, RetrievedObject]],
        echoes: tuple[CandidateRef, ...],
        gaps: list[str],
    ) -> None:
        """What the same question was told the last time it was asked goes before the best candidate of its time.

        Reached as any seed's turn is, at a first rank's fixed score, it would fall below every candidate two channels
        agree on: with vectors on, one person's questions asked again lost about 25 of 124 answers in a threshold sweep.
        Only the latest copy's turn is raised, so an answer that changed since is not raised beside the one that
        replaced it.  Of that turn at most half the packet is: the replies a channel ranked highest, which say what was
        asked, then the turn's last reply, which answers it, when the turn was read to its end.  An agent's turn opens
        with what it is about to do, and raising every reply would let an old turn's first messages fill the packet, as
        raising every reply a channel found would let its narration of the subject do; the last reply read of a turn cut
        by the window is narration too.  Raising only the best found reply lost 6 of the owner's 173 questions asked
        again with vectors on, where more replies are found and the answer is not always the best of them.  They go
        above the best candidate said up to that turn, the turn's other replies included, and not above what was said
        after it and ranks higher still, which may say the answer changed.  Kept below every such candidate outranking
        the best of the turn's time, the answer fell out of the packet for 5 of the owner's 173 questions asked again: a
        store keeps growing, and something said since matches most questions.
        """
        latest_turn = getattr(self.storage_reader, "latest_turn", None)
        if not echoes or not callable(latest_turn):
            return
        if self._remaining(working) <= 0:
            gaps.append("deadline_exceeded_relation")
            return
        read = latest_turn(tx, echoes, now=working.now)
        if read is None:
            return
        opened, replies, ended = read
        at = {candidate.key: index for index, (candidate, _obj) in enumerate(hydrated)}
        found = [reply for reply in replies if reply.key in at and hydrated[at[reply.key]][0].source != "relation"]
        # At most half the packet: the replies a channel ranked highest, then the last reply of a whole turn.
        room = max(1, working.limits.max_items // 2)
        final = replies[-1] if ended else None
        by_score = [
            reply
            for reply in sorted(found, key=lambda reply: -hydrated[at[reply.key]][0].fusion_score)
            if final is None or reply.key != final.key
        ]
        raised = [*by_score[: room - (final is not None)], *([final] if final is not None else [])]
        # The turn's other replies are of its time, and stay below what is raised of it.
        of_turn, lifted = {reply.key for reply in replies}, {reply.key for reply in raised}
        others = [
            (candidate.fusion_score, candidate.key not in of_turn and _said_after(obj, opened))
            for candidate, obj in hydrated
            if candidate.key not in lifted
        ]
        best = max((score for score, after in others if not after), default=0.0)
        for index, reply in enumerate(raised):
            score = best + rrf_score((index + 2,), k=self.policy.rrf_k)
            if reply.key in at:
                candidate, obj = hydrated[at[reply.key]]
                if candidate.fusion_score < score:
                    hydrated[at[reply.key]] = (replace(candidate, fusion_score=score), obj)
            elif (
                f"{reply.ref}@{reply.revision}" not in working.current_source_refs
                and (obj := self._hydrate_admit(tx, reply, working)) is not None
            ):
                hydrated.append((replace(reply, fusion_score=score), obj))

    @staticmethod
    def _behind_replacements(
        ranked: list[tuple[CandidateRef, RetrievedObject]], working: SearchContext
    ) -> list[tuple[CandidateRef, RetrievedObject]]:
        """``ranked`` with each message whose values were all replaced (``retrieval_hydration.replacements``) moved
        right behind the first item that states what holds now in their place, where that ranked below it; messages
        moved behind one item keep the order they were ranked in, as does the rest, so a small packet keeps the
        highest-ranked of them.

        A claim's correction moves the claim's head and leaves the message that said the old value as it was, and
        that message often holds the question's words better than the one that changed it: ranked first, it is the
        answer a reader gives.  Only replaced messages move, and none is dropped: each still says what was said then,
        and a question about the old value still finds it.  A message the caller names by ref keeps its place first.
        A question that names a day or a time, or asks what held before (``recall_scope.asks_what_held_before``), is
        left as it is, since what was said or held then may be the value replaced since."""
        if working.scope is not None or working.time is not None or asks_what_held_before(working.query):
            return ranked
        place = {candidate.key: index for index, (candidate, _obj) in enumerate(ranked)}
        behind: dict[int, list[tuple[CandidateRef, RetrievedObject]]] = {}
        for index, (candidate, obj) in enumerate(ranked):
            first = min((place[key] for key in replacements(obj) if key in place), default=index)
            # What states a value that holds is never itself moved, so every message moved stays behind its item.
            if first > index and candidate.source != "exact_ref" and not replacements(ranked[first][1]):
                behind.setdefault(first, []).append((candidate, obj))
        moved = {pair[0].key for pairs in behind.values() for pair in pairs}
        return [
            pair
            for index, kept in enumerate(ranked)
            if kept[0].key not in moved
            for pair in (kept, *behind.get(index, ()))
        ]

    def _select(
        self,
        tx,
        working: SearchContext,
        hydrated: list[tuple[CandidateRef, RetrievedObject]],
        gaps: list[str],
        *,
        include_background: bool = True,
    ):
        """Rank, fold duplicate bodies, fit the item budget, then add background."""
        # Distinct *content*, not distinct rows: candidates are already unique
        # by (kind, ref, revision), but a corpus where a legacy import
        # re-delivered the same bodies under fresh identities would let one
        # document hold four of six slots.  One filter spans both passes so
        # background cannot repeat the evidence either.
        distinct = DistinctContent()
        ordered = self._behind_replacements(self._rank_hydrated(hydrated, working, tx=tx), working)
        originals = prefer_originals(tx, ordered, working, clock=self.clock.monotonic, gaps=gaps)
        note_removed(ordered, originals, "restatement")
        ordered = originals
        if self._remaining(working) <= 0:
            gaps.append("deadline_exceeded_select")
        safe = withhold_secret_items(ordered, gaps, content=lambda pair: pair[1])
        note_removed(ordered, safe, "secret")
        visible = visible_query_pairs(
            tx,
            safe,
            working,
            now=getattr(self.clock, "utc_now", lambda: working.now)(),
        )
        note_removed(safe, visible, "private_or_muted")
        ordered = visible
        pool = distinct.filtered(ordered)
        note_removed(ordered, pool, "duplicate")
        defer_budget = working.packet_selection and has_conversation_summaries(pool)
        # A summary cannot be discarded until the original survives the packet's
        # full envelope and schema limits. Keep a bounded pool for that compiler,
        # which owns final selection and can refill slots released by duplicates.
        ranked = (
            pool[: working.limits.candidate_pool]
            if defer_budget
            else self._apply_budget(pool, working.limits, gaps, context=working)
        )
        note_removed(pool, ranked, "budget")
        query_items = tuple(obj for _candidate, obj in ranked)
        if include_background and working.mode == "auto" and self._remaining(working) > 0:
            if ranked and all(estimate_tokens(obj.content) > working.limits.budget_tokens for _c, obj in ranked):
                # Nothing that fits answered (as when the only hit that fits is a message the model still holds): one
                # hit too large for any packet stays as an expansion hint. The
                # compiler requires query evidence to survive before background
                # can share the final packet.
                ranked = ranked[:1]
            background = background_candidates(
                tx, working, self.storage_reader, self.clock, gaps, query_evidence=bool(query_items)
            )
            background = withhold_secret_items(background, gaps, content=lambda pair: pair[1])
            present = {candidate.key for candidate, _obj in ranked}
            # Query evidence comes first; background shares the final byte
            # budget and can only consume remaining packet slots.  What the
            # asking model still holds is not background either (``_in_context``).
            ranked.extend(
                pair
                for pair in background
                if pair[0].key not in present
                and not conflicting_record(tx, working, pair[1])
                and distinct.admits(pair[1])
                and not (working.in_context_since is not None and _in_context(tx, pair[1], working))
            )
            note_truncation(gaps, "packet_slots", considered=working.limits.max_items, available=len(ranked))
            if not defer_budget:
                ranked = ranked[: working.limits.max_items]
        note_duplicates(gaps, distinct.collapsed)
        # Content equality saves full-item slots, but cannot establish that
        # independent sessions describe the same event. Only the proven
        # restatement filter above removes sources from the neutral index.
        overflow = tuple(
            candidate
            for candidate, obj in ordered
            if obj.kind in {"event", "claim"} and candidate.source not in {"background", "exact_ref"}
        )
        return ranked, query_items, overflow

    @staticmethod
    def _result(
        working: SearchContext,
        epoch: int,
        ranked,
        query_items: tuple[RetrievedObject, ...],
        gaps: list[str],
        seed_count: int,
        hydrated_count: int,
        question: str,
    ) -> RetrievalResult:
        """The search's result and its verdict on its query evidence: unknown without any, partial with a need left
        open.  Then a question that names what answers it (``answering``) is unknown when nothing does, ambiguous when
        what answers a question that asks to choose rests on more than one source (``evidence_roots``: a claim and the
        message it was drawn from say one thing once), else supported.  One that names nothing is ambiguous by the same
        rule, else partial: what matched it was never held to the question.  ``question`` is the question without the
        owner's name, which their own words never carry.  A question about a day names its days and entries, not what
        answers it (``recall_scope``), and is left unjudged."""
        items = tuple(obj for _candidate, obj in ranked)
        needs = unmet_needs(working.query, query_items)
        if items and not query_items:
            needs = (*needs, "query_evidence_missing")
        answers = answering(question, query_items) if working.scope is None else Answers(False, query_items)
        sources = {root for item in answers.items for root in evidence_roots(item)}
        if not query_items:
            answerability = "unknown"
        elif needs:
            answerability = "partial"
        elif answers.judged and not answers.items:
            answerability = "unknown"
        elif len(sources) > 1 and mentions(working.query, CHOICE_MARKERS):
            answerability = "ambiguous"
        else:
            answerability = "supported" if answers.judged else "partial"
        return RetrievalResult(
            tuple(candidate for candidate, _obj in ranked),
            items,
            epoch,
            tuple(dict.fromkeys(gaps)),
            "partial" if gaps else "unknown",
            answerability,
            seed_count,
            hydrated_count,
            request_id=working.request_id,
            unmet_needs=needs,
        )

    def collection(self, context: SearchContext, query: CollectionQuery, cursor=None) -> CollectionPage:
        if not isinstance(context, SearchContext) or not isinstance(query, CollectionQuery):
            raise ContractError("INPUT_INVALID", "collection_request")
        remaining = self._remaining(context)
        if remaining <= 0:
            raise ContractError("DEADLINE_EXCEEDED", "collection")
        with self.storage.read(context.trusted_context, remaining_seconds=remaining) as tx:
            return self.storage_reader.collection(tx, context, query, cursor)


def recall(
    context: SearchContext, *, storage, vector_port=None, policy: RecallPolicy | None = None, clock=None
) -> RetrievalResult:
    """Small functional entry point for adapters that do not retain a service."""
    return RetrievalPipeline(storage, vector_port=vector_port, policy=policy, clock=clock).search(context)
