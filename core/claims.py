"""Pure Claim qualification and temporal selection; no adapters, SQL or models."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial

from ..contracts import (
    VALUE_FREE_KINDS,
    Basis,
    ClaimProposal,
    ContractError,
    Origin,
    SourcePrincipal,
    nfkc_letters_and_digits,
    utc_timestamp,
    verified_human_principal_ref,
)
from .candidate_lifecycle import SELF_SUBJECTS
from .claim_denial import DENIAL_REASON as DENIAL_REASON
from .claim_denial import DENIED_REASON as DENIED_REASON
from .claim_denial import NOT_A_VALUE_REASON as NOT_A_VALUE_REASON
from .claim_denial import Denial as Denial
from .claim_denial import _negation, is_denial
from .claim_denial import negates as negates
from .claim_denial import read_denial as read_denial
from .evidence_support import SupportVerdict, positioned, support_verdict
from .fact_actions import ClaimDraft, EvidenceReference
from .fact_evidence import evidence_supports_claim, evidence_supports_relation
from .fact_identity import fact_frame
from .fact_temporal_semantics import classify_durable_state_clause
from .source_qualification import (
    AUTHORITY_QUESTION,
    CONDITIONAL,
    POLARITY,
    RELATIVE_SCOPE,
    UNASSERTED_UNCERTAINTY,
    asserted_marker,
    attributed_to_another,
    bound_literal,
    condition_supports_value,
    in_pasted_text,
    is_transient_request,
    preserves_qualifiers,
    self_report_bound,
)
from .subject_binding import neighbourhood_binds


@dataclass(frozen=True)
class RootEvidence:
    ref: str
    revision: int
    origin: Origin
    original_origin: str | None
    content: str
    occurred_at: str | None
    capture_state: str
    session_id: str
    import_verified: bool = False
    capture_gaps: tuple[str, ...] = ()
    source_principal: SourcePrincipal | None = None


@dataclass(frozen=True)
class Qualification:
    state: str
    basis: Basis
    reason: str


@dataclass(frozen=True)
class ClaimVersion:
    ref: str
    revision: int
    current_revision: int
    scope_id: str
    project_id: str | None
    branch_id: str | None
    payload: ClaimProposal
    state: str
    basis: Basis
    reason: str
    valid_from: str | None
    valid_to: str | None
    recorded_from: str
    recorded_to: str | None
    replaces_revision: int | None
    conflict_revisions: tuple[int, ...]
    suppressed: bool = False
    importance: int = 3
    pinned: bool = False
    base_importance: int = 3


def persisted_claim_payload(proposal):
    """Keep salience metadata outside the version payload and replay identity."""
    return {key: value for key, value in proposal.items() if key != "importance"}


def canonical_time(value: str | None) -> str | None:
    return None if value is None else utc_timestamp(value)


def claim_slot(scope_id: str, project_id: str | None, branch_id: str | None, proposal: ClaimProposal) -> str:
    """Only identical explicit slots merge; aliases/semantic similarity confer no identity."""
    subject = proposal["subject"]
    relation = proposal.get("alias")
    if proposal["kind"] == "alias" and proposal["predicate"].startswith("entity_identity:") and relation is not None:
        # A name has one relation in an audience even when its evidence or target changes.
        subject = relation["name"]
    identity = [
        scope_id,
        project_id,
        branch_id,
        proposal["kind"],
        subject,
        proposal["predicate"],
        sorted(set(proposal["conditions"])),
    ]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


#: Kinds whose claims state one value for a subject's relation.  Which of them an extraction reads a statement as
#: varies from one message to the next ("we launch on March 3": a fact, then a decision), so a value's slot is looked
#: for among all four (``slot_kinds``).  A claim's own kind never changes.
STATED_VALUE_KINDS = frozenset({"fact", "preference", "constraint", "decision"})


def slot_kinds(kind: str) -> frozenset[str]:
    """The kinds under which a claim of ``kind`` may already hold its slot's value."""
    return STATED_VALUE_KINDS if kind in STATED_VALUE_KINDS else frozenset({kind})


def slot_reading(payload) -> tuple[frozenset[str], str, tuple[str, ...]]:
    """What a claim's kind, labels and conditions say its slot is, however an extraction wrote them.

    Claims that ``claim_slot`` files apart only for that writing read the same: they are candidates for one
    same-slot judgement (``semantic_proposals.merge_candidates``). Empty frames cannot nominate a candidate.
    A reading decides nothing by itself; health classification uses the strict normalized slot.
    """
    return (
        slot_kinds(payload["kind"]),
        fact_frame(payload["subject"], payload["predicate"]),
        tuple(sorted(set(payload["conditions"]))),
    )


def same_assertion(left: ClaimProposal, right: ClaimProposal) -> bool:
    fields = (
        "kind",
        "subject",
        "predicate",
        "value_text",
        "conditions",
        "statement_kind",
        "valid_from",
        "valid_to",
        "occurred_at",
        "procedure",
        "intention",
        "alias",
    )
    return all(
        (sorted(set(left[k])) == sorted(set(right[k]))) if k == "conditions" else left.get(k) == right.get(k)
        for k in fields
    )


def statement_key(payload):
    """Canonical assertion content, excluding evidence, for dependency comparisons."""
    from .fact_identity import normalize_fact_component

    value = {
        key: payload.get(key)
        for key in (
            "kind",
            "subject",
            "predicate",
            "value_text",
            "conditions",
            "statement_kind",
            "valid_from",
            "valid_to",
            "occurred_at",
            "procedure",
            "intention",
            "alias",
        )
    }
    for key in ("subject", "predicate", "value_text"):
        value[key] = normalize_fact_component(value[key], max_chars=2048, allow_empty=True)
    value["conditions"] = sorted(set(value["conditions"] or ()))
    for key in ("valid_from", "valid_to", "occurred_at"):
        value[key] = canonical_time(value[key])
    if value["kind"] in {"fact", "preference", "constraint", "decision", "observation"}:
        for key in ("statement_kind", "occurred_at", "procedure", "intention", "alias"):
            del value[key]
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


_HYPOTHETICAL = re.compile(r"假设|假如|设想|虚构|假定|\b(?:suppose|hypothetic|fictional|what if)\b", re.I)
_UNDECIDED = re.compile(r"先不要定|还[没未]决定|尚未确定|没有决定|待定|\b(?:not decided|undecided)\b", re.I)
_QUOTED_OTHER = re.compile(r"不是我的|不是我本人|客户原文|\b(?:not my preference|customer said)\b", re.I)
_LIMIT = re.compile(r"只限|仅限|本次|这次|今天|未授权|除非|\b(?:only this|this time|unless|without permission)\b", re.I)
_ACCEPTANCE = re.compile(
    r"认可|验收通过|按这个方法|就按这样|这个方法[很好对可]|\b(?:accept|approved|use this method)\b", re.I
)
_COMPLETION = re.compile(r"已经完成|已完成|检查完成|完成了|\b(?:completed|finished|done)\b", re.I)
_CANCELLATION = re.compile(r"取消|撤销|作废|\b(?:cancel|revoke|withdraw)\b", re.I)
_NEGATED_COMPLETION = re.compile(r"未完成|没[有]?完成|尚未完成|提醒|\b(?:not done|not completed|remind)\b", re.I)
_NEGATION = re.compile(r"不是|并非|没有|未曾|不喜欢|讨厌|\b(?:not|never|dislike)\b", re.I)


def evidence_refs(proposal: ClaimProposal) -> tuple[str, ...]:
    refs = [f"{s['source_ref']}@{s['source_revision']}" for s in proposal["evidence_spans"]]
    intention = proposal.get("intention")
    if intention is not None:
        refs.extend(intention["state_evidence_refs"])
    procedure = proposal.get("procedure")
    if procedure is not None:
        refs.extend(procedure["counterexample_refs"])
    return tuple(dict.fromkeys(refs))


def verified_human_ref(root: RootEvidence) -> str | None:
    return verified_human_principal_ref(root.source_principal)


def _unresolved_subject(proposal: ClaimProposal, roots: tuple[RootEvidence, ...]) -> str:
    refs = sorted(
        f"{root.ref}@{root.revision}"
        for root in roots
        if any(
            span["source_ref"] == root.ref and span["source_revision"] == root.revision
            for span in proposal["evidence_spans"]
        )
    )
    digest = hashlib.sha256(json.dumps(refs, separators=(",", ":")).encode()).hexdigest()
    return f"unresolved-source:{digest}"


def bind_claim_subject(
    proposal: ClaimProposal,
    roots: tuple[RootEvidence, ...],
    *,
    stored: bool = False,
    semantic_support: bool = False,
    verdict: SupportVerdict | None = None,
    nominate_only: bool = False,
) -> tuple[ClaimProposal, bool, str | None]:
    """Bind a first-person model label to C1 evidence, or isolate it safely.

    The model may emit ``current_user`` (or legacy self labels) for any language's
    first-person or omitted-subject self statement, but it cannot mint an authority
    key. Unresolved sources receive a source-specific non-authority slot so different
    speakers in one shared scope cannot collapse together.
    """
    cited_roots = tuple(
        root
        for root in roots
        if any(
            span["source_ref"] == root.ref and span["source_revision"] == root.revision
            for span in proposal["evidence_spans"]
        )
    )
    relevant = tuple(root for root in cited_roots if effective_origin(root) == "human_direct")
    relevant_keys = {(root.ref, root.revision) for root in relevant}
    all_human_direct = bool(proposal["evidence_spans"]) and all(
        (span["source_ref"], span["source_revision"]) in relevant_keys for span in proposal["evidence_spans"]
    )
    written = stored and all_human_direct and {verified_human_ref(root) for root in relevant} == {proposal["subject"]}
    if proposal["subject"].startswith("principal:") and not written:
        return proposal, False, "subject_not_bound"
    if proposal["subject"].casefold() not in SELF_SUBJECTS and not written:
        return proposal, False, None
    bound = deepcopy(proposal)

    def spans_of(root):
        return [
            span
            for span in proposal["evidence_spans"]
            if span["source_ref"] == root.ref and span["source_revision"] == root.revision
        ]

    not_pasted = all_human_direct and all(
        not in_pasted_text(root.content, span["quote"], start=span.get("start"), end=span.get("end"))
        for root in relevant
        for span in spans_of(root)
    )
    if verdict is not None and verdict.attribution in {"third_party", "quoted_or_example"}:
        bound["subject"] = _unresolved_subject(proposal, roots)
        return bound, False, "other_speaker"
    verdict_author_bound = (
        semantic_support
        and not_pasted
        and (
            nominate_only
            or (
                verdict is not None
                and verdict.state == "supported"
                and verdict.attribution == "author"
                and verdict.scope in {"durable", "conditional"}
            )
        )
    )
    # The context around a quote can leave out what marks pasted text, a speaker's label above it or the fence around
    # it: the whole message is asked.
    if not relevant or (
        not verdict_author_bound
        and not any(
            self_report_bound(
                "\n".join(evidence_span_context(root.content, span) for span in spans_of(root)),
                spans_of(root)[0]["quote"] if semantic_support else proposal["value_text"],
                kind=proposal["kind"],
            )
            and not any(
                in_pasted_text(root.content, span["quote"], start=span.get("start"), end=span.get("end"))
                for span in spans_of(root)
            )
            for root in relevant
        )
    ):
        bound["subject"] = _unresolved_subject(proposal, roots)
        return bound, False, "subject_not_bound"
    principals = tuple(verified_human_ref(root) for root in relevant)
    if any(principal is None for principal in principals):
        bound["subject"] = _unresolved_subject(proposal, roots)
        return bound, False, "source_identity_unresolved"
    unique = set(principals)
    if len(unique) != 1:
        bound["subject"] = _unresolved_subject(proposal, roots)
        return bound, False, "source_identity_conflict"
    bound["subject"] = next(iter(unique))
    return bound, True, None


def effective_origin(root: RootEvidence) -> str:
    return (
        (root.original_origin or "origin_unknown")
        if root.origin == "imported" and root.import_verified
        else ("origin_unknown" if root.origin == "imported" else root.origin)
    )


def grounded_time(value: str | None, roots: tuple[RootEvidence, ...]) -> bool:
    if value is None:
        return True
    canonical_value = canonical_time(value)
    if canonical_value is None:
        return False
    stamp = datetime.fromisoformat(canonical_value)
    for root in roots:
        if root.occurred_at is not None and canonical_time(root.occurred_at) == canonical_time(value):
            return True
        text = root.content
        dates = (stamp.strftime("%Y-%m-%d"), f"{stamp.year}年{stamp.month}月{stamp.day}日")
        if any(date in text for date in dates):
            return True
        if root.occurred_at is not None:
            canonical_source_time = canonical_time(root.occurred_at)
            if canonical_source_time is None:
                continue
            source_time = datetime.fromisoformat(canonical_source_time)
            if source_time.year == stamp.year and f"{stamp.month}月{stamp.day}日" in text:
                return True
            if (
                "今天" in text
                and source_time.date() == stamp.date()
                and stamp.hour == stamp.minute == stamp.second == stamp.microsecond == 0
            ):
                return True
            # End of the explicitly stated one-day exception is the following
            # UTC boundary only when the source itself supplies a UTC timestamp.
            if (
                "只限今天" in text
                and (stamp.date() - source_time.date()).days == 1
                and stamp.hour == stamp.minute == stamp.second == stamp.microsecond == 0
            ):
                return True
    return False


def evidence_context(content: str, quote: str) -> str:
    """Include the quote's sentence so trimming cannot discard its negation.

    Unrelated sentences are not evidence for this assertion. A preceding label
    ending in a colon belongs to the quote (for example a fictional scenario).
    Repeated indistinguishable quotations retain all their possible contexts.
    """
    contexts = []
    start = 0
    for _ in range(128):
        at = content.find(quote, start)
        if at < 0:
            break
        before = list(re.finditer(r"[。！？!?;；\n]", content[:at]))
        left = before[-1].end() if before else 0
        quote_end = at + len(quote)
        if quote.rstrip(" \t\r\"“”’'").endswith(tuple("。！？!?;；\n")):
            right = quote_end
        else:
            following = re.search(r"[。！？!?;；\n]", content[quote_end:])
            right = quote_end + following.end() if following else len(content)
        prefix = content[:left].rstrip()
        if prefix.endswith((":", "：")):
            header = re.split(r"[。！？!?;；\n]", prefix)[-1]
            left = max(0, left - len(header) - 1)
        contexts.append(content[left:right])
        start = at + max(1, len(quote))
    if content.find(quote, start) >= 0:
        raise ContractError("DERIVATION_INVALID", "ambiguous_quote_locations")
    return "\n".join(dict.fromkeys(contexts))


#: Real sentence punctuation *plus* the two-character escapes that stand in for
#: it when a tool envelope is stored as a JSON string body.
_ASSERTION_BREAK = re.compile(r"[。！？!?;；\n]|\\[nrt]")


def _ends_clause(quote: str) -> bool:
    """Whether the quote's own final character already closes its clause."""
    trimmed = quote.rstrip(" \t\r\"\u201c\u201d\u2019'")
    if not trimmed:
        return False
    breaks = list(_ASSERTION_BREAK.finditer(trimmed))
    return bool(breaks) and breaks[-1].end() == len(trimmed)


def assertion_clause(content: str, quote: str) -> str:
    """Return the clause the quote sits in, treating escaped breaks as breaks.

    Treating escaped line breaks as clause boundaries prevents an unrelated
    question elsewhere in a serialized JSON body from vetoing the quoted
    assertion.  ``evidence_context`` retains the wider context for negation,
    polarity, and condition checks.

    This narrower context is for checks of the assertion itself.  Splitting on
    ``\\n`` also splits a literal Windows path such as ``C:\\new``, which can
    only narrow what such a check reads.  Negation, polarity and condition checks
    use the wider context instead.
    """
    clauses = []
    start = 0
    closes = _ends_clause(quote)
    for _ in range(128):
        at = content.find(quote, start)
        if at < 0:
            break
        before = list(_ASSERTION_BREAK.finditer(content, 0, at))
        left = before[-1].end() if before else 0
        end = at + len(quote)
        # A quote whose own last character is the break already ends its
        # clause.  Searching onwards from there would find the *next*
        # sentence's terminator and pull that whole sentence in, so a question
        # standing in the sentence after a correct assertion would veto it.
        # ``evidence_context`` has the same guard.
        if closes:
            right = end
        else:
            after = _ASSERTION_BREAK.search(content, end)
            right = after.end() if after else len(content)
        clauses.append(content[left:right])
        start = at + max(1, len(quote))
    return "\n".join(dict.fromkeys(clauses)) if clauses else quote


#: Statement kinds that state no value of their slot; qualification keeps them proposed (``statement_not_asserted``).
UNASSERTED_STATEMENTS = frozenset({"request", "proposal", "hypothetical", "quotation", "fictional", "unknown"})
HUMAN_ONLY_KINDS = frozenset({"preference", "constraint", "decision", "intention", "alias"})
#: Origins a cited root may lend authority from; ``_authority`` needs one of them.
AUTHORITY_ORIGINS = ("human_direct", "tool_observation", "external_document")
_POLARITY_KINDS = frozenset({"fact", "preference", "constraint", "decision"})


@dataclass(frozen=True)
class _Cited:
    """One proposal against the complete root spans it actually cites.

    ``roots`` are narrowed to the sentence radius around each quote; ``quoted``
    is the quotes themselves; ``asserted`` is the clause each quote was lifted
    from, taken before narrowing because the negation and polarity rules keep
    the wider radius.  ``human``/``observed``/``documents`` partition the roots
    by effective origin.
    """

    proposal: ClaimProposal
    roots: tuple[RootEvidence, ...]
    quoted: str
    asserted: str
    cited_spans: tuple[tuple[str, str], ...]
    source_text: str
    human: tuple[RootEvidence, ...]
    observed: tuple[RootEvidence, ...]
    documents: tuple[RootEvidence, ...]
    subject_bound: bool
    explicit_attribute_correction: bool

    @property
    def kind(self) -> str:
        return self.proposal["kind"]

    @property
    def subject(self) -> str:
        return self.proposal["subject"]


def evidence_span_context(content, span):
    """Read the complete sentence around the selected occurrence, preserving a preceding speaker label."""
    if "start" not in span:
        return evidence_context(content, span["quote"])
    start, end = span["start"], span["end"]
    before = list(re.finditer(r"[。！？!?;；\n]", content[:start]))
    left = before[-1].end() if before else 0
    after = re.search(r"[。！？!?;；\n]", content[end:])
    right = end if span["quote"].endswith(tuple("。！？!?;；\n")) else end + after.end() if after else len(content)
    prefix = content[:left].rstrip()
    if prefix.endswith((":", "：")):
        left = max(0, left - len(re.split(r"[。！？!?;；\n]", prefix)[-1]) - 1)
    return content[left:right]


def evidence_span_neighbourhood(content, span):
    """Keep legacy source context or neighbours around the verified occurrence for entity binding."""
    if "start" not in span:
        return content
    start, end = span["start"], span["end"]
    before = list(re.finditer(r"[。！？!?;；\n]", content[:start]))
    left = before[-2].end() if len(before) >= 2 else 0
    after = re.search(r"[。！？!?;；\n]", content[end:])
    right = end + after.end() if after else len(content)
    return content[left:right]


def _cite(
    proposal: ClaimProposal,
    roots: tuple[RootEvidence, ...],
    *,
    subject_bound: bool,
    explicit_attribute_correction: bool,
) -> _Cited | None:
    """Pair every cited span with its complete root; None when no root qualifies."""
    spans = proposal["evidence_spans"]

    def cited(root: RootEvidence):
        return [span for span in spans if span["source_ref"] == root.ref and span["source_revision"] == root.revision]

    # A summary may link to a root for lineage, but cannot lend its own wording
    # that root's authority.  Promotion requires a span on the actual root.
    roots = tuple(r for r in roots if cited(r) and r.capture_state == "complete" and not r.capture_gaps)
    if not roots:
        return None
    narrowed = tuple(
        replace(r, content="\n".join(evidence_span_context(r.content, span) for span in cited(r))) for r in roots
    )
    by_origin = {origin: tuple(r for r in narrowed if effective_origin(r) == origin) for origin in AUTHORITY_ORIGINS}
    return _Cited(
        proposal,
        narrowed,
        quoted="\n".join(span["quote"] for span in spans),
        asserted="\n".join(
            assertion_clause(evidence_span_context(r.content, span), span["quote"]) for r in roots for span in cited(r)
        ),
        cited_spans=tuple(
            (span["quote"], evidence_span_neighbourhood(r.content, span)) for r in roots for span in cited(r)
        ),
        source_text="\n".join(r.content for r in narrowed),
        human=by_origin["human_direct"],
        observed=by_origin["tool_observation"],
        documents=by_origin["external_document"],
        subject_bound=subject_bound,
        explicit_attribute_correction=explicit_attribute_correction,
    )


# Each rule answers with the refusal reason, or None.  Order matters: the first
# refusal wins, and later rules assume the earlier ones passed.
def _not_asserted(c: _Cited) -> str | None:
    if c.proposal["statement_kind"] in UNASSERTED_STATEMENTS:
        return "statement_not_asserted"
    if AUTHORITY_QUESTION.search(c.asserted or c.source_text):
        return "question_not_asserted"
    if (
        _HYPOTHETICAL.search(c.source_text)
        or _UNDECIDED.search(c.source_text)
        or UNASSERTED_UNCERTAINTY.search(c.source_text)
    ):
        return "hypothetical_or_undecided"
    if is_transient_request(c.source_text):
        return "transient_request_not_durable"
    if (
        c.kind in {"preference", "constraint", "decision"} and _QUOTED_OTHER.search(c.source_text)
    ) or attributed_to_another(c.source_text):
        return "other_speaker"
    return None


def _scope_preserved(c: _Cited) -> str | None:
    conditions, valid_to = c.proposal["conditions"], c.proposal["valid_to"]
    if any(
        not condition_supports_value(c.source_text, condition, c.proposal["value_text"]) for condition in conditions
    ):
        return "condition_not_supported"
    # "要是值班表调整，告警频道是 X" holds only under its condition: a claim that drops it would say X holds now.
    if CONDITIONAL.search(c.source_text) and not conditions:
        return "condition_not_preserved"
    if _LIMIT.search(c.source_text) and not conditions and valid_to is None:
        return "missing_limitation"
    for marker in RELATIVE_SCOPE.finditer(c.source_text):
        if valid_to is None and not any(bound_literal(condition, marker.group()) for condition in conditions):
            return "relative_scope_not_preserved"
    return None


def _authority(c: _Cited, *, semantic_support: bool = False) -> str | None:
    if not (c.human or c.observed or c.documents):
        return "no_independent_authority"
    if c.kind in HUMAN_ONLY_KINDS and not c.human:
        return "requires_human_source"
    if not semantic_support and any(
        not grounded_time(c.proposal.get(field), c.roots) for field in ("valid_from", "valid_to", "occurred_at")
    ):
        return "time_not_grounded"
    return None


def _subject_binds(c: _Cited, *, normalized: bool = False) -> str | None:
    # Aliases have a separate, exact old-name -> target proof in apply_claim;
    # that proof may name the sourced old label instead of its internal subject.
    if c.subject_bound or c.kind == "alias":
        return None
    normalize = (lambda text: unicodedata.normalize("NFKC", text).casefold()) if normalized else (lambda text: text)
    subject, kind = normalize(c.subject), c.kind
    quoted = normalize(c.quoted)
    spans = tuple((normalize(quote), normalize(content)) for quote, content in c.cited_spans)
    self_bound = subject.casefold() in SELF_SUBJECTS and self_report_bound(
        c.source_text, c.proposal["value_text"], kind=kind
    )
    if subject in quoted and not bound_literal(quoted, subject):
        return "subject_not_bound"
    if subject == "我" and not self_bound:
        return "subject_not_bound"
    # The subject may also be stated verbatim in the sentence next to the quote,
    # in the same source; see core/subject_binding.py.  Nothing is inferred: only
    # literal text from the cited source counts, and a subject that merely
    # points ("you", "the agent") is refused at any distance.
    if not bound_literal(quoted, subject) and not neighbourhood_binds(subject, spans) and not self_bound:
        return "subject_not_bound"
    return None


def _intention_proved(c: _Cited) -> str | None:
    if c.kind != "intention":
        return None
    intention = c.proposal.get("intention")
    if intention is None:
        return "intention_missing"
    if intention["target"] not in c.source_text and c.proposal["predicate"] not in c.source_text:
        return "intention_target_unproved"
    state = intention["state"]
    if state == "pending" and intention["cue"] not in c.source_text:
        return "intention_cue_unproved"
    state_sources = set(intention["state_evidence_refs"])
    authorized = "\n".join(
        r.content
        for r in c.roots
        if f"{r.ref}@{r.revision}" in state_sources and effective_origin(r) in {"human_direct", "tool_observation"}
    )
    if state == "completed" and (
        not asserted_marker(_COMPLETION, authorized) or _NEGATED_COMPLETION.search(authorized)
    ):
        return "completion_unproved"
    if state == "cancelled" and not asserted_marker(_CANCELLATION, authorized):
        return "cancellation_unproved"
    if state == "expired" and c.proposal["valid_to"] is None:
        return "expiry_unproved"
    return None


def _procedure_proved(c: _Cited) -> str | None:
    if c.kind != "procedure":
        return None
    procedure = c.proposal.get("procedure")
    if procedure is None:
        return "procedure_missing"
    if any(step not in c.source_text for step in procedure["method"]):
        return "method_steps_unproved"
    if any(condition not in c.source_text for condition in procedure["non_applicable"]):
        return "method_exception_unproved"
    verification = procedure["verification_basis"]
    if verification == "user_accepted" and not any(asserted_marker(_ACCEPTANCE, r.content) for r in c.human):
        return "method_acceptance_unproved"
    if verification == "observed_once" and not c.observed:
        return "method_observation_unproved"
    if verification in {"inferred_suggestion", "unknown"}:
        return "method_is_suggestion"
    return None


def _fact_entailed(c: _Cited) -> str | None:
    """Ordinary durable facts keep the claim-frame support check.  A person's negation of themselves, a denial or a
    state ("I don't have a car", 我不吃辣, 我没有车), holds its verb in its value, so no frame orders subject, relation
    and value in it; bound to their own clause (``bind_claim_subject``), it stands on that as their preference does."""
    if c.kind != "fact" or c.explicit_attribute_correction or (c.subject_bound and _negation(c.proposal) is not None):
        return None
    try:
        draft = ClaimDraft.from_parts(
            subject=c.subject,
            predicate=c.proposal["predicate"],
            value=c.proposal["value_text"],
            scope_id="qualification-only",
        )
        evidence = tuple(
            EvidenceReference(
                "direct_user" if effective_origin(root) == "human_direct" else "external_record",
                root.ref,
                evidence_context(root.content, span["quote"]),
                c.subject if c.subject_bound and verified_human_ref(root) == c.subject else "",
            )
            for root in (*c.human, *c.observed, *c.documents)
            for span in c.proposal["evidence_spans"]
            if span["source_ref"] == root.ref and span["source_revision"] == root.revision
        )
    except ValueError:
        return "fact_entailment_unproved"
    if any(evidence_supports_claim(item, draft) for item in evidence) or any(
        evidence_supports_relation(item, draft) for item in evidence
    ):
        return None
    return "fact_entailment_unproved"


def _value_preserved(c: _Cited) -> str | None:
    value, conditions = c.proposal["value_text"], c.proposal["conditions"]
    if c.kind not in VALUE_FREE_KINDS and value not in c.quoted:
        return "value_not_supported_by_quote"
    if c.kind in _POLARITY_KINDS and not preserves_qualifiers(
        c.source_text, value, conditions=conditions, polarity_only=True
    ):
        return "value_polarity_not_preserved"
    if c.kind in {"preference", "decision"} and _NEGATION.search(c.source_text) and not _NEGATION.search(value):
        if not rejects_other_value(c.source_text, c.proposal):
            return "negation_not_preserved"
    return None


def rejects_other_value(content, proposal):
    """An explicit 'X, not Y' does not negate the positive X clause.

    A general denial of the statement remains a denial. Every negative clause
    must be an isolated rejected alternative; questions/uncertainty are still
    checked by the owning qualifier.
    """
    clauses = re.split(r"[，,;；。!?！？\n]", content)
    subject, predicate, value = (proposal[k] for k in ("subject", "predicate", "value_text"))
    frame = re.compile(
        re.escape(subject) + r"\s*(?:应当|应该|应|仍然|仍)?" + re.escape(predicate) + r"\s*" + re.escape(value)
    )
    if not any(frame.search(c) and not _NEGATION.search(c) for c in clauses):
        return False
    negatives = [c.strip() for c in clauses if _NEGATION.search(c)]
    if not negatives:
        return False
    for clause in negatives:
        match = re.fullmatch(r"(?:不是|并非)([^\s，,。;；!?！？]{1,120})", clause)
        if (
            not match
            or value in match.group(1)
            or re.search(r"事实|确认|真的|真实|认真|意思|偏好|断言|承诺|决定|成立|正确", match.group(1))
        ):
            return False
    return True


def _time_scope_known(c: _Cited) -> str | None:
    temporal = classify_durable_state_clause(c.source_text)
    valid_from, valid_to = c.proposal["valid_from"], c.proposal["valid_to"]
    if temporal == "past" and valid_from is None:
        return "historical_start_unknown"
    if temporal == "future" and valid_from is None and c.kind != "intention":
        return "future_start_unknown"
    if temporal == "temporary" and valid_to is None and not c.proposal["conditions"]:
        return "temporary_scope_unknown"
    return None


_REFUSALS = (
    _not_asserted,
    _scope_preserved,
    _authority,
    _subject_binds,
    _intention_proved,
    _procedure_proved,
    _fact_entailed,
    _value_preserved,
    _time_scope_known,
)


def _reconcile_structured_verdict(cited: _Cited, verdict: SupportVerdict) -> str | None:
    """Enforce structural governance against a 7-key semantic support verdict without wordlists."""
    if cited.proposal["statement_kind"] in UNASSERTED_STATEMENTS:
        return "statement_not_asserted"
    if verdict.attribution != "author":
        return "other_speaker"
    if verdict.scope == "unasserted":
        clause = cited.asserted or cited.source_text
        return "question_not_asserted" if ("?" in clause or "？" in clause) else "hypothetical_or_undecided"
    if verdict.scope == "ephemeral_turn" and cited.proposal["valid_to"] is None and not cited.proposal["conditions"]:
        return "relative_scope_not_preserved"
    if verdict.scope == "conditional" and not cited.proposal["conditions"]:
        return "qualified_statement"
    for rule in (
        _scope_preserved,
        partial(_authority, semantic_support=True),
        partial(_subject_binds, normalized=True),
        _intention_proved,
        _procedure_proved,
        _time_scope_known,
    ):
        reason = rule(cited)
        if reason is not None:
            return reason
    return None


def _qualification_reason(cited, normalized, support_state, verdict: SupportVerdict | None = None):
    """Select semantic support gates without allowing a model to override source governance."""
    rules = _REFUSALS
    # No semantic verdict must not erase a known literal refusal merely because
    # the quote now has positions. Capability failures retain their own state.
    if normalized and support_state != "unavailable":
        has_traits = verdict is not None and all(
            trait is not None for trait in (verdict.attribution, verdict.polarity, verdict.scope)
        )
        if has_traits:
            assert verdict is not None
            if support_state != "supported":
                return "other_speaker" if verdict.attribution != "author" else "evidence_support_" + str(support_state)
            return _reconcile_structured_verdict(cited, verdict)
        # Positions locate evidence; they do not invalidate a literal negative
        # state or a fully stated condition that the ordinary rules already prove.
        if support_state == "supported" and all(rule(cited) is None for rule in rules):
            return None
        if POLARITY.search(cited.source_text) or CONDITIONAL.search(cited.source_text):
            return "qualified_statement"
        if support_state != "supported":
            return "evidence_support_" + str(support_state)
        rules = (
            _not_asserted,
            _scope_preserved,
            partial(_authority, semantic_support=True),
            partial(_subject_binds, normalized=True),
            _intention_proved,
            _procedure_proved,
            _time_scope_known,
        )
    for rule in rules:
        reason = rule(cited)
        if reason is not None:
            return reason
    return None


def qualify(
    proposal: ClaimProposal,
    roots: tuple[RootEvidence, ...],
    *,
    project_id: str | None = None,
    explicit_attribute_correction: bool = False,
    _subject_bound: bool = False,
    _support_state: str | None = None,
    _verdict: SupportVerdict | None = None,
    _legacy: bool = False,
) -> Qualification:
    """Source roles are necessary, never sufficient, to promote a proposal.

    P01 has already checked exact evidence spans against these source versions.
    This rule retains unsupported interpretation as proposed. It neither invents
    dates nor converts the model's requested state into governance authority.
    """
    if proposal["kind"] == "observation":
        return Qualification("proposed", "derived_summary", "reflection_pending")
    normalized = positioned(proposal) and not _legacy
    verdict = _verdict
    if normalized and verdict is None:
        verdict = support_verdict(proposal, roots)
    if normalized and _support_state is None:
        _support_state = verdict.state if verdict else "unavailable"
    if not _subject_bound:
        bound_proposal, subject_bound, binding_issue = bind_claim_subject(
            proposal, roots, semantic_support=normalized, verdict=verdict
        )
        if binding_issue is not None:
            return Qualification("proposed", "inferred_suggestion", binding_issue)
        if subject_bound:
            return qualify(
                bound_proposal,
                roots,
                project_id=project_id,
                explicit_attribute_correction=explicit_attribute_correction,
                _subject_bound=True,
                _support_state=_support_state,
                _verdict=verdict,
                _legacy=_legacy,
            )
    cited = _cite(
        proposal, roots, subject_bound=_subject_bound, explicit_attribute_correction=explicit_attribute_correction
    )
    if cited is None:
        return Qualification("proposed", "inferred_suggestion", "no_complete_root_span")
    reason = _qualification_reason(cited, normalized, _support_state, verdict=verdict)
    if reason is not None:
        return Qualification("proposed", "inferred_suggestion", reason)
    if cited.human:
        reason = "normalized_evidence_supported" if normalized else "explicit_scoped_source"
        return Qualification("active", "direct_report", reason)
    return Qualification("active", "observed", "observation_at_source_time")


def valid_at(version: ClaimVersion, instant: str, *, historical: bool) -> bool:
    when = canonical_time(instant)
    start, end = canonical_time(version.valid_from), canonical_time(version.valid_to)
    if historical and start is None:
        return False
    return (start is None or (when is not None and start <= when)) and (
        end is None or (when is not None and when < end)
    )


def select_proposal(versions: tuple[ClaimVersion, ...], instant: str) -> ClaimVersion | None:
    """The head of a claim that qualification has never promoted.

    ``select_effective`` deliberately returns None for these, and it is right to:
    a proposal is not an effective version and must never be answered as if it
    were settled. But that alone removes them from recall entirely, and in a real
    store almost the whole derived layer can sit in this state (231 proposed
    against 11 active in one), so most of what consolidation produces would be
    invisible to every caller, with no error and no gap to notice it by.

    Returning the head here lets retrieval admit it *labelled*, so the reader can
    weigh it, instead of the system silently having nothing to show.

    Not for historical queries: ``as_of`` asks what was true at a past instant,
    and an unpromoted proposal cannot answer that.
    """
    if not versions:
        return None
    heads = [v for v in versions if v.revision == 1 or v.replaces_revision is not None]
    if not heads:
        return None
    head = max(heads, key=lambda v: v.revision)
    if head.state != "proposed" or not valid_at(head, instant, historical=False):
        return None
    return head


def select_effective(
    versions: tuple[ClaimVersion, ...], instant: str, *, as_of: bool = False, known_at: str | None = None
) -> ClaimVersion | None:
    if known_at is not None:
        known_time = canonical_time(known_at)
        if known_time is None:
            return None
        versions = tuple(
            v for v in versions if (recorded := canonical_time(v.recorded_from)) is not None and recorded <= known_time
        )
    if not versions:
        return None
    heads = [v for v in versions if v.revision == 1 or v.replaces_revision is not None]
    head = max(heads, key=lambda v: v.revision)
    if not as_of and known_at is None and head.state == "superseded" and head.reason == "health_superseded":
        return None
    # A replacement with an unknown effective date can establish an explicitly
    # current assertion, but supplies no answer about a past date.
    if as_of and head.valid_from is None and head.state != "proposed":
        return None
    if head.state != "proposed" and valid_at(head, instant, historical=as_of):
        return None if head.state == "retracted" else head
    candidates = [
        v
        for v in versions
        if v.state in {"active", "superseded", "disputed", "retracted"}
        and valid_at(v, instant, historical=as_of or head.state == "disputed")
    ]
    if not candidates:
        return None
    chosen = max(candidates, key=lambda v: (canonical_time(v.valid_from) or "", v.revision))
    return None if chosen.state == "retracted" else chosen


def dated_after(new_start, old_start, new_times, old_times) -> bool:
    """Whether an assertion is known to come after another: by its stated start, or every source of it said after
    every source of the other.  Starts and times are ``canonical_time`` values; None and empty lists say nothing."""
    return bool(new_start and old_start and new_start > old_start) or bool(
        new_times and old_times and min(new_times) > max(old_times)
    )


def states_a_value(version: ClaimVersion) -> bool:
    """Whether a claim version says what its slot holds: a kind that has a value (not a procedure, an intention or an
    alias), asserted (no hypothetical, quotation or other of ``UNASSERTED_STATEMENTS``), no retraction, and no denial
    (``is_denial``) unless it was taken as the slot's value, as the person's own "我不吃辣" is (active).  A proposal the
    fact gate could not prove still states its value."""
    payload = version.payload
    return (
        payload.get("kind") not in VALUE_FREE_KINDS
        and payload.get("statement_kind") not in UNASSERTED_STATEMENTS
        and version.state != "retracted"
        and (version.state == "active" or not is_denial(payload))
    )


def holds_instead(versions: tuple[ClaimVersion, ...], revision: int, instant: str) -> ClaimVersion | None:
    """The version of a claim that holds at ``instant`` in place of the value its version ``revision`` states: the
    effective one (``select_effective``) when it is active and its value differs in its letters and digits
    (``nfkc_letters_and_digits``).  None while that value still holds, a replacement that starts later included, and
    while no active version holds: the slot holds only proposals, a dispute or a retraction.  Whether the version was
    said after the stated one is the caller's to check (``dated_after``)."""
    stated = next((version for version in versions if version.revision == revision), None)
    effective = select_effective(versions, instant)
    if stated is None or effective is None or effective.state != "active" or effective.revision == revision:
        return None
    held, said = (nfkc_letters_and_digits(v.payload.get("value_text")) for v in (effective, stated))
    return None if held == said else effective
