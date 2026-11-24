"""Pure admission and ranking policy for the P08 retrieval pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

from ..contracts import ContractError
from .embedding_budget import bounded_embedding_text, kept_length
from .endpoint_policy import endpoint_url_shape_ok
from .events import CJK_RUN, lexical_terms, query_terms, version_suffixes
from .owner_view import OWNER_NAME_LENGTH, OWNER_NAMES_LIMIT
from .retrieval import CandidateRef, SearchContext
from .secret_patterns import secret_like_text_starts_before

EMBEDDING_SPACE = {
    "model": "gemini-embedding-2",
    "dimensions": 3072,
    "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents",
    "task_type": None,
    "prompt_encoding": {
        "document": "title: none | text: {NFKC source}",
        "query": "task: question answering | query: {NFKC query}",
    },
    "input_preprocessing": {"id": "nfkc-v1", "unicode_version": "15.0.0"},
    "metric": "cosine",
    "vector_normalization": "l2_at_scoring",
    "request_encoding": {
        "id": "gemini002-native-batch-v1",
        "task_type_field": None,
        "task_type_location": None,
        "prompt_encoding": "official-question-answering-v1",
    },
}
_SPACE_BYTES = json.dumps(EMBEDDING_SPACE, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
SPACE_ID = hashlib.sha256(_SPACE_BYTES).hexdigest()
VECTOR_SCORE_TOLERANCE = 1e-6
#: The score a recall's best vector hit must reach before the vector channel admits anything.  Gemini embedding 2
#: scores sit between 0.6 and 0.8: on LongMemEval-S a question's best hit in its own history has a median of 0.72-0.74,
#: its best hit in another question's history a median of 0.62 and a 75th percentile of 0.65, and over half of the
#: turns holding an answer score below 0.70.  On the 150-question tuning replay, 0.66 found 0.78 of the answer turns
#: against 0.69 at 0.70, and 0.64 only 0.01 more while injecting into 4 % more unrelated histories.
BUILTIN_VECTOR_THRESHOLD = 0.66
#: Once the best hit reaches the threshold, the hits ranked in the channel's first ``BUILTIN_VECTOR_NEAR_RANK`` are
#: admitted down to ``BUILTIN_VECTOR_MARGIN`` below the threshold.  The turns holding an answer rank in a question's
#: first six 80-89 % of the time, yet half of them score below 0.70.  Rank rather than distance from the best hit,
#: because an earlier copy of the question itself scores near 1 and would push every other hit out of a margin.
BUILTIN_VECTOR_NEAR_RANK = 6
BUILTIN_VECTOR_MARGIN = 0.10
#: Minimum source count before mature-corpus statistical rules engage; smaller stores keep young-instance floors.
MATURE_CORPUS_SIZE = 512
#: Corpus size at which ``BUILTIN_VECTOR_THRESHOLD`` was calibrated before extreme-value cosine drift sets in.
VECTOR_CALIBRATION_CORPUS_SIZE = 2048
#: Minimum self-information ratio relative to a query's rarest specific term for primary anchor membership.
AUTO_ANCHOR_INFO_RATIO = 0.88
#: Self-information (nats) above which a specific term (~<0.055% corpus prevalence) is intrinsically a primary anchor.
AUTO_INTRINSIC_ANCHOR_INFO = 7.5
#: Maximum cosine margin above the vector threshold required on large stores without a lexical primary anchor.
AUTO_STRONG_VECTOR_MARGIN = 0.06
#: Minimum specific or total self-information (nats) to anchor cross-session raw event recall on a mature store.
AUTO_ONGOING_SPECIFIC_INFO_FLOOR = 11.5
AUTO_ONGOING_PEAK_INFO_FLOOR = 8.2
AUTO_ONGOING_TOTAL_INFO_FLOOR = 45.0

_WEAK_QUERY = frozenset({"那次", "那件", "那个", "这个", "怎样", "如何", "what", "that", "it"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{1,239}")
_HARD_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]{1,12}\d[A-Za-z0-9._/-]*|\d+[A-Za-z][A-Za-z0-9._/-]*)(?![A-Za-z0-9_])"
)
_CJK_TEXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_IDENTIFIER_JOINT = re.compile(r"(?<=[A-Za-z])[ \t_-](?=\d)")
_CLAUSE_BOUNDARY = re.compile(r"[，,；;。！？!?\n]+")
# Stop at adjacent non-ASCII prose and common Markdown delimiters. Percent
# escapes stay in the URL, so an encoded path segment retains its identity.
_HTTP_LINK = re.compile(r"https?://[a-z0-9._~:/?#\[\]@!$&*+;=%-]+", re.ASCII | re.I)
_RECORD_PATH = re.compile(r"(?P<parent>.*)/(?P<number>[0-9]+)(?:/(?:files|diff|changes|commits|checks))?/?")
_POST_PATH = re.compile(r"/(?:[^/]+|i/web)/status/(?P<number>[0-9]+)(?:/(?:photo|video)/[0-9]+)?/?")
_POST_SITES = frozenset({"x.com", "www.x.com", "mobile.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"})


def _validate_prompt_encoding(prompt_encoding: object) -> None:
    if type(prompt_encoding) is not dict or set(prompt_encoding) != {"document", "query"}:
        raise ContractError("INPUT_INVALID", "embedding_space")
    for prompt_value in prompt_encoding.values():
        if type(prompt_value) is not str or len(prompt_value) > 256 or any(ch in prompt_value for ch in ("\r", "\x00")):
            raise ContractError("INPUT_INVALID", "embedding_space")


def canonical_embedding_space(value: dict) -> dict:
    """Return the frozen descriptor with a stable key order and strict fields."""

    if type(value) is not dict or set(value) != set(EMBEDDING_SPACE):
        raise ContractError("INPUT_INVALID", "embedding_space")
    # Shape and bounds, not identity: requiring every field to equal the module
    # default would make the embedding model unconfigurable, though the vector
    # store keys everything on the space digest and rebuilds cleanly for
    # another provider.  Identity is enforced one layer out, where it belongs —
    # a vector whose space digest differs from the one its `RecallPolicy` is
    # bound to (`embedding_space_id`, the configured route's space in a runtime
    # instance) is refused admission in `RecallPolicy.vector_admission`.
    if type(value.get("model")) is not str or not 1 <= len(value["model"]) <= 200:
        raise ContractError("INPUT_INVALID", "embedding_space")
    if type(value.get("dimensions")) is not int or not 8 <= value["dimensions"] <= 16384:
        raise ContractError("INPUT_INVALID", "embedding_space")
    endpoint = value.get("endpoint")
    if type(endpoint) is not str or not endpoint_url_shape_ok(endpoint) or len(endpoint) > 2048:
        raise ContractError("INPUT_INVALID", "embedding_space")
    if value.get("task_type") is not None:
        raise ContractError("INPUT_INVALID", "embedding_space")
    _validate_prompt_encoding(value.get("prompt_encoding"))
    if type(value.get("input_preprocessing")) is not dict or set(value["input_preprocessing"]) != {
        "id",
        "unicode_version",
    }:
        raise ContractError("INPUT_INVALID", "embedding_space")
    if value["input_preprocessing"] != EMBEDDING_SPACE["input_preprocessing"]:
        raise ContractError("INPUT_INVALID", "embedding_space")
    if value.get("metric") != "cosine" or value.get("vector_normalization") != "l2_at_scoring":
        raise ContractError("INPUT_INVALID", "embedding_space")
    if type(value.get("request_encoding")) is not dict or set(value["request_encoding"]) != {
        "id",
        "task_type_field",
        "task_type_location",
        "prompt_encoding",
    }:
        raise ContractError("INPUT_INVALID", "embedding_space")
    if type(value["request_encoding"].get("id")) is not str or not value["request_encoding"]["id"]:
        raise ContractError("INPUT_INVALID", "embedding_space")
    if (
        value["request_encoding"]["task_type_field"] is not None
        or value["request_encoding"]["task_type_location"] is not None
    ):
        raise ContractError("INPUT_INVALID", "embedding_space")
    if (
        type(value["request_encoding"].get("prompt_encoding")) is not str
        or not value["request_encoding"]["prompt_encoding"]
    ):
        raise ContractError("INPUT_INVALID", "embedding_space")
    result = {
        "model": value["model"],
        "dimensions": value["dimensions"],
        "endpoint": value["endpoint"],
        "task_type": None,
        "prompt_encoding": {
            "document": value["prompt_encoding"]["document"],
            "query": value["prompt_encoding"]["query"],
        },
        "input_preprocessing": {
            "id": value["input_preprocessing"]["id"],
            "unicode_version": value["input_preprocessing"]["unicode_version"],
        },
        "metric": value["metric"],
        "vector_normalization": value["vector_normalization"],
        "request_encoding": {
            "id": value["request_encoding"]["id"],
            "task_type_field": value["request_encoding"]["task_type_field"],
            "task_type_location": value["request_encoding"]["task_type_location"],
            "prompt_encoding": value["request_encoding"]["prompt_encoding"],
        },
    }
    return result


def embedding_space_id(value: dict) -> str:
    canonical = canonical_embedding_space(value)
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


#: Wire dialects the embedding adapter can speak. "gemini" is Google's native
#: batchEmbedContents shape; "openai" is the /v1/embeddings shape that MiniMax,
#: Qwen, OpenAI and most other providers accept.
EMBEDDING_DIALECTS = frozenset({"gemini", "openai"})


def build_embedding_space(
    *,
    model: str,
    dimensions: int,
    endpoint: str,
    dialect: str = "gemini",
    prompt_encoding: Mapping[str, str] | None = None,
) -> dict:
    """Describe one embedding space so its digest can key the vector store.

    Everything about how text reaches a model belongs in this descriptor,
    because the digest of it is the vector directory name. Change the model, the
    dimensionality or the request shape and the digest changes, the store moves
    to a new directory, and vectors from the old space are refused admission
    instead of being silently compared across incompatible geometries. That
    property is what makes swapping embedding models safe.
    """
    if dialect not in EMBEDDING_DIALECTS:
        raise ContractError("INPUT_INVALID", "embedding_dialect")
    resolved_encoding = dict(EMBEDDING_SPACE["prompt_encoding"]) if prompt_encoding is None else dict(prompt_encoding)
    encoding_id = (
        EMBEDDING_SPACE["request_encoding"]["prompt_encoding"]
        if resolved_encoding == EMBEDDING_SPACE["prompt_encoding"]
        else "configured-prefix-v1"
    )
    return canonical_embedding_space(
        {
            "model": model,
            "dimensions": dimensions,
            "endpoint": endpoint,
            "task_type": None,
            "prompt_encoding": resolved_encoding,
            "input_preprocessing": dict(EMBEDDING_SPACE["input_preprocessing"]),
            "metric": "cosine",
            "vector_normalization": "l2_at_scoring",
            "request_encoding": {
                "id": f"{dialect}-embed-v1",
                "task_type_field": None,
                "task_type_location": None,
                "prompt_encoding": encoding_id,
            },
        }
    )


def _bound_input(raw_text: object) -> str:
    """``raw_text`` as the input bound cuts it and every embedded body carries it: in NFKC form."""
    if type(raw_text) is not str:
        raise ContractError("INPUT_INVALID", "embedding_text")
    return unicodedata.normalize("NFKC", raw_text)


def encode_embedding_text(raw_text: str, *, kind: str, prompt_encoding: Mapping[str, str] | None = None) -> str:
    """Encode raw source/query text without changing raw identity.

    The one choke point every embedded body passes through -- source, claim and
    query alike -- which is why the input bound lives here rather than in each
    caller.  Without a bound, an oversized source is permanently unembeddable; see
    ``core/embedding_budget.py``.
    """

    if kind not in {"document", "query"}:
        raise ContractError("INPUT_INVALID", "embedding_text")
    bounded, _truncated = bounded_embedding_text(_bound_input(raw_text))
    if prompt_encoding is not None:
        template = prompt_encoding.get(kind)
        if type(template) is str:
            expected_suffix = "{NFKC query}" if kind == "query" else "{NFKC source}"
            if template.endswith(expected_suffix):
                return f"{template.removesuffix(expected_suffix)}{bounded}"
            for placeholder in ("{NFKC query}", "{NFKC source}", "{text}"):
                if placeholder in template:
                    return template.replace(placeholder, bounded, 1)
            return f"{template}{bounded}"
    if kind == "query":
        return f"task: question answering | query: {bounded}"
    return f"title: none | text: {bounded}"


def embedding_sends_secret_like_text(raw_text: str) -> bool:
    """Whether the part of ``raw_text`` that ``encode_embedding_text`` sends holds secret-like text.

    A run counts when it starts in the part the input bound keeps, and is judged with the rest of the line the cut
    falls in (``secret_like_text_starts_before``); the template around the text and the marker after a cut are this
    module's own words.  Judged in the encoded body, a cut right after a key name puts the marker where its value
    would be (``credential_env: …[truncated]``), and a cut inside a value exempt only whole leaves its start
    (``credential_env: nu`` of ``null``); either reads as a credential, and a text is cut in the same place every
    time, so its source could never be embedded.  Judged with the rest of its line, a key the cut runs through is
    caught although too few of its characters are sent for its pattern.
    """
    text = _bound_input(raw_text)
    return secret_like_text_starts_before(text, kept_length(text))


def claim_embedding_text(payload: dict) -> str:
    """Render one claim as the assertion it makes, for embedding.

    The stored payload is JSON; embedding it verbatim would index braces and
    field names rather than meaning. Subject, predicate and value are the
    assertion, so they lead; conditions follow because they qualify it rather
    than identify it.

    Derived objects are unreachable from automatic recall until they are in the
    vector index: the lexical, vector and recent channels all yield events, so a
    claim's only other route is relation expansion out of an event that happened
    to be retrieved first.
    """
    if not isinstance(payload, dict):
        raise ContractError("INPUT_INVALID", "claim_embedding_text")
    statement = " ".join(
        part for field in ("subject", "predicate", "value_text") if (part := str(payload.get(field) or "").strip())
    )
    conditions = payload.get("conditions")
    if isinstance(conditions, (list, tuple)):
        qualifiers = " ".join(text for item in conditions if (text := str(item).strip()))
        if qualifiers:
            statement = f"{statement}（{qualifiers}）" if statement else qualifiers
    if not statement:
        raise ContractError("DERIVATION_INVALID", "claim_embedding_text")
    return statement


def _finite_score(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return float(value)


@dataclass(frozen=True)
class RecallPolicy:
    """Resolve the built-in space's floor; explicit thresholds override it."""

    #: The score the best vector hit must reach (``BUILTIN_VECTOR_THRESHOLD`` in the built-in space).
    vector_threshold: float | None = None
    #: Ranked hits admitted below the threshold once the best reaches it, and how far below (``BUILTIN_VECTOR_*``).
    vector_near_rank: int = BUILTIN_VECTOR_NEAR_RANK
    vector_margin: float = BUILTIN_VECTOR_MARGIN
    #: Automatic injection needs independent lexical coverage or vector evidence;
    #: fusion ranks admitted evidence but cannot turn a weak hit into a strong one.
    auto_lexical_min_coverage: float = 0.20
    auto_lexical_min_terms: int = 2
    auto_vector_threshold: float = 0.30
    lexical_min_terms: int = 1
    rrf_k: int = 60
    #: Digest of the space the query is embedded in; only vectors from that
    #: space are comparable to it.  The shipped space is the default, so a Core
    #: or an installation that names no embedding route keeps its behavior.
    embedding_space_id: str = SPACE_ID
    #: Names the store's owner is called by besides the one the host attests for the person asking
    #: (``core/owner_view.py``); none by default.
    owner_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.vector_threshold is None and self.embedding_space_id == SPACE_ID:
            object.__setattr__(self, "vector_threshold", BUILTIN_VECTOR_THRESHOLD)
        threshold = _finite_score(self.vector_threshold)
        if self.vector_threshold is not None and (threshold is None or not -1.0 <= threshold <= 1.0):
            raise ContractError("INPUT_INVALID", "vector_threshold")
        for name, low in (("auto_lexical_min_coverage", 0.0), ("auto_vector_threshold", -1.0), ("vector_margin", 0.0)):
            value = _finite_score(getattr(self, name))
            if value is None or not low <= value <= 1.0:
                raise ContractError("INPUT_INVALID", name)
        if type(self.auto_lexical_min_terms) is not int or self.auto_lexical_min_terms < 2:
            raise ContractError("INPUT_INVALID", "auto_lexical_min_terms")
        if type(self.vector_near_rank) is not int or self.vector_near_rank < 0:
            raise ContractError("INPUT_INVALID", "vector_near_rank")
        if type(self.lexical_min_terms) is not int or self.lexical_min_terms < 1:
            raise ContractError("INPUT_INVALID", "lexical_min_terms")
        if type(self.rrf_k) is not int or self.rrf_k < 1:
            raise ContractError("INPUT_INVALID", "rrf_k")
        if type(self.embedding_space_id) is not str or not self.embedding_space_id.strip():
            raise ContractError("INPUT_INVALID", "embedding_space_id")
        if (
            type(self.owner_names) is not tuple
            or len(self.owner_names) > OWNER_NAMES_LIMIT
            or any(
                type(name) is not str
                or name != name.strip()
                or not 2 <= len(name) <= OWNER_NAME_LENGTH
                or not name.isprintable()
                for name in self.owner_names
            )
        ):
            raise ContractError("INPUT_INVALID", "owner_names")

    def _vector_automatic_admission(
        self,
        candidate: CandidateRef,
        query: str,
        *,
        term_frequencies: dict[str, int] | None,
        corpus_size: int,
    ) -> bool:
        """Admit strong vector hits directly; require primary-anchor coherence in the large-corpus marginal band."""
        score = _finite_score(candidate.vector_score)
        if score is None or score + VECTOR_SCORE_TOLERANCE < self.auto_vector_threshold:
            return False
        margin = extreme_value_vector_margin(corpus_size)
        if term_frequencies is None or margin <= 0.0:
            return True
        base = self.vector_threshold if self.vector_threshold is not None else BUILTIN_VECTOR_THRESHOLD
        if score + VECTOR_SCORE_TOLERANCE >= base + margin:
            return True
        present, specific, anchors = query_term_information(
            meaningful_query_terms(query), query, term_frequencies, corpus_size
        )
        if not specific:
            asked = set(asked_terms(query))
            return candidate.kind in {"claim", "procedure"} and bool(asked - set(present))
        return bool(anchors.intersection(candidate.matched_query_terms or ()))

    def _idf_lexical_admission(
        self,
        scope_terms: set[str] | tuple[str, ...],
        matched: set[str],
        query: str,
        term_frequencies: dict[str, int],
        corpus_size: int,
        *,
        dated: bool = False,
    ) -> bool:
        """Evaluate self-information coverage and primary-anchor presence for one query or substantive clause."""
        present, specific, anchors = query_term_information(scope_terms, query, term_frequencies, corpus_size)
        if not present:
            return dated or corpus_size < MATURE_CORPUS_SIZE
        if corpus_size >= MATURE_CORPUS_SIZE and not dated:
            if not specific or not matched.intersection(anchors):
                return False
            floor = self.auto_lexical_min_coverage
        else:
            if specific and not matched.intersection(specific):
                return False
            floor = self.auto_lexical_min_coverage if specific else max(0.60, self.auto_lexical_min_coverage)
        present_info = sum(present.values())
        matched_info = sum(present[term] for term in matched.intersection(present))
        return present_info > 0 and matched_info / present_info >= floor

    def automatic_admission(
        self,
        candidate: CandidateRef,
        query: str,
        *,
        term_frequencies: dict[str, int] | None = None,
        corpus_size: int = 0,
    ) -> bool:
        """Whether a channel's evidence is strong enough for automatic injection.

        Named references and a named day's scoped conversation are already
        narrowed by the request. Relations inherit admission from their seed.
        One-term questions may match one term; longer ones need several. When
        corpus document frequencies are supplied, CJK boundary bigrams are
        de-artifacted by character-cover monotonicity, mature stores require
        primary-anchor coherence for lexical and marginal vector hits, and
        queries composed solely of high-frequency terms cannot admit noise.
        """
        if candidate.source in {"exact_ref", "scoped", "relation"}:
            return True
        if candidate.source == "vector":
            return self._vector_automatic_admission(
                candidate, query, term_frequencies=term_frequencies, corpus_size=corpus_size
            )
        terms = set(meaningful_query_terms(query))
        matched = terms.intersection(candidate.matched_query_terms or ())
        if not terms or len(matched) < min(len(terms), self.auto_lexical_min_terms):
            return False
        substantive_clauses = [
            clause
            for clause in _substantive_chinese_clauses(query)
            if (hits := len(matched.intersection(clause)))
            >= max(5, _specificity_required(len(clause)), min(len(clause), self.auto_lexical_min_terms))
            and hits / len(clause) >= self.auto_lexical_min_coverage
        ]
        if not substantive_clauses and len(matched) / len(terms) < self.auto_lexical_min_coverage:
            return False
        if term_frequencies is not None and corpus_size >= 64:
            is_dated = candidate.source == "dated"
            whole_ok = len(matched) / len(terms) >= self.auto_lexical_min_coverage and self._idf_lexical_admission(
                terms, matched, query, term_frequencies, corpus_size, dated=is_dated
            )
            return whole_ok or any(
                self._idf_lexical_admission(clause, matched, query, term_frequencies, corpus_size, dated=is_dated)
                for clause in substantive_clauses
            )
        return True

    def vector_best(self, candidates) -> float | None:
        """The best valid score among ``candidates`` in this policy's space; None when there is none."""
        scores = [
            score
            for candidate in candidates
            if candidate.source == "vector"
            and candidate.embedding_space == self.embedding_space_id
            and (score := _finite_score(candidate.vector_score)) is not None
            and -1.0 - VECTOR_SCORE_TOLERANCE <= score <= 1.0 + VECTOR_SCORE_TOLERANCE
        ]
        return max(scores, default=None)

    def vector_admission(
        self, candidate: CandidateRef, *, best: float | None = None, rank: int | None = None
    ) -> tuple[bool, str | None]:
        """Admit a vector hit that reaches the threshold, or that ranks ``rank`` within ``vector_near_rank`` and scores
        no more than ``vector_margin`` below the threshold when the best hit of its recall (``best``) reaches it.

        The threshold says whether the store holds anything this close to the question; rank keeps the hits that
        stand with the best rather than only the ones that clear a fixed line.  Without ``best`` and ``rank`` a hit
        is judged alone, against the threshold.
        """
        if candidate.source != "vector":
            return False, "not_vector"
        if type(candidate.vector_id) is not str or not candidate.vector_id:
            return False, "vector_id_missing"
        if candidate.embedding_space != self.embedding_space_id:
            return False, "embedding_space_mismatch"
        score = _finite_score(candidate.vector_score)
        if score is None:
            return False, "vector_score_invalid"
        # Cosine similarity is mathematically bounded.  Allow only the same
        # fixed float32/calibration tolerance used at the threshold boundary;
        # an arbitrary out-of-range score must never become high-confidence.
        if score > 1.0 + VECTOR_SCORE_TOLERANCE or score < -1.0 - VECTOR_SCORE_TOLERANCE:
            return False, "vector_score_invalid"
        if self.vector_threshold is None:
            return False, "vector_threshold_unconfigured"
        near = (
            best is not None
            and rank is not None
            and rank <= self.vector_near_rank
            and best + VECTOR_SCORE_TOLERANCE >= self.vector_threshold
            and score + VECTOR_SCORE_TOLERANCE >= self.vector_threshold - self.vector_margin
        )
        if near or score + VECTOR_SCORE_TOLERANCE >= self.vector_threshold:
            return True, None
        return False, "vector_below_threshold"

    def lexical_admission(self, candidate: CandidateRef, query: str, *, exact: bool = False) -> tuple[bool, str | None]:
        if candidate.source == "exact_ref" or exact:
            return True, None
        if candidate.source not in {"lexical", "recent_raw", "relation"}:
            return False, "not_lexical"
        score = _finite_score(candidate.lexical_score)
        if score is None:
            return False, "lexical_score_invalid"
        terms = [term for term in query_terms(query) if term not in _WEAK_QUERY]
        if not terms and candidate.source == "recent_raw":
            return False, "weak_query_only"
        if candidate.matched_query_terms is not None:
            # Only the query's own terms count.  Storage credits a synonym hit
            # to the one query term it stands in for (``synonym_expansions``);
            # a synonym reported as itself adds nothing.
            matched = set(meaningful_query_terms(query)).intersection(candidate.matched_query_terms)
            score = min(score, float(len(matched)))
        if score < self.lexical_min_terms:
            return False, "lexical_below_minimum"
        if not query_is_specific(
            query,
            matched_terms=score,
            matched_query_terms=candidate.matched_query_terms,
        ):
            return False, "lexical_insufficient_specificity"
        return True, None


def parse_time(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError("INPUT_INVALID", "timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ContractError("INPUT_INVALID", "timestamp")
    return stamp.astimezone(timezone.utc)


def in_time_window(value: str | None, context: SearchContext) -> bool:
    """Check source/object time against one fixed query clock."""

    if value is None:
        return context.mode not in {"current", "as_of"}
    stamp = parse_time(value)
    now = parse_time(context.now)
    if stamp > now:
        return False
    if context.as_of is not None and stamp > parse_time(context.as_of):
        return False
    return True


def query_is_relevant(query: str, content: str) -> bool:
    """Use meaningful lexical anchors for recent raw admission.

    Weak conversational words alone do not force an unrelated same-project
    source into the result; vector candidates remain independent of this gate.
    """

    q = set(meaningful_query_terms(query))
    if not q:
        return False
    overlap = len(q.intersection(lexical_terms(content)))
    if hard_identifiers(query).intersection(hard_identifiers(content)):
        return True
    return shares_enough(overlap, len(q))


def shares_enough(overlap: int, terms: int) -> bool:
    """Whether ``overlap`` of a query's ``terms`` meaningful terms is enough for a source something else already
    narrowed down, as the session for its recent messages or the time a question names: one of one, else two and
    three in ten."""
    if terms == 1:
        return overlap == 1
    return overlap >= max(2, math.ceil(terms * 0.30))


def _specificity_required(term_count: int) -> int:
    required = max(2, math.ceil(term_count * 0.30))
    return max(required, 3) if term_count >= 3 else required


def _substantive_chinese_clauses(query: str) -> tuple[tuple[str, ...], ...]:
    if _CJK_TEXT.search(query) is None:
        return ()
    clauses = tuple(
        meaningful_query_terms(clause)
        for clause in _CLAUSE_BOUNDARY.split(query)
        if clause.strip() and _CJK_TEXT.search(clause) is not None
    )
    substantive = tuple(terms for terms in clauses if len(terms) >= 5)
    return substantive if len(substantive) >= 2 else ()


def _matched_substantive_clauses(query: str, matches: Iterable[str]) -> tuple[tuple[int, int], ...]:
    """Matched and total term counts for clauses meeting the lexical specificity minimum."""
    held = frozenset(matches)
    return tuple(
        (hits, len(terms))
        for terms in _substantive_chinese_clauses(query)
        if (hits := len(held.intersection(terms))) >= max(5, _specificity_required(len(terms)))
    )


def query_is_specific(
    query: str,
    *,
    matched_terms: float,
    matched_query_terms: Iterable[str] | None = None,
) -> bool:
    """Require global, or conservative Chinese clause-level, lexical coverage."""

    terms = meaningful_query_terms(query)
    if not terms:
        return False
    if hard_identifiers(query):
        return True
    if len(terms) == 1:
        return matched_terms >= 1.0
    required = _specificity_required(len(terms))
    if matched_terms >= required and matched_terms / len(terms) >= 0.30:
        return True
    return any(hits / count >= 0.30 for hits, count in _matched_substantive_clauses(query, matched_query_terms or ()))


#: Characters that only ask: 什么, 怎么, 哪个, 谁, 多少 and the sentence-final
#: particles.  The index holds overlapping bigrams, so "这个实例当前是什么模型"
#: would search 是什, 什么 and 么模 too -- terms every earlier question shares
#: and no answer does -- and "你现在是什么模型呀" would come back as the best match.
#: A bigram holding one of these asks rather than names, so it proves nothing.
#: The few nouns built on them (酒吧, 哪吒) are the price of that rule.
_RETRIEVAL_ASKING_CHARACTERS = frozenset("什么吗呢呀吧嘛哪啥谁")
#: Asking words whose characters also appear in naming words (怎 in 怎样 only,
#: but 多 in 多少 and 多个, 是 in 是否 and 于是).
_RETRIEVAL_ASKING_WORDS = frozenset({"怎样", "如何", "为何", "多少", "是否", "何时", "几时"})
#: English asking and helper words: in a question they carry no subject matter.
_RETRIEVAL_ASKING_ENGLISH = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "the",
        "an",
        "of",
        "to",
        "in",
        "on",
        "for",
        "at",
        "by",
        "with",
        "about",
    }
)


def _asks_only(term: str) -> bool:
    if term in _RETRIEVAL_ASKING_WORDS or term in _RETRIEVAL_ASKING_ENGLISH:
        return True
    return _CJK_TEXT.search(term) is not None and any(char in _RETRIEVAL_ASKING_CHARACTERS for char in term)


#: English words that only frame a request to be told about what a question names ("Describe Orchard", "Any updates
#: on Orchard?"), which an answer need not say: it names what it answers.  Read by ``asked_terms`` alone: recall
#: searches them as it searches any word (``meaningful_query_terms``), since a message that says one may be what was
#: asked for ("show me the summary" finds "Summary: the deployment ended at noon.").  status names something itself
#: (a status page, a status code) and is none of them.
_FRAMING_ENGLISH = frozenset(
    {
        "please",
        "tell",
        "show",
        "give",
        "you",
        "know",
        "remember",
        "remind",
        "say",
        "said",
        "describe",
        "explain",
        "summarize",
        "summarise",
        "any",
        "anything",
        "something",
        "everything",
        "update",
        "updates",
        "latest",
        "news",
        "progress",
        "situation",
        "overview",
        "summary",
        "info",
        "information",
        "details",
    }
)
#: What ``asked_terms`` reads a question without, whole, so that the bigrams joining a Chinese one to its neighbours go
#: with it (是多 in 端口是多少, 绍一 and 一下 in 介绍一下): the Chinese words that ask
#: (``_RETRIEVAL_ASKING_WORDS``); the Chinese
#: words that only frame a request (介绍一下, 说说, 告诉, 关于, 知道, 怎么样, 进展, 消息 ...), where 情况 frames
#: "X 的情况" but not 情况下, a condition an answer says too; and the person a request to be told is addressed to:
#: tell me, 给我讲讲, and me or us closing a request ("Explain Orchard to me").  Elsewhere me and 我 name whom a
#: request is about (contact me, 给我订), and an answer says them.  Recall reads the question with all of them.
_FRAMING_PHRASES = re.compile(
    "|".join(
        (
            *sorted(_RETRIEVAL_ASKING_WORDS),
            "怎么样",
            "(?:给我|跟我|帮我)?(?:介绍|说说|讲讲|谈谈|聊聊|看看)",
            "告诉我?",
            "一下",
            "关于",
            "知道",
            "还记得",
            "记得",
            "情况(?!下)",
            "进展",
            "消息",
            "最新",
            "动态",
            r"\b(?:tell|show|give|remind) (?:me|us)\b",
            r"\b(?:let (?:me|us) know|fill (?:me|us) in|catch (?:me|us) up|walk (?:me|us) through)\b",
            r"\b(?:to|for) (?:me|us)\b(?=\W*$)",
        )
    ),
    re.I,
)


def asked_terms(question: str) -> tuple[str, ...]:
    """The terms a question asks an answer to say (``recall_needs.answering``): its meaningful terms
    (``meaningful_query_terms``) read without what asks or only frames a request (``_FRAMING_PHRASES``,
    ``_FRAMING_ENGLISH``).  "Tell me about Orchard" asks nothing past its name, as "What is Orchard?" does; "What do you
    know about Orchard's port?" asks for its port."""
    return tuple(
        term for term in meaningful_query_terms(_FRAMING_PHRASES.sub(" ", question)) if term not in _FRAMING_ENGLISH
    )


#: Longest text still read as a bare question rather than a message that also
#: says something.  Replies, reports and tool output run far longer.
QUESTION_MAX_CHARS = 120
#: A question either carries a question mark or ends on an asking particle or word.
_QUESTION_MARK = re.compile(r"[？?]")
_QUESTION_ENDING = re.compile(r"(?:吗|呢|么|呀|没|多少|是否|如何|哪|哪里|哪儿)$")
#: Marks a question may trail off with before the mark or particle that makes it one.
_TRAILING_MARKS = frozenset("。！!~～…,.，、)）]】\"'”’")
#: Marks that may close a message without changing what it says, question marks included (a question mark does change
#: it, unless both messages ask: ``same_message``).
_CLOSING_MARKS = _TRAILING_MARKS | frozenset("？?;；:：")


def _without_closing(text: str, marks: frozenset[str] = _CLOSING_MARKS) -> str:
    """``text`` without the spaces and ``marks`` it ends with, read from the end once.  A pattern anchored at the end
    tried a run of them again from each of its positions: seconds for one long message."""
    end = len(text)
    while end and (text[end - 1].isspace() or text[end - 1] in marks):
        end -= 1
    return text[:end]


def asks_without_answering(text: str) -> bool:
    """Whether a message only asks.

    Asked again, a question can return five earlier questions like it and not
    the reply that answered one: a question shares every content word with
    a later one and is short, so it outranks its own answer.  A short message
    with a question mark, or ending on an asking particle, is such a question.
    "去看下那台机器怎么了" is a request, and "不管什么情况都要先备份" a rule.
    """
    if type(text) is not str:
        return False
    body = _without_closing(unicodedata.normalize("NFKC", text).strip(), _TRAILING_MARKS)
    if not body or len(body) > QUESTION_MAX_CHARS:
        return False
    return _QUESTION_MARK.search(body) is not None or _QUESTION_ENDING.search(body) is not None


def _ends_asking(text: str) -> bool:
    """Whether a message ends asking: on a question mark, or on an asking particle before its closing marks.  Read at
    its end only, whatever its length: a long statement asked back with a question mark still asks."""
    body = _without_closing(text, _TRAILING_MARKS)
    return body.endswith(("?", "？")) or _QUESTION_ENDING.search(_without_closing(body)) is not None


def same_message(text: str, other: str) -> bool:
    """Whether two messages say the same words, whatever their closing marks and surrounding spaces, as long as both
    end asking or neither does.

    The automatic recall sets an older copy of the current message aside (``RetrievalPipeline._hydrate_admit``):
    the message already says it.  Compared character for character, "我家窗外有什么" and "我家窗外有什么？" (both ask)
    would be two messages, and the copy without the question mark would take a packet slot as if it answered the one
    with it.
    "我的航班改到周五早上八点了。" asked back as "……八点了？" is not a copy: the person's statement stays found, at
    any length.  Letter case still tells two messages apart ("Release-2" is not "release-2").  A
    message of closing marks alone is compared as it is.
    """
    if type(text) is not str or type(other) is not str:
        return False
    plain, other_plain = unicodedata.normalize("NFKC", text).strip(), unicodedata.normalize("NFKC", other).strip()
    words, other_words = _without_closing(plain), _without_closing(other_plain)
    if not words and not other_words:
        return plain == other_plain
    return words == other_words and _ends_asking(plain) == _ends_asking(other_plain)


def meaningful_query_terms(query: str) -> tuple[str, ...]:
    """Terms that can establish lexical relevance for one candidate."""

    return tuple(
        term for term in query_terms(query) if term not in _WEAK_QUERY and len(term) > 1 and not _asks_only(term)
    )


def cjk_run_bigrams(text: str, *, limit: int = 256) -> tuple[str, ...]:
    """Bounded overlapping 2-character bigrams across contiguous CJK runs in NFKC-normalized ``text``."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    seen: dict[str, None] = {}
    for run in CJK_RUN.findall(normalized):
        for index in range(len(run) - 1):
            seen.setdefault(run[index : index + 2], None)
            if len(seen) >= limit:
                return tuple(seen)
    return tuple(seen)


def word_document_frequencies(query: str, raw_frequencies: dict[str, int]) -> dict[str, int]:
    """Bound length-1 and length-2 interior CJK cross-word bridge valleys without segmentation dictionaries.

    Two adjacent words separated by zero characters form one bridge bigram ``b_i`` bounded by
    ``min(df(b_{i-1}), df(b_{i+1}))``; separated by a single particle or conjunction character they form a
    two-bigram valley ``(b_i, b_{i+1})`` bounded by ``min(df(b_{i-1}), df(b_{i+2}))``. A genuine two-character
    rare word spans three low-frequency bigrams, so its core bigram retains its rarity, as do three-character
    proper names and prefix/suffix compounds.
    """
    adjusted = dict(raw_frequencies)
    normalized = unicodedata.normalize("NFKC", query).casefold()
    for run in CJK_RUN.findall(normalized):
        if len(run) < 3:
            continue
        if len(run) == 3:
            b0, b1 = run[:2], run[1:3]
            v0, v1 = raw_frequencies.get(b0, 0), raw_frequencies.get(b1, 0)
            if v0 > 0 and v1 > 0:
                geo = math.isqrt(v0 * v1)
                adjusted[b0] = max(v0, v1 if v1 >= 16 * v0 else geo)
                adjusted[b1] = max(v1, geo)
            continue
        count = len(run) - 1
        vals = [raw_frequencies.get(run[index : index + 2], 0) for index in range(count)]
        bound = list(vals)
        if vals[1] >= 4 * max(vals[0], vals[2], 1):
            bound[0] = max(bound[0], vals[1])
        if vals[-2] >= 4 * max(vals[-1], vals[-3], 1):
            bound[-1] = max(bound[-1], vals[-2])
        for index in range(1, count - 1):
            bound[index] = max(bound[index], min(vals[index - 1], vals[index + 1]))
        for index in range(1, count - 2):
            pair_cap = min(vals[index - 1], vals[index + 2])
            bound[index] = max(bound[index], pair_cap)
            bound[index + 1] = max(bound[index + 1], pair_cap)
        for index in range(count):
            if vals[index] > 0:
                adjusted[run[index : index + 2]] = bound[index]
    return adjusted


def ubiquity_ceiling(corpus_size: int) -> int:
    """Sublinear Heaps'-law document-frequency ceiling separating ubiquitous vocabulary from store-specific terms."""
    if corpus_size <= 0:
        return 64
    return max(64, int(min(corpus_size * 0.10, 5.0 * math.sqrt(corpus_size))))


def _is_cjk_bigram(term: str) -> bool:
    return len(term) == 2 and CJK_RUN.fullmatch(term) is not None


def _independent_specific_sum(query: str, present: dict[str, float], specific: set[str]) -> float:
    """Sum specific self-information without double-counting adjacent overlapping bigrams in the same CJK run."""
    cjk_specific = {term for term in specific if _is_cjk_bigram(term)}
    total = sum(present[term] for term in specific - cjk_specific)
    if not cjk_specific:
        return total
    normalized = unicodedata.normalize("NFKC", query).casefold()
    accounted: set[str] = set()
    for run in CJK_RUN.findall(normalized):
        prev2, prev1 = 0.0, 0.0
        for index in range(len(run) - 1):
            bigram = run[index : index + 2]
            weight = present[bigram] if bigram in cjk_specific and bigram not in accounted else 0.0
            if weight > 0.0:
                accounted.add(bigram)
            prev2, prev1 = prev1, max(prev1, prev2 + weight)
        total += prev1
    total += sum(present[term] for term in cjk_specific - accounted)
    return total


def extreme_value_vector_margin(corpus_size: int) -> float:
    """Scale-aware cosine margin above the calibration threshold as corpus size grows beyond ``2048``."""
    if corpus_size <= VECTOR_CALIBRATION_CORPUS_SIZE:
        return 0.0
    return min(AUTO_STRONG_VECTOR_MARGIN, 0.012 * math.log(corpus_size / VECTOR_CALIBRATION_CORPUS_SIZE))


def query_term_information(
    scope_terms: Iterable[str],
    query: str,
    term_frequencies: dict[str, int],
    corpus_size: int,
) -> tuple[dict[str, float], set[str], set[str]]:
    """Return ``(present_weights, specific_terms, primary_anchors)`` over observed store terms."""
    adjusted = word_document_frequencies(query, term_frequencies)
    ceiling = ubiquity_ceiling(corpus_size)
    mature = corpus_size >= MATURE_CORPUS_SIZE
    smoothing = 1e-6 if mature else 1.0
    present: dict[str, float] = {}
    for term in set(scope_terms):
        df = adjusted.get(term, 0)
        if df > 0:
            present[term] = math.log((corpus_size + 1) / (df + 1)) + smoothing
    specific = {term for term in present if adjusted.get(term, 0) < ceiling}
    peak = max((present[term] for term in specific), default=0.0)
    intrinsic = min(AUTO_INTRINSIC_ANCHOR_INFO, 0.58 * math.log(max(2, corpus_size + 1)))
    anchor_floor = min(intrinsic, AUTO_ANCHOR_INFO_RATIO * peak)
    anchors = {term for term in specific if present[term] >= anchor_floor} if peak > 0.0 else set()
    return present, specific, anchors


def query_has_specific_anchor(query: str, term_frequencies: dict[str, int] | None, corpus_size: int) -> bool:
    """Whether ``query`` carries sufficient scale-normalized self-information to anchor cross-session raw events."""
    if term_frequencies is None or corpus_size < MATURE_CORPUS_SIZE:
        return True
    all_terms = meaningful_query_terms(query)
    if not all_terms:
        return any(char.isalpha() and not char.isascii() and CJK_RUN.fullmatch(char) is None for char in query)
    content_terms = asked_terms(query) or all_terms
    present, specific, _anchors = query_term_information(content_terms, query, term_frequencies, corpus_size)
    spec_sum = _independent_specific_sum(query, present, specific)
    spec_peak = max((present[term] for term in specific), default=0.0)
    all_present, _all_spec, _all_anc = query_term_information(all_terms, query, term_frequencies, corpus_size)
    total_info = sum(all_present.values())
    scale = math.log(corpus_size + 1)
    return (
        spec_sum >= min(AUTO_ONGOING_SPECIFIC_INFO_FLOOR, 0.89 * scale)
        or spec_peak >= min(AUTO_ONGOING_PEAK_INFO_FLOOR, 0.63 * scale)
        or total_info >= min(AUTO_ONGOING_TOTAL_INFO_FLOOR, 3.5 * scale)
    )


#: Chinese words that agent-operations conversations use interchangeably.
#:
#: The lexical index holds overlapping CJK bigrams, so "这个实例装上了吗" and
#: "这个实例已经安装好了" share no term for the one word they both use.  A query
#: naming a member also searches for the others (``synonym_expansions``); that
#: happens at query time only, so nothing is stored and nothing is re-indexed.
#:
#: Curated and reviewed, not generated.  A group holds words interchangeable in
#: this domain, and a word with a common second sense stays out even when one of
#: its senses fits: 还是 is "still" but also "or"; 开始 is to begin, not to boot
#: like 启动; 回退 is also a regression; 删掉 deletes but does not uninstall
#: like 卸载.  Every member is exactly two CJK characters, one index term, so a
#: swap replaces bigrams one for one (and 没成功 or 版本号 cannot join).
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("仍然", "依然", "仍旧", "依旧"),  # still
    ("安装", "装上", "装好"),  # install, installed
    ("卸载", "卸掉"),  # uninstall
    ("删除", "删掉"),  # delete
    ("完成", "做完", "搞定"),  # done
    ("修复", "修好"),  # fix, fixed
    ("报错", "出错"),  # an error was raised
    ("停止", "停掉"),  # stop
    ("登录", "登陆"),  # log in; 登陆 is its common misspelling
    ("账号", "帐号"),  # account (variant character)
    ("账户", "帐户"),  # account (variant character)
    ("密钥", "秘钥"),  # secret key; 秘钥 is its common misspelling
    ("默认", "缺省"),  # default
    ("截图", "截屏"),  # screenshot
)
_SYNONYMS = {member: tuple(other for other in group if other != member) for group in SYNONYM_GROUPS for member in group}
#: Bound on the synonym terms one query adds to the lexical search.
_SYNONYM_TERM_LIMIT = 64


def synonym_expansions(query: str) -> dict[str, str]:
    """Index terms a synonym of a query word adds, each mapped to the query term it stands in for.

    Writing another member in place of a group member changes three bigrams:
    the word's own and the two it forms with its neighbours.  Swapping 安装 for
    装上 in "这个实例装上了吗" turns 姬装, 装上 and 上了 into 姬安, 安装 and 装了.
    Each changed bigram is credited to the query term at its position, so a
    source phrased with the synonym matches at most the terms the question
    written with that synonym would have matched, and one synonym term never
    counts for more than one query term.  Coverage and specificity are still
    measured against the query's own terms, and a synonym is a CJK word, so it
    never names or satisfies a hard identifier.
    """
    terms = frozenset(meaningful_query_terms(query))
    expansions: dict[str, str] = {}
    for run in CJK_RUN.findall(unicodedata.normalize("NFKC", query).casefold()):
        for start in range(len(run) - 1):
            for other in _SYNONYMS.get(run[start : start + 2], ()):
                variant = run[:start] + other + run[start + 2 :]
                for position in range(max(start - 1, 0), min(start + 2, len(run) - 1)):
                    original, replacement = run[position : position + 2], variant[position : position + 2]
                    # A neighbouring member is an occurrence of its own and is
                    # expanded as one; blending it with this swap makes noise.
                    if position != start and original in _SYNONYMS:
                        continue
                    if original not in terms or replacement in terms or replacement in _WEAK_QUERY:
                        continue
                    expansions.setdefault(replacement, original)
                    if len(expansions) >= _SYNONYM_TERM_LIMIT:
                        return expansions
    return expansions


def hard_identifiers(text: str) -> frozenset[str]:
    # A release is also identified by its version suffix: "rc28" for 3.1.0rc28.
    matches = frozenset(match.group(0).casefold() for match in _HARD_IDENTIFIER.finditer(text))
    return matches | version_suffixes(text)


def _written_forms(text: str) -> frozenset[str]:
    """Hard identifiers with their hyphens and underscores dropped, so H-100 and H100 are one name."""
    return frozenset(identifier.replace("-", "").replace("_", "") for identifier in hard_identifiers(text))


def _spelled_out_forms(text: str) -> frozenset[str]:
    """The same names, also as they read with a space or dash between letters and their digits.

    A reply writes "DLSS 5" and "glm-5.3-flash" where the question wrote DLSS5
    and glm5.3-flash; compared as written, the reply would be refused as naming
    a different thing.  Only the text being offered is read both ways: read that
    way, a query asking for "the top 5 options" would start demanding that an
    answer contain "top5", and a query about H100 would ask for "H100as".
    """
    return _written_forms(text) | _written_forms(_IDENTIFIER_JOINT.sub("", text))


def identifiers_compatible(query: str, content: str, *, numbered_urls: bool = False) -> bool:
    """Keep distinct hard identifiers separate while supporting comparisons.

    A query naming H100 may only admit content that carries H100.  A query
    naming both H100 and H200 can therefore admit one side of the comparison
    at a time, with the evidence identity preserved for each side.
    """

    # Account handles and tracking parameters inside a record's URL do not
    # name additional objects. Identifiers in the surrounding question still do.
    if numbered_urls and "://" in query:
        query = _HTTP_LINK.sub(lambda match: " " if _numbered_link(match[0]) else match[0], query)
    requested = _written_forms(query)
    if not requested:
        return True
    return bool(requested.intersection(_spelled_out_forms(content)))


def _numbered_link(link: str) -> tuple[str, str, str] | None:
    """Read a record's identity from its permalink, including views of the same record."""
    try:
        url = urlsplit(link.partition("?")[0].partition("#")[0].rstrip(".;:]!"))
    except ValueError:
        return None
    site = url.netloc.rpartition("@")[2].casefold()
    if site in _POST_SITES and (post := _POST_PATH.fullmatch(url.path.casefold())):
        return "x.com", "/status", post["number"]
    path = url.path.casefold() if site in {"github.com", "www.github.com"} else url.path
    if site and (record := _RECORD_PATH.fullmatch(path)):
        return site, record["parent"], record["number"]
    return None


def numbered_links(text: str) -> set[tuple[str, str, str]]:
    """Return canonical (site, collection path, record number) identities for absolute links."""
    return {record for match in _HTTP_LINK.finditer(text) if (record := _numbered_link(match[0])) is not None}


def numbered_link_conflict(requested: set[tuple[str, str, str]], content: str) -> bool:
    """Veto an automatic match naming a different record in the same collection.

    Replies may omit a link or cite another site. Only an explicit competing
    identity is negative evidence. A comparison may find either requested record.
    """
    if not requested or "://" not in content:
        return False
    families: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for site, parent, number in requested:
        families.setdefault((site, parent.casefold()), set()).add((parent, number))
    conflict = False
    for site, parent, number in numbered_links(content):
        targets = families.get((site, parent.casefold()), ())
        if any(other_number == number for _, other_number in targets):
            return False
        # GitHub normalizes repository names. On a generic site, only the
        # same spelling proves that different numbers compete in one collection.
        if any(other_parent == parent for other_parent, _ in targets):
            conflict = True
    return conflict


def applicability(context: SearchContext, project_id: str | None, branch_id: str | None) -> str:
    parts = []
    if project_id is not None:
        parts.append(f"project:{project_id}")
    if branch_id is not None:
        parts.append(f"branch:{branch_id}")
    return ",".join(parts) if parts else "trusted scope"


def rrf_score(ranks: Iterable[int], *, k: int = 60) -> float:
    return sum(1.0 / (k + rank) for rank in ranks)
