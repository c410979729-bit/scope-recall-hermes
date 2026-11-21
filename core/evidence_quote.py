r"""Resolve a model's evidence quote back to a literal slice of stored content.

The model never sees stored bytes.  ``consolidation_messages`` hands it
``json.dumps(body, ensure_ascii=False)``, so a source holding a double quote
or a newline reaches the model as ``\"`` and ``\n`` -- two characters, not
one.  A quote copied faithfully from what the model saw is therefore not
always a substring of what is stored, and the byte-strict ``in`` checks in
``core/claim_storage.py``, ``core/consolidate.py`` and the alias path in
``core/mutate.py`` reject it as ``evidence_span`` / ``fragment_evidence``.

Model evidence uses an exact occurrence or an unambiguous short quote anchor.
Canonical forms locate the raw slice and never become the stored evidence.
"""

from __future__ import annotations

import json
import unicodedata
from functools import cached_property

from ..contracts import ContractError, nfkc_letters_and_digits

#: What ``json.dumps(..., ensure_ascii=False)`` emits for each character it
#: rewrites.  Anything else below 0x20 becomes a ``\u00xx`` escape; everything
#: at or above it, non-ASCII included, is passed through unchanged.
_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _encoded_piece(char: str) -> str:
    piece = _ESCAPES.get(char)
    if piece is not None:
        return piece
    if char < " ":
        return json.dumps(char)[1:-1]
    return char


def encoded_with_origin(content: str) -> tuple[str, list[int]]:
    """Return the JSON-encoded body and, per encoded character, its raw index.

    The encoded text is what the model was shown.  ``origin`` is what lets a
    hit in that text be mapped back to a real slice of ``content`` rather than
    be trusted on its own.
    """
    pieces: list[str] = []
    origin: list[int] = []
    for index, char in enumerate(content):
        piece = _encoded_piece(char)
        pieces.append(piece)
        origin.extend([index] * len(piece))
    return "".join(pieces), origin


def _json_decoded(quote: str) -> str | None:
    """Read the quote as a JSON string body, or ``None`` if it is not one.

    A quote sliced out of the encoded text can end mid-escape; this then fails
    rather than guessing at the missing half.
    """
    try:
        value = json.loads('"' + quote + '"')
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None


def resolve_quote(quote: object, content: object) -> str | None:
    """Return the stored slice this quote names, or ``None`` if there is none.

    A returned value is always a literal substring of ``content``.
    """
    if type(quote) is not str or not quote or type(content) is not str or not content:
        return None
    if quote in content:
        return quote
    decoded = _json_decoded(quote)
    if decoded and decoded in content:
        return decoded
    encoded, origin = encoded_with_origin(content)
    position = encoded.find(quote)
    if position < 0:
        return None
    # A match that begins or ends inside an escape sequence maps back to the
    # whole character it belongs to, so the slice can be one character wider at
    # each end than what the model quoted.  It is still a literal slice of the
    # stored source, and a wider quote only ever makes the downstream
    # "value_text must appear inside the quote" check more permissive -- it
    # cannot make an unsupported value look supported by text that is not
    # there.
    start = origin[position]
    end = origin[position + len(quote) - 1] + 1
    return content[start:end]


def resolve_evidence_quotes(value: dict, contents: dict) -> int:
    """Rewrite every resolvable span quote in place; return how many moved.

    ``contents`` maps ``(source_ref, source_revision)`` to stored content and
    is built by the caller from sources it has *already* authorized.  Taking a
    map rather than a transaction is deliberate: a span naming a source outside
    that map is left exactly as it is, so this can never become a way to read a
    source the caller had not already cleared.  Rejecting such a span stays
    with the strict check downstream, which is the place that states it.
    """
    resolved = 0
    for proposal in value.get("claim_proposals") or ():
        if type(proposal) is not dict:
            continue
        for span in proposal.get("evidence_spans") or ():
            if type(span) is not dict:
                continue
            content = contents.get((span.get("source_ref"), span.get("source_revision")))
            if content is None:
                continue
            if "start" in span or "end" in span:
                continue
            stored = resolve_quote(span.get("quote"), content)
            if stored is not None and stored != span.get("quote"):
                span["quote"] = stored
                resolved += 1
    return resolved


_PUNCTUATION = str.maketrans({"。": ".", "‘": "'", "’": "'", "“": '"', "”": '"'})


def _normalized_origin(content: str) -> tuple[str, list[tuple[int, int]]]:
    """Map typographic equivalents back to complete raw character intervals."""
    characters: list[str] = []
    origins: list[tuple[int, int]] = []
    index = 0
    while index < len(content):
        end = index + 1
        while end < len(content) and unicodedata.category(content[end]).startswith("M"):
            end += 1
            if end - index > 64:
                # Pathological combining runs cannot be short quote anchors.
                return "", []
        # Adjacent Hangul jamo compose without a combining-mark category.
        while end < len(content) and unicodedata.normalize("NFKC", content[index:end] + content[end]) != (
            unicodedata.normalize("NFKC", content[index:end]) + unicodedata.normalize("NFKC", content[end])
        ):
            end += 1
        for char in unicodedata.normalize("NFKC", content[index:end]).translate(_PUNCTUATION):
            if char.isspace():
                char = " "
                if characters and characters[-1] == " ":
                    origins[-1] = (origins[-1][0], end)
                    continue
            # A space before a sentence terminator is typographic. An
            # internal dot in a version, domain or decimal keeps its spacing.
            if (
                char in ".!?"
                and characters
                and characters[-1] == " "
                and (content[index] == "。" or end == len(content) or content[end].isspace())
            ):
                characters.pop()
                origins.pop()
            characters.append(char)
            origins.append((index, end))
        index = end
    return "".join(characters), origins


class QuoteAnchors:
    """Resolve unique short anchors within one authorized source or source window."""

    def __init__(self, content: str) -> None:
        self.content = content

    @cached_property
    def _normalized(self) -> tuple[str, list[tuple[int, int]]]:
        return _normalized_origin(self.content)

    def locate(self, quote: str) -> tuple[int, int] | None:
        """Prefer literal text, then complete JSON escapes, then typography."""
        if not quote or len(quote) > 4096:
            return None
        decoded = _json_decoded(quote) if "\\" in quote else None
        for literal in (quote, decoded):
            if not literal:
                continue
            start = self.content.find(literal)
            if start >= 0:
                return (start, start + len(literal)) if self.content.find(literal, start + 1) < 0 else None
        return self._locate_normalized(decoded or quote)

    def _locate_normalized(self, quote: str) -> tuple[int, int] | None:
        needle, _ = _normalized_origin(quote)
        needle = needle.strip()
        if not needle:
            return None
        normalized, origins = self._normalized
        match = None
        start = normalized.find(needle)
        # Bound the work on repeated compatibility expansions that cannot
        # themselves supply an interval. A second valid hit is ambiguous.
        for _ in range(8):
            if start < 0:
                return match
            end = start + len(needle)
            if (start == 0 or origins[start - 1] != origins[start]) and (
                end == len(origins) or origins[end - 1] != origins[end]
            ):
                interval = origins[start][0], origins[end - 1][1]
                if match is not None or interval[1] - interval[0] > 4096:
                    return None
                match = interval
            start = normalized.find(needle, start + 1)
        return None


def quote_interval(quote: str, content: str) -> tuple[int, int] | None:
    """Return the exact Unicode interval selected by one unambiguous anchor."""
    return QuoteAnchors(content).locate(quote)


def align_claim_evidence(claim: dict, sources: dict, *, index: int, anchors: dict | None = None) -> None:
    """Bind model quotes to authorized source versions; offsets are optional hints.

    A window's content is searched before any omitted text and intervals are
    translated to the complete source. No source outside this map is read.
    A quote without a letter or digit (``{``, ``。``) can state no subject,
    relation or value, and is refused like a quote the source does not hold.
    """
    spans = claim.get("evidence_spans")
    if not isinstance(spans, list):
        return
    for span_index, span in enumerate(spans):
        if not isinstance(span, dict) or not isinstance(span.get("quote"), str):
            continue
        source = sources.get((span.get("source_ref"), span.get("source_revision")))
        if source is None:
            field = "source_revision" if any(key[0] == span.get("source_ref") for key in sources) else "source_ref"
            raise ContractError("DERIVATION_INVALID", f"claim_proposals[{index}].evidence_spans[{span_index}].{field}")
        if not nfkc_letters_and_digits(span["quote"]):
            raise ContractError("DERIVATION_INVALID", f"claim_proposals[{index}].evidence_spans[{span_index}].quote")
        content = source.event["content"]
        window = getattr(source, "consolidation_window", None)
        offset = window.start if window is not None else 0
        quote, start, end = span["quote"], span.get("start"), span.get("end")
        if (
            type(start) is int
            and type(end) is int
            and offset <= start < end <= offset + len(content)
            and content[start - offset : end - offset] == quote
        ):
            continue
        if anchors is None:
            interval = quote_interval(quote, content)
        else:
            interval = anchors[(source.ref, source.revision)].locate(quote)
        if interval is None:
            raise ContractError("DERIVATION_INVALID", f"claim_proposals[{index}].evidence_spans[{span_index}].quote")
        start, end = interval
        span.update(quote=content[start:end], start=start + offset, end=end + offset)


def _located_evidence(proposal, contents):
    spans = []
    for span in proposal["evidence_spans"]:
        located = dict(span)
        content = contents.get((span["source_ref"], span["source_revision"]))
        if "start" not in span and "end" not in span and content is not None:
            quote = resolve_quote(span["quote"], content)
            if quote is not None:
                start = content.index(quote)
                located.update(start=start, end=start + len(quote))
        spans.append(located)
    return spans


def same_evidence(left, right, contents) -> bool:
    """Adding deterministic positions to a legacy quote is not new evidence.

    Source versions, literal quotes, and verified occurrence choices must still
    match. A different occurrence or an independent source remains new evidence.
    """
    return _located_evidence(left, contents) == _located_evidence(right, contents)


def resolve_fragment_summaries(value: dict, page: str, seeds: tuple) -> None:
    """Restore summary strings to literal page slices before strict validation.

    Only a goal already accepted on an earlier page can use that page's text.
    Other summaries and references resolve on the current page only; unresolved
    text stays unchanged so the existing validator rejects it normally.
    """
    for proposal in value["resume_proposals"]:
        goal = proposal["goal"]
        if goal not in seeds:
            original = goal["text"]
            stored = resolve_quote(original, page)
            if stored is not None:
                # A current-page literal must not be decoded again against an
                # older goal: different page goals must stay distinguishable.
                goal["text"] = stored
            else:
                for seed in seeds:
                    if dict(goal, text=seed["text"]) == seed and resolve_quote(original, seed["text"]) == seed["text"]:
                        goal["text"] = seed["text"]
                        break
        for field in ("decisions", "verified_progress", "open_items", "blockers"):
            for item in proposal[field]:
                item["text"] = resolve_quote(item["text"], page) or item["text"]
        if proposal["next_step"]:
            proposal["next_step"] = resolve_quote(proposal["next_step"], page) or proposal["next_step"]
    for proposal in value["reference_proposals"]:
        proposal["mention"] = resolve_quote(proposal["mention"], page) or proposal["mention"]


__all__ = ["encoded_with_origin", "resolve_quote", "resolve_evidence_quotes", "resolve_fragment_summaries"]
