"""Pure, deterministic identity helpers for structured factual assertions.

A fact key identifies a subject/predicate slot. The value fingerprint identifies
one value asserted for that slot. This separation lets temporal evolution close
or supersede one value without losing the stable slot identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

FACT_IDENTITY_VERSION = 1
MAX_FACT_SUBJECT_CHARS = 200
MAX_FACT_PREDICATE_CHARS = 120
MAX_FACT_VALUE_CHARS = 2000
_WHITESPACE_RE = re.compile(r"\s+")


class FactIdentityError(ValueError):
    """Raised when a structured fact cannot form a safe, stable identity."""


def normalize_fact_component(
    value: Any,
    *,
    field_name: str = "component",
    max_chars: int = MAX_FACT_VALUE_CHARS,
    allow_empty: bool = False,
) -> str:
    """Normalize one identity component with NFKC, whitespace folding, and casefold."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip().casefold()
    if not normalized and not allow_empty:
        raise FactIdentityError(f"{field_name} is required")
    if len(normalized) > max_chars:
        raise FactIdentityError(f"{field_name} exceeds {max_chars} characters")
    return normalized


_WORD_RE = re.compile(r"[^\W_]+")


def _words(label: Any) -> list[str]:
    """A label's runs of letters and digits, case and width folded."""
    return _WORD_RE.findall(unicodedata.normalize("NFKC", str(label or "")).casefold())


def fact_frame(subject: Any, predicate: Any) -> str:
    """What a subject and its relation say together: their letters and digits in order, case and width folded.

    An extraction model writes these labels afresh for every message, so one slot comes back as "launch date" and
    "launch_date", divided as ("Atlas beta", "launch date") and ("Atlas", "beta launch date"), or with the words that
    open the relation also closing the subject, ("Atlas beta launch", "launch date"); those words are read once.
    Equal frames only nominate claims for a same-slot judgement (``claims.slot_reading``).  A frame is never a stored
    identity: ``claims.claim_slot`` keeps the labels as written.

    Labels undergo Unicode NFKC then casefold; runs of letters and digits are concatenated. Non-segmented scripts
    are not tokenized by language: overlap removal compares complete runs, never substrings within a run.
    Empty frames carry no candidate identity and must be excluded from loose matching.
    """
    named, relation = _words(subject), _words(predicate)
    # The slot lookup runs this on every claim of a scope, and few labels repeat a word across the two.
    if named and named[-1] in relation:
        for size in range(min(len(named), len(relation)), 0, -1):
            if named[-size:] == relation[:size]:
                relation = relation[size:]
                break
    return "".join(named + relation)


def _digest(kind: str, components: list[str]) -> str:
    material = json.dumps(
        {
            "components": components,
            "kind": kind,
            "version": FACT_IDENTITY_VERSION,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _subject(subject: Any) -> str:
    return normalize_fact_component(subject, field_name="subject", max_chars=MAX_FACT_SUBJECT_CHARS)


def _predicate(predicate: Any) -> str:
    return normalize_fact_component(predicate, field_name="predicate", max_chars=MAX_FACT_PREDICATE_CHARS)


def _value(value: Any) -> str:
    return normalize_fact_component(value, field_name="value", max_chars=MAX_FACT_VALUE_CHARS)


def canonical_fact_key(subject: Any, predicate: Any) -> str:
    """Return the versioned key for one subject/predicate fact slot."""

    return f"fact:v{FACT_IDENTITY_VERSION}:{_digest('slot', [_subject(subject), _predicate(predicate)])}"


def canonical_fact_fingerprint(subject: Any, predicate: Any, value: Any) -> str:
    """Return a versioned fingerprint for one normalized slot/value assertion."""

    components = [_subject(subject), _predicate(predicate), _value(value)]
    return f"assertion:v{FACT_IDENTITY_VERSION}:{_digest('assertion', components)}"


@dataclass(frozen=True, slots=True)
class FactIdentity:
    """Normalized identity of one fact slot and asserted value."""

    subject: str
    predicate: str
    value: str
    fact_key: str
    value_fingerprint: str

    @classmethod
    def from_parts(cls, subject: Any, predicate: Any, value: Any) -> "FactIdentity":
        subject, predicate, value = _subject(subject), _predicate(predicate), _value(value)
        return cls(
            subject=subject,
            predicate=predicate,
            value=value,
            fact_key=canonical_fact_key(subject, predicate),
            value_fingerprint=canonical_fact_fingerprint(subject, predicate, value),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "fact_key": self.fact_key,
            "value_fingerprint": self.value_fingerprint,
        }


def build_fact_identity(subject: Any, predicate: Any, value: Any) -> FactIdentity:
    """Build a validated normalized identity for one factual assertion."""

    return FactIdentity.from_parts(subject, predicate, value)
