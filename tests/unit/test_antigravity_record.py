"""Synthetic native steps preserve occurrences, not generated or partial text."""

import json
from uuid import uuid4

import pytest
from scope_recall.adapters.clients import transcript


def step(index=0, **values):
    """A complete synthetic user step, with explicit overrides for refusal cases."""
    return (
        dict(
            step_index=index,
            source="USER_EXPLICIT",
            type="USER_INPUT",
            status="DONE",
            content="TEST atlas code is bluebird",
            created_at="2026-11-17T08:00:00Z",
        )
        | values
    )


def test_native_occurrence_separates_identical_inputs():
    first = transcript.antigravity_said(step())
    second = transcript.antigravity_said(step(2))
    assert first and second and first.text == second.text
    assert first.entry_id != second.entry_id
    assert first.prompt_id == first.entry_id


@pytest.mark.parametrize(
    "changes",
    [
        {"source": "MODEL"},
        {"type": "GENERIC"},
        {"status": "RUNNING"},
        {"truncated_fields": ["content"]},
        {"content": ""},
        {"content": {"text": "TEST"}},
        {"step_index": True},
        {"created_at": "not-a-time"},
        {"type": []},
        {"source": {}},
    ],
)
def test_nonhuman_or_incomplete_steps_are_not_user_words(changes):
    assert transcript.antigravity_said(step(**changes)) is None


def test_visible_planner_text_excludes_reasoning_and_tools():
    entry = transcript.antigravity_said(
        step(source="MODEL", type="PLANNER_RESPONSE", thinking="TEST hidden", tool_calls=[{"args": "TEST tool"}])
    )
    assert entry and entry.role == "assistant" and entry.text == "TEST atlas code is bluebird"
    assert transcript.antigravity_said(step(source="MODEL", type="PLANNER_RESPONSE", content=None)) is None


def test_prefix_identity_survives_bad_and_incomplete_tail(tmp_path):
    path = tmp_path / "transcript.jsonl"
    prefix = json.dumps(step()).encode() + b"\n"
    path.write_bytes(prefix + b'\nnot-json\n{"step_index":')
    entries = transcript.read(path, 0, rows=transcript.antigravity_said)
    assert entries[0][1] == transcript.antigravity_said(step())
    assert entries[-1][0] < path.stat().st_size


@pytest.mark.parametrize("filename", ["transcript.jsonl", "transcript_full.jsonl"])
def test_native_path_requires_matching_conversation(tmp_path, filename):
    session = str(uuid4())
    path = tmp_path / "brain" / session / ".system_generated" / "logs" / filename
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(step()) + "\n", encoding="utf-8")
    assert transcript.antigravity_record_path(str(path), session) == path
    assert transcript.antigravity_record_path(str(path), str(uuid4())) is None


def test_native_user_envelope_keeps_only_the_exact_user_words():
    text = "  TEST user text\nsecond line  "
    wrapped = f"<USER_REQUEST>\n{text}\n</USER_REQUEST>\n<ADDITIONAL_METADATA>TEST generated context"
    entry = transcript.antigravity_said(step(content=wrapped))
    assert entry and entry.text == text
    assert transcript.antigravity_said(step(content=wrapped.replace("</USER_REQUEST>", ""))) is None
    assert transcript.antigravity_said(step(content=wrapped + wrapped)) is None


def test_running_visible_step_defers_byte_acknowledgment(tmp_path):
    path = tmp_path / "transcript_full.jsonl"
    first = json.dumps(step()).encode() + b"\n"
    next_step = dict(type="PLANNER_RESPONSE", source="MODEL", content="TEST final response")
    path.write_bytes(first + json.dumps(step(1, **next_step, status="RUNNING")).encode() + b"\n")
    page = transcript.read(path, 0, rows=transcript.antigravity_said)
    assert len(page) == 1 and page[-1][0] == len(first)
    path.write_bytes(first + json.dumps(step(1, **next_step)).encode() + b"\n")
    page = transcript.read(path, len(first), rows=transcript.antigravity_said)
    assert page[-1][1] == transcript.antigravity_said(step(1, **next_step))
