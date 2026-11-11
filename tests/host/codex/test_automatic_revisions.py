"""Synthetic user prompts through the client capture boundary and automatic consolidation."""

import json
from dataclasses import replace

import pytest
from scope_recall.adapters.clients import CodexHookHandler, install_codex_scope_recall
from scope_recall.adapters.hermes.shared_entries import attach_shared_record, client_entry_record
from scope_recall.contracts import TrustedSourcePrincipal
from scope_recall.core.worker_consolidation import build_consolidation_model

from tests.host.codex.test_shared_client import NOW, store


class LaunchPort:
    """Extract only the synthetic launch messages and judge the preceding value by content."""

    def propose(self, messages, **kwargs):
        body = json.loads(messages[-1]["content"])
        if "proposal" in body:
            targets = [c for c in body["candidates"] if c["payload"]["value_text"] == "March 3"]
            assert len(targets) == 1
            return json.dumps(
                dict(
                    support="supported",
                    action="update",
                    target_ref=targets[0]["ref"],
                    target_revision=targets[0]["revision"],
                )
            )
        claims = []
        for source in body["sources"]:
            text = source["content"]
            value = "March 17" if "March 17" in text else "March 3"
            claims.append(
                dict(
                    kind="decision",
                    subject="TEST-project",
                    predicate="launch date",
                    value_text=value,
                    statement_kind="decision",
                    conditions=[],
                    valid_from=None,
                    valid_to=None,
                    evidence_spans=[
                        dict(
                            source_ref=source["source_ref"],
                            source_revision=source["source_revision"],
                            quote=text,
                            start=0,
                            end=len(text),
                        )
                    ],
                )
            )
        return json.dumps(
            dict(
                protocol_version="1.1",
                source_refs=body["source_refs"],
                claim_proposals=claims,
                resume_proposals=[],
                reference_proposals=[],
            )
        )


@pytest.mark.parametrize("entry", ["independent", "shared", "different_people"])
def test_client_prompts_automatically_revise_only_when_speakers_allow_it(tmp_path, store, monkeypatch, entry):
    root, _homes, _client, scope = store
    project = tmp_path / "TEST-project"
    project.mkdir()
    if entry == "independent":
        config, core = install_codex_scope_recall(tmp_path / "TEST-independent", project_root=project)
        hook = CodexHookHandler(config, core=core)
    else:
        home = tmp_path / "TEST-codex-client"
        attach_shared_record(
            root,
            client_entry_record(
                host="codex",
                home=home,
                entry_id="test-codex",
                display_name="TEST client",
                attached_at=NOW,
                allowed_scope_ids=[scope],
                writable_scope_ids=[scope],
                capture_scope_id=scope,
            ),
            now=NOW,
        )
        hook = CodexHookHandler.from_home(str(home), "codex")
    # Drain synchronously with a fixed port; no resident process or external model participates.
    monkeypatch.setattr(hook, "ensure_runtime", lambda *a, **k: None)
    model = build_consolidation_model(LaunchPort())
    original_context = hook.context
    try:
        for index, value in enumerate(("March 3", "March 17")):
            if entry == "different_people" and index == 1:
                # Simulate a second authenticated sender at the host's trusted context boundary.
                monkeypatch.setattr(
                    hook,
                    "context",
                    lambda *a: replace(
                        original_context(*a),
                        source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-second-speaker"),
                    ),
                )
            text = f"TEST-project launch date {value}."
            payload = dict(
                hook_event_name="UserPromptSubmit",
                session_id="TEST-auto-session",
                cwd=str(project),
                turn_id=f"TEST-turn-{index}",
                prompt_id=f"TEST-prompt-{index}",
                prompt=text,
            )
            hook.handle_payload(payload)
            context = hook.context(hook.audience_of(payload), payload["session_id"], "human_direct")
            with hook.core.storage.read(context) as tx:
                row = (
                    tx._check()
                    .execute(
                        "SELECT event_id,source_revision FROM source_events WHERE session_id=? AND content=?",
                        (context.session_id, text),
                    )
                    .fetchone()
                )
            assert row is not None
            source = hook.core.source(context, row[0], row[1])
            assert source.event["origin"] == "human_direct"
            assert source.event["source_principal"]["resolution"] == (
                "unresolved" if entry == "independent" else "verified"
            )
            result = hook.core.drain_worker(context, max_items=32, remaining_seconds=30, consolidation=model)
            assert result.failed == 0 and not any(i.error_code for i in result.items)
            with hook.core.storage.read(context) as tx:
                heads = tx._check().execute("SELECT claim_id FROM claims WHERE kind='decision'").fetchall()
            assert len(heads) == 1
            head = hook.core.current_claim(context, heads[0][0])
            expected = "March 3" if index == 0 or entry == "different_people" else "March 17"
            assert head.payload["value_text"] == expected
        history = hook.core.claim_history(context, heads[0][0])
        assert len(history) == 2
        assert history[-1].state == ("proposed" if entry == "different_people" else "active")
    finally:
        hook.close()
