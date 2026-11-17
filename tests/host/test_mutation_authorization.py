"""Synthetic tool-boundary regressions; metadata here proves no native host capability."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from scope_recall.adapters.clients import install_codex_scope_recall
from scope_recall.adapters.clients.identity import trusted_context
from scope_recall.adapters.clients.mcp_server import build_server
from scope_recall.adapters.hermes import ScopeRecallHermesAdapter, install_hermes_scope_recall
from scope_recall.contracts import ContractError
from scope_recall.core.capture_inbox import enqueue, replay_inbox
from scope_recall.core.claim_storage import Claims
from scope_recall.core.delete_storage import Deletions

from tests.v11_support import source_event


@pytest.fixture(params=["hermes", "mcp"])
def surface(request, tmp_path):
    """Create a local synthetic installation and invoke its normal tool dispatch."""
    session = str(uuid4())
    if request.param == "hermes":
        home = tmp_path / "hermes"
        home.mkdir()
        settings = dict(platform="cli", user_id="TEST-user", agent_workspace="TEST-workspace")
        _, core = install_hermes_scope_recall(home, agent_id="TEST-agent", **settings)
        providers = {}

        def open_session(thread):
            provider = ScopeRecallHermesAdapter(core=core)
            provider.initialize(thread, hermes_home=str(home), agent_identity="TEST-agent", **settings)
            providers[thread] = provider
            return provider

        adapter = open_session(session)
        context = adapter._identity.trusted_context()
        scope = adapter._identity.local_scope_id

        def call(name, payload, thread=session):
            return json.loads(providers[thread].handle_tool_call(name, payload))

        def reopen():
            providers[session].shutdown()
            return open_session(session)

    else:
        project = tmp_path / "project"
        project.mkdir()
        config, core = install_codex_scope_recall(tmp_path / "client", project_root=project)
        adapter = build_server(config, workspace=project, core=core)
        context = trusted_context(config, adapter.audience, session_id=session, actor_origin="human_direct")
        scope = adapter.audience.capture_scope_id

        def call(name, payload, thread=session):
            # A same-turn append shares this turnId. It cannot attest a message occurrence.
            ctx = SimpleNamespace(request_context=SimpleNamespace(meta={"threadId": thread, "turnId": "turn-1"}))
            try:
                return {"result": getattr(adapter, name)(ctx, **payload)}
            except ContractError as exc:
                return {"error": {"code": exc.code, "field": exc.field}}

        def open_session(thread):
            return replace(context, session_id=thread)

        def reopen():
            nonlocal adapter
            adapter = build_server(config, workspace=project, core=core)
            return adapter

    def capture(key, text, actor=context):
        event = source_event(source_event_key=key, content=text, occurred_at=None, time_precision="unknown")
        return core.record_event(actor, event, scope_id=scope, remaining_seconds=5).event_refs[0]

    yield SimpleNamespace(
        core=core,
        context=context,
        scope=scope,
        capture=capture,
        call=call,
        adapter=adapter,
        open_session=open_session,
        reopen=reopen,
    )
    if request.param == "hermes":
        for provider in providers.values():
            provider.shutdown()


def mutation(surface, operation):
    """Persist an explicit command without applying its requested revision during capture."""
    source = surface.capture("TEST-fact", "TEST-project 配色 蓝色。")
    proposal = dict(
        kind="fact",
        subject="TEST-project",
        predicate="配色",
        value_text="蓝色",
        conditions=[],
        statement_kind="assertion",
        valid_from=None,
        valid_to=None,
        evidence_spans=[dict(source_ref=source.ref, source_revision=1, quote="TEST-project 配色 蓝色。")],
    )
    claim = surface.core.records.accept_claim_proposals(
        surface.context,
        dict(
            protocol_version="1.1",
            source_refs=[f"{source.ref}@1"],
            claim_proposals=[proposal],
            resume_proposals=[],
            reference_proposals=[],
        ),
        scope_id=surface.scope,
        remaining_seconds=5,
    ).items[0]
    if operation == "forget":
        surface.capture("TEST-command", f"删除 {claim.ref}。")
        body = dict(protocol_version="1.1", target_refs=[claim.ref], mode="delete", expected_revisions={claim.ref: 1})
    else:
        command = surface.capture("TEST-command", "Please correct TEST-project 配色: 银色。")
        body = dict(
            protocol_version="1.1",
            target_ref=claim.ref,
            expected_revision=1,
            new_value="银色",
            conditions=[],
            source_evidence_refs=[f"{command.ref}@1"],
            valid_from=None,
        )
    assert surface.core.current_claim(surface.context, claim.ref).revision == 1
    return claim, body


@pytest.mark.parametrize("operation", ["revise", "forget"])
@pytest.mark.parametrize("delivery", ["next_turn_failed", "same_turn_queued", "restored_history"])
def test_stored_command_cannot_authorize_an_unbound_model_request(surface, operation, delivery):
    """No message snapshot reaches tool dispatch, even when the model saw a newer objection."""
    claim, body = mutation(surface, operation)
    if delivery == "restored_history":
        restored = surface.reopen()
        if isinstance(restored, ScopeRecallHermesAdapter):
            restored.on_turn_start(11, "保留原来的记录。")
    elif delivery == "same_turn_queued":
        event = source_event(source_event_key="TEST-command", content="保留原来的记录。")
        token, _ = enqueue(
            surface.core.storage,
            surface.core.clock,
            surface.context,
            event,
            scope_id=surface.scope,
            host_scope={"synthetic": True},
            remaining_seconds=5,
        )
        assert token is not None
    # The failed delivery has no stored row; the same-turn append has only an inbox row.
    # Both must refuse without borrowing the persisted command's identity.
    before = surface.core.storage.path.read_bytes()
    result = surface.call(operation, body)
    assert result.get("error", {}).get("field") == "request_message_binding", result
    assert surface.core.current_claim(surface.context, claim.ref).revision == 1
    assert surface.core.storage.path.read_bytes() == before


@pytest.mark.parametrize("operation", ["revise", "forget"])
def test_unbound_entry_refuses_before_target_lookup(surface, operation, monkeypatch):
    claim, body = mutation(surface, operation)

    def unexpected(*args, **kwargs):
        pytest.fail("an unbound tool request looked up a target")

    monkeypatch.setattr(Claims, "versions", unexpected)
    monkeypatch.setattr(Deletions, "target", unexpected)
    monkeypatch.setattr(Deletions, "receipt", unexpected)
    for target in (claim.ref, "claim-" + "f" * 64):
        payload = dict(body)
        if operation == "revise":
            payload["target_ref"] = target
        else:
            payload.update(target_refs=[target], expected_revisions={target: 1})
        assert surface.call(operation, payload)["error"]["field"] == "request_message_binding"


@pytest.mark.parametrize("operation", ["revise", "forget"])
def test_two_sessions_cannot_supply_each_others_tool_authority(surface, operation):
    _, body = mutation(surface, operation)
    other = replace(surface.context, session_id=str(uuid4()))
    surface.open_session(other.session_id)
    surface.capture("TEST-other-session", "保留原来的记录。", other)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda session: surface.call(operation, body, session), (surface.context.session_id, other.session_id)
            )
        )
    assert [item.get("error", {}).get("field") for item in results] == ["request_message_binding"] * 2


@pytest.mark.parametrize("operation", ["revise", "forget"])
def test_background_capture_can_commit_but_cannot_attest_a_tool_request(surface, operation):
    """Exercise real inbox replay; tool retry remains closed without native request proof."""
    claim, body = mutation(surface, operation)
    event = source_event(source_event_key="TEST-later-command", content=f"删除 {claim.ref}。")
    token, _ = enqueue(
        surface.core.storage,
        surface.core.clock,
        surface.context,
        event,
        scope_id=surface.scope,
        host_scope={"synthetic": True},
        remaining_seconds=5,
    )
    assert token is not None
    assert surface.call(operation, body)["error"]["field"] == "request_message_binding"
    receipts = replay_inbox(
        surface.core.storage,
        surface.core.clock,
        surface.context,
        authorize=lambda _: surface.context.allowed_scope_ids,
        remaining_seconds=5,
    )
    assert len(receipts) == 1 and receipts[0].durability == "persisted"
    saved = receipts[0].event_refs[0]
    assert surface.core.source(surface.context, saved.ref, saved.revision).event["content"] == event["content"]
    assert surface.call(operation, body)["error"]["field"] == "request_message_binding"
    assert surface.core.current_claim(surface.context, claim.ref).revision == 1


def test_core_revision_target_isolation_with_current_stored_authority(surface, monkeypatch):
    """Check the trusted primitive separately from the model-entry refusal."""
    claim, body = mutation(surface, "revise")
    other = replace(surface.context, project_id="TEST-other-project")
    source = surface.capture("TEST-other-proof", "Please correct TEST-project 配色: 银色。", other)
    proof = f"{source.ref}@{source.revision}"
    with surface.core.storage.read(other) as tx:
        assert tx.claims.current_human((proof,), surface.scope).ref == source.ref
    seen = []
    original = Claims.versions

    def versions(self, ref):
        seen.append(ref)
        return original(self, ref)

    monkeypatch.setattr(Claims, "versions", versions)
    targets = (claim.ref, "claim-" + "f" * 64)
    for target in targets:
        payload = dict(body, target_ref=target, source_evidence_refs=[proof])
        with pytest.raises(ContractError) as error:
            surface.core.revise(other, payload, remaining_seconds=5)
        assert (error.value.code, error.value.field) == ("SOURCE_MISSING", "payload")
    assert seen == list(targets)
