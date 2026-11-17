"""Synthetic retained rollouts joined through real hooks and MCP dispatch."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from scope_recall.adapters.clients.config import load_shared_client
from scope_recall.adapters.clients.handler import CodexHookHandler
from scope_recall.adapters.clients.identity import resolve_runtime_audience, trusted_context
from scope_recall.adapters.clients.mcp_server import build_server
from scope_recall.contracts import ContractError
from scope_recall.core import CoreConfig, MemoryCore

from tests.host.test_local_account_pin import shared_world
from tests.host.test_mutation_authorization import mutation
from tests.v11_support import source_event


def user_row(text, turn, *, message="TEST-message", sequence=1, order=0):
    """Create the retained user shape observed in the native contract."""
    return dict(
        timestamp="2026-11-16T00:00:01Z",
        type="response_item",
        payload=dict(
            type="message",
            id=message,
            role="user",
            content=[dict(type="input_text", text=text)],
            internal_chat_message_metadata_passthrough=dict(
                turn_id=turn,
                content_item_kinds=["user.text"],
                content_item_metadata=[dict(provenance=dict(type="user"))],
            ),
        ),
        metadata=dict(
            history_sequence=sequence,
            sampled_through_sequence=0,
            retained_source=dict(
                id=dict(message_id=message, turn_id=turn, role="user"),
                revision="TEST-retained-" + message,
                complete=True,
            ),
            client_authored=False,
            user_input_order=order,
            mcp_attribution=dict(status="none"),
        ),
    )


@pytest.fixture
def native_codex(tmp_path):
    """Use a pinned shared entry without any live host, models or credentials."""
    shared_world(tmp_path)
    config = load_shared_client(tmp_path / "first", "codex")
    core = MemoryCore(CoreConfig(config.to_binding()))
    session, turn = str(uuid4()), str(uuid4())
    audience = resolve_runtime_audience(config, str(config.home))
    context = trusted_context(
        config, audience, session_id=session, actor_origin="human_direct", mutation=True, project_root=str(config.home)
    )
    with core.storage.write(context) as tx:
        tx.registry.register_entry(config.entry_id, config.entry_name, "codex", now=core.clock.utc_now())
    server = build_server(config, workspace=config.home, core=core)
    hook = CodexHookHandler(config, core=core)
    scope = audience.capture_scope_id

    def capture(key, text, actor=context):
        event = source_event(source_event_key=key, content=text, occurred_at=None, time_precision="unknown")
        return core.record_event(actor, event, scope_id=scope).event_refs[0]

    world = SimpleNamespace(core=core, context=context, scope=scope, capture=capture)
    claim, body = mutation(world, "revise")
    record = config.home / "sessions" / "TEST-rollout.jsonl"
    record.parent.mkdir(parents=True)
    rows = [dict(type="session_meta", payload=dict(id=session, cwd=str(config.home)))]

    def write():
        record.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def event(name, **fields):
        return hook.handle_payload(
            dict(
                hook_event_name=name,
                session_id=session,
                turn_id=turn,
                cwd=str(config.home),
                transcript_path=str(record),
                **fields,
            )
        )

    def prepare(text="Please correct TEST-project 配色: 银色。", *, tool="revise", payload=None, order=0):
        rows[:] = rows[:1] + [user_row(text, turn, order=order)]
        write()
        event("PreModelRequest")
        rows.append(
            dict(
                timestamp="2026-11-16T00:00:02Z",
                type="response_item",
                payload=dict(
                    type="custom_tool_call",
                    id="TEST-item",
                    name="exec",
                    namespace="functions",
                    internal_chat_message_metadata_passthrough=dict(turn_id=turn),
                ),
                metadata=dict(history_sequence=2, sampled_through_sequence=1, client_authored=False),
            )
        )
        write()
        if tool == "revise":
            from scope_recall.adapters.clients.boundary import host_source_key

            key = host_source_key(
                host="codex",
                installation_id=config.installation_id,
                session_id=session,
                event_kind="user",
                event_id=turn,
            )
            with core.storage.read(context) as tx:
                source = tx.sources.source_by_event_key(key)
            assert source is not None, hook.diagnostics
            body["source_evidence_refs"] = [f"{source.ref}@{source.revision}"]
        used = body if payload is None else payload
        event("PreToolUse", tool_use_id="TEST-call", tool_name="mcp__scope-recall__" + tool, tool_input=used)
        return used

    meta = dict(
        threadId=session,
        sessionId=session,
        windowId=session + ":0",
        callId="TEST-call",
        itemId="TEST-item",
        **{"x-codex-turn-metadata": dict(session_id=session, thread_id=session, turn_id=turn)},
    )

    def call(tool, payload, metadata=meta):
        return getattr(server, tool)(SimpleNamespace(request_context=SimpleNamespace(meta=metadata)), **payload)

    yield SimpleNamespace(
        **vars(world),
        config=config,
        server=server,
        hook=hook,
        session=session,
        turn=turn,
        claim=claim,
        body=body,
        rows=rows,
        record=record,
        write=write,
        event=event,
        prepare=prepare,
        meta=meta,
        call=call,
    )
    hook.close()


def test_codex_stdio_native_witness_allows_suppression(native_codex):
    """The actual MCP transport consumes hook evidence and commits a legal suppression."""
    w = native_codex
    body = dict(protocol_version="1.1", target_refs=[w.claim.ref], mode="suppress", expected_revisions={w.claim.ref: 1})
    w.prepare(f"Suppress {w.claim.ref}.", tool="forget", payload=body)

    async def run():
        env = dict(os.environ)
        repo = Path(__file__).parents[3]
        env["PYTHONPATH"] = os.pathsep.join(part for part in (str(repo), env.get("PYTHONPATH", "")) if part)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "scope_recall.adapters.codex.mcp_entry", "--home", str(w.config.home), "--host", "codex"],
            env=env,
            cwd=str(w.config.home),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("forget", body, meta=w.meta)
                assert not result.is_error, result
                replay = await session.call_tool("forget", body, meta=w.meta)
                assert replay.is_error

    asyncio.run(run())
    with w.core.storage.read(w.context) as tx:
        assert tx.claims.versions(w.claim.ref)[-1].suppressed


def test_codex_public_revision_uses_native_source(native_codex):
    w = native_codex
    w.prepare()
    result = w.call("revise", w.body)
    assert result["result"]["items"][0]["revision"] == 2
    assert w.core.current_claim(w.context, w.claim.ref).payload["value_text"] == "银色"


def test_codex_revision_receipt_requires_a_new_tool_witness(native_codex):
    """The same user and request ID survive a fresh native tool call ID."""
    w = native_codex
    w.body["request_id"] = "TEST-durable-revise"
    w.prepare()
    first = w.call("revise", w.body)
    assert first["result"]["items"][0]["revision"] == 2
    epoch = w.core.status(w.context).memory_epoch
    with pytest.raises(ContractError):
        w.call("revise", w.body)
    row = json.loads(json.dumps(w.rows[-1]))
    row["payload"]["id"] = "TEST-new-item"
    row["metadata"]["history_sequence"] = 3
    w.rows.append(row)
    w.write()
    w.event("PreToolUse", tool_use_id="TEST-new-call", tool_name="mcp__scope-recall__revise", tool_input=w.body)
    meta = dict(w.meta, callId="TEST-new-call", itemId="TEST-new-item")
    result = w.call("revise", w.body, meta)
    assert result["result"] == first["result"]
    assert w.core.status(w.context).memory_epoch == epoch
    assert w.core.current_claim(w.context, w.claim.ref).revision == 2


def test_codex_proposal_links_user_without_promoting_authority(native_codex):
    w = native_codex
    body = dict(protocol_version="1.1", content="TEST-proposed 配色 金色。")
    w.prepare("TEST user message", tool="propose_memory", payload=body)
    result = w.call("propose_memory", body)["result"]
    assert result["authority"] == "assistant_visible_only"
    ref = result["receipt"]["event_refs"][0]
    source = w.core.source(w.context, ref["ref"], ref["revision"])
    assert source.event["origin"] == "assistant_visible" and source.event["role"] == "assistant"
    assert len(source.event["evidence_refs"]) == 1
    linked, revision = source.event["evidence_refs"][0].rsplit("@", 1)
    assert w.core.source(w.context, linked, int(revision)).event["content"] == "TEST user message"
    proposal = dict(
        kind="fact",
        subject="TEST-proposed",
        predicate="配色",
        value_text="金色",
        conditions=[],
        statement_kind="assertion",
        valid_from=None,
        valid_to=None,
        evidence_spans=[dict(source_ref=source.ref, source_revision=source.revision, quote=body["content"])],
    )
    try:
        accepted = w.core.records.accept_claim_proposals(
            w.context,
            dict(
                protocol_version="1.1",
                source_refs=[f"{source.ref}@{source.revision}"],
                claim_proposals=[proposal],
                resume_proposals=[],
                reference_proposals=[],
            ),
            scope_id=w.scope,
        )
    except ContractError:
        pass
    else:
        assert all(item.state != "active" for item in accepted.items)


@pytest.mark.parametrize(
    "missing", ["pretool", "watermark", "new_request", "new_prompt", "new_user", "other_item", "bad_refs"]
)
def test_codex_nearby_invalid_witnesses_refuse(native_codex, missing):
    w = native_codex
    w.prepare()
    if missing == "pretool":
        w.meta["callId"] = "TEST-manual-rpc"
    elif missing == "watermark":
        w.rows[-1]["metadata"].pop("sampled_through_sequence")
        w.write()
    elif missing == "new_request":
        w.event("PreModelRequest")
    elif missing == "new_prompt":
        w.event("UserPromptSubmit", prompt="Keep the original record.")
    elif missing == "new_user":
        w.rows.append(user_row("Keep the original record.", w.turn, message="TEST-steer", sequence=3, order=1))
        w.write()
    elif missing == "other_item":
        w.meta["itemId"] = "TEST-other-item"
    else:
        w.body["source_evidence_refs"] = ["event-broken@not-a-revision"]
    with pytest.raises(ContractError):
        w.call("revise", w.body)
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


def test_codex_first_user_order_need_not_restart_at_zero(native_codex):
    w = native_codex
    w.prepare(order=7)
    assert w.call("revise", w.body)["result"]["items"][0]["revision"] == 2


def test_codex_identical_text_has_distinct_native_sources(native_codex):
    w = native_codex
    w.prepare()
    original = w.body["source_evidence_refs"][0]
    w.rows.append(
        user_row("Please correct TEST-project 配色: 银色。", w.turn, message="TEST-second", sequence=3, order=1)
    )
    w.write()
    w.event("PreModelRequest")
    from scope_recall.adapters.clients.boundary import host_source_key

    key = host_source_key(
        host="codex",
        installation_id=w.config.installation_id,
        session_id=w.session,
        event_kind="record",
        event_id="TEST-second",
    )
    with w.core.storage.read(w.context) as tx:
        source = tx.sources.source_by_event_key(key)
    assert source is not None
    w.body["source_evidence_refs"] = [f"{source.ref}@{source.revision}"]
    assert w.body["source_evidence_refs"][0] != original
    parent = dict(
        w.rows[2],
        payload=dict(w.rows[2]["payload"], id="TEST-new-item"),
        metadata=dict(history_sequence=4, sampled_through_sequence=3, client_authored=False),
    )
    w.rows.append(parent)
    w.write()
    w.event("PreToolUse", tool_use_id="TEST-new-call", tool_name="mcp__scope-recall__revise", tool_input=w.body)
    meta = dict(w.meta, callId="TEST-new-call", itemId="TEST-new-item")
    assert w.call("revise", w.body, meta)["result"]["items"][0]["revision"] == 2


def test_codex_deleted_opener_is_not_reimported_for_binding(native_codex):
    w = native_codex
    w.prepare()
    ref = w.body["source_evidence_refs"][0].rsplit("@", 1)[0]
    w.capture("TEST-delete-command", f"Delete {ref}.")
    w.core.forget(
        w.context, dict(protocol_version="1.1", target_refs=[ref], mode="delete", expected_revisions={ref: 1})
    )
    w.event("PreModelRequest")
    w.event("PreToolUse", tool_use_id="TEST-new-call", tool_name="mcp__scope-recall__revise", tool_input=w.body)
    with pytest.raises(ContractError):
        w.call("revise", w.body, dict(w.meta, callId="TEST-new-call"))
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1
