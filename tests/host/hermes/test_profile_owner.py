"""Synthetic shared-owner read views across independently attached local clients."""

from __future__ import annotations

import itertools
import json
from dataclasses import replace
from functools import partial

import pytest
from scope_recall.adapters.clients.config import load_shared_client
from scope_recall.adapters.clients.identity import resolve_runtime_audience, trusted_context
from scope_recall.adapters.host_identity import owner_subjects
from scope_recall.adapters.shared_entries import (
    attach_shared_record,
    client_entry_record,
    new_shared_payload,
    write_shared_payload,
)
from scope_recall.contracts import ContractError, TrustedSourcePrincipal
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.core.claims import Qualification
from scope_recall.core.recall_budget import canonical_render_json

from tests.contract.test_claims import Clock, draft
from tests.v11_support import source_event

VISIBLE = ("TEST-private", "TEST-project-a", "TEST-project-b")
DENIED = "TEST-denied-scope"
NOW = "2026-09-01T12:00:00Z"


def stdio_context(config, monkeypatch):
    """Bind through the actual stdio entry without starting a transport, worker or model."""
    from types import SimpleNamespace

    from scope_recall.adapters.clients import local_endpoint, mcp_entry

    context = trusted_context(
        config, resolve_runtime_audience(config, None), session_id="TEST-mcp", actor_origin="memory_reinjection"
    )
    tools = SimpleNamespace(context=context, server=object())
    with monkeypatch.context() as patch:
        patch.setattr(mcp_entry, "build_server", lambda *args, **kwargs: tools)
        patch.setattr(mcp_entry, "run_stdio", lambda server: None)
        patch.setattr(local_endpoint, "request_recall", lambda *args, **kwargs: None)
        assert mcp_entry.main(["--home", str(config.home), "--host", config.host]) == 0
    return tools.context


@pytest.fixture
def shared(tmp_path, monkeypatch):
    """Two real entry manifests approve different local principals as one owner."""
    root = tmp_path / "TEST-shared-profile"
    write_shared_payload(root, new_shared_payload(root, agent_id="TEST-agent", test_mode=True))
    manifests = []
    for host, scopes in (("claude-code", VISIBLE), ("codex", (*VISIBLE, DENIED))):
        record = client_entry_record(
            host=host,
            home=tmp_path / ("TEST-" + host),
            entry_id=host,
            display_name="TEST " + host,
            attached_at=NOW,
            allowed_scope_ids=scopes,
            writable_scope_ids=scopes,
            capture_scope_id=VISIBLE[0],
        )
        manifests.append(attach_shared_record(root, record, now=NOW))
    core = MemoryCore(CoreConfig(manifests[-1].to_binding()), clock=Clock())
    core.initialize()
    core.test_sequence = itertools.count(1)
    configs = [load_shared_client(manifest.hermes_home, manifest.entry_host) for manifest in manifests]
    contexts = [
        trusted_context(config, resolve_runtime_audience(config, None), session_id="TEST-conversation")
        for config in configs
    ]
    reader = stdio_context(configs[0], monkeypatch)
    binding = core.storage.binding
    return core, replace(reader, binding=binding), [replace(ctx, binding=binding) for ctx in contexts], configs


def fact(core, ctx, subject, value, *, scope=VISIBLE[0], predicate="喜欢", text=None):
    """Capture and admit one fully synthetic fact without a model or network."""
    text = text or (f"我{predicate}{value}。" if subject == "我" else f"{subject} {predicate} {value}.")
    event = source_event(source_event_key=f"TEST-profile/{next(core.test_sequence)}", content=text)
    receipt = core.record_event(ctx, event, scope_id=scope, remaining_seconds=10)
    source = core.source(ctx, receipt.event_refs[0].ref, 1)
    proposal = draft(source, value, subject=subject, kind="fact", statement_kind="assertion", predicate=predicate)
    accepted = core.records.accept_claim_proposals(
        ctx,
        dict(
            protocol_version="1.1",
            source_refs=[f"{source.ref}@1"],
            claim_proposals=[proposal],
            resume_proposals=[],
            reference_proposals=[],
        ),
        scope_id=scope,
        remaining_seconds=10,
    ).items[0]
    assert accepted.state == "active"
    return core.current_claim(ctx, accepted.ref)


def view(core, reader, subject, tool="profile", **changes):
    request = dict(protocol_version="1.1", request_id="TEST-profile", subject=subject, max_items=30, budget_tokens=8000)
    if tool == "entity":
        request["action"] = "probe"
    return getattr(core, tool)(reader, request | changes)


def items(result):
    if "statements" in result:
        return result["statements"]
    return [item for section in result["sections"].values() for item in section] + result["disputed"]


@pytest.mark.parametrize("tool", ["profile", "entity"])
@pytest.mark.parametrize("alias", ["me", "myself", "i", "I", "user", "current_user", "用户", "我"])
def test_first_person_words_resolve_without_a_conversation(shared, tool, alias):
    core, reader, (first, _second), _configs = shared
    target = fact(core, first, "我", "TEST-tea")
    result = view(core, reader, alias, tool)
    print(json.dumps(result, ensure_ascii=False))
    assert result["resolved_subject"] == first.source_principal.principal_ref
    assert {item["ref"] for item in items(result)} == {target.ref}


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_approved_owner_subjects_are_read_together_without_merging_storage(shared, tool):
    core, reader, (first, second), _configs = shared
    targets = [fact(core, first, "我", "TEST-tea"), fact(core, second, "我", "TEST-cycling", scope=VISIBLE[1])]
    stranger = replace(second, source_principal=TrustedSourcePrincipal("human", "verified", "TEST-other-person"))
    excluded = fact(core, stranger, "我", "TEST-stranger")
    hidden = fact(core, second, "我", "TEST-hidden-owner", scope=DENIED)
    result = view(core, first, "user", tool)
    print(json.dumps(result, ensure_ascii=False))
    assert {item["ref"] for item in items(result)} == {target.ref for target in targets}
    assert {item["subject"] for item in items(result)} == {target.payload["subject"] for target in targets}
    assert all(item.get("entries") for item in items(result))
    assert {entry["id"] for item in items(result) for entry in item["entries"]} == {"claude-code", "codex"}
    assert excluded.ref not in json.dumps(result) and hidden.ref not in json.dumps(result)
    assert "TEST-other-person" not in json.dumps(result) and DENIED not in json.dumps(result)
    assert all(core.current_claim(second, target.ref).payload == target.payload for target in targets)


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_same_name_returns_separate_readable_audiences(shared, tool):
    core, reader, (first, second), _configs = shared
    targets = [fact(core, first, "TEST-Alex", "TEST-" + scope, scope=scope) for scope in VISIBLE]
    fact(core, first, "我", "TEST-owner-only")
    before = view(core, reader, "TEST-Alex", tool)
    hidden = fact(core, second, "TEST-Alex", "TEST-hidden-name", scope=DENIED)
    result = view(core, reader, "TEST-Alex", tool)
    print(json.dumps(result, ensure_ascii=False))
    assert result["alias_resolution"] == "ambiguous" and result["resolved_subject"] is None
    assert items(result) == []
    groups = result.get("candidate_groups", [])
    assert {group["audience"]["scope_id"] for group in groups} == set(VISIBLE)
    assert {item["ref"] for group in groups for item in items(group)} == {target.ref for target in targets}
    assert all(len(items(group)) == 1 for group in groups)
    assert hidden.ref not in json.dumps(result) and DENIED not in json.dumps(result)
    assert result | {"memory_epoch": before["memory_epoch"]} == before


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_grouped_views_keep_global_item_and_byte_limits(shared, tool):
    core, reader, (first, _second), _configs = shared
    for scope in VISIBLE:
        fact(core, first, "TEST-Alex", "TEST-value", scope=scope)
    limited = view(core, reader, "TEST-Alex", tool, max_items=1)
    assert sum(len(items(group)) for group in limited["candidate_groups"]) == 1
    assert limited["truncated"] and "max_items_cap" in limited["gaps"]
    bounded = view(core, reader, "TEST-Alex", tool, budget_tokens=1500)
    assert len(canonical_render_json(bounded).encode("utf-8")) <= 1500
    assert bounded["truncated"] and "budget_token_cap" in bounded["gaps"]
    assert all(items(group) for group in bounded.get("candidate_groups", []))
    with pytest.raises(ContractError, match="budget_tokens"):
        view(core, reader, "TEST-Alex", tool, budget_tokens=64)


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_group_labels_preserve_project_and_branch_release_fences(shared, tool):
    core, reader, (first, _second), _configs = shared
    contexts = [first, replace(first, project_id="TEST-p"), replace(first, project_id="TEST-p", branch_id="TEST-b")]
    targets = [fact(core, ctx, "TEST-Alex", f"TEST-value-{index}") for index, ctx in enumerate(contexts)]
    fact(core, replace(first, project_id="TEST-unreadable-project"), "TEST-Alex", "TEST-unreadable")
    result = view(core, replace(reader, project_id="TEST-p", branch_id="TEST-b"), "TEST-Alex", tool)
    assert {item["ref"] for group in result["candidate_groups"] for item in items(group)} == {targets[-1].ref}
    assert {(g["audience"]["project_id"], g["audience"]["branch_id"]) for g in result["candidate_groups"]} == {
        ("TEST-p", "TEST-b"),
    }
    assert "TEST-unreadable" not in json.dumps(result)


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_message_routed_context_requires_a_verified_speaker(shared, tool):
    core, reader, (first, second), _configs = shared
    gateway = replace(reader, subject_resolver=partial(owner_subjects, reader.binding, reader.entry_id))
    with pytest.raises(ContractError, match="IDENTITY_UNBOUND"):
        view(core, gateway, "me", tool)
    stranger = replace(second, source_principal=TrustedSourcePrincipal("human", "verified", "TEST-visitor"))
    target = fact(core, stranger, "我", "TEST-visitor-tea")
    fact(core, first, "我", "TEST-owner-tea")
    result = view(core, replace(gateway, source_principal=stranger.source_principal), "me", tool)
    assert {item["ref"] for item in items(result)} == {target.ref}


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_owner_approval_changes_during_release_close_the_view(shared, tool, monkeypatch):
    import scope_recall.core.read_views as read_views
    from scope_recall.adapters.shared_entries import read_shared_payload

    core, reader, (first, second), _configs = shared
    first_fact = fact(core, first, "我", "TEST-tea")
    fact(core, second, "我", "TEST-cycling", scope=VISIBLE[1])
    payload = read_shared_payload(reader.binding.data_directory)
    retained = dict(platform="codex", user_id="TEST-owner-backup")
    payload["entries"][1]["owner_principals"].append(retained)
    write_shared_payload(reader.binding.data_directory, payload)
    release = read_views._release_selected

    def revoke(*args, **kwargs):
        released = release(*args, **kwargs)
        payload = read_shared_payload(reader.binding.data_directory)
        payload["entries"][1]["owner_principals"] = [retained]
        write_shared_payload(reader.binding.data_directory, payload)
        from scope_recall.adapters.shared_entries import shared_entry_manifest

        shared_entry_manifest(reader.binding.data_directory, second.entry_id)
        assert payload["entries"][1]["owner_principals"] == [retained]
        return released

    monkeypatch.setattr(read_views, "_release_selected", revoke)
    result = view(core, reader, "me", tool)
    assert result["status"] == "unavailable" and not items(result)
    assert not result.get("candidate_groups") and "TEST-tea" not in json.dumps(result)
    stable = view(core, reader, "me", tool)
    assert {item["ref"] for item in items(stable)} == {first_fact.ref}


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_group_release_failure_and_delivery_fence_remove_all_candidates(shared, tool, monkeypatch):
    import scope_recall.core.read_views as read_views
    from scope_recall.adapters.tool_common import FENCED_ENTITY, FENCED_PROFILE, fence_epoch

    core, reader, (first, _second), _configs = shared
    for scope in VISIBLE:
        fact(core, first, "TEST-Alex", "TEST-value", scope=scope)
    result = view(core, reader, "TEST-Alex", tool)
    fenced = fence_epoch(result, result["memory_epoch"] + 1, FENCED_ENTITY if tool == "entity" else FENCED_PROFILE)
    assert fenced["status"] == "unavailable" and not fenced.get("candidate_groups")
    assert "TEST-value" not in json.dumps(fenced)

    def reject(*args, **kwargs):
        raise ContractError("VERSION_CONFLICT", "memory_epoch")

    monkeypatch.setattr(read_views, "_release_selected", reject)
    unavailable = view(core, reader, "TEST-Alex", tool)
    assert unavailable["status"] == "unavailable" and not unavailable.get("candidate_groups")


def admitted_owner_alias(core, person, scope):
    """Seed an admitted owner name and relation, leaving identity admission outside this read contract."""
    from tests.contract.test_aliases import _alias

    captured = source_event(
        source_event_key=f"TEST-owner-name/{next(core.test_sequence)}", content="I use TEST-Owner as my name."
    )
    captured_ref = core.record_event(person, captured, scope_id=scope, remaining_seconds=10).event_refs[0]
    captured_source = core.source(person, captured_ref.ref, 1)
    name_payload = draft(
        captured_source,
        "TEST-Owner",
        subject=person.source_principal.principal_ref,
        kind="fact",
        statement_kind="assertion",
        predicate="name",
    )
    with core.storage.write(person) as tx:
        target = tx.claims.append(
            scope, name_payload, Qualification("active", "direct_report", "supported"), recorded_at=NOW
        )
    raw = source_event(
        source_event_key=f"TEST-alias/{next(core.test_sequence)}", content="TEST-Owner renamed to TEST-Alex."
    )
    source_ref = core.record_event(person, raw, scope_id=scope, remaining_seconds=10).event_refs[0]
    source = core.source(person, source_ref.ref, 1)
    # Seed a previously admitted relation; the reader still revalidates its exact
    # target and source proof. Identity admission and stored slot keys stay unchanged.
    payload = _alias(source, target, "TEST-Alex") | {"subject": target.payload["subject"]}
    with core.storage.write(person) as tx:
        alias = tx.claims.append(scope, payload, Qualification("active", "direct_report", "supported"), recorded_at=NOW)
    return target, alias


@pytest.mark.parametrize("tool", ["profile", "entity"])
@pytest.mark.parametrize("both_entries", [False, True])
def test_accepted_alias_to_owner_uses_approved_subjects_only(shared, tool, both_entries):
    core, reader, (first, second), _configs = shared
    admitted = [admitted_owner_alias(core, first, VISIBLE[0])]
    if both_entries:
        admitted.append(admitted_owner_alias(core, second, VISIBLE[1]))
    other = fact(core, second, "我", "TEST-cycling", scope=VISIBLE[1])
    result = view(core, reader, "TEST-Alex", tool)
    assert result["alias_resolution"] == "resolved"
    assert {item["ref"] for item in items(result)} == {other.ref, *(target.ref for target, _alias in admitted)}
    with core.storage.write(second) as tx:
        for _target, alias in admitted:
            tx.claims.append(
                alias.scope_id,
                alias.payload,
                Qualification("retracted", "direct_report", "user_withdrawal"),
                recorded_at=NOW,
                previous=alias,
            )
    assert not items(view(core, reader, "TEST-Alex", tool))


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_one_visible_name_does_not_disclose_an_unreadable_collision(shared, tool):
    core, reader, (first, second), _configs = shared
    fact(core, first, "TEST-Alex", "TEST-visible")
    before = view(core, reader, "TEST-Alex", tool)
    fact(core, second, "TEST-Alex", "TEST-hidden", scope=DENIED)
    result = view(core, reader, "TEST-Alex", tool)
    # The store-wide epoch is public existing metadata; name resolution and content must be identical.
    assert result | {"memory_epoch": before["memory_epoch"]} == before


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_local_stdio_fallback_uses_capture_approval_without_an_account_probe(shared, tool, monkeypatch):
    core, _reader, _contexts, configs = shared
    from scope_recall.adapters.clients import identity

    monkeypatch.setattr(identity, "host_account_principal", lambda: None, raising=False)
    reader = stdio_context(configs[0], monkeypatch)
    reader = replace(reader, binding=core.storage.binding)
    captured = trusted_context(configs[0], resolve_runtime_audience(configs[0], None), session_id="TEST-capture")
    assert view(core, reader, "me", tool)["resolved_subject"] == captured.source_principal.principal_ref


@pytest.mark.parametrize("host", ["codex", "claude-code", "workbuddy", "dsh"])
def test_only_approved_local_clients_supply_an_owner_fallback(shared, tmp_path, host, monkeypatch):
    core, _reader, _contexts, _configs = shared
    root = core.storage.binding.data_directory
    manifest = attach_shared_record(
        root,
        client_entry_record(
            host=host,
            home=tmp_path / ("TEST-local-" + host),
            entry_id="test-local",
            display_name="TEST local",
            attached_at=NOW,
            allowed_scope_ids=VISIBLE,
            writable_scope_ids=VISIBLE,
            capture_scope_id=VISIBLE[0],
        ),
        now=NOW,
    )
    config = load_shared_client(manifest.hermes_home, host)
    reader = stdio_context(config, monkeypatch)
    assert reader.source_principal.kind != "human"
    assert (
        reader.subject_resolver(None)[0]
        == trusted_context(
            config, resolve_runtime_audience(config, None), session_id="TEST-conversation"
        ).source_principal.principal_ref
    )


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_shared_owner_views_remain_read_only_and_obey_private_and_pause_states(shared, tool, monkeypatch):
    from scope_recall.core.management import MemoryManagement
    from scope_recall.core.restore import InstallationMaintenance

    core, reader, (first, second), _configs = shared
    fact(core, first, "我", "TEST-tea")
    fact(core, second, "我", "TEST-cycling", scope=VISIBLE[1])

    def refuse_write(*args, **kwargs):
        pytest.fail("a read view requested a write transaction")

    with monkeypatch.context() as patch:
        patch.setattr(type(core.storage), "write", refuse_write)
        assert items(view(core, reader, "me", tool))
    owner = MemoryManagement(core, InstallationMaintenance(second))
    owner.private_on(target="session", session=reader.session_id)
    with pytest.raises(ContractError, match="private_mode"):
        view(core, reader, "me", tool)
    owner.private_off(target="session", session=reader.session_id)
    owner.set_entry_state(reader.entry_id, "paused", actor="TEST-owner")
    paused = view(core, reader, "me", tool)
    assert paused["status"] == "unavailable" and paused["gaps"] == ["entry_state:paused"]
    assert not items(paused) and not paused.get("candidate_groups")


@pytest.mark.parametrize("thread", [None, "00000000-0000-4000-8000-000000000001"])
def test_remote_mcp_never_inherits_a_local_owner_fallback(shared, tmp_path, monkeypatch, thread):
    from types import SimpleNamespace

    from scope_recall.adapters.clients import mcp_server, remote_server
    from scope_recall.core.read_views import _profile_subject
    from starlette.applications import Starlette

    _core, _reader, _contexts, configs = shared
    config = configs[1]
    audience = resolve_runtime_audience(config, None)
    tools = object.__new__(mcp_server.CodexMCPServer)
    tools.config, tools.audience, tools.host = config, audience, config.host
    tools.session_id, tools.workspace, tools._host_runtime = "TEST-remote-mcp", tmp_path, None
    tools.context = trusted_context(config, audience, session_id=tools.session_id, actor_origin="memory_reinjection")
    with pytest.raises(ContractError, match="IDENTITY_UNBOUND"):
        _profile_subject(tools.context, "me")
    tools.server = SimpleNamespace(streamable_http_app=lambda **kwargs: Starlette())
    monkeypatch.setattr(mcp_server, "build_server", lambda *args, **kwargs: tools)
    # No socket, host runtime or background hook is started by constructing the app.
    remote_server.build_app(remote_server.RemoteServerConfig(config.home, config.host, "127.0.0.1", 19000, "0" * 64))
    call = SimpleNamespace(request_context=SimpleNamespace(meta={"threadId": thread} if thread else {}))
    context = tools._request_context(call)
    assert tools._context("assistant_visible").subject_resolver(None) == ()
    approved = trusted_context(config, audience, session_id="TEST-owner").source_principal.principal_ref
    assert len(context.subject_resolver(approved)) == 2
    with pytest.raises(ContractError, match="IDENTITY_UNBOUND"):
        _profile_subject(context, "me")
