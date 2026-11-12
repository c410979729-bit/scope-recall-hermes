"""Read-only owner boundaries and optional candidate budgets over synthetic shared data."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import scope_recall.core.read_views as read_views
from scope_recall.adapters.clients.identity import resolve_runtime_audience, trusted_context
from scope_recall.adapters.shared_entries import (
    attach_shared_record,
    client_entry_record,
    read_shared_payload,
    shared_entry_manifest,
    write_shared_payload,
)
from scope_recall.contracts import ContractError, TrustedSourcePrincipal
from scope_recall.core.claims import Qualification
from scope_recall.core.recall_budget import canonical_render_json

from tests.host.hermes.test_profile_owner import (
    NOW,
    VISIBLE,
    admitted_owner_alias,
    fact,
    items,
    shared,
    stdio_context,
    view,
)


def wire(result):
    """The exact public serialization that the view charges against its budget."""
    return canonical_render_json(result).encode("utf-8")


def empty_ambiguous(core, reader, tool, monkeypatch):
    """Render the baseline's empty ambiguity path through the real read-view pipeline."""
    load = read_views._load_view

    def without_candidates(storage, clock, context, lookup, select, **kwargs):
        return load(
            storage,
            clock,
            context,
            lookup,
            lambda _tx, _now, _subject, resolution: read_views._Selection((), resolution.scan_capped, False),
            **kwargs,
        )

    with monkeypatch.context() as patch:
        patch.setattr(read_views, "_load_view", without_candidates)
        return view(core, reader, "TEST-Alex", tool)


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_stdio_uses_capture_approval_without_reading_process_account(shared, tool, monkeypatch):
    from scope_recall.adapters.clients import identity

    core, _reader, (first, _second), configs = shared
    target = fact(core, first, "我", "TEST-owner")
    probes = []

    def account():
        probes.append("read")
        return None

    monkeypatch.setattr(identity, "host_account_principal", account, raising=False)
    reader = replace(stdio_context(configs[0], monkeypatch), binding=core.storage.binding)
    assert {item["ref"] for item in items(view(core, reader, "me", tool))} == {target.ref}
    assert probes == []


@pytest.mark.parametrize("host", ["codex", "claude-code", "workbuddy", "dsh"])
@pytest.mark.parametrize("origin", ["human_direct", "memory_reinjection", "assistant_visible", "host_generated"])
def test_generic_client_context_has_equivalences_without_current_owner(shared, tmp_path, host, origin):
    core, _reader, _people, _configs = shared
    root = core.storage.binding.data_directory
    manifest = attach_shared_record(
        root,
        client_entry_record(
            host=host,
            home=tmp_path / "TEST-generic",
            entry_id="test-generic",
            display_name="TEST generic",
            attached_at=NOW,
            allowed_scope_ids=VISIBLE,
            writable_scope_ids=VISIBLE,
            capture_scope_id=VISIBLE[0],
        ),
        now=NOW,
    )
    from scope_recall.adapters.clients.config import load_shared_client

    config = load_shared_client(manifest.hermes_home, host)
    context = trusted_context(
        config, resolve_runtime_audience(config, None), session_id="TEST-generic", actor_origin=origin
    )
    captured = trusted_context(config, resolve_runtime_audience(config, None), session_id="TEST-captured")
    assert context.subject_resolver(captured.source_principal.principal_ref)
    assert context.subject_resolver(None) == ()


@pytest.mark.parametrize("tool", ["profile", "entity"])
@pytest.mark.parametrize("budget_case", ["exact_empty", "slightly_larger", "one_candidate"])
def test_candidate_budget_can_return_the_baseline_empty_response(
    shared, tool, budget_case, monkeypatch, record_testsuite_property
):
    core, reader, (first, _second), _configs = shared
    for scope in VISIBLE[:2]:
        fact(core, first, "TEST-Alex", "TEST-value", scope=scope)
    baseline = empty_ambiguous(core, reader, tool, monkeypatch)
    full = view(core, reader, "TEST-Alex", tool)
    empty_bytes = len(wire(baseline))
    budget = {"exact_empty": empty_bytes, "slightly_larger": empty_bytes + 1, "one_candidate": len(wire(full)) - 1}[
        budget_case
    ]
    result = view(core, reader, "TEST-Alex", tool, budget_tokens=budget)
    record_testsuite_property(
        f"budget_{tool}_{budget_case}",
        json.dumps(dict(baseline_bytes=empty_bytes, budget=budget, response_bytes=len(wire(result)), response=result)),
    )
    if budget_case != "one_candidate":
        assert wire(result) == wire(baseline)
        assert "candidate_groups" not in result
    else:
        assert len(result["candidate_groups"]) == 1
        assert sum(len(items(group)) for group in result["candidate_groups"]) == 1
        assert result["truncated"] and "budget_token_cap" in result["gaps"]
    assert len(wire(result)) <= budget


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_mixed_owner_and_nonowner_alias_targets_stay_ambiguous(shared, tool):
    core, reader, (first, second), _configs = shared
    owner, _alias = admitted_owner_alias(core, first, VISIBLE[0])
    outsider = replace(
        second, source_principal=TrustedSourcePrincipal("human", "verified", "principal:zz-TEST-outsider")
    )
    foreign, _other = admitted_owner_alias(core, outsider, VISIBLE[1])
    result = view(core, reader, "TEST-Alex", tool)
    assert result["alias_resolution"] == "ambiguous" and result["resolved_subject"] is None
    assert not items(result) and not result.get("candidate_groups")
    assert owner.ref not in json.dumps(result) and foreign.ref not in json.dumps(result)


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_literal_name_in_another_audience_blocks_owner_expansion(shared, tool):
    core, reader, (first, second), _configs = shared
    owner, _alias = admitted_owner_alias(core, first, VISIBLE[0])
    other = fact(core, second, "我", "TEST-owner-fact", scope=VISIBLE[1])
    literal = fact(core, first, "TEST-Alex", "TEST-literal", scope=VISIBLE[2])
    result = view(core, reader, "TEST-Alex", tool)
    assert result["alias_resolution"] == "ambiguous" and result["resolved_subject"] is None
    grouped = [item for group in result["candidate_groups"] for item in items(group)]
    assert {item["ref"] for item in grouped} == {literal.ref}
    assert owner.ref not in json.dumps(result) and other.ref not in json.dumps(result)


@pytest.mark.parametrize("tool", ["profile", "entity"])
@pytest.mark.parametrize("surface", ["item", "audience"])
def test_secret_candidate_surfaces_are_withheld_with_a_gap(shared, tool, surface, monkeypatch):
    core, reader, (first, _second), _configs = shared
    synthetic = "password: Synthetic7Only"
    if surface == "audience":
        first, reader = replace(first, project_id=synthetic), replace(reader, project_id=synthetic)
    from scope_recall.core import events

    # Legacy identifiers can predate ingress filtering; exercise the read boundary itself.
    with monkeypatch.context() as patch:
        patch.setattr(events, "capture_contains_secret", lambda *args, **kwargs: False)
        targets = [fact(core, first, "TEST-Alex", "TEST-safe", scope=scope) for scope in VISIBLE[:2]]
    if surface == "item":
        # An admitted legacy payload can require withholding even when its source is harmless.
        with core.storage.write(first) as tx:
            tx.claims.append(
                targets[0].scope_id,
                targets[0].payload | {"value_text": synthetic},
                Qualification("active", "direct_report", "supported"),
                recorded_at=core.clock.utc_now(),
                previous=targets[0],
            )
    result = view(core, reader, "TEST-Alex", tool)
    assert any(gap.startswith("secrets_withheld:") for gap in result["gaps"])
    assert "Synthetic7Only" not in json.dumps(result)
    grouped = [item for group in result.get("candidate_groups", []) for item in items(group)]
    assert {item["ref"] for item in grouped} == ({targets[1].ref} if surface == "item" else set())


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_a_literal_I_is_reserved_for_the_current_person(shared, tool):
    core, reader, (first, _second), _configs = shared
    owner = fact(core, first, "我", "TEST-owner-tea")
    target = fact(core, first, "TEST-artifact", "TEST-literal-artifact")
    with core.storage.write(first) as tx:
        literal = tx.claims.append(
            target.scope_id,
            target.payload | {"subject": "I"},
            Qualification("active", "direct_report", "supported"),
            recorded_at=core.clock.utc_now(),
        )
    assert {item["ref"] for item in items(view(core, reader, "I", tool))} == {owner.ref}
    unbound = replace(reader, source_principal=None, subject_resolver=None)
    with pytest.raises(ContractError, match="IDENTITY_UNBOUND"):
        view(core, unbound, "I", tool)
    assert core.current_claim(first, literal.ref).payload["subject"] == "I"


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_readable_entry_without_its_local_owner_approval_has_no_fallback(shared, tool, monkeypatch):
    from scope_recall.adapters.clients.config import load_shared_client

    core, reader, (first, _second), configs = shared
    target = fact(core, first, "TEST-Alex", "TEST-readable")
    payload = read_shared_payload(reader.binding.data_directory)
    payload["entries"][0]["owner_principals"] = [dict(platform="claude-code", user_id="TEST-alternate-owner")]
    payload["entries"][0]["audiences"][0]["kind"] = "conversation"
    write_shared_payload(reader.binding.data_directory, payload)
    checked = shared_entry_manifest(reader.binding.data_directory, configs[0].entry_id, hermes_home=configs[0].home)
    assert checked.owner_principals
    config = load_shared_client(configs[0].home, configs[0].host)
    context = stdio_context(config, monkeypatch)
    context = replace(context, binding=core.storage.binding)
    assert {item["ref"] for item in items(view(core, context, "TEST-Alex", tool))} == {target.ref}
    with pytest.raises(ContractError, match="IDENTITY_UNBOUND"):
        view(core, context, "me", tool)
