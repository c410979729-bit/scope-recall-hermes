"""Synthetic read-view compatibility at strict schema and complete-proof boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from scope_recall.adapters.clients import local_endpoint, mcp_entry
from scope_recall.adapters.clients.config import load_shared_client
from scope_recall.adapters.clients.identity import resolve_runtime_audience, trusted_context
from scope_recall.adapters.clients.mcp_server import CodexMCPServer
from scope_recall.adapters.shared_entries import (
    attach_shared_record,
    client_entry_record,
    new_shared_payload,
    write_shared_payload,
)
from scope_recall.contracts import TrustedSourcePrincipal, validate_payload
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.core.claims import Qualification
from scope_recall.core.read_views import resolve_subject
from scope_recall.core.recall_budget import canonical_render_json

from tests.contract.test_aliases import _alias
from tests.contract.test_claims import Clock, draft
from tests.contract.test_profile_entity import app
from tests.contract.test_trace import edge, query
from tests.host.hermes.test_profile_owner import NOW, VISIBLE, fact, items, shared, view
from tests.v11_support import source_event

GOLDENS = Path(__file__).parents[2] / "fixtures" / "profile_read_compatibility.json"


def _wire(value):
    return canonical_render_json(value).encode("utf-8")


def test_shared_trace_retains_entry_labels_on_each_edge(shared):
    core, reader, (person, _other), _configs = shared
    target = fact(core, person, "TEST-A", "TEST-B", predicate="owns")
    result = query(core, reader, target="TEST-B", direction="outgoing", max_hops=1)
    assert result["status"] == "ok" and len(result["paths"]) == 1
    relation = result["paths"][0]["edges"][0]
    assert relation["ref"] == target.ref
    assert relation["entries"] == [{"id": "claude-code", "name": "TEST claude-code"}]
    validate_payload("trace_view", result)


def test_standalone_trace_matches_the_fixed_baseline_response(app):
    core, context = app
    edge(core, context, "TEST-A", "TEST-B", predicate="owns")
    result = query(core, context, target="TEST-B", direction="outgoing", max_hops=1)
    expected = json.loads(GOLDENS.read_text(encoding="utf-8"))["standalone_trace"].encode("utf-8")
    assert "entries" not in result["paths"][0]["edges"][0]
    assert _wire(result) == expected


@pytest.mark.parametrize("tool", ["profile", "entity"])
def test_small_candidate_budget_matches_the_fixed_baseline_response(shared, tool):
    core, reader, (person, _other), _configs = shared
    for scope in VISIBLE[:2]:
        fact(core, person, "TEST-Alex", "TEST-value", scope=scope)
    expected = json.loads(GOLDENS.read_text(encoding="utf-8"))[tool].encode("utf-8")
    result = view(core, reader, "TEST-Alex", tool, budget_tokens=len(expected))
    assert _wire(result) == expected


def _seed_alias(core, person, scope, index, *, semantic=False):
    """Seed admitted relations with separate slots, including optional pinned entity proofs."""
    raw = source_event(
        source_event_key=f"TEST-release-alias/{index}",
        content="TEST-Owner renamed to TEST-Alex. TEST-Alex is a person.",
    )
    receipt = core.record_event(person, raw, scope_id=scope, remaining_seconds=10)
    source = core.source(person, receipt.event_refs[0].ref, 1)
    payload = draft(
        source,
        "TEST-Owner",
        subject=person.source_principal.principal_ref,
        kind="fact",
        predicate="name",
        statement_kind="assertion",
        conditions=[f"TEST-context-{index}"],
    )
    qualification = Qualification("active", "direct_report", "supported")
    with core.storage.write(person) as tx:
        target = tx.claims.append(scope, payload, qualification, recorded_at=NOW)
        relation = _alias(source, target, "TEST-Alex") | {
            "subject": target.payload["subject"],
            "conditions": payload["conditions"],
        }
        alias = tx.claims.append(scope, relation, qualification, recorded_at=NOW)
        refs = {target.ref, alias.ref}
        if semantic:
            proof = tx.claims.append(
                scope,
                payload | {"subject": "TEST-Alex", "predicate": "category", "value_text": "person"},
                qualification,
                recorded_at=NOW,
            )
            tx._check(write=True).execute(
                "INSERT INTO entity_alias_proofs VALUES (?,?,?,?,?,?,?)",
                (alias.ref, alias.revision, proof.ref, proof.revision, target.ref, target.revision, "a" * 64),
            )
            refs.add(proof.ref)
    return refs


def _many_aliases(tmp_path, count, *, owner, semantic):
    """Use distinct approved scopes for an owner, or distinct slots in one non-owner audience."""
    root = tmp_path / "TEST-alias-release"
    scopes = tuple(f"TEST-scope-{n:03d}" for n in range(count if owner else 1))
    write_shared_payload(root, new_shared_payload(root, agent_id="TEST-agent", test_mode=True))
    manifest = attach_shared_record(
        root,
        client_entry_record(
            host="codex",
            home=tmp_path / "TEST-entry",
            entry_id="codex",
            display_name="TEST Codex",
            attached_at=NOW,
            allowed_scope_ids=scopes,
            writable_scope_ids=scopes,
            capture_scope_id=scopes[0],
        ),
        now=NOW,
    )
    core = MemoryCore(CoreConfig(manifest.to_binding()), clock=Clock())
    core.initialize()
    config = load_shared_client(manifest.hermes_home, "codex")
    reader = trusted_context(config, resolve_runtime_audience(config, None), session_id="TEST-reader")
    person = (
        reader
        if owner
        else replace(reader, source_principal=TrustedSourcePrincipal("human", "verified", "TEST-person"))
    )
    refs = set()
    for index in range(count):
        refs.update(_seed_alias(core, person, scopes[index] if owner else scopes[0], index, semantic=semantic))
    return core, reader, person.source_principal.principal_ref, refs


@pytest.mark.parametrize("tool", ["profile", "entity"])
@pytest.mark.parametrize("count", [100, 101])
@pytest.mark.parametrize("owner", [False, True], ids=["nonowner", "owner"])
def test_alias_targets_share_the_complete_release_object_limit(tmp_path, tool, count, owner, record_testsuite_property):
    core, reader, subject, refs = _many_aliases(tmp_path, count, owner=owner, semantic=False)
    assert len(refs) == count * 2
    if not owner:
        # Query-name rewriting does not publish these objects and keeps its existing resolver contract.
        with core.storage.read(reader) as tx:
            assert resolve_subject(tx, "TEST-Alex", core.clock.utc_now()).status == "resolved"
    result = view(core, reader, "TEST-Alex", tool, max_items=4)
    record_testsuite_property(f"alias_{tool}_{owner}_{count}", canonical_render_json(result))
    if count == 101:
        assert result["alias_resolution"] == "ambiguous" and result["resolved_subject"] is None
        assert result["scan_capped"] and "scan_capped" in result["gaps"]
        assert not items(result) and not result.get("candidate_groups")
    else:
        assert result["alias_resolution"] == "resolved" and result["resolved_subject"] == subject
        assert len(items(result)) == 4


@pytest.mark.parametrize("tool", ["profile", "entity"])
@pytest.mark.parametrize("count", [66, 67])
def test_entity_alias_source_proofs_also_count_towards_release(tmp_path, tool, count):
    core, reader, subject, refs = _many_aliases(tmp_path, count, owner=True, semantic=True)
    assert len(refs) == count * 3
    result = view(core, reader, "TEST-Alex", tool, max_items=16)
    if count == 67:
        assert result["alias_resolution"] == "ambiguous" and result["scan_capped"]
        assert not items(result) and not result.get("candidate_groups")
    else:
        assert result["alias_resolution"] == "resolved" and result["resolved_subject"] == subject
        assert items(result)


def test_real_stdio_server_keeps_owner_fallback_after_thread_rebinding(shared, monkeypatch):
    core, _reader, (_first, person), configs = shared
    config = configs[1]
    target = fact(core, person, "我", "TEST-thread-owner")
    server = CodexMCPServer(config, workspace=config.home, core=core)
    meta = SimpleNamespace(request_context=SimpleNamespace(meta={"threadId": "11111111-2222-4333-8444-555555555555"}))
    assert server._request_context(meta).subject_resolver(None) == ()
    monkeypatch.setattr(mcp_entry, "build_server", lambda *args, **kwargs: server)
    monkeypatch.setattr(mcp_entry, "run_stdio", lambda transport: None)
    monkeypatch.setattr(local_endpoint, "request_recall", lambda *args, **kwargs: None)
    assert mcp_entry.main(["--home", str(config.home), "--host", config.host]) == 0
    rebound = server._request_context(meta)
    assert rebound.session_id != server.context.session_id
    assert rebound.subject_resolver(None)[0] == person.source_principal.principal_ref
    assert {item["ref"] for item in items(view(core, rebound, "me"))} == {target.ref}
