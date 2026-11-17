"""Provenance repair covers temporal versions, final release, audience fences and read-only opens."""

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate
from scope_recall.contracts import ContractError
from scope_recall.core.management import MemoryManagement
from scope_recall.core.recall_packet import RecallPacketCompiler, render_recall_packet_context
from scope_recall.core.restore import InstallationMaintenance
from scope_recall.maintenance import cli
from test_claims import accept, app, capture, draft, initial
from test_restore import store


def temporal_claims(core, context, *, mixed=False):
    """Admit the ordinary rule and a one-day exception through the real fact application path."""
    core.clock.now = "2026-11-12T12:00:00Z"
    source = capture(core, context, "TEST-project 配色 白底。", when="2026-11-01T12:00:00Z")
    first = accept(core, context, draft(source, "白底")).items[0]
    middle = None
    if mixed:
        other = capture(core, context, "TEST-project 配色 改用蓝底。", when="2026-11-05T12:00:00Z")
        proposal = draft(other, "蓝底")
        proposal["evidence_spans"].extend(draft(source)["evidence_spans"])
        middle = accept(core, context, proposal).items[0]
        assert middle.state == "active"
    temporary = capture(core, context, "TEST-project 配色 今天破例用黑底，只限今天。", when="2026-11-10T00:00:00Z")
    last = accept(core, context, draft(temporary, "黑底", valid_to="2026-11-11T00:00:00Z")).items[0]
    assert first.state == last.state == "active" and first.ref == last.ref
    return first, middle, last, source


def test_final_release_uses_the_rechecked_attribution_in_packet_and_injection(app, monkeypatch):
    core, context = app
    source = capture(core, context, "TEST-project color blue.")
    owner = MemoryManagement(core, InstallationMaintenance(context))
    verify = RecallPacketCompiler._verify
    changed = []

    def interleave(compiler, storage, draft_packet, ranked):
        verified = verify(compiler, storage, draft_packet, ranked)
        assert any(obj.origin == "human_direct" for _, obj in verified)
        owner.apply_origin(owner.preview_origin([source.ref], origin="host_generated", reason="Host notification"))
        changed.append(True)
        return verified

    monkeypatch.setattr(RecallPacketCompiler, "_verify", interleave)
    packet = core.recall_packet(
        context,
        dict(
            protocol_version="1.1",
            request_id="TEST-final-release",
            query="color",
            mode="current",
            focus_refs=[f"{source.ref}@1"],
            max_items=6,
            budget_tokens=4096,
        ),
        deadline_seconds=10,
    )
    assert changed == [True]
    assert packet["items"] and all(s["origin"] != "human_direct" for s in packet["items"])
    rendered = render_recall_packet_context(
        packet, installation_id=context.binding.installation_id, session_id=context.session_id
    )
    assert rendered.canonical_text and "human_direct" not in rendered.canonical_text
    assert "host_generated" in rendered.canonical_text


@pytest.mark.parametrize("section,field", [("procedure", "counterexample_refs"), ("intention", "state_evidence_refs")])
def test_historical_secondary_evidence_keeps_project_and_branch_fences(app, section, field):
    core, context = app
    item, source = initial(core, context)
    global_context = replace(context, project_id=None, branch_id=None)
    outside = capture(core, global_context, "TEST global evidence")
    proposal = draft(source)
    proposal[section] = {field: [f"{outside.ref}@1"]}
    for historical in (False, True):
        with pytest.raises(ContractError) as error:
            with core.storage.write(context) as tx:
                tx.claims._link_evidence(
                    item.ref, item.revision, source.scope_id, proposal, allow_historical=historical
                )
        assert error.value.code in {"SOURCE_MISSING", "ACCESS_DENIED", "DERIVATION_INVALID"}


def management_schema():
    return json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/memory_management.schema.json").read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "action,data",
    [
        (
            "origin",
            dict(
                applied=False,
                preview=dict(
                    origin="host_generated",
                    reason="TEST",
                    memory_epoch=0,
                    sources=[None],
                    claims=[42],
                    evaluations=["bad"],
                    work=[False],
                    preview_id="a" * 64,
                ),
            ),
        ),
        (
            "origin-candidates",
            dict(items=[dict(ref="event-TEST", kind="event", revision=1)], examined=1, next_after="not-an-integer"),
        ),
    ],
)
def test_management_contract_rejects_malformed_origin_results(action, data):
    with pytest.raises(ValidationError):
        validate(dict(format="scope-recall-management/1", action=action, status="ok", data=data), management_schema())


@pytest.mark.parametrize("section,field", [("procedure", "counterexample_refs"), ("intention", "state_evidence_refs")])
def test_historical_secondary_evidence_in_the_same_partition_still_works(app, section, field):
    core, context = app
    item, source = initial(core, context)
    extra = capture(core, context, "TEST old same-project evidence")
    capture(core, context, "TEST updated same-project evidence", key=extra.event["source_event_key"], revision=2)
    proposal = draft(source)
    proposal[section] = {field: [f"{extra.ref}@1"]}
    with core.storage.write(context) as tx:
        tx.claims._link_evidence(item.ref, item.revision, source.scope_id, proposal, allow_historical=True)


@pytest.mark.parametrize("action", ["origin", "origin-candidates"])
def test_real_cli_previews_read_wal_without_initialization(store, capsys, monkeypatch, action):
    from scope_recall.core import MemoryCore

    keeper = sqlite3.connect(store.core.storage.path)
    keeper.execute("PRAGMA journal_mode=WAL")
    source = store.say("TEST source for a real local maintenance command")
    assert store.core.storage.path.with_name(store.core.storage.path.name + "-wal").stat().st_size > 0
    before = tuple(keeper.iterdump())

    def forbidden(self):
        raise AssertionError("a maintenance preview must not initialize the store")

    monkeypatch.setattr(MemoryCore, "initialize", forbidden)
    arguments = ["memory", action, *store.target(), "--json"]
    if action == "origin":
        arguments.extend([source.ref, "--origin", "host_generated", "--reason", "Host notification"])
    try:
        assert cli.main(arguments) == 0
        result = json.loads(capsys.readouterr().out)
        validate(result, management_schema())
        if action == "origin":
            assert result["data"]["preview"]["sources"][0]["ref"] == source.ref
        else:
            assert result["data"]["examined"] >= 1
        assert tuple(keeper.iterdump()) == before
    finally:
        keeper.close()
