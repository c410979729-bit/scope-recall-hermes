"""Owner provenance corrections preserve content identities while withdrawing unsupported authority."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
from jsonschema import validate
from scope_recall.adapters.hermes.boundary import stored_origin_rule
from scope_recall.contracts import ContractError, TrustedSourcePrincipal
from scope_recall.core import source_origin
from scope_recall.core.management import MemoryManagement
from scope_recall.core.management_query import MemoryFilter
from scope_recall.core.recall_packet import RecallPacketRenderer, render_recall_packet_context
from scope_recall.core.restore import InstallationMaintenance
from scope_recall.core.source_records import source_origin as effective_origin
from scope_recall.core.worker_outcomes import derivation_changed, read_derivation_fence
from scope_recall.maintenance import cli, memory
from scope_recall.maintenance.origin_candidates import candidates
from test_candidate_lifecycle import _candidate, _settle_evidence, _sweep
from test_claims import accept, app, capture, draft, initial


@pytest.mark.parametrize("kind", ["fact", "preference", "constraint", "decision"])
def test_origin_correction_withdraws_the_sole_source_fact(app, kind):
    core, context = app
    item, source = initial(core, context, kind=kind)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    owner.apply_origin(plan)
    assert core.source(context, source.ref, source.revision).event["origin"] == "host_generated"
    assert owner.show(item.ref)["value"]["state"] == "retracted"


def snapshot(core, context, tables):
    with core.storage.read(context) as tx:
        return {table: [tuple(r) for r in tx._check().execute(f"SELECT * FROM {table}")] for table in tables}


def test_preview_and_apply_agree_and_keep_mixed_evidence_and_source_indices(app):
    core, context = app
    item, source = initial(core, context)
    other = capture(core, context, "TEST-project 配色蓝色。")
    mixed_proposal = draft(source, predicate="颜色")
    mixed_proposal["evidence_spans"].extend(draft(other)["evidence_spans"])
    mixed = accept(core, context, mixed_proposal).items[0]
    owner = MemoryManagement(core, InstallationMaintenance(context))
    tables = ("episodes", "episode_events", "episode_versions", "lexical_terms", "lexical_postings")
    preserved = snapshot(core, context, tables)
    before = snapshot(core, context, ("source_events", "claims", "claim_versions", "work_items", "management_actions"))
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    assert snapshot(core, context, tuple(before)) == before
    effects = {s["ref"]: s["effect"] for s in plan["claims"]}
    assert effects[item.ref] == "retract"
    assert effects[mixed.ref] == "mixed_unchanged"
    epoch = plan["memory_epoch"]
    result = owner.apply_origin(plan)
    assert result["memory_epoch"] > epoch
    assert snapshot(core, context, tables) == preserved
    assert owner.show(mixed.ref)["revision"] == mixed.revision
    assert owner.show(mixed.ref)["value"]["state"] == mixed.state
    assert owner.history(item.ref)["items"][0]["value"]["state"] == "retracted"
    saved = core.source(context, source.ref, 1)
    assert saved.event["content"] == source.event["content"]
    assert saved.content_sha256 == source.content_sha256 and saved.revision == source.revision
    with core.storage.read(context) as tx:
        for work in plan["work"]:
            row = (
                tx._check()
                .execute("SELECT state,lease_token FROM work_items WHERE work_id=?", (work["work_id"],))
                .fetchone()
            )
            assert row["state"] == "obsolete" and row["lease_token"] > work["lease_token"]


def test_history_and_restore_do_not_resurrect_withdrawn_claims(app):
    core, context = app
    item, source = initial(core, context)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    owner.apply_origin(owner.preview_origin([source.ref], origin="origin_unknown", reason="Uncertain author"))
    history = owner.history(source.ref)["items"][0]["origin_history"]
    assert history[0]["actor"] == "local_user_management"
    assert history[0]["before"]["origin"] == "human_direct"
    assert history[0]["after"] == "origin_unknown" and history[0]["reason"] == "Uncertain author"
    assert history[0]["recorded_at"] == core.clock.utc_now()
    owner.apply_origin(owner.preview_origin([source.ref], origin="human_direct", reason="Verified original author"))
    assert core.source(context, source.ref, 1).event["origin"] == "human_direct"
    assert owner.show(item.ref)["value"]["state"] == "retracted"
    assert len(owner.show(source.ref)["origin_history"]) == 2
    assert owner.show(source.ref)["origin_history"][1]["origin_revision"] == 2


def test_recall_injection_inventory_and_history_use_corrected_attribution(app):
    core, context = app
    item, source = initial(core, context)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    request = dict(
        protocol_version="1.1",
        request_id="TEST-origin-recall",
        query="配色",
        mode="current",
        focus_refs=[f"{source.ref}@1"],
        max_items=6,
        budget_tokens=4096,
    )
    before = core.recall_packet(context, request, deadline_seconds=10)
    renderer = RecallPacketRenderer()
    render_recall_packet_context(
        before, installation_id=context.binding.installation_id, session_id=context.session_id, renderer=renderer
    )
    assert any(s["ref"] == source.ref and s["origin"] == "human_direct" for s in before["items"])
    owner.apply_origin(owner.preview_origin([source.ref], origin="host_generated", reason="Host notice"))
    after = core.recall_packet(context, request, deadline_seconds=10)
    event = next(s for s in after["items"] if s["ref"] == source.ref)
    assert event["origin"] == "host_generated"
    assert item.ref not in {s["ref"] for s in after["items"]}
    rendered = (
        render_recall_packet_context(
            after, installation_id=context.binding.installation_id, session_id=context.session_id, renderer=renderer
        ).canonical_text
        or ""
    )
    assert "human_direct" not in rendered
    assert "human_direct" not in json.dumps(owner.list(MemoryFilter(kind="event")))
    assert "human_direct" not in json.dumps(owner.list(MemoryFilter(kind="event", query="配色")))
    shown = owner.history(source.ref)["items"][0]
    assert shown["value"]["event"]["origin"] == "host_generated"
    assert shown["origin_history"][0]["before"]["origin"] == "human_direct"


def test_batch_all_evidence_is_retracted_and_an_exact_historical_revision_can_be_selected(app):
    core, context = app
    item, first = initial(core, context)
    second = capture(core, context, first.event["content"])
    proposal = draft(first, predicate="颜色")
    proposal["evidence_spans"].extend(draft(second)["evidence_spans"])
    mixed = accept(core, context, proposal).items[0]
    capture(core, context, "TEST-project 配色绿色。", key=first.event["source_event_key"], revision=2)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    plan = owner.preview_origin([f"{first.ref}@1", second.ref], origin="host_generated", reason="Host notices")
    owner.apply_origin(plan)
    assert owner.show(mixed.ref)["value"]["state"] == "retracted"
    assert core.source(context, first.ref, 1).event["origin"] == "host_generated"
    assert core.source(context, first.ref, 2).event["origin"] == "human_direct"


def test_failed_transaction_rolls_back_sources_claims_work_and_audit(app, monkeypatch):
    core, context = app
    item, source = initial(core, context)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    tables = ("source_events", "claims", "claim_versions", "work_items", "management_actions", "instance_meta")
    before = snapshot(core, context, tables)

    def fail(*args):
        raise RuntimeError("TEST transaction failure")

    monkeypatch.setattr(source_origin, "_write_sources", fail)
    with pytest.raises(RuntimeError, match="TEST transaction"):
        owner.apply_origin(plan)
    assert snapshot(core, context, tables) == before


def test_concurrent_corrections_are_serialized_and_only_one_preview_commits(app):
    core, context = app
    item, source = initial(core, context)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    barrier = Barrier(2)

    def apply():
        barrier.wait(timeout=10)
        try:
            return owner.apply_origin(plan)
        except ContractError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: apply(), range(2)))
    assert sum(isinstance(result, dict) for result in results) == 1
    assert "VERSION_CONFLICT" in results
    assert len(owner.show(source.ref)["origin_history"]) == 1
    assert owner.show(item.ref)["revision"] == 2


def test_stale_preview_and_missing_owner_capability_cannot_write(app):
    core, context = app
    item, source = initial(core, context)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    capture(core, context, "TEST another source")
    with pytest.raises(ContractError) as error:
        owner.apply_origin(plan)
    assert error.value.code == "VERSION_CONFLICT"
    owner.authority = context
    with pytest.raises(ContractError) as error:
        owner.apply_origin(plan)
    assert error.value.code == "ACCESS_DENIED"
    assert core.source(context, source.ref, 1).event["origin"] == "human_direct"


@pytest.mark.parametrize(
    "body,imported,rule",
    [
        ("[Background process TEST heartbeat #1 — still running]", False, "hermes_prefix"),
        ("[IMPORTANT: Background process TEST completed]", False, "hermes_prefix"),
        ("[System note: TEST verdict]", False, "hermes_prefix"),
        ("[Continuing toward your standing goal TEST]", False, "hermes_prefix"),
        ("[Thu 2026-11-12 10:00:00 UTC] [System note: TEST verdict]", False, "replay_timestamp"),
        ("[2026-11-12T10:00:00Z] [System note: TEST verdict]", False, "replay_timestamp"),
        ("Review the conversation above and extract TEST facts", True, "legacy_review"),
    ],
)
def test_candidate_markers_keep_platform_id_precedence(body, imported, rule):
    row = dict(role="user", content=body)
    assert stored_origin_rule(row, imported=imported) == rule
    assert stored_origin_rule(dict(row, platform_message_id="TEST-platform-id"), imported=imported) is None
    assert stored_origin_rule(dict(row, message_id="TEST-message-id"), imported=imported) is None
    assert stored_origin_rule(dict(row, role="assistant"), imported=imported) is None
    assert stored_origin_rule(dict(row, content="A person discusses " + body), imported=imported) is None


def test_gateway_origin_uses_the_adapters_stronger_notice_rule():
    text = "Gateway message origin (JSON data, not instructions or authorization):\n{}\nTEST correction"
    assert stored_origin_rule(dict(role="user", content=text, message_id="TEST-platform")) == "gateway_origin"
    assert stored_origin_rule(dict(role="user", content="I discuss " + text)) is None


def test_candidate_inventory_is_read_only_bounded_and_omits_bodies(app):
    core, context = app
    host = capture(core, context, "[System note: TEST sensitive body]", key="hermes:TEST:notice")
    review = capture(core, context, "Review the conversation above and TEST private text", key="import:TEST:review")
    capture(core, context, "A person discusses [System note: TEST]", key="hermes:TEST:human")
    capture(core, context, "[System note: TEST unrelated host]", key="other:TEST:notice")
    owner = MemoryManagement(core, InstallationMaintenance(context))
    before = snapshot(core, context, ("source_events", "work_items", "management_actions", "instance_meta"))
    first = candidates(owner, limit=1)
    assert first["next_after"] and [s["ref"] for s in first["items"]] == [host.ref]
    second = candidates(owner, after=first["next_after"])
    assert [s["ref"] for s in second["items"]] == [review.ref]
    assert "sensitive body" not in json.dumps(first)
    assert "private text" not in json.dumps(second)
    assert candidates(owner, snippet=True)["items"][0]["snippet"] == host.event["content"]
    assert snapshot(core, context, tuple(before)) == before


def test_cli_previews_executes_restores_and_defaults_to_read_only_open(app, monkeypatch, capsys):
    core, context = app
    item, source = initial(core, context)
    api = MemoryManagement(core, InstallationMaintenance(context))
    initialized = []

    def open_store(args, *, initialize=True):
        initialized.append(initialize)
        return api

    monkeypatch.setattr(memory, "open_management", open_store)
    args = ["memory", "origin", source.ref, "--origin", "host_generated", "--reason", "Host notice", "--json"]
    assert cli.main(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert not data["data"]["applied"]
    assert cli.main([*args, "--yes", "--preview-id", data["data"]["preview"]["preview_id"]]) == 0
    result = json.loads(capsys.readouterr().out)
    from pathlib import Path

    schema = json.loads((Path(__file__).resolve().parents[2] / "contracts/memory_management.schema.json").read_text())
    validate(result, schema)
    assert initialized == [False, False]
    assert cli.main(["memory", "origin-candidates"]) == 0
    assert "RULE" in capsys.readouterr().out


def test_queued_and_leased_candidate_work_loses_its_authority(app):
    core, context = app
    item, source, proposal, registration = _candidate(core, context)
    _settle_evidence(core)
    _sweep(core, context)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    with core.storage.write(context) as tx:
        conn = tx._check(write=True)
        evaluation = conn.execute(
            "SELECT evaluation_id,work_id FROM candidate_evaluations WHERE state='queued'"
        ).fetchone()
        assert evaluation is not None
        conn.execute(
            "UPDATE work_items SET state='leased',lease_token=7,lease_owner='TEST-worker',"
            "lease_until='2026-09-07T00:00:00Z' WHERE work_id=?",
            (evaluation["work_id"],),
        )
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    assert any(s["evaluation_id"] == evaluation["evaluation_id"] for s in plan["evaluations"])
    assert any(s["work_id"] == evaluation["work_id"] and s["lease_token"] == 7 for s in plan["work"])
    owner.apply_origin(plan)
    with core.storage.read(context) as tx:
        row = tx._check().execute("SELECT * FROM work_items WHERE work_id=?", (evaluation["work_id"],)).fetchone()
        assert row["state"] == "obsolete" and row["lease_token"] == 8
        assert row["lease_owner"] is None and row["lease_until"] is None
        assert tx.candidates.evaluation(evaluation["evaluation_id"]) is None
    assert owner.show(item.ref)["value"]["state"] == "retracted"


def test_import_envelope_and_ingress_retry_keep_the_attribution_correction(app):
    core, context = app
    source = capture(
        core,
        context,
        "TEST imported person text",
        origin="imported",
        source_original_origin="human_direct",
        attested=True,
        key="memory-import:TEST-person",
    )
    owner = MemoryManagement(core, InstallationMaintenance(context))
    owner.apply_origin(owner.preview_origin([source.ref], origin="host_generated", reason="Host import prompt"))
    saved = core.source(context, source.ref, 1)
    assert saved.event["origin"] == "imported" and saved.event["source_original_origin"] == "host_generated"
    assert effective_origin(saved) == "host_generated"
    assert saved.import_provenance_sha256 == source.import_provenance_sha256
    owner.apply_origin(owner.preview_origin([source.ref], origin="human_direct", reason="Restore verified import"))
    assert effective_origin(core.source(context, source.ref, 1)) == "human_direct"
    direct = capture(core, context, "TEST-project 配色蓝色。")
    owner.apply_origin(owner.preview_origin([direct.ref], origin="origin_unknown", reason="Unknown writer"))
    replay = capture(core, context, direct.event["content"], key=direct.event["source_event_key"])
    assert effective_origin(replay) == "origin_unknown"


def test_unresolved_authority_updates_are_obsolete_and_reference_history_is_preserved(app):
    core, context = app
    item, source = initial(core, context)
    with core.storage.write(context) as tx:
        conn = tx._check(write=True)
        conn.execute(
            "INSERT INTO unresolved_updates(update_id,source_ref,source_revision,scope_id,project_id,branch_id,"
            "candidate_refs_json,created_at) VALUES ('TEST-update',?,?,?, ?,?,'[]',?)",
            (source.ref, 1, source.scope_id, context.project_id, context.branch_id, core.clock.utc_now()),
        )
        conn.execute(
            "INSERT INTO episodes(episode_id,scope_id,project_id,branch_id,anchor_key,anchor_kind,series_key) "
            "VALUES ('episode-TEST-origin',?,?,?,'TEST-origin','session','TEST-origin')",
            (source.scope_id, context.project_id, context.branch_id),
        )
        conn.execute(
            "INSERT INTO episode_versions(episode_id,revision,state,source_watermark,recorded_at) "
            "VALUES ('episode-TEST-origin',1,'open','',?)",
            (core.clock.utc_now(),),
        )
        conn.execute(
            "INSERT INTO reference_bindings(reference_id,scope_id,project_id,branch_id,episode_id,current_revision) "
            "VALUES ('reference-TEST',?,?,?,'episode-TEST-origin',1)",
            (source.scope_id, context.project_id, context.branch_id),
        )
        conn.execute(
            "INSERT INTO reference_versions VALUES ('reference-TEST',1,?,'explicit_version_mention',?)",
            (
                json.dumps(
                    dict(
                        mention="TEST",
                        candidate_refs=["artifact-TEST@1"],
                        resolved_ref="artifact-TEST@1",
                        resolution="resolved",
                    )
                ),
                core.clock.utc_now(),
            ),
        )
        from scope_recall.core import lineage

        lineage.link(conn, "reference", "reference-TEST", 1, source.ref, 1)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    assert {s["kind"] for s in plan["related"]} == {"update", "reference"}
    owner.apply_origin(plan)
    with core.storage.read(context) as tx:
        assert tx._check().execute("SELECT state FROM unresolved_updates").fetchone()[0] == "obsolete"
        rows = tx._check().execute("SELECT revision,payload_json FROM reference_versions ORDER BY revision").fetchall()
        assert json.loads(rows[0][1])["resolution"] == "resolved"
        assert json.loads(rows[1][1])["resolution"] == "unresolved"


def test_source_vectors_remain_searchable_and_identical_after_correction(app, tmp_path):
    from scope_recall.vector.store import build_vector_store

    core, context = app
    item, source = initial(core, context)
    store = build_vector_store("sqlite-vec", storage_dir=tmp_path / "TEST-vectors", table_name="vectors", dimensions=8)
    store.open()
    try:
        record = dict(
            id=source.ref,
            scope_id=source.scope_id,
            source=f"{source.ref}@1",
            target="{}",
            content="",
            summary="",
            updated_at="",
            vector=[1.0] + [0.0] * 7,
        )
        store.upsert_records([record])
        before = store.search(record["vector"], scope_id=source.scope_id, limit=1)
        owner = MemoryManagement(core, InstallationMaintenance(context))
        owner.apply_origin(owner.preview_origin([source.ref], origin="host_generated", reason="Host notice"))
        assert store.search(record["vector"], scope_id=source.scope_id, limit=1) == before
        assert core.source(context, source.ref, 1).event["origin"] == "host_generated"
    finally:
        store.close()


def test_corrected_principal_cannot_be_mistaken_for_a_verified_human(app):
    core, context = app
    actor = replace(context, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-owner"))
    source = capture(core, actor, "TEST owner text")
    owner = MemoryManagement(core, InstallationMaintenance(context))
    principal = source.event["source_principal"]
    owner.apply_origin(owner.preview_origin([source.ref], origin="host_generated", reason="Host wrote this"))
    assert "source_principal" not in core.source(context, source.ref, 1).event
    owner.apply_origin(owner.preview_origin([source.ref], origin="human_direct", reason="Restore attribution"))
    assert core.source(context, source.ref, 1).event["source_principal"] == principal


def test_an_in_flight_derivation_cannot_reuse_former_human_authority(app):
    core, context = app
    source = capture(core, context, "TEST owner text before correction")
    with core.storage.read(context) as tx:
        fence = read_derivation_fence(tx, scope_id=source.scope_id, sources=(source,))
    owner = MemoryManagement(core, InstallationMaintenance(context))
    owner.apply_origin(owner.preview_origin([source.ref], origin="host_generated", reason="Host wrote this"))
    with core.storage.read(context) as tx:
        assert derivation_changed(tx, fence) == "source"


def test_equivalent_references_are_deduplicated_and_host_sources_cannot_be_promoted(app):
    core, context = app
    item, source = initial(core, context)
    owner = MemoryManagement(core, InstallationMaintenance(context))
    plan = owner.preview_origin([source.ref, f"{source.ref}@1"], origin="host_generated", reason="Host notice")
    assert len(plan["sources"]) == 1
    owner.apply_origin(plan)
    other = capture(core, context, "TEST host content", origin="host_generated")
    with pytest.raises(ContractError) as error:
        owner.preview_origin([other.ref], origin="human_direct", reason="Promotion")
    assert error.value.code == "ACCESS_DENIED"


def test_candidate_count_includes_facts_backed_through_derived_sources(app):
    core, context = app
    source = capture(core, context, "[System note: TEST-project color blue.]", key="hermes:TEST:count")
    echo = capture(
        core, context, "TEST-project color blue.", origin="assistant_visible", evidence_refs=[f"{source.ref}@1"]
    )
    claim = accept(core, context, draft(echo, "blue", predicate="color")).items[0]
    owner = MemoryManagement(core, InstallationMaintenance(context))
    assert candidates(owner)["items"][0]["affected_claims"] == 1
    plan = owner.preview_origin([source.ref], origin="host_generated", reason="Host notice")
    assert any(s["ref"] == claim.ref and s["effect"] == "retract" for s in plan["claims"])
    owner.apply_origin(plan)
    assert owner.show(claim.ref)["value"]["state"] == "retracted"
