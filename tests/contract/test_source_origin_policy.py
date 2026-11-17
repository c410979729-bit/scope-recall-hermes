"""Only unambiguous whole-claim withdrawals belong to provenance maintenance."""

import sqlite3
from collections import defaultdict
from contextlib import closing

import pytest
from jsonschema import validate
from scope_recall.core.claims import select_effective, select_proposal
from scope_recall.core.management import MemoryManagement
from scope_recall.core.requalify import restore_displaced_values
from scope_recall.core.restore import InstallationMaintenance, _replay_withdrawal, export_deletion_ledger
from scope_recall.core.storage import Transaction
from scope_recall.maintenance.legacy_claims import _insert_version
from scope_recall.maintenance.origin_candidates import candidates
from test_claims import _denial_at_the_head, _preferred, accept, app, capture, draft
from test_source_origin_repairs import management_schema, temporal_claims


def owner(core, context):
    return MemoryManagement(core, InstallationMaintenance(context))


def versions(core, context, ref):
    with core.storage.read(context) as tx:
        return tx.claims.versions(ref)


def backup(core, context):
    """Use SQLite's backup API without creating a second installation on disk."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    with core.storage.read(context) as tx:
        tx._check().backup(conn)
    return conn


def service(core, context, source, value, *extra):
    proposal = draft(source, value, kind="fact", predicate="CI service", statement_kind="assertion")
    for item in extra:
        proposal["evidence_spans"].extend(draft(item)["evidence_spans"])
    result = accept(core, context, proposal).items[0]
    assert result.state == "active"
    return result


def replay(core, context, conn):
    ledger = export_deletion_ledger(core.storage, InstallationMaintenance(context))
    for withdrawal in ledger["withdrawals"]:
        _replay_withdrawal(conn, context, ledger, withdrawal)
    return ledger


def test_same_period_shadowed_history_uses_one_ordinary_head_withdrawal(app):
    core, context = app
    source_a = capture(core, context, "The TEST-project CI service is Jenkins.")
    first = service(core, context, source_a, "Jenkins")
    source_b = capture(core, context, "The TEST-project CI service is CircleCI.")
    last = service(core, context, source_b, "CircleCI")
    authority = owner(core, context)
    core.clock.now = "2026-09-07T12:00:00Z"
    with closing(backup(core, context)) as saved:
        plan = authority.preview_origin([source_a.ref, source_b.ref], origin="host_generated", reason="Host rows")
        authority.apply_origin(plan)
        current = versions(core, context, last.ref)
        replay(core, context, saved)
        restored = Transaction(saved, context, writable=False).claims.versions(last.ref)
    for history in (current, restored):
        for instant in ("2026-08-31T12:00:00Z", "2026-09-01T12:00:00Z", "2026-09-06T12:00:00Z", "2027-01-01T00:00:00Z"):
            for as_of in (False, True):
                assert select_effective(history, instant, as_of=as_of) is None
            assert select_proposal(history, instant) is None
    assert (
        select_effective(current, core.clock.now, known_at="2026-09-06T12:00:00Z").payload["value_text"] == "CircleCI"
    )
    assert [(s["revision"], s["effect"]) for s in plan["claims"]] == [(last.revision, "retract")]
    assert [s["revision"] for s in plan["claims"][0]["versions"]] == [first.revision, last.revision]
    assert current[-1].current_revision == current[-1].revision == last.revision + 1
    assert current[-1].replaces_revision == last.revision
    assert current[-1].reason == "source_origin_corrected" and current[-1].payload == current[-2].payload


@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_different_periods_or_other_version_support_require_review(app, both, mixed):
    core, context = app
    first, middle, last, source = temporal_claims(core, context, mixed=mixed)
    before = versions(core, context, first.ref)
    selected = [source.ref]
    if both:
        selected.extend(s["source_ref"] for s in before[-1].payload["evidence_spans"])
    authority = owner(core, context)
    plan = authority.preview_origin(selected, origin="host_generated", reason="Host rows")
    assert [(s["revision"], s["effect"]) for s in plan["claims"]] == [(last.revision, "review")]
    details = plan["claims"][0]["versions"]
    assert [(s["revision"], s["state"], s["valid_from"], s["valid_to"]) for s in details] == [
        (s.revision, s.state, s.valid_from, s.valid_to) for s in before
    ]
    assert details[0]["evidence_distribution"] == "selected_only"
    assert details[-1]["evidence_distribution"] == ("selected_only" if both else "elsewhere")
    if middle:
        assert next(s for s in details if s["revision"] == middle.revision)["evidence_distribution"] == "mixed"
    validate(
        dict(format="scope-recall-management/1", action="origin", status="ok", data=dict(preview=plan, applied=False)),
        management_schema(),
    )
    authority.apply_origin(plan)
    assert versions(core, context, first.ref) == before


def test_review_does_not_change_governance_ledger_replay(app):
    core, context = app
    core.clock.now = "2026-11-12T12:00:00Z"
    source_a = capture(core, context, "The TEST-project CI service is Jenkins.")
    source_b = capture(core, context, "TEST-project uses Jenkins for CI too.")
    first = service(core, context, source_a, "Jenkins", source_b)
    source_d = capture(core, context, "The TEST-project CI service is CircleCI.", when="2026-11-05T00:00:00Z")
    last = service(core, context, source_d, "CircleCI")
    before = versions(core, context, first.ref)
    authority = owner(core, context)
    with closing(backup(core, context)) as saved:
        plan = authority.preview_origin([source_d.ref], origin="host_generated", reason="Host row")
        authority.apply_origin(plan)
        current = versions(core, context, first.ref)
        ledger = replay(core, context, saved)
        restored = Transaction(saved, context, writable=False).claims.versions(first.ref)
    values = tuple(
        select_effective(history, core.clock.now).payload["value_text"]
        if select_effective(history, core.clock.now)
        else None
        for history in (current, restored)
    )
    assert values == ("CircleCI", "CircleCI")
    assert plan["claims"][0]["effect"] == "review" and not ledger["withdrawals"]
    assert current == before and current[-1].revision == last.revision


def test_legacy_denial_head_requires_review_and_preserves_existing_repair(app):
    core, context = app
    source = capture(core, context, "The TEST-project CI service is Jenkins.")
    first = service(core, context, source, "Jenkins")
    _denial_at_the_head(core, context, first, "The TEST-project CI service is not CircleCI.", "not CircleCI")
    before = versions(core, context, first.ref)
    authority = owner(core, context)
    with closing(backup(core, context)) as saved:
        baseline = restore_displaced_values(
            Transaction(saved, context, writable=True), now=core.clock.now, dry_run=False
        )
        expected = Transaction(saved, context, writable=False).claims.versions(first.ref)
    plan = authority.preview_origin([source.ref], origin="host_generated", reason="Host row")
    assert plan["claims"][0]["effect"] == "review"
    authority.apply_origin(plan)
    assert versions(core, context, first.ref) == before
    actual = core.operations.restore_displaced_values(context, dry_run=False, remaining_seconds=10)
    assert actual == baseline.to_dict()
    assert versions(core, context, first.ref) == expected


def test_the_persons_own_negative_preference_is_an_ordinary_sole_source_claim(app):
    core, context = app
    item, context = _preferred(core, context, "I do not eat spicy food.", "do not eat spicy food")
    before = versions(core, context, item.ref)
    assert before[-1].state == "active"
    refs = [s["source_ref"] for s in before[-1].payload["evidence_spans"]]
    authority = owner(core, context)
    plan = authority.preview_origin(refs, origin="host_generated", reason="Host row")
    assert plan["claims"][0]["effect"] == "retract"
    authority.apply_origin(plan)
    assert select_effective(versions(core, context, item.ref), core.clock.now) is None


def test_a_sole_source_proposal_is_no_longer_selected_after_ordinary_withdrawal(app):
    core, context = app
    source = capture(core, context, "The TEST-project CI service is Jenkins.")
    # An unsupported earlier date keeps the real admission in proposal state.
    item = accept(
        core,
        context,
        draft(
            source,
            "Jenkins",
            kind="fact",
            predicate="CI service",
            statement_kind="assertion",
            valid_from="2026-08-01T00:00:00Z",
        ),
    ).items[0]
    before = versions(core, context, item.ref)
    assert item.state == "proposed" and select_proposal(before, core.clock.now) is not None
    authority = owner(core, context)
    plan = authority.preview_origin([source.ref], origin="host_generated", reason="Host row")
    assert plan["claims"][0]["effect"] == "retract"
    authority.apply_origin(plan)
    after = versions(core, context, item.ref)
    for instant in ("2026-07-31T00:00:00Z", "2026-08-01T00:00:00Z", core.clock.now, "2027-01-01T00:00:00Z"):
        assert select_proposal(after, instant) is None
        assert select_effective(after, instant) is None
        assert select_effective(after, instant, as_of=True) is None


@pytest.mark.parametrize("action", ["preview", "candidates"])
def test_origin_readers_see_committed_rows_still_in_wal(app, action):
    core, context = app
    # Keep a connection open so the final writer does not checkpoint the committed row on close.
    with closing(sqlite3.connect(core.storage.path)) as keeper:
        keeper.execute("PRAGMA journal_mode=WAL")
        source = capture(core, context, "[System note: TEST host row]", key="hermes:TEST:wal")
        assert core.storage.path.with_name(core.storage.path.name + "-wal").stat().st_size > 0
        authority = owner(core, context)
        if action == "preview":
            result = authority.preview_origin([source.ref], origin="host_generated", reason="Host row")
            assert result["sources"][0]["ref"] == source.ref
        else:
            result = candidates(authority)
            assert any(s["ref"] == source.ref for s in result["items"])


def migrated_version(core, context, payload, *, evidence):
    """Write one fact version the way a legacy store migration does: evidence only as links."""
    with core.storage.write(context) as tx:
        _insert_version(
            tx._check(write=True),
            claim_id="TEST-old-C",
            scope_id="TEST-scope",
            project_id="TEST-project",
            branch_id="TEST-main",
            payload=payload,
            state="active",
            basis="direct_report",
            reason="legacy_current",
            recorded="2026-09-05T12:00:00Z",
            valid_from=payload["valid_from"],
            valid_to=payload["valid_to"],
            evidence=evidence,
            counts=defaultdict(int),
        )


def jenkins(**changes):
    payload = dict(
        kind="fact",
        subject="TEST-project",
        predicate="CI service",
        value_text="Jenkins",
        conditions=[],
        statement_kind="assertion",
        valid_from="2026-09-01T12:00:00Z",
        valid_to=None,
        evidence_spans=[],
    )
    payload.update(changes)
    return payload


def test_a_migrated_fact_whose_evidence_is_only_linked_is_listed_for_review(app):
    core, context = app
    source = capture(core, context, "[System note: The TEST-project CI service is Jenkins.]", key="hermes:TEST:legacy")
    migrated_version(core, context, jenkins(), evidence=[(source.ref, source.event["content"], None)])
    authority = owner(core, context)
    plan = authority.preview_origin([source.ref], origin="host_generated", reason="Host note")
    assert [(c["ref"], c["effect"]) for c in plan["claims"]] == [("TEST-old-C", "review")]
    assert plan["claims"][0]["versions"][0]["evidence_distribution"] == "selected_only"
    authority.apply_origin(plan)
    assert [(v.revision, v.state) for v in versions(core, context, "TEST-old-C")] == [(1, "active")]


def test_one_instant_written_two_ways_is_one_validity_window(app):
    core, context = app
    source_a = capture(core, context, "The TEST-project CI service is Jenkins.")
    span = dict(source_ref=source_a.ref, source_revision=source_a.revision, quote=source_a.event["content"])
    migrated_version(
        core,
        context,
        jenkins(valid_from="2026-09-01T12:00:00.000000Z", evidence_spans=[span]),
        evidence=[(source_a.ref, source_a.event["content"], None)],
    )
    source_b = capture(core, context, "The TEST-project CI service is CircleCI.")
    core.clock.now = "2026-09-06T12:00:00Z"
    later = service(core, context, source_b, "CircleCI")
    before = versions(core, context, later.ref)
    assert before[0].valid_from != before[-1].valid_from
    authority = owner(core, context)
    plan = authority.preview_origin([source_a.ref, source_b.ref], origin="host_generated", reason="Host rows")
    assert [c["effect"] for c in plan["claims"]] == ["retract"]
    authority.apply_origin(plan)
    after = versions(core, context, later.ref)
    for instant in ("2026-09-01T12:00:00Z", "2026-09-06T12:00:00Z", "2027-01-01T00:00:00Z"):
        assert select_effective(after, instant) is None
        assert select_effective(after, instant, as_of=True) is None
