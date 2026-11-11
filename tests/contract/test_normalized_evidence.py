"""Normalized evidence contracts over synthetic sources and deterministic verdicts."""

import json
from copy import deepcopy

import pytest
from scope_recall.contracts import ContractError
from scope_recall.core.evidence_support import using_support
from scope_recall.core.semantic_proposals import prepare_support

from tests.contract.test_claims import accept, app, capture, draft


def normalized(source, **changes):
    proposal = draft(source, subject="TEST-project", predicate="database", value="PostgreSQL")
    proposal["evidence_spans"][0].update(start=0, end=len(source.event["content"]))
    proposal.update(changes)
    return proposal


def envelope(proposal):
    span = proposal["evidence_spans"][0]
    return dict(
        protocol_version="1.1",
        source_refs=[f"{span['source_ref']}@{span['source_revision']}"],
        claim_proposals=[proposal],
        resume_proposals=[],
        reference_proposals=[],
    )


def judge(messages, **kwargs):
    assert json.loads(messages[-1]["content"])["proposal"]["value_text"] == "PostgreSQL"
    return json.dumps(dict(support="supported", action="new", target_ref=None, target_revision=None))


def test_normalized_fields_need_a_source_bound_support_verdict(app):
    core, ctx = app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    proposal = normalized(source)
    verdicts = prepare_support(envelope(proposal), (source,), judge)
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    assert receipt.items[0].state == "active"
    stored = core.current_claim(ctx, receipt.items[0].ref)
    assert stored.payload["value_text"] == "PostgreSQL"
    assert stored.payload["evidence_spans"][0]["start"] == 0


def test_unavailable_support_keeps_normalized_proposal_as_candidate(app):
    core, ctx = app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    receipt = accept(core, ctx, normalized(source))
    assert receipt.items[0].state == "proposed"


def test_offsets_cannot_be_repaired_to_another_occurrence(app):
    core, ctx = app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    proposal = normalized(source)
    proposal["evidence_spans"][0]["start"] = 1
    with pytest.raises(ContractError) as error:
        accept(core, ctx, proposal)
    assert error.value.field == "evidence_position"


@pytest.mark.parametrize(
    "text",
    [
        "TEST-project 的数据库不采用 postgres。",
        "如果迁移成功，TEST-project 的数据库采用 postgres。",
        "同事说 TEST-project 的数据库采用 postgres。",
    ],
)
def test_semantic_support_cannot_override_core_authority(app, text):
    core, ctx = app
    source = capture(core, ctx, text)
    proposal = normalized(source)
    with using_support(prepare_support(envelope(proposal), (source,), judge)):
        receipt = accept(core, ctx, proposal)
    assert receipt.items[0].state == "proposed"


def test_changed_proposal_cannot_reuse_a_support_receipt(app):
    core, ctx = app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    proposal = normalized(source)
    verdicts = prepare_support(envelope(proposal), (source,), judge)
    changed = deepcopy(proposal)
    changed["value_text"] = "SQLite"
    with using_support(verdicts):
        receipt = accept(core, ctx, changed)
    assert receipt.items[0].state == "proposed"


def test_one_verified_owner_statement_promotes_without_a_second_witness(app):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal

    core, ctx = app
    owner = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-owner"))
    source = capture(core, owner, "我决定采用 postgres。")
    proposal = normalized(source, subject="user")
    with using_support(prepare_support(envelope(proposal), (source,), judge)):
        receipt = accept(core, owner, proposal)
    assert receipt.items[0].state == "active"
    assert core.current_claim(owner, receipt.items[0].ref).payload["subject"] == "principal:TEST-owner"


def test_support_failure_preserves_candidates_and_stops_later_requests(app):
    core, ctx = app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    proposal = normalized(source)
    value = envelope(proposal)
    value["claim_proposals"].append(dict(proposal, predicate="storage"))
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("synthetic unavailable route")

    verdicts = prepare_support(value, (source,), unavailable)
    assert len(calls) == 1
    assert all(v.state.startswith("unavailable") for v in verdicts)
    with using_support(verdicts):
        receipt = accept(core, ctx, *value["claim_proposals"])
    assert all(item.state == "proposed" for item in receipt.items)


def test_normalized_slot_lookup_and_model_update_preserve_history(app):
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    first = capture(core, ctx, "TEST-project database PostgreSQL。", when="2026-09-05T12:00:00Z")
    old = accept(core, ctx, draft(first, subject="TEST-project", predicate="database", value="PostgreSQL"))
    source = capture(core, ctx, "TEST-project 的数据库采用 sqlite。", when="2026-09-06T12:00:00Z")
    proposal = normalized(source, subject="test-project", value_text="SQLite")

    def update(messages, **kwargs):
        candidates = json.loads(messages[-1]["content"])["candidates"]
        target = next(c for c in candidates if c["ref"] == old.items[0].ref)
        return json.dumps(
            dict(support="supported", action="update", target_ref=target["ref"], target_revision=target["revision"])
        )

    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), update)
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    assert receipt.items[0].ref == old.items[0].ref
    assert receipt.items[0].revision == 2
    assert core.current_claim(ctx, old.items[0].ref).payload["value_text"] == "SQLite"
    assert core.claim_history(ctx, old.items[0].ref)[0].payload["value_text"] == "PostgreSQL"


def test_normalized_event_time_is_canonical_utc_and_keeps_semantic_evidence(app):
    from scope_recall.core.worker_consolidation import decode_consolidation_result

    core, ctx = app
    source = capture(core, ctx, "TEST-project chose postgres at 2026-09-05 23:00 in New York.")
    proposal = normalized(source, valid_from="2026-09-05T23:00:00-04:00", occurred_at="2026-09-05T23:00:00-04:00")
    value = decode_consolidation_result(json.dumps(envelope(proposal)), (source,))
    proposal = value["claim_proposals"][0]
    with using_support(prepare_support(value, (source,), judge)):
        receipt = accept(core, ctx, proposal)
    saved = core.current_claim(ctx, receipt.items[0].ref)
    assert saved.payload["occurred_at"] == "2026-09-06T03:00:00Z"
    assert saved.valid_from.startswith("2026-09-06T03:00:00")


def test_each_support_request_rechecks_the_outbound_guard(app):
    from scope_recall.core import private

    core, ctx = app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    proposal = normalized(source)
    checks, calls = [], []

    def blocked():
        checks.append(True)
        return len(checks) > 1

    def never_send(*args, **kwargs):
        calls.append(True)
        return judge(*args, **kwargs)

    with private.using_outbound_check(blocked), pytest.raises(ContractError) as error:
        prepare_support(envelope(proposal), (source,), never_send)
    assert error.value.field == "private_mode"
    assert not calls


@pytest.mark.parametrize("error_type,detail", [("http_status", "401"), ("model_unavailable", None)])
def test_support_account_refusal_stops_the_worker_batch(app, error_type, detail):
    from dataclasses import replace

    from scope_recall.core.worker_consolidation import build_consolidation_model

    core, ctx = app
    capture(core, ctx, "TEST-project selected postgres.", key="TEST-support/first")
    capture(
        core,
        replace(ctx, session_id="TEST-other-session"),
        "TEST-project selected postgres again.",
        key="TEST-support/second",
    )

    class Refusal(RuntimeError):
        pass

    Refusal.error_type = error_type
    Refusal.detail = detail

    class Port:
        def __init__(self):
            self.extractions, self.judgements = 0, 0

        def propose(self, messages, **kwargs):
            body = json.loads(messages[-1]["content"])
            if "proposal" in body:
                self.judgements += 1
                raise Refusal()
            self.extractions += 1
            source = body["sources"][0]
            claim = dict(
                kind="decision",
                subject="TEST-project",
                predicate="database",
                value_text="PostgreSQL",
                conditions=[],
                statement_kind="assertion",
                valid_from=None,
                valid_to=None,
                evidence_spans=[
                    dict(
                        source_ref=source["source_ref"],
                        source_revision=source["source_revision"],
                        quote=source["content"],
                        start=0,
                        end=len(source["content"]),
                    )
                ],
            )
            return json.dumps(
                dict(
                    protocol_version="1.1",
                    source_refs=body["source_refs"],
                    claim_proposals=[claim],
                    resume_proposals=[],
                    reference_proposals=[],
                )
            )

    port = Port()
    receipt = core.drain_worker(ctx, max_items=8, remaining_seconds=30, consolidation=build_consolidation_model(port))
    assert port.extractions == port.judgements == 1
    assert "consolidate" in receipt.unavailable_work_types
    with core.storage.read(ctx) as tx:
        conn = tx._check()
        assert conn.execute("SELECT count(*) FROM claims").fetchone()[0] == 1
        pending = conn.execute(
            "SELECT state,attempt FROM work_items WHERE work_type='consolidate' AND state='pending'"
        ).fetchall()
        assert len(pending) == 1
        assert pending[0][1] == 0


def test_merge_lookup_filters_before_limiting_unrelated_candidates(app):
    from scope_recall.core.semantic_proposals import merge_candidates

    core, ctx = app
    source = capture(core, ctx, "TEST-project 的数据库采用 postgres。")
    target = accept(core, ctx, draft(source, predicate="database", value="postgres"))
    with core.storage.write(ctx, remaining_seconds=30) as tx:
        conn = tx._check(write=True)
        original = {
            table: dict(conn.execute(f"SELECT * FROM {table} WHERE claim_id=?", (target.items[0].ref,)).fetchone())
            for table in ("claims", "claim_versions")
        }
        for index in range(205):
            ref = target.items[0].ref[:-1] + "!" + str(index)
            for table, template in original.items():
                row = dict(template, claim_id=ref)
                if table == "claims":
                    row.update(subject=f"OTHER-{index}", slot_key=f"TEST-unrelated-{index}")
                else:
                    payload = json.loads(row["payload_json"])
                    payload["subject"] = f"OTHER-{index}"
                    row.update(payload_json=json.dumps(payload), state="proposed")
                conn.execute(
                    f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})", tuple(row.values())
                )
        assert target.items[0].ref not in tx.claims.list_refs(kind="decision", limit=200)
    with core.storage.read(ctx) as tx:
        choices = merge_candidates(
            tx, normalized(source, subject="test-project"), "TEST-scope", now=core.clock.utc_now()
        )
    assert [item["ref"] for item in choices] == [target.items[0].ref]


def test_semantic_duplicate_keeps_the_stored_wording_and_revision(app):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    ctx = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-owner"))
    first = capture(core, ctx, "TEST-project database postgres.", when="2026-09-05T12:00:00Z")
    old = accept(core, ctx, draft(first, predicate="database", value="postgres"))
    source = capture(core, ctx, "TEST-project selected PostgreSQL.", when="2026-09-06T12:00:00Z")
    proposal = normalized(source)

    def duplicate(messages, **kwargs):
        candidates = json.loads(messages[-1]["content"])["candidates"]
        assert [candidate["ref"] for candidate in candidates] == [old.items[0].ref]
        target = next(candidate for candidate in candidates if candidate["payload"]["value_text"] == "postgres")
        return json.dumps(
            dict(support="supported", action="duplicate", target_ref=target["ref"], target_revision=target["revision"])
        )

    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), duplicate)
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    assert receipt.items[0].disposition == "duplicate"
    assert receipt.items[0].revision == 1
    assert core.current_claim(ctx, old.items[0].ref).payload["value_text"] == "postgres"
    assert len(core.claim_history(ctx, old.items[0].ref)) == 1


def test_merge_lookup_resolves_a_normalized_verified_alias(app):
    from scope_recall.core.semantic_proposals import merge_candidates

    core, ctx = app
    first = capture(core, ctx, "TEST-project database postgres.")
    old = accept(core, ctx, draft(first, predicate="database", value="postgres"))
    identity_source = capture(core, ctx, "TEST-project project name is TEST-oldname.")
    identity = accept(
        core,
        ctx,
        draft(identity_source, kind="fact", statement_kind="assertion", predicate="project name", value="TEST-oldname"),
    )
    rename = capture(core, ctx, "TEST-project is now called Orchard.")
    alias = draft(
        rename,
        kind="alias",
        subject="TEST-project",
        predicate="alias",
        value="Orchard",
        alias=dict(name="Orchard", target_ref=identity.items[0].ref, scope_description="TEST-project"),
    )
    assert accept(core, ctx, alias).items[0].state == "active"
    with core.storage.read(ctx) as tx:
        choices = merge_candidates(tx, normalized(first, subject="orchard"), "TEST-scope", now=core.clock.utc_now())
    assert [item["ref"] for item in choices] == [old.items[0].ref]


def test_support_and_merge_inputs_keep_principal_authority_keys_local(app):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    principal = "principal:TEST-owner"
    owner = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", principal))
    first = capture(core, owner, "我决定采用 postgres。")
    original = normalized(first, subject="user")
    with using_support(prepare_support(envelope(original), (first,), judge)):
        saved = accept(core, owner, original)
    second = capture(core, owner, "我决定继续采用 postgres。")
    proposal = normalized(second, subject="user")

    def safe_judge(messages, **kwargs):
        assert principal not in json.dumps(messages)
        body = json.loads(messages[-1]["content"])
        assert body["candidates"][0]["payload"]["subject"] == "current_user"
        return json.dumps(
            dict(
                support="supported",
                action="duplicate",
                target_ref=saved.items[0].ref,
                target_revision=saved.items[0].revision,
            )
        )

    verdicts = prepare_publication(core.storage, owner, envelope(proposal), (second,), safe_judge)
    with using_support(verdicts):
        receipt = accept(core, owner, proposal)
    assert receipt.items[0].disposition == "duplicate"


@pytest.mark.parametrize("period", ["expired", "future", "microsecond"])
def test_merge_alias_must_apply_at_publication_time(app, period):
    from scope_recall.core.claims import Qualification
    from scope_recall.core.semantic_proposals import merge_candidates

    core, ctx = app
    source = capture(core, ctx, "TEST-project database postgres.")
    target = accept(core, ctx, draft(source, predicate="database", value="postgres")).items[0]
    relation = draft(
        source,
        kind="alias",
        predicate="alias",
        value="Orchard",
        alias=dict(name="Orchard", target_ref=target.ref, scope_description="TEST-project"),
        valid_from="2026-09-07T00:00:00Z" if period == "future" else "2026-09-01T00:00:00Z",
        valid_to={"expired": "2026-09-06T12:00:00Z", "microsecond": "2026-09-06T12:00:00.000001Z"}.get(period),
    )
    with core.storage.write(ctx) as tx:
        tx.claims.append(
            "TEST-scope",
            relation,
            Qualification("active", "direct_report", "synthetic_relation"),
            recorded_at=core.clock.utc_now(),
        )
        choices = merge_candidates(tx, normalized(source, subject="orchard"), "TEST-scope", now=core.clock.utc_now())
    assert [c["ref"] for c in choices] == ([target.ref] if period == "microsecond" else [])


def test_merge_alias_is_rechecked_if_it_expires_during_judgement(app):
    from scope_recall.core.claims import Qualification
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    original = capture(core, ctx, "TEST-project database postgres.")
    target = accept(core, ctx, draft(original, predicate="database", value="postgres")).items[0]
    relation = draft(
        original,
        kind="alias",
        predicate="alias",
        value="Orchard",
        alias=dict(name="Orchard", target_ref=target.ref, scope_description="TEST-project"),
        valid_to="2026-09-07T00:00:00Z",
    )
    with core.storage.write(ctx) as tx:
        tx.claims.append(
            "TEST-scope",
            relation,
            Qualification("active", "direct_report", "synthetic_relation"),
            recorded_at=core.clock.utc_now(),
        )
    source = capture(core, ctx, "Orchard database postgres.")
    proposal = normalized(source, subject="Orchard", value_text="postgres")

    def expires(messages, **kwargs):
        assert json.loads(messages[-1]["content"])["candidates"][0]["ref"] == target.ref
        core.clock.now = "2026-09-07T00:00:00Z"
        return json.dumps(dict(support="supported", action="update", target_ref=target.ref, target_revision=1))

    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), expires, clock=core.clock)
    with using_support(verdicts), pytest.raises(ContractError) as error:
        accept(core, ctx, proposal)
    assert (error.value.code, error.value.field) == ("VERSION_CONFLICT", "merge_target")
    assert len(core.claim_history(ctx, target.ref)) == 1


def restated(source, value, **labels):
    """A located proposal of ``value`` under the kind, subject and relation one extraction gave it."""
    proposal = draft(source, value, **labels)
    proposal["evidence_spans"][0].update(start=0, end=len(source.event["content"]))
    return proposal


@pytest.mark.parametrize("kind,predicate", [("fact", "launch_date"), ("decision", "launch date")])
@pytest.mark.parametrize("speaker", ["TEST-B", None, "TEST-A"])
@pytest.mark.parametrize("owner_proof", [False, True])
def test_verified_import_preserves_speaker_when_revising_a_human_decision(app, kind, predicate, speaker, owner_proof):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    if owner_proof:
        ctx = replace(
            ctx,
            owner_resolver=lambda entry, principal, memo: (
                "TEST-owner" if principal and principal.get("principal_ref") == "principal:TEST-A" else None
            ),
        )
    owner = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-A"))
    first = capture(core, owner, "TEST-project launch date March 3.", when="2026-09-03T12:00:00Z")
    old = accept(core, owner, draft(first, "March 3", predicate="launch date")).items[0]
    principal = (
        TrustedSourcePrincipal("human", "verified", f"principal:{speaker}")
        if speaker
        else TrustedSourcePrincipal("human", "unresolved")
    )
    source = capture(
        core,
        replace(ctx, source_principal=principal),
        "TEST-project launch date March 17.",
        origin="imported",
        attested=True,
        source_original_origin="human_direct",
        when="2026-09-04T12:00:00Z",
    )
    proposal = restated(source, "March 17", kind=kind, predicate=predicate)
    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), revise_launch_candidate([]))
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    expected = "active" if speaker == "TEST-A" else "proposed"
    assert (receipt.items[0].ref, receipt.items[0].state) == (old.ref, expected)
    assert core.current_claim(ctx, old.ref).payload["value_text"] == ("March 17" if speaker == "TEST-A" else "March 3")
    if speaker:
        with core.storage.read(ctx) as tx:
            from scope_recall.core.claim_normalization import human_owner

            assert human_owner(tx.claims.roots([f"{source.ref}@{source.revision}"])) == f"principal:{speaker}"


def decoded_date_proposal(source, value, **labels):
    """Pass a synthetic extraction through the production normalization boundary."""
    from scope_recall.core.worker_consolidation import decode_consolidation_result

    proposal = restated(source, value, **labels)
    return decode_consolidation_result(json.dumps(envelope(proposal)), (source,))["claim_proposals"][0]


def _date_update(target, value):
    """A supported update to the one reviewed synthetic predecessor, checked by reference and content."""

    def decide(messages, **kwargs):
        candidates = json.loads(messages[-1]["content"])["candidates"]
        assert [(c["ref"], c["payload"]["value_text"]) for c in candidates] == [(target.ref, value)]
        return json.dumps(
            dict(support="supported", action="update", target_ref=target.ref, target_revision=target.revision)
        )

    return decide


def _supported_date_insert(core, ctx, source, proposal):
    """Persist a synthetic initial assertion without a merge target."""
    with using_support(
        prepare_support(
            envelope(proposal),
            (source,),
            lambda *a, **k: json.dumps(dict(support="supported", action="new", target_ref=None, target_revision=None)),
        )
    ):
        return accept(core, ctx, proposal).items[0]


@pytest.mark.parametrize(
    "old_value,value,text",
    [
        ("2026-11-10", "2026-11-05", "TEST-project launch date corrected to 2026-11-05."),
        ("2026年11月10日", "2026年11月5日", "TEST-project 上线日期更正为2026年11月5日。"),
        ("2026-11-10", "2026-11-05", "TEST-project 从2026-11-11起上线日期改为2026-11-05。"),
    ],
)
@pytest.mark.parametrize("offset", ["Z", "+08:00", "-08:00", "+14:00", "-12:00"])
def test_date_valued_correction_uses_source_order_and_preserves_validity(app, old_value, value, text, offset):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal
    from scope_recall.core.claims import canonical_time
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    ctx = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-date-owner"))
    core.clock.now = "2026-11-11T12:00:00Z"
    first = capture(core, ctx, f"TEST-project launch date {old_value}.", when="2026-11-10T12:00:00Z")
    start = f"2026-11-05T00:00:00{offset}"
    # Include the UTC calendar day too so grounding and timezone ordering are exercised independently.
    text += f" UTC record {canonical_time(start)[:10]}."
    # Capture before inserting the predecessor so the literal capture shortcut cannot bypass model publication.
    source = capture(core, ctx, text, when=core.clock.now)
    stored = restated(first, old_value, kind="fact", predicate="launch date", valid_from="2026-11-10T00:00:00Z")
    old = _supported_date_insert(core, ctx, first, stored)
    proposal = decoded_date_proposal(source, value, predicate="launch_date", valid_from=start)
    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), _date_update(old, old_value))
    with using_support(verdicts):
        result = accept(core, ctx, proposal).items[0]
    head = core.current_claim(ctx, old.ref)
    assert (result.ref, result.state, head.payload["value_text"]) == (old.ref, "active", value)
    assert head.valid_from == canonical_time(start)
    assert head.reason != "late_historical_evidence"


@pytest.mark.parametrize(
    "text",
    [
        "On 2027-01-01, TEST-policy cutoff will change to 2027-01-01.",
        "Beginning 2027-01-01, TEST-policy cutoff will change to 2027-01-01.",
    ],
)
def test_scheduled_date_valued_rule_keeps_old_value_until_its_start(app, text):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal
    from scope_recall.core.claims import canonical_time
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    ctx = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-date-owner"))
    core.clock.now = "2026-11-11T12:00:00Z"
    first = capture(core, ctx, "TEST-policy cutoff is 2026-12-01.", when="2026-11-10T12:00:00Z")
    old = _supported_date_insert(
        core, ctx, first, restated(first, "2026-12-01", subject="TEST-policy", predicate="cutoff", valid_from=None)
    )
    source = capture(core, ctx, text, when=core.clock.now)
    proposal = decoded_date_proposal(
        source, "2027-01-01", subject="TEST-policy", predicate="cutoff", valid_from="2027-01-01T00:00:00Z"
    )
    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), _date_update(old, "2026-12-01"))
    with using_support(verdicts):
        item = accept(core, ctx, proposal).items[0]
    assert (item.ref, item.state, item.revision) == (old.ref, "active", 2)
    assert core.claim_history(ctx, old.ref)[-1].valid_from == canonical_time("2027-01-01T00:00:00Z")
    assert core.current_claim(ctx, old.ref).payload["value_text"] == "2026-12-01"
    core.clock.now = "2027-01-01T00:00:00Z"
    assert core.current_claim(ctx, old.ref).payload["value_text"] == "2027-01-01"


def test_future_date_value_waits_for_its_copied_start(app):
    from scope_recall.core.claims import canonical_time

    core, ctx = app
    core.clock.now = "2026-11-11T12:00:00Z"
    source = capture(core, ctx, "TEST-project 下个月20号上线，日期为2026-12-20。", when=core.clock.now)
    proposal = decoded_date_proposal(
        source, "2026-12-20", kind="fact", predicate="launch date", valid_from="2026-12-20T00:00:00Z"
    )
    item = _supported_date_insert(core, ctx, source, proposal)
    assert item.state == "active"
    assert core.claim_history(ctx, item.ref)[-1].valid_from == canonical_time("2026-12-20T00:00:00Z")
    assert core.current_claim(ctx, item.ref) is None
    core.clock.now = "2026-12-20T00:00:00Z"
    assert core.current_claim(ctx, item.ref).payload["value_text"] == "2026-12-20"


def test_future_date_valued_rule_without_start_remains_proposed(app):
    core, ctx = app
    source = capture(core, ctx, "Starting next month, TEST-policy cutoff will change to 2027-01-01.")
    proposal = decoded_date_proposal(source, "2027-01-01", subject="TEST-policy", predicate="cutoff", valid_from=None)
    item = _supported_date_insert(core, ctx, source, proposal)
    assert item.state == "proposed"
    assert core.claim_history(ctx, item.ref)[-1].reason == "future_start_unknown"


@pytest.mark.parametrize(
    "value,text,start",
    [
        ("柏林", "TEST-project：我2020年住在柏林。", "2020-01-01T00:00:00Z"),
        ("2021-01-01", "In 2020, TEST-project cutoff was 2021-01-01.", "2020-01-01T00:00:00Z"),
        ("November 5", "TEST-project cutoff was November 5.", "2026-11-05T00:00:00Z"),
    ],
)
def test_non_date_different_day_and_partial_date_evidence_stays_historical(app, value, text, start):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    ctx = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-date-owner"))
    core.clock.now = "2026-11-11T12:00:00Z"
    first = capture(core, ctx, "TEST-project cutoff Boston.", when="2026-11-10T12:00:00Z")
    old = _supported_date_insert(
        core, ctx, first, restated(first, "Boston", predicate="cutoff", valid_from="2026-11-10T00:00:00Z")
    )
    source = capture(core, ctx, text, when=core.clock.now)
    proposal = decoded_date_proposal(source, value, predicate="cutoff", valid_from=start)
    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), _date_update(old, "Boston"))
    with using_support(verdicts):
        item = accept(core, ctx, proposal).items[0]
    assert (item.ref, item.state) == (old.ref, "active")
    assert core.current_claim(ctx, old.ref).payload["value_text"] == "Boston"
    assert core.claim_history(ctx, old.ref)[-1].reason == "late_historical_evidence"


@pytest.mark.parametrize(
    "value,start",
    [
        ("11-05", "2026-11-05T00:00:00Z"),
        ("2026年11月5日", "2026-11-05T00:00:00+14:00"),
        ("November 5, 2026", "2026-11-05T00:00:00Z"),
    ],
)
def test_model_date_validity_is_only_converted_to_utc(app, value, start):
    from scope_recall.contracts import utc_instant

    core, ctx = app
    source = capture(core, ctx, f"TEST-project launch date {value}.")
    proposal = decoded_date_proposal(source, value, valid_from=start)
    assert proposal["valid_from"] == utc_instant(start)


def test_empty_frame_does_not_nominate_merge_candidates(app):
    from scope_recall.core.semantic_proposals import merge_candidates

    core, ctx = app
    source = capture(core, ctx, "TEST punctuation only labels.")
    from scope_recall.core.claims import Qualification

    with core.storage.write(ctx) as tx:
        tx.claims.append(
            "TEST-scope",
            restated(source, "blue", subject="😀", predicate="!!!"),
            Qualification("active", "direct_report", "synthetic"),
            recorded_at=core.clock.utc_now(),
        )
    with core.storage.read(ctx) as tx:
        proposal = restated(source, "red", subject="🚀", predicate="???")
        assert merge_candidates(tx, proposal, "TEST-scope", now=core.clock.utc_now()) == ()


def revise_launch_candidate(shown):
    """Select the synthetic launch's preceding value by content and record all nominated references."""

    def decide(messages, **kwargs):
        body = json.loads(messages[-1]["content"])
        candidates = body["candidates"]
        shown.append([candidate["ref"] for candidate in candidates])
        previous = {
            "March 17": "March 3",
            "March 17, 2027": "March 3, 2027",
            "March 24, 2027": "March 17, 2027",
            "2026-11-05": "2026-11-10",
        }
        target = next(
            (c for c in candidates if c["payload"]["value_text"] == previous[body["proposal"]["value_text"]]),
            dict(ref=None, revision=None),
        )
        assert not candidates or target["ref"] is not None
        return json.dumps(
            dict(
                support="supported",
                action="update" if candidates else "new",
                target_ref=target["ref"],
                target_revision=target["revision"],
            )
        )

    return decide


def test_exact_labels_survive_candidate_truncation_and_paging_keeps_id_order(app):
    from scope_recall.core.claims import Qualification
    from scope_recall.core.semantic_proposals import merge_candidates

    core, ctx = app
    source = capture(core, ctx, "TEST-project launch date March 3.")
    with core.storage.write(ctx) as tx:
        versions = [
            tx.claims.append(
                "TEST-scope",
                restated(source, "March 3", subject="TEST" + "-" * index + "project"),
                Qualification("active", "direct_report", "synthetic"),
                recorded_at=core.clock.utc_now(),
            )
            for index in range(1, 22)
        ]
    exact = max(versions, key=lambda v: v.ref)
    proposal = dict(exact.payload, value_text="March 17")
    with core.storage.read(ctx) as tx:
        candidates = merge_candidates(tx, proposal, "TEST-scope", now=core.clock.utc_now())
        assert len(candidates) == 16
        assert candidates[0]["ref"] == exact.ref
        first, cursor = merge_candidates(tx, proposal, "TEST-scope", now=core.clock.utc_now(), paging=(None, 16))
        second, _ = merge_candidates(tx, proposal, "TEST-scope", now=core.clock.utc_now(), paging=(cursor, 16))
        assert [c["ref"] for c in (*first, *second)] == sorted(v.ref for v in versions)


def test_a_frame_reads_labels_by_their_letters_and_digits_in_order():
    from scope_recall.core.fact_identity import fact_frame

    same = [
        ("Atlas beta", "launch date"),
        ("atlas BETA", "launch_date"),
        ("Atlas beta", "launch-date"),
        ("Ａｔｌａｓ beta", "launch　date"),
        ("Atlas", "beta launch date"),
        ("Atlas beta launch", "date"),
        ("Atlas beta launch", "launch_date"),
    ]
    apart = [
        ("Atlas beta", "date launch"),
        ("Atlas beta", "launch owner"),
        ("Atlas", "launch date"),
        ("Atlas beta 2", "launch date"),
    ]
    assert len({fact_frame(*labels) for labels in same}) == 1
    assert len({fact_frame(*labels) for labels in (same[0], *apart)}) == len(apart) + 1
    assert fact_frame("项目甲", "发布 日期") == fact_frame("项目甲", "发布日期") != fact_frame("项目甲", "日期发布")


@pytest.mark.parametrize(
    "stored,labels",
    [
        # The relation written with another separator.
        ({}, dict(predicate="launch_date")),
        # The same words divided between subject and relation at another place.
        ({}, dict(subject="TEST-project launch", predicate="date")),
        # The relation's first word written into the subject as well.
        ({}, dict(subject="TEST-project launch", predicate="launch_date")),
        # One statement read as a fact by one extraction and as a decision by the next.
        (dict(kind="fact", statement_kind="assertion"), dict(predicate="launch_date")),
    ],
)
def test_a_changed_value_revises_its_claim_however_its_labels_were_written(app, stored, labels):
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    first = capture(core, ctx, "TEST-project launch date March 3.", when="2026-09-05T12:00:00Z")
    old = accept(core, ctx, draft(first, "March 3", predicate="launch date", **stored)).items[0]
    assert old.state == "active"
    source = capture(core, ctx, "TEST-project launch date March 17.", when="2026-09-06T12:00:00Z")
    proposal = restated(source, "March 17", **labels)
    shown = []
    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), revise_launch_candidate(shown))
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    assert shown == [[old.ref]]
    assert (receipt.items[0].ref, receipt.items[0].revision, receipt.items[0].state) == (old.ref, 2, "active")
    head, original = core.current_claim(ctx, old.ref), core.claim_history(ctx, old.ref)[0]
    assert (original.payload["value_text"], head.payload["value_text"]) == ("March 3", "March 17")
    # The claim keeps the kind, subject and relation it was filed under.
    assert all(head.payload[label] == original.payload[label] for label in ("kind", "subject", "predicate"))


def test_a_value_said_again_under_other_labels_adds_no_second_claim(app):
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    first = capture(core, ctx, "TEST-project launch date March 3.", when="2026-09-05T12:00:00Z")
    stored = draft(first, "March 3", kind="fact", statement_kind="assertion", predicate="launch date")
    old = accept(core, ctx, stored).items[0]
    source = capture(core, ctx, "TEST-project launch date is still March 3.", when="2026-09-06T12:00:00Z")
    proposal = restated(source, "March 3", predicate="launch_date")

    def duplicate(messages, **kwargs):
        candidates = json.loads(messages[-1]["content"])["candidates"]
        assert [candidate["ref"] for candidate in candidates] == [old.ref]
        target = next(candidate for candidate in candidates if candidate["payload"]["value_text"] == "March 3")
        return json.dumps(
            dict(support="supported", action="duplicate", target_ref=target["ref"], target_revision=target["revision"])
        )

    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (source,), duplicate)
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    assert (receipt.items[0].ref, receipt.items[0].revision, receipt.items[0].disposition) == (old.ref, 1, "duplicate")
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT count(*) FROM claims").fetchone()[0] == 1


def test_a_value_only_a_person_may_state_is_not_revised_from_a_tool_observation(app):
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    first = capture(core, ctx, "TEST-project launch date March 3.", when="2026-09-05T12:00:00Z")
    old = accept(core, ctx, draft(first, "March 3", predicate="launch date")).items[0]
    assert (old.state, core.current_claim(ctx, old.ref).payload["kind"]) == ("active", "decision")
    observed = capture(
        core, ctx, "TEST-project launch date March 17.", origin="tool_observation", when="2026-09-06T12:00:00Z"
    )
    proposal = restated(observed, "March 17", kind="fact", statement_kind="assertion", predicate="launch_date")
    verdicts = prepare_publication(core.storage, ctx, envelope(proposal), (observed,), revise_launch_candidate([]))
    with using_support(verdicts):
        receipt = accept(core, ctx, proposal)
    assert (receipt.items[0].ref, receipt.items[0].state) == (old.ref, "proposed")
    assert core.current_claim(ctx, old.ref).payload["value_text"] == "March 3"


def test_slot_lookup_keeps_other_relations_word_orders_conditions_and_kinds_apart(app):
    from scope_recall.core.semantic_proposals import merge_candidates

    core, ctx = app
    source = capture(core, ctx, "TEST-project launch date March 3.")
    old = accept(core, ctx, draft(source, "March 3", predicate="launch date")).items[0]
    apart = (
        dict(predicate="launch owner"),
        dict(predicate="date launch"),
        dict(conditions=["TEST-condition"]),
        dict(kind="observation"),
        dict(kind="procedure"),
    )
    with core.storage.read(ctx) as tx:

        def found(**labels):
            proposal = restated(source, "March 3", **{"predicate": "launch_date", **labels})
            return [item["ref"] for item in merge_candidates(tx, proposal, "TEST-scope", now=core.clock.utc_now())]

        assert found() == [old.ref]
        assert [found(**labels) for labels in apart] == [[]] * len(apart)


def test_a_changed_plan_stays_one_claim_across_differently_labelled_extractions(app):
    """Three messages of one conversation, each consolidated before the next is said, under the kind, subject and
    relation an extraction model gave them on one recorded occasion."""
    from scope_recall.core.worker_consolidation import build_consolidation_model

    core, ctx = app
    extracted = {
        "We're launching the TEST-atlas beta on March 3, 2027.": dict(
            kind="fact",
            statement_kind="assertion",
            subject="TEST-atlas beta",
            predicate="launch date",
            value_text="March 3, 2027",
        ),
        "Change of plan: we're launching the TEST-atlas beta on March 17, 2027 instead.": dict(
            kind="decision",
            statement_kind="decision",
            subject="TEST-atlas beta",
            predicate="launch_date",
            value_text="March 17, 2027",
        ),
        "Change of plan: the TEST-atlas beta launch date is now March 24, 2027.": dict(
            kind="decision",
            statement_kind="decision",
            subject="TEST-atlas beta launch",
            predicate="launch_date",
            value_text="March 24, 2027",
        ),
    }

    class Port:
        def propose(self, messages, **kwargs):
            body = json.loads(messages[-1]["content"])
            if "proposal" in body:
                return revise_launch_candidate([])(messages)
            claims = [
                dict(
                    extracted[source["content"]],
                    conditions=[],
                    valid_from=None,
                    valid_to=None,
                    evidence_spans=[
                        dict(
                            source_ref=source["source_ref"],
                            source_revision=source["source_revision"],
                            quote=source["content"],
                            start=0,
                            end=len(source["content"]),
                        )
                    ],
                )
                for source in body["sources"]
            ]
            return json.dumps(
                dict(
                    protocol_version="1.1",
                    source_refs=body["source_refs"],
                    claim_proposals=claims,
                    resume_proposals=[],
                    reference_proposals=[],
                )
            )

    model = build_consolidation_model(Port())
    for day, message in enumerate(extracted, start=3):
        capture(core, ctx, message, when=f"2026-09-0{day}T12:00:00Z")
        core.drain_worker(ctx, max_items=8, remaining_seconds=30, consolidation=model)
    with core.storage.read(ctx) as tx:
        versions = (
            tx._check()
            .execute(
                "SELECT c.claim_id,v.revision,v.state,json_extract(v.payload_json,'$.value_text') FROM claims c "
                "JOIN claim_versions v USING(claim_id) WHERE c.kind IN ('fact','decision') ORDER BY c.rowid,v.revision"
            )
            .fetchall()
        )
    assert [tuple(row)[1:] for row in versions] == [
        (1, "active", "March 3, 2027"),
        (2, "active", "March 17, 2027"),
        (3, "active", "March 24, 2027"),
    ]
    assert len({row[0] for row in versions}) == 1


OWNER_AT_CODEX = "principal:TEST-owner-at-codex"
OWNER_AT_DESKTOP = "principal:TEST-owner-at-desktop"
ORIGINAL = "We're launching the Atlas beta on March 3."
CHANGE = "Change of plan: the Atlas beta launch date is now March 17."
CHANGE_QUOTE = "the Atlas beta launch date is now March 17"


def one_owner(entry_id, principal, memo):
    """A shared store's ownership proof: both routes are approved as the one owner's."""
    return "TEST-installation" if principal["principal_ref"] in {OWNER_AT_CODEX, OWNER_AT_DESKTOP} else None


def speaker(ctx, principal, session, proof=None):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal

    person = TrustedSourcePrincipal("human", "verified", principal)
    return replace(ctx, session_id=session, source_principal=person, owner_resolver=proof)


def launch(source, value, quote, **changes):
    """A positioned proposal for the Atlas beta's launch date, quoting ``quote`` from ``source``."""
    proposal = draft(
        source, value, subject="Atlas beta", predicate="launch_date", quote_override=quote, valid_from=None
    )
    start = source.event["content"].index(quote)
    proposal["evidence_spans"][0].update(start=start, end=start + len(quote))
    proposal.update(changes)
    return proposal


def distinct(messages, **kwargs):
    """Judge the synthetic launch relation by candidate content; ambiguous pairs remain separate."""
    body = json.loads(messages[-1]["content"])
    choices = [
        c
        for c in body["candidates"]
        if c["payload"]["subject"] == "Atlas beta"
        and c["payload"]["predicate"] in {"launch_date", "launch date"}
        and c["payload"]["value_text"] == "March 3"
    ]
    target = choices[0] if len(body["candidates"]) == len(choices) == 1 else None
    return json.dumps(
        dict(
            support="supported",
            action="update" if target else "new",
            target_ref=target["ref"] if target else None,
            target_revision=target["revision"] if target else None,
        )
    )


def publish(core, person, source, proposal):
    from scope_recall.core.semantic_proposals import prepare_publication

    with using_support(prepare_publication(core.storage, person, envelope(proposal), (source,), distinct)):
        return accept(core, person, proposal).items[0]


def told_through_codex(core, ctx, proof=None):
    """The owner tells Codex the launch date; returns the stored claim."""
    codex = speaker(ctx, OWNER_AT_CODEX, "TEST-codex-session", proof)
    source = capture(core, codex, ORIGINAL, when="2026-09-03T12:00:00Z")
    stored = publish(core, codex, source, launch(source, "March 3", ORIGINAL))
    assert (stored.revision, stored.state) == (1, "active")
    return stored


def changed_through(core, ctx, principal, session, proof=None, **changes):
    """The change of plan said later by ``principal``; returns what publishing its claim wrote."""
    person = speaker(ctx, principal, session, proof)
    source = capture(core, person, CHANGE, when="2026-09-05T12:00:00Z")
    return publish(core, person, source, launch(source, "March 17", CHANGE_QUOTE, **changes))


def test_the_owner_s_later_word_through_another_entry_replaces_the_value(app):
    core, ctx = app
    stored = told_through_codex(core, ctx, one_owner)

    changed = changed_through(core, ctx, OWNER_AT_DESKTOP, "TEST-desktop-session", one_owner)

    assert (changed.ref, changed.revision, changed.disposition, changed.state) == (stored.ref, 2, "inserted", "active")
    assert core.current_claim(ctx, stored.ref).payload["value_text"] == "March 17"
    history = core.claim_history(ctx, stored.ref)
    assert [(v.revision, v.payload["value_text"], v.replaces_revision) for v in history] == [
        (1, "March 3", None),
        (2, "March 17", 1),
    ]


def test_another_person_s_later_word_stays_a_proposal_beside_the_value(app):
    """The proof names the first speaker the owner and says nothing of the second: two people."""
    core, ctx = app

    def only_codex(entry_id, principal, memo):
        return "TEST-installation" if principal["principal_ref"] == OWNER_AT_CODEX else None

    stored = told_through_codex(core, ctx, only_codex)

    said = changed_through(core, ctx, "principal:TEST-visitor", "TEST-visitor-session", only_codex)

    assert (said.ref, said.disposition, said.state) == (stored.ref, "historical", "proposed")
    assert core.claim_history(ctx, stored.ref)[-1].reason == "different_source_principal"
    assert core.current_claim(ctx, stored.ref).payload["value_text"] == "March 3"


def test_two_routes_with_no_ownership_proof_stay_two_people(app):
    core, ctx = app
    stored = told_through_codex(core, ctx)

    said = changed_through(core, ctx, OWNER_AT_DESKTOP, "TEST-desktop-session")

    assert (said.ref, said.state) == (stored.ref, "proposed")
    assert core.current_claim(ctx, stored.ref).payload["value_text"] == "March 3"


def test_a_pass_outside_a_host_session_reads_one_owner_from_the_proof_it_is_given(app):
    """The worker's context carries no ownership proof of its own; its pass is given the store's."""
    from scope_recall.core.claim_normalization import using_owner_proof

    core, ctx = app
    stored = told_through_codex(core, ctx)

    with using_owner_proof(one_owner):
        changed = changed_through(core, ctx, OWNER_AT_DESKTOP, "TEST-desktop-session")

    assert (changed.ref, changed.revision, changed.state) == (stored.ref, 2, "active")
    assert core.current_claim(ctx, stored.ref).payload["value_text"] == "March 17"


@pytest.mark.parametrize("spelling", ["launch date", "Launch-Date", "LAUNCH_DATE", "launch  date"])
def test_a_relation_spelled_another_way_takes_the_slot_that_holds_it(app, spelling):
    core, ctx = app
    stored = told_through_codex(core, ctx)

    changed = changed_through(core, ctx, OWNER_AT_CODEX, "TEST-codex-session", predicate=spelling)

    assert (changed.ref, changed.revision, changed.state) == (stored.ref, 2, "active")
    current = core.current_claim(ctx, stored.ref)
    assert (current.payload["predicate"], current.payload["value_text"]) == ("launch_date", "March 17")
    with core.storage.read(ctx) as tx:
        assert tx._check().execute("SELECT count(*) FROM claims WHERE subject='Atlas beta'").fetchone()[0] == 1


def test_another_relation_of_the_same_subject_keeps_a_slot_of_its_own(app):
    core, ctx = app
    stored = told_through_codex(core, ctx)

    other = changed_through(core, ctx, OWNER_AT_CODEX, "TEST-codex-session", predicate="launch_window")

    assert other.ref != stored.ref and (other.revision, other.state) == (1, "active")
    assert core.current_claim(ctx, stored.ref).payload["value_text"] == "March 3"


def test_a_spelling_two_stored_slots_already_share_is_given_to_neither(app):
    """A store can hold one relation as two claims that differ only in spelling.  A third spelling of it names
    neither claim, and choosing one is left to the owner."""
    from scope_recall.core.claims import Qualification

    core, ctx = app
    stored = told_through_codex(core, ctx)
    codex = speaker(ctx, OWNER_AT_CODEX, "TEST-codex-session")
    source = capture(core, codex, "The Atlas beta launch date is March 10.", when="2026-09-04T12:00:00Z")
    twin = launch(source, "March 10", source.event["content"], predicate="launch date")
    with core.storage.write(codex) as tx:
        second = tx.claims.append(
            "TEST-scope",
            twin,
            Qualification("active", "direct_report", "synthetic_twin"),
            recorded_at="2026-09-04T12:00:00Z",
        )

    third = changed_through(core, ctx, OWNER_AT_CODEX, "TEST-codex-session", predicate="Launch-Date")

    assert third.ref not in {stored.ref, second.ref} and third.revision == 1
    assert core.current_claim(ctx, stored.ref).payload["value_text"] == "March 3"
    assert core.current_claim(ctx, second.ref).payload["value_text"] == "March 10"


def test_merge_lookup_nominates_the_slot_of_a_relation_spelled_another_way(app):
    from scope_recall.core.semantic_proposals import merge_candidates

    core, ctx = app
    stored = told_through_codex(core, ctx)
    codex = speaker(ctx, OWNER_AT_CODEX, "TEST-codex-session")
    source = capture(core, codex, CHANGE, when="2026-09-05T12:00:00Z")
    proposal = launch(source, "March 17", CHANGE_QUOTE, predicate="launch date")

    with core.storage.read(codex) as tx:
        choices = merge_candidates(tx, proposal, "TEST-scope", now=core.clock.utc_now())

    assert [(item["ref"], item["payload"]["predicate"]) for item in choices] == [(stored.ref, "launch_date")]


@pytest.mark.parametrize(
    "start,echo",
    [
        ("2026-11-05T00:00:00+08:00", True),
        ("2026-11-05T00:00:00-08:00", True),
        ("2026-11-05T00:00:00+14:00", True),
        ("2026-11-05T23:59:59-12:00", True),
        ("2026-11-03T00:00:00Z", False),
    ],
)
def test_date_echo_covers_any_timezone_but_excludes_two_days_earlier(start, echo):
    from scope_recall.core.claims import canonical_time
    from scope_recall.core.mutate import _historical_start

    proposal = dict(valid_from=start, value_text="2026-11-05")
    assert _historical_start(proposal) == (None if echo else canonical_time(start))


def test_later_date_echo_from_imports_without_source_times_does_not_prove_order(app):
    from dataclasses import replace

    from scope_recall.contracts import TrustedSourcePrincipal
    from scope_recall.core.semantic_proposals import prepare_publication

    core, ctx = app
    ctx = replace(ctx, source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-date-owner"))
    core.clock.now = "2026-12-01T12:00:00Z"
    first = capture(
        core,
        ctx,
        "TEST-project cutoff 2026-11-10.",
        origin="imported",
        when=None,
        attested=True,
        source_original_origin="human_direct",
    )
    old = _supported_date_insert(
        core, ctx, first, restated(first, "2026-11-10", predicate="cutoff", valid_from="2026-11-10T00:00:00Z")
    )
    source = capture(
        core,
        ctx,
        "TEST-project cutoff 2026-11-20.",
        origin="imported",
        when=None,
        attested=True,
        source_original_origin="human_direct",
    )
    proposal = restated(source, "2026-11-20", predicate="cutoff", valid_from="2026-11-20T00:00:00Z")
    with using_support(
        prepare_publication(core.storage, ctx, envelope(proposal), (source,), _date_update(old, "2026-11-10"))
    ):
        result = accept(core, ctx, proposal).items[0]
    assert result.state == "disputed"
    latest = core.claim_history(ctx, old.ref)[-1]
    assert latest.reason == "conflicting_evidence_order_unknown"
    assert latest.valid_from == "2026-11-20T00:00:00.000000+00:00"
