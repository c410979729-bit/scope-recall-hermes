"""Durable model revision retries remain bound to the current native user occurrence."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from scope_recall.contracts import ContractError, RequestMessageBinding, TrustedSourcePrincipal
from scope_recall.core import CoreConfig, MemoryCore, private
from scope_recall.core.backup_primitives import backup_sqlite
from scope_recall.core.management import MemoryManagement
from scope_recall.core.model_revision_storage import ModelRevisionReceipts
from scope_recall.core.restore import InstallationMaintenance
from scope_recall.maintenance.restore import apply_restore
from test_claims import accept, app, capture, draft, initial, revise_request  # noqa: F401


def bound(ctx, source, *, occurrence="TEST-native-uid", live=lambda: True):
    """Attest one synthetic host occurrence without serializing its runtime liveness."""
    proof = RequestMessageBinding(
        ctx.binding,
        ctx.entry_id,
        ctx.session_id,
        ctx.project_id,
        ctx.branch_id,
        source.scope_id,
        source.ref,
        source.revision,
        hashlib.sha256(source.event["content"].encode()).hexdigest(),
        occurrence,
        ctx.source_principal,
        live,
    )
    return replace(ctx, request_message=proof)


def counts(core, ctx):
    """Read truth counts and the epoch, including the distinct human audit table."""
    with core.storage.read(ctx) as tx:
        return tuple(
            tx._check().execute("SELECT count(*) FROM " + table).fetchone()[0]
            for table in ("claim_versions", "source_events", "management_actions", "model_revision_receipts")
        ) + (tx.memory_epoch(),)


@pytest.fixture
def revision(app):
    """Capture an explicit correction without the capture fast path applying it."""
    core, ctx = app
    principal = TrustedSourcePrincipal("human", "verified", "principal:TEST-owner")
    ctx = replace(ctx, source_principal=principal)
    item, _ = initial(core, ctx, kind="fact")
    source = capture(core, ctx, "Please correct TEST-project 配色: 银色。", when=None)
    return core, bound(ctx, source), revise_request(item, source, valid_from=None)


def test_exact_retry_returns_original_receipt(revision):
    """A retry returns the original version and epoch instead of failing its old CAS."""
    core, ctx, body = revision
    first = core.revise(ctx, body, request_id="TEST-request")
    after = counts(core, ctx)
    core.clock.now = "2026-09-07T12:00:00Z"
    assert core.revise(ctx, body, request_id="TEST-request") == first
    assert counts(core, ctx) == after


@pytest.mark.parametrize(
    "change",
    [
        {"new_value": "TEST different"},
        {"new_value": None},
        {"expected_revision": 2},
        {"conditions": ["TEST condition"]},
        {"valid_from": "2026-09-01T00:00:00Z"},
        {"target_ref": "claim-TEST-different"},
        {"source_evidence_refs": ["event-TEST-different@1"]},
    ],
)
def test_same_key_different_request_conflicts(revision, change):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    after = counts(core, ctx)
    with pytest.raises(ContractError, match="request_id"):
        core.revise(ctx, body | change, request_id="TEST-request")
    assert counts(core, ctx) == after


@pytest.mark.parametrize("identity", ["session", "principal", "entry", "scope", "branch", "expired"])
def test_revoked_or_changed_authority_cannot_recover_result(revision, identity):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    after = counts(core, ctx)
    changes = {
        "session": dict(session_id="TEST-other"),
        "principal": dict(source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-other")),
        "entry": dict(entry_id="TEST-other"),
        "scope": dict(allowed_scope_ids=frozenset()),
        "branch": dict(branch_id="TEST-other"),
        "expired": dict(request_message=replace(ctx.request_message, live=lambda: False)),
    }[identity]
    with pytest.raises(ContractError):
        core.revise(replace(ctx, **changes), body, request_id="TEST-request")
    assert counts(core, ctx) == after


def test_new_same_text_occurrence_does_not_replay(revision):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    source = core.source(ctx, ctx.request_message.source_ref, 1)
    # Even the same stored text and source cannot change its native occurrence in the key.
    other = bound(ctx, source, occurrence="TEST-another-native-uid")
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        core.revise(other, body, request_id="TEST-request")
    source = capture(core, replace(ctx, request_message=None), source.event["content"], when=None)
    other = bound(ctx, source, occurrence="TEST-third-native-uid")
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        core.revise(other, body | {"source_evidence_refs": [f"{source.ref}@1"]}, request_id="TEST-request")
    with pytest.raises(ContractError):
        core.revise(ctx, body, request_id="TEST-request")


@pytest.mark.parametrize("request_id", [None, "TEST-new-request"])
def test_without_same_explicit_id_original_cas_applies(revision, request_id):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        core.revise(ctx, body, request_id=request_id)


def test_reopen_reads_receipt_from_sqlite(revision):
    core, ctx, body = revision
    first = core.revise(ctx, body, request_id="TEST-request")
    reopened = MemoryCore(CoreConfig(ctx.binding), clock=core.clock)
    reopened.initialize()
    assert reopened.revise(ctx, body, request_id="TEST-request") == first
    assert counts(core, ctx) == counts(reopened, ctx)


@pytest.mark.parametrize("mode", ["full", "read-only"])
def test_private_mode_revokes_receipt_access(revision, mode):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    private.open_interval(core.storage, core.clock, ctx, target="session", session=ctx.session_id, mode=mode)
    after = counts(core, ctx)
    with pytest.raises(ContractError, match="private_mode"):
        core.revise(ctx, body, request_id="TEST-request")
    assert counts(core, ctx) == after


@pytest.mark.parametrize("target", ["claim", "source", "scope"])
def test_deletion_cannot_be_bypassed_by_receipt(revision, target):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    owner = MemoryManagement(core, InstallationMaintenance(replace(ctx, request_message=None)))
    if target == "scope":
        with core.storage.read(ctx) as tx:
            refs = [
                row[0]
                for row in tx._check().execute(
                    "SELECT DISTINCT event_id FROM source_events WHERE scope_id=?", (ctx.request_message.scope_id,)
                )
            ]
        preview = owner.preview(refs, mode="delete")
    else:
        refs = [body["target_ref"] if target == "claim" else ctx.request_message.source_ref]
        preview = owner.preview(refs, mode="delete")
    owner.apply(preview)
    after = counts(core, ctx)
    with pytest.raises(ContractError):
        core.revise(ctx, body, request_id="TEST-request")
    assert counts(core, ctx) == after


def test_later_revision_invalidates_old_receipt(revision):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    owner = MemoryManagement(core, InstallationMaintenance(replace(ctx, request_message=None)))
    owner.revise(body["target_ref"], expected_revision=2, value="TEST later value")
    after = counts(core, ctx)
    with pytest.raises(ContractError, match="expected_revision"):
        core.revise(ctx, body, request_id="TEST-request")
    assert counts(core, ctx) == after


@pytest.mark.parametrize("when", ["before_insert", "after_insert"])
def test_revision_and_receipt_roll_back_together(revision, monkeypatch, when):
    core, ctx, body = revision
    before = counts(core, ctx)
    record = ModelRevisionReceipts.record

    def fail(self, item, epoch):
        if when == "after_insert":
            record(self, item, epoch)
        raise RuntimeError("TEST receipt failure")

    with monkeypatch.context() as patch:
        patch.setattr(ModelRevisionReceipts, "record", fail)
        with pytest.raises(RuntimeError, match="TEST receipt failure"):
            core.revise(ctx, body, request_id="TEST-request")
    assert counts(core, ctx) == before
    assert core.current_claim(ctx, body["target_ref"]).revision == 1
    assert core.revise(ctx, body, request_id="TEST-request").items[0].revision == 2


@pytest.mark.parametrize("suffix", ["@01", "@+1", "@1 ", "@1@1"])
def test_malformed_reference_is_rejected_before_receipt_lookup(revision, suffix):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    token = ctx.request_message.source_ref + suffix
    with pytest.raises(ContractError, match="INPUT_INVALID"):
        core.revise(ctx, body | {"source_evidence_refs": [token]}, request_id="TEST-request")


def test_receipt_contains_metadata_only(revision):
    core, ctx, body = revision
    core.revise(ctx, body, request_id="TEST-request")
    with core.storage.read(ctx) as tx:
        stored = json.dumps(
            dict(tx._check().execute("SELECT * FROM model_revision_receipts").fetchone()), ensure_ascii=False
        )
    assert "银色" not in stored and "Please correct" not in stored and "TEST-native-uid" not in stored


def test_missing_request_id_keeps_original_cas_without_receipt(revision):
    core, ctx, body = revision
    core.revise(ctx, body)
    assert counts(core, ctx)[3] == 0
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        core.revise(ctx, body)


@pytest.mark.parametrize("request_id", ["", " " * 3, 1, "x" * 101])
def test_invalid_request_id_never_mutates(revision, request_id):
    core, ctx, body = revision
    before = counts(core, ctx)
    with pytest.raises(ContractError, match="request_id"):
        core.revise(ctx, body, request_id=request_id)
    assert counts(core, ctx) == before


def test_request_id_requires_verified_native_binding(revision):
    core, ctx, body = revision
    with pytest.raises(ContractError, match="request_message_binding"):
        core.revise(replace(ctx, request_message=None), body, request_id="TEST-request")
    assert core.current_claim(ctx, body["target_ref"]).revision == 1


def test_null_withdrawal_has_an_exact_durable_receipt(revision):
    core, ctx, body = revision
    source = capture(
        core, replace(ctx, request_message=None), "撤回 TEST-project 配色 蓝色。改成 X；改成 Y。", when=None
    )
    ctx = bound(ctx, source, occurrence="TEST-withdraw-uid")
    body = body | {"new_value": None, "source_evidence_refs": [f"{source.ref}@1"]}
    first = core.revise(ctx, body, request_id="TEST-withdrawal")
    assert first.items[0].state == "retracted"
    after = counts(core, ctx)
    assert core.revise(ctx, body, request_id="TEST-withdrawal") == first
    assert counts(core, ctx) == after


@pytest.mark.parametrize("deleted", [False, True])
def test_backup_restore_keeps_receipt_and_honors_deletion_ledger(revision, tmp_path, deleted):
    core, ctx, body = revision
    first = core.revise(ctx, body, request_id="TEST-request")
    snapshot = tmp_path / "TEST-receipt-snapshot.sqlite3"
    backup_sqlite(core.storage.path, snapshot)
    if deleted:
        owner = MemoryManagement(core, InstallationMaintenance(replace(ctx, request_message=None)))
        owner.apply(owner.preview([ctx.request_message.source_ref], mode="delete"))
    result = apply_restore(ctx.binding, snapshot, backup_dir=tmp_path / "TEST-restore-backup", now=datetime.now(UTC))
    assert result["status"] == "restored", result
    reopened = MemoryCore(CoreConfig(ctx.binding), clock=core.clock)
    reopened.initialize()
    if deleted:
        with pytest.raises(ContractError):
            reopened.revise(ctx, body, request_id="TEST-request")
    else:
        assert reopened.revise(ctx, body, request_id="TEST-request") == first


def test_changed_conditions_can_recover_a_held_result(revision):
    """Review can return a version beside the head; later head changes still fence its receipt."""
    core, ctx, body = revision
    actor = replace(ctx, request_message=None)
    source = capture(core, actor, "TEST-project 配色仅在测试用蓝色。", when="2026-09-05T12:00:00Z")
    existing = accept(core, actor, draft(source, kind="fact", conditions=["仅在测试"])).items[0]
    owner = MemoryManagement(core, InstallationMaintenance(actor))
    owner.set_review_policy(ctx.request_message.scope_id, True, expected_version=0)
    command = capture(core, actor, "Please correct TEST-project 配色: 银色，仅在测试。", when="2026-09-03T12:00:00Z")
    ctx = bound(actor, command, occurrence="TEST-historical-command")
    body = body | {
        "conditions": ["仅在测试"],
        "valid_from": command.event["occurred_at"],
        "source_evidence_refs": [f"{command.ref}@1"],
    }
    first = core.revise(ctx, body, request_id="TEST-request")
    assert first.items[0].ref == existing.ref and first.items[0].disposition == "historical"
    after = counts(core, ctx)
    assert core.revise(ctx, body, request_id="TEST-request") == first
    assert counts(core, ctx) == after
    owner.revise(existing.ref, expected_revision=existing.revision, value="TEST later value")
    with pytest.raises(ContractError, match="expected_revision"):
        core.revise(ctx, body, request_id="TEST-request")
