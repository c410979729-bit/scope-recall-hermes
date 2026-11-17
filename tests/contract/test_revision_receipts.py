"""Exact local management retries retain their first durable outcome and authority."""

import json
from dataclasses import replace

import pytest
from scope_recall.contracts import ContractError, TrustedSourcePrincipal
from scope_recall.core import CoreConfig, MemoryCore
from scope_recall.core.restore import InstallationMaintenance
from test_claims import accept, app, capture, draft, initial
from test_memory_admin import management


def counts(core, context):
    """Read authoritative counts and epoch to detect duplicate effects."""
    with core.storage.read(context) as tx:
        conn = tx._check()
        return tuple(
            conn.execute("SELECT count(*) FROM " + table).fetchone()[0]
            for table in ("claim_versions", "source_events", "management_actions")
        ) + (tx.memory_epoch(),)


def revise(owner, item, **changes):
    """Send one synthetic operator occurrence; changes describe an actual different request."""
    return owner.revise(
        item.ref, **(dict(expected_revision=1, value="TEST replacement", request_id="TEST-correction") | changes)
    )


def test_management_exact_retry_returns_original_result(app):
    """A lost response can be recovered without another correction or source."""
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    first = revise(owner, item)
    after = counts(core, context)
    core.clock.now = "2026-09-07T12:00:00Z"
    assert revise(owner, item) == first
    assert counts(core, context) == after


@pytest.mark.parametrize("changes", [{"value": "TEST different"}, {"expected_revision": 2}, {"value": None}])
def test_same_key_different_payload_is_refused(app, changes):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    revise(owner, item)
    after = counts(core, context)
    with pytest.raises(ContractError, match="request_id"):
        revise(owner, item, **changes)
    assert counts(core, context) == after


@pytest.mark.parametrize(
    "change",
    [
        dict(session_id="TEST-another-session"),
        dict(source_principal=TrustedSourcePrincipal("human", "verified", "principal:TEST-other")),
        dict(actor_origin="host_generated"),
        dict(task_anchor="TEST-another-command-authority"),
    ],
)
def test_another_authority_cannot_obtain_receipt(app, change):
    core, context = app
    item, _ = initial(core, context)
    revise(management(core, context), item)
    after = counts(core, context)
    other = management(core, replace(context, **change))
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        revise(other, item)
    assert counts(core, context) == after


def test_reopening_core_recovers_persisted_response(app):
    core, context = app
    item, _ = initial(core, context)
    first = revise(management(core, context), item)
    after = counts(core, context)
    reopened = MemoryCore(CoreConfig(context.binding), clock=core.clock)
    reopened.initialize()
    assert revise(management(reopened, context), item) == first
    assert counts(reopened, context) == after


def test_revoked_capability_cannot_replay(app):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    revise(owner, item)
    after = counts(core, context)
    owner.authority = InstallationMaintenance(replace(context, allowed_scope_ids=frozenset()))
    with pytest.raises(ContractError):
        revise(owner, item)
    assert counts(core, context) == after


def test_hidden_target_cannot_be_recovered_from_receipt(app):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    revise(owner, item)
    owner.apply(owner.preview([item.ref], mode="delete"))
    after = counts(core, context)
    with pytest.raises(ContractError):
        revise(owner, item)
    assert counts(core, context) == after


def test_later_correction_prevents_old_receipt_replay(app):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    revise(owner, item)
    revise(owner, item, expected_revision=2, value="TEST newer", request_id="TEST-new-command")
    after = counts(core, context)
    with pytest.raises(ContractError, match="expected_revision"):
        revise(owner, item)
    assert counts(core, context) == after


@pytest.mark.parametrize("request_id", [None, "TEST-new-user-command"])
def test_new_occurrence_keeps_version_check(app, request_id):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    revise(owner, item)
    with pytest.raises(ContractError, match="VERSION_CONFLICT"):
        revise(owner, item, request_id=request_id)


def test_same_key_cannot_change_target(app):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    revise(owner, item)
    source = capture(core, context, "TEST-other palette TEST separate.")
    other = accept(core, context, draft(source, "TEST separate", subject="TEST-other", predicate="palette")).items[0]
    assert other.ref != item.ref and other.state == "active"
    with pytest.raises(ContractError, match="request_id"):
        revise(owner, other)


def test_receipt_preserves_truthful_actor_and_has_no_value_copy(app):
    core, context = app
    item, _ = initial(core, context)
    result = revise(management(core, context), item)
    with core.storage.read(context) as tx:
        audit = (
            tx._check().execute("SELECT * FROM management_actions WHERE action_id=?", (result["action_id"],)).fetchone()
        )
        assert audit["actor"] == result["actor"] == "local_user_management"
        assert "TEST replacement" not in audit["targets_json"]
        assert json.loads(audit["targets_json"])[0]["receipt"]["result"] == result
        source = tx.source(result["source_ref"], 1)
        assert source.event["origin"] == "host_generated"
        assert source.event["role"] == "system"


@pytest.mark.parametrize("request_id", ["", " ", 5, "x" * 241])
def test_invalid_request_key_cannot_write(app, request_id):
    core, context = app
    item, _ = initial(core, context)
    before = counts(core, context)
    with pytest.raises(ContractError, match="request_id"):
        revise(management(core, context), item, request_id=request_id)
    assert counts(core, context) == before


def test_private_interval_prevents_receipt_release(app):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    revise(owner, item)
    owner.private_on(target="global")
    before = counts(core, context)
    with pytest.raises(ContractError, match="private_mode"):
        revise(owner, item)
    assert counts(core, context) == before


def test_withdrawal_retry_keeps_single_revision_and_original_epoch(app):
    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    first = revise(owner, item, value=None)
    after = counts(core, context)
    assert revise(owner, item, value=None) == first
    assert counts(core, context) == after


def test_management_cli_passes_retry_id_to_durable_api(app, monkeypatch, capsys):
    from scope_recall.maintenance import memory

    core, context = app
    item, _ = initial(core, context)
    owner = management(core, context)
    monkeypatch.setattr(memory, "open_management", lambda *args, **kwargs: owner)
    command = [
        "revise",
        item.ref,
        "--expected-revision",
        "1",
        "--value",
        "TEST replacement",
        "--request-id",
        "TEST-cli-occurrence",
        "--json",
    ]
    assert memory.main(command) == 0
    first = json.loads(capsys.readouterr().out)["data"]
    after = counts(core, context)
    assert memory.main(command) == 0
    assert json.loads(capsys.readouterr().out)["data"] == first
    assert counts(core, context) == after
