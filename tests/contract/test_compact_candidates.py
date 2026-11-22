"""Candidate compaction keeps live questions, work receipts and future wake-ups intact."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace

from scope_recall.core import suppression
from scope_recall.core.backup_primitives import sha256
from scope_recall.core.candidate_lifecycle import RULE_VERSION
from scope_recall.core.candidate_tables import encode_refs
from scope_recall.core.storage import Transaction
from scope_recall.maintenance import compact_candidates
from scope_recall.maintenance.compact_candidates import measure_candidates, trim_candidates
from scope_recall.maintenance.compact_store import _trim_authorizer
from test_candidate_lifecycle import _candidate, _snapshot
from test_claims import app, capture, initial
from test_deletion import request
from v11_support import source_event

OLD = "2026-06-01T00:00:00Z"
CUTOFF = "2026-07-01T00:00:00+00:00"
NEW = "2026-08-01T00:00:00Z"


def test_candidate_history_does_not_rescan_all_unlinked_evaluations_per_revision(app):
    core, ctx = app
    with core.storage.read(ctx) as tx:
        db = tx._check()
        sql = compact_candidates._EVALUATION_HISTORY + " SELECT count(*) FROM (" + compact_candidates._DELETIONS + ")"
        plan = db.execute("EXPLAIN QUERY PLAN " + sql, {"cutoff": CUTOFF}).fetchall()
    # NULL work IDs are deliberately shared by unrelated candidates' terminal history.
    assert not any("SEARCH e " in row[3] and "(work_id=?)" in row[3] for row in plan)


def _finish_initial(db, candidate, *, state="waiting_evidence", reason="insufficient_evidence"):
    db.execute(
        "UPDATE candidate_lifecycle SET processing_state='waiting_evidence',reason=?, "
        "created_at=?,updated_at=?,last_evidence_at=?,last_evaluated_at=? WHERE candidate_ref=?",
        (reason, OLD, OLD, OLD, OLD, candidate.ref),
    )
    db.execute(
        "UPDATE candidate_evidence SET observed_at=? WHERE candidate_ref=?",
        (OLD, candidate.ref),
    )
    db.execute(
        "UPDATE work_items SET state='done',available_at=? WHERE subject_ref=?",
        (OLD, "candidate:" + candidate.ref),
    )
    db.execute(
        "UPDATE candidate_evaluations SET state=?,reason=?,created_at=?,completed_at=? WHERE candidate_ref=?",
        (state, reason, OLD, OLD, candidate.ref),
    )


def _evaluation(db, candidate, source, fingerprint, state, *, reason="retired_question", attempted=None):
    return db.execute(
        """INSERT INTO candidate_evaluations(candidate_ref,candidate_revision,evidence_fingerprint,evidence_refs_json,
           rule_version,memory_epoch,state,reason,created_at,completed_at,model_attempted_at)
           VALUES (?,?,?,?,?,0,?,?,?,?,?) RETURNING evaluation_id""",
        (
            candidate.ref,
            candidate.revision,
            fingerprint,
            json.dumps([f"{source.ref}@{source.revision}"]),
            RULE_VERSION,
            state,
            reason,
            OLD,
            OLD,
            attempted,
        ),
    ).fetchone()[0]


def _copy(source, db):
    with closing(sqlite3.connect(source)) as before:
        before.execute("PRAGMA query_only=ON")
        version = before.total_changes
        measured = measure_candidates(before, CUTOFF)
        assert before.total_changes == version
        before.backup(db)
    db.execute("PRAGMA foreign_keys=ON")
    db.set_authorizer(_trim_authorizer)
    try:
        with db:
            trim_candidates(db, CUTOFF, measured)
    finally:
        db.set_authorizer(None)
    assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert db.execute("PRAGMA foreign_key_check").fetchone() is None
    return measured


def _state(db, candidate):
    return db.execute(
        "SELECT processing_state,reason FROM candidate_lifecycle WHERE candidate_ref=? AND candidate_revision=?",
        (candidate.ref, candidate.revision),
    ).fetchone()


def test_closed_waiting_candidates_keep_real_evidence_and_wake_on_new_source(app, tmp_path):
    core, ctx = app
    dormant, old_source, _, _ = _candidate(core, ctx, key="TEST-compact/dormant")
    unfinished, _, _, _ = _candidate(core, ctx, key="TEST-compact/unfinished")
    recent, _, _, _ = _candidate(core, ctx, key="TEST-compact/recent")
    revoked, _, _, _ = _candidate(core, ctx, key="TEST-compact/revoked")
    promoted, _ = initial(core, ctx)
    assert promoted.state == "active"
    source = core.storage.path
    with closing(sqlite3.connect(source)) as db, db:
        db.execute(
            "DELETE FROM candidate_evidence WHERE candidate_ref=? AND source_ref<>?", (dormant.ref, old_source.ref)
        )
        _finish_initial(db, dormant)
        _finish_initial(db, unfinished)
        _finish_initial(db, recent)
        _finish_initial(db, revoked, state="obsolete", reason="authority_revoked")
        db.execute("UPDATE candidate_evaluations SET model_attempted_at=? WHERE candidate_ref=?", (OLD, dormant.ref))
        db.execute("UPDATE work_items SET state='failed' WHERE subject_ref=?", ("candidate:" + unfinished.ref,))
        db.execute("UPDATE candidate_evaluations SET state='failed' WHERE candidate_ref=?", (unfinished.ref,))
        db.execute("UPDATE candidate_lifecycle SET last_evidence_at=? WHERE candidate_ref=?", (NEW, recent.ref))
        obsolete = _evaluation(db, dormant, old_source, "superseded", "obsolete", reason="superseded_by_new_evidence")
        decided = _evaluation(db, dormant, old_source, "already-decided", "waiting_evidence")
        failed = _evaluation(db, dormant, old_source, "older-failure", "failed")
        judged = _evaluation(db, dormant, old_source, "old-verdict", "waiting_evidence", attempted=OLD)
        fenced = _evaluation(db, dormant, old_source, "suppressed", "obsolete", reason="authority_revoked")

    checksum = sha256(source)
    target = tmp_path / "TEST-compacted.sqlite3"
    with closing(sqlite3.connect(target)) as db:
        measured = _copy(source, db)
        assert measured["candidate_lifecycle"] == {"rows": 1, "estimated_bytes": 0}
        assert measured["candidate_evaluations"]["rows"] == 2
        assert measured["candidate_evaluations"]["estimated_bytes"] > 0
        assert _state(db, dormant) == ("archived", "dormant_no_evidence")
        assert _state(db, unfinished)[0] == _state(db, recent)[0] == "waiting_evidence"
        assert _state(db, revoked) == ("waiting_evidence", "authority_revoked")
        retained = dict(db.execute("SELECT evaluation_id,evidence_refs_json FROM candidate_evaluations"))
        assert all(retained[key] == "[]" for key in (obsolete, decided))
        assert all(json.loads(retained[key]) for key in (judged, failed, fenced))
        db.row_factory = sqlite3.Row
        from scope_recall.core.storage import Transaction

        with db:
            tx = Transaction(db, ctx, writable=True)
            _, previous = tx.candidates._model_verdicts(dormant.ref, dormant.revision)
            assert (old_source.ref, old_source.revision) in previous
            fresh = tx.sources.put_source(
                source_event(
                    source_event_key="TEST-compaction/after",
                    content=old_source.event["content"] + " Confirmed again.",
                ),
                scope_id=old_source.scope_id,
                persisted_at=core.clock.utc_now(),
            )
            wake = tx.candidates.observe_source(fresh.ref, fresh.revision, observed_at=core.clock.utc_now())
            assert wake.matched >= 1
            assert _state(db, dormant)[0] == "pending_evaluation", _state(db, dormant)
            assert (
                db.execute("SELECT state FROM claim_versions WHERE claim_id=?", (promoted.ref,)).fetchone()[0]
                == "active"
            )
    assert sha256(source) == checksum


def test_retired_revision_expiry_preserves_implicit_work_and_evaluation_ids(app):
    core, ctx = app
    candidate, evidence, proposal, _ = _candidate(core, ctx, key="TEST-compact/replaced")
    source = core.storage.path
    with closing(sqlite3.connect(source)) as db, db:
        _finish_initial(db, candidate)
        removable = {
            _evaluation(db, candidate, evidence, state, state) for state in ("waiting_evidence", "obsolete", "failed")
        }
        judged = _evaluation(db, candidate, evidence, "judged", "waiting_evidence", attempted=OLD)
        implicit_work = _evaluation(db, candidate, evidence, "implicit-work", "obsolete")
        db.execute(
            """INSERT INTO work_items(work_type,subject_ref,subject_revision,scope_id,project_id,branch_id,
                                      state,available_at)
               VALUES ('evaluate_candidate',?,?,?,?,?,'obsolete',?)""",
            (
                "candidate:" + candidate.ref,
                implicit_work,
                evidence.scope_id,
                evidence.project_id,
                evidence.branch_id,
                OLD,
            ),
        )

    from scope_recall.core.claims import Qualification

    with core.storage.write(ctx) as tx:
        head = tx.claims.append(
            evidence.scope_id,
            proposal,
            Qualification("proposed", "inferred_suggestion", "TEST_replaced"),
            previous=candidate,
            recorded_at=core.clock.utc_now(),
        )
        tx.candidates.register(head.ref, head.revision, observed_at=core.clock.utc_now(), schedule_initial=False)
    with closing(sqlite3.connect(source)) as db, db:
        db.execute("UPDATE candidate_lifecycle SET updated_at=?,dormant_at=? WHERE candidate_revision=1", (OLD, OLD))
        db.execute(
            "UPDATE claim_versions SET recorded_from=?,recorded_to=? WHERE claim_id=? AND revision=1",
            (OLD, OLD, candidate.ref),
        )
        high_watermark = _evaluation(db, candidate, evidence, "highest", "obsolete")

    with closing(sqlite3.connect(":memory:")) as db:
        measured = _copy(source, db)
        ids = {row[0] for row in db.execute("SELECT evaluation_id FROM candidate_evaluations")}
        assert ids.isdisjoint(removable)
        assert {judged, implicit_work, high_watermark}.issubset(ids)
        assert db.execute(
            "SELECT evidence_refs_json,model_attempted_at FROM candidate_evaluations WHERE evaluation_id=?", (judged,)
        ).fetchone() == ("[]", OLD)
        assert measured["candidate_evaluations"]["rows"] >= len(removable)
        assert db.execute(
            "SELECT count(*) FROM work_items WHERE work_type='evaluate_candidate' AND subject_revision=?",
            (implicit_work,),
        ).fetchone() == (1,)
        with db:
            assert _evaluation(db, candidate, evidence, "next", "obsolete") > high_watermark


def _retryable_evaluation(db, candidate, evidence):
    evaluation = _evaluation(db, candidate, evidence, "retryable", "failed", attempted=OLD)
    work_id = db.execute(
        """INSERT INTO work_items(work_type,subject_ref,subject_revision,scope_id,project_id,branch_id,
           state,available_at) VALUES ('evaluate_candidate',?,?,?,?,?,'failed',?) RETURNING work_id""",
        ("candidate:" + candidate.ref, evaluation, candidate.scope_id, candidate.project_id, candidate.branch_id, OLD),
    ).fetchone()[0]
    db.execute("UPDATE candidate_evaluations SET work_id=? WHERE evaluation_id=?", (work_id, evaluation))
    return evaluation, work_id


def test_current_candidate_trims_old_payloads_independently_of_current_work_and_recent_history(app):
    core, ctx = app
    candidate, evidence, _, registered = _candidate(core, ctx, key="TEST-compact/current")
    with core.storage.read(ctx) as tx:
        original_refs, original_question = tx.candidates._question(candidate.ref, candidate.revision)
    newer = capture(core, ctx, evidence.event["content"] + " And again.")
    with core.storage.read(ctx) as tx:
        newer_refs, new_question = tx.candidates._question(candidate.ref, candidate.revision)
        assert newer_refs != original_refs

    source = core.storage.path
    with closing(sqlite3.connect(source)) as db, db:
        _finish_initial(db, candidate)
    snapshot = _snapshot(core, candidate.ref, candidate.revision)
    with core.storage.write(ctx) as tx:
        current, queued = tx.candidates._schedule(snapshot, now=NEW, rule_version=RULE_VERSION)
        assert queued
    background = capture(core, ctx, evidence.event["content"] + " In a tool result.", origin="tool_observation")
    background_ref = (background.ref, background.revision)
    with closing(sqlite3.connect(source)) as db, db:
        obsolete = [_evaluation(db, candidate, evidence, f"history-{i}", "obsolete") for i in range(128)]
        first_attempt = _evaluation(db, candidate, evidence, "old-model-a", "waiting_evidence", attempted=OLD)
        keeper = _evaluation(db, candidate, evidence, "old-model-b", "obsolete", attempted=OLD)
        db.execute(
            """UPDATE candidate_evaluations SET evidence_refs_json=?,result_digest='TEST-result'
               WHERE evaluation_id=?""",
            (encode_refs(((evidence.ref, evidence.revision), background_ref)), first_attempt),
        )
        db.execute(
            "UPDATE candidate_evaluations SET evidence_refs_json=? WHERE evaluation_id=?",
            (encode_refs(newer_refs), keeper),
        )
        recent = _evaluation(db, candidate, evidence, "new-history", "waiting_evidence")
        db.execute("UPDATE candidate_evaluations SET completed_at=? WHERE evaluation_id=?", (NEW, recent))
        failed, failed_work = _retryable_evaluation(db, candidate, evidence)
        revoked = _evaluation(db, candidate, evidence, "revoked", "obsolete", reason="authority_revoked")
        preserved = {
            table: list(db.execute(f"SELECT * FROM {table}"))
            for table in ("claims", "claim_versions", "source_events", "work_items", "candidate_lifecycle")
        }
    snapshot = _snapshot(core, candidate.ref, candidate.revision)
    checksum = sha256(source)
    with closing(sqlite3.connect(":memory:")) as db:
        measured = _copy(source, db)
        counts = measured["candidate_evaluations"]
        assert (
            counts["rows"],
            counts["deleted_rows"],
            counts["trimmed_attempted_rows"],
            counts["trimmed_unattempted_rows"],
        ) == (130, 0, 1, 129)
        assert measured["candidate_lifecycle"]["rows"] == 0
        for table, rows in preserved.items():
            assert list(db.execute(f"SELECT * FROM {table}")) == rows
        retained = dict(db.execute("SELECT evaluation_id,evidence_refs_json FROM candidate_evaluations"))
        assert all(retained[key] == "[]" for key in (*obsolete, registered.evaluation_id))
        assert retained[first_attempt] == encode_refs((background_ref,))
        assert all(json.loads(retained[key]) for key in (keeper, failed, recent, revoked, current))
        assert db.execute(
            "SELECT result_digest,model_attempted_at FROM candidate_evaluations WHERE evaluation_id=?", (first_attempt,)
        ).fetchone() == ("TEST-result", OLD)
        assert measure_candidates(db, CUTOFF)["candidate_evaluations"]["rows"] == 0
        db.row_factory = sqlite3.Row
        with db:
            tx = Transaction(db, ctx, writable=True)
            assert tx.candidates._model_verdicts(candidate.ref, candidate.revision) == (
                3,
                frozenset((*newer_refs, background_ref)),
            )
            assert tx.candidates.evaluation(current).evidence_refs == newer_refs
            outsider = Transaction(db, replace(ctx, project_id="TEST-outsider"), writable=False)
            assert outsider.candidates.evaluation(current) is None
            target = tx.deletions.target(newer.ref)
            tx.deletions.block(request(newer, mode="suppress"), (target,), now=NEW)
            assert tx.candidates._question(candidate.ref, candidate.revision)[1] == original_question
            assert tx.candidates._nothing_new_to_ask(candidate.ref, candidate.revision, rule_version=RULE_VERSION)
            suppression.release(tx, (target,), now=NEW)
            assert tx.candidates._question(candidate.ref, candidate.revision)[1] == new_question
            assert tx.candidates._schedule(snapshot, now=NEW, rule_version=RULE_VERSION) == (None, False)
            assert tx.work.retry_failed(now=NEW, dry_run=True)["retried_work_ids"] == [failed_work]
            assert tx.work.retry_failed(now=NEW, dry_run=False)["retried_work_ids"] == [failed_work]
            assert tx.candidates.evaluation(failed).evidence_refs == ((evidence.ref, evidence.revision),)
    assert sha256(source) == checksum


def test_candidate_compaction_keeps_suppression_receipt_and_single_attempt_guard(app):
    core, ctx = app
    candidate, evidence, _, registration = _candidate(core, ctx, key="TEST-compact/paused")
    now = core.clock.utc_now()
    with core.storage.write(ctx) as tx:
        target = tx.deletions.target(evidence.ref)
        tx.deletions.block(request(evidence, mode="suppress"), (target,), now=now)

    source = core.storage.path
    with closing(sqlite3.connect(source)) as db, db:
        db.execute("UPDATE candidate_evaluations SET created_at=?,completed_at=?", (OLD, OLD))
        db.execute("UPDATE work_items SET available_at=? WHERE work_type='evaluate_candidate'", (OLD,))
    with closing(sqlite3.connect(":memory:")) as db:
        assert _copy(source, db)["candidate_evaluations"]["rows"] == 0
        db.row_factory = sqlite3.Row
        with db:
            tx = Transaction(db, ctx, writable=True)
            target = tx.deletions.target(evidence.ref)
            suppression.release(tx, (target,), now=now)
            resumed = tx.candidates.evaluation(registration.evaluation_id)
            assert resumed.state == "queued"
            assert resumed.evidence_refs == ((evidence.ref, evidence.revision),)
            (work,) = tx.work.claim_next(
                "TEST-released", now, lease_seconds=60, allowed_work_types=frozenset({"evaluate_candidate"})
            )
            assert work.subject_revision == resumed.evaluation_id
            assert tx.candidates.begin_model_attempt(resumed.evaluation_id, *work.lease, now=now)
            assert not tx.candidates.begin_model_attempt(resumed.evaluation_id, *work.lease, now=now)


def test_orphan_cleanup_handles_only_expired_missing_dependencies(app):
    core, ctx = app
    candidate, evidence, _, _ = _candidate(core, ctx, key="TEST-compact/orphan")
    fresh = capture(core, ctx, "TEST orphan second source")
    source = core.storage.path
    with closing(sqlite3.connect(source)) as db, db:
        db.execute(
            "INSERT INTO candidate_evidence VALUES (?,?,?,?,?)",
            (candidate.ref, candidate.revision, "src:missing", 1, OLD),
        )
        db.execute(
            "INSERT INTO candidate_evidence VALUES (?,?,?,?,?)",
            ("clm:missing", 1, evidence.ref, evidence.revision, OLD),
        )
        db.execute("INSERT INTO candidate_trigger_terms VALUES ('TEST-orphan','clm:missing',1)")
        db.execute("INSERT INTO candidate_source_triggers VALUES ('src:missing',1,0,0,1,?)", (OLD,))
        db.execute(
            "INSERT OR REPLACE INTO candidate_source_triggers VALUES (?,?,0,0,1,?)", (fresh.ref, fresh.revision, OLD)
        )
        db.execute(
            "INSERT OR REPLACE INTO candidate_source_triggers VALUES (?,?,0,0,0,?)",
            (evidence.ref, evidence.revision, OLD),
        )

    checksum = sha256(source)
    with closing(sqlite3.connect(":memory:")) as db:
        measured = _copy(source, db)
        assert measured["candidate_evidence"]["rows"] == 2
        assert measured["candidate_trigger_terms"]["rows"] == 1
        assert measured["candidate_source_triggers"]["rows"] == 1
        assert set(db.execute("SELECT source_ref,truncated FROM candidate_source_triggers")) >= {
            (fresh.ref, 1),
            (evidence.ref, 0),
        }
        assert db.execute(
            "SELECT COUNT(*) FROM candidate_evidence WHERE candidate_ref=? AND source_ref=?",
            (candidate.ref, evidence.ref),
        ).fetchone() == (1,)
        for table in ("candidate_evidence", "candidate_trigger_terms", "candidate_source_triggers"):
            assert measured[table]["estimated_bytes"] > 0
        assert sum(measure_candidates(db, CUTOFF)[table]["rows"] for table in measured) == 0
    assert sha256(source) == checksum
