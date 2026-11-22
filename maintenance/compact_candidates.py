"""Expire candidate processing history in a disposable SQLite copy."""

from __future__ import annotations

import sqlite3

from ..core.backup_primitives import BackupError
from ..core.candidate_lifecycle import RULE_VERSION
from ..core.candidate_tables import reachable_sql
from ..core.source_origin import ACTION as SOURCE_ORIGIN_ACTION

# Dormancy changes the whole candidate and therefore requires the whole revision
# to be quiet. Evaluation history is selected separately, one question at a time.
_QUIET = """
WITH quiet AS MATERIALIZED (
    SELECT l.*
    FROM candidate_lifecycle l
    CROSS JOIN claims c ON c.claim_id=l.candidate_ref
    CROSS JOIN claim_versions v ON v.claim_id=l.candidate_ref AND v.revision=l.candidate_revision
    WHERE l.processing_state='waiting_evidence' AND l.reason<>'authority_revoked'
      AND c.current_revision=l.candidate_revision
      AND c.read_blocked=0 AND c.suppressed=0 AND v.state IN ('proposed','disputed')
      AND l.candidate_ref NOT IN (
          SELECT claim_id FROM claim_versions WHERE state IN ('active','superseded','retracted'))
      AND julianday(l.created_at)<julianday(:cutoff) AND julianday(l.updated_at)<julianday(:cutoff)
      AND (l.last_evidence_at IS NULL OR julianday(l.last_evidence_at)<julianday(:cutoff))
      AND (l.last_evaluated_at IS NULL OR julianday(l.last_evaluated_at)<julianday(:cutoff))
      AND (l.dormant_at IS NULL OR julianday(l.dormant_at)<julianday(:cutoff))
      AND NOT EXISTS (
          SELECT 1 FROM candidate_evaluations q
          WHERE q.candidate_ref=l.candidate_ref AND q.candidate_revision=l.candidate_revision
            AND (q.state='queued'
              OR COALESCE(julianday(q.created_at)<julianday(:cutoff),0)=0
              OR COALESCE(julianday(q.completed_at)<julianday(:cutoff),0)=0
              OR (q.model_attempted_at IS NOT NULL AND COALESCE(julianday(q.model_attempted_at)<julianday(:cutoff),0)=0)
              OR EXISTS (
                  SELECT 1 FROM work_items w WHERE w.work_id=q.work_id
                    AND (w.state IN ('pending','leased','failed') OR w.lease_owner IS NOT NULL
                         OR w.lease_until IS NOT NULL OR COALESCE(julianday(w.available_at)<julianday(:cutoff),0)=0))
              OR EXISTS (
                  SELECT 1 FROM work_items w WHERE w.work_type='evaluate_candidate'
                    AND w.subject_ref='candidate:'||q.candidate_ref AND w.subject_revision=q.evaluation_id
                    AND (w.state IN ('pending','leased','failed') OR w.lease_owner IS NOT NULL
                         OR w.lease_until IS NOT NULL OR COALESCE(julianday(w.available_at)<julianday(:cutoff),0)=0)))
      )
)
"""

# Source provenance can change independently of an event's content revision.
# A changed origin can change the question even when all its source IDs match.
_CORRECTED_ORIGINS = """
    SELECT json_extract(s.value,'$.ref'),json_extract(s.value,'$.revision')
    FROM management_actions m CROSS JOIN json_each(m.targets_json,'$.sources') s
    WHERE m.action=:source_origin_action AND json_type(s.value,'$.ref')='text'
      AND json_type(s.value,'$.revision')='integer'
"""

# The window for new evidence starts when the current question was created.
# A worker can finish an old question after newer evidence arrived while it ran.
_DORMANT = (
    _QUIET
    + f"""
SELECT l.candidate_ref,l.candidate_revision,0 AS bytes
FROM quiet l
WHERE l.rule_version=:rule_version
  AND (
      (l.last_evidence_at IS NULL AND NOT EXISTS (
          SELECT 1 FROM candidate_evidence x
          WHERE x.candidate_ref=l.candidate_ref AND x.candidate_revision=l.candidate_revision))
      OR EXISTS (
          SELECT 1 FROM candidate_evaluations q
          WHERE q.candidate_ref=l.candidate_ref AND q.candidate_revision=l.candidate_revision
            AND q.evidence_fingerprint=l.evidence_fingerprint AND q.rule_version=l.rule_version
            AND q.state IN ('waiting_evidence','failed','obsolete') AND q.reason<>'authority_revoked'
            AND (l.last_evidence_at IS NULL OR julianday(l.last_evidence_at)<=julianday(q.created_at))
            AND json_type(q.evidence_refs_json)='array'
            AND json_array_length(q.evidence_refs_json)=(
                SELECT count(*) FROM candidate_evidence x
                WHERE x.candidate_ref=l.candidate_ref AND x.candidate_revision=l.candidate_revision)
            AND NOT EXISTS (
                SELECT 1 FROM candidate_evidence x
                WHERE x.candidate_ref=l.candidate_ref AND x.candidate_revision=l.candidate_revision
                  AND (COALESCE(julianday(x.observed_at)<=julianday(q.created_at),0)=0
                    OR NOT EXISTS (
                        SELECT 1 FROM json_each(q.evidence_refs_json) r
                        WHERE r.type='text' AND r.value=x.source_ref||'@'||x.source_revision)
                    OR NOT EXISTS (
                        SELECT 1 FROM source_events s WHERE s.event_id=x.source_ref
                          AND s.source_revision=x.source_revision AND s.read_blocked=0 AND s.suppressed=0)
                    OR EXISTS (
                        SELECT 1 FROM object_blocks b WHERE b.object_kind='event' AND b.object_ref=x.source_ref
                          AND (b.read_blocked=1 OR b.suppressed=1))
                    OR (x.source_ref,x.source_revision) IN ({_CORRECTED_ORIGINS})))
      )
  )
"""
)

# The subject lookup finds work even when the explicit link is missing. A
# different explicit link or audience is ambiguous and leaves the row intact.
_EVALUATION_HISTORY = f"""
WITH history AS NOT MATERIALIZED (
    SELECT e.*, w.work_id AS subject_work_id, c.current_revision>l.candidate_revision AS retired
    FROM candidate_lifecycle l
    CROSS JOIN claims c ON c.claim_id=l.candidate_ref
    CROSS JOIN claim_versions v ON v.claim_id=l.candidate_ref AND v.revision=l.candidate_revision
    CROSS JOIN candidate_evaluations e
      ON e.candidate_ref=l.candidate_ref AND e.candidate_revision=l.candidate_revision
    LEFT JOIN work_items w ON w.work_type='evaluate_candidate'
      AND w.subject_ref='candidate:'||e.candidate_ref AND w.subject_revision=e.evaluation_id
    WHERE c.read_blocked=0 AND c.suppressed=0 AND v.state IN ('proposed','disputed')
      AND c.scope_id=l.scope_id AND c.project_id IS l.project_id AND c.branch_id IS l.branch_id
      AND l.candidate_ref NOT IN (
          SELECT claim_id FROM claim_versions WHERE state IN ('active','superseded','retracted'))
      AND ((c.current_revision=l.candidate_revision AND {reachable_sql("l.")} AND l.reason<>'authority_revoked'
            AND (e.evidence_fingerprint<>l.evidence_fingerprint OR e.rule_version<>l.rule_version))
           OR (c.current_revision>l.candidate_revision AND l.processing_state='archived'
               AND l.reason='superseded_by_candidate_revision' AND julianday(v.recorded_to)<julianday(:cutoff)
               AND julianday(l.updated_at)<julianday(:cutoff)))
      AND e.state IN ('waiting_evidence','obsolete','failed','archived','resolved') AND e.reason<>'authority_revoked'
      AND julianday(e.created_at)<julianday(:cutoff) AND julianday(e.completed_at)<julianday(:cutoff)
      AND (e.model_attempted_at IS NULL OR julianday(e.model_attempted_at)<julianday(:cutoff))
      -- A missing explicit link is valid. Check it after selecting this revision:
      -- NULL work IDs otherwise form a large index range shared by unrelated candidates.
      AND CASE WHEN e.work_id IS NULL THEN 1 ELSE e.work_id=w.work_id END
      AND (w.work_id IS NULL OR (
          w.state IN ('done','obsolete') AND w.lease_owner IS NULL AND w.lease_until IS NULL
          AND julianday(w.available_at)<julianday(:cutoff)
          AND w.scope_id=l.scope_id AND w.project_id IS l.project_id AND w.branch_id IS l.branch_id))
)
"""

# A current question's fingerprint and rule version must survive: withdrawing
# some evidence can reveal that exact question again. Only a closed old
# revision can lose keys, and SQLite must not reuse its highest evaluation ID.
_DELETIONS = """
SELECT e.evaluation_id,
           length(CAST(e.candidate_ref AS BLOB)) + length(CAST(e.evidence_refs_json AS BLOB))
           + length(CAST(e.evidence_fingerprint AS BLOB)) + length(CAST(e.rule_version AS BLOB))
           + length(CAST(e.state AS BLOB)) + length(CAST(e.reason AS BLOB))
           + length(CAST(e.created_at AS BLOB)) + length(CAST(e.completed_at AS BLOB))
           + coalesce(length(CAST(e.failure_code AS BLOB)),0) AS bytes
FROM history e
WHERE e.retired AND e.subject_work_id IS NULL AND e.model_attempted_at IS NULL AND e.result_digest IS NULL
  AND e.state IN ('waiting_evidence','obsolete','failed')
  AND e.evaluation_id<(SELECT max(evaluation_id) FROM candidate_evaluations)
"""

# Model pacing uses the row count and latest attempt, while restatement uses
# the union of judged references. Keep each unique ref in the newest eligible
# terminal attempt that held it. A row that can reopen must never be a keeper.
# A malformed list is untouched and cannot vouch for a ref in another row.
_LISTS = """
, terminal_lists AS MATERIALIZED (
    SELECT evaluation_id, retired, model_attempted_at IS NOT NULL AS attempted,
           result_digest IS NOT NULL AS has_result
    FROM history
    WHERE state<>'failed' AND json_type(evidence_refs_json)='array' AND json_array_length(evidence_refs_json)>0
), attempt_refs AS (
    SELECT a.evaluation_id, r.value,
           max(a.evaluation_id) OVER (PARTITION BY a.candidate_ref,a.candidate_revision,r.value) AS keeper
    FROM terminal_lists e
    CROSS JOIN candidate_evaluations a USING (evaluation_id)
    CROSS JOIN json_each(a.evidence_refs_json) r
    WHERE NOT e.retired AND e.attempted
      AND NOT EXISTS (
          SELECT 1 FROM json_each(a.evidence_refs_json) j
          WHERE j.type<>'text' OR instr(j.value,'@')<2
             OR CAST(substr(j.value,instr(j.value,'@')+1) AS INTEGER)<1
             OR substr(j.value,instr(j.value,'@')+1)<>
                CAST(CAST(substr(j.value,instr(j.value,'@')+1) AS INTEGER) AS TEXT))
), deduplicated AS (
    SELECT evaluation_id,json_group_array(value) FILTER (WHERE evaluation_id=keeper) AS refs
    FROM attempt_refs GROUP BY evaluation_id HAVING max(evaluation_id<>keeper)
), shortened AS (
    SELECT evaluation_id, refs, 1 AS attempted FROM deduplicated
    UNION ALL
    SELECT evaluation_id, '[]', attempted FROM terminal_lists
    WHERE retired OR (NOT attempted AND NOT has_result)
)
SELECT s.evaluation_id,s.refs,s.attempted,
       length(CAST(e.evidence_refs_json AS BLOB))-length(CAST(s.refs AS BLOB)) AS bytes
FROM shortened s CROSS JOIN candidate_evaluations e USING (evaluation_id)
WHERE e.evaluation_id NOT IN (SELECT evaluation_id FROM deletions)
  AND length(CAST(e.evidence_refs_json AS BLOB))>length(CAST(s.refs AS BLOB))
"""

_EVALUATION_LISTS = f"{_EVALUATION_HISTORY}, deletions AS ({_DELETIONS}) {_LISTS}"

# Candidate membership is the source-page cursor. Presence of both parents
# suffices to preserve it, including blocked and non-current source versions.
_ORPHANS = {
    "candidate_evidence": (
        "candidate_ref,candidate_revision,source_ref,source_revision",
        """SELECT candidate_ref,candidate_revision,source_ref,source_revision,
                  length(CAST(candidate_ref AS BLOB)) + length(CAST(source_ref AS BLOB))
                  + length(CAST(observed_at AS BLOB)) AS bytes
           FROM candidate_evidence x
           WHERE julianday(x.observed_at)<julianday(:cutoff)
             AND (NOT EXISTS (
                      SELECT 1 FROM candidate_lifecycle l
                      WHERE l.candidate_ref=x.candidate_ref AND l.candidate_revision=x.candidate_revision)
                  OR NOT EXISTS (
                      SELECT 1 FROM source_events s
                      WHERE s.event_id=x.source_ref AND s.source_revision=x.source_revision))""",
    ),
    "candidate_trigger_terms": (
        "term,candidate_ref,candidate_revision",
        """SELECT term,candidate_ref,candidate_revision,
                  length(CAST(term AS BLOB)) + length(CAST(candidate_ref AS BLOB)) AS bytes
           FROM candidate_trigger_terms t
           WHERE NOT EXISTS (
               SELECT 1 FROM candidate_lifecycle l
               WHERE l.candidate_ref=t.candidate_ref AND l.candidate_revision=t.candidate_revision)""",
    ),
    "candidate_source_triggers": (
        "source_ref,source_revision",
        """SELECT source_ref,source_revision,
                  length(CAST(source_ref AS BLOB)) + length(CAST(processed_at AS BLOB)) AS bytes
           FROM candidate_source_triggers t
           WHERE julianday(t.processed_at)<julianday(:cutoff)
             AND NOT EXISTS (
                 SELECT 1 FROM source_events s
                 WHERE s.event_id=t.source_ref AND s.source_revision=t.source_revision)""",
    ),
}


def measure_candidates(conn: sqlite3.Connection, cutoff: str) -> dict[str, dict[str, int]]:
    """Count candidate changes and their minimum logical text bytes without writes."""
    params = {"cutoff": cutoff, "rule_version": RULE_VERSION, "source_origin_action": SOURCE_ORIGIN_ACTION}
    deleted, removed_bytes = conn.execute(
        f"{_EVALUATION_HISTORY} SELECT count(*),coalesce(sum(bytes),0) FROM ({_DELETIONS})", params
    ).fetchone()
    trimmed, saved_bytes, attempted = conn.execute(
        f"SELECT count(*),coalesce(sum(bytes),0),coalesce(sum(attempted),0) FROM ({_EVALUATION_LISTS})",
        params,
    ).fetchone()
    measured = {
        "candidate_evaluations": {
            "rows": deleted + trimmed,
            "estimated_bytes": removed_bytes + saved_bytes,
            "deleted_rows": deleted,
            "trimmed_attempted_rows": attempted,
            "trimmed_unattempted_rows": trimmed - attempted,
        }
    }
    selections = {"candidate_lifecycle": _DORMANT, **{table: query for table, (_, query) in _ORPHANS.items()}}
    for table, query in selections.items():
        count, size = conn.execute(f"SELECT count(*),coalesce(sum(bytes),0) FROM ({query})", params).fetchone()
        measured[table] = {"rows": count, "estimated_bytes": size}
    return measured


def trim_candidates(conn: sqlite3.Connection, cutoff: str, measured: dict[str, dict[str, int]]) -> None:
    """Apply exactly the measured candidate changes inside the caller's copy transaction."""
    params = {"cutoff": cutoff, "rule_version": RULE_VERSION, "source_origin_action": SOURCE_ORIGIN_ACTION}
    # A top-level WITH hides SQLite's statement row count from Python sqlite3.
    deleted = conn.execute(
        "DELETE FROM candidate_evaluations WHERE evaluation_id IN "
        f"({_EVALUATION_HISTORY} SELECT evaluation_id FROM ({_DELETIONS}))",
        params,
    ).rowcount
    cleared = conn.execute(
        "UPDATE candidate_evaluations AS e SET evidence_refs_json=l.refs FROM "
        f"({_EVALUATION_LISTS}) l WHERE e.evaluation_id=l.evaluation_id",
        params,
    ).rowcount
    evaluations = measured["candidate_evaluations"]
    _verify_count(deleted, evaluations["deleted_rows"])
    _verify_count(deleted + cleared, evaluations["rows"])

    archived = conn.execute(
        """UPDATE candidate_lifecycle SET processing_state='archived',reason='dormant_no_evidence',
              dormant_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
           WHERE (candidate_ref,candidate_revision) IN
           (SELECT candidate_ref,candidate_revision FROM ("""
        + _DORMANT
        + "))",
        params,
    ).rowcount
    _verify_count(archived, measured["candidate_lifecycle"]["rows"])

    for table, (keys, query) in _ORPHANS.items():
        removed = conn.execute(f"DELETE FROM {table} WHERE ({keys}) IN (SELECT {keys} FROM ({query}))", params).rowcount
        _verify_count(removed, measured[table]["rows"])


def _verify_count(actual: int, expected: int) -> None:
    if actual != expected:
        raise BackupError("candidate compaction selection changed on the staging copy")
