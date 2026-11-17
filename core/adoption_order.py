"""Conversation positions frozen at each message's first entrance."""


def recorded_key(column):
    """SQL ordering key for contract-validated UTC times, preserving every fractional digit.

    Removing the UTC suffix and trailing fractional zeroes makes equivalent Z
    and +00:00 spellings equal without SQLite's millisecond rounding. The
    argument is a code-owned SQL expression, never capture text.
    """
    return (
        f"(substr({column},1,19)||'.'||CASE WHEN substr({column},20,1)='.' "
        f"THEN rtrim(substr({column},21),'Z+0:') ELSE '' END)"
    )


SOURCE_ADOPTION_ORDER_INDEX = (
    "CREATE INDEX IF NOT EXISTS source_adoption_order ON source_events"
    f"(session_id,scope_id,project_id,branch_id,entry_id,{recorded_key('recorded_at')},source_id)"
)
_PARTITION = "e.session_id=? AND e.scope_id=? AND e.project_id IS ? AND e.branch_id IS ? AND e.entry_id=?"
_FIRST = "NOT EXISTS(SELECT 1 FROM source_events first WHERE first.event_id=e.event_id AND first.source_id<e.source_id)"
_POSITION = f"({recorded_key('e.recorded_at')},e.source_id)"


def partition(source):
    """Exact conversation audience; identity proof is checked separately."""
    return source.session_id, source.scope_id, source.project_id, source.branch_id, source.entry_id


def first_occurrence(conn, ref):
    """Read the immutable position even when the caller holds a later revision."""
    return conn.execute(
        f"SELECT *,{recorded_key('recorded_at')} AS moment FROM source_events "
        "WHERE event_id=? ORDER BY source_id LIMIT 1",
        (ref,),
    ).fetchone()


def adjacent_occurrence(conn, source, first, *, following=False):
    """Seek one neighboring first envelope through the conversation-order index."""
    comparison, direction = (">", "ASC") if following else ("<", "DESC")
    key = recorded_key("e.recorded_at")
    return conn.execute(
        f"SELECT e.*,{recorded_key('e.recorded_at')} AS moment FROM source_events e WHERE {_PARTITION} "
        f"AND {key}{comparison}=? AND {_POSITION}{comparison}(?,?) AND {_FIRST} "
        f"ORDER BY {key} {direction},e.source_id {direction} LIMIT 1",
        (*partition(source), first["moment"], first["moment"], first["source_id"]),
    ).fetchone()


def preceding_occurrence(conn, owner):
    """Find the preceding first envelope, retaining unavailable inbox positions.

    First capture orders occurrences: editing cannot move a message. Pending
    envelopes follow persisted rows at the same recorded instant; their
    insertion ordinal is unknown and they cannot lend words before persistence.
    """
    first = first_occurrence(conn, owner.ref)
    if first is None or (first["source_revision"], first["origin"], first["role"]) != (1, "human_direct", "user"):
        return None
    candidate = adjacent_occurrence(conn, owner, first)
    session, scope, project, branch, entry = partition(owner)
    key = recorded_key("json_extract(v.value,'$.recorded_at')")
    waiting = conn.execute(
        f"""SELECT max({key}) FROM capture_inbox i,json_each(i.payload_json,'$.events') v
        WHERE i.scope_id=? AND i.project_id IS ? AND i.branch_id IS ?
        AND json_extract(i.payload_json,'$.context.session_id')=?
        AND coalesce(json_extract(i.payload_json,'$.context.entry_id'),'local')=?
        AND {key}<? AND NOT EXISTS(SELECT 1 FROM source_events e
            WHERE e.source_event_key=json_extract(v.value,'$.source_event_key'))""",
        (scope, project, branch, session, entry, first["moment"]),
    ).fetchone()[0]
    if waiting is not None and (candidate is None or waiting >= candidate["moment"]):
        return None
    return candidate


def revision_before(conn, ref, owner):
    """Select text witnessed before the confirmation's first entrance, including insertion ties."""
    first = first_occurrence(conn, owner.ref)
    return conn.execute(
        f"SELECT max(e.source_revision) FROM source_events e WHERE e.event_id=? AND {_POSITION}<(?,?)",
        (ref, first["moment"], first["source_id"]),
    ).fetchone()[0]


def prior_assistant_contents(conn, owner):
    """Read earlier assistant text using the same frozen positions and revision cutoff."""
    first = first_occurrence(conn, owner.ref)
    return conn.execute(
        f"""SELECT text.content FROM source_events e JOIN source_events text ON text.event_id=e.event_id
        WHERE {_PARTITION} AND {_FIRST} AND {_POSITION}<(?,?)
        AND e.source_revision=1 AND e.origin='assistant_visible' AND e.role='assistant'
        AND text.origin='assistant_visible' AND text.role='assistant'
        AND text.source_revision=(SELECT max(e.source_revision) FROM source_events e
            WHERE e.event_id=text.event_id AND {_POSITION}<(?,?))""",
        (*partition(owner), first["moment"], first["source_id"], first["moment"], first["source_id"]),
    ).fetchall()
