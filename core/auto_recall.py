"""Conservative automatic-recall eligibility, shared by every host.

Only whole conversational messages are skipped. A self-contained topic following
a conversational preamble is searched in its own words.
"""

from __future__ import annotations

import re
import sqlite3
import time
import unicodedata
from typing import Protocol

from ..contracts import ContractError, TrustedContext

_SOCIAL = re.compile(
    r"(?:ok(?:ay)?|yes|yeah|yep|sure|alright|understood|got it|sounds good|"
    r"thanks?(?: you)?|thank you|well done|great(?: job)?|good (?:job|work)|excellent|perfect|approved|"
    r"好的?|行|可以|同意|没问题|嗯+|收到|明白了?|知道了|谢谢(?:你)?|辛苦了|太棒了|你?真棒|干得好|做得(?:很)?好|不错)",
    re.I,
)
_FOLLOW_THROUGH = re.compile(
    r"(?:please\s+)?(?:continue|resume|go on|go ahead|keep going|carry on|proceed|"
    r"(?:let['’]s )?do (?:that|it)|(?:do|proceed with) (?:what|as) you (?:recommend|suggest)(?:ed)?)"
    r"(?:\s+please)?|(?:请)?(?:继续|接着|继续执行|继续做|接着做)(?:吧|呀|啊)?|"
    r"(?:就)?按(?:照)?你(?:推荐|建议|说|刚才说)的(?:办|做|执行|设置|来|改|弄)(?:吧)?",
    re.I,
)
_SESSION_STATUS = re.compile(
    r"(?:(?:is|are) (?:it|that|this|you|the (?:agent|task|job|run|test|build)) "
    r"(?:done|finished|running|ready|working|started)(?: yet)?|"
    r"(?:did|have) you (?:finish|finished|start|started|delete|deleted)(?: it| that)?|"
    r"(?:any progress|how is it going|what(?:'s| is) the status))|"
    r"(?:(?:你|任务|测试|构建|进程|代理|codex|claude(?: code)?|workbuddy|dsh|hermes)"
    r"(?:已经|在|开始|还在)?(?:开始)?(?:跑|运行|执行|启动|完成|做完|结束|好了)(?:了|了吗|了没|没有|吗|没)?|"
    r"(?:删(?:除|掉)?|完成|做完|结束|启动|开始|跑起来|好了)(?:了吗|了没|没有|没)|"
    r"(?:现在)?(?:进度|进展)(?:怎么样|如何|呢)|现在怎么样了)",
    re.I,
)
# These follow-ups ask the host about its present exchange. Without a named
# object, even the best-scoring older conversation cannot identify their target.
_CONTEXT_ACTION = re.compile(
    r"(?:那么?)?(?:[这那][个件](?:问题|事)?)?(?:该|要)?(?:怎么|如何)(?:修(?:复|理)?|解决|处理)"
    r"(?:[这那][个件](?:问题|事)?)?(?:呢|呀|啊)?|"
    r"(?:请)?(?:帮我|帮忙)(?:修(?:复|理)?|处理|解决)(?:一下)?(?:这个(?:问题)?)?|"
    r"(?:然后|先|再)?(?:全部|都)?(?:停[止掉]|升级)(?:一下|吧)?|"
    r"(?:how (?:do|can|should) (?:we|i|you) |(?:please )?)(?:fix|repair|resolve) (?:this|that|it)",
    re.I,
)
_UNANCHORED_STATUS = re.compile(
    r"(?:还有|还剩|剩下)(?:哪些|什么)(?:事(?:情)?|工作|任务)?(?:还)?(?:没有|没|未)(?:做完|完成|做)(?:的)?(?:吗|呢)?|"
    r"what(?:['’]s| is) (?:still )?(?:left(?: to do)?|unfinished)",
    re.I,
)
_PRESENCE = re.compile(r"(?:测试[，,：:\s]+)?(?:你?在吗|有人吗)|(?:test[, ]+)?are you there", re.I)
_CLOSING = " \t\r\n。.!！?？,，;；~～"
_CONVERSATIONAL = re.compile(
    rf"(?:(?P<acknowledgment>{_SOCIAL.pattern})|(?P<session_continuation>{_FOLLOW_THROUGH.pattern})|"
    rf"(?P<session_status>{_SESSION_STATUS.pattern})|(?P<context_action>{_CONTEXT_ACTION.pattern})|"
    rf"(?P<unanchored_followup>{_UNANCHORED_STATUS.pattern})|(?P<presence>{_PRESENCE.pattern}))"
    rf"(?=[{re.escape(_CLOSING)}]|$)",
    re.I,
)
# ASCII punctuation inside a URL, file name or version need not end a clause.
_CLAUSE_END = re.compile(r"[ \t]*(?:[，；。！？]+|[.,;!?]+(?=\s|$)|\r?\n)\s*")
_QUESTION_END = re.compile(r"？+|\?+(?=\s|$)")


class _Clock(Protocol):
    """The recall request's monotonic clock."""

    def monotonic(self) -> float: ...


def _conversation_reason(text: str) -> str | None:
    """Recognize only complete sequences of social, continuation and status intents."""
    tail = text.strip(_CLOSING)
    reasons = set()
    while tail:
        match = _CONVERSATIONAL.match(tail)
        if match is None:
            return None
        reasons.add(match.lastgroup)
        tail = tail[match.end() :].lstrip(_CLOSING)
    return next(
        (
            reason
            for reason in (
                "context_action",
                "unanchored_followup",
                "session_status",
                "session_continuation",
                "presence",
            )
            if reason in reasons
        ),
        "acknowledgment",
    )


def automatic_query_topic(query: str) -> str:
    """Keep the named question verbatim when standalone conversational clauses surround it."""
    start = len(query) - len(query.lstrip())
    topic = 0
    while match := _CONVERSATIONAL.match(query, start):
        # An adjacent clause can supply the object or constraints of an action.
        if match.lastgroup in {"context_action", "session_continuation"}:
            break
        boundary = _CLAUSE_END.match(query, match.end())
        if match.end() == len(query) or (boundary and boundary.end() == len(query)):
            return query
        if boundary is None:
            break
        topic = start = boundary.end()
    # A slash at the start denotes a host command. A path following a preamble
    # is part of the user's question and must not turn into such a command.
    body = query if query.startswith("/", topic) else query[topic:]
    # A follow-up after a statement can supply its only question ("The pump
    # leaks. How do we fix this?"). Trim a tail only after an actual question.
    end = len(body)
    for mark in reversed(tuple(_QUESTION_END.finditer(body))):
        tail = body[mark.end() : end].strip()
        if not tail:
            continue
        if _conversation_reason(tail) not in {"acknowledgment", "unanchored_followup", "session_status", "presence"}:
            break
        end = mark.end()
    return body[:end]


def _prior_prompt(tx, context: TrustedContext, current_source_refs, *, deadline, clock) -> bool:
    """Find one eligible prior prompt in this session within the request's deadline."""
    conn = tx._check()

    def expired():
        return deadline is not None and clock.monotonic() >= deadline

    if expired():
        raise ContractError("DEADLINE_EXCEEDED")
    interruptible = deadline is not None and callable(getattr(conn, "set_progress_handler", None))
    if interruptible:
        conn.set_progress_handler(lambda: int(expired()), 1000)
    excluded = ""
    if current_source_refs:
        excluded = f" AND event_id || '@' || source_revision NOT IN ({','.join('?' for _ in current_source_refs)})"
    try:
        # Read-only access can precede the initialization that supplies this
        # optional index. When present, prefer it over the broad scope/role index.
        indexed = conn.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name='source_session'").fetchone()
        table = "source_events INDEXED BY source_session" if indexed else "source_events"
        for scope in sorted(context.allowed_scope_ids):
            prior = conn.execute(
                f"""SELECT 1 FROM {table} WHERE session_id=? AND scope_id=? AND entry_id=? AND role='user'
                  AND COALESCE(source_original_origin,origin)='human_direct'
                  AND project_id IS ? AND branch_id IS ? AND read_blocked=0{excluded} LIMIT 1""",
                (
                    context.session_id,
                    scope,
                    context.entry_id or "local",
                    context.project_id,
                    context.branch_id,
                    *current_source_refs,
                ),
            ).fetchone()
            if expired():
                raise ContractError("DEADLINE_EXCEEDED")
            if prior is not None:
                return True
    except sqlite3.OperationalError as exc:
        if interruptible and getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
            raise ContractError("DEADLINE_EXCEEDED") from exc
        raise
    finally:
        if interruptible:
            conn.set_progress_handler(None, 0)
    return False


def automatic_recall_route(
    query: str | None,
    *,
    tx=None,
    context: TrustedContext | None = None,
    current_source_refs: tuple[str, ...] = (),
    deadline: float | None = None,
    clock: _Clock = time,
) -> str | None:
    """Choose ordinary recall, a current-task lookup, or a reason to skip injection.

    A first prompt may resume a task from another session. The host can have
    captured it already, so its own source segments do not count as a prior turn.
    """
    text = unicodedata.normalize("NFKC", query or "").strip()
    if not text:
        return "blank_prompt"
    if text.startswith("/"):
        command = text.split(None, 1)[0]
        if len(command) > 1 and "/" not in command[1:] and not command[1:].startswith("."):
            return "slash_command"
    reason = _conversation_reason(text)
    if reason == "presence":
        return "unanchored_followup"
    if reason is None:
        return None
    if tx is None or context is None or not context.allowed_scope_ids:
        return None
    prior = _prior_prompt(tx, context, current_source_refs, deadline=deadline, clock=clock)
    if reason in {"context_action", "unanchored_followup"}:
        return "unanchored_followup" if prior else "current_task_only"
    if reason == "session_continuation" and not prior:
        return "current_task_only"
    return reason if prior else None


def automatic_recall_skip_reason(query: str | None, **session) -> str | None:
    """Expose only reasons to skip to hosts that do not perform retrieval here."""
    route = automatic_recall_route(query, **session)
    return None if route == "current_task_only" else route
