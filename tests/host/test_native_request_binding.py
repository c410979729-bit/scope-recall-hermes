"""Synthetic host seams exercising the production mutation dispatch."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from types import SimpleNamespace

import pytest
from scope_recall.adapters.hermes import hooks
from scope_recall.contracts import ContractError
from scope_recall.core.capture import CaptureReceipt
from scope_recall.core.claim_storage import Claims

from tests.host.test_mutation_authorization import mutation, surface  # noqa: F401

pytestmark = pytest.mark.parametrize("surface", ["hermes"], indirect=True)


class HookContext:
    """Weakly referenced public registration context."""

    def __init__(self, registered, middleware):
        self.register_hook = lambda name, fn: registered.__setitem__(name, fn)
        self.register_middleware = lambda name, fn: middleware.__setitem__(name, fn)


@pytest.fixture
def native_hermes(surface, monkeypatch):
    """Install public callbacks around an isolated provider with synthetic messages."""
    if not hasattr(surface.adapter, "handle_tool_call"):
        pytest.skip("Hermes middleware seam")
    claim, body = mutation(surface, "revise")
    adapter = surface.adapter
    command = "Please correct TEST-project 配色: 银色。"
    adapter.on_turn_start(1, command)
    registered, middleware = {}, {}
    ctx = HookContext(registered, middleware)
    monkeypatch.setattr(hooks, "_iter_hook_callbacks", lambda: ())
    hooks.register_capture_hooks(ctx, adapter)
    ids = dict(
        session_id=adapter._identity.session_id, turn_id="TEST-turn", api_request_id="TEST-api", task_id="TEST-task"
    )
    row = dict(role="user", content=command, message_uid="TEST-uid", _row_id=1)
    request = dict(messages=[dict(role="user", content=command)])
    world = SimpleNamespace(**vars(surface), claim=claim, body=body, command=command, row=row, request=request, ids=ids)

    def observe(rows=None, *, started=1.0, retry=0, opener=None, **fields):
        rows = [row] if rows is None else rows
        request["messages"] = [dict(role=item["role"], content=item["content"]) for item in rows]
        registered["pre_api_request"](
            **dict(ids, **fields),
            user_message=command if opener is None else opener,
            conversation_history=rows,
            request_messages=request["messages"],
            retry_count=retry,
            started_at=started,
        )

    def seal(*, fail=False, **fields):
        def respond(_):
            if fail:
                raise RuntimeError("TEST request failed")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[SimpleNamespace(id="TEST-tool", function=SimpleNamespace(name="revise"))]
                        )
                    )
                ]
            )

        return middleware["llm_execution"](**dict(ids, **fields), request=request, next_call=respond)

    def invoke(**fields):
        return middleware["tool_execution"](
            **dict(ids, **fields),
            tool_name="revise",
            tool_call_id="TEST-tool",
            args=body,
            next_call=lambda args: surface.call("revise", args),
        )

    def evidence():
        with surface.core.storage.read(adapter._tool_context(mutation=True)) as tx:
            source = tx.sources.source_by_event_key(adapter._native_requests.current.source_key)
        body["source_evidence_refs"] = [f"{source.ref}@{source.revision}"]
        return source

    world.observe, world.seal, world.invoke, world.evidence = observe, seal, invoke, evidence
    world.middleware = middleware
    return world


def test_hermes_successful_request_binds_public_revision(native_hermes):
    """A successful native request carries its captured user occurrence into Core."""
    w = native_hermes
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(copy_context().run, w.observe).result()
    w.evidence()
    w.seal()
    result = w.invoke()
    assert "error" not in result, json.dumps(result, ensure_ascii=False)
    assert w.core.current_claim(w.context, w.claim.ref).revision == 2
    assert w.adapter._native_requests.active.get() is None
    assert w.call("revise", w.body)["error"]["field"] == "request_message_binding"


@pytest.mark.parametrize("layout", ["attributes", "mapping"])
def test_hermes_responses_execution_binds_the_actual_tool(native_hermes, layout):
    """Responses transports return output items before the host normalizes them."""
    w = native_hermes
    w.observe()
    w.evidence()
    item = dict(
        type="function_call", call_id="TEST-tool", name="revise", arguments=json.dumps(w.body), status="completed"
    )
    response = dict(status="completed", output=[item])
    if layout == "attributes":
        response = SimpleNamespace(status="completed", output=[SimpleNamespace(**item)])
    request = dict(input=w.request["messages"])
    w.middleware["llm_execution"](**w.ids, request=request, next_call=lambda _: response)
    result = w.invoke()
    assert "error" not in result, result
    assert w.core.current_claim(w.context, w.claim.ref).revision == 2


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_hermes_unfinished_responses_never_seal_authority(native_hermes, status):
    """A terminal failure cannot grant authority merely by carrying a tool item."""
    w = native_hermes
    w.observe()
    w.evidence()
    response = SimpleNamespace(
        status=status, output=[SimpleNamespace(type="function_call", call_id="TEST-tool", name="revise")]
    )
    w.middleware["llm_execution"](**w.ids, request=w.request, next_call=lambda _: response)
    assert w.invoke()["error"]["field"] == "request_message_binding"
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


def test_hermes_revision_receipt_requires_a_fresh_native_attempt(native_hermes):
    """A new API attempt for the same user occurrence can recover its first result."""
    w = native_hermes
    w.body["request_id"] = "TEST-durable-revise"
    w.observe()
    w.evidence()
    w.seal()
    first = w.invoke()
    assert "error" not in first, first
    epoch = w.core.status(w.context).memory_epoch
    assert w.call("revise", w.body)["error"]["field"] == "request_message_binding"
    w.ids["api_request_id"] = "TEST-new-api-attempt"
    w.observe(started=2.0)
    w.seal()
    assert w.invoke() == first
    assert w.core.status(w.context).memory_epoch == epoch
    assert w.core.current_claim(w.context, w.claim.ref).revision == 2


@pytest.mark.parametrize("shape", ["uid_missing", "duplicate_words", "merged", "wire_omitted"])
def test_hermes_ambiguous_request_does_not_borrow_stored_command(native_hermes, shape):
    w = native_hermes
    rows = [dict(w.row)]
    if shape == "uid_missing":
        rows[0].pop("message_uid")
    elif shape == "duplicate_words":
        rows.insert(0, dict(w.row, message_uid="TEST-other-uid"))
    elif shape == "merged":
        rows[0]["_absorbed_message_uids"] = ["TEST-absorbed"]
    w.observe(rows)
    if shape == "wire_omitted":
        w.request["messages"] = []
    w.seal()
    assert w.invoke()["error"]["field"] == "request_message_binding"
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


@pytest.mark.parametrize("later", ["new_request", "failed_retry", "late_hook", "other_session"])
def test_hermes_attempt_seals_cannot_be_reused(native_hermes, later):
    w = native_hermes
    w.observe()
    w.evidence()
    w.seal()
    if later == "late_hook":
        w.seal()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(w.observe, started=2.0).result()
    elif later == "failed_retry":
        w.observe(started=2.0, retry=1)
        with pytest.raises(RuntimeError):
            w.seal(fail=True)
    elif later == "new_request":
        w.observe(started=2.0, api_request_id="TEST-new-api")
    result = w.invoke(**({"session_id": "TEST-other-session"} if later == "other_session" else {}))
    assert result["error"]["field"] == "request_message_binding"
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


def test_hermes_late_failed_attempt_hook_cannot_authorize_a_reused_id(native_hermes):
    w = native_hermes
    with pytest.raises(RuntimeError):
        w.seal(fail=True)
    w.observe()
    w.seal()
    assert w.invoke()["error"]["field"] == "request_message_binding"


def test_hermes_changed_uid_cannot_take_a_trimmed_openers_source(native_hermes):
    w = native_hermes
    w.observe()
    w.evidence()
    w.seal()
    w.ids["api_request_id"] = "TEST-second-api"
    w.observe([dict(w.row, message_uid="TEST-second-occurrence")], started=2.0)
    w.seal()
    assert w.invoke()["error"]["field"] == "request_message_binding"
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


def test_hermes_latest_steer_is_the_authority(native_hermes):
    w = native_hermes
    rows = [
        w.row,
        dict(
            role="user",
            content="[OUT-OF-BAND USER MESSAGE]\nTEST first steer\n[/OUT-OF-BAND USER MESSAGE]",
            display_kind="steer",
            message_uid="TEST-steer1",
            timestamp=1794787201.0,
        ),
        dict(
            role="user",
            content="[OUT-OF-BAND USER MESSAGE]\nKeep the original record.\n[/OUT-OF-BAND USER MESSAGE]",
            display_kind="steer",
            message_uid="TEST-steer2",
            timestamp=1794787202.0,
        ),
    ]
    w.observe(rows)
    w.seal()
    result = w.invoke()
    assert "error" in result
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


def test_hermes_multiple_steers_bind_the_latest_complete_occurrence(native_hermes):
    w = native_hermes
    w.adapter.on_turn_start(2, "TEST opening")
    rows = [
        dict(w.row, content="TEST opening"),
        dict(
            role="user",
            content="[OUT-OF-BAND USER MESSAGE]\nTEST first steer\n[/OUT-OF-BAND USER MESSAGE]",
            display_kind="steer",
            message_uid="TEST-steer1",
            timestamp=1794787201.0,
        ),
        dict(
            role="user",
            content="[OUT-OF-BAND USER MESSAGE]\n" + w.command + "\n[/OUT-OF-BAND USER MESSAGE]",
            display_kind="steer",
            message_uid="TEST-steer2",
            timestamp=1794787202.0,
        ),
    ]
    w.observe(rows, opener="TEST opening")
    source = w.evidence()
    assert source.event["content"] == w.command
    w.seal()
    assert "error" not in w.invoke()
    assert w.core.current_claim(w.context, w.claim.ref).revision == 2


def test_hermes_copied_context_expires_with_dispatch(native_hermes):
    w = native_hermes
    w.observe()
    w.evidence()
    w.seal()
    copied = []

    def capture_context(_):
        bound = w.adapter._native_requests.context(w.adapter, w.adapter._tool_context(mutation=True))
        copied.extend([copy_context(), bound])

    w.middleware["tool_execution"](
        **w.ids, tool_name="revise", tool_call_id="TEST-tool", args=w.body, next_call=capture_context
    )
    with pytest.raises(ContractError, match="request_message_binding"):
        copied[0].run(w.core.revise, copied[1], w.body)
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


def test_hermes_profile_route_cannot_borrow_a_seal(native_hermes):
    w = native_hermes
    w.observe()
    w.evidence()
    w.seal()
    assert w.invoke(hermes_home="TEST-other-profile")["error"]["field"] == "request_message_binding"


def test_core_rechecks_the_bound_source_before_target_lookup(native_hermes, monkeypatch):
    """A newly committed message revokes an earlier proof inside the write transaction."""
    w = native_hermes
    w.observe()
    w.evidence()
    w.seal()
    original = w.core.revise

    def unexpected(*args, **kwargs):
        pytest.fail("target lookup preceded native source validation")

    def revise(context, body, *, request_id=None):
        w.capture("TEST-concurrent-source", "Keep the original record.")
        with monkeypatch.context() as patch:
            patch.setattr(Claims, "versions", unexpected)
            return original(context, body, request_id=request_id)

    monkeypatch.setattr(w.core, "revise", revise)
    result = w.invoke()
    assert result["error"]["field"] == "current_human_evidence"
    assert w.core.current_claim(w.context, w.claim.ref).revision == 1


@pytest.mark.parametrize("disposition", ["queued", "private"])
def test_hermes_uncommitted_capture_cannot_authorize(native_hermes, monkeypatch, disposition):
    w = native_hermes
    receipt = CaptureReceipt(
        "accepted",
        (),
        "queued" if disposition == "queued" else "not_persisted",
        "not_indexed",
        "not_scheduled",
        error_code=None if disposition == "queued" else "PRIVATE_MODE",
    )
    monkeypatch.setattr(w.adapter._writer, "_store", lambda *args, **kwargs: receipt)
    w.observe()
    w.seal()
    assert w.invoke()["error"]["field"] == "request_message_binding"
