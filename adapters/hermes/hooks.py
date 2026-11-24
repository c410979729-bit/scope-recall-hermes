"""Hermes hook registration with one global, instance-aware dispatcher."""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
import weakref
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from ..host_profiles import HOSTS

_log = logging.getLogger(__name__)
_CURRENT_HOOK_DEADLINE: ContextVar[float | None] = ContextVar("_CURRENT_HOOK_DEADLINE", default=None)


def current_hook_deadline() -> float | None:
    """Return the active Hermes callback deadline for the executing thread, if any."""
    return _CURRENT_HOOK_DEADLINE.get()


_OBSERVERS = dict(HOSTS["hermes"].hook_observers)
_SUPPORTED_HOOKS = tuple(_OBSERVERS)
#: Fallback callback allowance when the host SDK cannot supply its configured timeout. Each session lock waits for
#: a bounded share so a busy session cannot stall unrelated callbacks.
_HOST_HOOK_TIMEOUT_S = 30.0
#: The documented upper bound for plugins.hook_callback_timeout.
_HOST_HOOK_TIMEOUT_MAX_S = 600.0
#: The longest a hook waits for its own session, at most a third of the host's timeout.
_SESSION_WAIT_CAP_S = 10.0
_REGISTRY_LOCK = threading.RLock()
_ADAPTERS: weakref.WeakSet[Any] = weakref.WeakSet()
#: When each adapter last bound its session (``initialize``, a session switch), in order: a session's hooks go to
#: the adapter that bound it last.
_BOUND: weakref.WeakKeyDictionary[Any, int] = weakref.WeakKeyDictionary()
_BINDINGS = itertools.count(1)
_REGISTERED_CONTEXTS: weakref.WeakSet[Any] = weakref.WeakSet()
_FALLBACK_CONTEXT_IDS: set[int] = set()


def _register_adapter_instance(adapter: Any) -> None:
    with _REGISTRY_LOCK:
        _ADAPTERS.add(adapter)
        _BOUND[adapter] = next(_BINDINGS)


def _unregister_adapter_instance(adapter: Any) -> None:
    with _REGISTRY_LOCK:
        _ADAPTERS.discard(adapter)


def _profile_matches(adapter: Any, kwargs: dict) -> bool:
    """Compare profile home keys silently before route matching."""
    from pathlib import Path

    identity = getattr(adapter, "_identity", None)
    home = (
        getattr(adapter, "_home", None) if hasattr(adapter, "remote_hook") else getattr(identity, "hermes_home", None)
    )
    if kwargs.get("hermes_home"):
        return home is not None and Path(kwargs["hermes_home"]).resolve() == Path(home).resolve()
    try:
        from agent.secret_scope import (  # pyright: ignore[reportMissingImports]
            current_secret_scope_home,
            is_multiplex_active,
        )
        from hermes_constants import (  # pyright: ignore[reportMissingImports]
            get_hermes_home,
            get_hermes_home_override,
            hermes_home_key,
        )
    except ImportError:
        return True
    caller = get_hermes_home_override() or current_secret_scope_home()
    if caller is None:
        if is_multiplex_active():
            return False
        caller = get_hermes_home()
    return home is not None and hermes_home_key(caller) == hermes_home_key(home)


def _route_key(adapter: Any) -> tuple:
    if hasattr(adapter, "remote_hook"):
        return "remote", adapter.hook_route
    identity = adapter._identity
    return (
        "local",
        identity.binding,
        tuple(sorted(identity.runtime_audience.allowed_scope_ids)),
        identity.local_scope_id,
        identity.read_only,
        identity.scope.chat_type,
        identity.scope.chat_id,
        identity.scope.thread_id,
    )


def _active_adapter(kwargs: dict[str, Any]) -> Any | None:
    """Choose once across local and remote providers; no candidate and ambiguity both refuse dispatch."""
    session = str(kwargs.get(HOSTS["hermes"].session_field) or "").strip()
    if not session:
        return None
    platform = str(kwargs.get("platform") or "").strip().lower()
    sender = str(kwargs.get("sender_id") or "").strip()
    with _REGISTRY_LOCK:
        matches = []
        for adapter in tuple(_ADAPTERS):
            if not getattr(adapter, "_initialized", False) or not _profile_matches(adapter, kwargs):
                continue
            if hasattr(adapter, "remote_hook"):
                matching = adapter.matches_hook(kwargs)
            else:
                identity = getattr(adapter, "_identity", None)
                matching = (
                    identity is not None
                    and identity.session_id == session
                    and (not platform or identity.scope.platform == platform)
                    and (not sender or identity.is_sender(sender))
                )
            if matching:
                matches.append(adapter)
        if not matches or len({_route_key(item) for item in matches}) != 1:
            return None
        return max(matches, key=lambda item: _BOUND.get(item, 0))


def host_hook_timeout() -> float | None:
    """Read the documented hook deadline through Hermes' public configuration API.

    Zero disables the deadline, negative/invalid values use the 30-second default, and positive values are capped
    at the documented 600 seconds. Prefer load_config_readonly, then load_config on hosts without that function;
    unavailable APIs use the default. NaN deliberately uses the default even when the host's resolver returns NaN.
    """
    try:
        try:
            from hermes_cli.config import load_config_readonly  # pyright: ignore[reportMissingImports]
        except ImportError:
            from hermes_cli.config import load_config as load_config_readonly  # pyright: ignore[reportMissingImports]

        plugins = (load_config_readonly() or {}).get("plugins")
        if not isinstance(plugins, dict) or plugins.get("hook_callback_timeout") is None:
            return _HOST_HOOK_TIMEOUT_S
        timeout = float(plugins["hook_callback_timeout"])
    except Exception:
        return _HOST_HOOK_TIMEOUT_S
    if timeout < 0 or math.isnan(timeout):
        return _HOST_HOOK_TIMEOUT_S
    return min(timeout, _HOST_HOOK_TIMEOUT_MAX_S) if timeout > 0 else None


def _input_hook_deadline(started: float, timeout: float | None) -> float | None:
    """Bound pre_llm_call and pre_api_request well inside Hermes' callback timeout so it never disables the hook."""
    if timeout is None:
        return None
    return started + min(timeout * 0.85, max(0.35, min(timeout * 0.7, timeout - min(5.0, timeout * 0.25))))


def _dispatch(event: str, kwargs: dict[str, Any], *, wait: float, deadline: float | None = None) -> Any | None:
    adapter = _active_adapter(kwargs)
    if adapter is None:
        return None
    if hasattr(adapter, "remote_hook"):
        adapter.remote_hook(event, kwargs)
        return adapter
    if event == "post_llm_call":
        # Runs before Hermes sends the reply and writes nothing: it keeps its copy under the adapter's own
        # small lock, which also drops it when a session switch has cleared the turn meanwhile.
        adapter.observe_post_llm_call(**kwargs)
        return adapter
    if event in {"pre_llm_call", "pre_api_request"}:
        adapter._observe_input(
            event, kwargs, wait=wait, deadline=deadline, selected=lambda: _active_adapter(kwargs) is adapter
        )
        return adapter
    # The hook waits for its own session only so long: waited out past the host's timeout, it is abandoned and the host
    # skips this hook for every session (a tool hook queued behind its session's prefetch can wait that long).  One it
    # cannot wait for is counted and said.
    if not adapter._lock.acquire(timeout=wait):
        adapter._calls.busy(event, kwargs)
        return adapter
    try:
        # Session switch can occur after selection; never send that old
        # callback through the replacement audience's identity.
        if _active_adapter(kwargs) is adapter:
            if event == "post_tool_call":
                # Held here exactly once, so the capture's store I/O runs without it and the step's other tool hooks
                # are not kept waiting behind it (``_observe_post_tool_call``).
                with adapter._calls.holding("observe_post_tool_call"):
                    adapter._observe_post_tool_call(**kwargs)
            else:
                getattr(adapter, _OBSERVERS.get(event, "observe_api_request_error"))(**kwargs)
    finally:
        adapter._lock.release()
    return adapter


def _global_callback(event: str) -> Callable[..., None]:
    def callback(**kwargs: Any) -> None:
        started, timeout = time.monotonic(), host_hook_timeout()
        hook_deadline = _input_hook_deadline(started, timeout)
        selected = _active_adapter(kwargs) if event == "pre_api_request" else None
        native = getattr(selected, "_native_requests", None)
        candidate = native.observe(selected, kwargs) if native is not None else None
        token = _CURRENT_HOOK_DEADLINE.set(hook_deadline)
        try:
            adapter = _dispatch(
                event,
                kwargs,
                wait=_SESSION_WAIT_CAP_S if timeout is None else min(_SESSION_WAIT_CAP_S, timeout / 3),
                deadline=hook_deadline if event in {"pre_llm_call", "pre_api_request"} else None,
            )
        finally:
            _CURRENT_HOOK_DEADLINE.reset(token)
        if native is not None and adapter is selected:
            native.finish(candidate)
        elapsed = time.monotonic() - started
        if timeout is not None and elapsed >= timeout:
            _log.warning(
                "scope-recall: %s took %.1f s, past the host's %g s hook timeout; the host skips it for "
                "every session for the next minute",
                event,
                elapsed,
                timeout,
            )
            if adapter is not None and hasattr(adapter, "_calls"):
                adapter._calls.count(f"{event}_overran")

    # Hermes names a callback in its timeout and skip lines; every plugin's closure called ``callback`` read alike.
    callback.__name__ = callback.__qualname__ = f"scope_recall_{event}"
    callback.scope_recall_registration_identity = ("scope-recall", "global-dispatcher")
    return callback


def _context_registered(ctx: Any) -> bool:
    try:
        return ctx in _REGISTERED_CONTEXTS
    except TypeError:
        return id(ctx) in _FALLBACK_CONTEXT_IDS


def _mark_context_registered(ctx: Any) -> None:
    try:
        _REGISTERED_CONTEXTS.add(ctx)
    except TypeError:
        _FALLBACK_CONTEXT_IDS.add(id(ctx))


def register_capture_hooks(ctx: Any, adapter: Any) -> list[str]:
    """Register global callbacks once; adapter selection happens per event."""

    _register_adapter_instance(adapter)
    iter_hook_callbacks = _iter_hook_callbacks()
    if iter_hook_callbacks is None or _context_registered(ctx):
        return []
    registered: list[str] = []
    if callable(getattr(ctx, "register_middleware", None)):
        for kind in ("llm_execution", "tool_execution"):
            ctx.register_middleware(kind, _execution_callback(kind))
    for event in _SUPPORTED_HOOKS:
        callback = _global_callback(event)
        ctx.register_hook(event, callback)
        registered.append(event)
    _mark_context_registered(ctx)
    return registered


def _execution_callback(kind):
    """Use the existing exact profile/session router for public middleware too."""

    def execute(*, next_call, **kwargs):
        adapter = _active_adapter(kwargs)
        native = getattr(adapter, "_native_requests", None)
        payload = kwargs.get("request" if kind == "llm_execution" else "args")
        if native is None:
            return next_call(payload)
        method = native.llm if kind == "llm_execution" else native.tool
        return method(adapter, kwargs, next_call, payload)

    return execute


def update_adapter_binding(adapter: Any) -> None:
    _register_adapter_instance(adapter)


def unregister_adapter(adapter: Any) -> None:
    _unregister_adapter_instance(adapter)


def _iter_hook_callbacks():
    try:
        from hermes_cli.plugins import iter_hook_callbacks  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return iter_hook_callbacks


def unsupported_host_fields() -> dict[str, str]:
    """Documented public gaps for this bounded slice."""

    return {
        "on_session_reset": "unsupported_in_adapter_slice_use_on_session_switch",
        "provider_queue_prefetch": "optional_noop_when_prefetch_is_synchronous",
        "png_raw_attachment_bytes": "unsupported_host_shape_metadata_only",
        "turn_cancelled_hook": "unsupported_public_hook_record_gap_only",
        "turn_interrupted_hook": "unsupported_public_hook_record_gap_only",
    }
