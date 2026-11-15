"""Hermes plugin entry for the bounded core adapter slice."""

from __future__ import annotations

from typing import Any


def register(ctx: Any) -> Any:
    """Assemble the public provider with its operator setup workflow and capture hooks."""
    from pathlib import Path

    from ...maintenance.setup import configure_hermes
    from ..clients.remote_client import load_client_config
    from .hooks import register_capture_hooks, unsupported_host_fields
    from .provider import ScopeRecallHermesAdapter
    from .remote import RemoteHermesProvider, config_path

    try:
        from hermes_constants import get_hermes_home  # pyright: ignore[reportMissingImports]
    except ImportError:
        import os

        home = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")
    else:
        home = get_hermes_home()
    remote_config = config_path(home)
    if remote_config.is_file():
        remote = RemoteHermesProvider(load_client_config(remote_config))
        ctx.register_memory_provider(remote)
        register_capture_hooks(ctx, remote)
        return remote

    adapter = ScopeRecallHermesAdapter(setup=configure_hermes)
    adapter._diagnostics.unsupported_fields = unsupported_host_fields()
    ctx.register_memory_provider(adapter)
    register_capture_hooks(ctx, adapter)
    return adapter
