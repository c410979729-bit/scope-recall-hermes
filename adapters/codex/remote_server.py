"""The server for a client on another machine, as its scheduled task runs it, under the name installed configurations
hold (see this package):
``python -m scope_recall.adapters.codex.remote_server``.  The code is ``adapters/clients/remote_server.py``."""

from ...core.process_environment import configure_numeric_threads

configure_numeric_threads()

from ..clients import remote_server as server  # noqa: E402


def _hermes_hook(config, body, **kwargs):
    from ..hermes.remote_handler import HermesRemoteError, handle_hook

    try:
        return handle_hook(config.home, body)
    except HermesRemoteError as exc:
        raise server.RemoteServerError("invalid Hermes facts") from exc


def _hermes_private(config, body):
    from ..hermes.remote_handler import check_private

    return check_private(config.home, body)


def build_app(config, *, warm=False):
    """Assemble the token-gated transport with the host's existing capture and tool adapter."""
    if config.host != "hermes":
        return server.build_app(config, warm=warm)
    from ..hermes.remote_handler import HermesMCPServer

    return server.build_app(
        config, hook_handler=_hermes_hook, private_handler=_hermes_private, tools_server=HermesMCPServer(config.home)
    )


def main(argv=None):
    """Run the established remote service CLI with the selected host's handlers."""
    return server.main(argv, app_factory=build_app)


if __name__ == "__main__":
    raise SystemExit(main())
