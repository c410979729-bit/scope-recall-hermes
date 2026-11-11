"""The server for a client on another machine, as its scheduled task runs it, under the name installed configurations
hold (see this package):
``python -m scope_recall.adapters.codex.remote_server``.  The code is ``adapters/clients/remote_server.py``."""

from ...core.process_environment import configure_numeric_threads

configure_numeric_threads()

from ..clients.remote_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
