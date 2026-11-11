"""The hook every installed client runs, under the name installed configurations hold (see this package):
``python -m scope_recall.adapters.codex.hook_entry``.  The code is ``adapters/clients/hook_entry.py``."""

from ...core.process_environment import configure_numeric_threads

configure_numeric_threads()

from ..clients.hook_entry import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
