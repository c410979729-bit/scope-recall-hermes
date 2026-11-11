"""Stable maintenance migration entry point.

The v1.1 migration contract has one implementation in :mod:`migrate_v2`.
This module remains as the public import and ``python -m maintenance.migrate``
entry point so existing callers do not acquire a second migration engine.
"""

from __future__ import annotations

from scope_recall.core.process_environment import configure_numeric_threads

configure_numeric_threads()

from .migrate_v2 import (  # noqa: E402
    LEGACY_BASELINE,
    MigrationError,
    main,
    migrate_legacy,
)

__all__ = ["LEGACY_BASELINE", "MigrationError", "main", "migrate_legacy"]


if __name__ == "__main__":
    raise SystemExit(main())
