"""Numerical-library defaults for processes owned by Scope Recall."""

import os


def configure_numeric_threads() -> None:
    """Avoid unused OpenBLAS thread buffers before the first NumPy import.

    Vector search already bounds its own scan workers. Explicit environment
    values remain authoritative, and this package's own worker processes inherit the chosen value.
    Host adapters must not call this while loading in another application's process.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
