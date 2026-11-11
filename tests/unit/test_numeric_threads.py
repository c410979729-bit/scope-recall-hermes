"""An explicit numerical thread policy always belongs to the operator."""

import pytest


@pytest.mark.parametrize("value", [None, "7", "", "0"])
def test_numeric_thread_default_preserves_explicit_values(monkeypatch, value):
    from scope_recall.core.process_environment import configure_numeric_threads

    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    if value is not None:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", value)
    configure_numeric_threads()
    configure_numeric_threads()
    import os

    assert os.environ["OPENBLAS_NUM_THREADS"] == ("1" if value is None else value)
    assert os.environ["OMP_NUM_THREADS"] == "5"
