"""Check that failed test phases are reported before the session ends."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    ("when", "expected"),
    [
        ("call", "FAILED tests/example.py::test_case"),
        ("setup", "FAILED tests/example.py::test_case (setup)"),
        ("teardown", "FAILED tests/example.py::test_case (teardown)"),
    ],
)
def test_failed_phase_is_reported_immediately(request, when, expected):
    conftest = request.config.pluginmanager.get_plugin(str(Path(__file__).resolve().parents[1] / "conftest.py"))
    reporter = Mock()
    item = SimpleNamespace(config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter)))
    report = SimpleNamespace(failed=True, when=when, nodeid="tests/example.py::test_case")
    hook = conftest.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    reporter.write.assert_called_once_with(f"\n{expected}\n", flush=True)


@pytest.mark.parametrize("outcome", ["passed", "skipped"])
def test_non_failed_phase_is_not_reported(request, outcome):
    conftest = request.config.pluginmanager.get_plugin(str(Path(__file__).resolve().parents[1] / "conftest.py"))
    reporter = Mock()
    item = SimpleNamespace(config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter)))
    report = SimpleNamespace(failed=False, outcome=outcome)
    hook = conftest.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    reporter.write.assert_not_called()


def test_failed_phase_without_terminal_reporter(request):
    conftest = request.config.pluginmanager.get_plugin(str(Path(__file__).resolve().parents[1] / "conftest.py"))
    item = SimpleNamespace(config=SimpleNamespace(pluginmanager=SimpleNamespace(get_plugin=lambda name: None)))
    hook = conftest.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: SimpleNamespace(failed=True)))
