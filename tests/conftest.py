"""pytest configuration for the tank_twin test suite.

M0 A1: the package is a scaffold and ships no tests yet. pytest returns exit
code 5 (NO_TESTS_COLLECTED) when a run collects nothing, which fails CI gates
that treat non-zero as failure. Map that single case to 0 so an empty suite is
green. Real failures (exit 1) and usage/internal errors are untouched. This
hook becomes a no-op the moment the first test lands in A6.
"""

from pytest import ExitCode


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = ExitCode.OK
