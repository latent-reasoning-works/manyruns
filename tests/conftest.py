"""Suite-wide guards. Today there is exactly one, and it is about not touching the developer.

THE TEST RUN MUST NOT WRITE TO A REAL HOME DIRECTORY. `state.roster()` calls
`inspected.put(...)` for every dropped file it reads, and `inspected.HOME` is
`Path.home() / ".manyruns"` — so a suite run on a machine with a populated drop folder
rewrites the developer's own inspection cache as a side effect of asserting on a screen.
Measured before this file existed: a full run mutated `~/.manyruns/inspected.json`.

That was already true before QC; QC made it heavier, because `qc.for_path` merges a whole
block of measurements into the same row. The failure is quiet either way — nothing errors, the
suite is green, and a file in someone's home directory is different afterwards.

Redirecting HOME rather than stubbing the reads is deliberate: a fixture that no-op'd the QC
call would make every test run against a code path no user ever takes, and the cache logic —
which has its own invariants about misses and corruption — would go unexercised. This way the
real code runs, against a directory that evaporates.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def pytest_terminal_summary(terminalreporter):
    """Optional full-stack coverage must not silently disappear under pytest -q."""
    skipped = [report for report in terminalreporter.stats.get("skipped", [])
               if "test_stack_integration.py" in report.nodeid]
    if skipped:
        terminalreporter.write_sep("!", "STACK INTEGRATION NOT RUN", red=True)
        for report in skipped:
            terminalreporter.write_line(str(report.longrepr), red=True)


@pytest.fixture(autouse=True)
def _inspection_cache_is_disposable(tmp_path, monkeypatch):
    """Point the inspection cache at a per-test directory.

    Autouse and unconditional: the tests that touch it are not the ones you would guess — any
    test that constructs the real app, or calls `state.roster()` with no explicit drop folder,
    reaches it through several layers.
    """
    from manyruns import inspected

    monkeypatch.setattr(inspected, "HOME", tmp_path / ".manyruns")


def pytest_collection_modifyitems(items):
    """Repository-only gates cannot run from an sdist, which deliberately omits tooling."""
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir(), "test root moved"
    if not (root / ".git").exists():
        skip = pytest.mark.skip(reason="requires a source checkout (omitted from sdist)")
        for item in items:
            if item.get_closest_marker("source_checkout"):
                item.add_marker(skip)


@pytest.fixture
def synthetic_manifest():
    """Small, invented acquisition catalogue; never used as an application default."""
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "acquire_manifest.tsv"
    assert path.is_file()
    return path
