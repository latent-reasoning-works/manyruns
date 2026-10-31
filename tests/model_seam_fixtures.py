"""Shared fixtures of the model seam tests.

A module the seam tests import, not a second conftest: the fixture folder
`tests/fixtures/model_seam/` holds the golden adapter and its manifests, and these helpers
point manyruns at it. Every tool runs under the current interpreter through the
`MANYRUNS_TOOL_<NAME>_PYTHON` override, so no lock and no tool environment is needed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "model_seam"

X12 = np.arange(36, dtype=float).reshape(12, 3)
IDS12 = np.array([f"c{i:02d}" for i in range(12)])
SHA12 = hashlib.sha256("\n".join(IDS12).encode("utf-8")).hexdigest()
T12 = np.repeat([0.0, 1.0, 2.0], 4)

#: Every manifest in the golden folder that runs in an interpreter.
RUNNABLE = ("golden", "golden_call_svd", "golden_call_class")


def golden_adapter():
    """The golden adapter, loaded by file path (it is not a package module)."""
    assert GOLDEN.is_dir(), "golden fixture folder moved"
    spec = importlib.util.spec_from_file_location("golden_adapter", GOLDEN / "golden_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def golden_result(base: Path, params: "dict | None" = None, **kwargs) -> dict:
    """Write the golden output into `base` and return the golden result document."""
    return golden_adapter().write_outputs(base, X12, IDS12, params or {}, **kwargs)


def override_pythons(monkeypatch) -> None:
    for name in RUNNABLE:
        monkeypatch.setenv(f"MANYRUNS_TOOL_{name.upper()}_PYTHON", sys.executable)


@pytest.fixture()
def golden_dir(monkeypatch):
    """`$MANYRUNS_TOOL_DIR` is the golden folder, and every golden tool runs here."""
    assert GOLDEN.is_dir(), "golden fixture folder moved"
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(GOLDEN))
    override_pythons(monkeypatch)
    return GOLDEN


@pytest.fixture()
def precomputed_dir(tmp_path, monkeypatch):
    """A tool folder with `golden_precomputed.yaml` and one precomputed output for IDS12."""
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy(GOLDEN / "golden_precomputed.yaml", tools / "golden_precomputed.yaml")
    golden_result(tools / "results" / "sample_a", facts=("clusters", "graph"))
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(tools))
    return tools


def round_trip_ctx(out_dir: Path) -> dict:
    return {"out_dir": out_dir, "seed": 0, "sample_ids": IDS12, "caveats": []}
