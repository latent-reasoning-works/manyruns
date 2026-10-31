"""The precomputed mode: the result already exists, and manyruns only reads and checks it."""
from __future__ import annotations

import subprocess

import pytest

from manyruns import toolchain
from manyruns.pipeline import external, runner, steps
import model_seam_fixtures as _seam
from model_seam_fixtures import IDS12, X12, golden_result, round_trip_ctx

precomputed_dir = _seam.precomputed_dir

NAME = "golden_precomputed"


def _run(tmp_path, **ctx_changes):
    state, g = runner._new_state(X=X12), {}
    ctx = {**round_trip_ctx(tmp_path / "run"), **ctx_changes}
    try:
        external.run_external_step(NAME, {}, state, g, ctx)
    finally:
        _run.last = (state, g, ctx)
    return state, g, ctx


@pytest.fixture()
def no_subprocess(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a precomputed step started a subprocess")
    monkeypatch.setattr(external.subprocess, "run", refuse)


def test_precomputed_output_enters_the_run(precomputed_dir, tmp_path, no_subprocess):
    state, _, _ = _run(tmp_path)
    assert state["clusters"].tolist() == [0, 1] * 6
    assert state["graph"]["node_ids"].tolist() == [0, 1, 2, 3]


def test_resolve_reports_precomputed(precomputed_dir):
    r = toolchain.resolve(NAME)
    assert r["status"] == "precomputed" and r["python"] is None and r["fix"] is None
    assert str(precomputed_dir / "results") in r["reason"]


def test_undeclared_precomputed_fact_is_refused(precomputed_dir, tmp_path, no_subprocess):
    golden_result(precomputed_dir / "results" / "sample_a")  # all six facts
    with pytest.raises(steps._StepSkipped) as e:
        _run(tmp_path)
    assert "embedding" in str(e.value) and "produces" in str(e.value)


def test_run_without_cell_list_is_refused(precomputed_dir, tmp_path, no_subprocess):
    with pytest.raises(steps._StepSkipped) as e:
        _run(tmp_path, sample_ids=None)
    assert "cell list" in str(e.value)


def test_output_for_other_cells_is_not_found(precomputed_dir, tmp_path, no_subprocess):
    with pytest.raises(steps._StepSkipped) as e:
        _run(tmp_path, sample_ids=IDS12[::-1])
    assert "no precomputed output" in str(e.value)


def test_two_outputs_for_same_cells_are_ambiguous(precomputed_dir, tmp_path, no_subprocess):
    golden_result(precomputed_dir / "results" / "sample_b", facts=("clusters", "graph"))
    with pytest.raises(steps._StepSkipped) as e:
        _run(tmp_path)
    assert "sample_a" in str(e.value) and "sample_b" in str(e.value)


def test_precomputed_step_is_marked(precomputed_dir, tmp_path, no_subprocess):
    _, g, ctx = _run(tmp_path)
    assert g[f"{NAME}.tool_origin"] == "precomputed"
    assert g[f"{NAME}.tool_reproducible"] is False
    assert f"{NAME}.tool_platform" not in g and f"{NAME}.tool_lock" not in g
    assert len(g[f"{NAME}.tool_result_sha256"]) == 64
    assert g[f"{NAME}.tool_licence"] == "CC-BY-NC-4.0"
    assert g[f"{NAME}.tool_commercial_use"] is False
    assert ctx["caveats"] == [f"step {NAME}: " + external.PRECOMPUTED_CAVEAT]


def test_refused_precomputed_step_is_still_recorded(precomputed_dir, tmp_path, no_subprocess):
    path = precomputed_dir / "results" / "sample_a" / "clusters.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(steps._StepSkipped):
        _run(tmp_path)
    _, g, _ = _run.last
    assert g[f"{NAME}.tool_origin"] == "precomputed"


def test_override_python_is_ignored(precomputed_dir, tmp_path, monkeypatch, no_subprocess):
    import sys

    monkeypatch.setenv("MANYRUNS_TOOL_GOLDEN_PRECOMPUTED_PYTHON", sys.executable)
    assert toolchain.resolve(NAME)["status"] == "precomputed"
    state, _, _ = _run(tmp_path)
    assert state["clusters"].tolist() == [0, 1] * 6


def test_subprocess_module_is_the_one_patched():
    assert external.subprocess is subprocess
