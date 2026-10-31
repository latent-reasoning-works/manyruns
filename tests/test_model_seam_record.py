"""The run row names its non-commercial steps, and `branch_points` counts a seam graph."""
from __future__ import annotations

import json

import numpy as np

from manyruns import store
from manyruns.pipeline import external, runner, steps
import model_seam_fixtures as _seam
from model_seam_fixtures import IDS12, X12, golden_result, round_trip_ctx

golden_dir = _seam.golden_dir
precomputed_dir = _seam.precomputed_dir


def _row(tmp_path, recipe):
    results = runner.run_inproc(X12, recipe, tmp_path / "out", sample_ids=IDS12)
    store.append(results, out_dir=tmp_path)
    return json.loads(store.index_path(tmp_path).read_text().splitlines()[-1])


def test_precomputed_non_commercial_step_reaches_the_row(precomputed_dir, tmp_path):
    recipe = {"name": "seam_probe", "steps": [
        {"name": "golden_precomputed", "group": "latent", "tool": "golden_precomputed"}]}
    row = _row(tmp_path, recipe)
    assert row["non_commercial_steps"] == ["golden_precomputed"]
    assert any(external.PRECOMPUTED_CAVEAT in c for c in row["caveats"]), row["caveats"]
    keys = list(row)
    assert keys.index("non_commercial_steps") == keys.index("caveats") + 1


def test_commercial_run_has_empty_list(golden_dir, tmp_path):
    recipe = {"name": "p", "steps": [{"name": "golden", "group": "latent", "tool": "golden"}]}
    row = _row(tmp_path, recipe)
    assert row["steps"][0]["outcome"] == "ok", row["steps"][0]
    assert row["non_commercial_steps"] == []


def _golden_graph(tmp_path):
    golden_result(tmp_path / "graph")
    with np.load(tmp_path / "graph" / "graph.npz") as z:
        return {k: z[k] for k in z.files}


def test_golden_graph_branch_counts(golden_dir, tmp_path):
    state, g = runner._new_state(X=X12), {}
    external.run_external_step("golden", {}, state, g, round_trip_ctx(tmp_path))
    steps._step_branch_points(state, g, {}, tmp_path, [], 3)
    assert g["branch_points.n_branch_points"] == 1
    assert g["branch_points.n_endpoints"] == 3
    assert "branch_points.n_branch_points_note" not in g


def test_reeb_keys_win(tmp_path):
    g = {"reeb_graph.structural_summary.n_branch_points": 2,
         "reeb_graph.structural_summary.n_endpoints": 4}
    steps._step_branch_points({"graph": _golden_graph(tmp_path)}, g, {}, tmp_path, [], 3)
    assert g["branch_points.n_branch_points"] == 2
    assert g["branch_points.n_endpoints"] == 4


def test_fallback_honours_source_parameter(tmp_path):
    g = {"reeb_graph.structural_summary.n_branch_points": 5}
    steps._step_branch_points({"graph": _golden_graph(tmp_path)}, g, {"source": "rg2"},
                              tmp_path, [], 3)
    assert g["branch_points.n_branch_points"] == 1
    assert g["branch_points.n_endpoints"] == 3


def test_queue_pr6_empty_state_still_writes_note(tmp_path):
    g: dict = {}
    steps._step_branch_points({}, g, {}, tmp_path, [], 3)
    assert g["branch_points.n_branch_points"] is None
    assert g["branch_points.n_branch_points_note"].startswith("MeasurementUnavailable:")
