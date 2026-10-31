"""The model seam in run mode: the golden adapter, a subprocess, a version 1 result."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

import model_seam_fixtures as _seam
from manyruns.pipeline import external, runner, steps
from model_seam_fixtures import GOLDEN, X12, override_pythons, round_trip_ctx

golden_dir = _seam.golden_dir


def test_golden_adapter_imports_no_scientific_stack():
    """The golden adapter needs numpy only, so CI tests the seam without any model."""
    assert GOLDEN.is_dir(), "golden fixture folder moved"
    code = (f"import sys; sys.path.insert(0, {str(GOLDEN)!r}); import golden_adapter; "
            "import json; print(json.dumps(sorted(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout
    loaded = {m.split(".")[0] for m in json.loads(out)}
    assert "golden_adapter" in loaded
    assert not loaded & {"anndata", "scanpy", "torch", "manylatents"}


SLOTS = ("emb", "clusters", "pseudotime", "trajectories", "graph")


def _round_trip(tmp_path, params=None):
    state, g = runner._new_state(X=X12), {}
    ctx = round_trip_ctx(tmp_path)
    external.run_external_step("golden", params or {}, state, g, ctx)
    return state, g, ctx


def test_golden_round_trip(golden_dir, tmp_path):
    state, g, _ = _round_trip(tmp_path)
    assert state["emb"].shape == (12, 2)
    assert state["clusters"].tolist() == [0, 1] * 6
    assert state["pseudotime"][-1] == 1.0
    assert state["trajectories"].shape == (3, 2, 3)
    assert state["graph"]["edges"].shape == (3, 2)
    assert g["golden.class_probabilities.multi-branching"] == 0.9
    assert g["golden.n_cells"] == 12
    assert state["extras"]["golden.graph.edges"].shape == (3, 2)
    req = (tmp_path / "tool-golden" / "request.json").read_text()
    assert '"result_schema": "result-v1"' in req and '"input_dim": 3' in req


def test_refused_result_changes_no_state(golden_dir, tmp_path):
    state, g = runner._new_state(X=X12), {}
    before = {k: state.get(k) for k in SLOTS}
    with pytest.raises(steps._StepSkipped) as e:
        external.run_external_step("golden", {"truncate": 5}, state, g, round_trip_ctx(tmp_path))
    assert "12" in str(e.value)
    for key in SLOTS:
        assert state.get(key) is before[key], key


def test_undeclared_fact_is_refused(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    cfg = yaml.safe_load((GOLDEN / "golden.yaml").read_text())
    cfg["produces"] = ["clusters"]
    (tools / "golden.yaml").write_text(yaml.safe_dump(cfg))
    (tools / "golden_adapter.py").write_text((GOLDEN / "golden_adapter.py").read_text())
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(tools))
    override_pythons(monkeypatch)
    with pytest.raises(steps._StepSkipped) as e:
        _round_trip(tmp_path / "run")
    assert "embedding" in str(e.value) and "produces" in str(e.value)


def test_reserved_scalar_name_is_refused(golden_dir, tmp_path):
    with pytest.raises(steps._StepSkipped) as e:
        _round_trip(tmp_path, {"scalars": {"tool_origin": "x"}})
    assert "tool_" in str(e.value)


def test_declined_result_is_skipped(golden_dir, tmp_path):
    with pytest.raises(steps._StepSkipped) as e:
        _round_trip(tmp_path, {"decline": "no time axis"})
    assert "no time axis" in str(e.value)


def test_ran_step_is_marked(golden_dir, tmp_path):
    _, g, _ = _round_trip(tmp_path)
    assert g["golden.tool_origin"] == "ran"
    assert g["golden.tool_licence"] == "MIT"
    assert g["golden.tool_commercial_use"] is True


def test_narrowing_clears_seam_slots(golden_dir, tmp_path):
    state, _, ctx = _round_trip(tmp_path)
    rec: dict = {}
    runner._apply_transition({"mask": np.arange(12) >= 2, "axis": "rows"}, state, rec, ctx)
    for key in ("clusters", "graph", "trajectories"):
        assert state[key] is None, key
        assert key in rec["invalidated"], rec


def test_request_carries_time_when_labels_are_time(golden_dir, tmp_path):
    from model_seam_fixtures import T12

    state, g = runner._new_state(X=X12, labels=T12), {}
    state["label_kind"] = "time"
    external.run_external_step("golden", {}, state, g, round_trip_ctx(tmp_path))
    import json

    req = json.loads((tmp_path / "tool-golden" / "request.json").read_text())
    np.testing.assert_array_equal(np.load(req["time"]), T12)
    assert req["n_cells"] == 12
