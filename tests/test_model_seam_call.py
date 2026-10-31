"""Call mode: a library is usable with a manifest and no adapter code."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from manyruns.pipeline import external, runner, steps
import model_seam_fixtures as _seam
from model_seam_fixtures import GOLDEN, X12, round_trip_ctx

golden_dir = _seam.golden_dir


def _run(name, params, tmp_path):
    state, g = runner._new_state(X=X12), {}
    external.run_external_step(name, params, state, g, round_trip_ctx(tmp_path))
    return state, g


@pytest.fixture()
def toy_importable(golden_dir, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(GOLDEN))
    return golden_dir


def test_function_call_needs_no_adapter(golden_dir, tmp_path):
    state, g = _run("golden_call_svd", {}, tmp_path)
    assert state["emb"].shape == (12, 3)
    expected = np.linalg.svd(X12, full_matrices=False)[0]
    assert np.allclose(np.abs(state["emb"]), np.abs(expected))
    assert g["golden_call_svd.tool_call_target"] == "numpy.linalg:svd"
    assert g["golden_call_svd.tool_origin"] == "ran"


def test_class_method_and_instance_attribute(toy_importable, tmp_path):
    state, _ = _run("golden_call_class", {"k": 4}, tmp_path)
    assert state["clusters"].tolist() == [0, 1, 2, 3] * 3


def test_option_default_applies(toy_importable, tmp_path):
    state, _ = _run("golden_call_class", {}, tmp_path)
    assert state["clusters"].tolist() == [0, 1, 2] * 4


def test_parameter_outside_list_is_refused(toy_importable, tmp_path):
    with pytest.raises(steps._StepSkipped) as e:
        _run("golden_call_class", {"n_init": 5}, tmp_path)
    assert "n_init" in str(e.value) and "k" in str(e.value)


def test_failing_library_becomes_declined_step(toy_importable, tmp_path):
    with pytest.raises(steps._StepSkipped) as e:
        _run("golden_call_class", {"k": 0}, tmp_path)
    message = str(e.value)
    assert message.startswith("golden_call_class declined:")
    assert "ValueError" in message and "k must be >= 1" in message


def test_call_adapter_imports_no_library():
    code = ("import json, sys; import manyruns.adapters.call_adapter; "
            "print(json.dumps(sorted(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout
    loaded = {m.split(".")[0] for m in json.loads(out)}
    assert not loaded & {"sklearn", "torch", "anndata", "call_target"}


def _call(tmp_path, call, params=None, *, time=False, n=12):
    from manyruns.adapters import call_adapter

    tmp_path.mkdir(parents=True, exist_ok=True)
    np.save(tmp_path / "input.npy", X12[:n])
    req = {"params": params or {}, "seed": 0, "input": str(tmp_path / "input.npy"),
           "result": str(tmp_path / "result.json"), "call": call}
    if time:
        np.save(tmp_path / "time.npy", np.zeros(n))
        req["time"] = str(tmp_path / "time.npy")
    (tmp_path / "request.json").write_text(json.dumps(req))
    call_adapter.main(str(tmp_path / "request.json"))
    return json.loads((tmp_path / "result.json").read_text())


def test_missing_slot_is_declined(tmp_path):
    doc = _call(tmp_path, {"target": "numpy:asarray", "args": ["time"],
                           "outputs": {"pseudotime": "return"}})
    assert doc["ok"] is False and "time" in doc["reason"]


def test_non_integral_clusters_are_declined(tmp_path):
    doc = _call(tmp_path, {"target": "numpy:linspace", "kwargs": {}, "options": {
        "start": 0.0, "stop": 1.0, "num": 12}, "outputs": {"clusters": "return"}})
    assert doc["ok"] is False and "clusters" in doc["reason"]


def test_key_and_attribute_selectors(tmp_path):
    doc = _call(tmp_path, {"target": "numpy.linalg:svd", "args": ["matrix"],
                           "options": {"full_matrices": False},
                           "outputs": {"embedding": "return.U"}})
    assert doc["ok"] is True
    assert external.check_result(doc, base=tmp_path, n_cells=12, input_dim=3,
                                 produces=["embedding"]) == []
