"""The original object and explicit role choices survive workflow routing."""

import numpy as np
import pytest

from manyruns import app, modes, pipeline
from manyruns.pipeline import runner, steps
from manyruns.serving import LocalServer
from manyruns.session import Session


def _original():
    return {"object": np.ones((30, 4)), "role_keys": {"time": "day", "condition": "cond"}}


def _capture(monkeypatch, seen):
    monkeypatch.setitem(steps._ANALYSIS_STEPS, "probe", lambda s, g, p, ctx: seen.append(ctx.copy()))
    monkeypatch.setattr(steps, "_CTX_ANALYSIS_STEPS", frozenset({"probe"}))


def test_read_inputs_returns_the_loaded_object_and_given_role_keys(tmp_path, monkeypatch):
    ad = pytest.importorskip("anndata")
    obj = ad.AnnData(np.ones((30, 4)))
    for key in ("day", "cond", "donor", "batch"):
        obj.obs[key] = np.arange(30) % 3
    path = tmp_path / "input.h5ad"
    obj.write_h5ad(path)
    seen = []
    load = pipeline.load_array

    def capture(path):
        loaded = load(path)
        seen.append(loaded)
        return loaded

    monkeypatch.setattr(pipeline, "load_array", capture)
    loaded = app._read_inputs("manylatents", path, None, "day", condition_key="cond",
                              sample_key="donor", batch_key="batch")
    original = loaded["original_input"]
    assert len(seen) == 1
    assert original["object"] is seen[0]
    assert original["role_keys"] == dict(time="day", condition="cond", sample="donor", batch="batch")
    assert list(original["object"].obs.columns) == ["day", "cond", "donor", "batch"]
    detected = app._read_inputs("manylatents", path, None, None)["original_input"]
    assert all(v is None for v in detected["role_keys"].values())


def test_empty_inputs_have_no_original_input():
    assert app._EMPTY_INPUTS["original_input"] is None


def test_session_puts_original_input_in_ctx(tmp_path):
    original = _original()
    session = Session("probe", engine="_inproc", out_dir=tmp_path, original_input=original)
    assert session.ctx["original_input"] is original


def test_branch_keeps_the_original_input(tmp_path, monkeypatch):
    original = _original()

    def embed(state, g, params, out_dir, plots, target_dim):
        state["emb"] = state["X"][:, :2].copy()

    monkeypatch.setitem(steps._INPROC_STEPS, "phate", embed)
    session = Session("probe", engine="_inproc", out_dir=tmp_path,
                      array=original["object"], original_input=original)
    assert session.step("phate")["outcome"] == "ok"
    assert session.branch(0).ctx["original_input"] is original


@pytest.mark.parametrize("annotated", [False, True])
def test_run_inproc_defaults_to_the_matrix_object(tmp_path, monkeypatch, annotated):
    obj = np.ones((30, 4))
    if annotated:
        obj = pytest.importorskip("anndata").AnnData(obj)
    seen = []
    _capture(monkeypatch, seen)
    runner.run_inproc(obj, {"steps": [{"name": "probe", "group": "analysis"}]}, tmp_path)
    assert seen[0]["original_input"]["object"] is obj
    assert seen[0]["original_input"]["role_keys"] == {}


def test_run_inproc_preserves_an_explicit_original(tmp_path, monkeypatch):
    original, seen = _original(), []
    _capture(monkeypatch, seen)
    runner.run_inproc(np.zeros((30, 4)), {"steps": [{"name": "probe", "group": "analysis"}]},
                      tmp_path, original_input=original)
    assert seen[0]["original_input"] is original


def test_serving_forwards_the_original_input(monkeypatch):
    original, seen = _original(), {}
    monkeypatch.setattr(pipeline, "run_manylatents", lambda **kw: seen.update(kw) or {})
    LocalServer(engine="manylatents").predict({"recipe": {"steps": []}, "original_input": original})
    assert seen["original_input"] is original


def test_modes_run_forwards_the_original_input(monkeypatch):
    original, seen = _original(), {}
    monkeypatch.setattr(app, "run_explorations", lambda *a, **kw: seen.update(kw) or {})
    modes.run("infer", {"recipe": {"steps": []}, "original_input": original})
    assert seen["original_input"] is original


@pytest.mark.parametrize("preloaded", [False, True])
def test_run_explorations_forwards_the_original_input(tmp_path, monkeypatch, preloaded):
    original, seen = _original(), {}
    loaded = dict(app._EMPTY_INPUTS, array=original["object"], original_input=original)
    monkeypatch.setattr(app, "_load_inputs", lambda *a, **kw: loaded)
    monkeypatch.setattr(app, "_live_step_panel", lambda *a: (None, None))

    class Server:
        def predict(self, payload):
            seen.update(payload)
            return {}

    kwargs = dict(array=original["object"], original_input=original) if preloaded else {}
    app.run_explorations(tmp_path, "unknown", server=Server(), recipe={"steps": []}, **kwargs)
    assert seen["original_input"] is original


def test_run_manylatents_keeps_the_original_input(tmp_path, monkeypatch):
    import sys
    import types

    original, seen = _original(), []
    api = types.ModuleType("manylatents.api")
    api.run = lambda **kw: None
    monkeypatch.setitem(sys.modules, "manylatents.api", api)
    _capture(monkeypatch, seen)
    runner.run_manylatents({"steps": [{"name": "probe", "group": "analysis"}]},
                           array=original["object"], out_dir=tmp_path, original_input=original)
    assert seen[0]["original_input"] is original
