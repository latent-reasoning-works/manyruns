"""Only a context analysis step may serialize the original dataset."""

import os
import pickle

import numpy as np
import pytest

import features_stub
from manyruns.pipeline import features, runner
from manyruns.session import Session


FEATURES = {"name": "dataset_features", "group": "analysis"}
EMBED = {"name": "pca", "group": "latent"}


class ParentOnlyInput:
    def __reduce__(self):
        raise AssertionError("the original input must stay in the parent")


class SerializedCompute:
    """Exercise the child entry point with the same pickle boundary as ComputeProcess."""

    def __init__(self):
        self.requests = []
        self.responses = []

    def call(self, fn, *args, on_step=None, **kwargs):
        self.requests.append((args, kwargs))
        fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs), protocol=5))
        result = fn(*args, **kwargs, on_step=on_step)
        self.responses.append(result)
        return pickle.loads(pickle.dumps(result, protocol=5))


def _embed(name, params, state, g, ctx):
    state["emb"] = state["X"][:, :2].copy()


def _session(tmp_path, original_input):
    session = Session(
        "features", engine="_inproc", out_dir=tmp_path,
        array=np.arange(120, dtype=float).reshape(30, 4),
        recipe={"steps": [EMBED]}, original_input=original_input,
    )
    session.dispatch = dict(session.dispatch, latent=_embed)
    return session


def test_recipe_without_features_never_sends_original_input(tmp_path):
    original = {"object": ParentOnlyInput(), "role_keys": {}}
    session = _session(tmp_path, original)
    session.compute = compute = SerializedCompute()
    session.run_recipe()
    session.step(EMBED)

    assert len(compute.requests) == 2
    assert all(rec["outcome"] == "ok" for rec in session.steps), session.steps
    for _, kwargs in compute.requests:
        assert "original_input" not in kwargs["ctx"]
        assert kwargs.get("original_input") is None
    assert "original_input" not in session.ctx
    assert session.original_input is original


def _checked_rows(data, roles, **kwargs):
    # Check fields unavailable in Session.state, including the unfiltered input.
    assert data.shape == (30, 4)
    np.testing.assert_array_equal(data.layers["counts"], np.full((30, 4), 7))
    np.testing.assert_array_equal(data.raw.X, np.full((30, 4), 11))
    np.testing.assert_array_equal(data.obsm["coordinates"], np.full((30, 2), 13))
    assert data.obs["donor"].tolist() == ["sample"] * 30
    assert data.var["symbol"].tolist() == ["a", "b", "c", "d"]
    assert data.uns == {"source": "original"}
    assert roles["sample"] == "donor"
    assert kwargs == {"layer": "counts", "overrides": {"species": "human"}}
    return features_stub.stub_rows(data, roles, **kwargs)


def _checked_analysis(name, params, state, g, ctx):
    # Install the helper inside the spawned child as well as the parent.
    previous = features._helper
    features._helper = lambda: _checked_rows
    try:
        result = runner._run_analysis_step(name, params, state, g, ctx)
        g["features_process"] = os.getpid()
        return result
    finally:
        features._helper = previous


@pytest.mark.parametrize("placement", ["in_process", "serialized_compute", "spawned_compute"])
def test_interactively_added_features_receives_original_input(tmp_path, placement):
    ad = pytest.importorskip("anndata")
    obj = ad.AnnData(np.ones((30, 4)))
    obj.layers["counts"] = np.full((30, 4), 7)
    obj.raw = ad.AnnData(np.full((30, 4), 11))
    obj.obsm["coordinates"] = np.full((30, 2), 13)
    obj.obs["donor"] = "sample"
    obj.var["symbol"] = ["a", "b", "c", "d"]
    obj.uns["source"] = "original"
    original = {"object": obj, "role_keys": {"sample": "donor"}}
    session = _session(tmp_path, original)
    session.dispatch = dict(session.dispatch, analysis=_checked_analysis)
    if placement == "serialized_compute":
        session.compute = SerializedCompute()
    elif placement == "spawned_compute":
        session.compute = runner.ComputeProcess()
    features_stub.CALLS.clear()
    try:
        session.run_recipe()
        assert session.steps[0]["outcome"] == "ok"
        # An interactive step must still describe the complete original input.
        session.state["X"] = session.state["X"][:15]
        rec = session.step(dict(FEATURES, params={
            "layer": "counts", "overrides": {"species": "human"},
        }))["record"]
        assert rec["outcome"] == "ok", rec.get("detail")
        assert "dataset_features" in rec["artifacts"]
        assert (session.g["features_process"] != os.getpid()) == (placement == "spawned_compute")
        if placement == "in_process":
            assert features_stub.CALLS[-1][0] is obj
        assert "original_input" not in session.ctx
        assert session.original_input is original

        # A subsequent ordinary step must not inherit the transient input.
        assert session.step(EMBED)["outcome"] == "ok"
        if placement == "serialized_compute":
            _, kwargs = session.compute.requests[1]
            assert "original_input" not in kwargs["ctx"]
            assert kwargs["original_input"] is original
            for index in (0, 2):
                _, kwargs = session.compute.requests[index]
                assert "original_input" not in kwargs["ctx"]
                assert kwargs.get("original_input") is None
            for result in session.compute.responses:
                assert "original_input" not in result[2]
    finally:
        if placement == "spawned_compute":
            session.compute.close()


@pytest.mark.parametrize("offloaded", [False, True])
@pytest.mark.parametrize("error, outcome", [(ValueError, "skipped"), (RuntimeError, "error")])
def test_failed_features_does_not_retain_input_in_context(tmp_path, monkeypatch, offloaded,
                                                       error, outcome):
    def fail(data, roles, **kwargs):
        assert data.shape == (30, 4)
        raise error("intentional helper failure")

    monkeypatch.setattr(features, "_helper", lambda: fail)
    original = {"object": np.ones((30, 4)), "role_keys": {}}
    session = _session(tmp_path, original)
    if offloaded:
        session.compute = SerializedCompute()
    rec = session.step(FEATURES)["record"]
    assert rec["outcome"] == outcome, rec.get("detail")
    assert "intentional helper failure" in rec["detail"]
    assert "original_input" not in session.ctx
    assert session.original_input is original
    if offloaded:
        assert "original_input" not in session.compute.responses[0][2]


def test_branch_keeps_original_input_for_later_features(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "_helper", lambda: features_stub.stub_rows)
    original = {"object": np.ones((30, 4)), "role_keys": {}}
    session = _session(tmp_path, original)
    session.run_recipe()
    child = session.branch(0)
    assert child.original_input is original
    assert "original_input" not in child.ctx
    assert child.step(FEATURES)["outcome"] == "ok"
    assert features_stub.CALLS[-1][0] is original["object"]
    assert "original_input" not in child.ctx
