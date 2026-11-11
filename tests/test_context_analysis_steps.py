"""Analysis readouts can opt into the full step context."""

import builtins

import numpy as np
import pytest

from manyruns import installation, vocab
from manyruns.pipeline import runner, steps
from manyruns.session import Session


def _probe(monkeypatch, seen):
    monkeypatch.setitem(steps._ANALYSIS_STEPS, "probe", lambda s, g, p, ctx: seen.append(ctx.copy()))
    monkeypatch.setattr(steps, "_CTX_ANALYSIS_STEPS", frozenset({"probe"}))


def test_a_context_step_receives_the_step_context(tmp_path, monkeypatch):
    seen = []
    _probe(monkeypatch, seen)
    out = runner.run_inproc(np.ones((30, 4)),
                            {"steps": [{"name": "probe", "group": "analysis"}]}, tmp_path)
    assert out["steps"][0]["outcome"] == "ok"
    assert seen[0]["step_index"] == 0
    assert seen[0]["run_id"] == out["run_id"]


def test_a_context_step_runs_with_no_working_matrix(tmp_path, monkeypatch):
    seen = []
    _probe(monkeypatch, seen)
    session = Session("probe", engine="_inproc", out_dir=tmp_path)
    assert session.state["X"] is session.state["emb"] is None
    assert session.step({"name": "probe", "group": "analysis"})["outcome"] == "ok"
    assert seen[0]["run_id"] == session.run_id


def test_dataset_features_is_legal_on_every_shape():
    for shape in vocab.SHAPES:
        assert vocab.unmet({"steps": [{"name": "dataset_features", "group": "analysis"}]},
                           shape) == frozenset()
    assert "dataset_features" not in vocab.STEP_NEEDS
    assert "dataset_features" not in vocab.STEP_PRODUCES


@pytest.mark.parametrize("engine", ["_inproc", "manylatents"])
def test_dataset_features_is_dispatchable(engine):
    assert runner.dispatchable(engine, "dataset_features", "analysis")


def test_without_the_helper_the_step_is_unsupported(tmp_path, monkeypatch):
    from manyruns.pipeline import features

    def unavailable():
        raise steps._StepUnsupported("dataset_features needs the helper; " + installation.RUNTIME_REPAIR)

    monkeypatch.setattr(features, "_helper", unavailable)
    out = runner.run_inproc(np.ones((30, 4)),
                            {"steps": [{"name": "dataset_features", "group": "analysis"}]}, tmp_path)
    rec = out["steps"][0]
    assert rec["outcome"] == "skipped"
    assert rec["unsupported"] is True
    assert installation.RUNTIME_REPAIR in rec["detail"]


def test_helper_import_failure_has_the_repair(monkeypatch):
    from manyruns.pipeline import features

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "manylatents.singlecell.dataset_features":
            raise ImportError("helper unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(steps._StepUnsupported, match="singlecell.dataset_features") as exc:
        features._helper()
    assert installation.RUNTIME_REPAIR in str(exc.value)
