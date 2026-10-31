"""Time is declared, never invented; MIOFlow readouts read the `trajectories` slot.

The in-process MIOFlow route used to compute a diffusion pseudotime when the data carried no
time axis, with no recipe step that said so. Now `mioflow` needs `time`: real time labels, or a
`discretize_time` step that bins an ordering the recipe computed (`cflows_inferred`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from manyruns import catalog, vocab
from manyruns.pipeline import runner
from manyruns.pipeline import steps as _steps

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_RECIPES = ROOT / "examples" / "recipes"
X12 = np.arange(36, dtype=float).reshape(12, 3)
T12 = np.repeat([0.0, 1.0, 2.0], 4)


# ── time is declared, never invented ─────────────────────────────────────────

def test_cflows_needs_real_time(monkeypatch):
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    assert vocab.unmet(catalog.load_recipe("cflows"), "manifold") == frozenset({"time"})


def test_cflows_legal_on_time_course(monkeypatch):
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    assert vocab.unmet(catalog.load_recipe("cflows"), "time-course") == frozenset()


def test_tables_declare_time():
    assert vocab.STEP_NEEDS["mioflow"] == ("embedding", "time")
    assert vocab.STEP_PRODUCES["discretize_time"] == ("time",)


@pytest.mark.source_checkout
def test_inferred_recipe_declares_its_axis(monkeypatch):
    assert EXAMPLE_RECIPES.is_dir(), "the example recipe folder moved"
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(EXAMPLE_RECIPES))
    recipe = catalog.load_recipe("cflows_inferred")
    assert vocab.unmet(recipe, "manifold") == frozenset()
    assert recipe["id"] == "sc.traj.mioflow_inferred"
    assert recipe["claims"] == "single-trajectory"
    assert recipe["suits"] == ["manifold", "unknown"]
    assert [s["name"] for s in recipe["steps"]] == [
        "normalize", "transform", "pca", "phate", "dpt", "discretize_time", "mioflow"]
    assert recipe["steps"][-1] == {"name": "mioflow", "group": "lightning", "params": {},
                                   "via": "manylatents"}
    assert "pseudotime" in recipe["question"]


@pytest.mark.source_checkout
def test_discretize_time_keeps_real_time(tmp_path, monkeypatch):
    assert EXAMPLE_RECIPES.is_dir(), "the example recipe folder moved"
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(EXAMPLE_RECIPES))
    received = {}

    def _record(state, g, params, out_dir, plots, target_dim):
        received["labels"] = state.get("labels")
        received["kind"] = state.get("label_kind")

    monkeypatch.setitem(_steps._INPROC_STEPS, "mioflow", _record)
    res = runner.run_inproc(X12, catalog.load_recipe("cflows_inferred"), tmp_path,
                            labels=T12, label_kind="time", seed=0)
    by_name = {s["name"]: s for s in res["steps"]}
    assert by_name["discretize_time"]["outcome"] == "skipped"
    assert "real time labels are already present" in by_name["discretize_time"]["detail"]
    assert received["kind"] == "time"
    assert np.array_equal(np.asarray(received["labels"], dtype=float), T12)


def test_route_refuses_to_invent_time():
    from manyruns.pipeline import mioflow

    before = {m for m in sys.modules if m.split(".")[0] == "manylatents"}
    with pytest.raises(_steps._StepUnsupported) as raised:
        mioflow._run_mioflow_experiment(X12, 0, True, {}, labels=None)
    assert "discretize_time" in str(raised.value)
    assert "time" in str(raised.value)
    after = {m for m in sys.modules if m.split(".")[0] == "manylatents"}
    assert after == before


def test_fallback_is_gone():
    from manyruns.pipeline import mioflow

    assert not hasattr(mioflow, "_TIME_PARAMS")
    assert not hasattr(mioflow, "_diffusion_pseudotime")


# ── MIOFlow readouts read the trajectories slot ──────────────────────────────

STUB = ROOT / "tests" / "fixtures" / "mioflow_tool"
EXAMPLE_TOOL = ROOT / "examples" / "tools" / "mioflow"
TOOLED = {"name": "tooled", "steps": [
    {"name": "pca", "group": "latent"},
    {"name": "mioflow", "group": "lightning", "tool": "mioflow"},
    {"name": "sample_trajectories", "group": "probe"}]}


def test_slot_tables():
    from manyruns.pipeline import probes

    assert probes.PROBE_READS == {"sample_trajectories": "trajectories"}
    assert vocab.STEP_PRODUCES["mioflow"] == ("model", "trajectories")
    assert vocab.STEP_NEEDS["sample_trajectories"] == ("trajectories",)


def test_probe_takes_the_array():
    from manyruns.pipeline import probes

    traj = np.zeros((4, 3, 2))
    traj[-1, :, 0] = [1.0, 2.0, 3.0]
    assert probes._probe_sample_trajectories(traj, {}) == {
        "n_steps": 4, "n_trajectories": 3, "dims": 2,
        "median_displacement": 2.0, "max_displacement": 3.0}


@pytest.mark.source_checkout
def test_stub_manifest_matches_example():
    import yaml

    assert STUB.is_dir(), "the stub MIOFlow folder moved"
    assert EXAMPLE_TOOL.is_dir(), "the MIOFlow example folder moved"
    stub = yaml.safe_load((STUB / "mioflow.yaml").read_text())
    example = yaml.safe_load((EXAMPLE_TOOL / "mioflow.yaml").read_text())
    assert stub == example
    assert stub["commercial_use"] is False


def test_tool_step_gets_same_readouts(tmp_path, monkeypatch):
    assert STUB.is_dir(), "the stub MIOFlow folder moved"
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(STUB))
    monkeypatch.setenv("MANYRUNS_TOOL_MIOFLOW_PYTHON", sys.executable)
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    res = runner.run_inproc(X12, TOOLED, tmp_path, labels=T12, label_kind="time", seed=0)
    g = res["g_vector"]
    assert g["sample_trajectories.n_steps"] == 4, res["steps"]
    assert g["sample_trajectories.n_trajectories"] == 3
    assert g["sample_trajectories.dims"] == 3
    assert any(Path(p).name == "mioflow.png" and Path(p).is_file() for p in res["plots"])


def test_stand_in_skips_probe(tmp_path, monkeypatch):
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    res = runner.run_inproc(X12, catalog.load_recipe("traced"), tmp_path, labels=T12,
                            label_kind="time", seed=0)
    probe = [s for s in res["steps"] if s["name"] == "sample_trajectories"]
    assert [s["outcome"] for s in probe] == ["skipped"]
    assert "trajectories" in probe[0]["detail"]


@pytest.mark.stack_integration
def test_in_process_route_fills_slot(tmp_path, monkeypatch, capsys):
    import importlib.util

    if importlib.util.find_spec("manylatents") is None:
        pytest.skip("manylatents is not installed")
    from manyruns import app, store
    from manyruns.pipeline import mioflow, probes

    captured = []
    real = mioflow.trajectories_of

    def _capture(model):
        out = real(model)
        captured.append(out)
        return out

    monkeypatch.setattr(mioflow, "trajectories_of", _capture)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    code = app.main(["run", "synthetic_timecourse", "--recipe", "traced", "--engine",
                     "manylatents", "--project", "slot-check"])
    assert code == 0, capsys.readouterr()
    row = [r for r in store.read("outputs") if r.get("recipe") == "traced"][-1]
    assert row["ok"], row["steps"]
    assert captured and captured[-1] is not None
    expected = probes._probe_sample_trajectories(captured[-1], {})
    for key, value in expected.items():
        assert row["g_vector"][f"sample_trajectories.{key}"] == pytest.approx(value)
    assert any(Path(p).name == "mioflow.png" for p in row.get("plots") or []) or \
        list(tmp_path.rglob("mioflow.png"))


def test_the_old_time_params_are_unknown():
    """`n_neighbors` and `n_timepoints` built the invented time axis. They are gone with it, so
    a recipe that still sets one gets the existing "unknown param" error."""
    pytest.importorskip("manylatents")
    from manyruns.pipeline import mioflow

    for name in ("n_neighbors", "n_timepoints"):
        with pytest.raises(ValueError, match="unknown param"):
            mioflow._resolve_kwargs({name: 3}, fast_dev_run=True)
