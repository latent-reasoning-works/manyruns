"""Each run keeps its own figures under its run folder (manyruns#25).

The per-dataset `plots/<stem>.png` (+ `.spec.json`, `.spec.npz`) is overwritten by the next run
of the same recipe on the same data. The run folder `state/<run_id>/plots/` keeps a copy per
run, and the step record (`run_plots`) and the run row (`run_plots`) name those copies, so a
reader reaches an older run's figure from the record alone.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from manyruns import figspec, store
from manyruns.pipeline import io as _io
from manyruns.pipeline import runner
from manyruns.pipeline import steps as _steps


@pytest.fixture(autouse=True)
def _in_tmp(monkeypatch, tmp_path):
    """Row paths are cwd-relative, so every test runs from its own temp directory."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def toy_step(monkeypatch):
    """A cheap in-process latent step whose figure differs on every call: call `k` draws
    `X[:, :2] * k`. One recipe and one dataset, two runs, two different figures."""
    calls = {"n": 0}

    def toy(state, g, params, out_dir, plots, target_dim):
        calls["n"] += 1
        state["emb"] = np.asarray(state["X"][:, :2], dtype=float) * calls["n"]
        _io._save_scatter(state["emb"], None, out_dir, "toy.png", plots, "toy")

    monkeypatch.setitem(_steps._INPROC_STEPS, "toy", toy)
    return calls


RECIPE = {"name": "toy-recipe", "steps": [{"name": "toy", "group": "latent"}]}


def _run(X, out_dir):
    results = runner.run_inproc(X, RECIPE, out_dir)
    store.append(results, out_dir="outputs")
    return results


def _rows():
    return [r for r in store.read("outputs")]


def _spec_files(paths):
    json_path = next(p for p in paths if p.endswith(figspec.META))
    npz_path = next(p for p in paths if p.endswith(figspec.ARRAYS))
    return json_path, npz_path


def test_two_runs_keep_their_own_figures_and_plots_holds_the_latest(toy_step):
    X = np.random.default_rng(0).normal(size=(40, 4))
    out_dir = Path("outputs") / "p"
    first = _run(X, out_dir)
    second = _run(X, out_dir)
    assert first["run_id"] != second["run_id"]

    rows = _rows()
    assert [r["run_id"] for r in rows] == [first["run_id"], second["run_id"]]
    coords = []
    for row in rows:
        paths = row["run_plots"]
        run_dir = (out_dir / "state" / row["run_id"] / "plots").as_posix()
        assert paths, "a run that drew a figure names it on its row"
        for p in paths:
            assert not Path(p).is_absolute(), f"row paths are cwd-relative: {p}"
            assert Path(p).parent.as_posix() == run_dir, f"{p} is not under {run_dir}"
            assert Path(p).is_file() and not Path(p).is_symlink(), p
        assert sorted(Path(p).name for p in paths) == [
            "toy.png", "toy.spec.json", "toy.spec.npz"]
        # The step record names the same files as the row.
        step_paths = [p for s in row["steps"] for p in s.get("run_plots") or []]
        assert step_paths == paths
        json_path, npz_path = _spec_files(paths)
        assert json.loads(Path(json_path).read_text())["title"] == "toy"
        with np.load(npz_path, allow_pickle=False) as z:
            coords.append(z["coords"].copy())

    # Each run's spec holds its own values, and they differ.
    np.testing.assert_allclose(coords[0], X[:, :2] * 1)
    np.testing.assert_allclose(coords[1], X[:, :2] * 2)
    assert not np.allclose(coords[0], coords[1])

    # The per-dataset copy is "the latest figure": byte-equal to the second run's.
    latest_dir = out_dir / "plots"
    assert latest_dir.is_dir()
    for p in rows[1]["run_plots"]:
        assert (latest_dir / Path(p).name).read_bytes() == Path(p).read_bytes(), p
    for p in rows[0]["run_plots"]:
        if p.endswith(figspec.ARRAYS):
            assert (latest_dir / Path(p).name).read_bytes() != Path(p).read_bytes()


def test_a_run_with_no_figures_records_an_empty_list(monkeypatch):
    """Behaviour chosen: the run row always carries `run_plots`, `[]` when the run drew
    nothing; the step record omits the key when that step drew nothing, the same rule as
    its `plots`."""
    def silent(state, g, params, out_dir, plots, target_dim):
        state["emb"] = np.asarray(state["X"][:, :1], dtype=float)  # one column: no scatter

    monkeypatch.setitem(_steps._INPROC_STEPS, "silent", silent)
    X = np.random.default_rng(1).normal(size=(30, 3))
    results = runner.run_inproc(X, {"name": "quiet", "steps": [{"name": "silent",
                                                                "group": "latent"}]},
                                Path("outputs") / "p")
    store.append(results, out_dir="outputs")
    (row,) = _rows()
    assert row["run_plots"] == []
    assert all("run_plots" not in s for s in row["steps"])
    state_root = Path("outputs") / "p" / "state"
    if state_root.is_dir():
        assert not list(state_root.glob("*/plots/*"))


def test_the_per_run_copy_survives_when_plots_is_rewritten(toy_step):
    """The per-run file is a real copy, so rewriting the per-dataset file leaves it intact."""
    X = np.random.default_rng(2).normal(size=(25, 3))
    out_dir = Path("outputs") / "p"
    results = _run(X, out_dir)
    png = next(p for p in results["steps"][0]["run_plots"] if p.endswith(".png"))
    before = Path(png).read_bytes()
    (out_dir / "plots" / "toy.png").write_bytes(b"overwritten")
    assert Path(png).read_bytes() == before
