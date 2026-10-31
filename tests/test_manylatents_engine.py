"""The manylatents engine (latent steps via manylatents.api.run; MIOFlow as a tool step).

manylatents isn't installed in CI, so we inject a FAKE `manylatents.api` module whose
`run()` records its calls and returns fake embeddings. That lets us assert the two things
that matter about the integration without any heavy deps: group dispatch
(`algorithms={group: name}`) and chaining (`input_data` = the previous step's embeddings).

`cflows`' `mioflow` step is a tool step: it runs the MIOFlow tool through the model seam, here
the stub MIOFlow folder (`tests/fixtures/mioflow_tool/`, numpy only). What the tool step was
handed is the `input.npy` / `time.npy` of its request.
"""
import sys
import types

import pytest

from manyruns import app
from manyruns.pipeline import step_group


def test_group_is_the_only_step_vocabulary():
    """`group` is declared directly — there is no second `kind` axis aliasing onto it."""
    assert step_group({"group": "latent", "name": "phate"}) == "latent"
    assert step_group({"group": "lightning", "name": "mioflow"}) == "lightning"
    assert step_group({"group": "analysis", "name": "separation"}) == "analysis"


def test_a_step_without_a_group_is_not_guessed_at():
    """It used to default to 'latent', so a typo'd step silently ran as an embedding."""
    assert step_group({"name": "phate"}) is None
    assert step_group({"kind": "module", "name": "phate"}) is None  # the old axis is gone


def test_a_grouplessstep_is_skipped_and_reported(tmp_path):
    """The loop must record it, not silently run or silently drop it."""
    np = pytest.importorskip("numpy")
    from manyruns import pipeline

    recipe = {"name": "x", "steps": [{"name": "phate"}]}   # no group
    out = pipeline.run_inproc(np.zeros((6, 3)), recipe, tmp_path)
    assert out["ok"] is False
    assert "no 'group'" in out["status"]["phate"]


def test_server_selects_manylatents_engine():
    from manyruns.serving import LocalServer

    srv = app._server_for("manylatents")
    assert isinstance(srv, LocalServer)
    assert srv.engine == "manylatents"


def _install_fake_manylatents(monkeypatch, calls):
    """Register a fake manylatents.api.run that logs calls and returns fake embeddings."""
    np = pytest.importorskip("numpy")

    def fake_run(**kwargs):
        calls.append(kwargs)
        n = 6
        # return a distinct 2-D embedding each call so chaining is observable
        return {"embeddings": np.arange(n * 2, dtype=float).reshape(n, 2) + len(calls),
                "scores": {"trustworthiness": 0.9}}

    api = types.ModuleType("manylatents.api")
    api.run = fake_run
    pkg = types.ModuleType("manylatents")
    monkeypatch.setitem(sys.modules, "manylatents", pkg)
    monkeypatch.setitem(sys.modules, "manylatents.api", api)
    return calls


def test_lightning_cmd_mirrors_the_cli():
    """The lightning step builds the exact manylatents CLI invocation (with +append)."""
    from manyruns.pipeline import _lightning_cmd

    cmd = _lightning_cmd("mioflow", "swissroll", fast_dev_run=True)
    assert "algorithms/lightning=mioflow" in cmd
    assert "data=swissroll" in cmd
    assert "+trainer.fast_dev_run=true" in cmd
    assert cmd[1:3] == ["-m", "manylatents.main"]
    # full training omits the smoke flag
    assert "+trainer.fast_dev_run=true" not in _lightning_cmd("mioflow", "swissroll", False)


def _stub_mioflow(monkeypatch):
    """The stub MIOFlow folder on the tool search path, run by this interpreter."""
    from pathlib import Path

    stub = Path(__file__).resolve().parent / "fixtures" / "mioflow_tool"
    assert stub.is_dir()
    monkeypatch.delenv("MANYRUNS_TOOL_DIR", raising=False)
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", str(stub))
    monkeypatch.setenv("MANYRUNS_TOOL_MIOFLOW_PYTHON", sys.executable)


def _sent(folder, name):
    """The array a tool step's request carried (`input.npy`, `time.npy`)."""
    np = pytest.importorskip("numpy")
    assert folder.is_dir()
    [path] = list(folder.rglob(name))
    return np.load(path, allow_pickle=False)


def test_latent_via_api_and_mioflow_via_the_tool(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from manyruns import pipeline

    calls: list = []
    _install_fake_manylatents(monkeypatch, calls)
    _stub_mioflow(monkeypatch)
    recipe = app.load_recipe("cflows")

    out = pipeline.run_manylatents(recipe, data_ref="swissroll", out_dir=tmp_path, seed=7,
                                   labels=np.repeat([0.0, 1.0, 2.0], 2), label_kind="time")

    # only the LATENT steps call manylatents.api.run; mioflow runs as a tool step. TWO of them
    # now, not one: manyruns#54 gave `cflows` a declared prep block, so the recipe reads
    # normalize -> transform -> pca(50) -> phate -> mioflow. `pca` is a `latent` step and goes
    # through the same api.run seam, so it is calls[0] and phate is calls[1]. The prep pair
    # never reaches the engine here — on a NAMED dataset the engine loads the matrix itself, so
    # manyruns holds nothing to prep and both steps decline (asserted below).
    assert len(calls) == 2
    assert calls[0]["algorithms"] == {"latent": "pca"}
    assert calls[1]["algorithms"] == {"latent": "phate"}
    assert out["status"]["normalize"].startswith("skipped")
    assert out["status"]["transform"].startswith("skipped")
    assert out["status"]["phate"] == "ok"
    assert out["status"]["mioflow"] == "ok"
    # Asserts the VALUE: `fake_run` offsets by its call count, so phate's output is
    # `arange + 2` and pca's is `arange + 1`. The tool step must be handed phate's.
    assert np.array_equal(_sent(tmp_path, "input.npy"),
                          np.arange(6 * 2, dtype=float).reshape(6, 2) + 2)
    assert out["g_vector"]["mioflow.n_bins"] == 4        # the tool's scalars in the g-vector
    assert out["g_vector"]["phate.trustworthiness"] == 0.9


def test_real_timepoint_labels_reach_mioflow(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from manyruns import pipeline

    _install_fake_manylatents(monkeypatch, [])
    _stub_mioflow(monkeypatch)
    recipe = app.load_recipe("cflows")
    labels = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 2.0])

    # `label_kind` is required for labels to reach MIOFlow at all (the seam sends `time.npy`
    # only for `label_kind == "time"`) — these ARE timepoints, so declaring the kind is
    # honest, not a workaround.
    out = pipeline.run_manylatents(
        recipe, data_ref="eb", out_dir=tmp_path, labels=labels, label_kind="time")

    assert out["status"]["mioflow"] == "ok"
    assert np.array_equal(_sent(tmp_path, "time.npy"), labels)   # real timepoints passed through


def test_mioflow_plot_uses_pre_mioflow_embedding_as_background(tmp_path, monkeypatch):
    """`mioflow.png`'s background scatter must be the embedding the MIOFlow step received
    (phate's output): the trajectories start at true positions in that frame. Regression test
    for the "trajectories in the wrong position" bug, when the in-process route plotted over
    MIOFlow's own post-flow `encode()` output. The tool returns trajectories only, and the runner
    draws them over the embedding the step received."""
    np = pytest.importorskip("numpy")
    from manyruns import pipeline

    _install_fake_manylatents(monkeypatch, [])
    _stub_mioflow(monkeypatch)

    seen_scatter = {}
    real_save_scatter = pipeline.io.save_display_scatter

    def spy_save_scatter(emb, color, out_dir, fname, plots, title, **kwargs):
        if fname == "mioflow.png":
            seen_scatter["emb"] = np.asarray(emb).copy()
            seen_scatter["trajectories"] = kwargs.get("trajectories")
        return real_save_scatter(emb, color, out_dir, fname, plots, title, **kwargs)

    monkeypatch.setattr(pipeline.io, "save_display_scatter", spy_save_scatter)
    recipe = app.load_recipe("cflows")

    out = pipeline.run_manylatents(recipe, data_ref="swissroll", out_dir=tmp_path, seed=7,
                                   labels=np.repeat([0.0, 1.0, 2.0], 2), label_kind="time")

    assert out["status"]["mioflow"] == "ok"
    # `+ 2`, not `+ 1`: fake_run offsets by its own call count, and manyruns#54's prep block put
    # a `pca` latent step ahead of `phate` in `cflows`, so phate is the SECOND api.run call.
    phate_emb = np.arange(6 * 2, dtype=float).reshape(6, 2) + 2  # fake_run's phate-call output
    assert np.array_equal(seen_scatter["emb"], phate_emb), (
        "mioflow.png must be plotted over the embedding the mioflow step received (phate's)")
    assert np.asarray(seen_scatter["trajectories"]).shape == (4, 3, 2)


def test_mioflow_first_without_embedding_is_skipped(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    from manyruns import pipeline

    _install_fake_manylatents(monkeypatch, [])
    _stub_mioflow(monkeypatch)
    # a recipe whose first step is the trajectory tool step → nothing to chain from → skipped
    recipe = {"name": "t", "steps": [{"kind": "model", "name": "mioflow", "group": "lightning",
                                      "tool": "mioflow"}]}
    out = pipeline.run_manylatents(recipe, data_ref="swissroll", out_dir=tmp_path)
    assert out["status"]["mioflow"].startswith("skipped")


def test_run_manylatents_file_uses_input_data(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    from manyruns import pipeline

    calls: list = []
    _install_fake_manylatents(monkeypatch, calls)
    recipe = app.load_recipe("cflows")

    arr = np.zeros((5, 3))
    pipeline.run_manylatents(recipe, array=arr, out_dir=tmp_path)

    # a loaded file/array → the first step to reach the engine passes input_data (not a named
    # dataset). That step is `pca` since manyruns#54 put a prep block at the head of `cflows`;
    # which algorithm it is does not matter here, only that an in-memory array is handed over
    # by value and no dataset name is invented for it.
    assert "input_data" in calls[0]
    assert "data" not in calls[0]
