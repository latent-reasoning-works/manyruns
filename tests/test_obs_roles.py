import numpy as np
import pytest

from tool_stubs import FRAME_ADAPTER, make_tool

from manyruns import modes, pipeline, toolchain
from manyruns.pipeline import external, frame, loading, runner, steps
from tests.test_display_routes import (
    _saved_display_session,
    runtime,  # noqa: F401
    source,  # noqa: F401
)


def _adata(n=4):
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    obs = pd.DataFrame(
        {"donor": ["a", "a", "b", "b"][:n], "lane": ["l1", "l2", "l1", "l2"][:n]},
        index=[f"c{i}" for i in range(n)],
    )
    return anndata.AnnData(X=np.ones((n, 2), dtype="float32"), obs=obs)


def test_obs_roles_of_reads_the_named_columns():
    roles = loading.obs_roles_of(_adata(), sample_key="donor", batch_key="lane")
    assert roles["sample"].tolist() == ["a", "a", "b", "b"] and roles["batch"].tolist() == [
        "l1",
        "l2",
        "l1",
        "l2",
    ]
    assert loading.obs_roles_of(_adata(), sample_key="donor").keys() == {"sample"}


def test_obs_roles_of_refuses_an_absent_column():
    with pytest.raises(ValueError, match="nosuch"):
        loading.obs_roles_of(_adata(), sample_key="nosuch")


def test_the_view_narrows_obs_with_the_rows():
    f = frame.new_frame(counts=np.ones((4, 2)), obs={"sample": np.array(["a", "a", "b", "b"])})
    v = frame.view(f, np.array([0, 2]), frame.full_selection(2))
    assert v["obs"]["sample"].tolist() == ["a", "b"]


def test_infer_carries_obs_to_the_runner(tmp_path, monkeypatch):
    class Seen(Exception):
        pass

    def capture(**kw):
        raise Seen(kw)

    monkeypatch.setattr(pipeline, "run_manylatents", capture)
    obs = {"sample": np.array(["a", "b"])}
    with pytest.raises(Seen) as e:
        modes.run(
            "infer",
            {
                "recipe": {"name": "r", "steps": []},
                "engine": "manylatents",
                "array": np.ones((2, 2)),
                "out_dir": str(tmp_path),
                "obs": obs,
            },
        )
    assert e.value.args[0]["obs"]["sample"].tolist() == ["a", "b"]


def test_obs_roles_are_never_detected():
    assert loading.obs_roles_of(_adata()) == {}
    assert loading.obs_roles_of(np.ones((2, 2))) == {}
    with pytest.raises(ValueError, match="lane"):
        loading.obs_roles_of(np.ones((2, 2)), batch_key="lane")


@pytest.mark.parametrize("engine", ["inproc", "manylatents"])
def test_runners_put_obs_in_the_frame(tmp_path, monkeypatch, engine):
    from manyruns.pipeline import runner

    class Seen(Exception):
        pass

    def capture(recipe, state, *args, **kwargs):
        raise Seen(state)

    monkeypatch.setattr(runner, "_run_steps", capture)
    obs = {"sample": np.array(["a", "b"])}
    matrix = np.ones((2, 2))
    with pytest.raises(Seen) as caught:
        if engine == "inproc":
            runner.run_inproc(matrix, {"name": "r", "steps": []}, tmp_path, obs=obs)
        else:
            runner.run_manylatents(
                {"name": "r", "steps": []}, array=matrix, out_dir=tmp_path, obs=obs
            )
    assert caught.value.args[0]["frame"]["obs"]["sample"].tolist() == ["a", "b"]


def test_session_puts_obs_in_the_frame(tmp_path):
    from manyruns.session import Session

    session = Session(
        "roles", out_dir=tmp_path, array=np.ones((2, 2)), obs={"sample": np.array(["a", "b"])}
    )
    assert session.state["frame"]["obs"]["sample"].tolist() == ["a", "b"]


@pytest.mark.parametrize("filtered", [False, True])
def test_reopen_aligns_source_channels_to_artifact_ids(runtime, source, monkeypatch, filtered):  # noqa: F811
    from manyruns import app, artifacts, figspec
    from manyruns.pipeline import runner

    obj, path = source
    proj, args = _saved_display_session(path)
    args.sample_key = "cell_type"
    args.batch_key = "condition"
    first = app._build_session(proj, args, resume=False)
    rows = np.arange(24)[::3] if filtered else np.arange(24)
    if filtered:
        monkeypatch.setitem(
            runner._prep._PREP_STEPS,
            "cut",
            lambda *a: {"mask": np.arange(24) % 3 == 0, "axis": "rows"},
        )
        first.apply({"name": "cut", "group": "prep"})
    first.step("pca")
    first.close()
    obj[np.arange(24)[::-1]].copy().write_h5ad(path)
    resumed = app._build_session(proj, args)
    np.testing.assert_array_equal(resumed.ctx["sample_ids"], obj.obs_names[rows])
    np.testing.assert_array_equal(resumed.state["X"], obj.X[rows])
    np.testing.assert_array_equal(resumed.state["counts"], obj.X[rows])
    np.testing.assert_array_equal(resumed.labels, obj.obs.timepoint.to_numpy()[rows])
    for role, column in (("sample", "cell_type"), ("batch", "condition")):
        np.testing.assert_array_equal(
            resumed.state["frame"]["obs"][role], obj.obs[column].to_numpy()[rows]
        )
    resumed.step("pca")
    spec = figspec.load(resumed.plots[0])
    np.testing.assert_array_equal(spec["coords"], obj.X[rows, :2])
    np.testing.assert_array_equal(spec["obs_names"], obj.obs_names[rows])
    np.testing.assert_array_equal(spec["color_values"], obj.obs.score.to_numpy()[rows])
    resumed.close()
    manifest = artifacts.read_manifest(resumed.out_dir, resumed.run_id)
    identities = [e["row_identity"]["sample_ids"] for e in manifest.values() if e["slot"] == "emb"]
    assert identities == [obj.obs_names[rows].tolist()]


def test_interactive_run_records_sample_and_batch_keys(runtime, source):  # noqa: F811
    from manyruns import app, store

    obj, path = source
    proj, args = _saved_display_session(path)
    args.sample_key = "cell_type"
    args.batch_key = "condition"
    session = app._build_session(proj, args, resume=False)
    inputs = iter(["pca", "accept", "quit"])
    rc = app.interactive_session(session, read=lambda _="": next(inputs), write=lambda *_: None)
    assert rc == 0
    (run,) = list(store.read("outputs"))
    assert run["sample_key"] == "cell_type" and run["batch_key"] == "condition"


def test_a_branch_carries_its_parent_s_sample_and_batch_keys(runtime, source):  # noqa: F811
    from manyruns import app, store

    obj, path = source
    proj, args = _saved_display_session(path)
    args.sample_key = "cell_type"
    args.batch_key = "condition"
    parent = app._build_session(proj, args, resume=False)
    parent_inputs = iter(["pca", "accept", "quit"])
    app.interactive_session(parent, read=lambda _="": next(parent_inputs), write=lambda *_: None)

    branch = parent.branch(0)
    branch_inputs = iter(["pca", "accept", "quit"])
    rc = app.interactive_session(
        branch, read=lambda _="": next(branch_inputs), write=lambda *_: None
    )
    assert rc == 0
    (branch_row,) = [r for r in store.read("outputs") if r["run_id"] == branch.run_id]
    assert branch_row["sample_key"] == "cell_type" and branch_row["batch_key"] == "condition"


def _roles_tool(tmp_path, monkeypatch, roles):
    make_tool(
        tmp_path,
        monkeypatch,
        "roles",
        {
            "input": {"kind": "state_frame", "obs": roles},
            "output": {"obsm": ["field"]},
            "produces": ["field"],
        },
        FRAME_ADAPTER,
    )


def _state(obs=None, labels=None, kind=None):
    X = np.arange(8, dtype=float).reshape(4, 2)
    state = runner._new_state(X=X, counts=X, genes=np.array(["g0", "g1"]), labels=labels, obs=obs)
    state["label_kind"] = kind
    return state


def test_the_tool_sees_the_roles_of_the_cells_the_run_kept(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    # filter_cells runs in manylatents; the base CI job installs no compute stack.
    pytest.importorskip("manylatents.singlecell.preprocessing")
    _roles_tool(tmp_path, monkeypatch, ["sample"])
    out = runner.run_inproc(
        np.arange(8, dtype=float).reshape(4, 2),
        {
            "name": "r",
            "steps": [
                {"name": "filter_cells", "group": "prep", "params": {"min_genes": 2}},
                {"name": "roles", "group": "lightning", "tool": "roles"},
            ],
        },
        tmp_path / "out",
        counts=np.array([[0, 0], [1, 1], [1, 1], [0, 1]], dtype=float),
        genes=np.array(["g0", "g1"]),
        obs={"sample": np.array(["a", "b", "c", "d"])},
    )
    assert out["g_vector"]["roles.samples"] == "b,c"


def test_condition_comes_from_labels_when_they_are_conditions(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    _roles_tool(tmp_path, monkeypatch, ["condition"])
    g = {}
    external.run_external_step(
        "roles",
        {},
        _state(labels=np.array(["x", "y", "x", "y"]), kind="condition"),
        g,
        {"out_dir": tmp_path, "seed": 0},
    )
    assert g["roles.conditions"] == "x,y"


def test_a_missing_role_skips_the_step_and_names_the_flag(tmp_path, monkeypatch):
    _roles_tool(tmp_path, monkeypatch, ["sample"])
    with pytest.raises(steps._StepSkipped, match="--sample-key"):
        external.run_external_step("roles", {}, _state(), {}, {"out_dir": tmp_path, "seed": 0})


@pytest.mark.parametrize("bad", [["a", "", "b", "b"], ["a", np.nan, "b", "b"]])
def test_a_role_with_missing_values_is_refused(tmp_path, monkeypatch, bad):
    _roles_tool(tmp_path, monkeypatch, ["sample"])
    with pytest.raises(steps._StepSkipped, match="1 cell"):
        external.run_external_step(
            "roles",
            {},
            _state(obs={"sample": np.array(bad, dtype=object)}),
            {},
            {"out_dir": tmp_path, "seed": 0},
        )


def test_input_obs_is_refused_outside_state_frame():
    bad = toolchain.check_tool(
        {
            "name": "x",
            "version": "1",
            "requires": ["x==1"],
            "entrypoint": "a.py",
            "selftest": "pass",
            "input": {"kind": "state_matrix", "obs": ["sample"]},
        },
        "x",
    )
    assert any("obs" in p and "state_frame" in p for p in bad), bad


def test_a_tool_may_need_samples():
    from manyruns import vocab

    assert {"samples", "batch"} <= vocab.facts()


@pytest.mark.parametrize("as_sparse", [False, True])
def test_the_frame_export_keeps_sparse_storage_and_integer_counts(tmp_path, as_sparse):
    anndata = pytest.importorskip("anndata")
    sparse = pytest.importorskip("scipy.sparse")
    from manyruns.pipeline import external, runner

    X = np.array([[2**24 + 1, 0], [1, 3]], dtype=np.int64)
    counts = sparse.csr_matrix(X) if as_sparse else X
    state = runner._new_state(
        X=X, counts=counts, genes=np.array(["g0", "g1"]), layers={"spliced": counts}
    )
    cfg = {"name": "t", "input": {"kind": "state_frame", "layers": ["spliced"]}}
    out = anndata.read_h5ad(external._input_path(cfg, state, {}, tmp_path / "work"))
    for m in (out.X, out.layers["spliced"]):
        assert sparse.issparse(m) == as_sparse
        dense = m.toarray() if sparse.issparse(m) else m
        assert dense.dtype == np.int64 and np.array_equal(dense, X)
