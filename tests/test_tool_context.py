from pathlib import Path

import numpy as np
import pytest

from manyruns import app, modes, pipeline
from manyruns.pipeline import external, runner, steps
from manyruns.session import Session
from test_choice_keys import _forward_to_mock
from test_choice_keys import store  # noqa: F401
from tool_stubs import FRAME_ADAPTER, make_tool


class _Seen(Exception):
    pass


def _recipe(tool=None):
    step = {"name": "dbl", "group": "prep", "params": {}}
    if tool:
        step["tool"] = tool
    return {"name": "r", "steps": [step]}


def test_spec_id_changes_with_the_tool_only():
    ids = {runner.identity(_recipe(t), engine="mock", seed=0, dataset=None)[1] for t in ("a", "b")}
    assert len(ids) == 2


def test_spec_id_without_a_tool_is_unchanged():
    # Measured on 42160dd to keep tool-free identities stable.
    assert runner.identity(_recipe(), engine="mock", seed=0, dataset=None)[1] == "b4b545cddfd2"


def test_performed_keeps_the_tool(tmp_path):
    session = Session(
        "roles", out_dir=tmp_path, array=np.ones((2, 2)), obs={"sample": np.array(["a", "b"])}
    )
    session.steps.append(
        {"index": 0, "name": "dbl", "group": "prep", "params": {}, "tool": "masker"}
    )
    assert session.performed()["steps"][-1]["tool"] == "masker"


@pytest.mark.parametrize("compute_error", [False, True])
def test_session_apply_performed_keeps_the_tool_and_identity(tmp_path, monkeypatch, compute_error):
    # Isolate record handling from tool execution; failed attempts need the same identity.
    monkeypatch.setattr(external, "run_external_step", lambda *args: None)

    class FailedCompute:
        def call(self, *args, **kwargs):
            raise RuntimeError("compute interrupted")

    ids = set()
    for tool in ("masker", "other"):
        recipe = _recipe(tool)
        session = Session(
            "roles",
            engine="mock",
            seed=0,
            recipe=recipe,
            out_dir=tmp_path / tool,
            array=np.ones((2, 2)),
            obs={"sample": np.array(["a", "b"])},
        )
        if compute_error:
            session.compute = FailedCompute()
        rec = session.apply(recipe["steps"][0])
        assert rec["outcome"] == ("error" if compute_error else "ok")
        assert session.performed()["steps"][-1]["tool"] == tool
        spec_id = session.results()["spec_id"]
        assert spec_id == runner.identity(recipe, engine="mock", seed=0, dataset=None)[1]
        ids.add(spec_id)
    assert len(ids) == 2


def _h5ad(path, n=8):
    import anndata
    import pandas as pd

    obs = pd.DataFrame(
        {"donor": ["a", "b"] * (n // 2), "lane": ["l1"] * n}, index=[f"c{i}" for i in range(n)]
    )
    anndata.AnnData(X=np.arange(n * 3, dtype="float32").reshape(n, 3), obs=obs).write_h5ad(path)
    return path


def test_tool_timeout_reads_the_environment(monkeypatch):
    monkeypatch.setenv("MANYRUNS_TOOL_TIMEOUT", "17")
    assert runner.tool_timeout() == 17
    monkeypatch.setenv("MANYRUNS_TOOL_TIMEOUT", "0")
    with pytest.raises(ValueError, match="MANYRUNS_TOOL_TIMEOUT"):
        runner.tool_timeout()
    monkeypatch.delenv("MANYRUNS_TOOL_TIMEOUT")
    assert runner.tool_timeout() == 7200


@pytest.mark.parametrize("value", ["-1", "abc", "1.5", ""])
def test_tool_timeout_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("MANYRUNS_TOOL_TIMEOUT", value)
    with pytest.raises(ValueError, match="MANYRUNS_TOOL_TIMEOUT"):
        runner.tool_timeout()


def test_a_source_file_tool_runs_in_a_real_run(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    src = _h5ad(tmp_path / "d.h5ad")
    make_tool(
        tmp_path,
        monkeypatch,
        "srcer",
        {"input": {"kind": "source_file"}, "output": {"obsm": ["field"]}, "produces": ["field"]},
        FRAME_ADAPTER,
    )
    X = np.arange(24, dtype=float).reshape(8, 3)
    out = runner.run_inproc(
        X,
        {"name": "r", "steps": [{"name": "srcer", "group": "lightning", "tool": "srcer"}]},
        tmp_path / "out",
        source_path=src,
    )
    assert out["g_vector"]["srcer.rows"] == 8


def test_a_source_file_tool_is_refused_when_the_frame_is_shorter_than_the_file(tmp_path):
    X = np.arange(24, dtype=float).reshape(8, 3)
    src = tmp_path / "source.npy"
    np.save(src, X)
    # A branch rebuilds its frame from surviving cells, but retains the original path.
    X4 = X[:4]
    state = runner._new_state(X=X4, counts=X4, genes=np.array(["a", "b", "c"]))
    ctx = {"source_path": str(src)}
    cfg = {"name": "src", "input": {"kind": "source_file"}}
    with pytest.raises(steps._StepSkipped, match="ORIGINAL file"):
        external._input_path(cfg, state, ctx, tmp_path / "work")


def test_a_source_file_tool_runs_when_the_run_holds_every_cell_of_the_file(tmp_path):
    X = np.arange(24, dtype=float).reshape(8, 3)
    src = tmp_path / "source.npy"
    np.save(src, X)
    state = runner._new_state(X=X, counts=X, genes=np.array(["a", "b", "c"]))
    ctx = {"source_path": str(src)}
    cfg = {"name": "src", "input": {"kind": "source_file"}}
    assert external._input_path(cfg, state, ctx, tmp_path / "work") == Path(ctx["source_path"])


def test_source_cell_count_reads_h5ad_and_npy(tmp_path):
    import anndata

    X = np.arange(15, dtype=float).reshape(5, 3)
    h5ad = tmp_path / "source.h5ad"
    npy = tmp_path / "source.npy"
    anndata.AnnData(X=X).write_h5ad(h5ad)
    np.save(npy, X)
    assert external._source_cell_count(h5ad) == 5 and external._source_cell_count(npy) == 5


@pytest.mark.parametrize("suffix", [".h5ad", ".npy"])
@pytest.mark.parametrize("failure", ["missing", "corrupt", "unreadable"])
def test_a_source_file_tool_refuses_an_unavailable_source(tmp_path, monkeypatch, suffix, failure):
    src = tmp_path / f"source{suffix}"
    if failure == "corrupt":
        src.write_bytes(b"not a readable source file")
    elif failure == "unreadable":
        # Simulate permissions so the refusal also holds when tests run as an administrator.
        def deny_read(*args, **kwargs):
            raise PermissionError("source is unreadable")

        if suffix == ".h5ad":
            import anndata

            monkeypatch.setattr(anndata, "read_h5ad", deny_read)
        else:
            monkeypatch.setattr(np, "load", deny_read)
    X = np.arange(24, dtype=float).reshape(8, 3)
    state = runner._new_state(X=X, counts=X, genes=np.array(["a", "b", "c"]))
    cfg = {"name": "src", "input": {"kind": "source_file"}}
    with pytest.raises(steps._StepSkipped, match="restore or replace") as refused:
        external._input_path(cfg, state, {"source_path": str(src)}, tmp_path / "work")
    assert str(src) in str(refused.value)


def test_read_inputs_names_the_file_it_loaded(tmp_path):
    pytest.importorskip("anndata")
    src = _h5ad(tmp_path / "d.h5ad")
    assert app._read_inputs("manylatents", src, None, None)["source_path"] == str(src)


def test_infer_carries_the_source_path_to_the_runner(tmp_path, monkeypatch):
    def capture(**kw):
        raise _Seen(kw)

    monkeypatch.setattr(pipeline, "run_manylatents", capture)
    with pytest.raises(_Seen) as e:
        modes.run(
            "infer",
            {
                "recipe": {"name": "r", "steps": []},
                "engine": "manylatents",
                "array": np.ones((4, 2)),
                "out_dir": str(tmp_path),
                "source_path": "x.h5ad",
            },
        )
    assert e.value.args[0]["source_path"] == "x.h5ad"


def test_run_explorations_names_the_file_it_loads_itself(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    src = _h5ad(tmp_path / "d.h5ad")

    def capture(**kw):
        raise _Seen(kw)

    monkeypatch.setattr(pipeline, "run_manylatents", capture)
    with pytest.raises(_Seen) as e:
        app.run_explorations(
            src, "scrna", recipe={"name": "r", "steps": []}, engine="manylatents", out_dir=tmp_path
        )
    assert e.value.args[0]["source_path"] == str(src)


def test_the_cli_request_carries_the_source_path(store, monkeypatch):  # noqa: F811
    seen = _forward_to_mock(monkeypatch)
    data = store.parent / "twoarms.h5ad"
    assert data.is_file()
    assert (
        app.main(
            ["run", str(data), "--recipe", "embed", "--engine", "manylatents", "--project", "src"]
        )
        == 0
    )
    assert seen[0]["source_path"] == str(data)


def test_a_session_carries_the_tool_context(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYRUNS_TOOL_TIMEOUT", "11")
    session = Session(
        project="p",
        engine="mock",
        modality="scrna",
        out_dir=tmp_path,
        recipe=app.load_recipe("cflows"),
        source_path=tmp_path / "d.h5ad",
    )
    assert session.ctx["tool_timeout"] == 11
    assert session.ctx["source_path"] == tmp_path / "d.h5ad"


@pytest.mark.parametrize("order, refused", [(["a", "b", "c"], False), (["b", "a", "c"], True)])
def test_a_source_file_tool_checks_the_file_s_cell_ids_and_their_order(tmp_path, order, refused):
    # A reopened run is aligned to its saved ids; the file may hold the same cells reordered.
    anndata = pytest.importorskip("anndata")
    X = np.arange(9, dtype=float).reshape(3, 3)
    src = tmp_path / "source.h5ad"
    adata = anndata.AnnData(X=X)
    adata.obs_names = ["a", "b", "c"]
    adata.write_h5ad(src)
    state = runner._new_state(X=X, counts=X, genes=np.array(["g0", "g1", "g2"]))
    ctx = {"source_path": str(src), "sample_ids": np.array(order)}
    cfg = {"name": "src", "input": {"kind": "source_file"}}
    if refused:
        with pytest.raises(steps._StepSkipped, match="cell ids"):
            external._input_path(cfg, state, ctx, tmp_path / "work")
    else:
        assert external._input_path(cfg, state, ctx, tmp_path / "work") == src
