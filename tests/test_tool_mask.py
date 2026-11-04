import sys
import tempfile
import types

import numpy as np
import pytest

from manyruns import vocab
from manyruns.pipeline import external, runner, steps
from tool_stubs import FRAME_ADAPTER, make_tool


class MeasurementUnavailable(ValueError):
    """Same name as manylatents' — so `detail` reads as the real record did."""


def _engine(monkeypatch, calls: list) -> None:
    api = types.ModuleType("manylatents.api")

    def run(**kw):
        calls.append(kw)
        n = kw["input_data"].shape[0]
        if kw["algorithms"]["latent"] == "leiden":
            if "metrics" in kw:
                raise MeasurementUnavailable("only 3 distinct points for k=20: need at least k+1")
            labels = np.repeat(np.arange(3, dtype=np.float32), n // 3).reshape(-1, 1)
            return {"embeddings": labels, "scores": {}}
        return {
            "embeddings": np.zeros((n, 2), dtype=np.float32),
            "scores": {m: 0.5 for m in kw.get("metrics", [])},
        }

    api.run = run
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)


@pytest.fixture(autouse=True)
def _temporary_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))


def _tool(tmp_path, monkeypatch):
    make_tool(
        tmp_path,
        monkeypatch,
        "masker",
        {"input": {"kind": "state_frame"}, "output": {"mask": "keep"}, "produces": []},
        FRAME_ADAPTER,
    )


def test_a_mask_tool_narrows_the_run(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    from manyruns import pipeline

    _tool(tmp_path, monkeypatch)
    calls = []
    _engine(monkeypatch, calls)
    X = np.arange(16, dtype=float).reshape(8, 2)
    out = pipeline.run_manylatents(
        {
            "name": "r",
            "steps": [
                {"name": "dbl", "group": "prep", "tool": "masker", "params": {"mask": "alternate"}},
                {"name": "pca", "group": "latent", "params": {}},
            ],
        },
        array=X,
        counts=X,
        genes=np.array(["g0", "g1"]),
        out_dir=tmp_path / "out",
    )
    by_name = {s["name"]: s for s in out["steps"]}
    assert by_name["dbl"]["outcome"] == "ok", by_name["dbl"].get("detail")
    assert calls and calls[-1]["input_data"].shape[0] == 4


def test_a_mask_tool_may_not_declare_other_outputs():
    from manyruns import toolchain

    bad = toolchain.check_tool(
        {
            "name": "x",
            "version": "1",
            "requires": ["x==1"],
            "entrypoint": "a.py",
            "selftest": "pass",
            "input": {"kind": "state_frame"},
            "output": {"mask": "keep", "obsm": ["field"]},
            "produces": ["field"],
        },
        "x",
    )
    assert any("mask" in p for p in bad), bad


def test_a_mask_tool_narrows_a_sparse_run(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    from scipy.sparse import csr_matrix

    from manyruns import pipeline

    _tool(tmp_path, monkeypatch)
    calls = []
    _engine(monkeypatch, calls)
    X = np.arange(16, dtype=float).reshape(8, 2)
    out = pipeline.run_manylatents(
        {
            "name": "r",
            "steps": [
                {"name": "dbl", "group": "prep", "tool": "masker", "params": {"mask": "alternate"}},
                {"name": "pca", "group": "latent", "params": {}},
            ],
        },
        array=X,
        counts=csr_matrix(X),
        genes=np.array(["g0", "g1"]),
        out_dir=tmp_path / "out",
    )
    by_name = {s["name"]: s for s in out["steps"]}
    assert by_name["dbl"]["outcome"] == "ok", by_name["dbl"].get("detail")
    assert calls and calls[-1]["input_data"].shape[0] == 4


@pytest.mark.parametrize("mask", ["reversed", "renamed"])
def test_a_mask_on_other_cells_is_refused(tmp_path, monkeypatch, mask):
    pytest.importorskip("anndata")
    from manyruns import pipeline

    _tool(tmp_path, monkeypatch)
    # The base CI job has no manylatents; the stub engine stands in, as in the tests above.
    _engine(monkeypatch, [])
    X = np.arange(16, dtype=float).reshape(8, 2)
    out = pipeline.run_manylatents(
        {
            "name": "r",
            "steps": [
                {"name": "dbl", "group": "prep", "tool": "masker", "params": {"mask": mask}},
            ],
        },
        array=X,
        counts=X,
        genes=np.array(["g0", "g1"]),
        out_dir=tmp_path / "out",
    )
    by_name = {s["name"]: s for s in out["steps"]}
    assert by_name["dbl"]["outcome"] != "ok"
    assert "cell ids" in by_name["dbl"]["detail"]


def test_a_mask_tool_reads_a_sparse_working_array(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    from scipy.sparse import csr_matrix

    _tool(tmp_path, monkeypatch)
    X = csr_matrix(np.arange(16, dtype=float).reshape(8, 2))
    state = runner._new_state(X=X, counts=X, genes=np.array(["g0", "g1"]))
    transition = external.run_external_step(
        "dbl", {"mask": "alternate"}, state, {}, {"out_dir": tmp_path, "seed": 0, "tool": "masker"}
    )
    assert transition["mask"].tolist() == [True, False] * 4


@pytest.mark.parametrize("sample_ids", [None, np.arange(100, 108)])
def test_frame_export_names_the_selected_cells(tmp_path, sample_ids):
    ad = pytest.importorskip("anndata")
    X = np.arange(16, dtype=float).reshape(8, 2)
    state = runner._new_state(X=X, counts=X, genes=np.array(["g0", "g1"]))
    state["rows"] = np.array([1, 3, 6])
    path = external._input_path(
        {"name": "masker", "input": {"kind": "state_frame"}},
        state,
        {"sample_ids": sample_ids},
        tmp_path,
    )
    expected = ["0", "1", "2"] if sample_ids is None else ["101", "103", "106"]
    assert list(ad.read_h5ad(path).obs_names) == expected


def test_a_mask_tool_narrows_at_plan_time(tmp_path, monkeypatch):
    _tool(tmp_path, monkeypatch)
    assert vocab.step_narrows({"name": "dbl", "group": "prep", "tool": "masker"}) is True
    assert vocab.step_narrows({"name": "dbl", "group": "prep", "tool": "nosuchtool"}) is False


def test_the_narrowing_clears_an_earlier_embedding(tmp_path, monkeypatch):
    _tool(tmp_path, monkeypatch)
    recipe = {
        "name": "r",
        "steps": [
            {"name": "pca", "group": "latent"},
            {"name": "dbl", "group": "prep", "tool": "masker"},
            {"name": "mioflow", "group": "lightning"},
        ],
    }
    assert "embedding" in vocab.unmet(recipe, "time-course")


@pytest.mark.parametrize("mask, word", [("none", "no cell"), ("strings", "bool"), ("short", "8")])
def test_a_bad_mask_is_refused(tmp_path, monkeypatch, mask, word):
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    X = np.arange(16, dtype=float).reshape(8, 2)
    state = runner._new_state(X=X, counts=X, genes=np.array(["g0", "g1"]))
    rows = state["rows"].copy()
    with pytest.raises(steps._StepSkipped, match=word):
        external.run_external_step(
            "dbl", {"mask": mask}, state, {}, {"out_dir": tmp_path, "seed": 0, "tool": "masker"}
        )
    assert np.array_equal(state["rows"], rows)
