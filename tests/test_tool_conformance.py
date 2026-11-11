from pathlib import Path

import numpy as np
import pytest

from manyruns import toolchain
from manyruns.pipeline import runner
from tool_stubs import FRAME_ADAPTER, TABLE_ADAPTER, make_tool

GENES = np.array(["g0", "g1"])
# Rows 0 and 3 fail the filter, so tools must see only rows 1, 2, 4 and 5.
COUNTS = np.array([[0, 0], [1, 1], [1, 1], [0, 1], [1, 1], [1, 1]], dtype=float)
OBS = {
    "sample": np.array(["a", "b", "c", "d", "e", "f"]),
    "batch": np.array(["x", "x", "y", "y", "x", "y"]),
}
FILTER = {"name": "filter_cells", "group": "prep", "params": {"min_genes": 2}}

CASES = {
    "matrix_field": (
        {"input": {"kind": "state_matrix"}, "output": {"obsm": ["field"]}, "produces": ["field"]},
        FRAME_ADAPTER,
        "lightning",
    ),
    "frame_roles": (
        {
            "input": {"kind": "state_frame", "obs": ["sample", "batch"]},
            "output": {"schema": "result-v1", "tables": {"de": ["gene"]}},
            "produces": [],
        },
        TABLE_ADAPTER,
        "analysis",
    ),
    "mask": (
        {"input": {"kind": "state_frame"}, "output": {"mask": "keep"}, "produces": []},
        FRAME_ADAPTER,
        "prep",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_stub_tool_round_trips_after_a_filter(tmp_path, monkeypatch, case):
    pytest.importorskip("anndata")
    # filter_cells runs in manylatents; the base CI job installs no compute stack.
    pytest.importorskip("manylatents.singlecell.preprocessing")
    manifest, adapter, group = CASES[case]
    make_tool(tmp_path, monkeypatch, "conf", manifest, adapter)
    out = runner.run_inproc(
        COUNTS.copy(),
        {"name": "r", "steps": [FILTER, {"name": "step_x", "group": group, "tool": "conf"}]},
        tmp_path / "out",
        counts=COUNTS,
        genes=GENES,
        obs=OBS,
    )
    step = {s["name"]: s for s in out["steps"]}["step_x"]
    assert step["outcome"] == "ok", step.get("detail")
    g = out["g_vector"]
    assert g["step_x.tool_name"] == "conf"
    for key in ("tool_version", "tool_reproducible", "tool_seconds"):
        assert f"step_x.{key}" in g, key
    if case == "frame_roles":
        assert "step_x.rows" not in g
        assert (tmp_path / "out" / Path(step["artifacts"]["de"])).is_file()
    else:
        assert g["step_x.rows"] == 4


def test_a_source_file_tool_runs_before_a_filter_and_is_refused_after_one(tmp_path, monkeypatch):
    anndata = pytest.importorskip("anndata")
    # filter_cells runs in manylatents; the base CI job installs no compute stack.
    pytest.importorskip("manylatents.singlecell.preprocessing")
    src = tmp_path / "d.h5ad"
    frame = anndata.AnnData(X=COUNTS.copy())
    frame.var_names = GENES
    frame.write_h5ad(src)
    make_tool(
        tmp_path,
        monkeypatch,
        "conf",
        {"input": {"kind": "source_file"}, "output": {"obsm": ["field"]}, "produces": ["field"]},
        FRAME_ADAPTER,
    )
    tool_step = {"name": "conf", "group": "lightning", "tool": "conf"}
    out = runner.run_inproc(
        COUNTS.copy(),
        {"name": "r", "steps": [tool_step]},
        tmp_path / "unfiltered",
        counts=COUNTS,
        genes=GENES,
        source_path=src,
    )
    (step,) = out["steps"]
    assert step["outcome"] == "ok", step.get("detail")
    assert out["g_vector"]["conf.rows"] == 6

    out = runner.run_inproc(
        COUNTS.copy(),
        {"name": "r", "steps": [FILTER, tool_step]},
        tmp_path / "filtered",
        counts=COUNTS,
        genes=GENES,
        source_path=src,
    )
    filtered, step = out["steps"]
    assert filtered["outcome"] == "ok", filtered.get("detail")
    assert filtered["dropped"] == {"axis": "rows", "n": 2, "remaining": 4}
    assert step["outcome"] == "skipped", step.get("detail")
    assert "narrowed" in step["detail"]
    assert "removed" in step["detail"]


def test_every_bundled_manifest_passes_check_tool():
    assert toolchain.tool_dir().is_dir()
    for name in toolchain.discover_tools():
        assert toolchain.check_tool(toolchain.load_tool(name), name) == [], name


def test_a_state_matrix_tool_reads_a_sparse_run(tmp_path):
    # The production loader keeps sparse counts sparse; run_inproc densifies before the state is
    # built, so the export is checked on a state that holds the sparse matrix itself.
    sparse = pytest.importorskip("scipy.sparse")
    from manyruns.pipeline import external

    state = runner._new_state(X=COUNTS, counts=COUNTS, genes=GENES)
    state["X"] = sparse.csr_matrix(COUNTS)
    cfg = {"name": "step_x", "input": {"kind": "state_matrix"}}
    path = external._input_path(cfg, state, {}, tmp_path / "work")
    assert np.array_equal(np.load(path, allow_pickle=False), COUNTS)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
@pytest.mark.parametrize("as_sparse", [False, True])
def test_the_matrix_export_keeps_dtype_and_values(tmp_path, dtype, as_sparse):
    sparse = pytest.importorskip("scipy.sparse")
    from manyruns.pipeline import external

    X = np.array([[2**53 + 1, 0], [1, 3]], dtype=dtype)
    state = runner._new_state(X=COUNTS, counts=COUNTS, genes=GENES)
    state["X"] = sparse.csr_matrix(X) if as_sparse else X
    cfg = {"name": "step_x", "input": {"kind": "state_matrix"}}
    back = np.load(external._input_path(cfg, state, {}, tmp_path / "work"), allow_pickle=False)
    assert back.dtype == dtype and np.array_equal(back, X)
