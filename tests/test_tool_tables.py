import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from manyruns import toolchain
from manyruns.pipeline import external, runner, steps
from tool_stubs import TABLE_ADAPTER, make_tool


@pytest.fixture(autouse=True)
def _local_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))


GENES = np.array(["g0", "g1", "g2"])


def _tool(tmp_path, monkeypatch, tables=None):
    make_tool(
        tmp_path,
        monkeypatch,
        "detab",
        {
            "input": {"kind": "state_frame"},
            "output": {
                "schema": "result-v1",
                "tables": tables or {"de": ["gene", "log2fc", "padj"]},
            },
            "produces": [],
        },
        TABLE_ADAPTER,
    )


def _state():
    X = np.arange(12, dtype=float).reshape(4, 3)
    return runner._new_state(X=X, counts=X, genes=GENES)


def _run(tmp_path, params=None, state=None, g=None):
    state, g = state or _state(), ({} if g is None else g)
    ctx = {"out_dir": tmp_path / "out", "seed": 0, "run_id": "r1", "step_index": 0, "caveats": []}
    external.run_external_step("detab", params or {}, state, g, ctx)
    return state, g


def test_a_declared_table_is_checked_saved_and_recorded(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    state, g = _run(tmp_path)
    assert g["detab.de.rows"] == 2 and g["detab.de.columns"] == ["gene", "log2fc", "padj"]
    path = state["tables"]["de"]
    assert path.endswith(".tsv") and open(path).readline().split("\t")[0] == "gene"


@pytest.mark.parametrize(
    "params, word",
    [
        ({"extra": True}, "extra"),
        ({"drop": "padj"}, "padj"),
        ({"bad_sha": True}, "sha256"),
        ({"genes": ["g0", "NOTAGENE"]}, "NOTAGENE"),
        ({"no_tables": True}, "facts"),
    ],
)
def test_a_bad_table_result_is_refused_and_changes_nothing(tmp_path, monkeypatch, params, word):
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    state = _state()
    with pytest.raises(steps._StepSkipped, match=word):
        _run(tmp_path, params, state)
    assert state["tables"] == {}


def test_a_gene_a_filter_removed_is_refused(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    state = _state()
    state["cols"] = np.array([0, 1])
    with pytest.raises(steps._StepSkipped, match="g2"):
        _run(tmp_path, {"genes": ["g0", "g2"]}, state)


@pytest.mark.parametrize("genes", [["7", "007"], ["NA", "g1"]])
def test_gene_identifiers_are_read_as_literal_strings(tmp_path, monkeypatch, genes):
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    X = np.arange(8, dtype=float).reshape(4, 2)
    state = runner._new_state(X=X, counts=X, genes=np.array(genes))
    state, g = _run(tmp_path, None, state)
    assert g["detab.de.rows"] == 2


def test_output_tables_is_refused_on_the_h5ad_path():
    bad = toolchain.check_tool(
        {
            "name": "x",
            "version": "1",
            "requires": ["x==1"],
            "entrypoint": "a.py",
            "selftest": "pass",
            "input": {"kind": "state_frame"},
            "output": {"tables": {"de": ["gene"]}},
        },
        "x",
    )
    assert any("tables" in p and "result-v1" in p for p in bad), bad


def test_a_table_only_result_is_valid_and_an_empty_one_is_not():
    base = {"schema_version": 1, "ok": True, "n_cells": 4}
    assert not any(
        "facts" in p
        for p in external.check_result(
            base | {"facts": {}, "tables": {"de": {"path": "de.tsv", "sha256": "0" * 64}}},
            produces=[],
            tables={"de": ["gene"]},
            check_files=False,
        )
    )
    assert any("facts" in p for p in external.check_result(base | {"facts": {}}, produces=[]))


def test_the_step_record_and_the_completion_manifest_name_the_table(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    X = np.arange(12, dtype=float).reshape(4, 3)
    out = runner.run_inproc(
        X,
        {"name": "r", "steps": [{"name": "detab", "group": "analysis", "tool": "detab"}]},
        tmp_path / "out",
        counts=X,
        genes=GENES,
    )
    (step,) = out["steps"]
    assert step["outcome"] == "ok", step.get("detail")
    path = step["artifacts"]["de"]
    assert (tmp_path / "out").is_dir()
    complete = next((tmp_path / "out").rglob("COMPLETE"))
    assert path in complete.read_text()
    assert json.loads(complete.read_text())[path]["slot"] == "table:de"


@pytest.mark.parametrize("facts", [{}, None])
def test_facts_may_be_absent_only_for_table_only_manifests(facts):
    doc = {
        "schema_version": 1,
        "ok": True,
        "n_cells": 4,
        "tables": {"de": {"path": "de.tsv", "sha256": "0" * 64}},
    }
    if facts is not None:
        doc["facts"] = facts
    assert external.check_result(doc, produces=[], tables={"de": ["gene"]}) == []
    assert any(
        "facts" in problem
        for problem in external.check_result(doc, produces=["embedding"], tables={"de": ["gene"]})
    )


@pytest.mark.parametrize("path", ["../de.tsv", "/de.tsv", "de.npy", "sub/../de.tsv", "d\x00e.tsv"])
def test_table_paths_are_checked_without_reading_files(path):
    doc = {
        "schema_version": 1,
        "ok": True,
        "n_cells": 4,
        "tables": {"de": {"path": path, "sha256": "0" * 64}},
    }
    assert any(
        "path" in problem
        for problem in external.check_result(
            doc, produces=[], tables={"de": ["gene"]}, check_files=False
        )
    )


@pytest.mark.parametrize("tables", [[], {"de": "gene"}, {"../de": ["gene"]}, {"de": [7]}])
def test_malformed_table_declarations_are_refused(tables):
    bad = toolchain.check_tool(
        {
            "name": "x",
            "version": "1",
            "requires": ["x==1"],
            "entrypoint": "a.py",
            "selftest": "pass",
            "input": {"kind": "state_frame"},
            "output": {"schema": "result-v1", "tables": tables},
            "produces": [],
        },
        "x",
    )
    assert any("tables" in problem for problem in bad)


def test_nondict_scalars_with_tables_is_refused_not_crashed():
    # scalars=1 is truthy, so `doc.get("scalars") or {}` does not fall back to {} and a plain
    # membership test against it raised TypeError instead of refusing. The structural check
    # already flags the bad scalars; the table check must not crash before that reaches the
    # caller.
    doc = {
        "schema_version": 1,
        "ok": True,
        "n_cells": 4,
        "facts": {},
        "tables": {"de": {"path": "de.tsv", "sha256": "0" * 64}},
        "scalars": 1,
    }
    bad = external.check_result(doc, produces=[], tables={"de": ["gene"]}, check_files=False)
    assert any("scalars" in problem for problem in bad)


RE_FORMATTED_TABLE_ADAPTER = """
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad

req = json.loads(open(sys.argv[1]).read())
frame = ad.read_h5ad(req["input"])
# Written with three decimal places on purpose: pandas re-serializes the float as "1.0"
# once it has been parsed back into a DataFrame, so the saved file's bytes differ from
# the bytes this adapter wrote and hashed.
text = "gene\\tlog2fc\\tpadj\\n" + "\\n".join(f"g{i}\\t1.000\\t0.010" for i in range(2)) + "\\n"
path = Path(req["result"]).parent / "de.tsv"
path.write_text(text)
result = {
    "schema_version": 1,
    "ok": True,
    "n_cells": frame.n_obs,
    "facts": {},
    "tables": {"de": {"path": "de.tsv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
}
json.dump(result, open(req["result"], "w"))
"""


def test_the_recorded_sha256_matches_the_saved_file_not_the_tool_s_original(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    make_tool(
        tmp_path,
        monkeypatch,
        "refmt",
        {
            "input": {"kind": "state_frame"},
            "output": {"schema": "result-v1", "tables": {"de": ["gene", "log2fc", "padj"]}},
            "produces": [],
        },
        RE_FORMATTED_TABLE_ADAPTER,
    )
    state, g = {}, {}
    state.update(_state())
    ctx = {"out_dir": tmp_path / "out", "seed": 0, "run_id": "r1", "step_index": 0, "caveats": []}
    external.run_external_step("refmt", {}, state, g, ctx)
    path = state["tables"]["de"]
    saved_sha256 = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert g["refmt.de.sha256"] == saved_sha256
    # The tool's declared hash described bytes that no longer exist on disk.
    assert "1.000" not in open(path).read()


def test_a_cancelled_attempt_does_not_leave_its_table_in_the_restored_baseline(
    tmp_path, monkeypatch
):
    # tune.run_tune_loop snapshots `baseline_state = dict(session.state)` once, BEFORE the
    # first attempt even runs, then restores state on cancel with `state.clear();
    # state.update(baseline_state)` (manyruns/tune.py:502, 520-529). That snapshot is shallow,
    # so baseline_state["tables"] is the SAME dict object as state["tables"] at that moment —
    # mutating it in place during the merge would corrupt the baseline too, and cancelling the
    # very first attempt would fail to remove the table it wrote.
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    state = _state()
    baseline = dict(state)
    external.run_external_step(
        "detab",
        {},
        state,
        {},
        {"out_dir": tmp_path / "out", "seed": 0, "run_id": "r1", "step_index": 0, "caveats": []},
    )
    assert state["tables"]["de"]
    state.clear()
    state.update(baseline)
    assert state["tables"] == {}


def test_a_hash_read_failure_after_save_is_refused_not_errored(tmp_path, monkeypatch):
    # If the saved file can be written but not read back for hashing, the review requires the
    # step to raise _StepSkipped (the refusal exception) and to leave NO measurements or state
    # behind — otherwise the record carries partial table metadata whose sha256 never arrived.
    # Reject only the saved-state path, so check_result's initial read of the tool's own file
    # still succeeds and the failure surfaces at the post-save hash step the review targets.
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    state, g = _state(), {}
    before_state = dict(state)
    before_tables = dict(state["tables"])

    real_read_bytes = Path.read_bytes
    saved_dir = tmp_path / "out" / "state" / "r1"

    def _refuse_saved(self, *args, **kwargs):
        if saved_dir in self.parents and str(self).endswith(".tsv"):
            raise OSError("simulated read failure")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", _refuse_saved)
    with pytest.raises(steps._StepSkipped, match="hash"):
        _run(tmp_path, None, state, g)
    assert state.keys() == before_state.keys()
    assert state["tables"] == before_tables
    assert not any(key.startswith("detab.de.") for key in g)


@pytest.mark.parametrize(
    "table_name, scalar, missing_context",
    [("emb", None, False), ("detab.graph.edges", None, False)]
    + [("de", f"de.{suffix}", False) for suffix in ("rows", "columns", "sha256")]
    + [("de", None, True)],
)
def test_table_conflicts_are_refused_before_saving_or_merging(
    tmp_path, table_name, scalar, missing_context
):
    path = tmp_path / "de.tsv"
    path.write_text("value\n1\n2\n")
    embedding = tmp_path / "embedding.npy"
    np.save(embedding, np.ones((4, 2)))
    doc = {
        "schema_version": 1,
        "ok": True,
        "n_cells": 4,
        "facts": {
            "embedding": {
                "path": "embedding.npy",
                "sha256": hashlib.sha256(embedding.read_bytes()).hexdigest(),
            }
        },
        "tables": {
            table_name: {"path": "de.tsv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        },
        "scalars": {scalar: 999} if scalar else {},
    }
    state, g = _state(), {}
    state["extras"] = {"detab.graph.edges": np.ones((2, 2))}
    before = dict(state)
    ctx = {"out_dir": tmp_path / "out", "run_id": "r1", "step_index": 0}
    if missing_context:
        del ctx["step_index"]
    with pytest.raises(steps._StepSkipped, match="step_index" if missing_context else "conflict"):
        external._merge_v1(
            doc,
            tmp_path,
            state,
            g,
            "detab",
            n_cells=4,
            cell_list=None,
            input_dim=3,
            produces=["embedding"],
            tables={table_name: ["value"]},
            ctx=ctx,
        )
    assert state.keys() == before.keys()
    assert all(state[key] is value for key, value in before.items())
    assert state["tables"] == {}
    assert g == {}
    assert not (tmp_path / "out").exists()


def test_a_table_path_in_a_symlink_loop_is_refused_not_errored(tmp_path):
    (tmp_path / "loop.tsv").symlink_to(tmp_path / "loop.tsv")
    doc = {
        "schema_version": 1,
        "ok": True,
        "n_cells": 4,
        "tables": {"de": {"path": "loop.tsv", "sha256": "0" * 64}},
    }
    problems = external.check_result(doc, base=tmp_path, produces=[], tables={"de": ["gene"]})
    assert any(problem.startswith("tables.de:") for problem in problems)
