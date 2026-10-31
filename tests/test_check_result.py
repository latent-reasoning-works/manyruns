"""`check_result` is the open contract of the version 1 result file.

No JSON Schema file describes the result. `external.check_result` states every key, type and
rule, and the golden adapter fixtures show a valid result. Each rule below has a document or
an output that breaks it.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

import manyruns
from manyruns.pipeline import external
from model_seam_fixtures import IDS12, SHA12, X12, golden_adapter, golden_result


@pytest.fixture()
def golden(tmp_path):
    base = tmp_path / "out"
    return base, golden_result(base)


def test_check_result_accepts_golden_result(golden):
    _, doc = golden
    assert external.check_result(doc) == []
    assert external.RESULT_SCHEMA_VERSION == 1


def test_package_holds_no_result_schema_file():
    root = Path(manyruns.__file__).parent
    assert root.is_dir()
    assert sorted(root.rglob("result-v1*.json")) == []


def _set(path, value):
    def edit(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return doc
    return edit


def _drop(path):
    def edit(doc):
        node = doc
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
        return doc
    return edit


def _only(*facts):
    def edit(doc):
        doc["facts"] = {k: v for k, v in doc["facts"].items() if k in facts}
        return doc
    return edit


def _replace(new):
    return lambda doc: copy.deepcopy(new)


_DECLINED = {"schema_version": 1, "ok": False, "reason": "no time axis"}

#: (row, edit of the golden result, produces, first problem contains; None means valid)
DOCUMENTS = [
    (1, lambda d: d, None, None),
    (2, _replace(_DECLINED), None, None),
    (3, _only("clusters", "graph"), None, None),
    (4, _set(["schema_version"], 2), None, "schema_version"),
    (5, _drop(["ok"]), None, "ok"),
    (6, _replace({"schema_version": 1, "ok": True, "n_cells": 12}), None, "facts"),
    (7, _set(["facts", "velocity"], {"path": "v.npy", "sha256": SHA12}), None, "velocity"),
    (8, _set(["facts", "embedding", "path"], "/x.npy"), None, "embedding"),
    (9, _set(["facts", "embedding", "path"], "../x.npy"), None, "embedding"),
    (10, _set(["facts", "embedding", "sha256"], SHA12[:63]), None, "sha256"),
    (11, _drop(["facts", "class_probabilities", "value", "surface"]), None, "surface"),
    (12, _set(["facts", "class_probabilities", "value", "clusters"], 1.2), None,
     "class_probabilities"),
    (13, _set(["scalars", "n_cells"], [12]), None, "scalars"),
    (14, _replace({"schema_version": 1, "ok": False}), None, "reason"),
    (15, _set(["notes"], "x"), None, "notes"),
    (16, _set(["schema_version"], True), None, "schema_version"),
    (17, _set(["ok"], "true"), None, "ok"),
    (18, _replace({"schema_version": 1, "ok": False, "reason": ""}), None, "reason"),
    (19, _replace({"schema_version": 1, "ok": False, "reason": 3}), None, "reason"),
    (20, _drop(["n_cells"]), None, "n_cells"),
    (21, _set(["n_cells"], 0), None, "n_cells"),
    (22, _set(["n_cells"], True), None, "n_cells"),
    (23, _drop(["cells_sha256"]), None, None),
    (24, _set(["cells_sha256"], SHA12.upper()), None, "cells_sha256"),
    (25, _set(["facts"], {}), None, "facts"),
    (26, _set(["facts", "embedding", "path"], "embedding.csv"), None, "embedding"),
    (27, _set(["facts", "embedding", "path"], "out/embedding.npy"), None, None),
    (28, _set(["facts", "graph", "path"], "graph.npy"), None, "graph"),
    (29, _drop(["facts", "embedding", "sha256"]), None, "sha256"),
    (30, _set(["facts", "embedding", "dtype"], "float32"), None, "dtype"),
    (31, _set(["facts", "class_probabilities"], {"path": "cp.npy", "sha256": SHA12}), None,
     "class_probabilities"),
    (32, _set(["facts", "class_probabilities", "value", "tree"], 0.1), None, "tree"),
    (33, _set(["scalars"], {"n_cells": 12, "converged": True, "method": "svd", "loss": 0.25}),
     None, None),
    (34, _drop(["schema_version"]), None, "schema_version"),
    (35, _set(["facts"], []), None, "facts"),
    (36, _set(["scalars"], 3), None, "scalars"),
    (37, _set(["facts", "embedding", "path"], 3), None, "embedding"),
    (38, _set(["facts", "class_probabilities", "value", "cycle"], -0.1), None,
     "class_probabilities"),
    (39, _set(["facts", "class_probabilities", "value", "cycle"], True), None,
     "class_probabilities"),
    (40, _set(["facts", "class_probabilities", "value", "cycle"], float("nan")), None,
     "class_probabilities"),
    (41, _set(["scalars", "loss"], float("nan")), None, "scalars"),
    (42, _set(["scalars", "loss"], float("inf")), None, "scalars"),
    (43, _set(["reason"], "x"), None, "reason"),
    (44, _replace({"schema_version": 1, "ok": False, "reason": "x", "n_cells": 12}), None,
     "n_cells"),
    (45, _set(["facts", "embedding", "path"], "out\\embedding.npy"), None, "embedding"),
    (46, _set(["facts", "embedding", "path"], "./embedding.npy"), None, "embedding"),
    (47, _set(["scalars"], {"tool_origin": "x"}), None, "tool_"),
    (48, lambda d: d, ["clusters"], "produces"),
    (49, _only("clusters", "graph"), ["clusters", "graph"], None),
]


def test_the_table_has_49_documents():
    assert [row for row, *_ in DOCUMENTS] == list(range(1, 50))


@pytest.mark.parametrize("row,edit,produces,expected", DOCUMENTS,
                         ids=[f"row{r[0]}" for r in DOCUMENTS])
def test_each_rule_has_a_breaking_document(golden, row, edit, produces, expected):
    _, doc = golden
    problems = external.check_result(edit(copy.deepcopy(doc)), produces=produces)
    if expected is None:
        assert problems == [], problems
    else:
        assert problems, f"row {row} was accepted"
        assert expected in problems[0], problems


def _resave(base, doc, fact, array=None, **arrays):
    path = base / doc["facts"][fact]["path"]
    if arrays:
        np.savez(path, **arrays)
    else:
        np.save(path, array)
    doc["facts"][fact]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _graph(base):
    with np.load(base / "graph.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _nan_embedding(base, doc, tmp):
    emb = np.load(base / "embedding.npy")
    emb[0, 0] = np.nan
    _resave(base, doc, "embedding", emb)


def _inf_pseudotime(base, doc, tmp):
    pt = np.load(base / "pseudotime.npy")
    pt[0] = np.inf
    _resave(base, doc, "pseudotime", pt)


def _float_clusters(base, doc, tmp):
    _resave(base, doc, "clusters", (np.arange(12) % 2).astype(float))


def _one_bin(base, doc, tmp):
    _resave(base, doc, "trajectories", np.zeros((1, 2, 3)))


def _dup_nodes(base, doc, tmp):
    g = _graph(base)
    g["node_ids"] = np.array([0, 1, 1, 3], dtype="int64")
    _resave(base, doc, "graph", **g)


def _bad_cell_node(base, doc, tmp):
    g = _graph(base)
    g["cell_node"][0] = 7
    _resave(base, doc, "graph", **g)


def _extra_array(base, doc, tmp):
    g = _graph(base)
    g["extra"] = np.zeros(2)
    _resave(base, doc, "graph", **g)


def _symlink_out(base, doc, tmp):
    outside = tmp / "elsewhere"
    outside.mkdir()
    target = outside / "embedding.npy"
    target.write_bytes((base / "embedding.npy").read_bytes())
    (base / "embedding.npy").unlink()
    (base / "embedding.npy").symlink_to(target)


FILE_EDITS = [
    ("nan_embedding", _nan_embedding, 12, "embedding"),
    ("inf_pseudotime", _inf_pseudotime, 12, "pseudotime"),
    ("float_clusters", _float_clusters, 12, "clusters"),
    ("one_time_bin", _one_bin, 12, "trajectories"),
    ("duplicate_node_ids", _dup_nodes, 12, "node_ids"),
    ("unknown_cell_node", _bad_cell_node, 12, "cell_node"),
    ("extra_graph_array", _extra_array, 12, "extra"),
    ("symlink_outside_base", _symlink_out, 12, "embedding"),
    ("other_cell_count", lambda base, doc, tmp: None, 11, "n_cells"),
]


@pytest.mark.parametrize("edit,n_cells,expected", [e[1:] for e in FILE_EDITS],
                         ids=[e[0] for e in FILE_EDITS])
def test_each_file_rule_has_an_output_that_breaks_it(golden, tmp_path, edit, n_cells, expected):
    base, doc = golden
    edit(base, doc, tmp_path)
    problems = external.check_result(doc, base=base, n_cells=n_cells, cell_list=IDS12,
                                     input_dim=3)
    assert problems, "the broken output was accepted"
    assert expected in problems[0], problems


def test_golden_output_passes_every_file_rule(golden):
    base, doc = golden
    assert external.check_result(doc, base=base, n_cells=12, cell_list=IDS12, input_dim=3,
                                 produces=list(doc["facts"])) == []


def test_truncated_array_is_refused(tmp_path):
    doc = golden_result(tmp_path, {"truncate": 5})
    problems = external.check_result(doc, base=tmp_path, n_cells=12)
    assert len(problems) == 1, problems
    assert all(s in problems[0] for s in ("clusters", "5", "12")), problems


def test_changed_payload_is_refused(golden):
    base, doc = golden
    path = base / "pseudotime.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    problems = external.check_result(doc, base=base, n_cells=12)
    assert len(problems) == 1, problems
    assert "pseudotime" in problems[0] and "sha256" in problems[0]


def test_unknown_edge_node_is_refused(tmp_path):
    doc = golden_result(tmp_path, {"bad_edge": True})
    problems = external.check_result(doc, base=tmp_path, n_cells=12)
    assert len(problems) == 1, problems
    assert all(s in problems[0] for s in ("graph", "edges", "9")), problems


def test_pickled_array_is_refused(golden):
    base, doc = golden
    path = base / "clusters.npy"
    np.save(path, np.array([{"a": 1}] * 12, dtype=object), allow_pickle=True)
    doc["facts"]["clusters"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    problems = external.check_result(doc, base=base, n_cells=12)
    assert len(problems) == 1, problems
    assert "clusters" in problems[0]


def test_other_cell_list_is_refused(golden):
    base, doc = golden
    problems = external.check_result(doc, base=base, n_cells=12, cell_list=IDS12[::-1])
    assert len(problems) == 1, problems
    assert "cells_sha256" in problems[0]


def test_trajectories_in_other_frame_are_refused(golden):
    base, doc = golden
    problems = external.check_result(doc, base=base, n_cells=12, input_dim=5)
    assert len(problems) == 1, problems
    assert all(s in problems[0] for s in ("trajectories", "3", "5")), problems


def test_flat_trajectories_are_refused(tmp_path):
    doc = golden_result(tmp_path, {"flat_trajectories": True})
    problems = external.check_result(doc, base=tmp_path, n_cells=12)
    assert len(problems) == 1, problems
    assert "trajectories" in problems[0] and "3-D" in problems[0]


def test_golden_adapter_matches_the_spec_values(golden):
    """The golden output is the table of the spec, value for value."""
    base, doc = golden
    assert doc["cells_sha256"] == SHA12 and doc["n_cells"] == 12
    np.testing.assert_array_equal(np.load(base / "embedding.npy"), X12[:, :2].astype("float32"))
    np.testing.assert_array_equal(np.load(base / "trajectories.npy"),
                                  np.arange(18, dtype=float).reshape(3, 2, 3))
    assert doc["facts"]["class_probabilities"]["value"] == golden_adapter().CLASS_PROBABILITIES
    assert doc["scalars"] == {"n_cells": 12}
