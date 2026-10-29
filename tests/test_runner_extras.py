"""A dict-of-scalars engine byproduct is flattened one level into the g-vector.

`_collect_extras` routes a byproduct by what it is: an array goes to the artifact channel, an
`int` or `float` goes to the g-vector as `<step>.<key>`. A dict fell through both tests and was
dropped. The Reeb graph step returns its graph counts as such a dict (`structural_summary`:
`n_nodes, n_edges, n_components, degree_sequence, n_branch_points, n_endpoints, n_loops`), so
none of the counts reached the record.

The rule here: open a dict once, and route each sub-value by the SAME scalar test as a
top-level value. The key is the flattened extra key `<step>.<key>.<subkey>`. A list such as
`degree_sequence` has a variable length, so it is not comparable across runs by a fixed key,
and it stays out. A dict inside the dict is not opened, and an array inside the dict is not
sent to the artifact channel.
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")

from manyruns import pipeline, store  # noqa: E402
from manyruns.pipeline import runner  # noqa: E402

SUMMARY = {"n_nodes": 5, "n_edges": 4, "n_components": 1,
           "degree_sequence": [3, 2, 1, 1, 1],
           "n_branch_points": 1, "n_endpoints": 3, "n_loops": 0}

COUNTS = {"n_nodes": 5, "n_edges": 4, "n_components": 1,
          "n_branch_points": 1, "n_endpoints": 3, "n_loops": 0}


def _state():
    return {"X": None, "emb": None, "pseudotime": None, "labels": None, "seed": 0}


def _reeb_out():
    return {"embeddings": np.zeros((4, 2)), "node_coordinates": np.zeros((3, 2)),
            "structural_summary": dict(SUMMARY)}


# ── a dict of scalars is flattened one level ────────────────────────────────
def test_the_reeb_graph_summary_reaches_the_g_vector():
    state, g = _state(), {}
    runner._collect_extras("reeb_graph", _reeb_out(), state, g)

    assert g == {f"reeb_graph.structural_summary.{k}": v for k, v in COUNTS.items()}
    assert "reeb_graph.node_coordinates" in state["extras"]
    assert not [k for k in state["extras"] if k.startswith("reeb_graph.structural_summary")]


def test_float_sub_values_are_kept():
    state, g = _state(), {}
    runner._collect_extras("x", {"summary": {"mean_degree": 2.5}}, state, g)
    assert g == {"x.summary.mean_degree": 2.5}


def test_a_non_string_sub_key_is_converted_with_str():
    state, g = _state(), {}
    runner._collect_extras("t", {"s": {1: 2}}, state, g)
    assert g == {"t.s.1": 2}


# ── non-scalar sub-values are not recorded ──────────────────────────────────
def test_degree_sequence_stays_out():
    state, g = _state(), {}
    runner._collect_extras("reeb_graph", _reeb_out(), state, g)
    assert not [k for k in g if k.endswith("degree_sequence")]


def test_mixed_sub_values_keep_only_the_scalar():
    state, g = _state(), {}
    runner._collect_extras("t", {"s": {"a": 1, "flag": True, "name": "x", "none": None,
                                       "inner": {"b": 2}, "seq": (1, 2),
                                       "arr": np.zeros(2)}}, state, g)
    assert g == {"t.s.a": 1}
    assert not state.get("extras")


def test_a_dict_of_arrays_is_neither_flattened_nor_sent_to_the_artifact_channel():
    """The shape of the planned engine `graph` output: four arrays in one dict."""
    state, g = _state(), {}
    runner._collect_extras("reeb_graph", {"graph": {"edges": np.zeros((4, 2), np.int64),
                                                    "edge_weights": np.ones(4),
                                                    "node_ids": np.arange(5),
                                                    "node_coordinates": np.zeros((5, 2))}},
                           state, g)
    assert g == {}
    assert not state.get("extras")


def test_an_empty_dict_records_nothing():
    state, g = _state(), {}
    runner._collect_extras("t", {"s": {}}, state, g)
    assert g == {}


# ── the existing routes are unchanged ───────────────────────────────────────
def test_a_claimed_key_that_holds_a_dict_is_not_flattened():
    state, g = _state(), {}
    runner._collect_extras("p", {"metadata": {"n": 3}, "scores": {"lid": 0.5}}, state, g)
    assert g == {}
    assert not state.get("extras")


def test_a_top_level_scalar_keeps_its_two_segment_key():
    state, g = _state(), {}
    runner._collect_extras("mioflow", {"n_branches": 3, "converged": True}, state, g)
    assert g == {"mioflow.n_branches": 3}


# ── end to end: the flattened keys reach the run row ────────────────────────
def _engine(monkeypatch) -> None:
    api = types.ModuleType("manylatents.api")

    def run(**kw):
        n = kw["input_data"].shape[0]
        return {"embeddings": np.zeros((n, 2), np.float32), "scores": {},
                "structural_summary": dict(SUMMARY)}

    api.run = run
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)


def test_flattened_extra_keys_reach_the_run_row(tmp_path, monkeypatch):
    _engine(monkeypatch)

    result = pipeline.run_manylatents(
        {"name": "r", "steps": [{"name": "reeb_graph", "group": "latent", "params": {}}]},
        array=np.arange(36, dtype=np.float32).reshape(9, 4), out_dir=tmp_path)
    assert result["ok"] is True

    store.append(result, out_dir=tmp_path)
    row = list(store.read(tmp_path))[-1]
    for k, v in COUNTS.items():
        assert row["g_vector"][f"reeb_graph.structural_summary.{k}"] == v
    assert not [k for k in row["g_vector"] if k.endswith("degree_sequence")]
