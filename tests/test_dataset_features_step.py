"""Publication and provenance of dataset descriptions, independent of compute."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import features_stub
from manyruns import artifacts
from manyruns.pipeline import features, runner, steps
from manyruns.session import Session

STEP = {"name": "dataset_features", "group": "analysis"}


@pytest.fixture(autouse=True)
def helper(monkeypatch):
    features_stub.CALLS.clear()
    monkeypatch.setattr(features, "_helper", lambda: features_stub.stub_rows)


def _run(tmp_path, params=None, **kwargs):
    return runner.run_inproc(np.ones((30, 4)),
                            {"steps": [dict(STEP, params=params or {})]}, tmp_path, **kwargs)


def _table(out, index=0):
    rec = out["steps"][index]
    assert rec["outcome"] == "ok", rec.get("detail")
    path = Path(rec["artifacts"]["dataset_features"])
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_filter=False)


def test_roles_default_to_the_runs_choice_keys_independently(tmp_path):
    roles = dict(sample="donor", batch="batch", time="day", condition="cond", labels="ignored")
    out = _run(tmp_path, original_input={"object": np.ones((30, 4)), "role_keys": roles})
    expected = dict(roles, labels=None)
    assert features_stub.CALLS[0][1] == expected
    assert all(json.loads(value) == expected for value in _table(out)["roles"])


def test_role_params_override_choice_keys(tmp_path):
    _run(tmp_path, {"roles": {"batch": "absent", "time": None, "labels": "cell_type"}},
         original_input={"object": np.ones((30, 4)),
                         "role_keys": dict(batch="batch", time="day", condition="cond")})
    assert features_stub.CALLS[0][1] == dict(sample=None, batch="absent", time=None,
                                           condition="cond", labels="cell_type")


@pytest.mark.parametrize("params, words", [
    ({"unknown": 1}, ("roles", "layer", "overrides")),
    ({"roles": {"unknown": "x"}}, ("sample", "condition", "batch", "time", "labels")),
    ({"roles": []}, ("roles",)),
    ({"roles": {"batch": 3}}, ("batch",)),
    ({"layer": 3}, ("layer",)),
    ({"overrides": []}, ("overrides",)),
])
def test_unknown_role_or_param_is_refused(tmp_path, params, words):
    out = _run(tmp_path, params)
    rec = out["steps"][0]
    assert rec["outcome"] == "skipped"
    assert all(word in rec["detail"] for word in words)
    assert not features_stub.CALLS
    assert "dataset_features" not in rec.get("artifacts", {})


def test_layer_and_overrides_reach_the_helper(tmp_path):
    out = _run(tmp_path, {"layer": "counts", "overrides": {"species": "human"}})
    assert features_stub.CALLS[0][2:] == ("counts", {"species": "human"})
    assert set(_table(out)["tested_matrix"]) == {"layers:counts"}


def test_the_table_has_the_documented_columns(tmp_path):
    out = _run(tmp_path)
    assert list(_table(out).columns) == list(features.COLUMNS)
    g = out["g_vector"]
    assert g["dataset_features.n_ok"] == 3
    assert g["dataset_features.dataset_features.rows"] == 4
    assert g["dataset_features.dataset_features.columns"] == sorted(features.COLUMNS)
    path = Path(out["steps"][0]["artifacts"]["dataset_features"])
    assert path.name == "00-dataset_features_dataset_features.tsv"
    assert g["dataset_features.dataset_features.sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "dataset_features.definition_version" not in g


def test_tsv_round_trip_preserves_types(tmp_path):
    frame = _table(_run(tmp_path))
    values = [json.loads(v) for v in frame["value"]]
    assert values == [True, 3, "X", None]
    assert [type(v) for v in values] == [bool, int, str, type(None)]
    assert frame["reason"].tolist() == ["", "", "", "Role not supplied: sample"]


def _assert_manifest(tmp_path, out):
    path = out["steps"][0]["artifacts"]["dataset_features"]
    manifest = artifacts.read_manifest(tmp_path, out["run_id"])
    assert manifest[path]["slot"] == "table:dataset_features"
    assert Path(path).is_file()


def test_the_table_is_in_the_step_artifacts_and_the_manifest(tmp_path):
    _assert_manifest(tmp_path, _run(tmp_path))


def test_the_table_is_published_when_a_later_step_fails(tmp_path, monkeypatch):
    def boom(*args):
        raise RuntimeError("intentional failure")

    monkeypatch.setitem(steps._ANALYSIS_STEPS, "boom", boom)
    out = runner.run_inproc(np.ones((30, 4)),
                            {"steps": [STEP, {"name": "boom", "group": "analysis"}]}, tmp_path)
    assert out["steps"][1]["outcome"] == "error"
    _assert_manifest(tmp_path, out)


def test_the_step_reads_the_original_input_after_a_filter(tmp_path, monkeypatch):
    obj = pytest.importorskip("anndata").AnnData(np.ones((30, 4)))
    monkeypatch.setitem(runner._prep._PREP_STEPS, "cut",
                        lambda *args: {"mask": np.arange(30) % 2 == 0, "axis": "rows"})
    out = runner.run_inproc(obj, {"steps": [STEP, {"name": "cut", "group": "prep"}, STEP]}, tmp_path)
    assert out["steps"][1]["dropped"]["n"] == 15
    assert len(features_stub.CALLS) == 2
    assert all(call[0] is obj and call[0].shape == (30, 4) for call in features_stub.CALLS)
    pd.testing.assert_frame_equal(_table(out, 0), _table(out, 2))
    manifest = artifacts.read_manifest(tmp_path, out["run_id"])
    assert sum(e["slot"] == "table:dataset_features" for e in manifest.values()) == 2


def test_input_sha256_of_a_file_source(tmp_path):
    path = tmp_path / "input.csv"
    path.write_bytes(b"a,b\n1,2\n")
    frame = _table(_run(tmp_path, source_path=path))
    assert set(frame["input_sha256"]) == {hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("source", [None, "directory", "missing"])
def test_input_sha256_uses_matrix_without_a_file(tmp_path, source):
    path = None if source is None else tmp_path if source == "directory" else tmp_path / "missing"
    assert set(_table(_run(tmp_path, source_path=path))["input_sha256"]) == {features.input_sha256(np.ones((30, 4)))}


def test_input_sha256_streams_the_file(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    data = b"x" * (2 * 1024 * 1024 + 17)
    path.write_bytes(data)
    sizes, real_open = [], Path.open

    class Reader:
        def __enter__(self):
            self.file = real_open(path, "rb")
            return self

        def __exit__(self, *args):
            self.file.close()

        def read(self, size):
            sizes.append(size)
            assert size == 1024 * 1024
            return self.file.read(size)

    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("whole-file read"))
    monkeypatch.setattr(Path, "open", lambda *a, **kw: Reader())
    assert features._file_sha256(path) == hashlib.sha256(data).hexdigest()
    assert len(sizes) == 4


def test_a_point_cloud_run_publishes_a_table(tmp_path):
    obj = np.random.default_rng(0).normal(size=(200, 5))
    obj -= obj.mean(axis=0)
    out = runner.run_inproc(obj, {"steps": [STEP]}, tmp_path)
    assert len(_table(out)) == 4
    assert features_stub.CALLS[0][0] is obj


def test_no_original_input_is_skipped_with_a_reason(tmp_path):
    session = Session("features", engine="_inproc", out_dir=tmp_path)
    rec = session.step(STEP)
    assert rec["outcome"] == "skipped"
    assert "this run has none" in rec["detail"]
    assert "unsupported" not in rec
    assert not features_stub.CALLS


@pytest.mark.parametrize("change", [
    lambda r: r.pop("status"),
    lambda r: r.update(status="maybe"),
    lambda r: r.update(value=np.float64("nan")),
    lambda r: r.update(value=np.int64(3)),
    lambda r: r.update(value=np.float64(3)),
    lambda r: r.update(value={"nested": 3}),
    lambda r: r.update(definition_version="other/1"),
    lambda r: r.update(extra=True),
])
def test_a_bad_helper_row_is_refused(tmp_path, monkeypatch, change):
    rows = features_stub.stub_rows(None, {})
    change(rows[0])
    monkeypatch.setattr(features, "_helper", lambda: lambda *a, **kw: rows)
    session = Session("features", engine="_inproc", out_dir=tmp_path,
                      original_input={"object": np.ones((30, 4)), "role_keys": {}})
    before = session.state["tables"]
    rec = session.step(STEP)
    assert rec["outcome"] == "skipped"
    assert session.state["tables"] is before
    assert not session.state["tables"]
    assert "dataset_features" not in rec.get("artifacts", {})
    assert not any(k.startswith("dataset_features.") for k in session.g)
    assert tmp_path.is_dir()
    assert not list(tmp_path.rglob("*.tsv"))


@pytest.mark.parametrize("error", [ValueError, TypeError])
def test_helper_errors_are_refusals(tmp_path, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error("invalid helper argument")

    monkeypatch.setattr(features, "_helper", lambda: fail)
    rec = _run(tmp_path)["steps"][0]
    assert rec["outcome"] == "skipped"
    assert "invalid helper argument" in rec["detail"]
