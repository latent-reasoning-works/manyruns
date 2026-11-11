"""Real-helper contract and timing; gated until the helper is released."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import features_stub
from manyruns.pipeline import features

helper_module = pytest.importorskip("manylatents.singlecell.dataset_features")


def _step(data, tmp_path, role_keys=None):
    from manyruns.pipeline import runner

    return runner.run_inproc(
        data, {"steps": [{"name": "dataset_features", "group": "analysis"}]}, tmp_path,
        original_input={"object": data, "role_keys": role_keys or {}},
    )


def test_the_real_helper_gives_27_rows_through_the_step(tmp_path):
    from anndata import AnnData

    data = AnnData(sparse.csr_matrix(np.ones((3, 4))))
    data.var_names = ["MT-A", "GENE1", "GENE2", "GENE3"]
    data.uns.update(modality="snRNA-seq", species="human", technology="droplet")
    data.obs["batch"] = ["a", "a", "b"]
    data.obs["condition"] = ["control", "treated", "treated"]
    data.obs["day"] = [0, 1, 2]
    for layer in ("counts", "spliced", "unspliced"):
        data.layers[layer] = data.X.copy()
    roles = dict(batch="batch", condition="condition", time="day")
    out = _step(data, tmp_path, roles)
    rec = out["steps"][0]
    assert rec["outcome"] == "ok", rec.get("detail")
    frame = pd.read_csv(rec["artifacts"]["dataset_features"], sep="\t",
                        keep_default_na=False, na_filter=False)
    assert len(frame) == 27
    assert list(frame["key"]) == [key for key, _ in helper_module.KEYS]
    assert frame["type"].tolist() == [kind for _, kind in helper_module.KEYS]
    assert set(frame["status"]) <= set(features.STATUSES)
    assert set(frame["definition_version"]) == {"dataset_features/1"}
    assert features.ROW_FIELDS == features_stub.FIELDS == helper_module.FIELDS
    assert features.STATUSES == helper_module.STATUSES
    assert features_stub.DEFINITION_VERSION == helper_module.DEFINITION_VERSION
    decoded = {row["key"]: {**row, "value": json.loads(row["value"]),
                            "reason": row["reason"] or None}
               for row in frame.to_dict("records")}
    # Pin the stub against the real values of these same four representative rows.
    for row in features_stub.stub_rows(data, roles):
        assert row == {field: decoded[row["key"]][field] for field in helper_module.FIELDS}
    assert decoded["modality"]["value"] == "snRNA-seq"
    assert decoded["has_condition_column"]["value"] is True
    assert decoded["has_time_column"]["value"] is True
    assert decoded["has_spliced_unspliced_layers"]["value"] is True


def test_timing_on_5000_by_20000_sparse_is_reported(tmp_path, record_property):
    from anndata import AnnData

    rng = np.random.default_rng(0)
    data = AnnData(sparse.random(5000, 20000, density=0.05, format="csr", random_state=0,
                                data_rvs=lambda n: rng.integers(1, 10, size=n)))
    state, g = {"tables": {}}, {}
    ctx = {"original_input": {"object": data, "role_keys": {}},
           "out_dir": tmp_path, "step_index": 0, "run_id": "timing"}
    started = time.perf_counter()
    features.step(state, g, {}, ctx)
    elapsed = time.perf_counter() - started
    record_property("dataset_features_seconds", elapsed)
    print(f"dataset_features_seconds={elapsed:.6f}")
    assert g["dataset_features.dataset_features.rows"] == 27
    assert Path(state["tables"]["dataset_features"]).is_file()
