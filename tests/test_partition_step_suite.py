"""A latent step that produces a partition is not scored by the geometry suite.

THE FAILURE, measured on the installed stack (manyruns 0.1.0, manylatents 0.1.7), bundled
`cluster` recipe over swissroll 800x10, seed 42, `--engine manylatents --device cpu`: step 4
`leiden` recorded `outcome='error'`, `detail='MeasurementUnavailable: only 17 distinct points
for k=20: need at least k+1'`, the run `ok=False`, exit code 1. The partition had been computed
and was lost with the exception.

THE CHAIN. `_ml_latent` forwards the declared suite to the engine on EVERY latent step
(`runner.py`, `kwargs["metrics"]`). The engine evaluates it inside `run()` on the step's own
output, with no per-metric catch on the list-of-names path (manylatents `evaluate.py`,
`_evaluate_registry`). `leiden` leaves an (N, 1) label column in `state["emb"]` — the shape
`test_analysis_readouts` measured and `vocab.STEP_PRODUCES["leiden"] == ("clusters",)` declares.
`lid` refuses fewer than k+1 distinct points (manylatents `lid.py`, the Levina-Bickel contract
since 0.1.7). One metric declining became the STEP's error — the collapse `suite.measure`
already refuses for the final suite ("one bad metric must not lose the other twelve").

The suite is geometry on coordinates. A partition has none to measure, so it is not sent; the
absence is SAID in the g-vector rather than implied by a missing key (`suite.py`, "declared,
not conditional").

The engine here is a stand-in that does what the real one does on this input: it refuses when
the suite is forwarded to a partition step, and returns the partition when it is not.
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")

from manyruns import pipeline  # noqa: E402
from manyruns.pipeline import runner  # noqa: E402

SUITE = ["lid", "anisotropy"]


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
        return {"embeddings": np.zeros((n, 2), dtype=np.float32),
                "scores": {m: 0.5 for m in kw.get("metrics", [])}}

    api.run = run
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)


def test_a_partition_step_is_not_scored_by_the_geometry_suite(tmp_path, monkeypatch):
    calls: list = []
    _engine(monkeypatch, calls)

    out = pipeline.run_manylatents(
        {"name": "r", "steps": [{"name": "pca", "group": "latent", "params": {}},
                                {"name": "leiden", "group": "latent", "params": {}}]},
        array=np.arange(36, dtype=np.float32).reshape(9, 4), out_dir=tmp_path, metrics=SUITE)

    by_name = {s["name"]: s for s in out["steps"]}
    assert by_name["leiden"]["outcome"] == "ok", by_name["leiden"]["detail"]
    assert by_name["leiden"]["produced"].startswith("ndarray(9, 1)")
    assert out["ok"] is True
    # The suite still travels with every COORDINATE step, and lands in the g-vector.
    assert [c["algorithms"]["latent"] for c in calls] == ["pca", "leiden"]
    assert calls[0]["metrics"] == SUITE
    assert "metrics" not in calls[1]
    assert out["g_vector"]["pca.lid"] == 0.5
    # The absence is stated, not implied by a missing `leiden.lid` key.
    assert "partition" in out["g_vector"]["leiden.suite_note"]
    assert "leiden.lid" not in out["g_vector"]


def test_the_exemption_is_read_off_the_calculus_not_a_name_list():
    """`STEP_PRODUCES` is the one table that already knows which step yields `clusters`; a
    second list of names here would drift from it the first time a clustering step is added."""
    from manyruns.vocab import STEP_PRODUCES

    assert [n for n in ("pca", "umap", "phate", "leiden") if runner._label_valued(n)] == ["leiden"]
    assert all(runner._label_valued(n) == ("clusters" in produces)
               for n, produces in STEP_PRODUCES.items())
