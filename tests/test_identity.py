"""Run identity — the precondition the stage derivation assumed and the record lacked.

Three capabilities (validate, contrast, apply) each reduce to reading the run store. An audit showed
the
reduction was unavailable: `store.append` had five call sites, all in tests, and the row it
wrote carried no run id, no dataset and no timestamp. Every one of those capabilities is an
EDGE between records — "B is A's null twin", "every run on this cohort", "has this spec run
before" — and an unaddressed record has no edges.

Two ids, because both semantics are wanted: dedupe and stopping rules need "same spec, same key";
seed stability and null
margins need "each execution is its own".
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

RECIPE = {"name": "r", "steps": [{"name": "phate", "group": "latent",
                                  "params": {"n_components": 2}}]}


def _run(tmp_path, **kw):
    from manyruns import pipeline

    return pipeline.run_inproc(np.random.default_rng(0).normal(size=(40, 5)),
                                 RECIPE, tmp_path, seed=1, **kw)


def test_every_run_is_addressable(tmp_path):
    out = _run(tmp_path, dataset="cohort_a")
    assert out["run_id"] and out["spec_id"]
    assert out["dataset"] == "cohort_a"
    assert out["at"].endswith("+00:00")          # UTC, so runs from two machines order


def test_the_same_spec_keeps_its_key_while_each_execution_gets_its_own(tmp_path):
    """The distinction that lets one record answer two different questions."""
    a, b = _run(tmp_path, dataset="d"), _run(tmp_path, dataset="d")

    assert a["spec_id"] == b["spec_id"]          # same question asked twice
    assert a["run_id"] != b["run_id"]            # ...and two distinct askings


def test_a_different_dataset_or_seed_is_a_different_spec(tmp_path):
    """`spec_id` must cover everything that changes what was asked — otherwise a re-run on
    another cohort would dedupe against the first and silently vanish."""
    from manyruns import pipeline

    base = _run(tmp_path, dataset="d")
    assert _run(tmp_path, dataset="other")["spec_id"] != base["spec_id"]

    seeded = pipeline.run_inproc(np.random.default_rng(0).normal(size=(40, 5)),
                                   RECIPE, tmp_path, seed=99, dataset="d")
    assert seeded["spec_id"] != base["spec_id"]

    other_params = {"name": "r", "steps": [{"name": "phate", "group": "latent",
                                            "params": {"n_components": 7}}]}
    changed = pipeline.run_inproc(np.random.default_rng(0).normal(size=(40, 5)),
                                    other_params, tmp_path, seed=1, dataset="d")
    assert changed["spec_id"] != base["spec_id"]


def test_the_store_now_carries_the_address(tmp_path):
    """Without these columns you cannot group by cohort or join a run to its control, which
    is what made `contrast` and `validate` unreachable rather than merely unbuilt."""
    from manyruns import store

    store.append(_run(tmp_path, dataset="cohort_a"), out_dir=tmp_path)
    row = next(iter(store.read(tmp_path)))

    assert row["run_id"] and row["spec_id"]
    assert row["dataset"] == "cohort_a"
    assert row["at"]


def test_the_product_actually_writes_the_store(tmp_path, monkeypatch):
    """THE gap. `store.append` shipped with five call sites, every one of them a test, so no
    run was ever recorded. A store nothing writes is not a store."""
    import inspect

    from manyruns import app

    src = inspect.getsource(app._explore_project)
    assert "_store.append" in src, "the product still does not record its runs"
