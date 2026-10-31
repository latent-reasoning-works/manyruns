"""C1 regression: a CONDITION axis must never be fed to MIOFlow as a TIME axis.

The trace of a real cflows run on case/control data showed `mioflow.n_timepoints: 2` — the
two disease groups (treated/healthy) mapped to two timepoints, training a fabricated
trajectory "from disease to health over time". `load_labeled` now returns the label KIND,
and the trajectory step ignores a condition axis (falling back to pseudotime, exactly as for
unlabelled manifold data). Verified end-to-end against the running engine in the audit; these
pin the seam with a stubbed MIOFlow so they need no stack.
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")

CFLOWS = {"name": "cflows", "steps": [
    {"name": "phate", "group": "latent"},
    {"name": "mioflow", "group": "lightning"},
]}


def _stub(monkeypatch, seen):
    from manyruns import pipeline

    api = types.ModuleType("manylatents.api")
    api.run = lambda **kw: {"embeddings": np.zeros((6, 2)), "scores": {}}
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)

    def fake_mioflow(X, seed, fast_dev_run, params, labels=None, device="cpu"):
        seen["labels"] = labels          # what MIOFlow was told is "time"
        # FOURTH return value: the trained model. `mioflow.py`'s docstring predicted this
        # break — a fake that kept working while the real signature moved would be a test that
        # had stopped testing the seam it was written for.
        return np.asarray(X), {"loss": 0.1}, {}, object()

    monkeypatch.setattr(pipeline.mioflow, "_run_mioflow_experiment", fake_mioflow)


def test_condition_labels_are_not_fed_to_mioflow_as_time(tmp_path, monkeypatch):
    from manyruns import pipeline

    seen = {}
    _stub(monkeypatch, seen)
    labels = np.array(["treated", "treated", "healthy", "healthy", "treated", "healthy"])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path,
        labels=labels, label_kind="condition",
    )
    assert seen["labels"] is None, "MIOFlow got the condition labels as a time axis (C1)"
    assert "mioflow.n_timepoints" not in out["g_vector"]   # no fabricated trajectory
    assert out["status"]["mioflow"] == "ok"                # still runs — on pseudotime


def test_real_time_labels_still_reach_mioflow(tmp_path, monkeypatch):
    from manyruns import pipeline

    seen = {}
    _stub(monkeypatch, seen)
    labels = np.array(["d0", "d0", "d3", "d3", "d7", "d7"])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path,
        labels=labels, label_kind="time",
    )
    assert list(seen["labels"]) == list(labels)            # a real time axis IS used
    assert out["g_vector"]["mioflow.n_timepoints"] == 3    # d0/d3/d7


def test_only_a_time_kind_reaches_mioflow_whatever_the_other_kind_is_called(tmp_path, monkeypatch):
    """The guard was `None if label_kind == "condition" else labels` — a DENYLIST of one, so
    any kind added later fell through the `else` and became a time axis. That is exactly the
    C1 failure this module exists to pin, arriving by nobody's mistake: the next kind is
    `group` (a cell type, a cluster, a branch), which is as much a trajectory as treated/healthy
    is. Tested against a kind this repo does not define, so the property is 'only time', not
    'time and group'."""
    from manyruns import pipeline

    seen = {}
    _stub(monkeypatch, seen)
    labels = np.array(["b1", "b1", "b2", "b2", "b3", "b3"])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path,
        labels=labels, label_kind="a-kind-nobody-has-defined",
    )
    assert seen["labels"] is None, "an unrecognised label kind was fed to MIOFlow as time"
    assert "mioflow.n_timepoints" not in out["g_vector"]
    assert out["status"]["mioflow"] == "ok"


def test_labels_with_no_declared_kind_do_not_reach_mioflow(tmp_path, monkeypatch):
    """The allowlist's other half. Production never produces this — `load_labeled` returns
    either no labels and no kind, or labels WITH a kind — so an untyped label array means a
    caller constructed one, and guessing that it is a time axis is the C1 failure with the
    declaration missing rather than wrong."""
    from manyruns import pipeline

    seen = {}
    _stub(monkeypatch, seen)
    labels = np.array(["day0", "day0", "day1", "day1", "day2", "day2"])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path, labels=labels)

    assert seen["labels"] is None
    assert "mioflow.n_timepoints" not in out["g_vector"]


def test_the_fix_is_label_typing_not_over_pruning():
    """cflows is legal on BOTH manifold and case/control — the guard is that MIOFlow ignores
    a condition axis, NOT that the planner prunes cflows off shapes with no time (which would
    wrongly kill cflows-on-swissroll, a valid pseudotime run)."""
    from manyruns.vocab import unmet

    # Mirrors the real recipe: phate → mioflow, no analysis step. It used to carry
    # `granger`, which was deleted; do NOT substitute another analysis step here, because
    # `separation`/`composition` DECLARE a `conditions` need in `vocab.STEP_NEEDS` and would
    # make `unmet(cflows, "manifold")` non-empty — turning this legality test into an
    # assertion about the substitute rather than about cflows.
    cflows = {"name": "cflows", "steps": [
        {"name": "phate", "group": "latent"}, {"name": "mioflow", "group": "lightning"}]}
    assert unmet(cflows, "manifold") == frozenset()
    assert unmet(cflows, "case-control") == frozenset()
