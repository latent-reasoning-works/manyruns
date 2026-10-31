"""C1 regression: a CONDITION axis must never be fed to MIOFlow as a TIME axis.

The trace of a real cflows run on case/control data showed `mioflow.n_timepoints: 2` — the
two disease groups (treated/healthy) mapped to two timepoints, training a fabricated
trajectory "from disease to health over time". `load_labeled` now returns the label KIND, and
only `label_kind == "time"` reaches the trajectory step as time.

MIOFlow now runs only as a tool step, through the model seam. The request carries a `time.npy`
only for a time axis (`external._write_request`), and the MIOFlow tool declines without one —
there is no pseudotime fallback any more. These tests pin that seam with the stub MIOFlow folder
(`tests/fixtures/mioflow_tool/`), so they need no MIOFlow and no stack: what the step was told is
"time" is the `time.npy` in its work folder.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

STUB = Path(__file__).resolve().parent / "fixtures" / "mioflow_tool"

CFLOWS = {"name": "cflows", "steps": [
    {"name": "phate", "group": "latent"},
    {"name": "mioflow", "group": "lightning", "tool": "mioflow"},
]}


def _stub(monkeypatch):
    """A fake latent engine, and the stub MIOFlow tool on the tool search path."""
    api = types.ModuleType("manylatents.api")
    api.run = lambda **kw: {"embeddings": np.zeros((6, 2)), "scores": {}}
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)
    assert STUB.is_dir()
    monkeypatch.delenv("MANYRUNS_TOOL_DIR", raising=False)
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", str(STUB))
    monkeypatch.setenv("MANYRUNS_TOOL_MIOFLOW_PYTHON", sys.executable)


def _sent_time(folder: Path):
    """The time axis the tool step was handed, or None when the request carried none."""
    assert folder.is_dir()
    sent = list(folder.rglob("time.npy"))
    assert len(sent) <= 1
    return np.load(sent[0], allow_pickle=False) if sent else None


def _mioflow(out):
    [rec] = [s for s in out["steps"] if s["name"] == "mioflow"]
    return rec


def _declined_without_time(out, tmp_path):
    assert _sent_time(tmp_path) is None, "MIOFlow got a non-time label axis as time (C1)"
    rec = _mioflow(out)
    assert rec["outcome"] == "skipped" and "needs a time axis" in rec["detail"]
    assert "mioflow.n_bins" not in out["g_vector"]       # no fabricated trajectory


def test_condition_labels_are_not_fed_to_mioflow_as_time(tmp_path, monkeypatch):
    from manyruns import pipeline

    _stub(monkeypatch)
    labels = np.array(["treated", "treated", "healthy", "healthy", "treated", "healthy"])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path,
        labels=labels, label_kind="condition",
    )
    _declined_without_time(out, tmp_path)


def test_real_time_labels_still_reach_mioflow(tmp_path, monkeypatch):
    from manyruns import pipeline

    _stub(monkeypatch)
    # NUMERIC, because the seam sends a time axis it can read as numbers
    # (`external._time_axis`); the loader turns `d0/d3/d7` into 0/3/7 before this point.
    labels = np.array([0.0, 0.0, 3.0, 3.0, 7.0, 7.0])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path,
        labels=labels, label_kind="time",
    )
    assert list(_sent_time(tmp_path)) == list(labels)      # a real time axis IS used
    assert _mioflow(out)["outcome"] == "ok", _mioflow(out)
    assert out["g_vector"]["mioflow.n_bins"] == 4          # the stub's bins


def test_only_a_time_kind_reaches_mioflow_whatever_the_other_kind_is_called(tmp_path, monkeypatch):
    """The guard was `None if label_kind == "condition" else labels` — a DENYLIST of one, so
    any kind added later fell through the `else` and became a time axis. That is exactly the
    C1 failure this module exists to pin, arriving by nobody's mistake: the next kind is
    `group` (a cell type, a cluster, a branch), which is as much a trajectory as treated/healthy
    is. Tested against a kind this repo does not define, so the property is 'only time', not
    'time and group'."""
    from manyruns import pipeline

    _stub(monkeypatch)
    labels = np.array(["b1", "b1", "b2", "b2", "b3", "b3"])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path,
        labels=labels, label_kind="a-kind-nobody-has-defined",
    )
    _declined_without_time(out, tmp_path)


def test_labels_with_no_declared_kind_do_not_reach_mioflow(tmp_path, monkeypatch):
    """The allowlist's other half. Production never produces this — `load_labeled` returns
    either no labels and no kind, or labels WITH a kind — so an untyped label array means a
    caller constructed one, and guessing that it is a time axis is the C1 failure with the
    declaration missing rather than wrong."""
    from manyruns import pipeline

    _stub(monkeypatch)
    labels = np.array(["day0", "day0", "day1", "day1", "day2", "day2"])

    out = pipeline.run_manylatents(
        CFLOWS, array=np.zeros((6, 4)), out_dir=tmp_path, labels=labels)

    _declined_without_time(out, tmp_path)


def test_cflows_needs_a_declared_time_axis_on_every_shape_without_one(monkeypatch):
    """The C1 fix was label typing, and it still is. What changed is the fallback: cflows used
    to be legal on manifold and case/control data, running MIOFlow on an invented pseudotime.
    The MIOFlow tool needs `time`, so with the tool declared, cflows is legal on a time course
    and blocked on `time` everywhere else — a refusal at plan time, not a fabricated clock."""
    from manyruns.vocab import unmet

    assert STUB.is_dir()
    monkeypatch.delenv("MANYRUNS_TOOL_DIR", raising=False)
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", str(STUB))
    assert unmet(CFLOWS, "time-course") == frozenset()
    assert unmet(CFLOWS, "manifold") == frozenset({"time"})
    assert unmet(CFLOWS, "case-control") == frozenset({"time"})
