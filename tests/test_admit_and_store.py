"""The admission gate and the run store.

The gate exists because an audit measured ~16 of 38 callables returning a
confident number for structurally wrong input. A harness that runs everything without it
produces plausible nonsense at 38x throughput.

The store exists because it was deferred *until there was a reader* — and the
panel and the gate are now readers.
"""
from __future__ import annotations

import pytest

from manyruns import admit, store

np = pytest.importorskip("numpy")


# ── the gate ─────────────────────────────────────────────────────────────────
def test_a_tool_that_never_fails_is_rejected():
    """THE check. A metric returning 0.7 for the string "not a matrix" will return 0.7 for a
    cohort too — that is what makes it unbelievable, not whether the number looks sensible."""
    v = admit.admit("always_confident", lambda X: 0.7)
    assert v.verdict == "rejected"
    assert v.checks["raises"] is False
    assert not v.ok


def test_a_nondeterministic_tool_is_flagged_but_not_rejected():
    """A sweep compares its own cells, so an unseeded tool cannot participate. It is still
    reported rather than hidden: a number you know not to compare remains usable."""
    counter = {"n": 0}

    def drifts(X):
        if not hasattr(X, "shape"):
            raise TypeError("not a matrix")
        counter["n"] += 1
        return float(counter["n"])

    v = admit.admit("drifts", drifts)
    assert v.verdict == "flagged"
    assert v.checks["raises"] is True and v.checks["stable"] is False
    assert v.ok                      # flagged does not fail a build


def test_a_tool_blind_to_structure_versus_noise_is_flagged():
    """The check four candidate readouts failed against a matched control this session."""
    def constant(X):
        if not hasattr(X, "shape"):
            raise TypeError("not a matrix")
        return 1.0

    v = admit.admit("constant", constant)
    assert v.checks["null"] is False
    assert "identical on structure and noise" in " ".join(v.notes)


def test_a_well_behaved_tool_is_admitted():
    """Variance concentrated in the top two components: high for a ring (2-D structure in an
    8-D ambient), low for iid noise. Deterministic, raises on garbage, separates."""
    def top2_variance(X):
        if not hasattr(X, "shape"):
            raise TypeError("not a matrix")
        s = np.linalg.svd(np.asarray(X) - np.asarray(X).mean(0), compute_uv=False)
        return float((s[:2] ** 2).sum() / (s**2).sum())

    v = admit.admit("top2_variance", top2_variance)
    assert v.verdict == "admitted", v.notes
    assert all(v.checks.values())
    assert v.ok


def test_nan_is_not_stability():
    """NaN == NaN is False, so a naive equality check calls two NaNs unstable — but a tool
    that only ever returns NaN would otherwise look 'stable' under any tolerant comparison.
    Both readings are wrong; it is reported as the failure it is."""
    v = admit.admit("nan", lambda X: float("nan") if hasattr(X, "shape") else 1 / 0)
    assert v.checks["stable"] is False
    assert "NaN" in " ".join(v.notes)


def test_the_gate_defers_rather_than_faking_when_the_engine_is_absent(monkeypatch):
    """Same discipline as `catalog.check_suite`: a gate that passes everything on a
    stackless machine is worse than no gate, because it reads as a clean bill of health."""
    import builtins

    real_import = builtins.__import__

    def no_manylatents(name, *a, **k):
        if name.startswith("manylatents"):
            raise ImportError("no manylatents")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_manylatents)
    verdicts = admit.admit_metrics(["lid"])
    assert [v.verdict for v in verdicts] == ["deferred"]
    assert verdicts[0].ok            # deferred is not a failure...
    assert verdicts[0].checks == {}  # ...but it claims no checks either


# ── the store ────────────────────────────────────────────────────────────────
def _result(recipe="contrast", ok=True):
    return {"recipe": recipe, "engine": "_inproc", "seed": None, "tool_version": "0.1.0",
            "ok": ok, "steps": [{"name": "phate", "outcome": "ok", "seconds": 1.0}],
            "g_vector": {"n_samples": 300}}


def test_runs_append_and_never_overwrite(tmp_path):
    """The failure mode of a run store is losing history to an overwrite."""
    store.append(_result("a"), out_dir=tmp_path)
    store.append(_result("b"), out_dir=tmp_path)
    rows = list(store.read(tmp_path))
    assert [r["recipe"] for r in rows] == ["a", "b"]


def test_a_corrupt_line_does_not_destroy_the_history(tmp_path):
    """A half-written row from a killed worker must not make the file unreadable."""
    store.append(_result("a"), out_dir=tmp_path)
    with open(store.index_path(tmp_path), "a") as fh:
        fh.write('{"truncated": \n')
    store.append(_result("b"), out_dir=tmp_path)

    assert [r["recipe"] for r in store.read(tmp_path)] == ["a", "b"]


def test_the_store_holds_the_record_not_the_payload(tmp_path):
    """Records already carry descriptions rather than matrices (see `watch.describe`), which
    is what keeps the index greppable after a thousand runs."""
    store.append(_result(), out_dir=tmp_path)
    row = next(iter(store.read(tmp_path)))

    assert row["steps"][0]["name"] == "phate"
    assert row["tool_version"] == "0.1.0"
    assert len(store.index_path(tmp_path).read_text()) < 1000


def test_reading_an_absent_index_is_empty_not_an_error(tmp_path):
    assert list(store.read(tmp_path / "nope")) == []
