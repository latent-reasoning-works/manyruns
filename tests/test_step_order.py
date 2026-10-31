"""An illegal step ORDER must be refused, and refused the SAME WAY by every engine.

`vocab.unmet` is the legal-move function. When this file was written it modelled only the
DATA axis — it unioned `STEP_NEEDS` and asked what a dataset's shape cannot provide, so
`mioflow -> phate` came back `frozenset()`, i.e. legal, and the engine's own refusal was the
only thing standing between that recipe and a published number. `unmet` has since been made
order-aware (`vocab.GROUP_PROVIDES`, folded along the recipe instead of unioned over it), so
the same recipe is now refused at PLAN time as well. The engine refusal is not thereby
redundant: `unmet` is advisory and every run path can be reached without consulting it.
The last test in this file pins BOTH levels — see its docstring for why that is the invariant
rather than the earlier "the fix is at the engine, not in unmet".

Measured before the fix, on the in-process loop, 120x8 gaussian noise, recipe `mioflow -> phate`:
every step reported `outcome=ok`, the run reported `ok=True`, and the g-vector carried
`pseudotime_range = [0.0, 1.0]` — a trajectory computed on the RAW input, because
`_step_mioflow` read `state["emb"] if ... is not None else state["X"]`. The manylatents
engine had refused this all along. Two engines, one recipe, opposite answers, and the
disagreeing one produced a number that the record could not distinguish from a real one.

These tests pin the refusal on BOTH engines and pin that the two give the same reason, so
the rule cannot drift back into one engine only. They are dep-free: the refusal happens
before any scientific import, which is itself part of the contract.
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")

#: The backwards recipe. `unmet` calls this legal (see the last test), so the engine's own
#: refusal is the only thing standing between it and a published pseudotime.
BACKWARDS = {"name": "backwards", "steps": [
    {"name": "mioflow", "group": "lightning", "params": {}},
    {"name": "phate", "group": "latent", "params": {"n_components": 3}},
]}


def _fake_manylatents(monkeypatch):
    """A manylatents.api that would happily embed if it were ever reached."""
    api = types.ModuleType("manylatents.api")
    api.run = lambda **kw: {"embeddings": np.zeros((6, 2)), "scores": {}}
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)


def test_the_in_process_loop_refuses_mioflow_before_a_latent_step(tmp_path):
    """THE regression. Before the fix this returned outcome=ok and a full-range pseudotime."""
    from manyruns import pipeline

    out = pipeline.run_inproc(np.zeros((12, 4)), BACKWARDS, tmp_path, seed=0)

    mioflow = out["steps"][0]
    assert mioflow["name"] == "mioflow"
    assert mioflow["outcome"] == "skipped", "mioflow ran with no coordinates to run on"
    # the refusal names the missing OBJECT, not a missing step: coordinates supplied
    # from anywhere satisfy it (see tests/test_supplied_embedding.py)
    assert "needs an embedding to run on" in mioflow["detail"]
    # The number itself must be ABSENT, not merely flagged: a g-vector carrying a
    # pseudotime range is indistinguishable downstream from one a legal run produced.
    assert "pseudotime_range" not in out["g_vector"]
    assert out["ok"] is False, "a run containing a refused step must not report ok"


def test_the_manylatents_engine_refuses_it_too(tmp_path, monkeypatch):
    """The engine that already had the rule keeps it — the fix moved the check, so this
    guards against the move having quietly dropped it."""
    from manyruns import pipeline

    _fake_manylatents(monkeypatch)
    out = pipeline.run_manylatents(BACKWARDS, array=np.zeros((6, 4)), out_dir=tmp_path)

    assert out["steps"][0]["outcome"] == "skipped"
    assert "needs an embedding to run on" in out["steps"][0]["detail"]


def test_both_engines_give_the_SAME_reason(tmp_path, monkeypatch):
    """One rule, one home, one sentence. Divergent refusal text is how a single rule becomes
    two rules that drift — which is exactly how the real engine came to have none."""
    from manyruns import pipeline

    _fake_manylatents(monkeypatch)
    real = pipeline.run_inproc(np.zeros((12, 4)), BACKWARDS, tmp_path, seed=0)
    ml = pipeline.run_manylatents(BACKWARDS, array=np.zeros((6, 4)), out_dir=tmp_path)

    assert real["steps"][0]["detail"] == ml["steps"][0]["detail"]
    assert real["status"]["mioflow"] == ml["status"]["mioflow"]


def test_the_legal_order_still_runs_on_the_in_process_loop(tmp_path, monkeypatch):
    """The guard must refuse the backwards recipe WITHOUT breaking the forward one.

    The latent step is stubbed rather than real PHATE so this stays dep-free in the half
    that matters; mioflow itself is the genuine `_step_mioflow`, so scipy/sklearn are
    required and skipped-for when absent."""
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    from manyruns import pipeline

    def _embed(state, g, params, out_dir, plots, target_dim):
        state["emb"] = np.asarray(state["X"])[:, :2]

    monkeypatch.setitem(pipeline._INPROC_STEPS, "stub_latent", _embed)
    forward = {"name": "forward", "steps": [
        {"name": "stub_latent", "group": "latent", "params": {}},
        {"name": "mioflow", "group": "lightning", "params": {}},
    ]}
    rng = np.random.default_rng(0)
    out = pipeline.run_inproc(rng.normal(size=(40, 5)), forward, tmp_path, seed=0)

    assert out["status"]["mioflow"] == "ok"
    assert "pseudotime_range" in out["g_vector"]


def test_the_refusal_is_at_BOTH_the_calculus_and_the_ENGINE(tmp_path):
    """Both levels refuse, and NEITHER is redundant.

    This assertion was inverted deliberately. As committed in df478cc it read
    `unmet(BACKWARDS, ...) == frozenset()` and recorded that leaving `unmet` order-blind was
    a scope line, not an oversight — "making it order-aware is a separate design change".
    That change has since been made: `vocab.GROUP_PROVIDES` declares that a `latent` step
    produces an `embedding`, `STEP_NEEDS` declares that `mioflow` requires one, and `unmet`
    folds along the recipe, so the backwards recipe is now illegal at plan time. The old
    assertion and the new behaviour cannot both hold; the two-level refusal is the one worth
    keeping, so the assertion moves rather than the behaviour.

    The engine check is still load-bearing. `unmet` prunes at PLAN time and only for callers
    that consult it (`experiment.plan`, `shell._explore`, `app.cmd_check`); `run_inproc`
    and `run_manylatents` are reachable directly — the four tests above call them that way —
    so deleting `_require_embedding` would restore the fabrication for every direct caller
    while this test still passed on the calculus half alone. Hence both halves, in one test:
    if either level stops refusing, this fails."""
    from manyruns import pipeline
    from manyruns.vocab import unmet

    # Level 1 — declarative. The recipe is illegal before anything runs, on every shape,
    # and the missing fact is NAMED so a caller can say why.
    for shape in ("manifold", "time-course", "case-control", "trajectory", "clusters", "unknown"):
        assert unmet(BACKWARDS, shape) == frozenset({"embedding"}), shape

    # Level 2 — imperative, and independent of level 1: this path never calls `unmet`.
    out = pipeline.run_inproc(np.zeros((12, 4)), BACKWARDS, tmp_path, seed=0)
    assert out["steps"][0]["outcome"] == "skipped"
    assert "pseudotime_range" not in out["g_vector"]
