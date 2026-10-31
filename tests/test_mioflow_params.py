"""A mioflow step's `params:` reach the tool, or the step says so — manyruns#55, on the tool route.

MEASURED BEFORE manyruns#55 WAS FIXED, on the in-process route that has since been removed:
`params={"n_bins": 7, "n_trajectories": 25, "lambda_density": 0.5}` trained a model with
100 / 100 / 0.0, and `apply_step` recorded `outcome='ok'`. Two failures came with it, and this
file pins that neither returns on the tool route:

1. **The caveat channel lied.** `bounds.fit` runs before the executor, so a `limits:` line
   clamped a param the executor then dropped, and the caveats told a reader about a clamp that
   never reached the engine.
2. **A typo ran the default.** `n_globl_epochs: 3` recorded `ok`.

A `mioflow` step now runs only as a tool step (`tool: mioflow`). The params reach the tool's
`request.json`, and the pinned adapter declines a parameter outside its allow-list with a
reason. Every test here runs the stub MIOFlow folder (`tests/fixtures/mioflow_tool/`), whose
adapter keeps the real adapter's allow-list and its reasons, under the current interpreter.

The tests of the in-process parameter routing (`_resolve_kwargs`, `_RESERVED`, `_TIME_PARAMS`
and the signature introspection of the manylatents MIOFlow port) are retired with that route:
requirement "The in-process route is gone".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from manyruns.pipeline import runner as _runner

np = pytest.importorskip("numpy")

STUB = Path(__file__).resolve().parent / "fixtures" / "mioflow_tool"


@pytest.fixture(autouse=True)
def _stub_tool(monkeypatch):
    assert STUB.is_dir(), "stub MIOFlow folder moved"
    monkeypatch.delenv("MANYRUNS_TOOL_DIR", raising=False)
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", str(STUB))
    monkeypatch.setenv("MANYRUNS_TOOL_MIOFLOW_PYTHON", sys.executable)


def _state(n=30, d=3):
    """A state as the tool step finds one: an embedding and a real time axis."""
    X = np.asarray(np.random.default_rng(0).normal(size=(n, d)), dtype=np.float32)
    state = _runner._new_state(X=X, seed=0)
    state["emb"] = X.copy()
    state["labels"] = np.repeat([0.0, 1.0, 2.0], n // 3)
    state["label_kind"] = "time"
    return state


def _ctx(tmp_path):
    return {"out_dir": tmp_path, "run_id": "RUN", "plots": [], "target_dim": 2,
            "caveats": [], "seed": 0, "fast_dev_run": True, "device": "cpu"}


def _apply(step, state, tmp_path, ctx=None):
    return _runner.apply_step(step, state, {}, dispatch=_runner.dispatch_for("manylatents"),
                              ctx=ctx or _ctx(tmp_path), index=0, carry=_runner.new_carry())


def _request(tmp_path) -> dict:
    assert tmp_path.is_dir()
    [path] = list(tmp_path.rglob("request.json"))
    return json.loads(path.read_text())


def _mioflow(params, **extra):
    return {"name": "mioflow", "group": "lightning", "tool": "mioflow", "params": params, **extra}


# ── the ticket ───────────────────────────────────────────────────────────────
def test_a_declared_mioflow_param_reaches_the_tool(tmp_path):
    """manyruns#55, inverted, on the tool route: what the recipe asked for is what the tool
    is asked to run with."""
    params = {"n_bins": 7, "n_trajectories": 25, "lambda_density": 0.5, "hidden_dim": 8}

    rec = _apply(_mioflow(params), _state(), tmp_path)

    assert rec["outcome"] == "ok", rec.get("detail")
    assert _request(tmp_path)["params"] == params


def test_a_bounded_param_reaches_the_tool_clamped(tmp_path):
    """THE FALSE CAVEAT stays closed. `bounds.fit` clamps before the executor runs, so the
    caveat, the record and the tool's request describe the same run."""
    state, ctx = _state(), _ctx(tmp_path)
    step = _mioflow({"n_trajectories": 250}, limits={"n_trajectories": ["n_samples"]})

    rec = _apply(step, state, tmp_path, ctx=ctx)

    assert rec["outcome"] == "ok", rec.get("detail")
    assert rec["params"]["n_trajectories"] == 30            # what the record says ran
    assert any("250 → 30" in c for c in ctx["caveats"])     # what the reader is told
    assert _request(tmp_path)["params"]["n_trajectories"] == 30   # what the tool was asked


# ── the refusal ──────────────────────────────────────────────────────────────
def test_an_unknown_mioflow_param_declines_the_step_instead_of_running_the_default(tmp_path):
    """The typo is not silent. The pinned adapter declines a parameter outside its allow-list,
    so the step records `skipped` with a reason that names the parameter and the legal names,
    instead of `ok` with a default."""
    rec = _apply(_mioflow({"n_globl_epochs": 3}), _state(), tmp_path)

    assert rec["outcome"] == "skipped", rec
    assert "n_globl_epochs" in rec["detail"]
    assert "n_epochs" in rec["detail"], "the legal names come from the adapter's allow-list"


def test_the_run_seed_reaches_the_tool_and_a_recipe_cannot_set_its_own(tmp_path):
    """`init_seed` would make the recorded seed describe a run that did not happen. On the tool
    route the run's seed travels in the request, and `init_seed` is outside the allow-list."""
    ctx = _ctx(tmp_path)
    ctx["seed"] = 7
    rec = _apply(_mioflow({}), _state(), tmp_path, ctx=ctx)
    assert rec["outcome"] == "ok", rec.get("detail")
    assert _request(tmp_path)["seed"] == 7

    other = tmp_path / "other"
    other.mkdir()
    rec = _apply(_mioflow({"init_seed": 7}), _state(), other)
    assert rec["outcome"] == "skipped" and "init_seed" in rec["detail"]


def test_the_lightning_cli_path_refuses_params_it_cannot_forward(tmp_path):
    """The same defect one branch over. `_ml_lightning` shells out to `manylatents.main` for a
    lightning name other than `mioflow`, and never passes `params` — so every param on a
    `cflows`/`latent_ode` step was dropped while the step recorded `reported`.

    `skipped` rather than `error` here, and the distinction is the vocabulary's: the params are
    not wrong, the ROUTE cannot carry them. Synthesising Hydra overrides instead would need the
    config-group path per algorithm, which is a product-side copy of the engine's config
    layout — the thing this whole change exists to not do."""
    step = {"name": "cflows", "group": "lightning", "params": {"n_bins": 7}}
    ctx = _ctx(tmp_path)
    ctx["data_ref"] = "swissroll"          # the CLI path's own precondition, satisfied

    rec = _apply(step, _state(), tmp_path, ctx=ctx)

    assert rec["outcome"] == "skipped"
    assert "n_bins" in rec["detail"]
    assert "forwards no params" in rec["detail"]
