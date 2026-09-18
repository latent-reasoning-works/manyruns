"""A mioflow step's `params:` reach the engine, or the step says so — manyruns#55.

MEASURED BEFORE THIS FILE EXISTED, on a 40x3 array through `_run_mioflow_experiment`:

    params={"n_bins": 7, "n_trajectories": 25, "lambda_density": 0.5}
    -> model.n_bins == 100, model.n_trajectories == 100, model.lambda_density == 0.0

and `apply_step` recorded `outcome='ok'`. Eight of `MIOFlow.__init__`'s fourteen arguments were
unreachable from a recipe because the call site named its kwargs by hand (7 of 14 — see
`mioflow._RESERVED` for the arithmetic), and the three it did
expose were the only three anybody had needed yet.

TWO THINGS MAKE THAT WORSE THAN A DROPPED SETTING, and both are what this file pins:

1. **The caveat channel was lying.** `bounds.fit` runs before the executor (runner.py:629), so a
   `limits:` line clamped a param the executor then dropped, and wrote a sentence about the
   clamp into `rec['bounded']` and the run's caveats. Measured on a 30x3 embedding with
   `params: {n_trajectories: 250}, limits: {n_trajectories: [n_samples]}` — "mioflow:
   n_trajectories 250 -> 30 ..." in the caveats, `outcome='ok'`, and the trained model at 100.
   The one channel a human reads to decide whether to trust a number was asserting a clamp that
   never reached the engine.
2. **The two groups on one engine disagreed about whether a typo is fatal.** `n_compnents: 3` on
   a `latent` step errors, with manylatents raising manyruns's own house argument back at it
   ("a silently ignored parameter makes two different configurations return identical results",
   `latent_module_base.py:53-58`). `n_globl_epochs: 3` on the `lightning` step ran the default
   and recorded `ok`.

The fix reads the engines' own `inspect.signature` at call time rather than carrying a
forwardable-name list — `bounds.py`'s standing argument ("two homes for one fact", "a table in
Python is unlearnable") applied to a second fact. That only stays safe while the signature stays
CLOSED and the recipe namespaces stay DISJOINT (with the explicit run-owned `init_seed`
exception below), which is why the cheap introspection test below is not
optional: if `MIOFlow.__init__` ever grows `**kwargs`, the leftover set silently empties and the
refusal stops firing — the exact silence #55 is about, restored by an upstream change nobody
here would see.
"""
from __future__ import annotations

import pytest

from manyruns.pipeline import runner as _runner

np = pytest.importorskip("numpy")


def _state(n=30, d=3):
    """A state as a lightning step finds one: an embedding, no real time axis."""
    X = np.asarray(np.random.default_rng(0).normal(size=(n, d)), dtype=np.float32)
    state = _runner._new_state(X=X, seed=0)
    state["emb"] = X.copy()
    return state


def _ctx(tmp_path):
    return {"out_dir": tmp_path, "run_id": "RUN", "plots": [], "target_dim": 2,
            "caveats": [], "seed": 0, "fast_dev_run": True, "device": "cpu"}


def _apply(step, state, tmp_path, ctx=None):
    return _runner.apply_step(step, state, {}, dispatch=_runner.dispatch_for("manylatents"),
                              ctx=ctx or _ctx(tmp_path), index=0, carry=_runner.new_carry())


# ── the ticket ───────────────────────────────────────────────────────────────
def test_a_declared_mioflow_param_reaches_the_engine(tmp_path):
    """manyruns#55, inverted. This is the measurement in the module docstring, run again
    against the fix: what the recipe asked for is what the trained object carries."""
    pytest.importorskip("manylatents")
    pytest.importorskip("torch")
    from manyruns.pipeline import mioflow

    # 80 rows, not 30: `lambda_density` turns on a k-NN hinge loss with a fixed `top_k=5`
    # (`mioflow_net.py:73`), so a timepoint bin thinner than five cells makes the ENGINE raise
    # "selected index k out of range". That it can raise at all is the fix working — the same
    # call on a 30-row array trained happily before, because the param never arrived.
    _, _, _, model = mioflow._run_mioflow_experiment(
        _state(n=80)["emb"], seed=0, fast_dev_run=True,
        params={"n_bins": 7, "n_trajectories": 25, "lambda_density": 0.5, "hidden_dim": 8,
                "n_timepoints": 3})

    assert model.n_bins == 7
    assert model.n_trajectories == 25
    assert model.lambda_density == 0.5
    # `hidden_dim` was the one of the four that already worked (it goes to the network, not to
    # MIOFlow) — asserted here so the split does not lose it. Read off the layer because
    # `MIOFlowODEFunc` keeps no `hidden_dim` attribute, and off `network_config` because
    # `.network` is None until the module builds it.
    assert model.network_config.net[0].out_features == 8


def test_a_bounded_param_reaches_the_engine_clamped(tmp_path):
    """THE FALSE CAVEAT, closed. `bounds.fit` clamps before the executor runs, so the caveat and
    the engine now describe the same run.

    Measured before the fix, this exact step: `rec['bounded']` and `ctx['caveats']` both said
    "mioflow: n_trajectories 250 -> 30 — the recipe caps it at n_samples, and this data is
    30 x 3", `outcome` was `'ok'`, and `state['model'].n_trajectories` was 100. A caveat about a
    clamp that never reached the engine is worse than a silent drop: the silent drop at least
    does not tell a reader it was handled."""
    pytest.importorskip("manylatents")
    pytest.importorskip("torch")
    state, ctx = _state(), _ctx(tmp_path)
    step = {"name": "mioflow", "group": "lightning", "params": {"n_trajectories": 250},
            "limits": {"n_trajectories": ["n_samples"]}}

    rec = _apply(step, state, tmp_path, ctx=ctx)

    assert rec["outcome"] == "ok", rec.get("detail")
    assert rec["params"]["n_trajectories"] == 30            # what the record says ran
    assert any("250 → 30" in c for c in ctx["caveats"])     # what the reader is told
    assert state["model"].n_trajectories == 30              # what actually ran


# ── the refusal ──────────────────────────────────────────────────────────────
def test_an_unknown_mioflow_param_errors_the_step_instead_of_running_the_default(tmp_path):
    """The asymmetry, closed. MEASURED before: the same class of typo on a `latent` step
    (`pca`, `n_compnents: 3`) recorded `outcome='error'`; on this step (`n_globl_epochs: 3`) it
    recorded `outcome='ok'` and trained with manyruns's default. Two groups on one engine
    must not disagree about whether a typo is fatal — a step reporting `ok` having done
    something other than what the recipe declared is the failure `prep.TRANSFORMS`,
    `bounds.check_limits` and `catalog.check_claims` each refuse in their own corner."""
    pytest.importorskip("manylatents")
    pytest.importorskip("torch")
    step = {"name": "mioflow", "group": "lightning", "params": {"n_globl_epochs": 3}}

    rec = _apply(step, _state(), tmp_path)

    assert rec["outcome"] == "error"
    assert "n_globl_epochs" in rec["detail"]
    assert "n_global_epochs" in rec["detail"], "the legal names come from the same signature"


def test_a_param_the_run_owns_is_refused_with_its_own_reason(tmp_path):
    """"You may not set this" and "there is no such param" are different mistakes and a reader
    has to be able to tell them apart. `init_seed` is the sharp one: it is `ctx['seed']`, which
    the lineage records, so a recipe overriding it makes the recorded seed describe a run that
    did not happen."""
    pytest.importorskip("manylatents")
    pytest.importorskip("torch")
    step = {"name": "mioflow", "group": "lightning", "params": {"init_seed": 7}}

    rec = _apply(step, _state(), tmp_path)

    assert rec["outcome"] == "error"
    assert "seed" in rec["detail"]
    assert "--seed" in rec["detail"]
    assert "network at construction" in rec["detail"]
    assert "algorithm wrapper" in rec["detail"]
    assert "unknown" not in rec["detail"], "a reserved param is not a typo"


def test_the_lightning_cli_path_refuses_params_it_cannot_forward(tmp_path):
    """The same defect one branch over. `_ml_lightning`'s non-mioflow branch shells out to
    `manylatents.main` and never passes `params` — so every param on a `cflows`/`latent_ode`
    step was dropped while the step recorded `reported`.

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


# ── what makes reading the signature safe ────────────────────────────────────
def test_the_param_namespaces_are_disjoint_and_the_signature_is_closed():
    """THE GUARD ON THE WHOLE APPROACH, and it is not optional.

    Reading `inspect.signature` instead of keeping a list is only sound while (a) the signature
    is CLOSED — a `**kwargs` on `MIOFlow.__init__` would empty the leftover set and silently
    retire the refusal, which is the exact silence #55 reports — and (b) the namespaces the flat
    `params` block addresses stay DISJOINT, since a name in two of them would be routed by
    precedence rather than by intent.

    `init_seed` is the sole exception: both constructors mean weight initialization, at
    different construction points. The wrapper seeds config construction but preserves an
    existing network. Manyruns constructs that network and passes the recorded run seed to
    both; `params.init_seed` is refused with directions to --seed. The behavior tests below
    pin that reason, not just the spelling. Any new overlap needs its own design decision;
    precedence cannot tell us which constructor a recipe author intended."""
    import inspect

    pytest.importorskip("manylatents")
    from manylatents.algorithms.lightning.mioflow import MIOFlow
    from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

    from manyruns.pipeline import mioflow

    algo = set(inspect.signature(MIOFlow.__init__).parameters) - {"self"}
    net = set(inspect.signature(MIOFlowODEFunc.__init__).parameters) - {"self"}
    local = set(mioflow._TIME_PARAMS)

    assert algo & net == {"init_seed"}, (
        "only the run-owned weight-initialization seed may name both constructors; "
        "a param the recipe means for one constructor would reach two")
    assert "init_seed" in mioflow._RESERVED, "the shared seed must come from the recorded run"
    assert algo & local == set() and net & local == set()
    for owner in (MIOFlow.__init__, MIOFlowODEFunc.__init__):
        kinds = [p.kind for p in inspect.signature(owner).parameters.values()]
        assert inspect.Parameter.VAR_KEYWORD not in kinds, (
            f"{owner.__qualname__} now takes **kwargs — nothing is 'left over' any more, so the "
            "refusal above stopped firing and every typo is silent again")


def test_init_seed_means_weight_initialization_on_both_construction_paths():
    """The exact overlap is safe only while upstream shares this construction contract."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("manylatents")
    from manylatents.algorithms.lightning.mioflow import MIOFlow
    from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        net = MIOFlowODEFunc(input_dim=3, hidden_dim=8, init_seed=0)
        weights = {k: v.clone() for k, v in net.state_dict().items()}

        torch.manual_seed(123)
        configured = MIOFlow(network={
            "_target_": "manylatents.algorithms.lightning.networks.mioflow_net.MIOFlowODEFunc",
            "input_dim": 3, "hidden_dim": 8,
        }, optimizer=None, init_seed=0)
        configured.configure_model()
        assert all(torch.equal(v, configured.network.state_dict()[k])
                   for k, v in weights.items()), "wrapper seeding must seed config construction"

        # A different wrapper seed must NOT reset supplied weights. Only the seed at
        # network construction can initialize them, which is why our call site must pass it.
        supplied = MIOFlow(network=net, optimizer=None, init_seed=7)
        supplied.configure_model()
        assert supplied.network is net
        assert all(torch.equal(v, supplied.network.state_dict()[k]) for k, v in weights.items())


def test_the_run_seed_reaches_network_construction_before_the_experiment(monkeypatch):
    """Same recorded seed, same initial weights regardless of earlier process RNG use.

    Stop at the experiment boundary: training would hide whether construction was seeded.
    Keep both real constructors so dropping either seed at our call site is observable.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("manylatents")
    from manylatents import experiment

    from manyruns.pipeline import mioflow

    snapshots = []

    def capture(*, datamodule, algorithm, trainer, seed):
        assert algorithm.init_seed == seed
        snapshots.append({k: v.clone() for k, v in algorithm.network_config.state_dict().items()})
        return {}

    monkeypatch.setattr(experiment, "run_experiment", capture)
    X = _state()["emb"]
    with torch.random.fork_rng(devices=[]):
        for prior_seed, run_seed in ((99, 0), (123, 0), (99, 7)):
            torch.manual_seed(prior_seed)
            mioflow._run_mioflow_experiment(
                X, seed=run_seed, fast_dev_run=True, params={}, labels=np.repeat([0, 1, 2], 10))

    first, repeated, changed = snapshots
    assert all(torch.equal(v, repeated[k]) for k, v in first.items()), (
        "the run seed arrived after network construction; weights depend on earlier RNG use")
    assert any(not torch.equal(v, changed[k]) for k, v in first.items()), (
        "changing the run seed must change the network's initial weights")


def test_another_constructor_overlap_is_still_refused(monkeypatch):
    """The init_seed decision grants no precedence or permission to future shared names."""
    import inspect

    pytest.importorskip("manylatents")
    from manylatents.algorithms.lightning.mioflow import MIOFlow

    from manyruns.pipeline import mioflow

    signature = inspect.signature(MIOFlow.__init__)
    monkeypatch.setattr(MIOFlow.__init__, "__signature__", signature.replace(parameters=[
        *signature.parameters.values(),
        inspect.Parameter("hidden_dim", inspect.Parameter.KEYWORD_ONLY, default=64),
    ]), raising=False)

    with pytest.raises(ValueError, match="hidden_dim.*name both the algorithm and its network"):
        mioflow._resolve_kwargs({"hidden_dim": 8}, fast_dev_run=True)


def test_manyruns_defaults_survive_a_silent_recipe():
    """The forward must MERGE onto manyruns's defaults, not replace them. The `fast_dev_run`
    epoch counts and the `sample_size=256` subsampling are load-bearing (CPU OT/ODE training on
    a large input), and `params: {}` is what every bundled recipe declares — `cflows.yaml:90` is
    the only lightning step manyruns ships."""
    pytest.importorskip("manylatents")
    from manyruns.pipeline import mioflow

    net, algo = mioflow._resolve_kwargs({}, fast_dev_run=True)
    assert net == {"hidden_dim": 16}
    assert algo["n_global_epochs"] == 2 and algo["n_local_epochs"] == 0
    assert algo["sample_size"] == 256

    _, real = mioflow._resolve_kwargs({}, fast_dev_run=False)
    assert real["n_global_epochs"] == 50 and real["n_local_epochs"] == 10


def test_the_time_axis_params_are_manyruns_own_and_are_not_forwarded():
    """`n_neighbors`/`n_timepoints` build the pseudotime axis MIOFlow is trained against when
    the data carries no real one (`steps._diffusion_pseudotime`). Manyruns genuinely implements
    that bit, so those two names are written down here — our vocabulary, not a copy of anyone's
    catalogue — and they must not be handed to a constructor that would reject them."""
    pytest.importorskip("manylatents")
    from manyruns.pipeline import mioflow

    net, algo = mioflow._resolve_kwargs({"n_neighbors": 4, "n_timepoints": 3}, fast_dev_run=True)

    assert "n_neighbors" not in algo and "n_neighbors" not in net
    assert "n_timepoints" not in algo and "n_timepoints" not in net


def test_an_explicit_null_reaches_the_engine_as_none(tmp_path):
    """`sample_size: null` is the only way to write MIOFlow's own "use every point", so an
    explicit null has to mean null here — the opposite of `prep._param`, which reads one as
    "not declared". MEASURED before: `params={'n_local_epochs': None}` raised `TypeError: int()
    argument must be ... not 'NoneType'` out of manyruns's own coercion, so the null was
    neither honoured nor refused."""
    pytest.importorskip("manylatents")
    from manyruns.pipeline import mioflow

    _, algo = mioflow._resolve_kwargs({"sample_size": None}, fast_dev_run=True)

    assert "sample_size" in algo and algo["sample_size"] is None


def test_every_manyruns_default_stays_overridable_after_the_spread():
    """The claim "all now overridable" held for only two of the four pins.

    MEASURED by mutation: re-appending `"n_local_epochs": ..., "n_global_epochs": ...` AFTER the
    `**declared` spread in `_resolve_kwargs` makes both un-overridable again, and the full suite
    still passed 1090. Nothing observed the regression, because the two tests that exercise a
    declared param both happen to name a param that is not one of the four.

    A default may be supplied BEFORE the spread and never after it — that is the whole ordering
    the fix rests on, and this is what watches it. All four pins at once, so one surviving
    re-append cannot hide behind three that do not.
    """
    pytest.importorskip("manylatents")   # `_resolve_kwargs` imports MIOFlow to read its signature

    from manyruns.pipeline import mioflow

    declared = {"n_local_epochs": 3, "n_global_epochs": 4, "sample_size": 12, "hidden_dim": 8}
    net_kwargs, algo_kwargs = mioflow._resolve_kwargs(declared, fast_dev_run=False)

    assert algo_kwargs["n_local_epochs"] == 3, "a default was re-applied after the spread"
    assert algo_kwargs["n_global_epochs"] == 4, "a default was re-applied after the spread"
    assert algo_kwargs["sample_size"] == 12
    assert net_kwargs["hidden_dim"] == 8


def test_every_keyword_the_call_site_passes_by_hand_is_reserved():
    """`_RESERVED` is subtracted from the two signatures, so a name in it is refused with a
    reason and a name absent from it is settable by a recipe. The invariant nothing checked:
    every keyword `_run_mioflow_experiment` passes LITERALLY must be reserved — otherwise a
    recipe can declare it, `_resolve_kwargs` merges it into the spread, and the constructor gets
    the same keyword twice.

    MEASURED by mutation: renaming `_RESERVED`'s `input_dim` key left the full suite green at
    1090, because `input_dim` is in `MIOFlowODEFunc`'s signature and so simply became settable —
    nothing compared the two lists. The resulting failure is loud but fires on a recipe at run
    time, and only for a recipe that happens to name it.

    Read off the SOURCE rather than restated, because a list here would be the third copy of the
    same fact and the one nobody updates.
    """
    pytest.importorskip("manylatents")   # reads MIOFlow's and MIOFlowODEFunc's signatures

    import ast
    import inspect

    from manyruns.pipeline import mioflow

    tree = ast.parse(inspect.getsource(mioflow._run_mioflow_experiment))
    literal_kwargs = {
        kw.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) in {"MIOFlow", "MIOFlowODEFunc"}
        for kw in node.keywords
        if kw.arg is not None          # `**algo_kwargs` has arg=None; that is the spread
    }

    assert literal_kwargs, "found no construction call — this test has stopped watching anything"
    unreserved = sorted(literal_kwargs - set(mioflow._RESERVED))
    assert unreserved == [], (
        f"{unreserved} is passed by hand at construction but absent from `_RESERVED`, so a "
        "recipe may also declare it and the constructor receives it twice")
