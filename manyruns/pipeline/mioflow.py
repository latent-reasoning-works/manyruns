"""MIOFlow: the real trained trajectory step, plus the manylatents CLI shell-out.

Quarantined in its own module ON PURPOSE. CLAUDE.md says training loops belong in
the learner/manylatents, and this hand-rolls a Lightning Trainer because the engine exposes
no trainer factory in Python (only a Hydra-internal builder; `run_experiment` demands an
already-built Trainer). Keeping the violation in one named file makes it visible and the
eventual upstream push mechanical, rather than burying it mid-pipeline.
"""
from __future__ import annotations

from typing import Any  # noqa: F401 - used by the nested datamodule shims

from manyruns.pipeline.steps import _diffusion_pseudotime

# NOTE (surfaced by the split): the REAL trained step depends on `_diffusion_pseudotime`,
# which lives with the public-library stand-ins — it computes MIOFlow's time label when the
# data carries no real time axis. So the trained path and the no-private-stack path share a
# geometry primitive. Acyclic (steps never imports mioflow) but worth naming: changing that
# pseudotime silently changes what MIOFlow is trained against.

#: Params this file consumes ITSELF and never forwards — the time axis it builds when the data
#: carries no real one (`steps._diffusion_pseudotime`; read at :199 and :201, and only when the
#: caller supplied no real timepoints). Written down here, unlike the two sets `_resolve_kwargs`
#: derives from a signature, because manyruns genuinely implements this bit: it is our
#: vocabulary, not a copy of somebody else's catalogue. A third one added below without being
#: added here reads as `unknown` and errors the step, which is the loud direction to fail in.
_TIME_PARAMS = ("n_neighbors", "n_timepoints")

#: Constructor args the RUN owns, not the recipe — refused with their own reason, because "you
#: may not set this" and "there is no such param" are different mistakes and a reader has to be
#: able to tell them apart. `init_seed` is the sharp one: it is `ctx["seed"]`, which the lineage
#: records, so a recipe overriding it makes the recorded seed describe a run that did not happen.
_RESERVED = {
    "network": "manyruns builds it from the data; set `hidden_dim` for its width",
    "input_dim": "the embedding's width, which is a fact about the data, not a choice",
    "optimizer": "manyruns builds it (Adam, lr=1e-3)",
    "datamodule": "manyruns builds it from the state's embedding and time labels",
    "init_seed": "the run's seed (--seed), which the lineage records; manyruns passes it to "
                 "the network at construction and to the algorithm wrapper",
}


def _resolve_kwargs(params: dict, fast_dev_run: bool) -> tuple[dict, dict]:
    """A mioflow step's flat `params`, routed to the two constructors that share the namespace.

    Returns `(network_kwargs, algorithm_kwargs)`, each merged ON TOP of manyruns's defaults —
    the `fast_dev_run` epoch counts and the `sample_size` subsampling are load-bearing (see the
    call site) and a recipe that names neither must still get them.

    THE TABLE IS THE ENGINE'S OWN SIGNATURE, read at call time. A tuple of forwardable names
    written down here would be a product-side copy of manylatents' constructor — the shape
    `bounds.py` refuses at length ("two homes for one fact… a table in Python is unlearnable")
    and `vocab.py` refuses twice — and it would drift in exactly the direction that produced
    manyruns#55: the hand-written kwarg list at the call site exposed 3 of MIOFlow's 14
    arguments, and MEASURED, `params: {n_bins: 7, n_trajectories: 25, lambda_density: 0.5}`
    trained a model with 100 / 100 / 0.0 while the step recorded `outcome='ok'`.

    Anything left over RAISES, so `apply_step` records `error`. The `latent` group already errors
    on exactly this class of typo — manylatents raises manyruns's own house argument back at it
    ("a silently ignored parameter makes two different configurations return identical results",
    `latent_module_base.py:53-58`) — and two groups on one engine must not disagree about whether
    a typo is fatal. It also closes a worse hole: `bounds.fit` runs BEFORE the executor
    (runner.py:629), so a `limits:` line on a dropped param clamped it, wrote "mioflow:
    n_trajectories 250 → 30 …" into `rec['bounded']` and the run's caveats, and the engine ran
    100 — the one channel a human reads to decide whether to trust a number, asserting a clamp
    that never reached the engine.
    """
    import inspect

    from manylatents.algorithms.lightning.mioflow import MIOFlow
    from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc

    def _accepts(fn) -> set:
        return {n for n, p in inspect.signature(fn).parameters.items()
                if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}

    algo_names = _accepts(MIOFlow.__init__) - set(_RESERVED)
    net_names = _accepts(MIOFlowODEFunc.__init__) - set(_RESERVED)
    declared = dict(params or {})

    reserved = sorted(k for k in declared if k in _RESERVED)
    if reserved:
        raise ValueError("mioflow: " + "; ".join(
            f"`{k}` is not the recipe's to set — {_RESERVED[k]}" for k in reserved))
    # The signatures overlap only at `init_seed`: one weight-initialization seed, applied
    # at network construction or by the wrapper when it builds a config. We supply an
    # existing network, so the run explicitly seeds BOTH calls below; a wrapper cannot
    # retroactively seed its weights. That name is reserved, not a flat recipe override.
    # `tests/test_mioflow_params.py` pins this exact exception and its construction semantics.
    # Any other overlap is refused, never resolved by precedence.
    both = sorted((algo_names & net_names) & set(declared))
    if both:
        raise ValueError(f"mioflow: {both} name both the algorithm and its network — the flat "
                         "`params` block cannot say which was meant")
    unknown = sorted(set(declared) - algo_names - net_names - set(_TIME_PARAMS))
    if unknown:
        raise ValueError(
            f"mioflow: unknown param(s) {unknown}. Check the spelling — a silently ignored "
            "parameter makes two different configurations return identical results. This step "
            f"takes {sorted(algo_names | net_names | set(_TIME_PARAMS))}")

    # NO NUMERIC COERCION SURVIVES HERE, and that is four changes rather than one. The old call
    # site wrapped every value in `int(...)`; none of them does now, so `sample_size: "256"` and
    # `n_local_epochs: "5"` reach the constructor as STRINGS where they used to be coerced, and
    # `sample_size: 0` reaches it as 0 where `int(x) if x else None` silently read it as "use
    # every point" — an inversion of meaning for that one value. YAML types these when a human
    # writes them, which is why the trade is worth it: a recipe that declares a number gets that
    # number, and one that declares nonsense finds out at the engine rather than after a run.
    #
    # `k in declared`, NOT `declared.get(k)`: an explicit `null` means null here. `prep._param`
    # (prep.py:45-49) chose the opposite ("an explicit None means not declared"), which is right
    # for a transform's optional field and wrong for this one — `sample_size: null` is the only
    # way to write MIOFlow's own "use every point". It also removes a crash: the previous
    # `int(params.get("n_local_epochs", …))` turned `n_local_epochs: null` into `TypeError:
    # int() argument must be … not 'NoneType'`, so the null was neither honoured nor refused.
    # No coercion either — YAML already types these, and `int(sample_size) if sample_size else
    # None` silently read `sample_size: 0` as "every point".
    #
    # `hidden_dim` is 16 against the network's own default of 64: this trains on CPU inside an
    # interactive session, so the width is manyruns's call for the same reason `sample_size` is.
    net_kwargs = {"hidden_dim": 16, **{k: declared[k] for k in net_names if k in declared}}
    algo_kwargs = {
        # fast_dev_run runs a single batch (a plumbing check — the flow barely moves, so the
        # embedding ≈ input + a constant drift). A real flow needs actual training epochs; use
        # meaningful defaults when not in smoke mode, and subsample per step so CPU OT/ODE
        # training stays tractable on large inputs.
        "n_local_epochs": 0 if fast_dev_run else 10,
        "n_global_epochs": 2 if fast_dev_run else 50,
        "sample_size": 256,
        # Three here plus `hidden_dim` above — four pins, and only four. The old call site also
        # passed `n_post_local_epochs=0`, `lambda_ot=1.0` and `lambda_energy=0.01`; MEASURED
        # against `inspect.signature(MIOFlow.__init__)`, all three are the engine's own defaults
        # restated, so dropping them changes nothing today and stops manyruns from pinning a
        # value it never chose. The four that stay are ones manyruns DOES choose — smoke mode,
        # and CPU tractability — and the merge below lets a recipe override any of them.
        **{k: declared[k] for k in algo_names if k in declared},
    }
    return net_kwargs, algo_kwargs


def _lightning_cmd(name: str, dataset: str, fast_dev_run: bool) -> list[str]:
    """The manylatents CLI invocation for a Lightning module (MIOFlow, …). Trained
    modules aren't name-addressable via `run()`, so they go through Hydra config groups —
    mirroring `python -m manylatents.main algorithms/lightning=mioflow data=swissroll`."""
    import sys

    cmd = [sys.executable, "-m", "manylatents.main", f"algorithms/lightning={name}", f"data={dataset}"]
    if fast_dev_run:
        # `+` appends: manylatents' trainer config doesn't predeclare fast_dev_run
        cmd.append("+trainer.fast_dev_run=true")
    return cmd


def _run_lightning_cli(name: str, dataset: str, fast_dev_run: bool) -> str:
    """Run a Lightning step via the manylatents CLI; return a status string."""
    import os
    import subprocess

    cmd = _lightning_cmd(name, dataset, fast_dev_run)
    env = {**os.environ, "HYDRA_FULL_ERROR": "1"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    except Exception as e:  # noqa: BLE001 - report + continue
        return f"error: {type(e).__name__}: {e}"
    if r.returncode == 0:
        return f"ok (manylatents.main {' '.join(cmd[3:])})"
    tail = " | ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])
    return f"error (exit {r.returncode}): {tail[-400:]}"


def trajectories_of(model) -> Any:
    """The sample paths a fitted flow left behind, as `(n_bins, n_trajectories, d)` — or None.

    AXIS ORDER MEASURED, because it is easy to read backwards and a plot that transposes it
    draws garbage: `n_trajectories=7, n_bins=23` on 3-d data returns `(23, 7, 3)`. The middle
    axis is also CAPPED by the earliest time bin's population — paths start at real points, so
    `n_trajectories=100` over a bin holding 20 cells returns 20, not 100.

    Defensive by design, and it is a FIGURE's read, not a probe's: a plot that cannot draw arrows
    should still draw its scatter, so every way of not having trajectories collapses to `None`
    here rather than raising. (`probes.sample_trajectories` asks the same question and DOES raise,
    via `_ask` — a probe that silently returns nothing would put an empty readout in the g-vector,
    which is the one thing a readout may not do.)

    `.trajectories` is a plain optional property on MIOFlow, populated by `on_train_end` on every
    fit including `fast_dev_run`; it is absent on any other model and `None` when `on_train_end`
    produced nothing (an empty datamodule). Duck-typed, never `isinstance` — same rule as
    `probes._ask`, so this keeps working for the next algorithm that answers the same question.
    """
    traj = getattr(model, "trajectories", None)
    if traj is None:
        return None
    detach = getattr(traj, "detach", None)
    return traj if detach is None else detach().cpu().numpy()


def _run_mioflow_experiment(X, seed: int, fast_dev_run: bool, params: dict, labels=None,
                            device: str = "cpu"):
    """Run the REAL MIOFlow on an (n, d) array via manylatents.experiment.run_experiment.

    MIOFlow is a flow between timepoints, so each sample needs a time `label`. If real
    per-cell timepoints are provided (e.g. EB collection days), use them; otherwise compute
    a diffusion pseudotime on the (phate) embedding and discretize it into T timepoints.
    Built to match manylatents' own MIOFlow run_experiment test.

    `params` is the step's flat block; `_resolve_kwargs` decides where each name goes and
    refuses the ones it cannot place. Returns (embeddings, scores, extras, model) — the
    docstring said "(embeddings, scores)" for two return values after that stopped being
    true.

    TRAJECTORIES COME OFF `model`, not a fifth return value: `model.trajectories` is
    `(n_bins, n_trajectories, d)` or None. See the return statement for why that is enough."""
    import functools

    import numpy as np
    import torch
    from lightning import LightningDataModule, Trainer
    from torch.utils.data import DataLoader, Dataset

    from manylatents.algorithms.lightning.mioflow import MIOFlow
    from manylatents.algorithms.lightning.networks.mioflow_net import MIOFlowODEFunc
    from manylatents.experiment import run_experiment

    # FIRST, before any work: `_resolve_kwargs` raises on a param it cannot place, and a run
    # that is going to be refused for a typo should be refused before it spends a kNN graph and
    # a shortest-path pass computing the pseudotime it will never train against.
    net_kwargs, algo_kwargs = _resolve_kwargs(params, fast_dev_run)

    # `_dense` first — torch rejects negative strides AND cannot see a sparse matrix at all;
    # `np.ascontiguousarray(csr)` yields a `(1,)` object array silently. See `loading.dense`.
    from manyruns.pipeline.loading import dense as _dense_matrix

    Xn = np.ascontiguousarray(_dense_matrix(X), dtype=np.float32)
    n, d = Xn.shape
    from manyruns import vocab

    if labels is not None:
        # real timepoints → [0, 1] with REAL spacing preserved (day0/day3/day9 → 0, ⅓, 1, not
        # the even 0, ½, 1 that ranks would give). Magnitude lives in vocab.numeric_time so the
        # name-side (which column is time) and the value-side (what the clock reads) agree.
        # This is also where an upstream `discretize_time` step's bins arrive: they are already
        # discrete integer tokens, and `numeric_time` reads bare numbers as real magnitudes, so
        # the bin SPACING (not just the bin count) survives into MIOFlow's clock.
        time_labels = vocab.numeric_time(labels)
    else:
        # No real time axis AND no upstream ordering was threaded in (`runner._ml_lightning`
        # only reaches this branch when `state["labels"]` is empty) — fall back to computing
        # our OWN pseudotime on the embedding, same binning rule `discretize_time` uses.
        n_neighbors = int(params.get("n_neighbors") or min(15, max(2, n - 1)))
        pt = _diffusion_pseudotime(Xn, n_neighbors)
        n_timepoints = int(params.get("n_timepoints") or 5)
        bins = vocab.discretize_pseudotime(pt, n_timepoints)
        time_labels = (bins / max(n_timepoints - 1, 1)).astype(np.float32)

    x_t = torch.tensor(Xn)
    y_t = torch.tensor(time_labels)

    class _TimeDS(Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            return {"data": x_t[i], "label": y_t[i]}  # 'label' = per-sample timepoint

        def get_labels(self):
            return time_labels  # numeric timepoints (not the raw string labels)

    class _TimeDM(LightningDataModule):
        def setup(self, stage=None):
            self.train_dataset = _TimeDS()
            self.test_dataset = _TimeDS()

        def _loader(self):
            return DataLoader(_TimeDS(), batch_size=n)

        def train_dataloader(self):
            return self._loader()

        def val_dataloader(self):
            return self._loader()

        def test_dataloader(self):
            return self._loader()

    dm = _TimeDM()
    # THE WHOLE OF #55, in these two calls: they used to name their kwargs by hand, so 7 of
    # MIOFlow's 14 arguments were unreachable from a recipe and a step that declared one still
    # recorded `ok`. (Seven, not eight: 14 = 4 the run owns — network, optimizer, datamodule,
    # init_seed — plus 3 that were already settable — n_local_epochs, n_global_epochs,
    # sample_size — leaves 7. `_RESERVED` and `_accepts` are where that arithmetic lives.) The engine's signature decides now — see `_resolve_kwargs` above. `lr`
    # stays pinned: forwarding it means introspecting `torch.optim.Adam`, a third namespace and
    # a third-party signature far less stable than manylatents'.
    # The wrapper preserves a supplied network's weights. Seed them HERE, before they
    # exist, with the same recorded run seed used by the wrapper and run_experiment.
    net = MIOFlowODEFunc(input_dim=d, init_seed=int(seed), **net_kwargs)
    # `n_trajectories`/`n_bins` size the SAMPLE PATHS `MIOFlow.on_train_end` integrates for its
    # `.trajectories` property. They do NOT need naming here: they are ordinary MIOFlow kwargs
    # and `_resolve_kwargs` places them off the engine's signature like every other one, which
    # is exactly what #55 bought. MEASURED, because a hand-written block that reached these was
    # the alternative: `_resolve_kwargs({"n_trajectories": 7, "n_bins": 9})` puts both in
    # `algo_kwargs`, and the engine's own defaults for them are 100/100.
    model = MIOFlow(
        network=net,
        optimizer=functools.partial(torch.optim.Adam, lr=1e-3),
        init_seed=int(seed),
        **algo_kwargs,
    )
    model.datamodule = dm  # lets encode() infer the time span
    # Accelerator comes from the resolved device (manyruns.devices) — NOT a blind cpu pin.
    # accelerator_for() maps mps→cpu because torchdiffeq float64 + cdist backward is unsupported
    # on Apple MPS (manylatents' own tests pin cpu for the same reason); cuda is honored.
    from manyruns.devices import accelerator_for

    trainer = Trainer(
        accelerator=accelerator_for(device),
        devices=1,
        max_epochs=model.total_epochs,
        fast_dev_run=bool(fast_dev_run),
        # The engine's own trainer contract (manylatents/configs/trainer/default.yaml) sets
        # `deterministic: true`; Lightning defaults to False. Hand-rolling the Trainer here
        # silently dropped that, so two runs of the same recipe on the same data could
        # return different g-vectors — and a metric-pruning experiment built on those would
        # produce rankings that don't replicate. Track the engine, don't drift from it.
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    result = run_experiment(datamodule=dm, algorithm=model, trainer=trainer, seed=int(seed))
    # THE PREVIOUS FAILURE: "the trained flow -- the simulatable
    # object, the model -- is discarded at the return statement, and only its coordinates
    # survive". The model is still discarded; what stops being discarded here is everything
    # ELSE the engine computed. `manylatents.experiment` merges `algorithm.extra_outputs()`
    # into the top level of its result (experiment.py:443), which is where MIOFlow's
    # trajectories and a decoded gene-space head would arrive once it implements one — and
    # where `Cflows` already puts its GRN edges today.
    #
    # A THIRD AND NOW FOURTH RETURN VALUE rather than a tolerant unpack: three tests supply a
    # fake for this function, and a fake that silently keeps working while the real signature
    # moves is a test that has stopped testing the seam it was written for.
    extras = {k: v for k, v in result.items()
              if k not in ("embeddings", "scores", "label", "metadata") and v is not None}
    # `model` IS the fourth; returning it fixes the previous failure: "the trained flow — the
    # simulatable object, the model — is discarded at the
    # return statement, and only its coordinates are kept." It was discarded HERE, by this
    # line, not by anything upstream — `run_experiment` takes the algorithm as an argument and
    # this function owns it throughout. `probe` steps are what it is returned for.
    #
    # The OBJECT, not a checkpoint: `Trainer` runs with `enable_checkpointing=False` above, and
    # turning that on is a disk-cost decision separate from this one. In-process retention is
    # what `runner.state["model"]` needs and it costs nothing that was not already allocated.
    #
    # AND IT IS WHY THERE IS NO FIFTH VALUE FOR TRAJECTORIES. `model.trajectories` is populated
    # by `MIOFlow.on_train_end` — UNCONDITIONALLY, on every fit, `fast_dev_run` included — and
    # was simply never read, because `run_experiment` returns `embeddings`/`scores` only and the
    # model went out of scope right here. No manylatents change is needed to get the paths a plot
    # draws as arrows; the ODE integration was already happening. Returning the model gets them
    # and the growth rate and everything else `on_train_end` leaves behind, in one value, which
    # is the difference between retaining an object and enumerating its outputs one at a time.
    # (`.trajectories` is a plain optional property, so it is `None` when `on_train_end` produced
    # nothing — an empty datamodule. A caller reads it defensively; `probes._ask` is how.)
    return np.asarray(result.get("embeddings")), (result.get("scores") or {}), extras, model
