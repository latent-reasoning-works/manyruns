"""The `probe` group — questions put to a trained model.

The trained flow, *"the simulatable object, the model"*, was discarded at
`_run_mioflow_experiment`'s return statement and only its coordinates kept. It was discarded by
MANYRUNS's line, not by anything upstream — `manylatents.experiment.run_experiment` takes the
algorithm as an argument and the caller owns it throughout. This module is what the model is
kept FOR.

**A probe is a step, not a new axis.** It consumes a fact another step produced (`model`), it
writes to the g-vector, and it transforms no data and moves no selection. `CLAUDE.md`'s
extension path for a new capability is a group plus one executor in the dispatch table, and
that is exactly what this is — not a fourth axis beside capability / substrate / fidelity, and
not a new `--engine` value.

**Duck-typed against the engine, on purpose.** A probe asks the model for a method by name and
declines when it has none. It must NOT import `MIOFlow` or test `isinstance`: manyruns does
not implement the compute and must not hold an opinion about the class, and a probe written
against one algorithm's type is a product-side copy of the engine's catalogue by another route
(`vocab.py` refuses that shape twice). What a probe knows is the QUESTION; the engine knows how
to answer it.

**What is answerable today, measured against the installed manylatents.** `MIOFlow` exposes
`trajectories` (a property, populated by `_generate_trajectories` at train end) and `encode`
(integrate an arbitrary tensor from t=0 to t=1). It does NOT expose `decode_to_gene_space` or
`growth_rate` — those are upstream work, and `decode_to_gene_space` in particular cannot arrive
through `extra_outputs()`, which takes no arguments and so cannot accept a gene set chosen
after the run (`Cflows` fixes its own at construction time via `grn_gene_names`). So this module
ships the probe that needs nothing upstream, and the others land when their methods do.
"""
from __future__ import annotations

from typing import Any


class _NoAnswer(Exception):
    """The model cannot answer this question — raised here, translated to a declined step by
    `runner._run_probe_step`, so this module never imports the step loop's vocabulary."""


def _ask(model: Any, *names: str) -> Any:
    """The first attribute the model actually has, by name. Never `isinstance`.

    Several names because the same question has different spellings across algorithms and
    versions, and a probe that hard-codes one spelling breaks on the next model that answers
    the same question. Reports every name it tried, so the failure names the contract rather
    than the attribute.
    """
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    raise _NoAnswer(
        f"this model answers none of {list(names)} — it exposes "
        f"{sorted(n for n in dir(model) if not n.startswith('_'))[:12]}…")


def _probe_sample_trajectories(model: Any, params: dict) -> dict:
    """The paths the trained flow integrates — `(T, n, d)`: T timesteps, n cells, d dims.

    Reported as SHAPE plus a displacement summary, never as the array. A probe's answer lands
    in the g-vector, which `store.py` writes to `index.jsonl` one line per run; putting an
    `(n_bins, n_cells, dim)` tensor there would put the scientist's coordinates in a shared
    corpus, which `the-decision-record.md`'s federated line rules out — *"No data, no
    coordinates, no counts."* The array stays in memory for a caller that wants it.
    """
    import numpy as np

    traj = np.asarray(_ask(model, "trajectories"))
    if traj.ndim != 3:
        raise _NoAnswer(f"trajectories has shape {traj.shape}; expected (steps, cells, dims)")
    # AXIS 0 IS TIME, axis 1 is cells, and this is MEASURED rather than inferred.
    # `MIOFlow._generate_trajectories` builds `t_bins = linspace(min(times), max(times),
    # self.n_bins)` and `X_0_sample = X_0[idx]` where `idx` selects `n_trajectories` cells, then
    # `odeint(self.network, X_0_sample, t_bins)` — and torchdiffeq returns `(len(t), *y0.shape)`.
    # Constructed directly with `n_bins=7, n_trajectories=25` to check, because BOTH default to
    # 100 and a run on the bundled defaults cannot tell the two axes apart. Naming them the
    # other way round reads perfectly and reports the bin count as a population size.
    n_steps, n_trajectories, dims = (int(x) for x in traj.shape)
    # Net displacement per cell, start to end: one number per cell reduced to two, so the
    # record says whether the flow MOVED anything without carrying where anything went.
    travel = np.linalg.norm(traj[-1] - traj[0], axis=-1)
    return {
        "n_steps": n_steps,
        "n_trajectories": n_trajectories,
        "dims": dims,
        "median_displacement": float(np.median(travel)),
        "max_displacement": float(np.max(travel)),
    }


#: `name -> probe`, the table `runner._NAME_TABLES` reads so `dispatchable` can answer "could
#: this engine run this step" without running it. A plain dict, so a test can substitute one.
_PROBE_STEPS = {
    "sample_trajectories": _probe_sample_trajectories,
}
