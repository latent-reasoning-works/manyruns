"""The `probe` group — questions put to what a model step returned.

**A probe is a step, not a new axis.** It consumes a fact another step produced, it writes to
the g-vector, and it transforms no data and moves no selection. `CLAUDE.md`'s extension path
for a new capability is a group plus one executor in the dispatch table, and that is exactly
what this is — not a fourth axis beside capability / substrate / fidelity, and not a new
`--engine` value.

**Most probes read a state slot, not a model.** manyruns never imports MIOFlow: a `mioflow` step
runs only as a tool step, and the fitted flow stays in the tool's interpreter. What crosses the
model seam is `trajectories`, which fills `state["trajectories"]`. `PROBE_READS` names the slot
each such probe reads; `runner._run_probe_step` passes that slot, and declines when it is empty.
A probe absent from `PROBE_READS` reads `state["model"]` (the declared stubs, `growth_rate` and
`decode_to_gene_space`, which still need a fitted model).

**Duck-typed against a model, on purpose.** A model probe asks for a method by name (`_ask`) and
declines when it has none. It must NOT import a model class or test `isinstance`: manyruns does
not implement the compute and must not hold an opinion about the class.
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


def _probe_sample_trajectories(traj: Any, params: dict) -> dict:
    """The paths a model step returned, `(T, n, d)`: T time bins, n trajectories, d dims.

    `traj` is `state["trajectories"]` (`PROBE_READS`), the array the model seam checked and
    merged. Reported as SHAPE plus a displacement summary, never as the array. A probe's answer
    lands in the g-vector, which `store.py` writes to `index.jsonl` one line per run; putting the
    array there would put the scientist's coordinates in a shared corpus, which the federated
    line of the decision record rules out — *"No data, no coordinates, no counts."*
    """
    import numpy as np

    traj = np.asarray(traj)
    if traj.ndim != 3:
        raise _NoAnswer(f"trajectories has shape {traj.shape}; expected (steps, cells, dims)")
    # AXIS 0 IS TIME, axis 1 is trajectories: the seam's `trajectories` fact is
    # (bins, trajectories, coordinates), which `external.check_result` checks before the merge.
    # Naming them the other way round reads perfectly and reports the bin count as a population.
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


#: `probe name -> the state slot it reads`. A probe listed here takes that slot's value instead
#: of `state["model"]`; `runner._run_probe_step` declines when the slot is empty.
PROBE_READS: dict[str, str] = {"sample_trajectories": "trajectories"}


#: `name -> probe`, the table `runner._NAME_TABLES` reads so `dispatchable` can answer "could
#: this engine run this step" without running it. A plain dict, so a test can substitute one.
_PROBE_STEPS = {
    "sample_trajectories": _probe_sample_trajectories,
}
