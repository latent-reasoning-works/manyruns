"""The empirical null — how big is this readout when the association is destroyed?

Every analysis readout in this product is an *extreme-value* statistic: a maximum over k
clusters and condition pairs (`composition`), or a score over a partition someone chose
(`separation`). The null expectation of such a statistic is positive by construction and
moves with n and k, so a bare magnitude is uninterpretable on its own. Measured: `composition`'s
null mean is 0.578 at n=600/k=8 and **5.589** at n=60/k=8 —
and at n=60 the real value (5.102) lands *below* its own null. The product rendered that as
*"one population is ~48× more abundant"*.

So the rule this module exists to enforce, from the audit: **report an empirical rank over
R ≥ 99 permutations, never a difference of two numbers.** A difference is inflatable by
shrinking the cohort and raising k, which is perfectly seed-stable and so passes a
reproducibility check while being meaningless.

What is permuted is the *association*, not the data. `permute_labels` shuffles the condition
assignment and recomputes; the embedding and the clustering are left exactly as they were.
That is the right null here precisely because neither step ever sees a label — clustering is
unsupervised, so re-using it across permutations is not double-dipping, it is holding the
one thing the null is not about fixed.

**The limit of a label-permutation null, stated once.** It can only test a statistic that
READS labels. A geometry-derived statistic — one computed from the embedding alone — is
bit-identical under label permutation, so its margin is zero by construction rather than by
absence of signal. The deleted `granger` is the worked example: that
step never read a label, so this harness would have scored it a perfect null while it
returned p < 1e-30 on pure noise. Such a statistic needs a *data-level* null propagated
through the whole pipeline (synthesise null data, re-run the embedding AND the ordering AND
the statistic), which is a different and much more expensive object. `NULL_KIND` in
`vocab.py` is the closed list of which readout gets which null, and it refuses to guess.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

#: Permutation count. The empirical p has a floor of 1/(R+1), so R=99 buys a floor of 0.01 —
#: enough to clear 0.05 with headroom, and cheap: the clustering is NOT recomputed, so each
#: rep is a bincount over labels.
DEFAULT_REPS = 199


def rank_against_null(
    statistic: Callable[[Any], Optional[float]],
    labels: Any,
    *,
    reps: int = DEFAULT_REPS,
    seed: int = 0,
) -> Optional[dict]:
    """Score `statistic` against `reps` label permutations. Returns the rank, not a margin.

    `statistic` takes a label vector and returns a float (or None if it cannot be computed,
    which aborts — a null for a statistic that did not run is meaningless).

    The returned `p_emp` is the standard (1 + #{null ≥ real}) / (R + 1): the +1 counts the
    observed value as one of its own permutations, which is what keeps the estimate from
    reaching an impossible p = 0.
    """
    import numpy as np

    real = statistic(labels)
    if real is None:
        return None

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    draws: list[float] = []
    for _ in range(reps):
        val = statistic(rng.permutation(labels))
        if val is not None:
            draws.append(float(val))
    if not draws:
        return None

    null = np.asarray(draws, dtype=float)
    n_ge = int((null >= float(real)).sum())
    return {
        "real": float(real),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)) if len(null) > 1 else 0.0,
        "null_p95": float(np.percentile(null, 95)),
        "reps": len(draws),
        "rank": len(draws) - n_ge,          # how many null draws the real value beats
        "p_emp": (1.0 + n_ge) / (len(draws) + 1.0),
    }


def record(g: dict, key: str, result: Optional[dict]) -> None:
    """Write a null result into the g-vector under `<key>__null_*`.

    Flat scalar keys on purpose: the g-vector is `dict[str, float]` and reaches the results
    table by that contract, so a nested dict here would be dropped silently by
    `experiment.metric_columns`. `__null_p` is the one the narration gates on.
    """
    if result is None:
        return
    g[f"{key}__null_p"] = round(float(result["p_emp"]), 4)
    g[f"{key}__null_mean"] = round(float(result["null_mean"]), 4)
    g[f"{key}__null_reps"] = int(result["reps"])
