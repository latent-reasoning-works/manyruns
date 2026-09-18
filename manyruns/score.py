"""The complexity ladder — scoring whether an analysis choice claimed the right amount.

Pure functions over plain dicts and lists. At module level it imports `vocab` and nothing
else: not `modes`, `pipeline`, `serving` or `app`. There are already several step loops in
this repo and a scorer that could reach the runner would become another one.

`catalog` is imported LAZILY, inside the two functions that need it, and it is the one
exception. The rule above is about reach, not about the import graph: `catalog` reads YAML and
imports `vocab`, `os`, `pathlib` and (lazily) `omegaconf` — there is no path from it to a
runner. The reason it is needed at all is that a recipe's rung is a per-recipe FACT, and the
table that used to hold it here was a second home for it (see `claimed_rungs`).

The idea: analyses form a ladder ordered by how much structure they
ASSUME. The right analysis is the highest rung the data actually supports. Under-shooting
loses signal; over-shooting *invents* it — a trajectory drawn through noise — so the two
errors are not symmetric and the score must not treat them as such.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from manyruns.vocab import TOPOLOGIES

#: topology → rung. The ordering is by strength of the structural claim, not by "difficulty".
#:
#: `surface` sits at 0 on purpose: for a smooth 2-manifold with no groups, no branch points
#: and no extremal vertices, the honest analysis IS just an embedding. That makes rung 0 a
#: correct answer sometimes rather than a universal floor, which is what stops "always predict
#: embed" from being a free win.
#:
#: `cycle` and `single-trajectory` share rung 2 — both claim a one-dimensional continuum; a
#: cycle merely closes it. They are not distinguished here because no recipe distinguishes
#: them either.
RUNG: dict[str, int] = {
    "surface": 0,
    "clusters": 1,
    "cycle": 2,
    "single-trajectory": 2,
    "multi-branching": 3,
    "archetypal": 4,
}

#: Rungs that are NOT comparable: archetypal assumes a simplex with extremal vertices, which
#: is a *different* claim from a branching tree rather than a strictly stronger one. Counting
#: 3↔4 as a distance of 1 would quietly reward a model for confusing them.
INCOMPARABLE: frozenset = frozenset({frozenset({3, 4})})


def recipe_rung(recipe: Mapping[str, Any]) -> Optional[int]:
    """The rung a recipe CLAIMS, read from its own `claims:`. None when it declares none.

    This replaced a hardcoded `RECIPE_RUNG = {"embed": 0, "contrast": 1, "cflows": 2}`. The
    dict was a SECOND HOME for a per-recipe fact and it drifted by construction: adding a
    recipe is adding a YAML file (see `catalog`'s header), the file reached the menu with no
    code change (`narrate.offer` derives from the catalog), and `predicted_rung` then scored
    it 0 — silently, because a name absent from the dict was indistinguishable from a name
    claiming rung 0. Ten new recipes would have drifted it ten ways.

    `claims` and `suits` are different axes and both are needed: `suits` is which data shapes
    the recipe is ADVISABLE on (it drives the ★ recommendation), `claims` is what structure it
    ASSERTS (it is what the ladder grades). Neither belongs in a Python dict beside the YAML.
    """
    if not isinstance(recipe, Mapping):
        return None
    return RUNG.get(recipe.get("claims"))


def claimed_rungs(config_dir: Any = None) -> dict[str, int]:
    """`recipe name -> claimed rung`, read from the catalog — the table that was a literal.

    A recipe declaring no `claims:` is absent rather than 0: "asserts nothing" and "asserts a
    surface" are different, and collapsing them is the defect this function exists to remove.

    Never raises on a bad file, for the same reason `catalog.known_steps` does not — a broken
    recipe in `$MANYRUNS_RECIPE_DIR` drops out of the scoring table instead of taking the
    scorer down. Not cached: the directory IS the registry, and a cache would mean a recipe
    added mid-session scores by a table that predates it, which is the drift again wearing a
    different hat.
    """
    from manyruns import catalog  # lazy: see the module docstring

    out: dict[str, int] = {}
    for name in catalog.discover_recipes(config_dir):
        try:
            cfg = catalog.load_recipe(name, config_dir)
        except (ValueError, OSError):
            continue
        rung = recipe_rung(cfg)
        if rung is not None:
            out[name] = rung
    return out


def truth_rung(topology: Optional[Sequence[str]]) -> Optional[int]:
    """The rung a dataset's ground-truth topology supports — the MAX over its labels.

    Max, not the set: the benchmark is multi-label because a dataset can exhibit several
    regimes at once, and the strongest supported claim is the one the ladder asks about.
    Returns None for an unlabelled dataset so it can be excluded rather than scored as 0.
    """
    if not topology:
        return None
    rungs = [RUNG[t] for t in topology if t in RUNG]
    return max(rungs) if rungs else None


def predicted_rung(recipes: Iterable[str], config_dir: Any = None) -> int:
    """The highest rung a set of offered recipes claims. No offer at all is rung 0.

    Unchanged in meaning and in every measured value on the bundled catalog — `embed` 0,
    `contrast` 1, `cflows` 2 — but now derived from those recipes' own `claims:` rather than
    from a table here. A recipe that claims nothing still contributes nothing, which is what
    keeps "no offer at all" at 0.
    """
    claimed = claimed_rungs(config_dir)
    rungs = [claimed[r] for r in recipes if r in claimed]
    return max(rungs) if rungs else 0


def incomparable(a: int, b: int) -> bool:
    return frozenset({a, b}) in INCOMPARABLE


def ladder_cost(truth: int, pred: int, *, overshoot_weight: float = 2.0) -> float:
    """Asymmetric cost of claiming rung `pred` when the data supports `truth`.

    Under-shooting costs one per rung: you left signal on the table. Over-shooting costs
    `overshoot_weight` per rung, because the analysis asserts structure the data cannot
    support — which is the failure `narrate.refusal` exists to prevent and exactly what
    the deleted `granger` step did on structureless input.

    Incomparable pairs (§`INCOMPARABLE`) cost a flat 1.0: a real error, but not a distance.
    """
    if truth == pred:
        return 0.0
    if incomparable(truth, pred):
        return 1.0
    gap = pred - truth
    return gap * overshoot_weight if gap > 0 else -gap


def unroutable_rungs(config_dir: Any = None) -> list[int]:
    """Rungs no recipe can claim. A perfect classifier is still capped by this.

    `[3, 4]` on the bundled catalog — nothing claims `multi-branching` or `archetypal`. That
    gap is a finding, not an oversight, and it is reported here rather than left to hide in
    the score. Now that it reads the catalog, a recipe file that closes the gap closes this
    too, with no edit here — which was the whole point of moving the claim into the YAML.
    """
    claimable = set(claimed_rungs(config_dir).values())
    return sorted({r for r in RUNG.values() if r not in claimable})


def score_offer(truth: Sequence[str], offered_topologies: Sequence[str]) -> dict:
    """Multi-label precision/recall/F1 of an offered topology SET against ground truth.

    Set-valued because `narrate.offer` is set-valued and because the benchmark aggregates
    nine annotators by union — the product's job is to offer what an expert would consider,
    not to pick one and hide the rest.
    """
    t, o = set(truth), set(offered_topologies)
    tp = len(t & o)
    precision = tp / len(o) if o else 0.0
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "precision": precision, "recall": recall, "f1": f1,
            "missed": sorted(t - o), "spurious": sorted(o - t)}


def summarize(records: Sequence[Mapping[str, Any]], *, overshoot_weight: float = 2.0) -> dict:
    """Aggregate per-dataset records into the ladder readout.

    A record needs `truth_rung` and `pred_rung`. Reports mean cost, exact-match rate, and —
    kept apart on purpose — how much of the error is under- versus over-shoot, because one
    mean hides the asymmetry the whole ladder exists to express.
    """
    scored = [r for r in records if r.get("truth_rung") is not None]
    if not scored:
        return {"n": 0}
    costs, under, over, exact, incomp = [], 0, 0, 0, 0
    for r in scored:
        t, p = int(r["truth_rung"]), int(r["pred_rung"])
        costs.append(ladder_cost(t, p, overshoot_weight=overshoot_weight))
        if t == p:
            exact += 1
        elif incomparable(t, p):
            incomp += 1
        elif p < t:
            under += 1
        else:
            over += 1
    n = len(scored)
    return {"n": n, "mean_cost": sum(costs) / n, "exact": exact, "exact_rate": exact / n,
            "under": under, "over": over, "incomparable": incomp,
            "unroutable_rungs": unroutable_rungs()}


def check_ladder() -> list[str]:
    """Problems with the ladder itself; empty means consistent.

    Guards the drift `vocab.py` exists to prevent: a topology added to `TOPOLOGIES` with no
    rung would be silently unscoreable.
    """
    bad = [f"topology {t!r} has no rung" for t in TOPOLOGIES if t not in RUNG]
    bad += [f"rung table has {t!r}, which is not in vocab.TOPOLOGIES"
            for t in RUNG if t not in TOPOLOGIES]
    return bad
