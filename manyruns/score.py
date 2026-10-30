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

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from manyruns.vocab import (
    BENCHMARK_CLASSES,
    LABELER_TOPOLOGY,
    SCSHAPEBENCH_TOPOLOGY,
    TOPOLOGIES,
    collapse_for_benchmark,
)

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


def unroutable_rungs(offer: Optional[Iterable[Any]] = None, *,
                     config_dir: Any = None) -> list[int]:
    """Rungs no recipe can claim. A perfect classifier is still capped by this.

    With `offer=None`, the CATALOGUE answer: the rungs no recipe in the catalog claims. A
    recipe file that closes a gap closes this too, with no edit here — which was the whole
    point of moving the claim into the YAML. `[]` on the bundled catalog since `branches`
    claims `multi-branching`.

    With an `offer`, the OFFER answer: the rungs no offered and runnable recipe claims. The
    catalogue and the offer come apart exactly where a recipe is claimed but not offered (a
    `blocked:` recipe, or one legal on no data). An entry is a recipe name (`str`) or a mapping
    with key `recipe` and optional key `can_run` — `narrate.offer` output,
    `LedgerRow.as_offer()` output, or a decision row's `offered` list, with no conversion. A
    mapping whose `can_run` is false does not count: a hollow row is not an offer anyone can
    take. A name that is not in the catalog, or that claims nothing, contributes no rung.

    `config_dir` is keyword-only, so a directory passed by position cannot be read as an offer.
    """
    claimed = claimed_rungs(config_dir)
    if offer is None:
        claimable = set(claimed.values())
    else:
        names: set = set()
        for entry in offer:
            if isinstance(entry, str):
                names.add(entry)
            elif isinstance(entry, Mapping):
                if entry.get("can_run", True):
                    names.add(entry.get("recipe"))
            else:
                raise TypeError(f"an offer entry is a recipe name or a mapping, not {entry!r}")
        claimable = {claimed[n] for n in names if n in claimed}
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


def check_raw_labels() -> list[str]:
    """Raw benchmark and labeler labels that do not resolve to one rung; empty means consistent.

    The style of `check_ladder`. Reads both maps at call time, so a drifted entry is reported
    the moment it lands rather than when a scorer first meets it.
    """
    bad = [f"scShapeBench label {k!r} maps to {v!r}, which has no rung"
           for k, v in SCSHAPEBENCH_TOPOLOGY.items() if v not in RUNG]
    bad += [f"labeler label {k!r} maps to {v!r}, which has no rung"
            for k, v in LABELER_TOPOLOGY.items() if v is not None and v not in RUNG]
    for t in TOPOLOGIES:
        try:
            collapse_for_benchmark(t)
        except (ValueError, KeyError):
            bad.append(f"topology {t!r} has no benchmark class")
    return bad


# ── verdict rows: a structure-class prediction about one run ────────────────
#
# A verdict row is a row of `outputs/index.jsonl` written by `manyruns claim`
# (`manyruns.verdict`). It reuses its run's `run_id` and carries the prediction in
# `row["extra"]["verdict"]`: six independent probabilities, one per `TOPOLOGIES` name.


def verdict_of(row: Mapping[str, Any]) -> Optional[dict]:
    """`row["extra"]["verdict"]` when it is a mapping, else `None`. The test for a verdict row."""
    extra = row.get("extra") if isinstance(row, Mapping) else None
    if not isinstance(extra, Mapping):
        return None
    verdict = extra.get("verdict")
    return verdict if isinstance(verdict, Mapping) else None


def pred_rung(row: Mapping[str, Any]) -> int:
    """The rung of a verdict's most probable topology, so `ladder_cost` reads a verdict row.

    A tie goes to the LOWER rung, then to the earlier name in `TOPOLOGIES`: overshoot costs
    twice undershoot, so the tie takes the cheaper error. All zero is rung 0, as no offer is in
    `predicted_rung`. A row that is not a verdict raises `ValueError`.
    """
    verdict = verdict_of(row)
    if verdict is None:
        raise ValueError("not a verdict row: row['extra']['verdict'] is missing")
    probs = verdict.get("topology_probs") or {}
    named = [(float(probs.get(t, 0.0)), t) for t in TOPOLOGIES if t in RUNG]
    if not named or max(p for p, _ in named) <= 0.0:
        return 0
    _, best = min(named, key=lambda pt: (-pt[0], RUNG[pt[1]], TOPOLOGIES.index(pt[1])))
    return RUNG[best]


def truth_from_annotations(annotation_rows: Iterable[Mapping[str, Any]],
                           run_rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Truth rows `{dataset, annotator, topology}` from `kind: "label"` annotation rows.

    The ONE reader of the annotation row shape (`manyruns.annotations`), so a change to that
    shape changes this function and its test only. Reads `payload["topology"]`, which already
    holds `TOPOLOGIES` names; the raw benchmark labels in `payload["raw"]` are not read here.
    A label on a run with no `payload["dataset"]` takes the `dataset_name` of the last run row
    with that `run_id`. A row with no dataset or no known topology is skipped.
    """
    by_run: dict[Any, Any] = {}
    for run in run_rows:
        if isinstance(run, Mapping) and run.get("run_id") is not None:
            by_run[run["run_id"]] = run.get("dataset_name")
    out: list[dict] = []
    for row in annotation_rows:
        if not isinstance(row, Mapping) or row.get("kind") != "label":
            continue
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
        dataset = payload.get("dataset")
        if not (isinstance(dataset, str) and dataset):
            dataset = by_run.get(row.get("run_id"))
        topology = [t for t in (payload.get("topology") or []) if t in TOPOLOGIES]
        if not (isinstance(dataset, str) and dataset) or not topology:
            continue
        out.append({"dataset": dataset, "annotator": row.get("chooser"), "topology": topology})
    return out


def soft_truth(truth_rows: Iterable[Mapping[str, Any]], *,
               collapse: Callable[[str], str] = collapse_for_benchmark) -> list[dict]:
    """Per dataset, the fraction of its annotators whose collapsed label set holds each class.

    One dict per dataset, sorted by dataset: `{dataset, n_annotators, soft}`, with `soft` keyed
    by `BENCHMARK_CLASSES` in that order. An annotator with several rows for one dataset counts
    once, by its LAST row.
    """
    latest: dict[str, dict[Any, set]] = {}
    for row in truth_rows:
        classes = {collapse(t) for t in row.get("topology") or []}
        latest.setdefault(row["dataset"], {})[row.get("annotator")] = classes
    out = []
    for dataset in sorted(latest):
        votes = latest[dataset]
        n = len(votes)
        soft = {c: sum(1 for s in votes.values() if c in s) / n for c in BENCHMARK_CLASSES}
        out.append({"dataset": dataset, "n_annotators": n, "soft": soft})
    return out


def _verdict_key(row: Mapping[str, Any]) -> Any:
    name = row.get("dataset_name")
    return row.get("dataset") if name is None else name


def _collapsed(verdict: Mapping[str, Any], collapse: Callable[[str], str]) -> dict[str, float]:
    """Each benchmark class's score: the MAX of its source probabilities (stays in [0, 1])."""
    probs = verdict.get("topology_probs") or {}
    out = {c: 0.0 for c in BENCHMARK_CLASSES}
    for t in TOPOLOGIES:
        c = collapse(t)
        if c in out:
            out[c] = max(out[c], float(probs.get(t, 0.0)))
    return out


def _matched(verdict_rows: Iterable[Mapping[str, Any]], soft_by: Mapping[Any, Mapping],
             collapse: Callable[[str], str]) -> tuple[list[tuple[dict, Mapping]], int]:
    """`([(collapsed scores, soft truth)], unmatched)` over the verdict rows that are verdicts."""
    scored, unmatched = [], 0
    for row in verdict_rows:
        verdict = verdict_of(row)
        if verdict is None:
            continue
        soft = soft_by.get(_verdict_key(row))
        if soft is None:
            unmatched += 1
            continue
        scored.append((_collapsed(verdict, collapse), soft))
    return scored, unmatched


def _average_precision(samples: Sequence[tuple[float, bool]]) -> Optional[float]:
    """AP = sum over distinct thresholds, high to low, of (R_n - R_(n-1)) * P_n. Equal scores
    enter at one threshold, as `sklearn.metrics.average_precision_score` does. `None` when
    there is no positive sample."""
    total = sum(1 for _, y in samples if y)
    if not total:
        return None
    ap, tp, seen, prev_recall = 0.0, 0, 0, 0.0
    for threshold in sorted({s for s, _ in samples}, reverse=True):
        at = [y for s, y in samples if s == threshold]
        seen += len(at)
        tp += sum(1 for y in at if y)
        recall = tp / total
        ap += (recall - prev_recall) * (tp / seen)
        prev_recall = recall
    return ap


def class_auprc(verdict_rows: Iterable[Mapping[str, Any]],
                truth_rows: Iterable[Mapping[str, Any]], *,
                collapse: Callable[[str], str] = collapse_for_benchmark) -> dict:
    """Macro and per-class AUPRC (average precision) of verdicts, pooled over datasets.

    Binary truth is `soft >= 0.5`, a majority of annotators. Each verdict is one sample; two
    verdicts on one dataset are two samples, so the caller filters by `chooser` or `model`
    first. A verdict whose dataset has no truth row is counted in `unmatched`, not scored. A
    class with no positive sample is `None` and the macro mean skips it.
    """
    soft_by = {s["dataset"]: s["soft"] for s in soft_truth(truth_rows, collapse=collapse)}
    scored, unmatched = _matched(verdict_rows, soft_by, collapse)
    per_class: dict[str, Optional[float]] = {}
    positives: dict[str, int] = {}
    for c in BENCHMARK_CLASSES:
        samples = [(scores[c], soft[c] >= 0.5) for scores, soft in scored]
        positives[c] = sum(1 for _, y in samples if y)
        per_class[c] = _average_precision(samples)
    defined = [v for v in per_class.values() if v is not None]
    macro = sum(defined) / len(defined) if defined else None
    return {"macro": macro, "per_class": per_class, "positives": positives,
            "n": len(scored), "unmatched": unmatched}


def brier(verdict_rows: Iterable[Mapping[str, Any]],
          soft_rows: Iterable[Mapping[str, Any]]) -> dict:
    """Mean squared error between collapsed verdict probabilities and soft truth.

    `soft_rows` is the output of `soft_truth`. Verdict selection, dataset keys and `unmatched`
    follow `class_auprc`. `brier` is the mean of the four per-class values; with no scored
    verdict, every value is `None`.
    """
    soft_by = {s["dataset"]: s["soft"] for s in soft_rows}
    scored, unmatched = _matched(verdict_rows, soft_by, collapse_for_benchmark)
    n = len(scored)
    if not n:
        return {"brier": None, "per_class": {c: None for c in BENCHMARK_CLASSES},
                "n": 0, "unmatched": unmatched}
    per_class = {c: sum((scores[c] - float(soft[c])) ** 2 for scores, soft in scored) / n
                 for c in BENCHMARK_CLASSES}
    return {"brier": sum(per_class.values()) / len(per_class), "per_class": per_class,
            "n": n, "unmatched": unmatched}
