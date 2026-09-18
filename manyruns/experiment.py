"""The metric-separation experiment: run every recipe over every dataset, then ask which
metrics actually separate them.

The deliverable is a `(recipe × dataset) → metric-vector` table plus a feature-selection
call — that is the whole thing. The pruned metric set IS the g-vector definition, learned
from data rather than hand-picked.

Deliberately thin. It sits ABOVE the seam that already exists
(`modes.run("infer", …) → g_vector`) and adds no second runner, no result store, and no
matcher. A k-NN reference library answers "which known workflow does this resemble"; the
question here is the inverse — "which metrics tell the workflows apart" — which is a
`groupby` and an F-test, not an abstraction.

Parallelism is PROCESSES, not threads, and that is not a preference: `pipeline` suppresses
warnings via `warnings.catch_warnings`/`simplefilter` (process-global, not thread-safe) and
matplotlib carries global figure/backend state. A cross-product of independent runs is
embarrassingly parallel with no shared mutable state, so one run per process is both the
robust and the simple choice. Workers pin their BLAS thread count so N processes each
spawning a full BLAS pool don't thrash.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


#: Columns the table carries besides the metrics themselves. `protocol` and `shape` are the
#: two labels the ranking scores against — "which metrics tell workflows apart" and "which
#: tell data shapes apart" — both answerable from the one table.
#: `complete` rides alongside `ok` because they answer different questions and a sweep needs
#: both: `ok` is "can I trust this row" (nothing errored) and `complete` is "did every declared
#: step run". A row that is `ok` but not `complete` is real data from a shorter analysis — fine
#: to aggregate, not always fine to compare against a longer one. See manyruns#66.
LABEL_COLUMNS = ("recipe", "dataset", "shape", "seed", "ok", "complete")


@dataclass(frozen=True)
class Run:
    """One cell of the cross product. Recipes and datasets are the plain dicts the catalog
    loads — no wrapper type, so nothing is converted on the way to the runner."""

    recipe: dict
    dataset: dict
    seed: int = 42

    @property
    def label(self) -> str:
        return f"{self.recipe['name']}__{self.dataset['name']}__s{self.seed}"


@dataclass(frozen=True)
class Result:
    run: Run
    g_vector: dict = field(default_factory=dict)
    #: Nothing errored — the sweep's failed-cell flag.
    ok: bool = False
    #: Every declared step ran. Defaults False for the SAME reason `ok` does: a cell that
    #: raised before reaching the runner produced no record, and claiming completeness for a
    #: run that never happened is the one direction this must not fail in.
    complete: bool = False
    status: dict = field(default_factory=dict)
    error: Optional[str] = None
    #: The positional step record from `pipeline.runner` — what actually executed, with
    #: params and durations. Previously `run_one` read `g_vector`/`ok`/`status` and let
    #: `res["trace"]` fall on the floor, which made step agreement and cost unscoreable
    #: from the sweep table no matter what the runner recorded.
    steps: list = field(default_factory=list)


def plan(
    recipes: Sequence[dict], datasets: Sequence[dict], seeds: Sequence[int] = (42,),
    prune: bool = True,
) -> list[Run]:
    """The cross product, minus the cells that cannot produce a result.

    A plain loop — what a Hydra multirun would buy us, without a process-global config
    singleton in the way. `prune` drops (recipe, dataset) pairs `vocab.unmet` refuses: a
    `contrast` recipe on data with no conditions runs, emits `None` + a note, and lands in the
    table as a MISSING metric — indistinguishable from a recipe that simply doesn't emit it.
    Pruning removes that conflation, and costs nothing because the cell had no result to give.
    `skipped_cells` says what was dropped, so it is never silent. Pass `prune=False` to plan
    the raw product.

    NOT ONLY A DATA QUESTION any more, which is why this reads `unmet` rather than describing
    it: since `prep` landed, a recipe can also be refused for its own step ORDER (a filter
    after the embedding it consumes), which is true of every dataset in the sweep at once. That
    cell is equally unable to produce a result and equally worth pruning; what changes is what
    `skipped_cells` has to be read as saying — see its own note."""
    from manyruns.vocab import unmet

    cells = product(recipes, datasets, seeds)
    if not prune:
        return [Run(r, d, s) for r, d, s in cells]
    return [Run(r, d, s) for r, d, s in cells
            if not unmet(r, d.get("shape", "unknown"), handle=d.get("handle"))]


def skipped_cells(recipes: Sequence[dict], datasets: Sequence[dict]) -> list[tuple]:
    """`(recipe, dataset, what's missing)` for every pair `plan` would prune.

    Report this. A pruned sweep that does not say what it pruned looks like a smaller
    experiment rather than a filtered one.

    The fact NAMES only, never a reason — a row here says which fact was unavailable at the
    point some step needed it, and it does not say whether the dataset lacks it or a narrowing
    took it away. Both are real ("no dataset here carries conditions" and "this recipe filters
    after it embeds"), and the second is a property of the recipe alone, so it repeats on every
    dataset row. `vocab.invalidated` is what tells the two apart; a reader chasing a recipe
    that appears against EVERY dataset with the same fact should ask it, or run
    `manyruns check`, which does.
    """
    from manyruns.vocab import unmet

    out = []
    for r, d in product(recipes, datasets):
        # The SAME arguments `plan` prunes on. A report that reads the pair differently from
        # the filter it explains is a row saying a cell was dropped for a fact the filter
        # accepted — the two-vocabularies drift `vocab.noncanonical`'s docstring records.
        missing = unmet(r, d.get("shape", "unknown"), handle=d.get("handle"))
        if missing:
            out.append((r["name"], d["name"], sorted(missing)))
    return out


def _request(run: Run, *, suite: Optional[Sequence[str]], engine: str, out_dir: Path) -> dict:
    """The `modes.run("infer", …)` request for one cell. The recipe goes through untouched —
    it is already the dict the runner consumes."""
    from manyruns import catalog

    handle = run.dataset.get("handle") or {}
    req: dict[str, Any] = {
        "modality": run.dataset.get("modality", "scrna"),
        "recipe": run.recipe,
        "engine": engine,
        "seed": run.seed,
        "out_dir": str(out_dir / run.label),
        "fast_dev_run": False,
    }
    # Inline rows may reuse a catalog name for a different source. Attach catalog
    # identity only when both the handle and generator settings match its declaration.
    if run.dataset.get("name") in catalog.discover_datasets():
        declared = catalog.load_dataset(run.dataset["name"])
        if (handle == declared.get("handle")
                and (run.dataset.get("params") or {}) == (declared.get("params") or {})):
            req["dataset_name"] = run.dataset["name"]
    if handle.get("kind") == "manylatents":
        req["dataset"] = handle.get("ref")   # a named engine dataset; the engine loads it
    else:
        req["data"] = Path(handle.get("ref", ""))   # a path the product loads
    # Generator kwargs (`n_branch`, `concentration`, `cluster_std`, …). Without these every
    # dataset of a given handle is the SAME point cloud, so a shape pack cannot vary difficulty
    # within a topology class — and within-class variation is what stops a classifier winning
    # by reading dataset size. `data_kwargs` already threads runner-side; only this was missing.
    if run.dataset.get("params"):
        req["data_kwargs"] = dict(run.dataset["params"])
    if suite:
        req["metrics"] = list(suite)
    return req


def run_one(
    run: Run, *, suite: Optional[Sequence[str]] = None, engine: str = "mock",
    out_dir: Path | str = "outputs/experiment", device: Optional[str] = None,
) -> Result:
    """Run one cell. Never raises: a failed cell is a Result with ok=False, so one bad
    (recipe, dataset) pair cannot kill a sweep — but it is never mistaken for data."""
    from manyruns import modes

    try:
        res = modes.run("infer", _request(run, suite=suite, engine=engine,
                                          out_dir=Path(out_dir)), device=device)
    except Exception as e:  # noqa: BLE001 - a cell may fail for many engine-side reasons
        return Result(run=run, ok=False, error=f"{type(e).__name__}: {e}")
    status = res.get("status") or {}
    return Result(
        run=run,
        g_vector=res.get("g_vector") or {},
        # Both come from the runner. `ok` is "nothing errored"; a run whose steps all failed
        # returns an empty g-vector and would otherwise read as a legitimate row of the table.
        ok=bool(res.get("ok", True)),
        # `.get("complete", …)` defaults to `ok` rather than to True: an OLDER record — one
        # written before the key existed, and `index.jsonl` is append-only so those are on real
        # disks — meant "every step ran" by its `ok`, which is exactly this key's question.
        # Defaulting to True instead would silently relabel every declined step in the archive
        # as having run.
        complete=bool(res.get("complete", res.get("ok", True))),
        status=status,
        steps=list(res.get("steps") or []),
    )


def _init_worker() -> None:
    """Pin BLAS threads in each worker. Without this, N processes each spawn a full BLAS
    pool and contend; the numeric libraries are imported lazily inside the run, so setting
    this at worker start takes effect."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def run_all(
    runs: Iterable[Run], *, suite: Optional[Sequence[str]] = None, engine: str = "mock",
    out_dir: Path | str = "outputs/experiment", device: Optional[str] = None,
    workers: int = 1,
) -> list[Result]:
    """Run the plan. `workers=1` runs in-process (debuggable); >1 uses a PROCESS pool.

    .. important::
       With ``workers > 1`` your driver script MUST guard its entry point::

           if __name__ == "__main__":
               ex.run_all(runs, workers=8)

       macOS and Windows spawn (rather than fork) workers, so each child re-imports the
       driver module. Without the guard the child re-runs the sweep and tries to spawn its
       own pool, which dies as a `BrokenProcessPool` with an unreadable traceback. This is
       the caller's responsibility per the multiprocessing docs, so we cannot fix it for
       you — but we can say so plainly instead of letting you read a stack dump.
    """
    runs = list(runs)
    if workers <= 1:
        return [run_one(r, suite=suite, engine=engine, out_dir=out_dir, device=device)
                for r in runs]

    from concurrent.futures import BrokenExecutor, ProcessPoolExecutor

    kw = {"suite": suite, "engine": engine, "out_dir": str(out_dir), "device": device}
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = [pool.submit(_run_one_kw, r, kw) for r in runs]
            return [f.result() for f in futures]
    except BrokenExecutor as e:
        raise RuntimeError(
            "the worker pool died before running anything. The usual cause is a driver "
            "script with no entry-point guard: on macOS/Windows each worker re-imports "
            "your module, so the sweep must sit under `if __name__ == \"__main__\":`. "
            f"Use workers=1 to rule out the pool itself. (original: {type(e).__name__}: {e})"
        ) from e


def _run_one_kw(run: Run, kw: dict) -> Result:
    """Module-level so it pickles (a lambda or closure would not)."""
    return run_one(run, **kw)


def to_rows(results: Iterable[Result]) -> list[dict]:
    """Tidy the results: one row per run, metric columns + label columns.

    Rows are deliberately RAGGED — a recipe that emits no embedding cannot produce a
    trajectory metric, so that column is simply absent for it. Which metrics a recipe can
    even populate is itself signal that separates workflows, so the sparsity is not imputed
    away here."""
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "recipe": r.run.recipe["name"],
            "dataset": r.run.dataset["name"],
            "shape": r.run.dataset.get("shape", "unknown"),
            "seed": r.run.seed,
            "ok": r.ok,
            "complete": r.complete,
        }
        for k, v in (r.g_vector or {}).items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue          # notes/strings are diagnostics, not metric columns
            row[k] = float(v)
        rows.append(row)
    return rows


def metric_columns(rows: Sequence[dict]) -> list[str]:
    """Every numeric metric column present in the table (labels excluded).

    Selection is by TYPE, not by name. It used to be "any key not in `LABEL_COLUMNS`", which
    made that tuple a closed list nobody could extend: attaching a second label to score
    against — `topology`, say — silently turned it into a metric column, and `rank_metrics`
    then died in `round()` on a string. A label is a label because it is not a number, so
    test for that instead of maintaining a registry of exceptions.
    """
    cols: set = set()
    for r in rows:
        cols |= {k for k, v in r.items()
                 if k not in LABEL_COLUMNS
                 and not isinstance(v, bool)          # bools are flags, not measurements
                 and isinstance(v, (int, float))}
    return sorted(cols)


def rank_metrics(rows: Sequence[dict], label: str = "recipe", *, min_coverage: float = 0.5) -> list[dict]:
    """Rank metrics by how well they SEPARATE the given label.

    For each metric: `coverage` (fraction of runs that produced it — a metric only some
    protocols can emit is still informative, but it is scored on the rows that have it),
    and a one-way ANOVA F over the label groups. Metrics below `min_coverage`, or with no
    variance, or with fewer than two populated groups, are reported with `f=None` rather
    than silently dropped.

    Sorted by F descending: the top-k are the g-vector definition, where k is chosen where
    the separation plateaus.
    """
    from scipy.stats import f_oneway

    ok_rows = [r for r in rows if r.get("ok")]
    n = len(ok_rows) or 1
    out = []
    for col in metric_columns(rows):
        present = [r for r in ok_rows if col in r]
        coverage = len(present) / n
        groups: dict = {}
        for r in present:
            groups.setdefault(r.get(label), []).append(r[col])
        usable = [v for v in groups.values() if len(v) >= 2]
        f = p = None
        if coverage >= min_coverage and len(usable) >= 2:
            spread = {round(x, 12) for v in usable for x in v}
            if len(spread) > 1:                       # f_oneway is undefined on a constant
                stat = f_oneway(*usable)
                f, p = float(stat.statistic), float(stat.pvalue)
        out.append({"metric": col, "coverage": round(coverage, 3), "n": len(present),
                    "groups": len(groups), "f": f, "p": p})
    return sorted(out, key=lambda d: (d["f"] is None, -(d["f"] or 0.0)))


def summarize(rows: Sequence[dict], labels: Sequence[str] = ("recipe", "shape")) -> str:
    """A short plain-text readout of the ranking against each label."""
    n_ok = sum(1 for r in rows if r.get("ok"))
    # Reported SEPARATELY, and only when it differs, because "48 ok" hid the thing worth
    # knowing: before #66 a sweep over the bundled synthetics aggregated ~zero rows while every
    # step it cared about had run. A count of partial runs is the reader's cue that the rows
    # being compared are not all the same length of analysis.
    partial = n_ok - sum(1 for r in rows if r.get("ok") and r.get("complete"))
    lines = [f"{len(rows)} runs · {n_ok} ok"
             + (f" ({partial} with a step declined)" if partial else "")
             + f" · {len(metric_columns(rows))} metric columns", ""]
    for label in labels:
        lines.append(f"── separates {label} ──")
        for d in rank_metrics(rows, label)[:12]:
            f = "  n/a" if d["f"] is None else f"{d['f']:7.2f}"
            lines.append(f"  F={f}  cov={d['coverage']:.2f}  {d['metric']}")
        lines.append("")
    return "\n".join(lines)


def write_table(rows: Sequence[dict], path: Path | str) -> Path:
    """Persist the tidy table as CSV — the resolved, post-run record of what was measured."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(LABEL_COLUMNS) + metric_columns(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def rerun(result: Result, **overrides: Any) -> Result:
    """Re-run one cell (e.g. with a different seed) — useful when a cell fails."""
    return run_one(replace(result.run, **overrides))
