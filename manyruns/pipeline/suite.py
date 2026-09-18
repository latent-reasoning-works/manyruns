"""The declared metric suite, measured on a run's embedding.

The suite is orthogonal to the recipe on purpose (`configs/metrics/default.yaml`): a recipe
says *what transforms run*, the suite says *what is measured on the result*. That separation
is what lets one suite score every recipe comparably — and until this module existed it did
not happen. The suite was forwarded to `engine=manylatents` and to nothing else, so the
g-vector's cross-recipe intersection was ten keys of pure provenance — dataset shape and
PHATE hyperparameters — and no geometry at all:

    embed 10 keys · cflows 11 · contrast 14
    shared by all three: final_dim, n_embedded, n_features, n_samples, phate_dims,
                         phate.{decay,gamma,knn,n_landmark,n_pca}

Nothing in that list is a property of the data. `separation`, `composition` and
`pseudotime_range` — the only real measurements — each appear in exactly one recipe, so no
two runs of different recipes were ever comparable. "Match this dataset to known workflows"
would have been matching on `n_samples` and `phate.knn`.

**Declared, not conditional.** Every name in the suite gets a key in every g-vector, whether
or not it could be computed. Absent-because-the-engine-is-missing, absent-because-there-is-no-
embedding, and absent-because-it-raised are three different facts; a schema that simply omits
the key collapses all three into "this recipe does not emit that metric", which is the
disagreement the audit found across four g-vector constructors. The value is `None` and the
reason is in `<name>_note`, following the `separation` / `separation_note` shape already in
the g-vector.

**manyruns declares WHICH metrics matter; manylatents computes them** (CLAUDE.md's split).
The only compute in this file is `_Ambient`, the adapter that hands manylatents the ambient
matrix its faithfulness metrics compare against.

**Nothing here is a finding.** Every metric in the suite is derived from the embedding alone,
so `vocab.NULL_KIND` says each needs a `data` null — synthesise null data, re-run the *whole*
pipeline — and nothing implements one. The measured reason for that rule:
a statistic computed from the embedding scored p < 1e-30 on pure noise in 12/12 seeds, and a
cheap label-null would have certified it. These values are recorded so runs can be compared;
they are not evidence about the biology until the data null exists.

**A number outside its possible range is now said so.** Measured here before that was true:
the `embed` recipe on 600×40 iid gaussian noise reports `lid` 3.419 / 3.425 / 3.419 (seeds
0/1/2) on a `final_dim=3` embedding, every step `outcome=ok`, run `ok=True`. An intrinsic
dimension cannot exceed the ambient dimension it is measured in, so that reading is not
surprising, it is impossible — and the record could not tell you. The admissible range of
every declared metric is now declared next to it (`configs/metrics/default.yaml`, `ranges:`)
and checked against the shape of the run that produced the value. A violation is a WARNING —
the value stays, the reason goes in `<name>_note` — because 3.419 is an estimator artifact
worth recording, not a corrupt reading worth discarding. See `_admissible`.
"""
from __future__ import annotations

import math
import time
import warnings
from typing import Any, Optional

#: One key, stated once, rather than the same sentence repeated on thirteen `_note` fields.
#: A reader that wants to know whether a suite value may be narrated checks this.
NULL_STATUS = "data-null unimplemented — comparable, not a finding"

#: The subset measured PER STEP (`live`), so a dashboard can say "phate took lid from X to Y"
#: while the run is still going. Chosen by MEASUREMENT, not by taste. Warm cost, min of 3, on
#: an n×3 embedding against an n×50 ambient (M4 Max, numpy 2.2.6, manylatents from .venv):
#:
#:                                       400      2,700     10,000     20,000
#:     trustworthiness                 0.014      0.524      7.764     33.630   ← O(n²)
#:     betti_1                         0.025      0.822      0.852      0.847
#:     betti_0                         0.008      0.221      0.220      0.215
#:     loglog_consistency              0.003      0.039      0.158      0.333
#:     continuity                      0.004      0.021      0.106      0.187
#:     knn_preservation                0.003      0.014      0.078      0.160
#:     participation_ratio             0.002      0.012      0.044      0.092
#:     lid                             0.001      0.006      0.025      0.052
#:     outlier_score                   0.001      0.006      0.024      0.050
#:     anisotropy                      0.000      0.000      0.000      0.000
#:     geodesic_distance_correlation   0.000      0.000      0.000      0.000
#:     kernel_sparsity                 0.000      0.000      0.000      0.000
#:     ── full suite                   0.059      1.665      9.271     35.567
#:     ── LIVE_SUBSET                  0.007      0.066      0.252      0.538
#:
#: Excluded, each for a measured reason:
#:   trustworthiness, betti_1, betti_0  — 1.57 s of the full suite's 1.66 s at 2,700 points.
#:       Paying that on every step of a 3-step recipe is 4.7 s of pure overlay.
#:   continuity, knn_preservation       — cheap against a PCA'd ambient (0.047 s / 0.016 s at
#:       2,700×50) and NOT cheap against a raw one: 1.163 s / 1.023 s at 2,700×32,738, a
#:       25-64x blow-up driven by the ambient WIDTH, which the row-count guard below cannot
#:       see. `run_inproc` takes any 2-D array, so a raw gene matrix reaching a step is not
#:       hypothetical. Everything kept here reads `emb` alone, so its cost is a function of
#:       exactly the one number the guard checks.
#:   geodesic_distance_correlation, kernel_sparsity — free, and structurally NaN in this call
#:       shape: `_Ambient` withholds `get_gt_dists` on purpose (see its docstring) and
#:       `kernel_sparsity` needs the fitted `module`, which a post-hoc measurement has not
#:       got. Both would render as a permanently `[None, None]` row. The g-vector still
#:       carries them, with their reason — that is the record's job, not the overlay's.
LIVE_SUBSET = ("lid", "loglog_consistency", "anisotropy", "participation_ratio", "outlier_score")

#: Above this many embedding rows, per-step geometry does not run at all. 20,000 is where the
#: subset above costs 0.538 s per step — a third of what the FULL twelve-metric suite costs at
#: 2,700 (1.665 s), and 8x its own cost there; at 50,000 it is 1.417 s. Past that point the
#: live preview costs more than the record it previews, which is not a live view. Nothing
#: manyruns ships comes close (pbmc3k is 2,700 rows), so this is a bound, not a policy.
LIVE_MAX_POINTS = 20_000


# ── admissible ranges ────────────────────────────────────────────────────────────────────
#
# The VALUE-level instance of "type-correct, plausible, and never checked". The bounds
# themselves — and the reason for each — are declared beside the metric names in
# `configs/metrics/default.yaml`; this is only the machinery that reads and applies them.

#: The closed identifier set a bound may name. Every one is already in every g-vector
#: (`runner.GVECTOR_CORE`), which is what lets a bound depend on the RUN'S SHAPE rather than
#: be a constant: `lid`'s ceiling is 3 on a 3-D embedding and 10 on a 10-D one.
SHAPE_VARS = ("n_samples", "n_features", "n_embedded", "final_dim")

#: A range declared ABSENT. Not the same as a metric with no entry: this is a claim — we
#: looked and found no bound we can defend — and `outlier_score` is the one that makes it,
#: because LOF is an uncapped density ratio. A missing entry is a gap, and `check_ranges`
#: reports it; nothing reports this, because it is the answer.
UNBOUNDED = "unbounded"


def _resolve(token: Any, env: dict) -> Optional[float]:
    """One side of a bound → a number, or None when it names a shape this run does not carry.

    THE ENTIRE GRAMMAR: a numeric literal, or one identifier from `SHAPE_VARS`. No `eval`, no
    `ast`, no import by string — what a config file can make this process do is two
    productions long and readable in one screen. CLAUDE.md forbids string-based dynamic
    import for the same reason a bound expression must be auditable: the config layer ships
    inside the wheel and is edited by hand.

    There are deliberately NO operators. All twelve bounds the shipped suite declares are a
    literal or a bare identifier — `test_every_declared_bound_is_a_literal_or_a_shape_name`
    pins that — so an arithmetic layer here would be machinery ahead of its first use, which
    is the mistake CLAUDE.md names about registries. This is the ONE function that grows when
    a bound genuinely needs `n_embedded - 1`, and the test above is what will fail first."""
    if isinstance(token, bool):
        return None                                  # `True` is not the literal 1 here
    if isinstance(token, (int, float)):
        return float(token)
    if isinstance(token, str) and token in SHAPE_VARS:
        value = env.get(token)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _legal(token: Any) -> bool:
    """Whether a token is in the grammar at all. `None` is legal and means "no bound here"."""
    if token is None or (isinstance(token, str) and token in SHAPE_VARS):
        return True
    return isinstance(token, (int, float)) and not isinstance(token, bool)


def _parse(spec: Any) -> Any:
    """One `ranges:` entry → `UNBOUNDED` or `(low, high)`, or None if it is not one."""
    if spec == UNBOUNDED or (isinstance(spec, dict) and spec.get("in") == UNBOUNDED):
        return UNBOUNDED
    bound = spec.get("in") if isinstance(spec, dict) else spec
    if not isinstance(bound, (list, tuple)) or len(bound) != 2:
        return None
    if not all(_legal(t) for t in bound):
        return None
    if bound[0] is None and bound[1] is None:
        return None                                  # say `unbounded`, don't imply it
    return (bound[0], bound[1])


def _raw_ranges(name: str = "default", config_dir: Any = None) -> dict:
    """The suite file's `ranges:` block, verbatim. `{}` on any failure.

    Read here rather than through `catalog.load_suite`, which returns the metric NAMES and is
    consumed by four callers that want exactly that; a second return value would change all
    of them. The import is local for the same reason `runner._finalize`'s is — this module is
    imported by the step loop and must stay cheap."""
    try:
        from omegaconf import OmegaConf

        from manyruns import catalog

        path = catalog.metrics_dir(config_dir) / f"{name}.yaml"
        if not path.is_file():
            return {}
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
        raw = cfg.get("ranges") if isinstance(cfg, dict) else None
        return raw if isinstance(raw, dict) else {}
    except Exception:  # noqa: BLE001 - a range check that can abort a run is worse than none
        return {}


def ranges(name: str = "default", config_dir: Any = None) -> dict:
    """`{metric: UNBOUNDED | (low, high)}`. Entries that do not parse are DROPPED, not guessed.

    Lenient on purpose, which is why this is not called `load_ranges`: `catalog`'s `load_*`
    contract is "strict, so a bad file cannot flow into a run", and that is the wrong contract
    here. A malformed bound must not fail the run it is a check on — the same rule
    `runner._live_suite` states for the overlay. `check_ranges` is what makes the drop
    visible, following this codebase's load-strict / check-reports split."""
    out = {}
    for metric, spec in _raw_ranges(name, config_dir).items():
        parsed = _parse(spec)
        if parsed is not None:
            out[metric] = parsed
    return out


def check_ranges(declared: list, name: str = "default", config_dir: Any = None) -> list:
    """Problems with the range block against `declared`; empty means it covers the suite.

    This is the anti-drift guard. `ranges()` silently drops a bound it cannot parse, so
    without this a typo in the config would disable a check and look exactly like a run where
    nothing was out of range — which is the same class of failure the ranges exist to catch,
    one level up. A test asserts this is empty for the shipped suite."""
    raw = _raw_ranges(name, config_dir)
    bad = []
    for metric, spec in raw.items():
        if _parse(spec) is None:
            bad.append(f"{metric}: {spec!r} is not a legal bound — [low, high] over numeric "
                       f"literals and {list(SHAPE_VARS)} (null for one side), or {UNBOUNDED!r}")
        elif metric not in (declared or []):
            bad.append(f"{metric}: has an admissible range but is not in the suite")
    for metric in (declared or []):
        if metric not in raw:
            bad.append(f"{metric}: declared in the suite with no admissible range — say "
                       f"{UNBOUNDED!r} if there is no bound you can defend")
    return bad


def _shape_env(emb: Any, X: Any, already: dict) -> dict:
    """The free variables a bound may name, for THIS run.

    The arrays win over `already` where both exist: `emb` is the thing being measured, so its
    own width is what `lid`'s ceiling has to be, whoever else wrote a `final_dim` key.
    `n_samples`/`n_features` fall back to the g-vector because `run_manylatents` with a named
    dataset never sees the input matrix (`runner` documents that absence) — and a bound that
    names a shape this run does not carry is reported as unchecked, not silently passed."""
    env = {k: already.get(k) for k in SHAPE_VARS}
    if getattr(X, "ndim", 0) == 2:
        env["n_samples"], env["n_features"] = int(X.shape[0]), int(X.shape[1])
    if getattr(emb, "ndim", 0) == 2:
        env["n_embedded"], env["final_dim"] = int(emb.shape[0]), int(emb.shape[1])
    return env


def _render(spec: tuple) -> str:
    low, high = spec
    return f"[{'-inf' if low is None else low}, {'inf' if high is None else high}]"


#: The two verdicts that earn a note. The third — in range, nothing to say — is `None`, and
#: it is the common one. They are kept apart because "the value left its range" and "the
#: bound could not be checked" are different facts, and only the first is a count a table can
#: rank on: collapsing them would make an unmeasurable bound look like a caught violation,
#: which is the same silent-pass the ranges exist to catch, one level up.
OUT_OF_RANGE, UNCHECKED = "out_of_range", "unchecked"


def _admissible(value: float, spec: Any, env: dict) -> tuple:
    """`(verdict, note)` for one value against its declared range. `(None, None)` for in-range.

    Silence in the ordinary case is the point: the record stays quiet about a number that is
    fine, so a note here is a signal and not a caption. The value is NEVER touched — that is
    the design decision. `lid` 3.419 on a 3-D embedding is the Levina–Bickel estimator's
    finite-sample bias (it overshoots on raw iid noise at every width: 2.29 at d=2, 3.45 at
    d=3, 5.47 at d=5), not a corrupt reading, and nulling it would throw away a number that is
    still comparable across runs in order to make a point about one of them."""
    if spec is None or spec == UNBOUNDED:
        return None, None
    for token, side, outside in ((spec[0], "below", float.__lt__),
                                 (spec[1], "above", float.__gt__)):
        if token is None:
            continue
        bound = _resolve(token, env)
        if bound is None:
            # Declared but not evaluable — reachable only when the run carries no such shape
            # at all (`run_manylatents` with a named dataset never sees the input matrix).
            return UNCHECKED, f"range unchecked: {token} is not in this g-vector"
        if outside(value, bound):
            shown = f"{token}={bound:g}" if isinstance(token, str) else f"{bound:g}"
            return OUT_OF_RANGE, (f"out of range: {value:.6g} {side} {shown} "
                                  f"(admissible {_render(spec)}) — value kept, not corrected")
    return None, None


class _Ambient:
    """What manylatents' faithfulness metrics expect a `dataset` to be.

    `continuity`, `knn_preservation` and `correlation` read `dataset.data` — the high-
    dimensional matrix the embedding is compared against — so passing `None` would make every
    faithfulness metric in the suite fail. Duck-typed on purpose: the alternative is importing
    a manylatents dataset class into the product layer to carry one array.

    `get_gt_dists` is deliberately NOT provided. `geodesic_distance_correlation` checks for it
    and returns NaN when it is missing, and a fabricated ground-truth distance matrix would be
    worse than the NaN — which `measure` records as an explicit absence anyway."""

    def __init__(self, data: Any) -> None:
        self.data = data


def _compute_metric() -> Optional[Any]:
    """manylatents' `compute_metric`, or None where the engine is not installed."""
    try:
        from manylatents.metrics.registry import compute_metric
    except Exception:  # noqa: BLE001 - stackless: defer, never fake
        return None
    return compute_metric


def live(emb: Any, declared: list) -> dict:
    """`{metric: float | None}` for the cheap subset — the per-step view, not the record.

    `measure` runs once, at the end, and is authoritative; this runs after every step that
    changed the embedding, so the geometry can be watched moving rather than appearing all at
    once when the last step finishes. The two are deliberately different shapes:

    - No `_note` keys, no `suite_*` keys, no absences-with-reasons. A run with no engine
      installed returns `{}` here and the caller attaches nothing — a live panel that printed
      "manylatents not installed" twelve times per step would be worse than a quiet one. The
      stated absence still reaches the g-vector, once, from `measure`.
    - `{}` means "no live view was taken" (no engine, no embedding, too big, nothing
      declared). A per-metric `None` inside a non-empty dict means "measured and it did not
      come back a number" — the reason for that one is in the g-vector's `<name>_note`.

    Takes no ambient matrix on purpose: every metric in `LIVE_SUBSET` reads `emb` alone, which
    is what makes `LIVE_MAX_POINTS` a sufficient guard (see the constant's comment)."""
    if getattr(emb, "ndim", 0) != 2 or emb.shape[0] > LIVE_MAX_POINTS:
        return {}
    names = [n for n in (declared or []) if n in LIVE_SUBSET]
    if not names:
        return {}
    compute = _compute_metric()
    if compute is None:
        return {}
    out: dict[str, Any] = {}
    for name in names:
        try:
            value = compute(name, embeddings=emb, dataset=None)
            out[name] = float(value) if value is not None and math.isfinite(float(value)) else None
        except Exception:  # noqa: BLE001 - an overlay never fails the run it is watching
            out[name] = None
    return out


def measure(emb: Any, X: Any, declared: list, *, already: Optional[dict] = None,
            admissible: Optional[dict] = None) -> dict:
    """Measure `declared` on `emb`, returning flat g-vector keys.

    Every declared name appears in the result exactly once. A name that could not be measured
    gets `None` plus a `<name>_note` saying why, so the key set is a property of the SUITE and
    not of what happened to work. A name that WAS measured and landed outside its declared
    range keeps its value and gets the same `<name>_note`, saying that.

    Args:
        emb: the embedding to measure. None (no embedding step, or it failed) is a legal
            input and produces a fully-noted result rather than an empty one.
        X: the ambient matrix, for faithfulness metrics that compare against it.
        declared: metric names, from `catalog.load_suite()`.
        already: g-vector keys the engine has already filled in. A value there wins — the
            manylatents engine computes the suite itself, and recomputing it here would both
            waste the work and risk two numbers for one name. It is still RANGE-CHECKED:
            an admissible range is a property of the metric, not of who computed it, and
            `lid > final_dim` is exactly as impossible on the manylatents path.
        admissible: `{metric: range}` as `ranges()` returns. Defaults to the shipped suite's
            `ranges:` block; injectable so a test can pin one bound without editing config.
    """
    out: dict[str, Any] = {}
    if not declared:
        return out
    started = time.perf_counter()
    already = already or {}
    compute = _compute_metric()
    admissible = ranges() if admissible is None else admissible
    env = _shape_env(emb, X, already)
    flagged = 0

    blanket = None
    if compute is None:
        blanket = "metric suite unavailable: no manylatents — this install is incomplete"
    elif getattr(emb, "ndim", 0) != 2:
        blanket = "no embedding to measure"

    ambient = _Ambient(X) if getattr(X, "ndim", 0) == 2 else None
    measured = 0
    for name in declared:
        prior = already.get(name)
        if isinstance(prior, (int, float)) and not isinstance(prior, bool):
            measured += 1
            # The engine's VALUE is not second-guessed — no key is written for it, so it
            # stays exactly as the engine left it — but its range still is. Only the note
            # is emitted here, which is why a flagged prior gains a `_note` with no `name`
            # beside it in this dict; the value is already in the g-vector being updated.
            verdict, note = _admissible(float(prior), admissible.get(name), env)
            if note:
                out[f"{name}_note"] = note
                if verdict == OUT_OF_RANGE:
                    flagged += 1
            continue
        if blanket:
            out[name], out[f"{name}_note"] = None, blanket
            continue
        try:
            # The reason a metric declined is emitted as a WARNING, not as the return value —
            # `geodesic_distance_correlation` warns "no get_gt_dists() available" and then
            # returns NaN; `kernel_sparsity` warns "NoneType does not expose a kernel_matrix"
            # and then returns NaN. Without capturing it both absences read `returned nan`,
            # which says what happened and not why, and the two become indistinguishable in
            # the panel — the exact collapse "declared, not conditional" exists to prevent.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = compute(name, embeddings=emb, dataset=ambient)
        except Exception as exc:  # noqa: BLE001 - one bad metric must not lose the other twelve
            out[name] = None
            # This cap applies to metric-suite notes; ordinary step errors retain
            # the full string in runner.apply_step. This is not a general error-text limit.
            out[f"{name}_note"] = f"raised {type(exc).__name__}: {exc}"[:200]
            continue
        # NaN is how several manylatents metrics report "I could not do this" —
        # `geodesic_distance_correlation` returns it whenever the dataset exposes no ground
        # truth, which is every manyruns dataset. Recording NaN as a value would put a
        # confident-looking non-number into a comparison table.
        if value is None or not math.isfinite(float(value)):
            out[name] = None
            said = "; ".join(str(w.message) for w in caught)
            out[f"{name}_note"] = (said or f"returned {value!r}")[:200]
            continue
        out[name] = float(value)
        measured += 1
        # Kept, then flagged — in that order, and both. An out-of-range value still counts as
        # measured because it IS a measurement: `suite_measured` answers "did the suite run",
        # `suite_out_of_range` answers "does any of it mean what it says", and collapsing the
        # two would make a warning look like a failure to compute.
        verdict, note = _admissible(out[name], admissible.get(name), env)
        if note:
            out[f"{name}_note"] = note
            if verdict == OUT_OF_RANGE:
                flagged += 1

    out["suite_measured"] = measured
    out["suite_declared"] = len(declared)
    # One number a table can rank on. Without it, finding the run whose geometry is impossible
    # means reading twelve `_note` keys per run — which is how `lid` 3.419 on a 3-D embedding
    # survived in the record: nothing summarised it, so nothing looked.
    # Only when something was actually checked. Emitting `0` from a config with no `ranges:`
    # block (reachable via MANYRUNS_METRICS_DIR) reads as "nothing is out of range" when the
    # truth is "nothing was examined" — the same collapse between absent and clean that
    # `<name>_note` exists to prevent for individual metrics.
    if admissible:
        out["suite_out_of_range"] = flagged
    out["suite_null"] = NULL_STATUS
    out["suite_seconds"] = round(time.perf_counter() - started, 3)
    return out
