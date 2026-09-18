"""The `prep` group — normalization, transformation and filtering, all delegated.

Every executor here is a PURE FUNCTION of a view and returns a result dict. It writes to
neither ``state`` nor the frame, and it never composes a selection itself:

    fn(view: dict, X: Any, params: dict) -> dict

with exactly one of

    {"X": ndarray}                              a new working matrix
    {"mask": ndarray[bool], "axis": "rows"}     which cells survive
    {"mask": ndarray[bool], "axis": "cols"}     which genes survive

plus an optional ``{"report": {key: number}}`` merged into the g-vector by the caller.

`runner.apply_step` is the only place a mask reaches a selection. One place to be correct, one
place to test, and a step author cannot forget a slot because a step author never touches one.
The alternative — each step updating `X`, `emb`, `pseudotime`, `labels`, `counts` and
`ctx["color"]` itself — is six chances to forget one, and the forgotten slot fails SILENTLY:
`io.py:106` drops the colouring on a length mismatch rather than raising.

**Two arguments, not one, and the distinction is load-bearing.** ``view`` is the frame through
the current selection — the UNTRANSFORMED counts, the gene names, the labels. ``X`` is the
WORKING matrix, which is whatever the last transforming step produced. A filter thresholds on
counts (a library size is a count, not a log-ratio); a transform operates on the working
matrix, or `normalize → transform` would silently discard the normalization.

**No compute lives here.** `manylatents.singlecell.preprocessing` owns the algorithms (CLAUDE.md;
they landed in manylatents-omics#61 as seven pure functions), and every executor below is a
name-and-shape adapter over one of them — see `_ops`. What this module owns is the parameter
names a recipe declares, the closed vocabularies it validates against, and the shape of the
answer. Two adapters do carry a decision rather than a rename, and both are commented where they
sit: `detect_doublets` INVERTS upstream's `is_doublet` into a survives-mask, and `normalize`
reads the counts where `transform` reads the working matrix.

**Imports stay function-local**, like every other module in this package: `pipeline` must be
importable with numpy, scipy, sklearn, matplotlib and torch all blocked (see `narrate.py`'s
note on the stackless path).
"""
from __future__ import annotations

from typing import Any

from manyruns.installation import RUNTIME_REPAIR

#: The transforms a recipe may name. Closed, because a typo'd method silently defaulting to
#: log1p is a run that reports `ok` having done something the recipe did not declare — the
#: same argument `catalog.check_claims` makes for refusing a present-but-wrong field.
TRANSFORMS = ("log1p", "sqrt")

#: The view columns `filter_cells(per=...)` may group by. Closed for `TRANSFORMS`' reason, and
#: for a sharper one: `frame.view` returns `{counts, genes, labels}` and nothing else, so a `per`
#: naming anything else resolved to `None` through `view.get` and the percentile ran GLOBALLY —
#: the exact operation `per` exists to avoid. Measured on a 20 x 20 two-group matrix whose groups
#: differ 20x in library size, `pct_library_size=25, per="condition"`: every dropped cell came
#: from the low-library group (d0 8/10 kept, d9 10/10), against 8/10 and 7/10 for `per="labels"`
#: — a whole timepoint thinned instead of each group's worst cells, reported as `grouped: 0`.
#: `counts` and `genes` are absent deliberately: `genes` indexes the OTHER axis (one entry per
#: column, not per cell) and upstream refuses a groups array of the wrong length, and grouping
#: cells by the matrix itself is not a question.
GROUP_COLUMNS = ("labels",)

# The report mixes chosen thresholds with observed casualties. Declare which is which at
# the producer, so a dotted measurement such as `filter_cells.dropped` cannot become a setting.
REPORT_SETTINGS = frozenset({"filter_cells.min_genes", "filter_cells.pct_library_size",
                             "filter_genes.min_cells", "filter_mito.max_pct",
                             "normalize.target_sum", "transform.method"})


def _param(params: dict, key: str, default: Any) -> Any:
    """A declared param, or its default. An explicit `None` means "not declared" rather than
    "set to null", which is how a recipe omits a field it does not care about."""
    value = (params or {}).get(key)
    return default if value is None else value


def _ops():
    """The engine owns preprocessing; its module import does not load all optional deps."""
    try:
        from manylatents.singlecell import preprocessing
    except ImportError as exc:
        raise ImportError(f"scRNA preprocessing unavailable; {RUNTIME_REPAIR}") from exc
    return preprocessing


def _call(operation: str, *args: Any, **kwargs: Any) -> Any:
    """Guard the operation too: normalize imports scanpy only when called, not at import.

    Preserve the cause for diagnosis. Import errors are dependency failures, whereas bad
    matrices and algorithm errors must retain their own explanation.
    """
    try:
        return getattr(_ops(), operation)(*args, **kwargs)
    except ImportError as exc:
        raise ImportError(f"{operation}: scRNA preprocessing unavailable; {RUNTIME_REPAIR}") from exc


def _counts(view: dict, X: Any) -> Any:
    """The UNTRANSFORMED counts through the current selection, falling back to the working
    matrix. A filter thresholds on counts — a library size is a count, not a log-ratio — so
    every filter below reads this and not `X`. See the module docstring's two-arguments note."""
    counts = view.get("counts")
    return X if counts is None else counts


def _genes_or_refuse(view: dict, step: str) -> Any:
    """Gene names, or a refusal naming the fact that is missing.

    Refuses rather than degrading, and that is the whole reason `genes` is a declared fact:
    "removed 0 genes" from a nameless matrix is indistinguishable from a dataset where nothing
    was removable, and a step that cannot tell those apart reports `ok` for both. `vocab.unmet`
    normally prunes these steps before they run — this is the backstop for a caller that
    assembled a state by hand.
    """
    genes = view.get("genes")
    if genes is None:
        raise ValueError(
            f"{step} needs gene names and this run carries none — declare `genes` (a dataset "
            "with a gene axis), or drop the step")
    return genes


def _step_filter_cells(view: dict, X: Any, params: dict) -> dict:
    """Which cells survive — by genes detected, or by a per-group library-size percentile.

    Both forms are the recipe author's declaration and not a law: `min_genes=200` is the Scanpy
    pbmc3k tutorial's number, and Heumos et al. 2023 (`10.1038/s41576-023-00586-w`) recommends
    MAD-based outlier detection over fixed cuts. The percentile form exists because measured on
    pbmc3k, `min_genes=200` drops **0** of 2700 cells.

    `per` names a column of the VIEW (`labels` today) rather than passing an array, so the
    recipe stays declarative. It matters because library size differs systematically between
    collection days, and one global cut drops a whole timepoint rather than its worst cells.

    WHICH IS WHY A `per` THIS VIEW CANNOT SERVE IS REFUSED, both halves of it. A name outside
    `GROUP_COLUMNS` (`per: condition`, say) used to reach `view.get` and come back `None`, and a
    declared name whose column is empty on this dataset does the same — either way the step
    quietly performs the GLOBAL cut, which is a different operation with a different casualty
    list, and writes `grouped: 0` without saying the column was not found. `TRANSFORMS` gets this
    treatment for a typo'd `method`; the harm here is larger, because a wrong `method` is at least
    still the transform the report names.
    """
    pct = params.get("pct_library_size")
    if pct is None:
        min_genes = int(_param(params, "min_genes", 200))
        keep = _call("filter_cells", _counts(view, X), min_genes=min_genes)
        return {"mask": keep, "axis": "rows",
                "report": {"filter_cells.min_genes": min_genes,
                           "filter_cells.dropped": int((~keep).sum())}}

    declared = (params or {}).get("per")
    if declared is not None and str(declared) not in GROUP_COLUMNS:
        raise ValueError(
            f"filter_cells: per must be one of {GROUP_COLUMNS}, got {declared!r} — the view "
            "carries only those columns, and a name it does not hold makes the percentile a "
            "GLOBAL cut that drops a whole group rather than each group's worst cells")
    groups = view.get(str(declared) if declared is not None else "labels")
    if declared is not None and groups is None:
        # DECLARED AND UNSERVABLE. Refuses rather than falling back, for `_genes_or_refuse`'s
        # reason: the fallback is not a degraded version of what was asked for, it is the
        # operation `per` was declared to prevent, and it reports the same `dropped` count either
        # way. Undeclared `per` still defaults to `labels` and still runs globally when a dataset
        # carries none — nobody asked for a per-group cut there, and `grouped: 0` says so.
        raise ValueError(
            f"filter_cells: per={declared!r} was declared and this run carries no {declared} — "
            "a per-group percentile needs one value per cell. Drop `per` to cut globally on "
            "purpose, or run on data that carries the column")
    keep = _call("filter_cells", _counts(view, X), pct_library_size=float(pct), groups=groups)
    return {"mask": keep, "axis": "rows",
            "report": {"filter_cells.pct_library_size": float(pct),
                       "filter_cells.grouped": int(groups is not None),
                       "filter_cells.dropped": int((~keep).sum())}}


def _step_filter_genes(view: dict, X: Any, params: dict) -> dict:
    """Which genes survive — those detected in at least `min_cells` cells.

    A COLUMN narrowing, which is why it declares `genes` in `STEP_NEEDS`: the names have to
    travel with the axis they index, and `runner._apply_transition` is what keeps them in step.
    Measured on pbmc3k: `min_cells=3` drops 19,024 of 32,738 genes.
    """
    _genes_or_refuse(view, "filter_genes")
    min_cells = int(_param(params, "min_cells", 3))
    keep = _call("filter_genes", _counts(view, X), min_cells=min_cells)
    return {"mask": keep, "axis": "cols",
            "report": {"filter_genes.min_cells": min_cells,
                       "filter_genes.dropped": int((~keep).sum())}}


def _step_filter_mito(view: dict, X: Any, params: dict) -> dict:
    """Which cells survive a mitochondrial-fraction threshold.

    Reads gene NAMES to find the `MT-` prefix, so it refuses without them (`_genes_or_refuse`).
    Measured on pbmc3k: 13 `MT-*` genes, median 2.03 % mitochondrial, **57** cells over 5 %.
    """
    genes = _genes_or_refuse(view, "filter_mito")
    max_pct = float(_param(params, "max_pct", 5.0))
    prefix = str(_param(params, "prefix", "MT-"))
    keep = _call("filter_mito", _counts(view, X), genes, max_pct=max_pct, prefix=prefix)
    return {"mask": keep, "axis": "rows",
            "report": {"filter_mito.max_pct": max_pct,
                       "filter_mito.dropped": int((~keep).sum())}}


def _step_detect_doublets(view: dict, X: Any, params: dict) -> dict:
    """Flag likely doublets — and by default remove NOTHING.

    THE INVERSION IS THE BUG-PRONE PART: upstream returns `is_doublet`, where True means
    SUSPECT. A mask here means SURVIVES. Returning the upstream array unchanged would drop every
    cell except the doublets, which is the exact inverse of the intent and would still report
    `ok`.

    `remove` defaults to False because a doublet call nobody has inspected should not silently
    narrow a dataset — the step writes how many it found and leaves the data alone. That default
    is also why `vocab.NARROWING_STEPS` keys this step on `remove` rather than listing it
    unconditionally: with `remove: false` the mask is all-True, so a recipe placing it before an
    embedding is legal and must not be refused for a narrowing that cannot happen.

    THE HEAVIEST STEP IN THIS MODULE, with additional requirements for threshold detection.
    Measured: upstream runs scrublet, which needs `scikit-image` to pick a threshold itself
    (`threshold=None`, the default) and a 30-component PCA regardless, so it refuses on any
    matrix narrower than ~31 genes. Missing imports receive `_call`'s repair message; algorithm
    failures retain upstream's explanation. Declaring `threshold` avoids the first, nothing
    avoids the second, and this
    module has no business second-guessing either. It is why `preprocess.yaml` leaves this step
    out of the standard preamble rather than defaulting it on.
    """
    import numpy as np

    is_doublet = _call("detect_doublets",
        _counts(view, X), threshold=params.get("threshold"),
        random_state=int(_param(params, "random_state", 0)))
    remove = bool(_param(params, "remove", False))
    keep = ~np.asarray(is_doublet, dtype=bool) if remove \
        else np.ones(np.asarray(is_doublet).shape[0], dtype=bool)
    return {"mask": keep, "axis": "rows",
            "report": {"detect_doublets.found": int(np.asarray(is_doublet, dtype=bool).sum()),
                       "detect_doublets.removed": int((~keep).sum())}}


def _step_normalize(view: dict, X: Any, params: dict) -> dict:
    """Library-size normalization — every cell scaled to `target_sum`.

    Reads the COUNTS, not the working matrix, and that is the one place this module's two-argument
    rule points the other way from `transform`. A library size is a sum of counts; normalizing an
    already-log-transformed matrix scales log-ratios, which is meaningless and silent. The
    canonical order is `normalize → transform`, and `transform` reads `X`, so the pair composes.

    IT DECLINES DATA THAT IS NOT COUNTS, and it DECLINES rather than errors — the distinction is
    the whole design of this guard.

    Until the cutover `loading._anndata_matrix` ran the preamble behind `looks_like_counts`, so a
    non-counts matrix simply skipped it. Loading hands over the frame now and every bundled
    recipe declares `normalize` unconditionally, so the guard has to move here or an
    already-normalized matrix gets library-scaled again — silently, reporting `ok`, producing
    numbers that are not wrong-looking.

    Erroring was the first implementation and it was WRONG, measured: eight of the bundled
    recipes run on synthetic Gaussian point clouds that were never counts, and a `ValueError`
    turned every one of those runs red. Erroring says "this recipe is broken"; the truth is "this
    step has nothing to do here". `_StepSkipped` records `outcome="skipped"` with the reason on
    screen, which is the same thing the old code did except VISIBLE — the preamble's silence is
    exactly what the cutover exists to end, and replacing it with a crash would be overcorrecting
    past the target.

    `force: true` is the escape hatch, because the discriminator is a heuristic (non-negative
    integers, first 100 rows) and a legitimately non-integer count matrix — imputed, or restored
    from CPM — should not be unusable. Declaring it is the recipe author saying they looked.
    """
    from manyruns.pipeline.steps import _StepSkipped

    counts = _counts(view, X)
    if not bool(_param(params, "force", False)) and not _looks_like_counts(counts):
        raise _StepSkipped(
            "not counts — this matrix holds negative or non-integer values, so it was either "
            "already normalized or was never a count matrix. Library-size scaling it would "
            "produce numbers that look fine and mean nothing. Declare `force: true` if it really "
            "is counts")
    target_sum = float(_param(params, "target_sum", 1e4))
    return {"X": _call("normalize", counts, target_sum=target_sum),
            "report": {"normalize.target_sum": target_sum}}


def _has_negatives(X: Any) -> bool:
    """Does this matrix hold a negative value? — the discriminator `transform` declines on.

    Cheaper and broader than `looks_like_counts`, deliberately: a normalized matrix is
    non-integer but perfectly transformable, so asking "is this counts" here would decline the
    canonical `normalize → transform` pair. The only question `log1p` and `sqrt` care about is
    the sign.

    Reads `.data` on a sparse matrix — the stored entries — which is both correct and cheap: the
    implicit zeros cannot be negative, and touching them would densify 353.6 MB of pbmc3k to
    learn nothing. Answers False on anything it cannot inspect, the conservative direction, since
    a False lets the transform proceed rather than declining a matrix it did not understand.
    """
    import numpy as np

    try:
        values = X.data if hasattr(X, "data") and hasattr(X, "toarray") else np.asarray(X)
        return bool(np.any(np.asarray(values) < 0))
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def _looks_like_counts(X: Any) -> bool:
    """Integers, all non-negative — the discriminator `loading.looks_like_counts` applies.

    A THIRD copy of that expression would be the drift `vocab.py` records twice, so this
    delegates to manylatents' `looks_like_counts`, which shipped with the seven operations and is
    the same test. Reads the first 100 rows only: densifying all of pbmc3k materialises 353.6 MB
    from an 18.3 MB CSR to answer a yes/no question.

    Dependency failures must escape detection's best-effort guard: answering False for an
    absent inspector would make an unavailable normalization look like a benign skip.
    """
    try:
        return bool(_call("looks_like_counts", X))
    except ImportError:
        raise
    except Exception:  # noqa: BLE001 - detection is best-effort; see the conservative direction
        return False


def _step_transform(view: dict, X: Any, params: dict) -> dict:
    """`log1p` or `sqrt` over the WORKING matrix.

    `sqrt` is not decoration: it is the PHATE/MIOFlow embryoid-body convention, and until now
    it was reachable only as `loading._anndata_matrix(transform=...)` — a load-time flag for a
    step-level fact, declared at `app.py:285` and threaded through five call sites. Here it is
    a declared param of a declared step, which is what makes it settable per recipe.

    Reads ``X`` and not ``view["counts"]``, so `normalize → transform` composes. Reading the
    counts would log1p the raw matrix and discard the normalization, in a run that reports
    both steps `ok`.

    TWO GUARDS, because one matrix can be untransformable for two unrelated reasons — it is
    signed (the transform is undefined) or it has already been transformed (the transform is a
    second application). Both DECLINE with `force: true` as the escape hatch, for `normalize`'s
    reason; the sign guard is `_has_negatives` and the second is the frame's counts, below.
    """
    from manyruns.pipeline.steps import _StepSkipped

    method = str(_param(params, "method", "log1p"))
    if method not in TRANSFORMS:
        raise ValueError(f"transform: method must be one of {TRANSFORMS}, got {method!r}")
    # BOTH TRANSFORMS ARE UNDEFINED BELOW ZERO, and numpy expresses that as NaN plus a warning
    # rather than as an exception. This is the defect manyruns#54 was opened over, found by
    # driving the GUI: `loading` handed steps `obsm["X_pca"]`, which is signed, and `log1p` of it
    # produced **818 NaNs of 3000** with the step reporting `ok`. The NaN then travelled to
    # `pca`, which is the only thing that raised — several steps downstream of the cause, with a
    # message about missing values that names neither this step nor the sign.
    #
    # Declines rather than errors, for `normalize`'s reason: on a synthetic point cloud there is
    # nothing to transform and that is not a broken recipe. `force: true` is the same escape
    # hatch, and means the author accepts the NaNs.
    forced = bool(_param(params, "force", False))
    if not forced and _has_negatives(X):
        raise _StepSkipped(
            f"{method} is undefined for negative values and this matrix holds some — it is "
            "already transformed, or was never counts. Applying it would write NaNs that only "
            "surface several steps later. Declare `force: true` to do it anyway")
    # THE OTHER HALF OF `normalize`'S GUARD, and it has to be here or the pair comes apart. The
    # sign test above asks only about the sign, so an ALREADY-NORMALIZED-AND-LOGGED matrix —
    # non-negative, non-integer — sails through it: `normalize` declines that matrix "not counts"
    # and `transform` then logs on top of the log while reporting `ok`. Measured on an 8 x 6
    # Poisson matrix put through `normalize(1e4)` + `log1p` and handed back as the frame's counts:
    # max 8.517 -> 2.253, both steps on the record and one of them a lie. Nothing downstream
    # reveals it, which is the property that makes it worth a guard rather than a caveat.
    #
    # READS THE FRAME'S COUNTS, NOT `X`, and that is what keeps `normalize → transform`
    # composing: after `normalize` the working matrix is non-integer BY CONSTRUCTION, so asking
    # `_looks_like_counts(X)` here would decline the canonical pair (the reason `_has_negatives`
    # exists as a separate, sign-only test at all). The frame is the same evidence `normalize`
    # reads, so the two steps now stand or fall together — which is exactly what the pre-cutover
    # loader did with ONE `looks_like_counts(adata.X)` test gating `normalize_total` AND `log1p`
    # (`loading._anndata_matrix`, removed in 39d7b84). Silently, is the part that changed.
    #
    # ONLY WHEN THE FRAME HAS COUNTS. A point cloud loads with `counts=None` — `loading.gene_axis`
    # returns `(None, None)` for anything that is not AnnData — and then there is no untransformed
    # original to judge, so the sign test above is the whole guard, as before.
    counts = view.get("counts")
    if not forced and counts is not None and not _looks_like_counts(counts):
        raise _StepSkipped(
            f"{method} would be the SECOND transform on this matrix — the counts this run was "
            "loaded from are not integer counts, so they were normalized or transformed before "
            "manyruns saw them, and `normalize` declined on the same evidence. Declare "
            "`force: true` if this matrix really has not been transformed yet")
    # The CHECK stays here and the MATHS does not. `TRANSFORMS` is manyruns's vocabulary — the
    # closed set a recipe may name — and refusing a typo before dispatch is this module's job.
    # `np.log1p`/`np.sqrt` are not: they moved upstream with the other six operations, so there
    # is one implementation rather than manyruns's and manylatents' drifting apart.
    return {"X": _call("transform", X, method=method), "report": {"transform.method": method}}


#: ``name -> executor``, the table `runner._NAME_TABLES` reads so `dispatchable` can answer
#: "could this engine run this step" without running it. A plain dict so a test can substitute
#: one step by key, exactly as `steps._ANALYSIS_STEPS` allows.
#:
#: `hvg` is deliberately absent: spec §12 hands it to Zach, and `vocab.NARROWING_STEPS` already
#: carries its slot so the name is reserved without being dispatchable.
_PREP_STEPS = {
    "filter_cells": _step_filter_cells,
    "filter_genes": _step_filter_genes,
    "filter_mito": _step_filter_mito,
    "detect_doublets": _step_detect_doublets,
    "normalize": _step_normalize,
    "transform": _step_transform,
}
