"""Quality control as arithmetic, so the step and the front door share one account of it.

`pipeline.steps._step_qc` has computed these numbers since the QC recipe landed, and it computed
them *inside a step* — reachable only by running a recipe. The roster wants the same numbers
before anything runs, to put a comparable fact beside every dropped file, so the arithmetic moves
here and both call it. A second implementation for the front door is the failure
`narrate.contents` documents from the other direction: two surfaces answering one question, and
nobody finds out until they disagree in front of someone.

**WHY QC IS WHAT THE FRONT DOOR CAN AFFORD, and the geometry suite is not.** The obvious thing to
show beside a dataset is the g-vector, and it cannot be computed here at all. `trustworthiness`,
`continuity` and `knn_preservation` are defined BETWEEN an original and an embedding, so before a
DR step has run there is no second thing to compare against; `betti_0`/`betti_1` are persistent
homology and do not survive 50k cells; `loglog_consistency` is a statement about whether the LID
estimator's scaling law holds, which is a fact about the estimator rather than about the data and
has no business in front of a practitioner.

What is left is QC, and it is the right answer rather than the consolation prize: three sparse
reductions that never densify, O(nnz) rather than O(n²), and every number in it is one a
single-cell practitioner already reads first — median genes per cell, median counts, the
mitochondrial fraction, and how much a standard filter would remove. The thresholds are the
Scanpy pbmc3k tutorial's and are declared rather than settled: Heumos et al. 2023
(`10.1038/s41576-023-00586-w`) prefers MAD-based outlier detection to fixed cuts.

**IT REPORTS AND NEVER FILTERS.** `_step_qc`'s docstring calls that architectural and it applies
unchanged here: a function that returned a smaller matrix would change a recipe's output type
from a g-vector to a dataset, desynchronise labels fetched separately, and make `n_samples` mean
two things depending on where you look. `qc_removed` is `0` in the record, as a number, so an
edit that starts removing has to change a line that says it does not.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

#: The Scanpy pbmc3k tutorial's cuts. Declared, not laws — see the module docstring.
MIN_GENES = 200
MIN_CELLS = 3
MAX_PCT_MITO = 5.0
MITO_PREFIX = "MT-"


def facts(counts: Any, genes: Any = None, *,
          min_genes: int = MIN_GENES, min_cells: int = MIN_CELLS,
          max_pct_mito: float = MAX_PCT_MITO, mito_prefix: str = MITO_PREFIX) -> dict:
    """Every `qc_*` fact for one counts matrix. Never densifies, never filters.

    `genes` may be None — a matrix with no gene axis still answers every cell-side question, and
    says so about the mitochondrial one rather than reporting a zero.
    """
    import numpy as np

    out: dict = {"qc.min_genes": min_genes, "qc.min_cells": min_cells,
                 "qc.max_pct_mito": max_pct_mito, "qc.mito_prefix": mito_prefix}
    if counts is None:
        out["qc_n_cells"] = None
        out["qc_n_cells_note"] = (
            "no counts matrix in this run — qc reads state['counts'], which only a dataset "
            "with a gene axis carries; state['X'] is transformed and cannot answer this")
        return out

    n_cells, n_genes = int(counts.shape[0]), int(counts.shape[1])
    # Three reductions over the sparse matrix. `counts > 0` on a CSR touches the stored entries
    # only; `np.asarray(...).ravel()` flattens scipy's np.matrix result for both the sparse and
    # the dense case, so there is one code path rather than a branch per type.
    per_cell_genes = np.asarray((counts > 0).sum(axis=1)).ravel()
    per_cell_total = np.asarray(counts.sum(axis=1)).ravel()
    per_gene_cells = np.asarray((counts > 0).sum(axis=0)).ravel()

    out["qc_n_cells"], out["qc_n_genes"] = n_cells, n_genes
    out["qc_median_genes_per_cell"] = float(np.median(per_cell_genes))
    out["qc_median_counts_per_cell"] = float(np.median(per_cell_total))
    out["qc_would_drop_cells"] = int((per_cell_genes < min_genes).sum())
    out["qc_would_drop_genes"] = int((per_gene_cells < min_cells).sum())
    # The contract, as a number in the record rather than only in a docstring.
    out["qc_removed"] = 0

    if not _looks_countslike(counts):
        out["qc_input"] = ("this matrix does not hold integer counts, so 'counts per cell' is a "
                           "sum of already-transformed values, not a library size")

    if genes is None:
        out["qc_pct_mito_median"] = None
        out["qc_pct_mito_median_note"] = "no gene names — the mitochondrial fraction needs them"
        return out

    names = np.char.upper(np.asarray(genes).ravel().astype(str))
    mito = np.char.startswith(names, mito_prefix.upper())
    out["qc_n_mito_genes"] = int(mito.sum())
    if not mito.any():
        # 0 % would be a claim about biology. "Not named that way" is a claim about the file, and
        # it is the true one: pbmc3k names 13 genes MT-*, a mouse dataset names them mt-*, and a
        # dataset carrying gene IDs rather than symbols names none of them anything.
        out["qc_pct_mito_median"] = None
        out["qc_pct_mito_median_note"] = (
            f"no gene name starts with {mito_prefix!r} — the mitochondrial fraction is not "
            f"computable here, which is not the same as it being zero")
        return out

    mito_total = np.asarray(counts[:, mito].sum(axis=1)).ravel()
    frac = np.divide(mito_total, per_cell_total,
                     out=np.zeros_like(mito_total, dtype=float), where=per_cell_total > 0)
    out["qc_pct_mito_median"] = float(np.median(frac) * 100.0)
    out["qc_would_drop_cells_mito"] = int((frac * 100.0 > max_pct_mito).sum())
    return out


def for_path(p: "Path | str") -> Optional[dict]:
    """QC for a dropped file, read from the inspection cache when it is there.

    THE CACHE IS THE WHOLE REASON THIS IS AFFORDABLE. The roster opens a file per row on every
    launch — `inspected.py` exists to stop that, and QC is heavier than the read it caches
    beside, so it lands in the same place under the same rule: the key is `(size, mtime_ns)`, a
    miss and a corrupt entry both cost exactly one read, and the cache is allowed to change a
    duration and never an answer.

    Returns None when the file carries no gene axis to count — a point cloud is not a failure
    here, it is a dataset QC has nothing to say about.
    """
    from manyruns import inspected

    p = Path(p)
    cached = inspected.get(p)
    if cached and isinstance(cached.get("qc"), dict):
        return cached["qc"]

    counts, genes = _counts_and_genes(p)
    if counts is None:
        return None
    measured = facts(counts, genes)

    # MERGED, never overwritten: `inspected` already holds what `read_data` saw for this file and
    # the roster still needs it. A put that replaced the row would trade one cold read for
    # another.
    row = dict(cached or {})
    row["qc"] = measured
    try:
        inspected.put(p, row)
    except Exception:  # noqa: BLE001 - a cache that cannot be written is still a working app
        pass
    return measured


def _counts_and_genes(p: Path) -> tuple:
    """The raw counts and the gene names, or `(None, None)` for anything without a gene axis."""
    try:
        from manyruns.pipeline.loading import gene_axis, load_array

        obj = load_array(p)
        counts, genes = gene_axis(obj)
        return counts, genes
    except Exception:  # noqa: BLE001 - an unreadable file must not empty the roster
        return None, None


def _looks_countslike(matrix) -> bool:
    """`loading.looks_like_counts`, imported lazily so this module stays importable without it."""
    from manyruns.pipeline.loading import looks_like_counts

    return looks_like_counts(matrix)
