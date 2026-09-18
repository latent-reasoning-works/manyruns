"""The frame and the selection — what `state` IS, once a step may remove cells.

`runner._STATE_KEYS` held five row-indexed slots that a filter would have to update together
(`X`, `emb`, `pseudotime`, `labels`, `counts`), plus `ctx["color"]`, which is built once at
`runner.py:855` and never rebuilt. Missing one of those six fails SILENTLY: `io.py:106` drops
the colouring on a length mismatch, and `separation`/`composition` mis-align labels against
clusters without erroring.

So there is no second copy to keep in step. One immutable FRAME holds the data as loaded, and
a SELECTION says which of it the current step sees. Every row-indexed value is
`frame[...][rows]`, computed on demand, so "do the labels match the counts" is not a property
that has to be maintained — it is a property of there being one array and one index.

Indices are in ORIGINAL space and only ever narrow, which is what makes the chain the
provenance: a selection of `[0, 3]` says which rows of the file survived. `filter_mito` does
not have to record what it dropped for the record to be able to say so.

**Plain data on purpose.** Dicts and ndarrays, no classes — an agent driving manylatents or
manyagents must not have to import a type to use a surface.
"""
from __future__ import annotations

from typing import Any


def new_frame(counts: Any = None, genes: Any = None, labels: Any = None,
              layers: "dict | None" = None) -> dict:
    """The immutable original: the data exactly as loaded. Written once, never again.

    `counts` is the caller's matrix and is NOT copied — `loading.py:436` records that
    `state["counts"] IS adata.X`, an 18.3 MB CSR on pbmc3k against 353.6 MB densified. A step
    that needs to transform it produces a new array; nothing writes here.

    `layers` is `{name: matrix}`, every matrix on the SAME TWO AXES as `counts` — a second
    assay over the same cells and the same features (spliced/unspliced counts, a protein panel).
    That sameness is the whole reason this is cheap: :func:`view` indexes a layer with the code
    that already indexes `counts`, and `_apply_transition` re-reads them on the line that
    already re-reads `counts`/`genes`/`labels`. One loop, not a new rule.

    THEY ARE CARRIED AND NARROWED, NEVER TRANSFORMED. `normalize` and `transform` operate on the
    working matrix `X` and must not touch these. Two reasons, and the first is sufficient: the
    TOOL that reads a layer owns its own preprocessing (measured — a velocity adapter runs
    normalize_total -> log1p -> PCA -> neighbours -> moments itself, so a pre-transformed layer
    would be double-transformed), and velocity models want RAW counts, which is the one use case
    layers exist for. So after `normalize` the working matrix is normalized and the layers are
    raw, deliberately, and that divergence is a CAVEAT rather than a docstring — "the recipe
    says transform: log1p" invites exactly the wrong inference.

    Not copied, like `counts` — these are references to the caller's arrays.
    """
    return {"counts": counts, "genes": genes, "labels": labels, "layers": dict(layers or {})}


def full_selection(n: int) -> Any:
    """Every index of an axis of length `n`, in order — what a run starts from."""
    import numpy as np

    return np.arange(int(n), dtype=np.int64)


def narrow(selection: Any, mask: Any) -> Any:
    """`selection` composed with a boolean `mask` over ITS OWN positions.

    The mask is sized against what the step SAW, not against the frame — a second filter over
    three surviving cells passes a length-3 mask. The result stays in original-index space, so
    chains compose without a translation step.

    A wrong-length mask is the one way a caller can desync a selection, so it raises rather
    than letting numpy broadcast it into a wrong answer.
    """
    import numpy as np

    m = np.asarray(mask, dtype=bool)
    if m.ndim != 1 or m.shape[0] != selection.shape[0]:
        raise ValueError(
            f"mask has {m.shape[0] if m.ndim == 1 else m.shape} entries but the selection "
            f"holds {selection.shape[0]} — a filter's mask is sized against what it SAW")
    return selection[m]


def _take(value: Any, index: Any, axis: int) -> Any:
    """One array indexed along one axis, or None passed through."""
    if value is None:
        return None
    if axis == 0:
        return value[index]
    return value[:, index]


def view(frame: dict, rows: Any, cols: Any) -> dict:
    """The frame as the current step sees it — `{counts, genes, labels}`.

    Row indexing a CSR is cheap; column indexing is not, so `cols` is applied only when it is
    actually narrower than the frame. That keeps the common case (rows filtered, all genes
    kept) at one sparse row-slice. Equal length IS fullness here: a selection starts at
    `full_selection` and `narrow` only ever removes, so one that still has `n_cols` entries
    can only be `arange(n_cols)`.

    The skip is keyed on the MATRIX, not on `genes`. Keying it on `genes is not None` conflates
    "has gene names" with "has a gene axis": measured against that version, a 5 × 4 frame with
    `genes=None` under a 2-column selection returned a 5 × 4 view — full width, no error. That
    is the silent wrong-shape answer this module exists to make impossible, and it is not
    hypothetical, because `loading.py` only fills `genes` when the file carries `var_names`.
    """
    counts = frame.get("counts")
    genes = frame.get("genes")
    out_counts = _take(counts, rows, 0)
    if getattr(out_counts, "ndim", 0) == 2 and cols.shape[0] != out_counts.shape[1]:
        out_counts = _take(out_counts, cols, 1)
    return {
        "counts": out_counts,
        "genes": _take(genes, cols, 0),
        "labels": _take(frame.get("labels"), rows, 0),
        # Same two axes as `counts`, so the same rule: rows always, columns only when the
        # selection is actually narrower. A layer that is NOT cell-by-gene has no business here
        # and would be silently mis-sliced, which is why `new_frame` says what a layer is.
        "layers": {name: _take_both(matrix, rows, cols)
                   for name, matrix in (frame.get("layers") or {}).items()},
    }


def _take_both(value: Any, rows: Any, cols: Any) -> Any:
    """One cell-by-gene matrix under the current selection — the `counts` rule, factored out."""
    out = _take(value, rows, 0)
    if getattr(out, "ndim", 0) == 2 and cols.shape[0] != out.shape[1]:
        out = _take(out, cols, 1)
    return out
