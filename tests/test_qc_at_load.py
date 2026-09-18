"""QC as the fact the front door can afford to measure.

**Why QC and not geometry.** The roster wants something comparable to put beside every dropped
file, and the obvious candidate — the g-vector's 12 metrics — cannot be computed there at all.
`trustworthiness`, `continuity` and `knn_preservation` are defined between an ORIGINAL and an
EMBEDDING, so before a DR step runs there is nothing to compare; `betti_0`/`betti_1` are
persistent homology and are hopeless on 50k cells. QC is three sparse reductions that never
densify (`_step_qc` keeps CSR — 18.3 MB against 353.6 MB dense on pbmc3k), so it is O(nnz) and
survives a file the geometry suite would not.

**One account, two callers.** `qc.facts` is the arithmetic; `pipeline.steps._step_qc` calls it
to fill a g-vector, and `qc.for_path` calls it to fill a roster row. The alternative — a second
implementation for the front door — is the failure `narrate.contents` already documents from the
other direction: two surfaces answering one question and disagreeing.
"""
from __future__ import annotations

import pathlib

import pytest

np = pytest.importorskip("numpy")
sparse = pytest.importorskip("scipy.sparse")

from manyruns import qc  # noqa: E402

PBMC = pathlib.Path(__file__).resolve().parent.parent / "data" / "pbmc3k_raw.h5ad"


def _toy():
    """Four cells, five genes, with the answers arranged to be countable by hand.

    gene 4 is all-zero and gene 3 appears in one cell, so at `min_cells=3` exactly two genes
    would be dropped. Cell 3 expresses one gene, so at `min_genes=2` exactly one cell would.
    `MT-a` carries half of cell 0's counts.
    """
    m = np.array([[5, 5, 1, 0, 0],
                  [4, 4, 2, 0, 0],
                  [3, 3, 3, 0, 0],
                  [0, 0, 0, 1, 0]], dtype=np.float32)
    return sparse.csr_matrix(m), np.array(["MT-a", "b", "c", "d", "e"])


def test_the_counts_are_never_densified():
    """THE PROPERTY THAT MAKES THIS AFFORDABLE AT THE FRONT DOOR. A dense copy of pbmc3k is
    353.6 MB against 18.3 MB stored, and the roster may hold a folder of them. Anything that
    calls `.toarray()`/`.todense()` here stops the whole idea from scaling, so it is pinned
    rather than left to review."""
    class Refuses(sparse.csr_matrix):
        """A CSR that will not become dense. Cheaper and more honest than counting calls: the
        test cannot pass by densifying something that merely was not watched."""

        def toarray(self, *a, **k):    # noqa: D102
            raise AssertionError("qc densified the counts matrix")

        def todense(self, *a, **k):    # noqa: D102
            raise AssertionError("qc densified the counts matrix")

    counts, genes = _toy()
    f = qc.facts(Refuses(counts), genes)
    assert f["qc_n_cells"] == 4     # it still answered


def test_the_thresholds_are_counted_and_nothing_is_removed():
    """QC REPORTS, IT DOES NOT FILTER — `_step_qc`'s docstring calls that architectural, because
    a step that returned fewer rows would change the output TYPE of a recipe from a g-vector to
    a smaller dataset. The counts are what would go at the declared thresholds."""
    counts, genes = _toy()
    f = qc.facts(counts, genes, min_genes=2, min_cells=3, max_pct_mito=5.0)

    assert (f["qc_n_cells"], f["qc_n_genes"]) == (4, 5)
    assert f["qc_would_drop_genes"] == 2      # the all-zero gene and the 1-cell gene
    assert f["qc_would_drop_cells"] == 1      # the cell expressing a single gene
    assert f["qc_removed"] == 0               # the contract, as a number in the record


def test_a_missing_mitochondrial_prefix_is_not_a_zero():
    """"0 %" is a claim about biology; "no gene is named that way" is a claim about the file, and
    only the second one is true when the prefix does not match. pbmc3k names 13 genes `MT-*`, a
    mouse dataset names them `mt-*`, and a dataset carrying gene IDs names none of them
    anything."""
    counts, genes = _toy()
    f = qc.facts(counts, genes, mito_prefix="ZZZ-")

    assert f["qc_pct_mito_median"] is None
    assert "not the same as it being zero" in f["qc_pct_mito_median_note"]


def test_no_gene_names_means_no_mitochondrial_answer():
    counts, _ = _toy()
    f = qc.facts(counts, None)

    assert f["qc_pct_mito_median"] is None
    assert f["qc_n_cells"] == 4               # the cell-side facts still answer


def test_the_step_and_the_front_door_are_one_account():
    """`_step_qc` must be `qc.facts` wearing a g-vector, not a second implementation of it.

    Two surfaces computing one number separately is exactly the failure `narrate.contents`
    records from the other direction, and it is invisible until the two disagree in front of
    someone."""
    from manyruns.pipeline.steps import _step_qc

    counts, genes = _toy()
    g: dict = {}
    _step_qc({"counts": counts, "genes": genes}, g, {}, None, [], None)
    direct = qc.facts(counts, genes)

    shared = {k: v for k, v in g.items() if k.startswith("qc_")}
    assert shared == {k: v for k, v in direct.items() if k.startswith("qc_")}


# ── the real file, where the numbers are already on the record ───────────────
@pytest.mark.skipif(not PBMC.is_file(),
                    reason="data/pbmc3k_raw.h5ad is a checkout-relative fixture; see docs/sidecars/")
def test_the_measured_numbers_match_what_the_step_already_documented():
    """`_step_qc`'s docstring states these, measured on this file in this checkout, as the
    argument for shipping QC first. If the extraction changed any of them it broke the thing it
    was extracted from — so the docstring is the fixture."""
    pytest.importorskip("anndata")
    f = qc.for_path(PBMC)

    assert (f["qc_n_cells"], f["qc_n_genes"]) == (2700, 32738)
    assert f["qc_median_genes_per_cell"] == 817.0
    assert f["qc_median_counts_per_cell"] == 2197.0
    assert f["qc_n_mito_genes"] == 13
    assert f["qc_would_drop_cells"] == 0            # none below 200 genes
    assert f["qc_would_drop_genes"] == 19024        # of 32,738 — the finding worth surfacing
    assert f["qc_would_drop_cells_mito"] == 57
    assert round(f["qc_pct_mito_median"], 2) == 2.03


@pytest.mark.skipif(not PBMC.is_file(),
                    reason="data/pbmc3k_raw.h5ad is a checkout-relative fixture; see docs/sidecars/")
def test_the_second_look_is_free():
    """The roster opens a file per row on EVERY launch, which is what `inspected.py` exists to
    stop. QC is heavier than the read it caches beside, so it has to land in the same cache or
    the front door pays for it forever.

    Asserted by counting READS rather than by timing: a duration test on a warm page cache
    proves nothing.
    """
    pytest.importorskip("anndata")
    from manyruns import inspected

    inspected.forget()
    reads = []
    original = qc.facts

    def counting(*a, **k):
        reads.append(1)
        return original(*a, **k)

    qc.facts = counting                      # type: ignore[assignment]
    try:
        first = qc.for_path(PBMC)
        second = qc.for_path(PBMC)
    finally:
        qc.facts = original                  # type: ignore[assignment]

    assert reads == [1], "the second look recomputed instead of reading the cache"
    assert first == second


def test_disjoint_cell_cuts_are_labelled_separately():
    """One cell has too few genes, the other excess mito: max(1, 1) hid one failure."""
    from manyruns import narrate

    counts = sparse.csr_matrix([[0, 1, 0], [10, 1, 1]])
    facts = qc.facts(counts, np.array(["MT-a", "b", "c"]),
                     min_genes=2, min_cells=1, max_pct_mito=5.0)
    assert facts["qc_would_drop_cells"] == facts["qc_would_drop_cells_mito"] == 1
    assert narrate.qc_drops(facts) == "1 cells (low genes) · 1 cells (high mito)"
