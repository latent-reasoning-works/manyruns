"""The act-2 fixture — `docs/sidecars/make_pbmc3k_annotated.py`.

Why a JOIN and not either file on its own: `data/pbmc3k_raw.h5ad` is counts but carries zero
`.obs` columns, so `labels_of` returns `(None, None)` and the scatter is coloured by nothing;
`pbmc3k_processed.h5ad` carries the cell types but its `.X` is SCALED (measured min -2.85,
max 10.0), and `prep._step_transform` re-reads the frame's ORIGINAL counts, so both `normalize`
and `transform` would skip on it. The fixture is the raw matrix wearing the processed file's
labels, and the assertion that keeps it honest is `looks_like_counts` — drop that and the
demo's preprocessing path silently turns into two no-ops.

No network. The annotation source is a 23.5 MB scanpy download; this test asserts on a cache a
human populated by running the sidecar once, and skips when it is absent.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.source_checkout

pytest.importorskip("anndata")
pytest.importorskip("scanpy")  # `[harness]` extra only — a bare `uv sync` must skip, not error


def test_the_fixture_carries_eight_cell_types_the_loader_reads_as_a_group_axis(tmp_path):
    import anndata

    from manyruns.pipeline import loading

    if not Path("data/pbmc3k_raw.h5ad").is_file():
        pytest.skip("pbmc3k is a checkout-relative ref; see manyruns#12 for fetch-and-verify")
    if not Path("data/pbmc3k_processed.h5ad").is_file():
        pytest.skip("the annotation source is a 23.5MB scanpy download; run the sidecar once to "
                    "populate scanpy's datasetdir, then this test is offline and hermetic")

    out = tmp_path / "pbmc3k_annotated.h5ad"
    subprocess.run([sys.executable, "docs/sidecars/make_pbmc3k_annotated.py", str(out)],
                   check=True, capture_output=True)
    a = anndata.read_h5ad(out)
    # 2,000 genes, not the raw 32,738: `suite.measure` reads the working MATRIX and the run calls
    # it four times, so the gene axis is what act 2's wall-clock is made of — measured 41.82s per
    # call at 32,738 against 3.93s at 2,000, i.e. 177s of arithmetic against 18s for the whole act.
    # Selecting highly variable genes before embedding is also what every scRNA workflow does.
    assert a.shape == (2638, 2000), "annotated barcodes x highly variable genes"
    assert len(set(a.obs["cell_type"])) == 8

    labels, kind = loading.labels_of(a)
    assert kind == "group", "the fixture's own column must be the one `labels_of` finds"
    assert len(set(labels)) == 8

    # Act 2's actual claim, and the one act 1 never has to make: the matrix is still COUNTS,
    # so `normalize` and `transform` run instead of skipping. This is why the labels are joined
    # onto the raw file rather than the processed one being used directly.
    assert loading.looks_like_counts(a.X) is True, "the scaled matrix would skip both prep steps"

    # The colour channel the demo turns on: eight labels must reach the scatter as eight
    # distinct values, with no NaN standing in for a cell that lost its label in the join.
    import numpy as np

    from manyruns.pipeline import io as _io

    color = np.asarray(_io._labels_to_numeric(labels), dtype=float)
    assert len(set(color.tolist())) == 8
    assert not np.isnan(color).any(), "a NaN here is a grey cell in the act-2 figure"

    # ...but eight DISTINCT values is not the same as eight READABLE ones, and that ramp is what
    # `_categorical` exists to replace: alphabetically ranked onto an even [0,1] viridis scale it
    # gave NK cells (154 cells) the brightest yellow and 15 Megakaryocytes the second brightest,
    # while CD4 T (1,144) and CD8 T (316) landed on adjacent teals. The kind is `group`, so the
    # run takes the categorical path and the NAMES — not ranks — are what the figure carries.
    values, category_colors, palette = _io._categorical(labels)
    assert len(category_colors) == 8, "one swatch per named type"
    assert len(set(category_colors.values())) == 8, "and eight DISTINCT swatches"
    # Okabe-Ito, not `tab10`: eight colour-blind-safe colours for exactly eight cell types.
    # `tab10` put CD4 T on green #2ca02c and CD8 T on red #d62728 — the deuteranope confusion
    # pair, and the two most interleaved populations in this figure.
    assert palette == "okabe-ito", "eight types fit the safe set exactly, so nothing wraps"
    assert set(category_colors) == set(map(str, labels)), "every drawn label is in the key"
    # dtype, not just contents: an object array pickles into the `.npz` and `figspec.load`'s
    # `allow_pickle=False` then refuses it, so the spec silently comes back None forever.
    assert values.dtype.kind == "U", values.dtype
