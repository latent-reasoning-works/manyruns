"""The demo's act-2 dataset: pbmc3k's raw counts, wearing the cell types it always had.

Act 2's whole claim is "same loop, same keys, real data", and `data/pbmc3k_raw.h5ad` undercuts
it: that file carries ZERO `.obs` columns (spec `[M2]`), so `loading.labels_of` returns
`(None, None)`, `ctx["color"]` is never populated from a label, and `figspec` falls back to
colouring the scatter by the embedding's own first axis — a picture that encodes nothing but
x-position. The labels exist; they are just in a different file.

THE MATRIX AND THE LABELS COME FROM DIFFERENT FILES ON PURPOSE, and this is the decision a
future reader will otherwise undo by reaching for `pbmc3k_processed()` alone. That file has the
cell types, but its `.X` is SCALED — measured min -2.85, max 10.0, not counts. Two guards then
fire: `prep._step_normalize` declines a matrix that is not counts, and `prep._step_transform`
re-reads the frame's ORIGINAL `.X` rather than the working matrix (`prep.py:379`, "would be the
SECOND transform on this matrix"), so it declines too. The demo's preprocessing path
(normalize -> log1p -> pca -> phate) would become two silent no-ops and a PCA of scaled data.
So: the RAW counts, subset to the barcodes the processed file kept, plus one label column.
That takes the same route act 1 does — `load_labeled` -> `labels_of` -> `ctx["color"]` -> the
scatter — and `looks_like_counts` (`loading.py:420`) still answers True.

`cell_type` rather than `louvain`, though both are in `vocab.GROUP_KEYS` (`vocab.py:221`):
`_find` returns the FIRST candidate present and `cell_type` heads that tuple, so it is the name
that wins deterministically. It is also the honest one — upstream already renamed these louvain
communities to curated type names, so they are not integer clusters. NOT `group`: that is in
`vocab.CONDITION_KEYS`, where it means a case/control arm, and a column named that would be
read as a CONDITION and colour the plot under the wrong kind — the same trap
`make_tree_fixture.py` dodges by calling act 1's column `branch`. No `stage` either: that is in
`TIME_KEYS`, checked FIRST, and would be handed to MIOFlow as a clock.

This is a SUBSET, not a relabelling: 2638 of 2700 barcodes match, and the 62 that do not are
the reference analysis's QC exclusions, reproduced exactly on the raw matrix — 57 cells with
>=5% mitochondrial reads over the 13 `MT-` genes, 5 with >=2500 genes detected, no overlap.
They are not cells of unknown type, so they are dropped rather than labelled "Unassigned": the
colour path has no null bucket (`vocab.numeric_time` ranks label strings alphabetically onto an
even [0,1] viridis ramp), so an "Unassigned" level would sort last, land on 1.0, and be the
BRIGHTEST thing in the act-2 figure — while also shifting the colour of all eight real types.

There is no RNG here to seed, so the checks below are what a seed would have been. An earlier
revision of this paragraph said the five asserts made a re-cut upstream file "fail loudly";
that was wrong, and it was measured. A stand-in where upstream PERMUTES `louvain` across the
same 2638 barcodes passes ALL FIVE — same shape, same join count, same row order, same eight
names, no NaN — and even reproduces `value_counts` exactly, while 1952 of 2638 cells (74.0%)
carry the wrong type. Permutation preserves every property a cardinality assert can see; the
analytic rate for this label distribution is 1 - sum(p^2) = 74.1%.

WHAT SEES IT IS THE MAPPING, HASHED. `LABELS_SHA256` is sha256 over the barcode->label pairs,
sorted and joined `barcode<TAB>label` — deliberately not over a file. The 24 MB download's
hash would pin `obsm`, `uns`, `louvain_colors` and a neighbour graph this fixture never reads,
and would fire on an anndata format bump that changed no label; the derived fixture's own hash
would additionally pin h5py, HDF5 and gzip (it IS byte-reproducible here — 06ec6d48... across
two rebuilds and the shipped file, anndata 0.13.2 / h5py 3.16.0 / HDF5 2.0.0 — which is exactly
the version set it would silently pin). Sorting the pairs makes the hash independent of row
order, which the assert above already covers. The sorted and unsorted hashes coincide TODAY
because these barcodes arrive already sorted, which is precisely why an order-dependent hash
would look correct until upstream reordered — the same trap the `.reindex` above dodges.

THE CACHE HASH IS A DIAGNOSTIC, NOT A GATE. `sc.datasets.pbmc3k_processed` is one line —
`read(settings.datasetdir / "pbmc3k_processed.h5ad", backup_url=...)` — and
`readwrite._check_datafile_present_and_download` short-circuits on `path.is_file()`: presence
is the ONLY check, and on a warm cache no network call happens at all. `readwrite._download`
does write to a NamedTemporaryFile and `.replace()` it, so a partial file cannot land, but
nothing verifies the bytes. `SOURCE_SHA256` is printed on failure because it discriminates the
two causes the label hash cannot tell apart: cache hash also changed => upstream re-cut its
file; cache hash unchanged => this script's derivation drifted under a library upgrade.

`raise SystemExit` rather than a sixth `assert`: `python -O` strips an assert, and this is the
one check this docstring promises will fire. It runs BEFORE the write, so a wrong annotation
cannot overwrite a correct `data/pbmc3k_annotated.h5ad` a presenter already has.

This is a sidecar. It is NOT a claim that manyruns should ship annotated data, and the result
is NOT a new named dataset — it is a file a person drops, passed by path.

    python docs/sidecars/make_pbmc3k_annotated.py data/pbmc3k_annotated.h5ad
"""
import hashlib
import sys
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

from manyruns.pipeline import loading as _loading

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/pbmc3k_annotated.h5ad"
RAW = sys.argv[2] if len(sys.argv) > 2 else "data/pbmc3k_raw.h5ad"
LABELS_SHA256 = "dbfc5c007cb8d2922a556b742ebfb82be7029e30707e9ba970bf10d57e4a6f9e"
SOURCE_SHA256 = "0db367b991dd95809732b218539ede489bea99113807f62ebd7ccc970025fe38"


def _labels_sha256(adata) -> str:
    """sha256 over the barcode->label MAPPING: sorted `barcode\\tlabel` pairs, utf-8."""
    pairs = sorted(zip(map(str, adata.obs_names), adata.obs["cell_type"].astype(str), strict=True))
    return hashlib.sha256("\n".join(f"{b}\t{c}" for b, c in pairs).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# `sc.settings.datasetdir` defaults to the CWD-relative `PosixPath('data')`, so the 24 MB cache
# lands wherever the script was launched from. Pinning it beside the raw file keeps the two
# inputs together and makes the download-once property hold when OUT is a tmp_path.
sc.settings.datasetdir = Path(RAW).parent

#: Genes kept. 2,000 is scanpy's own tutorial default and `pbmc3k_processed`'s 1,838
#: is the same decision; see the block above the selection for what it costs.
N_GENES = 2000

pro = sc.datasets.pbmc3k_processed()   # read-or-download; `.is_file()` short-circuits the network
raw = ad.read_h5ad(RAW)

keep = raw.obs_names.isin(pro.obs_names)
a = raw[keep].copy()                   # `.copy()`: assigning `.obs` on a sliced VIEW may not persist
# `.reindex` is label-based, so the column follows the barcodes rather than row position. The
# two orders do coincide today (asserted below), which is exactly why a positional `.values`
# assignment would look correct right up until an upstream reordering made it wrong.
a.obs["cell_type"] = pd.Categorical(pro.obs["louvain"].reindex(a.obs_names).astype(str))

assert pro.shape == (2638, 1838), f"upstream pbmc3k_processed changed shape: {pro.shape}"
assert int(keep.sum()) == 2638, f"barcode join matched {int(keep.sum())}, not 2638"
assert list(a.obs_names) == list(pro.obs_names), "the join no longer preserves row order"
assert a.obs["cell_type"].nunique() == 8, "the eight named cell types are the whole point"
assert not a.obs["cell_type"].isna().any(), "every kept barcode must carry a label"

got = _labels_sha256(a)
if got != LABELS_SHA256:
    cache = Path(sc.settings.datasetdir) / "pbmc3k_processed.h5ad"
    seen = _file_sha256(cache) if cache.is_file() else "(absent)"
    raise SystemExit(
        f"\nREFUSING TO WRITE {OUT}: these are not the labels this fixture was built from.\n"
        f"  labels sha256 expected : {LABELS_SHA256}\n"
        f"  labels sha256 got      : {got}\n"
        f"  {cache} : {seen}\n"
        f"  ...recorded            : {SOURCE_SHA256}\n"
        "Shape, join count, row order, the eight names AND their value_counts can all be\n"
        "unchanged while every cell is relabelled: a permutation of upstream's `louvain`\n"
        "mislabels 74% of cells (measured 1952/2638) and passes all five asserts above.\n"
        "If the cache hash ALSO differs, upstream re-cut its file; if it matches, this\n"
        "script's own derivation drifted under a library upgrade.\n"
        "THE DEMO DOES NOT NEED A REBUILD: an existing fixture on disk is untouched.\n"
        "Do not re-pin to make this pass. `data/` is gitignored (.gitignore:67), so this\n"
        "constant is the only committed record of the annotation.")

# THE GENE AXIS IS CUT TO 2,000, AND THAT IS WHAT MAKES ACT 2 A DEMO RATHER THAN A WAIT.
# `suite.measure` — which the run calls FOUR times on this recipe (inline in the `pca` step, twice
# in `_tuned_apart`'s rejected/accepted pair, and once in `_finalize`) — takes the working matrix,
# not the embedding, so its cost tracks the gene axis. Measured on this fixture, same cells, same
# metrics, one call:
#
#     32,738 genes   41.82 s        2,000 genes   3.93 s
#
# which is 177 s of arithmetic against 18 s for the whole act. The 177 s is not a projection: an
# adversarial pass drove the real app headless and clocked 45.6 s to the first question and 174.5 s
# end to end, against act 1's 1.6 s / 2.6 s.
#
# THE SUBSET IS ALSO THE MORE HONEST FIXTURE, which is why this is not a demo cheat. Selecting
# highly variable genes before embedding is what every scRNA workflow does — scanpy's own pbmc3k
# tutorial does it, and `pbmc3k_processed` ships 1,838 genes for exactly this reason. Handing PHATE
# 32,738 raw columns is the unrealistic path, and act 2 was on it.
#
# COMPUTED ON A NORMALISED COPY, APPLIED TO THE RAW COUNTS. `highly_variable_genes` wants
# log-normalised input, and the fixture must stay counts or `looks_like_counts` fails and the
# demo's own `normalize`/`transform` steps decline. So the selection is made on a throwaway copy
# and only the COLUMN NAMES cross back. `flavor="seurat"` on log data is scanpy's default pairing;
# it is named rather than left implicit because the default has moved before.
_hvg = a.copy()
sc.pp.normalize_total(_hvg, target_sum=1e4)
sc.pp.log1p(_hvg)
sc.pp.highly_variable_genes(_hvg, n_top_genes=N_GENES, flavor="seurat")
_keep_genes = list(_hvg.var_names[_hvg.var["highly_variable"].values])
assert len(_keep_genes) == N_GENES, f"gene selection returned {len(_keep_genes)}, not {N_GENES}"
a = a[:, _keep_genes].copy()
assert _loading.looks_like_counts(a.X), "the subset must still be counts, or the prep steps decline"

# `compression="gzip"`: 21.2 MB -> 7.7 MB, and scanpy wrote the raw file the same way. The tree
# fixture omits it only because 248 KB is not worth a flag.
a.write_h5ad(OUT, compression="gzip")
print(f"wrote {OUT}: {a.shape}, {a.obs['cell_type'].nunique()} cell types "
      f"({raw.n_obs - a.n_obs} barcodes dropped as the reference analysis's QC exclusions, "
      f"gene axis cut to {a.n_vars} highly variable)")
