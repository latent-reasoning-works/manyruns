"""The five readouts — what `analysis` says once a recipe stops ending at a picture.

THE GAP THESE CLOSE, measured before they existed: the `analysis` group had two entries
(`separation`, `composition`), both requiring a condition label, and **zero of the bundled
datasets declare `shape: case-control`** (still zero of
the 14 there are now — 9 `manifold`, 4 `clusters`, 1 `time-course`). So every
recipe that could actually run ended at an embedding. Worse, the one clustering algorithm the
engine offers reports nothing: measured on two gaussian blobs at 120×8,
`manylatents.api.run(algorithms={'latent': 'leiden'}, seed=0)` returns `embeddings` of shape
**(120, 1) float32 over {0., 1.}** and an **empty** `scores` dict — a clustering step whose
result reaches no record at all.

WHAT THESE TESTS ARE FOR, in order of what would hurt most if it broke:

  1. `qc` must never remove a cell or a gene. A step that silently shrinks the dataset breaks
     `check_aligned`, desynchronises labels fetched separately, and makes `n_samples` mean two
     things depending on where a reader looks (§5.5). `test_qc_reports_and_removes_nothing`.
  2. `rank_genes` must not normalize the caller's matrix in place. `state["counts"]` IS the
     loaded AnnData's `.X` (`loading.gene_axis`: *"the same object, not a copy"*) and
     `sc.pp.normalize_total` writes in place. `test_rank_genes_does_not_touch_the_callers_
     counts` fails if the `.copy()` goes.
  3. No readout here may be narrated as a finding. All five are computed from the embedding or
     from a partition of it, so `vocab.NULL_KIND` requires a `data` null and nothing implements
     one — `test_none_of_the_new_readouts_is_narrated_as_a_finding`.
  4. The preconditions must prune, not degrade. `cluster_quality`/`rank_genes` are illegal
     without a clustering step (`STEP_PRODUCES`), and since the cutover `rank_genes`/`qc` are
     illegal without a gene axis — a DATASET fact, declared by exactly one of the 14 bundled
     datasets (`configs/dataset/pbmc3k.yaml`'s `provides: [genes]`) and by no shape. `markers`
     must stay legal on that one: a precondition that fires on correct data is worse than none.

DEPENDENCIES ARE SKIPPED, NEVER FAKED. numpy is required (every other test file requires it);
sklearn / scanpy / anndata / scipy are asked for per test, because CI has numpy and not
anndata, and a suite whose coverage all skips on CI is coverage that exists on one laptop.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from manyruns.pipeline import steps as _steps  # noqa: E402
from manyruns.pipeline.steps import _StepSkipped  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────
def _blobs(n=120, d=6, k=3, seed=0):
    """`k` well-separated gaussian blobs, and the partition that generated them."""
    rng = np.random.default_rng(seed)
    per = n // k
    X = np.vstack([rng.normal(10 * i, 0.4, (per, d)) for i in range(k)])
    labels = np.repeat(np.arange(k), per)
    return X, labels


def _leiden_shaped(labels):
    """Labels in the shape manylatents' leiden actually leaves in `state["emb"]`:
    (N, 1) float32. Measured, not assumed — see this module's docstring."""
    return np.asarray(labels, dtype=np.float32).reshape(-1, 1)


def _counts(n=40, p=12, seed=0):
    """A small integer count matrix with gene names, two groups apart on the first 3 genes."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(2.0, (n, p)).astype(np.float32)
    X[: n // 2, :3] += 30          # the marker block, group 0
    genes = np.array([f"GENE{i}" for i in range(p)])
    return X, genes


# ── _partition: where a readout finds the clusters ───────────────────────────
def test_partition_prefers_the_state_slot_that_has_no_producer_yet():
    """`state["clusters"]` is read FIRST although nothing writes it (it is absent from
    `runner._STATE_KEYS`). That ordering is the whole point of the seam: when leiden stops
    overwriting `state["emb"]` (§5.1's option (a)), every readout follows with no edit."""
    labels, where = _steps._partition({"clusters": [0, 0, 1, 1], "emb": _leiden_shaped([9] * 4)})
    assert list(labels) == [0, 0, 1, 1], "the emb label column won over state['clusters']"
    assert where == _steps._FROM_CLUSTERS


def test_partition_reads_leidens_actual_output_shape():
    labels, where = _steps._partition({"emb": _leiden_shaped([0, 1, 1, 2])})
    assert list(labels) == [0, 1, 1, 2]
    assert labels.dtype.kind == "i", "a label used to index counts must be integral"
    assert where == _steps._FROM_EMB


def test_partition_refuses_coordinates_and_says_no_clustering_ran():
    """The failure that matters: a 3-column embedding is coordinates, and reading it as a
    partition would invent one cluster per distinct row."""
    labels, why = _steps._partition({"emb": np.zeros((10, 3))})
    assert labels is None
    assert "no clustering step has run" in why


def test_partition_refuses_a_one_column_embedding_of_real_values():
    """One column is necessary and not sufficient — a 1-D PHATE embedding is not a partition.
    Narrow on purpose: the test is integrality, so 0.5 disqualifies the whole column."""
    labels, why = _steps._partition({"emb": np.array([[0.0], [0.5], [1.0]])})
    assert labels is None
    assert "not a partition" in why


# ── 1. cluster_quality ───────────────────────────────────────────────────────
def test_cluster_quality_reports_the_partition_it_was_given():
    pytest.importorskip("sklearn")
    X, labels = _blobs()
    g = {}
    _steps._step_cluster_quality({"emb": _leiden_shaped(labels), "X": X}, g, {}, None, [], 3)
    assert g["cluster_quality_n"] == 3
    assert g["cluster_quality_min_size"] == g["cluster_quality_max_size"] == 40
    assert g["cluster_quality_singletons"] == 0
    assert g["cluster_quality_silhouette"] > 0.9, "three separated blobs must score high"


def test_cluster_quality_scores_in_the_input_because_leiden_destroyed_the_coordinates():
    """THE MEASURED COST of §5.1's option (b), pinned so it cannot be lost.

    When the partition comes out of `state["emb"]`, the coordinates leiden partitioned are
    gone — leiden wrote its label column over them. The silhouette is then computed in
    `state["X"]`, and the g-vector SAYS SO. Scoring in the label column instead would return
    ~1.0 for any partition whatsoever (each cluster is a single point there), which is the
    silent success this asserts against."""
    pytest.importorskip("sklearn")
    from sklearn.metrics import silhouette_score

    X, labels = _blobs()
    g = {}
    _steps._step_cluster_quality({"emb": _leiden_shaped(labels), "X": X}, g, {}, None, [], 3)
    assert g["cluster_quality_silhouette"] == pytest.approx(silhouette_score(X, labels))
    assert "state['X']" in g["cluster_quality_space"]
    assert g["cluster_quality_silhouette"] < 1.0


def test_cluster_quality_scores_in_the_embedding_when_the_partition_came_from_elsewhere():
    pytest.importorskip("sklearn")
    X, labels = _blobs()
    g = {}
    _steps._step_cluster_quality({"emb": X, "X": np.zeros_like(X), "clusters": labels},
                                 g, {}, None, [], 3)
    assert g["cluster_quality_space"] == "state['emb']"
    assert g["cluster_quality_silhouette"] > 0.9


def test_cluster_quality_counts_singletons_because_n_alone_cannot_say_overpartitioned():
    pytest.importorskip("sklearn")
    X, _ = _blobs(n=120, k=3)
    labels = np.arange(120) // 40
    labels[:2] = [98, 99]                      # two clusters of exactly one cell
    g = {}
    _steps._step_cluster_quality({"emb": _leiden_shaped(labels), "X": X}, g, {}, None, [], 3)
    assert g["cluster_quality_singletons"] == 2
    assert g["cluster_quality_min_size"] == 1


def test_cluster_quality_says_why_when_nothing_clustered():
    g = {}
    _steps._step_cluster_quality({"emb": np.zeros((10, 3)), "X": np.zeros((10, 3))},
                                 g, {}, None, [], 3)
    assert g["cluster_quality_n"] is None
    assert "no clustering step has run" in g["cluster_quality_n_note"]


def test_cluster_quality_refuses_a_silhouette_of_one_cluster():
    g = {}
    _steps._step_cluster_quality({"emb": _leiden_shaped([0] * 10), "X": np.zeros((10, 3))},
                                 g, {}, None, [], 3)
    assert g["cluster_quality_n"] == 1
    assert g["cluster_quality_silhouette"] is None
    assert "≥2 clusters" in g["cluster_quality_silhouette_note"]


# ── 3. rank_genes ────────────────────────────────────────────────────────────
def _scanpy():
    pytest.importorskip("anndata")
    pytest.importorskip("pandas")
    return pytest.importorskip("scanpy")


def test_rank_genes_names_the_genes_that_separate_the_groups():
    _scanpy()
    X, genes = _counts()
    labels = np.repeat([0, 1], 20)
    g = {}
    _steps._step_rank_genes({"counts": X, "genes": genes, "emb": _leiden_shaped(labels)},
                            g, {"top_n": 3}, None, [], 3)
    assert g["rank_genes_n_groups"] == 2
    assert g["rank_genes_n_tested"] == 12
    top0 = [line for line in g["rank_genes_top"] if line.startswith("0:")][0]
    # group 0 carries the +30 block on GENE0..GENE2 and nothing else does
    assert set(top0.split(": ")[1].split(", ")) == {"GENE0", "GENE1", "GENE2"}


def test_rank_genes_does_not_touch_the_callers_counts():
    """MUTATION CHECK. `state["counts"]` IS the loaded AnnData's `.X` — the same object, not a
    copy — and `sc.pp.normalize_total` writes in place. Drop the `.copy()` in `_step_rank_genes`
    and this fails: the caller's integer counts come back normalized to 1e4 per row, so every
    later step (and the file the user re-reads) sees a matrix nobody asked to transform."""
    _scanpy()
    X, genes = _counts()
    before = X.copy()
    state = {"counts": X, "genes": genes, "emb": _leiden_shaped(np.repeat([0, 1], 20))}
    _steps._step_rank_genes(state, {}, {}, None, [], 3)
    assert np.array_equal(state["counts"], before), "rank_genes normalized the caller's matrix"
    assert state["counts"] is X, "rank_genes replaced the caller's matrix object"


def test_rank_genes_records_no_p_value_anywhere():
    """Squair et al. 2021 (10.1038/s41467-021-25960-2): the clusters were DEFINED by the genes
    now being tested, so a p-value here is anticonservative by construction. It ships as an
    order over names plus a sentence saying so — a significance in the record is an invitation
    to promote it, and `vocab.NULL_KIND` has no null that could earn one."""
    _scanpy()
    X, genes = _counts()
    g = {}
    _steps._step_rank_genes({"counts": X, "genes": genes,
                             "emb": _leiden_shaped(np.repeat([0, 1], 20))}, g, {}, None, [], 3)
    banned = ("pval", "pvals", "p_emp", "__null", "qval", "score", "logfold")
    assert not [k for k in g if any(b in k.lower() for b in banned)]
    assert "not a test" in g["rank_genes_ranked_by"]


def test_rank_genes_says_why_without_a_gene_axis():
    g = {}
    _steps._step_rank_genes({"counts": None, "genes": None,
                             "emb": _leiden_shaped([0, 1])}, g, {}, None, [], 3)
    assert g["rank_genes_top"] is None
    assert "no gene axis" in g["rank_genes_top_note"]


def test_rank_genes_says_why_with_one_group():
    g = {}
    X, genes = _counts()
    _steps._step_rank_genes({"counts": X, "genes": genes, "emb": _leiden_shaped([0] * 40)},
                            g, {}, None, [], 3)
    assert g["rank_genes_top"] is None
    assert "≥2" in g["rank_genes_top_note"]


# ── 4. dpt ───────────────────────────────────────────────────────────────────
def _ribbon(n=400, seed=0):
    """A 1-D manifold in 3-D with a known parameter — the thing a pseudotime must recover."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1, n))
    Z = np.c_[t, 0.1 * np.sin(6 * t), rng.normal(0, 0.01, n)].astype(np.float32)
    return Z, t


def test_dpt_recovers_the_order_of_a_one_dimensional_manifold():
    _scanpy()
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    Z, t = _ribbon()
    state, g = {"emb": Z}, {}
    _steps._step_dpt(state, g, {}, None, [], 3)
    rho = abs(spearmanr(state["pseudotime"], t).statistic)
    assert rho > 0.9, f"dpt did not order a 1-D manifold (|rho| = {rho:.3f})"
    assert g["dpt_unreachable"] == 0
    assert g["dpt_range"] == [0.0, 1.0]


def test_the_dpt_default_is_two_components_and_scanpys_default_would_break_it():
    """THE MEASURED DEFAULT. DPT accumulates over the first `n_dcs` diffusion components; on a
    clean low-dimensional manifold the higher ones are harmonics of the first and fold the
    manifold back on itself. On this fixture |Spearman| against the true parameter is ~0.997 at
    n_dcs=2 and ~0.05 at scanpy's default of 10 — the pseudotime rises 0 → 0.99 and falls back
    to 0.35, which is not an ordering. The canonical `pseudotime` recipe feeds `dpt` a
    `diffusionmap` embedding, i.e. exactly the input the default fails on, so this pins the
    deviation rather than leaving it to a comment."""
    _scanpy()
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    Z, t = _ribbon()
    ours, theirs = {"emb": Z}, {"emb": Z}
    g_ours, g_theirs = {}, {}
    _steps._step_dpt(ours, g_ours, {}, None, [], 3)
    _steps._step_dpt(theirs, g_theirs, {"n_dcs": 10}, None, [], 3)
    assert g_ours["dpt.n_dcs"] == 2, "the default moved to scanpy's, which does not order this"
    assert g_theirs["dpt.n_dcs"] == 10
    rho_ours = abs(spearmanr(ours["pseudotime"], t).statistic)
    rho_theirs = abs(spearmanr(theirs["pseudotime"], t).statistic)
    assert rho_ours > 0.9 > rho_theirs, f"ours {rho_ours:.3f} theirs {rho_theirs:.3f}"


def test_dpt_counts_the_cells_it_cannot_reach():
    """The diagnostic that survived contact. Measured through `diffusionmap → dpt` on the
    ribbon: 81 of 400 cells had no path to the root, both orderings were wrong, and their
    AGREEMENT still read 0.907. The unreachable count is what said so."""
    _scanpy()
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0, 0.1, (60, 3)), rng.normal(60, 0.1, (60, 3))]).astype(np.float32)
    state, g = {"emb": Z}, {}
    _steps._step_dpt(state, g, {"n_neighbors": 5}, None, [], 3)
    assert g["dpt_unreachable"] == 60, "the far blob has no path to a root in the near one"
    assert g["dpt_unreachable_frac"] == 0.5
    assert np.isnan(state["pseudotime"]).sum() == 60, "unreachable cells must not take the max"


def test_dpt_refuses_to_order_something_with_no_coordinates():
    with pytest.raises(_StepSkipped) as e:
        _steps._step_dpt({"emb": None, "X": np.zeros((10, 3))}, {}, {}, None, [], 3)
    assert "needs an embedding" in str(e.value)


def test_dpt_writes_the_ordering_into_the_slot_that_reaches_disk():
    """The ordering IS the result; a scalar range is not. `artifacts.PERSISTED` carries
    `pseudotime`, so writing that slot is what makes the answer survive the run."""
    from manyruns import artifacts

    _scanpy()
    assert "pseudotime" in artifacts.PERSISTED
    Z, _ = _ribbon(n=120)
    state = {"emb": Z}
    _steps._step_dpt(state, {}, {}, None, [], 3)
    assert np.asarray(state["pseudotime"]).shape == (120,)


# ── discretize_time: what lets an ordering step actually reach MIOFlow ───────
def test_discretize_time_bins_an_ordering_into_the_requested_group_count():
    """The money property: MIOFlow needs discrete groups, not a continuum, and the bin count
    the recipe asked for is the bin count that lands in `state["labels"]`."""
    _scanpy()
    Z, t = _ribbon(n=200)
    state, g = {"emb": Z}, {}
    _steps._step_dpt(state, g, {}, None, [], 3)
    _steps._step_discretize_time(state, g, {"n_timepoints": 5}, None, [], 3)
    assert state["label_kind"] == "time"
    assert len(set(state["labels"].tolist())) <= 5
    assert g["discretize_time.n_timepoints"] == 5


def test_discretize_time_preserves_the_orders_spearman_rank():
    """Binning must not scramble the ordering it was handed — a MIOFlow trained on these bins
    should still be fitting the same 1-D progression `dpt` recovered, not noise."""
    _scanpy()
    spearmanr = pytest.importorskip("scipy.stats").spearmanr
    Z, t = _ribbon(n=200)
    state, g = {"emb": Z}, {}
    _steps._step_dpt(state, g, {}, None, [], 3)
    _steps._step_discretize_time(state, g, {"n_timepoints": 5}, None, [], 3)
    from manyruns import vocab

    numeric = vocab.numeric_time(state["labels"])
    rho = abs(spearmanr(numeric, t).statistic)
    assert rho > 0.9, f"binning destroyed the ordering (|rho| = {rho:.3f})"


def test_discretize_time_refuses_with_no_pseudotime():
    with pytest.raises(_StepSkipped) as e:
        _steps._step_discretize_time({"emb": np.zeros((10, 3))}, {}, {}, None, [], 3)
    assert "needs a pseudotime" in str(e.value)


def test_discretize_time_refuses_to_overwrite_real_time_labels():
    """A derived approximation must never replace real per-cell time data — the same
    precedence `runner._ml_lightning` already gives real labels over a computed pseudotime."""
    state = {"pseudotime": np.linspace(0, 1, 10), "labels": ["day0"] * 5 + ["day3"] * 5,
             "label_kind": "time"}
    with pytest.raises(_StepSkipped) as e:
        _steps._step_discretize_time(state, {}, {}, None, [], 3)
    assert "already present" in str(e.value)
    assert list(state["labels"]) == ["day0"] * 5 + ["day3"] * 5, "must not have been touched"


def test_discretize_time_plots_the_embedding_coloured_by_its_own_bins(tmp_path):
    """The picture of what MIOFlow is actually about to train against — bins, not `dpt`'s
    continuous pseudotime (`dpt.png` already draws that one). Coloured with a qualitative
    cmap and NO colorbar: the colour is a group index, not a magnitude a scale bar can read."""
    pytest.importorskip("matplotlib")
    from manyruns import figspec

    _scanpy()
    Z, t = _ribbon(n=80)
    state, g, plots = {"emb": Z}, {}, []
    _steps._step_dpt(state, g, {}, tmp_path, plots, 3)
    _steps._step_discretize_time(state, g, {"n_timepoints": 4}, tmp_path, plots, 3)
    png = tmp_path / "plots" / "discretize_time.png"
    assert png.exists()
    assert any("discretize_time.png" in p for p in plots)
    spec = figspec.load(png)
    assert spec["cmap"] == "tab20"
    assert bool(spec["labeled"]) is False, "a group index must not draw a colorbar"


def test_discretize_time_puts_unreachable_cells_in_the_last_bin():
    """`dpt` marks a cell with no path to the root as NaN (`_step_dpt`'s own diagnostic). A
    bin boundary can't place a NaN, so it must land somewhere accountable rather than vanish
    from MIOFlow's data — the last bin, counted, not silently dropped."""
    pt = np.array([0.0, 0.25, 0.5, 0.75, 1.0, np.nan, np.nan])
    state, g = {"pseudotime": pt}, {}
    _steps._step_discretize_time(state, g, {"n_timepoints": 5}, None, [], 3)
    assert len(state["labels"]) == 7
    assert g["discretize_time.unreachable"] == 2
    last_bin = state["labels"][-3]  # the finite pt=1.0 cell, also the last bin
    assert state["labels"][-1] == last_bin and state["labels"][-2] == last_bin


# ── 5. simplex ───────────────────────────────────────────────────────────────
def test_simplex_refuses_an_embedding_that_is_not_a_simplex():
    """The check that stops this from emitting an "archetype purity" for any 3-column PHATE
    embedding sitting in `state["emb"]`. Measured through the real engine: `pca → simplex` on
    a 400×3 fixture reports min weight -0.521 and row sums -0.622..0.543, and NO numbers."""
    rng = np.random.default_rng(0)
    g = {}
    _steps._step_simplex({"emb": rng.normal(size=(50, 3))}, g, {}, None, [], 3)
    assert g["simplex_purity"] is None
    assert "not a simplex coordinate system" in g["simplex_purity_note"]
    assert not [k for k in g if k.startswith("simplex_") and not k.endswith(("note", "purity"))]


def test_simplex_reads_weights_against_the_floor_that_makes_them_readable():
    """Purity is bounded [1/k, 1] by definition: 1 = every cell on a vertex, 1/k = every cell
    at the barycentre. 0.4 is near-perfect at k=2 and near-nothing at k=8, so the floor is
    emitted beside it — the number is uninterpretable alone and there is no null that could
    interpret it (see the module header)."""
    rng = np.random.default_rng(0)
    sharp, soft = {}, {}
    _steps._step_simplex({"emb": rng.dirichlet([0.1] * 3, 500)}, sharp, {}, None, [], 3)
    _steps._step_simplex({"emb": rng.dirichlet([5.0] * 3, 500)}, soft, {}, None, [], 3)
    assert sharp["simplex_k"] == soft["simplex_k"] == 3
    assert sharp["simplex_purity_floor"] == soft["simplex_purity_floor"] == pytest.approx(1 / 3, abs=1e-4)
    assert sharp["simplex_purity"] > 0.8 > soft["simplex_purity"]
    assert sharp["simplex_at_vertex"] > 0.5
    assert soft["simplex_at_vertex"] == 0.0


def test_simplex_names_archetypes_nobody_uses():
    """An archetype that is nobody's nearest is a vertex the data does not use — the signature
    of a `k` larger than the structure supports, and the one thing purity cannot say."""
    rng = np.random.default_rng(0)
    W = np.c_[rng.dirichlet([0.2] * 4, 500), np.zeros((500, 2))]
    g = {}
    _steps._step_simplex({"emb": W}, g, {}, None, [], 3)
    assert g["simplex_k"] == 6
    assert g["simplex_unused"] == 2
    assert g["simplex_min_occupancy"] == 0.0


# ── 6. qc ────────────────────────────────────────────────────────────────────
def test_qc_reports_and_removes_nothing():
    """THE CONTRACT, and the mutation check for it. §5.5: a recipe's output type is a
    g-vector, a filter's is a smaller dataset. Make `_step_qc` drop the cells it counts and
    this fails on the shape, the object identity and `qc_removed` at once."""
    X, genes = _counts(n=40, p=12)
    X[0, :] = 0                                   # a cell that any threshold would drop
    state = {"counts": X, "genes": genes, "X": np.zeros((40, 3))}
    before, before_X = X.copy(), state["X"]
    g = {}
    _steps._step_qc(state, g, {"min_genes": 5, "min_cells": 3}, None, [], 3)
    assert g["qc_would_drop_cells"] == 1, "the empty cell was not counted"
    assert g["qc_removed"] == 0
    assert state["counts"].shape == (40, 12) and state["X"] is before_X
    assert np.array_equal(state["counts"], before), "qc rewrote the counts matrix"


def test_qc_counts_what_the_declared_thresholds_would_remove():
    X, genes = _counts(n=40, p=12)
    X[:, 0] = 0
    X[:, 1] = 0
    X[:2, 1] = 7                                  # a gene present in exactly 2 cells
    g = {}
    _steps._step_qc({"counts": X, "genes": genes}, g, {"min_cells": 3}, None, [], 3)
    assert g["qc_n_cells"] == 40 and g["qc_n_genes"] == 12
    assert g["qc_would_drop_genes"] == 2, "an all-zero gene and a 2-cell gene"
    assert g["qc.min_cells"] == 3, "the threshold must be in the record beside the count"


def test_qc_keeps_a_declared_zero_threshold():
    """`params.get(k) or default` would turn `min_genes: 0` — a meaningful declaration —
    back into 200. `_param` exists for this and this is what pins it."""
    X, genes = _counts()
    g = {}
    _steps._step_qc({"counts": X, "genes": genes}, g, {"min_genes": 0}, None, [], 3)
    assert g["qc.min_genes"] == 0
    assert g["qc_would_drop_cells"] == 0


def test_qc_refuses_to_call_an_unnamed_mitochondrial_fraction_zero():
    """0 % would be a claim about biology. "Not named that way" is a claim about the file, and
    it is the true one — pbmc3k names 13 genes `MT-*` (measured), a mouse dataset names them
    `mt-*`, and a matrix with Ensembl IDs names none of them anything."""
    X, genes = _counts()
    g = {}
    _steps._step_qc({"counts": X, "genes": genes}, g, {}, None, [], 3)
    assert g["qc_n_mito_genes"] == 0
    assert g["qc_pct_mito_median"] is None
    assert "not the same as it being zero" in g["qc_pct_mito_median_note"]


def test_qc_measures_the_mitochondrial_fraction_when_the_genes_are_named():
    X, genes = _counts(n=10, p=6)
    X[:] = 10.0
    genes = np.array(["MT-CO1", "MT-ND1", "ACTB", "GAPDH", "CD3D", "LYZ"])
    g = {}
    _steps._step_qc({"counts": X, "genes": genes}, g, {}, None, [], 3)
    assert g["qc_n_mito_genes"] == 2
    assert g["qc_pct_mito_median"] == pytest.approx(100 * 2 / 6)


def test_qc_says_why_without_a_counts_matrix():
    g = {}
    _steps._step_qc({"counts": None, "genes": None, "X": np.zeros((5, 3))}, g, {}, None, [], 3)
    assert g["qc_n_cells"] is None
    assert "state['counts']" in g["qc_n_cells_note"]


def test_qc_never_densifies_the_whole_matrix():
    """pbmc3k is 18.3 MB as CSR and 353.6 MB dense (measured). Every reduction here is over
    the sparse object; only `looks_like_counts` densifies, and only its 100-row sample. A spy
    on `toarray` catches a regression that would otherwise show up as a laptop swapping."""
    sparse = pytest.importorskip("scipy.sparse")
    X, genes = _counts(n=300, p=40)
    seen: list = []

    class Spy(sparse.csr_matrix):
        def toarray(self, *a, **k):
            seen.append(self.shape)
            return super().toarray(*a, **k)

    g = {}
    _steps._step_qc({"counts": Spy(X), "genes": genes}, g, {}, None, [], 3)
    assert g["qc_n_cells"] == 300, "the sparse path did not produce the metrics"
    assert all(shape[0] <= 100 for shape in seen), f"densified {seen}"


def test_qc_reports_its_own_gene_count_and_never_rewrites_a_core_key():
    """RE-FRAMED: the disagreement this was written around is gone, the mutation check is not.
    `qc_n_genes` existed because `GVECTOR_CORE` defines `n_features` as the columns of the
    input data while pbmc3k reported 50 — `loading._anndata_matrix` PCA'd the input before the
    runner saw it — and correcting a core key in place breaks comparability across every stored
    run. The cutover took that break deliberately in `runner.open_gvector` instead: `n_features`
    is the true width now (32,738 on pbmc3k, where it was 50).

    Two honest widths remain, so the write must still land in `qc`'s own key: `n_features` is
    the shape of the data that ENTERED the run, and `qc_n_genes` is the width of
    `state["counts"]` where qc reads it — which a `prep` step that drops genes (`filter_genes`,
    `hvg`) makes genuinely different. A step correcting a core key would make the opening size
    unrecoverable, which is the failure `open_gvector` exists to prevent."""
    X, genes = _counts(n=20, p=37)
    g = {"n_features": 5}              # a width qc did not compute and must not overwrite
    _steps._step_qc({"counts": X, "genes": genes}, g, {}, None, [], 3)
    assert g["qc_n_genes"] == 37
    assert g["n_features"] == 5, "qc rewrote a core g-vector key"


# ── the rules that hold across all five ──────────────────────────────────────
def test_none_of_the_new_readouts_is_narrated_as_a_finding():
    """Five of the ten recipes produce statistics computed from the embedding alone, which
    `vocab.NULL_KIND` says need a `data` null that nothing implements. They ship as
    MEASUREMENTS: `narrate.narrate` must reach its "no groups or time labels" fallback rather
    than making a claim out of any key below."""
    from manyruns import narrate

    g = {"cluster_quality_n": 7, "cluster_quality_silhouette": 0.42,
         "rank_genes_top": ["0: CD3D"], "dpt_range": [0.0, 1.0], "dpt_root": 3,
         "simplex_purity": 0.8, "qc_n_cells": 2700, "qc_would_drop_genes": 19024}
    said = narrate.narrate({"g_vector": g, "final_dim": 3, "recipe": {"steps": []}})
    assert "Mapped the structure" in said
    for word in ("cluster", "gene", "order", "archetype", "trajectory", "significant"):
        assert word not in said.lower(), f"the narrator promoted a measurement: {said!r}"


def test_no_new_readout_claims_a_null_it_does_not_have():
    """`NULL_KIND` is the closed list of which readout gets which null, and it refuses to
    guess. A `labels` null here would CERTIFY these: measured this session on pure iid
    gaussian noise, KMeans + silhouette against 199 label permutations returns p_emp = 0.0050
    — the 1/(R+1) floor — at n=600/d=10/k=8, n=600/d=3/k=8 and n=200/d=10/k=4 alike. That is
    the granger failure mode, so the five stay out of the table until a `data` null exists."""
    from manyruns.vocab import NULL_KIND

    for name in ("cluster_quality", "rank_genes", "dpt", "simplex", "qc"):
        assert name not in NULL_KIND, (
            f"{name} was given a null; the only one implemented is `labels`, which certifies "
            "a partition of pure noise at the p-floor")


def test_every_new_readout_is_dispatchable_on_both_step_loops():
    """`dispatchable` derives what an engine can be OFFERED from these tables, so a readout
    that is not in `_ANALYSIS_STEPS` is a step the menu never shows."""
    from manyruns.pipeline.runner import dispatchable

    for name in ("cluster_quality", "rank_genes", "dpt", "simplex", "qc"):
        assert dispatchable("_inproc", name, "analysis"), name
        assert dispatchable("manylatents", name, "analysis"), name


def test_a_typo_in_an_analysis_step_name_is_still_refused():
    """The other half of the table above: adding five names must not have made the group
    accept anything. `dispatchable` answers False and the runner declines at run time."""
    from manyruns.pipeline.runner import dispatchable

    assert not dispatchable("_inproc", "cluster_qualtiy", "analysis")


# ── the precondition calculus: prune, do not degrade ─────────────────────────
def _recipe(*names):
    groups = {"pca": "latent", "leiden": "latent", "umap": "latent", "diffusionmap": "latent",
              "aa": "latent"}
    return {"name": "probe",
            "steps": [{"name": n, "group": groups.get(n, "analysis")} for n in names]}


def test_a_cluster_readout_without_a_clustering_step_is_illegal_on_every_shape():
    """The point of registering `clusters` at all. A readout of a partition that no step
    produced must be pruned at PLAN time — `experiment.plan`, `shell._explore`'s menu and
    `app`'s coverage table all ask this because they cannot afford to run it to find out."""
    from manyruns.vocab import SHAPES, unmet

    carries_genes = {"provides": ["genes"]}         # what configs/dataset/pbmc3k.yaml declares
    for shape in SHAPES:
        assert unmet(_recipe("pca", "cluster_quality"), shape) == {"clusters"}
        # WAS {"clusters"}. `STEP_NEEDS["rank_genes"]` gained `genes` at the cutover — now that
        # `loading._anndata_matrix` hands the gene axis over instead of PCA-ing it away, the
        # need is satisfiable and so is declarable — so a dataset with no gene axis is short
        # two facts, not one.
        assert unmet(_recipe("pca", "rank_genes"), shape) == {"clusters", "genes"}
        # Supply the axis and the missing PARTITION is still the refusal, which is the property
        # this test was written for and the one the second need must not mask.
        assert unmet(_recipe("pca", "rank_genes"), shape, handle=carries_genes) == {"clusters"}


def test_leiden_makes_both_cluster_readouts_legal_everywhere():
    """"Everywhere" quantifies over SHAPES, which is the axis `unmet` prunes on, and that is
    still exactly what leiden buys: it produces `clusters`, so neither readout is pruned for
    want of a partition on any of the six. The cutover added a SECOND, orthogonal axis for
    `rank_genes` only — `genes` is a DATASET fact (`vocab.DATASET_FACTS`), supplied by a
    handle's `provides:` and by no shape at all, so it is passed in here rather than earned by
    a step. `cluster_quality` reads coordinates, not gene names, and is untouched by it."""
    from manyruns.vocab import SHAPES, unmet

    carries_genes = {"provides": ["genes"]}         # what configs/dataset/pbmc3k.yaml declares
    for shape in SHAPES:
        assert unmet(_recipe("pca", "leiden", "cluster_quality"), shape) == frozenset()
        # WAS frozenset() on every shape. `rank_genes` declares `genes` since the cutover and
        # no SHAPE provides it, so leiden alone no longer makes this one legal.
        assert unmet(_recipe("pca", "leiden", "rank_genes"), shape) == {"genes"}
        assert unmet(_recipe("pca", "leiden", "rank_genes"), shape,
                     handle=carries_genes) == frozenset()


def test_a_caller_holding_a_partition_is_not_refused():
    """The same structural-not-canonical rule `embedding` has (`unmet`'s docstring): the test
    is whether the object exists, never whether a blessed step produced it."""
    from manyruns.vocab import unmet

    assert unmet(_recipe("cluster_quality"), "manifold", provided={"clusters"}) == frozenset()


def test_the_gene_axis_is_a_declared_need_with_a_declared_provider():
    """RE-FRAMED AT THE CUTOVER, because the omission it pinned is over. It used to assert that
    `rank_genes` and `qc` declare NO `genes` need: they require the gene axis as genuinely as
    ever, but nothing could supply one — every scRNA load went through
    `loading._anndata_matrix`'s undeclared PCA and reached the runner 50 columns wide with no
    names — so declaring the need would have returned `{genes}` on all six shapes and pruned
    the two recipes that run on the one real dataset in the repo.

    Both halves have landed. `loading` hands the frame over unreduced, and a dataset now STATES
    what its bytes carry (`handle: {provides: [genes]}`) instead of being sniffed by suffix, so
    `genes` is a DATASET fact (`vocab.DATASET_FACTS`) rather than the shape fact
    `SHAPE_PROVIDES` could never key. The property under test is the one that always mattered
    and it is unchanged: a precondition must fire on data that cannot answer and must NOT fire
    on data that can — a precondition firing on correct data is the `lid <= final_dim` bound
    again. Measured on the bundled catalog: 14 datasets, exactly one carries the axis, so
    `markers` and `qc` are pruned on the 13 synthetics and legal on pbmc3k (where `markers`
    reports 7 groups over 32,738 genes)."""
    from manyruns import catalog
    from manyruns.vocab import SHAPES, dataset_provides, unmet

    handle = catalog.load_dataset("pbmc3k")["handle"]
    for shape in SHAPES:
        # The refusal, which is new and is the point: pruned at PLAN time rather than running
        # and reporting "no genes to rank" — a sentence a swissroll and a real dataset whose
        # names failed to load produce alike.
        assert unmet(_recipe("pca", "leiden", "rank_genes"), shape) == {"genes"}
        assert unmet(_recipe("qc"), shape) == {"genes"}
        # The half that must NOT fire: the declaration clears both on every shape.
        assert unmet(_recipe("pca", "leiden", "rank_genes"), shape, handle=handle) == frozenset()
        assert unmet(_recipe("qc"), shape, handle=handle) == frozenset()

    # The supply side, pinned beside the demand it exists for: delete `provides: [genes]` from
    # the one dataset that declares it and both recipes above are legal on nothing this repo
    # ships, which is the state this test used to describe.
    declared = {name: catalog.load_dataset(name) for name in catalog.discover_datasets()}
    carriers = [name for name, ds in declared.items()
                if "genes" in dataset_provides(ds.get("shape", "unknown"), ds.get("handle"))]
    assert len(declared) == 14, "13 synthetics + pbmc3k, the first bundled dataset with genes"
    assert carriers == ["pbmc3k"], f"exactly one bundled dataset carries a gene axis: {carriers}"


def test_the_bundled_recipes_keep_their_legality():
    """`STEP_PRODUCES` is additive — a name absent from it contributes nothing — so no recipe
    that was legal before it existed may have been pruned by it. The golden table in
    `test_order_calculus.py` covers the same ground recipe by recipe; this is the cheap
    restatement that travels with the change that could break it.

    ONE PRUNING IS NOW DELIBERATE and is stated as data below rather than left to the default:
    `markers`, `qc` and `preprocess` are the bundled recipes that read gene names, and their
    steps declare `genes`, so on a dataset carrying no gene axis they are refused at plan time.
    Every one of them is legal again on a dataset that declares the axis, which the second
    subtraction asserts — a refusal that survived the supply landing would be the
    precondition-fires-on-correct-data failure, not the calculus working."""
    from manyruns import catalog
    from manyruns.vocab import SHAPES, unmet

    carries_genes = {"provides": ["genes"]}         # what configs/dataset/pbmc3k.yaml declares
    for name in catalog.discover_recipes():
        recipe = catalog.load_recipe(name)
        for shape in SHAPES:
            no_conditions = name == "contrast" and shape != "case-control"
            # WAS frozenset() for `markers` and `qc` on every shape: they used to declare no
            # `genes` need because nothing could supply one. Measured now: `{genes}` on all six.
            # `preprocess` joined them the day it landed rather than moving — `filter_genes` and
            # `filter_mito` have declared the need since Task 6, so the recipe was born refused
            # everywhere the axis is absent, which is the whole reason it is a recipe and not a
            # step: a step would have carried that refusal into every recipe that named it.
            expected = frozenset({"conditions"}) if no_conditions else (
                frozenset({"genes"}) if name in ("markers", "qc", "preprocess")
                else frozenset())
            assert unmet(recipe, shape) == expected, (name, shape)
            with_axis = frozenset({"conditions"}) if no_conditions else frozenset()
            assert unmet(recipe, shape, handle=carries_genes) == with_axis, (name, shape)
