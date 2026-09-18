"""In-process step implementations — the substrate's stand-ins and the analysis readouts.

Two tables live here:
  `_INPROC_STEPS`     — latent/lightning steps computed with public libraries, so the product
                      runs with no private stack;
  `_ANALYSIS_STEPS` — manyruns's own readouts. These are NOT engine algorithms, which is
                      why `analysis` stays a product-owned group on every engine. It held two
                      entries, both requiring a condition label that zero bundled datasets
                      carry; the five added below are the readouts the top-5 recipes need.

Both are plain dicts, so callers (and tests) can substitute a step by key.
"""
from __future__ import annotations

from pathlib import Path  # noqa: F401 - step signatures take out_dir
from typing import Any  # noqa: F401

from manyruns.pipeline import io as _io
from manyruns.pipeline import null as _null

class _StepSkipped(Exception):
    """An executor declining a step because a precondition isn't met — not an error.

    Defined HERE rather than in `runner`, which is where the step loop that catches it lives,
    purely because of the import graph: `runner` imports this module at module scope, so the
    reverse import cannot exist. `runner` re-exports it, so its documented name is unchanged.
    """


class _StepUnsupported(_StepSkipped):
    """A step NOTHING here can run — a capability gap, not a judgment about the data.

    THE DISTINCTION, and why it needed a second class. Both land as `outcome="skipped"`,
    because the loop must not die and the record vocabulary stays four wide
    (`watch.OUTCOMES`). But they are not the same event and the RUN-level `ok` treats them
    differently (manyruns#66):

      * `_StepSkipped` — the executor RAN and judged. `normalize` on a matrix that is not
        counts, `discretize_time` on a pseudotime with no finite values. The record says what
        it decided, the other steps' numbers are unaffected, and the run stays `ok`.
      * `_StepUnsupported` — no executor exists. A typo'd step name, a group this engine has
        no table for, a step declared before its maths was written. The run is NOT `ok`: the
        record does not describe the analysis that was asked for, so comparing it against
        another run of "the same recipe" is invalid, and that is a trust question rather than
        a completeness one.

    Subclassing rather than sitting beside it, so every `except _StepSkipped` already written
    keeps catching both. The only code that has to know the difference is the one place that
    sets the flag — a new raise site that forgets this class degrades into the benign reading,
    which is the safe direction for the LOOP and the reason `ok`'s computation reads the flag
    rather than re-deriving intent from a detail string.
    """


def _require_embedding(state, step: str) -> None:
    """Refuse a trajectory step that has no coordinates to run on. ONE home for that rule.

    Both engines need it and only one had it. `_ml_lightning` (manylatents) raised; the real
    engine's `_step_mioflow` fell back to `state["X"]` and ran the ordering on the RAW input.
    Measured on a `mioflow -> phate` recipe, 120x8 gaussian noise, in-process: every step
    reported `outcome=ok`, the run reported `ok=True`, and the g-vector carried
    `pseudotime_range = [0.0, 1.0]` — a full-range trajectory over data no embedding had
    touched, indistinguishable in the record from a legitimate one. `vocab.unmet` returns
    `frozenset()` for that recipe (it unions `STEP_NEEDS` and does not model order), so the
    engine's own refusal is the only thing standing between a backwards recipe and a
    published number.
    """
    if state.get("emb") is None:
        # Names the missing OBJECT, not a missing step. The rule is that a trajectory needs
        # coordinates to run on; it is indifferent to where they came from. Coordinates
        # computed by another tool, loaded from a file, or carried in as a graph with
        # compatible dimensions all satisfy it — `run_inproc(embedding=...)` is the seam.
        # The earlier wording ("needs a preceding latent step") described the conventional
        # route as though it were the requirement, which is the kind of rule that prunes a
        # composition for being unusual rather than for being ill-typed.
        raise _StepSkipped(
            f"{step} needs an embedding to run on; none was supplied and no step produced one"
        )


#: PHATE knobs a recipe may set. Closed on purpose: an unknown key is a typo, and silently
#: accepting it is how `knn: 15` reads as "took effect" when nothing received it.
_PHATE_PARAMS = ("knn", "decay", "t", "gamma", "n_pca", "n_landmark", "mds", "mds_solver")


def _step_phate(state, g, params, out_dir, plots, target_dim) -> None:
    """Real PHATE embedding via the public `phate` package.

    Honours the recipe's parameters and the run's seed (issue #29). Both were dropped: this
    read only `n_components` and passed no `random_state`, so on the DEFAULT engine every
    knob a recipe declared was silently ignored and no two runs were alike. That made the
    one thing the product exists to support — change a setting, see what moves — impossible
    to express, and made every stackless result unreproducible."""
    import phate

    k = int(params.get("n_components") or target_dim)
    kwargs = {n: params[n] for n in _PHATE_PARAMS if params.get(n) is not None}
    seed = state.get("seed")
    if seed is not None:
        kwargs["random_state"] = int(seed)
    op = phate.PHATE(n_components=k, verbose=False, **kwargs)
    emb = op.fit_transform(state["X"])
    state["emb"] = emb
    from manyruns.measurements import annotate

    g["phate_dims"] = int(emb.shape[1])
    annotate(g, "phate_dims", kind="setting", stage="phate configuration")
    # what the fit actually used, so the record shows the effective setting rather than only
    # what the recipe asked for
    for name in _PHATE_PARAMS:
        value = getattr(op, name, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            g[f"phate.{name}"] = value
            annotate(g, f"phate.{name}", kind="setting", stage="phate configuration")
    _io._save_scatter(emb, None, out_dir, "phate.png", plots, title="PHATE embedding")


def _extremal_root(Z) -> int:
    """The cell an ordering starts from — argmin of the coordinate sum. ONE home for the rule.

    Extracted so `_diffusion_pseudotime` and `_step_dpt` cannot pick different roots for the
    same embedding: they are two orderings whose AGREEMENT `_step_dpt` reports, and an
    agreement between orderings rooted at different cells measures the roots, not the method.

    The rule is a choice, not an inference, and the choice is the point of Weinreb et al.
    (PNAS 115:E2467, cited in `cflows.yaml`): direction is not identifiable from a static
    snapshot. An extremal point is reproducible and arbitrary, which is why `_step_dpt`
    records which cell it landed on rather than leaving the reader to re-derive it."""
    import numpy as np

    return int(np.argmin(np.asarray(Z).sum(axis=1)))


def _diffusion_pseudotime(Z, n_neighbors: int, info: "dict | None" = None):
    """Per-sample pseudotime: normalized geodesic distance from an extremal root on a
    kNN graph. Shared by the `real` engine's trajectory step and the MIOFlow time label.

    `info`, when a dict is passed, receives the diagnostics this function otherwise HIDES:
    `root` and `unreachable` (cells with no path to the root, which the return value silently
    fills with the maximum finite distance). A cell that is unreachable and a cell that is
    genuinely farthest away are indistinguishable in the returned vector — the caller has to
    be able to tell them apart. An out-parameter rather than a second return value because
    two other call sites (`_step_mioflow`, `mioflow._run_mioflow_experiment`) unpack this as
    a bare array, and widening the return type for a diagnostic is how a signature change
    reaches files that did not ask for one."""
    import numpy as np
    from scipy.sparse.csgraph import shortest_path
    from sklearn.neighbors import kneighbors_graph

    graph = kneighbors_graph(Z, n_neighbors=n_neighbors, mode="distance").tocsr()
    # scipy csgraph wants int32 index arrays; on some platforms kneighbors_graph yields
    # int64 → "Buffer dtype mismatch, expected 'const int' but got 'long'".
    graph.indices = graph.indices.astype(np.int32, copy=False)
    graph.indptr = graph.indptr.astype(np.int32, copy=False)
    root = _extremal_root(Z)  # an extremal point as the trajectory root
    dist = shortest_path(graph, method="D", directed=False, indices=root)
    finite = dist[np.isfinite(dist)]
    if info is not None:
        info["root"] = root
        info["unreachable"] = int((~np.isfinite(dist)).sum())
    pt = np.where(np.isfinite(dist), dist, finite.max() if finite.size else 0.0)
    return (pt - pt.min()) / (np.ptp(pt) or 1.0)  # normalize 0..1 (np.ptp: NumPy 2.0-safe)


def _step_mioflow(state, g, params, out_dir, plots, target_dim) -> None:
    """Trajectory step — diffusion pseudotime (a real MIOFlow stand-in for the in-process loop).

    The manylatents engine runs the *actual* MIOFlow (see _run_mioflow_experiment); this
    stand-in keeps the in-process loop dependency-light (no torch)."""
    _require_embedding(state, "mioflow")
    Z = state["emb"]
    n_neighbors = int(params.get("n_neighbors") or min(15, max(2, Z.shape[0] - 1)))
    pt = _diffusion_pseudotime(Z, n_neighbors)
    state["pseudotime"] = pt
    g["pseudotime_range"] = [float(pt.min()), float(pt.max())]
    # unconditional now — `_require_embedding` above has already refused the None case
    _io._save_scatter(Z, pt, out_dir, "trajectory.png", plots, title="pseudotime")


def _step_separation(state, g, params, out_dir, plots, target_dim) -> None:
    """Condition separation — how distinctly do the conditions separate in the manifold?

    Silhouette of the embedding grouped by the condition label (``sklearn``). Degrades gracefully (records a note, no
    error) when there's no embedding, no condition labels, or <2 groups — so a contrast run
    on data that lacks a condition column still completes.

    AN ALLOWLIST ON THE LABEL KIND, and it is the same rule — and the same reasoning — as
    `runner.py`'s trajectory guard: only `label_kind == "condition"` is a condition axis, and
    everything else is no condition axis at all. `state["labels"]` is one channel carrying
    three kinds (`time`, `condition`, `group`), and this step read it unconditionally, so a
    `group` axis (a cell type, a cluster, a branch — `data/tree8.h5ad` ships one) scored a real
    silhouette and emitted `separation_*` keys asserting a case/control contrast that does not
    exist in the data. A denylist (`!= "group"`) would be the same defect one kind later; the
    refusal below is what group-only data got before the kind existed, and is what it gets
    again."""
    import numpy as np

    Z = state.get("emb")
    if Z is None:
        Z = state.get("X")
    labels = state.get("labels") if state.get("label_kind") == "condition" else None
    if Z is None:
        g["separation"], g["separation_note"] = None, "no embedding to score"
        return
    if labels is None:
        g["separation"] = None
        g["separation_note"] = "no condition labels — needs ≥2 condition groups in obs"
        return
    labels = np.asarray(labels)
    n_groups = len({str(v) for v in labels.tolist()})
    if n_groups < 2:
        g["separation"], g["separation_note"] = None, f"needs ≥2 condition groups, got {n_groups}"
        return
    from sklearn.metrics import silhouette_score

    Z = np.asarray(Z)

    def stat(lab) -> float | None:
        lab = np.asarray(lab)
        if len({str(v) for v in lab.tolist()}) < 2:
            return None
        return float(silhouette_score(Z, lab))

    g["separation_silhouette"] = stat(labels)
    g["separation_n_groups"] = n_groups
    # A silhouette is bounded [-1, 1] but its null is NOT centred on zero: any partition of a
    # finite cloud scores something, and the value drifts with n and group balance. Rank it.
    _null.record(g, "separation_silhouette",
                 _null.rank_against_null(stat, labels, seed=int(params.get("seed") or 0)))


def _step_composition(state, g, params, out_dir, plots, target_dim) -> None:
    """Population composition shift — the case/control backmap to *populations*.

    Cluster the embedding, then measure how each cluster's abundance differs across the
    condition labels (which populations are enriched in which condition). sklearn KMeans;
    degrades gracefully (a note, not an error) with no embedding / no labels / <2 conditions.
    Gene-level DE (the backmap to *genes*) is the next step — it needs the original expression
    matrix + gene names threaded through, which this pipeline doesn't yet carry.

    THE SAME ALLOWLIST ON THE LABEL KIND as `_step_separation` — see there for why it is an
    allowlist and not a denylist. This step is worse than that one when it is wrong: it names
    the two groups in `composition_between`, so a `group` axis produced a sentence like
    "branch-1 vs branch-2" in a field a reader is entitled to read as case versus control."""
    import itertools

    import numpy as np

    Z = state.get("emb")
    raw = state.get("labels") if state.get("label_kind") == "condition" else None
    if Z is None:
        g["composition"], g["composition_note"] = None, "no embedding to cluster"
        return
    if raw is None:
        g["composition"], g["composition_note"] = None, "no condition labels"
        return
    labels = np.asarray([str(v) for v in np.asarray(raw).tolist()])
    conds = sorted(set(labels.tolist()))
    if len(conds) < 2:
        g["composition"], g["composition_note"] = None, f"needs ≥2 conditions, got {len(conds)}"
        return

    from sklearn.cluster import KMeans

    Z = np.asarray(Z)
    k = int(params.get("k") or min(8, max(2, Z.shape[0] // 50)))
    # Clustered ONCE, then held fixed across every permutation below. KMeans never sees a
    # label, so reusing the partition is not double-dipping — it is holding constant the one
    # thing the null is not about. Re-clustering per permutation would also test KMeans'
    # seed-sensitivity, which is a different question.
    cl = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)

    def abundances(lab) -> dict:
        """Per-condition cluster abundance, Haldane–Anscombe corrected.

        The pseudocount lives in COUNT space (+0.5 per cell, per Haldane–Anscombe), not in
        fraction space. The previous form added 1e-3 to the *fractions*, which made the
        reported fold a function of that constant whenever a cluster was empty in one
        condition — and a cluster empty in one condition is the ORDINARY case as soon as the
        conditions separate at all, i.e. exactly when the finding is real. Measured on one
        fixed 600×10 two-condition cloud: sweeping the constant 1e-1 → 1e-5 moved the
        narrated claim across 4× / 32× / 314× / 3134× / 31334× with the data untouched.
        In count space the bound is set by n instead, which is the actual evidence: zero out
        of 300 cells is stronger evidence of depletion than zero out of 30, and the fraction
        form scored those identically.
        """
        lab = np.asarray(lab)
        out = {}
        for c in conds:
            counts = np.bincount(cl[lab == c], minlength=k).astype(float) + 0.5
            out[c] = counts / counts.sum()
        return out

    def largest_shift(lab):
        fr = abundances(lab)
        best, at, pair = 0.0, -1, None
        for a, b in itertools.combinations(conds, 2):
            fold = np.abs(np.log2(fr[a] / fr[b]))
            i = int(np.argmax(fold))
            if fold[i] > best:
                best, at, pair = float(fold[i]), i, f"{a} vs {b}"
        return best, at, pair

    def stat(lab) -> float | None:
        return largest_shift(lab)[0]

    max_shift, at_cluster, between = largest_shift(labels)
    g["composition_n_clusters"] = k
    g["composition_max_log2_shift"] = round(max_shift, 3)
    g["composition_shift_cluster"] = at_cluster
    g["composition_between"] = between
    # The magnitude alone is uninterpretable: this is a MAXIMUM over k clusters × condition
    # pairs, so its null expectation is positive by construction and grows as counts shrink.
    # Measured: null mean 0.578 at n=600/k=8 but 5.589 at n=60/k=8, where the
    # real value landed BELOW its own null and the product still printed "~48× more abundant".
    # The rank is what survives that; the narration gates on it.
    _null.record(g, "composition_max_log2_shift",
                 _null.rank_against_null(stat, labels, seed=int(params.get("seed") or 0)))
    # Whether the winning cluster is actually absent in one condition — the narration needs
    # this to say "absent" rather than quoting a fold ratio that is really a bound.
    if at_cluster >= 0 and between:
        a, b = between.split(" vs ")
        raw_counts = {c: np.bincount(cl[labels == c], minlength=k) for c in (a, b)}
        g["composition_absent_in_one"] = bool(
            raw_counts[a][at_cluster] == 0 or raw_counts[b][at_cluster] == 0)


# ── the readouts a picture is missing ────────────────────────────────────────
#
# Everything below this line exists because of one measured fact: the `analysis` group had two
# entries, both requiring a CONDITION label, and zero of the 13 bundled datasets declare
# `shape: case-control`. Every other recipe therefore ended at an embedding — a picture with no
# readout. These are the readouts for the recipe ladder.
#
# NONE of them is a finding, and that is a rule rather than a caution. Each is computed from the
# embedding, or from a partition of it, so `vocab.NULL_KIND` requires a `data` null — synthesise
# null data, re-run the WHOLE pipeline — and nothing implements one. The cheap `labels` null is
# not merely unavailable here, it is dangerous: measured this session on pure iid gaussian noise
# (no clusters exist), KMeans + silhouette scored against 199 label permutations returns
# p_emp = 0.0050 — the 1/(R+1) FLOOR — at n=600/d=10/k=8, n=600/d=3/k=8 and n=200/d=10/k=4
# alike. A `labels` null would certify a partition of noise in every configuration tried. That
# is the granger failure mode exactly, so these ship as MEASUREMENTS:
# absent from `NULL_KIND`, and `narrate.backmap` reads none of their keys.


def _param(params: dict, key: str, default):
    """A declared parameter, where 0 and False are legal values.

    `params.get(k) or default` — the idiom used above for PHATE's knobs, where every knob is a
    positive number — silently replaces a declared `0` with the default. `qc`'s thresholds are
    the case that breaks: `min_genes: 0` is a meaningful declaration ("report, drop nothing")
    and would come back as 200."""
    value = (params or {}).get(key)
    return default if value is None else value


#: How `_partition` found the labels, as the string it reports in the g-vector.
_FROM_CLUSTERS = "state['clusters']"
_FROM_EMB = "state['emb'] — a 1-column integral vector, which is leiden's output shape"


def _partition(state) -> tuple:
    """``(labels, where)`` — the cluster assignment to report on, or ``(None, why)``.

    TWO sources, in this order, and the order is the design:

    1. ``state["clusters"]`` — the slot canonical-recipes.md §5.1 argues for (*"leiden writes
       state['clusters'] rather than overwriting state['emb'] — the cleaner fix"*). **Nothing
       writes it today**: it is absent from `runner._STATE_KEYS`, and `runner._ml_latent`
       assigns `state["emb"] = emb` for every latent step whatever that step computed. It is
       read FIRST so that the day a producer lands, every readout here follows with no edit.
    2. ``state["emb"]`` when it is a single column of integral values — what manylatents'
       leiden actually leaves behind. Measured in this checkout on two gaussian blobs at
       120×8, `api.run(algorithms={'latent': 'leiden'}, seed=0)`: `embeddings` is
       **(120, 1) float32 over {0., 1.}** and `scores` is **empty** — a clustering step whose
       result reaches no record at all.

    Reading a label column out of `emb` is not a design, it is what the state schema forces on
    a readout while (1) has no producer. The test is deliberately narrow — exactly one column,
    every value finite and integral — so ordinary 1-D coordinates are not mistaken for a
    partition. A 1-D embedding whose values happen to be all-integral WOULD be, and that
    residual is the argument for (1) rather than a reason to widen the test."""
    import numpy as np

    raw = state.get("clusters")
    if raw is not None:
        return np.asarray(raw).ravel(), _FROM_CLUSTERS
    emb = state.get("emb")
    if emb is None:
        return None, "no partition in this run — no clustering step has run"
    arr = np.asarray(emb)
    if arr.ndim != 2 or arr.shape[1] != 1:
        width = arr.shape[1] if arr.ndim == 2 else f"{arr.ndim}-D"
        return None, (f"state['emb'] holds {width}-column coordinates, not a label vector "
                      f"— no clustering step has run")
    col = arr[:, 0]
    if not np.all(np.isfinite(col)) or not np.allclose(col, np.round(col)):
        return None, "state['emb'] is a 1-D embedding of non-integral values, not a partition"
    return np.round(col).astype(int), _FROM_EMB


def _step_cluster_quality(state, g, params, out_dir, plots, target_dim) -> None:
    """The clusters, reported — how many, how big, how separated. (`cluster`'s readout.)

    `pca → leiden → umap` runs today on `engine=manylatents` and emits **zero** keys about the
    clusters (measured: leiden's `scores` is empty), so the recipe that answers *"what cell
    types are in here"* previously answered it with a scatter plot and nothing in the record.

    **Silhouette, and where it is computed.** sklearn's silhouette over the partition. The
    coordinate system is `state["emb"]` when the partition came from `state["clusters"]`, and
    `state["X"]` when the partition came from `emb` ITSELF — because in that case leiden has
    overwritten the coordinates it partitioned and they no longer exist anywhere in `state`.
    Those are two different questions ("how separated in the embedding" vs "…in the input"),
    so the space is recorded as `cluster_quality_space` rather than left for a reader to
    assume. This is the concrete cost of §5.1's option (b) and the reason (a) is right.

    **No null, deliberately** — see this section's header: a labels null certifies a partition
    of pure noise at the p-floor in every configuration measured. `cluster_quality_silhouette`
    is a description of THIS partition, and the sizes below are what make it readable: a
    silhouette of 0.5 over 30 clusters of which 22 are singletons is over-partitioning, and
    the number alone cannot say so.

    Modularity is NOT computed although §5.1 lists it. Leiden optimises its objective on ITS
    OWN kNN graph (default `n_neighbors=15`, its own resolution); a modularity computed on a
    graph manyruns rebuilds with manyruns's k is a different number about a different graph,
    and would read in the record as a score of leiden's optimisation. The honest version needs
    the graph leiden used, which no step hands back."""
    import numpy as np

    labels, where = _partition(state)
    if labels is None:
        g["cluster_quality_n"] = None
        g["cluster_quality_n_note"] = where
        return
    uniq, counts = np.unique(labels, return_counts=True)
    k = int(uniq.size)
    g["cluster_quality_source"] = where
    g["cluster_quality_n"] = k
    g["cluster_quality_min_size"] = int(counts.min())
    g["cluster_quality_max_size"] = int(counts.max())
    # Singletons are the over-partitioning signature, and they are invisible in `n` alone.
    g["cluster_quality_singletons"] = int((counts == 1).sum())
    Z = state.get("X") if where == _FROM_EMB else state.get("emb")
    space = "state['X'] (the input — leiden overwrote the coordinates it partitioned)" \
        if where == _FROM_EMB else "state['emb']"
    if Z is None:
        g["cluster_quality_silhouette"] = None
        g["cluster_quality_silhouette_note"] = (
            "no coordinates to score the partition in — leiden's label column replaced the "
            "embedding and this run carries no input matrix (a named engine dataset)")
        return
    if k < 2:
        g["cluster_quality_silhouette"] = None
        g["cluster_quality_silhouette_note"] = f"a silhouette needs ≥2 clusters, got {k}"
        return
    from sklearn.metrics import silhouette_score

    g["cluster_quality_space"] = space
    g["cluster_quality_silhouette"] = float(silhouette_score(np.asarray(Z), labels))


def _step_rank_genes(state, g, params, out_dir, plots, target_dim) -> None:
    """The genes that most distinguish each group — `markers`' readout. **Not a test.**

    Scanpy's `rank_genes_groups` (Wilcoxon) over the gene axis, grouped by the partition
    `_partition` found. This is the first readout in the product that speaks in the
    scientist's own vocabulary — gene symbols — rather than in coordinates.

    **NO p-values reach the g-vector, and that is the design.** Squair et al. 2021
    (`10.1038/s41467-021-25960-2`) is cited by canonical-recipes.md §5.2 precisely because
    per-cell DE between clusters derived from the same data is double-dipping: the clusters
    were DEFINED by the genes now being tested. `vocab.NULL_KIND` has no entry for it and a
    `labels` null would certify it the way it would have certified granger. A significance in
    the record is an invitation to promote it, and the ranking survives without one — so what
    lands is an ORDER over gene names plus `rank_genes_ranked_by`, which says in words what
    the order is and is not.

    **The caller's matrix is copied before it is normalized.** `state["counts"]` IS the loaded
    AnnData's `.X` — the same object, not a copy (`loading.gene_axis`: *"the same object, not
    a copy (state['counts'] is adata.X)"*) — and `sc.pp.normalize_total` writes in place. On
    pbmc3k a CSR copy is 18.3 MB against the 353.6 MB `_dense` would cost, so the copy is
    cheap and skipping it would rewrite the scientist's matrix under them mid-run.

    **Normalizing here is the first time this preprocessing is DECLARED.** `loading.
    _anndata_matrix` already runs normalize_total → log1p → PCA invisibly on every counts-like
    AnnData (§2.5: not a step, absent from the trace, the g-vector and the caveats). This step
    asks `looks_like_counts` first — the same predicate, one home — because `state["counts"]`
    is `.X` untransformed and an already-normalized `.h5ad` must not be normalized twice."""
    import numpy as np

    counts, genes = state.get("counts"), state.get("genes")
    labels, where = _partition(state)
    if counts is None or genes is None:
        g["rank_genes_top"] = None
        g["rank_genes_top_note"] = (
            "no gene axis in this run — `counts`/`genes` are None, so there are no gene names "
            "to rank (engine=manylatents cannot derive them; they must be passed in)")
        return
    if labels is None:
        g["rank_genes_top"] = None
        g["rank_genes_top_note"] = where
        return
    groups = np.asarray([str(v) for v in np.asarray(labels).ravel().tolist()])
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        g["rank_genes_top"] = None
        g["rank_genes_top_note"] = f"ranking a group against the rest needs ≥2, got {n_groups}"
        return
    try:
        import anndata as _anndata
        import pandas as pd
        import scanpy as sc
    except ImportError as e:
        from manyruns.installation import RUNTIME_REPAIR

        # A missing capability is not a benign decline on this data.
        raise _StepUnsupported(
            f"rank_genes needs scanpy + anndata + pandas (missing: {e.name}); "
            f"{RUNTIME_REPAIR}") from e
    from manyruns.pipeline.loading import looks_like_counts

    matrix = counts.copy()  # see the docstring: state["counts"] IS the caller's .X
    ad = _anndata.AnnData(X=matrix)
    ad.var_names = [str(x) for x in np.asarray(genes).ravel().tolist()]
    ad.obs["group"] = pd.Categorical(groups)
    if looks_like_counts(matrix):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    sc.tl.rank_genes_groups(ad, "group", method="wilcoxon")
    names = ad.uns["rank_genes_groups"]["names"]
    top_n = int(_param(params, "top_n", 5))
    g["rank_genes_n_groups"] = n_groups
    g["rank_genes_n_tested"] = int(ad.n_vars)
    g["rank_genes_groups_from"] = where
    # A list of strings, one per group. Not a nested dict: `null.record`'s note applies here
    # too — the g-vector reaches the results table as flat keys, and a dict is dropped in
    # silence by `experiment.metric_columns`.
    g["rank_genes_top"] = [f"{grp}: " + ", ".join(str(x) for x in names[grp][:top_n])
                           for grp in names.dtype.names]
    g["rank_genes_ranked_by"] = (
        "Wilcoxon rank-sum, each group against the rest. The groups were derived from this "
        "same data, so this is a RANKING of what distinguishes them, not a test that they "
        "differ (Squair et al. 2021, 10.1038/s41467-021-25960-2). No p-value is recorded.")


def _step_dpt(state, g, params, out_dir, plots, target_dim) -> None:
    """Diffusion pseudotime — an ORDER over cells, never a significance. (`pseudotime`'s step.)

    Scanpy's `sc.tl.dpt` (Haghverdi et al. 2016, `10.1038/nmeth.3971`) on the current
    embedding, rooted at `_extremal_root`. The per-cell ordering is written to
    `state["pseudotime"]`, which `artifacts.PERSISTED` already carries to disk — so the
    ORDER, which is the actual result, survives the run rather than being reduced to a scalar.

    **`n_dcs` defaults to 2, not to scanpy's 10, and the deviation is measured.** DPT's
    distance accumulates over the first `n_dcs` diffusion components; on a clean
    low-dimensional manifold the higher components are harmonics of the first, which fold the
    manifold back onto itself. Measured this session, |Spearman| against the generating
    parameter, 400 points, `n_neighbors=15`:

    | fixture                          | n_dcs=2 | n_dcs=10 (scanpy default) |
    |----------------------------------|---------|---------------------------|
    | straight line + noise            | 0.996   | **0.060**                 |
    | wiggly 1-D curve + noise         | 0.997   | **0.048**                 |
    | 1-D signal in 10-D gaussian noise| 0.985   | 0.991                     |

    At the default the pseudotime on a clean line rises 0 → 0.99 and falls back to 0.35: not
    an ordering. This matters HERE more than in general use because the canonical `pseudotime`
    recipe runs `dpt` straight after `diffusionmap` — a clean low-dimensional embedding is
    exactly the input the default fails on. `n_dcs` is a declared parameter and the effective
    value is recorded as `dpt.n_dcs`, so the record says which DPT ran.

    That default is chosen on those two fixtures and **is not a claim that 2 is right in
    general**. A third fixture — the same curve through manylatents' own `diffusionmap` step
    — scores 0.013 / 0.584 / 0.080 / 0.089 / 0.088 against truth at n_dcs 2 / 3 / 5 / 10 / 15:
    no value works, because that embedding's 15-NN graph falls into disconnected components
    (below). The failure there is upstream of `n_dcs`, and no default could have rescued it.

    **`dpt_geodesic_agreement` reads in ONE direction only, and the asymmetry is measured.**
    It is the |Spearman| between this DPT and a second ordering of the same points — the kNN
    geodesic from the same root (`_diffusion_pseudotime`, which uses no diffusion map at all).
    Low agreement means the order is not identified. **High agreement is not evidence that
    either is right**, and that is not a caution, it is a measurement: run through
    `diffusionmap → dpt` on the wiggly curve above, the two orderings agree at 0.907 while
    scoring 0.089 and 0.323 against the generating parameter. Both were wrong together,
    because both are built on the same fragmented kNN graph and inherit the same fragmentation.

    What DID fire on that run is `dpt_unreachable`: 81 of 400 cells had no path to the root
    (scanpy prints *"Transition matrix has many disconnected components"*), and an ordering
    that cannot reach a fifth of the data is not an ordering of the data. The count and its
    fraction are emitted for that reason — they are the diagnostic that survived contact.

    **That count is the reason this step exists, and here is the measurement.** On a PHATE
    embedding of the same 300-point ribbon, in-process:

        mioflow → {'pseudotime_range': [0.0, 1.0]}
        dpt     → {'dpt_unreachable': 100, 'dpt_unreachable_frac': 0.3333,
                   'dpt_geodesic_agreement': 0.034, 'dpt_range': [0.0, 1.0]}

    A THIRD of the cells have no path to the root, and the fragmentation is in the embedding
    rather than in either method — the raw 15-NN graph on those coordinates has three
    connected components of 200 / 58 / 42 (scipy `connected_components`, no scanpy involved).
    `_step_mioflow` fills unreachable cells with the largest finite distance and then
    normalizes, so it reports a full-range trajectory and nothing else: a run in three pieces
    is indistinguishable in its record from a clean one. That is the same shape of silent
    success `_require_embedding`'s docstring records, one layer further in.

    Neither number is a significance and none is available: Weinreb et al.
    (`10.1073/pnas.1714723115`) is the standing limit — direction is not identifiable from a
    static snapshot, however well two methods agree about it.

    Branch detection is deliberately absent. `sc.tl.dpt(n_branchings=…)` exists and would be
    the route to rung 3 (`multi-branching`), but scanpy documents it as experimental and this
    session did not test it; §5.3 names PAGA as the canonical branching tool and scopes it as
    a separate executor. So `pseudotime` claims an ordering and rung 3 stays unclaimable —
    §6's "rung 3 becomes claimable" over-promises against §5.3's own step table."""
    import numpy as np

    _require_embedding(state, "dpt")
    try:
        import anndata as _anndata
        import scanpy as sc
    except ImportError as e:
        from manyruns.installation import RUNTIME_REPAIR

        raise _StepUnsupported(
            f"dpt needs scanpy + anndata (missing: {e.name}); {RUNTIME_REPAIR}") from e

    Z = np.ascontiguousarray(np.asarray(state["emb"], dtype=np.float32))
    n = int(Z.shape[0])
    n_neighbors = int(_param(params, "n_neighbors", min(15, max(2, n - 1))))
    n_dcs = int(_param(params, "n_dcs", 2))
    root = _extremal_root(Z)
    # scanpy raises `Cannot instantiate using n_dcs=…` when the diffusion map holds fewer
    # components than dpt asks for (measured: n_comps=5 with the default n_dcs=10), so the
    # map is sized from the request rather than from a constant.
    n_comps = int(min(max(n_dcs, 3, 15), max(2, n - 1)))
    ad = _anndata.AnnData(X=Z)
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.diffmap(ad, n_comps=n_comps)
    ad.uns["iroot"] = root
    sc.tl.dpt(ad, n_dcs=n_dcs)
    pt = np.asarray(ad.obs["dpt_pseudotime"], dtype=float)
    finite = np.isfinite(pt)
    # Cells with no path to the root come back as inf. NaN rather than inf so a colour map
    # and a mean both refuse them instead of quietly taking the largest value in the run —
    # `_diffusion_pseudotime` fills them with the max, which is why it now reports the count.
    pt = np.where(finite, pt, np.nan)
    state["pseudotime"] = pt
    from manyruns.measurements import annotate

    g["dpt.n_dcs"] = n_dcs
    g["dpt.n_neighbors"] = n_neighbors
    for key in ("dpt.n_dcs", "dpt.n_neighbors"):
        annotate(g, key, kind="setting", stage="dpt configuration")
    g["dpt_root"] = root
    g["dpt_unreachable"] = int((~finite).sum())
    g["dpt_unreachable_frac"] = round(float((~finite).mean()), 4)
    # [0, 1] BY CONSTRUCTION whenever two cells are reachable — measured [0.0, 1.0] on all
    # three fixtures in the docstring, at every n_dcs tried. Recorded because the degenerate
    # case ([0, 0], nothing ordered) is a real outcome, NOT because the endpoints measure the
    # data. `_step_mioflow`'s `pseudotime_range` has the same shape and the same limit.
    g["dpt_range"] = [float(np.nanmin(pt)), float(np.nanmax(pt))] if finite.any() else None
    if not finite.any():
        g["dpt_range_note"] = "no cell is reachable from the root — the kNN graph is empty"
        return
    geo_info: dict = {}
    geodesic = _diffusion_pseudotime(Z, n_neighbors, info=geo_info)
    from scipy.stats import spearmanr

    rho = spearmanr(pt[finite], geodesic[finite]).statistic
    g["dpt_geodesic_agreement"] = None if np.isnan(rho) else round(abs(float(rho)), 4)
    if np.isnan(rho):
        g["dpt_geodesic_agreement_note"] = "one of the two orderings is constant"
    _io._save_scatter(Z, pt, out_dir, "dpt.png", plots, title="diffusion pseudotime")


def _step_discretize_time(state, g, params, out_dir, plots, target_dim) -> None:
    """Bin an upstream ordering (`state["pseudotime"]`, e.g. `dpt`'s output) into MIOFlow
    timepoints — MIOFlow trains on discrete timepoint GROUPS, not a continuous ordering.

    Without this step, `mioflow` on `engine=manylatents` (`runner._ml_lightning`) never sees
    `state["pseudotime"]` at all: it only reads `state["labels"]`, and when that is empty it
    computes its OWN pseudotime from scratch and discards whatever ordering step the recipe
    actually ran (`mioflow.py`'s no-real-time fallback). This step is what lets a recipe's own
    `dpt` — or any future ordering step — reach MIOFlow, by writing the binned result into the
    SAME `state["labels"]` / `state["label_kind"]` channel a real per-cell time axis uses
    (`runner.py`'s `label_kind == "time"` guard). The presence of `mioflow.n_timepoints`
    in the run's g-vector afterward is the check that it worked — that key is written only
    when `_ml_lightning` receives non-None labels (`runner.py`), which never happens on a
    `phate -> dpt -> mioflow` recipe without this step in between.

    Refuses (skips) rather than silently overwriting real per-cell time labels already present
    — a derived approximation should never replace real data."""
    import numpy as np

    pt = state.get("pseudotime")
    if pt is None:
        raise _StepSkipped(
            "discretize_time needs a pseudotime; run an ordering step (e.g. dpt) first"
        )
    if state.get("label_kind") == "time" and state.get("labels") is not None:
        raise _StepSkipped(
            "real time labels are already present; discretizing pseudotime would overwrite them"
        )
    pt = np.asarray(pt, dtype=float)
    finite = np.isfinite(pt)
    if not finite.any():
        raise _StepSkipped("discretize_time: pseudotime has no finite values to bin")
    lo, hi = pt[finite].min(), pt[finite].max()
    span = (hi - lo) or 1.0
    # unreachable cells (dpt's NaN for cells with no path to the root) go in the LAST bin —
    # the same fill policy `_diffusion_pseudotime` applies to unreachable cells elsewhere in
    # this module, so a cell with no ordering does not silently vanish from MIOFlow's data.
    norm = np.where(finite, (pt - lo) / span, 1.0)
    n_timepoints = int(params.get("n_timepoints") or 5)
    from manyruns import vocab

    bins = vocab.discretize_pseudotime(norm, n_timepoints)
    state["labels"] = bins.astype(str)
    state["label_kind"] = "time"
    g["discretize_time.n_timepoints"] = n_timepoints
    g["discretize_time.n_bins_populated"] = int(len(set(bins[finite].tolist())))
    g["discretize_time.unreachable"] = int((~finite).sum())
    # The embedding coloured by the BINS, not the continuous `pt` `dpt.png` already shows —
    # this is the picture of what MIOFlow is actually about to train against, so a bin that
    # ate the whole manifold (n_bins_populated far under n_timepoints) is visible before the
    # fit runs, not after. `state["emb"]` is present whenever `pseudotime` legitimately is
    # (every ordering step needs an embedding first), but it is not THIS step's job to enforce
    # that — a missing one just means no picture.
    Z = state.get("emb")
    if Z is not None:
        # A qualitative/discrete cmap and no colorbar: the colour is a GROUP index, not a
        # magnitude, and a viridis colorbar on a handful of integers invites reading it as one.
        _io._save_scatter(Z, bins, out_dir, "discretize_time.png", plots,
                          title=f"discretized time ({n_timepoints} bins)",
                          cmap="tab20", colorbar=False)


def _step_simplex(state, g, params, out_dir, plots, target_dim) -> None:
    """The archetype weights, read as a simplex — `archetypes`' readout.

    Archetypal analysis represents each cell as a convex combination of `k` extremal
    specialists (Cutler & Breiman 1994, `10.1080/00401706.1994.10485840`; Hart et al. 2015,
    `10.1038/nmeth.3254`). `manylatents`' `aa` step returns exactly those weights — its
    `transform` docstring: *"Transforms the input into similarity degrees to each archetype"*
    — and `runner._ml_latent` puts them in `state["emb"]`. This reads them.

    **It verifies its input rather than assuming it.** Every claim below is meaningless unless
    the rows really are convex weights, so non-negativity and row-sums are CHECKED and the
    step reports a note instead of numbers when they fail. That check is doing real work: it
    is what stops this from emitting an "archetype purity" for any 3-column PHATE embedding
    that happens to be sitting in `state["emb"]`.

    **The producer does not run in this checkout, and that is measured, not assumed.**
    `manylatents/algorithms/latent/aa.py` calls `archetypes.AA(method=…, method_kwargs=…)`;
    the installed `archetypes` 0.6.2 takes neither — `AA.__init__` is
    `(n_archetypes, n_init, max_iter, tol, algorithm_init, verbose, random_state)`. Calling it
    directly with only its real arguments then raises
    `AttributeError: 'AA' object has no attribute '_validate_data'` against scikit-learn 1.9.
    So `aa` is broken twice over, upstream of manyruns both times, and this readout has been
    exercised on constructed weight matrices only — never end-to-end. §6 stage 0 ("fix
    manylatents `aa` — upstream, not here") is still the blocker for rung 4.

    **What is NOT computed: the t-ratio.** §5.4 names it, and it is the ratio of the fitted
    simplex's volume to the convex hull's. Both need the archetype POSITIONS in the ambient
    space, which `state` does not carry (`_ml_latent` keeps `embeddings` and drops the fitted
    module), and a hull volume in more than a handful of dimensions is not computable at
    single-cell n. `simplex_purity` against its own floor of 1/k is what the weights alone
    can honestly support."""
    import numpy as np

    _require_embedding(state, "simplex")
    W = np.asarray(state["emb"], dtype=float)
    tol = float(_param(params, "tol", 1e-3))
    if W.ndim != 2 or W.shape[1] < 2:
        g["simplex_purity"] = None
        g["simplex_purity_note"] = "archetype weights need ≥2 columns; this is not a simplex"
        return
    rows = W.sum(axis=1)
    if not np.all(np.isfinite(W)) or W.min() < -tol or not np.allclose(rows, 1.0, atol=tol):
        g["simplex_purity"] = None
        g["simplex_purity_note"] = (
            f"state['emb'] is not a simplex coordinate system (min weight {W.min():.3g}, row "
            f"sums {rows.min():.3g}..{rows.max():.3g}) — an archetypal step must run first")
        return
    k, n = int(W.shape[1]), int(W.shape[0])
    top = W.max(axis=1)
    owner = W.argmax(axis=1)
    occupied = np.bincount(owner, minlength=k)
    threshold = float(_param(params, "vertex_threshold", 0.9))
    g["simplex_k"] = k
    # Mean largest weight. Bounded [1/k, 1] BY DEFINITION: 1 when every cell sits on a vertex
    # (pure specialists), 1/k at the barycentre (every cell an equal blend, i.e. no archetypal
    # structure at all). The floor is emitted beside it because the number is uninterpretable
    # without it — 0.4 is near-perfect at k=2 and near-nothing at k=8.
    g["simplex_purity"] = float(top.mean())
    g["simplex_purity_floor"] = round(1.0 / k, 4)
    g["simplex_at_vertex"] = float((top >= threshold).mean())
    # An archetype that is nobody's nearest is a vertex the data does not use — the signature
    # of a k larger than the structure supports.
    g["simplex_unused"] = int((occupied == 0).sum())
    g["simplex_min_occupancy"] = round(float(occupied.min() / n), 4)
    g["simplex.vertex_threshold"] = threshold


def _step_qc(state, g, params, out_dir, plots, target_dim) -> None:
    """Quality control — **reporting only. This step never removes a cell or a gene.**

    canonical-recipes.md §5.5 is the argument and it is architectural, not stylistic: a
    recipe's output type is a g-vector, while a filter's output type is a smaller dataset. A
    step that rewrote `state` with fewer rows would invalidate `runner.check_aligned`,
    desynchronise any labels fetched separately, and make `n_samples` mean two different
    things depending on where in the recipe a reader looks. So this computes the metrics and
    counts WHAT WOULD BE REMOVED at the declared thresholds, and removes nothing. Filtering is
    an `admit`-stage question (mode-reconciliation.md §0.2), not a step.

    Thresholds default to the Scanpy pbmc3k tutorial's (`min_genes=200`, `min_cells=3`,
    `max_pct_mito=5`) and are declared, not laws — Heumos et al. 2023
    (`10.1038/s41576-023-00586-w`) recommends MAD-based outlier detection over fixed cuts.
    They are recorded as `qc.*` settings so the counts below can be read against them.

    **Measured on `data/pbmc3k_raw.h5ad` in this checkout** (2700 × 32738 CSR float32), which
    is the whole argument for shipping this first: median 817 genes and 2197 counts per cell,
    13 genes named `MT-*`, median 2.03 % mitochondrial reads — and **0 cells below 200 genes**
    against **19,024 of 32,738 genes present in fewer than 3 cells**, with 57 cells over 5 %
    mitochondrial. Every `cluster` or `markers` number the product prints on unfiltered
    pbmc3k is computed over those 19,024 near-empty columns and those 57 cells.

    **`qc_n_genes` is the true width, and the g-vector's `n_features` is not.** This reports
    32,738 on pbmc3k where the g-vector says 50, because `loading._anndata_matrix` PCA'd the
    input before the runner ever saw it. `GVECTOR_CORE` defines `n_features` as the columns of
    the input data, so it is wrong there — but changing a core key breaks comparability across
    every stored run, which is its own decision (§7 of the hand-off). This puts the honest
    number in the record beside it rather than silently correcting it.

    Never densifies: `state["counts"]` stays CSR through every reduction here (`_dense` on
    pbmc3k is 353.6 MB against 18.3 MB stored), and doublet detection (`sc.pp.scrublet`,
    §5.5's second half) is absent because it was not run this session — a doublet score
    nobody has executed is exactly the plausible-looking number this file must not ship."""
    from manyruns import qc

    # THE ARITHMETIC MOVED, THE CONTRACT DID NOT. Every number below is computed by `qc.facts`
    # so the roster can put the same fact beside a dropped file before any recipe runs — one
    # account, two callers. What stays here is the step's own job: reading the thresholds a
    # recipe declared, and writing the result into the g-vector.
    g.update(qc.facts(
        state.get("counts"), state.get("genes"),
        min_genes=int(_param(params, "min_genes", qc.MIN_GENES)),
        min_cells=int(_param(params, "min_cells", qc.MIN_CELLS)),
        max_pct_mito=float(_param(params, "max_pct_mito", qc.MAX_PCT_MITO)),
        mito_prefix=str(_param(params, "mito_prefix", qc.MITO_PREFIX)),
    ))


def _looks_countslike(matrix) -> bool:
    """`loading.looks_like_counts`, imported lazily so this module stays importable without it."""
    from manyruns.pipeline.loading import looks_like_counts

    return looks_like_counts(matrix)


# Analysis steps run in-process (not manylatents algorithms). Dispatched by name so a recipe can
# pick separation / composition (contrast); mock handles them generically.
#
# There is deliberately NO trajectory readout here. `granger` was deleted because it ordered rows by
# a pseudotime derived from the embedding and
# then tested two coordinates of that same embedding against each other, which rejected in
# 12/12 seeds on pure noise (median p 2.9e-29) — and scored the no-lead-lag case as MORE
# significant than a true one (9.9e-38 against 5e-23). Swapping the statistic does not help; PCMCI+/ParCorr returns
# p < 1e-2 in BOTH directions on the identical fixture. Direction is not identifiable from a
# static snapshot at all (Weinreb et al., PNAS 115:E2467), so the honest output for every
# dataset currently in this repo is a refusal, which `narrate` now gives.
#
# `dispatchable` reads this table to decide what each engine can be OFFERED, so a name landing
# here reaches the menu on `real` and `manylatents` at once — the readouts are engine-agnostic
# by construction (they read `state`, not an engine), which is what makes one recipe comparable
# across engines. `mock` answers every analysis name generically through `serving._measure`.
# Stubs merged in BELOW the real entries, so a name defined here always wins: a stub is what a
# step is until someone implements it, never what it becomes again. See `pipeline/stubs.py`.
def _step_velocity_field(state, g, params, out_dir, plots, target_dim) -> None:
    """Read a SUPPLIED velocity field and report whether it is coherent — never compute one.

    **The field comes from another process.** A velocity model needs spliced/unspliced layers
    and, measured on this checkout, a dependency set that cannot coexist with this one
    (`pyrovelocity 0.4.5` resolves only against `numpy<2`; manyruns runs 2.2.6). So the tool
    runs in its own interpreter, leaves an `.h5ad`, and `loading.velocity_field` reads it. That
    is not a workaround — it is CLAUDE.md's rule arriving at its natural conclusion: manyruns
    owns the workflow and the record, and implements none of the compute.

    **What this ADDS over the file it read, and it is the only reason the step exists:**
    coherence. The producer knows the field; it does not know the embedding THIS run built. A
    velocity field is trustworthy to the extent that nearby cells agree about where they are
    going, so the readout is the mean cosine similarity between each cell's velocity and its
    k nearest neighbours' — computed in the ambient gene space, compared against neighbours
    taken from the run's own coordinates. A field of noise scores ~0; a real flow scores high.

    **NOT A FINDING, and deliberately.** `vocab.NULL_KIND` has no row for this step, which by
    that table's own rule ("a readout with no null must not be narrated as a finding") makes
    every number below a description rather than evidence. Coherence is computed from the
    embedding and the field with no label anywhere in it, so a label-permutation null would be
    bit-identical to the real run and would CERTIFY it — the exact failure `docs/granger-
    verdict.md` measured at 12/12 rejections on pure noise. It needs `NULL_KIND: data`, which
    is declared and unimplemented. Said here rather than discovered later.
    """
    import numpy as np

    field = state.get("velocity")
    if field is None:
        raise _StepSkipped(
            "no velocity field in this state — it is SUPPLIED, not computed: run a velocity "
            "tool (see docs/sidecars/) and load the .h5ad it writes")
    coords = state.get("emb")
    if coords is None:
        coords = state.get("X")
    if coords is None:
        raise _StepSkipped("no coordinates to take neighbours from")
    field = np.asarray(field, dtype=float)
    coords = np.asarray(_dense_like(coords), dtype=float)
    if field.shape[0] != coords.shape[0]:
        raise _StepSkipped(
            f"the velocity field has {field.shape[0]} rows and this state holds "
            f"{coords.shape[0]} cells — a narrowing cleared one and not the other")

    g["velocity_field.n_cells"], g["velocity_field.n_genes"] = field.shape
    norms = np.linalg.norm(field, axis=1)
    g["velocity_field.median_speed"] = float(np.median(norms))
    # A cell whose field is exactly zero is not moving in this model; reporting the fraction
    # separates "a slow flow" from "a field that is mostly empty", which the median cannot.
    g["velocity_field.frac_cells_moving"] = float((norms > 0).mean())

    k = int(_param(params, "n_neighbors", 15))
    k = max(1, min(k, coords.shape[0] - 1))
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    idx = nn.kneighbors(coords, return_distance=False)[:, 1:]     # drop self
    unit = field / np.where(norms[:, None] == 0, 1.0, norms[:, None])
    # Mean cosine between each cell's direction and each neighbour's. Zero-norm rows contribute
    # zero rather than NaN, which is the honest reading: a cell with no velocity agrees with
    # nothing, and dropping it would report coherence over the moving cells alone while calling
    # it coherence over the field.
    cos = np.einsum("ij,ikj->ik", unit, unit[idx])
    raw = float(cos.mean())
    g["velocity_field.coherence"] = raw
    g["velocity_field.coherence_k"] = k

    # THE BASELINE, AND IT IS NOT OPTIONAL. Measured on a real pyrovelocity field over 600
    # cells: raw coherence 0.936, and 0.669 after SHUFFLING which cell each vector belongs to
    # — 71% of the number survives destroying the very correspondence it claims to measure.
    # A field with a dominant overall direction scores high on mean-cosine whether or not it
    # tracks the geometry, so the raw number alone is the `granger` shape: large, stable, and
    # a function of the field's anisotropy rather than of local agreement. Gaussian noise at
    # the same scale scores -0.001, so the statistic is not broken — it is uncalibrated.
    #
    # The permutation holds the embedding and the neighbour graph fixed and permutes the
    # CORRESPONDENCE, which is legitimate here for the reason `vocab.NULL_KIND` requires of a
    # cheap null: the statistic reads that correspondence and nothing else. It is deliberately
    # NOT registered in `NULL_KIND` — that table's vocabulary is `labels` and `data`, and this
    # is neither (a velocity field is not a label, and the data is not re-synthesised).
    # Minting a third kind is a VOCABULARY decision and not a step's to take, so the excess is
    # reported and this step still narrates nothing as a finding.
    n_perm = int(_param(params, "n_permutations", 5))
    rng = np.random.default_rng(0 if state.get("seed") is None else int(state["seed"]))
    null = [float(np.einsum("ij,ikj->ik", unit[perm], unit[perm][idx]).mean())
            for perm in (rng.permutation(unit.shape[0]) for _ in range(max(1, n_perm)))]
    g["velocity_field.coherence_null"] = float(np.median(null))
    # What is left after the field's own anisotropy is accounted for. THIS is the number to
    # read; `coherence` alone is kept because dropping it would hide the size of the correction.
    g["velocity_field.coherence_excess"] = raw - float(np.median(null))

    unc = state.get("velocity_uncertainty") or {}
    for key, values in unc.items():
        values = np.asarray(values, dtype=float)
        if values.shape[0] != field.shape[0]:
            continue
        # `median`, not `mean`: these are per-cell posterior summaries and one badly-fit cell
        # should not move the number the record keeps.
        g[f"velocity_field.{key}.median"] = float(np.median(values))
    # Whether the producer kept the posterior AT ALL is itself a fact about the run — a scVelo
    # field and a Bayesian one are indistinguishable downstream without it.
    g["velocity_field.has_uncertainty"] = bool(unc)


def _dense_like(m):
    """A 2-D array from a possibly-sparse matrix, without importing scipy to ask."""
    return m.toarray() if hasattr(m, "toarray") else m


_ANALYSIS_STEPS = {
    "separation": _step_separation,
    "composition": _step_composition,
    "cluster_quality": _step_cluster_quality,
    "rank_genes": _step_rank_genes,
    "dpt": _step_dpt,
    "discretize_time": _step_discretize_time,
    "simplex": _step_simplex,
    "qc": _step_qc,
    # READS a field it did not compute. `analysis` rather than `probe` because it consumes
    # DATA — a supplied array — not a fitted object: there is nothing here to ask a question of,
    # only a field to describe.
    "velocity_field": _step_velocity_field,
}

# The in-process loop's local stand-in for the engine's algorithm catalogue: latent/lightning steps
# computed with public libraries so the product runs with no private stack. Resolved at call
# time by `_inproc_step`, so definition order here doesn't matter.
_INPROC_STEPS = {
    "phate": _step_phate,
    "mioflow": _step_mioflow,
}

