"""The vocabulary — every closed word list the product uses, in ONE dependency-free file.

This is the file to read before authoring a recipe or declaring a dataset. If a word isn't
here, nothing downstream will recognise it.

It exists because the alternative kept biting: a vocabulary declared twice drifts, and the
drift is silent. `kind` vs `group` cost a forked trace format and a default that ran typo'd
steps as embeddings; the `.obs` time words were forked between the analysis selector and the
loader, which turned batch IDs into a fake time axis. Both were the same mistake — two
homes for one list — so there is now one home.

Dependency-free on purpose: importing this must never pull the compute stack. `protocols`
used to reach into `pipeline` for the three step groups, dragging the entire pipeline
package (io, loading, mioflow, runner, steps) along to obtain a 3-tuple of strings.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

# ── recipe steps ─────────────────────────────────────────────────────────────
#: What a recipe step declares. The single step vocabulary — there is no parallel `kind`.
#:   latent    — an embedding/transform, run by the engine's algorithm catalogue
#:   lightning — a trained step, likewise
#:   analysis  — manyruns's own in-process readout; NOT an engine algorithm, which is why
#:               this stays a product-owned sentinel rather than an engine group name
#:   prep      — DATA CONDITIONING: it changes the data and produces NO fact. May remove rows
#:               or columns. Delegated in full; manyruns declares the order and the
#:               parameters and implements none of the compute.
#:   probe     — a QUESTION put to a trained model. Reads `model`, writes to the g-vector,
#:               transforms no data and moves no selection.
#:
#: THE LINE BETWEEN THE GROUPS IS THE FACT EACH PRODUCES, not the domain it works in. `latent`
#: yields an `embedding`; `lightning` yields a `model`; `analysis` and `probe` yield g-vector keys
#: and no fact; `prep` yields nothing at all — it hands on the same data, conditioned. That is why
#: `prep` is NOT "scRNA preprocessing" despite every member of it being one today: batch
#: correction, imputation and scaling are the same shape and belong to it when they arrive, with
#: no new group and no new rule. The name is narrower than the definition; the definition governs.
#:
#: Members do vary in whether they NARROW, and that variation is real rather than a sign the group
#: is wrong: `vocab.NARROWING_STEPS` carries it per step, keyed by the param that decides it, and
#: the calculus reads that rather than the group. Splitting `prep` in two by that axis was
#: considered and refused — a recipe interleaves both kinds in one block, and `detect_doublets`
#: narrows only when `remove: true`, so the axis is not a property of the step at all.
#:
#: `prep` is the group that made the state model change, and the reason is that it is the only
#: one whose steps can make the data SMALLER. `loading._anndata_matrix` used to apply
#: normalize_total → log1p → PCA(50) on every counts-like load as a step nothing declared, in
#: no trace and no g-vector, destroying the gene axis before the runner saw anything
#: (2700 × 32738 → 2700 × 50 on pbmc3k). Those operations are steps now. See
#: `pipeline/frame.py` for what had to become true of `state` first.
#:
#: `probe` is the group that makes the product answer something other than "what did the
#: embedding look like". A transcript of performed analyses differs from a model
#: of the phenomenon: the difference in practice is whether there is an
#: object left to ask. A probe is a step rather than a fourth axis because it IS a step: it
#: consumes a fact (`model`) that another step produced, and `CLAUDE.md`'s extension path for a
#: new capability is a group plus one executor.
STEP_GROUPS = ("latent", "lightning", "analysis", "prep", "probe")

# ── datasets ─────────────────────────────────────────────────────────────────
#: Ground-truth data shapes — the label the metric-separation experiment scores against
#: ("which metrics tell data shapes apart"). Closed, because a typo here silently invents a
#: spurious class in the results table.
SHAPES = ("manifold", "clusters", "time-course", "case-control", "single", "unknown")

#: How a dataset's bytes are reached. Both kinds are a plain string, so a dataset
#: declaration pickles into a worker process cleanly.
HANDLE_KINDS = ("manylatents", "path")

#: Facts a dataset DECLARATION carries, as opposed to its `shape`. `frame_facts` unions it, so
#: everything here survives a narrowing by construction — which is the property that made
#: declaring `genes` safe at all, and the reason this tuple exists separately from
#: `SHAPE_PROVIDES` rather than being folded into it.
#:
#: `genes` — the name of each column of the expression matrix. Carrying a gene axis is a
#: property of the FILE and not of the shape, which is exactly why it could not live in
#: `SHAPE_PROVIDES`: an `.h5ad` of any shape has one and a synthetic point cloud of the same
#: shape has none, so keying it on shape would return `{genes}` on all six and prune the
#: recipes that run on the one real dataset in the repo. See `dataset_provides`.
#: `splicing` — the file carries spliced AND unspliced counts as layers, the input every RNA
#: velocity method needs and no shape can supply. A frame fact for the same reason `genes` is
#: one, and unlike a velocity FIELD (which is derived and cleared by a narrowing): a layer is
#: raw counts per cell per gene, so subsetting rows and columns is exactly right and it survives
#: every narrowing by construction. Declared by the dataset, because it is a property of the
#: BYTES — a time-course `.h5ad` quantified with velocyto has it and a time-course point cloud
#: of the same shape does not.
DATASET_FACTS: tuple[str, ...] = ("genes", "splicing")

#: The TOPOLOGY a dataset carries — scShapeBench's label set (arXiv:2605.12662) plus the two
#: graph classes its label set omits but its generators cover. Additive alongside `SHAPES`,
#: not a replacement: `SHAPES` currently mixes topology (`manifold`/`clusters`/`single`) with
#: experimental DESIGN (`time-course`/`case-control`), and only the design half feeds
#: `SHAPE_PROVIDES`. Splitting them is a schema migration; declaring the topology axis is not,
#: so this lands first and the migration is decided with data in hand.
#:
#: A LIST, not a scalar — the benchmark is multi-label under union aggregation over nine
#: annotators, because a dataset frequently exhibits more than one organizational regime
#: (distinct cell types progressing through a cycle is both `clusters` and `single-trajectory`).
TOPOLOGIES = (
    "clusters",           # discrete, well-separated groups
    "single-trajectory",  # a one-dimensional ribbon, no bifurcation
    "multi-branching",    # a tree: one arm splits into two or more
    "archetypal",         # a simplex — extremal specialists at the vertices
    "cycle",              # a closed loop (not in scShapeBench's L; its graph classes cover it)
    "surface",            # a 2-manifold (likewise)
)


def check_topology(values: Any) -> list[str]:
    """Problems with a dataset's `topology` field; empty means valid (or absent).

    Absent is legal — the field is additive, so the bundled datasets predate it. Present but
    not a list is not, because a scalar would silently re-impose the single-label assumption
    the benchmark exists to reject.
    """
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        return [f"topology must be a LIST (the benchmark is multi-label), got {values!r}"]
    if not values:
        return ["topology is an empty list — omit the field instead of declaring nothing"]
    unknown = [v for v in values if v not in TOPOLOGIES]
    return [f"unknown topology {unknown} — must be from {TOPOLOGIES}"] if unknown else []


def check_provides(values: Any) -> list[str]:
    """Problems with a handle's `provides:` field; empty means valid (or absent).

    THE FIELD IS ONLY WORTH DECLARING IF A TYPO IS CAUGHT HERE. `dataset_provides` intersects
    the declaration with `DATASET_FACTS` on a hot path and ignores what it does not recognise
    — the right call there, and it is what makes this function load-bearing rather than
    decorative. Measured on `pbmc3k`'s handle (`{kind: path, ref: data/pbmc3k_raw.h5ad}`)
    before this existed, on both shapes of typo:

        provides: [genes]  -> {'genes'}   (correct)
        provides: [gene]   -> set()       (a letter short)
        provides: genes    -> set()       (a scalar: set("genes") == {'e','g','n','s'})
        no `provides:` key -> {'genes'}   (the `.h5ad` suffix fallback)

    A mistyped declaration therefore left the dataset with STRICTLY FEWER facts than declaring
    nothing at all — the `declared is not None` branch also suppresses the suffix fallback that
    would have supplied `genes` — and `check_dataset` returned `[]`, so `manyruns check` named
    nothing. Every gene recipe then went hollow on that dataset with no error anywhere.

    A LIST, and a scalar is refused rather than wrapped: `set("genes")` is four letters, so
    accepting the scalar quietly means accepting a set of characters as a set of facts, which
    is the failure above rather than a convenience. Unknown names are refused for the same
    reason the field exists — an ignored one is indistinguishable from an absent one.

    An EMPTY list is legal, unlike `topology`'s (see :func:`check_topology`), and the
    difference is that this field has a fallback to override. `provides: []` on an `.h5ad`
    written without `var_names` is the only way to say "these bytes carry no gene axis" and
    suppress the suffix guess; refusing it would leave that dataset no way to be honest.
    """
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, (list, tuple)):
        return [f"handle `provides` must be a LIST of dataset facts from {DATASET_FACTS}, "
                f"got {values!r} — a scalar is read as a set of its CHARACTERS, which silently "
                f"declares nothing"]
    unknown = [v for v in values if v not in DATASET_FACTS]
    if unknown:
        return [f"unknown handle provides {unknown} — must be from {DATASET_FACTS}; an "
                f"unrecognised fact is ignored by `dataset_provides`, so a typo here reads as "
                f"a dataset that carries less than it does"]
    return []


def check_claims(value: Any) -> list[str]:
    """Problems with a recipe's `claims` field; empty means valid (or absent).

    A SCALAR, unlike a dataset's multi-label `topology:` — a dataset can exhibit several
    regimes at once, and the benchmark aggregates nine annotators by union, but a recipe
    asserts ONE structure and that assertion is what `score.RUNG` grades. A list here would make
    "which rung
    does this recipe claim" ambiguous at exactly the point the ladder needs one answer.

    Absent is legal: the field is additive and a recipe that declares nothing simply does not
    enter the rung table, which is the same thing an unlisted recipe did before it existed.
    """
    if value is None:
        return []
    if not isinstance(value, str):
        return [f"claims must be a single topology (a recipe asserts one thing), got {value!r}"]
    if value not in TOPOLOGIES:
        return [f"unknown claims {value!r} — must be one of {TOPOLOGIES}"]
    return []


# ── provenance: where a recipe's workflow comes from ─────────────────────────
#: What kind of thing a recipe's `source:` points at. Closed, and `none` is a real member:
#: `contrast`'s two analysis steps are manyruns's own readouts with no publication behind
#: them, and a vocabulary with no way to say that invites a plausible-looking citation for a
#: method that was never run.
#:
#: `source` is provenance pointing OUT of the system, at literature no code will resolve —
#: which is why it is not `ref:` (taken: `handle.ref` is a string the compute layer resolves
#: to bytes) and not `provenance:` (taken: `runner.identity`'s run_id/spec_id, the provenance
#: of one execution).
SOURCE_KINDS = ("paper", "tutorial", "book", "preprint", "none")

# ── AnnData `.obs` columns ───────────────────────────────────────────────────
#: Columns that signal a real ORDERING → a trajectory is meaningful.
TIME_KEYS = ("timepoint", "time", "day", "days", "stage", "week", "pseudotime")

#: Columns that signal unordered GROUPS → case/control contrast, not a trajectory.
CONDITION_KEYS = ("disease", "condition", "group", "status", "diagnosis", "genotype", "treatment")

#: Deliberately NOT auto-detected as time. `sample`/`batch` are genuinely ambiguous — a
#: batch axis and a collection-time axis look identical in `.obs` — and guessing wrong
#: invents a trajectory out of batch structure. Pass `--time-key` to say so explicitly.
AMBIGUOUS_TIME_KEYS = ("sample", "batch")

#: Columns that name an unordered IDENTITY per row — a cell type, a cluster, a branch. They
#: COLOUR a plot and they are not an axis anything integrates over.
#:
#: `"group"` is deliberately absent: it is already in `CONDITION_KEYS`, where it means a
#: case/control arm. The KIND named `group` and a COLUMN named `group` are different things,
#: and listing the name in both tables would make one shadow the other by check order.
GROUP_KEYS = ("cell_type", "celltype", "cell_types", "annotation",
              "branch", "cluster", "clusters", "leiden", "louvain")

#: The label kinds whose colour is an IDENTITY, not a magnitude — so a legend is the honest
#: key and a colorbar is not. `time` is deliberately absent: `TIME_KEYS` is "a real ORDERING",
#: `numeric_time` preserves its real spacing, and a scale bar is what reads that.
#:
#: Measured on `data/pbmc3k_annotated.h5ad` before this existed: eight named immune types were
#: ranked ALPHABETICALLY onto an even [0,1] viridis ramp (`numeric_time`'s non-numeric fallback),
#: so NK cells (154 of 2,638) got the brightest yellow and 15 Megakaryocytes the second
#: brightest, while CD4 T (1,144) and CD8 T (316) landed on adjacent teals. The picture argued a
#: magnitude the data does not have, and the eight NAMES appeared in no file under `outputs/`.
#:
#: AN ALLOWLIST, the same shape and for the same stated reason as `runner._ml_lightning`'s
#: `== "time"` and `steps._step_separation`'s `== "condition"`: a denylist of one is the defect
#: one kind later. A new unordered kind must be added here to be drawn as one.
CATEGORICAL_KINDS = ("condition", "group")


def _find(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """First candidate present in `columns` (case-insensitive) → its ORIGINAL spelling."""
    cols = {str(c).lower(): str(c) for c in columns}
    for k in candidates:
        if k in cols:
            return cols[k]
    return None


def find_time_key(columns: Iterable[str]) -> Optional[str]:
    """The per-cell time column, or None. Never returns an ambiguous batch-ish column.

    Both the analysis selector (which decides WHICH analysis runs) and the loader (which
    decides WHAT LABELS the trajectory model gets) call this. The loader must not
    out-guess the selector — that is the whole reason this lives in one place.
    """
    return _find(columns, TIME_KEYS)


def find_condition_key(columns: Iterable[str]) -> Optional[str]:
    """The per-cell condition column (the case/control label), or None."""
    return _find(columns, CONDITION_KEYS)


def find_group_key(columns: Iterable[str]) -> Optional[str]:
    """The per-row identity column (the colouring axis), or None."""
    return _find(columns, GROUP_KEYS)


def has_time_axis(columns: Iterable[str]) -> bool:
    return find_time_key(columns) is not None


# ── numeric time — the coordinate MIOFlow integrates over ────────────────────
#
# `find_time_key` reads column NAMES to decide WHICH column is the time axis; this reads that
# column's VALUES to turn them into the per-cell coordinate. Both live here so the name-side and
# the value-side of "time" can never disagree.
#
# The point is magnitude preservation. `day0 / day3 / day9` are 0, 3, 9 — real, uneven sampling —
# so the trajectory model must see them as 0, ⅓, 1, not flattened to evenly spaced ranks (0, ½,
# 1). Collapsing real spacing to ranks was a silent leak: the ODE integrated over the wrong clock.

#: A single token reads as a timepoint if it is a bare number (`3`, `0.5`) or a prefixed token
#: (`t0`, `tp_2`, `day3`, `week1`, `stage2`). Free-form IDs (`donorA`) never match.
_TIMEPOINT_TOKEN = re.compile(
    r"^\s*(?:t|tp|d|day|days|stage|week|timepoint|time)[-_ ]?\d+(?:\.\d+)?\s*$"
    r"|^\s*\d+(?:\.\d+)?\s*$",
    re.IGNORECASE,
)
_NUMBER_IN_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def looks_like_timepoint(token: Any) -> bool:
    """True if a single label reads as a timepoint (`3`, `day3`, `t0`, `week1`)."""
    return bool(_TIMEPOINT_TOKEN.match(str(token)))


def parse_timepoints(values: Iterable[Any]) -> Optional[list]:
    """Real numeric magnitudes for a sequence of labels, or None if ANY is non-numeric.

    `day0 / day3 / day9 → [0., 3., 9.]`; `donorA / donorB → None`. All-or-nothing so a single
    free-form label falls the whole sequence back to categorical ranks (see `numeric_time`)."""
    out: list = []
    for v in values:
        if not looks_like_timepoint(v):
            return None
        m = _NUMBER_IN_TOKEN.search(str(v))
        out.append(float(m.group()) if m else 0.0)
    return out


def numeric_time(labels: Iterable[Any]) -> Any:
    """Per-cell time labels → float32 array in [0, 1], preserving REAL spacing.

    Numeric labels (`day0/day3/day9`, `0/3/9`) keep their magnitudes, so uneven sampling stays
    uneven. Non-numeric categories (`early/mid/late`) fall back to evenly spaced ranks over the
    sorted unique values (the old behavior, correct when there is no real magnitude). numpy is
    imported lazily so the name-resolution path above stays dependency-free."""
    import numpy as np

    labels = list(labels)
    raw = parse_timepoints(labels)
    if raw is not None:
        vals = np.asarray(raw, dtype=np.float64)
    else:
        uniq = sorted(set(map(str, labels)))
        idx = {u: i for i, u in enumerate(uniq)}
        vals = np.asarray([idx[str(v)] for v in labels], dtype=np.float64)
    span = float(vals.max() - vals.min()) if len(vals) else 0.0
    scaled = (vals - vals.min()) / span if span else np.zeros_like(vals, dtype=np.float64)
    return scaled.astype(np.float32)


def discretize_pseudotime(pt: Any, n_timepoints: int) -> Any:
    """Bin a normalized, finite `[0, 1]` pseudotime into `n_timepoints` equal-width integer
    groups — MIOFlow trains on discrete timepoints, never a continuous ordering.

    ONE home so MIOFlow's own no-real-time fallback (`mioflow._run_mioflow_experiment`) and an
    explicit `discretize_time` recipe step (binning an upstream ordering, e.g. `dpt`) cannot
    compute two different binnings of the same kind of value. `pt` must already be finite and
    scaled to `[0, 1]`; NaN/unreachable-cell handling is the caller's policy, not this
    function's — a bin boundary has nothing to say about a cell with no position at all."""
    import numpy as np

    pt = np.asarray(pt, dtype=np.float64)
    n_timepoints = max(int(n_timepoints), 1)
    return np.clip((pt * n_timepoints).astype(int), 0, n_timepoints - 1)


# ── the precondition calculus ────────────────────────────────────────────────
#
# Which steps can legally run on which data, IN WHICH ORDER. Small on purpose: these are
# DERIVED from what the steps actually do, not copied from the learner's `PRECONDITIONS`, which
# would over-prune us — their `mioflow` requires a `time` coord, ours computes a diffusion
# pseudotime when there is no time axis, so the cell is legal here and illegal there.
#
# This is the legal-move function. `plan()` uses it to prune cells that cannot produce a
# result, and it is the same mask a search over recipe space would need: legality has to be
# computable WITHOUT running the recipe, or the frontier can only be discovered by executing
# it. `needs` maps onto `manykinds.KindSpec(coords=…)` if/when we adopt kinds — this is that
# idea in its minimal form, in our own vocabulary, with no array-stack dependency.
#
# A fact reaches a step from one of TWO sources: the dataset carries it (`SHAPE_PROVIDES`),
# or an earlier step produced it (`GROUP_PROVIDES`). There is still ONE needs table and ONE
# subtraction — `unmet` just folds it along the recipe instead of unioning over it, so a
# step's needs are checked against the facts available AT ITS OWN POSITION. A second needs
# table, keyed by "order", would have been the two-homes mistake this whole file exists to
# prevent; there is one needs table and two sources that can satisfy it.

#: What a dataset of each SHAPE carries.
SHAPE_PROVIDES: dict[str, tuple[str, ...]] = {
    "time-course": ("time",),
    "case-control": ("conditions",),
    "manifold": (),
    "clusters": (),
    "single": (),
    "unknown": (),
}

#: What running a step ADDS to the facts available to the steps AFTER it — the second source,
#: and the whole of the order-awareness. Keyed by step GROUP, not by step name, because that
#: is how the engines actually dispatch: `runner._ml_latent` assigns `state["emb"] = emb` for
#: ANY `latent`-group step, and `catalog.check_recipe` validates a step's `group`
#: against `STEP_GROUPS` while leaving its `name` open — so a manylatents recipe may
#: legitimately name an embedder other than `phate`. Keying on the name would have called
#: such a recipe ILLEGAL: a currently-runnable recipe newly pruned from `plan()` and from the
#: shell's menu, which is the one thing adding order-awareness must not do. It is also the
#: smaller vocabulary — one entry, rather than a product-side copy of the engine's algorithm
#: catalogue that has to be maintained in step with it.
#:
#: `lightning` is absent although `runner._ml_lightning` also assigns `state["emb"] = emb`:
#: `steps._step_mioflow` does NOT — it reads `state["emb"]` and writes only
#: `state["pseudotime"]` (checked by source, not from memory). The engines disagree on that
#: one group, so declaring it would be true of one and a lie about the other — and no step
#: needs the fact anyway. A group lands here only when both engines agree, because `unmet`
#: takes no engine argument and its answer has to hold for every engine.
GROUP_PROVIDES: dict[str, tuple[str, ...]] = {
    "latent": ("embedding",),
}

#: What ONE NAMED step adds, on top of what its group adds. The narrow companion to
#: `GROUP_PROVIDES`, and the two are keyed differently for the same reason: key on the thing
#: that is actually true of the fact.
#:
#: `embedding` is true of the GROUP — every `latent` executor assigns `state["emb"]`, whatever
#: algorithm it ran — so keying it by name would call a recipe naming any embedder but `phate`
#: illegal (the argument `GROUP_PROVIDES` above makes at length). `clusters` is the opposite
#: shape of fact: a partition is produced by clustering and by nothing else, and of the 15
#: names `manylatents.algorithms.latent.list_algorithms()` resolves — archetypal_analysis,
#: classifier, diffusion_map, leiden, mds, merging, multiscale_phate, no_op, pca, phate,
#: reeb_graph, selective_correction, trajectory_aligner, tsne, umap
#: — exactly ONE clusters. Keying that by
#: group would say every PCA produces a partition; keying it by name says leiden does.
#:
#: The known cost, stated rather than discovered later: a clustering algorithm added upstream
#: under a new name is not in this table, so a recipe using it to feed `cluster_quality` or
#: `rank_genes` is reported illegal until a line lands here. That is the direction this table
#: is allowed to be wrong in — it prunes a step that would have run, it never admits one that
#: cannot. The escape hatch is the same one `embedding` has: `unmet(..., provided={"clusters"})`
#: for a caller who holds a partition already.
STEP_PRODUCES: dict[str, tuple[str, ...]] = {
    "leiden": ("clusters",),
    # The trained flow — the SIMULATABLE object, not its coordinates. `model-not-pipeline.md`
    # §3.1 names its discard as "the whole thesis, inside our own code"; this is the fact that
    # makes keeping it addressable, so a `probe` step can declare that it needs one.
    #
    # BY NAME, and the reason is the one this table's header states. `GROUP_PROVIDES` means "true
    # of every executor in this group", and it is not: `steps._step_mioflow` — the in-process
    # `lightning` executor — computes a diffusion pseudotime on a kNN graph and trains NOTHING,
    # so "a lightning step produces a model" is true on `manylatents` and a lie on `_inproc` and
    # `mock`. That is exactly the test which keeps `lightning` out of `GROUP_PROVIDES` for
    # `embedding` (see the note there), applied to a second fact.
    #
    # Keying by name does not make the remaining lie disappear, only smaller: `mioflow` on
    # `_inproc` is the stand-in, so a recipe reading `[mioflow, sample_trajectories]` is legal
    # here and declines at run time. `unmet` takes no engine argument BY DESIGN — its answer has
    # to hold for every engine — so "this engine trains nothing" is a run-time fact, reported by
    # `runner._run_probe_step` as a declined step with a reason, and pruned from the menu ahead
    # of time by `runner.dispatchable`, which is the mechanism that DOES know the engine.
    "mioflow": ("model",),
    # BY NAME, exactly as `mioflow` and `leiden` are: a velocity field is true of THIS
    # algorithm, not of its group. `pyrovelocity` is a `lightning` step — it fits a
    # probabilistic model by SVI — but the model stays in the TOOL's interpreter and only the
    # field crosses the process boundary. So it declares `velocity` and not `model`: the record
    # must describe what manyruns actually holds, and a `probe` placed after it would find
    # nothing to ask.
    "pyrovelocity": ("velocity",),
}

#: What a step needs BEFORE it can run — satisfiable from EITHER source above. One needs
#: table; which source supplies a fact is `unmet`'s business, not a second table's.
#: Deliberately not "everything a step wants":
#:   - `mioflow` declares `embedding` but NOT `time`: it prefers a real time axis and falls
#:     back to a diffusion pseudotime when there is none, so `time` is a preference. The
#:     embedding is not. Measured before this entry existed, on `{steps: [mioflow, phate]}`:
#:     `unmet` returned `frozenset()` — LEGAL — on ALL SIX shapes, because it unioned
#:     `STEP_NEEDS` over the steps and so could not see position. Commit
#:     df478cc's engine-side `_require_embedding` was then the only thing standing between
#:     that recipe and a published `pseudotime_range = [0.0, 1.0]` computed over data no
#:     embedding had touched. This is that same rule one layer earlier — the engine refuses
#:     at RUN time, this refuses at PLAN time — and plan-time is where a legality function
#:     has to answer, because a search over recipes cannot afford to run each one to find out.
#:   - `separation`/`composition` genuinely cannot run without ≥2 condition groups; they
#:     return None and write a note, which the results table then reads as a MISSING metric
#:     — indistinguishable from "this recipe doesn't emit that metric". That conflation is
#:     what this table exists to remove.
#:   - the five readouts added with §6's ladder declare exactly what they cannot run WITHOUT,
#:     which for three of them is a partition and for two is coordinates. What they do NOT
#:     declare is the gene axis, and that omission is deliberate — see the note below.
STEP_NEEDS: dict[str, tuple[str, ...]] = {
    "mioflow": ("embedding",),
    "separation": ("conditions",),
    "composition": ("conditions",),
    # The gene axis, declarable at last — see the note below this table for why it could not
    # be before. Both READ gene names: `filter_mito` prefix-matches `MT-` and `filter_genes`
    # counts cells per gene. Refused on data with no gene axis rather than matching no columns
    # and reporting that they filtered nothing, which is a sentence indistinguishable from a
    # dataset that had nothing to filter.
    #
    # THESE TWO FIRST, and no longer these two only. When the fact landed, `qc` and `rank_genes`
    # needed the axis just as genuinely and abstained, because declaring it then would have
    # pruned them on every bundled dataset — none carried genes, four commits before `loading`
    # stopped PCA-ing the axis away. That commit has landed, `pbmc3k` declares
    # `provides: [genes]`, and both now declare the need lower down in this table. The split is
    # kept in the record because it is the rule: a demand lands with the supply that can satisfy
    # it, never before.
    "filter_mito": ("genes",),
    "filter_genes": ("genes",),
    # A question put to a trained flow. Declaring the need is what refuses a probe placed before
    # any fit, and — because `model` is step-produced and so absent from `frame_facts` — what
    # refuses one placed after a filter that dropped the cells the flow was fitted over.
    "sample_trajectories": ("model",),
    # DECLARED-BUT-UNIMPLEMENTED (`pipeline/stubs.py`). Their needs are declared for the same
    # reason a real step's are: the ORDER is testable before the maths exists, so a recipe that
    # puts `granger` before the thing that produces its trajectory is refused at plan time today
    # rather than after someone writes the estimator.
    "granger": ("model",),
    "decode_to_gene_space": ("model", "genes"),
    "growth_rate": ("model",),
    # A readout OF a partition. Not `embedding`: a silhouette over the raw input coordinates
    # is a perfectly well-typed question (it is what you would compute before any DR), so
    # requiring an embedding would prune a legal composition for being unconventional —
    # the distinction `unmet`'s docstring draws between cannot-compose and merely-unusual.
    "cluster_quality": ("clusters",),
    # `genes` GAINED AT THE CUTOVER, and deferred to it deliberately (Task 4 shipped the supply
    # side only). Declaring it earlier would have pruned these two on EVERY bundled dataset,
    # because `loading._anndata_matrix` reduced every scRNA load to a 50-column PCA and the gene
    # axis was gone before a step could ask for it. Loading hands over the frame now, so the
    # demand is satisfiable and the refusal is real.
    #
    # What it costs, measured this checkout: 26 of 154 recipe x dataset cells — `markers` (which
    # is where `rank_genes` runs) and `qc` on 13 of the 14 bundled datasets each, i.e. on every
    # synthetic point cloud and not on `pbmc3k`. Before the declaration both steps RAN on a
    # nameless matrix and recorded a note, which is the conflation the fact exists to remove:
    # "no genes to rank" from a swissroll and from a real dataset whose names failed to load were
    # one sentence.
    "rank_genes": ("clusters", "genes"),
    "qc": ("genes",),
    # Orders cells along a manifold, so it needs coordinates for the same reason `mioflow`
    # does: reading a raw ambient matrix as a coordinate system is the composition that
    # cannot compose, and `steps._require_embedding` refuses it at run time. This refuses it
    # at plan time, where a search over recipes has to get its answer.
    # THE GATE. `splicing` is a DATASET fact with no step producer and none possible — it comes
    # off the sequencer, through velocyto or kallisto|bustools. Declaring the need is what makes
    # a velocity recipe on a point cloud illegal BEFORE it runs, rather than erroring three
    # steps in. Measured cost: refused on all 14 bundled datasets, exactly as `genes` was.
    "pyrovelocity": ("splicing",),
    # A readout OF a field, so it needs one. Satisfied by the tool step above, or by a caller
    # who already holds a field (`unmet(..., provided={"velocity"})`).
    "velocity_field": ("velocity",),
    "dpt": ("embedding",),
    "simplex": ("embedding",),
    # Bins an upstream ordering (`state["pseudotime"]`, e.g. `dpt`'s output) into MIOFlow
    # timepoints. Declares the same `embedding` need as `dpt`/`mioflow` for the same reason:
    # the calculus does not model order, so it cannot say "needs dpt specifically" — the
    # runtime guard (no `state['pseudotime']` -> skip) is what actually enforces the sequence.
    "discretize_time": ("embedding",),
}

# ── how `genes` came to be declarable, and what the declaration costs ────────────────────
#
# THIS BLOCK USED TO ARGUE THE OPPOSITE and is kept as the record of why, because the argument
# was right at the time and its shape is the rule. It read: `rank_genes` and `qc` genuinely
# require the gene axis and must NOT declare it, because nothing in the calculus could supply
# it. `SHAPE_PROVIDES` is keyed by data SHAPE and carrying a gene axis is a property of the
# FILE — an `.h5ad` of any shape has one, a synthetic point cloud of the same shape has none —
# so the subtraction returned `{genes}` on all six shapes and pruned the two recipes that
# actually run on the one real dataset in the repo. A precondition that fires on correct data is
# the `lid <= final_dim` bound again (configs/metrics/default.yaml records why that one was
# removed), and it is worse than none.
#
# WHAT CHANGED IS THE SUPPLY, exactly where that block said it belonged: with the DATASET
# declaration. `DATASET_FACTS` carries `genes`, `dataset_provides` reads it off the HANDLE, and
# `configs/dataset/pbmc3k.yaml` declares `provides: [genes]` — the way a `case-control` shape
# provides `conditions`. `loading._anndata_matrix` returning `adata.X` untouched is what made
# that declaration true of the bytes rather than of a 50-column PCA of them.
#
# WHAT IT COSTS, measured this checkout rather than estimated: 11 recipes x 14 datasets = 154
# cells, 53 pruned. `contrast` loses 14 for `conditions` (no bundled dataset is `case-control`);
# `markers`, `qc` and `preprocess` lose 13 each for `genes` — every dataset except `pbmc3k`,
# which is every synthetic point cloud we ship. Not one cell is pruned on data that carries what
# it asks for, which is the property that separates this from the `lid` bound. The refusal is
# real now: before it, both steps ran on a nameless matrix and recorded a note, so "no genes to
# rank" from a swissroll and from a real dataset whose names failed to load were one sentence.

#: How a readout's null is built — the closed vocabulary of null KINDS.
#:
#: `labels` — permute the condition assignment, hold the embedding and clustering fixed.
#:            Valid only for a statistic that READS labels.
#: `data`   — synthesise null data and re-run the *entire* pipeline (embedding, ordering,
#:            statistic). The only honest null for a statistic derived from the geometry
#:            alone. Nothing implements this yet; it is named so that a step needing it
#:            cannot quietly be given the cheap one instead.
NULL_KINDS = ("labels", "data")

#: Which readout gets which null. **A step absent from this table gets no null**, and a
#: readout with no null must not be narrated as a finding.
#:
#: This table is where a measured failure is encoded so it cannot be re-lost. The deleted
#: `granger` step ordered rows by a pseudotime computed from
#: the embedding and then tested two coordinates of that same embedding against each other.
#: It never read a label — so a `labels` null would have been bit-identical to the real run
#: and scored it a *perfect* null, while the statistic returned p < 1e-30 on pure noise in
#: 12/12 seeds. The lesson generalises past that one step: cheap nulls do not merely fail to
#: catch geometry-derived statistics, they actively certify them. Any future readout computed
#: from the embedding alone needs `data`, and until that exists it does not ship as a finding.
NULL_KIND: dict[str, str] = {
    "separation": "labels",
    "composition": "labels",
}


def facts() -> frozenset:
    """Every fact name the calculus can name — DERIVED from the four tables above.

    A function rather than a fifth tuple, for the reason this whole file exists: a literal
    `FACTS = ("time", "conditions", "embedding", "clusters")` would be the same closed list
    stated a second time, and the drift would be silent — add a `STEP_NEEDS` entry for a new
    fact and the literal keeps reporting the old four.

    What it is FOR: a renderer that phrases a fact for a human has to cover exactly these
    words and no others (`narrate.MISSING_FACT`), and "exactly these" has to be computable or
    the phrase table is checked against nothing. Same relationship `watch.OUTCOMES` has to
    `narrate._OUTCOME_MARKS`.

    Measured today: `{"clusters", "conditions", "embedding", "genes", "model", "time"}`. Note
    that these are not all reachable through :func:`unmet` — that returns only facts some step
    NEEDS, which is `{"clusters", "conditions", "embedding", "genes", "model"}`; `time` is
    provided by `time-course` data and required by nothing, because `mioflow` treats a real
    clock as a preference and falls back to a diffusion pseudotime (see `STEP_NEEDS`). Both
    halves are here because this answers "what words does the calculus use", not "what can
    currently be missing".
    """
    out: set = set()
    for table in (SHAPE_PROVIDES, GROUP_PROVIDES, STEP_PRODUCES, STEP_NEEDS):
        for values in table.values():
            out.update(values)
    return frozenset(out)


#: Handle refs that reach a file with a gene axis. A `path` handle is NOT sufficient on its
#: own: `synthetic_timecourse.yaml` declares `{kind: path, ref: "synthetic:time-course"}`,
#: whose ref is a GENERATOR rather than a file, and handing that a gene axis it does not have
#: is the same class of error as keying the fact on shape.
_GENE_AXIS_SUFFIXES = (".h5ad", ".h5")


def dataset_provides(shape: str, handle: "dict | None" = None) -> frozenset:
    """The facts a dataset carries — from its SHAPE, and from its HANDLE.

    Two sources because they answer different questions. `case-control` provides `conditions`
    because that is what the experimental DESIGN means; `time-course` provides `time` the same
    way. `genes` is a property of the BYTES — see `DATASET_FACTS`.

    `handle=None` is every caller that predates this, and answers exactly as it did. That is
    the migration: a reader with a dataset declaration in hand passes its handle and sees one
    more fact, and a reader without one is no worse off than before.

    Measured on the fourteen bundled declarations, this checkout: exactly ONE provides `genes`
    — `pbmc3k`, the only real file we ship, and it says so with `provides: [genes]` rather than
    being sniffed. Of the other thirteen, twelve are `kind: manylatents`, which the engine loads
    and manyruns never sees the columns of (the same limitation `bounds.shape_of` records by
    returning None), and the thirteenth is the generated ref above. That is correct rather than
    disappointing — a swissroll has no genes.
    """
    facts = set(SHAPE_PROVIDES.get(shape, ()))
    handle = handle or {}
    # A DECLARED `provides:` WINS OVER THE SUFFIX GUESS BELOW, and is the direction this should
    # grow. Sniffing `.h5ad` is wrong in both directions — an `.h5ad` written without `var_names`
    # claims a gene axis it does not have and the step fails at RUN time, and a 10x `matrix.mtx`
    # directory carries genes and is refused — and it cannot answer anything else a caller might
    # want to know without opening the file. A declaration is cheap to read, reviewable in a
    # diff, and is the seam an agent enumerating datasets needs (it must not import a loader, let
    # alone torch, to ask what exists).
    #
    # It lives on the HANDLE rather than beside it because that is what it describes: the handle
    # points at bytes, and this is what those bytes carry. Unknown names are ignored rather than
    # refused — a typo is caught by `check_provides`, which `catalog.check_dataset` calls, not by
    # a fact lookup on a hot path. That sentence was ASPIRATION for four commits: nothing read
    # the field, so `provides: [gene]` and `provides: genes` both intersected to the empty set,
    # AND took the branch below that suppresses the suffix fallback — leaving the dataset with
    # strictly fewer facts than no declaration at all, silently. `check_provides`' docstring
    # carries the measurement.
    declared = handle.get("provides")
    if declared is not None:
        return frozenset(facts | (set(declared) & set(DATASET_FACTS)))
    ref = str(handle.get("ref") or "")
    if handle.get("kind") == "path" and ref.lower().endswith(_GENE_AXIS_SUFFIXES):
        facts.add("genes")
    return frozenset(facts)


def step_provides(step: dict) -> frozenset:
    """The facts running this step makes available to the steps after it.

    The union of what its GROUP provides (true of every executor in that group) and what its
    NAME provides (true of that algorithm only) — see `STEP_PRODUCES` for why one fact is
    keyed each way. A name absent from `STEP_PRODUCES` contributes nothing, so every recipe
    that was legal before that table existed is legal now."""
    return (frozenset(GROUP_PROVIDES.get(step.get("group"), ()))
            | frozenset(STEP_PRODUCES.get(step.get("name"), ())))


def recipe_needs(recipe: dict) -> frozenset:
    """Every fact ANY step of this recipe needs, unioned — the ORDER-BLIND upper bound.

    NOT the legality test; `unmet` is. This ignores both the position of a step and the facts
    the recipe produces for itself, so it reports `embedding` for a recipe that embeds first.
    Kept because "which facts does this recipe touch at all" is a real question (it is what
    `unmet` would return against a dataset carrying nothing), and because it makes the
    relation `unmet(r, s) <= recipe_needs(r) - dataset_provides(s)` statable — that bound is
    pinned in `tests/test_order_calculus.py`, which is what keeps this from drifting into a
    second, wrong answer to the legality question.
    """
    needs: set = set()
    for step in recipe.get("steps") or []:
        needs.update(STEP_NEEDS.get(step.get("name"), ()))
    return frozenset(needs)


# ── narrowing: the one thing that can take a fact away ───────────────────────
#
# `have` grew and never shrank, which was true while no step could remove anything. A `prep`
# step can. What a narrowing takes away is read off the tables above rather than listed: a fact
# the FRAME carries survives (fewer cells still carry their condition label), a fact a STEP
# produced does not (PHATE's diffusion operator was built over cells that are now gone, and
# `emb[keep]` would look like an embedding of the survivors without being one).
#
# The rule needs exactly one input those tables cannot supply — WHICH STEPS NARROW — and the
# table below is it.

#: `step name -> the param that must be true for it to narrow`, `None` meaning "always".
#:
#: A name list, deliberately, and the two alternatives are worse. Keyed by GROUP it would be
#: wrong: `transform` and `normalize` are `prep` steps that return a new working matrix and
#: remove nothing, so a group-keyed rule refuses `[phate, transform, mioflow]` — a composition
#: that runs. DERIVED from the executors it cannot be: whether a step narrows is a property of
#: what `prep._PREP_STEPS[name]` RETURNS (a `mask`, not an `X`), and reading that would mean
#: importing `pipeline` from this module, which this file's header records as the measured
#: mistake that dragged io/loading/mioflow/runner/steps in to obtain a 3-tuple of strings.
#:
#: What keeps it from being the product-side copy of an algorithm catalogue that
#: `GROUP_PROVIDES` and `STEP_PRODUCES` argue against at length: it says NOTHING about what a
#: filter computes, what it thresholds, or how — only that running it makes the data smaller.
#: There is no parameter here to hold in step with the engine, and a name that no recipe ever
#: uses costs nothing.
#:
#: ONE of the five names a step does not exist for. Measured this checkout, `prep._PREP_STEPS`
#: has six entries — `detect_doublets`, `filter_cells`, `filter_genes`, `filter_mito`,
#: `normalize`, `transform` — so four of the five below are wired for real and `hvg` alone is
#: outstanding (the spec's hand-off to Zach; manylatents#292 wires it). Forward-declaring it is
#: the SAFE direction of wrong — a name here that nobody writes prunes nothing, while a name
#: missing from here lets a recipe pass plan-time holding an embedding the runner has already
#: cleared, which is the plan-time/run-time disagreement `STEP_NEEDS`' `mioflow` note says this
#: calculus exists to remove.
#:
#: `detect_doublets` carries a param rather than `None` because it FLAGS by default and removes
#: only on `remove: true` (`prep._step_detect_doublets` returns an all-True mask otherwise — a
#: step that silently drops cells is the inverse of this product's thesis, which `qc.yaml`
#: states). Listing it unconditionally would refuse `[phate, detect_doublets, mioflow]`, where
#: nothing is removed at all.
NARROWING_STEPS: dict[str, "str | None"] = {
    "filter_cells": None,
    "filter_genes": None,
    "filter_mito": None,
    "hvg": None,
    "detect_doublets": "remove",
}


def step_narrows(step: dict) -> bool:
    """Does running this step remove rows or columns from the selection?

    Reads the step's declared PARAMS as well as its name, because for one step the answer is in
    them (`NARROWING_STEPS`). The recipe is the declaration, so this is not a guess about what
    the executor will do — it is what the author wrote down.

    No axis, on purpose. `filter_genes` and `hvg` remove GENES: no cell is dropped, so `emb`
    keeps one row per selected cell and nothing DESYNCS. It is cleared anyway, because the
    premise the row case is decided on is FITTING and not length — on length alone `emb[keep]`
    would be a legal repair and a row narrowing would clear nothing either. A kNN graph over
    32,738 genes is not the graph over the 2,000 that `hvg` kept. Splitting the axes would mean
    holding the fitted premise on rows and a length premise on columns inside one fold: two
    definitions of stale in one function, which is the two-homes drift this file exists to
    prevent. The cost is a refusal of `[phate, hvg, mioflow]`, which is the direction this kind
    of table is allowed to be wrong in (`STEP_PRODUCES`).

    WHAT PAYS THAT COST, corrected TWICE. An early version of this note offered
    `unmet(..., provided={"embedding"})` as the escape hatch. The correction then said `provided`
    is None on 100% of production paths — that measurement is now FALSE and is struck: the
    cutover gave `LedgerScreen.__init__` a default of `entry.obs.provides()`, and `shell.py:554`
    passes `provided=obs.provides()`, so two front doors supply it. The CONCLUSION survives the
    correction, and this is the measurement that carries it, by an AST walk over `manyruns/`
    this checkout:

      * six `unmet` call sites. Four pass no `provided` at all (`app.py:844`, `app.py:853`,
        `experiment.py:89`, `experiment.py:113`) — those hold a dataset DECLARATION and supply
        its facts through `handle=` instead.
      * the other two are the front doors, and both pass `narrate.Observation.provides()`
        (`shell.py:554` directly; `narrate.py:276` via `tui/state.py:328` ← `state.ledger` ←
        `LedgerScreen.__init__`'s default at `tui/ledger.py:225`). That method returns
        `{"genes"}` or `frozenset()` and nothing else — it reads `n_vars`, which `read_data`
        sets only on the `.h5ad`/`.h5` branch.
      * `{"embedding"}` IS built on production paths, from a caller-supplied 2-D array
        (`runner.py:1238`, `runner.py:1356`, `session.py:251`) — and every one of them hands it
        to :func:`noncanonical`, which renders a caveat and decides no legality. Not one reaches
        `unmet`.

    So no user can unlock a narrowed embedding by declaring they hold one.

    So what makes the refusal affordable is not the hatch — it is that the refusal EXPLAINS
    itself. `invalidated` separates "a filter cleared it" from "nothing produces it", and
    `narrate.CLEARED_FACT` names the filter as the cause, so the advice a user gets is to move
    it before the embedding rather than to add an embedder the recipe already has.
    """
    name = step.get("name")
    if name not in NARROWING_STEPS:
        return False
    gate = NARROWING_STEPS[name]
    return True if gate is None else bool((step.get("params") or {}).get(gate))


def frame_facts() -> frozenset:
    """Facts that come from the DATA rather than from a step, and so survive a narrowing.

    Derived from `SHAPE_PROVIDES` and `DATASET_FACTS`, never listed a second time — the drift
    this module exists to prevent. Everything a step produces (`GROUP_PROVIDES`,
    `STEP_PRODUCES`) is by construction absent from both, so the rule needs no list of what to
    clear: it keeps this and drops the rest. `tests/test_narrowing_calculus.py` asserts the two
    sets stay disjoint, because a fact appearing in both would silently make the fold monotone
    again for that one word.

    Measured today: `{"conditions", "genes", "time"}` — the `SHAPE_PROVIDES` union
    (`conditions`, `time`) plus `DATASET_FACTS`' one member, `genes`.
    """
    out: set = set()
    for values in SHAPE_PROVIDES.values():
        out.update(values)
    out.update(DATASET_FACTS)
    return frozenset(out)


def step_facts() -> frozenset:
    """Facts a STEP produces — exactly what a narrowing can take away.

    The complement of :func:`frame_facts`, derived from the two producer tables for the same
    reason that one is derived from the two data tables: the split is the data model, and
    stating either half a second time is the drift this file exists to prevent.
    `tests/test_narrowing_calculus.py` asserts the two are disjoint.

    What it is FOR: a renderer explaining a refusal has to phrase every fact a narrowing can
    clear and no others (`narrate.CLEARED_FACT`), and "exactly these" has to be computable —
    the same relationship `facts()` has to `narrate.MISSING_FACT`. Writing a cleared-phrase for
    `time` or `conditions` would be a sentence no code path can reach, because a frame fact
    survives every narrowing by construction.

    Measured today: `{"clusters", "embedding", "model"}`.
    """
    out: set = set()
    for table in (GROUP_PROVIDES, STEP_PRODUCES):
        for values in table.values():
            out.update(values)
    return frozenset(out)


#: The in-process step loop's name in the engine-keyed tables (`runner._DISPATCH`,
#: `STAND_INS`). It is NOT an engine and must never become one: it is absent from
#: `app.ENGINES`, so argparse cannot produce it, the picker cannot offer it and
#: `app._default_engine` cannot return it; and it is absent from `serving.LocalServer.SERVES`,
#: so no request can be routed to it. The leading underscore is load-bearing — it is what makes
#: a record reading `engine=_inproc` legible as "this did not come from the product".
#:
#: What it IS: `runner.run_inproc` — phate + a diffusion pseudotime computed here, with public
#: libraries only. It survives the removal of `--engine real` for exactly one reason, stated
#: plainly because it is the whole argument for keeping it: it is the only step loop CI can run.
#: `mock` writes no arrays (`runner._mock_transform` never assigns `state["emb"]`), so it cannot
#: carry the lineage, artifact, rewind or branch tests; `manylatents` needs torch, which
#: `.github/workflows/ci.yml` excludes on purpose. Delete this and ~94 assertions over the
#: SHARED loop (`_run_steps`/`_persist`/`_finalize`) stop running anywhere.
INPROC = "_inproc"

#: Which source CONVENTIONALLY provides a fact. Not a rule — a note. `unmet` decides legality
#: on whether the object exists; this decides only whether to say "that is an unusual way to
#: get it" on the way past.
#:
#: The distinction is the whole reason both exist. A step reading a raw ambient matrix as a
#: coordinate system cannot compose and is refused. A step reading coordinates a different tool
#: produced composes perfectly well and is unusual — and unusual is where new method comes
#: from, so it runs. What it must not do is run SILENTLY: a result obtained off the beaten path
#: is exactly the one whose provenance a reader needs, both to trust it and to repeat it.
#:
#: (These ten lines sat ABOVE `INPROC` until 2026-08-17, so the `#:` block documenting the
#: in-process loop opened by describing a different table. Nothing read wrong; a reader did.)
CANONICAL_PROVIDER: dict[str, str] = {
    "embedding": "latent",
}

#: A bundled step that fills the canonical role, so the note can name something runnable
#: rather than a group. Nothing depends on this being complete.
CANONICAL_EXAMPLE: dict[str, str] = {
    "latent": "phate",
}

#: `(engine, step name) -> what that engine ACTUALLY computes under that name`, for the
#: cases where it is a DIFFERENT algorithm rather than another implementation of the same one.
#:
#: This is the other half of a recipe's per-step `via:`. `via` is ADVISORY — the executor is
#: still chosen by `group` × `--engine`, unchanged — and what it buys is that a substitution
#: can be *said*. The substitution itself is a
#: property of the executor, not of the recipe, so it lives here rather than in YAML: a recipe
#: cannot know which engine will run it.
#:
#: Read from source, not from memory. `steps._INPROC_STEPS` has exactly two entries
#: (`steps.py:884-887`):
#:   `phate`   — `_step_phate` runs the public `phate` package (`steps.py:62-87`) and
#:               manylatents runs `algorithms/latent/phate.py`. Both are PHATE, so this is
#:               NOT a stand-in and is deliberately absent: a caveat fired here would spend
#:               the channel on a non-event, and a channel that cries wolf stops being read.
#:   `mioflow` — `_step_mioflow` (`steps.py:137-149`) computes a diffusion pseudotime on a kNN
#:               graph (`_diffusion_pseudotime`, `steps.py:106-134`); its own docstring calls
#:               itself "a real MIOFlow stand-in for the in-process loop", while the manylatents
#:               engine runs the trained neural-ODE flow. Different algorithm, same step name.
#:
#: One entry, because one substitution exists. Rows for substitutions nobody has written
#: would be the registry-ahead-of-the-second-tool mistake CLAUDE.md names.
STAND_INS: dict[tuple, str] = {
    (INPROC, "mioflow"): "diffusion-pseudotime",
}


def standins(recipe: dict, engine: Optional[str]) -> list:
    """Steps this engine ran with a DIFFERENT algorithm than the recipe names, as sentences.

    The rule reads `STAND_INS` first and the recipe's `via:` second, and that order matters.
    `via` cannot say "this is a stand-in" — it says which implementation is canonical — so a
    recipe that declares no `via` still gets the note: running diffusion pseudotime under the
    step name `mioflow` is a substitution whether or not anyone wrote it down. What `via` can
    do is SUPPRESS the note, by naming this engine as the one the recipe wanted; that case is
    not a substitution at all, it is the recipe getting what it asked for.

    Deliberately not a blanket `via != engine` test. `phate` under `vocab.INPROC` is PHATE, so
    a blanket test would attach a caveat to a run where nothing was substituted — and the
    caveat list is read by a human deciding whether to trust a number.
    """
    notes: list = []
    if not engine:
        return notes
    for step in recipe.get("steps") or []:
        name = step.get("name")
        what = STAND_INS.get((engine, name))
        if what is None:
            continue
        via = step.get("via")
        if via == engine:  # the recipe asked for THIS engine's implementation; nothing stood in
            continue
        names_it = f"the {via} {name} the recipe names" if via else f"{name} itself"
        notes.append(
            f"{name}: ran on engine={engine}'s {what} stand-in, not {names_it} — "
            f"legal and not the cited method, running anyway"
        )
    return notes


def noncanonical(recipe: dict, shape: str = "unknown",
                 provided: "frozenset | set | tuple | None" = None,
                 engine: Optional[str] = None,
                 handle: "dict | None" = None) -> list:
    """Legal compositions that took an unusual route to a fact, as sentences for a human.

    Same fold as :func:`unmet`, tracking WHERE each fact came from rather than whether it
    arrived. A need met from somewhere other than `CANONICAL_PROVIDER` produces one line. An
    UNMET need produces nothing here — that is `unmet`'s job, and reporting it twice in two
    vocabularies is how the two answers drift apart.

    `engine`, when given, also appends :func:`standins` — a step whose executor is a different
    algorithm from the one its name and `via:` designate. Same class of event as the fact
    notes above (*legal, unusual, running anyway*) and therefore the SAME list: "that mismatch
    should go through the caveat
    channel that already exists, not a new one". Defaults to None so every caller that has no
    engine in hand keeps its current answer rather than being told to invent one.

    Returns `[]` for every bundled recipe on its declared shape when no engine is named, and
    for every bundled recipe on `engine=manylatents`. On `vocab.INPROC` exactly the three
    recipes that name `mioflow` — `cflows`, `traced`, `sandbox` — return one line each, the
    stand-in, which is the point: this is quiet until something genuinely unusual happens."""
    have: dict = {f: "the dataset" for f in dataset_provides(shape, handle)}
    have.update({f: "the caller" for f in (provided or ())})
    keep: frozenset = frame_facts() | frozenset(provided or ())
    notes: list = []
    for step in recipe.get("steps") or []:
        name = step.get("name")
        for fact in STEP_NEEDS.get(name, ()):
            source = have.get(fact)
            canonical = CANONICAL_PROVIDER.get(fact)
            if source is None or canonical is None or source == canonical:
                continue
            example = CANONICAL_EXAMPLE.get(canonical)
            how = f"a {canonical} step" + (f" (e.g. {example})" if example else "")
            notes.append(
                f"{name}: {fact} came from {source}, not from {how} — "
                f"legal and unusual, running anyway"
            )
        # THE SAME TWO LINES AS `unmet`, and they have to be. This fold answers "where did
        # each fact come from" over the walk `unmet` answers "did it arrive" over; if only one
        # of them subtracted, this one would describe a fact as "legal and unusual, running
        # anyway" for a recipe the other has already refused as illegal. That is the exact
        # failure this docstring names — one event reported in two vocabularies that then
        # drift. The re-seed on the next line is this fold's own extra: a fact the CALLER holds
        # survives, but the run's own copy of it did not, so the source reverts to the caller.
        # Saying nothing there would credit the fact to a step the calculus just declared dead.
        if step_narrows(step):
            have = {f: source for f, source in have.items() if f in keep}
            have.update({f: "the caller" for f in (provided or ())})
        for fact in step_provides(step):
            have[fact] = step.get("group")
    notes.extend(standins(recipe, engine))
    return notes


def unmet(recipe: dict, shape: str, provided: "frozenset | set | tuple | None" = None,
          handle: "dict | None" = None) -> frozenset:
    """What this recipe needs that nothing available at that point can provide. Empty = legal.

    The same subtraction it has always been, folded ALONG the recipe rather than unioned over
    it: each step's needs are checked against the facts on hand AT ITS OWN POSITION — what the
    dataset carries, plus what the steps BEFORE it produced, MINUS what a narrowing has taken
    away. A step never satisfies its own need, which is the entire point.

    `have` grows and, since `prep` landed, can also shrink — see `step_narrows` and
    `frame_facts`. One rule, derived from the tables that already exist: a fact the frame
    carries survives a narrowing, a fact a step produced does not, because the cells it was
    fitted over are gone. Measured consequence for the readers below, re-derived this checkout
    now that eight of the eleven bundled recipes DO declare `prep` steps: still none. Seven of
    those eight declare only `normalize`/`transform`, which are `prep` and do not narrow; the
    eighth, `preprocess`, is three narrowing filters followed by those two and produces no fact
    at any position, so `have` never holds anything for a narrowing to take. Every answer is
    byte-for-byte what it was — pinned in `test_order_calculus.py` recipe × shape and in
    `test_narrowing_calculus.py` on each recipe's declared suit.

    Signature and meaning are unchanged for the three production readers (`experiment.plan`
    and `experiment.skipped_cells`, `shell._explore`'s legal-recipe filter, `app`'s coverage
    table): a frozenset of fact names, empty means legal. What changes is that an ILLEGALLY
    ORDERED recipe is no longer silently legal. Every bundled recipe is correctly ordered, so
    every currently-legal cell stays legal — verified by re-running the suite, and pinned in
    `tests/test_order_calculus.py` recipe by recipe so a future edit to either table has to
    face those assertions.

    `provided` is what the CALLER already has in hand — a precomputed embedding, a graph built
    by another tool, anything supplied rather than produced here. It matters because the test
    this function applies is meant to be STRUCTURAL, not canonical: the question is whether the
    object a step consumes exists, never whether a blessed step produced it.

    The first version got that wrong. `GROUP_PROVIDES` made `embedding` a fact only a `latent`
    step could create, so `{steps: [mioflow]}` on a precomputed embedding was reported ILLEGAL
    — refusing a composition that is mathematically fine because it was not the conventional
    route. That is the wrong kind of refusal: an unconventional-but-well-typed composition is
    where new method comes from, and a calculus that prunes it is enforcing convention while
    claiming to enforce type. What must stay refused is the composition that cannot COMPOSE —
    `[mioflow, phate]` with no embedding anywhere, where the trajectory step would read a raw
    ambient matrix as a coordinate system. That case is still caught, because the fact is
    absent from BOTH sources rather than merely from the step list.

    The fold itself is `_walk`, which this and :func:`invalidated` are the two projections of.
    Two answers about one recipe — what is missing, and whether a narrowing is why — computed
    by one walk, because computing them separately is how a verdict and its reason come to
    describe different recipes.
    """
    return _walk(recipe, shape, provided, handle)[0]


def invalidated(recipe: dict, shape: str,
                provided: "frozenset | set | tuple | None" = None,
                handle: "dict | None" = None) -> frozenset:
    """Of what :func:`unmet` refuses, the facts that WERE on hand and a narrowing took away.

    A second PROJECTION of the one fold, never a second fold — `unmet` and this both return a
    slice of `_walk`, so they cannot disagree about the same recipe. `invalidated(...)` is
    always a subset of `unmet(...)`, pinned by test: a reason attached to no refusal would be
    the two-vocabularies drift `noncanonical`'s docstring records.

    WHY THE DISTINCTION IS WORTH DRAWING. `[phate, filter_cells, mioflow]` and `[mioflow,
    phate]` both come back `{"embedding"}` from `unmet`, and they are opposite problems: the
    second never lays the cells out, the first lays them out and then removes the cells they
    were fitted over. Every surface that explains a refusal used to say the second thing about
    both — "nothing here lays the cells out", about a recipe whose FIRST step is phate — so the
    advice it implied ("add a latent step") was already taken and changed nothing. The fix for
    the first is to move the filter before the embedding, or to re-embed after it, and a
    sentence can only say so if the calculus tells it which case this is
    (`narrate.MISSING_FACT` vs `narrate.CLEARED_FACT`).

    Empty for every bundled recipe on every member of `SHAPES`, measured this checkout — and no
    longer for the reason it once was ("no bundled recipe has a `prep` step", false since
    `preprocess` and the scRNA preamble landed). It is empty because a narrowing can only clear
    a fact something already produced, and the one bundled recipe whose filters narrow
    (`preprocess`) declares no producing step at all.
    """
    return _walk(recipe, shape, provided, handle)[1]


def _walk(recipe: dict, shape: str,
          provided: "frozenset | set | tuple | None",
          handle: "dict | None" = None) -> tuple[frozenset, frozenset]:
    """The fold `unmet` and `invalidated` are the two projections of — `(missing, cleared)`.

    ONE walk with two readers, rather than two walks: `unmet` decides legality and
    `invalidated` says which half of the refusal is an ORDERING problem, and a caller reads
    both to phrase one sentence. Computing them separately is how a verdict and its reason come
    to describe different recipes.

    `cleared` is a subset of `missing` by construction — a fact only enters it at the moment it
    enters `missing` — so a reason is never reported for a need that was met.
    """
    have: set = set(dataset_provides(shape, handle)) | set(provided or ())
    keep: frozenset = frame_facts() | frozenset(provided or ())
    missing: set = set()
    cleared: set = set()
    lost: set = set()      # facts a narrowing has already taken away, for the reason half
    for step in recipe.get("steps") or []:
        for fact in STEP_NEEDS.get(step.get("name"), ()):
            if fact in have:
                continue
            missing.add(fact)
            if fact in lost:
                cleared.add(fact)
        # A narrowing invalidates everything a STEP produced: the cells those results were
        # fitted over are gone. What the FRAME carries is untouched — fewer cells still have a
        # gene axis and still carry their condition label. `provided` is held too: a caller who
        # handed in an embedding is asserting they hold it, and this function does not get to
        # decide it went stale. Checked AFTER this step's own needs and before its produces, so
        # a filter's own preconditions are read against the facts it actually ran on and a step
        # that both narrows and produces (spec §3 makes `hvg` one) keeps its own selection.
        # Both orderings are pinned in `tests/test_narrowing_calculus.py`, because with today's
        # tables — no `NARROWING_STEPS` name appears in `STEP_NEEDS`, and all five are `prep`,
        # a group no producer table mentions — every permutation of these three lines gives the
        # same answer on every recipe in the repo.
        # `noncanonical` folds the same two lines; if only one of them did, the two would
        # report contradictory answers about the same recipe.
        if step_narrows(step):
            lost |= have - keep
            have &= keep
        have |= step_provides(step)
    return frozenset(missing), frozenset(cleared)
