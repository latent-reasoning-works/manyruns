"""`genes` — the fact that could not be declared until the gene axis survived the load.

`vocab.py`'s own note says where the provider belongs: *"with the DATASET declaration, not
here — a `handle: {kind: path, ref: …h5ad}` dataset provides `genes`, the way a `case-control`
shape provides `conditions`."* Carrying a gene axis is a property of the FILE, not of the
shape: an `.h5ad` of any shape has one, a synthetic point cloud of the same shape has none.
Keying it on shape would have returned `{genes}` on all six shapes and pruned the two recipes
that run on the one real dataset in the repo.

**SUPPLY LANDED FIRST; DEMAND HAS NOW LANDED TOO.** This file shipped with the provider and
the two filters that consume it, and said `qc`/`rank_genes` would declare `("genes",)` in the
same commit that stops `loading` from PCA-ing the axis away — because until then not one
declared dataset could satisfy the need. The cutover (#54) is that commit: `_anndata_matrix`
returns `adata.X` untouched, `configs/dataset/pbmc3k.yaml` declares `provides: [genes]`, and
`vocab.STEP_NEEDS` now carries `qc: ("genes",)` and `rank_genes: ("clusters", "genes")`. So
the refusal and its cause arrived together, as promised, and the two tests below that pinned
the pre-cutover sweep are re-framed rather than renumbered — see their docstrings.
"""
from __future__ import annotations

import pytest

from manyruns import catalog, vocab

H5AD = {"kind": "path", "ref": "data/pbmc3k_raw.h5ad"}
GENERATED = {"kind": "path", "ref": "synthetic:time-course"}
ENGINE_REF = {"kind": "manylatents", "ref": "swissroll"}


# ── the provider ─────────────────────────────────────────────────────────────
def test_a_path_handle_to_an_h5ad_provides_the_gene_axis():
    """The provider the note prescribed, in the place it prescribed."""
    assert "genes" in vocab.dataset_provides("clusters", H5AD)


def test_the_shape_is_irrelevant_to_whether_there_are_genes():
    """The whole argument for keying on the handle. The same file provides `genes` whatever
    experimental design it holds, and `SHAPE_PROVIDES` keeps answering the design question."""
    for shape in vocab.SHAPES:
        assert "genes" in vocab.dataset_provides(shape, H5AD), shape


def test_an_engine_dataset_provides_no_gene_axis():
    """A named manylatents dataset is loaded by the engine and manyruns never sees its
    columns — the same limitation `bounds.shape_of` records by returning None for one. It
    declares nothing rather than guessing."""
    assert "genes" not in vocab.dataset_provides("manifold", ENGINE_REF)


def test_a_generated_path_ref_is_not_a_file_and_carries_no_genes():
    """`synthetic:time-course` is a `path` handle whose ref is a GENERATOR, not a file
    (configs/dataset/synthetic_timecourse.yaml says so). Keying on `kind == "path"` alone
    would have handed it a gene axis it does not have."""
    assert "genes" not in vocab.dataset_provides("time-course", GENERATED)


def test_the_old_single_argument_call_still_answers():
    """Every existing caller passes a shape and nothing else. Widening a signature must not
    make them wrong — it makes them unable to see one fact, which is what they see today."""
    assert vocab.dataset_provides("case-control") == frozenset({"conditions"})
    assert vocab.dataset_provides("time-course") == frozenset({"time"})


def test_genes_is_a_frame_fact_so_no_narrowing_can_take_it_away():
    """The point of the frame. Filtering cells leaves every surviving cell with all its gene
    names; filtering genes leaves the axis itself. `frame_facts` unions `DATASET_FACTS`, so
    this follows from the declaration rather than from a second list."""
    assert "genes" in vocab.frame_facts()
    assert "genes" not in vocab.step_facts(), "nothing produces genes; the data carries them"


# ── the consumers that land now ──────────────────────────────────────────────
def test_the_two_new_filters_declare_the_gene_axis_they_read():
    """`filter_mito` prefix-matches `MT-` against gene NAMES and `filter_genes` counts cells
    per gene. Both are refused on data with no gene axis rather than matching no columns and
    reporting that they filtered nothing — a report indistinguishable from a dataset with no
    dying cells."""
    assert vocab.STEP_NEEDS["filter_mito"] == ("genes",)
    assert vocab.STEP_NEEDS["filter_genes"] == ("genes",)


def test_filter_mito_is_refused_on_a_synthetic_point_cloud():
    """A 150 × 3 swissroll has no genes, so the refusal is at plan time, where a search over
    recipes has to get its answer."""
    recipe = {"name": "t", "steps": [
        {"name": "filter_mito", "group": "prep", "params": {"max_pct": 5}},
        {"name": "phate", "group": "latent", "params": {}}]}

    assert vocab.unmet(recipe, "manifold") == frozenset({"genes"})


def test_filter_mito_is_legal_on_a_dataset_whose_handle_carries_genes():
    """The other half. A refusal that fires on correct data is worse than none — the
    `lid <= final_dim` bound `configs/metrics/default.yaml` records removing."""
    recipe = {"name": "t", "steps": [
        {"name": "filter_mito", "group": "prep", "params": {"max_pct": 5}},
        {"name": "phate", "group": "latent", "params": {}}]}

    assert vocab.unmet(recipe, "clusters", handle=H5AD) == frozenset()


def test_a_missing_gene_axis_is_not_reported_as_an_ordering_problem():
    """`invalidated` is the half of a refusal that says a NARROWING caused it. No narrowing
    can clear a frame fact, so a `genes` refusal must never be phrased as "move the filter" —
    the two-sentences distinction `narrate.CLEARED_FACT` exists to draw."""
    recipe = {"name": "t", "steps": [
        {"name": "filter_genes", "group": "prep", "params": {"min_cells": 3}}]}

    assert vocab.unmet(recipe, "manifold") == frozenset({"genes"})
    assert vocab.invalidated(recipe, "manifold") == frozenset()


# ── the readers that supply it ───────────────────────────────────────────────
def test_the_sweep_prices_the_gene_demand_and_every_pruned_cell_is_named():
    """PREMISE RE-FRAMED AT THE CUTOVER. This test used to assert the sweep was UNCHANGED,
    because `qc`/`rank_genes` still abstained from declaring `genes` — the split that let the
    refusal wait for its cause. The cutover landed both halves, so "unchanged" is no longer
    the property worth protecting. What is: the refusal costs exactly what the data cannot
    supply, and not one cell more — a precondition that fires on correct data is the
    `lid <= final_dim` bound again (`configs/metrics/default.yaml` records removing it).

    So the sweep is asserted as its BREAKDOWN rather than as one total: an unexplained cell
    is the failure, and a bare count cannot tell "qc is pruned on a swissroll" (correct) from
    "qc is pruned on pbmc3k" (the bug this whole fact exists to prevent).

    Measured 2026-08-16, this checkout: 11 recipes x 14 datasets = 154 cells (was 140 — the
    recipe count went 10 -> 11 when `preprocess` was declared), 53 pruned (was 40).
    The 13 new prunings are ALL `preprocess` on the 13 synthetics: `filter_genes` and
    `filter_mito` read gene names, so the scRNA preamble is legal on `pbmc3k` and on nothing
    else we ship. The recipe adds no cell to any other row — it is the third recipe refused for
    exactly one fact, not a third KIND of refusal, which is what the breakdown below shows and
    a total would have hidden.
    """
    recipes = [catalog.load_recipe(n) for n in catalog.discover_recipes()]
    datasets = [catalog.load_dataset(n) for n in catalog.discover_datasets()]
    pruned: dict[tuple[str, tuple[str, ...]], int] = {}
    for r in recipes:
        for d in datasets:
            missing = vocab.unmet(r, d.get("shape", "unknown"), handle=d.get("handle"))
            if missing:
                pruned[(r["name"], tuple(sorted(missing)))] = 1 + pruned.get(
                    (r["name"], tuple(sorted(missing))), 0)

    assert len(recipes) * len(datasets) == 154
    assert pruned == {
        ("contrast", ("conditions",)): 14,  # no bundled dataset declares `case-control`
        ("markers", ("genes",)): 13,        # every dataset but pbmc3k: 13 synthetic clouds
        ("qc", ("genes",)): 13,             # same 13 — pbmc3k is the only `provides: [genes]`
        ("preprocess", ("genes",)): 13,     # same 13, and the same one fact
    }
    assert sum(pruned.values()) == 53


@pytest.mark.parametrize("name", catalog.discover_recipes())
def test_every_bundled_recipe_is_legal_on_its_declared_shape_given_data_that_carries_genes(name):
    """The regression that matters most, restated for a fact SHAPE CANNOT SUPPLY.

    Same protection as before — this change must never newly prune something that runs today,
    checked per recipe so a failure names it — but "runs today" now has to name the data. Since
    the cutover `markers` and `qc` need `genes`, and `genes` is a DATASET fact: `SHAPE_PROVIDES`
    never supplies it (that is the whole argument at the top of this file), so `unmet(recipe,
    suit)` with no handle reports `{genes}` for those two on any shape. Asserting an empty set
    there would be asserting that no recipe may need a fact about the bytes, which is the
    opposite of what this file establishes.

    Two assertions, because the pruning has to be right in both directions:
      1. with a gene-carrying handle every bundled recipe is legal on every suit it declares —
         nothing is illegal EVERYWHERE, the failure mode a hard refusal risks;
      2. with no handle the only fact any of them can be missing is `genes` — never `embedding`,
         `clusters`, `model` or `conditions`, i.e. never an ORDERING or shape mistake, which is
         what the original assertion was really guarding and what no data can fix.
    """
    recipe = catalog.load_recipe(name)
    for suit in (recipe.get("suits") or ["unknown"]):
        assert vocab.unmet(recipe, suit, handle=H5AD) == frozenset(), f"{name} on {suit}"
        assert vocab.unmet(recipe, suit) <= frozenset({"genes"}), f"{name} on {suit}"


def test_the_observation_reports_a_gene_axis_for_an_h5ad_and_not_for_a_folder():
    """The shell's supplier. `experiment` and `app` hold a dataset DECLARATION and read its
    handle; the shell holds a loaded `Observation` instead, so the fact has to be reachable
    from that too — otherwise dropping real scRNA into the app would refuse the very steps
    that need it."""
    from manyruns.narrate import Observation

    assert Observation(shape="clusters", n_vars=32738).provides() == frozenset({"genes"})
    assert Observation(shape="manifold").provides() == frozenset()
