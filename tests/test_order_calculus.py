"""The precondition calculus must see step ORDER, and must not have changed anything else.

`vocab.unmet` used to be `union(STEP_NEEDS over steps) - dataset_provides(shape)`. A union
cannot see position, so the recipe `[mioflow, phate]` came back `frozenset()` — LEGAL — on
every one of the six shapes (measured against `git show HEAD:manyruns/vocab.py` before the
change). Commit df478cc had closed the same hole imperatively, inside the engines; until
this, nothing refused the recipe at PLAN time, which is where a legality function has to
answer — `experiment.plan`, `shell._explore`'s menu and `app`'s coverage table all ask
"is this cell legal" precisely because they cannot afford to run it to find out.

The fix is a trim, not a mechanism: facts reach a step from two sources — the dataset
(`SHAPE_PROVIDES`) and the steps that already ran (`GROUP_PROVIDES`) — and `unmet` folds
along the recipe instead of unioning over it. Same subtraction, checked at each position.

What these tests pin, in order of what would hurt most if it broke:
  1. the backwards recipe is illegal AND names the missing fact;
  2. every bundled recipe keeps its exact legality on every shape — the whole risk of this
     change is a recipe that used to run being silently pruned. The table below was measured
     BEFORE the change (all 18 cells) and is reproduced here as a golden table, so a future
     edit to either vocabulary table has to face it;
  3. a step never satisfies its own need;
  4. producers are keyed by GROUP, so a `latent` step named something other than `phate`
     still satisfies the need — that is not a detail, it is the reason a currently-runnable
     manylatents recipe was not newly pruned.

Dependency-free by construction: `vocab` imports nothing but `re`, and `experiment.plan` is
a loop over dicts. No learner, no engine, no data — legality is answerable without any of
them, which is the property the whole calculus exists to have.
"""
from __future__ import annotations

from manyruns import catalog
from manyruns.vocab import (
    GROUP_PROVIDES,
    SHAPES,
    STEP_NEEDS,
    STEP_PRODUCES,
    dataset_provides,
    recipe_needs,
    unmet,
)
from public_safety import forbidden_count

PHATE = {"name": "phate", "group": "latent", "params": {"n_components": 3}}
MIOFLOW = {"name": "mioflow", "group": "lightning", "params": {}}

#: The recipe from commit df478cc's finding. On the in-process loop it reported `outcome=ok` for
#: every step, `ok=True` for the run, and `pseudotime_range = [0.0, 1.0]` over data no
#: embedding had touched. `unmet` called it legal.
BACKWARDS = {"name": "backwards", "steps": [MIOFLOW, PHATE]}
FORWARD = {"name": "forward", "steps": [PHATE, MIOFLOW]}

#: Measured with `git show HEAD:manyruns/vocab.py` loaded side by side with the new module,
#: over all three bundled recipes × all six shapes: 18 of 18 cells identical. Written out
#: rather than computed so that a regression has to disagree with a number, not with a
#: reimplementation of the thing under test.
BUNDLED_LEGALITY = {
    # The five below were added with the recipe catalogue. Each row is MEASURED, not intended:
    # the guard exists so that adding a recipe forces someone to look at what the calculus now
    # says about it, rather than discovering later that a menu entry is legal on nothing —
    # which is what `contrast` has always been. Three blocks are `set()`-free now, not one:
    # `markers` and `qc` joined it when they declared a need for `genes` (see their rows).
    ('archetypes', 'manifold'): set(),
    ('archetypes', 'clusters'): set(),
    ('archetypes', 'time-course'): set(),
    ('archetypes', 'case-control'): set(),
    ('archetypes', 'single'): set(),
    ('archetypes', 'unknown'): set(),
    ('cluster', 'manifold'): set(),
    ('cluster', 'clusters'): set(),
    ('cluster', 'time-course'): set(),
    ('cluster', 'case-control'): set(),
    ('cluster', 'single'): set(),
    ('cluster', 'unknown'): set(),
    # `markers` and `qc` each moved from `set()` to `{'genes'}` on all six shapes — twelve
    # cells, the only rows the cutover (#54) moved. `qc` and `rank_genes` now declare
    # `("genes",)` in `STEP_NEEDS`, and `genes` is a DATASET fact — no `SHAPE_PROVIDES` row
    # supplies it, so no shape can. This is the one legality change the cutover made, and it is
    # deliberate: a differential-expression readout over a matrix with no gene axis names
    # nothing, which is what these two used to do. They are legal only where a handle declares
    # `provides: [genes]` — `configs/dataset/pbmc3k.yaml` is the first and only bundled dataset
    # that does, and none of the 13 synthetics do.
    ('markers', 'manifold'): {'genes'},
    ('markers', 'clusters'): {'genes'},
    ('markers', 'time-course'): {'genes'},
    ('markers', 'case-control'): {'genes'},
    ('markers', 'single'): {'genes'},
    ('markers', 'unknown'): {'genes'},
    ('pseudotime', 'manifold'): set(),
    ('pseudotime', 'clusters'): set(),
    ('pseudotime', 'time-course'): set(),
    ('pseudotime', 'case-control'): set(),
    ('pseudotime', 'single'): set(),
    ('pseudotime', 'unknown'): set(),
    # `{'genes'}` on all six for the reason spelled out at `markers` above.
    ('qc', 'manifold'): {'genes'},
    ('qc', 'clusters'): {'genes'},
    ('qc', 'time-course'): {'genes'},
    ('qc', 'case-control'): {'genes'},
    ('qc', 'single'): {'genes'},
    ('qc', 'unknown'): {'genes'},

    ("cflows", "manifold"): set(),
    ("cflows", "clusters"): set(),
    ("cflows", "time-course"): set(),
    ("cflows", "case-control"): set(),
    ("cflows", "single"): set(),
    ("cflows", "unknown"): set(),
    ("contrast", "manifold"): {"conditions"},
    ("contrast", "clusters"): {"conditions"},
    ("contrast", "time-course"): {"conditions"},
    ("contrast", "case-control"): set(),
    ("contrast", "single"): {"conditions"},
    ("contrast", "unknown"): {"conditions"},
    ("embed", "manifold"): set(),
    ("embed", "clusters"): set(),
    ("embed", "time-course"): set(),
    ("embed", "case-control"): set(),
    ("embed", "single"): set(),
    ("embed", "unknown"): set(),

    # `traced` — `cflows` plus a `probe` step. MEASURED on this checkout, legal on all six, and
    # the reason is the one the probe was designed around: `sample_trajectories` declares
    # `STEP_NEEDS = ("model",)`, `mioflow` declares `STEP_PRODUCES = ("model",)`, and the fold
    # sees the second before the first. Identical to `cflows`'s row, which is the point — adding
    # a readout OF a trajectory must not change whether the trajectory is legal.
    ("traced", "manifold"): set(),
    ("traced", "clusters"): set(),
    ("traced", "time-course"): set(),
    ("traced", "case-control"): set(),
    ("traced", "single"): set(),
    ("traced", "unknown"): set(),

    # `sandbox` — the declared-but-unimplemented fixture (`pipeline/stubs.py`). Legal on all six,
    # and that is the property it is built for: a recipe REFUSED at plan time never reaches the
    # step pane, which is the surface it exists to drive. Its two real steps produce the `model`
    # its stubs need, which is why the stubs are not first.
    ("sandbox", "manifold"): set(),
    ("sandbox", "clusters"): set(),
    ("sandbox", "time-course"): set(),
    ("sandbox", "case-control"): set(),
    ("sandbox", "single"): set(),
    ("sandbox", "unknown"): set(),

    # `preprocess` — the scRNA preamble as a recipe. `{'genes'}` on all six, MEASURED
    # 2026-08-16, and unlike `markers`/`qc` above these rows did not MOVE: the recipe was born
    # here, because `filter_genes` and `filter_mito` have declared `("genes",)` since they were
    # wired. A shape can never supply the fact, so the only cell where this recipe is legal is
    # against a handle declaring `provides: [genes]` — `configs/dataset/pbmc3k.yaml`, one of the
    # 14 bundled datasets. That is the intended reading and not a gap: a filter that
    # prefix-matches `MT-` over a nameless matrix reports "removed 0 genes", which is the
    # sentence a dataset with nothing to remove also produces.
    ("preprocess", "manifold"): {"genes"},
    ("preprocess", "clusters"): {"genes"},
    ("preprocess", "time-course"): {"genes"},
    ("preprocess", "case-control"): {"genes"},
    ("preprocess", "single"): {"genes"},
    ("preprocess", "unknown"): {"genes"},
}


# ── 1. the hole ──────────────────────────────────────────────────────────────
def test_the_backwards_recipe_is_illegal_on_every_shape():
    """THE regression. `[mioflow, phate]` returned `frozenset()` for all six shapes."""
    for shape in SHAPES:
        assert unmet(BACKWARDS, shape) == {"embedding"}, (
            f"[mioflow, phate] is legal on {shape!r} — a trajectory step with no preceding "
            "embedding, which is how a pseudotime over raw input gets published"
        )


def test_the_unmet_fact_is_NAMED_not_merely_counted():
    """`plan`/`skipped_cells`/`app`'s coverage table all print what is missing. A refusal
    that says only "illegal" cannot be acted on — the author has to be told which fact."""
    assert sorted(unmet(BACKWARDS, "time-course")) == ["embedding"]


def test_the_forward_recipe_is_legal_on_every_shape():
    """The other direction, and the one that makes the test above mean something: the rule
    must key on ORDER, not on the mere presence of `mioflow` in the recipe."""
    for shape in SHAPES:
        assert unmet(FORWARD, shape) == frozenset(), f"cflows order wrongly pruned on {shape}"


def test_a_step_cannot_satisfy_its_OWN_need():
    """A step's needs are checked BEFORE its own produces are folded in.

    Guards the off-by-one that makes the fold vacuous: fold produces first and the backwards
    recipe becomes legal again, because `mioflow` would hand itself the embedding. Written
    with a mislabelled `mioflow` (group `latent`, so it does produce) because that is the
    only single-step recipe that can tell the two orderings apart."""
    self_feeding = {"name": "x", "steps": [{"name": "mioflow", "group": "latent"}]}
    assert unmet(self_feeding, "manifold") == {"embedding"}


# ── 2. nothing else moved ────────────────────────────────────────────────────
def test_every_bundled_recipe_keeps_its_exact_legality_on_every_shape():
    """The regression guard for the three production readers (`experiment.plan` and
    `experiment.skipped_cells`, `shell._explore`, `app`'s coverage table). Order-awareness
    is only safe if it prunes nothing that used to run; 18/18 cells were measured identical
    before and after."""
    for name in catalog.discover_recipes():
        recipe = catalog.load_recipe(name)
        for shape in SHAPES:
            expected = BUNDLED_LEGALITY.get((name, shape))
            assert expected is not None, (
                f"new bundled recipe {name!r}: add its measured legality to BUNDLED_LEGALITY"
            )
            assert unmet(recipe, shape) == expected, (
                f"{name} on {shape}: legality changed from the measured baseline"
            )


def test_a_recipe_of_analysis_steps_still_reports_its_DATA_need():
    """The data axis is untouched: `contrast` still needs `conditions` and still gets them
    only from a `case-control` dataset. An ordered fold that dropped the dataset's own facts
    would show up here and nowhere in the order tests."""
    contrast = catalog.load_recipe("contrast")
    assert unmet(contrast, "manifold") == {"conditions"}
    assert unmet(contrast, "case-control") == frozenset()


# ── 3. why producers are keyed by group ──────────────────────────────────────
def test_any_latent_step_satisfies_the_embedding_need_not_just_phate():
    """`catalog.check_recipe` validates a step's `group` against the closed `STEP_GROUPS` but
    leaves its `name` OPEN, and `runner._ml_latent` assigns `state["emb"]` for any latent
    step — so a manylatents recipe may legitimately name another embedder. Keying producers
    by name would have made this recipe illegal: a currently-runnable recipe newly pruned."""
    other_embedder = {"name": "x", "steps": [{"name": "umap", "group": "latent"}, MIOFLOW]}
    assert unmet(other_embedder, "manifold") == frozenset()


def test_a_step_with_no_group_produces_nothing():
    """`check_recipe` rejects a groupless step and the runner skips it ("never guessed at"),
    so it must not silently satisfy a downstream need either. Three vocabularies agreeing."""
    groupless = {"name": "x", "steps": [{"name": "phate"}, MIOFLOW]}
    assert unmet(groupless, "manifold") == {"embedding"}


# ── 4. the two tables cannot drift apart ─────────────────────────────────────
def test_every_declared_need_is_providable_by_something():
    """A typo in either table is silent in the worst direction: a need no source can ever
    supply prunes its recipe on every shape forever, and reads exactly like a deliberate
    refusal. Derived from the existing tables rather than a fourth word list — a closed list
    of fact names would be one more home for the same vocabulary, which is the mistake this
    module's docstring opens by describing.

    THREE sources now, not two. `STEP_PRODUCES` joined the derivation when `cluster_quality`
    and `rank_genes` declared a need for `clusters`, which no dataset SHAPE and no step GROUP
    can supply — a partition is produced by one named algorithm (`leiden`) and by nothing
    else. Reading the third table here rather than special-casing that one fact is what keeps
    this assertion able to fire: it still fails for a need nothing anywhere provides, which is
    the silent-worst-direction failure it was written for."""
    from manyruns.vocab import DATASET_FACTS, SHAPE_PROVIDES

    providable = {f for facts in SHAPE_PROVIDES.values() for f in facts}
    providable |= {f for facts in GROUP_PROVIDES.values() for f in facts}
    providable |= {f for facts in STEP_PRODUCES.values() for f in facts}
    # FOUR sources now. `DATASET_FACTS` joined when `filter_mito` and `filter_genes` declared
    # `genes`, which no SHAPE can supply — carrying a gene axis is a property of the file, so
    # it comes from the dataset's `handle` (`vocab.dataset_provides`). Reading the fourth table
    # rather than special-casing that one fact is what keeps this able to fire.
    providable |= set(DATASET_FACTS)
    for step, needs in STEP_NEEDS.items():
        for fact in needs:
            assert fact in providable, (
                f"{step!r} needs {fact!r}, which no dataset shape, no step group and no named "
                "step provides — that recipe is unrunnable on every shape, and looks like a "
                "refusal"
            )


def test_unmet_is_bounded_by_the_order_blind_answer():
    """`recipe_needs` survives as the ORDER-BLIND upper bound, and this is the relation that
    keeps it from being mistaken for the legality test: what an ordered fold reports is
    always a subset of what the union reported, because producing a fact can only ever
    remove it from the missing set."""
    recipes = [catalog.load_recipe(n) for n in catalog.discover_recipes()]
    recipes += [BACKWARDS, FORWARD]
    for recipe in recipes:
        for shape in SHAPES:
            assert unmet(recipe, shape) <= recipe_needs(recipe) - dataset_provides(shape)


def test_legality_needs_no_engine_no_data_and_no_private_stack():
    """The property the calculus exists to have: answerable without running anything. This
    is also what lets `plan` prune a sweep before spending a single cell of compute."""
    import sys

    before = set(sys.modules)
    for name in catalog.discover_recipes():
        for shape in SHAPES:
            unmet(catalog.load_recipe(name), shape)
    new = {m.split(".")[0] for m in set(sys.modules) - before}
    forbidden = sum(forbidden_count(name) for name in new)
    assert forbidden == 0, "answering legality imported a forbidden module"
    compute = len(new & {"manylatents", "manyagents", "torch"})
    assert compute == 0, "answering legality imported the compute stack"


# ── 5. the consequence at the reader ─────────────────────────────────────────
def test_plan_prunes_the_backwards_recipe_and_says_so():
    """`experiment.plan` is the reader where this matters most: an illegally ordered cell in
    a sweep produces a row whose g-vector carries a fabricated `pseudotime_range`, and the
    table cannot tell it from a legitimate one. `skipped_cells` must NAME the drop — a
    pruned sweep that does not say what it pruned looks like a smaller experiment."""
    from manyruns.experiment import plan, skipped_cells

    datasets = [{"name": "d", "shape": "manifold", "handle": {"kind": "path", "ref": "x"}}]
    assert plan([BACKWARDS], datasets) == []
    assert plan([FORWARD], datasets) != []
    assert skipped_cells([BACKWARDS], datasets) == [("backwards", "d", ["embedding"])]
