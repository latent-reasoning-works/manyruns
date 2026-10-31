"""What a narrowing does to the facts on hand — `vocab.unmet` and `vocab.noncanonical`.

`have` used to only ever grow (`have |= step_provides(step)`, vocab.py:598), which was true
while no step could remove anything. A `prep` step can. And the subtraction is NOT a table of
which step names destroy what — that would be the product-side copy of the engine's algorithm
catalogue `vocab.py` refuses to make in both the `GROUP_PROVIDES` and `STEP_PRODUCES`
commentary. It is one rule read off tables that already exist:

    a fact from the FRAME survives a narrowing; a fact produced by a STEP does not.

The rule is the data model. `conditions` is carried by the data, so nothing can take it away —
after a filter there are fewer cells and each still carries its condition label. `embedding`
was fitted over cells a filter has now removed, so it is gone: PHATE's diffusion operator saw
them, and `emb[keep]` would LOOK like an embedding of the survivors without being one.

Two decisions this file pins because the plan left them open:

  1. **The rule is axis-blind.** A column narrowing clears the embedding too — see
     `test_a_column_narrowing_clears_the_embedding_for_the_same_reason_a_row_one_does`.
  2. **`noncanonical` folds identically or the two answers contradict each other** — see
     `test_noncanonical_cannot_call_a_fact_unusual_that_unmet_calls_missing`.

And two properties that the shipped tables cannot exercise, so the tests below construct the
day they can. The fold's THREE lines — check needs, clear, produce — give the same answer in
every permutation on every recipe in this repo, and the cutover did not change that even though
it moved both halves of the argument: two `NARROWING_STEPS` names DO appear in `STEP_NEEDS` now
(`filter_genes` and `filter_mito`, both `("genes",)`) and `DATASET_FACTS` is no longer `()` —
but `genes` is exactly what `DATASET_FACTS` puts in `frame_facts`, so `keep` holds it through
every clear and the check/clear order is still unobservable; and all five narrowing names are
`prep`, a group neither producer table mentions, so `step_provides` is empty for all of them.
Measured: each of those three mutations left the suite green at 1008 passed. Section 3 and the
two order tests are what those mutations now fail.

The other half of a refusal — WHY the fact is missing — is section 5. `unmet` says `embedding`
for a recipe that never embeds and for one that embeds and then filters, and the two want
opposite fixes.
"""
from __future__ import annotations

import pytest

from manyruns import catalog, vocab

PHATE = {"name": "phate", "group": "latent", "params": {}}
MIOFLOW = {"name": "mioflow", "group": "lightning", "params": {}}
LEIDEN = {"name": "leiden", "group": "latent", "params": {}}
FILTER = {"name": "filter_cells", "group": "prep", "params": {"min_genes": 200}}
FILTER_GENES = {"name": "filter_genes", "group": "prep", "params": {"min_cells": 3}}

#: A dataset declaration that carries a gene axis. `filter_genes` and `filter_mito` declare
#: `STEP_NEEDS[...] = ("genes",)`, so a recipe using one is refused for TWO reasons on a
#: point cloud — a missing gene axis and, separately, whatever the narrowing cleared. The
#: tests below are about the second, so they supply the first rather than conflating them.
#:
#: THE CUTOVER WIDENED WHO NEEDS THIS. `qc` gained `("genes",)` and `rank_genes` gained it
#: alongside `clusters`, now that `loading._anndata_matrix` hands over the raw matrix instead of
#: a 50-column PCA — so the two-reasons hazard reaches every test below that ranks genes after a
#: filter, not just the two filters that read gene names. Those tests pass this for the same
#: reason: to be refused for the narrowing and nothing else.
#:
#: No `provides:` key here, deliberately — this literal reaches `genes` through
#: `dataset_provides`' `.h5ad` suffix guess, the fallback for a declaration that says nothing.
#: `configs/dataset/pbmc3k.yaml` is the declared route and the only bundled dataset that takes
#: it; both land on the same fact, which is what makes this hand-written handle a fair stand-in.
H5AD = {"kind": "path", "ref": "data/pbmc3k_raw.h5ad"}
TRANSFORM = {"name": "transform", "group": "prep", "params": {"method": "log1p"}}


def _recipe(*steps):
    return {"name": "t", "steps": list(steps)}


# ── 1. the rule ──────────────────────────────────────────────────────────────
def test_a_filter_between_the_embedding_and_the_trajectory_is_refused():
    """The headline. `mioflow` declares `embedding`; the filter dropped the cells it was
    fitted over. Refused at PLAN time, which is where a search over recipes has to get its
    answer — it cannot afford to run each one to find out."""
    assert vocab.unmet(_recipe(PHATE, FILTER, MIOFLOW), "time-course") == frozenset({"embedding"})


def test_the_same_recipe_without_the_filter_stays_legal():
    """The guard on the guard: every currently-legal composition must stay legal. That is the
    one thing adding an invalidation rule must not break, and it is the reason
    `test_every_bundled_recipe_is_still_legal_on_the_data_its_declared_shape_describes` exists
    below (renamed at the cutover — see its docstring for why the shape alone stopped
    deciding)."""
    assert vocab.unmet(_recipe(PHATE, MIOFLOW), "time-course") == frozenset()


def test_filtering_BEFORE_the_embedding_is_legal():
    """Which is the whole standard workflow — filter, then embed (`preprocess.yaml` puts every
    filter first for exactly this reason). The rule refuses a STALE derived array, not the act
    of filtering."""
    assert vocab.unmet(_recipe(FILTER, PHATE, MIOFLOW), "time-course") == frozenset()


def test_a_prep_step_that_does_not_narrow_clears_nothing():
    """`transform` returns a new matrix, not a mask (`prep._step_transform`). Treating every
    prep step as a narrowing — keying the rule on GROUP, the way `GROUP_PROVIDES` is keyed —
    would refuse `[phate, transform, mioflow]`, which composes and runs."""
    assert vocab.unmet(_recipe(PHATE, TRANSFORM, MIOFLOW), "time-course") == frozenset()


def test_a_frame_fact_survives_a_narrowing():
    """`conditions` comes from the data, not from a step, so a filter cannot remove it — there
    are fewer cells carrying the condition label and the label is still there. This is the half
    of the rule that keeps it from being "a narrowing invalidates everything"."""
    contrast = _recipe(FILTER, PHATE, {"name": "separation", "group": "analysis", "params": {}})

    assert vocab.unmet(contrast, "case-control") == frozenset()


def test_a_narrowing_step_still_gets_its_own_needs_checked_before_it_clears(monkeypatch):
    """Order inside the fold. A narrowing clears facts for the steps AFTER it; its own needs
    are checked against what was on hand when it ran, exactly as every other step's are.

    THE FIRST TWO ASSERTIONS CANNOT SEE THAT ORDER and are kept for the property they do pin
    (a later step reads the cleared facts). Measured: with `have &= keep` moved ABOVE the
    `missing.update(...)` line in `vocab.unmet`, the whole suite stayed green at
    1008 passed — because none of the five names in `NARROWING_STEPS` appears in `STEP_NEEDS`,
    so no shipped filter has a precondition for the reordering to destroy.

    The third assertion constructs one. `filter_cells` with a `STEP_NEEDS` entry of
    `embedding` is filtering outliers in embedding space — a real step, and the two that landed
    escape this only by accident (`filter_genes` and `filter_mito` need `genes`, which
    `DATASET_FACTS` makes a FRAME fact and therefore a member of `keep`). Reading a
    filter's own precondition against the facts it has just destroyed would refuse
    `[phate, filter_cells]`, where the filter genuinely ran on phate's coordinates."""
    ranks = {"name": "rank_genes", "group": "analysis", "params": {}}

    # `handle=H5AD` ADDED AT THE CUTOVER: `rank_genes` went from `("clusters",)` to
    # `("clusters", "genes")` in `STEP_NEEDS`, so on a bare shape both lines below now also
    # report `genes` — the missing-gene-axis reason, not the narrowing this test is about.
    # The facts asserted are unchanged: `{}` before the filter, `{"clusters"}` after it.
    assert vocab.unmet(_recipe(LEIDEN, ranks, FILTER), "clusters", handle=H5AD) == frozenset()
    assert vocab.unmet(_recipe(LEIDEN, FILTER, ranks), "clusters",
                       handle=H5AD) == frozenset({"clusters"})

    monkeypatch.setitem(vocab.STEP_NEEDS, "filter_cells", ("embedding",))
    assert vocab.unmet(_recipe(PHATE, FILTER), "manifold") == frozenset()


def test_a_narrowing_clears_before_it_produces_so_it_keeps_what_it_selected(monkeypatch):
    """The OTHER half of the fold's order, and nothing shipped can tell the two apart.

    Measured: with `have &= keep` and `have |= step_provides(step)` SWAPPED in `vocab.unmet`,
    the suite stayed green at 1008 passed. Every name in `NARROWING_STEPS` has group `prep`,
    which is in neither `GROUP_PROVIDES` nor `STEP_PRODUCES`, so `step_provides` is empty for
    all five and the two orders are the same function.

    Spec §3's table makes that an accident with a date on it: it puts `hvg` in `STEP_PRODUCES`
    while `vocab.py:495` already lists `hvg` in `NARROWING_STEPS` — a step that BOTH narrows
    and produces. Clearing after producing erases the selection the step just made, refusing a
    recipe that runs; that is a `plan()` cell, a shell menu entry and an `app` coverage row
    lost to an ordering nobody would look at."""
    monkeypatch.setitem(vocab.STEP_PRODUCES, "hvg", ("hvg",))
    monkeypatch.setitem(vocab.STEP_NEEDS, "marker_panel", ("hvg",))
    recipe = _recipe({"name": "hvg", "group": "prep", "params": {"n_top_genes": 2000}},
                     {"name": "marker_panel", "group": "analysis", "params": {}})

    assert vocab.step_narrows(recipe["steps"][0]) is True
    assert vocab.unmet(recipe, "manifold") == frozenset()


# ── 2. the axis question, decided ────────────────────────────────────────────
def test_a_column_narrowing_clears_the_embedding_for_the_same_reason_a_row_one_does():
    """THE AXIS DECISION, pinned. `filter_genes` and `hvg` remove GENES: no cell is dropped, so
    `emb` still has one row per selected cell and nothing DESYNCS. The rule clears it anyway,
    and the deciding evidence is that the premise the row case was decided on is about FITTING,
    not about length: `runner._apply_transition`'s argument for clearing `emb` after a row
    filter is that "PHATE's diffusion operator saw them", NOT that the array got too long — on
    length alone `emb[keep]` would be a legal repair and nothing would ever need clearing.

    Applied to columns the same premise gives the same answer: a kNN graph over 32,738 genes is
    not the graph over the 2,000 that `hvg` kept, so the coordinates are not coordinates of the
    current view. Splitting the axes would mean holding the fitted premise on rows and a
    length premise on columns inside one fold — two definitions of stale in one function, which
    is the two-homes drift this module exists to prevent.

    It also errs in the direction `STEP_PRODUCES` says this kind of table is allowed to err in
    ("it prunes a step that would have run, it never admits one that cannot"), and the pruning
    is overridable: see the `provided` test below."""
    assert vocab.unmet(_recipe(PHATE, FILTER_GENES, MIOFLOW), "manifold",
                       handle=H5AD) == frozenset({"embedding"})


def test_a_caller_who_says_they_hold_the_embedding_keeps_it_across_a_narrowing():
    """The escape hatch, and the reason erring toward refusal is cheap. `provided` is the
    caller asserting they hold the object — a precomputed embedding, or one they will re-fit
    themselves — and this function does not get to decide their asset went stale. It is the
    same hatch `STEP_PRODUCES`' commentary names for `clusters`."""
    recipe = _recipe(PHATE, FILTER_GENES, MIOFLOW)

    assert vocab.unmet(recipe, "manifold", provided={"embedding"},
                       handle=H5AD) == frozenset()


def test_detect_doublets_narrows_only_when_the_recipe_asks_it_to():
    """`step_narrows` reads the step's declared PARAMS, not only its name, because one step's
    answer depends on them: `_step_detect_doublets` returns an all-True mask unless
    `remove: true` (spec §5 — "a step that silently drops cells is the inverse of this
    product's thesis"). Listing the name unconditionally would refuse
    `[phate, detect_doublets, mioflow]`, a recipe whose executor removes nothing."""
    flags = {"name": "detect_doublets", "group": "prep", "params": {"threshold": 0.3}}
    removes = {"name": "detect_doublets", "group": "prep", "params": {"remove": True}}

    assert vocab.step_narrows(flags) is False
    assert vocab.step_narrows(removes) is True
    assert vocab.unmet(_recipe(PHATE, flags, MIOFLOW), "manifold") == frozenset()
    assert vocab.unmet(_recipe(PHATE, removes, MIOFLOW), "manifold") == frozenset({"embedding"})


# ── 3. the rule is derived, not restated ─────────────────────────────────────
def test_frame_facts_is_the_shape_union_PLUS_what_a_dataset_declares():
    """`frame_facts` is the survivor set, and it has two sources for the same reason
    `dataset_provides` does: a shape says what the experimental DESIGN carries, a handle says
    what the BYTES carry. Both are properties of the data rather than of any step, which is
    exactly the definition of surviving a narrowing."""
    from_shapes = {f for facts in vocab.SHAPE_PROVIDES.values() for f in facts}

    # TWO now. `splicing` joined with the frame's `layers` slot: spliced/unspliced counts
    # are a property of the BYTES, exactly as gene names are, and no shape can supply either.
    # A velocity FIELD is deliberately NOT here — it is derived, so a narrowing clears it, and
    # that is the line between a frame fact and a step-produced one.
    assert vocab.DATASET_FACTS == ("genes", "splicing")
    assert from_shapes == {"conditions", "time"}
    # `splicing` joins the day the frame can carry layers. It is DECLARABLE by a dataset
    # from here (`check_provides` reads DATASET_FACTS) and not yet DEMANDABLE — no step
    # declares the need until a velocity tool does, which is the rule this file's neighbour
    # states: the demand lands with the supply, never the other way round. So it is a frame
    # fact and is deliberately absent from `vocab.facts()`.
    assert vocab.frame_facts() == {"conditions", "time", "genes", "splicing"}


def test_a_dataset_fact_on_hand_survives_a_narrowing():
    """The claim `frame_facts` is built on, asserted against the real tables rather than
    monkeypatched ones. This test used to construct the day `DATASET_FACTS` stopped being empty
    out of spec §3; that day is now, so it asserts it directly.

    Both directions, because the union line in `frame_facts` is what makes it true and a test
    that only checks the legal case cannot see the line go: `preprocess.yaml`'s own order
    (`filter_cells` then `filter_mito`) is legal when the data carries genes and refused for
    `genes` when it does not — never refused because the first filter cleared the second's
    precondition."""
    mito = {"name": "filter_mito", "group": "prep", "params": {"max_pct": 5}}

    assert vocab.unmet(_recipe(FILTER, mito), "manifold", handle=H5AD) == frozenset()
    assert vocab.unmet(_recipe(FILTER, mito), "manifold") == frozenset({"genes"})
    assert vocab.invalidated(_recipe(FILTER, mito), "manifold") == frozenset(), (
        "a gene axis the data never had was not TAKEN by the filter")


def test_no_step_produced_fact_is_also_a_frame_fact():
    """The disjointness the rule rests on, asserted rather than assumed. If a fact ever appears
    in both a shape table and a producer table, `have &= keep` silently stops clearing it and
    the calculus goes back to being monotone for that one word — the exact silent drift this
    module opens by describing."""
    produced = {f for facts in vocab.GROUP_PROVIDES.values() for f in facts}
    produced |= {f for facts in vocab.STEP_PRODUCES.values() for f in facts}

    assert produced & vocab.frame_facts() == set()


def test_every_prep_step_that_exists_is_classified_by_what_it_returns():
    """The drift guard between the vocabulary and the executors it describes. `vocab` cannot
    import `pipeline` (its header records what that cost last time: the whole package dragged
    in for a 3-tuple of strings), so this test is where the two are held together — a name
    belongs in `NARROWING_STEPS` iff its executor returns a `mask` rather than an `X`.

    Written as "account for every prep step", not "the list is exactly this", so that landing a
    filter keeps it green and landing an UNCLASSIFIED step does not. `normalize` and `transform`
    are the two matrix-returning steps the spec's operations table names (§5); anything else
    appearing in `_PREP_STEPS` has to be one or the other and someone has to say which."""
    from manyruns.pipeline import prep

    assert vocab.step_narrows({"name": "transform", "group": "prep", "params": {}}) is False

    returns_a_matrix = {"normalize", "transform"}
    unclassified = set(prep._PREP_STEPS) - set(vocab.NARROWING_STEPS) - returns_a_matrix
    assert unclassified == set(), (
        f"prep steps {sorted(unclassified)} are in neither list — a step that narrows and is "
        "not in NARROWING_STEPS leaves the calculus holding an embedding the runner cleared"
    )


# ── 4. the two answers cannot drift ──────────────────────────────────────────
def test_noncanonical_cannot_call_a_fact_unusual_that_unmet_calls_missing(monkeypatch):
    """`noncanonical` MUST fold the same way as `unmet` or the two contradict each other, which
    is the failure this module's own docstring records as having happened before ("reporting it
    twice in two vocabularies is how the two answers drift apart").

    Constructed rather than hoped for. With today's one-entry `CANONICAL_PROVIDER` the mirror
    changes no output — the only non-canonical source of `embedding` is `the caller`, and
    `provided` survives a narrowing by design — so this registers the second entry that table
    will one day have and exercises the fold underneath it. Without the mirror in
    `noncanonical`, this recipe is reported ILLEGAL by `unmet` (`clusters` was cleared) and
    simultaneously described by `noncanonical` as "legal and unusual, running anyway"."""
    monkeypatch.setitem(vocab.CANONICAL_PROVIDER, "clusters", "analysis")
    recipe = _recipe(LEIDEN, FILTER, {"name": "rank_genes", "group": "analysis", "params": {}})

    # `handle=H5AD` ADDED AT THE CUTOVER, same reason as the order test above: `rank_genes` now
    # declares `("clusters", "genes")`, so without a gene axis `unmet` answers
    # `{"clusters", "genes"}` and the contradiction under test — one fold calling `clusters`
    # missing while the other calls it merely unusual — is read through a second, unrelated
    # refusal. The fact asserted is still exactly `{"clusters"}`, the one the filter cleared.
    assert vocab.unmet(recipe, "clusters", handle=H5AD) == frozenset({"clusters"})
    assert vocab.noncanonical(recipe, "clusters", handle=H5AD) == []


def test_noncanonical_names_the_callers_embedding_once_the_runs_own_has_been_cleared():
    """The other half of the mirror: the fact survives, and WHERE IT CAME FROM changed. After
    the filter, the embedding `mioflow` reads cannot be phate's — the calculus just declared
    that one dead — so the only one left is the caller's, and a result obtained off the beaten
    path is exactly the one whose provenance a reader needs."""
    recipe = _recipe(PHATE, FILTER, MIOFLOW)

    notes = vocab.noncanonical(recipe, "manifold", provided={"embedding"})

    assert len(notes) == 1
    assert "embedding came from the caller" in notes[0]
    assert vocab.noncanonical(_recipe(PHATE, MIOFLOW), "manifold", provided={"embedding"}) == []


# ── 5. a refusal a human can act on ──────────────────────────────────────────
def test_invalidated_names_the_facts_a_narrowing_took_away():
    """`unmet` says WHICH fact is missing; this says WHY it is missing, and the two are one
    fold so they cannot disagree about the same recipe.

    It exists because the answer decides what a human is told. `[phate, filter_cells, mioflow]`
    and `[mioflow, phate]` both come back `{"embedding"}`, and they are opposite problems: the
    first embeds and then throws the coordinates away, the second never embeds at all. Told the
    second reason for the first recipe, the user adds a second embedder and is refused again."""
    assert vocab.invalidated(_recipe(PHATE, FILTER, MIOFLOW), "time-course") == frozenset(
        {"embedding"})
    assert vocab.invalidated(_recipe(PHATE, FILTER_GENES, MIOFLOW), "manifold",
                             handle=H5AD) == frozenset({"embedding"})

    ranks = {"name": "rank_genes", "group": "analysis", "params": {}}
    assert vocab.invalidated(_recipe(LEIDEN, FILTER, ranks), "clusters") == frozenset(
        {"clusters"})


def test_a_fact_nothing_ever_produced_is_missing_but_was_never_TAKEN_away():
    """The half that keeps the distinction worth drawing. A filter in a recipe that never
    embeds at all removed nothing from the facts on hand, so the honest reason is still that
    nothing here lays the cells out."""
    assert vocab.unmet(_recipe(MIOFLOW, PHATE), "manifold") == frozenset({"embedding"})
    assert vocab.invalidated(_recipe(MIOFLOW, PHATE), "manifold") == frozenset()
    assert vocab.invalidated(_recipe(FILTER, MIOFLOW), "manifold") == frozenset()
    assert vocab.invalidated(_recipe(PHATE, TRANSFORM, MIOFLOW), "manifold") == frozenset()


def test_what_a_narrowing_took_away_is_always_something_unmet_refuses():
    """One fold, two projections — so `invalidated` can never name a fact `unmet` calls
    satisfied. A caller reads the two together (`narrate.blocked` picks the sentence with
    them), and a fact in one and not the other would be a reason attached to no refusal."""
    recipes = [_recipe(PHATE, FILTER, MIOFLOW), _recipe(PHATE, FILTER_GENES, MIOFLOW),
               _recipe(MIOFLOW, PHATE), _recipe(FILTER, PHATE, MIOFLOW),
               _recipe(LEIDEN, FILTER, {"name": "rank_genes", "group": "analysis"})]
    recipes += [catalog.load_recipe(n) for n in catalog.discover_recipes()]

    for recipe in recipes:
        for shape in vocab.SHAPES:
            for handle in (None, H5AD):
                assert (vocab.invalidated(recipe, shape, handle=handle)
                        <= vocab.unmet(recipe, shape, handle=handle))


# ── 6. nothing that runs today was newly pruned ──────────────────────────────
@pytest.mark.parametrize("name", catalog.discover_recipes())
def test_every_bundled_recipe_is_still_legal_on_the_data_its_declared_shape_describes(name):
    """The regression that matters most: this table's job is to prune what cannot compose, and
    it must never newly prune something that runs today.

    RENAMED, because the cutover broke the old name's premise rather than one of its numbers.
    "Legal on its declared shape" assumed a shape decides legality by itself, which stopped
    being true when `qc` and `rank_genes` declared `("genes",)`: `genes` is a DATASET fact, so
    `SHAPE_PROVIDES` never supplies it and only a handle can. Measured with no handle, over the
    ten bundled recipes: `qc` and `markers` come back `{"genes"}` on their declared suits and
    the other eight come back empty. Asserting the shape half alone would now read
    "these two recipes were newly pruned",
    which is the opposite of what happened — the gene axis they need exists for the first time.

    So the property is asserted the way it was always meant: nothing bundled is refused on data
    that carries what its steps declare. The second assertion is the exact converse and is what
    keeps the first from hiding a real pruning behind a generous handle — on a bare shape a
    bundled recipe may be refused for the gene axis and for NOTHING else.

    The third is the narrowing rule's own share of the claim, and it holds with or without a
    handle: seven of the ten gained a `normalize -> transform` prep block at the cutover, and
    neither name is in `NARROWING_STEPS`, so `step_narrows` is still False at every position of
    every bundled recipe and `invalidated` is empty for all ten."""
    recipe = catalog.load_recipe(name)
    # `genes` is the one fact a bundled recipe can need that no shape carries — see H5AD.
    needs_genes = frozenset({"genes"}) & vocab.recipe_needs(recipe)

    for suit in recipe.get("suits") or ["unknown"]:
        assert vocab.unmet(recipe, suit, handle=H5AD) == frozenset(), f"{name} on {suit}"
        assert vocab.noncanonical(recipe, suit, handle=H5AD) == [], f"{name} on {suit}"
        assert vocab.unmet(recipe, suit) == needs_genes, f"{name} on {suit}, no gene axis"
        assert vocab.invalidated(recipe, suit) == frozenset(), f"{name} on {suit}"
