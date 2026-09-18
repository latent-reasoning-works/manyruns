"""A step's params resolved against the data in hand — `manyruns/pipeline/bounds.py`.

WHAT IS UNDER TEST IS MACHINERY, and that is the point of the file. The first version of this
module carried `CEILINGS = {("pca", "n_components"): (("n_samples", "n_features"), …)}` — which
is sklearn's contract for PCA, written down in manyruns. The limit is now declared by the
RECIPE, in the recipe, and this module does arithmetic over shape names it never interprets.

So most of the tests below deliberately use a made-up step name: if any of them started needing
`pca` in particular, the table would be back.
"""
from __future__ import annotations

import pytest

from manyruns.pipeline import bounds  # noqa: E402

#: A step as a recipe declares one. The name is nonsense on purpose — see the module docstring.
STEP = {"name": "widget", "group": "latent", "params": {"n_components": 10},
        "limits": {"n_components": ["n_samples", "n_features"]}}


# ── the arithmetic ───────────────────────────────────────────────────────────
def test_the_cap_is_the_minimum_of_the_names_the_recipe_gave():
    """`limits:` names shape identifiers; this resolves them and takes the smallest. The
    resolution is the whole of manyruns's contribution — the CHOICE of names was the recipe
    author's, and they are the one holding the method's documentation."""
    params, notes = bounds.fit(STEP, STEP["params"], (150, 3))

    assert params == {"n_components": 3}
    assert len(notes) == 1
    assert "10 → 3" in notes[0] and "150 × 3" in notes[0]
    assert "n_samples / n_features" in notes[0], "the note quotes the recipe's own cap"


def test_rows_cap_it_too_not_only_columns():
    """A minimum over every name listed. Four rows and six columns caps at four, and a
    resolution that only read the first name would sail past it."""
    assert bounds.fit(STEP, STEP["params"], (4, 6))[0] == {"n_components": 4}


def test_a_step_with_no_limits_is_bounded_by_nothing():
    """THE RELAXATION, and the load-bearing test in this file. There is no fallback table to
    fall back to: manyruns asserts no law about anyone's method, so an out-of-range parameter
    that no recipe capped reaches the engine, raises there, and is recorded verbatim.

    This fails the moment someone re-adds a per-method default."""
    bare = {"name": "pca", "params": {"n_components": 10}}          # a real method name

    assert bounds.fit(bare, bare["params"], (150, 3)) == ({"n_components": 10}, [])


def test_a_value_that_already_fits_is_left_exactly_alone():
    """Silence unless something is genuinely out of range. A note on every run is a note nobody
    reads, and a `bounded` record on a step that was never bounded is a false one."""
    params, notes = bounds.fit(STEP, {"n_components": 3}, (150, 32738))

    assert params == {"n_components": 3} and notes == []


def test_the_declaration_is_never_rewritten():
    """`fit` returns a NEW dict, so `recipe["steps"]` still says what was ASKED however many
    times it runs — otherwise a second run on wider data would inherit the first run's clamp."""
    declared = {"n_components": 10}

    bounds.fit(STEP, declared, (150, 3))

    assert declared == {"n_components": 10}


def test_an_unknown_shape_resolves_nothing():
    """No shape, no resolution, no claim. A named manylatents dataset is loaded inside the
    engine and manyruns never sees the array — `manylatents.api` exposes only `run`."""
    assert bounds.fit(STEP, STEP["params"], None) == ({"n_components": 10}, [])


def test_a_cap_never_goes_to_zero():
    """Clamping to zero would turn "too many components" into "no embedding at all", and the
    step would report `ok` having produced nothing."""
    assert bounds.fit(STEP, STEP["params"], (0, 0))[0] == {"n_components": bounds.FLOOR}


def test_a_boolean_is_not_a_number_to_cap():
    """`isinstance(True, int)` is True in Python. A flag silently clamped to 1 reads as set
    whatever it was."""
    step = {"name": "w", "params": {"flag": True}, "limits": {"flag": ["n_features"]}}

    assert bounds.fit(step, step["params"], (150, 3)) == ({"flag": True}, [])


# ── the vocabulary, which must stay one vocabulary ───────────────────────────
def test_the_limit_names_are_a_subset_of_the_suites_shape_names():
    """A recipe author reads ONE vocabulary. `INPUT_VARS` is narrower than `suite.SHAPE_VARS`
    because `n_embedded`/`final_dim` describe an embedding that exists only after a step has
    run, so they cannot bound its input — but narrower must never mean different, and the two
    are declared in separate files precisely so this test has something to check."""
    from manyruns.pipeline import suite

    assert set(bounds.INPUT_VARS) < set(suite.SHAPE_VARS)


# ── the declaration is checked, so a typo is loud ────────────────────────────
def test_a_misspelt_shape_name_is_refused_at_load():
    """`n_feature` would resolve to nothing, cap nothing, and read on the page as though it
    had. Absent-is-legal; present-but-wrong is refused, which is what stops the field from
    being decorative."""
    step = {"name": "w", "params": {"n_components": 10}, "limits": {"n_components": ["n_feature"]}}

    problems = bounds.check_limits(step)

    assert problems and "n_feature" in problems[0]


def test_a_cap_on_a_param_the_step_never_passes_is_reported():
    """Nearly always a rename done on one side. It would never fire and nothing would say so."""
    step = {"name": "w", "params": {"n_components": 10}, "limits": {"n_neighbors": ["n_samples"]}}

    assert any("does not" in p for p in bounds.check_limits(step))


def test_no_limits_block_is_no_problem():
    assert bounds.check_limits({"name": "w", "params": {}}) == []


def test_the_recipe_checker_runs_it():
    """Wired into `catalog.check_recipe`, which is what `load_recipe` raises on — otherwise the
    validation exists and nothing calls it."""
    from manyruns.catalog import check_recipe

    bad = check_recipe({"name": "r", "steps": [
        {"name": "w", "group": "latent", "params": {"n_components": 10},
         "limits": {"n_components": ["n_feature"]}}]}, name="r")

    assert any("n_feature" in p for p in bad)


def test_every_bundled_recipe_declares_valid_limits():
    """The shipped catalogue passes its own checker."""
    from manyruns.catalog import discover_recipes, load_recipe

    for name in discover_recipes():
        for step in load_recipe(name)["steps"]:
            assert bounds.check_limits(step) == [], f"{name}/{step.get('name')}"


# ── the shape it resolves against ────────────────────────────────────────────
def test_the_shape_is_the_matrix_the_step_will_actually_see():
    """The SAME choice `_ml_latent` makes — the embedding if one exists, else the input. A
    limit resolved against a different matrix from the one the step runs on is worse than
    none."""
    np = pytest.importorskip("numpy")
    X, emb = np.zeros((150, 32738)), np.zeros((150, 3))

    assert bounds.shape_of({"X": X, "emb": None}, {"array": X}) == (150, 32738)
    assert bounds.shape_of({"X": X, "emb": emb}, {"array": X}) == (150, 3)
    assert bounds.shape_of({"X": None, "emb": None}, {}) is None


# ── end to end, which is where it was reported ───────────────────────────────
@pytest.mark.parametrize("recipe_name", ["cluster", "markers"])
def test_the_recipes_that_broke_on_a_narrow_dataset_run(tmp_path, recipe_name):
    """Measured before: `pca` errored and took `cluster` and `markers` down with it on a 150×3
    array, because `n_components: 10` is a scRNA default. This is that array.

    RE-FRAMED, not loosened. This asserted `res["ok"] is True`, and `ok` is
    `all(status == "ok")` — which no synthetic array can satisfy since the cutover: both
    recipes now DECLARE the `normalize → transform` preamble `loading._anndata_matrix` used to
    run unannounced, and on a Gaussian point cloud both DECLINE (log1p of a negative is NaN,
    the bug the decline exists to prevent) and are recorded `skipped`. The property this test
    was written to protect is that the narrow array makes nothing ERROR, so that is what it
    asserts, together with WHICH steps stood down — so a third one going quiet is still loud."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("manylatents")
    # `cluster` and `markers` both declare a prep block since the cutover, and the prep executors
    # delegate to the `[omics]` extra — absent in CI, where this would otherwise read as the
    # narrow-dataset regression returning rather than as a missing install. See manyruns#68.
    pytest.importorskip("manylatents.singlecell.preprocessing",
                        reason="both recipes declare prep steps ([omics] extra)")
    from manyruns import catalog
    from manyruns.pipeline import runner

    X = np.asarray(np.random.default_rng(0).normal(size=(150, 3)))
    res = runner.run_manylatents(catalog.load_recipe(recipe_name), array=X, out_dir=tmp_path,
                                 seed=0)

    assert [(s["name"], s["detail"]) for s in res["steps"] if s["outcome"] == "error"] == []
    # The declined pair, in recipe order, and no others — measured on this 150×3 Gaussian.
    # `skipped` by NAME rather than `!= "ok"`: `runner.DID_NOT_RUN` holds two outcomes, and a
    # decline that silently became the other one is the failure this line exists to catch.
    stood_down = [(s["name"], s["outcome"]) for s in res["steps"] if s["outcome"] != "ok"]
    assert stood_down == [("normalize", "skipped"), ("transform", "skipped")]
    # A decline that says nothing is the unannounced preamble again, wearing a different hat.
    assert all(s["detail"] for s in res["steps"] if s["outcome"] == "skipped")
    # The step this test is about: bounded to 3 components and RAN, rather than raising.
    assert next(s for s in res["steps"] if s["name"] == "pca")["outcome"] == "ok"


def test_the_record_says_what_ran_and_the_run_says_it_out_loud(tmp_path):
    """Two channels, both required. The step record carries the EFFECTIVE params so
    `index.jsonl` says what ran rather than what was asked; the caveats carry the sentence to
    every surface through `narrate.caveats`. A silent clamp would be a number whose provenance
    a reader cannot see."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("manylatents")
    from manyruns import catalog, narrate
    from manyruns.pipeline import runner

    X = np.asarray(np.random.default_rng(0).normal(size=(150, 3)))
    res = runner.run_manylatents(catalog.load_recipe("cluster"), array=X, out_dir=tmp_path,
                                 seed=0)

    pca = next(s for s in res["steps"] if s["name"] == "pca")
    assert pca["params"]["n_components"] == 3          # what RAN
    assert pca["bounded"] and "10 → 3" in pca["bounded"][0]
    assert any("10 → 3" in c for c in res["caveats"])
    assert any("10 → 3" in c.text for c in narrate.caveats(res))

    # and the declaration is untouched, so the recipe still says what it asks for.
    # BY NAME, NOT BY POSITION: step 0 of `cluster` is now `normalize` — the cutover prepended
    # the declared `normalize → transform` preamble, so `pca` sits at index 2.
    declared = next(s for s in catalog.load_recipe("cluster")["steps"] if s["name"] == "pca")
    assert declared["params"]["n_components"] == 10


def test_resolved_caps_and_unknowns_are_serializable_and_partial():
    import json

    step = {**STEP, "params": {**STEP["params"], "knn": 5}}
    resolved = bounds.resolved_constraints(step, (150, 3))
    assert resolved == {"n_components": {"status": "resolved", "cap": 3},
                        "knn": {"status": "unknown", "cap": None, "reason": "no_limit"}}
    assert json.loads(json.dumps(resolved)) == resolved
    assert bounds.resolved_constraints(step, None) == {
        "n_components": {"status": "unknown", "cap": None, "reason": "unresolved_shape"},
        "knn": {"status": "unknown", "cap": None, "reason": "no_limit"}}
    # A value already below the cap still carries the constraint, with no change note.
    params, notes = bounds.fit(step, {"n_components": 2, "knn": 5}, (150, 3))
    assert params == {"n_components": 2, "knn": 5} and notes == []
    assert resolved["n_components"]["cap"] == 3


def test_step_records_caps_without_a_clamp_and_preserves_full_errors():
    import json
    import numpy as np
    from manyruns.pipeline import runner

    detail = "engine failure " + "x" * 400
    def fail(*args):
        raise ValueError(detail)
    step = {**STEP, "params": {"n_components": 2, "knn": 5}}
    rec = runner.apply_step(step, state={"emb": np.zeros((150, 3))}, g={}, ctx={},
                            dispatch={"latent": fail}, index=0, carry=runner.new_carry())
    assert rec["detail"] == "ValueError: " + detail
    assert rec["constraints"] == bounds.resolved_constraints(step, (150, 3))
    assert "bounded" not in rec
    assert json.loads(json.dumps(rec["constraints"])) == rec["constraints"]


def test_declared_shape_is_last_resort():
    import numpy as np
    assert bounds.shape_of({}, {'declared_shape': (5000, 3)}) == (5000, 3)
    assert bounds.shape_of({'emb': np.zeros((8, 2))}, {'declared_shape': (5000, 3)}) == (8, 2)


@pytest.mark.parametrize('shape', [(True, 3), (0, 3), (3,), (3, -1), (3, 2.5)])
def test_declared_shape_rejects_invalid_dimensions(shape):
    with pytest.raises(ValueError, match='two positive integers'):
        bounds.shape_of({}, {'declared_shape': shape})
