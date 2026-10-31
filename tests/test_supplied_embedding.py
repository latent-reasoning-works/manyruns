"""Refuse what cannot COMPOSE; never refuse what is merely unconventional.

The precondition calculus and the engine both check that a trajectory step has an embedding
to run on. The first version of that check asked WHO PRODUCED IT — `GROUP_PROVIDES` made
`embedding` a fact only a `latent` step could create, and `_new_state` hardcoded
`state["emb"] = None`, so an embedding could not enter the pipeline from outside at all.

That is a rule about convention wearing the clothes of a rule about types. A precomputed
embedding, or a graph built by another tool with compatible dimensions, composes with MIOFlow
perfectly well; refusing it prunes exactly the unconventional-but-well-typed compositions that
new method comes from. What must stay refused is the composition that cannot compose — a
trajectory step reading a raw ambient matrix as though it were a coordinate system.

Both halves are pinned here, because a fix for either one alone is a regression in the other.
"""
import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from manyruns import vocab  # noqa: E402
from manyruns.pipeline import runner  # noqa: E402

#: A trajectory step with no embedding-producing step in front of it. Legal or not depending
#: entirely on whether an embedding is supplied — which is the point.
MIOFLOW_ALONE = {"name": "supplied", "steps": [{"name": "mioflow", "group": "lightning", "params": {}}]}
BACKWARDS = {"name": "backwards", "steps": [
    {"name": "mioflow", "group": "lightning", "params": {}},
    {"name": "phate", "group": "latent", "params": {}},
]}


def _embedding(n=120, d=3, seed=0):
    """Stands in for coordinates computed anywhere else — another tool, a file, a graph."""
    return np.asarray(np.random.default_rng(seed).normal(size=(n, d)))


# ── the calculus ─────────────────────────────────────────────────────────────
def test_a_supplied_embedding_makes_a_bare_trajectory_recipe_legal():
    """`provided` is the caller's own inventory, and it satisfies the need like any other.

    Without it this returns {'embedding'} — correct, since nothing anywhere has one. With it
    the recipe is legal, and NOTHING about the step list changed. That is the distinction:
    the fact is about the data on hand, not about which step manufactured it."""
    assert vocab.unmet(MIOFLOW_ALONE, "manifold") == frozenset({"embedding"})
    assert vocab.unmet(MIOFLOW_ALONE, "manifold", provided={"embedding"}) == frozenset()


def test_supplying_a_fact_never_licenses_a_different_missing_one():
    """An embedding does not stand in for conditions. Otherwise `provided` would become a
    blanket override rather than an inventory, and the calculus would be advisory."""
    contrast = {"name": "c", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        {"name": "separation", "group": "analysis", "params": {}},
    ]}
    assert "conditions" in vocab.unmet(contrast, "manifold", provided={"embedding"})


def test_the_uncomposable_recipe_is_still_refused_with_an_embedding_nowhere():
    """The case that motivated the check keeps failing. `[mioflow, phate]` with no supplied
    embedding has one NOWHERE — not from the data, not from a preceding step — so the
    trajectory step would read the raw ambient matrix as a coordinate system. Measured before
    the engine refused: every step ok, run ok, `pseudotime_range = [0.0, 1.0]`."""
    assert vocab.unmet(BACKWARDS, "manifold") == frozenset({"embedding"})


# ── the engine ───────────────────────────────────────────────────────────────
def test_mioflow_runs_on_an_embedding_it_did_not_produce(tmp_path):
    """The bunnyhop: coordinates from elsewhere, straight into the trajectory step.

    This is the assertion that fails if `_new_state` ever hardcodes `emb=None` again."""
    emb = _embedding()

    res = runner.run_inproc(np.zeros((120, 8)), MIOFLOW_ALONE, tmp_path, seed=0, embedding=emb)

    assert [(s["name"], s["outcome"]) for s in res["steps"]] == [("mioflow", "ok")]
    assert res["ok"] is True
    assert "pseudotime_range" in res["g_vector"]
    # and it ran on the SUPPLIED coordinates — 3 columns, not the 8 of the raw matrix
    assert res["g_vector"]["final_dim"] == emb.shape[1]


def test_the_same_recipe_without_one_is_refused_by_the_engine(tmp_path):
    """Identical recipe, identical data, no embedding supplied → skipped with a reason.

    Two runs of one recipe differing only in what the caller had in hand, and the record says
    which happened. That is the whole distinction, expressed as a test.

    THE RUN-LEVEL FLAGS CHANGED HERE and the change is the point (manyruns#66). This asserted
    `ok is False`. `_require_embedding` raises a plain `_StepSkipped`: the executor RAN and
    judged that a precondition was not met — the same class of event as `normalize` standing
    down on a matrix that is not counts, and NOT a capability gap (`_StepUnsupported`, which
    does spoil `ok`). So the honest pair is `ok` — nothing errored, nothing was missing — and
    `complete=False`, which is where "you did not get your trajectory" is now recorded. The
    g-vector assertion below is unchanged and is still the load-bearing one: no pseudotime was
    invented to fill the hole."""
    res = runner.run_inproc(np.zeros((120, 8)), MIOFLOW_ALONE, tmp_path, seed=0)

    step = res["steps"][0]
    assert step["outcome"] == "skipped"
    assert "unsupported" not in step, "the executor ran and judged; it is not a capability gap"
    assert "embedding" in step["detail"]
    assert res["ok"] is True and res["complete"] is False
    assert "pseudotime_range" not in res["g_vector"]


def test_a_supplied_embedding_is_measured_like_any_other(tmp_path):
    """It reaches the g-vector as geometry, not as an untracked side input — otherwise a run
    on supplied coordinates would be incomparable with one that computed them."""
    res = runner.run_inproc(np.zeros((120, 8)), MIOFLOW_ALONE, tmp_path, seed=0,
                              embedding=_embedding())

    g = res["g_vector"]
    assert g["n_embedded"] == 120 and g["final_dim"] == 3
    assert g["suite_declared"] >= 1          # the declared suite ran on it


# ── the caveat: canon is reported, never enforced ────────────────────────────
def test_an_unusual_route_is_reported_and_still_runs(tmp_path):
    """The composition runs AND the record says it was unusual. Both halves matter.

    Refusing it would prune a well-typed move for being off-canon; running it silently would
    hand back a number whose provenance a reader cannot see. `unmet` decides legality on
    whether the object exists; this decides only whether to mention how it got there."""
    res = runner.run_inproc(np.zeros((120, 8)), MIOFLOW_ALONE, tmp_path, seed=0,
                              embedding=_embedding())

    assert res["ok"] is True                       # it ran
    assert len(res["caveats"]) == 1                # and it said so
    note = res["caveats"][0]
    assert "not from a latent step" in note and "phate" in note
    assert "running anyway" in note


def test_the_canonical_route_says_nothing(tmp_path):
    """Quiet unless something is genuinely unusual — a caveat on every run is a caveat
    nobody reads. Verified for the bundled recipes in `vocab.noncanonical`."""
    from manyruns import catalog

    # 12 columns → 60. `cflows` gained a declared `normalize → transform → pca(50)` prep block
    # in the cutover (#54), and `pca` carries `limits: {n_components: [n_samples, n_features]}`,
    # so on a 12-column array `bounds.fit` clamps 50 → 12 and says so in `caveats` — correctly:
    # that caveat reports the run not doing what the recipe declared. It is a fact about the
    # DATA being narrower than the recipe, which is exactly the "genuinely unusual" this test
    # allows; what it must not see is a caveat on a run where nothing was substituted. 60
    # columns clears the clamp, so an empty list stays the assertion. (The Gaussian point cloud
    # makes `normalize`/`transform` DECLINE — recorded as `outcome="skipped"` with a reason,
    # which is not a caveat and does not populate this column.)
    res = runner.run_inproc(np.random.default_rng(0).normal(size=(150, 60)),
                              catalog.load_recipe("cflows"), tmp_path, seed=0)

    assert res["caveats"] == []


def test_the_caveat_reaches_the_reader(tmp_path):
    """It is in `run_panel`, which is the plain-text rendering `shell.render_result` falls
    back to when output is piped — so the fact survives the surface a script sees."""
    from manyruns import narrate

    res = runner.run_inproc(np.zeros((120, 8)), MIOFLOW_ALONE, tmp_path, seed=0,
                              embedding=_embedding())

    panel = narrate.run_panel(res)
    assert "not from a latent step" in panel


def test_an_unmet_need_is_never_reported_as_merely_unusual():
    """`unmet` and `noncanonical` answer different questions and must not both answer one.
    A missing embedding is illegal, not unconventional — reporting it in two vocabularies is
    how the two answers drift into disagreement."""
    assert vocab.unmet(MIOFLOW_ALONE, "manifold") == frozenset({"embedding"})
    assert vocab.noncanonical(MIOFLOW_ALONE, "manifold") == []
