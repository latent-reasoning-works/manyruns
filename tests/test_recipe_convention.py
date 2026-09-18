"""The recipe config convention: `id:`, `source:`, `claims:` and per-step `via:`.

Recipes declare stable identity, sources, claims and suitability; `via` records the
canonical implementation so substitutions can be reported.

Four things are pinned here, and each of them is a defect that was measured before it was
fixed rather than a feature that seemed nice:

  * **`claims:` replaced `score.RECIPE_RUNG`.** That was a hardcoded three-entry dict, and a
    recipe added as a YAML file reached the menu automatically (`narrate.offer` derives from
    the catalog) while `predicted_rung` scored it **0 — silently**, because "absent from the
    dict" and "claims rung 0" were the same value. `test_a_new_recipe_reaches_the_rung_table`
    is that drift, made to fail.
  * **`via:` surfaces a STAND-IN.** the in-process loop runs a diffusion pseudotime under the step
    name `mioflow` (`steps._step_mioflow`, whose own docstring says so). Nothing said so in
    the record. It now rides the `caveats` channel that already existed.
  * **`source:` is a mapping and its identifier is not resolved at load.** A loader that needs
    the internet fails on a plane.
  * **Absent is legal.** All four fields are additive; making any mandatory would refuse every
    recipe written before them, including a user's own in `$MANYRUNS_RECIPE_DIR`.

No compute stack is imported: every assertion here is over YAML, dicts and strings, except the
one session test, which supplies its embedding rather than fitting one.
"""
import numpy as np
import pytest

from manyruns import catalog, score, vocab

BUNDLED = ("cflows", "contrast", "embed")


def _write(dir_path, name, body):
    (dir_path / f"{name}.yaml").write_text(body)
    return dir_path


# ── the bundled three carry the convention ───────────────────────────────────
def test_every_bundled_recipe_declares_the_convention():
    """`id`, `source` and `claims` on all three. The fields are optional to the VALIDATOR
    (see `test_a_recipe_without_the_convention_still_loads`) and mandatory to the shipped
    catalog — a shipped recipe with no provenance is the thing the convention exists to stop.
    """
    for name in BUNDLED:
        cfg = catalog.load_recipe(name)
        assert cfg.get("id"), f"{name} declares no id"
        assert isinstance(cfg.get("source"), dict), f"{name} declares no source mapping"
        assert cfg.get("claims") in vocab.TOPOLOGIES, f"{name} claims {cfg.get('claims')!r}"
        assert catalog.check_recipe(cfg, name) == []


def test_bundled_ids_are_distinct_and_dotted():
    """`id` is the stable external handle — `name` is bound to the filename, so it cannot be
    renamed. Two recipes sharing an id would make it useless as a handle."""
    ids = [catalog.load_recipe(n)["id"] for n in BUNDLED]
    assert sorted(ids) == ["sc.contrast.separation", "sc.embed.phate", "sc.traj.mioflow"]
    assert len(set(ids)) == len(ids)


def test_a_source_that_is_not_kind_none_carries_a_resolvable_identifier():
    """The rule that stops `source:` from being decorative: anything but `kind: none` must
    name something a reader can go and check. `contrast` is the `none` case and it is real —
    neither of its analysis steps implements a published method."""
    kinds = {n: catalog.load_recipe(n)["source"]["kind"] for n in BUNDLED}
    assert kinds == {"cflows": "paper", "contrast": "none", "embed": "paper"}
    for name in ("cflows", "embed"):
        src = catalog.load_recipe(name)["source"]
        assert src.get("doi") or src.get("url")
        assert src.get("verified"), "an identifier with no verification date is a claim"


def test_no_recipe_cites_the_acm_placeholder_doi():
    """`10.5555/3600270.3602424` is ACM's non-DOI
    placeholder for MIOFlow and 404s at both doi.org and CrossRef — re-checked 2026-07-29,
    404 at api.crossref.org AND api.datacite.org. Pinned so it cannot be helpfully
    reintroduced by anyone who finds it in a reference manager."""
    for name in BUNDLED:
        blob = str(catalog.load_recipe(name).get("source") or {})
        assert "10.5555/3600270.3602424" not in blob


# ── §3.4: the validations ────────────────────────────────────────────────────
def test_a_recipe_without_the_convention_still_loads():
    """Absent is legal for all four fields. `check_recipe` is what `load_recipe` RAISES on, so
    a mandatory field here would refuse every recipe written before the convention existed."""
    bare = {"name": "bare", "steps": [{"name": "phate", "group": "latent"}]}
    assert catalog.check_recipe(bare, "bare") == []


@pytest.mark.parametrize("bad_id", ["Cluster", "sc cluster", "sc", "sc.", "sc..leiden", 3])
def test_a_malformed_id_is_refused(bad_id):
    """Dotted lowercase `<domain>.<verb>.<method>`. `sc` alone is refused because a single
    segment is a name, not a namespaced handle."""
    cfg = {"name": "r", "id": bad_id, "steps": [{"name": "phate", "group": "latent"}]}
    assert any("id must be dotted" in p for p in catalog.check_recipe(cfg, "r"))


def test_a_source_that_is_not_a_mapping_is_refused():
    """A bare `cite:` string cannot carry the kind, cannot carry a URL when no DOI exists, and
    cannot record when the identifier was last resolved."""
    cfg = {"name": "r", "source": "Wolf et al. 2018",
           "steps": [{"name": "phate", "group": "latent"}]}
    assert any("must be a mapping" in p for p in catalog.check_recipe(cfg, "r"))


def test_an_unknown_source_kind_is_refused():
    cfg = {"name": "r", "source": {"kind": "blogpost", "url": "https://example.invalid"},
           "steps": [{"name": "phate", "group": "latent"}]}
    assert any("source kind must be one of" in p for p in catalog.check_recipe(cfg, "r"))


def test_a_source_with_no_doi_and_no_url_is_refused_unless_kind_is_none():
    """The half of the rule that has teeth: `kind: none` is the ONLY way to declare a workflow
    with nothing to cite, and it has to be said explicitly."""
    unciteable = {"name": "r", "source": {"kind": "paper", "cite": "Someone, somewhere"},
                  "steps": [{"name": "phate", "group": "latent"}]}
    assert any("needs a `doi` or a `url`" in p for p in catalog.check_recipe(unciteable, "r"))
    honest = {"name": "r", "source": {"kind": "none", "note": "ours"},
              "steps": [{"name": "phate", "group": "latent"}]}
    assert catalog.check_recipe(honest, "r") == []


def test_a_doi_is_not_resolved_at_load_time():
    """§3.4: "`source.doi` must NOT be network-validated at load time — a strict loader that
    needs the internet is a loader that fails on a plane." Resolution belongs in the `admit`
    verbs, reporting rather than raising. This DOI is syntactically fine and resolves to
    nothing; the loader must not care."""
    cfg = {"name": "r", "source": {"kind": "paper", "doi": "10.9999/definitely.not.a.doi"},
           "steps": [{"name": "phate", "group": "latent"}]}
    assert catalog.check_recipe(cfg, "r") == []


@pytest.mark.parametrize("bad", ["spiral", ["clusters"], 2])
def test_claims_must_be_one_known_topology(bad):
    """SCALAR, unlike a dataset's multi-label `topology:` — a recipe asserts one thing, and
    that assertion is what `score.RUNG` grades."""
    cfg = {"name": "r", "claims": bad, "steps": [{"name": "phate", "group": "latent"}]}
    assert catalog.check_recipe(cfg, "r") != []


@pytest.mark.parametrize("bad", ["clusters", ["clusters", "sausage"], 7])
def test_suits_must_be_a_list_of_known_shapes(bad):
    """`suits` was unvalidated entirely: a typo fell through to a recommendation that could
    never fire, with nothing said."""
    cfg = {"name": "r", "suits": bad, "steps": [{"name": "phate", "group": "latent"}]}
    assert catalog.check_recipe(cfg, "r") != []


def test_a_via_that_is_not_a_name_is_refused():
    """`via` is advisory and never picks an executor, but `vocab.standins` compares it to an
    engine NAME — a list there makes a real caveat silently unreachable rather than loudly
    wrong."""
    cfg = {"name": "r", "steps": [{"name": "mioflow", "group": "lightning",
                                   "via": ["manylatents"]}]}
    assert any("`via` must be the name" in p for p in catalog.check_recipe(cfg, "r"))


# ── §6 stage 8: `claims:` replaces the hardcoded rung table ──────────────────
def test_the_bundled_rungs_are_unchanged_and_now_derived():
    """The exact three values `score.RECIPE_RUNG` used to hold, read out of the YAML instead.
    A swap that changed any of them would be a silent re-grading of every past evaluation.

    The other four were ADDED, not swapped: Tier 1
    of the selector's metric is `-ladder_cost(truth, pred)`, and a recipe with no `claims:`
    contributes rung 0, so five of eight choices could not be graded at all. The three original
    values are asserted individually below precisely so a future edit cannot move one of them
    under cover of adding another."""
    assert score.claimed_rungs() == {"embed": 0, "contrast": 1, "cflows": 2,
                                     "archetypes": 4, "cluster": 1, "markers": 1,
                                     "pseudotime": 2,
                                     # same claim as `cflows`; a probe reports on the structure
                                     # rather than asserting another one.
                                     "traced": 2}
    # `qc` abstains and that is the answer, not a gap: quality control asserts no structure, and
    # `check_claims` makes absence legal so a recipe can say so rather than pick the nearest word
    assert "qc" not in score.claimed_rungs()
    assert score.predicted_rung(["embed"]) == 0
    assert score.predicted_rung(["cflows", "embed"]) == 2
    assert score.predicted_rung([]) == 0
    assert score.predicted_rung(["nonsense"]) == 0


def test_there_is_no_second_home_for_a_recipes_rung():
    """The point of the swap. A module-level `RECIPE_RUNG` beside the YAML is a second home
    for one per-recipe fact, and `catalog.py`'s header names that as the failure this codebase
    keeps repeating."""
    assert not hasattr(score, "RECIPE_RUNG")


def test_a_new_recipe_reaches_the_rung_table_with_no_code_change(tmp_path):
    """The measured drift, made to fail. Before this, adding `configs/recipe/tree.yaml` put
    `tree` in the menu (`narrate.offer` is catalog-derived) and scored it rung 0 — the same
    number a recipe claiming a plain surface gets. Now it grades what it claims."""
    _write(tmp_path, "tree", "name: tree\nclaims: multi-branching\n"
                             "steps: [{name: diffusionmap, group: latent}]\n")
    assert score.claimed_rungs(tmp_path) == {"tree": 3}
    assert score.predicted_rung(["tree"], tmp_path) == 3


def test_a_recipe_that_claims_nothing_is_absent_rather_than_zero(tmp_path):
    """"Asserts nothing" and "asserts a surface" are different, and collapsing them is
    exactly the conflation the old table could not express."""
    _write(tmp_path, "quiet", "name: quiet\nsteps: [{name: pca, group: latent}]\n")
    assert score.claimed_rungs(tmp_path) == {}
    assert score.predicted_rung(["quiet"], tmp_path) == 0


def test_unroutable_rungs_closes_when_a_recipe_claims_the_gap(tmp_path):
    """`unroutable_rungs()` reports a gap in the ACTION SPACE, and that gap is a finding rather
    than an oversight. It is a function of the catalog: the recipes that close it close it by
    existing, which is what makes the number worth reporting.

    `[3]` on the bundled catalog since `archetypes` declared `claims: archetypal` — and the
    honest reading of that move is in `test_decisions.py`: this function reads what recipes
    CLAIM, not what they can RUN, and `archetypes` is `blocked:` upstream. Rung 4 is closed on
    paper and unreachable in fact."""
    assert score.unroutable_rungs() == [3]
    _write(tmp_path, "tree", "name: tree\nclaims: multi-branching\n"
                             "steps: [{name: diffusionmap, group: latent}]\n")
    _write(tmp_path, "arch", "name: arch\nclaims: archetypal\n"
                             "steps: [{name: aa, group: latent}]\n")
    _write(tmp_path, "flat", "name: flat\nclaims: surface\n"
                             "steps: [{name: pca, group: latent}]\n")
    _write(tmp_path, "grp", "name: grp\nclaims: clusters\n"
                            "steps: [{name: leiden, group: latent}]\n")
    _write(tmp_path, "line", "name: line\nclaims: single-trajectory\n"
                             "steps: [{name: phate, group: latent}]\n")
    assert score.unroutable_rungs(tmp_path) == []


def test_a_broken_recipe_file_drops_out_instead_of_taking_the_scorer_down(tmp_path):
    """Same split `catalog.known_steps` draws: a bad file in `$MANYRUNS_RECIPE_DIR` is not
    scoreable and is not fatal either. `load_recipe` stays strict for a recipe being RUN."""
    _write(tmp_path, "ok", "name: ok\nclaims: clusters\nsteps: [{name: pca, group: latent}]\n")
    _write(tmp_path, "broken", "name: mismatched\nsteps: []\n")
    assert score.claimed_rungs(tmp_path) == {"ok": 1}


# ── §6 stage 9: `via:` surfaces a stand-in through the caveat channel ────────
def test_the_in_process_loop_announces_that_mioflow_is_a_stand_in():
    """§2.4's rule: a recipe does not count as running until it runs with the algorithm its
    cited source specifies. The in-process `mioflow` is a diffusion pseudotime on a kNN graph
    (`steps._step_mioflow`), not the neural-ODE flow the recipe's source cites — and until now
    it announced that nowhere, while still emitting `pseudotime_range` into the g-vector."""
    cflows = catalog.load_recipe("cflows")
    notes = vocab.standins(cflows, "_inproc")
    assert len(notes) == 1
    assert notes[0].startswith("mioflow: ran on engine=_inproc's diffusion-pseudotime stand-in")
    assert "not the manylatents mioflow the recipe names" in notes[0]
    assert "running anyway" in notes[0]          # a caveat, not a refusal


def test_the_engine_that_runs_the_cited_method_says_nothing():
    """Quiet unless something genuinely unusual happened. A caveat channel that fires on every
    run is a channel nobody reads."""
    assert vocab.standins(catalog.load_recipe("cflows"), "manylatents") == []
    assert vocab.standins(catalog.load_recipe("cflows"), None) == []


def test_phate_on_engine_real_is_not_reported_as_a_stand_in():
    """The reason `standins` reads `STAND_INS` first and `via` second, rather than testing
    `via != engine`. `phate` is PHATE on both engines — the public `phate` package on `real`,
    `algorithms/latent/phate.py` on manylatents — so a blanket mismatch test would attach a
    caveat to a run in which nothing was substituted."""
    for name in ("embed", "contrast"):
        assert vocab.standins(catalog.load_recipe(name), "_inproc") == []
        assert vocab.standins(catalog.load_recipe(name), "manylatents") == []


def test_via_can_suppress_the_note_by_asking_for_that_engine():
    """The other half of the rule. A recipe that declares the stand-in as its canonical
    implementation got what it asked for, and that is not a substitution."""
    asked_for_it = {"name": "r", "steps": [{"name": "mioflow", "group": "lightning",
                                            "via": "_inproc"}]}
    assert vocab.standins(asked_for_it, "_inproc") == []


def test_a_recipe_with_no_via_is_still_told_about_the_stand_in():
    """`via` cannot say "this is a stand-in" — it says which implementation is canonical. So
    an undeclared step still gets the note: running a diffusion pseudotime under the name
    `mioflow` is a substitution whether or not anyone wrote it down."""
    silent = {"name": "r", "steps": [{"name": "mioflow", "group": "lightning"}]}
    notes = vocab.standins(silent, "_inproc")
    assert len(notes) == 1 and "not mioflow itself" in notes[0]


def test_stand_ins_ride_the_existing_caveat_channel_not_a_new_one():
    """§3.3: "that mismatch should go through the caveat channel that already exists, not a
    new one." Both kinds of note come out of one call, in one list — a second list would be
    two vocabularies for one question, which is how two answers drift apart."""
    # mioflow ALONE on a supplied embedding: both notes are live at once. In `cflows` the
    # `phate` step supplies the embedding canonically, so only the stand-in note fires.
    reuse = {"name": "reuse", "steps": [{"name": "mioflow", "group": "lightning",
                                         "via": "manylatents"}]}
    both = vocab.noncanonical(reuse, "manifold", provided={"embedding"}, engine="_inproc")
    assert len(both) == 2
    assert any("came from the caller" in n for n in both)      # the fact-provenance note
    assert any("stand-in" in n for n in both)                  # the via note


def test_omitting_the_engine_leaves_every_existing_caller_unchanged():
    """`engine` defaults to None so a caller with no engine in hand keeps its current answer
    rather than being told to invent one. `pipeline.runner` is exactly that caller today."""
    cflows = catalog.load_recipe("cflows")
    assert vocab.noncanonical(cflows, "time-course") == []
    assert vocab.noncanonical(cflows, "manifold", provided={"embedding"}) == []


def test_the_stand_in_note_reaches_a_finished_record(tmp_path):
    """End to end through `runner._finalize`, on the session path (`Session.results`), which
    is where the engine is in scope. The embedding is SUPPLIED rather than fitted, so this
    test needs scipy/sklearn and not PHATE."""
    from manyruns.session import Session

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 3))
    s = Session("standin", engine="_inproc", out_dir=tmp_path, seed=0, embedding=emb,
                recipe=catalog.load_recipe("cflows"))
    s.apply({"name": "mioflow", "group": "lightning", "params": {}, "via": "manylatents"})
    res = s.results()
    assert res["g_vector"].get("pseudotime_range") is not None   # it really did compute
    assert any("stand-in" in c for c in res["caveats"]), res["caveats"]


@pytest.mark.xfail(
    reason="the RECIPE path is one keyword short: `pipeline.runner` calls "
           "`vocab.noncanonical(recipe, 'unknown', provided=supplied)` at two sites "
           "(run_manylatents and run_inproc) without `engine=`, and `manyruns/pipeline/` "
           "is owned by another change this round. Measured on this checkout: "
           "run_inproc(cflows) returns caveats=[] with "
           "pseudotime_range=[0.0, 1.0] in the g-vector. Adding engine='_inproc' / "
           "engine='manylatents' at those two calls is the whole fix — everything else it "
           "needs is here and green.",
    strict=False,
)
def test_the_stand_in_note_reaches_a_recipe_run(tmp_path):
    """The same guarantee as `test_the_stand_in_note_reaches_a_finished_record`, for the path
    a whole-recipe run takes through the step loop. XPASSes the moment the runner passes its
    engine; kept live rather than written in prose so the gap is visible in the test output and
    not only in a report."""
    from manyruns.pipeline import runner

    res = runner.run_inproc(np.random.default_rng(0).normal(size=(60, 5)),
                              catalog.load_recipe("cflows"), tmp_path, seed=0)
    assert res["g_vector"].get("pseudotime_range") is not None
    assert any("stand-in" in c for c in res["caveats"]), res["caveats"]


def test_known_steps_carries_via_only_when_declared(tmp_path):
    """A step issued one-at-a-time in a session has to be the step dict the recipe would have
    issued, or the two paths report differently about one run. An unconditional `"via": None`
    would make every step claim a field it does not have."""
    steps = catalog.known_steps()
    assert steps["mioflow"]["via"] == "manylatents"
    assert steps["separation"]["via"] == "manyruns"
    _write(tmp_path, "plain", "name: plain\nsteps: [{name: pca, group: latent}]\n")
    assert "via" not in catalog.known_steps(tmp_path)["pca"]
