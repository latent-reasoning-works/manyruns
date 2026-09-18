"""The ledger as an instrument, not a chat log.

Every primitive this pane needs is ALREADY declared by the recipes — `id` names the method,
`claims` names the geometry the analysis assumes, `source.cite` names the paper — and until now
`narrate.offer` read none of them, returning only the prose `question:` field. The pane rendered
two clauses of first-person English per row, so a reader had to parse a sentence to learn what a
row would run.

These tests pin the primitives, not the prose. The prose still exists (`ask`/`gloss`) because the
one-shot surface wants a sentence; what is new is that a caller can now group by ASSUMPTION and
name the METHOD without asking the catalog a second question of its own.
"""
from __future__ import annotations

import pathlib

import pytest

from manyruns import narrate
from manyruns.catalog import discover_recipes, load_recipe

RECIPES = pathlib.Path(__file__).resolve().parent.parent / "manyruns" / "configs" / "recipe"


def test_every_claimed_geometry_has_a_glyph_and_no_others():
    """The exhaustiveness check, in BOTH directions — the same shape as
    `test_every_fact_the_calculus_names_has_a_phrase_and_no_others`.

    `ASSUMPTION_GLYPH` is a PRESENTATION table over the topology vocabulary, not a second
    declaration of it. A glyph for a word the vocabulary does not have is a private word list
    starting to drift; a vocabulary word with no glyph is a row that renders blank.
    """
    assert set(narrate.ASSUMPTION_GLYPH) == set(narrate._TOPOLOGY_PHRASE), (
        "the glyph table and the topology vocabulary disagree: "
        f"only-glyph={set(narrate.ASSUMPTION_GLYPH) - set(narrate._TOPOLOGY_PHRASE)}, "
        f"only-vocab={set(narrate._TOPOLOGY_PHRASE) - set(narrate.ASSUMPTION_GLYPH)}"
    )


def test_every_glyph_is_exactly_one_column_wide():
    """The pane wraps badly at 100 columns (measured), so a two-cell glyph silently breaks the
    only alignment the layout has. `len()` is not the check — an emoji is one character and two
    cells — so this asks the width that will actually be used.

    ASKED OF `rich`, WHICH IS WHAT DRAWS IT, and that is a correction rather than a swap. This
    read `wcwidth.wcswidth` until CI failed on it: `wcwidth` is not in the base install (the job
    whose value is what it LACKS), so the property went unchecked in the one environment that
    proves the product runs standalone — and it was being measured with a library manyruns
    neither imports nor ships. `rich.cells.cell_len` is the function the ledger's own layout is
    computed with, through textual, so a glyph that passes here is a glyph the pane will align.
    """
    from rich.cells import cell_len

    for word, glyph in narrate.ASSUMPTION_GLYPH.items():
        assert cell_len(glyph) == 1, f"{word!r}'s glyph {glyph!r} is not one column wide"


def test_every_claim_a_recipe_makes_can_be_drawn():
    """The join that matters: a recipe's `claims:` must be a word the glyph table knows. This is
    what makes "group the menu by assumption" total rather than best-effort."""
    unknown = {}
    for name in discover_recipes():
        claims = load_recipe(name).get("claims")
        if claims and claims not in narrate.ASSUMPTION_GLYPH:
            unknown[name] = claims
    assert unknown == {}, f"recipes claim a geometry with no glyph: {unknown}"


@pytest.fixture
def entries():
    """The menu as a caller sees it, for data that carries labelled groups."""
    obs = narrate.Observation(source="pbmc3k", shape="clusters", modality="scrna",
                              n_obs=2638, n_vars=2000)
    return {e["recipe"]: e for e in narrate.offer(obs)}


def test_offer_names_the_method_not_only_the_question(entries):
    """`sc.cluster.leiden` -> `leiden`. A scientist knows the method names; the menu hid them
    behind "What groups are in here?" and made them unsearchable."""
    assert entries["cluster"]["method"] == "leiden"
    assert entries["cflows"]["method"] == "mioflow"
    assert entries["pseudotime"]["method"] == "dpt"
    assert entries["archetypes"]["method"] == "aa"


def test_offer_exposes_the_assumption_and_its_glyph(entries):
    """The grouping key. `claims:` is already written in the topology vocabulary, so this is a
    lookup rather than a new fact."""
    assert entries["cluster"]["assumption"] == "clusters"
    assert entries["cluster"]["glyph"] == narrate.ASSUMPTION_GLYPH["clusters"]
    assert entries["pseudotime"]["assumption"] == "single-trajectory"
    assert entries["archetypes"]["glyph"] == narrate.ASSUMPTION_GLYPH["archetypal"]


def test_offer_carries_the_citation_short(entries):
    """The papers are declared and were never shown. Short form only — the pane has one line."""
    assert entries["archetypes"]["cite"] == "Cutler 1994"
    assert entries["cflows"]["cite"] == "Huguet 2022"
    assert entries["pseudotime"]["cite"].startswith("Haghverdi ")


def test_a_dev_recipe_never_reaches_the_product_menu(entries):
    """`sandbox` declares `id: dev.sandbox.stubs`, and its own header says it is "a recipe of
    steps that DO NOT WORK YET, so the app can be driven against them". It reached the shipped
    menu as "Drive the app — every step here declines, because none of them is built yet".

    Filtered by the `dev.` PREFIX it already declares, not by its name: a second dev fixture
    should not need this test edited to be excluded."""
    assert "sandbox" not in entries
    assert all(not e["id"].startswith("dev.") for e in entries.values())


def test_conditioning_recipes_are_not_given_a_false_assumption(entries):
    """`preprocess` and `qc` declare `claims: None` and `suits: []` because they are not
    analyses — they condition the data. They read as odd items in a list of asks precisely
    because they are a different KIND sharing one list, so they carry no assumption and the
    caller groups them apart."""
    for name in ("preprocess", "qc"):
        assert entries[name]["assumption"] is None
        assert entries[name]["glyph"] is None


def test_no_blocked_reason_names_a_package_or_a_kwarg():
    """THE REGRESSION THIS EXISTS FOR. `archetypes.yaml` shipped

        blocked: "blocked upstream — manylatents' `aa` rejects the `method` kwarg
                  (archetypes-package skew)"

    to a scientist's screen. A blocked row explains what the DATA lacks — `contrast`'s "needs a
    healthy/disease column" is the register — because a dependency's API drift is not something
    the reader can act on, and reads as the product being broken.

    Checked over the YAML rather than the rendered row so it fails at the source, where the copy
    is written."""
    import yaml
    offenders = {}
    assert RECIPES.is_dir(), f"recipe config directory moved: {RECIPES}"
    for path in sorted(RECIPES.glob("*.yaml")):
        reason = (yaml.safe_load(path.read_text()) or {}).get("blocked")
        if not reason:
            continue
        lowered = reason.lower()
        for tell in ("kwarg", "manylatents", "upstream", "package", "traceback", "()"):
            if tell in lowered:
                offenders[path.name] = reason
    assert offenders == {}, (
        f"a blocked reason names the code instead of the data: {offenders}")


# ── the third input: what the numbers say about the result ───────────────────
def _measurement(**over):
    """A counts-like measurement shaped like `qc.facts` output, with the gene axis mostly
    empty — the finding `pbmc3k_raw.h5ad` actually carries (19,024 of 32,738)."""
    base = {"qc_n_cells": 2700, "qc_n_genes": 32738, "qc_would_drop_genes": 19024,
            "qc_would_drop_cells": 0, "qc_would_drop_cells_mito": 57,
            "qc.min_genes": 200, "qc.min_cells": 3, "qc.max_pct_mito": 5.0}
    return base | over


def test_a_gene_axis_concern_implicates_exactly_the_recipes_whose_steps_touch_it():
    """THE JOIN, PINNED BY WHAT IT DERIVES AND NOT BY A LIST. A recipe is implicated when its
    steps need or produce a fact the concern is computed into (`genes`, `clusters`) AND it
    produces something — so the conditioning recipes, which produce nothing and are what you
    would run to fix it, are exempt. Measured over the bundled catalog: exactly two."""
    from manyruns import vocab

    fired = {n for n in discover_recipes() if narrate.concerns(load_recipe(n), _measurement())}
    expected = set()
    for n in discover_recipes():
        r = load_recipe(n)
        produces = set().union(*(vocab.step_provides(s) for s in (r.get("steps") or [])))
        touches = (set(vocab.recipe_needs(r)) | produces) & {"genes", "clusters"}
        if produces and touches:
            expected.add(n)
    assert fired == expected
    assert len(fired) == 2, f"the bundled catalog should implicate two, got {sorted(fired)}"


def test_a_caution_never_disables_a_row():
    """FREESTYLE, PINNED. The marker informs; `can_run` is legality and is computed exactly as
    before. A row the numbers warn about is as selectable as one they do not."""
    obs = narrate.Observation(source="x", shape="clusters", modality="scrna",
                              n_obs=2700, n_vars=32738)
    from manyruns.tui import state
    rows = {r.recipe: r for r in state.ledger(obs, ("genes",), measured=_measurement())}
    warned = [r for r in rows.values() if r.caution]
    assert warned, "the fixture must warn about something or this proves nothing"
    for r in warned:
        assert r.can_run, f"{r.recipe}: a caution turned into a refusal"
        assert r.measured == "caution"


def test_a_measurement_over_not_counts_raises_no_concern():
    """The cuts are defined over counts. `pbmc3k_processed.h5ad` and `tree8.h5ad` are not, and
    `qc.facts` says so with `qc_input`; a concern that fired there would report a threshold
    artefact as a finding."""
    m = _measurement(qc_input="this matrix does not hold integer counts")
    assert all(narrate.concerns(load_recipe(n), m) == [] for n in discover_recipes())


def test_no_measurement_and_an_empty_one_are_different_blanks():
    """`None` is "nothing was asked" and `{}` is "asked, nothing to count" — a synthetic
    dataset must not be read as clean, and a point cloud must not be read as unmeasured."""
    from manyruns.tui import state
    obs = narrate.Observation(source="x", shape="clusters", modality="scrna")
    unasked = state.ledger(obs, measured=None)
    empty = state.ledger(obs, measured={})
    assert all(r.measured in ("unmeasured", "refused") for r in unasked)
    assert all(r.measured in ("clear", "refused") for r in empty)


def test_the_decision_record_carries_the_word_and_not_the_sentence():
    """The corpus records what the screen SHOWED. Without the word it cannot tell "picked the
    row the numbers warned about" from "picked a row nothing was known about"; with the
    sentence it would re-label itself on every rewording."""
    from manyruns.tui import state
    obs = narrate.Observation(source="x", shape="clusters", modality="scrna",
                              n_obs=2700, n_vars=32738)
    rows = state.ledger(obs, ("genes",), measured=_measurement())
    offers = {r.recipe: r.as_offer() for r in rows}
    assert {o["measured"] for o in offers.values()} >= {"caution", "clear"}
    assert all("caution" not in o or not isinstance(o.get("caution"), str) for o in offers.values())
    assert all("sentence" not in o for o in offers.values())


# ── the record as one page: what a model is handed when someone asks about the run ────────────
def _entry(**over):
    """The roster's own row for `pbmc3k_raw.h5ad`, as the mapping it already serialises to.

    Built through `DataEntry.to_dict()` rather than written out as a literal, because "a
    `DataEntry`-shaped mapping" is only a real contract if the shape comes from the class. A
    field renamed there should break this, not silently stop reaching the prompt.
    """
    from manyruns.tui.state import DataEntry

    obs = narrate.Observation(source="pbmc3k_raw.h5ad", shape="single", modality="scrna",
                              n_obs=2700, n_vars=32738)
    facts = _measurement(qc_median_genes_per_cell=817.0, qc_pct_mito_median=2.0296, **over)
    return DataEntry(name="pbmc3k_raw.h5ad", kind="file", obs=obs, qc=facts).to_dict()


def _step(name, state="ok", seconds=0.5, deltas=(), detail=""):
    """One `StepView` as `RunFeed.rows()` would hand it over, marks and all.

    The `(glyph, word)` pair comes from `narrate.step_mark` over a record shaped the way
    `RunFeed._view` shapes it — `None` for a declared step that has not started, a live `state`
    while it runs, an `outcome` once it settles. Those three are not interchangeable keys:
    `step_mark` reads a live record by `state` and a settled one by `outcome`, so a fixture
    that fed the outcome slot the word `queued` would agree with the code under test and with
    nothing that was ever on screen.
    """
    from manyruns.tui.state import StepView

    if state == "queued":
        rec = None
    elif state == "running":
        rec = {"state": state}
    else:
        rec = {"outcome": state}
    glyph, word = narrate.step_mark(rec)
    return StepView(index=0, name=name, state=state, glyph=glyph, word=word, seconds=seconds,
                    detail=detail,
                    deltas=tuple(narrate.GeometryRow(label, value) for label, value in deltas))


#: A run mid-flight: two settled steps, one that declined with a reason, one embedding step
#: carrying THREE of the five deltas an embedding produces — enough to prove every delta
#: reaches the record without the fixture pretending to be a measurement. (The real five, per
#: `runner.run_inproc` over `pbmc3k_raw.h5ad`: `lid`, `loglog_consistency`, `anisotropy`,
#: `participation_ratio`, `outlier_score`.)
def _steps():
    return [
        _step("normalize", seconds=0.55),
        _step("pca", state="skipped", seconds=0.0, detail="no in-process implementation"),
        _step("phate", seconds=3.25, deltas=[("lid", "— → 2.534"),
                                             ("anisotropy", "— → 0.7116"),
                                             ("outlier_score", "— → 1.347")]),
        _step("leiden", state="queued", seconds=None),
    ]


def test_every_step_and_every_number_on_screen_reaches_the_record():
    """THE ONE PROPERTY THE ANSWER RESTS ON. The ask is a single call with the whole record in
    the prompt (spec §1), so a fact that does not render here is a fact the model cannot have
    been asked about — and it would answer anyway."""
    recipe = load_recipe("cluster")
    text = narrate.record_for_prompt(entry=_entry(), recipe=recipe, steps=_steps(),
                                     tunables=[("knn", 40)], said=["kept attempt 1 as phate@1.png"])

    for name in ("normalize", "pca", "phate", "leiden"):
        assert name in text, f"{name} is not in the record the model reads"
    # every delta, every number — the label AND both sides of the pair
    for number in ("lid", "2.534", "anisotropy", "0.7116", "outlier_score", "1.347"):
        assert number in text
    # the state word and the duration, in the steps pane's own two decimals
    assert "skipped" in text and "queued" in text and "3.25s" in text
    assert "knn=40" in text and "kept attempt 1 as phate@1.png" in text
    # the recipe, its commitment and its chain
    assert "clusters — assume separate groups" in text
    for step in recipe["steps"]:
        assert step["name"] in text


def test_the_reason_a_step_declined_is_in_the_record_with_the_step():
    """A DEVIATION FROM §3'S LIST, PINNED. It names name/state/seconds/deltas; the detail line
    is rendered too, because `skipped` alone does not say why. Measured on this tree: three of
    the clustering recipe's six steps decline under `runner.run_inproc` and the only thing that
    says why is the sentence under each of them."""
    text = narrate.record_for_prompt(steps=_steps())

    assert "no in-process implementation" in text


def test_a_step_is_in_the_record_under_the_word_the_screen_showed_it_by():
    """`StepView` CARRIES BOTH AND ONLY ONE OF THEM IS THE ACCOUNT. The outcome `reported` is
    drawn as "ran" and the live `running` as "running…" (`narrate.step_mark`), so a record that
    printed the raw state would hand the model a second vocabulary for how a step ended — told
    it a step was "reported" where the person watching was told it ran. That is the failure
    "the model reads what the person read" names, arriving one word at a time."""
    settled = narrate.record_for_prompt(steps=[_step("phate", state="reported", seconds=1.0)])
    live = narrate.record_for_prompt(steps=[_step("phate", state="running", seconds=None)])

    assert "ran" in settled and "reported" not in settled
    assert "running…" in live


def test_every_qc_number_reaches_the_record_with_the_cut_it_was_measured_against():
    """`qc_thresholds`' argument, one layer out: a drop count without its cut is meaningless,
    and a model given the count alone will read 19,024 as a verdict rather than as the output of
    a threshold someone chose."""
    text = narrate.record_for_prompt(entry=_entry())

    assert "19,024 genes" in text        # what a standard filter would remove, gene axis
    assert "57 cells" in text            # and the cell axis — `qc_drops` names both halves
    assert "817" in text                 # median genes per cell
    assert "2.0 %" in text               # median mitochondrial share
    assert "2,700 × 32,738" in text      # the size the roster shows
    for cut in ("<200 genes/cell", "gene in <3 cells", ">5.0% mito"):
        assert cut in text, f"the record drops the cut {cut!r} the numbers were measured against"


def test_the_same_run_renders_the_same_bytes_and_the_same_name():
    """WHAT MAKES `context_sha256` MEAN ANYTHING. The event carries the hash instead of the
    prompt (spec §2), so "these two answers were asked over one record" is only checkable if
    equal inputs render equal bytes — no clock, no path, no set iteration.

    The arguments are BUILT TWICE rather than splatted twice, because the property is *equal*
    inputs and not the same objects: handing both calls one dict would pass for a rendering
    that reached for `id()` or mutated what it was given, which are two of the three things
    determinism here has to rule out.
    """
    def args():
        return dict(entry=_entry(), recipe=load_recipe("cluster"), steps=_steps(),
                    tunables=[("knn", 40)], said=["one"],
                    concerns=["a caution the ledger drew"])

    first = narrate.record_for_prompt(**args())
    second = narrate.record_for_prompt(**args())

    assert first == second
    assert narrate.context_sha256(first) == narrate.context_sha256(second)
    assert len(narrate.context_sha256(first)) == 64


def test_one_moved_number_renames_the_context():
    """The other half: a hash that did not move when the record did would let a reader join an
    answer to a run it was never given."""
    before = narrate.record_for_prompt(steps=_steps())
    after = narrate.record_for_prompt(steps=[_step("phate", seconds=3.26)])

    assert narrate.context_sha256(before) != narrate.context_sha256(after)


def test_the_panes_that_are_always_on_screen_are_always_in_the_record():
    """A run screen always shows a dataset, a recipe and a steps pane, so their absence has to
    be SAID. An omitted heading is the one thing this reader cannot survive — a person skims a
    gap, a model fills it in."""
    text = narrate.record_for_prompt()

    assert text.startswith("dataset: —")
    assert "recipe: —" in text
    # the steps pane's own sentence, verbatim — an engine that declared nothing and reported
    # nothing must not read here as a clean run either
    assert "no steps declared, and none reported" in text


def test_a_section_the_screen_collapses_is_absent_rather_than_headed():
    """`caveats_view`'s rule, applied to a reader that answers rather than skims: a placeholder
    is a claim. The strip, the said lines and the caution collapse to nothing on screen when
    they hold nothing, so they are not written at all here."""
    text = narrate.record_for_prompt(entry=_entry(), recipe=load_recipe("cluster"),
                                     steps=_steps())

    for heading in ("tunable:", "said:", "caution:"):
        assert heading not in text, f"{heading} is written over nothing"


def test_the_caution_in_the_record_is_the_one_the_ledger_drew():
    """PASSED IN, NEVER RECOMPUTED, even though `recipe` and the QC facts are both to hand.
    What belongs in the record is the sentence the person was shown — a second call to
    `concerns` here could disagree with the pane (a file filtered since it was measured) and
    the answer would be about a run nobody watched."""
    recipe = load_recipe("cluster")
    assert narrate.concerns(recipe, _measurement()), "the fixture must warn or this proves nothing"

    silent = narrate.record_for_prompt(entry=_entry(), recipe=recipe, steps=_steps())
    drawn = narrate.record_for_prompt(recipe=load_recipe("preprocess"),
                                      concerns=["a caution the ledger drew"])

    assert "caution:" not in silent
    assert "a caution the ledger drew" in drawn


def test_a_record_can_be_rendered_with_no_terminal():
    """The seam `tui.state` opens, held from this end. In a FRESH interpreter, so another
    test's import cannot pollute `sys.modules`: rendering a whole run must not load textual or
    rich, which is what lets the prompt — and commit 5's worker — be asserted on headless."""
    import os
    import subprocess
    import sys

    code = ("from manyruns import narrate; "
            "narrate.record_for_prompt(entry={'name': 'x'}, said=['hi']); "
            "import sys; "
            "leaked=[m for m in ('textual', 'rich') if m in sys.modules]; "
            "assert not leaked, leaked")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)

    assert proc.returncode == 0, proc.stderr



def test_the_prompts_qc_labels_are_the_rosters_labels():
    """`_QC_PROMPT_LABELS` repeats the roster's column headings so a reader who saw the roster
    recognises the same words in the record. The dependency runs one way (the roster imports
    narrate, never the reverse), so it cannot be a shared constant — this pins the copy
    instead, in both directions, the way `_TOPOLOGY_PHRASE` is pinned to its glyph table."""
    from manyruns.tui import roster

    theirs = {k: h for k, h in roster.COLUMNS if k in dict(narrate._QC_PROMPT_LABELS)}
    assert dict(narrate._QC_PROMPT_LABELS) == theirs
    assert [k for k, _ in narrate._QC_PROMPT_LABELS] == [k for k, _ in roster.COLUMNS if k in theirs]


def test_the_recipe_scan_fails_if_its_source_directory_moves(monkeypatch, tmp_path):
    """An absent recipe root used to scan zero YAMLs and report no offenders."""
    monkeypatch.setitem(globals(), "RECIPES", tmp_path / "moved-recipes")
    with pytest.raises(AssertionError, match="recipe config directory moved"):
        test_no_blocked_reason_names_a_package_or_a_kwarg()
