"""The app must not deny the labels it is drawing — and must not change a recipe to stop.

THE CONTRADICTION. Both demo fixtures carry one `.obs` column of eight names —
`data/tree8.h5ad` (`branch`) and `data/pbmc3k_annotated.h5ad` (`cell_type`) — and both are
plotted coloured by it. On the same screens the app said, verbatim:

    I see 2,638 cells, one population, no groups or time labels I recognize.   (narrate.describe)
    2,638 × 32,738 · one population                                            (the roster row)
    Mapped the structure into 3 dimensions — no groups or time labels to test
    against, so this is the shape itself, laid out for you to read.            (narrate.narrate)

The first two are about the FILE and were false. The third is about the RUN and was not even
derivable: measured on `runner.run_inproc(X, embed, labels=…, label_kind="group")`, `labels`
and `label_kind` reach the run state and stop — `runner._finalize` puts neither in `results` —
so `narrate` was asserting a fact about labels from a dict that has never contained one.

THE PRICE THAT WAS NOT PAID. The obvious fix is to read the group column as a data SHAPE, so
`single` becomes `clusters`. That moves no legality — `vocab.SHAPE_PROVIDES` maps both to `()`,
so `vocab.unmet` is byte-identical across the two on every discovered recipe — but it does flip
`app._choose_recipe`'s pick from `embed` to `cluster` (which runs leiden and *discovers* groups,
ignoring the eight curated ones), and it silences `narrate.refusal`, whose shape tuple does not
list `clusters`. So the shape did NOT move; only the sentences did. These tests pin both halves,
because the suite otherwise cannot tell them apart: `test_analysis_selection.py` is entirely
folder-based and never reaches `app._inspect_obs`.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from manyruns import narrate
from manyruns.narrate import Observation

#: The two fixtures the demo runs on, and the `.obs` column each one is coloured by.
_FIXTURES = (("data/tree8.h5ad", "branch"), ("data/pbmc3k_annotated.h5ad", "cell_type"))


# ── the sentences (dep-free: string logic over an Observation) ───────────────
def test_describe_names_the_column_it_is_about_to_colour_by():
    """The COLUMN, not just "8 groups": it is the fact a user can go and check for themselves,
    and `shell._loaded_stats` already spends the bare word "groups" on `obs.conditions`, which
    is a different axis."""
    obs = Observation(shape="single", modality="scrna", n_obs=2638, n_vars=32738,
                      group_key="cell_type",
                      groups=["CD4 T cells", "CD14+ Monocytes", "B cells", "CD8 T cells",
                              "NK cells", "FCGR3A+ Monocytes", "Dendritic cells",
                              "Megakaryocytes"])
    said = narrate.describe(obs)
    assert "cell_type" in said and "8 groups" in said and "CD4 T cells" in said
    assert "no groups" not in said, "it is drawing eight of them"


def test_describe_truncates_a_long_group_list_and_never_the_count():
    """A `leiden` column can carry 40 categories. The sentence must not become the vocabulary,
    and the elision must not hide how many there were."""
    obs = Observation(shape="single", modality="scrna", n_obs=5000,
                      group_key="leiden", groups=[f"c{i:02d}" for i in range(40)])
    said = narrate.describe(obs)
    assert "40 groups" in said and "…" in said
    assert "c00" in said and "c39" not in said


def test_describe_without_a_group_axis_is_the_sentence_it_always_was():
    """The fix is additive: a file with nothing to say still says the old thing."""
    obs = Observation(shape="single", modality="scrna", n_obs=2700)
    assert narrate.describe(obs) == (
        "I see 2,700 cells, one population, no groups or time labels I recognize.")


def test_contents_lets_a_declaration_outrank_the_sniff_and_the_sniff_outrank_the_shape():
    """`narrate.contents` is the string the TUI actually renders (the roster's "what's in it"
    column, re-rendered as the ledger's border subtitle) — `describe` is the rich shell's, and
    the TUI never calls it. A declared `topology:` is reviewed and still wins; a counted group
    axis beats the shape WORD, because the shape word was the thing lying."""
    assert narrate.contents("single", ["multi-branching"], ["a", "b"]) == "a branching tree"
    assert narrate.contents("single", None, ["a", "b", "c"]) == "3 labelled groups"
    assert narrate.contents("single", None, None) == "one population"


def test_the_run_readout_claims_the_recipe_and_not_the_data():
    """`results` carries no `labels` and no `label_kind` (measured: `runner._finalize` emits
    neither), so a sentence about what the DATA is labelled with was never derivable here. What
    is derivable is that no readout was produced — a fact about the steps that ran."""
    said = narrate.narrate({"final_dim": 3, "g_vector": {}})
    assert "Mapped the structure into 3 dimensions" in said
    assert "no groups or time labels to test against" not in said


# ── the invariant: the sentences moved, the offered analyses did not ─────────
@pytest.mark.parametrize("path,column", _FIXTURES)
def test_the_group_axis_is_read_for_narration_and_changes_no_recipe(path, column):
    """The whole point of the conservative fix, and the ONLY thing that catches it going wrong.

    A group fact leaking into `narrate.shape_of` leaves the rest of the suite green — nothing
    else asserts on the shape of these two files — and hands the demo a different recipe than it
    ran yesterday. So: same shape, same auto-selection, same recommendation, same refusal, and
    the sentence now names the column.
    """
    pytest.importorskip("anndata")
    p = Path(path)
    if not p.is_file():
        pytest.skip(f"{path} is a checkout-relative fixture; see docs/sidecars/")

    from manyruns import app

    obs = narrate.read_data(p, "scrna")
    assert obs.shape == "single", "reading the labels must not restate them as a data shape"
    assert obs.group_key == column and len(obs.groups) == 8

    assert app.select_analysis("scrna", p)[0]["name"] == "embed"
    assert [o["recipe"] for o in narrate.offer(obs) if o["recommended"]] == ["embed"]
    assert narrate.refusal("show me the trajectory over time", obs), \
        "the trajectory refusal is the app's headline honesty behaviour; it must still fire"

    said = narrate.describe(obs)
    assert column in said and "8 groups" in said
    assert "no groups or time labels I recognize" not in said


@pytest.mark.parametrize("path,column", _FIXTURES)
def test_the_legal_set_is_identical_to_the_one_a_group_blind_read_would_give(path, column):
    """Legality is the line this pass promised not to cross, so it is checked directly rather
    than inferred from the shape being unchanged: `vocab.unmet` over every discovered recipe,
    against the shape the group-blind derivation produces from the same two facts."""
    pytest.importorskip("anndata")
    p = Path(path)
    if not p.is_file():
        pytest.skip(f"{path} is a checkout-relative fixture; see docs/sidecars/")

    from manyruns import vocab
    from manyruns.catalog import discover_recipes, load_recipe

    obs = narrate.read_data(p, "scrna")
    blind = narrate.shape_of(obs.signals["has_time"], obs.signals["conditions"], obs.modality)
    for name in discover_recipes():
        recipe = load_recipe(name)
        assert (list(vocab.unmet(recipe, obs.shape, provided=obs.provides()))
                == list(vocab.unmet(recipe, blind, provided=obs.provides()))), name
