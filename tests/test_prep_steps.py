"""The `prep` group — `manyruns/pipeline/prep.py`.

A prep executor is a PURE FUNCTION: it receives a view and returns a result dict, and it
writes to neither `state` nor the frame. That is not style. It is what makes the step loop the
only place a selection is composed, and therefore the only place that has to be correct.

The old state held five independently writable row-indexed slots (`runner._STATE_KEYS`), so a
step that dropped cells had to update all of them plus `ctx["color"]`, and missing one failed
SILENTLY — `io.py:106` drops the colouring on a length mismatch rather than raising. A step
that returns a mask cannot forget a slot because it never touches one.
"""
from __future__ import annotations

import numpy as np
import pytest

from manyruns import vocab
from manyruns.pipeline import prep as _prep
from manyruns.vocab import INPROC


def _view(counts=None, genes=None, labels=None):
    """A view as `frame.view` builds one — the three slots, some of them absent."""
    return {"counts": counts, "genes": genes, "labels": labels}


def test_prep_is_a_declared_step_group():
    """A step's `group` is validated against this tuple by `catalog.check_recipe`, and
    `runner.dispatchable` reads it to prune a step an engine cannot run. A new capability is a
    group plus an executor — CLAUDE.md's prescription, used rather than added to."""
    assert "prep" in vocab.STEP_GROUPS


def test_transform_returns_a_new_matrix_and_writes_nothing():
    """The purity contract. `sc.pp.normalize_total` writes IN PLACE (steps.py:435-437), so a
    prep step that took `state` and mutated it would corrupt the frame the record claims is
    the original."""
    X = np.array([[0.0, 1.0], [3.0, 7.0]])
    before = X.copy()
    out = _prep._step_transform(_view(), X, {"method": "log1p"})

    assert set(out) <= {"X", "report"}
    assert np.allclose(out["X"], np.log1p(before))
    assert np.array_equal(X, before), "the caller's array is untouched"


def test_transform_supports_the_sqrt_convention_the_EB_data_uses():
    """`--transform sqrt` exists because the PHATE/MIOFlow embryoid-body convention is sqrt,
    not log1p (`loading._anndata_matrix`'s signature). It becomes this param."""
    out = _prep._step_transform(_view(), np.array([[4.0, 9.0]]), {"method": "sqrt"})

    assert np.allclose(out["X"], [[2.0, 3.0]])


def test_an_unknown_transform_method_is_refused_rather_than_defaulted():
    """Defaulting a typo'd method to log1p would produce a run that reports `ok` having
    silently done something other than what the recipe declared."""
    with pytest.raises(ValueError, match="method must be one of"):
        _prep._step_transform(_view(), np.array([[1.0]]), {"method": "lg1p"})


def test_transform_reads_the_working_matrix_not_the_untransformed_counts():
    """Order is load-bearing: `normalize` then `transform` must compose. `transform` reading
    `view["counts"]` would silently discard the normalization and log1p the raw counts — a run
    that reports two steps `ok` and applied one of them."""
    counts = np.array([[100.0, 100.0]])
    normalized = np.array([[0.5, 0.5]])
    out = _prep._step_transform(_view(counts=counts), normalized, {"method": "log1p"})

    assert np.allclose(out["X"], np.log1p(normalized))


def test_the_group_dispatches_by_name_from_a_table_manyruns_owns():
    """`runner._NAME_TABLES` makes "can this engine run this step" answerable WITHOUT running
    it, which is what `dispatchable` needs to prune the shell's menu honestly."""
    from manyruns.pipeline import runner

    assert runner.dispatchable(INPROC, "transform", "prep") is True
    assert runner.dispatchable(INPROC, "not_a_prep_step", "prep") is False


def test_every_engine_can_run_a_prep_step():
    """Unlike `latent`, prep is not an engine algorithm — it is scanpy, which every install
    has. An engine missing the executor would report `skipped (no 'prep' executor on this
    engine)` for the standard preamble, which is not a fidelity difference but a hole."""
    from manyruns.pipeline import runner

    # `INPROC` is not an engine (it is `vocab.INPROC`, the in-process step loop CI runs),
    # but it is a row of `_DISPATCH` and a prep step must reach it the same way.
    for engine in ("mock", INPROC, "manylatents"):
        assert runner.dispatchable(engine, "transform", "prep") is True, engine


def test_a_prep_step_on_a_frameless_state_declines_with_a_reason():
    """The frame arrives in Task 5, and until it does a hand-written prep recipe would record
    `KeyError: 'frame'` — an error that says nothing about what to do next. `_StepSkipped` is
    how a step declines, and `apply_step` turns it into `outcome="skipped"` with the reason
    attached, which is what the record is for."""
    from manyruns.pipeline import runner
    from manyruns.pipeline.steps import _StepSkipped

    with pytest.raises(_StepSkipped, match="carries no frame"):
        runner._run_prep_step("transform", {}, {"X": np.ones((2, 2))}, {}, {})


def test_a_recipe_declaring_a_prep_step_validates():
    """`check_recipe` refuses a group outside `STEP_GROUPS`, so this is the gate a recipe with
    a prep block has to pass before Task 8 can migrate the bundled eight."""
    from manyruns import catalog

    recipe = {"name": "t", "steps": [
        {"name": "transform", "group": "prep", "params": {"method": "log1p"}, "via": "scanpy"}]}

    assert catalog.check_recipe(recipe, "t") == []


# ── the five that delegate to manylatents (Tasks 6 + 7) ──────────────────────────────────
#
# These are ADAPTERS, so the tests below are about the adaptation and not about the maths:
# does the right upstream function get called, do its arguments come from the declared params,
# and does its answer arrive in this module's shape. The algorithms themselves are tested in
# manylatents-omics (#61) and re-testing them here would be a second copy of that suite that
# drifts.
pytestmark_ops = pytest.importorskip("manylatents.singlecell.preprocessing",
                                     reason="the prep steps delegate to the [omics] extra")


def test_every_declared_prep_step_is_dispatchable():
    """The registration half, which is what makes these reachable at all: `runner.dispatchable`
    reads `_PREP_STEPS` by name, so a step implemented but unregistered is invisible to the
    planner and to the shell. `hvg` is deliberately absent — spec §12 hands it to Zach."""
    assert set(_prep._PREP_STEPS) == {"filter_cells", "filter_genes", "filter_mito",
                                      "detect_doublets", "normalize", "transform"}
    assert "hvg" not in _prep._PREP_STEPS, "reserved in NARROWING_STEPS, not implemented here"


def test_a_doublet_call_removes_nothing_unless_the_recipe_says_remove(monkeypatch):
    """THE INVERSION, and the reason this test exists at all. Upstream returns `is_doublet`
    where True means SUSPECT; a mask here means SURVIVES. Handing the upstream array back
    unchanged would keep ONLY the doublets — the exact inverse — and still report `ok`.

    Default is flag-without-narrowing, which is also what `vocab.NARROWING_STEPS` keys on: it
    lists `detect_doublets` under `remove`, so with `remove: false` the all-True mask means a
    recipe placing it before an embedding is legal and must not be refused.

    UPSTREAM IS STUBBED WITH A KNOWN ANSWER, deliberately. The real call runs scrublet, which
    needs `scikit-image` for its automatic threshold and a 30-component PCA for anything else —
    measured, both refuse on a small synthetic matrix. Neither is what this adapter owns: the
    inversion is, and a stub is the only way to assert it against an answer whose truth is known
    rather than whichever cells scrublet happened to flag."""
    counts = np.ones((5, 4), dtype=np.float32)
    is_doublet = np.array([False, True, False, True, False])
    monkeypatch.setattr(_prep._ops(), "detect_doublets",
                        lambda *_a, **_k: is_doublet, raising=True)

    flagged = _prep._step_detect_doublets(_view(counts), counts, {})
    assert flagged["mask"].tolist() == [True] * 5, "remove defaults to False: nothing is dropped"
    assert flagged["report"]["detect_doublets.found"] == 2, "but it still SAYS what it found"
    assert flagged["report"]["detect_doublets.removed"] == 0

    removed = _prep._step_detect_doublets(_view(counts), counts, {"remove": True})
    assert removed["mask"].tolist() == [True, False, True, False, True], "survivors, not suspects"
    assert removed["report"] == {"detect_doublets.found": 2, "detect_doublets.removed": 2}


def test_a_column_filter_answers_over_genes_and_a_row_filter_over_cells():
    """The axis is what `runner._apply_transition` composes into a selection, so a step that
    returns the right mask on the wrong axis narrows the wrong dimension of the frame."""
    counts = np.array([[5, 0, 0, 3], [4, 0, 1, 2], [6, 0, 0, 1]], dtype=np.float32)
    genes = ["a", "b", "c", "d"]

    rows = _prep._step_filter_cells(_view(counts, genes), counts, {"min_genes": 3})
    cols = _prep._step_filter_genes(_view(counts, genes), counts, {"min_cells": 2})

    assert rows["axis"] == "rows" and rows["mask"].shape[0] == counts.shape[0]
    assert cols["axis"] == "cols" and cols["mask"].shape[0] == counts.shape[1]
    assert cols["mask"].tolist() == [True, False, False, True], "b is never seen, c only once"


def test_the_gene_reading_steps_refuse_rather_than_match_nothing():
    """`vocab.STEP_NEEDS` declares `genes` for both, so `unmet` normally prunes them first —
    this is the backstop. It REFUSES rather than degrading because "removed 0 genes" from a
    nameless matrix is indistinguishable from a dataset where nothing was removable, and that
    conflation is the exact thing the `genes` fact was added to remove."""
    counts = np.ones((4, 3), dtype=np.float32)

    for step in (_prep._step_filter_mito, _prep._step_filter_genes):
        with pytest.raises(ValueError, match="gene names"):
            step(_view(counts, genes=None), counts, {})


def test_normalize_reads_the_counts_and_transform_reads_the_working_matrix():
    """The two-argument rule, and the one place it points opposite ways. A library size is a sum
    of COUNTS, so `normalize` must not read a matrix an earlier `transform` already logged; a
    transform must read the WORKING matrix, or `normalize → transform` would silently discard
    the normalization. Measured by handing the two arguments different data."""
    counts = np.array([[1.0, 3.0], [2.0, 2.0]], dtype=np.float32)
    working = np.full((2, 2), 100.0, dtype=np.float32)

    normalized = _prep._step_normalize(_view(counts), working, {"target_sum": 10.0})
    assert np.allclose(np.asarray(normalized["X"]).sum(axis=1), 10.0), "scaled the COUNTS"

    transformed = _prep._step_transform(_view(counts), working, {"method": "log1p"})
    assert np.allclose(transformed["X"], np.log1p(working)), "logged the WORKING matrix"


def test_a_declared_param_reaches_upstream_rather_than_its_default():
    """The failure this whole layer is exposed to: an adapter that accepts a param, reports it,
    and never forwards it. Both halves are asserted — the report AND a result that could only
    come from the declared value."""
    counts = np.array([[5, 0, 1], [4, 0, 1], [6, 0, 0]], dtype=np.float32)

    loose = _prep._step_filter_genes(_view(counts, ["a", "b", "c"]), counts, {"min_cells": 2})
    tight = _prep._step_filter_genes(_view(counts, ["a", "b", "c"]), counts, {"min_cells": 3})

    assert loose["report"]["filter_genes.min_cells"] == 2
    assert tight["report"]["filter_genes.min_cells"] == 3
    assert loose["mask"].tolist() == [True, False, True]
    assert tight["mask"].tolist() == [True, False, False], "c is in 2 cells, not 3"


def test_transform_still_refuses_a_method_upstream_would_accept_silently():
    """The check stays in manyruns even though the maths left. `TRANSFORMS` is this product's
    closed vocabulary; a typo defaulting to log1p is a run that reports `ok` having done
    something the recipe did not declare."""
    counts = np.ones((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="method must be one of"):
        _prep._step_transform(_view(counts), counts, {"method": "logp1"})
