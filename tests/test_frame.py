"""The frame and the selection — `manyruns/pipeline/frame.py`.

WHAT IS UNDER TEST IS THE ABSENCE OF A SECOND COPY. The old state held `X`, `emb`,
`pseudotime`, `labels` and `counts` as five independently writable row-indexed slots, so a
step that dropped cells had to update all five plus `ctx["color"]`, and missing one failed
SILENTLY (`manyruns/pipeline/io.py:106` drops the colouring on a length mismatch rather than
raising). Under a frame there is one original and one selection, so the tests below are not
checking that five updates agree — they are checking there is only ever one.
"""
from __future__ import annotations

import numpy as np
import pytest

from manyruns.pipeline import frame as _frame


def _demo():
    counts = np.arange(20, dtype=np.float32).reshape(5, 4)
    genes = np.array(["a", "b", "c", "d"])
    labels = np.array(["d0", "d0", "d3", "d3", "d9"])
    return _frame.new_frame(counts=counts, genes=genes, labels=labels)


def test_a_fresh_selection_is_every_index_in_order():
    """A run starts seeing all of its data. `arange` rather than a boolean mask because a
    selection COMPOSES — `narrow` indexes it — and an index array is closed under that."""
    assert _frame.full_selection(5).tolist() == [0, 1, 2, 3, 4]
    assert _frame.full_selection(5).dtype == np.int64


def test_narrowing_composes_and_the_mask_is_over_the_CURRENT_view():
    """The second filter sees three cells, not five, so its mask is length 3. Indices stay in
    ORIGINAL space, which is what makes the chain the provenance: [0, 3] says which rows of
    the file survived, not which rows of some intermediate."""
    sel = _frame.full_selection(5)
    sel = _frame.narrow(sel, [True, False, True, True, False])
    assert sel.tolist() == [0, 2, 3]
    sel = _frame.narrow(sel, [True, False, True])
    assert sel.tolist() == [0, 3]


def test_a_mask_of_the_wrong_length_is_refused():
    """The one way a caller can desync a selection is a mask sized against the wrong frame.
    Refused loudly, because the alternative is numpy broadcasting it into a wrong answer."""
    with pytest.raises(ValueError, match="mask has 4 entries"):
        _frame.narrow(_frame.full_selection(5), [True, False, True, True])


def test_every_view_moves_together_because_there_is_one_selection():
    """The property the whole model exists for. `labels` cannot fall out of step with `counts`
    because neither is stored narrowed — both are the frame, read through `rows`."""
    f = _demo()
    rows = _frame.narrow(_frame.full_selection(5), [True, False, True, True, False])
    cols = _frame.narrow(_frame.full_selection(4), [True, True, False, False])
    v = _frame.view(f, rows, cols)

    assert v["counts"].shape == (3, 2)
    assert v["labels"].tolist() == ["d0", "d3", "d3"]
    assert v["genes"].tolist() == ["a", "b"]
    assert len(v["labels"]) == v["counts"].shape[0], "one selection, so this cannot fail"


def test_the_frame_is_never_written_by_taking_a_view():
    """Immutability is the contract the invalidation rule rests on: if a view could write
    back, `derived` would not be the only thing a narrowing has to clear."""
    f = _demo()
    rows = _frame.narrow(_frame.full_selection(5), [True, False, True, True, False])
    v = _frame.view(f, rows, _frame.full_selection(4))
    v["counts"][0, 0] = 999.0

    assert f["counts"][0, 0] == 0.0, "the frame is the original and stays the original"


def test_columns_narrow_even_when_the_frame_carries_no_gene_names():
    """NOT IN THE PLAN. The plan's `view` skipped its column slice unless `genes is not None`,
    which conflates "has gene names" with "has a gene axis". Measured against that version: a
    5 × 4 frame with `genes=None` under a 2-column selection returned a 5 × 4 view — full
    width, no error, the exact class of silent wrong-shape answer the frame model exists to
    remove. `loading.py` only fills `genes` when the file carries `var_names`, so the frame
    with a real gene axis and no names is a shape that reaches here."""
    f = _frame.new_frame(counts=np.arange(20, dtype=np.float32).reshape(5, 4),
                         genes=None, labels=None)
    cols = _frame.narrow(_frame.full_selection(4), [True, True, False, False])
    v = _frame.view(f, _frame.full_selection(5), cols)

    assert v["counts"].shape == (5, 2), "the selection is the truth, names or no names"
    assert v["genes"] is None


def test_an_absent_slot_views_as_absent_rather_than_raising():
    """A synthetic point cloud has no gene axis and no labels. Those runs are the majority of
    the test suite, so a view over them must be ordinary rather than exceptional."""
    f = _frame.new_frame(counts=np.zeros((3, 2)), genes=None, labels=None)
    v = _frame.view(f, _frame.full_selection(3), _frame.full_selection(2))

    assert v["genes"] is None and v["labels"] is None


# ── layers: a second matrix over the same cells and genes ────────────────────
def test_a_frame_with_no_layers_is_the_shape_it_always_was():
    """The migration is INERT until something populates it. A frame built the old way answers
    the old way, plus an empty dict that no reader has to special-case."""
    f = _frame.new_frame(counts=np.arange(12).reshape(3, 4), genes=np.array(list("abcd")))
    assert f["layers"] == {}
    v = _frame.view(f, _frame.full_selection(3), _frame.full_selection(4))
    assert v["layers"] == {}


def test_a_layer_narrows_on_both_axes_exactly_as_counts_does():
    """The whole reason layers are cheap: they are cell-by-gene, so they take the `counts` rule
    unchanged. If this ever diverges, a layer is being mis-sliced and every number a tool
    computes from it is attributed to the wrong cell."""
    counts = np.arange(12).reshape(3, 4)
    spliced = counts * 10
    f = _frame.new_frame(counts=counts, genes=np.array(list("abcd")), layers={"spliced": spliced})

    rows = _frame.narrow(_frame.full_selection(3), np.array([True, False, True]))
    cols = _frame.narrow(_frame.full_selection(4), np.array([True, True, False, False]))
    v = _frame.view(f, rows, cols)

    np.testing.assert_array_equal(v["layers"]["spliced"], v["counts"] * 10)
    assert v["layers"]["spliced"].shape == v["counts"].shape == (2, 2)


def test_the_frame_is_never_written_through_a_view():
    """`new_frame` is the immutable original. A view that aliased it would let a step mutate the
    data as loaded, and the selection chain would stop being the provenance."""
    spliced = np.arange(12).reshape(3, 4)
    f = _frame.new_frame(counts=np.zeros((3, 4)), layers={"spliced": spliced})
    rows = _frame.narrow(_frame.full_selection(3), np.array([True, False, True]))
    v = _frame.view(f, rows, _frame.full_selection(4))
    v["layers"]["spliced"][0, 0] = -999
    assert f["layers"]["spliced"][0, 0] == 0, "the view wrote through to the frame"


def test_the_layer_dict_a_caller_passes_is_not_held_by_reference():
    """A caller that keeps mutating its own dict must not be able to add a layer to a frame that
    has already been built — the frame is written once."""
    caller = {"spliced": np.zeros((2, 2))}
    f = _frame.new_frame(counts=np.zeros((2, 2)), layers=caller)
    caller["unspliced"] = np.ones((2, 2))
    assert set(f["layers"]) == {"spliced"}


def test_the_loader_supplies_only_the_layers_that_were_declared(tmp_path):
    """DECLARED-ONLY is the policy, and this is it. An `.h5ad` may carry a dozen layers; loading
    all of them makes memory a property of the FILE rather than of the analysis, and no
    threshold picks itself. So a step says what it reads and the loader supplies that."""
    ad = pytest.importorskip("anndata")
    from manyruns.pipeline import loading

    a = ad.AnnData(X=np.zeros((4, 3), dtype="float32"))
    for name in ("spliced", "unspliced", "decoy"):
        a.layers[name] = np.full((4, 3), len(name), dtype="float32")

    assert sorted(loading.layers_of(a, ("spliced", "unspliced"))) == ["spliced", "unspliced"]
    assert sorted(loading.layers_of(a, ("spliced",))) == ["spliced"]
    # Nothing asked for, nothing loaded — the inert default every recipe has today.
    assert loading.layers_of(a, None) == {}
    assert loading.layers_of(a, ()) == {}
    # A name asked for and ABSENT is simply not returned. It is not an error here because this
    # cannot know whether the step that wanted it was optional; `vocab.unmet` is where a missing
    # `splicing` becomes a refusal, at plan time, with a sentence.
    assert loading.layers_of(a, ("nope",)) == {}


def test_a_recipe_declares_the_layers_its_steps_read():
    """The union over the steps, so two tools asking for the same layer load it once."""
    from manyruns import app

    assert app.declared_layers(None) == ()
    assert app.declared_layers({"steps": [{"name": "phate"}]}) == ()
    assert app.declared_layers({"steps": [
        {"name": "a", "layers": ["spliced", "unspliced"]},
        {"name": "b", "layers": ["spliced"]},
    ]}) == ("spliced", "unspliced")
