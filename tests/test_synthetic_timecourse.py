"""`synthetic:time-course` — the first bundled dataset that can legally drive MIOFlow.

The gap these pin, measured over `catalog.load_datasets()` before this dataset existed: all
12 bundled datasets declared `shape: manifold` or `shape: clusters`, `vocab.SHAPE_PROVIDES`
maps `time-course → ("time",)` and nothing else provides it, so no bundled data carried a
time axis. Every MIOFlow run therefore reached the pseudotime fallback in
`steps._step_mioflow`, which derives an ordering FROM the embedding — and so reports a full
[0, 1] range on data with no time in it whatsoever.

What must not regress, in order of how quietly it would break:

  1. the timepoint stays an INPUT to the generator. A label read back out of the geometry is
     pseudotime wearing a time-course label, which is the failure the whole dataset exists to
     be a control for (`test_the_timepoint_is_an_input_...`);
  2. those labels reach a caller as `label_kind="time"`. The tuple form of this generator
     would have been dropped silently by `load_labeled`'s non-AnnData branch
     (`test_the_loader_delivers_a_real_time_axis`);
  3. the shared obs vocabulary — not a second private rule — is what recognises the column
     (`test_vocab_recognises_the_column_the_loader_writes`).

Nothing here imports a learner, and the two properties that matter most (1 and the numeric
coordinate) need only numpy, so they run in a bare environment; the AnnData-shaped
assertions say so and skip, matching `test_real_labels.py`'s convention for CI without
anndata.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REF = "synthetic:time-course"
DATASET = "synthetic_timecourse"


# ── 1. time is a generative parameter, not something inferred ────────────────
def test_the_timepoint_is_an_input_not_read_back_out_of_the_geometry():
    """The labels are the `t`s that were passed in, and they translated the cloud.

    Two halves, because either alone can be faked. The labels must be exactly the declared
    timepoints in equal blocks (a geometry-derived ordering would give ~n distinct values
    varying with X, not 3 exact ones); and the per-timepoint centroid must have moved by
    `drift * t`, which is what makes `t` generative rather than decorative — a generator
    that ignored `t` and only stamped the label would pass the first half alone.
    """
    from manyruns.pipeline import loading

    X, t = loading.time_course_arrays()
    n_per, dim = loading.TIME_COURSE_N_PER_TIME, loading.TIME_COURSE_DIM
    tps = loading.TIME_COURSE_TIMEPOINTS

    assert X.shape == (n_per * len(tps), dim)
    assert t.shape == (n_per * len(tps),)
    # exactly the declared timepoints, in equal blocks — not a continuum read off the data
    assert np.array_equal(t, np.repeat(np.asarray(tps, dtype=float), n_per))
    assert sorted(set(t.tolist())) == sorted(tps)

    # `t` actually moved the cloud — measured against the data's OWN spread, never against
    # the constant that generated it. The first version asserted
    # `shift ≈ TIME_COURSE_DRIFT * tp`, which is the generator compared with itself: setting
    # `TIME_COURSE_DRIFT = 0.0` turned the dataset into pure noise with a decorative time
    # label and the whole suite still passed, 437 of 437. A dataset whose entire claim is
    # "time is generative here" was pinned by nothing.
    #
    # The separation criterion below has no such escape: it is between-timepoint displacement
    # over within-timepoint noise, so it fails for any drift small enough to be undetectable,
    # whatever constant produced it.
    within = X[t == tps[0]].std(axis=0).mean()
    base = X[t == tps[0]].mean(axis=0)
    for tp in tps[1:]:
        shift = np.linalg.norm(X[t == tp].mean(axis=0) - base)
        assert shift > 3 * within, (
            f"timepoint {tp} is not separated from t=0: displacement {shift:.3f} is within "
            f"3x the within-timepoint noise {within:.3f} — `t` is not generative here"
        )


def test_the_generator_is_reproducible():
    """A fixed seed, so a g-vector measured on this dataset is comparable across runs."""
    from manyruns.pipeline import loading

    a, ta = loading.time_course_arrays()
    b, tb = loading.time_course_arrays()

    assert np.array_equal(a, b) and np.array_equal(ta, tb)


def test_the_time_labels_survive_as_the_coordinate_mioflow_integrates():
    """`vocab.numeric_time` is what turns the labels into MIOFlow's clock; it must not
    flatten them. Evenly spaced here by construction, so ranks and magnitudes agree — the
    assertion is that the values arrive as [0, ½, 1] over the right rows, not that spacing
    is preserved (that is `test_time_coord.py`'s job, on uneven day0/day3/day9 labels)."""
    from manyruns import vocab
    from manyruns.pipeline import loading

    _, t = loading.time_course_arrays()
    coord = vocab.numeric_time(t)

    assert coord.shape == t.shape
    assert sorted(set(np.round(coord, 6).tolist())) == [0.0, 0.5, 1.0]
    assert np.allclose(coord, t)   # labels already live in [0, 1]; nothing is rescaled away


# ── 2. the labels reach a caller as a TIME axis ──────────────────────────────
def test_the_loader_delivers_a_real_time_axis():
    """`load_labeled(synthetic:time-course)` → (matrix, labels, "time"), rows aligned.

    `kind` is the load-bearing assertion. Had the generator returned an `(X, labels)` tuple,
    `load_labeled`'s non-AnnData branch returns `(as_matrix(obj), None, None)` — the time
    axis would vanish without an error and MIOFlow would fall back to pseudotime on a
    dataset that declares `shape: time-course`."""
    pytest.importorskip("anndata")
    from manyruns.pipeline import loading

    matrix, labels, kind = loading.load_labeled(Path(REF))

    assert kind == "time"
    assert labels is not None and len(labels) == matrix.shape[0]
    assert sorted({float(v) for v in labels}) == list(loading.TIME_COURSE_TIMEPOINTS)


def test_the_matrix_reaches_the_recipe_ungroomed():
    """`load_labeled` applies the scRNA preamble (normalize → log1p → PCA) to count-like
    matrices. This one is Gaussian and must come through untouched, or the drift the
    timepoints encode is no longer the drift the recipe sees."""
    pytest.importorskip("anndata")
    from manyruns.pipeline import loading

    generated, _ = loading.time_course_arrays()
    matrix, _, _ = loading.load_labeled(Path(REF))

    assert np.allclose(np.asarray(matrix), generated)


def test_vocab_recognises_the_column_the_loader_writes():
    """The SHARED obs vocabulary decides this, not a rule private to the generator — the
    forked-time-words mistake `vocab`'s docstring records is exactly this shape."""
    pytest.importorskip("anndata")
    from manyruns import vocab
    from manyruns.pipeline import loading

    ad = loading.time_course_anndata()

    assert vocab.find_time_key(ad.obs.columns) == "timepoint"
    assert vocab.has_time_axis(ad.obs.columns)
    assert vocab.find_condition_key(ad.obs.columns) is None   # a time axis, not a contrast
    labels, kind = loading.labels_of(ad)
    assert kind == "time"
    assert np.allclose(np.asarray(labels, dtype=float), loading.time_course_arrays()[1])


# ── 3. the config, and what the precondition calculus makes of it ────────────
def test_the_config_declares_a_usable_time_course():
    from manyruns import catalog, vocab

    cfg = catalog.load_dataset(DATASET)

    assert catalog.check_dataset(cfg, DATASET) == []
    assert cfg["shape"] == "time-course"
    assert cfg["handle"] == {"kind": "path", "ref": REF}
    assert vocab.dataset_provides(cfg["shape"]) == frozenset({"time"})
    assert set(cfg["topology"]) <= set(vocab.TOPOLOGIES)


def test_the_bundled_set_now_carries_a_time_axis():
    """The gap itself, pinned: delete this dataset and no bundled data provides `time`."""
    from manyruns import catalog, vocab

    provided = set()
    for d in catalog.load_datasets():
        provided |= vocab.dataset_provides(d.get("shape", "unknown"))

    assert "time" in provided


def test_cflows_is_legal_here_and_contrast_still_is_not():
    """`unmet` is the legal-move function, and the point of this dataset is that `cflows`
    has nothing unmet on it. `contrast` is asserted loosely — only that `conditions` is
    still missing — because this dataset must not be mistaken for the case-control one that
    is still absent, and the rest of that recipe's needs are not this test's business."""
    from manyruns import catalog, vocab

    assert vocab.unmet(catalog.load_recipe("cflows"), "time-course") == frozenset()
    assert "conditions" in vocab.unmet(catalog.load_recipe("contrast"), "time-course")


# ── 4. the generated-ref seam does not eat real paths, and says so on a typo ──
def test_an_unknown_generated_ref_refuses_by_name():
    """The reason the prefix is checked before `p.exists()`: `synthetic:typo` must report
    the refs that exist, not "data path not found", which sends you looking on disk for
    something that was never going to be there."""
    from manyruns.pipeline import loading

    with pytest.raises(ValueError) as err:
        loading.load_array(Path("synthetic:tiem-course"))

    msg = str(err.value)
    assert "unknown generated dataset" in msg
    assert REF in msg
    assert "not found" not in msg


def test_a_real_path_is_still_read_from_disk(tmp_path):
    """Guard on the early return added to `load_array`: a file whose name merely contains
    the word, or any ordinary path, must still be loaded, and a missing one must still
    report a missing path."""
    from manyruns.pipeline import loading

    f = tmp_path / "synthetic_notes.npy"
    np.save(f, np.asarray([[1.0, 2.0]]))
    assert loading.load_array(f).tolist() == [[1.0, 2.0]]

    with pytest.raises(ValueError, match="data path not found"):
        loading.load_array(tmp_path / "absent.npy")
