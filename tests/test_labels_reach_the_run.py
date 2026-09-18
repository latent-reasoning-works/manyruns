"""Labels on the stackless engine — what makes `contrast` capable of scoring at all.

`run_inproc` took no labels, so `separation` and `composition` hit their "no condition
labels" branch on *every* in-process run and returned None with a note. The results table
reads a None as a MISSING metric — indistinguishable from "this recipe doesn't emit that
metric" — so the default stackless engine could never produce the case/control readouts the
product ships a whole recipe for.

The trap avoided here is as important as the fix: `load_labeled` would have supplied labels,
but it also returns a *transformed* matrix (normalize → log1p → PCA) where the in-process loop has
always run on what `load_array` returns. Taking labels from there would have silently changed
the numbers every existing real run produces. Labels are therefore read off the same object
the matrix came from, which is also what keeps the rows aligned.

CI has numpy and scikit-learn but no anndata, so the tests that matter most here are written
against plain arrays and run everywhere; the one that needs anndata says so and skips.
"""
from __future__ import annotations

import importlib.util

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

#: The module header is right that most of this runs anywhere — the label plumbing is asserted
#: against plain arrays. Two tests are not like the others: they assert on what a REAL fit did
#: (which kwargs reached PHATE, and whether one seed gives one answer), so they need the
#: embedder rather than merely numpy. Declared per-test so the other seven keep running in CI.
_needs_embedder = pytest.mark.skipif(importlib.util.find_spec("phate") is None,
                                     reason="asserts on a real PHATE fit")

CONTRAST = {
    "name": "contrast",
    "steps": [
        {"name": "phate", "group": "latent", "params": {"n_components": 2}},
        {"name": "separation", "group": "analysis"},
    ],
}


def _separable():
    """Two genuinely distinct condition groups, so silhouette has something to find."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (40, 6)), rng.normal(6, 1, (40, 6))])
    return X, np.array(["treated"] * 40 + ["healthy"] * 40)


def _stub_phate(monkeypatch):
    """Keep the embedding trivial: this is about the label channel, not about PHATE."""
    from manyruns import pipeline

    def step(state, g, params, out_dir, plots, target_dim):
        state["emb"] = np.asarray(state["X"], dtype=float)[:, :2]

    monkeypatch.setitem(pipeline._INPROC_STEPS, "phate", step)


def test_without_labels_contrast_cannot_score_which_is_the_defect(tmp_path, monkeypatch):
    """The behaviour being fixed, pinned so it cannot come back silently."""
    from manyruns import pipeline

    _stub_phate(monkeypatch)
    X, _ = _separable()
    g = pipeline.run_inproc(X, CONTRAST, tmp_path)["g_vector"]

    assert g["separation"] is None
    assert "no condition labels" in g["separation_note"]


def test_with_labels_contrast_scores_on_the_stackless_engine(tmp_path, monkeypatch):
    """THE fix. Same recipe, same data, labels threaded — a real number instead of a note."""
    from manyruns import pipeline

    _stub_phate(monkeypatch)
    X, labels = _separable()
    out = pipeline.run_inproc(X, CONTRAST, tmp_path, labels=labels, label_kind="condition")

    assert out["g_vector"]["separation_n_groups"] == 2
    assert out["g_vector"]["separation_silhouette"] > 0.5
    assert "separation_note" not in out["g_vector"]
    assert out["ok"] is True


def test_the_label_kind_travels_with_the_labels(tmp_path, monkeypatch):
    """A condition axis is not a time axis. The kind has to reach the state, or a trajectory
    step cannot refuse a condition and will invent a progression out of case/control."""
    from manyruns import pipeline

    _stub_phate(monkeypatch)
    seen = {}

    def spy(state, g, params, out_dir, plots, target_dim):
        seen["kind"] = state.get("label_kind")

    monkeypatch.setitem(pipeline._ANALYSIS_STEPS, "spy", spy)
    X, labels = _separable()
    recipe = {"name": "r", "steps": [
        {"name": "phate", "group": "latent"},
        {"name": "spy", "group": "analysis"},
    ]}
    pipeline.run_inproc(X, recipe, tmp_path, labels=labels, label_kind="condition")

    assert seen["kind"] == "condition"


def test_misaligned_labels_raise_rather_than_mislabel_cells(tmp_path, monkeypatch):
    """Matrix and labels are now fetched separately, which is exactly the seam `aligned`
    exists to guard: off by one row and cells are attributed to the wrong donor with
    nothing downstream noticing."""
    from manyruns.aligned import AlignmentError
    from manyruns import pipeline

    _stub_phate(monkeypatch)
    X, labels = _separable()
    with pytest.raises(AlignmentError, match="run_inproc"):
        pipeline.run_inproc(X, CONTRAST, tmp_path, labels=labels[:-1])


# DELETED WITH THE ENGINE: `test_the_real_engine_matrix_path_is_unchanged`.
# It pinned `_load_inputs`'s in-process branch (since removed) — matrix from `load_array`, never
# `load_labeled`, so a real run's numbers could not be re-preprocessed underneath it. That
# branch is gone (`app._read_inputs`), and with it the only production caller of
# `loading.gene_axis`. This is the one loader behaviour the removal drops; it is recorded
# here rather than left as a permanently-skipped test that reads as "temporarily broken".


def test_labels_of_reads_a_condition_column(tmp_path):
    """The detection itself, on a real AnnData. Skips where anndata is absent (CI)."""
    anndata = pytest.importorskip("anndata")
    pd = pytest.importorskip("pandas")
    from manyruns import pipeline

    ad = anndata.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=pd.DataFrame({"disease": ["treated", "treated", "healthy", "healthy"]}),
    )
    labels, kind = pipeline.labels_of(ad)

    assert kind == "condition"
    assert sorted(set(labels)) == ["healthy", "treated"]
    assert pipeline.labels_of(object()) == (None, None)     # not an AnnData → no guessing


# ── issue #29: params and the seed reach the stackless engine ────────────────
@_needs_embedder
def test_recipe_params_reach_the_fit(tmp_path, monkeypatch):
    """`_step_phate` read ONLY `n_components`, so every other knob a recipe declared was
    silently dropped on the default engine. The panel whose whole purpose is "change a
    setting, see what moves" could not show a changed setting."""
    import phate as _phate

    seen = {}
    real_init = _phate.PHATE.__init__

    def spy(self, *a, **kw):
        seen.update(kw)
        return real_init(self, *a, **kw)

    monkeypatch.setattr(_phate.PHATE, "__init__", spy)
    from manyruns import pipeline

    recipe = {"name": "r", "steps": [{"name": "phate", "group": "latent",
                                      "params": {"n_components": 2, "knn": 7, "decay": 15}}]}
    pipeline.run_inproc(np.random.default_rng(0).normal(size=(60, 5)), recipe, tmp_path, seed=42)

    assert seen["knn"] == 7
    assert seen["decay"] == 15
    assert seen["random_state"] == 42        # and the run is seeded


@_needs_embedder
def test_the_same_seed_gives_the_same_embedding(tmp_path):
    """The default engine was unreproducible by construction. Two runs, one seed, one answer."""
    from manyruns import pipeline

    X = np.random.default_rng(0).normal(size=(60, 5))
    recipe = {"name": "r", "steps": [{"name": "phate", "group": "latent",
                                      "params": {"n_components": 2, "knn": 5}}]}
    a = pipeline.run_inproc(X, recipe, tmp_path, seed=7)
    b = pipeline.run_inproc(X, recipe, tmp_path, seed=7)

    assert a["seed"] == 7                                  # and it is recorded
    assert a["g_vector"]["phate.knn"] == 5                 # the EFFECTIVE setting, not just declared
    assert np.allclose(a["g_vector"]["phate_dims"], b["g_vector"]["phate_dims"])
