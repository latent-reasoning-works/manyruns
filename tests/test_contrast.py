"""The contrast recipe (case/control) + the `separation` step.

Mock run is dep-free; the actual `separation` compute uses numpy + sklearn (both public and
both base dependencies, so CI's product-layer env has them), and its silhouette + graceful
degradation are therefore exercised for real here.
"""
from __future__ import annotations

import numpy as np

from manyruns import app, pipeline


def test_contrast_recipe_is_prep_then_phate_separation_composition():
    recipe = app.load_recipe("contrast")
    assert recipe["name"] == "contrast"
    # The list gained `normalize → transform → pca(50)` at the front in the cutover (#54): that
    # preamble always ran, but inside `loading._anndata_matrix`, where it reached no trace and no
    # g-vector. It is the SAME compute, now declared, so it is pinned here in the order it runs.
    # `phate → separation → composition` — the part this test was written to protect — is
    # unchanged, and phate must still precede the two analysis steps, which score the EMBEDDING.
    assert [s["name"] for s in recipe["steps"]] == [
        "normalize", "transform", "pca", "phate", "separation", "composition"]


def test_contrast_runs_in_mock(tmp_path):
    res = app.run_explorations(
        None, "scrna", recipe=app.load_recipe("contrast"), engine="mock", out_dir=tmp_path
    )
    # Six entries, not three, for the reason above — and the two prep steps carry `prep:`, the
    # group `session.UNOFFERED_GROUPS` hides from the interactive menu but the trace still names.
    # No prep COMPUTE happens on this route and none is claimed: `engine=mock` runs
    # `serving._run_recipe`, whose `_mock_step` dispatches on `group` with branches for
    # `latent`/`lightning`/`analysis` only, so a `prep` step contributes a trace entry and
    # nothing else. Measured: dropping the two prep steps from this recipe leaves the mock
    # g-vector bit-identical ({'separation': 0.03, 'composition': 0.03, 'final_dim': 3}). The
    # counts/negative-value guards that make `normalize`/`transform` decline live under
    # `runner._run_prep_step`, which this loop never calls — so the trace is the whole claim.
    assert res["trace"] == ["prep:normalize", "prep:transform", "latent:pca",
                            "latent:phate", "analysis:separation", "analysis:composition"]
    assert "separation" in res["g_vector"] and "composition" in res["g_vector"]


# ── the real separation step (numpy + sklearn) ───────────────────────────────
_EMB = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 9.9]])


def test_separation_scores_two_clear_groups(tmp_path):
    g: dict = {}
    labels = np.array(["treated", "treated", "healthy", "healthy"])
    pipeline._step_separation({"emb": _EMB, "labels": labels, "label_kind": "condition"},
                              g, {}, tmp_path, [], 2)
    assert g["separation_n_groups"] == 2
    assert g["separation_silhouette"] > 0.5           # two well-separated conditions


def test_separation_degrades_without_labels(tmp_path):
    g: dict = {}
    pipeline._step_separation({"emb": _EMB, "labels": None}, g, {}, tmp_path, [], 2)
    assert g["separation"] is None
    assert "condition labels" in g["separation_note"]  # a note, not an error


def test_separation_degrades_with_one_group(tmp_path):
    g: dict = {}
    pipeline._step_separation(
        {"emb": _EMB, "labels": np.array(["treated"] * 4), "label_kind": "condition"},
        g, {}, tmp_path, [], 2
    )
    assert g["separation"] is None and "≥2" in g["separation_note"]


# ── the real composition step (numpy + sklearn KMeans) ───────────────────────
def test_composition_reports_a_population_shift(tmp_path):
    rng = np.random.RandomState(0)
    # condition A is mostly cluster-0 cells, B mostly cluster-1 → a real abundance shift
    a = np.vstack([rng.randn(40, 2) * 0.1, rng.randn(10, 2) * 0.1 + [10, 10]])
    b = np.vstack([rng.randn(10, 2) * 0.1, rng.randn(40, 2) * 0.1 + [10, 10]])
    emb = np.vstack([a, b])
    labels = np.array(["treated"] * 50 + ["healthy"] * 50)
    g: dict = {}
    pipeline._step_composition({"emb": emb, "labels": labels, "label_kind": "condition"},
                               g, {"k": 2}, tmp_path, [], 2)
    assert g["composition_n_clusters"] == 2
    assert g["composition_max_log2_shift"] > 1.0         # a clear enrichment difference
    assert g["composition_between"] == "healthy vs treated"


def test_composition_degrades_without_labels(tmp_path):
    g: dict = {}
    pipeline._step_composition({"emb": _EMB, "labels": None}, g, {}, tmp_path, [], 2)
    assert g["composition"] is None and "condition labels" in g["composition_note"]


def test_condition_key_detected_before_falling_through():
    class _Obs:
        columns = ["cell_type", "disease", "sample"]

    class _AD:
        obs = _Obs()

    assert pipeline._detect_condition_key(_AD()) == "disease"


# ── the label KIND, not merely the presence of labels ────────────────────────
# C1's second half. `state["labels"]` is ONE channel carrying THREE kinds (`time`,
# `condition`, `group`), and `runner.py`'s trajectory guard was the only consumer that
# checked. These two read the same channel and emit `separation_*` / `composition_*` keys — a
# case/control claim — so a `group` axis (a cell type, a cluster, a branch) scored a real
# silhouette over cell types and put it in the g-vector as a contrast. The fixture that makes
# it reachable ships in this branch: `data/tree8.h5ad` carries a `branch` column.
_GROUPED = np.array(["b1", "b1", "b2", "b2"])


def test_only_a_condition_kind_is_scored_whatever_the_other_kind_is_called(tmp_path):
    """The property is 'only condition', NOT 'condition and not group'.

    Tested against a kind this repo does not define, exactly as
    `test_condition_as_time.py::test_only_a_time_kind_reaches_mioflow_whatever_the_other_kind_
    is_called` does, and for the same reason: a denylist of one is what let `group` through in
    the first place, so a test that only names `group` would pass on the code that has the
    defect one kind later.
    """
    g: dict = {}
    state = {"emb": _EMB, "labels": _GROUPED, "label_kind": "a-kind-nobody-has-defined"}
    pipeline._step_separation(state, g, {}, tmp_path, [], 2)
    pipeline._step_composition(state, g, {"k": 2}, tmp_path, [], 2)

    assert g["separation"] is None and "condition labels" in g["separation_note"]
    assert g["composition"] is None and "condition labels" in g["composition_note"]
    # the refusal is a REFUSAL: no scored key may appear beside it
    assert "separation_silhouette" not in g and "separation_n_groups" not in g
    assert "composition_max_log2_shift" not in g and "composition_between" not in g


def test_a_group_axis_is_not_a_contrast(tmp_path):
    """The reachable case, named: `manyruns init data/tree8.h5ad` then `separation`, or
    `manyruns run --recipe contrast` on anything carrying a cell-type column. Before the
    guard this wrote a real silhouette over cell types and `composition_between: "b1 vs b2"`,
    which reads as case versus control and is not."""
    g: dict = {}
    state = {"emb": _EMB, "labels": _GROUPED, "label_kind": "group"}
    pipeline._step_separation(state, g, {}, tmp_path, [], 2)
    pipeline._step_composition(state, g, {"k": 2}, tmp_path, [], 2)

    assert g["separation"] is None and g["composition"] is None
    assert "separation_silhouette" not in g and "composition_between" not in g


def test_labels_with_no_declared_kind_are_not_scored_either(tmp_path):
    """The allowlist's other half, and the same argument `test_condition_as_time.py` makes for
    it: production never produces this — `load_labeled` returns either no labels and no kind,
    or labels WITH a kind — so an untyped array means a caller constructed one, and guessing
    that it is a condition axis is this defect with the declaration missing rather than wrong.
    """
    g: dict = {}
    labels = np.array(["treated", "treated", "healthy", "healthy"])
    pipeline._step_separation({"emb": _EMB, "labels": labels}, g, {}, tmp_path, [], 2)
    assert g["separation"] is None and "condition labels" in g["separation_note"]
