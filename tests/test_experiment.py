"""The metric-separation harness: plan → run → table → rank.

Dep-free where it can be (the mock engine needs no stack); the ranking needs scipy.
"""
from __future__ import annotations

import pytest

from manyruns import experiment as ex
from manyruns import catalog


def _proto(name, *groups):
    """A recipe is just the dict the runner consumes — no wrapper type."""
    return {"name": name,
            "steps": [{"name": f"s{i}", "group": g} for i, g in enumerate(groups)]}


def _ds(name, shape="manifold"):
    return {"name": name, "handle": {"kind": "manylatents", "ref": name},
            "modality": "scrna", "shape": shape}


# ── the catalog: one representation, validation as a function ────────────────
def test_recipes_are_discovered_from_the_config_dir_not_a_python_list():
    """The directory IS the registry — no parallel list to drift."""
    found = catalog.discover_recipes()
    assert {"cflows", "contrast", "embed"} <= set(found)
    for name in found:
        assert catalog.load_recipe(name)["name"] == name


def test_a_loaded_recipe_IS_the_dict_the_runner_consumes():
    """No wrapper, so no conversion — this is the whole point of dropping the type."""
    recipe = catalog.load_recipe("cflows")
    assert isinstance(recipe, dict)
    # was `["latent", "lightning"]`: cflows gained the declared prep preamble (#54) —
    # normalize → transform → pca(50) — which `loading._anndata_matrix` used to apply on
    # every counts-like load without appearing in the recipe, the trace or the g-vector.
    assert [s["group"] for s in recipe["steps"]] == [
        "prep", "prep", "latent", "latent", "lightning"]


def test_check_recipe_reports_a_bad_step_group():
    bad = catalog.check_recipe({"name": "x", "steps": [{"name": "phate", "group": "module"}]})
    assert any("group must be one of" in p for p in bad)   # the old vocabulary is gone
    assert catalog.check_recipe({"name": "x", "steps": []}) == ["no steps"]


def test_check_dataset_reports_a_bad_shape_and_handle():
    bad = catalog.check_dataset({"name": "d", "handle": {"kind": "s3", "ref": "b"},
                                 "shape": "trajectoryish"})
    assert any("shape must be one of" in p for p in bad)
    assert any("handle kind must be one of" in p for p in bad)
    assert bad and len(bad) == 2, "every problem is reported, not just the first"


def test_loading_is_strict_so_an_invalid_file_cannot_reach_a_run(tmp_path):
    (tmp_path / "bad.yaml").write_text(
        "name: bad\nsteps: [{name: phate, group: module}]\n")
    with pytest.raises(ValueError, match="invalid recipe"):
        catalog.load_recipe("bad", tmp_path)


def test_check_suite_defers_the_name_check_rather_than_faking_it():
    assert catalog.check_suite(["anisotropy", "lid"]) == []   # registry absent → deferred
    assert any("duplicate" in p for p in catalog.check_suite(["lid", "lid"]))
    assert any("at least one" in p for p in catalog.check_suite([]))


# ── the plan ─────────────────────────────────────────────────────────────────
def test_plan_is_the_cross_product():
    runs = ex.plan([_proto("a", "latent"), _proto("b", "latent")],
                   [_ds("x"), _ds("y"), _ds("z")], seeds=(1, 2))
    assert len(runs) == 2 * 3 * 2
    assert len({r.label for r in runs}) == len(runs)      # labels are unique


def test_a_run_pickles_for_a_worker_process():
    import pickle

    r = ex.plan([_proto("a", "latent")], [_ds("x")])[0]
    assert pickle.loads(pickle.dumps(r)) == r


# ── running ──────────────────────────────────────────────────────────────────
def test_run_all_on_the_mock_engine_produces_a_row_per_cell(tmp_path):
    # a case-control dataset is included so `contrast` has a legal cell; on the manifold it
    # is pruned, which is why this is 3 and not 4
    runs = ex.plan([catalog.load_recipe("embed"), catalog.load_recipe("contrast")],
                   [_ds("swissroll", "manifold"), _ds("cohort", "case-control")])
    results = ex.run_all(runs, engine="mock", out_dir=tmp_path)
    rows = ex.to_rows(results)

    assert len(rows) == 3
    assert all(r.ok for r in results)
    assert {r["recipe"] for r in rows} == {"embed", "contrast"}
    assert ("contrast", "swissroll") not in {(r["recipe"], r["dataset"]) for r in rows}


def test_a_failing_cell_is_recorded_not_raised(tmp_path, monkeypatch):
    """One bad (recipe, dataset) pair must not kill the sweep — nor look like data."""
    from manyruns import modes

    monkeypatch.setattr(modes, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    (res,) = ex.run_all(ex.plan([_proto("a", "latent")], [_ds("x")]), out_dir=tmp_path)

    assert res.ok is False and "nope" in res.error
    assert ex.to_rows([res])[0]["ok"] is False


def test_a_run_whose_steps_all_failed_is_not_ok(tmp_path, monkeypatch):
    from manyruns import modes

    monkeypatch.setattr(modes, "run",
                        lambda *a, **k: {"g_vector": {}, "ok": False, "status": {"s0": "error: x"}})
    (res,) = ex.run_all(ex.plan([_proto("a", "latent")], [_ds("x")]), out_dir=tmp_path)
    assert res.ok is False          # empty g-vector + ok=False, never a legitimate row


# ── the table ────────────────────────────────────────────────────────────────
def test_rows_keep_only_numeric_metrics_and_stay_ragged():
    r = ex.Result(run=ex.plan([_proto("a", "latent")], [_ds("x")])[0], ok=True,
                  g_vector={"lid": 2.5, "note": "no labels", "flag": True, "n_samples": 100})
    (row,) = ex.to_rows([r])
    assert row["lid"] == 2.5 and row["n_samples"] == 100.0
    assert "note" not in row and "flag" not in row       # strings/bools are diagnostics
    assert ex.metric_columns([row]) == ["lid", "n_samples"]


def test_write_table_round_trips(tmp_path):
    import csv

    rows = [{"recipe": "a", "dataset": "x", "shape": "manifold", "seed": 1, "ok": True, "lid": 2.0},
            {"recipe": "b", "dataset": "x", "shape": "manifold", "seed": 1, "ok": True, "lid": 9.0}]
    p = ex.write_table(rows, tmp_path / "t.csv")
    back = list(csv.DictReader(p.open()))
    assert [r["recipe"] for r in back] == ["a", "b"]
    assert [float(r["lid"]) for r in back] == [2.0, 9.0]


# ── the ranking (the actual deliverable) ─────────────────────────────────────
def _rows(**by_protocol):
    out = []
    for proto, vals in by_protocol.items():
        for i, v in enumerate(vals):
            out.append({"recipe": proto, "dataset": f"d{i}", "shape": "manifold",
                        "seed": 1, "ok": True, **v})
    return out


def test_a_separating_metric_outranks_a_noise_metric():
    pytest.importorskip("scipy")
    # `signal` is cleanly split by protocol; `noise` is identical across both
    rows = _rows(
        a=[{"signal": 1.0, "noise": 5.0}, {"signal": 1.1, "noise": 5.1}, {"signal": 0.9, "noise": 4.9}],
        b=[{"signal": 9.0, "noise": 5.0}, {"signal": 9.1, "noise": 5.1}, {"signal": 8.9, "noise": 4.9}],
    )
    ranked = ex.rank_metrics(rows, "recipe")
    assert ranked[0]["metric"] == "signal"
    assert ranked[0]["f"] > ranked[1]["f"]


def test_a_metric_only_some_protocols_emit_reports_partial_coverage():
    pytest.importorskip("scipy")
    rows = _rows(
        a=[{"shared": 1.0, "only_a": 3.0}, {"shared": 1.1, "only_a": 3.1}],
        b=[{"shared": 5.0}, {"shared": 5.1}],
    )
    by = {d["metric"]: d for d in ex.rank_metrics(rows, "recipe")}
    assert by["shared"]["coverage"] == 1.0
    assert by["only_a"]["coverage"] == 0.5
    assert by["only_a"]["f"] is None        # one group → not scorable, reported not dropped


def test_a_constant_metric_is_not_scored():
    pytest.importorskip("scipy")
    rows = _rows(a=[{"flat": 1.0}, {"flat": 1.0}], b=[{"flat": 1.0}, {"flat": 1.0}])
    assert ex.rank_metrics(rows, "recipe")[0]["f"] is None


def test_ranking_against_shape_answers_a_different_question_than_protocol():
    """Both labels come off ONE table — that is why the harness collects both."""
    pytest.importorskip("scipy")
    rows = [
        {"recipe": "p", "dataset": "a", "shape": "manifold", "seed": 1, "ok": True, "m": 1.0},
        {"recipe": "q", "dataset": "a", "shape": "manifold", "seed": 1, "ok": True, "m": 1.1},
        {"recipe": "p", "dataset": "b", "shape": "clusters", "seed": 1, "ok": True, "m": 9.0},
        {"recipe": "q", "dataset": "b", "shape": "clusters", "seed": 1, "ok": True, "m": 9.1},
    ]
    by_shape = ex.rank_metrics(rows, "shape")[0]
    by_proto = ex.rank_metrics(rows, "recipe")[0]
    assert by_shape["f"] > by_proto["f"]     # `m` tracks the SHAPE, not which protocol ran


def test_failed_rows_are_excluded_from_the_ranking():
    pytest.importorskip("scipy")
    rows = _rows(a=[{"m": 1.0}, {"m": 1.1}], b=[{"m": 9.0}, {"m": 9.1}])
    rows.append({"recipe": "a", "dataset": "z", "shape": "manifold", "seed": 1,
                 "ok": False, "m": 999.0})
    assert ex.rank_metrics(rows, "recipe")[0]["n"] == 4     # the failed row is not counted




# ── the dataset registry (Brian's surface — symmetric with Zach's) ───────────
def test_datasets_are_discovered_from_the_config_dir():
    found = catalog.discover_datasets()
    assert {"swissroll", "gaussian_blob", "torus"} <= set(found)
    for name in found:
        assert catalog.load_dataset(name)["name"] == name


def test_a_missing_file_names_the_known_ones(tmp_path):
    with pytest.raises(ValueError, match="no dataset named"):
        catalog.load_dataset("nope", tmp_path)


def test_an_external_dir_overrides_the_bundled_one(tmp_path, monkeypatch):
    """How a collaborator points at their own datasets — and how a sweep's experimental
    ARMS stay out of the shipped product."""
    (tmp_path / "mine.yaml").write_text(
        "name: mine\nhandle: {kind: path, ref: /data/x.h5ad}\nshape: case-control\n")
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(tmp_path))
    assert catalog.discover_datasets() == ["mine"]
    assert catalog.load_dataset("mine")["shape"] == "case-control"


def test_the_bundled_set_plans_a_real_cross_product():
    recipes, datasets = catalog.load_recipes(), catalog.load_datasets()
    raw = ex.plan(recipes, datasets, prune=False)
    runs = ex.plan(recipes, datasets)

    assert len(raw) == len(recipes) * len(datasets)
    assert {r.dataset["shape"] for r in runs} >= {"manifold", "clusters"}
    # every pruned cell is accounted for, so a filtered sweep is never mistaken for a
    # smaller one
    assert len(raw) - len(runs) == len(ex.skipped_cells(recipes, datasets))


def test_the_bundled_set_cannot_yet_exercise_contrast():
    """KNOWN GAP, pinned so it is not forgotten: no bundled dataset is case-control, so
    `contrast` has no legal cell. Brian's first real task is a case-control dataset.

    The gene-axis gap is the same shape and is now HALF closed, which is why this test names
    four recipes and not one: `vocab.STEP_NEEDS` gives the `qc` step `("genes",)`, the
    `rank_genes` step `("clusters", "genes")` and `filter_genes`/`filter_mito` `("genes",)`, so
    the `qc`, `markers` and `preprocess` recipes need a gene axis. `genes` is a DATASET fact
    (`vocab.DATASET_FACTS`) that `SHAPE_PROVIDES` never supplies, and
    `configs/dataset/pbmc3k.yaml` is the first and only bundled entry declaring
    `provides: [genes]` — so those three are legal on exactly one dataset and pruned on the 13
    synthetic point clouds. Asserting the WHOLE pruning ledger (not just contrast's row) is
    what keeps a third gap from opening unnoticed.

    Measured 2026-08-16: 14 datasets x 11 recipes = 154 cells, 53 of them pruned —
    contrast 14, qc 13, markers 13, preprocess 13.
    """
    recipes, datasets = catalog.load_recipes(), catalog.load_datasets()
    names = {d["name"] for d in datasets}
    skipped = ex.skipped_cells(recipes, datasets)

    pruned: dict[str, dict[str, list[str]]] = {}
    for recipe, dataset, missing in skipped:
        pruned.setdefault(recipe, {})[dataset] = missing

    # was `{"contrast"}` — `qc`/`markers` joined the ledger when they declared `("genes",)`,
    # and `preprocess` was born in it for the same reason
    assert set(pruned) == {"contrast", "qc", "markers", "preprocess"}

    # contrast is still pruned on EVERY bundled dataset; pbmc3k is the 14th and is
    # `clusters`/`design: single`, so it does not close this gap either
    assert pruned["contrast"] == {name: ["conditions"] for name in names}
    assert "case-control" not in {d["shape"] for d in datasets}

    # ...and the three gene-axis recipes are pruned on 13 of 14 — everything but pbmc3k
    for recipe in ("qc", "markers", "preprocess"):
        assert pruned[recipe] == {name: ["genes"] for name in names - {"pbmc3k"}}


# ── the precondition calculus (the legal-move function) ─────────────────────
def test_a_contrast_recipe_is_illegal_on_data_with_no_conditions():
    """It would run, emit None + a note, and land in the table as a MISSING metric —
    indistinguishable from a recipe that simply doesn't emit it."""
    from manyruns.vocab import unmet

    contrast = catalog.load_recipe("contrast")
    assert unmet(contrast, "manifold") == {"conditions"}
    assert unmet(contrast, "clusters") == {"conditions"}
    assert unmet(contrast, "case-control") == frozenset()   # legal


def test_recipes_that_fall_back_are_not_over_pruned():
    """mioflow computes a diffusion pseudotime when there is no time axis, and the
    orders by row index — so cflows is legal everywhere. the learner's own PRECONDITIONS
    require a `time` coord for mioflow; copying that table would have dropped these cells."""
    from manyruns.vocab import unmet

    cflows = catalog.load_recipe("cflows")
    for shape in ("manifold", "clusters", "single", "time-course"):
        assert unmet(cflows, shape) == frozenset(), f"cflows wrongly pruned on {shape}"
    assert unmet(catalog.load_recipe("embed"), "unknown") == frozenset()


def test_plan_prunes_the_illegal_cells():
    recipes = [catalog.load_recipe("contrast"), catalog.load_recipe("embed")]
    datasets = [_ds("m", "manifold"), _ds("cc", "case-control")]

    pruned = ex.plan(recipes, datasets)
    raw = ex.plan(recipes, datasets, prune=False)

    assert len(raw) == 4 and len(pruned) == 3          # contrast×manifold dropped
    assert ("contrast", "m") not in {(r.recipe["name"], r.dataset["name"]) for r in pruned}
    assert ("contrast", "cc") in {(r.recipe["name"], r.dataset["name"]) for r in pruned}


def test_pruning_is_never_silent():
    """A pruned sweep that doesn't say what it pruned looks like a smaller experiment."""
    recipes = [catalog.load_recipe("contrast")]
    datasets = [_ds("m", "manifold"), _ds("cc", "case-control")]

    skipped = ex.skipped_cells(recipes, datasets)
    assert skipped == [("contrast", "m", ["conditions"])]


def test_legality_is_computable_without_running_anything():
    """The property a search needs: the frontier must be enumerable without execution."""
    from manyruns.vocab import unmet

    for r in catalog.load_recipes():
        for d in catalog.load_datasets():
            unmet(r, d["shape"])          # no engine, no data, no compute
