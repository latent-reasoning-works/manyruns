"""The ONE g-vector schema — both engines, same keys, same meanings.

Before the runner collapse there were two step loops, and they diverged in ways that are
invisible until you build a table out of the results:

  * different core key sets per engine;
  * `n_samples` meaning INPUT rows on one engine and EMBEDDING rows on the other — which
    genuinely differ, because MIOFlow subsamples (sample_size=256);
  * a step that failed still returned a result indistinguishable from a successful one.

Any experiment that compares g-vectors across engines is built on this, so it gets pinned.
"""
from __future__ import annotations

import sys
import types

import pytest

np = pytest.importorskip("numpy")


def _fake_manylatents(monkeypatch, *, n_out=40, dims=2):
    """Stub `manylatents.api.run` → a fixed-size embedding, so we control the row count."""

    def fake_run(**kw):
        return {"embeddings": np.zeros((n_out, dims)), "scores": {"trustworthiness": 0.9}}

    api = types.ModuleType("manylatents.api")
    api.run = fake_run
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)


def _fake_real_phate(monkeypatch, *, dims=2):
    """Stub the in-process loop's phate so it needs no scientific stack."""
    from manyruns import pipeline

    def step(state, g, params, out_dir, plots, target_dim):
        state["emb"] = np.zeros((state["X"].shape[0], dims))

    monkeypatch.setitem(pipeline._INPROC_STEPS, "phate", step)


EMBED = {"name": "embed", "steps": [{"kind": "module", "name": "phate", "group": "latent"}]}


def test_the_mock_engine_conforms_to_the_schema_too(tmp_path):
    """THREE engines, not two. The mock is what every stackless run and every CI test uses,
    so a table built from it must carry the same keys — this was missed when the two real
    runners were unified, and only surfaced when the harness was actually driven."""
    from pathlib import Path

    from manyruns import modes
    from manyruns.catalog import load_recipe

    res = modes.run("infer", {"data": Path(tmp_path), "recipe": load_recipe("embed"),
                              "engine": "mock", "out_dir": str(tmp_path)})
    g = res["g_vector"]
    assert "final_dim" in g, "the mock kept final_dim out of the g-vector"
    # it has no rows (a 1-D stand-in vector), so these are OMITTED, never invented
    assert "n_samples" not in g and "n_embedded" not in g


# ── the core schema agrees across engines ────────────────────────────────────
def test_both_engines_emit_the_same_core_keys(tmp_path, monkeypatch):
    from manyruns import pipeline

    X = np.zeros((100, 5))
    _fake_real_phate(monkeypatch)
    real = pipeline.run_inproc(X, EMBED, tmp_path)

    _fake_manylatents(monkeypatch, n_out=100)
    ml = pipeline.run_manylatents(EMBED, array=X, out_dir=tmp_path)

    core = set(pipeline.GVECTOR_CORE)
    assert core <= set(real["g_vector"]), f"real is missing {core - set(real['g_vector'])}"
    assert core <= set(ml["g_vector"]), f"manylatents is missing {core - set(ml['g_vector'])}"
    # and they agree on the values, because they mean the same thing now
    for k in pipeline.GVECTOR_CORE:
        assert real["g_vector"][k] == ml["g_vector"][k], f"{k} diverges across engines"


def test_n_samples_is_input_rows_and_n_embedded_is_embedding_rows(tmp_path, monkeypatch):
    """The exact collision: a subsampling step makes these differ. They must not be one key."""
    from manyruns import pipeline

    X = np.zeros((100, 5))
    _fake_manylatents(monkeypatch, n_out=40)  # the engine subsampled 100 → 40, as MIOFlow does
    out = pipeline.run_manylatents(EMBED, array=X, out_dir=tmp_path)

    assert out["g_vector"]["n_samples"] == 100    # input rows
    assert out["g_vector"]["n_embedded"] == 40    # embedding rows
    assert out["g_vector"]["n_features"] == 5


def test_named_dataset_omits_input_shape_rather_than_lying(tmp_path, monkeypatch):
    """With a named engine dataset we never see the input — absent beats a key meaning two things."""
    from manyruns import pipeline

    _fake_manylatents(monkeypatch, n_out=40)
    g = pipeline.run_manylatents(EMBED, data_ref="swissroll", out_dir=tmp_path)["g_vector"]

    assert "n_samples" not in g and "n_features" not in g
    assert g["n_embedded"] == 40 and g["final_dim"] == 2


# ── failures are visible, not silently empty ─────────────────────────────────
def test_a_failed_step_is_not_reported_as_a_clean_run(tmp_path, monkeypatch):
    from manyruns import pipeline

    def boom(state, g, params, out_dir, plots, target_dim):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(pipeline._INPROC_STEPS, "phate", boom)
    out = pipeline.run_inproc(np.zeros((10, 3)), EMBED, tmp_path)

    assert out["ok"] is False                       # the harness must treat this as a failed cell
    assert "kaboom" in out["status"]["phate"]


def test_an_unknown_step_is_recorded_not_silently_dropped(tmp_path, monkeypatch):
    from manyruns import pipeline

    recipe = {"name": "x", "steps": [{"kind": "module", "name": "nosuchalgo", "group": "latent"}]}
    out = pipeline.run_inproc(np.zeros((10, 3)), recipe, tmp_path)

    assert out["ok"] is False
    assert out["status"]["nosuchalgo"].startswith("skipped")


def test_a_clean_run_reports_ok(tmp_path, monkeypatch):
    from manyruns import pipeline

    _fake_real_phate(monkeypatch)
    assert pipeline.run_inproc(np.zeros((10, 3)), EMBED, tmp_path)["ok"] is True


# ── chaining survives a step that fails *after* producing an embedding ───────
def test_chaining_uses_the_embedding_even_if_that_step_errored_late(tmp_path, monkeypatch):
    """The `first`-flag bug: a step raising after setting `emb` sent the NEXT step back to
    the raw input, silently running it on unreduced data."""
    from manyruns import pipeline

    seen: list = []

    def fake_run(**kw):
        seen.append(kw.get("input_data"))
        return {"embeddings": np.zeros((10, 2)), "scores": {}}

    api = types.ModuleType("manylatents.api")
    api.run = fake_run
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)
    # the first step's *plotting* blows up — after the embedding is already in state
    monkeypatch.setattr(pipeline.io, "_save_scatter",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("plot failed")))

    two = {"name": "two", "steps": [
        {"kind": "module", "name": "phate", "group": "latent"},
        {"kind": "module", "name": "pca", "group": "latent"},
    ]}
    out = pipeline.run_manylatents(two, array=np.zeros((100, 5)), out_dir=tmp_path)

    assert out["status"]["phate"].startswith("error")     # step 1 did fail…
    assert seen[1] is not None and seen[1].shape == (10, 2)  # …but step 2 chained the embedding
    assert seen[1].shape != (100, 5), "step 2 fell back to the raw input"


def test_n_samples_is_the_count_that_entered_even_when_a_filter_removed_cells(tmp_path,
                                                                              monkeypatch):
    """`GVECTOR_CORE` defines `n_samples` as "rows of the INPUT data", and `_finalize` derived
    it from `state["X"]` — which a `prep` filter now narrows. So a run that dropped 218 of
    2700 cells reported `n_samples: 2482, n_embedded: 2482` and the number 2700 appeared
    nowhere in its g-vector.

    That silences the one surface built to show cells leaving a run: `narrate` prints `⇢`
    instead of `→` exactly when `n_samples != n_embedded` (narrate.py:541-544), so the
    filtered run printed `cells 2,482 → 2,482` and named 2,482 as the count that entered.
    Measured on `data/pbmc3k_raw.h5ad` with `[filter_cells(min_genes=500), phate, mioflow]`.
    The sizes are therefore seeded at run OPEN, before the loop can narrow anything."""
    from manyruns import pipeline
    from manyruns.pipeline import prep as _prep

    _fake_real_phate(monkeypatch)
    monkeypatch.setitem(_prep._PREP_STEPS, "cut",
                        lambda view, X, params: {"mask": np.arange(100) < 60, "axis": "rows"})
    recipe = {"name": "r", "steps": [
        {"name": "cut", "group": "prep", "params": {}},
        {"name": "phate", "group": "latent", "params": {}}]}

    g = pipeline.run_inproc(np.zeros((100, 5)), recipe, tmp_path)["g_vector"]

    assert g["n_samples"] == 100, "the count that ENTERED the run"
    assert g["n_features"] == 5
    assert g["n_embedded"] == 60, "and the count that came out of the embedding"
