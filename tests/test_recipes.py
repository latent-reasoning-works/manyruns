"""The branches recipe, the graph-valued Reeb step, the `branch_points` readout, the
archetypes unblock, and the recipe README's class-to-recipe map and gate table.

The engine here is a stand-in installed into `sys.modules`, as in
`tests/test_partition_step_suite.py`. `reeb_graph` refuses the metric suite, as the real step
would on its 0/1 membership matrix, and returns a fixed structural summary.
"""
from __future__ import annotations

import re
import sys
import tomllib
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from manyruns import catalog, pipeline, score, vocab  # noqa: E402
from manyruns.pipeline import runner, steps  # noqa: E402

SUMMARY = {"n_nodes": 6, "n_edges": 6, "n_components": 1,
           "degree_sequence": [3, 3, 2, 2, 1, 1], "n_branch_points": 2, "n_endpoints": 2,
           "n_loops": 1}

REPO = Path(__file__).resolve().parent.parent
README = REPO / "manyruns" / "configs" / "recipe" / "README.md"


def _engine(monkeypatch, calls: list) -> None:
    api = types.ModuleType("manylatents.api")

    def run(**kw):
        calls.append(kw)
        n = kw["input_data"].shape[0]
        algo = kw["algorithms"]["latent"]
        if algo == "pca":
            return {"embeddings": np.zeros((n, 10), np.float32), "scores": {}}
        if algo == "phate":
            return {"embeddings": np.ones((n, 3), np.float32),
                    "scores": {m: 0.5 for m in kw.get("metrics", [])}}
        if algo == "reeb_graph":
            if "metrics" in kw:
                raise AssertionError("the suite was sent to a graph-valued step")
            member = (np.arange(n * 6).reshape(n, 6) % 2).astype(np.float32)
            return {"embeddings": member, "scores": {}, "structural_summary": dict(SUMMARY)}
        raise AssertionError(f"unexpected algorithm {algo}")

    api.run = run
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)


# ── a graph-valued latent step keeps the embedding ───────────────────────────


def test_reeb_graph_after_phate_keeps_the_phate_coordinates(tmp_path, monkeypatch):
    calls: list = []
    _engine(monkeypatch, calls)
    titles: list = []
    real = runner._display_scatter

    def spy(emb, state, ctx, fname, title, **kw):
        titles.append(title)
        return real(emb, state, ctx, fname, title, **kw)

    monkeypatch.setattr(runner, "_display_scatter", spy)
    out = pipeline.run_manylatents(
        {"name": "r", "steps": [{"name": "phate", "group": "latent", "params": {}},
                                {"name": "reeb_graph", "group": "latent", "params": {}}]},
        array=np.random.default_rng(0).normal(size=(40, 10)).astype(np.float32),
        out_dir=tmp_path, metrics=["lid"])

    by_name = {s["name"]: s for s in out["steps"]}
    assert by_name["phate"]["outcome"] == "ok", by_name["phate"]["detail"]
    assert by_name["reeb_graph"]["outcome"] == "ok", by_name["reeb_graph"]["detail"]
    g = out["g_vector"]
    assert g["final_dim"] == 3
    assert "graph" in g["reeb_graph.suite_note"]
    assert g["reeb_graph.structural_summary.n_branch_points"] == 2
    reeb_call = [c for c in calls if c["algorithms"]["latent"] == "reeb_graph"][0]
    assert reeb_call["input_data"].shape == (40, 3)
    assert titles == ["phate"]
    assert "reeb_graph.membership" in by_name["reeb_graph"]["artifacts"]


def test_the_helper_follows_the_step_calculus():
    names = ("reeb_graph", "leiden", "phate", "pca")
    assert [n for n in names if runner._graph_valued(n)] == ["reeb_graph"]
    for name, produces in vocab.STEP_PRODUCES.items():
        assert runner._graph_valued(name) == ("graph" in produces), name


# ── the branch_points readout reads the Reeb summary ─────────────────────────


def test_branch_points_copies_the_summary(tmp_path):
    g = {"reeb_graph.structural_summary.n_branch_points": 2,
         "reeb_graph.structural_summary.n_endpoints": 3}
    steps._step_branch_points({}, g, {}, tmp_path, [], 3)
    assert g["branch_points.n_branch_points"] == 2
    assert g["branch_points.n_endpoints"] == 3
    assert "branch_points.n_branch_points_note" not in g


def test_branch_points_without_a_summary_writes_none_and_a_note(tmp_path):
    g: dict = {}
    steps._step_branch_points({}, g, {}, tmp_path, [], 3)
    assert g["branch_points.n_branch_points"] is None
    assert g["branch_points.n_endpoints"] is None
    note = g["branch_points.n_branch_points_note"]
    assert note.startswith("MeasurementUnavailable:")
    assert "reeb_graph.structural_summary.n_branch_points" in note


def test_branch_points_source_selects_another_step(tmp_path):
    g = {"rg2.structural_summary.n_branch_points": 1, "rg2.structural_summary.n_endpoints": 2}
    steps._step_branch_points({}, g, {"source": "rg2"}, tmp_path, [], 3)
    assert g["branch_points.n_branch_points"] == 1
    assert steps._ANALYSIS_STEPS["branch_points"] is steps._step_branch_points


def test_the_readout_is_illegal_before_a_reeb_step():
    assert vocab.STEP_NEEDS["branch_points"] == ("graph",)
    recipe = {"name": "r", "steps": [{"name": "phate", "group": "latent"},
                                     {"name": "branch_points", "group": "analysis"}]}
    assert vocab.unmet(recipe, "manifold") == frozenset({"graph"})


# ── the branches recipe claims multi-branching ───────────────────────────────


def test_the_branches_recipe_is_declared_as_specified():
    r = catalog.load_recipe("branches")
    assert catalog.check_recipe(r, "branches") == []
    assert (r["name"], r["id"], r["claims"], r["suits"]) == (
        "branches", "sc.branch.reeb", "multi-branching", ["manifold"])
    assert " — " in r["question"]
    src = r["source"]
    assert src["kind"] == "preprint" and src["url"] == "https://arxiv.org/abs/2605.12662"
    assert src["cite"] and src["verified"] and src["note"]
    assert [(s["name"], s["group"], s.get("params"), s.get("via")) for s in r["steps"]] == [
        ("normalize", "prep", {"target_sum": 10000}, "manylatents"),
        ("transform", "prep", {"method": "log1p"}, "manylatents"),
        ("pca", "latent", {"n_components": 50}, "manylatents"),
        ("phate", "latent", {"n_components": 3}, "manylatents"),
        ("reeb_graph", "latent", {}, "manylatents"),
        ("branch_points", "analysis", {}, "manyruns")]
    assert r["steps"][2]["limits"] == {"n_components": ["n_samples", "n_features"]}


def test_the_recipe_closes_rung_3_in_the_catalogue():
    assert score.claimed_rungs()["branches"] == 3
    assert score.unroutable_rungs() == []


def test_the_recipe_is_legal_on_every_shape():
    r = catalog.load_recipe("branches")
    for shape in vocab.SHAPES:
        assert vocab.unmet(r, shape) == frozenset(), shape


def test_the_recipe_runs_end_to_end_against_a_fake_engine(tmp_path, monkeypatch):
    calls: list = []
    _engine(monkeypatch, calls)
    r = catalog.load_recipe("branches")
    out = pipeline.run_manylatents(
        {**r, "steps": r["steps"][2:]},
        array=np.random.default_rng(0).normal(size=(60, 10)).astype(np.float32),
        seed=0, out_dir=tmp_path)
    assert not [s for s in out["steps"] if s["outcome"] == "error"], out["steps"]
    assert out["g_vector"]["final_dim"] == 3
    assert out["g_vector"]["branch_points.n_branch_points"] == 2


# ── the archetypes recipe is runnable ────────────────────────────────────────


def test_aa_step_names_archetype_count():
    aa = [s for s in catalog.load_recipe("archetypes")["steps"] if s["name"] == "aa"]
    assert len(aa) == 1 and aa[0]["params"] == {"n_components": 3}


def test_manylatents_bound_is_aa_fix():
    assert REPO.is_dir()
    meta = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert "manylatents>=0.1.9" in meta["project"]["dependencies"]


# ── the recipe README ────────────────────────────────────────────────────────


def _section(heading: str) -> str:
    assert README.parent.is_dir()
    text = README.read_text()
    start = text.index(f"\n{heading}\n")
    rest = text[start + len(heading) + 2:]
    nxt = re.search(r"^## ", rest, flags=re.M)
    return rest[:nxt.start()] if nxt else rest


def _table_rows(section: str) -> list[list[str]]:
    rows = [ln for ln in section.splitlines() if ln.startswith("|")]
    return [[c.strip().strip("`") for c in ln.strip("|").split("|")] for ln in rows[2:]]


def test_every_mapped_recipe_claims_its_class():
    section = _section("## Structure class to recipe")
    assert "| Structure class | Recipe |" in section
    rows = _table_rows(section)
    assert len(rows) == 4
    for cls, recipe in rows:
        assert catalog.load_recipe(recipe)["claims"] == cls, (cls, recipe)
    assert {cls for cls, _ in rows} == {"clusters", "single-trajectory", "multi-branching",
                                        "archetypal"}


def test_the_readme_table_matches_the_code():
    section = _section("## The rung-support gate")
    keys = {row[1] for row in _table_rows(section) if len(row) > 1}
    assert keys == {key for key, _ in vocab.RUNG_EVIDENCE.values()}
    for word in ("admitted", "MeasurementUnavailable", "necessary", "SUPPORT_REASONS"):
        assert word in section, word
