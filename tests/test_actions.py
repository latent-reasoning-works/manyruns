"""`manyruns actions`: the legal action space for one dataset, as one JSON document.

The verb is read-only. The ledger surface is `tui.state.ledger` serialized, never a second
account of it; the tune surface is the declared params and their one-press neighbours; the ladder
surface is the scorer's tables. These tests pin that the document equals what those functions
say, that it depends on no data file, and that it is the same document in process and on stdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from manyruns import actions, app, catalog, narrate, score, store, tune, vocab
from manyruns.tui import params, state

ROW_A = {"schema_version": 1, "run_id": "aaaaaaaaaaaa", "dataset_name": "swissroll", "seed": 42,
         "recipe": "embed", "ok": True, "complete": True,
         "g_vector": {"betti_0": 1, "lid": 2.1}}


def _digest(doc: dict) -> str:
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write_store(root: Path, rows: list[dict]) -> None:
    (root / "outputs").mkdir()
    (root / "outputs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _toy(root: Path) -> Path:
    ddir = root / "datasets"
    ddir.mkdir()
    (ddir / "toy.yaml").write_text(
        "name: toy\nshape: clusters\nmodality: scrna\ntopology: [clusters]\n"
        f"handle: {{kind: path, ref: {root / 'x.h5ad'}}}\n")
    return ddir


def _tunable_recipes() -> list[str]:
    assert catalog.recipe_dir().is_dir()
    return [r for r in catalog.discover_recipes()
            if any(params.tunable_for(s) for s in catalog.load_recipe(r)["steps"])]


def _without(doc: dict, *keys: str) -> dict:
    return {k: v for k, v in doc.items() if k not in keys}


# ── task 2: the module ──────────────────────────────────────────────────────────


def test_header_keys_come_first(tmp_path):
    for doc in (actions.enumerate_actions("ledger", dataset="pbmc3k", store=tmp_path),
                actions.enumerate_actions("tune", dataset="pbmc3k", recipe="cluster",
                                          store=tmp_path),
                actions.enumerate_actions("ladder")):
        assert list(doc)[:4] == ["schema_version", "tool_version", "surface", "vocab_version"]
        assert doc["schema_version"] == actions.SCHEMA_VERSION == 1
        assert doc["tool_version"] == app._installed_build()
        assert doc["vocab_version"] == vocab.VERSION
    assert doc["surface"] == "ladder"
    assert actions.SURFACES == ("ledger", "tune", "ladder")


def test_ledger_offered_equals_tui_ledger(tmp_path):
    assert catalog.dataset_dir().is_dir()
    doc = actions.enumerate_actions("ledger", dataset="pbmc3k", store=tmp_path)
    obs = narrate.declared_observation(catalog.load_dataset("pbmc3k"), "pbmc3k")
    assert doc["offered"] == [r.as_offer() for r in state.ledger(obs, obs.provides(),
                                                                  measured=None)]
    # The base-commit values: measured, so a later recipe that moves them updates them here.
    assert list(doc)[4:] == ["dataset", "gate", "evidence", "offered", "detail"]
    assert doc["surface"] == "ledger"
    assert doc["gate"] is True
    assert doc["evidence"] is None
    assert doc["dataset"] == {"name": "pbmc3k", "shape": "clusters", "topology": ["clusters"],
                              "modality": "scrna", "handle_kind": "path",
                              "provides": ["genes"], "dims": None}
    assert doc["offered"][0] == {"recipe": "cluster", "can_run": True, "blocked": [],
                                 "recommended": True, "measured": "unmeasured"}
    assert doc["detail"]["cluster"]["steps"] == [
        "prep:normalize", "prep:transform", "latent:pca", "latent:umap", "latent:leiden",
        "analysis:cluster_quality"]
    assert doc["detail"]["contrast"]["unmet"] == ["conditions"]
    assert doc["detail"]["archetypes"]["offer_rank"] == 2
    assert list(doc["detail"]) == [o["recipe"] for o in doc["offered"]]
    for entry in doc["detail"].values():
        assert list(entry) == ["id", "steps", "claims", "suits", "unmet", "invalidated",
                               "unavailable", "support", "offer_rank"]
        assert entry["support"] == []
    # No phrasing and no outcomes: a recipe's question is not a statement of legality.
    text = json.dumps(doc, indent=2)

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)

    assert "question" not in set(keys(doc))
    for r in catalog.discover_recipes():
        assert json.dumps(catalog.load_recipe(r)["question"])[1:-1] not in text


def test_can_run_agrees_with_detail(tmp_path):
    assert catalog.dataset_dir().is_dir()
    names = catalog.discover_datasets()
    assert len(names) == 14
    for name in names:
        doc = actions.enumerate_actions("ledger", dataset=name, store=tmp_path)
        for o in doc["offered"]:
            d = doc["detail"][o["recipe"]]
            assert o["can_run"] == (d["unmet"] == [] and d["unavailable"] == []
                                    and d["support"] == []), (name, o["recipe"])


def test_catalog_overrides_and_no_data_file_dependence(tmp_path, monkeypatch):
    ddir = _toy(tmp_path)
    assert ddir.is_dir()
    absent = actions.enumerate_actions("ledger", dataset="toy", dataset_dir=ddir, store=tmp_path)
    assert absent["dataset"]["name"] == "toy"
    (tmp_path / "x.h5ad").write_bytes(b"")
    present = actions.enumerate_actions("ledger", dataset="toy", dataset_dir=ddir,
                                        store=tmp_path)
    assert json.dumps(present, indent=2) == json.dumps(absent, indent=2)

    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(ddir))
    assert actions.enumerate_actions("ledger", dataset="toy",
                                     store=tmp_path)["dataset"]["name"] == "toy"
    monkeypatch.delenv("MANYRUNS_DATASET_DIR")

    rdir = tmp_path / "recipes"
    rdir.mkdir()
    assert catalog.recipe_dir().is_dir()
    (rdir / "embed.yaml").write_text((catalog.recipe_dir() / "embed.yaml").read_text())
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(rdir))
    assert rdir.is_dir()
    doc = actions.enumerate_actions("ledger", dataset="pbmc3k", store=tmp_path)
    assert [o["recipe"] for o in doc["offered"]] == ["embed"]


def test_tune_offers_declared_params_and_tune_label(tmp_path):
    doc = actions.enumerate_actions("tune", dataset="pbmc3k", recipe="cluster", store=tmp_path)
    assert list(doc)[4:] == ["dataset", "gate", "evidence", "recipe", "step",
                             "offered", "knobs"]
    assert doc["surface"] == "tune"
    assert doc["dataset"]["name"] == "pbmc3k"
    assert doc["recipe"] == "cluster"
    assert doc["step"] is None
    assert doc["offered"] == [
        {"step": "umap", "params": {"n_components": 2}, "label": "umap n_components=2"},
        {"step": "leiden", "params": {}, "label": "leiden"}]
    recipes = _tunable_recipes()
    assert "cluster" in recipes
    for r in recipes:
        steps = [s for s in catalog.load_recipe(r)["steps"] if params.tunable_for(s)]
        offered = actions.enumerate_actions("tune", dataset="pbmc3k", recipe=r,
                                            store=tmp_path)["offered"]
        assert [o["step"] for o in offered] == [s["name"] for s in steps]
        for o, s in zip(offered, steps):
            assert o["label"] == tune._label(s)
            assert o["params"] == dict(s.get("params") or {})


def test_tune_knobs_carry_neighbours_and_null_cap(tmp_path):
    doc = actions.enumerate_actions("tune", dataset="pbmc3k", recipe="cluster", store=tmp_path)
    umap = doc["knobs"]["umap"]
    assert (umap["index"], umap["order"], umap["group"]) == (3, 0, "latent")
    assert doc["knobs"]["leiden"]["index"] == 4
    assert doc["knobs"]["leiden"]["order"] == 1
    assert list(umap["params"]) == ["n_neighbors", "min_dist", "n_components"]
    assert umap["params"]["n_components"] == {
        "current": 2, "source": "declared", "floor": 1, "cap": None, "left": 1, "right": 3,
        "domain": {"kind": "int", "min": 1, "min_open": False, "max": None, "max_open": False,
                   "also": []}}
    assert umap["params"]["n_neighbors"] == {
        "current": 15, "source": "default", "floor": 2, "cap": None, "left": 14, "right": 16,
        "domain": {"kind": "int", "min": 2, "min_open": False, "max": None, "max_open": False,
                   "also": []}}

    only = actions.enumerate_actions("tune", dataset="pbmc3k", recipe="cluster", step="leiden",
                                     store=tmp_path)
    assert only["step"] == "leiden"
    assert [o["step"] for o in only["offered"]] == ["leiden"]
    assert list(only["knobs"]) == ["leiden"]
    assert only["knobs"]["leiden"]["order"] == 0

    for r in _tunable_recipes():
        knobs = actions.enumerate_actions("tune", dataset="pbmc3k", recipe=r,
                                          store=tmp_path)["knobs"]
        for knob in knobs.values():
            for value in knob["params"].values():
                assert value["cap"] is None

    other = actions.enumerate_actions("tune", dataset="swissroll", recipe="cluster",
                                      store=tmp_path)
    assert other["dataset"]["name"] == "swissroll"
    assert _without(doc, "dataset") == _without(other, "dataset")


def test_ladder_reports_score_tables():
    doc = actions.enumerate_actions("ladder")
    assert list(doc)[4:] == ["rung", "incomparable", "overshoot_weight",
                             "claimed", "unroutable"]
    assert doc["rung"] == dict(score.RUNG)
    assert doc["rung"]["archetypal"] == 4
    assert doc["rung"]["surface"] == 0
    assert doc["incomparable"] == [[3, 4]]
    assert doc["overshoot_weight"] == 2.0
    assert doc["claimed"] == score.claimed_rungs()
    assert doc["unroutable"] == score.unroutable_rungs()


def test_probe_run_rule(tmp_path):
    assert tmp_path.is_dir()
    found = tmp_path / "found"
    found.mkdir()
    _write_store(found, [ROW_A])
    doc = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, seed=42,
                                    store=found)
    assert doc["gate"] is True
    assert doc["evidence"] == {"run_id": "aaaaaaaaaaaa", "recipe": "embed", "seed": 42}
    assert list(doc["evidence"]) == ["run_id", "recipe", "seed"]

    none = tmp_path / "none"
    none.mkdir()
    _write_store(none, [{**ROW_A, "ok": False}, {**ROW_A, "complete": False},
                        {**ROW_A, "seed": 101}])
    assert actions.probe_run("swissroll", 42, none) is None
    assert actions.enumerate_actions("ledger", dataset="swissroll", store=none)["evidence"] is None

    first = tmp_path / "first"
    first.mkdir()
    _write_store(first, [ROW_A, {**ROW_A, "run_id": "bbbbbbbbbbbb"}])
    assert actions.probe_run("swissroll", 42, first)["run_id"] == "aaaaaaaaaaaa"

    assert actions.probe_run("swissroll", 42, tmp_path / "empty") is None


def test_gate_is_recorded_and_changes_the_digest(tmp_path, monkeypatch):
    _write_store(tmp_path, [ROW_A])
    on = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, store=tmp_path)
    off = actions.enumerate_actions("ledger", dataset="swissroll", gate=False, store=tmp_path)
    assert on["gate"] is True and off["gate"] is False
    assert off["evidence"] is None
    assert _digest(on) != _digest(off)
    assert _without(on, "gate", "evidence") == _without(off, "gate", "evidence")
    for entry in on["detail"].values():
        assert entry["support"] == []

    assert actions.enumerate_actions("ledger", dataset="swissroll", seed=101,
                                     store=tmp_path)["evidence"] is None
    assert actions.enumerate_actions("ledger", dataset="swissroll", seed=42,
                                     store=tmp_path)["evidence"]["run_id"] == "aaaaaaaaaaaa"

    def refuse(*_a, **_k):
        raise AssertionError("gate off read the run record")

    monkeypatch.setattr(store, "read", refuse)
    assert actions.enumerate_actions("ledger", dataset="swissroll", gate=False,
                                     store=tmp_path)["evidence"] is None
