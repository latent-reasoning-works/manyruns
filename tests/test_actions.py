"""`manyruns actions <DATASET>` — the legal moves for a dataset, as one JSON document.

What is pinned here is that the document is the LEDGER, serialized, and nothing else: the same
rows in the same order with the same keys a decision row records, computed from the declaration
alone so the answer does not depend on which files this machine happens to hold.
"""
from __future__ import annotations

import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pytest

from manyruns import actions, app, catalog, narrate, vocab
from manyruns.tui import state


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A cwd and drop folder with no data in them, so nothing can be found by accident."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "drop"))
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: pytest.fail("actions must not download"))
    return tmp_path


def _actions(capsys, *argv):
    assert app.main(["actions", *argv]) == 0
    out = capsys.readouterr().out
    return out, json.loads(out)


def _expected_offer(name):
    ds = catalog.load_dataset(name)
    obs = narrate.declared_observation(ds, name)
    return [r.as_offer() for r in state.ledger(obs, obs.provides(), measured=None)]


# ── the document ─────────────────────────────────────────────────────────────
def test_actions_prints_exactly_one_json_document_schema_version_first(elsewhere, capsys):
    out, doc = _actions(capsys, "pbmc3k")
    assert list(doc)[0] == "schema_version" and doc["schema_version"] == 1
    assert list(doc) == ["schema_version", "tool_version", "dataset", "offered", "detail",
                         "legal_id"]
    assert doc["tool_version"] == app._installed_build()
    json.loads(out)                                   # the whole stream is one document


def test_actions_dataset_block_is_the_declaration(elsewhere, capsys):
    _, doc = _actions(capsys, "pbmc3k")
    assert doc["dataset"] == {"name": "pbmc3k", "shape": "clusters", "topology": ["clusters"],
                              "modality": "scrna", "handle_kind": "path",
                              "provides": ["genes"], "dims": None}


def test_actions_offered_is_the_ledger_serialized(elsewhere, capsys):
    for name in ("pbmc3k", "swissroll"):
        _, doc = _actions(capsys, name)
        assert doc["offered"] == _expected_offer(name)
        keys = set(state.LedgerRow(recipe="x", ask="", gloss="", can_run=True).as_offer())
        assert all(set(o) == keys for o in doc["offered"])


def test_actions_pbmc3k_legal_set(elsewhere, capsys):
    _, doc = _actions(capsys, "pbmc3k")
    by = {o["recipe"]: o for o in doc["offered"]}
    assert by["contrast"]["can_run"] is False
    assert by["contrast"]["blocked"] == ["needs a healthy/disease column"]
    for name in ("embed", "cluster", "pseudotime"):
        assert by[name]["can_run"] is True


def test_actions_detail_fields_and_can_run_invariant(elsewhere, capsys):
    _, doc = _actions(capsys, "pbmc3k")
    assert set(doc["detail"]) == {o["recipe"] for o in doc["offered"]}
    for o in doc["offered"]:
        d = doc["detail"][o["recipe"]]
        assert set(d) == {"id", "steps", "claims", "suits", "unmet", "invalidated",
                          "unavailable", "offer_rank"}
        assert all(":" in s for s in d["steps"])
        assert o["can_run"] == (d["unmet"] == [] and d["unavailable"] == [])
    contrast = doc["detail"]["contrast"]
    assert contrast["unmet"] == ["conditions"]
    recipe = catalog.load_recipe("contrast")
    assert contrast["steps"] == [f"{s['group']}:{s['name']}" for s in recipe["steps"]]


def test_actions_offer_rank_is_offers_order(elsewhere, capsys):
    _, doc = _actions(capsys, "pbmc3k")
    obs = narrate.declared_observation(catalog.load_dataset("pbmc3k"), "pbmc3k")
    order = [o["recipe"] for o in narrate.offer(obs, available=catalog.discover_recipes())]
    assert {r: d["offer_rank"] for r, d in doc["detail"].items()} == {
        r: order.index(r) for r in doc["detail"]}


def test_actions_carries_no_question_text_and_no_outcomes(elsewhere, capsys):
    out, _ = _actions(capsys, "pbmc3k")
    for name in catalog.discover_recipes():
        question = catalog.load_recipe(name).get("question")
        if question:
            assert question not in out
    for word in ("g_vector", "run_id", "outcome"):
        assert word not in out


# ── legal_id ─────────────────────────────────────────────────────────────────
def test_legal_id_is_order_free():
    v = "0.1.0 · revision unavailable"
    assert actions.legal_id(["b", "a"], v) == actions.legal_id(["a", "b"], v)
    assert actions.legal_id(["a", "b"], v) != actions.legal_id(["a"], v)
    assert actions.legal_id(["a"], v) != actions.legal_id(["a"], "other build")
    assert len(actions.legal_id(["a"], v)) == 12


def test_actions_legal_id_is_the_runnable_set_and_build(elsewhere, capsys):
    import hashlib

    _, doc = _actions(capsys, "pbmc3k")
    legal = sorted(o["recipe"] for o in doc["offered"] if o["can_run"])
    raw = json.dumps(legal) + "|" + doc["tool_version"]
    assert doc["legal_id"] == hashlib.sha256(raw.encode()).hexdigest()[:12]
    assert doc["legal_id"] == actions.legal_id(legal, doc["tool_version"])


# ── declared, not sniffed ────────────────────────────────────────────────────
def test_actions_identical_without_data_file(elsewhere, capsys):
    without, _ = _actions(capsys, "pbmc3k")
    drop = elsewhere / "drop"
    drop.mkdir()
    ad.AnnData(np.ones((4, 2), dtype=np.float32)).write_h5ad(drop / "pbmc3k_raw.h5ad")
    with_file, _ = _actions(capsys, "pbmc3k")
    assert with_file == without


def test_actions_opens_no_data_and_writes_nothing(elsewhere, capsys, monkeypatch):
    def no_data(*a, **k):
        pytest.fail("actions opened data")

    monkeypatch.setattr(narrate, "read_data", no_data)
    monkeypatch.setattr("manyruns.shell.dataset_ref_path", no_data)
    monkeypatch.setattr("anndata.read_h5ad", no_data)
    _actions(capsys, "pbmc3k")
    assert elsewhere.is_dir()
    assert sorted(p.name for p in elsewhere.iterdir()) == []


def test_declared_observation_is_the_declaration_alone():
    ds = catalog.load_dataset("pbmc3k")
    obs = narrate.declared_observation(ds, "pbmc3k")
    assert obs == narrate.Observation(shape="clusters", modality="scrna", source="pbmc3k",
                                      declared=("genes",))
    assert obs.provides() == frozenset({"genes"}) == narrate.declared_facts(ds)


def test_actions_imports_neither_the_tui_toolkit_nor_the_engine():
    code = ("import sys; from manyruns import actions; actions.document('pbmc3k', 'v'); "
            "bad = [m for m in sys.modules if m.split('.')[0] in ('textual', 'manylatents')]; "
            "print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout
    assert out.strip() == "[]"


# ── exits ────────────────────────────────────────────────────────────────────
def test_actions_without_a_dataset_is_a_usage_error(elsewhere, capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["actions"])
    assert exc.value.code == 2


def test_actions_unknown_dataset_exits_2(elsewhere, capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["actions", "no-such-dataset"])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""


def test_actions_unknown_surface_exits_2(elsewhere, capsys):
    with pytest.raises(SystemExit) as exc:
        app.main(["actions", "pbmc3k", "--surface", "tune"])
    assert exc.value.code == 2


def test_actions_broken_declaration_exits_1_with_empty_stdout(elsewhere, capsys):
    ddir = elsewhere / "catalog"
    ddir.mkdir()
    (ddir / "broken.yaml").write_text("name: broken\nshape: not-a-shape\n")
    assert ddir.is_dir()
    assert app.main(["actions", "broken", "--dataset-dir", str(ddir)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "broken" in captured.err


def test_actions_reads_the_dataset_dir_flag_and_overlay(elsewhere, capsys, monkeypatch):
    ddir = elsewhere / "catalog"
    ddir.mkdir()
    (ddir / "mine.yaml").write_text(
        "name: mine\nhandle: {kind: manylatents, ref: swissroll}\nshape: case-control\n")
    assert ddir.is_dir()
    _, doc = _actions(capsys, "mine", "--dataset-dir", str(ddir))
    assert doc["dataset"]["name"] == "mine" and doc["dataset"]["handle_kind"] == "manylatents"
    assert {o["recipe"]: o for o in doc["offered"]}["contrast"]["can_run"] is True
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(ddir))
    _, again = _actions(capsys, "mine")
    assert again == doc


def test_ledger_offer_is_the_one_offer_set(elsewhere):
    assert actions.ledger_offer("pbmc3k") == _expected_offer("pbmc3k")
    assert [o["recipe"] for o in actions.ledger_offer("pbmc3k")] == [
        r.recipe for r in state.ledger(
            narrate.declared_observation(catalog.load_dataset("pbmc3k"), "pbmc3k"),
            frozenset({"genes"}))]


def test_unmet_detail_is_the_calculus(elsewhere, capsys):
    _, doc = _actions(capsys, "pbmc3k")
    for name, d in doc["detail"].items():
        recipe = catalog.load_recipe(name)
        assert d["unmet"] == sorted(vocab.unmet(recipe, "clusters", frozenset({"genes"})))
        assert d["invalidated"] == sorted(
            vocab.invalidated(recipe, "clusters", frozenset({"genes"})))
