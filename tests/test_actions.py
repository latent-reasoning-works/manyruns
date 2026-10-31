"""`manyruns actions`: the legal action space for one dataset, as one JSON document.

The verb is read-only. The ledger surface is `tui.state.ledger` serialized, never a second
account of it; the tune surface is the declared params and their one-press neighbours; the ladder
surface is the scorer's tables. These tests pin that the document equals what those functions
say, that it depends on no data file, and that it is the same document in process and on stdout.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

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
    """With the gate off: the gate-on document of a dataset with no probe run refuses every
    ruled claim (`cluster` with `betti_0:not_measured`), which the gate scenarios pin below."""
    assert catalog.dataset_dir().is_dir()
    doc = actions.enumerate_actions("ledger", dataset="pbmc3k", gate=False, store=tmp_path)
    obs = narrate.declared_observation(catalog.load_dataset("pbmc3k"), "pbmc3k")
    assert doc["offered"] == [r.as_offer() for r in state.ledger(obs, obs.provides(),
                                                                  measured=None)]
    # The base-commit values: measured, so a later recipe that moves them updates them here.
    assert list(doc)[4:] == ["dataset", "gate", "evidence", "offered", "detail"]
    assert doc["surface"] == "ledger"
    assert doc["gate"] is False
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
    for entry in off["detail"].values():
        assert entry["support"] == []
    # On the tune surface the gate changes only the keys `gate` and `evidence`.
    t_on = actions.enumerate_actions("tune", dataset="swissroll", recipe="cluster", gate=True,
                                     store=tmp_path)
    t_off = actions.enumerate_actions("tune", dataset="swissroll", recipe="cluster", gate=False,
                                      store=tmp_path)
    assert _without(t_on, "gate", "evidence") == _without(t_off, "gate", "evidence")

    assert actions.enumerate_actions("ledger", dataset="swissroll", seed=101,
                                     store=tmp_path)["evidence"] is None
    assert actions.enumerate_actions("ledger", dataset="swissroll", seed=42,
                                     store=tmp_path)["evidence"]["run_id"] == "aaaaaaaaaaaa"

    def refuse(*_a, **_k):
        raise AssertionError("gate off read the run record")

    monkeypatch.setattr(store, "read", refuse)
    assert actions.enumerate_actions("ledger", dataset="swissroll", gate=False,
                                     store=tmp_path)["evidence"] is None


# ── the gate applies the rung-support rule from the probe run ─────────────────


def _offered(doc: dict) -> dict:
    return {o["recipe"]: o for o in doc["offered"]}


def test_the_probe_run_refuses_unsupported_claims(tmp_path):
    _write_store(tmp_path, [ROW_A])
    doc = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, seed=42,
                                    store=tmp_path)
    offered = _offered(doc)
    assert offered["cluster"]["can_run"] is False
    assert "betti_0:fails_rule" in offered["cluster"]["blocked"]
    assert offered["pseudotime"]["can_run"] is False
    assert "lid:fails_rule" in offered["pseudotime"]["blocked"]
    assert doc["detail"]["cluster"]["support"] == ["betti_0:fails_rule"]
    assert offered["embed"]["can_run"] is True
    assert doc["detail"]["embed"]["support"] == []


def test_the_document_holds_no_measured_value(tmp_path):
    _write_store(tmp_path, [ROW_A])
    doc = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, seed=42,
                                    store=tmp_path)
    assert set(doc["evidence"]) == {"run_id", "recipe", "seed"}
    for entry in doc["detail"].values():
        assert all(r in vocab.SUPPORT_REASONS for r in entry["support"])
    keys = tuple(k for k, _ in vocab.RUNG_EVIDENCE.values())
    offered = _offered(doc)
    for name in ("cluster", "pseudotime"):
        for reason in offered[name]["blocked"]:
            if reason.startswith(keys):
                assert reason in vocab.SUPPORT_REASONS, reason
    assert "2.1" not in json.dumps(doc["offered"])
    assert "2.1" not in json.dumps(doc["detail"])


def test_the_reeb_readout_is_not_admitted_in_v1(tmp_path):
    _write_store(tmp_path, [ROW_A])
    doc = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, seed=42,
                                    store=tmp_path)
    assert doc["detail"]["branches"]["support"] == [
        "reeb_graph.structural_summary.n_branch_points:not_admitted"]


def test_gate_off_removes_the_mask(tmp_path):
    _write_store(tmp_path, [ROW_A])
    doc = actions.enumerate_actions("ledger", dataset="swissroll", gate=False, seed=42,
                                    store=tmp_path)
    offered = _offered(doc)
    assert offered["cluster"]["can_run"] is True
    assert offered["pseudotime"]["can_run"] is True
    assert all(entry["support"] == [] for entry in doc["detail"].values())
    assert doc["evidence"] is None


def test_no_probe_run_refuses_every_ruled_claim_and_keeps_the_rest(tmp_path):
    assert tmp_path.is_dir()
    doc = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, store=tmp_path)
    offered = _offered(doc)
    assert doc["evidence"] is None
    assert "betti_0:not_measured" in offered["cluster"]["blocked"]
    assert "lid:not_measured" in offered["pseudotime"]["blocked"]
    assert offered["embed"]["can_run"] is True


def test_the_admitted_set_follows_the_declared_suite(tmp_path, monkeypatch):
    mdir = tmp_path / "metrics"
    mdir.mkdir()
    (mdir / "default.yaml").write_text("metrics:\n  - trustworthiness\n")
    assert mdir.is_dir()
    monkeypatch.setenv("MANYRUNS_METRICS_DIR", str(mdir))
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    _write_store(store_dir, [ROW_A])
    doc = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, seed=42,
                                    store=store_dir)
    assert doc["detail"]["cluster"]["support"] == ["betti_0:not_admitted"]


def test_a_gate_off_choice_is_not_a_gate_on_choice(tmp_path):
    _write_store(tmp_path, [ROW_A])
    on = actions.enumerate_actions("ledger", dataset="swissroll", gate=True, store=tmp_path)
    off = actions.enumerate_actions("ledger", dataset="swissroll", gate=False, store=tmp_path)
    assert _digest(on) != _digest(off)


# ── task 3: the CLI ─────────────────────────────────────────────────────────────


def _cli(argv: list[str], capsys) -> tuple[int, str, str]:
    """`app.main(argv)` as a process would see it: exit code, stdout, stderr."""
    try:
        code = app.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    out, err = capsys.readouterr()
    return code, out, err


def test_actions_is_a_subcommand(capsys):
    assert app._normalize_argv(["actions", "pbmc3k"]) == ["actions", "pbmc3k"]
    args = app._build_parser().parse_args(["actions", "pbmc3k"])
    assert args.func is app.cmd_actions
    assert args.surface == "ledger"
    assert args.dataset == "pbmc3k"
    code, out, _ = _cli(["--help"], capsys)
    assert code == 0
    assert "actions" in out


def test_repeat_runs_are_byte_identical(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    first = _cli(["actions", "pbmc3k"], capsys)
    second = _cli(["actions", "pbmc3k"], capsys)
    assert first[0] == 0
    assert first[1].endswith("}\n")
    assert first[1].encode() == second[1].encode()


@pytest.mark.parametrize("argv, needle", [
    ([], None),
    (["--surface", "tune", "--recipe", "cluster"], None),
    (["no_such_dataset"], "no_such_dataset"),
    (["no_such_dataset", "--surface", "tune", "--recipe", "cluster"], "no_such_dataset"),
    (["pbmc3k", "--surface", "tune"], "--recipe"),
    (["pbmc3k", "--surface", "tune", "--recipe", "no_such_recipe"], "no_such_recipe"),
    (["pbmc3k", "--surface", "tune", "--recipe", "cluster", "--step", "pca"], "pca"),
    (["pbmc3k", "--recipe", "cluster"], "--recipe"),
    (["pbmc3k", "--step", "umap"], "--step"),
    (["pbmc3k", "--surface", "ladder"], "DATASET"),
    (["--surface", "ladder", "--recipe", "cluster"], "--recipe"),
    (["--surface", "ladder", "--step", "umap"], "--step"),
    (["--surface", "ladder", "--gate", "off"], "--gate"),
    (["--surface", "ladder", "--seed", "1"], "--seed"),
    (["pbmc3k", "--gate", "maybe"], "maybe"),
    (["pbmc3k", "--surface", "sweep"], "sweep"),
    (["--surface", "ladder", "--dataset-dir", "<empty>"], None),
])
def test_usage_errors_exit_2_with_empty_stdout(argv, needle, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    if "<empty>" in argv:
        empty = tmp_path / "empty"
        empty.mkdir()
        assert empty.is_dir()
        argv = [str(empty) if a == "<empty>" else a for a in argv]
        # `--dataset-dir` is accepted on every surface and changes nothing on the ladder.
        code, out, _ = _cli(["actions", *argv], capsys)
        assert code == 0
        assert out == _cli(["actions", "--surface", "ladder"], capsys)[1]
        return
    code, out, err = _cli(["actions", *argv], capsys)
    assert code == 2
    assert out == ""
    if needle is not None:
        assert needle in err


def test_load_failure_exits_1_with_empty_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ddir = tmp_path / "datasets"
    ddir.mkdir()
    (ddir / "broken.yaml").write_text("name: [")
    assert ddir.is_dir()
    code, out, err = _cli(["actions", "broken", "--dataset-dir", str(ddir)], capsys)
    assert (code, out) == (1, "")
    assert "dataset" in err and "broken" in err
    assert len(err.strip().splitlines()) == 1

    assert catalog.recipe_dir().is_dir()
    cluster = (catalog.recipe_dir() / "cluster.yaml").read_text()
    rdir = tmp_path / "recipes"
    rdir.mkdir()
    (rdir / "cluster.yaml").write_text(cluster)
    (rdir / "broken.yaml").write_text("name: [")
    assert rdir.is_dir()
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(rdir))
    code, out, err = _cli(["actions", "pbmc3k", "--surface", "tune", "--recipe", "broken"],
                          capsys)
    assert (code, out) == (1, "")
    assert "recipe" in err and "broken" in err
    assert len(err.strip().splitlines()) == 1

    # On the ledger a recipe that does not load is a missing row, not a failure of the verb.
    code, out, _ = _cli(["actions", "pbmc3k"], capsys)
    assert code == 0
    doc = json.loads(out)
    assert [o["recipe"] for o in doc["offered"]] == ["cluster"]
    assert "broken" not in doc["detail"]

    twice = yaml.safe_load(cluster)
    twice["name"] = "twice"
    umap = next(i for i, s in enumerate(twice["steps"]) if s["name"] == "umap")
    twice["steps"].insert(umap + 1, dict(twice["steps"][umap]))
    assert catalog.check_recipe(twice, "twice") == []
    (rdir / "twice.yaml").write_text(yaml.safe_dump(twice))
    code, out, err = _cli(["actions", "pbmc3k", "--surface", "tune", "--recipe", "twice"],
                          capsys)
    assert (code, out) == (1, "")
    assert "twice" in err and "umap" in err


def test_no_heavy_import_no_session_and_no_write(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    assert cwd.is_dir()
    probe = (
        "import json, sys\n"
        "import manyruns.actions\n"
        "manyruns.actions.enumerate_actions('ledger', dataset='pbmc3k')\n"
        "from manyruns import app\n"
        "code = app.main(['actions', 'pbmc3k'])\n"
        "bad = sorted(k for k in sys.modules if k.startswith(('textual', 'manylatents'))\n"
        "             or k == 'manyruns.session')\n"
        "print(json.dumps({'code': code, 'bad': bad}), file=sys.stderr)\n"
    )
    # CI supplies a relative PYTHONPATH; the child starts in a different directory.
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), *sys.path))}
    done = subprocess.run([sys.executable, "-c", probe], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stderr.strip().splitlines()[-1])
    assert report == {"code": 0, "bad": []}
    assert json.loads(done.stdout)["schema_version"] == 1
    assert list(cwd.iterdir()) == []


def test_returned_and_printed_documents_have_one_digest(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cases = [
        (["pbmc3k"], dict(surface="ledger", dataset="pbmc3k")),
        (["pbmc3k", "--surface", "tune", "--recipe", "cluster"],
         dict(surface="tune", dataset="pbmc3k", recipe="cluster")),
        (["--surface", "ladder"], dict(surface="ladder")),
    ]
    docs = {}
    for argv, kwargs in cases:
        surface = kwargs.pop("surface")
        doc = actions.enumerate_actions(surface, **kwargs)
        code, out, _ = _cli(["actions", *argv], capsys)
        assert code == 0
        printed = json.loads(out)
        assert doc == printed
        assert _digest(doc) == _digest(printed)
        docs[surface] = doc
    assert docs["ladder"]["incomparable"] == [[3, 4]]
    assert "dataset" not in docs["ladder"]
    # A choice is checkable from the document alone.
    assert docs["ledger"]["surface"] == "ledger"
    assert docs["ledger"]["dataset"]["name"] == docs["tune"]["dataset"]["name"] == "pbmc3k"
    assert docs["ledger"]["offered"]
    assert (docs["tune"]["recipe"], docs["tune"]["step"]) == ("cluster", None)
    assert set(docs["tune"]["knobs"]) == {"umap", "leiden"}
