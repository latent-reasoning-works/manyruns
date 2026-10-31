"""`manyruns.action_space`: one choice document in, one record out, in either mode.

A harness gets an action document, a policy returns a choice document, and manyruns judges the
choice against the document whose digest it names, runs it, and writes the record with its own
code. These tests pin the checks, the refusals (each before the first write), the rows of both
surfaces, the review rows, the run-row extra and the parity of the two modes.

Every test uses a fresh empty store, because the first complete run of a dataset and seed in a
store is the probe run that the gate reads.
"""
from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from manyruns import action_space, actions
from manyruns.action_space import Refused, check_choice, offer_sha256

#: Task 1.0, measured on the base commit with `manyruns actions swissroll` in an empty folder:
#: the gate-on legal set of the bundled catalog.
SWISS_LEGAL = ["archetypes", "embed"]

CHOOSER = "prior:test.0123456789ab"

#: A run row that makes a probe run (the same row as `tests/test_actions.py`).
ROW_A = {"schema_version": 1, "run_id": "aaaaaaaaaaaa", "dataset_name": "swissroll", "seed": 42,
         "recipe": "embed", "ok": True, "complete": True,
         "g_vector": {"betti_0": 1, "lid": 2.1}}

BIG_META = {"note": "x" * 2100}
EXTRA = {"my_tool": {"arm": "a1", "fold": 3, "tags": ["x", "y"]}, "verdict_like": 0.5}


def _offer(store: Path, surface: str = "ledger", **kw) -> dict:
    """The library-mode action document, built without the wrapper."""
    assert Path(store).is_dir()
    return json.loads(json.dumps(actions.enumerate_actions(
        surface, dataset="swissroll", seed=42, store=store, **kw)))


def _write_store(root: Path, rows: list[dict]) -> None:
    (root / "outputs").mkdir(exist_ok=True)
    (root / "outputs" / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def _ledger_choice(doc: dict, **kw) -> dict:
    choice = {"schema_version": 1, "surface": "ledger", "dataset": "swissroll",
              "offer_sha256": offer_sha256(doc), "chosen": {"recipe": "embed"},
              "chooser": CHOOSER, "bound": "logits"}
    choice.update(kw)
    return choice


def _tune_choice(doc: dict, chosen: dict | None = None, **kw) -> dict:
    choice = {"schema_version": 1, "surface": "tune", "dataset": "swissroll",
              "offer_sha256": offer_sha256(doc),
              "chosen": chosen or {"recipe": "embed", "step": "phate", "params": {"knn": 7}},
              "chooser": CHOOSER, "bound": "logits"}
    choice.update(kw)
    return choice


def _reviewed(choice: dict, final: dict | None = None) -> dict:
    """A person's review of a filler's choice: an accept, or a replacement."""
    return action_space.review(choice, ask=lambda c: final or "accept", human_id="a3")


def _accepted(choice: dict) -> dict:
    """The reviewed choice of a person who accepted the filler's choice."""
    return _reviewed(choice)


class Shared:
    """The shared values of the spec, built in one fresh store."""

    def __init__(self, store: Path):
        self.store = store
        self.SWISS_DOC = _offer(store)
        self.GATE_OFF_DOC = _offer(store, gate=False)
        self.TUNE_DOC = _offer(store, "tune", recipe="embed")
        self.CLUSTER_TUNE_DOC = _offer(store, "tune", recipe="cluster", gate=False)
        self.LEDGER_CHOICE = _ledger_choice(self.SWISS_DOC)
        self.TUNE_CHOICE = _tune_choice(self.TUNE_DOC)
        self.GATE_OFF_CHOICE = _ledger_choice(self.GATE_OFF_DOC)
        self.ACCEPTED = _accepted(self.LEDGER_CHOICE)
        self.OVERRIDDEN = action_space.review(
            self.GATE_OFF_CHOICE, ask=lambda c: {"recipe": "cluster"}, human_id="a3")


@pytest.fixture
def S(tmp_path) -> Shared:
    store = tmp_path / "store"
    store.mkdir()
    return Shared(store)


def _refused(choice: dict, store: Path, *texts: str, code: int = 2) -> Refused:
    with pytest.raises(Refused) as caught:
        check_choice(choice, store=store, seed=42)
    assert caught.value.exit_code == code
    for text in texts:
        assert text in caught.value.message, (text, caught.value.message)
    return caught.value


# ── task 1.0: the measured values still hold ─────────────────────────────────────


def test_the_measured_legal_set_holds(S):
    legal = [o["recipe"] for o in S.SWISS_DOC["offered"] if o["can_run"]]
    assert sorted(legal) == sorted(SWISS_LEGAL)
    assert "embed" in legal and "cluster" not in legal and "contrast" not in legal
    assert S.SWISS_DOC["gate"] is True and S.SWISS_DOC["evidence"] is None
    assert S.GATE_OFF_DOC["gate"] is False
    off = {o["recipe"]: o["can_run"] for o in S.GATE_OFF_DOC["offered"]}
    assert off["cluster"] is True and off["pseudotime"] is True
    assert S.TUNE_DOC["knobs"]["phate"]["params"]["knn"] == {
        "current": 5, "source": "default", "floor": 1, "cap": None, "left": 4, "right": 6,
        "domain": {"kind": "int", "min": 1, "min_open": False, "max": None, "max_open": False,
                   "also": []}}
    assert list(S.CLUSTER_TUNE_DOC["knobs"]) == ["umap", "leiden"]


# ── task 1.1: the choice check ───────────────────────────────────────────────────


def test_check_choice_accepts_valid(S):
    assert check_choice(S.LEDGER_CHOICE, store=S.store, seed=42) == S.SWISS_DOC


def _without(d: dict, key: str) -> dict:
    return {k: v for k, v in d.items() if k != key}


def _proposed(choice: dict, **changes) -> dict:
    out = copy.deepcopy(choice)
    for key, value in changes.items():
        if value is _DROP:
            del out["proposed"][key]
        else:
            out["proposed"][key] = value
    return out


_DROP = object()

SHAPE_TABLE = [
    (lambda S: _without(S.LEDGER_CHOICE, "bound"), ["bound"]),
    (lambda S: {**S.LEDGER_CHOICE, "note": 1}, ["note"]),
    (lambda S: {**S.LEDGER_CHOICE, "schema_version": 2}, ["schema_version"]),
    (lambda S: {**S.LEDGER_CHOICE, "surface": "ladder"}, ["ladder"]),
    (lambda S: {**S.LEDGER_CHOICE, "dataset": "x.h5ad"}, ["catalog dataset name"]),
    (lambda S: {**S.LEDGER_CHOICE, "chooser": "human"}, ["chooser"]),
    (lambda S: {**S.LEDGER_CHOICE, "chooser": "llm:"}, ["chooser"]),
    (lambda S: {**S.LEDGER_CHOICE, "chooser": "llm:model a"}, ["chooser"]),
    (lambda S: {**S.LEDGER_CHOICE, "chooser": "human:Ada Lovelace"}, ["chooser"]),
    (lambda S: {**S.LEDGER_CHOICE, "chooser": "human:A3"}, ["chooser"]),
    (lambda S: {**S.LEDGER_CHOICE, "bound": "enum"}, ["bound"]),
    (lambda S: {**S.LEDGER_CHOICE, "scores": [{"action": {"recipe": "embed"}, "score": 0.2},
                                              {"action": {"recipe": "cluster"}, "score": 0.9}]},
     ["scores"]),
    (lambda S: {**S.LEDGER_CHOICE, "meta": BIG_META}, ["meta", "2048"]),
    (lambda S: {**S.LEDGER_CHOICE, "meta": {"a": {"b": 1}}}, ["meta"]),
    (lambda S: {**S.LEDGER_CHOICE, "meta": {"a": [[1]]}}, ["meta"]),
    (lambda S: {**S.LEDGER_CHOICE, "meta": {"a": list(range(17))}}, ["meta"]),
    (lambda S: {**S.LEDGER_CHOICE, "meta": {"a": float("nan")}}, ["meta"]),
    (lambda S: {**S.LEDGER_CHOICE, "extra": ["a"]}, ["extra"]),
    (lambda S: {**S.LEDGER_CHOICE, "extra": {"verdict": {}}}, ["verdict"]),
    (lambda S: {**S.LEDGER_CHOICE, "extra": {"proposer": {}}}, ["proposer"]),
    (lambda S: {**S.LEDGER_CHOICE, "extra": {"blob": "x" * 9000}}, ["extra", "8192"]),
    (lambda S: {**S.ACCEPTED, "chooser": CHOOSER}, ["proposed", "human:"]),
    (lambda S: _proposed(S.ACCEPTED, offer_sha256=_DROP), ["proposed", "offer_sha256"]),
    (lambda S: _proposed(S.ACCEPTED, note=1), ["proposed", "note"]),
    (lambda S: _proposed(S.ACCEPTED, offer_sha256="0" * 64), ["proposed.offer_sha256"]),
    (lambda S: _proposed(S.ACCEPTED, bound="enum"), ["proposed.bound"]),
    (lambda S: _proposed(S.ACCEPTED, chooser="human:Ada Lovelace"), ["proposed.chooser"]),
    (lambda S: _proposed(S.ACCEPTED, meta=BIG_META), ["proposed.meta", "2048"]),
]


@pytest.mark.parametrize("make, texts", SHAPE_TABLE, ids=[str(i) for i in range(len(SHAPE_TABLE))])
def test_check_choice_shape_refusals(S, make, texts):
    _refused(make(S), S.store, *texts)


@pytest.mark.parametrize("chooser", ["llm:model-a:8b", "llm:org/model-b@rev1"])
def test_llm_chooser_is_accepted(S, chooser):
    assert check_choice({**S.LEDGER_CHOICE, "chooser": chooser}, store=S.store) == S.SWISS_DOC


def test_valid_scores_are_accepted(S):
    scores = [{"action": {"recipe": "embed"}, "score": 0.9},
              {"action": {"recipe": "cluster"}, "score": 0.2}]
    assert check_choice({**S.LEDGER_CHOICE, "scores": scores}, store=S.store) == S.SWISS_DOC


def test_meta_at_cap_is_accepted(S):
    at_cap = {"note": "x" * (2048 - len('{"note":""}'))}
    size = len(json.dumps(at_cap, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8"))
    assert size == action_space.META_MAX_BYTES == 2048
    for meta in (at_cap, {"confidence": 0.8, "prompt": "p1", "tags": ["a", "b"], "flag": True,
                          "none": None}):
        assert check_choice({**S.LEDGER_CHOICE, "meta": meta}, store=S.store) == S.SWISS_DOC


def test_stale_offer_is_refused(S, tmp_path, monkeypatch):
    _refused({**S.LEDGER_CHOICE, "offer_sha256": "0" * 64}, S.store, "stale")
    # A changed catalog: only two of the bundled recipes remain.
    from manyruns import catalog

    bundled = catalog.recipe_dir()
    assert bundled.is_dir()
    narrow = tmp_path / "recipes"
    narrow.mkdir()
    for name in ("embed", "cluster"):
        (narrow / f"{name}.yaml").write_text((bundled / f"{name}.yaml").read_text())
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(narrow))
    _refused(S.LEDGER_CHOICE, S.store, "stale")


def test_filtered_tune_offer_is_not_stale(S):
    filtered = _offer(S.store, "tune", recipe="embed", step="phate")
    assert offer_sha256(filtered) != offer_sha256(S.TUNE_DOC)
    assert check_choice(_tune_choice(filtered), store=S.store) == filtered


def test_gate_off_offer_is_not_stale(S):
    assert check_choice(S.GATE_OFF_CHOICE, store=S.store) == S.GATE_OFF_DOC
    assert offer_sha256(S.GATE_OFF_DOC) != offer_sha256(S.SWISS_DOC)


def test_new_probe_run_makes_choice_stale(S):
    _write_store(S.store, [ROW_A])
    _refused(S.LEDGER_CHOICE, S.store, "stale")


def test_gate_decides_on_named_document(S):
    cluster = {"recipe": "cluster"}
    _refused({**S.LEDGER_CHOICE, "chosen": cluster}, S.store, "cluster", "betti_0:not_measured")
    assert check_choice({**S.GATE_OFF_CHOICE, "chosen": cluster}, store=S.store) == S.GATE_OFF_DOC


@pytest.mark.parametrize("chosen, texts", [
    ({"recipe": "contrast"}, ["contrast", "needs a healthy/disease column"]),
    ({"recipe": "nosuch"}, ["nosuch", "not offered"]),
    ({"recipe": "embed", "step": "phate"}, ["step"]),
    ("embed", ["chosen"]),
])
def test_ledger_legality(S, chosen, texts):
    _refused({**S.LEDGER_CHOICE, "chosen": chosen}, S.store, *texts)


TUNE_TABLE = [
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"knn": 7}}, None),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"knn": 1000}}, None),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"t": 25}}, None),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"t": "auto"}}, None),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"decay": 2.5}}, None),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"knn": 0}}, "knn"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"knn": 5.5}}, "knn"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"knn": True}}, "knn"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"knn": float("nan")}}, "knn"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"decay": float("inf")}}, "decay"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"t": "fast"}}, "t"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"t": 2.5}}, "t"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {"alpha": 2}}, "alpha"),
    ("TUNE_DOC", {"recipe": "embed", "step": "phate", "params": {}}, "params"),
    ("TUNE_DOC", {"recipe": "embed", "step": "pca", "params": {"n_components": 4}}, "pca"),
    ("TUNE_DOC", {"recipe": "cluster", "step": "umap", "params": {"n_neighbors": 20}}, "cluster"),
    ("CLUSTER_TUNE_DOC", {"recipe": "cluster", "step": "umap", "params": {"min_dist": 0.5}}, None),
    ("CLUSTER_TUNE_DOC", {"recipe": "cluster", "step": "umap", "params": {"min_dist": 1.5}},
     "min_dist"),
    ("CLUSTER_TUNE_DOC", {"recipe": "cluster", "step": "umap", "params": {"n_neighbors": 1}},
     "n_neighbors"),
    ("CLUSTER_TUNE_DOC", {"recipe": "cluster", "step": "leiden", "params": {"resolution": 3.0}},
     None),
]


@pytest.mark.parametrize("doc_name, chosen, text", TUNE_TABLE,
                         ids=[str(i) for i in range(len(TUNE_TABLE))])
def test_tune_legality(S, doc_name, chosen, text):
    doc = getattr(S, doc_name)
    choice = _tune_choice(doc, chosen)
    if text is None:
        assert check_choice(choice, store=S.store) == doc
    else:
        _refused(choice, S.store, text)


def test_tune_on_blocked_recipe(S):
    doc = _offer(S.store, "tune", recipe="contrast")
    choice = _tune_choice(doc, {"recipe": "contrast", "step": "phate", "params": {"knn": 7}})
    _refused(choice, S.store, "contrast", "needs a healthy/disease column")


def test_tune_on_gate_refused_recipe(S):
    doc = _offer(S.store, "tune", recipe="cluster")
    choice = _tune_choice(doc, {"recipe": "cluster", "step": "umap", "params": {"min_dist": 0.5}})
    _refused(choice, S.store, "cluster", "betti_0:not_measured")


def test_gate_refused_proposal_is_refused(S):
    filler = {**S.LEDGER_CHOICE, "chosen": {"recipe": "cluster"}}
    reviewed = _reviewed(filler, {"recipe": "embed"})
    _refused(reviewed, S.store, "proposed.chosen", "betti_0:not_measured")


def test_check_choice_writes_nothing(S):
    check_choice(S.LEDGER_CHOICE, store=S.store)
    assert list(S.store.iterdir()) == []


def test_non_finite_scores_are_refused(S):
    scores = [{"action": {"recipe": "embed"}, "score": math.inf}]
    _refused({**S.LEDGER_CHOICE, "scores": scores}, S.store, "scores")


# ── task 2: `manyruns run --choice FILE` ────────────────────────────────────────

_DECISION_LINE = re.compile(r"^decision ([0-9a-f]{12}) recorded in ", re.M)
_RUN_LINE = re.compile(r"^run ([0-9a-f]{12}) recorded in ", re.M)


def _child_env(home: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MANYRUNS_")}
    env.update({"HOME": str(home), "OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    # The child must import the manyruns this test imported; CI gives a relative PYTHONPATH.
    root = Path(action_space.__file__).resolve().parents[1]
    assert root.is_dir()
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(root), env.get("PYTHONPATH")) if p)
    return env


def _run_choice(store: Path, choice, *args: str, raw: str | None = None, before=()):
    """RUN_CHOICE: `python -m manyruns run --choice FILE --engine mock` in the store."""
    assert store.is_dir()
    path = store.parent / "choice.json"
    path.write_text(raw if raw is not None else json.dumps(choice))
    home = store.parent / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        [sys.executable, "-m", "manyruns", "run", *before, "--choice", str(path),
         "--engine", "mock", *args],
        cwd=store, env=_child_env(home), capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=300)


def _rows(store: Path) -> tuple[list[dict], list[dict]]:
    from manyruns import decisions, store as records

    return list(decisions.read(store / "outputs")), list(records.read(store / "outputs"))


def _ok(done) -> None:
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


def _refused_run(done, store: Path, *texts: str) -> None:
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    for text in texts:
        assert text in done.stderr, (text, done.stderr)
    assert not (store / "outputs").exists()


def test_ledger_decision_row_and_join(S):
    _ok(_run_choice(S.store, S.LEDGER_CHOICE))
    (row,), (run,) = _rows(S.store)
    assert row["chooser"] == CHOOSER and row["bound"] == "logits"
    assert row["surface"] == "ledger" and row["dataset"] == "swissroll"
    assert row["shape"] == "manifold" and row["topology"] == ["surface"]
    assert row["chosen"] == "embed" and row["run_id"] is None
    assert row["offered"] == S.SWISS_DOC["offered"] and row["gate"] is True
    assert run["recipe"] == "embed" and run["decision_id"] == row["decision_id"]


@pytest.mark.parametrize("gate_off", [False, True])
def test_decision_row_records_gate(S, gate_off):
    choice = ({**S.GATE_OFF_CHOICE, "chosen": {"recipe": "cluster"}} if gate_off
              else S.LEDGER_CHOICE)
    _ok(_run_choice(S.store, choice))
    (row,), (run,) = _rows(S.store)
    assert row["gate"] is (not gate_off) and list(row)[-1] == "gate"
    assert row["chosen"] == ("cluster" if gate_off else "embed")
    assert run["recipe"] == row["chosen"]


def test_two_output_lines(S):
    done = _run_choice(S.store, S.LEDGER_CHOICE)
    _ok(done)
    decided, ran = _DECISION_LINE.findall(done.stdout), _RUN_LINE.findall(done.stdout)
    assert len(decided) == 1 and len(ran) == 1
    assert _DECISION_LINE.search(done.stdout).start() < _RUN_LINE.search(done.stdout).start()
    (row,), (run,) = _rows(S.store)
    assert decided == [row["decision_id"]] and ran == [run["run_id"]]


def test_tune_decision_row_and_tuned_step(S):
    _ok(_run_choice(S.store, S.TUNE_CHOICE))
    (row,), (run,) = _rows(S.store)
    assert row["surface"] == "tune" and row["chosen"] == "phate knn=7 n_components=3"
    assert row["offered"] == S.TUNE_DOC["offered"] and row["gate"] is True
    assert run["decision_id"] == row["decision_id"]
    project = yaml.safe_load((S.store / "outputs" / "swissroll" / "project.yaml").read_text())
    phate, = [s for s in project["recipe"]["steps"] if s["name"] == "phate"]
    assert phate["params"] == {"n_components": 3, "knn": 7}


def test_failed_run_keeps_decision(S, monkeypatch, capsys):
    from manyruns import app, modes

    real = modes.run

    def failing(*args, **kwargs):
        results = real(*args, **kwargs)
        results["steps"] = [{**results["steps"][-1], "outcome": "error", "detail": "broken"}]
        results["ok"] = False
        return results

    monkeypatch.setattr(modes, "run", failing)
    monkeypatch.chdir(S.store)
    path = S.store.parent / "choice.json"
    path.write_text(json.dumps(S.LEDGER_CHOICE))
    assert app.main(["run", "--choice", str(path), "--engine", "mock"]) == 1
    (row,), (run,) = _rows(S.store)
    assert run["decision_id"] == row["decision_id"]


_FLAGS = [
    (["--recipe", "cluster"], [], "--recipe"),
    (["--dataset", "swissroll"], [], "--dataset"),
    (["--data", "x.h5ad"], [], "--data"),
    (["--modality", "scrna"], [], "--modality"),
    (["--transform", "sqrt"], [], "--transform"),
    (["--data-set", "n=3"], [], "--data-set"),
    (["--time-key", "day"], [], "--time-key"),
    (["--color-by", "label"], [], "--color-by"),
    (["--ask", "map the structure"], [], "--ask"),
    ([], ["swissroll"], "DATA"),
]


@pytest.mark.parametrize("args, before, text", _FLAGS, ids=[t for _, _, t in _FLAGS])
def test_run_choice_refuses_replaced_flags(S, args, before, text):
    _refused_run(_run_choice(S.store, S.LEDGER_CHOICE, *args, before=before), S.store, text)


@pytest.mark.parametrize("case", ["missing", "not_json", "not_object", "empty", "blocked",
                                  "stale"])
def test_run_choice_refusals_write_nothing(S, case):
    if case == "missing":
        done = subprocess.run(
            [sys.executable, "-m", "manyruns", "run", "--choice", "nosuch.json", "--engine",
             "mock"], cwd=S.store, env=_child_env(S.store.parent), capture_output=True,
            text=True, stdin=subprocess.DEVNULL, timeout=300)
        _refused_run(done, S.store, "nosuch.json")
        return
    raw = {"not_json": "{nope", "not_object": "[1, 2]", "empty": ""}.get(case)
    choice = {"blocked": {**S.LEDGER_CHOICE, "chosen": {"recipe": "contrast"}},
              "stale": {**S.LEDGER_CHOICE, "offer_sha256": "0" * 64}}.get(case)
    texts = {"blocked": ["contrast", "needs a healthy/disease column"],
             "stale": ["stale"]}.get(case, ["choice.json"])
    _refused_run(_run_choice(S.store, choice, raw=raw), S.store, *texts)


def test_non_finite_value_from_choice_file(S):
    raw = json.dumps(S.TUNE_CHOICE).replace('"knn": 7', '"knn": NaN')
    assert "NaN" in raw
    _refused_run(_run_choice(S.store, None, raw=raw), S.store, "knn")


def test_accept_row(S):
    _ok(_run_choice(S.store, S.ACCEPTED))
    (row,), _ = _rows(S.store)
    assert row["chooser"] == "human:a3" and row["bound"] == "none"
    assert row["chosen"] == "embed" and row["gate"] is True
    assert row["proposed"] == {"chooser": CHOOSER, "bound": "logits", "chosen": "embed",
                               "offer_sha256": offer_sha256(S.SWISS_DOC)}
    assert row["chosen"] == row["proposed"]["chosen"] and "overrode" not in row


def test_override_row(S):
    _ok(_run_choice(S.store, S.OVERRIDDEN))
    (row,), (run,) = _rows(S.store)
    assert row["chooser"] == "human:a3" and row["chosen"] == "cluster" and row["gate"] is False
    assert row["proposed"]["chosen"] == "embed" and row["proposed"]["chooser"] == CHOOSER
    assert row["chosen"] != row["proposed"]["chosen"] and "overrode" not in row
    assert run["recipe"] == "cluster"


def test_reviewed_row_key_order(S):
    _ok(_run_choice(S.store, S.ACCEPTED))
    (row,), _ = _rows(S.store)
    assert list(row) == ["schema_version", "decision_id", "run_id", "at", "surface", "chooser",
                         "bound", "legal_id", "dataset", "shape", "topology", "offered",
                         "chosen", "gate", "proposed"]
    assert row["schema_version"] == 1


def test_scores_and_meta_inside_proposed_only(S):
    scores = [{"action": {"recipe": "embed"}, "score": 0.9}]
    filler = {**S.LEDGER_CHOICE, "scores": scores, "meta": {"confidence": 0.8}}
    _ok(_run_choice(S.store, _accepted(filler)))
    (row,), _ = _rows(S.store)
    assert row["proposed"]["scores"] == scores and row["proposed"]["meta"] == {"confidence": 0.8}
    assert "scores" not in row and "meta" not in row


def test_illegal_proposal_is_refused(S):
    filler = {**S.LEDGER_CHOICE, "chosen": {"recipe": "contrast"}}
    reviewed = _reviewed(filler, {"recipe": "embed"})
    _refused_run(_run_choice(S.store, reviewed), S.store,
                 "proposed.chosen", "contrast", "needs a healthy/disease column")


def test_meta_over_cap_is_refused(S):
    big = _proposed(S.ACCEPTED, meta=BIG_META)
    _refused_run(_run_choice(S.store, big), S.store, "proposed.meta", "2048")


def test_row_without_proposed_reads_as_missing(S):
    from manyruns import decisions

    _ok(_run_choice(S.store, {**S.LEDGER_CHOICE, "chooser": "human:a3", "bound": "none"}))
    decisions.append(offered=[{"recipe": "embed"}], chosen="embed", out_dir=S.store / "outputs")
    path = S.store / "outputs" / "decisions.jsonl"
    before = path.read_bytes()
    rows, _ = _rows(S.store)
    assert len(rows) == 2
    assert all("proposed" not in r and r.get("proposed") is None for r in rows)
    assert path.read_bytes() == before


def test_examples_ignores_proposed(tmp_path):
    import inspect

    from manyruns import decisions

    stores = []
    for name in ("reviewed", "plain"):
        store = tmp_path / name / "store"
        store.mkdir(parents=True)
        shared = Shared(store)
        choice = (shared.ACCEPTED if name == "reviewed"
                  else {**shared.LEDGER_CHOICE, "chooser": "human:a3", "bound": "none"})
        _ok(_run_choice(store, choice))
        stores.append(store / "outputs")
    reviewed, plain = (decisions.examples(s) for s in stores)
    assert reviewed == plain and reviewed
    assert list(inspect.signature(decisions.examples).parameters) == [
        "out_dir", "surface", "exclude_chooser"]
    assert decisions.examples(stores[0], exclude_chooser="human:") == []


def test_choice_file_with_extra(S):
    _ok(_run_choice(S.store, {**S.LEDGER_CHOICE, "extra": EXTRA}))
    (row,), (run,) = _rows(S.store)
    assert run["extra"] == EXTRA and "extra" not in row


def test_no_extra_gives_no_key(S):
    _ok(_run_choice(S.store, S.LEDGER_CHOICE))
    _, (run,) = _rows(S.store)
    assert "extra" not in run


def test_plain_run_is_unchanged(S):
    done = subprocess.run(
        [sys.executable, "-m", "manyruns.app", "run", "swissroll", "--recipe", "embed",
         "--engine", "mock", "--project", "p1"], cwd=S.store, env=_child_env(S.store.parent),
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    _ok(done)
    assert not (S.store / "outputs" / "decisions.jsonl").exists()
    _, (run,) = _rows(S.store)
    assert "decision_id" not in run
    blocked = subprocess.run(
        [sys.executable, "-m", "manyruns.app", "run", "swissroll", "--recipe", "contrast",
         "--engine", "mock", "--project", "p2"], cwd=S.store, env=_child_env(S.store.parent),
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
    _ok(blocked)


# ── task 3: the wrapper, in both modes ────────────────────────────────────────────

MODES = ("library", "cli")


def _space(store: Path, mode: str, dataset: str = "swissroll", **kw):
    return action_space.ActionSpace(dataset, mode=mode, store=store, **kw)


def test_offer_equal_in_both_modes(S):
    lib, cli = (_space(S.store, m).offer("ledger") for m in MODES)
    assert lib == cli and offer_sha256(lib) == offer_sha256(cli)
    assert lib["offered"] == S.SWISS_DOC["offered"]
    lib, cli = (_space(S.store, m).offer("tune", recipe="embed") for m in MODES)
    assert lib == cli == S.TUNE_DOC
    lib, cli = (_space(S.store, m).offer("ladder") for m in MODES)
    assert lib == cli and lib["surface"] == "ladder"


@pytest.mark.parametrize("mode", MODES)
def test_offer_refusals_in_both_modes(S, mode):
    with pytest.raises(Refused) as caught:
        _space(S.store, mode, "no_such_dataset").offer("ledger")
    assert caught.value.exit_code == 2 and "no_such_dataset" in caught.value.message


def test_gate_is_an_argument_in_both_modes(S):
    lib, cli = (_space(S.store, m).offer("ledger", gate=False) for m in MODES)
    assert lib == cli == S.GATE_OFF_DOC
    assert offer_sha256(S.GATE_OFF_DOC) != offer_sha256(S.SWISS_DOC)


def test_both_modes_read_the_probe_run(S):
    _write_store(S.store, [ROW_A])
    lib, cli = (_space(S.store, m, seed=42).offer("ledger") for m in MODES)
    assert lib == cli
    assert lib["evidence"] == {"run_id": "aaaaaaaaaaaa", "recipe": "embed", "seed": 42}


def test_unknown_offer_keyword(S):
    with pytest.raises(TypeError):
        _space(S.store, "library").offer("ledger", seed=1)


def test_unknown_mode(S):
    with pytest.raises(ValueError, match="rpc"):
        action_space.ActionSpace("swissroll", mode="rpc")
    with pytest.raises(ValueError, match="nosuch"):
        action_space.ActionSpace("swissroll", store=S.store / "nosuch")


def _ids(store: Path) -> tuple[str, str]:
    (row,), (run,) = _rows(store)
    assert run["decision_id"] == row["decision_id"]
    return row["decision_id"], run["run_id"]


def test_act_library_result(S):
    result = _space(S.store, "library", engine="mock").act(S.LEDGER_CHOICE)
    decision_id, run_id = _ids(S.store)
    assert result == {"schema_version": 1, "decision_id": decision_id, "run_id": run_id,
                      "exit_code": 0}


def test_act_cli_result(S):
    result = _space(S.store, "cli", engine="mock").act(S.LEDGER_CHOICE)
    decision_id, run_id = _ids(S.store)
    assert result == {"schema_version": 1, "decision_id": decision_id, "run_id": run_id,
                      "exit_code": 0}
    assert [p.name for p in S.store.iterdir()] == ["outputs"]


@pytest.mark.parametrize("mode", MODES)
def test_act_refusal_in_both_modes(S, mode):
    with pytest.raises(Refused) as caught:
        _space(S.store, mode, engine="mock").act({**S.LEDGER_CHOICE,
                                                  "chosen": {"recipe": "contrast"}})
    assert caught.value.exit_code == 2 and "contrast" in caught.value.message
    assert not (S.store / "outputs").exists()


@pytest.mark.parametrize("mode", MODES)
def test_act_load_failure_in_both_modes(S, tmp_path, monkeypatch, mode):
    from manyruns import catalog

    bundled = catalog.dataset_dir()
    assert bundled.is_dir()
    broken = tmp_path / "datasets"
    broken.mkdir()
    (broken / "swissroll.yaml").write_text("name: [unclosed\n")
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(broken))
    with pytest.raises(Refused) as caught:
        _space(S.store, mode, engine="mock").act(S.LEDGER_CHOICE)
    assert caught.value.exit_code == 1, caught.value.message
    rows, runs = _rows(S.store)
    assert rows == [] and runs == []


def test_act_writes_no_record_itself():
    source = Path(action_space.__file__)
    assert source.is_file()
    assert not re.search(r"decisions\.append|store\.append", source.read_text())


@pytest.mark.parametrize("mode", MODES)
def test_act_writes_extra_in_both_modes(tmp_path, mode):
    store = tmp_path / mode
    store.mkdir()
    shared = Shared(store)
    _space(store, mode, engine="mock").act(shared.LEDGER_CHOICE, extra=EXTRA)
    (row,), (run,) = _rows(store)
    assert run["extra"] == EXTRA and "extra" not in row


def test_manyruns_ignores_extra_keys(S):
    from manyruns import score

    _space(S.store, "library", engine="mock").act(S.LEDGER_CHOICE, extra=EXTRA)
    _, (run,) = _rows(S.store)
    assert score.verdict_of(run) is None
    assert list(run["extra"]) == ["my_tool", "verdict_like"]
    package = Path(action_space.__file__).parent
    assert package.is_dir()
    # `store.py` names `my_tool` once, as an example in its module docstring; no code does.
    readers = []
    for path in package.rglob("*.py"):
        text = path.read_text()
        docstring = ast.get_docstring(ast.parse(text), clean=False) or ""
        if "my_tool" in text.replace(docstring, "", 1):
            readers.append(path.name)
    assert readers == []


@pytest.mark.parametrize("mode", MODES)
def test_extra_given_twice(S, mode):
    with pytest.raises(Refused) as caught:
        _space(S.store, mode, engine="mock").act({**S.LEDGER_CHOICE, "extra": {"a": 1}},
                                                 extra={"b": 2})
    assert caught.value.exit_code == 2 and "extra" in caught.value.message
    assert not (S.store / "outputs").exists()


@pytest.mark.parametrize("mode", MODES)
def test_reserved_extra_name_is_refused(S, mode):
    with pytest.raises(Refused) as caught:
        _space(S.store, mode, engine="mock").act(S.LEDGER_CHOICE,
                                                 extra={"verdict": {"topology_probs": {}}})
    assert caught.value.exit_code == 2 and "verdict" in caught.value.message
    assert not (S.store / "outputs").exists()


def _normalized(value, store: Path):
    if isinstance(value, str):
        return value.replace(str(store), "<store>")
    if isinstance(value, dict):
        return {k: _normalized(v, store) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalized(v, store) for v in value]
    return value


def _parity(row: dict, drop: tuple[str, ...], store: Path) -> dict:
    out = {k: v for k, v in row.items() if k not in drop}
    if "steps" in out:
        out["steps"] = [{k: v for k, v in s.items() if k != "seconds"} for s in out["steps"]]
    if isinstance(out.get("g_vector"), dict):
        # A wall-clock time, like each step's `seconds`: the one other value that differs
        # between two runs of one choice.
        out["g_vector"] = {k: v for k, v in out["g_vector"].items() if k != "suite_seconds"}
    return _normalized(out, store)


@pytest.mark.parametrize("name", ["LEDGER_CHOICE", "TUNE_CHOICE", "ACCEPTED"])
def test_modes_write_equal_records(tmp_path, name):
    pairs = {}
    for mode in MODES:
        store = tmp_path / mode / "store"
        store.mkdir(parents=True)
        choice = getattr(Shared(store), name)
        result = _space(store, mode, engine="mock").act(choice)
        assert result["exit_code"] == 0
        (row,), (run,) = _rows(store)
        pairs[mode] = (_parity(row, ("decision_id", "at"), store),
                       _parity(run, ("run_id", "decision_id", "at"), store))
    assert pairs["library"] == pairs["cli"]
    if name == "TUNE_CHOICE":
        assert pairs["cli"][0]["chosen"] == "phate knn=7 n_components=3"


# ── task 3b: the review helper ───────────────────────────────────────────────────


def test_review_accept(S):
    assert action_space.review(S.LEDGER_CHOICE, ask=lambda c: "accept", human_id="a3") == {
        "schema_version": 1, "surface": "ledger", "dataset": "swissroll",
        "offer_sha256": offer_sha256(S.SWISS_DOC), "chosen": {"recipe": "embed"},
        "chooser": "human:a3", "bound": "none",
        "proposed": {"chooser": CHOOSER, "bound": "logits", "chosen": {"recipe": "embed"},
                     "offer_sha256": offer_sha256(S.SWISS_DOC)}}


def test_review_replacement(S):
    out = action_space.review(S.GATE_OFF_CHOICE, ask=lambda c: {"recipe": "cluster"},
                              human_id="a3")
    assert out["chosen"] == {"recipe": "cluster"} and out["chooser"] == "human:a3"
    assert out["proposed"]["chosen"] == {"recipe": "embed"}


def test_review_tune_replacement(S):
    replacement = {"recipe": "embed", "step": "phate", "params": {"knn": 9}}
    reviewed = action_space.review(S.TUNE_CHOICE, ask=lambda c: replacement, human_id="a3")
    _ok(_run_choice(S.store, reviewed))
    (row,), _ = _rows(S.store)
    assert row["chosen"] == "phate knn=9 n_components=3"
    assert row["proposed"]["chosen"] == "phate knn=7 n_components=3"


def test_review_ask_sees_copy(S):
    before = copy.deepcopy(S.LEDGER_CHOICE)

    def ask(choice):
        choice["chosen"]["recipe"] = "cluster"
        choice["chooser"] = "human:x"
        return "accept"

    out = action_space.review(S.LEDGER_CHOICE, ask=ask, human_id="a3")
    assert S.LEDGER_CHOICE == before and out["chosen"] == {"recipe": "embed"}


@pytest.mark.parametrize("case", ["Ada Lovelace", "A3", "", "proposed", "no_bound"])
def test_review_bad_arguments(S, case):
    def ask(choice):
        pytest.fail("ask was called")

    choice, human_id = S.LEDGER_CHOICE, case
    if case == "proposed":
        choice, human_id = S.ACCEPTED, "a3"
    elif case == "no_bound":
        choice, human_id = _without(S.LEDGER_CHOICE, "bound"), "a3"
    with pytest.raises(ValueError):
        action_space.review(choice, ask=ask, human_id=human_id)


@pytest.mark.parametrize("answer", [None, "yes", ["embed"]])
def test_review_bad_ask_result(S, answer):
    with pytest.raises(ValueError):
        action_space.review(S.LEDGER_CHOICE, ask=lambda c: answer, human_id="a3")


def test_review_calls_no_act(S, monkeypatch):
    import builtins

    from manyruns import annotations, decisions, store  # noqa: F401 - loaded before open fails

    def fail(*args, **kwargs):
        pytest.fail("review touched the record")

    monkeypatch.setattr(action_space.ActionSpace, "act", fail)
    monkeypatch.setattr(decisions, "append", fail)
    monkeypatch.setattr(store, "append", fail)
    monkeypatch.setattr(builtins, "open", fail)
    out = action_space.review(S.LEDGER_CHOICE, ask=lambda c: "accept", human_id="a3")
    assert out["chooser"] == "human:a3"


# ── task 3a: the tool view ─────────────────────────────────────────────────────────

from manyruns.action_space import as_tools, choice_from_tool_call  # noqa: E402


def test_ledger_tool_lists_legal_recipes(S):
    tools = as_tools(S.GATE_OFF_DOC)
    assert [t["name"] for t in tools] == ["choose_recipe"]
    assert set(tools[0]) == {"name", "description", "parameters"}
    enum = tools[0]["parameters"]["properties"]["recipe"]["enum"]
    assert enum == [o["recipe"] for o in S.GATE_OFF_DOC["offered"] if o["can_run"]]
    assert "contrast" not in enum and "cluster" in enum
    assert tools[0]["parameters"]["required"] == ["recipe"]
    assert tools[0]["parameters"]["additionalProperties"] is False
    assert tools[0]["description"] == ("Choose the next recipe that manyruns runs on swissroll. "
                                       "Only the listed recipes are legal.")


def test_gate_removes_recipe_from_tool(S):
    enum = as_tools(S.SWISS_DOC)[0]["parameters"]["properties"]["recipe"]["enum"]
    assert "cluster" not in enum and "embed" in enum


def test_empty_legal_set_gives_no_tool(S):
    doc = copy.deepcopy(S.SWISS_DOC)
    for entry in doc["offered"]:
        entry["can_run"] = False
    assert as_tools(doc) == []


def test_tune_tools_carry_bounds(S):
    phate, = as_tools(S.TUNE_DOC)
    assert phate["name"] == "tune_phate"
    assert phate["description"] == ("Set parameters of step phate in recipe embed. "
                                    "Values outside the stated bounds are refused.")
    params = phate["parameters"]
    assert params["type"] == "object" and params["minProperties"] == 1
    assert params["additionalProperties"] is False
    assert params["properties"]["knn"] == {"type": "integer", "minimum": 1,
                                           "description": "knn, current 5"}
    assert params["properties"]["t"] == {
        "anyOf": [{"type": "integer", "minimum": 1}, {"type": "string", "enum": ["auto"]}],
        "description": "t, current auto"}
    umap, leiden = as_tools(S.CLUSTER_TUNE_DOC)
    assert (umap["name"], leiden["name"]) == ("tune_umap", "tune_leiden")
    assert umap["parameters"]["properties"]["min_dist"] == {
        "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "min_dist, current 0.1"}
    assert umap["parameters"]["properties"]["n_neighbors"] == {
        "type": "integer", "minimum": 2, "description": "n_neighbors, current 15"}


@pytest.mark.parametrize("knob, domain, schema", [
    # An open domain minimum above the floor: `exclusiveMinimum` for a number ...
    ({"floor": None, "cap": None, "current": 1.0},
     {"kind": "number", "min": 0, "min_open": True, "max": None, "max_open": False, "also": []},
     {"type": "number", "exclusiveMinimum": 0}),
    # ... and the next integer for an int.
    ({"floor": None, "cap": None, "current": 2},
     {"kind": "int", "min": 0, "min_open": True, "max": 10, "max_open": True, "also": []},
     {"type": "integer", "minimum": 1, "maximum": 9}),
    # A tie between the floor and an open minimum: the open one wins.
    ({"floor": 0, "cap": 5, "current": 1},
     {"kind": "number", "min": 0, "min_open": True, "max": None, "max_open": False, "also": []},
     {"type": "number", "exclusiveMinimum": 0, "maximum": 5}),
])
def test_knob_schema_bounds(knob, domain, schema):
    out = action_space._knob_schema("k", {**knob, "domain": domain})
    assert out == {**schema, "description": f"k, current {knob['current']}"}


def test_ladder_has_no_tools(S):
    assert as_tools(_space(S.store, "library").offer("ladder")) == []


def test_schema_fits_a_function_tool(S):
    """The shape a function-calling loop sends: `{"type": "function", "function": spec}`."""
    for spec in as_tools(S.TUNE_DOC) + as_tools(S.SWISS_DOC):
        assert set(spec) == {"name", "description", "parameters"}
        assert isinstance(spec["name"], str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}",
                                                               spec["name"])
        assert isinstance(spec["description"], str) and spec["parameters"]["type"] == "object"
        wrapped = {"type": "function", "function": spec}
        assert json.loads(json.dumps(wrapped))["function"] == spec


def test_choice_from_ledger_call(S):
    choice = choice_from_tool_call(S.SWISS_DOC, "choose_recipe", {"recipe": "embed"},
                                   chooser="llm:model-a:8b", bound="schema")
    assert choice == {"schema_version": 1, "surface": "ledger", "dataset": "swissroll",
                      "offer_sha256": offer_sha256(S.SWISS_DOC), "chosen": {"recipe": "embed"},
                      "chooser": "llm:model-a:8b", "bound": "schema"}
    assert check_choice(choice, store=S.store) == S.SWISS_DOC


def test_choice_from_tune_call(S):
    choice = choice_from_tool_call(S.TUNE_DOC, "tune_phate", {"t": 25},
                                   chooser="llm:model-a:8b", bound="schema")
    assert choice["chosen"] == {"recipe": "embed", "step": "phate", "params": {"t": 25}}
    assert check_choice(choice, store=S.store) == S.TUNE_DOC


@pytest.mark.parametrize("doc_name, name, arguments", [
    ("TUNE_DOC", "tune_pca", {}),
    ("SWISS_DOC", "tune_phate", {"knn": 3}),
    ("TUNE_DOC", "choose_recipe", {"recipe": "embed"}),
    ("SWISS_DOC", "choose_recipe", ["embed"]),
])
def test_unknown_tool_name(S, doc_name, name, arguments):
    with pytest.raises(Refused) as caught:
        choice_from_tool_call(getattr(S, doc_name), name, arguments, chooser="llm:x",
                              bound="schema")
    assert caught.value.exit_code == 2
    assert (name if isinstance(arguments, dict) else "arguments") in caught.value.message


def test_no_provider_import(S, tmp_path):
    """A fresh interpreter: the wrapper and its tool view load nothing outside the standard
    library and manyruns, so no provider or agent package, and no Gymnasium."""
    saved = tmp_path / "tune.json"
    saved.write_text(json.dumps(S.TUNE_DOC))
    probe = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import manyruns.action_space as a\n"
        f"a.as_tools(json.load(open({str(saved)!r})))\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "outside = sorted(m for m in new if m not in sys.stdlib_module_names\n"
        "                 and m != 'manyruns' and not m.startswith('_'))\n"
        "gym = sorted(m for m in sys.modules if m.startswith(('gymnasium', 'gym.'))\n"
        "             or m == 'gym')\n"
        "print(json.dumps({'outside': outside, 'gym': gym}))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path,
                          env=_child_env(tmp_path), capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {"outside": [], "gym": []}


def test_no_new_dependency():
    """The wrapper imports only the standard library and manyruns, at any depth of the file,
    so it needs no new entry in `pyproject.toml`."""
    source = Path(action_space.__file__)
    assert source.is_file()
    names = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    assert names - {"manyruns", "__future__"} <= set(sys.stdlib_module_names)
    pyproject = Path(action_space.__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.is_file():
        assert "gymnasium" not in pyproject.read_text()


def test_importing_main_runs_nothing(tmp_path):
    """A package walk imports `manyruns.__main__`; that import must not start the CLI."""
    done = subprocess.run([sys.executable, "-c", "import manyruns.__main__; print('IMPORTED')"],
                          cwd=tmp_path, env=_child_env(tmp_path), capture_output=True, text=True,
                          stdin=subprocess.DEVNULL, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "IMPORTED"
