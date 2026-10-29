"""A driver's answers through the REAL tune loop: who moved a value reaches the decision row.

The gate records who answered (`Gate.answered_by`); the loop copies it onto the attempt whose
params it chose and onto the final choice; `_record_decision` writes a version 2 row when a
driver took part, and the version 1 row it always wrote when none did.
"""
from __future__ import annotations

import threading
import time

from manyruns import app, decisions, tune
from manyruns.session import Session
from manyruns.tui import gate as _gate

LLM = {"chooser": "llm:qwen3:1.7b", "bound": "schema", "legal_id": "abc123abc123"}


def _drive(tmp_path, monkeypatch, replies):
    monkeypatch.chdir(tmp_path)
    session = Session(project="p", engine="mock", modality="scrna",
                      recipe=app.load_recipe("cflows"))
    g = _gate.Gate()
    step = tune.parse_step_line("phate knn=5", session.engine)
    out: dict = {}
    t = threading.Thread(target=lambda: out.update(
        accepted=tune.run_tune_loop(session, step, g.read, g.write)), daemon=True)
    t.start()
    for text, by in replies:
        for _ in range(400):
            if g.prompt is not None:
                break
            time.sleep(0.005)
        assert g.prompt is not None, f"the loop never asked before {text!r}"
        assert g.answer(text, by=by)
    t.join(timeout=10)
    assert not t.is_alive()
    rows = [r for r in decisions.read(tune._corpus_dir(session)) if r.get("surface") == "tune"]
    return out["accepted"], rows


def test_a_driver_applied_value_and_accept_write_a_version_2_row(tmp_path, monkeypatch):
    accepted, rows = _drive(tmp_path, monkeypatch, [("knn=10", LLM), ("accept", LLM)])
    assert accepted["params"]["knn"] == 10
    row = rows[-1]
    assert row["schema_version"] == 2
    assert row["chooser"] == "llm:qwen3:1.7b" and row["bound"] == "schema"
    assert row["legal_id"] == "abc123abc123"
    assert [o.get("by") for o in row["offered"]] == [None, "llm:qwen3:1.7b"]


def test_the_scientist_accepting_a_drivers_card_is_the_humans_choice(tmp_path, monkeypatch):
    card = {"chooser": "human", "bound": "none", "legal_id": "card0000card",
            "proposed": {"params": {"knn": 10}, "chooser": "llm:qwen3:1.7b", "bound": "schema",
                         "rationale": "fewer neighbours keep the side branch"}}
    accepted, rows = _drive(tmp_path, monkeypatch, [("knn=10", card), ("accept", None)])
    row = rows[-1]
    assert row["schema_version"] == 2
    assert row["chooser"] == "human" and row["bound"] == "none"
    assert [o.get("by") for o in row["offered"]] == [None, "human"]
    # The card behind the kept attempt is recorded even though the accept came later and was
    # the scientist's own keystroke: a machine suggested the value that was kept.
    assert row["proposed"]["chooser"] == "llm:qwen3:1.7b"
    assert row["proposed"]["params"] == {"knn": 10}
    assert row["offered"][1]["proposed_by"] == "llm:qwen3:1.7b"
    assert row["legal_id"] == "card0000card"  # the offer the kept value was chosen against


def test_a_loop_no_driver_touched_writes_the_version_1_row(tmp_path, monkeypatch):
    accepted, rows = _drive(tmp_path, monkeypatch, [("knn=10", None), ("accept", None)])
    row = rows[-1]
    assert row["schema_version"] == 1
    assert all("by" not in o for o in row["offered"])


def test_a_driver_cancel_is_recorded_as_the_drivers(tmp_path, monkeypatch):
    accepted, rows = _drive(tmp_path, monkeypatch, [("knn=10", LLM), ("cancel", LLM)])
    assert accepted is None
    row = rows[-1]
    assert row["chosen"] is None and row["chooser"] == "llm:qwen3:1.7b"
