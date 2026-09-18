"""The live dashboard — a run you can watch, rather than a wall of output at the end.

`run_inproc` executes the whole recipe in one call, so a scientist waiting on a two-minute
embedding saw a frozen terminal and then everything at once. The step loop now reports each
transition to an optional observer, and the panel opens showing every DECLARED step as queued
before any of them has run.

The observer contract is the load-bearing part: a dashboard is an observer, and an observer
must never be able to take down the thing it observes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


def _install(monkeypatch, **impls):
    from manyruns import pipeline

    for name, fn in impls.items():
        monkeypatch.setitem(pipeline._INPROC_STEPS, name, fn)


def _ok(state, g, params, out_dir, plots, target_dim):
    state["emb"] = np.zeros((state["X"].shape[0], 2))


def _recipe(*names):
    return {"name": "r", "steps": [{"name": n, "group": "latent"} for n in names]}


def test_the_observer_sees_a_step_start_before_it_finishes(tmp_path, monkeypatch):
    """THE point of a live view: you learn which step you are waiting ON, not only which
    ones are done. `running` is a live STATE, deliberately not an outcome — the outcome
    vocabulary has to keep meaning "how it ended"."""
    from manyruns import pipeline

    seen = []

    def observe(rec, steps):
        seen.append((rec["name"], rec.get("state"), rec.get("outcome")))

    _install(monkeypatch, a=_ok, b=_ok)
    pipeline.run_inproc(np.zeros((6, 3)), _recipe("a", "b"), tmp_path, on_step=observe)

    assert seen == [
        ("a", "running", None), ("a", None, "ok"),
        ("b", "running", None), ("b", None, "ok"),
    ]


def test_a_finished_record_never_keeps_a_live_state(tmp_path, monkeypatch):
    """`state` is scaffolding for the wait. If it survived into the stored record, a reader
    could not tell a crashed mid-run step from a finished one."""
    from manyruns import pipeline

    _install(monkeypatch, a=_ok)
    out = pipeline.run_inproc(np.zeros((6, 3)), _recipe("a"), tmp_path, on_step=lambda *_: None)

    assert "state" not in out["steps"][0]


def test_a_step_that_never_dispatches_also_settles(tmp_path):
    """The bug the happy-path test could not see. `state`/`started` were cleared in the
    `finally:` block, which the two `continue` paths — no group declared, and no executor
    for the group — never reach. Those records shipped with `state="running"`, and the
    dashboard rendered them as running FOREVER while their outcome said `skipped`."""
    from manyruns import pipeline

    recipe = {"name": "r", "steps": [
        {"name": "nogroup"},                       # declares no group at all
        {"name": "orphan", "group": "sweep"},      # a group no engine implements
    ]}
    out = pipeline.run_inproc(np.zeros((6, 3)), recipe, tmp_path)

    for rec in out["steps"]:
        assert rec["outcome"] == "skipped"
        assert "state" not in rec, f"{rec['name']} still looks like it is running"
        assert "started" not in rec


def test_an_observer_that_raises_cannot_kill_the_run(tmp_path, monkeypatch):
    """A renderer dies for reasons that have nothing to do with the science — a closed
    terminal, a bad width, a resize. The run is what is being protected here, not the
    display."""
    from manyruns import pipeline

    def hostile(rec, steps):
        raise RuntimeError("the terminal went away")

    _install(monkeypatch, a=_ok)
    out = pipeline.run_inproc(np.zeros((6, 3)), _recipe("a"), tmp_path, on_step=hostile)

    assert out["ok"] is True
    assert out["steps"][0]["outcome"] == "ok"


def test_a_run_without_an_observer_is_unchanged(tmp_path, monkeypatch):
    """The observer is optional and must cost nothing when absent — it is a display seam,
    not a behaviour change."""
    from manyruns import pipeline

    _install(monkeypatch, a=_ok)
    quiet = pipeline.run_inproc(np.zeros((6, 3)), _recipe("a"), tmp_path)
    watched = pipeline.run_inproc(np.zeros((6, 3)), _recipe("a"), tmp_path,
                                    on_step=lambda *_: None)

    # `seconds` varies by definition, and an artifact PATH embeds `run_id`, which is fresh per
    # execution on purpose (`spec_id` is the stable half). Compared by basename rather than
    # dropped, so a regression that stopped writing artifacts entirely still fails here.
    def strip(r):
        out = []
        for s in r["steps"]:
            rec = {k: v for k, v in s.items() if k != "seconds"}
            if "artifacts" in rec:
                rec["artifacts"] = {k: Path(v).name for k, v in rec["artifacts"].items()}
            out.append(rec)
        return out

    assert strip(quiet) == strip(watched)
    assert strip(quiet)[0]["artifacts"] == {"emb": "00-a_emb.npy"}


def test_a_failing_step_is_reported_live_too(tmp_path, monkeypatch):
    """You should see it die when it dies, not at the end."""
    from manyruns import pipeline

    def boom(state, g, params, out_dir, plots, target_dim):
        raise ValueError("kaboom")

    outcomes = []
    _install(monkeypatch, a=_ok, b=boom)
    pipeline.run_inproc(np.zeros((6, 3)), _recipe("a", "b"), tmp_path,
                          on_step=lambda rec, _s: outcomes.append(rec.get("outcome")))

    assert outcomes == [None, "ok", None, "error"]


# ── the panel itself ─────────────────────────────────────────────────────────
def test_the_panel_shows_every_declared_step_before_any_has_run():
    """It opens EMPTY — the whole recipe visible as queued — so the shape of the wait is
    known up front rather than revealed one line at a time."""
    pytest.importorskip("rich")
    from rich.console import Console

    from manyruns import shell

    panel = shell._progress_panel(_recipe("phate", "separation")["steps"], [], "t")
    out = Console(force_terminal=False, width=80)
    with out.capture() as cap:
        out.print(panel)
    text = cap.get()

    assert "phate" in text and "separation" in text
    assert text.count("queued") == 2
    assert "0 of 2 complete" in text


def test_the_panel_marks_the_running_step_and_counts_only_finished_ones():
    pytest.importorskip("rich")
    from rich.console import Console

    from manyruns import shell

    records = [
        {"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.5},
        {"index": 1, "name": "separation", "state": "running", "outcome": None,
         "started": __import__("time").monotonic()},
    ]
    panel = shell._progress_panel(_recipe("phate", "separation")["steps"], records, "t")
    out = Console(force_terminal=False, width=80)
    with out.capture() as cap:
        out.print(panel)
    text = cap.get()

    assert "running" in text
    assert "1.50s" in text
    assert "1 of 2 complete" in text      # the running one is not counted as done


# The figure pane's own tests live in `tests/test_figure_pane.py`, against
# `manyruns/figures.py`, which is where the drawing moved. Two tests here duplicated them
# and monkeypatched `shell._emit_inline_image`, which the pane no longer routes through — so
# they passed by intercepting a function that had stopped doing the work.

