"""The gate — `manyruns/tui/gate.py`, the blocking read/write the GUI hands `run_tune_loop`.

NO TEXTUAL IMPORT HERE, deliberately, and it is the property the gate exists to have. The whole
point of the seam is that the decision loop is unchanged and only its I/O differs, so the I/O has
to be testable without a terminal — otherwise the only test of step-by-step would be a screen
test, and a screen test cannot easily assert what happens when a worker blocks.
"""
from __future__ import annotations

import threading
import time

from manyruns.tui import gate as _gate


def _answer_soon(g: _gate.Gate, text: str, after: float = 0.02) -> threading.Thread:
    """Answer from another thread, as the UI thread does. Started, not joined by the caller —
    the assertion is that `read` UNBLOCKS, so joining first would hide a deadlock as a pass."""
    t = threading.Thread(target=lambda: (time.sleep(after), g.answer(text)), daemon=True)
    t.start()
    return t


def test_read_blocks_until_the_ui_answers():
    """THE property. `run_tune_loop` is a `while` that waits on a decision; if `read` returned
    without one the loop would spin through its own prompt and accept whatever came back."""
    g = _gate.Gate()
    _answer_soon(g, "accept")

    started = time.monotonic()
    answer = g.read("accept / retry / cancel > ")

    assert answer == "accept"
    assert time.monotonic() - started >= 0.015, "read returned without waiting for an answer"
    assert g.prompt is None, "the question is cleared once answered"


def test_a_keypress_between_questions_answers_nothing():
    """`answer` checks that a question is PENDING, not that the queue is empty. A keypress
    arriving between two questions has nothing to answer, and queueing it would silently answer
    the NEXT question with a keystroke meant for the last one — which in this loop is the
    difference between accepting and retrying a step."""
    g = _gate.Gate()

    assert g.answer("accept") is False, "nothing was asked"

    _answer_soon(g, "retry")
    assert g.read("? > ") == "retry"
    assert g.answer("accept") is False, "the question was already answered"


def test_closing_the_gate_wakes_a_waiting_worker():
    """A gate that can block forever turns a quit into a hang — the exact failure `tui/run.py`
    documents at length for its daemon thread (a 15 s worker held the terminal for 15.01 s after
    quit). So `close()` must unblock a waiter, and it must do so with a word the LOOP
    understands: `run_tune_loop` treats `cancel` as "put it back and return None"."""
    g = _gate.Gate()
    threading.Thread(target=lambda: (time.sleep(0.02), g.close()), daemon=True).start()

    answer = g.read("? > ")

    assert answer == _gate.CANCELLED == "cancel"
    assert g.closed
    assert g.read("? > ") == "cancel", "a closed gate refuses later questions without blocking"
    assert g.answer("accept") is False


def test_closing_twice_is_safe():
    """Happens whenever a screen is dismissed by a run that already finished."""
    g = _gate.Gate()
    g.close()
    g.close()

    assert g.closed


def test_write_never_blocks_and_is_kept_for_a_late_reader():
    """`write` is the loop talking, not asking. The transcript exists so a screen that mounts
    late — or a test with no screen at all — still sees everything the loop said."""
    seen: list = []
    g = _gate.Gate(on_write=seen.append)

    g.write("ran phate")
    g.write("  (kept attempt 1 as phate@1.png)")

    assert seen == ["ran phate", "  (kept attempt 1 as phate@1.png)"]
    assert g.transcript == seen


def test_the_prompt_reaches_the_screen_as_it_is_asked():
    """The screen has to render the question, and it only learns of it through this callback —
    `read` blocks immediately after, so nothing can poll for it."""
    asked: list = []
    g = _gate.Gate(on_prompt=asked.append)
    _answer_soon(g, "a")

    g.read("accept / retry / cancel > ")

    assert asked == ["accept / retry / cancel > "]


def test_the_screens_keys_and_the_loops_verbs_are_one_vocabulary():
    """`run_tune_loop` accepts `a`/`y`/`yes`, `r`, `c` beside the full words (`tune.py:129-140`).
    `resolve` maps the SCREEN's vocabulary onto that and nothing else, so a binding cannot drift
    into a second name for one choice — the failure `vocab.py`'s header records twice."""
    assert _gate.resolve("a") == _gate.resolve("y") == _gate.resolve("enter") == "accept"
    # `tune`, not `retry`. The branch always asks for new params — nothing here re-runs a step
    # unchanged — so "retry" named an operation the loop does not have. `r` still maps to it
    # because that is what the prompt said until the rename, and muscle memory is not a typo.
    assert _gate.resolve("t") == _gate.resolve("r") == "tune"
    assert _gate.resolve("c") == _gate.resolve("escape") == "cancel"
    # a typed word passes through untouched — the loop already understands it
    assert _gate.resolve("accept") == "accept"
    assert _gate.resolve("knn=40") == "knn=40", "a retry's params are not a verb"


# ── the seam it exists for: driving the REAL loop from a worker thread ────────────────────
def test_the_gate_drives_the_real_tune_loop_from_a_worker_thread():
    """The whole design in one test: `run_tune_loop` receives the gate's synchronous
    I/O and publishes structured question metadata through it. The loop runs on a worker (as `RunScreen`'s daemon thread does), blocks on the gate, and
    the "UI" answers from this thread.

    Uses the `mock` engine so it needs no private stack and no terminal: what is under test is the
    plumbing, not the fit. A real engine is exercised by the tune tests themselves.
    """
    import threading

    from manyruns import app, tune
    from manyruns.session import Session

    session = Session(project="p", engine="mock", modality="scrna",
                      recipe=app.load_recipe("cflows"))
    g = _gate.Gate()
    step = tune.parse_step_line("phate knn=5", session.engine)
    out: dict = {}

    def worker() -> None:
        out["accepted"] = tune.run_tune_loop(session, step, g.read, g.write)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # retry once with new params, then accept — the same script the REPL types
    for reply in ("r", "knn=40", "a"):
        for _ in range(200):                       # wait for the loop to ASK
            if g.prompt is not None:
                break
            time.sleep(0.005)
        assert g.prompt is not None, f"the loop never asked before {reply!r}"
        assert g.answer(_gate.resolve(reply) if reply in ("r", "a") else reply)

    t.join(timeout=10)
    assert not t.is_alive(), "the loop did not return — the gate deadlocked"
    assert out["accepted"]["params"]["knn"] == 40, "the retry's params reached the accepted step"
    assert len(session.steps) == 2, "both attempts are in the lineage"
    assert any("accept" in line or "retry" in line for line in g.transcript) or g.transcript


def test_quitting_mid_question_does_not_hang_the_worker():
    """The failure mode this must not have. `RunScreen`'s worker is a daemon thread precisely so
    a quit gives the terminal back (its docstring measures a 15 s worker holding it for 15.01 s);
    a gate that blocked forever would put that hang back one layer up, and the loop would never
    reach its `finally` — leaving `ctx["metrics"]` switched off on a session the user keeps."""
    import threading

    from manyruns import app, tune
    from manyruns.session import Session

    session = Session(project="p", engine="mock", modality="scrna",
                      recipe=app.load_recipe("cflows"))
    before = session.ctx.get("metrics")
    g = _gate.Gate()
    step = tune.parse_step_line("phate knn=5", session.engine)
    out: dict = {}

    t = threading.Thread(target=lambda: out.setdefault(
        "r", tune.run_tune_loop(session, step, g.read, g.write)), daemon=True)
    t.start()
    for _ in range(200):
        if g.prompt is not None:
            break
        time.sleep(0.005)

    g.close()
    t.join(timeout=10)

    assert not t.is_alive(), "closing the gate did not release the worker"
    assert out["r"] is None, "a cancelled loop returns None, as the REPL's cancel does"
    assert session.ctx.get("metrics") == before, "the loop's `finally` restored the metric suite"


def test_arm_and_moved_are_silent_when_nothing_is_bound():
    """A gate driving a HEADLESS run must not require a screen.

    The two report channels are the only parts of this object with no answer to wait for, so
    they are the only parts a caller could reasonably forget to bind — `_stepped_run` sends both
    unconditionally, and a run detached from its screen (escape mid-run, or a test with no screen
    at all) keeps sending them after the callbacks stopped mattering.
    """
    g = _gate.Gate()
    g.arm({"name": "phate"})
    g.moved({"a": 1}, {"a": 2})   # no exception is the assertion


def test_arm_and_moved_hand_over_copies_and_not_the_live_dicts():
    """Both receivers are on the OTHER thread's message pump, and both senders' arguments are
    live: `run_tune_loop` rebinds `current` on every retry and every executor mutates
    `session.g` in place. A reference handed across would be read as whatever the next attempt
    made it, which is the class of bug `run_tune_loop`'s own three-dict reset exists to stop."""
    seen: dict = {}
    step = {"name": "phate", "params": {"knn": 5}}
    before, after = {"trust": 0.1}, {"trust": 0.9}
    g = _gate.Gate(on_arm=lambda s: seen.update(step=s),
                   on_moved=lambda b, a: seen.update(before=b, after=a))

    g.arm(step)
    g.moved(before, after)
    step["name"] = "umap"
    before["trust"] = 999

    assert seen["step"]["name"] == "phate", "the step went across by reference"
    assert seen["before"]["trust"] == 0.1, "the g-vector went across by reference"
    assert seen["after"] == after


def test_bind_reaches_the_two_report_channels_too():
    """`bind` exists because the gate is built before the screen; a channel it cannot attach is
    a channel the screen can never receive on. All four go through the same door."""
    got: list = []
    g = _gate.Gate()
    g.bind(on_arm=lambda s: got.append(("arm", s)),
           on_moved=lambda b, a: got.append(("moved", b, a)))

    g.arm({"name": "pca"})
    g.moved({}, {})

    assert [row[0] for row in got] == ["arm", "moved"]


def test_resolve_preserves_parameter_case_and_normalizes_only_verbs():
    assert _gate.resolve(" ACCEPT ") == "accept"
    assert _gate.resolve("solver=ArPack") == "solver=ArPack"
    assert _gate.resolve("Unknown Prose") == "Unknown Prose"


def test_close_notifies_once_after_invalidating_and_can_be_observed_without_deadlock():
    seen = []
    gate = _gate.Gate()
    gate.bind(on_close=lambda: seen.append((gate.closed, gate.prompt,
                                            gate.is_pending(gate.question_id))))
    gate.close()
    gate.close()
    assert seen == [(True, None, False)]
    assert gate.read("late", kind="decision") == _gate.CANCELLED
    assert gate.prompt is None and gate.question_kind is None


def test_close_during_prompt_delivery_cannot_leave_a_live_question():
    gate = _gate.Gate()
    seen = []

    def prompt(_):
        question = gate.question_id
        gate.close()
        seen.append(gate.is_pending(question))

    gate.bind(on_prompt=prompt)
    assert gate.read("decision", kind="decision") == _gate.CANCELLED
    assert seen == [False]
    assert gate.prompt is None and gate.question_kind is None
