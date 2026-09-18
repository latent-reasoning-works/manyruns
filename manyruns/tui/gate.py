"""The gate: a blocking `read`/`write` pair backed by a screen instead of a terminal.

`tune.run_tune_loop(session, step, read, write)` has always been parameterised on its I/O — the
REPL passes `input`/`print` (`app.py:566`) and nothing else had ever passed it anything. This is
the second caller. The loop keeps that synchronous I/O interface; a bound gate also receives
the question kind and current step as data, so the screen never parses prompt wording.

WHY A QUEUE AND NOT A CALLBACK. `run_tune_loop` is a `while` loop that BLOCKS on an answer — that
is what makes it a decision loop rather than a stream — and Textual's event loop must never block.
`RunScreen` already runs the session on a daemon thread (`tui/run.py:296-325`), so the block
belongs there: the worker calls `read`, waits on a `queue.Queue`, and the UI thread puts an answer
on it from a keypress. No new concurrency model — the thread and the `post_message` marshal are
the ones the run screen already has.

THE GATE IS CONSTRUCTED BY THE CALLER, not by the screen, and given to both. `RunScreen`'s `start`
is built before the screen exists (it closes over a `Session`), so a gate the screen owned could
not be reached from inside it. One object, two holders, no signature change to `start`.

**Every wait is bounded and every wait is cancellable.** A gate that can block forever turns a
quit into a hang, which is the exact failure `run.py`'s daemon-thread note documents at length —
so `close()` wakes every waiter with `Cancelled`, and `run_tune_loop` reads that as `cancel`.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

#: What `read` returns when the gate is closed under a waiting caller. `run_tune_loop` compares
#: the answer against its own verbs and treats anything unrecognised as a re-prompt, so this is
#: spelled as the loop's own cancel word rather than a sentinel it would not understand.
CANCELLED = "cancel"


class Gate:
    """One pending question at a time, asked from a worker thread and answered from the UI.

    Not a general RPC channel: the loop asks exactly one question and waits, so a single-slot
    queue is the whole requirement and a deeper one would only let answers pile up for questions
    nobody has asked yet.
    """

    def __init__(self, on_prompt: Optional[Callable[[str], None]] = None,
                 on_write: Optional[Callable[[str], None]] = None,
                 on_arm: Optional[Callable[[dict], None]] = None,
                 on_moved: Optional[Callable[[dict, dict], None]] = None,
                 on_close: Optional[Callable[[], None]] = None) -> None:
        self._answers: queue.Queue = queue.Queue(maxsize=1)
        # Guards the pending question and its kind. `read` blocks outside it — holding a lock across a blocking
        # `get()` would deadlock the answerer that is meant to release it.
        self._lock = threading.Lock()
        self._on_prompt = on_prompt
        self._on_write = on_write
        # NAMED CHANNELS, not one generic `tell(kind, payload)`. `on_prompt`/`on_write` set
        # the shape and a general registry ahead of a general need is the op-registry mistake
        # CLAUDE.md names; each of these has exactly one sender and one handler.
        self._on_arm = on_arm
        self._on_moved = on_moved
        self._on_close = on_close
        self.question_id = 0
        self.prompt: Optional[str] = None
        self.question_kind: Optional[str] = None
        self.transcript: list[str] = []
        self.closed = False

    def bind(self, on_prompt: Optional[Callable[[str], None]] = None,
             on_write: Optional[Callable[[str], None]] = None,
             on_arm: Optional[Callable[[dict], None]] = None,
             on_moved: Optional[Callable[[dict, dict], None]] = None,
             on_close: Optional[Callable[[], None]] = None) -> "Gate":
        """Attach the screen's callbacks after construction. Returns self, so a caller can wire
        and pass in one expression.

        Needed because the gate is built BEFORE the screen — `RunScreen`'s `start` closes over it,
        and `start` is an argument to the screen's constructor, so the screen cannot have made it.

        Callbacks may fire on either thread. Whatever is attached must be thread-safe;
        `RunScreen` attaches `post_message`, which is safe from any thread and drops silently on a
        closed pump (textual 8.2.8), so a detached run cannot deliver to a screen that is gone.
        """
        if on_prompt is not None:
            self._on_prompt = on_prompt
        if on_write is not None:
            self._on_write = on_write
        if on_arm is not None:
            self._on_arm = on_arm
        if on_moved is not None:
            self._on_moved = on_moved
        if on_close is not None:
            self._on_close = on_close
        return self

    # ── what the worker thread calls ─────────────────────────────────────────
    def read(self, prompt: str = "", *, kind: Optional[str] = None) -> str:
        """Ask, then BLOCK until the UI answers or the gate closes.

        Called on the run worker. Returns `CANCELLED` on a closed gate rather than raising,
        because the caller is `run_tune_loop`, which already has a word for "stop and put things
        back" — raising would take the loop's `finally` (which restores the metric suite) down a
        path it was not written for.
        """
        with self._lock:
            if self.closed:
                return CANCELLED
            self.question_id += 1
            self.prompt = prompt
            self.question_kind = kind
        if self._on_prompt is not None:
            self._on_prompt(prompt)
        answer = self._answers.get()
        # `prompt` is cleared by whoever ANSWERED, not here — see `answer`. Clearing it after
        # unblocking left a window where the question was already answered and still looked
        # pending, so a second keypress queued an answer for a question nobody had asked. Caught
        # by `test_the_gate_drives_the_real_tune_loop_from_a_worker_thread`, which drove three
        # replies through and had its second rejected.
        return CANCELLED if answer is None else answer

    def read_tune(self, prompt: str, step: dict, kind: str) -> str:
        """Publish current-attempt params and the question kind before waiting for an answer.

        `answer()` acknowledges delivery only. Only the loop can acknowledge a valid change,
        so every question republishes its current step, also discarding any rejected draft.
        """
        self.arm(step)
        return self.read(prompt, kind=kind)

    def write(self, text: str = "") -> None:
        """Say something without waiting. Kept in `transcript` so a screen that mounts late, or
        a test with no screen at all, can still read everything the loop said."""
        self.transcript.append(str(text))
        if self._on_write is not None:
            self._on_write(str(text))

    def arm(self, step: "dict | None") -> None:
        """Say which step is about to be tuned. NOT a question — nothing blocks on it.

        It exists because the screen must not read the step out of the PROMPT STRING:
        `on_run_screen_ask` refuses prompt-sniffing in terms ("detecting which is which from the
        prompt STRING"), and the step is a dict, not a sentence. The strip is built from the
        step's params (`params.tunable_for`), so a string could not carry it in any case.

        Named rather than generic — `on_prompt`/`on_write` set that shape, and a generic
        `tell(kind, payload)` channel would be the op-registry mistake in miniature.

        COPIED, not passed by reference: the receiver is a message on another thread's pump and
        `run_tune_loop` rebinds `current` on every retry, so handing the live dict across would
        let the strip read a step the loop has already moved on from.
        """
        if self._on_arm is not None:
            self._on_arm(dict(step or {}))

    def moved(self, before: dict, after: dict) -> None:
        """Report the g-vector either side of a tuned step.

        Copied for `arm`'s reason and then some: `session.g` is mutated in place by every
        executor (`run_tune_loop`'s own `_reset` clears and refills it), so a reference handed
        across would read as whatever the NEXT step made it by the time the handler ran.
        """
        if self._on_moved is not None:
            self._on_moved(dict(before), dict(after))

    # ── what the UI thread calls ─────────────────────────────────────────────
    def is_pending(self, question_id: Optional[int]) -> bool:
        """Check identity as well as liveness; identical prompt text can be asked twice."""
        with self._lock:
            return (not self.closed and self.prompt is not None
                    and question_id == self.question_id)

    def answer(self, text: str, *, question_id: Optional[int] = None) -> bool:
        """Deliver an answer. False when nothing was waiting for one.

        The check is `prompt is not None` rather than the queue's emptiness: a keypress that
        arrives between two questions has nothing to answer, and putting it on the queue would
        silently answer the NEXT question with a keystroke meant for the last one.
        """
        with self._lock:
            if (self.closed or self.prompt is None
                    or (question_id is not None and question_id != self.question_id)):
                return False
            try:
                self._answers.put_nowait(str(text))
            except queue.Full:
                return False
            # CONSUMED HERE, under the same lock that checked it, so the pending question and the
            # answer to it move together. A reader waiting for `prompt is not None` therefore sees
            # the NEXT question, never the answered one.
            self.prompt = None
            self.question_kind = None
            return True

    def close(self) -> None:
        """Wake a waiting worker and refuse every later question.

        Idempotent, and safe to call from either thread — closing twice happens whenever a screen
        is dismissed by a run that has already finished.
        """
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self.prompt = None
            self.question_kind = None
        try:
            self._answers.put_nowait(None)
        except queue.Full:
            pass
        if self._on_close is not None:
            self._on_close()


def resolve(answer: str) -> str:
    """A keypress or a typed word → one of `run_tune_loop`'s verbs.

    The loop already accepts `a`/`y`/`yes`, `r`, `c` alongside the full words (`tune.py:129-140`),
    so this maps the SCREEN's vocabulary onto that and nothing more. It exists so the key bindings
    and the loop cannot drift into two vocabularies for one choice — the failure `vocab.py`'s
    header records twice.
    """
    text = str(answer).strip()
    key = text.lower()
    return {"a": "accept", "y": "accept", "yes": "accept", "accept": "accept",
            "enter": "accept", "t": "tune", "r": "tune", "retry": "tune", "tune": "tune",
            "c": "cancel", "cancel": "cancel", "escape": "cancel"}.get(key, text)
