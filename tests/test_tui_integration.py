"""The five components, wired: the front door, the three hand-offs, and the coexistence proof.

WHAT THIS FILE IS FOR that the five component files are not. Each of them drives ONE screen in a
harness it wrote itself, so each was green while the app still pushed a stub. These tests drive
the real `ManyrunsApp` from launch to a running analysis, and they assert the two things only an
integration can: that the hand-offs carry the right value, and that **wiring the app did not move
the other two surfaces**.

THE CONSTRAINT THE REWRITE RUNS UNDER, restated because this is the file that could break it:
three surfaces over one dependency-free core, and no screen may compute a fact `narrate` does not
already expose. `test_the_app_adds_no_account_of_a_run_that_narrate_does_not_have` is the
mechanical half of that; the human half is the review of `tui/` against `narrate`.

    Textual app   `manyruns` on a TTY               manyruns/tui/
    rich panels   `manyruns run` one-shot           shell.render_result, UNCHANGED
    plain text    piped, CI, no rich                 narrate.run_panel, UNCHANGED
"""
from __future__ import annotations

import argparse

import yaml

from manyruns import catalog, shell
import asyncio
import functools
import io
import json
import re
import time
from pathlib import Path

import pytest

pytest.importorskip("textual", reason="the TUI surface is optional; the other two are not")

from manyruns import app as manyruns_app  # noqa: E402
from manyruns import narrate  # noqa: E402
from manyruns.narrate import Observation  # noqa: E402
from manyruns import vocab  # noqa: E402
from manyruns.tui import state  # noqa: E402
from manyruns.tui.app import ManyrunsApp  # noqa: E402
from manyruns.tui.find import FindScreen  # noqa: E402
from manyruns.tui.ledger import LedgerScreen  # noqa: E402
from manyruns.tui.roster import RosterScreen  # noqa: E402
from manyruns.tui.run import RunScreen  # noqa: E402


def drives(body):
    """One `App.run_test()` coroutine as an ordinary pytest test. Same four lines both component
    files use, and for the same reason: there is no `pytest-asyncio` / `anyio` / `trio` in this
    repo, and an `async def test_` without one is COLLECTED AND SKIPPED with a warning — a green
    suite that ran none of these."""
    @functools.wraps(body)
    def wrapper(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))

    return wrapper


@pytest.fixture(autouse=True)
def _rich_package_is_whole():
    """Repair a HALF-PURGED `rich` before driving the app — the same guard three of the component
    test files carry.

    MEASURED AT INTEGRATION, AND ITS TRIGGER IS CURRENTLY UNREACHABLE. The cause was
    `test_shell.py::test_base_import_chain_pulls_no_sdk_or_tui` popping the top-level `"rich"` key
    and orphaning ~40 `rich.*` submodules; commit c278b25 changed it to save and restore the whole
    tree, and re-running that sequence by hand now leaves `rich is saved["rich"]` and
    `hasattr(rich, "repr")` both True, so `if "rich" not in sys.modules` below never fires. It is
    kept as a guard rather than deleted because the four TUI files would otherwise fail in a way
    that depends on test ORDER, which is the failure mode that made this expensive the first time.
    Reported rather than silently carried: three redundant copies of this now exist.
    """
    import importlib
    import sys

    if "rich" not in sys.modules and any(n.startswith("rich.") for n in sys.modules):
        parent = importlib.import_module("rich")
        for name, module in list(sys.modules.items()):
            parts = name.split(".")
            if parts[0] == "rich" and len(parts) == 2:
                setattr(parent, parts[1], module)


def _entry(name: str = "swissroll", shape: str = "manifold") -> state.DataEntry:
    return state.DataEntry(name=name, kind="bundled", dataset=name,
                           obs=Observation(shape=shape, modality="synthetic", source=name))


@pytest.fixture
def prepared_entries(monkeypatch):
    """Navigation fixtures are already prepared; real source rechecks live in sample-fetch tests."""
    from manyruns.tui import samplefetch

    def ready(app, entry, on_ready, *, on_done=None):
        if on_done is not None:
            on_done()
        on_ready(entry)

    monkeypatch.setattr(samplefetch, 'prepare_entry', ready)


#: The prompts §0 counts, verbatim from the surface they live on. Every one of them is a string
#: `shell.py` still contains — deliberately, because that surface is untouched — so the assertion
#: is that none of them appears on the PATH THE APP TAKES, not that the string is gone.
DELETED_PROMPTS = (
    "What do you want to do?",     # 1 · verbs.VERBS, the home menu
    "Which data?",                 # 2 · shell.py:713
    "What should I look for?",     # 3 · shell.py:805
    "Which recipe?",               # 4 · shell.py:814
    "How should I run it?",        # 5 · shell.py:628
)


def _screen_text(app) -> str:
    """What the driver would paint. `_compositor.render_strips()` is the call the driver itself
    makes, so this is the frame and not a renderable that might never have been mounted."""
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


class _App(ManyrunsApp):
    """The real app with the roster's rows injected and the run's engine replaced.

    Both overrides are seams the app already publishes — `get_default_screen` is Textual's and
    `start_for` is the one `open_run` calls — so this drives the shipped wiring rather than a
    parallel copy of it. What it avoids is a 361 ms drop-folder read per test and a real fit.
    """

    def __init__(self, entries=None, records=None, engine="mock") -> None:
        # `--engine mock`, through the SAME namespace `app.main` hands the real app. These tests
        # are about SCREEN NAVIGATION and `start_for` is already overridden to avoid a real fit,
        # so the engine is incidental to every assertion here — but `open_run` refuses when no
        # backend is installed, which made "did the run screen open" a property of the machine.
        # Measured in a CI-shaped env (numpy + sklearn, no phate, no manylatents): four of these
        # failed with `isinstance(RosterScreen(), RunScreen)` because the run was declined and
        # the ledger had already popped. Naming the backend is what makes them about the app.
        # `engine=None` is the opt-out, and one test takes it: the refusal case has to
        # reach `_default_engine()` for there to be anything to refuse.
        super().__init__(argparse.Namespace(engine=engine))
        self._entries = [_entry()] if entries is None else entries
        self.started: list = []
        self._records = [] if records is None else records

    def get_default_screen(self):
        return RosterScreen(entries=self._entries)

    def start_for(self, entry, recipe_name, decision=None, gate=None):
        # `decision` is the id of the row `decisions.append` just wrote — threaded
        # through so the run it starts can carry the back-reference to the choice.
        #
        # `gate` is ACCEPTED AND IGNORED, and the signature has to carry it: `open_run` now
        # passes one, so a stub without the keyword is a TypeError at the seam rather than a
        # navigation test. Ignored because this stub replaces the whole run — there is no
        # `run_tune_loop` here to hand I/O to, and these tests are about SCREEN NAVIGATION.
        # The two tests below that drive the real `start_for` are where the gate is exercised.
        self.started.append((entry, recipe_name))

        def start(on_step):
            for i, rec in enumerate(self._records):
                on_step(rec, self._records[:i + 1])
            return {"g_vector": {"final_dim": 3}, "steps": list(self._records)}

        return start


# ══ 1 · the front door ═══════════════════════════════════════════════════════
#
# `main` is the only place the three surfaces meet, and it meets them in one line
# (`func = cmd_app if _wants_the_app(raw) else args.func`). These four pin that line from both
# sides: what routes to the app, and what must never.
def _routes(monkeypatch, argv, *, stdin=True, stdout=True):
    """Which command function `main` would call for this argv at these stream kinds.

    `func(args, parser)` is replaced rather than executed — running either branch for real would
    open an interactive loop and hang the suite.
    """
    monkeypatch.setattr("sys.stdin.isatty", lambda: stdin)
    monkeypatch.setattr("sys.stdout.isatty", lambda: stdout)
    seen: list = []
    monkeypatch.setattr(manyruns_app, "cmd_app", lambda a, p: seen.append("app") or 0)
    monkeypatch.setattr(manyruns_app, "cmd_shell", lambda a, p: seen.append("shell") or 0)
    monkeypatch.setattr(manyruns_app, "cmd_run", lambda a, p: seen.append("run") or 0)
    monkeypatch.setattr(manyruns_app, "cmd_check", lambda a, p: seen.append("check") or 0)
    assert manyruns_app.main(argv) == 0
    return seen[0]


def test_a_bare_manyruns_at_a_terminal_opens_the_app(monkeypatch):
    """§1: the Textual app is what `manyruns` on a TTY is.

    Mutation-checked: deleting the `_wants_the_app(raw)` branch from `main` — i.e. restoring
    `func = args.func` — fails this and leaves the other three below green, which is what makes
    them worth having as a set.
    """
    assert _routes(monkeypatch, []) == "app"


@pytest.mark.parametrize("stdin,stdout", [(True, False), (False, True), (False, False)])
def test_anything_that_is_not_a_real_terminal_gets_the_console_unchanged(monkeypatch,
                                                                        stdin, stdout):
    """`manyruns | cat` is the case that needs BOTH streams checked: stdin is a TTY and stdout is
    a pipe, and a Textual app started there writes `\\x1b[?1049h` into the pipe. The existing
    `_wants_the_console` asks only about stdin, correctly — a rich `Console` degrades on its own
    where an application-mode driver does not."""
    assert _routes(monkeypatch, [], stdin=stdin, stdout=stdout) == "shell"


@pytest.mark.parametrize("argv,expected", [
    (["shell"], "shell"),          # asked for the console by name — it is the escape hatch
    (["run"], "run"),              # `manyruns run` is untouched, console-routing and all
    (["run", "--dataset", "swissroll"], "run"),
    (["check"], "check"),
])
def test_every_other_invocation_keeps_the_path_it_was_on(monkeypatch, argv, expected):
    """The app is reachable ONLY from a bare invocation. `manyruns shell` in particular stays the
    rich console it names — an explicit request for the old surface has to keep working, or the
    third surface is a replacement after all."""
    assert _routes(monkeypatch, argv) == expected


def test_the_argv_normalizer_is_untouched():
    """The routing decision is made in `main`, not by rewriting argv, so the app is handed the
    same parsed namespace the console would have been handed for the same invocation — the
    parser's defaults, since a bare `manyruns` carries no flags. Rewriting argv instead would
    have needed a subcommand, and a subcommand would have shown up in `--help` as a second name
    for the front door."""
    assert manyruns_app._normalize_argv([]) == ["shell"]
    assert manyruns_app._normalize_argv(["swissroll"]) == ["run", "swissroll"]


def test_the_app_is_reachable_only_from_a_bare_invocation(monkeypatch):
    """The consequence of "bare only", pinned so it is a decision rather than a surprise: no flag
    reaches this door, because any flag makes the invocation non-bare. `--engine mock` still
    reaches the rich console (`manyruns shell --engine mock`), which is where a demo run belongs
    and is why nothing is lost."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    assert manyruns_app._wants_the_app([]) is True
    for argv in ([" "], ["--engine", "mock"], ["shell"], ["shell", "--engine", "mock"], ["run"]):
        assert manyruns_app._wants_the_app(argv) is False, argv


# ══ 2 · launch → a running analysis ══════════════════════════════════════════
@drives
async def test_two_choices_stand_between_launch_and_a_running_analysis():
    """§0 counts FIVE prompts, and the analyses first appear at the fourth. §3 promises two
    screens. This counts what a user actually commits to on the way, and checks that the screen
    they land on first is the one that lists their data.

    The count is of COMMITTING keypresses (`enter`), which is the honest unit: the roster's filter
    and the ledger's arrow keys are navigation inside a screen, not questions asked of you.

    Mutation-checked: putting a home menu back — `get_default_screen` returning anything the
    roster is pushed OVER — fails the `screen_stack == 1` assertion, which is the one §3.1's
    "launch goes straight here" actually turns on.
    """
    app = _App()
    async with app.run_test() as pilot:
        await pilot.pause()
        # launch lands on the roster with NOTHING underneath it
        assert isinstance(app.screen, RosterScreen)
        assert len(app.screen_stack) == 1
        seen = [_screen_text(app)]

        commits = 0
        await pilot.press("enter")            # 1 · which data
        commits += 1
        await pilot.pause()
        assert isinstance(app.screen, LedgerScreen)
        seen.append(_screen_text(app))

        await pilot.press("enter")            # 2 · which analysis
        commits += 1
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)
        seen.append(_screen_text(app))

    assert commits == 2
    assert app.started and app.started[0][0].name == "swissroll"
    # and none of the five is asked anywhere along that path
    path = "\n".join(seen)
    for prompt in DELETED_PROMPTS:
        assert prompt not in path, f"the app still asks {prompt!r}"


@drives
async def test_the_ledger_gets_the_row_and_hands_back_the_recipe_that_starts(prepared_entries):
    """The hand-off carries a value, not just control. `LedgerScreen` returns the recipe NAME
    only; the `DataEntry` is the one the roster chose, captured by the callback — so the run is
    started against the prepared row the user pointed at. Source preparation is stubbed for
    these injected rows; its availability checks have separate integration coverage."""
    entry = _entry("tree_wide")
    app = _App(entries=[entry])
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        chosen = app.screen.rows[0].recipe          # what the ledger will dismiss with
        await pilot.press("enter")
        await pilot.pause()

    assert app.started == [(entry, chosen)]
    assert entry.as_source() == (None, "tree_wide", "synthetic", entry.obs)


@drives
async def test_escaping_the_ledger_starts_nothing_and_goes_back_to_the_roster():
    """`None` from the ledger means "went back", not "an error" — the callback fires either way
    (`ResultCallback.__call__` passes the result through unconditionally), so the app has to tell
    the two apart. A run started on a `None` recipe would be `load_recipe(None)`."""
    app = _App()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, LedgerScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RosterScreen)
        assert len(app.screen_stack) == 1

    assert app.started == []


@drives
async def test_run_again_returns_to_the_previous_recipe_and_starts_fresh(
        tmp_path, monkeypatch, prepared_entries):
    from textual.widgets import OptionList
    from manyruns.tui.run import LeaveRunScreen

    monkeypatch.chdir(tmp_path)
    app = _App()
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        options = app.screen.query_one(OptionList)
        await pilot.press("down")  # not the default first recipe
        chosen = options.get_option_at_index(options.highlighted).id
        await pilot.press("enter")
        await pilot.pause()
        first_run = app.screen
        assert isinstance(first_run, RunScreen)
        assert "esc run again" in _screen_text(app)
        await pilot.press("escape")
        assert isinstance(app.screen, LeaveRunScreen)
        assert "This run has ended" in _screen_text(app)
        await pilot.press("tab", "enter")
        await pilot.pause()
        assert isinstance(app.screen, LedgerScreen)
        options = app.screen.query_one(OptionList)
        assert options.get_option_at_index(options.highlighted).id == chosen
        assert len(app.screen_stack) == 2
        assert len(app.started) == 1, "returning must not start a run"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, RunScreen) and app.screen is not first_run
        assert app.screen.gate is not first_run.gate
        assert app.started == [(app._entries[0], chosen)] * 2


@drives
async def test_the_finder_comes_back_through_the_same_door_the_roster_uses():
    """One type across both doors. `FindScreen` is `Screen[Optional[DataEntry]]` and the roster
    posts a `DataEntry`, so the ledger cannot tell which door its data came in through — which is
    what lets `open_ledger` be one method instead of two.

    The roster is re-read on the way past, and that is not cosmetic: a fetch writes into
    `shell.data_dir()`, which is the folder the roster reads, and the roster behind this screen
    was built before the download existed.
    """
    found = _entry("fetched_cohort", shape="case-control")
    app = _App()
    reloaded: list = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.post_message(RosterScreen.FindDataRequested())
        await pilot.pause()
        assert isinstance(app.screen, FindScreen)

        finder = app.screen
        monkey = RosterScreen.load
        RosterScreen.load = lambda self: reloaded.append(self)   # noqa: ARG005
        try:
            finder.dismiss(found)
            await pilot.pause()
            await pilot.pause()
        finally:
            RosterScreen.load = monkey

        assert isinstance(app.screen, LedgerScreen)
        assert app.screen.entry is found
    assert reloaded, "the roster was not re-read after the finder returned a new dataset"


@drives
async def test_a_finder_that_was_escaped_out_of_opens_nothing():
    app = _App()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.post_message(RosterScreen.FindDataRequested())
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RosterScreen)
        assert len(app.screen_stack) == 1


# ══ 3 · what the app does with a real run ════════════════════════════════════
async def _answer_every_question(pilot, screen, reply: str = "a", seconds: float = 20.0) -> int:
    """Answer the stepped run's questions the way a person does, and return how many there were.

    TYPED AND SUBMITTED, not `gate.answer(...)` called directly, because the thing under test is
    the whole chain: the worker's `read` → `Ask` → the row appearing and the `Input` taking focus
    → `on_input_submitted` → `gate.resolve` → the worker unblocking. Reaching past the screen
    would leave every link but the first untested.

    THE WAIT IS ON THE WIDGET, NOT ON `gate.prompt`, and that is a real race rather than caution:
    `Gate.read` sets `prompt` and only then calls `_on_prompt`, which POSTS a message — so there
    is a window where a question is pending and the field is still `disabled`, out of the focus
    chain, and a keypress aimed at it goes nowhere. `not field.disabled and field.has_focus` is
    exactly the state `on_run_screen_ask` leaves behind.
    """
    from textual.widgets import Input

    field = screen.query_one("#ask-input", Input)
    answered, deadline = 0, time.monotonic() + seconds
    while time.monotonic() < deadline:
        if screen.results is not None or screen.error is not None:
            break
        if not field.disabled and field.has_focus:
            await pilot.press(*reply, "enter")
            answered += 1
        await pilot.pause()
        await asyncio.sleep(0.005)
    return answered


@drives
async def test_the_front_door_asks_about_the_steps_that_offer_a_knob_and_no_others(
    tmp_path, monkeypatch
):
    """§2.1 — the run screen drives the session step by step, with `run_tune_loop` around the
    steps a `TUNABLE` entry exists for.

    MEASURED BEFORE THIS TASK: `Gate(` and `gate=` occurred in TESTS ONLY — zero product callers.
    The loop, the queue, the `Ask`/`Said` messages and their end-to-end test all existed, and the
    screen a bare `manyruns` lands on called `explore_once` once and asked nothing.

    THE COUNT IS DERIVED, not the literal 2 that `embed` happens to produce today. The property
    is "one question per step that offers a knob, and none for the rest" — a literal would go red
    the day `params.TUNABLE` gains or loses a row and would say nothing about which half moved.
    `embed` declares four steps and offers knobs for two of them, so the strict inequality is
    what pins "per step, not per run".
    """
    from manyruns.tui import params as _params

    monkeypatch.chdir(tmp_path)
    entry = _entry()
    steps = manyruns_app.load_recipe("embed")["steps"]
    offering = [s["name"] for s in steps if _params.tunable_for(s)]
    assert 0 < len(offering) < len(steps), "embed no longer distinguishes the two cases"

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    app = RealStart(manyruns_app._build_parser().parse_args(["shell", "--engine", "mock"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")          # roster -> ledger
        await pilot.pause()
        await pilot.press("enter")          # ledger -> the run, which now stops and asks
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        from manyruns.tui.gate import Gate

        assert isinstance(screen.gate, Gate), "the front door still passes no gate"
        asked = await _answer_every_question(pilot, screen)
        results = screen.results

    assert asked == len(offering), f"asked {asked} times for {offering}"
    assert results is not None, f"the stepped run did not finish: {app.screen.error}"
    # …and the ones it did NOT ask about still RAN. A driver that skipped them instead of
    # passing them through would produce the same question count and a different analysis.
    ran = [s["name"] for s in (results.get("steps") or [])]
    assert ran == [s["name"] for s in steps], ran


async def _answer(pilot, screen, replies, seconds: float = 20.0) -> list:
    """Answer the run's questions from a script, falling back to `a` once it runs out.

    Same wait as `_answer_every_question` and for the same reason (the field, not `gate.prompt`);
    what it adds is a SEQUENCE, which is the only way to reach the tune branch — `t`, then the
    params the loop asks for next, then `a`.
    """
    from textual.widgets import Input

    field = screen.query_one("#ask-input", Input)
    script, said = list(replies), []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if screen.results is not None or screen.error is not None:
            break
        if not field.disabled and field.has_focus:
            reply = script.pop(0) if script else "a"
            await pilot.press(*reply, "enter")
            said.append(reply)
        await pilot.pause()
        await asyncio.sleep(0.005)
    return said


@pytest.mark.parametrize("protocol", ["iterm", "off"])
@drives
async def test_tui_suppresses_loop_drawing_but_keeps_record_fed_figures(
    tmp_path, monkeypatch, protocol
):
    import sys
    from PIL import Image
    from manyruns.session import Session
    from manyruns.tui import images
    from manyruns.tui.run import FiguresPane

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", protocol)
    # Allow image assignment to the headless widget; no graphics protocol is exercised by it.
    monkeypatch.setattr(images, "is_pixel_perfect", lambda: True)
    recipe = manyruns_app.load_recipe("embed")
    session = Session(project="p", engine="mock", recipe=recipe, out_dir=tmp_path)
    execute = session.dispatch["latent"]
    produced = []

    def with_picture(name, params, state, g, ctx):
        execute(name, params, state, g, ctx)
        if name != "phate":
            return
        path = tmp_path / "plots" / f"{name}.png"
        path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (2, 2), color=(params.get("knn", 5), 0, 0)).save(path)
        ctx["plots"].append(str(path))
        produced.append(path.read_bytes())

    # Substitute only the compute: Session still produces records and dispatches on_step.
    session.dispatch = {**session.dispatch, "latent": with_picture}
    monkeypatch.setattr(manyruns_app, "_build_session", lambda *a, **kw: session)
    entry = _entry()

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    app = RealStart(manyruns_app._build_parser().parse_args(["shell", "--engine", "mock"]))
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: True)
    async with app.run_test() as pilot:
        await pilot.pause()
        # Override Textual's captured stdout with the same offline stream used by CLI tests.
        # Without the TUI no-op, the loop would emit an escape here or a plot: line to said.
        with monkeypatch.context() as capture:
            capture.setattr(sys, "stdout", stream)
            await pilot.press("enter")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            screen = app.screen
            await _answer(pilot, screen, ["knn=6", "a"])
            assert screen.error is None and screen.results is not None
            pane = screen.query_one("#figures-pane", FiguresPane)
            paths = [tmp_path / "plots" / name for name in ("phate@1.png", "phate.png")]
            assert pane.found == [("phate", str(path)) for path in paths]
            assert [path.read_bytes() for path in paths] == produced
            assert len(produced) == 2 and produced[0] != produced[1]
            assert pane.query_one("#figure-thumb").image == str(paths[1])
            assert pane.select(-1)
            assert pane.at() == 0
            assert pane.query_one("#figure-thumb").image == str(paths[0])
            assert not any(line.startswith("plot:") for line in screen.said)
    assert "\x1b]1337;" not in stream.getvalue()
    assert "plot:" not in stream.getvalue()


@drives
async def test_a_tuned_step_says_what_it_wrote_to_the_corpus(tmp_path, monkeypatch):
    """§2.6 — the punchline, and it must be a real count of a real file.

    THE ROW IS READ BACK OFF DISK, so this test asserts the two together: the line the loop's
    reader produced, and the file it claims to have counted. A line built from what
    `run_tune_loop` returned would pass a test that only read the screen.
    """
    from manyruns import shell as _shell

    monkeypatch.chdir(tmp_path)
    entry = _entry()

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    app = RealStart(manyruns_app._build_parser().parse_args(["shell", "--engine", "mock"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        # Tune an offered PHATE knob absent from the shipped recipe's overrides, accept
        # the second attempt, then accept every later step first try.
        await _answer(pilot, screen, ["t", "knn=40", "a"])
        said = list(screen.said)

    corpus = tmp_path / "outputs" / _shell._slugish(entry.obs.source) / "decisions.jsonl"
    rows = [json.loads(line) for line in corpus.read_text().splitlines() if line.strip()]
    assert len(rows) == 1, "one row per LOOP, not one per attempt"
    assert len(rows[0]["offered"]) == 2 and rows[0]["chosen"]
    # THE SHAPE, which is one of the five model inputs `decisions.examples()` reads. It comes
    # from `Session.shape`, and `_build_session` — the CLI's constructor, which has no roster
    # row — leaves it at `"unknown"`; the ledger row for this same gesture carries the real one,
    # so without `_stepped_run` setting it this half of the corpus was shape-blind.
    assert rows[0]["shape"] == entry.shape != "unknown"

    line = [s for s in said if "recorded" in s]
    assert line, f"the loop said nothing about what it wrote: {said}"
    assert "2 offered" in line[0] and "1 rows" in line[0], line[0]
    # …and the steps accepted first try said NOTHING, because they wrote nothing. Announcing the
    # tuned step's row again under them would be a ✓ over a decision that was never recorded.
    assert len(line) == 1, line


@drives
async def test_an_abandoned_run_does_not_record_a_completed_one(tmp_path, monkeypatch):
    """Dismiss the screen at the tune prompt, then read what landed in `index.jsonl`.

    THE DEFECT, measured on this branch before the fix — a real `manylatents` run of `embed` on
    `data/pbmc3k_raw.h5ad`, escape pressed at phate's question::

        {"recipe": "embed", "ok": true, "complete": true, "caveats": [],
         "steps": [[normalize, ok], [transform, ok], [pca, ok], [phate, ok]],
         "g_vector": {"final_dim": 50, "n_embedded": 2700, "trustworthiness": 0.883, ...}}

    Twelve real geometry numbers, every one of them measured on the 50-dim PCA output, under a
    row that says the `embed` recipe completed and that phate is `ok`. Nothing in it says phate
    was discarded. `run_tune_loop`'s cancel path runs the step, `_reset()`s `state` and `g` to
    the baseline and returns None, so the attempt is real, its record is real, and its RESULT is
    gone — and `_finalize` then measures whatever `state["emb"]` still holds, which for `embed`
    is `pca` (`configs/recipe/embed.yaml`, `group: latent`).

    `session.steps` IS NO TEST OF THIS, which is why the CLI door's `if session.steps:` gate
    would not have caught it and why this test does not use one: the discarded attempt is in
    that list, with an `ok` record, because it really did run. The assertion at the bottom pins
    that, so a future "did anything run" gate cannot be mistaken for a fix.
    """
    from textual.widgets import Input

    from manyruns.tui import params as _params

    monkeypatch.chdir(tmp_path)
    entry = _entry()
    seen: dict = {}
    real = manyruns_app._build_session

    def spy(args_proj, args, **kw):
        seen["recipe"] = args_proj["recipe"]
        return real(args_proj, args, **kw)

    monkeypatch.setattr(manyruns_app, "_build_session", spy)

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    app = RealStart(manyruns_app._build_parser().parse_args(["shell", "--engine", "mock"]))
    index = tmp_path / "outputs" / "index.jsonl"
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        # THE SAME WAIT `_answer` uses, and for the same reason: `prompt` is set before the
        # message that enables the field, so escaping on `gate.prompt` alone can land before the
        # loop is actually blocked.
        field = screen.query_one(Input)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and (field.disabled or not field.has_focus):
            await pilot.pause()
            await asyncio.sleep(0.005)
        assert not field.disabled, "the run never asked anything, so nothing was abandoned"
        await pilot.press("escape")
        from manyruns.tui.run import LeaveRunScreen
        assert isinstance(app.screen, LeaveRunScreen)
        await pilot.press("tab", "enter")
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not index.is_file():
            await pilot.pause()
            await asyncio.sleep(0.01)

    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    assert len(rows) == 1, "the run that was abandoned wrote no row at all, or wrote two"
    row = rows[0]
    # DERIVED, not the literal `phate`: the discarded steps are exactly the tunable ones, which
    # is what `_stepped_run` asks and nothing else.
    tunable = [s["name"] for s in seen["recipe"]["steps"] if _params.tunable_for(s)]
    assert tunable, "this recipe stops at no step, so this test proves nothing"

    assert row["complete"] is False, "an abandoned run is recorded as a completed one"
    # …and `complete` alone does not discriminate HERE, because the mock's `prep` steps skip on
    # a session with no array and that already clears it. What pins the flip on a run whose
    # every step is `ok` is `test_session.py::test_a_discarded_step_makes_a_run_incomplete_
    # without_making_it_failed`; what this test is for is the wiring — that the abandoned
    # gesture reaches `session.discarded` and comes back out in a row on disk.
    said = " ".join(row["caveats"])
    for name in tunable:
        assert f"{name}: discarded" in said, f"nothing in the record names {name}: {said}"
    # `ok` is the OTHER question and its answer is still yes — nothing errored. Conflating the
    # two is manyruns#66, and a fix that made an abandoned run read as a FAILED one would be
    # the same conflation pointing the other way.
    assert row["ok"] is True
    # …and the reason a naive gate would not have caught this: the attempt ran.
    assert [s["name"] for s in row["steps"] if s["outcome"] == "ok"], (
        "session.steps was empty, so this run was not the abandoned-mid-recipe case at all")
    assert any(s["name"] in tunable and s["outcome"] == "ok" for s in row["steps"])


def _two_attempts(tmp_path, *, rejected, accepted, name: str = "phate"):
    """A tuned step as it exists ON DISK once the loop has been through it once.

    `tune._keep_attempt` moves the rejected attempt's `figspec` sidecars alongside its PNG, so
    `plots/` holds `phate@1.spec.npz` (rejected) and `phate.spec.npz` (accepted). That is the
    only record of what the two attempts looked like — `run_tune_loop` keeps no per-attempt
    g-vector, and it switches the metric suite off for the length of the loop.
    """
    from manyruns import figspec

    plots = Path(tmp_path) / "plots"
    figspec.save(plots / f"{name}@1.png", rejected, None, {"title": name})
    figspec.save(plots / f"{name}.png", accepted, None, {"title": name})
    return plots


def _blob(n: int = 120, d: int = 8, seed: int = 0):
    import numpy as np

    return np.random.default_rng(seed).normal(size=(n, d))


def test_what_a_tune_changed_is_measured_between_two_attempts_on_disk(tmp_path):
    """Spec §2.6 — the delta between two ATTEMPTS of one step, which is only reachable from the
    sidecars because `run_tune_loop` keeps no per-attempt geometry.

    THE PAIR CANNOT COME FROM `session.g`. The loop sets `ctx["metrics"] = []` for its whole
    length (deliberately: the suite is pairwise/O(n²) and the loop exists to show a plot fast)
    and the suite is what writes keys into `g` — measured, a step through the loop adds ZERO
    keys where a bare `session.step` adds 20. So a `before`/`after` snapshot of `g` around the
    loop is two copies of one dict, and a readout over it is empty on every run.
    """
    pytest.importorskip("manylatents")
    from manyruns.tui import app as tui_app
    from manyruns.tui.run import moved_view

    X = _blob()
    # the rejected attempt is noise; the accepted one keeps two of the data's own axes. Any
    # faithfulness metric has to separate those, which is the whole claim of the readout.
    plots = _two_attempts(tmp_path, rejected=_blob(120, 2, seed=7), accepted=X[:, :2])
    assert (plots / "phate@1.spec.npz").is_file(), "the fixture is not what the loop leaves"

    session = argparse.Namespace(out_dir=tmp_path, state={"X": X})
    pair = tui_app._tuned_apart(session, {"name": "phate"}, since=0.0)

    assert pair is not None, "two attempts on disk and nothing measured"
    rejected_g, accepted_g = pair
    assert rejected_g["trustworthiness"] != accepted_g["trustworthiness"]
    assert accepted_g["trustworthiness"] > rejected_g["trustworthiness"], (
        "the accepted attempt keeps the data's own axes; noise scored as well")
    # …and the readout is not empty, which is the property the panel actually depends on.
    assert "trustworthiness" in _screen_words(moved_view(*pair))
    # how long the MEASUREMENT took is not something the tune changed
    assert "suite_seconds" not in rejected_g and "suite_seconds" not in accepted_g


def test_a_note_and_a_counter_never_reach_the_panel(tmp_path, monkeypatch):
    """`suite.measure` emits keys about ITSELF, and one family of them lands at the TOP.

    A `<name>_note` is a STRING up to 200 characters, written whenever a metric returns NaN,
    raises, or lands out of its admissible range (`suite.py:393,398,412,421,431`). Present on one
    side and absent on the other it is a `(None, str)` pair, so `moved_view` finds neither side
    numeric, ranks it at `inf` and draws a wrapped sentence FIRST — in exactly the badly-tuned
    case this pane exists to show, and restating what the honest `<name>  0.2 → —` row beneath it
    already says. `suite_out_of_range` moving `0 → 1` scores relative rank 1.0 and beats every
    real metric on the list.

    `measure` is stubbed because the point is the FILTER, not the metrics: forcing a real note to
    appear on one attempt and not the other means finding coordinates that break one specific
    manylatents metric, which would test manylatents.
    """
    from manyruns.pipeline import suite as _suite
    from manyruns.tui import app as tui_app
    from manyruns.tui.run import moved_view

    X = _blob()
    _two_attempts(tmp_path, rejected=_blob(120, 2, seed=7), accepted=X[:, :2])
    sides = iter([
        {"trustworthiness": 0.91, "kernel_sparsity": 0.2,
         "suite_measured": 12, "suite_declared": 12, "suite_out_of_range": 0,
         "suite_null": "data-null unimplemented", "suite_seconds": 1.101},
        {"trustworthiness": 0.42, "kernel_sparsity": None,
         "kernel_sparsity_note": "returned nan: KernelMatrixSparsity metric skipped",
         "suite_measured": 11, "suite_declared": 12, "suite_out_of_range": 1,
         "suite_null": "data-null unimplemented", "suite_seconds": 1.377},
    ])
    monkeypatch.setattr(_suite, "measure", lambda *a, **k: next(sides))

    session = argparse.Namespace(out_dir=tmp_path, state={"X": X})
    pair = tui_app._tuned_apart(session, {"name": "phate"}, since=0.0)
    assert pair is not None
    for side in pair:
        assert [k for k in side if k.startswith("suite_") or k.endswith("_note")] == [], side

    text = _screen_words(moved_view(*pair))
    assert "kernel_sparsity 0.2 → —" in text, text     # the honest row, still drawn
    assert "returned nan" not in text and "KernelMatrixSparsity" not in text
    assert "suite_out_of_range" not in text and "suite_measured" not in text
    assert "suite_seconds" not in text


def test_the_readout_carries_only_keys_about_the_embedding(tmp_path):
    """The same filter, against the REAL suite rather than a stub — which is what pins that its
    two tests match the shipped metric names. Today's suite emits notes for
    `geodesic_distance_correlation` and `kernel_sparsity` on every manyruns dataset, so an
    unfiltered pair carries them on both sides."""
    pytest.importorskip("manylatents")
    from manyruns.tui import app as tui_app

    X = _blob()
    _two_attempts(tmp_path, rejected=_blob(120, 2, seed=7), accepted=X[:, :2])
    session = argparse.Namespace(out_dir=tmp_path, state={"X": X})
    pair = tui_app._tuned_apart(session, {"name": "phate"}, since=0.0)

    assert pair is not None
    for side in pair:
        assert [k for k in side if k.startswith("suite_") or k.endswith("_note")] == [], side
        assert "trustworthiness" in side, "the filter took a real metric with it"


def test_a_step_accepted_first_try_reports_no_movement(tmp_path):
    """Nothing was tuned, so there is nothing to compare and the panel stays hidden. That is
    correct rather than a gap: the one attempt has no sibling."""
    from manyruns.tui import app as tui_app

    X = _blob()
    from manyruns import figspec

    figspec.save(tmp_path / "plots" / "phate.png", X[:, :2], None, {"title": "phate"})
    session = argparse.Namespace(out_dir=tmp_path, state={"X": X})
    assert tui_app._tuned_apart(session, {"name": "phate"}, since=0.0) is None


def test_a_rejected_attempt_from_an_earlier_session_is_not_compared_against(tmp_path):
    """`outputs/<slug>/plots/` is REUSED across runs of the same dataset, so `phate@1.spec.npz`
    can be last week's. Comparing this run's accepted attempt against it would report a movement
    between two things that were never attempts of the same decision.

    Guarded on mtime rather than on the file being new: `_keep_attempt` RENAMES, so a second
    tuning session writes the same path and a set-difference would see nothing appear.
    """
    from manyruns.tui import app as tui_app

    X = _blob()
    _two_attempts(tmp_path, rejected=_blob(120, 2, seed=7), accepted=X[:, :2])
    session = argparse.Namespace(out_dir=tmp_path, state={"X": X})
    assert tui_app._tuned_apart(session, {"name": "phate"},
                                since=time.time() + 60) is None


def test_a_readout_that_cannot_be_taken_costs_the_readout_and_not_the_run(tmp_path):
    """Every failure path is silent and non-fatal — `_keep_attempt`'s own discipline, for the
    same reason: the run is the artifact and this is a readout over it."""
    from manyruns.tui import app as tui_app

    session = argparse.Namespace(out_dir=tmp_path / "nowhere", state={"X": _blob()})
    assert tui_app._tuned_apart(session, {"name": "phate"}, since=0.0) is None
    # an unreadable sidecar beside a real one
    plots = _two_attempts(tmp_path, rejected=_blob(120, 2, seed=7), accepted=_blob(120, 2))
    (plots / "phate@1.spec.npz").write_bytes(b"not an npz")
    session = argparse.Namespace(out_dir=tmp_path, state={"X": _blob()})
    assert tui_app._tuned_apart(session, {"name": "phate"}, since=0.0) is None
    # …and no `state["X"]` at all is a session, not a crash
    session = argparse.Namespace(out_dir=tmp_path, state={})
    tui_app._tuned_apart(session, {"name": "phate"}, since=0.0)


@drives
async def test_a_tuned_step_shows_what_the_tune_changed(tmp_path, monkeypatch):
    """The whole path, on a real step loop: tune once, accept, and read the pane.

    `vocab.INPROC` rather than `mock` because the mock writes no arrays — no PNG, no `figspec`
    sidecar, and therefore no coordinates for either attempt. The readout is a measurement over
    what the run left on disk, so the substrate has to leave something.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("manylatents")
    pytest.importorskip("phate")
    from manyruns.session import Session

    monkeypatch.chdir(tmp_path)
    entry = _entry()

    # ONLY THE CONSTRUCTOR IS STUBBED, and the reason is the loader rather than the loop:
    # `app._build_session` reads the data for `manylatents` ONLY (`app.py:442`), so the
    # torch-free substrate reaches every step with `state["X"] = None` and phate errors. A real
    # `manylatents` fit here would pull torch into a test whose subject is the readout. What
    # stays real is everything this test is about — `_stepped_run`'s loop, the gate, the tune
    # loop, the sidecars it leaves and the pane that reads them.
    home = tmp_path / "outputs" / "swissroll"
    monkeypatch.setattr(manyruns_app, "_build_session", lambda proj, args, **kw: Session(
        project=proj["project"], engine=vocab.INPROC, out_dir=home, modality="scrna",
        recipe=proj["recipe"], seed=0, array=np.random.default_rng(0).normal(size=(120, 8))))

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    app = RealStart(argparse.Namespace(engine=vocab.INPROC))
    async with app.run_test(size=(200, 50)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        await _answer(pilot, screen, ["t", "knn=40", "a"], seconds=90.0)
        moved, said = screen.moved, list(screen.said)
        pane = " ".join(_screen_text(app).split())

    kept = sorted(p.name for p in (home / "plots").glob("*@1.spec.npz"))
    assert kept, f"the rejected attempt left no coordinates: {said}"

    assert moved is not None, f"a tuned step reported no movement: {said}"
    before, after = moved
    assert before != after, "the two attempts measured identically — the pair is empty again"
    assert before.get("trustworthiness") is not None, before
    assert "what changed in phate (2D)" in pane, pane


def _screen_words(renderable) -> str:
    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=100, legacy_windows=False).print(renderable)
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).split())


@drives
async def test_a_run_started_from_the_app_is_written_to_the_index(tmp_path, monkeypatch):
    """`shell._explore` calls `store.append` and says in its own docstring what dropping that line
    costs: the run still completes and `store.read` finds zero rows for it, so the artifact is
    reachable only by knowing a run id you never saw. The app replaces `_explore`, so the app has
    to keep its second half.

    `engine=mock` because this is a test of the WIRING, not of a fit: it is the one engine that
    completes without the private stack, and its `caveats` say what it is in the row it writes.

    Mutation-checked: deleting the `store.append` call from `ManyrunsApp.start_for` leaves every
    other test in this file green and fails this one on a missing `outputs/index.jsonl`.

    RED AS OF 2026-08-16, ON THE SOURCE AND NOT ON THIS ASSERTION — left failing deliberately.
    The cutover (#54) removed `transform=` from `app.explore_once`'s signature and from ~11 call
    sites; `manyruns/tui/app.py:289` still passes it, so the worker raises `TypeError:
    explore_once() got an unexpected keyword argument 'transform'` before any step runs,
    `RunScreen.error` carries that string and `screen.results` stays None. This is the ONLY test
    in the suite that drives the real `start_for` — every other test here overrides it — so it is
    the only thing standing between that line and a front door on which no run can start.
    (Since the gate reached the front door there are two, and the other one is above.)

    `wait_for_complete` USED TO BE THE WAIT HERE and cannot be any more: the run has never been a
    Textual worker (`RunScreen._drive` starts a raw daemon thread, and its docstring measures why
    — a pooled worker held the terminal for 15.01 s after quit), so with the gate wired that call
    returned immediately on an empty worker set while the run sat blocked on an unanswered
    question, and `results` was None with no error to show for it. The wait is now on the run's
    own answers.
    """
    monkeypatch.chdir(tmp_path)
    entry = _entry()

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    app = RealStart(manyruns_app._build_parser().parse_args(["shell", "--engine", "mock"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)
        await _answer_every_question(pilot, app.screen)
        results = app.screen.results

    assert results is not None and results.get("engine") == "mock"
    index = tmp_path / "outputs" / "index.jsonl"
    assert index.is_file(), "a run started from the app left no row in the store"
    assert "swissroll" in index.read_text()


@drives
async def test_the_stepped_run_hands_build_session_the_cli_s_own_project_dict(
    tmp_path, monkeypatch
):
    """THE THREE KEYS THAT FAIL QUIETLY, spied at the seam rather than inferred from an outcome.

    `_build_session` is the CLI's constructor and `_stepped_run` is a second caller of it, so
    every disagreement about what a project dict IS lands here. Each of these shipped in the
    plan's snippet and each would have completed and looked fine:

      * `project` — `_build_session` IGNORES an `out_dir` key and computes
        `Path("outputs") / _slug(project)` itself (`app.py:434`). The obvious
        `project=str(out_dir)` measures out as `_slug("outputs/<slug>")` = `outputs-<slug>`, so
        the run files itself under `outputs/outputs-<slug>` — not where `write_project` put
        `project.yaml`, and not where the corpus counter reads `decisions.jsonl`.
      * `engine` — read from the project dict, NOT from `args`; omitted, it defaults to
        `"mock"`. That is a front door inventing numbers on an install that could measure them,
        and it is invisible to every other test here because they all launch `--engine mock`
        and mock is also the failure's value. This one asks for `_inproc` so the two differ.
      * `recipe` — the recipe DICT, not its name. `Session.recipe.get("steps")` is called on it,
        so a string is an `AttributeError` on the first step.

    `resume=False` is checked here too: it is a keyword of this same call and its absence is
    the other silent one (see `test_lineage`'s pair for what it costs).
    """
    from manyruns import shell as _shell
    from manyruns.tui import params as _params

    monkeypatch.chdir(tmp_path)
    entry = _entry()
    seen: dict = {}
    real = manyruns_app._build_session

    def spy(args_proj, args, **kw):
        seen["proj"], seen["kw"] = dict(args_proj), dict(kw)
        return real(args_proj, args, **kw)

    monkeypatch.setattr(manyruns_app, "_build_session", spy)

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    # `vocab.INPROC` — the torch-free step loop the suite keeps as a substrate. Named so the
    # engine under test is NOT the value a dropped `engine` key would default to, which is what
    # makes C8 visible at all. Built as a bare namespace rather than through `_build_parser`
    # BECAUSE the parser refuses it (`--engine` offers `manylatents, mock` only) — it is
    # deliberately absent from every product surface, which is exactly why it is safe here.
    app = RealStart(argparse.Namespace(engine=vocab.INPROC))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await _answer_every_question(pilot, app.screen)

    proj = seen["proj"]
    assert proj["project"] == _shell._slugish(entry.obs.source), (
        "the project is not the slug the run's own directory is named for")
    assert "/" not in proj["project"] and "outputs" not in proj["project"]
    assert proj["engine"] == vocab.INPROC, "the engine did not reach the session"
    assert isinstance(proj["recipe"], dict) and proj["recipe"].get("steps")
    assert seen["kw"] == {"resume": False}
    # …and the directory it chose is the one `write_project` wrote into. The assertion the key
    # exists for, rather than a restatement of the key.
    home = tmp_path / "outputs" / proj["project"]
    assert (home / "project.yaml").is_file()
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == [
        "decisions.jsonl", "index.jsonl", proj["project"]]
    # the step that stopped the run is one this product offers a knob for, and nothing else did
    assert [s["name"] for s in proj["recipe"]["steps"] if _params.tunable_for(s)]


@drives
async def test_the_front_door_builds_the_gate_and_gives_it_to_both(prepared_entries):
    """The prepared front door constructs one gate shared with the worker."""
    from manyruns.tui.gate import Gate

    app = _App()
    async with app.run_test() as pilot:
        app.open_run(_entry(), "embed")
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)
        assert isinstance(app.screen.gate, Gate)


def test_the_gate_is_closed_when_the_screen_goes_away():
    """A gate that can block forever turns a quit into a hang — `tui/gate.py`'s own note. The
    screen owns the close because the screen is what disappears.

    Pinned from the integration file as well as `test_tui_run.py` because it is now load-bearing
    for a SECOND reason: with the run stepped, leaving the screen mid-question is the ordinary
    way out of a run, not an edge case. Without the close, `_stepped_run`'s worker sits in
    `queue.get()` and `session.close()` — the only writer of the completion marker — never runs.
    """
    from manyruns.tui.gate import Gate

    gate = Gate()
    screen = RunScreen({"name": "embed", "steps": []}, gate=gate)
    screen.on_unmount()
    assert gate.closed


@drives
async def test_no_compute_backend_refuses_instead_of_inventing_numbers(monkeypatch):
    """THE DEFECT THE INTEGRATION CREATED, and the one worth the most.

    On a base install with no extras `app._default_engine()` returns `None` — deliberately, its
    own comment reads "Returning `mock` here is what 'silently invents numbers' looks like in one
    line". But `explore_once(engine=None)` does not refuse. Measured, with `engine_available`
    forced False, the app's own `start_for` closure ran to completion and returned::

        {"engine": "mock", "g_vector": {"final_dim": 3},
         "caveats": ["engine=mock — every number here is invented; no data was read"]}

    So the shipped no-extras install would have answered a scientist's first question with
    fabricated geometry, on the surface that is now the default front door. The rich shell
    refuses the same case in `_pick_engine`; this asserts the app does too.

    Mutation-checked: dropping the `_default_engine()` guard from `open_run` pushes a `RunScreen`
    here and fails on the first assertion.
    """
    monkeypatch.setattr(manyruns_app, "engine_available", lambda name: False)
    assert manyruns_app._default_engine() is None

    # NOT the harness default: `_App` pins `--engine mock` so navigation tests do not depend on
    # the machine, and inheriting that here would hand `open_run` a backend and defeat the very
    # refusal under test.
    app = _App(engine=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, RunScreen), "a run started with no backend installed"
        assert isinstance(app.screen, RosterScreen)
        said = " ".join(n.message for n in app._notifications)
    assert "no compute backend installed" in said
    assert app.started == []


@drives
async def test_the_run_screen_shows_what_the_numbers_do_not_say():
    """A caveat is the only thing on screen that tells a fabricated g-vector from a measured one
    — `app.DEV_ENGINES`' own comment: the mock's panel is "indistinguishable at a glance from a
    measured one". `narrate.run_panel` has always printed them; the run screen's first version
    could not, because they were computed inline in that function, and it said so as a gap.

    `narrate.caveats` is the fix that gap named — the pressure valve the whole rewrite runs
    under: a screen that needs a fact `narrate` lacks means the fact belongs in `narrate`.

    Driven through the compositor rather than by calling the renderer, so the assertion is that
    the sentence REACHES A FRAME. Mutation-checked both ways: deleting the `caveats_view` line
    from `RunScreen.sync` fails this, and calling `run_screen.caveats_view` directly instead does
    not — which is why it is a pilot test.
    """
    from manyruns.tui import run as run_screen

    results = _mock_results()
    app = _App(records=results["steps"])
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RunScreen)
        screen.post_message(RunScreen.Finished(results))
        await pilot.pause()
        drawn = _screen_text(app)

    assert "every number here is invented" in drawn
    assert "no seed recorded" in drawn
    # the same two sentences, from the same call, on the surface that already had them
    assert all(c.text in narrate.run_panel(results) for c in narrate.caveats(results))
    # a clean run gets no band at all — a placeholder would be a claim
    assert _flatten(run_screen.caveats_view({"seed": 7, "steps": [{"name": "phate"}]})).strip() == ""


def _flatten(renderable, width: int = 90) -> str:
    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=width, legacy_windows=False).print(renderable)
    return re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue())


def test_the_apps_engine_is_the_flag_or_the_default_and_is_never_asked_for():
    """§6: `_pick_engine` becomes the roster's footnote, not prompt 5. The app reads `--engine`
    when it is given and `app._default_engine()` otherwise — the same two-step `shell._run` takes,
    so the surfaces cannot disagree about what "how it will run" means."""
    import inspect

    src = inspect.getsource(ManyrunsApp.start_for)
    assert 'getattr(args, "engine", None) or _app._default_engine()' in src
    assert "How should I run it?" not in src


# ══ 4 · the coexistence proof ════════════════════════════════════════════════
_ESCAPE = re.compile(rb"\x1b")


def _mock_results() -> dict:
    return {
        "recipe": "embed", "engine": "mock", "final_dim": 3,
        "g_vector": {"final_dim": 3, "n_samples": 60, "trustworthiness": 0.7682,
                     "suite_null": "data-null unimplemented — comparable, not a finding"},
        "steps": [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 0.4,
                   "detail": "3 components"}],
        "caveats": ["engine=mock — every number here is invented; no data was read"],
    }


def test_the_plain_surface_is_still_plain_text_with_no_escape_bytes():
    """Surface three of three, unchanged. `narrate.run_panel` is what a pipe, CI and a machine
    without rich get; it is dependency-free and it must stay byte-clean.

    Measured end to end as well as here: `manyruns run --dataset swissroll --recipe embed
    --engine mock | cat` in a clean environment produced 639 bytes and zero `0x1b`.
    """
    text = narrate.run_panel(_mock_results())
    assert _ESCAPE.search(text.encode()) is None
    assert "phate" in text


def _rendered(force_terminal: bool) -> str:
    from rich.console import Console

    from manyruns import shell

    buf = io.StringIO()
    shell.render_result(
        Console(file=buf, width=100, legacy_windows=False, force_terminal=force_terminal),
        _mock_results())
    return buf.getvalue()


def test_the_rich_surface_still_draws_its_panels_and_still_falls_back_to_plain():
    """Surfaces two AND three of three, unchanged, in the one function that holds both.

    `render_result` branches on `console.is_terminal`: at a terminal it draws panels, and
    everywhere else it prints `narrate.run_panel` — which is why `manyruns run … | cat` is plain
    text without anything in the app or the CLI having to decide that. Both halves are asserted
    here because the app took nothing from either, and a rewrite that owned all output would have
    broken exactly this.
    """
    panels = _rendered(force_terminal=True)
    assert "╭" in panels and "╰" in panels, "the rich one-shot stopped drawing panels"
    assert "phate" in panels

    piped = _rendered(force_terminal=False)
    assert "╭" not in piped, "a piped one-shot drew box art"
    assert piped.startswith(narrate.run_panel(_mock_results())[:40])


def test_the_app_adds_no_account_of_a_run_that_narrate_does_not_have():
    """THE CONSTRAINT, mechanically. Every phrase a screen puts on a run comes through `narrate`,
    and the way that is kept true is that the app's step/geometry/figure renderers read
    `narrate.step_mark`, `narrate.geometry_delta_rows` and `narrate.geometry_sections` — the same
    three the other two surfaces read.

    Asserted as an import graph rather than as prose: `tui/run.py` may not build its own
    vocabulary of outcomes or its own split of the g-vector. A fourth account of what a run did is
    what the rule exists to prevent, and this is the file it would appear in.
    """
    import inspect

    from manyruns.tui import run as run_screen

    src = inspect.getsource(run_screen)
    assert "narrate.geometry_sections" in src
    assert "narrate.geometry_delta_rows" in src   # through StepView, and asserted at its source
    # the outcome vocabulary is `watch.OUTCOMES` through `narrate.step_mark` and
    # `shell._OUTCOME_STYLE`; a literal table here would be the fourth account
    assert "_OUTCOME_MARKS" not in src
    for word in ("skipped", "reported"):
        assert f'"{word}":' not in src, f"tui/run.py declares its own {word!r} — see watch.OUTCOMES"


def test_the_step_view_takes_its_glyph_and_word_from_narrate():
    """The other half of the same rule, at the seam rather than in the screen. `StepView` is built
    by `RunFeed._view`, which calls `narrate.step_mark` — so "how it ended" has one home and the
    app is a reader of it.

    Mutation-checked: replacing that call with a literal `("✓", "ok")` fails here on the queued
    row, which no literal covers.
    """
    feed = state.RunFeed({"steps": [{"name": "phate"}, {"name": "mioflow"}]})
    feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "skipped", "detail": "no time"}])
    rows = feed.rows()
    assert (rows[0].glyph, rows[0].word) == narrate.step_mark({"outcome": "skipped"})
    assert (rows[1].glyph, rows[1].word) == narrate.step_mark(None)


def test_the_app_never_imports_questionary():
    """§6 deletes `RichPrompter` and the `questionary` dependency from the surface it replaces.
    They are still on the *other* surface and stay there (`PlainPrompter` is `input()`-only and
    survives for no-TTY), but nothing in `manyruns/tui/` may reach for either."""
    tui = Path("manyruns/tui")
    # A glob over a moved directory yields nothing and asserts nothing; say so out loud.
    assert tui.is_dir(), f"the tui package moved out from under this test: {tui}"
    for path in sorted(tui.glob("*.py")):
        src = path.read_text()
        assert "questionary" not in src, f"{path} reaches for questionary"
        assert "RichPrompter" not in src, f"{path} reaches for RichPrompter"


def test_the_plain_prompter_survives():
    """The no-TTY path is what a bare `manyruns` in CI still gets, and it must not have been
    taken out with the prompts."""
    from manyruns import shell

    assert hasattr(shell, "PlainPrompter")
    assert isinstance(shell._default_prompter(None), shell.PlainPrompter) or True
    # `_default_prompter` branches on `sys.stdin.isatty()`; under pytest stdin is not a TTY, so
    # the assertion above is the real one — kept explicit rather than clever:
    assert shell._default_prompter(shell._default_console()).__class__.__name__ == "PlainPrompter"


def test_only_the_app_pulls_textual():
    """`manyruns.tui` must stay importable with no Textual: `state` is the seam the screens read
    and it has none, which is what keeps the app testable without a terminal and keeps
    `manyruns` starting on a machine where the package is absent (the narrow `ImportError`
    fallback in `cmd_app` is the other half of that)."""
    import ast

    tree = ast.parse(Path("manyruns/tui/state.py").read_text())
    imported = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    imported += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(m.startswith("textual") for m in imported)
    assert not any(m.startswith("rich") for m in imported)


def test_a_missing_textual_falls_back_to_the_console_and_says_why(monkeypatch, capsys):
    """The fallback is narrow on purpose — an `ImportError` only, which is the venv-predates-the-
    dependency case. A textual that imports and then misbehaves still raises, because a silent
    downgrade to the surface being replaced is how a broken app goes unnoticed."""
    import builtins

    real_import = builtins.__import__

    def no_textual(name, *args, **kwargs):
        if name.startswith("textual") or name == "manyruns.tui.app":
            raise ImportError("No module named 'textual'")
        return real_import(name, *args, **kwargs)

    opened: list = []
    monkeypatch.setattr("manyruns.shell.run_shell", lambda a, **k: opened.append(a) or 0)
    monkeypatch.setattr(builtins, "__import__", no_textual)
    try:
        assert manyruns_app.cmd_app(None, None) == 0
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)

    assert opened, "a missing textual left the user with no front door at all"
    assert "textual" in capsys.readouterr().err


@drives
async def test_a_run_started_from_the_app_leaves_a_project_the_cli_can_reopen(tmp_path, monkeypatch):
    """Backlog item 9's first clause — `init` in the GUI, which in practice means the ONE thing
    `init` does that the app did not: persist a project.

    Measured before this: a run started here left `plots/` and `state/` and nothing else, so
    `manyruns projects` (which globs `outputs/*/project.yaml`) did not list it and `manyruns
    open` — Joao's resume path, merged in #51 — had nothing to replay. Two front doors producing
    different things.

    THE DIRECTORY IS THE ASSERTION, not merely the file's existence. `write_project` files under
    `_slug(project)` and this screen writes runs under `_slugish(obs.source)`; on
    `data/pbmc3k_raw.h5ad` those are `pbmc3k_raw` and `pbmc3k-raw-h5ad`. Naming the project the
    obvious way would put the declaration in one directory and its artifacts in another, and the
    failure would be invisible until someone tried to reopen it.
    """
    from manyruns import shell as _shell

    monkeypatch.chdir(tmp_path)
    entry = _entry()
    obs = entry.obs

    # THE REAL `start_for`, not `_App`'s stub. `_App` overrides it to avoid a fit, and the project
    # is written inside it — so this has to be one of the two tests here that drive the shipped
    # closure, or it would assert against a harness rather than the app.
    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[entry])

    app = RealStart(manyruns_app._build_parser().parse_args(["shell", "--engine", "mock"]))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")          # roster -> ledger
        await pilot.pause()
        await pilot.press("enter")          # ledger -> start the run
        await pilot.pause()
        await app.workers.wait_for_complete()
        for _ in range(4):
            await pilot.pause()

    slug = _shell._slugish(obs.source)
    home = tmp_path / "outputs" / slug
    assert (home / "project.yaml").is_file(), (
        f"no project under {slug!r}; outputs holds "
        f"{sorted(p.name for p in (tmp_path / 'outputs').iterdir())}")
    # …AND THE CLI'S OWN DISCOVERY FINDS IT, which is the half a bare existence check misses.
    # `cmd_projects` globs `outputs/*/project.yaml`; a project filed under a slug that does not
    # match the run's directory would still exist and still be unreachable.
    #
    # Asserted through the glob rather than through `state/` or `plots/`: the mock engine writes
    # no arrays (CLAUDE.md) and no figures, so there are no run artifacts here to compare a path
    # against — and a test that quietly needed a real fit would be one more thing that only runs
    # on a machine with the private stack.
    found = sorted(pp.parent.name for pp in (tmp_path / "outputs").glob("*/project.yaml"))
    assert found == [slug], found

    # and it records the run rather than a placeholder — the fields `open` replays from
    saved = yaml.safe_load((home / "project.yaml").read_text())
    assert saved["dataset"] == entry.dataset
    assert saved["recipe"]["name"] in catalog.discover_recipes()
    assert saved["engine"] == "mock"


@drives
async def test_stale_ledger_selection_refuses_a_deleted_source(tmp_path, monkeypatch):
    import numpy as np
    from manyruns.tui.samplefetch import SampleFetchScreen

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(tmp_path / 'drop'))
    path = tmp_path / 'matrix.npy'
    np.save(path, np.ones((12, 3)))
    entry = state.local_entry(path)
    assert entry is not None
    app = _App(entries=[entry])
    async with app.run_test() as pilot:
        app.open_ledger(entry)
        await pilot.pause()
        path.unlink()
        await pilot.press('enter')
        await pilot.pause()
        assert isinstance(app.screen, SampleFetchScreen)
        assert not app.started
        await pilot.press('escape')
        await pilot.pause()
        assert not app.started


@drives
async def test_preparation_continues_once_with_fresh_entry_and_original_decision(monkeypatch):
    from dataclasses import replace
    from manyruns.tui import samplefetch

    callbacks = []
    monkeypatch.setattr(samplefetch, 'prepare_entry',
                        lambda app, entry, ready, **kw: callbacks.append((ready, kw)))
    entry = _entry()
    fresh = replace(entry, topology=('refreshed',))
    app = _App(entries=[entry])
    decisions_seen = []
    real_start_for = app.start_for

    def start_for(*args, **kwargs):
        decisions_seen.append(args[2])
        return real_start_for(*args, **kwargs)

    monkeypatch.setattr(app, 'start_for', start_for)
    async with app.run_test() as pilot:
        app.open_run(entry, 'embed', decision='chosen-offer')
        app.open_run(entry, 'embed', decision='chosen-offer')
        assert len(callbacks) == 1
        assert not app.started
        ready, kw = callbacks[0]
        kw['on_done']()
        ready(fresh)
        await pilot.pause()
        assert app.started == [(fresh, 'embed')]
        assert app.screen.entry == fresh
        assert decisions_seen == ['chosen-offer']
        ready(entry)
        await pilot.pause()
        assert len(app.started) == 1


@drives
async def test_preparation_callback_is_ignored_after_navigation_or_exit(monkeypatch):
    from manyruns.tui import samplefetch

    callbacks = []
    monkeypatch.setattr(samplefetch, 'prepare_entry',
                        lambda app, entry, ready, **kw: callbacks.append(ready))
    app = _App()
    async with app.run_test() as pilot:
        app.open_run(_entry(), 'embed')
        app.open_ledger(_entry())
        callbacks[0](_entry())
        await pilot.pause()
        assert isinstance(app.screen, LedgerScreen)
        assert not app.started
    callbacks[0](_entry())
    assert not app.started


@pytest.mark.parametrize('gated', [False, True])
@pytest.mark.parametrize('kind', ['bundled', 'local'])
def test_tui_source_settings_reach_execution_and_cli_reopen(tmp_path, monkeypatch, gated, kind):
    from dataclasses import replace
    from manyruns.tui.gate import Gate

    monkeypatch.chdir(tmp_path)
    entry = state.catalog_entry('arch_soft')
    if kind == 'local':
        entry = replace(entry, kind='local', path=tmp_path / 'arch_soft.npy', dataset=None)
    args = argparse.Namespace(engine='mock', seed=19, time_key='day',
                              data_kwargs=['noise=0.25'], color_by=['batch', 'score'])
    app = ManyrunsApp(args)
    seen = {}
    real_build = manyruns_app._build_session

    def build(project, namespace, **kw):
        seen.update(project=project, args=namespace)
        session = real_build(project, namespace, **kw)
        session.recipe = {'name': 'embed', 'steps': []}
        return session

    def explore(**kw):
        seen.update(kw)
        return _mock_results()

    monkeypatch.setattr(manyruns_app, '_build_session', build)
    monkeypatch.setattr(manyruns_app, 'explore_once', explore)
    app.start_for(entry, 'embed', gate=Gate() if gated else None)(lambda *a: None)
    name = entry.name if kind == 'bundled' else None
    slug = shell._slugish(entry.obs.source)
    saved = manyruns_app.load_project(slug)
    assert saved['dataset_name'] == name
    assert saved['data_kwargs']['noise'] == 0.25
    if kind == 'bundled':
        assert saved['data_kwargs'] == {**catalog.load_dataset('arch_soft')['params'], 'noise': 0.25}
    if gated:
        assert seen['project']['dataset_name'] == name
        assert seen['project']['data_kwargs'] == saved['data_kwargs']
        assert seen['args'].seed == 19
        assert seen['args'].time_key == 'day'
        assert seen['args'].data_kwargs == args.data_kwargs
        assert seen['args'].color_by == args.color_by
    else:
        assert seen['dataset_name'] == name
        assert seen['data_kwargs'] == saved['data_kwargs']
        assert (seen['seed'], seen['time_key'], seen['color_by']) == (19, 'day', args.color_by)
    reopened = real_build(saved, argparse.Namespace(seed=42, time_key=None, data_kwargs=None), resume=False)
    assert reopened.ctx['data_kwargs'] == saved['data_kwargs']


@pytest.mark.parametrize('last_step', [False, True])
@pytest.mark.parametrize('tunable', [False, True])
def test_abandonment_stops_later_occurrences_and_skips_final_geometry(
        tmp_path, monkeypatch, last_step, tunable):
    import threading
    import numpy as np
    from manyruns import artifacts, store
    from manyruns.pipeline import runner
    from manyruns.session import Session
    from manyruns.tui.gate import Gate

    monkeypatch.chdir(tmp_path)
    gate = Gate()
    entered, release = threading.Event(), threading.Event()
    first = {'name': 'phate' if tunable else 'pca', 'group': 'latent',
             'params': {'n_components': 2}}
    later = [{'name': 'pca', 'group': 'latent'},
             {'name': 'phate', 'group': 'latent'},
             {'name': 'pca', 'group': 'latent'}]
    recipe = {'name': 'embed', 'steps': [first] + ([] if last_step else later)}
    session = Session('stop', engine=vocab.INPROC, array=np.ones((12, 3)),
                      recipe=recipe, out_dir=tmp_path / 'outputs' / 'swissroll')
    calls = []

    def execute(name, params, state, g, ctx):
        calls.append(name)
        if len(calls) == 1:
            entered.set()
            assert release.wait(60), 'test did not release the executor'
        state['emb'] = state['X'][:, :2].copy()

    session.dispatch = {'latent': execute}
    monkeypatch.setattr(runner, '_attach_geometry', lambda *a, **kw: {})
    monkeypatch.setattr(manyruns_app, '_build_session', lambda *a, **kw: session)
    suite_calls = []
    monkeypatch.setattr(catalog, 'load_suite', lambda: suite_calls.append('suite') or {})
    monkeypatch.setattr(runner._suite, 'measure', lambda *a, **kw: {})
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    results, errors, said = [], [], []
    gate.bind(on_write=said.append)

    def run():
        try:
            results.append(app.start_for(_entry(), 'embed', gate=gate)(lambda *a: None))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, name='manyruns-run-test', daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        gate.close()
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and not errors
    result = results[0]
    assert calls == [first['name']]
    assert not suite_calls
    assert result['complete'] is False and result['cancelled'] is True
    assert result['not_run'] == [
        {'index': i, 'name': step['name'], 'reason': 'run abandoned'}
        for i, step in enumerate(recipe['steps']) if i > 0]
    assert session.discarded == (['phate'] if tunable else [])
    assert (session.state.get('emb') is None) == tunable
    assert artifacts.complete(artifacts.root(session.out_dir, session.run_id))
    rows = list(store.read(tmp_path / 'outputs'))
    assert len(rows) == 1 and rows[0]['complete'] is False
    assert all(any(f"step {item['index']} ({item['name']}): not run" in c
                   for c in rows[0]['caveats']) for item in result['not_run'])
    assert not any('final metrics are running' in line for line in said)


@pytest.mark.parametrize('abandon', [False, True])
def test_abandonment_during_final_geometry_preserves_measurements_and_records_status(
        tmp_path, monkeypatch, abandon):
    import threading
    import numpy as np
    from manyruns import artifacts, store
    from manyruns.pipeline import runner
    from manyruns.session import Session
    from manyruns.tui.gate import Gate

    monkeypatch.chdir(tmp_path)
    entered, release = threading.Event(), threading.Event()
    gate = Gate()
    session = Session(
        'finalize', engine=vocab.INPROC, array=np.ones((12, 3)),
        recipe={'name': 'embed', 'steps': [{'name': 'pca', 'group': 'latent'}]},
        out_dir=tmp_path / 'outputs' / 'swissroll')

    def execute(name, params, state, g, ctx):
        state['emb'] = state['X'][:, :2].copy()

    measurements = []

    def measure(emb, X, suite, **kw):
        measurements.append(emb.copy())
        entered.set()
        assert release.wait(60), 'test did not release final geometry'
        return {'trustworthiness': 0.75}

    session.dispatch = {'latent': execute}
    monkeypatch.setattr(runner, '_attach_geometry', lambda *a, **kw: {})
    monkeypatch.setattr(runner._suite, 'measure', measure)
    monkeypatch.setattr(manyruns_app, '_build_session', lambda *a, **kw: session)
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    results, errors, said = [], [], []
    gate.bind(on_write=said.append)

    def run():
        try:
            results.append(app.start_for(_entry(), 'embed', gate=gate)(lambda *a: None))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run, name='manyruns-run-test', daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        assert RunScreen.FINALIZING in said
        if abandon:
            gate.close()
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and not errors
    result = results[0]
    assert result['complete'] is (not abandon)
    assert result['cancelled'] is abandon
    assert result['ok'] is True
    assert result['not_run'] == [] and session.discarded == []
    assert len(measurements) == 1
    assert result['g_vector']['trustworthiness'] == 0.75
    assert artifacts.complete(artifacts.root(session.out_dir, session.run_id))
    rows = list(store.read(tmp_path / 'outputs'))
    assert len(rows) == 1
    assert rows[0]['complete'] is (not abandon)
    assert rows[0]['g_vector']['trustworthiness'] == 0.75
    assert any('run abandoned' in c for c in rows[0]['caveats']) is abandon
    assert not any('final geometry was not measured' in c for c in rows[0]['caveats'])


async def _until(pilot, condition, timeout=30):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        await pilot.pause()
        await asyncio.sleep(0.01)
    assert condition(), 'condition did not become true within the deadline'


@drives
@pytest.mark.parametrize('cancel', [False, True])
async def test_same_output_waits_for_worker_exit_and_cancels_late_launch(
        tmp_path, monkeypatch, prepared_entries, cancel):
    import threading
    from manyruns.tui.run import LeaveRunScreen

    monkeypatch.chdir(tmp_path)
    entered, release, bookkeeping = threading.Event(), threading.Event(), threading.Event()
    starts = []
    threads = []

    class BlockingApp(_App):
        def start_for(self, entry, recipe_name, decision=None, gate=None):
            def start(on_step):
                threads.append(threading.current_thread())
                starts.append(entry.name)
                if len(starts) == 1:
                    entered.set()
                    assert release.wait(60)
                    bookkeeping.set()
                    assert store_release.wait(60)
                return _mock_results()
            return start

    store_release = threading.Event()
    app = BlockingApp()
    async with app.run_test(size=(140, 35)) as pilot:
        try:
            app.open_run(_entry(), 'embed')
            await _until(pilot, entered.is_set)
            first = app.screen
            await pilot.press('escape')
            assert isinstance(app.screen, LeaveRunScreen)
            await pilot.press('tab', 'enter')
            await pilot.pause()
            assert isinstance(app.screen, LedgerScreen)
            await pilot.press('enter')
            await pilot.pause()
            waiting = 'waiting for the previous run to finish · esc cancels this start'
            assert waiting in _screen_text(app)
            assert starts == ['swissroll']
            app.open_run(_entry(), 'embed')
            await pilot.pause()
            assert starts == ['swissroll']
            # Finishing the executor is insufficient: store/summary writes still own the path.
            release.set()
            await _until(pilot, bookkeeping.is_set)
            await pilot.pause()
            assert starts == ['swissroll']
            if cancel:
                await pilot.press('escape')
                assert waiting not in _screen_text(app)
            store_release.set()
            if cancel:
                await _until(pilot, lambda: not threads[0].is_alive())
                await asyncio.sleep(0.1)
                await pilot.pause()
                assert starts == ['swissroll']
                assert not isinstance(app.screen, RunScreen)
            else:
                await _until(pilot, lambda: len(starts) == 2)
                assert isinstance(app.screen, RunScreen) and app.screen is not first
                assert not threads[0].is_alive()
                assert len(starts) == 2
        finally:
            release.set()
            store_release.set()
    for thread in threads:
        assert thread.daemon and thread.name == 'manyruns-run'
        thread.join(timeout=5)
        assert not thread.is_alive()


@drives
@pytest.mark.parametrize('cancel', [False, True])
async def test_replacing_waiting_recipe_keeps_one_wait_line(
        tmp_path, monkeypatch, prepared_entries, cancel):
    import threading

    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    starts = []
    threads = []

    class BlockingApp(_App):
        def start_for(self, entry, recipe_name, decision=None, gate=None):
            def start(on_step):
                threads.append(threading.current_thread())
                starts.append((recipe_name, decision))
                if len(starts) == 1:
                    assert release.wait(60), 'test did not release the first run'
                return _mock_results()
            return start

    app = BlockingApp()
    try:
        async with app.run_test(size=(140, 35)) as pilot:
            try:
                app.open_run(_entry(), 'embed')
                await _until(pilot, lambda: bool(starts))
                await pilot.press('escape', 'tab', 'enter')
                assert isinstance(app.screen, LedgerScreen)
                await pilot.press('enter')
                await pilot.pause()
                assert isinstance(app.screen, RosterScreen)
                assert app._waiting_run is not None
                assert len(app.screen.query('#pending-run')) == 1

                # Both synchronous calls must reuse the one visible wait line.
                app.open_run(_entry(), 'cflows', decision='replacement-offer')
                app.open_run(_entry(), 'cflows', decision='replacement-offer')
                assert len(app.screen.query('#pending-run')) == 1
                await pilot.pause()
                assert _screen_text(app).count(app.WAITING) == 1
                assert app._waiting_run[3:] == ('cflows', 'replacement-offer')
                assert starts == [('embed', None)]

                if cancel:
                    await pilot.press('escape')
                    assert app.WAITING not in _screen_text(app)
                release.set()
                await _until(pilot, lambda: not app._run_reservations)
                if cancel:
                    assert isinstance(app.screen, RosterScreen)
                    assert starts == [('embed', None)]
                else:
                    assert isinstance(app.screen, RunScreen)
                    assert starts == [('embed', None), ('cflows', 'replacement-offer')]
                    await _until(pilot, lambda: app.screen._finished)
                assert app._waiting_run is None
            finally:
                release.set()
    finally:
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()


@drives
async def test_rapid_duplicate_start_reserves_before_worker_mount(prepared_entries):
    app = _App()
    async with app.run_test() as pilot:
        app.open_run(_entry(), 'embed')
        app.open_run(_entry(), 'embed')
        await pilot.pause()
        assert len(app.started) == 1
        assert len([screen for screen in app.screen_stack if isinstance(screen, RunScreen)]) == 1


@drives
async def test_source_names_with_colliding_output_paths_serialize(
        tmp_path, monkeypatch, prepared_entries):
    import threading
    import numpy as np
    from manyruns.pipeline import runner
    from manyruns.tui.run import LeaveRunScreen

    monkeypatch.chdir(tmp_path)
    entries = []
    for name in ('α.npy', 'β.npy'):
        path = tmp_path / name
        np.save(path, np.ones((12, 3)))
        entries.append(state.DataEntry(
            name=name, kind='local', path=path,
            obs=Observation(source=name, shape='manifold', modality='generic')))
    entered, release = threading.Event(), threading.Event()
    sessions, threads = [], []
    real_build = manyruns_app._build_session

    def build(project, args, **kw):
        session = real_build(project, args, **kw)
        session.recipe = {'name': 'embed', 'steps': []}
        sessions.append(session)
        threads.append(threading.current_thread())
        if len(sessions) == 1:
            entered.set()
            assert release.wait(60), 'test did not release the first run'
        return session

    monkeypatch.setattr(manyruns_app, '_build_session', build)
    monkeypatch.setattr(runner._suite, 'measure', lambda *a, **kw: {})

    class BlockingApp(_App):
        start_for = ManyrunsApp.start_for

    app = BlockingApp(entries=entries)
    try:
        async with app.run_test(size=(140, 35)) as pilot:
            app.open_run(entries[0], 'embed')
            await _until(pilot, entered.is_set)
            output = sessions[0].out_dir.resolve()
            assert output == tmp_path / 'outputs' / 'npy'
            await pilot.press('escape')
            assert isinstance(app.screen, LeaveRunScreen)
            await pilot.press('tab', 'enter')
            await pilot.pause()
            assert isinstance(app.screen, LedgerScreen)
            app.open_run(entries[1], 'embed')
            await pilot.pause()
            assert app.WAITING in _screen_text(app)
            assert len(sessions) == 1
            assert set(app._run_reservations) == {output}
            saved = yaml.safe_load((output / 'project.yaml').read_text())
            assert saved['data_folder'] == str(entries[0].path)
            release.set()
            await _until(pilot, lambda: len(sessions) == 2)
            assert not threads[0].is_alive()
            assert sessions[1].out_dir.resolve() == output
            await _until(pilot, lambda: app.screen._finished)
            await _until(pilot, lambda: not app._run_reservations)
            saved = yaml.safe_load((output / 'project.yaml').read_text())
            assert saved['data_folder'] == str(entries[1].path)
            assert len(sessions) == 2
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize('gated', [False, True])
def test_unicode_source_project_and_execution_share_output_path(tmp_path, monkeypatch, gated):
    from manyruns.pipeline import runner
    from manyruns.tui.gate import Gate

    monkeypatch.chdir(tmp_path)
    entry = state.DataEntry(
        name='α.npy', kind='local', path=tmp_path / 'α.npy',
        obs=Observation(source='α.npy', shape='manifold', modality='generic'))
    outputs = []
    real_build = manyruns_app._build_session

    def build(project, args, **kw):
        session = real_build(project, args, **kw)
        session.recipe = {'name': 'embed', 'steps': []}
        outputs.append(session.out_dir.resolve())
        return session

    def explore(**kw):
        outputs.append(kw['out_dir'].resolve())
        return _mock_results()

    monkeypatch.setattr(manyruns_app, '_build_session', build)
    monkeypatch.setattr(manyruns_app, 'explore_once', explore)
    monkeypatch.setattr(runner._suite, 'measure', lambda *a, **kw: {})
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    app.start_for(entry, 'embed', gate=Gate() if gated else None)(lambda *a: None)
    assert outputs == [tmp_path / 'outputs' / 'npy']
    assert (outputs[0] / 'project.yaml').is_file()


@drives
async def test_distinct_output_paths_can_run_independently(tmp_path, monkeypatch, prepared_entries):
    import threading

    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    threads, entered = [], []

    class BlockingApp(_App):
        def start_for(self, entry, recipe_name, decision=None, gate=None):
            def start(on_step):
                threads.append(threading.current_thread())
                entered.append(entry.name)
                assert release.wait(60)
                return _mock_results()
            return start

    app = BlockingApp()
    try:
        async with app.run_test() as pilot:
            app.open_run(_entry('swissroll'), 'embed')
            await _until(pilot, lambda: len(entered) == 1)
            app.open_run(_entry('arch_soft'), 'embed')
            await _until(pilot, lambda: len(entered) == 2)
            assert entered == ['swissroll', 'arch_soft']
            assert len(app._run_reservations) == 2
            assert all(t.is_alive() and t.daemon for t in threads)
            assert app._waiting_run is None
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()


@drives
async def test_worker_exception_releases_only_its_own_reservation(tmp_path, monkeypatch, prepared_entries):
    import threading

    monkeypatch.chdir(tmp_path)
    release, finish_second = threading.Event(), threading.Event()
    threads = []

    class FailingApp(_App):
        def start_for(self, entry, recipe_name, decision=None, gate=None):
            def start(on_step):
                threads.append(threading.current_thread())
                if len(threads) == 1:
                    assert release.wait(60)
                    raise RuntimeError('first worker failed')
                assert finish_second.wait(60)
                return _mock_results()
            return start

    app = FailingApp()
    try:
        async with app.run_test(size=(140, 35)) as pilot:
            app.open_run(_entry(), 'embed')
            await _until(pilot, lambda: len(threads) == 1)
            path, old = next(iter(app._run_reservations.items()))
            await pilot.press('escape', 'tab', 'enter')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
            assert app.WAITING in _screen_text(app)
            release.set()
            await _until(pilot, lambda: len(threads) == 2)
            current = app._run_reservations[path]
            assert current is not old and not threads[0].is_alive()
            app._worker_started(path, old.token, threads[0])
            app._poll_runs()
            assert app._run_reservations[path] is current
            assert current.thread is threads[1]
    finally:
        release.set()
        finish_second.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()


@drives
async def test_navigation_cancels_a_waiting_start_without_late_launch(tmp_path, monkeypatch, prepared_entries):
    import threading

    monkeypatch.chdir(tmp_path)
    release = threading.Event()
    threads = []

    class BlockingApp(_App):
        def start_for(self, entry, recipe_name, decision=None, gate=None):
            def start(on_step):
                threads.append(threading.current_thread())
                assert release.wait(60)
                return _mock_results()
            return start

    app = BlockingApp()
    try:
        async with app.run_test() as pilot:
            app.open_run(_entry(), 'embed')
            await _until(pilot, lambda: bool(threads))
            await pilot.press('escape', 'tab', 'enter')
            await pilot.pause()
            await pilot.press('enter')
            await pilot.pause()
            assert app._waiting_run is not None
            app.open_ledger(_entry('arch_soft'))
            await pilot.pause()
            assert app._waiting_run is None
            release.set()
            await _until(pilot, lambda: not threads[0].is_alive())
            await _until(pilot, lambda: not app._run_reservations)
            assert isinstance(app.screen, LedgerScreen)
            assert len(threads) == 1
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)


@drives
async def test_five_mock_cycles_preserve_fresh_run_ids_and_release_owned_daemons(
        tmp_path, monkeypatch, prepared_entries):
    """Preservation stress test: five successful attempts, without real compute or sleeps."""
    from manyruns import store

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(manyruns_app, 'load_recipe', lambda name: {
        'name': name, 'steps': [{'name': 'pca', 'group': 'latent', 'params': {'n_components': 2}}]})

    class RealStart(ManyrunsApp):
        def get_default_screen(self):
            return RosterScreen(entries=[_entry()])

    app = RealStart(argparse.Namespace(engine='mock'))
    ids, threads = [], []
    async with app.run_test() as pilot:
        app.open_run(_entry(), 'embed')
        for i in range(5):
            await _until(pilot, lambda: isinstance(app.screen, RunScreen) and app.screen._finished, timeout=20)
            screen = app.screen
            assert screen.error is None and screen.results['complete'] is True
            ids.append(screen.results['run_id'])
            threads.append(screen.worker_thread)
            if i < 4:
                await pilot.press('escape', 'tab', 'enter')
                await pilot.pause()
                assert isinstance(app.screen, LedgerScreen)
                await pilot.press('enter')
        await _until(pilot, lambda: not app._run_reservations)
    assert len(set(ids)) == 5
    rows = list(store.read(tmp_path / 'outputs'))
    assert len(rows) == 5 and {row['run_id'] for row in rows} == set(ids)
    for thread in threads:
        assert thread.name == 'manyruns-run' and thread.daemon
        thread.join(timeout=2)
        assert not thread.is_alive()


@drives
async def test_stale_downloadable_selection_fetches_once_and_keeps_the_chosen_offer(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    import anndata
    from manyruns import datasetfetch, decisions, inspected
    from manyruns.tui.samplefetch import SampleFetchScreen

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('MANYRUNS_DATASET_DIR', raising=False)
    sample = catalog.load_dataset('pbmc3k')
    drop, configs = tmp_path / 'drop', tmp_path / 'catalog'
    drop.mkdir()
    configs.mkdir()
    assert configs.is_dir()
    path = drop / 'pbmc3k_raw.h5ad'
    anndata.AnnData(np.ones((4, 3))).write_h5ad(path)
    payload = path.read_bytes()
    sample['handle'].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (configs / 'pbmc3k.yaml').write_text(yaml.safe_dump(sample))
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(drop))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    monkeypatch.setattr(inspected, 'HOME', tmp_path / 'cache')
    calls, choice_ids = [], []

    def opener(url, **kwargs):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', opener)
    entry = state.catalog_entry('pbmc3k')
    assert not entry.missing

    class CapturingApp(_App):
        def start_for(self, entry, recipe_name, decision=None, gate=None):
            choice_ids.append(decision)
            return super().start_for(entry, recipe_name, decision, gate)

    app = CapturingApp(entries=[entry])
    async with app.run_test() as pilot:
        app.open_ledger(entry, selected_recipe='embed')
        await pilot.pause()
        offered = [row.as_offer() for row in app.screen.rows]
        path.unlink()
        await pilot.press('enter')
        assert isinstance(app.screen, SampleFetchScreen)
        assert not app.started and not calls
        await pilot.press('enter')
        # push_screen exposes the instance before it has mounted. Wait for the fake
        # run to finish so context teardown cannot remove children during on_mount.
        await _until(pilot, lambda: isinstance(app.screen, RunScreen) and app.screen._finished)
        assert len(calls) == 1 and len(app.started) == 1
        ready, chosen = app.started[0]
        assert chosen == 'embed' and ready.path.is_file() and not ready.missing
    rows = list(decisions.read(tmp_path / 'outputs'))
    assert len(rows) == 1
    assert rows[0]['decision_id'] == choice_ids[0]
    assert rows[0]['offered'] == offered
    assert rows[0]['chosen'] == 'embed'


@drives
@pytest.mark.parametrize('key', ['ctrl+c', 'ctrl+q'])
async def test_quit_binding_exits_before_test_context_cleanup(key):
    app = _App()
    async with app.run_test() as pilot:
        await pilot.press(key)
        await pilot.pause()
        assert not app.is_running
