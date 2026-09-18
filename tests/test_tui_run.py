"""The run screen (`manyruns.tui.run`) — component 4 of the TUI rewrite.

Driven through `App.run_test()`, which is what §5 of the spec buys: a `Pilot` presses keys and
resizes the terminal, and there is no terminal. Two kinds of assertion, deliberately:

  * on the COMPOSITOR's strips (`_screen_text`) — what the driver would write, so a widget that
    was built but never reached the screen fails;
  * on a pane's renderable at the width Textual gave it (`_pane`) — the whole of a scrollable
    pane, including the part below the fold, which the compositor legitimately crops.

The end-to-end test drives a real in-process run of `embed` on a 60×8 array through a thread
worker (~1.4 s), because the properties that matter here — the observer arriving on another
thread, the figure landing on disk, the g-vector at the end — are exactly the ones a stubbed
run would not have.

WHAT WAS MEASURED BEFORE ANY OF THIS WAS WRITTEN, and it is §8's one remaining unknown: an
inline image escape does NOT survive Textual's compositor. `test_the_figure_is_a_path_because_an_image_escape_does_not_survive`
carries the three measurements and fails if anyone reaches for `figures.escape` here.
"""
from __future__ import annotations

import asyncio
import functools
import io
from pathlib import Path
import re
import threading

import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402 - after the importorskip, which is the point of it
from textual.widgets import Input, Static  # noqa: E402

from manyruns import agents, narrate, shell, watch  # noqa: E402
from manyruns.tui import run as run_screen  # noqa: E402
from manyruns.tui.run import RunScreen  # noqa: E402
from manyruns.tui.state import StepView  # noqa: E402

EMBED = {"name": "embed", "steps": [{"name": "phate", "group": "latent",
                                     "params": {"n_components": 3}}]}
THREE = {"name": "cflows", "steps": [{"name": "phate", "group": "latent"},
                                     {"name": "mioflow", "group": "lightning"},
                                     {"name": "separation", "group": "analysis"}]}


def drives(body):
    """Run an async test body on its own loop.

    `App.run_test()` is an async context manager and this repo has no `pytest-asyncio`, `anyio`
    or `trio` — measured, none of the three is installed. Adding one is a dependency decision,
    and §8 of the spec already records what an unexamined dependency costs (installing textual
    pulled `rich` 13 → 15 sideways). One `asyncio.run` per test costs nothing and takes the
    decision off the table. `functools.wraps` keeps the signature pytest reads, so fixtures and
    `parametrize` arrive exactly as they would on a plain test.
    """
    @functools.wraps(body)
    def wrapper(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))

    return wrapper


class _Harness(App):
    """The smallest app that can hold a screen. `app.py` is component 2's; this exists so the
    screen can be tested before there is one, and it records the `Finished` message so the
    bubbling that component 2 depends on is asserted rather than assumed."""

    def __init__(self, screen: RunScreen) -> None:
        super().__init__()
        self._run_screen = screen
        self.finished: list = []

    def on_mount(self) -> None:
        self.push_screen(self._run_screen)

    def on_run_screen_finished(self, message: RunScreen.Finished) -> None:
        self.finished.append(message)


def _screen_text(app: App) -> str:
    """What the driver would write. `_compositor.render_strips()` is the same call the driver
    makes to paint a frame, so this is the screen, not a renderable that might never reach it.
    Private, and used anyway: there is no public way to ask "what is on screen" without a
    terminal, and asserting on anything else would let a pane that is built but never mounted
    pass."""
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


#: Colour, and only colour. SGR sequences are stripped so an assertion can read `✓  phate`
#: instead of `\x1b[32m✓\x1b[0m  \x1b[1mphate`; OSC and APC are deliberately NOT stripped,
#: because the image-escape test's whole job is to notice one.
_SGR = re.compile(r"\x1b\[[0-9;]*m")


def _pane(app: App, selector: str) -> str:
    """One pane's full content, rendered at the width TEXTUAL gave it.

    The width is the load-bearing half: it is what makes an assertion about wrapping an
    assertion about the layout engine's answer rather than about a number this test picked.
    """
    from rich.console import Console

    widget = app.screen.query_one(selector, Static)
    buf = io.StringIO()
    Console(file=buf, width=max(widget.size.width, 1), legacy_windows=False).print(widget.content)
    return _SGR.sub("", buf.getvalue())


def _words(text: str) -> str:
    """Prose, with the line breaks the layout engine chose collapsed away."""
    return " ".join(_SGR.sub("", text).split())


def _unbroken(text: str) -> str:
    """All whitespace removed. For a PATH, which wraps mid-token: a 76-column pane folds
    `/private/var/…/plots/phate.png` wherever it runs out of room, and joining the lines with a
    space would put one inside the filename."""
    return "".join(_SGR.sub("", text).split())


async def _settled(pilot, app: App, timeout: float = 30.0) -> None:
    """Wait for the run to actually FINISH, then let its `Finished` message be processed.

    `app.workers.wait_for_complete()` WAS THE WHOLE WAIT, and it waits for nothing here.
    `RunScreen._drive` starts a raw `threading.Thread` (run.py:431), deliberately — a Textual
    thread worker runs on a non-daemon executor thread that is joined before the process can
    exit, and that left a quit holding the terminal for the length of the run (its docstring
    measures 15.01 s). A raw daemon thread is not in `app.workers`, so `wait_for_complete()`
    returned immediately and the only real wait was three `pilot.pause()` calls.

    So this passed on timing luck. Reported by an audit that ran it while nine other processes
    were saturating the machine: failed / passed / failed across three runs, where it passes 5/5
    on an idle one. A load-sensitive test is a flaky test — it will fail on a busy CI runner, and
    every "the suite is green" claim rests on it.

    Waits on the CONDITION instead: `Finished` sets `results` or `error` on the screen, and the
    harness records it. Bounded by a generous timeout so a genuine hang still fails the test
    rather than hanging the suite.
    """
    import asyncio
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(app, "finished", None) or app.screen.results or app.screen.error:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    for _ in range(3):
        await pilot.pause()


def _real_start(tmp_path):
    """A real `embed` run, bound to everything but the observer — the exact shape the screen's
    `start` argument takes, and the shape every existing run path already offers
    (`Session(on_step=…)`, `runner.run_inproc(on_step=…)`, `app.run_explorations(on_step=…)`).

    60×8 because it has to be a real PHATE fit and it has to be quick: measured, 1.3 s.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("phate")

    def start(on_step):
        from manyruns.session import Session

        session = Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                          recipe=EMBED, seed=0, on_step=on_step,
                          array=np.random.default_rng(0).normal(size=(60, 8)))
        session.run_recipe()
        return session.close()

    return start


# ── the landing state: declared steps before anything has run ────────────────
@drives
async def test_every_declared_step_is_on_screen_before_anything_runs():
    """The overlay is the whole reason a two-minute embedding is watchable, and it has to be
    there at MOUNT — a screen that renders only what has been reported opens empty and fills in
    from nothing.

    This is the test that caught the real bug in this component: the paint was guarded on
    `self.is_mounted`, which is False inside a pushed Screen's own `on_mount` (measured, textual
    8.2.8), so every row silently failed to draw and the screen opened blank.
    """
    app = _Harness(RunScreen(THREE))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        text = _screen_text(app)

    for name in ("phate", "mioflow", "separation"):
        assert name in text
    assert text.count("queued") == 3
    assert "0 of 3 steps" in text


@drives
async def test_the_queued_glyph_is_narrates_and_not_this_screens():
    """`narrate.step_mark(None)` is the one home for "declared and not started". A screen with
    its own glyph is a fourth account of a run — the rule §1 exists to enforce."""
    glyph, word = narrate.step_mark(None)
    app = _Harness(RunScreen(THREE))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        text = _screen_text(app)

    assert f"{glyph}  phate" in text
    assert word in text


# ── the live states and the outcomes, all from narrate ───────────────────────
@pytest.mark.parametrize("outcome", list(watch.OUTCOMES))
@drives
async def test_every_outcome_is_drawn_with_narrates_glyph_and_word(outcome):
    """Both halves of the mark come from `narrate.step_mark`, for every outcome the vocabulary
    declares. `test_watch.test_every_renderer_covers_the_whole_outcome_vocabulary` pins the two
    existing renderers to that vocabulary; this pins the third surface to the same function, so
    a new outcome cannot render as `?` here while reading correctly in the other two."""
    glyph, word = narrate.step_mark({"outcome": outcome})
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": outcome,
                                  "seconds": 1.5}])
        await pilot.pause()
        text = _screen_text(app)

    assert f"{glyph}  phate" in text
    assert word in text


@drives
async def test_a_running_step_says_running_and_shows_no_duration_yet():
    """`seconds` is a DURATION and the record does not have one until the step settles.
    `run_panel` prints `0.00s` for a record mid-flight because a plain line has nowhere to put
    "not yet"; a live view that did the same would report a finished step that took no time."""
    glyph, word = narrate.step_mark({"state": "running"})
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "state": "running",
                                  "started": 100.0, "seconds": 0.0}])
        await pilot.pause()
        text = _screen_text(app)

    assert f"{glyph}  phate" in text and word in text
    assert "0.00s" not in text


@drives
async def test_a_step_that_ran_past_the_declared_list_is_still_on_screen():
    """`Session.step` issues ad-hoc actions with no declaration behind them and `run_recipe`
    does not de-duplicate, so a re-run lands at its own index. `shell._progress_panel` drops
    those — it enumerates the declared list — and that is correct for a fixed recipe and wrong
    for a live lineage. A step that ran and is not on screen is the one failure a run view must
    not have. Fails if this screen ever iterates `feed.declared` instead of `feed.rows()`."""
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0},
                                 {"index": 1, "name": "leiden", "outcome": "ok", "seconds": 0.2}])
        await pilot.pause()
        text = _screen_text(app)

    assert "leiden" in text
    assert "2 of 2 steps" in text


# ── the ink is the rich surface's ink ────────────────────────────────────────
def test_the_screen_does_not_keep_its_own_outcome_colours():
    """It imports `shell._OUTCOME_STYLE` rather than restating it, so the one table
    `test_watch` already pins in both directions against `watch.OUTCOMES` is the only one that
    exists. A copy here would be a table nothing compares to the vocabulary — precisely the
    drift `watch.OUTCOMES`' own comment records having been paid for once already."""
    assert run_screen._OUTCOME_STYLE is shell._OUTCOME_STYLE
    assert run_screen._PENDING is shell._PENDING


# ── layout, which is what Textual was brought in for ─────────────────────────
def test_a_wrapped_detail_stays_attached_to_its_step():
    """A hanging indent, not a first-line one. With `"    " + detail` in a single string, only
    the first line is indented and the continuation starts at column 0 — where it reads as a
    line belonging to the next step rather than to this one. `Padding` indents every line."""
    from rich.console import Console

    detail = ("ValueError: mioflow refused the fit because the label vector it was handed is a "
              "condition axis and not a time axis")
    step = StepView(index=0, name="phate", state="error", glyph="✗", word="error",
                    seconds=0.1, detail=detail)
    buf = io.StringIO()
    Console(file=buf, width=50, legacy_windows=False).print(run_screen.steps_view([step]))
    lines = [_SGR.sub("", line) for line in buf.getvalue().splitlines()]

    # line 0 is the step's own head row; everything after it belongs to that step
    wrapped = [line for line in lines[1:] if line.strip()]
    assert len(wrapped) > 1, "the detail has to wrap for this test to mean anything"
    for line in wrapped:
        assert line.startswith("    "), f"continuation is not indented: {line!r}"


@pytest.mark.parametrize("width", [24, 30, 36, 38, 43, 44, 60, 80, 119])
def test_no_pane_loses_a_character_at_any_width(width):
    """The property §0 is actually about, tested where it broke.

    `rich.table` 15.0.0 deletes characters when it folds a word longer than its column. Measured
    before this test existed, with the geometry pane built as a two-column grid: at a 38-cell
    width `not measured — GeodesicDistanceCorrelation: no get_gt_dists` came out as
    `GeodesicDistance` / `relation: no` — the `Cor` gone, no ellipsis, nothing on screen to say
    so. The pane lost characters at every width from 24 to 43, and 36 is what a 100-column
    terminal gives it. The steps pane did the same to a path inside a traceback at 24.

    Both panes are now `Text` lines in a `Group`, which measured lossless at every width from 12
    to 89. This runs on the renderables directly, with no app: the failure is in rich's
    measurement of a cell, so a test that needs a terminal to see it would be testing the wrong
    layer. Mutation-checked — putting either pane back into a `Table.grid` fails this at 36.
    """
    from rich.console import Console

    token = "/tmp/example/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/plots/phate.png"
    detail = f"FileNotFoundError: {token} is gone"
    reason = "GeodesicDistanceCorrelation: no get_gt_dists on the dataset"
    step = StepView(index=0, name="phate", state="error", glyph="✗", word="error",
                    seconds=0.1, detail=detail, plots=(token,),
                    deltas=(narrate.GeometryRow("participation_ratio", "— → 1.722"),))
    g = {"n_samples": 60, "n_embedded": 60, "n_features": 8, "final_dim": 3,
         "trustworthiness": 0.7682, "geodesic_distance_correlation": None,
         "geodesic_distance_correlation_note": reason,
         "suite_null": "data-null unimplemented — comparable, not a finding"}

    def flat(renderable):
        buf = io.StringIO()
        Console(file=buf, width=width, legacy_windows=False).print(renderable)
        return _unbroken(buf.getvalue())

    panes = {"steps": (run_screen.steps_view([step]), [detail, "participation_ratio"]),
             "geometry": (run_screen.geometry_view({"g_vector": g}),
                          ["trustworthiness", "geodesic_distance_correlation", reason[:40]]),
             # THE FILENAME, not the path, and this is the contract change rather than a
             # weakening. A 74-character token cannot appear whole in a 24-column pane, and the
             # old pane "kept" it by wrapping across four of its seven rows — which starved the
             # picture to one row and is the bug `figures_view` was rewritten to fix. It shows
             # the full path when it FITS and the filename when it does not, and a filename is a
             # COMPLETE string: nothing is silently cut, which is the property this test holds.
             # `test_the_path_survives_whatever_happens_to_the_picture` asserts the other half —
             # that the full path is still reachable, in the viewer.
             "figures": (run_screen.figures_view([step], -1, width),
                         [token if len(token) <= width else Path(token).name])}
    for pane, (renderable, needles) in panes.items():
        drawn = flat(renderable)
        for needle in needles:
            assert needle.replace(" ", "") in drawn, f"{pane} pane dropped {needle!r} at {width}"



@drives
async def test_a_long_detail_wraps_and_is_never_clipped():
    """§0 opens with the panels wrapping mid-value, and `StepView.detail` is raw precisely so a
    layout engine can re-flow it. `run_panel` clips at 58 and `shell._steps_body` at 24, both
    correctly — neither can re-wrap. Clipping here would hand the pane a string it could not
    un-truncate at a wider terminal.

    Mutation-checked: putting `narrate._clip(row.detail)` back into `steps_view` fails this.
    """
    # `[ok]` is in there because a detail is engine output and rich would delete a bracketed
    # lowercase word outright — measured. Mutation-checked: dropping `escape()` fails this.
    detail = ("ValueError: mioflow refused the fit because the label vector it was handed is a "
              "condition axis and not [ok] a time axis, which it cannot integrate over")
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "error",
                                  "seconds": 0.1, "detail": detail}])
        await pilot.pause()
        wide = _pane(app, "#steps")
        await pilot.resize_terminal(100, 40)
        await pilot.pause()
        narrow = _pane(app, "#steps")

    # every word survives at both widths, and none of it is elided
    for text in (wide, narrow):
        assert "…" not in text
        assert all(word in _words(text) for word in detail.split())
    # and the layout genuinely re-solved: the same text needs more lines in a narrower pane
    assert narrow.count("\n") > wide.count("\n")


@drives
async def test_the_panes_stack_when_the_terminal_is_too_narrow_to_divide():
    """A fraction of a terminal too narrow to divide is `width=30` in a different unit. Measured
    at 100 columns the top-right content is 36 wide and every metric name the shipped suite
    produces fits on one line; at 80 it would be 28. Stacked, it is 76.

    The measurement is the g-vector's and the pane is the answer's now — `NARROW`'s own note
    says why the threshold is kept on it. THE TWO NUMBERS BELOW ARE WHAT MAKES THAT CHECKABLE:
    they are the pane's widths, not its content's, so they hold across the rename and would move
    the moment `#answer-pane` stopped being the same CSS slot."""
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not screen.has_class("narrow")
        side_by_side = app.screen.query_one("#answer", Static).size.width

        await pilot.resize_terminal(80, 40)
        await pilot.pause()
        assert screen.has_class("narrow")
        stacked = app.screen.query_one("#answer", Static).size.width

    assert side_by_side == 44           # 120 columns, 2fr of the 3:2 split
    assert stacked == 76                # 80 columns, full width
    assert stacked > side_by_side


@drives
async def test_the_screen_installs_no_timer_of_its_own():
    """§2: `rewardspy` polls at `set_interval(0.5, refresh_all)` because a training loop emits
    thousands of rewards; our steps push through `on_step` and take seconds, so the screen
    repaints on the event and on nothing else. Fails the moment anyone adds a poll.

    NOT "no interval exists" — measured, Textual installs one itself the moment a Screen is
    mounted: `Screen._on_timer_update` at 1/60 s, from `textual.screen`. That is its compositor's
    business and not ours. What is asserted is that no interval belonging to THIS package is
    installed, which is the claim §2 actually makes.
    """
    calls: list = []
    real = RunScreen.set_interval

    def spy(self, *args, **kwargs):
        # delegating, not stubbing: `set_interval` returns a `Timer` that Textual holds on to,
        # and a lambda returning None broke the app before the assertion could run.
        calls.append(args)
        return real(self, *args, **kwargs)

    screen = RunScreen(THREE)
    screen.set_interval = spy.__get__(screen, RunScreen)  # type: ignore[method-assign]
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "state": "running"}])
        await pilot.pause()
        await pilot.resize_terminal(80, 30)
        await pilot.pause()

    ours = [args for args in calls
            if getattr(getattr(args[1], "__func__", args[1]), "__module__", "")
            .startswith("manyruns")]
    assert ours == []
    assert calls, "the spy is wired: Textual's own screen-update interval was recorded"


# ── the figure: measured, and the measurement is the test ────────────────────
def _png(path, size=(640, 480), colour=(200, 60, 60)):
    """A real image on disk. Real, not a header and 4 KB of zeroes, because the pane now READS
    it — an unreadable file exercises the degradation path, not the drawing one."""
    Image = pytest.importorskip("PIL.Image", reason="pillow rides with matplotlib")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def test_the_tier_is_a_terminal_capability_and_not_a_property_of_the_app():
    """**This test replaces one that asserted the opposite, and the correction is the point.**

    What it used to say: no image escape may reach the pane — on the strength of three
    measurements against `textual 8.2.8`. Those measurements still hold and every one is about an
    ESCAPE PUT THROUGH THE COMPOSITOR: rich's `cell_len` counts a 51,681-character iTerm2
    sequence as 51,679 printable cells and Textual measures with the same function; mounted in a
    `Static` at 80×24 the sequence came back torn across 663 strips with 2,103 of its characters
    emitted; and `App.suspend()` writes `\\x1b[?1049l` and leaves the alternate screen.

    None of that is a property of the app. Measured in a pty on the same Textual, inside a
    two-pane layout with scroll containers: 4 kitty/TGP sequences reach the terminal, and 4 sixel
    on a sixel terminal — because `textual-image` composites a PLACEHOLDER and writes the
    graphics separately, so the escape never takes the path those three measurements describe.

    So what is asserted now is the shape of the decision, not an outcome: the tier is chosen from
    what the TERMINAL admits, and `is_pixel_perfect` answers the only question a screen needs —
    is this real pixels, or a grid of cells that cannot show where one point sits.
    """
    from manyruns.tui import images

    assert images.tier() in ("none", "tgp", "sixel", "halfcell", "unicode")
    # the two graphics tiers, and only those, are pointwise
    assert images.is_pixel_perfect() == (images.tier() in ("tgp", "sixel"))
    # `run_test` has no tty, so detection lands on the fallback — which is exactly why a test
    # cannot assert that pixels appear, and why this asserts the rule instead.
    assert images.tier() != "none", "textual-image is a base dependency; this suite has it"


@drives
async def test_the_path_survives_whatever_happens_to_the_picture(tmp_path):
    """The path is the half that was always load-bearing — it is what you hand to another
    program — so `figures_view` gives the LIST its rows and the picture whatever is left. A
    layout that measured the picture first would drop the useful half for the pretty one."""
    # a folder whose name is legal rich markup, because a path is user data. Measured: rich's
    # tag regex fires on `[` followed by `[a-z#/@]`, so `run[1]` is harmless and `run[old]` is
    # not — unescaped, `/tmp/run[old]/phate.png` renders as `/tmp/run/phate.png`, a path nobody
    # can open. Mutation-checked: dropping `escape()` from `figures_view` fails this.
    png = _png(tmp_path / "run[old]" / "phate.png")
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0,
                                  "plots": [str(png)]}])
        await pilot.pause()
        pane = _pane(app, "#figures")

    assert "phate.png" in _words(pane), "the filename identifies the figure in the pane"
    assert "phate ·" in _words(pane), "a figure has to say which step drew it"
    # THE WHOLE PATH LIVES IN THE VIEWER NOW, and that is where it can be read: the pane is 44
    # columns at this terminal size and this path is 140, so the old pane "kept" it by wrapping
    # across four of its seven rows and starving the picture to one. Asserted here rather than
    # dropped, because the path being REACHABLE is the property — which pane holds it is layout.
    from manyruns.tui.figure import figures_of

    assert str(png) in str(figures_of(screen.feed.rows())), "the whole path, so it can be opened"


@drives
async def test_the_pane_names_the_keys_that_go_further_than_the_terminal(tmp_path):
    """The old text read "the app cannot draw inside its own layout" — true of the approach and
    false of the problem, and on the front door it read as a dead end. What replaced it names
    KEYS, because a pane that only reports a limitation is still a dead end.

    `s` and `o` matter more than `f` here: no cell-based tier can show where an individual point
    sits, and pointwise is the entire content of an embedding scatter. So the pane's job when it
    cannot draw honestly is to say where the real figure is.
    """
    png = _png(tmp_path / "run" / "phate.png")
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0,
                                  "plots": [str(png)]}])
        await pilot.pause()
        said = _words(_render(run_screen.figures_view(screen.feed.rows())))
        heading = _words(_pane(app, "#heading"))

    # the exact wording differs by tier — what must hold is that all three keys are named,
    # because on a cell tier this line is the ONLY thing pointing at the figure
    for key in ("f", "s", "o"):
        assert f"{key} " in said or f" {key}" in said, said
    # Abbreviated since the pane grew a selector: at a 36-column pane (a 100-column terminal)
    # the spelled-out form wrapped, and a wrapped hint costs the picture a row. What must hold
    # is that each key is NAMED with what it does, not that it is spelled a particular way.
    assert "full" in said and "save" in said and "open" in said, said
    assert "←/→" in said, "the selector has to be discoverable from the pane"
    # and the same keys on the heading, which is the line a reader is already looking at
    for key in ("f figure", "s save", "o open"):
        assert key in heading, heading


@drives
async def test_the_picture_is_a_widget_beside_the_list_not_inside_it(tmp_path):
    """`AutoImage` is a WIDGET, so it cannot live in the rich `Group` the pane builds — which is
    why `figures_view` went back to being text only. The picture is its sibling.

    Asserting the wiring rather than the pixels: which tier draws depends on the terminal, and
    `run_test` has none (`textual_image.renderable` picks Unicode when `is_tty` is False). What
    must hold on every tier is that the newest figure reached the widget.
    """
    a, b = _png(tmp_path / "a.png"), _png(tmp_path / "b.png")
    screen = RunScreen(THREE)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [
            {"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0, "plots": [str(a)]},
            {"index": 1, "name": "mioflow", "outcome": "ok", "seconds": 1.0, "plots": [str(b)]}])
        await pilot.pause()
        thumb = app.screen.query_one("#figure-thumb")
        # `drawable` gates it: None on a cell tier (which is what `run_test` has), the newest
        # figure on a graphics one. Asserting the GATE rather than an outcome that depends on
        # the terminal the suite happens to run in.
        from manyruns.tui import images
        from manyruns.tui.figure import drawable

        assert thumb.image == drawable(str(b))
        assert (thumb.image is None) is not images.is_pixel_perfect()
        # both are still listed by path — the picture shows one, the list accounts for all
        pane = _unbroken(_pane(app, "#figures"))
        # ONE figure's path, not every figure's — the listing is a one-line INDEX now
        # (`figure.trajectory`), and the path shown belongs to the selected figure. `a` is
        # named by a mark on the strip; only `b` is spelled out.
        assert "b.png" in pane and "a.png" not in pane
        assert pane.count("○") + pane.count("◉") == 2, "one mark per figure on the index"


def _render(renderable, width: int = 40) -> str:
    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=width, legacy_windows=False).print(renderable)
    return buf.getvalue()


# ── the geometry, which is narrate's split and this screen's ink ─────────────
def test_an_absent_metric_is_a_row_and_carries_its_reason():
    """`geometry_sections`' own docstring: absences are rows, not gaps — `suite.measure`
    declares every metric whether or not it could be computed, and a renderer that skips `None`
    throws away the difference between "no engine", "no embedding" and "it raised". Mutation-
    checked: dropping the `not row.measured` branch from `geometry_view` fails this.

    READ OFF THE RENDERABLE RATHER THAN OFF A PANE, which is this commit's change to it and not
    a weakening: `#geometry-pane` is `#answer-pane` now, so no slot on this screen draws
    `geometry_view` and a pane assertion would have nothing to read. The claim was always the
    renderer's. 80 columns because that is what a stacked pane gets at an 80-column terminal
    (`RunScreen.NARROW`'s note) and nothing in it has to fold there; the narrow case has its
    own test. The screen-level half is now
    `test_the_freed_slot_is_the_answers_and_the_record_is_not_drawn_twice`.
    """
    g = {"n_samples": 60, "n_embedded": 60, "n_features": 8, "final_dim": 3,
         "lid": 1.816, "kernel_sparsity": None,
         "kernel_sparsity_note": "needs a fitted LatentModule",
         "suite_null": "data-null unimplemented — comparable, not a finding"}
    pane = _words(_render(run_screen.geometry_view({"g_vector": g}), width=80))

    assert "kernel_sparsity" in pane
    assert "not measured" in pane and "needs a fitted LatentModule" in pane
    # the caption that keeps the numbers from being read as evidence, verbatim
    assert "data-null unimplemented" in pane
    # and the sections narrate decided on, not sections this screen invented
    for section in narrate.geometry_sections(g):
        assert section.title in pane
    # A MEASURED ROW, POSITIVELY, on the same numbers the freed-slot test then asserts are NOT
    # on the screen. Without this the negative there is vacuous: `dimensions 8 → 3` missing from
    # a screen proves the pane dropped the g-vector only if that string is what the renderer
    # would have put in it, and this is where that is established. narrate's label and narrate's
    # arrow, from `n_features` and `final_dim` — this screen composes neither.
    assert "dimensions 8 → 3" in pane


@drives
async def test_the_answer_pane_says_nothing_rather_than_promising_something():
    """Before a question there is no answer. The empty state states what is true now; a pane
    that wrote "your answer appears here" would be stating aspiration as behaviour — and at this
    commit it would name a channel that does not exist yet, since the `?` line lands in the
    commit after this one. The same choice `geometry_view`'s "nothing recorded yet" made in this
    slot, for the same reason, which is why that sentence must not still be on screen: two empty
    states in one pane is the pane not having decided what it holds."""
    app = _Harness(RunScreen(EMBED))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        text = _screen_text(app)

    assert RunScreen.AT_REST in text
    assert "nothing recorded yet" not in text


@drives
async def test_the_freed_slot_is_the_answers_and_the_record_is_not_drawn_twice():
    """Spec §0 and §5: the top-right pane was a second, narrower account of a record the trace
    layer now holds, so it goes and the answer takes the slot.

    BY ID AND BY CONTENT, because either alone passes a rename that changed nothing: an id
    assertion is satisfied by a pane still painting the g-vector under a new name, and a content
    assertion is satisfied by a pane that was simply never filled in. The g-vector and the tuned
    pair are both set here first, so "not drawn" is a fact about a screen that had both to draw.
    """
    from textual.css.query import NoMatches

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = {"g_vector": {"n_samples": 60, "n_embedded": 60, "n_features": 8,
                                       "final_dim": 3, "lid": 1.816}}
        screen.moved = ({"trustworthiness": 0.50}, {"trustworthiness": 0.80})
        screen.tune_step = "phate"
        screen.sync()
        await pilot.pause()
        title = app.screen.query_one("#answer-pane").border_title
        for gone in ("#geometry-pane", "#geometry"):
            with pytest.raises(NoMatches):
                app.screen.query_one(gone)
        text = _words(_screen_text(app))
        answer = _words(_pane(app, "#answer"))

    # `dimensions 8 → 3` is narrate's own label for `final_dim` — what `geometry_view` drew in
    # this slot, asserted positively on these same numbers by
    # `test_an_absent_metric_is_a_row_and_carries_its_reason` so that this line is a fact about
    # the pane and not about a label that moved. THE RECORD IS GONE from the screen: it is the
    # sink's now.
    assert "dimensions 8 → 3" not in text
    # THE TUNED PAIR IS NOT. "what changed in phate (2D)" is the tune loop's feedback, not a
    # record — a loop that shows nothing after `tune` is not a loop — so it keeps its ink, in
    # this pane, above the rest line. The first cut of this commit dropped both and rewrote
    # the integration test that noticed to read a renderer nobody calls; this is the contract
    # that test was protecting.
    assert "what changed in phate" in answer
    # and beneath it, what the slot says with nothing asked of it yet
    assert RunScreen.AT_REST in answer
    # THE BORDER WENT WITH THE CONTENT. It said what this run recorded; it names the model that
    # would answer now. That it is the model's REAL name, and not a literal, is
    # `test_the_answer_pane_is_titled_with_the_model_that_would_answer`.
    assert title == agents.ask_model()
    assert "what this run recorded" not in text


@drives
async def test_the_answer_pane_is_titled_with_the_model_that_would_answer(monkeypatch):
    """Spec §4: this pane's border title is the model's name. §8 is why that is not decoration —
    an answer here is GENERATED and the deltas beside it are MEASURED, in one typeface, and the
    border is the only thing on the screen that says which of the two a reader is looking at.
    `agents.ask_model`'s own docstring carries the rest: a name shown that is not the name called
    would be worse than showing none.

    DRIVEN THROUGH THE OVERRIDE, because `== "qwen3:8b"` also passes against a literal typed
    into `on_mount`, which is exactly the bug worth catching. `$MANYRUNS_ASK_MODEL` moves the
    model the ask will really use, so the title has to move with it.

    AND WITH THE SERVER DOWN, second half: the title is who WOULD answer, not whether anything
    is listening. `ask_available` is patched `False` rather than left to whatever this machine
    is running, so the assertion means the same thing on a laptop with ollama up and in CI with
    nothing on 11434 — and it fails the moment someone gates the border on a socket, which would
    be a second account of a state the controller row is spec'd to hold (§4).
    """
    monkeypatch.setenv("MANYRUNS_ASK_MODEL", "llama3.2:1b")
    app = _Harness(RunScreen(EMBED))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        overridden = app.screen.query_one("#answer-pane").border_title

    monkeypatch.delenv("MANYRUNS_ASK_MODEL")
    monkeypatch.setattr(agents, "ask_available", lambda: False)
    app = _Harness(RunScreen(EMBED))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        with_nothing_listening = app.screen.query_one("#answer-pane").border_title

    assert overridden == "llama3.2:1b"
    assert with_nothing_listening == agents.ASK_MODEL


# ── live: the screen moves DURING the run, not only at the end ───────────────
@drives
async def test_the_screen_repaints_while_the_run_is_still_going():
    """The property the whole component exists for. The run is held open on an `Event` so there
    is a middle to observe: the step must read `running…` while the worker is still inside
    `start`, and `screen.results` must still be None when that is read.

    What this adds over the tests above is the MIDDLE. They all report through the same
    listener, so unwiring it (measured: `self.feed.listener = self._reported` deleted) kills
    nine of them — but every one of those reads the screen after its last report, and a screen
    that painted only when the run ended would look identical to them. This one reads it while
    the worker is still blocked, which is the only place the difference shows.
    """
    import threading

    reported, release = threading.Event(), threading.Event()

    def start(on_step):
        on_step({}, [{"index": 0, "name": "phate", "state": "running", "started": 1.0}])
        reported.set()
        release.wait(10)
        on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 2.0}])
        return {"g_vector": {"n_samples": 60, "n_embedded": 60}}

    screen = RunScreen(EMBED, start=start)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        for _ in range(200):                      # ≤ 2 s, then the test fails rather than hangs
            await pilot.pause()
            if reported.is_set():
                break
            await asyncio.sleep(0.01)
        assert reported.is_set(), "the worker never reported its first step"
        await pilot.pause()
        mid, mid_results = _screen_text(app), screen.results

        release.set()
        await _settled(pilot, app)
        end = _screen_text(app)

    assert narrate.step_mark({"state": "running"})[1] in mid, "mid-run, the step reads running"
    assert mid_results is None, "and the run had not finished when that was read"
    assert narrate.step_mark({"outcome": "ok"})[1] in _words(end)
    assert "1 of 1 steps" in end


# ── a real run, on a real thread ─────────────────────────────────────────────
@drives
async def test_a_real_run_fills_the_screen_in_and_tells_the_app_it_ended(tmp_path):
    """End to end: a real in-process run, `embed` on a 60×8 array, driven by the screen's own thread
    worker. What this pins that a stub cannot:

      * the observer arrives on the WORKER's thread and still repaints — the marshal is
        `post_message`, which branches on the thread id internally;
      * `RunFeed.observer_errors` stays empty, which is the only evidence the listener did not
        raise: `runner._report` swallows whatever an observer throws, so a screen bug would
        otherwise vanish and the test would still pass;
      * the per-step deltas, the figure that was actually written to disk, and the g-vector
        assembled by `_finalize` all reach the screen;
      * the app hears `Finished`, which is the seam component 2 navigates on.

    `manylatents` is required for the DELTAS half specifically, and only here. A real PHATE fit
    needs `phate`; the per-step geometry it reports comes from `suite.live()`, which reads
    manylatents' metric registry — so with phate installed and manylatents absent this ran green
    end to end and then failed on `rows[0].deltas == ()`. That is an absent metric suite, not a
    screen defect, and CI deliberately does not pull the engine chain (it would bring torch).
    """
    pytest.importorskip("manylatents", reason="the deltas are the metric suite's, not phate's")
    screen = RunScreen(EMBED, start=_real_start(tmp_path))
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        await _settled(pilot, app)
        steps, figures = _pane(app, "#steps"), _pane(app, "#figures")
        answer = _pane(app, "#answer")
        text = _screen_text(app)

    assert screen.feed.observer_errors == []
    assert screen.error is None
    assert "1 of 1 steps" in text

    # the step, settled, with a real duration
    glyph, word = narrate.step_mark({"outcome": "ok"})
    assert f"{glyph} phate" in _words(steps) and word in steps

    # what the step moved — narrate's rows, not this screen's arithmetic
    rows = screen.feed.rows()
    assert rows[0].deltas, "a real phate step records geometry deltas"
    flat = _words(steps)
    for delta in rows[0].deltas:
        assert delta.label in flat and delta.value in flat

    # the figure it drew, on disk, attributed to its step
    assert rows[0].plots, "a real phate step writes a plot"
    from pathlib import Path

    assert Path(rows[0].plots[0]).is_file()
    assert Path(rows[0].plots[0]).name in _words(figures)

    # the g-vector `_finalize` assembled
    assert app.finished and app.finished[0].results is screen.results
    assert screen.results["g_vector"]

    # WHERE THE PANE'S FACT WENT, AND WHAT IS ASSERTED IN ITS PLACE — not a like-for-like move,
    # so it is spelled out. This used to read `input → output`, one of `narrate.geometry_sections`'
    # own section titles, out of `#geometry`. That slot is the answer's, so NO PANE draws the
    # settled g-vector on a real run any more and those titles are now asserted on a synthetic
    # g-vector only (`test_an_absent_metric_is_a_row_and_carries_its_reason`, off the renderable).
    # What a real run measured is checked two ways here instead: the g-vector `_finalize`
    # assembled still splits into narrate's sections — data, not ink — and the per-step deltas
    # reach `narrate.record_for_prompt`, which is where a model reads them (spec §3). Those
    # deltas are also on `#steps` fifteen lines above; this is the serializer's copy of them.
    assert narrate.geometry_sections(screen.results["g_vector"])
    record = narrate.record_for_prompt(recipe=EMBED, steps=rows)
    for delta in rows[0].deltas:
        assert delta.label in record and delta.value in record

    # …and the ink half of it, on a REAL settled run rather than only on the synthetic one the
    # freed-slot test builds: the top-right pane holds the answer's empty state, and none of the
    # sections `_finalize`'s own g-vector splits into are in it. `sync()` ran on every report and
    # again at the end, so this is a pane that had the whole record in reach and painted none of
    # it — which is the pane being the answer's, not the pane being unwired.
    assert RunScreen.AT_REST in _words(answer)
    for section in narrate.geometry_sections(screen.results["g_vector"]):
        assert section.title not in _words(answer)


@drives
async def test_a_run_that_raises_says_so_without_inventing_an_outcome_for_the_step():
    """A worker that dies silently leaves a screen that looks like a run still in progress. The
    error reaches the heading and the app hears `Finished`, so navigation is not stranded.

    And the step it died on is left reading exactly what its record says — `running`. That is
    the interesting half. Marking it `error` would be more comfortable and would be the screen
    INVENTING an outcome: `apply_step` never raises, so a record stuck at `running` means the
    run died somewhere else entirely (`close()`, the engine failing to build), and "phate
    errored" would be a claim about a step nothing measured. `narrate._LIVE_MARKS`' own comment
    is the rule — `running` is a live state, not an outcome — and a screen that promotes one to
    the other is a fourth account of the run. What the user reads is true as written: phate was
    running when the run stopped, and here is why it stopped.
    """
    def boom(on_step):
        on_step({}, [{"index": 0, "name": "phate", "state": "running", "started": 1.0}])
        # markup on purpose: the heading interpolates this into rich markup, and an exception
        # message is exactly where a bracketed word turns up. Measured, `expected [ok] got
        # [error]` renders as `expected  got ` with both words deleted.
        raise RuntimeError("the fit exploded: expected [ok] got [error]")

    screen = RunScreen(EMBED, start=boom)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _settled(pilot, app)
        text = _screen_text(app)
        # asserted INSIDE the context: leaving it stops the app, so `is_running` outside would
        # be False whether or not the exception took it down.
        assert app.is_running, "a failed run must not take the app down"

    assert screen.error == "RuntimeError: the fit exploded: expected [ok] got [error]"
    assert "expected [ok] got [error]" in _words(text), "markup ate part of the message"
    assert app.finished and app.finished[0].results is None
    # the record said `running` and still says `running` — no outcome was invented for it
    glyph, word = narrate.step_mark({"state": "running"})
    assert f"{glyph}  phate" in text and word in text
    assert screen.feed.rows()[0].state == "running"


# ── quitting must give the terminal back ─────────────────────────────────────
@drives
async def test_the_run_thread_never_holds_the_process_open():
    """REPORTED FROM THE SHIPPED BUILD: "once i select a workflow and quit with ctrl c ... it
    just hangs after i try to run it again, it wont let it go thru."

    The cause is not manyruns's. Textual runs a `@work(thread=True)` worker through
    `loop.run_in_executor(None, ...)` (worker.py:326) — the DEFAULT `ThreadPoolExecutor`, whose
    threads are non-daemon and are joined before the process can exit. Measured with a bare
    Textual app, no manyruns involved, whose worker sleeps 15 s and which is told to quit at
    0.5 s: **`app.run()` returned at 15.01 s and the process at 15.13 s.** A real fit is longer
    than that, so quitting left the terminal hostage to a run the user had just abandoned — and
    the next `manyruns` launched behind it looked like it hung too.

    Same measurement after this fix, on the real `RunScreen`: **process exited at 0.81 s.**

    Asserted as a PROPERTY rather than by timing a subprocess, because the property is the
    mechanism: a daemon thread cannot hold interpreter shutdown, and nothing else here can
    become non-daemon by accident.
    """
    running, release = threading.Event(), threading.Event()

    def slow(on_step):
        running.set()
        release.wait(30)                       # stands in for a fit that outlives the quit
        return {"g_vector": {}, "steps": []}

    screen = RunScreen(EMBED, start=slow)
    app = _Harness(screen)
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert running.wait(5), "the run never started"
            worker = screen.worker_thread
            assert worker is not None and worker.name == "manyruns-run"
            assert worker.daemon, "a non-daemon run thread blocks process exit"
    finally:
        release.set()
        if screen.worker_thread is not None:
            screen.worker_thread.join(timeout=5)
            assert not screen.worker_thread.is_alive()


# ── the gate: stepping a run from the screen (backlog item 9) ─────────────────────────────
@drives
async def test_the_screen_answers_the_loop_and_the_run_completes(tmp_path):
    """END TO END through the REAL `tune.run_tune_loop`, on the screen, with no terminal.

    The loop runs on `RunScreen`'s own daemon thread and BLOCKS on `gate.read`; the pilot types
    into the ask row and the worker wakes. Retry once, then accept — the same script the REPL
    takes — and the run has to finish rather than deadlock.

    `mock` so this needs no private stack: what is under test is the plumbing between a blocked
    worker and a screen, not a fit.
    """
    from manyruns import app as _app
    from manyruns import tune
    from manyruns.session import Session
    from manyruns.tui.gate import Gate
    from textual.widgets import Input

    recipe = _app.load_recipe("cflows")
    gate = Gate()

    def start(_on_step):
        session = Session(project="p", engine="mock", modality="scrna",
                          recipe=recipe, out_dir=tmp_path)
        step = tune.parse_step_line("phate knn=5", session.engine)
        tune.run_tune_loop(session, step, gate.read, gate.write)
        return session.close()

    screen = RunScreen(recipe, start=start, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        for reply in ("r", "knn=40", "a"):
            for _ in range(400):                       # wait for the loop to ASK
                if gate.prompt is not None:
                    break
                await pilot.pause()
                await asyncio.sleep(0.01)
            assert gate.prompt is not None, f"the screen never showed a question before {reply!r}"
            assert app.screen.query_one("#ask").display is True, "the ask row is hidden"
            field = app.screen.query_one("#ask-input", Input)
            field.value = reply
            await pilot.press("enter")
            await pilot.pause()

        for _ in range(400):
            if app.finished:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)

        assert app.finished, "the run never completed — the gate deadlocked"
        assert app.screen.query_one("#ask").display is False, "the row outlived the question"

    assert screen.error is None
    assert gate.transcript, "the loop said nothing — `write` never reached the screen"


@drives
@pytest.mark.parametrize("gesture", ["t_then_arrow", "arrow_at_decision", "typed_at_decision", "rejected_then_arrow",
                                     "relative_at_decision"])
async def test_the_strip_and_typed_params_reach_the_loop(tmp_path, gesture):
    """Exercise all entry gestures against params actually received by Session.step.

    The old regression sent `t` first and missed the first-question trap. Both prompts now
    accept parameters, including knobs absent from the shipped embed recipe's overrides.
    """
    from manyruns import app as _app
    from manyruns import tune
    from manyruns.session import Session
    from manyruns.tui.gate import Gate
    from textual.widgets import Input

    recipe = _app.load_recipe("embed")
    gate = Gate()
    seen: list = []

    def start(_on_step):
        session = Session(project="p", engine="mock", modality="scrna",
                          recipe=recipe, out_dir=tmp_path)
        step = next(s for s in recipe["steps"] if s["name"] == "phate")
        issued = session.step

        def spy(action):
            seen.append(dict((action or {}).get("params") or {}))
            return issued(action)

        session.step = spy
        # Exactly what `_stepped_run` does: arm, then hand the loop the gate's I/O.
        gate.arm(step)
        tune.run_tune_loop(session, step, gate.read, gate.write)
        return session.close()

    screen = RunScreen(recipe, start=start, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()

        async def wait_for_the_question():
            for _ in range(400):
                field = app.screen.query_one("#ask-input", Input)
                if not field.disabled and field.has_focus:
                    return field
                await pilot.pause()
                await asyncio.sleep(0.01)
            raise AssertionError("the screen never showed a question")

        await wait_for_the_question()
        assert app.screen.tunables, "the strip was not armed for a step that offers knobs"
        if gesture == "t_then_arrow":
            await pilot.press("t", "enter")
            await wait_for_the_question()
            assert screen._question_kind == "params"
        else:
            assert screen._question_kind == "decision"
        if gesture == "rejected_then_arrow":
            await pilot.press(*"t=20 unknown=1", "enter")
            await wait_for_the_question()
            assert len(seen) == 1
            assert dict(screen.tunables)["t"] == "auto"
            assert any("unknown parameter" in line for line in gate.transcript)
        if gesture == "relative_at_decision":
            await pilot.press(*"increase the k by one")
        elif gesture == "typed_at_decision":
            await pilot.press(*"t=20")
        else:
            await pilot.press("shift+tab")
            assert screen.focused.id == "tune-strip"
            await pilot.press("right")
        composed = screen.pending_text()
        assert composed == ("increase the k by one" if gesture == "relative_at_decision" else "t=20")
        assert dict(screen.tunables)["t"] == "auto", "a draft changed accepted values"
        await pilot.press("enter")

        await wait_for_the_question()
        if gesture == "relative_at_decision":
            assert dict(screen.tunables)["knn"] == 6
        else:
            assert dict(screen.tunables)["t"] == 20
        await pilot.press("a", "enter")                 # accept the tuned attempt

        for _ in range(400):
            if app.finished:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        assert app.finished, "the run never completed"

    assert screen.error is None
    assert len(seen) == 2, f"expected an attempt and a retry, got {seen}"
    assert seen[0] != seen[1], (
        f"the retry re-fitted identical params — the strip never reached the loop: {seen}")
    name = "knn" if gesture == "relative_at_decision" else "t"
    assert seen[1].get(name) != seen[0].get(name), (
        f"{name!r} did not move between attempts: {seen}")


@drives
async def test_the_gate_arms_the_strip_from_the_step_and_never_from_the_prompt():
    """The driver says WHICH step is about to be tuned, and it says it as a dict.

    This is the channel `on_run_screen_ask` could not be: that handler is explicit that it will
    not detect anything "from the prompt STRING", and the strip is built from the step's params
    (`params.tunable_for`), which no sentence carries. So `Arm` exists beside `Ask` rather than
    the screen parsing one into the other.

    Sent through the GATE, not by the driver reaching for `app.screen`: `_stepped_run` runs on
    the run worker, and the gate's callbacks are `post_message` — safe from any thread and
    dropped on a closed pump. This drives it from the app's own thread, which `post_message`
    also handles (it branches on the calling thread internally).
    """
    from manyruns.tui import params as _params
    from manyruns.tui.gate import Gate

    gate = Gate()
    app = _Harness(RunScreen(EMBED, gate=gate))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.screen.tunables == [], "the strip was armed before anything asked"

        step = {"name": "phate", "group": "latent", "params": {"knn": 5}}
        gate.arm(step)
        await pilot.pause()
        assert app.screen.tunables == _params.tunable_for(step)

        # …AND A STEP THIS PRODUCT OFFERS NOTHING FOR ARMS AN EMPTY STRIP rather than keeping the
        # last one on screen. `_stepped_run` never arms for such a step, but a stale strip is the
        # failure that would look like it had.
        gate.arm({"name": "normalize", "group": "prep", "params": {}})
        await pilot.pause()
        assert app.screen.tunables == []
        assert app.screen.query_one("#tune-strip").display is False


@drives
async def test_the_screen_records_the_g_vector_either_side_of_a_tuned_step():
    """`Moved` is a REPORT, not a derivation: only the driver knows where one step's tuning began
    and ended, because `run_tune_loop` may have run the step three times in between. Copies, so a
    later step mutating `session.g` in place cannot rewrite what was already reported."""
    from manyruns.tui.gate import Gate

    gate = Gate()
    app = _Harness(RunScreen(EMBED, gate=gate))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.screen.moved is None, "a screen with nothing tuned claims a movement"

        gate.moved({"trustworthiness": 0.10}, {"trustworthiness": 0.90})
        await pilot.pause()
        assert app.screen.moved == ({"trustworthiness": 0.10}, {"trustworthiness": 0.90})


# ── what changed (2D) — the moved panel ──────────────────────────────────────
def test_only_the_metrics_that_moved_are_drawn():
    """The g-vector is 23-29 keys and a tune changes a handful. Drawing all of them makes the
    reader find the difference; drawing the difference is the whole job of this pane."""
    from manyruns.tui.run import moved_view

    out = moved_view({"trustworthiness": 0.5, "continuity": 0.9},
                     {"trustworthiness": 0.8, "continuity": 0.9})
    text = _render(out)
    assert "trustworthiness" in text
    assert "continuity" not in text, "an unchanged metric is not news"


def test_nothing_moved_draws_nothing_rather_than_an_empty_frame():
    from manyruns.tui.run import moved_view

    assert moved_view({"a": 1.0}, {"a": 1.0}) is None


def test_the_biggest_relative_move_is_first_and_the_list_is_capped():
    """Ranked by RELATIVE movement, because the suite mixes correlations near 1 with counts in
    the thousands and an absolute rank would only ever show the counts."""
    from manyruns.tui.run import moved_view

    before = {"tiny": 0.10, "huge": 1000.0, "mid": 1.0}
    after = {"tiny": 0.20, "huge": 1100.0, "mid": 1.5}
    text = _render(moved_view(before, after, cap=2))
    assert text.index("tiny") < text.index("mid")
    assert "huge" not in text, "capped at two, and huge moved least in relative terms"


def test_a_metric_that_appeared_or_vanished_is_shown_and_not_divided_by():
    from manyruns.tui.run import moved_view

    text = _render(moved_view({"a": 1.0}, {"a": 1.0, "b": 2.0}))
    assert "b" in text
    text = _render(moved_view({"a": 1.0, "b": 2.0}, {"a": 1.0}))
    assert "b" in text


def test_the_caption_admits_the_projection_it_measured_and_names_its_step():
    """TWO DIFFERENT NUMBERS UNDER ONE METRIC NAME, STACKED, unless the caption separates them.

    The sidecars hold `emb[:, :2]` (`io._save_scatter`, sliced again at `figspec.save`), so
    this panel is measured on the PLOTTED two columns while `geometry_view` — drawn directly
    beneath it in the same pane — reports the full embedding. A reader comparing the two halves
    must not have to discover that difference; the direction is honest either way, the absolute
    values are not comparable.

    The step is named because a run with two tuned steps draws this twice.
    """
    from manyruns.tui.run import moved_view

    text = _render(moved_view({"trustworthiness": 0.5}, {"trustworthiness": 0.8}, step="phate"),
                   width=80)
    assert "what changed in phate (2D)" in " ".join(text.split())
    bare = _render(moved_view({"a": 1.0}, {"a": 2.0}), width=80)
    assert "what changed (2D)" in " ".join(bare.split())


def test_the_caption_costs_one_row_at_the_width_the_pane_actually_has():
    """`RunScreen.NARROW`'s note records the pane's own reference widths: 36 columns at a
    100-column terminal's 3:2 split, 28 at an 80-column one. A caption that wraps there costs a
    row of the picture — the argument `figures_view` makes for bounding its own caption, and the
    reason `MOVED_CAP` exists.

    THIS TEST EXISTS BECAUSE ITS ABSENCE SHIPPED A REGRESSION. The first disclosure caption was
    "what the tune changed in phate (in the plotted 2D)" — 50 characters, two lines at BOTH
    reference widths — and every test rendered at 80, 100 or 200 columns, so none of them saw
    the pane's actual ~36.

    The step names are the ones that can actually reach this caption: `_stepped_run` arms only
    where `params.TUNABLE` has a row, so `leiden` is the longest a run can hand it.
    """
    from manyruns.tui import params as _params
    from manyruns.tui.run import moved_view

    for width in (28, 36):
        for step in ("", *_params.TUNABLE):
            out = moved_view({"trustworthiness": 0.9}, {"trustworthiness": 0.6}, step=step)
            first = _SGR.sub("", _render(out, width=width)).splitlines()[0]
            assert len(first.rstrip()) <= width, (step, width, first)
            assert "2D" in first, "the projection disclosure must survive on the first line"


def test_an_over_cap_metric_label_overflows_rather_than_wrapping_the_number_away():
    """`geodesic_distance_correlation` is 29 characters and `_LABEL_CAP` is 20, so its row is
    over the cap at every width this pane sees. `geometry_view` deliberately lets such a label
    overflow its field — "that row loses its alignment and keeps every character" — and this
    renderer shares the cap, so it must behave the same rather than truncating or eating the
    value. Checked at the pane's real widths, not a terminal's.

    MEASURED, both renderers, 2026-08-30: at 28 the label breaks mid-token after
    `…correlatio` and the value follows on the next line; at 36 the row survives and only the
    value wraps. Identical in both, which is the property — so `_unbroken` is the right reader
    here for the same reason it is for a path, and nothing needed changing.
    """
    from manyruns.tui.run import moved_view

    for width in (28, 36):
        text = _unbroken(_render(moved_view({"geodesic_distance_correlation": 0.9},
                                            {"geodesic_distance_correlation": 0.6}), width=width))
        assert "geodesic_distance_correlation" in text, "the label lost characters"
        assert "0.9→0.6" in text, "the value was folded away"


@drives
async def test_arming_the_next_step_drops_the_last_one_s_readout():
    """A run with two tuned steps arms twice, and `_tuned_apart` returns None for a step accepted
    first try. Without the clear, the FIRST step's pair is still on the screen when the second
    step is armed — a real readout attributed to the wrong step the moment anything draws it.

    ASSERTED ON THE STATE, because after this commit nothing draws it: the pane that did is the
    answer's. The bug is the same one and it is still live — `self.moved` and `self.tune_step`
    are still written by the gate — so the clear keeps its test rather than going with the ink.
    """
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.pause()
        screen.arm_tuning({"name": "phate", "group": "latent", "params": {"knn": 5}})
        screen.moved = ({"trustworthiness": 0.50}, {"trustworthiness": 0.80})
        screen.sync()
        await pilot.pause()
        assert screen.moved is not None and screen.tune_step == "phate"

        screen.arm_tuning({"name": "mioflow", "group": "lightning", "params": {}})
        await pilot.pause()

    assert screen.moved is None, "the last step's pair survived the next step arming"
    assert screen.tune_step == "mioflow", "and the caption's step is the one just armed"


# `test_the_moved_panel_is_drawn_above_the_settled_geometry` WAS HERE, and it is deleted rather
# than retargeted. Its claim was an ORDER inside one pane — the tuned pair over the settled
# g-vector, "one pane, two claims" — and that pane is the answer's now, so there is no surviving
# place for the order to be true. Spec §7.4 says to retarget it to the prompt serializer, and
# that is not available either: `narrate.record_for_prompt` carries the per-step deltas but
# neither the settled g-vector nor the tuned pair, so nothing there can carry this. What the two
# renderers still do separately is pinned by the seven `moved_view` tests above (the whole
# `what changed (2D) — the moved panel` section) and, for the g-vector half, by
# `test_an_absent_metric_is_a_row_and_carries_its_reason` — which asserts `dimensions 8 → 3`
# POSITIVELY off `geometry_view`, the one assertion this deletion would otherwise have dropped
# with nothing standing in its place, and which the freed-slot test's negative form depends on to
# mean anything. What the slot does now is
# `test_the_freed_slot_is_the_answers_and_the_record_is_not_drawn_twice`.


# ── the corpus line: what a finished tune loop left on disk ──────────────────
def test_the_corpus_line_counts_the_file_and_not_the_session(tmp_path):
    """A number that only goes up while you watch is a progress bar. The claim is that a
    corpus EXISTS on disk, so the count is read from it — including rows this session did not
    write."""
    from manyruns import decisions

    for i in range(3):
        decisions.append(offered=[{"a": 1}, {"b": 2}], chosen=f"r{i}",
                         surface="tune", out_dir=tmp_path)
    line = run_screen.corpus_line(tmp_path, offered=3, chosen="phate knn=40")
    assert "3 offered" in line
    assert "knn=40" in line
    assert "3 rows" in line


def test_a_cancelled_loop_says_so_rather_than_naming_a_choice(tmp_path):
    """`chosen=None` is a real row — 'none of these worked' is a signal, and rendering it as a
    choice would misreport the one outcome the corpus most needs to keep honest."""
    line = run_screen.corpus_line(tmp_path, offered=2, chosen=None)
    assert "none kept" in line


def test_an_unreadable_corpus_reports_the_decision_without_a_count(tmp_path):
    """The run is the artifact; a corpus that cannot be read must not take the line down.

    `decisions.read` yields NOTHING for a directory that does not exist — it returns early
    rather than raising — so the guard cannot be `except OSError` alone. Zero is dropped as
    well, and this test is what pins that: a line that has just announced a written row and
    then says `0 rows` is reporting the one number here that is not a measurement.
    """
    line = run_screen.corpus_line(tmp_path / "nope", offered=2, chosen="phate knn=40")
    assert "2 offered" in line and "rows" not in line


@drives
async def test_leaving_mid_question_releases_the_worker(tmp_path):
    """Unmounting must not strand a thread in `queue.get()`. It is a daemon thread so the process
    still exits, but `run_tune_loop`'s `finally` — which restores the metric suite on a session
    the user keeps — would never run."""
    import threading

    from manyruns import app as _app
    from manyruns import tune
    from manyruns.session import Session
    from manyruns.tui.gate import Gate

    recipe = _app.load_recipe("cflows")
    gate = Gate()
    done = threading.Event()

    def start(_on_step):
        session = Session(project="p", engine="mock", modality="scrna",
                          recipe=recipe, out_dir=tmp_path)
        try:
            tune.run_tune_loop(session, tune.parse_step_line("phate", session.engine),
                               gate.read, gate.write)
            return session.close()
        finally:
            done.set()

    app = _Harness(RunScreen(recipe, start=start, gate=gate))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        for _ in range(400):
            if gate.prompt is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        assert gate.prompt is not None, "the loop never asked"

    assert done.wait(timeout=10), "the worker is still blocked after the screen went away"
    assert gate.closed


@drives
async def test_the_ask_input_never_swallows_the_screens_own_keys():
    """A REGRESSION THIS PORT CAUSED AND ITS FIX, pinned so it cannot come back.

    An `Input` is focusable, so merely mounting one put it in the focus chain and it took the
    screen's bindings with it — measured, `f`/`s`/`o` stopped working the moment the widget
    existed, and four `test_tui_figure` tests caught it. `display=False` was not enough: display
    stops a widget being painted, `disabled` takes it out of the chain.

    Asserted as the PROPERTY (the field is out of the chain whenever nothing is being asked)
    rather than by pressing `f` again, so it holds for bindings this screen gains later.

    `action_ask` opens the field deliberately and is the one exception, which is why it closes it
    again on `escape` rather than leaving it enabled: the property here is what the screen
    returns to, not a state it can never leave.
    """
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        field = app.screen.query_one("#ask-input", Input)

        assert field.disabled is True, "the field is in the focus chain with nothing to answer"
        assert app.screen.query_one("#ask").display is False
        assert app.screen.focused is not field


@drives
async def test_the_ask_field_offers_the_loops_current_verbs():
    """The placeholder is the only place the screen SPELLS the vocabulary, so it is the one that
    goes stale silently — it read `a / r / c` for a commit after the verb became `tune`.

    Derived from `gate.resolve` rather than compared to a literal: the assertion is that every
    letter the field advertises actually resolves to a verb the loop accepts, which stays true
    through the next rename instead of pinning today's spelling.
    """
    import re

    from manyruns.tui.gate import resolve

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        placeholder = app.screen.query_one("#ask-input", Input).placeholder

    letters = re.findall(r"\b([a-z])\b", placeholder.split(",")[0])
    assert letters, f"the field advertises no verbs at all: {placeholder!r}"
    assert {resolve(x) for x in letters} == {"accept", "tune", "cancel"}, (
        f"{placeholder!r} advertises {[resolve(x) for x in letters]}")
    assert "r /" not in placeholder, "the pre-rename verb is still on screen"
    assert "knn=40" in placeholder
    assert "declared params" not in placeholder


# ── the figures index: an x axis of steps, walked with the arrow keys ─────────
def test_the_index_is_one_line_however_many_figures_there_are():
    """THE BUG THIS PANE HAD, as a property rather than a screenshot.

    `#figure-thumb` is `height: 1fr` above a caption that was `height: auto` and carried two
    lines per figure. `auto` beats `1fr`, so the caption ate the pane: measured on a 120x30
    terminal, the picture got 1 row at one figure, 1 at three and 1 at six while the listing ran
    to 27 rows and the whole thing scrolled. The figure was squeezed by its own caption and it
    got worse with every step that drew.
    """
    from rich.console import Console

    def height(n):
        rows = [StepView(index=i, name=f"s{i}", state="ok", glyph="✓", word="ok",
                         plots=(f"/p/s{i}.png",)) for i in range(n)]
        buf = io.StringIO()
        Console(file=buf, width=60, legacy_windows=False).print(
            run_screen.figures_view(rows, -1, 60))
        return len([ln for ln in buf.getvalue().splitlines() if ln.strip()])

    one = height(1)
    assert height(6) == one and height(40) == one, \
        "the caption grows with the run again — that is what starved the picture"
    assert one <= 3


def test_the_index_marks_every_figure_and_the_cursor_sits_on_the_shown_one():
    from manyruns.tui.figure import index_line, trajectory

    found = [(f"s{i}", f"/p/{i}.png") for i in range(5)]
    for sel in range(5):
        strip = trajectory(found, sel)
        assert strip.count("◉") == 1, "exactly one cursor"
        assert strip.count("○") == 4
        assert strip.index("◉") == sel * 2, "the cursor sits where the figure is in the order"
        assert index_line(found, sel) == f"s{sel} · {sel + 1} of 5"


def test_a_clipped_index_keeps_the_cursor_visible():
    """A forty-figure run still has to say WHERE you are. The strip clips around the cursor and
    `index_line` says how many are off the end."""
    from manyruns.tui.figure import trajectory

    found = [(f"s{i}", "") for i in range(40)]
    for sel in (0, 20, 39):
        assert "◉" in trajectory(found, sel, width=11)


@drives
async def test_the_arrows_walk_the_figures_and_stop_at_the_ends(tmp_path):
    """Stops rather than wraps. The strip is an ORDER — the trajectory the embedding took
    through the recipe — and one whose last element is next to its first is not one; a reader
    stepping right off the end onto `phate` would read that as the run having looped."""
    a, b = _png(tmp_path / "a.png"), _png(tmp_path / "b.png")
    screen = RunScreen(THREE)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [
            {"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0, "plots": [str(a)]},
            {"index": 1, "name": "mioflow", "outcome": "ok", "seconds": 1.0, "plots": [str(b)]}])
        await pilot.pause()
        pane = app.screen.query_one(run_screen.FiguresPane)

        assert pane.at() == 1, "a fresh pane shows the newest"
        screen.query_one("#figures-pane").focus()
        await pilot.press("left")
        assert pane.at() == 0
        screen.query_one("#figures-pane").focus()
        await pilot.press("left")
        assert pane.at() == 0, "stops at the start rather than wrapping to the end"
        await pilot.press("right")
        assert pane.at() == 1
        await pilot.press("right")
        assert pane.at() == 1, "stops at the end rather than wrapping to the start"


@drives
async def test_a_new_figure_moves_the_pane_until_the_reader_takes_over(tmp_path):
    """Follow-the-newest, until it is not. A live run should advance the picture as steps draw —
    and the moment a reader walks back to step 1, a step finishing must NOT yank them to step 3.
    Same rule a log tail follows, and the reason `selected` is a sentinel and not an index."""
    a, b, c = (_png(tmp_path / f"{n}.png") for n in "abc")
    screen = RunScreen(THREE)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok",
                                  "seconds": 1.0, "plots": [str(a)]}])
        await pilot.pause()
        pane = app.screen.query_one(run_screen.FiguresPane)
        assert pane.at() == 0

        # `on_step` takes the WHOLE row list each time, not a delta — so a step landing is the
        # same list one longer, which is exactly the case this test is about.
        two = [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0, "plots": [str(a)]},
               {"index": 1, "name": "mioflow", "outcome": "ok", "seconds": 1.0,
                "plots": [str(b)]}]
        screen.feed.on_step({}, two)
        await pilot.pause()
        assert pane.at() == 1, "an untouched pane follows the newest figure"

        screen.query_one("#figures-pane").focus()
        await pilot.press("left")                       # the reader takes over
        assert pane.at() == 0
        screen.feed.on_step({}, two + [{"index": 2, "name": "separation", "outcome": "ok",
                                        "seconds": 1.0, "plots": [str(c)]}])
        await pilot.pause()
        assert pane.at() == 0, "a finishing step yanked the reader off the figure they chose"


@drives
async def test_save_and_open_act_on_the_selected_figure_not_the_newest(tmp_path):
    """`s` and `o` are bound on the run screen so the moment someone decides they want a figure
    is the moment they can keep it. Once the pane can show any figure in the run, acting on "the
    newest" would save a picture the reader is not looking at."""
    a, b = _png(tmp_path / "a.png"), _png(tmp_path / "b.png")
    screen = RunScreen(THREE)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [
            {"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0, "plots": [str(a)]},
            {"index": 1, "name": "mioflow", "outcome": "ok", "seconds": 1.0, "plots": [str(b)]}])
        await pilot.pause()

        assert screen._selected_figure() == ("mioflow", str(b))
        screen.query_one("#figures-pane").focus()
        await pilot.press("left")
        assert screen._selected_figure() == ("phate", str(a)), \
            "s and o would have acted on a figure the reader had walked away from"


def test_the_strip_arrows_write_into_the_input_rather_than_answering_directly():
    """ONE ANSWER MECHANISM, NOT TWO (`run.py`'s `on_run_screen_ask`). The strip is an input
    helper: `→` edits the field, and nothing reaches the gate until `enter` submits it the way
    a typed answer does. A screen-level binding that called `gate.answer` would be the second
    mechanism that handler refuses in terms."""
    from manyruns.tui.gate import Gate
    from manyruns.tui.run import RunScreen

    gate = Gate()
    screen = RunScreen({"name": "embed", "steps": [{"name": "phate"}]}, gate=gate)
    screen.arm_tuning({"name": "phate", "params": {"knn": 15}})
    assert dict(screen.tunables)["knn"] == 15
    # Select by name so this test is independent of curated ordering.
    screen.tune_at = [name for name, _ in screen.tunables].index("knn")

    screen.move_value(+1)
    assert screen.pending_text() == "knn=16"
    assert gate.prompt is None, "nothing may reach the gate before enter"

    screen.move_value(+1)
    assert screen.pending_text() == "knn=17"


def test_the_strip_walks_its_rows_and_the_figures_keep_their_own_arrows():
    """Parameter stepping composes a draft; pane-local keys are exercised by the pilots."""
    from manyruns.tui.run import RunScreen

    screen = RunScreen({"name": "embed", "steps": [{"name": "phate"}]})
    # `decay=40.0`, a float, not `40`: `step_value` branches on the VALUE's own type
    # (`test_a_float_knob_moves_by_a_tenth_of_itself` pins the tenth-of-itself step for a float,
    # `test_an_int_knob_stays_an_int...` pins whole-unit steps for an int) — an int `40` would
    # step to `41`, not `44.0`.
    screen.arm_tuning({"name": "phate", "params": {"knn": 15, "decay": 40.0}})
    assert screen.tune_at == 0
    screen.move_row(+1)
    assert screen.tune_at == 1
    screen.move_value(+1)
    assert screen.pending_text() == "decay=44.0"
    last = len(screen.tunables) - 1
    screen.move_row(last + 5)
    assert screen.tune_at == last, "the last row is the last row, not a wrap to the first"


def test_a_step_with_no_offered_knobs_arms_no_strip():
    from manyruns.tui.run import RunScreen

    screen = RunScreen({"name": "embed", "steps": [{"name": "normalize"}]})
    screen.arm_tuning({"name": "normalize", "params": {"target_sum": 10000}})
    assert screen.tunables == []
    assert screen.pending_text() == ""


# ── finished above, live below — where the gate's rows sit ───────────────────
async def _asked(app: App, pilot, gate) -> None:
    """Arm the strip for a step this product offers knobs on, then ask. Two pauses: one for
    `Arm`/`Ask` to be handled, one for the layout to be re-solved with the rows now shown."""
    gate.arm({"name": "phate", "group": "latent", "params": {"knn": 5}})
    await _live_question(pilot, gate)


async def _live_question(pilot, gate, prompt="accept / tune / cancel?", kind="decision"):
    worker = threading.Thread(target=lambda: gate.read(prompt, kind=kind), daemon=True)
    worker.start()
    for _ in range(100):
        await pilot.pause()
        if gate.prompt is not None and not pilot.app.screen.query_one(Input).disabled:
            return worker
        await asyncio.sleep(0.005)
    raise AssertionError("the live question did not open its editor")


@drives
async def test_the_prompt_is_below_the_analysis_because_the_last_line_is_the_one_you_read():
    """The gate's rows follow the panes, on screen and not just in `compose`. Asserted on the
    REGIONS the layout engine solved — a compose order that CSS docked back to the top would
    pass a source-order check and fail this one."""
    from manyruns.tui.gate import Gate

    gate = Gate()
    app = _Harness(RunScreen(EMBED, gate=gate))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _asked(app, pilot, gate)
        panes = app.screen.query_one("#panes").region
        ask = app.screen.query_one("#ask").region
        strip = app.screen.query_one("#tune-strip").region
        assert strip.height and ask.height, "the rows were asked for and not laid out"
        assert strip.y >= panes.y + panes.height, f"strip {strip} is not below panes {panes}"
        assert ask.y >= panes.y + panes.height, f"ask {ask} is not below panes {panes}"
        assert ask.y >= strip.y + strip.height, "the question is not the last thing on screen"


@drives
@pytest.mark.parametrize("height", [30, 24])
async def test_the_prompt_sits_on_the_bottom_edge_without_being_docked(height):
    """`#panes { height: 1fr }` under `RunScreen { layout: vertical }` is the whole mechanism:
    the panes take every row the auto-height rows below them leave, so the question lands on
    the bottom edge at ANY height. Pinned at two heights so a fixed offset could not pass."""
    from manyruns.tui.gate import Gate

    gate = Gate()
    app = _Harness(RunScreen(EMBED, gate=gate))
    async with app.run_test(size=(120, height)) as pilot:
        await pilot.pause()
        await _asked(app, pilot, gate)
        ask = app.screen.query_one("#ask").region
        assert ask.y + ask.height == height, f"ask {ask} does not reach the bottom edge"
        assert app.screen.query_one("#ask").styles.dock in ("", "none"), \
            "the row is docked — the layout should not need it"


@drives
async def test_the_analysis_does_not_move_when_a_question_arrives():
    """The measured jump this order fixes: with the rows above the panes, the panes' top slid
    from y=3 to y=12 at 120×30 the moment a question appeared, under whoever was reading them.
    Now the top holds and only the height gives."""
    from manyruns.tui.gate import Gate

    gate = Gate()
    app = _Harness(RunScreen(EMBED, gate=gate))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        before = app.screen.query_one("#panes").region
        await _asked(app, pilot, gate)
        after = app.screen.query_one("#panes").region
        assert after.y == before.y, f"the panes moved: {before} -> {after}"
        assert after.height < before.height, "the question took no room from the panes"


@drives
async def test_answering_still_hands_the_keys_back_to_the_screen():
    """`on_input_submitted` ends in `focus_next`, and the input is now the LAST focusable
    thing in compose order — so the walk wraps rather than steps. It still has to land on the
    screen's own panes, or `f`/`s`/`o` are lost after every answer."""
    from manyruns.tui.gate import Gate

    gate = Gate()
    app = _Harness(RunScreen(EMBED, gate=gate))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _asked(app, pilot, gate)
        field = app.screen.query_one("#ask-input", Input)
        assert app.screen.focused is field, "the question did not take focus"
        field.value = "a"
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.focused is app.screen.query_one("#steps-pane"), \
            f"focus went to {app.screen.focused!r}, not back to the panes"
        assert field.disabled is True


@drives
@pytest.mark.parametrize("height", [13, 24])
async def test_the_prompt_is_whole_at_the_shortest_terminal_it_claims(height):
    """13 rows is the measured floor: heading (1) + caveats (1) + panes at their 1-row minimum +
    `#said` (1) + a strip of four knobs (6) + the bordered input (3). At 12 the input's bottom border is the
    row that clips — which is stated, not fixed, here."""
    from manyruns.tui.gate import Gate

    gate = Gate()
    app = _Harness(RunScreen(EMBED, gate=gate))
    async with app.run_test(size=(120, height)) as pilot:
        await pilot.pause()
        await _asked(app, pilot, gate)
        ask = app.screen.query_one("#ask").region
        assert ask.y + ask.height <= height, f"ask {ask} is clipped at {height} rows"


# ── the ask channel: a `?` line, an answer in the pane, an event in the sink ──
#: The stub every test here answers with, so an assertion reads as the pane's content rather
#: than as a fixture's. It is the measured demo answer from spec §2's example event, kept
#: verbatim: the one thing this channel is for is prose about THIS run's numbers.
CANNED = "No — 19,024 of 32,738 genes appear in fewer than 3 cells, so leiden would cluster noise."


def _no_sink(monkeypatch):
    """Make `trace.current()` a `NullTracer` no matter what the developer's shell exports.

    The suite must never write a real HOME (`tests/conftest.py` redirects `inspected.HOME` for
    the same reason), and `$MANYRUNS_TRACE=jsonl://~/traces` in a profile is exactly how a test
    run would come to append to one. Both prefixes, because `env.get` falls back to the
    pre-rename spelling.
    """
    for name in ("MANYRUNS_TRACE", "GEOMANCER_TRACE"):
        monkeypatch.delenv(name, raising=False)


def _sink(monkeypatch, tmp_path):
    """Point the tracer at a directory under `tmp_path` and return it. It does NOT exist yet —
    `JsonlTracer` creates it on the first `emit`, which is what makes "no file at all" an
    assertable outcome for an ask that emitted nothing."""
    where = tmp_path / "traces"
    monkeypatch.delenv("GEOMANCER_TRACE", raising=False)
    monkeypatch.setenv("MANYRUNS_TRACE", f"jsonl://{where}")
    return where


def _model_says(monkeypatch, text, *, listening=True):
    """Stand in for the local model. Returns the dict the call was made with, so a test can
    assert on the RECORD the screen handed over — the prompt is the whole feature.

    Patched on the module `tui/run.py` imports (`from manyruns import agents` inside the
    worker), so the seam under test is the real one and no socket is opened anywhere in this
    file: `ask_available` is the TCP probe and `answer` is the HTTP call.
    """
    seen: dict = {}

    def _answer(*, system, prompt):
        seen["system"], seen["prompt"] = system, prompt
        return text

    monkeypatch.setattr(agents, "ask_available", lambda: listening)
    monkeypatch.setattr(agents, "answer", _answer)
    return seen


async def _ask(app, pilot, line: str) -> None:
    """Type one line into the controller field and let the ask worker finish.

    The field is enabled by hand rather than by driving a real gate to a question, because what
    is under test is the CHANNEL and the gate's prompt is a different commit's wire — the tests
    that drive `run_tune_loop` end to end are below. THE SURFACE A PERSON ACTUALLY REACHES is
    covered on its own —
    `test_the_channel_is_open_exactly_while_the_loop_is_waiting_for_an_answer` enables nothing
    and drives the row up the way the gate does, so this shortcut cannot be the reason a channel
    nobody can reach looks tested. `wait_for_complete` is the ask worker's
    (`@work`), which is a Textual worker and therefore IS in `app.workers`, unlike the run's raw
    daemon thread — `_settled`'s docstring carries that distinction and the flake it caused.
    """
    field = app.screen.query_one("#ask-input", Input)
    field.disabled = False
    field.focus()
    field.value = line
    await pilot.pause()
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.pause()


@drives
async def test_a_question_never_answers_the_gate_and_never_takes_the_field_away(monkeypatch):
    """FREESTYLE, and the whole reason the prefix is checked before anything else parses the
    line. A question is not an answer: the gate stays pending, the row stays up, the field stays
    enabled and focused, and the person can now answer the question they asked about.

    THE STRIP IS THE SECOND HALF and it is not hypothetical: `on_input_submitted` runs
    `tune.parse_overrides` over the typed line, so the sentence "should I raise knn=40 here"
    would have moved the knob a person was only asking about — a run tuned by a question.

    Asserted on `gate.answer` itself (spied) rather than on what the loop did with it, because
    a gate with no question pending accepts and discards an answer silently: the failure this
    catches would look like nothing at all from outside.
    """
    from manyruns.tui.gate import Gate

    _no_sink(monkeypatch)
    _model_says(monkeypatch, CANNED, listening=False)   # nothing to ask; the channel still runs
    gate = Gate()
    answered: list = []
    monkeypatch.setattr(gate, "answer", lambda text, **kwargs: bool(answered.append(text)) or True)

    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _live_question(pilot, gate, "accept / tune / cancel >")
        await pilot.pause()
        screen.arm_tuning({"name": "phate", "group": "latent", "params": {"knn": 5}})
        await pilot.pause()
        armed = list(screen.tunables)
        field = app.screen.query_one("#ask-input", Input)
        # the only thing on screen that says the channel exists
        assert f"{run_screen.ASK_PREFIX} a question" in field.placeholder

        await _ask(app, pilot, "? should I raise knn=40 here")
        after = app.screen.query_one("#ask-input", Input)

        assert answered == [], "a question was spent on the gate"
        assert after.disabled is False, "the question disabled the field the gate is waiting on"
        assert app.screen.query_one("#ask").display is True, "the question hid the loop's row"
        assert screen.tunables == armed, "a sentence mentioning knn=40 moved the strip"
        assert after.value == "", "the question stayed in the field for the next enter to re-send"

        # …and the gate is still answerable, which is the point of leaving all of that alone
        after.value = "a"
        await pilot.press("enter")
        await pilot.pause()

    assert answered == ["accept"]


@drives
async def test_the_model_is_handed_the_record_the_screen_is_showing(monkeypatch):
    """SPEC §1 AND §3: one call with the whole record in the prompt, and the record is what the
    person is looking at. A fact that does not reach this string is a fact the model was not
    asked about — and it would answer anyway.

    Every item is state this screen holds and none of it is composed here:
    `narrate.record_for_prompt` renders and `ask_context` gathers, so the model reads the words
    the panes drew. The dataset arrives as a `DataEntry` and reaches the prompt through the
    class's own `to_dict`, which is what makes "a `DataEntry`-shaped mapping" a contract rather
    than a shape this screen invented.

    `said` IS SLICED to `SAID_LINES`: §3 asks for "the last `SAID_LINES` the loop said", the
    pane shows three, and a record carrying a fourth line nobody could see would be the same
    two-accounts failure from the other direction.
    """
    from manyruns.tui.state import DataEntry

    _no_sink(monkeypatch)
    seen = _model_says(monkeypatch, CANNED)
    obs = narrate.Observation(source="pbmc3k_raw.h5ad", shape="single", modality="scrna",
                              n_obs=2700, n_vars=32738)
    entry = DataEntry(name="pbmc3k_raw.h5ad", kind="file", obs=obs)

    screen = RunScreen(EMBED, entry=entry, concerns=["19,024 genes are near-empty"])
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 3.25}])
        for line in ("the first thing it said", "one", "two", "kept attempt 1 as phate@1.png"):
            screen.post_message(RunScreen.Said(line))
        screen.arm_tuning({"name": "phate", "group": "latent", "params": {"knn": 40}})
        await pilot.pause()

        await _ask(app, pilot, "? should I trust leiden here")

    prompt = seen["prompt"]
    assert prompt.startswith("dataset: pbmc3k_raw.h5ad"), prompt[:80]
    assert "2,700 × 32,738" in prompt          # the roster's own size string, via `to_dict`
    assert "recipe: embed" in prompt
    assert "phate" in prompt and "3.25s" in prompt
    assert "knn=40" in prompt                   # the strip, in the form a person could type back
    assert "kept attempt 1 as phate@1.png" in prompt
    assert "19,024 genes are near-empty" in prompt
    # the cap is the screen's: three said lines on screen, three in the record
    assert "the first thing it said" not in prompt
    # THE QUESTION IS OUTSIDE THE RECORD, which is what lets the event carry a hash of one and
    # the text of the other and still describe the whole call.
    assert "should I trust leiden here" in prompt
    # …and everything before it is the record, byte for byte — `record_for_prompt` ends with its
    # own newline, so the separator this splits on is the blank line the question sits after.
    assert narrate.context_sha256(screen.ask_context()) == narrate.context_sha256(
        prompt[:prompt.index("\nquestion: ")])
    assert seen["system"] is run_screen.ASK_SYSTEM


@drives
async def test_the_answer_lands_in_the_pane_and_the_record_gets_exactly_one_event(
        monkeypatch, tmp_path):
    """The demo, end to end with the model stubbed: a `?` line, prose in the pane it freed, and
    ONE event in the sink — spec §2's shape, field for field.

    `context_sha256` is checked against the record this screen would render RIGHT NOW, which is
    the property the whole event rests on: the answer is a function of a page a later reader can
    rebuild. A hash over anything else (the prompt including the question, a summary, the
    screen's own text) would still be 64 hex characters and would join nothing to anything.
    """
    import json

    where = _sink(monkeypatch, tmp_path)
    _model_says(monkeypatch, CANNED)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        # A settled run, so the event has a `run_id` to be filed under — the mid-run case is
        # the test below.
        screen.results = {"run_id": "2d730cc5c591"}
        await _ask(app, pilot, "? should I trust leiden here")
        pane = _words(_pane(app, "#answer"))

    assert CANNED in pane
    assert "? should I trust leiden here" in pane, "the pane holds an answer to nothing"
    assert RunScreen.AT_REST not in pane
    assert RunScreen.THINKING not in pane

    assert [p.name for p in sorted(where.iterdir())] == ["2d730cc5c591.jsonl"]
    lines = (where / "2d730cc5c591.jsonl").read_text().splitlines()
    assert len(lines) == 1, f"one question, {len(lines)} events"
    event = json.loads(lines[0])
    assert event["kind"] == "ask"
    assert event["run_id"] == "2d730cc5c591"
    assert event["agent_id"] == "human"          # nobody said otherwise, and a person typed it
    assert event["at"].endswith("+00:00")
    assert event["question"] == "should I trust leiden here"
    assert event["answer"] == CANNED
    assert event["model"] == agents.ask_model()
    assert event["backend"] == run_screen.ASK_BACKEND
    assert isinstance(event["latency_s"], (int, float)) and event["latency_s"] >= 0
    assert event["context_sha256"] == narrate.context_sha256(screen.ask_context())
    # THE PROMPT ITSELF IS NOT IN THE EVENT (§2): a corpus that stored it would be the same few
    # hundred tokens thousands of times, and the hash plus `record_for_prompt` rebuilds it.
    assert "dataset:" not in lines[0]


@drives
async def test_a_question_asked_before_the_run_ends_is_recorded_under_the_absence_of_an_id(
        monkeypatch, tmp_path):
    """MID-RUN IS WHEN PEOPLE ASK, and mid-run this screen has no run id: `results` is what
    carries one and it is None until `Finished`, while the step records the feed sees carry
    `index`/`name`/`group`/`params`/`outcome` and no identity at all.

    So the event is filed under `trace.UNKNOWN_RUN` — the file that constant exists for, and a
    named absence is greppable where a dropped event is not. Pinned rather than left implicit
    because the fix (the runner reporting its id through the feed) belongs to the trace layer's
    own sub-project, and whoever lands it should see this expectation move.
    """
    import json

    from manyruns import trace

    where = _sink(monkeypatch, tmp_path)
    _model_says(monkeypatch, CANNED)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert screen.results is None, "the fixture settled the run and proves nothing"
        await _ask(app, pilot, "? is this working")

    landed = where / f"{trace.UNKNOWN_RUN}.jsonl"
    assert landed.exists(), f"the event went somewhere else: {sorted(p.name for p in where.iterdir())}"
    assert json.loads(landed.read_text().splitlines()[0])["run_id"] is None


@drives
async def test_nothing_listening_says_so_in_the_pane_and_writes_no_event(monkeypatch, tmp_path):
    """§4: `ask_available()` False shows the sentence and emits NOTHING. The record is of
    answers; an outage is not one, and a corpus of them would carry a question with a null
    answer and no way to say why.

    The directory not existing at all is the assertion, which is `JsonlTracer`'s own design —
    it creates the directory on the first `emit`, so "a session that traces nothing leaves
    nothing behind" is checkable rather than a claim.
    """
    where = _sink(monkeypatch, tmp_path)
    _model_says(monkeypatch, CANNED, listening=False)

    app = _Harness(RunScreen(EMBED))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _ask(app, pilot, "? should I trust leiden here")
        pane = _words(_pane(app, "#answer"))

    assert RunScreen.UNAVAILABLE in pane
    assert CANNED not in pane, "the pane answered with a model that was never called"
    assert not where.exists(), f"an unavailable model still wrote a record: {where}"


@drives
async def test_a_server_that_answers_nothing_is_not_recorded_as_an_answer(monkeypatch, tmp_path):
    """THE CASE THE PROBE CANNOT SEE: something is listening and `agents.answer` still returns
    `None` — an unpulled model, a timeout, a thinking budget spent with `content` empty
    (measured in `agents`: 9.1 s and an empty string). The pane says the outcome and the sink
    stays empty for the reason above: `answer` collapses every one of those to one value, so an
    event could record that something failed but never what."""
    where = _sink(monkeypatch, tmp_path)
    _model_says(monkeypatch, None)

    app = _Harness(RunScreen(EMBED))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _ask(app, pilot, "? should I trust leiden here")
        pane = _words(_pane(app, "#answer"))

    assert RunScreen.UNANSWERED in pane
    assert RunScreen.UNAVAILABLE not in pane, "a server that answered nothing was reported as down"
    assert not where.exists()


@drives
async def test_the_answer_is_state_so_a_repaint_keeps_it_under_the_tuned_pair(monkeypatch):
    """COMMIT 4'S CORRECTION, held. `sync()` rebuilds this screen from state on every step
    report, and a run keeps reporting while a model thinks — so an answer painted straight into
    the widget would be wiped by the next `on_step`, which is the failure that made the tuned
    pair state in the first place.

    AND THE ORDER, which is the pane's whole layout decision: the tuned pair above, the answer
    below. The last thing written is the thing that gets read, and prose the person asked for
    two seconds ago is what they are looking for — not a readout that was already on screen.
    """
    _no_sink(monkeypatch)
    _model_says(monkeypatch, CANNED)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.tune_step = "phate"
        screen.moved = ({"trustworthiness": 0.50}, {"trustworthiness": 0.80})
        await _ask(app, pilot, "? should I trust leiden here")

        # a step lands while the answer is on screen — the repaint that used to wipe it
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0}])
        await pilot.pause()
        pane = _words(_pane(app, "#answer"))

    assert CANNED in pane, "the next step report wiped the answer"
    assert pane.index("what changed in phate") < pane.index("? should I trust leiden here")
    assert pane.index("? should I trust leiden here") < pane.index(CANNED)


@drives
async def test_a_bare_question_mark_asks_nothing_at_all(monkeypatch):
    """A `?` and nothing else is a keystroke somebody thought better of. The field is cleared
    and the model is not called — a record with no question is still a prompt a model would
    answer, and the event would carry `"question": ""`."""
    _no_sink(monkeypatch)
    seen = _model_says(monkeypatch, CANNED)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _ask(app, pilot, "?   ")
        pane = _words(_pane(app, "#answer"))
        field = app.screen.query_one("#ask-input", Input)

    assert seen == {}, "a bare `?` reached the model"
    assert screen.asked == ""
    assert RunScreen.AT_REST in pane
    assert field.value == ""


@drives
async def test_the_pane_ends_on_the_question_that_was_asked_last(monkeypatch, tmp_path):
    """ASK, WAIT, ASK AGAIN — and the second answer is the one on screen, whichever lands first.

    `exclusive=True` is not enough and that is the whole test. Textual cancels a superseded
    worker's RECORD; it cannot interrupt the thread, which runs to completion and posts. So the
    handler that took the last message to ARRIVE showed the older answer whenever the older call
    was the slower one — and at `agents`' measured latencies (2.4 s cold, 0.7 s warm) slow-then-
    fast is the ordinary case, not a corner: the first question pays for the model to load.

    Driven with a stub that BLOCKS rather than sleeps, so the ordering is forced instead of
    raced: the first call waits on an event this test holds, the second answers immediately, and
    only then is the first released.

    THE SINK KEEPS BOTH, which is the asymmetry the handler is built on. A superseded answer is a
    real answer to a question a person really asked and the corpus is the place for it; the pane
    is where somebody is looking, and there they are looking for the one they are waiting on.
    """
    import json

    where = _sink(monkeypatch, tmp_path)
    release = threading.Event()

    def _answer(*, system, prompt):
        question = prompt.rsplit("question: ", 1)[-1].strip()
        if question.startswith("slow"):
            assert release.wait(10), "the slow call was never released"
        return f"answered {question}"

    monkeypatch.setattr(agents, "ask_available", lambda: True)
    monkeypatch.setattr(agents, "answer", _answer)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = {"run_id": "2d730cc5c591"}
        field = app.screen.query_one("#ask-input", Input)
        field.disabled = False
        field.focus()
        field.value = "? slow one"
        await pilot.press("enter")
        await pilot.pause()
        field.value = "? fast two"
        await pilot.press("enter")

        landed = where / "2d730cc5c591.jsonl"
        for _ in range(200):                      # the fast answer, while the slow one blocks
            await pilot.pause(0.02)
            if screen.answer:
                break
        assert screen.answer == "answered fast two", (
            f"the pane is showing {screen.answer!r} while the slow call is still blocked")

        release.set()
        for _ in range(200):                      # …and now the superseded one comes back
            await pilot.pause(0.02)
            if landed.exists() and len(landed.read_text().splitlines()) == 2:
                break
        await pilot.pause()
        pane = _words(_pane(app, "#answer"))

    assert screen.asked == "fast two"
    assert "answered fast two" in pane
    assert "answered slow one" not in pane, "a superseded answer took the pane back"

    events = [json.loads(line) for line in landed.read_text().splitlines()]
    assert [e["question"] for e in events] == ["fast two", "slow one"], (
        "the record dropped an answer somebody was given")


@drives
async def test_a_probe_that_raises_leaves_the_app_up_rather_than_taking_it_down(
        monkeypatch, tmp_path):
    """A THREAD WORKER THAT RAISES IS FATAL, and one raise is a shell variable away.

    `agents.ask_available` computed `urlsplit(...).port` outside its `except OSError`, so
    `OLLAMA_BASE_URL=http://localhost:notaport/v1` in a profile raised `ValueError` from the
    probe. Measured before the catch: `WorkerFailed` out of the app, `app.is_running` False, and
    the pane frozen at `thinking…` — `agents.py`'s own contract ("the interactive shell must
    never fail because an optional model call did") broken through the one door that file did
    not guard. That source fix has since landed (`test_agents_ask.py` pins it), which is why the
    raise here is INJECTED: what is under test is the screen surviving a callee that breaks its
    promise, and the only honest way to test that is with one that does.

    The proof that the app is up is not `is_running` but a SECOND ASK that answers: a screen
    still taking input is the property, and it is the one a person would notice missing.
    """
    import json

    where = _sink(monkeypatch, tmp_path)

    def _boom():
        raise ValueError("Port could not be cast to integer value as 'notaport'")

    monkeypatch.setattr(agents, "ask_available", _boom)
    monkeypatch.setattr(agents, "answer", lambda *, system, prompt: CANNED)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _ask(app, pilot, "? should I trust leiden here")
        broke = _words(_pane(app, "#answer"))
        # READ BEFORE THE SECOND ASK, because the second one is meant to write: what is being
        # asserted is that the RAISE wrote nothing, not that the session did.
        wrote_on_the_raise = where.exists()

        # the endpoint gets fixed while the app is still up, which is the whole point of it
        # still being up
        monkeypatch.setattr(agents, "ask_available", lambda: True)
        await _ask(app, pilot, "? and now")
        after = _words(_pane(app, "#answer"))

    assert RunScreen.UNAVAILABLE in broke, "a raise left the pane saying nothing"
    assert RunScreen.THINKING not in broke, "the pane is stuck mid-thought"
    assert not wrote_on_the_raise, "a call that never happened was recorded as an answer"
    assert CANNED in after, "the app never took another question"
    events = [json.loads(line)
              for f in sorted(where.iterdir()) for line in f.read_text().splitlines()]
    assert [e["question"] for e in events] == ["and now"], events


@drives
async def test_the_channel_is_open_exactly_while_the_loop_is_waiting_for_an_answer(monkeypatch):
    """THE REACHABLE SURFACE, driven the way a person reaches it: nothing here enables the field.

    The `?` channel lives on the GATE's input, which `on_run_screen_ask` enables when the loop
    asks and `on_input_submitted` disables on the answer. So this test also pins the half that is
    NOT reachable BY TYPING — between questions there is no field to type into — because that is
    a cost of keeping the screen's own keys (an enabled `Input` sits in the focus chain and
    swallows `f`, `s`, `o` and the figure arrows; measured at `compose`), and a cost nobody wrote
    down is a cost somebody re-discovers. `action_ask` is the door through it, tested below: the
    key opens the same row and `escape` closes it again, so the disabled state this test asserts
    is where the screen SITS rather than where it is stuck.
    """
    from manyruns.tui.gate import Gate

    _no_sink(monkeypatch)
    _model_says(monkeypatch, CANNED)

    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        at_rest = app.screen.query_one("#ask-input", Input)
        assert app.screen.query_one("#ask").display is False
        assert at_rest.disabled is True, "the field is in the focus chain with nothing pending"

        await _asked(app, pilot, gate)            # the loop asks — exactly as `Gate.prompt` does
        field = app.screen.query_one("#ask-input", Input)
        assert field.disabled is False and field.has_focus

        field.value = f"{run_screen.ASK_PREFIX} should I trust leiden here"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()
        pane = _words(_pane(app, "#answer"))
        assert CANNED in pane, "the channel is unreachable from the surface a person has"

        field.value = "a"                          # …and answering the loop closes it again
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.query_one("#ask-input", Input).disabled is True
        assert app.screen.query_one("#ask").display is False


# ── the model line, and the key that gives it somewhere to render (spec §4) ──
def _row(app: App) -> str:
    """The controller row's model line. Read from the widget at the width the layout gave it,
    like every other pane assertion here."""
    return _words(_pane(app, "#ask-model"))


@drives
async def test_the_controller_row_says_which_model_and_then_how_long_it_took(monkeypatch):
    """§4's three states, in the order a person meets them: which model would answer, that it is
    answering, and how long it took.

    THE MIDDLE ONE IS DRIVEN WITH A BLOCKING STUB rather than raced. `thinking…` lives between
    two events — the worker starting and its message landing — so a test that let a fast stub run
    to completion would catch it or not depending on the machine, which is the shape of a flake
    that passes for a year and fails in CI. Here the stub waits on an event this test holds.

    The last assertion is on the COMPOSITOR, not on the widget: a row that was updated and never
    reached the screen is exactly what a `width: auto` widget in a hidden row looks like, and the
    whole point of this commit is that the model can be SEEN.
    """
    _no_sink(monkeypatch)
    release = threading.Event()

    def _answer(*, system, prompt):
        assert release.wait(10), "the call was never released"
        return CANNED

    monkeypatch.setattr(agents, "ask_available", lambda: True)
    monkeypatch.setattr(agents, "answer", _answer)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        at_rest = _row(app)

        field = app.screen.query_one("#ask-input", Input)
        field.value = f"{run_screen.ASK_PREFIX} should I trust leiden here"
        await pilot.press("enter")
        await pilot.pause()
        thinking = _row(app)

        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        answered = _row(app)
        on_screen = _words(_screen_text(app))

    model = agents.ask_model()
    assert at_rest == f"ask · {model} · {run_screen.ASK_BACKEND}"
    assert thinking == f"ask · {model} · {RunScreen.THINKING}"
    assert re.fullmatch(rf"ask · {re.escape(model)} · \d+\.\d s", answered), answered
    assert f"ask · {model}" in on_screen, "the row was written and never painted"


@drives
async def test_the_row_says_unavailable_and_names_no_model_at_all(monkeypatch):
    """§4's first state. The name is ABSENT rather than dimmed beside the word: nothing is
    listening on the port, which is not a fact about `qwen3:8b` — every model is equally out of
    reach — and `ask · qwen3:8b · unavailable` would read as this one being the trouble.

    The border still carries the name, and that is the division §5 draws: the border says WHO
    would answer, the row says WHETHER anything can.
    """
    _no_sink(monkeypatch)
    _model_says(monkeypatch, CANNED, listening=False)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        row = _row(app)
        on_screen = _words(_screen_text(app))
        title = app.screen.query_one("#answer-pane").border_title

    assert row == f"ask · {RunScreen.UNAVAILABLE}"
    assert agents.ask_model() not in row
    assert RunScreen.UNAVAILABLE in on_screen, "the row never reached the screen"
    assert title == agents.ask_model(), "the border stopped naming who would answer"


@drives
async def test_the_ask_key_opens_a_settled_run_and_escape_hands_the_screen_back(monkeypatch):
    """THE AFFORDANCE, and nothing here touches `disabled` — that is the assertion.

    Every other test in this file reaches the channel through a field the gate enabled or one it
    enabled by hand. A settled run is where spec §0's own demo question is most naturally typed,
    and until this key existed there was no field on screen to type it into. So this drives the
    whole gesture: press `?`, ask, read, and press `escape` — which must close the QUESTION and
    not the run. Leaving the run from here would throw away a screenful of analysis because
    somebody thought better of typing.
    """
    _no_sink(monkeypatch)
    _model_says(monkeypatch, CANNED)

    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.screen.query_one("#ask").display is False, "the row was up before anyone asked"

        await pilot.press("question_mark")
        await pilot.pause()
        field = app.screen.query_one("#ask-input", Input)
        assert app.screen.query_one("#ask").display is True
        assert field.disabled is False and field.has_focus
        # The key that opened the channel is IN the channel: the line a person completes here is
        # the line they would have typed by hand while the gate was waiting.
        assert field.value == f"{run_screen.ASK_PREFIX} ", f"{field.value!r} is not the prefix"

        field.value = f"{run_screen.ASK_PREFIX} should I trust leiden here"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()
        assert CANNED in _words(_pane(app, "#answer")), "the settled run could not be asked"

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen, "escape left the run instead of closing the question"
        assert app.screen.query_one("#ask-input", Input).disabled is True, \
            "the field kept the screen's own keys"
        assert app.screen.query_one("#ask").display is False
        assert CANNED in _words(_pane(app, "#answer")), "closing the row took the answer with it"


@drives
async def test_escape_from_text_then_navigation_confirms_leaving(monkeypatch):
    from manyruns.tui.gate import Gate

    _no_sink(monkeypatch)
    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await _asked(app, pilot, gate)
        field = screen.query_one(Input)
        field.value = "t=20"
        question = gate.question_id
        await pilot.press("escape")
        assert app.screen is screen and screen.focused is field
        assert field.value == "" and gate.is_pending(question)
        await pilot.press("escape")
        assert isinstance(app.screen, run_screen.LeaveRunScreen)
        wording = app.screen.query_one("#leave-explanation", Static).visual.plain
        for fact in ("in-flight step or measurement may finish", "later steps",
                     "final metrics that have not started are skipped", "incomplete", "fresh run"):
            assert fact in wording
        await pilot.press("escape")
        assert app.screen is screen and gate.is_pending(question)
        await pilot.press("escape", "tab", "enter")
        await pilot.pause()
        assert app.screen is not screen and gate.closed


@drives
async def test_the_loops_question_keeps_its_slot_and_the_key_types_itself(monkeypatch):
    """THE ROW CARRIES TWO FACTS AND NEITHER EVICTS THE OTHER.

    §4 named `#ask-prompt` for the model line, back when that row held one thing. It holds the
    loop's own question verbatim (`on_run_screen_ask`), so a model line written into it would
    delete the question a person is being asked in order to say which model is idle. The line got
    a second slot instead; this pins that the gate's wording is still there with it.

    And `ASK_PREFIX`'s second half: while the gate waits, the field has focus and swallows screen
    bindings, so `?` types a `?` rather than firing `action_ask`. One gesture, both places, no
    mode to be in.
    """
    from manyruns.tui.gate import Gate

    _no_sink(monkeypatch)
    _model_says(monkeypatch, CANNED)

    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        await _asked(app, pilot, gate)
        prompt = _words(_pane(app, "#ask-prompt"))
        row = _row(app)

        await pilot.press("question_mark")
        await pilot.pause()
        typed = app.screen.query_one("#ask-input", Input).value

    assert "accept" in prompt, f"the model line took the loop's question: {prompt!r}"
    assert row == f"ask · {agents.ask_model()} · {run_screen.ASK_BACKEND}"
    assert typed == run_screen.ASK_PREFIX, "the key did not reach the field the gate owns"


# ── the two facts the caller holds (spec rider 5b) ───────────────────────────
def test_the_app_hands_the_screen_the_dataset_and_the_ledgers_caution():
    """A `?` FROM THE SHIPPED PATH MUST CARRY THE QC NUMBERS, and until the app passed them it
    did not. `RunScreen` takes `entry` and `concerns` because neither can be re-derived here —
    re-measuring the file would be a second account of a dataset that may have changed since the
    roster read it, and re-deriving the caution would be a second legality judgement — so the
    one caller holding both passes them.

    Without them `record_for_prompt` renders the dataset block as six dashed values and no
    caution at all, which makes the spec's own demo question ("should I trust leiden on this
    before filtering?") unanswerable from the app: its whole answer IS the QC numbers.

    Asserted against `tui/app.py`'s REAL push arguments rather than a hand-built screen, because
    the defect was never in the screen — it was that nothing passed anything to it.
    """
    import ast
    import inspect

    from manyruns.tui import app as app_module

    source = inspect.getsource(app_module.ManyrunsApp._open_prepared_run)
    call = next(n for n in ast.walk(ast.parse(source.strip()))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "RunScreen")
    passed = {kw.arg for kw in call.keywords}
    assert {"entry", "concerns"} <= passed, (
        f"the app pushes RunScreen without the record's dataset block: {sorted(passed)}")


def test_a_screen_given_the_entry_names_the_dataset_instead_of_dashes():
    """The other half, measured on the record itself: the same screen with and without an
    `entry` renders a prompt that names the file or one that dashes six values."""
    from manyruns.tui.state import roster

    # A REAL ROSTER ROW, because the fact under test is that `entry.to_dict()` reaches the prompt
    # — a hand-built stand-in would pass while the real serialisation was broken. It is the same
    # call `app.py` makes to fill the screen, so a checkout with nothing in its drop folder has
    # nothing to assert on and says so.
    rows = [e for e in roster() if e.kind in {"file", "folder"}]
    if not rows:
        pytest.skip("no dropped file in this checkout to name")
    entry = rows[0]

    blind = RunScreen(EMBED)
    seeing = RunScreen(EMBED, entry=entry, concerns=("58% of genes sit in <3 cells",))

    assert entry.name not in blind.ask_context()
    assert entry.name in seeing.ask_context()
    assert "58% of genes sit in <3 cells" in seeing.ask_context()


@drives
async def test_unavailable_status_leaves_a_usable_input_at_80_columns(monkeypatch):
    """At 80×30, a gate prompt and unavailable label used up the horizontal input's cells."""
    _no_sink(monkeypatch)
    _model_says(monkeypatch, None, listening=False)
    from manyruns.tui.gate import Gate
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        await _live_question(pilot, gate, "accept / tune / cancel")
        await pilot.pause()
        field = screen.query_one("#ask-input", Input)
        status = screen.query_one("#ask-model", Static)
        assert RunScreen.UNAVAILABLE in status.visual.plain
        assert field.content_size.width >= 40
        assert status.region.y >= field.region.bottom
        assert field.region.right <= 80
        assert field.region.bottom <= 30
        await pilot.press("h", "e", "l", "l", "o")
        assert field.value == "hello" and field.has_focus


@drives
async def test_repeated_question_keeps_the_answer_from_the_newer_step(monkeypatch, tmp_path):
    """Hold the pre-step answer until the post-step answer lands: identical text is two asks."""
    import json

    where = _sink(monkeypatch, tmp_path)
    started, release = threading.Event(), threading.Event()
    prompts = []

    def answer(*, system, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            started.set()
            assert release.wait(10), "the first call was never released"
            return "before the step"
        return "after the step"

    monkeypatch.setattr(agents, "ask_available", lambda: True)
    monkeypatch.setattr(agents, "answer", answer)
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    async with app.run_test(size=(120, 30)) as pilot:
        try:
            await pilot.pause()
            screen.results = {"run_id": "repeat"}
            await pilot.press("question_mark")
            field = screen.query_one("#ask-input", Input)
            field.value = "? can I trust this"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause(0.02)
                if started.is_set():
                    break
            assert started.is_set()
            screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok",
                                      "seconds": 3.25}])
            await pilot.pause()
            field.value = "? can I trust this"
            await pilot.press("enter")
            for _ in range(200):
                await pilot.pause(0.02)
                if screen.answer:
                    break
            assert screen.answer == "after the step"
        finally:
            release.set()
        landed = where / "repeat.jsonl"
        for _ in range(200):
            await pilot.pause(0.02)
            if landed.exists() and len(landed.read_text().splitlines()) == 2:
                break
        assert screen.answer == "after the step"
        assert "after the step" in _words(_pane(app, "#answer"))
    events = [json.loads(line) for line in landed.read_text().splitlines()]
    assert [e["answer"] for e in events] == ["after the step", "before the step"]
    assert len({e["question"] for e in events}) == 1
    assert len({e["context_sha256"] for e in events}) == 2
    assert prompts[0] != prompts[1]


@drives
async def test_arrows_belong_to_each_pane_and_function_keys_work_from_text(tmp_path, monkeypatch):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(120, 40)) as pilot:
        a, b = _png(tmp_path / "a.png"), _png(tmp_path / "b.png")
        screen.feed.on_step({}, [
            {"index": 0, "name": "phate", "outcome": "ok", "plots": [str(a), str(b)]}])
        await _asked(app, pilot, gate)
        field = screen.query_one(Input)
        figures = screen.query_one(run_screen.FiguresPane)
        question = gate.question_id
        for text in ("? should I change t=20 here", "t=20"):
            field.value = text
            field.cursor_position = len(text)
            await pilot.press("left")
            assert field.cursor_position == len(text) - 1
            await pilot.press("up", "down", "right")
            assert field.value == text
            assert screen.tune_at == 0 and figures.at() == 1
            assert gate.is_pending(question)
        # A literal n/p remains text, never a figure shortcut.
        field.value = "? "
        await pilot.press("n", "p")
        assert field.value == "? np" and figures.at() == 1
        # Tab visits every available pane, with the active owner named on screen.
        for pane_id, name in (("steps-pane", "Trace"), ("answer-pane", "Answer"),
                              ("figures-pane", "Figures"), ("tune-strip", "Parameters"),
                              ("ask-input", "Text")):
            await pilot.press("tab")
            assert screen.focused.id == pane_id
            assert f"Active pane: {name}" in _screen_text(app)
            original = field.value
            await pilot.press("f7")
            assert figures.at() == 0
            await pilot.press("f8")
            assert figures.at() == 1
            assert screen.focused.id == pane_id and field.value == original
        assert "F7/F8 figures · Tab change pane" in _screen_text(app)
        await pilot.press("shift+tab")
        assert screen.focused.id == "tune-strip"
        await pilot.press("right")
        assert field.value == "t=20" and dict(screen.tunables)["t"] == "auto"
        assert "DRAFT, NOT SUBMITTED: 20" in _pane(app, "#tune-strip")
        await pilot.press("down", "right")
        assert screen.tune_at == 1 and field.value == "decay=41"
        await pilot.press("up")
        assert screen.tune_at == 0
        assert gate.is_pending(question)
        screen.query_one("#figures-pane").focus()
        await pilot.press("left")
        assert figures.at() == 0 and field.value == "decay=41"
        # Scroll ownership on both long panes, without changing parameters or figures.
        for pane_id, content_id in (("steps-pane", "steps"), ("answer-pane", "answer")):
            pane = screen.query_one(f"#{pane_id}")
            screen.query_one(f"#{content_id}", Static).update("\n".join(map(str, range(100))))
            pane.focus()
            await pilot.pause()
            await pilot.press("down", "down")
            await pilot.pause()
            assert pane.scroll_y > 0
            await pilot.press("up")
            assert field.value == "decay=41" and figures.at() == 0


@drives
async def test_draft_tracks_manual_text_and_only_publication_changes_current(monkeypatch):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        worker = await _asked_worker(app, pilot, gate)
        field = screen.query_one(Input)
        for text, expected in (("t=20", "20"), ("t=23", "23"), ("t=oops", "oops")):
            field.value = text
            await pilot.pause()
            strip = _words(_pane(app, "#tune-strip"))
            assert f"t current attempt: auto [DRAFT, NOT SUBMITTED: {expected}]" in strip
            assert dict(screen.tunables)["t"] == "auto"
        for text in ("knn=9", "? is t=20 useful", "a", "", "t="):
            field.value = text
            await pilot.pause()
            strip = _words(_pane(app, "#tune-strip"))
            assert "t current attempt: auto [DRAFT" not in strip
            if text == "knn=9":
                assert "knn current attempt: 5 [DRAFT, NOT SUBMITTED: 9]" in strip
            else:
                assert "DRAFT" not in strip
        field.value = "t=20"
        await pilot.press("enter")
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert dict(screen.tunables)["t"] == "auto", "delivery promoted a draft"
        assert field.value == "" and not screen.query_one("#tune-strip").display
        # A refusal republishes the unchanged declaration.
        worker = await _asked_worker(app, pilot, gate)
        assert dict(screen.tunables)["t"] == "auto"
        assert "DRAFT" not in _pane(app, "#tune-strip")
        field.value = "t=23"
        await pilot.press("escape")
        assert field.value == "" and "DRAFT" not in _pane(app, "#tune-strip")
        gate.arm({"name": "phate", "params": {"t": 20}})
        await pilot.pause()
        assert dict(screen.tunables)["t"] == 20
        assert "DRAFT" not in _pane(app, "#tune-strip")
        gate.close()
        worker.join(timeout=1)


async def _asked_worker(app, pilot, gate):
    gate.arm({"name": "phate", "params": {"knn": 5}})
    return await _live_question(pilot, gate)


@drives
@pytest.mark.parametrize("ending", ["close", "finish", "failure"])
@pytest.mark.parametrize("standalone", [False, True])
async def test_question_invalidation_preserves_only_explicit_standalone_composition(
        monkeypatch, ending, standalone):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(120, 40)) as pilot:
        if standalone:
            await pilot.press("question_mark")
            field = screen.query_one(Input)
            field.value = "? my unfinished question"
        worker = await _asked_worker(app, pilot, gate)
        field = screen.query_one(Input)
        old_question = RunScreen.Ask(gate.prompt, gate.question_kind, gate.question_id)
        if standalone:
            assert field.value == "? my unfinished question", "a gate question erased composition"
        else:
            field.value = "t=20"
        if ending == "close":
            gate.close()
        else:
            screen.post_message(RunScreen.Finished({} if ending == "finish" else None,
                                                   "boom" if ending == "failure" else None))
        await pilot.pause()
        worker.join(timeout=1)
        assert not worker.is_alive() and gate.closed
        screen.post_message(old_question)
        screen.post_message(RunScreen.Arm({"name": "phate", "params": {"t": 99}}))
        await pilot.pause()
        assert screen._question_kind is None and screen._question_id is None
        assert screen.tunables == [] and not screen.query_one("#tune-strip").display
        assert screen.query_one("#ask-prompt", Static).visual.plain == ""
        assert screen.query_one("#ask").display is standalone
        assert field.disabled is (not standalone)
        assert field.value == ("? my unfinished question" if standalone else "")
        if standalone:
            assert field.has_focus
            await pilot.press("left")
            assert field.value == "? my unfinished question"


@drives
async def test_late_identical_question_cannot_replace_a_new_question(monkeypatch):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test() as pilot:
        worker = await _asked_worker(app, pilot, gate)
        old = RunScreen.Ask(gate.prompt, gate.question_kind, gate.question_id)
        await pilot.press("a", "enter")
        worker.join(timeout=1)
        screen.post_message(old)
        await pilot.pause()
        assert screen.query_one(Input).disabled, "answered question reopened the editor"
        worker = await _asked_worker(app, pilot, gate)
        field = screen.query_one(Input)
        field.value = "knn=12"
        new_id = gate.question_id
        screen.post_message(old)
        await pilot.pause()
        assert field.value == "knn=12" and screen._question_id == new_id
        assert not gate.answer("accept", question_id=old.question_id)
        assert gate.is_pending(new_id)
        gate.close()
        worker.join(timeout=1)


@drives
async def test_escape_from_standalone_composition_keeps_a_new_gate_question_answerable(monkeypatch):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        field = screen.query_one(Input)
        field.value = "? unfinished"
        worker = await _asked_worker(app, pilot, gate)
        question = gate.question_id
        await pilot.press("escape")
        assert screen.focused is field
        assert not field.disabled and field.value == "" and gate.is_pending(question)
        # Clearing the draft keeps the original pending question directly answerable.
        assert field.has_focus
        await pilot.press("a", "enter")
        worker.join(timeout=1)
        assert not worker.is_alive() and not gate.is_pending(question)


@drives
async def test_closure_does_not_steal_focus_from_a_reader(monkeypatch):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test() as pilot:
        worker = await _asked_worker(app, pilot, gate)
        await pilot.press("tab", "tab")
        assert screen.focused.id == "answer-pane"
        gate.close()
        await pilot.pause()
        worker.join(timeout=1)
        assert screen.focused.id == "answer-pane"
        assert "Active pane: Answer" in _screen_text(app)


@drives
async def test_parameter_draft_replaces_standalone_composition_ownership(monkeypatch):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        worker = await _asked_worker(app, pilot, gate)
        await pilot.press("shift+tab", "right")
        assert screen.query_one(Input).value == "t=20"
        assert not screen._ask_open
        gate.close()
        await pilot.pause()
        worker.join(timeout=1)
        assert screen.query_one(Input).value == "" and screen.query_one(Input).disabled
        assert screen.focused.id == "steps-pane"


@drives
@pytest.mark.parametrize('draft', ['', 'knn=20'])
@pytest.mark.parametrize('navigation', [False, True])
async def test_pending_question_escape_opens_leave_or_says_exactly_why_not(monkeypatch, draft, navigation):
    from manyruns.tui.gate import Gate

    _model_says(monkeypatch, CANNED)
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate)
    app = _Harness(screen)
    async with app.run_test(size=(150, 35)) as pilot:
        worker = await _asked_worker(app, pilot, gate)
        screen.query_one(Input).value = draft
        if navigation:
            screen.query_one('#steps-pane').focus()
        question = gate.question_id
        try:
            await pilot.press('escape')
            if draft:
                hint = 'esc again to leave this run · the question is still waiting'
                assert app.screen is screen
                assert screen.said[-1] == hint
                assert hint in _screen_text(app)
                assert screen.query_one(Input).value == ''
                assert gate.is_pending(question)
                await pilot.press('escape')
            assert isinstance(app.screen, run_screen.LeaveRunScreen)
            assert gate.is_pending(question)
            await pilot.press('tab', 'enter')
            await pilot.pause()
            assert gate.closed
        finally:
            gate.close()
            worker.join(timeout=1)
        assert not worker.is_alive()


@drives
async def test_blocked_finalizer_shows_elapsed_status_until_finished(tmp_path, monkeypatch):
    import argparse
    import time
    from manyruns import app as cli
    from manyruns.session import Session
    from manyruns.tui.app import ManyrunsApp
    from manyruns.tui.gate import Gate
    from manyruns.tui import state

    monkeypatch.chdir(tmp_path)
    entered, release = threading.Event(), threading.Event()
    session = Session('finalize', engine='mock', recipe={'name': 'embed', 'steps': []})
    real_close = session.close

    def close(**kw):
        entered.set()
        assert release.wait(5), 'test did not release finalization'
        return real_close(**kw)

    monkeypatch.setattr(session, 'close', close)
    monkeypatch.setattr(cli, '_build_session', lambda *a, **kw: session)
    app_driver = ManyrunsApp(argparse.Namespace(engine='mock'))
    gate = Gate()
    screen = RunScreen(EMBED, gate=gate, start=app_driver.start_for(
        state.catalog_entry('swissroll'), 'embed', gate=gate))
    app = _Harness(screen)
    async with app.run_test(size=(150, 35)) as pilot:
        try:
            deadline = time.monotonic() + 3
            while not entered.is_set() and time.monotonic() < deadline:
                await pilot.pause()
            assert entered.is_set()
            await pilot.pause()
            frame = _words(_screen_text(app))
            assert 'measuring geometry… final metrics are running; duration depends on dataset and machine' in frame
            assert 'elapsed' in frame and not screen._finished
            before = screen.query_one('#said', Static).content
            await asyncio.sleep(0.25)
            await pilot.pause()
            assert screen.query_one('#said', Static).content != before
        finally:
            release.set()
        await _settled(pilot, app)
        assert screen._finished
        assert screen._finalizing_timer is None
