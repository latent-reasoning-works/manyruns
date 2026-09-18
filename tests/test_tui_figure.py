"""The full-screen figure viewer and the quickdrop — `manyruns/tui/figure.py`.

The gesture this exists for: you are watching a run, an embedding comes up that you want, and
you keep it — without leaving the app, without hunting through `outputs/`, and at a resolution
you can put in a paper rather than the 120 dpi a terminal wants.

`o` is here for a reason a graphics tier cannot cover: no cell-based rendering preserves
pointwise resolution, and pointwise is the whole content of an embedding scatter. On a terminal
with no image protocol, handing the PNG to the desktop is the only exact route there is.

NOTHING HERE LAUNCHES A VIEWER. `open_externally` is monkeypatched in the one test that presses
`o`; a suite that opened Preview twelve times would be a suite nobody runs.
"""
from __future__ import annotations

import asyncio
import functools
from pathlib import Path

import pytest

pytest.importorskip("textual")
np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from textual.app import App  # noqa: E402

from manyruns.pipeline import io as _io  # noqa: E402
from manyruns.tui import figure as figure_mod  # noqa: E402
from manyruns.tui.figure import FigureScreen, figures_of, quickdrop  # noqa: E402
from manyruns.tui.run import RunScreen  # noqa: E402
from manyruns.tui.state import StepView  # noqa: E402

EMBED = {"name": "embed", "steps": [{"name": "phate", "group": "latent"}]}


def drives(body):
    """One `asyncio.run` per test — this repo has no `pytest-asyncio`, and adding one is a
    dependency decision rather than a test decision (`test_tui_run.drives` makes the same call)."""
    @functools.wraps(body)
    def wrapper(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))

    return wrapper


class _Harness(App):
    def __init__(self, screen) -> None:
        super().__init__()
        self._screen = screen

    def on_mount(self) -> None:
        self.push_screen(self._screen)


def _figure(tmp_path, name="phate", n=200):
    """A real run's figure: the PNG a terminal shows, and the spec beside it."""
    plots: list = []
    emb = np.asarray(np.random.default_rng(0).normal(size=(n, 3)))
    _io._save_scatter(emb, None, tmp_path, f"{name}.png", plots, title=name)
    return plots[0]


def _step(i, name, plots):
    return StepView(index=i, name=name, state="ok", glyph="✓", word="ok", plots=plots)


def _text(app) -> str:
    return "\n".join(s.text for s in app.screen._compositor.render_strips())


# ── the quickdrop, which is what the keys are for ────────────────────────────
def test_the_quickdrop_names_the_step_and_the_run_it_came_from(tmp_path):
    """A folder of `phate.pdf` from five runs is a folder you cannot use."""
    png = _figure(tmp_path)

    written = quickdrop(png, "phate", "abc12345deadbeef", tmp_path / "figures")

    assert {Path(p).stem for p in written} == {"phate-abc12345"}
    assert [Path(p).suffix for p in written] == [".svg", ".pdf", ".png"]


def test_a_run_with_no_id_yet_still_saves(tmp_path):
    """`results` is None until a run ends, so a save mid-run has no id. It names the step alone
    rather than inventing one for a lineage still being written."""
    png = _figure(tmp_path)

    written = quickdrop(png, "phate", None, tmp_path / "figures")

    assert {Path(p).stem for p in written} == {"phate"}


@drives
async def test_s_on_the_run_screen_saves_the_figure_in_the_pane(tmp_path, monkeypatch):
    """The gesture, without leaving the run. The moment someone decides they want a figure is
    the moment it appears in the pane; making them open a screen first to keep it puts a step
    between the decision and the file."""
    monkeypatch.chdir(tmp_path)
    png = _figure(tmp_path)
    screen = RunScreen(EMBED)
    app = _Harness(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0,
                                  "plots": [str(png)]}])
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        told = [n.message for n in app._notifications]

    dropped = sorted(p.name for p in (tmp_path / "figures").iterdir())
    assert dropped == ["phate.pdf", "phate.png", "phate.svg"]
    # the toast names the FILES: "saved to figures/" is a sentence you have to act on to verify
    assert told and "phate.svg" in told[0] and "KB" in told[0]


@drives
async def test_saving_a_figure_with_no_spec_says_so_rather_than_lying(tmp_path, monkeypatch):
    """A figure drawn before specs existed has none. Reporting "saved" when nothing was written
    is the one outcome worse than not offering the key."""
    monkeypatch.chdir(tmp_path)
    old = tmp_path / "plots" / "old.png"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"\x89PNG\r\n\x1a\n")
    screen = RunScreen(EMBED)
    app = _Harness(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0,
                                  "plots": [str(old)]}])
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        told = [(n.severity, n.message) for n in app._notifications]

    assert not (tmp_path / "figures").exists()
    assert told and told[0][0] == "warning" and "nothing to re-render" in told[0][1]


@drives
async def test_o_hands_the_file_to_the_desktop_and_does_not_block(tmp_path, monkeypatch):
    """The only pointwise-exact route on a terminal with no graphics protocol. `Popen`, not
    `run`: a viewer that blocked until Preview was closed would look like a hang."""
    opened: list = []
    monkeypatch.setattr(figure_mod, "open_externally", lambda p: opened.append(str(p)))
    png = _figure(tmp_path)
    screen = RunScreen(EMBED)
    app = _Harness(screen)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0,
                                  "plots": [str(png)]}])
        await pilot.pause()
        await pilot.press("o")
        await pilot.pause()

    assert opened == [str(png)]


def test_the_opener_reports_a_failure_instead_of_raising(tmp_path, monkeypatch):
    """A missing `xdg-open` must cost the gesture, not the app."""
    import subprocess

    def boom(*a, **k):
        raise OSError("no such tool")

    monkeypatch.setattr(subprocess, "Popen", boom)

    assert "OSError" in (figure_mod.open_externally(tmp_path / "x.png") or "")


# ── the viewer ───────────────────────────────────────────────────────────────
def test_the_figures_come_from_the_step_that_drew_them():
    """`figures_of` is taken by both the pane and the viewer, so they cannot disagree about how
    many there are or which is newest. A viewer that globbed an output folder would be a second
    account of which step drew what."""
    rows = [_step(0, "phate", ["/a/phate.png"]),
            _step(1, "mioflow", ["/a/flow.png", "/a/traj.png"])]

    assert figures_of(rows) == [("phate", "/a/phate.png"), ("mioflow", "/a/flow.png"),
                                ("mioflow", "/a/traj.png")]


@drives
async def test_the_viewer_keeps_the_path_on_screen_beside_the_picture(tmp_path):
    """The path does not stop being useful because the picture arrived — it is still what you
    hand to another program. It has its own auto-height row because sharing a fixed row with the
    keys can clip a long output path and hide the navigation hints."""
    png = _figure(tmp_path)
    app = _Harness(FigureScreen([("phate", str(png))]))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        text = _text(app)

    assert "phate.png" in text and "· phate" in text
    assert str(png).replace(" ", "") in text.replace(" ", "").replace("\n", "")
    for key in ("esc back", "s save for a paper", "o open"):
        assert key in text


@drives
async def test_the_arrows_walk_the_runs_figures_and_wrap(tmp_path):
    """Two figures where `→` stops working on the second reads as broken, not as bounded."""
    figs = [("phate", str(_figure(tmp_path, "a"))), ("mioflow", str(_figure(tmp_path, "b")))]
    screen = FigureScreen(figs, index=1)
    app = _Harness(screen)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        await pilot.press("right")
        await pilot.pause()
        assert screen.index == 0, "it wraps"
        await pilot.press("left")
        await pilot.pause()
        assert screen.index == 1


@drives
async def test_one_figure_is_not_offered_a_key_that_would_do_nothing(tmp_path):
    """A key that does nothing is worse than an absent one: it reads as the screen being stuck."""
    screen = FigureScreen([("phate", str(_figure(tmp_path)))])
    app = _Harness(screen)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        text = _text(app)
        await pilot.press("right")
        await pilot.pause()
        assert screen.index == 0

    assert "between figures" not in text


@drives
async def test_a_cell_tier_draws_nothing_and_says_where_the_figure_is(tmp_path):
    """Reported from a real session on the unicode tier: "not blank but super choppy, literally
    nothing to do with the original one."

    That is the correct reaction. A graphics protocol draws at the cell area's real pixel size;
    a cell tier gets two pixels per cell — 864 px against 58,752 for the same pane, 68x apart.
    An embedding scatter is read pointwise, so 864 px is a different picture, not a smaller one,
    and showing it next to an exact path asks a reader to distrust the picture.

    `run_test` has no tty, so detection lands on the fallback — which is the case under test."""
    from manyruns.tui import images
    from manyruns.tui.figure import drawable

    assert not images.is_pixel_perfect(), "headless has no graphics tier; that is the point"
    assert drawable("/a/phate.png") is None
    app = _Harness(FigureScreen([("phate", str(_figure(tmp_path)))]))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        text = _text(app)
        assert app.screen.query_one("#figure").image is None, "nothing drawn"

    assert "no inline images" in text and "o" in text


@drives
async def test_f_opens_the_newest_figure_and_escape_comes_back(tmp_path):
    """`f` opens the figure the PANE was showing, not the first one of the run."""
    a, b = _figure(tmp_path, "a"), _figure(tmp_path, "b")
    screen = RunScreen({"name": "r", "steps": [{"name": "phate", "group": "latent"},
                                               {"name": "mioflow", "group": "lightning"}]})
    app = _Harness(screen)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen.feed.on_step({}, [
            {"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.0, "plots": [str(a)]},
            {"index": 1, "name": "mioflow", "outcome": "ok", "seconds": 1.0, "plots": [str(b)]}])
        await pilot.pause()
        await pilot.press("f")
        await pilot.pause()

        top = app.screen
        assert isinstance(top, FigureScreen)
        assert top.figures[top.index] == ("mioflow", str(b))

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)


# ── a tier that draws nothing must say so ────────────────────────────────────
def test_cell_tier_guidance_is_specific_to_the_tui(monkeypatch):
    from manyruns import figures
    from manyruns.tui import images

    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("STY", raising=False)
    monkeypatch.setattr(images, "is_pixel_perfect", lambda: False)
    note = figure_mod._tier_note()
    assert "kitty (TGP)" in note and "sixel-capable terminal" in note
    assert "restart manyruns" in note
    assert "MANYRUNS_IMAGE=tgp" in note and "MANYRUNS_IMAGE=sixel" in note
    assert "enableImages" not in note and "MANYRUNS_INLINE_IMAGES" not in note
    assert "terminal.integrated.enableImages" in figures.hint()


@drives
async def test_a_forced_tier_names_itself_because_it_can_draw_nothing(tmp_path, monkeypatch):
    """Reported from a real session: forced to `sixel` in a VS Code terminal,
    the escapes went out, the terminal swallowed them, and
    the pane was an empty rectangle — a black hole on a dark theme, with no explanation and no
    way to tell it from a bug.

    Detection cannot cause that: it only picks a graphics tier the terminal claimed. A FORCED
    one can, so a forced one says its name and what to do about it."""
    from manyruns.tui import images

    monkeypatch.setattr(images, "forced", lambda: True)
    monkeypatch.setattr(images, "is_pixel_perfect", lambda: True)
    monkeypatch.setattr(images, "tier", lambda: "sixel")
    app = _Harness(FigureScreen([("phate", str(_figure(tmp_path)))]))

    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        text = _text(app)

    assert "forced" in text and "sixel" in text
    assert "blank" in text                       # names the symptom the user actually sees
    assert "MANYRUNS_IMAGE" in text and "o" in text


@drives
async def test_a_detected_graphics_tier_says_nothing(tmp_path, monkeypatch):
    """Quiet when there is nothing to warn about. A note on every run is a note nobody reads,
    and "you are seeing real pixels" is not information to someone seeing real pixels."""
    from manyruns.tui import images

    monkeypatch.setattr(images, "forced", lambda: False)
    monkeypatch.setattr(images, "is_pixel_perfect", lambda: True)
    app = _Harness(FigureScreen([("phate", str(_figure(tmp_path)))]))

    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        note = app.screen.query_one("#figure-note").visual.plain

    assert note.strip() == ""


def test_forced_is_only_true_for_a_tier_the_env_var_actually_names(monkeypatch):
    """A junk value falls through to detection, so it must not be reported as forced —
    `MANYRUNS_IMAGE=yes` picking the auto tier and then claiming to be forced would send
    someone chasing an override that did nothing."""
    import importlib

    from manyruns.tui import images as mod

    for value, expected in (("sixel", True), ("", False), ("yes", False), ("TGP", True)):
        monkeypatch.setenv("MANYRUNS_IMAGE", value)
        reloaded = importlib.reload(mod)
        assert reloaded.forced() is expected, value
    monkeypatch.delenv("MANYRUNS_IMAGE", raising=False)
    importlib.reload(mod)


@drives
async def test_function_keys_also_navigate_the_full_figure_viewer(tmp_path):
    figs = [("phate", str(_figure(tmp_path, "a"))), ("umap", str(_figure(tmp_path, "b")))]
    screen = FigureScreen(figs, index=1)
    app = _Harness(screen)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("f7")
        assert screen.index == 0
        await pilot.press("f8")
        assert screen.index == 1
        assert "F7/F8" in _text(app)
    for key in ("f7", "f8"):
        assert any(binding.key == key and binding.priority for binding in screen.BINDINGS)

    assert not any(key in ("n", "p", "space")
                   for binding in screen.BINDINGS for key in binding.key.split(","))
