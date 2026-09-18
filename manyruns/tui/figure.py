"""One figure, at the size of the terminal — and the way out of the terminal.

`f` from the run screen puts one figure on the whole screen, `←/→` walk the run's figures,
`escape` goes back. Two keys take it further than the terminal can:

    s   save it for a paper — 300 dpi PDF + PNG, rebuilt from the figure's spec
    o   open the PNG in whatever this machine opens PNGs with

**REAL PIXELS, and the earlier conclusion here was wrong.** This module used to say the app
could not draw inside its own layout, on the strength of three measurements about an image
ESCAPE: rich counts a 51,681-character iTerm2 sequence as 51,679 printable cells, the compositor
clips it to the viewport and emits 2,103 of them torn across 663 strips, and `App.suspend()`
leaves the alternate screen. Every one of those still holds — and none of them is a property of
the app. They are properties of putting an escape through a compositor that measures by cell.
Measured in a pty at `textual 8.2.8`, inside a two-pane layout with scroll containers: 4 kitty
TGP sequences reach the terminal, 4 sixel on a sixel terminal. `textual-image` composites a
PLACEHOLDER and writes the graphics separately, so the escape never takes that path.

`AutoImage` picks the best tier the terminal admits — sixel → TGP → half-cell → unicode —
**at import time**, because querying the terminal is impossible once Textual has started (its
own note). That is why it is imported at module scope here and not inside a handler.

**On a cell tier NOTHING IS DRAWN**, and that is a decision rather than a gap. A graphics
protocol draws at the cell area's real pixel size; a cell tier gets two pixels per cell. Same
screen area, 68x apart — 864 px against 58,752 in the figures pane. An embedding scatter is read
pointwise, so 864 px is not a smaller version of it, and putting one on screen next to an exact
path asks a reader to distrust the picture. See `drawable`.

**It draws `plots` and never re-derives them.** The list arrives as `(step, path)` pairs from
`StepView.plots`, the per-step attribution the flat run-scoped list used to lose. A viewer that
globbed an output folder would be a second account of which step drew what.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Static

from manyruns.tui import colorby, images
from manyruns.tui.images import AutoImage


def _tier_note() -> str:
    """The line under the picture, and it exists for the case where there IS no picture.

    Three states, and only the middle one is quiet:

      * a FORCED graphics tier — `$MANYRUNS_IMAGE` was set, so the escapes go out whether or
        not this terminal can decode them. When it cannot, nothing is drawn and the pane is an
        empty rectangle: on a dark theme, a black hole with no explanation. Reported from a real
        session forced to `sixel` in a VS Code terminal. The CLI's iTerm image setting cannot
        change this renderer's tier. Detection cannot cause this, so only a forced tier
        says so.
      * a detected graphics tier — real pixels, nothing to say.
      * a cell tier — NOTHING IS DRAWN, and the note is the whole content. `images.hint()`
        names the terminals and overrides supported by this surface's renderer.
    """
    if images.forced() and images.is_pixel_perfect():
        return (f"[dim]forced [b]{images.tier()}[/b] — if this is blank your terminal cannot "
                f"decode it; unset MANYRUNS_IMAGE, or use [b]o[/b][/dim]")
    if images.is_pixel_perfect():
        return ""
    return f"[dim]no inline images here — press [b]o[/b] to open it. {images.hint()}[/dim]"


def drawable(path: "str | None") -> "str | None":
    """The image to hand a widget, or None when this terminal cannot show one honestly.

    **A cell tier is not a smaller version of the figure. It is a different picture.** Measured
    for a 600×480 matplotlib figure, same screen area either way:

        pane @100 cols   36×12 cells   halfcell      864 px  (0.3% of the figure)
                                       graphics   58,752 px
        full @100×30    100×28 cells   halfcell    5,600 px
                                       graphics  380,800 px  (essentially all of it)

    68× either way, because a graphics protocol draws at the cell area's real pixel size while a
    cell tier gets two pixels per cell whatever the font. Reported from a real session on the
    unicode tier: "not blank but super choppy, literally nothing to do with the original one".

    That is the correct reaction and the reason this returns None there. An embedding scatter is
    read POINTWISE — where one cell sits relative to its neighbours, the three points out on a
    branch tip — and 864 pixels cannot carry that for any number of points. Drawing it anyway
    puts a picture on screen that a reader has to distrust, next to a path that is exact.
    """
    return path if (path and images.is_pixel_perfect()) else None


#: Where a quickdrop lands. `figures/` under the working directory the app was launched from —
#: predictable, local, and greppable. NOT the run's own `outputs/<slug>/plots/` folder: the point
#: of the gesture is to get one chosen figure OUT of the pile a run leaves behind, and not the
#: home directory either, because a file that appears somewhere you did not choose is a file you
#: have to go looking for.
QUICKDROP_DIR = "figures"
_POSITIONAL_NOTE = "legacy figure: positional metadata alignment"


def figures_of(rows: list) -> list[tuple[str, str]]:
    """`(step, path)` for every figure in a run, in the order the steps drew them.

    Kept here and taken by both readers — this screen and `run.figures_view` — so the pane and
    the viewer cannot disagree about how many figures there are or which one is newest.
    """
    return [(row.name, str(plot)) for row in rows for plot in row.plots]


#: The mark for a figure on the index strip. `◉` is where the picture above is; `○` is one you
#: can walk to. Two glyphs and no colour, because this line has to survive the tier that cannot
#: draw the figure at all — on a cell terminal the strip IS the figure pane.
_HERE, _THERE = "\u25c9", "\u25cb"


def trajectory(found: list, selected: int, width: int = 0) -> str:
    """The run's figures as one line — an x axis of steps, with a cursor on the one being shown.

    THIS REPLACED A LIST THAT GREW, and the growth was the bug. The pane laid out an `AutoImage`
    at `height: 1fr` above a `Static` at `height: auto` that carried one line per figure, inside
    a `VerticalScroll`. `auto` wins that argument: measured on a 120x30 terminal, the pane is 7
    rows and the picture got **1 row** at one figure, 1 row at three, 1 row at six, while the
    listing ran to 27 rows and the pane scrolled. The figure was squeezed to a sliver by its own
    caption, and it got worse with every step that drew — which is why it looked like the pane
    was collapsing onto itself mid-run.

    So the index is ONE LINE whatever the run does, and the picture keeps the rest. Which is also
    the feature: the strip is an ordering, so walking it with the arrow keys is walking the
    trajectory the embedding took through the recipe.

    `width` clips the strip from the LEFT, keeping the cursor visible — a run with forty figures
    still shows where you are, and the count beside it says what is off-screen. Zero means no
    clipping, which is what the tests use and what a wide terminal gets.
    """
    if not found:
        return ""
    marks = [(_HERE if i == selected else _THERE) for i in range(len(found))]
    strip = "\u2500".join(marks)
    if width and len(strip) > width:
        # Keep the cursor in view. Each mark costs two columns (glyph + rule) except the last.
        at = selected * 2
        start = max(0, min(at - width // 2, len(strip) - width))
        strip = strip[start:start + width]
    return strip


def index_line(found: list, selected: int) -> str:
    """`mioflow · 3 of 5` — which step drew the figure on screen, and where it sits in the run.

    Separate from :func:`trajectory` because they answer different questions and one of them is
    the one a reader needs when the strip is clipped: the strip says WHERE, this says WHAT.
    """
    if not found:
        return ""
    step, _ = found[selected]
    return f"{step} \u00b7 {selected + 1} of {len(found)}"


def quickdrop(png: Path | str, step: str, run_id: Optional[str] = None,
              out_dir: Path | str = QUICKDROP_DIR, *, spec: Optional[dict] = None) -> list[str]:
    """Save this figure for a paper. Returns what was written; `[]` when there is no spec.

    SVG, PDF and PNG — an editable vector figure, the format a journal takes, and the one that
    goes in a slide deck — all from one artist tree, so they cannot disagree with each other or
    with the PNG on screen.

    Rebuilt from the spec beside the PNG rather than upscaled from it: the PNG is 120 dpi because
    that is what a terminal wants, and enlarging it would hand someone a blurry figure that looks
    like it was meant to be that size. `figspec.figure` is the same function the run drew with,
    so this is the figure they were looking at, re-emitted at 300 dpi.

    **Vector at any point count.** `figspec.export` never rasterises unless asked, so the SVG is
    a real scatter of paths however many cells are in it.

    Additional figures from the same step include their safe source-stem identity. A colour
    view's unique stem distinguishes even repeated selections of the same column, while the
    original step PNG keeps its established step/run export name.
    """
    from manyruns import figspec

    name = "-".join(p for p in (step, (run_id or "")[:8]) if p) or "figure"
    if Path(png).stem != step:
        name += "-" + figspec.safe_column(Path(png).stem)
    return figspec.export(png, out_dir, name, spec=spec)


class QuickdropWorker:
    """Screen-owned daemon export, shared with RunScreen.

    Construct on the screen, call ``save(png, step, run_id)`` from its save action, and
    ``cancel()`` when leaving/unmounting it. No spec reads, lock waits or file writes run
    on the UI thread. Cancellation stops a queued export; one already rendering may
    finish its requested files, but never sends a late notification to a closed screen.
    ``after_capture(callback)`` defers an action that could rename the source until its
    spec has been read, independently of the potentially long wait for the render lock.
    """

    def __init__(self, screen: Screen) -> None:
        self.screen = screen
        self._cancelled = threading.Event()
        self._busy = False
        self._capturing = False
        self._after_capture = []

    def cancel(self) -> None:
        self._cancelled.set()

    def after_capture(self, callback) -> None:
        """On the UI thread, continue once this export no longer reads the source."""
        if self._capturing:
            self._after_capture.append(callback)
        else:
            callback()

    def save(self, png: Path | str, step: str, run_id: Optional[str] = None) -> None:
        from manyruns import figspec
        from rich.markup import escape

        if self._cancelled.is_set():
            return
        if self._busy:
            self.screen.notify("figure export already in progress", severity="information")
            return
        self._busy = True
        self._capturing = True
        app = self.screen.app

        def captured():
            self._capturing = False
            callbacks, self._after_capture = self._after_capture, []
            for callback in callbacks:
                callback()

        def delivered(message, severity, timeout):
            self._busy = False
            if not self._cancelled.is_set() and self.screen.is_mounted:
                self.screen.notify(escape(message), severity=severity, timeout=timeout)

        def run():
            locked = False
            try:
                # Copy arrays and metadata before waiting for matplotlib. RunScreen holds
                # a tuning answer until captured() so _keep_attempt cannot rename either
                # sidecar halfway through this read. Rendering needs no source files.
                try:
                    spec = figspec.load(png)
                finally:
                    app.call_from_thread(captured)
                while not self._cancelled.is_set():
                    if figspec.RENDER_LOCK.acquire(timeout=0.05):
                        locked = True
                        break
                else:
                    return
                if self._cancelled.is_set():
                    return
                written = quickdrop(png, step, run_id, spec=spec) if spec is not None else []
                if not written:
                    message = ("no figure spec beside this PNG — nothing to re-render at print "
                               "quality. Re-run to get one.")
                    severity, timeout = "warning", 8
                else:
                    lines = [f"{Path(p).name}  {Path(p).stat().st_size / 1024:,.0f} KB"
                             for p in written]
                    message = "saved to " + str(Path(written[0]).parent) + "\n" + "\n".join(lines)
                    severity, timeout = "information", 10
            except Exception as exc:
                message, severity, timeout = f"could not export figure: {exc}", "error", 8
            finally:
                if locked:
                    figspec.RENDER_LOCK.release()
            if self._cancelled.is_set():
                return
            try:
                app.call_from_thread(delivered, message, severity, timeout)
            except RuntimeError:  # application exited while the daemon was completing
                pass

        threading.Thread(target=run, name="manyruns-export", daemon=True).start()


def open_externally(path: Path | str) -> Optional[str]:
    """Hand the file to the desktop. Returns an error string, or None on success.

    The only pointwise-exact route on a terminal with no graphics protocol, and the one that
    still works when someone wants to zoom. `Popen` and not `run`: a viewer that blocks the
    event loop until Preview is closed would look like the app had hung.
    """
    opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(
        sys.platform, ["xdg-open"])
    try:
        subprocess.Popen([*opener, str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, ValueError) as e:  # noqa: BLE001 - reported on screen, never fatal
        return f"{type(e).__name__}: {e}"
    return None


class FigureScreen(Screen):
    """A figure, full screen. `escape` back, `←/→` between them, `s` to keep one, `o` to open."""

    BINDINGS = [
        Binding("escape", "back", "back", show=True),
        Binding("f", "back", "back", show=False),
        Binding("right", "step(1)", "next", show=True),
        Binding("left", "step(-1)", "previous", show=True),
        Binding("f8", "step(1)", "next", show=False, priority=True),
        Binding("f7", "step(-1)", "previous", show=False, priority=True),
        Binding("s", "save", "save for a paper", show=True),
        Binding("o", "open", "open the file", show=True),
        Binding("c", "color_by", "colour by", show=True),
    ]

    #: Give the full figure path its own auto-height row. Sharing a fixed row with
    #: the keys can clip a long path and push the navigation hints off screen.
    #: A truncated path looks copyable but does not identify the file.
    DEFAULT_CSS = """
    FigureScreen { layout: vertical; background: $surface; }
    FigureScreen #figure-caption { height: 1; padding: 0 2; }
    FigureScreen #figure-body { height: 1fr; padding: 0 1; align: center middle; }
    FigureScreen #figure { width: auto; height: 1fr; }
    FigureScreen #figure-note { height: auto; padding: 0 2; }
    FigureScreen #figure-path { height: auto; padding: 0 2; color: $text-muted; }
    FigureScreen #figure-hint { height: 1; padding: 0 2; color: $text-muted; }
    """

    def __init__(self, figures: list[tuple[str, str]], index: int = 0,
                 *, run_id: Optional[str] = None, entry=None, on_view=None, can_recolor=None,
                 on_view_at=None,
                 name: Optional[str] = None,
                 id: Optional[str] = None, classes: Optional[str] = None) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.figures = list(figures)
        # Clamped rather than trusted: the caller passes "the newest one", and a run whose last
        # step drew nothing would otherwise index past the end.
        self.index = min(max(0, index), max(0, len(self.figures) - 1))
        #: Rides along so a quickdrop can name the run it came from. A folder of `phate.pdf`
        #: files from five runs is a folder you cannot use.
        self.run_id = run_id
        self.entry = entry
        # on_view(step, path) lets RunScreen retain the new view through later feed syncs.
        # can_recolor() is called only on the UI thread; omitted means an archived figure.
        self.on_view = on_view
        # on_view_at(source_index, step, path) identifies the selected occurrence even
        # when repeated steps share a basename. The index precedes insertion of the view.
        self.on_view_at = on_view_at
        self.can_recolor = can_recolor
        self._color_busy = False
        self._color_cancel = threading.Event()
        self._view_notes: dict[str, str] = {}
        self._export = QuickdropWorker(self)

    def compose(self) -> ComposeResult:
        yield Static(id="figure-caption")
        with Vertical(id="figure-body"):
            yield AutoImage(id="figure")
        yield Static(id="figure-note")
        yield Static(id="figure-path")
        yield Static(id="figure-hint")

    def on_mount(self) -> None:
        self.draw()

    # ── the paint ────────────────────────────────────────────────────────────
    def draw(self) -> None:
        from rich.markup import escape

        body = self.query_one("#figure", AutoImage)
        caption = self.query_one("#figure-caption", Static)
        note = self.query_one("#figure-note", Static)
        where = self.query_one("#figure-path", Static)
        hint = self.query_one("#figure-hint", Static)

        if not self.figures:
            caption.update("[dim]no figures[/dim]")
            note.update("")
            where.update("")
            hint.update("esc back")
            return

        step, path = self.figures[self.index]
        counter = f"  [dim]{self.index + 1} of {len(self.figures)}[/dim]" \
            if len(self.figures) > 1 else ""
        caption.update(f"[bold]{escape(Path(path).name)}[/bold]  [dim]· {escape(step)}[/dim]"
                       f"{counter}")
        # A cell-based tier cannot show a scatter honestly — 60×20 cells is 60×40 pixels for
        # thousands of points — so it is labelled a thumbnail and pointed at the key that is
        # exact everywhere. Silence on the graphics tiers: there is nothing to apologise for.
        self._restore_view_note(path)
        alignment = self._view_notes.get(path, "")
        note.update("\n".join(filter(None, (_tier_note(), alignment))))
        where.update(escape(path))
        keys = "esc back  ·  s save for a paper  ·  o open  ·  c colour by"
        hint.update(keys + ("  ·  F7/F8 or ←/→ between figures" if len(self.figures) > 1 else ""))
        body.image = drawable(path)

    def _restore_view_note(self, path: str) -> None:
        """Restore archived alignment metadata off-thread, without loading coordinate arrays."""
        if path in self._view_notes:
            return
        if self.can_recolor is not None and not self.can_recolor():
            return
        self._view_notes[path] = ""  # also marks this read as pending across redraws

        def read_note():
            import json
            from manyruns import figspec

            try:
                metadata = json.loads(figspec._paths(path)[1].read_text())
            except (OSError, ValueError):
                return ""
            return (_POSITIONAL_NOTE if isinstance(metadata, dict) and metadata.get("positional")
                    else "")

        def restored(note):
            self._view_notes[path] = note
            if self.figures and self.figures[self.index][1] == path:
                self.draw()

        self._color_work(read_note, restored)

    # ── actions ──────────────────────────────────────────────────────────────
    def _recolor_ready(self) -> bool:
        if self.can_recolor is not None and not self.can_recolor():
            self.notify("finish this run before recolouring", severity="warning")
            return False
        return True

    def _color_work(self, job, done, cleanup=lambda result: None) -> None:
        """One daemon operation, delivered on the UI thread; no executor join on exit."""
        from rich.markup import escape

        app, cancelled = self.app, self._color_cancel

        def deliver(result, error):
            if cancelled.is_set() or not self.is_mounted:
                if error is None:
                    cleanup(result)
                return
            if error is not None:
                self._color_busy = False
                self.notify(escape(str(error)), severity="error", timeout=8)
            else:
                done(result)

        def run():
            result, error = None, None
            try:
                result = job()
            except Exception as exc:
                error = exc
            if cancelled.is_set():
                if error is None:
                    cleanup(result)
                return
            try:
                app.call_from_thread(deliver, result, error)
            except RuntimeError:  # app closed while this daemon was completing
                if error is None:
                    cleanup(result)

        threading.Thread(target=run, name="manyruns-colour", daemon=True).start()

    def action_color_by(self) -> None:
        from manyruns import figspec
        from rich.markup import escape

        if not self._recolor_ready():
            return
        if self._color_busy:
            self.notify("colour view already in progress", severity="information")
            return
        if not self.figures:
            self.notify("no figure to recolour", severity="warning")
            return
        if self.entry is None:
            self.notify("no dataset source for this figure", severity="warning")
            return
        source_index = self.index
        step, path = self.figures[source_index]
        entry = self.entry
        self._color_busy = True
        self._color_cancel = threading.Event()

        def inspect():
            if not Path(path).is_file():
                raise ValueError(f"figure is no longer available: {path}")
            spec = figspec.load(path)
            if spec is None:
                raise ValueError("no figure spec beside this PNG — re-run to get one")
            return spec, colorby.load_metadata(entry)

        def inspected(result):
            spec, metadata = result
            if not metadata.columns:
                self._color_busy = False
                self.notify(escape(metadata.empty_message), severity="warning", timeout=8)
                return

            def chosen(key):
                if key is None or self._color_cancel.is_set() or not self.is_mounted:
                    self._color_busy = False
                    return
                if not self._recolor_ready():
                    self._color_busy = False
                    return

                def viewed(result):
                    new_path, positional = result
                    self._color_busy = False
                    selected_index = self.index
                    self.figures.insert(source_index + 1, (step, new_path))
                    stayed_on_source = selected_index == source_index
                    self.index = (source_index + 1 if stayed_on_source
                                  else selected_index + (selected_index > source_index))
                    self._view_notes[new_path] = _POSITIONAL_NOTE if positional else ""
                    self.draw()
                    if self.on_view is not None:
                        self.on_view(step, new_path)
                    if self.on_view_at is not None:
                        self.on_view_at(source_index, step, new_path)
                    hint = (" — press o to open" if stayed_on_source
                            else " — use ←/→ to view it")
                    self.notify(escape(f"saved colour view: {new_path}") + hint, timeout=8)

                self._color_work(lambda: colorby.recolor_figure(path, metadata, key, spec=spec),
                                 viewed, lambda result: figspec.remove_view(result[0]))

            self.app.push_screen(colorby.ColorByScreen(metadata, current=spec.get("color_by")), chosen)

        self._color_work(inspect, inspected)

    def on_unmount(self) -> None:
        self._color_cancel.set()
        self._export.cancel()

    def action_step(self, delta: int) -> None:
        """Wraps, because a viewer of two figures where `→` stops working on the second reads as
        broken rather than as bounded."""
        if len(self.figures) > 1:
            self.index = (self.index + delta) % len(self.figures)
            self.draw()

    def action_save(self) -> None:
        """The quickdrop: you found the embedding you wanted, keep it.

        The toast names the FILES, not the folder. "Saved to figures/" is a sentence you have to
        act on to verify; the paths are the answer.
        """
        if not self.figures:
            return
        step, path = self.figures[self.index]
        self._export.save(path, step, self.run_id)

    def action_open(self) -> None:
        if not self.figures:
            return
        error = open_externally(self.figures[self.index][1])
        if error:
            self.notify(f"could not open it: {error}", severity="error", timeout=8)

    def action_back(self) -> None:
        self._color_cancel.set()
        self._export.cancel()
        self.dismiss(None)
