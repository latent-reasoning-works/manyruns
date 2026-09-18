"""The interactive front door — `manyruns` with no arguments drops you here.

This is the harness made conversational: **read** what the data is → **offer** only the moves
the data can honestly support → **run** the chosen recipe with a live step trace → **narrate**
the finding in plain English → loop. The compute is delegated — a `Session` over
`runner.apply_step`, the same step seam the batch path's recipe loop drives; this file owns the
*loop and the surface*, nothing scientific.

No LLM is required. Intent is rule-based by default (`manyruns.intent` Tier A); a free-typed
request is upgraded to a one-call Anthropic router only when the `[agents]` extra and a key are
both present. The rich/questionary imports are lazy, so the module imports (and tests, with an
injected prompter + console) without a terminal or those packages installed.
"""
from __future__ import annotations

import contextlib
from manyruns import env as _env
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

from manyruns import figures as _figures
from manyruns.narrate import Observation


# ── I/O seams (injectable, so the whole loop is testable off a real terminal) ────────────────
class RichPrompter:
    """Default on a real terminal: arrow-key menus + inline text via questionary (lazy import)."""

    def __init__(self, console: Any):
        self.console = console

    def select(self, message: str, choices: list[tuple[str, Any]], default: Any = None) -> Any:
        """A filterable dropdown: type to narrow, a number to jump, arrows to browse.

        The plain arrow-key list made a twelve-dataset menu a scrolling wall you had to read
        end to end. Typing beats scanning once a list is longer than a handful, and the
        number shortcuts keep the muscle memory the numbered fallback already taught."""
        import questionary

        qs = [questionary.Choice(title=t, value=v) for t, v in choices]
        extras = {
            "use_search_filter": True,
            # j/k as vim-style motion is mutually exclusive with prefix filtering — both
            # want the same keystrokes, and questionary raises rather than choosing for you
            "use_jk_keys": False,
            "use_shortcuts": True,
            "show_selected": True,
            "instruction": "(type to filter · ↑↓ · enter)",
        }
        try:
            return questionary.select(message, choices=qs, default=default, **extras).ask()
        except (TypeError, ValueError):
            # an older questionary, or a combination it refuses: fall back to the plain
            # list rather than crashing the front door over a menu affordance
            return questionary.select(message, choices=qs, default=default).ask()

    def text(self, message: str, default: str = "") -> str:
        import questionary

        return questionary.text(message, default=default).ask() or ""


class PlainPrompter:
    """No-TTY fallback: a numbered `input()` menu — no questionary, no terminal needed.

    Used when stdin isn't a terminal (pipes, CI, dumb shells) or questionary is absent, so the
    front door degrades to a plain prompt instead of crashing. EOF / Ctrl-C reads as 'leave'."""

    def __init__(self, console: Any):
        self.console = console

    def select(self, message: str, choices: list[tuple[str, Any]], default: Any = None) -> Any:
        self.console.print(message)
        for i, (title, _v) in enumerate(choices, 1):
            self.console.print(f"  {i}. {title}")
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1][1]
        for title, v in choices:                       # accept a substring of a choice's label
            if raw.lower() in title.lower():
                return v
        return default

    def text(self, message: str, default: str = "") -> str:
        self.console.print(message)
        try:
            return input("> ").strip() or default
        except (EOFError, KeyboardInterrupt):
            return default


def _default_prompter(console: Any) -> Any:
    """RichPrompter on an interactive terminal with questionary; PlainPrompter otherwise."""
    import sys

    if sys.stdin.isatty():
        try:
            import questionary  # noqa: F401
        except ImportError:
            return PlainPrompter(console)
        return RichPrompter(console)
    return PlainPrompter(console)


class _PlainConsole:
    """Fallback console when rich isn't installed — strips markup and prints plainly."""

    is_terminal = False

    def print(self, *args: Any) -> None:
        import re

        s = " ".join(str(a) for a in args)
        print(re.sub(r"\[/?[a-z0-9 =#]+\]", "", s))


def _default_console() -> Any:
    try:
        from rich.console import Console
    except ImportError:
        return _PlainConsole()
    return Console()


#: outcome → (glyph, colour). One place, so the panel and any future surface agree.
_OUTCOME_STYLE = {
    "ok": ("✓", "green"),
    "skipped": ("⊘", "yellow"),
    "error": ("✗", "red"),
    "reported": ("•", "cyan"),
}


def state_rows() -> list[tuple[str, str]]:
    """Where you are, before you choose anything: where your data would come from, how it
    would run, and what is in the catalog.

    A bare menu answers none of those, so the first question — "where do I put my file" —
    had to be asked rather than read. Pure data, so the plain and panel renderings share it."""
    from manyruns import app
    from manyruns.catalog import discover_datasets, discover_recipes

    # The count is over EVERY drop folder (`data_dirs`), because that is what the roster lists;
    # the folder NAMED is where a fetch lands. Counting one folder while listing two is how a
    # header comes to disagree with the screen under it.
    found = drop_folder_entries()
    folders = data_dirs()
    ddir = data_dir()

    def _short(p: Path) -> str:
        return str(p).replace(str(Path.home()), "~")

    where = _short(ddir)
    also = [_short(p) for p in folders if not p.samefile(ddir)] if ddir.is_dir() else \
        [_short(p) for p in folders]
    data = f"{len(found)} in {where}" if found else f"empty — drop files into {where}"
    if also:
        data += f"  (+ {', '.join(also)})"

    # None when nothing is installed — `_default_engine` no longer falls back to a dev engine,
    # so the header says "install one" rather than naming a backend that invents numbers.
    engine = app._default_engine()
    missing = [n for n in app.ENGINE_NAMES
               if n not in app.DEV_ENGINES and not app.engine_available(n)]
    if engine is None:
        how = "[yellow]none installed[/yellow] — reinstall: the compute ships with manyruns"
    else:
        how = engine + (f"   [dim](not installed: {', '.join(missing)})[/dim]" if missing else "")

    return [
        ("data", data),
        ("engine", how),
        ("catalog", f"{len(discover_recipes())} recipes · {len(discover_datasets())} samples"),
    ]


def _render_state(console: Any) -> None:
    rows = state_rows()
    if not getattr(console, "is_terminal", False):
        for label, value in rows:
            console.print(f"  {label:<8} {value}")
        console.print("")
        return
    from rich.panel import Panel
    from rich.table import Table

    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="cyan", justify="right")
    grid.add_column()
    for label, value in rows:
        grid.add_row(label, value)
    # tight: this now prints above every prompt, so the vertical padding it carried when it
    # appeared once at startup would cost a third of the screen over a few rounds
    console.print(Panel(grid, border_style="dim", padding=(0, 2)))


def render_result(console: Any, results: dict) -> None:
    """Draw a run as PANELS — what ran, then what it means.

    `narrate.run_panel` builds the same content as a plain string and stays the fallback:
    it has no dependencies, so it works when rich is absent and when output is piped (where
    box-drawing is noise). This is the terminal rendering of the same facts, not a second
    source of them — which the geometry did not honour until now: the g-vector panel was
    drawn here and nowhere else, so piping the output silently dropped every measurement.
    Both paths now read `narrate.geometry_sections` / `geometry_delta_rows`."""
    from manyruns import narrate

    if not getattr(console, "is_terminal", False):
        console.print(narrate.run_panel(results))
        console.print(f"\n  {narrate.narrate(results)}\n")
        return

    from rich.panel import Panel
    from rich.table import Table

    g = results.get("g_vector") or {}
    shape = ""
    if g.get("n_samples") and g.get("n_features"):
        shape = f" · {int(g['n_samples']):,} × {int(g['n_features']):,}"
    title = f"[bold]{results.get('recipe') or '(adaptive)'}[/bold] · {results.get('engine')}{shape}"

    # top row: what ran (narrow, left) beside what it means (wide, right)
    top = Table.grid(padding=(0, 1), expand=True)
    top.add_column(width=30)
    top.add_column(ratio=1)
    finding = "\n".join(line.strip() for line in narrate.narrate(results).splitlines())
    top.add_row(
        Panel(_steps_body(results, narrate), title="steps", title_align="left",
              border_style="cyan", padding=(1, 1)),
        Panel(finding, title="[bold]what it means[/bold]", title_align="left",
              border_style="green", padding=(1, 2)),
    )
    console.print(Panel(top, title=title, title_align="left", border_style="dim", padding=(0, 0)))

    geometry = _geometry_panel(g)
    if geometry is not None:
        console.print(Panel(geometry, title="what this run recorded", title_align="left",
                            border_style="magenta", padding=(1, 2)))

    _render_plots(console, results)


#: A step nobody has reached yet. Not an OUTCOME — a live state, and keeping it out of the
#: outcome vocabulary is what stops a dashboard inventing a fifth way a step can have ended.
_PENDING = ("·", "dim", "queued")


class _RunningFor:
    """Elapsed time that recomputes at RENDER time, not at update time.

    This is what makes the dashboard alive without a worker thread. `rich.Live` repaints
    from its own refresh thread — measured, 7 repaints while the main thread was fully
    blocked in a matrix decomposition — so anything that reads the clock when it renders
    keeps ticking through a blocking fit. A pre-formatted string would freeze.

    That is the whole reason there is no threading here. An earlier draft of the spec moved
    the engine to a worker thread and gave the UI a `prompt_toolkit` event loop, justified
    by "a two-minute PHATE fit". Measured, a fit is 2.5s at 2,700 cells and 4.4s at 20,000,
    and the full `cflows` recipe on the in-process loop is 2.3s end to end. Against numbers like
    that, a spinner is the honest amount of machinery."""

    def __init__(self, started: float) -> None:
        self.started = started

    def __rich_console__(self, console: Any, options: Any) -> Any:
        from rich.text import Text

        yield Text(f"{time.monotonic() - self.started:5.1f}s", style="dim")


def live_dashboard(console: Any, recipe: dict, title: str) -> Any:
    """A dashboard that exists BEFORE the run does.

    It opens showing every declared step as queued, then fills in as each one lands — so a
    scientist watching a two-minute embedding sees which step is running and what the ones
    before it produced, instead of a frozen terminal and a wall of output at the end.

    Returns `(live, on_step)`. `on_step` is the observer handed to the step loop; it is only
    ever a renderer, so `runner._report` swallows anything it raises rather than taking the
    run down with the display."""
    from rich.live import Live

    declared = list(recipe.get("steps") or [])

    def frame(records: Optional[list] = None) -> Any:
        return _progress_panel(declared, records or [], title)

    live = Live(frame(), console=console, refresh_per_second=8, transient=False)
    # The figure pane rides the same observer. It draws a step's plot as that step lands —
    # which is the only moment it is the CURRENT picture rather than one of a pile at the end.
    # It owns the stop/start of the live region, because an image escape cannot be written
    # into a region rich repaints by line count (see `figures.FigurePane.show`).
    pane = _figures.FigurePane(console)

    def on_step(rec: dict, records: list) -> None:
        live.update(frame(records))
        if rec.get("plots"):
            pane.show(rec, live)
            live.update(frame(records))

    return live, on_step


def _progress_panel(declared: list, records: list, title: str) -> Any:
    """Declared steps, overlaid with whatever has actually happened so far."""
    from rich.panel import Panel
    from rich.table import Table

    from manyruns import narrate

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="center", width=1)
    grid.add_column(style="bold", min_width=14)
    grid.add_column(min_width=10)
    grid.add_column(justify="right", min_width=7)

    # Only the NEWEST step's geometry, unlike `_steps_body` which keeps every step's. Measured
    # on the shipped suite, five metrics land per embedding step, and `rich.Live` has to
    # repaint the whole panel inside one screen — so every step's rows would grow the live
    # view without bound. This one is scaffolding for the wait and shows what just landed;
    # the finished panels carry the whole record.
    newest = next((r for r in reversed(records) if r.get("geometry")), None)

    by_index = {r.get("index"): r for r in records}
    for i, step in enumerate(declared):
        rec = by_index.get(i)
        name = str(step.get("name") or "?")
        if rec is None:
            glyph, colour, word = _PENDING
            grid.add_row(f"[{colour}]{glyph}[/{colour}]", name,
                         f"[{colour}]{word}[/{colour}]", "")
        elif rec.get("state") == "running":
            # a live spinner and a live clock, both animated by Live's own refresh thread
            from rich.spinner import Spinner

            grid.add_row(Spinner("dots", style="cyan"), f"[cyan]{name}[/cyan]",
                         "[cyan]running…[/cyan]", _RunningFor(rec.get("started") or time.monotonic()))
        else:
            glyph, colour = _OUTCOME_STYLE.get(rec.get("outcome"), ("?", "white"))
            grid.add_row(f"[{colour}]{glyph}[/{colour}]", name,
                         f"[{colour}]{rec.get('outcome')}[/{colour}]",
                         f"{float(rec.get('seconds') or 0.0):.2f}s")
            if rec is newest:
                for row in narrate.geometry_delta_rows(rec):
                    grid.add_row("", f"  [dim]{row.label}[/dim]",
                                 f"[magenta]{row.value}[/magenta]", "")

    done = sum(1 for r in records if r.get("state") != "running" and r.get("outcome"))
    footer = f"\n[dim]{done} of {len(declared)} complete[/dim]"
    from rich.console import Group

    return Panel(Group(grid, footer), title=title, title_align="left",
                 border_style="cyan", padding=(1, 2))


def _steps_body(results: dict, narrate: Any) -> Any:
    """The left column: one line per step attempt, plus the honest count."""
    from rich.console import Group
    from rich.table import Table

    steps = results.get("steps") or []
    body = Table.grid(padding=(0, 1))
    body.add_column(justify="center", width=1)
    body.add_column(style="bold")
    body.add_column(justify="right", style="dim")

    if steps:
        for s in steps:
            glyph, colour = _OUTCOME_STYLE.get(s.get("outcome"), ("?", "white"))
            body.add_row(f"[{colour}]{glyph}[/{colour}]", str(s.get("name") or "?"),
                         f"{float(s.get('seconds') or 0.0):.2f}s")
            if s.get("detail"):
                body.add_row("", f"[dim]{narrate._clip(s['detail'], 24)}[/dim]", "")
            # what the step actually contributed — a step can report "ok" and produce
            # nothing at all, which the outcome alone cannot distinguish from real work.
            made = s.get("produced")
            emitted = s.get("emitted") or []
            if made:
                body.add_row("", f"[dim]→ {narrate._clip(made, 24)}[/dim]", "")
            if emitted:
                body.add_row("", f"[dim]+ {len(emitted)} value"
                                 f"{'s' if len(emitted) != 1 else ''}[/dim]", "")
            elif not made and s.get("outcome") == "ok":
                body.add_row("", "[yellow]produced nothing[/yellow]", "")
            # what the step did to the GEOMETRY, which `produced` cannot say: a shape can be
            # identical either side of a step that reorganised every point in it. Every row
            # is kept — picking a subset would be a ranking of the suite that nothing has
            # earned: values where computed, no ranking.
            for row in narrate.geometry_delta_rows(s):
                body.add_row("", f"[dim]{row.label}[/dim]", f"[magenta]{row.value}[/magenta]")
        ran = sum(1 for s in steps if s.get("outcome") == "ok")
        total = sum(float(s.get("seconds") or 0.0) for s in steps)
        footer = f"\n[bold]{ran} of {len(steps)}[/bold] ran · {total:.2f}s"
    else:
        # an engine that reports no outcomes must not look like one that ran cleanly
        for entry in results.get("trace") or []:
            body.add_row("[dim]·[/dim]", entry, "")
        footer = "\n[yellow]no per-step outcome from\nthis engine — not evidence\nanything ran[/yellow]"
    if results.get("seed") is None and steps:
        footer += "\n[yellow]⚠[/yellow] [dim]no seed — not\nreproducible[/dim]"
    return Group(body, footer)


def _geometry_panel(g: dict) -> Any:
    """The g-vector, drawn in sections. Returns None when there is nothing to draw.

    `narrate.geometry_sections` decides what belongs where and the plain fallback renders the
    same split — this only picks the ink. Two things the ink has to carry:

    * an ABSENT metric is drawn in a different colour from a measured one. Its value is a
      sentence, not a number, and in a right-aligned column of `0.6481`s a lone "returned nan"
      reads as a value until you look twice.
    * the caption on the measured section is `suite_null`, printed verbatim rather than
      paraphrased — it is the sentence that keeps a table of embedding-derived numbers from
      being read as evidence about the data (`vocab.NULL_KIND`)."""
    from rich.console import Group
    from rich.markup import escape
    from rich.table import Table

    from manyruns import narrate

    sections = narrate.geometry_sections(g)
    if not sections:
        return None
    blocks: list[Any] = []
    for section in sections:
        if blocks:
            blocks.append("")
        head = f"[bold]{section.title}[/bold]"
        if section.note:
            head += f"  [dim]{escape(section.note)}[/dim]"
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="cyan", justify="right")
        grid.add_column(justify="right")
        # The reason gets its OWN column so the numbers stay in a narrow aligned one. Sharing
        # a column, a 30-character sentence sets the width for the whole section and pushes
        # every value to the far right of it, away from the label it belongs to.
        grid.add_column()
        for row in section.rows:
            if row.measured:
                grid.add_row(row.label, f"[bold]{row.value}[/bold]", "")
            else:
                # escaped: an absence's reason can be an exception message, and
                # `raised KeyError: ['x']` would otherwise be parsed as markup by rich
                grid.add_row(row.label, "[yellow]—[/yellow]",
                             f"[yellow]not measured[/yellow] [dim]— {escape(row.value)}[/dim]")
        blocks.extend([head, grid])
    return Group(*blocks)


def _render_plots(console: Any, results: dict) -> None:
    """Pull the figure forward. It is the thing a user actually wants to look at, and it was
    previously reachable only by reading a path out of `summary.md`.

    Rendered inline where the terminal speaks a graphics protocol (iTerm2, kitty); elsewhere
    the path is printed prominently, because most terminals make it click-openable."""
    from rich.panel import Panel

    plots = [Path(p) for p in (results.get("plots") or [])]
    plots = [p for p in plots if p.exists()]
    if not plots:
        return
    shown = [p for p in plots if _emit_inline_image(console, p)]
    lines = [f"[bold]{p.name}[/bold]  [dim]{p}[/dim]" for p in plots]
    if not shown:
        lines.append(f"[dim]{_no_inline_hint()}[/dim]")
    console.print(Panel("\n".join(lines), title="figure", title_align="left",
                        border_style="yellow", padding=(1, 2)))


#: The inline-image machinery lives in `manyruns.figures` — it is a feature in its own right
#: (a pane that follows the running step), and the run-summary panel below is only one of its
#: two callers. Bound here under the names this module's callers already use.
_image_protocol = _figures.protocol
_image_escape = _figures.escape
_no_inline_hint = _figures.hint


def _emit_inline_image(console: Any, path: Path, cols: int = 60) -> bool:
    return _figures.draw(console, path, cols)


# ── the loop ─────────────────────────────────────────────────────────────────────────────────
def run_shell(args: Any, *, prompter: Any = None, console: Any = None) -> int:
    """The home loop: pick a move, do it, come back. Returns a process exit code."""
    console = console or _default_console()
    prompter = prompter or _default_prompter(console)

    from manyruns import narrate

    console.print(f"[bold]manyruns[/bold] — {narrate.TAGLINE}")
    console.print("[dim]interactive mode · type a number, or free text · Ctrl-C to leave[/dim]\n")

    while True:
        # Re-rendered every time round, not once at startup: the state it reports CHANGES
        # (a file dropped, a project run) and a header that scrolled away five screens ago
        # is not a header. Deliberately re-printed rather than pinned with a Live region —
        # questionary drives the terminal during a prompt, and two things owning the cursor
        # is how a TUI starts corrupting its own output.
        _render_state(console)
        verb = _home(prompter)
        if verb in (None, "quit"):
            console.print("bye.")
            return 0
        try:
            if verb == "explore":
                _explore(args, prompter, console)
            elif verb == "check":
                _check(console)
            elif verb == "recipes":
                _list_recipes(console)
            elif verb == "datasets":
                _list_datasets(console)
        except KeyboardInterrupt:
            console.print("[dim](cancelled — back to the menu)[/dim]")
        console.print("")


def _home(prompter: Any) -> Optional[str]:
    from manyruns import verbs

    choice = prompter.select(
        "What do you want to do?",
        [(v.summary, v.name) for v in verbs.VERBS],
        default="explore",
    )
    return choice


# ── explore: read → offer → run → narrate ──────────────────────────────────────────────────────
def _explore(args: Any, prompter: Any, console: Any) -> None:
    from manyruns import intent, narrate
    from manyruns.catalog import discover_recipes, load_recipe
    from manyruns.vocab import unmet

    picked = _pick_source(prompter, console)
    if picked is None:
        return
    data_folder, dataset, modality, obs = picked

    console.print(f"\n[bold]{obs.source}[/bold]")
    console.print(f"  [green]✓ data loaded[/green] — {_loaded_stats(obs)}")
    console.print(f"  {narrate.describe(obs)}\n")

    # the legal-move set: recipes whose preconditions the data actually meets (the calculus).
    # `obs.provides()` is the shell's supplier of dataset-level facts — the counterpart to a
    # declaration's `handle`, which this surface does not have. Without it, dropping an .h5ad
    # in would refuse every step that reads the gene axis it demonstrably has.
    legal = [n for n in discover_recipes()
             if not unmet(load_recipe(n), obs.shape, provided=obs.provides())]
    if not legal:
        console.print("[yellow]No recipe can run on this data as described.[/yellow] "
                      "Try `check` to see what each recipe needs.")
        return

    recipe_name = _pick_recipe(prompter, console, obs, legal, allow_llm=intent.llm_available())
    if recipe_name is None:
        return

    engine = _pick_engine(prompter, console, args)
    if engine is None:
        return

    results = _run(args, console, data_folder, dataset, modality, recipe_name, obs, engine)
    if results is not None:
        _record(console, results, obs, modality)
        console.print("")
        render_result(console, results)


def _record(console: Any, results: dict, obs: Observation, modality: str) -> None:
    """Append this run to the store — the front door's missing half of §4.3 P1.

    `store.append` had exactly ONE product caller (`app._explore_project`, the `run` /
    `init --batch` path), and `shell.py` contained no reference to `store` at all. So every
    run started from the bare `manyruns` front door — the default entry point — wrote its
    state folder under `outputs/<project>/state/<run_id>/` and left no index row pointing at
    it. Measured by deleting this one line again: the same `cflows` run through `run_shell`
    still completes and returns 0, and `store.read` finds zero rows for it.

    That is the prerequisite for resume, not a bookkeeping nicety: a resume menu is
    `store.read` → `artifacts.complete()` → `load_array`, and it starts at the row. An
    artifact whose row is missing is reachable only by knowing the run id you never saw.

    The DEFAULT `out_dir` on purpose, so this writes the same `outputs/index.jsonl` the
    `run` path writes. Two index files would be two homes for one history, and the questions
    the store exists for ("every run on this cohort", "has this spec run before") are exactly
    the ones a split history answers wrongly rather than not at all.
    """
    from manyruns import store as _store

    try:
        index = _store.append(results, extra={"source": str(obs.source), "modality": modality})
    except OSError as e:  # noqa: BLE001 - a read-only cwd must not lose a completed run
        console.print(f"[dim](run not recorded: {e})[/dim]")
        return
    console.print(f"[dim]run {results.get('run_id')} recorded in {index}[/dim]")


def _pick_engine(prompter: Any, console: Any, args: Any) -> Optional[str]:
    """How should this actually run?

    Backends whose dependencies are missing are shown but MARKED, rather than hidden: a user
    who wants a real run needs to know it is *installable*, not that it is absent. Development
    backends (`app.DEV_ENGINES`) are the exception — they are omitted entirely, because `mock`
    invents a g-vector without opening the file and the panel it produces is indistinguishable
    at a glance from a measured one. It is not a fidelity a scientist should be choosing
    between; it is a test fixture.

    `--engine` still wins, so the flag path and every test that passes `--engine mock` are
    unchanged."""
    from manyruns import app

    chosen = getattr(args, "engine", None)
    if chosen:
        return chosen

    options, default = [], None
    for name, description, _ in app.ENGINES:
        # A development backend is not a choice a scientist should be shown. `mock` invents a
        # g-vector and never opens the file, and the panel it produces is indistinguishable at
        # a glance from a measured one — see `app.DEV_ENGINES`.
        if name in app.DEV_ENGINES:
            continue
        ok = app.engine_available(name)
        if ok and default is None:
            default = name
        label = f"{name} — {description}" if ok else f"{name} — {description}  (not installed)"
        # an unavailable pick carries its own sentinel rather than None, which the caller
        # cannot distinguish from "the user backed out".
        options.append((label, name if ok else f"__missing__{name}"))
    options.append(("← back", "__back__"))

    # No usable backend: there is nothing to ask about, and offering a menu of things that
    # are all "(not installed)" is a question with no right answer.
    if default is None:
        from rich.markup import escape
        from manyruns.installation import REINSTALL

        console.print("[yellow]no compute backend installed[/yellow] — "
                      f"reinstall with `{escape(REINSTALL)}`")
        return None
    picked = prompter.select("How should I run it?", options, default=default)
    if picked is None or picked == "__back__":
        return None
    if isinstance(picked, str) and picked.startswith("__missing__"):
        name = picked.removeprefix("__missing__")
        console.print(f"[yellow]{name} isn't installed here.[/yellow] "
                      "Install it, or pick another way to run.")
        return None
    return picked


_DATA_SUFFIXES = {".h5ad", ".h5", ".csv", ".tsv", ".txt", ".loom", ".mtx", ".npy"}


def data_dir() -> Path:
    """The drop folder — where a user puts their own data, once.

    `$MANYRUNS_DATA_DIR` wins. Otherwise `./data` **when it already exists**, so a project
    checkout keeps its own; failing that `~/.manyruns/data`.

    The home-directory fallback is the point. A folder resolved from the current directory
    silently becomes a *different* folder every time you `cd`, so "drop your data here" only
    holds until you move — and `outputs/` lands somewhere else too. One stable place means
    the answer to "where do I put my file" does not depend on where you happened to launch."""
    env = _env.get("DATA_DIR")
    if env:
        return Path(env).expanduser()
    local = Path("data")
    if local.is_dir():
        return local
    return Path.home() / ".manyruns" / "data"


def ensure_drop_folder() -> Path:
    """Create the chosen drop folder, including parents; let OSError explain a refusal."""
    folder = data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def data_dirs() -> list[Path]:
    """EVERY drop folder, in the order a row should be offered. Usually one, sometimes two.

    `data_dir()` answers "where does a fetch land" and has to be a single path. This answers
    "where do I look for the scientist's files", which is a different question — and the two
    were the same function, which cost a real dataset.

    Measured: `data/pbmc3k_raw.h5ad` (5.6 MB) sat in the checkout, `/data/` is gitignored
    (`.gitignore:103`), and a git worktree therefore has no `data/`. `data_dir()` prefers
    `./data` **when it exists** and otherwise falls back to `~/.manyruns/data` — so from the
    worktree the fallback won, the roster read an empty folder, and the file was simply gone
    from the front door. Not lost, but unreachable, which looks the same from the app.

    That is the failure `data_dir`'s own docstring says the home fallback exists to prevent —
    "a folder resolved from the current directory silently becomes a *different* folder every
    time you `cd`". True, and the `./data`-wins branch in front of it reintroduces exactly that.

    So the two are ADDITIVE rather than exclusive: the stable home folder is always read, and a
    checkout that keeps its own `data/` is read as well. Nothing a person dropped anywhere stops
    being visible because of where they launched.
    """
    # AN EXPLICIT OVERRIDE MEANS THAT FOLDER, AND ONLY IT. `$MANYRUNS_DATA_DIR` is how a test
    # points the product at a tmp dir and how a user pins a cohort directory; adding the home
    # folder on top would mean a suite run picks up whatever is sitting in the developer's
    # `~/.manyruns/data` — isolation lost, and lost intermittently, which is worse.
    env = _env.get("DATA_DIR")
    if env:
        chosen = Path(env).expanduser()
        return [chosen] if chosen.is_dir() else []

    # `data_dir()` FIRST, because it is where a fetch lands: if the folder you download into
    # did not win a name clash, you could fetch `cohort.h5ad` and still be shown a different
    # file of that name. The rest follow so nothing dropped anywhere goes missing.
    seen: list[Path] = []
    for candidate in (data_dir(), Path.home() / ".manyruns" / "data", Path("data")):
        resolved = candidate.expanduser()
        if resolved.is_dir() and not any(resolved.samefile(s) for s in seen):
            seen.append(resolved)
    return seen


def drop_folder_entries(ddir: Optional[Path] = None) -> list[Path]:
    """Every droppable thing in the drop folder(s), by name. Shared by the shell's picker and
    the non-interactive `run`, so both see the same set.

    An explicit `ddir` reads that folder ALONE — a caller naming a folder means that one, and a
    test pointing at a tmp dir must not also pick up whatever is in the developer's home."""
    if ddir is not None:
        return [payload for _, (_, payload) in _scan_drop_folder(ddir)]
    out: list[Path] = []
    names: set[str] = set()
    for folder in data_dirs():
        for _, (_, payload) in _scan_drop_folder(folder):
            # First folder wins a name clash — `data_dirs()` puts `data_dir()` first, so the
            # folder a fetch lands in is also the one shown. Two identical rows that open
            # different files is worse than either.
            if payload.name not in names:
                names.add(payload.name)
                out.append(payload)
    return out


#: The prefix of a `kind: path` ref that names a GENERATOR rather than a file
#: (`synthetic:time-course`). `pipeline.loading._GENERATED` owns the set of them; the prefix is
#: all this module needs, and it is tested BEFORE any `.exists()` for the same reason it is
#: there — a generated ref was never going to be on disk, so "not found" is the wrong report.
_GENERATED_REF = "synthetic:"


def dataset_ref_path(ref: str) -> Optional[Path]:
    """Where a dataset declaration's `kind: path` ref actually IS on this machine — or None.

    THE REF IS NOT A PATH UNTIL SOMETHING ANCHORS IT, and nothing did. `pbmc3k.yaml` declares
    `ref: data/pbmc3k_raw.h5ad`, `discover_datasets()` reads that YAML out of the WHEEL via
    `importlib.resources`, so the row is on the menu of every install — while `Path(ref)` is
    resolved against the current directory. Measured on this checkout, 2026-08-17, calling
    `_resolve_dataset("pbmc3k")` from two directories of one install:

        cwd=<repo root>   ->  Observation(n_obs=2700, n_vars=32738)
        cwd=/private/tmp  ->  Observation(n_obs=None,  n_vars=None)

    Same name, same install, two different datasets — and the second one is a declaration with
    nothing behind it, which `Observation.provides()` then reports as carrying no gene axis. A
    `cd` is not supposed to change what a dataset is.

    So the ref is resolved against ANCHORS, in this order, first hit wins:

    1. the ref as written (absolute, or relative to the current directory) — this is the route
       that already worked from the repo root, and it keeps working;
    2. the ref's BASENAME inside each drop folder (`data_dirs()`, i.e. `$MANYRUNS_DATA_DIR`,
       `~/.manyruns/data`, `./data`).

    (2) also resolves verified first-use downloads: the drop folder is where the product tells
    you to put a file, and where a fetch lands. Basename rather than the whole ref, because
    `data/` is a fragment of
    THIS checkout's layout, not of anyone's home directory.

    Matching on a basename can in principle find the wrong file — a `~/.manyruns/data/
    matrix.mtx` is not necessarily the `cohorts/p01/matrix.mtx` some `$MANYRUNS_DATASET_DIR`
    declaration meant. That is why the ref as written wins, why the resolved path is PRINTED by
    `_list_datasets` when it differs from the ref. Existing files are user inputs, so nothing
    here verifies a checksum; only the downloader verifies the bytes it fetches.

    Returns None for a generated ref (there is no file to find) and for a ref that is nowhere;
    `missing_dataset_file` is the caller that needs to tell those two apart.
    """
    if not ref or ref.startswith(_GENERATED_REF):
        return None
    literal = Path(ref).expanduser()
    if literal.exists():
        return literal
    if literal.is_absolute():
        return None
    for folder in data_dirs():
        candidate = folder / literal.name
        if candidate.exists():
            return candidate
    return None


def missing_dataset_file(ds: Any) -> Optional[str]:
    """The ref of a declared dataset whose bytes are NOT on this machine, else None.

    The half of the fix that stops the listing and the run from disagreeing. Before it, an
    unresolvable `path` handle fell through to a declared-shape `Observation` — so `pbmc3k` was
    offered as a loadable sample on every install, printed "✓ data loaded", and the run then
    failed on a path that was never there. A row that cannot run has to SAY it cannot run.

    None for a `manylatents` handle (the engine loads it by name, there is no file here to
    check) and for a generated `synthetic:` ref (there is no file by construction) — inventing
    a "missing" for either would refuse the twelve bundled synthetics and the time course.
    """
    handle = (ds or {}).get("handle") or {}
    if handle.get("kind") != "path":
        return None
    ref = str(handle.get("ref") or "")
    if not ref or ref.startswith(_GENERATED_REF):
        return None
    return None if dataset_ref_path(ref) is not None else ref


def _scan_drop_folder(ddir: Path) -> list[tuple[str, Any]]:
    """Droppable entries: data files + subfolders (per-sample / per-condition), newest first."""
    if not ddir.is_dir():
        return []
    entries = []
    for p in ddir.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_dir():
            entries.append((p, f"{p.name}/  (folder)"))
        elif p.suffix.lower() in _DATA_SUFFIXES:
            entries.append((p, p.name))
    entries.sort(key=lambda e: e[0].name.lower())
    return [(label, ("path", p)) for p, label in entries]


def _pick_source(
    prompter: Any, console: Any
) -> Optional[tuple[Optional[Path], Optional[str], str, Observation]]:
    """Pick a data source: something you dropped in `./data`, a built-in sample, or a typed path.

    The drop folder is the primary surface — drag a file into it and pick "rescan". Returns
    (data_folder, dataset, modality, Observation), or None to go back."""
    from manyruns.catalog import discover_datasets, load_dataset
    from manyruns import datasetfetch
    from rich.markup import escape

    ddir = ensure_drop_folder()

    while True:
        dropped = _scan_drop_folder(ddir)
        choices: list[tuple[str, Any]] = list(dropped)
        for n in discover_datasets():
            try:
                ds = load_dataset(n)
            except Exception:  # noqa: BLE001 - a broken dataset file shouldn't hide the rest
                continue
            label = f"{n}  (sample · {ds.get('shape', '?')})"
            # A DECLARATION THAT CANNOT BE MATERIALISED HERE SAYS SO, on the row. The catalog
            # ships in the wheel and the bytes do not, so `pbmc3k` is on this menu on every
            # install while its file is in exactly one checkout; without this it read as a
            # loadable sample, printed "✓ data loaded" off the declared shape alone, and failed
            # at the run on a path that was never there.
            if missing_dataset_file(ds):
                label += "  — not on this machine"
                if datasetfetch.can_fetch(ds):
                    label += f" · download on selection · {ds['handle']['bytes'] / 1024**2:.1f} MB"
            choices.append((label, ("dataset", n)))
        choices.append(("a file or folder path…", ("path", None)))
        choices.append((f"↻ rescan {ddir}", ("rescan", None)))
        choices.append(("← back", ("back", None)))

        if not dropped:
            console.print(f"[dim]drop your data into {ddir.resolve()}, then pick “rescan” — "
                          f"or choose a sample below.[/dim]")

        kind, val = prompter.select("Which data?", choices) or ("back", None)
        if kind == "back":
            return None
        if kind == "rescan":
            continue
        if kind == "dataset":
            ds = load_dataset(val)
            if missing_dataset_file(ds) and datasetfetch.can_fetch(ds):
                _say_where_the_file_should_be(console, val)
                choice = prompter.select("Download this sample?", [("Fetch", "fetch"),
                                                                   ("Back", "back")],
                                         default="back")
                if choice != "fetch":
                    console.print("  Download declined. Select the sample again to retry.")
                    continue
                # At most eleven updates, independent of transport chunk size or speed.
                last_bucket = -1

                def progress(done: int, total: int) -> None:
                    nonlocal last_bucket
                    bucket = min(10, done * 10 // total)
                    if bucket > last_bucket:
                        last_bucket = bucket
                        console.print(f"  downloading {escape(Path(ds['handle']['ref']).name)}: "
                                      f"{done:,}/{total:,} bytes ({done * 100 // total}%)")

                try:
                    datasetfetch.fetch_dataset(ds, progress=progress)
                except datasetfetch.DatasetFetchError as error:
                    console.print(f"  [yellow]{escape(str(error))}[/yellow]")
                    _say_where_the_file_should_be(console, val)
                    continue
            if not _say_where_the_file_should_be(console, val):
                continue                                # explained; re-show the menu
            return _resolve_dataset(val)

        p = val
        if p is None:                                       # "a file or folder path…"
            raw = prompter.text("Path to a file or folder:").strip()
            if not raw:
                continue
            p = Path(raw).expanduser()
        resolved = _resolve_path_source(Path(p), console)
        if resolved is not None:
            return resolved                                 # else: re-show the menu


def where_the_file_should_be(name: str, ds: dict) -> list[str]:
    """Surface-neutral placement instructions, or no lines for local/generated data.

    Shared by UI/CLI callers: no printing, hashing or network. The direct download URL wins
    over a citation, and the full destination plus pins also supports an offline first use.
    """
    ref = missing_dataset_file(ds)
    if ref is None:
        return []
    handle = ds.get("handle") or {}
    target = data_dir().resolve() / Path(ref).name
    lines = [f"{name} is declared here, but its data is not on this machine.",
             f"looked for {Path(ref).name} at {ref} and in "
             f"{', '.join(str(d) for d in data_dirs()) or '(no drop folder yet)'}"]
    url = handle.get("url") or (ds.get("source") or {}).get("url")
    if url:
        lines.append(f"download {Path(ref).name} from {url}")
    if "bytes" in handle:
        lines.append(f"expected {handle['bytes']} bytes ({handle['bytes'] / 1024**2:.1f} MB)")
    if "sha256" in handle:
        lines.append(f"expected sha256: {handle['sha256']}")
    lines.append(f"put the file at {target}, then choose the sample again once it is in place.")
    return lines


def _say_where_the_file_should_be(console: Any, name: str) -> bool:
    """Print the refusal with rich-shell hints; true means the data is local/generated."""
    from manyruns.catalog import load_dataset
    from rich.markup import escape

    try:
        ds = load_dataset(name)
    except Exception:  # noqa: BLE001 - the picker already skips a YAML that will not load
        return True
    lines = where_the_file_should_be(name, ds)
    for line in lines:
        console.print(f"  {escape(line)}")
    if lines:
        console.print("  To refresh this menu after placing the file, pick “rescan”.")
        console.print("  Or choose “a file or folder path…” to use another local file.")
    return not lines


def _resolve_dataset(name: str) -> tuple[Optional[Path], Optional[str], str, Observation]:
    from manyruns import narrate
    from manyruns.catalog import load_dataset

    ds = load_dataset(name)
    handle = ds.get("handle", {})
    modality = ds.get("modality", "unknown")
    # a manylatents handle → the engine loads it by ref; a path handle → we load the file.
    if handle.get("kind") == "path":
        ref = str(handle["ref"])
        # ANCHORED, not `Path(ref)`: the ref is checkout-relative and this function is reached
        # from an installed tool. See `dataset_ref_path` for the anchors and the measurement.
        found = dataset_ref_path(ref)
        data_folder = Path(ref)
        # THE DECLARATION IS THE ALIAS, and a `path` ref that is not a real path has nothing to
        # infer from. `synthetic_timecourse`'s ref is `"synthetic:time-course"` — a generator,
        # not a file — so `read_data` saw nothing and returned `unknown` against a declared
        # `time-course`. Measured cost of that: on `unknown` the recommendation is `embed` and
        # `cflows`, the recipe that dataset exists to give a real time axis to, was not
        # recommended at all. Reported by `tui/state.roster`'s docstring since 2026-07-30 as
        # belonging HERE rather than in the roster, because preferring the YAML in the roster
        # alone would give it a different answer from this function — two accounts of one
        # dataset. This is that fix, in the one place both surfaces read.
        #
        # It also buys the landing screen an OPEN it no longer does: a bundled dataset that
        # declares its shape costs a YAML parse, not a file read.
        #
        # The returned path stays the ref AS DECLARED when nothing was found, so a caller that
        # prints it (`tui.state.DataEntry.path`) reports where the declaration pointed rather
        # than an invented location. Callers that must not offer it as loadable data ask
        # `missing_dataset_file` — an absent file is a fact about this machine, not about the
        # declaration, and the two are separate questions.
        if found is None:
            return data_folder, None, modality, Observation(
                shape=ds.get("shape", "unknown"), modality=modality, conditions=None, source=name,
                declared=tuple(narrate.declared_facts(ds)))
        # MERGE, DO NOT CHOOSE. `read_data` reports facts about the BYTES — `n_obs`, `n_vars`,
        # `conditions`, `n_timepoints`. The YAML reports the experimental DESIGN — `shape`. They
        # answer different questions, so taking one wholesale throws the other away.
        #
        # It used to take `read_data`'s answer entire, which meant THE MORE INFORMATION THE
        # PRODUCT HAD, THE MORE LIKELY IT WAS TO DISCARD THE DECLARATION: the branch above already
        # honours `shape:` when the file is missing, so a declaration won when nothing could be
        # read and lost the moment it could. Measured on `data/pbmc3k_raw.h5ad`, which declares
        # `shape: clusters`: the roster reported `single`, and `cluster` — the recipe that dataset
        # is for — was demoted out of the suited set on the one real dataset the repo ships.
        #
        # Same shape of defect as the gene axis at `3bc84f3`, and the same fix: a DECLARED fact
        # outranks a sniff for that fact alone, and the sniff fills in everything else.
        return found, None, modality, replace(
            narrate.read_data(found, modality, source=name),
            shape=ds.get("shape") or "unknown", declared=tuple(narrate.declared_facts(ds)))
    obs = Observation(shape=ds.get("shape", "unknown"), modality=modality,
                      conditions=None, source=name, declared=tuple(narrate.declared_facts(ds)))
    return None, handle.get("ref", name), modality, obs


def _resolve_path_source(
    p: Path, console: Any
) -> Optional[tuple[Optional[Path], Optional[str], str, Observation]]:
    from manyruns import app, narrate

    if not p.exists():
        console.print(f"[red]not found:[/red] {p}")
        return None
    try:
        modality = app.detect_modality(p)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        return None
    return p, None, modality, narrate.read_data(p, modality, source=p.name)


def _pick_recipe(
    prompter: Any, console: Any, obs: Observation, legal: list[str], *, allow_llm: bool
) -> Optional[str]:
    """Offer the legal recipes (friendly labels from `narrate.offer`), plus a free-text option.

    Free text is routed through `intent`: a request the data can't honestly support is *refused*
    with a reframe (and we re-ask); otherwise it maps to a recipe, falling back to the
    recommended one. Returns the chosen recipe name, or None to abort."""
    from manyruns import intent, narrate

    offered = narrate.offer(obs, available=legal)
    recommended = next((o["recipe"] for o in offered if o.get("recommended")), None) or legal[0]

    # The top menu asks what you want to LEARN. Only recipes SUITED to this data are phrased
    # here; everything else legal sits one level down behind a single agnostic entry.
    #
    # It used to mix the two. A recipe `narrate.offer` had no phrase for was appended as the
    # bare label "Run the cflows recipe", so a scientific question and a raw recipe name sat
    # side by side in one list — and the list grew by one row per recipe added, which is a menu
    # shape that cannot survive a catalog.
    # ONE starred row, not every match. `recommended` is a predicate — it marks each recipe
    # whose `suits:` covers this shape — so it is TRUE OF MANY once a catalog exists: measured
    # at eight recipes, a `clusters` observation recommends five (archetypes, cluster, embed,
    # markers, qc). Showing all of them puts the catalog back in the top menu, which is the
    # exact shape this branch was written to remove.
    #
    # The head is the recommendation. `offer` sorts `(not recommended, name)` — recommended
    # first, then ALPHABETICAL — so among equally-suited recipes the letter decides, which is
    # stable but is not a ranking. Keeping `suits:` narrow is what makes the head meaningful:
    # broad lists put five recipes in contention for a plain folder and `archetypes` won on
    # the letter A, silently replacing `embed` as the default. If a real ranking is ever
    # wanted, it belongs in the recipe (a declared specificity), not in this line.
    suited = ([o for o in offered if o.get("recommended")] or offered)[:1]
    choices: list[tuple[str, Any]] = [(o["label"] + "  ★", o["recipe"]) for o in suited]
    rest = [o for o in offered if o["recipe"] not in {o2["recipe"] for o2 in suited}]
    if rest:
        choices.append((f"Run a specific recipe…  ({len(rest)} more)", "__recipes__"))
    choices.append(("Type what you want in plain English…", "__free__"))
    choices.append(("← back", "__back__"))

    while True:
        pick = prompter.select("What should I look for?", choices, default=recommended)
        if pick in (None, "__back__"):
            return None
        if pick == "__recipes__":
            # The dropdown is the whole legal set, suited ones included — someone who opened it
            # to compare should not have to back out to find the recommended one again.
            sub = [(f"{o['recipe']} — {o['label']}" + ("  ★" if o["recommended"] else ""),
                    o["recipe"]) for o in offered]
            sub.append(("← back", "__back__"))
            chosen = prompter.select("Which recipe?", sub, default=recommended)
            if chosen in (None, "__back__"):
                continue
            return chosen
        if pick != "__free__":
            return pick
        text = prompter.text("In your own words:").strip()
        if not text:
            continue
        recipe, refusal = intent.route_recipe(text, obs, legal, allow_llm=allow_llm)
        if refusal:
            console.print(f"\n  [yellow]{refusal}[/yellow]\n")
            continue
        chosen = recipe or recommended
        if not recipe:
            console.print(f"  [dim](couldn't pin that down — going with {chosen})[/dim]")
        return chosen


def _run(
    args: Any, console: Any, data_folder: Optional[Path], dataset: Optional[str],
    modality: str, recipe_name: str, obs: Observation, engine: Optional[str] = None,
) -> Optional[dict]:
    """Execute one recipe. A steppable engine shows a live per-step trace; a non-steppable one runs as
    one blocking call. Returns the results dict, or None on setup failure."""
    from manyruns import app
    from manyruns.pipeline import runner as _runner

    engine = engine or getattr(args, "engine", None) or app._default_engine()
    recipe, _ = app.select_analysis(modality, data_folder, override=recipe_name)
    console.print(f"[dim]running {recipe_name} on {obs.source} · engine={engine}[/dim]")

    # `engine in ("mock", "manylatents")` — the same stale predicate `app.cmd_init` carried.
    # `real` is the default on a public install, so the front door's stepped path was
    # unreachable on the shipped configuration. One question now, asked of the dispatch table.
    if _runner.steppable(engine):
        dataset_name = None
        if dataset is not None and data_folder is None:
            from manyruns.catalog import load_dataset

            try:
                handle = load_dataset(obs.source).get("handle") or {}
            except (ValueError, OSError):
                handle = {}
            if handle.get("kind") == "manylatents" and handle.get("ref") == dataset:
                dataset_name = obs.source
        proj_like = {
            "project": _slugish(obs.source), "engine": engine, "dataset": dataset,
            "dataset_name": dataset_name,
            "data_folder": str(data_folder) if data_folder else None,
            "modality": modality, "recipe": recipe,
        }
        session = app._build_session(proj_like, args)
        session.source = obs.source
        return _run_stepped(console, session, recipe)

    # A live dashboard for the blocking engines. `run_inproc` executes the whole recipe in
    # one call, so a scientist watching a two-minute embedding previously saw a frozen
    # terminal and then a wall of output. The panel now opens with every declared step
    # queued and fills in as each lands.
    live, on_step = (None, None)
    if getattr(console, "is_terminal", False):
        live, on_step = live_dashboard(
            console, recipe, f"[bold]{recipe_name}[/bold] · {engine} · {obs.source}")
    try:
        if live is not None:
            live.start()
        results = app.explore_once(
            data_folder=data_folder, dataset=dataset, modality=modality, recipe_name=recipe_name,
            engine=engine, seed=getattr(args, "seed", 42), fast_dev_run=app._smoke(args),
            time_key=getattr(args, "time_key", None),
            # `None` default — the shim overrides a recipe's DECLARED method, so defaulting to
            # log1p here would silently rewrite a recipe that asked for `sqrt`.
            transform=getattr(args, "transform", None),
            device=getattr(args, "device", None), out_dir=Path("outputs") / _slugish(obs.source),
            on_step=on_step,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return None
    finally:
        if live is not None:
            # the progress view is scaffolding for the wait; the finished panels replace it
            live.stop()
    # the caller renders the step record; this used to re-derive per-step status by testing
    # `"ok" in str(st)`, which reads "skipped (…)" as a success on the substring alone.
    return results


def _run_stepped(console: Any, session: Any, recipe: dict) -> dict:
    """Drive the recipe one step at a time, updating a live trace as each completes.

    Each step is applied as its DICT, not its name. Passing `s["name"]` re-resolved the step
    through the catalog and threw away whatever the loaded recipe declared, which is how a
    `phate` step carrying `{n_components: 3}` reached the fit at the engine default of 2.
    Ends with `close()`, so the lineage's `COMPLETE` marker is written exactly once."""
    steps = recipe.get("steps", [])
    if getattr(console, "is_terminal", False):
        from rich.live import Live

        status: dict[str, Any] = {}
        with Live(_trace_table(steps, status), console=console, refresh_per_second=8) as live:
            for s in steps:
                status[s["name"]] = session.step(dict(s))
                live.update(_trace_table(steps, status))
    else:  # non-terminal (tests, pipes): print each step as it lands
        for s in steps:
            console.print(_fmt_step(session.step(dict(s))))
    return session.close()


def _trace_table(steps: list[dict], status: dict[str, Any]) -> Any:
    from rich.table import Table

    t = Table(show_header=False, box=None, pad_edge=False)
    for s in steps:
        name = s["name"]
        r = status.get(name)
        if r is None:
            mark, note = "[dim]…[/dim]", "[dim]queued[/dim]"
        elif r.get("ok"):
            mark, note = "[green]✓[/green]", r.get("detail", "ok")
        else:
            mark, note = "[red]✗[/red]", f"[red]{r.get('error', 'failed')}[/red]"
        t.add_row(mark, f"{s.get('group', '?')}:{name}", str(note))
    return t


def _fmt_step(r: dict) -> str:
    if r.get("ok"):
        return f"[green]✓[/green] {r.get('name')}: {r.get('detail', 'ok')}"
    return f"[red]✗[/red] {r.get('name', '?')}: {r.get('error', 'failed')}"


# ── the other verbs ────────────────────────────────────────────────────────────────────────────
def _check(console: Any) -> None:
    from types import SimpleNamespace

    from manyruns import app

    with _maybe_status(console, "validating the catalog…"):
        app.cmd_check(SimpleNamespace(dataset_dir=None), None)


def _list_recipes(console: Any) -> None:
    from manyruns.catalog import discover_recipes, load_recipe

    console.print("[bold]recipes[/bold]")
    for n in discover_recipes():
        try:
            steps = " → ".join(f"{s.get('group')}:{s.get('name')}" for s in load_recipe(n)["steps"])
        except Exception as e:  # noqa: BLE001
            steps = f"[red]{type(e).__name__}[/red]"
        console.print(f"  {n:<14} {steps}")


def _list_datasets(console: Any) -> None:
    from manyruns.catalog import discover_datasets, load_dataset

    console.print("[bold]datasets[/bold]")
    for n in discover_datasets():
        try:
            ds = load_dataset(n)
            h = ds.get("handle", {})
            line = f"  {n:<14} {h.get('kind')}:{h.get('ref')}  shape={ds.get('shape')}"
            # The ref is what the declaration SAYS; where the bytes are is a fact about this
            # machine. Both are printed, because a listing that shows only the first is the
            # listing that disagreed with the run — and one that showed only the second would
            # hide which declaration you are looking at.
            if missing_dataset_file(ds):
                line += "  [yellow](not on this machine)[/yellow]"
            elif h.get("kind") == "path":
                found = dataset_ref_path(str(h.get("ref") or ""))
                if found is not None and str(found) != str(h.get("ref")):
                    line += f"  [dim]→ {found}[/dim]"
            console.print(line)
        except Exception as e:  # noqa: BLE001
            console.print(f"  {n:<14} [red]{type(e).__name__}: {e}[/red]")


# ── helpers ──────────────────────────────────────────────────────────────────────────────────
def _loaded_stats(obs: Observation) -> str:
    """The terse validation line: the concrete facts read off the data (numbers when known)."""
    bits: list[str] = []
    if obs.n_obs:
        bits.append(f"{obs.n_obs:,} cells" + (f" × {obs.n_vars:,} genes" if obs.n_vars else ""))
    if obs.conditions:
        bits.append(f"{len(obs.conditions)} groups: {', '.join(map(str, obs.conditions))}")
    if obs.n_timepoints:
        bits.append(f"{obs.n_timepoints} time points")
    if obs.modality and obs.modality != "unknown":
        bits.append(f"modality {obs.modality}")
    bits.append(f"shape {obs.shape}")
    return " · ".join(bits)


def _slugish(source: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in str(source).lower()).strip("-") or "session"


def _maybe_status(console: Any, message: str):
    if getattr(console, "is_terminal", False) and hasattr(console, "status"):
        return console.status(message)
    return contextlib.nullcontext()
