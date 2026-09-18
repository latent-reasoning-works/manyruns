"""`find data…` — the roster row §3.1 draws at the bottom, answered. Component 5 of the rewrite.

TWO SCREENS AND ONE CONTRACT. `FindScreen` is `Screen[Optional[DataEntry]]`: it gives the app
back **the same thing the roster gives it** — a `state.DataEntry`, whose `as_source()` is
already `shell._run`'s `(data_folder, dataset, modality, obs)` tuple — or `None` for "went
back". Nothing downstream learns that this screen exists. `FetchScreen` is its internal step:
it confirms a download, runs it, and returns the `.h5ad` path.

WIRING IT IS TWO LINES, and they are deliberately not written here, exactly as components 2 and
3 left theirs. `manyruns/tui/app.py` is component 2's file and `open_finder` is the seam it
published::

    def open_finder(self) -> None:
        self.push_screen(FindScreen(), callback=lambda entry: entry and self.open_ledger(entry))

An app that imports four screens mid-edit is how four green components make one red tree; the
seam is the contract, and `tests/test_tui_find.py` drives it through a `ManyrunsApp` subclass
the same way `tests/test_tui_roster.py` does.

THREE TIERS, CHEAPEST FIRST, AND THE ORDERING IS THE FEATURE. The search is
`manyruns.tui.search` — dependency-free, no Textual, and measured: `pool()` is 44 ms (a TSV
parse plus the bundled dataset YAMLs, against an empty drop folder) and `rank()` is 0.42 ms, so
the pool is built once at mount and only the ranking runs per keystroke. Tier 3 is one optional
Anthropic call behind `intent.llm_available()`; with no key the row that offers it never
appears and tiers 1 and 2 are unaffected.

The confirm states both download and on-disk sizes: archives can expand substantially
when extracted and converted. Where expansion is not recorded, the screen says so;
it never scales one dataset's ratio onto another.

NOTHING HERE DOWNLOADS ANYTHING ITSELF. `harness/data_acquire.py` already fetches, extracts and
converts, and it works. The one thing it lacked was a way to see the bytes go by, so `fetch()`
gained a `reporthook=` keyword — `urllib.request.urlretrieve`'s own callback, forwarded — and
that is the entire change to it. It is also the cancel point: raising from the hook is the only
cooperative way out of a blocking `urlretrieve`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Button, Input, OptionList, ProgressBar, Static
from textual.widgets.option_list import Option
from textual.worker import get_current_worker

from manyruns.tui import search
from manyruns.tui.search import Candidate, FetchPlan
from manyruns.tui.state import DataEntry

#: The two marks §3.1's roster already taught: `▸`-style pointers are the cursor's job, so these
#: say what a row IS. `⤓` costs a download; `✦` is made here and costs nothing. A row that
#: cannot be fetched at all reuses the ledger's hollow `○` — same meaning on both screens ("on
#: screen to be read, not chosen"), which is why it is the same character.
FETCH_MARK = "⤓"
LOCAL_MARK = "✦"
BLOCKED_MARK = "○"

#: The tier-3 row's option id. A NAME rather than a position, because every other option's id is
#: its index into `self.results` and this row has no candidate behind it — an index would make
#: `int(event.option.id)` either an out-of-range crash or, worse, the wrong dataset.
ASK_ID = "ask-a-model"

#: The hint line, in one place because two handlers write it. `tab` is Textual's own focus
#: cycling and is advertised rather than rebound: `enter` in the query box moves to the list on
#: purpose (see `on_input_submitted`), so the way BACK to the query has to be discoverable.
HINT = "↑↓ move · enter chooses · tab to edit the query · esc back"


def mark(candidate: Candidate) -> str:
    return (BLOCKED_MARK if not candidate.available
            else FETCH_MARK if candidate.needs_fetch else LOCAL_MARK)


def detail(candidate: Candidate) -> str:
    """The second line of a row: what this is, and what standing it has.

    One place, like `LedgerRow.detail`, so the list and any future summary cannot drift about
    which sentence a row carries.
    """
    if not candidate.available:
        return candidate.cannot_fetch
    if candidate.by_model:
        # Provenance travels with the row: this one is here because a model read the manifest,
        # not because the words matched. Without the label it is indistinguishable from a
        # substring hit, and the two are worth different amounts of trust.
        return f"a model picked this from the manifest · {candidate.note}"
    return candidate.note


def _row(candidate: Candidate, name_width: int) -> Table:
    """One result: mark · name · summary, with the note folded underneath.

    A `Table.grid` per option and a shared `name_width`, both for the reasons
    `tui/ledger.py::_prompt` gives: a column that each row measures for itself is not a column,
    and prose in a cell needs `overflow="fold"` because rich 15 DELETES characters when it folds
    a word longer than its column (component 4 measured that).
    """
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)
    grid.add_column(width=name_width, no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    # Render user-supplied notes as Text so brackets cannot be interpreted as markup.
    grid.add_row(Text(mark(candidate)), Text(candidate.name), Text(candidate.summary))
    grid.add_row(Text(""), Text(""), Text(detail(candidate), style="dim"))
    return grid


class FindScreen(Screen[Optional[DataEntry]]):
    """Describe what you need. Returns a `DataEntry` — fetched, or generated here — or `None`."""

    BINDINGS = [Binding("escape", "back", "back", show=True)]

    DEFAULT_CSS = """
    FindScreen { layout: vertical; }
    FindScreen > #find-frame {
        height: 1fr; margin: 1 2; padding: 0 1;
        border: round $primary; border-title-align: left; border-subtitle-align: right;
    }
    FindScreen #find-results { height: 1fr; background: transparent; }
    FindScreen #find-query { border: none; background: transparent; padding: 0 1; }
    FindScreen #find-hint { dock: bottom; height: auto; margin: 0 3 1 3; color: $text-muted; }
    """

    def __init__(self, *, candidates: "Optional[list[Candidate]]" = None,
                 out_dir: "Optional[Path]" = None,
                 name: Optional[str] = None, id: Optional[str] = None,  # noqa: A002
                 classes: Optional[str] = None) -> None:
        """`candidates` is injectable so a test drives the screen without reading the manifest or
        the dataset YAMLs; `None` means "build the real pool after the first paint".

        `out_dir` overrides where a fetch lands. It defaults to `shell.data_dir()` — the drop
        folder — inside `search.plan`, which is the integration: what you fetch is a roster row
        the next time the app opens.
        """
        super().__init__(name=name, id=id, classes=classes)
        self._preloaded = candidates
        self.out_dir = out_dir
        self.pool: list[Candidate] = []
        self.results: list[Candidate] = []
        #: What tier 3 returned, kept apart from the pool so a second query does not inherit a
        #: model's answer to the first one.
        self.suggested: Optional[Candidate] = None
        self.asking = False

    # ── layout ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with Vertical(id="find-frame"):
            yield Input(placeholder="describe what you need", id="find-query")
            yield OptionList(id="find-results", compact=True)
        yield Static(HINT, id="find-hint")

    def on_mount(self) -> None:
        frame = self.query_one("#find-frame", Vertical)
        frame.border_title = "find data"
        frame.border_subtitle = "a public cohort, or one made here"
        self.query_one("#find-query", Input).focus()

        if self._preloaded is not None:
            self.pool = list(self._preloaded)
            self.refresh_results()
            return
        # FIRST PAINT BEFORE THE READ, for component 2's measured reason: `search.pool()` calls
        # `state.roster()`, which opens a file per dropped row (361 ms cold on a 5.9 MB h5ad).
        # 44 ms of that is the manifest and the dataset YAMLs; the rest is whatever is in the
        # drop folder, and neither should happen between a keypress and the first frame.
        self.query_one(OptionList).add_option(Option(Text("reading the catalogue…"),
                                                     disabled=True))
        self.call_after_refresh(self.load)

    def load(self) -> None:
        try:
            self.pool = search.pool()
        except Exception as e:  # noqa: BLE001 - a finder with no rows must still say why
            self.pool = []
            self.query_one("#find-hint", Static).update(f"[red]{type(e).__name__}: {e}[/red]")
        self.refresh_results()

    # ── the rows ─────────────────────────────────────────────────────────────
    def query_text(self) -> str:
        return self.query_one("#find-query", Input).value

    def refresh_results(self) -> None:
        """Re-rank and redraw. Pure string work — `rank` touches no disk (measured: 0.42 ms)."""
        text = self.query_text()
        found = search.rank(text, self.pool)
        if self.suggested is not None:
            # The model's pick goes to the TOP and its duplicate leaves the list, so the screen
            # never shows one cohort twice with two different reasons for being there.
            found = [self.suggested] + [c for c in found if c.name != self.suggested.name]
        self.results = found

        options: list[Option] = []
        name_width = max([len(c.name) for c in found] or [0])
        for i, candidate in enumerate(found):
            options.append(Option(_row(candidate, name_width), id=str(i),
                                  disabled=not candidate.available))
        if self.can_ask():
            options.append(Option(
                Text(f"  ask a model to read the manifest for {text!r}", style="italic"),
                id=ASK_ID))

        results = self.query_one(OptionList)
        results.clear_options()
        results.add_options(options)
        if options:
            results.action_first()

    def can_ask(self) -> bool:
        """Whether to offer tier 3 at all.

        THREE CONDITIONS, and each is the reason it is tier 3 rather than tier 1: there must be a
        query (a model cannot read an empty mind), the local search must have matched NOTHING (an
        LLM call to reorder rows that already matched is a network round-trip bought for
        nothing), and the SDK and key must both be present — `intent.llm_available()` checks that
        without importing the SDK.
        """
        from manyruns import intent

        if self.asking or self.suggested is not None or not self.query_text().strip():
            return False
        return not any(c.score for c in self.results) and intent.llm_available()

    # ── events ───────────────────────────────────────────────────────────────
    def on_input_changed(self, event: Input.Changed) -> None:
        # A new query retires the old model answer; it was an answer to a different question.
        self.suggested = None
        self.refresh_results()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the query box moves to the list rather than choosing.

        Deliberate, and the reason is the screen after this one: enter here choosing the top row
        would put a confirm dialog under a finger that is already pressing enter, and the button
        behind it starts a large download. One extra keystroke is the cost of that not
        happening by momentum.
        """
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id == ASK_ID:
            self.ask()
            return
        self._chosen(self.results[int(event.option.id)])

    def _chosen(self, candidate: Candidate) -> None:
        """Tier 2 finishes here; tier 1 goes through the confirm.

        A synthetic dismisses IMMEDIATELY with the roster's own `DataEntry` — no network, no
        confirm, nothing to warn about. That asymmetry is the point of listing the two kinds
        together: the cheap answer is one keystroke and the expensive one has a screen in front
        of it.
        """
        if not candidate.available:
            return
        if not candidate.needs_fetch:
            self.dismiss(candidate.entry)
            return
        plan = search.plan(candidate, out_dir=self.out_dir)
        self.app.push_screen(FetchScreen(plan), callback=self._fetched)

    def _fetched(self, path: "Optional[Path]") -> None:
        """The fetch screen came back. A path means it landed and verified."""
        if path is None:
            return                       # cancelled or failed — stay here, the list is intact
        entry = search.roster_entry(path, self.out_dir)
        if entry is None:
            # Verified-and-not-a-roster-row: the file exists but `shell._resolve_path_source`
            # will not resolve it, which is the silent skip component 1 reports as a known gap.
            # Said out loud here because this is the one moment someone can act on it.
            self.query_one("#find-hint", Static).update(
                f"[red]{path.name} downloaded, but it does not resolve as data — "
                f"it is at {path}[/red]")
            return
        self.dismiss(entry)

    # ── tier 3 ───────────────────────────────────────────────────────────────
    def ask(self) -> None:
        self.asking = True
        self.query_one("#find-hint", Static).update("asking a model…")
        self._ask_worker(self.query_text(), list(self.pool))

    @work(thread=True, exclusive=True, group="manyruns-find-llm")
    def _ask_worker(self, text: str, pool: list[Candidate]) -> None:
        """One network call, off the event loop, reported back as a message.

        A thread because `intent._llm_choose` is the synchronous SDK and a screen that freezes
        while it waits is worse than no tier 3 at all. `search.ask_a_model` swallows everything
        (no SDK, no key, no network) and answers None, which arrives here as "it had nothing".
        """
        self.post_message(self.Suggested(search.ask_a_model(text, pool)))

    class Suggested(Message):
        def __init__(self, candidate: Optional[Candidate]) -> None:
            super().__init__()
            self.candidate = candidate

    def on_find_screen_suggested(self, message: "FindScreen.Suggested") -> None:
        self.asking = False
        self.suggested = message.candidate
        self.query_one("#find-hint", Static).update(
            HINT if message.candidate is not None
            else "the model had nothing to add — the whole manifest is above")
        self.refresh_results()

    def action_back(self) -> None:
        self.dismiss(None)


# ══ the fetch ════════════════════════════════════════════════════════════════
class FetchScreen(Screen[Optional[Path]]):
    """Confirm a download, then run it. Returns the `.h5ad` path, or `None`.

    ONE SCREEN IN TWO STATES rather than two screens, because the second state has to answer a
    question the first one asked ("how big?") and a pushed second screen would either ask the
    server twice or carry the answer across as a duplicate of the plan.

    THE CONFIRM'S DEFAULT IS CANCEL, and that is not politeness. This screen is reached by
    pressing enter on a list, and a focused "fetch it" would put the start of a large download
    under the next enter. `test_enter_on_arrival_does_not_start_a_download` pins it.
    """

    BINDINGS = [Binding("escape", "back", "back", show=True)]

    DEFAULT_CSS = """
    FetchScreen { layout: vertical; align: center middle; }
    FetchScreen > #fetch-frame {
        width: 100%; height: auto; max-height: 100%; margin: 1 2; padding: 1 2;
        border: round $warning; border-title-align: left;
    }
    FetchScreen #fetch-sizes { height: auto; margin: 1 0; }
    FetchScreen #fetch-phases { height: auto; margin: 1 0; }
    FetchScreen #fetch-bar { height: auto; margin: 0 0 1 0; }
    FetchScreen #fetch-buttons { height: auto; align-horizontal: left; }
    FetchScreen Button { margin: 0 2 0 0; }
    FetchScreen #fetch-note { height: auto; color: $text-muted; }
    """

    #: One update per whole percent bounds UI work for large downloads.
    _REPORT_EVERY_PERCENT = 1

    class Progress(Message):
        def __init__(self, phase: str, read: int, total: int) -> None:
            super().__init__()
            self.phase, self.read, self.total = phase, read, total

    class Ended(Message):
        """Finished, cancelled or failed — one message, so a screen can never be left in a state
        where it is waiting for a worker that has already stopped."""

        def __init__(self, path: Optional[Path], error: Optional[str] = None,
                     cancelled: bool = False) -> None:
            super().__init__()
            self.path, self.error, self.cancelled = path, error, cancelled

    def __init__(self, plan: FetchPlan, *, probe: bool = True,
                 name: Optional[str] = None, id: Optional[str] = None,  # noqa: A002
                 classes: Optional[str] = None) -> None:
        """`probe=False` turns off the network size check — for tests, and for anyone who
        already holds a probed plan."""
        super().__init__(name=name, id=id, classes=classes)
        self.plan = plan
        self._probe = probe
        self.state = "confirm"          # confirm | running | done | failed | cancelled
        self.phase = search.PHASES[0]
        self.read = 0
        self.total = 0
        self.message = ""
        self.path: Optional[Path] = None

    # ── layout ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        with Vertical(id="fetch-frame"):
            yield Static(id="fetch-what")
            yield Static(id="fetch-sizes")
            yield ProgressBar(id="fetch-bar", show_eta=True)
            yield Static(id="fetch-phases")
            yield Static(id="fetch-note")
            with Horizontal(id="fetch-buttons"):
                # Cancel FIRST so it takes focus first — see the class docstring.
                yield Button("cancel", id="fetch-cancel")
                yield Button("fetch it", id="fetch-go", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#fetch-frame", Vertical).border_title = self.plan.accession
        self.query_one("#fetch-bar", ProgressBar).display = False
        self.query_one("#fetch-cancel", Button).focus()
        self.sync()
        if self._probe and self.plan.url:
            # Keep network latency off the event loop. The
            # screen is already up and reads "checking…" until it answers.
            self._probe_worker()

    @work(thread=True, exclusive=True, group="manyruns-fetch-probe")
    def _probe_worker(self) -> None:
        from dataclasses import replace

        self.post_message(self.Probed(replace(self.plan, probe=search.probe(self.plan.url))))

    class Probed(Message):
        def __init__(self, plan: FetchPlan) -> None:
            super().__init__()
            self.plan = plan

    def on_fetch_screen_probed(self, message: "FetchScreen.Probed") -> None:
        self.plan = message.plan
        self.sync()

    # ── the paint ────────────────────────────────────────────────────────────
    def sync(self) -> None:
        if not self.is_running:      # component 4's measured distinction: `is_mounted` is still
            return                   # False inside a pushed screen's own `on_mount`
        from rich.markup import escape

        plan = self.plan
        self.query_one("#fetch-what", Static).update(
            f"[bold]{escape(plan.accession)}[/bold]\n[dim]{escape(plan.url)}[/dim]")
        self.query_one("#fetch-sizes", Static).update(self._sizes())
        self.query_one("#fetch-phases", Static).update(self._phases())
        self.query_one("#fetch-note", Static).update(self._note())

        bar = self.query_one("#fetch-bar", ProgressBar)
        bar.display = self.state == "running" and self.phase == "download"
        if bar.display:
            # `total=None` when the server reported no length: Textual draws an indeterminate
            # bar, which is the honest picture of bytes arriving at an unknown fraction.
            bar.update(total=float(self.total) if self.total > 0 else None,
                       progress=float(self.read))

        confirming = self.state == "confirm"
        self.query_one("#fetch-go", Button).display = confirming
        self.query_one("#fetch-cancel", Button).label = (
            "cancel" if self.state in ("confirm", "running") else "back")

    def _sizes(self) -> str:
        """BOTH SIZES, ALWAYS, and never an estimate for either.

        `probing…` is a third state and it earns its place: with no `content-length` yet, "size
        unknown" would be a claim about the server rather than about this screen's own progress,
        and it would flip to a number 8 seconds later, which reads as the app changing its mind.
        """
        probe = self.plan.probe
        download = ("[dim]checking…[/dim]" if self._probing() else
                    f"[red]{probe.error}[/red]" if probe.missing else self.plan.download)
        return (f"[dim]download[/dim]   {download}\n"
                f"[dim]on disk [/dim]   {self.plan.disk}   [dim]after extract + convert[/dim]")

    def _probing(self) -> bool:
        return (self._probe and self.state == "confirm"
                and self.plan.probe.status is None and self.plan.probe.error is None)

    def _phases(self) -> str:
        """The four phases, always all four, with the live one marked.

        Extraction and conversion continue after the download reaches 100%, so
        show every phase from the first frame and highlight the active one.
        """
        if self.state == "confirm":
            # One line before anything starts: the confirm's job is to say what a "yes" buys,
            # and four labels in a row is that sentence without spending four lines on it.
            return "[dim]then:  " + "  →  ".join(search.PHASES) + "[/dim]"
        here = search.PHASES.index(self.phase)
        lines = []
        for i, phase in enumerate(search.PHASES):
            if self.state == "done" or i < here:
                lines.append(f"[green]{phase}[/green]")
            elif i == here and self.state == "running":
                lines.append(f"[bold]{phase}[/bold]  [dim]{search.PHASE_GLOSS[phase]}[/dim]")
            else:
                lines.append(f"[dim]{phase}[/dim]")
        return "\n".join(lines)

    def _note(self) -> str:
        from rich.markup import escape

        if self.state == "cancelled":
            # NEVER PROMISE A ROLLBACK NOTHING PERFORMS. `urlretrieve` streams into the
            # destination opened `'wb'`, so the partial tar is still there; a re-run truncates
            # it and starts from zero rather than resuming. Read from CPython's urllib, and the
            # reason this sentence names the path instead of saying "cleaned up".
            return (f"[yellow]cancelled.[/yellow] the part that downloaded is still at "
                    f"{escape(str(self.plan.partial))} — a re-run starts it over, it does not "
                    f"resume.")
        if self.state == "failed":
            return f"[red]{escape(self.message)}[/red]"
        if self.state == "done":
            return f"[green]ready:[/green] {escape(str(self.path))}"
        if self.state == "running":
            # `message` during a run is never an error — the run is still going. It is the one
            # thing a cancel that cannot be honoured has to say, and hiding it behind the work
            # dir would leave a pressed cancel button with no reply at all.
            return (f"[yellow]{escape(self.message)}[/yellow]" if self.message
                    else f"[dim]working in {escape(str(self.plan.work_dir))}[/dim]")
        if self.plan.already_here:
            return (f"[green]you already have this[/green] — {escape(str(self.plan.out_path))}. "
                    f"fetching again will not re-download it.")
        return ("[dim]archives can expand during extraction and conversion; "
                "on-disk size is shown only when recorded.[/dim]")

    # ── events ───────────────────────────────────────────────────────────────
    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "fetch-go":
            self.start()
        else:
            self.action_back()

    def action_back(self) -> None:
        """Escape and `cancel`. During a run it cancels the worker; otherwise it leaves.

        The worker stops at the next `reporthook` call, which is every 8 KB while bytes are
        moving — and NOT AT ALL during extract or convert, because neither `tarfile.extractall`
        nor scanpy's readers offer a callback to check from. Said plainly rather than shown as a
        cancel that appears to work and does not.
        """
        if self.state == "running":
            self.workers.cancel_group(self, "manyruns-fetch")
            if self.phase != "download":
                self.message = (f"{self.phase} cannot be interrupted — it has no checkpoint to "
                                f"stop at. it will finish.")
                self.sync()
            return
        self.dismiss(self.path if self.state == "done" else None)

    def start(self) -> None:
        if self.plan.already_here:
            # Nothing to download and nothing to convert: `data_acquire.convert` returns the
            # existing path untouched unless `force=True`, so the honest move is to hand it
            # back now rather than run three phases that each decide to do nothing.
            self.path, self.state = self.plan.out_path, "done"
            self.sync()
            self.dismiss(self.plan.out_path)
            return
        self.state, self.phase, self.read, self.total = "running", search.PHASES[0], 0, 0
        self.sync()
        self._fetch_worker()

    @work(thread=True, exclusive=True, group="manyruns-fetch")
    def _fetch_worker(self) -> None:
        """Download → extract → convert → verify, off the event loop.

        `data_acquire.fetch` + `.convert` rather than `.acquire`, because `acquire` is the two of
        them behind one call and this screen has to name the boundary between them. Everything
        else is theirs: the tar handling, the matrix location, the scanpy read, the concat.
        """
        from manyruns.harness import data_acquire as da

        worker = get_current_worker()
        plan = self.plan
        ticks = search.DownloadTicks(self._REPORT_EVERY_PERCENT)

        def hook(blocknum: int, blocksize: int, total: int) -> None:
            if worker.is_cancelled:
                # The ONLY cooperative exit from a blocking `urlretrieve`. It propagates out,
                # the destination file closes with whatever arrived, and `partial` says where.
                raise _Cancelled()
            tick = ticks(blocknum, blocksize, total)
            if tick is not None:
                self.post_message(self.Progress(*tick))

        try:
            plan.work_dir.mkdir(parents=True, exist_ok=True)
            root = da.fetch(plan.accession, plan.work_dir, url=plan.url or None, reporthook=hook)
            self.post_message(self.Progress("convert", 0, 0))
            out = da.convert(root, plan.accession, out_dir=plan.out_path.parent, verify=False)
            self.post_message(self.Progress("verify", 0, 0))
            # `verify=False` above and the roster's own read here: `assert_loadable` asks whether
            # manylatents can build an AnnDataModule, which is the harness's question. The one
            # this screen owes an answer to is "is it on my roster", and `search.roster_entry`
            # asks exactly that — see its docstring.
            entry = search.roster_entry(Path(out))
        except _Cancelled:
            self.post_message(self.Ended(None, cancelled=True))
        except Exception as e:  # noqa: BLE001 - shown on screen, never swallowed
            self.post_message(self.Ended(None, f"{type(e).__name__}: {e}"))
        else:
            self.post_message(self.Ended(
                Path(out) if entry is not None else None,
                None if entry is not None else
                f"downloaded to {out}, but it does not resolve as data"))

    def on_fetch_screen_progress(self, message: "FetchScreen.Progress") -> None:
        self.phase, self.read, self.total = message.phase, message.read, message.total
        self.sync()

    def on_fetch_screen_ended(self, message: "FetchScreen.Ended") -> None:
        self.path = message.path
        self.state = ("cancelled" if message.cancelled
                      else "failed" if message.error else "done")
        self.message = message.error or ""
        self.sync()
        if self.state == "done":
            self.dismiss(message.path)

    def on_key(self, event: Any) -> None:
        """`y` starts the fetch, and `enter` deliberately does not.

        A shortcut for the deliberate answer only. The undeliberate one — the enter still under
        the finger from the list — lands on the focused `cancel` button.
        """
        if event.key == "y" and self.state == "confirm":
            event.stop()
            self.start()


class _Cancelled(Exception):
    """Raised inside the download's `reporthook` to unwind a blocking `urlretrieve`. Private:
    it is a control signal between two functions in this file, not an error anyone catches."""


__all__ = ["FindScreen", "FetchScreen", "mark", "detail"]
