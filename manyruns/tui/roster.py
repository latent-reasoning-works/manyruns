"""The landing screen. `manyruns` on a TTY opens HERE.

No home menu, and no "What do you want to do?". §0 counts five prompts between launch and a
running analysis; the first of them asks a question whose answer is always "look at my data",
so the roster *is* the answer and it is the first thing drawn.

WHAT THIS SCREEN COMPUTES: no fact. Every row is a `tui.state.DataEntry` and every phrase in one
comes from `narrate` (`DataEntry.contents` is `narrate.contents`, `DataEntry.size` is the
Observation's own counts). The header footnote is `shell.state_rows()` verbatim — the existing
single source whose docstring already says "pure data, so the plain and panel renderings share
it", which makes this a third reader of it rather than a fourth account of what is installed.
The rule the rewrite runs under is honoured by *reading*, not by re-deriving.

WIDTHS ARE MEASURED, NEVER DECLARED. `shell.render_result` hardcodes `width=30` for its steps
column, which is why a real render splits `ndarray(18…` / `10) float32` across lines (§0). Here
`columns_that_fit` measures the strings that will actually be in the table against the width the
table actually has, and drops the rightmost column when they do not fit. Everything else is in
`manyruns.tcss`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

from rich.cells import cell_len
from rich.markup import escape
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.coordinate import Coordinate
from textual.message import Message
from textual.screen import Screen
from textual.widgets import DataTable, Input, Static

from manyruns import narrate
from manyruns.tui import dropwatch, samplefetch, state
from manyruns.tui.state import DataEntry

#: The columns, in the order §3.1's mockup draws them, and in DROP order right-to-left: when
#: the terminal is too narrow, "what's in it" goes before "size" and neither `mark` nor `data`
#: ever goes — a row you cannot name is not a row. `mark` carries the cursor (`▸`) and the
#: find-data row's `?`, which is why it is not folded into `data`: the cursor must be legible
#: even when the table does not hold focus (focus lives in the filter box, so `DataTable`'s own
#: cursor styling is the blurred one).
COLUMNS: tuple[tuple[str, str], ...] = (
    ("mark", ""),
    ("data", "data"),
    ("size", "size"),
    ("geometry", ""),
    ("contents", "what's in it"),
    # QC SITS TO THE RIGHT OF `contents`, ON PURPOSE. `columns_that_fit` drops the RIGHTMOST
    # column until the table fits, so at a narrow width these three go first and the roster
    # degrades to exactly the four columns it had before they existed — the new facts cost
    # nothing at the sizes where there was no room for them anyway.
    ("per_cell", "genes/cell"),
    ("mito", "% mito"),
    ("drops", "filter drops"),
)

#: Columns that are never dropped, counted from the left.
_KEEP = 2

#: Cells that `DataTable` adds around each column's content, measured on textual 8.2.8:
#: a table of `mark/data/size/what's in it` over the two rows in the probe reported
#: `virtual_size.width == 81` against a summed content width of 73 — 2 per column, which is
#: `cell_padding=1` on each side. If a future version changes it the cost is one column too many
#: or one too few, i.e. a horizontal scrollbar, not a crash.
_CELL_PADDING = 2

#: The find-data row. Not a `DataEntry`: it names an action, not something you could run.
#: Built over every column rather than written out, so adding a column cannot leave this row
#: short one key — which is exactly how it broke the first time the QC columns landed.
FIND_ROW = {k: "" for k, _ in COLUMNS} | {
    "mark": "?", "data": "find data…", "contents": "describe what you need"}

#: What a group-heading row carries where an entry would be.
#:
#: A DISTINCT SENTINEL AND NOT `None`, because `None` already means something here: `choose`
#: reads it as "the find-data row" and posts `FindDataRequested`. A heading sharing that value
#: would open the finder when someone pressed enter on the word "synthetic".
HEADING = object()

#: The three groups, in the order they are drawn, as (mark, title, predicate).
#:
#: FIND STAYS LAST, which is where it already was. Putting it between the two data groups reads
#: better on paper and is what breaks eight call sites: `tests/test_tui_integration.py` builds a
#: roster with ONE bundled entry and presses enter to reach the ledger, which works only while
#: the first row of the table is selectable data. Keeping find last leaves that true for every
#: roster that has any data at all, and costs the grouping nothing — "find new data" is an
#: escape hatch rather than a kind of dataset.
#: The gloss goes in the CONTENTS column, not in the title. A heading's text shares the `data`
#: column with every filename, so a long title widens that column for all of them — measured:
#: "synthetic — shapes with known answers" pushed every size and QC figure ~20 cells right.
GROUPS: tuple = (
    ("\u25a3", "measured data", "measured when it loaded",
     lambda e: e.obs.modality != "synthetic"),
    ("\u25c7", "synthetic", "shapes with a known answer",
     lambda e: e.obs.modality == "synthetic"),
)


def subtitle(filtering: bool) -> str:
    """The bottom-border hint. This app has no `Footer`, so a screen's own hint line is the
    ONLY place a key is advertised — which makes a wrong one expensive.

    It depends on state because `escape` does: with a filter typed it clears, and only on an
    empty box does it leave. A fixed string is therefore wrong in one of the two states, and the
    one this shipped with was wrong in a worse way — it named `ctrl+q`, which the VS Code
    integrated terminal swallows as "Quit Window", on the one screen that had no other way out.
    The app now binds `ctrl+c` explicitly (`tui/app.py`), so both keys named here are keys that
    work on the terminal this was launched in.
    """
    return ("↑↓ · type to filter · enter · ctrl+r rescan · esc clears the filter" if filtering
            else "↑↓ · type to filter · enter · ctrl+r rescan · esc or ctrl+c to quit")


def cells(entry: DataEntry) -> dict[str, str]:
    """One roster row as text. Reads the entry's own properties and phrases nothing itself."""
    # BUNDLED DOES NOT MEAN SYNTHETIC. The catalog declares thirteen synthetic modalities and
    # one scrna (`pbmc3k`, human cells); packaging cannot decide whether its QC is shown.
    measured = entry.obs.modality != "synthetic"
    qc = entry.qc_cells if measured else {"per_cell": "", "mito": "", "drops": ""}
    return {"mark": "·", "data": entry.name, "size": entry.size,
            "geometry": entry.glyphs, "contents": entry.contents,
            "per_cell": qc["per_cell"], "mito": qc["mito"], "drops": qc["drops"]}


def heading_cells(mark: str, title: str, gloss: str = "") -> dict:
    """A group heading, shaped like a row so one table can hold both.

    Every column but the first two is empty: a heading states what the rows under it have in
    common, and a heading carrying a size or a gene count would be stating a fact about a group
    rather than about data.
    """
    return {k: "" for k, _ in COLUMNS} | {"mark": mark, "data": title, "contents": gloss}


def table_width(keys: "tuple[str, ...] | list[str]", rows: list[dict]) -> int:
    """How wide this table renders — the widest cell in each column, plus the padding.

    Pure and takes plain dicts, so the fitting rule is testable with no terminal and no data.
    """
    total = 0
    for key in keys:
        header = dict(COLUMNS).get(key, "")
        total += max([cell_len(header)]
                     + [cell_len(str(r.get(key, ""))) for r in rows]) + _CELL_PADDING
    return total


def columns_that_fit(width: int, rows: list[dict]) -> tuple[str, ...]:
    """Which columns to draw at this width. Drops the rightmost until it fits.

    This is the replacement for the `width=30` literal, and the difference is not style: a
    declared width is wrong at every terminal size except one, and `DataTable` answers an
    overflow by scrolling sideways (measured: content 81 wide in a 60-wide table gives
    `virtual_size.width == 81`), so a column that does not fit does not wrap — it hides behind a
    scrollbar the user has no reason to look for. Dropping it is the honest version of that.

    `width <= 0` means the table has not been laid out yet; everything is kept, and the first
    real resize decides.
    """
    keys = [k for k, _ in COLUMNS]
    if width <= 0:
        return tuple(keys)
    while len(keys) > _KEEP and table_width(keys, rows) > width:
        keys.pop()
    return tuple(keys)


def matches(text: str, row: dict) -> bool:
    """The filter: case-insensitive substring over the row's VISIBLE text.

    Over what is on screen rather than over the name alone, because a filter that ignores the
    columns beside it makes "branching" fail on the row that literally reads "a branching tree".
    """
    needle = text.strip().lower()
    if not needle:
        return True
    return needle in " ".join(str(v) for v in row.values()).lower()


class RosterTable(DataTable):
    """The rows. It owns the resize event because the width that decides the columns is THIS
    widget's, not the screen's — the screen's includes a border, two margins and whatever
    scrollbar is up, and subtracting those would be three more magic numbers in place of the one
    we are removing."""

    class Resized(Message):
        """A trigger, with no payload: the width that decides is read back from the table's
        content region when the screen handles this, and a width carried here would be the
        widget's outer one — a second, wrong number in flight."""

    def on_resize(self, event: events.Resize) -> None:
        # `on_resize` is free on DataTable — the framework's own handler is `_on_resize`
        # (checked: `hasattr(DataTable, "on_resize") is False`), so this adds a handler rather
        # than replacing scrollbar bookkeeping.
        self.post_message(self.Resized())


class RosterScreen(Screen):
    """Choose your data. Rows are selectable, typing filters, enter opens the ledger."""

    BINDINGS = [
        Binding("ctrl+r", "rescan", "rescan", show=False),
        # Up/down are bound HERE, not on the table, because focus stays in the filter box so
        # that typing filters without a "press / to search" ceremony. Measured: `Input.BINDINGS`
        # binds neither `up` nor `down`, so both bubble out of it to this screen uncontested.
        Binding("down", "move(1)", "next", show=False),
        Binding("up", "move(-1)", "previous", show=False),
        # Escape CLEARS a filter, and quits when there is nothing to clear. The roster is the
        # root screen: with no filter typed there is nothing to go back to, so an escape that
        # only cleared left the landing screen with no exit at all — and the default `ctrl+q`
        # is eaten by the VS Code terminal. Every other screen binds escape to "back", so this
        # reads the same everywhere: escape means "out of whatever I am in".
        Binding("escape", "clear_or_quit", "clear filter / quit", show=True),
    ]

    class DataChosen(Message):
        """Enter on a data row. Carries the `DataEntry`, whose `as_source()` is already the
        tuple `shell._run` takes — so nothing downstream needs a second convention."""

        def __init__(self, entry: DataEntry) -> None:
            super().__init__()
            self.entry = entry

    class FindDataRequested(Message):
        """Enter on the `find data…` row. Component 5 owns what opens."""

    def __init__(self, *, entries: "Optional[list[DataEntry]]" = None) -> None:
        super().__init__()
        #: Injectable so a test can drive the screen without a drop folder. `None` means "read
        #: the real one on mount".
        self._preloaded = entries
        self.entries: list[DataEntry] = []
        #: What is on screen, row for row. `None` is the find-data row. The cursor is an index
        #: into this, which is why it is kept rather than re-derived from the widget.
        self.on_screen: list = []
        self.column_keys: tuple[str, ...] = ()
        self.load_error: Optional[str] = None
        self._watch = dropwatch.DropWatch()
        self._poll_timer = None
        self._loaded_poll = None
        self._failed_sources = {}
        self._folder_error = None
        self._skipped_declarations = set()
        self._selection_active = False
        self._announcement = ""
        self._started = False
        self._initial_population = entries is None
        self._initial_drops = None

    # ── layout ───────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        # The same sentence `shell.run_shell` greets with (shell.py:483). One tagline, two
        # front doors; if it ever becomes a constant both should read it.
        yield Static(f"[bold]manyruns[/bold]  ·  {narrate.TAGLINE}",
                     id="title")
        yield Static(id="footnote")
        with Vertical(id="roster-body"):
            yield RosterTable(id="roster")
            yield Input(placeholder="type to filter", id="filter")

    @dropwatch.filesystem_guard('_poll_failed')
    def on_mount(self) -> None:
        """Mount chrome. Filesystem: footnote catalog reads, stat/lstat/scandir/listdir,
        readlink, Path.resolve/exists/is_file/is_dir; source inspection waits for polling.
        """
        body = self.query_one("#roster-body", Vertical)
        body.border_title = "choose your data"
        self.refresh_hint()

        table = self.query_one(RosterTable)
        table.cursor_type = "row"
        table.show_cursor = True
        self.query_one("#footnote", Static).update(
            footnote() if self._preloaded is None else "preloaded data")
        self.query_one("#filter", Input).focus()

        if self._preloaded is not None:
            self.entries = list(self._preloaded)
            self.refresh_rows()
            return
        table.add_columns("", "reading your data folder…")
        self.call_after_refresh(self._start_polling)

    @dropwatch.filesystem_guard('_poll_failed')
    def _start_polling(self) -> None:
        """Start polling. Filesystem: os.stat/lstat/scandir/listdir/readlink/mkdir,
        Path.resolve/exists/is_file/is_dir, io.open for catalog/source/cache reads,
        and os.replace for cache writes.
        """
        if not self.is_mounted or self._started:
            return
        self._started = True
        self.poll()
        self._poll_timer = self.set_interval(1, self.poll)

    @dropwatch.filesystem_guard('_poll_failed')
    def load(self) -> None:
        """Finder refresh uses the settle rule. Filesystem: os.stat/lstat/scandir/listdir/
        readlink/mkdir, Path.resolve/exists/is_file/is_dir, io.open reads and os.replace cache writes.
        """
        self.poll(force=True)

    @dropwatch.filesystem_guard('_poll_failed')
    def poll(self, *, force: bool = False) -> None:
        """Observe and paint under one guard, including invalidation and final refresh.

        Filesystem: os.stat/lstat/scandir/listdir/readlink/mkdir,
        Path.resolve/exists/is_file/is_dir, io.open for catalog/source/cache reads,
        and os.replace for cache writes.
        """
        if self._preloaded is not None or not self.is_mounted:
            return
        from manyruns import shell

        previous_folder_error = self._folder_error
        try:
            dropwatch.path_call(shell.ensure_drop_folder)
            self._folder_error = None
        except (OSError, ValueError) as error:
            self._folder_error = f"Could not create the drop folder: {error}"
        snapshot_error = None
        try:
            current = self._watch.observe(dropwatch.snapshot())
        except OSError as error:
            # Losing a root mid-scan invalidates its evidence. Drop stale local rows and
            # require fresh settled observations when the filesystem becomes readable.
            current = self._watch.observe(dropwatch.Snapshot((), {}, ()))
            snapshot_error = f"Could not read the data folders: {type(error).__name__}: {error}"
        if self._initial_drops is None:
            self._initial_drops = {
                current.snapshot.references[p] for p in current.snapshot.drops or ()
                if p in current.snapshot.references}
        # Keep stat observations while covered, but do not compete with a run for QC reads.
        if self.app.screen is not self or self._selection_active:
            return
        if force:
            self._failed_sources.clear()
        if (not force and current == self._loaded_poll
                and not snapshot_error and not self._folder_error and not previous_folder_error):
            return
        before = {e.identity for e in self.entries}
        changed = set()
        failed = set()
        skipped = set()
        try:
            self.entries = state.roster(skip=current.held, snapshot=current.snapshot,
                                        on_changed=changed.add, on_failed=failed.add,
                                        on_skipped=skipped.add,
                                        previous=self.entries, failures=self._failed_sources)
            self._skipped_declarations = skipped
            self.load_error = snapshot_error or (
                f"{len(failed)} files could not be read" if failed else None)
        except dropwatch.read_errors() as error:
            self._poll_failed(error)
            return
        for path in changed:
            self._watch.invalidate(path)
        new = {e.identity for e in self.entries} - before
        self._announcement = "new data available" if before and new else ""
        self._loaded_poll = None if changed else current
        self.refresh_rows()
        if not self._initial_drops.intersection(current.held | changed):
            self._initial_population = False
        self.refresh_footnote(len(current.held | changed))

    def _poll_failed(self, error: Exception) -> None:
        """Recover without disk I/O; all uncertain rows need fresh settling evidence."""
        self._watch.observe(dropwatch.Snapshot((), {}, ()))
        self._loaded_poll = None
        self.load_error = f'{type(error).__name__}: {error}'
        self.entries = [replace(e, pending=True, observation=None) for e in self.entries]
        if self.is_mounted and self.app.screen is self and not self._selection_active:
            self.refresh_rows()
            self.query_one('#footnote', Static).update(escape(self.load_error))

    def refresh_footnote(self, pending: int = 0) -> None:
        if self._selection_active or self._preloaded is not None:
            return
        lines = [footnote()]
        if pending:
            lines.append(f"copying… {pending} data source(s) waiting to settle")
        if self._announcement:
            lines.append(self._announcement)
        if self._skipped_declarations:
            lines.append(escape('Skipped declaration(s): '
                                + ', '.join(sorted(self._skipped_declarations))))
        for error in (self._folder_error, self.load_error):
            if error:
                lines.append(escape(error))
        self.query_one("#footnote", Static).update("\n".join(lines))

    @dropwatch.filesystem_guard('_poll_failed')
    def on_screen_resume(self) -> None:
        """Resume observation. Filesystem: os.stat/lstat/scandir/listdir/readlink/mkdir,
        Path.resolve/exists/is_file/is_dir, io.open for catalog/source/cache reads,
        and os.replace for cache writes.
        """
        if self._started:
            self.poll()

    def on_unmount(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None

    @dropwatch.filesystem_guard('_poll_failed')
    def action_rescan(self) -> None:
        """Rescan without bypassing settling. Filesystem: os.stat/lstat/scandir/listdir/
        readlink/mkdir, Path.resolve/exists/is_file/is_dir, io.open reads and os.replace cache writes.
        """
        self.poll(force=True)

    # ── the rows ─────────────────────────────────────────────────────────────
    def filtered(self) -> list[tuple[Optional[DataEntry], dict]]:
        """What belongs on screen: the filtered entries, then the find-data row.

        Entry and cells travel together — an index into one list, never a name looked up in
        another. Two rows CAN share a name (a file dropped into the folder beside a bundled
        dataset of the same stem), and a name lookup would open the wrong one.

        The find-data row is ALWAYS last and always present: when the filter matches nothing it
        is the only move left, which is exactly when a screen that hid it would be a dead end.
        """
        text = self.query_one("#filter", Input).value
        kept = [(e, c) for e, c in ((e, cells(e)) for e in self.entries) if matches(text, c)]

        # A HEADING ONLY APPEARS OVER ROWS THAT SURVIVED THE FILTER. Drawing an empty group
        # would answer "what else is there" with a title and nothing under it, and while
        # filtering that is most of the screen.
        rows: list[tuple[Any, dict]] = []
        for mark, title, gloss, belongs in GROUPS:
            members = [(e, c) for e, c in kept if belongs(e)]
            if not members:
                continue
            rows.append((HEADING, heading_cells(mark, title, gloss)))
            rows.extend(members)
        rows.append((None, dict(FIND_ROW)))
        return rows

    def selectable(self, row: int) -> bool:
        """Whether `row` is something enter could act on — data, or the find row."""
        return 0 <= row < len(self.on_screen) and self.on_screen[row] is not HEADING

    def first_selectable(self) -> int:
        """The row the screen should open on: the first that is not a group heading."""
        return next((i for i in range(len(self.on_screen)) if self.selectable(i)), 0)

    def usable_width(self) -> int:
        """The width the CELLS get.

        `scrollable_content_region` and not `size` or `content_size`, both of which count the
        vertical scrollbar's column — and a roster of fourteen datasets in a short terminal has
        a scrollbar. Measured at a 100-wide terminal with the bar up: `size.width` 96,
        `content_size.width` 96, `scrollable_content_region.width` 95. One column, but it is one
        column at every boundary, and the failure it causes is the sideways scroll this whole
        function exists to prevent.

        Zero before the first layout, which `columns_that_fit` reads as "not laid out yet".
        """
        return self.query_one(RosterTable).scrollable_content_region.width

    def refresh_rows(self) -> None:
        """Rebuild the table: filter, then fit the columns to the width we actually have."""
        table = self.query_one(RosterTable)
        selected = (self.on_screen[table.cursor_row]
                    if 0 <= table.cursor_row < len(self.on_screen) else HEADING)
        identity = selected.identity if isinstance(selected, DataEntry) else selected
        if self._initial_population or not any(
                isinstance(entry, DataEntry) for entry in self.on_screen):
            identity = HEADING  # initial population chooses the first available dataset
        rows = self.filtered()
        cell_rows = [c for _, c in rows]
        keys = columns_that_fit(self.usable_width(), cell_rows)

        table.clear(columns=True)
        for key in keys:
            table.add_column(dict(COLUMNS)[key], key=key)
        for row in cell_rows:
            # `.get` and not `[]`: a row is a plain dict and the column set is the schema, so a
            # row assembled anywhere else in this file must not be able to crash the screen by
            # being one key short.
            table.add_row(*[row.get(k, "") for k in keys])
        self.column_keys = keys
        self.on_screen = [e for e, _ in rows]

        if table.row_count:
            # NOT ROW 0 ANY MORE — row 0 is a group heading whenever there is any data, and a
            # cursor parked on one would make the first `enter` do nothing.
            opening = next((i for i, entry in enumerate(self.on_screen)
                            if entry is not HEADING and
                            (entry.identity if isinstance(entry, DataEntry) else entry) == identity),
                           self.first_selectable())
            table.move_cursor(row=opening)
            self.mark_cursor(opening)


    def refresh_hint(self) -> None:
        """Re-state the keys for the state the screen is actually in. One `Vertical` lookup."""
        self.query_one("#roster-body", Vertical).border_subtitle = subtitle(
            bool(self.query_one("#filter", Input).value))

    def mark_cursor(self, row: int) -> None:
        """`▸` on the cursor row, `·` (or `?`) elsewhere.

        Drawn explicitly rather than left to `DataTable`'s cursor styling because focus lives in
        the filter box: an unfocused table renders its cursor in the blurred style, and "which
        row does enter act on" is not a question a landing screen may leave to a shade of grey.
        """
        table = self.query_one(RosterTable)
        for i, (_, row_cells) in enumerate(self.filtered()):
            if i >= table.row_count:
                break
            # A heading never takes the cursor mark: it keeps the group's own glyph, which is
            # what tells you which group you are standing in.
            here = i == row and self.selectable(i)
            table.update_cell_at(Coordinate(i, 0), "▸" if here else row_cells["mark"])

    # ── events ───────────────────────────────────────────────────────────────
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.value:
            self._initial_population = False
        self.refresh_rows()
        # `escape` changed meaning the moment this box stopped being empty; the hint says so.
        self.refresh_hint()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Enter reaches the screen as `Input.Submitted` rather than as a key, because the focused
        # Input binds `enter` itself. Same action either way: open what the cursor is on.
        self.choose()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.mark_cursor(event.cursor_row)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if isinstance(event.widget, RosterTable):
            self._initial_population = False

    def on_roster_table_resized(self, event: RosterTable.Resized) -> None:
        """§2: `on_resize → refresh_all` is one of the two mechanisms Textual buys us.

        The event is the TRIGGER; the width is re-read from `usable_width()` rather than taken
        from `event.width`, because the event carries the widget's outer size and the cells only
        get the content region.

        Rebuilt only when the COLUMN SET changes, not on every pixel: a rebuild resets the
        cursor, and a table that jumps back to the first row while you drag a window edge is a
        worse defect than the one this fixes.
        """
        if columns_that_fit(self.usable_width(), [c for _, c in self.filtered()]) \
                != self.column_keys:
            self.refresh_rows()

    # ── actions ──────────────────────────────────────────────────────────────
    def action_move(self, delta: int) -> None:
        table = self.query_one(RosterTable)
        if not table.row_count:
            return
        # Even an arrow at the edge confirms the row already under the cursor.
        self._initial_population = False
        # STEP OVER HEADINGS rather than landing on them: the same thing Textual's
        # `find_first_enabled` does for the ledger's disabled options, done by hand because a
        # `DataTable` row cannot be disabled. `delta` is ±1, so walking one at a time in that
        # direction lands on the next selectable row or stops at the end.
        row = max(0, min(table.row_count - 1, table.cursor_row + delta))
        step = 1 if delta > 0 else -1
        while 0 <= row < table.row_count and not self.selectable(row):
            row += step
        if not (0 <= row < table.row_count and self.selectable(row)):
            return                      # the edge is a heading; stay where we are
        table.move_cursor(row=row)
        self.mark_cursor(row)

    def action_clear_or_quit(self) -> None:
        box = self.query_one("#filter", Input)
        if box.value:
            box.value = ""
            return
        self.app.exit()

    def choose(self) -> None:
        """Act on the cursor row: a dataset opens the ledger, `find data…` opens the finder."""
        table = self.query_one(RosterTable)
        if not self.on_screen or table.cursor_row >= len(self.on_screen):
            return
        entry = self.on_screen[table.cursor_row]
        if entry is HEADING:
            return                      # a group name is not a thing you can open
        if self._selection_active:
            return
        self._initial_population = False
        if entry is None:
            self.post_message(self.FindDataRequested())
            return
        self._selection_active = True

        def done():
            """End selection. Filesystem primitives: none."""
            self._selection_active = False

        @dropwatch.filesystem_guard(self._poll_failed)
        def ready(resolved):
            """Refresh and hand off. Filesystem: footnote catalog/source metadata reads,
            os.stat/lstat/scandir/listdir/readlink and Path.resolve/exists/is_file/is_dir.
            """
            if not self.is_mounted or self.app.screen is not self:
                return
            # Selection rechecks the signature that made this row eligible for inspection.
            self.entries = [resolved if e.identity == resolved.identity else e for e in self.entries]
            self.refresh_rows()
            self.refresh_footnote()
            self.post_message(self.DataChosen(resolved))

        samplefetch.prepare_entry(self.app, entry, ready, on_done=done)


def footnote() -> str:
    """The header's footnote: what will run, and what is missing.

    §6 of the spec: `_pick_engine` becomes a footnote here rather than prompt 5, because
    `app._default_engine()` has already chosen and the user was being asked to ratify a decision
    they had no new information about. The three rows are `shell.state_rows()` unaltered — where
    your data would come from, how it would run, what is in the catalog.

    FIXED, AND WORTH KEEPING FIXED: the engine row used to read
    "uv pip install 'manyruns[real]'" when no backend was installed, and the bracketed token
    was swallowed as a markup tag — by Textual (measured: the Static rendered
    "uv pip install 'manyruns'") *and* by rich on the plain surface, same string, same result.
    The fix was in the STRING, not in any renderer: `shell.state_rows` now says "none installed
    — reinstall: the compute ships with manyruns", which carries no bracketed token for markup
    to eat, on any of the three surfaces. Pinned by
    `test_the_no_engine_line_survives_markup_now_that_it_has_no_brackets`.

    So: do not reintroduce a bracketed token into a row this renders. Escaping it here alone
    would make the TUI print a different install command from the other two surfaces.
    """
    from manyruns import shell

    try:
        rows = dropwatch.path_call(shell.state_rows)
    except dropwatch.read_errors() as e:
        # `state_rows` reads the catalog, and `$MANYRUNS_RECIPE_DIR` pointing at a folder that
        # is not one raises out of `discover_recipes`. A mistyped environment variable must cost
        # you the footnote, not the front door.
        return f"[red]could not read your setup: {type(e).__name__}: {e}[/red]"
    pad = max((len(label) for label, _ in rows), default=0)
    return "\n".join(f"[dim]{label:<{pad}}[/dim]  {value}" for label, value in rows)
