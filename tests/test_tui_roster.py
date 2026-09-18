"""The app shell and the roster screen — component 2 of the TUI rewrite.

Driven with `App.run_test()` and a `Pilot`: real keys, real layout, real resize, no terminal.
Nothing here shells out and nothing here needs `questionary`
— the injected-prompter pattern is replaced by the pilot for the app surface only, and the other
two surfaces keep theirs.

MOST TESTS INJECT THEIR ROWS (`RosterScreen(entries=…)`) rather than reading the real drop
folder, because `state.roster()` opens a file per dropped row (measured: 361 ms cold on the
5.9 MB `data/pbmc3k_raw.h5ad`) and because a screen test should fail when the SCREEN breaks, not
when the bundled catalog gains a dataset. The two tests that do read the real roster say so in
their names, and both point `$MANYRUNS_DATA_DIR` at a tmp folder so they do not depend on what
is sitting in this checkout's `data/`.
"""
from __future__ import annotations

import ast
import asyncio
import functools
import importlib
import inspect
from pathlib import Path

import pytest

pytest.importorskip("textual", reason="the TUI surface is optional; the other two are not")

from textual.widgets import Input, Static  # noqa: E402

from manyruns.narrate import Observation  # noqa: E402
from manyruns.tui import state  # noqa: E402
from manyruns.tui.app import ManyrunsApp  # noqa: E402
from manyruns.tui.roster import RosterScreen, RosterTable  # noqa: E402

#: `import_module`, not `from manyruns.tui import roster`, because those two names COLLIDE:
#: `state.roster` is a function and `manyruns/tui/roster.py` is this screen's module. Python
#: setattrs a submodule onto its parent package on import, so while the package re-exported the
#: function, `from manyruns.tui import roster` meant the function until something imported the
#: app and the module afterwards — measured here as
#: `AttributeError: 'function' object has no attribute 'columns_that_fit'`. The re-export has
#: since been dropped from `manyruns/tui/__init__.py`; `import_module` means the module either
#: way, which is why it stays.
roster_screen = importlib.import_module("manyruns.tui.roster")


def pilot_test(fn):
    """Run one `App.run_test()` coroutine as an ordinary pytest test.

    There is no `pytest-asyncio` in this repo and this component does not add one: an async
    plugin is a dev dependency for every test here, and `asyncio.run` around the one coroutine
    buys the same thing in four lines. `functools.wraps` keeps `__wrapped__`, which is how
    pytest still sees the real signature and injects `tmp_path` / `monkeypatch`.

    Without it an `async def test_` is COLLECTED AND SKIPPED with a warning, which is the worst
    outcome available: a green suite that ran none of these.
    """
    @functools.wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))

    return run


@pytest.fixture(autouse=True)
def _rich_package_is_whole():
    """Repair a HALF-PURGED `rich` before driving the app. Not this component's bug — reported,
    and worked around here because the fix belongs in a file this component does not own.

    `tests/test_shell.py::test_base_import_chain_pulls_no_sdk_or_tui` does
    `sys.modules.pop("rich")` to prove the base import chain stays SDK-free, but leaves
    `sys.modules["rich.repr"]` and friends behind. The next `import rich.repr` then re-executes
    the parent package into a NEW module object while the submodule is served from the cache, so
    the attribute is never set on it — and `textual.drivers.linux_driver`, which is imported
    lazily inside `App.run_test()`, is decorated `@rich.repr.auto`. Measured: that one test alone
    ahead of this file turns 22 passed into 18 failed with
    `AttributeError: module 'rich' has no attribute 'repr'`.

    The repair RE-ATTACHES the cached submodules rather than purging them, so there is never a
    second copy of rich in the process for anything else to fail an isinstance against.

    The one-line fix, for whoever owns that file: pop `rich` AND every `rich.*` key.
    """
    import sys

    if "rich" not in sys.modules and any(n.startswith("rich.") for n in sys.modules):
        parent = importlib.import_module("rich")
        for name, module in list(sys.modules.items()):
            parts = name.split(".")
            if parts[0] == "rich" and len(parts) == 2:
                setattr(parent, parts[1], module)


@pytest.fixture(autouse=True)
def _roster_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)


def _entry(name: str, shape: str = "single", **kw) -> state.DataEntry:
    obs = Observation(shape=shape, modality=kw.pop("modality", "scrna"),
                      n_obs=kw.pop("n_obs", None), n_vars=kw.pop("n_vars", None), source=name)
    return state.DataEntry(name=name, kind=kw.pop("kind", "bundled"), obs=obs, **kw)


#: Three rows that exercise the three "size" answers and two different `contents` phrases.
FIXTURE = [
    _entry("pbmc3k_raw.h5ad", kind="file", n_obs=2700, n_vars=32738, path=Path("pbmc3k_raw.h5ad")),
    _entry("swissroll", shape="manifold", modality="synthetic", dataset="swissroll"),
    _entry("tree_wide", shape="manifold", modality="synthetic", dataset="tree_wide",
           topology=("multi-branching",)),
]


class RosterApp(ManyrunsApp):
    """The real app with its rows injected, and both seams recorded instead of pushed.

    Overriding `open_ledger` / `open_finder` is exactly what components 3 and 5 will do, so a
    test that overrides them is testing the seam they will use rather than a test hook.
    """

    def __init__(self, entries=None) -> None:
        super().__init__()
        self.entries = FIXTURE if entries is None else entries
        self.opened: list = []

    def get_default_screen(self):
        return RosterScreen(entries=self.entries)

    def open_ledger(self, entry) -> None:
        self.opened.append(("ledger", entry))

    def open_finder(self) -> None:
        self.opened.append(("finder", None))


def _data_rows(app: ManyrunsApp) -> list[int]:
    """Row indices that are DATA or the find row — group headings excluded.

    The roster groups its rows under headings ("your own data", "synthetic") that are drawn in
    the same table. A heading is not something enter can act on and not a dataset, so every
    assertion below about "the rows on screen" means these.
    """
    return [i for i in range(app.screen.query_one(RosterTable).row_count)
            if app.screen.selectable(i)]


def _names(app: ManyrunsApp) -> list[str]:
    """The `data` column, as the table actually holds it — headings excluded."""
    table = app.screen.query_one(RosterTable)
    col = app.screen.column_keys.index("data")
    return [table.get_row_at(i)[col] for i in _data_rows(app)]


def _marks(app: ManyrunsApp) -> list[str]:
    table = app.screen.query_one(RosterTable)
    return [table.get_row_at(i)[0] for i in _data_rows(app)]


def _under_cursor(app: ManyrunsApp) -> str:
    """The name of the dataset the cursor is on. BY NAME, not by row index: headings shift
    every index, and an index assertion would pass or fail on where a heading happened to
    land rather than on what the keyboard actually selected."""
    screen = app.screen
    entry = screen.on_screen[screen.query_one(RosterTable).cursor_row]
    return "find data…" if entry is None else entry.name


# ── launch ───────────────────────────────────────────────────────────────────
@pilot_test
async def test_launch_lands_on_the_roster_with_nothing_underneath_it():
    """§3.1: "launch goes straight here. No home menu."

    The stack depth is the assertion that matters. A roster PUSHED over a menu would look
    identical on screen and leave `escape` going back to the question §0 counts as prompt 1.
    """
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, RosterScreen)
        assert len(app.screen_stack) == 1
        assert app.screen.query_one("#title", Static).visual.plain.startswith("manyruns")


def _syntax(module) -> tuple[list[str], list[str], list[str]]:
    """(string literals, imported module names, identifiers) — the module's CODE, with its prose
    left out. Asserting on raw source cannot work here: these files QUOTE the prompts they
    delete, in the docstrings that explain why they are gone."""
    tree = ast.parse(inspect.getsource(module))
    scopes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    docs = {ast.get_docstring(n, clean=False) for n in ast.walk(tree) if isinstance(n, scopes)}
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value not in docs]
    imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    imports += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    names += [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    names += [a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments) for a in n.args]
    return literals, imports, names


def test_the_app_asks_no_question_and_owns_no_prompter():
    """Prompts 1, 3, 4 and 5 are gone (§6), and the app cannot re-grow them by accident: it has
    no prompter to ask with. `PlainPrompter` survives for the no-TTY path (§5) and is untouched
    — this pins only that the Textual surface never reaches for one."""
    from manyruns.tui import app as app_module

    for module in (roster_screen, app_module):
        literals, imports, names = _syntax(module)
        for asked in ("What do you want to do?", "Which data?", "What should I look for?",
                      "Which recipe?", "How should I run it?"):
            assert asked not in literals, f"{module.__name__} still asks {asked!r}"
        assert not any("questionary" in i for i in imports), module.__name__
        assert not any("prompt" in n.lower() for n in names), module.__name__


@pilot_test
async def test_the_chrome_is_on_screen_before_the_data_folder_is_read(monkeypatch):
    """The measured first-paint decision, asserted rather than described.

    `state.roster()` costs 361 ms cold; the screen therefore composes, paints, and reads on the
    next refresh. This spy runs INSIDE the read and asserts the screen it will fill is already
    mounted and already showing its header — so moving the read into `__init__` or ahead of
    `compose` (which is what "just load it first" looks like) fails here.
    """
    seen: dict = {}
    app = ManyrunsApp()

    def spy(*a, **k):
        screen = app.screen
        seen["mounted"] = screen.is_mounted
        seen["footnote"] = screen.query_one("#footnote", Static).visual.plain
        seen["table"] = screen.query_one(RosterTable).is_mounted
        return []

    monkeypatch.setattr(state, "roster", spy)
    async with app.run_test() as pilot:
        await pilot.pause()
    assert seen["mounted"] is True and seen["table"] is True
    assert "engine" in seen["footnote"]


@pilot_test
async def test_a_drop_folder_that_cannot_be_read_says_so_instead_of_going_blank(monkeypatch):
    """`state.roster` already skips one unreadable entry. This is the whole read failing — and a
    landing screen with no rows and no reason is the worst possible first frame."""
    def boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(state, "roster", boom)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, RosterScreen)
        assert "permission denied" in app.screen.query_one("#footnote", Static).visual.plain
        assert _names(app) == ["find data…"]


# ── the rows ─────────────────────────────────────────────────────────────────
@pilot_test
async def test_the_rows_are_the_roster_plus_a_find_data_row_at_the_bottom():
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert _names(app) == ["pbmc3k_raw.h5ad", "swissroll", "tree_wide", "find data…"]
        assert _marks(app) == ["▸", "·", "·", "?"]
        table = app.screen.query_one(RosterTable)
        row = table.get_row_at(_data_rows(app)[0])   # row 0 is the "your own data" heading
        assert row[app.screen.column_keys.index("size")] == "2,700 × 32,738"
        assert row[app.screen.column_keys.index("contents")] == "one population"


@pilot_test
async def test_the_drop_folder_comes_first_then_the_bundled_set(tmp_path, monkeypatch):
    """The one test that reads the REAL roster. Same order as `shell._pick_source`, so moving
    between the two pickers is not a re-learn."""
    (tmp_path / "mine.csv").write_text("a,b\n1,2\n")
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path))

    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.poll()
        names = _names(app)
        assert names[0] == "mine.csv"
        assert names[-1] == "find data…"
        assert {r.name for r in state.roster()} == set(names[:-1])


# ── typing filters ───────────────────────────────────────────────────────────
@pilot_test
async def test_typing_filters_the_rows_without_a_search_mode():
    """No "press / to search": the filter box has focus from the first frame, so the first key
    a scientist presses narrows the list."""
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.focused, Input)
        await pilot.press("s", "w", "i")
        await pilot.pause()
        assert _names(app) == ["swissroll", "find data…"]
        assert _under_cursor(app) == "swissroll"


@pilot_test
async def test_the_filter_is_case_insensitive_and_reads_the_whole_row():
    """A filter over the name alone makes "branching" fail on the row that literally reads
    "a branching tree" — the user filters what they can see."""
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"BRANCH")
        await pilot.pause()
        assert _names(app) == ["tree_wide", "find data…"]


@pilot_test
async def test_escape_clears_the_filter_and_the_rows_come_back():
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"swi")
        await pilot.pause()
        assert len(_names(app)) == 2
        await pilot.press("escape")
        await pilot.pause()
        assert len(_names(app)) == 4


@pilot_test
async def test_the_find_data_row_survives_a_filter_that_matches_nothing():
    """When nothing matches, "find data…" is the only move left — which is exactly when a screen
    that filtered it away would be a dead end."""
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"zzzz")
        await pilot.pause()
        assert _names(app) == ["find data…"]
        await pilot.press("enter")
        await pilot.pause()
        assert app.opened == [("finder", None)]


# ── the cursor, and what enter opens ─────────────────────────────────────────
@pilot_test
async def test_the_cursor_is_drawn_because_the_table_never_holds_focus():
    """Focus stays in the filter box so typing filters; `DataTable` would therefore render its
    cursor in the blurred style. The `▸` is what actually answers "which row does enter act
    on", so it must move with the arrow keys."""
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert _marks(app)[0] == "▸"
        await pilot.press("down", "down")
        await pilot.pause()
        assert _marks(app) == ["·", "·", "▸", "?"]
        assert _under_cursor(app) == "tree_wide"
        assert isinstance(app.focused, Input), "the arrow keys must not steal focus"


@pilot_test
async def test_enter_opens_the_ledger_for_the_row_under_the_cursor(monkeypatch):
    """§3.1: enter opens the ledger for that dataset. The seam carries the `DataEntry` itself,
    whose `as_source()` is already the tuple `shell._run` takes."""
    def ready(app, entry, on_ready, *, on_done=None):
        if on_done:
            on_done()
        on_ready(entry)

    monkeypatch.setattr(roster_screen.samplefetch, 'prepare_entry', ready)
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        assert [kind for kind, _ in app.opened] == ["ledger"]
        entry = app.opened[0][1]
        assert entry.name == "swissroll"
        assert entry.as_source() == (None, "swissroll", "synthetic", entry.obs)


@pilot_test
async def test_the_cursor_survives_the_arrow_keys_at_both_ends():
    """`down` at the last row and `up` at the first must not wrap or go out of range — the row
    the cursor is on is what `enter` acts on, so an off-by-one here opens the wrong dataset."""
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()
        # `up` from the first dataset stays on it — the heading above it is stepped over,
        # never landed on.
        assert _under_cursor(app) == "pbmc3k_raw.h5ad"
        for _ in range(10):
            await pilot.press("down")
        await pilot.pause()
        assert _under_cursor(app) == "find data…"
        await pilot.press("enter")
        await pilot.pause()
        assert app.opened == [("finder", None)]


@pilot_test
async def test_the_default_seams_open_the_two_real_screens():
    """The real `ManyrunsApp` (not the test subclass): both hooks now push the screen they name.

    Until the components were wired together each pushed a `PendingScreen` that said which
    component owned what was missing; this is the same assertion against what replaced it, and it
    is the roster's half of the coexistence proof — the landing screen posts a message and knows
    nothing about what opens.
    """
    from manyruns.tui.find import FindScreen
    from manyruns.tui.ledger import LedgerScreen

    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.post_message(RosterScreen.DataChosen(FIXTURE[1]))
        await pilot.pause()
        assert isinstance(app.screen, LedgerScreen)
        assert app.screen.entry.name == "swissroll"
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RosterScreen)

        app.screen.post_message(RosterScreen.FindDataRequested())
        await pilot.pause()
        assert isinstance(app.screen, FindScreen)


# ── layout: the width=30 defect, and its replacement ─────────────────────────
def test_the_fitting_rule_drops_the_rightmost_column_and_never_the_name():
    """Pure, no terminal. The rule: drop right-to-left, never below `mark` + `data`."""
    rows = [{"mark": "·", "data": "pbmc3k_raw.h5ad", "size": "2,700 × 32,738",
             "contents": "one population"}]
    every = tuple(k for k, _ in roster_screen.COLUMNS)
    assert roster_screen.columns_that_fit(200, rows) == every
    # THE RULE, not a literal: narrower is always a PREFIX of wider (drop from the right), and
    # nothing goes below `mark` + `data`.
    wide, mid, tight = (roster_screen.columns_that_fit(w, rows) for w in (200, 40, 20))
    assert wide[:len(mid)] == mid and mid[:len(tight)] == tight
    assert len(mid) < len(wide)
    assert tight == ("mark", "data")
    # a name is not droppable: two columns is the floor, and the table scrolls instead
    assert roster_screen.columns_that_fit(1, rows) == ("mark", "data")
    # not laid out yet → decide nothing, and let the first resize decide
    assert roster_screen.columns_that_fit(0, rows) == tuple(k for k, _ in roster_screen.COLUMNS)


def test_the_fitting_rule_measures_the_content_and_not_a_declared_width():
    """The whole point of replacing `width=30`: the same column fits or does not fit depending
    on what is IN it, which a literal cannot know."""
    short = [{"mark": "·", "data": "ab", "size": "10", "contents": "a continuum"}]
    long = [{"mark": "·", "data": "a" * 40, "size": "10", "contents": "a continuum"}]
    # the same width keeps MORE columns for short content than for long — that is the point
    assert len(roster_screen.columns_that_fit(40, short)) > len(roster_screen.columns_that_fit(40, long))
    assert roster_screen.columns_that_fit(40, long) == ("mark", "data")


@pilot_test
async def test_a_narrow_terminal_drops_a_column_instead_of_scrolling_sideways():
    """§0's other defect: `render_result`'s hardcoded `width=30` splits `ndarray(18…` /
    `10) float32` across lines. Measured here instead of asserted in prose — at each width the
    table's rendered content must fit the region it has, so nothing hides behind a sideways
    scrollbar the user has no reason to look for."""
    app = RosterApp()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        table = app.screen.query_one(RosterTable)
        every = tuple(k for k, _ in roster_screen.COLUMNS)
        assert app.screen.column_keys == every[:len(app.screen.column_keys)]   # a prefix

        # The RULE at each width — a prefix of the full set, shrinking as the terminal does —
        # rather than a literal count, which changed the day the QC columns landed and would
        # change again with the next one.
        shown = len(app.screen.column_keys)
        for width in (100, 46, 30):
            await pilot.resize_terminal(width, 24)
            await pilot.pause()
            # narrower never shows MORE columns, and never fewer than mark + data
            assert roster_screen._KEEP <= len(app.screen.column_keys) <= shown, f"at {width}"
            shown = len(app.screen.column_keys)
            # THE ASSERTION THAT MATTERS, unchanged: whatever fits, nothing hides sideways.
            assert table.virtual_size.width <= table.scrollable_content_region.width, (
                f"the table scrolls sideways at width {width}")
        assert shown == roster_screen._KEEP, "30 columns must be down to the two kept ones"

        await pilot.resize_terminal(100, 24)
        await pilot.pause()
        # back at 100, every column that fits is back — a prefix of the full set, and at least
        # the four the roster always had
        every = tuple(k for k, _ in roster_screen.COLUMNS)
        assert app.screen.column_keys == every[:len(app.screen.column_keys)]
        assert len(app.screen.column_keys) >= 4


@pilot_test
async def test_a_resize_that_changes_nothing_does_not_rebuild_the_table():
    """A rebuild resets the cursor. A table that jumps back to row 0 while you drag a window
    edge is a worse defect than the one the resize handler fixes."""
    app = RosterApp()
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        await pilot.press("down", "down")
        await pilot.pause()
        assert _under_cursor(app) == "tree_wide"
        before = app.screen.column_keys
        await pilot.resize_terminal(98, 24)          # two cells narrower: same column set
        await pilot.pause()
        assert app.screen.column_keys == before
        assert _under_cursor(app) == "tree_wide", "a resize reset the cursor"


def test_no_width_is_declared_in_cells_anywhere_in_this_component():
    """The rule that replaces `render_result`'s `width=30`, enforced over both the screen and
    the stylesheet. `1fr` / `auto` / `100%` are constraints the layout engine solves; `width: 30`
    is a guess about a terminal nobody has.

    On the AST, not the text, because this component's comments quote the literal they replace.
    """
    import re

    tree = ast.parse(inspect.getsource(roster_screen))
    for node in ast.walk(tree):
        for kw in getattr(node, "keywords", []):
            assert not (kw.arg in ("width", "height")
                        and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, int)), f"declared {kw.arg} in the screen"

    css = Path(roster_screen.__file__).with_name("manyruns.tcss").read_text()
    for decl in re.findall(r"(?:min-|max-)?(?:width|height)\s*:\s*([^;]+);", css):
        assert not re.fullmatch(r"\s*\d+\s*", decl), f"declared cell size in the CSS: {decl!r}"


# ── the header footnote: an answer, not a question ───────────────────────────
@pilot_test
async def test_the_header_reports_the_engine_instead_of_asking_which_one():
    """§6: `_pick_engine` becomes a footnote — what will run, and what is missing. The user is
    never asked, because `app._default_engine()` has already decided and the question offered no
    new information."""
    from manyruns import app as product

    chosen = product._default_engine()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        text = app.screen.query_one("#footnote", Static).visual.plain
        assert "engine" in text
        if chosen:
            assert chosen in text
        assert "?" not in text, "the footnote is a statement, not a prompt"


@pilot_test
async def test_the_footnote_is_the_shell_state_rows_and_not_a_second_account(monkeypatch):
    """Three surfaces, one source. `shell.state_rows`'s own docstring says it is pure data so
    the plain and panel renderings can share it; this is the third reader, and it re-derives
    nothing — swap the source and the header changes with it."""
    from manyruns import shell

    monkeypatch.setattr(shell, "state_rows", lambda: [("engine", "borrowed"), ("data", "0 in x")])
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        text = app.screen.query_one("#footnote", Static).visual.plain
        assert "engine  borrowed" in text and "data    0 in x" in text


@pilot_test
async def test_a_setup_the_header_cannot_read_costs_the_footnote_and_not_the_front_door(
    monkeypatch,
):
    """`shell.state_rows` reads the catalog, and `$MANYRUNS_RECIPE_DIR` pointing at a folder
    that is not one raises out of `discover_recipes`. A mistyped environment variable must not
    be the reason `manyruns` will not open."""
    from manyruns import shell

    def boom():
        raise ValueError("no recipes in /nope")

    monkeypatch.setattr(shell, "state_rows", boom)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, RosterScreen)
        assert "no recipes in /nope" in app.screen.query_one("#footnote", Static).visual.plain
        assert "find data…" in _names(app)


@pilot_test
async def test_the_no_engine_line_survives_markup_now_that_it_has_no_brackets(monkeypatch):
    """**This replaces a test that pinned a defect, and the defect is fixed.**

    What it used to say: with no backend installed `shell.state_rows` returned
    "…uv pip install 'manyruns[real]'", and `[real]` was eaten as a markup tag — measured on
    Textual AND on rich, so the install command printed to the user who most needed it arrived
    missing its extra on both surfaces. Escaping it on one renderer would have made the three
    surfaces print different commands, so the defect was kept and named, with the note "this
    test fails when it lands".

    It landed sideways. The extra is gone — the compute is a hard dependency now — so the line
    no longer contains a bracketed token for markup to swallow. That is a better fix than
    escaping: a string with nothing to escape cannot be escaped wrongly on a fourth surface.
    """
    from manyruns import app as product

    monkeypatch.setattr(product, "engine_available", lambda name: False)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        text = app.screen.query_one("#footnote", Static).visual.plain

    assert "none installed" in text
    assert "reinstall" in text.lower(), "a missing backend is a broken install; say so"
    # nothing bracket-shaped is left for a renderer to eat
    assert "[" not in text and "]" not in text


# ── getting out ──────────────────────────────────────────────────────────────
@pilot_test
async def test_ctrl_c_quits_because_the_vscode_terminal_eats_ctrl_q():
    """Textual's DEFAULT quit chord is `ctrl+q`, and the VS Code integrated terminal claims it
    as "Quit Window" before the terminal sees it — so on the surface this app was first
    launched in, it had no exit. `ctrl+c` is the one chord every terminal forwards."""
    from manyruns.tui.app import ManyrunsApp

    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.return_value is None      # exited cleanly rather than raising


@pilot_test
async def test_escape_on_the_roster_clears_a_filter_then_leaves():
    """The roster is the ROOT screen — there is nothing behind it to go back to. Escape that
    only cleared a filter left the landing screen with no exit at all once the filter was
    already empty. Two presses: the first clears, the second leaves."""
    from textual.widgets import Input

    from manyruns.tui.app import ManyrunsApp

    app = ManyrunsApp()
    async with app.run_test() as pilot:
        box = app.screen.query_one("#filter", Input)
        box.value = "swiss"
        await pilot.pause()

        await pilot.press("escape")           # clears
        await pilot.pause()
        assert box.value == ""
        assert app.is_running, "escape with a filter typed must not quit"

        await pilot.press("escape")           # nothing left to clear -> leaves
        await pilot.pause()
    assert not app.is_running


#: How a key reads in a hint line, against how Textual names it in `active_bindings`.
SPELLINGS = {"esc": "escape", "ctrl+c": "ctrl+c", "ctrl+q": "ctrl+q", "enter": "enter"}


@pilot_test
async def test_the_roster_advertises_an_exit_that_survives_the_terminal_it_runs_in():
    """The hint line is the ONLY place this app states a key — there is no `Footer` — and the
    one it shipped with read "ctrl+q to quit", which is bound and which the VS Code integrated
    terminal swallows as "Quit Window". Advertised, bound, and unreachable all at once.

    A test cannot see the host terminal, so it pins the two halves it can: every key named is
    one the app actually binds, and at least one named exit is something OTHER than the chord a
    host is known to claim. Naming only `ctrl+q` is what the defect looked like."""
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        hint = app.screen.query_one("#roster-body").border_subtitle

        named = {key for token, key in SPELLINGS.items() if token in hint}
        assert named, f"the hint names no key at all: {hint!r}"
        assert not [k for k in named if k not in app.active_bindings], \
            f"{hint!r} advertises a key nothing binds"
        assert named & {"escape", "ctrl+c"}, \
            f"{hint!r} offers no exit that works where ctrl+q is eaten"


@pilot_test
async def test_the_hint_says_what_escape_means_right_now():
    """`escape` clears a filter when there is one and leaves when there is not, so a FIXED
    string has to be wrong in one of the two states. It follows the box."""
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        body = app.screen.query_one("#roster-body")
        assert "quit" in body.border_subtitle

        app.screen.query_one("#filter", Input).value = "swiss"
        await pilot.pause()
        assert "clears the filter" in body.border_subtitle
        assert "quit" not in body.border_subtitle       # escape does not quit right now


@pilot_test
async def test_ctrl_c_never_raises_the_question_it_cannot_answer():
    """Undeclared, `ctrl+c` fell through `Input`'s `copy` (it raises `SkipAction` with nothing
    selected) to Textual's `action_help_quit`, which posts a toast titled "Do you want to
    quit?". A toast takes no answer — `enter` does nothing to it — so the app read as hung by
    the person who had just tried to answer it. Reproduced in a real pty before this was bound.

    The assertion is on the ABSENCE: quitting outright is what stops the question."""
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running
        assert [n.title for n in app._notifications] == []


def test_filenames_are_measured_in_the_cells_the_table_draws():
    """Ten CJK characters occupy twenty cells; Python length under-budgeted the name by ten."""
    from rich.cells import cell_len

    name = "细胞" * 5 + ".h5ad"
    rows = [{"data": name, "size": "2 × 3"}]
    assert roster_screen.table_width(("data",), rows) == cell_len(name) + 2
    width = roster_screen.table_width(("mark", "data"), rows)
    assert roster_screen.columns_that_fit(width, rows) == ("mark", "data")


@pytest.mark.parametrize("kind", ["bundled", "file"])
def test_pbmc_is_measured_data_and_keeps_its_quality_columns(kind):
    """The bundled YAML declares scrna and genes; changing packaging must not erase its QC."""
    from manyruns.catalog import load_dataset

    ds = load_dataset("pbmc3k")
    assert ds["modality"] == "scrna" and "genes" in ds["handle"]["provides"]
    entry = _entry(ds["name"], modality=ds["modality"], kind=kind,
                   qc={"qc_n_cells": 2700, "qc_median_genes_per_cell": 817,
                       "qc_pct_mito_median": 2.0})
    groups = [title for _, title, _, belongs in roster_screen.GROUPS if belongs(entry)]
    assert groups == ["measured data"]
    assert roster_screen.cells(entry)["per_cell"] == "817"
    assert roster_screen.cells(entry)["mito"] == "2.0 %"
    synthetic = _entry("swissroll", modality="synthetic", kind=kind)
    assert [title for _, title, _, belongs in roster_screen.GROUPS if belongs(synthetic)] == [
        "synthetic"]
    assert roster_screen.cells(synthetic)["per_cell"] == ""


@pytest.fixture
def live_drop(tmp_path, monkeypatch):
    from manyruns import inspected
    monkeypatch.chdir(tmp_path)
    drop = tmp_path / 'drop'
    drop.mkdir()
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(drop))
    monkeypatch.delenv('MANYRUNS_DATASET_DIR', raising=False)
    monkeypatch.setattr(inspected, 'HOME', tmp_path / 'cache')
    return drop


@pytest.mark.parametrize('encoding', ['mtx', 'h5ad', 'legacy'])
@pytest.mark.parametrize('arrival', ['launch', 'drop'])
@pilot_test
async def test_sparse_roster_inspection_never_loads_matrix(
        live_drop, tmp_path, monkeypatch, encoding, arrival):
    import anndata
    import h5py
    import numpy as np
    import scipy.io
    import scipy.sparse as sparse
    import yaml
    from manyruns import datasetfetch
    from manyruns.tui import search
    from manyruns.tui.ledger import LedgerScreen

    assert live_drop.is_dir()
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    suffix = '.mtx' if encoding == 'mtx' else '.h5ad'
    path = live_drop / ('counts' + suffix)
    expected_size = '2,700 × 32,738' if encoding == 'legacy' else '2,000 × 20,000'
    (configs / 'counts.yaml').write_text(yaml.safe_dump({
        'name': 'counts', 'shape': 'single', 'modality': 'scrna',
        'handle': {'kind': 'path', 'ref': str(path)},
    }))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    matrix = sparse.csr_matrix(([1., 2.], ([0, 1999], [0, 19999])), shape=(2000, 20000))
    # Keep a second matrix in layers: anndata's backed reader eagerly reads these.
    staged = tmp_path / ('staged' + suffix)
    if suffix == '.mtx':
        scipy.io.mmwrite(staged, matrix)
    elif encoding == 'legacy':
        # Pre-0.7 AnnData layout used by the production sample, generated offline.
        with h5py.File(staged, 'w') as handle:
            node = handle.create_group('X')
            node.attrs['h5sparse_format'] = 'csr'
            node.attrs['h5sparse_shape'] = (2700, 32738)
            node['data'] = [1.]
            node['indices'] = [0]
            node['indptr'] = np.r_[0, np.ones(2700, dtype='i4')]
            handle['obs'] = np.array([(f'cell{i}'.encode(),) for i in range(2700)],
                                     dtype=[('index', 'S16')])
            handle['var'] = np.array([(f'gene{i}'.encode(),) for i in range(32738)],
                                     dtype=[('index', 'S16')])
        assert anndata.read_h5ad(staged).shape == (2700, 32738)
    else:
        anndata.AnnData(matrix, layers={'counts': matrix.copy()}).write_h5ad(staged)
    if arrival == 'launch':
        staged.rename(path)

    def forbidden(*args, **kwargs):
        pytest.fail('roster inspection loaded or densified the matrix')

    asarray = np.asarray
    def no_sparse_asarray(value, *args, **kwargs):
        if sparse.issparse(value):
            forbidden()
        return asarray(value, *args, **kwargs)

    reads = []
    read_h5ad = anndata.read_h5ad
    def tracked_read(*args, **kwargs):
        reads.append(kwargs.get('backed'))
        return read_h5ad(*args, **kwargs)

    getitem = h5py.Dataset.__getitem__
    def no_matrix_payload(node, *args, **kwargs):
        if node.name == '/X' or node.name.startswith(('/X/', '/layers/')):
            forbidden()
        return getitem(node, *args, **kwargs)

    for cls in (sparse.csr_matrix, sparse.csc_matrix, sparse.coo_matrix):
        monkeypatch.setattr(cls, 'todense', forbidden)
        monkeypatch.setattr(cls, 'toarray', forbidden)
    monkeypatch.setattr(np, 'asarray', no_sparse_asarray)
    monkeypatch.setattr(anndata, 'read_h5ad', tracked_read)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', no_matrix_payload)
    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', forbidden)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        assert not any(e.path == path and e.kind == 'file' for e in screen.entries)
        if arrival == 'drop':
            staged.rename(path)
            screen.poll()
            assert not any(e.path == path and e.kind == 'file' for e in screen.entries)
        screen.poll()
        rows = [e for e in screen.entries if e.path == path]
        assert len(rows) == 2  # drop and catalog alias
        for row in rows:
            assert not row.pending and not row.refusal
            assert row.size == expected_size
        screen.poll(force=True)
        assert search.roster_entry(path, live_drop).size == expected_size
        for kind in ('file', 'bundled'):
            index = next(i for i, e in enumerate(screen.on_screen)
                         if isinstance(e, state.DataEntry) and e.path == path and e.kind == kind)
            screen.query_one(RosterTable).move_cursor(row=index)
            await pilot.press('enter')
            assert isinstance(app.screen, LedgerScreen)
            assert app.screen.entry.size == expected_size
            await pilot.press('escape')
        assert app._exception is None
    assert all(mode == 'r' for mode in reads), reads


@pilot_test
async def test_string_counts_make_qc_unavailable_without_closing_roster(live_drop, monkeypatch):
    import anndata
    import numpy as np
    from manyruns import datasetfetch
    from manyruns.tui.ledger import LedgerScreen

    assert live_drop.is_dir()
    bad = live_drop / 'string-counts.h5ad'
    anndata.AnnData(np.array([['one', 'two'], ['three', 'four']], dtype=object)).write_h5ad(bad)
    good = live_drop / 'good.csv'
    good.write_text('a,b\n1,2\n3,4\n')

    def no_download(*args, **kwargs):
        raise AssertionError('local selection must not download')

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', no_download)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.call_later(screen.poll)
        await pilot.pause()
        assert app._exception is None
        entry = next(e for e in screen.entries if e.path == bad)
        assert entry.qc is None
        index = next(i for i, e in enumerate(screen.on_screen)
                     if isinstance(e, state.DataEntry) and e.path == good)
        screen.query_one(RosterTable).move_cursor(row=index)
        await pilot.press('enter')
        assert isinstance(app.screen, LedgerScreen) and app.screen.entry.path == good
        assert app._exception is None


@pilot_test
async def test_refused_sample_folder_retains_its_reason_without_repeated_inspection(
        live_drop, monkeypatch):
    import anndata
    import numpy as np
    from manyruns import shell
    from manyruns.tui import samplefetch
    from manyruns.pipeline import loading

    assert live_drop.is_dir()
    folder = live_drop / 'samples'
    for name in ('s1', 's2'):
        sample = folder / name
        sample.mkdir(parents=True)
        anndata.AnnData(np.ones((3, 2))).write_h5ad(sample / 'cells.h5ad')
    monkeypatch.setattr(loading, 'load_array', lambda *a, **k: pytest.fail('loaded matrix payload'))
    original = shell._resolve_path_source
    reads = []

    def tracked(path, console):
        if path == folder:
            reads.append(path)
        return original(path, console)

    monkeypatch.setattr(shell, '_resolve_path_source', tracked)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        row = next(e for e in screen.entries if e.path == folder)
        reason = '\n'.join(row.refusal)
        assert 'unsupported data folder' in reason and 'no recognized files' in reason
        assert '[red]' not in reason
        for _ in range(3):
            screen.poll()
        assert len(reads) == 1
        screen.query_one(RosterTable).move_cursor(row=screen.on_screen.index(row))
        await pilot.press('enter')
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert reason in app.screen.query_one('#sample-details', Static).visual.plain
        assert len(reads) == 1
        await pilot.press('escape', 'ctrl+r')
        assert len(reads) == 2


@pilot_test
async def test_unreadable_drops_keep_selectable_refusals_and_a_short_footnote(
        live_drop, monkeypatch):
    from manyruns.tui import samplefetch

    assert live_drop.is_dir()
    paths = [live_drop / 'unfinished.h5ad', live_drop / 'pointer.h5ad']
    paths[0].write_bytes(b'partial HDF5 download')
    paths[1].write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n')
    app = ManyrunsApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        await pilot.pause()
        rows = [e for e in screen.entries if e.path in paths]
        assert len(rows) == 2
        assert screen.load_error == '2 files could not be read'
        assert screen.query_one(RosterTable).content_size.height >= 10
        assert 'find data…' in _names(app)
        for row in rows:
            reason = '\n'.join(row.refusal)
            assert reason.count('Could not read') == 1
            assert reason.count(str(row.path)) == 1
            assert row.size == 'unavailable'
            index = screen.on_screen.index(row)
            assert screen.selectable(index)
            screen.query_one(RosterTable).move_cursor(row=index)
            with monkeypatch.context() as patch:
                patch.setattr(state, '_check_readable',
                              lambda *a: pytest.fail('selection retried an unchanged refusal'))
                await pilot.press('enter')
                assert isinstance(app.screen, samplefetch.SampleFetchScreen)
                assert reason in app.screen.query_one('#sample-details', Static).visual.plain
                assert not app.screen.query('#fetch-sample')
                await pilot.press('escape')


@pytest.mark.parametrize('damage', [
    'empty_h5ad', 'random_h5ad', 'random_csv', 'header_csv', 'text_csv',
    'ragged_tsv', 'object_npy', 'mixed_obs_h5ad', 'one_dimensional_h5ad', 'corrupt_sample',
])
@pilot_test
async def test_bad_drop_is_reported_without_disabling_other_rows(live_drop, monkeypatch, damage):
    import anndata
    import h5py
    import numpy as np
    from manyruns import datasetfetch
    from manyruns.tui.ledger import LedgerScreen

    assert live_drop.is_dir()
    good = live_drop / 'good.csv'
    good.write_text('a,b\n1,2\n3,4\n')
    suffix = damage.rsplit('_', 1)[-1]
    bad = live_drop / f'x.{suffix}'
    if damage == 'empty_h5ad':
        bad.touch()
    elif damage in {'random_h5ad', 'random_csv'}:
        bad.write_bytes(bytes(range(256)))
    elif damage == 'header_csv':
        bad.write_text('a,b\n')
    elif damage == 'text_csv':
        bad.write_text('a,b\nred,green\nblue,yellow\n')
    elif damage == 'ragged_tsv':
        bad.write_text('a\tb\n1\t2\n3\t4\t5\n')
    elif damage == 'object_npy':
        np.save(bad, np.array([{'a': 1}], dtype=object))
    elif damage in {'mixed_obs_h5ad', 'one_dimensional_h5ad'}:
        data = anndata.AnnData(np.ones((3, 2)))
        data.obs['condition'] = ['a', 'b', 'c']
        data.write_h5ad(bad)
        # AnnData's writer rejects these structures. Construct the malformed on-disk
        # encodings directly, then exercise the real reader rather than a reader stub.
        with h5py.File(bad, 'r+') as handle:
            if damage == 'one_dimensional_h5ad':
                del handle['X']
                node = handle.create_dataset('X', data=np.ones(3))
            else:
                del handle['obs/condition']
                mixed = np.dtype([('number', 'i4'), ('text', 'S1')])
                node = handle['obs'].create_dataset('condition', (3,), dtype=h5py.vlen_dtype(mixed))
                for i in range(3):
                    node[i] = np.array([(i, b'a')] * (i + 1), dtype=mixed)
            node.attrs['encoding-type'] = 'array'
            node.attrs['encoding-version'] = '0.2.0'
        if damage == 'mixed_obs_h5ad':
            assert anndata.read_h5ad(bad).obs['condition'].dtype == object
    else:
        assert damage == 'corrupt_sample'
        bad = live_drop / 'cohort'
        for name in ('sample1', 'sample2'):
            folder = bad / name
            folder.mkdir(parents=True)
            assert folder.is_dir()
        assert bad.is_dir()
        # Give the suffix-based modality inspection a recognized root file; the shared
        # loader must still visit both per-sample subfolders, including the corrupt one.
        (bad / 'metadata.csv').write_text('a,b\n1,2\n')
        anndata.AnnData(np.ones((3, 2))).write_h5ad(bad / 'sample1' / 'cells.h5ad')
        (bad / 'sample2' / 'cells.h5ad').write_bytes(b'corrupt sample')

    def no_download(*args, **kwargs):
        raise AssertionError('inspecting drops must not download')

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', no_download)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        for _ in range(3):
            screen.call_later(screen.poll, force=True)
            await pilot.pause()
            assert app._exception is None and app.screen is screen
            assert any(e.path == good and not e.pending and not e.refusal for e in screen.entries)
            assert any(e.name == 'swissroll' and not e.refusal for e in screen.entries)
            bad_rows = [e for e in screen.entries if e.path == bad]
            assert bad_rows and all(e.refusal for e in bad_rows)
            assert all('could not read' in ' '.join(e.refusal).lower() for e in bad_rows)
        # The finder's read-back and subsequent roster refresh reach the same boundary.
        from manyruns.tui import search
        assert search.roster_entry(bad, live_drop).refusal
        screen.call_later(app.refresh_roster)
        await pilot.pause()
        assert app.screen is screen and app._exception is None
        index = next(i for i, e in enumerate(screen.on_screen)
                     if isinstance(e, state.DataEntry) and e.path == good)
        screen.query_one(RosterTable).move_cursor(row=index)
        await pilot.press('enter')
        assert isinstance(app.screen, LedgerScreen) and app.screen.entry.path == good
        await pilot.press('escape')
        index = next(i for i, e in enumerate(screen.on_screen)
                     if isinstance(e, state.DataEntry) and e.name == 'swissroll')
        screen.query_one(RosterTable).move_cursor(row=index)
        await pilot.press('enter')
        assert isinstance(app.screen, LedgerScreen) and app.screen.entry.name == 'swissroll'
        assert app._exception is None


@pilot_test
async def test_launch_selects_initial_drop_after_settling_with_catalog_visible(
        live_drop, tmp_path, monkeypatch):
    from manyruns import datasetfetch
    from manyruns.tui.ledger import LedgerScreen

    assert live_drop.is_dir()
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv('MANYRUNS_DATASET_DIR', raising=False)
    calls = []

    def no_download(*args, **kwargs):
        calls.append(args)
        raise AssertionError('launching own data must not download')

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', no_download)
    path = live_drop / 'mine.csv'
    path.write_text('a,b\n1,2\n3,4\n')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        assert any(e.name == 'pbmc3k' for e in screen.entries)
        assert not any(e.path == path for e in screen.entries)
        screen.poll()
        assert _under_cursor(app) == 'mine.csv'
        await pilot.press('enter')
        assert isinstance(app.screen, LedgerScreen)
        assert app.screen.entry.path == path
        assert not calls


@pytest.mark.parametrize('key', ['up', 'down', 'click'])
@pilot_test
async def test_user_selection_during_launch_survives_initial_drop_settling(
        live_drop, tmp_path, monkeypatch, key):
    assert live_drop.is_dir()
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.delenv('MANYRUNS_DATASET_DIR', raising=False)
    path = live_drop / 'mine.csv'
    path.write_text('a,b\n1,2\n')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        assert not any(e.path == path for e in screen.entries)
        if key == 'click':
            await pilot.click(RosterTable, offset=(3, 2))
        else:
            await pilot.press(key)
        selected = _under_cursor(app)
        screen.poll()
        assert any(e.path == path for e in screen.entries)
        assert _under_cursor(app) == selected


@pilot_test
@pytest.mark.parametrize('failure', ['oserror', 'none'])
@pytest.mark.parametrize('retry', ['change', 'rescan'])
async def test_refused_drop_retries_only_on_change_or_rescan(
        live_drop, tmp_path, monkeypatch, failure, retry):
    import errno
    from manyruns import shell

    assert live_drop.is_dir()
    monkeypatch.setenv('HOME', str(tmp_path))
    original = shell._resolve_path_source
    path = live_drop / 'mine.csv'
    calls = []

    def read_once_unavailable(source, *args, **kwargs):
        if source == path:
            calls.append(source)
            if len(calls) == 1:
                if failure == 'oserror':
                    raise OSError(errno.EIO, 'temporarily unreadable')
                return None
        return original(source, *args, **kwargs)

    monkeypatch.setattr(shell, '_resolve_path_source', read_once_unavailable)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        path.write_text('a,b\n1,2\n3,4\n')
        screen.poll()
        assert not calls  # first observation is held
        screen.poll()
        assert len(calls) == 1
        assert not any(e.path == path and not e.refusal for e in screen.entries)
        failure_text = screen.query_one('#footnote', Static).visual.plain
        for i in range(3):
            # Even unrelated changes must not make the refused source read again.
            (live_drop / 'neighbour.csv').write_text('a,b\n' + '1,2\n' * (i + 1))
            screen.poll()
            assert len(calls) == 1
        if retry == 'change':
            path.write_text('a,b\n1,2\n3,4\n5,6\n')
            screen.poll()
            assert len(calls) == 1
            screen.poll()
        else:
            await pilot.press('ctrl+r')
        assert len(calls) == 2
        assert any(e.path == path for e in screen.entries)
        assert '1 files could not be read' in failure_text
        assert 'retrying' not in screen.query_one('#footnote', Static).visual.plain
        assert screen.load_error is None and app._exception is None


@pilot_test
async def test_large_drop_reuses_unchanged_entries_while_a_neighbour_grows(
        live_drop, tmp_path, monkeypatch):
    from manyruns import inspected, shell

    assert live_drop.is_dir()
    monkeypatch.setenv('HOME', str(tmp_path))
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    paths = [live_drop / f'sample-{i:04}.csv' for i in range(inspected.MAX_ROWS + 1)]
    for path in paths:
        path.write_text('a,b\n1,2\n')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        warm = {e.path: e for e in screen.entries}
        assert len(warm) == len(paths) > inspected.MAX_ROWS
        reads, cache_gets, cache_puts, qcs = [], [], [], []

        def track(owner, name, calls):
            original = getattr(owner, name)

            def counted(path, *args, **kwargs):
                calls.append(path)
                return original(path, *args, **kwargs)

            monkeypatch.setattr(owner, name, counted)

        track(shell, '_resolve_path_source', reads)
        track(inspected, 'get', cache_gets)
        track(inspected, 'put', cache_puts)
        track(state, '_qc', qcs)
        growing = live_drop / 'growing.csv'
        for content in ('a,b\n1,2\n', 'a,b\n1,2\n3,4\n'):
            growing.write_text(content)
            screen.poll()
            assert not reads and not cache_gets and not cache_puts and not qcs
            assert all(e is warm[e.path] for e in screen.entries)
        screen.poll()
        assert reads == cache_gets == cache_puts == qcs == [growing]
        fresh = next(e for e in screen.entries if e.path == growing)
        assert fresh.nbytes == growing.stat().st_size
        screen.poll(force=True)
        assert reads == cache_gets == cache_puts == qcs == [growing]
        assert next(e for e in screen.entries if e.path == growing) is fresh

        # Changed and deleted sources must still lose their old entries immediately.
        paths[0].write_text('a,b\n1,2\n3,4\n5,6\n')
        paths[1].unlink()
        screen.poll()
        assert not any(e.path in paths[:2] for e in screen.entries)
        screen.poll()
        changed = next(e for e in screen.entries if e.path == paths[0])
        assert changed is not warm[paths[0]]
        assert changed.nbytes == paths[0].stat().st_size != warm[paths[0]].nbytes
        assert reads == cache_gets == cache_puts == qcs == [growing, paths[0]]


@pilot_test
async def test_live_drop_settles_without_keys_and_preserves_filter_and_cursor(live_drop):
    assert live_drop.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.query_one('#filter', Input).value = 'swissroll'
        await pilot.pause()
        selected = screen.on_screen[screen.query_one(RosterTable).cursor_row].identity
        path = live_drop / 'swissroll.csv'
        path.write_text('a,b\n1,2\n')
        screen.poll()
        assert not any(e.path == path for e in screen.entries)
        screen.poll()
        assert any(e.path == path for e in screen.entries)
        assert screen.query_one('#filter', Input).value == 'swissroll'
        assert screen.on_screen[screen.query_one(RosterTable).cursor_row].identity == selected
        assert 'new data' in screen.query_one('#footnote', Static).visual.plain
        assert len(app.screen_stack) == 1


@pilot_test
async def test_preloaded_roster_never_reads_or_polls(monkeypatch):
    from manyruns import shell
    from manyruns.tui import dropwatch

    def forbidden(*args, **kwargs):
        pytest.fail('preloaded screen touched disk')

    monkeypatch.setattr(shell, 'ensure_drop_folder', forbidden)
    monkeypatch.setattr(shell, 'state_rows', forbidden)
    monkeypatch.setattr(dropwatch, 'snapshot', forbidden)
    monkeypatch.setattr(state, 'roster', forbidden)
    app = RosterApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.load()
        app.screen.poll()
        assert app.screen._poll_timer is None


@pilot_test
async def test_covered_roster_defers_inspection_until_resume(live_drop, monkeypatch):
    from textual.screen import Screen
    assert live_drop.is_dir()
    reads = []
    real = state.roster
    monkeypatch.setattr(state, 'roster', lambda **kw: reads.append(kw) or real(**kw))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await app.push_screen(Screen())
        before = len(reads)
        (live_drop / 'new.csv').write_text('a,b\n1,2\n')
        roster.poll()
        roster.poll()
        assert len(reads) == before
        app.pop_screen()
        await pilot.pause()
        assert len(reads) > before
        assert any(e.name == 'new.csv' for e in roster.entries)


@pilot_test
async def test_folder_creation_failure_recovers_and_ctrl_r_uses_settle_rule(
        live_drop, monkeypatch):
    from manyruns import shell
    assert live_drop.is_dir()
    real = shell.ensure_drop_folder
    monkeypatch.setattr(shell, 'ensure_drop_folder',
                        lambda: (_ for _ in ()).throw(PermissionError('cannot create drop')))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        assert 'cannot create drop' in screen.query_one('#footnote', Static).visual.plain
        monkeypatch.setattr(shell, 'ensure_drop_folder', real)
        (live_drop / 'new.csv').write_text('a,b\n1,2\n')
        await pilot.press('ctrl+r')
        assert not any(e.name == 'new.csv' for e in screen.entries)
        await pilot.press('ctrl+r')
        assert any(e.name == 'new.csv' for e in screen.entries)
        assert screen.load_error is None
        assert 'cannot create drop' not in screen.query_one('#footnote', Static).visual.plain
        assert 'ctrl+r' in str(screen.query_one('#roster-body').border_subtitle)


@pilot_test
async def test_creation_error_clears_even_when_the_snapshot_does_not_change(live_drop, monkeypatch):
    from manyruns import shell
    assert live_drop.is_dir()
    real = shell.ensure_drop_folder
    monkeypatch.setattr(shell, 'ensure_drop_folder',
                        lambda: (_ for _ in ()).throw(PermissionError('folder unavailable')))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        assert 'folder unavailable' in screen.query_one('#footnote', Static).visual.plain
        monkeypatch.setattr(shell, 'ensure_drop_folder', real)
        screen.poll()
        assert 'folder unavailable' not in screen.query_one('#footnote', Static).visual.plain


@pilot_test
async def test_same_name_kinds_keep_identity_and_removal_chooses_a_surviving_row(live_drop):
    assert live_drop.is_dir()
    folder = live_drop / 'swissroll'
    folder.mkdir()
    (folder / 'matrix.csv').write_text('a,b\n1,2\n')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        screen.query_one('#filter', Input).value = 'swissroll'
        await pilot.pause()
        choices = [e for e in screen.entries if e.name == 'swissroll']
        assert {e.kind for e in choices} == {'folder', 'bundled'}
        selected = next(e for e in choices if e.kind == 'folder')
        table = screen.query_one(RosterTable)
        table.move_cursor(row=next(i for i, e in enumerate(screen.on_screen) if e is selected))
        screen.refresh_rows()
        assert screen.on_screen[table.cursor_row].identity == selected.identity
        (folder / 'matrix.csv').unlink()
        folder.rmdir()
        screen.poll()
        assert screen.on_screen[table.cursor_row].identity == ('bundled', 'swissroll')
        assert screen.query_one('#filter', Input).value == 'swissroll'


@pilot_test
async def test_each_tick_uses_one_snapshot_and_inspection_race_resettles(live_drop, monkeypatch):
    from manyruns import shell
    from manyruns.tui import dropwatch
    assert live_drop.is_dir()
    snapshots = []
    snapshot = dropwatch.snapshot
    monkeypatch.setattr(dropwatch, 'snapshot', lambda: snapshots.append(1) or snapshot())
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        path = live_drop / 'moving.csv'
        path.write_text('a,b\n1,2\n')
        before = len(snapshots)
        screen.poll()
        assert len(snapshots) == before + 1
        resolve = shell._resolve_path_source

        def moving(p, console):
            value = resolve(p, console)
            path.write_text('a,b\n1,2\n3,4\n')
            return value

        monkeypatch.setattr(shell, '_resolve_path_source', moving)
        screen.poll()
        assert not any(e.path == path for e in screen.entries)
        monkeypatch.setattr(shell, '_resolve_path_source', resolve)
        screen.poll()
        assert not any(e.path == path for e in screen.entries)
        screen.poll()
        assert any(e.path == path for e in screen.entries)


@pilot_test
@pytest.mark.parametrize('change', [
    'symlink_cycle', 'symlink_target_deleted', 'file_deleted', 'drop_folder_removed',
    'drop_folder_replaced', 'permission_revoked', 'cycle_after_snapshot', 'snapshot_error',
])
async def test_live_roster_survives_invalidated_sources(live_drop, tmp_path, monkeypatch, change):
    """Cycle cases regress stale identity reads; deletion cases preserve graceful removal."""
    from manyruns import catalog
    from manyruns.tui import dropwatch

    assert live_drop.is_dir()
    # Keep a generated survivor without inspecting any developer catalog paths.
    declaration = catalog.load_dataset('swissroll')
    monkeypatch.setattr(catalog, 'discover_datasets', lambda: ['swissroll'])
    monkeypatch.setattr(catalog, 'load_dataset', lambda name: declaration)
    path = live_drop / 'moving.csv'
    target = tmp_path / 'target.csv'
    target.write_text('a,b\n1,2\n')
    if change in {'symlink_cycle', 'symlink_target_deleted', 'cycle_after_snapshot'}:
        path.symlink_to(target)
    else:
        path.write_text('a,b\n1,2\n')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        selected = next(e for e in screen.entries if e.path == path)
        table = screen.query_one(RosterTable)
        table.move_cursor(row=screen.on_screen.index(selected))
        snapshot = dropwatch.snapshot

        def make_cycle():
            path.unlink()
            path.symlink_to(path)

        if change == 'cycle_after_snapshot':
            def changed_snapshot():
                result = snapshot()
                make_cycle()
                return result

            monkeypatch.setattr(dropwatch, 'snapshot', changed_snapshot)
        elif change == 'snapshot_error':
            def unavailable_snapshot():
                raise FileNotFoundError('drop folder vanished during polling')

            monkeypatch.setattr(dropwatch, 'snapshot', unavailable_snapshot)
        elif change == 'symlink_cycle':
            make_cycle()
        elif change == 'symlink_target_deleted':
            target.unlink()
        elif change == 'permission_revoked':
            stat = Path.stat

            def denied(source, *args, **kwargs):
                if source == path:
                    raise PermissionError('source permission revoked')
                return stat(source, *args, **kwargs)

            monkeypatch.setattr(Path, 'stat', denied)
        else:
            path.unlink()
            if change in {'drop_folder_removed', 'drop_folder_replaced'}:
                live_drop.rmdir()
                if change == 'drop_folder_replaced':
                    live_drop.write_text('no longer a directory')

        # Dispatch through Textual, so escaping poll exceptions reach app._exception.
        screen.call_later(screen.poll, force=True)
        await pilot.pause()
        assert app._exception is None
        assert all(e.path != path for e in screen.entries)
        assert screen.on_screen[table.cursor_row].identity == ('bundled', 'swissroll')
        await pilot.press('s')
        assert screen.query_one('#filter', Input).value == 's'
        assert app.is_running and app._exception is None
        if change == 'snapshot_error':
            assert 'drop folder vanished' in screen.load_error
            monkeypatch.setattr(dropwatch, 'snapshot', snapshot)
            screen.poll()
            assert all(e.path != path for e in screen.entries)
            screen.poll()
            assert any(e.path == path for e in screen.entries)
            assert screen.load_error is None and app._exception is None


@pilot_test
async def test_queued_invalidation_survives_a_later_symlink_cycle(live_drop, monkeypatch):
    from dataclasses import replace
    from manyruns import catalog, inspected, shell

    assert live_drop.is_dir()
    monkeypatch.setattr(catalog, 'discover_datasets', lambda: [])
    first, second = live_drop / 'a.csv', live_drop / 'b.csv'
    for path in (first, second):
        path.write_text('a,b\n1,2\n')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        assert {e.path for e in screen.entries} == {first, second}
        resolve = shell._resolve_path_source
        queued = []
        read_roster = state.roster

        def tracked_roster(*args, on_changed, **kwargs):
            def changed(key):
                queued.append(key)
                on_changed(key)
            return read_roster(*args, on_changed=changed, **kwargs)

        def racing(path, console):
            result = resolve(path, console)
            if path == first:
                first.write_text('a,b\n1,2\n3,4\n')
            elif path == second:
                assert queued == [first]
                first.unlink()
                first.symlink_to(first)
            return result

        monkeypatch.setattr(inspected, 'get', lambda path: None)
        # This fault needs inspection, so discard both live and disk cache evidence.
        screen.entries = [replace(e, observation=None) for e in screen.entries]
        monkeypatch.setattr(state, 'roster', tracked_roster)
        monkeypatch.setattr(shell, '_resolve_path_source', racing)
        screen.call_later(screen.poll, force=True)
        await pilot.pause()
        assert queued == [first]
        assert app._exception is None
        assert first not in screen._watch._previous
        assert [e.path for e in screen.entries] == [second]
        await pilot.press('b')
        assert screen.query_one('#filter', Input).value == 'b'


ROSTER_FS_PRIMITIVES = [
    'os.stat', 'os.lstat', 'os.scandir', 'os.listdir', 'os.readlink', 'os.mkdir',
    'io.open', 'os.replace',
    'Path.resolve', 'Path.exists', 'Path.is_file', 'Path.is_dir',
    'Path.resolve:ValueError',
]


async def _roster_callback_fault(live_drop, tmp_path, monkeypatch, entry_point, primitive):
    from dataclasses import replace
    import io
    import os
    import yaml
    from manyruns import catalog, inspected

    assert live_drop.is_dir()
    target = tmp_path / 'target.csv'
    target.write_text('a,b\n1,2\n')
    (live_drop / 'a.csv').symlink_to(target)
    cohort = live_drop / 'cohort'
    cohort.mkdir()
    assert cohort.is_dir()
    (cohort / 'nested.csv').write_text('a,b\n1,2\n')
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    ds = catalog.load_dataset('swissroll')
    ds['handle'] = {'kind': 'path', 'ref': str(live_drop / 'a.csv')}
    (configs / 'alias.yaml').write_text(yaml.safe_dump(ds))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        # Force a complete refresh, including inspection and the footnote, on this tick.
        screen._loaded_poll = None
        screen.entries = [replace(e, observation=None) for e in screen.entries]
        monkeypatch.setattr(inspected, 'get', lambda path: None)
        if entry_point == '_start_polling':
            screen._started = False
        namespace, name = primitive.split(':')[0].split('.')
        owner = {'os': os, 'io': io, 'Path': Path}[namespace]
        original = getattr(owner, name)
        hits = []

        def fail_once(*args, **kwargs):
            if not hits:
                hits.append(args)
                error = ValueError if primitive.endswith(':ValueError') else OSError
                raise error(f'injected {primitive}')
            return original(*args, **kwargs)

        def tick():
            with monkeypatch.context() as patch:
                patch.setattr(owner, name, fail_once)
                getattr(screen, entry_point)()

        screen.call_later(tick)
        await pilot.pause()
        assert hits, f'{primitive} was not reached from {entry_point}'
        assert app._exception is None
        screen._poll_timer.pause()
        await pilot.press('a')
        assert screen.query_one('#filter', Input).value == 'a'
        # Transient faults cannot strand the watcher; two good ticks recover the row.
        screen.poll(force=True)
        screen.poll(force=True)
        assert any(e.path == live_drop / 'a.csv' for e in screen.entries)
        assert app.is_running and app._exception is None


@pytest.mark.parametrize('primitive', ROSTER_FS_PRIMITIVES)
@pilot_test
async def test_poll_filesystem_faults(live_drop, tmp_path, monkeypatch, primitive):
    await _roster_callback_fault(live_drop, tmp_path, monkeypatch, 'poll', primitive)


@pytest.mark.parametrize('primitive', ROSTER_FS_PRIMITIVES)
@pilot_test
async def test_resume_filesystem_faults(live_drop, tmp_path, monkeypatch, primitive):
    await _roster_callback_fault(live_drop, tmp_path, monkeypatch, 'on_screen_resume', primitive)


@pytest.mark.parametrize('primitive', ROSTER_FS_PRIMITIVES)
@pilot_test
async def test_rescan_filesystem_faults(live_drop, tmp_path, monkeypatch, primitive):
    await _roster_callback_fault(live_drop, tmp_path, monkeypatch, 'action_rescan', primitive)


@pytest.mark.parametrize('primitive', ROSTER_FS_PRIMITIVES)
@pilot_test
async def test_load_filesystem_faults(live_drop, tmp_path, monkeypatch, primitive):
    await _roster_callback_fault(live_drop, tmp_path, monkeypatch, 'load', primitive)


@pytest.mark.parametrize('primitive', ROSTER_FS_PRIMITIVES)
@pilot_test
async def test_start_polling_filesystem_faults(live_drop, tmp_path, monkeypatch, primitive):
    await _roster_callback_fault(live_drop, tmp_path, monkeypatch, '_start_polling', primitive)


@pytest.mark.parametrize('primitive', ['Path.resolve', 'Path.exists', 'Path.is_dir'])
@pilot_test
async def test_poll_late_refresh_filesystem_faults(live_drop, monkeypatch, primitive):
    """The final refresh is inside the guard, and stale entries lose their evidence."""
    assert live_drop.is_dir()
    path = live_drop / 'a.csv'
    path.write_text('a,b\n1,2\n')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        screen.poll()
        refresh = screen.refresh_rows
        hits = []

        def refresh_with_fault():
            if not hits:
                hits.append(1)
                with monkeypatch.context() as patch:
                    def failed(*args, **kwargs):
                        raise OSError('source vanished during refresh')
                    patch.setattr(Path, primitive.split('.')[1], failed)
                    getattr(path, primitive.split('.')[1])()
            refresh()

        monkeypatch.setattr(screen, 'refresh_rows', refresh_with_fault)
        screen.call_later(screen.poll, force=True)
        await pilot.pause()
        assert hits and app._exception is None
        assert not screen._watch._previous
        assert all(e.pending and e.observation is None for e in screen.entries if e.path == path)
        await pilot.press('a')
        assert screen.query_one('#filter', Input).value == 'a'


@pytest.mark.parametrize('error', [TypeError, AttributeError, KeyError])
@pilot_test
async def test_poll_does_not_swallow_programming_errors(live_drop, monkeypatch, error):
    assert live_drop.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()

        def broken(*args, **kwargs):
            raise error('programming defect')

        with monkeypatch.context() as patch:
            patch.setattr(state, 'roster', broken)
            with pytest.raises(error, match='programming defect'):
                screen.poll(force=True)


@pytest.mark.parametrize('source', ['catalog', 'drop_folder'])
@pilot_test
async def test_unknown_user_path_keeps_roster_available_with_reason(
        live_drop, tmp_path, monkeypatch, source):
    import pwd
    import yaml
    from manyruns import catalog

    def absent(name):
        raise KeyError(name)

    monkeypatch.setattr(pwd, 'getpwnam', absent)
    assert live_drop.is_dir()
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    (configs / 'swissroll.yaml').write_text(yaml.safe_dump(catalog.load_dataset('swissroll')))
    if source == 'catalog':
        (configs / 'unknown_home.yaml').write_text(yaml.safe_dump({
            'name': 'unknown_home', 'shape': 'single', 'modality': 'scrna',
            'handle': {'kind': 'path', 'ref': '~missing_test_user/cells.h5ad'},
        }))
    else:
        monkeypatch.setenv('MANYRUNS_DATA_DIR', '~missing_test_user/data')
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        assert app._exception is None and app.is_running
        assert isinstance(screen, RosterScreen) and screen._poll_timer is not None
        screen._poll_timer.pause()
        screen.poll()
        assert 'Could not determine home directory' in screen.query_one('#footnote', Static).visual.plain
        assert any(e.name == 'swissroll' and not e.refusal for e in screen.entries)
        assert 'find data…' in _names(app)
        await pilot.press('s', 'ctrl+r')
        assert screen.query_one('#filter', Input).value == 's'
        assert app._exception is None


@pytest.fixture
def malformed_catalog(live_drop, tmp_path, monkeypatch):
    import yaml
    from manyruns import catalog

    assert live_drop.is_dir()
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    valid = ['swissroll', 'tree_wide']
    for name in valid:
        (configs / f'{name}.yaml').write_text(yaml.safe_dump(catalog.load_dataset(name)))
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))

    def write_invalid(kind='top_level_list'):
        name = f'broken_{kind}'
        declaration = {
            'name': name, 'shape': 'single', 'modality': 'scrna',
            'handle': {'kind': 'path', 'ref': 'absent.csv'},
        }
        if kind == 'top_level_list':
            declaration = [declaration]
        elif kind == 'top_level_scalar':
            declaration = 'bare string'
        elif kind == 'source_scalar':
            declaration['source'] = 'https://example.org/data'
        elif kind == 'handle_scalar':
            declaration['handle'] = 'absent.csv'
        elif kind == 'provides_scalar':
            declaration['handle']['provides'] = 42
        elif kind == 'shape_number':
            declaration['shape'] = 42
        elif kind == 'topology_mapping':
            declaration['topology'] = {'branching': True}
        elif kind == 'missing_name':
            del declaration['name']
        elif kind == 'ref_list':
            declaration['handle']['ref'] = ['absent.csv']
        else:
            raise AssertionError(f'unknown malformed declaration: {kind}')
        (configs / f'{name}.yaml').write_text(yaml.safe_dump(declaration))
        return name

    return valid, write_invalid


@pytest.mark.parametrize('reader', ['roster', 'snapshot'])
def test_malformed_catalog_preserves_valid_rows_and_observations(
        malformed_catalog, live_drop, reader):
    from manyruns.tui import dropwatch

    valid, write_invalid = malformed_catalog
    write_invalid()
    path = live_drop / 'mine.csv'
    path.write_text('a,b\n1,2\n')
    assert live_drop.is_dir()
    if reader == 'roster':
        assert {e.name for e in state.roster()} == {*valid, path.name}
    else:
        assert dropwatch.snapshot().drops == (path,)


@pytest.mark.parametrize('malformation', ['top_level_list', 'source_scalar', 'ref_list'])
def test_malformed_catalog_entry_reports_a_read_error_to_selection(
        malformed_catalog, malformation):
    _, write_invalid = malformed_catalog
    name = write_invalid(malformation)
    with pytest.raises(ValueError, match=name):
        state.catalog_entry(name)


@pytest.mark.parametrize('arrival', ['launch', 'poll'])
@pytest.mark.parametrize('malformation', [
    'top_level_list', 'top_level_scalar', 'source_scalar', 'handle_scalar',
    'provides_scalar', 'shape_number', 'topology_mapping', 'missing_name', 'ref_list',
])
@pilot_test
async def test_malformed_catalog_keeps_valid_rows_selectable(
        malformed_catalog, arrival, malformation):
    from manyruns.tui.ledger import LedgerScreen

    valid, write_invalid = malformed_catalog
    if arrival == 'launch':
        bad_name = write_invalid(malformation)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        assert [e.name for e in screen.entries] == valid
        if arrival == 'poll':
            bad_name = write_invalid(malformation)
        for _ in range(3):
            screen.call_later(screen.poll, force=True)
            await pilot.pause()
            assert [e.name for e in screen.entries] == valid
            assert screen.load_error is None and app._exception is None
            footnote_text = screen.query_one('#footnote', Static).visual.plain
            assert 'Skipped declaration' in footnote_text and bad_name in footnote_text
        await pilot.press('enter')
        assert isinstance(app.screen, LedgerScreen)
        assert app.screen.entry.name == valid[0]
        assert app._exception is None


@pilot_test
@pytest.mark.parametrize('refresh', ['ctrl+r', 'load'])
async def test_explicit_rescan_reloads_repaired_catalog_without_drop_changes(
        live_drop, tmp_path, monkeypatch, refresh):
    import yaml
    from manyruns import catalog
    assert live_drop.is_dir()
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    valid = catalog.load_dataset('swissroll')
    declaration = configs / 'swissroll.yaml'
    declaration.write_text('handle: [broken')
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen._poll_timer.pause()
        assert not screen.entries and screen.load_error is None
        assert 'swissroll' in screen.query_one('#footnote', Static).visual.plain
        declaration.write_text(yaml.safe_dump(valid))
        # Background stat polling still avoids expensive unchanged-data reads.
        screen.poll()
        assert not screen.entries
        if refresh == 'ctrl+r':
            await pilot.press(refresh)
        else:
            screen.load()
        assert [e.name for e in screen.entries] == ['swissroll']
        assert 'Skipped declaration' not in screen.query_one('#footnote', Static).visual.plain
