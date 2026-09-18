"""The ledger screen (`manyruns.tui.ledger`) — component 3 of the TUI rewrite.

Driven through `App.run_test()`, which §5 of the spec picks as the replacement for the injected
prompter: real key presses, real layout, real widget state, no terminal. What is asserted is
what a scientist would SEE — the composited screen text — not a data structure the screen was
handed; `tests/test_tui_state.py` already covers the data structure, and a screen test that
re-asserted it would prove nothing about the screen.

The two properties worth the most here, both of them §0's third defect:

    a refused analysis is ON SCREEN with the reason it is refused
    and it still cannot be chosen, even when the highlight is forced onto it

`textual` is not declared in `pyproject.toml` (§8 measured it installed in `.venv`; nothing
added the dependency line), so this module skips rather than errors where it is absent —
`manyruns.tui.__init__` imports no Textual and the other two surfaces do not depend on it.
STATED LOUDLY because a skip is how a test file stops guarding anything: if CI has no
`textual`, this file is silent, and the fix is the dependency line, not a weaker test.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import sys
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402
from textual.widgets import OptionList  # noqa: E402

from manyruns import narrate, vocab  # noqa: E402
from manyruns.catalog import discover_recipes, load_recipe  # noqa: E402
from manyruns.narrate import Observation  # noqa: E402
from manyruns.tui import state  # noqa: E402

# `from manyruns.tui import ledger` does NOT give you this module — it gives you
# `state.ledger`, the function `manyruns/tui/__init__.py` re-exports under the same name. The
# `from <module> import <name>` form below resolves through `sys.modules` and is unambiguous;
# `test_the_screen_module_shadows_the_state_function_of_the_same_name` pins the hazard.
from manyruns.tui.ledger import (  # noqa: E402
    BLOCKED_MARK,
    RECOMMENDED_MARK,
    RUNNABLE_MARK,
    LedgerScreen,
    is_group_id,
)
from manyruns.tui.state import DataEntry, LedgerRow  # noqa: E402

LEDGER_MODULE = sys.modules[LedgerScreen.__module__]
FILLED, HOLLOW = RUNNABLE_MARK, BLOCKED_MARK
RECOMMENDED = RECOMMENDED_MARK


@pytest.fixture(autouse=True)
def _rich_is_whole():
    """Repair a `rich` that an earlier test dismantled. NOT this component's bug, and the fix
    below is a splint, not the cure.

    ROOT CAUSE, measured: `tests/test_shell.py::test_base_import_chain_pulls_no_sdk_or_tui` does
    `sys.modules.pop("rich", None)` to prove the base import chain stays SDK-free, and never puts
    it back. That pops only the top-level entry — the ~40 `rich.*` submodules stay in
    `sys.modules` — so the next `import rich` builds a FRESH package object with none of them
    attached, and `import rich.repr` will not reattach it because `sys.modules["rich.repr"]`
    already exists and short-circuits the import. Any later `rich.repr.auto` then raises
    `AttributeError: module 'rich' has no attribute 'repr'`, which Textual uses pervasively.

    Harmless until now: nothing in the suite touched `rich` attributively after that test. It
    detonated when Textual arrived — measured at 32 failures across `test_tui_roster.py` (18,
    component 2's) and this file (14), and NONE of them alone: both files are green when run on
    their own. The real fix is three lines in `test_shell.py` (re-import what it popped, in a
    `finally`), or one autouse fixture in a `tests/conftest.py` that does not exist yet; both are
    outside this component, and this fixture leaves the suite honest in the meantime.
    """
    import rich

    package = sys.modules["rich"]
    for name, module in list(sys.modules.items()):
        head, _, tail = name.partition(".")
        if head == "rich" and tail and "." not in tail:
            if getattr(package, tail, None) is not module:
                setattr(package, tail, module)
    assert rich is package


def astest(fn):
    """Run one `async def` test to completion.

    `pytest-asyncio` is not a dependency of this repo (`dev = ["pytest", "ruff"]`) and adding
    one for a screen test would be a heavier decision than the test is worth, so the loop is
    driven here. `functools.wraps` is what keeps pytest reporting the test's own name.
    """
    @functools.wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


def _entry(shape: str = "single", name: str = "pbmc3k_raw.h5ad", **kw) -> DataEntry:
    """A roster row, built by hand rather than by `state.roster()`.

    `roster()` opens a file per dropped row (361 ms cold, measured by component 1) and its
    contents depend on whatever is in the drop folder; this screen never opens anything, so a
    constructed row keeps the test about the screen. One test below uses a REAL roster row, to
    prove the two halves connect.
    """
    obs = Observation(shape=shape, modality=kw.pop("modality", "scrna"), **kw)
    return DataEntry(name=name, kind="file", obs=obs, path=Path("/tmp") / name, nbytes=5_855_727)


class _Harness(App):
    """The smallest app that can hold the screen: it pushes it and keeps what it returns.

    `dismiss` needs the screen to be PUSHED — Textual refuses to pop the last screen off the
    stack — so this mirrors what the real app (component 2) has to do, rather than testing a
    configuration nobody will ship. `picked` stays empty until the screen returns something,
    which is how "nothing was chosen" is told apart from "None was chosen".
    """

    def __init__(self, screen: LedgerScreen) -> None:
        super().__init__()
        self._ledger = screen
        self.picked: list = []

    def on_mount(self) -> None:
        self.push_screen(self._ledger, callback=self.picked.append)


def _screen_text(app: App) -> str:
    """What is actually on the terminal, composited.

    `_compositor.render_strips()` is private to Textual and is used deliberately: the public
    alternative is `export_screenshot`, which returns SVG. If a Textual upgrade removes it this
    raises loudly instead of quietly asserting on nothing.
    """
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


def _options(app: App) -> OptionList:
    return app.screen.query_one(OptionList)


def _analysis_options(app: App) -> list:
    """The options that are ANALYSES — group headings excluded.

    The pane groups rows under the geometry they assume, and a heading is an `Option` so it
    scrolls with its own rows. It is disabled, so the keyboard never lands on it; what it is not
    is a recipe, so every assertion about "the analyses on screen" has to say so. Filtered
    through `is_group_id` rather than on `o.id is None`, so this keeps working if a heading ever
    gains different content.
    """
    return [o for o in _options(app)._options if not is_group_id(o.id)]


def _index_of(app: App, recipe: str) -> int:
    """The OPTION index of a recipe's row. Row index and option index stopped agreeing when the
    headings arrived, and a test that assumes they do is testing arithmetic, not the screen."""
    return [o.id for o in _options(app)._options].index(recipe)


def _offered() -> set:
    """Every recipe that reaches the product menu — i.e. not a `dev.` fixture.

    `sandbox` declares `id: dev.sandbox.stubs` and is filtered by `narrate.offer` on that
    prefix, so the ledger no longer carries it. Derived from the catalog rather than subtracting
    a name, so a second fixture needs no edit here.
    """
    from manyruns.catalog import load_recipe
    return {n for n in discover_recipes()
            if not str(load_recipe(n).get("id") or "").startswith("dev.")}


def _rows_text(app: App) -> str:
    """Only the rows, without the frame or the legend.

    The legend under the list carries a `○` of its own, so counting marks over the whole screen
    counts the explanation as if it were an analysis. This renders the `OptionList` alone.
    """
    options = _options(app)
    return "\n".join(options.render_line(y).text for y in range(options.size.height))


def _executable_source(module) -> str:
    """The module with every docstring and comment removed — what it DOES, not what it says.

    `ast.unparse` drops comments outright; the docstrings are stripped node by node first. A
    guard over the raw text would trip over prose that names `vocab.unmet` or `contrast` while
    explaining why the screen goes nowhere near either.
    """
    import ast

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                node.body = node.body[1:]
    return ast.unparse(tree)


def _unmet(obs: Observation) -> dict:
    """The verdict the screen must agree with, recomputed for every bundled recipe — derived
    rather than transcribed.

    TWO SOURCES, and that is the point rather than an inconvenience. `vocab.unmet` asks whether
    THIS DATA carries the facts a recipe's steps need; `narrate.unavailable` asks whether those
    steps can run at all right now, which is a different question with no data in it — measured,
    `manylatents`' `aa` raises `TypeError: ... unexpected keyword argument 'method'`, and the
    ledger offered `archetypes` as the first analysis for `swissroll` regardless. A row is
    hollow if either says so; neither is folded into the other, because `blocked(...) == []` iff
    `unmet(...) == frozenset()` is an invariant with its own test.
    """
    # OVER THE OFFERED RECIPES ONLY. `dev.` fixtures never reach the pane, so counting the
    # screen's marks against a verdict that includes them compares two different populations —
    # which reads as the screen having lost a row rather than as the fixture never being drawn.
    return {name: (vocab.unmet(load_recipe(name), obs.shape)
                   or frozenset(narrate.unavailable(load_recipe(name))))
            for name in discover_recipes()
            if not str(load_recipe(name).get("id") or "").startswith("dev.")}


# ── every analysis is on screen, and the refused one says why ────────────────
@astest
async def test_every_analysis_is_on_screen_including_the_ones_that_cannot_run():
    """The screen replaces `shell.py:805` + `shell.py:814`, which between them show ONE recipe
    and hide the rest behind "Run a specific recipe… (7 more)". A screen is not a menu row."""
    screen = LedgerScreen(_entry())
    app = _Harness(screen)
    async with app.run_test(size=(120, 40)):
        listed = [option.id for option in _analysis_options(app)]
    assert set(listed) == _offered()
    assert len(listed) == len(screen.rows)


@astest
async def test_the_refused_analysis_is_on_screen_with_the_reason_it_is_refused():
    """§0's third defect and the one this component exists for: today `contrast` is simply
    ABSENT on unlabelled data, so the scientist never learns that a disease column would unlock
    it. The reason has to be READABLE, not merely present in a dataclass."""
    app = _Harness(LedgerScreen(_entry("single")))
    async with app.run_test(size=(120, 40)):
        text = _screen_text(app)

    # NAMED BY ITS METHOD NOW, not by a question. The pane used to spend both columns on prose
    # ("How different? — do the groups separate…"); it names `separation` and cites the paper,
    # and the REASON — which is what §0's third defect is about — is unchanged and still here.
    assert "separation" in text                           # the refused analysis, by method
    assert "needs a healthy/disease column" in text       # and what it is missing
    assert f"{HOLLOW}  separation" in text                # marked hollow, not filled
    assert narrate.MISSING_FACT["conditions"] in text     # narrate's words, not the screen's


@astest
async def test_the_mark_and_the_selectability_are_both_narrates_verdict():
    """The counts on screen ARE `vocab.unmet`, per recipe and in total.

    Two things are checked against the calculus, not against each other: the glyph a human reads
    and the `disabled` flag the widget enforces. A screen that drew a filled dot on a row it
    then refused to select, or the reverse, would be a second legality judgement — which is the
    failure the "no screen computes a fact narrate does not expose" rule exists to prevent.
    """
    obs = _entry("single").obs
    unmet = _unmet(obs)
    app = _Harness(LedgerScreen(_entry("single")))
    async with app.run_test(size=(120, 40)):
        rows = _rows_text(app)
        for option in _analysis_options(app):
            blocked = bool(unmet[option.id])
            assert option.disabled is blocked, f"{option.id}: widget disagrees with unmet"

    # THREE MARKS IN ONE COLUMN, so the star costs no width: ★ is the runnable row `offer`
    # advises and ● a runnable row it does not. Both mean selectable, which is what this test is
    # about, so they are counted together against `unmet`.
    filled = sum(1 for why in unmet.values() if not why)
    assert (rows.count(FILLED) + rows.count(RECOMMENDED),
            rows.count(HOLLOW)) == (filled, len(unmet) - filled)
    # 2 -> 4 at the cutover (#54): `qc` and `rank_genes` declare `("genes",)` in
    # `vocab.STEP_NEEDS`, and `_entry("single")` is a hand-built Observation with no `n_vars`, so
    # `qc` and `markers` join the hollow rows. 4 -> 5 with `preprocess`, whose `filter_genes` and
    # `filter_mito` want the same axis. The count is spelled out rather than derived because the
    # point of the line is that the fixture CONTAINS refusals — a screen with none would satisfy
    # the two assertions above vacuously.
    assert rows.count(HOLLOW) == 5, ("measured on `single`: `contrast` wants a condition column, "
                                     "`qc`, `markers` and `preprocess` want a gene axis, and "
                                     "`archetypes` is blocked upstream")


@astest
async def test_the_same_analysis_fills_in_once_the_data_carries_the_fact():
    """A `blocked` row is a statement about the DATA. Point the same screen at labelled data and
    the refusal becomes an offer, with the recipe's own gloss back in the second column.

    AND THE OTHER KINDS DO NOT MOVE, which is why they are asserted here too. `archetypes` is
    hollow for a reason no dataset can satisfy — `manylatents`' `aa` rejects the `method` kwarg —
    and since the cutover (#54) `qc`, `markers` and `preprocess` are hollow for a DIFFERENT data
    fact (`genes`, which `qc`, `rank_genes`, `filter_genes` and `filter_mito` declare in
    `vocab.STEP_NEEDS`). A condition column is not a gene axis, so better data fills in exactly
    the row that wanted what it added and leaves the other four where they were. A test that only
    checked "nothing is disabled" would pass by erasing that distinction; with four unmoved rows
    it now cannot.
    """
    entry = _entry("case-control", name="labelled.h5ad", conditions=["healthy", "disease"])
    app = _Harness(LedgerScreen(entry))
    async with app.run_test(size=(120, 40)):
        text = _screen_text(app)
        disabled = {o.id for o in _analysis_options(app) if o.disabled}

    assert "contrast" not in disabled, "the data now carries the fact it wanted"
    # {"archetypes"} -> {"archetypes", "qc", "markers", "preprocess"}: the three new rows are
    # refused for `genes`, which `case-control` does not supply and this hand-built row has no
    # `n_vars` for.
    assert disabled == {"archetypes", "qc", "markers", "preprocess"}, (
        "the upstream block and the rows missing a different fact are all that is left")
    assert narrate.MISSING_FACT["conditions"] not in text
    # The pane names the METHOD now — the prose gloss it used to print is the thing this
    # redesign removed. `separation` is `contrast`'s, off its `id: sc.contrast.separation`.
    assert "separation" in text


# ── a hollow row cannot be chosen ────────────────────────────────────────────
@astest
async def test_a_hollow_row_cannot_be_selected_even_with_the_highlight_forced_onto_it():
    """The adversarial case, not the happy path: the highlight is moved onto the refused row by
    hand — which the keyboard cannot do — and `enter` still returns nothing.

    This is enforced by Textual (`OptionList.action_select` checks `option.disabled`) rather
    than by a guard in the screen, which is why `OptionList` was chosen over `ListView`: measured
    in `textual 8.2.8`, `ListView.action_select_cursor` posts `Selected` for a disabled item.

    MUTATION-CHECKED: dropping `disabled=not r.can_run` from the option fails this with "a
    refused analysis was started", and fails three of its neighbours as well.
    """
    screen = LedgerScreen(_entry("single"))
    # the refused row is identified from NARRATE's verdict, not from the widget's own `disabled`
    # flag — keying on the flag would make the test agree with whatever the screen did.
    hollow = [row.recipe for row in screen.rows if not row.can_run]
    assert hollow, "the fixture must contain a refused analysis or this proves nothing"

    app = _Harness(screen)
    async with app.run_test(size=(120, 40)) as pilot:
        options = _options(app)
        # BY ID, not by row index: the two stopped agreeing when the group headings arrived, and
        # an index here would silently aim the highlight at whatever now sits at that offset —
        # which is how this test passed a runnable row to `enter` and reported a refusal starting.
        options.highlighted = _index_of(app, hollow[0])
        await pilot.press("enter")
        await pilot.pause()
        assert app.picked == [], "a refused analysis was started"
        assert app.screen is options.screen, "the screen was dismissed by a refused row"


@astest
async def test_the_keyboard_never_lands_on_a_hollow_row():
    """Every runnable row is reachable by arrow keys and no refused one is — including on the
    wrap, where a naive implementation walks straight off the end of the runnable block into the
    refusals (`state.ledger` sorts them last)."""
    screen = LedgerScreen(_entry("single"))
    # again from narrate's verdict, not from the widget: a test that read `option.disabled` here
    # would pass even if nothing were disabled at all.
    runnable = {row.recipe for row in screen.rows if row.can_run}
    assert len(runnable) < len(screen.rows), "the fixture must contain a refused analysis"

    app = _Harness(screen)
    async with app.run_test(size=(120, 40)) as pilot:
        options = _options(app)
        ids = [o.id for o in options._options]
        visited = {ids[options.highlighted]}
        for _ in range(2 * len(ids)):
            await pilot.press("down")
            visited.add(ids[options.highlighted])
    # By RECIPE rather than by index: this says "the keyboard reaches every analysis you can run
    # and no analysis you cannot, and never a heading", which is the invariant. The index form
    # said the same thing only while the option list and the row list were the same length.
    assert visited == runnable


@astest
async def test_a_ledger_with_nothing_runnable_refuses_everything_instead_of_crashing():
    """The degenerate screen: no analysis is legal. The bundled catalog cannot produce it today
    (only `contrast` declares a need), so it is constructed — the point being that `enter` on a
    screen with nothing to select must do nothing rather than raise, and the reasons must still
    be readable.

    MUTATION-CHECKED: replacing `action_first()` (which is `find_first_enabled`) with the obvious
    `highlighted = 0` fails this with "a refused row was pre-selected". On the bundled catalog
    the two are indistinguishable — row 0 is always runnable — which is exactly why this fixture
    is hand-built."""
    rows = [LedgerRow(recipe="embed", ask="Map it", gloss="g", can_run=False,
                      blocked=("needs coordinates first",)),
            LedgerRow(recipe="contrast", ask="How different?", gloss="g", can_run=False,
                      blocked=("needs a healthy/disease column",))]
    app = _Harness(LedgerScreen(_entry()))
    app._ledger.rows = rows
    async with app.run_test(size=(120, 40)) as pilot:
        assert _options(app).highlighted is None, "a refused row was pre-selected"
        await pilot.press("enter")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        text = _screen_text(app)
    assert app.picked == []
    assert "needs coordinates first" in text and "needs a healthy/disease column" in text


# ── choosing one, and going back ─────────────────────────────────────────────
@astest
async def test_choosing_a_runnable_row_hands_back_the_recipe_name():
    """The handoff to component 4 (the run screen), which does not exist yet: the screen returns
    the recipe NAME, and the caller pairs it with the `DataEntry` it pushed with —
    `entry.as_source()` is already the `(data_folder, dataset, modality, obs)` tuple
    `shell._run` takes. Returning anything else would be a second way to start a run."""
    screen = LedgerScreen(_entry("single"))
    app = _Harness(screen)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("enter")
        await pilot.pause()

    assert app.picked == [screen.rows[0].recipe]
    assert app.picked[0] in discover_recipes()
    # the head of the list is `narrate.offer`'s recommendation — the screen opens on it, so
    # `enter` straight away runs what the product advises. Derived, not transcribed.
    assert app.picked[0] == next(o["recipe"] for o in narrate.offer(_entry("single").obs)
                                 if o["recommended"])


@astest
async def test_the_screen_opens_on_the_recommendation_and_never_on_a_refusal():
    """`OptionList.highlighted` defaults to `None`, so the first `enter` would do nothing; and a
    naive `highlighted = 0` on data where the first row is refused would open on a row that
    cannot be chosen."""
    app = _Harness(LedgerScreen(_entry("single")))
    async with app.run_test(size=(120, 40)):
        options = _options(app)
        # NOT index 0 any more — index 0 is a group heading, and `action_first` is
        # `find_first_enabled`, so it steps over it. That is the whole reason the headings are
        # disabled rather than merely styled.
        opened = options._options[options.highlighted]
        assert opened.disabled is False
        assert not is_group_id(opened.id), "the highlight opened on a heading"
        assert opened.id == next(r.recipe for r in app._ledger.rows if r.can_run)


@astest
async def test_escape_goes_back_without_choosing_anything():
    """`None` means "went back" and is not the same as "not answered" — the caller has to be able
    to tell them apart, which is why the harness records into a list."""
    app = _Harness(LedgerScreen(_entry("single")))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
    assert app.picked == [None]


# ── the data stays identified, and the reason stays whole ────────────────────
@astest
async def test_the_data_this_is_about_stays_on_screen():
    """After the roster is gone, the screen still says WHICH dataset these analyses are about,
    in `narrate`'s words (`DataEntry.contents` is `narrate.contents`) — a scientist who came
    back to the terminal should not have to guess."""
    entry = _entry("single", n_obs=2700, n_vars=32738)
    app = _Harness(LedgerScreen(entry))
    async with app.run_test(size=(120, 40)):
        text = _screen_text(app)
    assert entry.name in text
    assert entry.size in text and entry.contents in text
    assert narrate.contents(entry.shape) in text


@astest
async def test_the_reason_and_the_gloss_are_wrapped_not_truncated():
    """§0 records the defect this replaces: `render_result`'s hardcoded `width=30` steps column
    splits `lid 6.999 →` / `2.072` mid-value, and `narrate.run_panel` clips detail at 58
    characters because plain text cannot re-wrap. A layout engine can, so the second column
    folds and every word survives at BOTH widths — component 1 deliberately hands the screen
    unclipped text and this is the other half of that decision.

    MUTATION-CHECKED, twice, and reported as measured rather than as intended:

      * making the detail column `no_wrap=True, overflow="ellipsis"` — the clip `run_panel` is
        obliged to do — fails this at 70 columns ("something was elided");
      * measuring the ask column over `row.ask` instead of over the rendered `_ask(row)` fails
        it at 120 columns, which is the bug that was actually in this file first: the starred
        row rendered as "Map it and look togethe…".

    A third mutation does NOT fail this test and is recorded so nobody assumes it would:
    replacing the folding column with a hardcoded `width=30` still wraps, because `expand=True`
    hands the leftover width back to it. It is caught by
    `test_the_refused_analysis_is_on_screen_with_the_reason_it_is_refused`, whose `○ How
    different?` stops matching once the mark column shifts.
    """
    entry = _entry("single")
    # MEASURED OVER THE REFUSED ROWS. `detail` is the gloss on a runnable row and the reason on a
    # refused one, and the pane no longer draws the gloss — a runnable row reads
    # "● leiden  Scanpy 2018  ready". The reason is still drawn in full, and it is the half that
    # must never be clipped: it is the only thing telling a scientist what their data lacks.
    refused = [r for r in state.ledger(entry.obs) if not r.can_run]
    assert refused, "the fixture must contain a refused analysis or this proves nothing"
    longest = max(refused, key=lambda r: len(r.detail)).detail
    tail = longest.split()[-4:]

    for width in (120, 70):
        app = _Harness(LedgerScreen(entry))
        async with app.run_test(size=(width, 40)):
            text = _screen_text(app)
        assert "…" not in text, f"something was elided at {width} columns"
        # the words survive the fold; at 70 they are on the next line, which is the point
        assert all(word in text for word in tail), f"lost the tail of a gloss at {width}"


# ── the screen owns layout, and no fact ──────────────────────────────────────
@astest
async def test_a_real_roster_row_drives_the_screen_unchanged():
    """The seam actually connects: a `DataEntry` straight out of `state.roster()` is what the
    screen takes, and the analyses it lists are the ones `state.ledger` computes for it. Uses a
    bundled dataset, which is present on any checkout."""
    entry = {r.name: r for r in state.roster()}["swissroll"]
    app = _Harness(LedgerScreen(entry))
    async with app.run_test(size=(120, 40)):
        listed = [o.id for o in _analysis_options(app)]
        disabled = {o.id for o in _analysis_options(app) if o.disabled}
    # SET, not list: grouping keeps `state.ledger`'s order inside each group but gathers a
    # group's rows together, so a refused row sorted last by the ledger is drawn beside the
    # runnable rows that share its assumption. Which analyses are listed is the invariant;
    # their order within the pane is the grouping's business.
    assert set(listed) == {r.recipe for r in state.ledger(entry.obs)}
    assert disabled == {name for name, why in _unmet(entry.obs).items() if why}


@astest
async def test_the_shared_stylesheet_does_not_break_the_screen():
    """The app (component 2) loads `manyruns.tcss`, and app-level CSS outranks a widget's
    `DEFAULT_CSS` in Textual — so the layout has to be checked under the stylesheet it will
    actually run beneath, not only under the bare harness the other tests use.

    Isolated to ONE test on purpose: it is the only place this file depends on another
    component's code, so a failure elsewhere in `manyruns.tui.app` cannot make the ledger's own
    tests look broken. `RosterScreen` is never constructed here — only the stylesheet is read.
    """
    from manyruns.tui.app import stylesheet

    class _Themed(_Harness):
        CSS = stylesheet()

    entry = _entry("single")
    app = _Themed(LedgerScreen(entry))
    async with app.run_test(size=(120, 40)) as pilot:
        text = _screen_text(app)
        assert not is_group_id(_options(app)._options[_options(app).highlighted].id)
        await pilot.press("enter")
        await pilot.pause()
    assert entry.name in text and narrate.MISSING_FACT["conditions"] in text
    assert app.picked == [next(r.recipe for r in state.ledger(entry.obs) if r.can_run)]


def test_the_screen_names_no_recipe_and_phrases_no_refusal():
    """The rule the whole rewrite runs under, as a source guard. A recipe added as a YAML file
    must reach this screen with no code change, and the sentence explaining a refusal must come
    from `narrate.MISSING_FACT` — a phrase written here would be a fourth account of what this
    product can and cannot do.

    Read through `ast` rather than as text, so that the guard is about what the module DOES.
    `test_tui_state`'s version of this subtracts the docstring by hand; here the prose names
    `vocab.unmet` and `contrast` on purpose (explaining a decision is not making one), and a
    line-prefix filter cannot tell a docstring's continuation line from code.
    """
    code = _executable_source(LEDGER_MODULE)

    for name in discover_recipes():
        assert name not in code, f"{name} is hardcoded in the ledger screen"
    # Both reasons a fact can be missing — `MISSING_FACT` (nothing produced it) and
    # `CLEARED_FACT` (a narrowing took it away). A second phrase table is a second chance for
    # the screen to grow its own copy of one.
    for phrase in (*narrate.MISSING_FACT.values(), *narrate.CLEARED_FACT.values()):
        assert phrase not in code, "a refusal is phrased in the screen instead of in narrate"
    assert "vocab" not in code, "the screen reaches past narrate into the calculus"


def test_the_screen_module_shadows_the_state_function_of_the_same_name():
    """A MEASURED HAZARD IN THE PACKAGE API, pinned rather than hidden — and it is not this
    component's to fix.

    `manyruns/tui/__init__.py` re-exports `state.ledger` and `state.roster` as
    `manyruns.tui.ledger` / `manyruns.tui.roster`, and the screens are modules of those exact
    names. Python's import machinery `setattr`s a submodule onto its parent package, so importing
    a screen REBINDS the package attribute from the function to the module — after which
    `from manyruns.tui import ledger; ledger(obs)` raises `TypeError: 'module' object is not
    callable`. Measured today, and it is live for `roster` independently of this component:
    `import manyruns.tui.app` alone is enough to break `from manyruns.tui import roster`.

    Renaming was considered and rejected: `roster.py` and `run.py` already set the convention, so
    a lone `ledger_screen.py` would trade a package-wide problem for an inconsistency. The fix is
    two lines in `__init__.py` — drop `ledger` and `roster` from the re-export and `__all__`,
    since every consumer measured today (`app.py`, `roster.py`, this module) already reaches them
    as `state.ledger` / `state.roster`. THIS TEST FAILS WHEN THAT IS DONE, which is the point.
    """
    import manyruns.tui as pkg

    assert pkg.ledger is LEDGER_MODULE, "fixed? drop this test with the re-export"
    assert pkg.ledger is not state.ledger
    with pytest.raises(TypeError):
        pkg.ledger(_entry().obs)


def test_the_app_and_the_shell_give_ONE_account_of_the_same_dataset():
    """The two front doors must not disagree about what the data carries.

    `shell` has always passed `provided=obs.provides()`; `LedgerScreen` passed nothing, so the
    observation's own facts were dropped on the app's side. Measured on `data/pbmc3k_raw.h5ad`
    (2700 x 32738) before the fix: the shell offered **8** recipes and the app offered **6**,
    drawing `qc` and `markers` hollow with "needs data with gene names — a point cloud has no
    genes to read" — about a file with 32,738 gene names.

    Invisible until the cutover, because nothing declared `("genes",)` until `qc` and
    `rank_genes` did. That is why this is pinned rather than left to the type: the divergence
    existed for as long as the seam did and cost nothing until one fact made it matter.
    """
    from manyruns.tui import state
    from manyruns.tui.ledger import LedgerScreen

    obs = Observation(shape="clusters", modality="scrna", conditions=None, n_timepoints=0,
                      n_obs=2700, n_vars=32738, source="pbmc3k", signals=())
    entry = state.DataEntry(name="pbmc3k", kind="bundled", dataset="pbmc3k", obs=obs)

    assert "genes" in obs.provides(), "the fixture has to carry the fact under test"
    app_rows = LedgerScreen(entry).rows
    shell_rows = state.ledger(obs, obs.provides())

    assert [r.can_run for r in app_rows] == [r.can_run for r in shell_rows]
    assert {r.recipe for r in app_rows if r.can_run} == {r.recipe for r in shell_rows if r.can_run}

    # and an EXPLICIT empty set still means empty — the default fills a gap, it does not override
    assert not all(r.can_run for r in LedgerScreen(entry, provided=frozenset()).rows)


# ── the declared fact, not the sniff ─────────────────────────────────────────
GENE_STEPS = ("qc", "markers", "preprocess")
#: The three recipes whose steps declare `("genes",)` in `vocab.STEP_NEEDS` — measured
#: 2026-08-17 over the bundled catalog by `test_the_gene_recipes_are_exactly_the_ones_named`
#: below, so a fourth arriving fails there rather than silently weakening the two tests above it.


@astest
async def test_a_declared_gene_axis_reaches_the_screen_when_the_file_is_not_there():
    """The declaration is the authority; the `n_vars` sniff is the fallback under it.

    THE BUG THIS PINS, measured 2026-08-17 by running `state.roster()` from `/tmp` with an empty
    drop folder: `configs/dataset/pbmc3k.yaml` declares `handle.provides: [genes]` and its `ref`
    is the checkout-relative `data/pbmc3k_raw.h5ad`, so wherever those bytes are not reachable
    `shell._resolve_dataset` returns `n_vars=None`. The screen's default was
    `Observation.provides()`, which read only `n_vars`, so `qc`, `markers` and `preprocess` were
    drawn HOLLOW reading "needs data with gene names — a point cloud has no genes to read" —
    about the one declaration in the folder that states it has them, 6 analyses offered where
    the declaration offers 9. That is the same defect the `provided=obs.provides()` default was
    added to fix, one source of truth further out: the YAML said it and no surface asked.

    THE OTHER DIRECTION IS ASSERTED TOO, and it is what makes this an authority rather than a
    second guess: a declaration is believed when it states a fact the file cannot show, and it
    is believed when it states the ABSENCE of one the sniff would have found. `n_vars` set with
    `declared=()` refuses all three, which is `vocab.dataset_provides`' own rule (an explicit
    `provides:` wins over the suffix guess) reaching the screen unchanged.
    """
    absent_file = _entry("clusters", name="pbmc3k", declared=("genes",))
    app = _Harness(LedgerScreen(absent_file))
    async with app.run_test(size=(120, 40)):
        disabled = {o.id for o in _options(app)._options if o.disabled}
        text = _screen_text(app)
    assert absent_file.obs.n_vars is None, "the fixture has to be the un-openable file"
    assert not (set(GENE_STEPS) & disabled), (
        "the YAML declares `genes`; the screen refused it because it could not open the file")
    assert narrate.MISSING_FACT["genes"] not in text

    # no declaration in hand (a dropped file) → the sniff, unchanged
    sniffed = _entry("clusters", name="dropped.h5ad", n_obs=2700, n_vars=32738)
    assert not (set(GENE_STEPS) & {r.recipe for r in LedgerScreen(sniffed).rows if not r.can_run})

    # a declaration that states the absence outranks a sniff that would have found one
    contradicted = _entry("clusters", name="no_var_names.h5ad", n_obs=2700, n_vars=32738,
                          declared=())
    refused = {r.recipe for r in LedgerScreen(contradicted).rows if not r.can_run}
    assert set(GENE_STEPS) <= refused, "an explicit declaration has to beat the suffix guess"


@astest
async def test_the_bundled_declaration_survives_a_cwd_that_is_not_the_checkout(monkeypatch,
                                                                               tmp_path):
    """End to end through the real seam: `state.roster()` → `DataEntry` → `LedgerScreen`.

    The test above builds the Observation by hand, which proves the screen reads the field and
    nothing about who fills it. This runs the actual catalog from a CWD where `pbmc3k`'s
    checkout-relative `ref` does not resolve — an installed `uv tool install` manyruns, or a
    git worktree, or simply launching from `~` — and asserts the declaration still reaches the
    screen. Measured before the fix, from `/tmp`: `entry.obs.provides()` was `frozenset()` and
    the screen offered 6 analyses; after, it is `{"genes"}` and the screen offers 9.

    The drop folder is pointed at an empty tmp dir because `$MANYRUNS_DATA_DIR` means THAT
    folder and only it (see `shell.data_dirs`), so whatever is in the developer's `data/` or
    `~/.manyruns/data` cannot add rows and change the count under this assertion.
    """
    from manyruns.catalog import load_dataset

    drop = tmp_path / "drop"
    drop.mkdir()
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(drop))
    monkeypatch.chdir(tmp_path)

    entries = {r.name: r for r in state.roster()}
    entry = entries["pbmc3k"]
    assert entry.obs.n_vars is None, "the ref must not resolve from here, or this proves nothing"
    assert "genes" in entry.obs.provides(), "the declared fact did not reach the roster row"

    app = _Harness(LedgerScreen(entry))
    async with app.run_test(size=(120, 40)):
        disabled = {o.id for o in _options(app)._options if o.disabled}
    assert not (set(GENE_STEPS) & disabled)

    # EVERY bundled row, not just the one: the declaration is what the screen's default resolves
    # to for a dataset that has one, so a sniff creeping back in fails here for whichever dataset
    # declares next. `contents`/`size` are unaffected — this is about the calculus fact alone.
    for name, row in entries.items():
        if row.kind != "bundled":
            continue
        assert row.obs.provides() == narrate.declared_facts(load_dataset(name)), name


def test_the_gene_recipes_are_exactly_the_ones_named():
    """`GENE_STEPS` is transcribed, so it gets a check that fails when the catalog moves.

    Derived from `vocab.STEP_NEEDS` through `recipe_needs`, which is the order-blind upper bound
    and therefore the right question here — "does this recipe touch the gene axis at all", not
    "can it run today". Measured 2026-08-17: `qc`, `markers`, `preprocess`.
    """
    named = {n for n in discover_recipes() if "genes" in vocab.recipe_needs(load_recipe(n))}
    assert named == set(GENE_STEPS)


def test_wide_citations_survive_and_align_with_ascii_citations():
    """Rich renders 李 as two cells. Both citations must survive with status at one column."""
    from io import StringIO
    from rich.cells import cell_len
    from rich.console import Console
    from manyruns.tui.ledger import _method_row
    from manyruns.tui.state import LedgerRow

    citations = ["李 2024", "Long 2024"]
    width = max(map(cell_len, citations))
    lines = []
    for citation in citations:
        row = LedgerRow(recipe="method", ask="", gloss="", can_run=True,
                        assumption="clusters", method="方法", cite=citation)
        output = StringIO()
        Console(file=output, width=80, color_system=None).print(_method_row(row, 6, width))
        line = output.getvalue().strip("\n")
        assert citation in line and "方法" in line
        lines.append(line)
    # Compare the suffix after each padded prefix; a clipped citation or shifted status differs.
    assert cell_len(lines[0].split("ready")[0]) == cell_len(lines[1].split("ready")[0])
