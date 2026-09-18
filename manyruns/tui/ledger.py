"""§3.2 · the ledger — every analysis this data admits, and why a refused one is refused.

ONE SCREEN REPLACING TWO PROMPTS. `shell.py:805` asks "What should I look for?" and
`shell.py:814` asks "Which recipe?", which are the same question asked twice: a menu row can
hold one recommendation, so the other seven analyses hide behind "Run a specific recipe…
(7 more)". A screen is not a menu row and does not need the fold.

**THE HOLLOW ROWS ARE THE POINT.** Today a recipe that cannot run is simply ABSENT from the
list (`shell._pick_recipe` renders `narrate.offer`, which only returns the legal set), so a
scientist never learns that a disease column would unlock `contrast` — §0 of the spec names
that as one of the three defects the rewrite exists to fix, and it is the sharp one: refusal
with a reason is the commitment this product actually keeps, and the interface hides it. A
hollow row is on screen carrying the reason precisely because its absence is the informative
part.

NO FACT IS COMPUTED HERE. Every string on this screen comes from `manyruns.tui.state`, which
gets it from `narrate`: legality and its reason from `narrate.blocked` (via `LedgerRow.can_run`
/ `.blocked`), the two columns from `narrate.split_question` (via `.ask` / `.gloss`), the
choice of *which* of gloss-or-reason occupies the second column from `LedgerRow.detail` — made
once there so that this screen and any other cannot drift — and the dataset line from
`DataEntry.size` / `.contents`. What this module owns is layout, three marks (`●`, `○`, `★`)
and the key handling.

WHY `OptionList` AND NOT `ListView`, measured in `textual 8.2.8` by reading the installed
source: the not-selectable half of "hollow" is enforced by the widget, not by a guard of mine.
`OptionList.action_select` returns without posting when `option.disabled`; `_on_click` checks
`disabled` before selecting; `action_cursor_up/down` move via
`_widget_navigation.find_next_enabled`, so the highlight cannot land on a hollow row at all.
`ListView` does NOT do this — its `action_select_cursor` and `_on_list_item__child_clicked`
post `Selected` for whatever is highlighted or clicked, disabled or not, and its default
`index = 0` highlights the first item regardless — so building on it would mean reintroducing
the check by hand in the one place a bug is most expensive.

WHY THE TWO GLYPHS LIVE HERE AND NOT IN `narrate`. `narrate._OUTCOME_MARKS` is in `narrate`
because the outcome VOCABULARY has five words and two surfaces render it, so a private copy
would drift into a second name for one outcome. `●`/`○` marks a two-valued boolean
(`LedgerRow.can_run`, which is `narrate.blocked` inverted) that exactly one surface draws;
there is no vocabulary to drift and nothing to disagree with. The property that matters — that
the mark is a function of narrate's verdict and never a second judgement of legality — is
pinned by `test_the_mark_and_the_selectability_are_both_narrates_verdict`, which is a stronger
guarantee than moving two characters to another file.

HANDOFF: this screen is `Screen[Optional[str]]` and returns **the chosen recipe name**, or
`None` for "went back". Push it and take the result::

    self.push_screen(LedgerScreen(entry), callback=self.start_run)   # entry: state.DataEntry

The caller already holds the `DataEntry` it pushed with, so pairing the two gives component 4
everything a run needs: `entry.as_source()` is the `(data_folder, dataset, modality, obs)`
tuple `shell._run` takes, and the recipe name is its `recipe_name`. Nothing here starts a run —
component 4 (the run screen) does not exist yet, and inventing a second way to start one is
exactly what `as_source()` was shaped to prevent.

`dismiss` requires this screen to be PUSHED, not to be the app's base screen — Textual refuses
to pop the last screen off the stack. That is deliberate: it fails loudly in the integrator's
face rather than silently doing nothing under a user pressing escape.

IMPORT IT AS `from manyruns.tui.ledger import LedgerScreen`, never as `from manyruns.tui
import ledger`. That second form gives you `state.ledger` — the FUNCTION `manyruns/tui/
__init__.py` re-exports under this module's name — and importing this module rebinds the
package attribute to the module, after which the function is unreachable through the package
and calling it raises `TypeError: 'module' object is not callable`. Measured, and already live
for `roster` independently of this component. `tests/test_tui_ledger.py::
test_the_screen_module_shadows_the_state_function_of_the_same_name` pins it with the two-line
fix; renaming the module instead was rejected because `roster.py` and `run.py` set the
convention.

`textual` IS NOT DECLARED IN `pyproject.toml` — it is installed in `.venv` only (§8 measured
it there). Importing this module on a fresh install raises `ModuleNotFoundError`, which is why
`manyruns.tui.__init__` does not import it and why `tests/test_tui_ledger.py` opens with an
`importorskip`. Adding the dependency line is the integrator's call, not this component's, and
§8 flags that installing it also pulled `rich` 13 → 15.0.0 sideways.
"""
from __future__ import annotations

from typing import Optional

from rich.cells import cell_len
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from manyruns import narrate
from manyruns.tui import state
from manyruns.tui.state import DataEntry, LedgerRow

#: Filled = you can run it and select it. Hollow = you cannot, and the row says what it needs.
#: The mockup in §3.2 draws exactly these two; see the module docstring for why they are not in
#: `narrate`.
RUNNABLE_MARK = "●"
BLOCKED_MARK = "○"

#: `narrate.offer`'s advice, carried across from the surface this screen replaces:
#: `shell._pick_recipe` appends `"  ★"` to every recommended recipe in its dropdown, and this
#: appends it to the same place — the end of the ask — so the mark stays beside what it is
#: advising instead of floating at the far edge of a wrapped row. It is ADVICE, never legality:
#: `recommended` is true of many recipes at once (measured: a `clusters` observation recommends
#: five), which is precisely why it cannot be the whole screen and why the fold it was hiding
#: behind had to go.
RECOMMENDED_MARK = "★"

#: A runnable row the numbers warn about. One column wide — the VS16 form ⚠️ is two cells and
#: would break the only alignment this pane has, so it is the bare U+26A0.
CAUTION_MARK = "\u26a0"


def _cell(text: str) -> Text:
    """A grid cell that is never re-interpreted as markup.

    `rich` parses square brackets in a `str` cell, so a recipe whose `question:` or whose
    blocked-reason contained `[...]` would render wrong or raise — from data, at the worst
    moment. `Text` is rendered literally.
    """
    return Text(text)


def _ask(row: LedgerRow) -> str:
    """The first column's text: the ask, starred when this is what `narrate.offer` advises.

    A hollow row never carries the star even when `suits:` covers the shape — advising an
    analysis the data cannot support is advice you cannot take. `offer` computes `recommended`
    from `suits:` alone and says so; the legality question is `can_run`, and this is the one
    place the two meet.
    """
    return f"{row.ask}  {RECOMMENDED_MARK}" if row.recommended and row.can_run else row.ask


def _prompt(row: LedgerRow, ask_width: int) -> Table:
    """One row: mark · ask · detail.

    A `Table.grid` per option rather than a padded string, because the detail column WRAPS and
    the wrap has to hang under itself. Measured on the bundled catalog against `single`: the
    longest ask is 24 characters and the longest detail 79, so mark + ask + detail is 108 and
    every row folds on an 80-column terminal. A single padded string would put the continuation
    at column 0, under the mark.

    `ask_width` is passed in rather than measured here because each option is its own table and
    `rich` sizes a table's columns from its own content: measured per-row, "Trace the path" (14)
    and "Map it and look together" (24) would produce two different column widths and the second
    column would not line up. One width computed once over all rows is what makes it a column.
    It is DERIVED from the rows, not a constant — the `width=30` in `shell.render_result` that
    §0 records splitting `lid 6.999 →` / `2.072` mid-value is the failure mode of the constant.

    Nothing here sets a colour: the disabled rendering is `OptionList`'s
    `option-list--option-disabled` component style, so hollow rows dim through CSS. §2 is
    explicit that layout as a constraint rather than a magic number is the whole reason for
    Textual, and a hardcoded style here would be the magic number in another costume.
    """
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=1, no_wrap=True)            # ● / ○
    grid.add_column(width=ask_width, no_wrap=True)    # the ask, ★ if it is the recommendation
    grid.add_column(ratio=1, overflow="fold")         # the gloss, or why it is refused
    grid.add_row(
        _cell(RUNNABLE_MARK if row.can_run else BLOCKED_MARK),
        _cell(_ask(row)),
        _cell(row.detail),
    )
    return grid



#: The conditioning group's mark. DELIBERATELY NOT in `narrate.ASSUMPTION_GLYPH`, which is
#: exhaustively checked against the topology vocabulary in both directions — `preprocess` and
#: `qc` assume no geometry at all, so a glyph for them there would be a word the vocabulary does
#: not have. They are a different KIND sharing one list, and this is the mark that says so.
CONDITION_MARK = "\u229e"     # ⊞

#: Group headings are options (so they scroll with their rows) and therefore need ids, because
#: an id-less `Option` reports `id=None` and `None` then turns up in every set of recipe ids a
#: caller collects. A recipe name cannot contain `:`, so this namespace cannot collide with one,
#: and `is_group_id` is the one place that knowledge lives.
GROUP_ID_PREFIX = "group:"


def is_group_id(option_id: "str | None") -> bool:
    """Whether an option id names a group heading rather than an analysis."""
    return bool(option_id) and str(option_id).startswith(GROUP_ID_PREFIX)


def _group(rows: "list[LedgerRow]") -> "list[tuple[str | None, list[LedgerRow]]]":
    """Rows bucketed by the geometry they assume, in the order the rows already had.

    ORDER IS INHERITED, NEVER RECOMPUTED. `state.ledger` already ranks — runnable first, then
    `offer`'s recommendation, then alphabetical — and its docstring is explicit that it must not
    become a second ranking. So a group's position is the position of its FIRST row, which means
    grouping reorders nothing: it only draws a line around runs of rows that were already
    adjacent for the same reason.

    Conditioning rows (`assumption is None` — `preprocess`, `qc`) always sort last regardless of
    where they fell, because they answer a different question than the rest of the pane and
    reading them as one more analysis is the confusion the grouping exists to remove.
    """
    order: list[str | None] = []
    buckets: dict[str | None, list[LedgerRow]] = {}
    for row in rows:
        if row.assumption not in buckets:
            buckets[row.assumption] = []
            order.append(row.assumption)
        buckets[row.assumption].append(row)
    order.sort(key=lambda a: a is None)      # stable: only conditioning moves, and only to last
    return [(a, buckets[a]) for a in order]


def _heading(assumption: "str | None", rows: "list[LedgerRow]") -> Table:
    """The group's line: the glyph, what it commits you to, and whether the data agrees.

    "assume …" rather than a noun, because that is the honest verb. Every method here imposes a
    geometry — leiden imposes discreteness, dpt imposes a continuum — and naming the imposition
    is what lets a reader decline it. The pane used to phrase the same choice as a question about
    intent ("What groups are in here?"), which hides that a commitment is being made at all.

    NO "fits your data" MARKER HERE, and the reason is a bug this pane had for exactly one
    render. `claims` (the geometry a method assumes) and `suits` (the data shapes it is advised
    for) are DIFFERENT AXES: `embed` claims `surface` and suits `[single, unknown, manifold,
    clusters]`, so marking the group produced "assume a curved sheet ← fits your data" over
    2,638 cells in 8 labelled groups — asserting the data is a curved sheet, which is false.
    Recommendation is a property of the RECIPE, so it is drawn on the row (`_method_row`'s
    mark), where `recommended` has always meant what it says.
    """
    glyph = rows[0].glyph or CONDITION_MARK
    if assumption is None:
        text = "condition the data first"
    else:
        text = f"assume {narrate._TOPOLOGY_PHRASE.get(assumption, assumption)}"
    # Spacing is BAKED INTO THE STRING rather than left to `Table.grid(padding=…)`. Measured:
    # with fixed-width columns and `expand=True`, rich consumes the padding and the glyph
    # renders flush against its own heading ("∿assume a curved sheet"). One column and explicit
    # spaces is deterministic at every width, which is the property this pane most lacks.
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1, overflow="ellipsis")
    # BOLD, and it is the one style set in this module. `_prompt`'s note — "nothing here sets a
    # colour: the disabled rendering is `OptionList`'s component style" — is about COLOUR, and
    # this sets none. A heading is disabled for a structural reason (the keyboard must step over
    # it), not because anything is unavailable, and the dim that `option-list--option-disabled`
    # applies says the opposite: measured in a real terminal, the group headings were the
    # faintest thing on a screen whose whole organising idea they carry. Weight lifts them back
    # out without claiming a second legality judgement.
    grid.add_row(Text(f"{glyph}  {text}", style="bold"))
    return grid


def _status(row: "LedgerRow") -> str:
    """The status column. `row.measured` decides what kind of thing it says; this adds only
    the glyph, which is the screen's to own the way the star in `_ask` is.

    THE CAUTION DOES NOT TOUCH `disabled`. A row the numbers warn about is exactly as
    selectable as one they do not — freestyle: the marker informs and never blocks. The
    reader sees "it runs, but…" and decides.
    """
    if row.measured == "refused":
        return row.detail
    if row.measured == "caution":
        return f"{CAUTION_MARK} {row.caution[0]}"
    return "ready"


def _method_row(row: "LedgerRow", label_width: int, cite_width: int) -> Table:
    """One analysis under its heading: mark · method · citation · status.

    THE METHOD IS NAMED. `leiden`, `dpt`, `mioflow` — the recipes have declared them all along
    in `id:` and no surface read it, so the menu offered "What groups are in here?" and a reader
    who knew exactly which algorithm they wanted could not find it. The citation is the other
    half: `source.cite` carries a paper and a DOI for nine of the eleven recipes, and showed
    nowhere.

    `preprocess` and `qc` are labelled by RECIPE rather than by method: both declare `osca`
    (they come from one paper), so the method column would print the same word twice and
    distinguish nothing.

    Widths are passed in, measured once over every row, for the reason `_prompt` records: a
    table sized from its own content gives each row its own column width, and columns that do
    not line up are not columns.
    """
    label = row.recipe if row.assumption is None else (row.method or row.recipe)
    # Three states in ONE column, so the star costs no width: ★ is the runnable row `offer`
    # advises, ● a runnable row it does not, ○ a row this data cannot support. A hollow row
    # never stars — advising an analysis the data cannot carry is advice you cannot take, which
    # is the split `_ask` documents and the one place legality and suitability meet.
    mark = (RECOMMENDED_MARK if row.recommended and row.can_run
            else RUNNABLE_MARK if row.can_run else BLOCKED_MARK)
    # The indent is what puts an analysis UNDER its assumption rather than beside it, so the
    # group reads as one block. Spacing baked in for the reason `_heading` records.
    # Rich measures terminal cells: "李 2024" is six characters but seven cells. Pad by that
    # same measure so the citation survives and the status starts at the same cell on each row.
    label += " " * max(0, label_width - cell_len(label))
    cite = row.cite + " " * max(0, cite_width - cell_len(row.cite))
    prefix = f"   {mark}  {label}  {cite}  "
    grid = Table.grid(expand=True)
    grid.add_column(width=cell_len(prefix), no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")           # ready, or why not — wraps under itself
    grid.add_row(_cell(prefix), _cell(_status(row)))
    return grid

class LedgerScreen(Screen[Optional[str]]):
    """The analysis picker. Returns the chosen recipe name, or `None` for back."""

    BINDINGS = [Binding("escape", "back", "back", show=True)]

    DEFAULT_CSS = """
    LedgerScreen {
        layout: vertical;
        align: center top;
    }
    LedgerScreen > #ledger-frame {
        width: 100%;
        height: auto;
        max-height: 100%;
        margin: 1 2;
        padding: 0 1;
        border: round $primary;
        border-title-align: left;
        border-subtitle-align: right;
    }
    LedgerScreen #ledger-rows {
        background: transparent;
    }
    LedgerScreen #ledger-hint {
        /* DOCKED, so the sentence that explains the hollow rows survives a catalog long enough
           to scroll. Docking also shrinks what `max-height: 100%` on the frame resolves to, so
           the box hugs its content the way §3.2 draws it and the OptionList — which is itself a
           ScrollView — takes over the scrolling when the analyses outgrow the terminal. */
        dock: bottom;
        height: auto;
        margin: 0 3 1 3;
        color: $text-muted;
    }
    """

    def __init__(self, entry: DataEntry,
                 provided: "frozenset | set | tuple | None" = None,
                 name: Optional[str] = None, id: Optional[str] = None,  # noqa: A002
                 classes: Optional[str] = None, *,
                 selected_recipe: Optional[str] = None) -> None:
        """`entry` is the roster row the previous screen selected — the whole row, not an
        unpacked `Observation`, so the dataset's name, size and contents stay with it and the
        caller can hand `entry.as_source()` to `shell._run` afterwards without a second lookup.

        `provided` is what the caller already holds (a precomputed embedding, a graph another
        tool built) and goes straight through to `state.ledger` → `vocab.unmet`, whose test is
        structural rather than canonical: holding the fact is enough to unlock the recipe.

        The ledger is computed HERE, not at mount, so `screen.rows` is readable before the
        screen is on screen. Measured: `state.ledger` is 49 ms cold and 21 ms warm (eight recipe
        YAMLs), against `roster()`'s 361 ms cold — the data was already inspected by then, and
        this screen opens no file.
        """
        super().__init__(name=name, id=id, classes=classes)
        self.entry = entry
        self.selected_recipe = selected_recipe
        # DEFAULTS TO THE OBSERVATION'S OWN FACTS, which is what the shell has always passed
        # (`shell.py`: `provided=obs.provides()`). The app passed nothing, so the two front doors
        # gave two accounts of one dataset — measured on `data/pbmc3k_raw.h5ad` (2700 x 32738):
        # the shell offered 8 recipes and the app offered 6, drawing `qc` and `markers` hollow
        # with "needs data with gene names — a point cloud has no genes to read" about a file with
        # 32,738 of them. Invisible until #54, because no step declared `("genes",)` before it.
        #
        # `obs.provides()` is the DECLARATION where the dataset carries one and the `n_vars`
        # sniff otherwise — see `narrate.Observation.provides`. That distinction is this screen's
        # to care about because the sniff only speaks where the bytes are reachable: measured
        # 2026-08-17 from `/tmp` with an empty drop folder, `pbmc3k`'s `ref` resolves to nothing,
        # `n_vars` is None, and this screen offered 6 analyses — drawing `qc`, `markers` and
        # `preprocess` hollow against a YAML that declares `genes` — where the declaration offers
        # 9. Nothing is read here: `state.roster` attaches the declaration to the Observation, so
        # this line is unchanged and the fact arrives through the one accessor both doors use.
        #
        # `None` means "not specified" rather than "nothing", so a caller that genuinely holds no
        # extra facts passes `frozenset()` and still gets an empty set. The observation's facts
        # are not an override — they are what the data says about itself.
        if provided is None:
            provided = entry.obs.provides()
        self.rows: list[LedgerRow] = state.ledger(entry.obs, provided,
                                                  measured=state.measurement(entry))

    def compose(self) -> ComposeResult:
        # `ask_width` over ALL rows, hollow ones included: a column that changed width when a
        # refused analysis scrolled past would not be a column. Measured over the RENDERED text
        # (`_ask`, star included) and not over `row.ask`, because the first version measured the
        # bare ask and cropped the one row that carries the star to "Map it and look togethe…".
        # Measured over EVERY row, hollow ones included, for the reason `_prompt` records: a
        # column that changed width when a refused analysis scrolled past would not be a column.
        label_width = max((cell_len(r.recipe if r.assumption is None else (r.method or r.recipe))
                           for r in self.rows), default=0)
        cite_width = max((cell_len(r.cite) for r in self.rows), default=0)
        options: list[Option] = []
        for assumption, group in _group(self.rows):
            # The heading is an OPTION so it scrolls with its rows rather than floating away
            # from them, and `disabled` is what keeps the keyboard from landing on it — Textual
            # skips disabled options when moving, so ↑↓ still steps analysis to analysis.
            options.append(Option(_heading(assumption, group), disabled=True,
                                  id=f"{GROUP_ID_PREFIX}{assumption or 'condition'}"))
            options.extend(
                Option(_method_row(r, label_width, cite_width), id=r.recipe,
                       disabled=not r.can_run)
                for r in group
            )
        # `compact=True` drops `OptionList`'s own border and padding — Textual ships the class
        # for exactly this (`.-textual-compact` in its `DEFAULT_CSS`), and the frame below is
        # already the box §3.2 draws. Two nested borders is one more than the mockup has.
        rows = OptionList(*options, id="ledger-rows", compact=True)
        frame = Vertical(rows, id="ledger-frame")
        frame.border_title = f"what you can ask {self.entry.name}"
        # the dataset stays identified after the roster is gone, in its own words:
        # `DataEntry.size` and `DataEntry.contents` (which is `narrate.contents`).
        frame.border_subtitle = f"{self.entry.size} · {self.entry.contents}"
        yield frame
        # The legend says what the hollow rows ARE — without it a dimmed unreachable row reads as
        # a bug in the list rather than as a statement about this data. It appears ONLY when
        # there is one to explain: a legend for a symbol that is not on screen is the same kind
        # of noise as §3.1's banned capability column, and on fully-labelled data every row is
        # filled and there is nothing to explain.
        parts = []
        if any(r.recommended and r.can_run for r in self.rows):
            parts.append(f"{RECOMMENDED_MARK} suits this data")
        if any(not r.can_run for r in self.rows):
            parts.append(f"{BLOCKED_MARK} your data lacks something")
        if any(r.measured == "caution" for r in self.rows):
            parts.append(f"{CAUTION_MARK} runs, but poorly")
        legend = ("     " + " · ".join(parts)) if parts else ""
        yield Static(f"↑↓ move · enter runs it · esc back{legend}", id="ledger-hint")

    def on_mount(self) -> None:
        """Restore the previous recipe when runnable, otherwise select the first runnable row.

        `action_first` is `find_first_enabled`, so the highlight never starts on a hollow row —
        and because `state.ledger` sorts runnable-first with `narrate.offer`'s order intact
        inside each group, that first row is `offer`'s recommendation. `highlighted` defaults to
        `None` in Textual, which would mean the first `enter` did nothing.
        """
        rows = self.query_one(OptionList)
        rows.focus()
        rows.action_first()
        if self.selected_recipe is not None:
            for index in range(rows.option_count):
                option = rows.get_option_at_index(index)
                if option.id == self.selected_recipe and not option.disabled:
                    rows.highlighted = index
                    rows.scroll_to_highlight()
                    break

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """A runnable row was chosen — hand the recipe name back and let the caller start it.

        This handler cannot be reached for a hollow row: `action_select` and `_on_click` both
        check `option.disabled` before posting (see the module docstring). The `id` is the
        recipe name, which is what `shell._run` takes as `recipe_name`.
        """
        event.stop()
        self.dismiss(event.option.id)

    def action_back(self) -> None:
        """Escape returns `None` — no recipe chosen, and the caller decides what back means."""
        self.dismiss(None)
