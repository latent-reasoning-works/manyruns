"""§3.3 — the run screen. Steps and figures as `on_step` fires, and an answer beside them.

**Layout and event wiring, not new rendering.** Every fact on this screen was computed before
it got here: the glyph and the word are `narrate.step_mark` (through `StepView`), the per-step
deltas are `narrate.geometry_delta_rows`, the settled g-vector `geometry_view` still holds is
`narrate.geometry_sections` (no pane draws that one after this commit — see below), and the
fold from `on_step` reports to rows is `tui.state.RunFeed`. This module picks ink and
puts things in boxes. If something here starts computing a fact, it belongs in `narrate` — a
fourth account of what a run did is what that rule exists to prevent.

**The top-right pane holds an ANSWER now, not a second account of the record.** It was
`#geometry-pane`, titled *what this run recorded*: what the last tune moved, over the settled
g-vector. THE TWO HALVES ARE NOT ALIKE, and the paragraph after this one turns on the
difference. The settled g-vector is narrate's (`geometry_sections`) and it survives off this
screen: `shell._geometry_panel` draws the same sections on the one-shot surface, `summary.md`
carries it off the record, and every per-step delta is drawn per step in `#steps` and
serialized whole by `narrate.record_for_prompt` for a model that is handed the run. Here it was
a second, narrower account of a record the trace layer now holds, so the slot is the ask's (spec
`2026-09-01-ask-the-record`, §0 and §5). The tuned pair is the other half and it is neither
narrate's nor drawn anywhere else: `tui/app.py`'s header names `_tuned_apart` as the ONE fact
that module computes outside narrate, `record_for_prompt` takes no pair, and `summary.md`
writes the g-vector only. Losing this pane loses that readout outright, which is a cost this
commit takes rather than a fact it relocates. The pane sits at `AT_REST` until somebody asks,
under a border carrying the name of the model that would answer (`agents.ask_model()`, set in
`on_mount`; the argument for naming it before it has said anything is there).

**A `?` LINE IS A QUESTION, AND IT NEVER ANSWERS THE GATE.** The one `Input` on this screen
carries three vocabularies now — the loop's verbs (`a`/`t`/`c`), a parameter (`knn=40`) and a
question — and the third has to be unmistakable at the FIRST character, because `gate.resolve`
passes anything it does not know through as itself: before this the typed question
`should I trust leiden here` reached `run_tune_loop` as an answer and came back as "didn't
understand". So `on_input_submitted` checks the prefix before `gate.answer` can consume the
pending decision, and returns. The field stays enabled, the row stays
up, the strip stays armed and the gate stays pending, so a person can ask, read, and then
answer. That is the whole point of asking mid-loop, and it is why a question does not share an
answer's exit path.

**The answer is STATE (`asked`/`answer`/`ask_note`), painted by `sync()` like everything else
here.** That is commit 4's correction taken at its word: a value that must survive a repaint
becomes state rather than getting its own paint path, and this one has to survive several — the
run keeps reporting steps while a model is thinking, and every report repaints. The model call
itself is off the event loop on `@work(thread=True, exclusive=True, group="manyruns-ask")`,
`find.py:295`'s pattern with its own group; what the worker is handed is a record rendered HERE,
on the event loop, BEFORE it starts (`ask_context`). A snapshot, deliberately: `context_sha256`
then names the screen as it was when the question was asked rather than as it became while the
model thought about it, and the worker touches no screen state from a thread.

**What that leaves, said out loud rather than left to be found.** `geometry_view` below still
renders and nothing on this screen calls it: the settled g-vector has no ink here any more,
because it is the record and the record is the sink's (§2) — `shell._geometry_panel` still
draws the same sections on the one-shot surface and `summary.md` carries it off the record. It
is not deleted, because `tests/test_tui_integration.py` requires `narrate.geometry_sections` to
appear in this file and that guard is not this commit's to move. `moved_view` IS called, from
`sync()`, and this paragraph said otherwise until now — a leftover from commit 4's first cut,
which dropped the tuned pair with the g-vector and was corrected: the pair is the tune loop's
FEEDBACK rather than a second account of the record, and a loop that shows nothing after `tune`
is not a loop.

WHAT THE PAIR COSTS IS WORTH KEEPING ON THE PAGE. `tui/app.py` calls `_tuned_apart` on every
accepted tune, which loads two `figspec` sidecars and measures each through `suite.measure`:
0.82 s for the pair on the 800-row `data/tree8.h5ad`, and **~7.9 s** (3.93 s per call, measured
2026-08-31) on act 2's 2,000-gene fixture, which is the demo path. Eight seconds of the demo
buys the two lines that now sit above the answer. Whether that trade is right belongs to
whoever owns `tui/app.py`; it is written down here rather than left to be found by someone
wondering where the time went.

Step reports repaint through `on_step`. Final geometry has no intermediate step reports,
so its status uses a UI timer for elapsed time until the run finishes or the screen leaves.

**What is here and what is not.** §3.3 names three things — steps, geometry, figures — and
this draws two of them, plus the run's *caveats* and the answer that took geometry's slot. The
caveats were a stated gap in this file's first version — `run_panel` computed them
inline, so a screen could only have them by re-deriving them, which is the fourth account the
rule forbids. They are here now because integration took the fix the gap named: `narrate.caveats`
exposes them as a list, `run_panel` reads it, and this screen reads the same list. They are not
decoration. The first entry is normally `engine=mock — every number here is invented; no data was
read`, which is the one thing on screen distinguishing a fabricated g-vector from a measured one.

**Finished above, live below.** Top to bottom: heading, caveats, the three panes, then what the
loop said (`#said`), the parameter strip (`#tune-strip`) and the question (`#ask`). The settled
record sits over the thing being acted on, and the input sits where the cursor is — the bottom
edge, as at `manyruns>`. This is an ORDER, not yet a static/live REGION: `sync` still repaints
every pane on every event, and `#ask` is not docked — it is at the bottom because `#panes` is
`1fr` under a vertical layout and takes the rows the live widgets leave. Making the top half
stop repainting and the bottom half a true docked region is a separate change.
"""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from manyruns.agents import Answer

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Static

from manyruns import narrate
from manyruns.tui.images import AutoImage
from manyruns.tui.state import RunFeed, StepView

#: Outcome → colour. NOT a third table: this is `shell._OUTCOME_STYLE`, the one
#: `test_watch.test_every_renderer_covers_the_whole_outcome_vocabulary` already pins against
#: `watch.OUTCOMES` in both directions. Reusing it is the same argument `state.roster` makes for
#: calling `shell._resolve_path_source` — reimplementing is how two surfaces come to disagree,
#: and here they would disagree about what a failed step LOOKS like. Only the colour is taken;
#: the glyph comes from `narrate.step_mark` through `StepView.glyph`, so "how it ended" still
#: has exactly one home. Both are plain rich style names, which is what a rich renderable inside
#: a Textual widget wants — Textual's `$error`/`$success` tokens are CSS, not markup.
from manyruns.shell import _OUTCOME_STYLE, _PENDING

#: The two live states, in the same ink the rich surface uses. `_PENDING`'s colour is `dim`;
#: `running` is cyan in `_progress_panel`. Not merged into the outcome table for the reason
#: `narrate._LIVE_MARKS` states: `running` is a live state, not an outcome.
_LIVE_STYLE = {"queued": _PENDING[1], "running": "cyan"}

#: What the pane says under the picture. Both name a KEY, because a pane that only reports a
#: limitation is a dead end and the whole point of the line is the next move.
#: The keys, in one place: the pane and the no-inline tier name the same four.
#: THIRTY COLUMNS, and the budget is the reason. The pane is 36 columns wide at a 100-column
#: terminal (measured; 44 at 120, 60 at 160), and this line wrapping to two rows costs the
#: picture one of the four it has. So the keys are abbreviated rather than spelled: `f full`
#: over `f full screen`, and no "thumbnail" prefix — that `f` offers a bigger view is what says
#: this one is small.
FIGURES_KEYS = "\u2190/\u2192 \u00b7 f full \u00b7 s save \u00b7 o open"
FIGURES_THUMBNAIL = FIGURES_KEYS
FIGURES_NO_IMAGES = "written to disk — this install has no image support"
#: No graphics protocol. Nothing is drawn (see `figure.drawable`), so this line is all
#: there is — and it names the two keys that still reach the figure.
FIGURES_NO_INLINE = f"not drawn here \u00b7 {FIGURES_KEYS}"

# ── the ask ──────────────────────────────────────────────────────────────────────────────────
#: The one character that makes a question a question. A PREFIX rather than a keybinding,
#: because the gesture has to work while an `Input` has focus and swallows this screen's
#: bindings — which is exactly when someone wants to ask (the gate is waiting, the field is
#: enabled). One character, checked before anything else parses the line, so no other
#: vocabulary on this field can claim it: `resolve` normalizes recognized verbs only, and
#: the override parser reads assignments or numeric phrases; neither handles a leading `?`.
#:
#: The editor is enabled for a live gate question or an explicitly opened standalone ask.
#: `?` in pane navigation opens a standalone ask when the editor is closed; in the editor it
#: is ordinary text. A leading `?` routes submission to the record question channel, leaving
#: the tuning gate unanswered. Text arrows edit; priority F7/F8 remain available for figures.
#: Escape leaves text for pane navigation, without answering the gate.
ASK_PREFIX = "?"

#: What the model is told it is doing, and it is HERE rather than in `narrate` on the seam
#: `narrate.record_for_prompt` already draws: that function renders the RECORD, which every
#: surface shares, and this is one surface's instruction to one model about how to answer into a
#: pane 36 columns wide. A second surface asking a different way would change this string and
#: not that one.
#:
#: THREE SENTENCES IS A LAYOUT FACT, not a style preference — and the binding case is the one
#: with a picture in it. Measured at 100×30: the pane's inside is 36 columns by 18 rows bare, and
#: 36 by **11** the moment a figure lands, because `.has-figure` takes it from `height: 3fr` to
#: `1fr` (44 and 60 columns at 120 and 160, 18/11 rows at both). Eleven rows is what three
#: sentences have to fit in, and prose past it scrolls under a fold nobody knows is there.
#: "Say the record does not say" is §8's mitigation carried into the prompt: the
#: one risk this feature has is a generated sentence sitting beside measured deltas in the same
#: typeface, and a model that will admit a gap is the cheapest half of the answer to it.
ASK_SYSTEM = (
    "You are reading the record of one analysis run, printed below exactly as the person "
    "running it sees it on screen. Answer their question from that record and from what you "
    "know about single-cell analysis. Quote the numbers you rely on. If the record does not "
    "say, say so rather than guessing. Three sentences at most \u2014 no headings, no lists."
)

# The idle row names the server; a completed answer carries its own client attribution.
ASK_BACKEND = "ollama"


class ParametersPane(Static, can_focus=True):
    """A focus target for choosing parameters and composing changes."""


class FiguresPane(Vertical, can_focus=True):
    """The figures pane: the newest figure as real pixels, with every figure listed by path.

    **It no longer measures itself, and losing that machinery is the point.** The previous
    version computed a row budget — subtract the caption lines, subtract the wrapped path, ask
    whether what is left `fits` — and passed `(cols, rows)` into a renderer that drew into a
    fixed grid. That produced a real ordering bug (`sync` toggled `.has-figure`, which changes
    this pane's height, then read the OLD height in the same call and reported a figure that had
    room as too small to draw) and it produced it because a widget was being told its size by
    something that could not know it yet.

    `AutoImage` is a widget with a CSS height. The layout engine solves it, on its own schedule,
    at whatever size the pane turns out to be — so there is no budget to compute, no resize
    handler, and no ordering to get wrong.
    """

    #: `#figures` is `height: 3` and NOT `auto`, and that one word is the bug this pane had.
    #: `auto` beats `1fr` when they compete, so a caption that grew by a line per figure ate the
    #: picture: measured on a 120x30 terminal, the pane is 7 rows and `#figure-thumb` resolved to
    #: **1 row** at one figure, 1 at three, 1 at six, while the listing ran to 27 rows and the
    #: whole pane scrolled. The figure was squeezed to a sliver by its own caption and it got
    #: worse with every step that drew — which is what "the pane smushes onto itself" looks like
    #: from the outside. Three fixed rows is the index, the path and the keys; everything else
    #: belongs to the image.
    #:
    #: `auto` WITH A CAP rather than a fixed `height: 3`, because the last line is a key hint and
    #: a hint wraps: at a 36-column pane the four keys take two rows, and a fixed 3 would clip
    #: the one line telling a reader which keys exist. The cap is what makes `1fr` solvable —
    #: `auto` alone is what starved it.
    DEFAULT_CSS = """
    FiguresPane #figure-thumb { width: auto; height: 1fr; min-height: 3; }
    FiguresPane #figures { height: auto; max-height: 4; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        #: The rows this pane last drew. Kept so the screen can ask what is on screen without
        #: re-deriving it — `action_save` needs the selected figure and its step.
        self.rows: list[StepView] = []
        #: Which figure is on screen. `-1` means "the newest", and it is a SENTINEL rather than
        #: an index so that a live run keeps advancing the picture as steps draw. The moment the
        #: user presses an arrow it becomes a real index and stops moving — a reader looking at
        #: step 2 must not be yanked to step 5 because the run finished. Same rule a log tail
        #: follows, and the reason `select` exists at all.
        self.selected: int = -1

    def compose(self) -> ComposeResult:
        yield AutoImage(id="figure-thumb")
        yield Static(id="figures")

    @property
    def found(self) -> list:
        from manyruns.tui.figure import figures_of

        return figures_of(self.rows)

    def at(self) -> int:
        """The RESOLVED index of the figure on screen — never the sentinel, never out of range.

        Clamped rather than wrapped: a run that loses a figure between repaints should show the
        nearest one, not jump to the other end of the strip.
        """
        found = self.found
        if not found:
            return 0
        return len(found) - 1 if self.selected < 0 else max(0, min(self.selected, len(found) - 1))

    def select(self, delta: int) -> bool:
        """Walk the index. Returns whether anything moved, so the screen can stay quiet if not.

        Stops at the ends instead of wrapping. The strip is an ORDER — it is the trajectory the
        embedding took through the recipe — and an order whose last element is next to its first
        is not one; a reader stepping right off the end and landing on `phate` would read that
        as the run having looped.
        """
        found = self.found
        if not found:
            return False
        target = max(0, min(self.at() + delta, len(found) - 1))
        if target == self.at() and self.selected >= 0:
            return False
        self.selected = target
        self.show(self.rows)
        return True

    def show(self, rows: list[StepView]) -> None:
        from manyruns.tui.figure import drawable

        self.rows = list(rows)
        found = self.found
        # ONE picture, and now it is the SELECTED one rather than always the newest. This pane is
        # one fraction of one column, so two pictures in it are two things too small to read
        # instead of one; which one it shows is the arrow keys' business.
        #
        # `drawable` is None on a cell tier — 864 px for a 600x480 figure — so nothing is drawn
        # rather than something a reader has to distrust. See its docstring for the numbers.
        self.query_one("#figure-thumb", AutoImage).image = \
            drawable(found[self.at()][1]) if found else None
        self.query_one("#figures", Static).update(figures_view(rows, self.at(), self.size.width))


class LeaveRunScreen(ModalScreen[bool]):
    """Confirm detachment without promising worker cancellation."""

    BINDINGS = [Binding("escape", "stay", "stay", priority=True)]
    DEFAULT_CSS = """
    LeaveRunScreen { align: center middle; background: $background 70%; }
    LeaveRunScreen > Vertical {
        width: 70; max-width: 95%; height: auto; padding: 1 2; border: round $accent;
    }
    LeaveRunScreen Static { height: auto; margin-bottom: 1; }
    LeaveRunScreen Horizontal { height: auto; }
    LeaveRunScreen Button { margin-right: 1; }
    """

    def __init__(self, *, draining: bool) -> None:
        super().__init__()
        self.draining = draining

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Run again?", id="leave-title")
            yield Static(
                ("Leaving closes the tuning gate. An in-flight step or measurement may finish; "
                 "later steps and final metrics that have not started are skipped. "
                 "The run is recorded as incomplete.\n\n" if self.draining else
                 "This run has ended.\n\n")
                + "Return to the ledger with this recipe selected. Choosing it starts a fresh run.",
                id="leave-explanation")
            with Horizontal():
                yield Button("Stay", id="stay", variant="primary")
                yield Button("Run again", id="run-again")

    def on_mount(self) -> None:
        self.query_one("#stay").focus()

    def action_stay(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, message: Button.Pressed) -> None:
        self.dismiss(message.button.id == "run-again")


class RunScreen(Screen):
    """One run, watched. Owns a `RunFeed` and repaints it as reports land.

    Two ways to drive it, and the second is the one the app uses:

        RunScreen(recipe)                 the caller owns the run and pushes reports into
                                          `screen.feed.on_step`
        RunScreen(recipe, start=fn)       the screen owns it: `fn(on_step) -> results` runs in
                                          a thread worker

    `start` is a callable of ONE argument because that is the shape every existing run path
    already takes — `Session(on_step=…)`, `runner.run_inproc(on_step=…)`,
    `app.run_explorations(on_step=…)` — so the caller binds its own data, engine and out_dir and
    hands over the one seam. It runs on a worker thread because the fit blocks: on the event
    loop a 2.3 s recipe would freeze input and repaint for 2.3 s, which is the whole thing the
    screen exists to avoid.

    The marshal back is `post_message`, not `call_from_thread`. Both are thread-safe, and
    `post_message` is the one that also works when the reporter is ALREADY on the app's thread —
    it branches on `self._thread_id != threading.get_ident()` internally, where
    `call_from_thread` raises `RuntimeError` in that case. A caller driving the feed
    synchronously from a test is exactly that case.

    `Finished` is not stopped, so it bubbles to the app: that is how whoever pushed this screen
    learns the run ended and gets the results dict.
    """

    BINDINGS = [
        Binding("escape", "back", "run again", show=True, priority=True),
        Binding("f8", "select(1)", "next figure", show=False, priority=True),
        Binding("f7", "select(-1)", "previous figure", show=False, priority=True),
        Binding("f", "figure", "figure", show=True),
        Binding("s", "save", "save for a paper", show=True),
        Binding("o", "open", "open the file", show=True),
        Binding("c", "color_by", "colour by metadata", show=True),
        # THE ASK, ON A RUN NOBODY IS BEING ASKED ABOUT. `question_mark` is Textual's own name
        # for this key (`textual.keys._character_to_key("?")`), spelled that way because the
        # binding table matches `event.key` and a `"?"` here would never fire.
        #
        # It is reachable exactly where `ASK_PREFIX` is not: a focused `Input` swallows screen
        # bindings, so while the gate is waiting this binding does not exist and the character
        # goes into the field — which is the channel. The two halves never both fire, and that is
        # the mechanism rather than a coincidence to be careful about.
        Binding("question_mark", "ask", "ask the record", show=True),
    ]

    DEFAULT_CSS = """
    RunScreen { layout: vertical; }
    RunScreen #pane-focus { height: auto; padding: 0 2; color: $accent; }
    RunScreen #tune-strip:focus { background: $boost; }
    RunScreen #figures-pane:focus { border: heavy $warning; }
    RunScreen #heading { height: auto; padding: 0 2; }
    /* Above the panes, not inside one: a caveat is about the whole run, and putting it in the
       top-right pane would make it read as a note on whatever that pane holds — one number
       when it held the g-vector, one answer now. `height: auto` collapses it to nothing while
       there is nothing to say. */
    RunScreen #caveats { height: auto; padding: 0 2; }
    /* WHY THE LIVE ROWS SIT ON THE BOTTOM EDGE: under a vertical layout `1fr` takes every row
       the `height: auto` widgets after `#panes` do not, so `#said`, the strip and the ask row
       land on the bottom with nothing docked. Both this rule and `RunScreen { layout: vertical; }`
       above RESTATE Textual 8.2.8 defaults (`Screen { layout: vertical }`, `Horizontal { height:
       1fr }`) — written out so the mechanism is on the page and does not ride on a library
       default; dropping both changes nothing today (measured: every region identical at 120×30). */
    RunScreen #panes { height: 1fr; }
    /* Widths as FRACTIONS, which is §2's point: `render_result` wraps because it hardcodes
       `width=30`, and a fraction is a constraint the layout engine re-solves at every size.
       3:2 rather than 2:1 for what the narrow side costs, measured at a 100-column terminal
       with the panes below: 2:1 leaves the geometry content 30 columns and 3:2 leaves it 36,
       where the section captions and the metric rows both stop wrapping (14 non-blank lines
       become 13, and `the only real before/after here` stops breaking after `before/after`).
       The steps side loses those 6 columns from its detail text, which folds rather than
       truncating and so loses nothing. THAT MEASUREMENT IS THE OLD CONTENT'S, and the fraction
       is kept rather than re-picked now that the 2fr side holds prose: an answer folds at any
       width, so nothing measured says a different split reads better. Re-measure it against a
       real answer or leave it alone — a fraction changed without a measurement is `width=30`
       in a different unit. */
    RunScreen #steps-pane { width: 3fr; border: round $accent; padding: 0 1; }
    RunScreen #side { width: 2fr; }
    RunScreen #answer-pane { height: 3fr; border: round $secondary; padding: 0 1; }
    RunScreen #figures-pane { height: 1fr; border: round $warning; padding: 0 1; }
    /* The live rows, in draw order — they follow the panes in `compose`, so they follow them
       here. Moved down for READABILITY only: every selector is a single id and no property
       overlaps another rule's, so the order of these blocks changes nothing (checked).
       `height: auto` for `#caveats`' reason: nothing to say collapses to nothing, so a run that
       is not being stepped is laid out exactly as it was before the gate existed. */
    RunScreen #said { height: auto; padding: 0 2; color: $text-muted; }
    RunScreen #tune-strip { height: auto; padding: 0 2; }
    RunScreen #ask { height: auto; padding: 0 2; }
    RunScreen #ask-prompt { width: auto; padding: 0 1 0 0; }
    RunScreen #ask-input { width: 1fr; }
    /* At 80 columns the prompt plus unavailable status consumed the entire field. On the
       narrow layout each gets its own row, so the input keeps the row's usable text cells. */
    RunScreen #ask-model { width: auto; padding: 0 0 0 1; }
    RunScreen.narrow #ask { layout: vertical; }
    RunScreen.narrow #ask-prompt { width: 1fr; height: auto; }
    RunScreen.narrow #ask-model { width: 1fr; height: auto; padding: 0; }
    /* A pane reading "no figures yet" needs one line; a pane with a picture in it needs every
       row it can get, and at 3:1 it got 7. The split follows the content: `sync` sets
       `.has-figure`. The answer loses less by it than the g-vector did — the pane still
       scrolls, and prose re-flows into whatever height is left, which twelve fixed metric rows
       could not do. */
    RunScreen.has-figure #answer-pane { height: 1fr; }
    RunScreen.has-figure #figures-pane { height: 1fr; }
    /* Below `NARROW` the two columns become two rows — see `on_resize`. */
    RunScreen.narrow #panes { layout: vertical; }
    RunScreen.narrow #steps-pane { width: 1fr; height: 1fr; }
    RunScreen.narrow #side { width: 1fr; height: 1fr; }
    """

    #: Terminal width below which side-by-side stops paying, measured on the same eleven-row
    #: g-vector: the geometry pane draws it in 11 non-blank lines at 76 columns, 14 at 36 and 15
    #: at 28. 36 is what a 100-column terminal gives the pane at 3:2; 28 is what an 80-column one
    #: would, and at 28 even `geodesic_distance_correlation` breaks across two lines. Stacked, an
    #: 80-column terminal gives that same pane 76 and nothing wraps at all — so below 100 the two
    #: columns become two rows. Nothing is ever LOST at any of these widths (see `steps_view`);
    #: this is about how much of the screen a reader has to reassemble.
    #:
    #: The g-vector is not in that pane any more and the threshold is kept on its measurement
    #: anyway: the WIDTHS did not move (44 at 120 columns, 76 stacked at 80 — the two numbers
    #: `test_the_panes_stack_when_the_terminal_is_too_narrow_to_divide` pins), only what the 2fr
    #: side holds, and the steps pane either side of the split is untouched.
    NARROW = 100

    #: What the answer pane says with nothing in it. A STATE, not a promise, which is the choice
    #: `geometry_view`'s "nothing recorded yet" made in the same slot for the same reason: a pane
    #: that wrote "your answer appears here" would describe a channel that did not exist when it
    #: was written. This sentence is true at every commit in the sequence, before the channel and
    #: after it: nothing has been asked.
    AT_REST = "nothing asked yet"

    #: The three things the pane says INSTEAD of an answer, each a state rather than a promise
    #: for `AT_REST`'s reason. All three are dim where an answer is not, which is the only ink
    #: separating "the model said this" from "nothing said anything" once a question is on screen
    #: above them.
    #:
    #: `UNAVAILABLE` is §4's controller-row wording minus its `ask ·` prefix, spelled once here so
    #: that the row the next commit adds reads this constant instead of typing a second copy — a
    #: pane and a row disagreeing about whether a model is reachable is exactly the two-accounts
    #: failure this file's header opens with. Under a border already titled `qwen3:8b`, the word
    #: has its subject.
    #:
    #: `UNANSWERED` is the case `ask_available` cannot see: something IS listening on the port and
    #: `agents.answer` still came back with nothing — a model name that was never pulled, a
    #: timeout, or a thinking budget spent with `content` left empty (measured in `agents`: 9.1 s
    #: and an empty string). It names the OUTCOME and not a cause, because `answer` collapses all
    #: of those to `None` and a screen that picked one would be guessing on the record's behalf.
    THINKING = "thinking…"
    UNAVAILABLE = "unavailable — start ollama"
    UNANSWERED = "the model gave no answer"

    #: The controller row's third segment at rest — the BACKEND, because the two facts either
    #: side of it are the model (segment two) and the fact that this is the ask (segment one),
    #: and what is left to say about a channel nobody has used yet is where it goes. §4 spells
    #: the whole row: `ask · qwen3:8b · ollama`.
    #:
    #: `UNANSWERED` has no counterpart HERE and that is the division between the row and the
    #: pane. A server that accepted the socket and returned nothing is a fact about the ANSWER —
    #: the pane says it, under a border naming the model — while the row's question is only
    #: whether there is a channel and how long it took to come back. So an empty answer leaves
    #: the row reading `ask · qwen3:8b · 9.1 s`, which is the true and useful thing: it took nine
    #: seconds and the pane will tell you it said nothing.
    AT_HAND = ASK_BACKEND

    class Progress(Message):
        """A step was reported. Carries NO payload: the feed is the payload, and a message
        holding a copy of the rows would be a second account of the run that could go stale
        between the post and the handler."""

    class Ask(Message):
        """The run is waiting for a decision. Carries the loop's own prompt text verbatim, so the
        screen never invents wording for a question `tune.run_tune_loop` asked."""

        def __init__(self, prompt: str, kind: Optional[str] = None,
                     question_id: Optional[int] = None) -> None:
            super().__init__()
            self.prompt = prompt
            self.kind = kind
            self.question_id = question_id

    class GateClosed(Message):
        """The gate can no longer accept decisions."""

    class Said(Message):
        """The loop said something without asking — a step summary, a kept-attempt rename."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class Arm(Message):
        """The step about to be tuned. Carries the step DICT, because the strip is built from
        its params and a prompt string is not a step."""

        def __init__(self, step: dict) -> None:
            super().__init__()
            self.step = step

    class Moved(Message):
        """The g-vector either side of a tuned step."""

        def __init__(self, before: dict, after: dict) -> None:
            super().__init__()
            self.before, self.after = before, after

    class Answered(Message):
        """A `?` line came back — with prose, or with nothing.

        CARRIES EVERYTHING THE EVENT NEEDS, and that is the reason it carries the question and
        the context it was already given rather than letting the handler read them off `self`.
        The worker is `exclusive=True`, so a second question cancels the first — and a screen
        that stamped an event from its own current state could pair question 2 with answer 1
        after nothing worse than a fast typist. What was asked, what was seen and what came back
        travel together or the record is not a record.

        `answer=None` means the model said nothing — every failure `agents.answer` swallows
        (no server, an unpulled model, a timeout, an empty thinking turn) arrives as this one
        value, and `available` is the only distinction the screen can honestly draw between
        them.
        """

        def __init__(self, question: str, answer: Optional[Answer | str], *, dispatch_id: int, context: str,
                     model: str, latency: float, available: bool = True) -> None:
            super().__init__()
            self.dispatch_id = dispatch_id
            self.question, self.answer, self.context = question, answer, context
            self.model, self.latency, self.available = model, latency, available
            from manyruns import agents

            self.route = answer if isinstance(answer, agents.Answer) else None
            if self.route is not None:
                self.answer, self.model = self.route.text, self.route.model

    class Finished(Message):
        """The run ended — cleanly (`results`) or not (`error`). One message for both, because
        a screen that only hears about success leaves the app with no way to tell a run that is
        still working from one that died, and navigation stranded on the second.

        What it deliberately does NOT do is settle the steps. A record still reading `running`
        when the error arrives is left alone: `apply_step` never raises, so such a record means
        the run died somewhere other than in that step, and rewriting it to `error` would be the
        screen inventing an outcome for a step nothing measured."""

        def __init__(self, results: Optional[dict], error: Optional[str] = None) -> None:
            super().__init__()
            self.results = results
            self.error = error

    def __init__(self, recipe: Optional[dict] = None,
                 start: Optional[Callable[[Callable[[dict, list], None]], dict]] = None,
                 *, gate: Optional[Any] = None,
                 on_worker: Optional[Callable[[threading.Thread], None]] = None,
                 feed: Optional[RunFeed] = None, heading: Optional[str] = None,
                 # WHAT THE MODEL IS HANDED THAT THIS SCREEN DOES NOT DRAW. The record in the
                 # prompt is spec §3's list, and two of its items are facts the run screen never
                 # had: the dataset row (name, size, contents and the QC numbers the roster
                 # measured) and the caution the LEDGER drew over this recipe. Both are known to
                 # whoever pushed this screen — `tui/app.py:295` already holds the `DataEntry`
                 # and passes `entry.name` into `heading` — and neither can be re-derived here
                 # without a second measurement of a file that may have changed, which is the
                 # failure `record_for_prompt`'s "the model reads what the person read" exists
                 # to prevent.
                 #
                 # THEY ARE NOT WIRED YET, and what follows is the wire itself rather than a
                 # note that one is wanted — commit 5's scope is `tui/run.py` and its tests, so
                 # the caller is a separate change, and a change nobody can copy out of the file
                 # that names it is how a seam stays half-built:
                 #
                 #     # tui/app.py, `open_run`, at the `push_screen` that is already there
                 #     from manyruns import narrate
                 #     from manyruns.tui import state
                 #     self.push_screen(RunScreen(recipe, gate=gate, start=…, heading=…,
                 #                                entry=entry,
                 #                                concerns=narrate.concerns(
                 #                                    recipe, state.measurement(entry))))
                 #
                 # Both calls already exist in the package — `tui/ledger.py:370` calls
                 # `state.measurement(entry)` and `tui/state.py:468` calls
                 # `narrate.concerns(recipe, measured)` to draw the ledger's own caution — so
                 # that line re-uses the two accounts a person already read instead of making a
                 # third one for the model.
                 #
                 # WHAT ITS ABSENCE COSTS, measured rather than described: `record_for_prompt`
                 # with `entry=None` renders the dataset section as SIX dashed values —
                 # `dataset:`, `size:`, `what's in it:`, `genes/cell:`, `% mito:`,
                 # `filter drops:` (it writes the headings and dashes the values, because an
                 # absent fact is never a zero) — and no caution block at all. So the channel
                 # works and the spec's own demo question (`should I trust leiden on this
                 # before filtering?`) cannot be answered from a run started in the app: its
                 # whole answer is the QC numbers, and they are not in the prompt yet.
                 entry: Optional[Any] = None, concerns: Optional[Any] = (),
                 # `name` / `id` / `classes` are Textual's own widget arguments, forwarded
                 # unchanged so this screen can be styled and queried like any other.
                 name: Optional[str] = None, id: Optional[str] = None,
                 classes: Optional[str] = None) -> None:
        super().__init__(name=name, id=id, classes=classes)
        import weakref

        from manyruns.tui.figure import QuickdropWorker

        self._export = QuickdropWorker(self)
        self._figure_exports = weakref.WeakSet([self._export])
        self._extra_views: dict[tuple[int, str], list[str]] = {}
        self._color_busy = False
        self._color_cancel = threading.Event()
        # KEPT, where before only `feed` and `heading` took anything from it. The recipe's name,
        # its `claims:` and its step chain are three of the record's lines, and `RunFeed` keeps
        # `declared` (the steps) rather than the recipe, so the feed cannot answer for it.
        self.recipe = recipe or {}
        self.entry = entry
        self.concerns = list(concerns or ())
        self.feed = feed if feed is not None else RunFeed(recipe)
        # Wired even when the caller supplied the feed: a feed nobody listens to renders once at
        # mount and never again, which looks exactly like a hung run.
        self.feed.listener = self._reported
        self._start = start
        self.worker_thread: Optional[threading.Thread] = None
        self._on_worker = on_worker
        if gate is not None:
            gate._issuance = threading.Event()
            gate._issuance.set()
        # THE GATE IS THE CALLER'S, not this screen's — `start` closes over it, and `start` is an
        # argument here, so a gate this screen made could not be reached from inside the run.
        # Callbacks use `post_message`: safe from either thread, and dropped on a closed
        # pump so a detached run cannot paint a screen that is
        # gone (the same property `_reported` relies on).
        self.gate = gate
        self.said: list[str] = []
        # Current-attempt values only change when the loop republishes its step.
        self.tunables: list = []
        self.tune_constraints: dict = {}
        self._tune_params: dict = {}
        self.tune_at: int = 0
        self._pending = ""
        self._question_kind: Optional[str] = None
        self._question_id: Optional[int] = None
        self._finished = False
        self._finalizing_started = None
        self._finalizing_timer = None
        #: The g-vector either side of the last tuned step, as `(before, after)` — or None while
        #: nothing has been tuned. It is a REPORT the driver sent, not something this screen
        #: derives: only the stepped driver knows where one step's tuning began and ended, and
        #: `feed`'s rows carry per-step deltas rather than per-DECISION ones.
        self.moved: Optional[tuple] = None
        #: Which step the pair above is about, from `gate.arm`. Empty until something is armed.
        self.tune_step: str = ""
        #: THE ASK, AS STATE — commit 4's correction applied to the value it was written for.
        #: `sync()` repaints this pane on every step report, and a run keeps reporting while a
        #: model is thinking, so an answer painted straight into the widget would be wiped by
        #: the next `on_step`. Three fields rather than one because the pane draws two things:
        #: the question the person asked (dim, above) and either the answer (in the pane's own
        #: ink) or the reason there is none (`THINKING`/`UNAVAILABLE`/`UNANSWERED`, dim).
        #: `asked == ""` is the whole "nothing has been asked" test; `answer is None` with an
        #: `asked` set is the case `ask_note` explains.
        self.asked: str = ""
        self.answer: Optional[str] = None
        self.ask_note: str = ""
        #: The dispatch the person is WAITING for, which is not always the one that answers
        #: first. `exclusive=True` cancels a superseded worker's RECORD and not the thread it
        #: runs on, so the answers can land out of order — measured with two stubs, 1.2 s then
        #: 0.05 s: the pane finished on the first question while the person sat waiting for the
        #: second, and ask-then-ask-again is the ordinary gesture at `agents`' own measured
        #: latencies (2.4 s cold, 0.7 s warm). Kept apart from `asked` because the handler
        #: overwrites `asked` with whatever it is painting, so `asked` cannot also be the test
        #: for what is outstanding.
        self._outstanding: int = 0
        #: THE ROW'S OWN TWO FIELDS, which are not the pane's. `_ask_up` is the last answer
        #: `agents.ask_available` gave — WHETHER there is a channel — and `_ask_status` is the
        #: third segment of `ask · <model> · <status>`. They are separate from `ask_note` because
        #: the row and the pane answer different questions about the same call: the pane says
        #: what came back (`UNANSWERED` when nothing did), the row says whether anything could
        #: and how long it took. Deriving one from the other would put `9.1 s` and "no answer" in
        #: one string and force a caller to pick which of the two it meant.
        self._ask_up: bool = False
        self._ask_status: str = self.AT_HAND
        # Explicit standalone composition survives gate closure and run completion.
        self._ask_open: bool = False
        if gate is not None:
            gate.bind(on_prompt=lambda p: self.post_message(
                          self.Ask(p, gate.question_kind, gate.question_id)),
                      on_write=lambda s: self.post_message(self.Said(s)),
                      on_arm=lambda step: self.post_message(self.Arm(step)),
                      on_moved=lambda b, a: self.post_message(self.Moved(b, a)),
                      on_close=lambda: self.post_message(self.GateClosed()))
        self.heading = heading or str((recipe or {}).get("name") or "run")
        self.results: Optional[dict] = None
        self.error: Optional[str] = None

    # ── the frame ────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Static(id="heading")
        # STAYS ABOVE THE PANES. A caveat is a settled fact about the whole run (`engine=mock —
        # every number here is invented`), not something anyone acts on, so it reads with the
        # heading rather than with the prompt.
        yield Static(id="caveats")
        with Horizontal(id="panes"):
            # Scrollable, and that is a capability the rich surface does not have.
            # `shell._progress_panel` shows only the NEWEST step's geometry, and its comment
            # says why: five metrics land per embedding step and `rich.Live` has to repaint the
            # whole panel inside one screen. A scroll container has no such ceiling, so every
            # step keeps its own deltas here.
            yield VerticalScroll(Static(id="steps"), id="steps-pane")
            with Vertical(id="side"):
                # THE FREED SLOT, and freed is the whole argument for the answer landing here:
                # three sentences need room `#said`'s three lines do not have, and a pane that
                # already exists at that size is the least disruptive place to put them (spec
                # §3). `VerticalScroll` for the reason the g-vector had it — an answer can be
                # longer than the pane, and the part below the fold must be reachable.
                yield VerticalScroll(Static(id="answer"), id="answer-pane")
                yield FiguresPane(id="figures-pane")
        # THE GATE'S ROWS, BELOW THE ANALYSIS. Two reasons, both about where a reader's eyes
        # are: the last line on a terminal is the one you read, and the input belongs where
        # the cursor is — at the bottom, as at `manyruns>`. They used to sit ABOVE `#panes`,
        # and the measured cost was a jump: at 120×30 the panes' top moved from y=3 to y=12 the
        # moment a question appeared, so the analysis someone was reading slid nine rows down
        # under them. Now the panes' top stays at y=2 and only their height gives (27 → 18).
        # Nothing is docked to get them here: the screen lays children in this order (vertical
        # layout — Textual's default, restated at the top of the stylesheet) and `#panes` is
        # `1fr` (also `Horizontal`'s default, restated), so it takes every row the three
        # `height: auto` widgets below it leave, which is what pushes them to the bottom edge.
        # `#ask` and `#tune-strip` are `display=False` from `on_mount`, so a run nobody is
        # stepping is laid out exactly as it was before the gate existed. At 13 rows the input
        # is still whole (measured; below that its bottom border clips) — a live region that is
        # actually docked, and panes that stop repainting under it, are a separate change.
        yield Static(id="pane-focus")
        yield Static(id="said")
        yield ParametersPane(id="tune-strip")
        with Horizontal(id="ask"):
            yield Static(id="ask-prompt")
            # DISABLED UNTIL SOMETHING ASKS, and `display=False` on the row is not enough.
            # An `Input` is focusable, so mounting one put it in the focus chain and it swallowed
            # the screen's own bindings — measured: `f`, `s` and `o` stopped working the moment
            # this widget existed, caught by the four `test_tui_figure` tests that press them.
            # `disabled` takes it out of the chain; `display` only stops it being painted.
            # THE PLACEHOLDER NAMES ALL THREE VOCABULARIES (§5's one added clause), because it
            # is the only thing on screen that says the third exists — `?` is a character
            # nobody guesses, and a channel nobody can find is a channel nobody has.
            yield Input(id="ask-input", disabled=True,
                        placeholder="a / t / c, offered params as key=value (e.g. knn=40), "
                                    f"or {ASK_PREFIX} a question")
            # THE MODEL LINE — §4's `ask · qwen3:8b · ollama`, and a SECOND widget rather than
            # `#ask-prompt`'s text, which is what §4 named when this row held one thing. The two
            # collided the moment both were real: `#ask-prompt` carries the loop's own question
            # verbatim (`on_run_screen_ask`), so a model line written into it would delete the
            # question a person is being asked in order to say which model is idle. §5's
            # division is the resolution and it survives intact — the border says WHO answered,
            # this row says WHETHER there is anything to ask — with the gate's prompt keeping the
            # slot it already had. Painted by `_paint_ask_row` and by nothing else.
            yield Static(id="ask-model")

    def on_mount(self) -> None:
        from manyruns import agents

        self.query_one("#steps-pane").border_title = "steps"
        # TITLED WITH THE MODEL'S NAME — spec §4's own words for this pane, and §8's whole
        # mitigation: what lands here is GENERATED prose sitting beside MEASURED deltas in the
        # same typeface, and the border is the only thing on the screen that says which is
        # which. So the name goes up at mount rather than with the first answer: a pane titled
        # `answer` says nothing a reader could discount, and it would be titled that way for
        # every second before anyone asks — which is most of them.
        #
        # `agents.ask_model()` is the constant plus `$MANYRUNS_ASK_MODEL` and nothing else: no
        # socket, no backend import — 72-77 µs over three runs to import the whole module once
        # `run.py`'s own chain is loaded, quoted as a range because one figure from one run does
        # not reproduce. The name of who WOULD answer is knowable before anything has been
        # asked, and that is what makes this a title rather than a status.
        #
        # NOT gated on `agents.ask_available()`, one TCP probe away though it is, and the second
        # reason is the deciding one. It puts a connect in the mount path of every run screen —
        # free when it is refused on loopback, but `$OLLAMA_BASE_URL` pointed at a host that
        # does not answer pays `agents.ASK_PROBE_S` (100 ms) on every mount, for a title. And
        # availability CHANGES — somebody starts ollama while the run is still going — while the
        # model's identity does not. A border that flipped between a name and "unavailable"
        # would be a second, worse account of a state the controller row is spec'd to hold
        # (`ask · unavailable — start ollama`, §4). The border says WHO; the row says WHETHER.
        #
        # THE BORDER IS SET HERE AND THE PANE'S INK IS NOT, which is the one asymmetry in this
        # method worth a line. A border title is not state this screen keeps — it is written
        # once and never rebuilt — while what is INSIDE `#answer` is `self.asked` and
        # `self.moved`, and `sync()` four lines below paints it from those (empty `asked` →
        # `AT_REST`). This used to paint that same string here as well; two paths writing one
        # widget is exactly what commit 4 corrected, and the second one being harmless is not
        # a reason to leave it. Exactly one line in this file writes `#answer` now, in `sync()`,
        # which makes "one paint path" greppable rather than argued.
        self.query_one("#answer-pane").border_title = agents.ask_model()
        self.query_one("#figures-pane").border_title = "figures"
        self.query_one("#ask").display = False
        self.query_one("#tune-strip").display = False
        self.sync()
        self.query_one("#steps-pane").focus()
        self._paint_focus()
        if self._start is not None:
            self._drive()

    def on_resize(self, event: Any) -> None:
        """Two columns or two rows, decided by the width there actually is.

        §2's second borrowed mechanism, and the one place a magic number would otherwise come
        back: `render_result`'s `width=30` wraps because it is a number, and a fraction of a
        terminal too narrow to divide is the same mistake in a different unit. The layout
        engine re-solves the fractions on its own; this only decides how many columns to solve
        for.
        """
        self.set_class(event.size.width < self.NARROW, "narrow")

    # ── the events ───────────────────────────────────────────────────────────
    def _reported(self, feed: RunFeed) -> None:
        """The `RunFeed` listener. Runs on whichever thread reported the step — the worker's,
        normally — so it does the one thread-safe thing and touches no widget."""
        self.post_message(self.Progress())

    def on_run_screen_progress(self, message: "RunScreen.Progress") -> None:
        self.sync()

    def on_run_screen_ask(self, message: "RunScreen.Ask") -> None:
        """Only a still-live question may open the decision editor."""
        if (self._finished or self.gate is None
                or not self.gate.is_pending(message.question_id)):
            return
        self._question_id = message.question_id
        self._question_kind = message.kind
        self.query_one("#ask-prompt", Static).update(message.prompt)
        self.query_one("#ask").display = True
        field = self.query_one("#ask-input", Input)
        field.disabled = False
        if not self._ask_open:
            self._set_pending("")
            field.focus()
        self._look_for_a_model()
        self._paint_strip()

    def _invalidate_question(self) -> None:
        """Remove obsolete gate UI without destroying a standalone composition."""
        parameters_focused = getattr(self.focused, "id", None) == "tune-strip"
        self._question_id = None
        self._question_kind = None
        self.tunables = []
        self._tune_params = {}
        self.tune_constraints = {}
        self.query_one("#ask-prompt", Static).update("")
        self.query_one("#tune-strip").display = False
        if not self._ask_open:
            self._close_ask()
        if parameters_focused:
            self.query_one("#steps-pane").focus()

    def on_run_screen_gate_closed(self, message: "RunScreen.GateClosed") -> None:
        self._stop_finalizing()
        self._invalidate_question()

    def on_input_submitted(self, message: Input.Submitted) -> None:
        """Hand the typed answer to the waiting worker and hide the row again — unless it was a
        question, which is not an answer to anything and leaves every one of those things alone.

        `answer` returns False when nothing was pending — a stray enter between questions — and
        the row is hidden either way, because a question that is no longer being asked must not
        keep sitting on screen offering to be answered.
        """
        # Questions consume no gate answer and edit no params. Route them before the gate
        # check so a standalone ask works too; when a tune question IS pending, asking about
        # it keeps that question and the field's focus available for the eventual answer.
        typed = str(message.value).strip()
        if typed.startswith(ASK_PREFIX):
            self._ask(typed[len(ASK_PREFIX):].strip())
            return
        if self.gate is None:
            return
        from manyruns.tui.gate import resolve

        # Delivery is not acceptance. The loop republishes current-attempt values through `Arm`
        # when it next asks; a rejected or stale submission cannot move the strip.
        if self._question_id is not None:
            answer, question_id = resolve(message.value), self._question_id

            def submit():
                self.gate.answer(answer, question_id=question_id)

            # Snapshot reads must finish before the loop can rename an attempt's files.
            # Only the short capture phase defers an answer; rendering remains independent.
            continuation = submit
            for exporter in list(self._figure_exports):
                continuation = partial(exporter.after_capture, continuation)
            continuation()
        self._question_kind = None
        self._question_id = None
        self._set_pending("")
        # Disabled again, not merely hidden: the run screen's own keys have to work between
        # questions, and a focusable widget in the chain is what took them away.
        field = self.query_one("#ask-input", Input)
        field.disabled = True
        self._ask_open = False
        self.query_one("#ask").display = False
        self.query_one("#tune-strip").display = False
        # Keep current-attempt values between questions; the next Arm replaces them from the loop.
        self.query_one("#steps-pane").focus()

    # ── the ask ──────────────────────────────────────────────────────────────
    def ask_context(self) -> str:
        """Everything this screen holds, as the one page a model is handed.

        `narrate.record_for_prompt` does the rendering and this method does the GATHERING, which
        is the whole division: no line of the record is composed here, so the model reads the
        same words the panes drew. The screen's own contribution is which state goes in, and
        every item is spec §3's list — the dataset row, the recipe, `RunFeed.rows()` with every
        delta, the strip's knobs, what the loop said, and the ledger's caution.

        `said` IS SLICED to `SAID_LINES` and the record is not: §3 asks for "the last
        `SAID_LINES` the loop said", and `record_for_prompt` states outright that the cap is the
        caller's because only the caller knows what it showed. Three lines is what `#said`
        paints, so three lines is what the model reads.

        The TUI suppresses the tune loop's drawing, including its `plot: <path>` fallback.
        We accept losing those incidental paths from `said` and the resulting context-hash
        changes: they previously reached the model only when inline drawing failed.
        `record_for_prompt` does not otherwise serialize `StepView.plots`; giving the model
        a deliberate account of figures is a separate change.

        Public and pure — no widget is touched, so a test can ask what a screen would send
        without a running app, and `_ask` can call it on the event loop before the worker starts.
        """
        return narrate.record_for_prompt(
            entry=_entry_mapping(self.entry), recipe=self.recipe, steps=self.feed.rows(),
            tunables=self.tunables, constraints=self.tune_constraints,
            said=self.said[-self.SAID_LINES:], concerns=self.concerns)

    def _ask(self, question: str) -> None:
        """Dispatch a `?` line. Touches the gate not at all — see `on_input_submitted`.

        THE FIELD IS CLEARED AND LEFT ENABLED. Cleared because the question is now on screen in
        the pane and a field still holding it would be re-submitted by the next `enter` a person
        presses to accept the run; enabled because the gate may still be waiting and freestyle's
        rule is that a question never blocks a run.

        An empty question — a bare `?` — asks nothing and is treated as nothing rather than sent:
        the model would answer it (a record with no question is still a prompt) and the record
        would carry an event whose `question` is `""`. Clearing the field is the whole response,
        which is also what typing `?` and thinking better of it looks like.
        """
        self._set_pending("")
        if not question:
            return
        # NAMED BEFORE THE WORKER STARTS, so the handler can tell an answer somebody is waiting
        # for from one they have already asked past. See `on_run_screen_answered`.
        self._outstanding += 1
        self.asked, self.answer, self.ask_note = question, None, self.THINKING
        # The row says the same word as the pane while the worker runs, and it is the same
        # constant: two surfaces reporting one call must not be able to disagree about whether it
        # is still out. `_ask_up` is left where the last look put it — this is not new news about
        # whether a model is there, and the worker will bring that back with the answer.
        self._ask_status = self.THINKING
        self._paint_ask_row()
        # Painted BEFORE the worker starts, because the worker is the slow part: measured on
        # this seam in `agents`, 2.4 s cold and 0.7 s warm on an M-series laptop. A screen that
        # showed nothing for that long would read as a swallowed keypress.
        self.sync()
        self._ask_worker(question, self.ask_context(), self._outstanding)

    @work(thread=True, exclusive=True, group="manyruns-ask")
    def _ask_worker(self, question: str, context: str, dispatch_id: int) -> None:
        """One model call, off the event loop — `find.py:295`'s pattern with its own group.

        A THREAD because `agents.answer` is a synchronous HTTP call and the screen must keep
        repainting while a run reports steps underneath it. `exclusive` because a second question
        supersedes the first: Textual cancels the worker's RECORD, the thread itself runs to
        completion (`BINDINGS`' note carries the same fact about the run thread), and the late
        answer arrives as a message carrying its own question — see the handler.

        `@work` rather than the raw daemon thread `_drive` uses, and the difference is the one
        `_drive`'s docstring measures: a Textual thread worker sits on a non-daemon executor
        thread that is joined before the process can exit, which held a terminal for the length
        of a *run*. A bounded call is what makes that acceptable here, and a worker Textual
        cancels on screen removal is exactly what `@work` is for.

        BOUNDED ON ONE OF THE TWO BACKENDS, said exactly because "bounded" is the argument
        above. `agents.ASK_TIMEOUT_S` (60 s) is passed to `urllib.request.urlopen` in
        `_ask_via_ollama` — the direct path, the one this checkout and CI take, since manyAgents
        is not installed here. `agents.answer` tries `_ask_via_manyagents` FIRST, and that path
        is `asyncio.run(OllamaAdapter().chat(…))` carrying whatever timeout the adapter's own
        client has and none of ours. So on a partner install a server that accepts the socket
        and then never answers pins this thread rather than this screen — the screen is fine
        either way (`post_message` is dropped on a closed pump), but the process joins the
        executor at exit, which is the terminal-holding failure `_drive` measured arriving
        through the other door. The bound belongs in `agents`, next to the call it would
        bound, and not in a second timer here.

        NOTHING ON THE SCREEN IS READ HERE. The record was rendered on the event loop and handed
        in; the only state this touches is `post_message`, which is thread-safe and returns False
        on a closed pump, so a question asked and then walked away from paints nothing.

        AND NOTHING HERE MAY RAISE — a belt, and it is worn because the alternative was measured
        rather than imagined. A Textual thread worker that raises is FATAL: `WorkerFailed` comes
        out of the app, `app.is_running` goes False and the pane freezes at `thinking…`, which is
        `agents.py`'s stated contract ("the interactive shell must never fail because an optional
        model call did") broken from the one direction that file cannot defend. `agents.answer`
        swallows its own failures by design; `agents.ask_available` does NOT — it computes
        `urlsplit(...).port` outside its `except OSError` (`agents.py:293`), so
        `OLLAMA_BASE_URL=http://localhost:notaport/v1` in a shell profile raised `ValueError` and
        took the app down. Driven live, that is exactly what happened. The fix at the source was
        one line in `agents` — the parsing inside the `try`, `except (OSError, ValueError)` — and
        it landed with the controller row, which calls the same function from the EVENT LOOP
        where this catch does not reach. This catch stays anyway, and the reason is not
        belt-and-braces: a screen that survives only while every callee keeps its promises is not
        a screen that survives, and the promise this one relies on is `agents.answer`'s, not
        `ask_available`'s.

        IT REPORTS `unavailable` RATHER THAN A FOURTH SENTENCE. Every raise reachable on this
        seam today is an endpoint that cannot be spoken to, and "unavailable — start ollama" is
        the true thing to say about one; a wording whose only instance is a malformed
        `$OLLAMA_BASE_URL` would be a state on this screen that nobody ever reads. Nothing is
        emitted either way — `answer=None` is the record's "not an answer" (see `_record_ask`).
        """
        import time

        from manyruns import agents

        # Bound before the `try` so the message can still name the model if the raise came from
        # the line that would have set it.
        model = ""
        try:
            model = agents.ask_model()
            # The probe is here rather than in `_ask` so its cost is the worker's: refused on
            # loopback it is free, but `$OLLAMA_BASE_URL` pointed at a host that does not answer
            # pays `agents.ASK_PROBE_S` (100 ms), and paying that on the event loop would drop
            # frames of a run that is still going.
            if not agents.ask_available():
                self.post_message(self.Answered(question, None, dispatch_id=dispatch_id,
                                                context=context, model=model,
                                                latency=0.0, available=False))
                return
            started = time.monotonic()
            # The question goes AFTER the record and OUTSIDE the hash. `context_sha256` names the
            # record alone, and the event carries the question in its own field — so a later
            # reader rebuilds exactly what the model saw from two things it has, rather than from
            # a hash over a string that mixed them.
            text = agents.answer(system=ASK_SYSTEM, prompt=f"{context}\nquestion: {question}\n")
            self.post_message(self.Answered(question, text, dispatch_id=dispatch_id,
                                            context=context, model=model,
                                            latency=time.monotonic() - started))
        except Exception:  # noqa: BLE001 - a thread worker that raises kills the app; measured
            self.post_message(self.Answered(question, None, dispatch_id=dispatch_id,
                                            context=context, model=model,
                                            latency=0.0, available=False))

    def on_run_screen_answered(self, message: "RunScreen.Answered") -> None:
        """Write the event, then paint — but paint only the answer somebody is still waiting for.

        THE RECORD TAKES BOTH AND THE PANE TAKES ONE, and the asymmetry is the whole handler. A
        superseded answer is a real answer to a question a person really asked, so the corpus
        keeps it; the pane is somewhere a person is LOOKING, and what they are looking for is the
        answer to the question they asked last.

        `exclusive=True` does not do this for us, which is the failure the guard was written
        from. Textual cancels a superseded worker's RECORD and cannot interrupt the thread, so
        the first call finishes and its `post_message` still lands — and a handler that took the
        last message to ARRIVE rather than the last question ASKED ends on the wrong one whenever
        the older call is the slower one. Measured with two stubs, 1.2 s then 0.05 s: the pane
        settled on `slow one` while the person waited on `fast two`.

        `self.asked` is set FROM THE MESSAGE rather than left as it is, so the pane can never
        show a pair that was never asked — and that is also why `asked` cannot be the
        outstanding-question test: this line overwrites it. `_outstanding` is.

        Identical words do not identify an ask: a step can change the context between them.
        With the first call held until the second answers, matching text repainted the older
        context. The dispatch counter travels with the worker and identifies only the latest.
        """
        if message.answer:
            self._record_ask(message)
        if message.dispatch_id != self._outstanding:
            return
        self.asked, self.answer = message.question, message.answer
        if message.answer:
            self.ask_note = message.route.attribution if message.route else "client unrecorded"
        else:
            self.ask_note = self.UNAVAILABLE if not message.available else self.UNANSWERED
        # THE ROW LEARNS FROM THE ANSWER RATHER THAN FROM A SECOND PROBE. The worker already
        # called `ask_available` on its own thread and carried the result here, so re-asking on
        # the event loop would pay `ASK_PROBE_S` for a question that has just been answered —
        # and could answer it differently, which is a row disagreeing with the pane beside it.
        #
        # AFTER THE GUARD, not before: a superseded answer is a real answer to a question nobody
        # is waiting for any more, and a row that took its latency would report the wrong call's
        # duration under a question still marked `thinking…`. The record takes both (above); the
        # row, like the pane, takes the one somebody is looking at.
        self._ask_up = message.available
        self._ask_status = f"{message.latency:.1f} s" if message.available else self.UNAVAILABLE
        self._paint_ask_row()
        self.sync()

    def _record_ask(self, message: "RunScreen.Answered") -> None:
        """The first event anything in this package emits — spec §2's `ask`, through
        `trace.current()`, which is `NullTracer` unless `$MANYRUNS_TRACE` names a sink.

        UNCONDITIONALLY, with no `if tracing:` — that is what the null sink is for, and a second
        code path here would be the one nobody runs. `trace.stamp` adds `at`, `run_id` and
        `agent_id`; nothing in this file may raise into the screen and nothing in `trace.py`
        does.

        ONLY WHEN THERE IS AN ANSWER. `ask_available()` False emits nothing (§4 says so), and a
        `None` from `agents.answer` emits nothing either, for a reason worth stating: `answer`
        collapses six unrelated failures into that one value — no server, an unpulled model, a
        timeout, an empty thinking turn — so an event carrying `"answer": null` would record
        that something went wrong without being able to say what, thousands of times over. The
        corpus is of ANSWERS. If a record of failed asks is ever wanted it needs a cause, and the
        cause lives in `agents`, not here.

        **`context_tokens` IS NOT EMITTED, and §2 no longer asks for it.** The key was on the
        spec's example line and came off in this commit rather than being quietly skipped here,
        because §2 is the wire format the next emitter's author reads and a promise nothing keeps
        is worse than an absent field. The reason is written there in full and in one sentence
        here: the honest count is the endpoint's own `usage.prompt_tokens`, which the
        OpenAI-compatible response carries and `agents.answer` throws away on the way back (it
        returns attributed text, not endpoint usage), and the alternatives at this seam are a
        tokenizer this venv does not have and a `len(context) / 4` guess that no reader could tell from a measurement.
        It comes back as a real number the day `answer` returns one — a change to that function's
        signature, and therefore its own commit — and until then the hash plus
        `record_for_prompt` recovers the prompt itself, which says more than a count would.

        `run_id` IS `None` UNTIL THE RUN ENDS, and mid-run is when someone asks. `results` is
        what carries it (`_run_id`), step records do not (`runner.apply_step` writes
        `index`/`name`/`group`/`params`/`outcome`), so those events land in
        `trace.UNKNOWN_RUN` — the file that name exists for. Giving this screen a second account
        of the run's identity is not the fix; the runner reporting its id through the feed is,
        and that is the trace layer's own sub-project.
        """
        from manyruns import trace

        trace.current().emit(trace.stamp({
            "kind": "ask",
            "question": message.question,
            "answer": message.answer,
            "model": message.model,
            "backend": message.route.backend if message.route else ASK_BACKEND,
            "client": message.route.client if message.route else "unrecorded",
            "endpoint": message.route.endpoint if message.route else None,
            "fallback": message.route.fallback if message.route else None,
            "latency_s": round(message.latency, 3),
            "context_sha256": narrate.context_sha256(message.context),
        }, run_id=self._run_id()))

    # ── the row: which model, and whether it is there at all ─────────────────
    def action_ask(self) -> None:
        """Open a standalone composition, or return to the existing text."""
        field = self.query_one("#ask-input", Input)
        if field.disabled:
            self._ask_open = True
            # The prompt slot belongs to the gate and may still hold the last question it asked.
            # That question was answered — leaving it up over a field opened for a question of
            # one's own would offer to answer it twice.
            self.query_one("#ask-prompt", Static).update("")
            self.query_one("#ask").display = True
            field.disabled = False
            self._set_pending(f"{ASK_PREFIX} ")
            field.cursor_position = len(field.value)
        self._look_for_a_model()
        field.focus()

    def _close_ask(self) -> None:
        """Hide and disable the editor; if it held focus, return to the steps scroll pane.

        Steps and answer are focusable scroll panes; disabling the field alone does not
        specify which pane receives focus. A reader already in another pane stays there.
        """
        field = self.query_one("#ask-input", Input)
        was_focused = self.focused is field
        self._ask_open = False
        self._set_pending("")
        field.disabled = True
        self.query_one("#ask").display = False
        if was_focused:
            self.query_one("#steps-pane").focus()

    def _look_for_a_model(self) -> None:
        """One TCP probe, at each of the two moments the row comes up. §4's `WHETHER`.

        ON THE EVENT LOOP, and that is a cost decision with a measurement under it.
        `agents.ask_available` connects to `$OLLAMA_BASE_URL`'s host and port with a
        `ASK_PROBE_S` (100 ms) bound: measured on this laptop, 0.39–4.5 ms over five runs with
        ollama up, and a refused connect on loopback is the same order. Only a host that accepts
        nothing and refuses nothing pays the bound — 101.3 ms, measured against an unroutable
        address. Hiding that rare 100 ms in a worker would buy it with a fourth row state
        (`checking…`) in every case, a message class, and a row that flickers from a model's name
        to `unavailable` after the person has read it. The stall is worse in exactly one
        configuration; the flicker is worse in all of them.

        NOT AT MOUNT — `on_mount` argues that at length for the border title and the argument is
        the same one: this is drawn only when the row is on screen, and the row is on screen only
        when the loop is waiting or somebody asked. And it is re-asked at every opening rather
        than cached, because availability is the one thing here that CHANGES: a person reads
        `unavailable — start ollama`, starts ollama, and asks again.

        UNBELTED, deliberately. `ask_available` returns a bool for every value of
        `$OLLAMA_BASE_URL` — including the malformed one it used to raise `ValueError` on, fixed
        in that file in this commit precisely because this line calls it from the event loop,
        where `_ask_worker`'s catch does not reach. One place owns that promise, and a second
        `except` here would mean neither did.
        """
        from manyruns import agents

        self._ask_up = agents.ask_available()
        self._ask_status = self.AT_HAND if self._ask_up else self.UNAVAILABLE
        self._paint_ask_row()

    def _paint_ask_row(self) -> None:
        """The ONLY writer of `#ask-model` — `sync()` does not touch it, and that is the one
        exception to this screen's paint-from-state rule worth stating.

        Everything in `sync()` is a function of the FEED, which changes under the screen on
        every step report; the row is a function of a call nobody but this screen makes, so it
        changes only at the four transitions that write `_ask_status` — the row opening (twice),
        the question going out, and the answer coming back. Painting it from `sync()` would put
        a probe's result on a repaint path that runs several times a second and derives nothing
        new by doing so.

        Tolerates a screen that is not mounted, for `_set_pending`'s reason: the transitions are
        worth testing without a running app.
        """
        from manyruns import agents

        try:
            widget = self.query_one("#ask-model", Static)
        except NoMatches:
            return
        # NO MODEL NAMED WHEN THERE IS NONE TO ASK. `ask · unavailable — start ollama` is §4's
        # own wording and it is a complete sentence about a channel; `ask · qwen3:8b ·
        # unavailable` would name a model as if the trouble were that one's, when the trouble is
        # that nothing is listening on the port and every model is equally out of reach.
        widget.update(ask_row_view(agents.ask_model() if self._ask_up else "", self._ask_status))

    def _paint_focus(self) -> None:
        names = {"steps-pane": "Trace", "answer-pane": "Answer", "figures-pane": "Figures",
                 "tune-strip": "Parameters", "ask-input": "Text"}
        focused = names.get(getattr(self.focused, "id", None), "Navigation")
        self.query_one("#pane-focus", Static).update(
            f"Active pane: {focused} · F7/F8 figures · Tab change pane")

    def on_descendant_focus(self) -> None:
        self._paint_focus()

    async def on_key(self, event: Any) -> None:
        """Arrows belong to the focused pane; text and scroll panes keep their own keys."""
        target = getattr(self.focused, "id", None)
        if target == "figures-pane" and event.key in ("left", "right"):
            self.action_select(-1 if event.key == "left" else 1)
        elif (target == "tune-strip" and self.tunables
              and self._question_kind in ("decision", "params")
              and self.gate is not None and self.gate.is_pending(self._question_id)):
            moved = {"up": (self.move_row, -1), "down": (self.move_row, 1),
                     "left": (self.move_value, -1), "right": (self.move_value, 1)}.get(event.key)
            if moved is not None:
                fn, delta = moved
                fn(delta)
            elif event.key == "enter":
                await self.query_one("#ask-input", Input).action_submit()
            else:
                return
        else:
            return
        event.stop()
        event.prevent_default()

    def on_input_changed(self, message: Input.Changed) -> None:
        if message.input.id == "ask-input":
            self._paint_strip()

    # ── the parameter strip ────────────────────────────────────────────────
    def arm_tuning(self, step: "dict | None") -> None:
        """Load the knobs this product offers for `step`. Empty list when it offers none,
        which is how a `normalize` or a `transform` step goes by without a strip.

        AND DROP THE LAST STEP'S READOUT. A run with two tuned steps arms twice, and
        `_tuned_apart` returns None for a step accepted first try — so without this the FIRST
        step's pair stays on screen, under a caption now naming the second, while the run has
        moved past both. A step with no readout must show none.
        """
        from manyruns.tui import params as _params

        if not self._ask_open:
            self._set_pending("")
        self.tunables = _params.tunable_for(step or {})
        self._tune_params = dict(self.tunables)
        self.tune_constraints = dict((step or {}).get("constraints") or {})
        self.tune_at = 0
        self.tune_step = str((step or {}).get("name") or "")
        self.moved = None
        self._paint_strip()
        # Repainted here rather than left to the next progress event: the loop is about to BLOCK
        # on a question, so "the next repaint" can be minutes of a stale panel away.
        self.sync()

    def move_row(self, delta: int) -> None:
        """Pick a different knob. CLAMPED, not wrapped: a strip of three that jumps from the
        last row to the first on one keypress makes a fast gesture land somewhere unintended."""
        if not self.tunables:
            return
        self.tune_at = max(0, min(self.tune_at + delta, len(self.tunables) - 1))
        self._paint_strip()

    def move_value(self, delta: int) -> None:
        """Turn the selected knob and WRITE THE RESULT INTO `#ask-input`.

        This is the whole reason there is no second answer mechanism: the field ends up holding
        a string a person could have typed, `enter` submits it through `on_input_submitted`
        exactly as a typed answer, through the loop's shared validation.
        """
        from manyruns.tui import params as _params

        if not self.tunables:
            return
        name, value = self.tunables[self.tune_at]
        # Repeated arrows build a draft in the field. The strip continues to show what the
        # loop published, so abandoning the draft or having it refused cannot falsify a value.
        from manyruns.tune import parse_overrides

        draft = parse_overrides(self.pending_text(), self._tune_params)
        moved = _params.step_value(name, draft.get(name, value), delta)
        # Choosing a parameter change replaces any standalone question composition.
        self._ask_open = False
        self._set_pending(_params.compose(name, moved))
        self._paint_strip()

    # ── the two seams the tests drive, and the screen uses ───────────────────
    def _set_pending(self, text: str) -> None:
        """Put text in the ask field. Tolerates a screen that is not mounted, because the
        strip's arithmetic is worth testing without a running app.

        ONLY THE LOOKUP IS GUARDED. Wrapping the `.value` assignment too would swallow a real
        failure from the widget and report it as "not mounted" — the shape of the incident at
        `tui/app.py:322-326`, where a bare `except Exception` hid a wrong signature for a full
        test cycle.
        """
        try:
            field = self.query_one("#ask-input", Input)
        except NoMatches:
            field = None
        if field is not None:
            field.value = text
        self._pending = text

    def pending_text(self) -> str:
        """What `enter` would send. Read from the widget when there is one, so a test that
        mounts the screen and one that does not are asking the same question."""
        try:
            field = self.query_one("#ask-input", Input)
        except NoMatches:
            return getattr(self, "_pending", "")
        return field.value

    def _paint_strip(self) -> None:
        from manyruns.tui import params as _params

        try:
            widget = self.query_one("#tune-strip", Static)
        except NoMatches:
            return
        from manyruns.tune import parse_overrides

        text = self.pending_text().strip()
        draft = {} if text.startswith(ASK_PREFIX) else parse_overrides(text, self._tune_params)
        view = _params.strip_view(self.tunables, self.tune_at, draft=draft)
        widget.display = view is not None and self._question_kind in ("decision", "params")
        if view is not None:
            widget.update(view)

    #: How many of the loop's lines stay on screen. Three, because most of what `write` says is
    #: ALREADY visible elsewhere — `_fmt_step` restates the step the pane just drew, `_show_plots`
    #: prints a path the figures pane lists — and the line that is genuinely new is the one
    #: `_keep_attempt` writes ("kept attempt 1 as phate@1.png"). A full transcript would be a
    #: fourth pane competing with three that already say it better.
    SAID_LINES = 3
    FINALIZING = ("measuring geometry… final metrics are running; "
                  "duration depends on dataset and machine")

    def on_run_screen_said(self, message: "RunScreen.Said") -> None:
        """The loop talking, as opposed to asking. Kept on its own line rather than folded into
        `#caveats`, which renders from `self.results` — a run-level fact, not a running log."""
        self.said.append(message.text)
        if (message.text == self.FINALIZING and not self._finished
                and (self.gate is None or not self.gate.closed)):
            self._stop_finalizing()
            self._finalizing_started = time.monotonic()
            self._finalizing_timer = self.set_interval(0.2, self._paint_said)
        self._paint_said()

    def _paint_said(self) -> None:
        lines = list(self.said[-self.SAID_LINES:])
        if self._finalizing_started is not None:
            elapsed = time.monotonic() - self._finalizing_started
            lines = [f"{line} · {elapsed:.1f}s elapsed" if line == self.FINALIZING else line
                     for line in lines]
        self.query_one("#said", Static).update(
            "\n".join(lines))

    def _stop_finalizing(self) -> None:
        if self._finalizing_timer is not None:
            self._finalizing_timer.stop()
        self._finalizing_timer = None
        self._finalizing_started = None

    def on_run_screen_arm(self, message: "RunScreen.Arm") -> None:
        """Load the strip for the step the driver is ABOUT to ask about.

        On the UI thread, which is the point: `arm_tuning` calls `_paint_strip`, and the driver
        that knows which step it is runs on the worker. The gate's `arm` is the marshal.
        """
        if not self._finished and (self.gate is None or not self.gate.closed):
            self.arm_tuning(message.step)

    def on_run_screen_moved(self, message: "RunScreen.Moved") -> None:
        self.moved = (message.before, message.after)
        self.sync()

    def on_unmount(self) -> None:
        """Release a worker that is waiting on an answer nobody will now give.

        Without this, leaving the screen mid-question leaves the run thread blocked in
        `queue.get()` forever. It is a daemon thread so the process still exits — but the
        session's `finally` never runs, and `run_tune_loop` restores the metric suite there.
        """
        self._stop_finalizing()
        self._color_cancel.set()
        self._export.cancel()
        self.cancel_run()

    def cancel_run(self) -> None:
        """Signal immediately, without joining a thread or process on the UI loop."""
        if self.gate is not None:
            self.gate.close()
            compute = getattr(self.gate, '_compute', None)
            if compute is not None:
                compute.cancel()

    def on_run_screen_finished(self, message: "RunScreen.Finished") -> None:
        if self._finished and self._run_id() != (message.results or {}).get("run_id"):
            self._color_cancel.set()
            self._color_cancel = threading.Event()
            self._color_busy = False
            self._extra_views.clear()
            self._export.cancel()
        self._stop_finalizing()
        self.said = [line for line in self.said if line != self.FINALIZING]
        self._paint_said()
        self.results, self.error = message.results, message.error
        self._finished = True
        if self.gate is not None:
            self.gate.close()
        self._invalidate_question()
        self.sync()

    def _drive(self) -> None:
        """Run the recipe off the event loop, on a DAEMON thread.

        **Not `@work(thread=True)`, and the reason is a hang someone hit on the shipped build:**
        quit the app mid-run and the terminal does not come back. Textual runs a thread worker
        through `loop.run_in_executor(None, …)` (worker.py:326) — the default
        `ThreadPoolExecutor`, whose threads are non-daemon and are joined before the process can
        exit. Measured with a bare Textual app whose worker sleeps 15 s and which is told to quit
        at 0.5 s: **`app.run()` returned at 15.01 s.** A real fit is longer than that, so quitting
        left a terminal hostage to a run the user had just abandoned — and a second `manyruns`
        launched behind it looked like it hung too.

        The real engine and its metrics run in a spawn child; this daemon waits on its pipe.
        Leaving or quitting signals that child immediately. The daemon reaps it and records
        the retained state as incomplete, keeping the output reservation until writes end.

        `post_message` is unchanged and is what makes this safe from any thread: it branches on
        the calling thread internally, and returns False on a closed pump (read in textual
        8.2.8) — so a detached worker's `Progress` and `Finished` are dropped rather than
        delivered to a screen that is gone.

        Every exception is caught and reported as a `Finished` with an error, including the ones
        `runner.apply_step` cannot swallow (it never raises, but `start` may fail before it gets
        there — a missing file, an engine that will not build). A worker that dies silently
        leaves the screen showing a step that is still `running…`, which is a lie about a run
        that has stopped.
        """
        self.worker_thread = threading.Thread(
            target=self._run_to_completion, name="manyruns-run", daemon=True)
        if self._on_worker is not None:
            self._on_worker(self.worker_thread)
        self.worker_thread.start()

    def _run_to_completion(self) -> None:
        try:
            results = self._start(self.feed.on_step)  # type: ignore[misc]
        except Exception as e:  # noqa: BLE001 - surfaced on screen, never swallowed
            self.post_message(self.Finished(None, f"{type(e).__name__}: {e}"))
            return
        self.post_message(self.Finished(results))

    # ── the paint ────────────────────────────────────────────────────────────
    def sync(self) -> None:
        """Repaint from the feed. Idempotent and complete — the screen is rebuilt from state,
        never appended to.

        That is why there is no `FigurePane` here and no `seen` set. `figures.FigurePane` keeps
        one because the rich surface draws into scrollback: `on_step` fires twice per step
        (`running`, then settled) and an immediate-mode pane would paint the same picture twice.
        A retained-mode screen re-renders the whole list from `feed.rows()`, so drawing each
        figure exactly once is structural rather than remembered.
        """
        # `is_running`, NOT `is_mounted`, and the difference is not academic — measured on
        # textual 8.2.8, inside a pushed Screen's own `on_mount` handler the composed children
        # already exist (`len(self._nodes) == 2`) while `is_mounted` is still False. Guarding on
        # `is_mounted` made this return early at mount, so the landing state — every declared
        # step on screen as `queued` before anything runs, which is the whole point of the
        # overlay — silently never drew. `is_running` is True there and False before the pump
        # starts, which is the question actually being asked.
        if not self.is_running:
            return
        rows = self._rows_with_views()
        # The split is content-driven. `set_class` queues a LAYOUT rather than performing one, so
        # nothing here may read a height afterwards and expect the new one; the pane redraws
        # itself off its own resize instead (see `FiguresPane`).
        self.set_class(any(r.plots for r in rows), "has-figure")

        self.query_one("#heading", Static).update(self._heading_text(rows))
        self.query_one("#caveats", Static).update(caveats_view(self.results))
        self.query_one("#steps", Static).update(steps_view(rows))
        # THE ANSWER PANE IS PAINTED FROM STATE, LIKE EVERYTHING ELSE HERE. The first cut of
        # this commit exempted it — "an answer is not a function of the feed, painting it from
        # here would wipe it" — which is true of an answer and beside the point: this method's
        # contract is that the screen is rebuilt from STATE, and the fix for a value that must
        # survive a repaint is to make it state, not to give one pane its own paint path.
        #
        # The pane holds two things and they are stacked in that order. The tuned pair is the
        # tune loop's FEEDBACK — "what changed in phate (2D)" — which is not a record and is
        # drawn nowhere else (`tui/app.py` spends ~7.9 s on the demo fixture computing it); the
        # answer is what the `?` channel put there, and it is state exactly so this line can
        # rebuild it. THE ANSWER GOES UNDERNEATH, which is `run_panel`'s own argument one pane
        # in: the last thing written is the thing that gets read, and between a readout that
        # was already on screen and prose the person asked for two seconds ago, the second is
        # the one they are looking for.
        from rich.console import Group
        from rich.text import Text

        # `moved_view` RETURNS None when nothing moved between the two attempts — identical
        # g-vectors, which the mock engine produces — and `rich.Group` raises
        # `NotRenderableError` on a None child (measured, both halves). The pair went in
        # UNGUARDED before this commit — `Group(moved_view(*self.moved, step=self.tune_step),
        # rest)` at 289f214 — so the crash was live, not hypothetical; filtering is a line, and a
        # screen that dies because a tune changed nothing is a worse report of that fact than an
        # empty pane.
        blocks = [moved_view(*self.moved, step=self.tune_step) if self.moved else None,
                  answer_view(self.asked, self.answer, self.ask_note)
                  if self.asked else Text(self.AT_REST, style="dim")]
        self.query_one("#answer", Static).update(Group(*[b for b in blocks if b is not None]))
        pane = self.query_one(FiguresPane)
        if pane.selected >= 0 and pane.found:
            # Names and paths can repeat, and tuning renames a rejected attempt before
            # reusing its path. Restore the occurrence and its plot position from the
            # rows last drawn, while leaving the newest-figure sentinel free to advance.
            previous = [(r.index, r.name, i) for r in pane.rows for i in range(len(r.plots))]
            selected = previous[pane.at()]
            current = [(r.index, r.name, i) for r in rows for i in range(len(r.plots))]
            if selected in current:
                pane.selected = current.index(selected)
        pane.show(rows)

    def _rows_with_views(self) -> list[StepView]:
        return [replace(row, plots=tuple(dict.fromkeys(
            (*row.plots, *self._extra_views.get((row.index, row.name), ())))))
            for row in self.feed.rows()]

    def _keep_view(self, occurrence: tuple[int, str], path: str) -> None:
        paths = self._extra_views.setdefault(occurrence, [])
        if path not in paths:
            paths.append(path)
        self.sync()
        pane = self.query_one(FiguresPane)
        pane.selected = pane.found.index((occurrence[1], path))
        pane.show(self._rows_with_views())

    def _color_current(self, cancelled, entry, run_id) -> bool:
        return (not cancelled.is_set() and self.is_mounted and self.entry is entry
                and self._run_id() == run_id)

    def _color_work(self, job, done, cleanup=lambda result: None) -> None:
        """Run metadata/rendering work on a daemon; deliver only to the captured run."""
        from rich.markup import escape

        app, cancelled, entry, run_id = self.app, self._color_cancel, self.entry, self._run_id()

        def deliver(result, error):
            if not self._color_current(cancelled, entry, run_id):
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
            except RuntimeError:
                if error is None:
                    cleanup(result)

        threading.Thread(target=run, name="manyruns-colour", daemon=True).start()

    def action_color_by(self) -> None:
        from rich.markup import escape
        from manyruns import figspec
        from manyruns.tui import colorby

        if not self._finished:
            self.notify("finish this run before recolouring", severity="warning")
            return
        if self._color_busy:
            self.notify("colour view already in progress")
            return
        selected = self._selected_figure()
        if selected is None:
            self.notify("no figure to recolour", severity="warning")
            return
        if self.entry is None:
            self.notify("no dataset source for this figure", severity="warning")
            return
        step, path = selected
        pane = self.query_one(FiguresPane)
        occurrences = [(r.index, r.name) for r in self._rows_with_views() for _ in r.plots]
        occurrence = occurrences[pane.at()]
        entry, run_id = self.entry, self._run_id()
        self._color_busy = True
        self._color_cancel = cancelled = threading.Event()

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
                if not self._color_current(cancelled, entry, run_id):
                    return
                if key is None:
                    self._color_busy = False
                    return

                def viewed(result):
                    new_path, positional = result
                    self._color_busy = False
                    self._keep_view(occurrence, new_path)
                    note = " · legacy figure: positional metadata alignment" if positional else ""
                    self.notify(escape(f"saved colour view: {new_path}{note}"), timeout=8)

                self._color_work(lambda: colorby.recolor_figure(path, metadata, key, spec=spec),
                                 viewed, lambda result: figspec.remove_view(result[0]))

            self.app.push_screen(colorby.ColorByScreen(metadata, current=spec.get("color_by")), chosen)

        self._color_work(inspect, inspected)

    def action_select(self, delta: int) -> None:
        """Walk the figures pane's index. Quiet when there is nowhere to go.

        The pane owns the position, not this screen: it is the thing that has to redraw, and a
        second copy of "which figure" here is the two-homes drift this codebase keeps paying for.
        """
        self.query_one(FiguresPane).select(delta)

    def _selected_figure(self) -> Optional[tuple[str, str]]:
        """`(step, path)` for the figure ON SCREEN — which is the selected one, not the newest.

        THIS USED TO BE `_newest_figure`, and the rename is the behaviour change. `s` and `o` are
        bound on this screen precisely so the moment someone decides they want a figure is the
        moment they can keep it — and once the pane can show any figure in the run, "the newest"
        is the wrong one to act on. It would save a picture the reader is not looking at.
        """
        pane = self.query_one(FiguresPane)
        found = pane.found
        return found[pane.at()] if found else None

    def action_figure(self) -> None:
        """Give the newest figure the whole terminal. Does nothing when there is none — a toast
        saying "no figures yet" would repeat what the pane already says."""
        from manyruns.tui.figure import FigureScreen

        pane = self.query_one(FiguresPane)
        found = pane.found
        if found:
            entry, run_id, cancelled = self.entry, self._run_id(), self._color_cancel
            occurrences = [(r.index, r.name) for r in self._rows_with_views() for _ in r.plots]

            def on_view_at(source_index, step, path):
                if not self._color_current(cancelled, entry, viewer.run_id):
                    return
                if (step, path) in self.query_one(FiguresPane).found:
                    return
                occurrence = occurrences[source_index]
                occurrences.insert(source_index + 1, occurrence)
                self._keep_view(occurrence, path)

            def can_recolor():
                if (not self._finished or not self.is_mounted or cancelled.is_set()
                        or self.entry is not entry):
                    return False
                # A viewer opened mid-run learns the identity on Finished, before it
                # can request its first view or report that view back to this screen.
                viewer.run_id = self._run_id()
                return True

            viewer = FigureScreen(found, index=pane.at(), entry=entry, run_id=run_id,
                                  can_recolor=can_recolor, on_view_at=on_view_at)
            self._figure_exports.add(viewer._export)
            self.app.push_screen(viewer)

    def _run_id(self) -> Optional[str]:
        """The run's identity, once it has one. `results` is None until the run ends, so a save
        mid-run names the step alone rather than inventing an id for a lineage still being
        written."""
        return (self.results or {}).get("run_id")

    def action_save(self) -> None:
        """The quickdrop. See `manyruns.tui.figure.quickdrop` — SVG, PDF and PNG at 300 dpi,
        rebuilt from the spec the run wrote beside the PNG."""
        selected = self._selected_figure()
        if selected is not None:
            self._export.save(selected[1], selected[0], self._run_id())

    def action_open(self) -> None:
        """Hand the PNG to the desktop — the pointwise-exact route on any terminal."""
        from manyruns.tui.figure import open_externally

        newest = self._selected_figure()
        if newest is None:
            return
        error = open_externally(newest[1])
        if error:
            self.app.notify(f"could not open it: {error}", severity="error", timeout=8)

    def action_back(self) -> None:
        """Clear a pending answer draft before offering departure, from any pane."""
        if self.gate is not None and self.gate.is_pending(self._question_id):
            self._ask_open = False
            if self.pending_text():
                self._set_pending("")
                self.query_one("#ask-input", Input).focus()
                self.post_message(self.Said(
                    "esc again to leave this run · the question is still waiting"))
                return
        elif self.focused is self.query_one("#ask-input", Input):
            self._close_ask()
            return
        if self.gate is not None:
            self.gate._issuance.clear()
        self.app.push_screen(LeaveRunScreen(draining=not self._finished), self._leave_run)

    def _leave_run(self, confirmed: bool) -> None:
        if confirmed:
            self._stop_finalizing()
            self._color_cancel.set()
            self._export.cancel()
            self.cancel_run()
            self.dismiss(None)
        elif self.gate is not None:
            self.gate._issuance.set()

    def _heading_text(self, rows: list[StepView]) -> str:
        from rich.markup import escape

        done = sum(1 for r in rows if r.state in ("ok", "reported"))
        # `esc run again` is advertised for the same reason the roster's subtitle and the ledger's
        # hint line advertise theirs: this screen has no `Footer`, so a binding nothing states is
        # a binding nobody finds.
        # `? ask` is unconditional where the figure keys are not: a figure key with no figure is
        # a key that does nothing, and there is always a record to ask about — the declared steps
        # and the dataset are on screen before the first one runs.
        keys = ("· f figure · c colour · s save · o open · ? ask · esc run again" if any(r.plots for r in rows)
                else "· ? ask · esc run again")
        head = (f"[bold]{escape(self.heading)}[/bold]  [dim]{done} of {len(rows)} steps successful[/dim]"
                f"  [dim]{keys}[/dim]")
        if self.results is not None:
            from manyruns.outcomes import run_verdict

            _, verdict = run_verdict(self.results)
            head += f"  {escape(verdict)}"
        if self.error:
            # escaped for the same reason `shell._geometry_panel` escapes an absence's reason,
            # and here it is not hypothetical: this string is `f"{type(e).__name__}: {e}"`, and
            # `KeyError: ['x']` is markup rich would swallow whole.
            # The run's failure is the headline, in the same ink a failed step gets.
            return f"{head}  [{_OUTCOME_STYLE['error'][1]}]{escape(self.error)}[/]"
        return head


# ── the renderables, kept out of the widget so they can be rendered without one ──────────────
#: **No prose inside a table cell, on either pane, and this is measured rather than stylistic.**
#:
#: `rich.table` 15.0.0 DELETES CHARACTERS when it folds a word longer than the column it is in.
#: Measured on this exact content: a two-column grid holding
#: `not measured — GeodesicDistanceCorrelation: no get_gt_dists` rendered at a 38-cell total
#: width produced `GeodesicDistance` / `relation: no` — the `Cor` is simply gone, with no ellipsis
#: and nothing on screen to say anything was dropped. Scanned across widths, the geometry pane
#: lost characters at every width from 24 to 43 inclusive, and 36 is what a 100-column terminal
#: gives it. Cell type does not matter (`str` and `rich.text.Text` behave identically), padding
#: does not matter, and `overflow` does not matter — with the default the same range elides
#: instead, which at least says so.
#:
#: A bare `Text(…, overflow="fold")` rendered at full pane width is LOSSLESS at every width from
#: 12 to 89. So the panes below are built from `Text` lines in a `Group`, with the column
#: alignment computed here in Python rather than negotiated by a table. That is not a
#: workaround dressed as a design: this screen's entire job is to be the surface where §0's
#: "the panels wrap and break mid-value" stops being true, and a table that silently eats three
#: characters out of a metric's name is that same defect wearing a nicer border.
#:
#: A second consequence, and a welcome one: `Text.append` takes LITERAL text, so a path
#: containing `[old]` or a traceback containing `[ok]` reaches the screen as itself. The markup
#: hazard `shell._geometry_panel` has to escape around cannot arise here at all.
def steps_view(rows: list[StepView]) -> Any:
    """One line per step, with its detail and what it moved underneath it.

    `detail` is drawn IN FULL and folded by the layout engine. `narrate.run_panel` clips it at
    58 characters and `shell._steps_body` at 24, both correctly — a plain line and a fixed-width
    rich column cannot re-flow. This surface can, and clipping here would hand the pane a string
    it could not un-truncate when the terminal got wider, which is exactly the defect §0 opens
    with.
    """
    from rich.console import Group
    from rich.padding import Padding
    from rich.text import Text

    if not rows:
        # An engine that declares nothing and reports nothing must not look like a clean run.
        return Text("no steps declared, and none reported", style="dim")

    name_w = max(len(row.name) for row in rows)
    delta_w = max((len(d.label) for row in rows for d in row.deltas), default=0)
    blocks: list[Any] = []
    for row in rows:
        colour = _LIVE_STYLE.get(row.state) or _OUTCOME_STYLE.get(row.state, ("?", "white"))[1]
        # `seconds` is None until the step settles — a DURATION, not a clock. Printing 0.00s
        # while it runs would report a finished step that took no time.
        secs = "" if row.seconds is None else f"{row.seconds:.2f}s"
        line = Text(overflow="fold")
        line.append(f"{row.glyph}  ", style=colour)
        line.append(f"{row.name:<{name_w}}  ", style="bold")
        line.append(f"{row.word:<9}", style=colour)
        line.append(f"{secs:>7}", style="dim")
        blocks.append(line)
        if row.detail:
            # `Padding` rather than four spaces in the string, so the indent is HANGING: a
            # detail long enough to wrap put its continuation at column 0 otherwise, which
            # reads as a line belonging to the next step. Measured lossless at every width
            # from 12 to 119 — Padding renders its child at width-4, which is still the bare
            # `Text` path and not a table cell.
            blocks.append(Padding(Text(row.detail, style="dim", overflow="fold"), (0, 0, 0, 4)))
        for delta in row.deltas:
            moved = Text(overflow="fold")
            moved.append(f"    {delta.label:<{delta_w}}  ", style="dim")
            moved.append(delta.value, style="magenta")
            blocks.append(moved)
    return Group(*blocks)


#: Which ink each kind of caveat gets. `ⓘ` is a NOTE about the route the run took and `⚠` is a
#: warning about the run itself, and `narrate.Caveat` already separates them — colouring them the
#: same would put "this composition was unusual but valid" and "this cannot be reproduced" in one
#: voice. Yellow for the warning is `shell._steps_body`'s own choice for the same sentence
#: (`[yellow]⚠[/yellow]`), taken rather than re-picked.
_CAVEAT_STYLE = {"ⓘ": "cyan", "⚠": "yellow"}


def caveats_view(results: Optional[dict]) -> Any:
    """What the numbers do not say — `narrate.caveats`, in this surface's ink.

    Empty until the run settles, and empty is EMPTY: `Text("")` renders as nothing and the CSS
    `height: auto` collapses the widget, so a clean run has no blank band where a warning would
    be. A placeholder here ("no caveats") would be a claim, and it would be made before the run
    has produced anything to have caveats about.
    """
    from rich.console import Group
    from rich.text import Text

    rows = narrate.caveats(results or {})
    if not rows:
        return Text("")
    blocks: list[Any] = []
    for row in rows:
        # `Text.append` takes LITERAL text, so a caveat that names a bracketed step or an
        # `[ok]` reaches the screen as itself — the markup hazard cannot arise here at all.
        line = Text(overflow="fold")
        style = _CAVEAT_STYLE.get(row.mark, "yellow")
        line.append(f"{row.mark} ", style=style)
        line.append(row.text, style=style)
        blocks.append(line)
    return Group(*blocks)


def answer_view(question: str, answer: Optional[str], note: str = "") -> Any:
    """A question and what came back of it, for the pane the record used to have.

    THE QUESTION IS DRAWN TOO, and it is not decoration: the field is cleared the moment a `?`
    line is dispatched (so the next `enter` cannot re-send it), the loop's own prompt is still
    sitting in `#ask-prompt` beside it, and `#said` scrolls. Without this line the pane would
    hold three sentences of prose with nothing on screen saying what they answer — and a person
    who asked two minutes and four steps ago is exactly the reader this pane has.

    Dim for the question and for `note`, plain for the answer, and that is the only ink
    separating them. §8's risk is that generated prose sits beside measured deltas in one
    typeface; the border carries the model's name for that, and here the rule is narrower — what
    the MODEL said is the one thing in this pane not written by this app.

    `overflow="fold"` on every line, `steps_view`'s measurement: `rich.table` deletes characters
    when it folds a long word in a narrow column and a bare `Text` at pane width does not, and
    this pane is 36 columns at a 100-column terminal. An answer is the longest prose on this
    screen, so it is the thing that would have lost the characters.
    """
    from rich.console import Group
    from rich.text import Text

    blocks: list[Any] = [Text(f"{ASK_PREFIX} {question}", style="dim", overflow="fold")]
    if answer:
        blocks.append(Text(answer, overflow="fold"))
    if note:
        blocks.append(Text(note, style="dim", overflow="fold"))
    return Group(*blocks)


def ask_row_view(model: str, status: str) -> Any:
    """`ask · <model> · <status>` — the controller row's account of the channel. §4's three
    states, as one string built from two.

    THE SEGMENTS ARE JOINED, NOT FORMATTED, and empties fall out: `ask · qwen3:8b · ollama` when
    there is a model and nothing has been asked, `ask · qwen3:8b · 1.5 s` after an answer, and
    `ask · unavailable — start ollama` when the probe found nothing — one string, no branch, and
    no `· ·` where a segment used to be. A caller that has no model to name passes none, which
    is the only way to say "the trouble is not this model's" in a row whose whole subject is
    which model would answer.

    DIM, AND THE WORD `ask` LEADS. The row sits under the panes and beside a field somebody is
    typing into: it is chrome that answers a question nobody asked out loud, so it must not
    compete with the loop's own prompt at the other end of the row for the same eye. The leading
    word is what makes the rest legible at all — `qwen3:8b · 1.5 s` alone on a controller row is
    a fact with no subject.

    A string in a `Text` rather than a `Static.update(str)`, so the style travels with the
    content and this function is the one place that decides how the row reads.
    """
    from rich.text import Text

    return Text(" · ".join(part for part in ("ask", model, status) if part), style="dim")


def _entry_mapping(entry: Any) -> Optional[dict]:
    """A roster row as the mapping `narrate.record_for_prompt` reads.

    `DataEntry.to_dict()` when there is one — the class's own serialisation, so a field renamed
    there reaches the prompt renamed rather than silently stopping — and a mapping passed
    straight through, which is what a caller with no roster (a test, a driver) has. Anything
    else is `None`: a record whose dataset section is three dashes is `record_for_prompt`'s
    stated behaviour for "nobody measured this", and it is a great deal better than a prompt
    with `<DataEntry object at 0x…>` in it.
    """
    if entry is None:
        return None
    to_dict = getattr(entry, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return dict(entry) if isinstance(entry, dict) else None


def geometry_view(results: Optional[dict]) -> Any:
    """The settled g-vector, in `narrate.geometry_sections`' own split.

    Sibling of `shell._geometry_panel`, which draws the same sections in rich for the one-shot
    surface; narrate decides what belongs where and each surface picks its own ink. Two things
    the ink has to carry, both taken from that sibling:

      * an ABSENT metric is drawn differently from a measured one. Its value is a sentence, not
        a number, and in a column of `0.6481`s a lone "returned nan" reads as a value.
      * the caption on the measured section is `suite_null` verbatim, because that is the
        sentence keeping a table of embedding-derived numbers from being read as evidence about
        the data (`vocab.NULL_KIND`).
    """
    from rich.console import Group
    from rich.text import Text

    sections = narrate.geometry_sections((results or {}).get("g_vector") or {})
    if not sections:
        # Said as a state, not as a promise about what will appear.
        return Text("nothing recorded yet", style="dim")
    blocks: list[Any] = []
    for section in sections:
        if blocks:
            blocks.append("")
        blocks.append(Text(section.title, style="bold"))
        if section.note:
            blocks.append(Text(section.note, style="dim", overflow="fold"))
        # Right-aligned to the longest label in the section, capped: measured, 20 fits every
        # metric name the shipped suite produces except `geodesic_distance_correlation` (29). A
        # section containing that one would otherwise indent all twelve of its rows to 29 and
        # leave a 36-column pane 5 columns for the numbers. Over the cap the label simply
        # overflows its field — that row loses its alignment and keeps every character, which is
        # the right way round.
        width = min(max(len(row.label) for row in section.rows), _LABEL_CAP)
        for row in section.rows:
            line = Text(overflow="fold")
            line.append(f"{row.label:>{width}}  ", style="cyan")
            if row.measured:
                line.append(row.value, style="bold")
            else:
                # an absence is LABELLED as one before its reason is given: its value is a
                # sentence, not a number, and in a column of `0.6481`s a lone "returned nan"
                # reads as a value until you look twice. Same call `shell._geometry_panel` makes.
                line.append("not measured", style="yellow")
                line.append(f" — {row.value}", style="dim")
            blocks.append(line)
    return Group(*blocks)


#: How far the geometry pane's label column is allowed to grow. See `geometry_view`.
_LABEL_CAP = 20


#: How many moved metrics the pane draws. Five, because this sits above the settled g-vector in
#: a pane beside a figure at `height: 1fr` and every row it takes is a row the picture loses —
#: the same argument `figures_view` makes for bounding its own caption.
MOVED_CAP = 5


def _caption(step: str = "") -> str:
    """The moved panel's heading, and it is SHORT because a wrapped one costs a row of the
    picture.

    `RunScreen.NARROW`'s own note records this pane's reference widths: 36 columns is what a
    100-column terminal gives it at 3:2, and 28 is what an 80-column one would. The first
    spelling of the disclosure — "what the tune changed in phate (in the plotted 2D)", 50
    characters — wrapped to two lines at BOTH, which is exactly the cost `figures_view` argues
    against for its own caption and the reason `MOVED_CAP` exists: every row this takes is a row
    the figure beside it loses. No test saw it because they all render at 80, 100 or 200 columns
    and the pane has ~36.

    `(2D)` IS THE WHOLE DISCLOSURE ON SCREEN, and that is enough for what it has to do: warn a
    reader comparing this panel against the `geometry_view` rows immediately below that the two
    are not measuring the same thing. The full account — that the sidecars hold `emb[:, :2]`,
    that the direction is honest because both sides are truncated identically, and that the
    absolute values are not the run's — lives in `moved_view`'s and `app._tuned_apart`'s
    docstrings, where it costs no rows.

    27 CHARACTERS AT THE WORST CASE THIS CAN ACTUALLY BE HANDED, so it fits the 28-column case
    and therefore every wider one. The worst case is not "any step name": the only steps that
    ever arm this panel are the ones `params.TUNABLE` has a row for (`_stepped_run` asks
    `tunable_for` and nothing else) — `phate`, `umap`, `leiden`, at 26, 25 and 27
    characters. A longer
    name added to that table would push the caption over 28 and is the one thing to re-measure
    here; the test at the pane's real widths is what would say so.
    """
    return f"what changed in {step} (2D)" if step else "what changed (2D)"


def moved_view(before: dict, after: dict, cap: int = MOVED_CAP, step: str = "") -> Any:
    """What a tune changed, in `geometry_view`'s own `[from, to]` form.

    **THE TWO SIDES ARE TWO ATTEMPTS OF ONE STEP** — the last one the user REJECTED and the one
    they ACCEPTED — which is what spec §2.6 asked for. Neither is a snapshot of `session.g`, and
    it cannot be one: `run_tune_loop` sets `session.ctx["metrics"] = []` for the length of the
    loop (deliberately — the suite is pairwise/O(n²) and the loop exists to show a plot fast) and
    the suite is what writes keys into `g`. Measured twice, independently: a bare
    `session.step(phate)` adds 20 keys to `g` and the same step through `run_tune_loop` adds
    NONE. A renderer over `g` either side of a tuned step therefore draws nothing, always.

    What makes the real comparison reachable is on disk. `tune._keep_attempt` moves each rejected
    attempt's `figspec` sidecars alongside its PNG, so a tuned step leaves `plots/phate@1.spec.npz`
    (rejected), `plots/phate@2.spec.npz` (rejected) and `plots/phate.spec.npz` (accepted) — the
    PLOTTED TWO COLUMNS of every attempt, kept precisely so something could say what moved between
    two of them. Not the embedding: `io._save_scatter` builds the spec from `emb[:, :2]` and
    `figspec.save` slices again (`figspec.save`), so a step run at `n_components=3` leaves a
    2-column sidecar. Hence the caption — the direction of every row here is honest, because both
    sides are truncated identically, and the absolute values are NOT the ones `geometry_view`
    draws below in this same pane, which are measured on the full embedding.
    `app._tuned_apart` loads two of those and measures each through the seam
    `pipeline/runner.py:974` already uses (`suite.measure` → manylatents), so no geometry is
    computed in manyruns. This function only renders the pair.

    `step` NAMES THE SUBJECT. A run with two tuned steps fires this twice, and a caption that
    said only "what changed" would leave the reader to guess which one — the screen takes the
    name from `gate.arm`, which is already the channel that carries the step across the thread
    boundary. See `_caption` for why the heading is as short as it is.

    RANKED BY RELATIVE MOVEMENT, not absolute. The declared suite mixes correlations that live
    near 1.0 with counts in the thousands, so an absolute rank would show the counts every time
    and never the trustworthiness that a neighbourhood-size change actually moves.

    A metric that APPEARED or VANISHED is shown and is not divided by: it ranks first (there is
    no ratio to take), which is right — a metric the run stopped being able to measure is the
    most interesting thing on the list, not the least.

    None when nothing moved, so the caller can hide the row entirely. An empty frame captioned
    "what changed in phate (2D)" is a claim that the pane looked and found nothing, which is
    true but reads as a measurement rather than as silence. (The caption used to read "what
    the tune changed"; it was shortened for a measured reason — see `_caption`, which owns
    the wording and the column budget behind it.)
    """
    from rich.console import Group
    from rich.text import Text

    rows: list[tuple[float, str, Any, Any]] = []
    for key in sorted(set(before) | set(after)):
        was, now = before.get(key), after.get(key)
        if was == now:
            continue
        if not isinstance(was, (int, float)) or not isinstance(now, (int, float)) \
                or isinstance(was, bool) or isinstance(now, bool):
            # appeared, vanished, or not a number: infinite rank, shown first
            rows.append((float("inf"), key, was, now))
            continue
        scale = abs(was) or 1.0
        rows.append((abs(now - was) / scale, key, was, now))
    if not rows:
        return None
    rows.sort(key=lambda r: -r[0])
    kept = rows[:max(1, cap)]
    width = min(max(len(k) for _, k, _, _ in kept), _LABEL_CAP)

    def _fmt(v: Any) -> str:
        if v is None:
            return "—"
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    blocks: list[Any] = [Text(_caption(step), style="bold", overflow="fold")]
    for _, key, was, now in kept:
        line = Text(overflow="fold")
        line.append(f"{key:>{width}}  ", style="cyan")
        line.append(f"{_fmt(was)} → {_fmt(now)}", style="bold")
        blocks.append(line)
    if len(rows) > len(kept):
        blocks.append(Text(f"…and {len(rows) - len(kept)} more", style="dim"))
    return Group(*blocks)


def corpus_line(out_dir: Any, offered: int, chosen: "str | None") -> str:
    """What a finished tune loop leaves behind, said in one line.

    THE COUNT IS READ FROM THE FILE. A session counter would go up while you watch and prove
    nothing; the claim this surface is making is that a corpus exists on disk, so it is counted
    there — including the rows this session did not write.

    A count that cannot be taken is DROPPED rather than guessed: the decision still happened and
    the row was still written, and reporting `0 rows` over an unreadable corpus would be the one
    number here that is not a measurement. Zero is dropped for the same reason and not only on
    an `OSError` — `decisions.read` returns EARLY for a path that is not a file rather than
    raising, so a wrong directory arrives here as an empty iterator, which is indistinguishable
    from an empty corpus and is not what a line announcing a written row should claim.
    """
    from manyruns import decisions

    kept = f"{chosen} chosen" if chosen else "none kept"
    line = f"✓ recorded — {offered} offered, {kept}"
    try:
        total = sum(1 for _ in decisions.read(out_dir))
    except OSError:
        return line
    return f"{line} · {total} rows" if total else line


def figures_view(rows: list[StepView], selected: int = -1, width: int = 0) -> Any:
    """THREE LINES, whatever the run does: the index, the selected figure's path, and the keys.

    IT USED TO BE TWO LINES PER FIGURE and that was the pane's bug rather than its content. This
    renderable is `height: auto` under an `AutoImage` at `height: 1fr`, and `auto` wins that
    argument — so a caption growing by two lines per figure starved the picture to a single row
    (measured on 120x30: 1 row at one figure, 1 at three, 1 at six, listing 27 rows and the pane
    scrolling). Bounding it is what gives the image its height back.

    Nothing is lost by bounding it. The full listing was never readable in a 7-row pane, and the
    ONE thing it carried that mattered — a path you can hand to another program — is still here,
    for the figure actually on screen. `f` opens the viewer, which walks all of them.

    TEXT ONLY. The picture is a sibling widget, because `AutoImage` is a widget and cannot live
    inside a rich `Group`.
    """
    from rich.console import Group
    from rich.markup import escape
    from rich.text import Text

    from manyruns.tui.figure import figures_of, index_line, trajectory
    from manyruns.tui.images import HAVE_IMAGES, forced, is_pixel_perfect, tier

    found = figures_of(rows)
    if not found:
        return "[dim]no figures yet[/dim]"
    at = len(found) - 1 if selected < 0 else max(0, min(selected, len(found) - 1))

    # THE X AXIS. One mark per figure in the order the steps drew them, so walking it with the
    # arrow keys is walking the trajectory the embedding took through the recipe. Clipped from
    # the left around the cursor when the run has more figures than the pane has columns; the
    # count in `index_line` is what says how many are off the end.
    #
    # The strip is left to the pane's own width rather than measured here: `width=0` means no
    # clipping, which is what a caller with nothing to say about width gets.
    strip = trajectory(found, at, max(0, width - 2))

    def line(markup: str) -> Any:
        """One row, and exactly one. NOTHING in this caption may wrap: it sits under an image at
        `height: 1fr`, so every extra row it takes is a row the picture loses — which is the
        starvation this whole renderable was rewritten to stop. A path is the reason it matters:
        a real outputs path runs to 118 characters against a 36-column pane (measured; the pane
        is 36 at a 100-column terminal), so it alone would take four rows."""
        text = Text.from_markup(markup)
        text.no_wrap, text.overflow = True, "ellipsis"
        return text

    # THE FULL PATH WHEN IT FITS, THE FILENAME WHEN IT DOES NOT — and never a cut fragment of
    # either. Both halves of that matter and they pull in opposite directions:
    #
    #   * the path is the thing you hand to another program, and `figures_view`'s first version
    #     called it "the only part of this pane that was ever load-bearing". So it is shown
    #     whenever the pane is wide enough to show it whole.
    #   * a real outputs path is 118 characters against a 36-column pane (measured; that is what
    #     a 100-column terminal gives this pane), and wrapping it took FOUR of the pane's seven
    #     rows. That is the starvation this renderable was rewritten to stop, and it is not
    #     fixed by ellipsising the path instead — `/tmp/example/…` is not
    #     something you can hand to anything either.
    #
    # So below the fitting width it degrades to the filename, which is a COMPLETE string rather
    # than a truncation: nothing on this line is ever silently cut, which is the property
    # `test_no_pane_loses_a_character_at_any_width` exists to hold. The full path keeps its own
    # row in the viewer (`f`), which is where `figure.FigureScreen` widened a row to fit it.
    path = found[at][1]
    shown = path if (width and len(path) <= width) or not width else Path(path).name
    lines: list[Any] = [
        line(f"[bold]{strip}[/bold]  [dim]{escape(index_line(found, at))}[/dim]"),
        # escaped, because a path is user data: `outputs/run[1]/plots/phate.png` is legal,
        # `[1]` is legal markup, and rich would eat it rather than print it.
        line(f"[dim]{escape(shown)}[/dim]"),
    ]

    if not HAVE_IMAGES:
        tail = FIGURES_NO_IMAGES
    elif not is_pixel_perfect():
        # Nothing was drawn: the tail is the only thing telling you where the figure is.
        tail = FIGURES_NO_INLINE
    elif forced():
        # A forced tier can draw nothing at all, silently — see `figure._tier_note`. The pane
        # names it, because an empty rectangle with no caption is indistinguishable from a bug.
        tail = f"{tier()} (forced) · {FIGURES_KEYS}"
    else:
        tail = FIGURES_THUMBNAIL
    # THE TAIL IS THE ONE LINE ALLOWED TO WRAP, and only it. The strip and the path are pinned
    # to a row each because they degrade gracefully — the strip clips around the cursor, the path
    # falls back to a filename. The tail cannot: it is a list of KEY NAMES, and an ellipsis
    # through it deletes one. Measured at a 44-column pane, `not drawn here · ←/→ · f full ·
    # s save · o open` is 47 columns and came out as `… s save · o …` — `o open` gone, which on a
    # tier that draws nothing is the only line telling a reader the figure is reachable at all.
    # Two rows here costs the picture one; a missing key costs the figure.
    return Group(*lines, f"[dim]{tail}[/dim]")
