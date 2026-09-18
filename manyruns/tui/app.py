"""The Textual app — surface three of three, and the launch path for `manyruns` on a TTY.

    Textual app     this module + `roster` / `ledger` / `run` / `find`   `manyruns` on a TTY
    rich panels     `shell.render_result`                         `manyruns run`, UNCHANGED
    plain text      `narrate.run_panel`                           piped, CI, UNCHANGED

Importing `manyruns.tui` still pulls no Textual (the package exports only `state`); Textual
arrives only when this module is imported, and `manyruns.app.cmd_app` imports it inside the
function body so that the base chain — and every other subcommand — stays as it was.

WIRED TO THE FRONT DOOR, as of this integration. `manyruns.app.main` routes a bare `manyruns`
at a real terminal here (`app._wants_the_app`). Everything else is byte-for-byte the path it was
on: a pipe, a script, `manyruns run`, and the explicit `manyruns shell` all still reach
`shell.run_shell`, and `manyruns run <data>` still reaches the one-shot. Three surfaces, one
core, and this module is the only new reader of it.

THE FOUR SCREENS AND THE THREE HAND-OFFS. Each screen is `Screen[T]` and hands back a value; the
app is the only thing that knows what comes next, which is why no screen imports another:

    roster  --DataChosen(entry)---->  open_ledger(entry)
    roster  --FindDataRequested--->   open_finder()
    ledger  --dismiss(recipe|None)->  open_run(entry, recipe)
    finder  --dismiss(entry|None)-->  found(entry)  -> open_ledger(entry)
    run     --run again------------>  ledger, previous recipe selected

A `push_screen` callback is safe to push from, and that is measured rather than assumed:
`Screen.dismiss` invokes the callback through `ResultCallback.__call__`, which does
`requester.call_next(self.callback, result)` — a message queued on the APP's pump — and only then
calls `pop_screen()`. So the pop of the dismissed screen always happens before the callback runs,
and the screen the callback pushes is never the one popped. Calling the callback inline would
have popped the newly-pushed screen instead, which is exactly the bug this note exists to stop
someone re-introducing by "simplifying" it.

ONE FACT IS COMPUTED HERE, AND IT IS NAMED. This module chooses screens and binds one run;
every string on screen comes from the screens, which get theirs from `narrate` (see
`tui/state.py`'s header) — with a single exception, `_tuned_apart`, which loads two of a tuned
step's `figspec` sidecars and measures each through `suite.measure`. The numbers are still not
computed *in manyruns*: that is the same seam `pipeline/runner.py` uses to reach manylatents'
metrics, which is what CLAUDE.md's layering requires ("manyruns owns the workflow and the
recipe, not the math"). What this module decides is WHICH TWO ATTEMPTS to compare and which
measured keys are about the embedding rather than about the suite (`_is_geometry`);
`run.moved_view` renders the pair.

Why the exception exists rather than the rule being kept: `narrate`'s whole vocabulary is over
ONE FINISHED RUN — a record, its g-vector, its steps — and this compares two ATTEMPTS of one
step, neither of which is a run and only one of which the run contains. There is no record to
narrate. And the numbers cannot be snapshotted off `session.g` either side of the loop, because
`run_tune_loop` switches the metric suite off for its own length, so nothing writes them (see
`_tuned_apart`, which carries the measurement).

Why it was not moved beside `moved_view` in `tui/run.py`, which renders it: that file's own
header states this same rule in stronger terms ("If something here starts computing a fact, it
belongs in `narrate`"), so the move would relocate the exception rather than remove it — and it
would put a `Session` into a screen module that deliberately holds none (`tui/gate.py`'s header:
the screen cannot reach the session, which is why the gate is the caller's). One named exception
in the module that already binds the run beats two half-true headers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Thread
from pathlib import Path
from typing import Any, Callable, Optional

from textual.app import App
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static

from manyruns import narrate
from manyruns.tui.find import FindScreen
from manyruns.tui.gate import Gate
from manyruns.tui.ledger import LedgerScreen
from manyruns.tui.roster import RosterScreen
from manyruns.tui.theme import MANYRUNS
from manyruns.tui.run import RunScreen, corpus_line
from manyruns.tui import state
from manyruns.tui.state import DataEntry


@dataclass
class _RunReservation:
    """UI-owned lease, retained through the daemon's final store writes."""
    token: object = field(default_factory=object)
    thread: Optional[Thread] = None
    screen: Optional[RunScreen] = None


def stylesheet() -> str:
    """The CSS, read from `manyruns.tcss` through `importlib.resources`.

    Textual's own `CSS_PATH` resolves relative to the file the App class is defined in, which
    CLAUDE.md rules out for bundled data ("never `__file__`-relative — see train.py"): it is the
    thing that makes distribution path "B" expensive later, and the config layer is already
    located this way for exactly that reason. The stylesheet stays a real `.tcss` file — the
    point of §2 is that layout is declared, not hardcoded — it is just located the way the rest
    of this package's bundled data is.
    """
    from importlib import resources

    return resources.files("manyruns.tui").joinpath("manyruns.tcss").read_text(encoding="utf-8")


class ManyrunsApp(App):
    """`manyruns`, as an application. Lands on the roster; there is no home menu to land on."""

    CSS = stylesheet() + "\n#pending-run { dock: bottom; height: auto; padding: 1 2; }"
    TITLE = "manyruns"
    WAITING = "waiting for the previous run to finish · esc cancels this start"

    # Textual's DEFAULT quit binding is `ctrl+q`, and the VS Code integrated terminal — where
    # this was first launched — swallows it as "Quit Window" before the terminal ever sees the
    # key. The app looked unquittable. `ctrl+c` is the one chord every terminal forwards, so it
    # is bound explicitly rather than left to a default; `ctrl+q` stays for terminals that do
    # deliver it.
    #
    # WHAT THE DEFAULT DOES INSTEAD, measured in a real pty rather than inferred: `App.BINDINGS`
    # carries `Binding("ctrl+c", "help_quit", system=True)`, and with the roster's filter box
    # focused `Input`'s own `ctrl+c -> copy` runs first and raises `SkipAction` (nothing is
    # selected), so the key falls through to `action_help_quit`. That posts a NOTIFICATION
    # titled "Do you want to quit?" reading "Press ctrl+q to quit the app". It is phrased as a
    # question and is not one — a toast takes no answer, so `enter` on it does nothing, and the
    # app reads as hung by the person who just tried to answer it. Declaring the binding is what
    # stops the unanswerable question from being asked.
    #
    # `show=True` renders nowhere today: this app has no `Footer`. Each screen advertises its own
    # keys on its bottom border (`roster.subtitle`, `ledger`'s hint, `find.HINT`, `run`'s line),
    # which is where the wrong key was found. Kept true so the binding appears if a Footer lands.
    #
    # Plain `q` is deliberately NOT bound: the roster keeps focus in its filter box so typing
    # narrows the list without a "press / to search" ceremony, and a bare letter would be typed
    # rather than obeyed.
    BINDINGS = [
        Binding("ctrl+c", "quit", "quit", show=True, priority=True),
        Binding("ctrl+q", "quit", "quit", show=False, priority=True),
        Binding("escape", "cancel_start", "cancel waiting start", show=False, priority=True),
    ]

    def __init__(self, args: Any = None, **kwargs: Any) -> None:
        """`args` is the argparse namespace `manyruns.app.main` already parsed — the same object
        `shell.run_shell` would have been handed for the same invocation.

        WHAT IS ACTUALLY IN IT TODAY, stated because the plumbing suggests more: this door is
        reachable only from a BARE `manyruns` (`app._wants_the_app` requires an empty argv), and
        a bare invocation carries no flags, so the namespace is `_normalize_argv([]) == ["shell"]`
        parsed — the parser's defaults and nothing else. `--engine mock` still reaches the rich
        console as `manyruns shell --engine mock`; it cannot reach this one, and per §6 of the
        spec that is the intended shape rather than a gap: the engine stops being a question here.

        It is threaded rather than hardcoded anyway, for two reasons that are not aspiration.
        Every read is the same `getattr(args, …, default)` call `shell._run` makes, so the two
        surfaces share ONE set of defaults instead of drifting into two; and the tests below pass
        a real namespace (`--engine mock`), which is what keeps those reads exercised.

        `None` is legal and is what `ManyrunsApp()` gets: `getattr(None, "engine", None)` is
        `None`, so an app with no namespace behaves as one whose flags were all left alone.
        """
        super().__init__(**kwargs)
        self.args = args
        self._preparing_run = None
        self._run_reservations: dict[Path, _RunReservation] = {}
        self._waiting_run = None
        self._waiting_widget = None
        self._run_poll = None

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action == "cancel_start":
            return self._waiting_run is not None
        return super().check_action(action, parameters)

    def action_cancel_start(self) -> None:
        self._cancel_pending_start()

    def _cancel_pending_start(self) -> None:
        self._preparing_run = None
        self._waiting_run = None
        if self._waiting_widget is not None:
            # Keep one reusable status node on its screen. Textual removes nodes
            # asynchronously, so a replacement start could mount a duplicate ID.
            self._waiting_widget.display = False
            self._waiting_widget = None

    async def action_quit(self) -> None:
        self._cancel_pending_start()
        self._cancel_runs()
        await super().action_quit()

    def _cancel_runs(self) -> None:
        for reservation in self._run_reservations.values():
            if reservation.screen is not None:
                reservation.screen.cancel_run()

    def on_unmount(self) -> None:
        self._cancel_pending_start()
        self._cancel_runs()
        if self._run_poll is not None:
            self._run_poll.stop()
            self._run_poll = None

    def _poll_runs(self) -> None:
        """Reap only stopped handles. Never join on the UI or an executor thread."""
        for path, reservation in list(self._run_reservations.items()):
            if reservation.thread is not None and not reservation.thread.is_alive():
                if self._run_reservations.get(path) is reservation:
                    del self._run_reservations[path]
        if self._waiting_run is not None:
            path, origin, entry, recipe, decision = self._waiting_run
            if not self.is_running or self.screen is not origin:
                self._cancel_pending_start()
            elif path not in self._run_reservations:
                self._cancel_pending_start()
                # Waiting may outlast the source. Prepare again before launching.
                self.open_run(entry, recipe, decision)
        if not self._run_reservations and self._waiting_run is None and self._run_poll is not None:
            self._run_poll.stop()
            self._run_poll = None

    def _worker_started(self, path: Path, token: object, thread: Thread) -> None:
        reservation = self._run_reservations.get(path)
        if reservation is not None and reservation.token is token:
            reservation.thread = thread

    def on_mount(self) -> None:
        # REGISTERED THEN SELECTED, in that order, because `App.theme` is a reactive whose setter
        # looks the name up in the registry and raises on a miss. Doing it here rather than as a
        # class attribute is what Textual supports: the registry does not exist until the app is
        # running, so `theme = "manyruns"` on the class would resolve against the stock set.
        #
        # This is the whole binding. `manyruns.tcss` and every screen's `DEFAULT_CSS` are already
        # written against tokens, so re-pointing the tokens re-skins all four screens at once; no
        # widget names a colour, and none had to be edited to make this work.
        self.register_theme(MANYRUNS)
        self.theme = MANYRUNS.name

    def get_default_screen(self) -> Screen:
        # The landing screen IS the first screen, not something pushed onto a menu underneath
        # it — §3.1: "launch goes straight here". A pushed screen would leave a back-stack
        # position for a question the rewrite deleted.
        return RosterScreen()

    # ── the seams ────────────────────────────────────────────────────────────
    def on_roster_screen_data_chosen(self, message: RosterScreen.DataChosen) -> None:
        self.open_ledger(message.entry)

    def on_roster_screen_find_data_requested(
        self, message: RosterScreen.FindDataRequested
    ) -> None:
        self.open_finder()

    def open_ledger(self, entry: DataEntry, selected_recipe: Optional[str] = None) -> None:
        """What `enter` on a dataset opens — §3.2, the analysis picker.

        `entry` goes across whole rather than unpacked, and is captured by the callback rather
        than looked up again: `LedgerScreen` returns only the recipe NAME, and pairing it with
        the row that was already in hand is what lets `open_run` call `entry.as_source()` — the
        `(data_folder, dataset, modality, obs)` tuple `shell._run` takes — instead of resolving
        the dataset a second time and risking a second answer about what it is.
        """
        # The SCREEN is captured, not just the entry, because its `rows` are the offer the
        # decision record has to carry — the rows it actually drew, not a recomputed ledger. It
        # outlives the pop (the callback runs after `pop_screen`, see this module's header), so
        # reading `.rows` there is reading what was on screen a moment earlier.
        self._cancel_pending_start()
        screen = LedgerScreen(entry, selected_recipe=selected_recipe)
        self.push_screen(screen, callback=lambda recipe: self.chose(entry, screen, recipe))

    def chose(self, entry: DataEntry, screen: LedgerScreen,
              recipe_name: Optional[str]) -> None:
        """Record the choice, then act on it.

        BEFORE `open_run`, not after, and that ordering is the point. `open_run` can decline (no
        compute backend), and a declined choice is still a decision: the person was offered eight
        analyses and picked one. Writing the record afterwards would keep exactly the decisions
        that happened to succeed, which is the selection bias a corpus can least afford.

        Backing out returns without a row: one line per COMMITTED choice — an escape is
        ambiguous (interrupted, changed their mind, mis-keyed) and recording it is the first step
        onto interaction telemetry this product has no reason to collect.
        """
        if not recipe_name:
            return
        from manyruns import decisions

        decision = decisions.append(
            offered=[row.as_offer() for row in screen.rows], chosen=recipe_name,
            dataset=entry.dataset or entry.name, shape=entry.shape,
            topology=entry.topology, surface="ledger",
        )
        self.open_run(entry, recipe_name, decision=decision)

    def open_finder(self) -> None:
        """What `enter` on `find data…` opens — the manifest search, size confirm and fetch."""
        self._cancel_pending_start()
        self.push_screen(FindScreen(), callback=self.found)

    def found(self, entry: Optional[DataEntry]) -> None:
        """The finder's answer: a `DataEntry` for something fetched or generated, or `None`.

        One type across both doors — the roster hands `open_ledger` a `DataEntry` and so does
        this — so the ledger cannot tell which door its data came in through.

        The roster is RE-READ first, and that is an interaction defect rather than a nicety:
        component 5 writes the fetched `.h5ad` into `shell.data_dir()`, which is the folder
        component 2's `state.roster()` reads, and the roster behind this screen was built before
        the download existed. Without the re-read, escaping back from the ledger lands on a
        roster that does not list the file the user just downloaded.
        """
        if entry is None:
            return
        self.refresh_roster()
        self.open_ledger(entry)

    def refresh_roster(self) -> None:
        """Re-read the landing screen's rows in place. Costs one `state.roster()` — measured by
        component 2 at 361 ms cold, 18 ms warm — which is why it is called after a fetch and not
        on every screen change. `RosterScreen.load` catches its own failures and reports them in
        the footnote, so this cannot be why a screen change does not happen."""
        base = self.screen_stack[0] if self.screen_stack else None
        if isinstance(base, RosterScreen):
            base.load()

    def open_run(self, entry: DataEntry, recipe_name: Optional[str],
                 decision: Optional[str] = None) -> None:
        """Start the chosen analysis — §3.3.

        `recipe_name` is `None` when the ledger was escaped out of, which means "no recipe
        chosen" and not "an error": the callback fires either way (`ResultCallback.__call__`
        passes the result through unconditionally), so the check is here rather than at the
        ledger, which is right — the caller decides what back means, and back from the ledger is
        the roster it was pushed over.

        The recipe dict is `load_recipe(recipe_name)`, which is exactly what
        `app.select_analysis(..., override=recipe_name)` returns for the same name, so the steps
        drawn as `queued` before the run starts are the steps the run will actually take.

        IT REFUSES WHEN THERE IS NO COMPUTE BACKEND, and that check is not defensive tidiness —
        it is measured. `app._default_engine()` returns `None` on a base install (its own comment:
        "Returning `mock` here is what 'silently invents numbers' looks like in one line"), and
        `explore_once(engine=None)` does NOT refuse: driven with `engine_available` forced False,
        it returned `{"engine": "mock", "g_vector": {"final_dim": 3}, "caveats": ["engine=mock —
        every number here is invented; no data was read"]}`. So without this the shipped, no-extras
        install would answer a scientist's first question with fabricated geometry. The rich shell
        refuses the same case in `_pick_engine` (returns `None`, and `_explore` returns without
        running); this is that refusal, on this surface.

        The toast does NOT repeat the install command. The header footnote already carries it
        (`shell.state_rows`), and one spelling in one place is the point: a second copy here
        would drift from it. That row once lost a bracketed `[real]` to markup on every
        surface; it was fixed in the string rather than per-renderer, so there is now exactly
        one wording and nothing for a second copy to disagree with.
        """
        if not recipe_name:
            return
        from manyruns.tui import samplefetch

        if self._waiting_run is not None:
            _, _, waiting_entry, waiting_recipe, _ = self._waiting_run
            if (waiting_entry, waiting_recipe) == (entry, recipe_name):
                return
            self._cancel_pending_start()
        if self._preparing_run is not None:
            return
        token = self._preparing_run = object()
        origin = self.screen

        def done():
            # prepare_entry calls on_done before on_ready. Clear a declined selection
            # on the next UI turn, after a successful continuation has consumed it.
            def clear():
                if self._preparing_run is token:
                    self._preparing_run = None
            self.call_next(clear)

        def ready(current):
            if (not self.is_running or self._preparing_run is not token
                    or self.screen is not origin):
                return
            self._preparing_run = None
            self._open_prepared_run(current, recipe_name, decision)

        samplefetch.prepare_entry(self, entry, ready, on_done=done)

    def _open_prepared_run(self, entry: DataEntry, recipe_name: str,
                           decision: Optional[str] = None) -> None:
        from manyruns import app as _app
        from manyruns.catalog import load_recipe

        if not (getattr(self.args, "engine", None) or _app._default_engine()):
            self.notify("no compute backend installed — nothing can run yet. "
                        "The engine line at the top of the roster says what to install.",
                        severity="error", timeout=10)
            return
        from manyruns import shell

        recipe = load_recipe(recipe_name)
        slug = _app._slug(shell._slugish(entry.as_source()[3].source))
        path = (Path("outputs") / slug).resolve()
        previous = self._run_reservations.get(path)
        if previous is not None:
            if previous.screen is self.screen:
                return
            self._waiting_run = (path, self.screen, entry, recipe_name, decision)
            self._waiting_widget = next(iter(self.screen.query("#pending-run")), None)
            if self._waiting_widget is None:
                self._waiting_widget = Static(self.WAITING, id="pending-run", markup=False)
                self.screen.mount(self._waiting_widget)
            self._waiting_widget.display = True
            self.refresh_bindings()
            return
        reservation = _RunReservation()
        self._run_reservations[path] = reservation
        if self._run_poll is None:
            self._run_poll = self.set_interval(0.05, self._poll_runs)
        # THE GATE IS THE CALLER'S, and that is not a preference. `start` closes over it and
        # `start` is an argument to the screen's constructor, so a gate the screen made could
        # not be reached from inside the run (`tui/gate.py`'s header states the constraint).
        # One object, two holders.
        #
        # MEASURED BEFORE THIS LINE: `Gate(` and `gate=` occurred in TESTS ONLY. The loop, the
        # queue, the `Ask`/`Said` messages and their end-to-end test all existed, and the screen
        # a bare `manyruns` lands on ran recipes straight through and asked nothing. This
        # argument is the whole wire.
        gate = Gate()
        # THE TWO FACTS THE MODEL CANNOT RE-DERIVE, and the only place that holds both.
        #
        # `record_for_prompt` renders the dataset block from `entry` and the cautions from
        # `concerns`; with neither, the prompt a `?` sends carries six dashed values and no
        # caution — measured — which makes "should I trust leiden on this before filtering?"
        # unanswerable from the app, since its whole answer IS the QC numbers. The screen
        # cannot fetch them: re-measuring the file would be a second account of a dataset that
        # may have changed since the roster read it, and re-deriving the caution would be a
        # second legality judgement. So they are passed, from the one caller that already holds
        # the `DataEntry` (it puts `entry.name` in the heading on the next line).
        #
        # Both calls exist already and neither is new work here: `state.measurement` is the
        # accessor the ledger reads, and `narrate.concerns` is the function that drew the ⚠ the
        # person saw. The model is handed what the person was shown, not a third opinion.
        try:
            screen = RunScreen(
                recipe, gate=gate,
                start=self.start_for(entry, recipe_name, decision, gate=gate),
                on_worker=lambda thread: self._worker_started(path, reservation.token, thread),
                heading=f"{recipe_name} · {entry.name}", entry=entry,
                concerns=narrate.concerns(recipe, state.measurement(entry)))
            reservation.screen = screen
            self.push_screen(screen, callback=lambda _: self.open_ledger(
                entry, selected_recipe=recipe_name))
        except Exception:
            if self._run_reservations.get(path) is reservation and reservation.thread is None:
                del self._run_reservations[path]
            raise

    def start_for(self, entry: DataEntry, recipe_name: str,
                  decision: Optional[str] = None,
                  gate: Optional[Any] = None) -> Callable[[Any], dict]:
        """The `start` callable `RunScreen` takes: `fn(on_step) -> results`, run on a worker
        thread. Everything but the observer is bound here, which is the shape every existing run
        path already offers (`Session(on_step=…)`, `runner.run_inproc(on_step=…)`,
        `app.run_explorations(on_step=…)`).

        THIS IS THE APP'S `shell._explore`, and it does what that function does AFTER the run as
        well as during it. `shell._explore` calls `_record` → `store.append`, and its docstring
        is explicit about what dropping that line costs: the run still completes and returns
        results, and `store.read` finds zero rows for it, so the artifact is reachable only by
        knowing a run id you never saw. A front door that recorded nothing would have made that
        true again for every run started from this surface.

        Every read of `self.args` is `getattr(..., default)` with the same default `shell._run`
        uses, so this surface's defaults and the rich shell's are one set of defaults rather than
        two. The engine is `--engine` if the namespace carries one, else `app._default_engine()` —
        the same two-step `shell._run` takes. On the shipped front door it is always the second
        branch, because a bare `manyruns` has no flags (see `__init__`), and §6 says that is the
        point: the engine stops being prompt 5 and becomes the roster's footnote, because
        `_default_engine()` had already chosen and the user was being asked to ratify a decision
        they had no new information about.

        `gate` IS THE ONLY BRANCH IN HERE, and it selects the DRIVER, not the bookkeeping.
        Everything either side of the run — `write_project` above it, `store.append` with the
        `decision_id` below it — is the same code on both paths, because those are this
        surface's two obligations that `app.interactive_session` does not have and a second run
        path that quietly dropped them would be two front doors producing different things
        again. `gate=None` (every caller before this, and every test that overrides this method)
        is byte-for-byte the `explore_once` call it always was.
        """
        args = self.args
        folder, dataset, modality, obs = entry.as_source()
        source_options = self._source_options(entry)

        def start(on_step: Any) -> dict:
            from manyruns import app as _app
            from manyruns import shell as _shell
            from manyruns import store as _store

            # THE PROJECT, which the app did not write and the CLI always has. Without it a run
            # started here left `plots/` and `state/` and nothing else: `manyruns projects`
            # (which globs `outputs/*/project.yaml`) did not list it, and `manyruns open` — the
            # resume path — had nothing to replay. Two front doors producing different things,
            # which is the seam this package exists to keep closed.
            #
            # Normalize the source slug exactly as write_project and _build_session do.
            # _slugish retains Unicode letters that _slug removes, so reservations, project
            # writes and both execution routes must use the same final name.
            #
            # No name PROMPT either: the directory is already chosen by the line below, and asking
            # for a name the app then ignores would be that same split with a dialog on top. The
            # CLI prompts because it has no screen; here the roster row is the name.
            slug = _app._slug(_shell._slugish(obs.source))
            out_dir = Path("outputs") / slug
            try:
                _app.write_project(slug, folder, modality, _app.load_recipe(recipe_name),
                                   engine=getattr(args, "engine", None) or _app._default_engine(),
                                   dataset=dataset, **source_options)
            except OSError:
                # OSError ONLY — an unwritable `outputs/`, a full disk, a permission problem.
                # Those must not stop a run that is otherwise fine; the project is a convenience
                # for reopening, not a prerequisite for computing.
                #
                # A bare `except Exception` here was the first draft and it hid a real bug for a
                # test cycle: the call never ran at all under the test harness and the swallow
                # made that look like a missing feature rather than a wiring mistake. A TypeError
                # from a wrong signature is a defect and must reach the screen, where
                # `_run_to_completion` turns it into a visible `Finished(error)`.
                pass

            if gate is None:
                results = _app.explore_once(
                    data_folder=folder, dataset=dataset, modality=modality,
                    recipe_name=recipe_name,
                    engine=getattr(args, "engine", None) or _app._default_engine(),
                    seed=getattr(args, "seed", 42), fast_dev_run=_app._smoke(args),
                    time_key=getattr(args, "time_key", None),
                    color_by=getattr(args, "color_by", None),
                    **source_options,
                    # `None`, not `"log1p"`: the shim OVERRIDES the recipe's declared method, so
                    # a default here would rewrite every recipe's `transform` step to log1p on
                    # every run from the app — including one that declared `sqrt` — with nobody
                    # having typed the flag.
                    transform=getattr(args, "transform", None),
                    device=getattr(args, "device", None),
                    out_dir=out_dir,
                    on_step=on_step,
                )
            else:
                results = self._stepped_run(entry, recipe_name, slug, gate, on_step)
            try:
                # `decision_id` rides the EXISTING `extra=` channel — no schema change in
                # `store`, and it is the back-reference that joins this run to the choice that
                # started it. Absent (None) for every row written before decisions existed and
                # for every run that did not come through the ledger, which is the honest
                # answer: those runs were not chosen from anything.
                _store.append(results, extra={"source": str(obs.source), "modality": modality,
                                              "decision_id": decision})
            except OSError as e:  # noqa: BLE001 - a read-only cwd must not lose a completed run
                # Said out loud rather than swallowed, and said as a TOAST rather than into the
                # run's own panes: the run succeeded and its geometry is correct, so writing the
                # failure where the results go would misattribute it. `App.notify` is
                # thread-safe — it is one `post_message`, which branches on the calling thread
                # itself — and this is on the worker.
                self.notify(f"run not recorded: {e}", severity="warning")
            return results

        return start

    def _source_options(self, entry: DataEntry) -> dict:
        """Use the execution layer's catalog contract for both project writes and runs."""
        from manyruns import app as _app

        name = entry.name if entry.kind == "bundled" else None
        params, _ = _app._dataset_contract(
            name, entry.dataset, _app._parse_kwargs(getattr(self.args, "data_kwargs", None)))
        return {"dataset_name": name, "data_kwargs": params}

    def _stepped_run(self, entry: DataEntry, recipe_name: str, slug: str,
                     gate: Any, on_step: Any) -> dict:
        """The recipe, one step at a time, with `run_tune_loop` around the steps that offer a
        knob — and around no others.

        WHY NOT `explore_once`. That function runs the whole recipe in one call and the loop is
        per STEP, so there is no seam inside it to stop at. `app.interactive_session` already
        drives a `Session` this way and this is that loop with the terminal's `input`/`print`
        replaced by the gate's `read`/`write` and loop drawing suppressed in favour of the
        record-fed pane. `run_tune_loop` owns the same decisions on both surfaces.

        THE GATE IS ARMED PER STEP, not per run. A recipe of five steps with a question on each
        is four interruptions nobody asked for, and `normalize`/`transform`/`pca` are not what
        anybody came to tune. `params.tunable_for` returning empty is the signal to pass
        straight through, which keeps the decision in one table rather than in a second list of
        "steppable" names.

        `slug`, NOT an `out_dir`, and that is the one thing about `_build_session` a caller has
        to know: it IGNORES an `out_dir` key and computes `Path("outputs") / _slug(project)`
        itself (`app.py:434`), so the project NAME is what chooses the directory. Passing
        `project=str(out_dir)` — the obvious spelling — measures out as
        `_slug("outputs/data-pbmc3k-raw-h5ad")` = `outputs-data-pbmc3k-raw-h5ad`, i.e. a run
        filed under `outputs/outputs-data-pbmc3k-raw-h5ad`, which is neither where `start_for`
        wrote `project.yaml` nor where the corpus counter reads `decisions.jsonl`. It would have
        completed and looked fine, which is the whole danger. The caller applies both
        `_slugish` and `_slug` before reserving the directory or executing, so project.yaml
        and the run's artifacts share one `outputs/<slug>/`, including Unicode sources.

        Abandonment terminates offloaded compute and stops issuing steps. A cancelled
        tune attempt keeps its rollback and discarded record; unissued occurrences are named
        separately in not_run. Session.close skips final geometry and records incomplete while
        preserving the artifact completion-marker contract.

        ONE MEASURED BEHAVIOUR DIFFERENCE FROM `explore_once`, named here rather than left to
        be discovered: **`--transform` does not reach this path.** `explore_once` applies
        `apply_transform_shim` to the resolved recipe; `_build_session` never does, and the
        recipe goes to `Session` as loaded. It is inert on this surface today — `start_for`
        passes `transform=None` deliberately (see its own comment on why a default there would
        rewrite every recipe's declared `transform` step) and a bare `manyruns` carries no
        flags — but it is a real difference between the two run paths, and the day this surface
        grows a transform control it is the line to change.

        AND ITS TWIN, with the difference that this one is UNREACHABLE rather than merely
        inert. `--device` is taken literally here: `run_explorations` resolves it through
        `devices.resolve_device(device, requires_fp64=recipe_requires_fp64(recipe))`
        (`app.py:1846`), which downgrades a device the recipe's float64 needs rule out and
        returns a note saying so; `_build_session` reads `getattr(args, "device", None) or
        "cpu"` (`app.py:500`) and hands that to `Session` as given.

        NO INVOCATION CAN CURRENTLY PUT A DEVICE ON THIS PATH. `_wants_the_app` requires a BARE
        argv — its own docstring: *"every flag keeps the exact path they were on"* — so
        `manyruns --device mps` and `manyruns shell --device mps` both route to the rich
        console, which resolves correctly. The app door parses `["shell"]` and nothing else, so
        `args.device` is `None` here and the value is always `"cpu"`. This is a latent
        divergence, not a live one, and it is recorded so it does not become live silently.

        WHAT IT WOULD COST IF THIS DOOR EVER TAKES FLAGS: the case `resolve_device` exists for
        is `mps` against a `group: lightning` recipe — `cflows`, `traced`, `sandbox` — where
        MIOFlow needs float64 and MPS has none (`devices.py:52-53`, `recipe_requires_fp64` at
        `devices.py:32`). Passed through unresolved, that reaches MIOFlow's trainer as `"mps"`
        (`runner.py:1237`). The fix is the one call, in `_build_session`, which is the CLI's
        constructor and therefore where both doors would get it from.

        An earlier revision of this paragraph claimed `manyruns --device auto` reaches the app
        and that `manyruns run --device auto` "resolves to MPS or CUDA". Both were wrong and
        neither was checked: `--device` is declared `choices=("cpu", "cuda", "mps")`
        (`app.py:279`), so `auto` is an argparse error on every subcommand, and
        `resolve_device("auto")` takes the `if requested:` branch and falls out at
        `return "cpu", f"unknown device {requested!r} -> cpu"` (`devices.py:60`) — auto-selection
        happens only when nothing was requested. Struck rather than edited away, because a
        docstring that invents a flag is the failure this file's own conventions exist to catch.
        """
        import argparse

        from manyruns import app as _app
        from manyruns import tune
        from manyruns.pipeline.runner import ComputeCancelled
        from manyruns.tui import params as _params

        folder, dataset, modality, _obs = entry.as_source()
        engine = getattr(self.args, "engine", None) or _app._default_engine()
        # EXACTLY the attributes `_build_session` reads, and three of them BARE rather than
        # through `getattr`: `args.time_key` (:441), `args.seed` (:471) and `args.data_kwargs`
        # (:479, via `_parse_kwargs`). A namespace missing `data_kwargs` raises AttributeError
        # on the first call. Preserve explicit values; `_parse_kwargs(None)` returns `{}`.
        # `smoke`/`full` are the pair `_smoke` reads: resolved here against this app's own
        # namespace and restated so `_build_session`'s second `_smoke(args)` gets the same
        # answer rather than re-deciding it from flags this namespace does not carry.
        args = argparse.Namespace(
            seed=getattr(self.args, "seed", 42),
            time_key=getattr(self.args, "time_key", None),
            data_kwargs=getattr(self.args, "data_kwargs", None),
            color_by=getattr(self.args, "color_by", None),
            device=getattr(self.args, "device", None),
            smoke=_app._smoke(self.args), full=False)
        # THE PROJECT DICT IS THE CLI'S, in the CLI's spelling — `_setup_project`'s own keys, so
        # this is not a second definition of what a project is. Two of them are easy to get
        # wrong and both fail quietly: `recipe` is the recipe DICT (`Session.recipe.get("steps")`
        # is called on it, so a name string is an AttributeError), and `engine` is read from
        # HERE and not from `args` — omit it and `_build_session` defaults to `"mock"`, i.e. a
        # front door that invents numbers on an install that could have measured them.
        #
        # `resume=False` — THIS DOOR'S VERB IS NOT "CONTINUE". `_build_session` is the CLI's
        # resume point: a second run of the same project seeds the previous run's `emb` and
        # `pseudotime` into `state` and stamps `parent`. That is right for `manyruns open`
        # ("continue where I left off") and wrong here, where the gesture is "pick data, pick a
        # recipe, run" — a fresh analysis. `explore_once` never resumes and writes
        # `parent: None`, so leaving it on would make the two front doors record different
        # lineage for the same gesture. Worse: `run_tune_loop` snapshots `baseline_state` and
        # `_reset()` restores it on CANCEL, so cancelling the embedding step would leave the
        # RESUMED embedding in `state` and `session.close()` → `_finalize` would then report
        # `n_embedded`, `final_dim` and the whole metric suite over an embedding this run never
        # computed.
        session = _app._build_session(
            {"project": slug, "data_folder": folder, "dataset": dataset, "engine": engine,
             "modality": modality, "recipe": _app.load_recipe(recipe_name),
             **self._source_options(entry)}, args,
            resume=False)
        # After construction rather than through it: `Session.apply` reads `self.on_step` at
        # call time, and `_build_session` owns the constructor call.
        reported_caveats = set()

        def reported(rec, records):
            from rich.markup import escape

            on_step(rec, records)
            # WS2 retains DisplayScatterResult diagnostics in the session's caveats.
            # Publish them while the run is live, where captured stderr is invisible.
            for note in session.ctx.get("caveats", ()):
                if note.startswith("skipping colour channel ") and note not in reported_caveats:
                    reported_caveats.add(note)
                    gate.write(escape(note))

        session.on_step = reported
        # THE SHAPE, for the same reason and from the same place `open_run`'s LEDGER row takes
        # it — `entry.shape`. `_build_session` is the CLI's constructor and the CLI has no
        # roster row to read it off, so it leaves `Session.shape` at its `"unknown"` default;
        # `tune._record_decision` then reads `session.shape` and every TUNE row this door wrote
        # carried `shape: "unknown"` while the ledger row for the same gesture carried the real
        # one. `shape` is one of five model inputs in `decisions.examples()`, so that made half
        # the corpus this branch exists to produce blind to it.
        session.shape = entry.shape
        if session.engine == 'manylatents':
            from manyruns.pipeline.runner import ComputeProcess

            session.compute = ComputeProcess(
                cancelled=lambda: gate.closed,
                log_path=session.out_dir / 'state' / session.run_id / 'compute.log')
            # The UI only signals cancellation; the daemon run thread reaps the child.
            gate._compute = session.compute

        def may_issue():
            # Escape opens a confirmation modal. Hold the next step (and final geometry)
            # until that choice is resolved, even if the current fit finishes meanwhile.
            issuance = getattr(gate, '_issuance', None)
            while not gate.closed:
                if issuance is None or issuance.wait(.05):
                    return not gate.closed
            return False

        declared_steps = list(session.recipe.get("steps") or [])
        issued = 0
        try:
            for index, declared in enumerate(declared_steps):
                if not may_issue():
                    break
                issued = index + 1
                step = dict(declared)
                if not _params.tunable_for(step):
                    session.step(step)      # nothing offered here — run it and move on
                    continue
                # ARMED THROUGH THE GATE, not by reaching for `self.screen`. This runs on the
                # run worker, and the gate is already the one object both threads hold — its
                # callbacks are `post_message`, which is safe from any thread and drops
                # silently on a closed pump. Reading `self.screen` here would be a cross-thread
                # widget read.
                gate.arm(step)
                # WHERE THE LOOP ITSELF WRITES, asked of the loop rather than re-derived. The
                # answer is NOT `outputs/`: `tune._corpus_dir` sends a tune row to the SESSION's
                # own directory (`outputs/<slug>/decisions.jsonl`) while a ledger row goes to
                # `outputs/decisions.jsonl` — two files, and counting the wrong one would report
                # a number that has nothing to do with the row just written.
                corpus = tune._corpus_dir(session)
                # READ BEFORE AS WELL AS AFTER, and that is not belt-and-braces. A loop that was
                # accepted on the first attempt writes NO row (`_record_decision`: "accepting the
                # first thing you tried is not a decision between things"), so "the last tune row
                # on disk" is then some earlier step's — or an earlier session's, since this
                # directory is reused across runs of the same dataset. Announcing it would put a
                # ✓ under a step that decided nothing.
                seen = (_last_tune_row(corpus) or {}).get("decision_id")
                # THE CLOCK IS READ BEFORE THE LOOP because `plots/` outlives the run: a
                # `phate@1` sidecar in this directory may be an earlier session's, and the only
                # thing that separates the two is when it was written. See `_tuned_apart`.
                since = time.time()
                # BOUND, AND USED FOR ONE THING: `None` is a CANCEL, and on a cancel the loop
                # has already put `state` back — the step is not in this run at all, and
                # `plots/<step>.png` is a discarded attempt nobody accepted. Reporting what it
                # "changed" would describe a step the run does not contain.
                # The pane owns presentation through on_step records; the loop must not draw
                # to stdout, a terminal stream this surface does not own. This closes a real
                # seam, but its causal link to reported terminal residue is unproven:
                # App._print forwards to original stdout only headlessly, and manyruns has no
                # forwarding handler. Plot production and _keep_attempt remain independent.
                accepted = tune.run_tune_loop(session, step, gate.read, gate.write,
                                              show_plots=lambda rec, write: None)
                row = _last_tune_row(corpus)
                if row is not None and row.get("decision_id") != seen:
                    gate.write(corpus_line(corpus,
                                           offered=len(row.get("offered") or ()),
                                           chosen=row.get("chosen")))
                # MEASURED FROM DISK, ATTEMPT AGAINST ATTEMPT — not `session.g` either side of
                # this call, which is the same dict twice: the loop switches the metric suite
                # off for its own length and the suite is what writes keys into `g`. The
                # sidecars `_keep_attempt` moves are what make the real comparison reachable.
                # `_tuned_apart` carries the measurement and the reasons; None here means
                # nothing was tuned, which is a hidden panel rather than a gap.
                if accepted is not None and not gate.closed:
                    if session.compute is None:
                        apart = _tuned_apart(session, step, since)
                    else:
                        from types import SimpleNamespace

                        try:
                            apart = session.compute.call(
                                _tuned_apart, SimpleNamespace(out_dir=session.out_dir,
                                                             state={'X': session.state.get('X')}),
                                step, since)
                        except ComputeCancelled:
                            apart = None
                        except RuntimeError as exc:
                            from rich.markup import escape

                            # The comparison is a best-effort readout. A dead worker
                            # must still reach finalization and the normal store path.
                            note = f'tune comparison unavailable: {exc}'
                            session.ctx['caveats'].append(note)
                            gate.write(escape(note))
                            apart = None
                    if apart is not None:
                        gate.moved(*apart)
                elif accepted is None:
                    # SAID INTO THE RECORD, not only on screen. `run_tune_loop` returns None on
                    # a cancel, having already `_reset()` the state — so this step ran, left an
                    # `ok` entry in `session.steps`, and contributed NOTHING. `session.close()`
                    # below assembles the record from the state as it now stands, which is the
                    # state before this step; without this line that record claims the recipe
                    # completed and attributes the geometry of the previous step to it. See
                    # `Session.discarded`; later unissued steps are recorded separately.
                    session.discarded.append(str(step.get("name") or "?"))
        except ComputeCancelled:
            # The in-flight occurrence has no committed result, just like every
            # later occurrence. Reach the normal store path after closing the session.
            issued = index
            gate.close()
        finally:
            # The session owns the incomplete record and artifact completion marker.
            if may_issue():
                gate.write(RunScreen.FINALIZING)
            not_run = [{"index": i, "name": str(step.get("name") or "?"),
                        "reason": "run abandoned"}
                       for i, step in enumerate(declared_steps) if i >= issued]
            abandoned = gate.closed
            results = session.close(abandoned=abandoned,
                                    not_run=not_run if abandoned else ())
            # Final metrics may already be in flight when the user leaves. Keep their
            # measurements, but reconcile cancellation before start_for stores the record.
            if gate.closed and not abandoned and not results.get('cancelled'):
                results["complete"] = False
                results["cancelled"] = True
                results["caveats"].append(
                    "run abandoned during finalization: completed measurements were retained")
        return results


def _is_geometry(key: str) -> bool:
    """A key `suite.measure` emitted ABOUT THE EMBEDDING, as opposed to about itself.

    `suite_seconds` alone was the first spelling and it filtered one of six. `measure` also
    emits `<name>_note` — a STRING, written whenever a metric returns NaN, raises, or lands out
    of its admissible range (`suite.py:393,398,412,421,431`) — and the `suite_*` counters
    (`:435,436,445`). The notes are what made this load-bearing: `moved_view` ranks a
    non-numeric pair at `inf`, so a metric that became unmeasurable put a 200-character sentence
    at the TOP of a five-row panel, in precisely the badly-tuned case this pane exists to show —
    and restated a fact the honest `<name>  0.9 → —` row beneath it already carried.
    `suite_out_of_range` moving `0 → 1` is worse than it looks: it scores relative rank 1.0 and
    beats every real metric on the list.

    A PREFIX AND A SUFFIX RATHER THAN A LIST, because the list is open: `<name>_note` exists for
    every declared metric and the suite's names are config (`configs/metrics/default.yaml`), so
    a literal set would filter the twelve shipped today and leak the thirteenth. Checked against
    the shipped suite — `trustworthiness`, `continuity`, `knn_preservation`, `lid`,
    `loglog_consistency`, `anisotropy`, `participation_ratio`, `betti_0`, `betti_1`,
    `geodesic_distance_correlation`, `outlier_score`, `kernel_sparsity` — none of which starts
    with `suite_` or ends with `_note`, so nothing real is caught by either test.
    """
    return not key.startswith("suite_") and not key.endswith("_note")


def _tuned_apart(session: Any, step: dict, since: float) -> "tuple[dict, dict] | None":
    """Measure the last REJECTED attempt of `step` against the ACCEPTED one. None if it cannot.

    THIS IS WHAT SPEC §2.6 ASKED FOR, and the reason it is read off disk rather than snapshotted
    around the loop is that the loop keeps no geometry to snapshot. `run_tune_loop` sets
    `session.ctx["metrics"] = []` for its whole length — deliberately, its own docstring gives
    the reason: the suite is pairwise/O(n²) and the loop exists to show a plot fast — and the
    suite is what writes keys into `g`. Measured twice, independently: a bare
    `session.step(phate)` adds 20 keys to `g`; the same step through `run_tune_loop` adds NONE.
    So `dict(session.g)` either side of a tuned step is the same dict twice, and a readout over
    that pair renders nothing on every run. `run_tune_loop` is not modified to fix it — that
    constraint is the whole reason the gate exists.

    WHERE THE TWO COORDINATE SETS COME FROM. `tune._keep_attempt` moves each rejected attempt's
    `figspec` sidecars alongside its PNG, so a tuned step leaves `plots/phate@1.spec.npz`,
    `plots/phate@2.spec.npz` (rejected, in order) and `plots/phate.spec.npz` (accepted). Those
    exist precisely so something could say what moved between two attempts — `_keep_attempt`'s
    own note: "nothing could say what MOVED between two attempts, because only one of the two
    arrays existed". The highest `@n` is the attempt the user was looking at when they decided
    to tune once more, which is the one the accepted attempt is an answer to.

    IT IS THE PLOTTED 2-D, NOT THE EMBEDDING, and the caption says so. `io._save_scatter`
    builds the spec as `{"coords": emb[:, :2], …}` and `figspec.save` slices again
    (`figspec.save`), so a sidecar holds the two columns that were DRAWN whatever
    `n_components` the step ran at — the accepted label can read `phate n_components=3` over a
    pair measured on two. There is no alternative source: the full embedding of a REJECTED
    attempt is persisted nowhere. The DIRECTION is honest, because both sides are truncated
    identically and that is what the panel claims; the absolute values are not, and they will
    not match the g-vector `geometry_view` draws directly beneath them in the same pane
    (`runner.py:974` measures the full `emb`). Shape-dependent bounds follow the same
    truncation: `participation_ratio: {in: [1, final_dim]}` evaluates at `final_dim = 2` here,
    so it can flag a value that is legal on the run's 3-D embedding. That flag reaches nothing —
    `_is_geometry` drops every `_note` — but it is the reason the drop is not cosmetic.

    MTIME, NOT PRESENCE. `outputs/<slug>/plots/` is reused by every run of the same dataset, so
    a `@1` sidecar can be last week's; comparing against it would report movement between two
    things that were never attempts of the same decision. A set-difference cannot see this —
    `_keep_attempt` RENAMES, so the second session writes the same path — so the filter is "was
    this written after the loop started". KNOWN LIMIT, stated rather than guarded: a sidecar
    whose mtime is in the FUTURE (clock skew, an NFS server ahead of its client) passes the
    filter, so a stale attempt could still be compared against. It is the one wrong-readout path
    left; the fix is a clock this function does not have.

    NO GEOMETRY IS COMPUTED HERE. `suite.measure` is the seam `pipeline/runner.py:974` already
    uses to reach manylatents' metrics; this calls the same function with the same declared
    suite. Cost, MEASURED IN PLACE around this function on 2026-08-30 — a real `manylatents`
    run of `embed` on the 800-row `data/tree8.h5ad`, phate tuned once and accepted: **0.82 s for
    the PAIR**, i.e. ~0.41 s per `measure` call.

    THAT NUMBER IS ACT 1's, AND THE COST TRACKS THE GENE AXIS, not the row count. `measure` reads
    the working MATRIX, so the same pair on act 2's original 32,738-gene fixture was **84.9 s** —
    104x — against a 1.5 s PHATE fit, which is what made act 2 undemoable. Act 2 now ships 2,000
    highly variable genes and the pair costs **~7.9 s** (3.93 s per call, measured 2026-08-31).
    An earlier revision of this paragraph quoted 1.48 s / 1.14 s, inherited from the plan and
    taken on a 60-COLUMN TRUNCATION of the same data — a 19x under-estimate presented as measured.
    Struck rather than edited away: a cost that scales with an axis nobody varied is exactly the
    kind of number this file's conventions exist to stop being quoted.
    been measured by the thing presenting it.)

    BEST EFFORT, LIKE `_keep_attempt`, and the blanket `except` around the LOAD-AND-MEASURE
    region is deliberate where this file argues against one elsewhere (`start_for`: "a bare
    `except Exception` here was the first draft and it hid a real bug"). The difference is what
    the failure costs: that one was inside a run, this is a readout OVER a finished one and
    nothing downstream reads its return. A truncated npz, coords that no longer match `X`, an
    install with no manylatents — each means no panel, never a broken run.
    `pipeline/io.py:_save_scatter` makes the same call for the same reason ("plotting is
    best-effort"). It is SCOPED to that region rather than the whole body: with the imports, the
    name lookup and the glob inside it, an `AttributeError` from a wrong signature or a
    `figspec` rename would be permanently invisible, which is the incident `start_for` describes.
    """
    from manyruns import catalog, figspec
    from manyruns.pipeline import suite as _suite

    name = str(step.get("name") or "")
    plots = Path(getattr(session, "out_dir", None) or ".") / "plots"
    if not name or not plots.is_dir():
        return None
    prefix, at = f"{name}@", -1
    for path in plots.glob(f"{prefix}*{figspec.ARRAYS}"):
        tail = path.name[len(prefix):-len(figspec.ARRAYS)]
        if tail.isdigit() and int(tail) > at and path.stat().st_mtime >= since:
            at = int(tail)
    if at < 0:
        return None          # accepted first try: nothing was tuned, so nothing moved
    try:
        rejected = figspec.load(plots / f"{prefix}{at}")
        accepted = figspec.load(plots / name)
        if not rejected or not accepted:
            return None
        X = (getattr(session, "state", None) or {}).get("X")
        declared = catalog.load_suite()

        def _measured(coords: Any) -> dict:
            # `already={}` — the engine's own g-vector is about the ACCEPTED attempt (when it is
            # about anything at all here), and letting it win would score one side of the
            # comparison with numbers the other side could not have.
            g = _suite.measure(coords, X, declared, already={})
            return {k: v for k, v in g.items() if _is_geometry(k)}

        return _measured(rejected.get("coords")), _measured(accepted.get("coords"))
    except Exception:  # noqa: BLE001 - a readout must not cost the run it is a readout of
        return None


def _last_tune_row(out_dir: Any) -> "dict | None":
    """The newest tune row in `out_dir`'s corpus, or None when there is none.

    READ BACK RATHER THAN RECONSTRUCTED from what `run_tune_loop` returned, and the difference
    is a line that would announce a row that does not exist: `_record_decision` declines to
    write when only ONE attempt was made ("accepting the first thing you tried is not a decision
    between things"), so a loop can return an accepted step and leave the corpus untouched.

    `surface == "tune"` because two kinds of row share a file. Ledger rows land in `outputs/`
    and tune rows in the session's own directory today, so the filter is currently redundant
    there — it is here because the row this returns is about to be described as a TUNE, and a
    reader who later points both surfaces at one directory should not have to find that out from
    a wrong caption.

    Never raises: a corpus that cannot be read costs the line, not the run.
    """
    from manyruns import decisions

    try:
        rows = [r for r in decisions.read(out_dir) if r.get("surface") == "tune"]
    except OSError:
        return None
    return rows[-1] if rows else None


def prewarm_multiprocessing() -> bool:
    """Start the multiprocessing resource tracker BEFORE Textual takes over stderr.

    **Without this, every algorithm that needs a subprocess fails inside the app.** Reported from
    a real run of `embed` on `gaussian_blob`:

        ✗ phate  error  1.88s
          ValueError: bad value(s) in fds_to_keep

    The stack ends at `multiprocessing/resource_tracker.py:173 ensure_running` →
    `util.spawnv_passfds` → `_posixsubprocess.fork_exec`. `ensure_running` passes
    `sys.stderr.fileno()` to the child it spawns, and inside a running Textual app
    `sys.stderr` is `textual.app._PrintCapture`, whose `fileno()` returns **-1** (measured; and
    `sys.__stderr__.fileno()` is still 2). `fork_exec` rejects a negative descriptor, so the
    first `multiprocessing.Lock` anything creates raises — and sklearn's `Pool` creates one
    immediately.

    Textual's headless context can also install `_PrintCapture` with fileno -1. Tests
    must prewarm before entering that context too; a headless driver alone is no protection.

    The tracker is a per-process singleton and `ensure_running` short-circuits once it is up, so
    starting it here — while stderr is still the terminal's — is the whole fix. It is not a
    workaround for our code: any Textual app that shells out to multiprocessing has this bug.

    Returns whether the tracker is running, for the test; never raises, because failing to
    pre-warm must cost a subprocess-backed step, not the front door.
    """
    try:
        from multiprocessing import resource_tracker

        resource_tracker.ensure_running()
        return resource_tracker._resource_tracker._fd is not None
    except Exception:  # noqa: BLE001 - a private-ish API; degrade to the old behaviour
        return False


def run(args: Any = None) -> int:
    """Launch the app. Returns a process exit code, like `shell.run_shell`."""
    # BEFORE `App.run()`, not inside it: the whole point is to do this while `sys.stderr` is
    # still the terminal's. See `prewarm_multiprocessing`.
    prewarm_multiprocessing()
    ManyrunsApp(args).run()
    return 0


if __name__ == "__main__":  # `python -m manyruns.tui.app`
    raise SystemExit(run())
