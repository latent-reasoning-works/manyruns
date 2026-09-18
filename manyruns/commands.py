"""The manyruns command interface — one source of truth for commands + aliases.

Both surfaces render from this list:
  - `manyruns --help` (the CLI) shows it as the epilog.
  - the `manyruns>` REPL's `help` / `?` prints it.

A *command* is either a **step** (a recipe action dispatched to `Session.step`: phate /
mioflow / separation) or a **meta** command (session control: run / summary / trace / help /
quit). Each carries its aliases and a one-line summary, so the alias list a user sees is
derived here — never hand-typed in two places.

The **step half is derived from the recipe catalog** (`catalog.known_steps`), not listed.
There used to be three hand-kept copies of the step vocabulary — `session._ACTIONS`,
`session.available_actions()` and the `step` rows of `COMMANDS` — and a test that asserted
they matched, i.e. one that fails *after* someone forgets rather than making forgetting
impossible. A step added to `configs/recipe/` now appears in `--help` and in the REPL's
`help` with no edit here.

What is still hand-written is `_SUMMARIES` — the one-line English for a step. That is
deliberate and it is not the same failure: a step missing from it is still LISTED, with its
group as the description, so the list cannot become a filter on what a person can reach.

**The registry is the whole catalog; the MENU is smaller, and the help list says which is
which.** `session.UNOFFERED_GROUPS` keeps `prep` and `probe` steps out of the interactive
menu (the session banner's `steps: …`, and the `try …` half of `unknown action`) while
leaving them typeable — see that constant's docstring for why. The registry does not shrink
to match, because `--help` describes every step that EXISTS. What it must not do is print the
two kinds identically: measured on the bundled catalog, that put `normalize`, `transform`,
`filter_cells`, `filter_genes`, `filter_mito`, `growth_rate` and `sample_trajectories` in
`--help` under the same `steps` heading as `phate`, and out of the banner, with nothing on
either page accounting for the gap — 23 listed, 16 offered. `Command.offered` (derived from
`session.offered`, the one rule) is what `render_help` groups them by.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _norm(token: str) -> str:
    """Normalize a typed token the way the session does: lowercase, spaces removed.

    So `"MIIOFlow"`, `"?"` all match their command."""
    return "".join(str(token).lower().split())


@dataclass(frozen=True)
class Command:
    name: str  # canonical form
    summary: str
    kind: str  # "step" | "meta"
    aliases: tuple[str, ...] = field(default_factory=tuple)
    #: Is this offered as a MOVE — i.e. does it appear in the session's menu? False for a step
    #: in an unoffered group (`session.offered`); always True for a meta command, which is this
    #: REPL's own verb and has no menu to be absent from. NOT a filter on what resolves:
    #: `resolve` and `session.resolve_step` answer for an unoffered step exactly as before.
    offered: bool = True

    def matches(self, token: str) -> bool:
        t = _norm(token)
        return t == _norm(self.name) or any(t == _norm(a) for a in self.aliases)

    def label(self) -> str:
        """`name` or `name  (alias1, alias2)` — the left column in the help list."""
        return self.name if not self.aliases else f"{self.name}  ({', '.join(self.aliases)})"


#: Typo/spelling tolerance for a derived step name. A UI concern, not a vocabulary — which is
#: why it is keyed by canonical name and may name a step this install does not have: an alias
#: for an absent step simply never matches anything.
STEP_ALIASES: dict[str, tuple[str, ...]] = {
    # `granger`/`grangercausality` are deliberately NOT here. The step was deleted
    # and an alias resolving to nothing is a worse answer than
    # "unknown action", which the session already reports along with what IS available.
    "mioflow": ("miioflow",),
}

#: One line of English per step, for the help list. Optional by construction — see the module
#: docstring. Keyed by step name because that is what the catalog yields.
_SUMMARIES: dict[str, str] = {
    "phate": "PHATE manifold embedding",
    "mioflow": "MIOFlow trajectory / flow (neural ODE)",
    "pyrovelocity": "Bayesian RNA velocity (runs in its own pinned environment)",
    "velocity_field": "coherence of a velocity field, against a permuted baseline",
    "separation": "condition separation in the embedding (silhouette)",
    "composition": "population composition shift between conditions (cluster abundance)",
}

#: Session control. Hand-written and staying that way: these are this REPL's own verbs, not a
#: vocabulary any config can grow.
META: tuple[Command, ...] = (
    # No "auto-step the REMAINING actions" any more: `Session.run_recipe` no longer skips a
    # step whose name already ran. Re-running a step with different params is the gesture
    # mix-and-match is made of, and the old de-dup silently refused it.
    Command("run", "run the recipe's steps, in order", "meta", ("all",)),
    Command("summary", "write the markdown data summary", "meta"),
    Command("trace", "show the emergent trace + g-vector", "meta", ("status",)),
    Command("help", "show this command list", "meta", ("?",)),
    Command("quit", "leave the session", "meta", ("exit", "q")),
)


def step_commands() -> tuple[Command, ...]:
    """The `step` half of the registry, derived from the recipe catalog.

    Every step the catalog declares, offered or not — `--help` describes what EXISTS. Which
    of them the session offers as a move is carried on `offered`, read from `session.offered`
    rather than re-decided here, and rendered as a separate section by `render_help`.
    """
    from manyruns import catalog, session

    return tuple(
        Command(name, _SUMMARIES.get(name) or f"{step.get('group') or 'unknown'} step",
                "step", STEP_ALIASES.get(name, ()), session.offered(step))
        for name, step in catalog.known_steps().items()
    )


def registry() -> tuple[Command, ...]:
    """The whole registry: the derived steps, then the fixed meta commands.

    Re-derived per call rather than cached. Measured on the bundled catalog: 2.98 ms for a
    `resolve()`, of which 2.79 ms is reading the three recipe YAMLs. That is once per line a
    person types at a prompt, i.e. free — and a cache here is not an optimisation but a
    reintroduction of the snapshot bug this replaced, since `$MANYRUNS_RECIPE_DIR` changes
    after import. If this ever needs to be cheap, key the cache on the recipe DIRECTORY's
    contents, not on nothing.
    """
    return step_commands() + META


def __getattr__(name: str):
    """`commands.COMMANDS` — the registry, re-derived on every access.

    It was a module-level tuple, evaluated once at import. The step half now comes from the
    recipe directory, and `$MANYRUNS_RECIPE_DIR` is set per test and per sweep arm AFTER
    manyruns is imported — so a tuple frozen at import would answer with the wrong catalog for
    the rest of the process. A module `__getattr__` keeps the name (every caller and test uses
    it) while making it a question rather than a snapshot.
    """
    if name == "COMMANDS":
        return registry()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def resolve(token: str) -> Command | None:
    """Return the command a typed token refers to (canonical or alias), or None."""
    for cmd in registry():
        if cmd.matches(token):
            return cmd
    return None


def _kind(kind: str) -> tuple[Command, ...]:
    return tuple(c for c in registry() if c.kind == kind)


#: `(kind, offered, heading)`. The step half is TWO sections, not one, because the session
#: offers only part of it as a move and a reader of `--help` otherwise has no way to tell which
#: names the banner will and will not repeat. `offered=None` means "don't split on it" — a meta
#: command is always offered.
#:
#: "typeable" is the ENGINE-FREE claim, which is the only kind this page makes — `--help` and
#: the REPL's `help` describe what exists, not what one install can dispatch (see
#: `session.available_actions`' `engine=None` docstring). Measured on this checkout: with
#: `engine=mock`, `session.step("normalize")` and `("filter_mito")` both record a step (an
#: honest decline — "no data in this state to prep"), while `("growth_rate")` answers "unknown
#: action" because `mock` has no `probe` executor. That last gap is the ENGINE axis, not this
#: one: it refuses the offered `umap` on `_inproc` in exactly the same way
#: (`test_session.py::test_a_step_no_executor_can_run_is_not_offered`), and `mock` is a
#: `DEV_ENGINE` — on `manylatents`, `dispatchable` is True for every step in this section.
_SECTIONS: tuple[tuple[str, bool | None, str], ...] = (
    ("step", True, "steps  (from the recipe catalog)"),
    ("step", False, "steps  (recipe-only — typeable, but not offered as a move)"),
    ("meta", None, "meta   (session control)"),
)


def render_help() -> str:
    """The alias list, grouped into offered steps, unoffered steps and meta, as aligned columns.

    An EMPTY section prints nothing, heading included: `$MANYRUNS_RECIPE_DIR` pointed at a
    single `latent` recipe declares no prep or probe step at all, and a heading over no rows
    would advertise a distinction that install does not have.
    """
    all_commands = registry()
    width = max(len(c.label()) for c in all_commands)
    lines = ["Commands — type these at the `manyruns>` prompt:", ""]
    for kind, offered, heading in _SECTIONS:
        rows = [c for c in all_commands
                if c.kind == kind and (offered is None or c.offered is offered)]
        if not rows:
            continue
        lines.append(f"  {heading}")
        lines += [f"    {c.label().ljust(width)}   {c.summary}" for c in rows]
        lines.append("")
    return "\n".join(lines).rstrip()
