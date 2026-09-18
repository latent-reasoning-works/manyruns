"""What this product offers to tune, and the strip that shows it.

The engine owns what is LEGAL; manyruns owns what is OFFERED, in `TUNABLE` order.
NOT the recipe's `params:`: those are overrides. `embed.yaml` declares `{n_components: 3}`
for PHATE, but the owner must still be able to turn `knn`. Declared values win over
`DEFAULTS`; the strip and prompt use `tunable_for` for the same vocabulary and baseline.

NOTHING HERE TOUCHES A WIDGET. `strip_view` returns a rich renderable, the same shape
`run.geometry_view` and `run.figures_view` have, so all of it is testable with no screen.
"""
from __future__ import annotations

from typing import Any, Optional

#: Step name -> the parameters this product offers. Short on purpose: a knob nobody turns is
#: a row that costs the picture a line. Values come from the step, then DEFAULTS.
TUNABLE: dict[str, tuple[str, ...]] = {
    # ORDERED BY HOW MUCH EACH ACTUALLY MOVES THE PICTURE, measured on the act-1 fixture
    # (spec [M8]): `t` swings branch-purity 0.366 -> 0.934 across its range, `decay` 0.830 ->
    # 0.930, and `knn` 0.894 -> 0.939, which is noise. The spec's first draft led with `knn`
    # on the strength of it being the knob everyone talks about; it is not the knob that works.
    "phate": ("t", "decay", "knn", "n_components"),
    "umap": ("n_neighbors", "min_dist", "n_components"),
    # NO `pca` ROW, and its absence is a decision rather than an omission. This table is now the
    # ONE thing deciding which steps stop a stepped run (`tui/app.py:_stepped_run` asks
    # `tunable_for` and nothing else), so a row here is an interruption, not merely a strip.
    # Spec §2.1 names the three steps that must not stop the run — `normalize`, `transform` and
    # `pca` — because "a recipe of five steps with a gate on each is four interruptions nobody
    # asked for". Measured with the row present: `embed` stopped TWICE, at `pca` and then at
    # `phate`, and `pca`'s `n_components` is a pipeline width nobody came to tune.
    "leiden": ("resolution",),
}

#: A missing override still needs a displayed value to step from. These library defaults
#: supply the interaction baseline; they never overwrite the recipe's declarations.
#: `t="auto"` keeps its nonnumeric value until the first arrow uses CONCRETE_FROM.
DEFAULTS: dict[str, Any] = {
    "t": "auto", "decay": 40, "knn": 5, "n_components": 2,
    "n_neighbors": 15, "min_dist": 0.1, "resolution": 1.0,
}

#: Below this a knob position is not a knob position — PHATE and UMAP both need at least one
#: neighbour, and arrowing into 0 turns a tuning gesture into a traceback.
#:
#: EVERY OFFERED KNOB IS IN HERE, and that is the rule rather than a preference: a knob missing
#: a floor is only safe by accident of arithmetic, and `decay` was the accident that did not
#: hold. A declared `decay: 40` is an INT, so `step_value` takes the `value + delta` branch
#: and eleven presses walk it 40 → 0 → −1 — PHATE's alpha must be positive, so the strip could
#: compose a line that fails the fit it exists to drive. `min_dist` was safe only because a
#: float steps multiplicatively and 0.1 → 0.09 never reaches zero; declared here anyway,
#: because a recipe that declares `min_dist: 0.0` breaks even that (`abs(value or 1.0)` reads
#: 0.0 as falsy and steps by a whole 0.1, straight to −0.1, which UMAP rejects).
FLOOR: dict[str, Any] = {"knn": 1, "n_neighbors": 1, "n_components": 1, "resolution": 0.1,
                         "t": 1, "decay": 1, "min_dist": 0.0}

#: Where an arrow starts when the current value is not a number. `t="auto"` is honest on screen
#: and cannot be stepped, and `t` is the knob that moves this product's picture most — so
#: without this the one knob worth turning is the one the strip cannot turn. The first press
#: makes it concrete and the user owns the number from then on.
CONCRETE_FROM: dict[str, Any] = {"t": 20}


def tunable_for(step: dict) -> list[tuple[str, Any]]:
    """`[(name, current_value), ...]` for one step, in `TUNABLE` order. Empty when this
    product offers no knob for it — which is how the run loop decides not to ask at all."""
    names = TUNABLE.get(str(step.get("name")), ())
    declared = dict(step.get("params") or {})
    return [(n, declared[n] if n in declared else DEFAULTS[n]) for n in names]


def step_value(name: str, value: Any, delta: int) -> Any:
    """One arrow-press. `+1` right, `-1` left.

    An int moves by one and a float by a tenth of itself — the SAME convention
    `tune.llm_override` already uses for a model-chosen direction, and for the same reason:
    the magnitude is the caller's, so nobody has to check a number they did not choose.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return CONCRETE_FROM.get(name, value)
    moved = value + delta if isinstance(value, int) else value + delta * abs(value or 1.0) * 0.1
    floor = FLOOR.get(name)
    if floor is not None and moved < floor:
        return type(value)(floor)
    return type(value)(moved)


def compose(name: str, value: Any) -> str:
    """The string the strip puts in `#ask-input`.

    THE WHOLE CONTRACT with the loop is that this is a string a person could have typed:
    `run_tune_loop` validates it through the same parser as typed text, and the strip is an input
    helper rather than a second answer channel (`tui/run.py`'s `on_input_submitted`, and its
    `move_value`, which composes into the same `#ask-input`).
    """
    return f"{name}={value}"


def strip_view(entries: list[tuple[str, Any]], selected: int = 0,
               draft: Optional[dict] = None) -> Optional[Any]:
    """The rows between the analysis and the prompt, or None when there is nothing to offer.

    None rather than an empty renderable so the caller can hide the whole row: a labelled but
    empty strip reads as "this step has no knobs" where the truth is "this product offers
    none", and those are different sentences.
    """
    if not entries:
        return None
    from rich.console import Group
    from rich.text import Text

    at = max(0, min(selected, len(entries) - 1))
    width = max(len(n) for n, _ in entries)
    rows: list[Any] = []
    for i, (name, value) in enumerate(entries):
        line = Text(overflow="fold")
        line.append(" ▸ " if i == at else "   ", style="bold" if i == at else "")
        line.append(f"{name:<{width}}  ", style="cyan")
        line.append(f"current attempt: {value}", style="bold" if i == at else "dim")
        if draft and name in draft:
            line.append(f"   [DRAFT, NOT SUBMITTED: {draft[name]}]", style="bold yellow")
        rows.append(line)
    rows.append(Text("Parameters: ↑↓ choose · ←→ draft · Enter submit · Tab text", style="dim"))
    return Group(*rows)
