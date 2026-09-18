"""The deck's palette, as a Textual theme.

WHY THIS FILE EXISTS AT ALL, since `manyruns.tcss` already avoids literals: that stylesheet is
written entirely against design tokens (`$primary`, `$text-muted`, `$surface`), which is what
makes it skinnable — but tokens have to resolve to something, and unthemed they resolve to
Textual's stock dark palette. Measured by recording the app with `vhs`: the canvas came back
near-black with an #0178D4 selection bar, inside a forest-green terminal. The terminal's palette
had been set and the app ignored it, because ANSI 0-15 is not where a Textual app gets its
colours from. This binds the tokens instead, and nothing else has to change.

BACKGROUND IS THE DECK'S SLIDE GREEN, NOT SOMETHING DARKER. The recording puts the app inside a
terminal whose background is `--slide-bg`; if the app painted its own canvas any other shade the
padding would show as a frame around it. Matching exactly is what makes the recording read as one
surface rather than a screenshot sitting on a slide.

ONE ACCENT, WHICH IS THE DECK'S RULE AND NOT AN OVERSIGHT: `--accent` is reserved in the deck for
punctuation — kickers, em-dashes, inline emphasis. Here it lands on `primary`, so it marks the
border of the panel you are in and the row `enter` acts on, and nothing else competes with it.

WHERE THE DECK RUNS OUT. It carries a surface, an ink, one accent and a wood, which is enough for
slides and not enough for a TUI: `success`, `warning` and `error` have no canonical source, and a
screen that cannot say "this is blocked" differently from "this is selected" is worse than one
slightly off-brand. Those three are derived, staying inside the deck's green/amber/orange range,
and are the only values here without a counterpart in `manyruns-vsp_2.html`.
"""

from __future__ import annotations

from textual.theme import Theme

#: Sourced from the deck's `:root` — see manifoldwerks/presentations/manyruns/manyruns-vsp_2.html
_SLIDE_BG = "#232E26"  # --slide-bg
_BG_ALT = "#2E3D30"  # --bg-dark-alt
_INK_DARK = "#1E2820"  # --ink-dark
_INK_CREAM = "#F0E8D2"  # --ink-cream
_ACCENT = "#C07030"  # --accent
_WOOD = "#7A4E24"  # --wood

MANYRUNS = Theme(
    name="manyruns",
    dark=True,
    background=_SLIDE_BG,
    surface=_BG_ALT,
    panel=_INK_DARK,
    foreground=_INK_CREAM,
    primary=_ACCENT,
    secondary=_WOOD,
    accent=_ACCENT,
    # derived — no deck counterpart, see the module docstring
    success="#8FA98A",
    warning="#C9A227",
    error="#D98A4A",
)
