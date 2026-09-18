"""The figure pane — the run's picture, drawn as each step makes it.

A figure used to arrive twice removed from the thing it shows: written to disk mid-run, then
mentioned as a file PATH in a panel after every step had already finished. On a `phate →
mioflow` run the interesting moment is what the embedding looked like *before* the trajectory
step touched it, and that is precisely the thing the terminal never showed you.

This draws each step's figure inline, as that step lands, captioned with the step that made
it — so the manifold appears, then the trajectory over it, in the order they were computed.

**Why this cannot simply be printed into the live dashboard.** `rich.Live` repaints by moving
the cursor up N lines and rewriting them, where N is what rich measured its own renderable to
be. An inline-image escape is opaque to that measurement: rich counts the ~100k base64
characters as printable width if it sees them at all, and it cannot know the image's height in
terminal cells. Either way the next repaint erases the wrong region and the display corrupts.
So the pane STOPS the live region, draws at a clean cursor, and starts a fresh one — the
figure scrolls into history where it belongs and the status panel stays pinned below it.

**Where no protocol is available the pane still runs.** It prints a captioned path attributed
to its step, which is strictly more than the flat unattributed list it replaces: with two
figures from one run, the old panel could not tell you which step produced which.
"""
from __future__ import annotations

import os
from manyruns import env as _env
from pathlib import Path
from typing import Any, Optional

#: Terminals that speak the iTerm2 inline-image protocol, by `TERM_PROGRAM`. VS Code is on the
#: list because its integrated terminal implements it behind `terminal.integrated.enableImages`
#: — and the VS Code terminal is where "why is this just a path?" was first reported.
ITERM_PROGRAMS = frozenset({"iTerm.app", "WezTerm", "vscode", "Hyper", "Tabby"})

#: Width in character cells. Wide enough to read a 3-D scatter, narrow enough that two figures
#: from one run still fit on a standard terminal without one scrolling past the other.
DEFAULT_COLS = 60


def protocol() -> Optional[str]:
    """Which graphics protocol this terminal speaks — ``"kitty"``, ``"iterm"``, or None.

    `MANYRUNS_INLINE_IMAGES` overrides: `off` disables, `kitty`/`iterm` forces one. Detection
    is env-var sniffing and env vars lie; a user who knows their terminal should be able to say
    so rather than watch us guess harder.

    Declines under tmux/screen: the multiplexer swallows the escape unless passthrough is
    configured, and a half-written payload dumped as text is worse than a path."""
    override = (_env.get("INLINE_IMAGES") or "").strip().lower()
    if override in ("0", "off", "no", "none"):
        return None
    if override in ("kitty", "iterm"):
        return override
    if os.environ.get("TMUX") or os.environ.get("STY"):
        return None
    term, program = os.environ.get("TERM", ""), os.environ.get("TERM_PROGRAM", "")
    if "kitty" in term or os.environ.get("KITTY_WINDOW_ID") or program == "ghostty":
        return "kitty"
    if program in ITERM_PROGRAMS or os.environ.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    return None


def escape(path: Path | str, proto: str, cols: int = DEFAULT_COLS) -> Optional[str]:
    """The terminal escape that draws `path` inline, scaled to `cols` character cells."""
    import base64

    try:
        payload = base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return None
    if proto == "iterm":
        return f"\033]1337;File=inline=1;width={cols};preserveAspectRatio=1:{payload}\a"
    # kitty: PNG (`f=100`), transmit-and-display (`a=T`), payload inline (`t=d`), scaled to
    # `cols` columns. Chunked because the protocol caps one escape at 4096 base64 bytes —
    # `m=1` means "more follows", `m=0` closes it. Control keys ride the first chunk only.
    chunks = [payload[i:i + 4096] for i in range(0, len(payload), 4096)] or [""]
    head = f"f=100,a=T,t=d,c={cols},"
    return "".join(
        f"\033_G{head if i == 0 else ''}m={int(i < len(chunks) - 1)};{chunk}\033\\"
        for i, chunk in enumerate(chunks)
    )


def draw(console: Any, path: Path | str, cols: int = DEFAULT_COLS) -> bool:
    """Draw one figure at the cursor. True if a picture was actually emitted.

    Written straight to the console's stream rather than through `console.print`: rich measures
    a string to wrap it, counts the base64 payload as printable width, and will insert newlines
    into the middle of it."""
    proto = protocol()
    stream = getattr(console, "file", None)
    if not proto or not getattr(console, "is_terminal", False) or stream is None:
        return False
    seq = escape(path, proto, cols)
    if seq is None:
        return False
    try:
        stream.write(seq + "\n")
        stream.flush()
    except (OSError, ValueError):  # closed or redirected stream
        return False
    return True


def hint() -> str:
    """Why the figure is a path and not a picture — actionable, not an apology."""
    if os.environ.get("TMUX") or os.environ.get("STY"):
        return "inline images off under tmux/screen (the multiplexer eats the escape)"
    if os.environ.get("TERM_PROGRAM") == "vscode":
        return "for inline images, turn on terminal.integrated.enableImages in VS Code settings"
    return ("inline images need iTerm2, kitty, Ghostty or WezTerm — "
            "or set MANYRUNS_INLINE_IMAGES=iterm")


class FigurePane:
    """Draws each step's figure as that step lands, exactly once.

    Stateful on purpose. `on_step` fires more than once per step — at `running`, then again
    when the step settles — so a pane that drew on every event would paint the same picture
    repeatedly into the scrollback. `seen` is what makes it once-per-FIGURE rather than
    once-per-event, and it is keyed on the path because that is what identifies a figure.
    """

    def __init__(self, console: Any, cols: int = DEFAULT_COLS) -> None:
        self.console = console
        self.cols = cols
        self.seen: set = set()
        self.drawn = 0
        self.inline = 0

    def pending(self, rec: dict) -> list:
        """Figures this step produced that have not been shown yet."""
        return [f for f in (rec.get("plots") or []) if f not in self.seen]

    def show(self, rec: dict, live: Any = None) -> int:
        """Draw whatever this step just produced. Returns how many figures were shown.

        `live` is the dashboard's region: stopped around the draw, restarted after. Without
        that the escape lands inside a region rich is about to erase by line count, and the
        next repaint corrupts the display — see the module docstring. Restarting in a `finally`
        so a failed draw cannot leave the dashboard dead for the rest of the run."""
        new = self.pending(rec)
        if not new:
            return 0
        running = bool(getattr(live, "is_started", False))
        if running:
            live.stop()
        try:
            for fig in new:
                self.seen.add(fig)
                name = rec.get("name") or "?"
                if draw(self.console, fig, self.cols):
                    self.inline += 1
                    self.console.print(f"[dim]{name} · {Path(fig).name}[/dim]")
                else:
                    self.console.print(f"[bold]{name}[/bold]  [dim]{fig}[/dim]")
                self.drawn += 1
        finally:
            if running:
                live.start()
        return len(new)

    def summary(self) -> str:
        """What the pane managed to show, for a footer that explains a run of bare paths."""
        if self.drawn and not self.inline:
            return hint()
        return ""
