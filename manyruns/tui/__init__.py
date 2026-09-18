"""The Textual front door — a THIRD surface, never a replacement.

Three surfaces read one dependency-free core:

    Textual app     `manyruns` on a TTY              this package (new)
    rich panels     `manyruns run` one-shot output   `shell.render_result`, unchanged
    plain text      piped, CI, no rich                `narrate.run_panel`, unchanged

The rule that makes that safe, and the reason this package is thin: **no screen may compute a
fact `narrate` does not already expose.** `narrate` is dependency-free and is already the
source the other two surfaces read (`geometry_sections`, `geometry_delta_rows`, `run_panel`).
A screen that needs something narrate lacks means the fact belongs in `narrate` — a fourth
account of what a run did is what this rule exists to prevent.

Importing this package pulls **no Textual**. `state` is the seam the screens read and it has no
Textual import at all (see its docstring), so the whole app is testable without a terminal and
`manyruns` still starts on a machine where `textual` is absent.
"""
from __future__ import annotations

from manyruns.tui.state import (
    DataEntry,
    LedgerRow,
    RunFeed,
    StepView,
)

# `ledger` and `roster` are NOT re-exported, and that is load-bearing rather than tidiness.
# `state.ledger` and `state.roster` are functions; `tui/ledger.py` and `tui/roster.py` are
# modules of the same name. Python setattrs a submodule onto its parent package on import, so
# `import manyruns.tui.app` silently replaced the function with the module and
# `from manyruns.tui import roster; roster()` raised "'module' object is not callable" —
# reproduced before this change. Reach them as `state.roster(...)`, or import the screen
# explicitly: `from manyruns.tui.ledger import LedgerScreen`.
__all__ = ["DataEntry", "LedgerRow", "RunFeed", "StepView"]
