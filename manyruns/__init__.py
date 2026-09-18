"""Manyruns — the shippable application, and the environment a practitioner works in.

Manyruns is the *product*: it accepts a user's data folder, runs automatic
exploration, and produces a first data summary. The heavy compute is delegated
one layer down, to ``manylatents``.

Layering (each arrow = "depends on / drives"):

    manyruns (this) ──▶ { manylatents, manyagents, shop }

**Nothing in this package names a learner**, and that is architecture rather than
oversight. manyruns is the ENVIRONMENT (what is legal, and the record of what
happened); the LEARNER decides what is good. The learner observes manyruns's
artifacts and drives it through the CLI as any other user would — the arrow never
points from here to there. An environment that imports its agent is not an
environment. See ``docs/environment-contract.md``
§3.5, and CLAUDE.md's "No track to a learner".

Manyruns owns: distribution, the user-facing flow — including the ``harness``
CLI (lifted from shop; ``python -m manyruns.harness``), which drives the
run → label → store product workflow and STOPS at the store — and the *config
layer* it drives the compute with (``manyruns/configs/**``, bundled into the wheel).
Manyruns does NOT implement the *compute*: DR algorithms and geometric metrics
live in ``manylatents``. The harness only *delegates* to them — it contributes the
workflow, not the math — and it does not train at all.
"""

__version__ = "0.1.0"
