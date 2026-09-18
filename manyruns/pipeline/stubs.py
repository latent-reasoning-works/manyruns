"""Steps that are DECLARED and not implemented — so the app can be driven before they exist.

A step only reaches a surface if a recipe declares it (`catalog.known_steps`), and a recipe is
only legal if the calculus can place it. That is the right rule and it has a cost: a capability
cannot be designed against in the GUI until someone writes its compute. The step pane, the
ledger, the refusal sentence and the record are all downstream of a step EXISTING, and none of
them needs the maths to be real to be worth looking at.

So a stub is a real step in every respect except that it computes nothing:

  * it is in its group's name table, so `runner.dispatchable` answers True and the shell offers it;
  * it declares its `STEP_NEEDS` / `STEP_PRODUCES`, so `vocab.unmet` places it in an order and
    refuses a recipe that puts it in the wrong one — the ordering is testable before the maths;
  * and it DECLINES when run, with the reason, which `apply_step` records as
    `outcome="skipped"` and `narrate` renders as `⊘`.

**A stub can never report `ok`.** That is the whole safety property and it is pinned by test. The
mock ENGINE already exists for "invent a plausible number"; this is the opposite — a step that
is honest about being empty, so a screen built against it cannot be built against a lie. A stub
that returned a plausible array would put fabricated geometry in the g-vector and the record
would not be able to tell it from a real run, which is the failure this product refuses
everywhere else.

**The reason string is the interesting part.** It is what a user reads in the pane, so it says
what is missing and who it is waiting on rather than "not implemented" — see `_STUBS`.
"""
from __future__ import annotations

from typing import Any

from manyruns.pipeline.steps import _StepUnsupported

#: `name -> (group, reason)`. The reason lands in the step record's `detail` and on screen.
#:
#: Only things that exist NOWHERE go here. A capability whose compute has landed upstream is
#: wired for real instead — `hvg`, `filter_cells`, `filter_genes`, `filter_mito`, `normalize` and
#: `detect_doublets` all exist in `manylatents-omics` as of #60 and are Task 6/7's job, not this
#: file's. Stubbing something that exists would hide working code behind a refusal.
_STUBS: dict[str, tuple[str, str]] = {
    # Brian's, manylatents#292. The ESTIMATOR exists (`manylatents/algorithms/cflows_granger.py`)
    # and nothing wires it to a manyruns step — deliberately. manyruns had a `granger` step and
    # DELETED it: it ordered rows by a pseudotime computed from the
    # embedding, then tested two coordinates of that same embedding against each other, and
    # rejected on pure noise in 12/12 seeds. The reason below says so, because a stub that read
    # "coming soon" would invite exactly the re-introduction the verdict argues against.
    "granger": ("analysis",
                "declared, not implemented — a directional readout needs a gene trajectory whose "
                "time axis was MEASURED, not one derived from the embedding being tested, "
                "and a data-level null that does not exist yet"),
    # RE-GROUNDED 2026-08-16 against the team backlog's "decode_to_gene_space should take a set of
    # highly variable genes as input", and the first clause of the old reason was WRONG to keep:
    # it said MIOFlow has no encoder/decoder, which manylatents#296 (open) makes false by
    # composing a GAGA one. The blocker was never the absence of a decoder. It is the FIT SPACE.
    #
    # MEASURED, twice and independently: GAGA's decoder is built as `_linear(prev, input_dim)`
    # with `input_dim` taken from the first batch, so it maps back to whatever the flow was fitted
    # on — live fits at widths 5, 15 and 3 decode to 5, 15 and 3. Through manyruns's own
    # `_run_mioflow_experiment` on a 3-D PHATE frame the decode returns 3 columns: the embedding,
    # not genes. So a decode is necessary and nowhere near sufficient.
    #
    # That is what makes the backlog's HVG framing the actual design rather than a detail — fit
    # the flow over an hvg-narrowed GENE matrix and the same decode lands in gene space by
    # construction. Two open pieces, and neither is manyruns's: the hvg step (backlog item 5,
    # Zachary Warren; manylatents#292 wires it) and a decode manyruns can call (#296 keeps the
    # GAGA network private).
    #
    # `STEP_NEEDS` is deliberately UNCHANGED at `("model", "genes")`. An `hvg` need cannot be
    # declared: `hvg` produces no fact — it is in `NARROWING_STEPS` and in neither
    # `STEP_PRODUCES` nor `GROUP_PROVIDES` — and minting one whose only producer is unimplemented
    # would prune this probe on every dataset.
    "decode_to_gene_space": ("probe",
                             "declared, not implemented — a decode lands in the space the flow "
                             "was FITTED on, and manyruns fits MIOFlow on the embedding, so gene "
                             "space needs the flow fitted over an hvg-narrowed gene matrix (hvg "
                             "is manylatents#292, open) plus a decode manyruns can call "
                             "(manylatents#296 composes one and keeps it private, open)"),
    "growth_rate": ("probe",
                    "declared, not implemented — the engine's MIOFlow exposes no growth rate; "
                    "upstream work, manylatents"),
}


def _decline(name: str) -> Any:
    """An executor for `name` that refuses, whatever signature its group calls it with.

    `*args, **kwargs` because the three groups call their executors differently — an analysis
    step takes `(state, g, params, out_dir, plots, target_dim)`, a probe takes `(model, params)`
    — and a stub has no business knowing which. It never reads an argument.
    """
    group, reason = _STUBS[name]

    def stub(*_args: Any, **_kwargs: Any) -> Any:
        # UNSUPPORTED, not a plain decline: a stub is a capability that does not exist yet,
        # which is the same event as a typo'd step name from the run's point of view. It means
        # the record does not describe the analysis the recipe asked for, so the run is not
        # `ok` (manyruns#66). A stub reporting `ok` was already impossible; this extends the
        # same property one level up, to the RUN that contains one.
        raise _StepUnsupported(reason)

    stub.__name__ = f"stub_{name}"
    stub.__doc__ = f"Declared `{group}` step, not implemented: {reason}"
    stub.__manyruns_stub__ = True   # what `test_a_stub_can_never_report_ok` keys off
    return stub


def for_group(group: str) -> dict:
    """Every stub belonging to `group`, ready to merge into that group's name table."""
    return {name: _decline(name) for name, (g, _) in _STUBS.items() if g == group}


def is_stub(fn: Any) -> bool:
    """Is this executor a stub? Read off the marker rather than the name, so a real
    implementation replacing a stub stops being one by construction."""
    return bool(getattr(fn, "__manyruns_stub__", False))


def names() -> tuple[str, ...]:
    """Every declared-but-unimplemented step name, for a surface that wants to say so."""
    return tuple(sorted(_STUBS))
