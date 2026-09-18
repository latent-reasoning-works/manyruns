"""What a NEW TOOL must satisfy before it can land — the gate, in one file.

Adding `velocity_field` (an `analysis` executor, ~80 lines of compute) broke **30 tests** in a
single run. None of them was a bug: each was a golden table that pins a measured fact, and each
had a legitimate opinion about the set the new step had just joined. But thirty scattered
failures is a scavenger hunt, not a checklist — the author reads them one at a time and learns
the contract by archaeology.

So the contract is stated HERE, once, and every violation is reported in ONE message. A new step
or a new fact that is missing a row fails this file first, with the list.

**One of these guards is load-bearing beyond legibility.** (The other — module-scope import
discipline — lives in `test_base_install.py`, which must import nothing from this package; see
that file's header for why it cannot live here.)

  * `test_every_demanded_fact_can_be_supplied_by_something` stops the precondition calculus
    from being poisoned. `vocab.py:427` states the rule — "a demand lands with the supply that
    can satisfy it, never before" — and the cost of breaking it is measured there: a need
    nothing can supply prunes its step on EVERY dataset, silently, and the recipe simply stops
    appearing in the menu.
"""
from __future__ import annotations

import pytest

from manyruns import commands, narrate, toolchain as _toolchain, vocab
from manyruns.pipeline import prep, probes, runner, steps, stubs

def _steps_with_an_executor() -> set[str]:
    """Every step name manyruns can resolve to a function of its own. Deliberately NOT the
    `latent`/`lightning` names — those resolve against manylatents' catalogue, which this
    process cannot enumerate (`runner.dispatchable` says so, and `vocab.py:348` argues why a
    product-side copy of it must not exist)."""
    return (set(steps._ANALYSIS_STEPS) | set(steps._INPROC_STEPS) | set(prep._PREP_STEPS)
            | set(probes._PROBE_STEPS) | set(stubs._STUBS)
            # THE THIRD CATEGORY, and it is why this is not simply the name tables. A `tool:`
            # step resolves to no manyruns function at all — `external.run_external_step` hands
            # it to another interpreter, and `configs/tool/<name>.yaml` is what says the step
            # exists. Its vocabulary rows are as real as any other step's, so a manifest counts
            # as an executor for the purpose of "does this row name something that exists".
            | set(_toolchain.discover_tools()))


#: Names that legitimately appear in a vocabulary table with NO manyruns executor, because the
#: engine resolves them. Measured, and each one is here for a stated reason rather than to make
#: the test pass:
#:   leiden — a manylatents `latent` algorithm; `STEP_PRODUCES` keys `clusters` by NAME because
#:            exactly one of the 15 catalogued algorithms partitions (`vocab.py:385`).
#:   hvg    — a RESERVED slot in `NARROWING_STEPS`, deliberately unimplemented (spec §12,
#:            manylatents#292). `preprocess.yaml` records that naming it in a recipe would make
#:            the recipe illegal rather than incomplete.
ENGINE_RESOLVED = frozenset({"leiden", "hvg"})


def test_no_vocabulary_row_names_a_step_that_does_not_exist():
    """A dead row is the silent drift `vocab.py`'s header exists to prevent: it is read by
    nothing, it agrees with nothing, and a rename leaves it behind pointing at a name that was."""
    known = _steps_with_an_executor() | ENGINE_RESOLVED
    orphans = []
    for label, table in (("vocab.STEP_NEEDS", vocab.STEP_NEEDS),
                         ("vocab.STEP_PRODUCES", vocab.STEP_PRODUCES),
                         ("vocab.NULL_KIND", vocab.NULL_KIND),
                         ("vocab.NARROWING_STEPS", vocab.NARROWING_STEPS),
                         ("commands._SUMMARIES", commands._SUMMARIES)):
        for name in sorted(set(table) - known):
            orphans.append(f"{label}[{name!r}] names no step with an executor")
    assert not orphans, "\n  ".join(["dead vocabulary rows:"] + orphans)


def test_every_demanded_fact_can_be_supplied_by_something():
    """THE CALCULUS GATE — see the module docstring. A fact demanded and unsuppliable prunes
    its step on every dataset, and the recipe just stops appearing."""
    suppliable = (
        {f for v in vocab.SHAPE_PROVIDES.values() for f in v}
        | {f for v in vocab.GROUP_PROVIDES.values() for f in v}
        | {f for v in vocab.STEP_PRODUCES.values() for f in v}
        | set(vocab.DATASET_FACTS))
    unsuppliable = {f for v in vocab.STEP_NEEDS.values() for f in v} - suppliable
    assert not unsuppliable, (
        f"{sorted(unsuppliable)} is demanded by a step and produced by NOTHING — no shape, no "
        f"group, no step and no dataset declaration supplies it, so every recipe naming that "
        f"step is refused on every dataset. Land the supply with the demand (vocab.py:427).")


def test_every_fact_the_calculus_names_can_be_explained_to_a_human():
    """`narrate.MISSING_FACT` has to cover exactly `vocab.facts()`. A new fact with no phrase
    reaches a user as a bare word in a refusal."""
    missing = sorted(vocab.facts() - set(narrate.MISSING_FACT))
    extra = sorted(set(narrate.MISSING_FACT) - vocab.facts())
    assert not missing, f"facts with no refusal phrase: {missing} — add to narrate.MISSING_FACT"
    assert not extra, f"phrases for facts the calculus cannot name: {extra}"


def test_every_step_produced_fact_can_be_explained_when_a_filter_clears_it():
    """The other half: `CLEARED_FACT` covers exactly the facts a narrowing can take away."""
    assert set(narrate.CLEARED_FACT) == set(vocab.step_facts()), (
        "narrate.CLEARED_FACT must cover exactly vocab.step_facts() — a fact a filter can "
        "clear with no phrase leaves the user a refusal that names no cause")


def test_every_step_with_an_executor_is_reachable_on_some_engine():
    """A step no engine can dispatch is in the menu and can only ever decline — which is
    precisely what `runner.dispatchable` exists to prevent."""
    groups = {name: group for group, table in (
        ("analysis", steps._ANALYSIS_STEPS), ("prep", prep._PREP_STEPS),
        ("probe", probes._PROBE_STEPS)) for name in table}
    unreachable = [
        f"{name} ({group})" for name, group in sorted(groups.items())
        if not any(runner.dispatchable(e, name, group) for e in runner._DISPATCH)]
    assert not unreachable, f"steps no engine can run: {unreachable}"


def test_the_state_schema_is_declared_and_derived_together():
    """`_STATE_KEYS` is documentation; `_new_state` is the truth. A new slot added to one and
    not the other is the two-homes drift, one layer down from `vocab.py`'s."""
    assert set(runner._new_state()) == set(runner._STATE_KEYS)


@pytest.mark.parametrize("name", sorted(stubs._STUBS))
def test_a_stub_can_never_report_ok(name):
    """The safety property `stubs.py` opens with, pinned per stub rather than in aggregate so a
    new one that forgets it names itself."""
    group, reason = stubs._STUBS[name]
    with pytest.raises(steps._StepSkipped):
        stubs._decline(name)(None, None, {}, None, [], 3)
    assert reason and "not implemented" in reason
