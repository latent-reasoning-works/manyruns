"""Steps declared before their compute exists — `manyruns/pipeline/stubs.py`.

A step reaches a screen only if a recipe declares it, so a capability could not be designed
against in the GUI before someone wrote its maths. A stub closes that: it is a real step in every
respect except that it computes nothing, so the step pane, the ledger row, the refusal sentence
and the record can all be built and looked at first.

THE SAFETY PROPERTY IS THAT IT CANNOT LIE. The mock ENGINE exists for "invent a plausible
number"; this is the opposite. A stub that returned a plausible array would put fabricated
geometry in the g-vector, and the record could not tell it from a real run — the failure this
product refuses everywhere else. So: a stub declines, always, and the tests below are what make
that a guarantee rather than a convention.
"""
from __future__ import annotations

import pytest

from manyruns import catalog, vocab
from manyruns.pipeline import runner as _runner
from manyruns.pipeline import stubs as _stubs
from manyruns.pipeline.steps import _StepSkipped

np = pytest.importorskip("numpy")


# ── the safety property ──────────────────────────────────────────────────────
@pytest.mark.parametrize("name", _stubs.names())
def test_a_stub_can_never_report_ok(name):
    """THE guarantee. Called with any signature, from any group, a stub raises `_StepSkipped` —
    which `apply_step` records as `outcome="skipped"` with the reason in `detail`. There is no
    argument list that gets a number out of one.

    Parametrized over `stubs.names()` rather than a written list, so a stub added tomorrow is
    covered the day it lands and cannot be added without this holding for it.
    """
    fn = _stubs._decline(name)

    for args in ((), (1, 2, 3), ({}, {}, {}, "out", [], 2)):
        with pytest.raises(_StepSkipped):
            fn(*args)


@pytest.mark.parametrize("name", _stubs.names())
def test_the_reason_says_what_is_missing_rather_than_coming_soon(name):
    """The reason is what a user reads in the pane, so it carries the actual blocker. `granger`'s
    is the one that matters most: manyruns DELETED its granger step on measured evidence, so a stub reading
    "coming soon" would invite exactly the
    re-introduction that evidence argues against."""
    _group, reason = _stubs._STUBS[name]

    assert reason.startswith("declared, not implemented — ")
    assert len(reason) > 60, "a reason short enough to be useless is a reason nobody acts on"


def test_granger_s_reason_carries_the_condition_that_deleted_it():
    """Not decoration. The estimator exists upstream and wiring it is a few lines; what stops it
    is the null, and the stub is the only place a person will read that before doing it."""
    _group, reason = _stubs._STUBS["granger"]

    assert "MEASURED" in reason and "not one derived from the embedding" in reason
    assert "null" in reason


# ── a stub is a real step in every other respect ─────────────────────────────
@pytest.mark.parametrize("name", _stubs.names())
def test_a_stub_is_dispatchable_so_the_shell_offers_it(name):
    """The point of the whole file. `runner.dispatchable` reads the group's name table, a stub is
    in it, so the step appears in `session.available_actions` and can be driven from the app."""
    group = _stubs._STUBS[name][0]

    assert _runner.dispatchable("manylatents", name, group) is True


@pytest.mark.parametrize("name", _stubs.names())
def test_a_stub_declares_its_needs_so_the_ORDER_is_testable_first(name):
    """A stub's preconditions are real even though its maths is not, which is most of the value:
    a recipe that puts `granger` before the step producing its trajectory is refused at plan time
    today, rather than after someone writes the estimator."""
    assert vocab.STEP_NEEDS.get(name), f"{name} declares no needs, so nothing constrains its order"


def test_a_stub_before_its_producer_is_refused_at_plan_time():
    """The ordering, exercised. Every stub needs `model`; only a lightning step makes one."""
    for name, (group, _) in _stubs._STUBS.items():
        recipe = {"name": "t", "steps": [{"name": name, "group": group, "params": {}}]}
        assert "model" in vocab.unmet(recipe, "time-course"), name


def test_a_real_implementation_replaces_a_stub_silently():
    """`setdefault`, not assignment, where the stubs are merged in `runner`. Someone landing the
    real `granger` must not also have to remember to delete a line in `stubs.py` — the name in the
    group's own table wins, and `is_stub` then answers False by construction."""
    real = _runner._steps._ANALYSIS_STEPS["separation"]

    assert _stubs.is_stub(real) is False
    assert _stubs.is_stub(_stubs._decline("granger")) is True


def test_nothing_that_already_exists_is_stubbed():
    """A stub hides working code behind a refusal, so the registry must only hold things that
    exist NOWHERE. The preprocessing operations landed in manylatents-omics (#60) and are Task
    6/7's job to wire; if one of them appears here, someone stubbed over a real implementation."""
    landed = {"hvg", "filter_cells", "filter_genes", "filter_mito", "normalize",
              "transform", "detect_doublets"}

    assert landed.isdisjoint(set(_stubs.names()))


# ── the sandbox recipe ───────────────────────────────────────────────────────
def test_the_sandbox_recipe_is_legal_on_every_shape():
    """A recipe refused at plan time never reaches the step pane, which is the surface it exists
    to drive. `decode_to_gene_space` is deliberately absent from it for exactly this reason — it
    needs `genes`, which no bundled dataset carries."""
    recipe = catalog.load_recipe("sandbox")

    assert catalog.check_recipe(recipe, "sandbox") == []
    for shape in vocab.SHAPES:
        assert vocab.unmet(recipe, shape) == frozenset(), shape


def test_the_sandbox_recommends_nothing():
    """`suits: []`, like `qc`. It answers no question about anybody's data and must never compete
    for the top row of the menu."""
    recipe = catalog.load_recipe("sandbox")

    assert recipe.get("suits") == []
    assert recipe.get("claims") is None, "it computes nothing, so it asserts no structure"


def test_the_sandbox_carries_at_least_one_real_step_before_its_stubs():
    """Every stub needs `model`, so a recipe of stubs alone is ILLEGAL rather than skipped — it
    would exercise the refusal path instead of the step pane. The real steps are load-bearing."""
    steps = catalog.load_recipe("sandbox")["steps"]
    stub_names = set(_stubs.names())

    first_stub = next(i for i, s in enumerate(steps) if s["name"] in stub_names)
    assert first_stub > 0
    assert any(s["name"] not in stub_names for s in steps[:first_stub])
