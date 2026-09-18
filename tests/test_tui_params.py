"""The parameter strip — `manyruns/tui/params.py`.

What is pinned here is that the strip is a DISPLAY over a table manyruns owns, and that its
arrow keys produce the same string a person could have typed. The moment it produces anything
else it is a second answer mechanism, which `tui/run.py` refuses in terms at its two answer
sites — `on_input_submitted` and `move_value`. (Named, not line-numbered: the numbers rotted
once already.)
"""
from __future__ import annotations

import pytest

from manyruns.tui import params


def test_the_offered_knobs_are_not_the_recipes_declared_params():
    """Recipe params are overrides, not the product's vocabulary. The shipped embed recipe
    declares only n_components, but knn and the knobs that move the picture most must remain
    offered. Silently hiding them removes the workflow this strip exists to support.
    """
    from manyruns import app

    step = next(s for s in app.load_recipe("embed")["steps"] if s["name"] == "phate")
    assert step["params"] == {"n_components": 3}
    assert params.tunable_for(step) == [
        ("t", "auto"), ("decay", 40), ("knn", 5), ("n_components", 3)]


@pytest.mark.parametrize("name", params.TUNABLE)
def test_every_offered_knob_has_a_default_and_can_be_arrowed(name):
    offered = params.tunable_for({"name": name, "params": {}})
    assert tuple(key for key, _ in offered) == params.TUNABLE[name]
    for key, value in offered:
        moved = params.step_value(key, value, +1)
        assert isinstance(moved, (int, float))
        assert moved != value


def test_the_strip_keeps_curated_order_regardless_of_override_order():
    step = {"name": "phate", "params": {
        "n_components": 3, "knn": 40, "decay": 15, "t": 20, "verbose": False}}
    assert params.tunable_for(step) == [
        ("t", 20), ("decay", 15), ("knn", 40), ("n_components", 3)]


def test_the_strip_uses_the_declared_value():
    step = {"name": "phate", "params": {"knn": 40}}
    assert dict(params.tunable_for(step))["knn"] == 40


def test_a_step_with_no_entry_offers_nothing_rather_than_guessing():
    """A step this product has not chosen knobs for is passed straight through by the run
    loop — the gate is armed per step, and no entry means no question."""
    assert params.tunable_for({"name": "normalize", "params": {}}) == []


def test_an_int_knob_stays_an_int_and_never_goes_below_its_floor():
    assert params.step_value("knn", 15, +1) == 16
    assert params.step_value("knn", 15, -1) == 14
    assert isinstance(params.step_value("knn", 15, +1), int)
    assert params.step_value("knn", 1, -1) == 1, "a neighbour count of 0 is not a knob position"


def test_a_float_knob_moves_by_a_tenth_of_itself():
    """`min_dist`, because it is the only FLOAT this product actually offers. This test used to
    pass `decay=40.0`, testing float stepping on a knob usually declared as integer 40.
    Use the float-valued knob itself to keep that path exercised.
    """
    assert params.step_value("min_dist", 0.1, +1) == pytest.approx(0.11)
    assert params.step_value("min_dist", 0.1, -1) == pytest.approx(0.09)
    assert isinstance(params.step_value("min_dist", 0.1, +1), float)


def test_an_int_knob_cannot_be_arrowed_to_zero_or_below():
    """`decay` is PHATE's alpha and must be positive. It is an INT (a declared `decay: 40`),
    so `step_value` takes the `value + delta` branch and walked 40 → 0 → −1 without a floor —
    the strip composing a line that fails the fit it exists to drive. Every offered knob has a
    floor now; see `params.FLOOR`."""
    assert params.step_value("decay", 40, -1) == 39
    assert params.step_value("decay", 1, -1) == 1, "PHATE's alpha must stay positive"
    assert isinstance(params.step_value("decay", 1, -1), int)
    # the float knob's own floor, which its arithmetic reaches only from a declared 0.0
    assert params.step_value("min_dist", 0.0, -1) == 0.0


def test_a_non_numeric_default_becomes_concrete_on_the_first_press():
    """PHATE's `t` defaults to the string "auto", and `t` is the knob that moves this picture
    most — measured, branch-purity 0.366 at t=1 against 0.934 at t=20, where `knn` moves it
    0.04 across its whole range (spec [M8]). The strip shows "auto" because that is what ran;
    the first press makes it a number."""
    assert params.step_value("t", "auto", +1) == 20
    assert params.step_value("t", 20, +1) == 21
    assert params.step_value("t", 1, -1) == 1, "t has a floor of 1"
    assert params.step_value("knn", "weird", +1) == "weird", "no concrete start -> unchanged"


def test_the_arrows_produce_exactly_what_a_person_could_have_typed():
    """THE property. `compose` is the whole contract between the strip and the loop: the
    answer arrives at `run_tune_loop` as one string that `tune.parse_overrides` parses, so the
    strip is an input helper and not a second channel."""
    from manyruns import tune

    text = params.compose("knn", 25)
    assert text == "knn=25"
    assert tune.parse_overrides(text, {"knn": 15}) == {"knn": 25}


def test_arrowing_a_relative_draft_uses_the_same_default_as_the_prompt():
    from manyruns import app, tune
    from manyruns.tui.run import RunScreen

    step = next(s for s in app.load_recipe("embed")["steps"] if s["name"] == "phate")
    screen = RunScreen()
    screen.arm_tuning(step)
    screen.tune_at = [key for key, _ in screen.tunables].index("knn")
    screen._set_pending("increase the k by one")
    screen.move_value(+1)
    assert screen.pending_text() == "knn=7"
    assert dict(screen.tunables)["knn"] == 5, "a draft must not change accepted values"
    assert tune.validated_overrides(screen.pending_text(), dict(params.tunable_for(step))) == {
        "knn": 7}


def test_the_strip_marks_the_selected_row_and_survives_an_empty_list():
    out = params.strip_view([("knn", 15), ("decay", 40.0)], selected=1)
    assert out is not None
    assert params.strip_view([], selected=0) is None


def test_draft_has_distinct_ink_and_does_not_replace_the_current_attempt():
    view = params.strip_view([("t", "auto")], draft={"t": 20})
    row = list(view.renderables)[0]
    assert "t  current attempt: auto   [DRAFT, NOT SUBMITTED: 20]" in row.plain
    assert "→" not in row.plain
    draft_start = row.plain.index("[DRAFT")
    assert any(span.start <= draft_start < span.end and span.style == "bold yellow"
               for span in row.spans)
