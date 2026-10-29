"""The driver seam's pure half: levels, the step offer, the check on a driver's choice.

Nothing here opens a socket or mounts a screen. `manyruns.driver` decides what a request may
do and what the screen should do about it; the socket and the screen only carry that out.
"""
from __future__ import annotations

import pytest

from manyruns import driver

PHATE = {"group": "latent", "name": "phate", "params": {"knn": 5, "t": "auto"}}
LEIDEN = {"group": "analysis", "name": "leiden", "params": {}}


def _offer(step=PHATE, constraints=None, question_id=3):
    return driver.step_surface(step, constraints or {}, question_id, "test-build")


# ── levels and control ───────────────────────────────────────────────────────────────────────

def test_a_new_control_is_the_human_at_interpret():
    ctrl = driver.control()
    assert ctrl["level"] == "interpret"
    assert driver.tag(ctrl) == "HUMAN"


def test_the_tag_names_the_driver_only_while_it_drives():
    ctrl = driver.attach(driver.control(), "llm:qwen3:1.7b", "schema")
    assert driver.tag(ctrl) == "HUMAN"
    ctrl = driver.cycle(driver.cycle(ctrl))
    assert ctrl["level"] == "drive"
    assert driver.tag(ctrl) == "MACHINE · llm:qwen3:1.7b"


def test_cycle_goes_interpret_propose_drive_and_back():
    ctrl = driver.attach(driver.control(), "llm:x", "schema")
    seen = []
    for _ in range(4):
        ctrl = driver.cycle(ctrl)
        seen.append(ctrl["level"])
    assert seen == ["propose", "drive", "interpret", "propose"]


def test_cycle_without_a_driver_stays_at_interpret():
    """No driver attached: there is nobody to hand the wheel to."""
    assert driver.cycle(driver.control())["level"] == "interpret"


def test_manual_is_one_press_away_and_auto_is_one_press_back():
    """Like a paddle shift: one press takes control, the same press hands it back at the level
    the driver had, with no cycling through the levels in between."""
    ctrl = driver.cycle(driver.cycle(driver.attach(driver.control(), "llm:x", "schema")))
    manual = driver.handbrake(ctrl)
    assert manual["level"] == "interpret" and manual["braked"] is True
    assert driver.tag(manual) == "MANUAL · llm:x standing by"
    # The driver stays attached: it may still talk, it may no longer act.
    assert manual["chooser"] == "llm:x"
    auto = driver.handbrake(manual)
    assert auto["level"] == "drive" and auto["braked"] is False
    assert driver.tag(auto) == "MACHINE · llm:x"


def test_manual_from_propose_returns_to_propose():
    ctrl = driver.cycle(driver.attach(driver.control(), "llm:x", "schema"))
    assert driver.handbrake(driver.handbrake(ctrl))["level"] == "propose"


def test_the_handbrake_with_no_driver_changes_nothing():
    assert driver.handbrake(driver.control()) == driver.control()


def test_a_detach_returns_control_to_the_human():
    ctrl = driver.cycle(driver.cycle(driver.attach(driver.control(), "llm:x", "schema")))
    assert driver.tag(driver.detach(ctrl)) == "HUMAN"
    assert driver.detach(ctrl)["level"] == "interpret"


def test_attach_refuses_a_bound_the_contract_does_not_have():
    with pytest.raises(ValueError, match="bound"):
        driver.attach(driver.control(), "llm:x", "telepathy")


# ── the step offer ───────────────────────────────────────────────────────────────────────────

def test_the_step_offer_lists_this_products_knobs_with_current_values():
    offer = _offer()
    assert offer["surface"] == "step"
    assert offer["step"] == "phate"
    assert list(offer["params"]) == ["t", "decay", "knn", "n_components"]
    assert offer["params"]["knn"]["current"] == 5
    assert offer["params"]["t"]["current"] == "auto"
    assert offer["params"]["decay"]["current"] == 40  # the product default, as the strip shows


def test_a_knob_at_auto_is_offered_as_the_number_it_takes_with_auto_kept():
    """`t` reads `auto` until someone sets it, and takes an integer; an offer typed `string`
    hid PHATE's most relevant knob from any numeric tool schema."""
    t = _offer()["params"]["t"]
    assert t["current"] == "auto" and t["type"] == "integer" and t["also"] == ["auto"]
    offer = _offer()
    assert driver.check_choice(offer, {"legal_id": offer["legal_id"], "params": {"t": 20}}) == []


def test_a_minimum_is_the_floor_and_an_unknown_maximum_is_said_to_be_unknown():
    p = _offer()["params"]["knn"]
    assert p["minimum"] == 1
    assert p["maximum"] is None
    assert p["maximum_known"] is False


def test_a_resolved_cap_becomes_the_maximum():
    constraints = {"knn": {"status": "resolved", "cap": 37}}
    p = _offer(constraints=constraints)["params"]["knn"]
    assert p["maximum"] == 37
    assert p["maximum_known"] is True


def test_presets_come_only_from_the_steps_own_declaration():
    step = {**PHATE, "presets": {"knn": [5, 10, 20]}}
    assert _offer(step)["params"]["knn"]["presets"] == [5, 10, 20]
    assert "presets" not in _offer()["params"]["knn"]


def test_the_legal_id_changes_with_the_question_and_the_current_values():
    a = _offer()["legal_id"]
    assert _offer(question_id=4)["legal_id"] != a
    assert _offer({**PHATE, "params": {"knn": 6, "t": "auto"}})["legal_id"] != a
    assert _offer()["legal_id"] == a


def test_a_step_with_no_knobs_offers_nothing():
    offer = _offer(LEIDEN | {"name": "pca"})
    assert offer["params"] == {}


# ── checking a driver's choice ───────────────────────────────────────────────────────────────

def test_a_valid_choice_has_no_problems_and_becomes_the_line_a_person_could_type():
    offer = _offer()
    choice = {"legal_id": offer["legal_id"], "params": {"knn": 10}}
    assert driver.check_choice(offer, choice) == []
    assert driver.answer_line(choice["params"]) == "knn=10"


@pytest.mark.parametrize("params, needle", [
    ({}, "no parameters"),
    ({"colour": "viridis"}, "unknown parameter"),
    ({"knn": "ten"}, "invalid value"),
    ({"knn": 0}, "below the minimum"),
    ({"knn": 5}, "unchanged"),
    ({"knn": 2.5}, "invalid value"),
])
def test_a_bad_choice_is_refused_with_its_reason(params, needle):
    offer = _offer()
    problems = driver.check_choice(offer, {"legal_id": offer["legal_id"], "params": params})
    assert problems and needle in " ".join(problems)


def test_a_value_above_a_known_maximum_is_refused_not_clamped():
    offer = _offer(constraints={"knn": {"status": "resolved", "cap": 37}})
    problems = driver.check_choice(offer, {"legal_id": offer["legal_id"], "params": {"knn": 500}})
    assert problems and "above the maximum" in problems[0]


def test_a_stale_legal_id_is_refused():
    offer = _offer()
    problems = driver.check_choice(offer, {"legal_id": "000000000000", "params": {"knn": 10}})
    assert problems and "stale" in problems[0]


def test_a_decision_needs_no_params():
    offer = _offer()
    assert driver.check_choice(offer, {"legal_id": offer["legal_id"], "decision": "accept"}) == []
    bad = driver.check_choice(offer, {"legal_id": offer["legal_id"], "decision": "maybe"})
    assert bad and "decision" in bad[0]


# ── requests: what each level may do ─────────────────────────────────────────────────────────

def _driving(level):
    ctrl = driver.attach(driver.control(), "llm:x", "schema")
    while ctrl["level"] != level:
        ctrl = driver.cycle(ctrl)
    return ctrl


def test_say_is_allowed_at_every_level():
    for level in driver.LEVELS:
        reply, effect = driver.handle({"verb": "say", "text": "branches look cleaner"},
                                      _driving(level), _offer())
        assert reply["ok"] is True
        assert effect == {"kind": "say", "text": "branches look cleaner", "by": "llm:x"}


def test_propose_needs_the_propose_level():
    offer = _offer()
    req = {"verb": "propose", "legal_id": offer["legal_id"], "params": {"knn": 10},
           "rationale": "fewer neighbours keep the side branch"}
    reply, effect = driver.handle(req, _driving("interpret"), offer)
    assert reply["ok"] is False and "level" in reply["error"]
    assert effect is None
    reply, effect = driver.handle(req, _driving("propose"), offer)
    assert reply["ok"] is True
    assert effect["kind"] == "propose" and effect["line"] == "knn=10"
    assert effect["rationale"] == "fewer neighbours keep the side branch"


def test_apply_needs_the_drive_level():
    offer = _offer()
    req = {"verb": "apply", "legal_id": offer["legal_id"], "params": {"knn": 10}}
    for level in ("interpret", "propose"):
        reply, effect = driver.handle(req, _driving(level), offer)
        assert reply["ok"] is False and effect is None
    reply, effect = driver.handle(req, _driving("drive"), offer)
    assert reply["ok"] is True
    assert effect == {"kind": "apply", "line": "knn=10", "question_id": 3, "by": "llm:x",
                      "bound": "schema", "legal_id": offer["legal_id"]}


def test_apply_after_the_handbrake_is_refused_until_the_human_hands_the_wheel_back():
    offer = _offer()
    braked = driver.handbrake(_driving("drive"))
    req = {"verb": "apply", "legal_id": offer["legal_id"], "params": {"knn": 10}}
    reply, effect = driver.handle(req, braked, offer)
    assert reply["ok"] is False and effect is None


def test_an_invalid_apply_is_refused_with_the_checks_reason_and_runs_nothing():
    offer = _offer()
    reply, effect = driver.handle({"verb": "apply", "legal_id": offer["legal_id"],
                                   "params": {"knn": 0}}, _driving("drive"), offer)
    assert reply["ok"] is False and "minimum" in reply["error"]
    assert effect is None


def test_apply_with_no_question_waiting_is_refused():
    reply, effect = driver.handle({"verb": "apply", "legal_id": "x", "params": {"knn": 10}},
                                  _driving("drive"), None)
    assert reply["ok"] is False and "no step" in reply["error"]
    assert effect is None


def test_actions_returns_the_current_offer_and_the_level():
    offer = _offer()
    reply, effect = driver.handle({"verb": "actions"}, _driving("propose"), offer)
    assert reply["ok"] is True and reply["offer"] == offer and reply["level"] == "propose"
    assert effect is None


def test_an_unknown_verb_is_refused():
    reply, effect = driver.handle({"verb": "rm -rf"}, _driving("drive"), _offer())
    assert reply["ok"] is False and "unknown verb" in reply["error"]
    assert effect is None


def test_a_request_that_is_not_an_object_is_refused():
    reply, effect = driver.handle(["apply"], _driving("drive"), _offer())
    assert reply["ok"] is False and effect is None


# ── conformance fixtures: the verdicts the seam owes every driver ─────────────────────────────

def _conformance():
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "driver_conformance.json"
    assert path.parent.is_dir()
    return json.loads(path.read_text())["cases"]


@pytest.mark.parametrize("case", _conformance(), ids=lambda c: c["id"])
def test_every_conformance_probe_gets_its_verdict(case):
    if case["probe"] is None:
        pytest.skip("no choice to probe: the right move is to say something, not act")
    offer = driver.step_surface(case["step"], case["constraints"], 1, "conformance")
    problems = driver.check_choice(offer, {"legal_id": offer["legal_id"], **case["probe"]})
    assert (problems == []) == (case["verdict"] == "ok"), problems


def test_every_intent_names_a_knob_the_offer_has():
    for case in _conformance():
        if isinstance(case["intent"], dict):
            offer = driver.step_surface(case["step"], case["constraints"], 1, "conformance")
            assert set(case["intent"]["params"]) <= set(offer["params"]), case["id"]


def test_the_fixtures_are_pinned_to_the_protocol_they_test():
    import json
    from pathlib import Path

    doc = json.loads((Path(__file__).parent / "fixtures" / "driver_conformance.json").read_text())
    assert doc["protocol_version"] == driver.PROTOCOL_VERSION


def test_every_fixture_carries_the_offer_the_seat_would_send():
    """A driver's conformance harness lives in another repository and imports nothing from
    manyruns, so each case carries the exact offer (question 1, build `conformance`)."""
    for case in _conformance():
        assert case["offer"] == driver.step_surface(case["step"], case["constraints"], 1,
                                                    "conformance"), case["id"]
