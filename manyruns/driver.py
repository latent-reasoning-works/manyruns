"""The driver seam's pure half: who has control, what is legal now, and what a request may do.

A DRIVER ATTACHES; manyruns never calls one (environment contract §3.5). A running session
opens a socket (`tui/socket.py`), any program connects to it, and every request it sends comes
through `handle` here. This module opens nothing and paints nothing: it returns a reply for the
driver and, when a request is allowed to change the screen, one EFFECT for the screen to carry
out. That split is what makes the whole seam testable without a socket or a terminal.

**Three levels, set by the scientist with one key.** `interpret` may read and talk; `propose`
may also put a choice on screen for the scientist to accept; `drive` may also apply its own
choices. The level is the scientist's, never the driver's: no request can raise it.

**The handbrake** drops a driving driver to `interpret` at once and marks the control `braked`.
The driver stays attached, so it can still talk; it cannot act until the scientist hands the
wheel back with the same key that set the level.

**The step offer is built from what this product already offers**, never from a list a caller
supplies: `params.tunable_for` for the names and current values, `params.FLOOR` for the
minimum, and `bounds.resolved_constraints` for a cap. A maximum nobody declared is said to be
unknown (`maximum_known: False`); it is never invented, because unknown is not permission
(`bounds.resolved_constraints`). Presets exist only when the step declares them.

**A choice is checked by the parser typed text goes through** (`tune.validated_overrides`), so
a driver's `knn=10` and a person's `knn=10` are refused for the same reasons. A range check
follows it, and a value out of range is REFUSED, not clamped: a clamp would run a number the
driver never chose and record it as the driver's choice.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

#: The scientist's three settings for an attached driver, in the order the key cycles them.
LEVELS = ("interpret", "propose", "drive")

#: The verbs each level may use. Every level may read and talk.
VERBS = {
    "interpret": frozenset({"hello", "actions", "say"}),
    "propose": frozenset({"hello", "actions", "say", "propose"}),
    "drive": frozenset({"hello", "actions", "say", "propose", "apply"}),
}

#: Every verb `handle` knows, so an unknown one is refused by name rather than by level.
KNOWN = frozenset().union(*VERBS.values())

#: The two words a driving driver may answer a decision question with.
DECISIONS = ("accept", "cancel")

#: Who holds the wheel when no driver drives.
HUMAN = "HUMAN"


def control(level: str = "interpret") -> dict:
    """A fresh control: the human at `interpret`, nothing attached."""
    return {"level": level, "chooser": None, "bound": None, "braked": False}


def attach(ctrl: dict, chooser: str, bound: str) -> dict:
    """A driver says who it is. The level does not change: only the scientist raises it.

    `chooser` and `bound` are recorded as told (contract §2.1, §2.2); `bound` must be one of
    `decisions.BOUNDS` so a row never carries a tier the contract has no meaning for."""
    from manyruns.decisions import BOUNDS

    if not isinstance(chooser, str) or not chooser.strip():
        raise ValueError("`chooser` must be a non-empty string")
    if bound not in BOUNDS:
        raise ValueError(f"`bound` must be one of {list(BOUNDS)}, not {bound!r}")
    return {**ctrl, "chooser": chooser.strip(), "bound": bound}


def detach(ctrl: dict) -> dict:
    """The driver hung up: control is the human's again, at `interpret`."""
    return control()


def cycle(ctrl: dict) -> dict:
    """The scientist's key: the next level, wrapping. Releases the handbrake.

    With nothing attached there is nobody to hand the wheel to, so it stays at `interpret`."""
    if not ctrl.get("chooser"):
        return {**ctrl, "level": "interpret"}
    at = LEVELS.index(ctrl.get("level", "interpret"))
    return {**ctrl, "level": LEVELS[(at + 1) % len(LEVELS)], "braked": False}


def handbrake(ctrl: dict) -> dict:
    """Stop a driving driver now: `interpret`, and `braked` until the scientist cycles again."""
    return {**ctrl, "level": "interpret", "braked": True}


def tag(ctrl: dict) -> str:
    """The on-air tag: who has control right now.

    The driver's name only while it drives; at `propose` every choice still passes through the
    scientist's key, so the human has control."""
    if ctrl.get("level") == "drive" and ctrl.get("chooser") and not ctrl.get("braked"):
        return f"MACHINE · {ctrl['chooser']}"
    return HUMAN


def step_surface(step: dict, constraints: dict, question_id: Optional[int],
                 tool_version: str) -> dict:
    """The legal moves for the step the tune loop is asking about, as one JSON document.

    `legal_id` hashes everything that makes this offer the one it is: the step, its current
    values, the caps, the question and the build. So an answer to an earlier question, or to the
    same question before a value moved, is stale by construction."""
    from manyruns.tui.params import CONCRETE_FROM, FLOOR, tunable_for

    presets = step.get("presets") if isinstance(step.get("presets"), dict) else {}
    params: dict[str, dict] = {}
    for name, current in tunable_for(step):
        cap = (constraints or {}).get(name) or {}
        known = cap.get("status") == "resolved" and cap.get("cap") is not None
        entry = {
            "current": current,
            "type": _type_name(current),
            "minimum": FLOOR.get(name),
            "maximum": cap.get("cap") if known else None,
            "maximum_known": known,
        }
        # A KNOB AT A WORD THAT TAKES A NUMBER (`t: auto`, which `CONCRETE_FROM` makes 20 on the
        # first arrow press) is offered as the number, with the word kept in `also`.
        if isinstance(current, str) and name in CONCRETE_FROM:
            entry["type"] = _type_name(CONCRETE_FROM[name])
            entry["also"] = [current]
        if name in presets:
            entry["presets"] = list(presets[name])
        params[name] = entry
    identity = {"step": step.get("name"), "params": {n: p["current"] for n, p in params.items()},
                "caps": {n: p["maximum"] for n, p in params.items()},
                "question_id": question_id, "tool_version": tool_version}
    return {
        "surface": "step",
        "step": step.get("name"),
        "question_id": question_id,
        "params": params,
        "legal_id": hashlib.sha256(json.dumps(identity, sort_keys=True, default=str)
                                   .encode("utf-8")).hexdigest()[:12],
        "tool_version": tool_version,
    }


def _type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def answer_line(params: dict) -> str:
    """The line a person could have typed for these params — the only thing the gate accepts."""
    return " ".join(f"{name}={value}" for name, value in params.items())


def check_choice(offer: dict, choice: Any) -> list[str]:
    """What is wrong with a driver's choice against the current offer. `[]` if it may run."""
    from manyruns import tune

    if not isinstance(choice, dict):
        return ["the choice must be a JSON object"]
    if choice.get("legal_id") != offer.get("legal_id"):
        return [f"stale: this offer is {offer.get('legal_id')}, the choice was made against "
                f"{choice.get('legal_id')!r}; ask for `actions` again"]
    if "decision" in choice:
        if choice["decision"] not in DECISIONS:
            return [f"`decision` must be one of {list(DECISIONS)}, not {choice['decision']!r}"]
        return []
    params = choice.get("params")
    if not isinstance(params, dict) or not params:
        return ["no parameters: send `params` as an object, or a `decision`"]
    current = {n: p["current"] for n, p in offer["params"].items()}
    try:
        overrides = tune.validated_overrides(answer_line(params), current)
    except ValueError as error:
        return [str(error)]
    problems = []
    for name, value in overrides.items():
        p = offer["params"][name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if p["minimum"] is not None and value < p["minimum"]:
            problems.append(f"{name}={value} is below the minimum {p['minimum']}")
        if p["maximum_known"] and value > p["maximum"]:
            problems.append(f"{name}={value} is above the maximum {p['maximum']}")
    return problems


def handle(request: Any, ctrl: dict, offer: Optional[dict]) -> tuple[dict, Optional[dict]]:
    """One request → `(reply, effect)`. `effect` is None unless the screen must act.

    `offer` is the step offer for the question the tune loop is waiting on, or None when
    nothing is waiting. A request never changes `ctrl`: the level is the scientist's."""
    if not isinstance(request, dict):
        return {"ok": False, "error": "a request must be a JSON object"}, None
    verb = request.get("verb")
    if verb not in KNOWN:
        return {"ok": False, "error": f"unknown verb {verb!r}; known: {sorted(KNOWN)}"}, None
    level = ctrl.get("level", "interpret")
    if verb not in VERBS[level]:
        why = " (the handbrake is on)" if ctrl.get("braked") else ""
        return {"ok": False, "error": f"`{verb}` needs a higher level than {level!r}{why}; "
                                      "only the scientist can raise it"}, None
    by = ctrl.get("chooser")
    if verb == "hello":
        return {"ok": True, "level": level, "tag": tag(ctrl)}, None
    if verb == "actions":
        return {"ok": True, "level": level, "tag": tag(ctrl), "offer": offer}, None
    if verb == "say":
        text = request.get("text")
        if not isinstance(text, str) or not text.strip():
            return {"ok": False, "error": "`say` needs non-empty `text`"}, None
        return {"ok": True}, {"kind": "say", "text": text, "by": by}
    if offer is None:
        return {"ok": False, "error": "no step is waiting for a choice right now"}, None
    problems = check_choice(offer, request)
    if problems:
        return {"ok": False, "error": "; ".join(problems)}, None
    line = request["decision"] if "decision" in request else answer_line(request["params"])
    if verb == "propose":
        rationale = request.get("rationale")
        return {"ok": True}, {"kind": "propose", "line": line, "by": by,
                              "rationale": rationale if isinstance(rationale, str) else None,
                              "question_id": offer.get("question_id"),
                              "legal_id": offer["legal_id"]}
    return {"ok": True}, {"kind": "apply", "line": line, "question_id": offer.get("question_id"),
                          "by": by, "bound": ctrl.get("bound"), "legal_id": offer["legal_id"]}
