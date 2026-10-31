"""The action space wrapper: one choice document in, one record out, in either mode.

Every harness, whatever it is, follows one intended use:

1. get the action document (`ActionSpace.offer`, or `manyruns actions`);
2. choose: a policy returns a choice document;
3. let manyruns run the choice (`ActionSpace.act`, or `manyruns run --choice FILE`);
4. let manyruns' own code write the record.

This module names no harness. It judges a choice against the action document whose digest the
choice names (`check_choice`), and it never writes a record itself: both modes reach
`app._run_choice`, which writes the decision row before the run and joins the run row to it.

`mode="library"` calls manyruns in this process. `mode="cli"` runs `python -m manyruns` as a
child process with the store as its working directory. Both return equal documents.

Nothing here imports Textual or the compute engine at module level, and nothing here reads a
provider key or retries a call: those belong to whatever runs the filler.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

SCHEMA_VERSION = 1

#: The prefixes a chooser may have. The full grammar is `verdict.CHOOSER_PATTERN`.
CHOOSER_PREFIXES = ("human:", "prior:", "learner:", "llm:")

#: The largest serialized size of a `meta` object, in bytes of compact JSON.
META_MAX_BYTES = 2048

#: The largest serialized size of a run-row `extra` object, in bytes of compact JSON.
EXTRA_MAX_BYTES = 8192

#: The surfaces a choice can be made on. `ladder` is read only.
_CHOICE_SURFACES = ("ledger", "tune")

_REQUIRED = ("schema_version", "surface", "dataset", "offer_sha256", "chosen", "chooser", "bound")
_OPTIONAL = ("scores", "meta", "proposed", "extra")
_PROPOSED_REQUIRED = ("chooser", "bound", "chosen", "offer_sha256")
_PROPOSED_OPTIONAL = ("scores", "meta")

#: Keys of a run-row extra that manyruns' own readers read, so a caller may not use them.
_RESERVED_EXTRA = ("verdict", "proposer")

_META_LIST_MAX = 16
_KEY_MAX = 64


class Refused(Exception):
    """manyruns refused a request. Nothing was written.

    `exit_code` is 2 for a usage error or an illegal choice, and 1 for a load failure.
    """

    def __init__(self, exit_code: int, message: str):
        super().__init__(message)
        self.exit_code = exit_code
        self.message = message


def offer_sha256(doc: dict) -> str:
    """The SHA-256 hex digest of the canonical JSON of an action document."""
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _compact_size(value: Any) -> int:
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8"))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _refuse(message: str) -> Refused:
    return Refused(2, message)


# ── checks 1 to 10: the shape, which reads no file ────────────────────────────────


def _check_keys(choice: Any) -> None:
    if not isinstance(choice, dict):
        raise _refuse(f"a choice document is a JSON object, got {type(choice).__name__}")
    missing = [k for k in _REQUIRED if k not in choice]
    if missing:
        raise _refuse(f"the choice document has no key {', '.join(missing)}")
    unknown = sorted(str(k) for k in choice if k not in _REQUIRED + _OPTIONAL)
    if unknown:
        raise _refuse(f"the choice document has the unknown key {', '.join(unknown)}")


def _check_chooser(chooser: Any, where: str) -> None:
    from manyruns import verdict

    if not isinstance(chooser, str) or not verdict.CHOOSER_PATTERN.fullmatch(chooser):
        raise _refuse(f"{where} {chooser!r} does not match the chooser grammar "
                      f"({', '.join(p + '<id>' for p in CHOOSER_PREFIXES)})")


def _check_bound(bound: Any, where: str) -> None:
    from manyruns import decisions

    if bound not in decisions.BOUNDS:
        raise _refuse(f"{where} {bound!r} is not one of {', '.join(decisions.BOUNDS)}")


def _check_scores(scores: Any, where: str) -> None:
    if not isinstance(scores, list):
        raise _refuse(f"{where} must be a list of {{\"action\", \"score\"}} objects")
    previous = math.inf
    for i, entry in enumerate(scores):
        if (not isinstance(entry, dict) or set(entry) != {"action", "score"}
                or not isinstance(entry["action"], dict) or not _is_number(entry["score"])):
            raise _refuse(f"{where}[{i}] must be {{\"action\": object, \"score\": number}}")
        if not math.isfinite(entry["score"]):
            raise _refuse(f"{where}[{i}] has a score that is not finite")
        if entry["score"] > previous:
            raise _refuse(f"{where} must not increase: entry {i} is above entry {i - 1}")
        previous = entry["score"]


def _is_meta_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _check_meta(meta: Any, where: str = "meta") -> None:
    """Scalars and short lists of scalars, at most `META_MAX_BYTES` of compact JSON."""
    if not isinstance(meta, dict):
        raise _refuse(f"{where} must be a JSON object")
    for key, value in meta.items():
        if not isinstance(key, str) or not 1 <= len(key) <= _KEY_MAX:
            raise _refuse(f"{where} key {key!r} must be a string of 1 to {_KEY_MAX} characters")
        if isinstance(value, list):
            if len(value) > _META_LIST_MAX:
                raise _refuse(f"{where}[{key!r}] holds {len(value)} items; "
                              f"at most {_META_LIST_MAX}")
            if not all(_is_meta_scalar(v) for v in value):
                raise _refuse(f"{where}[{key!r}] may hold only scalars, not lists or objects")
        elif not _is_meta_scalar(value):
            raise _refuse(f"{where}[{key!r}] must be a finite scalar or a list of scalars")
    size = _compact_size(meta)
    if size > META_MAX_BYTES:
        raise _refuse(f"{where} is {size} bytes of JSON; at most {META_MAX_BYTES}")


def _check_extra(extra: Any) -> None:
    """A JSON object for the run row. manyruns reads none of its keys after this check."""
    if not isinstance(extra, dict):
        raise _refuse("extra must be a JSON object")
    for key in extra:
        if not isinstance(key, str) or not 1 <= len(key) <= _KEY_MAX:
            raise _refuse(f"extra key {key!r} must be a string of 1 to {_KEY_MAX} characters")
        if key in _RESERVED_EXTRA:
            raise _refuse(f"extra key {key!r} is reserved: manyruns reads row['extra'][{key!r}]")
    try:
        size = _compact_size(extra)
    except (TypeError, ValueError) as exc:
        raise _refuse(f"extra must be JSON with finite numbers only: {exc}") from None
    if size > EXTRA_MAX_BYTES:
        raise _refuse(f"extra is {size} bytes of JSON; at most {EXTRA_MAX_BYTES}")


def _check_proposed(choice: dict) -> None:
    """Rules a to g of a proposed block. Rule 13, its legality, runs after the recompute."""
    proposed = choice["proposed"]
    if not isinstance(proposed, dict):
        raise _refuse("proposed must be a JSON object")
    missing = [k for k in _PROPOSED_REQUIRED if k not in proposed]
    if missing:
        raise _refuse(f"proposed has no key {', '.join(missing)}")
    unknown = sorted(str(k) for k in proposed if k not in _PROPOSED_REQUIRED + _PROPOSED_OPTIONAL)
    if unknown:
        raise _refuse(f"proposed has the unknown key {', '.join(unknown)}")
    if not str(choice["chooser"]).startswith("human:"):
        raise _refuse("a choice with proposed is a person's review: its chooser must start "
                      f"with human:, got {choice['chooser']!r}")
    _check_chooser(proposed["chooser"], "proposed.chooser")
    _check_bound(proposed["bound"], "proposed.bound")
    if proposed["offer_sha256"] != choice["offer_sha256"]:
        raise _refuse("proposed.offer_sha256 differs from offer_sha256: a proposal and its "
                      "review answer one action document")
    if "scores" in proposed:
        _check_scores(proposed["scores"], "proposed.scores")
    if "meta" in proposed:
        _check_meta(proposed["meta"], "proposed.meta")


def _check_shape(choice: Any) -> None:
    from manyruns import catalog

    _check_keys(choice)
    if choice["schema_version"] != SCHEMA_VERSION or isinstance(choice["schema_version"], bool):
        raise _refuse(f"schema_version {choice['schema_version']!r} is not {SCHEMA_VERSION}")
    if choice["surface"] not in _CHOICE_SURFACES:
        raise _refuse(f"surface {choice['surface']!r} takes no choice; choose on "
                      f"{' or '.join(_CHOICE_SURFACES)} (ladder is read only)")
    dataset = choice["dataset"]
    if not isinstance(dataset, str) or dataset not in catalog.discover_datasets():
        raise _refuse(f"dataset {dataset!r} is not a catalog dataset name")
    _check_chooser(choice["chooser"], "chooser")
    _check_bound(choice["bound"], "bound")
    if "scores" in choice:
        _check_scores(choice["scores"], "scores")
    if "meta" in choice:
        _check_meta(choice["meta"])
    if "extra" in choice:
        _check_extra(choice["extra"])
    if "proposed" in choice:
        _check_proposed(choice)


# ── check 11: the recompute ──────────────────────────────────────────────────────


def _candidates(choice: dict, store: Path | str, seed: int):
    """The action documents the choice may have been made on, in order (design decision 3)."""
    from manyruns import actions

    dataset = choice["dataset"]
    if choice["surface"] == "ledger":
        for gate in (True, False):
            yield lambda gate=gate: actions.enumerate_actions(
                "ledger", dataset=dataset, gate=gate, seed=seed, store=store)
        return
    chosen = choice["chosen"]
    recipe = chosen.get("recipe") if isinstance(chosen, dict) else None
    if not isinstance(recipe, str):
        raise _refuse(f"chosen {chosen!r} on tune must be "
                      "{\"recipe\": R, \"step\": S, \"params\": P}")
    step = chosen.get("step")
    for gate in (True, False):
        yield lambda gate=gate: actions.enumerate_actions(
            "tune", dataset=dataset, recipe=recipe, gate=gate, seed=seed, store=store)
        if isinstance(step, str):
            yield lambda gate=gate: actions.enumerate_actions(
                "tune", dataset=dataset, recipe=recipe, step=step, gate=gate, seed=seed,
                store=store)


def _recompute(choice: dict, store: Path | str, seed: int) -> dict:
    from manyruns import actions

    want = choice["offer_sha256"]
    for build in _candidates(choice, store, seed):
        try:
            doc = json.loads(json.dumps(build()))
        except actions.UsageError:
            # A step filter that names no tunable step gives no document: not this candidate.
            continue
        except actions.LoadError as exc:
            raise Refused(1, f"cannot load {exc}") from None
        if offer_sha256(doc) == want:
            return doc
    what = f"{choice['surface']} document of {choice['dataset']!r}"
    if choice["surface"] == "tune":
        what += f" for recipe {choice['chosen']['recipe']!r}"
    raise _refuse(f"stale offer: no {what}, with the gate on or off, has offer_sha256 "
                  f"{want!r}; get the action document again and choose on it")


# ── checks 12 and 13: legality, one function for a choice and a proposal ─────────


def _ledger_reason(doc: dict, recipe: str) -> Optional[str]:
    """Why `recipe` cannot run on a ledger document, or None when it can."""
    entry = next((o for o in doc["offered"] if o.get("recipe") == recipe), None)
    if entry is None:
        return f"{recipe!r} is not offered"
    if not entry.get("can_run"):
        reasons = entry.get("blocked") or ["blocked"]
        return f"{recipe!r} cannot run: {reasons[0]}"
    return None


def _knob_reason(name: str, value: Any, knob: dict) -> Optional[str]:
    """Design decision 5, rule 4, checks a to f. None when the value is legal."""
    domain = knob["domain"]
    if isinstance(value, str):
        if value in domain.get("also", []):
            return None
        return f"{name}={value!r} is not a number and not one of {domain.get('also', [])}"
    if not _is_number(value):
        return f"{name}={value!r} must be a number"
    if not math.isfinite(value):
        return f"{name}={value!r} is not finite"
    if domain["kind"] == "int" and not isinstance(value, int):
        return f"{name}={value!r} must be an integer"
    if knob.get("floor") is not None and value < knob["floor"]:
        return f"{name}={value!r} is below the floor {knob['floor']}"
    if knob.get("cap") is not None and value > knob["cap"]:
        return f"{name}={value!r} is above the cap {knob['cap']}"
    low, high = domain.get("min"), domain.get("max")
    if low is not None and (value <= low if domain.get("min_open") else value < low):
        return f"{name}={value!r} is outside the domain: the minimum is {low}"
    if high is not None and (value >= high if domain.get("max_open") else value > high):
        return f"{name}={value!r} is outside the domain: the maximum is {high}"
    return None


def _illegal(choice: dict, chosen: Any, doc: dict, store: Path | str, seed: int) -> Optional[str]:
    """Why `chosen` is not in the legal set of `doc`, or None when it is."""
    from manyruns import actions

    if choice["surface"] == "ledger":
        if not isinstance(chosen, dict) or set(chosen) != {"recipe"}:
            extra = sorted(set(chosen) - {"recipe"}) if isinstance(chosen, dict) else []
            return (f"chosen {chosen!r} on ledger must be exactly {{\"recipe\": R}}"
                    + (f"; unknown key {', '.join(extra)}" if extra else ""))
        return _ledger_reason(doc, chosen["recipe"])
    if not isinstance(chosen, dict) or set(chosen) != {"recipe", "step", "params"}:
        return f"chosen {chosen!r} on tune must be exactly {{\"recipe\", \"step\", \"params\"}}"
    recipe, step, params = chosen["recipe"], chosen["step"], chosen["params"]
    if recipe != doc["recipe"]:
        return f"recipe {recipe!r} is not the document's recipe {doc['recipe']!r}"
    ledger = actions.enumerate_actions("ledger", dataset=doc["dataset"]["name"],
                                       gate=doc["gate"], seed=seed, store=store)
    reason = _ledger_reason(ledger, recipe)
    if reason is not None:
        return reason
    if step not in doc["knobs"]:
        return f"step {step!r} is not a tunable step of {recipe!r}: {', '.join(doc['knobs'])}"
    if not isinstance(params, dict) or not params:
        return "params must name at least one knob to change"
    knobs = doc["knobs"][step]["params"]
    for name, value in params.items():
        if name not in knobs:
            return f"{name!r} is not a knob of step {step!r}: {', '.join(knobs)}"
        reason = _knob_reason(name, value, knobs[name])
        if reason is not None:
            return reason
    return None


def check_choice(choice: dict, *, store: Path | str = ".", seed: int = 42) -> dict:
    """Judge a choice document. Returns the recomputed action document it was made on.

    Raises `Refused(2, ...)` at the first failed check, and `Refused(1, ...)` when the action
    document cannot be loaded. Reads only `<store>/outputs/index.jsonl`; writes nothing.
    """
    _check_shape(choice)
    doc = _recompute(choice, store, seed)
    reason = _illegal(choice, choice["chosen"], doc, store, seed)
    if reason is not None:
        raise _refuse(f"chosen is not legal: {reason}")
    if "proposed" in choice:
        reason = _illegal(choice, choice["proposed"]["chosen"], doc, store, seed)
        if reason is not None:
            raise _refuse(f"proposed.chosen is not legal: {reason}")
    return doc


def review(choice: dict, *, ask: Callable[[dict], Any], human_id: str) -> dict:
    """Show a filler's choice to a person and return the reviewed choice.

    `ask` gets a deep copy of `choice` and returns the string `"accept"` or a replacement
    `chosen` dict. The result has `chooser: "human:<human_id>"`, `bound: "none"` (the screen
    that `ask` stands for is the bound) and the proposed block, which holds the filler's
    `chooser`, `bound`, `chosen`, `offer_sha256`, and its `scores` and `meta` when present.

    It checks no legality, reads and writes no file, and runs nothing: `act` judges both the
    final choice and the proposal.
    """
    from manyruns import annotations

    if not isinstance(human_id, str) or not re.fullmatch(annotations.PSEUDONYM_PATTERN, human_id):
        raise ValueError(f"human_id {human_id!r} is not a pseudonym: lowercase letters, digits, "
                         "_ and -, at most 32 characters, starting with a letter or digit")
    if not isinstance(choice, dict):
        raise ValueError(f"a choice document is a dict, got {type(choice).__name__}")
    missing = [k for k in _REQUIRED if k not in choice]
    if missing:
        raise ValueError(f"the choice document has no key {', '.join(missing)}")
    if "proposed" in choice:
        raise ValueError("the choice already has proposed: one review gives one row")
    answer = ask(copy.deepcopy(choice))
    if isinstance(answer, str) and answer == "accept":
        final = copy.deepcopy(choice["chosen"])
    elif isinstance(answer, dict):
        final = copy.deepcopy(answer)
    else:
        raise ValueError(f"ask must return 'accept' or a chosen dict, got "
                         f"{type(answer).__name__} {answer!r}")
    proposed = {k: copy.deepcopy(choice[k]) for k in _PROPOSED_REQUIRED}
    proposed.update({k: copy.deepcopy(choice[k]) for k in _PROPOSED_OPTIONAL if k in choice})
    out = {k: copy.deepcopy(choice[k])
           for k in ("schema_version", "surface", "dataset", "offer_sha256")}
    out.update(chosen=final, chooser="human:" + human_id, bound="none", proposed=proposed)
    if "extra" in choice:
        out["extra"] = copy.deepcopy(choice["extra"])
    return out


def label(choice: dict, chosen: dict, doc: dict) -> str:
    """The row form of a legal `chosen`: the recipe on ledger, the attempt label on tune."""
    if choice["surface"] == "ledger":
        return chosen["recipe"]
    from manyruns import tune

    declared = next(o["params"] for o in doc["offered"] if o["step"] == chosen["step"])
    return tune._label({"name": chosen["step"], "params": {**declared, **chosen["params"]}})



# ── the tool view ─────────────────────────────────────────────────────────────────


def _bound(candidates: list[tuple[Any, bool]], pick) -> Optional[tuple[Any, bool]]:
    """The tightest of `(value, open)` bounds; on a tie the open one wins."""
    present = [c for c in candidates if c[0] is not None]
    if not present:
        return None
    best = pick(value for value, _ in present)
    return best, any(is_open for value, is_open in present if value == best)


def _knob_schema(name: str, knob: dict) -> dict:
    """The JSON Schema of one knob, from its floor, cap, current value and domain only."""
    domain = knob["domain"]
    integer = domain["kind"] == "int"
    numeric: dict[str, Any] = {"type": "integer" if integer else "number"}
    low = _bound([(knob.get("floor"), False), (domain.get("min"), bool(domain.get("min_open")))],
                 max)
    if low is not None:
        value, is_open = low
        if is_open and integer:
            numeric["minimum"] = math.floor(value) + 1
        else:
            numeric["exclusiveMinimum" if is_open else "minimum"] = value
    high = _bound([(knob.get("cap"), False), (domain.get("max"), bool(domain.get("max_open")))],
                  min)
    if high is not None:
        value, is_open = high
        if is_open and integer:
            numeric["maximum"] = math.ceil(value) - 1
        else:
            numeric["exclusiveMaximum" if is_open else "maximum"] = value
    also = list(domain.get("also") or [])
    schema = ({"anyOf": [numeric, {"type": "string", "enum": also}]} if also else numeric)
    return {**schema, "description": f"{name}, current {knob['current']}"}


def as_tools(doc: dict) -> list[dict]:
    """The legal actions of one action document as tool schemas for a model filler.

    Each tool is `{"name", "description", "parameters"}`: a function tool without its
    callable, which the filler adds. The schema is a hint to the model; `check_choice` stays
    the judge. Reads no file and calls no other manyruns function.
    """
    surface = doc.get("surface")
    if surface == "ledger":
        legal = [o["recipe"] for o in doc["offered"] if o.get("can_run")]
        if not legal:
            return []
        return [{
            "name": "choose_recipe",
            "description": (f"Choose the next recipe that manyruns runs on "
                            f"{doc['dataset']['name']}. Only the listed recipes are legal."),
            "parameters": {"type": "object",
                           "properties": {"recipe": {"type": "string", "enum": legal}},
                           "required": ["recipe"], "additionalProperties": False},
        }]
    if surface == "tune":
        return [{
            "name": f"tune_{step}",
            "description": (f"Set parameters of step {step} in recipe {doc['recipe']}. "
                            "Values outside the stated bounds are refused."),
            "parameters": {"type": "object",
                           "properties": {name: _knob_schema(name, knob)
                                          for name, knob in knobs["params"].items()},
                           "minProperties": 1, "additionalProperties": False},
        } for step, knobs in doc["knobs"].items()]
    return []


def choice_from_tool_call(doc: dict, name: str, arguments: Any, *, chooser: str,
                          bound: str) -> dict:
    """The choice document of one tool call on `doc`. Checks the tool name, not legality."""
    known = [tool["name"] for tool in as_tools(doc)]
    if name not in known:
        raise _refuse(f"tool {name!r} is not a tool of this document: "
                      f"{', '.join(known) or '(none)'}")
    if not isinstance(arguments, dict):
        raise _refuse(f"the arguments of tool {name!r} must be an object, got "
                      f"{type(arguments).__name__}")
    if doc["surface"] == "ledger":
        chosen: dict = {"recipe": arguments.get("recipe")}
    else:
        chosen = {"recipe": doc["recipe"], "step": name[len("tune_"):],
                  "params": dict(arguments)}
    return {"schema_version": SCHEMA_VERSION, "surface": doc["surface"],
            "dataset": doc["dataset"]["name"], "offer_sha256": offer_sha256(doc),
            "chosen": chosen, "chooser": chooser, "bound": bound}


# ── the wrapper ───────────────────────────────────────────────────────────────────

_MODES = ("library", "cli")
_DECISION_LINE = re.compile(r"^decision ([0-9a-f]{12}) recorded in ", re.M)
_RUN_LINE = re.compile(r"^run ([0-9a-f]{12}) recorded in ", re.M)


def _child_env() -> dict:
    """The child imports the same manyruns as this process, whatever the working directory."""
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    return env


def _stderr_text(text: str) -> str:
    return text.strip() or "(no message)"


class ActionSpace:
    """The action space of one dataset, in one store, in one of two modes.

    `offer` returns the action document of a surface. `act` runs a choice document and returns
    the result document `{"schema_version", "decision_id", "run_id", "exit_code"}`. Both modes
    reach the same code, so they return equal documents and write equal records.

    Library mode changes the working directory of the whole process for the duration of
    `act`, so a threaded caller must not call `act` from two threads at once.
    """

    def __init__(self, dataset: str, *, mode: str = "library", store: str | Path = ".",
                 engine: Optional[str] = None, seed: int = 42, device: Optional[str] = None):
        if mode not in _MODES:
            raise ValueError(f"mode {mode!r} is not one of {', '.join(_MODES)}")
        if not Path(store).is_dir():
            raise ValueError(f"store {str(store)!r} is not an existing directory")
        self.dataset = dataset
        self.mode = mode
        self.store = Path(store)
        self.engine = engine
        self.seed = seed
        self.device = device

    # ── offer ──

    def offer(self, surface: str = "ledger", *, gate: bool = True,
              recipe: Optional[str] = None, step: Optional[str] = None) -> dict:
        """The action document of `surface` for this dataset. Raises `Refused` on a failure."""
        if self.mode == "library":
            return self._offer_library(surface, gate, recipe, step)
        return self._offer_cli(surface, gate, recipe, step)

    def _offer_library(self, surface, gate, recipe, step) -> dict:
        from manyruns import actions

        kwargs: dict[str, Any] = {}
        if recipe is not None:
            kwargs["recipe"] = recipe
        if step is not None:
            kwargs["step"] = step
        if surface != "ladder":
            kwargs.update(dataset=self.dataset, gate=gate, seed=self.seed, store=self.store)
        try:
            return json.loads(json.dumps(actions.enumerate_actions(surface, **kwargs)))
        except actions.UsageError as exc:
            raise Refused(2, str(exc)) from None
        except actions.LoadError as exc:
            raise Refused(1, f"cannot load {exc}") from None

    def _offer_cli(self, surface, gate, recipe, step) -> dict:
        args = ["actions"]
        if surface != "ladder":
            args.append(self.dataset)
        args += ["--surface", surface]
        if recipe is not None:
            args += ["--recipe", recipe]
        if step is not None:
            args += ["--step", step]
        if surface != "ladder":
            args += ["--gate", "on" if gate else "off", "--seed", str(self.seed)]
        done = self._child(args)
        if done.returncode != 0:
            raise Refused(2 if done.returncode == 2 else 1, _stderr_text(done.stderr))
        return json.loads(done.stdout)

    # ── act ──

    def act(self, choice: dict, *, extra: Optional[dict] = None) -> dict:
        """Run one choice document. Returns the result document; raises `Refused`.

        `extra`, when given, is a JSON object that `run --choice` writes as the run row's
        `extra` key. manyruns checks its shape and size and reads none of its keys.
        """
        if extra is not None:
            if isinstance(choice, dict) and "extra" in choice:
                raise Refused(2, "extra is given twice: as an argument and in the choice")
            choice = {**choice, "extra": extra}
        if self.mode == "library":
            return self._act_library(choice)
        return self._act_cli(choice)

    def _act_library(self, choice: dict) -> dict:
        import contextlib
        import io

        from manyruns import app

        captured = io.StringIO()
        try:
            with contextlib.chdir(self.store), contextlib.redirect_stderr(captured):
                result = app._run_choice(choice, engine=self.engine, seed=self.seed,
                                         device=self.device, project=None,
                                         parser=app._build_parser())
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            raise Refused(code, _stderr_text(captured.getvalue())) from None
        if result["decision_id"] is None:
            raise Refused(1, _stderr_text(captured.getvalue()))
        return result

    def _act_cli(self, choice: dict) -> dict:
        import tempfile

        fd, name = tempfile.mkstemp(prefix="choice-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(choice, handle)
            args = ["run", "--choice", name, "--seed", str(self.seed)]
            if self.engine is not None:
                args += ["--engine", self.engine]
            if self.device is not None:
                args += ["--device", self.device]
            done = self._child(args)
        finally:
            Path(name).unlink(missing_ok=True)
        decided = _DECISION_LINE.search(done.stdout)
        ran = _RUN_LINE.search(done.stdout)
        if done.returncode != 0 and decided is None:
            raise Refused(2 if done.returncode == 2 else 1, _stderr_text(done.stderr))
        return {"schema_version": SCHEMA_VERSION,
                "decision_id": decided.group(1) if decided else None,
                "run_id": ran.group(1) if ran else None,
                "exit_code": done.returncode}

    def _child(self, args: list[str]):
        return subprocess.run([sys.executable, "-m", "manyruns", *args], cwd=self.store,
                              capture_output=True, encoding="utf-8",
                              stdin=subprocess.DEVNULL, env=_child_env())
