"""The decision store — one line per committed choice, and what it was chosen FROM.

`index.jsonl` is an EXECUTION record: recipe, engine, steps, g-vector, what ran. Measured on
2026-07-31 it held 7 rows and **not one recorded what else was on offer** — so a selector
trained on it learns "pick `embed` or `archetypes`" and cannot see that `archetypes` was one of
eight, that six others were legal at the time, or that `contrast` was refused for a stated
reason. The choice is the label, and the label was missing.

That is the whole gap between the store and a training corpus. Everything else the selector needs — the truth side
(13/13 datasets declare `topology:`), the ladder, the outcome vocabulary, the fixed g-vector key
set, the lineage edge — was already on disk.

**`offered` is `state.ledger()` SERIALIZED, never re-derived.** `LedgerRow.as_offer()` is the one
conversion and the app hands this module the rows the screen actually drew. Recomputing the
ledger here would be a second account of what was offered, which is the same failure the TUI's
"no screen computes a fact `narrate` does not expose" rule exists to prevent — one direction
further out. `actions.ledger_offer` is the same serialization for a dataset's declaration — what
`manyruns actions` prints as `offered` — so the CLI offers the rows the ledger would draw.

**The join runs decision → run, not the other way.** `runner.identity` mints `run_id` as
`uuid4().hex[:12]` (read, not assumed), so a decision written BEFORE the run cannot know it.
Inverting it costs nothing: the decision carries its own `decision_id`, and `store.append`'s
existing `extra=` channel carries that id onto the run row. The decision is therefore complete
the moment it is written, which is the property that matters — a decision can outlive the run it
started, and if the run dies before `store.append`, the choice is still on disk.
That held for every row until tune rows arrived: a tune decision is written mid-run, so it CAN
name its run and does (`run_id`, added for the tuning loop). The direction is therefore a
property of WHEN a surface decides, not a rule about the format — see `append`.

**One line per COMMITTED choice.** Backing out of the ledger without choosing is not recorded:
it is ambiguous — interrupted, changed their mind, mis-keyed — and recording it is the first
step onto interaction telemetry, which this product has no reason to collect.

Version 1 core top-level keys are ``schema_version`` (written first), ``decision_id``,
``run_id``, ``at``, ``surface``, ``dataset``, ``shape``, ``topology``, ``offered`` and ``chosen``.
Version 2 keeps all of them, in that order, and adds ``chooser``, ``bound``, ``legal_id``,
``run_before``, ``state_before``, ``proposed``, ``chosen_params`` and ``rationale``
(environment contract §2). A row is version 2 exactly when a ``chooser`` was given; without
one, ``append`` writes the version 1 row unchanged.

**Version 2 records a proposal and the choice made against it, side by side.** ``proposed`` is
what a driver put forward (``action``, optional ``params``, ``probabilities``, ``confidence``
and a flat ``meta``, plus its own ``chooser`` and ``bound``); ``chosen`` is what was taken, and
it stays a recipe-name string in every version. Every value is recorded as supplied:
manyruns never computes a probability, a confidence or a chooser, and never fills one in.
``check_proposal`` says what is wrong with a proposal; it does not repair one.

Missing versions mean 1; readers preserve unknown keys and explicit versions for ordinary
JSON read/write round trips. Future downstream fields reserve ``extra.<caller>`` only;
no existing core field moves there. This writer creates new decisions, not row rewrites.

**What is in a row, and what is deliberately not.** Dataset name, shape, topology, recipe names,
legality reasons. No data, no coordinates, no counts — inside the federated line as currently
drawn (geometry and steps leave; the person's data does not).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

#: How a choice was constrained, strongest first (environment contract §2.2). `none` is a
#: person choosing from a rendered list.
BOUNDS = ("logits", "schema", "prompt", "none")

#: The most keys a proposal's `meta` may carry. It is stored verbatim and never interpreted,
#: so it stays small and flat.
META_KEYS = 16

#: How far a proposal's probabilities may sum from 1.
PROBABILITY_TOLERANCE = 1e-3


def index_path(out_dir: Path | str = "outputs") -> Path:
    """Sibling of `store.index_path`, and separate for the reason the module docstring gives."""
    return Path(out_dir) / "decisions.jsonl"


def new_id() -> str:
    """A decision's address. Twelve hex characters, matching `runner.identity`'s `run_id` so the
    two ids look like the same kind of thing in a row, because they are."""
    return uuid.uuid4().hex[:12]


def append(*, offered: Sequence[dict], chosen: Optional[str],
           dataset: Optional[str] = None, shape: Optional[str] = None,
           topology: Any = None, surface: str = "ledger",
           run_id: Optional[str] = None,
           out_dir: Path | str = "outputs",
           chooser: Optional[str] = None, bound: Optional[str] = None,
           proposed: Optional[dict] = None, chosen_params: Optional[dict] = None,
           rationale: Optional[str] = None, run_before: Optional[str] = None,
           state_before: Optional[dict] = None,
           legal_id: Optional[str] = None) -> Optional[str]:
    """Write one decision. Returns its `decision_id`, or `None` if nothing was written.

    Returns `None` rather than raising on any failure path: a scientist's chosen analysis must
    not fail to start because a corpus file could not be opened. The caller carries on with
    `decision_id=None`, which reads on the run row exactly as every row written before this
    module existed does.

    `topology` is COPIED here rather than joined from the dataset file later, because a
    declaration can be edited after the fact and a corpus whose labels move is not a corpus.

    `run_id` is the JOIN, and which rows can carry one follows from WHEN each is written, not
    from how the id is made. A LEDGER row is written BEFORE the run starts at all — `chose()`
    calls `append` and only then `open_run` (`tui/app.py:207,212`) — so there is no run to name
    and the field is `None`. A TUNE row is written MID-RUN, inside a loop a session is already
    driving, so the id is in hand (`tune._record_decision` reads `session.run_id`). `None` is
    likewise the honest value for every row written before this field existed: those decisions
    were not joined to anything.

    THE MINTING SITE IS DELIBERATELY NOT NAMED. There are five `identity()` call sites and which
    one a surface reaches differs per surface — the app's run path goes through `explore_once`
    to `run_manylatents`, the REPL's through `Session`, and the mock server mints none at all.
    Two successive attempts to state the mechanism in this docstring were both factually wrong.
    The timing is the reason; the mechanism is not, and it is not stable enough to cite here.

    `chooser` selects the version. None writes the version 1 row exactly as before; any value
    writes version 2 with all eight added keys, `None` where not given. The values are the
    caller's, recorded as told — `proposed` is copied, never checked or completed here (that
    is `check_proposal`, which the caller runs first).
    """
    if not offered:
        # A choice from an empty offer set is not a decision anyone can learn from, and writing
        # it would put rows in the corpus whose label has no alternatives to be a label against.
        return None
    decision_id = new_id()
    row = {
        "schema_version": 1,
        "decision_id": decision_id,
        "run_id": run_id,
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "surface": surface,
        "dataset": dataset,
        "shape": shape,
        # the TRUTH side, and the only field here that is not about the offer
        "topology": list(topology or ()),
        "offered": [dict(o) for o in offered],
        "chosen": chosen,
    }
    if chooser is not None:
        row["schema_version"] = 2
        row.update({
            "chooser": chooser,
            "bound": bound,
            "legal_id": legal_id,
            "run_before": run_before,
            "state_before": _copy(state_before),
            "proposed": _copy(proposed),
            "chosen_params": _copy(chosen_params),
            "rationale": rationale,
        })
    try:
        path = index_path(out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, separators=(",", ":")) + "\n"
        # One atomic append, no read-modify-write — `store.append`'s discipline, for the same
        # reason: the sweep runs cells in a process pool and they all write here.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except OSError:
        return None
    return decision_id


def _copy(value: Any) -> Any:
    """A JSON-plain deep copy, so a caller mutating its dict later cannot move a written row."""
    return None if value is None else json.loads(json.dumps(value))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def check_proposal(proposed: Any, offered: Sequence[dict]) -> list[str]:
    """What is wrong with a proposal against an offer set, in plain English. `[]` if valid.

    `action` must be a recipe the offer set marks `can_run`; a refused one is reported with
    its reason. `chooser` is a non-empty string and `bound` one of `BOUNDS`. `meta`, if
    present, is a flat object of at most `META_KEYS` strings and numbers. `probabilities` and
    `confidence` are optional, because not every chooser has them (a rule, a chat model, a
    person); when present, each probability names a legal recipe, every value is in [0, 1],
    and the probabilities sum to 1 within `PROBABILITY_TOLERANCE`. Nothing is filled in.
    """
    if not isinstance(proposed, dict):
        return ["the proposal must be a JSON object"]
    problems: list[str] = []
    by_recipe = {o.get("recipe"): o for o in offered}
    legal = {r for r, o in by_recipe.items() if o.get("can_run")}

    action = proposed.get("action")
    if not isinstance(action, str) or not action:
        problems.append("`action` must name a recipe")
    elif action not in by_recipe:
        problems.append(f"`action` {action!r} is not on offer; offered: {sorted(by_recipe)}")
    elif action not in legal:
        why = "; ".join(by_recipe[action].get("blocked") or []) or "not runnable here"
        problems.append(f"`action` {action!r} cannot run here: {why}")

    chooser = proposed.get("chooser")
    if not isinstance(chooser, str) or not chooser.strip():
        problems.append("`chooser` must be a non-empty string")
    if proposed.get("bound") not in BOUNDS:
        problems.append(f"`bound` must be one of {list(BOUNDS)}, not {proposed.get('bound')!r}")

    if "params" in proposed and not isinstance(proposed["params"], dict):
        problems.append("`params` must be an object")

    if "meta" in proposed:
        meta = proposed["meta"]
        if not isinstance(meta, dict):
            problems.append("`meta` must be an object")
        else:
            if len(meta) > META_KEYS:
                problems.append(f"`meta` has {len(meta)} keys; at most {META_KEYS}")
            bad = sorted(str(k) for k, v in meta.items()
                         if not isinstance(k, str) or not (isinstance(v, str) or _number(v)))
            if bad:
                problems.append(f"`meta` values must be strings or numbers: {bad}")

    if "confidence" in proposed:
        c = proposed["confidence"]
        if not _number(c) or not 0 <= c <= 1:
            problems.append(f"`confidence` must be a number in [0, 1], not {c!r}")

    if "probabilities" in proposed:
        probs = proposed["probabilities"]
        if not isinstance(probs, dict) or not probs:
            problems.append("`probabilities` must be a non-empty object of recipe → number")
        else:
            outside = sorted(str(k) for k in probs if k not in legal)
            if outside:
                problems.append(f"`probabilities` name recipes that cannot run here: {outside}")
            values = list(probs.values())
            if not all(_number(v) and 0 <= v <= 1 for v in values):
                problems.append("every probability must be a number in [0, 1]")
            elif abs(sum(values) - 1) > PROBABILITY_TOLERANCE:
                problems.append(f"`probabilities` sum to {sum(values):.4f}, not 1 "
                                f"(within {PROBABILITY_TOLERANCE})")
    return problems


def read(out_dir: Path | str = "outputs") -> Iterator[dict]:
    """Every decision, oldest first. A corrupt line is skipped rather than fatal — a half-written
    row from a killed process must not make the whole corpus unreadable. Missing versions
    mean version 1; unknown keys and explicit versions are preserved for JSON round trips."""
    path = index_path(out_dir)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            row.setdefault("schema_version", 1)
            yield row
        except json.JSONDecodeError:
            continue


def examples(out_dir: Path | str = "outputs") -> list[dict]:
    """The corpus as `(inputs, label)` pairs — one per decision that had a real alternative.

    THE TRAINING SHAPE, kept here so the selector's notion of an example and this store's row
    format cannot drift apart in two repos. Each pair is what a single-op selector sees at
    inference (the observation and the legal moves) plus what a human actually did:

        {"shape", "topology", "legal": [...], "blocked": {recipe: reason},
         "recommended", "chosen", "took_default", "overrode"}

    **A row a policy chose is not an example** (environment contract §6 item 3): a version 2
    row whose `chooser` is anything but `"human"` is left out here, by construction. Version 1
    rows carry no chooser and were all a person's choice from the ledger, so they stay.

    `overrode` is True when a person took something other than what was proposed, False when
    they took the proposal, and None when nothing was proposed (every version 1 row).

    **Decisions with fewer than two legal moves are dropped**, and the count of dropped ones is
    not hidden — `len(list(read())) - len(examples())` is the difference. A decision where only
    one analysis could run has no counterfactual: the human "chose" the only option, and an
    example like that teaches a selector the prior, not the policy.

    **`took_default` is the confound, carried rather than discovered later.** The ledger puts the
    recommendation first and the cursor starts there, so `enter` selects it — the very first real
    decision this store recorded, from driving the shipped binary, was `chosen == "embed"` with
    `embed` recommended, out of six legal moves. If every row looks like that, the corpus teaches
    `narrate.offer`'s prior with a human's name on it, and a selector trained to high accuracy on
    it has learned to agree with the default. A trainer that cannot see the flag cannot report
    the rate, so the flag is part of the example rather than something to reconstruct.
    """
    out: list[dict] = []
    for row in read(out_dir):
        if row.get("chooser") not in (None, "human"):
            continue
        offered = row.get("offered") or []
        legal = [o["recipe"] for o in offered if o.get("can_run")]
        chosen = row.get("chosen")
        if len(legal) < 2 or not chosen:
            continue
        recommended = [o["recipe"] for o in offered if o.get("recommended")]
        out.append({
            "shape": row.get("shape"),
            "topology": tuple(row.get("topology") or ()),
            "legal": legal,
            "blocked": {o["recipe"]: "; ".join(o.get("blocked") or [])
                        for o in offered if not o.get("can_run")},
            "recommended": recommended,
            "chosen": chosen,
            "took_default": chosen in recommended,
            "overrode": _overrode(row),
        })
    return out


def _overrode(row: dict) -> Optional[bool]:
    proposed = row.get("proposed")
    if not isinstance(proposed, dict) or not proposed.get("action"):
        return None
    return row.get("chosen") != proposed["action"]


def default_rate(out_dir: Path | str = "outputs") -> Optional[float]:
    """Fraction of examples where the human took the recommendation. `None` with no examples.

    The number to look at BEFORE reporting a selector's accuracy. At 1.0 the corpus is
    `narrate.offer` wearing a human's name, and a model scoring well on it has learned the
    default rather than the judgement — which is the failure mode that looks most like success.
    """
    rows = examples(out_dir)
    return sum(1 for e in rows if e["took_default"]) / len(rows) if rows else None
