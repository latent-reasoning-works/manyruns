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
further out. `manyruns actions --surface ledger` prints the same conversion for a dataset's
declaration, so `offered` here and `offered` there both come from `LedgerRow.as_offer`.

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
``run_id``, ``at``, ``surface``, ``chooser``, ``bound``, ``legal_id``, ``dataset``, ``shape``,
``topology``, ``offered`` and ``chosen``. ``chooser``, ``bound`` and ``legal_id`` (who chose,
which tier enforced the legal set, and which task space the choice was over) are core fields
from the environment contract §2, not downstream fields, so the ``extra.<caller>`` rule below
does not apply to them. They were added without a version bump; a row written before them
simply lacks them, and ``read`` does not fill them in.
Two more core keys are optional, and only ``run --choice`` writes them, after ``chosen``:
``gate`` (whether the action document the choice was judged on had the geometry gate on) and
then ``proposed`` (the filler's choice that a person reviewed: ``chooser``, ``bound``,
``chosen``, ``offer_sha256``, and ``scores`` and ``meta`` when present). A row without
``gate`` reads as missing, not as ``false``. A row without ``proposed`` reads as missing, not
as an accept. No row stores whether the person overrode the filler; a reader compares
``proposed.chosen`` with ``chosen``.
Missing versions mean 1; readers preserve unknown keys and explicit versions for ordinary
JSON read/write round trips. Future downstream fields reserve ``extra.<caller>`` only;
no existing core field moves there. This writer creates new decisions, not row rewrites.

**What is in a row, and what is deliberately not.** Dataset name, shape, topology, recipe names,
legality reasons. No data, no coordinates, no counts — inside the federated line as currently
drawn (geometry and steps leave; the person's data does not).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from manyruns import vocab

#: The tiers that can enforce a legal set, strongest first (environment contract §2.2). `none`
#: is a person choosing from a rendered list.
BOUNDS = ("logits", "schema", "prompt", "none")


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
           chooser: str = "human", bound: str = "none",
           gate: Optional[bool] = None, proposed: Optional[dict] = None,
           out_dir: Path | str = "outputs") -> Optional[str]:
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

    `chooser` is set by the CALLER that made the choice: `"human"` for a person, or a policy
    identifier such as `"llm:<model>"`. It is checked only for being a non-empty string.
    `bound` must come from the code path that ENFORCED the legal set (logit mask, schema,
    prompt, or `none`), never from what the caller intended. These two raise `ValueError`
    instead of returning `None`: a bad value is a caller bug, not an I/O failure, and a wrong
    `bound` is a silent lie in the corpus. `legal_id` is computed here and cannot be passed in,
    so every row's id is made the same way (see `_legal_set`).

    `gate` and `proposed` are written only when they are not `None`, after `chosen`, in that
    order; `proposed` is stored as a JSON copy and its keys are not checked here
    (`action_space.check_choice` checks them). A `gate` that is not a `bool`, or a `proposed`
    that is not a dict, raises `ValueError` with the other caller-bug checks.
    """
    if not isinstance(chooser, str) or not chooser:
        raise ValueError(f"chooser must be a non-empty string, got {chooser!r}")
    if bound not in BOUNDS:
        raise ValueError(f"bound must be one of {BOUNDS}, got {bound!r}")
    if gate is not None and not isinstance(gate, bool):
        raise ValueError(f"gate must be True, False or None, got {gate!r}")
    if proposed is not None and not isinstance(proposed, dict):
        raise ValueError(f"proposed must be a dict or None, got {type(proposed).__name__}")
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
        "chooser": chooser,
        "bound": bound,
        "legal_id": _legal_id(offered),
        "dataset": dataset,
        "shape": shape,
        # the TRUTH side, and the only field here that is not about the offer
        "topology": list(topology or ()),
        "offered": [dict(o) for o in offered],
        "chosen": chosen,
    }
    if gate is not None:
        row["gate"] = gate
    if proposed is not None:
        row["proposed"] = json.loads(json.dumps(proposed))
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
    # Only after the row is on disk, so an event never names a decision the corpus lacks. The
    # trace envelope's `agent_id` is this row's `chooser`: one name for who chose, and the row
    # itself gains no `agent_id` key. A ledger row has no run yet, so its event lands in the
    # sink's `unknown-run` file.
    from manyruns import trace as _trace

    _trace.current().emit(_trace.stamp(
        {"kind": "decision", "decision_id": decision_id, "surface": surface,
         "chosen": chosen, "n_offered": len(offered)},
        run_id=run_id, agent_id=chooser))
    return decision_id


def _legal_set(offered: Sequence[dict]) -> list[str]:
    """The sorted, distinct names of the entries that could run.

    An entry's name is its `recipe` (ledger), else its `label` (tune), else its `step`; an entry
    with none of them is skipped. A tune attempt is named by `label` rather than `step` because
    two attempts of one step differ only by params, and the label renders them. An entry is
    legal unless it says `can_run` is false: a tune attempt carries no `can_run`, and it ran.
    """
    names = set()
    for entry in offered:
        name = entry.get("recipe", entry.get("label", entry.get("step")))
        if name is not None and entry.get("can_run", True):
            names.add(name)
    return sorted(names)


def _legal_id(offered: Sequence[dict]) -> str:
    """Twelve hex characters of a hash of the legal set and `vocab.VERSION`, read now so a
    version bump takes effect. Another repository can recompute it from a row's `offered` and
    a pinned `VERSION` without importing manyruns."""
    legal = _legal_set(offered)
    return hashlib.sha256((json.dumps(sorted(set(legal))) + vocab.VERSION).encode()).hexdigest()[:12]


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


def examples(out_dir: Path | str = "outputs", *, surface: Optional[str] = "ledger",
             exclude_chooser: str | Iterable[str] | None = None) -> list[dict]:
    """The corpus as `(inputs, label)` pairs — one per decision that had a real alternative.

    THE TRAINING SHAPE, kept here so the selector's notion of an example and this store's row
    format cannot drift apart in two repos. Each pair is what a single-op selector sees at
    inference (the observation and the legal moves) plus what a human actually did:

        {"shape", "topology", "legal": [...], "blocked": {recipe: reason},
         "recommended", "chosen", "took_default"}

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

    **Filters select rows; they never widen the pair.** `surface` keeps one surface (`None`
    keeps all); a row with no `surface` key is a ledger row. `exclude_chooser` drops rows whose
    `chooser` equals a value, or starts with a value that ends in `":"` — so `"llm:"` drops
    every model row. A row with no `chooser` key was a person's, written before the field
    existed. That is how a trainer keeps a policy's own choices out of its training data
    (environment contract §2.1) without a `chooser` key in the pair.
    """
    if exclude_chooser is None:
        excluded: tuple[str, ...] = ()
    elif isinstance(exclude_chooser, str):
        excluded = (exclude_chooser,)
    else:
        excluded = tuple(exclude_chooser)
    out: list[dict] = []
    for row in read(out_dir):
        if surface is not None and row.get("surface", "ledger") != surface:
            continue
        chooser = row.get("chooser", "human")
        if any(chooser == v or (v.endswith(":") and str(chooser).startswith(v)) for v in excluded):
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
        })
    return out


def default_rate(out_dir: Path | str = "outputs") -> Optional[float]:
    """Fraction of examples where the human took the recommendation. `None` with no examples.

    The number to look at BEFORE reporting a selector's accuracy. At 1.0 the corpus is
    `narrate.offer` wearing a human's name, and a model scoring well on it has learned the
    default rather than the judgement — which is the failure mode that looks most like success.
    """
    rows = examples(out_dir)
    return sum(1 for e in rows if e["took_default"]) / len(rows) if rows else None
