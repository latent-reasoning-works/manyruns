"""Per-step accept/retry loop for the `manyruns>` REPL (`app.interactive_session`).

A typed line is `<step> [key=value ...]` — strict, not natural language (e.g. `phate
knn=40`, not "run phate and change knn to 40"). The step runs, its plot (if any) is shown,
and the REPL waits for `accept` / `tune` / `cancel` before the session moves on. A tune is
not a redo: `Session.step` appends a new lineage entry rather than overwriting the last one
(see `session.py`'s `run_recipe` docstring), so every rejected attempt still lands on disk —
only the accepted attempt's embedding is what the next step consumes.
"""
from __future__ import annotations

import math
import pathlib
from typing import Any, Callable, Optional


def _coerce(raw: str) -> Any:
    """`"40"` -> 40, `"0.5"` -> 0.5, `"true"` -> True, else the string as typed."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    return raw


#: Spoken numbers a person actually types at a prompt. Not a general parser — `parse_overrides`
#: reads digits first, so this only has to cover the words that appear in "by one", "by two".
#: Deliberately short: a word list that tries to be complete becomes a second number parser.
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "half": 0.5}

#: Words that mean "make it bigger" / "make it smaller", and the sign each carries.
_DIRECTIONS = {"increase": 1, "raise": 1, "up": 1, "more": 1, "bump": 1,
               "decrease": -1, "lower": -1, "reduce": -1, "down": -1, "less": -1}

#: Words a spoken phrase carries that are never a parameter name.
_STOPWORDS = frozenset({"the", "a", "an", "to", "by", "set", "make", "it", "please",
                        *_DIRECTIONS, *_WORD_NUMBERS})


def _resolve_param(spoken: str, current: "dict | None") -> Optional[str]:
    """A spoken parameter name -> one of THIS step's actual params, or None.

    RESOLVED AGAINST THE STEP, never a global table, and that is the whole reason this works:
    "the k" is `knn` on PHATE and `n_neighbors` on UMAP, so any fixed mapping would be wrong for
    one of them. `current` is this step's offered knobs and their current values, supplied by
    `params.tunable_for` (declared overrides over defaults).

    Returns None rather than guessing when nothing matches or SEVERAL do. A wrong parameter
    silently tuned is the failure this product refuses everywhere else — `prep.TRANSFORMS` is a
    closed vocabulary for the same reason — and the caller degrades to "no change", which the
    loop refuses before another attempt starts.
    """
    if not current:
        return None
    spoken = spoken.strip().lower().lstrip("_")
    if not spoken:
        return None
    names = list(current)
    if spoken in names:
        return spoken
    # `k` -> `knn`, `neighbors` -> `n_neighbors`: a name CONTAINS the spoken word, on a token
    # boundary, so `k` does not also match `decay`.
    hits = [n for n in names
            if spoken in [part for part in n.lower().replace("-", "_").split("_")]]
    if len(hits) == 1:
        return hits[0]
    hits = [n for n in names if n.lower().startswith(spoken)]
    return hits[0] if len(hits) == 1 else None


def _spoken_override(text: str, current: "dict | None") -> dict:
    """A sentence -> `{param: value}`, or `{}` when it does not clearly name one.

    Three forms, and they are the ones the backlog asks for:

        set knn to 40 / knn to 40      an absolute value
        increase the k to 5            an absolute value, spoken
        increase the k by one          RELATIVE, so it needs the current value

    The relative form is why `current` is not optional in practice: "by one" has nothing to add
    to without it, and inventing a base would produce a number the user did not ask for. With no
    current value it returns `{}` — no change — rather than a guess.
    """
    words = [w.strip(".,!?") for w in (text or "").lower().split()]
    if not words:
        return {}
    direction = next((_DIRECTIONS[w] for w in words if w in _DIRECTIONS), None)
    relative = "by" in words and direction is not None

    # the VALUE: the last number in the sentence, digits or a word
    value: Optional[float] = None
    for w in words:
        if w in _WORD_NUMBERS:
            value = _WORD_NUMBERS[w]
        else:
            got = _coerce(w)
            if isinstance(got, (int, float)) and not isinstance(got, bool):
                value = got
    if value is None:
        return {}

    # the NAME: the last word that is not furniture and not the number itself
    spoken = next((w for w in reversed(words)
                   if w not in _STOPWORDS and _coerce(w) == w), None)
    name = _resolve_param(spoken or "", current)
    if name is None:
        return {}

    if not relative:
        return {name: _coerce(str(value))}
    base = (current or {}).get(name)
    if not isinstance(base, (int, float)) or isinstance(base, bool):
        return {}
    moved = base + direction * value
    # An int parameter stays an int — `knn` is a neighbour COUNT, and `knn=5.0` is a float where
    # the engine wants a count. `_coerce` cannot know that; the current value does.
    return {name: type(base)(moved) if isinstance(base, int) else moved}


def llm_override(text: str, current: "dict | None", *, step: "str | None" = None) -> dict:
    """TIER B: ask a model which offered parameter a free sentence is about. `{}` on any failure.

    Only reached when TIER A DECLINED — `parse_overrides` returns `{}` for a sentence that names
    no parameter it can resolve. "this plot looks too clustered" is the case: it is a real request
    and it names no knob, so deciding which one moves needs judgement rather than rules.

    THE CHOICE IS CONSTRAINED TO THIS STEP'S OWN PARAMS, which is the same discipline
    `intent.route_recipe` uses for recipes: the enum is built from what is legal right now, so the
    model cannot name a parameter that does not exist. The reply is then checked against `current`
    anyway — a schema the model ignored is exactly what `agents.call_tool` returns `None` for, and
    a caller that trusted it would be the bug.

    RELATIVE ONLY, deliberately. The model returns a DIRECTION and the arithmetic happens here,
    against the value the step actually holds. Letting it return an absolute number would put a
    magnitude nobody can check into a run that reports `ok` — and since #45, into the decision
    corpus as a value a human chose.
    """
    if not current:
        return {}
    from manyruns import agents

    names = sorted(current)
    got = agents.call_tool(
        system=("You map a scientist's request about a plot to ONE parameter of the step that "
                "produced it, and a direction. You never invent a parameter name and you never "
                "choose a magnitude."),
        prompt=(f"Step: {step or 'the current step'}\n"
                f"Its parameters and current values: {dict(current)}\n"
                f"The scientist said: {text!r}\n"
                "Which single parameter should move, and which way?"),
        name="adjust",
        description="Name one parameter of this step to move, and the direction to move it.",
        schema={"type": "object",
                "properties": {"param": {"type": "string", "enum": names},
                               "direction": {"type": "string", "enum": ["up", "down"]}},
                "required": ["param", "direction"], "additionalProperties": False})
    name = (got or {}).get("param")
    direction = (got or {}).get("direction")
    if name not in current or direction not in ("up", "down"):
        return {}
    base = current[name]
    if not isinstance(base, (int, float)) or isinstance(base, bool):
        return {}
    # ONE STEP, and the step is the caller's convention rather than the model's: +/-1 for a count,
    # +/-10% for a float. A model-chosen magnitude is a number nobody can check.
    delta = 1 if isinstance(base, int) else abs(base) * 0.1
    moved = base + delta if direction == "up" else base - delta
    return {name: type(base)(moved) if isinstance(base, int) else moved}


def parse_overrides(text: str, current: "dict | None" = None) -> dict:
    """`"knn=40 decay=15"` -> `{"knn": 40, "decay": 15}`. Blank input -> `{}` (no change).

    ALSO ACCEPTS ONE SPOKEN PHRASE — "increase the k to 5", "knn to 40", "increase the k by one" —
    which is backlog item 7's simplest case, and it is RULE-BASED: no SDK, no key, no network, so
    it works in the base install and in CI. `intent.py` sets that pattern for recipe selection
    ("three tiers, and only the top one touches an LLM"); this is the same Tier A for parameters.
    The tier that reads "this plot looks too clustered" is a different problem — it names no
    parameter at all — and is deliberately not here.

    `key=value` WINS, and is tried first: it is unambiguous, it is what the prompt advertises,
    and a sentence that happens to contain an `=` should be read the way the user typed it.

    `current` is the step's offered knobs and values. It makes a spoken NAME resolvable ("the k" is
    `knn` here and `n_neighbors` elsewhere) and what makes a RELATIVE phrase possible at all.
    With no current params, only explicit assignments can name a parameter.
    """
    if "=" not in text:
        return _spoken_override(text, current)
    import shlex

    overrides: dict[str, Any] = {}
    try:
        tokens = shlex.split(text)
    except ValueError:
        return {}
    for tok in tokens:
        key, sep, value = tok.partition("=")
        # All of the line must parse: accepting the good half hides a typo in the other half.
        if not sep or not key or not value:
            return {}
        overrides[key] = _coerce(value)
    return overrides


def validated_overrides(text: str, current: dict, initial: Optional[dict] = None) -> dict:
    """A changed override, or a refusal BEFORE any fit or figure rename.

    Both mappings come from `params.tunable_for`: TUNABLE names, declared values over DEFAULTS.
    This is partial validation, not an engine parameter domain. The initial scalar supplies a
    coercion check; strings may also become numbers (`t=auto` → `t=20`). Retain that baseline
    across attempts so `t=auto` can be restored. Missing bounds assert nothing about legal ranges.
    """
    overrides = parse_overrides(text, current)
    example = next(iter(current), "knn")
    hint = f"use an offered parameter like {example}=40 (key=value, no spaces around '=')"
    if not overrides:
        raise ValueError(f"didn't understand {text!r} — {hint}")
    for name, value in overrides.items():
        if name not in current:
            raise ValueError(f"unknown parameter {name!r}; offered: {', '.join(current)} — {hint}")
        base = (current if initial is None else initial)[name]
        valid = True
        if isinstance(base, bool):
            valid = isinstance(value, bool)
        elif isinstance(base, int):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(base, float):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif base is not None and not isinstance(base, str):
            valid = False  # this prompt's scalar parser cannot replace a structured value
        if isinstance(value, float) and not math.isfinite(value):
            valid = False
        if not valid:
            raise ValueError(f"invalid value for {name!r}: {value!r} — {hint}")
    if all(value == current[name] for name, value in overrides.items()):
        raise ValueError(f"params unchanged — {hint} with a different value")
    return overrides


def parse_step_line(line: str, engine: str, recipe: Optional[dict] = None) -> Optional[dict]:
    """`"phate knn=40"` -> the catalog's `phate` step with `knn` overridden to `40`.

    The first token resolves exactly as a bare step name always has
    (`session.resolve_step`, engine-checked); trailing `key=value` tokens layer onto that
    step's own declared `params`, so an unlisted key keeps its recipe/catalog default. `None`
    if the first token isn't a step this engine can dispatch — the caller degrades exactly
    like an unrecognised bare command does today."""
    from manyruns.session import resolve_step

    first, *rest = line.split()
    step = resolve_step(first, engine=engine, recipe=recipe)
    if step is None:
        return None
    from manyruns.tui.params import tunable_for

    # Spoken names and relative phrases use the strip's offered vocabulary and baseline.
    # Keep params as overrides, so simply naming a step does not inject display defaults.
    declared = dict(step.get("params") or {})
    params = {**declared, **parse_overrides(" ".join(rest), dict(tunable_for(step)))}
    return {**step, "params": params}


def _fmt_step(r: dict) -> str:
    if r.get("ok"):
        return f"✓ {r['name']}: {r.get('detail', 'ok')}"
    return f"✗ {r.get('name', '?')}: {r.get('error', 'failed')}"


def _show_plots(rec: dict, write: Any) -> None:
    """Draw whatever this step just produced — inline where the terminal supports it, else
    the path. `figures.draw` already degrades headlessly; the shim below is the two
    attributes it actually reads (`.file`, `.is_terminal`), not a Rich console."""
    import sys

    from manyruns import figures

    paths = (rec.get("record") or {}).get("plots") or []
    if not paths:
        return
    stream = sys.stdout
    shim = type("_StdoutConsole", (), {"file": stream, "is_terminal": stream.isatty()})()
    for p in paths:
        if not figures.draw(shim, p):
            write(f"plot: {p}")


def _corpus_dir(session: Any) -> Any:
    """Where this session's decision rows belong.

    `Session.out_dir` DEFAULTS TO `"."`, and passing that straight through put tune rows at
    `./decisions.jsonl` while `decisions.append`'s own default sends ledger rows to
    `outputs/decisions.jsonl` — two corpora, in two places, for one product. Measured the ugly
    way: 87 rows accumulated in the repository root from test runs before anyone noticed, because
    an untracked file that only grows is invisible until someone reads `git status`.

    So a session that never named an out_dir falls back to `decisions`' default rather than to the
    current directory. A session that DID name one keeps it — that is where its outputs live and
    the corpus belongs beside them.
    """
    out_dir = getattr(session, "out_dir", None)
    return "outputs" if out_dir in (None, "", ".", pathlib.Path(".")) else out_dir


def _label(step: dict) -> str:
    """A tuning attempt's name: the step plus the params that make it this attempt.

    Sorted, so two attempts with the same params in a different order are one label rather than
    two — the corpus counts distinct ATTEMPTS, not distinct spellings.
    """
    params = step.get("params") or {}
    rendered = " ".join(f"{k}={params[k]}" for k in sorted(params))
    return f"{step.get('name')} {rendered}".strip()


def _record_decision(session: Any, attempts: list[dict], chosen: Optional[str]) -> None:
    """Write what was tried and what was kept — manyruns#45.

    THE LOOP EXISTS TO PRODUCE THIS AND WAS THROWING IT AWAY. `decisions.append` has been the
    corpus writer since the ledger got one, and `tune.py` contained no reference to it: every
    retry and its replacement were lost the moment the loop returned. The stated value of tuning
    is precisely "what did not work, and what was chosen when it did".

    ONE ROW PER LOOP, not one per attempt. A decision IS a choice among alternatives, so three
    attempts ending in an accept are one row whose `offered` has three entries and whose `chosen`
    names the third. (An earlier draft of the spec said three rows; that was wrong, and it would
    also have thrown away which attempts competed with which.)

    A CANCEL IS A REAL ROW with `chosen=None` — "none of these worked" is a signal, and dropping
    it would bias the corpus toward loops that happened to end well.

    Not written when only ONE attempt was made: `decisions.append` already refuses an empty offer
    set for the reason that applies here too — a choice with no alternative has no label to learn
    against. Accepting the first thing you tried is not a decision between things.

    `surface="tune"` distinguishes these from ledger rows. `decisions.examples()` currently
    extracts LEDGER rows only — it reads `o["recipe"]` and `o["can_run"]`, which these deliberately
    do not carry, so tune rows are skipped there rather than mis-parsed. Turning them into training
    pairs is separate work; recording them is what stops the data being unrecoverable.
    """
    if len(attempts) < 2:
        return
    from manyruns import decisions

    decisions.append(
        offered=[{"step": a["name"], "params": a["params"], "label": a["label"]}
                 for a in attempts],
        chosen=chosen, surface="tune", run_id=getattr(session, "run_id", None),
        dataset=getattr(session, "dataset", None), shape=getattr(session, "shape", None),
        out_dir=_corpus_dir(session))


def _keep_attempt(rec: dict, attempt: int, write: Any) -> None:
    """Preserve a rejected attempt's figures under an attempt-suffixed name.

    `plots/phate.png` -> `plots/phate@1.png`, so three retries leave three pictures and the one
    the user ACCEPTS keeps the plain name every other surface already expects (`figspec.stem`,
    the run pane, the summary). Renaming the accepted one instead would have made this loop the
    only producer of a filename nothing else knows.

    `@` rather than `-`: `artifacts.save_array` uses `{index:02d}-{step}_{slot}` and a second
    dash-separated convention over the same names invites a reader to parse one as the other.
    A rejected attempt is not a step index — it is a discarded sibling of one.

    MOVES rather than copies: the file is about to be overwritten by the next attempt, so a copy
    would leave the same bytes twice and double the figure directory over a long tuning session.

    THE SIDECARS MOVE WITH IT. `figspec.save` writes `<stem>.spec.npz` and `<stem>.spec.json`
    beside the png and neither is in `record["plots"]`, so renaming only the picture left every
    rejected attempt sharing one coordinate file — the last one written. Measured before the
    fix: three attempts, three pngs, one `.spec.npz`. That cost two things `figspec.py`'s header
    claims for the format ("the durable object is the INPUT"): a rejected attempt could not be
    re-exported, and nothing could say what MOVED between two attempts, because only one of the
    two arrays existed.

    Best-effort by design. A plot that is missing, unreadable, or already moved must not take
    down a tuning session — the run is the artifact, and this is the loop's own bookkeeping.
    The rename is REPORTED so a user can find the file; a silent one would be indistinguishable
    from the overwrite this exists to stop.
    """
    from pathlib import Path

    from manyruns import figspec as _figspec

    def _move(src: Path, dst: Path) -> "str | None":
        """Rename if it is there. Returns an error string, or None on success/absence."""
        try:
            if src.is_file():
                src.rename(dst)
            return None
        except OSError as e:  # noqa: BLE001 - bookkeeping must not kill the loop
            return str(e)

    # `rec` is `session.step`'s REPL-shaped wrapper, not the record — the paths are one level
    # in, which is the same reach `_show_plots` makes just above the call site.
    plots = (rec.get("record") or {}).get("plots") or []
    for i, path in enumerate(plots):
        src = Path(str(path))
        dst = src.with_name(f"{src.stem}@{attempt}{src.suffix}")
        failed = _move(src, dst)
        if failed is not None:
            write(f"  (could not keep attempt {attempt}'s {src.name}: {failed})")
            continue
        if not dst.is_file():
            continue
        # THE COORDINATES, and they are what makes this attempt comparable to the next one.
        # `figspec` writes `<stem>.spec.npz` / `.spec.json` BESIDE the png and neither is in
        # `record["plots"]`, so before this they were overwritten by the following attempt:
        # three pictures, one coordinate set. Moved silently — the png's rename is already
        # reported, and a second line per attempt about a sidecar nobody names would be noise.
        for suffix in (_figspec.ARRAYS, _figspec.META):
            spec_src = _figspec.stem_for(src)
            spec_dst = _figspec.stem_for(dst)
            _move(spec_src.with_name(spec_src.name + suffix),
                  spec_dst.with_name(spec_dst.name + suffix))
        # THE RECORD FOLLOWS THE FILE, and without this the figures pane lies. Each attempt is
        # its own step record and the pane walks `rec["plots"]` (`tui/state.py`'s `StepView`), so
        # a record still naming `phate.png` after the rename points at whatever the NEXT attempt
        # writes there. Measured on a two-attempt loop before this line existed: both records
        # named `plots/phate.png`, so the pane offered two entries over ONE file and drew attempt
        # 2 twice, while attempt 1's real picture sat on disk as `phate@1.png` with nothing
        # referencing it. Flipping between the rejected and the kept picture is the gesture the
        # whole loop exists to support.
        #
        # Mutated IN PLACE rather than rebound: `RunFeed.on_step` takes `list(records)`, a SHALLOW
        # copy, so the record dicts are shared with the runner's lineage and an in-place edit is
        # what reaches the screen. Guarded on `list` because a caller that handed over a tuple
        # should get the rename, not a TypeError.
        if isinstance(plots, list):
            plots[i] = str(dst)
        write(f"  (kept attempt {attempt} as {dst.name})")


def run_tune_loop(session: Any, step: dict, read: Any, write: Any, *,
                  show_plots: Callable[[dict, Any], None] = _show_plots) -> Optional[dict]:
    """Run `step`, show its plot, then loop on `tune` until `accept` or `cancel`.

    Returns the step dict that produced the accepted run, or `None` on cancel. Every retry
    calls `session.step` again — a new lineage entry, never a mutation of the last one.

    `show_plots(rec, write)` owns inline presentation AND its textual fallback. Its return
    value is ignored. A surface with its own record-fed figures can pass a no-op without
    changing plot production, `session.on_step`, or preservation of rejected attempts.

    EVERY ATTEMPT SEES THE SAME INPUT. `session.step` REPLACES `state["emb"]` (real/
    manylatents) or `state["vec"]` (mock) rather than mutating it in place, so holding a
    shallow copy taken before the first attempt is enough to reset it before each later one.
    Without this, a retry's `session.step` call found the PRIOR — not yet accepted, quite
    possibly about to be rejected — attempt's own output already sitting in `state` and
    chained onto it: retrying `phate` with a different `knn` silently re-embedded the
    rejected PHATE output instead of the original preprocessed data, one PHATE stacked on
    another. `g` is reset the same way and for the same reason: every executor writes its
    own keys into it on every call (`_ml_latent` writes `g["phate.<key>"]`), so a rejected
    attempt's numbers would otherwise still be there once a later attempt is accepted. This
    is exactly the `run_recipe`/branch chaining `session.py` documents — DIFFERENT steps in a
    declared sequence are meant to chain on the one before — the reset here only undoes
    chaining onto attempts of THIS SAME step invocation that were never kept.

    The declared metric suite (`manyruns/configs/metrics/default.yaml` — trustworthiness,
    continuity, knn_preservation, ...) is switched off for every step run through this loop.
    `_ml_latent` (`pipeline/runner.py:617`) computes it inline, before the plot is even
    saved, and several of those metrics are pairwise/O(n²) — on a real dataset that can take
    far longer than PHATE's own fit, which is exactly backwards for a loop whose entire
    point is "show me the plot so I can decide." `_finalize` (`session.results`/`close`)
    computes the suite anyway, once, on whatever the session ends on — nothing is lost, it's
    just not paid for on every rejected attempt."""
    from manyruns.tui.params import tunable_for

    current = dict(step)
    initial_params = dict(tunable_for(current))
    attempt = 0
    # Every params combination actually RUN, in order, for the decision row on the way out.
    # Appended after `session.step` rather than before, so an attempt that never executed is not
    # offered to the corpus as one that did.
    attempts: list[dict] = []
    saved_metrics = session.ctx.get("metrics")
    session.ctx["metrics"] = []
    baseline_state = dict(session.state)
    baseline_array = session.ctx.get("array")
    baseline_alignment = session.ctx.get("metadata_aligned", True)
    from manyruns.pipeline import bounds
    from manyruns.pipeline.runner import ComputeCancelled

    # Retain the input's caps alongside its arrays. The current output may be a rejected
    # lower-dimensional embedding and cannot resolve the next tuning question's limits.
    constraints = bounds.resolved_constraints(current, bounds.shape_of(baseline_state, session.ctx))
    baseline_g = dict(session.g)
    # THE THIRD DICT, and it is manyruns#44. `state` and `g` were reset and this was not.
    # `carry["geometry"]` is the run-scoped "last measured values" accumulator that turns each
    # step's geometry into a `[from, to]` delta (`runner.py:235`), so a rejected attempt's
    # numbers stayed in it and the NEXT attempt subtracted against them — the accepted attempt
    # reporting movement from a starting point nobody kept, rendered with an arrow like a real
    # measurement. Copied per key because the values are dicts the runner mutates in place.
    baseline_geometry = dict((session.carry or {}).get("geometry") or {})

    def _reset() -> None:
        session.state.clear()
        session.state.update(baseline_state)
        session.ctx["array"] = baseline_array
        session.ctx["metadata_aligned"] = baseline_alignment
        # The legacy numeric cache is derived; explicit channels retain their original axis.
        from manyruns.pipeline.io import _labels_to_numeric

        session.ctx["color"] = _labels_to_numeric(session.state.get("labels"))
        session.g.clear()
        session.g.update(baseline_g)
        geometry = (session.carry or {}).get("geometry")
        if geometry is not None:
            geometry.clear()
            geometry.update(baseline_geometry)

    try:
        while True:
            _reset()
            rec = session.step(current)
            attempts.append({"name": current.get("name"), "params": dict(current.get("params") or {}),
                             "label": _label(current)})
            show_plots(rec, write)
            write(_fmt_step(rec))
            # Both questions accept changed params. Validate inside this inner loop so a
            # blank, typo, unknown name or unchanged value neither fits nor renames a figure.
            # `tune`/`t` (and the old `retry`/`r`) still open a dedicated params question.
            kind = "decision"
            while True:
                prompt = ("accept / tune / cancel (or key=value) > " if kind == "decision" else
                          f"new params for {current['name']}, key=value (cancel to go back) > ")
                # A bound Gate supplies structured question metadata; CLI input keeps its
                # one-string interface. The loop owns accepted values on both surfaces.
                read_tune = getattr(getattr(read, "__self__", None), "read_tune", None)
                prompt_step = {**current, "constraints": constraints}
                text = (read_tune(prompt, prompt_step, kind) if read_tune else read(prompt)).strip()
                choice = text.lower()
                if kind == "params" and choice in ("cancel", "c"):
                    kind = "decision"
                    continue
                if kind == "decision":
                    if choice in ("accept", "a", "y", "yes"):
                        _record_decision(session, attempts, _label(current))
                        return current
                    if choice in ("cancel", "c"):
                        _reset()
                        # A cancellation still records that none of these attempts was kept.
                        _record_decision(session, attempts, None)
                        write(f"  (cancelled — {current['name']} discarded, back to what it was before)")
                        return None
                    if choice in ("tune", "t", "retry", "r"):
                        kind = "params"
                        continue
                try:
                    overrides = validated_overrides(text, dict(tunable_for(current)), initial_params)
                except ValueError as error:
                    write(f"  ({error})")
                    continue
                break
            # Keep the old picture only once a changed declaration has been accepted. The next
            # outer iteration resets the input BEFORE fitting, preserving attempt lineage and
            # preventing a rejected embedding from becoming its replacement's input.
            attempt += 1
            _keep_attempt(rec, attempt, write)
            current = {**current, "params": {**(current.get("params") or {}), **overrides}}
    except ComputeCancelled:
        if not attempts:
            # Nothing completed: the driver records this occurrence as not run.
            raise
        # The interrupted attempt never committed. Discard the completed alternatives
        # exactly as at the cancel prompt; None lets the driver mark this step discarded.
        _reset()
        _record_decision(session, attempts, None)
        write(f"  (cancelled — {current['name']} discarded, back to what it was before)")
        return None
    finally:
        session.ctx["metrics"] = saved_metrics
