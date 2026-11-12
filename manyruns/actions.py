"""`manyruns actions`: the legal action space for one dataset, as one JSON document.

THE ONE ACTION-ENUMERATION INTERFACE. Any program that decides what manyruns runs next reads
this document instead of parsing `manyruns check` or re-deriving legality from the YAMLs. It
does not know who calls it and takes no argument that says.

Three surfaces:

- `ledger`: `tui.state.ledger` serialized. Legality, the recommendation, membership and order
  all come from there, and each `offered` entry is exactly `LedgerRow.as_offer()`, the
  conversion a decision row records. Nothing here recomputes them.
- `tune`: the declared params of each tunable step, their one-press neighbours, floor, cap and
  domain. The `left` and `right` fields are nullable: unchanged sides are null, and a duplicate
  left side is null. No candidate grid: legality is free inside the domain.
- `ladder`: the scorer's rung tables.

**Declared, never sniffed.** The Observation is `narrate.declared_observation`, built from the
YAML alone, so the document is the same on a machine that holds the data file and on one that
does not. Nothing here opens a data file, writes anything, builds a `Session` or runs a step.
With the gate on, the run record is read through `store.read`, and only to find the probe run
(`probe_run`): the first `ok` run of the dataset that was not cancelled and had no step in
error, at the requested seed if there is one, else at any seed. Its g-vector is the gate's
evidence (`vocab.rung_support`); the document names the probe run in `evidence` (its `run_id`,
`recipe` and `seed`, plus `requested_seed` when the two seeds differ) and copies no measured
value.

**Plain JSON only.** Lists, never tuples, so the document a caller gets in process hashes
equal to the one it parses from stdout.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

from manyruns import catalog, narrate, vocab
# `store` is the name of a keyword argument below, so the module goes by another name.
from manyruns import store as records

SCHEMA_VERSION = 1

#: The views of the action space. `ledger` is the default.
SURFACES = ("ledger", "tune", "ladder", "derive")


class UsageError(ValueError):
    """A request the verb does not accept: a bad name, or a flag on the wrong surface."""


class LoadError(Exception):
    """A discovered name whose file does not load. `kind` is "dataset" or "recipe"."""

    def __init__(self, kind: str, name: str, message: str):
        # One line: a YAML parser error spans several, and stderr gets exactly one.
        super().__init__(" ".join(f"{kind} {name!r}: {message}".split()))
        self.kind = kind
        self.name = name


def is_probe(row: dict, dataset: str) -> bool:
    """True when `row` may serve as the gate's evidence for `dataset`.

    The rule (#37): the row is of `dataset`, `ok is True`, `cancelled` is not True, `not_run` is
    empty, and no step record has the outcome `"error"`.

    - A derived row is never gate evidence: its recipe starts with `derive:`.
    - A step that SKIPPED does not disqualify the row. `normalize` and `transform` decline a file
      whose values are already log-normalised; the run is `ok` and not `complete`, and its
      g-vector still holds the measurements of its embedding. `complete` stays the run's own
      account of whether every declared step ran, and this rule does not read it.
    - A step in `"error"` disqualifies the row, even when `ok` says otherwise: the probe is the
      run that nothing went wrong in.
    - A cancelled run disqualifies it: a cancel, a discarded tune attempt or an abandoned run
      leaves geometry that is not the geometry of the run that was asked for.
    - `not_run` disqualifies it: a declared step that was never issued means the run stopped
      before its recipe ended, so its g-vector is the geometry of a partial state, not a run
      that ran and declined. Today a non-empty `not_run` comes only with an abandoned run, so
      this check is the same as the `cancelled` check, and it holds for another writer too.
    - A compute error disqualifies it through `ok`: `runner._finalize` sets `ok` to False when
      the final geometry could not be measured, and the run row has no `compute_error` field.

    A row written before `cancelled` and `not_run` reached the run row has neither key and reads
    as not cancelled. The gate still refuses each measurement that is missing from the probe's
    g-vector (`vocab.rung_support`, `not_measured`), so a row that passes this rule never lets a
    claim through without its number.
    """
    if row.get("kind") in ("profile", "derived") or str(row.get("recipe") or "").startswith("derive:"):
        return False
    if row.get("dataset_name") != dataset or row.get("ok") is not True:
        return False
    if row.get("cancelled") is True or row.get("not_run"):
        return False
    return not any(isinstance(s, dict) and s.get("outcome") == "error"
                   for s in row.get("steps") or [])


def probe_run(dataset: str, seed: int, store: Path | str = ".") -> Optional[dict]:
    """The probe run of `dataset` for a request at `seed` in `<store>/outputs`, or None.

    A probe is a row that `is_probe` accepts. The first probe at `seed` comes first. With no
    probe at `seed`, the first probe of `dataset` at any seed stands in, and its own `seed`
    differs from the requested one; the action document then names both (`evidence.seed` and
    `evidence.requested_seed`).

    FIRST, in file order: the record is append-only, so the first matching row never changes and
    every document of one episode names the same evidence. A later run at the requested seed
    replaces a stand-in from another seed, which makes a choice made on the older document
    stale. The rule does not look at the recipe: whoever made the run, it is the probe run.
    """
    fallback = None
    for row in records.read(Path(store) / "outputs"):
        if not is_probe(row, dataset):
            continue
        if row.get("seed") == seed:
            return row
        if fallback is None:
            fallback = row
    return fallback


def enumerate_actions(surface: str, *, dataset: Optional[str] = None,
                      recipe: Optional[str] = None, step: Optional[str] = None,
                      dataset_dir: Optional[Path | str] = None, gate: Optional[bool] = None,
                      seed: int = 42, store: Path | str = ".", run: Optional[str] = None) -> dict:
    """The document for one surface. Raises `UsageError` or `LoadError`; never prints."""
    from manyruns.app import _installed_build

    if surface not in SURFACES:
        raise UsageError(f"unknown surface {surface!r}; known: {', '.join(SURFACES)}")
    if surface != "derive" and run is not None:
        raise UsageError("--run is allowed only on derive")
    if surface == "derive":
        if not run:
            raise UsageError("the derive surface needs --run")
        if recipe is not None or step is not None or gate is not None:
            raise UsageError("the derive surface takes no --recipe, --step or --gate")
    else:
        gate = True if gate is None else gate
    doc: dict = {"schema_version": SCHEMA_VERSION, "tool_version": _installed_build(),
                 "surface": surface, "vocab_version": vocab.VERSION}
    if surface == "ladder":
        given = [flag for flag, value in (("DATASET", dataset), ("--recipe", recipe),
                                          ("--step", step)) if value is not None]
        if given:
            raise UsageError(f"the ladder surface takes no {', '.join(given)}")
        return {**doc, **_ladder()}

    if dataset is None:
        raise UsageError(f"the {surface} surface needs a DATASET")
    known = catalog.discover_datasets(dataset_dir)
    if dataset not in known and surface != "derive":
        raise UsageError(f"unknown dataset {dataset!r}; known: {', '.join(known) or '(none)'}")
    if surface == "ledger" and (recipe is not None or step is not None):
        raise UsageError("the ledger surface takes no --recipe or --step")
    if surface == "tune":
        if recipe is None:
            raise UsageError("the tune surface needs --recipe")
        recipes = catalog.discover_recipes()
        if recipe not in recipes:
            raise UsageError(f"unknown recipe {recipe!r}; known: {', '.join(recipes) or '(none)'}")
    try:
        if surface == "derive":
            from manyruns import derive

            ds = derive.dataset_description(derive.root_row(derive.parent_row(run, store), store), dataset_dir)
        else:
            ds = catalog.load_dataset(dataset, dataset_dir)
    except UsageError:
        raise
    except Exception as exc:  # noqa: BLE001 - a malformed YAML raises a parser error, not ValueError
        raise LoadError("dataset", dataset, str(exc)) from exc
    obs = narrate.declared_observation(ds, dataset)
    doc["dataset"] = {
        "name": dataset,
        "shape": obs.shape,
        "topology": list(ds.get("topology") or []),
        "modality": obs.modality,
        "handle_kind": ds["handle"]["kind"],
        "provides": sorted(obs.provides()),
        "dims": ds.get("dims"),
    }
    if surface == "derive":
        from manyruns import derive

        return {**doc, **derive.offer(dataset, run, store, dataset_dir=dataset_dir)}
    doc["gate"] = bool(gate)
    probe = probe_run(dataset, seed, store) if gate else None
    doc["evidence"] = _evidence(probe, seed)
    if surface == "ledger":
        if not gate:
            return {**doc, **_ledger(obs, None, ())}
        # The gate is the geometry mask. Evidence is the probe run's g-vector, or `{}` when
        # there is none: absence refuses every ruled claim, it never passes. The admitted set
        # is the declared suite (the audit's verified-safe set, written down) plus the
        # readouts admitted by hand. Neither copies a measured value into the document.
        evidence = dict((probe or {}).get("g_vector") or {})
        admitted = set(catalog.load_suite()) | set(vocab.ADMITTED_READOUTS)
        return {**doc, **_ledger(obs, evidence, admitted)}
    return {**doc, **_tune(recipe, step)}


def _evidence(probe: Optional[dict], seed: int) -> Optional[dict]:
    """The document's account of the probe run: its `run_id`, `recipe` and `seed`.

    `requested_seed` is added only when the probe came from another seed than the request. A
    document whose probe is at the requested seed therefore keeps the keys, and the digest, it
    had before the fallback existed.
    """
    if probe is None:
        return None
    named = {"run_id": probe.get("run_id"), "recipe": probe.get("recipe"),
             "seed": probe.get("seed")}
    if named["seed"] != seed:
        named["requested_seed"] = seed
    return named


def _ledger(obs, evidence: Optional[dict], admitted) -> dict:
    from manyruns.tui import state

    provided = obs.provides()
    rows = state.ledger(obs, provided, measured=None, evidence=evidence, admitted=admitted)
    rank = {o["recipe"]: i for i, o in
            enumerate(narrate.offer(obs, available=catalog.discover_recipes(),
                                    evidence=evidence, admitted=admitted))}
    detail = {}
    for row in rows:
        # The ledger already loaded every row's recipe and skipped any that does not load.
        recipe = catalog.load_recipe(row.recipe)
        detail[row.recipe] = {
            "id": recipe.get("id"),
            "kind": recipe.get("kind", "recipe"),
            "steps": [f"{s['group']}:{s['name']}" for s in recipe["steps"]],
            "claims": recipe.get("claims"),
            "suits": list(recipe.get("suits") or []),
            # The same call `narrate.blocked` makes for the row, so `unmet == []` is exactly
            # the data half of `can_run`.
            "unmet": sorted(vocab.unmet(recipe, obs.shape, provided)),
            "invalidated": sorted(vocab.invalidated(recipe, obs.shape, provided)),
            "unavailable": list(narrate.unavailable(recipe)),
            # The rung-support reasons the gate refuses a recipe for: fixed codes from
            # `vocab.SUPPORT_REASONS`, the same ones the row's `blocked` list ends with. `[]`
            # with the gate off.
            "support": ([] if evidence is None else
                        sorted(vocab.rung_support(recipe, evidence, admitted))),
            "offer_rank": rank.get(row.recipe),
        }
    return {"offered": [r.as_offer() for r in rows], "detail": detail}


def _tune(name: str, only: Optional[str]) -> dict:
    from manyruns import tune
    from manyruns.pipeline import bounds
    from manyruns.tui import params

    try:
        recipe = catalog.load_recipe(name)
    except Exception as exc:  # noqa: BLE001 - a malformed YAML raises a parser error, not ValueError
        raise LoadError("recipe", name, str(exc)) from exc
    tunable = [] if recipe.get("kind") == "profile" else [(i, s) for i, s in enumerate(recipe["steps"]) if params.tunable_for(s)]
    seen: set = set()
    for _, s in tunable:
        if s["name"] in seen:
            raise LoadError("recipe", name, f"two tunable steps are named {s['name']!r}")
        seen.add(s["name"])
    if only is not None:
        tunable = [(i, s) for i, s in tunable if s["name"] == only]
        if not tunable:
            raise UsageError(f"recipe {name!r} has no tunable step {only!r}")
    offered, knobs = [], {}
    for order, (index, s) in enumerate(tunable):
        declared = dict(s.get("params") or {})
        offered.append({"step": s["name"], "params": declared, "label": tune._label(s)})
        caps = bounds.resolved_constraints(s, None)
        knob_params = {}
        for k, v in params.tunable_for(s):
            left = params.step_value(k, v, -1)
            right = params.step_value(k, v, +1)
            knob_params[k] = {
                "current": v,
                "source": "declared" if k in declared else "default",
                "floor": params.FLOOR.get(k),
                "cap": caps[k]["cap"] if k in caps else None,
                "left": left if left != v and left != right else None,
                "right": right if right != v else None,
                "domain": {**params.DOMAIN[k], "also": list(params.DOMAIN[k]["also"])},
            }
        knobs[s["name"]] = {
            "index": index,
            "group": s["group"],
            "order": order,
            "params": knob_params,
        }
    return {"recipe": name, "step": only, "offered": offered, "knobs": knobs}


def _ladder() -> dict:
    from manyruns import score

    weight = inspect.signature(score.ladder_cost).parameters["overshoot_weight"].default
    return {
        "rung": dict(score.RUNG),
        "incomparable": sorted(sorted(pair) for pair in score.INCOMPARABLE),
        "overshoot_weight": weight,
        "claimed": score.claimed_rungs(),
        "unroutable": score.unroutable_rungs(),
    }
