"""A step's parameters, resolved against the data in hand. **Machinery only — no method knowledge.**

`suite.py` states the rule this file now follows, one directory over: *"The bounds themselves —
and the reason for each — are declared beside the metric names in `configs/metrics/default.yaml`;
this is only the machinery that reads and applies them."* The first version of this module took
that file's grammar and left its principle behind — it carried a table

    CEILINGS = {("pca", "n_components"): (("n_samples", "n_features"), …)}

which is **sklearn's contract for PCA, written down in manyruns**. Three things are wrong with
that, and only the first is about tidiness:

  1. **Two homes for one fact.** sklearn owns the bound and enforces it; manylatents wraps
     sklearn. The older rationale cited a k clamp; the reviewed manylatents b44ee30
     `compute_knn` refuses invalid k instead. A product-side copy drifts when an engine
     changes, silently, in the direction of pruning what would have run.
  2. **It is the harness holding an opinion about a method it does not implement.** `vocab.py`'s
     own comments already flag this hazard twice — keying a table by algorithm name would be "a
     product-side copy of the engine's algorithm catalogue that has to be maintained in step
     with it" — and chose coarser keys to delay it. This module walked straight into it.
  3. **A table in Python is unlearnable.** The planner that is supposed to select and eventually
     PROPOSE compositions lives in the learner (a single-op selector). It can read a recipe;
     it cannot read a dict literal in
     another repo's source. Legality that a learner cannot see is legality it cannot improve.

**So the limit is declared by the recipe, in the recipe.** A step says what it wants and what
caps it:

    - name: pca
      group: latent
      params: {n_components: 10}
      limits: {n_components: [n_samples, n_features]}

and this module does arithmetic: resolve the names against the shape of the matrix the step will
actually see, take the minimum, clamp. It knows nothing about PCA, and a recipe naming an
algorithm nobody here has heard of works the same way. The knowledge sits with the file that
already carries the method's DOI, its `via:`, and the human who chose it.

**This is explicitly partial.** Missing limits mean "manyruns asserts nothing", not "all
values are legal". A resolved cap says only what the recipe's shape cap resolves to: no types,
lower bounds, exclusive endpoints, categoricals or inter-parameter relations are described.
Ordinary step exceptions retain the full engine string; metric-suite notes have a
200-character cap (`suite.py`), so that limit is not a reason to copy engine validation here.

The current recipe clamp is a stopgap. An explicitly adaptive default and an explicit user
override need different policies (adapt / refuse) and engine-side requested/effective reporting.
This module does not implement that cross-repository contract.

"""
from __future__ import annotations

from typing import Any, Optional

#: The identifiers a `limits:` entry may name — the shape of the matrix the step will SEE.
#:
#: A strict subset of `suite.SHAPE_VARS`, and pinned as one by test, so a recipe author reads one
#: vocabulary rather than two that nearly agree. The other two (`n_embedded`, `final_dim`)
#: describe an embedding that exists only AFTER a step has run, so they cannot bound its input.
#:
#: Not imported from `suite` on purpose: this module is in the step loop's hot path and `suite`
#: reaches for numpy and the metric registry. The test is what keeps them one vocabulary.
INPUT_VARS = ("n_samples", "n_features")

#: The floor a resolved limit may not go below. Clamping to zero would turn "too many components"
#: into "no embedding at all", and the step would report `ok` having produced nothing — an
#: outcome the record cannot tell apart from real work.
FLOOR = 1


def check_limits(step: Any) -> list[str]:
    """Problems with one step's `limits:` block; empty means valid (or absent).

    Absent is legal — the field is additive and every recipe written before it stays valid —
    but present-but-wrong is refused, which is the half that stops it from being decorative. A
    typo (`n_feature`) would otherwise resolve to nothing, bound nothing, and read on the page
    as though it had.
    """
    limits = step.get("limits") if hasattr(step, "get") else None
    if limits is None:
        return []
    name = step.get("name", "?")
    if not isinstance(limits, dict):
        return [f"step {name!r}: `limits` must be a mapping of param → cap, got {limits!r}"]

    bad: list[str] = []
    params = step.get("params") or {}
    for param, cap in limits.items():
        names = [cap] if isinstance(cap, str) else cap
        if not isinstance(names, (list, tuple)) or not names:
            bad.append(f"step {name!r}: limits[{param!r}] must name at least one of {INPUT_VARS}")
            continue
        unknown = [n for n in names if n not in INPUT_VARS]
        if unknown:
            bad.append(f"step {name!r}: limits[{param!r}] names {unknown} — must be from "
                       f"{list(INPUT_VARS)}")
        # A cap on a parameter the step never passes is a line that will never fire. Reported
        # rather than ignored: it is nearly always a rename that only got done on one side.
        if param not in params:
            bad.append(f"step {name!r}: limits[{param!r}] caps a param this step does not "
                       f"declare — params are {sorted(params)}")
    return bad


def shape_of(state: dict, ctx: dict) -> Optional[tuple[int, int]]:
    """`(n_samples, n_features)` of the matrix the next step will see, or `None` if unknown.

    The SAME choice `_ml_latent` makes when it builds its kwargs — the embedding if one exists,
    else the input — because a limit resolved against a different matrix from the one the step
    runs on is worse than no limit at all.

    Verified generator declarations are the last resort when no actual array is available.
    """
    for candidate in (state.get("emb"), ctx.get("array"), state.get("X")):
        shape = getattr(candidate, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return declared_shape(ctx.get("declared_shape"))


def declared_shape(value: Any) -> Optional[tuple[int, int]]:
    """Validate a caller's optional fit-set declaration before compute."""
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in value)):
        raise ValueError("declared_shape must contain two positive integers (not bool)")
    return tuple(value)


def resolved_constraints(step: Any, shape: Optional[tuple[int, int]]) -> dict:
    """Serializable recipe caps for each declared param, including explicit unknowns.

    `resolved` describes ONLY this cap, never a complete parameter domain. Unknown is not
    permission: a missing declaration and an unavailable input shape both assert nothing.
    Shared with `fit` so a record or ask consumer never has to recover a number from prose.
    """
    limits = step.get("limits") or {}
    env = dict(zip(INPUT_VARS, shape)) if shape is not None else {}
    constraints = {}
    for param in step.get("params") or {}:
        cap = limits.get(param) if isinstance(limits, dict) else None
        names = [cap] if isinstance(cap, str) else cap
        if not names:
            constraints[param] = {"status": "unknown", "cap": None, "reason": "no_limit"}
        elif not env or any(n not in env for n in names):
            constraints[param] = {"status": "unknown", "cap": None, "reason": "unresolved_shape"}
        else:
            constraints[param] = {"status": "resolved", "cap": max(FLOOR, min(env[n] for n in names))}
    return constraints


def fit(step: Any, params: dict, shape: Optional[tuple[int, int]]) -> tuple[dict, list[str]]:
    """This step's params with its own declared limits applied, and one sentence per change.

    Returns a NEW dict — the recipe's declaration is never rewritten, so `recipe["steps"]` still
    says what was ASKED for however many times it is run, and a second run on wider data does not
    inherit the first run's clamp.

    Partial only: no `limits:`, no shape, or a limit naming nothing this run carries → the
    params come back untouched. There is no fallback table to fall back to.
    """
    out = dict(params or {})
    limits = step.get("limits") if hasattr(step, "get") else None
    if not shape or not isinstance(limits, dict):
        return out, []

    constraints = resolved_constraints(step, shape)
    name = (step.get("name") if hasattr(step, "get") else None) or "step"
    notes: list[str] = []
    for param, cap in sorted(limits.items()):
        value = out.get(param)
        if not isinstance(value, int) or isinstance(value, bool):
            # `isinstance(True, int)` is True in Python: a flag clamped to 1 reads as set
            # whatever it was, so booleans are left alone rather than bounded.
            continue
        names = [cap] if isinstance(cap, str) else list(cap or ())
        ceiling = constraints.get(param, {}).get("cap")
        if ceiling is None:
            continue
        if value <= ceiling:
            continue
        out[param] = ceiling
        notes.append(f"{name}: {param} {value} → {ceiling} — the recipe caps it at "
                     f"{' / '.join(names)}, and this data is {shape[0]} × {shape[1]}")
    return out, notes
