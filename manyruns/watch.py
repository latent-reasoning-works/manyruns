"""`watch(fn)` — record what a tool did, without it having to be a recipe step.

A recipe declares a workflow, and `pipeline.runner._run_steps` records what that workflow
actually executed. That covers the tools manyruns *drives*. It does not cover the ~38
callables in manylatents' registry, each of which would otherwise need a recipe file and an
executor before anything could be said about it — 38 files to learn one fact per tool.

`watch` is the wrapper form of the same record: call anything, get one entry with its
parameters, its outcome, its duration, and a compact description of what came back.

**One record shape, two producers.** The entries here are byte-compatible with the step
record, so `narrate.run_panel` and the panel renderer read them with no idea which produced
them — pinned by a test. A second record format for "tools" would be exactly the drift
`vocab.py` exists to prevent, one layer up.

Parameters are DERIVED, never declared. `inspect.signature` already knows what a tool takes
(measured: 15 params for `diffusion_map`, 11 for `multiscale_phate`, clean across the whole
latent registry), so a hand-written schema per tool would be a second copy that rots on the
seventh tool.

Dependency-free on purpose — no numpy, no rich. A recorder that needs the compute stack
cannot wrap the stackless path.
"""
from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, Iterator, Optional

#: THE outcome vocabulary — the ways a step or a watched call can END. Declared here because
#: this module is the record's home and BOTH producers write into it: `recorder_entry` below,
#: and `pipeline.runner._run_steps`, which already imports `describe` from here.
#:
#:   ok       — the executor returned and claimed nothing more
#:   skipped  — it declined (`_StepSkipped`), or its group has no executor on this engine
#:   error    — it raised; `detail` carries `Type: message`
#:   reported — it ran and returned a status string of its own (the CLI lightning path).
#:              It ran, but it did not report plain "ok".
#:
#: This was declared twice: `runner.STEP_OUTCOMES` was the same 4-tuple, and — measured by
#: grep across manyruns, the learner, manylatents and shop — NEITHER copy had a reader. The
#: live copies were two dict-key restatements, `narrate._OUTCOME_MARKS` and
#: `shell._OUTCOME_STYLE`, checked against neither declaration, so an added outcome could
#: reach one renderer and render as `?` on the other. Both are now pinned to this tuple by
#: `test_every_renderer_covers_the_whole_outcome_vocabulary`.
#:
#: Kept dependency-free with the rest of this module: a renderer must be able to key off the
#: vocabulary with nothing installed.
OUTCOMES = ("ok", "skipped", "error", "reported")

_MAX_REPR = 48


def describe(value: Any) -> Any:
    """A compact, JSON-safe account of a value — the SHAPE of a matrix, not the matrix.

    A record that inlined a 2700×32738 array would be unwritable and unreadable, and one
    that inlined nothing could not tell you a tool returned the wrong thing. Shape, dtype
    and type are what distinguish "it ran" from "it ran on what I meant"."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dtype = getattr(getattr(value, "dtype", None), "name", None)
            return f"{type(value).__name__}{tuple(shape)}" + (f" {dtype}" if dtype else "")
        except Exception:  # noqa: BLE001 - a hostile __shape__ must not break recording
            pass
    if isinstance(value, str):
        return value if len(value) <= _MAX_REPR else value[: _MAX_REPR - 1] + "…"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in list(value.items())[:12]}
    text = f"{type(value).__name__}"
    return text


def bound_params(fn: Callable, args: tuple, kwargs: dict) -> dict:
    """What the tool was actually called with, named — derived from its signature.

    Falls back to positional keys when a signature cannot be read (C extensions, builtins),
    because an unnamed record still beats no record."""
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        items = bound.arguments.items()
    except (TypeError, ValueError):
        items = [(f"arg{i}", v) for i, v in enumerate(args)] + list(kwargs.items())
    return {k: describe(v) for k, v in items if k not in ("self", "cls")}


class Recorder:
    """Collects records. Explicitly passed, never global.

    A module-level list would be shared state, and the sweep runs cells in a process pool
    where shared state is either wrong or invisible depending on the start method."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def __len__(self) -> int:
        return len(self.records)

    def watch(self, fn: Callable, *, name: Optional[str] = None, group: str = "tool") -> Callable:
        return watch(fn, recorder=self, name=name, group=group)

    def as_result(self, *, engine: str = "watch", recipe: Optional[str] = None) -> dict:
        """The records in the shape `narrate.run_panel` and the panel renderer already read,
        so a watched session displays with no renderer changes."""
        return {
            "served_by": "local", "engine": engine, "recipe": recipe,
            "num_steps": len(self.records), "steps": list(self.records),
            "trace": [f"{r['group']}:{r['name']}" for r in self.records],
            "status": {r["name"]: r["outcome"] for r in self.records},
            "g_vector": {}, "plots": [], "seed": None,
            "ok": all(r["outcome"] == "ok" for r in self.records),
        }


def watch(fn: Callable, *, recorder: Recorder, name: Optional[str] = None,
          group: str = "tool") -> Callable:
    """Wrap `fn` so every call appends one record. Return value and exceptions pass through
    untouched — a recorder that swallows an error would be worse than no recorder."""
    label = name or getattr(fn, "__name__", None) or type(fn).__name__

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with recorder_entry(recorder, label, group=group,
                            params=bound_params(fn, args, kwargs)) as rec:
            out = fn(*args, **kwargs)
            # `produced`, not `returned` — the runner and `watch_tool` both use that
            # name, and a second word for one concept is the drift this module exists
            # to prevent. A reader must not have to know which producer wrote a record.
            rec["produced"] = describe(out)
            return out

    return wrapped


def watch_tool(cls: Any, *, recorder: Recorder, name: Optional[str] = None,
               group: str = "tool", method: str = "fit_transform") -> Callable:
    """Wrap an algorithm CLASS — one record per use, carrying its REAL parameter surface.

    `watch(instance.fit_transform)` records the *method's* signature, which for manylatents
    is `(x, y)` — so the tool's actual knobs (`knn`, `decay`, `t`, `n_components`) never
    appear. They live on the constructor. This splits a single call: keyword arguments that
    the constructor accepts configure the tool and are recorded as its params; the rest go
    to the method.

    That split is what makes "parse params for every tool" work without a schema per tool —
    `inspect.signature` already knows, and it is right by construction."""
    label = name or getattr(cls, "__name__", None) or type(cls).__name__
    accepted, defaults = _params_of(cls.__init__)
    method_params, _ = _params_of(getattr(cls, method, None))

    def run(*args: Any, **kwargs: Any) -> Any:
        config = {k: v for k, v in kwargs.items() if k in accepted}
        rest = {k: v for k, v in kwargs.items() if k not in accepted}
        # A keyword the method does not accept is FILTERED, not forwarded — forwarding it
        # raises a TypeError that tells the caller nothing about which of their parameters
        # was wrong. `api.run` drops such kwargs silently; recording them is how a parameter
        # that never took effect becomes visible instead of looking like it did.
        passthrough = {k: v for k, v in rest.items() if k in method_params} if method_params else rest
        ignored = sorted(set(rest) - set(passthrough))

        # defaults included: what a tool ran with is not only what you passed it
        recorded = {**defaults, **config}
        with recorder_entry(recorder, label, group=group,
                            params={k: describe(v) for k, v in recorded.items()}) as rec:
            if ignored:
                rec["ignored"] = ignored
            instance = cls(**config)
            out = getattr(instance, method)(*args, **passthrough)
            rec["produced"] = describe(out)
            return out

    run.__name__ = f"watched_{label}"
    return run


def _params_of(fn: Any) -> tuple[set, dict]:
    """`(accepted names, defaults)` for a callable — the derived parameter surface."""
    if fn is None:
        return set(), {}
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return set(), {}
    names = {n for n in params if n not in ("self", "cls")}
    defaults = {n: p.default for n, p in params.items()
                if n in names and p.default is not inspect.Parameter.empty}
    return names - {"kwargs", "args"}, defaults


class _Entry(dict):
    pass


def recorder_entry(recorder: Recorder, name: str, *, group: str = "tool",
                   params: Optional[dict] = None) -> Any:
    """Context manager yielding one record; sets `outcome` and `seconds` on the way out.

    Same ordering rule as the step loop: the outcome is written AFTER the work returns or
    raises, never before, so the record cannot claim something ran clean because it was
    attempted."""
    from contextlib import contextmanager

    @contextmanager
    def _cm() -> Iterator[dict]:
        rec: dict = _Entry({
            "index": len(recorder.records), "name": name, "group": group,
            "params": dict(params or {}), "outcome": None, "detail": None, "seconds": 0.0,
        })
        recorder.records.append(rec)
        started = time.perf_counter()
        try:
            yield rec
        except Exception as e:  # noqa: BLE001 - recorded, then re-raised unchanged
            rec["outcome"] = "error"
            rec["detail"] = f"{type(e).__name__}: {e}"
            raise
        else:
            if rec["outcome"] is None:
                rec["outcome"] = "ok"
        finally:
            rec["seconds"] = round(time.perf_counter() - started, 6)

    return _cm()
