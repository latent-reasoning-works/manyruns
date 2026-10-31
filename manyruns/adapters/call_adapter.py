"""The call adapter: one call to an unmodified library, described by a manifest's `call` block.

A library that takes an array and returns arrays in one call needs no adapter of its own. Its
manifest says `mode: call` and names the target, the constant options, how the request's slots
bind to arguments, and which selector gives each output. This adapter makes that call INSIDE
the tool's interpreter and writes a version 1 result.

It runs as `python call_adapter.py request.json`. At module level it imports only the standard
library and numpy. It imports the target by name inside `main`, so importing this module never
imports the library. Like every adapter here, it never imports manyruns.

`main` does eight things, in order:

  1. load the slots: `matrix` from `input`, `time` from `time` when present, `seed`
  2. decline when `args` or `kwargs` uses a slot the request does not carry
  3. merge the step's params over `options`; decline a key outside the block's `params`
  4. import `<module>` and resolve `<qualname>` attribute by attribute
  5. with `method`: build the instance with the options, then call the method with the bound
     arguments; without it: call the target with the bound arguments and the options
  6. apply each selector: `return`, `return[<int>]`, `return.<attr>`, `return["<key>"]`,
     `instance.<attr>`
  7. convert each value with `numpy.asarray`: `embedding`, `pseudotime` and `trajectories` to
     float64, `clusters` to int64 only when every value is integral
  8. write each array to `<fact>.npy` beside `result.json`, with its sha256

An exception raised by the target becomes `ok: false`, with a reason that starts with the
exception's type name. `KeyboardInterrupt` and `SystemExit` are not caught.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

#: The selector grammar; `toolchain.SELECTOR_PATTERN` with named groups.
_SELECTOR = (r'^(?:return(?:\[(?P<index>-?\d+)\]|\.(?P<attr>[A-Za-z_]\w*)|\["(?P<key>[^"]+)"\])?'
             r'|instance\.(?P<inst>[A-Za-z_]\w*))$')

_FLOAT_FACTS = ("embedding", "pseudotime", "trajectories")


def _write(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2))
    return 0


def _decline(req, reason):
    return _write(req["result"], {"schema_version": 1, "ok": False, "reason": reason})


def _resolve(target):
    module_name, qualname = target.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _select(selector, returned, instance):
    import re

    m = re.match(_SELECTOR, selector)
    if m is None:
        raise ValueError(f"{selector!r} is not a selector")
    if m.group("inst"):
        return getattr(instance, m.group("inst"))
    if m.group("index") is not None:
        return returned[int(m.group("index"))]
    if m.group("attr"):
        return getattr(returned, m.group("attr"))
    if m.group("key") is not None:
        return returned[m.group("key")]
    return returned


def _convert(fact, value):
    """`(array, None)` or `(None, reason)`."""
    try:
        array = np.asarray(value)
        if fact in _FLOAT_FACTS:
            return array.astype("float64"), None
        floats = array.astype("float64")
    except (TypeError, ValueError) as e:
        return None, f"{fact}: the selected value is not numeric ({type(e).__name__})"
    if np.issubdtype(array.dtype, np.integer):
        return array.astype("int64"), None
    if not (np.isfinite(floats).all() and (floats == np.round(floats)).all()):
        return None, f"{fact}: the selected values are not all integers, so they are not a " \
                     f"partition"
    return floats.astype("int64"), None


def main(request_path):
    req = json.loads(Path(request_path).read_text())
    call = req.get("call") or {}

    # 1. The slots this request carries.
    slots = {"matrix": np.load(req["input"], allow_pickle=False),
             "seed": int(req.get("seed") or 0)}
    if req.get("time"):
        slots["time"] = np.load(req["time"], allow_pickle=False)

    # 2. A slot the call binds and the request lacks.
    args = list(call.get("args") or [])
    kwargs = dict(call.get("kwargs") or {})
    for slot in [*args, *kwargs.values()]:
        if slot not in slots:
            return _decline(req, f"the call binds the slot {slot!r}, and this run does not "
                                 f"carry it")

    # 3. The step's params over the constant options, only for the keys the block allows.
    allowed = list(call.get("params") or [])
    params = dict(req.get("params") or {})
    outside = sorted(set(params) - set(allowed))
    if outside:
        return _decline(req, f"parameter(s) {outside} are not in the call block's params "
                             f"{allowed}")
    options = {**(call.get("options") or {}), **params}

    bound_args = [slots[s] for s in args]
    bound_kwargs = {name: slots[s] for name, s in kwargs.items()}
    try:
        # 4. The target, imported here and only here.
        target = _resolve(call["target"])
        # 5. One call.
        instance = None
        if call.get("method"):
            instance = target(**options)
            returned = getattr(instance, call["method"])(*bound_args, **bound_kwargs)
        else:
            returned = target(*bound_args, **bound_kwargs, **options)
        # 6. One selector per output.
        selected = {fact: _select(sel, returned, instance)
                    for fact, sel in (call.get("outputs") or {}).items()}
    except Exception as e:  # noqa: BLE001 - the library's failure is the step's reason
        return _decline(req, f"{type(e).__name__}: {e}")

    # 7 and 8. Convert, write each array with its sha256, then the result.
    base = Path(req["result"]).parent
    facts = {}
    for fact, value in selected.items():
        array, problem = _convert(fact, value)
        if problem:
            return _decline(req, problem)
        path = base / f"{fact}.npy"
        np.save(path, array)
        facts[fact] = {"path": path.name,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return _write(req["result"], {"schema_version": 1, "ok": True,
                                  "n_cells": int(slots["matrix"].shape[0]), "facts": facts})


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
