"""The commercial flag: one setting that keeps every non-commercial step out of the process.

manyruns is Apache-2.0. Seven modules of manylatents carry the Yale non-commercial licence
(`NON_COMMERCIAL_MODULES`), and manylatents ships their notices. Someone who builds a
proprietary application on the free components sets `MANYRUNS_COMMERCIAL=1` in the process
that runs manyruns. Then:

- every recipe with a non-commercial step is unavailable (`narrate.unavailable`): a hollow row
  with `REASON`, never recommended, and refused by `manyruns run`;
- `runner.apply_step`, `registry.apply_weights` and `toolchain.check_tool` refuse before any
  import, so manyruns imports none of the seven modules.

The flag covers manyruns' own imports only. A host that imports one of those modules itself
still has it in `sys.modules`. Without the flag everything runs.

This is the ONLY module that reads the flag or spells its name. Every other module uses
`enabled()`, `REASON` or `TOOL_PROBLEM`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from manyruns import env as _env

_NAME = "MANYRUNS_COMMERCIAL"

#: The seven manylatents modules under the Yale non-commercial licence, sorted.
NON_COMMERCIAL_MODULES: tuple[str, ...] = tuple(sorted((
    "manylatents.algorithms.lightning.mioflow",
    "manylatents.algorithms.lightning.networks.mioflow_net",
    "manylatents.algorithms.lightning.networks.gaga_net",
    "manylatents.algorithms.lightning.cflows",
    "manylatents.algorithms.cflows_granger",
    "manylatents.algorithms.lightning.losses.cflows",
    "manylatents.algorithms.lightning.networks.aanet",
)))

#: Step names that manyruns routes to a Yale-derived module, on any engine. Closed: a test
#: reads every lightning config of the installed manylatents and requires the stem of each one
#: that names a Yale-derived module to be here.
#:
#:   mioflow               the in-process route (`pipeline/mioflow.py`): mioflow, mioflow_net
#:   mioflow_gaga          the lightning CLI route: mioflow, mioflow_net, gaga_net
#:   cflows                the lightning CLI route: lightning.cflows, losses.cflows
#:   aanet_reconstruction  the lightning CLI route: networks.aanet
#:   granger               a stub today; its estimator is cflows_granger
#:
#: The in-process stand-in for `mioflow` is not Yale code, but legality is decided at plan
#: time and does not know the engine, so the name is non-commercial on every engine.
NON_COMMERCIAL_STEPS: tuple[str, ...] = (
    "aanet_reconstruction", "cflows", "granger", "mioflow", "mioflow_gaga")

#: The one reason every refusal gives.
REASON = (f"this recipe has a step under a non-commercial licence, and {_NAME}=1 is set, "
          f"so it does not run in this process")

#: The problem `toolchain.check_tool` reports for a `commercial_use: false` manifest.
TOOL_PROBLEM = f"commercial_use is false and {_NAME}=1 is set"


def enabled() -> bool:
    """Whether commercial mode is on. Read at each call, so a test can set the flag.

    `1` is on. Unset, empty and `0` are off. Any other value raises `ValueError`: a value such
    as `true` that turned the flag off would let non-commercial code run in a process that
    asked for none."""
    raw = _env.get("COMMERCIAL")
    if raw is None or raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{_NAME}={raw!r} is not an accepted value: set {_NAME}=1 to turn "
                     f"commercial mode on, or leave it unset, empty or 0 to turn it off")


def _tool_manifest(tool: str) -> "dict | None":
    """The tool's manifest as plain data, read with `yaml.safe_load`.

    Not `toolchain.load_tool`: that refuses a `commercial_use: false` manifest in commercial
    mode, which is the very manifest this needs to read."""
    import yaml

    from manyruns import toolchain

    path = Path(toolchain.tool_dir()) / f"{tool}.yaml"
    if not path.is_file():
        return None
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return doc if isinstance(doc, dict) else None


def is_non_commercial(step: Any) -> bool:
    """True for a step whose name is in `NON_COMMERCIAL_STEPS`, or whose `tool:` manifest says
    `commercial_use: false`."""
    if not isinstance(step, dict):
        return False
    if step.get("name") in NON_COMMERCIAL_STEPS:
        return True
    tool = step.get("tool")
    if not tool:
        return False
    manifest = _tool_manifest(str(tool))
    return manifest is not None and manifest.get("commercial_use") is False


def refusal(recipe: Any) -> list[str]:
    """`[REASON]` in commercial mode when any step of `recipe` is non-commercial; else `[]`."""
    if not enabled():
        return []
    steps = (recipe.get("steps") or []) if isinstance(recipe, dict) else []
    return [REASON] if any(is_non_commercial(s) for s in steps) else []
