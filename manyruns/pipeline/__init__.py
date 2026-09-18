"""Real data loading + the CFlows compute pipeline (public deps only).

manyruns stays thin, but the product ask is: *paste a filepath to your data and get the
shape back*. The engine selects datasets by config-group NAME, not by an arbitrary path,
so this package is the direct path — load an array from a file (or a generator `.py`) and
run the recipe with public libraries, no private stack and no `shop`.

Split by concern:
    io       — console suppression + figure writing (the global-state side effects)
    loading  — paths/names/scripts → arrays (+ labels); the future `sources.resolve()` seam
    steps    — in-process step implementations (`_INPROC_STEPS`, `_ANALYSIS_STEPS`)
    mioflow  — the real trained trajectory step, quarantined and named
    runner   — the ONE step loop, state schema, and g-vector assembler both engines share

Every heavy import (numpy, torch, lightning, phate, scanpy, matplotlib, manylatents) stays
function-local, so importing this package is cheap and the stackless/mock path is intact.
"""
from __future__ import annotations

from manyruns.pipeline import io, loading, mioflow, runner, steps
from manyruns.pipeline.io import quiet
from manyruns.pipeline.loading import (
    as_matrix,
    gene_axis,
    labels_of,
    layers_of,
    load_array,
    load_labeled,
    load_named_dataset,
    looks_like_counts,
)
from manyruns.pipeline.runner import (
    GVECTOR_CORE,
    STEP_GROUPS,
    run_manylatents,
    run_inproc,
    step_group,
)

__all__ = [
    "GVECTOR_CORE", "STEP_GROUPS", "as_matrix", "gene_axis", "io", "labels_of", "layers_of", "load_array",
    "load_labeled", "load_named_dataset", "loading", "looks_like_counts", "mioflow", "quiet",
    "run_manylatents", "run_inproc", "runner", "step_group", "steps",
]

# The module-private helpers (`_ANALYSIS_STEPS`, `_save_scatter`, `_detect_time_key`, …) are
# resolved from whichever submodule now owns them, so callers that predate the split keep
# working. Note this returns the LIVE object, so patching a submodule attribute is seen
# through this path too — one patch target (`manyruns.pipeline.<mod>.<name>`) serves every
# caller, rather than the split silently creating two.
_SUBMODULES = (runner, steps, loading, mioflow, io)


def __getattr__(name: str):
    for m in _SUBMODULES:
        if hasattr(m, name):
            return getattr(m, name)
    raise AttributeError(f"module 'manyruns.pipeline' has no attribute {name!r}")
