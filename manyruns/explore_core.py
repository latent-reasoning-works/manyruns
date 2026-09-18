"""Bind a versioned Core (`manyruns.core`) to the exploration loop (`manyruns.session`).

This is the glue between the two halves that converged: the **Core** (a snapshot-versioned
*collection* of `.h5ad` datasets, `core.py`) and **`Session`** (the steppable driver over
`runner.apply_step`, `session.py`). It exposes the two tools the prompt router
will call:

    read_core(core_dir)                    → data/metadata from a Core snapshot (NO compute)
    run_geometry(core_dir, accession, …)   → run the CFlows recipe over one Core member

The human REPL and (later) the LLM router drive the **same** `session.step` seam over a
Core-bound session. Compute stays delegated (manylatents / scanpy / the learner); this module
only wires a Core member's `.h5ad` into the session's data input. Import-clean without the
heavy stack — `session`/`pipeline`/`app` are imported lazily inside `run_geometry`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from manyruns import core


def read_core(core_dir: Path | str) -> dict:
    """The router's ``read_core`` tool: data/metadata from a Core snapshot, no compute."""
    snap = core.load_snapshot(core_dir)
    return {
        "name": snap.name,
        "version": snap.version,
        "members": [
            {
                "accession": m.accession,
                "n_obs": m.n_obs,
                "n_vars": m.n_vars,
                "obs_vocab": m.obs_vocab,
                "tissue": m.tissue,
                "disease": m.disease,
            }
            for m in snap.members
        ],
    }


def _find_member(snap: "core.CoreSnapshot", accession: str) -> "core.CoreMember":
    for m in snap.members:
        if m.accession.lower() == accession.lower():
            return m
    raise KeyError(
        f"{accession!r} not in Core {snap.name}@v{snap.version}; "
        f"have {[m.accession for m in snap.members]}"
    )


def run_geometry(
    core_dir: Path | str,
    accession: str,
    project: str,
    *,
    engine: str = "manylatents",
    out_dir: Path | str = ".",
    recipe: str | None = None,
    array: Any = None,
    labels: Any = None,
    fast_dev_run: bool = False,
) -> dict:
    """The router's ``run_geometry`` tool: run a recipe over one Core member.

    Loads the member's ``.h5ad`` into the session's data input (delegated to
    ``pipeline.load_labeled``) unless an ``array`` is injected or the ``mock`` engine is
    used (which synthesizes its own vector, so no data read / no heavy deps). Returns
    ``session.results()`` — the standard shape that feeds ``app.write_summary``.
    """
    from manyruns.app import load_recipe, select_analysis
    from manyruns.session import Session

    snap = core.load_snapshot(core_dir)
    member = _find_member(snap, accession)

    # Auto-select the recipe from the member's data shape (trajectory vs contrast vs embed)
    # unless the caller forced one — so the router doesn't silently re-impose the trajectory.
    recipe_dict = load_recipe(recipe) if recipe else select_analysis("scrna", Path(member.path))[0]

    label_kind = None
    if array is None and engine != "mock":
        from manyruns import pipeline

        array, labels, label_kind = pipeline.load_labeled(Path(member.path))

    session = Session(
        project=project,
        engine=engine,
        out_dir=Path(out_dir),
        modality="scrna",
        recipe=recipe_dict,
        array=array,
        labels=labels,
        label_kind=label_kind,
        fast_dev_run=fast_dev_run,
    )
    session.run_recipe()
    # `close`, not `results`: this tool runs the recipe to completion and hands back a
    # finished record, so the lineage's state folder gets its `COMPLETE` marker. `results`
    # alone would leave every router-driven run looking interrupted on disk.
    return session.close()
