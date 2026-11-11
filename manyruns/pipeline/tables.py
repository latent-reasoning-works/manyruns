"""Validate and publish run tables for external tools and in-process steps."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from manyruns import artifacts
from manyruns.pipeline.steps import _StepSkipped

if TYPE_CHECKING:
    import pandas


def publish(
    step: str,
    tables: dict[str, pandas.DataFrame],
    *,
    required: dict[str, Sequence[str]],
    state: dict,
    g: dict,
    ctx: dict,
    reserved: Iterable[str] = (),
) -> dict[str, str]:
    """Check, save and hash tables, then record their paths and measurements.

    Return only the paths published by this call. A refusal leaves state and g
    unchanged; files already saved before an I/O failure can remain on disk.
    """
    from manykinds import Table

    artifact_names = set(artifacts.PERSISTED) | set(state.get("extras") or {}) | set(reserved)
    for name in tables:
        if name in artifact_names:
            raise _StepSkipped(
                f"{step}: table {name} has an artifact name conflict; rename the table"
            )
    for name, df in tables.items():
        try:
            Table(df, provenance=(name,)).require(*required[name])
        except ValueError as exc:
            raise _StepSkipped(f"{step}: table {name} missing columns: {exc}") from exc
    if tables and (not ctx or ctx.get("out_dir") is None or "step_index" not in ctx):
        raise _StepSkipped(
            f"{step}: supply out_dir and step_index in the step context to save tables"
        )
    saved = {}
    for name, df in tables.items():
        try:
            saved[name] = artifacts.save_table(
                ctx["out_dir"], ctx.get("run_id"), ctx["step_index"], step, name, df
            )
        except OSError as exc:
            raise _StepSkipped(
                f"{step}: cannot save table {name}; fix the output directory: {exc}"
            ) from exc
    # Hash the saved bytes: serialization can change the tool's original bytes.
    # Finish every hash before writing any measurements or state.
    digests: dict[str, str] = {}
    for name, path in saved.items():
        try:
            digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise _StepSkipped(
                f"{step}: cannot read saved table {name} back to hash it: {exc}"
            ) from exc
    paths = {name: str(path) for name, path in saved.items()}
    if saved:
        # Replace the dict so a shallow snapshot used for cancellation stays intact.
        tables_state = dict(state.get("tables") or {})
        tables_state.update(paths)
        for name in saved:
            g[f"{step}.{name}.rows"] = len(tables[name])
            g[f"{step}.{name}.columns"] = sorted(tables[name].columns)
            g[f"{step}.{name}.sha256"] = digests[name]
        state["tables"] = tables_state
    return paths
