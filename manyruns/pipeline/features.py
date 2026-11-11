"""Publish descriptive features of the original run input."""

import hashlib
import json
from pathlib import Path

from manyruns import installation
from manyruns.pipeline.steps import _StepSkipped, _StepUnsupported

TABLE = "dataset_features"
COLUMNS = ("key", "value", "type", "status", "reason", "source", "definition_version",
           "input_sha256", "tested_matrix", "roles")
ROW_FIELDS = COLUMNS[:7]
STATUSES = ("ok", "not_supplied", "not_in_input", "not_applicable", "undefined")
_ROLES = ("sample", "condition", "batch", "time", "labels")
_PARAMS = ("roles", "layer", "overrides")


def _helper():
    """Load the optional helper only when the step runs."""
    try:
        from manylatents.singlecell.dataset_features import dataset_features
    except ImportError as exc:
        raise _StepUnsupported(
            "dataset_features needs manylatents-omics with singlecell.dataset_features; "
            + installation.RUNTIME_REPAIR
        ) from exc
    return dataset_features


def _file_sha256(path) -> str:
    """Hash a file with at most 1 MiB in each read."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_rows(rows):
    """Refuse malformed helper output before any table is published."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("the helper must return a nonempty list of rows")
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(ROW_FIELDS):
            raise ValueError(f"helper rows must have exactly these fields: {', '.join(ROW_FIELDS)}")
        if row["status"] not in STATUSES:
            raise ValueError(f"helper row status must be one of {', '.join(STATUSES)}")
        version = row["definition_version"]
        if not isinstance(version, str) or not version.startswith("dataset_features/"):
            raise ValueError("helper definition_version must start with dataset_features/")
        if type(row["value"]) not in (bool, int, float, str, type(None)):
            raise ValueError("helper values must be native Python bool, int, float, str, or None")
        json.dumps(row["value"], allow_nan=False)
    tested = [r for r in rows if r["key"] == "tested_matrix"]
    if len(tested) != 1 or tested[0]["status"] != "ok" or not isinstance(tested[0]["value"], str):
        raise ValueError("the helper must return one tested_matrix row with an ok string value")
    return tested[0]["value"]


def step(state, g, params, ctx) -> None:
    """Resolve explicit roles, call the helper, and publish the run table."""
    from manyruns.pipeline import tables

    unknown = set(params) - set(_PARAMS)
    if unknown:
        raise _StepSkipped(f"dataset_features: allowed params are {', '.join(_PARAMS)}")
    supplied = params.get("roles", {})
    if not isinstance(supplied, dict) or set(supplied) - set(_ROLES):
        raise _StepSkipped(f"dataset_features: roles must be a dict with keys {', '.join(_ROLES)}")
    for role, column in supplied.items():
        if column is not None and not isinstance(column, str):
            raise _StepSkipped(f"dataset_features: role {role} must be a column name, 'absent', or None")
    layer, overrides = params.get("layer"), params.get("overrides")
    if layer is not None and not isinstance(layer, str):
        raise _StepSkipped("dataset_features: layer must be a string or None")
    if overrides is not None and not isinstance(overrides, dict):
        raise _StepSkipped("dataset_features: overrides must be a dict or None")
    original = ctx.get("original_input")
    if original is None or original.get("object") is None:
        raise _StepSkipped(
            "dataset_features reads the input the run loaded; this run has none "
            "(a named dataset is loaded inside the engine)"
        )
    defaults = original.get("role_keys") or {}
    roles = {role: supplied.get(role, defaults.get(role) if role != "labels" else None)
             for role in _ROLES}
    try:
        rows = _helper()(original["object"], roles, layer=layer, overrides=overrides)
        tested_matrix = _check_rows(rows)
    except (ValueError, TypeError) as exc:
        raise _StepSkipped(f"dataset_features: {exc}") from exc
    source = ctx.get("source_path")
    try:
        input_sha256 = _file_sha256(source) if source is not None and Path(source).is_file() else ""
    except OSError as exc:
        raise _StepSkipped(f"dataset_features: cannot hash the input file: {exc}") from exc
    import pandas as pd

    encoded_roles = json.dumps(roles, sort_keys=True)
    frame = pd.DataFrame([
        {**row, "value": json.dumps(row["value"], allow_nan=False), "reason": row["reason"] or "",
         "input_sha256": input_sha256, "tested_matrix": tested_matrix, "roles": encoded_roles}
        for row in rows
    ], columns=COLUMNS)
    tables.publish(TABLE, {TABLE: frame}, required={TABLE: COLUMNS}, state=state, g=g, ctx=ctx)
    g["dataset_features.n_ok"] = sum(row["status"] == "ok" for row in rows)
