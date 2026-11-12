"""Validate and publish the compute helper's closed shape table."""
import math
import re

from manyruns import installation
from manyruns.pipeline.steps import _StepSkipped, _StepUnsupported

COLUMNS = ("entity_id", "parent_entity_id", "entity_kind", "stat_name", "scale_kind",
           "scale_value", "aggregation", "value", "n_cells_used", "status", "reason",
           "replicate", "input_space", "preprocessing_sha256", "knn_backend", "dtype",
           "metric_version")
FIELDS = set(COLUMNS) - {"entity_id", "parent_entity_id", "replicate", "input_space",
                         "preprocessing_sha256"}
GRID = (5, 10, 20, 40, 80, 160)


def _helper():
    try:
        from manylatents.metrics.shape_profile import shape_profile
    except ImportError as exc:
        raise _StepUnsupported("shape_profile needs manylatents.metrics.shape_profile; "
                               + installation.RUNTIME_REPAIR) from exc
    return shape_profile


def _keys(selection):
    core = [("log_rk_r15", "k", k, a) for k in GRID for a in ("median", "iqr")]
    core += [(name, "none", None, "value") for name in ("window_lo", "window_hi", "n_cells")]
    result = [("dataset", *key) for key in core]
    if selection:
        result += [("selection", *key) for key in core]
        result += [("selection", name, "k", k, a)
                   for name, a in (("conductance", "share"), ("selection_fidelity", "value"))
                   for k in GRID]
        result += [("selection", "relative_"+name, sk, k, a) for name, sk, k, a in core[:14]]
    return result


def _check_rows(rows, *, selection):
    if type(rows) is not list or len(rows) != len(_keys(selection)):
        raise ValueError("shape_profile: wrong row count")
    keys = []
    for r in rows:
        if type(r) is not dict or set(r) != FIELDS:
            raise ValueError("shape_profile: wrong fields")
        text_fields = FIELDS - {"scale_value", "value", "n_cells_used"}
        if any(type(r[key]) is not str for key in text_fields):
            raise ValueError("shape_profile: non-native text")
        keys.append(tuple(r[k] for k in ("entity_kind", "stat_name", "scale_kind", "scale_value", "aggregation")))
        if type(r["n_cells_used"]) is not int or r["n_cells_used"] < 0:
            raise ValueError("shape_profile: invalid count")
        if r["scale_value"] is not None and type(r["scale_value"]) is not int:
            raise ValueError("shape_profile: invalid scale")
        status, reason, value = r["status"], r["reason"], r["value"]
        if status not in ("ok", "censored", "empty", "undefined") or type(reason) is not str:
            raise ValueError("shape_profile: invalid status")
        if (status == "ok") != (reason == ""):
            raise ValueError("shape_profile: inconsistent reason")
        bound = r["stat_name"] == "window_hi" and status == "censored" and reason == "window_right_censored"
        if status == "ok" or bound:
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("shape_profile: invalid value")
        elif value is not None:
            raise ValueError("shape_profile: non-ok value")
        if r["dtype"] != "float64" or r["knn_backend"] != "scipy_exact":
            raise ValueError("shape_profile: invalid backend")
        if type(r["metric_version"]) is not str or not re.fullmatch(r"manylatents==[^:]+:shape_profile", r["metric_version"]):
            raise ValueError("shape_profile: invalid version")
    if keys != _keys(selection):
        raise ValueError("shape_profile: wrong row keys or order")


def step(state, g, params, ctx):
    from manyruns.pipeline import tables
    from manyruns.pipeline.features import input_sha256
    from manyruns.profile import preprocessing
    import pandas as pd

    if set(params) - {"extended"} or type(params.get("extended", False)) is not bool:
        raise _StepSkipped("shape_profile: invalid_parameters")
    if params.get("extended", False):
        raise _StepSkipped("not_implemented")
    if ctx.get("profile") and not ctx.get("profile_prepared"):
        raise _StepSkipped("profile_preparation_unavailable")
    selection = ctx.get("shape_selection")
    rows = _helper()(state["X"], selection=selection, layout=ctx.get("shape_layout"), extended=False)
    _check_rows(rows, selection=selection is not None)
    original = (ctx.get("original_input") or {}).get("object")
    digest = ctx.get("input_sha256") or input_sha256(
        state["X"] if original is None else original, ctx.get("source_path"))
    descriptor, preprocessing_digest = preprocessing(ctx)
    space = ctx.get("input_space_name", "input")
    selection_id = g.get("derive.selection.sha256")
    if not re.fullmatch("[0-9a-f]{64}", digest) or (selection is not None and not re.fullmatch("[0-9a-f]{64}", selection_id or "")):
        raise ValueError("shape_profile: invalid entity identity")
    encoded = []
    for r in rows:
        selected = r["entity_kind"] == "selection"
        encoded.append({**r, "entity_id": selection_id if selected else digest,
                        "parent_entity_id": digest if selected else "", "replicate": 0,
                        "input_space": space, "preprocessing_sha256": preprocessing_digest,
                        "scale_value": "" if r["scale_value"] is None else r["scale_value"],
                        "value": "" if r["value"] is None else repr(r["value"])})
    frame = pd.DataFrame(encoded, columns=COLUMNS)
    tables.publish("shape_profile", {"shape_profile": frame}, required={"shape_profile": COLUMNS},
                   state=state, g=g, ctx=ctx)
    g["profile.preprocessing_sha256"] = preprocessing_digest
    g["profile.input_space"] = space
