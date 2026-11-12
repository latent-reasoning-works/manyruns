"""Pinned profile preparation and provenance of the measured matrix."""
import hashlib
import json


def preprocessing(ctx):
    inherited = ctx.get("input_space_manifest")
    if inherited is not None:
        return inherited["preprocessing"], inherited["preprocessing_sha256"]
    descriptor = {"branch": "point_cloud" if ctx.get("profile") else "none",
                  "steps": ctx.get("executed_steps", []), "definition_version": "profile/1"}
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return descriptor, hashlib.sha256(encoded).hexdigest()

PINS = (
    ("dataset_features", "analysis", {"roles": {}, "layer": None, "overrides": None}),
    ("profile_prepare", "prep", {"max_components": 50, "center": True, "scale": False,
                                "dtype": "float64", "solver": "full"}),
    ("shape_profile", "analysis", {"extended": False}),
)


def check_pins(recipe, *, overrides=None):
    if recipe.get("kind", "recipe") != "profile" and recipe.get("name") != "profile":
        return []
    problems = []
    if recipe.get("kind") != "profile":
        problems.append("pinned_profile_parameter:kind")
    steps = recipe.get("steps") or []
    if len(steps) != len(PINS):
        return problems + ["pinned_profile_parameter:steps"]
    for step, (name, group, params) in zip(steps, PINS):
        if step.get("name") != name or step.get("group") != group or step.get("tool") or step.get("limits"):
            problems.append(f"pinned_profile_parameter:{name}")
        actual = step.get("params") or {}
        for key in set(params) | set(actual):
            if key not in actual or key not in params or type(actual[key]) is not type(params[key]) or actual[key] != params[key]:
                problems.append(f"pinned_profile_parameter:{name}.{key}")
    for name, params in (overrides or {}).items():
        for key in params:
            problems.append(f"pinned_profile_parameter:{name}.{key}")
    return problems


def classify_branch(feature_rows, *, input_kind):
    evidence = [r for r in feature_rows if r["key"] == "counts_like" and
                r["status"] == "ok" and (type(r["value"]) is bool or r["value"] in ("raw_integer", "processed", "unknown"))]
    if len(evidence) != 1:
        raise ValueError("classification_unavailable")
    if evidence[0]["value"] is True or evidence[0]["value"] == "raw_integer":
        return "counts"
    return "processed" if input_kind in ("processed", "counts", "annotated") else "point_cloud"


def _pca_helper():
    from manyruns import installation
    from manyruns.pipeline.steps import _StepUnsupported
    try:
        from manylatents.utils.profile import point_cloud_pca
    except ImportError as exc:
        raise _StepUnsupported("profile_prepare needs manylatents.utils.profile; "
                               + installation.RUNTIME_REPAIR) from exc
    return point_cloud_pca


def prepare(state, g, params, ctx):
    from manyruns.pipeline.steps import _StepSkipped
    rows = ctx.get("feature_rows")
    if not rows:
        raise _StepSkipped("classification_unavailable")
    try:
        branch = classify_branch(rows, input_kind="annotated" if state.get("genes") is not None else ctx.get("input_kind", "point_cloud"))
    except ValueError as exc:
        raise _StepSkipped(str(exc)) from exc
    if branch != "point_cloud":
        raise _StepSkipped("branch_not_built:" + branch)
    result = _pca_helper()(state["X"], max_components=params["max_components"],
                           center=params["center"], scale=params["scale"],
                           dtype=params["dtype"], solver=params["solver"])
    ctx["profile_prepared"] = True
    ctx["input_space_name"] = f"pca{result.shape[1]}"
    ctx["effective_params"] = {**params, "n_components": result.shape[1]}
    return {"X": result}


def save_input_space(state, g, ctx, carry, steps):
    """Persist the eligible working matrix once with its actual executed history."""
    from manyruns import artifacts
    from manyruns.pipeline.features import input_sha256
    from scipy import sparse
    import numpy as np

    if not any(s.get("outcome") in ("ok", "reported") for s in steps):
        return
    if ctx.get("out_dir") is None or ctx.get("input_space_attempted") or ctx.get("derive"):
        return
    ctx["input_space_attempted"] = True
    x = state.get("X")
    reason = ("unavailable" if x is None else "sparse" if sparse.issparse(x)
              else "too_many_columns" if x.shape[1] > 512 else "")
    if reason:
        g["input_space.saved"] = False
        g["input_space.reason"] = reason
        return
    descriptor, digest = preprocessing(ctx)
    original = (ctx.get("original_input") or {}).get("object")
    entity = ctx.get("input_sha256") or input_sha256(x if original is None else original, ctx.get("source_path"))
    manifest = {"input_space": ctx.get("input_space_name", "input"), "input_sha256": entity,
                "preprocessing_sha256": digest, "preprocessing": descriptor}
    value = np.asarray(x, dtype=np.float64).copy()
    index = steps[-1].get("index", max(0, len(steps)-1))
    name = steps[-1]["name"] if steps else "input"
    path = artifacts.save_array(ctx["out_dir"], ctx.get("run_id"), index, name,
                                "input_space", value, max_bytes=value.nbytes)
    g["input_space.saved"] = path is not None
    if path is None:
        g["input_space.reason"] = "write_failed"
        return
    state["input_space"] = value
    ids = ctx.get("sample_ids")
    rows = state.get("rows", np.arange(len(value)))
    ids = np.asarray(ids)[rows].tolist() if ids is not None else list(map(str, rows))
    carry["written"][path] = {"step": name, "index": index, "slot": "input_space",
                              "input_space": manifest,
                              "row_identity": {"sample_ids": ids, "labels": None, "label_kind": None}}
    if ctx.get("sample_ids") is None or ctx.get("metadata_aligned", True) is not True:
        carry["written"][path].pop("row_identity", None)
    if steps:
        steps[-1].setdefault("artifacts", {})["input_space"] = path


def resolve_parent_space(parent, store, *, parent_index=None, _seen=None):
    """Resolve only a persisted measured matrix; never reopen source data."""
    from manyruns import derive
    import numpy as np

    seen = set() if _seen is None else set(_seen)
    if parent.get("run_id") in seen:
        raise ValueError("parent_input_space_unavailable")
    seen.add(parent.get("run_id"))
    if parent.get("ok") is not True or parent.get("cancelled"):
        raise ValueError("parent_input_space_unavailable")
    try:
        manifest = derive._manifest(parent, store)
    except (OSError, ValueError) as exc:
        raise ValueError("parent_input_space_unavailable") from exc
    entries = [(p, e) for p, e in manifest.items() if e["slot"] == "input_space"]
    if not entries and str(parent.get("recipe", "")).startswith("derive:"):
        ref = parent.get("parent") or {}
        if not ref.get("run_id") or ref["run_id"] == parent["run_id"]:
            raise ValueError("parent_input_space_unavailable")
        return resolve_parent_space(derive.parent_row(ref["run_id"], store), store, _seen=seen)
    if parent_index is not None:
        entries = [(p,e) for p,e in entries if e["index"] <= parent_index]
    if entries:
        latest = max(e["index"] for _, e in entries)
        entries = [(p,e) for p,e in entries if e["index"] == latest]
    if len(entries) != 1:
        raise ValueError("parent_input_space_unavailable")
    path, entry = entries[0]
    metadata = entry.get("input_space") or {}
    if set(metadata) != {"input_space", "input_sha256", "preprocessing_sha256", "preprocessing"}:
        raise ValueError("parent_input_space_unavailable")
    descriptor = metadata["preprocessing"]
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    if set(descriptor) != {"branch", "steps", "definition_version"} or hashlib.sha256(encoded).hexdigest() != metadata["preprocessing_sha256"]:
        raise ValueError("parent_input_space_unavailable")
    x = np.load(derive._path(path, store), allow_pickle=False)
    ids = (entry.get("row_identity") or {}).get("sample_ids")
    if x.ndim != 2 or x.dtype != np.float64 or x.shape[1] > 512 or not isinstance(ids, list) or len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError("parent_input_space_unavailable")
    layout = None
    layouts = [(p, e) for p, e in manifest.items() if e["slot"] == "emb" and
               (parent_index is None or e["index"] == parent_index)]
    if layouts:
        last = max(e["index"] for _, e in layouts)
        layouts = [(p,e) for p,e in layouts if e["index"] == last]
    if len(layouts) == 1:
        p, e = layouts[0]
        other_ids = (e.get("row_identity") or {}).get("sample_ids") or []
        candidate = np.load(derive._path(p, store), allow_pickle=False)
        if candidate.shape == (len(x),2) and len(set(other_ids)) == len(x) and set(other_ids) == set(ids):
            positions = {cell:i for i,cell in enumerate(other_ids)}
            layout = candidate[[positions[cell] for cell in ids]]
    return {"X": x, "ids": ids, "manifest": metadata, "layout": layout,
            "index": entry["index"], "run_id": parent["run_id"]}
