"""The `tool:` seam — a recipe step whose implementation lives in another interpreter.

WHY THIS EXISTS. CLAUDE.md's extension path is "a tool is a recipe step, dispatched by its
group", and it silently assumes the tool can be imported. Some cannot: `pyrovelocity 0.4.5`
resolves only against `numpy<2` and this product runs 2.2.6, so no extra, marker or optional
import can host it — the environments are mutually exclusive, measured. Without this seam the
only answers are "decline the tool" or "fork the product", and both are wrong.

WHAT IT IS NOT. Not an `--engine` value: the engine axis is substrate/fidelity for ONE verb and
must not become the seam for calling things (CLAUDE.md). Not a new `group` either — "runs
elsewhere" is a LOCATION, and groups are keyed on the FACT a step produces. The proof that
location cannot be a group is the queue itself: an external scGPT produces an `embedding`
(`latent`), an external doublet caller produces nothing (`prep`), an external velocity model
produces a field. They share no fact, so they cannot share a group.

So `tool:` is the **substrate axis extended from machine to environment** — the same axis the
`dispatcher` skill already owns for local/gpu/cluster. A step keeps its honest `group`, the
calculus keeps working on it unchanged, and this executor is chosen ahead of the group's
engine executor when the step names a tool.

THE PROTOCOL IS A FILE, IN BOTH DIRECTIONS, and that is forced rather than chosen — two
irreconcilable dependency closures cannot share objects, only bytes:

    request.json  ──▶  <tool-env>/bin/python  <adapter>  request.json
    input.h5ad    ──▶                                     ──▶  output.h5ad + result.json

THE MODEL SEAM. A manifest with `output.schema: result-v1` is a model manifest: the tool writes
a version 1 `result.json` that names each fact it returns, with each array as a payload file by
relative path and sha256, and `check_result` checks all of it against the run before anything is
merged. manyruns adapts to an unmodified library by one of two routes. A library that takes an
array and returns arrays in ONE call needs no adapter: its manifest says `mode: call`, and the
bundled call adapter (`adapters/call_adapter.py`) makes the call its `call` block describes, in
the tool's interpreter. A library that needs several steps, a special input object, or returns
a graph gets a PINNED ADAPTER, written on the manyruns side for one exact upstream version.
A third mode, `precomputed`, starts nothing: the result already exists, and it is found by the
sha256 of the run's cell list and marked as precomputed in the record.

REFUSAL, NEVER IMPROVISATION. A tool that is not built, not locked, or not declared makes the
step `_StepUnsupported` — the record does not describe the analysis the recipe asked for, so the
run is not `ok` — and the reason carries the exact command that fixes it. Nothing here builds an
environment: the realized env for one tool is 2.4 GB (measured), and a step that downloads that
mid-run is an outage, not a step.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from manyruns import toolchain as _toolchain
from manyruns.pipeline.steps import _StepSkipped, _StepUnsupported

#: `fact -> (destination, how it arrives)`. It held ONE entry, `velocity`, and its comment said
#: that a second entry would tell us the general shape. The model seam is that second entry, with
#: five concrete callers (an embedding, a partition, a pseudotime, trajectories, a graph and
#: class probabilities from outside models), so the shape is now written down:
#:
#:   obsm    an `.h5ad` matrix, the `velocity` path, unchanged. Its per-cell posterior summaries
#:           ride alongside in `obs` and land in a companion slot.
#:   npy     one `.npy` file beside `result.json`, into the named state slot.
#:   npz     the graph interchange arrays in one `.npz`, into `state["graph"]` as a dict, and
#:           each array into `state["extras"]` as `<step>.graph.<array>`.
#:   inline  a value inside `result.json`, into the g-vector as `<step>.<fact>.<name>`.
FACT_SLOTS: dict[str, tuple[str, str]] = {
    "velocity": ("velocity", "obsm"),
    "embedding": ("emb", "npy"),
    "clusters": ("clusters", "npy"),
    "pseudotime": ("pseudotime", "npy"),
    "trajectories": ("trajectories", "npy"),
    "graph": ("graph", "npz"),
    "class_probabilities": ("g_vector", "inline"),
}

#: The caveat a precomputed step adds to its run. The `serving.MOCK_CAVEAT` pattern: the result
#: says what it is in its own record, so a precomputed output never reads as one manyruns ran.
PRECOMPUTED_CAVEAT = ("precomputed output: manyruns did not run this model; it read and checked "
                      "its files only")

#: Companion slot for a fact's per-cell uncertainty, when the manifest declares `output.obs`.
UNCERTAINTY_SLOT = {"velocity": "velocity_uncertainty"}


# ── the version 1 result ─────────────────────────────────────────────────────
# `check_result` is the open contract of `result.json`, version 1. It states every key, type and
# rule. With the golden adapter fixtures (`tests/fixtures/model_seam/`) it is the whole contract:
# no JSON Schema file describes the result, and an adapter in another language checks its output
# with `manyruns tools verify`.

#: The version of `result.json` that `check_result` reads.
RESULT_SCHEMA_VERSION = 1

#: The facts a model can return through the seam. `velocity` stays on the `.h5ad` path.
SEAM_FACTS = ("embedding", "clusters", "pseudotime", "trajectories", "graph",
              "class_probabilities")

#: The seam facts with one row per cell.
PER_CELL_FACTS = ("embedding", "clusters", "pseudotime")

#: The top-level keys of a version 1 result. Closed.
RESULT_KEYS = ("schema_version", "ok", "reason", "n_cells", "cells_sha256", "facts", "scalars")

#: The keys a declined result (`ok: false`) may hold.
DECLINED_KEYS = ("schema_version", "ok", "reason", "scalars")

#: The arrays of the `graph` fact, as the graph interchange names them. `cell_node` is optional.
GRAPH_ARRAYS = ("edges", "edge_weights", "node_ids", "node_coordinates")
GRAPH_OPTIONAL = ("cell_node",)

_HEX = frozenset("0123456789abcdef")


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    import math

    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _path_problem(fact: str, path: Any) -> "str | None":
    suffix = ".npz" if fact == "graph" else ".npy"
    if not isinstance(path, str) or not path:
        return f"facts.{fact}.path must be a non-empty string, got {path!r}"
    if "\\" in path:
        return f"facts.{fact}.path {path!r} holds a backslash; use a relative POSIX path"
    if path.startswith("/"):
        return f"facts.{fact}.path {path!r} is absolute; it must be relative to result.json"
    if any(part in ("..", ".", "") for part in path.split("/")):
        return f"facts.{fact}.path {path!r} has a '.', '..' or empty part"
    if not path.endswith(suffix):
        return f"facts.{fact}.path {path!r} must end in {suffix}"
    return None


def _entry_problems(fact: str, entry: Any) -> list[str]:
    from manyruns.vocab import TOPOLOGIES

    if not isinstance(entry, dict):
        return [f"facts.{fact} must be an object, got {type(entry).__name__}"]
    want = ("value",) if fact == "class_probabilities" else ("path", "sha256")
    bad = [f"facts.{fact}: unknown key {k!r} (allowed: {list(want)})"
           for k in entry if k not in want]
    bad += [f"facts.{fact}: missing key {k!r}" for k in want if k not in entry]
    if bad:
        return bad
    if fact != "class_probabilities":
        problem = _path_problem(fact, entry["path"])
        if problem:
            bad.append(problem)
        if not _is_hex64(entry["sha256"]):
            bad.append(f"facts.{fact}.sha256 must be 64 lowercase hex characters")
        return bad
    value = entry["value"]
    if not isinstance(value, dict):
        return [f"facts.class_probabilities.value must be an object of {list(TOPOLOGIES)}"]
    bad += [f"facts.class_probabilities.value: unknown name {k!r} (allowed: {list(TOPOLOGIES)})"
            for k in value if k not in TOPOLOGIES]
    bad += [f"facts.class_probabilities.value: missing name {k!r}"
            for k in TOPOLOGIES if k not in value]
    for name in TOPOLOGIES:
        v = value.get(name)
        if name in value and not (_is_finite_number(v) and 0.0 <= v <= 1.0):
            bad.append(f"facts.class_probabilities.value.{name} must be a finite number in "
                       f"[0, 1], got {v!r}")
    return bad


def _structure_problems(doc: Any, produces: Any) -> list[str]:
    if not isinstance(doc, dict):
        return [f"the result must be an object, got {type(doc).__name__}"]
    bad = [f"unknown key {k!r} (allowed: {list(RESULT_KEYS)})" for k in doc
           if k not in RESULT_KEYS]
    version = doc.get("schema_version")
    if "schema_version" not in doc:
        bad.append("schema_version is required")
    elif not _is_int(version) or version != RESULT_SCHEMA_VERSION:
        bad.append(f"schema_version must be the integer {RESULT_SCHEMA_VERSION}, "
                   f"got {version!r}")
    ok = doc.get("ok")
    if "ok" not in doc:
        bad.append("ok is required")
    elif not isinstance(ok, bool):
        bad.append(f"ok must be a boolean, got {ok!r}")

    if ok is False:
        bad += [f"{k} is not allowed when ok is false" for k in doc
                if k in RESULT_KEYS and k not in DECLINED_KEYS]
        reason = doc.get("reason")
        if not isinstance(reason, str) or not reason:
            bad.append(f"reason must be a non-empty string when ok is false, got {reason!r}")
    elif ok is True:
        if "reason" in doc:
            bad.append("reason is not allowed when ok is true")
        n = doc.get("n_cells")
        if "n_cells" not in doc:
            bad.append("n_cells is required when ok is true")
        elif not _is_int(n) or n < 1:
            bad.append(f"n_cells must be an integer >= 1, got {n!r}")
        if "cells_sha256" in doc and not _is_hex64(doc["cells_sha256"]):
            bad.append("cells_sha256 must be 64 lowercase hex characters")
        facts = doc.get("facts")
        if not isinstance(facts, dict) or not facts:
            bad.append(f"facts must be a non-empty object of seam facts {list(SEAM_FACTS)}")
        else:
            for fact, entry in facts.items():
                if fact not in SEAM_FACTS:
                    bad.append(f"facts: {fact!r} is not a seam fact (allowed: "
                               f"{list(SEAM_FACTS)})")
                else:
                    bad += _entry_problems(fact, entry)
            if produces is not None:
                declared = list(produces)
                bad += [f"facts: {f!r} is not in the manifest's produces {declared}"
                        for f in facts if f not in declared]
                bad += [f"facts: {f!r} is in the manifest's produces {declared} and absent "
                        f"from the result" for f in declared if f not in facts]

    if "scalars" in doc:
        scalars = doc["scalars"]
        if not isinstance(scalars, dict):
            bad.append(f"scalars must be an object, got {type(scalars).__name__}")
        else:
            for key, value in scalars.items():
                if str(key).startswith("tool_"):
                    bad.append(f"scalars: {key!r} starts with the reserved prefix tool_, "
                               f"which manyruns writes for provenance")
                elif not (isinstance(value, (bool, str)) or _is_finite_number(value)):
                    bad.append(f"scalars: {key!r} must be a finite number, a boolean or a "
                               f"string, got {value!r}")
    return bad


def _load_array(fact: str, path: Path) -> "tuple[Any, str | None]":
    """One payload file, loaded WITHOUT pickle: a precomputed file can come from anyone."""
    import numpy as np

    try:
        loaded = np.load(path, allow_pickle=False)
        if fact != "graph":
            if not isinstance(loaded, np.ndarray):
                return None, f"facts.{fact}: {path.name} is not a .npy array"
            return loaded, None
        if not hasattr(loaded, "files"):
            return None, f"facts.graph: {path.name} is not an .npz archive"
        with loaded:
            return {k: loaded[k] for k in loaded.files}, None
    except Exception as e:  # noqa: BLE001 - any unreadable file is one problem, never a crash
        return None, f"facts.{fact}: cannot load {path.name} without pickle ({type(e).__name__})"


def _array_problems(fact: str, a: Any, n_cells: int, input_dim: Any) -> list[str]:
    import numpy as np

    floating = np.issubdtype(a.dtype, np.floating)
    if fact == "embedding":
        if a.ndim != 2 or a.shape[1] < 1:
            return [f"facts.embedding: must be 2-D (n_cells, d), got shape {a.shape}"]
        if a.shape[0] != n_cells:
            return [f"facts.embedding: {a.shape[0]} rows for {n_cells} cells"]
        if not floating:
            return [f"facts.embedding: dtype {a.dtype} is not a floating dtype"]
        if not np.isfinite(a).all():
            return ["facts.embedding: holds non-finite values"]
    elif fact in ("clusters", "pseudotime"):
        if a.ndim != 1:
            return [f"facts.{fact}: must be 1-D (n_cells,), got shape {a.shape}"]
        if a.shape[0] != n_cells:
            return [f"facts.{fact}: {a.shape[0]} rows for {n_cells} cells"]
        if fact == "clusters" and not np.issubdtype(a.dtype, np.integer):
            return [f"facts.clusters: dtype {a.dtype} is not an integer dtype"]
        if fact == "pseudotime":
            if not floating:
                return [f"facts.pseudotime: dtype {a.dtype} is not a floating dtype"]
            if not np.isfinite(a).all():
                return ["facts.pseudotime: holds non-finite values"]
    elif fact == "trajectories":
        if a.ndim != 3:
            return [f"facts.trajectories: must be 3-D (B, T, d), got shape {a.shape}"]
        if a.shape[0] < 2 or a.shape[1] < 1:
            return [f"facts.trajectories: needs at least 2 time bins and 1 trajectory, got "
                    f"shape {a.shape}"]
        if not floating:
            return [f"facts.trajectories: dtype {a.dtype} is not a floating dtype"]
        if not np.isfinite(a).all():
            return ["facts.trajectories: holds non-finite values"]
        if input_dim is not None and a.shape[2] != int(input_dim):
            return [f"facts.trajectories: d is {a.shape[2]} and the input matrix has "
                    f"{int(input_dim)} columns; a trajectory must be in the input's frame"]
    return []


def _graph_problems(arrays: dict, n_cells: int) -> list[str]:
    import numpy as np

    unknown = [k for k in arrays if k not in GRAPH_ARRAYS + GRAPH_OPTIONAL]
    if unknown:
        return [f"facts.graph: unknown array {k!r} (allowed: "
                f"{list(GRAPH_ARRAYS + GRAPH_OPTIONAL)})" for k in unknown]
    missing = [k for k in GRAPH_ARRAYS if k not in arrays]
    if missing:
        return [f"facts.graph: missing array {k!r}" for k in missing]
    edges, weights = arrays["edges"], arrays["edge_weights"]
    nodes, coords = arrays["node_ids"], arrays["node_coordinates"]
    if edges.dtype != np.int64 or edges.ndim != 2 or edges.shape[1] != 2:
        return [f"facts.graph: edges must be int64 of shape (E, 2), got {edges.dtype} "
                f"{edges.shape}"]
    if (not np.issubdtype(weights.dtype, np.floating) or weights.ndim != 1
            or weights.shape[0] != edges.shape[0]):
        return [f"facts.graph: edge_weights must be floating of shape ({edges.shape[0]},), got "
                f"{weights.dtype} {weights.shape}"]
    if nodes.dtype != np.int64 or nodes.ndim != 1:
        return [f"facts.graph: node_ids must be int64 of shape (M,), got {nodes.dtype} "
                f"{nodes.shape}"]
    if len(np.unique(nodes)) != len(nodes):
        return ["facts.graph: node_ids are not unique"]
    if (not np.issubdtype(coords.dtype, np.floating) or coords.ndim != 2
            or coords.shape[0] != nodes.shape[0]):
        return [f"facts.graph: node_coordinates must be floating of shape ({nodes.shape[0]}, D), "
                f"got {coords.dtype} {coords.shape}"]
    known = set(nodes.tolist())
    bad = []
    stray = sorted(set(edges.ravel().tolist()) - known)
    if stray:
        bad.append(f"facts.graph: edges reference node(s) {stray} that are not in node_ids")
    if "cell_node" in arrays:
        cell_node = arrays["cell_node"]
        if cell_node.dtype != np.int64 or cell_node.shape != (n_cells,):
            bad.append(f"facts.graph: cell_node must be int64 of shape ({n_cells},), got "
                       f"{cell_node.dtype} {cell_node.shape}")
        else:
            stray = sorted(set(cell_node.tolist()) - known)
            if stray:
                bad.append(f"facts.graph: cell_node references node(s) {stray} that are not "
                           f"in node_ids")
    return bad


def _check_and_load(doc: Any, *, base: Any = None, n_cells: Any = None, cell_list: Any = None,
                    input_dim: Any = None, produces: Any = None) -> tuple[list[str], dict]:
    """`check_result`, and the arrays it loaded. The merge uses the arrays that were checked."""
    import hashlib

    bad = _structure_problems(doc, produces)
    if bad or base is None or doc.get("ok") is not True:
        return bad, {}
    base = Path(base)
    if n_cells is not None and doc["n_cells"] != int(n_cells):
        return [f"n_cells is {doc['n_cells']} in the result and {int(n_cells)} in the run"], {}
    n = int(n_cells) if n_cells is not None else doc["n_cells"]
    if "cells_sha256" in doc and cell_list is not None:
        joined = "\n".join(str(c) for c in cell_list)
        if hashlib.sha256(joined.encode("utf-8")).hexdigest() != doc["cells_sha256"]:
            return ["cells_sha256 does not match the run's cell list: the result describes "
                    "other cells, or the same cells in another order"], {}
    root = base.resolve()
    loaded: dict = {}
    for fact, entry in doc["facts"].items():
        if fact == "class_probabilities":
            loaded[fact] = dict(entry["value"])
            continue
        path = base / entry["path"]
        real = path.resolve()
        if real != root and root not in real.parents:
            bad.append(f"facts.{fact}: {entry['path']!r} resolves outside the result folder")
            continue
        if not real.is_file():
            bad.append(f"facts.{fact}: {entry['path']!r} does not exist")
            continue
        if hashlib.sha256(real.read_bytes()).hexdigest() != entry["sha256"]:
            bad.append(f"facts.{fact}: sha256 does not match the bytes of {entry['path']!r}")
            continue
        value, problem = _load_array(fact, real)
        if problem:
            bad.append(problem)
            continue
        found = (_graph_problems(value, n) if fact == "graph"
                 else _array_problems(fact, value, n, input_dim))
        bad += found
        if not found:
            loaded[fact] = value
    return bad, loaded


def check_result(doc: Any, *, base: Any = None, n_cells: Any = None, cell_list: Any = None,
                 input_dim: Any = None, produces: Any = None) -> list[str]:
    """The problems of a version 1 result document. An empty list means valid.

    THE OPEN CONTRACT of `result.json`. Every rule is stated here:

    Top-level keys (closed; any other key is a problem):

      schema_version  integer, required, equals RESULT_SCHEMA_VERSION; a boolean is refused
      ok              boolean, required
      reason          non-empty string, required when ok is false; absent when ok is true
      n_cells         integer >= 1, required when ok is true; a boolean is refused
      cells_sha256    optional; 64 lowercase hex characters: sha256 of the cell ids joined by
                      "\\n", in UTF-8
      facts           non-empty object, required when ok is true; each key is a seam fact
      scalars         optional object; each value is a finite number, a boolean or a string;
                      no key starts with `tool_`, which `toolchain.provenance` writes

    With ok false, only schema_version, ok, reason and scalars are allowed.

    A fact entry is `{"path", "sha256"}` for embedding, clusters, pseudotime, trajectories and
    graph, and `{"value"}` for class_probabilities. `path` is a relative POSIX path with no `.`
    or `..` part and no backslash, ending in `.npz` for graph and `.npy` otherwise. `sha256` is
    64 lowercase hex characters. `value` holds exactly the six names of `vocab.TOPOLOGIES`, each
    a finite number in [0, 1] (they need not sum to 1).

    With `produces` given, the facts must equal it: an undeclared fact and a declared fact that
    is absent are both problems that name the fact and the word `produces`.

    With `base` given (the folder of `result.json`), each payload file is checked too. It must
    resolve inside `base`, match its sha256, and load with `allow_pickle=False`. `n_cells` must
    equal the run's, and `cells_sha256`, when present and `cell_list` is given, must match it.

      embedding     2-D floating (n_cells, d >= 1), finite
      clusters      1-D integer (n_cells,)
      pseudotime    1-D floating (n_cells,), finite
      trajectories  3-D floating (B >= 2, T >= 1, d), finite; d == input_dim when given.
                    Not per cell.
      graph         edges int64 (E, 2), edge_weights floating (E,), node_ids int64 (M,) unique,
                    node_coordinates floating (M, D); every edge value is a node id; optional
                    cell_node int64 (n_cells,) of node ids; no other array
    """
    return _check_and_load(doc, base=base, n_cells=n_cells, cell_list=cell_list,
                           input_dim=input_dim, produces=produces)[0]


def _refuse_unbuilt(name: str, resolved: dict) -> None:
    fix = resolved.get("fix")
    raise _StepUnsupported(
        f"{resolved.get('reason')}"
        + (f" — fix: {fix}" if fix else "")
    )


def _write_request(workdir: Path, cfg: dict, params: dict, seed: Any,
                   input_path: Path, v1: "dict | None" = None) -> tuple[Path, Path, Path]:
    """`request.json`. On the version 1 path (`v1` given) it also carries `result_schema`,
    `n_cells`, `input_dim`, the optional `time` file and, in call mode, the `call` block. It
    carries no cell list, so a run-mode adapter may omit `cells_sha256`."""
    workdir.mkdir(parents=True, exist_ok=True)
    req_path, out_path, res_path = (workdir / "request.json", workdir / "output.h5ad",
                                    workdir / "result.json")
    merged = dict(cfg.get("params") or {})
    merged.update(params or {})
    request = {
        "tool": cfg["name"], "version": cfg["version"],
        "params": merged, "seed": int(seed) if seed is not None else 0,
        "input": str(input_path), "output": str(out_path), "result": str(res_path),
        "workdir": str(workdir),
    }
    if v1 is not None:
        request.update({"result_schema": _toolchain.RESULT_SCHEMA,
                        "n_cells": v1["n_cells"], "input_dim": v1["input_dim"]})
        if v1.get("time") is not None:
            import numpy as np

            time_path = workdir / "time.npy"
            np.save(time_path, v1["time"], allow_pickle=False)
            request["time"] = str(time_path)
        if _toolchain.tool_mode(cfg) == "call":
            request["call"] = cfg["call"]
    req_path.write_text(json.dumps(request, indent=2))
    return req_path, out_path, res_path


def _time_axis(state: dict) -> Any:
    """The run's time labels as floats, when the run's labels are a time axis; else None."""
    if state.get("label_kind") != "time" or state.get("labels") is None:
        return None
    import numpy as np

    try:
        return np.asarray(state["labels"], dtype=float)
    except (TypeError, ValueError):
        return None


def _input_shape(input_path: Path, state: dict) -> tuple[int, int]:
    """`(n_cells, input_dim)` of what the tool reads: the rows and columns of the input."""
    import numpy as np

    if input_path.suffix == ".npy":
        shape = np.load(input_path, mmap_mode="r", allow_pickle=False).shape
        return int(shape[0]), int(shape[1]) if len(shape) > 1 else 1
    rows, cols = state.get("rows"), state.get("cols")
    return (int(len(rows)) if rows is not None else 0,
            int(len(cols)) if cols is not None else 0)


def _cell_list(state: dict, ctx: dict) -> Any:
    """The ordered ids of the cells the run holds at this step, or None without ids."""
    ids = ctx.get("sample_ids")
    if ids is None:
        return None
    import numpy as np

    ids = np.asarray(ids)
    rows = state.get("rows")
    # No selection (a state built without a matrix) means the run holds every cell it was given.
    if rows is None or len(rows) == 0:
        return ids
    return ids[rows]


def _input_path(cfg: dict, state: dict, ctx: dict, workdir: Path) -> Path:
    """Assemble what the tool reads — and refuse the compositions that would mislead it."""
    kind = (cfg.get("input") or {}).get("kind")
    if kind == "source_file":
        src = ctx.get("source_path")
        if not src:
            raise _StepSkipped(
                f"{cfg['name']} reads the original file (it needs "
                f"{(cfg.get('input') or {}).get('layers') or 'fields'} that manyruns's frame "
                f"does not carry), and this run has no file path — it was handed an array or a "
                f"named engine dataset")
        # THE REFUSAL THAT MATTERS. The file still holds every cell; the run may not. A tool fed
        # the file after a filter reports on a population this run has already rejected, and the
        # numbers come back indexed to a different set of cells than `state` holds.
        rows = state.get("rows")
        frame = state.get("frame") or {}
        counts = frame.get("counts")
        full_rows = rows is None or counts is None or len(rows) == getattr(counts, "shape", [0])[0]
        if not full_rows:
            raise _StepSkipped(
                f"{cfg['name']} reads the ORIGINAL file, and this run has already narrowed "
                f"({len(rows)} cells selected) — the file still holds the cells a filter "
                f"removed, so it would report on a population this run rejected. Put the tool "
                f"step before the prep block")
        return Path(src)
    if kind == "state_frame":
        # THE VIEW, not the file. This is what the frame's `layers` slot exists for: the tool
        # gets exactly the cells and genes the run currently holds, so a prep block BEFORE it is
        # composition rather than a contradiction. Written per-run under `out_dir` and never
        # anywhere shared — it is the scientist's counts, which `artifacts.PERSISTED` refuses
        # for the same reason.
        import anndata as ad
        import numpy as np

        from manyruns.pipeline import frame as _frame

        view = _frame.view(state["frame"], state["rows"], state["cols"])
        counts = view["counts"]
        if counts is None:
            raise _StepSkipped(
                f"{cfg['name']} reads the run's frame and this state has none — the engine "
                f"loads a named dataset itself, so manyruns never sees the matrix")
        want = list((cfg.get("input") or {}).get("layers") or ())
        have = view["layers"] or {}
        missing = [name for name in want if name not in have]
        if missing:
            raise _StepSkipped(
                f"{cfg['name']} needs the {missing} layer(s) and this run carries "
                f"{sorted(have)}. Declare them on the step (`layers: {want}`) and load data "
                f"that has them")

        def _dense(m):
            return np.asarray(m.toarray() if hasattr(m, "toarray") else m, dtype="float32")

        out = ad.AnnData(X=_dense(counts))
        for name in want:
            out.layers[name] = _dense(have[name])
        if view["genes"] is not None:
            out.var_names = np.asarray(view["genes"]).astype(object)
        workdir.mkdir(parents=True, exist_ok=True)
        path = workdir / "input.h5ad"
        out.write_h5ad(path)
        return path
    if kind == "state_matrix":
        import numpy as np

        X = state.get("emb") if state.get("emb") is not None else state.get("X")
        if X is None:
            raise _StepSkipped(f"{cfg['name']} takes the working matrix and this state has none")
        path = workdir / "input.npy"
        workdir.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ascontiguousarray(np.asarray(X)))
        return path
    raise _StepUnsupported(f"{cfg['name']}: unknown input.kind {kind!r}")


def _merge(cfg: dict, out_path: Path, state: dict, g: dict, name: str) -> None:
    """Read what the tool wrote, CHECK it against the manifest, then merge. Never the reverse."""
    import anndata as ad
    import numpy as np

    if not out_path.is_file():
        raise _StepSkipped(f"{name} reported success and wrote no output at {out_path}")
    out = ad.read_h5ad(out_path)
    n_cells = None
    for slot in ("emb", "X"):
        if state.get(slot) is not None:
            n_cells = int(np.asarray(state[slot]).shape[0])
            break
    declared = cfg.get("output") or {}
    for key in declared.get("obsm") or []:
        if key not in out.obsm:
            raise _StepSkipped(f"{name} declares obsm[{key!r}] and did not write it")
        field = np.asarray(out.obsm[key])
        if n_cells is not None and field.shape[0] != n_cells:
            # Silent misalignment is the one failure this whole seam must not permit: every
            # per-cell number would be attributed to the wrong cell, with nothing to notice.
            raise _StepSkipped(
                f"{name} returned {field.shape[0]} rows for {n_cells} cells — refusing to "
                f"merge a field that is not indexed to this run's cells")
        slot, _ = FACT_SLOTS.get(key, (key, "obsm"))
        state[slot] = field
        g[f"{name}.{key}_shape"] = list(field.shape)
    obs_keys = [k for k in (declared.get("obs") or []) if k in out.obs]
    if obs_keys:
        unc = {k: np.asarray(out.obs[k], dtype=float) for k in obs_keys}
        for fact in cfg.get("produces") or []:
            if fact in UNCERTAINTY_SLOT:
                state[UNCERTAINTY_SLOT[fact]] = unc


def _merge_v1(doc: dict, base: Path, state: dict, g: dict, name: str, *, n_cells: int,
              cell_list: Any, input_dim: Any, produces: Any) -> None:
    """Merge a version 1 result. CHECK EVERYTHING, THEN ASSIGN: `_merge` assigns each key as it
    goes, and a result refused halfway would leave half a model in the state. Here a refused
    result changes no slot at all."""
    problems, arrays = _check_and_load(doc, base=base, n_cells=n_cells, cell_list=cell_list,
                                       input_dim=input_dim, produces=produces)
    if problems:
        more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
        raise _StepSkipped(f"{name} returned a result that check_result refuses: "
                           f"{problems[0]}{more}")
    for fact, value in arrays.items():
        slot, form = FACT_SLOTS[fact]
        if form == "inline":
            for key, v in value.items():
                g[f"{name}.{fact}.{key}"] = float(v)
        elif form == "npz":
            state[slot] = value
            extras = state.setdefault("extras", {})
            for key, array in value.items():
                extras[f"{name}.{fact}.{key}"] = array
        else:
            state[slot] = value
    for key, value in (doc.get("scalars") or {}).items():
        g[f"{name}.{key}"] = value


def _run_precomputed(name: str, resolved: dict, state: dict, g: dict, ctx: dict) -> None:
    """A precomputed step: find the one result that describes THESE cells, check it, merge it.

    Lookup by content, not by name: the sha256 of the run's cell list selects the one
    `<output.root>/*/result.json` whose `cells_sha256` equals it. One check finds the output and
    proves that it describes these cells in this order. No interpreter starts."""
    import hashlib

    cfg = resolved["cfg"]
    # PROVENANCE FIRST, so a refused precomputed step is still recorded.
    g.update(_toolchain.provenance(name, resolved))
    cells = _cell_list(state, ctx)
    if cells is None:
        raise _StepSkipped(
            f"{name} is precomputed and is found by the run's cell list, and this run has no "
            f"cell list (an array without cell names)")
    cells = [str(c) for c in cells]
    want = hashlib.sha256("\n".join(cells).encode("utf-8")).hexdigest()
    root = _toolchain.precomputed_root(cfg)
    found = []
    for candidate in sorted(root.glob("*/result.json")) if root.is_dir() else []:
        try:
            doc = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and doc.get("cells_sha256") == want:
            found.append(candidate)
    if not found:
        raise _StepSkipped(
            f"{name}: no precomputed output under {root} describes this run's {len(cells)} "
            f"cells (cells_sha256 {want})")
    if len(found) > 1:
        raise _StepSkipped(
            f"{name}: {len(found)} precomputed outputs describe this run's cells, so the choice "
            f"is ambiguous: {', '.join(p.parent.name for p in found)}")
    raw = found[0].read_bytes()
    g[f"{name}.tool_result_sha256"] = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw)
    if doc.get("ok") is False:
        raise _StepSkipped(f"{name} declined: {doc.get('reason') or 'no reason given'}")
    X = state.get("emb") if state.get("emb") is not None else state.get("X")
    input_dim = int(X.shape[1]) if getattr(X, "ndim", 0) == 2 else None
    _merge_v1(doc, found[0].parent, state, g, name, n_cells=len(cells), cell_list=cells,
              input_dim=input_dim, produces=cfg.get("produces") or [])
    caveats = ctx.get("caveats")
    if not isinstance(caveats, list):
        caveats = ctx["caveats"] = []
    caveats.append(f"step {name}: {PRECOMPUTED_CAVEAT}")


def _adapter_path(cfg: dict) -> "Path | None":
    """Where the adapter lives — BESIDE THE MANIFEST FIRST, then the bundled set.

    That order is what makes a third-party tool possible without forking the wheel. A user
    declaring their own tool puts `mytool.yaml` and `mytool_adapter.py` together in
    `$MANYRUNS_TOOL_DIR` or in a folder of `$MANYRUNS_TOOL_PATH` (`toolchain.manifest_dir`),
    exactly the way `$MANYRUNS_RECIPE_DIR` already works for recipes
    (`catalog.py` header: "the DIRECTORY is the registry — adding one is adding a file").
    Without this, adding a tool would mean editing an installed package, which is the one thing
    the config layer exists to avoid.

    An absolute path is honoured as given, for a tool whose adapter ships with the tool.

    A `mode: call` manifest has no adapter of its own: it gets the bundled call adapter, which
    makes the one call the manifest's `call` block describes.
    """
    from importlib import resources  # never __file__-relative — survives `uv tool install`

    from manyruns import toolchain as _tc

    if _tc.tool_mode(cfg) == "call":
        path = Path(str(resources.files("manyruns") / "adapters" / "call_adapter.py"))
        return path if path.is_file() else None
    entry = str(cfg["entrypoint"])
    candidates = [Path(entry)] if Path(entry).is_absolute() else [
        _tc._home(str(cfg["name"])) / entry,   # manifest_dir, or the base folder if undeclared
        Path(str(resources.files("manyruns") / "adapters" / entry)),
    ]
    return next((c for c in candidates if c.is_file()), None)


def run_external_step(name: str, params: dict, state: dict, g: dict, ctx: dict) -> None:
    """A `tool:` step: resolve the environment, hand it a file, read one back, record what ran."""
    resolved = _toolchain.resolve(name)
    if resolved["status"] == "precomputed":
        _run_precomputed(name, resolved, state, g, ctx)
        return
    if resolved["status"] not in ("ready", "overridden"):
        _refuse_unbuilt(name, resolved)
    cfg = resolved["cfg"]
    v1 = (cfg.get("output") or {}).get("schema") == _toolchain.RESULT_SCHEMA
    workdir = Path(ctx["out_dir"]) / f"tool-{name}"
    input_path = _input_path(cfg, state, ctx, workdir)
    shape = _input_shape(input_path, state) if v1 else None
    req_path, out_path, res_path = _write_request(
        workdir, cfg, params, ctx.get("seed"), input_path,
        v1={"n_cells": shape[0], "input_dim": shape[1], "time": _time_axis(state)}
        if v1 else None)

    adapter = _adapter_path(cfg)
    if adapter is None:
        raise _StepUnsupported(
            f"{name}: adapter {cfg.get('entrypoint') or 'call_adapter.py'!r} is neither beside "
            f"the manifest nor in this install's `manyruns/adapters/`")

    started = time.time()
    try:
        proc = subprocess.run([str(resolved["python"]), str(adapter), str(req_path)],
                              capture_output=True, text=True, timeout=int(ctx.get("tool_timeout")
                                                                          or 7200))
    except subprocess.TimeoutExpired:
        raise _StepSkipped(f"{name} exceeded its timeout") from None
    seconds = time.time() - started

    # PROVENANCE FIRST, so a FAILED tool run is still recorded as having been attempted with a
    # named environment. A record that only describes successes cannot answer "what did we try".
    g.update(_toolchain.provenance(name, resolved))
    g[f"{name}.tool_seconds"] = round(seconds, 2)
    if _toolchain.tool_mode(cfg) == "call":
        g[f"{name}.tool_call_target"] = cfg["call"]["target"]

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
        raise _StepSkipped(f"{name} exited {proc.returncode}: {tail[0][:200]}")
    result = json.loads(res_path.read_text()) if res_path.is_file() else {}
    if not result.get("ok"):
        raise _StepSkipped(f"{name} declined: {result.get('reason') or 'no reason given'}")
    if v1:
        _merge_v1(result, res_path.parent, state, g, name, n_cells=shape[0],
                  cell_list=_cell_list(state, ctx), input_dim=shape[1],
                  produces=cfg.get("produces") or [])
        return
    _merge(cfg, out_path, state, g, name)
    for key, value in (result.get("scalars") or {}).items():
        g[f"{name}.{key}"] = value
