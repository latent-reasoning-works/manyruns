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

#: `fact -> (state slot, how it arrives)`. ONE entry, and deliberately not a generic mapping
#: language: a second entry is what would tell us what the general shape is, and inventing it
#: now is the op-registry mistake CLAUDE.md names. `velocity` arrives as an `obsm` matrix; its
#: per-cell posterior summaries ride alongside in `obs` and land in a companion slot.
FACT_SLOTS: dict[str, tuple[str, str]] = {
    "velocity": ("velocity", "obsm"),
}

#: Companion slot for a fact's per-cell uncertainty, when the manifest declares `output.obs`.
UNCERTAINTY_SLOT = {"velocity": "velocity_uncertainty"}


def _refuse_unbuilt(name: str, resolved: dict) -> None:
    fix = resolved.get("fix")
    raise _StepUnsupported(
        f"{resolved.get('reason')}"
        + (f" — fix: {fix}" if fix else "")
    )


def _write_request(workdir: Path, cfg: dict, params: dict, seed: Any,
                   input_path: Path) -> tuple[Path, Path, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    req_path, out_path, res_path = (workdir / "request.json", workdir / "output.h5ad",
                                    workdir / "result.json")
    merged = dict(cfg.get("params") or {})
    merged.update(params or {})
    req_path.write_text(json.dumps({
        "tool": cfg["name"], "version": cfg["version"],
        "params": merged, "seed": int(seed) if seed is not None else 0,
        "input": str(input_path), "output": str(out_path), "result": str(res_path),
        "workdir": str(workdir),
    }, indent=2))
    return req_path, out_path, res_path


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


def _adapter_path(cfg: dict) -> "Path | None":
    """Where the adapter lives — BESIDE THE MANIFEST FIRST, then the bundled set.

    That order is what makes a third-party tool possible without forking the wheel. A user
    declaring their own tool puts `mytool.yaml` and `mytool_adapter.py` together in
    `$MANYRUNS_TOOL_DIR`, exactly the way `$MANYRUNS_RECIPE_DIR` already works for recipes
    (`catalog.py` header: "the DIRECTORY is the registry — adding one is adding a file").
    Without this, adding a tool would mean editing an installed package, which is the one thing
    the config layer exists to avoid.

    An absolute path is honoured as given, for a tool whose adapter ships with the tool.
    """
    from importlib import resources  # never __file__-relative — survives `uv tool install`

    from manyruns import toolchain as _tc

    entry = str(cfg["entrypoint"])
    candidates = [Path(entry)] if Path(entry).is_absolute() else [
        _tc.tool_dir() / entry,
        Path(str(resources.files("manyruns") / "adapters" / entry)),
    ]
    return next((c for c in candidates if c.is_file()), None)


def run_external_step(name: str, params: dict, state: dict, g: dict, ctx: dict) -> None:
    """A `tool:` step: resolve the environment, hand it a file, read one back, record what ran."""
    resolved = _toolchain.resolve(name)
    if resolved["status"] not in ("ready", "overridden"):
        _refuse_unbuilt(name, resolved)
    cfg = resolved["cfg"]
    workdir = Path(ctx["out_dir"]) / f"tool-{name}"
    input_path = _input_path(cfg, state, ctx, workdir)
    req_path, out_path, res_path = _write_request(
        workdir, cfg, params, ctx.get("seed"), input_path)

    adapter = _adapter_path(cfg)
    if adapter is None:
        raise _StepUnsupported(
            f"{name}: adapter {cfg['entrypoint']!r} is neither beside the manifest nor in this "
            f"install's `manyruns/adapters/`")

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

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
        raise _StepSkipped(f"{name} exited {proc.returncode}: {tail[0][:200]}")
    result = json.loads(res_path.read_text()) if res_path.is_file() else {}
    if not result.get("ok"):
        raise _StepSkipped(f"{name} declined: {result.get('reason') or 'no reason given'}")
    _merge(cfg, out_path, state, g, name)
    for key, value in (result.get("scalars") or {}).items():
        g[f"{name}.{key}"] = value
