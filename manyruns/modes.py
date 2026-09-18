"""Mode routing — one entry (`run(mode=…)`) dispatches the train / eval / infer stages.

This is the "mode with routing" seam. Three concerns are kept in three layers:

  - **which mode** to invoke (train / eval / infer, and baseline vs discovery) — the
    *policy/router* (LLM/RL), one level up. NOT here.
  - **what stage** runs — the *mode dispatch*. THIS module. `run(mode=…)` → a handler.
  - **where** it runs (cpu / mps / cuda / slurm) — the *substrate dispatch* (the dispatcher
    skill). It hands us a `device` (the `device_override`); we thread it to the delegated
    compute. We do NOT choose the device ourselves.

Compute stays delegated (manyruns owns the routing, not the loops):
  - **infer** → the serving / recipe path (`app.run_explorations` → a `ModelServer`).
  - **eval**  → geometry metrics over an infer result (the g-vector today; retention next).
  - **train** → NOBODY. This used to read "the learner: local via `manyruns.train.run`, cluster
    via shop's SLURM launcher". Both are deleted: manyruns holds no track to the learner
    (the environment-contract spec, §3.5). The stage PLANS — it records the ask and returns
    the store path, which the learner reads on its own. See `_train`.

The stages are connected by the **versioned artifact**, not by shared control flow: `train`
writes `<model>@vN`; `eval`/`infer` load it. That glue is the model-version registry
(serving-router.md) + provenance tagging — referenced here via `model_version`.
"""
from __future__ import annotations

from typing import Optional

MODES = ("infer", "eval", "train")


def run(mode: str, request: Optional[dict] = None, *, device: Optional[str] = None) -> dict:
    """Dispatch a stage. `device` is the substrate dispatcher's `device_override` (or None).

    `request` carries the stage inputs (a `recipe` for infer, a prior `result` for eval, a
    `core_snapshot`/`model_version` for train). Returns a mode-tagged result dict."""
    request = request or {}
    handlers = {"infer": _infer, "eval": _eval, "train": _train}
    if mode not in handlers:
        raise ValueError(f"unknown mode {mode!r}; choose from {sorted(handlers)}")
    return handlers[mode](request, device=device)


# ── infer: run a recipe → embedding + g-vector (the serving path) ────────────
def _infer(req: dict, *, device: Optional[str]) -> dict:
    from manyruns.app import load_recipe, run_explorations

    recipe = req.get("recipe")
    model_version = req.get("model_version")

    # Version routing — pull a registered model version (concurrent-versions serving):
    #   kind="recipe"  → this version's recipe drives the loop;
    #   kind="weights" → apply the version's trained weights to the input (stack-gated).
    if req.get("version"):
        from manyruns import registry

        v = registry.resolve(req["model"], req["version"], root=req.get("registry_root"))
        model_version = f"{v.model}@{v.version}"
        if v.kind == "weights":
            import numpy as np

            emb = np.asarray(registry.apply_weights(v, req["array"]))
            return {"mode": "infer", "model_version": model_version, "kind": "weights",
                    "device": device or "cpu", "embedding": emb,
                    "final_dim": int(emb.shape[1]) if emb.ndim == 2 else 0}
        recipe = load_recipe(v.spec["recipe"])   # recipe-kind: pull the version's recipe

    if not recipe:
        raise ValueError("infer needs a 'recipe' or a registered 'version' in the request")
    res = run_explorations(
        req.get("data"),
        req.get("modality", "scrna"),
        recipe=recipe,
        engine=req.get("engine", "mock"),
        dataset=req.get("dataset"),
        dataset_name=req.get("dataset_name"),
        declared_shape=req.get("declared_shape"),
        color=req.get("color"), obs_names=req.get("obs_names"), label_key=req.get("label_key"),
        color_by=req.get("color_by"),
        array=req.get("array"),
        out_dir=req.get("out_dir", "."),
        seed=req.get("seed", 42),
        fast_dev_run=req.get("fast_dev_run", True),
        data_kwargs=req.get("data_kwargs"),
        labels=req.get("labels"),
        metrics=req.get("metrics"),   # the declared metric suite → manylatents' registry
        # "time" | "condition" | "group" — so a trajectory step reads only a declared time
        # axis, and `separation`/`composition` read only a declared condition axis.
        label_kind=req.get("label_kind"),
        # The GENE AXIS the caller already read off the file. `run_explorations` re-reads the
        # path only when it was handed no array, and the CLI always hands it one — so without
        # these two the axis is loaded by `app._read_inputs` and dropped one frame later.
        counts=req.get("counts"),
        genes=req.get("genes"),
        layers=req.get("layers"),   # declared-only; empty unless a step asked for one
        device=device,   # ← actually threaded to compute; run_explorations resolves it (fp64/MPS)
    )
    # `res` carries the RESOLVED device + rationale (not the echoed request) — spread it last so
    # "device" is the one compute actually used, closing the cosmetic-device gap.
    return {"mode": "infer", "model_version": model_version, **res}


# ── eval: geometry readout over an infer result ──────────────────────────────
def _eval(req: dict, *, device: Optional[str]) -> dict:
    res = req.get("result") or _infer(req, device=device)
    return {
        "mode": "eval",
        "device": res.get("device", device),   # the device the evaluated run actually used
        "model_version": req.get("model_version") or res.get("model_version"),
        "recipe": res.get("recipe"),
        "metrics": res.get("g_vector") or {},
        "note": "eval reports the geometry g-vector today; retention R vs the ambient space "
        "is the next readout.",
    }


# ── train: delegated to the engine (local) or the cluster (shop/SLURM) ───────
def _train(req: dict, *, device: Optional[str]) -> dict:
    """The train stage: manyruns PLANS it and never runs it.

    THE HARNESS WORKFLOW STOPS AT STORE — `run -> label -> store`. Training is the learner's, and
    the learner observes manyruns rather than being called by it (the environment-contract spec,
    §3.5): manyruns has no track to it, so there is nothing here to delegate TO.

    This used to import `manyruns.train` and, with `execute`, call into the engine in-process.
    Both are gone. What survives is the STAGE — the product still has a train step in its
    vocabulary, still records that it was asked for, and still says where the work is picked up
    from — because deleting the stage would make the workflow lie about its own shape.

    Returns the store the learner reads. It is the only handoff, and it is a path rather than a
    call, which is the whole point.
    """
    return {
        "mode": "train",
        "device": device or "cpu",
        "core_snapshot": req.get("core_snapshot"),
        "model_version_out": req.get("model_version", "<model>@vN"),
        "delegate": "not manyruns's to run — the learner reads the store and drives manyruns "
                    "through its CLI; see docs/environment-contract.md §3.5",
        "store": str(req.get("store") or "outputs"),
        "status": "planned — manyruns records the ask and stops at the store",
    }
