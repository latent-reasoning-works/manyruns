# manyruns/harness/agent_tools.py
"""Tools the agentic chat loop can call.

These are thin wrappers that expose manyruns's compute to the manyAgents loop. The
loop lives in manyAgents; the tool body delegates to manyruns's existing DR path
(manyLatents). manyruns contributes the tool *definition*, not the orchestration.
"""

from __future__ import annotations

from typing import Any

# Preset names that resolve to DR workflows (see manyruns.harness.configs.PRESETS).
_PRESETS = ["pca_umap", "pca_tsne", "phate", "pca_only", "umap_2d", "deep_pca_umap"]


def _run_dr_workflow(preset: str = "umap_2d", dataset: str = "swissroll") -> str:
    """Run a DR workflow via the real manyLatents backend and summarize it."""
    from manyruns.harness.configs import resolve_config
    from manyruns.harness.interface import Request
    from manyruns.harness.interface import run as harness_run

    try:
        cfg = resolve_config(preset=preset, metrics=None)
    except (KeyError, ValueError, FileNotFoundError) as e:
        return f"Error: bad preset {preset!r} ({e}). Available: {', '.join(_PRESETS)}"

    request = Request(dataset=dataset, workflow=cfg["workflow"], metrics=cfg["metrics"] or [])
    try:
        resp = harness_run(request, backend="dr")
    except ImportError:
        return (
            "Error: manyLatents not installed — cannot run the DR backend "
            "(uv sync --extra harness)."
        )
    except Exception as e:
        return f"Error: DR run failed: {e}"

    steps = " → ".join(
        f"{s['algorithm']}({s.get('params', {}).get('n_components', '?')})" for s in resp.workflow
    )
    shape = getattr(resp.embeddings, "shape", None)
    scores = ", ".join(f"{k}={v:.4f}" for k, v in (resp.scores or {}).items()) or "none"
    return f"Ran {steps} on '{dataset}'. Embedding shape: {shape}. Metrics: {scores}."


def make_dr_tool() -> Any:
    """Build the run_dr_workflow Tool (imports manyAgents lazily)."""
    from manyagents.tools import Tool

    return Tool(
        name="run_dr_workflow",
        description=(
            "Run a dimensionality-reduction (DR) workflow on a dataset and return "
            "the embedding shape and geometric metric scores (trustworthiness, "
            "participation_ratio, ...). Use this when the user asks to embed, "
            "reduce, visualize, or compute the geometry/structure of a dataset."
        ),
        parameters={
            "type": "object",
            "properties": {
                "preset": {
                    "type": "string",
                    "enum": _PRESETS,
                    "description": "DR workflow preset (e.g. umap_2d, pca_umap, phate).",
                },
                "dataset": {
                    "type": "string",
                    "description": "Dataset name, e.g. swissroll or embryoid.",
                },
            },
            "required": ["preset", "dataset"],
        },
        run=_run_dr_workflow,
    )
