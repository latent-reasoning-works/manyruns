# manyruns/harness/runner.py
"""Execute DR workflows via manyLatents in-memory API.

LEGACY — the shop-lifted DR path. It uses its OWN step schema (`step["algorithm"]`) and
returns `{embeddings, scores, metadata}`, which is NOT the canonical recipe/g-vector shape.
The canonical step loop is `manyruns.pipeline.runner` (`_run_steps` → one g-vector schema
across every engine, trace-proven end-to-end). Do NOT build the new orchestration harness on
this module; build it on `pipeline.runner`. This exists only to keep the older
`python -m manyruns.harness` CLI (run/label/rlhf) working until it is migrated onto the
recipe layer. A cross-repo audit flagged this as the divergent third runner (the others:
`pipeline.runner`, and the learner's untested expert-workflow runner).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _suppress_noise() -> None:
    """Kill all handlers that manylatents/lightning/numba install on the root logger."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    for name in (
        "manylatents",
        "lightning",
        "lightning.pytorch",
        "lightning_utilities",
        "lightning.fabric",
        "numba",
        "manyruns.harness",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)


def _build_algorithm_config(step: dict[str, Any]) -> dict[str, Any]:
    """Build manyLatents algorithm config from workflow step.

    Follows manyAgents pattern: algorithm name -> {algo_name.upper()}Module
    """
    algo_name = step["algorithm"].lower()
    # Pattern from manyAgents: manylatents.algorithms.latent.{name}.{NAME}Module
    target = f"manylatents.algorithms.latent.{algo_name}.{algo_name.upper()}Module"

    config = {"_target_": target}
    if "params" in step:
        config.update(step["params"])
    return config


def _flatten_scores(scores: dict[str, Any]) -> dict[str, float]:
    """Flatten raw manyLatents scores to simple {name: float} dict.

    Handles group prefixes (e.g. "embedding.participation_ratio"),
    (scalar, per_sample_array) tuples, numpy arrays, and tensors.
    """
    flat: dict[str, float] = {}
    for key, value in scores.items():
        metric_name = key.split(".")[-1] if "." in key else key

        if isinstance(value, tuple) and len(value) == 2:
            scalar = value[0]
            if hasattr(scalar, "item"):
                flat[metric_name] = float(scalar.item())
            else:
                flat[metric_name] = float(scalar)
        elif isinstance(value, (int, float)):
            flat[metric_name] = float(value)
        elif hasattr(value, "mean"):
            flat[metric_name] = float(value.mean())
        elif hasattr(value, "item"):
            flat[metric_name] = float(value.item())
        elif hasattr(value, "__float__"):
            flat[metric_name] = float(value)

    return flat


def run_workflow(
    workflow: list[dict[str, Any]],
    dataset: str,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a DR workflow via manyLatents in-memory API.

    Args:
        workflow: List of workflow steps. Each step is a dict with:
            - algorithm: Name of the algorithm (e.g., "PCA", "UMAP")
            - params: Dict of algorithm parameters (e.g., {"n_components": 2})
        dataset: Name of the dataset (e.g., "swissroll", "embryoid")
        metrics: Optional list of metrics to compute (e.g., ["participation_ratio"])

    Returns:
        Dict with keys: embeddings, scores, metadata, workflow, dataset.

    Example:
        >>> result = run_workflow(
        ...     workflow=[
        ...         {"algorithm": "PCA", "params": {"n_components": 50}},
        ...         {"algorithm": "UMAP", "params": {"n_components": 2}},
        ...     ],
        ...     dataset="swissroll",
        ... )
        >>> result["embeddings"].shape
        (1000, 2)
    """
    try:
        from manylatents.api import run as ml_run
    except ImportError as e:
        raise ImportError(
            "manylatents is required for harness functionality. "
            "Install with: uv sync --extra harness"
        ) from e

    # Suppress manylatents/lightning internal logging noise
    import warnings

    warnings.filterwarnings("ignore", message=".*tensorboardX.*")

    start_time = time.perf_counter()
    step_times = []
    current_data = None
    steps_completed = 0
    result = None

    # Execute each step in the workflow
    for i, step in enumerate(workflow):
        step_start = time.perf_counter()

        # Note: GlobalHydra clearing is handled inside manylatents.api.run()

        # Build algorithm config
        algo_config = _build_algorithm_config(step)

        # Prepare kwargs for manyLatents API
        kwargs: dict[str, Any] = {
            "algorithms": {"latent": algo_config},
        }

        # First step loads data from dataset, subsequent steps use previous embedding
        if i == 0:
            kwargs["data"] = dataset

        # Add metrics on the last step only. manyLatents resolves metric names
        # through its Python registry now (the configs/metrics/embedding/*.yaml
        # files were removed when api.run went Hydra-free), so pass names directly.
        is_last_step = i == len(workflow) - 1
        if metrics and is_last_step:
            kwargs["metrics"] = [m.lower() for m in metrics]

        # Suppress logging before each ml_run call — manylatents reconfigures
        # handlers internally, so we must clear them every time
        _suppress_noise()

        # Run the step
        if current_data is None:
            result = ml_run(**kwargs)
        else:
            result = ml_run(input_data=current_data, **kwargs)

        _suppress_noise()

        # Extract embedding and convert to numpy if needed
        embedding = result["embeddings"]
        if hasattr(embedding, "detach"):
            # torch.Tensor -> numpy
            current_data = embedding.detach().cpu().numpy()
        elif hasattr(embedding, "numpy"):
            current_data = embedding.numpy()
        else:
            current_data = np.asarray(embedding)

        steps_completed += 1
        step_times.append(time.perf_counter() - step_start)
        log.debug(f"Step {i} complete: embedding shape = {current_data.shape}")

    total_time = time.perf_counter() - start_time

    # Extract metrics from result
    result_scores = {}
    if result and "scores" in result and result["scores"]:
        result_scores = _flatten_scores(result["scores"])

    return {
        "embeddings": current_data,
        "scores": result_scores,
        "metadata": {
            "total_time": total_time,
            "step_times": step_times,
            "steps_completed": steps_completed,
        },
        "workflow": workflow,
        "dataset": dataset,
    }


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a trajectory from disk and return the last step as a run result dict.

    Reads a trajectory manifest written by manyLatents' SaveTrajectory callback,
    and returns the final step in the same shape as ``run_workflow()`` output.

    Args:
        run_dir: Directory containing ``trajectory.json`` and step NPY files.

    Returns:
        Dict with keys: embeddings, scores, metadata, workflow, dataset.
    """
    from pathlib import Path as _Path

    from manylatents.callbacks.embedding.save_trajectory import load_trajectory

    run_dir = _Path(run_dir)
    steps = load_trajectory(run_dir)
    last = steps[-1]

    # Reconstruct workflow from step metadata
    workflow = []
    for s in steps:
        meta = s.get("metadata", {})
        workflow.append(
            {
                "algorithm": meta.get("algorithm", "unknown"),
                "params": {},
            }
        )

    step_times = [s.get("metadata", {}).get("step_time") for s in steps]

    return {
        "embeddings": last["embeddings"],
        "scores": last.get("scores", {}),
        "metadata": {
            "total_time": sum(t for t in step_times if t is not None),
            "step_times": step_times,
            "steps_completed": len(steps),
        },
        "workflow": workflow,
        "dataset": run_dir.name,
    }
