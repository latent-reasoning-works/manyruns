# manyruns/harness/configs.py
"""Workflow configuration loading and preset definitions.

Supports three config sources:
1. Built-in presets (high-signal workflows)
2. Custom YAML config files
3. manyLatents experiment configs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Built-in workflow presets
# These are high-signal workflows identified through experimentation
PRESETS: dict[str, dict[str, Any]] = {
    "pca_umap": {
        "name": "PCA → UMAP",
        "description": "Standard 2-stage DR: linear reduction then manifold",
        "workflow": [
            {"algorithm": "PCA", "params": {"n_components": 50}},
            {"algorithm": "UMAP", "params": {"n_components": 2}},
        ],
        "metrics": ["participation_ratio", "trustworthiness"],
    },
    "pca_tsne": {
        "name": "PCA → t-SNE",
        "description": "PCA preprocessing with t-SNE visualization",
        "workflow": [
            {"algorithm": "PCA", "params": {"n_components": 50}},
            {"algorithm": "TSNE", "params": {"n_components": 2}},
        ],
        "metrics": ["participation_ratio", "trustworthiness"],
    },
    "phate": {
        "name": "PHATE",
        "description": "Potential of Heat-diffusion for Affinity-based Transition Embedding",
        "workflow": [
            {"algorithm": "PHATE", "params": {"n_components": 2}},
        ],
        "metrics": ["participation_ratio"],
    },
    "pca_only": {
        "name": "PCA",
        "description": "Simple linear dimensionality reduction",
        "workflow": [
            {"algorithm": "PCA", "params": {"n_components": 10}},
        ],
        "metrics": ["participation_ratio"],
    },
    "umap_2d": {
        "name": "UMAP (2D)",
        "description": "Direct UMAP to 2 dimensions",
        "workflow": [
            {"algorithm": "UMAP", "params": {"n_components": 2}},
        ],
        "metrics": ["participation_ratio", "trustworthiness"],
    },
    "deep_pca_umap": {
        "name": "Deep PCA → UMAP",
        "description": "More aggressive PCA before UMAP",
        "workflow": [
            {"algorithm": "PCA", "params": {"n_components": 100}},
            {"algorithm": "UMAP", "params": {"n_components": 2, "n_neighbors": 30}},
        ],
        "metrics": ["participation_ratio", "trustworthiness", "continuity"],
    },
}


def list_presets() -> list[dict[str, str]]:
    """List available preset workflows.

    Returns:
        List of dicts with 'name', 'key', and 'description'.
    """
    return [
        {"key": key, "name": preset["name"], "description": preset["description"]}
        for key, preset in PRESETS.items()
    ]


def load_preset(name: str) -> dict[str, Any]:
    """Load a preset workflow by name.

    Args:
        name: Preset key (e.g., 'pca_umap').

    Returns:
        Dict with 'workflow' and 'metrics' keys.

    Raises:
        KeyError: If preset not found.
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")

    preset = PRESETS[name]
    return {
        "workflow": preset["workflow"],
        "metrics": preset.get("metrics", []),
    }


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load workflow config from YAML file.

    Expected format:
    ```yaml
    name: My Workflow
    workflow:
      - algorithm: PCA
        params:
          n_components: 50
      - algorithm: UMAP
        params:
          n_components: 2
    metrics:
      - participation_ratio
      - trustworthiness
    ```

    Args:
        path: Path to YAML config file.

    Returns:
        Dict with 'workflow' and 'metrics' keys.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if "workflow" not in config:
        raise ValueError(f"Config must have 'workflow' key: {path}")

    return {
        "workflow": config["workflow"],
        "metrics": config.get("metrics", []),
        "name": config.get("name", path.stem),
    }


def load_manylatents_config(path: Path | str) -> dict[str, Any]:
    """Load workflow from manyLatents experiment config.

    Parses manyLatents Hydra config format and extracts the workflow.

    Args:
        path: Path to manyLatents config YAML.

    Returns:
        Dict with 'workflow' and 'metrics' keys.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manyLatents config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # manyLatents config structure:
    # algorithms:
    #   latent:
    #     _target_: manylatents.algorithms.latent.pca.PCAModule
    #     n_components: 50

    workflow = []

    if "algorithms" in config and "latent" in config["algorithms"]:
        latent_config = config["algorithms"]["latent"]

        # Could be a single algorithm or a list
        if isinstance(latent_config, dict):
            latent_config = [latent_config]

        for alg in latent_config:
            target = alg.get("_target_", "")
            # Extract algorithm name from target
            # manylatents.algorithms.latent.pca.PCAModule -> PCA
            algo_name = target.split(".")[-1].replace("Module", "")

            # Extract params (everything except _target_)
            params = {k: v for k, v in alg.items() if not k.startswith("_")}

            workflow.append({"algorithm": algo_name, "params": params})

    if not workflow:
        raise ValueError(f"Could not extract workflow from manyLatents config: {path}")

    # Try to extract metrics from config
    metrics = []
    if "metrics" in config:
        metrics_config = config["metrics"]
        if isinstance(metrics_config, list):
            metrics = metrics_config
        elif isinstance(metrics_config, dict):
            metrics = list(metrics_config.keys())

    return {
        "workflow": workflow,
        "metrics": metrics,
        "name": config.get("name", path.stem),
    }


def resolve_config(
    preset: str | None = None,
    config_file: Path | str | None = None,
    manylatents_config: Path | str | None = None,
    algorithm: str | None = None,
    n_components: int | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve workflow config from various sources.

    Priority order:
    1. preset (named preset)
    2. config_file (custom YAML)
    3. manylatents_config (manyLatents experiment config)
    4. algorithm + n_components (CLI args, legacy)

    Args:
        preset: Preset name.
        config_file: Path to custom YAML config.
        manylatents_config: Path to manyLatents config.
        algorithm: DR algorithm (legacy CLI).
        n_components: Number of components (legacy CLI).
        metrics: Metrics to compute (can override config).

    Returns:
        Dict with 'workflow' and 'metrics' keys.

    Raises:
        ValueError: If no valid config source provided.
    """
    if preset:
        log.debug(f"Loading preset: {preset}")
        result = load_preset(preset)
    elif config_file:
        log.debug(f"Loading config file: {config_file}")
        result = load_yaml_config(config_file)
    elif manylatents_config:
        log.debug(f"Loading manyLatents config: {manylatents_config}")
        result = load_manylatents_config(manylatents_config)
    elif algorithm and n_components is not None:
        log.debug(f"Using CLI args: {algorithm}({n_components})")
        result = {
            "workflow": [{"algorithm": algorithm, "params": {"n_components": n_components}}],
            "metrics": [],
        }
    else:
        raise ValueError(
            "Must specify one of: --preset, --config, --manylatents-config, "
            "or --algorithm with --n-components"
        )

    # Override metrics if provided
    if metrics:
        result["metrics"] = list(metrics)

    return result
