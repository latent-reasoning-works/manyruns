# manyruns/harness/labeler.py
"""Interactive labeling for DR workflow results."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)

# THE topology vocabulary for DR embeddings, declared once: `label_workflow` validates
# against it and `interactive_label` offers it as the prompt's choices.
#
# A `LabelType = Literal[...]` alias restated the same five names on the next line. Grepped
# across manyruns, the learner, manylatents and shop, it had ZERO readers — nothing was
# annotated with it — so it could not fail a type check and could only ever drift away from
# the frozenset that is actually enforced. Removed rather than wired up: this file's `label`
# parameters are `str` and the runtime check is what rejects a bad label.
VALID_LABELS = frozenset({"cluster", "trajectory", "cycle", "surface", "other"})


def label_workflow(
    result: dict[str, Any],
    label: str,
    selected_metrics: list[str],
    labeler: str,
) -> dict[str, Any]:
    """Create a labeled entry dict from a run result dict.

    This is the programmatic API for labeling. For interactive CLI labeling,
    use ``interactive_label()`` which prompts the user.

    Args:
        result: Dict from run_workflow() with keys embeddings, scores, etc.
        label: Topology label (cluster, trajectory, cycle, surface, other).
        selected_metrics: List of metric names to include in G-vector.
        labeler: Name/ID of the person providing the label.

    Returns:
        Labeled entry dict ready for storage.

    Raises:
        ValueError: If label is invalid or metrics not found.
    """
    # Validate label
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: '{label}'. Must be one of: {sorted(VALID_LABELS)}")

    scores = result.get("scores", {})

    # Validate and extract selected metrics
    metrics_dict = {}
    for metric_name in selected_metrics:
        if metric_name not in scores:
            available = list(scores.keys()) if scores else []
            raise ValueError(
                f"Metric '{metric_name}' not found in result. Available metrics: {available}"
            )
        metrics_dict[metric_name] = scores[metric_name]

    return {
        "id": uuid4().hex,
        "workflow": result["workflow"],
        "dataset": result["dataset"],
        "label": label,
        "selected_metrics": metrics_dict,
        "embedding_shape": list(result["embeddings"].shape),
        "labeler": labeler,
        "labeled_at": datetime.now().isoformat(),
        "embedding_path": None,
    }


def interactive_label(
    result: dict[str, Any],
    labeler: str,
    skip_viz: bool = False,
    viz_mode: str = "auto",
) -> dict[str, Any]:
    """Interactively label a workflow result via CLI prompts.

    Shows the results, optionally visualizes the embedding, then prompts
    the user for label and metric selection.

    Args:
        result: Dict from run_workflow() with keys embeddings, scores, etc.
        labeler: Name/ID of the person providing the label.
        skip_viz: If True, skip visualization.
        viz_mode: Visualization mode (browser, terminal, wandb, file, auto).

    Returns:
        Labeled entry dict with user-provided label and metrics.
    """
    from rich.console import Console
    from rich.prompt import Prompt

    from manyruns.harness.display import display_results, visualize_embedding

    console = Console()

    # Display results
    display_results(result)

    # Visualize if requested
    if not skip_viz:
        console.print("\n[dim]Opening visualization...[/dim]")
        viz_path = visualize_embedding(result["embeddings"], mode=viz_mode)
        if viz_path:
            console.print(f"[dim]Plot: {viz_path}[/dim]")

    console.print()

    # Prompt for label
    label = Prompt.ask(
        "What topology do you see?",
        choices=list(VALID_LABELS),
        default="other",
    )

    # Prompt for metrics selection
    scores = result.get("scores", {})
    selected_metrics = []
    if scores:
        console.print("\n[bold]Select metrics for G-vector:[/bold]")
        for name, value in scores.items():
            include = Prompt.ask(
                f"  Include {name} ({value:.4f})?",
                choices=["y", "n"],
                default="y",
            )
            if include == "y":
                selected_metrics.append(name)
    else:
        console.print("[dim]No metrics to select[/dim]")

    # Create labeled entry
    labeled = label_workflow(
        result=result,
        label=label,
        selected_metrics=selected_metrics,
        labeler=labeler,
    )

    console.print(f"\n[green]✓[/green] Labeled as [bold]{label}[/bold]")

    return labeled
