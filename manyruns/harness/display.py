# manyruns/harness/display.py
"""Display DR workflow results and visualize embeddings."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

log = logging.getLogger(__name__)


def _format_workflow_step(step: dict[str, Any]) -> str:
    """Format a workflow step as Algorithm(n_components).

    Args:
        step: Workflow step dict with 'algorithm' and optional 'params'.

    Returns:
        Formatted string like "PCA(50)" or "UMAP(2)".
    """
    algo = step["algorithm"]
    params = step.get("params", {})
    n_comp = params.get("n_components")

    if n_comp is not None:
        return f"{algo}({n_comp})"
    return algo


def _format_workflow(workflow: list[dict[str, Any]]) -> str:
    """Format workflow as PCA(50) -> UMAP(2).

    Args:
        workflow: List of workflow step dicts.

    Returns:
        Arrow-separated string of formatted steps.
    """
    return " → ".join(_format_workflow_step(s) for s in workflow)


def format_results(result: dict[str, Any]) -> str:
    """Format a run result dict as plain text string.

    This function returns a string for testing purposes.
    Use display_results() for Rich-formatted terminal output.

    Args:
        result: Dict from run_workflow() with keys embeddings, scores,
            metadata, workflow, dataset.

    Returns:
        Formatted string with dataset, workflow, shape, metrics, and timing.
    """
    lines = []

    metadata = result.get("metadata", {})
    scores = result.get("scores", {})
    workflow = result["workflow"]
    embeddings = result["embeddings"]
    total_time = metadata["total_time"]
    steps_completed = metadata["steps_completed"]
    step_times = metadata.get("step_times", [])

    # Header
    lines.append("=" * 50)
    lines.append("DR Workflow Results")
    lines.append("=" * 50)
    lines.append("")

    # Basic info
    lines.append(f"Dataset: {result['dataset']}")
    lines.append(f"Workflow: {_format_workflow(workflow)}")
    lines.append(f"Shape: {embeddings.shape}")
    lines.append(f"Steps: {steps_completed} completed in {total_time:.2f}s")
    lines.append("")

    # Metrics
    if scores:
        lines.append("Metrics:")
        for name, value in scores.items():
            lines.append(f"  {name}: {value:.4f}")
    else:
        lines.append("Metrics: (none computed)")
    lines.append("")

    # Step timing
    if step_times:
        lines.append("Step Timing:")
        for i, (step, time) in enumerate(zip(workflow, step_times)):
            step_str = _format_workflow_step(step)
            lines.append(f"  {i + 1}. {step_str}: {time:.2f}s")

    return "\n".join(lines)


def display_results(result: dict[str, Any]) -> None:
    """Display run result dict with Rich formatting.

    Args:
        result: Dict from run_workflow() with keys embeddings, scores,
            metadata, workflow, dataset.
    """
    console = Console()

    metadata = result.get("metadata", {})
    scores = result.get("scores", {})
    workflow = result["workflow"]
    embeddings = result["embeddings"]
    total_time = metadata["total_time"]
    steps_completed = metadata["steps_completed"]
    step_times = metadata.get("step_times", [])

    # Header panel
    console.print(Panel("DR Workflow Results", style="bold blue"))

    # Basic info
    console.print(f"[bold]Dataset:[/bold] {result['dataset']}")
    console.print(f"[bold]Workflow:[/bold] {_format_workflow(workflow)}")
    console.print(f"[bold]Shape:[/bold] {embeddings.shape}")
    console.print(f"[bold]Time:[/bold] {steps_completed} steps in {total_time:.2f}s")
    console.print()

    # Metrics table
    if scores:
        table = Table(title="Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for name, value in scores.items():
            table.add_row(name, f"{value:.4f}")

        console.print(table)
    else:
        console.print("[dim]No metrics computed[/dim]")

    console.print()

    # Step timing
    if step_times and len(step_times) > 1:
        console.print("[bold]Step Timing:[/bold]")
        for i, (step, time) in enumerate(zip(workflow, step_times)):
            step_str = _format_workflow_step(step)
            console.print(f"  {i + 1}. {step_str}: {time:.2f}s")


def _detect_viz_mode() -> Literal["browser", "terminal", "wandb", "file"]:
    """Auto-detect best visualization mode based on environment.

    Returns:
        Detected mode string.
    """
    # In a WandB run - use wandb logging
    if os.environ.get("WANDB_RUN_ID"):
        return "wandb"

    # Not a TTY (piped, headless) - save to file
    if not sys.stdout.isatty():
        return "file"

    # SSH session without display - use terminal
    if os.environ.get("SSH_CLIENT") and not os.environ.get("DISPLAY"):
        return "terminal"

    # Default: open in browser
    return "browser"


def visualize_embedding(
    embedding: np.ndarray,
    labels: np.ndarray | None = None,
    mode: Literal["browser", "terminal", "wandb", "file", "auto"] = "auto",
    title: str = "Embedding",
    output_path: Path | str | None = None,
) -> str | None:
    """Visualize 2D embedding with configurable output mode.

    Args:
        embedding: 2D array of shape (n_samples, 2).
        labels: Optional labels for coloring points.
        mode: Visualization mode:
            - "browser": Save PNG and open in system viewer
            - "terminal": Inline sixel image in terminal
            - "wandb": Log to Weights & Biases
            - "file": Save PNG to output_path
            - "auto": Auto-detect based on environment
        title: Plot title.
        output_path: Output path for "file" mode (auto-generated if None).

    Returns:
        Path or URL to visualization, or None for terminal mode.
    """
    if mode == "auto":
        mode = _detect_viz_mode()

    log.debug(f"Visualizing embedding with mode={mode}")

    # Subsample if too many points
    max_points = 10000
    if len(embedding) > max_points:
        log.info(f"Subsampling {len(embedding)} points to {max_points}")
        indices = np.random.choice(len(embedding), max_points, replace=False)
        embedding = embedding[indices]
        if labels is not None:
            labels = labels[indices]

    if mode == "terminal":
        return _visualize_terminal(embedding, labels, title)
    elif mode == "wandb":
        return _visualize_wandb(embedding, labels, title)
    elif mode == "browser":
        return _visualize_browser(embedding, labels, title)
    elif mode == "file":
        return _visualize_file(embedding, labels, title, output_path)
    else:
        raise ValueError(f"Unknown visualization mode: {mode}")


def _visualize_terminal(embedding: np.ndarray, labels: np.ndarray | None, title: str) -> str | None:
    """Render scatter plot inline via sixel escape codes.

    Works in Windows Terminal, mlterm, foot, and other sixel-capable
    terminals. Falls back to text stats if libsixel is not available.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))

    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, alpha=0.6, s=10, cmap="tab10"
        )
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=10)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Dim 1", fontsize=8)
    ax.set_ylabel("Dim 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

    # Render to PNG bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Try sixel output
    try:
        from PIL import Image

        img = Image.open(buf)
        # Resize to fit laptop terminal (~500px wide)
        max_w = 500
        if img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)))

        _print_sixel(img)
        return None
    except ImportError:
        from manyruns.installation import ENVIRONMENT_REPAIR

        # No PIL — fall back to text
        print(f"\n{title}")
        print(f"  X range: [{embedding[:, 0].min():.2f}, {embedding[:, 0].max():.2f}]")
        print(f"  Y range: [{embedding[:, 1].min():.2f}, {embedding[:, 1].max():.2f}]")
        print(f"  Points: {len(embedding)}")
        print(f"  Pillow is unavailable for inline sixel images; {ENVIRONMENT_REPAIR}.")
        return None


def _print_sixel(img) -> None:
    """Encode a PIL Image as sixel and print to stdout."""
    import sys

    from PIL import Image

    # Convert to palette mode (sixel needs indexed color, max 256)
    img = img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    palette = img.getpalette()  # flat list: [r0, g0, b0, r1, g1, b1, ...]
    pixels = img.load()
    w, h = img.size

    parts = ["\x1bPq"]  # DCS (Device Control String) start

    # Set raster attributes: pan;pad;width;height
    parts.append(f'"1;1;{w};{h}')

    # Define colors. quantize() may return fewer than 256 colors (a simple plot
    # needs only a handful), so iterate the actual palette length — hardcoding
    # 256 IndexErrors on a short palette.
    n_colors = len(palette) // 3
    for i in range(n_colors):
        r, g, b = palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]
        # Sixel uses 0-100 percentage for RGB
        parts.append(f"#{i};2;{r * 100 // 255};{g * 100 // 255};{b * 100 // 255}")

    # Encode pixel data in bands of 6 rows
    for band_top in range(0, h, 6):
        band_bottom = min(band_top + 6, h)
        band_h = band_bottom - band_top

        # For each color used in this band
        colors_in_band = set()
        for y in range(band_top, band_bottom):
            for x in range(w):
                colors_in_band.add(pixels[x, y])

        for color in sorted(colors_in_band):
            parts.append(f"#{color}")
            sixel_row = []
            for x in range(w):
                bits = 0
                for dy in range(band_h):
                    if pixels[x, band_top + dy] == color:
                        bits |= 1 << dy
                sixel_row.append(chr(63 + bits))
            parts.append("".join(sixel_row))
            parts.append("$")  # carriage return (same band)

        parts.append("-")  # newline (next band)

    parts.append("\x1b\\")  # ST (String Terminator)

    sys.stdout.write("".join(parts))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _visualize_wandb(embedding: np.ndarray, labels: np.ndarray | None, title: str) -> str | None:
    """Log plot to Weights & Biases."""
    try:
        import wandb

        if wandb.run is None:
            log.warning("No active WandB run, falling back to file mode")
            return _visualize_file(embedding, labels, title, None)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        if labels is not None:
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=labels, alpha=0.5, s=10, cmap="tab10"
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=10)

        ax.set_title(title)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

        wandb.log({f"harness/{title}": wandb.Image(fig)})
        plt.close(fig)

        return wandb.run.url
    except ImportError:
        log.warning("wandb not installed, falling back to file mode")
        return _visualize_file(embedding, labels, title, None)


def _visualize_browser(embedding: np.ndarray, labels: np.ndarray | None, title: str) -> str:
    """Save plot and open in system browser."""
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, alpha=0.6, s=15, cmap="tab10"
        )
        plt.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=15)

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.3)

    # Save to temp file
    path = Path(tempfile.mktemp(suffix=".png", prefix="embedding_"))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Open in browser (non-blocking)
    webbrowser.open(f"file://{path}")
    log.info(f"Opened plot in browser: {path}")

    return str(path)


def _visualize_file(
    embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    output_path: Path | str | None,
) -> str:
    """Save plot to file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, alpha=0.6, s=15, cmap="tab10"
        )
        plt.colorbar(scatter, ax=ax, label="Label")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=15)

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.3)

    # Determine output path
    if output_path is None:
        from uuid import uuid4

        output_path = Path(f"embedding_{uuid4().hex[:8]}.png")
    else:
        output_path = Path(output_path)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"Saved plot to: {output_path}")
    return str(output_path)
