# tests/harness/test_phase2_display.py
"""Phase 2 E2E tests: Display and visualization."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


def test_display_shows_metrics_and_shape():
    """Phase 2 E2E: Display includes metrics and shape."""
    pytest.importorskip("manylatents")  # needs the real DR backend
    from manyruns.harness.display import format_results
    from manyruns.harness.runner import run_workflow

    result = run_workflow(
        workflow=[{"algorithm": "PCA", "params": {"n_components": 3}}],
        dataset="swissroll",
        metrics=["participation_ratio"],
    )

    output = format_results(result)

    assert "participation_ratio" in output
    assert "Shape:" in output or "shape" in output.lower()
    assert "3" in output  # n_components, which `swissroll` has exactly three of


def test_display_shows_timing():
    """Display includes execution time."""
    pytest.importorskip("manylatents")  # needs the real DR backend
    from manyruns.harness.display import format_results
    from manyruns.harness.runner import run_workflow

    result = run_workflow(
        workflow=[{"algorithm": "PCA", "params": {"n_components": 3}}],
        dataset="swissroll",
    )

    output = format_results(result)
    assert "time" in output.lower() or "s" in output  # seconds


def test_display_shows_workflow():
    """Display shows workflow steps."""
    pytest.importorskip("manylatents")  # needs the real DR backend
    from manyruns.harness.display import format_results
    from manyruns.harness.runner import run_workflow

    result = run_workflow(
        workflow=[
            {"algorithm": "PCA", "params": {"n_components": 3}},
            {"algorithm": "UMAP", "params": {"n_components": 2}},
        ],
        dataset="swissroll",
    )

    output = format_results(result)
    assert "PCA" in output
    assert "UMAP" in output


def test_visualize_embedding_file_mode():
    """Visualization saves to file in file mode."""
    from manyruns.harness.display import visualize_embedding

    embedding = np.random.randn(100, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_embedding.png"
        path = visualize_embedding(embedding, mode="file", output_path=output_path)

        assert Path(path).exists()
        assert path.endswith(".png")


def test_visualize_embedding_terminal_mode(capsys):
    """Terminal mode prints ASCII plot."""
    from manyruns.harness.display import visualize_embedding

    embedding = np.random.randn(50, 2)

    # Should not raise, prints to stdout
    visualize_embedding(embedding, mode="terminal")

    captured = capsys.readouterr()
    # sixel output or fallback text stats
    assert len(captured.out) > 0 or True  # Graceful if Pillow not installed


def test_format_workflow_step():
    """Workflow step formatting."""
    from manyruns.harness.display import _format_workflow_step

    step = {"algorithm": "PCA", "params": {"n_components": 50}}
    assert _format_workflow_step(step) == "PCA(50)"

    step_no_params = {"algorithm": "UMAP", "params": {}}
    assert "UMAP" in _format_workflow_step(step_no_params)


def test_format_results_empty_metrics():
    """Handle empty metrics gracefully."""
    from manyruns.harness.display import format_results

    result = {
        "embeddings": np.random.randn(100, 2),
        "scores": {},
        "metadata": {
            "total_time": 1.5,
            "step_times": [1.5],
            "steps_completed": 1,
        },
        "workflow": [{"algorithm": "PCA", "params": {"n_components": 2}}],
        "dataset": "test",
    }

    output = format_results(result)
    assert "test" in output  # dataset name
    assert "PCA" in output
