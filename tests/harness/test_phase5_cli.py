# tests/harness/test_phase5_cli.py
"""Phase 5 E2E tests: CLI entry point."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from manyruns.harness.labeler import label_workflow
from manyruns.harness.storage import save_labeled


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner."""
    return CliRunner()


def _make_result(**overrides) -> dict:
    """Build a sample run result dict with optional overrides."""
    base = {
        "embeddings": np.random.randn(100, 5),
        "scores": {"pr": 0.5},
        "metadata": {"total_time": 1.0, "step_times": [1.0], "steps_completed": 1},
        "workflow": [{"algorithm": "PCA", "params": {"n_components": 5}}],
        "dataset": "swissroll",
    }
    base.update(overrides)
    return base


def test_cli_help(cli_runner):
    """CLI shows help."""
    from manyruns.harness.__main__ import cli

    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "harness" in result.output.lower() or "workflow" in result.output.lower()


def test_cli_run_basic(cli_runner):
    """CLI run command executes workflow."""
    from manyruns.harness.__main__ import cli

    result = cli_runner.invoke(
        cli,
        [
            "run",
            "--algorithm",
            "PCA",
            "--n-components",
            "10",
            "--dataset",
            "swissroll",
        ],
    )

    assert result.exit_code == 0
    assert "shape" in result.output.lower() or "Shape" in result.output


def test_cli_list_shows_entries(cli_runner):
    """CLI list command shows dataset contents."""
    from manyruns.harness.__main__ import cli

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        # Create test dataset
        result = _make_result()
        labeled = label_workflow(result, label="cluster", selected_metrics=["pr"], labeler="test")
        save_labeled(path, labeled, embedding=np.random.randn(100, 5))

        # Run list command
        cli_result = cli_runner.invoke(cli, ["list", "--dataset", str(path)])

        assert cli_result.exit_code == 0
        assert "cluster" in cli_result.output or "1" in cli_result.output  # entry count or label


def test_cli_list_empty_dataset(cli_runner):
    """CLI list handles empty dataset."""
    from manyruns.harness.__main__ import cli

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.json"
        # Write an empty dataset JSON directly
        import json
        from datetime import datetime

        path.write_text(
            json.dumps(
                {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "entries": [],
                }
            )
        )

        result = cli_runner.invoke(cli, ["list", "--dataset", str(path)])

        assert result.exit_code == 0
        assert "0" in result.output or "empty" in result.output.lower()


def test_cli_run_with_metrics(cli_runner):
    """CLI run can include metrics."""
    from manyruns.harness.__main__ import cli

    result = cli_runner.invoke(
        cli,
        [
            "run",
            "--algorithm",
            "PCA",
            "--n-components",
            "5",
            "--dataset",
            "swissroll",
            "--metric",
            "participation_ratio",
        ],
    )

    assert result.exit_code == 0
    assert "participation_ratio" in result.output


def test_cli_run_with_viz_file(cli_runner):
    """CLI run can save visualization to file."""
    from manyruns.harness.__main__ import cli

    with tempfile.TemporaryDirectory() as tmpdir:
        viz_path = Path(tmpdir) / "plot.png"

        result = cli_runner.invoke(
            cli,
            [
                "run",
                "--algorithm",
                "PCA",
                "--n-components",
                "5",
                "--dataset",
                "swissroll",
                "--viz",
                "file",
                "--viz-output",
                str(viz_path),
            ],
        )

        assert result.exit_code == 0
        assert viz_path.exists()


def test_mock_round_trip(cli_runner):
    """Full round-trip: resolve preset → MockBackend → display → save PNG."""
    from manyruns.harness.__main__ import cli

    with tempfile.TemporaryDirectory() as tmpdir:
        viz_path = Path(tmpdir) / "embedding.png"

        result = cli_runner.invoke(
            cli,
            [
                "--no-banner",
                "run",
                "--preset",
                "pca_umap",
                "--dataset",
                "swissroll",
                "--backend",
                "mock",
                "--viz",
                "file",
                "--viz-output",
                str(viz_path),
            ],
        )

        # CLI exits cleanly
        assert result.exit_code == 0, result.output

        # Text output contains the scores MockBackend returns
        assert "trustworthiness" in result.output
        assert "participation_ratio" in result.output

        # Plot was saved and the path was printed
        assert "Plot saved to:" in result.output
        assert viz_path.exists()
        assert viz_path.stat().st_size > 0
