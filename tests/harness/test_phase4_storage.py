# tests/harness/test_phase4_storage.py
"""Phase 4 E2E tests: Labeled dataset storage."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from manyruns.harness.labeler import label_workflow


def _make_result(**overrides) -> dict:
    """Build a sample run result dict with optional overrides."""
    base = {
        "embeddings": np.random.randn(100, 5),
        "scores": {"participation_ratio": 0.5},
        "metadata": {"total_time": 1.0, "step_times": [1.0], "steps_completed": 1},
        "workflow": [{"algorithm": "PCA", "params": {"n_components": 5}}],
        "dataset": "swissroll",
    }
    base.update(overrides)
    return base


@pytest.fixture
def sample_labeled() -> dict:
    """Create a sample labeled entry dict for testing."""
    result = _make_result()
    return label_workflow(
        result,
        label="cluster",
        selected_metrics=["participation_ratio"],
        labeler="test_user",
    )


def test_save_and_load_labeled_workflow(sample_labeled):
    """Phase 4 E2E: Labeled workflows persist and reload."""
    from manyruns.harness.storage import load_labeled, save_labeled

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "labeled.json"

        save_labeled(path, sample_labeled, embedding=np.random.randn(100, 5))

        entries = load_labeled(path)
        assert len(entries) == 1
        assert entries[0]["label"] == "cluster"
        assert "participation_ratio" in entries[0]["selected_metrics"]


def test_append_to_existing_dataset(sample_labeled):
    """Can append entries to existing dataset."""
    from manyruns.harness.storage import load_labeled, save_labeled

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "labeled.json"

        # First entry
        save_labeled(path, sample_labeled, embedding=np.random.randn(100, 5))

        # Second entry
        result2 = _make_result(
            embeddings=np.random.randn(200, 2),
            workflow=[{"algorithm": "UMAP", "params": {"n_components": 2}}],
            dataset="embryoid",
            scores={"trustworthiness": 0.9},
        )
        labeled2 = label_workflow(result2, label="trajectory", selected_metrics=[], labeler="user2")
        save_labeled(path, labeled2, embedding=np.random.randn(200, 2))

        # Verify both entries
        entries = load_labeled(path)
        assert len(entries) == 2
        labels = {e["label"] for e in entries}
        assert labels == {"cluster", "trajectory"}


def test_embedding_saved_as_npy(sample_labeled):
    """Embeddings are saved as separate NPY files."""
    from manyruns.harness.storage import save_labeled

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "labeled.json"
        embedding = np.random.randn(100, 5)

        save_labeled(path, sample_labeled, embedding=embedding)

        # Check NPY file exists
        embeddings_dir = Path(tmpdir) / "embeddings"
        assert embeddings_dir.exists()
        npy_files = list(embeddings_dir.glob("*.npy"))
        assert len(npy_files) == 1

        # Verify content
        loaded_embedding = np.load(npy_files[0])
        np.testing.assert_array_equal(loaded_embedding, embedding)


def test_load_embedding(sample_labeled):
    """Can load embedding from stored dataset."""
    from manyruns.harness.storage import load_embedding, load_labeled, save_labeled

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "labeled.json"
        embedding = np.random.randn(100, 5)

        save_labeled(path, sample_labeled, embedding=embedding)

        entries = load_labeled(path)
        loaded_embedding = load_embedding(path, entries[0])

        np.testing.assert_array_almost_equal(loaded_embedding, embedding)


def test_dataset_iteration(sample_labeled):
    """Can iterate over dataset entries."""
    from manyruns.harness.storage import load_labeled, save_labeled

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "labeled.json"

        for i in range(3):
            result = _make_result(
                embeddings=np.random.randn(50, 2),
                workflow=[{"algorithm": "PCA", "params": {"n_components": 2}}],
                dataset=f"dataset_{i}",
                scores={},
            )
            labeled = label_workflow(result, label="cluster", selected_metrics=[], labeler="user")
            save_labeled(path, labeled, embedding=np.random.randn(50, 2))

        entries = load_labeled(path)
        datasets = [e["dataset"] for e in entries]
        assert datasets == ["dataset_0", "dataset_1", "dataset_2"]


def test_dataset_metadata():
    """Dataset includes metadata (version, created timestamp)."""
    from manyruns.harness.storage import dataset_summary, save_labeled

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "labeled.json"

        # Create a minimal entry to have a valid dataset file
        result = _make_result(scores={})
        labeled = label_workflow(result, label="other", selected_metrics=[], labeler="test")
        save_labeled(path, labeled, embedding=np.random.randn(100, 5))

        summary = dataset_summary(path)
        assert summary["version"] == "1.0"
        assert summary["created"] != ""
