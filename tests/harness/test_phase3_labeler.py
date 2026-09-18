# tests/harness/test_phase3_labeler.py
"""Phase 3 E2E tests: Interactive labeling."""

from datetime import datetime

import numpy as np
import pytest


@pytest.fixture
def sample_result() -> dict:
    """Create a sample run result dict for testing."""
    return {
        "embeddings": np.random.randn(100, 2),
        "scores": {"participation_ratio": 1.93, "trustworthiness": 0.89},
        "metadata": {
            "total_time": 1.5,
            "step_times": [1.5],
            "steps_completed": 1,
        },
        "workflow": [{"algorithm": "PCA", "params": {"n_components": 2}}],
        "dataset": "swissroll",
    }


def test_label_workflow_returns_labeled_entry(sample_result):
    """Phase 3 E2E: Labeling returns a labeled dict."""
    from manyruns.harness.labeler import label_workflow

    labeled = label_workflow(
        sample_result,
        label="trajectory",
        selected_metrics=["participation_ratio"],
        labeler="test_user",
    )

    assert isinstance(labeled, dict)
    assert labeled["label"] == "trajectory"
    assert "participation_ratio" in labeled["selected_metrics"]
    assert labeled["selected_metrics"]["participation_ratio"] == pytest.approx(1.93, rel=0.01)
    assert labeled["labeler"] == "test_user"


def test_label_workflow_validates_label(sample_result):
    """Invalid labels are rejected."""
    from manyruns.harness.labeler import label_workflow

    with pytest.raises(ValueError, match="Invalid label"):
        label_workflow(
            sample_result,
            label="invalid_topology",
            selected_metrics=[],
            labeler="test_user",
        )


def test_label_workflow_validates_metrics(sample_result):
    """Selected metrics must exist in result."""
    from manyruns.harness.labeler import label_workflow

    with pytest.raises(ValueError, match="not found"):
        label_workflow(
            sample_result,
            label="cluster",
            selected_metrics=["nonexistent_metric"],
            labeler="test_user",
        )


def test_labeled_workflow_has_timestamp(sample_result):
    """Labeled entry includes ISO timestamp."""
    from manyruns.harness.labeler import label_workflow

    labeled = label_workflow(
        sample_result,
        label="cycle",
        selected_metrics=["trustworthiness"],
        labeler="expert",
    )

    assert labeled["labeled_at"] is not None
    # Verify it's a valid ISO timestamp
    datetime.fromisoformat(labeled["labeled_at"])


def test_labeled_workflow_has_uuid(sample_result):
    """Labeled entries have unique IDs."""
    from manyruns.harness.labeler import label_workflow

    labeled1 = label_workflow(sample_result, label="cluster", selected_metrics=[], labeler="a")
    labeled2 = label_workflow(sample_result, label="cluster", selected_metrics=[], labeler="b")

    assert labeled1["id"] != labeled2["id"]


def test_labeled_workflow_preserves_workflow_info(sample_result):
    """Labeled entry contains original workflow and dataset."""
    from manyruns.harness.labeler import label_workflow

    labeled = label_workflow(
        sample_result,
        label="surface",
        selected_metrics=["participation_ratio"],
        labeler="user",
    )

    assert labeled["workflow"] == sample_result["workflow"]
    assert labeled["dataset"] == sample_result["dataset"]
    assert labeled["embedding_shape"] == list(sample_result["embeddings"].shape)


def test_valid_labels():
    """Check all valid label values."""
    from manyruns.harness.labeler import VALID_LABELS

    assert "cluster" in VALID_LABELS
    assert "trajectory" in VALID_LABELS
    assert "cycle" in VALID_LABELS
    assert "surface" in VALID_LABELS
    assert "other" in VALID_LABELS
