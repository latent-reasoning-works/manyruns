# tests/harness/test_phase1_runner.py
"""Phase 1 E2E tests: Run DR workflow and return embedding."""

import numpy as np
import pytest

# The runner exercises the real manyLatents DR backend — a private editable
# sibling CI can't install. Skip the whole module when it's absent.
pytest.importorskip("manylatents")


def test_run_workflow_returns_embedding():
    """Phase 1 E2E: Running a workflow returns an embedding."""
    from manyruns.harness.runner import run_workflow

    # `swissroll` is 3-dimensional. This asked for 50 components of a 3-column dataset
    # and raised — a nonsense request that was never what this test is about, and the
    # same failure the front door hit on `pca` (see `manyruns/pipeline/bounds.py`).
    # The gap it exposed is REAL and is pinned by `test_a_named_dataset_is_unbounded`
    # below rather than by leaving five reds standing.
    result = run_workflow(
        workflow=[
            {"algorithm": "PCA", "params": {"n_components": 3}},
            {"algorithm": "UMAP", "params": {"n_components": 2}},
        ],
        dataset="swissroll",
    )

    assert result["embeddings"] is not None
    assert isinstance(result["embeddings"], np.ndarray)
    assert result["embeddings"].shape[1] == 2  # Final dim from UMAP
    assert result["metadata"]["steps_completed"] == 2
    assert result["metadata"]["total_time"] > 0


def test_run_workflow_with_metrics():
    """Workflow can compute metrics at each step."""
    from manyruns.harness.runner import run_workflow

    result = run_workflow(
        workflow=[{"algorithm": "PCA", "params": {"n_components": 3}}],
        dataset="swissroll",
        metrics=["participation_ratio", "trustworthiness"],
    )

    assert "participation_ratio" in result["scores"]
    assert "trustworthiness" in result["scores"]


@pytest.mark.xfail(strict=True, reason="a named dataset is loaded inside manylatents, so "
                                       "manyruns never sees the array to bound against")
def test_a_named_dataset_is_unbounded():
    """THE GAP, written down as the behaviour we want rather than as five standing reds.

    `manyruns/pipeline/bounds.py` clamps a step's parameters to what the data admits, and on
    the recipe path it fixed exactly this failure — `n_components: 10` on a 3-column array broke
    three of eight bundled recipes. It cannot reach here: `run_workflow`'s first step passes
    `data=<name>` and `manylatents.api` exposes only `run`, so the array is loaded inside the
    engine and manyruns has no shape to bound against. `manylatents.data.get_datamodule` builds
    a module but leaves `dataset` unpopulated after `setup()`, and going further would mean
    depending on another repo's internal structure.

    STRICT xfail on purpose: when the upstream API grows a way to ask a named dataset its shape
    — or when `run_workflow` takes an array — this starts passing and pytest reports it, which
    is the signal to delete the marker. A skip would go quiet instead.
    """
    from manyruns.harness.runner import run_workflow

    result = run_workflow(
        workflow=[{"algorithm": "PCA", "params": {"n_components": 50}}],
        dataset="swissroll",       # three columns; 50 components do not exist
    )

    assert result["embeddings"].shape[1] == 3
