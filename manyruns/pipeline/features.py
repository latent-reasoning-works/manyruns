"""Publish descriptive features of the original run input."""

from manyruns import installation
from manyruns.pipeline.steps import _StepUnsupported


def _helper():
    """Load the optional helper only when the step runs."""
    try:
        from manylatents.singlecell.dataset_features import dataset_features
    except ImportError as exc:
        raise _StepUnsupported(
            "dataset_features needs manylatents-omics with singlecell.dataset_features; "
            + installation.RUNTIME_REPAIR
        ) from exc
    return dataset_features


def step(state, g, params, ctx):
    _helper()
    raise _StepUnsupported("dataset_features table publication is not implemented yet")
