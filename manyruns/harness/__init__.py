# manyruns/harness/__init__.py
"""DR Workflow Labeling Harness.

This module provides tools for running DR workflows, collecting expert labels,
saving labeled datasets, and launching Manyruns training.

Requires the [harness] extra: uv sync --extra harness
"""

from __future__ import annotations

# Lazy imports - only import when accessed
# This allows the CLI to load without all dependencies installed


def __getattr__(name: str):
    """Lazy import attributes on first access."""
    if name == "run":
        from manyruns.harness.interface import run

        return run
    elif name == "Request":
        from manyruns.harness.interface import Request

        return Request
    elif name == "Response":
        from manyruns.harness.interface import Response

        return Response
    elif name == "Backend":
        from manyruns.harness.interface import Backend

        return Backend
    elif name == "MockBackend":
        from manyruns.harness.interface import MockBackend

        return MockBackend
    elif name == "DRBackend":
        from manyruns.harness.interface import DRBackend

        return DRBackend
    elif name == "AgentBackend":
        from manyruns.harness.interface import AgentBackend

        return AgentBackend
    elif name == "call_agent":
        from manyruns.harness.agents import call_agent

        return call_agent
    elif name == "run_workflow":
        from manyruns.harness.runner import run_workflow

        return run_workflow
    elif name == "load_run":
        from manyruns.harness.runner import load_run

        return load_run
    elif name == "format_results":
        from manyruns.harness.display import format_results

        return format_results
    elif name == "display_results":
        from manyruns.harness.display import display_results

        return display_results
    elif name == "visualize_embedding":
        from manyruns.harness.display import visualize_embedding

        return visualize_embedding
    elif name == "label_workflow":
        from manyruns.harness.labeler import label_workflow

        return label_workflow
    elif name == "interactive_label":
        from manyruns.harness.labeler import interactive_label

        return interactive_label
    elif name == "VALID_LABELS":
        from manyruns.harness.labeler import VALID_LABELS

        return VALID_LABELS
    elif name == "save_labeled":
        from manyruns.harness.storage import save_labeled

        return save_labeled
    elif name == "load_labeled":
        from manyruns.harness.storage import load_labeled

        return load_labeled
    elif name == "load_embedding":
        from manyruns.harness.storage import load_embedding

        return load_embedding
    elif name == "dataset_summary":
        from manyruns.harness.storage import dataset_summary

        return dataset_summary
    elif name == "PRESETS":
        from manyruns.harness.configs import PRESETS

        return PRESETS
    elif name == "list_presets":
        from manyruns.harness.configs import list_presets

        return list_presets
    elif name == "load_preset":
        from manyruns.harness.configs import load_preset

        return load_preset
    elif name == "resolve_config":
        from manyruns.harness.configs import resolve_config

        return resolve_config
    elif name == "acquire":
        from manyruns.harness.data_acquire import acquire

        return acquire
    elif name == "convert":
        from manyruns.harness.data_acquire import convert

        return convert
    elif name == "load_manifest":
        from manyruns.harness.data_acquire import load_manifest

        return load_manifest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Interface
    "Backend",
    "MockBackend",
    "DRBackend",
    "AgentBackend",
    "call_agent",
    "run",
    "Request",
    "Response",
    # Phase 1: Runner
    "run_workflow",
    "load_run",
    # Phase 2: Display
    "format_results",
    "display_results",
    "visualize_embedding",
    # Phase 3: Labeler
    "label_workflow",
    "interactive_label",
    "VALID_LABELS",
    # Phase 4: Storage
    "save_labeled",
    "load_labeled",
    "load_embedding",
    "dataset_summary",
    # Phase 6: Trainer
    # Configs
    "PRESETS",
    "list_presets",
    "load_preset",
    "resolve_config",
    # Data acquire-read shim (raw omics → canonical .h5ad)
    "acquire",
    "convert",
    "load_manifest",
]
