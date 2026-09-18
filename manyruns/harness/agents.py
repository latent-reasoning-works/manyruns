# manyruns/harness/agents.py
"""Orchestrate a manyAgents adapter call.

manyruns runs no inference itself — this only dispatches a prompt to manyAgents and
normalizes the result. All computation lives in manyAgents (and the model
backend it wraps: vLLM, HF, ollama, Claude, ...).

This is the agent-side mirror of ``runner.py`` (the manyLatents/DR side):
    agents.py : manyAgents  ::  runner.py : manyLatents
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


def call_agent(
    prompt: str,
    *,
    agent: str = "vllm",
    model: str | None = None,
    build_trace: bool = False,
    system_prompt: str | None = None,
    input_files: dict[str, Path] | None = None,
    **task_config: Any,
) -> dict[str, Any]:
    """Dispatch a prompt to a manyAgents adapter; return a normalized result.

    The adapter's ``AdapterResult`` puts the *full* generation in
    ``output_files["raw_response"]`` (its ``summary`` is truncated), so we read
    it back. Extra manyAgents knobs pass straight through via ``**task_config``.

    Returns a dict: ``{answer, trace_path, hidden_states_path, metadata}``.
    """
    try:
        from manyagents.adapters import ADAPTER_REGISTRY
    except ImportError as e:  # pragma: no cover - exercised via the agents extra
        raise ImportError(
            "manyagents is required for the prompt stream. Install with: uv sync --extra agents"
        ) from e

    if agent not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown agent {agent!r}. Choose from: {sorted(ADAPTER_REGISTRY)}")

    cfg: dict[str, Any] = {"prompt": prompt, "build_trace": build_trace}
    if model:
        cfg["model"] = model
    if system_prompt:
        cfg["system_prompt"] = system_prompt
    cfg.update(task_config)

    result = asyncio.run(ADAPTER_REGISTRY[agent]().run(cfg, input_files or {}))

    if not result.get("success"):
        raise RuntimeError(result.get("summary", "agent call failed"))

    files = result.get("output_files", {}) or {}
    raw = files.get("raw_response")
    answer = Path(raw).read_text() if raw else result.get("summary", "")

    return {
        "answer": answer,
        "trace_path": files.get("trace"),
        "hidden_states_path": files.get("hidden_states"),
        "metadata": result.get("metadata", {}),
    }
