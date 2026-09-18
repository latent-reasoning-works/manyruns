from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(eq=False)
class Request:
    # DR stream (manyLatents)
    dataset: str = ""
    workflow: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    run_dir: Path | None = None
    # Prompt stream (manyAgents) — routes to AgentBackend
    prompt: str | None = None
    agent: str = "vllm"
    model: str | None = None


@dataclass(eq=False)
class Response:
    embeddings: np.ndarray | None = None
    scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    workflow: list[dict[str, Any]] = field(default_factory=list)
    dataset: str = ""
    request: Request | None = None
    # Prompt stream
    answer: str | None = None
    trace_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "embeddings": self.embeddings,
            "scores": self.scores,
            "metadata": self.metadata,
            "workflow": self.workflow,
            "dataset": self.dataset,
        }


class Backend(ABC):
    """ReadingInterface — the one surface the CLI uses to reach a backend.

    Any package can implement this to plug into the harness.
    """

    @abstractmethod
    def run(self, request: Request) -> Response: ...


class MockBackend(Backend):
    """Canned responses — no downstream dependencies needed.

    Use this to develop and test the full harness loop without manyLatents
    or Manyruns installed.
    """

    def run(self, request: Request) -> Response:
        rng = np.random.default_rng(seed=42)
        embeddings = rng.standard_normal((200, 2))
        return Response(
            embeddings=embeddings,
            scores={"trustworthiness": 0.95, "participation_ratio": 0.42},
            metadata={
                "total_time": 0.0,
                "step_times": [0.0] * max(len(request.workflow), 1),
                "steps_completed": len(request.workflow),
            },
            workflow=request.workflow,
            dataset=request.dataset,
            request=request,
        )


class DRBackend(Backend):
    """Dimensionality Reduction via manyLatents.

    Requires: uv sync --extra harness
    """

    def run(self, request: Request) -> Response:
        if request.run_dir is not None:
            from manyruns.harness.runner import load_run

            result = load_run(request.run_dir)
        else:
            from manyruns.harness.runner import run_workflow

            result = run_workflow(
                workflow=request.workflow,
                dataset=request.dataset,
                metrics=request.metrics or None,
            )
        return Response(
            embeddings=result["embeddings"],
            scores=result["scores"],
            metadata=result["metadata"],
            workflow=result["workflow"],
            dataset=result["dataset"],
            request=request,
        )


class AgentBackend(Backend):
    """Natural-language prompt -> manyAgents.

    manyruns only orchestrates: inference lives in manyAgents (vLLM, HF, ollama,
    Claude, ...). Slice 1 is text-only; hidden-state traces arrive with the
    vLLM/HF backends on GPU.
    """

    def run(self, request: Request) -> Response:
        if not request.prompt:
            raise ValueError("AgentBackend requires request.prompt")
        from manyruns.harness.agents import call_agent

        out = call_agent(request.prompt, agent=request.agent, model=request.model)
        return Response(
            metadata=out["metadata"],
            dataset=request.dataset,
            answer=out["answer"],
            trace_path=out["trace_path"],
            request=request,
        )


_BACKENDS: dict[str, type[Backend]] = {
    "mock": MockBackend,
    "dr": DRBackend,
    "agent": AgentBackend,
}


def run(request: Request, backend: str | Backend | None = None) -> Response:
    if backend is None:
        # Route by what the request carries: a prompt → manyAgents; a DR workflow → the
        # geometry stream (dr) — NOT mock, which would silently return canned embeddings for a
        # real workflow; only a bare request (neither) falls back to the dep-free mock.
        backend = "agent" if request.prompt else ("dr" if request.workflow else "mock")
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend {backend!r}. Choose from: {list(_BACKENDS)}")
        backend = _BACKENDS[backend]()
    return backend.run(request)
