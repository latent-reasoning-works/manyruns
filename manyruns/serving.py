"""Pluggable model-serving backends for manyruns.

KEY SHAPE: a manyruns "model" is NOT a weight blob, and NOT a fixed declarative
sequence (that was an over-abstraction — manylatents dropped its "workflow" mode for
exactly this reason). It's a DRIVER LOOP: a `while` that repeatedly asks for the next
step and calls the open engine API with it. A step is open-ended — a `module`, `model`,
`metric`, or anything else — and the loop runs to a dynamic stopping condition ("a
proper signature for analysis"). The executed steps are an emergent *trace*, not a
pre-declared list. (the learner's own executor step is the real driver; the RL policy
decides each next step.)

The product talks to a backend through ONE interface (`ModelServer.predict`); the
backend is chosen by Hydra `_target_` via `serving=` — mirroring shop's `cluster=`
launcher swap. Start free (`LocalServer`, in-process, $0); swap later, no caller
changes.

    serving=local   -> in-process driver loop        ($0, default)
    serving=modal   -> remote serverless (stub)       (scale-to-zero, add when needed)

LocalServer engines:
    engine=mock     -> self-contained loop over an open step API (runnable today)
    (an engine naming the learner used to delegate the loop downstream; there is none — §3.5)
    (removed) engine=real -> loaded a file/generator and ran the recipe with public libs;
                       public libs (phate/statsmodels/…), bypassing the learner + shop
                       (see manyruns/pipeline.py). For "paste a path, get the shape."
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

DEFAULT_METRICS = ["trustworthiness", "continuity"]  # mock stand-ins

#: What a mock result says about itself, in its own record. `mock` is a development backend
#: (`app.DEV_ENGINES`): it never opens the data and every number it reports is invented, while
#: the panel it produces looks exactly like a measured one. It is no longer offered by the
#: picker or chosen as a default, so reaching it takes an explicit `--engine mock` — and this
#: is what stops that result OUTLIVING the intent, in `index.jsonl`, a summary, or a
#: screenshot someone reads a month later.
MOCK_CAVEAT = "engine=mock — every number here is invented; no data was read"


@runtime_checkable
class ModelServer(Protocol):
    """The whole contract. Callers depend on this, never on a backend."""

    def predict(self, inputs: Any) -> Any: ...


# --- mock open-API primitives (stand in for manylatents ops + metrics) ----------
def _apply_op(name: str, params: dict, vec: list[float]) -> list[float]:
    """A module/model step: transform the data (deterministic stand-in for DR)."""
    n_out = int(params.get("n_components") or params.get("n_neighbors") or len(vec))
    n_out = max(1, min(n_out, len(vec)))
    f = sum(map(ord, name)) % 5 + 1
    return [round(((v * f) % 9) / 9, 3) for v in vec[:n_out]]


def _mock_step(step: dict, state: dict) -> str:
    """Run one recipe step against the mock state; returns its trace entry.

    Dispatches on `group` — the same single vocabulary the real engines use, so a recipe
    reads identically on every backend and the traces are comparable."""
    name, params = step["name"], step.get("params", {})
    group = step.get("group")
    if group in ("latent", "lightning"):
        state["vec"] = _apply_op(name, params, state["vec"])
    elif group == "analysis":
        state["g"][name] = _measure(name, state["vec"])
    return f"{group}:{name}"


def _measure(name: str, vec: list[float]) -> float:
    """A metric step: read the current data, don't transform it."""
    n = len(vec) or 1
    mean = sum(vec) / n
    if name in ("continuity", "variance"):
        return round(sum((x - mean) ** 2 for x in vec) / n, 4)
    return round(mean, 3)


def _mark_mock(result: dict) -> dict:
    """Stamp a mock result with what it is, without clobbering caveats it already carries."""
    result.setdefault("caveats", [])
    if MOCK_CAVEAT not in result["caveats"]:
        result["caveats"].insert(0, MOCK_CAVEAT)
    return result


class LocalServer:
    """In-process driver loop. $0 — the default backend."""

    def __init__(
        self,
        engine: str = "mock",
        max_steps: int = 8,
        target_dim: int = 3,
        metrics: Any = None,
        algorithm: str = "pca",
        device: str = "cpu",
        **engine_kwargs: Any,
    ):
        self.engine = engine
        self.max_steps = int(max_steps)
        self.target_dim = int(target_dim)
        self.metrics = list(metrics) if metrics else list(DEFAULT_METRICS)
        self.algorithm = algorithm
        self.device = device
        # `mode` and `data_key` USED TO LIVE HERE — the engine-contract knobs for the learner
        # path (which Hydra mode to run, which key names the dataset), correctable via
        # the learner's serving config. They went with the path. They are named here because they were
        # a public keyword argument, and a caller still passing one now lands in `engine_kwargs`
        # and is ignored rather than refused — the only silent thing this removal leaves behind.
        self.engine_kwargs = engine_kwargs

    #: Every backend this server can actually serve. The list is here rather than derived from
    #: `app.ENGINES` because `serving` must not import `app`; `tests/test_public_path.py::
    #: test_the_two_engine_lists_cannot_drift_apart` pins the two together. `pipeline`'s in-process step loop (`run_inproc`) is deliberately ABSENT:
    #: it is the suite's substrate, not a backend, and nothing may route a request to it.
    #: NO learner, and that is a RULE rather than an omission — the environment-contract spec
    #: §3.5. manyruns has no track to a learner: the learner observes manyruns and drives it through
    #: the CLI, so a backend here that called INTO it was the arrow pointing backwards. An
    #: environment that imports its agent is not an environment.
    SERVES = frozenset({"mock", "manylatents"})

    def predict(self, inputs: Any) -> Any:
        # An unknown engine used to fall through to `_run_mock`, so `LocalServer(engine="real")`
        # answered a real request with invented numbers and stamped `engine=mock` on the result
        # while `self.engine` still read `real`. That is the failure `app._default_engine`
        # refuses to commit one layer up; refusing it here too closes the library-caller door.
        if self.engine not in self.SERVES:
            raise ValueError(
                f"engine={self.engine!r} is not a serving backend. "
                f"Available: {', '.join(sorted(self.SERVES))}."
            )
        if self.engine == "manylatents":
            return self._run_manylatents(inputs)
        return self._run_mock(inputs)

    # real compute delegated to manylatents.api.run (phate → mioflow chained, analysis
    # in-process). engine=manylatents. See manyruns/pipeline.run_manylatents.
    def _run_manylatents(self, data: Any) -> dict:
        from manyruns import pipeline

        if not isinstance(data, dict):
            raise RuntimeError("engine=manylatents needs a recipe payload (internal wiring error)")
        return pipeline.run_manylatents(
            recipe=data.get("recipe") or {},
            data_ref=data.get("dataset"),
            dataset_name=data.get("dataset_name"),
            array=data.get("array"),
            out_dir=Path(data.get("out_dir") or "."),
            seed=int(42 if data.get("seed") is None else data["seed"]),
            fast_dev_run=bool(data.get("fast_dev_run", True)),
            data_kwargs=data.get("data_kwargs") or {},
            declared_shape=data.get("declared_shape"),
            color=data.get("color"), sample_ids=data.get("obs_names"),
            label_key=data.get("label_key"),
            labels=data.get("labels"),
            # The gene axis. `run_manylatents` has carried these two parameters since the frame
            # landed, with nothing on any product path supplying them; `app._read_inputs` is now
            # the producer, and this is the last hop.
            counts=data.get("counts"),
            genes=data.get("genes"),
            layers=data.get("layers"),   # a second matrix over the same cells, declared-only
            device=self.device,   # the resolved device reaches MIOFlow's Trainer (no longer a cpu pin)
            metrics=data.get("metrics"),   # the declared metric suite (manyruns.protocols.Suite)
            on_step=data.get("on_step"),
            # "time" | "condition" | "group". THREE kinds, and every consumer takes an
            # ALLOWLIST on the one it wants: only "time" reaches a trajectory step
            # (`runner.py`), only "condition" reaches `separation`/`composition`
            # (`steps.py`). The old note here said "mioflow refuses condition" — a
            # denylist of one, which is exactly the form that let `group` through.
            label_kind=data.get("label_kind"),
        )

    def _run_mock(self, data: Any) -> dict:
        # A caller can inject an explicit ordered recipe (the product-owned CFlows
        # trace: phate -> mioflow) via a structured dict input. Bare inputs
        # keep the adaptive driver loop below — the two paths never cross.
        if isinstance(data, dict) and data.get("recipe"):
            # BOTH mock paths get the caveat. It lived only on the adaptive branch below, and
            # this is the branch the product actually takes — a recipe run through the mock
            # came back unmarked, which is precisely the case that matters.
            return _mark_mock(self._run_recipe(data))
        state: dict[str, Any] = {"vec": self._as_vector(data), "g": {}}
        trace: list[str] = []
        i = 0
        while True:
            step = self._next_step(state, i)  # open-ended; None => done (while-condition)
            if step is None:
                break
            trace.append(_mock_step(step, state))
            i += 1
        return {
            "served_by": "local",
            "engine": "mock",
            "num_steps": len(trace),
            "trace": trace,          # emergent, heterogeneous, dynamic length
            # The mock conforms to the same g-vector schema as the real engines: emit what
            # it genuinely has (`final_dim`) and OMIT what it doesn't. It works on a 1-D
            # stand-in vector, so it has no rows — n_samples/n_features/n_embedded are
            # absent rather than invented, the same rule the engines follow for a named
            # dataset whose input shape is never seen.
            "g_vector": {**state["g"], "final_dim": len(state["vec"])},
            "final_dim": len(state["vec"]),
            "caveats": ["engine=mock — every number here is invented; no data was read"],
        }

    # recipe path: run a fixed, ordered list of steps (the default static exploration).
    # Same open step API as the adaptive loop, but the trace is pre-declared by the
    # product's recipe config rather than emergent. Compute is still the mock stand-in;
    # the real algorithms arrived via the learner's engine (delegated downstream).
    def _run_recipe(self, payload: dict) -> dict:
        from manyruns.pipeline import runner

        recipe = payload["recipe"]
        seed = payload.get("seed", 42)
        dataset = payload.get("dataset") or payload.get("data")
        identity_args = dict(engine="mock", seed=seed, dataset=dataset,
                             dataset_name=payload.get("dataset_name"),
                             data_kwargs=payload.get("data_kwargs"))
        ident = runner.identity(recipe, **identity_args)
        # Use the same record loop as interactive mock sessions. No input arrays are
        # supplied, so prep declines honestly and the mock executors only mutate a list.
        state = runner._new_state()
        state["vec"] = self._as_vector(payload.get("data", ""))
        g: dict = {}
        ctx = {"out_dir": Path(payload.get("out_dir") or "."), "plots": [],
               "target_dim": self.target_dim, "seed": seed, "run_id": ident[0], "caveats": []}
        steps = runner._run_steps(recipe, state, g, dispatch=runner.dispatch_for("mock"),
                                  ctx=ctx, on_step=payload.get("on_step"))
        g["final_dim"] = len(state["vec"])
        return runner._finalize(state, g, recipe=recipe, steps=steps, plots=[],
                                caveats=ctx["caveats"], ident=ident, **identity_args)

    def _next_step(self, state: dict, i: int) -> dict | None:
        """The driver: pick the next open-ended step from current state. This is the
        decision the RL policy makes for real; here it's a simple deterministic rule."""
        if i >= self.max_steps:
            return None
        dim = len(state["vec"])
        if dim > self.target_dim:  # keep reducing — alternate latent/lightning (heterogeneous)
            name = ["pca", "umap", "autoencoder", "phate"][i % 4]
            group = "lightning" if name == "autoencoder" else "latent"
            return {"group": group, "name": name,
                    "params": {"n_components": max(self.target_dim, dim // 2)}}
        for m in self.metrics:     # then measure until the signature is complete
            if m not in state["g"]:
                return {"group": "analysis", "name": m, "params": {}}
        return None                # nothing left to do -> stop

    @staticmethod
    def _as_vector(data: Any) -> list[float]:
        if isinstance(data, (list, tuple)):
            return [float(x) for x in data] or [0.0]
        return [round((ord(c) % 10) / 10, 3) for c in str(data)] or [0.0]


class ModalServer:
    """Remote serverless backend (scale-to-zero). Stub — proves the swap point:
    same interface, different infra. Implement when hosted serving is needed."""

    def __init__(self, app_name: str = "manyruns-core", **kwargs: Any):
        self.app_name = app_name
        self.kwargs = kwargs

    def predict(self, inputs: Any) -> Any:
        raise NotImplementedError(
            "ModalServer is a stub. Add the remote call when you need hosted serving — "
            "only the `serving=` config flips; callers are untouched."
        )


def load_server(cfg: Any) -> ModelServer:
    """Instantiate the backend from a Hydra/OmegaConf node carrying `_target_`."""
    from hydra.utils import instantiate

    return instantiate(cfg)


def default_server() -> ModelServer:
    """Zero-config default backend (local, in-process, $0) for callers without Hydra."""
    return LocalServer()
