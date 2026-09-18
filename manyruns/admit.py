"""The admission gate — which tools have earned the right to be believed.

The problem this exists for: running every metric against structurally wrong input found **~16 of 38 distinct callables return a confident number
rather than raising**. A harness that runs "all the tools" without a gate is therefore a
machine for producing plausible nonsense at 38x throughput, and the more tools you add the
worse it gets.

`configs/metrics/default.yaml` is the same discipline applied by hand — 110 registry names,
38 callables, **12 admitted**, each removal carrying the measurement that killed it. This
module is that process made executable, so a new tool can be judged in seconds instead of an
afternoon, and so the judgement is reproducible rather than remembered.

Three checks, deliberately mechanical. None asks whether a tool is *useful* — that is a
scientific question no gate can answer. They ask whether its output can be TRUSTED AT ALL:

  raises    — given structurally wrong input, does it fail? A tool that returns 0.7 for a
              matrix of the wrong shape will return 0.7 for your cohort too.
  stable    — same input twice, same answer? An unseeded stochastic tool cannot be compared
              across runs, which is what a sweep does for a living.
  null      — on iid noise, is the answer distinguishable from its own value on structure?
              A metric that scores noise the same as a manifold measures nothing. (This is
              the check the deleted `granger` step failed, and the one four candidate
              readouts failed against a matched control this session.)

A tool that fails `raises` is not admitted. A tool that fails `stable` or `null` is admitted
with the failure recorded, because a number you know to be scale-free is still usable if you
never compare it — the point is that the caller is told, not that the tool is hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

#: What a verdict can be. `deferred` is not a failure — it is the honest answer when the
#: check could not run here (no engine installed), and it must never read as a pass.
VERDICTS = ("admitted", "flagged", "rejected", "deferred")


@dataclass
class Verdict:
    name: str
    verdict: str
    checks: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Does this pass the gate? Only `rejected` fails, and `cmd_audit`'s exit code says
        the same — a tool that is merely unstable or scale-blind is reported, not blocked,
        because a number you have been told not to compare is still usable. These two
        definitions disagreeing would make the CLI and the API give different verdicts on
        the same tool."""
        return self.verdict != "rejected"


def _noise(n: int = 120, d: int = 8, seed: int = 0):
    import numpy as np

    return np.random.default_rng(seed).normal(size=(n, d))


def _structure(n: int = 120, d: int = 8, seed: int = 0):
    """A ring — genuine low-dimensional structure in the same ambient shape as the noise,
    so a metric that cannot tell them apart is not being asked a trick question."""
    import numpy as np

    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    xy = np.c_[np.cos(t), np.sin(t)]
    basis, _ = np.linalg.qr(rng.normal(size=(d, 2)))
    return xy @ basis.T + rng.normal(0, 0.02, (n, d))


def check_raises(fn: Callable) -> tuple[bool, str]:
    """Structurally wrong input must fail, not produce a number."""
    garbage: list[Any] = ["not a matrix", [], 42]
    for bad in garbage:
        try:
            out = fn(bad)
        except Exception:  # noqa: BLE001 - raising IS the pass condition
            continue
        return False, f"returned {out!r} for {type(bad).__name__} input"
    return True, ""


def check_stable(fn: Callable) -> tuple[bool, str]:
    """Same input twice, same answer — or a sweep cannot compare its own cells."""
    X = _structure()
    try:
        a, b = fn(X), fn(X)
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if a != a and b != b:  # NaN == NaN is False; two NaNs are "stable" and useless
        return False, "returns NaN"
    return (a == b), "" if a == b else f"{a!r} then {b!r}"


def check_null(fn: Callable) -> tuple[bool, str]:
    """Structure and iid noise must not score the same.

    Deliberately weak: it asks only that the two DIFFER, not that they differ in a
    particular direction, because the direction is a scientific claim and this is a gate."""
    try:
        on_structure, on_noise = fn(_structure()), fn(_noise())
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if on_structure != on_structure or on_noise != on_noise:
        return False, "returns NaN"
    if on_structure == on_noise:
        return False, f"identical on structure and noise ({on_structure!r})"
    spread = abs(on_structure - on_noise) / max(abs(on_structure), abs(on_noise), 1e-12)
    return spread > 0.01, f"structure {on_structure:.4g} vs noise {on_noise:.4g}"


CHECKS = (("raises", check_raises), ("stable", check_stable), ("null", check_null))


def admit(name: str, fn: Callable) -> Verdict:
    """Run the gate on one scalar-returning callable."""
    v = Verdict(name=name, verdict="admitted")
    for label, check in CHECKS:
        passed, note = check(fn)
        v.checks[label] = passed
        if note:
            v.notes.append(f"{label}: {note}")
    if not v.checks.get("raises"):
        v.verdict = "rejected"          # a tool that never fails cannot be believed
    elif not all(v.checks.values()):
        v.verdict = "flagged"
    return v


def admit_metrics(names: Optional[list] = None) -> list[Verdict]:
    """Run the gate over the declared suite, against manylatents' live registry.

    DEFERS rather than fakes when the engine is absent — the same discipline as
    `catalog.check_suite`, because a gate that passes everything on a stackless machine is
    worse than no gate."""
    from manyruns import catalog

    names = list(names or catalog.load_suite())
    try:
        from manylatents.metrics.registry import get_metric
    except Exception:  # noqa: BLE001 - stackless: defer, never fake
        return [Verdict(name=n, verdict="deferred",
                        notes=["manylatents not importable — not checked here"])
                for n in names]

    out = []
    for name in names:
        try:
            metric = get_metric(name)
        except Exception as e:  # noqa: BLE001
            out.append(Verdict(name=name, verdict="rejected",
                               notes=[f"not in the registry: {type(e).__name__}"]))
            continue

        def scalar(X: Any, _m: Any = metric) -> Any:
            result = _m(X)
            if isinstance(result, dict):        # several metrics return a dict of scalars
                result = next((v for v in result.values()
                               if isinstance(v, (int, float))), float("nan"))
            return float(result)

        out.append(admit(name, scalar))
    return out
