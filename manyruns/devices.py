"""Device resolution — make the dispatcher's device choice REAL, not cosmetic.

The mode-router / substrate dispatcher pick a device (cpu / cuda / mps); this resolves the
device compute will **actually** use, honoring the one hard constraint manyruns's CFlows
recipe carries: adaptive neural-ODE solvers (torchdiffeq `dopri5`, inside MIOFlow) need
**float64**, and Apple MPS has none. So an fp64 workload is never placed on MPS — it is
*resolved away* (to cpu or cuda) with a note, never silently echoed. This is the single source
of the fp64/device rule; the pipeline Trainer derives its accelerator from the resolved device
(via `accelerator_for`) instead of hard-pinning cpu. Fixes the audit's "device_override is
cosmetic" / "substrate router unwired" gaps.
"""
from __future__ import annotations

from typing import Optional


def _available_devices() -> dict:
    """``{'cuda': bool, 'mps': bool}`` via torch (cpu is always available). torch-free → all
    False, so a base/CI install resolves everything to cpu. Monkeypatched in tests."""
    try:
        import torch

        mps = getattr(torch.backends, "mps", None)
        return {
            "cuda": bool(torch.cuda.is_available()),
            "mps": bool(mps and torch.backends.mps.is_available()),
        }
    except Exception:  # noqa: BLE001 - torch absent (base install / CI)
        return {"cuda": False, "mps": False}


def recipe_requires_fp64(recipe: Optional[dict]) -> bool:
    """True if the recipe runs a neural-ODE step (MIOFlow, ``group: lightning``) → needs float64."""
    if not recipe:
        return False
    return any(
        s.get("group") == "lightning" or s.get("name") == "mioflow"
        for s in recipe.get("steps", [])
    )


def resolve_device(requested: Optional[str] = None, *, requires_fp64: bool = False) -> tuple[str, str]:
    """Return ``(device, note)`` — the device compute will ACTUALLY use.

    Never echoes a device it can't honor: an fp64 workload on MPS resolves to cpu/cuda with a
    reason; a requested device that isn't present falls back to cpu with a reason. ``note`` is
    the honest explanation to surface (so "device: cuda" can never secretly mean "ran on cpu").
    """
    avail = _available_devices()
    if requested:
        r = requested.lower()
        if r == "mps" and requires_fp64:
            return _fp64_off_mps(avail, "mps requested, but this workload needs float64 (MPS has none)")
        if r == "cuda":
            return ("cuda", "cuda") if avail["cuda"] else ("cpu", "cuda requested but no CUDA device → cpu")
        if r == "mps":
            return ("mps", "mps") if avail["mps"] else ("cpu", "mps requested but unavailable → cpu")
        if r == "cpu":
            return "cpu", "cpu"
        return "cpu", f"unknown device {requested!r} → cpu"
    # auto-select
    if avail["cuda"]:
        return "cuda", "auto: CUDA"
    if avail["mps"] and requires_fp64:
        return "cpu", "auto: cpu (fp64 workload excludes MPS)"
    if avail["mps"]:
        return "mps", "auto: Apple MPS"
    return "cpu", "auto: cpu"


def _fp64_off_mps(avail: dict, why: str) -> tuple[str, str]:
    return ("cuda", f"{why} → cuda") if avail["cuda"] else ("cpu", f"{why} → cpu")


def accelerator_for(device: Optional[str]) -> str:
    """Lightning ``accelerator`` for a device. MIOFlow never runs on MPS (fp64) → map mps to cpu."""
    return "gpu" if (device or "cpu").lower() == "cuda" else "cpu"
