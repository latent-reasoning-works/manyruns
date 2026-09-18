"""Device resolution — the single source of the fp64/MPS rule (manyruns.devices).

Dep-free: monkeypatches device availability, so the fp64 exclusion + fallbacks are tested
without torch/a GPU."""
from __future__ import annotations

import pytest

from manyruns import devices


@pytest.fixture
def avail(monkeypatch):
    def _set(cuda=False, mps=False):
        monkeypatch.setattr(devices, "_available_devices", lambda: {"cuda": cuda, "mps": mps})
    return _set


# ── the fp64 / MPS rule (the load-bearing part) ──────────────────────────────
def test_fp64_workload_never_lands_on_mps(avail):
    avail(cuda=False, mps=True)
    dev, note = devices.resolve_device("mps", requires_fp64=True)
    assert dev == "cpu" and "float64" in note              # resolved away, not echoed

    dev, note = devices.resolve_device(None, requires_fp64=True)   # auto
    assert dev == "cpu" and "excludes MPS" in note


def test_fp64_prefers_cuda_over_cpu_when_present(avail):
    avail(cuda=True, mps=True)
    assert devices.resolve_device("mps", requires_fp64=True)[0] == "cuda"
    assert devices.resolve_device(None, requires_fp64=True)[0] == "cuda"


def test_non_fp64_can_use_mps(avail):
    avail(cuda=False, mps=True)
    assert devices.resolve_device(None, requires_fp64=False)[0] == "mps"


# ── requested-device honesty (no silent echo) ────────────────────────────────
def test_requested_device_absent_falls_back_to_cpu(avail):
    avail(cuda=False, mps=False)
    dev, note = devices.resolve_device("cuda")
    assert dev == "cpu" and "cuda requested" in note       # not a silent "cuda"


def test_auto_prefers_cuda(avail):
    avail(cuda=True, mps=False)
    assert devices.resolve_device()[0] == "cuda"


# ── recipe fp64 detection + accelerator mapping ──────────────────────────────
def test_recipe_requires_fp64_only_for_mioflow():
    cflows = {"steps": [{"name": "phate", "group": "latent"},
                        {"name": "mioflow", "group": "lightning"}]}
    contrast = {"steps": [{"name": "phate", "group": "latent"},
                          {"name": "separation", "group": "analysis"}]}
    assert devices.recipe_requires_fp64(cflows) is True
    assert devices.recipe_requires_fp64(contrast) is False
    assert devices.recipe_requires_fp64(None) is False


def test_accelerator_for_never_mps():
    assert devices.accelerator_for("cuda") == "gpu"
    assert devices.accelerator_for("mps") == "cpu"     # MIOFlow never on MPS
    assert devices.accelerator_for("cpu") == "cpu"
    assert devices.accelerator_for(None) == "cpu"
