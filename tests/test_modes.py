"""The mode-router seam (manyruns.modes): train / eval / infer dispatch + device threading.

Dep-free: infer/eval run over the mock recipe path; train returns its delegation descriptor
without executing (no engine here). Proves the *routing*, not the compute."""
from __future__ import annotations

import pytest

from manyruns import modes

_RECIPE = {"name": "t", "steps": [{"kind": "module", "name": "phate", "group": "latent"},
                                  {"kind": "metric", "name": "separation", "group": "analysis"}]}


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        modes.run("finetune")


def test_infer_threads_a_REAL_device_not_an_echo(monkeypatch):
    # device is now resolved by the compute path, not echoed. Pin availability so the assert is
    # env-independent: with no CUDA/MPS present, a requested "mps" resolves to cpu with a reason.
    from manyruns import devices

    monkeypatch.setattr(devices, "_available_devices", lambda: {"cuda": False, "mps": False})
    res = modes.run("infer", {"recipe": _RECIPE, "engine": "mock"}, device="mps")
    assert res["mode"] == "infer"
    assert res["device"] == "cpu"                       # NOT "mps" — the echo is gone
    assert "mps" in res["device_note"]                  # honest reason it isn't mps
    assert res["trace"] == ["latent:phate", "analysis:separation"]
    assert "separation" in res["g_vector"]


def test_infer_requires_a_recipe():
    with pytest.raises(ValueError, match="needs a 'recipe'"):
        modes.run("infer", {"engine": "mock"})


def test_eval_reads_an_infer_result():
    infer = modes.run("infer", {"recipe": _RECIPE, "engine": "mock"})
    ev = modes.run("eval", {"result": infer}, device="cpu")
    # eval reflects the device the evaluated run actually used (whatever this env resolved to)
    assert ev["mode"] == "eval" and ev["device"] == infer["device"]
    assert ev["metrics"] == infer["g_vector"]


def test_eval_without_a_result_runs_infer_itself():
    ev = modes.run("eval", {"recipe": _RECIPE, "engine": "mock"})
    assert ev["mode"] == "eval" and "separation" in ev["metrics"]


def test_train_is_a_delegation_descriptor_not_executed():
    plan = modes.run("train", {"core_snapshot": "treated-condition@v0", "model_version": "treated-core@v1"},
                     device="cuda")
    assert plan["mode"] == "train" and plan["device"] == "cuda"
    assert plan["model_version_out"] == "treated-core@v1"
    # "planned", not "delegated". The word changed with the removal and the change is the
    # point: there is no longer anyone to delegate TO. manyruns records the ask and stops at
    # the store, which the learner reads (the environment-contract spec, §3.5).
    assert "planned" in plan["status"]
    assert "result" not in plan                    # nothing executed, and nothing to execute


# ── version routing: pull between registered model versions ──────────────────
def test_infer_pulls_between_recipe_versions(tmp_path):
    from manyruns import registry

    registry.register("treated-core", "v0", "recipe", {"recipe": "embed"}, root=tmp_path)
    registry.register("treated-core", "v1", "recipe", {"recipe": "contrast"}, root=tmp_path)
    common = {"model": "treated-core", "engine": "mock", "registry_root": tmp_path}
    v0 = modes.run("infer", {**common, "version": "v0"})
    v1 = modes.run("infer", {**common, "version": "v1"})
    assert v0["model_version"] == "treated-core@v0" and v0["recipe"] == "embed"
    assert v1["model_version"] == "treated-core@v1" and v1["recipe"] == "contrast"
    assert v0["trace"] != v1["trace"]          # pulling a different version changes the run


def test_infer_weights_version_applies_the_trained_weights(tmp_path, monkeypatch):
    import numpy as np

    from manyruns import registry

    registry.register("treated-core", "v2", "weights",
                      {"weights": "/w.pt", "input_dim": 5, "hidden_dim": 16}, root=tmp_path)
    # stack-gated apply_weights faked (real one loads torch/manylatents) — proves the routing
    monkeypatch.setattr(registry, "apply_weights", lambda v, X: np.zeros((3, 5)))
    res = modes.run("infer", {"model": "treated-core", "version": "v2",
                              "array": np.ones((3, 5)), "registry_root": tmp_path})
    assert res["kind"] == "weights" and res["model_version"] == "treated-core@v2"
    assert res["embedding"].shape == (3, 5)
