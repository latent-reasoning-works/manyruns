"""A model manifest is a tool manifest with `output.schema: result-v1` and three more fields."""
from __future__ import annotations

from pathlib import Path

import pytest

from manyruns import toolchain
from model_seam_fixtures import GOLDEN

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tools" / "mioflow"


def _read(path: Path) -> dict:
    from omegaconf import OmegaConf

    assert path.parent.is_dir(), f"{path.parent} moved"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _golden(name: str) -> dict:
    return _read(GOLDEN / f"{name}.yaml")


def test_missing_commercial_use_is_refused():
    cfg = _golden("golden")
    del cfg["commercial_use"]
    bad = toolchain.check_tool(cfg, "golden")
    assert len(bad) == 1 and "commercial_use" in bad[0], bad


def test_missing_licence_is_refused():
    cfg = _golden("golden")
    del cfg["licence"]
    bad = toolchain.check_tool(cfg, "golden")
    assert len(bad) == 1 and "licence" in bad[0], bad


def test_non_commercial_licence_cannot_claim_commercial_use():
    cfg = _golden("golden_precomputed")
    cfg["commercial_use"] = True
    bad = toolchain.check_tool(cfg, "golden_precomputed")
    assert len(bad) == 1 and "CC-BY-NC-4.0" in bad[0], bad


def test_golden_manifests_are_valid():
    for name in ("golden", "golden_precomputed"):
        assert toolchain.check_tool(_golden(name), name) == [], name


def test_precomputed_manifest_needs_no_input_kind():
    cfg = _golden("golden_precomputed")
    assert "input" not in cfg
    bad = toolchain.check_tool(cfg, "golden_precomputed")
    assert bad == [] and not any("input.kind" in p for p in bad)


def test_precomputed_manifest_with_adapter_is_refused():
    cfg = _golden("golden_precomputed")
    cfg["entrypoint"] = "x.py"
    bad = toolchain.check_tool(cfg, "golden_precomputed")
    assert len(bad) == 1 and "entrypoint" in bad[0] and "precomputed" in bad[0], bad


def test_model_manifest_cannot_produce_velocity():
    cfg = _golden("golden")
    cfg["produces"] = ["velocity"]
    bad = toolchain.check_tool(cfg, "golden")
    assert len(bad) == 1 and "velocity" in bad[0], bad


def test_distribution_satisfies_pin_rule():
    cfg = _golden("golden")
    assert (cfg["name"], cfg["distribution"], cfg["requires"]) == (
        "golden", "numpy", ["numpy==2.2.6"])
    assert toolchain.check_tool(cfg, "golden") == []
    del cfg["distribution"]
    bad = toolchain.check_tool(cfg, "golden")
    assert len(bad) == 1 and "golden" in bad[0] and "pin" in bad[0], bad


@pytest.mark.source_checkout
def test_range_is_not_a_pin():
    cfg = _read(EXAMPLE / "mioflow.yaml")
    cfg["requires"] = ["mioflow>=0.1.14"]
    bad = toolchain.check_tool(cfg, "mioflow")
    assert len(bad) == 1 and "mioflow>=0.1.14" in bad[0] and "pin" in bad[0], bad


@pytest.mark.source_checkout
def test_mioflow_example_manifest_is_valid():
    assert toolchain.check_tool(_read(EXAMPLE / "mioflow.yaml"), "mioflow") == []


def test_golden_call_manifests_are_valid():
    for name in ("golden_call_svd", "golden_call_class"):
        assert toolchain.check_tool(_golden(name), name) == [], name


def test_call_mode_selftest_defaults_to_importing_the_target_module():
    cfg = _golden("golden_call_class")
    assert "selftest" not in cfg
    assert toolchain.selftest_command(cfg) == "import call_target"


def test_unknown_slot_is_refused():
    cfg = _golden("golden_call_svd")
    call = dict(cfg["call"], args=["counts"])
    bad = toolchain.check_call(call, cfg["produces"])
    assert len(bad) == 1 and "counts" in bad[0], bad


def test_instance_selector_needs_method():
    cfg = _golden("golden_call_svd")
    call = dict(cfg["call"], outputs={"embedding": "instance.components_"})
    bad = toolchain.check_call(call, cfg["produces"])
    assert len(bad) == 1 and "instance" in bad[0], bad


def test_graph_cannot_be_selected():
    call = {"target": "numpy.linalg:svd", "args": ["matrix"], "outputs": {"graph": "return"}}
    bad = toolchain.check_call(call, ["graph"])
    assert len(bad) == 1 and "graph" in bad[0], bad
