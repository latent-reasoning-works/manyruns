"""Tests for the Core ↔ Session glue (manyruns.explore_core).

Dep-free: builds a real Core snapshot from hand-constructed members (no anndata — the
Core snapshot is plain JSON), then drives the CFlows recipe over it via the **mock**
engine (which needs no data read and no private stack). This validates the convergence
seam end-to-end without network, scanpy, or manylatents.
"""
from __future__ import annotations

import pytest

from manyruns import core, explore_core


def _snapshot(tmp_path):
    member = core.CoreMember(
        accession="GSE000001",
        path=str(tmp_path / "GSE000001.h5ad"),
        n_obs=100,
        n_vars=2000,
        obs_vocab={"sample": ["s1", "s2"], "disease": ["treated", "healthy"]},
        source="geo",
        tissue="sample material",
        disease="condition",
    )
    core.assemble_core(
        [member], name="dataset-core", version=0, root=tmp_path,
        created="2026-07-10T00:00:00", tool_version="test",
    )
    return tmp_path / "dataset-core@v0"


def test_read_core_returns_member_metadata(tmp_path):
    info = explore_core.read_core(_snapshot(tmp_path))
    assert info["name"] == "dataset-core" and info["version"] == 0
    m = info["members"][0]
    assert m["accession"] == "GSE000001"
    assert m["obs_vocab"]["disease"] == ["treated", "healthy"]
    assert m["tissue"] == "sample material"


def test_run_geometry_mock_runs_full_cflows_trace(tmp_path):
    # `recipe=` is forced: this test is about a recipe running IN ORDER over a Core-bound
    # session, not about which recipe gets auto-selected. The stub member carries no readable
    # time axis, so auto-selection now (correctly, #26) yields `embed` — a one-step trace that
    # would exercise none of what this test exists to check.
    res = explore_core.run_geometry(
        _snapshot(tmp_path), "GSE000001", "treated-v0", engine="mock", out_dir=tmp_path,
        recipe="cflows",
    )
    assert res["engine"] == "mock"
    assert res["recipe"] == "cflows"
    # The CFlows recipe ran, in order, over the Core-bound session. Was
    # `[latent:phate, lightning:mioflow]` / 2 steps; the three leading steps are the preamble
    # `loading._anndata_matrix` used to apply invisibly (normalize_total(1e4) → log1p →
    # PCA(50)), declared in the recipe as of #54, so the trace is 5 long.
    assert res["trace"] == ["prep:normalize", "prep:transform",
                            "latent:pca", "latent:phate", "lightning:mioflow"]
    assert res["num_steps"] == 5


def test_run_geometry_unknown_accession_raises(tmp_path):
    with pytest.raises(KeyError, match="GSE999"):
        explore_core.run_geometry(_snapshot(tmp_path), "GSE999", "x", engine="mock")
