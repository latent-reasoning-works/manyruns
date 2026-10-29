"""`narrate.declared_observation`: the Observation a dataset declaration states, no file read.

One construction, called by `tui.state.catalog_entry`, `shell._resolve_dataset` and
`manyruns actions`, so that no two surfaces disagree about what a declaration says.
"""
from __future__ import annotations

import builtins
import pathlib

from manyruns import catalog, narrate, shell


def test_pbmc3k_declaration():
    assert catalog.dataset_dir().is_dir()
    obs = narrate.declared_observation(catalog.load_dataset("pbmc3k"), "pbmc3k")
    assert obs.shape == "clusters"
    assert obs.modality == "scrna"
    assert obs.source == "pbmc3k"
    assert obs.declared == ("genes",)
    assert obs.provides() == frozenset({"genes"})


def test_missing_shape_and_modality_read_unknown():
    obs = narrate.declared_observation(
        {"name": "x", "handle": {"kind": "manylatents", "ref": "x"}}, "x")
    assert obs.shape == "unknown"
    assert obs.modality == "unknown"
    assert obs.declared == ()


def test_opens_no_file(monkeypatch):
    assert catalog.dataset_dir().is_dir()
    ds = catalog.load_dataset("pbmc3k")

    def refuse(*_a, **_k):
        raise AssertionError("declared_observation opened a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(pathlib.Path, "open", refuse)
    obs = narrate.declared_observation(ds, "pbmc3k")
    assert obs.source == "pbmc3k"


def test_shell_resolve_dataset_agrees_with_the_helper():
    assert catalog.dataset_dir().is_dir()
    assert shell._resolve_dataset("swissroll")[3] == narrate.declared_observation(
        catalog.load_dataset("swissroll"), "swissroll")
