"""The model-version registry (manyruns.registry) — dep-free register/resolve/list.

`apply_weights` (trained-weight inference) is stack-gated and not exercised here."""
from __future__ import annotations

import pytest

from manyruns import registry


def test_register_resolve_roundtrip(tmp_path):
    registry.register("treated-core", "v0", "recipe", {"recipe": "cflows"}, root=tmp_path, created="t")
    v = registry.resolve("treated-core", "v0", root=tmp_path)
    assert (v.model, v.version, v.kind) == ("treated-core", "v0", "recipe")
    assert v.spec["recipe"] == "cflows" and v.created == "t"


def test_versions_are_immutable_unless_forced(tmp_path):
    registry.register("m", "v0", "recipe", {"recipe": "embed"}, root=tmp_path)
    with pytest.raises(FileExistsError, match="immutable"):
        registry.register("m", "v0", "recipe", {"recipe": "cflows"}, root=tmp_path)
    registry.register("m", "v0", "recipe", {"recipe": "cflows"}, root=tmp_path, force=True)
    assert registry.resolve("m", "v0", root=tmp_path).spec["recipe"] == "cflows"


def test_list_versions_sorted_across_kinds(tmp_path):
    registry.register("m", "v1", "weights", {"weights": "/w.pt"}, root=tmp_path)
    registry.register("m", "v0", "recipe", {"recipe": "cflows"}, root=tmp_path)
    vs = registry.list_versions("m", root=tmp_path)
    assert [v.version for v in vs] == ["v0", "v1"]
    assert {v.kind for v in vs} == {"recipe", "weights"}


def test_resolve_unknown_raises(tmp_path):
    with pytest.raises(KeyError):
        registry.resolve("m", "vX", root=tmp_path)


def test_bad_kind_rejected(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        registry.register("m", "v0", "bogus", root=tmp_path)
