"""Unit tests for the Core assembler (manyruns.core).

Pure logic only — snapshot pathing, the member/snapshot data model, dedup, required-obs
validation, immutable-snapshot behavior, and JSON round-trip. No anndata/scanpy/manylatents
(the one heavy edge, `inspect_member`, opens a `.h5ad` and is not exercised here), so this
runs in the product-layer CI.
"""
from __future__ import annotations

import json

import pytest

from manyruns import core


def _member(acc: str, **over) -> core.CoreMember:
    base = dict(
        accession=acc,
        path=f"/data/single_cell/{acc}.h5ad",
        n_obs=100,
        n_vars=2000,
        obs_vocab={"sample": ["s1", "s2"], "disease": ["treated", "healthy"]},
        source="geo",
        tissue="sample material",
        disease="treated",
    )
    base.update(over)
    return core.CoreMember(**base)


# ── on-disk home ─────────────────────────────────────────────────────────────
def test_reference_library_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYRUNS_EXPERIMENTS", str(tmp_path))
    assert core.reference_library_root() == tmp_path / "reference_library"


def test_snapshot_dir_shape(tmp_path):
    d = core.snapshot_dir("dataset-core", 0, root=tmp_path)
    assert d == tmp_path / "dataset-core@v0"


# ── data model round-trip ────────────────────────────────────────────────────
def test_member_roundtrip_ignores_unknown_keys():
    m = _member("GSE000001")
    d = m.to_dict()
    d["future_column"] = "ignored"  # forward-compat: unknown keys don't break from_dict
    assert core.CoreMember.from_dict(d) == m


def test_snapshot_roundtrip():
    snap = core.CoreSnapshot(
        name="dataset-core", version=0, members=[_member("A"), _member("B")],
        provenance={"created": "2026-07-08T00:00:00", "n_members": 2},
    )
    restored = core.CoreSnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
    assert restored == snap


# ── pure operations ──────────────────────────────────────────────────────────
def test_dedupe_by_accession_keeps_first_case_insensitive():
    members = [_member("GSE1"), _member("gse1", n_obs=999), _member("GSE2")]
    out = core.dedupe_by_accession(members)
    assert [m.accession for m in out] == ["GSE1", "GSE2"]
    assert out[0].n_obs == 100  # first kept, not the later dup


def test_missing_required_obs_detected():
    ok = _member("A")
    bad = _member("B", obs_vocab={"sample": ["s1"]})  # no 'disease'
    assert ok.missing_required_obs() == []
    assert bad.missing_required_obs() == ["disease"]


# ── assembly ─────────────────────────────────────────────────────────────────
def test_assemble_core_writes_and_loads(tmp_path):
    out = core.assemble_core(
        [_member("GSE000001"), _member("GSE000003")],
        root=tmp_path, created="2026-07-08T00:00:00", tool_version="test",
    )
    assert out == tmp_path / "dataset-core@v0"
    snap = core.load_snapshot(out)
    assert [m.accession for m in snap.members] == ["GSE000001", "GSE000003"]
    assert snap.provenance["n_members"] == 2
    assert snap.provenance["created"] == "2026-07-08T00:00:00"
    assert snap.provenance["canonical_obs"]["required"] == ["sample", "disease"]
    # load_snapshot also accepts the json file directly
    assert core.load_snapshot(out / "snapshot.json") == snap


def test_assemble_core_refuses_missing_required_obs(tmp_path):
    with pytest.raises(ValueError, match="missing required obs"):
        core.assemble_core([_member("B", obs_vocab={"sample": ["s1"]})], root=tmp_path)


def test_assemble_core_rejects_empty(tmp_path):
    with pytest.raises(ValueError, match="empty Core"):
        core.assemble_core([], root=tmp_path)


def test_assemble_core_immutable_unless_force(tmp_path):
    args = dict(root=tmp_path, created="t", tool_version="test")
    core.assemble_core([_member("A")], **args)
    with pytest.raises(FileExistsError, match="immutable"):
        core.assemble_core([_member("A")], **args)
    # force overwrites, and a new version is always a fresh dir
    core.assemble_core([_member("A")], force=True, **args)
    core.assemble_core([_member("A")], version=1, **args)
    assert (tmp_path / "dataset-core@v1" / "snapshot.json").exists()


def test_import_is_clean_of_heavy_stack():
    # In a FRESH interpreter (so other tests' importorskip can't pollute sys.modules),
    # importing manyruns.core must not pull anndata/scanpy/manylatents — they are the
    # lazy edges. Deterministic regardless of test order or what's installed.
    import os
    import subprocess
    import sys

    code = (
        "import manyruns.core, sys; "
        "leaked=[m for m in ('anndata','scanpy','manylatents') if m in sys.modules]; "
        "assert not leaked, leaked"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
