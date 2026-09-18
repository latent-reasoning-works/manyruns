"""Acquire-read edges that need the REAL stack (anndata/scanpy) — the `[harness]` extra.

Separate from test_data_acquire.py, which is deliberately stack-free so it runs in the
product-layer CI. These skip there and run wherever the harness extra is installed.

They cover the silent-corruption class: bugs that produce a plausible-looking matrix and
would never fail a smoke test — which is exactly why they need a fixture with real bytes.
"""
from __future__ import annotations

import pytest

from manyruns.harness import data_acquire as da

ad = pytest.importorskip("anndata", reason="needs the [harness] extra (anndata)")
pytest.importorskip("scanpy", reason="needs the [harness] extra (scanpy)")  # _read_matrix
np = pytest.importorskip("numpy")


def _write_h5ad(path, n_obs=4, n_vars=3, sample=None):
    a = ad.AnnData(np.arange(n_obs * n_vars, dtype="float32").reshape(n_obs, n_vars))
    a.obs_names = [f"cell{i}" for i in range(n_obs)]
    a.var_names = [f"gene{j}" for j in range(n_vars)]
    if sample is not None:
        a.obs["sample"] = sample
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)
    return path


def test_convert_preserves_existing_sample_labels(tmp_path):
    """C3 regression — convert() must NEVER clobber a collaborator's own `sample`.

    It previously did `adata.obs["sample"] = m.sample` unconditionally, replacing real
    donor IDs with the filename stem. That destroys the required provenance AND the batch
    axis, and it defeats core._assert_required_obs — a single-valued column still "has"
    the key, so the gate greens on the very data it exists to catch.
    """
    src = tmp_path / "raw"
    _write_h5ad(src / "processed.h5ad", n_obs=4, sample=["donorA", "donorA", "donorB", "donorB"])

    out = da.convert(src, "TEST", out_dir=tmp_path / "out")
    got = list(ad.read_h5ad(out).obs["sample"])

    assert got == ["donorA", "donorA", "donorB", "donorB"], (
        f"convert() clobbered the donor labels with the filename stem: {got}"
    )
    assert "processed" not in got


def test_convert_sets_sample_when_absent(tmp_path):
    """The documented behavior (mvp-workflow.md:161): set `sample` *if absent*."""
    src = tmp_path / "raw"
    _write_h5ad(src / "cohort.h5ad", n_obs=3, sample=None)

    out = da.convert(src, "TEST2", out_dir=tmp_path / "out")
    assert set(ad.read_h5ad(out).obs["sample"]) == {"cohort"}  # from the filename stem


def test_convert_keeps_sample_as_the_batch_axis_across_parts(tmp_path):
    """Fusion invariant in miniature: every row keeps the sample it came from.

    Two parts with their own donor labels must concat WITHOUT either being flattened —
    this is the property the orchestrator's row-alignment guarantee is built on.
    """
    src = tmp_path / "raw"
    _write_h5ad(src / "a.h5ad", n_obs=2, sample=["donorA", "donorA"])
    _write_h5ad(src / "b.h5ad", n_obs=2, sample=["donorB", "donorB"])

    obs = ad.read_h5ad(da.convert(src, "TEST3", out_dir=tmp_path / "out")).obs
    assert set(obs["sample"]) == {"donorA", "donorB"}, "the batch axis collapsed on concat"
    assert len(obs) == 4


def test_acquire_with_explicit_manifest_end_to_end(tmp_path, synthetic_manifest, monkeypatch):
    """A local archive exercises download, extraction, conversion and the manifest URL."""
    import csv
    import tarfile

    monkeypatch.setenv("MANYRUNS_ACQUIRE_MANIFEST", str(tmp_path / "missing.tsv"))
    source = _write_h5ad(tmp_path / "input" / "matrix.h5ad", sample=["a", "a", "b", "b"])
    archive = tmp_path / "input.tar"
    with tarfile.open(archive, "w") as bundle:
        bundle.add(source, arcname="matrix.h5ad")
    with synthetic_manifest.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = reader.fieldnames
        row = next(reader)
    row["url"] = archive.as_uri()
    manifest = tmp_path / "manifest.tsv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    result = da.acquire(row["accession"], manifest=manifest,
                        work_dir=tmp_path / "work", out_dir=tmp_path / "out")

    assert result == tmp_path / "out" / f"{row['accession']}.h5ad"
    actual = ad.read_h5ad(result)
    expected = ad.read_h5ad(source)
    np.testing.assert_array_equal(actual.X, expected.X)
    assert list(actual.obs_names) == list(expected.obs_names)
    assert list(actual.var_names) == list(expected.var_names)
    assert list(actual.obs["sample"]) == ["a", "a", "b", "b"]
    assert (tmp_path / "work" / f"{row['accession']}_RAW.tar").is_file()
