"""Unit tests for the acquire-read shim (manyruns.harness.data_acquire).

Pure logic only — manifest parsing, matrix location, path/URL builders, config
emission. Deliberately exercises NO scanpy / manylatents / network, so it runs in
the product-layer CI (which installs neither). The scanpy/manylatents edges are
imported lazily inside their functions and are not touched here.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from manyruns.harness import data_acquire as da


# ── manifest ─────────────────────────────────────────────────────────────────
def test_manifest_loads_synthetic_rows(synthetic_manifest):
    entries = da.load_manifest(synthetic_manifest)
    assert len(entries) == 6
    assert all(e.accession and e.source and e.url for e in entries)
    # every scrna row carries a concrete matrix format
    scrna = [e for e in entries if e.modality == "scrna"]
    assert scrna and all(e.fmt in {"mtx", "txt", "tenx_h5"} for e in scrna)


def test_manifest_meta_carries_extra_columns(synthetic_manifest):
    entry = da.get_entry("GSE000001", synthetic_manifest)
    assert entry.source == "geo"
    assert entry.fmt == "mtx"
    assert entry.meta["tissue"] == "sample material"
    assert "note" in entry.meta


def test_get_entry_case_insensitive_and_unknown(synthetic_manifest):
    assert da.get_entry("gse000001", synthetic_manifest).accession == "GSE000001"
    with pytest.raises(KeyError):
        da.get_entry("GSE000000", synthetic_manifest)


def test_manifest_env_override(tmp_path, monkeypatch):
    custom = tmp_path / "m.tsv"
    custom.write_text(
        "accession\tsource\tmodality\tformat\turl\n"
        "X1\tgeo\tscrna\tmtx\thttp://example/x1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MANYRUNS_ACQUIRE_MANIFEST", str(custom))
    entries = da.load_manifest()
    assert [e.accession for e in entries] == ["X1"]


def test_missing_manifest_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYRUNS_ACQUIRE_MANIFEST", str(tmp_path / "nope.tsv"))
    with pytest.raises(FileNotFoundError):
        da.load_manifest()


# ── URL builders ─────────────────────────────────────────────────────────────
def test_url_builders():
    assert da.geo_supplementary_url("GSE000001") == (
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE000001&format=file"
    )
    assert da.arrayexpress_files_url("E-MTAB-0002") == (
        "https://www.ebi.ac.uk/biostudies/files/E-MTAB-0002/"
    )


# ── matrix location (the pure ETL core) ──────────────────────────────────────
def _touch(p: Path, gz: bool = False) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if gz:
        with gzip.open(p, "wb") as f:
            f.write(b"")
    else:
        p.write_bytes(b"")


def test_find_matrices_flat_geo_triplets(tmp_path):
    # GEO ships flat GSM-prefixed, gzipped 10x triplets.
    for gsm in ("GSM100_", "GSM101_"):
        _touch(tmp_path / f"{gsm}matrix.mtx.gz", gz=True)
        _touch(tmp_path / f"{gsm}barcodes.tsv.gz", gz=True)
        _touch(tmp_path / f"{gsm}features.tsv.gz", gz=True)

    found = da.find_matrices(tmp_path)
    assert [m.kind for m in found] == ["mtx", "mtx"]
    assert {m.sample for m in found} == {"GSM100", "GSM101"}
    assert all(m.prefix.endswith("_") and m.path == tmp_path for m in found)


def test_find_matrices_prefers_structured_over_text(tmp_path):
    # A triplet's own barcodes/features .tsv must NOT be picked up as a text matrix.
    _touch(tmp_path / "matrix.mtx.gz", gz=True)
    _touch(tmp_path / "barcodes.tsv.gz", gz=True)
    _touch(tmp_path / "features.tsv.gz", gz=True)
    found = da.find_matrices(tmp_path)
    assert [m.kind for m in found] == ["mtx"]


def test_find_matrices_h5_and_h5ad(tmp_path):
    _touch(tmp_path / "filtered_feature_bc_matrix.h5")
    _touch(tmp_path / "processed.h5ad")
    kinds = {m.kind for m in da.find_matrices(tmp_path)}
    assert kinds == {"tenx_h5", "h5ad"}


def test_find_matrices_text_fallback(tmp_path):
    _touch(tmp_path / "GSE000002_counts.txt.gz", gz=True)
    _touch(tmp_path / "GSE000002_series_matrix.txt.gz", gz=True)  # must be ignored
    found = da.find_matrices(tmp_path)
    assert [m.kind for m in found] == ["txt"]
    assert "series_matrix" not in found[0].path.name


# ── path contract + config emission ──────────────────────────────────────────
def test_canonical_h5ad_path_with_explicit_out_dir(tmp_path):
    p = da.canonical_h5ad_path("GSE000001", out_dir=tmp_path)
    assert p == tmp_path / "GSE000001.h5ad"


def test_default_out_dir_without_omics_raises(monkeypatch):
    # When the omics data-root can't be imported, the resolver must fail loudly with
    # actionable guidance, not guess a path. Force the import to fail deterministically
    # (setting the module to None makes `from ... import` raise ImportError).
    import sys

    monkeypatch.setitem(sys.modules, "manylatents._data_paths", None)
    with pytest.raises(RuntimeError, match="out_dir"):
        da.default_out_dir()


def test_acquire_forwards_orientation_to_convert(tmp_path, monkeypatch, synthetic_manifest):
    """C2 regression — the orientation must reach convert(), not be hardcoded.

    A cells-by-genes table must retain its orientation through acquisition.
    Pure logic: fetch/convert are stubbed, no scanpy touched.
    """
    seen = {}

    def fake_convert(root, accession, **kw):
        seen.update(kw)
        return tmp_path / f"{accession}.h5ad"

    monkeypatch.setattr(da, "fetch", lambda acc, wd, **kw: tmp_path)
    monkeypatch.setattr(da, "convert", fake_convert)
    da.acquire("GSE000002", manifest=synthetic_manifest, out_dir=tmp_path, genes_are_rows=False)
    assert seen["genes_are_rows"] is False, "acquire() dropped the orientation on the floor"


def test_write_config_uses_omics_ref_and_target(tmp_path):
    h5ad = tmp_path / "single_cell" / "GSE000001.h5ad"
    h5ad.parent.mkdir(parents=True)
    h5ad.write_bytes(b"")
    cfg = da.write_config("GSE000001", h5ad, tmp_path / "configs")
    text = cfg.read_text()
    assert cfg.name == "GSE000001.yaml"
    assert "_target_: manylatents.singlecell.data.anndata.AnnDataModule" in text
    assert "adata_path: ${omics_data:}/single_cell/GSE000001.h5ad" in text
    assert "label_key: null" in text


def test_write_config_literal_path_when_not_under_single_cell(tmp_path):
    h5ad = tmp_path / "GSE000001.h5ad"
    h5ad.write_bytes(b"")
    cfg = da.write_config("GSE000001", h5ad, tmp_path / "configs", label_key="cell_type")
    text = cfg.read_text()
    assert f"adata_path: {h5ad}" in text
    assert "label_key: 'cell_type'" in text


@pytest.mark.parametrize("value", [None, ""])
def test_manifest_is_required(monkeypatch, tmp_path, value):
    if value is None:
        monkeypatch.delenv("MANYRUNS_ACQUIRE_MANIFEST", raising=False)
    else:
        monkeypatch.setenv("MANYRUNS_ACQUIRE_MANIFEST", value)
    for call in (lambda: da.load_manifest(),
                 lambda: da.acquire("TEST", out_dir=tmp_path, work_dir=tmp_path)):
        with pytest.raises(FileNotFoundError, match="Pass an explicit manifest path.*MANYRUNS_ACQUIRE_MANIFEST"):
            call()


def test_explicit_manifest_takes_precedence(synthetic_manifest, tmp_path, monkeypatch):
    monkeypatch.setenv("MANYRUNS_ACQUIRE_MANIFEST", str(tmp_path / "missing.tsv"))
    assert da.load_manifest(synthetic_manifest)[0].accession == "GSE000001"
