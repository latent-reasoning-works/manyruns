# manyruns/harness/data_acquire.py
"""Acquire-read shim: raw omics archives → a canonical ``.h5ad`` (the load contract).

This is the **ETL half** of the data seam documented in ``data/local.md``. It lives
in manyruns because it is I/O / distribution, not compute: it downloads a public
GEO / ArrayExpress supplementary archive, parses it with **scanpy** (the same reader
manylatents uses internally), and writes a canonical ``<accession>.h5ad``.

Layering (why this is here and not in manylatents):

    raw bytes ──[this shim + scanpy]──▶  <accession>.h5ad  ──[manylatents]──▶ tensors → DR
      (acquire-read, impure, once)        THE CONTRACT         (load-read, at run() time)

The contract with manylatents is the **``.h5ad`` path** returned by :func:`acquire` /
:func:`convert` — *not* a dataset name. This module never imports manylatents' DR
internals and never constructs a ``LabeledArray``/kind; it only produces the artifact
that ``AnnDataModule`` reads. scanpy is a direct dependency of the ``[harness]`` extra
(declared in ``pyproject.toml``) — this module does not rely on it arriving transitively.

Name resolution is deliberately *not* wired here. ``manylatents.api.run(data=<name>)``
(the harness fast path) resolves names only from manylatents' **core synthetic**
registry (``get_datamodule`` → swissroll/torus/blobs/…); it does **not** read the
singlecell ``configs/data/*.yaml`` and cannot load an AnnData dataset by name. So the
optional :func:`write_config` below serves the Hydra/serving pipeline only, and even
there the always-works handoff is the path. Wiring a name into the fast path is an
open decision (see ``data/local.md`` → "Reader / caller separation") and is out of
scope for this module.
"""
from __future__ import annotations

import csv
import logging
from manyruns import env as _env
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

_MANIFEST_FIELDS = {"accession", "source", "modality", "format", "url"}


# ── manifest ────────────────────────────────────────────────────────────────
@dataclass
class DatasetEntry:
    """One manifest row. ``meta`` carries every other column verbatim (tissue,
    disease, n_samples, note, …) so the manifest can grow without a code change."""

    accession: str
    source: str  # geo | arrayexpress | cellxgene
    modality: str  # scrna | spatial-visium | spatial-xenium | atlas
    fmt: str  # mtx | txt | tenx_h5 | visium | xenium | census
    url: str
    meta: dict[str, str] = field(default_factory=dict)


def manifest_path(path: Path | str | None = None) -> Path:
    """Resolve a caller-supplied manifest, then MANYRUNS_ACQUIRE_MANIFEST."""
    selected = path if path is not None else _env.get("ACQUIRE_MANIFEST")
    if not selected:
        raise FileNotFoundError(
            "No dataset manifest configured. Pass an explicit manifest path or set "
            "MANYRUNS_ACQUIRE_MANIFEST to a TSV or CSV file."
        )
    return Path(selected)


def load_manifest(path: Path | str | None = None) -> list[DatasetEntry]:
    """Parse the manifest into :class:`DatasetEntry` rows.

    Raises ``FileNotFoundError`` if absent so a missing manifest fails loudly.
    Reads TSV or CSV by file extension.
    """
    path = manifest_path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset manifest not found at {path}. Set MANYRUNS_ACQUIRE_MANIFEST "
            "to an existing TSV or CSV file, or pass an explicit manifest path."
        )
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    entries: list[DatasetEntry] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=delimiter):
            acc = (row.get("accession") or "").strip()
            if not acc or acc.startswith("#"):
                continue
            entries.append(
                DatasetEntry(
                    accession=acc,
                    source=(row.get("source") or "").strip(),
                    modality=(row.get("modality") or "").strip(),
                    fmt=(row.get("format") or "").strip(),
                    url=(row.get("url") or "").strip(),
                    meta={
                        k: (v or "").strip()
                        for k, v in row.items()
                        if k not in _MANIFEST_FIELDS and k
                    },
                )
            )
    return entries


def get_entry(accession: str, path: Path | str | None = None) -> DatasetEntry:
    """Look up one manifest entry by accession (case-insensitive)."""
    acc = accession.lower()
    for entry in load_manifest(path):
        if entry.accession.lower() == acc:
            return entry
    known = ", ".join(e.accession for e in load_manifest(path))
    raise KeyError(f"accession {accession!r} not in manifest. Known: {known}")


# ── URL builders (pure) ──────────────────────────────────────────────────────
def geo_supplementary_url(accession: str) -> str:
    """URL for a GEO series' bundled supplementary TAR (the raw matrices)."""
    return f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}&format=file"


def arrayexpress_files_url(accession: str) -> str:
    """URL for an ArrayExpress/BioStudies study's file listing."""
    return f"https://www.ebi.ac.uk/biostudies/files/{accession}/"


# ── output path (the contract lives here) ────────────────────────────────────
def default_out_dir() -> Path:
    """The omics data root's ``single_cell/`` dir — where manylatents reads ``.h5ad``.

    Reuses manylatents' single source of truth (``omics_data_root``) so writer and
    reader always agree on the location. Requires the omics package; when it is
    absent, callers must pass an explicit ``out_dir`` instead.
    """
    try:
        from manylatents._data_paths import omics_data_root
    except Exception as e:  # noqa: BLE001 - optional at ETL time
        raise RuntimeError(
            "omics data root unavailable (manylatents-omics not installed). "
            "Pass an explicit out_dir=... to acquire()/convert()."
        ) from e
    return omics_data_root() / "single_cell"


def canonical_h5ad_path(accession: str, out_dir: Path | str | None = None) -> Path:
    """Canonical ``<out_dir>/<accession>.h5ad`` path (the artifact the reader loads)."""
    base = Path(out_dir) if out_dir is not None else default_out_dir()
    return base / f"{accession}.h5ad"


# ── locating matrices in an extracted archive (pure) ─────────────────────────
@dataclass
class MatrixSource:
    """A single matrix located in an extracted archive tree.

    ``kind`` ∈ {mtx, tenx_h5, h5ad, txt}. For ``mtx`` the ``path`` is the *directory*
    and ``prefix`` the shared filename prefix (GEO ships flat ``GSM..._matrix.mtx.gz``
    triplets); for the others ``path`` is the file. ``sample`` labels the obs on concat.
    """

    kind: str
    path: Path
    sample: str
    prefix: str = ""


def _strip_gz(name: str) -> str:
    return name[:-3] if name.lower().endswith(".gz") else name


def find_matrices(root: Path | str) -> list[MatrixSource]:
    """Walk an extracted archive tree and locate the count matrices within.

    Handles the three shapes GEO/ArrayExpress ship: 10x MTX triplets (flat with
    ``GSM..._matrix.mtx`` prefixes or in per-sample subdirs), 10x ``.h5`` feature
    matrices, and ``.h5ad``. Pure filesystem/naming logic — no scanpy — so it is
    unit-testable with fabricated files.
    """
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    found: list[MatrixSource] = []

    for p in files:
        stem = _strip_gz(p.name)
        low = stem.lower()
        if low.endswith(".h5ad"):
            found.append(MatrixSource("h5ad", p, sample=p.stem))
        elif low.endswith(".h5"):
            found.append(MatrixSource("tenx_h5", p, sample=Path(stem).stem))

    # 10x MTX triplets: anchor on each ``matrix.mtx`` and derive the shared prefix.
    for p in files:
        stem = _strip_gz(p.name)
        low = stem.lower()
        if low.endswith("matrix.mtx"):
            prefix = stem[: -len("matrix.mtx")]  # e.g. "GSM1234_" or ""
            sample = prefix.rstrip("_-.") or p.parent.name
            found.append(MatrixSource("mtx", p.parent, sample=sample, prefix=prefix))

    # Dense text count tables only if no structured matrix was found (avoids
    # grabbing a triplet's own barcodes/features .tsv or a series_matrix.txt).
    if not found:
        for p in files:
            low = _strip_gz(p.name).lower()
            if low.endswith((".txt", ".csv", ".tsv")) and "series_matrix" not in low:
                found.append(MatrixSource("txt", p, sample=p.stem))

    found.sort(key=lambda m: (m.kind, m.sample, str(m.path)))
    return found


# ── download + extract (impure edge) ─────────────────────────────────────────
def fetch(accession: str, work_dir: Path | str, *, url: str | None = None,
          reporthook=None, manifest: Path | str | None = None) -> Path:
    """Download and extract an accession's raw archive into ``work_dir``.

    Currently implements the GEO bundled-TAR path (a single downloadable URL).
    ArrayExpress/CELLxGENE need per-file curation — call :func:`convert` on an
    already-local matrix for those. Returns the extraction root.

    ``reporthook`` is ``urlretrieve``'s own ``(blocknum, blocksize, total)`` callback,
    forwarded unchanged. Added for the TUI's fetch screen (``manyruns/tui/find.py``),
    which cannot draw a progress bar over bytes it never sees; it is also the only
    cooperative cancel point — raising from the hook propagates out of ``urlretrieve``
    and leaves the partial ``<accession>_RAW.tar`` on disk (it opens the destination
    ``'wb'`` and streams, so a re-run truncates rather than resuming). Default ``None``
    keeps every existing caller byte-identical.
    """
    entry = None
    try:
        entry = get_entry(accession, manifest)
    except KeyError:
        pass

    source = entry.source if entry else "geo"
    url = url or (entry.url if entry else geo_supplementary_url(accession))

    if source == "cellxgene":
        raise NotImplementedError(
            f"{accession} is a CELLxGENE atlas source — no file download. Use the "
            f"census DataModule (see write_census_config()) instead of acquire()."
        )
    if source == "arrayexpress":
        raise NotImplementedError(
            f"{accession} is ArrayExpress — file listing at {url}. Download the "
            f"processed matrix and call convert(src=<path>, accession={accession!r})."
        )

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    archive = work_dir / f"{accession}_RAW.tar"
    log.info("downloading %s → %s", url, archive)
    urllib.request.urlretrieve(url, archive, reporthook)  # noqa: S310 - trusted GEO/EBI hosts

    extract_root = work_dir / accession
    extract_root.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(extract_root, filter="data")  # filter="data" — no path escape
    log.info("extracted %s to %s", accession, extract_root)
    return extract_root


# ── read (scanpy, lazy) ──────────────────────────────────────────────────────
def _read_matrix(src: MatrixSource, *, genes_are_rows: bool = True):
    """Read one located matrix into an AnnData (cells × genes). scanpy imported lazily."""
    import scanpy as sc

    if src.kind == "h5ad":
        return sc.read_h5ad(src.path)
    if src.kind == "tenx_h5":
        return sc.read_10x_h5(src.path)
    if src.kind == "mtx":
        return sc.read_10x_mtx(src.path, prefix=src.prefix or None, var_names="gene_symbols")
    if src.kind == "visium":
        return sc.read_visium(src.path)
    if src.kind == "txt":
        adata = sc.read_csv(src.path, first_column_names=True)
        # GEO text tables are usually genes × cells; AnnData wants cells × genes.
        return adata.T if genes_are_rows else adata
    raise ValueError(f"unsupported matrix kind: {src.kind!r}")


def convert(
    src: Path | str,
    accession: str,
    *,
    out_dir: Path | str | None = None,
    genes_are_rows: bool = True,
    force: bool = False,
    verify: bool = False,
) -> Path:
    """Read local matrices under ``src`` and write the canonical ``<accession>.h5ad``.

    ``src`` may be a single matrix file/dir or an extracted archive tree containing
    per-sample matrices; multiple same-kind matrices are concatenated with an
    ``obs['sample']`` label (outer join on genes). Returns the ``.h5ad`` path.
    """
    out_path = canonical_h5ad_path(accession, out_dir)
    if out_path.exists() and not force:
        log.info("%s already exists — skipping (force=True to overwrite)", out_path)
        return out_path

    src = Path(src)
    matrices = find_matrices(src) if src.is_dir() else [_lone_source(src)]
    if not matrices:
        raise FileNotFoundError(f"no count matrix found under {src}")

    kinds = {m.kind for m in matrices}
    if len(kinds) > 1:
        raise ValueError(
            f"{accession}: mixed matrix kinds {sorted(kinds)} under {src}; "
            f"point convert() at a single kind."
        )

    import anndata as ad

    parts = []
    for m in matrices:
        adata = _read_matrix(m, genes_are_rows=genes_are_rows)
        # Set `sample` only if ABSENT — never clobber. A collaborator's own sample/donor
        # labels ARE the provenance and the batch axis; overwriting them with the filename
        # stem destroys both and silently satisfies core._assert_required_obs (a
        # single-valued column still "has" the key), so the gate greens on the very data
        # it exists to catch.
        if "sample" not in adata.obs:
            adata.obs["sample"] = m.sample
        parts.append(adata)
    adata = parts[0] if len(parts) == 1 else ad.concat(parts, join="outer", label=None)

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    log.info("wrote %s (%d cells × %d genes)", out_path, adata.n_obs, adata.n_vars)

    if verify:
        assert_loadable(out_path)
    return out_path


def _lone_source(src: Path) -> MatrixSource:
    """Wrap a single explicit file as a MatrixSource by sniffing its suffix."""
    low = _strip_gz(src.name).lower()
    if low.endswith(".h5ad"):
        return MatrixSource("h5ad", src, sample=src.stem)
    if low.endswith(".h5"):
        return MatrixSource("tenx_h5", src, sample=src.stem)
    if low.endswith("matrix.mtx"):
        prefix = _strip_gz(src.name)[: -len("matrix.mtx")]
        return MatrixSource("mtx", src.parent, sample=prefix.rstrip("_-.") or src.parent.name,
                            prefix=prefix)
    if low.endswith((".txt", ".csv", ".tsv")):
        return MatrixSource("txt", src, sample=src.stem)
    raise ValueError(f"cannot infer matrix kind from {src.name!r}")


def acquire(
    accession: str,
    *,
    manifest: Path | str | None = None,
    work_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    genes_are_rows: bool = True,
    force: bool = False,
    verify: bool = False,
) -> Path:
    """Fetch → convert an accession end-to-end. Returns the canonical ``.h5ad`` path.

    The one-call convenience over :func:`fetch` + :func:`convert`; use those two
    directly when an accession needs manual matrix selection. Supply ``manifest``
    or set ``MANYRUNS_ACQUIRE_MANIFEST``; no dataset catalogue is bundled.

    ``genes_are_rows`` sets the orientation for text counts and is forwarded to
    :func:`convert`. Reading cells by genes as genes by cells would silently run
    dimensionality reduction on the wrong axis.
    """
    manifest = manifest_path(manifest)
    out_path = canonical_h5ad_path(accession, out_dir)
    if out_path.exists() and not force:
        log.info("%s already exists — skipping", out_path)
        return out_path
    import tempfile

    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix=f"{accession}_"))
    root = fetch(accession, work_dir, manifest=manifest)
    return convert(
        root, accession, out_dir=out_dir, genes_are_rows=genes_are_rows,
        force=force, verify=verify,
    )


# ── optional: config emission (Hydra/serving route) + load check ─────────────
def write_config(
    accession: str,
    h5ad_path: Path | str,
    configs_dir: Path | str,
    *,
    label_key: str | None = None,
) -> Path:
    """Emit an ``AnnDataModule`` data-config yaml for ``accession``.

    SECONDARY to the path. This wires the dataset into the **Hydra/serving** pipeline
    only — ``manylatents.api.run(data=<name>)`` (the harness fast path) will NOT pick
    it up (see module docstring). Returns the written yaml path.
    """
    h5ad_path = Path(h5ad_path)
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the portable ${omics_data:} form when the artifact is under single_cell/.
    if h5ad_path.parent.name == "single_cell":
        adata_ref = f"${{omics_data:}}/single_cell/{h5ad_path.name}"
    else:
        adata_ref = str(h5ad_path)

    label = "null" if label_key is None else repr(label_key)
    out = configs_dir / f"{accession}.yaml"
    out.write_text(
        f"# Auto-generated by manyruns.harness.data_acquire for {accession}.\n"
        f"# For the Hydra/serving pipeline. The reliable handoff is the .h5ad path;\n"
        f"# api.run(data='{accession}') does NOT resolve this (see data/local.md).\n"
        "defaults:\n  - default\n\n"
        "_target_: manylatents.singlecell.data.anndata.AnnDataModule\n"
        f"adata_path: {adata_ref}\n"
        f"label_key: {label}\n"
        "mode: full\n",
        encoding="utf-8",
    )
    return out


def assert_loadable(h5ad_path: Path | str) -> None:
    """Verify manylatents can load the artifact (constructs AnnDataModule, sets up).

    A load-read acceptance check at the seam boundary. Imports manylatents lazily so
    this module stays importable without the private stack.
    """
    from manylatents.singlecell.data.anndata import AnnDataModule

    dm = AnnDataModule(adata_path=str(h5ad_path))
    dm.setup()
    log.info("verified %s loads via AnnDataModule", h5ad_path)
