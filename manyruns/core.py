"""The Core — manyruns's product-side reference library of collaborator datasets.

Assemble the validated per-source `.h5ad`s (produced by `manyruns.harness.data_acquire`) into a
**snapshot-versioned collection** — "a true core" that manyruns owns, reuses, and trains
against, and from which model versions stream (see the serving seam in `serving.py`).

Design (per the locked MVP decisions):

- **The Core is a *collection*, not one integrated matrix.** Each dataset stays its own
  `.h5ad`; the snapshot *references* them by path and records provenance + the canonical
  `obs` vocabulary. Cross-dataset integration/batch-correction is a downstream **model**
  output (a trained version v1+), NOT part of the core build — so this module stays
  compute-free and product-side (no DR, no training, no matrix merge).
- **Snapshots are immutable and versioned:** `experiments/reference_library/<name>@v<n>/`.
  Rebuilding a version is refused unless `force=True`; a new version is a new directory.
- **manylatents only *reads* the members.** This module writes a manifest, never weights.

Import-clean without the heavy stack: `anndata` is imported lazily inside `inspect_member`
(the only function that opens a `.h5ad`); everything else is pure and unit-testable.
"""
from __future__ import annotations

import json
import os
from manyruns import env as _env
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Canonical obs schema for the Core. `sample` + `disease` are required (the analysis
# label + provenance axis); the rest interpret the embedding. Harmonizing collaborator
# annotations onto these keys happens at validate time (Step 2); the assembler records
# the observed vocabulary per member so mismatches are visible before training.
REQUIRED_OBS = ("sample", "disease")
RECOMMENDED_OBS = ("cell_type", "tissue")
CANONICAL_OBS = REQUIRED_OBS + RECOMMENDED_OBS

_MAX_VOCAB = 50  # cap recorded value-sets so a huge cell_type space doesn't bloat the manifest


# ── on-disk home ─────────────────────────────────────────────────────────────
def reference_library_root() -> Path:
    """Base dir holding Core snapshots, install-location-independent.

    Resolution mirrors ``manylatents._data_paths.omics_data_root``:
    1. ``$MANYRUNS_EXPERIMENTS`` — explicit override.
    2. ``<repo>/experiments/reference_library`` in a source checkout (repo root carries
       ``pyproject.toml``).
    3. ``~/.cache/manyruns/experiments/reference_library`` once installed.
    """
    env = _env.get("EXPERIMENTS")
    if env:
        return Path(env).expanduser() / "reference_library"

    repo_root = Path(__file__).resolve().parents[1]  # <repo> in a checkout; site-packages once installed
    if (repo_root / "pyproject.toml").is_file():
        return repo_root / "experiments" / "reference_library"

    cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache).expanduser() if cache else Path.home() / ".cache"
    return base / "manyruns" / "experiments" / "reference_library"


def snapshot_dir(name: str, version: int, root: Path | str | None = None) -> Path:
    """Directory for a specific Core snapshot: ``<root>/<name>@v<version>/``."""
    base = Path(root) if root is not None else reference_library_root()
    return base / f"{name}@v{version}"


# ── data model ───────────────────────────────────────────────────────────────
@dataclass
class CoreMember:
    """One dataset in a Core snapshot — referenced by path, never copied."""

    accession: str
    path: str  # the validated .h5ad (str for clean JSON round-trip)
    n_obs: int
    n_vars: int
    obs_vocab: dict[str, list[str]] = field(default_factory=dict)  # canonical key → observed values
    source: str = ""
    tissue: str = ""
    disease: str = ""

    def missing_required_obs(self) -> list[str]:
        """Required canonical `obs` keys absent from this member (empty ⇒ ok)."""
        return [k for k in REQUIRED_OBS if k not in self.obs_vocab]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CoreMember":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


@dataclass
class CoreSnapshot:
    """An immutable, versioned collection of `CoreMember`s + provenance."""

    name: str
    version: int
    members: list[CoreMember]
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "members": [m.to_dict() for m in self.members],
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoreSnapshot":
        return cls(
            name=d["name"],
            version=d["version"],
            members=[CoreMember.from_dict(m) for m in d.get("members", [])],
            provenance=d.get("provenance", {}),
        )


# ── pure operations ──────────────────────────────────────────────────────────
def dedupe_by_accession(members: list[CoreMember]) -> list[CoreMember]:
    """Drop duplicate accessions, keeping first occurrence (order-preserving).

    Cross-dataset *sample* overlap (same donor in two series) is a real dedup axis but
    needs a donor mapping to detect reliably — deferred.
    """
    seen: set[str] = set()
    out: list[CoreMember] = []
    for m in members:
        key = m.accession.lower()
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _assert_required_obs(members: list[CoreMember]) -> None:
    """Raise if any member lacks a required canonical `obs` key (fail loud, don't guess)."""
    bad = {m.accession: miss for m in members if (miss := m.missing_required_obs())}
    if bad:
        raise ValueError(
            f"Core assembly blocked — members missing required obs {list(REQUIRED_OBS)}: "
            f"{bad}. Harmonize obs at validate time (Step 2) before assembling."
        )


# ── heavy edge (opens a .h5ad; anndata imported lazily) ──────────────────────
def inspect_member(
    h5ad_path: Path | str,
    accession: str,
    *,
    source: str = "",
    tissue: str = "",
    disease: str = "",
) -> CoreMember:
    """Read shape + canonical `obs` vocabulary from a validated `.h5ad`.

    Uses backed mode so the expression matrix is not loaded into memory — only `obs`.
    """
    import anndata as ad

    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"member .h5ad not found: {path}")

    adata = ad.read_h5ad(path, backed="r")
    try:
        vocab: dict[str, list[str]] = {}
        for key in CANONICAL_OBS:
            if key in adata.obs.columns:
                values = [str(v) for v in adata.obs[key].unique()[:_MAX_VOCAB]]
                vocab[key] = sorted(values)
        return CoreMember(
            accession=accession,
            path=str(path),
            n_obs=int(adata.n_obs),
            n_vars=int(adata.n_vars),
            obs_vocab=vocab,
            source=source,
            tissue=tissue,
            disease=disease,
        )
    finally:
        if adata.isbacked and adata.file is not None:
            adata.file.close()


# ── assembly ─────────────────────────────────────────────────────────────────
def assemble_core(
    members: list[CoreMember],
    *,
    name: str = "dataset-core",
    version: int = 0,
    root: Path | str | None = None,
    harmonization_map: dict | None = None,
    source_manifest: str | None = None,
    created: str | None = None,
    tool_version: str | None = None,
    force: bool = False,
) -> Path:
    """Write an immutable Core snapshot from already-inspected members.

    Returns the snapshot directory. Refuses to overwrite an existing version unless
    ``force`` — snapshots are immutable; a new version is a new directory.
    """
    members = dedupe_by_accession(members)
    if not members:
        raise ValueError("cannot assemble an empty Core — no members provided.")
    _assert_required_obs(members)

    out = snapshot_dir(name, version, root)
    manifest = out / "snapshot.json"
    if manifest.exists() and not force:
        raise FileExistsError(
            f"Core snapshot {name}@v{version} already exists at {out} (immutable). "
            f"Bump the version or pass force=True."
        )

    if tool_version is None:
        try:
            from manyruns import __version__ as tool_version  # type: ignore
        except Exception:  # noqa: BLE001
            tool_version = "unknown"

    provenance = {
        "created": created,  # caller injects an ISO timestamp; kept explicit for reproducible tests
        "tool_version": tool_version,
        "n_members": len(members),
        "canonical_obs": {"required": list(REQUIRED_OBS), "recommended": list(RECOMMENDED_OBS)},
        "harmonization_map": harmonization_map or {},
        "source_manifest": source_manifest,
    }
    snapshot = CoreSnapshot(name=name, version=version, members=members, provenance=provenance)

    out.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    return out


def load_snapshot(snapshot_path: Path | str) -> CoreSnapshot:
    """Load a Core snapshot from its directory or its ``snapshot.json``."""
    p = Path(snapshot_path)
    if p.is_dir():
        p = p / "snapshot.json"
    return CoreSnapshot.from_dict(json.loads(p.read_text(encoding="utf-8")))


def build_from_manifest(
    accessions: list[str] | None = None,
    *,
    name: str = "dataset-core",
    version: int = 0,
    root: Path | str | None = None,
    omics_dir: Path | str | None = None,
    created: str | None = None,
    force: bool = False,
) -> Path:
    """Convenience: assemble a Core from acquired datasets named in the acquire manifest.

    For each accession, locates its canonical `.h5ad` (via `data_acquire`), inspects it,
    and carries source/tissue/disease provenance from the manifest row. scRNA/atlas rows
    without a produced `.h5ad` are skipped with no silent success — a missing file raises.
    """
    from manyruns.harness import data_acquire as da

    entries = {e.accession: e for e in da.load_manifest()}
    picks = accessions or [
        e.accession for e in entries.values() if e.modality in {"scrna", "atlas"}
    ]

    members: list[CoreMember] = []
    for acc in picks:
        entry = entries.get(acc)
        h5ad = da.canonical_h5ad_path(acc, omics_dir)
        members.append(
            inspect_member(
                h5ad,
                acc,
                source=entry.source if entry else "",
                tissue=entry.meta.get("tissue", "") if entry else "",
                disease=entry.meta.get("disease", "") if entry else "",
            )
        )

    return assemble_core(
        members,
        name=name,
        version=version,
        root=root,
        source_manifest=str(da.manifest_path()),
        created=created,
        force=force,
    )
