"""Model-version registry — the minimal serving-router substrate: name a version, pull it.

A *version* pins how a served model runs:
  - ``kind="recipe"``  — a fixed driver loop (e.g. ``cflows`` / ``contrast``), no weights;
  - ``kind="weights"`` — trained weights (a MIOFlow checkpoint) applied at inference time.

Versions are immutable JSON under ``experiments/manifests/<model>/<version>.json``, so infer
can **pull** a version by name and one model can serve concurrent versions (v0 fixed, v1
trained). Dep-free to register/resolve. Weight *application* (`apply_weights`) is refused:
manyruns never imports MIOFlow, which runs only as a tool step through the model seam.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from manyruns import core

_KINDS = ("recipe", "weights")


def registry_root() -> Path:
    """``experiments/manifests/`` — the version registry, sibling of the Core's reference_library."""
    return core.reference_library_root().parent / "manifests"


@dataclass
class Version:
    model: str
    version: str  # e.g. "v0", "v1"
    kind: str  # "recipe" | "weights"
    spec: dict = field(default_factory=dict)  # {"recipe": "cflows"} | {"weights": path, dims…}
    created: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Version":
        return cls(**{k: d[k] for k in d if k in cls.__dataclass_fields__})


def _path(model: str, version: str, root: Path | str | None) -> Path:
    base = Path(root) if root is not None else registry_root()
    return base / model / f"{version}.json"


def register(
    model: str, version: str, kind: str, spec: Optional[dict] = None,
    *, root: Path | str | None = None, created: Optional[str] = None, force: bool = False,
) -> Path:
    """Register an immutable version record. Refuses to overwrite unless ``force``."""
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    p = _path(model, version, root)
    if p.exists() and not force:
        raise FileExistsError(f"{model}@{version} already exists at {p} (immutable); force=True to overwrite")
    rec = Version(model=model, version=version, kind=kind, spec=spec or {}, created=created)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
    return p


def resolve(model: str, version: str, *, root: Path | str | None = None) -> Version:
    """Pull a version by name."""
    p = _path(model, version, root)
    if not p.exists():
        raise KeyError(f"no version {model}@{version} at {p}")
    return Version.from_dict(json.loads(p.read_text(encoding="utf-8")))


def list_versions(model: str, *, root: Path | str | None = None) -> list[Version]:
    """All registered versions of a model (version-sorted)."""
    base = (Path(root) if root is not None else registry_root()) / model
    if not base.is_dir():
        return []
    return sorted(
        (Version.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in base.glob("*.json")),
        key=lambda v: v.version,
    )


def apply_weights(version: Version, X: Any):
    """Inference through a trained-weights version — refused: manyruns never imports MIOFlow.

    This loaded a MIOFlow ODE network from manylatents and integrated ``X`` in this process.
    MIOFlow runs only through the model seam, in a tool's own interpreter, so a weights-kind
    version cannot be applied here. Registering and resolving versions stays dep-free and
    unchanged."""
    raise NotImplementedError(
        f"{version.model}@{version.version} is a weights version, and manyruns does not apply "
        "trained MIOFlow weights in process; run MIOFlow as a tool step (`tool: mioflow`, "
        "examples/tools/mioflow) instead")
