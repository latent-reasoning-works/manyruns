"""Provenance beside the flat g-vector; numeric consumers keep their existing keys."""
from __future__ import annotations


def annotate(g: dict, key: str, *, kind: str, stage: str) -> None:
    # Session and tune snapshots shallow-copy g. Replace the map so a retry cannot rewrite
    # the provenance of an earlier accepted state through a shared nested dictionary.
    g["_provenance"] = {**g.get("_provenance", {}), key: {"kind": kind, "stage": stage}}
