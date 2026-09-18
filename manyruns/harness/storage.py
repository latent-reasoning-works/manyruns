# manyruns/harness/storage.py
"""Labeled dataset storage for DR workflow results."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DATASET_VERSION = "1.0"


def save_labeled(path: Path | str, entry: dict[str, Any], embedding: np.ndarray) -> None:
    """Append a labeled entry to a JSON dataset file and save its embedding as NPY.

    Creates the file if it doesn't exist, appends if it does.

    Args:
        path: Path to JSON metadata file.
        entry: Labeled entry dict from label_workflow().
        embedding: The embedding array to save.
    """
    path = Path(path)
    embeddings_dir = path.parent / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Save embedding NPY
    embedding_filename = f"{entry['id']}.npy"
    entry = {**entry, "embedding_path": f"embeddings/{embedding_filename}"}
    np.save(embeddings_dir / embedding_filename, embedding)
    log.debug(f"Saved embedding to {embeddings_dir / embedding_filename}")

    # Load existing or create new
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = {
            "version": DATASET_VERSION,
            "created": datetime.now().isoformat(),
            "entries": [],
        }

    data["entries"].append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    log.info(f"Saved entry to {path} ({len(data['entries'])} entries total)")


def load_labeled(path: Path | str) -> list[dict[str, Any]]:
    """Load all labeled entries from a JSON dataset file.

    Args:
        path: Path to JSON metadata file.

    Returns:
        List of labeled entry dicts.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return data.get("entries", [])


def load_embedding(dataset_path: Path | str, entry: dict[str, Any]) -> np.ndarray:
    """Load the embedding array for a labeled entry.

    Args:
        dataset_path: Path to the JSON metadata file (embeddings are resolved
            relative to this file's parent directory).
        entry: Labeled entry dict with an ``embedding_path`` key.

    Returns:
        Embedding numpy array.

    Raises:
        ValueError: If entry has no embedding_path.
        FileNotFoundError: If embedding file doesn't exist.
    """
    embedding_path = entry.get("embedding_path")
    if embedding_path is None:
        raise ValueError(f"Entry {entry.get('id', '?')} has no embedding path")

    full_path = Path(dataset_path).parent / embedding_path
    if not full_path.exists():
        raise FileNotFoundError(f"Embedding not found: {full_path}")

    return np.load(full_path)


def dataset_summary(path: Path | str) -> dict[str, Any]:
    """Get summary statistics for a labeled dataset.

    Args:
        path: Path to JSON metadata file.

    Returns:
        Dict with total_entries, labels, datasets, version, created.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    entries = data.get("entries", [])
    labels = Counter(e["label"] for e in entries)
    datasets = Counter(e["dataset"] for e in entries)

    return {
        "total_entries": len(entries),
        "labels": dict(labels),
        "datasets": dict(datasets),
        "version": data.get("version", "1.0"),
        "created": data.get("created", ""),
    }
