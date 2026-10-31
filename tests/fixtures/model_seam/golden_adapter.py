"""The golden adapter: a test-only model that needs numpy and nothing else.

It writes a fixed small output for every seam fact, so CI tests the model seam without a
model. Its parameters make it fail in each way the seam must catch:

  truncate           writes only the first k rows of `clusters.npy`
  bad_edge           writes a graph edge to a node that does not exist
  flat_trajectories  writes 2-D trajectories
  decline            writes `ok: false` with this reason
  scalars            replaces the scalars of the result

It runs in run mode as `python golden_adapter.py request.json`. It never imports manyruns.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SEAM_FACTS = ("embedding", "clusters", "pseudotime", "trajectories", "graph",
              "class_probabilities")

CLASS_PROBABILITIES = {"clusters": 0.1, "single-trajectory": 0.2, "multi-branching": 0.9,
                       "archetypal": 0.0, "cycle": 0.0, "surface": 0.0}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_outputs(base, X, cell_ids, params, *, facts=SEAM_FACTS):
    """Write the golden output into `base`, restricted to `facts`; return the result document."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    params = dict(params or {})
    if params.get("decline"):
        doc = {"schema_version": 1, "ok": False, "reason": str(params["decline"])}
        (base / "result.json").write_text(json.dumps(doc, indent=2))
        return doc

    X = np.asarray(X, dtype=float)
    n, d = X.shape
    arrays = {
        "embedding": X[:, :2].astype("float32"),
        "clusters": (np.arange(n) % 2).astype("int64"),
        "pseudotime": np.linspace(0.0, 1.0, n),
        "trajectories": np.arange(3 * 2 * d, dtype=float).reshape(3, 2, d),
    }
    if params.get("truncate") is not None:
        arrays["clusters"] = arrays["clusters"][: int(params["truncate"])]
    if params.get("flat_trajectories"):
        arrays["trajectories"] = arrays["trajectories"].reshape(6, d)

    out_facts = {}
    for fact in ("embedding", "clusters", "pseudotime", "trajectories"):
        if fact in facts:
            path = base / f"{fact}.npy"
            np.save(path, arrays[fact])
            out_facts[fact] = {"path": path.name, "sha256": _sha256(path)}
    if "graph" in facts:
        graph = {
            "edges": np.array([[0, 1], [1, 2], [1, 3]], dtype="int64"),
            "edge_weights": np.array([1.0, 1.0, 1.0]),
            "node_ids": np.array([0, 1, 2, 3], dtype="int64"),
            "node_coordinates": np.array([[0, 0], [1, 0], [2, 1], [2, -1]], dtype="float64"),
            "cell_node": (np.arange(n) % 4).astype("int64"),
        }
        if params.get("bad_edge"):
            graph["edges"] = np.array([[0, 9]], dtype="int64")
            graph["edge_weights"] = np.array([1.0])
        path = base / "graph.npz"
        np.savez(path, **graph)
        out_facts["graph"] = {"path": path.name, "sha256": _sha256(path)}
    if "class_probabilities" in facts:
        out_facts["class_probabilities"] = {"value": dict(CLASS_PROBABILITIES)}

    doc = {"schema_version": 1, "ok": True, "n_cells": int(n)}
    if cell_ids is not None:
        joined = "\n".join(str(c) for c in cell_ids)
        doc["cells_sha256"] = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    doc["facts"] = out_facts
    doc["scalars"] = (dict(params["scalars"]) if params.get("scalars") is not None
                      else {"n_cells": int(n)})
    (base / "result.json").write_text(json.dumps(doc, indent=2))
    return doc


def main(request_path):
    """Run mode: read `request.json`, write the golden output beside `result.json`."""
    req = json.loads(Path(request_path).read_text())
    X = np.load(req["input"], allow_pickle=False)
    base = Path(req["result"]).parent
    # `request.json` carries no cell list, so this result has no `cells_sha256`.
    write_outputs(base, X, None, req.get("params") or {})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
