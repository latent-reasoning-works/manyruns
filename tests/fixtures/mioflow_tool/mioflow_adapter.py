"""A numpy-only stand-in for the pinned MIOFlow adapter, for the tests.

It reads the same request as `examples/tools/mioflow/mioflow_adapter.py` and declines in the same
two ways, with the same reasons. When it runs, it fits nothing: it writes
`trajectories = np.zeros((4, 3, d))` for an input of d columns, with the scalars `n_bins` and
`n_trajectories`. It never imports manyruns or MIOFlow. It runs as
`python mioflow_adapter.py request.json`.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

#: The allow-list of the real adapter, copied so that a declined parameter declines here too.
ALLOWED_PARAMS = ("n_epochs", "n_trajectories", "n_bins", "hidden_dim", "learning_rate",
                  "lambda_ot", "lambda_energy", "use_density_loss", "lambda_density",
                  "sample_size")

N_BINS, N_TRAJECTORIES = 4, 3


def _write(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2))
    return 0


def _decline(req, reason):
    return _write(req["result"], {"schema_version": 1, "ok": False, "reason": reason})


def main(request_path):
    req = json.loads(Path(request_path).read_text())
    if not req.get("time"):
        return _decline(req, "MIOFlow needs a time axis, and this run's labels are not time "
                             "points")
    params = dict(req.get("params") or {})
    unknown = sorted(set(params) - set(ALLOWED_PARAMS))
    if unknown:
        return _decline(req, f"parameter(s) {unknown} are outside the allowed list "
                             f"{list(ALLOWED_PARAMS)}")

    X = np.load(req["input"], allow_pickle=False)
    trajectories = np.zeros((N_BINS, N_TRAJECTORIES, int(X.shape[1])), dtype="float64")
    path = Path(req["result"]).parent / "trajectories.npy"
    np.save(path, trajectories)
    return _write(req["result"], {
        "schema_version": 1, "ok": True, "n_cells": int(X.shape[0]),
        "facts": {"trajectories": {"path": path.name,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        "scalars": {"n_bins": N_BINS, "n_trajectories": N_TRAJECTORIES},
    })


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
