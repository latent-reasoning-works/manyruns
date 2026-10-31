# SPDX-License-Identifier: Apache-2.0
"""The pinned adapter for upstream MIOFlow, `mioflow==0.1.14`.

It calls the Yale-licensed `mioflow` package at run time, inside the tool's own interpreter, and
contains no MIOFlow code. The package is used as published: this file builds the input object
the library expects, makes the calls its documentation shows, and writes a version 1 result.

  1. decline when the request carries no time axis
  2. decline a parameter outside ALLOWED_PARAMS
  3. import mioflow, anndata and torch; seed numpy and torch
  4. build an AnnData with obs["time_bin"] and obsm["X_pca"]
  5. MIOFlow(adata, gaga_model=None, ..., use_cuda=False).fit()
  6. write `trajectories` and the scalars n_bins and n_trajectories

It never imports manyruns. It runs as `python mioflow_adapter.py request.json`.
"""
import json
import sys
from pathlib import Path

import numpy as np

#: The MIOFlow parameters a recipe step may set. Any other key is declined, so a step cannot
#: reach an option this adapter was not written for (for example `use_cuda`).
ALLOWED_PARAMS = ("n_epochs", "n_trajectories", "n_bins", "hidden_dim", "learning_rate",
                  "lambda_ot", "lambda_energy", "use_density_loss", "lambda_density",
                  "sample_size")


def _write(path, doc):
    Path(path).write_text(json.dumps(doc, indent=2))
    return 0


def _decline(req, reason):
    return _write(req["result"], {"schema_version": 1, "ok": False, "reason": reason})


def main(request_path):
    import hashlib

    req = json.loads(Path(request_path).read_text())
    # 1. MIOFlow interpolates BETWEEN time points, so it cannot run without them.
    if not req.get("time"):
        return _decline(req, "MIOFlow needs a time axis, and this run's labels are not time "
                             "points")
    # 2. The allow-list, checked before anything is imported.
    params = dict(req.get("params") or {})
    unknown = sorted(set(params) - set(ALLOWED_PARAMS))
    if unknown:
        return _decline(req, f"parameter(s) {unknown} are outside the allowed list "
                             f"{list(ALLOWED_PARAMS)}")

    # 3. The library, imported only now and only here, in the tool's interpreter.
    import anndata
    import mioflow
    import torch

    seed = int(req.get("seed") or 0)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 4. The input object the library expects.
    X = np.load(req["input"], allow_pickle=False)
    time = np.load(req["time"], allow_pickle=False)
    adata = anndata.AnnData(X=np.asarray(X, dtype="float32"))
    adata.obs["time_bin"] = time
    adata.obsm["X_pca"] = np.asarray(X)

    # 5. The upstream call, without GAGA: with gaga_model=None, MIOFlow reads obsm["X_pca"].
    workdir = Path(req.get("workdir") or Path(req["result"]).parent)
    fitted = mioflow.MIOFlow(adata, gaga_model=None, gaga_input_key="X_pca",
                             obs_time_key="time_bin", use_cuda=False, exp_dir=str(workdir),
                             **params).fit()

    # 6. The trajectories, de-normalized by MIOFlow into the frame of the input.
    trajectories = np.asarray(fitted.trajectories, dtype="float64")
    base = Path(req["result"]).parent
    path = base / "trajectories.npy"
    np.save(path, trajectories)
    return _write(req["result"], {
        "schema_version": 1, "ok": True, "n_cells": int(X.shape[0]),
        "facts": {"trajectories": {"path": path.name,
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        "scalars": {"n_bins": int(trajectories.shape[0]),
                    "n_trajectories": int(trajectories.shape[1])},
    })


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
