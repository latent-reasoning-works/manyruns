"""Local tool manifests and adapters for tests that exercise the tool protocol."""

import sys
from pathlib import Path

import yaml

from manyruns import toolchain


def make_tool(tmp_path, monkeypatch, name: str, manifest: dict, adapter: str) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    cfg = {
        "name": name,
        "version": "1.0",
        "licence": "MIT",
        "commercial_use": True,
        "python": "3.12",
        "requires": [f"{name}==1.0"],
        "entrypoint": f"{name}_adapter.py",
        "selftest": "pass",
        **manifest,
    }
    (tools / f"{name}.yaml").write_text(yaml.safe_dump(cfg))
    (tools / cfg["entrypoint"]).write_text(adapter)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(tools))
    monkeypatch.setenv("MANYRUNS_TOOL_HOME", str(tmp_path / "envs"))
    monkeypatch.setenv(f"MANYRUNS_TOOL_{name.upper()}_PYTHON", sys.executable)
    lock = toolchain.lock_path(toolchain.load_tool(name))
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{name}==1.0 --hash=sha256:deadbeef\n")
    return tools


FRAME_ADAPTER = """
import json
import sys

import anndata as ad
import numpy as np

req = json.loads(open(sys.argv[1]).read())
if req["input"].endswith(".npy"):
    out = ad.AnnData(X=np.load(req["input"]))
else:
    out = ad.read_h5ad(req["input"])
n = out.n_obs
obs = out.obs.copy()
out.obsm["field"] = out.X[:, :1].copy()
mask = req["params"].get("mask")
if mask in ("alternate", "reversed"):
    keep = [True, False] * (n // 2)
elif mask == "none":
    keep = [False] * n
elif mask == "strings":
    keep = ["yes"] * n
elif mask == "short":
    # AnnData requires obs and X to have the same row count, even for a bad mask.
    out = out[:-1].copy()
    keep = [True] * (n - 1)
else:
    keep = [True] * n
out.obs["keep"] = keep
if mask == "reversed":
    out = out[::-1].copy()
elif mask == "renamed":
    out.obs_names = ["x" + n for n in out.obs_names]
out.write_h5ad(req["output"])
scalars = {"rows": n}
for key, column in [("samples", "sample"), ("conditions", "condition"), ("batches", "batch")]:
    scalars[key] = ",".join(sorted(set(obs[column]))) if column in obs else ""
json.dump({"ok": True, "scalars": scalars}, open(req["result"], "w"))
"""


TABLE_ADAPTER = """
import hashlib
import json
import sys
from pathlib import Path

import anndata as ad
import pandas as pd

req = json.loads(open(sys.argv[1]).read())
frame = ad.read_h5ad(req["input"])
params = req["params"]
result = {"schema_version": 1, "ok": True, "n_cells": frame.n_obs, "facts": {}}
if not params.get("no_tables"):
    genes = params.get("genes", list(frame.var_names[:2]))
    table = pd.DataFrame({"gene": genes, "log2fc": [1.0] * len(genes), "padj": [0.01] * len(genes)})
    if params.get("drop"):
        table = table.drop(columns=[params["drop"]])
    result["tables"] = {}
    for name in (["de", "extra"] if params.get("extra") else ["de"]):
        path = Path(req["result"]).parent / f"{name}.tsv"
        table.to_csv(path, sep="\t", index=False)
        digest = "0" * 64 if params.get("bad_sha") else hashlib.sha256(path.read_bytes()).hexdigest()
        result["tables"][name] = {"path": path.name, "sha256": digest}
json.dump(result, open(req["result"], "w"))
"""
