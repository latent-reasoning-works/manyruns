"""Pyro-Velocity, behind the tool protocol. Runs in the tool's env — see `adapters/__init__`.

Reads a request JSON, trains, and writes the fields the manifest declares. Imports pyrovelocity
at module scope, which is legal HERE and nowhere else in this package.
"""
import json
import sys
import warnings

warnings.filterwarnings("ignore")


def main(request_path: str) -> int:
    req = json.loads(open(request_path).read())
    params = req.get("params") or {}
    import anndata as ad
    import numpy as np
    import scanpy as sc

    a = ad.read_h5ad(req["input"])
    missing = [k for k in ("spliced", "unspliced") if k not in a.layers]
    if missing:
        json.dump({"ok": False, "reason": f"input carries no {missing} layer(s) — RNA velocity "
                                          f"needs spliced/unspliced counts"},
                  open(req["result"], "w"))
        return 0

    # The input contract PyroVelocity.setup_anndata requires, built here rather than asked of
    # the caller: manyruns must not know this model's field names.
    a.layers["raw_unspliced"] = a.layers["unspliced"].copy()
    a.layers["raw_spliced"] = a.layers["spliced"].copy()
    a.obs["u_lib_size_raw"] = np.asarray(a.layers["unspliced"]).sum(1)
    a.obs["s_lib_size_raw"] = np.asarray(a.layers["spliced"]).sum(1)

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    n_pcs = int(min(30, min(a.shape) - 1))
    sc.pp.pca(a, n_comps=n_pcs)
    sc.pp.neighbors(a, n_neighbors=int(params.get("n_neighbors", 30)), n_pcs=n_pcs)
    for key, layer in (("Ms", "spliced"), ("Mu", "unspliced")):
        conn = a.obsp["connectivities"].copy()
        conn.setdiag(1.0)
        conn = conn.multiply(1.0 / conn.sum(1))
        a.layers[key] = np.asarray(conn @ np.asarray(a.layers[layer], dtype=float))

    from pyrovelocity.tasks.train import train_model

    adata, _model, post = train_model(
        a,
        max_epochs=int(params.get("max_epochs", 1500)),
        num_samples=int(params.get("num_samples", 12)),
        log_every=250, use_gpu="cpu", random_seed=int(req.get("seed") or 0),
        loss_plot_path=req["workdir"] + "/loss.png",
        loss_csv_path=req["workdir"] + "/loss.csv")

    # pyrovelocity/analysis/analyze.py:73's velocity, WITHOUT its `.mean(0)`. Their postprocess
    # collapses the posterior at the source; the spread is the reason to run this model at all.
    ut, st = np.asarray(post["ut"]), np.asarray(post["st"])
    beta, gamma = np.asarray(post["beta"]), np.asarray(post["gamma"])
    v = (ut * beta / np.asarray(post["u_scale"]) if "u_scale" in post else ut * beta) - st * gamma

    out = ad.AnnData(X=np.asarray(adata.layers["Ms"], dtype="float32"))
    out.obs_names = adata.obs_names.astype(object)
    out.var_names = adata.var_names.astype(object)
    out.obsm["velocity"] = v.mean(0).astype("float32")
    lo, hi = np.percentile(v, [2.5, 97.5], axis=0)
    out.obs["velocity_speed"] = np.linalg.norm(v.mean(0), axis=1)
    out.obs["velocity_frac_genes_ci_excludes_zero"] = ((lo > 0) | (hi < 0)).mean(1)
    ct = np.asarray(post["cell_time"])
    out.obs["velocity_pseudotime"] = ct.mean(0).ravel()
    out.obs["velocity_pseudotime_sd"] = ct.std(0).ravel()
    out.write_h5ad(req["output"])

    json.dump({"ok": True,
               "scalars": {"posterior_samples": int(v.shape[0]),
                           "epochs": int(params.get("max_epochs", 1500))}},
              open(req["result"], "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
