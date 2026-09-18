"""A synthetic dataset WITH spliced/unspliced layers — the input Gate 1.1 says we cannot get.

Two-state kinetic model, the one scVelo/pyro-velocity assume:
    du/dt = alpha(t) - beta*u        unspliced
    ds/dt = beta*u    - gamma*s      spliced
Cells are sampled along a latent time; a fraction of genes are induced, the rest repressed,
so the field has a real direction to recover. Counts are Poisson draws.

Written so the gate can be TESTED rather than argued about. It is NOT a claim that manyruns
should ship synthetic velocity data.
"""
import numpy as np, anndata as ad, pandas as pd, sys

rng = np.random.default_rng(0)
n_cells, n_genes = 600, 120
t = np.sort(rng.uniform(0, 1, n_cells))            # latent time
alpha = rng.uniform(3, 8, n_genes)
beta  = rng.uniform(0.4, 1.2, n_genes)
gamma = rng.uniform(0.2, 0.8, n_genes)
switch = rng.uniform(0.4, 0.9, n_genes)            # induction -> repression time per gene

def solve(tau, u0, s0, a, b, g):
    eb, eg = np.exp(-b * tau), np.exp(-g * tau)
    u = u0 * eb + a / b * (1 - eb)
    c = (a - u0 * b) / (g - b)
    s = s0 * eg + a / g * (1 - eg) + c * (eg - eb)
    return u, s

U = np.zeros((n_cells, n_genes)); S = np.zeros((n_cells, n_genes))
for j in range(n_genes):
    ts = switch[j]
    u_s, s_s = solve(np.array([ts]), 0.0, 0.0, alpha[j], beta[j], gamma[j])
    ind = t <= ts
    u1, s1 = solve(t[ind], 0.0, 0.0, alpha[j], beta[j], gamma[j])
    u2, s2 = solve(t[~ind] - ts, u_s[0], s_s[0], 0.0, beta[j], gamma[j])
    U[ind, j], S[ind, j] = u1, s1
    U[~ind, j], S[~ind, j] = u2, s2

Uc = rng.poisson(np.clip(U, 0, None) * 4).astype(np.float32)
Sc = rng.poisson(np.clip(S, 0, None) * 4).astype(np.float32)

a = ad.AnnData(X=Sc.copy())
a.layers["spliced"], a.layers["unspliced"] = Sc, Uc
a.var_names = [f"g{j:03d}" for j in range(n_genes)]
a.obs_names = [f"c{i:04d}" for i in range(n_cells)]
a.obs["latent_time"] = t
a.obs["stage"] = pd.Categorical(np.where(t < 1/3, "early", np.where(t < 2/3, "mid", "late")))
out = sys.argv[1]
a.write_h5ad(out)
print("wrote", out, a.shape, "layers:", list(a.layers), "obs:", list(a.obs.columns))
