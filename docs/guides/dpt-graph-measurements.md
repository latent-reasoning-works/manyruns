# Pseudotime graph measurements

The public SwissRoll fixture reproduced disconnected graphs after a diffusion map.
PCA means principal component analysis. DPT means diffusion pseudotime.
A connected component is a set of points joined by paths in the neighbour graph.

The fixture uses `manylatents.data.synthetic_dataset.SwissRoll` with
`n_distributions=n`, `n_points_per_distribution=1`, `noise=0`,
`manifold_noise=0`, and `random_state=seed`. Other generator settings retain
their defaults. No normalization or logarithm is applied.

`sklearn.decomposition.PCA(n_components=3)` transforms the raw points.
The diffusion map receives that output through `manylatents.api.run` with
`algorithms={'latent': 'diffusionmap'}`, `seed=seed`, `n_components=3`, and
`n_landmark=None`. Other algorithm settings retain the bundled defaults.

For each embedding, coordinates are converted to float32. Scanpy constructs
the graph with `sc.pp.neighbors(ad, n_neighbors=15, use_rep='X')`.
`scipy.sparse.csgraph.connected_components(ad.obsp['connectivities'])`
counts its components. DPT uses `sc.tl.diffmap(ad, n_comps=15)` followed by
`sc.tl.dpt(ad, n_dcs=2)`. Unreachable means the resulting pseudotime is not finite.
The table reports both the old coordinate-sum root and the new root.
The new root minimizes `ad.obsm['X_diffmap'][:, 1]`.
Scanpy 1.12.2 documents column zero as the steady-state solution.

| Points | Seed | Connected components, PCA graph | PCA unreachable, either root | Connected components, diffusion map graph | Diffusion map unreachable, old root | Diffusion map unreachable, new root |
|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0 | 1 | 0 | 1 | 0 | 0 |
| 400 | 1 | 1 | 0 | 2 | 139 | 261 |
| 400 | 42 | 1 | 0 | 1 | 0 | 0 |
| 1000 | 0 | 1 | 0 | 2 | 655 | 345 |
| 1000 | 1 | 1 | 0 | 2 | 108 | 892 |
| 1000 | 42 | 1 | 0 | 1 | 0 | 0 |

Changing the root does not repair a disconnected graph. The recipe therefore
runs DPT on PCA coordinates and retains the later diffusion map for its own
embedding and measurements. These measurements do not establish that either
embedding always has a connected graph.

The regression tests also use a NumPy curve: seed 1 generates 1000 sorted
uniform values `t`, with coordinates `(t, sin(t), cos(t))`. After PCA, its
15-neighbour graph had one component and DPT produced no unreachable values.

The shared `_require_embedding` check now refuses stale coordinates in these steps:

- `mioflow`, including its manylatents executor
- `separation`
- `composition`
- `cluster_quality`
- `rank_genes`, which can read a partition stored in the embedding slot
- `dpt`
- `simplex`
- `velocity_field`
- latent steps that chain through `_ml_latent`

Before this change, `_ml_latent` used any nonempty embedding slot as the next
algorithm's input. Its `source_fit` condition checked only whether that slot
was empty. A failed coordinate step could therefore leave older coordinates
in use without a refusal. The runner now marks retained coordinates stale
after an error. A successful step that supplies new coordinates clears the mark.
Partition and graph failures do not mark coordinates stale.

`discretize_time` does not refuse: it bins a pseudotime that an earlier step
computed, so a later coordinate failure does not change its input.
