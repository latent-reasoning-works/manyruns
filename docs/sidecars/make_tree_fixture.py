"""The demo's act-1 dataset: DLA-tree ground truth, written as an .h5ad a person can drop.

`dla_tree` is in manylatents' registry and ships everything the demo needs — 800 points on 8
branches, `get_labels()`, and a colormap with names. None of it reaches manyruns through
`--dataset dla_tree`, because a NAMED dataset is loaded inside the engine and this process
never sees the array (`manyruns/pipeline/runner.py:1087`). Written to a file it takes the
same route pbmc3k does: `load_labeled` -> `labels_of` -> `ctx["color"]` -> the scatter.

Sixty dimensions rather than the native three, because the demo's point is that a knob moves
an embedding — a 3-D input embedded to 3-D shows nothing about neighbourhood size. `sigma=2`
keeps the branches separable at knn=40 and fragmenting at knn=5, which is the contrast the
demo turns on.

This is a sidecar. It is NOT a claim that manyruns should ship synthetic data.

    python docs/sidecars/make_tree_fixture.py data/tree8.h5ad
"""
import sys

import anndata as ad
import numpy as np
import pandas as pd
from manylatents.data import get_dataset

OUT = sys.argv[1] if len(sys.argv) > 1 else "data/tree8.h5ad"

d = get_dataset("dlatree", n_branch=8, sigma=2, n_dim=60, random_state=0)
X = np.asarray(d.data, dtype=np.float32)
labels = np.asarray(d.get_labels())

# `branch` rather than `group`: "group" is in `vocab.CONDITION_KEYS`, where it means a
# case/control arm, so a column named that would be read as a CONDITION and colour the plot
# under the wrong kind. `branch` is in `GROUP_KEYS` and says what these actually are.
names = getattr(d.get_colormap_info(), "label_names", None) or {}
obs = pd.DataFrame(
    {"branch": pd.Categorical([str(names.get(int(v), f"Branch {v}")) for v in labels])},
    index=[f"cell{i:04d}" for i in range(X.shape[0])])

a = ad.AnnData(X=X, obs=obs)
a.var_names = [f"f{j:03d}" for j in range(X.shape[1])]
a.write_h5ad(OUT)
print(f"wrote {OUT}: {a.shape}, {len(set(obs['branch']))} branches")
