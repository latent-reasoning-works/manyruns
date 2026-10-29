# Start from a published figure, then make it your own

**The point:** do not start from a blank page. Start from what is already published. In a few
minutes you get the structure that the authors showed, with their settings, recorded. From
there you change things yourself, and every change is a new recorded run that you can compare
with the published starting point.

**What this is:** a worked example. You take a published embedding, find the settings that give
the same structure, save them as a recipe, run that recipe on your own similar data, and then
change it. Everything marked **[measured]** was run on 2026-10-29, macOS arm64, CPU only.

| Step | Time **[measured]** |
|---|---|
| Download the authors' data | 3 s |
| First run with the authors' settings | 1 min 36 s |
| The same settings on a second file | 37 s |

"Match" here means the same structure: the same groups, in the same order, with the same
branches. It does not mean the same pixels. UMAP and PHATE layouts change with the seed and
the library version, and a branch can appear mirrored or on the other side.

---

## 1 · The figure and its data

The example is pancreatic endocrinogenesis at E15.5 (Bastidas-Ponce et al., *Development*
146:dev173849, 2019, [doi:10.1242/dev.173849](https://doi.org/10.1242/dev.173849)). The scVelo
paper (Bergen et al., *Nat Biotechnol* 38:1408–1414, 2020,
[doi:10.1038/s41587-020-0591-3](https://doi.org/10.1038/s41587-020-0591-3)) and its
[tutorial](https://scvelo.readthedocs.io/en/stable/vignettes/Pancreas.html) use the same data.

The authors published one `.h5ad` file. It has the raw counts, the cell-type labels and the
authors' own UMAP coordinates:

```bash
curl -L -o endocrinogenesis_day15.h5ad \
  https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad
```

**[measured]** 52 MB, 3 s. 3696 cells × 27998 genes. `obs` has `clusters` with eight cell
types: Ductal, Ngn3 low EP, Ngn3 high EP, Pre-endocrine, Alpha, Beta, Delta and Epsilon.
`obsm['X_umap']` holds the authors' layout.

The structure to match is one trunk (Ductal → Ngn3 low EP → Ngn3 high EP → Pre-endocrine) that
ends in four separate fates (Alpha, Beta, Delta, Epsilon).

## 2 · The first run: the default recipe

```bash
manyruns run endocrinogenesis_day15.h5ad --project pancreas --recipe embed --color-by clusters --seed 1
```

**[measured]** 2 min 37 s, of which 73 s were the geometry measurements. `embed` is PHATE on
all genes. The trunk matched, in the right order. The four fates did not: they were compressed
along one edge, and Alpha and Beta overlapped.

## 3 · Settings that match: the tutorial's own

The tutorial keeps genes seen in at least 20 cells and builds the UMAP on 30 principal
components with 30 neighbours. Written as a recipe:

```yaml
# umap_scvelo.yaml
name: umap_scvelo
id: sc.embed.umap_scvelo
source:
  kind: tutorial
  cite: "scVelo endocrine pancreas tutorial settings: 30 PCs, 30 neighbours"
  url: https://scvelo.readthedocs.io/en/stable/vignettes/Pancreas.html
  verified: 2026-10-29
claims: surface
steps:
  - {name: filter_genes, group: prep,   params: {min_cells: 20},     via: manylatents}
  - {name: normalize,    group: prep,   params: {target_sum: 10000}, via: manylatents}
  - {name: transform,    group: prep,   params: {method: log1p},     via: manylatents}
  - {name: pca,          group: latent, params: {n_components: 30},
     limits: {n_components: [n_samples, n_features]},                via: manylatents}
  - {name: umap,         group: latent, params: {n_components: 2, n_neighbors: 30}, via: manylatents}
question: "Lay it out the way the scVelo tutorial does"
suits:    [single, unknown, manifold, clusters]
```

Put it in a folder of your own and point `$MANYRUNS_RECIPE_DIR` at that folder. `manyruns
check` validates it. A `source` needs a `doi` or a `url`.

```bash
export MANYRUNS_RECIPE_DIR=$PWD/recipes
manyruns check
manyruns run endocrinogenesis_day15.h5ad --project pancreas_umap --recipe umap_scvelo --color-by clusters --seed 1
```

**[measured]** 1 min 36 s, on the first try. `filter_genes` dropped 15,242 genes and kept
12,756. The layout has the trunk and four separate fates: Beta and Alpha at the end of the
trunk, Epsilon and Delta as their own small groups.

If the paper does not state its settings, change one value in the recipe file (for example
`n_neighbors`) and run again. Each run is a new row in the record, so you can go back to the
settings of any earlier attempt.

## 4 · Reuse the settings on similar data

The recipe is now the saved setting. To apply it to a second pancreas experiment, run the same
recipe on that file:

```bash
manyruns run my_pancreas.h5ad --project my_pancreas --recipe umap_scvelo --color-by clusters --seed 1
```

**[measured]** As a stand-in for a second experiment, a random 2000-cell subset of the same
file ran in 37 s. The layout had the same trunk and four separate fates. Alpha and Beta changed
sides, which is a layout difference, not a structure difference.

## 5 · Show that both runs used the same settings

Each run writes one row to `outputs/index.jsonl`. The row has the recipe, the seed, every
step's parameters and a `spec_id`. The `spec_id` is a hash of the recipe name, each step with
its parameters, the engine and the seed. It does not include the data file. So two runs on
different files with the same `spec_id` used the same settings and the same seed.

```python
from manyruns import store

for row in store.read("outputs"):
    print(row["run_id"], row["recipe"], row["spec_id"], [(s["name"], s["params"]) for s in row["steps"]])
```

**[measured]** Both runs above have `spec_id` `01a092dfa939`, with identical step parameters.
Each row also has the geometry measurements of each step (for example `umap.trustworthiness`
and `umap.betti_0`), so you can compare the two structures by number as well as by eye.

## 6 · Then make it your own

The published setting is now your starting point, not your answer. Some things to try next:

- **Change one value.** Copy `umap_scvelo.yaml` to a new name, change `n_neighbors` or the
  `filter_genes` threshold, and run it. The new row has a different `spec_id`, and its geometry
  measurements sit next to those of the published setting.
- **Change the method.** Run `embed` (PHATE) on the same file and compare the two rows. In
  this example, PHATE with default settings kept the trunk and compressed the four fates.
- **Ask a different question of the same data.** Run another recipe, such as `cluster` or
  `pseudotime`, on the same file.
- **Step through it.** `manyruns open <project>` reopens a saved project and steps through it
  interactively.

Every attempt is a row in `outputs/index.jsonl`. The published setting stays one `spec_id` that
you can always go back to.

## 7 · Limits

- **No highly-variable-gene step yet.** The recipe language has no `hvg` step, so
  `filter_genes` stands in for gene selection. Most published scRNA pipelines select highly
  variable genes, and a figure that depends on that selection may not match.
- **Colours.** `--color-by` uses manyruns' palette, not the colours stored in the file
  (`uns['clusters_colors']`). The groups are the same, and the colours differ.
- **Same method.** Match the paper's method: PHATE did not separate these fates with default
  settings, and UMAP with the paper's settings did.
- **Your data must be similar.** A recipe fixes the settings, not the biology. On data with a
  different cell-type mix, the same settings can give a different structure, and that
  difference is a result, not a failure.
