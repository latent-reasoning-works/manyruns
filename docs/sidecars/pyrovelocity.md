# Pyro-Velocity as a sidecar

**What this is:** the worked example of a tool that manyruns wants the *output* of and cannot
host. Everything marked **[measured]** was run on 2026-08-20, macOS arm64, Python 3.12.13.

---

## 1 · Why it is out of process, and it is not a preference

**[measured]** `uv pip install pyrovelocity` resolves and installs cleanly — exit 0, 148
packages — and then **does not import**:

```
ModuleNotFoundError: No module named 'scvi.model.base._utils'
```

`pyrovelocity 0.4.5` declares `scvi-tools>=1.1.1` and imports four private scvi paths
(`scvi.data._constants`, `scvi.model._utils`, `scvi.model.base._utils`, `scvi.module.base`).
`scvi/model/base/_utils.py` exists in `scvi-tools 1.1.x` and **not** in 1.2.2 or later
**[measured]**, so the unbounded lower pin resolves to a version whose private API moved.

Chasing it down took **ten interventions** and every one exposed the next:

| # | pin | what it broke |
|---|---|---|
| 1 | *(none)* | `scvi.model.base._utils` gone at scvi-tools 1.5.0 |
| 2 | `scvi-tools==1.2.2` | `pooch` missing (an unlisted scvi extra) |
| 3 | `+ pooch` | still no `_utils` — 1.2.2 does not have it either |
| 4 | `scvi-tools==1.1.6` | dragged `numpy<2`; `astropy 8` needs `numpy.lib.array_utils` |
| 5 | `numpy>=2` | `lightning 2.1.4` needs `pkg_resources` |
| 6 | `setuptools<81` | (fixed 5; setuptools 81 removed `pkg_resources`) |
| 7 | `astropy<7` | back to scvi 1.1.6 + numpy 1.26 |
| 8 | `jax==0.4.35` | scvi 1.1.6 imports `jaxlib.xla_extension.Device`, gone in jaxlib 0.7 |
| 9 | `numpyro==0.15.3`, `flax<0.10`, `optax<0.2.4`, `orbax-checkpoint<0.7` | numpyro 0.21 needs `jax.api_util.debug_info` |
| 10 | `altair<6` | `altair.vegalite.v5` gone in altair 6 |

**The working set [measured]:**

```
pyrovelocity==0.4.5  scvi-tools==1.1.6  numpy==1.26.4  scipy==1.13.1  pandas==2.3.3
anndata==0.12.19     astropy==6.1.7     jax==jaxlib==0.4.35           numpyro==0.15.3
flax==0.9.0          optax==0.2.3       orbax-checkpoint==0.6.4       chex==0.1.90
altair==5.5.0        setuptools==80.10.2  pooch==1.9.0   torch==2.13.0
```

**`numpy==1.26.4` is the line that settles it.** manyruns runs `numpy 2.2.6`. There is no pin
set that satisfies both, so this is not "a heavy extra we chose not to add" — it is an
environment that **cannot coexist** with the product's. The `[omics]`-style extra is not
available as an option here at any price.

---

## 2 · The contract

```
 raw .h5ad  ──[ pyrovelocity_sidecar.py, its OWN venv ]──▶  <name>.h5ad
                                                              obsm["velocity"]   (cells x genes)
                                                              obs["velocity_*"]  (per-cell posterior)
                                                                     │
 manyruns  ◀──[ loading.velocity_field, THIS venv ]────────────────┘
```

Nothing crosses but a file. The sidecar imports no manyruns; manyruns imports no
pyrovelocity, names it in no dependency, and reaches it through no code path. **[measured]**
manyruns's venv (anndata 0.13.2 / numpy 2.2.6) reads an `.h5ad` written by the sidecar's
(anndata 0.12.19 / numpy 1.26.4) with no shim.

The producer writes, and `loading._VELOCITY_OBS` is the closed list that reads:

| key | meaning |
|---|---|
| `obsm["velocity"]` | posterior **mean** field, `(n_cells, n_genes)` |
| `obs["velocity_pseudotime"]` / `_sd` | shared latent time, posterior mean and spread |
| `obs["velocity_speed"]` | ‖velocity‖ per cell |
| `obs["velocity_frac_genes_ci_excludes_zero"]` | fraction of genes whose 95% posterior CI excludes zero |

**The sidecar keeps the posterior on purpose.** pyrovelocity's own postprocess collapses it —
`(ut * beta / u_scale - st * gamma).mean(0)` at `pyrovelocity/analysis/analyze.py:73` — so the
uncertainty that is the entire reason to run a Bayesian velocity model is averaged away at the
source. `docs/sidecars/pyrovelocity_sidecar.py` computes the same quantity **without** the
`.mean(0)` and summarises the spread into the four `obs` columns above.

---

## 3 · Running it

```bash
# once: an isolated venv, NEVER the project venv (CLAUDE.md — no `uv pip install` here)
uv venv --python 3.12 /tmp/pv/.venv
VIRTUAL_ENV=/tmp/pv/.venv uv pip install pyrovelocity \
  "scvi-tools==1.1.6" "numpy<2" "astropy<7" "setuptools<81" pooch \
  "jax==0.4.35" "jaxlib==0.4.35" "numpyro==0.15.3" "flax<0.10" "optax<0.2.4" \
  "orbax-checkpoint<0.7" "altair<6"

# a fixture with spliced/unspliced layers (two-state kinetic model, Poisson counts)
uv run python docs/sidecars/make_velocity_fixture.py /tmp/synth.h5ad

# the producer
/tmp/pv/.venv/bin/python docs/sidecars/pyrovelocity_sidecar.py /tmp/synth.h5ad /tmp/vel.h5ad 1500

# the consumer — a user recipe, not a bundled one (see §5)
MANYRUNS_RECIPE_DIR=docs/sidecars manyruns run /tmp/vel.h5ad --recipe velocity --engine manylatents
```

---

## 4 · Does it work? — measured, not asserted

The fixture carries a **known** latent time, so recovery is checkable rather than plausible.

| budget | Spearman ρ (recovered vs true latent time) | p |
|---|---|---|
| 120 epochs | −0.347 | 2.2e-18 |
| **1500 epochs** | **+0.820** | 9.2e-147 |

Sign is not identifiable in a velocity model, so the magnitude is the number. At the default
budget the model genuinely recovers the ordering; at 120 epochs it does not, which is worth
recording because a short run looks exactly like a working one from the outside.

**The readout, and the correction it needed.** `velocity_field` reports the mean cosine
similarity between each cell's velocity and its k nearest neighbours' — "do nearby cells agree
about where they are going". Measured on the 1500-epoch field and two controls:

| field | `coherence` | `coherence_null` | **`coherence_excess`** |
|---|---|---|---|
| real | 0.936 | 0.672 | **+0.264** |
| same vectors, **shuffled onto the wrong cells** | 0.669 | 0.674 | **−0.005** |
| gaussian noise, same scale | −0.001 | −0.001 | **−0.001** |

**71% of the raw number survived destroying the correspondence it claims to measure.** A field
with a dominant overall direction scores high on mean-cosine whether or not it tracks the
geometry — the `granger` shape (`docs/granger-verdict.md`): large, stable, and a function of
the field's anisotropy rather than of local agreement. The permutation baseline is computed
in-executor and `coherence_excess` is the number to read. Both nulls collapse to ~0; the real
field does not. `tests/test_velocity_field.py` pins that separation rather than the values.

**It is still not a finding.** `vocab.NULL_KIND` has no row for `velocity_field`, deliberately:
the baseline above permutes the *correspondence*, which is neither of that table's two kinds
(`labels`, `data`). Minting a third kind is a vocabulary decision, not a step's. So the number
is a description, and `narrate` says nothing about it.

---

## 5 · Why the recipe is NOT in the wheel

`docs/sidecars/velocity.yaml`, not `manyruns/configs/recipe/velocity.yaml`. That is not
tidiness — it is an invariant the suite enforces, discovered by violating it:

> `tests/test_recipe_catalog.py::test_every_recipe_is_legal_on_some_bundled_dataset`
> — *"velocity is legal on none of the 14 bundled datasets; it needs ['velocity'] and nothing
> declares them"* **[measured]**

**A bundled recipe must be exercisable on bundled data.** `velocity` needs a field no shipped
file can carry, so bundling it made **16 tests fail** across the catalog, the ledger, the rung
derivation and the shape-legality tables. Moving one file out fixed all 16.

The split is the right one and it generalises: **the step, the loader and the fact ship in the
wheel; the recipe ships beside the producer.** `velocity_field` is generic — it reads any
velocity field, from scVelo or anything else — so it belongs to the product. The recipe names a
workflow the product cannot run on its own data, so it belongs to whoever supplies the data.

---

## 6 · Licence

`pyrovelocity 0.4.5` is **AGPL-3.0-only** **[measured]**. Nothing in this arrangement links it:
it runs as a separate program, in a separate interpreter, communicating through a file. That is
the ordinary boundary, and it is a second independent reason the sidecar shape is the right one.
manyruns still declares **no licence at all** (no `LICENSE` file, no `license` field in
`pyproject.toml` — **[measured]**), which should be fixed before the question is asked in anger.
