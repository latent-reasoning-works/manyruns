# Dataset registry

One YAML per dataset. **The directory is the registry** — adding a dataset is adding a file;
there is no Python list to keep in sync. `manyruns check` validates every file here.

    name:     <unique name; must match the filename stem>
    handle:   kind: manylatents | path        # where the bytes come from
              ref:  <engine dataset name>  |  <filesystem path>
    modality: scrna | bulk | synthetic
    shape:    the GROUND-TRUTH shape — see manyruns/vocab.py SHAPES

`shape` is load-bearing: it is the label the metric-separation experiment scores against
("which metrics tell data shapes apart"). A wrong or invented shape silently creates a
spurious class in the results table, so it is a closed vocabulary and `manyruns check`
rejects anything outside it.

The catalog includes synthetic point-cloud generators, a locally generated
`synthetic_timecourse` with timepoint metadata, and the real `pbmc3k` single-cell sample
from [10x Genomics](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0).
The pbmc3k declaration is bundled, but the data is fetched only on explicit first use and
verified against its declared SHA-256 and byte count. No sample H5AD files ship in the wheel.

To use your own datasets without editing the wheel, point `$MANYRUNS_DATASET_DIR` at your
own directory.

## Known gap

The bundled shapes include `manifold`, `clusters`, and `time-course`, but no
`case-control` cohort, so the `contrast` recipe has **no legal cell** — `manyruns`'s
planner prunes it everywhere (`experiment.skipped_cells` reports it).
A `case-control` dataset is the first real addition needed here.

The experiment path now threads condition labels (`run_explorations` loads them for any
`path:` handle), so a case-control `.h5ad` with a `disease`/`condition` obs column is all
that is missing — add one and `contrast` runs.
