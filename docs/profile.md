# Shape profiles

Run a profile with the existing recipe interface:

```sh
manyruns run DATA --recipe profile --engine manylatents
```

The profile recipe publishes descriptive features of the original input, prepares
an unscaled centred principal component matrix with min(50, d) columns, and
publishes the shape table. Its parameters are pinned. Counts and annotated
processed inputs stop at preparation, with the features table retained.
Classification comes from the dataset_features helper's counts_like/1 output.
The string raw_integer means counts. A processed classification alone does not
make a plain numeric point cloud an annotated processed input.

A profile run has kind=profile and cannot provide probe evidence. Ordinary runs
have kind=recipe. Runs named derive:<connector> have kind=derived. These are
additive fields; historical rows without kind retain their previous reading.
New action offers include kind.

## Measured matrix

An eligible working matrix is captured before the first latent or lightning step
and saved as input_space after that step succeeds. A profile saves the matrix
prepared for shape_profile. A session with no successful steps writes no saved matrix.
A run with no such step saves its final working matrix. The matrix is float64
and carries row identity and a closed input_space manifest object with fields:

- input_space
- input_sha256
- preprocessing_sha256
- preprocessing

Sparse matrices and matrices above 512 columns are not saved. The run records
input_space.saved=false and reason sparse or too_many_columns. An unavailable
matrix records reason unavailable. A failed write records reason write_failed.
Named datasets use the engine's optional return_input_data result from its existing
load. Older engines without that option cannot supply the matrix; no second load
is performed. The separate CLI route for other lightning steps does not expose
its matrix and also records unavailable. Resume discovery excludes input_space;
it remains available for selections. Child sessions inherit the preprocessing
that produced their working matrix.

The preprocessing descriptor has exactly branch, steps, and definition_version.
Each successful prior step contributes name and effective params in execution
order. The definition version is profile/1. The branch is point_cloud for profile
and none for an ordinary run. The digest hashes UTF-8 JSON with sorted keys,
no spaces, ASCII escapes, and no nonfinite numbers.

File-backed inputs retain the SHA-256 of their file bytes. Without a file, a dense
matrix is hashed as b"manyruns-array/1\n", dtype string, newline, shape n,d,
newline, and C-order float64 bytes. A sparse matrix is converted to canonical CSR:
duplicate coordinates summed, explicit zeros eliminated, and indices sorted.
Its encoding is b"manyruns-csr/1\n", data dtype string, newline, shape n,d,
newline, int64 indptr bytes, int64 indices bytes, and float64 data bytes.
Sparse hashing never densifies. Equal dense and sparse values can have different
identities.

## Table and selections

The shape_profile table has exactly these columns, in order:

```text
entity_id parent_entity_id entity_kind stat_name scale_kind scale_value
aggregation value n_cells_used status reason replicate input_space
preprocessing_sha256 knn_backend dtype metric_version
```

The file uses tab separators. Values use Python float repr for exact round trips.
Statuses are ok, censored, empty, and undefined. Non-ok values are empty except
window_hi with reason window_right_censored, whose value is a lower bound.
The dataset has 15 rows. A selection adds 41 rows, including parent-relative
statistics, conductance, and selection fidelity. Counts precede deduplication;
core aggregation uses distinct points with equal weight.

The exact graph uses float64 Euclidean distances and row-index tie breaking.
The helper computes every statistic. The step validates the closed helper
contract and publishes the table atomically.

Use the existing derive choice interface for a selection:

```sh
manyruns run --choice CHOICE.json --selection SELECTION.tsv
```

The connector is shape_profile and its execution is inprocess. Choices contain
connector and params; execution cannot be selected by the caller. The selection
uses cell and group columns. Groups form one union. The saved selection table's
hash identifies the selection.

A selection uses the parent's saved input_space and preprocessing digest.
It recomputes both aggregations together on that matrix. It never refits selected
cells or reads another run's shape values. A parent without an eligible saved
matrix is refused with parent_input_space_unavailable. A missing aligned
two-dimensional layout makes selection fidelity undefined.

The extended parameter defaults to false. True is refused as not_implemented.

The timing test is skipped unless MANYRUNS_PROFILE_TIMING=1. A pytest marker
selection alone does not enable it.
