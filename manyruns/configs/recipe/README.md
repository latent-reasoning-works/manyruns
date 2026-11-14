# Bundled recipes

Each file in this folder is one recipe: an ordered list of steps, the question it answers
(`question:`), the data shapes it suits (`suits:`), and the structure it claims (`claims:`).
`manyruns check` validates every file. `score.RUNG` grades each claim on the complexity
ladder.

## Structure class to recipe

For each of the four scShapeBench structure classes, the bundled recipe that claims it:

| Structure class | Recipe |
|---|---|
| `clusters` | `cluster` |
| `single-trajectory` | `pseudotime` |
| `multi-branching` | `branches` |
| `archetypal` | `archetypes` |

Each recipe's `claims:` equals its structure class. `tests/test_recipes.py` checks this table
against the files.

## The rung-support gate

`vocab.unmet` decides whether a recipe's steps can run on the data. The rung-support gate,
`vocab.rung_support`, decides whether the measured geometry supports the recipe's claim. A
claim in `vocab.BLOCKING_CLAIMS` is refused unless its evidence passes the rule:

| Recipe `claims` | Evidence key | Admission name | Supported when the value `v` is |
|---|---|---|---|
| `clusters` | `betti_0` | `betti_0` | `v > 1` |
| `multi-branching` | `reeb_graph.structural_summary.n_branch_points` | `n_branch_points` | `v >= 1` |
| `single-trajectory` | `lid` | `lid` | Nonblocking: `LID_1D_BAND` is a placeholder |

The gate gives these guarantees:

1. The gate applies only when a caller passes `evidence`; without it, only `vocab.unmet` and `narrate.unavailable` decide whether a row can run, and `manyruns actions` passes it whenever its gate is on (the default), from the probe run in the store.
2. With evidence, a missing, `None`, `MeasurementUnavailable` or non-finite value refuses a blocking claim.
3. A metric that is not in `admitted` refuses a blocking claim, and `admit.admit_metrics` cannot admit `n_branch_points`, because it audits registry metrics only.
4. Claims outside `vocab.BLOCKING_CLAIMS` are never refused by this gate. This includes `single-trajectory`, `surface`, `cycle`, and `archetypal`.
5. A pass is necessary evidence, not proof of the structure.
6. `ADMITTED_READOUTS` is empty, so the multi-branching rule refuses in v1 as shipped.
7. A refusal reason is one of the nine fixed strings of `vocab.SUPPORT_REASONS`, and it never holds a measured value, a threshold or a note. The `lid:*` reasons remain for compatibility but are not emitted.

`LID_1D_BAND` is `(0.5, 1.5)`, a placeholder: it is not calibrated. Single-trajectory cannot
be refused by this gate, regardless of missing or out-of-band `lid` evidence. The `lid` value
remains in the evidence g-vector. After calibration, adding `single-trajectory` to
`vocab.BLOCKING_CLAIMS` makes the rule blocking again. In `manyruns actions`, the
admitted set is the declared metric suite (`configs/metrics/default.yaml`, the audit's
verified-safe set) plus `vocab.ADMITTED_READOUTS`. A caller that has audited the Reeb summary
by other means adds `n_branch_points` to `admitted` explicitly.
