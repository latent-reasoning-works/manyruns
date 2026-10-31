# The MIOFlow example tool

This folder runs upstream MIOFlow, unmodified, as a manyruns tool step. It is an example tool
folder: it is in the repository, and it is not in the manyruns wheel or sdist.

| Item | Value |
|---|---|
| Pin | `mioflow==0.1.14`, from PyPI |
| Licence of MIOFlow | `LicenseRef-Yale-Non-Commercial`: research use only |
| Licence of the adapter | Apache-2.0, the licence of manyruns |
| Needs | `embedding` and `time` |
| Produces | `trajectories`, shape (bins, trajectories, coordinates) |

The manifest says `commercial_use: false`. So every run that uses this tool names the step in
the row field `non_commercial_steps`.

## Files

- `mioflow.yaml` is the tool manifest.
- `mioflow_adapter.py` is the pinned adapter. It builds the AnnData that MIOFlow expects, calls
  `MIOFlow(...).fit()` without GAGA, and writes a version 1 `result.json`. It contains no
  MIOFlow code.

## Use

`$MANYRUNS_TOOL_DIR` replaces the bundled tool folder (`toolchain.tool_dir`). While it is set,
the bundled tools, such as `pyrovelocity`, are not visible.

In a manyruns checkout, point manyruns at this folder for the current shell:

```sh
export MANYRUNS_TOOL_DIR="$PWD/examples/tools/mioflow"
```

Then show the state of the tool:

```sh
manyruns tools list
```

Without a lock for your platform, the status is `no-lock`. The lock pulls torch, so a person
compiles it with `uv pip compile --generate-hashes` into `locks/` and builds the environment
with `manyruns tools install mioflow`.

A recipe step uses the tool with `tool: mioflow`, for example
`{name: mioflow, group: lightning, tool: mioflow, params: {n_epochs: 50}}`. The step params
that the adapter accepts are `n_epochs`, `n_trajectories`, `n_bins`, `hidden_dim`,
`learning_rate`, `lambda_ot`, `lambda_energy`, `use_density_loss`, `lambda_density` and
`sample_size`.

## After a pin change

The pin is the contract. A new upstream release changes nothing until a person moves the pin.
After every pin change, run the seam check on the built-in fixture:

```sh
manyruns tools verify mioflow
```

If it exits 0, the new version still returns a result that `check_result` accepts.
