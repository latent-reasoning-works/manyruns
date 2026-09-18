# manyruns — instructions for coding agents

manyruns is the environment a practitioner works in and the record of what happened.
It owns the user-facing workflow, harness, config catalog, and run artifacts. Keep it
thin on compute: training loops, geometric metrics, and dimensionality-reduction
algorithms belong in `manylatents`.

`manyruns` and `co-science` are aliases for the same program, both invoking
`manyruns.app:main`. Keep their behavior identical. The interaction loop is
**observe → legal moves → step → record**; `Session.step` accepts a step name or a
step dictionary, and `vocab.unmet` checks preconditions.

## No track to a learner — the arrow points one way

manyruns names no learner: no dependency, import, engine value, config, or CLI path.
A learner, or any agent, observes the artifacts (`index.jsonl`, g-vectors, decision
rows) and drives the CLI from outside, like any user. There is no privileged caller.
The environment defines what is legal and records what happened; the caller decides
what to do. See [the environment contract](docs/environment-contract.md), §3.5.

Use roles in prose. `tests/learners.py` holds only SHA-256 digests of forbidden
strings; keep that guard. The public-safety gate checks every tracked file,
including prose, lockfiles, and binary payloads. Failures report category counts,
never identifying text. Do not add machine-specific home or session paths to
shipped files or documentation.

## Engine, tool, and substrate

Keep these concerns separate:

- **Capability:** a recipe step's group and name identify the tool. Add a capability
  as a step in an appropriate group (`latent`, `lightning`, `analysis`, etc.). When
  a new group is needed, add its executor to `pipeline.runner` and declare its data
  needs in `vocab.STEP_NEEDS`, so preconditions govern its availability.
- **Substrate:** local, GPU, or cluster execution determines where a step runs.
  Cluster dispatch through `shop` is optional; local use must work without it.
- **Fidelity:** `--engine` selects `manylatents` or `mock` for the same verb, running
  a recipe to produce a g-vector. A new capability is not a new engine value.

Delegate recipe computation to the compute libraries. The harness workflow is
`run → label → store` and stops at the store; `modes.run("train", …)` plans without
executing. Keep the mock for workflow tests. It writes no arrays, so artifact,
lineage, rewind, and branch tests use `pipeline.run_inproc` (`vocab.INPROC`). That
in-process loop is a test substrate, absent from product engine choices and serving
backends. Mock output is not scientific evidence.

Tools currently run as declared recipe steps. Add planning or action-enumeration
interfaces only against a concrete caller's needs.

## Config and record handling

Ship the config layer inside the wheel at `manyruns/configs/`. Locate bundled
resources with `importlib.resources` (see `catalog.py`), never checkout-relative
or `__file__`-relative paths. Keep compute tools config-agnostic and inject the
config directory they need. Retain console-script entry points and avoid
string-`__import__` plugin loading so packaging remains straightforward.

`store.append` writes one run per line to cwd-relative `outputs/index.jsonl`.
TUI menu choices use `outputs/decisions.jsonl`; tuning choices use the session's
output directory. Both writers put `schema_version: 1` first. Missing versions
read as 1, and unknown fields survive reading and JSON serialization. Core row
shapes are documented in `manyruns/store.py` and `manyruns/decisions.py`.

`store.append(extra=...)` is a top-level merge: `source`, `modality`, and
`decision_id` remain core fields even when passed through that argument. Put future
downstream fields in nested `extra.<caller>` namespaces.

Environment variables use `MANYRUNS_*` and are read through `manyruns.env.get`,
passing the suffix after the prefix. User cache state lives in `~/.cache/manyruns/`.

## Distribution

The distribution name is `manyruns`; `co-science` is a console-script alias.
Once published to PyPI:

```sh
uv tool install --python 3.12 manyruns
co-science
```

The public Git install is also the runtime repair command:

```sh
uv tool install --python 3.12 --force git+https://github.com/latent-reasoning-works/manyruns
```

Supported platforms: macOS and Linux, Python 3.11–3.12. Windows is unsupported
because the required single-cell dependency `scikit-misc` has no Windows wheels.
Base dependencies run every bundled recipe. Keep `manylatents-omics[singlecell]`:
bare omics does not supply scanpy. Public dependencies resolve from PyPI without
checkout source overrides. Diagnose tool installs using installed versions and
`direct_url.json`.

`uv tool install` supplies an isolated application. For a developer checkout, use
`uv sync --extra harness --extra dev` and run commands through `uv run`. Sync
reconciles the environment to exactly the extras named: name every extra already
in use, including `--extra agents` if present. See [Development](#development).

The CLI and record schema are open beta and can change. Follow
[CONTRIBUTING.md](CONTRIBUTING.md) for trusted publishing, the required reviewer on
the `release` environment, version tags, and deployment approval. The release
workflow builds and verifies artifacts before publishing them.

## Development

```sh
uv sync --extra harness --extra dev
uv run pytest tests/ -q
uv run ruff check manyruns tests
```

`dev` is an extra, not a dependency group: use `--extra dev`, not `--dev`.
**Never `uv pip install` in this project.** It mutates the environment without
updating `uv.lock`; the next sync removes those packages, and a fresh checkout
cannot reproduce the environment.

Adding a dependency has three steps:

1. Declare it in `pyproject.toml`, in the base dependencies or extra that needs it.
2. Run `uv lock` to update the lockfile without changing the environment.
3. Run `uv sync --extra harness --extra dev`, including every other extra in use.

A bare `uv sync` strips extras because it reconciles to exactly the requested set.
Keep track of the checkout's extras before syncing; use `uv lock` when only
relocking. Consumers inherit project dependency metadata, not `tool.uv.sources`
or this checkout's lockfile. Declare direct dependencies where they are used;
do not rely on another package's source overrides or incidental dependencies.

Write a failing test before changing behavior, then run affected tests, the full
dev suite, and ruff. **Every test that scans a path must first assert `is_dir()`
on its scan root.** A renamed directory can otherwise make `rglob` scan nothing
and pass silently. Keep this protection when moving packages or config trees.
