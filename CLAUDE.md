# manyruns — instructions for coding agents

manyruns is the environment a practitioner works in and the record of what happened.
It owns the user-facing workflow, harness, config catalog, and run artifacts. Keep it
thin on compute: training loops, geometric metrics, and dimensionality-reduction
algorithms belong in `manylatents`.

`manyruns` and `co-science` are aliases for the same program, both invoking
`manyruns.app:main`. Keep their behavior identical. The interaction loop is
**observe → legal moves → step → record**; `Session.step` accepts a step name or a
step dictionary, and `vocab.unmet` checks preconditions.

**Bring your own intelligence.** Any model, trained on anything, may drive a run.
manyruns defines what is legal, checks every move, runs it, and records who made it;
the model brings the judgement. Swapping, retraining, or cloning a model changes
nothing about what counts as a valid or reproducible run.

## Who is at the wheel

A stepped run has one wheel and a tag under its heading that always says who holds it.
A **driver** is any outside program attached to the run's **seat** (below).

| Mode | Who is there | Driver level | Tag |
|---|---|---|---|
| person alone | scientist | no driver attached | `HUMAN` |
| interpret | scientist + driver | `interpret`: talks only | `HUMAN` |
| propose | scientist + driver | `propose`: puts up cards the scientist takes | `HUMAN` |
| drive, watched | scientist + driver | `drive`: applies its own choices | `ON AIR · MACHINE · <chooser>` |
| manual | scientist, driver standing by | `interpret` until `ctrl+b` again | `MANUAL · <chooser> standing by` |
| driver alone | driver | `drive`, granted at launch | `ON AIR · MACHINE · <chooser>` |

Keys on the run screen, all priority bindings so they work while the ask field has
focus: `ctrl+g` cycles the level (interpret → propose → drive), `ctrl+b` toggles
manual and auto, `ctrl+y` takes the driver's card. With a driver attached, `? question` in
the ask field goes to the driver, and the answer pane's border names it.

## The driver seam

**A driver attaches to manyruns; manyruns never calls a driver.** This keeps the arrow
of the next section. A stepped run opens a seat: a Unix socket (mode 0600) in a short
temp folder, with a pointer at `outputs/<project>/driver.json`. Any program may
connect, as it may run the CLI. The seat names no caller.

The protocol is §8 of [the environment contract](docs/environment-contract.md); that is
what a driver author reads. In short: JSON lines; `hello` (with `chooser` and `bound`)
first; requests `actions`, `say`, `propose`, `apply`; events pushed by the session
(`control`, `offer`, `question`, `message`, `said`, `applied`, `taken`, `refused`,
`finished`). `hello` returns the offer already waiting, because events are not replayed.

| File | Holds |
|---|---|
| `manyruns/driver.py` | levels, the step offer, the check on a choice, request handling; pure |
| `manyruns/tui/driver_socket.py` | the seat: one driver at a time, threads, pointer file |
| `manyruns/tui/run.py` | the tag, the keys, the conversation pane, cards, applying a choice |
| `manyruns/tui/gate.py`, `manyruns/tune.py` | who answered, carried into the decision row |
| `manyruns/tui/app.py` | a seat for every stepped run |
| `tests/fixtures/driver_conformance.json` | cases with the verdict the seam owes each choice |

Rules:

- **The level is the scientist's.** No request can raise it. An unattended run gets its
  level from whoever starts it (`MANYRUNS_DRIVER_LEVEL`), before the run, never after.
- **manyruns checks every choice itself**, whatever schema the driver used, through the
  parser typed text goes through (`tune.validated_overrides`) plus a range check. A stale
  `legal_id`, an unknown parameter, a wrong type, or a value out of range is refused with
  its reason. Refuse, never clamp: a clamp runs a number the driver did not choose.
- **The offer states only what is known.** Names and current values come from
  `params.tunable_for`, minimums from `params.FLOOR`, caps from
  `bounds.resolved_constraints`. An undeclared maximum is `maximum_known: false`, never
  invented. A knob at a word that takes a number (`t: auto`) is offered as the number.
- **An apply is checked twice**: by the seat, and again on the UI thread, so a switch to
  manual or a human answer that lands in between wins.
- **Manual is a toggle** (`driver.handbrake`), like a paddle shift: one press drops the
  driver to `interpret`, keeps its level in `resume`, and refuses its next move; the tag
  reads `MANUAL · <chooser> standing by`. The same press returns it to `resume`.
  A fit already running is left to finish: cancelling mid-fit would discard the whole
  step (`run_tune_loop`'s `ComputeCancelled` path), the opposite of keeping the values.
- **Every move is attributed.** A tune loop a driver touched writes a version 2 decision
  row: each offered value names who set it (`by`) and whose card it came from
  (`proposed_by`); `chooser`/`bound` name who made the final choice; `proposed` and
  `rationale` carry the card behind the kept value. A loop no driver touched writes the
  version 1 row it always wrote.
- **The protocol is versioned** (`driver.PROTOCOL_VERSION`, on the pointer and on
  `hello`). Bump it for any change an existing driver could misread, and re-pin the
  conformance fixtures.
- **Name no driver, model, or trainer here.** Conformance fixtures ship here; the code
  that calls a model and scores it lives with the driver (manyagents, or the driver's
  own repository).
- The driver owns its loop. manyruns answers one request at a time. `agents.py` stays the
  only place manyruns itself asks a model: a rented one, one call at a time.

## No track to a learner — the arrow points one way

manyruns names no learner: no dependency, import, engine value, config, or CLI path.
A learner, or any agent, observes the artifacts (`index.jsonl`, g-vectors, decision
rows) and drives the CLI or the seat from outside, like any user. There is no privileged
caller. The environment defines what is legal and records what happened; the caller
decides what to do. See [the environment contract](docs/environment-contract.md), §3.5.
Attaching to the seat is not a track: the seat is open to every program and manyruns
never learns who is on the other end.

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

Two interfaces enumerate legal moves: `manyruns actions <DATASET>` (`manyruns/actions.py`)
prints a dataset's ledger as one JSON document, and the seat's `actions` returns the
offer for the step the tune loop is waiting on. The ledger is not on the seat yet. Add
further planning or enumeration interfaces only against a concrete caller's needs.

## Config and record handling

Ship the config layer inside the wheel at `manyruns/configs/`. Locate bundled
resources with `importlib.resources` (see `catalog.py`), never checkout-relative
or `__file__`-relative paths. Keep compute tools config-agnostic and inject the
config directory they need. Retain console-script entry points and avoid
string-`__import__` plugin loading so packaging remains straightforward.

`store.append` writes one run per line to cwd-relative `outputs/index.jsonl`.
TUI menu choices use `outputs/decisions.jsonl`; tuning choices use the session's
output directory. Missing schema versions read as 1, and unknown fields survive
reading and JSON serialization. Core row shapes are documented in `manyruns/store.py`
and `manyruns/decisions.py`; a decision row is version 2 exactly when a `chooser` is
given.

`store.append(extra=...)` is a top-level merge: `source`, `modality`, and
`decision_id` remain core fields even when passed through that argument. Put future
downstream fields in nested `extra.<caller>` namespaces.

Environment variables use `MANYRUNS_*` and are read through `manyruns.env.get`,
passing the suffix after the prefix. User cache state lives in `~/.cache/manyruns/`.
The driver seam reads two: `MANYRUNS_DRIVER_SOCKET` (`0` or empty: no seat) and
`MANYRUNS_DRIVER_LEVEL` (the launch grant: `interpret`, `propose`, or `drive`; a value
that is not a level opens no seat and says so).

## Distribution

The distribution name is `manyruns`; `co-science` is a console-script alias.
manyruns is Apache-2.0; releases up to 0.1.0 are MIT (see `NOTICE`). Contributions
come in under section 5 of the license. Add no code whose license restricts use, such
as non-commercial terms; `NOTICE` lists the MIOFlow files already carved out.
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
because the required single-cell dependency `scikit-misc` has no Windows wheels, and
the seat is a Unix socket. Base dependencies run every bundled recipe. Keep
`manylatents-omics[singlecell]`: bare omics does not supply scanpy. Public
dependencies resolve from PyPI without checkout source overrides. Diagnose tool
installs using installed versions and `direct_url.json`.

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

The driver seam's tests run in seconds and need no model:

```sh
uv run pytest -q tests/test_driver.py tests/test_driver_socket.py tests/test_tui_driver.py tests/test_driver_tune.py
```

`tests/test_tui_driver.py` attaches a real Unix-socket client to a real run screen
running the real tune loop on the mock engine; copy its harness for new screen tests.

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
A module-scope import must be in `tests/test_base_install.py`'s stdlib list or its
allowlist; defer anything else into the function that needs it.
