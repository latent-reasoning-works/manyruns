# Changelog

All notable changes to manyruns are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
During the open beta, the CLI and record schema can still change.

## [Unreleased]

- Relicensed from MIT to Apache-2.0. Releases up to and including 0.1.0 stay under MIT; NOTICE keeps the MIT notice and lists the MIOFlow files it does not cover.


### Removed

- The second console-script name. The wheel installs one command, `manyruns`; run
  `manyruns` wherever you ran the other name.

### Added

- `manyruns claim RUN --topology NAME=P[,NAME=P...] --chooser C [--bound B]` appends a
  structure-class verdict about a stored run to `outputs/index.jsonl`: one independent
  probability per topology, nested under `extra.verdict` with the chooser, the bound, the
  `legal_id` and the model. The chooser is `human:<pseudonym>`, `llm:<model>`, `learner:<id>` or
  `prior:<id>`; a non-human chooser needs `--bound`. A usage error exits 2 and an unknown run
  exits 1. A verdict row shares its run's `run_id`; a reader that counts runs skips rows where
  `score.verdict_of(row)` is not `None`.
- Scorer functions in `manyruns.score`: `verdict_of`, `pred_rung`, `truth_from_annotations`,
  `soft_truth`, `class_auprc` (macro and per-class average precision against the annotator
  majority), `brier` and `check_raw_labels`. `vocab.BENCHMARK_CLASSES` names the four benchmark
  classes.
- `MANYRUNS_BLIND_DATASET`: when on, run rows, the model prompt and the roster payload carry a
  blinded name (`ds-` plus 12 hex characters of a SHA-256) in place of the true dataset name,
  and `outputs/blind_names.jsonl` keeps the true name so `blind.unblind_rows` can join verdicts
  to labels. Step artifact paths are not rewritten.
- `manyruns actions [DATASET] [--surface ledger|tune|ladder]` prints the legal action space as
  one JSON document: the ledger of a dataset, the knobs of a recipe's tunable steps with their
  domains, or the scorer's rung tables. It is read-only. `--gate on|off` and `--seed N` record
  the geometry mask and select the probe run it reads. Every tunable knob declares its domain
  in `tui.params.DOMAIN`, and the `n_neighbors` arrow floor moves from 1 to 2.
- Decision rows carry `chooser`, `bound` and `legal_id` (still `schema_version` 1;
  older rows are read unchanged). `decisions.examples()` gains `surface` and
  `exclude_chooser` filters. `vocab` gains `VERSION` and the scShapeBench and harness
  labeler topology mappings. The labeler's `VALID_LABELS` is now derived from
  `vocab.LABELER_TOPOLOGY`, and labeled entries gain a canonical `topology`.
- An engine byproduct that is a dict of scalars, such as the Reeb graph
  `structural_summary`, is flattened one level into the g-vector as
  `<step>.<key>.<subkey>`; non-scalar sub-values are not recorded.
- An annotation store, `outputs/annotations.jsonl` (`manyruns.annotations`), records a
  judgment about a run, which is not a committed choice: a topology `label`, a `rating`, or a
  `correction`. Each row is `schema_version` 1 and names its chooser as `human:<pseudonym>` or
  `llm:<model>`. `decisions.jsonl` and `index.jsonl` are unchanged.
- Four verbs write it: `manyruns label RUN --topology T[,T...] --annotator P`,
  `manyruns import-labels scshapebench --source FILE --revision SHA` (one row per dataset per
  annotator; the file's SHA-256 is kept, its path is not), `manyruns rate RUN [RUN ...]
  --annotator P` with `--rank`, `--rubric` or `--blind`, and
  `manyruns correct RUN --step N --should STEP --annotator P`.
- The run screen labels a finished run: keys `1` to `6` toggle the six topologies, and `enter`
  records them as `human:` plus `MANYRUNS_ANNOTATOR`.
- `MANYRUNS_PROPOSER_MODEL` sets the proposer model. `agents.proposer_model(fallback=None)`
  returns it, else `fallback`, else `agents.MODEL`; a blank value counts as unset.
  `agents.proposer_extra(model, agent)` puts the model on a run row under `extra.proposer`.
- `agents.propose(session, prompt, ...)` runs manyAgents' `run_agent_loop` once with that
  model and records it under the session's `run_id`: the transcript in
  `<run folder>/agent/messages.jsonl` and trace events. It returns `None` and raises nothing
  when manyAgents is absent, an event loop is running, or the loop fails.
- New trace kinds: `step` and `run_complete` (`Session`), `decision` (`decisions.append`,
  with the row's `chooser` as `agent_id`), and `proposer_assistant`, `proposer_tool_call`,
  `proposer_tool_result` and `proposer_call` (`agents.propose`). `Session` gains `agent_id`
  (default `human`), and tune decision rows name the session's driver as `chooser`.
- A manylatents latent step writes the engine's `outputs.json` into
  `<run folder>/engine/<NN>-<step>/` through `pipeline.runner.RunStepLogger`, with an
  `outputs.done` marker. `COMPLETE` keeps its meaning.
- Section 8 of the environment contract documents the run folder layout and names `run_id` as
  the join key.

### Removed

- The `label` and `rlhf` stubs of `python -m manyruns.harness`, which printed "not yet
  implemented". Use `manyruns label` and the other annotation verbs.

### Fixed

- Installation and runtime repair guidance consistently uses uv tool installs for
  the application and `uv sync` for developer checkouts, with uv's official installer
  for first-time setup.
- A latent step that produces a partition, such as `leiden`, is no longer scored by
  the declared geometry suite, so one declining metric no longer fails the run; the
  g-vector records `<step>.suite_note` instead.

## [0.1.0] - 2026-09-17

First public release: an open beta for macOS and Linux, Python 3.11–3.12.

### Removed

- The second console-script name. The wheel installs one command, `manyruns`; run
  `manyruns` wherever you ran the other name.

### Added

- Interactive data exploration, 11 bundled recipes, and the run → label → store
  workflow, available through `manyruns`.
- Run and decision records carry `schema_version`; missing versions read as 1.
- Public install instructions, contribution and community guides, and citation metadata.
- Verified wheel and sdist builds with PyPI trusted publishing through GitHub Actions.

### Changed

- The harness no longer bundles a default dataset manifest; provide one explicitly or
  set `MANYRUNS_ACQUIRE_MANIFEST`.

- Mock runs record run identity and step outcomes as well as their invented results.
- `.env` is read only from the working directory and no longer overrides exported variables.
- Dependency floors require the published single-cell and trajectory runtime modules.

### Fixed

- Tabular `.csv` and `.tsv` inputs load instead of crashing.
- An explicit seed of 0 is preserved.
- Displayed figures survive tuning renames and file changes without crashing the TUI.

[Unreleased]: https://github.com/latent-reasoning-works/manyruns/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/latent-reasoning-works/manyruns/releases/tag/v0.1.0
