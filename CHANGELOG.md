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
