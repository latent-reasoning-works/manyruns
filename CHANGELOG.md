# Changelog

All notable changes to manyruns are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
During the open beta, the CLI and record schema can still change.

## [Unreleased]

### Fixed

- Installation and runtime repair guidance consistently uses uv tool installs for
  the application and `uv sync` for developer checkouts, with uv's official installer
  for first-time setup.

## [0.1.0] - 2026-09-17

First public release: an open beta for macOS and Linux, Python 3.11–3.12.

### Added

- Interactive data exploration, 11 bundled recipes, and the run → label → store
  workflow, available through both `manyruns` and `co-science`.
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
