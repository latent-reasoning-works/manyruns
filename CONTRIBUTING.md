# Contributing

manyruns owns the workflow, config catalog, and run record. Compute belongs in
manylatents. Start with a small issue or pull request that explains what a user
tried, what happened, and what should happen.

## Development

Target pull requests at `dev`, where development work lands once all three CI
checks pass; no review is required. `main` is the stable line and receives release
PRs only. Cut release tags (`v0.1.x`) from `main`. Both branches require the three
checks and refuse force-pushes and deletion.

Consumers depending on manyruns from Git should pin a release tag. Use `dev` only
while co-developing a change.

Use macOS or Linux with Python 3.11–3.12:

```sh
git clone https://github.com/latent-reasoning-works/manyruns
cd manyruns
git switch dev
uv sync --extra harness --extra dev
uv run pytest tests/ -q
uv run ruff check manyruns tests
```

`dev` is an extra, not a dependency group. On later syncs, name every extra already
in use, including `--extra agents` if you use it (installs `manyagents>=0.0.1` from PyPI).
Declare dependency changes in `pyproject.toml`, run `uv lock`, then sync with the full
extra list. Do not install
packages into the checkout environment out of band.

Write a failing test for a behavior change or bug fix before changing the code.
Run the affected tests while working, then the full suite and lint before sending
a pull request. Use mock runs for workflow tests and the in-process test substrate
for artifact tests. A mock result is not scientific evidence.

The [environment contract](docs/environment-contract.md) is one-way: a learner may
read the artifacts and drive the CLI from outside. manyruns must never name or
import a learner, depend on one, or expose a config or engine path to one. Use the
role in prose too. The hash-based guard checks every tracked file without printing
forbidden names. Read [CLAUDE.md](CLAUDE.md) for the layering rules.

This is an open beta. The CLI and record schema can still change; describe any
compatibility impact in the pull request and changelog. Include `manyruns --version`
output, platform, and Python version when reporting a problem. Share a
small synthetic reproduction instead of private data or credentials.

## Maintainers: releases

Keep the GitHub Actions environment **release** protected with a **required
reviewer**. PyPI trusted publishing must use project **manyruns**, owner
**latent-reasoning-works**, repository **manyruns**, workflow **release.yml**, and
environment **release**. Before the project exists on PyPI, configure these values
as a pending publisher. No API token or publishing secret is needed; Actions
exchanges its OIDC identity for publishing access.

1. On `main`, confirm the release version in `pyproject.toml` and
   `manyruns/__init__.py`.
2. Run `release.yml` manually on `main` with `publish: false`. Review the wheel and
   sdist verification jobs and SHA-256 hashes in the build summary.
3. Cut and push the matching tag `vX.Y.Z` from `main`. A `v*` tag triggers
   publication; a manual run publishes only when `publish: true`.
4. After build and verification both pass, the required reviewer approves the
   deployment to the **release** environment.

The publish job has only `id-token: write`; the separate GitHub release job has
`contents: write` and attaches the same artifacts using `GITHUB_TOKEN`.

After publishing, check a clean consumer install on the supported platforms:

```sh
uv tool install --python 3.12 manyruns
manyruns --version
manyruns check
```

Do not announce on social media for this open beta. Track feedback in the public
issues and changes in [CHANGELOG.md](CHANGELOG.md).
