# Contributing

manyruns owns the workflow, config catalog, and run record. Compute belongs in
manylatents. Start with a small issue or pull request that explains what a user
tried, what happened, and what should happen.

## Development

Target pull requests at `dev`, where development work lands once all three CI
checks pass; no review is required. `main` is the stable line. It receives only the
sync pull request from `dev` (see [Maintainers: sync `dev` into `main`](#maintainers-sync-dev-into-main)).
Cut release tags (`v0.1.x`) from `main`. Both branches require the three checks and
refuse force-pushes and deletion.

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

Run `git config core.hooksPath .githooks` once in each clone. It turns on the
name hook, which runs the same hash-based matcher before a commit exists. The hook
blocks a commit whose staged file paths, staged file contents or commit message hold
a forbidden name. It prints only the locations, never the name.

This is an open beta. The CLI and record schema can still change; describe any
compatibility impact in the pull request and changelog. Include `manyruns --version`
output, platform, and Python version when reporting a problem. Share a
small synthetic reproduction instead of private data or credentials.

## Maintainers: sync `dev` into `main`

A merge into `main` publishes nothing. Only a `v*` tag, or a manual `release.yml` run
with `publish: true`, publishes. Sync `dev` into `main` whenever `dev` has new merges
and its CI passes, so that the Git install of `main` stays current.

1. Fetch both branches:

   ```sh
   git fetch origin
   ```

2. List what moves. If the output is empty, `main` already holds all of `dev`; stop.

   ```sh
   git diff --stat origin/main origin/dev
   ```

3. Make a sync branch from `main`, named for the day:

   ```sh
   git switch -c sync/dev-YYYY-MM-DD origin/main
   ```

4. Make its files exactly the files of `dev`. This also deletes the files that `dev`
   deleted:

   ```sh
   git restore --source=origin/dev --staged --worktree :/
   ```

5. Commit and push the branch:

   ```sh
   git commit -m "Sync dev into main: <summary> (#NN, #NN)" && git push -u origin HEAD
   ```

6. Open a pull request from the sync branch into `main`. Do not change the version.
7. When the three checks pass, merge it as a **squash**. Do not push a tag.
8. Confirm that the two branches hold the same files. The command exits 0 when they do:

   ```sh
   git fetch origin && git diff --quiet origin/main origin/dev
   ```

Do not open the pull request from `dev` itself. Each sync is a squash, so `main` and
`dev` hold the same files but share no recent commit. GitHub then merges from an old
common commit, and files that both branches changed, such as CHANGELOG.md, conflict.
The sync branch starts from `main`, so its pull request has no conflict. For the same
reason, compare the branches with `git diff`, never with `git log main..dev`.

Never commit to `main` by another route: a fix goes to `dev` first and reaches `main`
with the next sync. A release is a sync followed by the steps in the next section.

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

## License of contributions

Unless you state otherwise, any contribution you intentionally submit for inclusion in manyruns is licensed under the Apache License, Version 2.0, as section 5 of the license describes, without any additional terms or conditions. That includes the license's patent grant (section 3).
