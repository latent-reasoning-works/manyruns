"""Milestone 1 acceptance test (#14): a full sample run on EB data.

Dep-free end-to-end: `manyruns run <eb-fixture>` detects scRNA, runs the CFlows
recipe on the mock backend, and writes a markdown summary. Real EB loading +
compute is delegated to the private stack (manylatents-omics / the learner) and is
NOT exercised here — see the learner's bundled experiment config.
"""
from pathlib import Path

EB_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eb"


def test_eb_sample_run_end_to_end(tmp_path, monkeypatch):
    from manyruns import app

    monkeypatch.chdir(tmp_path)

    # `--recipe cflows` is forced. This test is the Milestone-1 acceptance run — a full
    # three-step pipeline end to end — not a test of which recipe gets auto-selected. The eb
    # fixture is a placeholder file with no readable time axis, so auto-selection now
    # (correctly, #26) returns `embed`, and asserting a trajectory here pinned the guess
    # rather than the case.
    # `--engine mock` is pinned. This test is dep-free by construction — the fixture is a
    # 12-byte placeholder, not a real .h5ad — so it only ever passed because the machine
    # running it had no private stack. `_default_engine()` prefers manylatents when it is
    # importable, which made the outcome a property of the developer's install: green on
    # CI, an h5py 'file signature not found' traceback locally. Pin the backend the
    # docstring already claims.
    rc = app.main(["run", str(EB_FIXTURE), "--project", "eb", "--recipe", "cflows",
                   "--engine", "mock"])
    assert rc == 0

    summary = tmp_path / "outputs" / "eb" / "summary.md"
    assert summary.is_file()

    text = summary.read_text()
    assert "modality: **scrna**" in text
    # was `pipeline: phate → mioflow` — 2 steps, now 5. The cutover (#54) gave `cflows` a
    # DECLARED `normalize → transform → pca(50)` prep block, in place of the undeclared
    # normalize_total → log1p → PCA(50) that `loading._anndata_matrix` used to run on every
    # counts-like load and report nowhere. The summary renders every declared step, so the
    # preamble that was invisible here is now the first three names on this line.
    assert "pipeline: normalize → transform → pca → phate → mioflow" in text
    assert "granger" not in text   # deleted; the summary must not advertise it
    # project.yaml was written by the init half of `run`
    assert (tmp_path / "outputs" / "eb" / "project.yaml").is_file()


def test_bare_invocation_maps_to_run(tmp_path, monkeypatch):
    """`manyruns <folder> --project x` (no subcommand) still works (back-compat)."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    # `--engine mock` is pinned. This test is dep-free by construction — the fixture is a
    # 12-byte placeholder, not a real .h5ad — so it only ever passed because the machine
    # running it had no private stack. `_default_engine()` prefers manylatents when it is
    # importable, which made the outcome a property of the developer's install: green on
    # CI, an h5py 'file signature not found' traceback locally. Pin the backend the
    # docstring already claims.
    rc = app.main([str(EB_FIXTURE), "--project", "eb2", "--engine", "mock"])
    assert rc == 0
    assert (tmp_path / "outputs" / "eb2" / "summary.md").is_file()
