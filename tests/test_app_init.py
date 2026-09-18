"""`manyruns init` (#11) — minimal project setup, dep-free (no learner import).

Writes outputs/<slug>/project.yaml capturing project name, data folder, detected
modality, and the selected recipe. Uses tmp_path + chdir so nothing touches the repo.
"""
import pytest


def _eb_fixture(tmp_path):
    """A tiny scRNA-like folder (suffix-only detection keys on .h5ad)."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "matrix.h5ad").write_text("placeholder")
    return d


def test_init_writes_project_file(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from manyruns import app

    monkeypatch.chdir(tmp_path)
    data = _eb_fixture(tmp_path)

    # `--engine mock` is pinned. This test is dep-free by construction — the fixture is a
    # 12-byte placeholder, not a real .h5ad — so it only ever passed because the machine
    # running it had no private stack. `_default_engine()` prefers manylatents when it is
    # importable, which made the outcome a property of the developer's install: green on
    # CI, an h5py 'file signature not found' traceback locally. Pin the backend the
    # docstring already claims.
    rc = app.main(["init", str(data), "--project", "EB Study", "--engine", "mock"])
    assert rc == 0

    proj_file = tmp_path / "outputs" / "eb-study" / "project.yaml"
    assert proj_file.is_file()

    proj = OmegaConf.load(proj_file)
    assert proj.project == "EB Study"
    assert proj.modality == "scrna"
    # The fixture exposes no time axis and no conditions, so it gets `embed` (#26). This
    # assertion is incidental to what the test is named for — that `init` writes project.yaml
    # — but it should state the truth rather than the old always-a-trajectory default.
    assert proj.recipe.name == "embed"
    # init does NOT explore — no summary yet
    assert not (tmp_path / "outputs" / "eb-study" / "summary.md").exists()


def test_dataset_arg_accepts_path_or_name(tmp_path, monkeypatch):
    """One data source, auto-detected: --dataset (or positional) resolves to a file path
    when it exists on disk, otherwise to a named dataset."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from manyruns import app

    monkeypatch.chdir(tmp_path)
    data = _eb_fixture(tmp_path)

    # --dataset pointing at an existing PATH → treated as a file/folder
    # `--engine mock` is pinned. This test is dep-free by construction — the fixture is a
    # 12-byte placeholder, not a real .h5ad — so it only ever passed because the machine
    # running it had no private stack. `_default_engine()` prefers manylatents when it is
    # importable, which made the outcome a property of the developer's install: green on
    # CI, an h5py 'file signature not found' traceback locally. Pin the backend the
    # docstring already claims.
    app.main(["init", "--dataset", str(data), "--project", "p-path", "--engine", "mock"])
    p = OmegaConf.load(tmp_path / "outputs" / "p-path" / "project.yaml")
    assert p.dataset is None
    assert p.data_folder == str(data)
    assert p.modality == "scrna"

    # --dataset with a bare NAME → treated as a named dataset. `--engine mock` for the same
    # reason as above, and it was missing here: with no backend installed `init` reports "no
    # compute backend" and returns before it writes project.yaml, so this asserted on a file
    # that was never created. Green on a machine with manylatents, red everywhere else.
    app.main(["init", "--dataset", "swissroll", "--project", "p-name", "--engine", "mock"])
    q = OmegaConf.load(tmp_path / "outputs" / "p-name" / "project.yaml")
    assert q.dataset == "swissroll"
    assert q.data_folder is None


def test_missing_path_like_arg_errors(tmp_path, monkeypatch):
    """A path-shaped source that doesn't exist errors clearly (not treated as a name).

    `--engine mock` because the SUBJECT here is path validation, and with no backend installed
    `init` refuses on the engine first — printing "no compute backend" and returning without
    raising, so the path was never checked and `DID NOT RAISE SystemExit` was the only thing the
    test could report. Pinning the engine is what makes the assertion about the path again.
    """
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        app.main(["init", "./does-not-exist.csv", "--project", "p", "--engine", "mock"])


def test_init_rejects_missing_folder(tmp_path, monkeypatch):
    """`--engine mock` for the reason above: the folder check is the subject."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        app.main(["init", str(tmp_path / "nope"), "--project", "p", "--engine", "mock"])


def test_init_rejects_unsupported_folder(tmp_path, monkeypatch):
    """`--engine mock` for the reason above: what is in the folder is the subject."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        app.main(["init", str(empty), "--project", "p", "--engine", "mock"])
