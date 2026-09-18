"""`manyruns open` / `manyruns projects` — reopen a saved project by NAME, no data source
needed a second time, and list what's been saved.

`open` reuses `_build_session`, the exact helper `init` already builds its own interactive
session from — the only difference is where the project dict comes from (`load_project`,
reading `outputs/<slug>/project.yaml`, instead of `_setup_project`, which prompts for a data
source and rewrites the file). So the dep-free (mock-engine) tests here pin down the wiring —
that `open` finds the saved project and never asks for a path — while the artifact-producing
test uses `engine=manylatents` against a small real `.h5ad` fixture to confirm a step run
through `open` lands its plot and its embedding on disk exactly where `init` would have put
them.
"""
import pytest


def _eb_fixture(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "matrix.h5ad").write_text("placeholder")
    return d


def test_open_reuses_the_saved_project_without_a_data_source(tmp_path, monkeypatch):
    """`init` writes project.yaml once; `open --project NAME` (no path, no --dataset) reads
    it back and builds the same kind of session `init` would have — the whole point of
    `open` is that the data source travels with the saved project, not the invocation."""
    pytest.importorskip("omegaconf")
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    data = _eb_fixture(tmp_path)
    rc = app.main(["init", str(data), "--project", "Demo 1", "--engine", "mock", "--batch"])
    assert rc == 0

    captured = {}

    def _fake_interactive_session(session, read=input, write=print):
        captured["session"] = session
        return 0

    monkeypatch.setattr(app, "interactive_session", _fake_interactive_session)
    # No data_folder/--dataset anywhere on this argv — if `open` needed one, this would
    # either prompt (blocking) or error.
    rc = app.main(["open", "--project", "Demo 1"])
    assert rc == 0

    session = captured["session"]
    assert session.project == "Demo 1"
    assert session.engine == "mock"
    assert session.modality == "scrna"
    assert session.recipe.get("name") == "embed"


def test_open_does_not_rewrite_the_project_file(tmp_path, monkeypatch):
    """`init`/`run` persist via `write_project`; `open` only reads. A second `open` must not
    touch `project.yaml` — if it did, `open` would silently re-derive the recipe/modality
    instead of trusting what `init` decided."""
    pytest.importorskip("omegaconf")
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    data = _eb_fixture(tmp_path)
    app.main(["init", str(data), "--project", "Demo 1", "--engine", "mock", "--batch"])

    proj_file = tmp_path / "outputs" / "demo-1" / "project.yaml"
    before = proj_file.read_text()

    monkeypatch.setattr(app, "interactive_session", lambda session, read=input, write=print: 0)
    app.main(["open", "--project", "Demo 1"])

    assert proj_file.read_text() == before


def test_open_unknown_project_errors_cleanly(tmp_path, monkeypatch):
    pytest.importorskip("omegaconf")
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        app.main(["open", "--project", "does-not-exist"])


def _h5ad_fixture(tmp_path, n=150, d=12, seed=0):
    """A real (not placeholder) .h5ad — a raw float matrix, no obs columns. Named datasets
    (`swissroll` etc.) load through a different path (`pipeline.load_named_dataset`, a
    manylatents DataModule) that the plain `array`/`data_folder` loading this test targets
    does not go through, so a real file is what exercises it."""
    anndata = pytest.importorskip("anndata")
    import numpy as np

    path = tmp_path / "data.h5ad"
    rng = np.random.default_rng(seed)
    anndata.AnnData(X=rng.normal(size=(n, d)).astype("float32")).write_h5ad(path)
    return path


def test_open_a_step_saves_its_plot_and_its_embedding(tmp_path, monkeypatch):
    """The end-to-end case this command exists for: open a saved project by name, run
    `phate` through the REPL, accept it, and find the plot + the embedding it produced on
    disk under the SAME `outputs/<slug>/` tree `init` created.

    `--engine manylatents`, not `real`: it's what `_default_engine()` actually picks (`real`
    exists for dev/no-private-stack use, per `app.DEV_ENGINES`), and, measured directly, only
    `manylatents`'s loader (`pipeline.load_labeled`) returns a plain matrix `Session` can use
    — `real`'s (`pipeline.load_array`, unconverted) hands back the AnnData object itself,
    which has no `.ndim` and is silently dropped to `state["X"] = None`
    (`Session.__init__`'s `array if getattr(array, "ndim", 0) == 2 else None`). That gap is
    pre-existing and independent of `open`/`init` — reproduces identically through
    `manyruns init data.h5ad --engine real` on `main` — and out of scope here."""
    pytest.importorskip("omegaconf")
    pytest.importorskip("manylatents")
    data = _h5ad_fixture(tmp_path)
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    rc = app.main(["init", str(data), "--project", "Demo 1", "--engine", "manylatents"])
    assert rc == 0  # no script fed to stdin — `init` opens and immediately closes on EOF

    # `cmd_open` calls `interactive_session(session)` with no `read=`/`write=` of its own, so
    # it falls back to that function's OWN default (`read=input`) — a default bound once, at
    # `app.py` import time, to the real builtin. Patching `builtins.input` after the fact
    # cannot reach an already-bound default; patching the *function* `cmd_open` looks up (a
    # fresh global lookup on every call) can, so this wraps the real REPL with a scripted
    # read/write and installs the wrapper under `app.interactive_session`.
    real_interactive_session = app.interactive_session
    script = iter(["phate", "accept", "quit"])

    def _scripted(session, read=input, write=print):
        return real_interactive_session(session, read=lambda *_: next(script), write=lambda *_: None)

    monkeypatch.setattr(app, "interactive_session", _scripted)
    rc = app.main(["open", "--project", "Demo 1"])
    assert rc == 0

    out_dir = tmp_path / "outputs" / "demo-1"
    assert (out_dir / "plots" / "phate.png").is_file()
    state_dirs = [p for p in (out_dir / "state").iterdir() if p.is_dir()]
    assert len(state_dirs) == 1
    assert list(state_dirs[0].glob("00-phate_emb.npy"))


def test_reopening_resumes_the_embedding_the_prior_open_computed(tmp_path, monkeypatch):
    """The bug report this feature exists for, through the real `open` command twice in a
    row: `init` -> open #1 runs `phate`, accepts, quits -> open #2 must find that embedding
    in `state["emb"]` with no `phate` run in ITS OWN lineage, since a stochastic step
    (`init`/learner-backed PHATE included) is exactly what a person reopening does not
    want to pay for again.
    """
    pytest.importorskip("omegaconf")
    pytest.importorskip("manylatents")
    data = _h5ad_fixture(tmp_path)
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    rc = app.main(["init", str(data), "--project", "Demo 1", "--engine", "manylatents"])
    assert rc == 0

    real_interactive_session = app.interactive_session

    # First `open`: run `phate`, accept it, quit — same shape as
    # `test_open_a_step_saves_its_plot_and_its_embedding` above.
    first_script = iter(["phate", "accept", "quit"])

    def _first(session, read=input, write=print):
        return real_interactive_session(
            session, read=lambda *_: next(first_script), write=lambda *_: None)

    monkeypatch.setattr(app, "interactive_session", _first)
    rc = app.main(["open", "--project", "Demo 1"])
    assert rc == 0

    # Second `open`: a FRESH process-shaped call (new argv, new Session) that runs nothing —
    # `dpt` cold is what motivated this feature, but the point under test is what the session
    # has in `state["emb"]` before any step of ITS OWN runs at all.
    captured = {}

    def _second(session, read=input, write=print):
        captured["session"] = session
        return real_interactive_session(session, read=lambda *_: "quit", write=lambda *_: None)

    monkeypatch.setattr(app, "interactive_session", _second)
    rc = app.main(["open", "--project", "Demo 1"])
    assert rc == 0

    session = captured["session"]
    assert session.steps == []                     # nothing ran in this (second) lineage
    assert session.state["emb"] is not None
    assert session.state["emb"].shape[0] == 150     # the input's n_samples, PHATE's own axis 0
    # `emb` is the one this test is about, and it is asserted by MEMBERSHIP rather than by
    # equality against the whole list: a resumed step now restores everything the engine handed
    # back, not just its headline array. Measured here — `phate` resumes as
    # `[("phate", "emb", 0), ("phate", "phate.affinity", 0), ("phate", "phate.kernel", 0)]`,
    # because `_collect_extras` keeps the affinity and kernel matrices PHATE computes. Pinning
    # the exact list would make every future extra a failure of the resume feature.
    assert session.resumed
    assert ("phate", "emb", 0) in session.resumed["steps"]
    assert all(step == "phate" for step, _key, _i in session.resumed["steps"])


def test_a_later_clean_open_does_not_shadow_an_older_ones_pseudotime(tmp_path, monkeypatch):
    """THE bug report, through three real `open` commands in a row — no crash, no dirty exit
    anywhere. open #1 runs `phate` then `dpt` and quits clean. open #2 resumes that embedding,
    re-runs `phate` (a fresh fit; never touches `dpt`), and ALSO quits clean. open #3 does
    nothing but open. Before the fix, `open #3` inherited only `open #2`'s manifest — emb,
    no pseudotime — even though `open #1`'s pseudotime is sitting on disk, untouched, one
    folder over. `discretize_time` cold on `open #3` reproduces the exact refusal this test
    guards against."""
    pytest.importorskip("omegaconf")
    pytest.importorskip("manylatents")
    pytest.importorskip("scanpy")
    data = _h5ad_fixture(tmp_path)
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    rc = app.main(["init", str(data), "--project", "Demo 1", "--engine", "manylatents"])
    assert rc == 0

    real_interactive_session = app.interactive_session

    # open #1: phate, then dpt — both computed and persisted in ONE completed run.
    first_script = iter(["phate", "accept", "dpt", "accept", "quit"])

    def _first(session, read=input, write=print):
        return real_interactive_session(
            session, read=lambda *_: next(first_script), write=lambda *_: None)

    monkeypatch.setattr(app, "interactive_session", _first)
    assert app.main(["open", "--project", "Demo 1"]) == 0

    # open #2: resumes #1's embedding, re-fits `phate` (never runs `dpt`), quits clean. Its
    # OWN completed manifest therefore holds `emb` only — the shadow.
    second_script = iter(["phate", "accept", "quit"])

    def _second(session, read=input, write=print):
        return real_interactive_session(
            session, read=lambda *_: next(second_script), write=lambda *_: None)

    monkeypatch.setattr(app, "interactive_session", _second)
    assert app.main(["open", "--project", "Demo 1"]) == 0

    # open #3: does nothing of its own — only what resuming hands it before step 0.
    captured = {}

    def _third(session, read=input, write=print):
        captured["session"] = session
        return real_interactive_session(session, read=lambda *_: "quit", write=lambda *_: None)

    monkeypatch.setattr(app, "interactive_session", _third)
    assert app.main(["open", "--project", "Demo 1"]) == 0

    session = captured["session"]
    assert session.state["emb"] is not None, "open #2's fresh embedding must still resume"
    assert session.state["pseudotime"] is not None, (
        "open #1's pseudotime must survive open #2's unrelated clean completion"
    )
    from manyruns.pipeline.steps import _step_discretize_time

    _step_discretize_time(session.state, {}, {"n_timepoints": 3}, None, [], 3)  # must not raise


def test_projects_lists_saved_projects(tmp_path, monkeypatch, capsys):
    pytest.importorskip("omegaconf")
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    data = _eb_fixture(tmp_path)
    app.main(["init", str(data), "--project", "Demo 1", "--engine", "mock", "--batch"])

    rc = app.main(["projects"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Demo 1" in out
    assert "engine=mock" in out
    assert "embed" in out


def test_projects_with_none_saved_says_so(tmp_path, monkeypatch, capsys):
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    rc = app.main(["projects"])
    assert rc == 0
    assert "no saved projects" in capsys.readouterr().out
