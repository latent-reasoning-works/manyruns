"""`manyruns explore` / `run` (#12/#13) — the recipe trace + the summary aggregation.

Dep-free: exercises the in-process mock backend on the product-owned CFlows recipe
(normalize → transform → pca → phate → mioflow). No learner import.

The first three steps used to run UNDECLARED inside `loading._anndata_matrix` on every
counts-like load (#54, the cutover); they are recipe steps now, so they appear in the trace
and in the summary table.

What these tests pin is that DECLARED list, not the arithmetic: `engine=mock` reads no data
and invents every number, and says so — the run's only caveat is measured as "engine=mock —
every number here is invented; no data was read". The list is the right thing to pin for the
computing engines too, because `runner._finalize` derives `trace` from one record per
ATTEMPT: a `normalize`/`transform` that DECLINES (`_StepSkipped`, as it does on every
synthetic fixture in this repo — they are Gaussian point clouds, not counts) still appears in
the trace, with the reason in `status`. So the trace is 5 long either way.
"""


def _eb_fixture(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "matrix.h5ad").write_text("placeholder")
    return d


def test_explore_yields_cflows_trace_in_order(tmp_path):
    from manyruns import app

    recipe = app.load_recipe("cflows")
    results = app.run_explorations(_eb_fixture(tmp_path), "scrna", recipe=recipe)

    assert results["served_by"] == "local"
    assert results["engine"] == "mock"
    assert results["recipe"] == "cflows"
    # The pre-declared CFlows trace, in order. Was `[latent:phate, lightning:mioflow]`; the
    # three leading steps are the preamble `loading._anndata_matrix` used to apply invisibly
    # (normalize_total(1e4) → log1p → PCA(50)), now declared in the recipe (#54).
    assert results["trace"] == ["prep:normalize", "prep:transform",
                                "latent:pca", "latent:phate", "lightning:mioflow"]


def test_recipe_path_does_not_disturb_adaptive_loop():
    """A bare-string input still runs the emergent adaptive loop (regression guard)."""
    from manyruns.serving import LocalServer

    out = LocalServer().predict("scrna:some-data")
    assert "recipe" not in out  # adaptive path has no fixed recipe
    assert out["num_steps"] == len(out["trace"]) > 0


def test_summary_has_expected_sections(tmp_path, monkeypatch):
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    data = _eb_fixture(tmp_path)
    recipe = app.load_recipe("cflows")
    results = app.run_explorations(data, "scrna", recipe=recipe)

    summary = app.write_summary("eb", data, "scrna", results, recipe=recipe)
    # path is relative to cwd (chdir'd to tmp_path); resolve before comparing
    assert summary.resolve() == (tmp_path / "outputs" / "eb" / "summary.md").resolve()
    text = summary.read_text()

    assert "modality: **scrna**" in text
    assert "recipe: **cflows**" in text
    # was `pipeline: phate → mioflow` — the summary renders every declared step, so the
    # prep preamble (#54) is now visible to the reader instead of applied behind their back
    assert "pipeline: normalize → transform → pca → phate → mioflow" in text
    assert "`phate`" in text and "`mioflow`" in text
    assert "`normalize`" in text and "`transform`" in text and "`pca`" in text
    assert "## plots" in text
    # the plots dir is created locally (milestone allows plots saved on disk)
    assert (tmp_path / "outputs" / "eb" / "plots").is_dir()


# ── --transform on the argv doors ────────────────────────────────────────────
# `tests/test_loading.py::test_the_transform_flag_reaches_the_declared_step_from_both_front_doors`
# pins the two INTERACTIVE doors (the rich shell and the TUI), both of which reach the flag
# through `app.explore_once`. Nothing reached the argv doors: `run`, `explore`, `init --batch`,
# `init` and `open` all build or load a project dict and hand its recipe to `_explore_project`
# / `_build_session`, none of which ever read `args.transform`. Measured on this checkout before
# the fix, with the recorder below: `manyruns run --transform sqrt <folder>` served the engine
# `transform.method=log1p` and printed nothing about the flag it dropped.


def _method_served(recipe: dict) -> str:
    """The `method` the `transform` step was actually served with."""
    return next(s["params"]["method"] for s in recipe["steps"] if s["name"] == "transform")


def _spy_on_the_served_payload(monkeypatch):
    """Record the payload the engine is served, and still run the real mock backend.

    A wrapper rather than a stub: `_explore_project` writes a summary and appends to the run
    store off the result, so a hand-made return value would test less of the path than the
    engine that ships."""
    from manyruns import app

    real = app._server_for

    class _Spy:
        def __init__(self, inner):
            self.inner, self.inputs = inner, None

        def predict(self, inputs):
            self.inputs = inputs
            return self.inner.predict(inputs)

    spies: list = []
    monkeypatch.setattr(
        app, "_server_for", lambda *a, **k: spies.append(_Spy(real(*a, **k))) or spies[-1]
    )
    return spies


def _spy_on_the_session(monkeypatch):
    """The REPL doors (`init` without `--batch`, `open`) never reach an engine — the session
    drives the executors directly — so their recipe is read off the `Session` instead."""
    from manyruns import app

    built: list = []
    monkeypatch.setattr(app, "interactive_session", lambda s, *a, **k: built.append(s) or 0)
    return built


def _saved(tmp_path, name="saved"):
    """A project.yaml on disk, the way `explore`/`open` find one — written by the product's own
    writer so the shape is whatever `write_project` currently persists."""
    from manyruns import app

    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    data = _eb_fixture(tmp_path / name)
    app.write_project(name, data, "scrna", app.load_recipe("cflows"), engine="mock")
    return data


def test_transform_reaches_the_recipe_from_run(tmp_path, monkeypatch):
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    spies = _spy_on_the_served_payload(monkeypatch)

    rc = app.main(["run", str(_eb_fixture(tmp_path)), "--project", "r", "--engine", "mock",
                   "--transform", "sqrt"])

    assert rc == 0
    assert _method_served(spies[-1].inputs["recipe"]) == "sqrt"


def test_transform_reaches_the_recipe_from_explore(tmp_path, monkeypatch):
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    _saved(tmp_path)
    spies = _spy_on_the_served_payload(monkeypatch)

    rc = app.main(["explore", "--project", "saved", "--engine", "mock", "--transform", "sqrt"])

    assert rc == 0
    assert _method_served(spies[-1].inputs["recipe"]) == "sqrt"


def test_transform_reaches_the_recipe_from_init_batch(tmp_path, monkeypatch):
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    spies = _spy_on_the_served_payload(monkeypatch)

    rc = app.main(["init", str(_eb_fixture(tmp_path)), "--project", "b", "--engine", "mock",
                   "--batch", "--transform", "sqrt"])

    assert rc == 0
    assert _method_served(spies[-1].inputs["recipe"]) == "sqrt"


def test_transform_reaches_the_recipe_from_the_init_repl(tmp_path, monkeypatch):
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    built = _spy_on_the_session(monkeypatch)

    rc = app.main(["init", str(_eb_fixture(tmp_path)), "--project", "i", "--engine", "mock",
                   "--transform", "sqrt"])

    assert rc == 0
    assert _method_served(built[-1].recipe) == "sqrt"


def test_transform_reaches_the_recipe_from_open(tmp_path, monkeypatch):
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    _saved(tmp_path)
    built = _spy_on_the_session(monkeypatch)

    rc = app.main(["open", "--project", "saved", "--engine", "mock", "--transform", "sqrt"])

    assert rc == 0
    assert _method_served(built[-1].recipe) == "sqrt"


def test_the_project_file_records_the_transform_that_ran(tmp_path, monkeypatch):
    """`project.yaml` is what `explore`/`open` replay, so a flag applied only in memory would
    make the record disagree with the run it is the record of — the same project name, re-run
    tomorrow, quietly preprocessing differently. The shim is applied before `write_project`."""
    from omegaconf import OmegaConf

    from manyruns import app

    monkeypatch.chdir(tmp_path)
    _spy_on_the_served_payload(monkeypatch)

    app.main(["run", str(_eb_fixture(tmp_path)), "--project", "rec", "--engine", "mock",
              "--transform", "sqrt"])

    saved = OmegaConf.to_container(OmegaConf.load(tmp_path / "outputs" / "rec" / "project.yaml"))
    assert _method_served(saved["recipe"]) == "sqrt"


def test_no_transform_flag_leaves_the_recipe_declaration_alone(tmp_path, monkeypatch):
    """The control. `--transform` defaults to None precisely so an untyped flag never rewrites
    what the recipe declares; a fix that folded the flag in unconditionally would override
    every recipe's own `method` with the default nobody asked for."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    spies = _spy_on_the_served_payload(monkeypatch)

    rc = app.main(["run", str(_eb_fixture(tmp_path)), "--project", "n", "--engine", "mock"])

    assert rc == 0
    declared = _method_served(app.load_recipe("cflows"))
    assert declared == "log1p"                      # what configs/recipe/cflows.yaml says today
    assert _method_served(spies[-1].inputs["recipe"]) == declared
