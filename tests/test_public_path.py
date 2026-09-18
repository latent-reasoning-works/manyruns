"""Real engine + paste-a-filepath loading. The numeric compute needs a scientific
stack (phate/scipy/statsmodels) not present in CI, so here we cover the parts that are
dep-free: file/generator loading, backend selection, payload wiring, and graceful
errors. The heavy pipeline is validated on a machine with the stack installed.
"""
import pytest

from manyruns import app
from manyruns.pipeline import load_array
from manyruns.serving import LocalServer


def _write(path, text):
    path.write_text(text)
    return path


def test_load_array_from_py_generator(tmp_path):
    gen = _write(tmp_path / "swissroll.py", "def load():\n    return [[1.0, 2.0], [3.0, 4.0]]\n")
    assert load_array(gen) == [[1.0, 2.0], [3.0, 4.0]]


def test_load_array_py_module_attr(tmp_path):
    gen = _write(tmp_path / "gen.py", "X = [[1, 2, 3]]\n")
    assert load_array(gen) == [[1, 2, 3]]


def test_load_array_py_datamodule_class(tmp_path):
    """A manylatents-style DataModule class (no load()) is detected and instantiated."""
    gen = _write(
        tmp_path / "dm.py",
        "class MyDataModule:\n"
        "    def setup(self, stage=None):\n"
        "        self.train_dataset = [{'data': [1.0, 2.0]}, {'data': [3.0, 4.0]}]\n"
        "    def train_dataloader(self):\n"
        "        return None\n",
    )
    obj = load_array(gen)
    assert type(obj).__name__ == "MyDataModule"  # returns the instance; as_matrix extracts


def test_datamodule_extracted_to_matrix(tmp_path):
    """With numpy present, the DataModule instance coerces to the stacked matrix."""
    import importlib.util

    if importlib.util.find_spec("numpy") is None:
        pytest.skip("numpy not installed")
    from manyruns.pipeline import as_matrix

    gen = _write(
        tmp_path / "dm.py",
        "class MyDataModule:\n"
        "    def setup(self, stage=None):\n"
        "        self.train_dataset = [{'data': [1.0, 2.0]}, {'data': [3.0, 4.0]}]\n"
        "    def train_dataloader(self):\n"
        "        return None\n",
    )
    X = as_matrix(load_array(gen))
    assert X.shape == (2, 2)
    assert X[1].tolist() == [3.0, 4.0]


def test_load_array_py_without_entrypoint_errors(tmp_path):
    bad = _write(tmp_path / "empty.py", "FOO = 1\n")
    with pytest.raises(ValueError, match="no recognized data entrypoint"):
        load_array(bad)


def test_load_array_unsupported_suffix_errors(tmp_path):
    f = _write(tmp_path / "data.weird", "x")
    with pytest.raises(ValueError, match="don't know how to load"):
        load_array(f)


def test_load_array_missing_path_errors(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_array(tmp_path / "nope.csv")


def test_load_array_directory_picks_a_file(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    _write(d / "gen.py", "def generate():\n    return [[9, 8]]\n")
    assert load_array(d) == [[9, 8]]


def test_an_unknown_engine_refuses_instead_of_answering_with_the_mock():
    """What replaced the `real` backend. `predict` used to end in `return self._run_mock(...)`,
    so ANY unrecognised engine — `real` after its removal, a typo, an engine string read back
    off a stored record — was served invented numbers stamped `engine=mock`, while the server's
    own `.engine` still read what the caller asked for. That is the failure `_default_engine`
    refuses to commit at the front door, one layer below it."""
    for name in ("real", "manylatnets", ""):
        srv = LocalServer(engine=name)
        with pytest.raises(ValueError, match="not a serving backend"):
            srv.predict({"recipe": {"name": "cflows", "steps": []}})


def test_the_in_process_step_loop_is_not_a_backend():
    """`vocab.INPROC` is the suite's substrate, and the seam that keeps it one. It has a
    dispatch table (`runner.steppable`) and no server: nothing a user types can route a
    request to it, and `_server_for` cannot be talked into building one."""
    from manyruns.pipeline.runner import steppable
    from manyruns.vocab import INPROC

    assert steppable(INPROC) is True
    assert INPROC not in LocalServer.SERVES
    with pytest.raises(ValueError, match="not a serving backend"):
        LocalServer(engine=INPROC).predict({"recipe": {}})


def test_the_product_path_refuses_a_removed_engine_rather_than_building_a_mock():
    """The frame the product ACTUALLY goes through, which `LocalServer.predict` never sees.

    `app._server_for` ended in an unconditional `return default_server()` — a `LocalServer`
    with `engine="mock"` — so every unrecognised name got a working mock instead of a
    refusal, and `predict`'s guard above was unreachable from the CLI. MEASURED without this
    guard, driving `modes.run("infer", {..., "engine": "real"})` (the shape `_explore_project`
    builds from a stored `project.yaml`, app.py:815): the run completed and returned
    `engine='mock'` with a populated g_vector. An `init --engine real` project written before
    the removal is exactly that request."""
    from manyruns import app

    # The learner joined this list after `real`, for a different reason and by the same
    # mechanism: not "it could not run the catalogue" but "manyruns must not hold a track to
    # it at all" (the environment-contract spec, §3.5). A stored `project.yaml` naming it is
    # the same artifact as one naming `real`, and gets the same refusal.
    for name in ("real", "external-policy", "manylatnets", "_inproc", ""):
        with pytest.raises(ValueError, match="not a serving backend"):
            app._server_for(name)
    # the live ones still build
    assert app._server_for("mock").engine == "mock"
    assert app._server_for("manylatents").engine == "manylatents"


def test_the_two_engine_lists_cannot_drift_apart():
    """`serving.LocalServer.SERVES` and `app.ENGINES` are two hand-written spellings of one
    list, and `serving` may not import `app` (app imports serving). The failure is asymmetric:
    a name in `app` but not in `SERVES` makes the front door offer an engine that refuses,
    and a name in `SERVES` but not in `app` is how `real` survived in the serving layer for a
    whole PR after being deleted from the CLI. Pinned here because `app` may import both."""
    from manyruns import app

    assert set(app.ENGINE_NAMES) == set(LocalServer.SERVES)


class _StubServer:
    def __init__(self):
        self.seen = None

    def predict(self, inputs):
        self.seen = inputs
        return {"served_by": "stub", "engine": "manylatents", "trace": [], "g_vector": {}}


def test_run_explorations_puts_array_and_outdir_in_payload(tmp_path):
    recipe = app.load_recipe("cflows")
    stub = _StubServer()
    app.run_explorations(
        tmp_path / "x.csv",
        "bulk",
        server=stub,
        recipe=recipe,
        engine="manylatents",
        array=[[1, 2], [3, 4]],
        out_dir=tmp_path / "outputs" / "p",
    )
    assert stub.seen["array"] == [[1, 2], [3, 4]]
    assert stub.seen["out_dir"].endswith("outputs/p")


def test_run_inproc_without_stack_errors_cleanly(tmp_path):
    """Missing numpy/phate/etc → a clear RuntimeError, not a bare ModuleNotFoundError.
    (Skips if the scientific stack happens to be installed.)"""
    import importlib.util

    if importlib.util.find_spec("numpy") is not None:
        pytest.skip("numpy present; this guards the not-installed path")
    from manyruns.pipeline import run_inproc

    with pytest.raises(RuntimeError, match="scientific stack"):
        run_inproc([[1.0, 2.0], [3.0, 4.0]], {"name": "cflows", "steps": []}, tmp_path)


def test_detect_modality_on_a_single_file(tmp_path):
    assert app.detect_modality(_write(tmp_path / "m.h5ad", "x")) == "scrna"
    assert app.detect_modality(_write(tmp_path / "m.csv", "x")) == "bulk"
    assert app.detect_modality(_write(tmp_path / "gen.py", "x")) == "unknown"


def test_package_member_import_does_not_leak_sys_path(tmp_path):
    """A pooled worker loading many generator scripts must not grow sys.path unboundedly
    (nor shadow modules for every later run in that process)."""
    import sys

    from manyruns.pipeline import _import_as_package_member

    pkg = tmp_path / "mypkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "gen.py").write_text("from . import __name__ as _n\nX = [[1.0, 2.0]]\n")

    before = list(sys.path)
    mod = _import_as_package_member(pkg / "gen.py")
    assert mod.X == [[1.0, 2.0]]
    assert sys.path == before, "sys.path leaked an entry"


def test_a_stale_project_yaml_naming_a_removed_engine_says_so(tmp_path, monkeypatch):
    """`init --engine real` wrote `engine: real` into a file that outlives the flag, and
    `load_project` is its only reader. Unvalidated, that string reached `_explore_project`
    (app.py:815) and then `_server_for`, which built a mock — so the project kept saying
    `real` while the run reported `mock`. `_inproc` is checked too: `runner.steppable`
    answers True for the substrate, so the REPL branch would have opened a prompt on it."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    d = tmp_path / "outputs" / "stale"
    d.mkdir(parents=True)
    for bad in ("real", "_inproc"):
        (d / "project.yaml").write_text(
            f"project: stale\nengine: {bad}\nmodality: generic\nrecipe: embed\n"
        )
        with pytest.raises(ValueError, match="this build cannot run"):
            app.load_project("stale")
    (d / "project.yaml").write_text(
        "project: stale\nengine: mock\nmodality: generic\nrecipe: embed\n"
    )
    assert app.load_project("stale")["engine"] == "mock"
