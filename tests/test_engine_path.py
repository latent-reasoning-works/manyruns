"""What `--engine` still accepts, now that one of its values is gone.

THIS FILE USED TO TEST THE LEARNER SEAM — override construction,
backend selection for the learner's `--engine` value, and that a named dataset flowed, all
without importing the private stack. That seam is deleted: the environment-contract spec
§3.5 says manyruns holds no track to a learner, so there is no override to construct and no
backend to select.

What is left is the part that was never about the learner — that `--engine` refuses a name it
cannot serve rather than quietly answering with the mock, and that the flags around it still
behave. The refusal is the load-bearing half, and it is the reason removing an engine is safe
to do at all: an old `project.yaml` naming the removed one now fails loudly.
"""
import pytest

from manyruns import app
from manyruns.serving import LocalServer
from public_safety import forbidden_count


def test_no_learner_is_an_engine():
    """Both surfaces, because they are two lists that a reader could see disagree: the CLI's
    `--engine` choices come from `app.ENGINES`, the server's from `LocalServer.SERVES`."""
    forbidden = sum(forbidden_count(name)
                    for name in (*app.ENGINE_NAMES, *LocalServer.SERVES))
    assert forbidden == 0


def test_asking_for_it_is_refused_rather_than_served_by_the_mock():
    """A `project.yaml` written by an older `init` naming the learner is a real artifact on
    real disks. `_server_for` used to end in `return default_server()`, which answered ANY
    name — so that project would have come back with a full invented g-vector stamped
    `engine=mock` while the file still named the learner. The error names what IS available."""
    with pytest.raises(ValueError, match="not a serving backend") as err:
        app._server_for("external-policy")

    assert "manylatents" in str(err.value) and "mock" in str(err.value)


def test_the_hydra_passthrough_flags_went_with_the_path():
    """`--mode trace` and `--set data=swissroll` existed ONLY to reach the learner's `api.run`.
    Keeping them would leave two flags that parse, are recorded, and reach nothing — which is
    worse than removing them, because a user reading `--help` would believe an override had
    been applied."""
    parser = app._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "x", "--mode", "trace"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "x", "--set", "data=swissroll"])


class _StubServer:
    """Captures what the app hands the backend, so we can assert without running compute."""

    def __init__(self):
        self.seen = None

    def predict(self, inputs):
        self.seen = inputs
        return {"served_by": "stub", "engine": "manylatents", "trace": [], "g_vector": None}


def test_run_explorations_passes_dataset_and_recipe_to_backend():
    """The payload the backend receives, which is the same shape whichever engine serves it."""
    recipe = app.load_recipe("cflows")
    stub = _StubServer()

    app.run_explorations(
        None, "unknown", server=stub, recipe=recipe, engine="manylatents", dataset="swissroll"
    )

    assert isinstance(stub.seen, dict)
    assert stub.seen["dataset"] == "swissroll"
    assert stub.seen["recipe"]["name"] == "cflows"
    assert stub.seen["data"] == "swissroll"  # label is the dataset name


def test_smoke_default_is_full_training():
    """MIOFlow trains 50 epochs by default now; `--smoke` opts into fast_dev_run. `--full`
    is still accepted and always means full."""
    parser = app._build_parser()

    default = parser.parse_args(["run", "x", "--engine", "manylatents"])
    assert app._smoke(default) is False  # default → full (fast_dev_run off)

    smoke = parser.parse_args(["run", "x", "--engine", "manylatents", "--smoke"])
    assert app._smoke(smoke) is True

    full = parser.parse_args(["run", "x", "--engine", "manylatents", "--full"])
    assert app._smoke(full) is False

    # --full wins over --smoke (explicit full request)
    both = parser.parse_args(["run", "x", "--engine", "manylatents", "--smoke", "--full"])
    assert app._smoke(both) is False


def test_init_with_named_dataset_writes_engine_and_dataset(tmp_path, monkeypatch):
    """`--engine` and `--dataset` reach `project.yaml`. Named `manylatents` rather than
    the learner since the removal — `init` validates against `ENGINE_NAMES`, so the old value
    now exits 2 at the parser."""
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    monkeypatch.chdir(tmp_path)
    rc = app.main(["init", "--engine", "manylatents", "--dataset", "swissroll", "--project", "sr"])
    assert rc == 0

    proj = OmegaConf.load(tmp_path / "outputs" / "sr" / "project.yaml")
    assert proj.engine == "manylatents"
    assert proj.dataset == "swissroll"
    assert proj.data_folder is None
    # A named engine dataset exposes no time axis or conditions to the selector, so it gets
    # `embed` (#26). This assertion is incidental to what the test is named for — that
    # --engine and --dataset reach project.yaml — but it should still state the truth.
    assert proj.recipe.name == "embed"
