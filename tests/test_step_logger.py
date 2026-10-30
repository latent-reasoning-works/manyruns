"""The engine's own record of a latent step lands in the run folder.

`RunStepLogger` implements the manylatents `StepLogger` protocol, so `SaveOutputs` writes
`outputs.json` into `<run folder>/engine/<NN>-<step>/`. `NN` is the same index the run folder's
array names use, so a reader pairs `00-phate_emb.npy` with `engine/00-phate/outputs.json`.

Only the tests that need the real engine import it; the logger itself must work without it.
"""
from __future__ import annotations

import datetime as dt

import pytest

from manyruns import artifacts
from manyruns.pipeline.runner import RunStepLogger

RUN = "abc123def456"


def _logger(tmp_path, index=3, step="phate"):
    return RunStepLogger(tmp_path / "outputs" / "project", RUN, index, step)


# ── the protocol ─────────────────────────────────────────────────────────────────────────────

def test_the_step_folder_is_under_the_run_folder(tmp_path):
    got = _logger(tmp_path).get_step_dir(create=True)
    assert got == tmp_path / "outputs" / "project" / "state" / RUN / "engine" / "03-phate"
    assert got.is_dir()


def test_create_false_touches_nothing(tmp_path):
    got = _logger(tmp_path).get_step_dir()
    assert got == tmp_path / "outputs" / "project" / "state" / RUN / "engine" / "03-phate"
    assert not (tmp_path / "outputs").exists()


def test_the_experiment_id_is_the_run_id(tmp_path):
    assert _logger(tmp_path).get_experiment_id() == RUN


def test_the_marker_is_not_complete(tmp_path):
    logger = _logger(tmp_path)
    step_dir = logger.get_step_dir(create=True)
    logger.write_completion_marker(step_dir)

    marker = step_dir / "outputs.done"
    assert marker.is_file()
    stamp = dt.datetime.fromisoformat(marker.read_text().strip())
    assert stamp.utcoffset() == dt.timedelta(0)
    run_folder = tmp_path / "outputs" / "project" / "state" / RUN
    assert run_folder.is_dir()
    assert not [p for p in run_folder.rglob("*") if p.name == artifacts.DONE]
    assert artifacts.completed_runs(tmp_path / "outputs" / "project") == []


def test_an_unwritable_marker_does_not_raise(tmp_path):
    assert _logger(tmp_path).write_completion_marker(tmp_path / "missing" / "dir") is None
    assert not (tmp_path / "missing").exists()


def test_the_protocol_check_passes(tmp_path):
    save_outputs = pytest.importorskip("manylatents.callbacks.embedding.save_outputs")
    assert isinstance(RunStepLogger(tmp_path, RUN, 0, "phate"), save_outputs.StepLogger)


# ── SaveOutputs in the latent step ───────────────────────────────────────────────────────────

EMBED = {"name": "embed", "steps": [{"name": "phate", "group": "latent", "params": {}}]}


@pytest.fixture
def engine(monkeypatch):
    """The real manylatents, with `api.run` replaced: no fit, no network, no dataset."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("manylatents.callbacks.embedding.save_outputs")
    api = pytest.importorskip("manylatents.api")

    def fake_run(**kwargs):
        return {"embeddings": np.zeros((20, 3)), "scores": {}}

    monkeypatch.setattr(api, "run", fake_run)
    return np


def _session(tmp_path, np):
    from manyruns.session import Session

    return Session(project="p", engine="manylatents", out_dir=tmp_path, modality="scrna",
                   recipe=EMBED, array=np.random.default_rng(0).normal(size=(20, 4)),
                   seed=0, metrics=[])


def test_outputs_json_lands_beside_the_arrays(tmp_path, engine):
    session = _session(tmp_path, engine)
    rec = session.step({"name": "phate", "group": "latent", "params": {}})["record"]

    assert rec["outcome"] == "ok"
    step_dir = artifacts.root(tmp_path, session.run_id) / "engine" / "00-phate"
    assert (step_dir / "outputs.json").is_file()
    assert (step_dir / "outputs.done").is_file()
    assert (artifacts.root(tmp_path, session.run_id) / "00-phate_emb.npy").is_file()


def test_a_failing_write_does_not_fail_the_step(tmp_path, engine, monkeypatch):
    from manylatents.callbacks.embedding.save_outputs import SaveOutputs

    step = {"name": "phate", "group": "latent", "params": {}}
    normal = _session(tmp_path / "a", engine)
    normal_rec = normal.step(dict(step))["record"]

    def boom(self, dataset, embeddings):
        raise OSError("disk full")

    monkeypatch.setattr(SaveOutputs, "on_latent_end", boom)
    failing = _session(tmp_path / "b", engine)
    failing_rec = failing.step(dict(step))["record"]

    assert normal_rec["outcome"] == failing_rec["outcome"] == "ok"
    assert set(normal.g) == set(failing.g)


def test_a_failing_import_creates_no_engine_folder(tmp_path, engine, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "manylatents.callbacks.embedding.save_outputs", None)
    session = _session(tmp_path, engine)
    rec = session.step({"name": "phate", "group": "latent", "params": {}})["record"]

    assert rec["outcome"] == "ok"
    assert not (artifacts.root(tmp_path, session.run_id) / "engine").exists()


def test_complete_keeps_its_meaning(tmp_path, engine):
    session = _session(tmp_path, engine)
    session.step({"name": "phate", "group": "latent", "params": {}})
    assert (artifacts.root(tmp_path, session.run_id) / "engine" / "00-phate").is_dir()
    assert session.run_id not in artifacts.completed_runs(tmp_path)

    session.close()
    assert session.run_id in artifacts.completed_runs(tmp_path)


def test_a_fake_manylatents_package_is_tolerated(tmp_path, monkeypatch):
    """`tests/test_manylatents_engine.py` replaces `manylatents` with a module that has only
    `api`. The step must still pass and leave no `engine` folder, even when an earlier test
    already imported the real `save_outputs` into this process."""
    import sys
    import types

    np = pytest.importorskip("numpy")
    pytest.importorskip("manylatents.callbacks.embedding.save_outputs")   # cached, as in a suite

    api = types.ModuleType("manylatents.api")
    api.run = lambda **kwargs: {"embeddings": np.zeros((20, 3)), "scores": {}}
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)

    session = _session(tmp_path, np)
    rec = session.step({"name": "phate", "group": "latent", "params": {}})["record"]
    assert rec["outcome"] == "ok"
    assert not (artifacts.root(tmp_path, session.run_id) / "engine").exists()
