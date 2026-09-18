"""Serving backend = a module swap (Hydra _target_), mirroring shop's `cluster=` launcher swap.

A manyruns "model" is an open-ended driver LOOP (not a fixed sequence), so
LocalServer.predict runs a `while` over heterogeneous steps and returns the emergent
trace + a G-vector."""
from pathlib import Path

import pytest

CFG = Path(__file__).resolve().parent.parent / "manyruns" / "configs" / "serving"
assert CFG.is_dir(), f"the config layer moved out from under this test: {CFG}"


def _load(name):
    pytest.importorskip("hydra")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    return instantiate(OmegaConf.load(CFG / f"{name}.yaml"))


def test_local_runs_open_ended_loop():
    from manyruns.serving import LocalServer, ModelServer

    srv = _load("local")
    assert isinstance(srv, LocalServer)
    assert isinstance(srv, ModelServer)  # runtime_checkable protocol

    out = srv.predict("some-data")
    assert out["served_by"] == "local"
    # the loop ran heterogeneous steps to a stopping condition
    assert out["num_steps"] == len(out["trace"]) > 0
    assert any(t.startswith(("latent:", "lightning:")) for t in out["trace"])
    assert any(t.startswith("analysis:") for t in out["trace"])
    assert set(srv.metrics) <= set(out["g_vector"])  # signature complete
    assert "final_dim" in out["g_vector"]            # …plus the shared core schema key
    assert out["final_dim"] <= srv.target_dim


def test_loop_length_is_dynamic():
    """Length depends on the input (more dims => more reduction steps) — not fixed."""
    from manyruns.serving import LocalServer

    short = LocalServer().predict("ab")
    long = LocalServer().predict("a-much-longer-input-string-here")
    assert long["num_steps"] >= short["num_steps"]


def test_swap_to_modal_is_config_only():
    from manyruns.serving import ModalServer, ModelServer

    srv = _load("modal")
    assert isinstance(srv, ModalServer)
    assert isinstance(srv, ModelServer)  # same contract, different infra
    with pytest.raises(NotImplementedError):
        srv.predict("x")


@pytest.mark.parametrize('payload, expected', [
    ({'seed': 0}, 0), ({'seed': 7}, 7), ({'seed': None}, 42), ({}, 42),
])
def test_manylatents_receives_the_explicit_seed_including_zero(monkeypatch, payload, expected):
    from manyruns import pipeline
    from manyruns.serving import LocalServer

    received = {}

    def run(**kwargs):
        received.update(kwargs)
        return {'ok': True}

    monkeypatch.setattr(pipeline, 'run_manylatents', run)
    assert LocalServer(engine='manylatents').predict(payload) == {'ok': True}
    assert received['seed'] == expected
