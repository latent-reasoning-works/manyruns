"""Smoke tests for the manyruns product seam.

These used to be phrased as "do NOT import the learner, so they run in CI without the private
ecosystem" — a discipline the whole suite had to keep. It is no longer a discipline: nothing
in the package can import a learner, because nothing in the package names one (the
environment-contract spec, §3.5). What is left to smoke-test is that the product's own
surfaces are importable and its own config layer ships.
"""
from importlib import resources
from pathlib import Path
from public_safety import forbidden_count


def test_app_exposes_main():
    from manyruns import app

    assert callable(app.main)


def test_the_train_seam_is_gone_and_the_stage_remains():
    """`manyruns/train.py` was a passthrough to the learner's `api.run` — the track itself. The
    STAGE survives it: the harness workflow still has a train step to record, it just stops at
    the store instead of calling anyone."""
    import pytest

    from manyruns import modes

    with pytest.raises(ImportError):
        from manyruns import train  # noqa: F401

    assert "train" in modes.MODES


def test_config_layer_is_bundled():
    """manyruns's OWN config layer, which is what `importlib.resources` has to reach for an
    installed wheel to work (CLAUDE.md, Config and record handling).

    It used to check a bundled learner config tree — a copy of the ENGINE's Hydra tree, composed
    by the deleted `train.py`. Both are gone. `configs/recipe/` is the layer that was always
    manyruns's own, and the one a run actually reads."""
    cfg_dir = Path(str(resources.files("manyruns") / "configs"))
    assert (cfg_dir / "recipe" / "cflows.yaml").is_file()
    assert (cfg_dir / "dataset").is_dir()
    assert cfg_dir.is_dir()
    assert sum(forbidden_count(str(p.relative_to(cfg_dir)))
               for p in cfg_dir.rglob("*")) == 0
