"""The CFlows recipe ships in the wheel and declares the expected ordered steps.

Mirrors test_smoke.py's importlib.resources pattern — dep-free, no learner import.
"""
from importlib import resources
from pathlib import Path

import pytest


def test_cflows_recipe_bundled_and_ordered():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    cfg_path = Path(str(resources.files("manyruns") / "configs" / "recipe" / "cflows.yaml"))
    assert cfg_path.is_file()

    recipe = OmegaConf.load(cfg_path)
    assert recipe.name == "cflows"

    names = [s.name for s in recipe.steps]
    groups = [s.group for s in recipe.steps]
    # granger was deleted from this recipe — see the long
    # comment in cflows.yaml. The trajectory recipe embeds and flows; it makes no
    # directional claim, because direction is not identifiable from a static snapshot.
    #
    # The list GREW BY THREE AT THE CUTOVER (was `["phate", "mioflow"]`), and the three are
    # not new work: `normalize → transform → pca(50)` is the preamble
    # `loading._anndata_matrix` used to run on every counts-like load, in no trace, no
    # g-vector and no caveat. Declaring it here is what makes it visible and settable, so the
    # step list starting at `phate` was exactly the reading this test should no longer accept.
    assert names == ["normalize", "transform", "pca", "phate", "mioflow"]
    assert groups == ["prep", "prep", "latent", "latent", "lightning"]
    assert "granger" not in names
