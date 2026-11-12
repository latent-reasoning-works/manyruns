"""Bundled recipe steps are flow mappings that start with `- {`, never block mappings."""

from importlib import resources
from pathlib import Path

import pytest
import yaml


def _bundled_recipes():
    directory = Path(str(resources.files("manyruns") / "configs" / "recipe"))
    assert directory.is_dir(), directory
    recipes = sorted(directory.glob("*.yaml"))
    assert len(recipes) >= 13
    return recipes


@pytest.mark.parametrize("recipe", _bundled_recipes(), ids=lambda path: path.name)
def test_bundled_steps_are_flow_mappings(recipe):
    """Each step starts its line with `- {` and is one flow mapping.

    A flow mapping may continue on following lines (several recipes wrap `limits:`),
    but a step never uses the block form `- name: x` with keys on later lines.
    """
    text = recipe.read_text(encoding="utf-8")
    document = yaml.compose(text)
    steps = next(value for key, value in document.value if key.value == "steps")
    assert isinstance(steps, yaml.SequenceNode)
    lines = text.splitlines()
    for step in steps.value:
        location = f"{recipe.name}:{step.start_mark.line + 1}"
        assert lines[step.start_mark.line].lstrip().startswith("- {"), location
        assert isinstance(step, yaml.MappingNode) and step.flow_style, location


def test_profile_contents_are_unchanged():
    profile = next(path for path in _bundled_recipes() if path.name == "profile.yaml")
    assert yaml.safe_load(profile.read_text(encoding="utf-8")) == {
        "name": "profile",
        "id": "data.profile.shape",
        "kind": "profile",
        "source": {
            "kind": "none",
            "note": "Descriptive shape profile, definition profile/1.",
        },
        "question": "Describe the shape — report scale-dependent geometry",
        "suits": [],
        "steps": [
            {
                "name": "dataset_features",
                "group": "analysis",
                "params": {"roles": {}, "layer": None, "overrides": None},
                "via": "manyruns",
            },
            {
                "name": "profile_prepare",
                "group": "prep",
                "params": {
                    "max_components": 50,
                    "center": True,
                    "scale": False,
                    "dtype": "float64",
                    "solver": "full",
                },
                "via": "manylatents",
            },
            {
                "name": "shape_profile",
                "group": "analysis",
                "params": {"extended": False},
                "via": "manyruns",
            },
        ],
    }
