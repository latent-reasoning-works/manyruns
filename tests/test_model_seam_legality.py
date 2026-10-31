"""Legality reads a tool step's needs and produces from its manifest, not from the name tables."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from manyruns import narrate, toolchain, vocab
import model_seam_fixtures as _seam

golden_dir = _seam.golden_dir

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tools" / "mioflow"

GOLDEN_THEN_BRANCHES = [{"name": "golden", "group": "latent", "tool": "golden"},
                        {"name": "branch_points", "group": "analysis"}]


def test_seam_graph_satisfies_branch_points(golden_dir):
    assert vocab.unmet({"steps": GOLDEN_THEN_BRANCHES}, "manifold") == frozenset()
    untooled = [{k: v for k, v in s.items() if k != "tool"} for s in GOLDEN_THEN_BRANCHES]
    assert vocab.unmet({"steps": untooled}, "manifold") == frozenset({"graph"})


def test_narrowing_clears_seam_fact(golden_dir):
    recipe = {"steps": [GOLDEN_THEN_BRANCHES[0], {"name": "filter_cells", "group": "prep"},
                        GOLDEN_THEN_BRANCHES[1]]}
    assert vocab.unmet(recipe, "manifold") == frozenset({"graph"})
    assert vocab.invalidated(recipe, "manifold") == frozenset({"graph"})


def test_manifest_facts_reads_the_manifest(golden_dir):
    needs, produces = toolchain.manifest_facts("golden")
    assert needs == frozenset()
    assert produces == frozenset({"embedding", "clusters", "pseudotime", "trajectories",
                                  "graph", "class_probabilities"})
    with pytest.raises(ValueError):
        toolchain.manifest_facts("no_such_tool")


def test_step_needs_falls_back_to_the_name_table():
    assert vocab.step_needs({"name": "branch_points", "group": "analysis"}) == frozenset(
        {"graph"})
    assert vocab.step_needs({"name": "unknown", "group": "analysis"}) == frozenset()


@pytest.mark.source_checkout
def test_manifest_replaces_name_table(monkeypatch):
    assert EXAMPLE.is_dir(), "the example tool folder moved"
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(EXAMPLE))
    # `growth_rate` reads the fitted `model`, which the tool keeps in its own interpreter: the
    # manifest produces only `trajectories`, while the name table's `mioflow` also produces a
    # `model`. (`sample_trajectories` reads `trajectories`, which both supply.)
    steps = [{"name": "mioflow", "group": "lightning", "tool": "mioflow"},
             {"name": "growth_rate", "group": "probe"}]
    given = {"embedding", "time"}
    assert vocab.unmet({"steps": steps}, "manifold", provided=given) == frozenset({"model"})
    untooled = [{k: v for k, v in s.items() if k != "tool"} for s in steps]
    assert vocab.unmet({"steps": untooled}, "manifold", provided=given) == frozenset()


@pytest.mark.source_checkout
def test_mioflow_tool_step_needs_time(monkeypatch):
    assert EXAMPLE.is_dir(), "the example tool folder moved"
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(EXAMPLE))
    steps = [{"name": "mioflow", "group": "lightning", "tool": "mioflow"}]
    assert vocab.unmet({"steps": steps}, "manifold", provided={"embedding"}) == frozenset(
        {"time"})


def test_vocab_import_stays_light():
    out = subprocess.run(
        [sys.executable, "-c",
         "import manyruns.vocab, sys; print(json.dumps('manyruns.toolchain' in sys.modules))"
         .replace("sys;", "sys, json;")],
        capture_output=True, text=True, check=True).stdout
    assert json.loads(out) is False


def test_every_fact_has_a_phrase():
    assert set(narrate.MISSING_FACT) >= {"pseudotime", "trajectories", "class_probabilities"}
    assert set(narrate.CLEARED_FACT) >= {"pseudotime", "trajectories", "class_probabilities"}
    assert vocab.facts() >= {"pseudotime", "trajectories", "class_probabilities", "graph"}
    assert vocab.step_facts() >= set(vocab.TOOL_OUTPUT_FACTS)
