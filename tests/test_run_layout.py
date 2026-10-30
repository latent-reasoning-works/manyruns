"""The run folder layout is written down as the join contract.

A reader joins every record of one run on `run_id`. Section 8 of the environment contract
lists each path that carries it and the writer of each, and this test fails when a path is
missing from that section.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.source_checkout

HEADING = "## 8 · The run folder and the join key"

PATHS = [
    "outputs/index.jsonl",
    "outputs/decisions.jsonl",
    "outputs/<slug>/state/<run_id>/NN-<step>_<slot>.npy",
    "outputs/<slug>/state/<run_id>/COMPLETE",
    "outputs/<slug>/state/<run_id>/agent/messages.jsonl",
    "outputs/<slug>/state/<run_id>/engine/<NN>-<step>/outputs.json",
    "outputs/<slug>/state/<run_id>/engine/<NN>-<step>/outputs.done",
    "<MANYRUNS_TRACE dir>/<run_id>.jsonl",
]

WRITERS = ["store.append", "decisions.append", "artifacts.save_array", "runner.finish_carry",
           "agents.propose", "SaveOutputs", "RunStepLogger.write_completion_marker",
           "trace.JsonlTracer"]


def _section() -> str:
    docs = Path(__file__).resolve().parents[1] / "docs"
    assert docs.is_dir()
    text = (docs / "environment-contract.md").read_text()
    assert HEADING in text, "section 8 is missing"
    return text.split(HEADING, 1)[1]


def test_every_path_is_documented_after_the_heading():
    section = _section()
    missing = [p for p in PATHS if p not in section]
    assert missing == []


def test_every_writer_is_named_and_run_id_is_the_join_key():
    section = _section()
    assert [w for w in WRITERS if w not in section] == []
    assert "join key" in section and "`run_id`" in section


def test_section_8_is_the_last_section():
    assert "\n## " not in _section()
