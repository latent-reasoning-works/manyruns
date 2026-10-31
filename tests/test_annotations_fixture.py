"""`tests/fixtures/annotations.jsonl` is cut by the real writer and reads back whole."""
from __future__ import annotations

import shutil
from pathlib import Path

from manyruns import annotations

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_the_fixture_reads_back(tmp_path):
    assert FIXTURES.is_dir()
    shutil.copy(FIXTURES / "annotations.jsonl", tmp_path / "annotations.jsonl")
    rows = list(annotations.read(tmp_path))
    assert {r["kind"] for r in rows} == set(annotations.KINDS)
    raw = (tmp_path / "annotations.jsonl").read_text().splitlines()
    assert len(raw) == len(rows)
    for line in raw:
        assert line.startswith('{"schema_version":1,')
    assert len(annotations.examples(tmp_path)) == len(rows)


def test_the_fixture_covers_the_join_edges(tmp_path):
    assert FIXTURES.is_dir()
    shutil.copy(FIXTURES / "annotations.jsonl", tmp_path / "annotations.jsonl")
    rows = list(annotations.read(tmp_path))
    assert any(r["kind"] == "label" and r["run_id"] is None for r in rows)
    assert any(r["chooser"].startswith("llm:") for r in rows)
    corrections = [r for r in rows if r["kind"] == "correction"]
    assert {r["decision_id"] is None for r in corrections} == {True, False}
    joins = {e["join"] for e in annotations.examples(tmp_path)}
    assert joins == {"run_id", "dataset", "decision_id"}
