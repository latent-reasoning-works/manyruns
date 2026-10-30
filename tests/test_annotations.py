"""The annotation store: judgments about runs, kept apart from committed choices."""
from __future__ import annotations

import json
import re

import pytest

from manyruns import annotations

KEYS = ["schema_version", "annotation_id", "kind", "at", "chooser", "run_id", "decision_id",
        "step", "should", "payload"]
LABEL = {"dataset": "pbmc3k", "topology": ["clusters"], "raw": []}


def _rows(tmp_path):
    path = tmp_path / "annotations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# ── the module's line ─────────────────────────────────────────────────────────
def test_docstring_carries_the_line():
    doc = annotations.__doc__
    for phrase in ("not a committed choice", "no data", "no coordinates", "no counts",
                   "interaction telemetry", "human:<pseudonym>", "llm:<model>"):
        assert phrase in doc, phrase


def test_kinds_are_fixed():
    assert annotations.KINDS == ("label", "rating", "correction")


def test_index_path(tmp_path):
    assert annotations.index_path(tmp_path) == tmp_path / "annotations.jsonl"
    assert str(annotations.index_path()) == "outputs/annotations.jsonl"


# ── append ───────────────────────────────────────────────────────────────────
def test_label_row_is_written_with_schema_version_first(tmp_path):
    aid = annotations.append("label", run_id="abc123def456", chooser="human:a3",
                             payload=LABEL, out_dir=tmp_path)
    rows = _rows(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert list(row) == KEYS
    assert row["schema_version"] == 1
    assert row["at"].endswith("+00:00")
    assert aid == row["annotation_id"]
    assert re.fullmatch(r"[0-9a-f]{12}", aid)
    assert row["kind"] == "label" and row["run_id"] == "abc123def456"
    assert row["decision_id"] is None and row["step"] is None and row["should"] is None
    assert row["payload"] == LABEL


def test_decision_and_run_files_are_not_touched(tmp_path):
    assert annotations.append("label", run_id="abc123def456", chooser="human:a3",
                              payload=LABEL, out_dir=tmp_path)
    assert not (tmp_path / "decisions.jsonl").exists()
    assert not (tmp_path / "index.jsonl").exists()


def test_unwritable_directory_returns_none(tmp_path):
    (tmp_path / "f").write_text("a regular file")
    out = tmp_path / "f" / "out"
    assert annotations.append("label", run_id="r", chooser="human:a3", payload={},
                              out_dir=out) is None


def test_a_correction_row(tmp_path):
    annotations.append("correction", run_id="r1", chooser="human:a3", payload={"did": "phate"},
                       decision_id="d1", step=0, should="umap", out_dir=tmp_path)
    row = _rows(tmp_path)[0]
    assert (row["step"], row["should"], row["decision_id"]) == (0, "umap", "d1")


@pytest.mark.parametrize("chooser", ["llm:org/model-4b@1cfa9a72", "llm:local-model-7b",
                                     "human:a3", "human:scshapebench_a4", "human:x-y_z9"])
def test_accepted_choosers(tmp_path, chooser):
    assert annotations.append("label", run_id="r", chooser=chooser, payload={},
                              out_dir=tmp_path)
    assert _rows(tmp_path)[0]["chooser"] == chooser


def test_an_llm_rating_is_accepted(tmp_path):
    payload = {"mode": "blind", "blind": True, "accuracy": 4, "completeness": 3}
    assert annotations.append("rating", run_id="r", chooser="llm:local-model-7b",
                              payload=payload, out_dir=tmp_path)
    rows = _rows(tmp_path)
    assert len(rows) == 1 and rows[0]["chooser"] == "llm:local-model-7b"


@pytest.mark.parametrize("kwargs", [
    {"kind": "approval"},
    {"chooser": "human:Jane Doe"},
    {"chooser": "a3"},
    {"chooser": "expert:a3"},
    {"chooser": "human"},
    {"chooser": "human:"},
    {"chooser": "human:" + "a" * 33},
    {"chooser": "llm:"},
    {"chooser": "llm:has space"},
    {"chooser": None},
    {"kind": "correction", "should": "umap"},
    {"kind": "correction", "step": -1, "should": "umap"},
    {"kind": "correction", "step": True, "should": "umap"},
    {"kind": "correction", "step": 1.0, "should": "umap"},
    {"kind": "correction", "step": 0},
    {"kind": "correction", "step": 0, "should": ""},
    {"step": 2},
    {"should": "umap"},
    {"payload": ["clusters"]},
    {"payload": {"topology": ["cluster"]}},
    {"payload": {"topology": []}},
])
def test_append_refuses(tmp_path, kwargs):
    args = {"kind": "label", "run_id": "r", "chooser": "human:a3", "payload": {},
            "out_dir": tmp_path}
    args.update(kwargs)
    kind = args.pop("kind")
    with pytest.raises(ValueError):
        annotations.append(kind, **args)
    assert not (tmp_path / "annotations.jsonl").exists()


def test_pseudonym_pattern_is_the_part_after_human():
    pat = re.compile(annotations.PSEUDONYM_PATTERN)
    assert pat.fullmatch("a3")
    assert pat.fullmatch("scshapebench_a4")
    assert not pat.fullmatch("Jane")
    assert not pat.fullmatch("jane doe")
    assert not pat.fullmatch("_a")
    assert not pat.fullmatch("a" * 33)
    assert pat.fullmatch("a" * 32)


# ── read ─────────────────────────────────────────────────────────────────────
def test_read_missing_file_yields_nothing(tmp_path):
    assert list(annotations.read(tmp_path)) == []


def test_a_half_written_line_is_skipped(tmp_path):
    annotations.append("label", run_id="r1", chooser="human:a3", payload={}, out_dir=tmp_path)
    with open(tmp_path / "annotations.jsonl", "a") as fh:
        fh.write('{"kind": "lab\n')
        fh.write("[1, 2]\n")
        fh.write("\n")
        fh.write('"a string"\n')
    annotations.append("label", run_id="r2", chooser="human:a3", payload={}, out_dir=tmp_path)
    rows = list(annotations.read(tmp_path))
    assert [r["run_id"] for r in rows] == ["r1", "r2"]


def test_a_row_without_a_version_reads_as_version_1(tmp_path):
    (tmp_path / "annotations.jsonl").write_text('{"kind": "label", "future_key": 7}\n')
    [row] = list(annotations.read(tmp_path))
    assert row["schema_version"] == 1 and row["future_key"] == 7


def test_an_explicit_version_is_kept(tmp_path):
    (tmp_path / "annotations.jsonl").write_text('{"schema_version": 2, "kind": "label"}\n')
    [row] = list(annotations.read(tmp_path))
    assert row["schema_version"] == 2


# ── examples ─────────────────────────────────────────────────────────────────
def _write(tmp_path, *rows):
    (tmp_path / "annotations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_examples_join_edges(tmp_path):
    _write(tmp_path,
           {"kind": "label", "chooser": "human:a3", "run_id": "r1", "payload": {"dataset": "x"}},
           {"kind": "label", "chooser": "human:a1", "run_id": None,
            "payload": {"dataset": "scd_0047"}},
           {"kind": "rating", "chooser": "human:a3", "run_id": "r2", "payload": {}},
           {"kind": "correction", "chooser": "human:a3", "run_id": "r1", "decision_id": "d1",
            "step": 0, "should": "umap", "payload": {"did": "phate"}},
           {"kind": "correction", "chooser": "human:a3", "run_id": "r3", "decision_id": None,
            "step": 1, "should": "pca", "payload": {"did": "umap"}})
    ex = annotations.examples(tmp_path)
    assert [(e["join"], e["key"]) for e in ex] == [
        ("run_id", "r1"), ("dataset", "scd_0047"), ("run_id", "r2"),
        ("decision_id", "d1"), ("run_id", "r3")]
    for e in ex:
        assert set(e) == {"kind", "chooser", "join", "key", "payload", "step", "should"}
    assert (ex[3]["step"], ex[3]["should"]) == (0, "umap")


def test_examples_leave_out_rows_with_no_key(tmp_path):
    _write(tmp_path,
           {"kind": "rating", "chooser": "human:a3", "run_id": None, "payload": {}},
           {"kind": "label", "chooser": "human:a3", "run_id": None, "payload": {}},
           {"kind": "correction", "chooser": "human:a3", "run_id": None, "decision_id": None,
            "payload": {}},
           {"kind": "approval", "chooser": "human:a3", "run_id": "r1", "payload": {}},
           {"kind": "rating", "chooser": "human:a3", "run_id": "r9", "payload": {}})
    ex = annotations.examples(tmp_path)
    assert [(e["join"], e["key"]) for e in ex] == [("run_id", "r9")]


def test_examples_docstring_names_the_step_and_should_readers():
    doc = annotations.examples.__doc__
    assert "Session.branch(index)" in doc
    assert "session.resolve_step" in doc


# ── benchmark_topology ───────────────────────────────────────────────────────
def test_benchmark_topology_in_vocab_order():
    assert annotations.benchmark_topology(["multi_branch", "clusters", "clusters"]) == [
        "clusters", "multi-branching"]


def test_benchmark_topology_ignores_unknown_labels():
    assert annotations.benchmark_topology(["blob", "clusters"]) == ["clusters"]
    assert annotations.benchmark_topology(["Clusters", "simple-traj"]) == []


@pytest.mark.parametrize("raw", [["blob"], []])
def test_benchmark_topology_no_mappable_class(raw):
    assert annotations.benchmark_topology(raw) == []


def test_benchmark_topology_defines_no_map_of_its_own():
    """The one map is `vocab.SCSHAPEBENCH_TOPOLOGY`; this module holds no second account."""
    import inspect

    src = inspect.getsource(annotations)
    for raw in ("simple_traj", "multi_branch"):
        assert f'"{raw}"' not in src and f"'{raw}'" not in src
