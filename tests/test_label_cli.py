"""`manyruns label`, `import-labels`, `rate` and `correct`: the verbs that write annotations."""
from __future__ import annotations

import hashlib
import json

import pytest

from manyruns import annotations, app, commands, store


def _seed(*runs):
    """Run rows in `outputs/index.jsonl`; each item is `(results, extra)`."""
    for results, extra in runs:
        store.append(results, out_dir="outputs", extra=extra)


def _run(run_id, *, steps=("phate",), dataset_name="pbmc3k", extra=None):
    return ({"run_id": run_id, "dataset_name": dataset_name,
             "steps": [{"name": s, "group": "latent"} for s in steps]}, extra)


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(_run("abc123def456"), _run("r1", extra={"decision_id": "d1"}), _run("r2"), _run("r3"))
    return tmp_path / "outputs"


def _rows(outputs):
    return list(annotations.read(outputs))


def _exit2(argv):
    with pytest.raises(SystemExit) as info:
        app.main(argv)
    assert info.value.code == 2


# ── the verbs are verbs ──────────────────────────────────────────────────────
@pytest.mark.parametrize("verb", ["label", "import-labels", "rate", "correct"])
def test_the_verb_is_a_subcommand(verb):
    assert verb in app._SUBCOMMANDS


def test_the_verb_is_not_rewritten_to_run():
    argv = ["label", "r1", "--topology", "clusters", "--annotator", "a3"]
    assert app._normalize_argv(argv) == argv


def test_help_lists_the_verbs(capsys):
    with pytest.raises(SystemExit):
        app.main(["--help"])
    out = capsys.readouterr().out
    for verb in ("label", "import-labels", "rate", "correct"):
        assert verb in out


def test_each_verb_routes_to_its_command():
    parser = app._build_parser()
    cases = {
        ("label", "r1", "--topology", "clusters", "--annotator", "a3"): app.cmd_label,
        ("import-labels", "scshapebench", "--source", "f", "--revision", "abcdef0"):
            app.cmd_import_labels,
        ("rate", "r1", "--blind", "--accuracy", "4", "--completeness", "3", "--annotator", "a3"):
            app.cmd_rate,
        ("correct", "r1", "--step", "0", "--should", "umap", "--annotator", "a3"):
            app.cmd_correct,
    }
    for argv, func in cases.items():
        assert parser.parse_args(list(argv)).func is func


def test_the_prompt_command_list_is_unchanged():
    assert commands.resolve("rate") is None
    assert commands.resolve("correct") is None


# ── label ────────────────────────────────────────────────────────────────────
def test_two_topologies_on_a_stored_run(outputs, capsys):
    rc = app.main(["label", "abc123def456", "--topology", "multi-branching,clusters",
                   "--annotator", "a3"])
    assert rc == 0
    [row] = _rows(outputs)
    assert row["kind"] == "label" and row["chooser"] == "human:a3"
    assert row["run_id"] == "abc123def456"
    assert row["payload"] == {"dataset": "pbmc3k", "topology": ["clusters", "multi-branching"],
                              "raw": []}
    assert capsys.readouterr().out.strip() == row["annotation_id"]


def test_label_keeps_raw_labels_in_order_and_falls_back_to_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(({"run_id": "r1", "dataset": "eb"}, None))
    assert app.main(["label", "r1", "--topology", "clusters,clusters", "--annotator", "a3",
                     "--raw-label", "blob", "--raw-label", "clusters"]) == 0
    [row] = _rows(tmp_path / "outputs")
    assert row["payload"] == {"dataset": "eb", "topology": ["clusters"],
                              "raw": ["blob", "clusters"]}


def test_label_dataset_is_null_when_the_run_names_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(({"run_id": "r1"}, None))
    assert app.main(["label", "r1", "--topology", "cycle", "--annotator", "a3"]) == 0
    assert _rows(tmp_path / "outputs")[0]["payload"]["dataset"] is None


def test_label_uses_the_last_row_of_a_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(({"run_id": "r1", "dataset_name": "old"}, None),
          ({"run_id": "r1", "dataset_name": "new"}, None))
    assert app.main(["label", "r1", "--topology", "cycle", "--annotator", "a3"]) == 0
    assert _rows(tmp_path / "outputs")[0]["payload"]["dataset"] == "new"


def test_the_benchmarks_singular_spelling_is_refused(outputs, capsys):
    _exit2(["label", "abc123def456", "--topology", "cluster", "--annotator", "a3"])
    assert "cluster" in capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


@pytest.mark.parametrize("annotator", ["Jane", "jane doe", "_a3", "a" * 33, ""])
def test_label_refuses_a_bad_annotator(outputs, annotator):
    _exit2(["label", "abc123def456", "--topology", "clusters", "--annotator", annotator])
    assert not (outputs / "annotations.jsonl").exists()


def test_label_refuses_an_empty_topology(outputs):
    _exit2(["label", "abc123def456", "--topology", ",", "--annotator", "a3"])
    assert not (outputs / "annotations.jsonl").exists()


def test_an_unknown_run_is_refused(outputs, capsys):
    rc = app.main(["label", "nosuchrun000", "--topology", "clusters", "--annotator", "a3"])
    assert rc == 1
    assert "nosuchrun000" in capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


def test_label_reports_a_failed_write(outputs, monkeypatch, capsys):
    monkeypatch.setattr(annotations, "append", lambda *a, **k: None)
    assert app.main(["label", "abc123def456", "--topology", "clusters",
                     "--annotator", "a3"]) == 1
    assert capsys.readouterr().err


# ── import-labels ────────────────────────────────────────────────────────────
def _source(tmp_path, *lines, name="labels.jsonl"):
    path = tmp_path / name
    path.write_text("".join((line if isinstance(line, str) else json.dumps(line)) + "\n"
                            for line in lines))
    return path


def _import(path, revision="abcdef0"):
    return app.main(["import-labels", "scshapebench", "--source", str(path),
                     "--revision", revision])


def test_two_annotators_on_one_dataset(outputs, tmp_path, capsys):
    src = _source(tmp_path,
                  {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]},
                  {"dataset": "d1", "annotator": "a2", "labels": ["multi_branch", "clusters"]})
    assert _import(src) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "2"
    assert "skipped" not in captured.err
    rows = _rows(outputs)
    assert len(rows) == 2
    second = rows[1]
    assert second["chooser"] == "human:a2" and second["run_id"] is None
    assert second["payload"]["topology"] == ["clusters", "multi-branching"]
    assert second["payload"]["raw"] == ["multi_branch", "clusters"]
    assert second["payload"]["dataset"] == "d1"
    assert second["payload"]["source"] == {
        "name": "scshapebench", "revision": "abcdef0",
        "sha256": hashlib.sha256(src.read_bytes()).hexdigest()}


def test_a_second_import_of_the_same_file_is_refused(outputs, tmp_path, capsys):
    src = _source(tmp_path,
                  {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]},
                  {"dataset": "d1", "annotator": "a2", "labels": ["clusters"]})
    assert _import(src) == 0
    capsys.readouterr()
    assert _import(src) == 1
    assert "already imported" in capsys.readouterr().err
    assert len((outputs / "annotations.jsonl").read_text().splitlines()) == 2


def test_rows_with_no_mappable_class_are_skipped_and_counted(outputs, tmp_path, capsys):
    src = _source(tmp_path,
                  {"dataset": "d1", "annotator": "a1", "labels": ["blob", "clusters"]},
                  {"dataset": "d1", "annotator": "a2", "labels": ["blob"]},
                  {"dataset": "d1", "annotator": "a3", "labels": []})
    assert _import(src) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "1"
    assert "skipped 2 rows with no mappable class" in captured.err
    [row] = _rows(outputs)
    assert row["payload"]["topology"] == ["clusters"]
    assert row["payload"]["raw"] == ["blob", "clusters"]


def test_the_source_path_does_not_reach_the_row(outputs, tmp_path):
    src = _source(tmp_path, {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]})
    assert _import(src) == 0
    for line in (outputs / "annotations.jsonl").read_text().splitlines():
        assert str(tmp_path) not in line
        assert "labels.jsonl" not in line


def test_import_refuses_another_benchmark(outputs, tmp_path):
    src = _source(tmp_path, {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]})
    _exit2(["import-labels", "otherbench", "--source", str(src), "--revision", "abcdef0"])
    assert not (outputs / "annotations.jsonl").exists()


@pytest.mark.parametrize("revision", ["abc", "ABCDEF0", "g" * 7, "a" * 41, "main"])
def test_import_refuses_a_bad_revision(outputs, tmp_path, revision):
    src = _source(tmp_path, {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]})
    with pytest.raises(SystemExit) as info:
        _import(src, revision=revision)
    assert info.value.code == 2
    assert not (outputs / "annotations.jsonl").exists()


def test_import_accepts_a_full_revision(outputs, tmp_path):
    src = _source(tmp_path, {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]})
    assert _import(src, revision="a" * 40) == 0


def test_import_refuses_a_missing_file(outputs, tmp_path, capsys):
    assert _import(tmp_path / "nope.jsonl") == 1
    assert capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


@pytest.mark.parametrize("bad", [
    "{not json",
    "[1, 2]",
    json.dumps({"annotator": "a2", "labels": ["clusters"]}),
    json.dumps({"dataset": "d2", "labels": ["clusters"]}),
    json.dumps({"dataset": "d2", "annotator": "a2"}),
    json.dumps({"dataset": 3, "annotator": "a2", "labels": ["clusters"]}),
    json.dumps({"dataset": "d2", "annotator": "a2", "labels": "clusters"}),
    json.dumps({"dataset": "d2", "annotator": "a2", "labels": ["clusters", 4]}),
])
def test_import_refuses_a_bad_line_and_writes_nothing(outputs, tmp_path, capsys, bad):
    src = _source(tmp_path, {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]}, bad)
    assert _import(src) == 1
    assert "line 2" in capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


def test_import_refuses_a_duplicate_pair(outputs, tmp_path, capsys):
    src = _source(tmp_path,
                  {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]},
                  {"dataset": "d1", "annotator": "a1", "labels": ["archetypal"]})
    assert _import(src) == 1
    assert capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


def test_import_refuses_a_bad_annotator(outputs, tmp_path, capsys):
    src = _source(tmp_path,
                  {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]},
                  {"dataset": "d1", "annotator": "Jane Doe", "labels": ["clusters"]})
    assert _import(src) == 1
    assert capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


def test_import_accepts_the_benchmark_pseudonym(outputs, tmp_path):
    src = _source(tmp_path,
                  {"dataset": "d1", "annotator": "scshapebench_a4", "labels": ["simple_traj"]})
    assert _import(src) == 0
    [row] = _rows(outputs)
    assert row["chooser"] == "human:scshapebench_a4"
    assert row["payload"]["topology"] == ["single-trajectory"]


def test_import_reports_a_failed_write(outputs, tmp_path, monkeypatch, capsys):
    src = _source(tmp_path, {"dataset": "d1", "annotator": "a1", "labels": ["clusters"]})
    monkeypatch.setattr(annotations, "append", lambda *a, **k: None)
    assert _import(src) == 1
    assert capsys.readouterr().err


# ── rate ─────────────────────────────────────────────────────────────────────
def test_three_runs_ranked(outputs):
    rc = app.main(["rate", "r1", "r2", "r3", "--rank", "2,1,3", "--novelty", "4,5,2",
                   "--impact", "3,3,1", "--annotator", "a3"])
    assert rc == 0
    rows = _rows(outputs)
    assert [r["run_id"] for r in rows] == ["r1", "r2", "r3"]
    assert all(r["kind"] == "rating" and r["chooser"] == "human:a3" for r in rows)
    r2 = rows[1]
    assert r2["payload"] == {"mode": "rank", "rank": 1, "compared": ["r1", "r2", "r3"],
                             "novelty": 5, "impact": 3}


def test_a_rank_without_scores_carries_nulls(outputs):
    assert app.main(["rate", "r1", "r2", "--rank", "1,2", "--annotator", "a3"]) == 0
    rows = _rows(outputs)
    assert rows[0]["payload"]["novelty"] is None and rows[0]["payload"]["impact"] is None


def test_rate_prints_each_annotation_id(outputs, capsys):
    assert app.main(["rate", "r1", "r2", "--rank", "1,2", "--annotator", "a3"]) == 0
    ids = capsys.readouterr().out.split()
    assert ids == [r["annotation_id"] for r in _rows(outputs)]


def test_a_blind_rating(outputs):
    assert app.main(["rate", "r1", "--blind", "--accuracy", "4", "--completeness", "3",
                     "--annotator", "a3"]) == 0
    [row] = _rows(outputs)
    assert row["payload"] == {"mode": "blind", "blind": True, "accuracy": 4, "completeness": 3}


def test_a_rubric_rating(outputs, tmp_path):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("clarity: 4\nrigor: 2\n")
    assert app.main(["rate", "r1", "--rubric", str(rubric), "--annotator", "a3"]) == 0
    [row] = _rows(outputs)
    assert row["payload"] == {"mode": "rubric", "scores": {"clarity": 4, "rigor": 2},
                              "rubric_sha256": hashlib.sha256(rubric.read_bytes()).hexdigest()}
    for line in (outputs / "annotations.jsonl").read_text().splitlines():
        assert str(rubric) not in line


@pytest.mark.parametrize("argv", [
    ["r1", "--annotator", "a3"],                                           # no mode
    ["r1", "r2", "--rank", "1,2", "--blind", "--accuracy", "4",
     "--completeness", "3", "--annotator", "a3"],                          # two modes
    ["r1", "--rank", "1", "--annotator", "a3"],                            # rank needs 2+
    ["r1", "r2", "--blind", "--accuracy", "4", "--completeness", "3",
     "--annotator", "a3"],                                                 # blind is one run
    ["r1", "r2", "--rubric", "x.yaml", "--annotator", "a3"],               # rubric is one run
    ["r1", "r1", "--rank", "1,2", "--annotator", "a3"],                    # same run twice
    ["r1", "r2", "--rank", "1,1", "--annotator", "a3"],                    # not a permutation
    ["r1", "r2", "--rank", "1,3", "--annotator", "a3"],                    # not 1..n
    ["r1", "r2", "--rank", "1,2,3", "--annotator", "a3"],                  # length differs
    ["r1", "r2", "--rank", "1,x", "--annotator", "a3"],                    # not an int
    ["r1", "r2", "--rank", "1,2", "--novelty", "4", "--annotator", "a3"],  # length differs
    ["r1", "r2", "--rank", "1,2", "--impact", "4,6", "--annotator", "a3"],  # out of range
    ["r1", "r2", "--rank", "1,2", "--novelty", "0,3", "--annotator", "a3"],  # out of range
    ["r1", "--novelty", "4", "--blind", "--accuracy", "4", "--completeness", "3",
     "--annotator", "a3"],                                                 # novelty w/o rank
    ["r1", "--impact", "4", "--annotator", "a3"],                          # impact w/o rank
    ["r1", "r2", "--rank", "1,2", "--accuracy", "4", "--annotator", "a3"],  # accuracy w/o blind
    ["r1", "--completeness", "3", "--rubric", "x.yaml",
     "--annotator", "a3"],                                                 # completeness w/o blind
    ["r1", "--blind", "--accuracy", "4", "--annotator", "a3"],             # blind needs both
    ["r1", "--blind", "--accuracy", "6", "--completeness", "3",
     "--annotator", "a3"],                                                 # score out of range
    ["r1", "--blind", "--accuracy", "4", "--completeness", "3",
     "--annotator", "A3"],                                                 # bad annotator
])
def test_rate_refuses_with_exit_2(outputs, argv):
    _exit2(["rate", *argv])
    assert not (outputs / "annotations.jsonl").exists()


def test_a_rank_that_is_not_a_permutation(outputs):
    _exit2(["rate", "r1", "r2", "--rank", "1,1", "--annotator", "a3"])
    assert not (outputs / "annotations.jsonl").exists()


def test_rate_refuses_an_unknown_run(outputs, capsys):
    assert app.main(["rate", "r1", "nosuchrun000", "--rank", "1,2", "--annotator", "a3"]) == 1
    assert "nosuchrun000" in capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


@pytest.mark.parametrize("content", ["clarity: 9\n", "clarity: 0\n", "clarity: high\n",
                                     "clarity: 4.5\n", "clarity: true\n", "- 1\n- 2\n", "{}\n",
                                     "4\n"])
def test_rate_refuses_a_bad_rubric(outputs, tmp_path, content):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text(content)
    assert app.main(["rate", "r1", "--rubric", str(rubric), "--annotator", "a3"]) == 1
    assert not (outputs / "annotations.jsonl").exists()


def test_a_rubric_score_out_of_range(outputs, tmp_path):
    rubric = tmp_path / "rubric.yaml"
    rubric.write_text("clarity: 9\n")
    assert app.main(["rate", "r1", "--rubric", str(rubric), "--annotator", "a3"]) == 1
    assert not (outputs / "annotations.jsonl").exists()


def test_rate_refuses_a_missing_rubric(outputs, tmp_path):
    assert app.main(["rate", "r1", "--rubric", str(tmp_path / "nope.yaml"),
                     "--annotator", "a3"]) == 1
    assert not (outputs / "annotations.jsonl").exists()


def test_rate_reports_a_failed_write(outputs, monkeypatch, capsys):
    monkeypatch.setattr(annotations, "append", lambda *a, **k: None)
    assert app.main(["rate", "r1", "--blind", "--accuracy", "4", "--completeness", "3",
                     "--annotator", "a3"]) == 1
    assert capsys.readouterr().err


# ── correct ──────────────────────────────────────────────────────────────────
def test_a_correction_on_a_ledger_run(outputs, capsys):
    rc = app.main(["correct", "r1", "--step", "0", "--should", "UMAP", "--annotator", "a3"])
    assert rc == 0
    [row] = _rows(outputs)
    assert row["kind"] == "correction" and row["chooser"] == "human:a3"
    assert (row["run_id"], row["step"], row["should"], row["decision_id"]) == (
        "r1", 0, "umap", "d1")
    assert row["payload"] == {"did": "phate"}
    assert capsys.readouterr().out.strip() == row["annotation_id"]


def test_a_correction_without_a_decision(outputs):
    assert app.main(["correct", "r2", "--step", "0", "--should", "pca",
                     "--annotator", "a3"]) == 0
    [row] = _rows(outputs)
    assert row["decision_id"] is None and row["should"] == "pca"


def test_an_unknown_step_name(outputs, capsys):
    _exit2(["correct", "r1", "--step", "0", "--should", "nosuchstep", "--annotator", "a3"])
    assert "nosuchstep" in capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


@pytest.mark.parametrize("step", ["-1", "x", "1.5"])
def test_correct_refuses_a_bad_step_index(outputs, step):
    _exit2(["correct", "r1", "--step", step, "--should", "umap", "--annotator", "a3"])
    assert not (outputs / "annotations.jsonl").exists()


def test_correct_refuses_a_bad_annotator(outputs):
    _exit2(["correct", "r1", "--step", "0", "--should", "umap", "--annotator", "Jane Doe"])
    assert not (outputs / "annotations.jsonl").exists()


def test_correct_refuses_an_unknown_run(outputs, capsys):
    assert app.main(["correct", "nosuchrun000", "--step", "0", "--should", "umap",
                     "--annotator", "a3"]) == 1
    assert "nosuchrun000" in capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


def test_a_step_past_the_end_of_the_run(outputs, capsys):
    assert app.main(["correct", "r1", "--step", "1", "--should", "umap",
                     "--annotator", "a3"]) == 1
    assert "1" in capsys.readouterr().err
    assert not (outputs / "annotations.jsonl").exists()


def test_correct_reports_a_failed_write(outputs, monkeypatch, capsys):
    monkeypatch.setattr(annotations, "append", lambda *a, **k: None)
    assert app.main(["correct", "r1", "--step", "0", "--should", "umap",
                     "--annotator", "a3"]) == 1
    assert capsys.readouterr().err


# ── the verbs leave the other records alone ──────────────────────────────────
def test_the_verbs_touch_neither_decisions_nor_runs(outputs):
    before = (outputs / "index.jsonl").read_text()
    assert app.main(["label", "r1", "--topology", "clusters", "--annotator", "a3"]) == 0
    assert app.main(["rate", "r1", "r2", "--rank", "1,2", "--annotator", "a3"]) == 0
    assert app.main(["correct", "r1", "--step", "0", "--should", "umap",
                     "--annotator", "a3"]) == 0
    assert (outputs / "index.jsonl").read_text() == before
    assert not (outputs / "decisions.jsonl").exists()
