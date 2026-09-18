"""The batch and TUI writers share a cwd ledger, separate from project artifacts."""
from argparse import Namespace
from pathlib import Path

from manyruns import app, decisions, store, tune
from manyruns.narrate import Observation
from manyruns.tui.app import ManyrunsApp, _last_tune_row
from manyruns.tui.state import DataEntry


def test_batch_and_tui_share_one_working_directory_ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert app.main(["run", "swissroll", "--engine", "mock", "--project", "batch"]) == 0
    ui = ManyrunsApp(args=Namespace(engine="mock", seed=42))
    entry = DataEntry(name="swissroll", kind="bundled", dataset="swissroll",
                      obs=Observation(shape="manifold", modality="synthetic", source="swissroll"))
    result = ui.start_for(entry, "embed")(None)
    rows = list(store.read())
    assert len(rows) == 2 and rows[-1]["run_id"] == result["run_id"]
    assert store.index_path().resolve() == tmp_path / "outputs/index.jsonl"
    for project in ("batch", "swissroll"):
        folder = tmp_path / "outputs" / project
        assert (folder / "project.yaml").is_file()
        assert not (folder / "index.jsonl").exists()
    assert (tmp_path / "outputs/batch/summary.md").is_file()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert list(store.read()) == []
    assert list(store.read(tmp_path / "outputs")) == rows


def test_decision_readers_match_menu_and_tuning_scopes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    offered = [{"name": "a"}, {"name": "b"}]
    decisions.append(offered=offered, chosen="a", surface="ledger")
    corpus = tune._corpus_dir(Namespace(out_dir=Path("outputs/project")))
    decisions.append(offered=offered, chosen="b", surface="tune", out_dir=corpus)
    assert [row["surface"] for row in decisions.read()] == ["ledger"]
    assert [row["surface"] for row in decisions.read(corpus)] == ["tune"]
    assert _last_tune_row(corpus)["chosen"] == "b"
    assert tune._corpus_dir(Namespace(out_dir=Path("."))) == "outputs"
