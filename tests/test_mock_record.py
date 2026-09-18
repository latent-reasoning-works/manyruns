"""A mock batch run is a recorded simulation, with no array artifacts."""
from datetime import datetime

import pytest

from manyruns import app, store


def test_mock_cli_records_identity_steps_and_invented_numbers(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    argv = ["run", "swissroll", "--engine", "mock", "--project", "audit-mock", "--seed", "17"]
    assert app.main(argv) == 0
    first, = store.read()
    output = capsys.readouterr().out
    assert first["run_id"] and f"run {first['run_id']} recorded" in output
    assert first["spec_id"] and datetime.fromisoformat(first["at"]).tzinfo is not None
    assert first["seed"] == 17 and first["engine"] == "mock"
    assert first["ok"] is True and first["complete"] is False
    assert first["dataset"] == "swissroll" and first["dataset_name"] == "swissroll"
    assert [step["outcome"] for step in first["steps"]] == ["skipped", "skipped", "ok", "ok"]
    assert all(step["seconds"] >= 0 for step in first["steps"])
    assert "0 successful, 0 benign skips" not in output
    assert any("invented" in caveat for caveat in first["caveats"])
    assert "invented" in (tmp_path / "outputs/audit-mock/summary.md").read_text()
    assert app.main(argv) == 0
    _, second = store.read()
    assert second["run_id"] != first["run_id"]
    assert second["spec_id"] == first["spec_id"]
    assert second["g_vector"]["final_dim"] == first["g_vector"]["final_dim"]
    assert tmp_path.is_dir()
    assert not list(tmp_path.rglob("*.npy"))
    assert not list(tmp_path.rglob("*.npz"))
    assert not (tmp_path / "outputs/audit-mock/state").exists()


def test_mock_cli_reports_prep_as_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert app.main(["run", "swissroll", "--engine", "mock", "--project", "prep",
                     "--recipe", "cflows"]) == 0
    row, = store.read()
    assert row["ok"] is True and row["complete"] is False
    assert [(s["name"], s["outcome"]) for s in row["steps"]] == [
        ("normalize", "skipped"), ("transform", "skipped"),
        ("pca", "ok"), ("phate", "ok"), ("mioflow", "ok")]
    assert all(s["detail"] for s in row["steps"] if s["outcome"] == "skipped")


@pytest.mark.parametrize("fails", [False, True])
def test_mock_cli_status_matches_execution(tmp_path, monkeypatch, fails):
    from manyruns import serving

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "load_recipe", lambda *a, **k: {
        "name": "mock-step", "steps": [{"name": "phate", "group": "latent", "params": {}}]})
    if fails:
        def failed_op(*args):
            raise ValueError("mock operation failed")
        monkeypatch.setattr(serving, "_apply_op", failed_op)
    rc = app.main(["run", "swissroll", "--engine", "mock", "--project", "one",
                   "--recipe", "mock-step"])
    row, = store.read()
    assert rc == (1 if fails else 0)
    assert row["ok"] is (not fails) and row["complete"] is (not fails)
    assert row["steps"][0]["outcome"] == ("error" if fails else "ok")
