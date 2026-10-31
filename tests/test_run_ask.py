"""`manyruns run --ask TEXT` — a free-typed request, routed by the rule tier, never a model.

The refusal is the case that matters: a trajectory asked of cluster data must print the reframe
and stop, with no fetch, no project file and no run row. A routed request runs the recipe it
routed to, exactly as `--recipe` would.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from manyruns import app


@pytest.fixture
def cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "drop"))
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda *a, **k: pytest.fail("--ask must not fetch on a refusal"))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("must not prompt"))
    ran: list[dict] = []
    monkeypatch.setattr(app, "_explore_project", lambda proj, **kw: ran.append(proj) or 0)
    return tmp_path, ran


def test_run_ask_refuses_trajectory_on_pbmc3k_without_running(cli, capsys):
    tmp_path, ran = cli
    rc = app.main(["run", "pbmc3k", "--engine", "mock", "--project", "asked",
                   "--ask", "show me the developmental trajectory"])
    assert rc == 0
    assert ran == []
    out = capsys.readouterr().out
    assert "no clock" in out and "what cell types are here" in out
    assert not (tmp_path / "outputs").exists()     # no project.yaml, no index.jsonl row


def test_run_ask_routes_structure_to_embed(cli):
    tmp_path, ran = cli
    (tmp_path / "drop").mkdir()
    ad.AnnData(np.ones((5, 3), dtype=np.float32)).write_h5ad(tmp_path / "drop" / "pbmc3k_raw.h5ad")
    rc = app.main(["run", "pbmc3k", "--engine", "mock", "--project", "asked",
                   "--ask", "map the structure"])
    assert rc == 0
    assert [p["recipe"]["name"] for p in ran] == ["embed"]
    assert Path("outputs/asked/project.yaml").is_file()


def test_run_ask_with_recipe_is_a_usage_error(cli):
    _, ran = cli
    with pytest.raises(SystemExit) as exc:
        app.main(["run", "pbmc3k", "--engine", "mock", "--project", "asked",
                  "--recipe", "embed", "--ask", "map the structure"])
    assert exc.value.code == 2
    assert ran == []
