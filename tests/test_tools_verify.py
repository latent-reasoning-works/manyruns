"""`manyruns tools verify <name>` runs a tool on a built-in fixture of 12 cells and checks the
result with `check_result`. The owner of a tool runs it after every pin change."""
from __future__ import annotations

import shutil

import yaml

from manyruns import app
from model_seam_fixtures import GOLDEN, override_pythons


def _verify(name, capsys):
    code = app.main(["tools", "verify", name])
    return code, capsys.readouterr().out


def test_golden_tools_verify(capsys, tmp_path, monkeypatch):
    assert GOLDEN.is_dir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(GOLDEN))
    override_pythons(monkeypatch)
    for name in ("golden", "golden_call_svd"):
        code, out = _verify(name, capsys)
        assert code == 0, out
    assert not (tmp_path / "outputs" / "index.jsonl").exists()


def test_broken_tool_fails_verification(capsys, tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    cfg = yaml.safe_load((GOLDEN / "golden.yaml").read_text())
    cfg["params"] = {"truncate": 5}
    (tools / "golden.yaml").write_text(yaml.safe_dump(cfg))
    shutil.copy(GOLDEN / "golden_adapter.py", tools / "golden_adapter.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(tools))
    override_pythons(monkeypatch)
    code, out = _verify("golden", capsys)
    assert code == 1, out
    assert "clusters" in out


def test_declining_tool_fails_verification(capsys, tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    cfg = yaml.safe_load((GOLDEN / "golden.yaml").read_text())
    cfg["params"] = {"decline": "no model here"}
    (tools / "golden.yaml").write_text(yaml.safe_dump(cfg))
    shutil.copy(GOLDEN / "golden_adapter.py", tools / "golden_adapter.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(tools))
    override_pythons(monkeypatch)
    code, out = _verify("golden", capsys)
    assert code == 1 and "no model here" in out, out


def test_precomputed_and_unknown_tools_exit_2(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(GOLDEN))
    assert _verify("golden_precomputed", capsys)[0] == 2
    assert _verify("no_such_tool", capsys)[0] == 2


def test_unbuilt_tool_exits_4_with_the_fix(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(GOLDEN))
    monkeypatch.delenv("MANYRUNS_TOOL_GOLDEN_PYTHON", raising=False)
    code, out = _verify("golden", capsys)
    assert code == 4, out
    assert "manyruns tools lock golden" in out


def test_pyrovelocity_cannot_use_fixture(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_TOOL_DIR", raising=False)
    code, out = _verify("pyrovelocity", capsys)
    assert code == 3, out
    assert "state_frame" in out
