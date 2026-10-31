"""The commercial flag: `MANYRUNS_COMMERCIAL=1` refuses every non-commercial step.

Someone who builds a proprietary application on the free components sets the flag. Then every
recipe with a non-commercial step is a hollow row with one reason, and manyruns imports none of
the Yale-derived manylatents modules. Without the flag everything runs.
"""
from __future__ import annotations

import argparse
import ast
import importlib.resources
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from manyruns import app, catalog, commercial, narrate, toolchain
from manyruns.pipeline import runner

PACKAGE = Path(__file__).resolve().parent.parent / "manyruns"


# ── the flag module ──────────────────────────────────────────────────────────

def test_three_values(monkeypatch):
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    seen = [commercial.enabled()]
    for value in ("", "0", "1"):
        monkeypatch.setenv("MANYRUNS_COMMERCIAL", value)
        seen.append(commercial.enabled())
    assert seen == [False, False, False, True]


def test_typo_does_not_turn_flag_off(monkeypatch):
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "true")
    with pytest.raises(ValueError) as raised:
        commercial.enabled()
    message = str(raised.value)
    for part in ("MANYRUNS_COMMERCIAL", "true", "1", "0"):
        assert part in message


def _commercial_reads(tree: ast.AST) -> list[int]:
    """Line numbers of every `env.get("COMMERCIAL")` / `env.is_set("COMMERCIAL")` call."""
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        first = node.args[0]
        if (name in ("get", "is_set") and isinstance(first, ast.Constant)
                and first.value == "COMMERCIAL"):
            lines.append(node.lineno)
    return lines


def test_one_reader():
    assert PACKAGE.is_dir(), "the manyruns package folder moved"
    readers, spellers = set(), set()
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(PACKAGE).as_posix()
        if _commercial_reads(ast.parse(text)):
            readers.add(rel)
        if "MANYRUNS_COMMERCIAL" in text:
            spellers.add(rel)
    assert readers == {"commercial.py"}
    assert spellers == {"commercial.py"}


def test_module_constants():
    assert commercial.NON_COMMERCIAL_STEPS == (
        "aanet_reconstruction", "cflows", "granger", "mioflow", "mioflow_gaga")
    assert commercial.NON_COMMERCIAL_MODULES == tuple(sorted(commercial.NON_COMMERCIAL_MODULES))
    assert len(commercial.NON_COMMERCIAL_MODULES) == 7
    assert "MANYRUNS_COMMERCIAL=1" in commercial.REASON


def _targets(node) -> list[str]:
    """Every `_target_` value of a parsed config, at any depth."""
    if isinstance(node, dict):
        found = [node["_target_"]] if isinstance(node.get("_target_"), str) else []
        for value in node.values():
            found.extend(_targets(value))
        return found
    if isinstance(node, list):
        return [t for item in node for t in _targets(item)]
    return []


def test_steps_cover_yale_lightning_configs():
    if importlib.util.find_spec("manylatents") is None:
        pytest.skip("manylatents is not installed")
    folder = importlib.resources.files("manylatents") / "configs" / "algorithms" / "lightning"
    assert folder.is_dir(), "manylatents moved its lightning configs"
    yale = commercial.NON_COMMERCIAL_MODULES
    named = set()
    for entry in folder.iterdir():
        if not entry.name.endswith(".yaml"):
            continue
        doc = yaml.safe_load(entry.read_text(encoding="utf-8"))
        if any(t == m or t.startswith(m + ".") for t in _targets(doc) for m in yale):
            named.add(entry.name[:-len(".yaml")])
    assert named, "no lightning config names a Yale-derived module; the scan reads nothing"
    assert named <= set(commercial.NON_COMMERCIAL_STEPS), sorted(
        named - set(commercial.NON_COMMERCIAL_STEPS))


def test_is_non_commercial_by_name_and_refusal(monkeypatch):
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    recipe = {"name": "r", "steps": [{"name": "phate", "group": "latent"},
                                     {"name": "mioflow", "group": "lightning"}]}
    assert commercial.is_non_commercial({"name": "mioflow", "group": "lightning"})
    assert not commercial.is_non_commercial({"name": "phate", "group": "latent"})
    assert commercial.refusal(recipe) == []
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "1")
    assert commercial.refusal(recipe) == [commercial.REASON]
    assert commercial.refusal({"name": "e", "steps": [{"name": "phate"}]}) == []


# ── refusal by legality ─────────────────────────────────────────────────────────

STUB = Path(__file__).resolve().parent / "fixtures" / "mioflow_tool"
MIOFLOW_RECIPES = ("cflows", "sandbox", "traced")
#: The CLI in a fresh interpreter. The package has no `__main__`, so not `python -m manyruns`.
CLI = (sys.executable, "-c", "import sys; from manyruns.app import main; sys.exit(main(sys.argv[1:]))")
TOOLED = {"name": "tooled", "steps": [
    {"name": "pca", "group": "latent"},
    {"name": "mioflow", "group": "lightning", "tool": "mioflow"},
    {"name": "sample_trajectories", "group": "probe"}]}


def _cli(argv, capsys):
    try:
        code = app.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    out, err = capsys.readouterr()
    return code, out, err


def _actions(capsys) -> dict:
    code, out, _ = _cli(["actions", "synthetic_timecourse"], capsys)
    assert code == 0
    return json.loads(out)


def _subprocess_env(tmp_path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MANYRUNS_")}
    env.update({"HOME": str(tmp_path), "OMP_NUM_THREADS": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    env.update(extra)
    return env


def test_mioflow_recipes_are_hollow_rows(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    plain = _actions(capsys)
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "1")
    refused = _actions(capsys)
    plain_rows = {o["recipe"]: o for o in plain["offered"]}
    refused_rows = {o["recipe"]: o for o in refused["offered"]}
    assert set(plain_rows) == set(refused_rows)
    # `sandbox` (`suits: []`) is never offered, so it has no row; its refusal is checked on
    # the recipe itself.
    assert "sandbox" not in refused_rows
    assert narrate.unavailable(catalog.load_recipe("sandbox")) == [commercial.REASON]
    for name in set(MIOFLOW_RECIPES) - {"sandbox"}:
        assert refused_rows[name]["can_run"] is False
        assert commercial.REASON in refused_rows[name]["blocked"]
        assert refused["detail"][name]["unavailable"] == [commercial.REASON]
    for name in set(plain_rows) - set(MIOFLOW_RECIPES):
        assert refused_rows[name] == plain_rows[name], name


def test_nothing_changes_without_flag(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    unset = _actions(capsys)
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "0")
    assert _actions(capsys) == unset


def test_run_refuses_before_any_step(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(
        [*CLI, "run", "synthetic_timecourse", "--recipe", "cflows",
         "--engine", "mock", "--project", "commercial-check"],
        cwd=work, capture_output=True, text=True, timeout=300,
        env=_subprocess_env(tmp_path, MANYRUNS_COMMERCIAL="1"))
    assert result.returncode == 2, result.stdout + result.stderr
    assert "MANYRUNS_COMMERCIAL" in result.stderr
    assert not (work / "outputs").exists()


def _mock_ctx(tmp_path) -> dict:
    return {"out_dir": tmp_path, "plots": [], "seed": 0, "fast_dev_run": True,
            "target_dim": 3, "caveats": []}


def test_added_step_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "1")
    state = runner._new_state(X=np.arange(36, dtype=float).reshape(12, 3))
    rec = runner.apply_step({"name": "mioflow", "group": "lightning"}, state, {},
                            dispatch=runner.dispatch_for("mock"), ctx=_mock_ctx(tmp_path),
                            index=0, carry=runner.new_carry())
    assert rec["outcome"] == "skipped"
    assert rec.get("unsupported") is True
    assert rec["detail"] == commercial.REASON


def _tools_list(tmp_path, commercial_flag: bool) -> str:
    extra = {"MANYRUNS_TOOL_DIR": str(STUB)}
    if commercial_flag:
        extra["MANYRUNS_COMMERCIAL"] = "1"
    result = subprocess.run([*CLI, "tools", "list"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=120,
                            env=_subprocess_env(tmp_path, **extra))
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.split()[:1] == ["mioflow"]]
    assert len(lines) == 1, result.stdout
    return lines[0]


def test_non_commercial_manifest_is_refused(tmp_path, monkeypatch):
    assert STUB.is_dir(), "the stub MIOFlow folder moved"
    assert "no-manifest" in _tools_list(tmp_path, commercial_flag=True)
    assert "no-manifest" not in _tools_list(tmp_path, commercial_flag=False)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(STUB))
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "1")
    with pytest.raises(ValueError, match="commercial_use is false"):
        toolchain.load_tool("mioflow")
    assert commercial.TOOL_PROBLEM in toolchain.check_tool(
        yaml.safe_load((STUB / "mioflow.yaml").read_text()), "mioflow")


def test_tooled_non_commercial_step_is_refused(monkeypatch):
    assert STUB.is_dir(), "the stub MIOFlow folder moved"
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(STUB))
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    assert narrate.unavailable(TOOLED) == []
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "1")
    assert narrate.unavailable(TOOLED) == [commercial.REASON]


def test_weights_inference_is_refused(tmp_path):
    script = (
        "import sys, numpy as np\n"
        "from manyruns import commercial, registry\n"
        "v = registry.Version(model='m', version='1', kind='weights',"
        " spec={'weights': 'absent.pt'}, created='now')\n"
        "try:\n"
        "    registry.apply_weights(v, np.arange(36, dtype=float).reshape(12, 3))\n"
        "except ValueError as e:\n"
        "    assert str(e) == commercial.REASON, e\n"
        "else:\n"
        "    raise SystemExit('not refused')\n"
        "print(sorted(set(commercial.NON_COMMERCIAL_MODULES) & set(sys.modules)))\n")
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True,
                            text=True, timeout=120,
                            env=_subprocess_env(tmp_path, MANYRUNS_COMMERCIAL="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_ladder_stays_routable(monkeypatch, capsys):
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    monkeypatch.setenv("MANYRUNS_COMMERCIAL", "1")
    rc = app.cmd_check(SimpleNamespace(dataset_dir=None), argparse.ArgumentParser())
    out = capsys.readouterr().out
    assert "  ok    unroutable rungs: none" in out
    assert rc == 0


# ── no Yale-derived import in commercial mode ───────────────────────────────────

#: A fresh interpreter's prelude: every manyruns module, then the session below.
_WALK = """
import json, pkgutil, importlib, subprocess, sys
import numpy as np
import manyruns
from manyruns import commercial
for info in pkgutil.walk_packages(manyruns.__path__, "manyruns."):
    importlib.import_module(info.name)
"""

_REPORT = """
print("YALE=" + json.dumps(sorted(set(commercial.NON_COMMERCIAL_MODULES) & set(sys.modules))))
"""


def _fresh(tmp_path, body: str, **env) -> tuple[list, str]:
    """Run `body` in a fresh interpreter after the walk; return the Yale names it loaded."""
    result = subprocess.run([sys.executable, "-c", _WALK + body + _REPORT], cwd=tmp_path,
                            capture_output=True, text=True, timeout=1800,
                            env=_subprocess_env(tmp_path, **env))
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("YALE=")][-1]
    return json.loads(line[len("YALE="):]), result.stdout


def test_full_session_leaves_modules_out(tmp_path):
    body = f"""
from pathlib import Path
from manyruns import actions
from manyruns.pipeline import runner
for surface, kw in (("ledger", {{"dataset": "synthetic_timecourse"}}),
                    ("tune", {{"dataset": "synthetic_timecourse", "recipe": "cflows"}}),
                    ("ladder", {{}})):
    actions.enumerate_actions(surface, store=Path("store"), **kw)
from manyruns import catalog
X12 = np.arange(36, dtype=float).reshape(12, 3)
res = runner.run_inproc(X12, catalog.load_recipe("embed"), Path("embed-out"), seed=0)
assert [s["outcome"] for s in res["steps"] if s["name"] == "phate"] == ["ok"]
cli = {CLI!r}
child = subprocess.run([*cli, "run", "synthetic_timecourse", "--recipe", "cflows",
                        "--engine", "mock", "--project", "refused"], capture_output=True, text=True)
assert child.returncode == 2, child.stderr
"""
    loaded, _ = _fresh(tmp_path, body, MANYRUNS_COMMERCIAL="1")
    assert loaded == []


def _stack():
    if importlib.util.find_spec("manylatents") is None:
        pytest.skip("manylatents is not installed")


def _run_and_report(recipe: str) -> str:
    return f"""
from manyruns import app, store
code = app.main(["run", "synthetic_timecourse", "--recipe", {recipe!r}, "--engine",
                 "manylatents", "--project", "stack-check"])
rows = [r for r in store.read("outputs") if r.get("recipe") == {recipe!r}]
print("CODE=" + str(code))
print("OK=" + json.dumps(bool(rows) and bool(rows[-1].get("ok"))))
"""


@pytest.mark.stack_integration
def test_real_engine_leaves_modules_out(tmp_path):
    _stack()
    loaded, out = _fresh(tmp_path, _run_and_report("pseudotime"), MANYRUNS_COMMERCIAL="1")
    assert "OK=true" in out, out[-2000:]
    assert loaded == []


@pytest.mark.stack_integration
def test_route_imports_port_without_flag(tmp_path):
    _stack()
    loaded, out = _fresh(tmp_path, _run_and_report("traced"))
    assert "OK=true" in out, out[-2000:]
    assert "manylatents.algorithms.lightning.mioflow" in loaded


# ── the run row ──────────────────────────────────────────────────────────────

def _mock_row(tmp_path, monkeypatch, capsys, dataset: str, recipe: str) -> dict:
    from manyruns import store

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_COMMERCIAL", raising=False)
    code, _, err = _cli(["run", dataset, "--recipe", recipe, "--engine", "mock",
                         "--project", "row-check"], capsys)
    assert code == 0, err
    rows = [r for r in store.read("outputs") if r.get("recipe") == recipe]
    assert len(rows) == 1
    return rows[0]


def test_in_process_mioflow_step_is_named(tmp_path, monkeypatch, capsys):
    row = _mock_row(tmp_path, monkeypatch, capsys, "synthetic_timecourse", "cflows")
    assert row["non_commercial_steps"] == ["mioflow"]


def test_commercial_recipe_names_none(tmp_path, monkeypatch, capsys):
    row = _mock_row(tmp_path, monkeypatch, capsys, "swissroll", "embed")
    assert row["non_commercial_steps"] == []
