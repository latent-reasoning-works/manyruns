"""`manyruns tools` — the operator's view of the external-tool layer, and it must EXIST.

THIS FILE IS A REGRESSION TEST FOR A SHIPPED DEFECT. `toolchain.install_command` puts
`manyruns tools install <name>` into every refusal a missing tool produces — it is the entire
"how do I fix this" half of the message. The subcommand was added on a working branch and left
out of the merge, so on `main` for one commit the refusal named a command that did not exist.

Worse than not existing: it did something. `_normalize_argv` treats a bare word as a DATA
argument, so `manyruns tools` fell through to a run, wrote `outputs/tools/summary.md` and
appended a row to `index.jsonl` — a recorded run of a dataset nobody has. An unknown verb must
refuse, not improvise, and the only reason it improvised is that `tools` was not a verb.
"""

from __future__ import annotations

import pathlib
import tempfile
from pathlib import Path

import pytest
from tool_stubs import FRAME_ADAPTER, make_tool

from manyruns import app, toolchain

_REPO = pathlib.Path(__file__).resolve().parent.parent


def test_lock_compiles_the_manifest_requires_with_hashes(tmp_path, monkeypatch):
    make_tool(tmp_path, monkeypatch, "locky", {"input": {"kind": "state_matrix"}}, FRAME_ADAPTER)
    toolchain.lock_path(toolchain.load_tool("locky")).unlink()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    seen = {}

    def fake_run(argv, **kw):
        seen["argv"] = list(argv)
        seen["cwd"] = Path(kw["cwd"])
        assert seen["cwd"].is_dir()
        seen["in"] = (seen["cwd"] / argv[-1]).read_text()

        class R:
            returncode, stderr = 0, ""

        return R()

    monkeypatch.setattr(toolchain.subprocess, "run", fake_run)
    monkeypatch.setattr(toolchain, "uv", lambda: "/usr/bin/uv")
    out = toolchain.lock("locky")
    cfg = toolchain.load_tool("locky")
    argv = seen["argv"]
    assert argv[:3] == ["/usr/bin/uv", "pip", "compile"] and "--generate-hashes" in argv
    assert argv[argv.index("--python-version") + 1] == "3.12"
    assert argv[argv.index("--output-file") + 1] == str(toolchain.lock_path(cfg))
    assert "--custom-compile-command" in argv
    assert argv[argv.index("--custom-compile-command") + 1] == "manyruns tools lock locky"
    assert seen["in"] == "locky==1.0\n"
    assert argv[-1] == "requirements.in"
    assert not seen["cwd"].exists()
    assert out == {"lock": toolchain.lock_path(cfg), "returncode": 0, "stderr": ""}


def test_lock_without_uv_is_refused(tmp_path, monkeypatch):
    make_tool(tmp_path, monkeypatch, "locky", {"input": {"kind": "state_matrix"}}, FRAME_ADAPTER)
    monkeypatch.setattr(toolchain, "uv", lambda: None)
    with pytest.raises(ValueError, match="uv"):
        toolchain.lock("locky")


def test_the_no_lock_fix_names_an_action_that_exists(tmp_path, monkeypatch):
    make_tool(tmp_path, monkeypatch, "locky", {"input": {"kind": "state_matrix"}}, FRAME_ADAPTER)
    monkeypatch.delenv("MANYRUNS_TOOL_LOCKY_PYTHON")
    toolchain.lock_path(toolchain.load_tool("locky")).unlink()
    r = toolchain.resolve("locky")
    assert r["status"] == "no-lock" and r["fix"] == "manyruns tools lock locky"
    from manyruns import app

    assert "lock" in app._TOOLS_ACTIONS


def test_tools_is_a_declared_subcommand():
    """The registry `_normalize_argv` reads. Absent from it, `tools` is data, not a verb."""
    assert "tools" in app._SUBCOMMANDS


def test_the_parser_accepts_the_three_actions():
    parser = app._build_parser()
    for argv in (
        ["tools"],
        ["tools", "list"],
        ["tools", "check"],
        ["tools", "install", "pyrovelocity"],
    ):
        args = parser.parse_args(argv)
        assert args.func is app.cmd_tools, argv
    assert parser.parse_args(["tools"]).action == "list", "bare `tools` lists"


def test_every_command_a_refusal_names_is_a_command_that_exists():
    """THE POINT. A refusal that names a command nobody can run is a dead end wearing the
    costume of a next step — and this is generated text, so nothing else checks it."""
    cfg = toolchain.load_tool(toolchain.discover_tools()[0])
    fix = toolchain.install_command(cfg)
    assert fix.startswith("manyruns tools install ")

    verb, action = fix.split()[1], fix.split()[2]
    assert verb in app._SUBCOMMANDS
    args = app._build_parser().parse_args([verb, action, cfg["name"]])
    assert args.func is app.cmd_tools and args.name == cfg["name"]


def test_listing_tools_runs_nothing_and_records_nothing(capsys, tmp_path, monkeypatch):
    """`manyruns tools` is a READ. The defect this file exists for was it starting a run."""
    monkeypatch.chdir(tmp_path)
    assert app.main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "summary written" not in out and "recorded in" not in out
    assert not (tmp_path / "outputs").exists(), "listing tools wrote a run to disk"


def test_no_tool_scratch_file_is_tracked_in_the_repo():
    """A tool writes into whatever directory it was launched from, and this is a git repo.

    `pyrovelocity`'s trainer opens an mlflow tracking store on import and drops `mlflow.db` and
    `mlruns/` in the cwd, so every `manyruns run` of a tool recipe leaves them in the root.
    One was committed by a `git add -A` (#86) and only surfaced because it then blocked a
    branch switch. They are a TOOL's scratch state, not this project's source: nothing here
    reads them, and a binary that changes on every run is the worst possible thing to track.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=str(_REPO)
    ).stdout.split()
    strays = [
        f for f in tracked if f == "mlflow.db" or f.startswith("mlruns/") or f.endswith(".db")
    ]
    assert not strays, f"tool scratch state is tracked: {strays}"


def test_parser_accepts_verify():
    args = app._build_parser().parse_args(["tools", "verify", "golden"])
    assert args.func is app.cmd_tools
    assert args.action == "verify" and args.name == "golden"


def test_list_shows_precomputed(capsys, tmp_path, monkeypatch):
    from model_seam_fixtures import GOLDEN

    assert GOLDEN.is_dir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(GOLDEN))
    assert app.main(["tools", "list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert any(
        "golden_precomputed" in line and "precomputed" in line.split()[-1] for line in lines
    ), lines


def test_check_runs_no_selftest_for_precomputed(capsys, tmp_path, monkeypatch):
    from model_seam_fixtures import GOLDEN

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(GOLDEN))
    monkeypatch.setattr(
        toolchain,
        "selftest",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a selftest ran")),
    )
    app.main(["tools", "check"])
    line = next(x for x in capsys.readouterr().out.splitlines() if "golden_precomputed" in x)
    assert "selftest" not in line


def test_install_refuses_precomputed(capsys, tmp_path, monkeypatch):
    from model_seam_fixtures import GOLDEN

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(GOLDEN))
    assert app.main(["tools", "install", "golden_precomputed"]) == 2
    out = capsys.readouterr().out
    assert "precomputed" in out and "nothing to build" in out
