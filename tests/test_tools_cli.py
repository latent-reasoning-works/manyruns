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

from manyruns import app, toolchain

_REPO = pathlib.Path(__file__).resolve().parent.parent


def test_tools_is_a_declared_subcommand():
    """The registry `_normalize_argv` reads. Absent from it, `tools` is data, not a verb."""
    assert "tools" in app._SUBCOMMANDS


def test_the_parser_accepts_the_three_actions():
    parser = app._build_parser()
    for argv in (["tools"], ["tools", "list"], ["tools", "check"],
                 ["tools", "install", "pyrovelocity"]):
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

    tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                             cwd=str(_REPO)).stdout.split()
    strays = [f for f in tracked
              if f == "mlflow.db" or f.startswith("mlruns/") or f.endswith(".db")]
    assert not strays, f"tool scratch state is tracked: {strays}"
