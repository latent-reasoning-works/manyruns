"""Offline contracts for the commands a first-time user is told to run."""
from pathlib import Path
from importlib import resources
import json
import re
import shlex
import signal
import stat
import subprocess
import sys

import pytest

pytestmark = pytest.mark.source_checkout


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_URL = "git+https://github.com/latent-reasoning-works/manyruns"
TOOL_INSTALL = f"uv tool install --python 3.12 --force {PUBLIC_URL}"


def read_surface(name):
    assert ROOT.is_dir()
    return (ROOT / name).read_text()


def test_readme_installs_the_public_distribution_without_org_access():
    readme = read_surface("README.md")
    install = readme.split("## install\n", 1)[1].split("\n## ", 1)[0]
    assert "uv tool install --python 3.12 manyruns" in install
    assert "https://docs.astral.sh/uv/getting-started/installation/" in install
    assert not re.search(r"\bpip(?:3)?\b", install, re.I)
    assert TOOL_INSTALL in readme
    assert "repository is private" not in readme.lower()
    assert "gh auth" not in readme
    assert "open beta" in readme.lower()
    assert "https://github.com/latent-reasoning-works/manyruns/issues" in readme
    assert "CLI and record schema can still change" in readme


def test_install_opens_with_the_pypi_quickstart():
    install = read_surface("README.md").split("## install\n", 1)[1].split("\n## ", 1)[0]
    intro, block = install.strip().split("```sh\n", 1)
    assert "once published" in intro.lower()
    assert "https://docs.astral.sh/uv/getting-started/installation/" in intro
    commands, following = block.split("```", 1)
    assert commands.splitlines() == ["uv tool install --python 3.12 manyruns", "co-science"]
    assert "`uv tool upgrade manyruns` updates an installed tool in place (keeping its extras)" in following
    assert ("`sh packaging/install.sh` reinstalls the base tool from scratch, "
            "which drops any extras you added") in following


def test_docs_explain_supported_platforms_bootstrap_and_checkout_resolution():
    readme = read_surface("README.md")
    assert "https://astral.sh/uv/install.sh" in readme
    assert "powershell" not in readme.lower()
    assert "macOS and Linux, Python 3.11–3.12" in readme
    assert "Windows is unsupported" in readme
    assert "scikit-misc" in readme
    assert "new terminal" in readme.lower()
    assert "uv tool update-shell" in readme
    assert "isolated" in readme
    assert "uv sync --extra harness --extra dev" in readme
    for name in ("README.md", "CLAUDE.md"):
        source = read_surface(name)
        assert TOOL_INSTALL in source
        assert "gh auth" not in source
        assert "direct_url.json" in source


def test_distribution_sync_advice_requires_every_existing_extra():
    distribution = read_surface("CLAUDE.md").split(
        '## Distribution\n', 1)[1].split("\n## ", 1)[0]
    advice = " ".join(distribution.split())
    assert "exactly the extras named" in advice
    assert "name every extra" in advice
    assert "--extra agents" in advice
    assert "[Development](#development)" in advice
    assert "preserving all other already used extras" not in advice


def test_readme_explains_local_drop_precedence_and_verified_first_use():
    readme = read_surface("README.md")
    assert "$MANYRUNS_DATA_DIR" in readme
    assert "existing `./data`" in readme
    assert "~/.manyruns/data" in readme
    assert "ctrl+r" in readme.lower()
    assert "stable" in readme.lower()
    assert "first explicit selection" in readme
    assert "sha256" in readme
    assert "5.6 MB" in readme
    assert "https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad" in readme
    assert "co-science check" in readme
    assert "never downloads" in readme
    assert "no .h5ad files are bundled" in readme.lower()


def test_readme_distinguishes_check_validation_from_sample_presence(tmp_path, monkeypatch, capsys):
    import argparse

    from manyruns import app, catalog, shell

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "absent-data"))
    declarations = tmp_path / "datasets"
    declarations.mkdir()
    (declarations / "pbmc3k.yaml").write_text(
        read_surface("manyruns/configs/dataset/pbmc3k.yaml"))
    monkeypatch.setattr(catalog, "discover_recipes", lambda: [])

    def no_data_io(*args, **kwargs):
        pytest.fail("declaration checking must not resolve or download sample files")

    monkeypatch.setattr(shell, "dataset_ref_path", no_data_io)
    monkeypatch.setattr("urllib.request.urlopen", no_data_io)
    assert declarations.is_dir()
    result = app.cmd_check(argparse.Namespace(dataset_dir=declarations), argparse.ArgumentParser())
    assert result == 0
    assert "ok    pbmc3k" in capsys.readouterr().out

    readme = " ".join(read_surface("README.md").split())
    assert "validates declarations" in readme
    assert "never downloads" in readme
    assert "does not tell you whether a sample file is present" in readme
    assert "roster marks missing samples" in readme
    assert "reports absence" not in readme


def test_readme_uses_generated_metadata_for_the_colour_example():
    readme = read_surface("README.md")
    assert "co-science run synthetic_timecourse --recipe embed --color-by timepoint" in readme
    assert "--color-by cell_type --color-by score" in readme
    assert "raw pbmc3k has no obs metadata" in readme.lower()
    assert "F7/F8" in readme
    assert "full-screen" in readme
    assert "clamped" in readme


def test_readme_explains_repeated_attempts_and_actual_waiting_status():
    from manyruns.tui.app import ManyrunsApp
    from manyruns.tui.run import RunScreen

    readme = " ".join(read_surface("README.md").split())
    assert "**Repeated attempts.**" in readme
    assert "**Run again?**" in readme
    assert "same recipe selected" in readme
    assert "incomplete" in readme
    assert "in-flight step or measurement may finish" in readme
    assert "later steps and final metrics that have not started are skipped" in readme
    assert "same output directory" in readme
    assert ManyrunsApp.WAITING in readme
    assert RunScreen.FINALIZING in readme
    assert "elapsed time" in readme


def test_readme_documents_run_screen_colour_views_and_save():
    readme = " ".join(read_surface("README.md").split())
    assert "`c` on either the run screen or the full-screen viewer" in readme
    assert "finish this run before recolouring" in readme
    assert "`s` on either screen" in readme
    assert "SVG, PDF and PNG" in readme
    assert "`figures/`" in readme
    assert "no dataset source for this figure" not in readme


def make_recipes():
    recipes = {}
    target = None
    for line in read_surface("Makefile").splitlines():
        if line and not line.startswith(("\t", "#", ".")) and line.endswith(":"):
            target = line[:-1]
            recipes[target] = []
        elif line.startswith("\t") and target is not None:
            recipes[target].append(line.strip())
    assert recipes
    return recipes


def test_make_installs_committed_git_and_runs_a_real_demo():
    recipes = make_recipes()
    assert recipes["install"] == [
        'uv tool install --python 3.12 --force --reinstall-package manyruns "git+file://$(CURDIR)"'
    ]
    assert "committed HEAD" in " ".join(recipes["help"])
    assert "install-real" not in recipes
    assert "demo-real" not in recipes
    assert recipes["install-public"] == [TOOL_INSTALL]
    assert recipes["demo"] == [
        "co-science run swissroll --recipe embed --engine manylatents --project demo"
    ]
    assert recipes["test"] == ["uv run pytest tests/ -q"]
    assert recipes["lint"] == ["uv run ruff check manyruns tests"]


def test_runtime_repair_uses_the_tool_or_preserves_checkout_extras():
    from manyruns import installation

    assert installation.REINSTALL == TOOL_INSTALL
    advice = installation.RUNTIME_REPAIR
    assert TOOL_INSTALL in advice
    assert "uv sync --extra harness --extra dev" in advice
    assert "all already used extras" in advice
    assert "--extra agents" in advice
    assert "gh auth" not in advice
    assert "public" in advice
    assert "singlecell" in advice


# Every command visible to the installer is a fake, including its bootstrap effects.
# The absolute Python shebang and /bin/sh entry point do not depend on the replaced PATH.
FAKE_COMMAND = r'''
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")

def fail(message, code):
    print(message, file=sys.stderr)
    sys.exit(code)

if name == "uv":
    if args[:2] == ["tool", "install"]:
        if os.environ.get("FAKE_INSTALL_FAIL"):
            fail("fixture resolver failure", 7)
    elif args == ["tool", "dir", "--bin"]:
        if os.environ.get("FAKE_BIN_FAIL"):
            fail("fixture bin lookup failure", 9)
        print("" if os.environ.get("FAKE_BIN_EMPTY") else os.environ["FAKE_TOOL_BIN"])
    else:
        fail("unexpected uv arguments: " + repr(args), 99)
elif name == "co-science":
    assert args == ["--version"], args
    if os.environ.get("FAKE_VERSION_FAIL"):
        fail("fixture entry-point failure", 8)
    print("co-science fixture version")
elif name == "curl":
    assert "https://astral.sh/uv/install.sh" in args, args
    destination = Path(args[args.index("-o") + 1])
    destination.write_text("#!/bin/sh\nfake-bootstrap\n")
    if os.environ.get("FAKE_DOWNLOAD_FAIL"):
        fail("fixture download failure", 22)
    if os.environ.get("FAKE_SIGNAL"):
        os.kill(os.getppid(), getattr(signal, os.environ["FAKE_SIGNAL"]))
elif name == "fake-bootstrap":
    if os.environ.get("FAKE_BOOTSTRAP_FAIL"):
        fail("fixture bootstrap failure", 23)
    if os.environ.get("FAKE_BOOTSTRAP_NO_UV"):
        sys.exit(0)
    destination = Path(os.environ["HOME"]) / ".local" / "bin" / "uv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(Path(sys.argv[0]).read_text())
    destination.chmod(0o755)
elif name == "mktemp":
    template = Path(args[0])
    fd, path = tempfile.mkstemp(prefix=template.name.rstrip("X"), dir=template.parent)
    os.close(fd)
    print(path)
elif name == "rm":
    for arg in args:
        if not arg.startswith("-"):
            Path(arg).unlink(missing_ok=True)
else:
    fail("unexpected fake command: " + name, 99)
'''


@pytest.fixture
def installer(tmp_path):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    task_home = tmp_path / "home"
    task_home.mkdir()
    scratch = tmp_path / "temporary scripts"
    scratch.mkdir()
    log = tmp_path / "commands.jsonl"
    env = {"PATH": str(fake_bin), "HOME": str(task_home), "TMPDIR": str(scratch),
           "FAKE_LOG": str(log), "FAKE_TOOL_BIN": str(task_home / "custom tool bin")}

    def run(*, uv_present=True, tool_present=True, **settings):
        names = ["curl", "fake-bootstrap", "mktemp", "rm"]
        if uv_present:
            names.append("uv")
        if tool_present:
            names.append("co-science")
        for name in names:
            path = fake_bin / name
            path.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
            path.chmod(0o755)
        assert ROOT.is_dir()
        result = subprocess.run(["/bin/sh", str(ROOT / "packaging/install.sh")],
                                env={**env, **settings}, cwd=tmp_path, capture_output=True,
                                text=True, timeout=15)
        commands = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert scratch.is_dir()
        assert list(scratch.iterdir()) == [], "bootstrap temporary scripts must be cleaned up"
        return result, commands, env

    return run


def installs(commands):
    return [command for command in commands if command[:3] == ["uv", "tool", "install"]]


def test_installer_uses_existing_uv_once_and_checks_the_command(installer):
    result, commands, _ = installer()
    assert result.returncode == 0, result.stderr
    assert installs(commands) == [["uv", "tool", "install", "--python", "3.12", "--force",
                                   PUBLIC_URL]]
    assert commands[-1] == ["co-science", "--version"]
    assert not any(command[0] == "curl" for command in commands)
    assert "co-science fixture version" in result.stdout


def test_installer_bootstraps_uv_then_installs_once(installer):
    result, commands, _ = installer(uv_present=False)
    assert result.returncode == 0, result.stderr
    assert len(installs(commands)) == 1
    assert ["fake-bootstrap"] in commands
    assert commands[-1] == ["co-science", "--version"]


@pytest.mark.parametrize("setting,diagnostic", [
    ("FAKE_DOWNLOAD_FAIL", "fixture download failure"),
    ("FAKE_BOOTSTRAP_FAIL", "fixture bootstrap failure"),
])
def test_installer_stops_on_failed_bootstrap(installer, setting, diagnostic):
    result, commands, _ = installer(uv_present=False, **{setting: "1"})
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not installs(commands)
    if setting == "FAKE_DOWNLOAD_FAIL":
        assert ["fake-bootstrap"] not in commands, "a partial installer must never run"


def test_installer_keeps_install_stderr_and_points_to_the_public_source(installer):
    result, commands, _ = installer(FAKE_INSTALL_FAIL="1")
    assert result.returncode != 0
    assert "fixture resolver failure" in result.stderr
    assert "installation failed" in result.stderr.lower()
    assert "gh auth" not in result.stderr
    assert PUBLIC_URL.removeprefix("git+") in result.stderr
    assert "clone failed" not in result.stderr.lower()
    assert len(installs(commands)) == 1
    assert not any(command[0] == "co-science" for command in commands)


def test_installer_succeeds_with_path_advice_if_command_is_not_visible(installer):
    result, commands, env = installer(tool_present=False)
    assert result.returncode == 0, result.stderr
    assert "uv tool update-shell" in result.stdout
    assert env["FAKE_TOOL_BIN"] in result.stdout
    assert len(installs(commands)) == 1
    assert ["uv", "tool", "dir", "--bin"] in commands
    assert not any(command[0] == "co-science" for command in commands)


@pytest.mark.parametrize("setting", ["FAKE_BIN_FAIL", "FAKE_BIN_EMPTY"])
def test_installer_falls_back_to_the_default_tool_bin(installer, setting):
    """Preservation: a failed/empty bin lookup is PATH advice, not an install failure."""
    result, commands, env = installer(tool_present=False, **{setting: "1"})
    assert result.returncode == 0, result.stderr
    assert str(Path(env["HOME"]) / ".local" / "bin") in result.stdout
    assert "uv tool update-shell" in result.stdout
    assert "new terminal" in result.stdout
    assert len(installs(commands)) == 1


@pytest.mark.parametrize("signal_name", ["SIGHUP", "SIGINT", "SIGTERM"])
def test_interrupted_download_cleans_up_without_running_the_partial(installer, signal_name):
    """Preservation: the script's signal traps also run its temporary-file cleanup."""
    result, commands, _ = installer(uv_present=False, FAKE_SIGNAL=signal_name)
    assert result.returncode == 128 + getattr(signal, signal_name)
    assert ["fake-bootstrap"] not in commands
    assert not installs(commands)


def test_bootstrap_without_uv_gives_path_advice_and_stops(installer):
    """Preservation: an installer exiting successfully does not prove uv is on PATH."""
    result, commands, _ = installer(uv_present=False, FAKE_BOOTSTRAP_NO_UV="1")
    assert result.returncode != 0
    assert "PATH" in result.stderr
    assert "rerun this script" in result.stderr
    assert not installs(commands)


def test_broken_installed_entry_point_is_not_reported_as_success(installer):
    """Preservation: the version check's failure stays visible and nonzero."""
    result, commands, _ = installer(FAKE_VERSION_FAIL="1")
    assert result.returncode == 8
    assert "fixture entry-point failure" in result.stderr
    assert len(installs(commands)) == 1
    assert commands[-1] == ["co-science", "--version"]


@pytest.mark.parametrize("name", ["README.md", "CLAUDE.md", "packaging/install.sh"])
def test_executable_install_lines_pin_python_and_public_sources(name):
    """Preservation: prose about Python does not count as a pinned command."""
    source = read_surface(name)
    if name.endswith(".md"):
        # Only executable examples in fenced blocks count, never inline advice/prose.
        source = "\n".join(re.findall(r"```(?:sh|bash|powershell)\n(.*?)```", source, re.S))
    commands = []
    for line in source.splitlines():
        match = re.fullmatch(r"(?:if )?(uv tool install .*?)(?:; then)?", line.strip())
        if match:
            commands.append(shlex.split(match[1]))
    assert commands, f"{name} contains no actual install command"
    for command in commands:
        assert command[:5] == ["uv", "tool", "install", "--python", "3.12"]
        requirement = command[5:]
        if requirement[0] == "--force":
            requirement = requirement[1:]
        assert requirement in (
            ["manyruns"], ["manyruns[harness]"], ["manyruns[agents]"], [PUBLIC_URL],
        )


def test_install_and_repair_surfaces_do_not_recommend_pip():
    for name in ("README.md", "CONTRIBUTING.md", "CHANGELOG.md", "Makefile",
                 "manyruns/installation.py", "packaging/install.sh", "manyruns/app.py",
                 "manyruns/harness/display.py"):
        source = read_surface(name)
        assert not re.search(r"\bpip(?:3)?\s+install\b", source, re.I), name


def test_command_surfaces_do_not_mutate_the_venv_or_configure_plaintext_credentials():
    """Preservation: scan the owned command surfaces, including their comments."""
    for name in ("README.md", "Makefile", "manyruns/installation.py", "packaging/install.sh"):
        source = read_surface(name)
        assert " ".join(("uv", "pip", "install")) not in source
        assert "--engine real" not in source
        assert "credential.helper" not in source
        assert "raw.githubusercontent.com" not in source
    make = read_surface("Makefile")
    assert "--with" not in make
    assert "install-real" not in make
    assert "demo-real" not in make


def test_readme_recipe_count_matches_nonempty_packaged_catalog():
    """Preservation: the documented count follows the actual bundled recipes."""
    root = resources.files("manyruns").joinpath("configs", "recipe")
    assert root.is_dir()
    recipes = [path for path in root.iterdir() if path.name.endswith(".yaml")]
    assert recipes
    assert f"{len(recipes)} bundled recipes" in read_surface("README.md")


def test_convenience_script_is_executable_posix_shell():
    """Preservation: syntax validation executes none of the installation commands."""
    assert ROOT.is_dir()
    script = ROOT / "packaging" / "install.sh"
    assert script.stat().st_mode & stat.S_IXUSR
    assert script.read_text().startswith("#!/bin/sh\n")
    assert "set -eu" in script.read_text().splitlines()
    result = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True,
                            timeout=5)
    assert result.returncode == 0, result.stderr
