"""`docs/harness.md`: every name it lists imports, every verb it names exists, and the wording of
four files says that a harness drives manyruns through the CLI or as a library.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.source_checkout

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "harness.md"

SECTIONS = ["## The intended use", "## The action space wrapper", "## The choice document",
            "## CLI mode", "## Library mode", "## The record", "## Who is a harness"]


def _section(text: str, heading: str) -> str:
    start = text.index(heading + "\n")
    rest = text[start + len(heading) + 1:]
    end = re.search(r"^## ", rest, re.M)
    return rest[:end.start()] if end else rest


def _library_names(text: str) -> list[str]:
    rows = re.findall(r"^\| `(manyruns\.[A-Za-z0-9_.]+)` \|", _section(text, "## Library mode"),
                      re.M)
    return rows


def _unresolved(names: list[str]) -> list[str]:
    bad = []
    for name in names:
        module, _, attr = name.rpartition(".")
        try:
            getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError):
            bad.append(name)
    return bad


def test_page_has_the_sections_in_order():
    assert ROOT.is_dir() and PAGE.is_file()
    text = PAGE.read_text()
    found = [line for line in text.splitlines() if line.startswith("## ")]
    assert found == SECTIONS


def test_library_names_import():
    assert ROOT.is_dir()
    names = _library_names(PAGE.read_text())
    assert len(names) == len(set(names)) == 24
    assert _unresolved(names) == []


def test_a_removed_name_fails_the_check():
    assert ROOT.is_dir()
    copy = PAGE.read_text().replace("| `manyruns.store.read` |", "| `manyruns.store.write` |")
    assert _unresolved(_library_names(copy)) == ["manyruns.store.write"]


def test_cli_verbs_exist(capsys):
    from manyruns import app

    assert ROOT.is_dir()
    cli = _section(PAGE.read_text(), "## CLI mode")
    for verb in ("actions", "run"):
        assert f"manyruns {verb}" in cli
        assert verb in app._SUBCOMMANDS
    assert "--choice" in cli
    assert "decision <decision_id> recorded in" in cli and "run <run_id> recorded in" in cli
    with pytest.raises(SystemExit):
        app.main(["run", "--help"])
    out = capsys.readouterr().out
    assert "--choice" in out and "--chooser" not in out


def test_the_record_section_states_the_missing_rule():
    assert ROOT.is_dir()
    record = " ".join(_section(PAGE.read_text(), "## The record").split())
    assert "A row without `proposed` reads as missing, not as an accept." in record
    assert "A row without `gate` reads as missing, not as `false`." in record


WORDING = [
    ("CLAUDE.md",
     ["drives manyruns from outside, through the CLI or as a library, like any user.",
      "manyruns names no learner: no dependency, import, engine value, config, or CLI path.",
      "docs/harness.md"],
     ["drives the CLI from outside"]),
    ("README.md",
     ["outside this package", "manyruns's CLI or its Python API, as any other user would"],
     ["a separate process"]),
    ("manyruns/__init__.py",
     ["drives it through the CLI or as a library, as any other user would"],
     ["drives it through the CLI as any other user would"]),
    ("docs/environment-contract.md", ["drives it through the CLI or as a library"], []),
]


@pytest.mark.parametrize("path, present, absent", WORDING, ids=[w[0] for w in WORDING])
def test_wording_names_both_modes(path, present, absent):
    assert ROOT.is_dir()
    text = " ".join((ROOT / path).read_text().split())
    for sentence in present:
        assert sentence in text, sentence
    for sentence in absent:
        assert sentence not in text, sentence
