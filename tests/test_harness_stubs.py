"""The harness `label` and `rlhf` stubs are gone; `manyruns label` writes real rows instead."""
from __future__ import annotations

import pytest

pytest.importorskip("click")

from click.testing import CliRunner  # noqa: E402

from manyruns.harness.__main__ import cli  # noqa: E402


@pytest.mark.parametrize("argv", [["label", "-d", "x", "-o", "y"], ["rlhf", "-d", "x"]])
def test_the_stub_is_gone(argv):
    result = CliRunner().invoke(cli, argv)
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_the_other_commands_remain():
    assert sorted(cli.commands) == ["chat", "list", "presets", "prompt", "run"]


def test_the_docstrings_name_no_stub():
    import manyruns.harness.__main__ as main
    import manyruns.harness.runner as runner

    assert "harness label" not in main.__doc__ and "harness rlhf" not in main.__doc__
    assert "(run/label/rlhf)" not in runner.__doc__ and "(run)" in runner.__doc__
