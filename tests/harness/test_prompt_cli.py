"""Prompt stream: `harness prompt` -> AgentBackend -> manyAgents.

The mock round-trip needs manyAgents installed (MockAdapter) but no model,
network, or GPU — same spirit as test_phase5_cli's mock DR test.
"""

from click.testing import CliRunner


def test_prompt_round_trip_mock():
    """prompt subcommand -> AgentBackend -> MockAdapter -> printed answer."""
    import pytest

    pytest.importorskip("manyagents")  # MockAdapter lives in manyAgents
    from manyruns.harness.__main__ import cli

    result = CliRunner().invoke(
        cli, ["--no-banner", "prompt", "hello manyruns", "--agent", "mock"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip()  # a response was printed


def test_prompt_request_routes_to_agent_backend():
    """A Request carrying a prompt selects AgentBackend in the module-level run()."""
    from manyruns.harness.interface import _BACKENDS, AgentBackend

    assert _BACKENDS["agent"] is AgentBackend


def test_agent_backend_requires_prompt():
    import pytest

    from manyruns.harness.interface import AgentBackend, Request

    with pytest.raises(ValueError):
        AgentBackend().run(Request())
