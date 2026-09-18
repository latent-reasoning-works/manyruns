"""harness/manyruns chat — agentic REPL. CI-safe (manyAgents loop mocked)."""

import pytest


def test_dr_tool_schema_and_run():
    """The run_dr_workflow tool exposes a schema and runs via the DR path."""
    pytest.importorskip("manyagents")
    from manyruns.harness.agent_tools import make_dr_tool

    tool = make_dr_tool()
    schema = tool.to_openai_schema()
    assert schema["function"]["name"] == "run_dr_workflow"
    assert "preset" in schema["function"]["parameters"]["properties"]


def test_chat_repl_drives_loop(monkeypatch):
    """chat reads a line, runs the loop, prints the answer, exits on EOF."""
    pytest.importorskip("manyagents")
    from click.testing import CliRunner

    # Stub the loop so no model/network is needed; assert it's invoked with our tool.
    import manyruns.harness.__main__ as cli_mod

    seen = {}

    class FakeResult:
        answer = "Trustworthiness is 0.998."
        messages = [{"role": "assistant", "content": "Trustworthiness is 0.998."}]

    async def fake_loop(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["tools"] = [t.name for t in kwargs.get("tools", [])]
        seen["agent"] = kwargs.get("agent")
        return FakeResult()

    monkeypatch.setattr("manyagents.agent_loop.run_agent_loop", fake_loop, raising=False)

    # Feed one line then EOF.
    result = CliRunner().invoke(
        cli_mod.cli,
        ["--no-banner", "chat", "--agent", "mock"],
        input="embed swissroll with umap\n",
    )
    assert result.exit_code == 0, result.output
    assert "Trustworthiness is 0.998." in result.output
    assert seen["prompt"] == "embed swissroll with umap"
    assert seen["tools"] == ["run_dr_workflow"]
    assert seen["agent"] == "mock"


def test_chat_claude_uses_api_key_flag(monkeypatch):
    """--agent claude with --api-key sets ANTHROPIC_API_KEY for the adapter."""
    pytest.importorskip("manyagents")
    import os

    from click.testing import CliRunner

    import manyruns.harness.__main__ as cli_mod

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seen = {}

    class FakeResult:
        answer = "done"
        messages = []

    async def fake_loop(prompt, **kwargs):
        seen["agent"] = kwargs.get("agent")
        seen["key"] = os.environ.get("ANTHROPIC_API_KEY")
        return FakeResult()

    monkeypatch.setattr("manyagents.agent_loop.run_agent_loop", fake_loop, raising=False)

    result = CliRunner().invoke(
        cli_mod.cli,
        ["--no-banner", "chat", "--agent", "claude", "--api-key", "sk-test-123"],
        input="hi\n",
    )
    assert result.exit_code == 0, result.output
    assert seen["agent"] == "claude"
    assert seen["key"] == "sk-test-123"  # flag flowed into the env for ClaudeAdapter


def test_chat_help_lists_claude():
    """`chat --help` mentions the claude backend and the api-key option."""
    from click.testing import CliRunner

    import manyruns.harness.__main__ as cli_mod

    result = CliRunner().invoke(cli_mod.cli, ["--no-banner", "chat", "--help"])
    assert result.exit_code == 0
    assert "claude" in result.output
    assert "--api-key" in result.output


def test_prompt_claude_uses_api_key_flag(monkeypatch):
    """prompt --agent claude --api-key sets ANTHROPIC_API_KEY for the adapter."""
    pytest.importorskip("manyagents")
    import os

    from click.testing import CliRunner

    import manyruns.harness.__main__ as cli_mod

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    seen = {}

    class _Resp:
        answer = "ok"
        trace_path = None

    def fake_run(request, backend=None):
        seen["agent"] = request.agent
        seen["key"] = os.environ.get("ANTHROPIC_API_KEY")
        return _Resp()

    monkeypatch.setattr("manyruns.harness.interface.run", fake_run, raising=False)

    result = CliRunner().invoke(
        cli_mod.cli,
        ["--no-banner", "prompt", "hi", "--agent", "claude", "--api-key", "sk-test-xyz"],
    )
    assert result.exit_code == 0, result.output
    assert seen["agent"] == "claude"
    assert seen["key"] == "sk-test-xyz"


def test_prompt_help_lists_claude_and_api_key():
    from click.testing import CliRunner

    import manyruns.harness.__main__ as cli_mod

    result = CliRunner().invoke(cli_mod.cli, ["--no-banner", "prompt", "--help"])
    assert result.exit_code == 0
    assert "claude" in result.output
    assert "--api-key" in result.output
