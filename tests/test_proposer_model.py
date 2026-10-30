"""The proposer: one model value, one loop entry point, and the record it leaves.

`proposer_model` resolves the model a proposer call names. `propose` runs the agent loop once,
stamps every loop event with the session's `run_id`, and appends the transcript to the run
folder. manyagents is not a dependency of this checkout, so the loop is a fake installed into
`sys.modules`; a skip on the real package would let these tests pass while testing nothing.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import types
from dataclasses import dataclass, field

import pytest

from manyruns import agents, artifacts, store

AGENTS_MODEL = agents.MODEL


@pytest.fixture
def no_model_env(monkeypatch):
    monkeypatch.delenv("MANYRUNS_PROPOSER_MODEL", raising=False)
    monkeypatch.delenv("MANYAGENTS_MODEL", raising=False)


# ── one proposer model value ─────────────────────────────────────────────────────────────────

def test_the_manyruns_variable_wins(monkeypatch):
    monkeypatch.setenv("MANYRUNS_PROPOSER_MODEL", "qwen3:8b")
    monkeypatch.setenv("MANYAGENTS_MODEL", "claude-opus-5")
    assert agents.proposer_model() == "qwen3:8b"


def test_the_manyagents_variable_is_not_read(monkeypatch):
    monkeypatch.delenv("MANYRUNS_PROPOSER_MODEL", raising=False)
    monkeypatch.setenv("MANYAGENTS_MODEL", "claude-opus-5")
    assert agents.proposer_model() == AGENTS_MODEL == "claude-opus-4-8"


def test_a_blank_variable_falls_through(monkeypatch):
    monkeypatch.setenv("MANYRUNS_PROPOSER_MODEL", "  ")
    monkeypatch.delenv("MANYAGENTS_MODEL", raising=False)
    assert agents.proposer_model() == "claude-opus-4-8"


def test_the_fallback_argument_comes_before_agents_model(monkeypatch):
    monkeypatch.delenv("MANYRUNS_PROPOSER_MODEL", raising=False)
    monkeypatch.delenv("MANYAGENTS_MODEL", raising=False)
    assert agents.proposer_model("qwen3") == "qwen3"
    assert agents.proposer_model() == "claude-opus-4-8"
    assert agents.proposer_model("   ") == "claude-opus-4-8"


def test_the_value_is_returned_stripped(monkeypatch):
    monkeypatch.setenv("MANYRUNS_PROPOSER_MODEL", "  qwen3:8b ")
    monkeypatch.delenv("MANYAGENTS_MODEL", raising=False)
    assert agents.proposer_model() == "qwen3:8b"


def test_the_variable_is_read_through_the_accessor(monkeypatch):
    """The other half of `tests/test_env.py::test_no_module_reads_the_environment_around_the_accessor`,
    which catches a read around `env.get` by its shape: this one names the call."""
    monkeypatch.delenv("MANYAGENTS_MODEL", raising=False)
    calls = []

    def get(key, default=None):
        calls.append(key)
        return "from-accessor"

    monkeypatch.setattr(agents._env, "get", get)
    assert agents.proposer_model() == "from-accessor"
    assert calls == ["PROPOSER_MODEL"]


# ── the run row names the proposer model ─────────────────────────────────────────────────────

def test_extra_proposer_model_is_on_the_row(tmp_path):
    store.append({"run_id": "abc123def456"}, out_dir=tmp_path,
                 extra=agents.proposer_extra("qwen3:8b", "ollama"))
    rows = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    row = rows[-1]
    assert row["extra"]["proposer"] == {"model": "qwen3:8b", "agent": "ollama"}
    assert row["run_id"] == "abc123def456"
    assert row["schema_version"] == 1


def test_proposer_extra_shape():
    assert agents.proposer_extra("m", "claude") == {
        "extra": {"proposer": {"model": "m", "agent": "claude"}}}


# ── the proposer loop, against a fake manyagents ─────────────────────────────────────────────

@dataclass
class FakeResult:
    answer: str = "a"
    messages: list = field(default_factory=list)
    steps: int = 2
    stopped: str = "end_turn"
    tool_calls: list = field(default_factory=list)


def _messages():
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "p"},
            {"role": "assistant", "content": "hi"}, {"role": "tool", "content": "ok"}]


class Loop:
    """The fake `run_agent_loop` and what it saw."""

    def __init__(self, session):
        self.session = session
        self.calls: list[dict] = []
        self.agent_ids: list[str] = []
        self.raises: Exception | None = None

    async def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        self.agent_ids.append(self.session.agent_id)
        if self.raises is not None:
            raise self.raises
        on_event = kwargs["on_event"]
        on_event("assistant", {"text": "hi"})
        on_event("tool_call", {"name": "echo", "arguments": {}})
        on_event("tool_result", {"name": "echo", "result": "ok"})
        return FakeResult(messages=_messages())


@pytest.fixture
def session(tmp_path):
    return types.SimpleNamespace(run_id="abc123def456", out_dir=tmp_path / "out",
                                 agent_id="human")


@pytest.fixture
def traces(tmp_path, monkeypatch):
    monkeypatch.setenv("MANYRUNS_TRACE", f"jsonl://{tmp_path / 'traces'}")
    return tmp_path / "traces"


@pytest.fixture
def fake_loop(monkeypatch, session, no_model_env):
    loop = Loop(session)

    class Claude:
        DEFAULT_MAX_TOKENS = 4096

    class Ollama:
        DEFAULT_MODEL = "qwen3"
        DEFAULT_MAX_TOKENS = 2000

    pkg = types.ModuleType("manyagents")
    pkg.__path__ = []
    agent_loop = types.ModuleType("manyagents.agent_loop")
    agent_loop.run_agent_loop = loop
    agent_loop.AgentResult = FakeResult
    adapters = types.ModuleType("manyagents.adapters")
    adapters.ADAPTER_REGISTRY = {"claude": Claude, "ollama": Ollama}
    monkeypatch.setitem(sys.modules, "manyagents", pkg)
    monkeypatch.setitem(sys.modules, "manyagents.agent_loop", agent_loop)
    monkeypatch.setitem(sys.modules, "manyagents.adapters", adapters)
    return loop


def _events(traces, kind=None, run_id="abc123def456"):
    path = traces / f"{run_id}.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def _transcript(session):
    return artifacts.root(session.out_dir, session.run_id) / "agent" / "messages.jsonl"


def test_the_model_is_passed_explicitly(fake_loop, session, traces, monkeypatch):
    monkeypatch.setenv("MANYRUNS_PROPOSER_MODEL", "qwen3:8b")
    result = agents.propose(session, "embed this")
    assert isinstance(result, FakeResult)
    (call,) = fake_loop.calls
    assert call["prompt"] == "embed this"
    assert call["model"] == "qwen3:8b"
    assert call["agent"] == "claude"
    assert callable(call["on_event"])
    assert call["tools"] is None and call["system_prompt"] is None and call["max_steps"] == 8


def test_a_non_claude_agent_gets_its_own_default(fake_loop, session, traces):
    agents.propose(session, "x", agent="ollama")
    assert fake_loop.calls[0]["model"] == "qwen3"
    assert fake_loop.calls[0]["agent"] == "ollama"


def test_the_session_is_attributed_to_the_model_during_the_call_only(fake_loop, session, traces):
    agents.propose(session, "x")
    assert fake_loop.agent_ids == ["llm:claude-opus-4-8"]
    assert session.agent_id == "human"


def test_the_agent_is_restored_when_the_loop_raises(fake_loop, session, traces):
    fake_loop.raises = ValueError("boom")
    agents.propose(session, "x")
    assert fake_loop.agent_ids == ["llm:claude-opus-4-8"]
    assert session.agent_id == "human"


def test_three_loop_events_become_three_trace_lines(fake_loop, session, traces):
    agents.propose(session, "x")
    loop_events = [e for e in _events(traces) if e["kind"] != "proposer_call"]
    assert [e["kind"] for e in loop_events] == [
        "proposer_assistant", "proposer_tool_call", "proposer_tool_result"]
    call_ids = {e["call_id"] for e in loop_events}
    assert len(call_ids) == 1 and len(next(iter(call_ids))) == 12
    for e in loop_events:
        assert e["run_id"] == "abc123def456"
        assert e["agent_id"] == "llm:claude-opus-4-8"
        assert e["model"] == "claude-opus-4-8"
    assert loop_events[0]["text"] == "hi"
    assert loop_events[1]["name"] == "echo" and loop_events[1]["arguments"] == {}
    assert loop_events[2]["result"] == "ok"
    (summary,) = _events(traces, "proposer_call")
    assert summary["call_id"] in call_ids


def test_one_transcript_line_per_message(fake_loop, session, traces):
    agents.propose(session, "x")
    lines = [json.loads(line) for line in _transcript(session).read_text().splitlines()]
    assert [line["index"] for line in lines] == [0, 1, 2, 3]
    for line, message in zip(lines, _messages()):
        assert line["kind"] == "message"
        assert line["run_id"] == session.run_id
        assert line["agent_id"] == "llm:claude-opus-4-8"
        assert line["model"] == "claude-opus-4-8"
        assert line["agent"] == "claude"
        assert line["max_tokens"] == 4096
        assert re.fullmatch(r"[0-9a-f]{64}", line["context_sha256"])
        assert line["message"] == message
    assert len({line["call_id"] for line in lines}) == 1


def test_a_second_call_appends(fake_loop, session, traces):
    agents.propose(session, "x")
    agents.propose(session, "x")
    lines = [json.loads(line) for line in _transcript(session).read_text().splitlines()]
    assert len(lines) == 8
    assert len({line["call_id"] for line in lines}) == 2


def test_the_context_hash_is_stable(fake_loop, session, traces):
    agents.propose(session, "p", system_prompt="s")
    agents.propose(session, "p", system_prompt="s")
    hashes = {json.loads(line)["context_sha256"]
              for line in _transcript(session).read_text().splitlines()}
    assert len(hashes) == 1
    assert re.fullmatch(r"[0-9a-f]{64}", hashes.pop())


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_folder_does_not_raise(fake_loop, session, traces):
    session.out_dir.mkdir(parents=True)
    session.out_dir.chmod(0o500)
    try:
        result = agents.propose(session, "x")
    finally:
        session.out_dir.chmod(0o700)
    assert isinstance(result, FakeResult)
    (summary,) = _events(traces, "proposer_call")
    assert summary["messages_path"] is None
    assert not _transcript(session).exists()


def test_a_finished_call_is_summarised(fake_loop, session, traces):
    agents.propose(session, "x")
    (summary,) = _events(traces, "proposer_call")
    assert summary["steps"] == 2
    assert summary["stopped"] == "end_turn"
    assert summary["n_messages"] == 4
    assert summary["error"] is None
    assert summary["model"] == "claude-opus-4-8"
    assert summary["agent"] == "claude"
    assert summary["max_tokens"] == 4096
    assert re.fullmatch(r"[0-9a-f]{64}", summary["context_sha256"])
    assert summary["messages_path"] == str(_transcript(session))
    assert summary["run_id"] == "abc123def456"
    assert summary["agent_id"] == "llm:claude-opus-4-8"


def _assert_failed(traces, session, error):
    (summary,) = _events(traces, "proposer_call")
    assert summary["stopped"] == "error"
    assert summary["error"] == error
    assert summary["steps"] == 0
    assert summary["n_messages"] == 0
    assert summary["messages_path"] is None
    assert not _transcript(session).exists()


def test_manyagents_is_absent(monkeypatch, session, traces, no_model_env):
    monkeypatch.setitem(sys.modules, "manyagents.agent_loop", None)
    assert agents.propose(session, "x") is None
    _assert_failed(traces, session, "ImportError")
    (summary,) = _events(traces, "proposer_call")
    assert summary["model"] == "claude-opus-4-8"
    assert session.agent_id == "human"


def test_the_loop_raises(fake_loop, session, traces):
    fake_loop.raises = ValueError("boom")
    assert agents.propose(session, "x") is None
    _assert_failed(traces, session, "ValueError")


def test_a_running_event_loop_is_refused(fake_loop, session, traces):
    async def inside():
        return agents.propose(session, "x")

    assert asyncio.run(inside()) is None
    assert fake_loop.calls == []
    _assert_failed(traces, session, "RuntimeError")
    assert session.agent_id == "human"
