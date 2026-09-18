"""`manyruns/agents.py` — the one place this product asks a model a question.

NO NETWORK, NO SDK, NO KEY in any test here. The seam's whole contract is that it is optional and
guarded, so what is tested is the GUARDS and the SHAPE of the request — the two things that decide
whether the shell survives a model tier that is absent, broken, or wrong.

`manyagents` and `anthropic` are absent from this checkout and from CI (the workflow installs the
product layer only), so the unavailable path is the DEFAULT one here rather than a special case.
"""
from __future__ import annotations

import asyncio

import pytest

from manyruns import agents


def test_injected_intelligence_owns_its_model_and_receives_portable_tools():
    class Intelligence:
        async def chat(self, messages, **kwargs):
            assert "model" not in kwargs
            assert messages[-1]["content"] == "measured evidence"
            tool = kwargs["tools"][0]
            assert tool["type"] == "function"
            assert tool["function"]["parameters"]["required"] == ["choice"]
            return {"tool_calls": [{"name": "choose", "arguments": '{"choice":"a"}'}]}

    assert asyncio.run(agents.call_tool_async(
        system="s", prompt="measured evidence", name="choose", description="d",
        schema={"type": "object", "required": ["choice"]}, adapter=Intelligence(),
    )) == {"choice": "a"}


def test_injected_intelligence_outage_does_not_fall_back_to_another_provider(monkeypatch):
    class Unavailable:
        async def chat(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    def forbidden(*_args, **_kwargs):
        pytest.fail("an explicit adapter must not disclose evidence to a fallback provider")

    monkeypatch.setattr(agents, "_via_anthropic", forbidden)
    assert asyncio.run(agents.call_tool_async(
        system="s", prompt="p", name="n", description="d",
        schema={}, adapter=Unavailable())) is None


def test_a_missing_backend_returns_none_rather_than_raising():
    """THE contract. Every caller treats `None` as "the rule tier decides", so a missing package
    must not reach them as an exception — the interactive shell would die on an optional call."""
    got = agents.call_tool(system="s", prompt="p", name="choose",
                           description="d", schema={"type": "object"})

    assert got is None


def test_choose_one_refuses_an_empty_choice_set_without_calling_anything(monkeypatch):
    """A forced choice among nothing is a call worth not making. Asserted by making any call
    fail loudly — if `choose_one` reached the backend, this raises instead of returning None."""
    def explode(**_kw):
        raise AssertionError("called the model for an empty choice set")

    monkeypatch.setattr(agents, "call_tool", explode)

    assert agents.choose_one(system="s", prompt="p", key="recipe", choices=[]) is None


def test_a_reply_outside_the_choice_set_is_refused(monkeypatch):
    """The enum makes it unlikely; this makes it impossible. A model that answers with something
    not on offer must not have that answer acted on — `intent.route_recipe` would otherwise route
    to a recipe the data cannot run."""
    monkeypatch.setattr(agents, "call_tool", lambda **_kw: {"recipe": "not_a_recipe"})

    assert agents.choose_one(system="s", prompt="p", key="recipe",
                             choices=["embed", "cflows"]) is None


def test_a_reply_inside_the_choice_set_is_taken(monkeypatch):
    """The control: the refusals above must not be refusing everything."""
    monkeypatch.setattr(agents, "call_tool", lambda **_kw: {"recipe": "cflows"})

    assert agents.choose_one(system="s", prompt="p", key="recipe",
                             choices=["embed", "cflows"]) == "cflows"


def test_one_model_name_for_every_caller():
    """Two callers ask this module questions (`intent.route_recipe`, `tune.llm_override`). A
    model name per caller would make their answers incomparable to each other — the same argument
    `vocab.py` makes for one home per vocabulary."""
    import inspect

    from manyruns import intent, tune

    for module in (intent, tune):
        assert "claude-" not in inspect.getsource(module), (
            f"{module.__name__} names a model directly; `agents.MODEL` is the one home")


def test_intent_no_longer_calls_the_sdk_itself():
    """The consolidation, asserted rather than described. `intent._llm_choose` held manyruns's
    only LLM call until `tune` needed a second; two direct SDK calls would have meant two sets of
    guards and two places to add a provider."""
    import inspect

    from manyruns import intent

    src = inspect.getsource(intent)
    assert "import anthropic" not in src
    assert "agents.choose_one" in src


# ── the constraint must survive the backend that reads a different shape ──────────────────
#: `ClaudeAdapter._to_anthropic_tools`, copied VERBATIM from the commit `uv.lock` pins
#: (`manyagents@f44d852`, `adapters/claude_adapter.py`). A copy rather than an import because
#: manyagents is absent here and in CI — and a copy is what makes this a CONTRACT test: if
#: upstream changes the shape it reads, this keeps passing while reality diverges, so the copy is
#: dated. Re-derive this fixture when changing the manyagents version or tool schema.
def _as_manyagents_would(tools):
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append({"name": fn["name"], "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}})})
    return out


def test_the_schema_survives_the_manyagents_converter():
    """THE DEFECT PR #77 SHIPPED, pinned so it cannot return.

    `_to_anthropic_tools` reads the OPENAI shape — `fn.get("parameters", ...)` — while the direct
    SDK reads Anthropic's `input_schema`. Sending only `input_schema` meant the default fired and
    the tool arrived with `{"type": "object", "properties": {}}`: no enum, no `required`, no
    `additionalProperties`, no `strict`.

    So "a forced choice over a fixed set" was, on the PREFERRED backend, a free-text call. Nothing
    failed loudly — the caller-side check rejected the answer or the fallback produced a better
    one — which is exactly why a test that mocked the seam could not have caught it.
    """
    schema = {"type": "object",
              "properties": {"recipe": {"type": "string", "enum": ["embed", "cflows"]}},
              "required": ["recipe"], "additionalProperties": False}

    forwarded = _as_manyagents_would([agents._tool("choose", "pick one", schema)])[0]

    assert forwarded["input_schema"] == schema, "the constraint was dropped in translation"
    assert forwarded["input_schema"]["properties"]["recipe"]["enum"] == ["embed", "cflows"]
    assert forwarded["name"] == "choose"


def test_the_tool_carries_the_schema_under_both_names():
    """Both keys, not a branch on which backend is about to run: the tool dict is built before a
    backend is chosen, and a converter that ignores an extra key is the cheaper contract."""
    schema = {"type": "object", "properties": {}}
    tool = agents._tool("adjust", "d", schema)

    assert tool["input_schema"] is schema, "the direct SDK reads this one"
    assert tool["parameters"] is schema, "manyagents reads this one"


def test_an_unforced_backend_degrades_into_the_forced_one(monkeypatch):
    """`ClaudeAdapter.chat` has NO `tool_choice` — measured, its request is built from model,
    messages, max_tokens, system and tools and nothing else — so the model may answer with prose
    instead of calling the tool. The direct SDK cannot.

    The ordering makes that survivable rather than fatal: a manyagents turn with no matching tool
    call yields None, the loop moves on, and the SDK answers the same question with the call
    forced. The unforced backend degrades into the forced one, never into a guess.
    """
    calls: list = []

    def no_tool_call(system, prompt, tool):
        calls.append("manyagents")
        return None                      # answered, but not with the tool

    def forced(system, prompt, tool):
        calls.append("anthropic")
        return {"recipe": "cflows"}

    monkeypatch.setattr(agents, "_via_manyagents", no_tool_call)
    monkeypatch.setattr(agents, "_via_anthropic", forced)

    got = agents.call_tool(system="s", prompt="p", name="choose", description="d",
                           schema={"type": "object"})

    assert got == {"recipe": "cflows"}
    assert calls == ["manyagents", "anthropic"], "the fallthrough did not happen in order"


def test_availability_needs_a_key_AND_a_backend(monkeypatch):
    """`available()` checked the KEY ALONE and so answered its own question wrong: measured with
    `ANTHROPIC_API_KEY=sk-fake` and neither package installed, it returned True while an actual
    call returned None — the tier reported ON in exactly the case where nothing can work.

    The key is not a proxy for intent either. `app._load_dotenv()` reads a repo-local `.env` at
    startup and `harness/__main__.py` writes a prompted key into `os.environ`, so a stale file or
    an earlier session sets it without anyone asking for a model call now.
    """
    import importlib.util

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    # Model absence explicitly: the integration environment has both backends installed.
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    assert agents.available() is False, "no backend installed, so nothing can work"

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    assert agents.available() is True

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert agents.available() is False


def test_one_availability_predicate_not_two():
    """There were two and they disagreed: `intent.llm_available()` gated on the raw SDK while the
    seam prefers manyAgents, so with a key and nothing installed they answered False and True to
    the same question — and `intent`'s is the one the product actually reads (`shell.py:560`,
    `tui/search.py:316`, `tui/find.py:233`), so the seam's own predicate had no callers at all."""
    import inspect

    from manyruns import intent

    assert intent.llm_available() == agents.available()
    assert "agents.available" in inspect.getsource(intent.llm_available)
    assert "find_spec" not in inspect.getsource(intent.llm_available), (
        "the second predicate grew its own logic back")


def test_injected_adapter_reused_on_the_callers_loop():
    class LoopBoundIntelligence:
        loop = None
        calls = 0

        async def chat(self, *_args, **_kwargs):
            loop = asyncio.get_running_loop()
            if self.loop is None:
                self.loop = loop
                self.pending = loop.create_future()
                loop.call_soon(self.pending.set_result, "ready")
            assert self.loop is loop and not self.loop.is_closed()
            assert await self.pending == "ready"
            self.calls += 1
            return {"tool_calls": [{"name": "choose", "arguments": {"call": self.calls}}]}

    async def reuse():
        adapter = LoopBoundIntelligence()
        for number in (1, 2):
            assert await agents.call_tool_async(
                system="s", prompt="p", name="choose", description="d", schema={},
                adapter=adapter) == {"call": number}
        assert adapter.loop is asyncio.get_running_loop()
        assert not adapter.loop.is_closed()  # caller still owns client cleanup on this loop

    asyncio.run(reuse())
