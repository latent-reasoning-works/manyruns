"""The one place manyruns asks a model a question.

**One tool call. Never a loop.** Every caller here wants the same thing: a constrained
answer from a fixed set of legal options, decided by the shell and validated on the way back.
`intent.route_recipe` picks one recipe from the ones the data admits; `tune` picks one parameter
of the step in front of it. Neither hands the model control of what happens next.

THAT IS A DELIBERATE LIMIT, not an oversight. manyAgents owns a provider-agnostic agentic
tool-loop (`manyagents/agent_loop.py`) and manyruns does NOT call it: a loop that chooses
manyruns's next step IS the search/planner layer, and CLAUDE.md defers that until there is a
second real tool and real data — "a general tool registry ahead of a second tool is the
op-registry mistake". When that layer arrives it should use that loop rather than grow one here.

TWO BACKENDS, ONE SEAM, and the preference is manyAgents:

  1. `manyagents.adapters.claude_adapter.ClaudeAdapter` — the house adapter, which is where
     observability and provider-routing live and where a second provider would be added once.
  2. a direct `anthropic` call — what `intent.py` did before this module existed. Kept as the
     fallback rather than deleted, because the `[agents]` extra installs both but an environment
     with only the SDK used to work and must not stop working.

A SECOND PAIR OF BACKENDS, FOR A DIFFERENT QUESTION. `answer` asks a LOCAL model for prose
about the run in front of the person — manyAgents' `OllamaAdapter`, else a direct `urllib` POST
to the same OpenAI-compatible endpoint. It is STILL ONE CALL: the limit this file defends is the
loop, not the tool, and a call with no tool at all is further from a loop rather than nearer to
one. It keeps its own model (`ASK_MODEL`), its own availability (`ask_available`) and its own
outage, because a local model needs no key and a stopped `ollama` is not a missing
`ANTHROPIC_API_KEY`.

EVERYTHING IS GUARDED. A missing package, a missing key, a network failure, a malformed reply,
or a schema the model ignored all resolve to `None`, and every caller treats `None` as "the rule
tier decides". The interactive shell must never fail because an optional model call did.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from manyruns import env as _env

#: The model these calls use. One place, so the two callers cannot drift onto different models
#: and produce answers that are not comparable to each other.
MODEL = "claude-opus-4-8"

#: Small, because every call here returns one short forced tool call and nothing else.
MAX_TOKENS = 256


def available() -> bool:
    """Could a model call succeed at all? — cheap, no import of the backend, no network.

    BOTH HALVES, because the first draft checked the KEY ALONE and therefore answered its own
    question wrong: measured with `ANTHROPIC_API_KEY=sk-fake` and neither package installed,
    `available()` was True while `choose_one(...)` returned None. It reported the tier was on in
    exactly the case where nothing can work.

    The key is not a proxy for intent either, so "the user set it, so they want it" does not
    rescue it: `app._load_dotenv()` reads a repo-local `.env` at startup, and
    `harness/__main__.py` writes the prompted key into `os.environ` — so a stale file or an
    earlier session sets it without anyone asking for a model call now.

    `find_spec` rather than an import: this is called to render a menu, and importing the backend
    to answer "is the backend there" would pay the eager adapter-registry cost (numpy, pandas,
    nine adapter modules) just to draw a row. Either backend suffices — the seam falls through
    from one to the other, so the tier is on if EITHER can run.
    """
    import importlib.util

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    return any(importlib.util.find_spec(m) is not None
               for m in ("manyagents", "anthropic"))


def _tool(name: str, description: str, schema: dict) -> dict:
    """One strict tool definition, carrying the schema under BOTH keys the two backends read.

    THE SCHEMA IS THE CONSTRAINT, and it was silently dropped on the preferred backend until this
    carried `parameters` too. `ClaudeAdapter._to_anthropic_tools` reads the OPENAI shape —
    `fn.get("parameters", {"type": "object", "properties": {}})` — while the direct SDK reads
    Anthropic's `input_schema`. Sending only `input_schema` meant the default fired.

    Measured against the real converter, extracted from the commit `uv.lock` pins and run over
    this function's own output:

        sent      {"input_schema": {..., "properties": {"recipe": {"enum": [...]}}, "required": [...]}}
        forwarded {"input_schema": {"type": "object", "properties": {}}}

    The enum, `required`, `additionalProperties` and `strict` all gone — so the "forced choice
    over a fixed set" was, on that path, a free-text call. The answer still arrived, from the
    caller-side check rejecting it or from the fallback backend, which is exactly why nothing
    failed loudly and a test that only mocked the seam could not have caught it.

    Both keys, not a branch on which backend is about to run: the tool dict is built before a
    backend is chosen, and a converter that ignores an extra key is the cheaper contract.
    """
    return {"name": name, "description": description, "strict": True,
            "input_schema": schema, "parameters": schema}


def _via_manyagents(system: str, prompt: str, tool: dict) -> Optional[dict]:
    """The house adapter. `ClaudeAdapter.chat` is ASYNC and returns the OpenAI-shaped turn
    `{"message", "tool_calls": [{id, name, arguments}], "content"}`.

    **Runs the coroutine only when there is no event loop already running.** `asyncio.run` raises
    inside one, and manyruns calls this from two places: the REPL (no loop) and the tune loop on
    `RunScreen`'s worker thread (also no loop — the loop is on the UI thread). If a future caller
    asks from inside a loop, this returns None and the direct backend answers instead, rather than
    raising into a surface that cannot show it.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass                      # no loop on this thread: safe to drive one
    else:
        return None               # already inside one: let the sync backend handle it

    from manyagents.adapters.claude_adapter import ClaudeAdapter

    return asyncio.run(_chat_tool(ClaudeAdapter(), system, prompt, tool, model=MODEL))


async def _chat_tool(adapter, system: str, prompt: str, tool: dict, **kwargs) -> Optional[dict]:
    # manyagents' provider-neutral chat vocabulary is OpenAI-shaped. Each adapter
    # translates it, and an injected adapter owns its model default.
    chat_tool = {"type": "function", "function": {
        key: tool[key] for key in ("name", "description", "parameters", "strict")}}

    turn = await adapter.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tools=[chat_tool], max_tokens=MAX_TOKENS, **kwargs)
    for call in (turn or {}).get("tool_calls") or []:
        if call.get("name") == tool["name"]:
            args = call.get("arguments")
            if isinstance(args, str):
                import json

                args = json.loads(args)
            return args if isinstance(args, dict) else None
    return None


def _via_anthropic(system: str, prompt: str, tool: dict) -> Optional[dict]:
    """The direct SDK call `intent.py` made before this module existed, moved here unchanged in
    behaviour. The fallback, not the default: an install with the SDK and no manyAgents worked
    before and still does."""
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=MODEL, max_tokens=MAX_TOKENS, system=system, tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": prompt}])
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None


def call_tool(*, system: str, prompt: str, name: str, description: str,
              schema: dict) -> Optional[dict]:
    """One tool call. Returns its arguments, or `None` on any failure whatsoever.

    **ONLY THE SDK BACKEND CAN FORCE THE CALL**, and the docstring used to claim both did.
    `ClaudeAdapter.chat` builds its request from `model`, `messages`, `max_tokens`, `system` and
    `tools` — measured, there is no `tool_choice` parameter anywhere in that file — so on that
    path the model MAY answer with prose instead of calling the tool. `_via_anthropic` passes
    `tool_choice={"type": "tool", ...}` and cannot.

    That is survivable rather than fatal, and the ordering below is why: a manyAgents turn with no
    matching tool call yields None, the loop moves on, and the SDK answers the same question with
    the call forced. The unforced backend degrades into the forced one instead of into a guess.

    `schema` is a JSON Schema for the tool's input and is the CONSTRAINT, not a hint: callers
    build it from what is legal right now — the recipes this data admits, the params this step
    declares — so the model cannot name something that does not exist. Validating the reply
    against reality is still the caller's job, because a schema the model ignored is exactly the
    case this returns `None` for and a caller that trusted it would be the bug.

    Tries manyAgents first, then the direct SDK. Both are attempted because a failure in the
    first is usually "not installed", which the second may not share.

    For an injected adapter use ``await call_tool_async(...)``. Its caller owns the
    event loop and client lifetime; this synchronous fallback API owns one-shot backends.
    """
    tool = _tool(name, description, schema)
    for backend in (_via_manyagents, _via_anthropic):
        try:
            out = backend(system, prompt, tool)
        except Exception:  # noqa: BLE001 - an optional tier must never break the shell
            continue
        if isinstance(out, dict) and out:
            return out
    return None


async def call_tool_async(*, system: str, prompt: str, name: str, description: str,
                          schema: dict, adapter) -> Optional[dict]:
    """Ask an injected adapter on the caller's running loop, returning arguments or None.

    Reuse the adapter for consecutive awaited calls on that loop, then close its client
    before the caller closes the loop. Synchronous drivers can own an ``asyncio.Runner``
    around the entire adapter lifetime. No loop is created or closed here, no model is
    overridden, and an outage never sends evidence to another provider.
    """
    try:
        out = await _chat_tool(adapter, system, prompt, _tool(name, description, schema))
    except Exception:  # noqa: BLE001 - optional intelligence must not break the caller
        return None
    return out if isinstance(out, dict) and out else None


def choose_one(*, system: str, prompt: str, key: str, choices: list) -> Optional[str]:
    """A forced pick of exactly one of `choices` — the shape `intent.route_recipe` needs.

    Returns None for an empty choice set, because a forced choice among nothing is a call worth
    not making, and None when the model answers with something outside the set — the enum makes
    that unlikely and the check makes it impossible.
    """
    if not choices:
        return None
    got = call_tool(
        system=system, prompt=prompt, name="choose",
        description="Pick the single option that best matches the request.",
        schema={"type": "object",
                "properties": {key: {"type": "string", "enum": list(choices)}},
                "required": [key], "additionalProperties": False})
    picked = (got or {}).get(key)
    return picked if picked in choices else None


# ── the ask: prose from a local model, one call, still never a loop ───────────────────────────

#: The model the ask uses, and deliberately NOT `MODEL` above: the two are different kinds of
#: call. The tool calls are constrained, remote, rare and comparable to each other; the ask is
#: prose about one run, local, on screen, and answered while a person waits. Measured on the demo
#: record (147 tokens, thinking off): qwen3:8b 1.5 s, qwen3:14b 3.4 s, qwen3:32b 29 s — all three
#: named the right reason, and only the first answers in the time it takes to read the question
#: back. Through this seam, on an M-series laptop with the server already up: 2.4 s for the first
#: call and 0.7 s for the second, byte-identical answers both times. Overridable with
#: `$MANYRUNS_ASK_MODEL`; a wrong name reads as unavailable.
ASK_MODEL = "qwen3:8b"

#: manyAgents' `OllamaAdapter.DEFAULT_BASE_URL`, repeated rather than imported because the whole
#: point of the direct backend is to run with manyAgents absent.
ASK_BASE_URL = "http://localhost:11434/v1"

#: Room for the three sentences the pane can show, and a ceiling on a model that decides to write
#: an essay. The measured demo answer was 238 characters.
ASK_MAX_TOKENS = 400

#: A bound on a hung socket, not a limit on a slow answer: warm is 1.5 s and the slowest measured
#: call — the first after a model swap, with thinking left on — was 9.1 s.
ASK_TIMEOUT_S = 60.0

#: The availability probe is drawn into a row, so it must cost nothing on a laptop with nothing
#: listening: a refused TCP connect on loopback returns in well under this.
ASK_PROBE_S = 0.1


def ask_model() -> str:
    """The model name the ask will actually use.

    Public because the screen prints it beside the answer and the event records it, and a name
    shown that is not the name called would be worse than showing none. An empty override means
    "no opinion" here rather than "off" — unlike `$MANYRUNS_INLINE_IMAGES`, an empty model name is
    not a value the endpoint can do anything with.
    """
    return _env.get("ASK_MODEL") or ASK_MODEL


def _ask_base_url() -> str:
    """The OpenAI-compatible endpoint, read from manyAgents' OWN variable and not through
    `env.get`.

    That looks like a violation of "every override is `$MANYRUNS_*`" and is the opposite: the
    endpoint belongs to manyAgents (`OllamaAdapter.__init__` reads `$OLLAMA_BASE_URL`), so a
    second name for it here would mean a person who moved their ollama could move only one of the
    two backends and would get different answers depending on which one ran.
    """
    return os.environ.get("OLLAMA_BASE_URL") or ASK_BASE_URL


def _completions_url(base: str) -> str:
    """`<base>/v1/chat/completions`, tolerating a base that already carries the `/v1`.

    Both spellings are in circulation and both are correct: manyAgents' default ends in `/v1`
    (the OpenAI client appends the path itself), while a hand-set `$OLLAMA_BASE_URL` usually does
    not. Appending blindly gives `/v1/v1/chat/completions` and a 404 that reads as "the model is
    broken".
    """
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def ask_available() -> bool:
    """Is there a local model to ask? — one TCP connect, no import of a backend, no model call.

    NOT `available()`, and the difference is the point: that one requires `ANTHROPIC_API_KEY`,
    which a model running on this laptop has no use for. Reusing it would have hidden the ask
    behind a key it never sends.

    `available()`'s second half — `find_spec` for a backend — has no counterpart here, and the
    honest version of it is a constant. The fallback backend's dependency is `urllib`, so "no
    backend installed" is not a reachable state and the branch could never be taken; what can
    actually be missing is the SERVER, which both backends need equally. So the probe is the whole
    predicate.

    A connect rather than a request: this is consulted to draw a row, and `/api/tags` is a real
    round-trip against a server that may be mid-generation, while a refused connect on loopback
    returns before the timeout is worth naming.

    **IT RETURNS FALSE FOR A MALFORMED ENDPOINT AND DOES NOT RAISE**, and the parsing is inside
    the `try` for that reason rather than by style. `urlsplit` is lazy — it accepts
    `http://localhost:notaport/v1` and raises `ValueError: Port could not be cast to integer` at
    `.port`, which is the attribute access, not the parse. That used to sit outside the guard, so
    one wrong character in a shell profile turned a *predicate* into an exception; measured live,
    it came out of a Textual thread worker as `WorkerFailed` and took the app down, and the
    screen's own belt (`tui/run.py`'s `_ask_worker`) is what caught it. This function is now
    consulted from the EVENT LOOP as well — the controller row asks it whether to say
    `unavailable — start ollama` — where no belt covers it, so the fix belongs here: a question
    that can be asked to draw a row must have an answer for every value of `$OLLAMA_BASE_URL`,
    and for a port that is not a number the answer is "no, there is nothing to ask".
    """
    import socket
    from urllib.parse import urlsplit

    try:
        parsed = urlsplit(_ask_base_url())
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=ASK_PROBE_S):
            return True
    except (OSError, ValueError):  # refused, unroutable, too slow to wait for — or unparseable
        return False


def _ask_via_manyagents(system: str, prompt: str, model: str) -> Optional[str]:
    """The house adapter again, pointed at the local server. `OllamaAdapter.chat` is ASYNC and
    returns the same OpenAI-shaped turn `{"message", "tool_calls", "content"}` — no tools sent, so
    only `content` is read.

    The event-loop guard is `_via_manyagents`'s, for its reason: the ask runs on a Textual worker
    thread, which has no loop, and a future caller that does have one falls through to the direct
    backend rather than raising into a screen.

    `reasoning_effort` is the load-bearing argument, so a `TypeError` from an adapter that predates
    it means FALL THROUGH, not retry without it: Qwen3 through the OpenAI-compatible endpoint puts
    its thinking in a separate field and leaves `content` EMPTY until the budget is spent —
    measured 9.1 s and nothing to show, unchanged by `/no_think` in the prompt. The direct backend
    below always sends the field, so falling through is the path that answers.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass                      # no loop on this thread: safe to drive one
    else:
        return None               # already inside one: let the direct backend handle it

    from manyagents.adapters.openai_adapter import OllamaAdapter

    try:
        turn = asyncio.run(OllamaAdapter().chat(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            model=model, temperature=0, max_tokens=ASK_MAX_TOKENS, reasoning_effort="none"))
    except TypeError:             # an adapter from before the port: the direct backend answers
        return None
    content = (turn or {}).get("content")
    return content if isinstance(content, str) else None


def _ask_via_ollama(system: str, prompt: str, model: str) -> Optional[str]:
    """One `urllib` POST to the same endpoint — the path that runs in an install with no
    manyAgents, which is this checkout and CI.

    `reasoning_effort: "none"` is what makes an 8B thinking model return prose at all. Measured on
    qwen3:8b with the same prompt: the default 9.1 s with empty `content`; `{"think": false}` in
    the body 4.1 s, still empty; `reasoning_effort` 0.9 s and 238 characters of answer. Without it
    the ask looks like a broken app rather than a slow one.

    `temperature: 0` because the event pins the prompt by hash and a reader will assume the answer
    is a function of it; sampling would make that assumption false in a way no event could show.
    """
    import json
    import urllib.request

    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": ASK_MAX_TOKENS,
        "reasoning_effort": "none",
        "stream": False,
    }).encode()
    request = urllib.request.Request(
        _completions_url(_ask_base_url()), data=body, method="POST",
        # ollama itself needs no key at all; a proxy or a shared box in front of it may, and
        # manyAgents sends this same literal (`OllamaAdapter.__init__`, `api_key="ollama"`).
        headers={"Content-Type": "application/json", "Authorization": "Bearer ollama"})

    with urllib.request.urlopen(request, timeout=ASK_TIMEOUT_S) as response:
        payload = json.loads(response.read().decode("utf-8"))
    choice = (payload.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    return content if isinstance(content, str) else None


@dataclass(frozen=True)
class Answer:
    """The text and the route that actually answered, rather than the route we hoped to use."""

    text: str
    model: str
    endpoint: str
    client: str
    fallback: bool
    backend: str = "ollama"

    @property
    def attribution(self) -> str:
        route = f"{self.model} · {self.backend} · {self.client}"
        return route + (" · fallback from manyagents/OllamaAdapter" if self.fallback else "")


def answer(*, system: str, prompt: str) -> Optional[Answer]:
    """One call, attributed text back, no tool and no loop. `None` on every failure.

    The whole record is in `prompt` — that is the design, not a shortcut: a single run's record is
    a few hundred tokens, so there is nothing in it for a query tool to select among, and a model
    handed the whole page gives an answer that is a function of a page a later reader can
    reconstruct. A tool loop would buy no capability at this size and would cost the one thing a
    corpus cannot recover: what the model actually looked at.

    Empty content is a failure, not an answer. A thinking model with its budget spent returns
    `""` (see `_ask_via_manyagents`), and painting that into the pane would report an outage as a
    reply. The caller says "unavailable" instead, and emits nothing.

    manyAgents first, then the direct POST, for `call_tool`'s reason: a failure in the first is
    usually "not installed", which the second does not share.
    """
    try:
        model, endpoint = ask_model(), _ask_base_url()
        for backend, client, fallback in (
            (_ask_via_manyagents, "manyagents/OllamaAdapter", False),
            (_ask_via_ollama, "urllib", True),
        ):
            try:
                out = backend(system, prompt, model)
            except Exception:  # an optional tier must never break the screen
                continue
            if isinstance(out, str) and out.strip():
                return Answer(out.strip(), model, endpoint, client, fallback)
    except Exception:  # configuration failures share the unavailable contract too
        return None
    return None
