"""`agents.answer` / `agents.ask_available` — the ask, against a stub that is not a model.

WHAT IS TESTED IS THE WIRE AND THE GUARDS, because that is all this seam owns. Whether qwen3:8b
answers the question well is not a property of this file and cannot be asserted from one; whether
the request said "do not think", whether a stopped server is an outage instead of a traceback, and
whether the key the ask never sends can turn it off — those are.

A LOCAL HTTP STUB rather than a monkeypatched `urllib`: the argument this commit turns on is that
`reasoning_effort` reaches the endpoint (without it an 8B thinking model returns empty `content`
and the app looks broken), and a patched sender proves that the code SAID it, not that it was
sent. The stub reads the bytes off the socket.

NOTHING HERE MAY REACH A REAL OLLAMA. `_no_real_ollama` is autouse and points `$OLLAMA_BASE_URL`
at a port nothing is on, so a developer running this on the laptop the spec was measured on — one
with the server up and qwen3:8b pulled — gets the same result as CI rather than a 2.4 s network
call and a different answer every release.
"""
from __future__ import annotations

import http.server
import json
import socket
import threading

import pytest

from manyruns import agents


def _completion(content: str) -> dict:
    """A full OpenAI-shaped chat completion, spelled out rather than minimised: the manyAgents
    backend parses this with the `openai` client's own models, so a body trimmed to the two keys
    manyruns reads would pass here and fail on an install that has the adapter."""
    return {"id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": "qwen3:8b",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 147, "completion_tokens": 42, "total_tokens": 189}}


class _Endpoint(http.server.BaseHTTPRequestHandler):
    """One POST, recorded and answered. The recording is the point — see `self.server.requests`."""

    def do_POST(self) -> None:                                    # noqa: N802 - stdlib's name
        length = int(self.headers.get("Content-Length", 0))
        self.server.requests.append((self.path, json.loads(self.rfile.read(length))))
        payload = json.dumps(self.server.completion).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args) -> None:
        """stdlib logs every request to stderr; the test output is the test's own."""


def _closed_port() -> int:
    """A port nothing is listening on — bind, read the number, hand it back closed."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(autouse=True)
def _no_real_ollama(monkeypatch):
    """The default for every test in this file: an endpoint that is not there."""
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{_closed_port()}/v1")


@pytest.fixture
def ollama(monkeypatch, _no_real_ollama):
    """A server on an ephemeral port, pointed at by manyAgents' own `$OLLAMA_BASE_URL`.

    Naming the autouse fixture is what orders the two `setenv` calls: this one must win.
    """
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    server.requests = []
    server.completion = _completion(
        "No — 19,024 of 32,738 genes appear in fewer than 3 cells, so the graph is mostly noise.")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_the_answer_is_the_text_the_endpoint_returned(ollama):
    """Plain text out of a call with no tool in it — the one shape the pane can paint."""
    got = agents.answer(system="you answer questions about one run", prompt="trust leiden here?")

    assert got.text == ollama.completion["choices"][0]["message"]["content"]
    assert len(ollama.requests) == 1, "one call, never a loop"


def test_the_request_carries_the_instruction_not_to_think(ollama):
    """THE LOAD-BEARING FIELD, asserted off the socket rather than off the code.

    Measured on qwen3:8b with the demo's 147-token record: the default call takes 9.1 s and comes
    back with EMPTY `content` (the thinking lands in a separate field and the budget is spent
    before prose starts), unchanged by `/no_think` in the prompt; `{"think": false}` in the body is
    4.1 s and still empty; `reasoning_effort: "none"` is 0.9 s and 238 characters of answer. Drop
    this field and every ask on an 8B model returns nothing and reads as a broken app.
    """
    agents.answer(system="a system line", prompt="a question")

    path, body = ollama.requests[0]
    assert path.endswith("/v1/chat/completions"), "manyAgents' base already carries the /v1"
    assert body["reasoning_effort"] == "none"
    assert body["model"] == "qwen3:8b"
    assert body["temperature"] == 0, "the event pins the prompt by hash; sampling would unpin it"
    assert [m["content"] for m in body["messages"]] == ["a system line", "a question"]


def test_nothing_listening_answers_none_rather_than_raising():
    """The contract every caller relies on. The screen shows a sentence about the server being
    down; it does not show a traceback, and the app does not fail because an optional call did."""
    assert agents.answer(system="s", prompt="p") is None


def test_an_empty_reply_is_an_outage_not_an_answer(ollama):
    """A thinking model with its budget spent returns `""`. Painting that into the pane would
    report an outage as a reply, so it resolves to None like any other failure."""
    ollama.completion = _completion("")

    assert agents.answer(system="s", prompt="p") is None


def test_availability_is_the_server_being_there(ollama):
    """True with the stub up, False against the port it was on once it is gone — availability is
    a live fact, not a fact about the install."""
    assert agents.ask_available() is True

    ollama.shutdown()
    ollama.server_close()
    assert agents.ask_available() is False


def test_nothing_listening_reads_as_unavailable():
    assert agents.ask_available() is False


@pytest.mark.parametrize("endpoint", [
    "http://localhost:notaport/v1",       # the measured one: a typo in a shell profile
    "http://localhost:99999/v1",          # a port number that is not a port
    "not a url at all",
])
def test_an_endpoint_nobody_can_parse_is_an_answer_and_not_an_exception(monkeypatch, endpoint):
    """A PREDICATE MUST HAVE AN ANSWER FOR EVERY VALUE OF `$OLLAMA_BASE_URL`.

    `urlsplit` is lazy: it accepts `http://localhost:notaport/v1` and raises `ValueError: Port
    could not be cast to integer` at `.port`, which is the attribute access rather than the
    parse. That used to sit outside the guard, and the failure was measured rather than imagined
    — the raise came out of a Textual thread worker as `WorkerFailed`, `app.is_running` went
    False and the run screen froze mid-thought, which is `agents.py`'s own contract ("the
    interactive shell must never fail because an optional model call did") broken through the one
    door this file did not guard.

    It matters more now than it did then: the controller row consults this from the EVENT LOOP to
    decide whether to say `unavailable — start ollama`, where the screen's own catch does not
    reach. For an endpoint that cannot be parsed the true answer is "no, there is nothing to
    ask" — the same answer as for a port with nothing on it.
    """
    monkeypatch.setenv("OLLAMA_BASE_URL", endpoint)

    assert agents.ask_available() is False
    assert agents.answer(system="s", prompt="p") is None, "the call itself must not raise either"


def test_neither_half_of_the_ask_consults_the_anthropic_key(ollama, monkeypatch):
    """THE REASON `ask_available` EXISTS AT ALL instead of reusing `available()`.

    A model on this laptop has no use for `ANTHROPIC_API_KEY`, and `available()` returns False
    without one. Gating the ask on that predicate would have hidden a working local model behind a
    key it never sends — so with the key deleted the ask still answers, and the two predicates
    disagree about the same machine, correctly.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert agents.ask_available() is True
    assert agents.answer(system="s", prompt="p")
    assert agents.available() is False, "the tool-call tier is off; the ask is not"


def test_the_model_is_overridable_through_the_product_prefix(ollama, monkeypatch):
    """`$MANYRUNS_ASK_MODEL`, read through `env.get` like every other override here — so a
    practitioner with a 14B they prefer, or a machine where the 8B is not pulled, changes one
    variable. The endpoint keeps manyAgents' name; the model is manyruns' own choice."""
    monkeypatch.setenv("MANYRUNS_ASK_MODEL", "qwen3:14b")

    agents.answer(system="s", prompt="p")

    assert agents.ask_model() == "qwen3:14b"
    assert ollama.requests[0][1]["model"] == "qwen3:14b"
