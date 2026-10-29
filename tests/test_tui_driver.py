"""A driver attached to a live run screen, end to end: the tag, the levels, the handbrake.

A real `RunScreen` runs the real `tune.run_tune_loop` on the mock engine, and a real Unix-socket
client plays the driver. What is under test is who may move a value, what the screen says about
it, and what the decision row records; the fit itself is the mock's.
"""
from __future__ import annotations

import asyncio
import json
import socket

from manyruns import app as _app
from manyruns import decisions, tune
from manyruns.session import Session
from manyruns.tui.driver_socket import DriverSocket
from manyruns.tui.gate import Gate
from manyruns.tui.run import RunScreen
from test_tui_run import _Harness, _screen_text, drives


class Client:
    def __init__(self, path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(path)
        self.buf = b""
        self.events: list[dict] = []

    def _line(self) -> dict:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        raw, self.buf = self.buf.split(b"\n", 1)
        return json.loads(raw)

    def ask(self, obj: dict) -> dict:
        self.sock.sendall((json.dumps(obj) + "\n").encode())
        while True:
            msg = self._line()
            if "event" in msg:
                self.events.append(msg)
                continue
            return msg

    def close(self) -> None:
        self.sock.close()


def _screen(tmp_path):
    recipe = _app.load_recipe("cflows")
    gate = Gate()
    seat = DriverSocket(tmp_path / "session", tool_version="test-build")
    holder: dict = {}

    def start(_on_step):
        session = Session(project="p", engine="mock", modality="scrna", recipe=recipe,
                          out_dir=tmp_path / "session")
        holder["session"] = session
        step = tune.parse_step_line("phate knn=5", session.engine)
        tune.run_tune_loop(session, step, gate.read, gate.write)
        return session.close()

    seat.start()
    screen = RunScreen(recipe, start=start, gate=gate, seat=seat)
    return screen, gate, seat, holder


async def _until(pilot, cond, what, tries=600):
    for _ in range(tries):
        if cond():
            return
        await pilot.pause()
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


async def _question(pilot, gate, seat):
    await _until(pilot, lambda: gate.prompt is not None and seat.offer is not None
                 and seat.offer.get("question_id") == gate.question_id, "a question with an offer")
    return seat.offer


def _tune_rows(holder):
    return [r for r in decisions.read(tune._corpus_dir(holder["session"]))
            if r.get("surface") == "tune"]


@drives
async def test_a_screen_without_a_seat_shows_no_tag_and_the_keys_do_nothing(tmp_path):
    screen = RunScreen(_app.load_recipe("cflows"))
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert app.screen.query_one("#on-air").display is False
        await pilot.press("ctrl+g")
        await pilot.press("ctrl+b")
        await pilot.pause()
        assert "ON AIR" not in _screen_text(app)


@drives
async def test_the_tag_says_human_until_the_scientist_hands_over_the_wheel(tmp_path):
    screen, gate, seat, holder = _screen(tmp_path)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        c = Client(seat.path)
        assert c.ask({"verb": "hello", "chooser": "llm:qwen3:1.7b", "bound": "schema"})["ok"]
        await _until(pilot, lambda: screen.ctrl.get("chooser"), "the attach to reach the screen")
        text = _screen_text(app)
        assert "HUMAN" in text and "llm:qwen3:1.7b attached · interpret" in text
        await pilot.press("ctrl+g")
        await pilot.press("ctrl+g")
        await _until(pilot, lambda: screen.ctrl.get("level") == "drive", "the drive level")
        assert "ON AIR · MACHINE · llm:qwen3:1.7b" in _screen_text(app)
        c.close()
        await _until(pilot, lambda: not screen.ctrl.get("chooser"), "the hang-up")
        assert "ON AIR" not in _screen_text(app)
        gate.close()


@drives
async def test_a_driving_driver_tunes_and_accepts_and_the_row_is_its(tmp_path):
    screen, gate, seat, holder = _screen(tmp_path)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        c = Client(seat.path)
        c.ask({"verb": "hello", "chooser": "llm:qwen3:1.7b", "bound": "schema"})
        await _until(pilot, lambda: screen.ctrl.get("chooser"), "the attach")
        await pilot.press("ctrl+g")
        await pilot.press("ctrl+g")
        await _until(pilot, lambda: screen.ctrl.get("level") == "drive", "the drive level")

        offer = await _question(pilot, gate, seat)
        assert c.ask({"verb": "apply", "legal_id": offer["legal_id"], "params": {"t": 20}})["ok"]
        await _until(pilot, lambda: gate.question_id > offer["question_id"]
                     and gate.prompt is not None, "the loop to fit and ask again")

        offer = await _question(pilot, gate, seat)
        assert offer["params"]["t"]["current"] == 20  # the driver's value is the current attempt
        assert c.ask({"verb": "apply", "legal_id": offer["legal_id"], "decision": "accept"})["ok"]
        await _until(pilot, lambda: app.finished, "the run to finish")
        c.close()

    row = _tune_rows(holder)[-1]
    assert row["schema_version"] == 2 and row["chooser"] == "llm:qwen3:1.7b"
    assert [o.get("by") for o in row["offered"]] == [None, "llm:qwen3:1.7b"]


@drives
async def test_at_propose_the_card_waits_for_the_scientists_key(tmp_path):
    screen, gate, seat, holder = _screen(tmp_path)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        c = Client(seat.path)
        c.ask({"verb": "hello", "chooser": "llm:qwen3:1.7b", "bound": "schema"})
        await _until(pilot, lambda: screen.ctrl.get("chooser"), "the attach")
        await pilot.press("ctrl+g")
        await _until(pilot, lambda: screen.ctrl.get("level") == "propose", "the propose level")

        offer = await _question(pilot, gate, seat)
        apply = c.ask({"verb": "apply", "legal_id": offer["legal_id"], "params": {"t": 20}})
        assert apply["ok"] is False  # propose may not apply
        reply = c.ask({"verb": "propose", "legal_id": offer["legal_id"], "params": {"t": 20},
                       "rationale": "a longer diffusion draws the trajectory as one line"})
        assert reply["ok"]
        await _until(pilot, lambda: screen.proposal is not None, "the card")
        assert "proposes t=20" in _screen_text(app)
        qid = gate.question_id
        await pilot.press("ctrl+y")
        await _until(pilot, lambda: gate.question_id > qid and gate.prompt is not None,
                     "the loop to take the card")
        gate.answer("accept")
        await _until(pilot, lambda: app.finished, "the run to finish")
        c.close()

    row = _tune_rows(holder)[-1]
    assert row["chooser"] == "human" and [o.get("by") for o in row["offered"]] == [None, "human"]
    assert row["proposed"]["chooser"] == "llm:qwen3:1.7b" and row["proposed"]["params"] == {"t": 20}
    assert row["rationale"] == "a longer diffusion draws the trajectory as one line"


@drives
async def test_the_handbrake_takes_the_wheel_back_and_refuses_the_next_apply(tmp_path):
    screen, gate, seat, holder = _screen(tmp_path)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        c = Client(seat.path)
        c.ask({"verb": "hello", "chooser": "llm:qwen3:1.7b", "bound": "schema"})
        await _until(pilot, lambda: screen.ctrl.get("chooser"), "the attach")
        await pilot.press("ctrl+g")
        await pilot.press("ctrl+g")
        await _until(pilot, lambda: screen.ctrl.get("level") == "drive", "the drive level")
        offer = await _question(pilot, gate, seat)

        await pilot.press("ctrl+b")
        await _until(pilot, lambda: screen.ctrl.get("braked"), "the handbrake")
        text = _screen_text(app)
        assert "ON AIR" not in text and "HUMAN" in text and "handbrake on" in text
        reply = c.ask({"verb": "apply", "legal_id": offer["legal_id"], "params": {"t": 20}})
        assert reply["ok"] is False and "handbrake" in reply["error"]
        assert gate.prompt is not None, "the question still waits for the scientist"
        assert offer["params"]["knn"]["current"] == 5  # nothing moved
        gate.answer("accept")
        await _until(pilot, lambda: app.finished, "the run to finish")
        c.close()


@drives
async def test_with_a_driver_attached_a_question_goes_to_it(tmp_path):
    from textual.widgets import Input

    screen, gate, seat, holder = _screen(tmp_path)
    app = _Harness(screen)
    async with app.run_test(size=(140, 40)) as pilot:
        c = Client(seat.path)
        c.ask({"verb": "hello", "chooser": "llm:qwen3:1.7b", "bound": "schema"})
        await _until(pilot, lambda: screen.ctrl.get("chooser"), "the attach")
        await _question(pilot, gate, seat)
        field = app.screen.query_one("#ask-input", Input)
        field.value = "? why does this look like two clusters"
        await pilot.press("enter")
        await pilot.pause()
        assert ("you", "why does this look like two clusters") in screen.conversation
        assert c.ask({"verb": "say", "text": "the diffusion time is short"})["ok"]
        assert any(e.get("event") == "message" and "two clusters" in e.get("text", "")
                   for e in c.events)
        await _until(pilot, lambda: any(w == "llm:qwen3:1.7b" for w, _ in screen.conversation),
                     "the driver's reply on screen")
        assert "the diffusion time is short" in _screen_text(app)
        gate.answer("accept")
        await _until(pilot, lambda: app.finished, "the run to finish")
        c.close()
