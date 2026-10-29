"""The driver socket, over a real Unix socket, with a stand-in for the screen.

The socket only carries requests to `manyruns.driver.handle` and effects back to whoever owns
the screen; these tests pin the carrying: attach, levels the scientist sets, one driver at a
time, events pushed to the driver, and a clean close.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import pytest

from manyruns.tui.driver_socket import DriverSocket

PHATE = {"group": "latent", "name": "phate", "params": {"knn": 5}}


class Client:
    def __init__(self, path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(path)
        self.buf = b""

    def send(self, obj) -> None:
        self.sock.sendall((json.dumps(obj) + "\n").encode())

    def line(self) -> dict:
        while b"\n" not in self.buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        raw, self.buf = self.buf.split(b"\n", 1)
        return json.loads(raw)

    def reply(self) -> dict:
        """The next reply, skipping pushed events (they carry `event`)."""
        while True:
            msg = self.line()
            if "event" not in msg:
                return msg

    def ask(self, obj) -> dict:
        self.send(obj)
        return self.reply()

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def seat(tmp_path):
    effects: list[dict] = []
    controls: list[dict] = []
    s = DriverSocket(tmp_path, on_effect=effects.append, on_control=controls.append,
                     tool_version="test-build")
    s.effects, s.controls = effects, controls
    s.start()
    yield s
    s.close()


def _wait(pred, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_the_session_folder_names_the_socket(seat, tmp_path):
    pointer = json.loads((tmp_path / "driver.json").read_text())
    assert pointer["socket"] == seat.path
    assert len(seat.path.encode()) < 104  # the macOS sun_path limit
    assert Path(seat.path).exists()


def test_hello_attaches_the_driver_at_interpret(seat):
    c = Client(seat.path)
    reply = c.ask({"verb": "hello", "chooser": "llm:qwen3:1.7b", "bound": "schema"})
    assert reply["ok"] is True and reply["level"] == "interpret" and reply["tag"] == "HUMAN"
    assert _wait(lambda: seat.controls and seat.controls[-1]["chooser"] == "llm:qwen3:1.7b")
    c.close()


def test_a_driver_that_arrives_mid_question_gets_the_offer_in_its_hello(seat):
    """Events are not replayed, so a driver that attaches while a question is already waiting
    would otherwise never see the offer for it."""
    seat.set_offer(PHATE, {}, question_id=4)
    c = Client(seat.path)
    reply = c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    assert reply["offer"]["question_id"] == 4 and reply["offer"]["step"] == "phate"
    c.close()


def test_nothing_but_hello_is_allowed_before_hello(seat):
    c = Client(seat.path)
    reply = c.ask({"verb": "say", "text": "hi"})
    assert reply["ok"] is False and "hello" in reply["error"]
    c.close()


def test_a_bad_hello_is_refused_and_attaches_nothing(seat):
    c = Client(seat.path)
    reply = c.ask({"verb": "hello", "chooser": "llm:x", "bound": "telepathy"})
    assert reply["ok"] is False and "bound" in reply["error"]
    assert seat.ctrl["chooser"] is None
    c.close()


def test_a_second_driver_is_turned_away_while_one_is_attached(seat):
    a = Client(seat.path)
    a.ask({"verb": "hello", "chooser": "llm:a", "bound": "schema"})
    b = Client(seat.path)
    reply = b.reply()
    assert reply["ok"] is False and "already attached" in reply["error"]
    b.close()
    a.close()


def test_say_reaches_the_screen_as_an_effect(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    assert c.ask({"verb": "say", "text": "branches look cleaner"})["ok"] is True
    assert _wait(lambda: any(e["kind"] == "say" for e in seat.effects))
    c.close()


def test_the_scientist_raises_the_level_and_the_driver_hears_it(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    seat.cycle()
    seat.cycle()
    events = []
    while len(events) < 2:
        msg = c.line()
        if msg.get("event") == "control":
            events.append(msg)
    assert [e["level"] for e in events] == ["propose", "drive"]
    assert events[-1]["tag"] == "MACHINE · llm:x"
    c.close()


def test_apply_runs_only_against_the_current_offer_at_drive(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    seat.set_offer(PHATE, {}, question_id=7)
    offer = c.ask({"verb": "actions"})["offer"]
    req = {"verb": "apply", "legal_id": offer["legal_id"], "params": {"knn": 10}}
    assert c.ask(req)["ok"] is False  # interpret
    seat.cycle()
    seat.cycle()
    assert c.ask(req)["ok"] is True
    assert _wait(lambda: any(e["kind"] == "apply" for e in seat.effects))
    applied = [e for e in seat.effects if e["kind"] == "apply"][-1]
    assert applied["line"] == "knn=10" and applied["question_id"] == 7
    c.close()


def test_an_offer_change_makes_the_old_legal_id_stale(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    seat.cycle()
    seat.cycle()
    seat.set_offer(PHATE, {}, question_id=7)
    old = c.ask({"verb": "actions"})["offer"]["legal_id"]
    seat.set_offer({**PHATE, "params": {"knn": 10}}, {}, question_id=8)
    reply = c.ask({"verb": "apply", "legal_id": old, "params": {"knn": 20}})
    assert reply["ok"] is False and "stale" in reply["error"]
    c.close()


def test_with_no_question_waiting_apply_is_refused(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    seat.cycle()
    seat.cycle()
    seat.set_offer(PHATE, {}, question_id=7)
    legal = c.ask({"verb": "actions"})["offer"]["legal_id"]
    seat.clear_offer()
    reply = c.ask({"verb": "apply", "legal_id": legal, "params": {"knn": 10}})
    assert reply["ok"] is False and "no step" in reply["error"]
    c.close()


def test_the_handbrake_stops_apply_at_once(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    seat.cycle()
    seat.cycle()
    seat.set_offer(PHATE, {}, question_id=7)
    legal = c.ask({"verb": "actions"})["offer"]["legal_id"]
    seat.handbrake()
    reply = c.ask({"verb": "apply", "legal_id": legal, "params": {"knn": 10}})
    assert reply["ok"] is False and "handbrake" in reply["error"]
    assert seat.ctrl["braked"] is True
    c.close()


def test_events_reach_the_driver(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    seat.emit({"event": "message", "text": "more like branches"})
    while True:
        msg = c.line()
        if msg.get("event") == "message":
            break
    assert msg["text"] == "more like branches"
    c.close()


def test_a_driver_hanging_up_hands_control_back(seat):
    c = Client(seat.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    seat.cycle()
    seat.cycle()
    c.close()
    assert _wait(lambda: seat.ctrl["chooser"] is None and seat.ctrl["level"] == "interpret")
    assert seat.controls[-1]["chooser"] is None


def test_garbage_is_refused_and_the_connection_survives(seat):
    c = Client(seat.path)
    c.sock.sendall(b"not json\n")
    assert c.reply()["ok"] is False
    assert c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})["ok"] is True
    c.close()


def test_close_removes_the_socket_and_the_pointer(tmp_path):
    s = DriverSocket(tmp_path, on_effect=lambda e: None, on_control=lambda c: None,
                     tool_version="t")
    s.start()
    path = s.path
    s.close()
    assert not Path(path).exists()
    assert not (tmp_path / "driver.json").exists()
    s.close()  # twice is fine


def test_close_while_a_driver_is_attached_does_not_hang(tmp_path):
    s = DriverSocket(tmp_path, on_effect=lambda e: None, on_control=lambda c: None,
                     tool_version="t")
    s.start()
    c = Client(s.path)
    c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    t = threading.Thread(target=s.close)
    t.start()
    t.join(3)
    assert not t.is_alive()
    c.close()


def test_a_launch_grant_sets_the_level_a_driver_gets_on_attach(tmp_path):
    """Unattended runs: whoever starts the run grants a level up front. Nobody is at the keys
    to raise it, and no request can."""
    s = DriverSocket(tmp_path, tool_version="t", grant="drive")
    s.start()
    c = Client(s.path)
    reply = c.ask({"verb": "hello", "chooser": "llm:x", "bound": "schema"})
    assert reply["level"] == "drive" and reply["tag"] == "MACHINE · llm:x"
    c.close()
    s.close()


def test_an_unknown_grant_is_refused_at_launch(tmp_path):
    with pytest.raises(ValueError, match="grant"):
        DriverSocket(tmp_path, tool_version="t", grant="root")
