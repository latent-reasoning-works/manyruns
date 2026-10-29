"""The driver socket: where an outside program attaches to a running session.

**manyruns never calls a driver** (environment contract §3.5). This listens; a driver connects,
as it would run the CLI. It names no caller and is the same for every program. Every request is
decided by `manyruns.driver.handle`; this module only carries requests in and effects out.

ONE DRIVER AT A TIME. The screen has one wheel, and two programs taking turns at it would make
the on-air tag a lie about who moved a value. A second connection gets one refusal and is closed.

THE SOCKET LIVES IN A SHORT TEMP FOLDER, and the session folder holds a pointer to it
(`driver.json`). A Unix socket path is capped at 104 bytes on macOS (`sun_path`), and a session
folder under a project path is routinely longer than that, so the socket cannot sit beside the
run it belongs to. The pointer can, and it is how a driver finds the session it means.

**Every wait is bounded and every thread is a daemon**, for the reason `tui/run.py`'s `_drive`
documents at length: a non-daemon thread holds the terminal hostage after quit. `close()` shuts
the listener and the client, and unblocks both threads.

Callbacks (`on_effect`, `on_control`) run on this module's threads. The screen passes
`post_message`, which is safe from any thread.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from manyruns import driver

#: The pointer file in the session folder.
POINTER = "driver.json"

#: The longest request line accepted. A step offer is a few hundred bytes; a megabyte line is a
#: driver that has lost the protocol, and buffering it would let one client exhaust memory.
MAX_LINE = 1 << 20


class DriverSocket:
    """A listening socket, the control state, and the current step offer."""

    def __init__(self, folder: Path | str, *, tool_version: str,
                 on_effect: Callable[[dict], None] = lambda effect: None,
                 on_control: Callable[[dict], None] = lambda ctrl: None,
                 grant: str = "interpret") -> None:
        # THE LAUNCH GRANT: the level a driver gets the moment it attaches. `interpret` unless
        # whoever STARTED the run said otherwise, which is how an unattended run (nobody at the
        # keys to press ctrl+g) can be driven at all. It is still a person's decision, made
        # before the run instead of during it; no request can raise a level past it.
        if grant not in driver.LEVELS:
            raise ValueError(f"grant must be one of {list(driver.LEVELS)}, not {grant!r}")
        self.grant = grant
        self.folder = Path(folder)
        self._on_effect = on_effect
        self._on_control = on_control
        self.tool_version = tool_version
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self.ctrl: dict = driver.control()
        self.offer: Optional[dict] = None
        self.path: str = ""
        self._dir: Optional[str] = None
        self._server: Optional[socket.socket] = None
        self._client: Optional[socket.socket] = None
        self._closed = False

    def bind(self, *, on_effect: Callable[[dict], None],
             on_control: Callable[[dict], None]) -> "DriverSocket":
        """Attach the screen's callbacks after construction, as `Gate.bind` does and for its
        reason: the seat is built beside the gate, before the screen that receives its effects."""
        self._on_effect = on_effect
        self._on_control = on_control
        return self

    # ── life cycle ───────────────────────────────────────────────────────────
    def start(self) -> str:
        """Listen, write the pointer, and return the socket path."""
        self._dir = tempfile.mkdtemp(prefix="mr-", dir="/tmp" if os.path.isdir("/tmp") else None)
        self.path = os.path.join(self._dir, "d.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self.path)
        os.chmod(self.path, 0o600)  # this user's programs only
        server.listen(2)
        server.settimeout(0.5)
        self._server = server
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / POINTER).write_text(json.dumps(
            {"socket": self.path, "pid": os.getpid(), "tool_version": self.tool_version,
             "protocol_version": driver.PROTOCOL_VERSION}))
        threading.Thread(target=self._accept, name="manyruns-driver-accept", daemon=True).start()
        return self.path

    def close(self) -> None:
        """Stop listening, hang up on the driver, and remove the socket and the pointer."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            client, self._client = self._client, None
            server, self._server = self._server, None
        for s in (client, server):
            if s is not None:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()
        for p in (self.path, str(self.folder / POINTER)):
            try:
                os.unlink(p)
            except (FileNotFoundError, OSError):
                pass
        if self._dir:
            try:
                os.rmdir(self._dir)
            except OSError:
                pass

    # ── what the screen calls ────────────────────────────────────────────────
    def set_offer(self, step: dict, constraints: dict, question_id: Optional[int]) -> None:
        """The tune loop is asking about `step`: this is now the only offer a choice may match."""
        offer = driver.step_surface(step, constraints, question_id, self.tool_version)
        with self._lock:
            self.offer = offer
        self.emit({"event": "offer", "offer": offer})

    def clear_offer(self) -> None:
        """Nothing is waiting for a choice."""
        with self._lock:
            self.offer = None
        self.emit({"event": "offer", "offer": None})

    def cycle(self) -> dict:
        """The scientist's level key."""
        return self._set_ctrl(driver.cycle)

    def handbrake(self) -> dict:
        """The scientist's manual/auto toggle (`driver.handbrake`)."""
        return self._set_ctrl(driver.handbrake)

    def emit(self, event: dict) -> None:
        """Push one event to the attached driver, if any. Never raises."""
        with self._lock:
            client = self._client
        if client is not None:
            self._send(client, event)

    # ── internals ────────────────────────────────────────────────────────────
    def _set_ctrl(self, change: Callable[[dict], dict]) -> dict:
        with self._lock:
            self.ctrl = change(self.ctrl)
            ctrl = dict(self.ctrl)
        self._on_control(ctrl)
        self.emit({"event": "control", "level": ctrl["level"], "tag": driver.tag(ctrl),
                   "braked": ctrl["braked"], "resume": ctrl.get("resume")})
        return ctrl

    def _send(self, client: socket.socket, obj: dict) -> None:
        data = (json.dumps(obj, default=str) + "\n").encode("utf-8")
        try:
            with self._send_lock:
                client.sendall(data)
        except OSError:
            pass  # the driver hung up; the reader thread notices and detaches

    def _accept(self) -> None:
        while True:
            with self._lock:
                server = self._server
            if server is None:
                return
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                busy = self._client is not None or self._closed
                if not busy:
                    self._client = conn
            if busy:
                self._send(conn, {"ok": False, "error": "a driver is already attached to this "
                                                        "session; one at a time"})
                conn.close()
                continue
            threading.Thread(target=self._serve, args=(conn,), name="manyruns-driver",
                             daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        attached = False
        buf = b""
        try:
            while True:
                try:
                    chunk = conn.recv(65536)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                if len(buf) > MAX_LINE and b"\n" not in buf:
                    self._send(conn, {"ok": False, "error": "request line too long"})
                    break
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    if raw.strip():
                        reply, attached = self._request(raw, attached)
                        self._send(conn, reply)
        finally:
            with self._lock:
                mine = self._client is conn
                if mine:
                    self._client = None
            conn.close()
            if mine and attached:
                self._set_ctrl(driver.detach)

    def _request(self, raw: bytes, attached: bool) -> tuple[dict, bool]:
        try:
            request: Any = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            return {"ok": False, "error": f"not JSON: {error}"}, attached
        verb = request.get("verb") if isinstance(request, dict) else None
        if verb == "hello":
            if attached:
                return {"ok": False, "error": "already attached; hello once"}, attached
            try:
                with self._lock:
                    self.ctrl = {**driver.attach(self.ctrl, request.get("chooser"),
                                                 request.get("bound")), "level": self.grant}
                    ctrl = dict(self.ctrl)
            except ValueError as error:
                return {"ok": False, "error": str(error)}, attached
            self._on_control(ctrl)
            with self._lock:
                offer = self.offer
            # THE CURRENT OFFER RIDES ON THE REPLY. Events are not replayed, and a driver that
            # attaches while a question is already waiting would otherwise never see it.
            return {"ok": True, "protocol_version": driver.PROTOCOL_VERSION,
                    "level": ctrl["level"], "tag": driver.tag(ctrl),
                    "verbs": sorted(driver.VERBS[ctrl["level"]]), "offer": offer}, True
        if not attached:
            return {"ok": False, "error": "send `hello` with your `chooser` and `bound` first"}, attached
        with self._lock:
            ctrl, offer = dict(self.ctrl), self.offer
        reply, effect = driver.handle(request, ctrl, offer)
        if effect is not None:
            self._on_effect(effect)
        return reply, attached
