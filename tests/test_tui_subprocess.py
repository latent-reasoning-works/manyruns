"""Subprocess-backed algorithms need prewarming before Textual captures stderr.

Reported from a real run of `embed` on `gaussian_blob`:

    ✗ phate  error  1.88s
      ValueError: bad value(s) in fds_to_keep

The stack ends at `multiprocessing/resource_tracker.py:173 ensure_running` →
`util.spawnv_passfds` → `_posixsubprocess.fork_exec`. `ensure_running` passes
`sys.stderr.fileno()` to the child process it spawns, and inside a running Textual app
`sys.stderr` is `textual.app._PrintCapture`, whose `fileno()` returns **-1** — measured, with
`sys.__stderr__.fileno()` still 2. `fork_exec` rejects a negative descriptor, so the first
`multiprocessing.Lock` anything creates raises, and sklearn's `Pool` creates one immediately.

**Narrowed by elimination**, because the record keeps only `f"{type(e).__name__}: {e}"` and the
stack was gone:

    plain script                     phate -> ok
    worker thread, no Textual        phate -> ok
    Textual app on a real tty        ValueError: bad value(s) in fds_to_keep

And confirmed fixed the same way — same app, same fit, on a tty:

    prewarmed=False   ValueError: bad value(s) in fds_to_keep
    prewarmed=True    OK — embedding (300, 3)

`App.run_test` can also replace stderr with `_PrintCapture` (fileno -1). Prewarm order
and captured descriptors are tested here, along with process exit during a pending run.

"""
from __future__ import annotations

import pytest

pytest.importorskip("textual")

from manyruns.tui import app as tui_app  # noqa: E402


def test_the_resource_tracker_is_running_after_a_prewarm():
    """The whole fix. `ensure_running` short-circuits once the tracker is up, so starting it
    while `sys.stderr` is still the terminal's is all that is needed — the app can then replace
    stderr freely and no later `Lock` will try to spawn anything."""
    assert tui_app.prewarm_multiprocessing() is True


def test_prewarming_twice_is_harmless():
    """It is a per-process singleton, and `run()` is not the only thing that may call it."""
    assert tui_app.prewarm_multiprocessing() is True
    assert tui_app.prewarm_multiprocessing() is True


def test_it_happens_before_the_app_takes_the_terminal(monkeypatch):
    """ORDER IS THE ENTIRE FIX. Prewarming after `App.run()` would run against the replaced
    stderr and do nothing — and would look exactly like a fix that worked, because the call
    still succeeds. This asserts the sequence rather than the call."""
    order: list[str] = []
    monkeypatch.setattr(tui_app, "prewarm_multiprocessing",
                        lambda: order.append("prewarm") or True)
    monkeypatch.setattr(tui_app.ManyrunsApp, "run", lambda self: order.append("app.run"))

    assert tui_app.run() == 0
    assert order == ["prewarm", "app.run"]


def test_a_prewarm_that_fails_costs_a_step_and_not_the_front_door(monkeypatch):
    """It reaches for a private-ish singleton (`_resource_tracker._fd`). If a future CPython
    moves it, the app must still open — a subprocess-backed step then fails the way it did
    before, which is a bad outcome and not a fatal one."""
    import multiprocessing.resource_tracker as rt

    def boom():
        raise RuntimeError("moved in 3.14")

    monkeypatch.setattr(rt, "ensure_running", boom)

    assert tui_app.prewarm_multiprocessing() is False


def test_the_bug_is_a_property_of_textuals_stderr_not_of_this_app():
    """Pinned so the fix is not mistaken for a workaround for manyruns's own code. ANY Textual
    app that reaches for multiprocessing has this, because the fd it hands the child comes from
    `sys.stderr`, and Textual replaces `sys.stderr` for the life of the app."""
    import inspect
    import multiprocessing.resource_tracker as rt

    source = inspect.getsource(rt.ResourceTracker.ensure_running)

    assert "sys.stderr.fileno()" in source, \
        "CPython stopped passing stderr's fd; re-measure whether the prewarm is still needed"


def test_headless_textual_also_captures_stderr_with_an_unusable_descriptor():
    import asyncio
    import sys
    from textual.app import App

    async def captured():
        async with App().run_test():
            return type(sys.stderr).__name__, sys.stderr.fileno()

    assert asyncio.run(captured()) == ('_PrintCapture', -1)


def test_quitting_during_an_output_wait_does_not_join_the_compute_daemon(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    program = textwrap.dedent('''
        import argparse
        import asyncio
        import threading
        import time
        from manyruns.narrate import Observation
        from manyruns.tui import samplefetch, state
        from manyruns.tui.app import ManyrunsApp
        from manyruns.tui.roster import RosterScreen
        from manyruns.tui.run import RunScreen

        entry = state.DataEntry('swissroll', 'bundled',
                                Observation('manifold', source='swissroll'), dataset='swissroll')
        started = threading.Event()
        threads = []
        def prepare(app, entry, ready, *, on_done=None):
            if on_done:
                on_done()
            ready(entry)
        samplefetch.prepare_entry = prepare

        class SlowApp(ManyrunsApp):
            def get_default_screen(self):
                return RosterScreen(entries=[entry])
            def start_for(self, *args, **kwargs):
                def start(on_step):
                    threads.append(threading.current_thread())
                    started.set()
                    threading.Event().wait(60)
                    return {}
                return start

        async def main():
            app = SlowApp(argparse.Namespace(engine='mock'))
            async with app.run_test(size=(140, 35)) as pilot:
                app.open_run(entry, 'embed')
                deadline = time.monotonic() + 3
                while not started.is_set() and time.monotonic() < deadline:
                    await pilot.pause(0.01)
                assert started.is_set()
                await pilot.press('escape', 'tab', 'enter')
                await pilot.pause()
                await pilot.press('enter')
                await pilot.pause()
                assert app._waiting_run is not None
                assert len(threads) == 1 and threads[0].daemon
                assert 'esc cancels this start' in str(app._waiting_widget.content)
                await pilot.press('ctrl+c')
                await pilot.pause()
                assert not app.is_running, 'quit must work before run_test cleans up'
            assert app._waiting_run is None
            assert threads[0].is_alive()
            print('quit-with-live-daemon', flush=True)
        asyncio.run(main())
    ''')
    env = dict(os.environ, PYTHONPATH=str(root))
    completed = subprocess.run([sys.executable, '-c', program], cwd=tmp_path, env=env,
                               capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert 'quit-with-live-daemon' in completed.stdout
