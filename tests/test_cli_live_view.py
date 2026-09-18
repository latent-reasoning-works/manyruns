"""The live step view on the COMMAND-LINE paths (`run`, `explore`, `init --batch`).

`shell.py` has had a working dashboard for a while and exactly one caller — the interactive
explore path. Everything reached from argv runs `app → modes.run → _infer → run_explorations`,
and `modes._infer` has no `on_step` parameter at all, so those paths printed nothing until the
run was over no matter how hard `app.py` threaded an observer. `run_explorations` is the one
function every path funnels into (`session.py` drives `pipeline` directly and never arrives
here), which is where the panel is now opened.

The guards are the load-bearing part. A live region that opens at the wrong moment is worse
than none: two things owning the cursor corrupts the display, and a repaint per update turns
piped output into megabytes of escape codes.
"""
from __future__ import annotations

import pytest


def _recipe(*names):
    return {"name": "r", "steps": [{"name": n, "group": "latent"} for n in names]}


class _Recorder:
    """A stand-in engine: records the payload it was served and runs no compute."""

    def __init__(self, boom: Exception | None = None) -> None:
        self.inputs: dict | None = None
        self.boom = boom

    def predict(self, inputs):
        self.inputs = inputs
        if self.boom is not None:
            raise self.boom
        return {"engine": "manylatents"}


def _console(monkeypatch, *, terminal: bool):
    """Point the panel's console at a string buffer with `is_terminal` under our control.

    The whole decision turns on that flag, so the test must set it rather than inherit
    whatever the runner happened to attach to stdout."""
    pytest.importorskip("rich")
    import io

    from rich.console import Console

    from manyruns import shell

    buf = io.StringIO()
    monkeypatch.setattr(
        shell, "_default_console", lambda: Console(file=buf, force_terminal=terminal, width=80)
    )
    return buf


def _spy_on_live_regions(monkeypatch) -> list:
    """Every live region that gets opened. An empty list is the assertion that nothing took
    the cursor."""
    from manyruns import shell

    opened: list = []
    real = shell.live_dashboard

    def spy(console, recipe, title):
        live, on_step = real(console, recipe, title)
        opened.append(live)
        return live, on_step

    monkeypatch.setattr(shell, "live_dashboard", spy)
    return opened


def _run(server, monkeypatch, *, engine="manylatents", recipe=None, on_step=None):
    from manyruns import app

    return app.run_explorations(
        None, "scrna", server=server, recipe=recipe if recipe is not None else _recipe("phate"),
        engine=engine, on_step=on_step,
    )


def test_a_command_line_run_at_a_terminal_reaches_the_engine_with_an_observer(monkeypatch):
    """The gap this closes: `manyruns run` used to serve a payload with no
    `on_step` key, so the step loop had nobody to report to and the terminal sat silent."""
    _console(monkeypatch, terminal=True)
    rec = _Recorder()

    _run(rec, monkeypatch)

    assert callable(rec.inputs["on_step"])


def test_piped_output_gets_no_live_region_and_no_observer(monkeypatch):
    """Redirected to a file, rich writes a full repaint of the panel per update. The payload
    must come out exactly as it did before this existed — no `on_step` key at all."""
    _console(monkeypatch, terminal=False)
    opened = _spy_on_live_regions(monkeypatch)
    rec = _Recorder()

    _run(rec, monkeypatch)

    assert opened == []
    assert "on_step" not in rec.inputs


def test_an_observer_the_caller_already_supplied_is_never_displaced(monkeypatch):
    """`shell._run` opens its own dashboard and passes the observer down. Opening a second
    one is two things owning one stdout — the failure shell.py's own comment (l. 507) names.
    The caller's observer must survive untouched, and nothing new may be opened."""
    _console(monkeypatch, terminal=True)
    opened = _spy_on_live_regions(monkeypatch)
    mine = lambda *_: None  # noqa: E731
    rec = _Recorder()

    _run(rec, monkeypatch, on_step=mine)

    assert rec.inputs["on_step"] is mine
    assert opened == [], "a second live region was opened on top of the shell's"


def test_the_live_region_is_closed_even_when_the_run_raises(monkeypatch):
    """A failed run that keeps the cursor leaves the user's terminal hidden-cursor and
    half-repainted, and only a `reset` gets it back. The panel is scaffolding for the wait;
    it comes down however the wait ends."""
    _console(monkeypatch, terminal=True)
    opened = _spy_on_live_regions(monkeypatch)
    rec = _Recorder(boom=RuntimeError("the fit exploded"))

    with pytest.raises(RuntimeError, match="the fit exploded"):
        _run(rec, monkeypatch)

    assert len(opened) == 1
    assert opened[0].is_started is False


@pytest.mark.parametrize("engine", ["mock"])
def test_engines_that_never_report_a_step_get_no_panel(monkeypatch, engine):
    """`serving.LocalServer` forwards `on_step` only on the `manylatents` path. `mock` runs the
    whole recipe in one call, so a panel opened for it would sit at "queued" for the entire run
    and then vanish — a progress bar that never moves is a worse answer than silence.

    The learner's engine was the second parameter here and is no longer one (§3.5). The parametrize
    is kept with one value rather than inlined: `_REPORTING_ENGINES` is a list, and the next
    non-reporting backend belongs in this row, not in a rewritten test."""
    _console(monkeypatch, terminal=True)
    opened = _spy_on_live_regions(monkeypatch)
    rec = _Recorder()

    _run(rec, monkeypatch, engine=engine)

    assert opened == []


def test_a_recipe_with_no_declared_steps_gets_no_panel(monkeypatch):
    """Nothing to animate, so the panel would be an empty box around a blocking call."""
    _console(monkeypatch, terminal=True)
    opened = _spy_on_live_regions(monkeypatch)

    _run(_Recorder(), monkeypatch, recipe={"name": "r", "steps": []})

    assert opened == []


def test_the_mode_router_path_gets_the_view_too(monkeypatch):
    """The actual command-line route, end to end through the layer that drops the parameter.

    `cmd_run` / `cmd_explore` / `cmd_init --batch` all go through `_explore_project` →
    `modes.run("infer", …)` → `modes._infer`, which forwards fifteen keys and no `on_step`.
    This test fails if the panel is ever moved up into `_explore_project`, where the mode
    router would silently drop it again."""
    from manyruns import app, modes

    _console(monkeypatch, terminal=True)
    rec = _Recorder()
    monkeypatch.setattr(app, "_server_for", lambda *a, **k: rec)

    modes.run("infer", {"recipe": _recipe("phate"), "engine": "manylatents", "modality": "scrna"})

    assert callable(rec.inputs["on_step"])


def test_the_installed_observer_actually_draws_the_declared_steps(monkeypatch):
    """It has to be wired to the real dashboard, not to a no-op that merely satisfies the
    `callable` checks above. Feeding it one finished record must repaint the panel."""
    buf = _console(monkeypatch, terminal=True)
    rec = _Recorder()

    _run(rec, monkeypatch, recipe=_recipe("phate", "separation"))
    rec.inputs["on_step"]({}, [{"index": 0, "name": "phate", "outcome": "ok", "seconds": 1.5}])

    text = buf.getvalue()
    assert "phate" in text and "separation" in text
    assert "queued" in text  # the steps nobody has reached yet
