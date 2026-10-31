"""The figure pane — a picture per step, drawn as that step lands.

The old behaviour was a flat list of paths in a panel after the run. Two failures in that: a
run with two figures could not say which step drew which, and the picture arrived only once
everything had finished — so the one moment worth seeing on a `phate → mioflow` run, the
embedding BEFORE the trajectory step touched it, was never shown live.
"""
import importlib.util

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
#: A REAL EMBEDDER, needed by the two tests here that RUN a recipe rather than hand the pane a
#: file. Per-test rather than module-level on purpose: the rest of this file is about the pane's
#: own behaviour — what it writes, when it stops the live region, what it says with no protocol —
#: and none of that needs an embedder to be installed to be worth checking.
_needs_embedder = pytest.mark.skipif(importlib.util.find_spec("phate") is None,
                                     reason="runs a real recipe; an absent embedder is not a "
                                            "defect in this code")

from manyruns import catalog, figures  # noqa: E402
from manyruns.pipeline import runner  # noqa: E402


class _Console:
    """A terminal that records what was written, both ways it can be written to."""

    is_terminal = True

    def __init__(self):
        self.printed = []
        self.file = self
        self.raw = ""

    def print(self, *a):
        self.printed.extend(str(x) for x in a)

    def write(self, s):
        self.raw += s

    def flush(self):
        pass


class _Live:
    """Stands in for `rich.Live`, recording the stop/start the pane must perform."""

    def __init__(self):
        self.is_started = True
        self.events = []

    def stop(self):
        self.is_started = False
        self.events.append("stop")

    def start(self):
        self.is_started = True
        self.events.append("start")

    def update(self, _frame):
        self.events.append("update")


def _png(tmp_path, name="f.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return str(p)


# ── attribution: which step drew which figure ────────────────────────────────
@_needs_embedder
def test_each_step_records_the_figure_it_drew(tmp_path):
    """`_io._save_scatter` appends to ONE list shared by every step, so nothing attributed a
    figure to its author. That is the same gap `emitted` closed for g-vector keys, and it is
    what a pane following the run needs."""
    res = runner.run_inproc(np.random.default_rng(0).normal(size=(150, 12)),
                              catalog.load_recipe("cflows"), tmp_path, seed=0)

    by_step = {s["name"]: [f.rsplit("/", 1)[-1] for f in (s.get("plots") or [])]
               for s in res["steps"]}
    assert by_step["phate"] == ["phate.png"]
    assert by_step["mioflow"] == ["trajectory.png"]


def test_a_step_that_drew_nothing_carries_no_figure_key(tmp_path):
    """Absent, not an empty list — the same rule the rest of the record follows, so a reader
    cannot mistake "drew nothing" for "drew something that vanished"."""
    recipe = {"name": "r", "steps": [{"name": "separation", "group": "analysis", "params": {}}]}

    res = runner.run_inproc(np.random.default_rng(0).normal(size=(60, 8)), recipe,
                              tmp_path, seed=0)

    assert "plots" not in res["steps"][0]


# ── the pane ─────────────────────────────────────────────────────────────────
def test_the_pane_draws_a_figure_once_however_often_the_observer_fires(tmp_path, monkeypatch):
    """`on_step` fires at `running` and again when the step settles. A pane keyed on the event
    rather than on the figure would paint the same picture into the scrollback twice."""
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "iterm")
    con = _Console()
    pane = figures.FigurePane(con)
    rec = {"name": "phate", "plots": [_png(tmp_path)]}

    assert pane.show(rec) == 1
    assert pane.show(rec) == 0            # same record, second event
    assert pane.drawn == 1


def test_the_pane_stops_the_live_region_around_the_draw(tmp_path, monkeypatch):
    """`rich.Live` repaints by erasing N lines, where N is what rich measured. An image escape
    is opaque to that measurement, so drawing INTO the region makes the next repaint erase the
    wrong lines. Stop, draw at a clean cursor, start a fresh region."""
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "iterm")
    con, live = _Console(), _Live()
    pane = figures.FigurePane(con)

    pane.show({"name": "phate", "plots": [_png(tmp_path)]}, live)

    assert live.events == ["stop", "start"]
    assert live.is_started is True


def test_the_live_region_is_restarted_even_if_the_draw_fails(tmp_path, monkeypatch):
    """A dead dashboard for the rest of the run is a worse failure than a missing picture."""
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "iterm")
    live = _Live()

    class _Boom(_Console):
        def print(self, *a):
            raise RuntimeError("render exploded")

    pane = figures.FigurePane(_Boom())
    with pytest.raises(RuntimeError):
        pane.show({"name": "phate", "plots": [_png(tmp_path)]}, live)

    assert live.is_started is True and live.events == ["stop", "start"]


def test_a_terminal_with_no_protocol_still_attributes_the_figure(tmp_path, monkeypatch):
    """The fallback is not the old behaviour. It names the STEP beside the path, which is what
    the flat list could never do."""
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "off")
    # The HINT is terminal-specific — `figures.hint()` names VS Code's `enableImages` setting
    # when `TERM_PROGRAM=vscode`, and iTerm2 otherwise — so a test that asserts on it has to say
    # which terminal it is asking about. Left to the environment it passed in CI and failed for
    # anyone running the suite from the VS Code terminal, which makes it a test about the
    # developer's setup rather than about the pane.
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    con = _Console()
    pane = figures.FigurePane(con)

    pane.show({"name": "mioflow", "plots": [_png(tmp_path, "trajectory.png")]})

    assert con.raw == ""                                    # nothing binary was written
    assert any("mioflow" in p for p in con.printed)
    assert any("trajectory.png" in p for p in con.printed)
    assert "iTerm2" in pane.summary()                       # and it says why, once


def test_nothing_binary_reaches_a_pipe(tmp_path, monkeypatch):
    """Redirected output must stay clean — the check the rest of this repo uses."""
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "iterm")

    class _Piped(_Console):
        is_terminal = False

    con = _Piped()
    figures.FigurePane(con).show({"name": "phate", "plots": [_png(tmp_path)]})

    assert con.raw == ""
    assert any("phate" in p for p in con.printed)           # the fact still arrives


# ── the pane inside the dashboard ────────────────────────────────────────────
@_needs_embedder
def test_the_dashboard_shows_each_figure_as_its_step_lands(tmp_path, monkeypatch):
    """End to end through the observer the runner actually calls: two steps, two figures, each
    drawn at the moment its own step finished rather than both at the end."""
    pytest.importorskip("rich")
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "off")
    from manyruns import shell

    con = _Console()
    live, on_step = shell.live_dashboard(con, catalog.load_recipe("cflows"), "t")

    order = []
    real_print = con.print

    def spy(*a):
        order.append(" ".join(str(x) for x in a))
        real_print(*a)

    con.print = spy
    runner.run_inproc(np.random.default_rng(0).normal(size=(150, 12)),
                        catalog.load_recipe("cflows"), tmp_path, seed=0, on_step=on_step)

    drawn = [o for o in order if "png" in o]
    assert len(drawn) == 2
    assert "phate" in drawn[0] and "phate.png" in drawn[0]
    assert "mioflow" in drawn[1] and "trajectory.png" in drawn[1]
