"""The run screen's labeling keys: `1`-`6` toggle a topology, `enter` commits one label row."""
from __future__ import annotations

import asyncio
import functools
import json

import pytest

pytest.importorskip("textual")

from textual.app import App  # noqa: E402 - after the importorskip, which is the point of it
from textual.widgets import Input, Static  # noqa: E402

from manyruns import agents, annotations, vocab  # noqa: E402
from manyruns.tui.run import RunScreen  # noqa: E402

EMBED = {"name": "embed", "steps": [{"name": "phate", "group": "latent",
                                     "params": {"n_components": 3}}]}
RESULTS = {"run_id": "r1", "dataset_name": "pbmc3k"}


def drives(body):
    """Run an async test body on its own loop (copied from `tests/test_tui_run.py`: this repo
    has no async pytest plugin)."""
    @functools.wraps(body)
    def wrapper(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))

    return wrapper


class _Harness(App):
    """The smallest app that can hold a screen (copied from `tests/test_tui_run.py`), and it
    records every notice so a test can read what the person was told."""

    def __init__(self, screen: RunScreen) -> None:
        super().__init__()
        self._run_screen = screen
        self.notices: list[str] = []

    def on_mount(self) -> None:
        self.push_screen(self._run_screen)

    def notify(self, message, *args, **kwargs):  # type: ignore[override]
        self.notices.append(str(message))
        return super().notify(message, *args, **kwargs)


@pytest.fixture
def here(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_ANNOTATOR", "a3")
    # no socket: the ask row probes for a model when `?` opens it
    monkeypatch.setattr(agents, "ask_available", lambda: False)
    return tmp_path


def _rows(here):
    return list(annotations.read(here / "outputs"))


async def _screen(results=RESULTS):
    screen = RunScreen(EMBED)
    app = _Harness(screen)
    return screen, app


def _strip(app) -> str:
    return str(app.screen.query_one("#label-strip", Static).render())


def test_bindings_are_not_priority():
    keys = {b.key: b for b in RunScreen.BINDINGS}
    for k in ("1", "2", "3", "4", "5", "6", "enter"):
        assert k in keys, k
        assert not keys[k].priority, k
    assert [keys[str(i + 1)].action for i in range(6)] == [
        f"toggle_topology({i})" for i in range(6)]
    assert len(vocab.TOPOLOGIES) == 6


@drives
async def test_two_keys_and_enter_write_one_row(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("3", "1", "enter")
        await pilot.pause()
        [row] = _rows(here)
        assert row["kind"] == "label" and row["chooser"] == "human:a3"
        assert row["run_id"] == "r1"
        assert row["payload"] == {"dataset": "pbmc3k",
                                  "topology": ["clusters", "multi-branching"], "raw": []}
        assert screen.topologies == set(), "a committed set is cleared"
        assert any(row["annotation_id"] in n for n in app.notices)


@drives
async def test_the_strip_shows_the_toggled_set(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("3", "1")
        await pilot.pause()
        text = _strip(app)
        assert "clusters" in text and "multi-branching" in text
        assert text.index("clusters") < text.index("multi-branching")
    assert not (here / "outputs" / "annotations.jsonl").exists()


@drives
async def test_a_key_pressed_twice_toggles_off(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("1", "1", "enter")
        await pilot.pause()
    assert not (here / "outputs" / "annotations.jsonl").exists()


@drives
async def test_each_key_toggles_its_topology(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("6", "5", "4", "3", "2", "1", "enter")
        await pilot.pause()
    [row] = _rows(here)
    assert row["payload"]["topology"] == list(vocab.TOPOLOGIES)


@drives
async def test_digits_typed_into_the_ask_field_do_not_toggle(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("question_mark")
        await pilot.pause()
        field = app.screen.query_one("#ask-input", Input)
        assert app.screen.focused is field
        await pilot.press("2")
        await pilot.pause()
        assert "2" in field.value
        assert screen.topologies == set()
        await pilot.press("enter")
        await pilot.pause()
        assert screen.topologies == set()
    assert not (here / "outputs" / "annotations.jsonl").exists()


@drives
async def test_no_annotator_set(here, monkeypatch):
    monkeypatch.delenv("MANYRUNS_ANNOTATOR")
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("1", "enter")
        await pilot.pause()
        assert any("MANYRUNS_ANNOTATOR" in n for n in app.notices)
    assert not (here / "outputs" / "annotations.jsonl").exists()


@drives
async def test_a_bad_annotator_is_refused(here, monkeypatch):
    monkeypatch.setenv("MANYRUNS_ANNOTATOR", "Jane Doe")
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("1", "enter")
        await pilot.pause()
        assert any("MANYRUNS_ANNOTATOR" in n for n in app.notices)
    assert not (here / "outputs" / "annotations.jsonl").exists()


@drives
async def test_an_unfinished_run(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert screen.results is None
        await pilot.press("1", "enter")
        await pilot.pause()
        assert any("not finished" in n for n in app.notices)
        assert screen.topologies == {"clusters"}
    assert not (here / "outputs" / "annotations.jsonl").exists()


@drives
async def test_a_failed_write_keeps_the_set(here, monkeypatch):
    monkeypatch.setattr(annotations, "append", lambda *a, **k: None)
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = dict(RESULTS)
        await pilot.press("1", "enter")
        await pilot.pause()
        assert any("not recorded" in n for n in app.notices)
        assert screen.topologies == {"clusters"}


@drives
async def test_the_dataset_falls_back_to_the_dataset_key(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        screen.results = {"run_id": "r1", "dataset_name": None, "dataset": "eb"}
        await pilot.press("2", "enter")
        await pilot.pause()
    [row] = _rows(here)
    assert row["payload"]["dataset"] == "eb"
    assert row["payload"]["topology"] == ["single-trajectory"]
    assert json.loads((here / "outputs" / "annotations.jsonl").read_text())["run_id"] == "r1"


@drives
async def test_the_strip_takes_no_row_until_something_is_toggled(here):
    screen, app = await _screen()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        strip = app.screen.query_one("#label-strip", Static)
        assert strip.display is False
        await pilot.press("1")
        await pilot.pause()
        assert strip.display is True
        await pilot.press("1")
        await pilot.pause()
        assert strip.display is False
