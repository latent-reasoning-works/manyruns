"""Run-screen colour views and exports through real UI and offline display helpers."""
import argparse
import asyncio
import functools
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from textual.app import App
from textual.widgets import Input

from manyruns import figspec
from manyruns.narrate import Observation
from manyruns.pipeline import io, loading
from manyruns.tui import colorby, state
from manyruns.tui.figure import FigureScreen
from manyruns.tui.run import RunScreen, FiguresPane, LeaveRunScreen


RECIPE = {'name': 'embed', 'steps': [{'name': 'pca', 'group': 'latent'}]}


def drives(body):
    @functools.wraps(body)
    def run(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))
    return run


class Harness(App):
    def __init__(self, screen):
        super().__init__()
        self.initial = screen

    def on_mount(self):
        self.push_screen(self.initial)


async def until(pilot, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        await pilot.pause(0.02)
    assert predicate(), 'UI operation did not finish'


def notices(app):
    return '\n'.join(str(n.message) for n in app._notifications)


def frame(app):
    return '\n'.join(strip.text for strip in app.screen._compositor.render_strips())


def example(tmp_path, *, empty=False, generated=False):
    if generated:
        obj = loading.time_course_anndata()
        entry = state.catalog_entry('synthetic_timecourse')
    else:
        obs = pd.DataFrame(index=[f'cell-{i}' for i in range(6)])
        if not empty:
            obs['group'] = ['a', 'b'] * 3
            obs['day'] = [0, 1, 2] * 2
        obj = AnnData(np.arange(18).reshape(6, 3).astype(float), obs=obs)
        path = tmp_path / 'source.h5ad'
        obj.write_h5ad(path)
        entry = state.DataEntry('source', 'file', Observation('single'), path=path)
    png = io.save_display_scatter(obj.X[:, :2], [], tmp_path, 'pca.png', [], 'PCA',
                                  obs_names=obj.obs_names)[0]
    record = {'index': 0, 'name': 'pca', 'outcome': 'ok', 'seconds': 0.1, 'plots': [png]}
    screen = RunScreen(RECIPE, entry=entry)
    screen.feed.on_step(record, [record])
    return screen, png, record


async def finish(pilot, screen):
    screen.post_message(screen.Finished({'run_id': 'run-123', 'steps': screen.feed.records}))
    await until(pilot, lambda: screen._finished)


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('click', [False, True])
@drives
async def test_colour_views_survive_sync_fullscreen_return_and_export(tmp_path, monkeypatch, full, click):
    monkeypatch.chdir(tmp_path)
    screen, png, record = example(tmp_path)
    originals = {path: path.read_bytes() for path in (
        Path(png), Path(png).with_suffix('.spec.npz'), Path(png).with_suffix('.spec.json'))}
    app = Harness(screen)
    async with app.run_test(size=(140, 35)) as pilot:
        await finish(pilot, screen)
        assert 'c colour' in frame(app)
        if full:
            await pilot.press('f')
            assert isinstance(app.screen, FigureScreen)
        await pilot.press('c')
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        assert [c['key'] for c in app.screen.metadata.columns] == ['group', 'day']
        if click:
            await pilot.click('#color-options', offset=(2, 1))
        else:
            await pilot.press('enter')
        await until(pilot, lambda: len(screen.query_one(FiguresPane).found) == 2)
        if full:
            await pilot.press('escape')
        assert app.screen is screen
        new_path = screen._selected_figure()[1]
        assert new_path != png and Path(new_path).is_file()
        assert figspec.load(new_path)['color_by'] in ('group', 'day')
        screen.feed.on_step(record, [record])
        await pilot.pause()
        assert screen._selected_figure()[1] == new_path
        assert len(screen.query_one(FiguresPane).found) == 2
        assert record['plots'] == [png] and screen.feed.records == [record]
        await pilot.press('f7')
        assert screen._selected_figure()[1] == png
        await pilot.press('f8')
        assert screen._selected_figure()[1] == new_path
        await pilot.press('s')
        await until(pilot, lambda: 'saved to' in notices(app))
        exported = tmp_path / 'figures'
        assert exported.is_dir()
        assert {p.suffix for p in exported.iterdir()} == {'.png', '.pdf', '.svg'}
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize('full', [False, True])
@drives
async def test_live_recolour_is_refused_on_both_surfaces(tmp_path, monkeypatch, full):
    screen, _, _ = example(tmp_path)
    monkeypatch.setattr(colorby, 'load_metadata', lambda *a: pytest.fail('live metadata read'))
    app = Harness(screen)
    async with app.run_test() as pilot:
        if full:
            await pilot.press('f')
        await pilot.press('c')
        await until(pilot, lambda: 'finish this run before recolouring' in notices(app))
        assert not isinstance(app.screen, colorby.ColorByScreen)


@pytest.mark.parametrize('mode', ['no-figure', 'empty', 'generated', 'stale'])
@drives
async def test_colour_action_explains_absence_or_loads_synthetic_timepoint(tmp_path, mode):
    screen, _, _ = example(tmp_path, empty=mode == 'empty', generated=mode == 'generated')
    if mode == 'no-figure':
        screen.feed.records = []
    if mode == 'stale':
        screen.entry.path.unlink()
    app = Harness(screen)
    async with app.run_test() as pilot:
        await finish(pilot, screen)
        await pilot.press('c')
        if mode == 'generated':
            await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
            assert app.screen.metadata.columns[0]['key'] == 'timepoint'
            await pilot.press('enter')
            await until(pilot, lambda: len(screen.query_one(FiguresPane).found) == 2)
            assert figspec.load(screen._selected_figure()[1])['color_by'] == 'timepoint'
        else:
            expected = {'no-figure': 'no figure to recolour',
                        'empty': 'drop an annotated .h5ad to pick one',
                        'stale': 'could not read metadata'}[mode]
            await until(pilot, lambda: expected in notices(app))
            assert app.screen is screen


@pytest.mark.parametrize('dismiss', [False, True])
@drives
async def test_run_save_waiting_on_renderer_keeps_heartbeat_and_can_cancel(tmp_path, monkeypatch, dismiss):
    monkeypatch.chdir(tmp_path)
    screen, png, _ = example(tmp_path)
    app = Harness(screen)
    held, release, expired = threading.Event(), threading.Event(), threading.Event()

    def hold():
        with figspec.RENDER_LOCK:
            held.set()
            if not release.wait(3):
                expired.set()

    holder = threading.Thread(target=hold, daemon=True)
    async with app.run_test() as pilot:
        await finish(pilot, screen)
        holder.start()
        try:
            await until(pilot, held.is_set)
            await pilot.press('s')
            heartbeat = []
            app.call_later(lambda: heartbeat.append(True))
            await pilot.pause(0.02)
            assert heartbeat and not expired.is_set(), 'save blocked the event loop'
            await pilot.press('s')
            if dismiss:
                await pilot.press('escape')
                assert isinstance(app.screen, LeaveRunScreen)
                await pilot.press('tab', 'enter')
                assert app.screen is not screen
            release.set()
            if dismiss:
                await until(pilot, lambda: not holder.is_alive())
                await pilot.pause(0.1)
                assert not (tmp_path / 'figures').exists()
                assert 'saved to' not in notices(app)
            else:
                await until(pilot, lambda: 'saved to' in notices(app))
                out = tmp_path / 'figures'
                assert out.is_dir()
                files = list(out.iterdir())
                assert len(files) == 3
                assert all('run-123' in p.name for p in files)
        finally:
            release.set()
            holder.join(timeout=4)
        assert not holder.is_alive()


@pytest.mark.parametrize('pending', [False, True])
@drives
async def test_c_remains_text_in_answer_and_ask_inputs(tmp_path, monkeypatch, pending):
    from manyruns.tui.gate import Gate
    from manyruns import agents

    monkeypatch.setattr(agents, 'ask_available', lambda: False)
    screen, _, _ = example(tmp_path)
    gate = Gate()
    screen = RunScreen(RECIPE, entry=screen.entry, gate=gate)
    app = Harness(screen)
    worker = None
    async with app.run_test() as pilot:
        try:
            if pending:
                worker = threading.Thread(target=lambda: gate.read('accept / tune / cancel?'), daemon=True)
                worker.start()
                await until(pilot, lambda: not screen.query_one(Input).disabled)
            else:
                await finish(pilot, screen)
                await pilot.press('question_mark')
            field = screen.query_one(Input)
            before = field.value
            await pilot.press('end', 'c')
            assert field.value == before + 'c'
            assert app.screen is screen
        finally:
            gate.close()
            if worker is not None:
                worker.join(timeout=2)


@drives
async def test_all_missing_colour_channel_is_said_during_the_run(tmp_path, monkeypatch):
    from manyruns import app as cli, agents
    from manyruns.pipeline import runner
    from manyruns.session import Session
    from manyruns.tui.app import ManyrunsApp
    from manyruns.tui.gate import Gate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agents, 'ask_available', lambda: False)
    key = '[score]'
    recipe = {'name': 'embed', 'steps': [{'name': 'phate', 'group': 'latent'}]}
    session = Session('missing', engine='_inproc', array=np.ones((6, 3)), recipe=recipe,
                      out_dir=tmp_path, color=[{'key': key, 'kind': 'continuous',
                                               'values': np.full(6, np.nan)}])

    def execute(name, params, state, g, ctx):
        state['emb'] = state['X'][:, :2].copy()
        runner._display_scatter(state['emb'], state, ctx, 'pca.png', 'PCA')

    session.dispatch = {'latent': execute}
    monkeypatch.setattr(runner, '_attach_geometry', lambda *a, **kw: {})
    monkeypatch.setattr(cli, '_build_session', lambda *a, **kw: session)
    driver = ManyrunsApp(argparse.Namespace(engine='mock'))
    gate = Gate()
    screen = RunScreen(recipe, gate=gate, start=driver.start_for(
        state.catalog_entry('swissroll'), 'embed', gate=gate))
    app = Harness(screen)
    async with app.run_test(size=(150, 35)) as pilot:
        try:
            await until(pilot, lambda: not screen.query_one(Input).disabled)
            assert not screen._finished
            assert "skipping colour channel '[score]': all plotted values are missing or nonfinite" in frame(app)
        finally:
            gate.close()
            await until(pilot, lambda: screen._finished)


@drives
async def test_viewer_opened_during_run_can_add_views_after_finished(tmp_path):
    screen, png, _ = example(tmp_path)
    app = Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press('f')
        assert isinstance(app.screen, FigureScreen)
        await finish(pilot, screen)
        await pilot.press('c')
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        await pilot.press('enter')
        await until(pilot, lambda: 'saved colour view' in notices(app))
        await pilot.press('escape')
        assert len(screen.query_one(FiguresPane).found) == 2
        assert screen._selected_figure()[1] != png


@pytest.mark.parametrize('unmount', [False, True])
@drives
async def test_late_colour_result_is_removed_after_departure_or_new_run(tmp_path, monkeypatch, unmount):
    screen, _, _ = example(tmp_path)
    app = Harness(screen)
    entered, release, ended = threading.Event(), threading.Event(), threading.Event()
    real = colorby.recolor_figure
    produced = []

    def recolor(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(60)
            result = real(*args, **kwargs)
            produced.append(result[0])
            return result
        finally:
            ended.set()

    monkeypatch.setattr(colorby, 'recolor_figure', recolor)
    async with app.run_test() as pilot:
        await finish(pilot, screen)
        await pilot.press('c')
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        await pilot.press('enter')
        try:
            await until(pilot, entered.is_set)
            if unmount:
                await pilot.press('escape', 'tab', 'enter')
                assert app.screen is not screen
            else:
                screen.post_message(screen.Finished({'run_id': 'different-run'}))
                await pilot.pause()
            release.set()
            await until(pilot, ended.is_set)
            await until(pilot, lambda: bool(produced) and not Path(produced[0]).exists())
            assert 'saved colour view' not in notices(app)
            assert not screen._extra_views
        finally:
            release.set()


@pytest.mark.parametrize('full', [False, True])
@drives
async def test_extra_views_belong_to_step_occurrences_with_repeated_names(tmp_path, full):
    screen, first, record = example(tmp_path)
    second = io.save_display_scatter(np.arange(12).reshape(6, 2), [], tmp_path, 'other.png', [],
                                     'PCA again', obs_names=[f'cell-{i}' for i in range(6)])[0]
    records = [record, {**record, 'index': 1, 'plots': [second]}]
    screen.feed.declared.append({'name': 'pca', 'group': 'latent'})
    screen.feed.on_step(records[-1], records)
    app = Harness(screen)
    async with app.run_test() as pilot:
        await finish(pilot, screen)
        await pilot.press('f7')
        assert screen._selected_figure()[1] == first
        if full:
            await pilot.press('f')
        await pilot.press('c')
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        await pilot.press('enter')
        await until(pilot, lambda: len(screen.query_one(FiguresPane).found) == 3)
        if full:
            await pilot.press('escape')
        view = screen._selected_figure()[1]
        screen.feed.on_step(records[-1], records)
        await pilot.pause()
        assert [path for _, path in screen.query_one(FiguresPane).found] == [first, view, second]
        assert screen._extra_views == {(0, 'pca'): [view]}
        await pilot.press('f8')
        assert screen._selected_figure()[1] == second
        assert records[0]['plots'] == [first] and records[1]['plots'] == [second]


@drives
async def test_run_export_captures_selection_serializes_duplicates_and_notifies_on_ui_thread(
        tmp_path, monkeypatch):
    from manyruns.tui import figure

    monkeypatch.chdir(tmp_path)
    screen, png, record = example(tmp_path)
    other = io.save_display_scatter(np.arange(12).reshape(6, 2), [], tmp_path, 'other.png', [],
                                    'PCA again', obs_names=[f'cell-{i}' for i in range(6)])[0]
    screen.feed.on_step(record, [{**record, 'plots': [png, other]}])
    entered, release = threading.Event(), threading.Event()
    calls, threads, notifications = [], [], []
    ui_thread = threading.get_ident()
    real = figure.quickdrop
    notify = screen.notify

    def checked_notify(message, **kwargs):
        assert threading.get_ident() == ui_thread
        notifications.append(message)
        return notify(message, **kwargs)

    def blocked(*args, **kwargs):
        threads.append(threading.current_thread())
        assert threading.get_ident() != ui_thread
        calls.append(args)
        entered.set()
        assert release.wait(60)
        return real(*args, **kwargs)

    monkeypatch.setattr(figure, 'quickdrop', blocked)
    monkeypatch.setattr(screen, 'notify', checked_notify)
    app = Harness(screen)
    async with app.run_test() as pilot:
        await finish(pilot, screen)
        try:
            await pilot.press('s')
            await until(pilot, entered.is_set)
            await pilot.press('f7', 's')
            assert screen._selected_figure()[1] == png
            assert calls == [(other, 'pca', 'run-123')]
            assert all(t.daemon and t.name == 'manyruns-export' for t in threads)
            release.set()
            await until(pilot, lambda: any('saved to' in str(n) for n in notifications))
            message = next(str(n) for n in notifications if 'saved to' in str(n))
            assert all(suffix in message for suffix in ('.png', '.pdf', '.svg'))
        finally:
            release.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('navigate', [False, True])
@drives
async def test_recolour_second_occurrence_with_shared_output_basename(
        tmp_path, monkeypatch, full, navigate):
    screen, png, record = example(tmp_path)
    records = [record, {**record, 'index': 1, 'plots': [png]}]
    screen.feed.declared.append({'name': 'pca', 'group': 'latent'})
    screen.feed.on_step(records[-1], records)
    entered, release = threading.Event(), threading.Event()
    recolor = colorby.recolor_figure

    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(60)
        return recolor(*args, **kwargs)

    monkeypatch.setattr(colorby, 'recolor_figure', delayed)
    app = Harness(screen)
    async with app.run_test() as pilot:
        try:
            await finish(pilot, screen)
            pane = screen.query_one(FiguresPane)
            assert pane.at() == 1
            if full:
                await pilot.press('f')
                viewer = app.screen
                assert viewer.index == 1
            await pilot.press('c')
            await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
            await pilot.press('enter')
            await until(pilot, entered.is_set)
            if navigate:
                await pilot.press('right' if full else 'f7')
            release.set()
            await until(pilot, lambda: len(pane.found) == 3)
            view = screen._selected_figure()[1]
            assert screen._extra_views == {(1, 'pca'): [view]}
            assert pane.found == [('pca', png), ('pca', png), ('pca', view)]
            if full:
                assert viewer.figures == pane.found
                assert viewer.index == (0 if navigate else 2)
                await pilot.press('escape')
            screen.feed.on_step(records[-1], records)
            await pilot.pause()
            assert screen._selected_figure()[1] == view
            assert pane.found == [('pca', png), ('pca', png), ('pca', view)]
            assert all(r['plots'] == [png] for r in records)
        finally:
            release.set()


@pytest.mark.parametrize('slow_capture', [False, True])
@drives
async def test_queued_export_keeps_attempt_spec_across_tuning_retry(
        tmp_path, monkeypatch, slow_capture):
    from manyruns import agents, tune
    from manyruns.session import Session
    from manyruns.tui.gate import Gate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agents, 'ask_available', lambda: False)
    initial, png, _ = example(tmp_path)
    for suffix in ('.png', '.spec.npz', '.spec.json'):
        Path(png).with_suffix(suffix).rename(tmp_path / ('phate' + suffix))
    png = str(tmp_path / 'phate.png')
    recipe = {'name': 'embed', 'steps': [{'name': 'phate', 'group': 'latent'}]}
    original = figspec.load(png)
    png_bytes = Path(png).read_bytes()
    gate = Gate()
    held, release_render = threading.Event(), threading.Event()
    capturing, release_capture = threading.Event(), threading.Event()
    retried, exported = threading.Event(), []
    real_load, real_figure = figspec.load, figspec.figure
    ui_thread = threading.get_ident()

    def load(path):
        if threading.current_thread().name == 'manyruns-export':
            assert threading.get_ident() != ui_thread
            capturing.set()
            if slow_capture:
                assert release_capture.wait(60)
        return real_load(path)

    def figure(spec, **kwargs):
        exported.append(spec['coords'].copy())
        return real_figure(spec, **kwargs)

    def hold_render():
        with figspec.RENDER_LOCK:
            held.set()
            assert release_render.wait(60)

    def start(on_step):
        session = Session(project='p', engine='mock', modality='scrna',
                          recipe=recipe, out_dir=tmp_path / 'run')

        def reported(record, records):
            if record.get('outcome') is None:
                on_step(record, records)
                return
            coords = original['coords'] + (100 if len(records) > 1 else 0)
            Path(png).write_bytes(png_bytes)
            figspec.save(png, coords, None, original, obs_names=original['obs_names'])
            record['plots'] = [png]
            on_step(record, records)
            if len(records) > 1:
                retried.set()

        session.on_step = reported
        tune.run_tune_loop(session, {'name': 'phate', 'group': 'latent',
                                    'params': {'knn': 5}},
                           gate.read, gate.write, show_plots=lambda *a: None)
        return {'run_id': 'run-123', 'steps': session.steps}

    monkeypatch.setattr(figspec, 'load', load)
    monkeypatch.setattr(figspec, 'figure', figure)
    screen = RunScreen(recipe, entry=initial.entry, start=start, gate=gate)
    app = Harness(screen)
    holder = threading.Thread(target=hold_render, daemon=True)
    async with app.run_test() as pilot:
        try:
            await until(pilot, lambda: screen._question_id is not None)
            holder.start()
            await until(pilot, held.is_set)
            screen.query_one('#steps-pane').focus()
            await pilot.press('s')
            if slow_capture:
                # Reading the snapshot must start even while another renderer owns the lock.
                await until(pilot, capturing.is_set)
            field = screen.query_one(Input)
            field.value = 'knn=10'
            field.focus()
            await pilot.press('enter')
            heartbeat = []
            app.call_later(lambda: heartbeat.append(True))
            await pilot.pause(0.02)
            assert heartbeat
            if slow_capture:
                assert not retried.is_set(), 'retry renamed the spec before capture finished'
                release_capture.set()
            await until(pilot, retried.is_set)
            assert Path(png).with_name('phate@1.png').is_file()
            np.testing.assert_array_equal(real_load(png)['coords'], original['coords'] + 100)
            assert not exported, 'export did not wait for the renderer'
            release_render.set()
            await until(pilot, lambda: 'saved to' in notices(app))
            assert len(exported) == 1
            np.testing.assert_array_equal(exported[0], original['coords'])
            out = tmp_path / 'figures'
            assert out.is_dir()
            assert {p.suffix for p in out.iterdir()} == {'.png', '.pdf', '.svg'}
        finally:
            release_capture.set()
            release_render.set()
            gate.close()
            if holder.ident is not None:
                holder.join(timeout=2)
            await until(pilot, lambda: not screen.worker_thread.is_alive())


@pytest.mark.parametrize('full', [False, True])
@drives
async def test_ask_refresh_keeps_second_shared_figure_occurrence_before_recolour(
        tmp_path, monkeypatch, full):
    from manyruns import agents

    monkeypatch.setattr(agents, 'ask_available', lambda: False)
    screen, png, record = example(tmp_path)
    records = [record, {**record, 'index': 1, 'plots': [png]}]
    screen.feed.declared.append({'name': 'pca', 'group': 'latent'})
    screen.feed.on_step(records[-1], records)
    app = Harness(screen)
    async with app.run_test() as pilot:
        await finish(pilot, screen)
        # Explicit selection must survive an ask's repaint before any view exists.
        await pilot.press('f7', 'f8')
        pane = screen.query_one(FiguresPane)
        assert pane.selected == 1
        await pilot.press('?')
        screen.query_one(Input).value = '?Which step is this?'
        await pilot.press('enter')
        await until(pilot, lambda: screen.asked == 'Which step is this?')
        assert pane.at() == 1
        await pilot.press('escape')
        screen.feed.on_step(records[-1], records)
        await pilot.pause()
        assert pane.at() == 1
        if full:
            await pilot.press('f')
            assert app.screen.index == 1
        await pilot.press('c')
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        await pilot.press('enter')
        await until(pilot, lambda: len(pane.found) == 3)
        if full:
            await pilot.press('escape')
        view = screen._selected_figure()[1]
        assert screen._extra_views == {(1, 'pca'): [view]}
        assert pane.found == [('pca', png), ('pca', png), ('pca', view)]
        assert all(r['plots'] == [png] for r in records)


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('plot_index', [0, 1])
@drives
async def test_tuning_refresh_keeps_selected_attempt_and_plot_before_recolour(
        tmp_path, monkeypatch, full, plot_index):
    from manyruns import agents, tune
    from manyruns.tui import images

    monkeypatch.setattr(agents, 'ask_available', lambda: False)
    monkeypatch.setattr(images, 'is_pixel_perfect', lambda: True)
    monkeypatch.setattr(images.AutoImage, '_Renderable', images._Half)
    initial, png, _ = example(tmp_path)
    original = figspec.load(png)
    plots = [io.save_display_scatter(
        original['coords'] + offset, [], tmp_path, name, [], 'PHATE',
        obs_names=original['obs_names'])[0]
        for offset, name in enumerate(('phate.png', 'phate-other.png'))]
    record = {'index': 0, 'name': 'phate', 'outcome': 'ok', 'plots': list(plots)}
    recipe = {'name': 'embed', 'steps': [{'name': 'phate', 'group': 'latent'}]}
    screen = RunScreen(recipe, entry=initial.entry)
    screen.feed.on_step(record, [record])
    app = Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press('f7')
        if plot_index:
            await pilot.press('f8')
        pane = screen.query_one(FiguresPane)
        assert pane.selected == plot_index
        selected_coords = figspec.load(plots[plot_index])['coords'].copy()

        # Tuning can archive and reuse both paths before the next UI repaint.
        # Exercise its real rename, including the in-place record and spec updates.
        messages = []
        tune._keep_attempt({'record': record}, 1, messages.append)
        assert len(messages) == 2
        archived = list(record['plots'])
        thumb = pane.query_one('#figure-thumb', images.AutoImage)
        assert thumb.image == plots[plot_index]
        thumb.refresh(layout=True)
        await pilot.resize_terminal(100, 35)
        await pilot.pause()
        assert app.is_running
        for path in plots:
            io.save_display_scatter(original['coords'] + 100, [], tmp_path,
                                    Path(path).name, [], 'PHATE retry',
                                    obs_names=original['obs_names'])
        retry = {**record, 'index': 1, 'plots': list(plots)}
        screen.feed.on_step(retry, [record, retry])
        await until(pilot, lambda: len(pane.found) == 4)
        assert pane.at() == plot_index
        assert screen._selected_figure() == ('phate', archived[plot_index])
        await finish(pilot, screen)
        assert pane.at() == plot_index
        if full:
            await pilot.press('f')
            assert app.screen.index == plot_index
        await pilot.press('c')
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        await pilot.press('enter')
        await until(pilot, lambda: len(pane.found) == 5)
        if full:
            await pilot.press('escape')
        view = screen._selected_figure()[1]
        assert screen._extra_views == {(0, 'phate'): [view]}
        np.testing.assert_array_equal(figspec.load(view)['coords'], selected_coords)
        assert [path for _, path in pane.found] == [*archived, view, *plots]
        assert record['plots'] == archived and retry['plots'] == plots
