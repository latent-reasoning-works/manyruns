"""WS7 preservation walks across the assembled app, with disposable data and no network.

The fixes already land in this branch. These are integration preservation tests, not
new red regressions. Essential walks use mock or the in-process test substrate; only
the explicitly optional runtime checks may require manylatents.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import io
import os
import shutil
import subprocess
import sys
import threading
import time
from importlib.resources import files
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import pytest
import yaml
from textual.widgets import Input, OptionList, Static

from manyruns import agents, app as cli, catalog, datasetfetch, env, figspec, inspected, store, vocab
from manyruns.pipeline import loading
from manyruns.session import Session
from manyruns.tui import colorby, state
from manyruns.tui.app import ManyrunsApp, prewarm_multiprocessing
from manyruns.tui.figure import FigureScreen
from manyruns.tui.ledger import LedgerScreen
from manyruns.tui.roster import RosterScreen, RosterTable
from manyruns.tui.run import FiguresPane, LeaveRunScreen, RunScreen
from manyruns.tui.samplefetch import SampleFetchScreen


def drives(body):
    @functools.wraps(body)
    def run(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))
    return run


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for kind in ('dataset', 'recipe', 'metrics'):
        source = files('manyruns').joinpath('configs', kind)
        assert source.is_dir()
        destination = tmp_path / kind
        shutil.copytree(str(source), destination)
        assert destination.is_dir()
        monkeypatch.setenv(f'MANYRUNS_{kind.upper()}_DIR', str(destination))
    # This walk tests transport and records, not geometry; _smoke is not a metric switch.
    (tmp_path / 'metrics' / 'default.yaml').write_text('metrics: []\n')
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(tmp_path / 'drop'))
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    monkeypatch.setattr(inspected, 'HOME', tmp_path / 'inspection-cache')
    monkeypatch.setattr(agents, 'ask_available', lambda: False)
    requests = []

    def no_network(url, **kwargs):
        requests.append(url)
        raise AssertionError('a demo walk tried an uninjected download')

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', no_network)
    yield requests
    assert not requests, 'every possible sample fetch must use an injected transport'


async def until(pilot, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'demo walk did not reach the expected state'
        await pilot.pause(0.01)


def frame(app):
    return '\n'.join(strip.text for strip in app.screen._compositor.render_strips())


def records(tmp_path):
    root = tmp_path / 'outputs'
    assert root.is_dir()
    return list(store.read(root))


async def roster_ready(app, pilot):
    await until(pilot, lambda: isinstance(app.screen, RosterScreen)
                and bool(app.screen.entries) and app.screen._poll_timer is not None)
    roster = app.screen
    roster._poll_timer.pause()
    return roster


async def select_source(app, pilot, name):
    roster = app.screen
    roster.query_one('#filter', Input).value = name
    await pilot.pause()
    index = next(i for i, entry in enumerate(roster.on_screen)
                 if isinstance(entry, state.DataEntry) and entry.name == name)
    roster.query_one(RosterTable).move_cursor(row=index)
    await pilot.press('enter')


async def select_embed(app, pilot):
    await until(pilot, lambda: isinstance(app.screen, LedgerScreen))
    await pilot.pause()  # let the ledger finish its deferred initial highlight
    options = app.screen.query_one(OptionList)
    options.highlighted = options.get_option_index('embed')
    await pilot.press('enter')
    await until(pilot, lambda: isinstance(app.screen, RunScreen))
    return app.screen


async def finish_run(pilot, screen):
    deadline = time.monotonic() + 30
    while not screen._finished:
        assert time.monotonic() < deadline, 'run did not finish'
        if not screen.query_one(Input).disabled:
            screen.query_one(Input).focus()
            await pilot.press('a', 'enter')
        await pilot.pause(0.01)
    assert screen.error is None
    assert screen.results['ok'], screen.results
    assert screen.recipe['name'] == 'embed'
    await until(pilot, lambda: not screen.worker_thread.is_alive())
    return screen.results


async def run_again(app, pilot):
    await pilot.press('escape')
    assert isinstance(app.screen, LeaveRunScreen)
    await pilot.press('tab', 'enter')
    await until(pilot, lambda: isinstance(app.screen, LedgerScreen))
    await pilot.pause()
    options = app.screen.query_one(OptionList)
    assert options.get_option_at_index(options.highlighted).id == 'embed'


def write_h5ad(path, *, labelled=True, legacy=False):
    obs = pd.DataFrame(index=[f'cell-{i}' for i in range(12)])
    if labelled:
        obs['condition'] = ['control', 'treated'] * 6
        obs['day'] = [0, 1, 2] * 4
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(12, 5)) if labelled else rng.poisson(3, size=(12, 5)).astype(float)
    obj = anndata.AnnData(matrix, obs=obs)
    if legacy:
        with h5py.File(path, 'w') as handle:
            node = handle.create_group('X')
            node.attrs['h5sparse_format'] = 'csr'
            node.attrs['h5sparse_shape'] = obj.shape
            node['data'] = [1.0]
            node['indices'] = [0]
            node['indptr'] = [0] + [1] * 12
            handle['obs'] = np.array([(s.encode(), b'control') for s in obs.index],
                                     dtype=[('index', 'S8'), ('condition', 'S7')])
            handle['var'] = np.array([(f'g{i}'.encode(),) for i in range(5)],
                                     dtype=[('index', 'S2')])
    else:
        obj.write_h5ad(path)
    return obj


def test_production_sample_declaration_is_pinned():
    root = files('manyruns').joinpath('configs', 'dataset')
    assert root.is_dir()
    row = yaml.safe_load(root.joinpath('pbmc3k.yaml').read_text())
    handle = row['handle']
    assert handle['ref'] == 'data/pbmc3k_raw.h5ad'
    assert handle['url'] == 'https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad'
    assert handle['sha256'] == '89a96f1beaa2dd83a687666d3f19a4513ac27a2a2d12581fcd77afed7ea653a1'
    assert handle['bytes'] == 5855727


@drives
async def test_launch_creates_drop_folder_and_lists_unfetched_samples(tmp_path, isolated):
    assert not (tmp_path / 'drop').exists()
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    async with app.run_test() as pilot:
        roster = await roster_ready(app, pilot)
        assert (tmp_path / 'drop').is_dir()
        rows = {entry.name: entry for entry in roster.entries}
        assert len(rows) == 14
        assert rows['pbmc3k'].missing and rows['pbmc3k'].downloadable
        assert 'download' in rows['pbmc3k'].size.lower()
        assert not rows['synthetic_timecourse'].missing
        assert not isolated


@pytest.mark.parametrize('legacy', [False, True], ids=['modern', 'legacy'])
@drives
async def test_h5ad_drop_settles_without_moving_filter_or_selection(tmp_path, monkeypatch, legacy):
    source = tmp_path / 'own.h5ad'
    write_h5ad(source, legacy=legacy)
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    async with app.run_test() as pilot:
        roster = await roster_ready(app, pilot)
        roster.query_one('#filter', Input).value = 'swissroll'
        await pilot.pause()
        selected = roster.on_screen[roster.query_one(RosterTable).cursor_row].identity
        destination = tmp_path / 'drop' / 'swissroll-own.h5ad'
        reads = []
        inspect = state._h5ad_observation

        def observed(path, *args, **kwargs):
            reads.append(Path(path))
            return inspect(path, *args, **kwargs)

        monkeypatch.setattr(state, '_h5ad_observation', observed)
        shutil.copyfile(source, destination)
        roster.poll()
        assert not reads
        assert not any(entry.path == destination for entry in roster.entries)
        roster.poll()
        row = next(entry for entry in roster.entries if entry.path == destination)
        assert reads == [destination]
        assert (row.obs.n_obs, row.obs.n_vars) == (12, 5)
        assert '12' in row.size and '5' in row.size
        assert not row.missing and not row.pending
        assert roster.query_one('#filter', Input).value == 'swissroll'
        assert roster.on_screen[roster.query_one(RosterTable).cursor_row].identity == selected
        assert 'new data' in roster.query_one('#footnote', Static).visual.plain
        assert app.screen is roster and len(app.screen_stack) == 1


@drives
async def test_nested_drop_restarts_settling_when_a_child_grows(tmp_path):
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    async with app.run_test() as pilot:
        roster = await roster_ready(app, pilot)
        folder = tmp_path / 'drop' / 'own-samples'
        for name in ('T0', 'T1'):
            nested = folder / name
            nested.mkdir(parents=True)
            (nested / 'matrix.mtx').write_text(
                '%%MatrixMarket matrix coordinate real general\n5 3 1\n1 1 1\n')
            (nested / 'features.tsv').write_text(''.join(f'g{i}\tg{i}\n' for i in range(5)))
            (nested / 'barcodes.tsv').write_text('c1\nc2\nc3\n')
        roster.poll()
        assert not any(e.path == folder for e in roster.entries)
        (nested / 'matrix.mtx').write_text(
            '%%MatrixMarket matrix coordinate real general\n5 6 1\n1 1 1\n')
        (nested / 'barcodes.tsv').write_text(''.join(f'c{i}\n' for i in range(6)))
        roster.poll()
        assert not any(e.path == folder for e in roster.entries)
        roster.poll()
        entry = next(e for e in roster.entries if e.path == folder)
        assert not entry.pending and not entry.refusal
        assert app.screen is roster
        assert 'new data' in roster.query_one('#footnote', Static).visual.plain


@pytest.mark.parametrize('name,cap', [('swissroll', 3), ('tree_narrow', 3),
                                      ('arch_soft', 5), ('torus', 50)])
@drives
async def test_named_embed_keeps_bounds_and_settings_in_record_and_reopened_project(tmp_path, name, cap):
    """Mock executes the shared bound/record path without a process-local engine spy."""
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    async with app.run_test() as pilot:
        await roster_ready(app, pilot)
        await select_source(app, pilot, name)
        screen = await select_embed(app, pilot)
        result = await finish_run(pilot, screen)
        pca = next(step for step in result['steps'] if step['name'] == 'pca')
        assert pca['outcome'] == 'ok' and pca['params']['n_components'] == cap
        assert bool(pca.get('bounded')) == (cap != 50)
        if cap != 50:
            assert any(f'50 → {cap}' in note for note in result['caveats'])
        for step in result['steps'][:2]:
            assert step['outcome'] == 'skipped' and not step.get('unsupported')
        assert result['complete'] is False  # synthetic prep declines are truthful
    saved = cli.load_project(name.replace('_', '-'))
    expected = catalog.load_dataset(name).get('params', {})
    assert saved['dataset_name'] == name and saved['data_kwargs'] == expected
    reopened = cli._build_session(saved, argparse.Namespace(seed=42, data_kwargs=None), resume=False)
    assert reopened.dataset_name == name and reopened.data_kwargs == expected
    row, = records(tmp_path)
    assert row['dataset_name'] == name and row['data_kwargs'] == expected
    assert row['run_id'] == result['run_id'] and row['caveats'] == result['caveats']


@pytest.fixture
def sample(tmp_path, monkeypatch):
    source = tmp_path / 'tiny.h5ad'
    write_h5ad(source, labelled=False)
    payload = source.read_bytes()
    row = catalog.load_dataset('pbmc3k')
    row['handle'].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (tmp_path / 'dataset' / 'pbmc3k.yaml').write_text(yaml.safe_dump(row))
    calls = []

    def opener(url, **kwargs):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', opener)
    return row, payload, calls


@drives
async def test_fetch_progress_refreshes_ledger_runs_and_reuses_verified_file(tmp_path, monkeypatch, sample):
    row, payload, calls = sample
    held, release = threading.Event(), threading.Event()

    class Stream(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                held.set()
                assert release.wait(60), 'download barrier was not released'
            return super().read(min(size, len(payload) // 2))

    def opener(url, **kwargs):
        calls.append(url)
        return Stream(payload)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', opener)
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    modal = None
    try:
        async with app.run_test() as pilot:
            roster = await roster_ready(app, pilot)
            stale = next(e for e in roster.entries if e.name == 'pbmc3k')
            await select_source(app, pilot, 'pbmc3k')
            assert isinstance(app.screen, SampleFetchScreen) and not calls
            modal = app.screen
            await pilot.press('enter')
            await until(pilot, held.is_set)
            await until(pilot, lambda: f'{len(payload) // 2:,}' in
                        modal.query_one('#sample-status', Static).visual.plain)
            assert not (tmp_path / 'drop' / 'pbmc3k_raw.h5ad').exists()
            assert not app._run_reservations
            release.set()
            await until(pilot, lambda: isinstance(app.screen, LedgerScreen))
            ready = app.screen.entry
            assert ready is not stale and not ready.missing
            assert ready.path == (tmp_path / 'drop' / 'pbmc3k_raw.h5ad').resolve()
            assert (ready.obs.n_obs, ready.obs.n_vars) == (12, 5)
            result = await finish_run(pilot, await select_embed(app, pilot))
            assert result['status']['pca'] == 'ok'
            await run_again(app, pilot)
            await pilot.press('escape')
            assert app.screen is roster
            await select_source(app, pilot, 'pbmc3k')
            assert isinstance(app.screen, LedgerScreen)
            assert calls == [row['handle']['url']]
            assert ready.path.read_bytes() == payload
    finally:
        release.set()
        if modal is not None and modal._thread is not None:
            modal._thread.join(timeout=2)
            assert not modal._thread.is_alive()
    assert len(records(tmp_path)) == 1


@pytest.mark.parametrize('stale_ledger', [False, True], ids=['first-use', 'deleted-after-ledger'])
@drives
async def test_failed_fetch_is_actionable_and_never_starts_a_run(tmp_path, monkeypatch, sample, stale_ledger):
    row, payload, calls = sample
    destination = tmp_path / 'drop' / 'pbmc3k_raw.h5ad'
    if stale_ledger:
        destination.parent.mkdir()
        destination.write_bytes(payload)

    def offline(url, **kwargs):
        calls.append(url)
        raise OSError('offline test transport')

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', offline)
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    async with app.run_test() as pilot:
        roster = await roster_ready(app, pilot)
        if stale_ledger:
            roster.poll()  # settle the already present file before selecting it
        await select_source(app, pilot, 'pbmc3k')
        if stale_ledger:
            assert isinstance(app.screen, LedgerScreen)
            destination.unlink()
            options = app.screen.query_one(OptionList)
            options.highlighted = options.get_option_index('embed')
            await pilot.press('enter')
        assert isinstance(app.screen, SampleFetchScreen)
        await pilot.press('enter')
        modal = app.screen
        await until(pilot, lambda: 'offline test transport' in
                    modal.query_one('#sample-status', Static).visual.plain)
        text = (modal.query_one('#sample-details', Static).visual.plain
                + modal.query_one('#sample-status', Static).visual.plain)
        assert row['handle']['url'] in text
        assert destination.name in text and str(destination.resolve()) in text
        assert not app._run_reservations
        assert not any(isinstance(screen, RunScreen) for screen in app.screen_stack)
        assert destination.parent.is_dir()
        assert not list(destination.parent.iterdir())
        assert calls == [row['handle']['url']]
        await until(pilot, lambda: not modal._thread.is_alive())
    if (tmp_path / 'outputs').exists():
        assert records(tmp_path) == []


@pytest.fixture
def inproc_scatter(tmp_path, monkeypatch):
    """Select the existing in-process substrate; no manylatents function is patched.

    Keep the production builder and its input transport. Only Session's engine changes
    at construction; metadata loading, gate, records and figures are real.
    """
    recipe = catalog.load_recipe('embed')
    recipe['steps'] = [{'name': 'phate', 'group': 'latent',
                        'params': {'n_components': 2, 'knn': 3, 't': 1, 'mds': 'classic'}}]
    (tmp_path / 'recipe' / 'embed.yaml').write_text(yaml.safe_dump(recipe))

    init = Session.__init__

    def inproc(session, *args, **kwargs):
        kwargs['engine'] = vocab.INPROC
        init(session, *args, **kwargs)

    monkeypatch.setattr(Session, '__init__', inproc)
    prewarm_multiprocessing()


@pytest.mark.parametrize('source', ['synthetic_timecourse', 'own', 'empty'])
@drives
async def test_finished_scatter_colour_views_survive_navigation_and_export(tmp_path, inproc_scatter, source):
    if source == 'synthetic_timecourse':
        original = loading.time_course_anndata()
        keys = ['timepoint']
        name = source
    else:
        drop = tmp_path / 'drop'
        drop.mkdir()
        name = 'own.h5ad'
        original = write_h5ad(drop / name, labelled=source != 'empty')
        keys = [] if source == 'empty' else ['condition', 'day']
    labels_before, kind_before = loading.labels_of(original)
    app = ManyrunsApp(argparse.Namespace(engine='manylatents'))
    async with app.run_test(size=(140, 35)) as pilot:
        roster = await roster_ready(app, pilot)
        roster.poll()
        await select_source(app, pilot, name)
        screen = await select_embed(app, pilot)
        result = await finish_run(pilot, screen)
        assert result['status']['phate'] == 'ok'
        png = screen._selected_figure()[1]
        np.testing.assert_array_equal(figspec.load(png).get('obs_names'), original.obs_names)
        paths = [Path(png), Path(png).with_suffix('.spec.npz'), Path(png).with_suffix('.spec.json')]
        originals = {path: path.read_bytes() for path in paths}
        before_rows = records(tmp_path)
        for key in keys or [None]:
            await pilot.press('c')
            if key is None:
                await until(pilot, lambda: any('no metadata columns to colour by' in str(n.message)
                            for n in app._notifications))
                assert app.screen is screen
                assert len(screen.query_one(FiguresPane).found) == 1
                break
            await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
            assert [col['key'] for col in app.screen.metadata.columns] == keys
            options = app.screen.query_one(OptionList)
            options.highlighted = keys.index(key)
            count = len(screen.query_one(FiguresPane).found)
            await pilot.press('enter')
            await until(pilot, lambda: len(screen.query_one(FiguresPane).found) == count + 1)
            view = screen._selected_figure()[1]
            assert view != png and Path(view).is_file()
            spec = figspec.load(view)
            assert spec['color_by'] == key
            np.testing.assert_array_equal(spec['obs_names'], original.obs_names)
            await pilot.press('f7')
            previous = screen._selected_figure()[1]
            assert previous != view
            await pilot.press('f8')
            assert screen._selected_figure()[1] == view
            for gesture, expected in [('f', view), ('left', previous), ('right', view)]:
                await pilot.press(gesture)
                assert isinstance(app.screen, FigureScreen)
                viewer = app.screen
                assert viewer.figures[viewer.index][1] == expected
                assert viewer.query_one('#figure-path', Static).visual.plain == expected
            await pilot.press('escape')
            assert app.screen is screen and screen._selected_figure()[1] == view
        if keys:
            await pilot.press('s')
            await until(pilot, lambda: any('saved to' in str(n.message) for n in app._notifications))
            exports = tmp_path / 'figures'
            assert exports.is_dir()
            assert {path.suffix for path in exports.iterdir()} == {'.png', '.pdf', '.svg'}
        assert all(path.read_bytes() == contents for path, contents in originals.items())
        assert records(tmp_path) == before_rows  # a view is not another run
        metadata = colorby.load_metadata(screen.entry)
        np.testing.assert_array_equal(metadata.obs, original.obs)
        labels_after, kind_after = loading.labels_of(loading.load_array(screen.entry.path))
        np.testing.assert_array_equal(labels_before, labels_after)
        assert kind_before == kind_after


@pytest.fixture
def two_step_recipe(tmp_path):
    recipe = catalog.load_recipe('embed')
    recipe['steps'] = [
        {'name': 'phate', 'group': 'latent', 'params': {'n_components': 2}},
        {'name': 'pca', 'group': 'latent', 'params': {'n_components': 2}},
    ]
    (tmp_path / 'recipe' / 'embed.yaml').write_text(yaml.safe_dump(recipe))


@drives
async def test_run_again_records_distinct_attempts_and_cancels_a_waiting_restart(
        tmp_path, monkeypatch, two_step_recipe):
    """Hold parent-side close, so process offloading cannot bypass the lifetime barrier."""
    entered, release = threading.Event(), threading.Event()
    close = Session.close

    def held_close(session, **kwargs):
        if kwargs.get('abandoned'):
            entered.set()
            assert release.wait(60), 'abandoned close was not released'
        return close(session, **kwargs)

    monkeypatch.setattr(Session, 'close', held_close)
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    screens = []
    try:
        async with app.run_test(size=(140, 35)) as pilot:
            await roster_ready(app, pilot)
            await select_source(app, pilot, 'swissroll')
            for attempt in range(5):
                screen = await select_embed(app, pilot)
                screens.append(screen)
                if attempt == 3:
                    await until(pilot, lambda: not screen.query_one(Input).disabled)
                    await run_again(app, pilot)
                    await until(pilot, entered.is_set)
                    await pilot.press('enter')
                    await until(pilot, lambda: app._waiting_run is not None)
                    assert app.WAITING in frame(app)
                    assert screen.worker_thread.is_alive()
                    assert len(app._run_reservations) == 1
                    heartbeat = []
                    app.call_later(lambda: heartbeat.append(True))
                    await until(pilot, lambda: bool(heartbeat))
                    await pilot.press('escape')
                    assert app._waiting_run is None and app.WAITING not in frame(app)
                    release.set()
                    await until(pilot, lambda: not screen.worker_thread.is_alive())
                    await until(pilot, lambda: not app._run_reservations)
                    assert len(records(tmp_path)) == 4  # the cancelled start wrote nothing
                    assert not isinstance(app.screen, RunScreen)
                    await select_source(app, pilot, 'swissroll')
                else:
                    result = await finish_run(pilot, screen)
                    assert result['complete'] is True
                    if attempt < 4:
                        await run_again(app, pilot)
            await until(pilot, lambda: not app._run_reservations)
    finally:
        release.set()
        for screen in screens:
            screen.worker_thread.join(timeout=2)
            assert not screen.worker_thread.is_alive()
    rows = records(tmp_path)
    assert len(rows) == 5 and len({row['run_id'] for row in rows}) == 5
    abandoned, = [row for row in rows if not row['complete']]
    assert any('run abandoned' in note for note in abandoned['caveats'])
    assert any('step 1 (pca): not run' in note for note in abandoned['caveats'])
    assert [step['name'] for step in abandoned['steps']] == ['phate']
    assert all(screen.worker_thread.name == 'manyruns-run' for screen in screens)


def test_quit_signal_uses_environment_accessor(tmp_path, monkeypatch):
    """The quit check must use the same override accessor as the application."""
    reads, launches = [], []
    get = env.get

    def read(name, default=None):
        reads.append(name)
        return get(name, default)

    monkeypatch.delenv('MANYRUNS_TEST_QUIT_SIGNAL', raising=False)
    monkeypatch.setattr(env, 'get', read)
    monkeypatch.setattr(sys.modules[__name__], 'assert_quit_process_exits', launches.append)
    test_finalization_status_repaints_and_quit_returns_during_a_wait(
        tmp_path, monkeypatch, None)
    assert reads == ['TEST_QUIT_SIGNAL']
    assert launches == [tmp_path]


def assert_quit_process_exits(tmp_path):
    """Time process exit after the child reaches Ctrl+C, excluding app startup."""
    quitting = tmp_path / 'quitting'
    output = tmp_path / 'quit-process.log'
    # CI supplies a relative PYTHONPATH; the child starts in a different directory.
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    env = {**os.environ, 'MANYRUNS_TEST_QUIT_SIGNAL': str(quitting),
           'PYTHONPATH': os.pathsep.join((str(root), *sys.path))}
    node = (f'{Path(__file__).resolve()}::'
            'test_finalization_status_repaints_and_quit_returns_during_a_wait')
    with output.open('w') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'pytest', node, '-q', '-s', '--tb=short'],
            cwd=tmp_path, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while not quitting.exists() and process.poll() is None:
                assert time.monotonic() < deadline, (
                    'child did not reach the waiting restart\n' + output.read_text())
                time.sleep(0.01)
            assert quitting.exists(), output.read_text()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pytest.fail('quit left the process running with its worker blocked\n'
                            + output.read_text())
            assert process.returncode == 0, output.read_text()
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


@drives
async def test_finalization_status_repaints_and_quit_returns_during_a_wait(
        tmp_path, monkeypatch, two_step_recipe):
    """The child quits with Session.close held through interpreter shutdown."""
    quitting = env.get('TEST_QUIT_SIGNAL')
    if quitting is None:
        assert_quit_process_exits(tmp_path)
        return

    entered, release = threading.Event(), threading.Event()
    close = Session.close

    def held_close(session, **kwargs):
        entered.set()
        release.wait()  # Deliberately never released, including during test teardown.
        return close(session, **kwargs)

    monkeypatch.setattr(Session, 'close', held_close)
    app = ManyrunsApp(argparse.Namespace(engine='mock'))
    async with app.run_test(size=(140, 35)) as pilot:
        await roster_ready(app, pilot)
        await select_source(app, pilot, 'swissroll')
        screen = await select_embed(app, pilot)
        await until(pilot, lambda: not screen.query_one(Input).disabled)
        await pilot.press('a', 'enter')
        await until(pilot, entered.is_set)
        await until(pilot, lambda: 'measuring geometry' in frame(app) and 'elapsed' in frame(app))
        assert not screen._finished
        await run_again(app, pilot)
        await pilot.press('enter')
        await until(pilot, lambda: app._waiting_run is not None)
        assert app.WAITING in frame(app)
        assert screen.worker_thread.is_alive() and not release.is_set()
        Path(quitting).touch()
        await pilot.press('ctrl+c')
        await until(pilot, lambda: not app.is_running, timeout=1)
    assert screen.worker_thread.is_alive() and not release.is_set()


def test_optional_runtime_generator_fit_dimensions():
    """Optional preservation oracle; essential UI walks above never skip for this engine."""
    api = pytest.importorskip('manylatents.api', reason='optional installed-engine dimension oracle')
    root = catalog.dataset_dir()
    assert root.is_dir()
    generators = [row for row in catalog.load_datasets() if row['handle']['kind'] == 'manylatents']
    assert len(generators) == 12
    for row in generators:
        dm = api._resolve_datamodule(data=row['handle']['ref'], seed=42, **row.get('params', {}))
        dm.setup()
        assert loading._stack_dataset(dm.train_dataset).shape == (
            row['dims']['n_samples'], row['dims']['n_features']), row['name']


@pytest.mark.parametrize('source', ['synthetic_timecourse', 'own'])
def test_optional_runtime_cli_pca_colour_and_branch(tmp_path, source):
    """Real CLI → runner plus reopened session/branch; local inputs and an empty metric suite."""
    pytest.importorskip('manylatents.api', reason='optional installed-engine PCA/branch walk')
    recipe = catalog.load_recipe('embed')
    recipe['steps'] = [step for step in recipe['steps'] if step['name'] == 'pca']
    assert len(recipe['steps']) == 1
    (tmp_path / 'recipe' / 'embed.yaml').write_text(yaml.safe_dump(recipe))
    if source == 'own':
        path = tmp_path / 'own.h5ad'
        original = write_h5ad(path)
        entry = state.local_entry(path)
        source_arg, key = str(path), 'condition'
    else:
        original = loading.time_course_anndata()
        entry = state.catalog_entry(source)
        source_arg, key = source, 'timepoint'
    assert entry is not None and not entry.refusal
    prewarm_multiprocessing()
    assert cli.main(['run', source_arg, '--project', 'runtime', '--recipe', 'embed',
                     '--engine', 'manylatents', '--color-by', key]) == 0
    row, = records(tmp_path)
    assert row['ok'] and row['complete']
    step, = row['steps']
    assert step['name'] == 'pca' and step['outcome'] == 'ok'
    assert step['params']['n_components'] == 5
    assert any('50 → 5' in note for note in row['caveats'])
    png, = step['plots']
    before = {path: path.read_bytes() for path in (
        Path(png), Path(png).with_suffix('.spec.json'), Path(png).with_suffix('.spec.npz'))}
    assert figspec.load(png)['color_by'] == key
    view, positional = colorby.recolor_figure(png, colorby.load_metadata(entry), key)
    assert not positional and view != png
    np.testing.assert_array_equal(figspec.load(view)['obs_names'], original.obs_names)
    assert all(path.read_bytes() == contents for path, contents in before.items())

    args = argparse.Namespace(seed=42, data_kwargs=None, time_key=None, color_by=[key])
    session = cli._build_session(cli.load_project('runtime'), args, resume=False)
    session.step('pca')
    assert session.steps[-1]['outcome'] == 'ok'
    labels, kind = loading.labels_of(original)
    np.testing.assert_array_equal(session.labels, labels)
    assert session.label_kind == kind
    child = session.branch(0)
    child.step({'name': 'pca', 'group': 'latent', 'params': {'n_components': 2}})
    assert child.steps[-1]['outcome'] == 'ok'
    np.testing.assert_array_equal(child.labels, labels)
    assert child.label_kind == kind
    for lineage in (session, child):
        store.append(lineage.close())
    rows = records(tmp_path)
    assert len(rows) == 3 and len({item['run_id'] for item in rows}) == 3
    assert rows[-1]['parent'] == {'run_id': session.run_id, 'index': 0}
    assert rows[-1]['g_vector']['final_dim'] == 2
