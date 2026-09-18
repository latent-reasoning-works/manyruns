"""First-use selection is explicit; all sample bytes use an injected local transport."""
from __future__ import annotations

import asyncio
import functools
import hashlib
import io
import threading
import time

import anndata
import numpy as np
import pytest
import yaml
from textual.widgets import Static

from manyruns import catalog, datasetfetch, inspected
from manyruns.tui import samplefetch, state
from manyruns.tui.app import ManyrunsApp
from manyruns.tui.ledger import LedgerScreen
from manyruns.tui.roster import RosterScreen, RosterTable


def pilot_test(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


@pytest.fixture
def sample(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    drop = tmp_path / 'drop'
    drop.mkdir()
    configs = tmp_path / 'catalog'
    configs.mkdir()
    assert configs.is_dir()
    monkeypatch.delenv('MANYRUNS_DATASET_DIR', raising=False)
    ds = catalog.load_dataset('pbmc3k')
    path = tmp_path / 'tiny.h5ad'
    if getattr(request, 'param', 'modern') == 'legacy':
        import h5py
        with h5py.File(path, 'w') as handle:
            node = handle.create_group('X')
            node.attrs['h5sparse_format'] = 'csr'
            node.attrs['h5sparse_shape'] = (4, 3)
            node['data'] = [1.]
            node['indices'] = [0]
            node['indptr'] = [0, 1, 1, 1, 1]
            handle['obs'] = np.array([(b'c1', b'control'), (b'c2', b'control'),
                                     (b'c3', b'treated'), (b'c4', b'treated')],
                                    dtype=[('index', 'S2'), ('condition', 'S7')])
            handle['var'] = np.array([(b'g1',), (b'g2',), (b'g3',)], dtype=[('index', 'S2')])
        assert anndata.read_h5ad(path).shape == (4, 3)
    else:
        anndata.AnnData(np.ones((4, 3))).write_h5ad(path)
    payload = path.read_bytes()
    ds['handle'].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (configs / 'pbmc3k.yaml').write_text(yaml.safe_dump(ds))
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(drop))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    monkeypatch.setattr(inspected, 'HOME', tmp_path / 'cache')
    calls = []

    def opener(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', opener)
    return drop, ds, payload, calls


async def until(pilot, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'UI did not reach the expected state'
        await pilot.pause(0.01)


def metadata_reads(monkeypatch):
    """Observe the actual header and obs readers, including H5AD's shell bypass."""
    reads = {name: [] for name in ('_check_readable', '_h5ad_observation')}
    for name, calls in reads.items():
        original = getattr(state, name)

        def tracked(*args, _original=original, _calls=calls, **kwargs):
            _calls.append(args[0])
            return _original(*args, **kwargs)

        monkeypatch.setattr(state, name, tracked)
    return reads


@pytest.mark.parametrize('sample', ['modern', 'legacy'], indirect=True)
@pilot_test
async def test_back_never_fetches_and_success_opens_newly_resolved_ledger(sample, monkeypatch):
    import h5py

    getitem = h5py.Dataset.__getitem__
    def no_matrix_payload(node, *args, **kwargs):
        assert node.name != '/X' and not node.name.startswith(('/X/', '/layers/'))
        return getitem(node, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, '__getitem__', no_matrix_payload)
    drop, ds, _, calls = sample
    assert drop.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        stale = roster.entries[0]
        await pilot.press('enter')
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        text = app.screen.query_one('#sample-details', Static).visual.plain
        assert ds['handle']['url'] in text
        assert str(drop / 'pbmc3k_raw.h5ad') in text
        assert str(ds['handle']['bytes']) in text
        await pilot.press('escape')
        assert app.screen is roster and calls == []
        await pilot.press('enter', 'enter')
        await until(pilot, lambda: isinstance(app.screen, LedgerScreen))
        entry = app.screen.entry
        assert entry is not stale and entry.path == drop / 'pbmc3k_raw.h5ad'
        assert entry.path.is_absolute() and not entry.missing
        assert (entry.obs.n_obs, entry.obs.n_vars) == (4, 3)
        if entry.obs.conditions:
            assert entry.obs.conditions == ['control', 'treated']
        assert len(calls) == 1 and 0 < calls[0][1] <= 30
        await pilot.press('escape')
        await pilot.pause()
        index = next(i for i, e in enumerate(roster.on_screen)
                     if isinstance(e, state.DataEntry) and e.kind == 'bundled')
        roster.query_one(RosterTable).move_cursor(row=index)
        await pilot.press('enter')
        await until(pilot, lambda: isinstance(app.screen, LedgerScreen))
        assert len(calls) == 1


@pilot_test
async def test_fetch_declaration_changed_during_selection_is_refused(sample, tmp_path, monkeypatch):
    _, _, _, calls = sample
    configs = tmp_path / 'catalog'
    assert configs.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        original = samplefetch._resolve

        def changed(entry):
            current = original(entry)
            (configs / 'pbmc3k.yaml').write_text('- invalid declaration\n')
            return current

        monkeypatch.setattr(samplefetch, '_resolve', changed)
        await pilot.press('enter')
        assert app._exception is None
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert 'Could not read' in app.screen.query_one('#sample-details', Static).visual.plain
        assert not app.screen.query('#fetch-sample') and not calls
        await pilot.press('escape')
        assert app.screen is roster and not app._samplefetch_active


@pilot_test
async def test_verified_but_unreadable_sample_stays_in_refusal_modal(sample, tmp_path, monkeypatch):
    drop, ds, _, calls = sample
    assert drop.is_dir()
    configs = tmp_path / 'catalog'
    assert configs.is_dir()
    payload = b'not an HDF5 file'
    ds['handle'].update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (configs / 'pbmc3k.yaml').write_text(yaml.safe_dump(ds))

    def opener(url, *, timeout):
        calls.append(url)
        return io.BytesIO(payload)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', opener)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await pilot.press('enter', 'enter')
        modal = app.screen
        await until(pilot, lambda: modal._download_path is not None)
        modal._settle_timer.pause()
        modal.call_later(modal._settle_download)
        await pilot.pause()
        assert app.screen is modal and app._exception is None
        assert 'Could not read' in modal.query_one('#sample-status', Static).visual.plain
        assert (drop / 'pbmc3k_raw.h5ad').read_bytes() == payload and len(calls) == 1
        await pilot.press('escape')
        assert app.screen is roster and not app._samplefetch_active


@pilot_test
async def test_hash_failure_stays_put_and_retry_is_explicit(sample, monkeypatch):
    drop, _, payload, calls = sample
    assert drop.is_dir()
    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen',
                        lambda *a, **k: io.BytesIO(b'x' * len(payload)))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press('enter', 'enter')
        modal = app.screen
        await until(pilot, lambda: 'sha256 mismatch' in
                    modal.query_one('#sample-status', Static).visual.plain)
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert not list(drop.iterdir())
        assert 'put the file at' in modal.query_one('#sample-status', Static).visual.plain
        monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen',
                            lambda *a, **k: io.BytesIO(payload))
        await pilot.press('enter')
        await until(pilot, lambda: isinstance(app.screen, LedgerScreen))


@pilot_test
async def test_missing_local_and_non_url_rows_cannot_open_a_ledger(sample):
    drop, _, _, calls = sample
    assert drop.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ready = []
        row = state.DataEntry(name='removed.csv', kind='file',
                              obs=state.Observation(shape='single', modality='tabular'),
                              path=drop / 'removed.csv')
        samplefetch.prepare_entry(app, row, ready.append)
        await pilot.pause()
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert 'not on this machine' in app.screen.query_one('#sample-details', Static).visual.plain
        assert not app.screen.query('#fetch-sample')
        await pilot.press('escape')
        assert not ready and not calls


@pilot_test
async def test_progress_concurrent_enter_and_dismissal_cancel_without_late_navigation(sample, monkeypatch):
    drop, _, payload, calls = sample
    assert drop.is_dir()
    reading = threading.Event()
    release = threading.Event()
    stopped = threading.Event()

    class Slow(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                reading.set()
                assert release.wait(60), 'test did not release transport'
            return super().read(min(size, len(payload) // 2))

        def close(self):
            stopped.set()
            super().close()

    def opener(url, *, timeout):
        calls.append(url)
        return Slow(payload)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', opener)
    app = ManyrunsApp()
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            roster = app.screen
            roster._poll_timer.pause()
            await pilot.press('enter', 'enter')
            modal = app.screen
            await until(pilot, reading.is_set)
            assert modal._thread.daemon
            fetch_thread = modal._thread
            await until(pilot, lambda: f'{len(payload) // 2:,}' in
                        modal.query_one('#sample-status', Static).visual.plain)
            await pilot.press('enter', 'enter', 'enter')
            # The downloader's lock can hide duplicate UI workers from opener counts.
            assert modal._thread is fetch_thread
            assert app.screen is modal and len(calls) == 1
            roster.poll()
            from manyruns.tui import dropwatch
            assert not dropwatch.snapshot().entries
            assert not any(e.kind == 'file' for e in roster.entries)
            assert not any(p.name == 'pbmc3k_raw.h5ad' for p in drop.iterdir())
            await pilot.press('escape')
            assert app.screen is roster
            release.set()
            await until(pilot, stopped.is_set)
            await until(pilot, lambda: not modal._thread.is_alive())
            assert not list(drop.iterdir())
            assert app.screen is roster and len(app.screen_stack) == 1
    finally:
        release.set()


@pilot_test
async def test_exit_does_not_join_a_download_thread(sample, monkeypatch):
    drop, _, payload, _ = sample
    assert drop.is_dir()
    reading, release = threading.Event(), threading.Event()

    class Blocked(io.BytesIO):
        def read(self, size=-1):
            reading.set()
            assert release.wait(60), 'test did not release transport'
            return super().read(size)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', lambda *a, **k: Blocked(payload))
    app = ManyrunsApp()
    modal = None
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press('enter', 'enter')
            modal = app.screen
            await until(pilot, reading.is_set)
            app.exit()
        # The transport is still blocked after run_test has returned: no hidden thread join.
        assert modal._thread.is_alive()
    finally:
        release.set()
        if modal is not None:
            deadline = time.monotonic() + 3
            while modal._thread.is_alive():
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
    assert not list(drop.iterdir())


@pilot_test
async def test_non_url_catalog_ghost_offers_placement_and_no_fetch(sample):
    drop, ds, _, calls = sample
    assert drop.is_dir()
    ghost = dict(ds, name='ghost', handle={'kind': 'path', 'ref': 'data/ghost.h5ad'})
    configs = catalog.dataset_dir()
    assert configs.is_dir()
    (configs / 'ghost.yaml').write_text(yaml.safe_dump(ghost))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        row = state.catalog_entry('ghost')
        assert row.missing and not row.downloadable and row.size == 'not on this machine'
        samplefetch.prepare_entry(app, row, lambda _: pytest.fail('missing ghost navigated'))
        await pilot.pause()
        text = app.screen.query_one('#sample-details', Static).visual.plain
        assert str(drop / 'ghost.h5ad') in text
        assert not app.screen.query('#fetch-sample')
        await pilot.press('enter')
        assert isinstance(app.screen, RosterScreen) and not calls


@pytest.mark.parametrize('damage', ['removed', 'malformed'])
@pilot_test
async def test_stale_catalog_failure_refuses_and_releases_selection(sample, damage):
    _, declaration, _, calls = sample
    configs = catalog.dataset_dir()
    assert configs.is_dir()
    config = configs / 'pbmc3k.yaml'
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        row = roster.entries[0]
        assert row.missing and row.downloadable
        if damage == 'removed':
            config.unlink()
        else:
            config.write_text('handle: [unterminated')
        # Choose synchronously so an escaping exception fails at the selection boundary.
        roster.choose()
        await pilot.pause()
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert 'Could not read pbmc3k' in app.screen.query_one('#sample-details', Static).visual.plain
        assert not app.screen.query('#fetch-sample') and not calls
        assert app._samplefetch_active and roster._selection_active
        await pilot.press('escape')
        assert app.screen is roster
        assert not app._samplefetch_active and not roster._selection_active
        config.write_text(yaml.safe_dump(declaration))
        roster.poll(force=True)
        await pilot.press('enter')
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert app.screen.query('#fetch-sample')
        await pilot.press('escape')
        assert not app._samplefetch_active and not roster._selection_active and not calls


@pilot_test
async def test_generated_refs_need_no_file_and_stale_local_selection_rechecks(sample, monkeypatch):
    drop, _, _, calls = sample
    assert drop.is_dir()
    monkeypatch.delenv('MANYRUNS_DATASET_DIR')
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        ready = []
        generated = state.catalog_entry('synthetic_timecourse')
        samplefetch.prepare_entry(app, generated, ready.append)
        assert len(ready) == 1 and not ready[0].missing
        path = drop / 'mine.csv'
        path.write_text('a,b\n1,2\n')
        local = state.local_entry(path)
        path.unlink()
        samplefetch.prepare_entry(app, local, ready.append)
        await pilot.pause()
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert len(ready) == 1 and not calls


@pilot_test
async def test_copying_catalog_selection_offers_only_back(sample, monkeypatch):
    drop, _, payload, calls = sample
    assert drop.is_dir()
    path = drop / 'pbmc3k_raw.h5ad'
    path.write_bytes(payload)
    pending = state.catalog_entry('pbmc3k', skip={path})
    assert pending.pending and pending.downloadable
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        roster.entries = [pending]
        roster.refresh_rows()
        reads = metadata_reads(monkeypatch)
        await pilot.press('enter')
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert not app.screen.query('#fetch-sample')
        assert not any(reads.values()), 'copying alias was inspected'
        await pilot.press('enter')
        assert app.screen is roster and not calls
        roster.poll()
        assert all(reads.values()), 'settled metadata must exercise both spies'


@pilot_test
async def test_existing_file_returned_by_fetch_must_settle_before_inspection(sample, monkeypatch):
    drop, _, payload, calls = sample
    assert drop.is_dir()
    path = drop / 'pbmc3k_raw.h5ad'
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await pilot.press('enter')
        modal = app.screen
        # A manual copy wins after the missing-source modal has opened.
        path.write_bytes(payload)
        reads = metadata_reads(monkeypatch)
        await pilot.press('enter')
        await until(pilot, lambda: not modal._fetching)
        assert app.screen is modal and not any(reads.values()) and not calls
        modal._settle_timer.pause()
        # Growing after completion must restart the two-observation wait.
        anndata.AnnData(np.ones((6, 3))).write_h5ad(path)
        modal._settle_download()
        assert app.screen is modal and not any(reads.values())
        modal._settle_download()
        await until(pilot, lambda: isinstance(app.screen, LedgerScreen))
        assert app.screen.entry.obs.n_obs == 6 and not calls
        assert all(reads.values()), 'settled metadata must exercise both spies'


@pilot_test
async def test_enter_on_back_cancels_a_blocked_download(sample, monkeypatch):
    from textual.widgets import Button
    drop, _, payload, calls = sample
    assert drop.is_dir()
    reading, release = threading.Event(), threading.Event()

    class Blocked(io.BytesIO):
        def read(self, size=-1):
            reading.set()
            assert release.wait(60), 'test did not release transport'
            return super().read(size)

    def opener(*args, **kwargs):
        calls.append(1)
        return Blocked(payload)

    monkeypatch.setattr(datasetfetch.urllib.request, 'urlopen', opener)
    app = ManyrunsApp()
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            roster = app.screen
            roster._poll_timer.pause()
            await pilot.press('enter', 'enter')
            modal = app.screen
            await until(pilot, reading.is_set)
            modal.query_one('#back-sample', Button).focus()
            await pilot.press('enter')
            assert app.screen is roster and modal._cancelled.is_set()
            release.set()
            await until(pilot, lambda: not modal._thread.is_alive())
            assert app.screen is roster and len(calls) == 1
            assert not list(drop.iterdir())
    finally:
        release.set()


@pilot_test
async def test_download_becoming_a_cycle_during_settling_offers_retry(sample):
    from textual.widgets import Button

    drop, _, _, calls = sample
    assert drop.is_dir()
    path = drop / 'pbmc3k_raw.h5ad'
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await pilot.press('enter', 'enter')
        modal = app.screen
        await until(pilot, lambda: modal._settle_timer is not None)
        modal._settle_timer.pause()
        assert len(calls) == 1
        path.unlink()
        path.symlink_to(path)
        modal.call_later(modal._settle_download)
        await pilot.pause()
        assert app._exception is None and app.screen is modal
        assert modal._settle_timer is None
        retry = modal.query_one('#fetch-sample', Button)
        assert not retry.disabled and str(retry.label) == 'Retry'
        assert 'Could not read the downloaded data' in str(
            modal.query_one('#sample-status', Static).content)
        assert str(path) in str(modal.query_one('#sample-status', Static).content)
        await pilot.press('escape')
        assert app.screen is roster and not app._samplefetch_active


SETTLE_FS_PRIMITIVES = [
    'os.stat', 'os.lstat', 'os.scandir', 'os.listdir', 'os.readlink',
    'io.open',
    'Path.resolve', 'Path.exists', 'Path.is_file', 'Path.is_dir',
    'Path.resolve:ValueError',
]


async def _sample_callback_fault(sample, monkeypatch, entry_point, primitive):
    import os
    from pathlib import Path
    from textual.widgets import Button

    drop, _, payload, _ = sample
    assert drop.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await pilot.press('enter')
        modal = app.screen
        path = drop / 'pbmc3k_raw.h5ad'
        target = drop.parent / 'returned.h5ad'
        target.write_bytes(payload)
        path.symlink_to(target)
        # Exercise nested fingerprint lstat/scandir as well as catalog/path resolution.
        cohort = drop / 'cohort'
        cohort.mkdir()
        assert cohort.is_dir()
        (cohort / 'matrix.csv').write_text('a,b\n1,2\n')
        modal._download_path = path
        namespace, name = primitive.split(':')[0].split('.')
        owner = {'os': os, 'io': io, 'Path': Path}[namespace]
        original = getattr(owner, name)
        hits = []

        def fail_once(*args, **kwargs):
            if not hits:
                hits.append(args)
                error = ValueError if primitive.endswith(':ValueError') else OSError
                raise error(f'injected {primitive}')
            return original(*args, **kwargs)

        def tick():
            with monkeypatch.context() as patch:
                patch.setattr(owner, name, fail_once)
                if entry_point == '_settle_download':
                    modal._settle_download()
                else:
                    modal.on_sample_fetch_screen_finished(modal.Finished(None, path))

        modal.call_later(tick)
        await pilot.pause()
        assert hits, f'{primitive} was not reached from {entry_point}'
        assert app._exception is None and app.screen is modal
        assert modal.query_one('#back-sample', Button).disabled is False
        await pilot.press('escape')
        assert app.screen is roster and not app._samplefetch_active


@pytest.mark.parametrize('primitive', SETTLE_FS_PRIMITIVES)
@pilot_test
async def test_settle_callback_filesystem_faults(sample, monkeypatch, primitive):
    await _sample_callback_fault(sample, monkeypatch, '_settle_download', primitive)


@pytest.mark.parametrize('primitive', SETTLE_FS_PRIMITIVES)
@pilot_test
async def test_finished_callback_filesystem_faults(sample, monkeypatch, primitive):
    await _sample_callback_fault(sample, monkeypatch, 'finished', primitive)


@pytest.mark.parametrize('primitive', [
    'os.stat', 'os.lstat', 'os.readlink', 'os.mkdir', 'os.replace', 'os.link', 'os.unlink',
    'Path.resolve', 'Path.exists', 'Path.is_file', 'Path.is_dir', 'Path.resolve:ValueError',
])
@pilot_test
async def test_fetch_worker_filesystem_faults(sample, monkeypatch, primitive):
    import errno
    import os
    from pathlib import Path
    from textual.widgets import Button

    drop, _, payload, _ = sample
    assert drop.is_dir()
    alias = drop.parent / 'drop-link'
    alias.symlink_to(drop, target_is_directory=True)
    assert alias.is_dir()
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(alias))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await pilot.press('enter')
        modal = app.screen
        if primitive == 'Path.is_file':
            # The existing-file branch is the worker's only is_file call.
            (drop / 'pbmc3k_raw.h5ad').write_bytes(payload)
        namespace, name = primitive.split(':')[0].split('.')
        owner = os if namespace == 'os' else Path
        original = getattr(owner, name)
        hits = []

        def fail_once(*args, **kwargs):
            if not hits:
                hits.append(args)
                error = ValueError if primitive.endswith(':ValueError') else OSError
                raise error(f'injected {primitive}')
            return original(*args, **kwargs)

        def worker_tick():
            # Run the worker body synchronously with injected transport so process-global
            # primitive patches cannot also fault Textual or a concurrent background tick.
            with monkeypatch.context() as patch:
                if primitive == 'os.replace':
                    # Atomic rename publishes only when hard links are unsupported.
                    def no_links(*args, **kwargs):
                        raise OSError(errno.ENOTSUP, 'hard links unsupported')
                    patch.setattr(os, 'link', no_links)
                patch.setattr(owner, name, fail_once)
                modal._fetch()

        modal.call_later(worker_tick)
        await pilot.pause()
        assert hits, f'{primitive} was not reached by the worker'
        assert app._exception is None and app.screen is modal
        assert not modal._fetching
        assert not modal.query_one('#back-sample', Button).disabled
        await pilot.press('escape')
        assert app.screen is roster and app._exception is None


async def _progress_without_filesystem(sample, monkeypatch, worker):
    import os
    from pathlib import Path

    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await pilot.press('enter')
        modal = app.screen
        calls = []

        def unavailable(*args, **kwargs):
            calls.append(args)
            raise OSError('progress must not consult the filesystem')

        def tick():
            with monkeypatch.context() as patch:
                for name in ('stat', 'lstat', 'scandir', 'listdir', 'readlink'):
                    patch.setattr(os, name, unavailable)
                for name in ('resolve', 'exists', 'is_file', 'is_dir'):
                    patch.setattr(Path, name, unavailable)
                if worker:
                    modal._progress(12, 24)
                else:
                    modal.on_sample_fetch_screen_progress(modal.Progress(12, 24))

        modal.call_later(tick)
        await pilot.pause()
        assert not calls and app._exception is None
        assert '12 / 24 bytes' in str(modal.query_one('#sample-status', Static).content)
        await pilot.press('escape')
        assert app.screen is roster


@pilot_test
async def test_worker_progress_callback_uses_no_filesystem(sample, monkeypatch):
    """Preservation: no filesystem primitive is reachable from this callback."""
    await _progress_without_filesystem(sample, monkeypatch, worker=True)


@pilot_test
async def test_ui_progress_callback_uses_no_filesystem(sample, monkeypatch):
    """Preservation: no filesystem primitive is reachable from this callback."""
    await _progress_without_filesystem(sample, monkeypatch, worker=False)


@pytest.mark.parametrize('error', [TypeError, AttributeError, RuntimeError])
@pytest.mark.parametrize('callback', ['_fetch', '_settle_download', 'finished', 'progress'])
@pilot_test
async def test_sample_callbacks_do_not_swallow_programming_errors(
        sample, monkeypatch, error, callback):
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        await pilot.press('enter')
        modal = app.screen

        def broken(*args, **kwargs):
            raise error('programming defect')

        with monkeypatch.context() as patch:
            if callback == '_fetch':
                patch.setattr(datasetfetch, 'fetch_dataset', broken)
                invoke = modal._fetch
            elif callback in {'_settle_download', 'finished'}:
                patch.setattr(samplefetch.dropwatch, 'snapshot', broken)
                invoke = (modal._settle_download if callback == '_settle_download' else
                          lambda: modal.on_sample_fetch_screen_finished(modal.Finished(None)))
            else:
                patch.setattr(modal, 'query_one', broken)
                def invoke():
                    modal.on_sample_fetch_screen_progress(modal.Progress(1, 2))
            with pytest.raises(error, match='programming defect'):
                invoke()
        await pilot.press('escape')


@pytest.mark.parametrize('primitive', ['resolve', 'exists', 'is_file', 'is_dir'])
@pilot_test
async def test_selection_completion_callback_filesystem_faults(sample, monkeypatch, primitive):
    from pathlib import Path

    drop, _, _, _ = sample
    assert drop.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        completed, hits = [], []

        def ready(entry):
            def fail_once(*args, **kwargs):
                hits.append(1)
                raise OSError(f'injected {primitive}')
            with monkeypatch.context() as patch:
                patch.setattr(Path, primitive, fail_once)
                getattr(drop, primitive)()

        samplefetch.prepare_entry(app, roster.entries[0], ready,
                                  on_done=lambda: completed.append(1))
        await pilot.pause()
        modal = app.screen
        def complete_selection():
            modal.dismiss(roster.entries[0])
        modal.call_later(complete_selection)
        await pilot.pause()
        assert hits and completed == [1]
        assert app.screen is roster and app._exception is None and not app._samplefetch_active
        await pilot.press('p')
        from textual.widgets import Input
        assert roster.query_one('#filter', Input).value == 'p'


@pytest.mark.parametrize('error', [TypeError, AttributeError])
@pilot_test
async def test_prepare_entry_does_not_swallow_programming_errors(sample, monkeypatch, error):
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()

        def broken(*args, **kwargs):
            raise error('programming defect')

        with monkeypatch.context() as patch:
            patch.setattr(samplefetch, '_resolve', broken)
            with pytest.raises(error, match='programming defect'):
                samplefetch.prepare_entry(app, roster.entries[0], lambda entry: None)


@pilot_test
async def test_selection_survives_a_cyclic_download_destination(sample):
    drop, _, _, calls = sample
    assert drop.is_dir()
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        drop.rmdir()
        drop.symlink_to(drop)
        roster.call_later(roster.choose)
        await pilot.pause()
        assert app._exception is None
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert not app.screen.query('#fetch-sample')
        assert 'Could not read pbmc3k' in str(
            app.screen.query_one('#sample-details', Static).content)
        await pilot.press('escape')
        assert app.screen is roster and not app._samplefetch_active
        assert not roster._selection_active and not calls


@pytest.mark.parametrize('primitive', [
    'os.stat', 'os.lstat', 'os.scandir', 'os.listdir', 'os.readlink', 'os.mkdir', 'os.replace',
    'io.open', 'Path.resolve', 'Path.exists', 'Path.is_file', 'Path.is_dir',
    'Path.resolve:ValueError',
])
@pilot_test
async def test_prepare_callback_filesystem_faults(sample, monkeypatch, primitive):
    import os
    from pathlib import Path

    drop, _, payload, calls = sample
    assert drop.is_dir()
    path = drop / 'pbmc3k_raw.h5ad'
    path.write_bytes(payload)
    alias = drop.parent / 'drop-link'
    alias.symlink_to(drop, target_is_directory=True)
    assert alias.is_dir()
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(alias))
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        roster.poll()
        row = next(e for e in roster.entries if e.kind == 'file')
        roster.query_one(RosterTable).move_cursor(row=roster.on_screen.index(row))
        monkeypatch.setattr(inspected, 'get', lambda path: None)
        namespace, name = primitive.split(':')[0].split('.')
        owner = {'os': os, 'io': io, 'Path': Path}[namespace]
        original = getattr(owner, name)
        hits = []

        def fail_once(*args, **kwargs):
            if not hits:
                hits.append(args)
                error = ValueError if primitive.endswith(':ValueError') else OSError
                raise error(f'injected {primitive}')
            return original(*args, **kwargs)

        def tick():
            with monkeypatch.context() as patch:
                patch.setattr(owner, name, fail_once)
                roster.choose()

        roster.call_later(tick)
        await pilot.pause()
        assert hits, f'{primitive} was not reached from selection preparation'
        assert app._exception is None
        assert isinstance(app.screen, (samplefetch.SampleFetchScreen, LedgerScreen))
        await pilot.press('escape')
        assert app.screen is roster and not roster._selection_active
        assert not app._samplefetch_active and not calls


@pytest.mark.parametrize('source', ['file', 'bundled', 'directory_alias', 'appeared'])
@pilot_test
async def test_selection_after_roster_paint_requires_the_settled_signature(
        sample, monkeypatch, source):
    drop, ds, payload, calls = sample
    assert drop.is_dir()
    path = drop / 'pbmc3k_raw.h5ad'
    if source == 'directory_alias':
        folder = drop / 'cohort'
        folder.mkdir()
        path = folder / path.name
        ds['handle'] = {'kind': 'path', 'ref': str(path)}
        (catalog.dataset_dir() / 'pbmc3k.yaml').write_text(yaml.safe_dump(ds))
    if source != 'appeared':
        path.write_bytes(payload)
    app = ManyrunsApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        roster = app.screen
        roster._poll_timer.pause()
        roster.poll()
        kind = 'file' if source == 'file' else 'bundled'
        index = next(i for i, entry in enumerate(roster.on_screen)
                     if isinstance(entry, state.DataEntry) and entry.kind == kind)
        roster.query_one(RosterTable).move_cursor(row=index)
        assert not roster.on_screen[index].pending

        reads = metadata_reads(monkeypatch)
        if source == 'directory_alias':
            # The whole watched directory must settle, even when this alias is unchanged.
            (path.parent / 'another.csv').write_text('a,b\n1,2\n')
        else:
            anndata.AnnData(np.ones((6, 3))).write_h5ad(path)
        await pilot.press('enter')
        assert not any(reads.values()), 'selection inspected bytes changed since roster paint'
        assert isinstance(app.screen, samplefetch.SampleFetchScreen)
        assert app.screen.entry.pending
        assert not app.screen.query('#fetch-sample')
        await pilot.press('escape')
        # Resume supplies the first observation of the changed source.
        assert not any(reads.values())
        roster.poll()
        index = next(i for i, entry in enumerate(roster.on_screen)
                     if isinstance(entry, state.DataEntry) and entry.kind == kind)
        roster.query_one(RosterTable).move_cursor(row=index)
        await pilot.press('enter')
        assert isinstance(app.screen, LedgerScreen)
        assert app.screen.entry.obs.n_obs == (4 if source == 'directory_alias' else 6)
        assert all(reads.values()), 'settled metadata must exercise both spies'
        assert not calls
