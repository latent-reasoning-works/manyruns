"""Two observations are a copy heuristic, independently for each supported entry."""
from pathlib import Path

import pytest

from manyruns.tui import dropwatch


@pytest.mark.parametrize('resolve', [dropwatch.absolute, dropwatch.canonical])
def test_unknown_user_is_a_path_error(resolve, monkeypatch):
    import pwd

    def absent(name):
        raise KeyError(name)

    monkeypatch.setattr(pwd, 'getpwnam', absent)
    with pytest.raises(ValueError, match='Could not determine home directory'):
        resolve(Path('~missing_test_user/data'))


@pytest.mark.parametrize('guard', [dropwatch.row_guard, lambda fn: dropwatch.filesystem_guard(lambda e: None)(fn)])
def test_guards_preserve_unrelated_runtime_errors(guard):
    @guard
    def broken():
        raise RuntimeError('programming defect')

    with pytest.raises(RuntimeError, match='programming defect'):
        broken()


@pytest.fixture(autouse=True)
def isolated_sources(tmp_path_factory, monkeypatch):
    # Keep anchors outside each test's explicit snapshot root, where directories themselves
    # are discoverable drops. Catalog-source behavior has controlled coverage in state tests.
    anchors = tmp_path_factory.mktemp('watcher-anchors')
    cwd, configs, drop = (anchors / name for name in ('cwd', 'catalog', 'drop'))
    for folder in (cwd, configs, drop):
        folder.mkdir()
        assert folder.is_dir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(configs))
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(drop))


def test_add_growth_settled_neighbour_and_immediate_delete(tmp_path):
    assert tmp_path.is_dir()
    watch = dropwatch.DropWatch()
    a = tmp_path / 'a.csv'
    b = tmp_path / 'b.csv'
    a.write_text('1')
    b.write_text('1')
    first = watch.observe(dropwatch.snapshot([tmp_path]))
    assert first.held == {a, b} and not first.ready
    b.write_text('12')
    second = watch.observe(dropwatch.snapshot([tmp_path]))
    assert second.ready == {a} and second.held == {b}
    assert watch.observe(dropwatch.snapshot([tmp_path])).ready == {a, b}
    a.unlink()
    last = watch.observe(dropwatch.snapshot([tmp_path]))
    assert a not in last.ready | last.held


def test_nested_growth_is_seen_without_parent_mtime_change(tmp_path):
    assert tmp_path.is_dir()
    folder = tmp_path / 'cohort'
    folder.mkdir()
    matrix = folder / 'matrix.csv'
    matrix.write_text('1')
    watch = dropwatch.DropWatch()
    watch.observe(dropwatch.snapshot([tmp_path]))
    assert watch.observe(dropwatch.snapshot([tmp_path])).ready == {folder}
    before = folder.stat().st_mtime_ns
    matrix.write_text('123')
    assert folder.stat().st_mtime_ns == before
    assert watch.observe(dropwatch.snapshot([tmp_path])).held == {folder}


def test_nested_file_symlink_target_growth_restarts_settling(tmp_path, monkeypatch):
    from manyruns import catalog

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, 'discover_datasets', lambda: [])
    drop = tmp_path / 'drop'
    drop.mkdir()
    folder = drop / 'cohort'
    nested = folder / 'day1'
    nested.mkdir(parents=True)
    assert drop.is_dir() and folder.is_dir() and nested.is_dir()
    target = tmp_path / 'target.h5ad'
    target.write_bytes(b'first chunk')
    link = nested / 'cells.h5ad'
    link.symlink_to(target)
    (nested / 'cycle').symlink_to(folder, target_is_directory=True)
    watch = dropwatch.DropWatch()
    assert watch.observe(dropwatch.snapshot([drop])).held == {folder}
    original_link = link.lstat()
    target.write_bytes(b'first chunk; second chunk')
    assert link.lstat() == original_link
    assert watch.observe(dropwatch.snapshot([drop])).held == {folder}
    assert watch.observe(dropwatch.snapshot([drop])).ready == {folder}
    target.write_bytes(b'first chunk; second chunk; third chunk')
    assert watch.observe(dropwatch.snapshot([drop])).held == {folder}
    assert watch.observe(dropwatch.snapshot([drop])).ready == {folder}


def test_hidden_partials_and_directory_symlink_cycles_are_ignored(tmp_path):
    assert tmp_path.is_dir()
    (tmp_path / '.sample.h5ad.part').write_text('partial')
    folder = tmp_path / 'cohort'
    folder.mkdir()
    (folder / 'cycle').symlink_to(folder, target_is_directory=True)
    snap = dropwatch.snapshot([tmp_path / '..' / tmp_path.name])
    assert snap.roots == (tmp_path.resolve(),)
    assert set(snap.entries) == {folder.resolve()}


def test_vanished_or_denied_entries_do_not_crash_a_snapshot(tmp_path, monkeypatch):
    assert tmp_path.is_dir()
    bad = tmp_path / 'denied.csv'
    bad.write_text('1')
    good = tmp_path / 'good.csv'
    good.write_text('1')
    original = Path.stat

    def stat(path, *args, **kwargs):
        if path == bad:
            raise PermissionError('denied')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', stat)
    snap = dropwatch.snapshot([tmp_path, tmp_path / 'gone'])
    assert set(snap.entries) == {good}


def test_inspection_race_requires_two_fresh_observations(tmp_path):
    assert tmp_path.is_dir()
    source = tmp_path / 'a.csv'
    source.write_text('1')
    watch = dropwatch.DropWatch()
    watch.observe(dropwatch.snapshot([tmp_path]))
    assert watch.observe(dropwatch.snapshot([tmp_path])).ready == {source}
    watch.invalidate(source)
    assert watch.observe(dropwatch.snapshot([tmp_path])).held == {source}
    assert watch.observe(dropwatch.snapshot([tmp_path])).ready == {source}


def test_a_descendant_disappearing_during_fingerprint_is_retried(tmp_path, monkeypatch):
    assert tmp_path.is_dir()
    folder = tmp_path / 'cohort'
    folder.mkdir()
    matrix = folder / 'matrix.csv'
    matrix.write_text('1')
    real = Path.lstat

    def vanished(p, *args, **kwargs):
        if p == matrix:
            raise FileNotFoundError('copy replaced this descendant')
        return real(p, *args, **kwargs)

    monkeypatch.setattr(Path, 'lstat', vanished)
    assert dropwatch.fingerprint(folder) is None
    monkeypatch.setattr(Path, 'lstat', real)
    watch = dropwatch.DropWatch()
    assert watch.observe(dropwatch.snapshot([tmp_path])).held == {folder}
    assert watch.observe(dropwatch.snapshot([tmp_path])).ready == {folder}


def test_drop_folder_resolution_race_is_an_empty_observation(monkeypatch):
    from manyruns import shell

    def vanished():
        raise FileNotFoundError('drop folder removed during anchor resolution')

    monkeypatch.setattr(shell, 'data_dirs', vanished)
    assert dropwatch.snapshot().entries == {}


def test_symlinked_drop_root_retains_source_references_and_canonical_identity(tmp_path):
    assert tmp_path.is_dir()
    storage = tmp_path / 'storage'
    storage.mkdir()
    blob = tmp_path / 'blob'
    blob.write_text('a,b\n1,2\n')
    (storage / 'experiment.csv').symlink_to(blob)
    folder = tmp_path / 'drop'
    folder.symlink_to(storage, target_is_directory=True)
    watch = dropwatch.DropWatch()
    first = watch.observe(dropwatch.snapshot([folder]))
    assert first.held == {blob}
    assert first.snapshot.drops == (folder / 'experiment.csv',)
    assert watch.observe(dropwatch.snapshot([folder])).ready == {blob}
    watch.invalidate(first.snapshot.references[folder / 'experiment.csv'])
    assert watch.observe(dropwatch.snapshot([folder])).held == {blob}


@pytest.mark.parametrize('catalog_anchor', ['cwd', 'override'])
def test_pure_watcher_tests_ignore_developer_catalog_anchors(tmp_path, catalog_anchor):
    import os
    import subprocess
    import sys
    import yaml

    checkout = tmp_path / 'checkout'
    data = checkout / 'data'
    data.mkdir(parents=True)
    configs = tmp_path / 'catalog'
    configs.mkdir()
    drop = tmp_path / 'drop'
    drop.mkdir()
    assert checkout.is_dir() and data.is_dir() and configs.is_dir() and drop.is_dir()
    # Snapshotting needs only metadata; these are never opened as datasets.
    source = data / 'pbmc3k_raw.h5ad'
    source.write_bytes(b'developer-owned sample stand-in')
    (configs / 'external.yaml').write_text(yaml.safe_dump({
        'name': 'external', 'shape': 'single', 'modality': 'scrna',
        'handle': {'kind': 'path', 'ref': str(source)},
    }))
    test_file = Path(__file__).resolve()
    repo = test_file.parents[1]
    assert repo.is_dir() and test_file.parent.is_dir()
    env = dict(os.environ, MANYRUNS_DATA_DIR=str(drop), PYTHONPATH=str(repo))
    env.pop('MANYRUNS_DATASET_DIR', None)
    if catalog_anchor == 'override':
        env['MANYRUNS_DATASET_DIR'] = str(configs)
    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        f'{test_file}::test_add_growth_settled_neighbour_and_immediate_delete',
        f'{test_file}::test_drop_folder_resolution_race_is_an_empty_observation',
        '-q', '-p', 'no:cacheprovider', '--tb=line',
    ], cwd=checkout, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
