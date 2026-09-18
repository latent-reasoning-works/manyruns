"""Explicit CLI sample use authorizes a single, hermetic verified download."""
import io
import hashlib
from pathlib import Path
from urllib.error import URLError
import anndata as ad
import numpy as np
import pytest
from omegaconf import OmegaConf
from manyruns import app, shell


@pytest.fixture
def sample(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(tmp_path / 'drop'))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(tmp_path / 'catalog'))
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: pytest.fail('uninjected transport'))
    monkeypatch.setattr('builtins.input', lambda *a: pytest.fail('fetch must not prompt'))
    original = tmp_path / 'payload.h5ad'
    ad.AnnData(np.ones((5, 3))).write_h5ad(original)
    payload = original.read_bytes()
    row = {'name': 'pbmc3k', 'modality': 'scrna', 'shape': 'clusters', 'provides': ['genes'],
           'handle': {'kind': 'path', 'ref': 'data/pbmc3k_raw.h5ad',
                      'url': 'https://example.org/pbmc3k_raw.h5ad',
                      'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}
    (tmp_path / 'catalog').mkdir()
    OmegaConf.save(row, tmp_path / 'catalog/pbmc3k.yaml')
    calls = []
    monkeypatch.setattr('urllib.request.urlopen', lambda url, **kw: calls.append(url) or io.BytesIO(payload))
    computes = []
    def explore(proj, **kw):
        loaded = app._load_inputs('manylatents', Path(proj['data_folder']), None, None)
        assert loaded['array'].shape == (5, 3)
        computes.append(proj)
        return 0
    monkeypatch.setattr(app, '_explore_project', explore)
    monkeypatch.setattr(app, 'interactive_session', lambda session: computes.append(session) or 0)
    return row, payload, calls, computes


@pytest.mark.parametrize('argv', [['run', 'pbmc3k'], ['run', '--dataset', 'pbmc3k'], ['init', 'pbmc3k', '--batch'], ['init', '--dataset', 'pbmc3k']])
def test_explicit_first_use_and_local_reuse(sample, argv, capsys):
    row, payload, calls, computes = sample
    argv = argv + ['--engine', 'manylatents', '--project', 'sample', '--recipe', 'embed']
    assert app.main(argv) == 0
    assert calls == [row['handle']['url']]
    assert shell.data_dir().joinpath('pbmc3k_raw.h5ad').read_bytes() == payload
    assert app.main(argv) == 0
    assert len(calls) == 1
    assert len(computes) == 2
    assert app.load_project('sample')['dataset_name'] == 'pbmc3k'
    out = capsys.readouterr().out
    for value in ('pbmc3k_raw.h5ad', str(len(payload)), row['handle']['url'], str(shell.data_dir().resolve()), '100%'):
        assert value in out


@pytest.mark.parametrize('failure', ['offline', 'hash', 'size', 'no_url'])
def test_fetch_failure_has_recovery_and_writes_no_project(sample, monkeypatch, failure, capsys):
    row, payload, calls, computes = sample
    if failure == 'offline':
        def fail(*a, **k):
            raise URLError('offline test transport')
        monkeypatch.setattr('urllib.request.urlopen', fail)
    elif failure in ('hash', 'size'):
        bad = bytes([payload[0] ^ 1]) + payload[1:] if failure == 'hash' else payload[:-1]
        monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: io.BytesIO(bad))
    else:
        row['handle'].pop('url')
        row['handle'].pop('sha256')
        row['handle'].pop('bytes')
        OmegaConf.save(row, Path('catalog/pbmc3k.yaml'))
    with pytest.raises(SystemExit) as exc:
        app.main(['run', 'pbmc3k', '--engine', 'manylatents', '--project', 'failed'])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert str(shell.data_dir().resolve() / 'pbmc3k_raw.h5ad') in error
    if failure != 'no_url':
        assert row['handle']['url'] in error
        assert {'offline': 'offline test transport', 'hash': 'sha256 mismatch', 'size': 'bytes'}[failure] in error
    assert not Path('outputs').exists()
    assert not computes
    if shell.data_dir().exists():
        assert shell.data_dir().is_dir()
        assert list(shell.data_dir().iterdir()) == []


@pytest.mark.parametrize('verb', ['open', 'explore'])
def test_saved_catalog_source_recovers_when_removed(sample, verb):
    row, payload, calls, computes = sample
    assert app.main(['run', 'pbmc3k', '--engine', 'manylatents', '--project', 'saved', '--recipe', 'embed']) == 0
    shell.data_dir().joinpath('pbmc3k_raw.h5ad').unlink()
    assert app.main([verb, '--project', 'saved']) == 0
    assert len(calls) == 2


@pytest.mark.parametrize('verb', ['open', 'explore'])
@pytest.mark.parametrize('replacement', [False, True])
def test_existing_saved_source_wins_after_drop_folder_changes(sample, monkeypatch, tmp_path, verb, replacement):
    row, payload, calls, computes = sample
    assert app.main(['run', 'pbmc3k', '--engine', 'manylatents', '--project', 'saved', '--recipe', 'embed']) == 0
    saved_path = app.load_project('saved')['data_folder']
    other = tmp_path / 'other-drop'
    other.mkdir()
    if replacement:
        other.joinpath('pbmc3k_raw.h5ad').write_bytes(payload)
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(other))
    assert app.main([verb, '--project', 'saved']) == 0
    assert len(calls) == 1
    if verb == 'open':
        assert str(computes[-1].source) == saved_path
    else:
        assert computes[-1]['data_folder'] == saved_path


def test_listing_and_existing_user_file_do_not_fetch(sample, monkeypatch):
    row, payload, calls, computes = sample
    app.main(['projects'])
    app._build_parser().parse_args(['run', 'pbmc3k'])
    assert calls == []
    shell.ensure_drop_folder().joinpath('pbmc3k_raw.h5ad').write_bytes(payload)
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **k: pytest.fail('local file fetched'))
    row['handle']['sha256'] = '0' * 64
    OmegaConf.save(row, Path('catalog/pbmc3k.yaml'))
    assert app.main(['run', 'pbmc3k', '--engine', 'manylatents', '--project', 'local']) == 0


def test_legacy_missing_path_does_not_fetch(sample, capsys):
    app.write_project('legacy', Path('gone.h5ad'), 'scrna', app.load_recipe('embed'), engine='manylatents')
    assert app.main(['open', '--project', 'legacy']) != 0
    assert 'data path not found' in capsys.readouterr().err
    assert sample[2] == []
