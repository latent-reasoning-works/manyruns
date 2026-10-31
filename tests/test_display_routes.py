"""Colour is a view channel, independent of analysis labels and original row identities."""
import argparse
import json
import sys
import types
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from manyruns import app, figspec, tune
from manyruns.pipeline import loading, runner
from manyruns.session import Session


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    def run(**kwargs):
        calls.append(kwargs)
        return {'embeddings': np.asarray(kwargs['input_data'])[:, :2].copy(), 'scores': {}}
    monkeypatch.setitem(sys.modules, 'manylatents.api', types.SimpleNamespace(run=run))
    monkeypatch.setattr(runner._suite, 'measure', lambda *a, **k: {})
    monkeypatch.setattr(runner, '_attach_geometry', lambda *a, **k: {})
    return calls


@pytest.fixture
def source(tmp_path):
    obj = ad.AnnData(np.random.default_rng(4).normal(size=(24, 4)), obs=pd.DataFrame({
        'timepoint': [0., .5, 1.] * 8,
        'cell_type': ['a', 'b'] * 12,
        'condition': ['control', 'case'] * 12,
        'score': np.arange(24, dtype=float),
        'flag': [True, False] * 12,
    }, index=[f'cell-{i}' for i in range(24)]))
    obj.layers['counts_copy'] = obj.X.copy()
    path = tmp_path / 'annotated.h5ad'
    obj.write_h5ad(path)
    return obj, path


def pca_recipe():
    step = next(s for s in app.load_recipe('embed')['steps'] if s['name'] == 'pca')
    return {'name': 'pca-only', 'steps': [step]}


@pytest.fixture
def display_recipe(monkeypatch):
    # These routes test display transport, so keep the real PCA step without scRNA prep.
    recipe = pca_recipe()
    monkeypatch.setattr(app, 'load_recipe', lambda name: recipe)


def test_one_object_load_supplies_all_channels(source, monkeypatch):
    obj, path = source
    calls = []
    from manyruns import pipeline
    monkeypatch.setattr(pipeline, 'load_array', lambda p: calls.append(p) or obj)
    loaded = app._load_inputs('manylatents', path, None, None, ('counts_copy',), color_by=['cell_type', 'score'])
    assert calls == [path]
    assert loaded['label_key'] == 'timepoint'
    assert loaded['kind'] == 'time'
    np.testing.assert_array_equal(loaded['labels'], obj.obs.timepoint)
    np.testing.assert_array_equal(loaded['obs_names'], obj.obs_names)
    assert [c['key'] for c in loaded['color']] == ['cell_type', 'score']
    assert loaded['counts'] is obj.X
    assert loaded['genes'].tolist() == obj.var_names.tolist()
    assert loaded['layers']['counts_copy'] is obj.layers['counts_copy']


def test_sparse_source_stays_sparse_at_load_time(source, monkeypatch):
    from scipy import sparse
    from manyruns import pipeline

    obj, path = source
    obj.X = sparse.csr_matrix(obj.X, dtype=np.float32)
    monkeypatch.setattr(pipeline, 'load_array', lambda p: obj)
    loaded = app._load_inputs('manylatents', path, None, None, color_by=['score'])
    assert sparse.issparse(loaded['array'])
    assert loaded['array'] is obj.X
    assert loaded['array'].dtype == np.float32
    assert loaded['counts'] is obj.X
    np.testing.assert_array_equal(loaded['obs_names'], obj.obs_names)
    np.testing.assert_array_equal(loaded['color'][0]['values'], obj.obs.score)


def _saved_display_session(path):
    app.write_project('resume', path, 'scrna', pca_recipe(), engine='manylatents')
    args = argparse.Namespace(seed=42, smoke=True, time_key=None, data_kwargs=None, color_by=['score'])
    return app.load_project('resume'), args


@pytest.mark.parametrize('filtered', [False, True])
def test_reopen_aligns_source_channels_to_artifact_ids(runtime, source, monkeypatch, filtered):
    from manyruns import artifacts

    obj, path = source
    proj, args = _saved_display_session(path)
    first = app._build_session(proj, args, resume=False)
    rows = np.arange(24)[::3] if filtered else np.arange(24)
    if filtered:
        monkeypatch.setitem(runner._prep._PREP_STEPS, 'cut',
                            lambda *a: {'mask': np.arange(24) % 3 == 0, 'axis': 'rows'})
        first.apply({'name': 'cut', 'group': 'prep'})
    first.step('pca')
    first.close()
    obj[np.arange(24)[::-1]].copy().write_h5ad(path)
    resumed = app._build_session(proj, args)
    np.testing.assert_array_equal(resumed.ctx['sample_ids'], obj.obs_names[rows])
    np.testing.assert_array_equal(resumed.state['X'], obj.X[rows])
    np.testing.assert_array_equal(resumed.state['counts'], obj.X[rows])
    np.testing.assert_array_equal(resumed.labels, obj.obs.timepoint.to_numpy()[rows])
    resumed.step('pca')
    spec = figspec.load(resumed.plots[0])
    np.testing.assert_array_equal(spec['coords'], obj.X[rows, :2])
    np.testing.assert_array_equal(spec['obs_names'], obj.obs_names[rows])
    np.testing.assert_array_equal(spec['color_values'], obj.obs.score.to_numpy()[rows])
    resumed.close()
    manifest = artifacts.read_manifest(resumed.out_dir, resumed.run_id)
    identities = [e['row_identity']['sample_ids'] for e in manifest.values() if e['slot'] == 'emb']
    assert identities == [obj.obs_names[rows].tolist()]


def test_reopen_refuses_missing_artifact_sample_ids(runtime, source, capsys):
    obj, path = source
    proj, args = _saved_display_session(path)
    first = app._build_session(proj, args, resume=False)
    first.step('pca')
    first.close()
    obj.obs_names = ['replacement'] + obj.obs_names[1:].tolist()
    obj.write_h5ad(path)
    with pytest.raises(ValueError, match='cannot resume.*missing sample IDs'):
        app._build_session(proj, args)
    assert app.main(['open', '--project', 'resume']) == 1
    assert 'missing sample IDs' in capsys.readouterr().err


@pytest.mark.parametrize('source_kind', ['matrix', 'duplicate_ids', 'legacy'])
def test_reopen_omits_nameless_filtered_embedding(
        runtime, tmp_path, monkeypatch, source_kind):
    pytest.importorskip('manylatents.singlecell.preprocessing',
                        reason='the real percentile filter delegates to scRNA preprocessing')
    from manyruns import artifacts

    # Distinct library sizes make the real percentile filter retain exactly 42 rows.
    X = np.arange(1, 61, dtype=float)[:, None] * np.arange(1, 6)[None, :]
    path = tmp_path / 'counts.npy'
    if source_kind == 'duplicate_ids':
        path = path.with_suffix('.h5ad')
        obj = ad.AnnData(X, obs=pd.DataFrame(index=['duplicate'] * len(X)))
        obj.write_h5ad(path)
    else:
        np.save(path, X)
    proj, args = _saved_display_session(path)
    args.color_by = None
    if source_kind == 'legacy':
        # The pre-WS2 caller supplied only the matrix; its artifacts have no identities.
        first = Session(project='resume', engine='manylatents', out_dir='outputs/resume',
                        array=X, recipe=pca_recipe(), metrics=[])
    else:
        first = app._build_session(proj, args, resume=False)
    assert first.step({'name': 'filter_cells', 'group': 'prep',
                       'params': {'pct_library_size': 30}})['ok']
    assert first.state['X'].shape == (42, 5)
    assert first.step('pca')['ok']
    first.close()
    manifest = artifacts.read_manifest(first.out_dir, first.run_id)
    assert manifest and all('row_identity' not in entry for entry in manifest.values())
    seen = []

    def interact(session):
        seen.append(session)
        np.testing.assert_array_equal(session.state['X'], X)
        assert session.state['emb'] is None
        assert session.parent is None
        assert session.ctx['metadata_aligned'] is True
        for key in ('arrays', 'origin', 'index', 'row_identity'):
            assert 'emb' not in session.resumed[key]
        assert not session.resumed['steps']
        assert any('omitted resumed emb' in c for c in session.close()['caveats'])
        return 0

    monkeypatch.setattr(app, 'interactive_session', interact)
    with pytest.warns(UserWarning, match='omitted resumed emb.*row count.*source'):
        assert app.main(['open', '--project', 'resume']) == 0
    assert len(seen) == 1
    assert artifacts.read_manifest(first.out_dir, first.run_id) == manifest


def test_reopen_aligns_pseudotime_from_a_different_completed_run(runtime, source, monkeypatch):
    obj, path = source
    proj, args = _saved_display_session(path)
    first = app._build_session(proj, args, resume=False)
    first.step('pca')
    def dpt(state, *a):
        state['pseudotime'] = np.arange(24, dtype=float)
    monkeypatch.setitem(runner._steps._ANALYSIS_STEPS, 'dpt', dpt)
    first.step('dpt')
    assert first.steps[-1]['outcome'] == 'ok'
    first.close()
    obj[::-1].copy().write_h5ad(path)
    second = app._build_session(proj, args, resume=False)
    second.step('pca')
    second.close()
    resumed = app._build_session(proj, args)
    np.testing.assert_array_equal(resumed.ctx['sample_ids'], obj.obs_names[::-1])
    np.testing.assert_array_equal(resumed.state['pseudotime'], np.arange(24)[::-1])


@pytest.mark.parametrize('legacy_pseudotime', [False, True])
def test_reopen_omits_filtered_pseudotime_after_full_embedding(
        runtime, source, monkeypatch, legacy_pseudotime):
    from manyruns import artifacts

    obj, path = source
    proj, args = _saved_display_session(path)
    first = app._build_session(proj, args, resume=False)
    monkeypatch.setitem(runner._prep._PREP_STEPS, 'cut',
                        lambda *a: {'mask': np.arange(24) % 3 == 0, 'axis': 'rows'})
    assert first.apply({'name': 'cut', 'group': 'prep'})['outcome'] == 'ok'
    assert first.step('pca')['ok']
    def dpt(state, *a):
        state['pseudotime'] = np.arange(len(state['emb']), dtype=float)
    monkeypatch.setitem(runner._steps._ANALYSIS_STEPS, 'dpt', dpt)
    assert first.step('dpt')['ok']
    first.close()
    if legacy_pseudotime:
        manifest = artifacts.read_manifest(first.out_dir, first.run_id)
        for entry in manifest.values():
            if entry['slot'] == 'pseudotime':
                entry.pop('row_identity', None)
        artifacts.finish(first.out_dir, first.run_id, manifest)

    second = app._build_session(proj, args, resume=False)
    assert second.step('pca')['ok']
    second.close()
    with pytest.warns(UserWarning, match='omitted resumed pseudotime'):
        resumed = app._build_session(proj, args)
    np.testing.assert_array_equal(resumed.state['emb'], second.state['emb'])
    np.testing.assert_array_equal(resumed.ctx['sample_ids'], obj.obs_names)
    assert resumed.state['pseudotime'] is None
    assert resumed.parent == {'run_id': second.run_id, 'index': 0}
    for key in ('arrays', 'origin', 'index', 'row_identity'):
        assert 'pseudotime' not in resumed.resumed[key]
    assert all(slot != 'pseudotime' for _, slot, _ in resumed.resumed['steps'])
    assert resumed.step('pca')['ok']
    spec = figspec.load(resumed.plots[0])
    np.testing.assert_array_equal(spec['obs_names'], obj.obs_names)
    np.testing.assert_array_equal(spec['color_values'], obj.obs.score)
    assert any('omitted resumed pseudotime' in c for c in resumed.close()['caveats'])
    assert any(e['slot'] == 'pseudotime'
               for e in artifacts.read_manifest(first.out_dir, first.run_id).values())


def test_legacy_resume_discloses_unverified_alignment(runtime, source):
    from manyruns import artifacts

    _, path = source
    proj, args = _saved_display_session(path)
    first = app._build_session(proj, args, resume=False)
    first.step('pca')
    first.close()
    manifest = artifacts.read_manifest(first.out_dir, first.run_id)
    for entry in manifest.values():
        entry.pop('row_identity', None)
    artifacts.finish(first.out_dir, first.run_id, manifest)
    with pytest.warns(UserWarning, match='positional.*unverified'):
        resumed = app._build_session(proj, args)
    assert resumed.ctx['metadata_aligned'] is None
    resumed.step('pca')
    spec = figspec.load(resumed.plots[0])
    assert spec.get('obs_names') is None
    assert spec.get('color_values') is None
    assert any('positional' in c and 'unverified' in c for c in resumed.close()['caveats'])


@pytest.mark.parametrize('output_rows', ['aligned', 'missing', 'reordered'])
def test_legacy_resume_fresh_fit_rechecks_source_alignment(
        runtime, source, monkeypatch, output_rows):
    from manyruns import artifacts

    obj, path = source
    proj, args = _saved_display_session(path)
    first = app._build_session(proj, args, resume=False)
    assert first.step('pca')['ok']
    first.close()
    manifest = artifacts.read_manifest(first.out_dir, first.run_id)
    for entry in manifest.values():
        entry.pop('row_identity', None)
    artifacts.finish(first.out_dir, first.run_id, manifest)
    with pytest.warns(UserWarning, match='positional.*unverified'):
        resumed = app._build_session(proj, args)
    assert resumed.ctx['metadata_aligned'] is None
    rows = np.arange(len(obj))[::3]
    monkeypatch.setitem(runner._prep._PREP_STEPS, 'cut',
                        lambda *a: {'mask': np.arange(len(obj)) % 3 == 0, 'axis': 'rows'})
    assert resumed.step({'name': 'cut', 'group': 'prep'})['ok']
    assert resumed.state['emb'] is None

    def fit(**kwargs):
        np.testing.assert_array_equal(kwargs['input_data'], obj.X[rows])
        emb = kwargs['input_data'][:, :2].copy()
        if output_rows == 'missing':
            return {'embeddings': emb[:-1]}
        if output_rows == 'reordered':
            return {'embeddings': emb[::-1], 'obs_names': obj.obs_names[rows][::-1]}
        return {'embeddings': emb}

    monkeypatch.setattr(sys.modules['manylatents.api'], 'run', fit)
    assert resumed.step('pca')['ok']
    result = resumed.close()
    spec = figspec.load(resumed.plots[0])
    manifest = artifacts.read_manifest(resumed.out_dir, resumed.run_id)
    embeddings = [entry for entry in manifest.values() if entry['slot'] == 'emb']
    assert len(embeddings) == 1
    if output_rows == 'aligned':
        np.testing.assert_array_equal(spec.get('color_values'), obj.obs.score.to_numpy()[rows])
        np.testing.assert_array_equal(spec['obs_names'], obj.obs_names[rows])
        assert embeddings[0]['row_identity']['sample_ids'] == obj.obs_names[rows].tolist()
        assert resumed.ctx['metadata_aligned'] is True
    else:
        assert spec.get('color_values') is None and spec.get('obs_names') is None
        assert 'row_identity' not in embeddings[0]
        assert resumed.ctx['metadata_aligned'] is False
        assert any('metadata alignment unavailable' in c for c in result['caveats'])


@pytest.mark.parametrize('annotated', [False, True])
def test_branch_after_reopen_preserves_unverified_identity(runtime, source, annotated):
    from manyruns import artifacts

    obj, path = source
    if not annotated:
        path = path.with_suffix('.npy')
        np.save(path, obj.X)
    proj, args = _saved_display_session(path)
    args.color_by = ['score'] if annotated else None
    first = app._build_session(proj, args, resume=False)
    assert first.step('pca')['ok']
    first.close()
    if annotated:
        manifest = artifacts.read_manifest(first.out_dir, first.run_id)
        for entry in manifest.values():
            entry.pop('row_identity', None)
        artifacts.finish(first.out_dir, first.run_id, manifest)
    with pytest.warns(UserWarning, match='positional.*unverified'):
        resumed = app._build_session(proj, args)
    assert resumed.step('pca')['ok']
    child = resumed.branch(0)
    np.testing.assert_array_equal(child.state['emb'], resumed.state['emb'])
    assert child.parent == {'run_id': resumed.run_id, 'index': 0}
    assert child.step('pca')['ok']
    spec = figspec.load(child.plots[0])
    assert spec.get('obs_names') is None
    assert spec.get('color_values') is None
    assert any('positional' in c and 'unverified' in c for c in child.close()['caveats'])
    manifest = artifacts.read_manifest(child.out_dir, child.run_id)
    assert manifest and all('row_identity' not in e for e in manifest.values())
    resumed.close()


def test_branch_after_positional_reopen_refuses_incompatible_engine_rows(
        runtime, source, monkeypatch):
    obj, path = source
    path = path.with_suffix('.npy')
    np.save(path, obj.X)
    proj, args = _saved_display_session(path)
    args.color_by = None
    first = app._build_session(proj, args, resume=False)
    assert first.step('pca')['ok']
    first.close()
    with pytest.warns(UserWarning, match='positional.*unverified'):
        resumed = app._build_session(proj, args)
    monkeypatch.setattr(sys.modules['manylatents.api'], 'run',
                        lambda **kw: {'embeddings': kw['input_data'][::2, :2]})
    assert resumed.step('pca')['ok']
    with pytest.raises(ValueError, match='cannot branch: metadata alignment unavailable'):
        resumed.branch(0)
    assert any('engine coordinates have 12 rows for 24 input rows' in c
               for c in resumed.close()['caveats'])


@pytest.mark.parametrize('metadata', ['bare', 'annotated', 'positional'])
@pytest.mark.parametrize('mismatch', ['rows', 'order'])
def test_branch_uses_selected_artifact_alignment(
        runtime, source, tmp_path, monkeypatch, metadata, mismatch):
    obj, _ = source
    annotated = metadata != 'bare'
    session = Session(project='rewind', engine='manylatents', out_dir=tmp_path,
                      array=obj.X, recipe=pca_recipe(), metrics=[],
                      sample_ids=obj.obs_names if annotated else None,
                      color=[loading.color_channel_of(obj, 'score')] if annotated else None)
    assert session.step('pca')['ok']
    if metadata == 'positional':
        session.ctx['metadata_aligned'] = None
    assert session.step('pca')['ok']

    def misaligned(**kwargs):
        emb = kwargs['input_data'][:, :2].copy()
        if mismatch == 'rows':
            return {'embeddings': emb[:-1]}
        return {'embeddings': emb[::-1], 'obs_names': obj.obs_names[::-1]}

    with monkeypatch.context() as patch:
        patch.setattr(sys.modules['manylatents.api'], 'run', misaligned)
        assert session.step('phate')['ok']
    assert session.ctx['metadata_aligned'] is False
    for index in (0, 1):
        child = session.branch(index)
        expected_alignment = None if metadata == 'positional' and index == 1 else True
        assert child.ctx.get('metadata_aligned', True) is expected_alignment
        np.testing.assert_array_equal(child.state['emb'], obj.X[:, :2])
        assert child.parent == {'run_id': session.run_id, 'index': index}
        assert child.step('pca')['ok']
        spec = figspec.load(child.plots[0])
        if annotated and expected_alignment:
            np.testing.assert_array_equal(spec['obs_names'], obj.obs_names)
            np.testing.assert_array_equal(spec['color_values'], obj.obs.score)
        else:
            assert spec.get('obs_names') is None and spec.get('color_values') is None
        child.close()
    with pytest.raises(ValueError, match='cannot branch: metadata alignment unavailable'):
        session.branch(2)
    assert session.ctx['metadata_aligned'] is False
    session.close()


@pytest.mark.parametrize('keys', [[], ['cell_type', 'score']])
def test_cli_modes_runner_display_transport(runtime, source, keys, tmp_path, display_recipe):
    obj, path = source
    argv = ['run', str(path), '--project', 'view', '--engine', 'manylatents', '--recipe', 'embed']
    for key in keys:
        argv += ['--color-by', key]
    assert app.main(argv) == 0
    plots = tmp_path / 'outputs/view/plots'
    assert plots.is_dir()
    specs = [figspec.load(p) for p in plots.glob('pca*.png')]
    assert len(specs) == max(1, len(keys))
    assert {s['color_by'] for s in specs} == set(keys or ['timepoint'])
    for spec in specs:
        np.testing.assert_array_equal(spec['obs_names'], obj.obs_names)
    if not keys:
        assert specs[0]['color_kind'] == 'continuous'  # even with only three time bins
    proj = app.load_project('view')
    args = argparse.Namespace(seed=42, smoke=True, time_key=None, data_kwargs=None, color_by=keys)
    session = app._build_session(proj, args, resume=False)
    session.step('pca')
    assert session.state['counts'] is not None
    np.testing.assert_array_equal(session.labels, obj.obs.timepoint)
    assert session.ctx['label_key'] == 'timepoint'
    np.testing.assert_array_equal(session.ctx['sample_ids'], obj.obs_names)


@pytest.mark.parametrize(('source_kind', 'expected', 'unexpected'), [
    ('annotated', "unknown colour column 'unknown'; available columns: timepoint, cell_type, condition, score, flag",
     'cannot inspect metadata'),
    ('generator', "cannot inspect metadata for 'swissroll'", 'available columns'),
])
def test_bad_cli_column_and_uninspectable_generator_refuse(
        runtime, source, capsys, source_kind, expected, unexpected):
    data = str(source[1]) if source_kind == 'annotated' else 'swissroll'
    with pytest.raises(SystemExit) as exc:
        app.main(['run', data, '--project', 'bad', '--engine', 'manylatents',
                  '--recipe', 'embed', '--color-by', 'unknown'])
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert expected in message
    assert unexpected not in message
    assert runtime == []


@pytest.mark.parametrize('command', ['run', 'init'])
@pytest.mark.parametrize('existing', [False, True])
def test_refused_colour_leaves_saved_project_unchanged(
        runtime, source, tmp_path, capsys, command, existing, display_recipe):
    path = tmp_path / 'outputs/keep/project.yaml'
    if existing:
        assert app.main(['run', str(source[1]), '--project', 'keep', '--engine',
                         'manylatents', '--recipe', 'embed']) == 0
        before = path.read_bytes()
        runtime.clear()
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        app.main([command, 'synthetic_timecourse', '--project', 'keep', '--engine',
                  'manylatents', '--recipe', 'cflows', '--color-by', 'timepoint',
                  '--color-by', 'nope'])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "unknown colour column 'nope'; available columns: timepoint" in captured.err
    assert runtime == []
    if existing:
        assert path.read_bytes() == before
    else:
        assert not path.exists()
    assert 'project initialized' not in captured.out


@pytest.mark.parametrize('missing_rows', [True, False])
@pytest.mark.parametrize('other_channel', [True, False])
@pytest.mark.parametrize('terminal', [True, False])
def test_cli_reports_and_records_skipped_colour_channels(
        runtime, source, tmp_path, monkeypatch, capsys, missing_rows, other_channel, terminal):
    from rich.console import Console
    from manyruns import shell, store

    obj, path = source
    key = 'score[missing]'
    obj.obs[key] = obj.obs.score.copy()
    obj.obs.loc[obj.obs_names[[0, 3]], key] = np.nan
    obj.write_h5ad(path)
    assert loading.color_channel_of(obj, key)['kind'] == 'continuous'
    mask = obj.obs[key].isna().to_numpy()
    if not missing_rows:
        mask = ~mask
    recipe = {'name': 'filtered-pca', 'steps': [
        {'name': 'cut', 'group': 'prep'}, *pca_recipe()['steps']]}
    monkeypatch.setattr(app, 'load_recipe', lambda name: recipe)
    monkeypatch.setitem(runner._prep._PREP_STEPS, 'cut',
                        lambda *a: {'mask': mask, 'axis': 'rows'})
    monkeypatch.setattr(shell, '_default_console',
                        lambda: Console(force_terminal=terminal, color_system=None, width=200))
    argv = ['run', str(path), '--project', 'skips', '--engine', 'manylatents',
            '--recipe', 'embed', '--color-by', key]
    if other_channel:
        argv += ['--color-by', 'flag']
    assert app.main(argv) == 0
    np.testing.assert_array_equal(runtime[0]['input_data'], obj.X[mask])
    rows = list(store.read(tmp_path / 'outputs'))
    assert len(rows) == 1
    expected = ([f"skipping colour channel {key!r}: all plotted values are missing or nonfinite"]
                if missing_rows else [])
    caveats = [c for c in rows[0]['caveats'] if c.startswith('skipping colour channel ')]
    output = capsys.readouterr().out
    assert caveats == expected
    assert output.count('skipping colour channel ') == len(expected)
    for message in expected:
        assert message in output


def test_synthetic_timecourse_is_inspectable(runtime, tmp_path, display_recipe):
    result = app.explore_once(data_folder=Path('synthetic:time-course'), dataset=None,
                              modality='synthetic', recipe_name='embed', engine='manylatents',
                              out_dir=tmp_path, color_by=['timepoint'])
    assert result['ok']
    assert figspec.load(result['plots'][0])['color_by'] == 'timepoint'


def test_analysis_axes_and_coordinates_do_not_change_with_display(runtime, source, monkeypatch, tmp_path):
    obj, path = source
    times, conditions, coordinates = [], [], []
    def flow(X, seed, fast_dev_run, params, labels=None, device='cpu'):
        times.append(labels.copy())
        return X.copy(), {}, {}, object()
    def contrast(state, g, params, out_dir, plots, target_dim):
        conditions.append(state['labels'].copy())
    monkeypatch.setattr(runner._mioflow, '_run_mioflow_experiment', flow)
    monkeypatch.setitem(runner._steps._ANALYSIS_STEPS, 'separation', contrast)
    for keys in [[], ['cell_type'], ['score']]:
        loaded = app._load_inputs('manylatents', path, None, None, color_by=keys)
        for kind, labels in [('time', loaded['labels']), ('condition', obj.obs.condition.to_numpy())]:
            result = runner.run_manylatents(
                {'steps': pca_recipe()['steps'] + [{'name': 'mioflow' if kind == 'time' else 'separation',
                                                   'group': 'lightning' if kind == 'time' else 'analysis'}]},
                array=loaded['array'], labels=labels, label_kind=kind,
                color=loaded['color'], sample_ids=loaded['obs_names'], label_key=loaded['label_key'], out_dir=tmp_path)
            assert result['ok']
            assert len(result['steps']) == 2
            assert all(step['outcome'] in ('ok', 'reported') for step in result['steps'])
            coordinates.append(figspec.load(result['plots'][0])['coords'])
    assert len(times) == 3
    assert len(conditions) == 3
    assert len(coordinates) == 6
    assert len(runtime) == 6
    for time in times:
        np.testing.assert_array_equal(time, obj.obs.timepoint)
    for condition in conditions:
        np.testing.assert_array_equal(condition, obj.obs.condition)
    for coords in coordinates:
        np.testing.assert_array_equal(coords, obj.X[:, :2])


@pytest.mark.parametrize('explicit', [False, True])
def test_derived_analysis_labels_do_not_claim_the_source_column(runtime, source, tmp_path, explicit):
    obj, _ = source
    session = Session('derived', engine='manylatents', out_dir=tmp_path,
                      array=obj.X, labels=obj.obs.condition.to_numpy(), label_kind='condition',
                      label_key='condition', sample_ids=obj.obs_names,
                      color=[loading.color_channel_of(obj, 'condition')] if explicit else [],
                      recipe=pca_recipe())
    session.state['pseudotime'] = np.arange(24, dtype=float)
    session.step('discretize_time')
    assert session.steps[-1]['outcome'] == 'ok'
    session.step('pca')
    assert session.steps[-1]['outcome'] == 'ok'
    spec = figspec.load(session.plots[-1])
    assert spec['color_by'] == ('condition' if explicit else None)
    assert spec['color_kind'] == ('categorical' if explicit else 'continuous')
    channel = runner.current_display(session.state, session.ctx)[0][0]
    np.testing.assert_array_equal(channel['values'], obj.obs.condition if explicit else
                                  runner._io._labels_to_numeric(session.state['labels']))
    child = session.branch(1)
    child.step('pca')
    assert figspec.load(child.plots[-1])['color_by'] == ('condition' if explicit else None)


def test_cancelled_label_derivation_restores_source_provenance(runtime, source, tmp_path):
    obj, _ = source
    session = Session('cancel', engine='manylatents', out_dir=tmp_path,
                      array=obj.X, labels=obj.obs.condition.to_numpy(), label_kind='condition',
                      label_key='condition', sample_ids=obj.obs_names)
    session.state['pseudotime'] = np.arange(24, dtype=float)
    tune.run_tune_loop(session, {'name': 'discretize_time', 'group': 'analysis'},
                       read=lambda *a: 'cancel', write=lambda *a: None)
    assert runner.current_display(session.state, session.ctx)[0][0]['key'] == 'condition'
    session.step('pca')
    assert figspec.load(session.plots[-1])['color_by'] == 'condition'


def test_original_axes_survive_two_filters_genes_branch_and_tune(runtime, source, tmp_path, monkeypatch):
    obj, _ = source
    color = [loading.color_channel_of(obj, 'cell_type'), loading.color_channel_of(obj, 'score')]
    s = Session('axis', engine='manylatents', out_dir=tmp_path, array=obj.X, labels=obj.obs.timepoint.to_numpy(),
                label_kind='time', label_key='timepoint', color=color, sample_ids=obj.obs_names,
                counts=obj.X, genes=obj.var_names.to_numpy(), layers={'copy': obj.X.copy()}, recipe=pca_recipe())
    original = s.ctx['display_channels'][1]['values'].copy()
    masks = [np.arange(24) % 3 != 1, np.arange(16) % 2 == 1]
    monkeypatch.setitem(runner._prep._PREP_STEPS, 'cut', lambda *a: {'mask': masks.pop(0), 'axis': 'rows'})
    for _ in range(2):
        s.apply({'name': 'cut', 'group': 'prep'})
    monkeypatch.setitem(runner._prep._PREP_STEPS, 'genes', lambda *a: {'mask': [True, False, True, True], 'axis': 'cols'})
    s.apply({'name': 'genes', 'group': 'prep'})
    s.step('pca')
    rows = np.arange(24)[np.arange(24) % 3 != 1][np.arange(16) % 2 == 1]
    np.testing.assert_array_equal(s.state['rows'], rows)
    np.testing.assert_array_equal(s.ctx['sample_ids'], obj.obs_names)
    np.testing.assert_array_equal(s.ctx['display_channels'][1]['values'], original)
    spec = figspec.load(s.plots[0])
    np.testing.assert_array_equal(spec['obs_names'], obj.obs_names[rows])
    child = s.branch(3)
    np.testing.assert_array_equal(child.ctx['sample_ids'], obj.obs_names[rows])
    np.testing.assert_array_equal(child.ctx['display_channels'][1]['values'], obj.obs.score.to_numpy()[rows])
    before = s.state['emb'].copy()
    tune.run_tune_loop(s, pca_recipe()['steps'][0], read=lambda *a: 'cancel', write=lambda *a: None)
    np.testing.assert_array_equal(s.state['emb'], before)
    s.step('pca')
    np.testing.assert_array_equal(figspec.load(s.plots[0])['obs_names'], obj.obs_names[rows])


@pytest.mark.parametrize('constructor', ['session', 'manylatents', 'inproc'])
@pytest.mark.parametrize('channel', ['ids', 'color'])
def test_explicit_display_alignment_is_validated(runtime, tmp_path, constructor, channel):
    kw = {'sample_ids': ['x'] * 5} if channel == 'ids' else {'color': [{'key': 'x', 'kind': 'continuous', 'values': [1, 2]}]}
    with pytest.raises(ValueError, match='unique|match.*rows'):
        if constructor == 'session':
            Session('bad', array=np.zeros((5, 3)), **kw)
        elif constructor == 'manylatents':
            runner.run_manylatents({}, array=np.zeros((5, 3)), **kw)
        else:
            runner.run_inproc(np.zeros((5, 3)), {}, tmp_path, **kw)


def test_implicit_duplicate_ids_disclose_positional_alignment(runtime, source):
    obj, path = source
    obj.obs_names = ['same'] * 24
    obj.write_h5ad(path)
    with pytest.warns(UserWarning, match='positional alignment'):
        loaded = app._load_inputs('manylatents', path, None, None)
    assert loaded['obs_names'] is None
    assert loaded['array'].shape == (24, 4)


def test_real_inproc_phate_uses_display_scatter(runtime, tmp_path):
    X = np.random.default_rng(13).normal(size=(40, 4))
    labels = np.tile([0., .5, 1., 2.], 10)
    result = runner.run_inproc(X, {'steps': [{'name': 'phate', 'group': 'latent',
                                             'params': {'n_components': 2, 'knn': 5, 't': 2}}]},
                               tmp_path, labels=labels, label_kind='time', label_key='timepoint',
                               sample_ids=[f'row-{i}' for i in range(40)])
    assert result['ok'] and len(result['plots']) == 1
    spec = figspec.load(result['plots'][0])
    assert spec['color_by'] == 'timepoint'
    assert spec['color_kind'] == 'continuous'
    np.testing.assert_array_equal(spec['color_values'], labels / 2)
    assert len(spec['obs_names']) == 40


@pytest.mark.parametrize('label_key', [None, 'condition'])
def test_inproc_explicit_labels_keep_caller_provenance(
        runtime, source, tmp_path, monkeypatch, label_key):
    obj, _ = source
    def embed(state, g, params, out_dir, plots, target_dim):
        state['emb'] = state['X'][:, :2].copy()
        runner._io._save_scatter(state['emb'], None, out_dir, 'phate.png', plots, 'PHATE')
    monkeypatch.setitem(runner._steps._INPROC_STEPS, 'phate', embed)
    result = runner.run_inproc(
        obj, {'steps': [{'name': 'phate', 'group': 'latent'}]}, tmp_path,
        labels=obj.obs.condition.to_numpy(), label_kind='condition', label_key=label_key)
    assert result['ok'] and len(result['plots']) == 1
    spec = figspec.load(result['plots'][0])
    assert spec.get('color_by') == label_key
    assert spec['color_kind'] == 'categorical'
    np.testing.assert_array_equal(spec['color_values'], obj.obs.condition)
    np.testing.assert_array_equal(spec['obs_names'], obj.obs_names)


def test_unmapped_engine_rows_receive_no_metadata(runtime, source, tmp_path, monkeypatch):
    obj, _ = source
    monkeypatch.setattr(sys.modules['manylatents.api'], 'run',
                        lambda **kw: {'embeddings': kw['input_data'][::2, :2]})
    result = runner.run_manylatents(pca_recipe(), array=obj.X,
                                    color=[loading.color_channel_of(obj, 'score')],
                                    sample_ids=obj.obs_names, out_dir=tmp_path)
    assert result['ok']
    spec = figspec.load(result['plots'][0])
    assert spec.get('obs_names') is None and spec.get('color_values') is None
    assert any('metadata alignment unavailable' in c for c in result['caveats'])


def test_folder_time_axis_and_concatenated_ids_are_preserved(runtime, tmp_path):
    root = tmp_path / 'timepoints'
    for name in ['day0', 'day9']:
        directory = root / name
        directory.mkdir(parents=True)
        ad.AnnData(np.ones((3, 2)), obs=pd.DataFrame({'cell_type': ['a', 'b', 'a']},
                                                   index=['one', 'two', 'three'])).write_h5ad(directory / 'data.h5ad')
    loaded = app._load_inputs('manylatents', root, None, None, color_by=['cell_type'])
    assert loaded['labels'].tolist() == ['day0'] * 3 + ['day9'] * 3
    assert loaded['kind'] == 'time' and loaded['label_key'] is None
    assert len(set(loaded['obs_names'])) == 6
    np.testing.assert_array_equal(loaded['obs_names'], loading.load_array(root).obs_names)


def test_cancelled_filter_restores_the_compute_input(runtime, source, tmp_path, monkeypatch):
    obj, _ = source
    session = Session('filter', engine='manylatents', array=obj.X, out_dir=tmp_path,
                      color=[loading.color_channel_of(obj, 'score')], sample_ids=obj.obs_names)
    monkeypatch.setitem(runner._prep._PREP_STEPS, 'cut', lambda *a: {'mask': np.arange(24) % 3 != 1, 'axis': 'rows'})
    tune.run_tune_loop(session, {'name': 'cut', 'group': 'prep', 'params': {}},
                       read=lambda *a: 'cancel', write=lambda *a: None)
    session.apply(pca_recipe()['steps'][0])
    np.testing.assert_array_equal(runtime[-1]['input_data'], obj.X)
    np.testing.assert_array_equal(figspec.load(session.plots[0])['obs_names'], obj.obs_names)


def test_reported_engine_reordering_does_not_stamp_source_metadata(runtime, source, monkeypatch, tmp_path):
    from manyruns import artifacts
    obj, _ = source
    monkeypatch.setattr(sys.modules['manylatents.api'], 'run', lambda **kw: {
        'embeddings': kw['input_data'][::-1, :2], 'obs_names': obj.obs_names[::-1].to_numpy()})
    session = Session('reordered', engine='manylatents', array=obj.X, sample_ids=obj.obs_names,
                      color=[loading.color_channel_of(obj, 'score')], out_dir=tmp_path)
    session.apply(pca_recipe()['steps'][0])
    result = session.close()
    spec = figspec.load(result['plots'][0])
    assert spec.get('obs_names') is None and spec.get('color_values') is None
    assert any('metadata alignment unavailable' in c for c in result['caveats'])
    folder = artifacts.root(tmp_path, session.run_id)
    manifest = json.loads((folder / 'COMPLETE').read_text())
    assert 'row_identity' not in str(manifest)


def test_mioflow_receives_the_actual_time_tensor_for_each_display(runtime, source, monkeypatch, tmp_path):
    """Preservation: the installed adapter supplies time through its datamodule's labels."""
    experiment = pytest.importorskip('manylatents.experiment',
                                     reason='verifies the installed engine datamodule adapter')
    obj, path = source
    tensors = []
    def run_experiment(*, datamodule, algorithm, trainer, seed):
        batch = next(iter(datamodule.train_dataloader()))
        tensors.append(batch['label'].numpy().copy())
        return {'embeddings': batch['data'].numpy(), 'scores': {}}
    monkeypatch.setattr(experiment, 'run_experiment', run_experiment)
    for keys in [[], ['cell_type'], ['score']]:
        loaded = app._load_inputs('manylatents', path, None, None, color_by=keys)
        result = runner.run_manylatents(
            {'steps': pca_recipe()['steps'] + [{'name': 'mioflow', 'group': 'lightning'}]},
            array=loaded['array'], labels=loaded['labels'], label_kind=loaded['kind'],
            color=loaded['color'], sample_ids=loaded['obs_names'], out_dir=tmp_path)
        assert result['ok'], result['status']
    assert len(tensors) == 3
    for tensor in tensors:
        np.testing.assert_array_equal(tensor, obj.obs.timepoint)


def test_raw_matrix_has_no_invented_column_key(runtime, tmp_path):
    result = runner.run_manylatents(pca_recipe(), array=np.ones((12, 3)),
                                    labels=np.arange(12), label_kind='time', out_dir=tmp_path)
    spec = figspec.load(result['plots'][0])
    assert spec['color_by'] is None and spec.get('obs_names') is None


@pytest.mark.parametrize('verb', ['init', 'open', 'explore'])
def test_other_cli_doors_transport_colour(runtime, source, tmp_path, monkeypatch, verb):
    obj, path = source
    recipe = pca_recipe()
    app.write_project('saved', path, 'scrna', recipe, engine='manylatents')
    seen = []
    def interactive(session):
        session.run_recipe()
        seen.extend(session.plots)
        return 0
    monkeypatch.setattr(app, 'interactive_session', interactive)
    argv = [verb, '--project', 'saved', '--color-by', 'flag']
    if verb == 'init':
        argv += [str(path), '--engine', 'manylatents', '--recipe', 'embed']
    assert app.main(argv) == 0
    png = seen[0] if seen else tmp_path / 'outputs/saved/plots/pca.png'
    spec = figspec.load(png)
    assert spec['color_by'] == 'flag' and spec['color_kind'] == 'categorical'


def test_display_choices_do_not_enter_decision_rows(runtime, source, tmp_path):
    import json
    obj, _ = source
    rows = []
    for i, key in enumerate(['cell_type', 'score']):
        session = Session('decision', engine='manylatents', array=obj.X, out_dir=tmp_path / str(i),
                          color=[loading.color_channel_of(obj, key)], recipe=pca_recipe())
        answers = iter(['retry', 'knn=7', 'accept'])
        tune.run_tune_loop(session, {'name': 'phate', 'group': 'latent', 'params': {'knn': 5}},
                           read=lambda *a: next(answers), write=lambda *a: None)
        assert session.steps[-1]['outcome'] == 'ok'
    rows = [json.loads((tmp_path / str(i) / 'decisions.jsonl').read_text()) for i in range(2)]
    assert len(rows) == 2
    for row in rows:
        assert 'color_by' not in str(row) and 'cell_type' not in str(row) and 'score' not in str(row)


def test_explicit_metadata_for_a_declared_generator_keeps_its_original_axis(runtime, source, monkeypatch, tmp_path):
    obj, _ = source
    monkeypatch.setattr(sys.modules['manylatents.api'], 'run', lambda **kw: {'embeddings': obj.X[:, :2]})
    session = Session('declared', engine='manylatents', dataset='external_generator',
                      declared_shape=obj.shape, sample_ids=obj.obs_names,
                      color=[loading.color_channel_of(obj, 'score')], out_dir=tmp_path)
    session.apply(pca_recipe()['steps'][0])
    assert session.steps[0]['outcome'] == 'ok'
    spec = figspec.load(session.plots[0])
    np.testing.assert_array_equal(spec['obs_names'], obj.obs_names)
    np.testing.assert_array_equal(spec['color_values'], obj.obs.score)
