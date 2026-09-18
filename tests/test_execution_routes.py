"""Exercise actual execution routes with a deterministic, row-preserving engine."""
import argparse
import sys
import types
import numpy as np
import pytest
from manyruns import app, catalog, modes
from manyruns.pipeline import runner
from manyruns.session import Session


@pytest.fixture
def engine(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from importlib.resources import files
    import shutil
    source = files('manyruns').joinpath('configs/dataset')
    assert source.is_dir()
    shutil.copytree(str(source), tmp_path / 'catalog')
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(tmp_path / 'catalog'))
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('MANYRUNS_STORE_DIR', str(tmp_path / 'store'))
    calls = []
    def run(**kw):
        calls.append(kw)
        n = len(kw['input_data']) if 'input_data' in kw else kw.get('data_kwargs', {}).get('n_samples', kw.get('data_kwargs', {}).get('n_obs', kw.get('data_kwargs', {}).get('num_points', 5000)))
        if kw.get('data') == 'dla_tree':
            n = kw['data_kwargs']['n_branch'] * 100
        if kw.get('data') == 'torus':
            n = 794
        return {'embeddings': np.arange(n * 2, dtype=float).reshape(n, 2), 'scores': {}}
    monkeypatch.setitem(sys.modules, 'manylatents.api', types.SimpleNamespace(run=run))
    monkeypatch.setattr(runner._suite, 'measure', lambda *a, **k: {})
    monkeypatch.setattr(runner, '_attach_geometry', lambda *a, **k: {})
    monkeypatch.setattr(runner._io, '_save_scatter', lambda *a, **k: None)
    return calls


def args(**kw):
    return argparse.Namespace(seed=42, smoke=True, time_key=None, data_kwargs=None, **kw)


@pytest.mark.parametrize('name,cap', [('swissroll', 3), ('tree_narrow', 3), ('arch_soft', 5), ('blob_loose', 2), ('torus', 50)])
def test_cli_modes_server_runner_source_contract(engine, name, cap):
    assert app.main(['run', name, '--project', name, '--engine', 'manylatents', '--recipe', 'embed']) == 0
    row = catalog.load_dataset(name)
    assert engine[0].get('data_kwargs', {}) == row.get('params', {})
    assert engine[0]['n_components'] == cap
    assert engine[1]['input_data'].shape[1] == 2
    assert engine[1]['n_components'] == 3  # PHATE has no feature-count cap
    proj = app.load_project(name)
    assert proj['dataset_name'] == name
    assert proj['data_kwargs'] == row.get('params', {})
    session = app._build_session(proj, args(), resume=False)
    assert session.data_kwargs == row.get('params', {})
    session.step('pca')
    rec = session.steps[-1]
    assert rec['params']['n_components'] == cap
    assert bool(rec.get('bounded')) == (cap != 50)
    child = session.branch(0)
    assert child.ctx['declared_shape'] == session.ctx['declared_shape']
    assert child.data_kwargs == session.data_kwargs
    session.step('pca')
    second = session.steps[-1]
    assert second['params']['n_components'] == 2  # actual embedding wins
    assert second['bounded']


def test_persisted_override_and_explicit_override_merge(engine):
    recipe = {'name': 'unbounded', 'steps': []}
    app.write_project('custom', None, 'synthetic', recipe, engine='manylatents',
                      dataset='archetypal', dataset_name='arch_soft', data_kwargs={'n_obs': 17})
    proj = app.load_project('custom')
    session = app._build_session(proj, args(), resume=False)
    assert session.data_kwargs == {**catalog.load_dataset('arch_soft')['params'], 'n_obs': 17}
    overridden = args()
    overridden.data_kwargs = ['n_obs=19']
    session = app._build_session(proj, overridden, resume=False)
    assert session.data_kwargs['n_obs'] == 19
    assert session.ctx['declared_shape'] is None


@pytest.mark.parametrize('dataset,name,overrides', [('archetypal', None, {}), ('dla_tree', None, {}), ('archetypal', 'arch_soft', {'n_obs': 11})])
def test_unresolved_bounded_source_refuses_before_compute(engine, dataset, name, overrides):
    with pytest.raises(ValueError, match='matching catalog declaration.*explicit generated data file'):
        app.run_explorations(None, 'synthetic', engine='manylatents', dataset=dataset,
                            dataset_name=name, data_kwargs=overrides, recipe=app.load_recipe('embed'))
    assert engine == []


def test_path_identity_and_actual_shape(engine, tmp_path, monkeypatch):
    # Source identity and dimensional bounds do not depend on scRNA preprocessing.
    recipe = app.load_recipe('embed')
    recipe['steps'] = [s for s in recipe['steps'] if s['group'] == 'latent']
    monkeypatch.setattr(app, 'load_recipe', lambda name: recipe)
    path = tmp_path / 'arch_soft.npy'
    np.save(path, np.zeros((7, 4)))
    assert app.main(['run', str(path), '--project', 'path', '--engine', 'manylatents', '--recipe', 'embed']) == 0
    assert app.load_project('path')['dataset_name'] is None
    assert engine[0]['n_components'] == 4


@pytest.mark.parametrize('constructor', [Session, runner.run_manylatents])
def test_public_shape_validation(constructor, tmp_path):
    with pytest.raises(ValueError, match='two positive integers'):
        if constructor is Session:
            constructor('p', declared_shape=(True, 3))
        else:
            constructor({}, declared_shape=(True, 3), out_dir=tmp_path)


def test_modes_directly_transports_declared_shape_and_generator_settings(engine, tmp_path):
    row = catalog.load_dataset('arch_soft')
    result = modes.run('infer', {'engine': 'manylatents', 'dataset': 'archetypal',
                                'dataset_name': 'arch_soft', 'data_kwargs': row['params'],
                                'declared_shape': (600, 5), 'recipe': app.load_recipe('embed'),
                                'out_dir': tmp_path})
    assert result['ok']
    assert engine[0]['data_kwargs'] == row['params']
    assert engine[0]['n_components'] == 5
    assert any('50 → 5' in note for note in result['caveats'])


def test_cli_parameter_mismatch_refuses_before_project_or_compute(engine, tmp_path, capsys):
    assert app.main(['run', 'arch_soft', '--project', 'mismatch', '--engine', 'manylatents',
                     '--recipe', 'embed', '--data-set', 'n_obs=17']) != 0
    assert 'matching catalog declaration' in capsys.readouterr().err
    assert not (tmp_path / 'outputs').exists()
    assert engine == []


def test_session_result_keeps_bound_explanations(engine):
    session = Session('bounded', engine='manylatents', dataset='swissroll',
                      declared_shape=(5000, 3), recipe=app.load_recipe('embed'))
    session.step('pca')
    assert any('50 → 3' in note for note in session.close()['caveats'])


@pytest.mark.parametrize('name', ['torus', 'torus_noisy', 'torus_tight'])
def test_experiment_run_one_keeps_catalog_identity(engine, tmp_path, name):
    from manyruns import experiment

    row = catalog.load_dataset(name)
    result = experiment.run_one(experiment.Run(app.load_recipe('embed'), row),
                                engine='manylatents', out_dir=tmp_path)
    assert result.ok, result.error
    assert len(engine) == 2
    assert engine[0]['data'] == 'torus'
    assert engine[0].get('data_kwargs', {}) == row.get('params', {})
    assert engine[0]['n_components'] == 50
    assert result.status['pca'] == 'ok'


@pytest.mark.parametrize('kind', ['path', 'manylatents'])
def test_experiment_inline_name_collision_keeps_its_source(engine, tmp_path, kind):
    from manyruns import experiment

    path = tmp_path / 'local-torus.npy'
    matrix = np.arange(28, dtype=float).reshape(7, 4)
    np.save(path, matrix)
    row = {'name': 'torus', 'modality': 'synthetic', 'shape': 'manifold',
           'handle': {'kind': kind, 'ref': str(path) if kind == 'path' else 'swissroll'}}
    recipe = {'name': 'embed-only', 'steps': [s for s in app.load_recipe('embed')['steps']
                                           if s['group'] == 'latent']}
    run = experiment.Run(recipe, row)
    request = experiment._request(run, suite=None, engine='manylatents', out_dir=tmp_path)
    result = experiment.run_one(run, engine='manylatents', out_dir=tmp_path)
    assert result.ok, result.error
    assert request.get('dataset_name') is None
    assert len(engine) == 2
    if kind == 'path':
        np.testing.assert_array_equal(engine[0]['input_data'], matrix)
        assert engine[0]['n_components'] == 4
    else:
        assert engine[0]['data'] == 'swissroll'
        assert engine[0]['n_components'] == 3


def test_experiment_modified_generator_does_not_claim_catalog_identity(engine, tmp_path):
    from manyruns import experiment

    row = catalog.load_dataset('torus_noisy')
    row['params'] = {'num_points': 12}
    run = experiment.Run({'name': 'empty', 'steps': []}, row)
    request = experiment._request(run, suite=None, engine='manylatents', out_dir=tmp_path)
    assert request.get('dataset_name') is None
    assert request['data_kwargs'] == {'num_points': 12}


def test_generator_contract_survives_project_overwrite_in_durable_records(engine):
    from manyruns import store

    names = ['torus_noisy', 'torus_tight', 'torus_noisy']
    for name in names:
        assert app.main(['run', name, '--project', 'same', '--engine', 'manylatents', '--recipe', 'embed']) == 0
    rows = list(store.read())
    assert len(rows) == 3
    for row, name in zip(rows, names):
        assert row.get('dataset_name') == name
        assert row.get('data_kwargs') == catalog.load_dataset(name)['params']
        assert row['dataset'] == 'torus'
    assert rows[0]['spec_id'] != rows[1]['spec_id']
    assert rows[0]['spec_id'] == rows[2]['spec_id']
    assert len({row['run_id'] for row in rows}) == 3


def test_session_and_branch_record_effective_generator_contract(engine, tmp_path):
    from manyruns import store

    name = 'torus_noisy'
    app.write_project('saved', None, 'synthetic', app.load_recipe('embed'), engine='manylatents',
                      dataset='torus', dataset_name=name, data_kwargs=catalog.load_dataset(name)['params'])
    session = app._build_session(app.load_project('saved'), args(), resume=False)
    session.step('pca')
    branch = session.branch(0)
    for s in (session, branch):
        result = s.close()
        assert result.get('dataset_name') == name
        assert result.get('data_kwargs') == catalog.load_dataset(name)['params']
        store.append(result)
    rows = list(store.read())
    assert len(rows) == 2
    assert all(row.get('dataset_name') == name for row in rows)
    assert all(row.get('data_kwargs') == session.data_kwargs for row in rows)


def test_specification_identity_includes_effective_generator_settings(engine, tmp_path):
    results = [runner.run_manylatents({'name': 'empty', 'steps': []}, data_ref='torus',
                                      data_kwargs=params, out_dir=tmp_path)
               for params in [{'noise': .1, 'major_radius': 2}, {'noise': .4, 'major_radius': 2},
                              {'major_radius': 2, 'noise': .1}]]
    assert results[0]['spec_id'] != results[1]['spec_id']
    assert results[0]['spec_id'] == results[2]['spec_id']
    assert results[0].get('data_kwargs') == {'noise': .1, 'major_radius': 2}
