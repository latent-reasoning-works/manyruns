"""Saved spaces describe successful execution and inherited working matrices."""
import numpy as np
import pytest
from manyruns import artifacts
from manyruns.session import Session


def test_branch_inherits_working_matrix_preprocessing(tmp_path):
    session = Session('p', engine='_inproc', out_dir=tmp_path,
                      array=np.arange(120.).reshape(30, 4))
    def prep(name, params, state, g, ctx):
        return {'X': state['X'] + params['offset']}
    def latent(name, params, state, g, ctx):
        state['emb'] = state['X'][:, :2].copy()
    session.dispatch = dict(session.dispatch, prep=prep, latent=latent)
    session.apply({'name':'shift', 'group':'prep', 'params':{'offset':3}})
    session.apply({'name':'layout', 'group':'latent'})
    child = session.branch(1)
    assert child.ctx['input_sha256'] == session.ctx['input_sha256']
    child.dispatch = session.dispatch
    child.apply({'name':'layout', 'group':'latent'})
    child.close()
    entries = artifacts.read_manifest(tmp_path, child.run_id)
    path, entry = next((p,e) for p,e in entries.items() if e['slot']=='input_space')
    np.testing.assert_array_equal(np.load(path), session.state['X'])
    assert entry['input_space']['preprocessing']['steps'] == [
        {'name':'shift','params':{'offset':3}}]


def test_named_engine_saves_matrix_from_the_single_load(tmp_path, monkeypatch):
    api = pytest.importorskip('manylatents.api')
    from manyruns.pipeline import runner
    calls = []
    x = np.arange(120.).reshape(30,4)
    def run(*, return_input_data=False, **kwargs):
        calls.append(kwargs)
        assert return_input_data
        return {'embeddings':x[:,:2], 'input_data':x}
    monkeypatch.setattr(api, 'run', run)
    out = runner.run_manylatents({'steps':[{'name':'pca','group':'latent'}]},
        out_dir=tmp_path, data_ref='swissroll', metrics=[])
    assert out['ok'], out['steps']
    entries = artifacts.read_manifest(tmp_path,out['run_id'])
    path = next(p for p,e in entries.items() if e['slot']=='input_space')
    np.testing.assert_array_equal(np.load(path), x)
    assert len(calls)==1
    assert not any(e['slot'].endswith('input_data') for e in entries.values())


def test_real_named_engine_loads_once(tmp_path, monkeypatch):
    import inspect
    api = pytest.importorskip('manylatents.api')
    if 'return_input_data' not in inspect.signature(api.run).parameters:
        pytest.skip('installed engine does not expose its input matrix')
    from manyruns.pipeline import runner
    calls = []
    resolve = api._resolve_datamodule
    def counted(**kwargs):
        calls.append(kwargs)
        return resolve(**kwargs)
    monkeypatch.setattr(api, '_resolve_datamodule', counted)
    monkeypatch.setattr(runner._suite, 'measure', lambda *a, **k: {})
    out = runner.run_manylatents({'steps':[{'name':'pca','group':'latent',
                                          'params':{'n_components':2}}]},
        out_dir=tmp_path, data_ref='swissroll', metrics=[])
    assert out['ok'], [s.get('detail') for s in out['steps']]
    entries = artifacts.read_manifest(tmp_path,out['run_id'])
    path, entry = next((p,e) for p,e in entries.items() if e['slot']=='input_space')
    assert np.load(path).shape[1] == 3
    assert entry['input_space']['preprocessing']['steps'] == []
    assert len(calls) == 1
