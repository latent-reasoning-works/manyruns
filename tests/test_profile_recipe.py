import numpy as np
import pytest
import features_stub
import shape_profile_stub
from manyruns import catalog, artifacts
from manyruns.pipeline import features, runner, shape_profile


@pytest.fixture
def helpers(monkeypatch):
    def rows(*args, **kwargs):
        return features_stub.stub_rows(*args, **kwargs) + [
            dict(key='counts_like', value=False, type='bool', status='ok', reason=None,
                 source='counts_like/1', definition_version='dataset_features/1')]
    from manyruns import profile
    monkeypatch.setattr(profile, "_pca_helper", lambda: lambda points, **kw: points.astype(np.float64))
    monkeypatch.setattr(features, '_helper', lambda: rows)
    monkeypatch.setattr(shape_profile, '_helper', lambda: shape_profile_stub.stub_rows)


def test_profile_runs_without_choice_and_writes_both_tables(tmp_path, helpers):
    recipe = catalog.load_recipe('profile')
    out = runner.run_inproc(np.random.default_rng(2).normal(size=(600,5)), recipe, tmp_path)
    assert out['complete'], out['steps']
    assert out['kind'] == 'profile'
    assert set(out['steps'][0]['artifacts']) >= {'dataset_features'}
    assert 'shape_profile' in out['steps'][2]['artifacts']
    manifest = artifacts.read_manifest(tmp_path, out['run_id'])
    entries = [e for e in manifest.values() if e['slot'] == 'input_space']
    assert len(entries) == 1
    assert entries[0]['input_space']['input_space'] == 'pca5'


def test_profile_kind_is_a_declared_field():
    from manyruns import vocab
    assert vocab.RECIPE_KINDS == ('recipe', 'profile')
    assert vocab.RUN_KINDS == ('recipe', 'profile', 'derived')


def test_profile_pins_reject_mutated_declaration(tmp_path, helpers):
    recipe = catalog.load_recipe('profile')
    recipe['steps'][1]['params']['scale'] = True
    with pytest.raises(ValueError, match='pinned_profile_parameter'):
        runner.run_inproc(np.ones((600,5)), recipe, tmp_path)


def test_every_run_saves_input_space(tmp_path, helpers):
    x = np.ones((30,4))
    out = runner.run_inproc(x, {'steps':[{'name':'dataset_features','group':'analysis'}]}, tmp_path)
    manifest = artifacts.read_manifest(tmp_path, out['run_id'])
    entries = [(p,e) for p,e in manifest.items() if e['slot'] == 'input_space']
    assert len(entries) == 1
    np.testing.assert_array_equal(np.load(entries[0][0]), x)
    assert entries[0][1]['input_space']['preprocessing']['branch'] == 'none'


@pytest.mark.parametrize('value,kind,expected', [('raw_integer','point_cloud','counts'), ('processed','point_cloud','point_cloud'), ('processed','annotated','processed')])
def test_classifier_enum_routes_by_input_kind(value, kind, expected):
    from manyruns.profile import classify_branch
    assert classify_branch([{'key':'counts_like','value':value,'status':'ok'}], input_kind=kind) == expected


@pytest.mark.parametrize('sparse_input,width,reason', [(False,513,'too_many_columns'), (True,5,'sparse')])
def test_input_space_limits(tmp_path, sparse_input, width, reason):
    from scipy import sparse
    from manyruns.profile import save_input_space
    x = np.ones((20,width))
    if sparse_input:
        x = sparse.csr_matrix(x)
    state = runner._new_state(X=x)
    g, ctx, carry = {}, {'out_dir':tmp_path}, {'written':{}}
    save_input_space(state,g,ctx,carry,[{"name":"prep", "outcome":"ok"}])
    assert g == {'input_space.saved':False, 'input_space.reason':reason}
    assert not carry['written']


def test_equal_profile_override_is_refused():
    from manyruns.app import apply_step_params
    with pytest.raises(ValueError, match='pinned_profile_parameter:shape_profile.extended'):
        apply_step_params(catalog.load_recipe('profile'), 'shape_profile', {'extended':False})


def test_profile_session_preserves_kind(tmp_path, helpers):
    from manyruns.session import Session
    session = Session(project='profile', engine='_inproc', out_dir=tmp_path,
                      array=np.ones((600,5)), recipe=catalog.load_recipe('profile'))
    session.run_recipe()
    out = session.close()
    assert out['kind'] == 'profile' and out['complete'], out['steps']


def test_profile_is_legal_for_every_catalog_dataset():
    from manyruns import vocab
    assert catalog.dataset_dir().is_dir()
    recipe = catalog.load_recipe('profile')
    for name in catalog.discover_datasets():
        data = catalog.load_dataset(name)
        assert not vocab.unmet(recipe, data.get('shape','unknown'), handle=data.get('handle'))


def test_saved_space_matches_table_preprocessing(tmp_path, helpers):
    import pandas as pd
    out = runner.run_inproc(np.ones((600,5)), catalog.load_recipe('profile'), tmp_path)
    manifest = artifacts.read_manifest(tmp_path, out['run_id'])
    saved = next(e['input_space'] for e in manifest.values() if e['slot'] == 'input_space')
    table = pd.read_csv(out['steps'][2]['artifacts']['shape_profile'], sep='\t')
    assert set(table['preprocessing_sha256']) == {saved['preprocessing_sha256']}


def test_input_space_is_a_declared_slot():
    assert 'input_space' in artifacts.PERSISTED


def test_narrowing_invalidates_saved_input_space(tmp_path):
    from manyruns.pipeline.runner import _apply_transition
    state = runner._new_state(X=np.ones((30,4)))
    state['input_space'] = state['X'].copy()
    ctx = {'array': state['X'], 'input_space_attempted':True, 'color':None}
    _apply_transition({'mask':np.arange(30)<20, 'axis':'rows'},state,{'name':'cut'},ctx)
    assert state['input_space'] is None
    assert ctx['input_space_attempted'] is False
