"""Recipe and CLI acceptance with contract stubs and importable real helpers."""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manyruns import app, catalog, profile
from manyruns.pipeline import features, runner, shape_profile

pytest_plugins = ['connector_fixtures']


@pytest.fixture(params=['stub', 'real'])
def helpers(request, monkeypatch):
    if request.param == 'real':
        pytest.importorskip('manylatents.metrics.shape_profile')
        try:
            features._helper()
        except ImportError:
            raise
        except Exception as exc:
            from manyruns.pipeline.steps import _StepUnsupported
            if not isinstance(exc, _StepUnsupported):
                raise
            import features_stub
            def feature_rows(*args, **kwargs):
                return features_stub.stub_rows(*args, **kwargs) + [
                    dict(key='counts_like', value=False, type='bool', status='ok', reason=None,
                         source='counts_like/1', definition_version='dataset_features/1')]
            monkeypatch.setattr(features, '_helper', lambda: feature_rows)
    else:
        import features_stub
        import shape_profile_stub
        def rows(*args, **kwargs):
            return features_stub.stub_rows(*args, **kwargs) + [
                dict(key='counts_like', value=False, type='bool', status='ok', reason=None,
                     source='counts_like/1', definition_version='dataset_features/1')]
        monkeypatch.setattr(features, '_helper', lambda: rows)
        monkeypatch.setattr(shape_profile, '_helper', lambda: shape_profile_stub.stub_rows)
        monkeypatch.setattr(profile, '_pca_helper', lambda: lambda x, **kw: x.copy())
    return request.param


def test_2000_by_50_recipe_matches_raw_oracle(tmp_path, helpers):
    from scipy.spatial.distance import cdist
    x = np.random.default_rng(914).normal(size=(2000,50))
    recipe = catalog.load_recipe('profile')
    first = runner.run_inproc(x, recipe, tmp_path / 'first')
    second = runner.run_inproc(x, recipe, tmp_path / 'second')
    assert first['ok'] and first['complete'], first['steps']
    path = Path(first['steps'][2]['artifacts']['shape_profile'])
    assert path.read_bytes() == Path(second['steps'][2]['artifacts']['shape_profile']).read_bytes()
    table = pd.read_csv(path, sep='\t')
    assert len(table) == 15 and not table.value.isin([np.inf, -np.inf]).any()
    if helpers == 'real':
        distances = cdist(x,x)
        np.fill_diagonal(distances, np.inf)
        distances.sort(axis=1)
        expected = np.median(np.log(distances[:,4] / distances[:,14]))
        actual = table.query("stat_name == 'log_rk_r15' and scale_value == 5 and aggregation == 'median'").value.item()
        assert abs(actual - expected) <= 1e-9


def test_cli_selection_on_finished_recipe_with_layout(store_with_parent, capsys, helpers):
    from test_derive_end_to_end import cli_derive
    fixture = store_with_parent
    fixture['selection'].write_text('cell\tgroup\n' + ''.join(f'c{i}\ta\n' for i in range(30)))
    row = cli_derive(fixture, 'shape_profile', {'extended':False}, capsys)
    table = pd.read_csv(row['steps'][0]['artifacts']['shape_profile'], sep='\t', keep_default_na=False)
    assert len(table) == 56
    selected = table[table.entity_kind == 'selection']
    assert set(selected.entity_id) == {row['g_vector']['derive.selection.sha256']}
    assert set(selected.parent_entity_id) == set(table[table.entity_kind == 'dataset'].entity_id)
    assert set(selected.stat_name) >= {'conductance', 'selection_fidelity', 'relative_log_rk_r15'}
    assert 'layout_unavailable' not in set(selected.reason)


PINS = [(name,key,value) for name,group,params in profile.PINS for key,value in params.items()]


@pytest.mark.parametrize('name,key,value', PINS)
def test_each_explicit_parameter_override_is_refused(name, key, value):
    recipe = catalog.load_recipe('profile')
    with pytest.raises(ValueError, match=f'pinned_profile_parameter:{name}.{key}'):
        app.apply_step_params(recipe, name, {key:value})
    assert profile.check_pins(recipe, overrides={name:{key:value}}) == [
        f'pinned_profile_parameter:{name}.{key}']


@pytest.mark.parametrize('entry', ['catalog','session','execution','session_mutation'])
@pytest.mark.parametrize('name,key,value', PINS)
def test_changed_pins_are_refused_at_each_boundary(tmp_path, entry, name, key, value):
    from manyruns.session import Session
    recipe = catalog.load_recipe('profile')
    changed = copy.deepcopy(recipe)
    step = next(s for s in changed['steps'] if s['name']==name)
    step['params'][key] = 'invalid'
    if entry == 'catalog':
        assert any('pinned_profile_parameter' in e for e in catalog.check_recipe(changed, 'profile'))
    elif entry == 'session':
        with pytest.raises(ValueError, match='pinned_profile_parameter'):
            Session('p', engine='_inproc', out_dir=tmp_path, recipe=changed, array=np.ones((30,4)))
    elif entry == 'execution':
        with pytest.raises(ValueError, match='pinned_profile_parameter'):
            runner.run_inproc(np.ones((30,4)), changed, tmp_path)
    else:
        session = Session('p', engine='_inproc', out_dir=tmp_path, recipe=recipe, array=np.ones((30,4)))
        session.recipe['steps'] = changed['steps']
        with pytest.raises(ValueError, match='pinned_profile_parameter'):
            session.run_recipe()


def test_profile_transform_override_is_refused():
    with pytest.raises(ValueError, match='pinned_profile_parameter:profile_prepare.transform'):
        app.apply_transform_shim(catalog.load_recipe('profile'), 'sqrt')


def test_profile_exposes_no_tuning_path():
    from manyruns import actions
    document = actions._tune('profile', None)
    assert document['offered'] == []
    for name, _, _ in profile.PINS:
        with pytest.raises(actions.UsageError, match='no tunable step'):
            actions._tune('profile', name)


def test_cli_refuses_transform_override(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'points.npy'
    np.save(source, np.ones((30,4)))
    assert app.main(['run', str(source), '--recipe', 'profile', '--engine', 'mock',
                     '--transform', 'sqrt']) == 1
    assert 'pinned_profile_parameter:profile_prepare.transform' in capsys.readouterr().err
    assert not (tmp_path / 'outputs').exists()


def test_cli_refuses_replaced_profile_recipe(tmp_path, monkeypatch, capsys):
    import yaml
    recipe = catalog.load_recipe('profile')
    recipe['steps'][1]['params']['scale'] = True
    recipes = tmp_path / 'recipes'
    recipes.mkdir()
    (recipes / 'profile.yaml').write_text(yaml.safe_dump(recipe))
    monkeypatch.setenv('MANYRUNS_RECIPE_DIR', str(recipes))
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'points.npy'
    np.save(source, np.ones((30,4)))
    assert app.main(['run', str(source), '--recipe', 'profile', '--engine', 'mock']) == 1
    assert 'pinned_profile_parameter:profile_prepare.scale' in capsys.readouterr().err
    assert not (tmp_path / 'outputs').exists()
