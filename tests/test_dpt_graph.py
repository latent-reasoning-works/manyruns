"""Public SwissRoll and seeded NumPy fixtures for trajectory inputs."""
import numpy as np
import pytest

from manyruns import catalog
from manyruns.pipeline import runner, steps


def test_recipe_orders_dpt_on_pca():
    names = [step['name'] for step in catalog.load_recipe('pseudotime')['steps']]
    assert names == ['normalize', 'transform', 'pca', 'dpt', 'diffusionmap', 'discretize_time']


@pytest.mark.parametrize('fixture', ['swissroll', 'curve'])
def test_connected_input_and_diffusion_root(fixture, monkeypatch):
    sc = pytest.importorskip('scanpy')
    from anndata import AnnData
    from scipy.sparse.csgraph import connected_components
    from sklearn.decomposition import PCA

    if fixture == 'swissroll':
        from manylatents.data.synthetic_dataset import SwissRoll
        X = SwissRoll(n_distributions=1000, n_points_per_distribution=1,
                      noise=0, manifold_noise=0, random_state=1).data
    else:
        t = np.sort(np.random.default_rng(1).uniform(size=1000))
        X = np.column_stack((t, np.sin(t), np.cos(t)))
    Z = PCA(n_components=3).fit_transform(X).astype(np.float32)
    ad = AnnData(Z)
    sc.pp.neighbors(ad, n_neighbors=15, use_rep='X')
    assert connected_components(ad.obsp['connectivities'])[0] == 1
    sc.tl.diffmap(ad, n_comps=15)
    expected = int(np.argmin(ad.obsm['X_diffmap'][:, 1]))
    roots = []
    original = steps._diffusion_pseudotime

    def geodesic(*args, **kwargs):
        roots.append(kwargs.get('root'))
        return original(*args, **kwargs)

    monkeypatch.setattr(steps, '_diffusion_pseudotime', geodesic)
    for _ in range(2):
        state, g = {'emb': Z, 'emb_from': 'pca'}, {}
        steps._step_dpt(state, g, {}, None, [], 3)
        assert np.isfinite(state['pseudotime']).all()
        assert g['dpt_root'] == expected
        assert g['dpt.root_rule'] == 'diffusion_component_1_min'
        assert g['dpt.graph_on'] == 'pca'
        for key in ('dpt.root_rule', 'dpt.graph_on'):
            assert g['_provenance'][key]['kind'] == 'setting'
    assert roots == [expected, expected]


@pytest.mark.parametrize('recover', [False, True])
def test_failed_coordinates_refuse_stale_input(tmp_path, monkeypatch, recover):
    X = np.random.default_rng(1).normal(size=(400, 3))
    states = []

    def pca(state, *args):
        state['emb'] = state['X'].copy()

    def fail(state, *args):
        states.append(state)
        raise ValueError('coordinate failure')

    monkeypatch.setitem(steps._INPROC_STEPS, 'pca', pca)
    monkeypatch.setitem(steps._INPROC_STEPS, 'diffusionmap', fail)
    recipe_steps = [{'name': name, 'group': 'latent'} for name in ['pca', 'diffusionmap']]
    if recover:
        recipe_steps.append({'name': 'pca', 'group': 'latent'})
    recipe_steps.append({'name': 'dpt', 'group': 'analysis'})
    result = runner.run_inproc(X, {'name': 'public_curve', 'steps': recipe_steps}, tmp_path, seed=1)
    rec = result['steps'][-1]
    assert rec['outcome'] == ('ok' if recover else 'skipped')
    if recover:
        assert result['g_vector']['dpt.graph_on'] == 'pca'
    else:
        assert 'diffusionmap' in rec['detail'] and 'pca' in rec['detail']
        assert states[0].get('pseudotime') is None


def test_latent_chain_refuses_stale_input():
    state = {'emb': np.random.default_rng(1).normal(size=(400, 3)),
             'emb_from': 'pca', 'emb_stale': 'diffusionmap'}
    with pytest.raises(steps._StepSkipped, match='diffusionmap.*pca'):
        runner._ml_latent('pca', {}, state, {}, {'seed': 1})


def test_latent_can_refit_raw_input_when_no_embedding_remains(monkeypatch, tmp_path):
    import manylatents.api

    X = np.random.default_rng(1).normal(size=(400, 3))
    received = []

    def run(**kwargs):
        received.append(kwargs['input_data'])
        return {'embeddings': X.copy()}

    monkeypatch.setattr(manylatents.api, 'run', run)
    state = runner._new_state(X=X)
    state['emb_stale'] = 'diffusionmap'
    runner._ml_latent('pca', {}, state, {},
                      {'seed': 1, 'array': X, 'out_dir': tmp_path, 'plots': []})
    np.testing.assert_array_equal(received[0], X)


@pytest.mark.parametrize('name,replace,stale', [
    ('diffusionmap', False, True),
    ('diffusionmap', True, False),
    ('leiden', False, False),
    ('reeb_graph', False, False),
    ('mioflow', False, False),
])
def test_only_failed_coordinates_retained_from_before_are_stale(tmp_path, name, replace, stale):
    X = np.random.default_rng(1).normal(size=(400, 3))
    state = runner._new_state(X=X, emb=X)
    state['emb_from'] = 'pca'

    def fail(name, params, state, g, ctx):
        if replace:
            state['emb'] = X.copy()
        raise ValueError('coordinate failure')

    group = 'lightning' if name == 'mioflow' else 'latent'
    rec = runner.apply_step(
        {'name': name, 'group': group}, state, {}, dispatch={group: fail},
        ctx={'out_dir': tmp_path, 'plots': [], 'target_dim': 3}, index=0,
        carry=runner.new_carry(),
    )
    assert rec['outcome'] == 'error'
    assert bool(state.get('emb_stale')) is stale


def test_supplied_embedding_has_input_provenance():
    state = {'emb': np.random.default_rng(1).normal(size=(400, 3))}
    g = {}
    steps._step_dpt(state, g, {}, None, [], 3)
    assert g['dpt.graph_on'] == 'input'


@pytest.mark.parametrize('name', ['mioflow', 'separation', 'composition', 'cluster_quality',
                                  'rank_genes', 'dpt', 'simplex', 'velocity_field'])
def test_readouts_refuse_stale_coordinates(name):
    state = {'emb': np.random.default_rng(1).normal(size=(400, 3)),
             'emb_from': 'pca', 'emb_stale': 'diffusionmap'}
    with pytest.raises(steps._StepSkipped, match='diffusionmap.*pca'):
        getattr(steps, '_step_' + name)(state, {}, {}, None, [], 3)


def test_discretize_time_keeps_a_pseudotime_computed_before_the_failure():
    """The new order runs dpt before diffusionmap; a later diffusionmap failure must not
    discard the bins of a pseudotime that was computed on valid coordinates."""
    rng = np.random.default_rng(1)
    state = {'emb': rng.normal(size=(400, 3)), 'emb_from': 'pca', 'emb_stale': 'diffusionmap',
             'pseudotime': rng.uniform(size=400)}
    g = {}
    steps._step_discretize_time(state, g, {'n_timepoints': 5}, None, [], 3)
    assert g['discretize_time.n_timepoints'] == 5


def test_an_unavailable_engine_score_is_recorded_as_unavailable(tmp_path, monkeypatch):
    """manylatents keeps a declined metric as a MeasurementUnavailable value. The record keeps
    its `to_dict()` form, never a bare string, and the gate reads it as not measured."""
    import json

    import manylatents.api
    from manylatents.utils.exceptions import MeasurementUnavailable

    from manyruns import vocab

    X = np.random.default_rng(1).normal(size=(400, 3)).astype(np.float32)

    def run(**kwargs):
        return {'embeddings': X.copy(), 'scores': {
            'lid': MeasurementUnavailable('only 4 distinct points for k=20'),
            'trustworthiness': 0.9}}

    monkeypatch.setattr(manylatents.api, 'run', run)
    state, g = runner._new_state(X=X), {}
    runner._ml_latent('diffusionmap', {}, state, g,
                      {'seed': 1, 'array': X, 'out_dir': tmp_path, 'plots': [],
                       'metrics': ['lid', 'trustworthiness']})
    assert g['diffusionmap.lid'] == {'status': 'unavailable',
                                     'reason': 'only 4 distinct points for k=20'}
    assert g['diffusionmap.trustworthiness'] == 0.9
    assert json.loads(json.dumps(g['diffusionmap.lid'])) == g['diffusionmap.lid']
    assert vocab._measured_number(g['diffusionmap.lid']) is None
