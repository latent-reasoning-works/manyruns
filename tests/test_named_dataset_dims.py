"""Pinned generator fit sets; only the runtime oracle needs the engine."""
import pytest
from manyruns import app, catalog

DIMS = {
    'swissroll': (5000, 3), 'saddlesurface': (4000, 3), 'torus': (794, 100),
    'torus_noisy': (474, 100), 'torus_tight': (474, 100),
    'gaussian_blob': (100, 2), 'blob_loose': (600, 2), 'blob_tight': (600, 2),
    'arch_sharp': (600, 4), 'arch_soft': (600, 5),
    'tree_narrow': (800, 3), 'tree_wide': (2000, 3),
}
assert DIMS


def test_exact_generator_declarations():
    rows = catalog.load_datasets()
    generators = {row['name']: row for row in rows if row['handle']['kind'] == 'manylatents'}
    assert set(generators) == set(DIMS)
    for name, shape in DIMS.items():
        assert generators[name]['dims'] == dict(zip(('n_samples', 'n_features'), shape))


@pytest.mark.parametrize('name', DIMS)
def test_generator_contract(name):
    row = catalog.load_dataset(name)
    params, shape = app._dataset_contract(name, row['handle']['ref'], {})
    assert params == row.get('params', {})
    assert shape == DIMS[name]
    params, shape = app._dataset_contract(name, row['handle']['ref'], {'seed': 7})
    assert params == {**row.get('params', {}), 'seed': 7}
    assert shape is None


def test_contract_does_not_guess_identity():
    assert app._dataset_contract(None, 'archetypal', {}) == ({}, None)
    assert app._dataset_contract(None, 'dla_tree', {}) == ({}, None)
    assert app._dataset_contract(None, 'swissroll', {}) == ({}, (5000, 3))
    with pytest.raises(ValueError, match='does not match'):
        app._dataset_contract('tree_narrow', 'torus', {})


@pytest.mark.parametrize('name', DIMS)
def test_runtime_fit_tensor_oracle(name):
    api = pytest.importorskip('manylatents.api')
    from manyruns.pipeline.loading import _stack_dataset
    row = catalog.load_dataset(name)
    dm = api._resolve_datamodule(data=row['handle']['ref'], seed=42, **row.get('params', {}))
    dm.setup()
    assert _stack_dataset(dm.train_dataset).shape == DIMS[name]
