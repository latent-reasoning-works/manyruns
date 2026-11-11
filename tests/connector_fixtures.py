"""Small local data and tools for connector tests."""
import numpy as np
import pytest


def planted_h5ad(path, ensembl=True):
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(12)
    x = rng.poisson(5, (60, 6))
    x[:30, :2] += 30
    data = ad.AnnData(x, obs=pd.DataFrame({
        'sample': [f's{i // 10}' for i in range(60)],
        'condition': ['case'] * 30 + ['control'] * 30,
    }, index=[f'c{i}' for i in range(60)]),
        var=pd.DataFrame(index=[f'g{i}' for i in range(6)]))
    if ensembl:
        data.var['gene_ids'] = [f'ENSG{i + 1:011d}.2' for i in range(6)]
    data.write_h5ad(path)
    return data


def tiny_gene_ids(tmp_path, monkeypatch):
    from tool_stubs import make_resource
    return make_resource(tmp_path, monkeypatch, 'conn', 'gene_ids', {
        'symbol': ['g0', 'g1', 'g2'],
        'gene_id': [f'ENSG{i + 1:011d}' for i in range(3)],
    })


@pytest.fixture
def store_with_parent(tmp_path, monkeypatch):
    import yaml
    from manyruns import store
    from manyruns.pipeline import runner
    from tool_stubs import INPUT_TABLE_ADAPTER, make_tool

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'matplotlib'))
    source = tmp_path / 'cells.h5ad'
    data = planted_h5ad(source)
    datasets = tmp_path / 'datasets'
    datasets.mkdir()
    (datasets / 'cells.yaml').write_text(yaml.safe_dump({
        'name': 'cells', 'shape': 'clusters', 'modality': 'scrna', 'topology': ['clusters'],
        'handle': {'kind': 'path', 'ref': str(source), 'provides': ['genes', 'samples']},
    }))
    monkeypatch.setenv('MANYRUNS_DATASET_DIR', str(datasets))
    for name, inputs, needs in [
        ('conn', {'kind': 'state_frame', 'obs': ['sample', 'condition'],
                  'tables': {'selection': ['cell', 'group']}}, ['genes', 'samples']),
        ('chain', {'kind': 'none', 'tables': {'genes': ['gene', 'gene_id', 'score']}}, []),
        ('missing', {'kind': 'none', 'tables': {'absent': ['gene']}}, []),
        ('unmet', {'kind': 'state_frame'}, ['velocity']),
    ]:
        make_tool(tmp_path, monkeypatch, name, {
            'input': inputs, 'needs': needs, 'produces': [],
            'output': {'schema': 'result-v1', 'tables': {'genes': ['gene', 'gene_id', 'score']}},
            'params': {'top_n': 3},
            'declaration': {'schema_version': 1, 'origin': 'test', 'method': name, 'group': 'tool', 'params': {
                'top_n': {'legal': {'type': 'int', 'min': 1, 'max': 10, 'min_open': False,
                                    'max_open': False, 'choices': [], 'also': []},
                          'default': 3, 'sensible': None, 'ladder': None, 'limits': [],
                          'role': 'knob', 'precomputed': False, 'cost': None,
                          'meaning': 'Number of genes', 'more': None, 'less': None},
            }},
        }, INPUT_TABLE_ADAPTER)
    make_tool(tmp_path, monkeypatch, 'embedder', {
        'input': {'kind': 'state_matrix'}, 'produces': ['embedding'],
        'output': {'schema': 'result-v1'},
    }, """import json, sys, hashlib
from pathlib import Path
import numpy as np
req = json.loads(Path(sys.argv[1]).read_text())
x = np.load(req['input'])
u, s, v = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
path = Path(req['workdir']) / 'embedding.npy'
np.save(path, u[:, :2] * s[:2])
Path(req['result']).write_text(json.dumps({'schema_version': 1, 'ok': True,
    'n_cells': len(x), 'facts': {'embedding': {'path': path.name,
    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}))
""")
    recipe = {'name': 'embed', 'steps': [{'name': 'embedder', 'group': 'latent', 'tool': 'embedder'}]}
    with monkeypatch.context() as local:
        local.setattr(runner._suite, 'measure', lambda *a, **k: {})
        local.setattr(runner, '_attach_geometry', lambda *a, **k: {})
        result = runner.run_inproc(data.X, recipe, tmp_path / 'outputs/project',
                                   sample_ids=np.asarray(data.obs_names), counts=data.X,
                                   genes=np.asarray(data.var_names), dataset='cells')
    result['dataset_name'] = 'cells'
    store.append(result, extra={'sample_key': 'sample', 'condition_key': 'condition',
                                'source': str(source), 'modality': 'scrna'})
    parent = list(store.read())[-1]
    selection = tmp_path / 'selection.tsv'
    selection.write_text('cell\tgroup\nc0\ta\nc1\tb\n')
    return {'store': tmp_path, 'parent': parent, 'source': source, 'selection': selection,
            'manifest': next(iter(parent['steps'][0]['artifacts'].values()))}
