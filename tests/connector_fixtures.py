"""Small local data and tools for connector tests."""
import numpy as np


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
