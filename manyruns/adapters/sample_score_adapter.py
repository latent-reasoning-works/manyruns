"""Describe a selected population across samples, using samples as replicates."""
import hashlib
import json
from pathlib import Path
import sys



def analyze(req):
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse
    from scipy.stats import mannwhitneyu

    data = sc.read_h5ad(req['input'])
    selection = pd.read_csv(req['tables']['selection'], sep='\t', dtype=str, keep_default_na=False)
    selected = np.asarray(data.obs_names.isin(selection.cell))
    if not selected.any() or selected.all():
        raise ValueError('sample_score needs selected cells and cells outside the selection')
    if data.obs.groupby('sample', observed=True).condition.nunique().max() != 1:
        raise ValueError('each sample must have exactly one condition')
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('sample_score requires finite non-negative counts')
    data.X = data.X.astype(float)
    sc.pp.normalize_total(data, target_sum=1e6)
    sc.pp.log1p(data)
    difference = np.asarray(data.X[selected].mean(axis=0) - data.X[~selected].mean(axis=0)).reshape(-1)
    genes = np.argsort(-difference, kind='stable')[:req['params'].get('top_genes', 20)]
    scores = np.asarray(data.X[:, genes].mean(axis=1)).reshape(-1)
    rows = []
    for sample, indices in data.obs.groupby('sample', observed=True, sort=True).indices.items():
        rows.append({'sample': str(sample), 'n_cells': len(indices), 'fraction': float(selected[indices].mean()),
                     'score': float(scores[indices].mean()), 'condition': str(data.obs.condition.iloc[indices[0]])})
    table = pd.DataFrame(rows)
    levels = sorted(table.condition.unique())
    stat, pvalue, reason = '', '', ''
    if len(levels) != 2:
        reason = 'association needs exactly two condition levels'
    else:
        fractions = [table.loc[table.condition == level, 'fraction'] for level in levels]
        if min(map(len, fractions)) < 2:
            reason = 'each condition needs at least 2 samples'
        else:
            test = mannwhitneyu(*fractions, alternative='two-sided')
            stat, pvalue = float(test.statistic), float(test.pvalue)
    return table, {'assoc_stat': stat, 'assoc_pvalue': pvalue, 'assoc_reason': reason}


def main(request_path):
    req = json.loads(Path(request_path).read_text())
    try:
        table, scalars = analyze(req)
    except ValueError as exc:
        Path(req['result']).write_text(json.dumps({'ok': False, 'reason': str(exc)}))
        return 0
    path = Path(req['workdir']) / 'sample_scores.tsv'
    table.to_csv(path, sep='\t', index=False)
    result = {'schema_version': 1, 'ok': True, 'n_cells': req['n_cells'], 'facts': {},
              'scalars': scalars, 'tables': {'sample_scores': {
                  'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    Path(req['result']).write_text(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
