"""Descriptive marker genes for selected cells in the isolated tool environment."""
import hashlib
import json
from pathlib import Path
import sys


def analyze(req):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.stats import rankdata

    data = ad.read_h5ad(req['input'])
    selection = pd.read_csv(req['tables']['selection'], sep='\t', dtype=str, keep_default_na=False)
    groups = sorted(selection.group.unique())
    if len(groups) not in (1, 2) or '' in groups:
        raise ValueError('selection_markers needs one or two group values')
    inside = data.obs_names.get_indexer(selection.loc[selection.group == groups[0], 'cell'])
    if len(groups) == 2:
        outside = data.obs_names.get_indexer(selection.loc[selection.group == groups[1], 'cell'])
        contrast = 'a_vs_b'
    else:
        outside = np.flatnonzero(~data.obs_names.isin(selection.cell))
        contrast = 'a_vs_rest'
    if not selection.cell.isin(data.obs_names).all():
        raise ValueError('selection cells are absent from the frame')
    counts = data.X
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('selection_markers requires non-negative finite counts')
    params = req['params']
    fraction = params.get('subsample', 1.0)
    if fraction < 1:
        rng = np.random.default_rng(req['seed'])
        inside = rng.choice(inside, round(fraction * len(inside)), replace=False)
        outside = rng.choice(outside, round(fraction * len(outside)), replace=False)
    n_in, n_out = len(inside), len(outside)
    minimum = params.get('min_cells', 10)
    if min(n_in, n_out) < minimum:
        raise ValueError(f'each side needs at least {minimum} cells: n_in={n_in}, n_out={n_out}')
    counts = counts[np.concatenate([inside, outside])]
    libraries = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    scale = np.divide(np.median(libraries), libraries, out=np.zeros_like(libraries), where=libraries > 0)
    # Zero-library cells remain zero. Process genes in blocks to bound dense rank storage.
    rows = []
    for start in range(0, data.n_vars, 256):
        block = counts[:, start:start + 256]
        raw = block.toarray() if sparse.issparse(block) else np.asarray(block)
        prepared = np.log1p(raw * scale[:, None])
        frac_in = (raw[:n_in] > 0).mean(axis=0)
        frac_out = (raw[n_in:] > 0).mean(axis=0)
        tested = ((np.maximum(frac_in, frac_out) >= params.get('min_frac', 0.1)) &
                  (np.ptp(prepared, axis=0) > 0))
        indices = np.flatnonzero(tested)
        prepared = prepared[:, tested]
        ranks = rankdata(prepared, axis=0, method='average')
        scores = (ranks[:n_in].sum(axis=0) - n_in * (n_in + 1) / 2) / (n_in * n_out) - 0.5
        rows.append(pd.DataFrame({
            'gene': data.var_names[start + indices].astype(str),
            'gene_id': data.var['gene_id'].iloc[start + indices].fillna('').astype(str).to_numpy(),
            'score': scores,
            'log2fc': (prepared[:n_in].mean(axis=0) - prepared[n_in:].mean(axis=0)) / np.log(2),
            'mean_expr': prepared.mean(axis=0),
            'frac_in': frac_in[tested], 'frac_out': frac_out[tested],
        }))
    columns = ['gene', 'gene_id', 'score', 'log2fc', 'mean_expr', 'frac_in', 'frac_out']
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)
    table = table.sort_values(['score', 'gene_id', 'gene'], ascending=[False, True, True])
    table['rank'] = np.arange(1, len(table) + 1)
    table['n_in'], table['n_out'] = n_in, n_out
    table = table[['gene', 'gene_id', 'score', 'rank', 'log2fc', 'mean_expr',
                   'frac_in', 'frac_out', 'n_in', 'n_out']]
    return table, {'contrast': contrast, 'n_in': n_in, 'n_out': n_out,
                   'genes_tested': len(table), 'genes_untested': data.n_vars - len(table)}


def main(request_path):
    req = json.loads(Path(request_path).read_text())
    try:
        table, scalars = analyze(req)
    except ValueError as exc:
        Path(req['result']).write_text(json.dumps({'ok': False, 'reason': str(exc)}))
        return 0
    path = Path(req['workdir']) / 'genes.tsv'
    table.to_csv(path, sep='\t', index=False, na_rep='')
    result = {'schema_version': 1, 'ok': True, 'n_cells': req['n_cells'], 'facts': {},
              'scalars': scalars, 'tables': {'genes': {
                  'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    Path(req['result']).write_text(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
