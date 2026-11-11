"""Differential expression over sample sums, in the isolated tool environment."""
import hashlib
import json
from pathlib import Path
import sys



def pair(value, name):
    if isinstance(value, str):
        value = json.loads(value)
    if (not isinstance(value, list) or len(value) != 2 or
            any(not isinstance(v, str) or not v for v in value) or value[0] == value[1]):
        raise ValueError(f'{name} must name two distinct values')
    return value


def analyze(req):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    data = ad.read_h5ad(req['input'])
    selection = pd.read_csv(req['tables']['selection'], sep='\t', dtype=str, keep_default_na=False)
    params = req['params']
    contrast = params.get('contrast', 'condition')
    minimum = params.get('min_cells_per_sample', 10)
    selected = data[selection.cell].copy()
    selected.obs['group'] = selection.group.to_numpy()
    counts = selected.X
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError('pseudobulk_de requires non-negative integer counts')
    if selected.obs.groupby('sample', observed=True).condition.nunique().max() != 1:
        raise ValueError('each sample must have exactly one condition')
    if contrast == 'condition':
        levels = pair(params.get('levels', ['case', 'control']), 'levels')
        groupby, factor, design = ['sample'], 'condition', '~condition'
    elif contrast == 'groups':
        groups = sorted(selection.group.unique())
        if len(groups) != 2 or '' in groups:
            raise ValueError('groups contrast needs exactly two group values')
        levels = pair(params.get('groups') or groups, 'groups')
        if set(levels) != set(groups):
            raise ValueError('groups must name the two selection group values')
        groupby, factor, design = ['sample', 'group'], 'group', '~sample + group'
    else:
        raise ValueError('contrast must be condition or groups')
    rows, metadata = [], []
    for _, indices in selected.obs.groupby(groupby, observed=True, sort=True).indices.items():
        obs = selected.obs.iloc[indices[0]]
        if str(obs[factor]) not in levels:
            continue
        rows.append(np.asarray(counts[indices].sum(axis=0)).reshape(-1))
        metadata.append({'sample': str(obs['sample']), factor: str(obs[factor]), 'n_cells': len(indices)})
    if not rows:
        raise ValueError('no selected cells have the requested levels')
    metadata = pd.DataFrame(metadata, index=[f'r{i}' for i in range(len(rows))])
    sums = pd.DataFrame(np.asarray(rows, dtype=np.int64), index=metadata.index, columns=data.var_names)
    if any(not (metadata[factor] == level).any() for level in levels):
        raise ValueError('both contrast levels need selected cells')
    eligible = metadata.n_cells >= minimum if contrast == 'condition' else pd.Series(True, index=metadata.index)
    n_samples = [int(((metadata[factor] == level) & eligible).sum()) for level in levels]
    reason = ''
    if contrast == 'condition' and min(n_samples) < 2:
        # Descriptive fallback uses all selected sample sums. A unit pseudocount is in CPM units.
        cpm = sums.div(sums.sum(axis=1).clip(lower=1), axis=0) * 1e6
        means = [cpm.loc[metadata[factor] == level].mean() for level in levels]
        log2fc = np.log2((means[0] + 1) / (means[1] + 1))
        table = pd.DataFrame({'score': log2fc, 'log2fc': log2fc, 'padj': '', 'base_mean': cpm.mean()})
        reason = f'each condition needs at least 2 samples with at least {minimum} selected cells'
    else:
        if contrast == 'groups':
            reason = 'groups were drawn on the same data'
        dds = DeseqDataSet(counts=sums.loc[eligible], metadata=metadata.loc[eligible],
                           design=design, n_cpus=1, quiet=True)
        dds.deseq2()
        stats = DeseqStats(dds, contrast=[factor, *levels], n_cpus=1, quiet=True)
        stats.summary()
        table = stats.results_df.rename(columns={'stat': 'score', 'log2FoldChange': 'log2fc',
                                                   'baseMean': 'base_mean'})
        if contrast == 'groups':
            table['padj'] = ''
    table['gene'] = table.index.astype(str)
    table['gene_id'] = data.var['gene_id'].reindex(table.index).fillna('')
    table = table[['gene', 'gene_id', 'score', 'log2fc', 'padj', 'base_mean']]
    table = table.sort_values('score', ascending=False, kind='stable', na_position='last')
    return table, {'padj_reason': reason, 'n_samples_case': n_samples[0],
                   'n_samples_control': n_samples[1], 'n_selected': len(selection)}


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
