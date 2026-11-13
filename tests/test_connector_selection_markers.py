import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from manyruns import toolchain
from manyruns.pipeline import external, steps
from connector_fixtures import bundled_connector, connector_table

NAME = 'selection_markers'
COLUMNS = ['gene', 'gene_id', 'score', 'rank', 'log2fc', 'mean_expr',
           'frac_in', 'frac_out', 'n_in', 'n_out']


def setup(tmp_path, monkeypatch, groups=2, mutate=None):
    data, _, _ = bundled_connector(tmp_path, monkeypatch, NAME)
    if mutate:
        mutate(data)
    selection = pd.DataFrame({'cell': data.obs_names, 'group': ['A'] * 120 + ['B'] * 120})
    if groups == 1:
        selection = selection.iloc[:120]
    elif groups == 3:
        selection['group'] = ['A', 'B', 'C'] * 80
    return bundled_connector(tmp_path, monkeypatch, NAME, data, selection)


def run(tmp_path, state, ctx, **params):
    scalars = {}
    external.run_external_step(NAME, params, state, scalars, ctx)
    return connector_table(tmp_path, NAME, 'genes'), scalars


def test_manifest():
    cfg = toolchain.load_tool(NAME)
    assert not toolchain.check_tool(cfg, NAME)
    assert 'seed' not in cfg['params']
    assert 'seed' not in cfg['declaration']['params']
    assert not cfg['input'].get('obs')
    assert cfg['needs'] == ['genes']
    assert cfg['output']['tables']['genes'] == COLUMNS


@pytest.mark.parametrize('groups,contrast', [(1, 'a_vs_rest'), (2, 'a_vs_b')])
def test_planted_genes_and_columns(tmp_path, monkeypatch, groups, contrast):
    _, state, ctx = setup(tmp_path, monkeypatch, groups)
    table, scalars = run(tmp_path, state, ctx)
    assert set(table.head(5).gene) == {f'g{i}' for i in range(5)}
    assert (table.head(5).log2fc > 0).all()
    assert list(table.columns) == COLUMNS
    assert 'padj' not in table
    assert scalars[f'{NAME}.contrast'] == contrast
    assert scalars[f'{NAME}.n_in'] == scalars[f'{NAME}.n_out'] == 120
    assert scalars[f'{NAME}.genes_tested'] == len(table) == 200
    assert table.set_index('gene').loc['g3', 'gene_id'] == 'ENSG00000000004'


def test_three_groups_refused(tmp_path, monkeypatch):
    _, state, ctx = setup(tmp_path, monkeypatch, 3)
    with pytest.raises(steps._StepSkipped, match='one or two group values'):
        run(tmp_path, state, ctx)


def test_absent_cell_refused(tmp_path, monkeypatch):
    _, state, ctx = setup(tmp_path, monkeypatch)
    pd.DataFrame({'cell': ['absent'], 'group': ['A']}).to_csv(
        ctx['input_tables']['selection'], sep='\t', index=False)
    with pytest.raises(steps._StepSkipped, match='absent from the frame'):
        run(tmp_path, state, ctx)


@pytest.mark.parametrize('fraction', [0.5, 1.0])
def test_run_seed(tmp_path, monkeypatch, fraction):
    _, state, ctx = setup(tmp_path, monkeypatch)
    contents = []
    for seed in [17, 17, 29]:
        ctx['seed'] = seed
        table, _ = run(tmp_path, state, ctx, subsample=fraction)
        assert set(table.n_in) == set(table.n_out) == {round(120 * fraction)}
        contents.append((tmp_path / f'out/tool-{NAME}/genes.tsv').read_bytes())
    assert contents[0] == contents[1]
    assert (contents[0] == contents[2]) == (fraction == 1.0)


def test_ties_constants_and_filter(tmp_path, monkeypatch):
    def mutate(data):
        data.X[:, 6] = data.X[:, 5]
        data.var.loc['g5', 'gene_ids'] = 'ENSG99999999999'
        data.X[:, 7] = 0
        data.X[:, 8] = 0
        data.X[0, 8] = 1
    _, state, ctx = setup(tmp_path, monkeypatch, mutate=mutate)
    table, scalars = run(tmp_path, state, ctx)
    assert not {'g7', 'g8'} & set(table.gene)
    assert scalars[f'{NAME}.genes_untested'] == 2
    assert len(table) == 198
    tied = table.set_index('gene')
    assert tied.loc['g5', 'score'] == tied.loc['g6', 'score']
    assert tied.loc['g6', 'rank'] < tied.loc['g5', 'rank']
    assert list(table['rank']) == list(range(1, len(table) + 1))


def test_minimum_counts(tmp_path, monkeypatch):
    _, state, ctx = setup(tmp_path, monkeypatch)
    with pytest.raises(steps._StepSkipped, match=r'n_in=120.*n_out=120'):
        run(tmp_path, state, ctx, min_cells=121)
    with pytest.raises(steps._StepSkipped, match=r'n_in=0.*n_out=0'):
        run(tmp_path, state, ctx, subsample=0.001)


def test_scores_and_expression_match_reference(tmp_path, monkeypatch):
    data, state, ctx = setup(tmp_path, monkeypatch)
    table, _ = run(tmp_path, state, ctx)
    counts = np.asarray(data.X, dtype=float)
    libraries = counts.sum(axis=1)
    prepared = np.log1p(counts * (np.median(libraries) / libraries)[:, None])
    for gene in ['g0', 'g5', 'g42']:
        j = data.var_names.get_loc(gene)
        row = table.set_index('gene').loc[gene]
        expected = mannwhitneyu(prepared[:120, j], prepared[120:, j]).statistic / 120**2 - 0.5
        assert row.score == pytest.approx(expected)
        assert row.log2fc == pytest.approx((prepared[:120, j].mean() - prepared[120:, j].mean()) / np.log(2))
        assert row.mean_expr == pytest.approx(prepared[:, j].mean())
        assert row.frac_in == pytest.approx((counts[:120, j] > 0).mean())
        assert row.frac_out == pytest.approx((counts[120:, j] > 0).mean())


@pytest.mark.parametrize('value', [-1, np.nan, np.inf])
def test_invalid_counts(tmp_path, monkeypatch, value):
    def mutate(data):
        data.X = data.X.astype(float)
        data.X[0, 0] = value
    _, state, ctx = setup(tmp_path, monkeypatch, mutate=mutate)
    with pytest.raises(steps._StepSkipped, match='non-negative.*finite|finite.*non-negative'):
        run(tmp_path, state, ctx)


def test_no_sample_or_condition_roles_and_sparse_counts(tmp_path, monkeypatch):
    from scipy import sparse

    data, state, ctx = setup(tmp_path, monkeypatch)
    dense, _ = run(tmp_path, state, ctx)
    data.obs = pd.DataFrame(index=data.obs_names)
    data.X = sparse.csr_matrix(data.X)
    data.write_h5ad(ctx['source_path'])
    state['obs'] = {}
    state['counts'] = data.X
    state['X'] = data.X
    table, _ = run(tmp_path, state, ctx)
    pd.testing.assert_frame_equal(dense, table)


def test_zero_libraries_and_all_constant_genes(tmp_path, monkeypatch):
    def mutate(data):
        data.X[:] = 0
    _, state, ctx = setup(tmp_path, monkeypatch, mutate=mutate)
    table, scalars = run(tmp_path, state, ctx, min_frac=0)
    assert table.empty
    assert list(table.columns) == COLUMNS
    assert scalars[f'{NAME}.genes_tested'] == 0
    assert scalars[f'{NAME}.genes_untested'] == 200
