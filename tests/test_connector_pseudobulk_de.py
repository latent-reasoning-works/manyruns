import subprocess
import sys
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from manyruns import toolchain
from manyruns.pipeline import external, steps
from connector_fixtures import bundled_connector, connector_table


@pytest.fixture(autouse=True)
def needs_pydeseq2():
    pytest.importorskip('pydeseq2')


def test_the_bundled_manifest_passes_check_tool_and_declares_a_source():
    cfg = toolchain.load_tool('pseudobulk_de')
    assert not toolchain.check_tool(cfg, 'pseudobulk_de')
    assert cfg['source'] and cfg['params']['levels'] == ['case', 'control']


def test_condition_contrast_ranks_the_planted_genes_first(tmp_path, monkeypatch):
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de')
    g = {}
    external.run_external_step('pseudobulk_de', {}, state, g, ctx)
    genes = connector_table(tmp_path, 'pseudobulk_de', 'genes')
    assert set(genes.head(5).gene) == {f'g{i}' for i in range(5)}
    assert all(genes.head(5).padj != '')
    assert (genes.head(5).log2fc > 0).all()
    assert g['pseudobulk_de.n_samples_case'] == g['pseudobulk_de.n_samples_control'] == 4
    assert g['pseudobulk_de.n_selected'] == 240


def test_condition_contrast_with_one_sample_per_level_writes_no_padj_and_says_why(tmp_path, monkeypatch):
    data, _, _ = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de')
    data.obs['sample'] = data.obs['condition'].astype(str)
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de', data)
    g = {}
    external.run_external_step('pseudobulk_de', {}, state, g, ctx)
    genes = connector_table(tmp_path, 'pseudobulk_de', 'genes')
    assert set(genes.padj) == {''}
    assert g['pseudobulk_de.padj_reason']
    np.testing.assert_allclose(genes.score, genes.log2fc)


def test_groups_contrast_writes_no_padj(tmp_path, monkeypatch):
    data, _, _ = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de')
    selection = pd.DataFrame({'cell': data.obs_names, 'group': ['A', 'B'] * 120})
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de', data, selection)
    g = {}
    external.run_external_step('pseudobulk_de', {'contrast': 'groups'}, state, g, ctx)
    assert set(connector_table(tmp_path, 'pseudobulk_de', 'genes').padj) == {''}
    assert g['pseudobulk_de.padj_reason'] == 'groups were drawn on the same data'


@pytest.mark.parametrize('groups', [['A'], ['A', 'B', 'C']])
def test_groups_contrast_needs_exactly_two_group_values(tmp_path, monkeypatch, groups):
    data, _, _ = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de')
    selection = pd.DataFrame({'cell': data.obs_names, 'group': (groups * 240)[:240]})
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de', data, selection)
    with pytest.raises(steps._StepSkipped, match='exactly two group values'):
        external.run_external_step('pseudobulk_de', {'contrast': 'groups'}, state, {}, ctx)


def test_gene_id_comes_from_the_frame(tmp_path, monkeypatch):
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de')
    external.run_external_step('pseudobulk_de', {'min_cells_per_sample': 1000}, state, {}, ctx)
    table = connector_table(tmp_path, 'pseudobulk_de', 'genes').set_index('gene')
    assert table.loc['g3', 'gene_id'] == 'ENSG00000000004'


def test_gene_ids_prepare_writes_symbol_and_gene_id_columns(tmp_path):
    source = tmp_path / 'hgnc.tsv'
    source.write_text('symbol\tensembl_gene_id\tprev_symbol\talias_symbol\n'
                      'ONE\tENSG00000000001\tOLD\tALIAS|SHARED\n'
                      'TWO\tENSG00000000002\t\tSHARED\n')
    out = tmp_path / 'prepared.tsv'
    script = files('manyruns') / 'adapters/gene_ids_prepare.py'
    subprocess.run([sys.executable, str(script), '--url', source.as_uri(), '--release', 'test',
                    '--out', str(out)], check=True)
    table = pd.read_csv(out, sep='\t').set_index('symbol')
    assert table.loc['OLD', 'gene_id'] == table.loc['ONE', 'gene_id'] == 'ENSG00000000001'
    assert table.loc['ALIAS', 'gene_id'] == 'ENSG00000000001'
    assert 'SHARED' not in table.index


def test_custom_condition_levels_accept_a_json_pair(tmp_path, monkeypatch):
    data, _, _ = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de')
    data.obs['condition'] = data.obs['condition'].map({'case': 'treated', 'control': 'baseline'})
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de', data)
    external.run_external_step('pseudobulk_de', {'levels': '["treated", "baseline"]',
                                               'min_cells_per_sample': 1000}, state, {}, ctx)
    assert set(connector_table(tmp_path, 'pseudobulk_de', 'genes').head(5).gene) == {f'g{i}' for i in range(5)}


def test_groups_contrast_does_not_apply_the_condition_replication_threshold(tmp_path, monkeypatch):
    data, _, _ = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de')
    selection = pd.DataFrame({'cell': data.obs_names, 'group': ['A', 'B'] * 120})
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'pseudobulk_de', data, selection)
    g = {}
    external.run_external_step('pseudobulk_de', {'contrast': 'groups', 'min_cells_per_sample': 1000}, state, g, ctx)
    assert set(connector_table(tmp_path, 'pseudobulk_de', 'genes').padj) == {''}
    assert g['pseudobulk_de.n_samples_case'] == g['pseudobulk_de.n_samples_control'] == 8
