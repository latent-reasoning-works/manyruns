import numpy as np
import pandas as pd

from manyruns import toolchain
from manyruns.pipeline import external
from connector_fixtures import bundled_connector, connector_table


def sample_input(tmp_path, monkeypatch, single=False):
    data, _, _ = bundled_connector(tmp_path, monkeypatch, 'sample_score')
    if single:
        data.obs['sample'] = data.obs['condition'].astype(str)
    selected = [cell for i, cell in enumerate(data.obs_names) if i % 30 < (20 if i < 120 else 5)]
    selection = pd.DataFrame({'cell': selected, 'group': 'selected'})
    return bundled_connector(tmp_path, monkeypatch, 'sample_score', data, selection)


def test_the_bundled_manifest_passes_check_tool():
    cfg = toolchain.load_tool('sample_score')
    assert not toolchain.check_tool(cfg, 'sample_score')
    assert cfg['licence'] == 'BSD-3-Clause'
    assert cfg['needs'] == ['samples']


def test_each_sample_gets_one_row_with_its_fraction(tmp_path, monkeypatch):
    _, state, ctx = sample_input(tmp_path, monkeypatch)
    g = {}
    external.run_external_step('sample_score', {}, state, g, ctx)
    result = connector_table(tmp_path, 'sample_score', 'sample_scores')
    assert len(result) == result['sample'].nunique() == 8
    assert set(result.n_cells) == {30}
    np.testing.assert_allclose(result.loc[result.condition == 'case', 'fraction'], 20 / 30)
    np.testing.assert_allclose(result.loc[result.condition == 'control', 'fraction'], 5 / 30)
    assert 0 <= g['sample_score.assoc_pvalue'] <= 1
    assert g['sample_score.assoc_reason'] == ''


def test_the_condition_with_the_planted_population_scores_higher(tmp_path, monkeypatch):
    _, state, ctx = sample_input(tmp_path, monkeypatch)
    external.run_external_step('sample_score', {'top_genes': 5}, state, {}, ctx)
    result = connector_table(tmp_path, 'sample_score', 'sample_scores')
    assert result.loc[result.condition == 'case', 'score'].min() > result.loc[result.condition == 'control', 'score'].max()


def test_score_uses_all_sample_cells_and_log1p_cpm(tmp_path, monkeypatch):
    data, state, ctx = sample_input(tmp_path, monkeypatch)
    external.run_external_step('sample_score', {'top_genes': 5}, state, {}, ctx)
    result = connector_table(tmp_path, 'sample_score', 'sample_scores').set_index('sample')
    normalized = np.log1p(data.X / data.X.sum(axis=1)[:, None] * 1e6)
    for sample in data.obs['sample'].unique():
        expected = normalized[np.asarray(data.obs['sample'] == sample), :5].mean()
        np.testing.assert_allclose(result.loc[sample, 'score'], expected)


def test_one_sample_per_level_gives_no_pvalue_and_says_why(tmp_path, monkeypatch):
    _, state, ctx = sample_input(tmp_path, monkeypatch, single=True)
    g = {}
    external.run_external_step('sample_score', {}, state, g, ctx)
    assert g['sample_score.assoc_pvalue'] == ''
    assert g['sample_score.assoc_reason']
