import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from manyruns import action_space, app, decisions, derive, store

pytest_plugins = ['connector_fixtures']


def cli_derive(fixture, connector, params, capsys, selection=True):
    capsys.readouterr()
    assert app.main(['actions', 'cells', '--surface', 'derive', '--run', fixture['parent']['run_id']]) == 0
    doc = json.loads(capsys.readouterr().out)
    choice = {'schema_version': 1, 'surface': 'derive', 'dataset': 'cells',
              'parent_run': fixture['parent']['run_id'], 'offer_sha256': action_space.offer_sha256(doc),
              'chosen': {'connector': connector, 'params': params}, 'chooser': 'human:test', 'bound': 'none'}
    if selection:
        choice['selection_sha256'] = hashlib.sha256(fixture['selection'].read_bytes()).hexdigest()
    path = fixture['store'] / 'choice.json'
    path.write_text(json.dumps(choice))
    argv = ['run', '--choice', str(path)]
    if selection:
        argv += ['--selection', str(fixture['selection'])]
    assert app.main(argv) == 0
    row = list(store.read())[-1]
    assert row['ok'] and row['complete'], row['steps']
    decision = list(decisions.read())[-1]
    assert row['recipe'] == f'derive:{connector}'
    assert decision['surface'] == 'derive' and decision['decision_id'] == row['decision_id']
    return row


def test_derive_end_to_end_with_stub_connectors(store_with_parent, capsys):
    fixture = store_with_parent
    first = cli_derive(fixture, 'conn', {}, capsys)
    directory = derive.state_dir(first, '.')
    assert (directory / '00-derive_selection.tsv').is_file()
    assert (directory / '00-conn_genes.tsv').is_file()
    second = cli_derive({**fixture, 'parent': first}, 'chain', {}, capsys, selection=False)
    assert second['parent']['run_id'] == first['run_id']
    assert Path(derive.parent_tables(second, '.')['genes']).is_file()


def test_derive_end_to_end_ranks_the_planted_genes_then_chains_target_evidence(planted_store, capsys):
    pytest.importorskip('pydeseq2')
    fixture = planted_store
    first = cli_derive(fixture, 'pseudobulk_de', {'contrast': 'condition'}, capsys)
    directory = derive.state_dir(first, '.')
    assert (directory / '00-derive_selection.tsv').is_file()
    assert (directory / '00-pseudobulk_de_genes.tsv').is_file()
    genes = pd.read_csv(directory / '00-pseudobulk_de_genes.tsv', sep='\t')
    assert set(genes.head(5).gene) == {f'g{i}' for i in range(5)}
    assert genes.head(5).padj.notna().all()
    second = cli_derive({**fixture, 'parent': first}, 'target_evidence',
                        {'disease': 'EFO_fixture', 'top_n': 5}, capsys, selection=False)
    assert second['parent']['run_id'] == first['run_id']
    evidence = pd.read_csv(derive.parent_tables(second, '.')['gene_evidence'], sep='\t')
    assert set(evidence.gene_id) == {f'ENSG{i + 1:011d}' for i in range(5)}
    assert second['g_vector']['derive.root_run'] == fixture['parent']['run_id']
