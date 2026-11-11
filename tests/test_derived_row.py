import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from manyruns import action_space, actions, app, artifacts, catalog, decisions, derive, store
from manyruns.pipeline import runner
from test_derive_choice import choice_for

pytest_plugins = ['connector_fixtures']


def run_choice(fixture, connector='conn', **keys):
    choice = choice_for(fixture, connector, **keys)
    return app._run_choice(choice, engine=None, seed=42, device=None, project=None,
                           parser=app._build_parser(),
                           selection=fixture['selection'] if 'selection_sha256' in choice else None)


def last_row():
    return list(store.read())[-1]


def change_manifest(fixture, change):
    path = derive.state_dir(fixture['parent'], fixture['store']) / artifacts.DONE
    entries = json.loads(path.read_text())
    change(entries)
    path.write_text(json.dumps(entries))


def test_a_derived_run_is_a_new_lineage_whose_parent_is_the_named_step(store_with_parent):
    result = run_choice(store_with_parent, parent_index=0)
    row = last_row()
    assert result['exit_code'] == 0 and result['run_id'] == row['run_id']
    assert row['parent'] == {'run_id': store_with_parent['parent']['run_id'], 'index': 0}
    assert row['run_id'] != row['parent']['run_id']
    assert derive.state_dir(row, '.') != derive.state_dir(store_with_parent['parent'], '.')


def test_the_recipe_is_derive_colon_connector_and_the_decision_surface_is_derive(store_with_parent):
    run_choice(store_with_parent)
    row, decision = last_row(), list(decisions.read())[-1]
    assert row['recipe'] == decision['chosen'] == 'derive:conn'
    assert decision['surface'] == 'derive' and decision.get('gate') is None
    assert row['decision_id'] == decision['decision_id']
    assert list(row)[0] == list(decision)[0] == 'schema_version'


def test_a_derived_run_measures_no_geometry(store_with_parent, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('derived run measured geometry')
    monkeypatch.setattr(runner._suite, 'measure', forbidden)
    run_choice(store_with_parent)
    g = last_row()['g_vector']
    assert not set(catalog.load_suite()) & set(g)
    assert 'n_embedded' not in g and 'final_dim' not in g
    assert g['conn.selected'] == 2


def test_the_selection_is_kept_as_the_first_artifact(store_with_parent):
    run_choice(store_with_parent)
    row = last_row()
    directory = derive.state_dir(row, '.')
    path = directory / '00-derive_selection.tsv'
    entries = json.loads((directory / artifacts.DONE).read_text())
    assert entries[str(path)] == {'step': 'derive', 'index': 0, 'slot': 'table:selection'}
    assert row['g_vector']['derive.selection.sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert row['g_vector']['derive.selection.rows'] == 2
    assert path.read_text() == store_with_parent['selection'].read_text()
    assert directory.is_dir()
    assert not list(directory.glob('*.h5ad'))


def test_a_derived_row_is_not_a_probe(store_with_parent):
    run_choice(store_with_parent)
    assert not actions.is_probe(last_row(), 'cells')
    assert actions.probe_run('cells', 42)['run_id'] == store_with_parent['parent']['run_id']


def test_the_derived_frame_holds_the_parent_s_cells_in_order(store_with_parent):
    source = store_with_parent['source']
    original = ad.read_h5ad(source)
    original[::-1].copy().write_h5ad(source)
    run_choice(store_with_parent)
    exported = ad.read_h5ad(derive.state_dir(last_row(), '.').parent.parent / 'tool-conn/input.h5ad')
    assert list(exported.obs_names) == list(original.obs_names)
    np.testing.assert_array_equal(exported.X, original.X)
    assert list(exported.obs['sample']) == list(original.obs['sample'])
    assert list(exported.obs['condition']) == list(original.obs['condition'])


def test_a_derived_run_inherits_the_parent_s_role_keys(store_with_parent):
    run_choice(store_with_parent)
    assert last_row()['sample_key'] == 'sample' and last_row()['condition_key'] == 'condition'
    assert last_row()['engine'] == store_with_parent['parent']['engine']


def test_a_parent_without_recorded_cell_ids_is_refused_before_any_row(store_with_parent, capsys):
    change_manifest(store_with_parent, lambda entries: [e.pop('row_identity', None) for e in entries.values()])
    path = store_with_parent['store'] / 'choice.json'
    path.write_text(json.dumps(choice_for(store_with_parent)))
    assert app.main(['run', '--choice', str(path), '--selection', str(store_with_parent['selection'])]) == 1
    assert 'recorded cell ids at step 0' in capsys.readouterr().err
    assert len(list(store.read())) == 1 and not list(decisions.read())


def two_slots(fixture):
    directory = derive.state_dir(fixture['parent'], '.')
    path = directory / '03-filtered_emb.npy'
    np.save(path, np.zeros((40, 2)))
    def change(entries):
        old = next(iter(entries.values()))
        old['index'] = 1
        entries[str(path)] = {'step': 'filtered', 'index': 3, 'slot': 'emb',
                             'row_identity': {'sample_ids': [f'c{i}' for i in range(20, 60)]}}
    change_manifest(fixture, change)
    parent = dict(fixture['parent'])
    parent['steps'] = [*parent['steps'], {'index': 3, 'name': 'filtered', 'group': 'latent',
                                        'artifacts': {'emb': str(path)}, 'outcome': 'ok'}]
    store.append(parent, extra={'sample_key': 'sample', 'condition_key': 'condition'})
    fixture['parent'] = last_row()


def test_the_cells_come_from_the_slot_the_selection_was_drawn_on(store_with_parent):
    two_slots(store_with_parent)
    run_choice(store_with_parent, parent_index=1)
    row = last_row()
    assert row['parent']['index'] == 1 and row['g_vector']['derive.root_index'] == 1
    out = derive.state_dir(row, '.').parent.parent / 'tool-conn/input.h5ad'
    assert ad.read_h5ad(out).n_obs == 60
    with pytest.raises(action_space.Refused, match='step 3'):
        run_choice(store_with_parent)
    assert len(list(decisions.read())) == 1
    store_with_parent['selection'].write_text('cell\tgroup\nc20\ta\nc21\tb\n')
    run_choice(store_with_parent)
    assert last_row()['parent']['index'] == 3
    assert ad.read_h5ad(out).n_obs == 40


def test_a_named_slot_without_row_identity_is_refused(store_with_parent):
    two_slots(store_with_parent)
    def change(entries):
        for entry in entries.values():
            if entry['index'] == 1:
                entry.pop('row_identity', None)
    change_manifest(store_with_parent, change)
    with pytest.raises(action_space.Refused, match='recorded cell ids at step 1'):
        run_choice(store_with_parent, parent_index=1)
    assert not list(decisions.read())


def test_a_chain_reaches_the_root_s_cells_and_binds_the_first_derived_run_s_table(store_with_parent):
    two_slots(store_with_parent)
    run_choice(store_with_parent, parent_index=1)
    first = last_row()
    second_fixture = {**store_with_parent, 'parent': first}
    run_choice(second_fixture, connector='chain')
    row = last_row()
    assert row['parent'] == {'run_id': first['run_id'], 'index': 0}
    assert row['g_vector']['derive.root_run'] == store_with_parent['parent']['run_id']
    assert row['g_vector']['derive.root_index'] == 1 and row['g_vector']['n_samples'] == 60
    workdir = derive.state_dir(row, '.').parent.parent / 'tool-chain'
    request = json.loads((workdir / 'request.json').read_text())
    assert request['input'] is None and request['n_cells'] == 60
    assert Path(request['tables']['genes']).read_bytes() == Path(derive.parent_tables(first, '.')['genes']).read_bytes()


def test_readers_of_the_store_tolerate_a_derived_row(store_with_parent):
    run_choice(store_with_parent)
    assert actions.enumerate_actions('ledger', dataset='cells', gate=True)['surface'] == 'ledger'
    assert isinstance(decisions.examples(), list)
    assert isinstance(decisions.examples(surface=None), list)
    assert decisions._legal_set(list(decisions.read())[-1]['offered']) == ['derive:conn']


@pytest.mark.parametrize('fault', ['missing_file', 'missing_cell', 'engine_dataset'])
def test_unreachable_parent_data_is_refused_before_any_row(store_with_parent, fault):
    if fault == 'missing_file':
        store_with_parent['source'].unlink()
    elif fault == 'missing_cell':
        data = ad.read_h5ad(store_with_parent['source'])
        data[1:].copy().write_h5ad(store_with_parent['source'])
    else:
        import yaml
        path = store_with_parent['store'] / 'datasets/cells.yaml'
        cfg = yaml.safe_load(path.read_text())
        store_with_parent['source'].unlink()
        cfg['handle'] = {'kind': 'manylatents', 'ref': 'swissroll', 'provides': ['genes', 'samples']}
        path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(action_space.Refused) as exc:
        run_choice(store_with_parent)
    assert exc.value.exit_code == 1
    assert len(list(store.read())) == 1 and not list(decisions.read())


def test_an_explicit_different_engine_is_refused(store_with_parent):
    with pytest.raises(action_space.Refused, match='engine'):
        app._run_choice(choice_for(store_with_parent), engine='mock', seed=42, device=None,
                        project=None, parser=app._build_parser(), selection=store_with_parent['selection'])
    assert not list(decisions.read())


def test_a_skipped_connector_keeps_its_selection_and_is_not_gate_evidence(store_with_parent, monkeypatch):
    from manyruns.pipeline import external, steps
    def skip(*a, **k):
        raise steps._StepSkipped('fixture declines')
    monkeypatch.setattr(external, 'run_external_step', skip)
    run_choice(store_with_parent)
    row = last_row()
    assert row['ok'] is True and row['complete'] is False
    assert derive.state_dir(row, '.').joinpath('00-derive_selection.tsv').is_file()
    assert not actions.is_probe(row, 'cells')


def test_run_choice_cli_writes_the_derived_row(store_with_parent):
    path = store_with_parent['store'] / 'choice.json'
    path.write_text(json.dumps(choice_for(store_with_parent)))
    assert app.main(['run', '--choice', str(path), '--selection', str(store_with_parent['selection'])]) == 0
    assert last_row()['recipe'] == 'derive:conn'


def test_choice_role_keys_override_inherited_keys(store_with_parent):
    source = store_with_parent['source']
    data = ad.read_h5ad(source)
    data.obs['other_sample'] = ['new'] * len(data)
    data.write_h5ad(source)
    run_choice(store_with_parent, sample_key='other_sample', extra={'caller': {'tag': 'test'}})
    row = last_row()
    assert row['sample_key'] == 'other_sample' and row['extra'] == {'caller': {'tag': 'test'}}
    exported = ad.read_h5ad(derive.state_dir(row, '.').parent.parent / 'tool-conn/input.h5ad')
    assert set(exported.obs['sample']) == {'new'}


def test_selection_cannot_save_expression_columns_in_state(store_with_parent):
    store_with_parent['selection'].write_text('cell\tgroup\tcounts\nc0\ta\t300\n')
    with pytest.raises(action_space.Refused, match='only cell and group'):
        run_choice(store_with_parent)
    assert not list(decisions.read())


def test_reopen_errors_name_choice_keys(store_with_parent, capsys):
    source = store_with_parent['source']
    data = ad.read_h5ad(source)
    data.obs['sample'] = [''] * len(data)
    data.write_h5ad(source)
    run_choice(store_with_parent)
    row = last_row()
    assert row['complete'] is False
    assert 'sample_key' in row['steps'][0]['detail']
    assert '--sample-key' not in row['steps'][0]['detail']


def test_cli_wrapper_returns_the_ids_of_a_table_only_chain(store_with_parent):
    run_choice(store_with_parent)
    fixture = {**store_with_parent, 'parent': last_row()}
    space = action_space.ActionSpace('cells', mode='cli', store=fixture['store'])
    cli_offer = space.offer('derive', run=fixture['parent']['run_id'])
    library_offer = actions.enumerate_actions('derive', dataset='cells', run=fixture['parent']['run_id'])
    assert cli_offer == library_offer
    result = space.act(choice_for(fixture, connector='chain'))
    assert result['exit_code'] == 0
    assert result['run_id'] == last_row()['run_id']
    assert result['decision_id'] == last_row()['decision_id']


def test_reopen_resolves_a_relative_catalog_ref_in_the_drop_folder(store_with_parent, monkeypatch):
    import yaml
    fixture = store_with_parent
    path = fixture['store'] / 'datasets/cells.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['handle']['ref'] = 'data/cells.h5ad'
    path.write_text(yaml.safe_dump(cfg))
    drop = fixture['store'] / 'drop'
    drop.mkdir()
    fixture['source'].rename(drop / 'cells.h5ad')
    monkeypatch.setenv('MANYRUNS_DATA_DIR', str(drop))
    assert run_choice(fixture)['exit_code'] == 0


def test_reopen_prefers_the_recorded_source_even_without_a_catalog_name(store_with_parent):
    parent = {**store_with_parent['parent'], 'dataset_name': 'user-cells'}
    session = derive.open_derived(parent, 'conn', {}, {}, '.')
    assert session.ctx['source_path'] == store_with_parent['source']


def test_missing_source_names_the_paths_tried(store_with_parent):
    store_with_parent['source'].unlink()
    with pytest.raises(action_space.Refused, match=str(store_with_parent['source'])):
        run_choice(store_with_parent)


def test_a_verdict_does_not_replace_the_parent_run(store_with_parent):
    from manyruns import verdict, vocab
    verdict.write_verdict(store_with_parent['parent']['run_id'],
                          dict.fromkeys(vocab.TOPOLOGIES, 0.0), chooser='human:test')
    assert run_choice(store_with_parent)['exit_code'] == 0
    assert last_row()['parent']['run_id'] == store_with_parent['parent']['run_id']


@pytest.mark.parametrize('explicit, expected', [(False, 0), (True, 42)])
def test_derive_inherits_seed_unless_explicit(store_with_parent, explicit, expected):
    parent = {**store_with_parent['parent'], 'seed': 0}
    store.append(parent, extra={'sample_key': 'sample', 'condition_key': 'condition',
                                'source': str(store_with_parent['source'])})
    choice = choice_for(store_with_parent)
    path = store_with_parent['store'] / 'choice.json'
    path.write_text(json.dumps(choice))
    argv = ['run', '--choice', str(path), '--selection', str(store_with_parent['selection'])]
    assert app.main(argv + (['--seed', '42'] if explicit else [])) == 0
    assert last_row()['seed'] == expected


def test_failed_decision_write_reports_why_nothing_ran(store_with_parent, monkeypatch, capsys):
    monkeypatch.setattr(decisions, 'append', lambda **kw: None)
    assert run_choice(store_with_parent)['exit_code'] == 1
    assert 'error: decision not recorded; nothing was run' in capsys.readouterr().err
    assert len(list(store.read())) == 1


def test_a_file_parent_can_be_offered_and_derived_without_a_catalog_entry(store_with_parent):
    parent = {**store_with_parent['parent'], 'dataset_name': None}
    source = str(store_with_parent['source'])
    store.append(parent, extra={'source': source, 'modality': 'scrna',
                                'sample_key': 'sample', 'condition_key': 'condition'})
    doc = actions.enumerate_actions('derive', dataset=source, run=parent['run_id'])
    choice = {'schema_version': 1, 'surface': 'derive', 'dataset': source,
              'parent_run': parent['run_id'], 'offer_sha256': action_space.offer_sha256(doc),
              'chosen': {'connector': 'conn', 'params': {}}, 'chooser': 'human:test', 'bound': 'none',
              'sample_key': 'sample',
              'selection_sha256': hashlib.sha256(store_with_parent['selection'].read_bytes()).hexdigest()}
    path = store_with_parent['store'] / 'choice.json'
    path.write_text(json.dumps(choice))
    assert app.main(['run', '--choice', str(path), '--selection', str(store_with_parent['selection'])]) == 0
