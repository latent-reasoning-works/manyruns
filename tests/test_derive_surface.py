import json

import pytest

from manyruns import action_space, actions, app, store
pytest_plugins = ["connector_fixtures"]


def offer(fixture, **kwargs):
    return actions.enumerate_actions('derive', dataset='cells', run=fixture['parent']['run_id'], store=fixture['store'], **kwargs)


def test_the_derive_document_lists_each_connector_with_its_legality(store_with_parent):
    doc = offer(store_with_parent)
    assert doc['surface'] == 'derive' and doc['gate'] is None and doc['evidence'] is None
    assert doc['parent'] == {'run_id': store_with_parent['parent']['run_id'], 'recipe': 'embed', 'index': 0}
    entries = {e['connector']: e for e in doc['offered']}
    assert list(entries) == sorted(entries)
    assert entries['conn']['can_run'] and entries['conn']['selection']
    assert entries['conn']['binds'] == []
    assert entries['conn']['execution'] == 'external'
    assert entries['shape_profile']['execution'] == 'inprocess'
    assert entries['shape_profile']['blocked'] == []
    assert entries['conn']['role_keys'] == {'sample': 'sample', 'condition': 'condition'}
    assert not entries['chain']['can_run'] and 'genes' in ' '.join(entries['chain']['blocked'])
    assert not entries['missing']['can_run'] and 'absent' in ' '.join(entries['missing']['blocked'])
    assert not entries['unmet']['can_run'] and 'velocity' in ' '.join(entries['unmet']['blocked'])
    assert 'columns' not in doc['dataset']


def test_a_role_key_absent_from_the_parent_is_null_not_blocking(store_with_parent):
    parent = dict(store_with_parent['parent'])
    parent.pop('sample_key')
    store.append(parent)
    entry = next(e for e in offer(store_with_parent)['offered'] if e['connector'] == 'conn')
    assert entry['role_keys']['sample'] is None and entry['can_run']


def test_the_derive_document_is_deterministic_and_has_a_digest(store_with_parent):
    doc = offer(store_with_parent)
    assert doc == offer(store_with_parent)
    assert len(action_space.offer_sha256(doc)) == 64


@pytest.mark.parametrize('flags', [[], ['--recipe', 'embed'], ['--step', 'pca'], ['--gate', 'off']])
def test_derive_needs_run_and_refuses_recipe_step_and_gate(store_with_parent, flags):
    args = ['actions', 'cells', '--surface', 'derive']
    if flags:
        args += ['--run', store_with_parent['parent']['run_id']] + flags
    with pytest.raises(SystemExit) as exc:
        app.main(args)
    assert exc.value.code == 2


def test_run_is_refused_on_other_surfaces(store_with_parent):
    with pytest.raises(actions.UsageError, match='--run'):
        actions.enumerate_actions('ledger', dataset='cells', run=store_with_parent['parent']['run_id'])


def test_an_unknown_run_or_a_run_of_another_dataset_is_refused(store_with_parent):
    with pytest.raises(actions.UsageError, match='run'):
        actions.enumerate_actions('derive', dataset='cells', run='absent')
    store.append({**store_with_parent['parent'], 'dataset_name': 'other'})
    with pytest.raises(actions.UsageError, match='dataset'):
        offer(store_with_parent)


def test_library_and_cli_offers_are_equal(store_with_parent, capsys):
    run = store_with_parent['parent']['run_id']
    assert app.main(['actions', 'cells', '--surface', 'derive', '--run', run]) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli == action_space.ActionSpace('cells', store=store_with_parent['store']).offer('derive', run=run)


def test_pyrovelocity_is_not_offered_as_a_connector(store_with_parent, monkeypatch):
    monkeypatch.delenv('MANYRUNS_TOOL_DIR')
    assert 'pyrovelocity' not in [e['connector'] for e in offer(store_with_parent)['offered']]
