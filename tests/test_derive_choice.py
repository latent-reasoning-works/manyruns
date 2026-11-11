import hashlib
import json

import pytest

from manyruns import action_space, actions, app, store
from manyruns.action_space import Refused

pytest_plugins = ['connector_fixtures']


def choice_for(fixture, connector='conn', **keys):
    doc = actions.enumerate_actions('derive', dataset='cells', run=fixture['parent']['run_id'])
    choice = {'schema_version': 1, 'surface': 'derive', 'dataset': 'cells',
              'parent_run': fixture['parent']['run_id'], 'offer_sha256': action_space.offer_sha256(doc),
              'chosen': {'connector': connector, 'params': {}}, 'chooser': 'human:test', 'bound': 'none'}
    entry = next(e for e in doc['offered'] if e['connector'] == connector)
    if entry['selection']:
        choice['selection_sha256'] = hashlib.sha256(fixture['selection'].read_bytes()).hexdigest()
    return {**choice, **keys}


def test_parent_run_is_required_on_derive_and_refused_elsewhere(store_with_parent):
    choice = choice_for(store_with_parent)
    del choice['parent_run']
    with pytest.raises(Refused, match='parent_run'):
        action_space.check_choice(choice)
    choice['surface'] = 'ledger'
    choice['parent_run'] = 'p'
    with pytest.raises(Refused, match='parent_run'):
        action_space.check_choice(choice)


@pytest.mark.parametrize('chosen', [{}, {'connector': 'conn'}, {'connector': 'conn', 'params': {}, 'recipe': 'embed'}])
def test_chosen_on_derive_is_exactly_connector_and_params(store_with_parent, chosen):
    with pytest.raises(Refused, match='exactly'):
        action_space.check_choice(choice_for(store_with_parent, chosen=chosen))


def test_a_blocked_connector_is_not_legal(store_with_parent):
    with pytest.raises(Refused, match='genes'):
        action_space.check_choice(choice_for(store_with_parent, connector='chain'))


@pytest.mark.parametrize('params', [{'unknown': 3}, {'top_n': 0}, {'top_n': 'three'}])
def test_a_param_outside_the_declaration_is_not_legal(store_with_parent, params):
    with pytest.raises(Refused, match='unknown|top_n'):
        action_space.check_choice(choice_for(store_with_parent, chosen={'connector': 'conn', 'params': params}))


def test_a_missing_role_key_must_be_given_in_the_choice(store_with_parent):
    parent = dict(store_with_parent['parent'])
    parent.pop('sample_key')
    store.append(parent, extra={'condition_key': parent['condition_key']})
    choice = choice_for(store_with_parent)
    with pytest.raises(Refused, match='sample_key'):
        action_space.check_choice(choice)
    action_space.check_choice({**choice, 'sample_key': 'sample'})


@pytest.mark.parametrize('digest', [None, 'abc', 'A' * 64, 1])
def test_selection_sha256_is_required_and_lowercase(store_with_parent, digest):
    choice = choice_for(store_with_parent)
    if digest is None:
        choice.pop('selection_sha256')
    else:
        choice['selection_sha256'] = digest
    with pytest.raises(Refused, match='selection_sha256'):
        action_space.check_choice(choice)


def test_selection_sha256_is_refused_for_a_connector_without_selection(store_with_parent, monkeypatch):
    from manyruns import toolchain
    import yaml
    path = toolchain.tool_dir() / 'chain.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['input']['tables'] = {}
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(Refused, match='selection_sha256'):
        action_space.check_choice(choice_for(store_with_parent, connector='chain', selection_sha256='0' * 64))


def test_selection_file_hash_must_equal_selection_sha256(store_with_parent, capsys):
    choice = choice_for(store_with_parent)
    path = store_with_parent['store'] / 'choice.json'
    path.write_text(json.dumps(choice))
    store_with_parent['selection'].write_text('cell\tgroup\nc2\ta\n')
    actual = hashlib.sha256(store_with_parent['selection'].read_bytes()).hexdigest()
    with pytest.raises(SystemExit) as exc:
        app.main(['run', '--choice', str(path), '--selection', str(store_with_parent['selection'])])
    assert exc.value.code == 2
    text = capsys.readouterr().err
    assert actual in text and choice['selection_sha256'] in text


def test_selection_flag_is_refused_without_a_derive_choice(store_with_parent, capsys):
    with pytest.raises(SystemExit):
        app.main(['run', '--selection', str(store_with_parent['selection'])])
    assert '--choice' in capsys.readouterr().err
    choice = choice_for(store_with_parent, surface='ledger')
    path = store_with_parent['store'] / 'choice.json'
    path.write_text(json.dumps(choice))
    with pytest.raises(SystemExit):
        app.main(['run', '--choice', str(path), '--selection', str(store_with_parent['selection'])])
    assert 'derive' in capsys.readouterr().err


def test_a_missing_selection_file_is_refused(store_with_parent, capsys):
    path = store_with_parent['store'] / 'choice.json'
    path.write_text(json.dumps(choice_for(store_with_parent)))
    with pytest.raises(SystemExit):
        app.main(['run', '--choice', str(path)])
    assert '--selection' in capsys.readouterr().err


def test_a_stale_derive_offer_is_refused(store_with_parent):
    choice = choice_for(store_with_parent)
    store.append({**store_with_parent['parent'], 'ok': False})
    with pytest.raises(Refused, match='stale'):
        action_space.check_choice(choice)


def test_review_keeps_parent_run_and_selection_sha256(store_with_parent):
    choice = choice_for(store_with_parent, parent_index=0)
    reviewed = action_space.review(choice, ask=lambda c: 'accept', human_id='reviewer')
    for key in ('parent_run', 'parent_index', 'selection_sha256'):
        assert reviewed[key] == choice[key]
    action_space.check_choice(reviewed)


@pytest.mark.parametrize('index', [True, '0', -1, 5])
def test_parent_index_must_name_a_persisted_slot(store_with_parent, index):
    with pytest.raises(Refused, match='parent_index'):
        action_space.check_choice(choice_for(store_with_parent, parent_index=index))


def test_parent_index_is_refused_off_derive(store_with_parent):
    with pytest.raises(Refused, match='parent_index'):
        choice = choice_for(store_with_parent, parent_index=0, surface='ledger')
        choice.pop('parent_run')
        choice.pop('selection_sha256')
        action_space.check_choice(choice)


def test_the_offer_lists_embedding_indices_and_row_counts(store_with_parent):
    doc = action_space.check_choice(choice_for(store_with_parent, parent_index=0))
    assert doc['embeddings'] == [{'index': 0, 'rows': 60}]


def test_the_label_of_a_derive_choice_is_derive_colon_connector(store_with_parent):
    choice = choice_for(store_with_parent)
    doc = action_space.check_choice(choice)
    assert action_space.label(choice, choice['chosen'], doc) == 'derive:conn'
