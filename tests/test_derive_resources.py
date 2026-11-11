"""Derive offers describe local resources before a caller chooses a connector."""
import hashlib
import json

import pytest
import yaml

from manyruns import action_space, app, resources, toolchain
from test_derive_choice import choice_for
from test_derive_surface import offer

pytest_plugins = ['connector_fixtures']


@pytest.fixture
def resource_connector(store_with_parent, tmp_path, monkeypatch):
    monkeypatch.setenv('MANYRUNS_RESOURCE_HOME', str(tmp_path / 'resources'))

    def declare(name='knowledge', key=None):
        path = toolchain.tool_dir() / 'conn.yaml'
        cfg = yaml.safe_load(path.read_text())
        block = {'release': '1', 'url': 'https://example.org/resource.tsv',
                 'licence': 'CC0-1.0', 'commercial_use': True, 'prepare': 'prepare.py'}
        if key:
            block['key'] = key
            cfg['params'][key] = 'EFO_default'
            cfg['declaration']['params'][key] = {
                **cfg['declaration']['params']['top_n'], 'default': 'EFO_default',
                'legal': {'type': 'str', 'choices': [], 'also': [], 'min': None,
                          'max': None, 'min_open': False, 'max_open': False},
            }
        cfg.setdefault('resource', {})[name] = block
        path.write_text(yaml.safe_dump(cfg))
        return block

    return store_with_parent, declare


def built_resource(name, block, key=None, *, mismatch=False):
    path = resources.path_for(name, block, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('gene_id\tscore\nENSG00000000001\t1\n')
    path.with_suffix('.tsv.sha256').write_text(
        '0' * 64 if mismatch else hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def entry_for(fixture):
    return next(e for e in offer(fixture)['offered'] if e['connector'] == 'conn')


def test_required_keyless_resource_blocks_until_built_and_changes_digest(resource_connector):
    fixture, declare = resource_connector
    block = declare()
    before = offer(fixture)
    entry = entry_for(fixture)
    assert not entry['can_run']
    assert entry['blocked'] == ['resource_not_built:knowledge:1']
    assert entry['resources'] == [{
        'name': 'knowledge', 'release': '1', 'key_param': None, 'required': 'always',
        'built': False, 'status': 'not-built', 'build': 'manyruns tools build-resource knowledge',
    }]
    choice = choice_for(fixture)
    with pytest.raises(action_space.Refused, match='^resource_not_built:knowledge:1') as exc:
        action_space.check_choice(choice)
    assert exc.value.exit_code == 2
    built_resource('knowledge', block)
    entry = entry_for(fixture)
    assert entry['can_run'] and entry['blocked'] == []
    assert entry['resources'][0]['built'] is True
    assert entry['resources'][0]['status'] == 'ready'
    assert action_space.offer_sha256(before) != action_space.offer_sha256(offer(fixture))
    with pytest.raises(action_space.Refused, match='stale offer'):
        action_space.check_choice(choice)
    action_space.check_choice(choice_for(fixture))


def test_keyed_resource_lists_only_ready_keys_and_checks_choices(resource_connector):
    fixture, declare = resource_connector
    block = declare(key='disease')
    before = choice_for(fixture)
    for key in ['EFO_z', 'EFO_a', 'EFO_default']:
        built_resource('knowledge', block, key)
    built_resource('knowledge', block, 'EFO_bad', mismatch=True)
    unsigned = built_resource('knowledge', block, 'EFO_unsigned')
    unsigned.with_suffix('.tsv.sha256').unlink()
    # Other releases are not ready keys for this release.
    built_resource('knowledge', {**block, 'release': '2'}, 'EFO_other')
    entry = entry_for(fixture)
    assert entry['can_run']
    assert entry['resources'] == [{
        'name': 'knowledge', 'release': '1', 'key_param': 'disease', 'required': 'always',
        'built_keys': ['EFO_a', 'EFO_default', 'EFO_z'],
        'build': 'manyruns tools build-resource knowledge --param disease=<disease>',
    }]
    with pytest.raises(action_space.Refused, match='stale offer'):
        action_space.check_choice(before)
    for key in ['EFO_missing', 'EFO_unsigned', 'EFO_other']:
        choice = choice_for(fixture, chosen={'connector': 'conn', 'params': {'disease': key}})
        with pytest.raises(action_space.Refused, match=f'^resource_not_built:knowledge:1:{key}') as exc:
            action_space.check_choice(choice)
        assert exc.value.exit_code == 2
    action_space.check_choice(choice_for(fixture))
    action_space.check_choice(choice_for(fixture, chosen={'connector': 'conn', 'params': {'disease': 'EFO_a'}}))


@pytest.mark.parametrize('key', [None, 'EFO_bad'])
def test_mismatched_resource_is_refused_with_its_code(resource_connector, key):
    fixture, declare = resource_connector
    block = declare(key='disease' if key else None)
    built_resource('knowledge', block, key, mismatch=True)
    entry = entry_for(fixture)
    if key is None:
        assert not entry['can_run']
        assert entry['blocked'] == ['resource_mismatch:knowledge:1']
        assert entry['resources'][0]['status'] == 'mismatch'
        assert entry['resources'][0]['built'] is False
    else:
        assert entry['resources'][0]['built_keys'] == []
    choice = choice_for(fixture, chosen={'connector': 'conn', 'params': {'disease': key} if key else {}})
    code = 'resource_mismatch:knowledge:1' + (f':{key}' if key else '')
    with pytest.raises(action_space.Refused, match=f'^{code}') as exc:
        action_space.check_choice(choice)
    assert exc.value.exit_code == 2


def test_resource_refusal_exits_two_before_running(resource_connector, capsys):
    fixture, declare = resource_connector
    declare(key='disease')
    path = fixture['store'] / 'choice.json'
    path.write_text(json.dumps(choice_for(fixture)))
    with pytest.raises(SystemExit) as exc:
        app.main(['run', '--choice', str(path), '--selection', str(fixture['selection'])])
    assert exc.value.code == 2
    assert 'resource_not_built:knowledge:1:EFO_default' in capsys.readouterr().err


def test_gene_ids_remains_conditional_and_resources_are_sorted(resource_connector):
    fixture, declare = resource_connector
    block = declare('z_reference')
    built_resource('z_reference', block)
    declare('gene_ids')
    entry = entry_for(fixture)
    assert [r['name'] for r in entry['resources']] == ['gene_ids', 'z_reference']
    mapping = entry['resources'][0]
    assert mapping['required'] == 'if_no_gene_ids'
    assert mapping['built'] is False
    assert entry['can_run']
    action_space.check_choice(choice_for(fixture))


def test_connector_without_resources_has_an_empty_list(store_with_parent):
    assert entry_for(store_with_parent)['resources'] == []
