import pytest

from manyruns import toolchain


def problems(**changes):
    return toolchain.check_tool({
        'name': 'conn', 'version': '1', 'licence': 'MIT', 'commercial_use': True,
        'requires': ['conn==1'], 'entrypoint': 'adapter.py', 'selftest': 'pass',
        'input': {'kind': 'state_frame'}, 'produces': [],
        'output': {'schema': 'result-v1', 'tables': {'genes': ['gene']}}, **changes,
    }, 'conn')


def test_input_tables_map_file_safe_names_to_column_lists():
    assert not problems(input={'kind': 'state_frame', 'tables': {'selection': ['cell', 'group']}})
    for tables in ({'../x': ['a']}, {'genes': 'gene'}, {'': []}):
        assert 'input.tables' in ' '.join(problems(input={'kind': 'state_frame', 'tables': tables}))
    assert toolchain.input_tables({}) == {}


def test_input_kind_none_is_legal_and_takes_no_obs_gene_id_or_layers():
    assert not problems(input={'kind': 'none', 'tables': {'genes': ['gene']}})
    for extra in ({'obs': ['sample']}, {'gene_id': True}, {'layers': ['spliced']},
                  {'tables': {'selection': ['cell', 'group']}}):
        assert problems(input={'kind': 'none', **extra})


def test_input_gene_id_must_be_true_on_a_state_frame():
    assert not problems(input={'kind': 'state_frame', 'gene_id': True})
    assert problems(input={'kind': 'state_matrix', 'gene_id': True})
    assert problems(input={'kind': 'state_frame', 'gene_id': 'yes'})


def test_network_none_is_the_only_legal_value_and_absence_means_none():
    assert not problems(network='none') and not problems()
    assert problems(network=['gene_id'])


def test_a_resource_block_requires_its_fields_and_a_declared_key_param():
    block = {'release': '25.03', 'url': 'https://example.org/data', 'licence': 'CC0-1.0',
             'commercial_use': True, 'prepare': 'prepare.py', 'key': 'disease'}
    assert not problems(resource={'knowledge': block}, params={'disease': 'EFO_1'})
    assert problems(resource={'knowledge': block})
    for changes in ({'release': '../x'}, {'colour': 1}, {'licence': 'CC-BY-NC-4.0'},
                    {'key': '../x'}, {'prepare': ''}):
        assert problems(resource={'knowledge': {**block, **changes}}, params={'disease': 'EFO_1'})
    assert problems(resource={'knowledge': {'release': '1'}})


@pytest.mark.parametrize('name', ['', '.', '..', '../x', 'a/b', 'a\\b'])
def test_resource_names_are_file_safe(name):
    assert problems(resource={name: {}})


def test_a_connector_is_a_table_only_manifest():
    assert toolchain.is_connector({'output': {'tables': {'genes': ['gene']}}, 'produces': []})
    assert not toolchain.is_connector({'output': {'obsm': ['velocity']}, 'produces': ['velocity']})


def test_the_bundled_pyrovelocity_manifest_still_loads():
    assert toolchain.tool_dir().is_dir()
    assert not toolchain.check_tool(toolchain.load_tool('pyrovelocity'), 'pyrovelocity')


def test_resource_and_network_rules_also_apply_to_precomputed_tools():
    cfg = {'name': 'conn', 'version': '1', 'licence': 'MIT', 'commercial_use': True,
           'mode': 'precomputed', 'produces': [],
           'output': {'schema': 'result-v1', 'tables': {'genes': ['gene']}, 'root': '.'}}
    assert toolchain.check_tool({**cfg, 'network': 'internet'}, 'conn')
    assert toolchain.check_tool({**cfg, 'resource': {'unsafe': {'release': '1'}}}, 'conn')


def test_a_resource_licence_of_the_wrong_type_is_a_problem():
    assert problems(resource={'knowledge': {'release': '1', 'url': 'https://example.org',
                                           'licence': {}, 'commercial_use': True, 'prepare': 'prepare.py'}})
