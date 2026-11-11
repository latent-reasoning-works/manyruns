import json
import numpy as np
import anndata as ad
import pytest

from manyruns import toolchain
from manyruns.pipeline import external, steps
from connector_fixtures import planted_h5ad, tiny_gene_ids
from test_tool_table_inputs import setup_tool
from tool_stubs import make_resource


def gene_tool(tmp_path, monkeypatch, mapping=False, source=True):
    resource = {'gene_ids': {'release': '1', 'url': 'https://example.org/genes',
                            'licence': 'CC0-1.0', 'commercial_use': True, 'prepare': 'prepare.py'}} if mapping else {}
    state, ctx = setup_tool(tmp_path, monkeypatch, input={'kind': 'state_frame', 'gene_id': True}, resource=resource)
    if source:
        path = tmp_path / 'source.h5ad'
        planted_h5ad(path)
        ctx['source_path'] = path
    return state, ctx


def exported(tmp_path):
    return ad.read_h5ad(tmp_path / 'out/tool-conn/input.h5ad')


def test_an_ensembl_var_column_is_copied_to_gene_id_without_version(tmp_path, monkeypatch):
    state, ctx = gene_tool(tmp_path, monkeypatch)
    state['cols'] = np.array([2, 0])
    external.run_external_step('conn', {}, state, {}, ctx)
    assert list(exported(tmp_path).var['gene_id']) == ['ENSG00000000003', 'ENSG00000000001']


def test_a_frame_without_ids_maps_names_through_the_gene_ids_resource(tmp_path, monkeypatch):
    state, ctx = gene_tool(tmp_path, monkeypatch, mapping=True, source=False)
    tiny_gene_ids(tmp_path, monkeypatch)
    g = {}
    external.run_external_step('conn', {}, state, g, ctx)
    assert list(exported(tmp_path).var['gene_id']) == [f'ENSG{i + 1:011d}' for i in range(3)]
    assert g['conn.resource.gene_ids.sha256']
    assert 'gene_ids' not in json.loads((tmp_path / 'out/tool-conn/request.json').read_text()).get('resources', {})


def test_unmapped_genes_get_an_empty_id_and_are_counted(tmp_path, monkeypatch):
    state, ctx = gene_tool(tmp_path, monkeypatch, mapping=True, source=False)
    make_resource(tmp_path, monkeypatch, 'conn', 'gene_ids', {'symbol': ['g0', 'g1'], 'gene_id': ['ENSG00000000001', 'ENSG00000000002']})
    g = {}
    external.run_external_step('conn', {}, state, g, ctx)
    assert g['conn.genes_unmapped'] == 1 and list(exported(tmp_path).var['gene_id'])[-1] == ''


def test_fewer_than_half_mapped_skips_the_step_with_the_count(tmp_path, monkeypatch):
    state, ctx = gene_tool(tmp_path, monkeypatch, mapping=True, source=False)
    make_resource(tmp_path, monkeypatch, 'conn', 'gene_ids', {'symbol': ['g0'], 'gene_id': ['ENSG00000000001']})
    with pytest.raises(steps._StepSkipped, match='1 of 3 genes'):
        external.run_external_step('conn', {}, state, {}, ctx)
    assert not (tmp_path / 'out/tool-conn/ran.marker').exists()


def test_a_missing_gene_ids_resource_is_unsupported_with_the_build_command(tmp_path, monkeypatch):
    state, ctx = gene_tool(tmp_path, monkeypatch, mapping=True, source=False)
    monkeypatch.setenv('MANYRUNS_RESOURCE_HOME', str(tmp_path / 'missing'))
    with pytest.raises(steps._StepUnsupported, match='manyruns tools build-resource gene_ids'):
        external.run_external_step('conn', {}, state, {}, ctx)


def test_absent_column_and_resource_are_named(tmp_path, monkeypatch):
    state, ctx = gene_tool(tmp_path, monkeypatch, source=False)
    with pytest.raises(steps._StepSkipped, match='column.*resource'):
        external.run_external_step('conn', {}, state, {}, ctx)


def test_a_tool_that_does_not_ask_sees_an_unchanged_frame(tmp_path, monkeypatch):
    state, ctx = setup_tool(tmp_path, monkeypatch, tables={})
    external.run_external_step('conn', {}, state, {}, ctx)
    assert 'gene_id' not in exported(tmp_path).var


@pytest.mark.parametrize('ids, refused', [(['ENSG99999999999'] * 3, True), ([''] * 3, False)])
def test_result_gene_ids_are_checked_except_empty_ids(tmp_path, monkeypatch, ids, refused):
    state, ctx = gene_tool(tmp_path, monkeypatch)
    if refused:
        with pytest.raises(steps._StepSkipped, match='gene_id'):
            external.run_external_step('conn', {'ids': ids}, state, {}, ctx)
        assert not state['tables']
    else:
        external.run_external_step('conn', {'ids': ids}, state, {}, ctx)
        assert state['tables']


def test_a_kind_none_result_gene_id_must_come_from_its_input_tables(tmp_path, monkeypatch):
    state, ctx = setup_tool(tmp_path, monkeypatch, kind='none', tables={'genes': ['gene_id']})
    path = tmp_path / 'genes.tsv'
    path.write_text('gene_id\nENSG00000000001\n')
    ctx['input_tables'] = {'genes': str(path)}
    with pytest.raises(steps._StepSkipped, match='gene_id'):
        external.run_external_step('conn', {'ids': ['ENSG99999999999']}, state, {}, ctx)
    external.run_external_step('conn', {'ids': ['ENSG00000000001']}, state, {}, ctx)
    assert toolchain.load_tool('conn')['input']['kind'] == 'none'


def test_a_keyed_gene_ids_resource_is_unsupported_with_the_reason(tmp_path, monkeypatch):
    state, ctx = gene_tool(tmp_path, monkeypatch, mapping=True, source=False)
    import yaml
    path = toolchain.tool_dir() / 'conn.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['resource']['gene_ids']['key'] = 'release_key'
    cfg['params'] = {'release_key': 'human'}
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(steps._StepUnsupported, match='key'):
        external.run_external_step('conn', {}, state, {}, ctx)
