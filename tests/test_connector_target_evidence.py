import json
import subprocess
import sys
from importlib.resources import files

import pandas as pd
import pytest

from manyruns import toolchain
from manyruns.pipeline import external, steps
from connector_fixtures import bundled_connector, connector_table, tiny_opentargets


def evidence_input(tmp_path, monkeypatch, build=True):
    _, state, ctx = bundled_connector(tmp_path, monkeypatch, 'target_evidence')
    genes = pd.DataFrame({'gene': ['g0', 'g1', 'g2'],
                          'gene_id': [f'ENSG{i + 1:011d}' for i in range(3)], 'score': [2., -10., 5.]})
    path = tmp_path / 'genes.tsv'
    genes.to_csv(path, sep='\t', index=False)
    ctx['input_tables'] = {'genes': str(path)}
    if build:
        tiny_opentargets(tmp_path, monkeypatch)
    return state, ctx


def test_the_bundled_manifest_passes_check_tool():
    cfg = toolchain.load_tool('target_evidence')
    assert not toolchain.check_tool(cfg, 'target_evidence')
    assert cfg['licence'] == 'BSD-3-Clause'
    assert cfg['network'] == 'none'


def test_evidence_is_joined_on_gene_id_for_the_top_genes(tmp_path, monkeypatch):
    state, ctx = evidence_input(tmp_path, monkeypatch)
    g = {}
    external.run_external_step('target_evidence', {'disease': 'EFO_fixture', 'top_n': 2}, state, g, ctx)
    table = connector_table(tmp_path, 'target_evidence', 'gene_evidence')
    assert list(table.gene) == ['g1', 'g2']
    assert set(table.disease_id) == {'EFO_fixture'}
    assert g['target_evidence.genes_queried'] == g['target_evidence.genes_with_evidence'] == 2


def test_a_disease_without_a_built_resource_is_unsupported_with_the_command(tmp_path, monkeypatch):
    state, ctx = evidence_input(tmp_path, monkeypatch, build=False)
    with pytest.raises(steps._StepUnsupported, match='manyruns tools build-resource opentargets --param disease=EFO_fixture'):
        external.run_external_step('target_evidence', {'disease': 'EFO_fixture'}, state, {}, ctx)


@pytest.mark.parametrize('disease', ['', '../outside', 'a/b'])
def test_invalid_disease_is_refused(tmp_path, monkeypatch, disease):
    state, ctx = evidence_input(tmp_path, monkeypatch, build=False)
    with pytest.raises(steps._StepSkipped, match='key'):
        external.run_external_step('target_evidence', {'disease': disease}, state, {}, ctx)


def test_an_evidence_gene_id_not_in_the_input_genes_is_refused(tmp_path, monkeypatch):
    state, ctx = evidence_input(tmp_path, monkeypatch)
    # A faulty adapter must not publish a foreign gene, even with a valid digest.
    script = tmp_path / 'faulty.py'
    script.write_text('''import hashlib, json, sys
from pathlib import Path
req = json.loads(Path(sys.argv[1]).read_text())
p = Path(req['workdir']) / 'gene_evidence.tsv'
p.write_text('gene_id\\tsymbol\\tdisease_id\\tassoc_score\\tgene\\tscore\\nENSG99999999999\\tbad\\tEFO_fixture\\t1\\tg0\\t1\\n')
Path(req['result']).write_text(json.dumps({'schema_version': 1, 'ok': True, 'n_cells': req['n_cells'],
 'tables': {'gene_evidence': {'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}}}))
''')
    monkeypatch.setattr(external, '_adapter_path', lambda *a, **kw: script)
    with pytest.raises(steps._StepSkipped, match='gene_id'):
        external.run_external_step('target_evidence', {'disease': 'EFO_fixture'}, state, {}, ctx)
    assert not state['tables']


def test_opentargets_prepare_filters_to_one_disease(tmp_path):
    pytest.importorskip('pyarrow')
    source = tmp_path / 'associations.parquet'
    pd.DataFrame({'targetId': ['ENSG00000000001', 'ENSG00000000002'],
                  'diseaseId': ['EFO_fixture', 'EFO_other'], 'score': [0.8, 0.2]}).to_parquet(source)
    out = tmp_path / 'prepared.tsv'
    script = files('manyruns') / 'adapters/opentargets_prepare.py'
    subprocess.run([sys.executable, str(script), '--url', source.as_uri(), '--release', '25.03',
                    '--key', 'EFO_fixture', '--out', str(out)], check=True)
    result = pd.read_csv(out, sep='\t', keep_default_na=False)
    assert list(result.gene_id) == ['ENSG00000000001']
    assert list(result.columns) == ['gene_id', 'symbol', 'disease_id', 'assoc_score']


def test_opentargets_prepare_filters_a_tiny_local_fixture(tmp_path):
    from manyruns.adapters.opentargets_prepare import prepare_frame
    path = tmp_path / 'associations.json'
    path.write_text(json.dumps([{'targetId': 'ENSG00000000001', 'diseaseId': 'EFO_fixture', 'score': 0.8},
                                {'targetId': 'ENSG00000000002', 'diseaseId': 'EFO_other', 'score': 0.2}]))
    result = prepare_frame(pd.read_json(path), 'EFO_fixture')
    assert result.to_dict('records') == [{'gene_id': 'ENSG00000000001', 'symbol': '',
                                         'disease_id': 'EFO_fixture', 'assoc_score': 0.8}]
