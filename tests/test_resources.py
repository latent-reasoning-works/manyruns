import hashlib
import json
from pathlib import Path

import pytest

from manyruns import app, resources
from manyruns.pipeline import external, steps
from test_tool_table_inputs import setup_tool


def resource_tool(tmp_path, monkeypatch):
    source = tmp_path / 'source.tsv'
    source.write_text('gene_id\tsymbol\nENSG00000000001\tg0\n')
    block = {'release': '1', 'url': source.as_uri(), 'licence': 'CC0-1.0',
             'commercial_use': True, 'prepare': 'prepare.py', 'key': 'disease'}
    state, ctx = setup_tool(tmp_path, monkeypatch, tables={},
                            resource={'knowledge': block}, params={'disease': 'EFO_1'})
    (tmp_path / 'tools/prepare.py').write_text('''import argparse, shutil
from urllib.parse import urlparse
p = argparse.ArgumentParser()
for k in ('url', 'release', 'out', 'key'): p.add_argument('--' + k)
a = p.parse_args()
shutil.copyfile(urlparse(a.url).path, a.out)
''')
    monkeypatch.setenv('MANYRUNS_RESOURCE_HOME', str(tmp_path / 'resources'))
    return block, state, ctx


def test_build_resource_writes_the_tsv_and_its_sha256(tmp_path, monkeypatch):
    block, _, _ = resource_tool(tmp_path, monkeypatch)
    built = resources.build('knowledge', {'disease': 'EFO_1'})
    path = resources.path_for('knowledge', block, 'EFO_1')
    assert Path(built['path']) == path
    assert built['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.with_suffix('.tsv.sha256').read_text().strip() == built['sha256']


def test_build_resource_refuses_when_the_tool_environment_is_not_built(tmp_path, monkeypatch):
    resource_tool(tmp_path, monkeypatch)
    monkeypatch.delenv('MANYRUNS_TOOL_CONN_PYTHON')
    with pytest.raises(ValueError, match='manyruns tools install conn'):
        resources.build('knowledge', {'disease': 'EFO_1'})


@pytest.mark.parametrize('mismatch', [False, True])
def test_missing_or_mismatched_resource_is_unsupported(tmp_path, monkeypatch, mismatch):
    block, state, ctx = resource_tool(tmp_path, monkeypatch)
    if mismatch:
        resources.build('knowledge', {'disease': 'EFO_1'})
        resources.path_for('knowledge', block, 'EFO_1').write_text('changed')
    with pytest.raises(steps._StepUnsupported, match='manyruns tools build-resource knowledge --param disease=EFO_1'):
        external.run_external_step('conn', {}, state, {}, ctx)
    assert not (tmp_path / 'out/tool-conn/ran.marker').exists()


@pytest.mark.parametrize('key', ['../x', 'a/b', ''])
def test_a_resource_key_that_is_not_file_safe_is_refused(tmp_path, monkeypatch, key):
    _, state, ctx = resource_tool(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='file-safe'):
        resources.build('knowledge', {'disease': key})
    with pytest.raises(steps._StepSkipped, match='file-safe'):
        external.run_external_step('conn', {'disease': key}, state, {}, ctx)
    assert app.main(['tools', 'build-resource', 'knowledge', '--param', 'disease=' + key]) == 1
    assert not (tmp_path / 'x.tsv').exists()


def test_the_request_gives_the_adapter_the_path_and_the_record_pins_it(tmp_path, monkeypatch):
    _, state, ctx = resource_tool(tmp_path, monkeypatch)
    built = resources.build('knowledge', {'disease': 'EFO_1'})
    g = {}
    external.run_external_step('conn', {}, state, g, ctx)
    request = json.loads((tmp_path / 'out/tool-conn/request.json').read_text())
    assert request['resources']['knowledge'] == str(built['path'])
    assert g['conn.resource.knowledge.sha256'] == built['sha256']
    assert g['conn.resource.knowledge.release'] == '1'


def test_build_resource_cli_parses_name_and_params():
    args = app._build_parser().parse_args(['tools', 'build-resource', 'knowledge', '--param', 'disease=EFO_1'])
    assert args.name == 'knowledge' and args.param == ['disease=EFO_1']


def test_build_resource_cli_exit_codes(tmp_path, monkeypatch):
    resource_tool(tmp_path, monkeypatch)
    assert app.main(['tools', 'build-resource']) == 2
    assert app.main(['tools', 'build-resource', 'knowledge', '--param', 'broken']) == 2
    assert app.main(['tools', 'build-resource', 'absent']) == 1
    assert app.main(['tools', 'build-resource', 'knowledge', '--param', 'disease=EFO_1']) == 0
