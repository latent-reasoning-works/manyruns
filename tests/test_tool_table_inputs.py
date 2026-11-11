import json

import numpy as np
import pytest

from manyruns.pipeline import external, runner, steps
from tool_stubs import INPUT_TABLE_ADAPTER, make_tool


def setup_tool(tmp_path, monkeypatch, kind='state_frame', tables=None, **extra):
    make_tool(tmp_path, monkeypatch, 'conn', {
        'input': {'kind': kind, 'tables': {'selection': ['cell', 'group']} if tables is None else tables},
        'produces': [], 'output': {'schema': 'result-v1', 'tables': {'genes': ['gene']}},
        **extra,
    }, INPUT_TABLE_ADAPTER)
    x = np.arange(12).reshape(4, 3)
    state = runner._new_state(X=x, counts=x, genes=np.array(['g0', 'g1', 'g2']))
    ctx = {'out_dir': tmp_path / 'out', 'run_id': 'r1', 'step_index': 0,
           'sample_ids': ['c0', 'c1', 'c2', 'c3'], 'caveats': []}
    return state, ctx


def bind(tmp_path, ctx, text):
    path = tmp_path / 'selection.tsv'
    path.write_text(text)
    ctx['input_tables'] = {'selection': str(path)}


def test_a_bound_table_reaches_the_adapter_by_name(tmp_path, monkeypatch):
    state, ctx = setup_tool(tmp_path, monkeypatch)
    bind(tmp_path, ctx, 'cell\tgroup\nc1\ta\nc3\tb\n')
    g = {}
    external.run_external_step('conn', {}, state, g, ctx)
    request = json.loads((tmp_path / 'out/tool-conn/request.json').read_text())
    assert request['tables']['selection'].endswith('tables/selection.tsv')
    assert g['conn.selected'] == 2


@pytest.mark.parametrize('text, fault', [
    (None, 'selection'), ('cell\nc0\n', 'group'), ('cell\tgroup\nzz\ta\n', 'zz'),
    ('cell\tgroup\nc0\ta\nc0\tb\n', 'duplicate'), ('cell\tgroup\n', 'empty'),
])
def test_invalid_selection_is_refused_before_adapter(tmp_path, monkeypatch, text, fault):
    state, ctx = setup_tool(tmp_path, monkeypatch)
    if text is not None:
        bind(tmp_path, ctx, text)
    with pytest.raises(steps._StepSkipped, match=fault):
        external.run_external_step('conn', {}, state, {}, ctx)
    assert not (tmp_path / 'out/tool-conn/ran.marker').exists()


def test_a_selection_on_a_run_without_cell_ids_is_refused(tmp_path, monkeypatch):
    state, ctx = setup_tool(tmp_path, monkeypatch)
    del ctx['sample_ids']
    bind(tmp_path, ctx, 'cell\tgroup\n0\ta\n')
    with pytest.raises(steps._StepSkipped, match='cell ids'):
        external.run_external_step('conn', {}, state, {}, ctx)
    assert not (tmp_path / 'out/tool-conn/ran.marker').exists()


def test_an_input_kind_none_tool_gets_no_input_file_and_the_run_s_cell_count(tmp_path, monkeypatch):
    state, ctx = setup_tool(tmp_path, monkeypatch, kind='none', tables={})
    external.run_external_step('conn', {}, state, {}, ctx)
    folder = tmp_path / 'out/tool-conn'
    request = json.loads((folder / 'request.json').read_text())
    assert request['input'] is None and request['n_cells'] == 4
    assert not (folder / 'input.h5ad').exists() and not (folder / 'input.npy').exists()


def test_a_tool_without_input_tables_writes_the_request_it_wrote_before(tmp_path, monkeypatch):
    state, ctx = setup_tool(tmp_path, monkeypatch, tables={})
    external.run_external_step('conn', {}, state, {}, ctx)
    assert 'tables' not in json.loads((tmp_path / 'out/tool-conn/request.json').read_text())
