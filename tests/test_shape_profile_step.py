from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import shape_profile_stub
from manyruns.pipeline import shape_profile, runner


def test_step_publishes_exact_closed_table(tmp_path, monkeypatch):
    monkeypatch.setattr(shape_profile, '_helper', lambda: shape_profile_stub.stub_rows)
    out = runner.run_inproc(np.ones((600,5)), {'steps':[{'name':'shape_profile','group':'analysis'}]}, tmp_path)
    rec = out['steps'][0]
    assert rec['outcome'] == 'ok', rec['detail']
    table = pd.read_csv(rec['artifacts']['shape_profile'], sep='\t', keep_default_na=False)
    assert tuple(table.columns) == shape_profile.COLUMNS
    assert len(table) == 15
    assert all(len(s) == 64 for s in table['entity_id'])
    assert '0.12345678901234568' in Path(rec['artifacts']['shape_profile']).read_text()


@pytest.mark.parametrize('change', [
    lambda rows: rows.append(rows[0]),
    lambda rows: rows[0].update(extra=1),
    lambda rows: rows[0].update(value=float('nan')),
    lambda rows: rows[0].update(status='censored'),
    lambda rows: rows[0].update(n_cells_used=True),
    lambda rows: rows[0].update(stat_name='unknown'),
])
def test_bad_rows_refused_before_publication(tmp_path, monkeypatch, change):
    rows = shape_profile_stub.stub_rows(None)
    change(rows)
    monkeypatch.setattr(shape_profile, '_helper', lambda: lambda *a, **kw: rows)
    state, g = {'X':np.ones((600,5))}, {}
    with pytest.raises(ValueError):
        shape_profile.step(state,g,{}, {'out_dir':tmp_path,'step_index':0})
    assert g == {} and 'tables' not in state
    assert tmp_path.is_dir()
    assert not list(tmp_path.rglob('*.tsv'))


def test_numpy_string_is_not_a_native_helper_value():
    rows = shape_profile_stub.stub_rows(None)
    rows[0]['entity_kind'] = np.str_('dataset')
    with pytest.raises(ValueError):
        shape_profile._check_rows(rows, selection=False)
