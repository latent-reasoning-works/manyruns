import numpy as np
import pytest
from manyruns import derive
from manyruns.pipeline import runner


def test_selection_uses_saved_parent_space(tmp_path, monkeypatch):
    from manyruns.pipeline import features
    import features_stub
    monkeypatch.setattr(features, "_helper", lambda: features_stub.stub_rows)
    x = np.random.default_rng(2).normal(size=(60,5))
    parent = runner.run_inproc(x, {'name':'plain', 'steps':[{'name':'dataset_features','group':'analysis'}]},
                               tmp_path, sample_ids=np.array([f'cell{i}' for i in range(60)]))
    session = derive.open_derived(parent, 'shape_profile', {'extended':False}, {}, tmp_path)
    np.testing.assert_array_equal(session.state['X'], x)
    assert session.recipe['name'] == 'derive:shape_profile'
    assert session.measure is False
    assert session.ctx['input_space_manifest']['preprocessing']['branch'] == 'none'


def test_parent_without_saved_space_is_refused(tmp_path):
    with pytest.raises(ValueError, match='parent_input_space_unavailable'):
        derive.open_derived({'run_id':'missing','ok':True,'steps':[]}, 'shape_profile', {}, {}, tmp_path)


def test_broken_derived_cycle_is_refused(tmp_path, monkeypatch):
    parent = {'run_id':'cycle','recipe':'derive:shape_profile','ok':True,
              'parent':{'run_id':'other'},'steps':[]}
    other = {**parent,'run_id':'other','parent':{'run_id':'cycle'}}
    monkeypatch.setattr(derive,'_manifest', lambda *args: {})
    monkeypatch.setattr(derive,'parent_row',lambda run,*args: parent if run == 'cycle' else other)
    from manyruns.profile import resolve_parent_space
    with pytest.raises(ValueError,match='parent_input_space_unavailable'):
        resolve_parent_space(parent,tmp_path)
