"""Abandonment finalizes the record without starting final geometry work."""
import json
import numpy as np
import pytest
from manyruns import artifacts, store
from manyruns.pipeline import runner
from manyruns.session import Session


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Session('stop', engine='_inproc', array=np.ones((8, 3)), out_dir=tmp_path,
                recipe={'name': 'repeated', 'steps': [
                    {'name': 'embed', 'group': 'latent'},
                    {'name': 'read', 'group': 'analysis'},
                    {'name': 'read', 'group': 'analysis'}]})
    monkeypatch.setattr(runner, '_attach_geometry', lambda *a, **k: {})
    s.dispatch = {'latent': lambda n, p, state, g, ctx: state.update(emb=state['X'][:, :2].copy()),
                  'analysis': lambda n, p, state, g, ctx: g.update(already_measured=7)}
    return s


def forbid_suite(*a, **kw):
    pytest.fail('abandonment invoked final geometry')


def test_abandoned_close_skips_suite_and_persists_not_run(session, monkeypatch, tmp_path):
    session.apply(session.recipe['steps'][0])
    session.g['already_measured'] = 7
    monkeypatch.setattr(runner._suite, 'measure', forbid_suite)
    remaining = [{'index': 1, 'name': 'read', 'reason': 'run abandoned'},
                 {'index': 2, 'name': 'read', 'reason': 'run abandoned'}]
    result = session.close(abandoned=True, not_run=remaining)
    assert result['cancelled'] is True and result['complete'] is False
    assert result['g_vector']['already_measured'] == 7
    assert result['not_run'] == remaining
    assert len(result['steps']) == 1
    assert session.discarded == []
    assert any('final geometry was not measured' in note for note in result['caveats'])
    assert not any('discarded' in note for note in result['caveats'])
    index = store.append(result, out_dir=tmp_path)
    row = json.loads(index.read_text())
    assert row['complete'] is False and len(row['steps']) == 1
    for i in (1, 2):
        assert any(f'step {i} (read): not run' in note and 'run abandoned' in note for note in row['caveats'])
    assert artifacts.complete(artifacts.root(tmp_path, session.run_id))


def test_normal_close_still_measures_when_step_metrics_are_empty(session, monkeypatch):
    calls = []
    monkeypatch.setattr(runner._suite, 'measure', lambda *a, **k: calls.append(1) or {'measured': 12})
    session.ctx['metrics'] = []
    session.apply(session.recipe['steps'][0])
    result = session.close()
    assert calls == [1]
    assert result['g_vector']['measured'] == 12
    assert result['cancelled'] is False


def test_all_not_run_creates_no_success_records_or_marker(session, monkeypatch, tmp_path):
    monkeypatch.setattr(runner._suite, 'measure', forbid_suite)
    result = session.close(abandoned=True, not_run=[{'index': i, 'name': step['name'], 'reason': 'closed gate'}
                                                  for i, step in enumerate(session.recipe['steps'])])
    assert result['steps'] == []
    assert result['cancelled'] is True and result['complete'] is False
    assert not artifacts.complete(artifacts.root(tmp_path, session.run_id))


def test_abandonment_after_last_untunable_step_is_incomplete(session, monkeypatch):
    session.run_recipe()
    monkeypatch.setattr(runner._suite, 'measure', forbid_suite)
    result = session.close(abandoned=True)
    assert result['cancelled'] is True and result['complete'] is False
    assert result['not_run'] == []
    assert len(result['steps']) == 3


def test_discarded_attempt_is_distinct_from_unexecuted_occurrence(session, monkeypatch):
    session.apply(session.recipe['steps'][0])
    session.discarded.append('embed')
    monkeypatch.setattr(runner._suite, 'measure', forbid_suite)
    result = session.results(abandoned=True, not_run=[{'index': 1, 'name': 'read', 'reason': 'closed gate'}])
    assert session.discarded == ['embed']
    assert result['steps'][0]['outcome'] == 'ok'
    assert any('embed: discarded' in note for note in result['caveats'])
    assert any('step 1 (read): not run' in note for note in result['caveats'])
