"""Spawn transport preserves the session's record and authoritative parent state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from manyruns import artifacts, store
from manyruns.pipeline import runner
from manyruns.session import Session


def embed(name, params, state, g, ctx):
    state['emb'] = state['X'][:, :2].copy()
    state.setdefault('extras', {})['worker_pid'] = os.getpid()
    g['pca.score'] = float(state['emb'].sum())
    ctx['caveats'].append('fixture: measured on the selected rows')


def make_session(path):
    session = Session('transport', engine='manylatents', out_dir=path,
                      array=np.arange(36, dtype=float).reshape(12, 3),
                      sample_ids=np.array([f'cell-{i}' for i in range(12)]),
                      recipe={'name': 'embed', 'steps': [
                          {'name': 'pca', 'group': 'latent'}]})
    session.dispatch = {'latent': embed}
    session.carry['declared'] = []
    return session


def retain_model(name, params, state, g, ctx):
    # MIOFlow retains a function-local datamodule. Standard pickle cannot serialize
    # that object graph; it must remain in the worker, including across tune rollback.
    class Model:
        trajectories = np.arange(24.).reshape(3, 4, 2) * params['scale']

    state['model'] = Model()


def test_retained_model_survives_another_attempt_and_parent_rollback(tmp_path):
    session = make_session(tmp_path)
    session.dispatch.update(lightning=retain_model, probe=runner._run_probe_step)
    session.compute = runner.ComputeProcess()
    try:
        first = session.apply({'name': 'flow', 'group': 'lightning', 'params': {'scale': 1}})
        assert first['outcome'] == 'ok', first
        baseline = dict(session.state)
        session.apply({'name': 'flow', 'group': 'lightning', 'params': {'scale': 2}})
        session.state.clear()
        session.state.update(baseline)
        probe = session.apply({'name': 'sample_trajectories', 'group': 'probe'})
        assert probe['outcome'] == 'ok', probe
        assert session.g['sample_trajectories.median_displacement'] == np.sqrt(512)
    finally:
        session.compute.close()


def test_real_mioflow_retains_its_model_for_a_later_probe(tmp_path):
    pytest.importorskip('manylatents')
    array = np.random.default_rng(0).normal(size=(30, 3)).astype(np.float32)
    session = Session('flow', engine='manylatents', out_dir=tmp_path,
                      array=array, embedding=array.copy(), labels=np.repeat([0., .5, 1.], 10),
                      label_kind='time', metrics=[])
    session.carry['declared'] = []
    session.compute = runner.ComputeProcess()
    try:
        fitted = session.apply({'name': 'mioflow', 'group': 'lightning',
                                'params': {'n_bins': 7, 'n_trajectories': 4, 'hidden_dim': 8}})
        assert fitted['outcome'] == 'ok', fitted.get('detail')
        probe = session.apply({'name': 'sample_trajectories', 'group': 'probe'})
        assert probe['outcome'] == 'ok', probe.get('detail')
        assert session.g['sample_trajectories.n_steps'] == 7
        assert session.g['sample_trajectories.n_trajectories'] == 4
    finally:
        session.compute.close()


def child_exit():
    os._exit(9)


def test_child_failure_is_reported_and_reaped():
    compute = runner.ComputeProcess()
    try:
        with pytest.raises(RuntimeError, match='exited|disconnected'):
            compute.call(child_exit)
    finally:
        compute.close()
    assert compute.process is None


@pytest.mark.parametrize('source', ['cancel', 'callback'])
@pytest.mark.parametrize('loss', ['dead', 'dead-reap', 'send', 'poll', 'recv'])
def test_worker_loss_during_cancellation_is_not_a_failed_attempt(monkeypatch, source, loss):
    cancelled = threading.Event()
    compute = runner.ComputeProcess(cancelled=cancelled.is_set)
    process = Mock(pid=123, exitcode=-15)
    process.is_alive.return_value = False
    compute.process = process
    compute.connection = Mock()
    monkeypatch.setattr(compute, '_signal', Mock())
    cancel = compute.cancel if source == 'cancel' else cancelled.set

    def disconnected(*args):
        if loss == 'dead':
            cancel()  # The UI cancels after the loop check, inside poll's wait.
        if loss in ('dead', 'dead-reap'):
            return False
        error = {'send': BrokenPipeError, 'poll': OSError, 'recv': EOFError}[loss]
        raise error('fixture: worker disconnected')

    if loss != 'dead':
        # Pipe errors already checked cancellation, but can then lose that race
        # during the diagnostic join. Every disconnect path must check again.
        process.join.side_effect = lambda timeout: cancel()
    compute.connection.poll.return_value = True
    operation = {'send': 'send_bytes', 'recv': 'recv_bytes'}.get(loss, 'poll')
    getattr(compute.connection, operation).side_effect = disconnected

    with pytest.raises(runner.ComputeCancelled, match='run abandoned'):
        compute.call(child_exit)


class SlowReapProcess:
    """SIGKILL has been sent, but waitpid cannot report the exit within the bound."""
    pid = 123

    def __init__(self):
        self.alive = True
        self.closed = threading.Event()
        self.reaping = threading.Event()
        self.release = threading.Event()
        self.joins = []
        self.reaper = None
        self.close_count = 0

    def is_alive(self):
        if self.closed.is_set():
            raise ValueError('process object is closed')
        return self.alive

    def join(self, timeout=None):
        self.joins.append(timeout)
        if timeout is None:
            self.reaper = threading.current_thread()
            self.reaping.set()
            assert self.release.wait(5), 'test did not release the reap'
            self.alive = False

    def close(self):
        if self.alive:
            raise ValueError('Cannot close a process while it is still running')
        self.close_count += 1
        self.closed.set()


@pytest.mark.skipif(os.name != 'posix', reason='checks process-group shutdown signals')
@pytest.mark.parametrize('record_session', [False, True])
def test_slow_reap_preserves_shutdown_and_run_record(monkeypatch, tmp_path, record_session):
    import multiprocessing.process
    import signal

    compute = runner.ComputeProcess()
    process = SlowReapProcess()
    compute.process = process
    connection = compute.connection = Mock()
    signals = []
    monkeypatch.setattr(os, 'killpg', lambda pid, sig: signals.append(sig))
    # A daemon reaper alone is insufficient: multiprocessing's exit handler
    # would still synchronously join a handed-off child in this registry.
    multiprocessing.process._children.add(process)
    try:
        if record_session:
            session = make_session(tmp_path / 'outputs' / 'transport')
            session.run_recipe()
            assert session.carry['written']
            session.compute = compute
            result = session.close(abandoned=True)
            row = json.loads(store.append(result, out_dir=tmp_path / 'outputs').read_text())
            assert row['run_id'] == session.run_id
            assert row['steps'][0]['outcome'] == 'ok'
            assert artifacts.complete(artifacts.root(session.out_dir, session.run_id))
            saved = artifacts.load_array(row['steps'][0]['artifacts']['emb'])
            np.testing.assert_array_equal(saved, session.state['emb'])
        else:
            compute.close()
        assert compute.process is None
        assert compute.connection is None
        connection.close.assert_called_once()
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        assert process.reaping.wait(5), 'slow child was not handed to a reaper'
        assert process.reaper.daemon, 'reaping must not hold interpreter exit open'
        assert process.joins == [.2, .2, .2, None]
        assert process not in multiprocessing.process._children
        assert not process.closed.is_set(), 'a live handle must stay open'
        compute.cancel()  # UI cancellation after handoff must not touch its handle.
        compute.close()  # Repeated shutdown must not take ownership back.
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        process.release.set()
        process.reaper.join(5)
        assert not process.reaper.is_alive()
        assert process.closed.is_set()
        assert process.close_count == 1
        compute.cancel()  # Also safe after the reaper closes the handle.
        assert signals == [signal.SIGTERM, signal.SIGKILL]
    finally:
        process.release.set()
        if process.reaper is not None:
            process.reaper.join(5)
        multiprocessing.process._children.discard(process)


def test_finalization_oserror_still_records_retained_work(monkeypatch, tmp_path):
    session = make_session(tmp_path / 'run')
    session.run_recipe()
    session.compute = runner.ComputeProcess()

    def unavailable(*args, **kwargs):
        raise PermissionError('fixture: cannot start compute process')

    def no_parent_geometry(*args, **kwargs):
        pytest.fail('failed offload must not retry expensive geometry in the UI process')

    monkeypatch.setattr(session.compute, 'call', unavailable)
    monkeypatch.setattr(runner._suite, 'measure', no_parent_geometry)
    result = session.close()
    row = json.loads(store.append(result, out_dir=tmp_path).read_text())
    assert row['run_id'] == session.run_id
    assert not row['complete'] and not row['ok']
    assert not result['cancelled']
    assert row['steps'][0]['outcome'] == 'ok'
    assert row['g_vector']['pca.score'] == session.g['pca.score']
    assert any('final geometry was not measured' in note and 'cannot start compute' in note
               for note in row['caveats'])
    assert session.compute.process is None


def test_finalization_programming_error_remains_visible(monkeypatch, tmp_path):
    session = make_session(tmp_path)
    session.compute = runner.ComputeProcess()

    def broken(*args, **kwargs):
        raise TypeError('fixture: invalid compute call')

    monkeypatch.setattr(session.compute, 'call', broken)
    with pytest.raises(TypeError, match='invalid compute call'):
        session.close()


@pytest.mark.parametrize('phase', ['step', 'finalize'])
def test_spawn_start_oserror_still_records_run(monkeypatch, tmp_path, phase):
    import multiprocessing.process

    session = make_session(tmp_path / 'run')
    if phase == 'finalize':
        session.run_recipe()
    session.compute = runner.ComputeProcess()

    def unavailable(*args, **kwargs):
        raise PermissionError('fixture: process startup denied')

    monkeypatch.setattr(multiprocessing.process.BaseProcess, 'start', unavailable)
    if phase == 'step':
        session.run_recipe()
    result = session.close()
    row = json.loads(store.append(result, out_dir=tmp_path).read_text())
    assert row['run_id'] == session.run_id
    assert not row['complete'] and not row['ok']
    assert row['steps'][0]['outcome'] == ('error' if phase == 'step' else 'ok')
    assert any('final geometry was not measured' in note and 'startup denied' in note
               for note in row['caveats'])
    assert session.compute.process is None
    assert session.compute.connection is None


@pytest.mark.skipif(os.name != 'posix', reason='PTY regression requires POSIX')
def test_spawned_tune_comparison_does_not_probe_the_parent_terminal(tmp_path):
    """Unpickling the comparison imports TUI image detection in a fresh worker."""
    import pty
    import termios
    import threading

    program = '''
import time
from pathlib import Path
from types import SimpleNamespace
from manyruns.pipeline.runner import ComputeProcess
from manyruns.tui.app import _tuned_apart

compute = ComputeProcess(log_path=Path('compute.log'))
try:
    assert compute.call(_tuned_apart, SimpleNamespace(out_dir=Path('.')),
                        {'name': 'phate'}, time.time()) is None
finally:
    compute.close()
'''
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    master, slave = pty.openpty()
    termios.tcsetwinsize(slave, (40, 120))
    output = bytearray()

    def drain():
        try:
            while chunk := os.read(master, 65536):
                output.extend(chunk)
        except OSError:
            pass

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        result = subprocess.run(
            [sys.executable, '-c', program], cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=str(root)),
            stdin=slave, stdout=slave, stderr=slave, timeout=10)
        assert result.returncode == 0, output.decode(errors='replace')
    finally:
        os.close(slave)
        os.close(master)
        reader.join(timeout=1)


def test_spawn_keeps_parent_state_progress_and_record_parity(tmp_path):
    direct = make_session(tmp_path / 'direct')
    remote = make_session(tmp_path / 'remote')
    remote.compute = runner.ComputeProcess()
    reports = []
    remote.on_step = lambda rec, rows: reports.append((rec.get('state', 'done'), len(rows)))
    try:
        direct.run_recipe()
        remote.run_recipe()
        assert remote.state['extras']['worker_pid'] != os.getpid()
        np.testing.assert_array_equal(remote.state['emb'], direct.state['emb'])
        assert reports == [('running', 1), ('done', 1)]
        assert remote.ctx['plots'] is remote.plots
        rows = []
        for session in (direct, remote):
            result = session.close()
            path = store.append(result, out_dir=session.out_dir)
            rows.append(json.loads(path.read_text()))
            saved = artifacts.load_array(result['steps'][0]['artifacts']['emb'])
            np.testing.assert_array_equal(saved, session.state['emb'])
        # Wall-clock duration is telemetry, not a computed geometric value.
        for row in rows:
            row['g_vector'].pop('suite_seconds')
        assert rows[0]['g_vector'] == rows[1]['g_vector']
        assert rows[0]['caveats'] == rows[1]['caveats']
        assert direct.trace == remote.trace
        for left, right in zip(rows[0]['steps'], rows[1]['steps']):
            for key in ('name', 'group', 'params', 'outcome', 'emitted', 'produced'):
                assert left[key] == right[key]
    finally:
        remote.compute.close()
