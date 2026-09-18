"""The real app must process keys while a step or metric monopolizes its GIL."""
import argparse
import asyncio
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

from manyruns import app as cli
from manyruns.narrate import Observation
from manyruns.pipeline import runner
from manyruns.session import Session
from manyruns.tui.app import ManyrunsApp, prewarm_multiprocessing
from manyruns.tui.gate import Gate
from manyruns.tui.roster import RosterScreen
from manyruns.tui.run import LeaveRunScreen, RunScreen
from manyruns.tui.state import DataEntry

REAL_FINALIZE = runner._finalize


def hold_gil(folder):
    folder = Path(folder)
    (folder / 'entered').touch()
    # Only the driver may release or cancel this attempt. A local timeout turns a
    # slow cancellation into a completed error attempt and changes the decision row.
    # The driver's bounded waits fail a hang; app teardown terminates the worker.
    while not (folder / 'release').exists():
        time.sleep(.01)
    previous = sys.getswitchinterval()
    try:
        # Pure Python normally yields every 5 ms; pin the switch interval to model
        # the measured extension that retains the GIL for the entire calculation.
        sys.setswitchinterval(5)
        deadline = time.monotonic() + 3
        value = 0
        while time.monotonic() < deadline:
            value += 1
    finally:
        sys.setswitchinterval(previous)
    (folder / 'completed').touch()
    return 7.0


def busy_step(name, params, state, g, ctx):
    folder = Path(state['extras']['phase_dir'])
    with (folder / 'issued').open('a') as stream:
        stream.write(name + '\n')
    if state['extras']['phase'] == 'step' and name in ('pca', 'phate'):
        hold_gil(folder)
    state['emb'] = state['X'][:, :2].copy()
    g[f'{name}.score'] = 7.0
    state['extras']['worker_pid'] = os.getpid()


def tune_step(name, params, state, g, ctx):
    if params.get('knn') == 7:
        hold_gil(state['extras']['phase_dir'])
    busy_step(name, params, state, g, ctx)


def measured_finalize(state, g, **kwargs):
    if state['extras']['phase'] == 'crash-finalize' and not kwargs.get('compute_error'):
        print('fixture: final metric process failed', file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)
    def metric(name, **unused):
        if state['extras']['phase'] == 'finalize':
            return hold_gil(state['extras']['phase_dir'])
        return 7.0

    g['fixture.worker_pid'] = os.getpid()
    with patch('manyruns.catalog.load_suite', return_value=['fixture']), \
            patch.object(runner._suite, '_compute_metric', return_value=metric):
        return REAL_FINALIZE(state, g, **kwargs)


def crashed_comparison(session, step, since):
    print('fixture: tune comparison process failed', file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)


async def until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, 'timed out waiting for the run'
        await asyncio.sleep(.01)


def prepare(monkeypatch, tmp_path, phase):
    monkeypatch.chdir(tmp_path)
    entry = DataEntry(name='swissroll', kind='bundled', dataset='swissroll',
                      obs=Observation(shape='manifold', modality='synthetic', source='swissroll'))
    recipe = {'name': 'embed', 'steps': [{'name': 'pca', 'group': 'latent'},
                                        {'name': 'later', 'group': 'latent'}]}
    session = Session('swissroll', engine='manylatents', recipe=recipe,
                      array=np.arange(36.).reshape(12, 3), out_dir=tmp_path / 'outputs/swissroll')
    session.state['extras'] = dict(phase=phase, phase_dir=str(tmp_path))
    session.dispatch = {'latent': busy_step}
    session.carry['declared'] = []
    monkeypatch.setattr(cli, '_build_session', lambda *a, **kw: session)
    monkeypatch.setattr(cli, 'load_recipe', lambda *a: recipe)
    monkeypatch.setattr('manyruns.catalog.load_recipe', lambda *a: recipe)
    monkeypatch.setattr(runner, '_finalize', measured_finalize)
    monkeypatch.setattr(ManyrunsApp, 'get_default_screen', lambda self: RosterScreen(entries=[entry]))
    prewarm_multiprocessing()
    return ManyrunsApp(argparse.Namespace(engine='manylatents')), entry, session


@pytest.mark.parametrize('phase', ['tiny', 'crash-finalize', 'crash-comparison'])
def test_spawn_inside_app_records_finalization(monkeypatch, tmp_path, phase):
    app, entry, session = prepare(monkeypatch, tmp_path, phase)
    if phase == 'crash-comparison':
        session.recipe['steps'][0]['name'] = 'phate'
        monkeypatch.setattr('manyruns.tui.app._tuned_apart', crashed_comparison)

    async def drive():
        async with app.run_test(size=(140, 40)) as pilot:
            app._open_prepared_run(entry, 'embed')
            await until(lambda: isinstance(app.screen, RunScreen))
            screen = app.screen
            if phase == 'crash-comparison':
                await until(lambda: screen.gate.is_pending(screen._question_id))
                await pilot.press('a', 'enter')
            await until(lambda: screen._finished)
            await until(lambda: not screen.worker_thread.is_alive())
            index = tmp_path / 'outputs/index.jsonl'
            assert index.exists(), screen.error
            rows = [json.loads(line) for line in index.read_text().splitlines()]
            assert len(rows) == 1
            row = rows[0]
            assert row['run_id'] == session.run_id
            assert session.state['extras']['worker_pid'] != os.getpid()
            if phase == 'tiny':
                assert screen.error is None
                assert row['complete']
                assert row['g_vector']['fixture'] == 7.0
                assert row['g_vector']['fixture.worker_pid'] == session.state['extras']['worker_pid']
            else:
                assert not row['complete'] and not row['ok']
                assert not screen.results['cancelled']
                assert any('SIGTERM' in note and 'exit code -15' in note for note in row['caveats'])
                assert 'fixture' not in row['g_vector']
                log = session.out_dir / 'state' / session.run_id / 'compute.log'
                failure = 'tune comparison' if phase == 'crash-comparison' else 'final metric'
                assert f'fixture: {failure} process failed' in log.read_text()

    asyncio.run(drive())


@pytest.mark.parametrize('failure', ['directory', 'file'])
def test_unwritable_compute_diagnostics_still_records_run(monkeypatch, tmp_path, failure):
    app, entry, session = prepare(monkeypatch, tmp_path, 'tiny')
    state_dir = session.out_dir / 'state'
    state_dir.mkdir(parents=True)
    log_path = state_dir / session.run_id / 'compute.log'
    if failure == 'directory':
        state_dir.chmod(0o500)
    else:
        log_path.mkdir(parents=True)  # A directory cannot be opened as the log file.

    async def drive():
        async with app.run_test(size=(140, 40)):
            app._open_prepared_run(entry, 'embed')
            await until(lambda: isinstance(app.screen, RunScreen))
            screen = app.screen
            await until(lambda: screen.worker_thread is not None)
            await until(lambda: not screen.worker_thread.is_alive())
            index = tmp_path / 'outputs/index.jsonl'
            assert index.exists(), screen.error
            row = json.loads(index.read_text())
            assert row['run_id'] == session.run_id
            assert row['complete'] and row['ok']
            assert [step['outcome'] for step in row['steps']] == ['ok', 'ok']
            assert row['g_vector']['fixture'] == 7.0
            assert session.state['extras']['worker_pid'] != os.getpid()
            assert session.compute.process is None

    try:
        with pytest.warns(RuntimeWarning, match='compute diagnostics unavailable') as warnings:
            asyncio.run(drive())
        assert len(warnings) == 1
        assert str(log_path) in str(warnings[0].message)
    finally:
        state_dir.chmod(0o700)


@pytest.mark.parametrize('first', ['pca', 'phate'])
def test_abandon_first_offloaded_step_records_not_run(monkeypatch, tmp_path, first):
    app, entry, session = prepare(monkeypatch, tmp_path, 'step')
    session.recipe['steps'][0]['name'] = first

    async def drive():
        async with app.run_test(size=(140, 40)) as pilot:
            app._open_prepared_run(entry, 'embed')
            await until(lambda: isinstance(app.screen, RunScreen))
            screen = app.screen
            await until(lambda: (tmp_path / 'entered').exists())
            await pilot.press('escape')
            await until(lambda: isinstance(app.screen, LeaveRunScreen))
            await pilot.click('#run-again')
            await until(lambda: not screen.worker_thread.is_alive())
            index = tmp_path / 'outputs/index.jsonl'
            assert index.exists(), screen.error
            rows = [json.loads(line) for line in index.read_text().splitlines()]
            assert len(rows) == 1
            row = rows[0]
            assert row['run_id'] == session.run_id
            assert not row['complete']
            assert row['steps'] == [], row['steps']
            assert session.steps == []
            assert any(f'step 0 ({first}): not run' in note and 'run abandoned' in note
                       for note in row['caveats'])
            assert any('step 1 (later): not run' in note and 'run abandoned' in note
                       for note in row['caveats'])
            assert 'run abandoned: final geometry was not measured' in row['caveats']
            assert (tmp_path / 'issued').read_text().splitlines() == [first]
            assert session.compute.process is None

    asyncio.run(drive())


async def abandon_third_tune_attempt(monkeypatch, folder, *, offloaded=True, during_attempt=True):
    folder.mkdir()
    app, entry, session = prepare(monkeypatch, folder, 'tiny')
    session.recipe['steps'][0] = {'name': 'phate', 'group': 'latent', 'params': {'knn': 5}}
    session.recipe['steps'].append(dict(session.recipe['steps'][1]))
    session.dispatch = {'latent': tune_step}
    saved_metrics = session.ctx['metrics']
    cancelled = threading.Event()
    if not offloaded:
        # Keep the same engine/records, but execute the completed fits in process.
        # An in-process fit cannot be killed: inject its cooperative interruption
        # at the step seam, before it commits the third attempt's result.
        monkeypatch.setattr(runner, 'ComputeProcess', lambda **kwargs: None)
        original_step = session.step

        def interruptible_step(step):
            if step.get('params', {}).get('knn') == 7:
                (folder / 'entered').touch()
                cancelled.wait()  # The driver's finally also releases this on failure.
                raise runner.ComputeCancelled('fixture: tune attempt interrupted')
            return original_step(step)

        monkeypatch.setattr(session, 'step', interruptible_step)

    async with app.run_test(size=(140, 40)) as pilot:
        try:
            app._open_prepared_run(entry, 'embed')
            await until(lambda: isinstance(app.screen, RunScreen))
            screen = app.screen
            finished = []
            post_message = screen.post_message

            def capture_finished(message):
                if isinstance(message, RunScreen.Finished):
                    finished.append(message)
                return post_message(message)

            # Departure unmounts the screen, so its Finished handler may never run.
            monkeypatch.setattr(screen, 'post_message', capture_finished)
            previous = 0
            for answer in (('knn=6', 'knn=7') if during_attempt else ('knn=6',)):
                await until(lambda: screen.gate.is_pending(screen._question_id)
                            and screen._question_id > previous)
                previous = screen._question_id
                await pilot.press(*answer, 'enter')
            if during_attempt:
                await until(lambda: (folder / 'entered').exists())
            else:
                # Baseline: abandoning the unanswered prompt uses Gate's normal cancel
                # response, with the same two completed alternatives and no exception.
                await until(lambda: screen.gate.is_pending(screen._question_id)
                            and screen._question_id > previous)
            await pilot.press('escape')
            await until(lambda: isinstance(app.screen, LeaveRunScreen))
            try:
                await pilot.click('#run-again')
            finally:
                cancelled.set()
            await until(lambda: not screen.worker_thread.is_alive())
            assert len(finished) == 1
            assert finished[0].error is None, finished[0].error
            assert screen.error is None
            rows = [json.loads(line) for line in (folder / 'outputs/index.jsonl').read_text().splitlines()]
            assert len(rows) == 1
            row = rows[0]
            assert row['run_id'] == session.run_id
            assert session.state['emb'] is None
            assert 'phate.score' not in session.g
            assert session.ctx['metrics'] == saved_metrics
            assert session.compute is None or session.compute.process is None
            decisions = session.out_dir / 'decisions.jsonl'
            decision_rows = ([json.loads(line) for line in decisions.read_text().splitlines()]
                             if decisions.exists() else [])
            return session, row, decision_rows
        finally:
            cancelled.set()


def test_abandon_third_tune_attempt_keeps_completed_alternatives(monkeypatch, tmp_path):
    session, row, decisions = asyncio.run(
        abandon_third_tune_attempt(monkeypatch, tmp_path / 'run'))
    assert len(row['steps']) == 2
    assert [step['params']['knn'] for step in row['steps']] == [5, 6]
    assert all(step['outcome'] == 'ok' for step in row['steps'])
    assert session.discarded == ['phate']
    assert any('phate: discarded' in note for note in row['caveats'])
    assert not any('(phate): not run' in note for note in row['caveats'])
    for index in (1, 2):
        assert f'step {index} (later): not run — run abandoned' in row['caveats']
    assert not row['complete']
    assert 'run abandoned: final geometry was not measured' in row['caveats']
    assert len(decisions) == 1
    assert decisions[0]['run_id'] == session.run_id
    assert decisions[0]['chosen'] is None
    assert [offer['params']['knn'] for offer in decisions[0]['offered']] == [5, 6]


@pytest.mark.parametrize('during_attempt', [False, True])
def test_cancel_during_tune_attempt_has_inprocess_record_parity(monkeypatch, tmp_path, during_attempt):
    records = []
    for offloaded in (True, False):
        with monkeypatch.context() as patches:
            session, row, decisions = asyncio.run(abandon_third_tune_attempt(
                patches, tmp_path / str(offloaded), offloaded=offloaded,
                during_attempt=offloaded or during_attempt))
        assert len(decisions) == 1
        # Each run has its own identity and timestamp; all decision content must match.
        decision = {key: value for key, value in decisions[0].items()
                    if key not in ('run_id', 'decision_id', 'at')}
        records.append((row['caveats'], decision, session.discarded))
    assert records[0] == records[1]


@pytest.mark.parametrize('phase', ['step', 'finalize'])
@pytest.mark.parametrize('action', ['complete', 'leave', 'quit', 'stay'])
def test_compute_keeps_heartbeat_and_keys_live(monkeypatch, tmp_path, phase, action):
    app, entry, session = prepare(monkeypatch, tmp_path, phase)

    async def drive():
        beats = []
        async with app.run_test(size=(140, 40)) as pilot:
            app.set_interval(.05, lambda: beats.append(time.monotonic()))
            app._open_prepared_run(entry, 'embed')
            await until(lambda: isinstance(app.screen, RunScreen))
            screen = app.screen
            await until(lambda: (tmp_path / 'entered').exists())
            beats.append(time.monotonic())
            (tmp_path / 'release').touch()
            await asyncio.sleep(.1)
            if action in ('leave', 'stay'):
                started = time.monotonic()
                await pilot.press('escape')
                await until(lambda: isinstance(app.screen, LeaveRunScreen), timeout=1)
                assert time.monotonic() - started < 1
                if action == 'stay':
                    await until(lambda: (tmp_path / 'completed').exists())
                    # The fit can finish while its leave modal is open. No later step
                    # or final suite may start until the user chooses to stay.
                    await asyncio.sleep(.15)
                    if phase == 'step':
                        assert (tmp_path / 'issued').read_text().splitlines() == ['pca']
                    await pilot.press('escape')
                    await until(lambda: screen._finished)
                    assert screen.error is None
                    assert screen.results['complete']
                else:
                    await pilot.press('tab', 'enter')
            elif action == 'quit':
                started = time.monotonic()
                await pilot.press('ctrl+c')
                assert time.monotonic() - started < 1
            else:
                await until(lambda: screen._finished)
                assert screen.error is None
                assert screen.results['g_vector']['fixture'] == 7.0
            await until(lambda: not screen.worker_thread.is_alive())
            beats.append(time.monotonic())
            assert max(b - a for a, b in zip(beats, beats[1:])) < 1
            if action in ('leave', 'quit'):
                assert not (tmp_path / 'completed').exists(), 'cancel did not terminate compute'
                if phase == 'step':
                    assert (tmp_path / 'issued').read_text().splitlines() == ['pca']
            return screen

    screen = asyncio.run(drive())
    assert not screen.worker_thread.is_alive()
    assert session.compute.process is None


def test_real_swissroll_embed_keeps_heartbeat_live(monkeypatch, tmp_path):
    pytest.importorskip('manylatents')
    monkeypatch.chdir(tmp_path)
    entry = DataEntry(name='swissroll', kind='bundled', dataset='swissroll',
                      obs=Observation(shape='manifold', modality='synthetic', source='swissroll'))
    monkeypatch.setattr(ManyrunsApp, 'get_default_screen', lambda self: RosterScreen(entries=[entry]))
    prewarm_multiprocessing()
    app = ManyrunsApp(argparse.Namespace(engine='manylatents'))

    async def drive():
        beats = [time.monotonic()]
        async with app.run_test(size=(140, 40)) as pilot:
            app.set_interval(.05, lambda: beats.append(time.monotonic()))
            app._open_prepared_run(entry, 'embed')
            await until(lambda: isinstance(app.screen, RunScreen))
            screen = app.screen
            # Completion is a precondition for record checks, not a speed claim.
            deadline = time.monotonic() + 900
            while not screen._finished:
                assert time.monotonic() < deadline, 'real engine did not finish'
                if screen.gate.is_pending(screen._question_id):
                    await pilot.press('a', 'enter')
                await asyncio.sleep(.05)
            assert screen.error is None
            steps = screen.results['steps']
            assert [s['name'] for s in steps] == ['normalize', 'transform', 'pca', 'phate']
            assert all(s['outcome'] == 'ok' for s in steps[-2:]), steps
            assert screen.results['g_vector']['suite_measured'] > 0
            rows = [json.loads(line) for line in (tmp_path / 'outputs/index.jsonl').read_text().splitlines()]
            assert len(rows) == 1 and rows[0]['run_id'] == screen.results['run_id']
            beats.append(time.monotonic())
            assert max(b - a for a, b in zip(beats, beats[1:])) < 1
            await until(lambda: not screen.worker_thread.is_alive())

    asyncio.run(drive())


@pytest.mark.parametrize('phase', ['step', 'finalize'])
def test_quit_exits_the_interpreter_with_compute_in_flight(tmp_path, phase):
    """A returned Textual loop must not leave multiprocessing's exit join blocked."""
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    test_root = root / 'tests'
    assert test_root.is_dir()
    program = '''
import asyncio
import os
from pathlib import Path
import sys
import pytest
from test_tui_responsiveness import prepare, until, RunScreen

folder, phase = Path(sys.argv[1]), sys.argv[2]
patches = pytest.MonkeyPatch()
app, entry, session = prepare(patches, folder, phase)

async def drive():
    async with app.run_test(size=(140, 40)) as pilot:
        app._open_prepared_run(entry, 'embed')
        await until(lambda: isinstance(app.screen, RunScreen))
        await until(lambda: (folder / 'entered').exists())
        (folder / 'worker-pid').write_text(str(session.compute.process.pid))
        (folder / 'release').touch()
        await asyncio.sleep(.1)
        (folder / 'quitting').touch()
        await pilot.press('ctrl+c')

asyncio.run(drive())
'''
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(test_root))))
    process = subprocess.Popen([sys.executable, '-c', program, str(tmp_path), phase],
                               cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / 'quitting').exists() and process.poll() is None:
            assert time.monotonic() < deadline, 'subprocess did not reach compute'
            time.sleep(.01)
        stdout, stderr = process.communicate(timeout=1.5)
        assert process.returncode == 0, stdout + stderr
        assert (tmp_path / 'quitting').exists()
        pid = int((tmp_path / 'worker-pid').read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize('navigate', [False, True])
def test_draft_escape_keeps_the_waiting_answer_typeable(monkeypatch, navigate):
    monkeypatch.setattr(ManyrunsApp, 'get_default_screen', lambda self: RosterScreen(entries=[]))
    monkeypatch.setattr(RunScreen, '_look_for_a_model', lambda self: None)
    gate = Gate()
    screen = RunScreen(gate=gate, start=lambda on_step: {'answer': gate.read('accept or cancel?')})
    app = ManyrunsApp(argparse.Namespace(engine='mock'))

    async def drive():
        async with app.run_test(size=(140, 40)) as pilot:
            app.push_screen(screen)
            await until(lambda: gate.is_pending(screen._question_id))
            field = screen.query_one('#ask-input')
            await pilot.press(*'draft')
            assert field.value == 'draft'
            if navigate:
                screen.query_one('#steps-pane').focus()
            await pilot.press('escape')
            assert field.value == ''
            assert screen.focused is field
            await pilot.press('a', 'enter')
            await until(lambda: screen._finished)
            assert screen.results['answer'] == 'accept'
            assert screen.error is None
        await until(lambda: not screen.worker_thread.is_alive())

    asyncio.run(drive())
