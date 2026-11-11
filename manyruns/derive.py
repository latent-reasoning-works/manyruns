"""Connector offers and derived runs over the recorded cells of a finished run."""
from __future__ import annotations

import json
from pathlib import Path

from manyruns import actions, artifacts, catalog, store as records, toolchain, vocab


def parent_row(run_id: str, store) -> dict:
    rows = [row for row in records.read(Path(store) / 'outputs') if row.get('run_id') == run_id]
    if not rows:
        raise actions.UsageError(f'unknown parent run {run_id!r}')
    return rows[-1]


def _path(path, store) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(store) / path


def state_dir(row: dict, store) -> Path:
    for step in row.get('steps') or []:
        for path in (step.get('artifacts') or {}).values():
            if not isinstance(path, str) or path.startswith('not written:'):
                continue
            directory = _path(path, store).parent
            if directory.name == row['run_id'] and artifacts.complete(directory):
                return directory
    raise ValueError(f"parent run {row['run_id']} has no COMPLETE state folder")


def _manifest(row, store) -> dict:
    return json.loads((state_dir(row, store) / artifacts.DONE).read_text())


def parent_tables(row, store) -> dict[str, str]:
    entries = sorted(_manifest(row, store).items(), key=lambda item: (item[1]['index'], item[0]))
    return {entry['slot'][6:]: str(_path(path, store)) for path, entry in entries
            if entry['slot'].startswith('table:')}


def offer(dataset: str, run: str, store, *, dataset_dir=None) -> dict:
    parent = parent_row(run, store)
    if parent.get('dataset_name') != dataset:
        raise actions.UsageError(f'parent run {run} belongs to another dataset')
    try:
        tables = parent_tables(parent, store)
    except (OSError, ValueError) as exc:
        raise actions.LoadError('parent run', run, str(exc)) from exc
    ds = catalog.load_dataset(dataset, dataset_dir)
    offered = []
    for connector in toolchain.discover_tools():
        cfg = toolchain.load_tool(connector)
        if not toolchain.is_connector(cfg):
            continue
        inputs = toolchain.input_tables(cfg)
        binds = sorted(set(inputs) - {'selection'})
        step = {'name': connector, 'group': 'analysis', 'tool': connector}
        blocked = [f'parent has no table {name}' for name in binds if name not in tables]
        blocked += [f'needs {fact}' for fact in sorted(vocab.unmet(
            {'steps': [step]}, ds['shape'], handle=ds['handle']))]
        if parent.get('ok') is not True:
            blocked.append('parent run is not ok')
        roles = list((cfg.get('input') or {}).get('obs') or [])
        offered.append({
            'connector': connector, 'can_run': not blocked, 'blocked': blocked,
            'params': dict(cfg.get('params') or {}),
            'declaration': (cfg.get('declaration') or {}).get('params') or {},
            'binds': binds, 'selection': 'selection' in inputs, 'roles': roles,
            'role_keys': {role: parent.get(f'{role}_key') for role in roles},
        })
    return {'gate': None, 'evidence': None,
            'parent': {'run_id': run, 'recipe': parent['recipe'],
                       'index': max((s['index'] for s in parent.get('steps') or []), default=0)},
            'embeddings': _embeddings(parent, store), 'offered': offered}


def _embeddings(parent, store) -> list[dict]:
    entries = _manifest(parent, store)
    result = []
    for path, entry in sorted(entries.items(), key=lambda item: (item[1]['index'], item[0])):
        if entry['slot'] != 'emb':
            continue
        ids = (entry.get('row_identity') or {}).get('sample_ids')
        if ids is not None:
            rows = len(ids)
        else:
            import numpy as np

            rows = int(np.load(_path(path, store), mmap_mode='r', allow_pickle=False).shape[0])
        result.append({'index': entry['index'], 'rows': rows})
    return result


def parameter_problems(connector: str, declaration: dict, params: dict) -> list[str]:
    """Check a connector's choice without requiring dataset dimensions."""
    from manykinds.params import check

    return check({'method': connector, 'params': declaration}, params,
                 n_samples=None, n_features=None)
