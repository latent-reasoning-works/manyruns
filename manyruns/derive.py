"""Connector offers and derived runs over the recorded cells of a finished run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from manyruns import actions, artifacts, catalog, resources, score, store as records, toolchain, vocab


def parent_row(run_id: str, store) -> dict:
    rows = [row for row in records.read(Path(store) / 'outputs') if row.get('run_id') == run_id and score.verdict_of(row) is None]
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


def _resource_offer(cfg: dict) -> list[dict]:
    """Describe declared resources without preparing files or opening the dataset."""
    import shlex

    result = []
    for name, block in sorted((cfg.get('resource') or {}).items()):
        key_param = block.get('key')
        item = {'name': name, 'release': block['release'], 'key_param': key_param,
                # external._gene_ids reads this mapping only without an Ensembl column.
                'required': 'if_no_gene_ids' if name == 'gene_ids' else 'always'}
        if key_param:
            directory = resources.path_for(name, block, key_param).parent
            built_keys = []
            for path in sorted(directory.glob('*.tsv')):
                try:
                    found = resources.resolve(name, block, path.stem, cfg['name'])
                except ValueError:
                    # Ignore invalid keys and paths outside the resource cache.
                    continue
                if found['status'] == 'ready':
                    built_keys.append(path.stem)
            item.update(built_keys=built_keys,
                        build=f'manyruns tools build-resource {shlex.quote(name)}'
                              f' --param {shlex.quote(key_param)}=<{key_param}>')
        else:
            found = resources.resolve(name, block, None, cfg['name'])
            item.update(built=found['status'] == 'ready', status=found['status'], build=found['fix'])
        result.append(item)
    return result


def resource_problem(entry: dict, params: dict) -> str | None:
    """Return a refusal code for an unavailable required resource at choice time."""
    merged = {**entry['params'], **params}
    for item in entry['resources']:
        if item['required'] != 'always':
            continue
        name, release, key_param = item['name'], item['release'], item['key_param']
        suffix = f'{name}:{release}'
        if key_param:
            key = merged.get(key_param)
            suffix += f':{key}'
            try:
                status = resources.resolve(name, {'release': release, 'key': key_param},
                                           key, entry['connector'])['status']
            except ValueError:
                status = 'not-built'
        else:
            status = item['status']
        if status != 'ready':
            code = 'resource_mismatch' if status == 'mismatch' else 'resource_not_built'
            return f'{code}:{suffix}'
    return None


def offer(dataset: str, run: str, store, *, dataset_dir=None) -> dict:
    parent = parent_row(run, store)
    if (parent.get('dataset_name') or root_row(parent, store).get('source')) != dataset:
        raise actions.UsageError(f'parent run {run} belongs to another dataset')
    try:
        tables = parent_tables(parent, store)
    except (OSError, ValueError) as exc:
        raise actions.LoadError('parent run', run, str(exc)) from exc
    ds = dataset_description(root_row(parent, store), dataset_dir)
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
        resource_items = _resource_offer(cfg)
        blocked += [f"resource_not_built:{item['name']}:{item['release']}"
                    for item in resource_items
                    if item['key_param'] is None and item['required'] == 'always' and not item['built']]
        roles = list((cfg.get('input') or {}).get('obs') or [])
        offered.append({
            'connector': connector, 'can_run': not blocked, 'blocked': blocked,
            'params': dict(cfg.get('params') or {}),
            'declaration': (cfg.get('declaration') or {}).get('params') or {},
            'binds': binds, 'selection': 'selection' in inputs, 'roles': roles,
            'role_keys': {role: parent.get(f'{role}_key') for role in roles},
            'resources': resource_items,
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


def root_row(row, store) -> dict:
    """Follow derived parents to the run that recorded the cell identities."""
    seen = set()
    while str(row.get('recipe') or '').startswith('derive:'):
        if row['run_id'] in seen:
            raise ValueError('derived parent chain contains a cycle')
        seen.add(row['run_id'])
        parent = row.get('parent') or {}
        if not parent.get('run_id'):
            raise ValueError('derived parent chain has no root run')
        row = parent_row(parent['run_id'], store)
    return row


def cell_list(root, index, store) -> list[str]:
    """Read identities from the named persisted slot, never from a nearby step."""
    entries = [e for e in _manifest(root, store).values() if e['index'] == index]
    embeddings = [e for e in entries if e['slot'] == 'emb']
    # An index can have several byproducts. The embedding is the plotted slot.
    candidates = embeddings or entries
    ids = ((candidates[0].get('row_identity') or {}).get('sample_ids')
           if len(candidates) == 1 else None)
    if ids is None:
        raise ValueError(f'parent run has no recorded cell ids at step {index}; select a slot with row_identity')
    if not isinstance(ids, list) or not ids or any(not isinstance(c, str) or not c for c in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'parent run has invalid recorded cell ids at step {index}')
    return ids


def _root_reference(parent, keys, store):
    root = root_row(parent, store)
    entries = list(_manifest(parent, store).values())
    chained = str(parent.get('recipe') or '').startswith('derive:')
    if 'parent_index' in keys:
        index = keys['parent_index']
    else:
        candidates = entries if chained else [e for e in entries if e['slot'] == 'emb']
        if not candidates:
            raise ValueError('parent run has no persisted emb; supply parent_index naming a recorded slot')
        index = max(e['index'] for e in candidates)
    if type(index) is not int or index not in {e['index'] for e in entries}:
        raise ValueError('parent_index must name a persisted slot')
    root_index = index
    if chained:
        g = parent.get('g_vector') or {}
        root_index = g.get('derive.root_index')
        if g.get('derive.root_run') != root['run_id'] or type(root_index) is not int:
            raise ValueError('derived parent has no recorded root run and root index')
    return root, root_index, index


def dataset_description(root, dataset_dir=None):
    """Describe a catalog dataset or the source and roles recorded for a user file."""
    name = root.get('dataset_name')
    if name in catalog.discover_datasets(dataset_dir):
        return catalog.load_dataset(name, dataset_dir)
    ref = str(root.get('source') or '')
    handle = {'kind': 'path', 'ref': ref}
    provided = set(vocab.dataset_provides('unknown', handle))
    provided.update(fact for fact in ('samples', 'batch')
                    if root.get('sample_key' if fact == 'samples' else 'batch_key'))
    handle['provides'] = sorted(provided)
    return {'shape': 'unknown', 'modality': root.get('modality') or 'unknown', 'handle': handle}


def source_path(root, ds, store):
    """Prefer the recorded source, then the CLI's catalog and drop-folder anchors."""
    from manyruns import shell

    tried = []
    if root.get('source'):
        recorded = _path(Path(root['source']).expanduser(), store)
        tried.append(recorded)
        if recorded.is_file():
            return recorded
    handle = ds['handle']
    ref = handle.get('ref') or ''
    if handle['kind'] == 'path' and not ref.startswith('synthetic:') and ref:
        literal = Path(ref).expanduser()
        candidates = [_path(literal, store)]
        if not literal.is_absolute():
            candidates.extend(folder / literal.name for folder in shell.data_dirs())
        for candidate in candidates:
            tried.append(candidate)
            if candidate.is_file():
                return candidate
    paths = ', '.join(dict.fromkeys(map(str, tried))) or '(no file paths recorded)'
    raise ValueError(f'parent dataset file is unavailable; paths tried: {paths}')


def open_derived(parent: dict, connector: str, params: dict, keys: dict, store):
    """Reload and align the source before creating a derived lineage."""
    import numpy as np
    from manyruns import app
    from manyruns.pipeline import runner
    from manyruns.session import Session

    root, root_index, index = _root_reference(parent, keys, store)
    cells = cell_list(root, root_index, store)
    dataset = root.get('dataset_name')
    ds = dataset_description(root)
    source = source_path(root, ds, store)
    roles = {key: keys.get(key, parent.get(key)) for key in
             ('time_key', 'condition_key', 'sample_key', 'batch_key')}
    cfg = toolchain.load_tool(connector)
    loaded = app._read_inputs(engine='manylatents', data_folder=source, dataset=None,
                              layers=tuple((cfg.get('input') or {}).get('layers') or ()), **roles)
    source_ids = loaded.get('obs_names')
    if source_ids is None:
        raise ValueError('parent source has no cell ids')
    source_ids = list(map(str, source_ids))
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('parent source has duplicate cell ids')
    positions = {cell: i for i, cell in enumerate(source_ids)}
    missing = [cell for cell in cells if cell not in positions]
    if missing:
        raise ValueError(f'parent source is missing cell ids from step {root_index}: {missing[:5]}')
    rows = np.asarray([positions[cell] for cell in cells])
    for key in ('array', 'counts', 'labels', 'obs_names'):
        if loaded.get(key) is not None:
            loaded[key] = loaded[key][rows]
    for key in ('layers', 'obs'):
        loaded[key] = {k: v[rows] for k, v in (loaded.get(key) or {}).items()}
    loaded['color'] = [{**c, 'values': c['values'][rows]} for c in loaded.get('color') or []]
    out_dir = state_dir(parent, store).parent.parent
    step = {'name': connector, 'group': 'analysis', 'tool': connector, 'params': params}
    engine = parent['engine'] if runner.steppable(parent['engine']) else 'manylatents'
    session = Session(
        project=out_dir.name, engine=engine, out_dir=out_dir,
        seed=keys.get('seed', parent['seed'] if parent.get('seed') is not None else 42), modality=ds['modality'], shape=ds['shape'],
        recipe={'name': f'derive:{connector}', 'steps': [step]},
        parent={'run_id': parent['run_id'], 'index': index}, dataset_name=dataset,
        sample_ids=np.asarray(cells), array=loaded['array'], counts=loaded.get('counts'),
        genes=loaded.get('genes'), labels=loaded.get('labels'), label_kind=loaded.get('kind'),
        label_key=loaded.get('label_key'), layers=loaded.get('layers'), obs=loaded.get('obs'),
        color=loaded.get('color'), source_path=source,
    )
    session.measure = False
    session.sample_key = roles['sample_key']
    session.batch_key = roles['batch_key']
    session.ctx.update(derive=True, role_keys=app._used_keys(loaded))
    session.g.update({'derive.root_run': root['run_id'], 'derive.root_index': root_index})
    return session


def run_derived(choice: dict, doc: dict, selection: Path | None, *, seed: int | None, store='.') -> dict:
    """Run one checked connector and append its decision and derived record."""
    import hashlib
    import io
    import pandas as pd
    from manyruns import action_space, decisions
    from manyruns.pipeline import external
    from manyruns.pipeline.steps import _StepSkipped

    parent = parent_row(choice['parent_run'], store)
    chosen = choice['chosen']
    keys = {key: choice[key] for key in ('parent_index', *action_space._COLUMN_KEYS) if key in choice}
    selection_df = None
    # Reopen and check the selection before either append-only record is written.
    try:
        session = open_derived(parent, chosen['connector'], chosen['params'], {**keys, **({'seed': seed} if seed is not None else {})}, store)
        bindings = parent_tables(parent, store)
        if selection is not None:
            raw = Path(selection).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != choice.get('selection_sha256'):
                raise action_space.Refused(2, f"selection sha256 {digest} differs from selection_sha256 {choice.get('selection_sha256')}")
            selection_df = pd.read_csv(io.BytesIO(raw), sep='\t', dtype=str, keep_default_na=False, na_filter=False)
            external._check_selection(f"parent step {session.g['derive.root_index']}", selection_df,
                                      session.ctx['sample_ids'])
            if set(selection_df.columns) != {'cell', 'group'}:
                raise ValueError('selection must contain only cell and group columns')
    except (OSError, ValueError, _StepSkipped) as exc:
        raise action_space.Refused(1, str(exc)) from exc
    proposed = None
    if 'proposed' in choice:
        filler = choice['proposed']
        proposed = {**filler, 'chosen': action_space.label(choice, filler['chosen'], doc)}
    decision_id = decisions.append(
        offered=doc['offered'], chosen=action_space.label(choice, chosen, doc),
        dataset=doc['dataset']['name'], shape=doc['dataset']['shape'], topology=doc['dataset']['topology'],
        surface='derive', gate=None, chooser=choice['chooser'], bound=choice['bound'],
        proposed=proposed, out_dir=Path(store) / 'outputs',
    )
    result = {'schema_version': 1, 'decision_id': decision_id, 'run_id': None, 'exit_code': 1}
    if decision_id is None:
        print("error: decision not recorded; nothing was run", file=sys.stderr)
        return result
    print(f"decision {decision_id} recorded in {decisions.index_path(Path(store) / 'outputs')}")
    selection_path = None
    if selection_df is not None:
        selection_path = artifacts.save_table(session.out_dir, session.run_id, 0, 'derive', 'selection', selection_df)
        session.carry['written'][str(selection_path)] = {'step': 'derive', 'index': 0, 'slot': 'table:selection'}
        bindings['selection'] = str(selection_path)
        session.g['derive.selection.sha256'] = hashlib.sha256(selection_path.read_bytes()).hexdigest()
        session.g['derive.selection.rows'] = len(selection_df)
    session.ctx['input_tables'] = bindings
    session.g['derive.cells_sha256'] = hashlib.sha256('\n'.join(session.ctx['sample_ids']).encode()).hexdigest()
    rec = session.apply(session.recipe['steps'][0])
    if selection_path is not None:
        rec.setdefault('artifacts', {})['selection'] = str(selection_path)
    row = session.close()
    index = records.append(row, out_dir=Path(store) / 'outputs', extra={
        'source': str(session.ctx['source_path']), 'modality': session.modality,
        'decision_id': decision_id, **session.ctx['role_keys'],
        **({'extra': choice['extra']} if 'extra' in choice else {}),
    })
    print(f"run {session.run_id} recorded in {index}")
    result.update(run_id=session.run_id, exit_code=0 if row['ok'] else 1)
    return result


def parameter_problems(connector: str, declaration: dict, params: dict) -> list[str]:
    """Check a connector's choice without requiring dataset dimensions."""
    from manykinds.params import check

    return check({'method': connector, 'params': declaration}, params,
                 n_samples=None, n_features=None)
