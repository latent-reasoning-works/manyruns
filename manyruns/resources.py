"""Pinned resource files. Resolution reads; only an explicit build prepares files."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

from manyruns import env, toolchain


def home() -> Path:
    return Path(env.get('RESOURCE_HOME') or Path.home() / '.cache/manyruns/resources')


def path_for(name: str, block: dict, key: str | None) -> Path:
    for label, value in [('name', name), ('release', block['release']),
                         *([('key', key)] if key is not None else [])]:
        if not toolchain._file_safe(value):
            raise ValueError(f'resource {label} must be a non-empty file-safe name')
    if block.get('key') and key is None:
        raise ValueError('resource key must be a non-empty file-safe name')
    path = home() / name / block['release'] / f'{key if key is not None else name}.tsv'
    if home().resolve() not in path.resolve().parents:
        raise ValueError('resource path must stay within the resource cache')
    return path


def find(name: str, config_dir=None) -> tuple[dict, dict]:
    for tool in toolchain.discover_tools(config_dir):
        cfg = toolchain.load_tool(tool, config_dir)
        if name in (cfg.get('resource') or {}):
            return cfg, cfg['resource'][name]
    raise ValueError(f'no tool declares resource {name!r}')


def resolve(name: str, block: dict, key: str | None, tool: str) -> dict:
    import shlex

    path = path_for(name, block, key)
    fix = f'manyruns tools build-resource {shlex.quote(name)}'
    if block.get('key'):
        fix += ' --param ' + shlex.quote(f"{block['key']}={key}")
    out = {'status': 'not-built', 'path': str(path), 'sha256': None, 'fix': fix}
    sidecar = path.with_suffix('.tsv.sha256')
    if not path.is_file() or not sidecar.is_file():
        return out
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    out.update(status='ready' if sidecar.read_text().strip() == digest else 'mismatch', sha256=digest)
    return out


def build(name: str, params: dict[str, str], config_dir=None, timeout=3600) -> dict:
    cfg, block = find(name, config_dir)
    key = params.get(block['key'], (cfg.get('params') or {}).get(block['key'])) if block.get('key') else None
    path = path_for(name, block, key)
    resolved = toolchain.resolve(cfg['name'], config_dir)
    if resolved['status'] not in ('ready', 'overridden'):
        raise ValueError(toolchain.install_command(cfg, config_dir))
    from manyruns.pipeline.external import _adapter_path

    prepare = _adapter_path({**cfg, 'mode': 'run', 'entrypoint': block['prepare']}, config_dir=config_dir)
    if prepare is None:
        raise ValueError(f"resource {name}: prepare script {block['prepare']} not found")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='.prepare-', suffix='.tsv', dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        command = [str(resolved['python']), str(prepare), '--url', block['url'],
                   '--release', block['release'], '--out', str(tmp)]
        if key is not None:
            command += ['--key', key]
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if done.returncode:
            raise ValueError(f'resource {name}: prepare failed: {done.stderr.strip()}')
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        os.replace(tmp, path)
        sidecar = path.with_suffix('.tsv.sha256')
        staging = tmp.with_suffix('.sha256')
        try:
            staging.write_text(digest)
            os.replace(staging, sidecar)
        finally:
            staging.unlink(missing_ok=True)
        return {'path': str(path), 'sha256': digest}
    finally:
        tmp.unlink(missing_ok=True)


def provenance(step: str, name: str, block: dict, key, sha256) -> dict:
    prefix = f'{step}.resource.{name}'
    return {f'{prefix}.release': block['release'], f'{prefix}.key': key,
            f'{prefix}.sha256': sha256}
