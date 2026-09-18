"""Stat-only drop-folder snapshots and a two-observation copy heuristic.

No reads of data, timers or UI here. A settled signature is evidence of a pause in
writing, not proof a writer has closed the file. Inspection rechecks its signature.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
import os
from pathlib import Path
from typing import Callable, Iterable


def path_call(callback, *args, **kwargs):
    """Normalize pathlib's known filesystem RuntimeErrors, preserving programming errors."""
    try:
        return callback(*args, **kwargs)
    except RuntimeError as error:
        import errno

        if str(error) == 'Could not determine home directory.':
            raise ValueError(str(error)) from error
        cause = error.__context__
        if isinstance(cause, OSError) and cause.errno == errno.ELOOP:
            raise OSError(errno.ELOOP, str(error)) from error
        raise


def read_errors() -> tuple[type[Exception], ...]:
    """Expected read failures; optional config parsers stay off the module-import path."""
    from omegaconf.errors import OmegaConfBaseException
    from yaml import YAMLError

    return OSError, ValueError, YAMLError, OmegaConfBaseException


def row_errors() -> tuple[type[Exception], ...]:
    """Data failures at one source boundary, including HDF5, AnnData and pandas reads.

    Most parser errors inherit ValueError or OSError; AnnData's unknown-encoding
    exception does not. Keep these here, not in filesystem_guard: a broken tick or
    UI callback must still raise. Do not import optional readers to handle a YAML error.
    """
    import sys

    registry = sys.modules.get('anndata._io.specs.registry')
    encoding_errors = (registry.IORegistryError,) if registry is not None else ()
    return (*read_errors(), AttributeError, TypeError, KeyError, IndexError, EOFError,
            *encoding_errors)


def row_guard(build_row):
    """Normalize one source's inspection, QC and row labels for every caller.

    Catalog rows, drops, finder refreshes and fetch read-back all use the decorated
    state constructors. Their callers can skip/refuse this source without hiding
    unrelated failures in the surrounding polling or selection transaction.
    """
    @wraps(build_row)
    def guarded(*args, **kwargs):
        try:
            return path_call(build_row, *args, **kwargs)
        except row_errors() as error:
            source = args[0] if args else kwargs.get('name', kwargs.get('path'))
            raise ValueError(
                f'Could not read {source}: {type(error).__name__}: {error}') from error
    return guarded


def filesystem_guard(error_handler: str | Callable[[Exception], None]):
    """Guard a whole synchronous callback; recovery methods must perform no filesystem I/O.

    Normalize pathlib's symlink-loop and unknown-home failures; unrelated runtime
    and programming errors propagate.
    """
    def decorate(callback):
        @wraps(callback)
        def guarded(*args, **kwargs):
            def recover(error):
                if isinstance(error_handler, str):
                    getattr(args[0], error_handler)(error)
                else:
                    error_handler(error)

            try:
                return path_call(callback, *args, **kwargs)
            except read_errors() as error:
                recover(error)
        return guarded
    return decorate


def canonical(path: Path) -> Path:
    """Resolve a watcher identity, normalizing pathlib's filesystem failures."""
    source = path_call(Path(path).expanduser)
    return path_call(source.resolve)


def absolute(path: Path) -> Path:
    """Keep the source spelling: its suffix determines how inspection/loading dispatch."""
    return path_call(Path(path).expanduser).absolute()


def fingerprint(path: Path) -> tuple | None:
    """Include descendants and link targets, visiting each directory identity only once."""
    import stat

    try:
        info = path.stat()
        root = (info.st_size, info.st_mtime_ns)
        if not stat.S_ISDIR(info.st_mode):
            return root
        rows = []
        visited = set()

        def failed(error):
            raise error

        for parent, dirs, files in os.walk(path, followlinks=True, onerror=failed):
            directory = Path(parent).stat()
            identity = (directory.st_dev, directory.st_ino)
            if identity in visited:
                dirs.clear()
                continue
            visited.add(identity)
            dirs.sort()  # choose the same traversal path for repeated targets each poll
            # Keep link metadata as well as the target metadata a loader follows.
            # Repeated directory identities above stop cycles without hiding file growth.
            for name in sorted(dirs + files):
                child = Path(parent) / name
                st = child.lstat()
                target = ()
                if stat.S_ISLNK(st.st_mode):
                    followed = child.stat()
                    target = (followed.st_mode, followed.st_size, followed.st_mtime_ns,
                              followed.st_dev, followed.st_ino, followed.st_ctime_ns)
                rows.append((str(child.relative_to(path)), st.st_mode,
                             st.st_size, st.st_mtime_ns, target))
        return root + (tuple(sorted(rows)),)
    except (OSError, ValueError):
        return None


@dataclass(frozen=True)
class Snapshot:
    roots: tuple[Path, ...]
    entries: dict[Path, tuple]
    # Catalog-only observations participate in settling, not ordinary drop discovery.
    drops: tuple[Path, ...] | None = None
    # Original absolute references map to their canonical watcher identity. Multiple
    # differently named symlinks can refer to one observed file.
    references: dict[Path, Path] = field(default_factory=dict)


def snapshot(folders: Iterable[Path] | None = None) -> Snapshot:
    """Observe drops and explicit catalog paths, without opening source contents."""
    from manyruns import catalog, shell

    roots = []
    entries = {}
    drops = []
    references = {}
    try:
        sources = path_call(shell.data_dirs) if folders is None else folders
    except (OSError, ValueError):
        sources = ()
    for folder in sources:
        try:
            source_root = absolute(folder)
            root = canonical(source_root)
            if root in roots or not root.is_dir():
                continue
            roots.append(root)
            for path in sorted(source_root.iterdir(), key=lambda p: p.name.lower()):
                if path.name.startswith('.'):
                    continue
                try:
                    if not path.is_dir() and path.suffix.lower() not in shell._DATA_SUFFIXES:
                        continue
                    key = canonical(path)
                    value = fingerprint(path)
                    if value is not None:
                        entries[key] = value
                        drops.append(path)
                        references[path] = key
                except (OSError, ValueError):
                    continue
        except (OSError, ValueError):
            continue
    try:
        names = catalog.discover_datasets()
    except read_errors():  # a broken catalog does not stop ordinary drop observations
        names = ()
    for name in names:
        try:
            ds = catalog.load_dataset(name)
            handle = ds.get('handle') or {}
            if handle.get('kind') != 'path':
                continue
            path = path_call(shell.dataset_ref_path, handle['ref'])
            if path is None:
                continue
            path = absolute(path)
            key = canonical(path)
            watched = next((p for p in (key, *key.parents) if p in entries), None)
            if watched is not None:
                references[path] = watched
                continue  # an existing directory observation also holds its aliases
            value = fingerprint(path)
            if value is not None:
                entries[key] = value
                references[path] = key
        except row_errors():  # guard this declaration's entire observation
            continue
    return Snapshot(tuple(roots), entries, tuple(drops), references)


@dataclass(frozen=True)
class Poll:
    snapshot: Snapshot
    ready: frozenset[Path]
    held: frozenset[Path]


class DropWatch:
    def __init__(self) -> None:
        self._previous: dict[Path, tuple] = {}

    def observe(self, current: Snapshot) -> Poll:
        ready = frozenset(p for p, sig in current.entries.items()
                          if self._previous.get(p) == sig)
        self._previous = dict(current.entries)
        return Poll(current, ready, frozenset(current.entries) - ready)

    def invalidate(self, key: Path) -> None:
        """Forget a captured watcher key without touching a potentially changed filesystem."""
        self._previous.pop(key, None)
