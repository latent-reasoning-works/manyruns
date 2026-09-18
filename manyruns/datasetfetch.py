"""Missing-only, verified sample downloads into the existing drop folder.

Catalog loading and local resolution never fetch. Call this only after an explicit fetch
gesture. Existing files are user inputs, and are returned without hashing or replacing them.
"""
from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import BinaryIO

from manyruns import catalog, shell

_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()
_TIMEOUT = 30
_CHUNK_SIZE = 64 * 1024


class DatasetFetchError(RuntimeError):
    """An actionable download refusal suitable for plain-text UI/CLI rendering."""


def _open_partial(folder: Path, name: str) -> BinaryIO:
    """Create an exclusive hidden temporary file with normal umask/ACL permissions.

    NamedTemporaryFile fixes permissions at 0600. Ordinary exclusive creation lets the OS
    apply the caller's umask without reading/changing that process-wide state in a thread.
    """
    import secrets

    while True:
        partial = folder / f".{name}.{secrets.token_hex(16)}.part"
        try:
            return partial.open("xb")
        except FileExistsError:
            continue


def can_fetch(ds: Mapping) -> bool:
    """Whether this declaration has a complete, valid relative-file download contract.

    Pure schema inspection: no filesystem reads, hashing or network. Absolute references
    are intentionally ineligible; downloads materialise basenames, never arbitrary paths.
    """
    if not isinstance(ds, Mapping):
        return False
    handle = ds.get("handle")
    return (isinstance(handle, Mapping)
            and all(key in handle for key in ("url", "sha256", "bytes"))
            and not catalog._check_download_handle(handle))


def fetch_dataset(
    ds: Mapping, *, opener: Callable[..., BinaryIO] | None = None, url: str | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Return a local file or atomically publish verified bytes into the drop folder.

    ``opener`` accepts ``(url, timeout=seconds)`` and returns a closable binary stream.
    ``url`` overrides transport only for tests (e.g. a local file URI); the declaration must
    still validate as HTTPS. The default opener is resolved at call time. Progress starts
    at zero and reports received bytes against the declared total, regardless of headers.
    Cancellation is checked while waiting, between reads, and before publication.
    Cleanup failures raise DatasetFetchError with the remaining partial's path, preserving
    any download refusal and its recovery instructions.
    Publication uses an atomic hard link where supported, otherwise an atomic rename of
    the completed partial in the destination folder. The fallback rechecks for an existing
    file under the per-target process lock, but another process can still create a file
    between that recheck and the rename and have it replaced. Readers never see a partially
    written final file from this download, even if the publishing process is killed.
    """
    import errno
    from http.client import HTTPException

    handle = ds.get("handle") or {}
    ref = handle.get("ref")
    if handle.get("kind") != "path" or not isinstance(ref, str) or not ref:
        raise DatasetFetchError("Choose a file dataset with a declared download URL and pins.")
    target = shell.data_dir().resolve() / Path(ref).name
    partial: Path | None = None
    failure: DatasetFetchError | None = None
    acquired = False

    def check_cancelled() -> None:
        if cancelled is not None and cancelled():
            raise DatasetFetchError("download cancelled")

    def existing_file() -> Path | None:
        found = shell.dataset_ref_path(ref)
        if found is not None:
            if not found.is_file():
                raise DatasetFetchError(f"{found} exists but is not a usable file")
            return found.resolve()
        # A dangling symlink must not be replaced either (.exists() cannot see it).
        for candidate in (Path(ref).expanduser(), target):
            if os.path.lexists(candidate):
                raise DatasetFetchError(f"{candidate} exists but is not a usable file")
        return None

    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(target, threading.Lock())
    try:
        while not acquired:
            check_cancelled()
            acquired = lock.acquire(timeout=0.1)
        found = existing_file()
        if found is not None:
            return found
        if not can_fetch(ds):
            raise DatasetFetchError("no validated download URL, sha256 and byte count for this ref")
        folder = shell.ensure_drop_folder().resolve()
        total = handle["bytes"]
        digest = hashlib.sha256()
        done = 0
        with _open_partial(folder, target.name) as output:
            partial = Path(output.name)
            if progress is not None:
                progress(0, total)
            check_cancelled()
            open_url = opener if opener is not None else urllib.request.urlopen
            with closing(open_url(url if url is not None else handle["url"],
                                  timeout=_TIMEOUT)) as response:
                while True:
                    check_cancelled()
                    chunk = response.read(min(_CHUNK_SIZE, total - done + 1))
                    if not chunk:
                        break
                    done += len(chunk)
                    if done > total:
                        raise DatasetFetchError(f"received more than the declared {total} bytes")
                    digest.update(chunk)
                    output.write(chunk)
                    if progress is not None:
                        progress(done, total)
            if done != total:
                raise DatasetFetchError(f"received {done} bytes; expected {total} bytes")
            if digest.hexdigest() != handle["sha256"]:
                raise DatasetFetchError(f"sha256 mismatch; expected {handle['sha256']}")
            output.flush()
        check_cancelled()
        # A manual placement (or another process) can win while the stream is in flight.
        found = existing_file()
        if found is not None:
            return found
        # Both names are on the same filesystem. Linking publishes verified bytes
        # atomically without replacing an entry created after the last check.
        try:
            try:
                os.link(partial, target)
            except OSError as error:
                if error.errno not in {
                    errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM, errno.EXDEV, errno.ENOSYS,
                }:
                    raise
                found = existing_file()
                if found is not None:
                    return found
                # Keep the final name absent until all verified bytes become visible at
                # once, including on filesystems without hard links. See the race above.
                os.replace(partial, target)
        except FileExistsError:
            found = existing_file()
            if found is not None:
                return found
            raise
        return target
    except (DatasetFetchError, OSError, HTTPException, EOFError, ValueError) as error:
        source = handle.get("url") or (ds.get("source") or {}).get("url") or "the data source"
        failure = DatasetFetchError(
            f"Could not fetch {Path(ref).name}: {error}. Retry, or download from {source} "
            f"and put the file at {target} "
            f"(expected {handle.get('bytes', 'declared')} bytes, sha256 {handle.get('sha256', 'unknown')})."
        )
        raise failure from error
    finally:
        try:
            if partial is not None:
                try:
                    partial.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    message = (f"Could not remove partial file {partial}: {cleanup_error}. "
                               "Remove it manually once the folder is writable.")
                    if failure is not None:
                        raise DatasetFetchError(f"{failure} {message}") from failure
                    raise DatasetFetchError(message) from cleanup_error
        finally:
            if acquired:
                lock.release()
