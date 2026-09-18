"""First-use downloads are transactions; every byte comes from an injected transport."""
from __future__ import annotations

import errno
import hashlib
import io
import json
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.client import IncompleteRead
from pathlib import Path
from threading import Event
from time import monotonic
from urllib.error import HTTPError, URLError

import pytest

from manyruns import shell


PAYLOAD = b"a small sample\n"


@pytest.fixture(autouse=True)
def isolated_fetch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "new" / "drop"))
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(tmp_path / "catalog"))

    def no_network(*args, **kwargs):
        pytest.fail("a fetch test attempted an uninjected request")

    monkeypatch.setattr("urllib.request.urlopen", no_network)


@pytest.fixture
def ds():
    return {"name": "sample", "shape": "clusters", "handle": {
        "kind": "path", "ref": "data/sample.h5ad", "bytes": len(PAYLOAD),
        "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
        "url": "https://example.org/sample.h5ad",
    }}


def destination():
    return shell.data_dir().resolve() / "sample.h5ad"


def assert_empty_drop():
    root = shell.data_dir()
    assert root.is_dir()
    assert list(root.iterdir()) == []


def test_verified_first_use_is_invisible_until_complete_and_reused(ds):
    from manyruns import datasetfetch

    events = []
    calls = []
    response = io.BytesIO(PAYLOAD)

    def opener(url, *, timeout):
        calls.append(url)
        assert 0 < timeout <= 60
        return response

    def progress(done, total):
        events.append((done, total))
        assert not destination().exists(), "unverified bytes must never have the final name"
        root = shell.data_dir()
        assert root.is_dir()
        for partial in root.iterdir():
            assert partial.name.startswith(".")
            assert partial.suffix not in shell._DATA_SUFFIXES

    assert not shell.data_dir().exists()
    result = datasetfetch.fetch_dataset(ds, opener=opener, progress=progress)
    assert result == destination()
    assert result.read_bytes() == PAYLOAD
    assert response.closed
    assert events[0] == (0, len(PAYLOAD))
    assert events[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert calls == [ds["handle"]["url"]]
    assert datasetfetch.fetch_dataset(ds) == result  # default opener must not be called
    assert shell.data_dir().is_dir()
    assert list(shell.data_dir().iterdir()) == [result]


def test_existing_user_copy_is_never_verified_or_replaced(ds):
    """Preservation: a local file is user input, even when it differs from catalog pins."""
    from manyruns import datasetfetch

    target = destination()
    target.parent.mkdir(parents=True)
    target.write_bytes(b"my edited dataset")
    before = target.stat()
    assert datasetfetch.fetch_dataset(ds) == target
    assert target.read_bytes() == b"my edited dataset"
    assert target.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("code", sorted({
    errno.ENOTSUP, errno.EOPNOTSUPP, errno.EPERM, errno.EXDEV, errno.ENOSYS,
}))
def test_verified_fetch_succeeds_without_hard_link_support(ds, monkeypatch, code):
    from manyruns import datasetfetch

    def unsupported(source, target):
        assert Path(source).read_bytes() == PAYLOAD
        assert not Path(target).exists()
        raise OSError(code, "hard links unavailable")

    monkeypatch.setattr(datasetfetch.os, "link", unsupported)
    result = datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: io.BytesIO(PAYLOAD))
    assert result == destination()
    assert result.read_bytes() == PAYLOAD
    assert hashlib.sha256(result.read_bytes()).hexdigest() == ds["handle"]["sha256"]
    assert datasetfetch.fetch_dataset(ds) == result
    assert result.parent.is_dir()
    assert list(result.parent.iterdir()) == [result]


@pytest.mark.parametrize("mask", [0o022, 0o002, 0o077])
@pytest.mark.parametrize("publication", ["link", "rename"])
def test_fetched_sample_permissions_follow_umask(ds, mask, publication):
    """Change umask in a child only; the downloader must not mutate the process-wide mask."""
    from manyruns import datasetfetch

    script = """
import errno
import io
import json
import os
import stat
import sys
sys.path.insert(0, sys.argv[1])
from manyruns import datasetfetch

os.umask(int(sys.argv[2]))
def forbid_umask(*args):
    raise AssertionError("the downloader must not change the process-wide umask")
os.umask = forbid_umask
if sys.argv[3] == "rename":
    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "hard links unavailable")
    datasetfetch.os.link = unsupported

target = datasetfetch.fetch_dataset(
    json.loads(sys.argv[4]), opener=lambda *a, **k: io.BytesIO(sys.argv[5].encode()),
)
print(stat.S_IMODE(target.stat().st_mode))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(datasetfetch.__file__).resolve().parents[1]),
         str(mask), publication, json.dumps(ds), PAYLOAD.decode()],
        check=True, capture_output=True, text=True,
    )
    assert int(result.stdout) == 0o666 & ~mask
    assert destination().read_bytes() == PAYLOAD


@pytest.mark.parametrize("collision", ["file", "directory", "dangling_symlink"])
def test_fallback_publication_preserves_concurrent_placements(ds, monkeypatch, collision):
    from manyruns import datasetfetch

    publications = []

    def unsupported(source, target):
        publications.append(target)
        if collision == "file":
            target.write_bytes(b"user copy")
        elif collision == "directory":
            target.mkdir()
        else:
            target.symlink_to("absent-target")
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    def forbidden_replace(*args):
        pytest.fail("an existing user placement must prevent fallback publication")

    monkeypatch.setattr(datasetfetch.os, "link", unsupported)
    monkeypatch.setattr(datasetfetch.os, "replace", forbidden_replace)

    def fetch():
        return datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: io.BytesIO(PAYLOAD))

    if collision == "file":
        assert fetch() == destination()
        assert destination().read_bytes() == b"user copy"
    else:
        with pytest.raises(datasetfetch.DatasetFetchError, match="usable file"):
            fetch()
        assert destination().is_dir() if collision == "directory" else destination().is_symlink()
    assert publications == [destination()]
    assert destination().parent.is_dir()
    assert list(destination().parent.iterdir()) == [destination()]


@pytest.mark.parametrize("error", [OSError("rename failed"), KeyboardInterrupt()])
def test_interrupted_fallback_rename_leaves_no_destination_or_partial(ds, monkeypatch, error):
    from manyruns import datasetfetch

    def unsupported(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    def interrupted(source, target):
        assert Path(source).read_bytes() == PAYLOAD
        assert not Path(target).exists()
        raise error

    monkeypatch.setattr(datasetfetch.os, "link", unsupported)
    monkeypatch.setattr(datasetfetch.os, "replace", interrupted)
    expected = KeyboardInterrupt if isinstance(error, KeyboardInterrupt) else datasetfetch.DatasetFetchError
    with pytest.raises(expected) as raised:
        datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: io.BytesIO(PAYLOAD))
    if isinstance(error, OSError):
        for part in (str(error), "Retry", ds["handle"]["url"], str(destination())):
            assert part in str(raised.value)
    assert_empty_drop()


def test_fallback_readers_cannot_resolve_incomplete_publication(ds, monkeypatch):
    from manyruns import datasetfetch

    publishing, release = Event(), Event()
    open_path, replace = Path.open, datasetfetch.os.replace

    def pause():
        publishing.set()
        assert release.wait(5), "test did not release publication"

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "hard links unavailable")

    def paused_open(path, mode="r", *args, **kwargs):
        output = open_path(path, mode, *args, **kwargs)
        # Catch the old copy path at the point it first exposes an empty final file.
        if path == destination() and "x" in mode:
            pause()
        return output

    def paused_replace(source, target):
        assert Path(source).read_bytes() == PAYLOAD
        pause()
        replace(source, target)

    monkeypatch.setattr(datasetfetch.os, "link", unsupported)
    monkeypatch.setattr(Path, "open", paused_open)
    monkeypatch.setattr(datasetfetch.os, "replace", paused_replace)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(datasetfetch.fetch_dataset, ds,
                             opener=lambda *a, **k: io.BytesIO(PAYLOAD))
        try:
            assert publishing.wait(5), "fetch never reached publication"
            assert shell.dataset_ref_path(ds["handle"]["ref"]) is None
            assert not destination().exists()
        finally:
            release.set()
        assert future.result(timeout=5) == destination()
    assert hashlib.sha256(destination().read_bytes()).hexdigest() == ds["handle"]["sha256"]
    assert list(destination().parent.iterdir()) == [destination()]


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_killed_fallback_publisher_leaves_only_complete_final_file(ds, boundary):
    """SIGKILL bypasses finally: before rename only a hidden verified partial may survive."""
    from manyruns import datasetfetch

    script = """
import errno
import io
import json
import os
from pathlib import Path
import signal
import sys
sys.path.insert(0, sys.argv[1])
from manyruns import datasetfetch, shell

def unsupported(*args):
    raise OSError(errno.ENOTSUP, "hard links unavailable")

def kill():
    os.kill(os.getpid(), signal.SIGKILL)

open_path, replace = Path.open, os.replace
def killed_open(path, mode="r", *args, **kwargs):
    output = open_path(path, mode, *args, **kwargs)
    if path == shell.data_dir() / "sample.h5ad" and "x" in mode:
        kill()
    return output

def killed_replace(source, target):
    if sys.argv[2] == "before":
        kill()
    replace(source, target)
    kill()

datasetfetch.os.link = unsupported
datasetfetch.os.replace = killed_replace
Path.open = killed_open
datasetfetch.fetch_dataset(
    json.loads(sys.argv[3]), opener=lambda *a, **k: io.BytesIO(sys.argv[4].encode()),
)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(datasetfetch.__file__).resolve().parents[1]),
         boundary, json.dumps(ds), PAYLOAD.decode()],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == -signal.SIGKILL, result.stderr
    root = destination().parent
    assert root.is_dir()
    if boundary == "before":
        assert shell.dataset_ref_path(ds["handle"]["ref"]) is None
        assert not destination().exists()
        partials = list(root.iterdir())
        assert len(partials) == 1
        assert partials[0].name.startswith(".") and partials[0].suffix == ".part"
        assert partials[0].read_bytes() == PAYLOAD
    else:
        assert datasetfetch.fetch_dataset(ds) == destination()  # no transport on reuse
        assert hashlib.sha256(destination().read_bytes()).hexdigest() == ds["handle"]["sha256"]
        assert list(root.iterdir()) == [destination()]


@pytest.mark.parametrize(("payload", "reason"), [
    (b"x" * len(PAYLOAD), "sha256"), (PAYLOAD[:-1], "bytes"), (PAYLOAD + b"x", "bytes"),
])
def test_failed_verification_removes_only_the_partial(ds, payload, reason):
    from manyruns import datasetfetch

    response = io.BytesIO(payload)
    with pytest.raises(datasetfetch.DatasetFetchError, match=reason):
        datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: response)
    assert response.closed
    assert not destination().exists()
    assert_empty_drop()


def test_a_directory_collision_is_refused_without_network(ds):
    from manyruns import datasetfetch

    destination().mkdir(parents=True)
    with pytest.raises(datasetfetch.DatasetFetchError, match="file"):
        datasetfetch.fetch_dataset(ds)
    assert destination().is_dir()


def test_cancellation_closes_response_and_cleans_partial(ds):
    from manyruns import datasetfetch

    response = io.BytesIO(PAYLOAD)
    cancel = False

    def progress(done, total):
        nonlocal cancel
        cancel = done > 0

    with pytest.raises(datasetfetch.DatasetFetchError, match="cancel"):
        datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: response,
                                   progress=progress, cancelled=lambda: cancel)
    assert response.closed
    assert_empty_drop()


@pytest.mark.parametrize("error", [
    OSError("interrupted read"), TimeoutError("read timed out"),
    IncompleteRead(b"partial", 100),
])
def test_read_failures_close_and_clean_up(ds, error):
    from manyruns import datasetfetch

    class BrokenStream(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise error
            return super().read(2)

    response = BrokenStream(PAYLOAD)
    with pytest.raises(datasetfetch.DatasetFetchError) as raised:
        datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: response)
    assert str(error) in str(raised.value)
    assert response.closed
    assert_empty_drop()


@pytest.mark.parametrize("error", [
    TimeoutError("connection timed out"), URLError("offline"),
    HTTPError("https://example.org/sample.h5ad", 503, "unavailable", None, None),
    UnicodeEncodeError("ascii", "é", 0, 1, "URL cannot be encoded"),
    UnicodeError("label empty or too long"), ValueError("invalid transport URL"),
])
def test_transport_refusals_include_manual_recovery(ds, error):
    from manyruns import datasetfetch

    def opener(*args, **kwargs):
        raise error

    with pytest.raises(datasetfetch.DatasetFetchError) as raised:
        datasetfetch.fetch_dataset(ds, opener=opener)
    message = str(raised.value)
    for part in ("sample.h5ad", ds["handle"]["url"], str(destination()), "Retry"):
        assert part in message
    assert_empty_drop()


@pytest.mark.parametrize("url", [
    "https://example.org/données.h5ad", "https://exa\u200bmple.org/file",
    "https://example..org/file", "https://" + "a" * 64 + ".org/file",
])
def test_invalid_url_is_not_fetchable_or_sent_to_transport(ds, url):
    from manyruns import datasetfetch

    ds["handle"]["url"] = url
    assert not datasetfetch.can_fetch(ds)
    with pytest.raises(datasetfetch.DatasetFetchError, match="no validated download URL") as raised:
        datasetfetch.fetch_dataset(ds)
    assert url in str(raised.value)
    assert str(destination()) in str(raised.value)
    assert not shell.data_dir().exists()


@pytest.mark.parametrize("stage", ["mkdir", "tempfile", "publication"])
def test_filesystem_failures_are_actionable_and_leave_no_partial(ds, monkeypatch, stage):
    from manyruns import datasetfetch

    def denied(*args, **kwargs):
        raise PermissionError(f"{stage} denied")

    if stage == "mkdir":
        monkeypatch.setattr(shell, "ensure_drop_folder", denied)
    elif stage == "tempfile":
        monkeypatch.setattr(datasetfetch, "_open_partial", denied)
    else:
        monkeypatch.setattr(datasetfetch.os, "link", denied)
    response = io.BytesIO(PAYLOAD)
    with pytest.raises(datasetfetch.DatasetFetchError, match=f"{stage} denied"):
        datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: response)
    assert not destination().exists()
    if stage == "mkdir":
        assert not shell.data_dir().exists()
    else:
        assert_empty_drop()
    if stage == "publication":
        assert response.closed


@pytest.mark.parametrize("publication_fails", [True, False])
def test_cleanup_failure_reports_remaining_partial_and_allows_retry(ds, monkeypatch,
                                                                  publication_fails):
    from manyruns import datasetfetch

    response = io.BytesIO(PAYLOAD)
    leftovers = []
    unlink = Path.unlink

    def deny_publication(*args, **kwargs):
        raise PermissionError("publication denied")

    def deny_cleanup(path, *args, **kwargs):
        if path.parent == destination().parent and path.suffix == ".part":
            leftovers.append(path)
            raise PermissionError("cleanup denied")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as faults:
        if publication_fails:
            faults.setattr(datasetfetch.os, "link", deny_publication)
        faults.setattr(Path, "unlink", deny_cleanup)
        with pytest.raises(datasetfetch.DatasetFetchError) as raised:
            datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: response)

    assert response.closed
    assert len(leftovers) == 1
    partial = leftovers[0]
    assert partial.read_bytes() == PAYLOAD
    message = str(raised.value)
    assert str(partial) in message
    assert "cleanup denied" in message
    assert "remove" in message.lower()
    if publication_fails:
        assert not destination().exists()
        for part in ("publication denied", "Retry", ds["handle"]["url"], str(destination()),
                     str(len(PAYLOAD)), ds["handle"]["sha256"]):
            assert part in message
    else:
        assert destination().read_bytes() == PAYLOAD

    # Restoring write access permits a bounded retry; a leaked lock would cancel it.
    deadline = monotonic() + 2
    assert datasetfetch.fetch_dataset(
        ds, opener=lambda *a, **k: io.BytesIO(PAYLOAD),
        cancelled=lambda: monotonic() >= deadline,
    ) == destination()
    assert destination().read_bytes() == PAYLOAD
    assert partial.read_bytes() == PAYLOAD, "retry must not remove another attempt's partial"
    root = shell.data_dir()
    assert root.is_dir()
    assert set(root.iterdir()) == {destination(), partial}


@pytest.mark.parametrize("collision", ["file", "directory", "dangling_symlink"])
def test_manual_placement_during_download_wins_without_being_overwritten(ds, collision):
    from manyruns import datasetfetch

    def progress(done, total):
        if done == total:
            if collision == "file":
                destination().write_bytes(b"user copy")
            elif collision == "directory":
                destination().mkdir()
            else:
                destination().symlink_to("absent-target")

    def fetch():
        return datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: io.BytesIO(PAYLOAD),
                                          progress=progress)

    if collision == "file":
        assert fetch() == destination()
        assert destination().read_bytes() == b"user copy"
    else:
        with pytest.raises(datasetfetch.DatasetFetchError, match="usable file"):
            fetch()
        assert destination().is_dir() if collision == "directory" else destination().is_symlink()
    root = shell.data_dir()
    assert root.is_dir()
    assert list(root.iterdir()) == [destination()]


@pytest.mark.parametrize("collision", ["file", "directory", "dangling_symlink"])
def test_manual_placement_at_publication_is_never_overwritten(ds, monkeypatch, collision):
    """Create the competing entry after all checks, at the filesystem operation itself."""
    from manyruns import datasetfetch

    publications = []

    def intercept(publish):
        def competing_publish(source, target, *args, **kwargs):
            assert Path(source).read_bytes() == PAYLOAD
            assert Path(target) == destination()
            publications.append(target)
            if collision == "file":
                destination().write_bytes(b"user copy")
            elif collision == "directory":
                destination().mkdir()
            else:
                destination().symlink_to("absent-target")
            return publish(source, target, *args, **kwargs)
        return competing_publish

    # Exercise the same race with either replacing or non-replacing publication.
    for operation in ("replace", "link"):
        monkeypatch.setattr(datasetfetch.os, operation,
                            intercept(getattr(datasetfetch.os, operation)))

    def fetch():
        return datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: io.BytesIO(PAYLOAD))

    if collision == "file":
        assert fetch() == destination()
        assert destination().read_bytes() == b"user copy"
    else:
        with pytest.raises(datasetfetch.DatasetFetchError, match="usable file"):
            fetch()
        assert destination().is_dir() if collision == "directory" else destination().is_symlink()
    assert publications == [destination()]
    root = shell.data_dir()
    assert root.is_dir()
    assert list(root.iterdir()) == [destination()]


def test_a_literal_local_copy_keeps_precedence_over_the_drop_folder(ds):
    """Preservation: literal ref wins and noncanonical local bytes remain user-owned."""
    from manyruns import datasetfetch

    literal = Path(ds["handle"]["ref"])
    literal.parent.mkdir()
    literal.write_bytes(b"literal data")
    destination().parent.mkdir(parents=True)
    destination().write_bytes(b"drop data")
    assert datasetfetch.fetch_dataset(ds) == literal.resolve()
    assert literal.read_bytes() == b"literal data"
    assert destination().read_bytes() == b"drop data"


def test_failed_fetch_leaves_other_attempts_and_user_files_alone(ds):
    from manyruns import datasetfetch

    root = shell.ensure_drop_folder()
    other = root / ".another-attempt.part"
    other.write_bytes(b"somebody else's partial")
    with pytest.raises(datasetfetch.DatasetFetchError, match="bytes"):
        datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: io.BytesIO(b""))
    assert other.read_bytes() == b"somebody else's partial"
    assert root.is_dir()
    assert list(root.iterdir()) == [other]


def test_declared_size_is_authoritative_even_when_http_headers_disagree(ds):
    from manyruns import datasetfetch

    response = io.BytesIO(PAYLOAD)
    response.headers = {"Content-Length": "999999"}
    assert datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: response).read_bytes() == PAYLOAD


def test_oversize_stream_is_refused_without_reading_to_eof(ds):
    from manyruns import datasetfetch

    class Oversize(io.BytesIO):
        def read(self, size=-1):
            assert self.tell() == 0, "oversize must stop immediately, before another read"
            return super().read(size)

    response = Oversize(PAYLOAD + b"excess" * 100)
    with pytest.raises(datasetfetch.DatasetFetchError, match="bytes"):
        datasetfetch.fetch_dataset(ds, opener=lambda *a, **k: response)
    assert response.closed
    assert_empty_drop()


def test_local_url_override_and_runtime_opener_injection(ds, tmp_path, monkeypatch):
    from manyruns import datasetfetch

    local = tmp_path / "transport.bin"
    local.write_bytes(PAYLOAD)
    calls = []

    def opener(url, *, timeout):
        assert url == local.as_uri()
        calls.append(url)
        return local.open("rb")

    monkeypatch.setattr("urllib.request.urlopen", opener)
    assert datasetfetch.fetch_dataset(ds, url=local.as_uri()).read_bytes() == PAYLOAD
    assert calls == [local.as_uri()]


def test_same_destination_requests_share_one_download(ds):
    from manyruns import datasetfetch

    in_transport, release, waiting = Event(), Event(), Event()
    calls = []

    def opener(url, *, timeout):
        calls.append(url)
        in_transport.set()
        assert release.wait(5), "test did not release the transport"
        return io.BytesIO(PAYLOAD)

    def waiter_cancelled():
        waiting.set()
        return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(datasetfetch.fetch_dataset, ds, opener=opener)
        try:
            assert in_transport.wait(5)
            second = pool.submit(datasetfetch.fetch_dataset, ds, opener=opener,
                                 cancelled=waiter_cancelled)
            assert waiting.wait(5)
            with pytest.raises(FutureTimeout):
                second.result(timeout=0.1)
            assert len(calls) == 1, "a second transport started for the same destination"
        finally:
            release.set()
        assert first.result(timeout=5) == second.result(timeout=5) == destination()
    assert len(calls) == 1
    assert destination().read_bytes() == PAYLOAD
    assert shell.data_dir().is_dir()
    assert list(shell.data_dir().iterdir()) == [destination()]


def test_a_waiting_request_can_cancel_without_disturbing_the_active_download(ds):
    from manyruns import datasetfetch

    entered, release, cancel, waiting = Event(), Event(), Event(), Event()

    def opener(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return io.BytesIO(PAYLOAD)

    def waiter_cancelled():
        waiting.set()
        return cancel.is_set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(datasetfetch.fetch_dataset, ds, opener=opener)
        try:
            assert entered.wait(5)
            second = pool.submit(datasetfetch.fetch_dataset, ds, cancelled=waiter_cancelled)
            assert waiting.wait(5)
            cancel.set()
            with pytest.raises(datasetfetch.DatasetFetchError, match="cancel"):
                second.result(timeout=5)
            assert not destination().exists()
        finally:
            release.set()
        assert first.result(timeout=5).read_bytes() == PAYLOAD


@pytest.mark.parametrize("handle", [
    {}, {"kind": "manylatents", "ref": "swissroll"},
    {"kind": "path", "ref": "synthetic:time-course"},
    {"kind": "path", "ref": "/absolute/sample.h5ad"},
    {"kind": "path", "ref": "../sample.h5ad"},
])
def test_can_fetch_only_accepts_valid_relative_file_contracts(ds, handle):
    from manyruns import datasetfetch

    assert datasetfetch.can_fetch(ds)
    ds["handle"].update(handle)
    if not handle:
        del ds["handle"]["sha256"]
    assert not datasetfetch.can_fetch(ds)
    with pytest.raises(datasetfetch.DatasetFetchError):
        datasetfetch.fetch_dataset(ds)
