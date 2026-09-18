"""UI-thread selection guard shared by roster and stale ledger/run-again selections.

Only the explicit Fetch gesture starts transport. Daemon threads post messages; they
never touch widgets, and application exit never joins a blocked network read.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from typing import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from manyruns import catalog, datasetfetch, shell
from manyruns.tui import dropwatch, state
from manyruns.tui.state import DataEntry


@dropwatch.row_guard
def _fetch_declaration(name: str) -> dict:
    """The declaration can change after resolving its row; guard this read-back too."""
    return catalog.load_dataset(name)


def _resolve(entry: DataEntry) -> DataEntry:
    if entry.kind == 'bundled':
        return state.catalog_entry(entry.name, observation=entry.observation,
                                   require_observation=True)
    if entry.path is None or not entry.path.exists():
        return replace(entry, missing=True, refusal=(
            f'{entry.path or entry.name} is not on this machine.',
            f'Put the file at {entry.path or shell.data_dir()} and rescan.',))
    changed = []
    resolved = state.local_entry(
        entry.path, observation=entry.observation, require_observation=True,
        on_changed=changed.append, previous=entry if entry.refusal else None)
    if resolved is None:
        return replace(entry, pending=bool(changed), refusal=(
            'Copying… wait for two stable observations, then select again.' if changed else
            f'Could not read {entry.path}. Rescan after copying ends.',))
    return resolved


def prepare_entry(app, entry: DataEntry, on_ready: Callable[[DataEntry], None], *,
                  on_done: Callable[[], None] | None = None) -> None:
    """Recheck local availability and deliver a fresh entry only when it can be used.

    ``on_done`` optionally ends a caller's selection transaction on success or decline.
    Generated refs resolve through the catalog without filesystem-existence requirements.

    Filesystem: os.stat/lstat/scandir/listdir/readlink/mkdir,
    Path.resolve/exists/is_file/is_dir, io.open reads and os.replace cache writes.
    """
    if getattr(app, '_samplefetch_active', False):
        return
    app._samplefetch_active = True

    def completion_failed(error):
        """Report a failed handoff. Filesystem primitives: none."""
        app.notify(f'Could not open the selected data: {error}. Rescan and select again.',
                   severity='error')

    @dropwatch.filesystem_guard(completion_failed)
    def finished(ready):
        """Release selection and hand off. Filesystem: caller's roster refresh may reach
        os.stat/lstat/scandir/listdir/readlink and Path.resolve/exists/is_file/is_dir.
        """
        app._samplefetch_active = False
        if on_done is not None:
            on_done()
        if ready is not None and app.is_running:
            on_ready(ready)

    def refuse(error):
        """Refuse without filesystem access; the selection remains dismissible."""
        current = replace(entry, downloadable=False, pending=False, refusal=(
            f'Could not read {entry.name}: {type(error).__name__}: {error}',))
        app.push_screen(SampleFetchScreen(current), finished)

    @dropwatch.filesystem_guard(refuse)
    def prepare():
        """Prepare the modal/handoff. Filesystem: os.stat/lstat/scandir/listdir/readlink/
        mkdir/replace, Path.resolve/exists/is_file/is_dir and io.open catalog/source/cache I/O.
        """
        ds = None
        current = entry if entry.pending else _resolve(entry)
        if (current.kind == 'bundled' and current.downloadable
                and current.missing and not current.pending):
            ds = _fetch_declaration(current.name)
        if current.pending:
            current = replace(current, refusal=(
                'Copying… wait for two stable observations, then select again.',))
        if not current.missing and not current.pending and not current.refusal:
            finished(current)
            return
        app.push_screen(SampleFetchScreen(current, ds), finished)

    prepare()


class SampleFetchScreen(ModalScreen[DataEntry | None]):
    BINDINGS = [Binding('escape', 'back', 'Back', priority=True),
                Binding('enter', 'fetch', 'Fetch', priority=True)]
    DEFAULT_CSS = """
    SampleFetchScreen { align: center middle; background: $background 70%; }
    SampleFetchScreen > Vertical {
        width: 80; max-width: 95%; max-height: 95%; height: auto;
        padding: 1 2; border: round $accent; overflow-y: auto;
    }
    SampleFetchScreen Static { height: auto; margin-bottom: 1; }
    SampleFetchScreen Horizontal { height: auto; }
    SampleFetchScreen Button { margin-right: 1; }
    """

    class Progress(Message):
        def __init__(self, done: int, total: int):
            super().__init__()
            self.done, self.total = done, total

    class Finished(Message):
        def __init__(self, error: str | None, path: Path | None = None):
            super().__init__()
            self.error, self.path = error, path

    def __init__(self, entry: DataEntry, ds: dict | None = None):
        super().__init__()
        self.entry = entry
        self.ds = ds if entry.missing and not entry.pending else None
        self._cancelled = threading.Event()
        self._dismissed = False
        self._fetching = False
        self._thread: threading.Thread | None = None
        self._watch = dropwatch.DropWatch()
        self._download_path: Path | None = None
        self._settle_timer = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f'Get {self.entry.name}' if self.ds else self.entry.name,
                         markup=False, id='sample-title')
            lines = list(self.entry.refusal)
            if self.ds:
                handle = self.ds['handle']
                try:
                    target = dropwatch.canonical(dropwatch.path_call(shell.data_dir)) / Path(handle['ref']).name
                    lines = [f"{Path(handle['ref']).name} · {handle['bytes']} bytes "
                             f"({state._human_bytes(handle['bytes'])})", handle['url'],
                             f'Destination: {target}', *lines]
                except (OSError, ValueError) as error:
                    lines = [*lines, f'Could not read the destination: {error}. Rescan and retry.']
                    self.ds = None
            yield Static('\n'.join(lines), markup=False, id='sample-details')
            yield Static('Choose Fetch to download and verify.' if self.ds else 'Back to choose other data.',
                         markup=False, id='sample-status')
            with Horizontal():
                if self.ds:
                    yield Button('Fetch', id='fetch-sample', variant='primary')
                yield Button('Back', id='back-sample')

    @dropwatch.filesystem_guard('_filesystem_failed')
    def on_mount(self) -> None:
        """Focus the available action. Filesystem primitives: none."""
        self.query_one('#fetch-sample' if self.ds else '#back-sample', Button).focus()

    @dropwatch.filesystem_guard('_filesystem_failed')
    def action_fetch(self) -> None:
        """Start a worker. Filesystem primitives: none on the UI thread; see _fetch."""
        # Enter while Back has focus retains the button's advertised meaning.
        if isinstance(self.focused, Button) and self.focused.id == 'back-sample':
            self.action_back()
            return
        if self.ds is None or self._fetching or self._dismissed or self._settle_timer:
            return
        self._fetching = True
        self.query_one('#fetch-sample', Button).disabled = True
        # Disabling Fetch must not move focus to Back and turn repeated Enter into cancel.
        self.set_focus(None)
        self.query_one('#sample-status', Static).update('Downloading… 0 bytes; checking size and sha256.')

        self._thread = threading.Thread(target=self._fetch, name='manyruns-sample-fetch', daemon=True)
        self._thread.start()

    @dropwatch.filesystem_guard('_fetch_failed')
    def _progress(self, done: int, total: int) -> None:
        """Worker progress callback. Filesystem primitives: none; posts a message only."""
        if not self._dismissed:
            self.post_message(self.Progress(done, total))

    @dropwatch.filesystem_guard('_fetch_failed')
    def _fetch(self) -> None:
        """Fetch on the worker under one filesystem guard.

        Filesystem: os.stat/lstat/readlink/mkdir/link/replace/unlink,
        Path.resolve/exists/is_file/is_dir, temporary-file open/write/flush/close.
        Transport is supplied by datasetfetch; tests always inject it.
        """
        try:
            path = datasetfetch.fetch_dataset(self.ds, progress=self._progress,
                                             cancelled=self._cancelled.is_set)
        except datasetfetch.DatasetFetchError as error:
            self._fetch_failed(error)
            return
        if not self._dismissed:
            self.post_message(self.Finished(None, path))

    def _fetch_failed(self, error: Exception) -> None:
        """Worker recovery. Filesystem primitives: none; no widget access."""
        if not self._dismissed:
            self.post_message(self.Finished(str(error)))

    @dropwatch.filesystem_guard('_filesystem_failed')
    def on_sample_fetch_screen_progress(self, message: Progress) -> None:
        """Paint byte progress. Filesystem primitives: none; widget update only."""
        if not self._dismissed:
            self.query_one('#sample-status', Static).update(
                f'Downloading… {message.done:,} / {message.total:,} bytes; checking size and sha256.')

    @dropwatch.filesystem_guard('_filesystem_failed')
    def on_sample_fetch_screen_finished(self, message: Finished) -> None:
        """Start settling or refuse. Filesystem: os.stat/lstat/scandir/listdir/readlink,
        Path.resolve/exists/is_file/is_dir and io.open for catalog/source reads.
        """
        if self._dismissed:
            return
        self._fetching = False
        if message.error is not None:
            self._show_error(message.error)
            return
        # A successful return may be an existing manual copy, even when transport ran.
        # Observe all returned files independently before permitting inspection.
        self._download_path = message.path
        self._settle_timer = self.set_interval(1, self._settle_download)
        self._settle_download()

    @dropwatch.filesystem_guard('_filesystem_failed')
    def _settle_download(self) -> None:
        """Settle then inspect; all filesystem failures leave Retry/Back available.

        Reaches os.stat/lstat/scandir/listdir/readlink, Path.resolve/exists/is_file/is_dir,
        and io.open for catalog/source reads through snapshot and state.
        """
        if self._dismissed:
            return
        current = dropwatch.snapshot()
        path = self._download_path
        if path is not None:
            key = dropwatch.canonical(path)
            if not any(p == key or p in key.parents for p in current.entries):
                signature = dropwatch.fingerprint(path)
                if signature is not None:
                    current.entries[key] = signature
                    current.references[dropwatch.absolute(path)] = key
        poll = self._watch.observe(current)
        if path is not None and state._held(path, poll.held):
            self.query_one('#sample-status', Static).update(
                'Copying… waiting for two stable observations before opening.')
            return
        ready = state.catalog_entry(self.entry.name, skip=poll.held, snapshot=current,
                                    on_changed=self._watch.invalidate)
        if ready.pending:
            return
        if ready.missing or ready.refusal:
            self._show_error('\n'.join(ready.refusal) or 'The downloaded data is no longer available.')
            return
        self._dismissed = True
        self._stop_settling()
        self.dismiss(ready)

    def _stop_settling(self) -> None:
        """Stop the timer. Filesystem primitives: none."""
        if self._settle_timer is not None:
            self._settle_timer.stop()
            self._settle_timer = None

    def _show_error(self, error: str) -> None:
        """Show recovery actions. Filesystem primitives: none."""
        self._stop_settling()
        self.query_one('#sample-status', Static).update(error)
        buttons = self.query('#fetch-sample')
        if buttons:
            button = buttons.first(Button)
            button.disabled = False
            button.label = 'Retry'
            button.focus()
        else:
            self.query_one('#back-sample', Button).focus()

    def _filesystem_failed(self, error: Exception) -> None:
        """Refuse an uncertain source without filesystem reads or writes."""
        self._fetching = False
        self._watch.observe(dropwatch.Snapshot((), {}, ()))
        if not self._dismissed:
            self._show_error(f'Could not read the downloaded data: {error}. Retry or go Back.')

    @dropwatch.filesystem_guard('_filesystem_failed')
    def action_back(self) -> None:
        """Cancel and dismiss. Filesystem primitives: none; worker owns partial cleanup."""
        if self._dismissed:
            return
        self._dismissed = True
        self._cancelled.set()
        self._stop_settling()
        self.dismiss(None)

    @dropwatch.filesystem_guard('_filesystem_failed')
    def on_button_pressed(self, message: Button.Pressed) -> None:
        """Dispatch an action. Filesystem primitives: none; fetch runs on its worker."""
        if message.button.id == 'fetch-sample':
            self.action_fetch()
        else:
            self.action_back()

    @dropwatch.filesystem_guard('_filesystem_failed')
    def on_unmount(self) -> None:
        """Cancel late updates. Filesystem primitives: none; worker owns partial cleanup."""
        self._dismissed = True
        self._cancelled.set()
        self._stop_settling()
