"""The trace sink — one JSON line per event, and by default nowhere at all.

An EVENT is a moment INSIDE a run that a later reader would want. Today there is exactly one
kind, `ask`: the question a person typed at the controller and the answer a local model gave.
The `index.jsonl` row is by contrast the run's RESULT, written once, after it is over — an
event is written while the run is still going, and there can be many.

**Not `results["trace"]`.** That one is the step chain (`latent:phate → lightning:mioflow`), a
field on a run row, and the collision of names is why this docstring opens with the distinction
instead of leaving a reader to find it.

**Tracing is a sink, not a mode.** The default is `NullTracer`, whose `emit` does nothing, so an
emit site is written unconditionally: there is no `if tracing:` anywhere, no second code path to
keep working, and no way for the traced app and the shipped app to drift. Switching it on is
`MANYRUNS_TRACE=jsonl://~/traces`; switching it off is unsetting that.

**A sink may only ever change a duration, never an answer.** Nothing in this file raises. An
unparseable setting reads as off, an unwritable directory drops the event, a value `json` cannot
hold is stringified. A typo in a shell profile that crashed the app — or, worse, changed what the
app said — would make the record this exists to keep untrustworthy. That is `decisions.append`'s
contract (it returns `None` rather than raise, for the same reason), one layer further out.

**`agent_id` enters the package here.** Measured immediately before this file: zero files in
`manyruns/` carried the word. Every event names both the run and WHO caused it from the first
one, used or not — a corpus that discovers later that some of its rows were a person and some
were a driver cannot go back and label the ones already written.

**What is deliberately absent, so that it is not added by accident**: a SQLite tracer,
an index command, any reader over the sink, `span()`, and — the one worth stating loudest —
any emit site at all.
Nothing imports this module yet. The trace layer grows by adding EMITTERS; this file is finished
when the wire format is.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from manyruns import env as _env

#: The one URL scheme `current()` understands. A SCHEME rather than a bare directory because the
#: setting names a sink and not a path: the next tracer's value is `sqlite://…`, and a machine
#: configured for that one must read as "off" here rather than write JSONL into a directory
#: literally named `sqlite:`.
SCHEME = "jsonl://"

#: Where an event whose `run_id` is missing or unusable lands. Dropping it instead would make the
#: sink lossy in precisely the case somebody would be investigating, and a file named for the
#: absence is as greppable as one named for a run.
UNKNOWN_RUN = "unknown-run"

#: Who an event says caused it when the caller does not say. The other value this field will ever
#: take is a driver's, and the environment cannot tell which it is being used by — that is the
#: environment contract, not an omission — so the default is the person and a caller that knows
#: better passes `agent_id=`.
HUMAN = "human"


@runtime_checkable
class Tracer(Protocol):
    """The whole contract, and callers depend on this rather than on a sink.

    One method, because an event is fire-and-forget: no handle, no close, no flush. Anything a
    caller could do with a return value — retry, report, count — is a caller deciding whether
    tracing worked, which is the coupling this file exists to not have.
    """

    def emit(self, event: dict) -> None: ...


class NullTracer:
    """THE DEFAULT. Every call site can emit unconditionally because of this class.

    Not a debug convenience: it is what makes "tracing is a sink, not a mode" true in the code
    rather than in the docstring. An emit site guarded by a flag would mean the shipped path and
    the traced path are two paths, and the one nobody runs is the one that breaks.
    """

    def emit(self, event: dict) -> None:
        """Nothing, deliberately, and it is not an error — see the class docstring."""


class JsonlTracer:
    """One file per run: `<directory>/<run_id>.jsonl`, appended to, never rewritten.

    PER RUN rather than one file for everything, because the reader that matters first is a
    person with `cat` and the question is always about one run. The index that joins them across
    runs is sub-project 2 and reads these files; it does not need them pre-joined.

    The directory is created on the first `emit`, not here, for two reasons: `current()` is
    consulted to decide whether to bother building a payload, so constructing a tracer must cost
    nothing and touch nothing; and a session that traces nothing then leaves nothing behind — an
    empty directory that appears because a variable is exported reads as a bug in the sink.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def emit(self, event: dict) -> None:
        """Append one line. Failures are dropped, never raised — see the module docstring.

        Hold the file lock across every write and any rollback. A stub that writes seven bytes
        then raises leaves half an event; the next append would join onto it. PIPE_BUF only
        promises atomicity for pipes, not these regular files, so retries need the same lock.

        `default=str` rather than a `_jsonable` walk (`store.py:26`): an event is a handful of
        scalars the emitter already chose, not a record built from a result dict, so the only
        thing this has to survive is the occasional `Path` or `datetime` — and a stringified
        value in the corpus is worth more than a dropped event.
        """
        try:
            line = json.dumps(dict(event or {}), separators=(",", ":"), default=str) + "\n"
        except (TypeError, ValueError):        # a cycle; nothing str() can be talked into
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.directory / self._filename(event),
                         os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                start = os.lseek(fd, 0, os.SEEK_END)
                # A prior crash or failed rollback can leave a tail we cannot safely extend.
                # Measured with an unterminated fixture: refusing preserves its bytes and keeps
                # the next event from becoming part of the damaged line.
                if start and os.pread(fd, 1, start - 1) != b"\n":
                    return
                pending = memoryview(line.encode())
                try:
                    while pending:
                        written = os.write(fd, pending)
                        if written <= 0:
                            raise OSError("trace write made no progress")
                        pending = pending[written:]
                except OSError:
                    os.ftruncate(fd, start)
                    raise
            finally:
                os.close(fd)
        except OSError:                        # symlink, unwritable, out of space, gone
            return

    @staticmethod
    def _filename(event: dict) -> str:
        """`<run_id>.jsonl`, with the id reduced to what is safe to put in a path.

        A `run_id` is minted as `uuid4().hex[:12]` (`pipeline/runner.py:883`) and so is already
        twelve hex characters — but this class writes whatever the EVENT carries, and a sink that
        can be pointed at `../../.zshrc` by a field is a sink that can do more than change a
        duration. The filter is three characters wide and closes it.
        """
        run = str((event or {}).get("run_id") or "")
        return (("".join(c for c in run if c.isalnum() or c in "-_")) or UNKNOWN_RUN) + ".jsonl"


def stamp(event: dict, *, run_id: str | None, agent_id: str = HUMAN) -> dict:
    """The envelope every event carries: `kind`, `at`, `run_id`, `agent_id`, then the payload.

    Returns a NEW dict; the caller's is not mutated, so an emitter may keep and reuse the
    payload it built.

    The envelope wins over the payload on its own four keys, and `at` is this clock's rather
    than the caller's: events from one run are read in order, and two callers stamping their own
    times is how a record becomes unorderable. `kind` is the exception — it is the one envelope
    field only the emitter knows — and an event that arrives without one is named rather than
    refused, because a reader dispatching on `kind` would otherwise meet a `KeyError` from a line
    that is on disk and cannot be fixed.

    Seconds, no finer, and `+00:00` rather than the `Z` the spec's example line prints — both
    are ISO-8601 UTC and SQLite's `datetime()` reads either (checked, same instant out of both),
    so the tie goes to the two records already on disk: `decisions.append` and the run row both
    write `+00:00`, and an ingester joining a trace to a decision should not have to know which
    file it is reading to parse a time. The ordering that matters is the append order in the
    file, which is exact; `at` is for a person reading one line, and for bucketing by day.
    """
    out = {
        "kind": str((event or {}).get("kind") or "event"),
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "agent_id": agent_id,
    }
    out.update({k: v for k, v in (event or {}).items() if k not in out})
    return out


def current() -> Tracer:
    """The sink this process is configured for — `NullTracer` unless told otherwise.

    Read fresh on every call rather than cached in a module global: a test sets the variable per
    case, and a long-lived TUI would otherwise be stuck with whatever the environment said at
    import time. The read costs a dict lookup.

    ANYTHING NOT UNDERSTOOD IS OFF, and this is the file's central promise, not leniency: unset,
    empty, a bare path, a scheme belonging to a tracer that does not exist yet, a typo — all of
    them are `NullTracer`, none of them raise. The person who mistyped this in a shell profile
    finds out because their trace directory is empty, not because the app they were running died
    on a line that has nothing to do with their work.

    There is already a near neighbour to mistype it as: `TRACEBACK` (`app.py:372`) is a
    different variable with a different job, and someone reaching for that one and stopping a
    few characters early lands on `1`, which is not a sink and therefore reads as off.
    """
    raw = (_env.get("TRACE") or "").strip()
    if not raw.startswith(SCHEME):
        return NullTracer()
    where = raw[len(SCHEME):].strip()
    if not where:
        return NullTracer()
    try:
        return JsonlTracer(Path(where).expanduser())
    except (OSError, RuntimeError, ValueError):   # `~nosuchuser`, a NUL byte, an unresolvable $HOME
        return NullTracer()
