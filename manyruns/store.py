"""The run store — one append-only line per run.

Product surfaces share one ledger per working directory at ``outputs/index.jsonl``;
artifacts live under ``outputs/<project>/``. Paths are relative to the process's current
working directory, not the source data or project file. All four writers (batch, REPL,
shell and TUI) use this default, as does ``read()``. Library callers may explicitly supply
``out_dir`` to both append and read; existing rows are never relocated.

Decision records have separate scopes: TUI menu choices use ``outputs/decisions.jsonl``;
tuning choices use the session's output directory (``outputs/<project>/`` on product
surfaces, falling back to ``outputs/`` for a session with no directory). TUI tuning
readers and corpus counters use that same directory; ``decisions.read/examples`` default
to ``outputs/`` and also accept an explicit directory. These are all cwd-relative unless
the caller supplies an absolute path. Opening a project does not change cwd.

Version 1 row shape (core top-level keys): ``schema_version`` (written first),
``run_id``, ``spec_id``, ``parent``, ``dataset``, ``dataset_name``, ``data_kwargs``, ``at``,
``recipe``, ``engine``, ``seed``, ``tool_version``, ``ok``, ``complete``, ``caveats``,
``non_commercial_steps``, ``steps`` and ``g_vector``. ``non_commercial_steps`` is the sorted
list of the run's non-commercial steps, whatever their outcome and engine: a step whose name is
in ``commercial.NON_COMMERCIAL_STEPS`` (an in-process ``mioflow`` step, for example), or whose
tool manifest says ``commercial_use: false``; ``[]`` means none. A row written before that
field existed has no such key, and a missing key means unknown. Product callers also supply the core top-level ``source``,
``modality`` and ``decision_id`` fields through the existing ``append(extra=...)`` merge.
These stay at the top level; absent optional core fields retain their existing meaning.
``run``, ``init --batch``, ``explore`` and ``run --choice`` also write the core ``time_key`` and
``condition_key``: the metadata column whose values the run used as its time labels or its
condition labels (a given ``--time-key``/``--condition-key`` or choice key, else the column
found by name). At most one is a name. Both are ``null`` when the run read no such column: a
``mock`` run, a dataset that the engine generates, or time labels taken from sub-folder names.
A row without these keys was written before they existed, or by another writer; a missing key
means unknown.
Missing ``schema_version`` means 1. Readers preserve unknown keys and explicit versions.

Reserve nested ``extra.<caller>`` for FUTURE downstream fields only, e.g.
``append(results, extra={"extra": {"my_tool": {"annotation": "..."}}})``.
The ``extra=`` argument remains a top-level merge, not a new envelope around core fields.
Readers return ordinary dictionaries that can be serialized without dropping unknown keys;
``append`` projects execution results into new records, rather than rewriting stored rows.

**Verdict rows.** `manyruns claim` (`manyruns.verdict`) appends a structure-class prediction
about a stored run as a row of this file, nested under ``extra.verdict``. A verdict row copies
its run's identity fields, so it SHARES that run's ``run_id``: the file is no longer one line
per run, and ``run_id`` is not unique per row. It has ``ok`` and ``complete`` null and empty
``steps`` and ``g_vector``. A reader that counts runs skips every row for which
``score.verdict_of(row)`` is not ``None``.

**Blinding.** With ``MANYRUNS_BLIND_DATASET`` on (`manyruns.blind`), ``append`` writes a
blinded name in ``dataset_name`` and in the true-name occurrences of ``dataset``, ``source``
and the string leaves of ``data_kwargs``, and records the true name in
``outputs/blind_names.jsonl``. It does NOT rewrite ``steps``, ``caveats``, ``recipe`` or
``g_vector``: step records hold artifact paths that ``artifacts.complete()`` reads back, so a
step path can still carry the true name. Off, ``append`` writes exactly what it wrote before.

The store was deferred until it had a reader: *"blocked on having a reader. Write
`index.jsonl` when the scorer or a stopping rule needs it, not before."* That condition is
now met — the panel reads run records, and the admission gate produces verdicts worth
keeping — so the store earns its place. It stays deliberately smaller than the cut proposal:
no content-addressing, no five-store join, no adoption of stray artifacts.

Append-only and one line per run, because the two things a store like this gets wrong are
losing history to an overwrite and holding a lock. `O_APPEND` on a single `write` of a line
shorter than PIPE_BUF is atomic between processes, which matters because the sweep runs
cells in a process pool and they all write here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional

from manyruns import blind as _blind


def index_path(out_dir: Path | str = "outputs") -> Path:
    return Path(out_dir) / "index.jsonl"


def _jsonable(value: Any) -> Any:
    """Records already hold descriptions rather than payloads (see `watch.describe`), so
    this only has to survive the occasional stray object — never a matrix."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _leaves(value: Any, names: list[str]) -> Any:
    if isinstance(value, str):
        return _blind.blind_text(value, names)
    if isinstance(value, dict):
        return {k: _leaves(v, names) for k, v in value.items()}
    if isinstance(value, list):
        return [_leaves(v, names) for v in value]
    return value


def _blind_row(row: dict, out_dir: Path | str) -> None:
    """`MANYRUNS_BLIND_DATASET`: the true name out of `dataset_name`, `dataset`, `source` and
    the string leaves of `data_kwargs`, and into the side file. `steps` is NOT rewritten (see
    `manyruns.blind`)."""
    true = row.get("dataset_name")
    if not isinstance(true, str) or not true:
        return
    row["dataset_name"] = _blind.remember(true, out_dir=out_dir)
    names = [true]
    for key in ("dataset", "source"):
        if isinstance(row.get(key), str):
            row[key] = _blind.blind_text(row[key], names)
    row["data_kwargs"] = _leaves(row.get("data_kwargs"), names)


def append(results: dict, *, out_dir: Path | str = "outputs",
           extra: Optional[dict] = None) -> Path:
    """Append one run. Returns the index path.

    Writes the RECORD, not the g-vector's payload: recipe, engine, seed, tool version, and
    the per-step account of what ran and what it produced. That is what a later reader — a
    scorer, a stopping rule, `manyruns stale` — actually needs, and it stays small enough
    that the file is greppable after a thousand runs.

    ``extra=`` still merges top-level fields, including core source/modality/decision_id.
    Only future downstream additions belong in a nested ``extra.<caller>`` namespace.
    The schema version is owned by this writer, not by the caller."""
    path = index_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1,
        # identity first — a row without it cannot be referred to, and every question that
        # is an EDGE between runs is inexpressible rather than merely unbuilt
        "run_id": results.get("run_id"),
        "spec_id": results.get("spec_id"),
        # The ONE schema addition the interactive loop needs, and it is None for every row
        # written before it existed: `run_id` names one STATE LINEAGE, so a person who
        # rewinds to an earlier embedding and takes a different next step is writing a SECOND
        # row, not extending the first — `{"run_id", "index"}` is the edge back to the state
        # it started from. An atomic run is the degenerate lineage (no parent), which is why
        # every row already on disk still means exactly what it meant.
        #
        # Both questions a comparison reader has to ask are now answerable from the row alone:
        # *same analysis?* is `spec_id`, *same starting state?* is `parent`. Neither implies
        # the other, which is why they are two columns and `parent` is not hashed into
        # `spec_id` (see `runner._finalize`).
        "parent": results.get("parent"),
        "dataset": results.get("dataset"),
        "dataset_name": results.get("dataset_name"),
        "data_kwargs": _jsonable(results.get("data_kwargs") or {}),
        "at": results.get("at"),
        "recipe": results.get("recipe"),
        "engine": results.get("engine"),
        "seed": results.get("seed"),
        "tool_version": results.get("tool_version"),
        "ok": results.get("ok"),
        # Alongside `ok`, not folded into it. `ok` is "nothing errored"; this is "every declared
        # step ran". They were ONE field until manyruns#66, which conflated a run that failed
        # with a run where a step correctly stood down — and `index.jsonl` is the record the
        # learner reads, so a row that cannot tell those apart makes the distinction
        # unlearnable. Absent on rows written before the split; a reader must treat a missing
        # key as unknown rather than as True.
        "complete": results.get("complete"),
        # What the run says about ITSELF — an unusual composition, or an engine that invented
        # its numbers. The row schema never had this column, so a `mock` lineage landed in the
        # history indistinguishable from a measured one, and a `test_front_door` docstring
        # claiming the mock caveat "rides into index.jsonl" was inaccurate the day it was
        # written. It matters more now than it did: the interactive doors call `append`, so
        # mock runs reach the store rather than only the screen.
        "caveats": _jsonable(results.get("caveats") or []),
        "non_commercial_steps": _jsonable(results.get("non_commercial_steps") or []),
        "steps": _jsonable(results.get("steps") or []),
        "g_vector": _jsonable(results.get("g_vector") or {}),
        **(_jsonable(extra or {})),
    }
    row["schema_version"] = 1
    if _blind.enabled():
        _blind_row(row, out_dir)
    line = json.dumps(row, separators=(",", ":")) + "\n"
    # one atomic append; no read-modify-write, so concurrent pool workers cannot interleave
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)
    return path


def read(out_dir: Path | str = "outputs") -> Iterator[dict]:
    """Every run, oldest first. A corrupt line is skipped rather than fatal — a half-written
    row from a killed process must not make the whole history unreadable. Missing versions
    mean version 1; unknown keys and explicit versions are preserved for JSON round trips."""
    path = index_path(out_dir)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            row.setdefault("schema_version", 1)
            yield row
        except json.JSONDecodeError:
            continue
