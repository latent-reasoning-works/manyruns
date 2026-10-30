"""Blinded dataset names — hide a true dataset name from the model and from the run record.

``MANYRUNS_BLIND_DATASET`` switches blinding on. It is read only through `env.get`. Any value
other than unset, ``""``, ``0``, ``false``, ``no`` or ``off`` (stripped, any case) is on.

A **blinded name** is ``"ds-"`` plus the first 12 hex characters of the SHA-256 of the true
name. It has no salt: the control is against a model that recognizes a dataset by name, not
against a person who wants to invert the hash.

Blinding is applied only where a run or an observation LEAVES the process, never in memory:

  store.append                 dataset_name, dataset, source, string leaves of data_kwargs
  narrate.record_for_prompt    the ``dataset:`` line and every other occurrence of the name
  tui.state.DataEntry.to_dict  name, dataset, path and obs.source

`narrate.Observation.source` is not changed, because the shell loads the dataset from it.
`runner.identity` is not changed either, so a blinded run and a clear run of one spec share a
`spec_id`. `store.append` does not rewrite `steps`: step records hold artifact paths that
`artifacts.complete()` reads back, and the default project folder is named after the data
source, so those paths can still carry the true name. That is a known limit.

The **side file** ``outputs/blind_names.jsonl`` keeps one row per blinded name with its true
name, so a scorer can join verdict rows to truth rows after the run (`unblind_rows`). It is
written by `remember`, which `store.append` calls when blinding is on.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from manyruns import env

_OFF = ("", "0", "false", "no", "off")
_BLINDED = re.compile(r"ds-[0-9a-f]{12}")


def enabled() -> bool:
    """True when `MANYRUNS_BLIND_DATASET` is set to a value that is not an off word."""
    value = env.get("BLIND_DATASET")
    return value is not None and value.strip().lower() not in _OFF


def blind_name(name: str) -> str:
    """`"ds-" + sha256(name)[:12]`; a name that is already blinded comes back unchanged."""
    if _BLINDED.fullmatch(name):
        return name
    return "ds-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def blind_text(text: str, names: Iterable[str]) -> str:
    """`text` with each occurrence of each true name replaced by its blinded name.

    Longer names first, so a name that contains another is replaced whole."""
    for name in sorted({n for n in names if isinstance(n, str) and n}, key=len, reverse=True):
        if not _BLINDED.fullmatch(name):
            text = text.replace(name, blind_name(name))
    return text


def side_path(out_dir: Path | str = "outputs") -> Path:
    return Path(out_dir) / "blind_names.jsonl"


def _side_rows(out_dir: Path | str) -> Iterable[dict]:
    path = side_path(out_dir)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row


def remember(name: str, *, out_dir: Path | str = "outputs") -> str:
    """Return `blind_name(name)`, and record it in the side file the first time it is seen.

    Nothing is written for a name that is already blinded, or for one the file already holds.
    One `O_APPEND` write per line, as `store.append` does."""
    blinded = blind_name(name)
    if blinded == name:
        return blinded
    if any(row.get("blinded") == blinded for row in _side_rows(out_dir)):
        return blinded
    path = side_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"schema_version": 1, "blinded": blinded, "dataset_name": name},
                      separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)
    return blinded


def unblind(token: str, *, out_dir: Path | str = "outputs") -> str:
    """The true name behind a blinded name, from the side file; `token` when no row matches."""
    for row in _side_rows(out_dir):
        if row.get("blinded") == token and isinstance(row.get("dataset_name"), str):
            return row["dataset_name"]
    return token


def unblind_rows(rows: Iterable[Mapping[str, Any]], *,
                 out_dir: Path | str = "outputs") -> list[dict]:
    """Shallow copies of `rows` with string `dataset_name` and `dataset` values unblinded.

    One call serves verdict rows (`dataset_name`) and truth rows (`dataset`), so both meet in
    the true-name namespace. The input rows are not changed."""
    table: dict[Any, str] = {}
    for row in _side_rows(out_dir):  # the first row wins, as in `unblind`
        if isinstance(row.get("dataset_name"), str):
            table.setdefault(row.get("blinded"), row["dataset_name"])
    out = []
    for row in rows:
        copy = dict(row)
        for key in ("dataset_name", "dataset"):
            value = copy.get(key)
            if isinstance(value, str) and value in table:
                copy[key] = table[value]
        out.append(copy)
    return out
