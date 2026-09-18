"""What a dropped file turned out to be, remembered — so the landing screen opens files once.

The roster opens a file per dropped row (`narrate.read_data`, 361 ms cold on the 5.9 MB
`pbmc3k_raw.h5ad`, 18 ms warm) and it does it on **every launch**, because nothing was written
down. One file is a blink. A real drop folder is not one file, and the screen a scientist meets
first is the screen that pays for all of them before it can draw a row.

**The key is identity, not the path.** `(size, mtime_ns)` beside the path, because a file that
was replaced under the same name is a different file and must be re-read — that is the failure a
plain path cache has, and it is silent: the roster would describe the old contents of a name that
now holds new data. Anything that does not match exactly is a miss.

**A miss and a corrupt cache cost the same thing: one read.** Every failure path here returns
"not cached" or declines to write, so the roster's behaviour with no cache, an unreadable cache
and a stale cache is identical — the only difference is speed. That is deliberate: a cache that
can change an ANSWER is worse than no cache, and this one is only allowed to change a duration.

**Bundled datasets are not in here.** They declare their shape in YAML and `shell._resolve_dataset`
reads the declaration rather than the file, so there is nothing to remember — the declaration is
already the alias.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

#: Where the cache lives. Beside the fallback drop folder rather than in the working directory,
#: because `shell.data_dir()` is relative (`data`) and a cache that moved with the cwd would be
#: cold every time you launched from somewhere new — which is exactly when a big folder hurts.
HOME = Path.home() / ".manyruns"

#: Bumped when the stored shape changes meaning. An older file is ignored wholesale rather than
#: migrated: the cost of being wrong is a wrong ANSWER on the landing screen, and the cost of
#: ignoring it is one read.
VERSION = 1

#: A ceiling on remembered rows, so a folder someone points at once does not grow this forever.
#: Oldest-out by insertion order, which `dict` preserves.
MAX_ROWS = 512


def path() -> Path:
    return HOME / "inspected.json"


def key(p: Path) -> Optional[str]:
    """`path|size|mtime_ns` — the identity a cached answer is valid for.

    `None` when the file cannot be stat'd, which reads as a miss. Directories get a key too:
    their mtime changes when their contents change, which is the same question one level up.
    """
    try:
        st = p.stat()
    except OSError:
        return None
    return f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"


def _load() -> dict:
    try:
        raw = json.loads(path().read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or raw.get("version") != VERSION:
        return {}
    rows = raw.get("rows")
    return rows if isinstance(rows, dict) else {}


def get(p: Path) -> Optional[dict]:
    """The remembered inspection for this exact file, or `None`. Never raises."""
    k = key(p)
    if k is None:
        return None
    row = _load().get(k)
    return row if isinstance(row, dict) else None


def put(p: Path, value: dict) -> None:
    """Remember one inspection. Silent on any failure — a read-only home directory must cost
    speed, not the screen."""
    k = key(p)
    if k is None:
        return
    rows = _load()
    rows.pop(k, None)                      # re-insert so the newest is last for the trim below
    rows[k] = value
    for stale in list(rows)[:-MAX_ROWS]:
        del rows[stale]
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = path().with_suffix(".json.part")
        tmp.write_text(json.dumps({"version": VERSION, "rows": rows}, separators=(",", ":")))
        os.replace(tmp, path())            # atomic: two launches must not read a half-written file
    except OSError:
        return


def forget() -> None:
    """Drop the whole cache. The escape hatch for a reader who suspects it — and the reason
    `manyruns` never needs a `--no-cache` flag."""
    try:
        path().unlink()
    except OSError:
        return


def as_row(modality: str, obs: Any) -> dict:
    """One inspection, flattened to JSON.

    EVERY field of the dataclass, read off `dataclasses.fields` rather than hand-listed. A
    hand-list drops whatever gets added next, silently, and the thing it drops is a field some
    caller keys a decision on — `conditions` is what `vocab.unmet` uses to decide whether
    `contrast` can run, so a cache that lost it would make the ledger refuse a recipe that
    should run. Wrong answers are the one thing this module is not allowed to produce.
    """
    import dataclasses

    return {"modality": modality,
            "obs": {f.name: getattr(obs, f.name) for f in dataclasses.fields(obs)}}


def to_observation(row: dict, source: str) -> Optional[Any]:
    """The stored row back into an `Observation`, or `None` if it cannot be rebuilt exactly.

    `None` rather than a partial object: a half-restored Observation is a wrong answer wearing
    the shape of a right one, and the caller's fallback — read the file — is cheap and correct.
    """
    import dataclasses

    from manyruns.narrate import Observation

    stored = row.get("obs")
    if not isinstance(stored, dict):
        return None
    names = {f.name for f in dataclasses.fields(Observation)}
    if set(stored) != names:                 # the schema moved under us; re-read instead
        return None
    kwargs = dict(stored)
    kwargs["source"] = source
    try:
        return Observation(**kwargs)
    except TypeError:
        return None
