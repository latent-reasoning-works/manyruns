"""The annotation store — one line per judgment about a run or a dataset.

An annotation is what a person or a model THINKS of a run: which topologies it shows, how it
ranks against other runs, or which step should have been taken instead. It is a judgment,
not a committed choice. Committed choices, and the set each one was chosen from, stay in
`decisions.jsonl` (`manyruns.decisions`); this module never writes there, and `store.append`'s
`index.jsonl` stays the record of what ran. Three files, three meanings, one reader each.

Three kinds share one row shape (`KINDS`):

  label       the topologies a run (or, for an imported label, a dataset) shows
  rating      one run's score, rank or rubric, among the runs it was compared with
  correction  the step an expert would have taken at one position of a run

**What is in a row, and what is deliberately not.** Run ids, a decision id, a step position and
name, topology names, scores, and a digest of an imported source file.
A row holds no data, no coordinates and no counts — the same line `decisions.py` draws for its
rows. No source path either: a path on a laptop names its owner's home directory.

**No interaction telemetry is recorded.** A toggle, a back-out or an abandoned form writes
nothing; only a committed judgment is a row.

**The chooser is a pseudonym or a model id, never a real name.** It is `human:<pseudonym>` or
`llm:<model>`, and `append` refuses any other shape. `PSEUDONYM_PATTERN` is the one check for the
part after `human:`, shared by the CLI verbs and the run screen. The pattern refuses a space and
a capital letter, which catches the common mistake; it cannot prove that a lowercase pseudonym is
not someone's name, so the caller still owns that rule.

Version 1 rows carry ten top-level keys, in this order: ``schema_version`` (written first),
``annotation_id``, ``kind``, ``at``, ``chooser``, ``run_id``, ``decision_id``, ``step``,
``should`` and ``payload``. Missing versions read as 1; unknown keys survive reading. Future
downstream fields go under ``payload.extra.<caller>``, the rule `decisions.py` states.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from manyruns import decisions, vocab

#: The three kinds of judgment one row can hold.
KINDS = ("label", "rating", "correction")

#: The part of a `human:` chooser after the prefix: lowercase letters, digits, `_` and `-`, at
#: most 32 characters, starting with a letter or digit. Full-match it.
PSEUDONYM_PATTERN = r"[a-z0-9][a-z0-9_-]{0,31}"

#: The part of an `llm:` chooser after the prefix. Wide enough for `org/model@revision`.
_MODEL_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:/@-]{0,79}"

_CHOOSER = re.compile(rf"human:{PSEUDONYM_PATTERN}|llm:{_MODEL_PATTERN}")


def index_path(out_dir: Path | str = "outputs") -> Path:
    """Sibling of `decisions.index_path`, and separate for the reason the module docstring gives."""
    return Path(out_dir) / "annotations.jsonl"


def _check_chooser(chooser: Any) -> None:
    if not isinstance(chooser, str) or not _CHOOSER.fullmatch(chooser):
        raise ValueError(f"chooser must be 'human:<pseudonym>' or 'llm:<model>', got {chooser!r}")


def append(kind: str, *, run_id: Optional[str], chooser: str, payload: dict,
           decision_id: Optional[str] = None, step: Optional[int] = None,
           should: Optional[str] = None,
           out_dir: Path | str = "outputs") -> Optional[str]:
    """Write one annotation. Returns its `annotation_id`, or `None` if the write failed.

    A bad argument raises `ValueError` before the file is opened, so no line is written: a wrong
    kind or chooser is a caller bug, and a silent row would be a lie in the record. An I/O
    failure returns `None` instead, as `decisions.append` does.

    `step` and `should` belong to a correction and only to one: `step` is the position in the
    run row's `steps` list, and `should` is the canonical name of the step that should have run.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    _check_chooser(chooser)
    if kind == "correction":
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"a correction needs a non-negative int step, got {step!r}")
        if not should or not isinstance(should, str):
            raise ValueError(f"a correction needs the step it should have been, got {should!r}")
    elif step is not None or should is not None:
        raise ValueError(f"step and should are for a correction, not a {kind}")
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    if "topology" in payload:
        problems = vocab.check_topology(payload["topology"])
        if problems:
            raise ValueError("; ".join(problems))

    annotation_id = decisions.new_id()
    row = {
        "schema_version": 1,
        "annotation_id": annotation_id,
        "kind": kind,
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "chooser": chooser,
        "run_id": run_id,
        "decision_id": decision_id,
        "step": step,
        "should": should,
        "payload": payload,
    }
    try:
        line = json.dumps(row, separators=(",", ":")) + "\n"
        path = index_path(out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        # One atomic append, no read-modify-write — the discipline of `decisions.append`.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except OSError:
        return None
    return annotation_id


def read(out_dir: Path | str = "outputs") -> Iterator[dict]:
    """Every annotation, oldest first. A blank line, a corrupt line, and a line that is valid JSON
    but not an object are skipped rather than fatal. Missing versions mean version 1; unknown
    keys and explicit versions are preserved."""
    path = index_path(out_dir)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        row.setdefault("schema_version", 1)
        yield row


def _join(row: dict) -> tuple[str, Any]:
    kind = row.get("kind")
    if kind == "label" and row.get("run_id") is None:
        return "dataset", (row.get("payload") or {}).get("dataset")
    if kind == "correction" and row.get("decision_id") is not None:
        return "decision_id", row["decision_id"]
    return "run_id", row.get("run_id")


def examples(out_dir: Path | str = "outputs") -> list[dict]:
    """Every annotation with its join edge named: `{kind, chooser, join, key, payload, step,
    should}`.

    `join` names the field that links the row to the rest of the record, and `key` is its value:

      label with a run_id       run_id       (a judgment about one run)
      label without a run_id    dataset      (an imported label; `payload["dataset"]`)
      rating                    run_id
      correction + decision_id  decision_id  (the ledger decision that started the run)
      correction, no decision   run_id

    `step` is the position in the run row's `steps` list — the `index` that
    `Session.branch(index)` takes to rewind there. `should` is a step name that
    `session.resolve_step` resolves. A row whose key is `None`, or whose kind is not in `KINDS`,
    is left out.
    """
    out: list[dict] = []
    for row in read(out_dir):
        if row.get("kind") not in KINDS:
            continue
        join, key = _join(row)
        if key is None:
            continue
        out.append({
            "kind": row["kind"],
            "chooser": row.get("chooser"),
            "join": join,
            "key": key,
            "payload": row.get("payload"),
            "step": row.get("step"),
            "should": row.get("should"),
        })
    return out


def benchmark_topology(raw: list[str]) -> list[str]:
    """Raw scShapeBench labels → canonical topologies, through `vocab.scshapebench_to_topology`.

    A label that is not a key of `vocab.SCSHAPEBENCH_TOPOLOGY` is ignored. The match is exact.
    The result is de-duplicated and in `vocab.TOPOLOGIES` order; `[]` when nothing maps.
    """
    found = {vocab.scshapebench_to_topology(label) for label in raw
             if isinstance(label, str) and label in vocab.SCSHAPEBENCH_TOPOLOGY}
    return [t for t in vocab.TOPOLOGIES if t in found]
