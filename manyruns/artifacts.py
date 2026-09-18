"""The state channel — intermediate arrays, written down and addressable.

A run used to leave three things behind: `project.yaml`, `summary.md`, and a PNG. The
embedding itself — the object every downstream question is actually about — lived in
`state["emb"]` for the duration of the process and was then dropped. What survived was a
*picture* of it.

That is one gap with four faces:

  - **`embedding=` had no producer.** `run_inproc(embedding=...)` accepts coordinates from
    anywhere, which is what makes an unconventional composition runnable — but nothing this
    product produced could be handed back to it. You could only supply state you got
    elsewhere.
  - **`curate` has no input.** `harness/storage.save_labeled(path, entry, embedding)` has been
    callable for as long as it has existed and nothing has ever produced the embedding it
    takes. A human cannot label a run whose object was thrown away.
  - **The federated payload has nothing to carry.** Pooling per cell type × condition needs a
    per-cell object to stratify; a g-vector of scalars cannot be stratified after the fact.
  - **MIOFlow's trained flow is discarded at the return statement**
    — coordinates kept, the simulatable object dropped.
    The same discard one level up, and this is the channel that would hold it.

**The convention is manylatents', not ours.** `manylatents.experiment.run` returns a
`LatentOutputs` dict — `embeddings`, `label`, `metadata`, `scores`, optionally
`callback_outputs` — and `callbacks/embedding/save_outputs.py` persists it as one file per
key (`{base}_{key}.npy` for arrays, `.json` otherwise) written atomically, with a completion
marker so a reader can tell a finished write from an interrupted one. That shape is followed
here rather than reinvented. It is re-implemented rather than imported because the in-process loop
must work with no private stack, and a state channel that only exists when manylatents is
installed would be missing on exactly the install the product ships.

**Addressed by `run_id`, which is why identity had to come first.** An artifact nothing can
name is a temp file. The path rides in the step record, so `index.jsonl` already carries it —
a reader goes from a row in the run store to the array that row describes with no filesystem
convention in its head.

**Never the input.** Only DERIVED state is written: `emb`, `pseudotime`. `state["X"]` is the
scientist's data — 2,700 × 32,738 for pbmc3k — and copying it into an outputs folder is a
data-handling decision this module has no business making silently. It is also the line the
federated design turns on: geometry leaves, the person's data does not.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

#: Slots worth persisting. `X` is deliberately absent — see the module docstring. `labels`
#: too, unless the caller explicitly supplies Session(sample_ids=...). That opt-in stores
#: row annotations in the manifest, tied to each derived array's actual row selection.
PERSISTED = ("emb", "pseudotime")

#: A separate, much smaller ceiling for ENGINE BYPRODUCTS (`runner._collect_extras`), because
#: they are a different size class from manyruns's own state and the cap below was chosen for
#: the wrong one. Measured on a `phate` step: the embedding is `(120, 3) float32` = 1.5 KiB and
#: the affinity is `(120, 120) float64` = 112.6 KiB — the byproduct is 75x the result, and it
#: grows as N**2 while the embedding grows as N:
#:
#:      cells      affinity + kernel, per latent step
#:        120      0.2 MiB
#:      2,700      111.2 MiB     <- pbmc3k, the one real dataset in the repo
#:      5,000      381.5 MiB
#:
#: At 64 MiB both of pbmc3k's matrices would be written, on every latent step, for every run —
#: `cluster` has three of them. 16 MiB is roughly a 1,450-cell square: synthetics and small
#: cohorts keep their matrices, anything larger is RECORDED as a stated absence instead. The
#: choice being made is that a byproduct nobody asked for should not silently outweigh the
#: result by three orders of magnitude; raise this constant if a caller wants the bytes.
EXTRAS_MAX_BYTES = 16 * 1024 * 1024

#: A run that would write more than this to one array is doing something the caller did not
#: ask for. 64 MiB is ~2.8M cells at 3 float64 columns — far past anything manyruns ships
#: (pbmc3k is 2,700) and far short of a raw expression matrix. Over it the array is skipped
#: WITH a reason, because a missing artifact and a too-large one are different facts.
MAX_BYTES = 64 * 1024 * 1024

#: Written last, after every array. Its presence is the reader's only guarantee that the
#: folder is complete — taken from manylatents' `write_completion_marker`, and the reason a
#: killed run leaves an obviously-partial directory rather than a plausible one.
DONE = "COMPLETE"


def root(out_dir: Path | str, run_id: Optional[str] = None) -> Path:
    """Where this run's state lives. Sibling of `plots/`, so one run is one folder."""
    base = Path(out_dir) / "state"
    return base / run_id if run_id else base


def too_large(arr: Any, max_bytes: Optional[int] = None) -> bool:
    """Would this array be skipped for size? Separate so the caller can say WHY.

    `max_bytes` defaults to `MAX_BYTES`; the extras channel passes `EXTRAS_MAX_BYTES`."""
    try:
        import numpy as np
    except ImportError:
        return False
    cap = MAX_BYTES if max_bytes is None else max_bytes
    return arr is not None and hasattr(arr, "shape") and np.asarray(arr).nbytes > cap


def save_array(out_dir: Path | str, run_id: Optional[str], index: int, step: str,
               slot: str, arr: Any, max_bytes: Optional[int] = None) -> Optional[str]:
    """Write one intermediate array atomically. Returns its path, or None if nothing was written.

    Returns None rather than raising on every failure path: a run that produced a valid
    embedding must not become a failed run because a disk was full or a directory was
    read-only. The caller records the absence, and `_run_steps` never sees an exception here.

    Atomic by write-then-rename, so a reader never observes a truncated `.npy`. `np.save` to
    the final path directly would leave one on any interruption, and the sweep runs cells in a
    process pool — concurrent readers are not hypothetical.
    """
    try:
        import numpy as np
    except ImportError:  # noqa: BLE001 - no numpy, no arrays to save
        return None
    if arr is None or not hasattr(arr, "shape") or too_large(arr, max_bytes):
        return None
    a = np.asarray(arr)
    try:
        d = root(out_dir, run_id)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{index:02d}-{step}_{slot}.npy"
        # The temp name must ALSO end in `.npy`: `np.save` appends the suffix when the
        # filename lacks it, so a `.part` name produced `<name>.npy.part.npy` on disk while
        # the rename looked for `<name>.npy.part` and failed — silently, since every OSError
        # here degrades to "not written". Leading dot keeps a partial out of a glob.
        tmp = path.with_name(f".{path.name}.part.npy")
        np.save(tmp, a)
        os.replace(tmp, path)
        return str(path)
    except OSError:
        return None


def finish(out_dir: Path | str, run_id: Optional[str], written: dict) -> Optional[str]:
    """Mark the run's state folder complete, and index what is in it.

    Without this a reader cannot distinguish "this run produced one artifact" from "this run
    was killed after its first artifact". The manifest is the same question the g-vector's
    declared-not-conditional rule answers for metrics: what was expected, and what arrived."""
    if not written:
        return None
    try:
        d = root(out_dir, run_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / DONE).write_text(json.dumps(written, indent=2, sort_keys=True))
        return str(d / DONE)
    except OSError:
        return None


def complete(state_dir: Path | str) -> bool:
    """Is this state folder safe to read? False for an interrupted or in-flight run."""
    return (Path(state_dir) / DONE).is_file()


def completed_runs(out_dir: Path | str) -> list[str]:
    """Every COMPLETEd run under `out_dir/state/`, oldest first.

    Ordered by the COMPLETE marker's own mtime, not the folder name — `run_id` is a random
    uuid4 hex (`runner.identity`), so it carries no order of its own. An interrupted run (no
    COMPLETE) is invisible here on purpose, for the same reason `complete()` exists: a killed
    process left a plausible-looking folder, not a safe one to resume into.

    ONE home for "which runs exist and in what order" — `latest_run` is the degenerate
    `[-1]` of this, and `session.resumable` folds over all of it (a project's history is more
    than its single most recent run; see that function's docstring)."""
    base = root(out_dir)
    if not base.is_dir():
        return []
    marks = sorted(base.glob(f"*/{DONE}"), key=lambda p: p.stat().st_mtime)
    return [p.parent.name for p in marks]


def latest_run(out_dir: Path | str) -> Optional[str]:
    """The most recently COMPLETEd run under `out_dir/state/`, or None."""
    runs = completed_runs(out_dir)
    return runs[-1] if runs else None


def read_manifest(out_dir: Path | str, run_id: str) -> dict:
    """The COMPLETE marker's own content, parsed back: `{path: {"step", "index", "slot"}}` —
    `finish()`'s `written` argument, round-tripped. The other half of `finish`, the way
    `load_array` is the other half of `save_array`."""
    return json.loads((root(out_dir, run_id) / DONE).read_text())


def load_array(path: Path | str) -> Any:
    """Read an artifact back. The other half of `run_inproc(embedding=...)`.

    This is what makes a saved embedding a first-class INPUT rather than a record of one:
    measure a cohort once, then run five trajectory variants against the same coordinates
    without recomputing them — and, because the g-vector's key set is fixed, compare them.

    `allow_pickle=False` on purpose: an artifact is data, and a `.npy` that can execute on
    load is a code path from the filesystem into the process."""
    import numpy as np

    p = Path(path)
    if not p.exists():
        raise ValueError(f"artifact not found: {p}")
    return np.load(p, allow_pickle=False)


def load_labeled(out_dir: Path | str, run_id: str, index: int,
                 slot: str = "emb") -> dict:
    """Read a completed artifact and its recorded row identity, without reopening inputs.

    Old/unannotated artifacts refuse: neither positional integers nor labels reloaded from
    a mutable source can reconstruct the identity that was never recorded.
    """
    entries = [(path, entry) for path, entry in read_manifest(out_dir, run_id).items()
               if entry["index"] == index and entry["slot"] == slot]
    if len(entries) != 1:
        raise ValueError(f"expected one artifact at {run_id}/{index}/{slot}")
    path, entry = entries[0]
    identity = entry.get("row_identity")
    if identity is None:
        raise ValueError(f"artifact has no recorded row identity: {path}")
    array = load_array(path)
    ids, labels = identity["sample_ids"], identity["labels"]
    if (len(ids) != len(array) or len(set(ids)) != len(ids)
            or (labels is not None and len(labels) != len(ids))):
        raise ValueError(f"artifact row identity is misaligned: {path}")
    return {"array": array, **identity}
