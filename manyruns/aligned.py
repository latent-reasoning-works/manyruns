"""The row-alignment guard — the one check that stands between a fused matrix and a lie.

Every row of a data matrix must carry the ``(stream, sample)`` it came from. If the matrix
and its per-row metadata disagree by even one row, cells are attributed to the wrong donor
and *nothing downstream notices*: the only row-alignment check anywhere in the stack is a
``warnings.warn`` inside manylatents' ``PrecomputedDataModule`` ("Metadata length (N) does
not match embeddings length (M)") — and manyruns's own ``pipeline.quiet()`` blanket-ignores
warnings around the load. So today misalignment is completely silent. This raises.

Deliberately ~40 lines and dep-free. An earlier draft wrapped this in a typed-op registry
mirroring the learner's (schema checks, content-hash cache, lineage, snapshot/verify parity)
— ~280 lines that were cut, because an audit showed the registry contributed **zero**
alignment safety: the invariant is enforced at construction, before any schema check sees
the value, so `output_schema=Aligned` could only ever check the *type*, never the rows.
Nothing was registered on it either. When there are real ops to register (resolve →
materialize → fuse → load), that registry is copyable from the learner's ``registry.py``
verbatim — and by then their true signatures will be known, which today they are not.
"""
from __future__ import annotations

from typing import Any, Optional


class AlignmentError(ValueError):
    """A matrix and its per-row metadata disagree on how many rows there are."""


def n_rows(obj: Any) -> Optional[int]:
    """Row count of an array-like or a sequence of rows. ``None`` if it has no honest one.

    STRICT on purpose. Duck-typing this is how the guard gets holes:
      - a **dict** of columns (``{"sample": [...], "stream_id": [...]}`` — exactly what you
        build before ``pd.DataFrame(...)``) has ``len() == n_columns``, so a permissive
        version validates the *wrong axis* and passes a misaligned payload;
      - a **str** has ``len() == n_chars``, so ``"abc"`` would validate a 3-row matrix.
    Both are rejected rather than guessed at. ``.shape[0]`` (ndarray, DataFrame) and
    list/tuple-of-rows are the only shapes with an unambiguous row count.
    """
    if isinstance(obj, (str, bytes, dict, set)):
        return None
    shape = getattr(obj, "shape", None)  # ndarray / DataFrame
    if shape is not None:
        try:
            return int(shape[0])
        except (TypeError, IndexError, ValueError):
            return None
    if isinstance(obj, (list, tuple)):  # a sequence of rows
        return len(obj)
    return None


def check_aligned(matrix: Any, meta: Any, *, what: str = "payload") -> int:
    """Raise :class:`AlignmentError` unless ``meta`` has exactly one row per matrix row.

    Returns the row count so it can be used inline. ``meta`` may be an ndarray/DataFrame
    (``.shape[0]``) or a list/tuple of rows — see :func:`n_rows` for why not a dict.
    """
    n = n_rows(matrix)
    if n is None:
        raise AlignmentError(
            f"{what}: matrix has no row count (got {type(matrix).__name__}); "
            "expected an array-like with .shape, or a list/tuple of rows"
        )
    m = n_rows(meta)
    if m is None:
        raise AlignmentError(
            f"{what}: per-row metadata has no honest row count (got {type(meta).__name__}). "
            "A dict of columns counts COLUMNS and a str counts CHARACTERS — both would "
            "validate the wrong axis. Pass a DataFrame or a list of per-row records."
        )
    if m != n:
        raise AlignmentError(
            f"{what}: row misalignment — matrix has {n} rows but its per-row metadata has "
            f"{m}. Every row must carry the (stream, sample) it came from; off by even one "
            "and cells are silently attributed to the wrong donor."
        )
    return n
