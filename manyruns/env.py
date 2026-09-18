"""Environment-variable reads, through one accessor.

Every override this product honours is `MANYRUNS_*`, and every read of one goes through
`get` / `is_set` here rather than through `os.environ` at the call site.

THE PRE-RENAME PREFIX IS GONE. The package was called `geomancer` until 2026-08-30 and this
module carried a fallback to the old prefix through the transition; it was retired on
2026-09-01, once nothing anywhere set the old names. `tests/test_env.py` now pins the inverse
of what it used to — the old prefix appears in no source file at all, this one included, which
is why it is not spelled out here.

The module outlives the shim it was built for, and the reason is
`test_no_module_reads_the_environment_around_the_shim`: one accessor is what makes a change
to override semantics — a new prefix, a config file, a precedence rule — a one-file edit
instead of eleven call sites to find.
"""
from __future__ import annotations

import os

_PREFIX = "MANYRUNS_"


def get(name: str, default: str | None = None) -> str | None:
    """`$MANYRUNS_<name>`, or `default` when it is unset.

    `name` is the part AFTER the prefix — `get("DATA_DIR")` reads `$MANYRUNS_DATA_DIR`.
    An empty string is a real value and is returned as one; only a genuinely unset variable
    falls through to `default`, so `MANYRUNS_X=""` deliberately means "off" rather than
    "unspecified".
    """
    value = os.environ.get(_PREFIX + name)
    return default if value is None else value


def is_set(name: str) -> bool:
    """Whether the override is present, empty string included."""
    return get(name) is not None
