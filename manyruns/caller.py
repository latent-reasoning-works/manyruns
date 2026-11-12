"""Optional wrapping-program provenance, for display and run records only."""
from __future__ import annotations

from manyruns import env


def read() -> dict[str, str | None] | None:
    """Read and validate the current declaration; an empty name means unknown."""
    import unicodedata

    name = env.get("CALLER")
    version = env.get("CALLER_VERSION")
    if name is not None and (
        not isinstance(name, str) or len(name) > 100
        or any(unicodedata.category(char) == "Cc" for char in name)
    ):
        raise ValueError("MANYRUNS_CALLER must be a string of at most 100 characters without control characters")
    if version is not None and (not isinstance(version, str) or len(version) > 100):
        raise ValueError("MANYRUNS_CALLER_VERSION must be a string of at most 100 characters")
    return {"name": name, "version": version or None} if name else None
