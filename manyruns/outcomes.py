"""Completion is an account of outcomes, shared by the CLI and the live headline."""
from __future__ import annotations


def run_verdict(results: dict) -> tuple[int, str]:
    """0: successful or benignly declined; 1: failed/unsupported; 130: cancelled.

    Cancellation has its own status because a discarded attempt is neither an engine error
    nor retained work. `complete` means every step ran, so it cannot decide exit status: a
    normalization correctly declining signed data is incomplete but not a failure.
    """
    steps = results.get("steps") or []
    succeeded = sum(s.get("outcome") in ("ok", "reported") for s in steps)
    errors = sum(s.get("outcome") == "error" for s in steps)
    unsupported = sum(bool(s.get("unsupported")) for s in steps)
    skipped = sum(s.get("outcome") == "skipped" and not s.get("unsupported") for s in steps)
    if results.get("cancelled"):
        return 130, f"cancelled · {succeeded} successful steps retained"
    if errors or unsupported:
        return 1, f"failed · {succeeded} successful, {errors} errors, {unsupported} unsupported, {skipped} benign skips"
    if any(s.get("outcome") not in ("ok", "reported", "skipped") for s in steps):
        return 1, f"incomplete · {succeeded} successful steps"
    if not steps and results.get("ok") is False:
        return 1, "failed · no successful steps"
    return 0, f"finished · {succeeded} successful, {skipped} benign skips"
