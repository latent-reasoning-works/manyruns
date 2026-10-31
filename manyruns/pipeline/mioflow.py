"""The manylatents command-line route for a `lightning` step that is not MIOFlow.

manyruns never imports MIOFlow. A `mioflow` step runs only as a tool step
(`{name: mioflow, group: lightning, tool: mioflow}`), through the model seam
(`pipeline/external.py`), in the tool's own interpreter; the MIOFlow example folder
(`examples/tools/mioflow`) declares that tool. The in-process route that lived here, and its
invented time axis (a diffusion pseudotime binned into 5 time points when the data carried no
time labels), is removed: `runner._ml_lightning` refuses an untooled `mioflow` with the fix.

What stays is the shell-out for the other lightning step names (`latent_ode`, …), which already
ran in a subprocess. No bundled recipe uses it.
"""
from __future__ import annotations


def _lightning_cmd(name: str, dataset: str, fast_dev_run: bool) -> list[str]:
    """The manylatents CLI invocation for a Lightning module (MIOFlow, …). Trained
    modules aren't name-addressable via `run()`, so they go through Hydra config groups —
    mirroring `python -m manylatents.main algorithms/lightning=mioflow data=swissroll`."""
    import sys

    cmd = [sys.executable, "-m", "manylatents.main", f"algorithms/lightning={name}", f"data={dataset}"]
    if fast_dev_run:
        # `+` appends: manylatents' trainer config doesn't predeclare fast_dev_run
        cmd.append("+trainer.fast_dev_run=true")
    return cmd


def _run_lightning_cli(name: str, dataset: str, fast_dev_run: bool) -> str:
    """Run a Lightning step via the manylatents CLI; return a status string."""
    import os
    import subprocess

    cmd = _lightning_cmd(name, dataset, fast_dev_run)
    env = {**os.environ, "HYDRA_FULL_ERROR": "1"}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
    except Exception as e:  # noqa: BLE001 - report + continue
        return f"error: {type(e).__name__}: {e}"
    if r.returncode == 0:
        return f"ok (manylatents.main {' '.join(cmd[3:])})"
    tail = " | ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])
    return f"error (exit {r.returncode}): {tail[-400:]}"
