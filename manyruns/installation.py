"""Pasteable runtime repairs. These commands are advice, never executed by the product."""

# The base metadata requests the singlecell extra. Consumer resolution remains an owner-run
# release check; a checkout's source pins alone do not establish an installed tool's sources.
REINSTALL = (
    "uv tool install --python 3.12 --force "
    "git+https://github.com/latent-reasoning-works/manyruns"
)
CHECKOUT_SYNC = "uv sync --extra harness --extra dev"
ENVIRONMENT_REPAIR = (
    f"for an installed tool, run `{REINSTALL}`; for a checkout, run `uv sync` with "
    f"all already used extras (standard development: `{CHECKOUT_SYNC}`; "
    "keep `--extra agents` if already used). The Git source is public; no organisation "
    "access or GitHub login is required"
)
RUNTIME_REPAIR = (
    "repair the single-cell runtime (manylatents-omics[singlecell]): "
    f"{ENVIRONMENT_REPAIR}"
)
