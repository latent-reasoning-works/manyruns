# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — manyruns's base/mock tier as one self-contained binary.

Distribution Path "B" (see CLAUDE.md): a Python-free single file that runs the interactive
shell, `check`, and the mock flow — no interpreter, no `uv`, no private stack. Only the
SDK-free base tier compiles small; the real-engine tiers (`[harness]`/`[engine]`, torch/jax/
manylatents) stay a `uv tool install`.

The trick: build in a MINIMAL environment (base deps + rich + questionary) so the heavy and
private packages are simply *absent* and can't be pulled in. The excludes below just silence
the guarded lazy-import references the mock path never executes.

    uv run --no-project \
        --with pyinstaller --with rich --with questionary \
        --with omegaconf --with hydra-core --with numpy \
        pyinstaller --clean --noconfirm manyruns.spec
    ./dist/manyruns          # the interactive front door
    ./dist/manyruns check    # validate the bundled registries
"""
import os
import sys

from PyInstaller.utils.hooks import collect_submodules

# Build from the repo root. In a `--no-project` env manyruns isn't installed, so put the repo
# root on the path — both for `collect_submodules("manyruns")` here and for Analysis to resolve
# the entry's `from manyruns.app import main`.
_ROOT = os.path.abspath(os.getcwd())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# manyruns's own modules are imported lazily inside functions (so the base app stays light),
# which static analysis can't see — name them explicitly. The `harness` subpackage is excluded:
# it needs click + the acquire/engine deps that don't belong in the base binary.
hiddenimports = (
    collect_submodules("manyruns", filter=lambda n: not n.startswith("manyruns.harness"))
    + collect_submodules("omegaconf")
    + collect_submodules("questionary")
    + collect_submodules("prompt_toolkit")
    + ["rich"]
)

# The private + deep-learning stack — reached only by guarded lazy imports the mock path never
# runs. Absent from a minimal build env anyway; listed so PyInstaller doesn't warn or bloat.
#
# NO LEARNER IS LISTED, and it is not an oversight. `tests/test_no_track_to_a_learner.py`
# forbids any learner name in `manyruns/` code, so nothing here can import one and PyInstaller
# can never reach one to bundle. An exclude would be guarding a door the guard already welds.
excludes = [
    "manylatents", "manyagents", "shop", "heatgeo", "anthropic",
    "torch", "jax", "tensorflow", "scanpy", "anndata", "phate",
    "sklearn", "matplotlib", "pandas", "IPython", "pytest",
    # scipy is pulled only by manyruns.experiment (the metric-separation sweep), which is a
    # Python-API entry with no CLI verb — the binary's shell/check/mock flow never imports it.
    "scipy",
]

a = Analysis(
    ["packaging/manyruns_entry.py"],
    pathex=[_ROOT],
    binaries=[],
    datas=[("manyruns/configs", "manyruns/configs")],  # the YAML registries + READMEs
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

# One-file: passing a.binaries + a.datas into EXE (no COLLECT) makes it a single executable.
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="manyruns",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
