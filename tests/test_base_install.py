"""Can `import manyruns` survive an environment that lacks the compute stack?

**THIS FILE IMPORTS NOTHING FROM MANYRUNS, AND THAT IS THE WHOLE DESIGN.** It reads the package
as TEXT and parses it, because the failure it guards against is an import failure — and a guard
that imports the thing it is guarding dies with it. Measured, by injecting `import pyrovelocity`
at the top of `pipeline/steps.py` with this check living beside the other contract tests:

    !!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!

No failing test, no message, no name of the offending module — pytest could not collect the file
that would have explained it. Split out, the same injection produces a named assertion with the
file and line. That is the difference between a gate and a crash.

WHY IT MATTERS. CI installs almost none of the compute stack — no torch, no manylatents
(`.github/workflows/ci.yml` — `--no-project` plus an explicit `--with` list). A package imported
at module scope makes `import manyruns` raise for every test in every file that touches the
module, and for every user with a base install.

MEASURE CI WITH `git archive`, NEVER `rsync`. `/data/` is gitignored (`.gitignore:103`), so a
real checkout has no `data/pbmc3k_raw.h5ad` and an rsync of a working tree does — which changes
the answer: `test_tui_state.py::test_the_declared_shape_survives_wherever_there_is_nothing_to_infer_from`
inspects that file when it is present and fails without anndata. Measured both ways, the
rsync-based figure was wrong by 60-odd tests. On a faithful checkout of this branch:
**1261 passed / 69 skipped** under main's `--with` list, **1287 / 45** under this one.
"""
from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "manyruns"
# A rename once pointed this at a directory that no longer existed. `rglob` then yielded
# nothing and the gate below passed over an empty set — green, and testing nothing. The
# negative control covers the parser, not the path, so the path asserts for itself.
assert PACKAGE.is_dir(), f"the package moved out from under this test: {PACKAGE}"

#: Third-party packages manyruns may import AT MODULE SCOPE. Deny-by-default, and that is the
#: point: an allowlist catches a package nobody thought to denylist, which is exactly what a new
#: tool is. Measured on this tree — these four and nothing else.
#:
#:   click / rich / textual — the interface. Pure Python, base dependencies, and the front door
#:                            cannot lazy-import widget classes it subclasses at class-body scope.
#:   numpy                  — base dependency; `harness/display.py` types against it.
#:
#: EVERYTHING ELSE IS DEFERRED, INCLUDING BASE DEPENDENCIES — scipy, sklearn, matplotlib, phate,
#: anndata and pandas are all `dependencies` in `pyproject.toml` and all deferred here. Being a
#: dependency is not the test. The test is whether `import manyruns` still works without it,
#: because that is the environment CI runs in and the one a partial install leaves behind.
MODULE_SCOPE_ALLOWED = frozenset({"click", "rich", "textual", "numpy"})

_STDLIB = frozenset({
    "__future__", "abc", "argparse", "asyncio", "collections", "contextlib", "copy", "csv",
    "dataclasses", "datetime", "enum", "functools", "glob", "hashlib", "importlib", "inspect",
    # Unix-only stdlib: the product supports macOS and Linux, Python 3.11–3.12. Windows is
    # unsupported because required single-cell dependency scikit-misc ships no Windows wheels.
    # It is here rather than in the allowlist because that is for THIRD-PARTY packages a base
    # install may lack; `fcntl` is present on every platform this ships to, so a deferred import
    # would buy nothing. `trace.py` locks with it.
    "fcntl",
    "io", "itertools", "json", "logging", "math", "os", "pathlib", "platform", "queue",
    "random", "re",
    "shutil", "signal", "string", "subprocess", "sys", "tarfile", "tempfile", "textwrap",
    "threading", "time", "traceback", "types", "typing", "urllib", "uuid", "warnings",
    "webbrowser", "zipfile",
})


def module_scope_imports(source: str) -> set:
    """Top-level imports only — a deferred import inside a function is exactly what we want."""
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return {n for n in found if n and n not in _STDLIB and n != "manyruns"}


def test_no_module_scope_import_outside_the_allowlist():
    """THE BASE-INSTALL GATE. A new tool's package belongs inside the function that needs it."""
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for name in sorted(module_scope_imports(path.read_text()) - MODULE_SCOPE_ALLOWED):
            offenders.append(
                f"{path.relative_to(PACKAGE.parent)}: `import {name}` at module scope")
    assert not offenders, (
        "module-scope import of a package outside the allowlist. Move it inside the function "
        "that needs it, so `import manyruns` survives an install without it:\n  "
        + "\n  ".join(offenders))


def test_the_gate_actually_fires():
    """A guard that cannot fail is theatre. The negative control for the test above."""
    assert module_scope_imports("import pyrovelocity\n") == {"pyrovelocity"}
    assert module_scope_imports("from scvelo.tools import velocity\n") == {"scvelo"}
    assert module_scope_imports("import anndata as ad, torch\n") == {"anndata", "torch"}
    # …and the deferred form, which is the shape the product requires, is invisible to it.
    assert module_scope_imports("def f():\n    import pyrovelocity\n") == set()
    # A relative import is manyruns' own and never a third-party package.
    assert module_scope_imports("from .steps import _StepSkipped\n") == set()

