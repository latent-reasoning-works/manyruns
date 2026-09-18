"""The one accessor every environment override is read through, and the prefix that is gone.

`manyruns/env.py` used to be a COMPAT SHIM: it read the pre-rename prefix when `$MANYRUNS_*`
was unset, so a shell profile or CI job written before the 2026-08-30 rename did not silently
fall back to a default. That shim was retired on 2026-09-01 — nothing in the environment,
in any shell config, or in CI set the old names, so the fallback protected nobody and the last
`geomancer` string in the package was carrying that cost for it.

What the file pins now is the INVERSE of what it used to, plus the one thing that was never
about the shim: every `$MANYRUNS_*` read goes through `env.get` / `env.is_set` rather than
through `os.environ` at the call site. That was the shim's real failure mode once already
(see `test_no_module_reads_the_environment_around_the_accessor`) and it outlives the shim —
one accessor is what keeps a change to override semantics a one-file edit.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from manyruns import env

#: Built at runtime rather than written out, because the guard below scans source files for
#: it — spelling it here would make this file its own first offender.
_PRE_RENAME = "GEO" + "MANCER_"


def _package() -> pathlib.Path:
    package = pathlib.Path(__file__).resolve().parent.parent / "manyruns"
    # A test that walks a DIRECTORY passes green when the directory moves, having scanned
    # nothing. The rename found three of those; this assert is the fix.
    assert package.is_dir(), f"the package moved out from under this test: {package}"
    return package


def test_the_prefix_is_read(monkeypatch):
    monkeypatch.setenv("MANYRUNS_DATA_DIR", "/new")

    assert env.get("DATA_DIR") == "/new"
    assert env.is_set("DATA_DIR")


def test_an_empty_value_is_a_value_and_not_an_absence(monkeypatch):
    """`MANYRUNS_X=""` is a decision meaning "off", not an unset variable meaning
    "unspecified" — otherwise there is no way to turn an override off without unsetting a
    variable the caller may not control."""
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", "")

    assert env.get("INLINE_IMAGES") == ""
    assert env.get("INLINE_IMAGES", "fallback") == ""
    assert env.is_set("INLINE_IMAGES")


def test_unset_returns_the_default(monkeypatch):
    monkeypatch.delenv("MANYRUNS_NOTHING", raising=False)

    assert env.get("NOTHING") is None
    assert env.get("NOTHING", "fallback") == "fallback"
    assert not env.is_set("NOTHING")


def _bypasses_accessor(source):
    """Dynamic reads must use env.get; only literal non-product keys may bypass it.

    Resolve import aliases and reject computed keys so a parameter or a named MANYRUNS
    key cannot hide the read on a different line from the prefix.
    """
    tree = ast.parse(source)
    os_names, environ_names, getenv_names = {'os'}, set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            os_names.update(a.asname or a.name for a in node.names if a.name == 'os')
        elif isinstance(node, ast.ImportFrom) and node.module == 'os':
            for alias in node.names:
                if alias.name == 'environ':
                    environ_names.add(alias.asname or alias.name)
                elif alias.name == 'getenv':
                    getenv_names.add(alias.asname or alias.name)

    def os_attr(node, attr):
        return (isinstance(node, ast.Attribute) and node.attr == attr
                and isinstance(node.value, ast.Name) and node.value.id in os_names)

    def environ(node):
        return (os_attr(node, 'environ')
                or isinstance(node, ast.Name) and node.id in environ_names)

    def unsafe(key):
        value = key.value if isinstance(key, ast.Constant) else None
        return not isinstance(value, str) or value.startswith('MANYRUNS_')

    lines = []
    for node in ast.walk(tree):
        key = None
        if isinstance(node, ast.Call):
            fn = node.func
            if (os_attr(fn, 'getenv')
                    or isinstance(fn, ast.Name) and fn.id in getenv_names
                    or isinstance(fn, ast.Attribute) and fn.attr in ('get', '__getitem__')
                    and environ(fn.value)):
                key = node.args[0] if node.args else next(
                    (kw.value for kw in node.keywords if kw.arg in ('key', 'name')), ast.Constant(None))
        elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and environ(node.value):
            key = node.slice
        elif isinstance(node, ast.Compare) and any(environ(n) for n in node.comparators):
            key = node.left
        if key is not None and unsafe(key):
            lines.append(node.lineno)
    return lines


def test_no_module_reads_the_environment_around_the_accessor():
    package = _package()
    offenders = [f"{p.relative_to(package)}:{n}"
                 for p in package.rglob('*.py') if p != package / 'env.py'
                 for n in _bypasses_accessor(p.read_text())]
    assert offenders == [], f"environment reads bypass manyruns.env: {offenders}"


@pytest.mark.parametrize('source', [
    'os.environ.get("MANYRUNS_X")', 'os.environ["MANYRUNS_X"]',
    'def read(key): return os.environ.get(key)',
    'def read(key): return os.environ[key]',
    'KEY = "MANYRUNS_X"\nos.environ.get(KEY)',
    'key = "TERM"\ndef read(key): return os.environ.get(key)',
    'import os as system\nsystem.getenv("MANYRUNS_X")',
    'from os import environ as values\nvalues.get(key)',
    'from os import getenv as read\nread(key)',
    'os.getenv(key="MANYRUNS_X")', '"MANYRUNS_X" in os.environ',
])
def test_parameterised_and_aliased_reads_cannot_evade_the_guard(source):
    assert _bypasses_accessor(source)


def test_other_libraries_environment_keys_and_writes_are_allowed():
    assert not _bypasses_accessor('os.environ.get("TERM")\nos.environ["MPLBACKEND"] = "Agg"')


@pytest.mark.parametrize('kind', ['recipe', 'dataset', 'metrics'])
def test_catalog_directory_overrides_use_the_shared_accessor(monkeypatch, tmp_path, kind):
    from manyruns import catalog

    calls = []

    def get(key):
        calls.append(key)
        return str(tmp_path)

    monkeypatch.setattr(env, 'get', get)
    assert getattr(catalog, f'{kind}_dir')() == tmp_path
    assert calls == [f'{kind.upper()}_DIR']


def test_the_pre_rename_prefix_lives_nowhere():
    """The retirement, executed rather than asserted in a changelog. `env.py` is scanned too:
    it is the file that used to be allowed to name the old prefix, so it is the one most
    likely to grow it back."""
    package = _package()

    offenders = sorted(
        f"{p.relative_to(package)}:{n}"
        for p in package.rglob("*.py")
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if _PRE_RENAME in line
    )

    assert offenders == [], f"the pre-rename environment prefix is back: {offenders}"
