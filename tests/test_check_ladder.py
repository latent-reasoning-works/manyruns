"""`manyruns check` reports whether the offered set leaves a rung of the ladder unroutable.

The check offer is the valid recipes that are not `blocked:` and are legal on at least one
declared dataset. A gap in the bundled catalogue is a FAIL; a gap in a `$MANYRUNS_RECIPE_DIR`
is a WARN, because a sweep's recipe directory does not have to cover the ladder.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

from manyruns import app, catalog, score

HEADING = "ladder (rungs the offered recipes can claim)"


def _check(dataset_dir=None):
    return app.cmd_check(SimpleNamespace(dataset_dir=dataset_dir), argparse.ArgumentParser())


def _last_line(out: str) -> str:
    return [ln for ln in out.splitlines() if ln.strip()][-1]


def _override(tmp_path, monkeypatch, arch: str):
    rdir, ddir = tmp_path / "recipes", tmp_path / "datasets"
    rdir.mkdir()
    ddir.mkdir()
    for name, claim in (("s0", "surface"), ("c1", "clusters"), ("t2", "single-trajectory"),
                        ("b3", "multi-branching")):
        (rdir / f"{name}.yaml").write_text(
            f"name: {name}\nclaims: {claim}\nsteps: [{{name: pca, group: latent}}]\n")
    (rdir / "arch.yaml").write_text(arch)
    (ddir / "m.yaml").write_text(
        "name: m\nhandle: {kind: manylatents, ref: swissroll}\nshape: manifold\n")
    assert rdir.is_dir() and ddir.is_dir()
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(rdir))
    return ddir


def test_a_blocked_rung_4_recipe_is_a_warning_in_an_override_directory(
        tmp_path, monkeypatch, capsys):
    ddir = _override(tmp_path, monkeypatch,
                     "name: arch\nclaims: archetypal\nsteps: [{name: aa, group: latent}]\n"
                     'blocked: "not available yet"\n')
    rc = _check(str(ddir))
    out = capsys.readouterr().out
    assert HEADING in out
    assert "  WARN  unroutable rungs: [4]" in out
    assert _last_line(out).endswith("unroutable rungs [4]")
    assert rc == 0


def test_a_recipe_legal_on_no_dataset_does_not_count(tmp_path, monkeypatch, capsys):
    ddir = _override(tmp_path, monkeypatch,
                     "name: arch\nclaims: archetypal\n"
                     "steps: [{name: separation, group: analysis}]\n")
    _check(str(ddir))
    assert "  WARN  unroutable rungs: [4]" in capsys.readouterr().out


def test_a_gap_in_the_bundled_catalogue_fails_the_check(monkeypatch, capsys):
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    monkeypatch.setattr(score, "unroutable_rungs", lambda offer=None, *, config_dir=None: [3])
    rc = _check()
    out = capsys.readouterr().out
    assert "  FAIL  unroutable rungs: [3]" in out
    assert _last_line(out).endswith("unroutable rungs [3]")
    assert rc == 1


def test_no_coverage_pass_no_ladder_line(monkeypatch, capsys):
    monkeypatch.setattr(catalog, "discover_recipes", lambda *a, **k: [])
    _check()
    assert "unroutable rungs" not in capsys.readouterr().out


def test_the_bundled_catalogue_passes(monkeypatch, capsys):
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    rc = _check()
    out = capsys.readouterr().out
    assert HEADING in out
    assert "  ok    unroutable rungs: none" in out
    assert rc == 0


def test_the_existing_coverage_test_keeps_its_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MANYRUNS_RECIPE_DIR", raising=False)
    (tmp_path / "swiss.yaml").write_text(
        "name: swiss\nhandle: {kind: manylatents, ref: swissroll}\nshape: manifold\n")
    assert tmp_path.is_dir()
    rc = _check(str(tmp_path))
    assert "  ok    unroutable rungs: none" in capsys.readouterr().out
    assert rc == 0
