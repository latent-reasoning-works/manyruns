"""Preservation gates: samples are declarations, with first-use bytes outside the package."""
from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

from manyruns import catalog


EXPECTED_DATASETS = {
    "arch_sharp", "arch_soft", "blob_loose", "blob_tight", "gaussian_blob", "pbmc3k",
    "saddlesurface", "swissroll", "synthetic_timecourse", "torus", "torus_noisy",
    "torus_tight", "tree_narrow", "tree_wide",
}


def test_packaged_catalog_keeps_all_fourteen_existing_declarations(monkeypatch):
    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)
    config_root = resources.files("manyruns") / "configs" / "dataset"
    assert config_root.is_dir()
    names = {item.name.removesuffix(".yaml") for item in config_root.iterdir()
             if item.name.endswith(".yaml")}
    assert names == EXPECTED_DATASETS
    assert len(names) == 14 and "tree8" not in names
    assert catalog.dataset_dir().is_dir()
    assert set(catalog.discover_datasets()) == names
    pbmc = catalog.load_dataset("pbmc3k")
    assert pbmc["handle"] == {
        "kind": "path", "ref": "data/pbmc3k_raw.h5ad", "provides": ["genes"],
        "sha256": "89a96f1beaa2dd83a687666d3f19a4513ac27a2a2d12581fcd77afed7ea653a1",
        "bytes": 5855727, "url": "https://exampledata.scverse.org/scanpy/pbmc3k_raw.h5ad",
    }
    assert pbmc["source"]["url"] != pbmc["handle"]["url"], "keep the source citation"
    assert catalog.load_dataset("synthetic_timecourse")["handle"]["ref"] == "synthetic:time-course"


def test_no_sample_data_is_tracked():
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    try:
        checkout = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                                  capture_output=True, text=True)
    except FileNotFoundError:
        pytest.skip("tracked sample check requires Git and a checkout")
    if checkout.returncode or Path(checkout.stdout.strip()).resolve() != root:
        pytest.skip("tracked sample check requires a Git checkout; source export has no index")
    tracked = subprocess.run(["git", "ls-files", "-z", "--", "*.h5ad"], cwd=root,
                             check=True, capture_output=True).stdout.decode().split("\0")
    # This 12-byte test stub was already tracked at 15e45ca; the plan's absolute "no .h5ad
    # tracked" assertion was false on its base. No real sample or new fixture is added.
    assert {name for name in tracked if name} == {"tests/fixtures/eb/matrix.h5ad"}


def test_no_sample_data_is_present_under_the_package():
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    assert (root / "tests/fixtures/eb/matrix.h5ad").stat().st_size == 12
    package = root / "manyruns"
    assert package.is_dir()
    assert list(package.rglob("*.h5ad")) == []
    assert not (package / "data").exists()


@pytest.mark.parametrize("bundled", [None, "sample.h5ad", "data"])
def test_distribution_checks_in_source_export(tmp_path, bundled):
    """A gitless export must still reject bundled bytes or a package data directory."""
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    export = tmp_path / "export"
    package = export / "manyruns"
    package.mkdir(parents=True)
    tests = export / "tests"
    tests.mkdir()
    exported_test = tests / "test_sample_distribution.py"
    shutil.copyfile(__file__, exported_test)
    fixture = tests / "fixtures" / "eb" / "matrix.h5ad"
    fixture.parent.mkdir(parents=True)
    shutil.copyfile(root / "tests/fixtures/eb/matrix.h5ad", fixture)
    if bundled == "data":
        (package / bundled).mkdir()
    elif bundled is not None:
        (package / bundled).write_bytes(b"sample bytes")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(exported_test), "-q", "-rs",
         "-p", "no:cacheprovider", "-k", "no_sample_data"],
        cwd=export, env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True, text=True,
    )
    assert result.returncode == (0 if bundled is None else 1), result.stdout + result.stderr
    if bundled is None:
        assert "1 passed, 1 skipped" in result.stdout
        assert "checkout" in result.stdout
    else:
        assert "AssertionError" in result.stdout
        assert "CalledProcessError" not in result.stdout


def test_build_configuration_keeps_data_out_of_wheels():
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    config = tomllib.loads((root / "pyproject.toml").read_text())
    build = config["tool"]["hatch"]["build"]
    assert "artifacts" not in build
    for target in build.get("targets", {}).values():
        assert "artifacts" not in target


@pytest.mark.source_checkout
def test_binary_configuration_keeps_data_out():
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    spec = ast.parse((root / "manyruns.spec").read_text())
    literals = [node.value for node in ast.walk(spec)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert "manyruns/configs" in literals, "the config layer must still ship"
    assert not any("manyruns/data" in value for value in literals)
