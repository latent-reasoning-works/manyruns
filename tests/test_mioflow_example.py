"""The MIOFlow example: upstream `mioflow==0.1.14`, unmodified, outside the wheel and the sdist.

No test here imports MIOFlow. The adapter tests put stub modules at `mioflow`, `torch` and
`anndata`, so the base CI job (no torch) runs them. The real call is checked out of band with
`manyruns tools verify mioflow` after each pin change.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from manyruns.pipeline import external
from model_seam_fixtures import T12, X12

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "tools" / "mioflow"


def test_manyruns_never_reaches_mioflow():
    code = ("import json, sys; import manyruns.pipeline.external, manyruns.toolchain, "
            "manyruns.vocab, manyruns.adapters.call_adapter; "
            "print(json.dumps(sorted(sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout
    loaded = json.loads(out)
    assert not [m for m in loaded if m.startswith("mioflow")]
    assert not [m for m in loaded if m.startswith("manylatents.algorithms.lightning")]


@pytest.mark.source_checkout
def test_wheel_holds_no_mioflow_file(tmp_path, monkeypatch):
    build = pytest.importorskip("hatchling.build", reason="requires the build backend")
    assert ROOT.is_dir() and EXAMPLE.is_dir()
    monkeypatch.chdir(ROOT)
    wheel = tmp_path / build.build_wheel(str(tmp_path))
    sdist = tmp_path / build.build_sdist(str(tmp_path))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    with tarfile.open(sdist) as archive:
        names += archive.getnames()
    assert names
    assert not [n for n in names if "mioflow_adapter" in n or "examples/" in n]


class _Fitted:
    trajectories = np.zeros((100, 5, 3))


class _StubMIOFlow:
    calls: list = []

    def __init__(self, adata, **kwargs):
        type(self).calls.append({"adata": adata, **kwargs})

    def fit(self):
        return _Fitted()


class _StubAnnData:
    def __init__(self, X=None, **kwargs):
        self.X, self.obs, self.obsm = X, {}, {}


@pytest.fixture()
def adapter(monkeypatch):
    assert EXAMPLE.is_dir(), "the example tool folder moved"
    _StubMIOFlow.calls = []
    monkeypatch.setitem(sys.modules, "mioflow", types.SimpleNamespace(MIOFlow=_StubMIOFlow))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=lambda s: None))
    monkeypatch.setitem(sys.modules, "anndata", types.SimpleNamespace(AnnData=_StubAnnData))
    spec = importlib.util.spec_from_file_location("mioflow_adapter",
                                                  EXAMPLE / "mioflow_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(workdir: Path, *, time=True, params=None) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    np.save(workdir / "input.npy", X12)
    req = {"tool": "mioflow", "version": "0.1.14", "params": params or {}, "seed": 0,
           "input": str(workdir / "input.npy"), "result": str(workdir / "result.json"),
           "workdir": str(workdir), "result_schema": "result-v1", "n_cells": 12,
           "input_dim": 3}
    if time:
        np.save(workdir / "time.npy", T12)
        req["time"] = str(workdir / "time.npy")
    path = workdir / "request.json"
    path.write_text(json.dumps(req))
    return path


@pytest.mark.source_checkout
def test_adapter_declines_without_time_axis(adapter, tmp_path):
    adapter.main(str(_request(tmp_path, time=False)))
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] is False and "time" in result["reason"]
    assert _StubMIOFlow.calls == []
    assert external.check_result(result) == []


@pytest.mark.source_checkout
def test_adapter_writes_valid_trajectories(adapter, tmp_path):
    adapter.main(str(_request(tmp_path)))
    result = json.loads((tmp_path / "result.json").read_text())
    assert external.check_result(result, base=tmp_path, n_cells=12, input_dim=3,
                                 produces=["trajectories"]) == []
    (call,) = _StubMIOFlow.calls
    assert call["gaga_model"] is None
    assert call["obs_time_key"] == "time_bin" and call["use_cuda"] is False
    np.testing.assert_array_equal(call["adata"].obsm["X_pca"], X12)
    np.testing.assert_array_equal(call["adata"].obs["time_bin"], T12)
    assert result["scalars"] == {"n_bins": 100, "n_trajectories": 5}


@pytest.mark.source_checkout
def test_adapter_refuses_parameter_outside_list(adapter, tmp_path):
    adapter.main(str(_request(tmp_path, params={"use_cuda": True})))
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] is False and "use_cuda" in result["reason"]
    assert _StubMIOFlow.calls == []


@pytest.mark.source_checkout
def test_adapter_imports_only_the_standard_library_and_numpy_at_module_level():
    assert EXAMPLE.is_dir()
    source = (EXAMPLE / "mioflow_adapter.py").read_text()
    assert source.splitlines()[0] == "# SPDX-License-Identifier: Apache-2.0"
    top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert sorted(top) == ["from pathlib import Path", "import json", "import numpy as np",
                           "import sys"]
