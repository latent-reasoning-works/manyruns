"""The bundled MIOFlow recipes run on the tool route, and manyruns never imports MIOFlow.

Every tool-route test here uses the stub MIOFlow folder (`tests/fixtures/mioflow_tool/`): a copy
of the example manifest and a numpy-only adapter. It runs as a subprocess under the current
interpreter through `MANYRUNS_TOOL_MIOFLOW_PYTHON`, so no MIOFlow, torch, lock or network is
needed.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
STUB = Path(__file__).resolve().parent / "fixtures" / "mioflow_tool"
EXAMPLE = ROOT / "examples" / "tools" / "mioflow"


# ── 2. the stub MIOFlow folder ───────────────────────────────────────────────
@pytest.mark.source_checkout
def test_stub_manifest_matches_example():
    """The stub must not drift from the example it stands in for."""
    assert STUB.is_dir() and EXAMPLE.is_dir()
    stub = yaml.safe_load((STUB / "mioflow.yaml").read_text(encoding="utf-8"))
    example = yaml.safe_load((EXAMPLE / "mioflow.yaml").read_text(encoding="utf-8"))
    assert stub == example


@pytest.fixture(autouse=True)
def _no_tool_variables(monkeypatch):
    """Every test starts on a plain install: no tool folder and no tool search path."""
    for name in ("MANYRUNS_TOOL_DIR", "MANYRUNS_TOOL_PATH", "MANYRUNS_TOOL_MIOFLOW_PYTHON",
                 "MANYRUNS_RECIPE_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def stub_path(monkeypatch):
    """`$MANYRUNS_TOOL_PATH` is the stub MIOFlow folder, and the stub runs here."""
    import sys

    assert STUB.is_dir(), "stub MIOFlow folder moved"
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", str(STUB))
    monkeypatch.setenv("MANYRUNS_TOOL_MIOFLOW_PYTHON", sys.executable)
    return STUB


def _copy_stub(dest: Path) -> Path:
    import shutil

    assert STUB.is_dir()
    shutil.copytree(STUB, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


# ── 3. the tool search path ──────────────────────────────────────────────────
def test_bundled_tools_stay_visible(stub_path):
    from manyruns import toolchain

    tools = toolchain.discover_tools()
    assert "pyrovelocity" in tools and "mioflow" in tools


def test_lock_found_beside_manifest(tmp_path, monkeypatch):
    from manyruns import toolchain

    folder = _copy_stub(tmp_path / "mioflow_tool")
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", str(folder))
    monkeypatch.setenv("MANYRUNS_TOOL_HOME", str(tmp_path / "home"))
    lock = folder / "locks" / f"mioflow-0.1.14-{toolchain.platform_tag()}.lock"
    lock.parent.mkdir()
    lock.write_text("numpy==2.2.6\n", encoding="utf-8")

    assert toolchain.manifest_dir("mioflow") == folder
    assert toolchain.lock_path(toolchain.load_tool("mioflow")) == lock
    assert toolchain.resolve("mioflow")["status"] == "not-built"


def test_adapter_found_beside_manifest(stub_path):
    from manyruns import toolchain
    from manyruns.pipeline import external

    assert external._adapter_path(toolchain.load_tool("mioflow")) == STUB / "mioflow_adapter.py"


def test_duplicate_name_is_refused(tmp_path, monkeypatch):
    import os

    from manyruns import toolchain

    one = _copy_stub(tmp_path / "one")
    two = _copy_stub(tmp_path / "two")
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", os.pathsep.join([str(one), str(two)]))
    with pytest.raises(ValueError) as err:
        toolchain.load_tool("mioflow")
    assert str(one) in str(err.value) and str(two) in str(err.value)
    with pytest.raises(ValueError):
        toolchain.manifest_dir("mioflow")


def test_empty_path_entry_is_ignored(monkeypatch):
    import os

    from manyruns import toolchain

    assert STUB.is_dir()
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", os.pathsep + str(STUB) + os.pathsep)
    path = toolchain.tool_search_path()
    assert len(path) == 2
    assert path[0] == toolchain.tool_dir() and path[1] == STUB


# ── 4. a missing tool is an unmet fact ───────────────────────────────────────
MIOFLOW_RECIPES = ("cflows", "traced", "sandbox")


def test_recipes_stay_valid_on_plain_install():
    from manyruns import catalog

    for name in MIOFLOW_RECIPES:
        cfg = catalog._read(catalog.recipe_dir(), name, "recipe", catalog.discover_recipes())
        assert catalog.check_recipe(cfg, name) == [], name


def test_a_recipe_naming_an_undeclared_tool_stays_valid():
    from manyruns import catalog

    recipe = {"name": "r", "steps": [{"name": "x", "group": "latent", "tool": "nosuchtool"}]}
    assert catalog.check_recipe(recipe, "r") == []
    recipe["steps"][0]["tool"] = ""
    assert any("tool" in p for p in catalog.check_recipe(recipe, "r"))


def test_cflows_blocked_on_plain_install():
    from manyruns import catalog, vocab

    assert vocab.unmet(catalog.load_recipe("cflows"), "time-course") == frozenset({"tool:mioflow"})


def test_untooled_mioflow_blocked_at_plan_time():
    from manyruns import vocab

    recipe = {"steps": [{"name": "phate", "group": "latent"},
                        {"name": "mioflow", "group": "lightning"}]}
    assert vocab.unmet(recipe, "time-course") == frozenset({"tool:mioflow"})
    assert vocab.step_needs({"name": "mioflow", "group": "lightning"}) == frozenset({"tool:mioflow"})


def test_a_missing_tool_provides_nothing():
    from manyruns import vocab

    step = {"name": "mioflow", "group": "lightning", "tool": "mioflow"}
    assert vocab.step_needs(step) == frozenset({"tool:mioflow"})
    assert vocab.step_provides(step) == frozenset()


def test_the_tool_phrase_names_the_fix():
    from manyruns import catalog, narrate

    [why] = narrate.blocked(catalog.load_recipe("cflows"), "time-course")
    assert "mioflow" in why and "MANYRUNS_TOOL_PATH" in why and "examples/tools/mioflow" in why
    [other] = narrate.blocked(
        {"steps": [{"name": "x", "group": "latent", "tool": "nosuchtool"}]}, "manifold")
    assert "nosuchtool" in other and "MANYRUNS_TOOL_PATH" in other
    assert "examples/tools/mioflow" not in other


def _time_course_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "datasets"
    folder.mkdir()
    (folder / "tc.yaml").write_text(
        "name: tc\nhandle: {kind: path, ref: tc.npy}\nshape: time-course\n", encoding="utf-8")
    assert folder.is_dir()
    return folder


def _check(tmp_path: Path, env: dict) -> str:
    import os
    import subprocess
    import sys

    folder = _time_course_folder(tmp_path)
    clean = {k: v for k, v in os.environ.items()
             if k not in ("MANYRUNS_TOOL_DIR", "MANYRUNS_TOOL_PATH", "MANYRUNS_RECIPE_DIR")}
    cmd = [sys.executable, "-c", "import sys; from manyruns.app import main; sys.exit(main())",
           "check", "--dataset-dir", str(folder)]
    proc = subprocess.run(cmd,
                          capture_output=True, text=True, env={**clean, **env}, cwd=tmp_path,
                          timeout=300)
    return proc.stdout


def test_check_shows_why(tmp_path):
    out = _check(tmp_path, {})
    [line] = [ln for ln in out.splitlines() if ln.strip().startswith("WARN  cflows")]
    assert "mioflow" in line and "MANYRUNS_TOOL_PATH" in line


def test_declaring_the_tool_unblocks_it(stub_path):
    from manyruns import catalog, vocab

    assert vocab.unmet(catalog.load_recipe("cflows"), "time-course") == frozenset()


# ── 5. the recipes and the route removal ─────────────────────────────────────
def _x12():
    from model_seam_fixtures import IDS12, T12, X12

    return X12, IDS12, T12


def _step(out: dict, name: str) -> dict:
    [rec] = [s for s in out["steps"] if s["name"] == name]
    return rec


def test_every_bundled_mioflow_step_names_the_tool():
    from manyruns import catalog

    found = {}
    for name in catalog.discover_recipes():
        for step in catalog.load_recipe(name)["steps"]:
            if step.get("name") == "mioflow":
                found[name] = step
    assert sorted(found) == sorted(MIOFLOW_RECIPES)
    for name, step in found.items():
        assert step.get("tool") == "mioflow", name
        assert step.get("group") == "lightning" and step.get("params") == {}, name


@pytest.mark.source_checkout
def test_no_module_reaches_mioflow():
    import subprocess

    assert (ROOT / "manyruns").is_dir()
    proc = subprocess.run(
        ["git", "grep", "-n", "-E", r"manylatents\.algorithms\.lightning|import mioflow|from mioflow",
         "manyruns/"], cwd=ROOT, capture_output=True, text=True)
    assert proc.stdout == "", proc.stdout


def test_mioflow_module_keeps_only_the_cli_route():
    from manyruns.pipeline import mioflow

    for gone in ("_run_mioflow_experiment", "_resolve_kwargs", "trajectories_of"):
        assert not hasattr(mioflow, gone), gone
    assert callable(mioflow._lightning_cmd) and callable(mioflow._run_lightning_cli)


def test_untooled_mioflow_refused_with_fix(tmp_path):
    from manyruns.pipeline import runner

    X12, IDS12, T12 = _x12()
    out = runner.run_inproc(X12, {"name": "r", "steps": [{"name": "mioflow", "group": "lightning"}]},
                            tmp_path, labels=T12, label_kind="time", sample_ids=IDS12)
    rec = _step(out, "mioflow")
    assert rec["outcome"] == "skipped" and rec.get("unsupported") is True
    assert "tool: mioflow" in rec["detail"]


def test_ml_lightning_refuses_untooled_mioflow(tmp_path):
    from manyruns.pipeline import runner

    X12, _, _ = _x12()
    state = runner._new_state(X=X12)
    state["emb"] = X12
    with pytest.raises(runner._StepUnsupported) as err:
        runner._ml_lightning("mioflow", {}, state, {}, {"out_dir": tmp_path, "seed": 0})
    assert "tool: mioflow" in str(err.value) and "examples/tools/mioflow" in str(err.value)


def test_recipe_loads_without_mioflow_port():
    import subprocess
    import sys

    code = (
        "import importlib.abc, sys\n"
        "BLOCKED = 'manylatents.algorithms.lightning.mioflow'\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == BLOCKED or name.startswith(BLOCKED + '.'):\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import manyruns.pipeline.runner\n"
        "from manyruns import catalog\n"
        "assert catalog.load_recipe('cflows')['steps'][-1]['tool'] == 'mioflow'\n"
        "print('ok')\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def _mock_cflows(tmp_path, monkeypatch) -> dict:
    from manyruns import app
    from manyruns.pipeline import external

    def boom(*a, **k):
        raise AssertionError("the mock engine started a tool")

    monkeypatch.setattr(external.subprocess, "run", boom)
    return app.run_explorations(None, "scrna", recipe=app.load_recipe("cflows"), engine="mock",
                                out_dir=tmp_path)


def test_mock_cflows_starts_no_subprocess(stub_path, tmp_path, monkeypatch):
    out = _mock_cflows(tmp_path, monkeypatch)
    rec = _step(out, "mioflow")
    assert rec["outcome"] == "ok"
    assert "mock stand-in for tool mioflow" in rec["detail"]


def test_mock_ignores_whether_tool_is_declared(tmp_path, monkeypatch):
    out = _mock_cflows(tmp_path, monkeypatch)
    rec = _step(out, "mioflow")
    assert rec["outcome"] == "ok"
    assert "mock stand-in for tool mioflow" in rec["detail"]


# ── 6. the trajectories readouts ─────────────────────────────────────────────
def test_traced_legal_with_tool_and_time(stub_path):
    from manyruns import catalog, vocab

    assert vocab.unmet(catalog.load_recipe("traced"), "time-course") == frozenset()


def test_readout_tables_read_the_slot():
    from manyruns import vocab
    from manyruns.pipeline import probes

    assert probes.PROBE_READS == {"sample_trajectories": "trajectories"}
    assert vocab.STEP_NEEDS["sample_trajectories"] == ("trajectories",)
    assert "mioflow" not in vocab.STEP_PRODUCES


def test_probe_takes_the_array():
    import numpy as np

    from manyruns.pipeline import probes

    traj = np.zeros((4, 3, 2))
    traj[-1, :, 0] = [1.0, 2.0, 3.0]
    out = probes._probe_sample_trajectories(traj, {})
    assert out == {"n_steps": 4, "n_trajectories": 3, "dims": 2,
                   "median_displacement": 2.0, "max_displacement": 3.0}


def test_probe_skips_without_the_slot(tmp_path):
    from manyruns.pipeline import runner

    state = runner._new_state()
    with pytest.raises(runner._StepSkipped):
        runner._run_probe_step("sample_trajectories", {}, state, {},
                               {"out_dir": tmp_path, "seed": 0})


@pytest.fixture()
def traced_run(stub_path, tmp_path):
    from manyruns import catalog
    from manyruns.pipeline import runner

    X12, IDS12, T12 = _x12()
    out = runner.run_inproc(X12, catalog.load_recipe("traced"), tmp_path, labels=T12,
                            label_kind="time", sample_ids=IDS12, seed=0)
    return out, tmp_path


def test_probe_answers_from_slot(traced_run):
    out, folder = traced_run
    g = out["g_vector"]
    assert _step(out, "mioflow")["outcome"] == "ok", _step(out, "mioflow")
    assert g["sample_trajectories.n_steps"] == 4
    assert g["sample_trajectories.n_trajectories"] == 3
    assert g["sample_trajectories.dims"] == 3
    assert folder.is_dir()
    assert list(folder.rglob("mioflow.png")), sorted(p.name for p in folder.rglob("*.png"))


def test_embedding_is_not_replaced(traced_run):
    """The tool returns `trajectories` only, so the embedding after `mioflow` is phate's."""
    from manyruns import artifacts

    out, folder = traced_run
    assert folder.is_dir()
    assert _step(out, "mioflow")["produced"] is None
    manifest = artifacts.read_manifest(folder, out["run_id"])
    embs = sorted((v["index"], v["step"]) for v in manifest.values() if v["slot"] == "emb")
    assert embs and all(step != "mioflow" for _, step in embs)
    assert embs[-1][1] == "phate"   # the last embedding written, so the run's final one


def test_row_names_non_commercial_step(traced_run, tmp_path):
    import json

    from manyruns import store

    out, _ = traced_run
    index = store.append(out, out_dir=tmp_path / "outputs")
    [row] = [json.loads(line) for line in Path(index).read_text().splitlines()]
    assert row["non_commercial_steps"] == ["mioflow"]


# ── 7. time is declared, never invented ──────────────────────────────────────
def test_cflows_needs_real_time(stub_path):
    from manyruns import catalog, vocab

    assert vocab.unmet(catalog.load_recipe("cflows"), "manifold") == frozenset({"time"})


def test_discretize_time_provides_time():
    from manyruns import vocab

    assert vocab.STEP_PRODUCES["discretize_time"] == ("time",)


@pytest.fixture()
def inferred(stub_path, monkeypatch):
    folder = EXAMPLE / "recipes"
    assert folder.is_dir()
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(folder))
    from manyruns import catalog

    return catalog.load_recipe("cflows_inferred")


@pytest.mark.source_checkout
def test_inferred_recipe_declares_its_axis(inferred):
    from manyruns import catalog, vocab

    assert vocab.unmet(inferred, "manifold") == frozenset()
    assert inferred["id"] == "sc.traj.mioflow_inferred"
    assert inferred["claims"] == "single-trajectory"
    assert inferred["suits"] == ["manifold", "unknown"]
    assert [s["name"] for s in inferred["steps"]] == [
        "normalize", "transform", "pca", "phate", "dpt", "discretize_time", "mioflow"]
    assert inferred["steps"][-1]["tool"] == "mioflow"
    assert "pseudotime" in inferred["question"].lower()
    assert catalog.check_recipe(inferred, "cflows_inferred") == []


@pytest.mark.source_checkout
def test_discretize_time_keeps_real_time(inferred, tmp_path):
    import numpy as np

    from manyruns.pipeline import runner

    X12, IDS12, T12 = _x12()
    out = runner.run_inproc(X12, inferred, tmp_path, labels=T12, label_kind="time",
                            sample_ids=IDS12, seed=0)
    rec = _step(out, "discretize_time")
    assert rec["outcome"] == "skipped" and "real time labels are already present" in rec["detail"]
    assert _step(out, "mioflow")["outcome"] == "ok", _step(out, "mioflow")
    assert tmp_path.is_dir()
    [sent] = list(tmp_path.rglob("time.npy"))
    assert np.array_equal(np.load(sent, allow_pickle=False), T12)


# ── 8. auto-selection never picks a blocked recipe ───────────────────────────
def test_time_course_without_tool_gets_pseudotime():
    from manyruns import app

    name, why = app._choose_recipe(has_time_axis=True, conditions=None)
    assert name == "pseudotime"
    assert "cflows" in why and "mioflow" in why


def test_time_course_with_tool_gets_cflows(stub_path):
    from manyruns import app

    name, why = app._choose_recipe(has_time_axis=True, conditions=None)
    assert name == "cflows"
    assert "blocked" not in why


def test_a_step_issued_alone_keeps_its_tool():
    """A session issues a catalog step by name (`catalog.known_steps`). It must carry `tool:` as
    the recipe's step does, or a `mioflow` issued alone is refused for lacking it."""
    from manyruns import catalog

    assert catalog.known_steps()["mioflow"]["tool"] == "mioflow"
    assert "tool" not in catalog.known_steps()["phate"]
