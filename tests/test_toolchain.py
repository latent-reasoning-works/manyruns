"""The `tool:` seam — calling a tool manyruns cannot install, without installing it.

**THE WHOLE PROTOCOL IS EXERCISED HERE WITH NO HEAVY ENVIRONMENT.** The real case (pyrovelocity)
needs a 2.4 GB interpreter that CI will never build, and a mechanism that can only be tested
where it is expensive is a mechanism nobody tests. So these use a STUB tool: a manifest in a
tmp `$MANYRUNS_TOOL_DIR`, an adapter beside it, and the running interpreter as its "env" via
the `MANYRUNS_TOOL_<NAME>_PYTHON` override. Same resolve, same request JSON, same subprocess,
same contract check, same provenance — a different amount of maths at the far end.

What that leaves untested is stated rather than implied: whether a REAL lock reproduces a REAL
environment. That is `toolchain.build` + `selftest`, verified out of band and recorded in
`docs/sidecars/pyrovelocity.md`; no CI job can afford it.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from manyruns import toolchain
from manyruns.pipeline import external, runner, steps

ADAPTER = '''
import json, sys
def main(p):
    req = json.loads(open(p).read())
    import numpy as np
    X = np.load(req["input"])
    try:
        import anndata as ad
    except ImportError:
        json.dump({"ok": False, "reason": "no anndata"}, open(req["result"], "w")); return 0
    # `truncate` exists so a test can make a WELL-BEHAVED tool misbehave in the one way that
    # would otherwise be silent: returning rows that are not this run\'s cells.
    keep = int(req["params"].get("truncate") or X.shape[0])
    X = X[:keep]
    out = ad.AnnData(X=X.astype("float32"))
    out.obsm["stubfield"] = (X[:, :2] * float(req["params"].get("scale", 1.0))).astype("float32")
    out.obs["stub_sd"] = np.full(X.shape[0], 0.5)
    out.write_h5ad(req["output"])
    json.dump({"ok": True, "scalars": {"rows": int(X.shape[0])}}, open(req["result"], "w"))
    return 0
if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
'''

MANIFEST = textwrap.dedent("""
    name: stub
    version: "1.0"
    produces: [stubfield]
    python: "3.12"
    requires: [stub==1.0]
    entrypoint: stub_adapter.py
    selftest: "pass"
    input:  {kind: state_matrix}
    output: {obsm: [stubfield], obs: [stub_sd]}
    params: {scale: 1.0}
""")


@pytest.fixture()
def stub(tmp_path, monkeypatch):
    d = tmp_path / "tools"
    (d / "locks").mkdir(parents=True)
    (d / "stub.yaml").write_text(MANIFEST)
    (d / "stub_adapter.py").write_text(ADAPTER)
    monkeypatch.setenv("MANYRUNS_TOOL_DIR", str(d))
    monkeypatch.setenv("MANYRUNS_TOOL_HOME", str(tmp_path / "envs"))
    return d


def _lock(d, body="stub==1.0 --hash=sha256:deadbeef\n"):
    cfg = toolchain.load_tool("stub")
    p = toolchain.lock_path(cfg)
    p.write_text(body)
    return p


def test_a_manifest_with_no_selftest_is_refused():
    """A lock nobody executed is not a working pin — measured: pyrovelocity 0.4.5 installs with
    exit 0 and fails to import. The manifest may not skip the check that catches that."""
    bad = toolchain.check_tool({"name": "x", "version": "1", "requires": ["x==1"],
                                "entrypoint": "a.py", "input": {"kind": "state_matrix"}}, "x")
    assert any("selftest" in p for p in bad), bad


def test_a_manifest_with_no_version_is_refused():
    bad = toolchain.check_tool({"name": "x", "requires": ["x==1"], "entrypoint": "a.py",
                                "selftest": "pass", "input": {"kind": "state_matrix"}}, "x")
    assert any("version" in p for p in bad), bad


def test_resolve_reports_every_state_and_never_builds(stub, tmp_path):
    """`resolve` is pure. The realized env for one real tool is 2.4 GB — a step that downloads
    that mid-run is an outage, so nothing on the run path may create anything."""
    r = toolchain.resolve("stub")
    assert r["status"] == "no-lock" and r["python"] is None
    _lock(stub)
    r = toolchain.resolve("stub")
    assert r["status"] == "not-built"
    assert "manyruns tools install stub" in r["fix"]
    assert not (tmp_path / "envs").exists(), "resolve() created something"


def test_the_lock_digest_is_the_identity_not_the_version(stub):
    """Two runs are comparable when they name the same lock, not the same version string —
    re-locking for a patched transitive dep IS a different tool."""
    _lock(stub, "stub==1.0 --hash=sha256:aaa\n")
    first = toolchain.resolve("stub")["digest"]
    _lock(stub, "stub==1.0 --hash=sha256:bbb\n")   # same version, different closure
    assert toolchain.resolve("stub")["digest"] != first


def test_an_unbuilt_tool_makes_the_step_unsupported_with_the_fix(stub, tmp_path):
    """`_StepUnsupported`, not a plain decline: the record does not describe the analysis the
    recipe asked for, so the RUN is not `ok` (manyruns#66)."""
    _lock(stub)
    state = runner._new_state(X=__import__("numpy").zeros((4, 3)))
    with pytest.raises(steps._StepUnsupported) as e:
        external.run_external_step("stub", {}, state, {}, {"out_dir": tmp_path})
    assert "manyruns tools install stub" in str(e.value)


def test_the_round_trip_merges_the_field_and_records_the_environment(stub, tmp_path, monkeypatch):
    """THE PROTOCOL, end to end: request JSON out, `.h5ad` back, contract checked, field merged,
    environment recorded."""
    import numpy as np

    pytest.importorskip("anndata")
    monkeypatch.setenv("MANYRUNS_TOOL_STUB_PYTHON", sys.executable)
    _lock(stub)
    X = np.arange(24, dtype=float).reshape(8, 3)
    state, g = runner._new_state(X=X), {}
    external.run_external_step("stub", {"scale": 2.0}, state, g, {"out_dir": tmp_path, "seed": 1})

    assert state["stubfield"].shape == (8, 2)
    np.testing.assert_allclose(state["stubfield"], X[:, :2] * 2.0)
    assert g["stub.rows"] == 8
    assert g["stub.tool_version"] == "1.0"
    # An operator-supplied interpreter may be perfect and the record still cannot vouch for it.
    assert g["stub.tool_reproducible"] is False
    req = json.loads((Path(tmp_path) / "tool-stub" / "request.json").read_text())
    assert req["params"]["scale"] == 2.0 and req["seed"] == 1


def test_a_field_that_is_not_indexed_to_this_run_is_refused_not_merged(stub, tmp_path, monkeypatch):
    """The one failure this seam must never permit: every per-cell number attributed to the
    wrong cell, silently. The stub is asked to hand back 5 rows for an 8-cell run."""
    import numpy as np

    pytest.importorskip("anndata")
    monkeypatch.setenv("MANYRUNS_TOOL_STUB_PYTHON", sys.executable)
    _lock(stub)
    state, g = runner._new_state(X=np.zeros((8, 3))), {}
    with pytest.raises(steps._StepSkipped) as e:
        external.run_external_step("stub", {"truncate": 5}, state, g, {"out_dir": tmp_path})
    assert "not indexed to this run" in str(e.value)
    assert state.get("stubfield") is None, "a refused field must not be half-merged"
    # The environment is still recorded: a run that ATTEMPTED a tool and was refused is part of
    # the record, or the record only describes successes.
    assert g["stub.tool_version"] == "1.0"


def test_a_source_file_tool_is_refused_after_a_narrowing(tmp_path):
    """A tool reading the ORIGINAL file after a filter reports on cells this run rejected."""
    import numpy as np

    cfg = {"name": "t", "input": {"kind": "source_file", "layers": ["spliced"]}}
    state = runner._new_state(counts=np.zeros((10, 4)))
    state["rows"] = np.arange(3)               # narrowed
    with pytest.raises(steps._StepSkipped) as e:
        external._input_path(cfg, state, {"source_path": str(tmp_path / "x.h5ad")}, tmp_path)
    assert "already narrowed" in str(e.value)


def test_a_tool_step_keeps_its_group_so_the_calculus_is_unchanged():
    """`tool:` selects an EXECUTOR. It is not a group and not an engine — the fact a step
    produces is still declared by its group and name, so `unmet` needs no knowledge of it."""
    from manyruns import vocab

    assert vocab.STEP_PRODUCES["pyrovelocity"] == ("velocity",)
    recipe = {"steps": [{"name": "pyrovelocity", "group": "lightning", "tool": "pyrovelocity"},
                        {"name": "velocity_field", "group": "analysis"}]}
    # The tool's own need is the DATA it cannot run without, and it is refused at plan time on
    # anything that does not carry it — the gate, not an accident of ordering.
    assert vocab.unmet(recipe, "manifold") == frozenset({"splicing"})
    # Given data that does carry it, the pair composes with nothing outstanding: the tool
    # produces `velocity` and the readout consumes it, which is the ORDER being checked here.
    assert vocab.unmet(recipe, "manifold", handle={"provides": ["splicing"]}) == frozenset()
    # Swapped, it is refused — the readout cannot precede the step that makes the field.
    swapped = {"steps": list(reversed(recipe["steps"]))}
    assert "velocity" in vocab.unmet(swapped, "manifold", handle={"provides": ["splicing"]})


def test_the_bundled_manifests_are_valid_and_declare_a_licence():
    """A tool runs as a separate program partly BECAUSE of its licence (pyrovelocity is AGPL),
    so the manifest records it — and every bundled manifest must load."""
    for name in toolchain.discover_tools():
        cfg = toolchain.load_tool(name)
        assert cfg.get("licence"), f"{name} declares no licence"
        assert cfg.get("source"), f"{name} cites no source"


def test_a_state_frame_tool_sees_the_NARROWED_run_not_the_file(tmp_path):
    """THE TEST THIS WHOLE FRAME CHANGE EXISTS FOR.

    With the frame carrying one matrix, a velocity tool had to read the ORIGINAL file, and
    `external._input_path` refused it the moment the run narrowed — correctly, because the file
    still holds the cells a filter removed. The cost was that the ordinary scRNA order died:
    measured, `filter_cells -> pyrovelocity -> velocity_field` reported `dropped 40` and then
    two declined steps, so the whole recipe collapsed after its first line.

    `state_frame` writes the run's CURRENT VIEW instead, layers included. The tool sees the
    survivors and nothing else, and the ordinary order composes.
    """
    ad = pytest.importorskip("anndata")
    import numpy as np

    from manyruns.pipeline import frame as _frame

    n, g = 20, 6
    counts = np.arange(n * g, dtype="float32").reshape(n, g)
    layers = {"spliced": counts * 2, "unspliced": counts * 3}
    state = runner._new_state(counts=counts, genes=np.array([f"g{i}" for i in range(g)]),
                              layers=layers)

    keep = np.zeros(n, bool)
    keep[:7] = True                       # a filter drops 13 of 20
    state["rows"] = _frame.narrow(state["rows"], keep)

    cfg = {"name": "t", "input": {"kind": "state_frame", "layers": ["spliced", "unspliced"]}}
    written = external._input_path(cfg, state, {}, tmp_path)

    out = ad.read_h5ad(written)
    assert out.shape == (7, g), "the tool was handed cells this run had already dropped"
    np.testing.assert_allclose(np.asarray(out.X), counts[:7])
    np.testing.assert_allclose(np.asarray(out.layers["spliced"]), counts[:7] * 2)
    np.testing.assert_allclose(np.asarray(out.layers["unspliced"]), counts[:7] * 3)


def test_a_state_frame_tool_is_refused_when_the_run_carries_no_such_layer(tmp_path):
    """Declared-only cuts both ways: a tool that needs `spliced` and a run that loaded none is
    a refusal with the fix in it, not a crash inside somebody else's interpreter."""
    pytest.importorskip("anndata")
    import numpy as np

    state = runner._new_state(counts=np.zeros((4, 3), dtype="float32"))
    cfg = {"name": "t", "input": {"kind": "state_frame", "layers": ["spliced"]}}
    with pytest.raises(steps._StepSkipped) as e:
        external._input_path(cfg, state, {}, tmp_path)
    assert "needs the ['spliced'] layer(s)" in str(e.value)
    assert "layers: ['spliced']" in str(e.value), "the refusal must say how to fix it"
