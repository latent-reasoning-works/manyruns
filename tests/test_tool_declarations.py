import copy

import manykinds.params
import numpy as np
import pytest

from manyruns import toolchain
from manyruns.pipeline import external, runner, steps
from tool_stubs import FRAME_ADAPTER, make_tool

SCALE = {
    "legal": {
        "type": "float",
        "min": 0,
        "min_open": False,
        "max": None,
        "max_open": False,
        "choices": [],
        "also": [],
    },
    "default": 1.0,
    "sensible": None,
    "ladder": None,
    "limits": [],
    "role": "knob",
    "precomputed": False,
    "cost": None,
    "meaning": "multiplies the returned field",
    "more": None,
    "less": None,
}
DECL = {
    "schema_version": 1,
    "method": "decl",
    "group": "tool",
    "origin": "test",
    "params": {"scale": SCALE},
}


def _tool(tmp_path, monkeypatch, decl=DECL, params=None):
    make_tool(
        tmp_path,
        monkeypatch,
        "decl",
        {
            "input": {"kind": "state_frame"},
            "output": {"obsm": ["field"]},
            "produces": ["field"],
            "params": params if params is not None else {"scale": 1.0},
            "declaration": decl,
        },
        FRAME_ADAPTER,
    )
    return toolchain.load_tool("decl")


def test_a_valid_declaration_passes(tmp_path, monkeypatch):
    assert toolchain.check_tool(_tool(tmp_path, monkeypatch), "decl") == []


def test_a_declaration_for_another_method_is_refused(tmp_path, monkeypatch):
    bad = copy.deepcopy(DECL) | {"method": "other"}
    with pytest.raises(ValueError, match="method"):
        _tool(tmp_path, monkeypatch, decl=bad)


def test_a_default_that_disagrees_with_the_declaration_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="scale"):
        _tool(tmp_path, monkeypatch, params={"scale": 2.0})


def test_an_illegal_step_value_is_refused_before_the_tool_starts(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    _tool(tmp_path, monkeypatch)
    X = np.ones((4, 2))
    state = runner._new_state(X=X, counts=X, genes=np.array(["g0", "g1"]))
    with pytest.raises(steps._StepUnsupported, match="scale"):
        external.run_external_step(
            "decl", {"scale": -1.0}, state, {}, {"out_dir": tmp_path / "run", "seed": 0}
        )
    assert not (tmp_path / "run" / "tool-decl" / "request.json").exists()


def test_every_bundled_declaration_is_valid():
    assert toolchain.tool_dir().is_dir()
    for name in toolchain.discover_tools():
        cfg = toolchain.load_tool(name)
        if cfg.get("declaration") is not None:
            assert manykinds.params.check_declaration(cfg["declaration"]) == [], name


def test_pyrovelocity_declares_its_parameters():
    decl = toolchain.load_tool("pyrovelocity")["declaration"]
    # Every key the adapter reads is declared, or a legal override is refused.
    assert set(decl["params"]) == {"max_epochs", "num_samples", "n_neighbors"}
    assert decl["params"]["n_neighbors"]["default"] == 30
    assert manykinds.params.check(decl, {"n_neighbors": 15}, n_samples=100, n_features=50) == []
    assert manykinds.params.check(decl, {"n_neighbors": 1}, n_samples=100, n_features=50)
    assert decl["params"]["max_epochs"]["default"] == 1500


@pytest.mark.parametrize(
    "change,match",
    [
        ({"group": "latent"}, "group"),
        ({"schema_version": 2}, "schema_version"),
        ({"params": []}, "params"),
    ],
)
def test_invalid_declarations_are_refused(tmp_path, monkeypatch, change, match):
    with pytest.raises(ValueError, match=match):
        _tool(tmp_path, monkeypatch, decl=copy.deepcopy(DECL) | change)


def test_undeclared_manifest_parameter_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unknown"):
        _tool(tmp_path, monkeypatch, params={"unknown": 1})


def test_only_step_params_are_checked_with_input_dimensions(tmp_path, monkeypatch):
    decl = copy.deepcopy(DECL)
    decl["params"]["scale"]["role"] = "fixed"
    cfg = _tool(tmp_path, monkeypatch, decl=decl)
    cfg["input"]["kind"] = "state_matrix"
    monkeypatch.setattr(toolchain, "resolve", lambda _: {"status": "overridden", "cfg": cfg})
    state = runner._new_state(X=np.ones((4, 2)), emb=np.ones((4, 3)))
    calls = []
    check = manykinds.params.check

    def capture(doc, values, **facts):
        calls.append((values, facts))
        return check(doc, values, **facts)

    monkeypatch.setattr(manykinds.params, "check", capture)

    class RequestReached(Exception):
        pass

    def stop(*args, **kwargs):
        raise RequestReached

    monkeypatch.setattr(external, "_write_request", stop)
    with pytest.raises(RequestReached):
        external.run_external_step("decl", {}, state, {}, {"out_dir": tmp_path})
    assert calls == [({}, {"n_samples": 4, "n_features": 3})]


def test_all_step_problems_are_reported(tmp_path, monkeypatch):
    _tool(tmp_path, monkeypatch)
    state = runner._new_state(X=np.ones((4, 2)), counts=np.ones((4, 2)))
    with pytest.raises(steps._StepUnsupported) as exc:
        external.run_external_step(
            "decl", {"scale": -1, "unknown": 2}, state, {}, {"out_dir": tmp_path}
        )
    assert "scale" in str(exc.value)
    assert "unknown" in str(exc.value)
    assert not (tmp_path / "tool-decl" / "request.json").exists()


@pytest.mark.parametrize("value", [True, 1.0])
def test_integer_manifest_default_rejects_equal_values_of_wrong_type(tmp_path, monkeypatch, value):
    decl = copy.deepcopy(DECL)
    decl["params"]["scale"]["legal"]["type"] = "int"
    decl["params"]["scale"]["default"] = 1
    with pytest.raises(ValueError, match="scale"):
        _tool(tmp_path, monkeypatch, decl=decl, params={"scale": value})


@pytest.mark.parametrize("params", [{"scale": -1.0}, {"unknown": 2}])
def test_precomputed_parameters_are_refused_before_results_are_read(tmp_path, monkeypatch, params):
    from model_seam_fixtures import GOLDEN
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(GOLDEN / "golden_precomputed.yaml"))
    cfg["declaration"] = copy.deepcopy(DECL) | {"method": "golden_precomputed"}
    cfg["params"] = {"scale": 1.0}
    assert toolchain.check_tool(cfg, "golden_precomputed") == []
    monkeypatch.setattr(toolchain, "resolve", lambda _: {"status": "precomputed", "cfg": cfg})

    def refuse_read(*args, **kwargs):
        pytest.fail("precomputed results were read before parameter validation")

    monkeypatch.setattr(external, "_run_precomputed", refuse_read)
    state = runner._new_state(X=np.ones((4, 2)))
    with pytest.raises(steps._StepUnsupported, match=next(iter(params))):
        external.run_external_step("golden_precomputed", params, state, {}, {"out_dir": tmp_path})


def test_precomputed_dimension_limit_uses_the_embedding_when_the_gene_selection_is_empty(
    tmp_path, monkeypatch
):
    from model_seam_fixtures import GOLDEN
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(GOLDEN / "golden_precomputed.yaml"))
    decl = copy.deepcopy(DECL) | {"method": "golden_precomputed"}
    decl["params"]["scale"]["limits"] = [{"fact": "n_features", "op": "le", "offset": 0}]
    cfg["declaration"] = decl
    cfg["params"] = {"scale": 1.0}
    assert toolchain.check_tool(cfg, "golden_precomputed") == []
    monkeypatch.setattr(toolchain, "resolve", lambda _: {"status": "precomputed", "cfg": cfg})
    monkeypatch.setattr(external, "_run_precomputed", lambda *a, **k: None)
    # An embedding-only state (no counts, no gene axis): the limit must read the embedding's
    # own 3 columns, not the empty gene selection `runner._new_state` seeds for it.
    state = runner._new_state(emb=np.ones((4, 3)))
    external.run_external_step(
        "golden_precomputed", {"scale": 3.0}, state, {}, {"out_dir": tmp_path}
    )
    with pytest.raises(steps._StepUnsupported, match="scale"):
        external.run_external_step(
            "golden_precomputed", {"scale": 4.0}, state, {}, {"out_dir": tmp_path}
        )


def test_precomputed_dimension_limit_uses_the_embedding_not_the_wider_counts(tmp_path, monkeypatch):
    from model_seam_fixtures import GOLDEN
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(GOLDEN / "golden_precomputed.yaml"))
    decl = copy.deepcopy(DECL) | {"method": "golden_precomputed"}
    decl["params"]["scale"]["limits"] = [{"fact": "n_features", "op": "le", "offset": 0}]
    cfg["declaration"] = decl
    cfg["params"] = {"scale": 1.0}
    assert toolchain.check_tool(cfg, "golden_precomputed") == []
    monkeypatch.setattr(toolchain, "resolve", lambda _: {"status": "precomputed", "cfg": cfg})
    monkeypatch.setattr(external, "_run_precomputed", lambda *a, **k: None)
    # Counts are 5 columns wide, the embedding only 3: the limit must read the embedding, not
    # the gene selection's wider count — a value legal only against the counts is refused.
    state = runner._new_state(X=np.ones((4, 5)), counts=np.ones((4, 5)), emb=np.ones((4, 3)))
    external.run_external_step(
        "golden_precomputed", {"scale": 3.0}, state, {}, {"out_dir": tmp_path}
    )
    with pytest.raises(steps._StepUnsupported, match="scale"):
        external.run_external_step(
            "golden_precomputed", {"scale": 5.0}, state, {}, {"out_dir": tmp_path}
        )


def test_source_file_parameters_use_source_dimensions_after_gene_filtering(tmp_path, monkeypatch):
    ad = pytest.importorskip("anndata")
    cfg = _tool(tmp_path, monkeypatch)
    cfg["input"]["kind"] = "source_file"
    monkeypatch.setattr(toolchain, "resolve", lambda _: {"status": "overridden", "cfg": cfg})
    source = tmp_path / "source.h5ad"
    X = np.ones((4, 5))
    ad.AnnData(X=X).write_h5ad(source)
    state = runner._new_state(X=X, counts=X)
    state["cols"] = np.array([0, 1])
    calls = []
    check = manykinds.params.check

    def capture(doc, values, **facts):
        calls.append((values, facts))
        return check(doc, values, **facts)

    monkeypatch.setattr(manykinds.params, "check", capture)

    class RequestReached(Exception):
        pass

    def stop(*args, **kwargs):
        raise RequestReached

    monkeypatch.setattr(external, "_write_request", stop)
    with pytest.raises(RequestReached):
        external.run_external_step(
            "decl", {}, state, {}, {"out_dir": tmp_path, "source_path": source}
        )
    assert calls == [({}, {"n_samples": 4, "n_features": 5})]
