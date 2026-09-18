"""Offline regressions for the filmed CLI and the provenance it displays."""
from pathlib import Path

import pytest


def _setup(source, monkeypatch):
    from manyruns import app

    monkeypatch.setattr(app, "write_project", lambda *a, **k: Path("project.yaml"))
    parser = app._build_parser()
    args = parser.parse_args(["run", source, "--recipe", "embed", "--project", "camera",
                              "--engine", "mock"])
    return app._setup_project(args, parser)


def test_catalog_file_and_path_construct_the_same_project(tmp_path, monkeypatch):
    from manyruns import narrate, shell

    source = tmp_path / "counts.h5ad"
    source.touch()
    monkeypatch.setattr(shell, "dataset_ref_path", lambda ref: source)
    monkeypatch.setattr(narrate, "read_data", lambda *a, **k: narrate.Observation(shape="unknown"))
    named = _setup("pbmc3k", monkeypatch)
    direct = _setup(str(source), monkeypatch)
    assert named["dataset_name"] == "pbmc3k"
    assert direct["dataset_name"] is None
    # The resolved data and recipe agree; only explicit catalog selection has a name.
    assert {**named, "dataset_name": None} == direct


def test_generator_keeps_its_engine_reference(monkeypatch):
    result = _setup("swissroll", monkeypatch)
    assert result["dataset"] == "swissroll"
    assert result["data_folder"] is None


def test_unknown_source_refuses_with_search_and_remedy(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        _setup("no-such-camera-data", monkeypatch)
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert "local path and a catalog handle" in message
    assert "Supply an existing data path" in message


@pytest.mark.parametrize("steps,cancelled,code,word", [
    ([{"outcome": "error"}] * 4, False, 1, "4 errors"),
    ([{"outcome": "skipped", "unsupported": True}], False, 1, "1 unsupported"),
    ([{"outcome": "skipped"}], False, 0, "1 benign skips"),
    ([{"outcome": "ok"}, {"outcome": "reported"}], False, 0, "2 successful"),
    ([], True, 130, "cancelled"),
])
def test_cli_exit_and_record_follow_outcomes(steps, cancelled, code, word, tmp_path, monkeypatch, capsys):
    from manyruns import app, modes, shell, store

    result = {"steps": steps, "cancelled": cancelled, "ok": True, "run_id": "camera"}
    recorded = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(modes, "run", lambda *a, **k: result)
    monkeypatch.setattr(app, "write_summary", lambda *a, **k: "summary.md")
    monkeypatch.setattr(store, "append", lambda row, **k: recorded.append(row) or "runs.jsonl")
    monkeypatch.setattr(shell, "render_result", lambda *a: None)
    proj = {"engine": "mock", "project": "camera", "modality": "unknown", "recipe": {}}
    assert app._explore_project(proj) == code
    assert word in capsys.readouterr().out
    assert recorded == [result], "even failures must reach the run store"


def test_tui_headline_counts_successes_not_errors():
    from manyruns.tui.run import RunScreen
    from manyruns.tui.state import RunFeed

    feed = RunFeed({"steps": [{"name": "pca"}]})
    feed.records = [{"index": 0, "name": "pca", "outcome": "error"}]
    screen = RunScreen.__new__(RunScreen)
    screen.heading, screen.error, screen.results = "camera", None, {"steps": feed.records}
    heading = screen._heading_text(feed.rows())
    assert "0 of 1 steps successful" in heading
    assert "failed" in heading


def test_engine_scores_are_step_measurements_with_final_absence_separate(monkeypatch, tmp_path):
    # The base-install job runs this suite with NO compute stack, so the ecosystem is skipped
    # rather than imported (CI's own note: "Tests importorskip / lazy-import the ecosystem, so
    # the manyruns seam runs standalone"). The provenance rule under test is manyruns's; the
    # engine is only the source of the scores it classifies.
    api = pytest.importorskip("manylatents.api")
    import numpy as np
    from manyruns import narrate
    from manyruns.measurements import annotate
    from manyruns.pipeline import runner

    monkeypatch.setattr(api, "run", lambda **k: {"embeddings": np.ones((4, 2)),
                        "scores": {"trustworthiness": .9817, "continuity": None,
                                   "continuity_note": "MeasurementUnavailable: no reference"}})
    monkeypatch.setattr(runner._io, "_save_scatter", lambda *a, **k: None)
    g = {"trustworthiness": None, "trustworthiness_note": "no ambient matrix", "pca.n_components": 2}
    annotate(g, "pca.n_components", kind="setting", stage="pca configuration")
    runner._ml_latent("pca", {}, {"emb": None}, g,
                      {"seed": 0, "array": np.ones((4, 3)), "out_dir": tmp_path, "plots": []})
    sections = narrate.geometry_sections(g)
    settings = next(s for s in sections if s.title == "settings")
    assert [r.label for r in settings.rows] == ["pca.n_components"]
    stage = next(s for s in sections if s.title == "step measurements")
    assert stage.note == "pca engine evaluation"
    assert stage.rows[0].value == "0.9817"
    assert "MeasurementUnavailable" in stage.rows[1].value
    final = next(s for s in sections if s.title == "final measurements")
    assert final.rows[0].value == "no ambient matrix"


def test_legacy_dotted_values_do_not_claim_to_be_settings():
    from manyruns.narrate import geometry_sections

    sections = geometry_sections({"pca.trustworthiness": .9817, "filter_cells.dropped": 12})
    assert all(s.title != "settings" for s in sections)
    assert sections[0].note == "stage unrecorded (legacy)"


@pytest.mark.parametrize("operation", ["filter_cells", "filter_genes", "filter_mito",
                                      "detect_doublets", "normalize", "transform", "looks_like_counts"])
def test_lazy_optional_dependency_has_remedy_and_original_cause(operation, monkeypatch):
    from types import SimpleNamespace
    from manyruns.pipeline import prep

    missing = ModuleNotFoundError("No module named 'scanpy'", name="scanpy")

    def lazy(*args, **kwargs):
        raise missing

    monkeypatch.setattr(prep, "_ops", lambda: SimpleNamespace(**{operation: lazy}))
    with pytest.raises(ImportError, match=r"uv sync --extra harness --extra dev") as exc:
        prep._call(operation, None)
    assert exc.value.__cause__ is missing
    assert "No module named" not in str(exc.value)


def test_normalize_operation_boundary_is_guarded(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from manyruns.pipeline import prep

    missing = ModuleNotFoundError("No module named 'scanpy'", name="scanpy")

    def normalize(*args, **kwargs):
        raise missing

    monkeypatch.setattr(prep, "_ops", lambda: SimpleNamespace(normalize=normalize))
    with pytest.raises(ImportError, match="singlecell") as exc:
        prep._step_normalize({"counts": np.ones((4, 3))}, None, {"force": True})
    assert exc.value.__cause__ is missing


def test_dependency_failure_in_counts_detection_is_not_a_benign_skip(monkeypatch):
    from types import SimpleNamespace
    from manyruns.pipeline import prep

    def inspect(*args):
        raise ModuleNotFoundError("absent", name="scanpy")

    monkeypatch.setattr(prep, "_ops", lambda: SimpleNamespace(looks_like_counts=inspect))
    with pytest.raises(ImportError, match="singlecell"):
        prep._looks_like_counts(None)


@pytest.mark.parametrize("adapter_answers,client,fallback", [
    (True, "manyagents/OllamaAdapter", False), (False, "urllib", True),
])
def test_answer_records_actual_route_without_network(adapter_answers, client, fallback, monkeypatch):
    from manyruns import agents

    calls = []

    def adapter(*args):
        calls.append("adapter")
        if not adapter_answers:
            raise ImportError("manyagents absent")
        return " measured answer "

    def direct(*args):
        calls.append("urllib")
        return " measured answer "

    monkeypatch.setattr(agents, "_ask_via_manyagents", adapter)
    monkeypatch.setattr(agents, "_ask_via_ollama", direct)
    monkeypatch.setattr(agents, "ask_model", lambda: "qwen3:8b")
    monkeypatch.setattr(agents, "_ask_base_url", lambda: "http://local.example/v1")
    reply = agents.answer(system="s", prompt="p")
    assert reply == agents.Answer("measured answer", "qwen3:8b", "http://local.example/v1", client, fallback)
    assert calls == (["adapter"] if adapter_answers else ["adapter", "urllib"])
    assert ("fallback" in reply.attribution) is fallback


@pytest.mark.parametrize("bad", [None, "", "   ", {}, RuntimeError("failed")])
def test_answer_all_backend_failures_remain_none(bad, monkeypatch):
    from manyruns import agents

    def fail(*args):
        if isinstance(bad, Exception):
            raise bad
        return bad

    monkeypatch.setattr(agents, "_ask_via_manyagents", fail)
    monkeypatch.setattr(agents, "_ask_via_ollama", fail)
    assert agents.answer(system="s", prompt="p") is None


def test_answer_configuration_failure_remains_none(monkeypatch):
    from manyruns import agents

    def fail():
        raise ValueError("bad config")

    monkeypatch.setattr(agents, "ask_model", fail)
    assert agents.answer(system="s", prompt="p") is None


@pytest.mark.parametrize("client,fallback", [("manyagents/OllamaAdapter", False), ("urllib", True)])
def test_answer_pane_and_event_keep_the_actual_route(client, fallback, monkeypatch):
    from io import StringIO
    from types import SimpleNamespace
    from rich.console import Console
    from manyruns import agents, trace
    from manyruns.tui.run import RunScreen, answer_view

    events = []
    monkeypatch.setattr(trace, "current", lambda: SimpleNamespace(emit=events.append))
    screen = RunScreen({"name": "embed", "steps": []})
    screen._outstanding = 1
    monkeypatch.setattr(screen, "_paint_ask_row", lambda: None)
    monkeypatch.setattr(screen, "sync", lambda: None)
    reply = agents.Answer("measured answer", "qwen3:8b", "http://local.example/v1", client, fallback)
    message = RunScreen.Answered("why?", reply, dispatch_id=1, context="record", model="stale",
                                latency=.1)
    screen.on_run_screen_answered(message)
    output = StringIO()
    Console(file=output, width=120).print(answer_view(screen.asked, screen.answer, screen.ask_note))
    assert reply.attribution in output.getvalue()
    assert events[0]["model"] == "qwen3:8b"
    assert events[0]["client"] == client
    assert events[0]["endpoint"] == "http://local.example/v1"
    assert events[0]["fallback"] is fallback
    assert events[0]["answer"] == "measured answer"


def test_metadata_requests_the_shipped_singlecell_runtime():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert "manylatents-omics[singlecell]>=0.1.3" in metadata["project"]["dependencies"]
    assert "manylatents>=0.1.7" in metadata["project"]["dependencies"]
    pipeline_root = root / "manyruns" / "pipeline"
    assert pipeline_root.is_dir()
    for name in ("loading.py", "runner.py", "prep.py"):
        source = (pipeline_root / name).read_text()
        assert "git+<manyruns>" not in source
        assert "install `manylatents[omics]`" not in source


@pytest.mark.source_checkout
def test_install_docs_pin_python():
    root = Path(__file__).resolve().parents[1]
    assert root.is_dir()
    for name in ("README.md", "CLAUDE.md"):
        commands = [line for line in (root / name).read_text().splitlines()
                    if line.startswith("uv tool install ")]
        assert commands
        assert all("--python 3.12" in line for line in commands)


@pytest.mark.parametrize("direct,revision", [
    ('{"vcs_info":{"commit_id":"196cf6a0123456789"}}', "196cf6a0123456789"),
    (None, "unavailable"), ('{"url":"file:///wheel"}', "unavailable"),
    ("malformed", "unavailable"),
])
@pytest.mark.parametrize("program", ["co-science", "manyruns"])
def test_version_reports_installed_build_without_becoming_a_run(direct, revision, program, monkeypatch, capsys):
    from importlib import metadata
    from types import SimpleNamespace
    from manyruns import app

    def read(name):
        assert name == "direct_url.json"
        return direct

    monkeypatch.setattr(metadata, "distribution", lambda name: SimpleNamespace(version="0.1.0", read_text=read))
    monkeypatch.setattr("sys.argv", [program, "--version"])
    assert app._normalize_argv(None) == ["--version"]
    with pytest.raises(SystemExit) as exc:
        app.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"{program} 0.1.0 · revision {revision}"


def test_measurement_provenance_does_not_mutate_a_session_snapshot():
    from manyruns.measurements import annotate

    g = {"pca.trustworthiness": .9}
    annotate(g, "pca.trustworthiness", kind="measurement", stage="first evaluation")
    snapshot = dict(g)
    annotate(g, "pca.trustworthiness", kind="measurement", stage="retry evaluation")
    assert snapshot["_provenance"]["pca.trustworthiness"]["stage"] == "first evaluation"
    assert g["_provenance"]["pca.trustworthiness"]["stage"] == "retry evaluation"


def test_manyagents_attribution_requires_the_adapter_to_answer(monkeypatch):
    import sys
    from types import ModuleType
    from manyruns import agents

    calls = []
    module = ModuleType("manyagents.adapters.openai_adapter")

    class Adapter:
        async def chat(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return {"content": "adapter answered"}

    module.OllamaAdapter = Adapter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(agents, "_ask_via_ollama", lambda *a: pytest.fail("unexpected fallback"))
    reply = agents.answer(system="s", prompt="p")
    assert reply.client == "manyagents/OllamaAdapter"
    assert reply.text == "adapter answered"
    assert reply.fallback is False
    assert calls[0][1]["model"] == reply.model
    assert calls[0][1]["reasoning_effort"] == "none"


def test_catalog_and_path_execute_identical_steps_and_scientific_vector(tmp_path, monkeypatch):
    """Exercise both CLI routes through loading and the step loop, with deterministic compute.

    The stand-ins isolate routing from the unavailable biology stack. The real biology
    integration tests still require scanpy and still fail when it is absent.
    """
    api = pytest.importorskip("manylatents.api")
    anndata = pytest.importorskip("anndata")
    import numpy as np
    from manyruns import app, shell, store
    from manyruns.pipeline import prep, runner

    source = tmp_path / "counts.h5ad"
    anndata.AnnData(np.ones((8, 4), dtype=np.float32)).write_h5ad(source)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shell, "dataset_ref_path", lambda ref: source)
    ops = prep._ops()
    monkeypatch.setattr(ops, "normalize", lambda X, **k: X.copy())
    monkeypatch.setattr(ops, "transform", lambda X, **k: X.copy())
    monkeypatch.setattr(api, "run", lambda **k: {"embeddings": k["input_data"][:, :2].copy(),
                                               "scores": {"trustworthiness": .9817}})
    monkeypatch.setattr(runner._suite, "measure", lambda *a, **k: {"trustworthiness": .5})
    monkeypatch.setattr(runner._io, "_save_scatter", lambda *a, **k: None)
    monkeypatch.setattr(shell, "render_result", lambda *a: None)
    recorded = []
    monkeypatch.setattr(store, "append", lambda row, **k: recorded.append(row) or "runs.jsonl")
    for handle in ("pbmc3k", str(source)):
        project = _setup(handle, monkeypatch)
        project["engine"] = "manylatents"
        assert app._explore_project(project) == 0
    named, direct = recorded
    assert [(s["name"], s["outcome"]) for s in named["steps"]] == [
        (s["name"], s["outcome"]) for s in direct["steps"]]
    assert len(named["steps"]) == 4
    assert all(s["outcome"] == "ok" for s in named["steps"])
    assert named["g_vector"] == direct["g_vector"]


@pytest.mark.parametrize("name", ["rank_genes", "dpt"])
def test_missing_biology_runtime_is_unsupported_not_benign(name, tmp_path, monkeypatch):
    import sys
    import numpy as np
    from manyruns.outcomes import run_verdict
    from manyruns.pipeline import runner

    monkeypatch.setitem(sys.modules, "scanpy", None)
    state = runner._new_state(X=np.ones((4, 3)), counts=np.ones((4, 3)),
                              genes=np.array(["A", "B", "C"]), emb=np.array([[0], [0], [1], [1]]))
    rec = runner.apply_step({"name": name, "group": "analysis"}, state, {},
                            index=0, carry=runner.new_carry(),
                            dispatch={"analysis": runner._run_analysis_step},
                            ctx={"out_dir": tmp_path, "plots": [], "target_dim": 2})
    assert rec["outcome"] == "skipped"
    assert rec["unsupported"] is True
    assert "uv sync --extra harness --extra dev" in rec["detail"]
    assert "uv tool install --python 3.12" in rec["detail"]
    assert run_verdict({"steps": [rec]})[0] == 1


def test_urllib_attribution_comes_from_the_direct_response(monkeypatch):
    import io
    import json
    import urllib.request
    from manyruns import agents

    requests = []

    def absent(*args):
        raise ImportError("manyagents absent")

    def response(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "direct answer"}}]}).encode())

    monkeypatch.setattr(agents, "_ask_via_manyagents", absent)
    monkeypatch.setattr(urllib.request, "urlopen", response)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://local.example/v1")
    reply = agents.answer(system="s", prompt="p")
    assert reply.client == "urllib"
    assert reply.fallback is True
    assert reply.text == "direct answer"
    assert requests[0].full_url == "http://local.example/v1/chat/completions"
    body = json.loads(requests[0].data)
    assert body["model"] == reply.model
    assert body["reasoning_effort"] == "none"
