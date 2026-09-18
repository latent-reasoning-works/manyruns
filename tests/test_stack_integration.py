"""Real biology -> compute -> intelligence -> record; no model activations are claimed.

The fixture uses 240 synthetic cells, 24 genes and two populations. Independent
sklearn PCA coordinates keyed by cell ID check correspondence across live output,
persistence, filtering and branching. Shuffled population labels check evidence
sensitivity; they cannot establish per-cell identity. This is a PCA wiring check,
not a general detector of arbitrary output misalignment.

Run with `python -m pytest tests/test_stack_integration.py`. Missing sibling
packages skip; broken imports and stale compute contracts fail. One asyncio Runner
owns both injected requests and client cleanup, using an in-memory HTTP transport.

The only stand-in is the intelligence behind an in-memory HTTP transport.
manyruns.call_tool_async -> manyagents.ClaudeAdapter -> real SDK serialization all execute. Replace
the injected adapter with a real provider without changing product code. This proves the
seam and evidence sensitivity, not language-model competence.

Omics' from_anndata returns manykinds.LabeledArray at the ingestion boundary. The
current Session/api.run/metric APIs still accept arrays, and agent prompts/replies and
store records are JSON, not kinds. We explicitly unwrap there; no invented activation path.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.stack_integration


@pytest.fixture
def stack():
    # Only ABSENT siblings skip. Broken imports or stale contracts must fail loudly.
    for name in ("manylatents", "manyagents", "manykinds"):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"STACK INTEGRATION NOT RUN: missing sibling {name}")
        importlib.import_module(name)
    if importlib.util.find_spec("manylatents.singlecell") is None:
        pytest.skip("STACK INTEGRATION NOT RUN: missing manylatents.singlecell omics namespace")


class _ScriptedIntelligence:
    """Scripted numeric comparison, never a model and never a source of metric values."""

    def __init__(self):
        self.requests = []

    def respond(self, wire_request):
        import httpx

        request = json.loads(wire_request.content)
        self.requests.append(request)
        evidence = json.loads(request["messages"][0]["content"])
        candidates = evidence["candidates"]
        winner = max(candidates, key=lambda candidate: candidate["value"])
        answer = {"chosen_artifact": winner["artifact"], "metric": evidence["metric"],
                  "citations": candidates, "intelligence": "scripted-stand-in"}
        body = {
            "id": "msg-scripted", "type": "message", "role": "assistant",
            "model": request["model"], "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "tool_use", "id": "tool-scripted",
                         "name": request["tools"][0]["name"], "input": answer}],
        }
        return httpx.Response(200, json=body)


@pytest.fixture
def intelligence(stack):
    import asyncio
    import httpx
    from anthropic import AsyncAnthropic
    from manyagents.adapters.claude_adapter import ClaudeAdapter

    stand_in = _ScriptedIntelligence()
    client = AsyncAnthropic(api_key="scripted-integration-only",
                            http_client=httpx.AsyncClient(
                                transport=httpx.MockTransport(stand_in.respond)))
    stand_in.adapter = ClaudeAdapter(api_key="scripted-integration-only")
    stand_in.adapter.client = client
    with asyncio.Runner() as loop:
        stand_in.loop = loop
        try:
            yield stand_in
        finally:
            loop.run(client.close())


def _interpret(candidates, intelligence):
    from manyruns import agents

    evidence = {"metric": "silhouette", "candidates": candidates}
    answer = intelligence.loop.run(agents.call_tool_async(
        adapter=intelligence.adapter,
        system="Choose the artifact with the highest measured silhouette. Cite every exact value.",
        prompt=json.dumps(evidence), name="compare_evidence",
        description="Return a choice and exact citations from the supplied measured evidence.",
        schema={"type": "object", "properties": {
            "chosen_artifact": {"type": "string", "enum": [c["artifact"] for c in candidates]},
            "metric": {"type": "string", "enum": ["silhouette"]},
            "citations": {"type": "array", "items": {"type": "object", "properties": {
                "artifact": {"type": "string"}, "value": {"type": "number"}},
                "required": ["artifact", "value"], "additionalProperties": False}},
            "intelligence": {"type": "string"}},
            "required": ["chosen_artifact", "metric", "citations", "intelligence"],
            "additionalProperties": False}))
    assert answer is not None, "real manyagents adapter/SDK did not return the scripted tool call"
    assert answer["metric"] == "silhouette"
    assert answer["citations"] == candidates
    assert answer["intelligence"] == "scripted-stand-in"
    return answer


def _pca_oracle(X, sample_ids, labels, dimension):
    """Independent of Session/output: fit the public estimator to known input cells.

    This checks PCA row correspondence, not the correctness of sklearn's PCA math.
    No per-cell oracle is claimed for other latent/trajectory algorithms here.
    """
    from sklearn.decomposition import PCA

    embedding = PCA(n_components=dimension, random_state=13).fit(X).transform(X)
    return {str(cell): (row.copy(), str(label))
            for cell, row, label in zip(sample_ids, embedding, labels)}


def _assert_cells(embedding, sample_ids, labels, oracle):
    """Compare by ID, allowing a consistent reorder but no omitted/extra/wrong cells."""
    import numpy as np

    ids = list(map(str, sample_ids))
    assert len(ids) == len(set(ids)) == len(oracle), "cell identity cardinality"
    assert set(ids) == set(oracle), "cell identity membership"
    assert len(embedding) == len(labels) == len(ids), "cell identity lengths"
    for cell, row, label in zip(ids, embedding, labels):
        expected_row, expected_label = oracle[cell]
        assert str(label) == expected_label, f"cell label mismatch: {cell}"
        np.testing.assert_allclose(row, expected_row, rtol=1e-5, atol=1e-4,
                                   err_msg=f"cell coordinates mismatch: {cell}")


def test_biology_compute_intelligence_record(tmp_path, stack, intelligence):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from manykinds import LabeledArray
    from manylatents.metrics import compute_metric
    from manylatents.singlecell.data.adapters.formats.adapters import from_anndata
    from manyruns import artifacts, decisions, store
    from manyruns.pipeline import suite
    from manyruns.session import Session

    rng = np.random.default_rng(2026)
    n, genes = 240, 24
    labels = np.repeat(["population-A", "population-B"], n // 2)
    # Positive synthetic expression: two populations with distinct marker genes and
    # nuisance variation. Shuffle once so input order cannot stand in for population.
    rates = np.full((n, genes), 12.0)
    rates[:n // 2, :6] = 40
    rates[n // 2:, 6:12] = 40
    counts = rng.poisson(rates).astype(np.float32)
    order = rng.permutation(n)
    ids = np.asarray([f"cell-{i:04d}" for i in range(n)])[order]
    labels, counts = labels[order], counts[order]
    biology = ad.AnnData(counts, obs=pd.DataFrame({"population": labels}, index=ids),
                        var=pd.DataFrame(index=[f"gene-{i}" for i in range(genes)]))
    payload = from_anndata(biology, coords={
        "cell": biology.obs_names.to_numpy(), "gene": biology.var_names.to_numpy(),
        "population": ("cell", biology.obs["population"].to_numpy()),
    })
    assert isinstance(payload, LabeledArray)
    payload.require("cell", "gene", coords=("cell", "population"))
    np.testing.assert_array_equal(payload.da.coords["cell"], ids)
    np.testing.assert_array_equal(payload.da.coords["population"], labels)
    np.testing.assert_array_equal(payload.da.values, counts)

    # Explicit array boundary: these production APIs do not accept kinds today.
    X = payload.da.values
    sample_ids = payload.da.coords["cell"].values
    aligned_labels = payload.da.coords["population"].values
    control_labels = aligned_labels[rng.permutation(n)]
    output = tmp_path / "records"
    runs, candidates, expected = [], [], []
    for dimension, population in ((2, aligned_labels), (8, aligned_labels), (2, control_labels)):
        oracle = _pca_oracle(X, sample_ids, population, dimension)
        session = Session("stack-integration", engine="manylatents", out_dir=output,
                          array=X, labels=population, label_kind="group", sample_ids=sample_ids,
                          dataset="synthetic-two-populations", seed=13,
                          metrics=["trustworthiness"])
        step = session.apply({"name": "pca", "group": "latent",
                              "params": {"n_components": dimension}})
        assert step["outcome"] == "ok", step
        embedding = session.state["emb"]
        assert embedding.shape == (n, dimension)
        live_ids = session.ctx["sample_ids"][session.state["rows"]]
        _assert_cells(embedding, live_ids, session.state["labels"], oracle)
        # Metric implementation stays in compute. A dataset-shaped argument is the
        # registry's current array/metadata API, not a substitute metric or reader.
        measured = compute_metric("silhouette", embedding,
                                  dataset=SimpleNamespace(metadata=session.state["labels"]))
        assert np.isfinite(measured)
        session.g["population_silhouette"] = measured
        unavailable = suite.measure(embedding, X, ["silhouette"])
        assert unavailable["silhouette"] is None  # suite has no population labels
        assert unavailable["silhouette_note"]
        session.g.update(unavailable)
        result = session.close()
        assert result["ok"] and result["complete"]
        assert 0 < result["g_vector"]["pca.trustworthiness"] <= 1
        runs.append(result)
        candidates.append({"artifact": step["artifacts"]["emb"], "value": measured})
        expected.append({"run_id": result["run_id"], "oracle": oracle})

    assert candidates[0]["value"] > candidates[1]["value"] > 0.3
    assert abs(candidates[2]["value"]) < 0.1
    assert candidates[0]["value"] - candidates[2]["value"] > 0.5
    # Only the association to populations changes in the control, not the geometry.
    a, b = (artifacts.load_labeled(output, runs[i]["run_id"], 0) for i in (0, 2))
    np.testing.assert_array_equal(a["array"][np.argsort(a["sample_ids"])],
                                  b["array"][np.argsort(b["sample_ids"])])
    aligned = _interpret(candidates[:2], intelligence)
    control = _interpret([candidates[2], candidates[1]], intelligence)
    assert aligned["chosen_artifact"] == candidates[0]["artifact"]
    assert control["chosen_artifact"] == candidates[1]["artifact"]
    assert aligned["chosen_artifact"] != control["chosen_artifact"]
    assert len(intelligence.requests) == 2
    for request in intelligence.requests:
        # ClaudeAdapter converts the product's tool schema; a fallback direct-SDK
        # request instead forces tool_choice. Pin which real backend ran.
        assert "tool_choice" not in request
        assert request["tools"][0]["input_schema"]["required"] == [
            "chosen_artifact", "metric", "citations", "intelligence"]

    for result, answer in zip(runs, (aligned, None, control)):
        store.append(result, out_dir=output,
                     extra={"interpretation": answer} if answer else None)
    for result, answer in ((runs[0], aligned), (runs[2], control)):
        decision_id = decisions.append(
            offered=answer["citations"], chosen=answer["chosen_artifact"],
            surface="scripted-stand-in", run_id=result["run_id"], out_dir=output)
        assert decision_id

    # The child gets no fixture, AnnData, labels or embeddings. It uses production
    # readers only; expectations are compared by the parent after the child exits.
    child = subprocess.run([sys.executable, "-c", """
import json, sys
from manyruns import artifacts, decisions, store
from manyruns.session import resumable
out = sys.argv[1]
rows = list(store.read(out))
payloads = []
for row in rows[:3]:
    item = artifacts.load_labeled(out, row['run_id'], 0)
    payloads.append({'run_id': row['run_id'], 'sample_ids': item['sample_ids'],
                     'labels': item['labels'], 'embedding': item['array'].tolist()})
print(json.dumps({'rows': rows, 'payloads': payloads,
                  'decisions': list(decisions.read(out)),
                  'resumed': resumable(out)['arrays']['emb'].tolist()}))
""", str(output)], capture_output=True, text=True, timeout=60, check=True)
    reopened = json.loads(child.stdout)
    for item, want in zip(reopened["payloads"], expected):
        assert item["run_id"] == want["run_id"]
        _assert_cells(item["embedding"], item["sample_ids"], item["labels"], want["oracle"])
    assert len(reopened["payloads"]) == len(expected)
    # Resume exposes arrays only: this assertion is round-trip fidelity, not identity.
    assert reopened["resumed"] == reopened["payloads"][-1]["embedding"]
    assert [d["chosen"] for d in reopened["decisions"]] == [
        aligned["chosen_artifact"], control["chosen_artifact"]]
    for row, candidate in zip(reopened["rows"][:3], candidates):
        assert row["g_vector"]["population_silhouette"] == candidate["value"]
        assert row["g_vector"]["silhouette"] is None
        assert row["g_vector"]["silhouette_note"]
        assert row["steps"][0]["artifacts"]["emb"] == candidate["artifact"]
    assert len(reopened["rows"]) == 3
    assert [row["interpretation"] for row in reopened["rows"]
            if "interpretation" in row] == [aligned, control]
    # The reader must never manufacture identity for an older unannotated record.
    path = artifacts.save_array(output, "unannotated", 0, "pca", "emb", X[:, :2])
    artifacts.finish(output, "unannotated", {path: {"step": "pca", "index": 0, "slot": "emb"}})
    with pytest.raises(ValueError, match="no recorded row identity"):
        artifacts.load_labeled(output, "unannotated", 0)


def test_compute_measurement_unavailable_contract(stack):
    """A stale installed compute package FAILS; absence alone is allowed to skip."""
    import numpy as np
    from manylatents.metrics import compute_metric
    from manyruns.pipeline import suite

    try:
        from manylatents.utils.exceptions import MeasurementUnavailable
    except ImportError:
        pytest.fail("Installed manylatents lacks MeasurementUnavailable; update the compute "
                    "dependency to main's no-invented-measurements contract. Do not replace "
                    "this assertion with a numeric/NaN sentinel or an xfail.")
    X = np.random.default_rng(17).normal(size=(8, 4))
    embedding = X[:, :2].copy()
    # n=8, k=5 makes the old implementation's denominator zero: it returned a
    # perfect 1.0 for an undefined measurement. The new contract must refuse it.
    with pytest.raises(MeasurementUnavailable, match="n_neighbors"):
        compute_metric("trustworthiness", embedding,
                       dataset=SimpleNamespace(data=X), n_neighbors=5)
    unavailable = suite.measure(embedding, X, ["trustworthiness"])
    assert unavailable["trustworthiness"] is None
    assert "MeasurementUnavailable" in unavailable["trustworthiness_note"]


def _reverse_within_population(labels):
    import numpy as np

    order = np.arange(len(labels))
    for label in np.unique(labels):
        positions = np.flatnonzero(np.asarray(labels) == label)
        order[positions] = positions[::-1]
    assert np.all(order != np.arange(len(labels))), "control must move every cell"
    return order


def test_headline_rejects_within_population_permutation(tmp_path, stack, intelligence, monkeypatch):
    """Mutation regression: the previous headline PASSED with every cell misidentified."""
    from manyruns.pipeline import runner

    persist = runner._persist
    order = None

    def corrupt(rec, state, ctx, *args, **kwargs):
        nonlocal order
        if state.get("emb") is not None:
            if order is None:
                order = _reverse_within_population(state["labels"])
            # Keep the same geometric corruption in the label-shuffled control too.
            state["emb"] = state["emb"][order]
        return persist(rec, state, ctx, *args, **kwargs)

    monkeypatch.setattr(runner, "_persist", corrupt)
    with pytest.raises(AssertionError, match="cell coordinates mismatch"):
        test_biology_compute_intelligence_record(tmp_path, stack, intelligence)


def test_headline_accepts_consistent_reorder(tmp_path, stack, intelligence, monkeypatch):
    """Move coordinates AND identity together; the oracle must not insist on input order."""
    from manyruns.pipeline import runner

    persist = runner._persist

    def reorder(rec, state, ctx, *args, **kwargs):
        order = _reverse_within_population(state["labels"])
        state["emb"] = state["emb"][order]
        state["rows"] = state["rows"][order]
        state["labels"] = state["labels"][order]
        return persist(rec, state, ctx, *args, **kwargs)

    monkeypatch.setattr(runner, "_persist", reorder)
    test_biology_compute_intelligence_record(tmp_path, stack, intelligence)


@pytest.mark.parametrize("stage", ["filter", "branch"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_cell_correspondence_after_filter_and_branch(tmp_path, stack, stage, corrupt):
    """Known mask and PCA oracle check identities after narrowing and disk-backed branching.

    The mask is a test input to the production transition, not a claim to validate the
    biological decision of a filter. PCA is the only computed step with a per-cell oracle.
    """
    import numpy as np
    from manyruns import artifacts
    from manyruns.pipeline import runner
    from manyruns.session import Session

    X = np.random.default_rng(93).normal(size=(80, 6)).astype(np.float32)
    ids = np.asarray([f"cell-{i}" for i in range(len(X))])
    labels = np.repeat(["A", "B"], 40)
    mask = np.arange(len(X)) % 2 == 0
    oracle = _pca_oracle(X[mask], ids[mask], labels[mask], 2)
    session = Session("correspondence", engine="manylatents", out_dir=tmp_path,
                      array=X, labels=labels, sample_ids=ids, seed=13,
                      metrics=["trustworthiness"])
    # Exercise composition too: two masks expressed in their current view's positions.
    first = np.arange(len(X)) % 4 != 3
    runner._apply_transition({"mask": first}, session.state, {"name": "cut"}, session.ctx)
    runner._apply_transition({"mask": mask[first]}, session.state, {"name": "cut"}, session.ctx)
    step = session.apply({"name": "pca", "group": "latent", "params": {"n_components": 2}})
    assert step["outcome"] == "ok", step
    session.close()
    if stage == "branch":
        # Corrupt the actual artifact the branch reads, leaving its identities unchanged.
        if corrupt:
            path = step["artifacts"]["emb"]
            embedding = artifacts.load_array(path)
            np.save(path, embedding[_reverse_within_population(labels[mask])])
        child = session.branch(0)
        embedding = child.state["emb"]
        selected_ids = child.ctx["sample_ids"][child.state["rows"]]
        selected_labels = child.state["labels"]
    else:
        item = artifacts.load_labeled(tmp_path, session.run_id, 0)
        embedding, selected_ids, selected_labels = item["array"], item["sample_ids"], item["labels"]
        if corrupt:
            # Same count, same populations, wrong surviving cell IDs after filtering.
            selected_ids = np.asarray(selected_ids)[_reverse_within_population(selected_labels)]
    if corrupt:
        with pytest.raises(AssertionError, match="cell coordinates mismatch"):
            _assert_cells(embedding, selected_ids, selected_labels, oracle)
    else:
        _assert_cells(embedding, selected_ids, selected_labels, oracle)
