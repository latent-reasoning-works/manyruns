"""The step record — what actually EXECUTED, as opposed to what was declared.

The old loop appended to `trace` *before* the group check, the dispatch lookup and the
executor call, so a recipe naming three steps produced a three-entry trace whether or not
any of them ran. Every consumer — the summary, the sweep table, any future scorer — read
that as evidence of execution. The measured consequence: cost currently
*rewards declining*, because three named-but-failed steps yield an identical trace, a lower
wall time, and a row recorded as successful.

`steps` is the fix: one entry per attempt, positional, with `outcome` and `seconds` written
only after the executor returns or raises. `trace` and `status` are now derived views of it,
which is what these tests pin — the legacy shape must stay byte-identical while the record
underneath it becomes honest.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _install(monkeypatch, **impls):
    """Put stub executors into the in-process loop's table, so no scientific stack is needed."""
    from manyruns import pipeline

    for name, fn in impls.items():
        monkeypatch.setitem(pipeline._INPROC_STEPS, name, fn)


def _ok(state, g, params, out_dir, plots, target_dim):
    state["emb"] = np.zeros((state["X"].shape[0], 2))


def _boom(state, g, params, out_dir, plots, target_dim):
    raise ValueError("kaboom")


def _recipe(*names, group="latent", params=None):
    return {
        "name": "r",
        "steps": [{"name": n, "group": group, "params": dict(params or {})} for n in names],
    }


def test_the_record_distinguishes_attempts_from_executions(tmp_path, monkeypatch):
    """THE regression. A middle step that raises leaves the trace at full length — the
    trace cannot express an outcome — while the record says exactly which one died.

    Mutation-checked, and the result is worth recording because the obvious mutation is
    the WRONG one: pre-setting `rec["outcome"] = "ok"` above the executor call does NOT
    fail this test, because the except/else handlers overwrite it. The mutation that does
    fail is recording a raised step as "ok" (verified: fails this test and
    `test_the_legacy_views_are_byte_identical`). So what is pinned here is the record's
    *honesty*, not the assignment's position — the position only matters for `seconds`,
    which `test_duration_is_measured_around_the_executor` covers."""
    from manyruns import pipeline

    _install(monkeypatch, first=_ok, boom=_boom, last=_ok)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("first", "boom", "last"), tmp_path)

    assert out["trace"] == ["latent:first", "latent:boom", "latent:last"]  # unchanged
    assert [s["outcome"] for s in out["steps"]] == ["ok", "error", "ok"]
    assert out["steps"][1]["detail"] == "ValueError: kaboom"
    assert sum(s["outcome"] == "ok" for s in out["steps"]) == 2
    assert out["ok"] is False


def test_a_repeated_step_name_collapses_in_status_but_not_in_the_record(tmp_path, monkeypatch):
    """`status` is keyed by name, so a recipe running one step twice silently loses one.
    The record is positional and keeps both — pinning the collision loudly rather than
    leaving a caller to discover that two steps produced one status entry."""
    from manyruns import pipeline

    _install(monkeypatch, twice=_ok)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("twice", "twice"), tmp_path)

    assert len(out["steps"]) == 2
    assert [s["index"] for s in out["steps"]] == [0, 1]
    assert len(out["status"]) == 1          # the lossy view
    assert out["num_steps"] == 2            # ...but the count still comes from the record


def test_params_round_trip_and_the_recipe_cannot_be_rewritten(tmp_path, monkeypatch):
    """Params are recorded (step agreement needs them at the params tier) and COPIED, so an
    executor that mutates its params dict cannot rewrite the recipe's account of what it
    was asked to do."""
    from manyruns import pipeline

    def mutates(state, g, params, out_dir, plots, target_dim):
        params["n_components"] = 999          # a badly-behaved executor
        state["emb"] = np.zeros((state["X"].shape[0], 2))

    _install(monkeypatch, p=mutates)
    recipe = _recipe("p", params={"n_components": 7})
    out = pipeline.run_inproc(np.zeros((10, 3)), recipe, tmp_path)

    assert recipe["steps"][0]["params"] == {"n_components": 7}   # recipe untouched
    assert out["steps"][0]["params"]["n_components"] == 999      # what the step actually saw


def test_a_step_no_executor_implements_is_recorded_as_skipped(tmp_path):
    """A typo'd step name is a skip, not a success — and the record says so in one field
    rather than requiring a caller to parse the status string."""
    from manyruns import pipeline

    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("nosuchalgo"), tmp_path)

    assert out["steps"][0]["outcome"] == "skipped"
    assert out["status"]["nosuchalgo"] == "skipped (no in-process implementation)"
    assert out["ok"] is False


def test_a_step_with_no_group_is_recorded_without_dispatching(tmp_path):
    from manyruns import pipeline

    recipe = {"name": "r", "steps": [{"name": "orphan"}]}
    out = pipeline.run_inproc(np.zeros((10, 3)), recipe, tmp_path)

    rec = out["steps"][0]
    assert (rec["outcome"], rec["group"]) == ("skipped", None)
    assert rec["seconds"] == 0.0                       # never dispatched, so never timed
    assert out["status"]["orphan"].startswith("skipped (step declares no 'group'")


def test_the_record_carries_duration_and_provenance(tmp_path, monkeypatch):
    """Cost (audit axis ⑤) is a sum over durations, so they have to be on the record. And
    `seed` is None deliberately: `run_inproc` has no seed channel and
    `_step_phate` passes no `random_state` (issue #29), so those runs are NOT reproducible.
    Recording 42 here would be inventing provenance that does not exist."""
    from manyruns import pipeline

    _install(monkeypatch, s=_ok)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("s"), tmp_path)

    assert isinstance(out["steps"][0]["seconds"], float)
    assert out["steps"][0]["seconds"] >= 0.0
    assert out["tool_version"]
    assert out["seed"] is None


def test_duration_is_measured_around_the_executor(tmp_path, monkeypatch):
    """The one assertion that genuinely pins WHERE the timing happens. A step that sleeps
    must report at least that sleep; start the clock after the executor (or stop it before)
    and this collapses to ~0. Cost scoring is a sum over this field, so a clock that is
    merely present but measures nothing would be worse than no field at all."""
    import time as _time

    from manyruns import pipeline

    def slow(state, g, params, out_dir, plots, target_dim):
        _time.sleep(0.02)
        state["emb"] = np.zeros((state["X"].shape[0], 2))

    _install(monkeypatch, slow=slow)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("slow"), tmp_path)

    assert out["steps"][0]["seconds"] >= 0.01


def test_the_legacy_views_are_byte_identical(tmp_path, monkeypatch):
    """`trace` and `status` are now derived rather than accumulated. Every existing consumer
    (summary writer, sweep table, ~nine test assertions) reads those two, so their exact
    shape is the compatibility contract for this change."""
    from manyruns import pipeline

    _install(monkeypatch, a=_ok, b=_boom)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("a", "b"), tmp_path)

    assert out["trace"] == ["latent:a", "latent:b"]
    assert out["status"] == {"a": "ok", "b": "error: ValueError: kaboom"}


def test_the_sweep_table_no_longer_drops_the_record(tmp_path, monkeypatch):
    """`experiment.run_one` read g_vector/ok/status and let the trace fall on the floor, so
    step agreement and cost were unscoreable from the table no matter what the runner
    recorded. The record has to survive the trip into `Result`."""
    from manyruns import experiment, pipeline

    _install(monkeypatch, phate=_ok)

    def fake_run(mode, req, device=None):
        return pipeline.run_inproc(np.zeros((10, 3)), _recipe("phate"), tmp_path)

    monkeypatch.setattr("manyruns.modes.run", fake_run)
    res = experiment.run_one(
        experiment.Run(recipe=_recipe("phate"), dataset={"name": "d", "shape": "manifold"}),
        out_dir=tmp_path,
    )

    assert [s["name"] for s in res.steps] == ["phate"]
    assert res.steps[0]["outcome"] == "ok"


def test_the_record_says_what_each_step_produced(tmp_path, monkeypatch):
    """A step can report "ok" and contribute nothing — no embedding, no g-vector key — and
    the outcome alone cannot tell that apart from real work. `produced`/`emitted` close the
    output half of the record, and share `watch.describe` so a wrapped tool and a recipe
    step account for their output the same way."""
    from manyruns import pipeline

    def embeds(state, g, params, out_dir, plots, target_dim):
        state["emb"] = np.zeros((state["X"].shape[0], 2))
        g["thing"] = 0.5

    def does_nothing(state, g, params, out_dir, plots, target_dim):
        return None

    _install(monkeypatch, embeds=embeds, idle=does_nothing)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("embeds", "idle"), tmp_path)

    made, idle = out["steps"]
    assert made["produced"] == "ndarray(10, 2) float64"
    assert made["emitted"] == ["thing"]

    assert idle["outcome"] == "ok"          # it did not fail...
    assert idle["produced"] is None         # ...and it did not do anything either
    assert idle["emitted"] == []


# ── per-step geometry ────────────────────────────────────────────────────────────────────
#
# `suite.measure` runs ONCE, from `_finalize`, after every step has already run — so all
# twelve geometry metrics were attributed to no step, and nothing could say "phate took lid
# from X to Y" until the run was over. These pin the delta onto the step that caused it.
# The g-vector is untouched and stays authoritative; this is a view.


def _metrics(monkeypatch, fn=None):
    """A stand-in for manylatents' registry — these must run with no private stack."""
    from manyruns.pipeline import suite

    monkeypatch.setattr(
        suite, "_compute_metric",
        lambda: fn or (lambda name, embeddings=None, dataset=None: np.float32(embeddings.shape[1])),
    )


def _embeds(dim):
    def step(state, g, params, out_dir, plots, target_dim):
        state["emb"] = np.zeros((state["X"].shape[0], dim))
    return step


def test_each_step_carries_its_own_geometry_delta(tmp_path, monkeypatch):
    """THE contract. `[from, to]` per metric, and `from` is None on the first embedding step:
    that step is an APPEARANCE, not a change. A baseline of 0 would be an invention — `lid`
    did not go from 0 to 3.4, it did not exist."""
    from manyruns import pipeline
    from manyruns.pipeline import suite

    _metrics(monkeypatch)                              # value == emb.shape[1]
    _install(monkeypatch, first=_embeds(2), second=_embeds(5))
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("first", "second"), tmp_path)

    first, second = out["steps"]
    assert first["geometry"]["lid"] == [None, 2.0]     # appeared
    assert second["geometry"]["lid"] == [2.0, 5.0]     # ...then moved, from the real prior
    assert set(first["geometry"]) <= set(suite.LIVE_SUBSET)
    assert "trustworthiness" not in first["geometry"]  # 0.524 s/step at 2,700 — see LIVE_SUBSET
    # the record, not the g-vector: the shared geometry core just went from 10 provenance
    # keys to 28 and must not be re-cut per step
    assert "geometry" not in out["g_vector"]
    assert out["g_vector"]["lid"] == 2.0 or out["g_vector"]["lid"] == 5.0


def test_a_step_that_left_the_embedding_alone_has_no_geometry_key(tmp_path, monkeypatch):
    """Omitted entirely, not an empty dict. A `{}` reads as "measured, found nothing", which
    is a different claim from "this step did not move the geometry"."""
    from manyruns import pipeline

    def reads_only(state, g, params, out_dir, plots, target_dim):
        g["read"] = 1.0

    _metrics(monkeypatch)
    _install(monkeypatch, embeds=_embeds(2), reads=reads_only)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("embeds", "reads"), tmp_path)

    assert "geometry" in out["steps"][0]
    assert "geometry" not in out["steps"][1]


def test_a_re_embedding_of_the_same_shape_still_reports_a_delta(tmp_path, monkeypatch):
    """A second embedder refining the first at the same dimensionality is the step most likely
    to have a delta, and BOTH halves of the record have to show it.

    CHANGED HERE: `produced` used to be asserted `None` on this step, because it compared
    `describe` output — shape and dtype — while `_persist` and `geometry` keyed off object
    identity. Measured on the bundled `cflows` over pbmc3k (run 60921b8e06b6): step 1
    `mioflow` recorded `outcome='ok'`, `produced=None` and, in the same record,
    `artifacts={'emb': '…/01-mioflow_emb.npy'}` holding a (2700, 3) array — the trajectory
    step reading as contributing nothing while its embedding was written to disk. All three
    fields key off identity now, so what moved is `produced`; the delta below is unchanged."""
    from manyruns import pipeline

    def fill(v):
        def step(state, g, params, out_dir, plots, target_dim):
            state["emb"] = np.full((state["X"].shape[0], 2), float(v))
        return step

    _metrics(monkeypatch, lambda name, embeddings=None, dataset=None: float(embeddings[0, 0]))
    _install(monkeypatch, a=fill(1), b=fill(9))
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("a", "b"), tmp_path)

    assert out["steps"][1]["produced"] == out["steps"][0]["produced"]  # same shape, new array
    assert out["steps"][1]["produced"] == "ndarray(10, 2) float64"
    assert out["steps"][1]["geometry"]["lid"] == [1.0, 9.0]    # ...and the delta with it


def test_the_observer_is_told_about_the_geometry_not_a_record_without_it(tmp_path, monkeypatch):
    """Ordering, and the whole reason this exists: `on_step` is how a live panel learns a
    step is done. Attach the delta after `_report` and the panel renders the step once,
    without it, and never comes back."""
    import copy

    from manyruns import pipeline

    seen = []

    def observe(rec, steps):
        if rec.get("outcome"):                     # the settled report, not the "running" one
            seen.append(copy.deepcopy(rec))        # rec is mutated in place — snapshot it

    _metrics(monkeypatch)
    _install(monkeypatch, first=_embeds(2), second=_embeds(5))
    pipeline.run_inproc(np.zeros((10, 3)), _recipe("first", "second"), tmp_path,
                          on_step=observe)

    assert [s["geometry"]["lid"] for s in seen] == [[None, 2.0], [2.0, 5.0]]


def test_without_the_engine_there_is_no_geometry_and_no_noise(tmp_path, monkeypatch):
    """The stackless path — manyruns's default, and every CI run. Per-step geometry simply
    does not happen: no key, no note, no crash."""
    from manyruns import pipeline
    from manyruns.pipeline import suite

    monkeypatch.setattr(suite, "_compute_metric", lambda: None)
    _install(monkeypatch, first=_embeds(2))
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("first"), tmp_path)

    assert out["ok"] is True
    assert not [s for s in out["steps"] if "geometry" in s]


def test_the_delta_rides_into_the_index_as_clean_json(tmp_path, monkeypatch):
    """`store.append` serialises `steps` through `_jsonable`, so this arrives in
    `index.jsonl` for free — but only if the values are JSON scalars. A `np.float32` is not:
    `_jsonable` would `str()` it and the index would hold "2.0" as a string."""
    import json

    from manyruns import pipeline, store

    _metrics(monkeypatch)                          # returns np.float32 on purpose
    _install(monkeypatch, first=_embeds(2), second=_embeds(5))
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("first", "second"), tmp_path)
    store.append(out, out_dir=tmp_path)

    row = json.loads(store.index_path(tmp_path).read_text().splitlines()[-1])
    pairs = [v for s in row["steps"] for v in s.get("geometry", {}).values()]
    assert pairs, "the geometry did not survive the trip into the index"
    assert all(isinstance(p, list) and len(p) == 2 for p in pairs)
    assert all(x is None or isinstance(x, float) for p in pairs for x in p)
    assert row["steps"][1]["geometry"]["lid"] == [2.0, 5.0]


def test_produced_reports_a_change_not_a_snapshot(tmp_path, monkeypatch):
    """A later step that leaves the embedding untouched must not re-report the previous
    step's output as its own."""
    from manyruns import pipeline

    def embeds(state, g, params, out_dir, plots, target_dim):
        state["emb"] = np.zeros((state["X"].shape[0], 2))

    def reads_only(state, g, params, out_dir, plots, target_dim):
        assert state["emb"] is not None     # it saw the embedding
        g["read"] = 1.0                     # ...and emitted a value without changing it

    _install(monkeypatch, embeds=embeds, reads=reads_only)
    out = pipeline.run_inproc(np.zeros((10, 3)), _recipe("embeds", "reads"), tmp_path)

    assert out["steps"][0]["produced"] == "ndarray(10, 2) float64"
    assert out["steps"][1]["produced"] is None       # unchanged → not claimed
    assert out["steps"][1]["emitted"] == ["read"]
