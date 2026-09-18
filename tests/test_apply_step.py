"""The step/run seam — `apply_step` + `new_carry` / `finish_carry`.

`_run_steps` used to be the smallest unit anything could drive: a recipe went in, a whole
run came out. That is the wrong grain for a session, which applies ONE step per user action
and cannot pay a run's lifecycle each time. `apply_step` is the step-shaped half; `new_carry`
and `finish_carry` name the run-shaped half that used to be three locals in the loop.

These pin the seam's contract from the OUTSIDE — driving `apply_step` by hand, the way a
session will (stage 2), rather than through `run_inproc`. The existing suite
(`test_step_record.py`, `test_artifacts.py`, `test_live_dashboard.py`, …) already pins the
recipe path and stayed green through the extraction unedited; what it cannot see is whether
the seam still holds when the loop is somebody else's.

Every failure asserted here was measured on this checkout by deliberately breaking the seam
and re-running — the mechanism is named in each docstring, not guessed at.
"""
from __future__ import annotations

import json

import pytest

from manyruns import vocab as _vocab

np = pytest.importorskip("numpy")


def _ctx(tmp_path, run_id="RUN"):
    return {"out_dir": tmp_path, "run_id": run_id, "plots": [], "target_dim": 2}


def _embeds(value, dim=2, rows=8):
    """An executor that assigns a NEW embedding object — so `_persist` writes it and
    `_attach_geometry` fires, both of which key off object identity."""
    def step(name, params, state, g, ctx):
        state["emb"] = np.full((rows, dim), float(value))
    return step


def _metrics(monkeypatch, fn=None):
    """A stand-in for manylatents' registry — these must run with no private stack."""
    from manyruns.pipeline import suite

    monkeypatch.setattr(
        suite, "_compute_metric",
        lambda: fn or (lambda name, embeddings=None, dataset=None: np.float32(embeddings[0, 0])),
    )


def _drive(steps, executors, ctx, *, indices=None, carry=None, state=None, g=None,
           on_step=None, finish=True):
    """Apply a list of steps one at a time the way a session will: one carry, opened once,
    and an index the CALLER owns. Returns `(records, carry, state, g)`."""
    from manyruns.pipeline import runner

    state = runner._new_state(X=np.zeros((8, 3))) if state is None else state
    g = {} if g is None else g
    carry = runner.new_carry() if carry is None else carry
    recs: list = []
    for i, (step, fn) in enumerate(zip(steps, executors)):
        runner.apply_step(step, state, g, dispatch={"latent": fn}, ctx=ctx,
                          index=indices[i] if indices else i,
                          carry=carry, on_step=on_step, steps=recs)
    if finish:
        runner.finish_carry(carry, ctx)
    return recs, carry, state, g


def _latent(*names):
    return [{"name": n, "group": "latent", "params": {}} for n in names]


# ── never raises ─────────────────────────────────────────────────────────────────────────
#
# The atomic path has a `for` loop around every step and a `_finalize` after it, so a
# raising step is contained by construction. A session has neither: `apply_step` IS the call
# the REPL makes, and an exception out of it ends the session and loses the state lineage
# with it. This is the property the extraction most has to preserve.


@pytest.mark.parametrize("step,executor,outcome,detail_has", [
    ({"name": "x", "group": "latent"},
     lambda *a: (_ for _ in ()).throw(ValueError("kaboom")), "error", "kaboom"),
    ({"name": "x"}, None, "skipped", "no 'group'"),
    ({"name": "x", "group": "sweep"}, None, "skipped", "executor"),
])
def test_apply_step_reports_a_bad_step_instead_of_raising(tmp_path, step, executor,
                                                          outcome, detail_has):
    """Three ways a step can fail to run, and none of them may reach the caller as an
    exception. Verified by mutation: re-raising from the `except Exception` handler makes the
    first case propagate `ValueError` out of `apply_step`."""
    from manyruns.pipeline import runner

    dispatch = {"latent": executor} if executor else {}
    rec = runner.apply_step(step, runner._new_state(X=np.zeros((8, 3))), {},
                            dispatch=dispatch, ctx=_ctx(tmp_path), index=0,
                            carry=runner.new_carry())

    assert rec["outcome"] == outcome
    assert detail_has in rec["detail"]
    assert "state" not in rec and "started" not in rec   # settled on every exit path


def test_a_step_that_declined_lets_the_next_one_run(tmp_path):
    """The point of not raising: the lineage survives its own bad step. A session that lost
    `state` on a typo would make every experiment one mistake from the start."""
    def boom(name, params, state, g, ctx):
        raise RuntimeError("no")

    recs, _, state, _ = _drive(_latent("bad", "good"), [boom, _embeds(7)], _ctx(tmp_path))

    assert [r["outcome"] for r in recs] == ["error", "ok"]
    assert state["emb"][0, 0] == 7.0


# ── index belongs to the caller ──────────────────────────────────────────────────────────


def test_the_caller_owns_the_index_so_a_repeated_step_keeps_both_artifacts(tmp_path):
    """`index` addresses the artifact file (`NN-name_slot.npy`), so a driver that restarts it
    at 0 per action makes the second `phate` overwrite the first's embedding — silently, with
    both records reporting `ok` and both paths still resolving.

    Measured both ways on this checkout: monotonic 0/1/2 leaves three files and the first
    `a` artifact loads back as the FIRST array; index pinned at 0 leaves two files and the
    first `a` artifact loads back as the THIRD array."""
    from manyruns import artifacts

    steps, execs = _latent("a", "b", "a"), [_embeds(1), _embeds(2), _embeds(3)]

    recs, _, _, _ = _drive(steps, execs, _ctx(tmp_path, "MONO"))
    files = sorted(p.name for p in artifacts.root(tmp_path, "MONO").iterdir())
    assert files == ["00-a_emb.npy", "01-b_emb.npy", "02-a_emb.npy", "COMPLETE"]
    # the first `a`'s array is still the first `a`'s array
    assert np.load(recs[0]["artifacts"]["emb"])[0, 0] == 1.0
    assert np.load(recs[2]["artifacts"]["emb"])[0, 0] == 3.0

    pinned, _, _, _ = _drive(steps, execs, _ctx(tmp_path, "PINNED"), indices=[0, 0, 0])
    assert np.load(pinned[0]["artifacts"]["emb"])[0, 0] == 3.0    # destroyed, silently
    assert pinned[0]["outcome"] == "ok"                           # ...and nothing says so


# ── carry: the three run-scoped accumulators ─────────────────────────────────────────────


def test_one_carry_across_calls_makes_geometry_a_delta_not_an_appearance(tmp_path,
                                                                        monkeypatch):
    """`carry["geometry"]` is what makes `rec["geometry"]` a `[from, to]` rather than a
    snapshot. Held across calls it reads `[None, 1] [1, 2] [2, 3]`; re-created per call it
    reads `[None, 1] [None, 2] [None, 3]` — every step an appearance, no change ever
    expressible, and nothing in the record admitting the difference."""
    from manyruns.pipeline import runner

    _metrics(monkeypatch)                       # lid == emb[0, 0]
    steps, execs = _latent("a", "b", "c"), [_embeds(1), _embeds(2), _embeds(3)]

    shared, _, _, _ = _drive(steps, execs, _ctx(tmp_path, "SHARED"))
    assert [r["geometry"]["lid"] for r in shared] == [[None, 1.0], [1.0, 2.0], [2.0, 3.0]]

    # the same steps with a fresh carry per call — what dropping the accumulator costs
    ctx, per_call = _ctx(tmp_path, "FRESH"), []
    state = runner._new_state(X=np.zeros((8, 3)))
    for i, (step, fn) in enumerate(zip(steps, execs)):
        runner.apply_step(step, state, {}, dispatch={"latent": fn}, ctx=ctx, index=i,
                          carry=runner.new_carry(), steps=per_call)
    assert [r["geometry"]["lid"] for r in per_call] == [[None, 1.0], [None, 2.0], [None, 3.0]]


def test_the_manifest_describes_the_whole_folder_because_written_accumulates(tmp_path):
    """`artifacts.finish` writes the manifest WHOLE, so `carry["written"]` must hold every
    path the lineage wrote. With a fresh carry per call the last write wins and a 3-file
    folder documents itself as holding 1 — measured."""
    from manyruns import artifacts

    _drive(_latent("a", "b", "c"), [_embeds(1), _embeds(2), _embeds(3)], _ctx(tmp_path))
    folder = artifacts.root(tmp_path, "RUN")
    manifest = json.loads((folder / artifacts.DONE).read_text())

    assert len(manifest) == 3
    assert len(manifest) == len([p for p in folder.iterdir() if p.name != artifacts.DONE])


# ── the lifecycle stays out of the step ──────────────────────────────────────────────────


def test_apply_step_never_marks_the_lineage_complete(tmp_path):
    """`COMPLETE` is the reader's only guarantee that a state folder is done. Written per
    step it is True while the session is still open — the one thing the marker exists to
    rule out. Measured: calling `finish_carry` after every step makes `complete()` True with
    two steps still to run."""
    from manyruns import artifacts
    from manyruns.pipeline import runner

    ctx = _ctx(tmp_path)
    recs, carry, state, g = _drive(_latent("a", "b"), [_embeds(1), _embeds(2)], ctx,
                                   finish=False)

    assert artifacts.complete(artifacts.root(tmp_path, "RUN")) is False
    runner.finish_carry(carry, ctx)
    assert artifacts.complete(artifacts.root(tmp_path, "RUN")) is True


def test_a_lineage_that_wrote_nothing_leaves_no_marker(tmp_path):
    """An empty state folder and an interrupted one must not look alike."""
    from manyruns import artifacts
    from manyruns.pipeline import runner

    def reads_only(name, params, state, g, ctx):
        g["read"] = 1.0

    ctx = _ctx(tmp_path)
    _drive(_latent("a"), [reads_only], ctx)

    assert runner.finish_carry(runner.new_carry(), ctx) is None
    assert artifacts.complete(artifacts.root(tmp_path, "RUN")) is False


def test_apply_step_does_not_finalize_or_mint_identity(tmp_path):
    """The run-scoped work `apply_step` deliberately does not do. A step that finalised would
    re-run the full suite and re-stamp `at`/`run_id` after every user action, making a
    session's g-vector a different object each time it was looked at."""
    recs, _, _, g = _drive(_latent("a"), [_embeds(1)], _ctx(tmp_path))

    assert set(recs[0]) >= {"index", "name", "group", "params", "outcome", "detail",
                            "seconds", "produced", "emitted", "artifacts"}
    # none of `_finalize`'s run-scoped keys leak into a step record...
    assert not {"run_id", "spec_id", "at", "g_vector", "trace", "status"} & set(recs[0])
    # ...and the executor's own keys are the ONLY thing in g: no suite was measured
    assert set(g) == set()


# ── the step is a dict, which is the whole point ─────────────────────────────────────────


def test_declared_params_reach_the_executor(tmp_path):
    """The defect the seam exists to make impossible: an interaction surface whose unit is a
    bare step NAME cannot carry `params`, so `n_components: 3` never reaches the fit and
    `final_dim` silently stays at the engine default."""
    from manyruns.pipeline import runner

    seen = {}

    def capture(name, params, state, g, ctx):
        seen.update(params)
        state["emb"] = np.zeros((8, int(params.get("n_components", 2))))

    runner.apply_step({"name": "phate", "group": "latent", "params": {"n_components": 3}},
                      (state := runner._new_state(X=np.zeros((8, 3)))), {},
                      dispatch={"latent": capture}, ctx=_ctx(tmp_path), index=0,
                      carry=runner.new_carry())

    assert seen == {"n_components": 3}
    assert state["emb"].shape[1] == 3


def test_the_record_carries_a_copy_of_the_params_not_the_recipe_s_own_dict(tmp_path):
    """An executor that mutates its params must not rewrite the recipe's record of what it
    was asked to do — a session re-issuing the same step dict would then be issuing a
    different step the second time."""
    from manyruns.pipeline import runner

    def mutates(name, params, state, g, ctx):
        params["injected"] = True

    step = {"name": "x", "group": "latent", "params": {"k": 1}}
    rec = runner.apply_step(step, runner._new_state(X=np.zeros((8, 3))), {},
                            dispatch={"latent": mutates}, ctx=_ctx(tmp_path), index=0,
                            carry=runner.new_carry())

    assert step["params"] == {"k": 1}
    assert rec["params"] == {"k": 1, "injected": True}


# ── the observer contract survives the extraction ────────────────────────────────────────


def test_the_observer_sees_the_running_record_inside_the_list_it_renders(tmp_path):
    """`on_step(rec, steps)` hands a live panel the list it renders, and `_progress_panel`
    reads a declared step as `queued` unless a record for it is in that list. So `rec` has to
    join the list BEFORE the executor runs, inside `apply_step` — appending the returned
    record in the caller instead gives the observer a one-element list every time (measured:
    lengths 1, 1, 1 instead of 1, 2, 3), and the panel re-renders every FINISHED step as
    queued for the whole run."""
    lengths = []

    def observe(rec, steps):
        if rec.get("state") == "running":
            lengths.append(len(steps))
            assert steps[-1] is rec

    _drive(_latent("a", "b", "c"), [_embeds(1), _embeds(2), _embeds(3)],
           _ctx(tmp_path), on_step=observe)

    assert lengths == [1, 2, 3]


def test_an_observer_that_raises_cannot_kill_the_step(tmp_path):
    """Same rule as the run loop's, one grain finer: in a session the observer is between the
    scientist and their own state lineage."""
    from manyruns.pipeline import runner

    def hostile(rec, steps):
        raise RuntimeError("the terminal went away")

    rec = runner.apply_step({"name": "a", "group": "latent"},
                            runner._new_state(X=np.zeros((8, 3))), {},
                            dispatch={"latent": _embeds(1)}, ctx=_ctx(tmp_path), index=0,
                            carry=runner.new_carry(), on_step=hostile)

    assert rec["outcome"] == "ok"


# ── the extraction itself ────────────────────────────────────────────────────────────────


def test_the_recipe_loop_is_apply_step_and_nothing_else(tmp_path, monkeypatch):
    """THE acceptance property for the extraction: driving `apply_step` by hand — one carry,
    caller's index, one `finish_carry` — reproduces `_run_steps` record for record. If these
    ever diverge, manyruns has two step loops again, which `runner`'s own header records as
    the failure that produced three divergent g-vector key sets once already."""
    from manyruns import artifacts
    from manyruns.pipeline import runner

    _metrics(monkeypatch)
    steps, execs = _latent("a", "b", "a"), [_embeds(1), _embeds(2), _embeds(3)]

    # the loop — one dispatch entry that hands each successive step its own executor, so
    # both sides below run byte-identical compute
    calls = iter(execs)
    dispatch = {"latent": lambda n, p, s, g, c: next(calls)(n, p, s, g, c)}
    looped = runner._run_steps({"name": "r", "steps": steps},
                               runner._new_state(X=np.zeros((8, 3))), {},
                               dispatch=dispatch, ctx=_ctx(tmp_path, "LOOP"))

    # the same thing driven a step at a time
    by_hand, _, _, _ = _drive(steps, execs, _ctx(tmp_path, "HAND"))

    def strip(recs):
        out = []
        for r in recs:
            d = {k: v for k, v in r.items() if k != "seconds"}
            d["artifacts"] = {k: v.rsplit("/", 1)[-1] for k, v in d.get("artifacts", {}).items()}
            out.append(d)
        return out

    assert strip(looped) == strip(by_hand)
    for run_id in ("LOOP", "HAND"):
        assert artifacts.complete(artifacts.root(tmp_path, run_id)) is True
        assert len(json.loads((artifacts.root(tmp_path, run_id) / artifacts.DONE).read_text())) == 3


# ── the frame reaches the step loop, and a mask reaches the selection ─────────────────────
#
# The `prep` group, its executors and the narrowing rule in `vocab` all landed first
# and NOTHING RAN:
# `_run_prep_step` declined every prep step with "this state carries no frame", because
# `_new_state` seeded no frame to view. These pin the wiring that makes it run.
#
# What a narrowing has to move together is FIVE row-indexed things — `rows`, `X`,
# `state["labels"]`, `ctx["color"]` and `ctx["array"]` — plus two derived arrays it has to
# clear. `frame.py`'s docstring is the argument for why that count is not something a step
# author can be asked to get right: every one of them fails SILENTLY when missed
# (`io.py:106` drops the colouring on a length mismatch; `_ml_latent` embeds `ctx["array"]`
# and would embed the cells the filter just removed).


def _framed(n=6, d=4, labels=("d0", "d0", "d3", "d3", "d9", "d9")):
    """A state as `run_inproc` builds one: a working matrix, and a frame behind it."""
    from manyruns.pipeline import runner

    X = np.arange(n * d, dtype=np.float32).reshape(n, d)
    return runner._new_state(X=X, labels=np.array(labels[:n]), seed=0,
                             counts=X.copy(), genes=np.array(list("wxyz")[:d]))


def _prep_ctx(tmp_path=None, **extra):
    ctx = {"out_dir": tmp_path, "run_id": "PREP", "plots": [], "target_dim": 2, "caveats": []}
    ctx.update(extra)
    return ctx


def _apply_prep(state, out, ctx=None, name="f", group="prep", params=None, g=None):
    """One step whose executor returns `out`, applied the way the loop applies it.

    `g` is a parameter because the g-vector is part of what a refused transition must not
    move — see `test_a_refused_transition_leaves_no_report_in_the_g_vector`."""
    from manyruns.pipeline import runner

    ctx = _prep_ctx() if ctx is None else ctx
    dispatch = {group: lambda name, params, st, g, c: out}
    return runner.apply_step({"name": name, "group": group, "params": params or {}},
                             state, {} if g is None else g, dispatch=dispatch, ctx=ctx,
                             index=0, carry=runner.new_carry())


def test_a_run_starts_with_every_row_and_column_selected():
    """A selection is the run's view of its own data, and it starts complete. Seeded from
    `counts` rather than `X`, because on pbmc3k those are 2700 x 32738 and 2700 x 50 — the
    gene axis a column filter narrows lives only in the first."""
    state = _framed()

    assert state["rows"].tolist() == [0, 1, 2, 3, 4, 5]
    assert state["cols"].tolist() == [0, 1, 2, 3]
    assert state["frame"]["counts"].shape == (6, 4)


def test_an_executor_returning_a_mask_narrows_the_selection():
    """The mask meets the selection HERE and nowhere else. A step author returns which rows
    survive; they never touch `labels`, `counts`, `emb` or `ctx["color"]`, which is why they
    cannot forget one."""
    state = _framed()
    rec = _apply_prep(state, {"mask": np.array([True, True, False, True, False, False]),
                              "axis": "rows"})

    assert state["rows"].tolist() == [0, 1, 3]
    assert rec["outcome"] == "ok"
    assert rec["dropped"] == {"axis": "rows", "n": 3, "remaining": 3}


def test_narrowing_clears_every_derived_array_and_records_which():
    """The invalidation rule, in the record. An embedding fitted over six cells is not an
    embedding of the three that remain — PHATE's diffusion operator saw the other three, so
    `emb[keep]` would LOOK like an embedding of the survivors and not be one."""
    state = _framed()
    state["emb"], state["pseudotime"] = np.zeros((6, 2)), np.zeros(6)
    rec = _apply_prep(state, {"mask": np.array([True, False, True, False, True, False]),
                              "axis": "rows"})

    assert state["emb"] is None and state["pseudotime"] is None
    assert rec["invalidated"] == ["emb", "pseudotime"]


def test_a_column_mask_narrows_genes_and_leaves_cells_alone():
    """Zach's HVG is this branch. One primitive, two axes."""
    state = _framed()
    _apply_prep(state, {"mask": np.array([True, False, True, False]), "axis": "cols"})

    assert state["cols"].tolist() == [0, 2]
    assert state["rows"].tolist() == [0, 1, 2, 3, 4, 5]
    assert state["genes"].tolist() == ["w", "y"]


def test_an_executor_returning_a_matrix_replaces_X_and_narrows_nothing():
    """`normalize` and `transform` produce a new working matrix. Treating every prep result as
    a narrowing would invalidate the embedding on a step that removed no cells."""
    state = _framed()
    state["emb"] = np.zeros((6, 2))
    rec = _apply_prep(state, {"X": np.ones((6, 4))})

    assert np.allclose(state["X"], 1.0)
    assert state["emb"] is not None, "nothing was removed, so nothing went stale"
    assert "dropped" not in rec and "invalidated" not in rec


def test_a_string_return_still_means_reported():
    """The executor contract is WIDENED, not replaced — every analysis step returns a status
    string or None and must keep meaning exactly what it meant."""
    rec = _apply_prep(_framed(), "nothing to measure", group="analysis")

    assert rec["outcome"] == "reported" and rec["detail"] == "nothing to measure"


def test_labels_and_counts_cannot_desync_because_neither_is_stored_narrowed():
    """The property the model exists for, asserted at the loop rather than in the primitives."""
    from manyruns.pipeline import frame as _frame

    state = _framed()
    _apply_prep(state, {"mask": np.array([True, False, True, False, True, False]),
                        "axis": "rows"})
    view = _frame.view(state["frame"], state["rows"], state["cols"])

    assert view["counts"].shape[0] == len(view["labels"]) == 3
    assert state["X"].shape[0] == 3, "the working matrix moved with them"
    assert view["labels"].tolist() == ["d0", "d3", "d9"]


def test_a_narrowing_writes_no_artifact_for_the_array_it_just_cleared(tmp_path):
    """`_persist` runs AFTER the executor and writes any `PERSISTED` slot this step replaced.
    A narrowing replaces `emb` with None, so the question is whether the manifest gains a
    None. Measured here rather than assumed: `_persist` skips a `None` value, so the record
    of the narrowing step carries `invalidated` and no `artifacts` — the embedding's own file,
    written by the step that produced it, stays where it is."""
    from manyruns.pipeline import runner

    ctx = _prep_ctx(tmp_path)
    state = _framed()
    carry = runner.new_carry()
    runner.apply_step({"name": "e", "group": "latent", "params": {}}, state, {},
                      dispatch={"latent": _embeds(1, rows=6)}, ctx=ctx, index=0, carry=carry)
    rec = runner.apply_step({"name": "f", "group": "prep", "params": {}}, state, {},
                            dispatch={"prep": lambda *a: {"mask": np.array([True] * 3 + [False] * 3),
                                                          "axis": "rows"}},
                            ctx=ctx, index=1, carry=carry)

    assert rec["invalidated"] == ["emb"]
    assert "artifacts" not in rec, rec.get("artifacts")
    written = sorted(p.name for p in (tmp_path / "state").rglob("*.npy"))
    assert written == ["00-e_emb.npy"], written


def test_the_shape_a_later_steps_limits_resolve_against_moves_with_the_selection(tmp_path):
    """`bounds.shape_of` reads `state["emb"]`, then `ctx["array"]`, then `state["X"]` — and
    `ctx["array"]` is the run-scoped input the manylatents engine also EMBEDS
    (`runner.py:790`). Left alone by a narrowing it reports the pre-filter shape, so a
    `limits: {n_components: [n_samples, …]}` resolves against cells that are gone and the
    engine fits the ones the filter removed. Both are the same key, so both move here."""
    from manyruns.pipeline import bounds, runner

    array = np.arange(24, dtype=np.float32).reshape(6, 4)
    ctx = _prep_ctx(tmp_path, array=array)
    state = runner._new_state(X=array, labels=None, seed=0)
    _apply_prep(state, {"mask": np.array([True, True, False, True, False, False]),
                        "axis": "rows"}, ctx=ctx)

    assert bounds.shape_of(state, ctx) == (3, 4)
    assert ctx["array"].shape == (3, 4)
    assert ctx["array"].tolist() == array[[0, 1, 3]].tolist()


def test_the_matrix_a_transform_produced_is_the_one_the_next_step_sees(tmp_path):
    """The same key, on the other branch. `_ml_latent` embeds `ctx["array"]` when no
    embedding exists yet, so a `transform` step that replaced `state["X"]` alone would be a
    step reporting `ok` while the engine embedded the RAW counts — the exact failure
    `prep.TRANSFORMS` refuses a typo'd method to avoid."""
    ctx = _prep_ctx(tmp_path, array=np.zeros((6, 4), dtype=np.float32))
    state = _framed()
    _apply_prep(state, {"X": np.ones((6, 4))}, ctx=ctx)

    assert np.allclose(ctx["array"], 1.0)
    assert ctx["array"] is state["X"]


def test_a_prep_step_declines_on_a_state_that_carries_no_data_at_all():
    """A named engine dataset is loaded INSIDE manylatents, so manyruns holds no matrix
    (`run_manylatents`: `X` is None, `counts` is None, and `bounds.shape_of` records the same
    limitation). `transform` on that state would hand `np.log1p` a None; a declined step with
    a reason is what the record is for."""
    from manyruns.pipeline import runner

    rec = runner.apply_step(
        {"name": "transform", "group": "prep", "params": {}},
        runner._new_state(), {}, dispatch=runner.dispatch_for(_vocab.INPROC),
        ctx=_prep_ctx(), index=0, carry=runner.new_carry())

    assert rec["outcome"] == "skipped"
    assert "no data" in rec["detail"]


def test_a_prep_step_actually_runs_in_a_recipe_now(tmp_path, monkeypatch):
    """END TO END on the default engine, which is the whole point of this task: before it,
    every prep step in every recipe reported `skipped (this state carries no frame…)`."""
    # The prep executors DELEGATE to `manylatents.singlecell.preprocessing`, which rides the
    # `[omics]` extra. CI installs `manylatents` and not that extra, so without this the step
    # errors with the install message and the assertion below reads as a logic failure.
    pytest.importorskip("manylatents.singlecell.preprocessing",
                        reason="prep steps delegate to the [omics] extra")
    from manyruns import pipeline

    seen = {}

    def embeds(state, g, params, out_dir, plots, target_dim):
        seen["X"] = np.array(state["X"], copy=True)
        state["emb"] = np.zeros((state["X"].shape[0], 2))

    monkeypatch.setitem(pipeline._INPROC_STEPS, "embeds", embeds)
    counts = np.arange(24, dtype=np.float32).reshape(6, 4)
    out = pipeline.run_inproc(
        counts, {"name": "r", "steps": [
            {"name": "transform", "group": "prep", "params": {"method": "log1p"}},
            {"name": "embeds", "group": "latent", "params": {}}]},
        tmp_path, labels=np.array(["d0"] * 3 + ["d9"] * 3))

    assert [s["outcome"] for s in out["steps"]] == ["ok", "ok"]
    assert np.allclose(seen["X"], np.log1p(counts))
    assert out["g_vector"]["transform.method"] == "log1p"


def test_the_colour_the_plot_is_drawn_with_is_rebuilt_after_a_narrowing(tmp_path, monkeypatch):
    """HAZARD 7, end to end and read back off disk. `ctx["color"]` is built ONCE at run open
    (`runner.py:855`) from the full label array. `_save_scatter` drops the colouring entirely
    on a length mismatch (`io.py:106`) rather than raising, so a narrowing that did not
    rebuild it produces a grey plot and a run that says nothing about why.

    The assertion is the figure SPEC beside the PNG, not a spy on the call: it is what the
    published figure is remade from, so a colour that reaches it has reached the plot."""
    import sys
    import types

    from manyruns import figspec, pipeline
    from manyruns.pipeline import prep as _prep

    api = types.ModuleType("manylatents.api")
    api.run = lambda **kw: {"embeddings": np.zeros((kw["input_data"].shape[0], 2)), "scores": {}}
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)
    monkeypatch.setitem(_prep._PREP_STEPS, "half",
                        lambda view, X, params: {"mask": np.array([True, False] * 3),
                                                 "axis": "rows"})

    out = pipeline.run_manylatents(
        {"name": "r", "steps": [{"name": "half", "group": "prep", "params": {}},
                                {"name": "phate", "group": "latent", "params": {}}]},
        array=np.arange(24, dtype=np.float32).reshape(6, 4),
        labels=np.array(["d0", "d0", "d3", "d3", "d9", "d9"]), out_dir=tmp_path)

    spec = figspec.load(out["plots"][0])
    assert spec["labeled"] is True, "the colouring survived the filter"
    assert spec["coords"].shape[0] == 3
    # float32, and the SPACING is the assertion: `numeric_time` keeps day0/day3/day9 at 0, ⅓, 1
    # rather than the even ranks a re-derivation from three surviving categories would give.
    assert np.allclose(spec["color_values"], [0.0, 1 / 3, 1.0]), spec["color_values"]


# ── what a narrowing is, and what it is not ──────────────────────────────────────────────
#
# Every test below was measured by breaking the line it names and re-running. Four of them
# cover mutations that left the whole suite green when Task 5 landed, which is the reason
# they exist as separate cases rather than as extra assertions on the ones above.


def test_a_filter_that_removes_no_cell_leaves_the_embedding_where_it_is():
    """`vocab.step_narrows` has an explicit carve-out for this — `NARROWING_STEPS` is a DICT
    keyed by the param that makes a step narrow, so `detect_doublets` with `remove: false`
    (spec §5: "flags without narrowing") is NOT a narrowing at plan time. The run loop has to
    agree, or a recipe `unmet` calls legal comes back with the embedding cleared and the next
    step recorded as `skipped (mioflow needs an embedding to run on)`.

    Measured before the guard, on `[embeds, detect_doublets(remove=False), mioflow]`: plan
    time `step_narrows -> False` and `unmet -> frozenset()`, run time `invalidated: ['emb']`
    and mioflow skipped. `narrate.CLEARED_FACT` cannot explain it either, because
    `vocab.invalidated` is empty — nothing anywhere names the filter as the cause."""
    state = _framed()
    state["emb"], state["pseudotime"] = np.zeros((6, 2)), np.zeros(6)
    rec = _apply_prep(state, {"mask": np.ones(6, dtype=bool), "axis": "rows"})

    assert state["emb"] is not None and state["pseudotime"] is not None
    assert "invalidated" not in rec
    # The count is still RECORDED: "filter_cells dropped 0 cells" is a measurement (on pbmc3k
    # `min_genes=200` drops exactly 0), and an absent record cannot say it.
    assert rec["dropped"] == {"axis": "rows", "n": 0, "remaining": 6}


def test_a_column_filter_moves_the_working_matrix_the_next_step_actually_fits():
    """The mirror of the row rule, and without it a gene filter reaches nothing that computes.

    `state["X"]` and `ctx["array"]` are what a latent step embeds (`_ml_latent` passes
    `ctx["array"]` as `input_data`) and what `bounds.shape_of` resolves `limits:` against.
    Measured before the fix, with `hvg` keeping 3 of 6 genes: the record said
    `dropped {'axis': 'cols', 'n': 3}` and `invalidated ['emb']`, and the matrix the next
    `embeds` received was (12, 6) — every gene — with the re-fitted embedding
    `np.array_equal` to the one just thrown away and `n_features` still 6."""
    array = np.arange(24, dtype=np.float32).reshape(6, 4)
    ctx = _prep_ctx(array=array)
    state = _framed()
    state["X"] = array.copy()
    _apply_prep(state, {"mask": np.array([True, False, True, False]), "axis": "cols"}, ctx=ctx)

    assert state["X"].shape == (6, 2) and ctx["array"].shape == (6, 2)
    assert state["X"][:, 0].tolist() == array[:, 0].tolist()
    from manyruns.pipeline import bounds

    assert bounds.shape_of(state, ctx) == (6, 2), "a later step's limits resolve against it"


def test_a_column_filter_is_refused_when_the_working_matrix_has_no_gene_axis():
    """The case the width guard cannot repair, refused rather than half-applied.

    Today `loading._anndata_matrix` hands over `obsm["X_pca"]` — measured (2700, 50) from a
    (2700, 32738) file — so a `hvg`/`filter_genes` step declared AFTER a reduction narrows
    `counts`/`genes` while the matrix every downstream step fits keeps all 32,738. Recording
    `dropped {'n': 30738}` for a narrowing nothing computes over is the silent-failure mode
    the frame model exists to remove, and the honest answer is that a gene filter has to run
    before the gene axis is destroyed."""
    from manyruns.pipeline import runner

    state = runner._new_state(X=np.zeros((6, 2), dtype=np.float32), labels=None, seed=0,
                              counts=np.arange(24, dtype=np.float32).reshape(6, 4),
                              genes=np.array(list("wxyz")))
    rec = _apply_prep(state, {"mask": np.array([True, False, True, False]), "axis": "cols"})

    assert rec["outcome"] == "error"
    assert "6 x 2" in rec["detail"] and "4" in rec["detail"]
    assert state["cols"].tolist() == [0, 1, 2, 3], "refused before anything moved"
    assert state["genes"].tolist() == ["w", "x", "y", "z"]


def test_a_column_narrowing_clears_the_derived_arrays_too():
    """The clear is OUTSIDE the `axis == "rows"` guard and must stay there, because
    `vocab.step_narrows` is axis-blind: it clears `embedding` on a column narrowing too, on
    the premise of FITTING rather than length ("a kNN graph over 32,738 genes is not the graph
    over the 2,000 `hvg` kept"). Clearing only on rows leaves `unmet` refusing a recipe whose
    `emb` is still populated — and measured on `[phate, cols-prep, phate]`, the second latent
    step CHAINS off the stale embedding: `input_data` (6, 2) instead of the filtered matrix."""
    state = _framed()
    state["X"] = np.arange(24, dtype=np.float32).reshape(6, 4)
    state["emb"] = np.zeros((6, 2))
    rec = _apply_prep(state, {"mask": np.array([True, False, True, False]), "axis": "cols"})

    assert state["emb"] is None
    assert rec["invalidated"] == ["emb"]


def test_two_narrowings_compose_in_the_frame_s_own_index_space():
    """What makes the selection chain the provenance. The second mask is sized against what
    the second step SAW (five cells), and the result says which rows OF THE FILE survived.

    Mutation-checked: `state[axis] = np.flatnonzero(mask)` — the mask applied without being
    composed — leaves the whole suite green at 1054 and gives rows [2, 3, 4] here instead of
    [3, 4, 5], with `labels` ['c2','c3','c4'] beside an `X` holding original rows 3, 4, 5.
    Nothing else drives a SECOND narrowing through `apply_step`, and `preprocess.yaml`
    declares three filters in a row."""
    state = _framed()
    _apply_prep(state, {"mask": np.array([False, True, True, True, True, True]),
                        "axis": "rows"})
    _apply_prep(state, {"mask": np.array([False, False, True, True, True]), "axis": "rows"})

    assert state["rows"].tolist() == [3, 4, 5]
    assert state["labels"].tolist() == ["d3", "d9", "d9"]
    assert state["X"].tolist() == np.arange(24, dtype=np.float32).reshape(6, 4)[3:].tolist()


def test_the_view_a_prep_step_sees_is_the_gene_axis_and_not_the_reduction():
    """`rows`/`cols` are seeded from `counts` first, and on pbmc3k that is the difference
    between 32,738 columns and 50. Seeding from `X` instead leaves the suite green — measured
    — because the only test of the seeding passes `counts=X.copy()`, the same shape. Here a
    filter would then threshold on the first 3 of 8 genes and report a gene count it never
    looked at."""
    from manyruns.pipeline import runner

    seen = {}
    counts = np.arange(48, dtype=np.float32).reshape(6, 8)
    state = runner._new_state(X=np.zeros((6, 3), dtype=np.float32), labels=None, seed=0,
                              counts=counts, genes=np.array(list("abcdefgh")))

    def spy(view, X, params):
        seen["counts"], seen["genes"] = view["counts"], view["genes"]
        return {"X": X}

    from manyruns.pipeline import prep as _prep

    _prep._PREP_STEPS["spy"] = spy
    try:
        runner.apply_step({"name": "spy", "group": "prep", "params": {}}, state, {},
                          dispatch=runner.dispatch_for(_vocab.INPROC), ctx=_prep_ctx(), index=0,
                          carry=runner.new_carry())
    finally:
        _prep._PREP_STEPS.pop("spy", None)

    assert seen["counts"].shape == (6, 8)
    assert seen["genes"].tolist() == list("abcdefgh")


def test_a_mask_sized_against_the_wrong_frame_errors_before_anything_is_cleared():
    """The refusal has to come BEFORE the invalidation, or a step that ends `error` still
    destroys the embedding it never legitimately narrowed. `frame.narrow` raises on a
    wrong-length mask (it is the one way a caller can desync a selection), and `apply_step`
    turns that into this step's `error` — which is only a safe answer if the state it leaves
    behind is the state the step started from."""
    state = _framed()
    state["emb"] = np.zeros((6, 2))
    rec = _apply_prep(state, {"mask": np.array([True, False, True]), "axis": "rows"})

    assert rec["outcome"] == "error" and "mask has 3 entries" in rec["detail"]
    assert state["emb"] is not None and state["rows"].tolist() == [0, 1, 2, 3, 4, 5]
    assert "invalidated" not in rec and "dropped" not in rec


# ── the g-vector is part of the state a refused transition must not move ─────────────────


def test_a_refused_transition_leaves_no_report_in_the_g_vector():
    """A prep step's `report` reaches `g` only if its transition APPLIED. It used to reach `g`
    inside `_run_prep_step`, one frame before `_apply_transition` ran, so a step recorded as
    `error` still contributed its numbers.

    Measured on this checkout before the fix, with the REAL `filter_genes` against a state
    whose working matrix had already been reduced (`X` 4 x 2 beside a 6-gene frame): the
    record came back `outcome='error'`, detail *"'filter_genes' filters genes, but 'X' is
    4 x 2 against a selection of 6 genes"*, `state['cols']` untouched at [0..5] — and `g` held
    `{'filter_genes.min_cells': 2, 'filter_genes.dropped': 3}`. Three genes dropped, recorded,
    for a narrowing that never happened. That is the "record for work that was rolled back"
    mode `_apply_transition` refuses one slot over ("refused BEFORE the selection moves, so the
    state a step leaves on `error` is the state it started from"); the g-vector is state too.

    The second half is the positive control, and it is what stops the fix from being "never
    write the report at all": the same step on a state whose matrix DOES carry the gene axis
    drops the same three genes and says so."""
    # The prep executors DELEGATE to `manylatents.singlecell.preprocessing`, which rides the
    # `[omics]` extra. CI installs `manylatents` and not that extra, so without this the step
    # errors with the install message and the assertion below reads as a logic failure.
    pytest.importorskip("manylatents.singlecell.preprocessing",
                        reason="prep steps delegate to the [omics] extra")
    from manyruns.pipeline import runner

    # col0 [5,0,2,0] and col4 [7,0,9,0] are in 2 cells, col2 [3,4,0,6] in 3; col1/col3 are
    # empty and col5 [0,0,1,0] is in 1 — so `min_cells=2` drops exactly three of the six.
    counts = np.array([[5, 0, 3, 0, 7, 0],
                       [0, 0, 4, 0, 0, 0],
                       [2, 0, 0, 0, 9, 1],
                       [0, 0, 6, 0, 0, 0]], dtype=np.float32)
    genes = np.array(list("abcdef"))
    step = {"name": "filter_genes", "group": "prep", "params": {"min_cells": 2}}

    # the working matrix has no gene axis left — the transition is refused
    refused_state = runner._new_state(X=np.zeros((4, 2), dtype=np.float32), seed=0,
                                      counts=counts, genes=genes)
    refused_g: dict = {}
    refused = runner.apply_step(step, refused_state, refused_g,
                                dispatch=runner.dispatch_for(_vocab.INPROC), ctx=_prep_ctx(),
                                index=0, carry=runner.new_carry())

    assert refused["outcome"] == "error" and "gene axis is reduced" in refused["detail"]
    assert refused_state["cols"].tolist() == [0, 1, 2, 3, 4, 5], "refused before anything moved"
    assert refused_g == {}, refused_g
    assert refused["emitted"] == [], "and the record agrees it emitted nothing"

    # the same step, applied: the report is the whole point of the step, so it must land
    applied_state = runner._new_state(X=counts.copy(), seed=0, counts=counts, genes=genes)
    applied_g: dict = {}
    applied = runner.apply_step(step, applied_state, applied_g,
                                dispatch=runner.dispatch_for(_vocab.INPROC), ctx=_prep_ctx(),
                                index=0, carry=runner.new_carry())

    assert applied["outcome"] == "ok"
    assert applied["dropped"] == {"axis": "cols", "n": 3, "remaining": 3}
    assert applied_g == {"filter_genes.min_cells": 2, "filter_genes.dropped": 3,
                         "_provenance": {
                             "filter_genes.min_cells": {"kind": "setting", "stage": "step 1: filter_genes"},
                             "filter_genes.dropped": {"kind": "measurement", "stage": "step 1: filter_genes"}}}
    assert applied["emitted"] == ["filter_genes.dropped", "filter_genes.min_cells"]


def test_a_report_beside_a_mask_the_frame_refuses_reaches_nothing():
    """The same rule on the other refusal — `frame.narrow` raising on a wrong-length mask.
    Pinned separately from the case above because it takes a different route out of
    `_apply_transition` (the mask never reaches the width guard), and because a stub executor
    proves the ordering without depending on what upstream's `filter_genes` computes."""
    state = _framed()
    g: dict = {}
    rec = _apply_prep(state, {"mask": np.array([True, False, True]), "axis": "rows",
                              "report": {"f.dropped": 3}}, g=g)

    assert rec["outcome"] == "error" and "mask has 3 entries" in rec["detail"]
    assert g == {}, "a step that removed nothing recorded that it removed three"
    assert state["rows"].tolist() == [0, 1, 2, 3, 4, 5]


def test_a_matrix_whose_row_count_moved_takes_its_report_down_with_it():
    """The third refusal in `_apply_transition` — a transforming step that silently changed
    the cell count. `normalize`/`transform` return `{"X": …, "report": …}`, so this is the
    branch a real prep step reaches, and its report has to be refused with it: a
    `transform.method: log1p` in the g-vector for a transform the loop threw away is a claim
    about what the numbers downstream were computed from."""
    state = _framed()
    g: dict = {}
    rec = _apply_prep(state, {"X": np.ones((4, 4)), "report": {"transform.method": "log1p"}},
                      g=g)

    assert rec["outcome"] == "error" and "returned a matrix with 4 rows" in rec["detail"]
    assert g == {}
    assert state["X"].shape == (6, 4), "the working matrix is the one it started with"


def test_a_step_that_re_embeds_to_the_same_shape_still_says_it_produced_one():
    """`produced` is the record's output half and `_persist` writes off the same event, so the
    two must agree on what "this step produced an embedding" means. `describe` collapses to
    shape + dtype, so comparing descriptions makes a re-embed to the same shape read as
    "produced nothing" while its array is written to disk from the same record.

    Measured on the bundled `cflows` over pbmc3k (run 60921b8e06b6): step 1 `mioflow` recorded
    `outcome='ok'`, `produced=None` and `artifacts={'emb': '…/01-mioflow_emb.npy'}` holding a
    (2700, 3) array — the trajectory step the whole recipe exists for, reading as contributing
    nothing. `session._produced` is what the REPL prints after each step."""
    from manyruns.pipeline import runner

    state, ctx, carry = _framed(), _prep_ctx(), runner.new_carry()
    first = runner.apply_step({"name": "a", "group": "latent", "params": {}}, state, {},
                              dispatch={"latent": _embeds(1.0, rows=6)}, ctx=ctx, index=0,
                              carry=carry)
    second = runner.apply_step({"name": "b", "group": "latent", "params": {}}, state, {},
                               dispatch={"latent": _embeds(2.0, rows=6)}, ctx=ctx, index=1,
                               carry=carry)

    assert first["produced"] == "ndarray(6, 2) float64"
    assert second["produced"] == first["produced"], "same shape, and still a new embedding"
    assert state["emb"][0, 0] == 2.0


def test_a_step_that_only_reads_the_embedding_produced_nothing():
    """The other half, and the reason `produced` exists at all: a step could report `ok` while
    contributing nothing, and the record could not tell that apart from real work."""
    from manyruns.pipeline import runner

    state, ctx, carry = _framed(), _prep_ctx(), runner.new_carry()
    runner.apply_step({"name": "a", "group": "latent", "params": {}}, state, {},
                      dispatch={"latent": _embeds(1.0, rows=6)}, ctx=ctx, index=0, carry=carry)
    rec = runner.apply_step({"name": "reads", "group": "analysis", "params": {}}, state, {},
                            dispatch={"analysis": lambda *a: None}, ctx=ctx, index=1,
                            carry=carry)

    assert rec["produced"] is None
