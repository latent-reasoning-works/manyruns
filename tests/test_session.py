"""The interactive loop, as a driver over `runner.apply_step` rather than a second loop.

The former session loop was a SECOND implementation of "run steps and chain state", and the
weaker one. Measured on this checkout before the rewrite — `embed` (one `phate` step declaring
`n_components: 3`), the same 120x12 array, seed 42, `engine=manylatents`:

    former session loop  10 top-level keys,  1 g-vector key, final_dim 2, files: plots/phate.png
    runner.run_manylatents  18 top-level keys, 23 g-vector keys, final_dim 3, files: the plot,
                            state/<run_id>/00-phate_emb.npy and COMPLETE

and after, with the declared suite reaching both engines identically: 18 top-level keys both
sides, 43 g-vector keys both sides, the same key sets, the same `spec_id`, and step records
equal modulo `seconds`.

Every assertion below was checked by breaking the thing it names and re-running; the mechanism
is stated in each docstring rather than assumed. The in-process loop and `engine=mock` are used
throughout so the file runs with no private stack — `_suite.measure` names every declared
metric whether or not it can compute it, which is exactly why the KEY SET is the comparable
thing and the values are not.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from manyruns import app, catalog
from manyruns.session import (UNOFFERED_GROUPS, Session, available_actions, catalog_steps,
                               resolve_step)

np = pytest.importorskip("numpy")


EMBED = {"name": "embed", "steps": [{"name": "phate", "group": "latent",
                                     "params": {"n_components": 3}}]}


def _data(n=150, d=12, seed=0):
    return np.asarray(np.random.default_rng(seed).normal(size=(n, d)))


def _mock(recipe=None, **kw):
    return Session(project="p", engine="mock", modality="scrna",
                   recipe=recipe if recipe is not None else app.load_recipe("cflows"), **kw)


def _real(tmp_path, recipe=EMBED, **kw):
    pytest.importorskip("sklearn")
    pytest.importorskip("phate")
    return Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                   recipe=recipe, array=_data(), seed=0, **kw)


# ── the payload: one record, not two ─────────────────────────────────────────────────────


def test_a_session_and_a_recipe_produce_the_same_record(tmp_path):
    """THE acceptance property. The same steps, the same data, the same seed, one engine —
    the session's record and the recipe path's record must agree on everything that is not
    the wall clock or the run's own id.

    This is the sibling of `runner`'s own "one loop, one state schema, one assembler" rule.
    Before the rewrite the two disagreed on 8 top-level keys and 22 g-vector keys.
    """
    from manyruns.pipeline import runner

    session = _real(tmp_path / "sess")
    session.run_recipe()
    got = session.close()
    want = runner.run_inproc(_data(), EMBED, tmp_path / "run", seed=0)

    assert set(got) == set(want)                                  # 18 keys, both sides
    assert set(got["g_vector"]) == set(want["g_vector"])          # the comparability core
    assert got["trace"] == want["trace"] and got["status"] == want["status"]
    assert got["recipe"] == want["recipe"] == "embed"
    assert got["ok"] is want["ok"] is True

    def strip(recs):
        # `seconds` is a clock and `artifacts` carries a run-scoped id in its path. `plots`
        # carries a path too — these two runs write to different out_dirs — but it is compared
        # by BASENAME rather than dropped, so a regression that stopped attributing figures to
        # the step that drew them still fails here.
        out = []
        for r in recs:
            rec = {k: v for k, v in r.items() if k not in ("seconds", "artifacts")}
            if "plots" in rec:
                rec["plots"] = [Path(x).name for x in rec["plots"]]
            out.append(rec)
        return out

    assert strip(got["steps"]) == strip(want["steps"])


def test_a_session_that_ran_the_recipe_hashes_to_the_recipe_s_spec(tmp_path):
    """`spec_id` answers "has this exact analysis been done before". A session that issued a
    recipe's steps DID do that analysis, so it must hash to the same spec — otherwise the
    dedupe and stopping-rule semantics `runner.identity` exists for stop at the session's edge.

    Hashed over the sequence AS ISSUED, which is why it can be equal at all: `performed()`
    rebuilds `{name, group, params}` per step from the record."""
    from manyruns.pipeline import runner

    session = _real(tmp_path / "sess")
    session.run_recipe()

    assert session.close()["spec_id"] == runner.run_inproc(
        _data(), EMBED, tmp_path / "run", seed=0)["spec_id"]


def test_declared_params_reach_the_fit_so_final_dim_is_what_the_recipe_asked_for(tmp_path):
    """The sharpest defect the rewrite closes, and it changes a published number.

    `session.step` took a bare NAME and built its engine call from it, so `n_components: 3`
    had no channel to travel down: measured before the rewrite, `final_dim` was **2** out of
    the session and **3** out of the runner for the identical recipe, both reporting `ok`, with
    nothing in either record saying they differed.

    The second half uses a recipe declaring **4** — deliberately NOT the catalog's default of
    3 for `phate`. Asserting only against `embed` would pass just as well if the session
    re-resolved every step through the catalog and threw the loaded recipe away, because those
    two numbers happen to coincide."""
    session = _real(tmp_path)
    session.run_recipe()
    res = session.close()

    assert res["steps"][0]["params"] == {"n_components": 3}
    assert res["g_vector"]["final_dim"] == 3
    assert res["final_dim"] == 3
    assert session.state["emb"].shape[1] == 3

    # a recipe that disagrees with the catalog default: the RECIPE has to win
    odd = {"name": "odd", "steps": [{"name": "phate", "group": "latent",
                                     "params": {"n_components": 4}}]}
    other = _real(tmp_path / "odd", recipe=odd)
    other.run_recipe()

    assert other.steps[0]["params"] == {"n_components": 4}
    assert other.close()["g_vector"]["final_dim"] == 4


def test_a_step_issued_by_name_still_carries_the_catalog_s_params(tmp_path):
    """A person types `phate`, not a dict. The name is resolved against the catalog, so the
    typed form carries the same declared params the recipe form does — otherwise the REPL
    would be the one surface where `final_dim` silently drops back to the engine default."""
    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})
    rec = session.step("phate")["record"]

    assert rec["params"] == {"n_components": 3}
    assert session.state["emb"].shape[1] == 3


def test_the_declared_metric_suite_reaches_the_engine(tmp_path):
    """`configs/metrics/default.yaml` was inert on this path: the session had no `metrics`
    channel at all, so every stepped manylatents run silently took the engine's own defaults
    and the 20 `phate.*` scores the recipe path collects never appeared.

    Measured on the same 120x12 array, `embed`, seed 42, engine=manylatents: with the suite
    forwarded both sides report 43 g-vector keys and identical key sets; without it the runner
    reports 23 and the step emits nothing. Asserted here on the CTX rather than through the
    engine, so it holds with no private stack — `_ml_latent` reads `ctx["metrics"]` and this is
    the only thing that puts it there.

    Verified by mutation: dropping the `_default_metrics()` fallback leaves `ctx["metrics"]`
    None and fails both halves."""
    suite = catalog.load_suite()
    assert suite, "the shipped suite is empty — this test would assert nothing"

    session = Session(project="p", engine="manylatents", out_dir=tmp_path, modality="scrna",
                      recipe=EMBED, array=_data(), seed=0)
    assert session.ctx["metrics"] == suite

    seen: dict = {}
    session.dispatch["latent"] = lambda name, params, state, g, ctx: seen.update(
        metrics=ctx.get("metrics"))
    session.step("phate")
    assert seen["metrics"] == suite


# ── every step loop is reachable at all ──────────────────────────────────────────────────


def test_the_default_public_engine_is_steppable(tmp_path):
    """HISTORY, and why the gate is written as a capability rather than a list:
    `_default_engine()` used to return `real` on a public install while the session gate was
    `engine in ("mock", "manylatents")` — the inverse of that preference order. So the
    interactive mode did not exist on the shipped configuration: `manyruns init` printed
    "(interactive mode is mock/manylatents-only …)" and exited 0. `real` is gone and the
    default is now `manylatents`, but the lesson is the gate: it asks `steppable()`, so it
    cannot fall out of step with the engine list again.

    Verified by mutation: restoring that predicate in `app.cmd_init` makes this test's
    `steppable("_inproc")` assertion fail directly."""
    from manyruns.pipeline import runner

    assert runner.steppable("_inproc") is True
    assert runner.steppable("mock") is True
    assert runner.steppable("manylatents") is True
    # There used to be one honest False — the learner's engine, which ran a recipe in a single
    # downstream call and reported no per-step outcome. It is no longer an engine (§3.5), so
    # every engine is steppable and the refusal is left guarding names that are not engines.
    assert runner.steppable("external-policy") is False
    with pytest.raises(ValueError, match="no step loop"):
        runner.dispatch_for("external-policy")

    session = _real(tmp_path)
    assert session.step("phate")["ok"] is True


def test_init_opens_the_prompt_on_a_public_install(tmp_path, monkeypatch, capsys):
    """End to end at the front door: an explicit `--engine` must reach the REPL, not a
    refusal. The refusal's message was pinned by no test at all, which is how it survived
    `_default_engine` changing underneath it.

    This used to pass `--engine real`, and that is the coverage the removal costs: there is no
    longer any engine a checkout without manylatents can name here, so the only value that
    still opens the prompt is `mock`. What this now pins is the REPL's plumbing, not that a
    stackless install can compute."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "toy.npy").write_bytes(b"")
    np.save(tmp_path / "toy.npy", _data(40, 6))
    monkeypatch.setattr("builtins.input", lambda *a, **k: "quit")

    rc = app.main(["init", "--project", "demo", "--data", str(tmp_path / "toy.npy"),
                   "--engine", "mock"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "interactive mode is" not in out          # the deleted refusal
    assert "✓ initialized" in out                    # the prompt actually opened


# ── the lineage: index, artifacts, deltas, COMPLETE ──────────────────────────────────────


def test_running_one_step_twice_keeps_both_embeddings(tmp_path):
    """`index` addresses the artifact file, and a session's index is `len(self.steps)` — a
    monotonic counter over the lineage, not a per-action 0.

    This is the failure the scope measured when a session was prototyped as one `_run_steps`
    call per action: two `phate` steps both wrote `00-phate_emb.npy` and the second silently
    overwrote the first, with both records reporting `ok` and both paths still resolving."""
    from manyruns import artifacts

    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})
    first = session.step("phate")["record"]
    second = session.step({"name": "phate", "group": "latent", "params": {"n_components": 2}})
    second = second["record"]
    session.close()

    assert (first["index"], second["index"]) == (0, 1)
    assert first["artifacts"]["emb"] != second["artifacts"]["emb"]
    assert artifacts.load_array(first["artifacts"]["emb"]).shape[1] == 3    # not clobbered
    assert artifacts.load_array(second["artifacts"]["emb"]).shape[1] == 2


def test_a_geometry_delta_survives_across_actions(tmp_path):
    """`carry` is opened ONCE, at session open. The second embedding step's `geometry` must
    read `[from, to]`, not `[None, to]` — a session whose carry were re-created per action
    could only ever report an appearance, never a change, and nothing in the record would
    admit the difference."""
    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})
    first = session.step("phate")["record"]
    second = session.step({"name": "phate", "group": "latent",
                           "params": {"n_components": 2}})["record"]

    if not first.get("geometry"):
        pytest.skip("no live metric backend here — the delta has nothing to carry")
    metric = next(iter(first["geometry"]))
    assert first["geometry"][metric][0] is None            # an appearance
    assert second["geometry"][metric][0] is not None       # a change, against the first


def test_complete_appears_only_when_the_session_closes(tmp_path):
    """The marker's one guarantee is "this state folder is finished". A session that wrote it
    per step would report `complete()` True with the prompt still open — which is what calling
    `_run_steps` once per action does, measured."""
    from manyruns import artifacts

    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})
    session.step("phate")
    folder = artifacts.root(tmp_path, session.run_id)

    assert artifacts.complete(folder) is False     # mid-session
    session.close()
    assert artifacts.complete(folder) is True


def test_a_discarded_step_makes_a_run_incomplete_without_making_it_failed(tmp_path):
    """`ok` and `complete` are two questions, and a run that threw a step away answers them
    differently: nothing errored (`ok`), and the declared recipe did not get through
    (`complete`). See manyruns#66 for why those are two fields.

    THE STEP'S OWN RECORD STILL SAYS `ok`, AND SHOULD. `tune.run_tune_loop`'s cancel path RUNS
    the step and then `_reset()`s `state` and `g` back to the baseline, so the attempt really
    happened — it just contributed nothing. That is why `complete` cannot be read off the step
    records here and why `session.discarded` exists: every field derived from the records
    (`status`, `trace`, `ok`) reads as though the step had contributed.
    """
    session = _real(tmp_path, recipe=EMBED)
    session.run_recipe()
    kept = session.results()
    assert kept["complete"] is True and kept["ok"] is True     # the control: nothing discarded

    session.discarded.append("phate")
    out = session.results()

    assert out["ok"] is True, "an abandoned run is not a failed one"
    assert out["complete"] is False, "the declared recipe did not get through"
    assert [s["name"] for s in out["steps"] if s["outcome"] == "ok"] == ["phate"], (
        "the attempt's own record must still say it ran — it did")
    assert any("phate: discarded" in c for c in out["caveats"]), out["caveats"]


def test_results_mid_session_does_not_close_the_lineage(tmp_path):
    """The REPL's `summary` command calls `results()` while more steps may still be coming.
    It must assemble the record without ending the lineage."""
    from manyruns import artifacts

    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})
    session.step("phate")
    mid = session.results()

    assert mid["g_vector"]["final_dim"] == 3
    assert artifacts.complete(artifacts.root(tmp_path, session.run_id)) is False


def test_the_suite_is_re_measured_each_time_not_pinned_by_the_first_call(tmp_path, monkeypatch):
    """`suite.measure(already=g)` lets a value already in the g-vector win — deliberately, so
    the manylatents engine's own numbers are not recomputed. That makes WHICH dict `_finalize`
    merges into load-bearing for a session, because unlike a run a session is finalized more
    than once: `results()` is the REPL's `summary` command, and `close()` calls it again.

    So `_finalize` is handed a COPY. Verified by mutation — passing `self.g` instead of
    `dict(self.g)` makes the last assertion below read `lid == 1.0`: the FIRST embedding's
    geometry, reported as the current record after a second step has replaced it.

    Note the shape of what this rules out. `final_dim` would be 2 either way, because
    `_finalize` assigns the size keys unconditionally — only the SUITE goes stale, and it goes
    stale silently and in the direction of "nothing changed"."""
    from manyruns.pipeline import suite

    # stand in for manylatents' registry, so this runs with no private stack: lid == emb[0, 0]
    monkeypatch.setattr(suite, "_compute_metric", lambda: (
        lambda name, embeddings=None, dataset=None: float(embeddings[0, 0])))

    session = Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                      recipe={"name": "adhoc", "steps": []}, array=_data(), seed=0)
    session.dispatch["latent"] = lambda name, params, state, g, ctx: state.update(
        emb=np.full((8, 2), float(params["v"])))

    session.step({"name": "a", "group": "latent", "params": {"v": 1}})
    assert session.results()["g_vector"]["lid"] == 1.0

    session.step({"name": "a", "group": "latent", "params": {"v": 2}})
    assert session.results()["g_vector"]["final_dim"] == 2   # true under the mutation too
    assert session.results()["g_vector"]["lid"] == 2.0       # this is the one that goes stale


def test_the_run_id_is_minted_at_open_because_step_zero_writes_with_it(tmp_path):
    """`_persist` reads `ctx["run_id"]` to address the artifact step 0 writes. Minted at close
    it would be None for every write, so a lineage's whole state folder would be misaddressed."""
    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})

    assert session.run_id and session.ctx["run_id"] == session.run_id
    rec = session.step("phate")["record"]
    assert session.run_id in rec["artifacts"]["emb"]
    assert session.close()["run_id"] == session.run_id


# ── the record's identity when a session deviates from its recipe ────────────────────────


def test_a_session_that_deviates_stops_claiming_the_recipe_s_name():
    """`recipe` and `spec_id` must describe the same thing. A session that issued something
    other than what `cflows` declares is adaptive, and `None` is what both readers already
    render as "(adaptive)"."""
    session = _mock()
    assert session.results()["recipe"] == "cflows"        # nothing issued yet: still a prefix

    # The prefix starts at the PREAMBLE now, not at `phate`: since the cutover `cflows`
    # declares `normalize -> transform -> pca(50)` ahead of it (five steps, not two), so a
    # session that opens on `phate` deviates at position 0. Issued as DICTS, the way
    # `run_recipe` issues them — typing `pca` would resolve the catalog's `n_components: 10`
    # (archetypes' `pca`, first alphabetically) and deviate on params alone.
    for declared in session.recipe["steps"][:3]:
        session.step(dict(declared))
    session.step("phate")                                 # cflows' FOURTH step, typed by name
    assert session.results()["recipe"] == "cflows"

    session.step("separation")                            # cflows' fifth step is mioflow
    assert session.results()["recipe"] is None


def test_run_recipe_no_longer_refuses_to_re_run_a_step():
    """The old loop skipped any step already in `status`, so `run` after a manual `phate`
    silently dropped it. Re-running a step with different params is the whole of "mix and
    match"; the lineage records it twice, at two indices."""
    session = _mock()
    session.step("phate")
    again = session.run_recipe()

    # Five names and six indices, not two and three: `cflows` declares
    # `normalize -> transform -> pca(50)` ahead of `phate -> mioflow` since the cutover — the
    # preamble `loading._anndata_matrix` used to run unannounced on every counts-like load.
    # What this test is about is untouched by that: `phate` still appears TWICE, at two
    # indices, because `run_recipe` de-duplicates against nothing.
    assert [r["name"] for r in again] == ["normalize", "transform", "pca", "phate", "mioflow"]
    assert [s["index"] for s in session.steps] == [0, 1, 2, 3, 4, 5]
    assert session.trace == ["latent:phate", "prep:normalize", "prep:transform", "latent:pca",
                             "latent:phate", "lightning:mioflow"]


# ── views over the record, never a second accumulator ────────────────────────────────────


def test_trace_and_status_are_derived_from_the_step_record():
    """They were accumulated alongside the record, which is three sources of truth for one
    fact. `trace` appended BEFORE the step ran, so it reported an attempt as an execution."""
    session = _mock()
    session.step("phate")
    session.step("nope")                       # unknown: never reaches the lineage

    assert session.trace == ["latent:phate"]
    assert session.status == {"phate": "ok"}
    assert len(session.steps) == 1


def test_an_unknown_action_is_reported_and_the_session_survives():
    session = _mock()
    # NOT `umap` — it is a real action now, in the `cluster` recipe. A test whose "unknown"
    # input becomes known as the catalogue grows stops testing anything.
    r = session.step("definitely-not-a-step")

    assert not r["ok"] and "unknown action" in r["error"]
    assert "phate" in r["error"]               # it says what IS available
    assert session.step("phate")["ok"]         # still usable


def test_a_step_that_raises_is_a_record_not_an_exception(tmp_path, monkeypatch):
    """A session has no outer `try` for an exception to land in: one raised out of `step`
    ends the session and takes the state lineage with it."""
    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})

    def boom(name, params, state, g, ctx):
        raise RuntimeError("kaboom")

    session.dispatch["latent"] = boom
    r = session.step("phate")

    assert r["ok"] is False and "kaboom" in r["error"]
    assert session.steps[0]["outcome"] == "error"
    assert session.results()["ok"] is False    # and the row is not read as data


def test_results_reports_ok_so_a_failed_session_is_not_recorded_as_data(tmp_path):
    """`results()` had no `ok` key while `experiment.run_one` defaults it to True, so a session
    whose every step errored produced a row the sweep table read as successful. Same definition
    as the recipe path's, derived from the same records."""
    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []})
    assert session.results()["ok"] is True     # nothing has failed yet

    session.dispatch["latent"] = lambda *a: (_ for _ in ()).throw(ValueError("x"))
    session.step("phate")
    assert session.results()["ok"] is False


def test_a_live_observer_sees_each_step_as_it_runs(tmp_path):
    """`on_step(rec, steps)` is the seam the shell's live panel reads. It reaches the session
    because `apply` passes `steps=self.steps` — the observer gets the lineage list, growing,
    with the running record already in it."""
    seen: list = []
    session = _real(tmp_path, recipe={"name": "adhoc", "steps": []},
                    on_step=lambda rec, steps: seen.append((rec["name"], rec.get("state"),
                                                            len(steps))))
    session.step("phate")
    session.step("phate")

    running = [s for s in seen if s[1] == "running"]
    assert [s[2] for s in running] == [1, 2]   # 1,1 would mean the record joined late


# ── the step vocabulary is derived, not listed ───────────────────────────────────────────


def test_available_actions_comes_from_the_catalog_not_a_literal(tmp_path, monkeypatch):
    """It was `return ["phate", "mioflow", "separation", "composition"]` — one of three
    hand-kept copies. A recipe added to `$MANYRUNS_RECIPE_DIR` must show up in the menu with
    no source edit, because a catalogue of ten recipes will not survive a hardcoded four."""
    # Derived, so it GROWS with the catalogue — pinning the old literal four here would
    # contradict this test's own docstring the moment a recipe was added, which is exactly
    # what happened. What must hold is that every step of every bundled recipe is offerable,
    # EXCEPT the two groups the session declares unofferable: since the cutover seven recipes
    # declare a `normalize -> transform` prep block, and `session.UNOFFERED_GROUPS =
    # ("prep", "probe")` keeps those out of the menu on purpose (data conditioning is not a
    # question a person asks; a probe before anything is fitted can only decline). The
    # exclusion is asserted from BOTH sides here, so a step that quietly stopped being
    # dispatchable cannot hide inside it: unoffered means "absent from the menu and still
    # typeable", which is exactly what `UNOFFERED_GROUPS`' docstring promises.
    actions = set(available_actions())
    for name in catalog.discover_recipes():
        for step in catalog.load_recipe(name).get("steps") or []:
            if step["group"] in UNOFFERED_GROUPS:
                assert step["name"] not in actions, (
                    f"{name} declares {step['name']} in an unoffered group, yet it is a move")
                assert resolve_step(step["name"]) is not None, (
                    f"{name} declares {step['name']}, unoffered AND untypeable")
                continue
            assert step["name"] in actions, f"{name} declares {step['name']}, not offerable"

    (tmp_path / "solo.yaml").write_text(
        "name: solo\nsteps:\n  - {name: umap, group: latent, params: {n_components: 5}}\n"
    )
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(tmp_path))

    assert available_actions() == ["umap"]
    assert catalog_steps()["umap"]["params"] == {"n_components": 5}


def test_a_step_no_executor_can_run_is_not_offered(tmp_path, monkeypatch):
    """Derived from what can be DISPATCHED, not merely from what is declared. The in-process loop
    resolves a latent step by name out of `_INPROC_STEPS`, so a recipe naming an algorithm it
    does not have must not be offered — the session would only decline it."""
    (tmp_path / "solo.yaml").write_text(
        "name: solo\nsteps:\n  - {name: phate, group: latent, params: {}}\n"
        "  - {name: umap, group: latent, params: {}}\n"
    )
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(tmp_path))

    assert available_actions("_inproc") == ["phate"]     # umap has no in-process implementation
    assert available_actions() == ["phate", "umap"]   # but it does exist in the catalog
    assert resolve_step("umap", engine="_inproc") is None
    assert resolve_step("umap")["group"] == "latent"


def test_a_broken_recipe_file_drops_out_of_the_menu_instead_of_taking_it_down(tmp_path,
                                                                             monkeypatch):
    """`load_recipe` is strict and must stay strict — a recipe being RUN has to refuse. The
    MENU is a view, and a view never fails the thing it is a view of."""
    (tmp_path / "good.yaml").write_text(
        "name: good\nsteps:\n  - {name: phate, group: latent, params: {}}\n")
    (tmp_path / "bad.yaml").write_text("name: bad\nsteps: []\n")
    monkeypatch.setenv("MANYRUNS_RECIPE_DIR", str(tmp_path))

    assert available_actions() == ["phate"]
    with pytest.raises(ValueError):
        catalog.load_recipe("bad")


def test_aliases_still_resolve_and_a_deleted_step_is_not_kept_as_one():
    """An alias resolving to nothing is a worse answer than "unknown action", which lists what
    IS available.

    `granger` USED to be in the second category: removed with its step and asserted to resolve to
    nothing. It now resolves to a DECLARED
    STUB (`pipeline/stubs.py`) — which serves that same intent better rather than reversing it.
    A user who types it gets the measured refusal reason ("needs a gene trajectory whose time axis
    was MEASURED … and a data-level null that does not exist yet") instead of "unknown action",
    and the step still computes nothing: `test_stubs.py::test_a_stub_can_never_report_ok` is what
    keeps that true. The alias `Granger Causality` still resolves to nothing, because no alias
    exists — only the step does.
    """
    from manyruns.pipeline import stubs

    assert resolve_step("MIIOFlow")["name"] == "mioflow"
    assert resolve_step("MII OFlow")["name"] == "mioflow"      # spacing
    assert resolve_step("Granger Causality") is None
    assert resolve_step("nope") is None

    granger = resolve_step("granger")
    assert granger is not None and granger["name"] == "granger"
    assert "granger" in stubs.names(), "it resolves, so it must be a stub and not an implementation"


# ── the mock engine goes through the same loop ───────────────────────────────────────────


def test_the_mock_engine_is_a_dispatch_table_not_a_second_step_method():
    """`_step_mock` was 10 lines of loop inside the session. It is now two executors in
    `runner`'s dispatch table delegating to `serving`'s primitives, so a mock session produces
    a real step record — outcomes, params, indices — instead of a `status` string.

    FIVE records now, not two: `cflows` declares `normalize -> transform -> pca(50)` ahead of
    `phate -> mioflow` since the cutover. The prep pair DECLINES on this engine (the mock loads
    a named dataset itself, so manyruns never sees a matrix to prep) and a decline is recorded
    as a step WITH A REASON — which is the record shape this test is about, so it is asserted
    rather than skipped over.

    The `ok is True` line is RE-FRAMED, not loosened (`test_bounds.py` re-framed its own the
    same way): `ok` is `all(status == "ok")`, so a declining preamble makes it False on every
    engine that never gets a count matrix. The property this test protects is that the mock
    reaches the end of the recipe without ERRORING, so that is what it asserts — together with
    WHICH steps stood down and WHY, so a third one going quiet is still loud. Measured detail
    on this session: "no data in this state to prep — the engine loads a named dataset itself,
    so manyruns never sees the matrix"."""
    session = _mock()
    session.run_recipe()
    res = session.close()

    assert [s["outcome"] for s in res["steps"]] == ["skipped", "skipped", "ok", "ok", "ok"]
    # A decline without a reason is silence, which is the failure the cutover exists to end:
    # the preamble used to run unannounced, so it must now decline ANNOUNCED. Asserted on the
    # substring, as at `test_a_prep_step_that_never_RAN…`, so the wording can still improve.
    assert all("no data in this state" in (s["detail"] or "") for s in res["steps"][:2])
    assert res["trace"] == ["prep:normalize", "prep:transform", "latent:pca", "latent:phate",
                            "lightning:mioflow"]
    assert res["steps"][3]["params"] == {"n_components": 3}   # phate is step 3 now, not 0
    assert [(s["name"], s["detail"]) for s in res["steps"] if s["outcome"] == "error"] == []


def test_the_mock_writes_no_state_folder_because_it_has_no_arrays(tmp_path):
    """The mock transforms a 1-D stand-in vector, not an embedding. `_persist` writes only
    slots that have a `.shape`, so this is absence by construction rather than by exclusion —
    and an empty state folder must not be mistaken for an interrupted lineage."""
    from manyruns import artifacts

    session = _mock(out_dir=tmp_path)
    session.run_recipe()
    session.close()

    assert artifacts.complete(artifacts.root(tmp_path, session.run_id)) is False
    assert not (tmp_path / "state").exists()


def test_the_repl_loop_drives_the_session(tmp_path, monkeypatch):
    """The REPL end to end, on the dep-free engine: steps, then a summary on disk."""
    monkeypatch.chdir(tmp_path)
    inputs = iter(["phate", "accept", "MIIOFlow", "accept", "separation", "accept",
                   "summary", "quit"])
    out: list = []
    session = _mock(out_dir=tmp_path)
    session.source = "data"

    rc = app.interactive_session(session, read=lambda _="": next(inputs), write=out.append)

    joined = "\n".join(out)
    assert rc == 0
    assert "phate" in joined and "mioflow" in joined and "separation" in joined
    assert (tmp_path / "outputs" / "p" / "summary.md").is_file()
    # the typed aliases landed as real records, in order, at their own indices
    assert [s["name"] for s in session.steps] == ["phate", "mioflow", "separation"]
    assert [s["index"] for s in session.steps] == [0, 1, 2]


def test_a_prep_step_that_never_RAN_does_not_rewrite_the_runs_provenance(tmp_path):
    """The caveats are an account of THIS RUN, so they are folded over what ran.

    `results()` handed `vocab.noncanonical` the sequence AS ISSUED — `performed()`'s own
    docstring says it includes "steps that skipped or errored" — and `noncanonical` clears the
    run's step-produced facts on every DECLARED narrowing. Measured on this session before the
    fix: `filter_cells` has no executor, so `apply_step` records `outcome=skipped`,
    `detail="no prep step named 'filter_cells'"` and the run continues; `phate` had already
    assigned a fresh embedding that `mioflow` then read, and the record nonetheless carried
    "mioflow: embedding came from the caller, not from a latent step (e.g. phate)". A step that
    removed nothing changed the record's account of where the embedding came from, in the
    channel a human reads to decide whether to trust the number.

    TASKS 6/7 LANDED, AND THE SKIP REASON MOVED WITH THEM — the line above anticipated it.
    `filter_cells` is a real dispatchable step now, so "no prep step named 'filter_cells'" is no
    longer reachable; on this MOCK session it skips for the reason underneath, which is that the
    mock engine loads a named dataset itself and manyruns never sees a matrix to prep.

    The test's subject is untouched by that. What it pins is that a prep step which never RAN
    must not rewrite the run's provenance, and "never ran" is what both reasons describe. It is
    asserted on the substring "no data in this state" rather than the whole sentence, so the
    wording can be improved without failing a test about provenance.
    """
    from manyruns import vocab

    recipe = {"name": "filtered", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        {"name": "filter_cells", "group": "prep", "params": {"min_genes": 200}},
        {"name": "mioflow", "group": "lightning", "params": {}}]}
    session = _mock(recipe=recipe, out_dir=tmp_path, shape="time-course",
                    embedding=np.zeros((60, 3)))
    session.run_recipe()
    res = session.results()

    assert [s["outcome"] for s in res["steps"]] == ["ok", "skipped", "ok"]
    assert "no data in this state" in res["steps"][1]["detail"]
    assert [c for c in res["caveats"] if "came from the caller" in c] == []

    # The fold itself is unchanged — only what it is folded over. Over the DECLARED sequence,
    # which is what a recipe means before it runs, the same three steps still say so.
    declared = vocab.noncanonical(session.performed(), "time-course", provided={"embedding"})
    assert any("embedding came from the caller" in c for c in declared)


def test_a_typed_step_carries_THIS_recipe_s_params_not_the_alphabet_s(tmp_path):
    """manyruns#65. A step NAME is global across the catalog; a step's PARAMS belong to the
    recipe that declared them. Nothing made those two disagree until the cutover gave four
    recipes `pca n_components: 50` while `archetypes`/`cluster`/`markers` keep their own `10`.

    `catalog.known_steps` is first-declaration-wins over `discover_recipes()`, which is
    ALPHABETICAL — so `archetypes` won and every typed `pca` resolved to 10, including one typed
    while walking `cflows`, whose own file says 50. Measured before the fix:

        typed pca  -> {'n_components': 10}
        cflows pca -> {'n_components': 50}

    …with nothing in the record saying which of the two ran, and the session then dropping its
    claim to the recipe's name because the step it issued was not the step the recipe declares.
    """
    cflows = catalog.load_recipe("cflows")
    declared = next(s for s in cflows["steps"] if s["name"] == "pca")["params"]
    assert declared != catalog_steps()["pca"]["params"], (
        "this test is only meaningful while two bundled recipes declare `pca` differently; "
        "if that stopped being true, find another colliding name or delete this test")

    assert resolve_step("pca", recipe=cflows)["params"] == declared
    # …and a name THIS recipe does not declare still resolves, or mix-and-match breaks
    assert resolve_step("leiden", recipe=cflows) is not None
    # …and with no recipe in hand the catalog answers, unchanged for `tune` and every old caller
    assert resolve_step("pca")["params"] == catalog_steps()["pca"]["params"]


def test_a_session_walking_its_recipe_by_hand_still_claims_the_recipe(tmp_path):
    """The user-visible half of #65, and the reason it is not merely cosmetic: a person who types
    every step of `cflows` in order HAS run `cflows`, so the record must say so. Before the fix
    the typed `pca` carried different params from the declared one, `performed()` no longer
    matched the recipe, and `results()["recipe"]` came back `None` — rendered `(adaptive)` — for
    someone who followed the file exactly."""
    cflows = catalog.load_recipe("cflows")
    session = _mock(recipe=cflows, out_dir=tmp_path)

    for declared in cflows["steps"]:
        session.step(declared["name"])          # typed BY NAME, as a person would

    assert [s["name"] for s in session.steps] == [s["name"] for s in cflows["steps"]]
    assert session.results()["recipe"] == "cflows"


def test_embedding_only_identity_survives_continuation_and_branch(tmp_path):
    from manyruns import artifacts

    embedding = np.arange(8, dtype=float).reshape(4, 2)
    ids = ["cell-c", "cell-a", "cell-d", "cell-b"]
    labels = np.array(["A", "A", "B", "B"])
    session = Session("derived", engine="_inproc", out_dir=tmp_path,
                      embedding=embedding, labels=labels, sample_ids=ids, metrics=[])
    np.testing.assert_array_equal(session.state["rows"], np.arange(4))
    assert len(session.state["cols"]) == 0  # embedding dimensions are not genes

    # Deterministic stand-in checks identity plumbing; this is no compute oracle.
    def translate(name, params, state, g, ctx):
        state["emb"] = state["emb"] + 10

    session.dispatch["latent"] = translate
    assert session.apply({"name": "translate", "group": "latent"})["outcome"] == "ok"
    session.close()
    saved = artifacts.load_labeled(tmp_path, session.run_id, 0)
    assert saved["sample_ids"] == ids
    assert saved["labels"] == labels.tolist()
    np.testing.assert_array_equal(saved["array"], embedding + 10)
    branch = session.branch(0)
    np.testing.assert_array_equal(branch.ctx["sample_ids"], ids)
    np.testing.assert_array_equal(branch.state["emb"], embedding + 10)


@pytest.mark.parametrize("ids, labels, message", [
    (["a", "b", "c"], ["A"] * 4, "match the input rows"),
    (["a", "b", "c", "c"], ["A"] * 4, "unique"),
    (["a", "b", "c", "d"], ["A"] * 3, "labels must match"),
])
def test_embedding_only_identity_validation(tmp_path, ids, labels, message):
    with pytest.raises(ValueError, match=message):
        Session("derived", out_dir=tmp_path, embedding=np.zeros((4, 2)),
                sample_ids=ids, labels=labels)
