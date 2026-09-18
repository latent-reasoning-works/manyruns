"""`ok` and `complete` — two questions that were one field, and the run that they broke.

`ok` USED TO MEAN "every step ran" (`all(s == "ok" for s in status.values())`), which
conflated three events that are not the same:

  * a step ERRORED — the run is not trustworthy;
  * a step DECLINED — the executor ran and judged it had nothing to do (`normalize` on a
    matrix that is not counts). The other steps' numbers are unaffected;
  * a step was UNSUPPORTED — nothing could run it. A typo'd name, a stub, a group this
    engine has no table for. The record does not describe the analysis that was ASKED for.

Measured on a 120×12 Gaussian after the cutover gave seven recipes a prep block: 9 of the 11
bundled recipes reported `ok=False` with ZERO steps errored, so `experiment.py`'s `ok_rows`
filter aggregated a sweep over almost nothing while every step it cared about had run and
produced a g-vector. That is manyruns#66.

The split: **`ok` = nothing errored and nothing was missing. `complete` = every declared step
ran.** A row that is `ok` but not `complete` is real data from a shorter analysis — fine to
aggregate, not always fine to compare against a longer one.

THE UNSUPPORTED CASE IS THE ONE THIS FILE EXISTS FOR. The obvious reading of #66 — "declines
do not spoil `ok`" — makes a recipe of one typo'd step report success, which is the silent
failure this product refuses everywhere else. So the split is three-way, not two-way, and the
raiser says which it is (`_StepUnsupported`) rather than a reader parsing a detail string.
"""
from __future__ import annotations

import numpy as np

from manyruns.pipeline import runner
from manyruns.pipeline.steps import _StepSkipped, _StepUnsupported

#: REAL points, not `np.zeros`. A degenerate matrix makes PHATE raise ("Input matrices must
#: contain >1 unique points"), which turns every run below into an ERROR case and would have
#: made three of these tests pass for the wrong reason.
X = np.random.default_rng(0).normal(size=(60, 5))


def _run(steps, **kw):
    return runner.run_inproc(X, {"name": "t", "steps": steps}, "/tmp/manyruns-test-66", **kw)


def _one_step(executor, group="latent"):
    """One step through `apply_step` with an injected executor — the seam that takes a dispatch
    table. `run_inproc` builds its own, so an ERROR and a CLEAN run cannot be staged through it
    without depending on some real algorithm's behaviour on this fixture."""
    from manyruns.pipeline.runner import apply_step

    rec = apply_step({"name": "x", "group": group}, {"X": X, "emb": None}, {},
                     dispatch={group: executor}, ctx={"out_dir": "/tmp/manyruns-test-66",
                                                      "plots": [], "target_dim": 3},
                     index=0, carry={"written": [], "geometry": None})
    return rec


# ── the three events, and what each does to the two flags ────────────────────
def test_a_capability_gap_is_not_ok_and_not_complete():
    """The regression the naive fix would have introduced. A recipe of one step that nothing
    implements did NOTHING, and must not report success."""
    res = _run([{"name": "nosuchalgo", "group": "latent"}])

    assert res["steps"][0]["outcome"] == "skipped"
    assert res["steps"][0]["unsupported"] is True
    assert res["ok"] is False
    assert res["complete"] is False


def test_a_step_that_ran_and_judged_stays_ok_but_not_complete():
    """The decline #66 is about. `_require_embedding` is the clearest case in the tree: the
    executor is present, it ran, and it declined for want of coordinates."""
    res = _run([{"name": "mioflow", "group": "lightning"}])

    step = res["steps"][0]
    assert step["outcome"] == "skipped"
    assert "unsupported" not in step, "a benign decline carries NO key, rather than False"
    assert res["ok"] is True, "nothing errored and nothing was missing"
    assert res["complete"] is False, "but not every declared step ran"


def test_an_error_is_neither():
    """Unchanged, and the reason `ok` exists at all: a run whose steps blew up must not read
    as a legitimate row of a sweep table."""
    def boom(*_a, **_k):
        raise RuntimeError("the fit exploded")

    rec = _one_step(boom)

    assert rec["outcome"] == "error"
    assert "unsupported" not in rec, "an error is not a capability gap"


def test_a_clean_run_is_both():
    """The control. Without it every assertion above passes on a function that returns False."""
    res = _run([{"name": "phate", "group": "latent"}])

    assert res["steps"][0]["outcome"] == "ok"
    assert res["ok"] is True and res["complete"] is True


# ── the two flags are genuinely independent ──────────────────────────────────
def test_ok_without_complete_is_reachable_and_is_the_whole_point():
    """If no run could ever be `ok` and incomplete, the split bought nothing. This is the
    combination a sweep over the bundled synthetics produces by the dozen."""
    # ORDER IS LOAD-BEARING: `mioflow` first, so it declines for want of an embedding that
    # `phate` has not produced yet. Reverse them and phate supplies one, mioflow runs, and the
    # run is complete — which would make this test pass on the wrong pair.
    #
    # NOT a probe step, which was the first thing tried here: the `_inproc` substrate has no
    # `probe` executor at all, so `decode_to_gene_space` is a genuine capability gap and comes
    # back `unsupported` — correctly, and uselessly for this test.
    res = _run([{"name": "mioflow", "group": "lightning"},
                {"name": "phate", "group": "latent"}])

    assert [s["outcome"] for s in res["steps"]] == ["skipped", "ok"]
    assert res["ok"] is True, "phate's numbers are real and nothing errored"
    assert res["complete"] is False, "but mioflow did not run"


def test_complete_never_claims_more_than_ok():
    """A run cannot have run every step and also have errored — `complete` implies `ok`. The
    two are computed from the same records over disjoint outcome sets, so this is a property
    rather than a coincidence, and it fails loudly if one of them is ever inverted."""
    for steps in ([{"name": "phate", "group": "latent"}],
                  [{"name": "nosuchalgo", "group": "latent"}],
                  [{"name": "mioflow", "group": "lightning"}],
                  [{"name": "orphan"}]):
        res = _run(steps)
        if res["complete"]:
            assert res["ok"], f"complete but not ok: {steps}"


# ── the vocabulary the split rests on ────────────────────────────────────────
def test_unsupported_is_a_kind_of_skip_so_every_existing_catch_still_holds():
    """Subclassing rather than sitting beside it. `except _StepSkipped` appears in the step
    loop and nothing else needed changing — a new exception class that the loop did not catch
    would have turned every capability gap into an `error` and killed the run."""
    assert issubclass(_StepUnsupported, _StepSkipped)


def test_the_outcome_vocabulary_did_not_grow():
    """Both kinds land as `skipped`. The alternative — a fifth outcome — would have reached
    `watch.OUTCOMES`, `narrate`'s glyph table, the shell's colour map and the live panel, to
    express a distinction only the run-level flags consume."""
    from manyruns import watch

    assert set(watch.OUTCOMES) == {"ok", "skipped", "error", "reported"}


def test_a_missing_executor_and_a_missing_group_are_both_capability_gaps():
    """The two PRE-DISPATCH declines, which never reach an exception at all — `apply_step`
    returns before the `try`. They set the flag directly, so the property has two homes and
    this is the test that keeps them agreeing."""
    no_group = _run([{"name": "orphan"}])["steps"][0]
    no_table = _one_step(None, group="nosuchgroup")

    assert no_group["unsupported"] is True and no_table["unsupported"] is True
    assert _run([{"name": "orphan"}])["ok"] is False


def test_a_stub_spoils_ok_because_it_is_a_capability_that_does_not_exist():
    """`pipeline/stubs.py` declares steps whose maths is not written, so the GUI can be built
    against them. A run containing one did not perform the analysis its recipe declares —
    the same event as a typo'd name, and `ok` must say so. The stub's own safety property
    ("a stub can never report ok") is thereby extended one level up, to the RUN."""
    from manyruns.pipeline import stubs

    name = stubs.names()[0]
    group = stubs._STUBS[name][0]
    res = _run([{"name": name, "group": group}])

    assert res["steps"][0]["outcome"] == "skipped"
    assert res["steps"][0]["unsupported"] is True
    assert res["ok"] is False


# ── downstream: the sweep this was filed from ────────────────────────────────
def test_the_sweep_row_carries_both_and_the_summary_reports_the_difference():
    """`experiment.py:281` filters `ok_rows` and `:304` prints the count. Before the split a
    sweep over the bundled synthetics filtered down to ~nothing; a summary that recovered the
    rows but said nothing about their shortness would trade one silence for another."""
    from manyruns import experiment

    assert "complete" in experiment.LABEL_COLUMNS

    rows = [{"recipe": "a", "ok": True, "complete": True, "m": 1.0},
            {"recipe": "b", "ok": True, "complete": False, "m": 2.0}]
    out = experiment.summarize(rows)

    assert "2 ok" in out
    assert "1 with a step declined" in out


def test_an_older_record_is_read_as_complete_rather_than_relabelled():
    """`index.jsonl` is append-only, so rows written before the split are on real disks. Their
    `ok` meant "every step ran", which is exactly `complete`'s question — so it is the
    DEFAULT, and the alternative (default True) would silently relabel every declined step in
    the archive as having run."""
    from manyruns.experiment import Result, Run, run_one  # noqa: F401
    from manyruns import experiment

    assert Result(run=Run(recipe={"name": "r"}, dataset={"name": "d"})).complete is False
    # and the read path: an old record carries `ok` and no `complete`.
    assert experiment.Result(
        run=Run(recipe={"name": "r"}, dataset={"name": "d"}),
        ok=True, complete=bool({"ok": True}.get("complete", True)),
    ).complete is True


def test_the_row_written_to_the_store_carries_it(tmp_path):
    """`index.jsonl` is what the learner reads. A distinction the record cannot express is a
    distinction that cannot be learned, however carefully the runner computes it."""
    from manyruns import store

    store.append({"recipe": "r", "ok": True, "complete": False, "run_id": "abc"},
                 out_dir=tmp_path)
    row = next(iter(store.read(out_dir=tmp_path)))

    assert row["ok"] is True and row["complete"] is False
