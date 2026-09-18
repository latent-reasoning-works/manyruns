"""`watch(fn)` — the wrapper form of the step record.

The load-bearing claim is not that wrapping works; it is that wrapping produces the SAME
record the recipe path produces, so there is one shape and one set of readers. A second
record format for "tools" would be the drift `vocab.py` exists to prevent, one layer up.
"""
from __future__ import annotations

import pytest

from manyruns import narrate
from manyruns.watch import Recorder, bound_params, describe, watch


def test_a_watched_call_produces_the_same_record_the_step_loop_does():
    """THE claim. The panel reads a watched record with no idea it wasn't a recipe step —
    same keys, same outcome vocabulary, same duration field."""
    rec = Recorder()
    watch(lambda x, k=3: x * k, recorder=rec, name="scale")(2)

    entry = rec.records[0]
    assert set(entry) >= {"index", "name", "group", "params", "outcome", "detail", "seconds"}
    assert (entry["name"], entry["outcome"]) == ("scale", "ok")

    panel = narrate.run_panel(rec.as_result())
    assert "scale" in panel and "1 of 1 steps produced a result" in panel


def test_an_exception_is_recorded_and_re_raised_unchanged():
    """A recorder that swallowed the error would be worse than no recorder."""
    rec = Recorder()

    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        watch(boom, recorder=rec)()

    assert rec.records[0]["outcome"] == "error"
    assert rec.records[0]["detail"] == "ValueError: kaboom"
    assert rec.as_result()["ok"] is False


def test_parameters_are_derived_from_the_signature_not_declared():
    """`inspect.signature` already knows what a tool takes, including defaults it was not
    called with. A hand-written schema per tool is a second copy that rots on the seventh."""
    def phate_like(data, n_components=2, knn=5, decay=40):
        return data

    rec = Recorder()
    watch(phate_like, recorder=rec)([1, 2, 3], knn=25)
    params = rec.records[0]["params"]

    assert params["knn"] == 25             # what it was called with
    assert params["n_components"] == 2     # and what it defaulted to
    assert params["decay"] == 40


def test_a_matrix_is_recorded_by_shape_not_by_value():
    """A record that inlined a 2700x32738 array is unwritable; one that inlined nothing
    cannot tell you a tool ran on the wrong thing."""
    np = pytest.importorskip("numpy")

    described = describe(np.zeros((2700, 30), dtype=np.float32))
    assert "2700" in described and "30" in described and "float32" in described
    assert describe("x" * 200).endswith("…")
    assert describe([1, 2, 3]) == "list[3]"
    assert describe(0.5) == 0.5


def test_an_unreadable_signature_still_records():
    """C extensions and builtins have no introspectable signature. An unnamed record still
    beats no record."""
    params = bound_params(max, ([3, 1, 2],), {})
    assert params, "a builtin produced no parameter record at all"


def test_the_outcome_is_written_after_the_call_not_before():
    """Same ordering rule as the step loop. If the outcome were set on entry, a call that
    raised would still read 'ok' — the attempts-versus-executions defect, re-committed."""
    rec = Recorder()

    seen = {}

    def slow():
        seen["outcome_during_call"] = rec.records[0]["outcome"]
        return 1

    watch(slow, recorder=rec)()
    assert seen["outcome_during_call"] is None      # not yet decided while running
    assert rec.records[0]["outcome"] == "ok"


def test_records_are_per_recorder_never_global():
    """The sweep runs cells in a process pool; module-level shared state is either wrong or
    invisible depending on the start method."""
    a, b = Recorder(), Recorder()
    watch(lambda: 1, recorder=a, name="one")()
    assert len(a) == 1 and len(b) == 0


def test_watch_tool_records_the_constructor_params_not_the_method_signature():
    """The limitation this closes. `watch(instance.fit_transform)` records `(x, y)` — the
    method's signature — so a tool's actual knobs (`knn`, `decay`, `t`) never appear. They
    live on the constructor. Splitting the call is what makes "parse params for every tool"
    work with no schema per tool."""
    from manyruns.watch import Recorder, watch_tool

    class PhateLike:
        def __init__(self, n_components=2, knn=5, decay=40):
            self.knn = knn

        def fit_transform(self, x, y=None):
            return [[0.0]] * len(x)

    rec = Recorder()
    watch_tool(PhateLike, recorder=rec, name="phate")([1, 2, 3], knn=25)
    params = rec.records[0]["params"]

    assert params["knn"] == 25          # a real knob, from the constructor
    assert params["decay"] == 40        # including one it defaulted to
    assert "x" not in params            # not the method's positional data argument
    assert rec.records[0]["produced"] == "list[3]"


def test_a_parameter_the_tool_does_not_accept_is_recorded_as_ignored():
    """`api.run` pops exactly one loose kwarg and silently drops the rest, so a declared
    parameter can look like it took effect when nothing received it. Recording the drop is
    how that becomes visible."""
    from manyruns.watch import Recorder, watch_tool

    class Narrow:
        def __init__(self, knn=5):
            pass

        def fit_transform(self, x):
            return x

    rec = Recorder()
    watch_tool(Narrow, recorder=rec, name="narrow")([1], nonsense=7)
    assert rec.records[0].get("ignored") == ["nonsense"]


# ── ONE outcome vocabulary: two producers, two renderers ─────────────────────
def test_every_renderer_covers_the_whole_outcome_vocabulary():
    """The drift guard. `watch.OUTCOMES` is the declaration; `narrate._OUTCOME_MARKS` (the
    plain panel) and `shell._OUTCOME_STYLE` (the rich panel) are presentation tables keyed
    by it, and both were hand-written against neither declaration.

    The same four names were stated FOUR times — those two tables plus `watch.OUTCOMES` and
    `runner.STEP_OUTCOMES`, the last two with no readers at all — so nothing failed while
    they happened to agree. Add an outcome and it would reach one renderer and show up as
    `?` on the other, in two renderings of ONE record. Fails in both directions: a missing
    key and an extra key are each drift."""
    from manyruns import shell
    from manyruns.watch import OUTCOMES

    declared = set(OUTCOMES)
    assert len(OUTCOMES) == len(declared), "the declaration repeats an outcome"
    assert set(narrate._OUTCOME_MARKS) == declared, "the plain panel and the declaration differ"
    assert set(shell._OUTCOME_STYLE) == declared, "the rich panel and the declaration differ"

    # and the consequence, not just the shape: every declared outcome renders as a word,
    # never as the `?` both tables fall back to for something they have not heard of.
    for outcome in OUTCOMES:
        glyph, word = narrate._OUTCOME_MARKS[outcome]
        assert glyph != "?" and word != "?"
        assert shell._OUTCOME_STYLE[outcome][0] != "?"
        panel = narrate.run_panel({"recipe": "r", "engine": "_inproc", "seed": 1, "steps": [
            {"index": 0, "name": "s", "group": "latent", "params": {},
             "outcome": outcome, "detail": None, "seconds": 0.0}]})
        assert f"{glyph} s" in panel and word in panel


def test_every_outcome_is_classified_as_having_run_or_not_run():
    """`runner.DID_NOT_RUN` splits the outcome vocabulary in two, and the split decides what a
    run's caveats are folded over (`runner.executed`). A fifth outcome would default to the
    "it ran" side silently, which is the direction that puts a false provenance note in the one
    channel a human reads to decide whether to trust a number.

    Both halves asserted against the declaration, so an outcome missing from either — or a name
    in `DID_NOT_RUN` that is not an outcome at all — fails here rather than at a reader."""
    from manyruns.pipeline.runner import DID_NOT_RUN
    from manyruns.watch import OUTCOMES

    ran = set(OUTCOMES) - set(DID_NOT_RUN)
    assert set(DID_NOT_RUN) <= set(OUTCOMES), "DID_NOT_RUN names something that is not an outcome"
    assert ran == {"ok", "reported"}, f"a new outcome is unclassified: {sorted(ran)}"


def test_the_outcome_vocabulary_has_exactly_one_declaration():
    """`runner.STEP_OUTCOMES` was a byte-identical second copy. An alias would be harmless;
    a restatement is what drifts, so this pins identity rather than absence."""
    from manyruns.pipeline import runner
    from manyruns.watch import OUTCOMES

    assert getattr(runner, "STEP_OUTCOMES", OUTCOMES) is OUTCOMES


def test_the_step_loop_only_ever_writes_a_declared_outcome():
    """The other producer. `watch` writes `ok`/`error` from `recorder_entry`; the step loop
    writes all four as literals, and this is what pins those literals to the declaration —
    rename one in the loop and it stops being renderable by either panel.

    Driven through `_run_steps` directly rather than an engine, because `reported` (an
    executor returning its own status string) is unreachable via `_INPROC_STEPS`: `_real_step`
    discards its implementation's return value, so a stub installed there can only ever be
    `ok`, `skipped` or `error`. No numpy: the stubs never touch an array."""
    from manyruns.pipeline import runner
    from manyruns.watch import OUTCOMES

    def fine(name, params, state, g, ctx):
        return None

    def declined(name, params, state, g, ctx):
        raise runner._StepSkipped("nothing to work on")

    def boom(name, params, state, g, ctx):
        raise ValueError("kaboom")

    def submitted(name, params, state, g, ctx):
        return "submitted job 7"

    recipe = {"name": "r", "steps": [
        {"name": "fine", "group": "latent"},
        {"name": "declined", "group": "latent"},
        {"name": "boom", "group": "latent"},
        {"name": "submitted", "group": "lightning"},
        {"name": "orphan"},                              # declares no group
        {"name": "unserved", "group": "analysis"},       # no executor on this engine
    ]}
    latent = {"fine": fine, "declined": declined, "boom": boom}
    dispatch = {"latent": lambda name, *rest: latent[name](name, *rest),
                "lightning": submitted}
    steps = runner._run_steps(recipe, runner._new_state(), {}, dispatch=dispatch, ctx={})

    seen = [s["outcome"] for s in steps]
    assert set(seen) <= set(OUTCOMES), f"undeclared outcome: {set(seen) - set(OUTCOMES)}"
    assert set(seen) == set(OUTCOMES), "an outcome in the vocabulary is unreachable here"
    assert seen == ["ok", "skipped", "error", "reported", "skipped", "skipped"]
