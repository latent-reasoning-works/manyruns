"""The accept/retry/cancel loop `manyruns>` runs a step through (`manyruns/tune.py`)."""
import pytest

from manyruns import app, tune
from manyruns.session import Session


@pytest.fixture(autouse=True)
def _corpus_stays_out_of_the_repo(monkeypatch, tmp_path):
    """Every test in this file runs from a temp directory.

    The loop writes a decision row (#45), and a session that never named an `out_dir` writes it
    under `outputs/` — relative to the CURRENT directory. Without this, running the suite from a
    checkout appends to `outputs/decisions.jsonl` in the repository, forever: 87 rows accumulated
    in the repo root before anyone read `git status`, because an untracked file that only grows is
    invisible until you look for it.

    Autouse rather than a per-test argument, so a test added later cannot forget and reintroduce
    it. `chdir` rather than threading `out_dir` everywhere, because what is under test is the
    loop's behaviour and not its file paths.
    """
    monkeypatch.chdir(tmp_path)


def _mock(recipe=None, **kw):
    kw.setdefault("engine", "mock")
    return Session(project="p", modality="scrna",
                   recipe=recipe if recipe is not None else app.load_recipe("cflows"), **kw)


# ── parsing ──────────────────────────────────────────────────────────────────────────────


def test_parse_overrides_coerces_types():
    assert tune.parse_overrides("knn=40 decay=0.5 verbose=true") == {
        "knn": 40, "decay": 0.5, "verbose": True}


def test_parse_overrides_blank_is_no_change():
    assert tune.parse_overrides("") == {}
    assert tune.parse_overrides("   ") == {}


def test_parse_step_line_bare_name_keeps_catalog_defaults():
    step = tune.parse_step_line("phate", "mock")
    assert step["name"] == "phate"
    assert step["params"] == {"n_components": 3}


def test_parse_step_line_overrides_layer_onto_defaults():
    step = tune.parse_step_line("phate knn=40", "mock")
    assert step["params"] == {"n_components": 3, "knn": 40}


def test_parse_step_line_unknown_first_token_is_none():
    assert tune.parse_step_line("bogus knn=40", "mock") is None


# ── the loop ─────────────────────────────────────────────────────────────────────────────


def test_accept_on_first_try_runs_once():
    session = _mock()
    inputs = iter(["accept"])
    out: list = []
    step = tune.parse_step_line("phate", session.engine)

    accepted = tune.run_tune_loop(session, step, lambda _="": next(inputs), out.append)

    assert accepted["params"] == {"n_components": 3}
    assert len(session.steps) == 1


@pytest.mark.parametrize("protocol,terminal,inline", [
    ("iterm", True, True), ("kitty", True, True),
    ("off", True, False), ("iterm", False, False),
])
def test_loop_default_owns_inline_drawing_and_exact_path_fallback(
    tmp_path, monkeypatch, protocol, terminal, inline
):
    import base64
    import io
    import sys
    from PIL import Image

    path = tmp_path / "phate.png"
    Image.new("RGB", (2, 2)).save(path)
    session = _mock()
    issued = session.step

    def with_plot(step):
        rec = issued(step)
        rec["record"]["plots"] = [str(path)]
        return rec

    monkeypatch.setattr(session, "step", with_plot)
    # _show_plots builds its two-attribute shim from this stream. No terminal or probe.
    stream = io.StringIO()
    monkeypatch.setattr(stream, "isatty", lambda: terminal)
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setenv("MANYRUNS_INLINE_IMAGES", protocol)
    out = []
    tune.run_tune_loop(session, tune.parse_step_line("phate", "mock"),
                       lambda _: "accept", out.append)

    payload = base64.b64encode(path.read_bytes()).decode()
    expected = (f"\x1b]1337;File=inline=1;width=60;preserveAspectRatio=1:{payload}\a\n"
                if protocol == "iterm" else f"\x1b_Gf=100,a=T,t=d,c=60,m=0;{payload}\x1b\\\n")
    assert stream.getvalue() == (expected if inline else "")
    assert [line for line in out if line.startswith("plot:")] == (
        [] if inline else [f"plot: {path}"])


def test_loop_callback_owns_fallback_even_when_it_returns_none(monkeypatch):
    session = _mock()
    rec = {"ok": True, "name": "phate", "record": {"plots": ["figure.png"]}}
    monkeypatch.setattr(session, "step", lambda step: rec)
    out, shown = [], []

    def show(record, write):
        shown.append(record)
        write("surface-owned caption")

    tune.run_tune_loop(session, tune.parse_step_line("phate", "mock"),
                       lambda _: "accept", out.append, show_plots=show)
    assert shown == [rec]
    assert out == ["surface-owned caption", "✓ phate: ok"]


def test_retry_then_accept_keeps_both_attempts_as_separate_lineage_entries():
    """A rejected attempt is not overwritten — `Session.step` appends, so tuning leaves an
    audit trail of every params combination actually tried."""
    session = _mock()
    inputs = iter(["retry", "knn=40", "accept"])
    out: list = []
    step = tune.parse_step_line("phate knn=5", session.engine)

    accepted = tune.run_tune_loop(session, step, lambda _="": next(inputs), out.append)

    assert accepted["params"]["knn"] == 40
    assert [s["params"].get("knn") for s in session.steps] == [5, 40]


def test_retry_reruns_the_original_input_not_the_rejected_attempts_output():
    """Regression: retrying `phate` with a different `knn` must re-run against what the step
    saw BEFORE this loop started — not the just-rejected attempt's own output. Compares a
    retried run against a DIRECT single run at the same final params: before the fix these
    diverged (the retried run was double-transformed, PHATE stacked on the rejected PHATE);
    after the fix they land on the same state."""
    direct = _mock()
    tune.run_tune_loop(direct, tune.parse_step_line("phate knn=40", direct.engine),
                        lambda _="": "accept", lambda *_: None)

    retried = _mock()
    inputs = iter(["retry", "knn=40", "accept"])
    tune.run_tune_loop(retried, tune.parse_step_line("phate knn=5", retried.engine),
                        lambda _="": next(inputs), lambda *_: None)

    assert retried.state["vec"] == direct.state["vec"]


def test_cancel_fully_restores_the_state_from_before_this_step():
    """`cancel` must undo the attempt entirely, matching its own message ("discarded, back to
    what it was before") — not just stop retrying while leaving the last (rejected)
    computation's output as the session's live state."""
    session = _mock()
    inputs = iter(["cancel"])

    tune.run_tune_loop(session, tune.parse_step_line("phate", session.engine),
                        lambda _="": next(inputs), lambda *_: None)

    assert "vec" not in session.state  # nothing had run before this step; nothing lingers now


def test_cancel_returns_none_and_does_not_crash():
    session = _mock()
    inputs = iter(["cancel"])
    out: list = []
    step = tune.parse_step_line("phate", session.engine)

    accepted = tune.run_tune_loop(session, step, lambda _="": next(inputs), out.append)

    assert accepted is None
    assert len(session.steps) == 1  # the cancelled attempt still ran and still landed


def test_metric_suite_is_suppressed_during_the_loop_and_restored_after():
    """The declared suite is expensive (pairwise metrics) and runs inline before the plot is
    saved — paying for it on every rejected attempt defeats the loop's own point. It must be
    off during tuning and back to normal once tuning is done, so a later step (or the
    session's own close/finalize) still gets it."""
    session = _mock(metrics=["trustworthiness"])
    seen_during: list = []
    real_step = session.step

    def spying_step(action):
        seen_during.append(list(session.ctx.get("metrics") or []))
        return real_step(action)

    session.step = spying_step
    inputs = iter(["accept"])
    step = tune.parse_step_line("phate", session.engine)

    tune.run_tune_loop(session, step, lambda _="": next(inputs), lambda *_: None)

    assert seen_during == [[]]
    assert session.ctx["metrics"] == ["trustworthiness"]


def test_unrecognised_answer_reprompts_instead_of_crashing():
    session = _mock()
    inputs = iter(["huh", "accept"])
    out: list = []
    step = tune.parse_step_line("phate", session.engine)

    accepted = tune.run_tune_loop(session, step, lambda _="": next(inputs), out.append)

    assert accepted is not None
    assert any("didn't understand" in line for line in out)


# ── through the REPL end to end ─────────────────────────────────────────────────────────


def test_repl_gates_every_step_on_accept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = iter(["phate knn=5", "retry", "knn=40", "accept", "mioflow", "accept", "quit", ""])
    out: list = []
    session = _mock(out_dir=tmp_path)
    session.source = "data"

    rc = app.interactive_session(session, read=lambda _="": next(inputs), write=out.append)

    assert rc == 0
    assert [s["name"] for s in session.steps] == ["phate", "phate", "mioflow"]
    assert [s["params"].get("knn") for s in session.steps if s["name"] == "phate"] == [5, 40]
    assert any("next: mioflow" in line for line in out)


def test_declining_the_save_prompt_with_a_word_writes_no_file(tmp_path, monkeypatch):
    """Regression: a bare `if dest:` took ANY non-blank answer as a filename, so answering
    `no` to "save as a recipe? (path or blank to skip)" wrote a recipe literally named `no`
    to the cwd — reproduced verbatim from a real session (`no`/`save` both landed in the repo
    root this way). Blank and a declining word must both mean "skip"."""
    monkeypatch.chdir(tmp_path)
    inputs = iter(["phate knn=5", "retry", "knn=40", "accept", "quit", "no"])
    out: list = []
    session = _mock(out_dir=tmp_path)
    session.source = "data"

    rc = app.interactive_session(session, read=lambda _="": next(inputs), write=out.append)

    assert rc == 0
    assert not (tmp_path / "no").exists()
    assert not any("✓ saved" in line for line in out)


def test_repl_unknown_step_does_not_enter_the_tune_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inputs = iter(["bogus", "quit"])
    out: list = []
    session = _mock(out_dir=tmp_path)
    session.source = "data"

    rc = app.interactive_session(session, read=lambda _="": next(inputs), write=out.append)

    assert rc == 0
    assert session.steps == []
    assert any("unknown action" in line for line in out)


# ── what a rejected attempt must not leave behind ────────────────────────────────────────


def test_a_rejected_attempt_keeps_its_own_picture(tmp_path):
    """manyruns#43. `_save_scatter` names a plot for the STEP (`runner.py:1119`,
    `f"{name}.png"`), so every attempt of `phate` wrote `plots/phate.png` and only the last one
    survived. Measured before the fix: three attempts, one file.

    IT MATTERS MORE IN THE GUI THAN IN THE REPL, which is why it gates the port. The REPL is a
    stream — the plot is printed, you look at it, the next is printed below. A screen holds a
    PATH and re-reads it, so a retry silently swaps the image under a row describing a different
    attempt, and you compare attempt 3's picture against attempt 1's numbers.

    THE ACCEPTED ATTEMPT KEEPS THE PLAIN NAME. Everything else — `figspec.stem`, the run pane,
    the summary — already expects `plots/<step>.png`, so suffixing the winner would make this
    loop the only producer of a filename nothing else knows.
    """
    import pytest

    pytest.importorskip("phate")
    pytest.importorskip("sklearn")
    import numpy as np

    session = Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                      recipe={"name": "r", "steps": []}, seed=0,
                      array=np.asarray(np.random.default_rng(0).normal(size=(60, 6))))
    inputs = iter(["retry", "knn=6", "retry", "knn=7", "accept"])
    out: list = []
    step = tune.parse_step_line("phate knn=5", session.engine)

    tune.run_tune_loop(session, step, lambda _="": next(inputs), out.append)

    pngs = sorted(p.name for p in tmp_path.rglob("phate*.png"))
    assert pngs == ["phate.png", "phate@1.png", "phate@2.png"], pngs
    assert any("kept attempt 1" in line for line in out), "the rename is reported, not silent"


def test_a_rejected_attempts_numbers_are_not_the_next_attempts_baseline(tmp_path):
    """manyruns#42's sibling, manyruns#44. `run_tune_loop` reset two dicts and there are
    three: `carry["geometry"]` is the run-scoped accumulator that turns each step's geometry
    into a `[from, to]` delta (`runner.py:235`), and `tune.py` never touched it.

    So a rejected attempt's numbers stayed in the carry and the accepted attempt subtracted
    against them — reporting movement from a starting point nobody kept, with an arrow, exactly
    like a real measurement.

    ASSERTED AGAINST A CLEAN RUN of the same accepted params rather than against a hardcoded
    number: the property is "tuning leaves no trace in the delta", and the only thing that can
    state it is the run that never tuned.
    """
    import pytest

    pytest.importorskip("phate")
    pytest.importorskip("sklearn")
    # AND THE ENGINE, for the DELTAS specifically. The per-step geometry comes from
    # `runner._live_suite` -> `catalog.load_suite`, which reads manylatents' metric registry; CI
    # installs the product layer only (`.github/workflows/ci.yml:24-27`), so a `_inproc` run there
    # fits PHATE happily and reports NO geometry. Measured: this test failed in CI and passed
    # locally, on its own "both runs must report geometry" guard — which is what that guard is
    # for, since without it the loop below iterates an empty dict and asserts nothing.
    pytest.importorskip("manylatents", reason="the deltas are the metric suite's, not phate's")
    import numpy as np

    def _session(where):
        return Session(project="p", engine="_inproc", out_dir=where, modality="scrna",
                       recipe={"name": "r", "steps": []}, seed=0,
                       array=np.asarray(np.random.default_rng(0).normal(size=(60, 6))))

    tuned = _session(tmp_path / "tuned")
    inputs = iter(["retry", "knn=9", "accept"])
    tune.run_tune_loop(tuned, tune.parse_step_line("phate knn=5", tuned.engine),
                       lambda _="": next(inputs), lambda *_: None)

    clean = _session(tmp_path / "clean")
    tune.run_tune_loop(clean, tune.parse_step_line("phate knn=9", clean.engine),
                       lambda _="": "accept", lambda *_: None)

    got = tuned.steps[-1].get("geometry") or {}
    want = clean.steps[-1].get("geometry") or {}
    assert got and want, "both runs must report geometry or this asserts nothing"
    assert sorted(got) == sorted(want)
    for key in want:
        assert got[key][0] == want[key][0], (
            f"{key}: the accepted attempt's delta started from {got[key][0]} where an untuned "
            f"run starts from {want[key][0]} — a rejected attempt's number leaked into the carry")


# ── what the loop is FOR: the decision it produces ───────────────────────────────────────


def test_a_tuning_loop_records_what_was_tried_and_what_was_kept(tmp_path):
    """manyruns#45. `decisions.append` has been the corpus writer since the ledger got one, and
    `tune.py` contained no reference to it — every retry and its replacement were lost the moment
    the loop returned. The stated value of this loop is exactly "what did NOT work, and what was
    chosen when it did".

    ONE ROW PER LOOP, not one per attempt: a decision IS a choice among alternatives, so three
    attempts ending in an accept are one row whose `offered` has three entries and whose `chosen`
    names the third. Recording them separately would lose which attempts competed with which.
    """
    from manyruns import decisions

    session = _mock(out_dir=tmp_path)
    inputs = iter(["retry", "knn=40", "retry", "knn=80", "accept"])
    step = tune.parse_step_line("phate knn=5", session.engine)

    tune.run_tune_loop(session, step, lambda _="": next(inputs), lambda *_: None)

    rows = [r for r in decisions.read(tmp_path) if r.get("surface") == "tune"]
    assert len(rows) == 1, rows
    row = rows[0]
    # The label carries EVERY param, not just the one being tuned — `parse_step_line` layers an
    # override onto the catalog's declared defaults, so `n_components=3` rides along. That is
    # correct rather than noise: two attempts differing only in a default are different attempts,
    # and a label that dropped the defaults would collapse them into one.
    assert [o["label"] for o in row["offered"]] == [
        "phate knn=5 n_components=3",
        "phate knn=40 n_components=3",
        "phate knn=80 n_components=3"]
    assert row["chosen"] == "phate knn=80 n_components=3"
    assert [o["params"]["knn"] for o in row["offered"]] == [5, 40, 80]


def test_a_cancelled_loop_is_recorded_with_no_choice(tmp_path):
    """"None of these worked" is a signal. Dropping it would bias the corpus toward loops that
    happened to end well — the same selection effect that makes a corpus of successes useless
    for learning when to refuse."""
    from manyruns import decisions

    session = _mock(out_dir=tmp_path)
    inputs = iter(["retry", "knn=40", "cancel"])
    step = tune.parse_step_line("phate knn=5", session.engine)

    assert tune.run_tune_loop(session, step, lambda _="": next(inputs), lambda *_: None) is None

    row = [r for r in decisions.read(tmp_path) if r.get("surface") == "tune"][0]
    assert row["chosen"] is None
    assert len(row["offered"]) == 2


def test_accepting_the_first_attempt_records_nothing(tmp_path):
    """A choice with no alternative has no label to learn against — the reason
    `decisions.append` already refuses an empty offer set. Accepting the first thing you tried is
    not a decision between things, and a corpus full of those teaches a selector to always take
    the default it was already going to take."""
    from manyruns import decisions

    session = _mock(out_dir=tmp_path)
    step = tune.parse_step_line("phate", session.engine)

    tune.run_tune_loop(session, step, lambda _="": "accept", lambda *_: None)

    assert [r for r in decisions.read(tmp_path) if r.get("surface") == "tune"] == []


def test_a_tune_row_is_not_mistaken_for_a_ledger_row(tmp_path):
    """`decisions.examples()` extracts LEDGER rows: it reads `o["recipe"]` and `o["can_run"]`.
    A tune row carries neither, so it is SKIPPED there rather than mis-parsed into a training
    pair claiming a step name was a recipe. Turning tune rows into pairs is separate work; this
    pins that recording them does not corrupt what already reads the file."""
    from manyruns import decisions

    session = _mock(out_dir=tmp_path)
    inputs = iter(["retry", "knn=40", "accept"])
    tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine),
                       lambda _="": next(inputs), lambda *_: None)

    row = [r for r in decisions.read(tmp_path) if r.get("surface") == "tune"][0]
    assert not any("recipe" in o or "can_run" in o for o in row["offered"])
    assert decisions.examples(tmp_path) == []


def test_the_loop_offers_tune_and_still_answers_to_retry():
    """The verb is `tune`, not `retry`, and the distinction is what the branch DOES: it always
    asks for new params, so no path re-runs a step unchanged. "Retry" named an operation that
    consumes nothing; "tune" names the one that consumes the params declaration.

    `retry`/`r` remain accepted and unadvertised — they are what the prompt said until the
    rename, so refusing them would punish muscle memory for a change the product made. Only the
    current vocabulary is offered back on a miss, which is the half that must not drift.
    """
    session = _mock()
    prompts: list = []

    def read(prompt=""):
        prompts.append(prompt)
        # a miss first, so the refusal line is exercised, then the OLD verb, then accept
        return {1: "wat", 2: "retry", 3: "knn=40", 4: "accept"}[len(prompts)]

    out: list = []
    tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine), read, out.append)

    assert any("accept / tune / cancel" in p for p in prompts), prompts
    assert not any("retry" in p for p in prompts), "the old verb is no longer advertised"
    assert any("didn't understand 'wat'" in line and "key=value" in line for line in out)
    assert len(session.steps) == 2, "`retry` still drove a second attempt"


def test_a_typo_does_not_re_run_the_step(tmp_path):
    """Found while renaming the verb, and it is the sharper half of that change.

    The refusal used to fall out of the bottom of the outer loop, so an unrecognised answer sent
    control back to `session.step` — a typo re-FITTED the step. Measured before the fix: one
    `wat` in a two-attempt session left THREE entries in the lineage. On a real dataset that is
    minutes for a keystroke.

    It got worse when the loop started writing a decision row (#45): the spurious attempt was
    `offered` to the corpus as a params combination a human had chosen to try, which is exactly
    the label the corpus exists to be trusted on.
    """
    from manyruns import decisions

    session = _mock(out_dir=tmp_path)
    inputs = iter(["wat", "?", "tune", "knn=40", "nope", "accept"])
    out: list = []

    tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine),
                       lambda _="": next(inputs), out.append)

    assert len(session.steps) == 2, "three typos cost three extra fits"
    row = [r for r in decisions.read(tmp_path) if r.get("surface") == "tune"][0]
    assert len(row["offered"]) == 2, "a typo was offered to the corpus as an attempt"
    assert sum("didn't understand" in line for line in out) == 3


def test_a_session_with_no_out_dir_does_not_write_a_corpus_into_the_cwd(tmp_path, monkeypatch):
    """`Session.out_dir` DEFAULTS TO `"."`, and passing that through put tune rows at
    `./decisions.jsonl` while `decisions.append`'s own default sends LEDGER rows to
    `outputs/decisions.jsonl` — two corpora, two places, one product.

    Found the ugly way: 87 rows had accumulated in the repository root from test runs before
    anyone read `git status`. An untracked file that only ever grows is invisible until you look
    for it, which is why this is pinned rather than left to the fixture above.
    """
    from manyruns import decisions

    monkeypatch.chdir(tmp_path)
    session = _mock()                       # deliberately no out_dir
    inputs = iter(["tune", "knn=40", "accept"])

    tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine),
                       lambda _="": next(inputs), lambda *_: None)

    assert not (tmp_path / "decisions.jsonl").exists(), "a corpus landed in the working directory"
    # …and it went where the ledger's rows go, so there is ONE corpus rather than two
    rows = [r for r in decisions.read(tmp_path / "outputs") if r.get("surface") == "tune"]
    assert len(rows) == 1


# ── item 7, tier A: a spoken hyperparameter ──────────────────────────────────────────────


def test_a_spoken_parameter_sets_an_absolute_value():
    """Backlog item 7's simplest case — "hyperparameter mentioned explicitly" — and it is
    RULE-BASED: no SDK, no key, no network, so it works in the base install and in CI.

    `intent.py` set this pattern for recipe selection ("three tiers, and only the top one touches
    an LLM"); this is the same Tier A for parameters."""
    current = {"knn": 5, "n_components": 3, "decay": 40}

    assert tune.parse_overrides("set knn to 40", current) == {"knn": 40}
    assert tune.parse_overrides("knn to 40", current) == {"knn": 40}
    assert tune.parse_overrides("increase the k to 5", current) == {"knn": 5}


def test_a_relative_phrase_counts_from_the_current_value():
    """"by one" has nothing to add to without the step's current params, which is why they are
    threaded in rather than the parser holding a table. Inventing a base would produce a number
    the user did not ask for."""
    current = {"knn": 5, "n_components": 3}

    assert tune.parse_overrides("increase the k by one", current) == {"knn": 6}
    assert tune.parse_overrides("lower the k by two", current) == {"knn": 3}
    assert tune.parse_overrides("decrease knn by one", current) == {"knn": 4}
    # …and with nothing to count from it declines rather than guessing
    assert tune.parse_overrides("increase the k by one") == {}


def test_a_spoken_name_resolves_against_THIS_step_not_a_global_table():
    """"the k" is `knn` on PHATE and `n_neighbors` on UMAP, so a fixed mapping would be wrong for
    one of them. The step's own params are the vocabulary."""
    assert tune.parse_overrides("increase the k to 9", {"knn": 5}) == {"knn": 9}
    assert tune.parse_overrides("increase the k to 9", {"n_neighbors": 15}) == {}
    assert tune.parse_overrides("neighbors to 30", {"n_neighbors": 15}) == {"n_neighbors": 30}
    # a token-boundary match, so `k` does not also claim `decay`
    assert tune.parse_overrides("k to 7", {"knn": 5, "decay": 40}) == {"knn": 7}


def test_an_int_parameter_stays_an_int():
    """`knn` is a neighbour COUNT. `_coerce` cannot know that from the text — the current value
    can, and a float where the engine wants a count is the kind of thing that surfaces as a
    library error three layers down."""
    out = tune.parse_overrides("increase the k by one", {"knn": 5})

    assert out == {"knn": 6}
    assert isinstance(out["knn"], int) and not isinstance(out["knn"], bool)
    # a float parameter keeps being a float
    assert tune.parse_overrides("increase decay by half", {"decay": 1.0}) == {"decay": 1.5}


def test_a_sentence_that_names_no_parameter_changes_nothing():
    """THE SAFETY PROPERTY, and the reason this is worth having as a rule tier at all: a wrong
    parameter silently tuned is the failure this product refuses everywhere else.

    "this plot looks too clustered" is a real request and Tier B's job — it names no parameter,
    so deciding which knob moves needs the geometry. Guessing here would be worse than declining,
    because the loop would report an attempt the user never asked for."""
    current = {"knn": 5, "n_components": 3}

    assert tune.parse_overrides("this plot looks too clustered", current) == {}
    assert tune.parse_overrides("increase the x by one", current) == {}
    assert tune.parse_overrides("make it better", current) == {}
    assert tune.parse_overrides("", current) == {}


def test_key_value_still_wins_and_is_unchanged():
    """The advertised form is tried FIRST and is untouched — it is unambiguous, it is what the
    prompt tells you to type, and a sentence containing an `=` should be read as typed."""
    current = {"knn": 5}

    assert tune.parse_overrides("knn=40 decay=0.5 verbose=true", current) == {
        "knn": 40, "decay": 0.5, "verbose": True}
    # …and with no `current` at all, exactly as before this existed
    assert tune.parse_overrides("knn=40") == {"knn": 40}


def test_a_spoken_phrase_drives_the_loop_end_to_end(tmp_path):
    """Through `run_tune_loop`, at the prompt a person actually types into — and the same seam
    the GUI's gate feeds free text to, so both surfaces get this without extra wiring."""
    session = _mock(out_dir=tmp_path)
    inputs = iter(["tune", "increase the k to 40", "accept"])

    accepted = tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine),
                                  lambda _="": next(inputs), lambda *_: None)

    assert accepted["params"]["knn"] == 40
    assert len(session.steps) == 2


# ── item 7, tier B: the sentence that names no parameter ─────────────────────────────────


def test_tier_b_is_only_reached_when_the_rules_decline():
    """Tier A stays the default: an explicit phrase must never spend a model call. Asserted by
    making any call raise — if `parse_overrides` consulted a model for "increase the k to 5",
    this fails instead of passing."""
    import manyruns.agents as agents_mod

    current = {"knn": 5}
    saved = agents_mod.call_tool
    try:
        agents_mod.call_tool = lambda **_kw: (_ for _ in ()).throw(
            AssertionError("tier A should have answered this"))
        assert tune.parse_overrides("increase the k to 5", current) == {"knn": 5}
        assert tune.parse_overrides("knn=40", current) == {"knn": 40}
    finally:
        agents_mod.call_tool = saved


def test_tier_b_moves_one_declared_parameter_by_one_step(monkeypatch):
    """"this plot looks too clustered" names no parameter, so the rules decline and a model
    decides WHICH knob — but not by how much.

    THE MAGNITUDE IS OURS, deliberately: the model returns a direction and the arithmetic happens
    here against the value the step actually holds. A model-chosen number is one nobody can check,
    and since #45 it would land in the decision corpus as a value a human chose."""
    import manyruns.agents as agents_mod

    monkeypatch.setattr(agents_mod, "call_tool",
                        lambda **_kw: {"param": "knn", "direction": "up"})

    assert tune.llm_override("this plot looks too clustered", {"knn": 5}) == {"knn": 6}
    monkeypatch.setattr(agents_mod, "call_tool",
                        lambda **_kw: {"param": "knn", "direction": "down"})
    assert tune.llm_override("too smooth", {"knn": 5}) == {"knn": 4}
    # a float moves by a fraction rather than by one — 1 is not a step for `decay=1.0`
    monkeypatch.setattr(agents_mod, "call_tool",
                        lambda **_kw: {"param": "decay", "direction": "up"})
    assert tune.llm_override("x", {"decay": 1.0}) == {"decay": 1.1}


def test_tier_b_refuses_a_parameter_this_step_does_not_have(monkeypatch):
    """The enum is built from `current`, so this should not happen — and it is checked anyway. A
    schema the model ignored is exactly what the seam returns None for, and a caller that trusted
    it would be the bug. Same discipline `intent.route_recipe` applies to recipe names."""
    import manyruns.agents as agents_mod

    monkeypatch.setattr(agents_mod, "call_tool",
                        lambda **_kw: {"param": "not_a_param", "direction": "up"})
    assert tune.llm_override("anything", {"knn": 5}) == {}

    monkeypatch.setattr(agents_mod, "call_tool",
                        lambda **_kw: {"param": "knn", "direction": "sideways"})
    assert tune.llm_override("anything", {"knn": 5}) == {}


def test_tier_b_is_absent_by_default_and_changes_nothing():
    """No key, no packages — this checkout and CI both. The whole tier resolves to "no change",
    which is what the loop already renders as an attempt that moved nothing."""
    assert tune.llm_override("this plot looks too clustered", {"knn": 5}) == {}
    assert tune.llm_override("anything", None) == {}


# ── the decision row names the run it belongs to ─────────────────────────────────────────


def test_a_tune_row_carries_the_run_that_produced_it(tmp_path):
    """The loop is the one surface where the params ARE the decision, so a tune row that
    cannot be joined to its run records a preference over settings nobody can recover."""
    import json

    from manyruns import tune

    session = _mock(out_dir=tmp_path)
    inputs = iter(["tune", "knn=40", "accept"])
    tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine),
                       lambda _="": next(inputs), lambda *_: None)
    rows = [json.loads(line) for line in
            (tmp_path / "decisions.jsonl").read_text().splitlines() if line.strip()]
    tune_rows = [r for r in rows if r["surface"] == "tune"]
    assert len(tune_rows) == 1
    assert tune_rows[0]["run_id"] == session.run_id


# ── a rejected attempt keeps its coordinates, not only its picture ──────────────────────


def test_a_rejected_attempt_keeps_its_coordinates_not_only_its_picture(tmp_path):
    """MEASURED before this fix: three attempts left three PNGs and ONE `.spec.npz`, because
    `_keep_attempt` renamed the paths in `record["plots"]` and `figspec` writes its sidecars
    beside them rather than into that list. Two things were lost — a rejected attempt could
    not be re-exported for a paper (contradicting `figspec.py`'s own "the durable object is
    the INPUT"), and no before/after readout could be computed, because only one of the two
    coordinate arrays survived."""
    import numpy as np

    from manyruns import tune, vocab

    session = _mock(engine=vocab.INPROC, out_dir=tmp_path)
    session.state["X"] = np.random.RandomState(0).randn(120, 8)
    inputs = iter(["tune", "knn=7", "tune", "knn=15", "accept"])
    tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine),
                       lambda _="": next(inputs), lambda *_: None)

    pngs = sorted(p.name for p in tmp_path.rglob("*.png"))
    npzs = sorted(p.name for p in tmp_path.rglob("*.spec.npz"))
    jsons = sorted(p.name for p in tmp_path.rglob("*.spec.json"))
    assert len(pngs) == 3, pngs
    assert len(npzs) == 3, f"{len(pngs)} pictures kept but {len(npzs)} coordinate sets: {npzs}"
    assert len(jsons) == 3, jsons
    # The @n convention is shared, so a reader can pair a picture with its coordinates by name.
    assert [p.rsplit(".", 1)[0] for p in pngs] == [n[: -len(".spec.npz")] for n in npzs]


def test_each_attempt_points_at_its_own_picture_so_the_pane_can_flip_between_them(tmp_path):
    """THE DEMO'S CORE GESTURE. Tune a step, then flip between the rejected picture and the kept
    one — the figures pane walks each step record's `plots` (`tui/state.py`'s `StepView`, off
    `runner`'s per-step `rec["plots"]`).

    `_keep_attempt` renamed the FILE and left the record holding the old path, which the next
    attempt then overwrote. Measured before this fix, on a two-attempt loop: both records named
    `out/plots/phate.png`, so the pane offered two entries over ONE file and showed attempt 2
    twice, while attempt 1's real picture sat on disk as `phate@1.png` with nothing pointing at
    it. Not merely unreachable — the pane asserted it was showing a picture it was not.
    """
    import pathlib

    import numpy as np

    from manyruns import tune, vocab

    session = _mock(engine=vocab.INPROC, out_dir=tmp_path)
    session.state["X"] = np.random.RandomState(0).randn(150, 8)
    inputs = iter(["tune", "knn=7", "accept"])
    tune.run_tune_loop(session, tune.parse_step_line("phate knn=5", session.engine),
                       lambda _="": next(inputs), lambda *_: None)

    named = [p for rec in session.steps for p in (rec.get("plots") or [])]
    assert len(named) == 2, f"one record per attempt, each with its picture: {named}"
    assert len(set(named)) == 2, f"both records name the same file, so the pane cannot flip: {named}"
    for path in named:
        assert pathlib.Path(path).is_file(), f"a record names a file that is not there: {path}"
    assert any("@1" in p for p in named), f"the rejected attempt keeps its own name: {named}"


@pytest.mark.parametrize("second_prompt", [False, True])
@pytest.mark.parametrize("bad", ["", "wat", "knn = 40", "=40", "knn=", "knn=oops",
                                 "knn=5", "knn=6 junk", "unknown=4", "knn=nan", "knn=5.5"])
def test_invalid_or_unchanged_params_never_fit_or_keep_a_figure(monkeypatch, second_prompt, bad):
    session = _mock()
    kept = []
    monkeypatch.setattr(tune, "_keep_attempt", lambda *args: kept.append(args))
    inputs = iter((["tune"] if second_prompt else []) + [bad] +
                  (["cancel"] if second_prompt else []) + ["accept"])
    out = []
    step = tune.parse_step_line("phate knn=5", session.engine)
    assert tune.run_tune_loop(session, step, lambda _: next(inputs), out.append) == step
    assert len(session.steps) == 1
    assert kept == []
    assert any("key=value" in line for line in out)


def test_cancel_at_params_returns_to_decision_without_renaming(monkeypatch):
    session = _mock()
    monkeypatch.setattr(tune, "_keep_attempt", lambda *args: pytest.fail("cancel renamed a figure"))
    answers = iter(["t", "cancel", "accept"])
    prompts = []
    def read(prompt):
        prompts.append(prompt)
        return next(answers)
    accepted = tune.run_tune_loop(session, tune.parse_step_line("phate", session.engine), read, print)
    assert accepted is not None
    assert prompts[0] == prompts[2]
    assert len(session.steps) == 1


def test_direct_overrides_preserve_case_and_use_the_offered_relative_baseline():
    assert tune.validated_overrides("label=MixedCase", {"label": "Original"}) == {
        "label": "MixedCase"}
    session = _mock()
    step = tune.parse_step_line("phate knn=15", session.engine)
    answers = iter(["increase the k by one", "accept"])
    accepted = tune.run_tune_loop(session, step, lambda _: next(answers), print)
    assert accepted["params"]["knn"] == 16
    assert [r["params"]["knn"] for r in session.steps] == [15, 16]


@pytest.mark.parametrize("second_prompt", [False, True])
@pytest.mark.parametrize("gesture,expected", [("knn=40", 40), ("increase the k by one", 6)])
def test_embed_accepts_offered_knobs_absent_from_recipe_params(second_prompt, gesture, expected):
    recipe = app.load_recipe("embed")
    step = next(s for s in recipe["steps"] if s["name"] == "phate")
    session = _mock(recipe=recipe)
    answers = iter((["t"] if second_prompt else []) + [gesture, "a"])
    accepted = tune.run_tune_loop(session, step, lambda _: next(answers), print)
    assert [r["params"] for r in session.steps] == [
        {"n_components": 3}, {"n_components": 3, "knn": expected}]
    assert accepted["params"]["knn"] == expected
    assert step["params"] == {"n_components": 3}


@pytest.mark.parametrize("bad", ["knnn=40", "verbose=true", "knn=5", "knn=5.5"])
def test_embed_refuses_typos_unoffered_declarations_and_invalid_or_unchanged_defaults(monkeypatch, bad):
    recipe = app.load_recipe("embed")
    step = next(s for s in recipe["steps"] if s["name"] == "phate")
    step = {**step, "params": {**step["params"], "verbose": False}}
    session = _mock(recipe=recipe)
    monkeypatch.setattr(tune, "_keep_attempt", lambda *args: pytest.fail("refusal kept a figure"))
    answers = iter([bad, "a"])
    out = []
    assert tune.run_tune_loop(session, step, lambda _: next(answers), out.append) == step
    assert len(session.steps) == 1
    if bad in ("knnn=40", "verbose=true"):
        assert any("offered: t, decay, knn, n_components" in line for line in out)
    else:
        assert any("params unchanged" in line or "invalid value" in line for line in out)


def test_step_line_relative_phrase_uses_the_same_default_as_the_strip():
    assert tune.parse_step_line("phate increase the k by one", "mock")["params"] == {
        "n_components": 3, "knn": 6}


def test_ask_caps_follow_retained_input_and_survive_the_said_window():
    import numpy as np
    from manyruns.tui.gate import Gate
    from manyruns.tui.run import RunScreen

    session = _mock()
    session.state["emb"] = np.zeros((100, 8))
    step = {"name": "phate", "group": "latent", "params": {"n_components": 3, "knn": 15},
            "limits": {"n_components": ["n_samples", "n_features"]}}
    screen = RunScreen()
    answers = iter(["knn=16", "accept"])
    contexts = []
    issued = session.step
    def produce_lower_dimensional_output(action):
        rec = issued(action)
        session.state["emb"] = np.zeros((100, 2))
        return rec
    session.step = produce_lower_dimensional_output

    class ScriptedGate(Gate):
        def read(self, prompt="", *, kind=None):
            # The latest output cannot bound a retry: it has two features, the input eight.
            assert session.state["emb"].shape == (100, 2)
            screen.said = ["old bounds note", "one", "two", "three"]
            contexts.append(screen.ask_context())
            return next(answers)

    gate = ScriptedGate(on_arm=screen.arm_tuning)
    tune.run_tune_loop(session, step, gate.read, gate.write)
    for context in contexts:
        assert '"n_components": {"cap": 8, "status": "resolved"}' in context
        assert '"knn": {"cap": null, "reason": "no_limit", "status": "unknown"}' in context
        assert "old bounds note" not in context
        assert "unknown means manyruns asserts nothing" in context
    assert [rec["constraints"]["n_components"]["cap"] for rec in session.steps] == [8, 8]


@pytest.mark.parametrize("declaration", ["phate", "phate t=auto"])
def test_an_initial_string_can_be_restored_after_trying_a_number(declaration):
    session = _mock()
    step = tune.parse_step_line(declaration, session.engine)
    answers = iter(["t=20", "t=auto", "accept"])
    accepted = tune.run_tune_loop(session, step, lambda _: next(answers), print)
    assert accepted["params"]["t"] == "auto"
    assert [rec["params"].get("t", "auto") for rec in session.steps] == ["auto", 20, "auto"]


@pytest.mark.parametrize('line,requested', [('pca', 50), ('pca n_components=7', 7)])
def test_typed_pca_uses_recipe_before_bounds(line, requested, tmp_path):
    import numpy as np
    session = _mock(recipe=app.load_recipe('embed'), array=np.zeros((20, 4)), out_dir=tmp_path)
    step = tune.parse_step_line(line, session.engine, recipe=session.recipe)
    assert step['params']['n_components'] == requested
    rec = session.apply(step)
    assert rec['params']['n_components'] == 4
    assert rec['bounded']
    assert tune.parse_step_line('pca', 'mock')['params']['n_components'] == 10


def test_repl_passes_current_recipe_to_typed_step(monkeypatch):
    session = _mock(recipe=app.load_recipe('embed'))
    seen = []
    monkeypatch.setattr(tune, 'run_tune_loop', lambda s, step, *a: seen.append(step) or None)
    inputs = iter(['pca', 'quit'])
    app.interactive_session(session, read=lambda *a: next(inputs), write=lambda *a: None)
    assert seen[0]['params']['n_components'] == 50
