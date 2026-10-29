"""The decision store — `manyruns/decisions.py` and the one call site that writes to it.

Measured 2026-07-31, before any of this existed: `index.jsonl` held 7 rows and every one recorded
what RAN. Not one recorded what else was on offer, so a selector trained on that corpus learns
"pick `embed` or `archetypes`" and cannot see that `archetypes` was one of eight or that
`contrast` was refused for a stated reason. The choice was the label, and the label was missing.

What is pinned here is the corpus's integrity, because a corpus is the one artifact whose bugs
are invisible at the point of use: an offer set that was recomputed rather than recorded, a
decision dropped because the run it started died, or a selection bias introduced by writing the
row after the outcome is known would each produce a file that looks fine and teaches the wrong
thing.
"""
from __future__ import annotations

import json

import pytest

from manyruns import decisions  # noqa: E402

OFFERED = [
    {"recipe": "embed", "can_run": True, "blocked": [], "recommended": True},
    {"recipe": "cluster", "can_run": True, "blocked": [], "recommended": False},
    {"recipe": "contrast", "can_run": False, "recommended": False,
     "blocked": ["needs a healthy/disease column"]},
]


def _write(tmp_path, **kw):
    kw.setdefault("offered", OFFERED)
    kw.setdefault("chosen", "embed")
    kw.setdefault("dataset", "swissroll")
    kw.setdefault("shape", "manifold")
    kw.setdefault("topology", ["surface"])
    return decisions.append(out_dir=tmp_path, **kw)


def test_a_row_names_the_run_it_belongs_to(tmp_path):
    """manyruns-as-environment.md §5: decisions and runs are two streams that do not join,
    so a learner reading the corpus sees choices without settings. This is the join, and it
    runs decision -> run because `runner.identity` mints the run_id first."""
    _write(tmp_path, run_id="0123456789ab")
    row = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert row["run_id"] == "0123456789ab"


def test_a_row_written_without_a_run_says_so_rather_than_omitting_the_field(tmp_path):
    """`null` is the honest answer for every row written before this existed: that decision
    was not joined to anything. An ABSENT key would make a reader guess which."""
    _write(tmp_path)
    row = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert "run_id" in row and row["run_id"] is None


# ── the row ──────────────────────────────────────────────────────────────────
def test_the_row_carries_what_was_chosen_and_what_it_was_chosen_from(tmp_path):
    """The whole point. An execution record says what ran; this says what the alternatives
    were, which is what makes the choice a label rather than an observation."""
    decision_id = _write(tmp_path)

    (row,) = list(decisions.read(tmp_path))
    assert row["decision_id"] == decision_id
    assert row["chosen"] == "embed"
    assert [o["recipe"] for o in row["offered"]] == ["embed", "cluster", "contrast"]
    assert row["offered"][2]["blocked"] == ["needs a healthy/disease column"]


def test_the_truth_side_is_copied_not_joined(tmp_path):
    """`topology` is in the row, not looked up later from the dataset file. A declaration can be
    edited after the fact, and a corpus whose labels move is not a corpus."""
    _write(tmp_path, topology=["cycle", "surface"])

    (row,) = list(decisions.read(tmp_path))
    assert row["topology"] == ["cycle", "surface"]


def test_the_row_carries_no_data(tmp_path):
    """The federated line: geometry and steps leave, the person's data does not. Names, shape,
    topology and legality reasons only — no coordinates, no counts, no matrix."""
    _write(tmp_path)

    text = decisions.index_path(tmp_path).read_text()
    row = json.loads(text)
    assert set(row) == {"schema_version", "decision_id", "run_id", "at", "surface", "dataset", "shape",
                        "topology", "offered", "chosen"}


def test_a_choice_from_nothing_is_not_a_decision(tmp_path):
    """An empty offer set has no alternatives for the label to be a label against."""
    assert decisions.append(offered=[], chosen="embed", out_dir=tmp_path) is None
    assert list(decisions.read(tmp_path)) == []


def test_a_corpus_file_that_cannot_be_written_never_fails_the_run(tmp_path):
    """A scientist's chosen analysis must not fail to start because a corpus file could not be
    opened. The caller carries on with `decision_id=None`, which reads on the run row exactly as
    every row written before this module existed."""
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        assert _write(blocked) is None
    finally:
        blocked.chmod(0o700)


def test_rows_accumulate_and_a_corrupt_one_is_skipped(tmp_path):
    """Append-only, and a half-written line from a killed process must not make the whole corpus
    unreadable — `store.read`'s rule, for the same reason."""
    _write(tmp_path, chosen="embed")
    _write(tmp_path, chosen="cluster")
    with decisions.index_path(tmp_path).open("a") as fh:
        fh.write("{not json\n")
    _write(tmp_path, chosen="markers")

    assert [r["chosen"] for r in decisions.read(tmp_path)] == ["embed", "cluster", "markers"]


# ── the training shape ───────────────────────────────────────────────────────
def test_examples_are_the_pairs_a_selector_sees(tmp_path):
    """Inputs a selector has at inference (the observation and the legal moves) plus what a
    human actually did. Kept in this module so the selector's notion of an example and the row
    format cannot drift apart in two repos."""
    _write(tmp_path)

    (ex,) = decisions.examples(tmp_path)
    assert ex == {"shape": "manifold", "topology": ("surface",),
                  "legal": ["embed", "cluster"],
                  "blocked": {"contrast": "needs a healthy/disease column"},
                  "recommended": ["embed"], "chosen": "embed", "took_default": True,
                  "overrode": None}


def test_the_corpus_says_how_often_the_human_just_took_the_default(tmp_path):
    """THE CONFOUND, carried in the example rather than discovered after training.

    The ledger puts the recommendation first and the cursor starts there, so `enter` takes it —
    the first real decision this store recorded, from driving the shipped binary, was `embed`
    chosen with `embed` recommended out of six legal moves. At `default_rate == 1.0` the corpus
    is `narrate.offer`'s prior with a human's name on it, and a selector scoring well on it has
    learned to agree with the default. That is the failure mode that looks most like success,
    so the number is computable before anyone reports an accuracy."""
    assert decisions.default_rate(tmp_path) is None          # nothing to divide by

    _write(tmp_path, chosen="embed")                          # embed is the recommended row
    assert decisions.default_rate(tmp_path) == 1.0

    _write(tmp_path, chosen="cluster")                        # a human who went elsewhere
    assert decisions.default_rate(tmp_path) == 0.5
    assert [e["took_default"] for e in decisions.examples(tmp_path)] == [True, False]


def test_a_decision_with_one_legal_move_is_not_an_example(tmp_path):
    """No counterfactual: the human "chose" the only option. An example like that teaches the
    prior, not the policy — and the drop is countable rather than hidden, because `read` still
    returns the row."""
    _write(tmp_path, offered=[OFFERED[0], OFFERED[2]])       # one runnable, one hollow

    assert len(list(decisions.read(tmp_path))) == 1
    assert decisions.examples(tmp_path) == []


# ── the call site ────────────────────────────────────────────────────────────
def test_the_offer_is_the_rows_the_screen_drew(tmp_path):
    """`LedgerRow.as_offer` is the ONE conversion, and it lives beside the row rather than in
    the store, so the corpus records what was on screen instead of a recomputed ledger."""
    pytest.importorskip("textual")
    from manyruns.tui.state import LedgerRow

    row = LedgerRow(recipe="contrast", ask="How different?", gloss="", can_run=False,
                    blocked=("needs a healthy/disease column",), recommended=False)

    # `measured` joined the record so the corpus can tell "picked the row the numbers warned
    # about" from "picked a row nothing was known about". A refused row reads `refused`.
    assert row.as_offer() == {"recipe": "contrast", "can_run": False,
                              "blocked": ["needs a healthy/disease column"],
                              "recommended": False, "measured": "refused"}
    # the PHRASING is deliberately absent: a corpus carrying it would re-label itself every
    # time a recipe's `question:` was reworded
    assert "ask" not in row.as_offer() and "gloss" not in row.as_offer()


def test_the_decision_is_written_before_the_run_can_decline(tmp_path, monkeypatch):
    """Ordering, and it is the point rather than a detail. `open_run` declines when no compute
    backend is installed, and a declined choice is still a decision — the person was offered
    eight analyses and picked one. Writing the row afterwards would keep exactly the decisions
    that happened to succeed, which is the selection bias a corpus can least afford."""
    pytest.importorskip("textual")
    from manyruns.tui.app import ManyrunsApp
    from manyruns.tui.state import DataEntry, LedgerRow
    from manyruns import narrate

    monkeypatch.chdir(tmp_path)
    obs = narrate.Observation(source="s", shape="manifold", modality="scrna")
    entry = DataEntry(name="swissroll", kind="bundled", obs=obs, dataset="swissroll",
                      topology=("surface",))

    class _Screen:
        rows = [LedgerRow(recipe="embed", ask="a", gloss="g", can_run=True),
                LedgerRow(recipe="cluster", ask="b", gloss="g", can_run=True)]

    app = ManyrunsApp()
    declined: list = []
    app.open_run = lambda e, r, decision=None: declined.append((r, decision))

    app.chose(entry, _Screen(), "embed")

    (row,) = list(decisions.read(tmp_path / "outputs"))
    assert row["chosen"] == "embed" and row["topology"] == ["surface"]
    assert declined == [("embed", row["decision_id"])], "the id reaches the run"


def test_backing_out_of_the_ledger_writes_nothing(tmp_path, monkeypatch):
    """One line per COMMITTED choice. An escape is ambiguous — interrupted, changed their mind,
    mis-keyed — and recording it is the first step onto interaction telemetry."""
    pytest.importorskip("textual")
    from manyruns.tui.app import ManyrunsApp
    from manyruns.tui.state import DataEntry
    from manyruns import narrate

    monkeypatch.chdir(tmp_path)
    obs = narrate.Observation(source="s", shape="manifold", modality="scrna")
    app = ManyrunsApp()

    app.chose(DataEntry(name="d", kind="bundled", obs=obs), object(), None)

    assert list(decisions.read(tmp_path / "outputs")) == []


# ── the ladder, which is what grades the corpus ──────────────────────────────
def test_every_recipe_that_asserts_a_structure_now_declares_it():
    """Measured before: 3 of 8. Tier 1 of the metric is `-ladder_cost(truth, pred)`, and a
    recipe with no `claims:` contributes rung 0 — so five of eight choices could not be graded
    at all. `qc` still abstains, and that is the correct answer rather than a gap: quality
    control asserts no structure, and `check_claims` makes absence legal so it can say so."""
    from manyruns import score
    from manyruns.catalog import discover_recipes

    claimed = score.claimed_rungs()
    # Three abstain, for the same reason stated three ways: `qc` measures and asserts no
    # structure, `sandbox` computes nothing at all (`pipeline/stubs.py`), and `preprocess`
    # filters and rescales — operations that are as true of a tree as of a blob, so it asserts
    # nothing about which. `check_claims` makes absence legal precisely so a recipe can say so
    # rather than pick the nearest word.
    assert set(discover_recipes()) - set(claimed) == {"qc", "sandbox", "preprocess"}
    assert claimed == {"archetypes": 4, "cflows": 2, "cluster": 1, "contrast": 1,
                       "embed": 0, "markers": 1, "pseudotime": 2,
                       # `traced` is `cflows` plus a probe, and claims the same structure: the
                       # probe REPORTS on the trajectory rather than asserting a second one, so
                       # it grades the same rung rather than a higher one.
                       "traced": 2}


def test_the_unreachable_rung_is_reported_rather_than_hidden():
    """`unroutable_rungs()` went [3, 4] → [3]: `archetypes` now claims rung 4. It reads what
    recipes CLAIM, not what they can RUN, and `archetypes` is `blocked:` upstream — so rung 4 is
    closed on paper and unreachable in fact. Pinned so the distinction is not lost: a perfect
    selector is still capped by both, and `multi-branching` has no recipe at all while
    `tree_narrow`/`tree_wide` ship as datasets whose correct answer nothing can give."""
    from manyruns import narrate, score
    from manyruns.catalog import load_recipe

    assert score.unroutable_rungs() == [3]
    assert narrate.unavailable(load_recipe("archetypes")), "rung 4's only claimant is blocked"


# ── schema version 2: a proposal and the choice made against it ─────────────
V1_KEYS = ["schema_version", "decision_id", "run_id", "at", "surface", "dataset", "shape",
           "topology", "offered", "chosen"]
V2_EXTRA = ["chooser", "bound", "legal_id", "run_before", "state_before", "proposed",
            "chosen_params", "rationale"]

PROPOSAL = {"action": "embed", "params": {}, "probabilities": {"embed": 0.6, "cluster": 0.4},
            "confidence": 0.4, "chooser": "agent:x", "bound": "schema"}


def _row(tmp_path, index=-1):
    return json.loads(decisions.index_path(tmp_path).read_text().splitlines()[index])


def test_v1_row_unchanged_without_chooser(tmp_path):
    """No `chooser`, no change: the row a person's ledger choice writes today, key for key."""
    _write(tmp_path)
    row = _row(tmp_path)
    assert list(row) == V1_KEYS
    assert row["schema_version"] == 1


def test_v2_row_has_proposal_and_choice_side_by_side(tmp_path):
    from manyruns import actions

    lid = actions.legal_id(["embed", "cluster"], "0.1.0 · revision unavailable")
    _write(tmp_path, chosen="cluster", chooser="human", bound="none", legal_id=lid,
           proposed=PROPOSAL, run_before="e89076c7b8e4",
           state_before={"n_obs": 2700, "n_vars": 32738, "g_vector": {}},
           rationale="the groups matter more than the layout", surface="cli")
    row = _row(tmp_path)
    assert list(row) == V1_KEYS + V2_EXTRA
    assert row["schema_version"] == 2
    assert row["chosen"] == "cluster" and row["proposed"] == PROPOSAL
    assert row["chooser"] == "human" and row["bound"] == "none" and row["legal_id"] == lid
    assert row["run_before"] == "e89076c7b8e4"
    assert row["state_before"] == {"n_obs": 2700, "n_vars": 32738, "g_vector": {}}
    assert row["chosen_params"] is None
    assert row["rationale"] == "the groups matter more than the layout"


def test_v2_row_fills_nothing_it_was_not_given(tmp_path):
    _write(tmp_path, chooser="rule:my-lab")
    row = _row(tmp_path)
    assert row["schema_version"] == 2
    assert all(row[k] is None for k in V2_EXTRA if k != "chooser")


def test_check_proposal_accepts_a_valid_proposal():
    assert decisions.check_proposal(PROPOSAL, OFFERED) == []


def test_check_proposal_rejects_bad_probabilities():
    for probabilities in ({"embed": 0.5, "cluster": 0.4},          # sums to 0.9
                          {"embed": 0.6, "contrast": 0.4},         # contrast is not legal here
                          {"embed": 1.1, "cluster": -0.1}):        # a value outside [0, 1]
        problems = decisions.check_proposal({**PROPOSAL, "probabilities": probabilities},
                                            OFFERED)
        assert problems, probabilities
    assert decisions.check_proposal(
        {**PROPOSAL, "probabilities": {"embed": 0.6004, "cluster": 0.4}}, OFFERED) == []


def test_check_proposal_rejects_illegal_action():
    problems = decisions.check_proposal({**PROPOSAL, "action": "contrast"}, OFFERED)
    assert problems and "needs a healthy/disease column" in " ".join(problems)
    assert decisions.check_proposal({**PROPOSAL, "action": "nonesuch"}, OFFERED)


def test_check_proposal_rejects_bad_chooser_and_bound():
    assert decisions.check_proposal({**PROPOSAL, "chooser": ""}, OFFERED)
    assert decisions.check_proposal({**PROPOSAL, "chooser": None}, OFFERED)
    assert decisions.check_proposal({**PROPOSAL, "bound": "trust me"}, OFFERED)
    assert decisions.BOUNDS == ("logits", "schema", "prompt", "none")


def test_check_proposal_accepts_no_probabilities():
    proposal = {"action": "embed", "chooser": "rule:my-lab", "bound": "prompt"}
    assert decisions.check_proposal(proposal, OFFERED) == []


def test_proposal_meta_stored_verbatim(tmp_path):
    meta = {"model": "m", "stage": 0}
    proposal = {**PROPOSAL, "meta": meta}
    assert decisions.check_proposal(proposal, OFFERED) == []
    _write(tmp_path, chooser="human", bound="none", proposed=proposal)
    assert _row(tmp_path)["proposed"]["meta"] == meta
    nested = {**PROPOSAL, "meta": {"model": {"name": "m"}}}
    assert decisions.check_proposal(nested, OFFERED)
    wide = {**PROPOSAL, "meta": {f"k{i}": i for i in range(17)}}
    assert decisions.check_proposal(wide, OFFERED)
    assert decisions.check_proposal({**PROPOSAL, "meta": {f"k{i}": i for i in range(16)}},
                                    OFFERED) == []


def test_examples_excludes_policy_rows(tmp_path):
    """Environment contract §6 item 3: a row a policy chose is not an example of a person's
    choice, so `examples` leaves it out by construction."""
    _write(tmp_path, chosen="embed", chooser="agent:x", bound="schema")
    _write(tmp_path, chosen="cluster", chooser="human", bound="none")
    assert [e["chosen"] for e in decisions.examples(tmp_path)] == ["cluster"]


def test_examples_marks_override(tmp_path):
    _write(tmp_path)                                                       # version 1
    _write(tmp_path, chosen="cluster", chooser="human", bound="none", proposed=PROPOSAL)
    _write(tmp_path, chosen="embed", chooser="human", bound="none", proposed=PROPOSAL)
    _write(tmp_path, chosen="embed", chooser="human", bound="none")        # nothing proposed
    assert [e["overrode"] for e in decisions.examples(tmp_path)] == [None, True, False, None]


def test_examples_reads_mixed_versions(tmp_path):
    """Review focus: one file, both versions, and a version 1 line with no version at all."""
    _write(tmp_path, chosen="embed")
    _write(tmp_path, chosen="cluster", chooser="human", bound="none", proposed=PROPOSAL)
    legacy = json.loads(decisions.index_path(tmp_path).read_text().splitlines()[0])
    legacy.pop("schema_version")
    legacy["chosen"] = "cluster"
    with decisions.index_path(tmp_path).open("a") as fh:
        fh.write(json.dumps(legacy) + "\n")
    rows = list(decisions.read(tmp_path))
    assert [r["schema_version"] for r in rows] == [1, 2, 1]
    examples = decisions.examples(tmp_path)
    assert [e["chosen"] for e in examples] == ["embed", "cluster", "cluster"]
    assert [e["overrode"] for e in examples] == [None, True, None]
    assert decisions.default_rate(tmp_path) == 1 / 3
