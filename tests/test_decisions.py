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

from manyruns import decisions, vocab  # noqa: E402

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
    assert set(row) == {"schema_version", "decision_id", "run_id", "at", "surface", "chooser",
                        "bound", "legal_id", "dataset", "shape", "topology", "offered", "chosen"}


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


# ── who chose, how the choice was bound, and over which task space ──────────
def _row(tmp_path):
    (row,) = list(decisions.read(tmp_path))
    return row


def test_an_existing_caller_records_a_human_with_no_bound(tmp_path):
    """Environment contract §2.1-2.2: the two callers that exist pass neither field, and both
    are a person choosing from a rendered list."""
    decisions.append(offered=OFFERED, chosen="embed", out_dir=tmp_path)
    row = _row(tmp_path)
    assert row["chooser"] == "human" and row["bound"] == "none"


def test_a_model_chooser_is_recorded(tmp_path):
    decisions.append(offered=OFFERED, chosen="embed", chooser="llm:claude-opus-4-8",
                     bound="schema", out_dir=tmp_path)
    row = _row(tmp_path)
    assert row["chooser"] == "llm:claude-opus-4-8" and row["bound"] == "schema"


def test_an_unknown_bound_is_a_caller_bug_not_a_row(tmp_path):
    """A wrong `bound` is the silent lie §2.2 warns about, so it raises rather than records."""
    assert decisions.BOUNDS == ("logits", "schema", "prompt", "none")
    with pytest.raises(ValueError, match="enum"):
        decisions.append(offered=OFFERED, chosen="embed", bound="enum", out_dir=tmp_path)
    assert list(decisions.read(tmp_path)) == []


def test_an_empty_chooser_raises_before_the_file_is_opened(tmp_path):
    with pytest.raises(ValueError):
        decisions.append(offered=OFFERED, chosen="embed", chooser="", out_dir=tmp_path)
    assert not (tmp_path / "decisions.jsonl").exists()


def test_a_non_string_chooser_raises(tmp_path):
    with pytest.raises(ValueError):
        decisions.append(offered=OFFERED, chosen="embed", chooser=None, out_dir=tmp_path)
    assert list(decisions.read(tmp_path)) == []


def test_validation_runs_before_the_empty_offer_check(tmp_path):
    """A caller bug does not hide behind the empty-offer no-op."""
    with pytest.raises(ValueError):
        decisions.append(offered=[], chosen="embed", bound="enum", out_dir=tmp_path)


def test_the_legal_id_of_a_ledger_offer_with_one_blocked_recipe(tmp_path):
    """The id is a hash of the legal set and the vocabulary version, and nothing else: the
    blocked `contrast` is not in it."""
    assert decisions._legal_set(OFFERED) == ["cluster", "embed"]
    _write(tmp_path)
    assert _row(tmp_path)["legal_id"] == "d293a2ccd28b"


def test_the_order_of_the_offer_does_not_change_the_legal_id(tmp_path):
    _write(tmp_path, offered=list(reversed(OFFERED)))
    assert _row(tmp_path)["legal_id"] == "d293a2ccd28b"


def test_an_entry_without_can_run_is_legal(tmp_path):
    """The offer `tests/test_record_schema.py` writes."""
    assert decisions._legal_set([{"recipe": "embed"}]) == ["embed"]
    _write(tmp_path, offered=[{"recipe": "embed"}])
    assert _row(tmp_path)["legal_id"] == "85075bbaef66"


def test_an_offer_with_no_named_entry_has_an_empty_legal_set(tmp_path):
    """The offer `tests/test_tui_run.py` writes. The row is still written."""
    assert decisions._legal_set([{"a": 1}, {"b": 2}]) == []
    _write(tmp_path, offered=[{"a": 1}, {"b": 2}])
    assert _row(tmp_path)["legal_id"] == "25931395e736"


def test_the_vocabulary_version_changes_the_legal_id(tmp_path, monkeypatch):
    """`append` reads `vocab.VERSION` at call time, so a bump takes effect without a reload."""
    monkeypatch.setattr(vocab, "VERSION", "2")
    _write(tmp_path)
    assert _row(tmp_path)["legal_id"] == "63c20f253f6b"


def test_a_tune_offer_is_named_by_its_label():
    """Two attempts of one step differ only by params, and the label renders the params."""
    offered = [{"step": "pca", "params": {}, "label": "b"},
               {"step": "pca", "params": {}, "label": "a"}]
    assert decisions._legal_set(offered) == ["a", "b"]


def test_duplicate_tune_labels_count_once(tmp_path):
    offered = [{"step": "pca", "params": {"n_components": 2}, "label": "pca n_components=2"}] * 2
    assert decisions._legal_set(offered) == ["pca n_components=2"]
    _write(tmp_path, offered=offered)
    assert _row(tmp_path)["legal_id"] == "e28f4684e82c"


def test_a_caller_cannot_set_the_legal_id(tmp_path):
    """A caller that passes its own id can compute it another way, and the corpus cannot tell."""
    with pytest.raises(TypeError):
        decisions.append(offered=OFFERED, chosen="embed", legal_id="x", out_dir=tmp_path)
    assert list(decisions.read(tmp_path)) == []


def test_the_new_keys_follow_surface_and_the_version_stays_1(tmp_path):
    _write(tmp_path)
    row = json.loads(decisions.index_path(tmp_path).read_text())
    assert list(row) == ["schema_version", "decision_id", "run_id", "at", "surface", "chooser",
                         "bound", "legal_id", "dataset", "shape", "topology", "offered", "chosen"]
    assert row["schema_version"] == 1


def test_a_legacy_row_is_read_unchanged(tmp_path):
    """No migration: a row written before these fields existed reads without them."""
    path = decisions.index_path(tmp_path)
    path.write_text(json.dumps({"offered": OFFERED, "chosen": "embed"}) + "\n")
    before = path.read_bytes()

    row = next(decisions.read(tmp_path))
    assert not {"chooser", "bound", "legal_id"} & set(row)
    assert path.read_bytes() == before


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
                  "recommended": ["embed"], "chosen": "embed", "took_default": True}


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


# ── filtering the training pairs ─────────────────────────────────────────────
PAIR_KEYS = {"shape", "topology", "legal", "blocked", "recommended", "chosen", "took_default"}


def _hand_written(tmp_path, **row):
    """A row as an older writer left it: no `surface`, no `chooser`."""
    row = {"offered": OFFERED, "chosen": "embed", **row}
    with decisions.index_path(tmp_path).open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def test_examples_keep_the_ledger_surface_by_default(tmp_path):
    _write(tmp_path, surface="ledger")
    _write(tmp_path, surface="audit")
    assert len(decisions.examples(tmp_path)) == 1
    assert len(decisions.examples(tmp_path, surface="audit")) == 1
    assert len(decisions.examples(tmp_path, surface=None)) == 2


def test_a_row_without_a_surface_counts_as_ledger(tmp_path):
    _hand_written(tmp_path)
    assert len(decisions.examples(tmp_path)) == 1


def _three_choosers(tmp_path):
    """Each chooser takes a different recipe, so a surviving pair says whose row it came from."""
    for chooser, chosen in (("human", "embed"), ("llm:claude-opus-4-8", "cluster"),
                            ("llm:qwen3:8b", "contrast")):
        _write(tmp_path, chooser=chooser, chosen=chosen)


def test_examples_drop_one_model_by_exact_chooser(tmp_path):
    """Environment contract §2.1: a row a policy chose is not training data for that policy."""
    _three_choosers(tmp_path)
    assert len(decisions.examples(tmp_path, exclude_chooser="llm:claude-opus-4-8")) == 2


def test_examples_drop_every_model_by_prefix(tmp_path):
    _three_choosers(tmp_path)
    (ex,) = decisions.examples(tmp_path, exclude_chooser="llm:")
    assert ex["chosen"] == "embed"


def test_examples_drop_by_a_list_of_choosers(tmp_path):
    _three_choosers(tmp_path)
    (ex,) = decisions.examples(tmp_path, exclude_chooser=["human", "llm:qwen3:8b"])
    assert ex["chosen"] == "cluster"


def test_a_chooser_without_a_trailing_colon_matches_exactly(tmp_path):
    _three_choosers(tmp_path)
    assert len(decisions.examples(tmp_path, exclude_chooser="llm")) == 3


def test_a_legacy_row_counts_as_human(tmp_path):
    _hand_written(tmp_path)
    assert decisions.examples(tmp_path, exclude_chooser="human") == []


def test_a_pair_does_not_carry_the_chooser(tmp_path):
    """The pair keeps its seven keys; a trainer filters with `exclude_chooser` instead."""
    _write(tmp_path, chooser="llm:x")
    (ex,) = decisions.examples(tmp_path)
    assert set(ex) == PAIR_KEYS


def test_the_default_rate_uses_the_default_filters(tmp_path):
    _write(tmp_path, chosen="embed")
    _write(tmp_path, chosen="embed", surface="audit")
    assert decisions.default_rate(tmp_path) == 1.0
    assert len(decisions.examples(tmp_path)) == 1


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
    assert claimed == {"archetypes": 4, "branches": 3, "cflows": 2, "cluster": 1, "contrast": 1,
                       "embed": 0, "markers": 1, "pseudotime": 2,
                       # `traced` is `cflows` plus a probe, and claims the same structure: the
                       # probe REPORTS on the trajectory rather than asserting a second one, so
                       # it grades the same rung rather than a higher one.
                       "traced": 2}


def test_the_unreachable_rung_is_reported_rather_than_hidden():
    """`unroutable_rungs()` went [3, 4] → [3] → []: `archetypes` claimed rung 4 while it was
    `blocked:` upstream, and `branches` now claims rung 3. With manylatents 0.1.9 `archetypes`
    runs, so the catalogue answer and the offer answer agree: rung 4 is closed in fact, not only
    on paper. Pinned so a regression on either rung is reported rather than hidden."""
    from manyruns import narrate, score
    from manyruns.catalog import load_recipe

    assert score.unroutable_rungs() == []
    assert narrate.unavailable(load_recipe("archetypes")) == []


# ── the gate and the proposed block: two optional core keys that `run --choice` writes ──
_THIRTEEN = ["schema_version", "decision_id", "run_id", "at", "surface", "chooser", "bound",
             "legal_id", "dataset", "shape", "topology", "offered", "chosen"]


@pytest.mark.parametrize("gate", [True, False])
def test_append_records_gate(tmp_path, gate):
    _write(tmp_path, gate=gate)
    row, = decisions.read(tmp_path)
    assert row["gate"] is gate
    assert list(row) == [*_THIRTEEN, "gate"]


def test_append_without_gate_writes_no_key(tmp_path):
    _write(tmp_path)
    row, = decisions.read(tmp_path)
    assert list(row) == _THIRTEEN


@pytest.mark.parametrize("gate", ["on", 1, 0])
def test_append_refuses_non_bool_gate(tmp_path, gate):
    with pytest.raises(ValueError, match="gate"):
        _write(tmp_path, gate=gate)
    assert not (tmp_path / "decisions.jsonl").exists()


def test_row_without_gate_reads_as_missing(tmp_path):
    hand = {"offered": [{"recipe": "embed"}], "chosen": "embed"}
    (tmp_path / "decisions.jsonl").write_text(json.dumps(hand) + "\n")
    decisions.append(offered=[{"recipe": "embed"}], chosen="embed", out_dir=tmp_path)
    before = (tmp_path / "decisions.jsonl").read_bytes()
    rows = list(decisions.read(tmp_path))
    assert len(rows) == 2
    assert all("gate" not in row and row.get("gate") is None for row in rows)
    assert list(rows[1]) == _THIRTEEN
    assert (tmp_path / "decisions.jsonl").read_bytes() == before


def test_append_records_proposed_last(tmp_path):
    proposed = {"chooser": "llm:x", "bound": "schema", "chosen": "embed",
                "offer_sha256": "0" * 64}
    decisions.append(offered=[{"recipe": "embed"}], chosen="embed", chooser="human:a3",
                     gate=True, proposed=proposed, out_dir=tmp_path)
    row, = decisions.read(tmp_path)
    assert list(row)[-2:] == ["gate", "proposed"]
    assert row["proposed"]["chooser"] == "llm:x"
    assert row["proposed"] == proposed


def test_append_without_proposed_writes_no_key(tmp_path):
    _write(tmp_path, gate=True)
    row, = decisions.read(tmp_path)
    assert "proposed" not in row and row.get("proposed") is None


@pytest.mark.parametrize("proposed", ["embed", ["embed"]])
def test_append_refuses_non_dict_proposed(tmp_path, proposed):
    with pytest.raises(ValueError, match="proposed"):
        decisions.append(offered=[{"recipe": "embed"}], chosen="embed", proposed=proposed,
                         out_dir=tmp_path)
    assert not (tmp_path / "decisions.jsonl").exists()


def test_append_copies_proposed(tmp_path):
    proposed = {"chooser": "llm:x", "bound": "schema", "chosen": "embed",
                "offer_sha256": "0" * 64, "meta": {"tags": ["a"]}}
    decisions.append(offered=[{"recipe": "embed"}], chosen="embed", chooser="human:a3",
                     proposed=proposed, out_dir=tmp_path)
    proposed["chosen"] = "cluster"
    proposed["meta"]["tags"].append("b")
    row, = decisions.read(tmp_path)
    assert row["proposed"]["chosen"] == "embed"
    assert row["proposed"]["meta"] == {"tags": ["a"]}
