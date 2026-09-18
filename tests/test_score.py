"""The complexity ladder (manyruns/score.py) — pure, dependency-free scoring."""
import pytest

from manyruns import score
from manyruns.vocab import TOPOLOGIES


def test_ladder_covers_the_vocabulary():
    """Every topology has a rung and vice versa — the drift vocab.py exists to prevent."""
    assert score.check_ladder() == []


def test_truth_rung_is_the_max_over_a_multilabel_set():
    """A dataset that is both a cycle and a surface supports the STRONGER claim.

    Max, not min or mean: the ladder asks what the data can support, and a torus supports a
    one-dimensional continuum even though it is also a surface.
    """
    assert score.truth_rung(["cycle", "surface"]) == 2
    assert score.truth_rung(["surface"]) == 0
    assert score.truth_rung(["archetypal"]) == 4


def test_unlabelled_dataset_scores_as_none_not_zero():
    """None so it can be EXCLUDED. Scoring it as rung 0 would silently count an unlabelled
    dataset as evidence that `embed` was the right call."""
    assert score.truth_rung([]) is None
    assert score.truth_rung(None) is None


def test_overshoot_costs_more_than_undershoot():
    """The asymmetry is the whole point: under-shooting loses signal, over-shooting invents it."""
    assert score.ladder_cost(truth=3, pred=1) == 2.0     # 2 rungs under → 2
    assert score.ladder_cost(truth=1, pred=3) == 4.0     # 2 rungs over  → 4
    assert score.ladder_cost(truth=2, pred=2) == 0.0
    assert score.ladder_cost(truth=1, pred=3) > score.ladder_cost(truth=3, pred=1)


def test_branching_and_archetypal_are_incomparable():
    """Rungs 3 and 4 are different claims, not a distance of one — counting them as adjacent
    would reward a model for confusing a tree with a simplex."""
    assert score.incomparable(3, 4)
    assert score.ladder_cost(truth=3, pred=4) == 1.0
    assert score.ladder_cost(truth=4, pred=3) == 1.0
    # ... and a genuine 2-rung over-shoot still costs more than an incomparable swap
    assert score.ladder_cost(truth=1, pred=3) > score.ladder_cost(truth=3, pred=4)


def test_predicted_rung_is_the_highest_offer():
    """`narrate.offer` is set-valued; the claim it makes is its strongest member."""
    assert score.predicted_rung(["embed"]) == 0
    assert score.predicted_rung(["cflows", "embed"]) == 2
    assert score.predicted_rung([]) == 0
    assert score.predicted_rung(["nonsense"]) == 0


def test_unroutable_rungs_names_the_action_space_gap():
    """No recipe claims multi-branching, so a perfect classifier is still capped. Pinned so the
    gap is reported rather than hidden inside a mean.

    `[3]`, not `[3, 4]`, since `archetypes` declared `claims: archetypal` — and the caveat that
    goes with that is worth carrying: this reads what recipes CLAIM, not what they can RUN.
    `archetypes` is `blocked:` upstream and never reaches the ledger, so rung 4 is closed on
    paper and unreachable in fact. `tree_narrow`/`tree_wide` are the datasets rung 3 leaves with
    no correct answer available at all."""
    assert score.unroutable_rungs() == [3]


def test_score_offer_is_multilabel():
    got = score.score_offer(["clusters", "multi-branching"], ["clusters"])
    assert got["recall"] == pytest.approx(0.5)
    assert got["precision"] == pytest.approx(1.0)
    assert got["missed"] == ["multi-branching"]
    assert got["spurious"] == []


def test_summarize_separates_under_from_over():
    """One mean would hide the asymmetry the ladder exists to express."""
    recs = [
        {"truth_rung": 0, "pred_rung": 2},   # over  by 2 → 4.0
        {"truth_rung": 3, "pred_rung": 2},   # under by 1 → 1.0
        {"truth_rung": 2, "pred_rung": 2},   # exact     → 0.0
        {"truth_rung": None, "pred_rung": 2},  # excluded
    ]
    s = score.summarize(recs)
    assert s["n"] == 3
    assert s["over"] == 1 and s["under"] == 1 and s["exact"] == 1
    assert s["mean_cost"] == pytest.approx(5.0 / 3)


def test_every_topology_in_the_registry_is_scoreable():
    """Guards the real failure: a dataset labelled with a valid topology the ladder forgot."""
    for t in TOPOLOGIES:
        assert score.truth_rung([t]) is not None
