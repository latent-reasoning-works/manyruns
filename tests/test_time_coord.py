"""Time is a magnitude-preserving per-cell coordinate, resolved from ONE vocabulary (fixes #27).

Two invariants:
  1. Real timepoints keep their real spacing — `day0/day3/day9` become 0, ⅓, 1, not the even
     0, ½, 1 that ranks give. Collapsing spacing to ranks fed MIOFlow the wrong clock.
  2. The recipe selector and the label loader resolve time/condition through the SAME `vocab`
     lists, so they can never classify one column into two roles (the C1 bug: `init` says
     "conditions, no time" while `run` feeds that column to MIOFlow as time).
"""
from __future__ import annotations

import pytest


# ── magnitude preservation ───────────────────────────────────────────────────
def test_real_timepoints_keep_real_spacing():
    pytest.importorskip("numpy")
    from manyruns import vocab

    t = vocab.numeric_time(["day0", "day3", "day9"])
    assert t[0] == pytest.approx(0.0)
    assert t[2] == pytest.approx(1.0)
    assert t[1] == pytest.approx(1 / 3, abs=1e-5)   # 3/9 — NOT the even-rank 0.5


def test_bare_numbers_are_timepoints_too():
    pytest.importorskip("numpy")
    from manyruns import vocab

    assert list(vocab.numeric_time([0, 3, 9])) == pytest.approx([0.0, 1 / 3, 1.0], abs=1e-5)


def test_categories_fall_back_to_even_ranks():
    pytest.importorskip("numpy")
    from manyruns import vocab

    t = vocab.numeric_time(["early", "mid", "late"])   # non-numeric → no real magnitude
    assert set(round(float(x), 3) for x in t) == {0.0, 0.5, 1.0}   # evenly spaced


def test_discretize_pseudotime_bins_into_the_requested_count():
    np = pytest.importorskip("numpy")
    from manyruns import vocab

    pt = np.linspace(0.0, 1.0, 100)
    bins = vocab.discretize_pseudotime(pt, 5)
    assert set(bins.tolist()) == {0, 1, 2, 3, 4}


def test_discretize_pseudotime_endpoints_land_in_first_and_last_bin():
    """`pt == 1.0` must clip into the LAST bin (index `n_timepoints - 1`), not overflow into a
    phantom `n_timepoints`-th bin — the off-by-one `np.clip` in the shared helper guards."""
    np = pytest.importorskip("numpy")
    from manyruns import vocab

    bins = vocab.discretize_pseudotime(np.array([0.0, 1.0]), 5)
    assert list(bins) == [0, 4]


def test_looks_like_timepoint_and_parse():
    from manyruns import vocab

    assert vocab.looks_like_timepoint("day3") and vocab.looks_like_timepoint("t0")
    assert vocab.looks_like_timepoint("3") and vocab.looks_like_timepoint("week1")
    assert not vocab.looks_like_timepoint("donorA")
    assert vocab.parse_timepoints(["day0", "day3"]) == [0.0, 3.0]
    assert vocab.parse_timepoints(["donorA", "donorB"]) is None   # any non-numeric → None


# ── one vocabulary for both sides ─────────────────────────────────────────────
def test_selector_and_loader_share_one_time_vocabulary():
    from manyruns import app, vocab

    # the selector's constants ARE vocab's, not a drifting second copy
    assert app._TIME_KEYS is vocab.TIME_KEYS
    assert app._CONDITION_KEYS is vocab.CONDITION_KEYS


def test_a_condition_column_is_never_time_and_carriers_stay_conservative():
    from manyruns import vocab

    assert vocab.find_time_key(["disease"]) is None          # selector: not a trajectory
    assert vocab.find_condition_key(["disease"]) == "disease"
    assert vocab.find_time_key(["day"]) == "day"             # a real time column is time
    # main's deliberate conservatism: sample/batch are NOT auto-detected as time
    assert vocab.find_time_key(["sample"]) is None
    assert vocab.find_time_key(["batch"]) is None
