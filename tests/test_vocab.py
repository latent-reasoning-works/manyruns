"""The shared vocabulary (manyruns.vocab) — the loader must not out-guess the analysis selector.

These two used to disagree: `app` excluded `sample`/`batch` from the time vocabulary as
ambiguous, while `pipeline` included them *and* `condition`, and lacked `week`/`pseudotime`.
The consequence was not cosmetic — a dataset whose only obs column was `sample` got routed
to `contrast` (correctly: no time axis) and then had its batch IDs handed to MIOFlow as
evenly-spaced timepoints. A trajectory invented out of batch structure is exactly what the
`contrast` recipe exists to prevent.
"""
from __future__ import annotations

import pytest

from manyruns import vocab as obs_keys


class _FakeAdata:
    """Just enough AnnData for the detectors (they only read `.obs.columns`)."""

    class _Obs:
        def __init__(self, cols):
            self.columns = cols

    def __init__(self, *cols):
        self.obs = self._Obs(list(cols))


# ── the vocabulary itself ────────────────────────────────────────────────────
def test_time_and_condition_vocabularies_are_disjoint():
    """`condition` sat in the *time* candidates and was checked first — so condition
    labels were consumed as a time axis."""
    assert set(obs_keys.TIME_KEYS).isdisjoint(obs_keys.CONDITION_KEYS)


def test_ambiguous_batch_columns_are_not_time():
    assert set(obs_keys.AMBIGUOUS_TIME_KEYS).isdisjoint(obs_keys.TIME_KEYS)
    assert obs_keys.find_time_key(["sample"]) is None
    assert obs_keys.find_time_key(["batch"]) is None
    assert obs_keys.has_time_axis(["sample", "batch"]) is False


def test_real_time_columns_are_found_case_insensitively():
    assert obs_keys.find_time_key(["Timepoint"]) == "Timepoint"   # original spelling back
    assert obs_keys.find_time_key(["WEEK"]) == "WEEK"             # was missing from the fork
    assert obs_keys.find_time_key(["pseudotime"]) == "pseudotime"  # likewise


def test_condition_columns_are_found():
    assert obs_keys.find_condition_key(["disease"]) == "disease"
    assert obs_keys.find_condition_key(["Diagnosis"]) == "Diagnosis"
    assert obs_keys.find_condition_key(["sample"]) is None


# ── the loader now agrees with the selector ──────────────────────────────────
def test_loader_does_not_treat_sample_as_a_time_axis():
    """THE bug: obs has only `sample`, so there is no time axis. The loader must agree —
    otherwise MIOFlow receives batch IDs as timepoints and draws a trajectory from them."""
    from manyruns.pipeline import _detect_time_key

    assert _detect_time_key(_FakeAdata("sample")) is None
    assert _detect_time_key(_FakeAdata("batch", "sample")) is None


def test_loader_does_not_consume_the_condition_label_as_time():
    from manyruns.pipeline import _detect_time_key

    assert _detect_time_key(_FakeAdata("condition")) is None
    assert _detect_time_key(_FakeAdata("disease")) is None


def test_loader_finds_the_time_columns_the_selector_promises():
    """The mirror failure: the selector promised a time course on `week`/`pseudotime` but
    the loader returned None, so MIOFlow silently fell back to computed pseudotime."""
    from manyruns.pipeline import _detect_time_key

    for col in ("timepoint", "day", "week", "pseudotime", "stage"):
        assert _detect_time_key(_FakeAdata(col)) == col


def test_selector_and_loader_agree_on_every_vocabulary_word():
    """One vocabulary, one answer — asserted over the whole word list, not a sample."""
    from manyruns.pipeline import _detect_condition_key, _detect_time_key

    for col in obs_keys.TIME_KEYS:
        assert _detect_time_key(_FakeAdata(col)) == col
        assert obs_keys.has_time_axis([col]) is True
    for col in obs_keys.CONDITION_KEYS:
        assert _detect_condition_key(_FakeAdata(col)) == col
        assert obs_keys.has_time_axis([col]) is False  # a condition is never a time axis


# ── the vocabulary version and the topology mappings ─────────────────────────
def test_vocab_version_is_pinned_to_topologies():
    """`decisions.append` hashes `VERSION` into `legal_id`, so an edit to the topologies that
    keeps the old version would give a new task space the id of the old one."""
    assert (obs_keys.VERSION, obs_keys.TOPOLOGIES) == (
        "1", ("clusters", "single-trajectory", "multi-branching", "archetypal", "cycle",
              "surface")), "TOPOLOGIES changed: bump vocab.VERSION and update this pin"


def test_scshapebench_table():
    """The benchmark's four label columns, each to one canonical name."""
    assert obs_keys.SCSHAPEBENCH_TOPOLOGY == {
        "multi_branch": "multi-branching", "simple_traj": "single-trajectory",
        "archetypal": "archetypal", "clusters": "clusters"}
    assert obs_keys.scshapebench_to_topology("multi_branch") == "multi-branching"
    assert obs_keys.check_topology(list(obs_keys.SCSHAPEBENCH_TOPOLOGY.values())) == []


def test_scshapebench_refuses_unknown():
    """`cycle` is a canonical name, not a benchmark column."""
    with pytest.raises(ValueError, match="cycle"):
        obs_keys.scshapebench_to_topology("cycle")


def test_labeler_table():
    """The harness labeler's five words; `other` claims no topology."""
    assert obs_keys.LABELER_TOPOLOGY == {
        "cluster": "clusters", "trajectory": "single-trajectory", "cycle": "cycle",
        "surface": "surface", "other": None}
    assert obs_keys.labeler_to_topology("trajectory") == "single-trajectory"
    assert obs_keys.labeler_to_topology("other") is None
    for value in obs_keys.LABELER_TOPOLOGY.values():
        if value is not None:
            assert obs_keys.check_topology([value]) == []


def test_labeler_refuses_canonical_names():
    """The two vocabularies differ by one letter; the plural is not a labeler word."""
    with pytest.raises(ValueError, match="clusters"):
        obs_keys.labeler_to_topology("clusters")


def test_collapse_table():
    """scShapeBench's own scoring folds a cycle into a trajectory and a surface into a simplex."""
    assert obs_keys.collapse_for_benchmark("cycle") == "single-trajectory"
    assert obs_keys.collapse_for_benchmark("surface") == "archetypal"
    for name in ("clusters", "single-trajectory", "multi-branching", "archetypal"):
        assert obs_keys.collapse_for_benchmark(name) == name


def test_collapse_range_is_the_four_benchmark_classes():
    results = {obs_keys.collapse_for_benchmark(t) for t in obs_keys.TOPOLOGIES}
    assert results == set(obs_keys.SCSHAPEBENCH_TOPOLOGY.values()) == {
        "clusters", "single-trajectory", "multi-branching", "archetypal"}


def test_collapse_refuses_unknown():
    with pytest.raises(ValueError, match="tree"):
        obs_keys.collapse_for_benchmark("tree")
