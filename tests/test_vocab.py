"""The shared vocabulary (manyruns.vocab) — the loader must not out-guess the analysis selector.

These two used to disagree: `app` excluded `sample`/`batch` from the time vocabulary as
ambiguous, while `pipeline` included them *and* `condition`, and lacked `week`/`pseudotime`.
The consequence was not cosmetic — a dataset whose only obs column was `sample` got routed
to `contrast` (correctly: no time axis) and then had its batch IDs handed to MIOFlow as
evenly-spaced timepoints. A trajectory invented out of batch structure is exactly what the
`contrast` recipe exists to prevent.
"""
from __future__ import annotations

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
