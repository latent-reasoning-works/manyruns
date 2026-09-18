"""The empirical null — the gate between "we measured something" and "we found something".

These tests pin the property the audit asked for: a readout is narrated as a FINDING only
when it beats its own permutation null. The magnitudes here are deliberately extreme in the
null cases, because that is the failure being prevented — the old code printed
"~48× more abundant" about sixty random points.
"""
from __future__ import annotations

import numpy as np
import pytest

from manyruns import vocab
from manyruns.narrate import narrate
from manyruns.pipeline import null as _null
from manyruns.pipeline.steps import _step_composition, _step_separation


def _separated(n=400, d=8, seed=0, gap=6.0):
    """Two conditions that genuinely occupy different regions."""
    rng = np.random.default_rng(seed)
    Z = np.vstack([rng.normal(0, 1, (n // 2, d)), rng.normal(0, 1, (n // 2, d)) + gap])
    return Z, np.array(["case"] * (n // 2) + ["control"] * (n // 2))


def _noise(n=400, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.permutation(
        np.array(["case"] * (n // 2) + ["control"] * (n // 2)))


def _run(Z, labels, params=None):
    g: dict = {}
    # `label_kind` because both steps take an ALLOWLIST on it: only a declared
    # condition axis is scored (`steps.py`), so a bare label array is refused.
    state = {"emb": Z, "labels": labels, "label_kind": "condition"}
    _step_separation(state, g, params or {}, None, False, 2)
    _step_composition(state, g, params or {}, None, False, 2)
    return g


# ── the harness itself ───────────────────────────────────────────────────────
def test_p_is_bounded_and_has_the_expected_floor():
    """An empirical p can never be 0 — the observed value counts as one of its own draws."""
    res = _null.rank_against_null(lambda lab: float(np.sum(np.asarray(lab) == "a")),
                                  np.array(["a", "b"] * 20), reps=99)
    assert 0.0 < res["p_emp"] <= 1.0
    assert res["p_emp"] >= 1.0 / (res["reps"] + 1)
    assert res["reps"] == 99


def test_a_statistic_that_cannot_run_yields_no_null():
    """A null for a statistic that did not produce a value is meaningless, not zero."""
    assert _null.rank_against_null(lambda lab: None, np.array(["a", "b"]), reps=9) is None


def test_record_writes_flat_scalars_only():
    """The g-vector contract is dict[str, float]; a nested dict is dropped by the table."""
    g: dict = {}
    _null.record(g, "thing", {"real": 1.0, "null_mean": 0.5, "null_sd": 0.1,
                              "null_p95": 0.9, "reps": 99, "rank": 98, "p_emp": 0.01})
    assert g == {"thing__null_p": 0.01, "thing__null_mean": 0.5, "thing__null_reps": 99}
    assert all(isinstance(v, (int, float)) for v in g.values())


def test_record_of_nothing_writes_nothing():
    g: dict = {}
    _null.record(g, "thing", None)
    assert g == {}


# ── the readouts ─────────────────────────────────────────────────────────────
def test_real_separation_beats_its_null():
    g = _run(*_separated())
    assert g["separation_silhouette"] > 0.5
    assert g["separation_silhouette__null_p"] <= 0.05


def test_noise_separation_does_not_beat_its_null():
    g = _run(*_noise())
    assert g["separation_silhouette__null_p"] > 0.05


def test_real_composition_shift_beats_its_null():
    g = _run(*_separated())
    assert g["composition_max_log2_shift__null_p"] <= 0.05


def test_noise_composition_does_not_beat_its_null_even_when_the_magnitude_is_large():
    """The regression that matters: a big number on noise must not read as a finding.

    Small n with k forced high is the audit's failure configuration — the statistic is a
    maximum over k clusters × condition pairs, so it is large by construction there.
    """
    Z, labels = _noise(n=60, d=8, seed=3)
    g = _run(Z, labels, params={"k": 8})
    assert g["composition_max_log2_shift"] > 1.0, "expected an inflated magnitude at n=60/k=8"
    assert g["composition_max_log2_shift__null_p"] > 0.05


# ── the sentences ────────────────────────────────────────────────────────────
def test_noise_is_narrated_as_a_non_finding():
    """'We looked and found nothing' — not silence, and not a confident claim."""
    text = narrate({"g_vector": _run(*_noise())})
    assert "more abundant" not in text
    assert "do not separate beyond chance" in text
    assert "No population shifts more than chance" in text


def test_a_real_effect_is_still_narrated_as_a_finding():
    text = narrate({"g_vector": _run(*_separated())})
    assert "The groups separate" in text
    assert "biggest driver" in text


def test_a_real_effect_reports_its_rank():
    """A magnitude without a rank is what the audit called out; it must not come back."""
    text = narrate({"g_vector": _run(*_separated())})
    assert "shuffled labels beat it" in text


def test_an_unranked_magnitude_says_so():
    """A g-vector with no null (e.g. the mock) must not borrow the ranked language."""
    text = narrate({"g_vector": {"composition_max_log2_shift": 8.76,
                                 "composition_between": "case vs control"}})
    assert "unranked" in text


def test_an_absent_population_is_not_quoted_as_a_fold_ratio():
    """When the cluster is empty in one condition the ratio is a bound set by n, not a
    measured ratio — the old text printed it as '~434× more abundant' regardless."""
    g = _run(*_separated())
    assert g["composition_absent_in_one"] is True
    text = narrate({"g_vector": g})
    assert "essentially absent" in text
    assert "× more abundant" not in text


@pytest.mark.parametrize("n", [200, 400, 800])
def test_magnitude_is_bounded_by_the_cohort_not_by_a_constant(n):
    """The old form added 1e-3 to FRACTIONS, so sweeping that constant moved the narrated
    claim across 4×/32×/314×/3134× on untouched data — the number was set by the knob.

    In count space the extreme case is a cluster holding none of one condition and all of
    the other: (n_b + 0.5)/0.5 = 2·n_b + 1, i.e. log2(n + 1) for balanced groups. The bound
    is now set by how many cells were counted, which is the actual evidence — zero out of
    400 is stronger evidence of depletion than zero out of 40, and the fraction form scored
    those two identically.
    """
    g = _run(*_separated(n=n))
    assert g["composition_max_log2_shift"] <= np.log2(n + 1) + 1e-6


# ── the vocabulary ───────────────────────────────────────────────────────────
def test_every_readout_with_a_null_kind_declares_a_known_kind():
    assert set(vocab.NULL_KIND.values()) <= set(vocab.NULL_KINDS)


def test_label_permuting_readouts_are_exactly_those_that_read_labels():
    """A step absent from NULL_KIND gets no null — and must not be narrated as a finding.
    This is the granger lesson encoded: it never read a label, so a label-permutation null
    would have certified it while it returned p < 1e-30 on noise."""
    assert vocab.NULL_KIND == {"separation": "labels", "composition": "labels"}
    assert set(vocab.NULL_KIND) <= set(vocab.STEP_NEEDS), (
        "a label-permutation null is only valid for a step that requires conditions")
