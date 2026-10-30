"""The rung-support gate: a recipe's claim is refused when the measured geometry does not
support it.

`vocab.unmet` is the structural half of the shield: a recipe whose steps need a fact the data
does not carry is illegal. `vocab.rung_support` is the evidence half: a recipe whose CLAIM the
measured g-vector does not support is refused. A refusal reason is a fixed code string
`<key>:<code>`; it never holds a measured value, a threshold or a note, because the reasons
reach surfaces that a caller who must not see the geometry reads.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from manyruns import admit, narrate, vocab  # noqa: E402
from manyruns.narrate import Observation  # noqa: E402
from manyruns.tui import state  # noqa: E402
from manyruns.vocab import rung_support  # noqa: E402

BRANCH_KEY = "reeb_graph.structural_summary.n_branch_points"


class MeasurementUnavailable(ValueError):
    """Same name as manylatents' exception, so the gate's by-name test is exercised."""


def _obs(shape: str = "single", **kw) -> Observation:
    return Observation(shape=shape, modality=kw.pop("modality", "scrna"), **kw)


# ── the rule table ───────────────────────────────────────────────────────────


def test_the_table_covers_three_claims():
    assert set(vocab.RUNG_EVIDENCE) == {"clusters", "multi-branching", "single-trajectory"}
    assert set(vocab.RUNG_EVIDENCE) <= set(vocab.TOPOLOGIES)
    assert vocab.RUNG_EVIDENCE["clusters"] == ("betti_0", "betti_0")
    assert vocab.RUNG_EVIDENCE["multi-branching"] == (BRANCH_KEY, "n_branch_points")
    assert vocab.RUNG_EVIDENCE["single-trajectory"] == ("lid", "lid")
    assert vocab.LID_1D_BAND == (0.5, 1.5)
    assert vocab.ADMITTED_READOUTS == frozenset()


def test_the_reason_set_is_closed():
    assert vocab.SUPPORT_CODES == ("not_admitted", "not_measured", "fails_rule")
    assert vocab.SUPPORT_REASONS == frozenset(
        f"{key}:{code}" for key, _ in vocab.RUNG_EVIDENCE.values() for code in vocab.SUPPORT_CODES)
    assert len(vocab.SUPPORT_REASONS) == 9
    assert "betti_0:fails_rule" in vocab.SUPPORT_REASONS
    assert f"{BRANCH_KEY}:not_admitted" in vocab.SUPPORT_REASONS
    assert not any(ch.isdigit() for code in vocab.SUPPORT_CODES for ch in code)


def test_vocab_stays_import_free():
    path = Path(vocab.__file__)
    assert path.parent.is_dir()
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add((node.module or "").split(".")[0])
    assert names == {"__future__", "re", "typing"}


# ── the rule check ───────────────────────────────────────────────────────────

#: Every (recipe, evidence, admitted) input used in this file, for the closed-set scenario.
CALLS: list = []


def _rs(recipe, evidence, admitted):
    CALLS.append((recipe, evidence, admitted))
    return rung_support(recipe, evidence, admitted)


def test_two_components_support_clusters():
    assert _rs({"claims": "clusters"}, {"betti_0": 3}, {"betti_0"}) == frozenset()


def test_one_component_refuses_clusters():
    assert _rs({"claims": "clusters"}, {"betti_0": 1}, {"betti_0"}) == frozenset(
        {"betti_0:fails_rule"})


def test_a_branch_point_supports_multi_branching():
    claim = {"claims": "multi-branching"}
    assert _rs(claim, {BRANCH_KEY: 1}, {"n_branch_points"}) == frozenset()
    assert _rs(claim, {BRANCH_KEY: 0}, {"n_branch_points"}) == frozenset(
        {f"{BRANCH_KEY}:fails_rule"})


def test_the_lid_band_bounds_a_trajectory():
    claim = {"claims": "single-trajectory"}
    for v in (1.0, 0.5, 1.5):
        assert _rs(claim, {"lid": v}, {"lid"}) == frozenset(), v
    for v in (2.1, 0.3):
        assert _rs(claim, {"lid": v}, {"lid"}) == frozenset({"lid:fails_rule"}), v


def test_a_reason_does_not_depend_on_the_measured_value():
    reasons = set()
    for v in (1, 0, -7, 1.0):
        got = _rs({"claims": "clusters"}, {"betti_0": v}, {"betti_0"})
        assert got == frozenset({"betti_0:fails_rule"}), v
        reasons |= got
    for v in (0.3, 2.1, 17.25):
        got = _rs({"claims": "single-trajectory"}, {"lid": v}, {"lid"})
        assert got == frozenset({"lid:fails_rule"}), v
        reasons |= got
    for r in reasons:
        assert not any(ch.isdigit() for ch in r.rsplit(":", 1)[1]), r


def test_a_numpy_scalar_is_a_number():
    assert _rs({"claims": "clusters"}, {"betti_0": np.int64(3)}, {"betti_0"}) == frozenset()
    assert _rs({"claims": "single-trajectory"}, {"lid": np.float32(1.0)}, {"lid"}) == frozenset()


def test_a_claim_with_no_rule_is_never_refused():
    for c in ("surface", "cycle", "archetypal"):
        assert _rs({"claims": c}, {}, ()) == frozenset(), c
    assert _rs({}, {}, ()) == frozenset()


# ── an unavailable measurement refuses rather than passes ────────────────────


def test_a_declined_suite_metric_refuses():
    got = _rs({"claims": "clusters"},
              {"betti_0": None, "betti_0_note": "MeasurementUnavailable: only 3 distinct points"},
              {"betti_0"})
    assert got == frozenset({"betti_0:not_measured"})
    assert not any("only 3 distinct points" in r for r in got)


def test_the_engine_exception_refuses():
    claim = {"claims": "single-trajectory"}
    assert _rs(claim, {"lid": MeasurementUnavailable("k too large")}, {"lid"}) == frozenset(
        {"lid:not_measured"})


def test_the_real_engine_exception_refuses():
    exceptions = pytest.importorskip("manylatents.utils.exceptions")
    claim = {"claims": "single-trajectory"}
    value = exceptions.MeasurementUnavailable("k too large")
    assert _rs(claim, {"lid": value}, {"lid"}) == frozenset({"lid:not_measured"})


def test_the_serialised_form_refuses():
    got = _rs({"claims": "clusters"},
              {"betti_0": {"status": "unavailable", "reason": "no reference"}}, {"betti_0"})
    assert got == frozenset({"betti_0:not_measured"})


def test_nan_refuses():
    for v in (float("nan"), float("inf"), np.float64("nan")):
        assert _rs({"claims": "clusters"}, {"betti_0": v}, {"betti_0"}) == frozenset(
            {"betti_0:not_measured"}), v


def test_an_array_value_refuses_and_is_not_echoed():
    got = _rs({"claims": "clusters"}, {"betti_0": np.array([4.0, 5.0])}, {"betti_0"})
    assert got == frozenset({"betti_0:not_measured"})


def test_absence_a_bool_and_a_string_refuse():
    claim = {"claims": "clusters"}
    for evidence in ({}, {"betti_0": True}, {"betti_0": "3"}, {"betti_0": np.zeros(3)}):
        assert _rs(claim, evidence, {"betti_0"}) == frozenset({"betti_0:not_measured"}), evidence


# ── an unadmitted metric cannot open a rung ──────────────────────────────────


def test_a_passing_but_unadmitted_value_refuses():
    assert _rs({"claims": "clusters"}, {"betti_0": 4}, set()) == frozenset(
        {"betti_0:not_admitted"})


def test_the_admitted_set_comes_from_the_audits_gate():
    verdicts = [admit.Verdict("betti_0", "rejected"), admit.Verdict("lid", "flagged")]
    admitted = {v.name for v in verdicts if v.ok}
    assert _rs({"claims": "clusters"}, {"betti_0": 4}, admitted) == frozenset(
        {"betti_0:not_admitted"})
    assert _rs({"claims": "single-trajectory"}, {"lid": 1.0}, admitted) == frozenset()


def test_not_admitted_comes_before_not_measured():
    assert _rs({"claims": "clusters"}, {}, ()) == frozenset({"betti_0:not_admitted"})


def test_every_reason_is_in_the_closed_set():
    """Runs last in file order, over every input the scenarios above used."""
    assert CALLS, "no scenario recorded its inputs"
    for recipe, evidence, admitted in CALLS:
        for reason in rung_support(recipe, evidence, admitted):
            assert reason in vocab.SUPPORT_REASONS, reason


# ── the ledger and the offer apply the gate when evidence is given ───────────


def test_unsupported_clusters_is_hollow_with_its_reason():
    obs = _obs("clusters", modality="synthetic")
    rows = {r.recipe: r for r in state.ledger(obs, evidence={"betti_0": 1},
                                              admitted={"betti_0"})}
    row = rows["cluster"]
    assert row.can_run is False
    assert "betti_0:fails_rule" in row.blocked
    assert "betti_0:fails_rule" in row.as_offer()["blocked"]
    offered = {o["recipe"]: o for o in narrate.offer(obs, evidence={"betti_0": 1},
                                                     admitted={"betti_0"})}
    assert offered["cluster"]["recommended"] is False


def test_no_evidence_changes_nothing():
    obs = _obs("clusters", modality="synthetic")
    assert ([r.to_dict() for r in state.ledger(obs)]
            == [r.to_dict() for r in state.ledger(obs, evidence=None, admitted={"betti_0"})])
    assert narrate.offer(obs) == narrate.offer(obs, evidence=None, admitted={"betti_0"})


def test_recipes_without_a_rule_keep_their_legality_under_evidence():
    """The spec's scenario passes `admitted=()`, which by the rule order yields `not_admitted`,
    not `not_measured`. The metrics are admitted here so the scenario tests what it names: an
    empty g-vector refuses every ruled claim as not measured."""
    from manyruns.catalog import load_recipe

    obs = _obs("manifold", modality="synthetic")
    before = {r.recipe: r for r in state.ledger(obs)}
    after = {r.recipe: r for r in state.ledger(obs, evidence={}, admitted={"betti_0", "lid"})}
    for name, row in before.items():
        if load_recipe(name).get("claims") not in vocab.RUNG_EVIDENCE:
            assert after[name].can_run == row.can_run, name
    assert before["embed"].can_run is True and after["embed"].can_run is True
    assert "betti_0:not_measured" in after["cluster"].blocked
    assert "lid:not_measured" in after["pseudotime"].blocked


def test_the_reasons_follow_the_data_reasons():
    """The gate's reasons are appended after `narrate.blocked` and `narrate.unavailable`."""
    from manyruns.catalog import load_recipe

    obs = _obs("manifold", modality="synthetic")
    rows = {r.recipe: r for r in state.ledger(obs, evidence={}, admitted=())}
    for name, row in rows.items():
        recipe = load_recipe(name)
        head = narrate.blocked(recipe, obs.shape) + narrate.unavailable(recipe)
        assert list(row.blocked) == head + sorted(rung_support(recipe, {}, ())), name


def test_the_two_surfaces_agree_under_evidence():
    """A ledger row that cannot run is never recommended by the offer, with evidence too."""
    for shape in vocab.SHAPES:
        obs = _obs(shape, modality="synthetic")
        for evidence, admitted in (({}, ()), ({"betti_0": 1, "lid": 2.1}, {"betti_0", "lid"}),
                                   ({"betti_0": 3, "lid": 1.0}, {"betti_0", "lid"})):
            offered = {o["recipe"]: o for o in narrate.offer(obs, evidence=evidence,
                                                             admitted=admitted)}
            for row in state.ledger(obs, evidence=evidence, admitted=admitted):
                if not row.can_run:
                    assert not offered[row.recipe]["recommended"], (shape, row.recipe)
