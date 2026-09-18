"""The declared metric suite (`pipeline.suite`) — the shared geometric core of the g-vector.

The property under test is the one the whole thing exists for: the KEY SET is a property of
the suite, not of what happened to work. Two runs of different recipes must be comparable.
"""
import pytest

np = pytest.importorskip("numpy")

from manyruns.pipeline import suite  # noqa: E402

DECLARED = ["lid", "anisotropy", "betti_0"]


def _emb(n=40, d=3):
    return np.asarray(np.random.default_rng(0).normal(size=(n, d)))


def test_every_declared_name_appears_even_when_nothing_can_be_measured(monkeypatch):
    """No engine → still one key per declared metric, each with a reason.

    Omitting the key instead would make "the engine is missing" indistinguishable from "this
    recipe does not emit that metric" — the disagreement the audit found across four g-vector
    constructors, and the reason a cross-recipe comparison silently returned nothing."""
    monkeypatch.setattr(suite, "_compute_metric", lambda: None)

    g = suite.measure(_emb(), _emb(40, 10), DECLARED)

    assert [g[n] for n in DECLARED] == [None, None, None]
    assert all("manylatents" in g[f"{n}_note"] for n in DECLARED)
    assert g["suite_measured"] == 0 and g["suite_declared"] == 3


def test_no_embedding_is_a_stated_absence_not_an_empty_result(monkeypatch):
    monkeypatch.setattr(suite, "_compute_metric", lambda: (lambda *a, **k: 1.0))

    g = suite.measure(None, _emb(40, 10), DECLARED)

    assert set(DECLARED) <= set(g)
    assert g["lid_note"] == "no embedding to measure"
    assert g["suite_measured"] == 0


def test_nan_is_recorded_as_absent_rather_than_as_a_number(monkeypatch):
    """Several manylatents metrics return NaN to mean "I could not do this" —
    `geodesic_distance_correlation` does it whenever the dataset exposes no ground truth,
    which is every manyruns dataset. NaN in a comparison table is a confident-looking
    non-number; an explicit absence is not."""
    monkeypatch.setattr(suite, "_compute_metric",
                        lambda: (lambda name, **k: float("nan") if name == "lid" else 0.5))

    g = suite.measure(_emb(), _emb(40, 10), DECLARED)

    assert g["lid"] is None and "nan" in g["lid_note"].lower()
    assert g["anisotropy"] == 0.5
    assert g["suite_measured"] == 2          # the NaN does not count as measured


def test_one_metric_raising_does_not_lose_the_others(monkeypatch):
    def flaky(name, **kwargs):
        if name == "betti_0":
            raise RuntimeError("ripser exploded")
        return 1.5

    monkeypatch.setattr(suite, "_compute_metric", lambda: flaky)

    g = suite.measure(_emb(), _emb(40, 10), DECLARED)

    assert g["betti_0"] is None
    assert "RuntimeError" in g["betti_0_note"] and "ripser exploded" in g["betti_0_note"]
    assert g["lid"] == 1.5 and g["anisotropy"] == 1.5
    assert g["suite_measured"] == 2


def test_a_value_the_engine_already_produced_is_never_recomputed(monkeypatch):
    """`engine=manylatents` computes the suite itself. Recomputing here would waste the work
    and, worse, risk two different numbers for one metric name in one g-vector."""
    calls = []

    def spy(name, **kwargs):
        calls.append(name)
        return 9.9

    monkeypatch.setattr(suite, "_compute_metric", lambda: spy)

    g = suite.measure(_emb(), _emb(40, 10), DECLARED, already={"lid": 0.25})

    assert "lid" not in g                     # left alone entirely, not overwritten
    assert calls == ["anisotropy", "betti_0"]
    assert g["suite_measured"] == 3           # the engine's value still counts as measured


def test_suite_values_are_marked_comparable_but_not_findings(monkeypatch):
    """Every suite metric is derived from the embedding alone, so `vocab.NULL_KIND` says each
    needs a `data` null and nothing implements one. The measured reason for that
    rule: an embedding-derived statistic scored p < 1e-30 on pure noise in 12/12
    seeds, and a cheap label-null would have certified it. Stated once as a key rather than
    thirteen times as a note."""
    monkeypatch.setattr(suite, "_compute_metric", lambda: (lambda *a, **k: 1.0))

    g = suite.measure(_emb(), _emb(40, 10), DECLARED)

    assert g["suite_null"] == suite.NULL_STATUS
    assert "not a finding" in suite.NULL_STATUS
    # no `__null_p` is emitted for any of them — the key `narrate` gates a finding on
    assert not [k for k in g if k.endswith("__null_p")]


# ── the per-step live view (`suite.live`) ────────────────────────────────────────────────
#
# `measure` runs once, at the end, and is the record. `live` runs after every step that
# changed the embedding, so a dashboard can show geometry MOVING. Everything below pins the
# two apart: the live view is cheap, silent about what it could not do, and never the record.


def _spy(monkeypatch, value=1.0):
    """A stand-in registry that records which metrics were actually asked for."""
    calls = []

    def compute(name, embeddings=None, dataset=None):
        calls.append(name)
        return value

    monkeypatch.setattr(suite, "_compute_metric", lambda: compute)
    return calls


def test_the_live_view_measures_only_the_cheap_subset(monkeypatch):
    """THE cost regression. The full suite costs 1.665 s at 2,700 points, and 1.57 s of that
    is three metrics: trustworthiness (0.524 s, and O(n²) — 33.6 s at 20,000),
    betti_1 (0.822 s) and betti_0 (0.221 s). Paying those on every step of a 3-step recipe
    buys nothing the end-of-run measurement does not already record. `LIVE_SUBSET` is 0.066 s
    at 2,700; adding any of the three back would multiply that by 25."""
    from manyruns import catalog

    declared = catalog.load_suite()
    calls = _spy(monkeypatch)

    g = suite.live(_emb(), declared)

    assert calls == [n for n in declared if n in suite.LIVE_SUBSET]
    assert not ({"trustworthiness", "betti_0", "betti_1", "continuity", "knn_preservation"}
                & set(g)), "an expensive metric got into the per-step path"
    assert set(g) <= set(suite.LIVE_SUBSET)


def test_the_live_view_is_a_bare_mapping_with_no_notes(monkeypatch):
    """`measure`'s shape — a `<name>_note` for every absence plus four `suite_*` keys — is
    right for the record and wrong for an overlay redrawn on every step. The reasons still
    reach the g-vector, once."""
    _spy(monkeypatch)

    g = suite.live(_emb(), list(suite.LIVE_SUBSET))

    assert not [k for k in g if k.endswith("_note") or k.startswith("suite_")]
    assert all(isinstance(v, float) for v in g.values())


def test_no_engine_returns_nothing_rather_than_a_dict_of_absences(monkeypatch):
    """A live panel that printed "install manylatents" five times per step would be worse
    than a quiet one. `{}` here is what makes the caller omit the key entirely."""
    monkeypatch.setattr(suite, "_compute_metric", lambda: None)

    assert suite.live(_emb(), list(suite.LIVE_SUBSET)) == {}


def test_a_metric_that_cannot_be_measured_keeps_its_none_beside_the_others(monkeypatch):
    """One failure is per-metric, not per-view — the contract says values are floats or None,
    and the four that worked must still be shown."""
    def flaky(name, embeddings=None, dataset=None):
        if name == "lid":
            raise RuntimeError("no")
        return float("nan") if name == "anisotropy" else 2.0

    monkeypatch.setattr(suite, "_compute_metric", lambda: flaky)

    g = suite.live(_emb(), list(suite.LIVE_SUBSET))

    assert g["lid"] is None and g["anisotropy"] is None      # raised / NaN, both absent
    assert g["outlier_score"] == 2.0
    assert len(g) == len(suite.LIVE_SUBSET)


def test_the_guard_skips_a_large_embedding_without_measuring_anything(monkeypatch):
    """At 20,000 rows the subset costs 0.538 s per step — a third of what the FULL suite
    costs at 2,700 — and 1.417 s at 50,000. Past the bound the preview costs more than the
    record it previews, so it is not taken at all. The boundary is inclusive: exactly
    LIVE_MAX_POINTS still measures."""
    calls = _spy(monkeypatch)

    assert suite.live(np.zeros((suite.LIVE_MAX_POINTS + 1, 3)), list(suite.LIVE_SUBSET)) == {}
    assert calls == [], "the guard measured before deciding not to"
    assert suite.live(np.zeros((suite.LIVE_MAX_POINTS, 3)), list(suite.LIVE_SUBSET)) != {}


def test_the_live_view_never_measures_a_name_the_suite_did_not_declare(monkeypatch):
    """The suite is config (`configs/metrics/default.yaml`). A live value for a metric the
    g-vector does not carry would be a number with nothing to compare it to."""
    calls = _spy(monkeypatch)

    assert suite.live(_emb(), ["anisotropy"]) == {"anisotropy": 1.0}
    assert calls == ["anisotropy"]
    assert suite.live(_emb(), []) == {}


# ── admissible ranges (`suite.ranges` / `_admissible_note`) ──────────────────────────────
#
# The VALUE-level instance of "type-correct, plausible, and never checked". Measured on this
# repo before any of this existed: the `embed` recipe on 600×40 iid gaussian noise reported
# NOTE ON THE VEHICLE. These used `lid` with a ceiling of `final_dim`, on the reasoning that
# an intrinsic dimension cannot exceed its ambient dimension. That is true of the QUANTITY and
# false of the ESTIMATOR: measured on a filled 3-ball in R^3 (true intrinsic dimension 3) at
# final_dim=3, lid = 3.137 / 3.119 / 3.099 for seeds 0/1/2 — the bound fired on all three,
# while the genuine 2-D control passed at 2.093. lid's ceiling was removed for that reason, so
# the mechanism is exercised here with `participation_ratio`, which keeps a `final_dim`
# ceiling because it is an algebraic function of the local eigenvalues rather than a
# statistical estimator. Verified on the same 3-ball: 2.616 / 2.572 / 2.592 against a ceiling
# of 3, no flags.
# `lid` 3.419 / 3.425 / 3.419 (seeds 0/1/2) on a `final_dim=3` embedding, every step
# `outcome=ok`, run `ok=True`. An intrinsic dimension cannot exceed the ambient dimension it
# is measured in. Nothing in the record said so.


def _fixed(monkeypatch, values):
    """A stand-in registry returning a pinned value per metric name."""
    monkeypatch.setattr(suite, "_compute_metric",
                        lambda: (lambda name, **k: values.get(name, 0.5)))


def test_a_value_outside_its_declared_range_is_flagged_and_kept(monkeypatch):
    """THE motivating case, reproduced as a unit: lid 3.419 on a three-dimensional embedding.

    Flagged, not corrected — the design decision. The Levina–Bickel MLE overshoots on finite
    samples and does it worst on NOISE: called directly on raw iid gaussian it reads
    2.293/2.255/2.256 at d=2, 3.451/3.375/3.465 at d=3, 5.466/5.487/5.318 at d=5. That is an
    estimator artifact, not a corrupt reading, and it is still comparable across runs — so
    nulling it would discard a usable number to make a point about it."""
    _fixed(monkeypatch, {"participation_ratio": 3.419})

    g = suite.measure(_emb(600, 3), _emb(600, 40), ["participation_ratio"])

    assert g["participation_ratio"] == 3.419          # kept, in full, not nulled
    note = g["participation_ratio_note"]
    assert "out of range" in note and "final_dim=3" in note
    assert "value kept" in note
    assert g["suite_measured"] == 1                   # it WAS measured; that is a fact too
    assert g["suite_out_of_range"] == 1


def test_the_ceiling_is_an_expression_over_the_run_shape_not_a_constant(monkeypatch):
    """The same 3.419 is impossible in R^3 and unremarkable in R^10.

    A constant bound cannot express that, which is why the declaration resolves identifiers
    from the g-vector's own shape keys. This is the test that fails if `lid`'s ceiling is ever
    hard-coded to a number."""
    _fixed(monkeypatch, {"participation_ratio": 3.419})

    narrow = suite.measure(_emb(600, 3), _emb(600, 40), ["participation_ratio"])
    wide = suite.measure(_emb(600, 10), _emb(600, 40), ["participation_ratio"])

    assert narrow["suite_out_of_range"] == 1 and "participation_ratio_note" in narrow
    assert wide["suite_out_of_range"] == 0 and "participation_ratio_note" not in wide
    assert narrow["participation_ratio"] == wide["participation_ratio"] == 3.419


def test_a_value_inside_its_range_says_nothing_at_all(monkeypatch):
    """Silence is what makes the note a signal. A caption on every in-range metric would put
    twelve strings in every g-vector and bury the one that matters."""
    _fixed(monkeypatch, {"lid": 2.1, "anisotropy": 0.35, "betti_0": 19.0})

    g = suite.measure(_emb(600, 3), _emb(600, 40), DECLARED)

    assert not [k for k in g if k.endswith("_note")]
    assert g["suite_out_of_range"] == 0


def test_a_metric_declared_unbounded_is_never_flagged(monkeypatch):
    """`outlier_score` is the one that says "we looked and there is no bound we can defend".

    LOF is a ratio of local reachability densities with ~1 as the inlier baseline and no cap.
    A floor of 0 is derivable and could never bind, so declaring it would look like a check
    and be none. Measured 1.075 / 1.104 / 1.122 on the iid-noise runs above."""
    _fixed(monkeypatch, {"outlier_score": 1.0753})
    modest = suite.measure(_emb(600, 3), _emb(600, 40), ["outlier_score"])
    _fixed(monkeypatch, {"outlier_score": 1e6})
    absurd = suite.measure(_emb(600, 3), _emb(600, 40), ["outlier_score"])

    assert suite.ranges()["outlier_score"] == suite.UNBOUNDED
    assert modest["suite_out_of_range"] == absurd["suite_out_of_range"] == 0
    assert not [k for k in modest if k.endswith("_note")]


def test_a_bound_naming_a_shape_the_run_lacks_is_reported_unchecked(monkeypatch):
    """"The bound could not be checked" and "the value was in range" are two different facts.

    Same rule as the module's absences: a check that quietly did not happen is the failure
    class one level up. Reachable when a value arrives from the engine with no embedding
    beside it to take `final_dim` from."""
    _fixed(monkeypatch, {"participation_ratio": 3.419})

    g = suite.measure(None, None, ["participation_ratio"], already={"participation_ratio": 3.419})

    assert g["participation_ratio_note"] == "range unchecked: final_dim is not in this g-vector"
    assert g["suite_out_of_range"] == 0     # not a violation — nobody looked


def test_a_value_the_engine_produced_is_range_checked_but_never_overwritten(monkeypatch):
    """An admissible range is a property of the METRIC, not of who computed it. `lid` >
    `final_dim` is exactly as impossible on the `engine=manylatents` path, where the engine
    measures the suite itself and this module writes no value at all."""
    _fixed(monkeypatch, {"participation_ratio": 0.0})

    g = suite.measure(_emb(600, 3), _emb(600, 40), ["participation_ratio"], already={"participation_ratio": 9.9})

    assert "participation_ratio" not in g   # the engine's value is left exactly where it was
    assert "out of range" in g["participation_ratio_note"] and "9.9" in g["participation_ratio_note"]
    assert g["suite_measured"] == 1 and g["suite_out_of_range"] == 1


def test_every_metric_in_the_shipped_suite_has_an_admissible_range():
    """The anti-drift guard, and the reason `ranges()` is allowed to be lenient.

    A bound that does not parse is DROPPED, so without this a typo in `default.yaml` would
    disable a check and produce a g-vector identical to one where nothing was out of range —
    the same silent-pass class the ranges exist to catch, one level up."""
    from manyruns import catalog

    declared = catalog.load_suite()

    assert declared, "the shipped suite declares no metrics"
    assert suite.check_ranges(declared) == []
    assert set(suite.ranges()) == set(declared)


def test_every_declared_bound_is_a_literal_or_a_shape_name():
    """Pins the grammar's whole footprint, which is what justifies it having no operators.

    Named in `_resolve`'s docstring as the test that fails first when a bound needs
    arithmetic — at which point that one function grows and this assertion is the notice."""
    for metric, spec in suite.ranges().items():
        if spec == suite.UNBOUNDED:
            continue
        for token in spec:
            assert token is None or isinstance(token, (int, float)) \
                or token in suite.SHAPE_VARS, f"{metric}: {token!r} is not in the grammar"


def test_the_grammar_evaluates_nothing_it_does_not_recognise():
    """No `eval`, no `ast`, no import by string — the accepted language is a numeric literal
    or one of four names. Anything shaped like an expression is not computed, it is refused,
    so a hand-edited config file cannot make this process do arbitrary work."""
    env = {"final_dim": 3, "n_embedded": 600}

    assert suite._resolve("final_dim", env) == 3.0
    assert suite._resolve(3, env) == 3.0
    for hostile in ("final_dim * 2", "__import__('os').getcwd()", "1+1", "n_embedded ",
                    "FINAL_DIM", True, None, [3], "os.getcwd()"):
        assert suite._resolve(hostile, env) is None, f"{hostile!r} resolved"


def test_a_bound_that_does_not_parse_is_dropped_and_named(tmp_path, monkeypatch):
    """Lenient at load, loud at check. A malformed bound must not fail the run it is a check
    on (`runner._live_suite`'s rule), but it must not vanish either."""
    (tmp_path / "wonky.yaml").write_text(
        "metrics: [lid, anisotropy, betti_0]\n"
        "ranges:\n"
        "  lid: {in: [0, final_dim]}\n"
        "  anisotropy: {in: [0, 1, 2]}\n"          # not two sides
        "  betti_1: {in: [0, 1]}\n"                # not in this suite
    )
    monkeypatch.setenv("MANYRUNS_METRICS_DIR", str(tmp_path))

    parsed = suite.ranges("wonky")
    problems = " | ".join(suite.check_ranges(["lid", "anisotropy", "betti_0"], "wonky"))

    assert set(parsed) == {"lid", "betti_1"}       # the malformed one is gone, not guessed
    assert "anisotropy: " in problems and "not a legal bound" in problems
    assert "betti_0: declared in the suite with no admissible range" in problems
    assert "betti_1: has an admissible range but is not in the suite" in problems


def test_a_missing_range_block_leaves_the_measurement_untouched(tmp_path, monkeypatch):
    """A suite file with no `ranges:` — every sweep arm under `$MANYRUNS_METRICS_DIR` is
    one. The values must still be measured and recorded; only the check is absent."""
    (tmp_path / "bare.yaml").write_text("metrics: [lid]\n")
    monkeypatch.setenv("MANYRUNS_METRICS_DIR", str(tmp_path))
    _fixed(monkeypatch, {"lid": 99.0})

    assert suite.ranges("bare") == {}
    g = suite.measure(_emb(600, 3), _emb(600, 40), ["lid"], admissible=suite.ranges("bare"))
    assert g["lid"] == 99.0 and "lid_note" not in g
    # ABSENT, not 0. `suite_out_of_range: 0` from a config with no `ranges:` block reads as
    # "nothing is out of range" when the truth is "nothing was examined" — the same collapse
    # between absent and clean that `<name>_note` prevents for individual metrics.
    assert "suite_out_of_range" not in g


def test_the_key_set_is_the_same_for_every_recipe(monkeypatch):
    """The point of the whole module: two runs of different recipes are comparable.

    Before this, the g-vector's cross-recipe intersection was ten keys of provenance —
    dataset shape and PHATE hyperparameters — and no geometry, because the only real
    measurements each belonged to exactly one recipe."""
    monkeypatch.setattr(suite, "_compute_metric", lambda: (lambda name, **k: 1.0))

    embed_like = suite.measure(_emb(), _emb(40, 10), DECLARED)
    contrast_like = suite.measure(_emb(60), _emb(60, 10), DECLARED,
                                  already={"separation": 0.4, "composition": 0.2})

    assert set(DECLARED) <= set(embed_like) & set(contrast_like)
