"""The complexity ladder (manyruns/score.py) — pure, dependency-free scoring."""
import inspect
from pathlib import Path

import pytest

from manyruns import decisions, score
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
    """Every rung has a claimant in the bundled catalogue: `branches` claims multi-branching
    and `archetypes` claims archetypal. Pinned so a gap that reopens is reported rather than
    hidden inside a mean. This reads what recipes CLAIM; the offer form below reads what is
    offered and runnable."""
    assert score.unroutable_rungs() == []


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


def test_the_benchmark_collapse_is_not_on_the_ladder():
    """`vocab.collapse_for_benchmark` folds `surface` into `archetypal` for scShapeBench's scorer.
    On the ladder that would move `surface` from rung 0 to rung 4 and reward an over-claim, so
    `truth_rung` must not call it and the decision store must not either. Only these two are
    read: other scorer functions in `score.py` may call the collapse."""
    assert score.truth_rung(["surface"]) == 0
    assert score.truth_rung(["cycle"]) == 2
    assert score.RUNG == {"surface": 0, "clusters": 1, "cycle": 2, "single-trajectory": 2,
                          "multi-branching": 3, "archetypal": 4}
    assert "collapse_for_benchmark" not in inspect.getsource(score.truth_rung)
    source = Path(decisions.__file__)
    assert source.is_file()
    assert "collapse_for_benchmark" not in source.read_text()


# ── raw labels resolve to one rung ───────────────────────────────────────────
def test_every_raw_label_has_one_rung():
    assert score.check_raw_labels() == []


def test_drifted_raw_label_is_reported(monkeypatch):
    from manyruns import vocab

    monkeypatch.setitem(vocab.SCSHAPEBENCH_TOPOLOGY, "blob", "blob")
    problems = score.check_raw_labels()
    assert len(problems) == 1
    assert "blob" in problems[0]


# ── verdict rows ─────────────────────────────────────────────────────────────
def _probs(**given):
    """Six probabilities in `TOPOLOGIES` order; keyword names use `_` for `-`."""
    named = {k.replace("_", "-"): v for k, v in given.items()}
    return {t: float(named.get(t, 0.0)) for t in TOPOLOGIES}


def _verdict_row(dataset_name, probs, *, dataset=None):
    return {"run_id": f"run-{dataset_name}", "dataset_name": dataset_name, "dataset": dataset,
            "ok": None, "complete": None, "steps": [], "g_vector": {},
            "extra": {"verdict": {"topology_probs": probs, "chooser": "human:a3",
                                  "bound": "none", "legal_id": "a37fab6bef12",
                                  "model": None}}}


def test_pred_rung_argmax():
    row = _verdict_row("ds", _probs(multi_branching=0.6, clusters=0.8))
    assert score.pred_rung(row) == 1


def test_pred_rung_tie_takes_lower_rung():
    """Overshoot costs twice undershoot, so a tie takes the cheaper error."""
    row = _verdict_row("ds", _probs(archetypal=0.5, clusters=0.5))
    assert score.pred_rung(row) == 1


def test_pred_rung_all_zero_is_rung_0():
    assert score.pred_rung(_verdict_row("ds", _probs())) == 0


def test_plain_run_row_is_not_a_verdict(tmp_path):
    from manyruns import store

    store.append({"run_id": "r1"}, out_dir=tmp_path)
    (row,) = list(store.read(tmp_path))
    assert score.verdict_of(row) is None
    with pytest.raises(ValueError):
        score.pred_rung(row)


def test_summarize_reads_a_verdict_row():
    row = _verdict_row("ds", _probs(clusters=0.9))
    record = {"truth_rung": score.truth_rung(["multi-branching"]),
              "pred_rung": score.pred_rung(row)}
    out = score.summarize([record])
    assert out["under"] == 1
    assert out["mean_cost"] == 2.0


# ── truth rows from label annotations ────────────────────────────────────────
def _label(chooser, topology, *, dataset=None, run_id=None, kind="label"):
    return {"kind": kind, "run_id": run_id, "chooser": chooser,
            "payload": {"dataset": dataset, "topology": topology, "raw": []}}


def test_imported_label_keeps_its_dataset():
    row = {"kind": "label", "run_id": None, "chooser": "human:a3",
           "payload": {"dataset": "scd_0001", "topology": ["clusters", "multi-branching"],
                       "raw": ["multi_branch", "clusters"]}}
    assert score.truth_from_annotations([row], []) == [
        {"dataset": "scd_0001", "annotator": "human:a3",
         "topology": ["clusters", "multi-branching"]}]

    # The real row shape, cut by `tests/fixtures/cut_annotations.py`.
    import json

    fixture = Path(__file__).parent / "fixtures" / "annotations.jsonl"
    assert fixture.is_file()
    rows = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    imported = [r for r in rows if r.get("kind") == "label" and r.get("run_id") is None]
    assert imported, "the fixture holds no imported label row"
    (truth,) = score.truth_from_annotations(imported[:1], [])
    assert truth == {"dataset": imported[0]["payload"]["dataset"],
                     "annotator": imported[0]["chooser"],
                     "topology": imported[0]["payload"]["topology"]}


def test_run_label_takes_the_run_dataset():
    rows = [_label("human:a3", ["clusters"], run_id="r1")]
    runs = [{"run_id": "r1", "dataset_name": "other"},
            {"run_id": "r1", "dataset_name": "pbmc3k"}]
    (truth,) = score.truth_from_annotations(rows, runs)
    assert truth["dataset"] == "pbmc3k"


def test_unknown_topology_name_is_dropped():
    rows = [_label("human:a3", ["clusters", "blob"], dataset="ds")]
    (truth,) = score.truth_from_annotations(rows, [])
    assert truth["topology"] == ["clusters"]


def test_rating_rows_are_ignored():
    rows = [_label("human:a3", ["clusters"], dataset="ds", kind="rating")]
    assert score.truth_from_annotations(rows, []) == []


def _truth(dataset, annotator, topology):
    return {"dataset": dataset, "annotator": annotator, "topology": list(topology)}


def test_cycle_votes_count_as_trajectory():
    rows = [_truth("ds_c", "a1", ["single-trajectory", "clusters"]),
            _truth("ds_c", "a2", ["cycle", "clusters"])]
    (out,) = score.soft_truth(rows)
    assert out["dataset"] == "ds_c"
    assert out["n_annotators"] == 2
    assert list(out["soft"]) == list(score.BENCHMARK_CLASSES)
    assert out["soft"]["single-trajectory"] == 1.0
    assert out["soft"]["clusters"] == 1.0
    assert out["soft"]["multi-branching"] == 0.0


def test_last_row_per_annotator_wins():
    rows = [_truth("ds_x", "a1", ["clusters"]), _truth("ds_x", "a1", ["archetypal"])]
    (out,) = score.soft_truth(rows)
    assert out["soft"]["clusters"] == 0.0
    assert out["soft"]["archetypal"] == 1.0
    assert out["n_annotators"] == 1


# ── per-class AUPRC ──────────────────────────────────────────────────────────
def test_auprc_hand_computed():
    verdicts = [_verdict_row("d1", _probs(clusters=0.9)),
                _verdict_row("d2", _probs(clusters=0.8)),
                _verdict_row("d3", _probs(clusters=0.7))]
    truth = [_truth("d1", "a1", ["clusters"]), _truth("d2", "a1", ["archetypal"]),
             _truth("d3", "a1", ["clusters"])]
    out = score.class_auprc(verdicts, truth)
    assert set(out) == {"macro", "per_class", "positives", "n", "unmatched"}
    assert out["per_class"]["clusters"] == pytest.approx(5 / 6, abs=1e-9)
    assert out["positives"]["clusters"] == 2
    assert out["n"] == 3


def test_auprc_ties_at_one_threshold():
    verdicts = [_verdict_row("d1", _probs(clusters=0.5)),
                _verdict_row("d2", _probs(clusters=0.5))]
    truth = [_truth("d1", "a1", ["clusters"]), _truth("d2", "a1", ["archetypal"])]
    out = score.class_auprc(verdicts, truth)
    assert out["per_class"]["clusters"] == pytest.approx(0.5, abs=1e-9)


def test_class_without_positive_leaves_macro():
    verdicts = [_verdict_row("d1", _probs(clusters=0.9)),
                _verdict_row("d2", _probs(clusters=0.8, multi_branching=0.6)),
                _verdict_row("d3", _probs(clusters=0.7, single_trajectory=0.4))]
    truth = [_truth("d1", "a1", ["clusters"]), _truth("d2", "a1", ["multi-branching"]),
             _truth("d3", "a1", ["clusters", "single-trajectory"])]
    out = score.class_auprc(verdicts, truth)
    assert out["per_class"]["archetypal"] is None
    assert out["per_class"]["clusters"] == pytest.approx(5 / 6, abs=1e-9)
    assert out["per_class"]["multi-branching"] == pytest.approx(1.0, abs=1e-9)
    assert out["per_class"]["single-trajectory"] == pytest.approx(1.0, abs=1e-9)
    assert out["macro"] == pytest.approx(17 / 18, abs=1e-9)


def test_unmatched_verdict_is_counted():
    verdicts = [_verdict_row("ds_unlabeled", _probs(clusters=0.9)),
                _verdict_row("d1", _probs(clusters=0.9))]
    out = score.class_auprc(verdicts, [_truth("d1", "a1", ["clusters"])])
    assert out["unmatched"] == 1
    assert out["n"] == 1


def test_auprc_ignores_plain_run_rows_and_reads_dataset_fallback():
    plain = {"run_id": "r0", "dataset_name": "d1", "extra": None}
    by_dataset = _verdict_row(None, _probs(clusters=0.9), dataset="d1")
    out = score.class_auprc([plain, by_dataset], [_truth("d1", "a1", ["clusters"])])
    assert out["n"] == 1
    assert out["unmatched"] == 0


# ── Brier score ──────────────────────────────────────────────────────────────
def test_brier_perfect_verdict():
    truth = [_truth("d1", "a1", ["clusters"]), _truth("d1", "a2", ["clusters", "cycle"])]
    soft = score.soft_truth(truth)
    verdict = _verdict_row("d1", _probs(clusters=1.0, single_trajectory=0.5))
    out = score.brier([verdict], soft)
    assert set(out) == {"brier", "per_class", "n", "unmatched"}
    assert out["brier"] == 0.0
    assert out["n"] == 1


def test_brier_no_verdicts():
    out = score.brier([], [])
    assert out["brier"] is None
    assert out["n"] == 0
    assert out["unmatched"] == 0
    assert all(v is None for v in out["per_class"].values())


# ── one index drives every scorer ────────────────────────────────────────────
def test_one_index_drives_every_scorer():
    """One hand-built three-row index, four scorers, values from the class-scoring spec."""
    from manyruns.vocab import collapse_for_benchmark

    index = [_verdict_row("ds_a", _probs(clusters=0.9, surface=0.1)),
             _verdict_row("ds_b", _probs(clusters=0.8, multi_branching=0.6)),
             _verdict_row("ds_c", _probs(clusters=0.7, cycle=0.6, single_trajectory=0.3))]
    truth = [_truth("ds_a", "a1", ["clusters"]), _truth("ds_a", "a2", ["clusters"]),
             _truth("ds_b", "a1", ["multi-branching"]),
             _truth("ds_b", "a2", ["multi-branching", "clusters"]),
             _truth("ds_b", "a3", ["multi-branching"]),
             _truth("ds_c", "a1", ["single-trajectory", "clusters"]),
             _truth("ds_c", "a2", ["cycle", "clusters"])]

    auprc = score.class_auprc(index, truth)
    assert auprc["per_class"]["clusters"] == pytest.approx(5 / 6, abs=1e-9)
    assert auprc["per_class"]["multi-branching"] == pytest.approx(1.0, abs=1e-9)
    assert auprc["per_class"]["single-trajectory"] == pytest.approx(1.0, abs=1e-9)
    assert auprc["per_class"]["archetypal"] is None
    assert auprc["macro"] == pytest.approx(17 / 18, abs=1e-9)
    assert (auprc["n"], auprc["unmatched"]) == (3, 0)

    soft = score.soft_truth(truth)
    assert score.brier(index, soft)["brier"] == pytest.approx(583 / 10800, abs=1e-9)

    soft_by = {s["dataset"]: s["soft"] for s in soft}
    cost, f1 = {}, {}
    for row in index:
        name = row["dataset_name"]
        truth_set = [c for c, v in soft_by[name].items() if v >= 0.5]
        cost[name] = score.ladder_cost(score.truth_rung(truth_set), score.pred_rung(row))
        probs = score.verdict_of(row)["topology_probs"]
        offered = [collapse_for_benchmark(t) for t, p in probs.items() if p >= 0.5]
        f1[name] = score.score_offer(truth_set, offered)["f1"]
    assert cost == pytest.approx({"ds_a": 0.0, "ds_b": 2.0, "ds_c": 1.0}, abs=1e-9)
    assert f1 == pytest.approx({"ds_a": 1.0, "ds_b": 2 / 3, "ds_c": 1.0}, abs=1e-9)


def test_score_module_imports_only_vocab():
    """A scorer that could reach a runner would become another step loop."""
    import ast

    tree = ast.parse(Path(score.__file__).read_text())
    modules = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert modules == ["__future__", "typing", "manyruns.vocab"]
    assert not [n for n in tree.body if isinstance(n, ast.Import)]


# ── unroutable_rungs on an offered set ───────────────────────────────────────
# Only recipes that were bundled before `branches` existed: `embed` 0, `cluster` 1,
# `pseudotime` 2, `archetypes` 4.


def test_unroutable_rungs_of_an_offer_of_names():
    assert score.unroutable_rungs(["embed", "cluster", "pseudotime"]) == [3, 4]


def test_unroutable_rungs_does_not_count_a_hollow_row():
    offer = [{"recipe": "embed", "can_run": True}, {"recipe": "cluster", "can_run": True},
             {"recipe": "pseudotime", "can_run": True}, {"recipe": "archetypes", "can_run": False}]
    assert score.unroutable_rungs(offer) == [3, 4]
    offer[-1] = {"recipe": "archetypes", "can_run": True}
    assert score.unroutable_rungs(offer) == [3]


def test_unroutable_rungs_counts_a_mapping_without_can_run():
    assert score.unroutable_rungs([{"recipe": "archetypes"}]) == [0, 1, 2, 3]


def test_unroutable_rungs_of_an_empty_offer_closes_nothing():
    assert score.unroutable_rungs([]) == [0, 1, 2, 3, 4]


def test_unroutable_rungs_ignores_unknown_names():
    assert score.unroutable_rungs(["nonsense", "qc"]) == [0, 1, 2, 3, 4]


def test_unroutable_rungs_takes_config_dir_by_keyword_only(tmp_path):
    for name, claim in (("s0", "surface"), ("c1", "clusters"), ("t2", "single-trajectory"),
                        ("b3", "multi-branching"), ("a4", "archetypal")):
        (tmp_path / f"{name}.yaml").write_text(
            f"name: {name}\nclaims: {claim}\nsteps: [{{name: pca, group: latent}}]\n")
    assert tmp_path.is_dir()
    assert score.unroutable_rungs(config_dir=tmp_path) == []
    assert score.unroutable_rungs(None, config_dir=tmp_path) == []
    with pytest.raises(TypeError):
        score.unroutable_rungs(None, tmp_path)
