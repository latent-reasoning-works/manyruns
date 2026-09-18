"""The human-facing layer (manyruns.narrate): read → describe → offer → narrate → refuse.

Dep-free: dir-shape reads reuse app's structural detection; describe/offer/narrate/refusal are
string logic over an Observation or a g-vector."""
from __future__ import annotations

from manyruns import narrate
from manyruns.narrate import Observation


def _sample_dir(root, name):
    d = root / name
    d.mkdir()
    (d / "matrix.mtx.gz").write_bytes(b"")
    return d


# ── read_data: shape from directory layout ───────────────────────────────────
def test_reads_case_control_from_condition_folders(tmp_path):
    _sample_dir(tmp_path, "treated")
    _sample_dir(tmp_path, "healthy")
    obs = narrate.read_data(tmp_path, "scrna")
    assert obs.shape == "case-control" and obs.conditions == ["healthy", "treated"]


def test_reads_time_course_from_timepoint_folders(tmp_path):
    _sample_dir(tmp_path, "day0")
    _sample_dir(tmp_path, "day3")
    obs = narrate.read_data(tmp_path, "scrna")
    assert obs.shape == "time-course" and obs.n_timepoints == 2


# ── describe / offer ─────────────────────────────────────────────────────────
def test_describe_speaks_the_shape():
    cc = Observation(shape="case-control", conditions=["treated", "healthy"], n_obs=48900)
    assert "48,900 cells" in narrate.describe(cc) and "treated" in narrate.describe(cc)
    assert "snapshot" in narrate.describe(cc)
    tc = Observation(shape="time-course", n_timepoints=5, n_obs=31000)
    assert "time course" in narrate.describe(tc)


def test_offer_recommends_the_right_question():
    cc = narrate.offer(Observation(shape="case-control", conditions=["a", "b"]))
    assert cc[0]["recipe"] == "contrast" and cc[0]["recommended"] and "different" in cc[0]["label"].lower()
    assert "phate" not in " ".join(m["label"] for m in cc).lower()   # never an algorithm name
    tc = narrate.offer(Observation(shape="time-course"))
    assert tc[0]["recipe"] == "cflows"


# ── the refusal (the killer moment) ──────────────────────────────────────────
def test_refuses_a_trajectory_on_case_control():
    cc = Observation(shape="case-control", conditions=["treated", "healthy"])
    msg = narrate.refusal("show me how the disease progresses over time", cc)
    assert msg is not None and "no clock" in msg and "how the groups differ" in msg


def test_does_not_refuse_a_trajectory_on_a_time_course():
    tc = Observation(shape="time-course", n_timepoints=5)
    assert narrate.refusal("trace the progression", tc) is None      # valid — no refusal
    cc = Observation(shape="case-control", conditions=["a", "b"])
    assert narrate.refusal("compare the groups", cc) is None         # not a trajectory ask


def test_interpret_maps_plain_questions_to_recipes():
    cc = Observation(shape="case-control", conditions=["treated", "healthy"])
    assert narrate.interpret("which cells differ in treated?", cc) == "contrast"
    assert narrate.interpret("just map the structure", cc) == "embed"


# ── narrate: g-vector → plain English (only what was computed) ────────────────
def test_narrate_translates_separation_and_composition():
    r = {"g_vector": {"separation_silhouette": 0.71, "composition_max_log2_shift": 2.0,
                      "composition_between": "treated vs healthy"}}
    out = narrate.narrate(r)
    assert "0.71 out of 1" in out and "strong, real difference" in out
    assert "~4× more abundant" in out and "treated" in out


def test_narrate_refuses_direction_on_a_trajectory_run():
    """The sentence that replaced a false one.

    `granger` narrated "some signals lead others — there's a directional order" off a
    statistic that rejected in 12/12 seeds on pure noise. It is deleted; a trajectory run
    now says direction is not assessable, and says WHY and what would fix it, because a
    user who gets silence assumes the answer was boring rather than unavailable.
    """
    out = narrate.narrate({"g_vector": {"pseudotime_range": [0.0, 1.0]}})
    assert "can't tell you" in out
    assert "spliced/unspliced" in out or "collected at known times" in out
    assert "Mapped the structure" not in out          # NOT the embed fallback


def test_narrate_never_claims_a_direction():
    """No input may resurrect the directional sentence — including the old g-vector keys."""
    for g in ({"granger": 0.007}, {"granger_min_p": 1e-38}, {"pseudotime_range": [0.0, 1.0]}):
        out = narrate.narrate({"g_vector": g})
        assert "lead others" not in out
        assert "directional order" not in out


def test_direction_refusal_reads_the_trace_not_the_recipe_name():
    """The mock writes the g-vector only for `analysis` steps, so a mock trajectory run has
    a trace but an empty g. Reading the trace is what makes the refusal fire there too."""
    out = narrate.narrate({"g_vector": {}, "trace": ["latent:phate", "lightning:mioflow"]})
    assert "can't tell you" in out
    assert "Mapped the structure" not in out


def test_a_run_with_no_trajectory_step_gets_no_direction_refusal():
    """Refusing a direction for a path that was never built would be its own invention."""
    out = narrate.narrate({"g_vector": {}, "trace": ["latent:phate"], "final_dim": 3})
    assert "can't tell you" not in out
    assert "Mapped the structure" in out


def test_narrate_embed_says_it_only_mapped():
    out = narrate.narrate({"g_vector": {}, "final_dim": 3})
    assert "Mapped the structure" in out and "3 dimensions" in out


def test_narrate_never_invents_when_gvector_is_empty():
    # no separation/composition keys → it must NOT claim a finding
    out = narrate.narrate({"g_vector": {}})
    assert "separate" not in out.lower() and "abundant" not in out.lower()
