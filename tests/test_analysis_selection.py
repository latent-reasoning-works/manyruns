"""Automatic analysis selection (manyruns.app.select_analysis).

Dep-free: the decision logic + structural detection don't need anndata (the obs-inspection
edge is lazy and only fires for a real single .h5ad). Covers the core promise — stop always
imposing a trajectory: time axis → CFlows, unordered conditions → embed, override respected.
"""
from __future__ import annotations

from manyruns import app


# ── the pure decision ────────────────────────────────────────────────────────
def test_choose_time_axis_is_trajectory():
    name, why = app._choose_recipe(has_time_axis=True, conditions=None, modality="scrna")
    assert name == "cflows" and "time" in why.lower()


def test_choose_conditions_without_time_is_contrast():
    name, why = app._choose_recipe(
        has_time_axis=False, conditions=["treated", "healthy"], modality="scrna"
    )
    assert name == "contrast" and "treated" in why


def test_choose_no_signal_declines_to_claim_a_trajectory():
    """No time axis and no conditions → embed, for EVERY modality.

    This previously returned `cflows` (issue #26): `_RECIPE_FOR_MODALITY` mapped scrna/bulk to
    a trajectory and the fallback for everything else was `cflows` too, so the one branch
    reached when we can tell the LEAST about the data made the STRONGEST claim about it. The
    interactive path never agreed — `narrate.offer` returned `embed` for the same array — and
    both now derive from that single menu.
    """
    for modality in ("scrna", "bulk", "synthetic", "unknown"):
        name, why = app._choose_recipe(has_time_axis=False, conditions=None, modality=modality)
        assert name == "embed", f"{modality} should not get a trajectory with no ordering"
        assert "invent" in why


def test_choose_override_wins_and_reports():
    name, why = app._choose_recipe(
        has_time_axis=True, conditions=["a", "b"], modality="scrna", override="embed"
    )
    assert name == "embed" and "forced" in why


# ── structural detection (dep-free) ──────────────────────────────────────────
def _sample_dir(root, name):
    d = root / name
    d.mkdir()
    (d / "matrix.mtx.gz").write_bytes(b"")
    return d


def test_timepoint_subdirs_detected(tmp_path):
    _sample_dir(tmp_path, "T0")
    _sample_dir(tmp_path, "T1")
    assert app._has_timepoint_subdirs(tmp_path) is True


def test_single_or_no_subdirs_not_a_trajectory(tmp_path):
    _sample_dir(tmp_path, "only")
    assert app._has_timepoint_subdirs(tmp_path) is False        # one sample ≠ a series
    flat = tmp_path / "m.h5ad"
    flat.write_text("x")
    assert app._has_timepoint_subdirs(flat) is False            # a file
    assert app._has_timepoint_subdirs(None) is False


# ── end-to-end selection ─────────────────────────────────────────────────────
def test_select_analysis_timepoint_folder_gets_cflows(tmp_path):
    _sample_dir(tmp_path, "day0")
    _sample_dir(tmp_path, "day3")
    recipe, why = app.select_analysis("scrna", tmp_path)
    assert recipe["name"] == "cflows"
    # `normalize → transform → pca(50)` joined the front of this list in the cutover (#54). The
    # same three ran before, hidden in `loading._anndata_matrix`; declaring them changed where
    # they are written down, not what runs. The claim under test — a timepoint folder selects the
    # TRAJECTORY recipe, so `mioflow` is in it — is read off the tail.
    assert [s["name"] for s in recipe["steps"]] == [
        "normalize", "transform", "pca", "phate", "mioflow"]


def test_select_analysis_override_embed(tmp_path):
    recipe, why = app.select_analysis("scrna", tmp_path, override="embed")
    assert recipe["name"] == "embed" and "forced" in why


def test_select_analysis_plain_folder_declines_a_trajectory(tmp_path):
    # An eb-like folder (one placeholder .h5ad, no timepoint subdirs, no readable obs) carries
    # no ordering we can see, so it resolves to `embed`. This asserted `cflows` until #26 was
    # fixed, on the grounds of "preserving the Milestone-1 acceptance run" — but the Milestone-1
    # dataset is a real time course, and pinning its RECIPE via a folder with no readable
    # metadata pinned the guess rather than the case. Force `--recipe cflows` to run that path.
    (tmp_path / "matrix.h5ad").write_text("placeholder")
    recipe, why = app.select_analysis("scrna", tmp_path)
    assert recipe["name"] == "embed"
    assert "invent" in why


def test_embed_recipe_is_prep_then_phate_only():
    """`embed` embeds and claims nothing further — no trajectory, no causality, no contrast.

    This read `steps == ["phate"]` until the cutover (#54), which put the preamble
    `normalize → transform → pca(50)` in front of it. That preamble is not a new claim: it is
    the scRNA prep that `loading._anndata_matrix` ran on every counts-like load without telling
    anyone, now declared so it appears in the trace and its params are reachable from the recipe.
    So the property is no longer "one step" — it is that every step is `prep` or `latent`, i.e.
    the recipe only ever gets you a representation. A `lightning` or `analysis` step appearing
    here would be `embed` starting to assert something, which is what this test exists to catch.
    """
    recipe = app.load_recipe("embed")
    assert recipe["name"] == "embed"
    assert [s["name"] for s in recipe["steps"]] == ["normalize", "transform", "pca", "phate"]
    assert {s["group"] for s in recipe["steps"]} == {"prep", "latent"}


# ── the ② fix: case/control DIRECTORY layouts must not read as a time course ──
def test_per_condition_folders_route_to_contrast(tmp_path):
    # treated/ + healthy/ — a case/control layout, NOT a time course. Before the fix this was
    # read as ≥2 timepoints → cflows (a trajectory over a non-temporal axis); now → contrast.
    _sample_dir(tmp_path, "treated")
    _sample_dir(tmp_path, "healthy")
    recipe, why = app.select_analysis("scrna", tmp_path)
    assert recipe["name"] == "contrast" and "treated" in why


def test_condition_dir_is_not_temporal_but_temporal_dir_still_is(tmp_path):
    _sample_dir(tmp_path, "treated")
    _sample_dir(tmp_path, "healthy")
    assert app._has_timepoint_subdirs(tmp_path) is False        # names aren't temporal
    assert app._dir_conditions(tmp_path) == ["healthy", "treated"]

    tc = tmp_path / "course"
    tc.mkdir()
    _sample_dir(tc, "day0")
    _sample_dir(tc, "day3")                                     # temporal names → time course
    assert app._has_timepoint_subdirs(tc) is True
    assert app._dir_conditions(tc) is None
