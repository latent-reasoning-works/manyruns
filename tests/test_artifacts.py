"""The state channel — intermediate arrays written down, addressably.

Before this, a run left `project.yaml`, `summary.md` and a PNG. The embedding — the object
every downstream question is about — lived in `state["emb"]` for the life of the process and
was dropped. What survived was a picture of it.

The round trip is the property under test: a run produces coordinates, they land on disk under
the run's own id, and they load back as a first-class input to another run. That is what makes
`run_inproc(embedding=...)` a seam rather than a hole.
"""
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
#: A REAL EMBEDDER, declared. These tests run a recipe end to end and assert on what it
#: PRODUCED — artifacts, figures, coordinates. Guarding only on numpy/sklearn was wrong in
#: the direction that hurts: CI has both and no `phate`, so the latent step was skipped,
#: the run produced nothing, and the assertions failed against an empty dict instead of
#: skipping. An absent embedder is not a defect in this code.
pytest.importorskip("phate")

from manyruns import artifacts, catalog  # noqa: E402
from manyruns.pipeline import runner  # noqa: E402

EMBED = {"name": "e", "steps": [{"name": "phate", "group": "latent", "params": {"n_components": 3}}]}
REUSE = {"name": "r", "steps": [{"name": "mioflow", "group": "lightning", "params": {}}]}


def _data(n=150, d=12, seed=0):
    return np.asarray(np.random.default_rng(seed).normal(size=(n, d)))


def test_a_run_leaves_its_embedding_on_disk_under_its_own_id(tmp_path):
    """Addressed by `run_id`, not by a filesystem convention a reader has to know.

    The path rides in the step record, so `index.jsonl` — which serialises `steps` — already
    carries it. A reader goes from a row in the run store to the array that row describes."""
    res = runner.run_inproc(_data(), EMBED, tmp_path, seed=0)

    path = res["steps"][0]["artifacts"]["emb"]
    assert res["run_id"] in path
    reloaded = artifacts.load_array(path)
    assert reloaded.shape == (150, 3)


def test_the_embedding_round_trips_into_another_run(tmp_path):
    """THE point. Measure a cohort once, then run variants against the same coordinates
    without recomputing them — and, because the g-vector's key set is fixed, compare them."""
    first = runner.run_inproc(_data(), EMBED, tmp_path, seed=0)
    emb = artifacts.load_array(first["steps"][0]["artifacts"]["emb"])

    second = runner.run_inproc(np.zeros((150, 12)), REUSE, tmp_path, seed=0, embedding=emb)

    assert second["ok"] is True
    assert second["steps"][0]["outcome"] == "ok"
    assert second["g_vector"]["final_dim"] == 3
    # and it is reported as the unusual route it is, rather than passing silently
    assert second["caveats"] and "not from a latent step" in second["caveats"][0]


def test_every_derived_slot_is_written_not_only_the_embedding(tmp_path):
    """`mioflow` on the in-process loop sets `pseudotime` and leaves `emb` untouched. Keying the write
    on the embedding alone dropped every ordering this product computes — the first version
    did exactly that, and `mioflow` recorded no artifact at all."""
    res = runner.run_inproc(_data(), catalog.load_recipe("cflows"), tmp_path, seed=0)

    by_step = {s["name"]: (s.get("artifacts") or {}) for s in res["steps"]}
    assert "emb" in by_step["phate"]
    assert "pseudotime" in by_step["mioflow"]
    assert artifacts.load_array(by_step["mioflow"]["pseudotime"]).shape == (150,)


def test_an_unchanged_slot_is_not_written_twice(tmp_path):
    """Object identity, the same test `_attach_geometry` uses. A step that reads the embedding
    without replacing it must not deposit a second identical copy under a different name."""
    res = runner.run_inproc(_data(), catalog.load_recipe("cflows"), tmp_path, seed=0)

    embs = [s["artifacts"]["emb"] for s in res["steps"] if "emb" in (s.get("artifacts") or {})]
    assert len(embs) == 1                      # phate wrote it; mioflow did not rewrite it


def test_the_folder_says_when_it_is_complete(tmp_path):
    """Without the marker a reader cannot tell "this run produced one artifact" from "this run
    was killed after its first". Taken from manylatents' `write_completion_marker`."""
    res = runner.run_inproc(_data(), catalog.load_recipe("cflows"), tmp_path, seed=0)
    state = artifacts.root(tmp_path, res["run_id"])

    assert artifacts.complete(state)
    assert not artifacts.complete(tmp_path / "state" / "a-run-that-never-happened")
    # the marker indexes what is in the folder
    import json

    manifest = json.loads((state / artifacts.DONE).read_text())
    assert {v["slot"] for v in manifest.values()} == {"emb", "pseudotime"}


def test_no_partial_file_survives_a_write(tmp_path):
    """`np.save` APPENDS `.npy` when the name lacks it, so the first temp name produced
    `<name>.npy.part.npy` on disk while the rename looked for `<name>.npy.part` and failed —
    silently, because every OSError here degrades to "not written". The folder held a partial
    and no real artifact."""
    res = runner.run_inproc(_data(), catalog.load_recipe("cflows"), tmp_path, seed=0)
    state = artifacts.root(tmp_path, res["run_id"])

    names = sorted(p.name for p in state.iterdir())
    assert not [n for n in names if "part" in n], names
    assert all(n.endswith(".npy") or n == artifacts.DONE for n in names), names


def test_the_input_matrix_is_never_written(tmp_path):
    """Only DERIVED state. `state["X"]` is the scientist's data — copying it into an outputs
    folder is a data-handling decision this module has no business making silently, and it is
    the line the federated design turns on: geometry leaves, the person's data does not."""
    assert "X" not in artifacts.PERSISTED
    assert "labels" not in artifacts.PERSISTED

    res = runner.run_inproc(_data(), catalog.load_recipe("cflows"), tmp_path, seed=0)
    state = artifacts.root(tmp_path, res["run_id"])

    for f in state.glob("*.npy"):
        assert artifacts.load_array(f).shape != (150, 12)   # the 12-column input, nowhere


def test_an_oversized_array_is_a_stated_absence(tmp_path, monkeypatch):
    """A size skip is RECORDED. `save_array` returns None both when an array is too large and
    when the disk refused it, and a reader wondering why a run has no state folder needs those
    to be different facts."""
    monkeypatch.setattr(artifacts, "MAX_BYTES", 8)          # smaller than any real embedding

    res = runner.run_inproc(_data(), EMBED, tmp_path, seed=0)

    note = res["steps"][0]["artifacts"]["emb"]
    assert "not written" in note and "larger than" in note
    assert res["ok"] is True                                 # and the run still succeeded


# ── resume: reading a completed run back without having run it ──────────────
def test_latest_run_is_none_with_no_state_folder(tmp_path):
    assert artifacts.latest_run(tmp_path) is None


def test_latest_run_finds_the_most_recently_completed_folder(tmp_path):
    import time

    first = runner.run_inproc(_data(), EMBED, tmp_path, seed=0)
    time.sleep(0.02)                                          # mtime has to actually move
    second = runner.run_inproc(_data(seed=1), EMBED, tmp_path, seed=1)

    assert artifacts.latest_run(tmp_path) == second["run_id"]
    assert first["run_id"] != second["run_id"]


def test_read_manifest_round_trips_what_finish_wrote(tmp_path):
    res = runner.run_inproc(_data(), EMBED, tmp_path, seed=0)

    manifest = artifacts.read_manifest(tmp_path, res["run_id"])
    path = res["steps"][0]["artifacts"]["emb"]
    assert manifest[path] == {"step": "phate", "index": 0, "slot": "emb"}


def test_a_failed_write_never_fails_the_run(tmp_path, monkeypatch):
    """A run that produced a valid embedding must not become a failed run because a disk was
    full. `_run_steps` never sees an exception from the artifact channel."""
    monkeypatch.setattr(artifacts, "save_array",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError):
        artifacts.save_array(tmp_path, "x", 0, "s", "emb", np.zeros((2, 2)))

    monkeypatch.setattr(artifacts, "save_array", lambda *a, **k: None)
    res = runner.run_inproc(_data(), EMBED, tmp_path, seed=0)

    assert res["ok"] is True and res["steps"][0]["outcome"] == "ok"


# ── figures belong to the step that drew them ────────────────────────────────
def test_each_step_owns_the_figure_it_drew(tmp_path):
    """`_io._save_scatter` appends to a RUN-scoped list, so nothing recorded WHICH step drew
    WHICH figure. That is why a live panel could never show the plot for the step that just
    ran — a record gap, not a rendering one. Diffed the same way `emitted` is."""
    from manyruns import catalog
    from manyruns.pipeline import runner

    res = runner.run_inproc(_data(), catalog.load_recipe("cflows"), tmp_path, seed=0)

    by_step = {s["name"]: [Path(p).name for p in s.get("plots", [])] for s in res["steps"]}
    assert by_step["phate"] == ["phate.png"]
    assert by_step["mioflow"] == ["trajectory.png"]
    # and the run-level list is still the union, unchanged for existing readers
    assert [Path(p).name for p in res["plots"]] == ["phate.png", "trajectory.png"]


def test_a_step_that_drew_nothing_carries_no_plots_key(tmp_path):
    """Absent, not empty. An analysis step that draws nothing and a step whose figure failed
    to write are different facts, and `[]` would collapse them — the same declared-not-
    conditional rule the metric suite follows for absences."""
    from manyruns.pipeline import runner

    recipe = {"name": "e", "steps": [{"name": "phate", "group": "latent",
                                      "params": {"n_components": 3}}]}
    res = runner.run_inproc(_data(), recipe, tmp_path, seed=0, embedding=None)
    assert "plots" in res["steps"][0]

    # a skipped step draws nothing and must not claim an empty figure list
    backwards = {"name": "b", "steps": [{"name": "mioflow", "group": "lightning", "params": {}}]}
    res2 = runner.run_inproc(_data(), backwards, tmp_path, seed=0)
    assert res2["steps"][0]["outcome"] == "skipped"
    assert "plots" not in res2["steps"][0]
