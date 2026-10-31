"""What the engine hands back besides the embedding — `runner._collect_extras`.

MEASURED BEFORE ANY OF THIS EXISTED, on a plain PHATE run through `manylatents.api.run`:

    returned : embeddings, label, metadata, affinity, kernel
    manyruns read : embeddings, scores

`manylatents.experiment` merges `algorithm.extra_outputs()` into the top level of its result
(experiment.py:443), so `affinity` and `kernel` — two N×N matrices the engine had already
computed — were dropped on the line that read `out.get("embeddings")`. Every latent step this
product has ever run threw them away.

It is not only affinity. `LatentModuleBase.extra_outputs` also collects `trajectories` and
`adjacency`, and `Cflows` — a LightningModule — emits `grn_edges`/`grn_weights`/`grn_node_ids`,
a gene regulatory network decoded back to gene space. That one is a FINDING, and it was going in
the bin too.

THE SIZE STORY IS THE OTHER HALF and is why `EXTRAS_MAX_BYTES` exists. On a `phate` step the
embedding is `(120, 3) float32` = 1.5 KiB and the affinity is `(120, 120) float64` = 112.6 KiB:
the byproduct is 75× the result, and it grows as N² while the embedding grows as N. At pbmc3k's
2,700 cells that is 111.2 MiB of affinity + kernel per latent step, and `cluster` has three.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from manyruns import artifacts  # noqa: E402
from manyruns.pipeline import runner  # noqa: E402


def _state():
    return {"X": None, "emb": None, "pseudotime": None, "labels": None, "seed": 0}


# ── the routing rule ─────────────────────────────────────────────────────────
def test_an_array_byproduct_is_kept_and_namespaced_by_the_step_that_made_it():
    """Namespaced because two steps can both emit `affinity` — `cluster` runs `pca` then `umap`
    — and an unnamespaced slot would have the second silently overwrite the first."""
    state, g = _state(), {}

    runner._collect_extras("phate", {"embeddings": np.zeros((4, 2)),
                                     "affinity": np.zeros((4, 4))}, state, g)

    assert list(state["extras"]) == ["phate.affinity"]
    assert state["extras"]["phate.affinity"].shape == (4, 4)


def test_a_scalar_byproduct_goes_to_the_g_vector_like_a_score():
    """Same namespacing `scores` already uses, so it is comparable across runs by the fixed-key
    rule everything else in the g-vector obeys."""
    state, g = _state(), {}

    runner._collect_extras("mioflow", {"n_branches": 3, "converged": True}, state, g)

    assert g["mioflow.n_branches"] == 3
    assert "mioflow.converged" not in g, "a bool is a flag, not a measurement"


@pytest.mark.parametrize("claimed", ["embeddings", "scores", "label", "metadata"])
def test_the_keys_manyruns_takes_for_itself_are_never_duplicated(claimed):
    """`label` is the caller's OWN labels echoed back — manyruns already holds them in
    `state["labels"]`, and writing the engine's copy beside them would put two accounts of one
    input in the outputs folder with nothing saying which won. `metadata` is manylatents'
    provenance for its own run, where manyruns has `run_id`/`spec_id`/`tool_version`."""
    state, g = _state(), {}

    runner._collect_extras("phate", {claimed: np.zeros((4, 4))}, state, g)

    assert not state.get("extras")
    assert g == {}


def test_a_none_value_is_not_a_byproduct():
    state, g = _state(), {}
    runner._collect_extras("phate", {"affinity": None}, state, g)
    assert not state.get("extras")


def test_a_non_dict_return_is_survived_rather_than_unpacked():
    """Engines are other people's code. A shape change upstream must not take a run down."""
    state, g = _state(), {}
    runner._collect_extras("phate", [1, 2, 3], state, g)
    assert not state.get("extras") and g == {}


# ── the size discipline ──────────────────────────────────────────────────────
def test_a_byproduct_has_its_own_much_smaller_ceiling():
    """`MAX_BYTES` was chosen for embeddings — "~2.8M cells at 3 float64 columns". An N×N
    affinity is a different size class, and at 64 MiB both of pbmc3k's matrices would be
    written on every latent step of every run."""
    assert artifacts.EXTRAS_MAX_BYTES < artifacts.MAX_BYTES

    # a 1,450-cell square is about the line; pbmc3k's 2,700 is well past it
    assert not artifacts.too_large(np.zeros((1000, 1000)), artifacts.EXTRAS_MAX_BYTES)
    assert artifacts.too_large(np.zeros((2700, 2700)), artifacts.EXTRAS_MAX_BYTES)
    # and the embedding cap is unmoved for manyruns's own slots
    assert not artifacts.too_large(np.zeros((2700, 3)))


def test_an_oversized_byproduct_is_a_stated_absence_that_names_the_shape(tmp_path,
                                                                        monkeypatch):
    """"affinity was too big" and "affinity was 300x300" are different amounts of help to
    someone deciding whether to raise the cap — and the shape is the whole reason it was."""
    monkeypatch.setattr(artifacts, "EXTRAS_MAX_BYTES", 1024)
    rec, state = {}, _state()
    state["extras"] = {"phate.affinity": np.zeros((300, 300))}

    runner._persist(rec, state, {"out_dir": tmp_path, "run_id": "r"}, 0, "phate", {}, {}, {})

    note = rec["artifacts"]["phate.affinity"]
    assert "not written" in note and "300x300" in note and "MiB" in note
    assert not list(artifacts.root(tmp_path, "r").glob("*affinity*"))


def test_an_unchanged_byproduct_is_not_written_twice(tmp_path):
    """Object identity, the same test the state slots use: a step that reads an extra another
    step produced must not deposit a second copy under its own name."""
    rec, state = {}, _state()
    shared = np.zeros((4, 4))
    state["extras"] = {"phate.affinity": shared}
    ctx = {"out_dir": tmp_path, "run_id": "r"}

    runner._persist(rec, state, ctx, 0, "phate", {}, {}, {})
    first = dict(rec["artifacts"])
    rec2: dict = {}
    runner._persist(rec2, state, ctx, 1, "mioflow", {}, {}, {"phate.affinity": shared})

    assert "phate.affinity" in first
    assert not rec2.get("artifacts"), "the second step rewrote an extra it only inherited"


# ── end to end ───────────────────────────────────────────────────────────────
def test_a_real_run_keeps_what_it_used_to_drop(tmp_path):
    """The whole point, against the real engine. Before this, the only artifact a `phate` step
    left was `emb`."""
    pytest.importorskip("manylatents")
    pytest.importorskip("phate")
    from manyruns import catalog

    res = runner.run_manylatents(catalog.load_recipe("embed"),
                                 array=np.asarray(np.random.default_rng(0).normal(size=(60, 8))),
                                 out_dir=tmp_path, seed=0)

    # Looked up by NAME, not `steps[0]`: manyruns#54 gave `embed` a declared prep block, so the
    # recipe is normalize -> transform -> pca(50) -> phate and phate sits at index 3. On this
    # Gaussian fixture the first two DECLINE (a point cloud is not counts) and record no
    # artifacts at all, which is why indexing found no `artifacts` key. The claim is about what
    # a phate step keeps, so address the phate step.
    written = next(s for s in res["steps"] if s["name"] == "phate")["artifacts"]
    assert "emb" in written
    assert "phate.affinity" in written, "the engine computed it and we dropped it again"
    assert artifacts.load_array(written["phate.affinity"]).shape == (60, 60)
    # and the caller's labels are still not on disk — see `artifacts.PERSISTED`
    assert not any("label" in k for k in written)


def test_the_mioflow_seam_returns_its_byproducts_instead_of_dropping_them():
    """Previously, "the trained flow — the simulatable object, the model
    — is discarded at the return statement, and only its coordinates survive." The model is
    still discarded; everything else the engine computed is not.

    A THIRD return value rather than a tolerant unpack, so a stale fake fails loudly instead of
    quietly continuing to pass — three tests supply one for this function."""
    import inspect

    from manyruns.pipeline import mioflow

    src = inspect.getsource(mioflow._run_mioflow_experiment)
    assert "return np.asarray(result.get(\"embeddings\")), (result.get(\"scores\") or {}), extras" \
        in src
