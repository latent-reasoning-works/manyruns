"""The gene axis — the genes and the untransformed counts, threaded to where a step can read them.

THE MEASURED GAP these tests were written against. On `data/pbmc3k_raw.h5ad` (2700 × 32738,
CSR float32, `var_names[:3] == ['MIR1302-10', 'FAM138A', 'OR4F5']`), every route out of
`pipeline.loading` returned `(2700, 50)`:

    loading._anndata_matrix(adata)  -> (2700, 50)
    loading.as_matrix(adata)        -> (2700, 50)
    loading.load_labeled(path)      -> (2700, 50)

because the counts branch of `_anndata_matrix` ran normalize_total → log1p → PCA(50) and
returned `obsm["X_pca"]`. The 32,738-column gene axis was destroyed there, and the NAMES were
never returned by anything in that module — not even on the branch that kept the columns.

THE COLUMN HALF OF THAT GAP IS CLOSED; THE NAME HALF IS NOT — and which is which is the only
thing that changed in this file at the cutover (manyruns#54). `_anndata_matrix` is now
`return adata.X`: raw, unreduced, not densified, with the three operations promoted to declared
steps (`normalize`/`transform` in the `prep` group, `pca` in `latent`). So the columns survive
the load and the two tests that pinned their destruction are re-framed rather than deleted —
same property, opposite direction, see their docstrings. The names still arrive by exactly one
route: `gene_axis`, which `app._load_inputs` still does not call (`app.py:1560` says so in
those words), so `steps._step_composition`'s standing note ("it needs the original expression
matrix + gene names threaded through, which this pipeline doesn't yet carry") is now half
right — `markers` / `de` / `communicate` wait on the WIRING, no longer on the axis itself.

MOST OF THIS FILE IS DEP-FREE ON PURPOSE. CI has numpy but not anndata, and a suite whose
gene-axis coverage all skips there is coverage that only exists on one laptop. `loading`
identifies an AnnData structurally (`_is_anndata`: the type is named `AnnData`, or its module
mentions anndata), so a local stand-in class exercises every branch of `gene_axis` with no
import at all. The two tests that need the real library say so and skip.

ONE MEASURED CONSEQUENCE IS NOT PINNED HERE, DELIBERATELY. Now that `_anndata_matrix` returns
`.X` as stored, `as_matrix(adata)` — the second of the three routes above — raises
`ValueError: setting an array element with a sequence` for any AnnData whose `.X` is sparse,
because `np.asarray(csr, dtype=float)` is not a densification (measured 2026-08-16, this
checkout, 20 × 10 CSR; it is the shape pbmc3k has). That is a source defect, reported rather
than asserted: a test that pinned it would pin the bug.
"""
from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")


class AnnData:  # noqa: N801 - the NAME is the fixture: `loading._is_anndata` matches on it
    """The smallest object `loading._is_anndata` accepts. Not a mock of anndata's behaviour —
    only of the two attributes `gene_axis` reads, so a test that passes here is a test about
    manyruns's code rather than about anndata's."""

    def __init__(self, X, var_names):
        self.X = X
        self.var_names = list(var_names)


class Sparse:
    """A CSR stand-in: the ONE property that matters here is that `loading._dense` has to
    build a new object to read it, which an ndarray never does.

    Without this the no-copy test below is vacuous and says so by passing under mutation —
    measured: replacing `return counts` with `return _dense(counts)` in `gene_axis` left all
    16 tests green, because `np.asarray(ndarray)` returns the same object. `test_the_real_
    csr_behaves_like_the_stand_in` pins it against scipy where scipy is installed."""

    def __init__(self, dense):
        self._dense = dense
        self.shape = dense.shape
        self.ndim = 2

    def toarray(self):
        return self._dense.copy()

    def __getitem__(self, key):
        return Sparse(self._dense[key])


def _counts(n=6, p=4):
    return np.arange(n * p, dtype=float).reshape(n, p)


def _adata(n=6, p=4):
    return AnnData(_counts(n, p), [f"GENE{i}" for i in range(p)])


def _install(monkeypatch, **impls):
    from manyruns import pipeline

    for name, fn in impls.items():
        monkeypatch.setitem(pipeline._INPROC_STEPS, name, fn)


def _embeds(state, g, params, out_dir, plots, target_dim):
    state["emb"] = np.zeros((state["X"].shape[0], 2))


def _recipe(*names, group="latent"):
    return {"name": "r", "steps": [{"name": n, "group": group} for n in names]}


# ── the accessor ─────────────────────────────────────────────────────────────
def test_a_thing_with_no_gene_axis_reports_none_rather_than_guessing():
    """Both halves absent, not one. An ndarray has columns but no names for them, and
    inventing `["0", "1", …]` would give a downstream readout gene symbols to print."""
    from manyruns.pipeline.loading import gene_axis

    assert gene_axis(np.zeros((5, 3))) == (None, None)
    assert gene_axis(None) == (None, None)
    assert gene_axis([[1, 2], [3, 4]]) == (None, None)


def test_the_gene_axis_is_the_matrix_itself_never_a_copy():
    """THE constraint that makes this safe to thread at all.

    `counts` is `.X`, the same object — not densified, not copied. Measured on pbmc3k the
    difference is 18.3 MB of CSR against 353.6 MB dense: a 19× copy of the scientist's data,
    made on their behalf, with nothing asking. Mutation-checked with a sparse-like matrix:
    returning `_dense(counts)` from `gene_axis` fails this test. It does NOT fail on a dense
    ndarray, which is why `Sparse` exists — measured, and the reason this assertion is
    written twice.
    """
    from manyruns.pipeline.loading import gene_axis

    ad = _adata()
    counts, genes = gene_axis(ad)
    assert counts is ad.X
    assert list(genes) == list(ad.var_names)

    sparse = AnnData(Sparse(_counts(6, 4)), list("ABCD"))
    counts, _ = gene_axis(sparse)
    assert counts is sparse.X
    assert hasattr(counts, "toarray")   # still sparse: nothing densified it on the way through


def test_the_counts_and_the_genes_are_returned_together_or_not_at_all():
    """Two halves of one fact. Fetching them from two places is how they come to disagree —
    the argument `loading.labels_of` makes for rows, one axis over."""
    from manyruns.pipeline.loading import gene_axis

    counts, genes = gene_axis(_adata(6, 4))
    assert counts is not None and genes is not None
    assert genes.shape[0] == counts.shape[1]
    # a gene axis with no matrix behind it is not half an answer, it is no answer
    assert gene_axis(AnnData(None, ["A", "B"])) == (None, None)
    assert gene_axis(AnnData(np.zeros(5), ["A"])) == (None, None)  # 1-D: not a gene axis


def test_a_gene_name_per_column_is_checked_not_assumed():
    """The column analogue of `aligned.check_aligned`, which guards ROWS and cannot see this.

    anndata guarantees `len(var_names) == n_vars`, so this can only fire for a stand-in or a
    hand-built object — which is precisely the case no other check in the repo covers. Off by
    one column and every gene a readout names is the wrong gene, silently."""
    from manyruns.pipeline.loading import gene_axis

    with pytest.raises(ValueError, match="gene axis misalignment"):
        gene_axis(AnnData(_counts(6, 4), ["A", "B", "C"]))


# ── the one discriminator ────────────────────────────────────────────────────
def test_looks_like_counts_is_one_test_with_two_readers():
    """THE TWO READERS ARE DIFFERENT READERS SINCE THE CUTOVER, and the property is not.

    It used to decide whether `_anndata_matrix` ran the preamble AND whether a gene-level step
    must normalize what `gene_axis` handed it. The preamble is gone (`_anndata_matrix` returns
    `.X`), so both readers are now gene-level steps asking the same question of `state["counts"]`:
    `steps._step_rank_genes` (`steps.py:495`, before it normalizes its copy) and
    `steps._looks_countslike` (`steps.py:861`, for `qc`). The `normalize` step asks it a third
    time (`prep._looks_like_counts`) and answers by delegating to manylatents' implementation
    rather than re-writing the expression — which is the same reason this test exists. A second
    copy of the test is what `vocab.py`'s docstring records as drifting in silence, so there is one.

    Mutation-checked: dropping the `>= 0` clause makes the negative case below report True.
    """
    from manyruns.pipeline.loading import looks_like_counts

    assert looks_like_counts(np.array([[0.0, 3.0], [7.0, 2.0]])) is True
    assert looks_like_counts(np.array([[0.5, 3.1], [7.2, 2.4]])) is False
    assert looks_like_counts(np.array([[-1.0, 3.0], [7.0, 2.0]])) is False  # log-normalized
    assert looks_like_counts("not a matrix") is False  # unreadable → conservative False


def test_the_loader_hands_over_the_frame_and_only_gene_axis_names_it():
    """PREMISE INVERTED BY THE CUTOVER; the property underneath is the one that survived.

    This was `test_the_preamble_and_the_gene_axis_disagree_by_construction`, and its point was
    that `_anndata_matrix` transformed while `gene_axis` did not, so the two returned different
    matrices for the same object. `_anndata_matrix` is `return adata.X` now, so they return the
    SAME object — they cannot disagree, which is strictly better than the old test's guarantee
    and is asserted below. What did NOT change, and is what this file is about: the names come
    back from `gene_axis` and from nothing else in `loading`.

    THE SPARSE CASE IS THE ASSERTION THAT DISCRIMINATES. The old body densified on every path
    (`_dense(.X)`); the new one does not, and on pbmc3k that is 353.6 MB against an 18.3 MB CSR.
    A dense ndarray cannot see the difference — `np.asarray(ndarray)` returns the same object,
    the measurement `Sparse`'s docstring records — so the stand-in carries it here too."""
    from manyruns.pipeline import loading

    ad = AnnData(np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]), ["A", "B", "C"])
    matrix = loading._anndata_matrix(ad)
    assert matrix is ad.X                  # the frame itself: was `_dense(.X)` before the cutover
    assert matrix.shape == (2, 3)          # columns survive …
    counts, genes = loading.gene_axis(ad)
    assert counts is matrix                # … and both accessors now hand over one object
    assert list(genes) == ["A", "B", "C"]  # … but only gene_axis says what the columns are

    sparse = AnnData(Sparse(_counts(2, 3)), ["A", "B", "C"])
    # Sparse in, sparse out. The old branch returned `_dense(.X)` here and this was `.toarray()`.
    assert loading._anndata_matrix(sparse) is sparse.X
    assert hasattr(loading._anndata_matrix(sparse), "toarray")


# ── the thread through run_inproc ──────────────────────────────────────────
def test_run_inproc_derives_the_gene_axis_from_the_object_it_already_held(tmp_path,
                                                                            monkeypatch):
    """THE regression. `run_inproc` receives the loaded OBJECT and collapsed it with
    `as_matrix` one line in. The genes were in the function's hands and were dropped.

    Mutation-checked: deleting the `gene_axis(matrix)` derivation leaves `counts`/`genes`
    None here while every other assertion in the suite still passes.

    SCOPE, SINCE `engine=real` WAS REMOVED — read this as pinning the LOOP's contract, not a
    product capability. The caller that used to hand this function an AnnData was
    `app._read_inputs`' `real` branch, and it went with the engine; `run_inproc` is now the
    test substrate (`vocab.INPROC`), which no product surface routes to. So
    `loading.gene_axis` has **no production caller today**: `counts`/`genes` reach no
    user-visible run, and `rank_genes` records `rank_genes_top_note` on every engine until
    someone re-wires the axis into `_read_inputs` and widens its tuple across the call sites.
    This test stays green and stays honest because what it asserts — that the loop derives the
    axis from the object it already holds — is still exactly true of the loop.
    """
    from manyruns import pipeline

    seen: dict = {}

    def spy(state, g, params, out_dir, plots, target_dim):
        seen.update(counts=state.get("counts"), genes=state.get("genes"), X=state.get("X"))

    _install(monkeypatch, spy=spy)
    ad = _adata(6, 4)
    pipeline.run_inproc(ad, _recipe("spy"), tmp_path)

    assert seen["counts"] is ad.X                       # the same object, still
    assert list(seen["genes"]) == list(ad.var_names)
    assert seen["X"].shape == (6, 4)                    # what steps run on is unchanged


def test_a_plain_array_still_runs_and_simply_has_no_genes(tmp_path, monkeypatch):
    """Most data manyruns runs on is a point cloud with no gene axis at all. The absence has
    to be an honest None, not a crash and not an invented axis."""
    from manyruns import pipeline

    seen: dict = {}

    def spy(state, g, params, out_dir, plots, target_dim):
        seen.update(counts=state.get("counts"), genes=state.get("genes"))

    _install(monkeypatch, spy=spy)
    out = pipeline.run_inproc(np.zeros((6, 3)), _recipe("spy"), tmp_path)
    assert out["ok"] is True
    assert seen == {"counts": None, "genes": None}


def test_an_explicitly_supplied_gene_axis_wins(tmp_path, monkeypatch):
    """The channel `run_manylatents` needs: a caller holding the axis and only a matrix.

    Both halves are overridden together — passing one and letting the other be derived would
    pair genes with counts from two different objects, which is the misalignment this whole
    file is about."""
    from manyruns import pipeline

    seen: dict = {}

    def spy(state, g, params, out_dir, plots, target_dim):
        seen.update(counts=state.get("counts"), genes=state.get("genes"))

    _install(monkeypatch, spy=spy)
    mine = _counts(6, 2)
    pipeline.run_inproc(_adata(6, 4), _recipe("spy"), tmp_path,
                          counts=mine, genes=np.asarray(["X1", "X2"]))
    assert seen["counts"] is mine
    assert list(seen["genes"]) == ["X1", "X2"]


def test_a_supplied_counts_matrix_with_the_wrong_row_count_is_refused(tmp_path, monkeypatch):
    """`X` and `counts` are two matrices over the SAME cells. Off by one row and every gene a
    readout attributes goes to the wrong cell, with nothing downstream noticing — which is
    the exact sentence `aligned.check_aligned` exists to be able to say."""
    from manyruns import pipeline
    from manyruns.aligned import AlignmentError

    _install(monkeypatch, spy=lambda *a: None)
    with pytest.raises(AlignmentError):
        pipeline.run_inproc(np.zeros((6, 3)), _recipe("spy"), tmp_path,
                              counts=_counts(5, 4), genes=np.asarray(list("ABCD")))


# ── what the gene axis must NOT reach ────────────────────────────────────────
def test_the_gene_axis_is_never_written_to_disk(tmp_path, monkeypatch):
    """`artifacts.PERSISTED` excludes `X` because copying the scientist's data into an outputs
    folder is a data-handling decision (`artifacts.py`: "geometry leaves, the person's data
    does not"). `counts` is MORE of that data than `X` is, not less.

    The run below DOES write a state folder — the step assigns an embedding — so this is the
    difference between "nothing was persisted" and "the gene axis was not". Mutation-checked:
    adding `"counts"` to `artifacts.PERSISTED` fails this test.
    """
    from manyruns import artifacts, pipeline

    assert "counts" not in artifacts.PERSISTED and "genes" not in artifacts.PERSISTED
    _install(monkeypatch, embeds=_embeds)
    pipeline.run_inproc(np.zeros((6, 3)), _recipe("embeds"), tmp_path,
                          counts=_counts(6, 4), genes=np.asarray(list("ABCD")))
    written = sorted(p.name for p in (tmp_path / "state").rglob("*") if p.is_file())
    assert any(n.endswith("_emb.npy") for n in written), written   # the folder is real
    assert not [n for n in written if "counts" in n or "genes" in n], written


def test_the_gene_axis_never_reaches_the_record(tmp_path, monkeypatch):
    """A record is JSON and is what leaves the machine. Threading a 32,738-column expression
    matrix into state must not put one byte of it into the run store — the same line the
    federated design turns on."""
    from manyruns import pipeline

    _install(monkeypatch, embeds=_embeds)
    out = pipeline.run_inproc(np.zeros((6, 3)), _recipe("embeds"), tmp_path,
                                counts=_counts(6, 4), genes=np.asarray(list("ABCD")))
    assert "counts" not in out and "genes" not in out
    assert not [k for k in out["g_vector"] if "counts" in k or "gene" in k]
    json.dumps(out)  # raises TypeError if an array got in anywhere


# ── the schema change, stated ────────────────────────────────────────────────
def test_new_state_declares_every_key_it_writes():
    """`_STATE_KEYS` is a closed vocabulary and `counts`/`genes` are a deliberate addition to
    it. Nothing in the repo READ that tuple before this test — it was documentation, and
    documentation drifts: `label_kind` is written into `state` by both runners and by
    `session.py`, is read by `runner._ml_lightning`, and is still not declared there (the
    mock's `vec` is undeclared on purpose and says so). This pins the half that is
    enforceable — that `_new_state`, the one seeder both engines share, invents no key.

    Mutation-checked: removing `"counts"` from `_STATE_KEYS` fails this test.
    """
    from manyruns.pipeline.runner import _STATE_KEYS, _new_state

    state = _new_state()
    assert set(state) <= set(_STATE_KEYS)
    assert {"counts", "genes"} <= set(_STATE_KEYS)
    assert state["counts"] is None and state["genes"] is None  # absent by default, not empty


def test_both_loops_seed_the_gene_axis_through_the_same_function(tmp_path, monkeypatch):
    """One state schema, two loops — the divergence `runner`'s header exists to record.

    Two LOOPS, not two engines: since `engine=real` was removed, `manylatents` is the engine
    and `run_inproc` (`vocab.INPROC`) is the test substrate. They still share one state schema,
    which is the thing worth pinning.

    `run_manylatents` cannot DERIVE the axis (its `array` is already a matrix; the AnnData went
    out of scope inside `load_labeled`), so it takes it explicitly. What is pinned here is that
    when it is given one, it arrives in state identically to the in-process loop's. The fake
    `manylatents.api` is the convention `tests/test_manylatents_engine.py` sets, so this runs
    with no private stack.
    """
    import sys
    import types

    from manyruns import pipeline

    def fake_run(**kwargs):
        return {"embeddings": np.zeros((6, 2)), "scores": {}}

    api = types.ModuleType("manylatents.api")
    api.run = fake_run
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)

    seen: dict = {}

    def spy(state, g, params, out_dir, plots, target_dim):
        seen.update(counts=state.get("counts"), genes=state.get("genes"))

    monkeypatch.setitem(pipeline._ANALYSIS_STEPS, "spy", spy)
    mine = _counts(6, 4)
    pipeline.run_manylatents(_recipe("spy", group="analysis"), array=np.zeros((6, 3)),
                             out_dir=tmp_path, counts=mine,
                             genes=np.asarray(list("ABCD")))
    assert seen["counts"] is mine
    assert list(seen["genes"]) == list("ABCD")


# ── against the real library, where it is installed ──────────────────────────
def test_the_real_csr_behaves_like_the_stand_in():
    """`Sparse` is only worth testing against if `_dense` treats a real CSR the same way it
    treats the stand-in — a new object, every time. Skips where scipy is absent."""
    sp = pytest.importorskip("scipy.sparse")
    from manyruns.pipeline.loading import _dense, gene_axis

    csr = sp.csr_matrix(_counts(6, 4))
    assert _dense(csr) is not csr and _dense(Sparse(_counts(6, 4))) is not None
    counts, genes = gene_axis(AnnData(csr, list("ABCD")))
    assert counts is csr and hasattr(counts, "toarray")
    assert list(genes) == list("ABCD")


def test_the_stand_in_matches_a_real_anndata():
    """The dep-free tests above are only worth anything if the stand-in is faithful on the two
    attributes `gene_axis` reads. Skips where anndata is absent (CI)."""
    anndata = pytest.importorskip("anndata")
    from manyruns.pipeline.loading import gene_axis

    ad = anndata.AnnData(X=_counts(6, 4))
    ad.var_names = ["A", "B", "C", "D"]
    counts, genes = gene_axis(ad)
    assert counts is ad.X
    assert list(genes) == ["A", "B", "C", "D"]
    assert genes.shape[0] == ad.n_vars


def test_the_load_preserves_the_gene_axis_on_real_counts():
    """The measurement this whole file is a response to, re-derived rather than quoted — and
    RE-FRAMED, because the cutover flipped its sign.

    It was `test_the_preamble_destroys_the_gene_axis_on_real_counts`, and it asserted
    `matrix.shape[1] < ad.n_vars`: a counts-like AnnData went in, a PCA'd matrix with no gene
    axis came out of every route, and `gene_axis` returned the axis that had been lost. Its own
    docstring named the trigger for this edit — *"if the preamble is ever promoted to a declared
    step (canonical-recipes.md §2.5), THIS is the test to update"*. manyruns#54 is that
    promotion: `normalize` and `transform` are `prep` steps, `pca` is a `latent` step, and
    `_anndata_matrix` returns `adata.X`.

    The property is unchanged — WHAT A STEP IS HANDED MUST STILL CARRY THE GENE AXIS — so the
    same fixture is measured in the other direction: 200 columns in, 200 columns out (it was
    50 out, `PCA(n_comps=50)`'s width, on any input with more than 50 genes).

    Needs anndata only. `scanpy` was in the skip list because the preamble called it; nothing
    in this body does now, so dropping it moves this test off the one-laptop list that the
    module docstring warns about."""
    anndata = pytest.importorskip("anndata")
    from manyruns.pipeline import loading

    rng = np.random.default_rng(0)
    X = rng.poisson(3.0, size=(120, 200)).astype(float)
    ad = anndata.AnnData(X=X)
    ad.var_names = [f"G{i}" for i in range(200)]

    assert loading.looks_like_counts(ad.X) is True   # still counts-like: the discriminator stands
    matrix = loading._anndata_matrix(ad)
    assert matrix.shape[1] == ad.n_vars              # every gene reaches a step; was `< n_vars`
    assert matrix is ad.X                            # and reaches it as the stored frame, uncopied
    counts, genes = loading.gene_axis(ad)
    assert counts.shape == (120, 200) and genes.shape == (200,)
    assert list(genes[:3]) == ["G0", "G1", "G2"]
