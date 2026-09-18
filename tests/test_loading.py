"""`pipeline.loading` — how the loader declines what it cannot read.

`load_array` already refuses an unknown suffix in one readable sentence. This covers the
other half: a suffix it *does* support, over bytes it cannot parse.
"""


def test_pickled_npy_is_refused_before_its_callback_runs(tmp_path):
    """The front door must obey the data-only rule artifacts and figspec already enforce.

    A harmless write proves execution, not just an object dtype. Before the fix, pointing
    `load_array` at this file ran the callback and returned its result without a refusal.
    """
    from pathlib import Path

    import numpy as np
    import pytest

    from manyruns.pipeline.loading import load_array

    marker = tmp_path / "callback-ran"

    class Payload:
        def __reduce__(self):
            return Path.write_text, (marker, "executed")

    path = tmp_path / "objects.npy"
    np.save(path, np.array([Payload()], dtype=object))
    assert not marker.exists()

    with pytest.raises(ValueError) as err:
        load_array(path)

    assert not marker.exists(), "refusal must happen before unpickling"
    msg = str(err.value)
    assert str(path) in msg
    assert "refus" in msg.lower()
    assert "pickle" in msg.lower()
    assert "execute" in msg.lower()
    assert "numeric" in msg.lower()
    assert "allow_pickle=False" in msg
    assert isinstance(err.value.__cause__, ValueError)
    assert str(err.value.__cause__) != msg, "the refusal is ours, not numpy's raw error"


def test_numeric_npy_loads_unchanged(tmp_path):
    import numpy as np

    from manyruns.pipeline.loading import load_array

    expected = np.arange(12, dtype=np.int16).reshape(3, 4)
    path = tmp_path / "numeric.npy"
    np.save(path, expected, allow_pickle=False)

    actual = load_array(path)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_placeholder_h5ad_refuses_with_a_reason(tmp_path):
    """A .h5ad suffix over non-HDF5 bytes declines legibly instead of raising raw h5py.

    This is the front-door case: `manyruns init ./data` on a placeholder, a truncated
    download, or an unfetched git-LFS pointer. It used to surface as `OSError: Unable to
    synchronously open file (file signature not found)` — which names neither the file nor
    the cause — through a traceback in three libraries."""
    import pytest

    pytest.importorskip("anndata")
    from manyruns.pipeline.loading import load_array

    bad = tmp_path / "matrix.h5ad"
    bad.write_text("placeholder")

    with pytest.raises(ValueError) as err:
        load_array(bad)

    msg = str(err.value)
    assert "matrix.h5ad" in msg          # which file
    assert "11 bytes" in msg             # the tell: far too small to be a matrix
    assert "git-LFS" in msg              # and the likely causes
    # the original is preserved rather than swallowed — the h5py detail is still reachable
    assert isinstance(err.value.__cause__, OSError)


def test_front_door_reports_the_message_not_a_traceback(tmp_path, monkeypatch, capsys):
    """A user-facing `ValueError` exits 1 with the sentence, not a stack.

    `main` caught only RuntimeError, so every `ValueError` the product raises to explain
    itself — `catalog`'s "Known: ...", the loader's byte count, `vocab`'s legal topologies —
    arrived as a traceback with the human-written line buried at the bottom.

    An unknown recipe is the vehicle because it refuses identically on every machine. The
    unreadable-file case above reaches this same handler, but only under `--engine
    manylatents` (the one backend that loads eagerly), so testing it here would make the
    result depend on whether the private stack happens to be installed — which is what put
    four tests in this suite into the state that prompted the fix."""
    import pytest

    from manyruns import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_TRACEBACK", raising=False)
    data = tmp_path / "data"
    data.mkdir()
    (data / "matrix.h5ad").write_text("placeholder")
    argv = ["init", str(data), "--project", "p", "--engine", "mock", "--recipe", "nonesuch"]

    rc = app.main(argv)

    assert rc == 1
    err = capsys.readouterr().err
    assert "no recipe named 'nonesuch'" in err
    assert "Known:" in err          # the part that tells the user what to type instead

    # …and the stack is one env var away, for when the message reads like an internal bug
    monkeypatch.setenv("MANYRUNS_TRACEBACK", "1")
    with pytest.raises(ValueError):
        app.main(argv)


# ── the cutover: loading hands over the frame (#54) ──────────────────────────────────────
def test_loading_returns_the_gene_axis_rather_than_a_hidden_pca():
    """The gap this whole change exists to close.

    `_anndata_matrix`'s own docstring called the preamble UNDECLARED — "not a recipe step, so it
    appears in no trace, no g-vector and no caveat, and `n_comps` is not settable from a recipe"
    — and it returned `obsm["X_pca"]` at 50 columns from a 32,738-gene file. Three operations
    (`normalize_total(1e4)` → `log1p` → `PCA(50)`) ran on every counts-like load, before the
    runner saw anything, and no surface could name them.

    What runs is now what the recipe declares.
    """
    import pytest

    ad = pytest.importorskip("anndata")
    import numpy as np

    from manyruns.pipeline import loading

    counts = np.random.default_rng(0).poisson(3, size=(20, 60)).astype(np.float32)
    adata = ad.AnnData(counts)
    adata.var_names = [f"g{i}" for i in range(60)]

    out = loading._anndata_matrix(adata)

    assert out.shape == (20, 60), "every gene survives the load"
    assert np.allclose(loading._dense(out), counts), "and is untransformed"


def test_the_loader_takes_no_transform_argument_at_all():
    """The parameter is GONE, not defaulted. A surviving `transform=` would be a second place to
    say what `configs/recipe/*.yaml`'s `transform` step says — the drift `vocab.py` refuses
    twice — and the one that wins would depend on which entry point the user came through."""
    import inspect

    from manyruns.pipeline import loading

    assert "transform" not in inspect.signature(loading._anndata_matrix).parameters
    assert "transform" not in inspect.signature(loading.load_labeled).parameters


def test_the_transform_flag_reaches_the_declared_step_from_both_front_doors():
    """`--transform` survived the cutover as a SHIM over the `transform` step's `method`, and a
    shim with no caller is worse than no shim: the flag still parses, still shows in `--help`,
    and silently does nothing.

    Measured after the cutover's first pass: `app.apply_transform_shim` had **zero** callers
    anywhere in the repo, and `tui/app.py` was still passing `transform=` to `explore_once`,
    whose parameter had been removed — so the shipped no-arg `manyruns` TUI raised
    `TypeError: explore_once() got an unexpected keyword argument 'transform'` on its worker
    thread before any step ran, leaving no row in the run index. Both front doors are pinned
    here because they are separate call sites and only one of them was covered by a test.
    """
    import inspect

    from manyruns import app, shell
    from manyruns.tui import app as tui_app

    assert "transform" in inspect.signature(app.explore_once).parameters
    # The two FORWARDING call sites. `app` itself is neither — it defines `explore_once` and
    # applies the shim inside it, which is the point: one seam, so a third front door added later
    # inherits the behaviour instead of having to remember the flag.
    for module in (shell, tui_app):
        src = inspect.getsource(module)
        assert 'transform=getattr(args, "transform", None)' in src, (
            f"{module.__name__} does not forward --transform to explore_once")
    assert "apply_transform_shim(recipe, transform)" in inspect.getsource(app.explore_once)

    recipe = {"name": "r", "steps": [
        {"name": "normalize", "group": "prep", "params": {"target_sum": 1e4}},
        {"name": "transform", "group": "prep", "params": {"method": "log1p"}}]}

    assert app.apply_transform_shim(recipe, "sqrt")["steps"][1]["params"]["method"] == "sqrt"
    assert app.apply_transform_shim(recipe, "sqrt")["steps"][0]["params"] == {"target_sum": 1e4}
    # None means "not typed" and must leave the recipe's own declaration alone — the reason the
    # flag's default moved off "log1p", which would have overridden every `sqrt` recipe silently.
    assert app.apply_transform_shim(recipe, None) is recipe


def test_numpy_wraps_a_sparse_matrix_instead_of_converting_it():
    """The trap `loading.dense` exists for, pinned as the MEASUREMENT rather than the fix.

    `np.asarray`/`np.ascontiguousarray` on a scipy sparse matrix produce `shape=(1,),
    dtype=object` — they wrap it, and nothing raises. This shipped twice: upstream in
    manylatents-omics#61, where `normalize` returned a 0-d object array on sparse input, and here
    when the cutover made `_anndata_matrix` return `.X` as stored and the densification that used
    to happen inside the removed preamble went with it. On pbmc3k that is what reached the engine.

    Asserting numpy's behaviour and not just ours is deliberate: the day `np.asarray` learns to
    densify, this fails and `dense` becomes deletable. Until then it is load-bearing.
    """
    import numpy as np
    import pytest

    sp = pytest.importorskip("scipy.sparse")
    from manyruns.pipeline import loading

    csr = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]]))

    assert np.ascontiguousarray(csr).shape == (1,), "numpy still wraps rather than converts"
    assert loading.dense(csr).shape == (2, 3)
    assert np.allclose(loading.dense(csr), [[1, 0, 2], [0, 3, 0]])
    # and the two callers that hand data to an engine must not reintroduce it
    assert loading.as_matrix(csr).shape == (2, 3)


def test_the_loader_hands_on_the_gene_axis_it_read(tmp_path):
    """`runner.run_manylatents` has carried `counts=`/`genes=` since the frame landed, with its
    own docstring reading "MEASURED GAP: nothing populates these yet". This is that producer.

    Without it the calculus said `preprocess`/`qc`/`markers` were legal on `pbmc3k` at PLAN time
    while `prep._genes_or_refuse` refused them at RUN time — the worst of both, and it made the
    `STEP_NEEDS` commit message ("the demand is satisfiable") false on every product path.

    A temporary annotated matrix verifies the actual channels without a downloaded fixture.
    """
    import anndata
    import numpy as np

    from manyruns import app

    # A generator loaded inside the engine carries no local gene axis.
    assert app._load_inputs("manylatents", None, "swissroll", None) == app._EMPTY_INPUTS
    obj = anndata.AnnData(np.arange(12, dtype=np.float32).reshape(4, 3))
    obj.var_names = ['gene-a', 'gene-b', 'gene-c']
    obj.layers['requested'] = obj.X.copy()
    obj.layers['unused'] = obj.X.copy()
    obj.obs['score'] = np.arange(4, dtype=float)
    path = tmp_path / 'counts.h5ad'
    obj.write_h5ad(path)
    loaded = app._load_inputs('manylatents', path, None, None, ('requested',), color_by=['score'])
    np.testing.assert_array_equal(loaded['counts'], obj.X)
    np.testing.assert_array_equal(loaded['genes'], obj.var_names)
    assert set(loaded['layers']) == {'requested'}
    np.testing.assert_array_equal(loaded['layers']['requested'], obj.X)
    assert [c['key'] for c in loaded['color']] == ['score']


def test_an_identity_column_is_a_group_axis_and_not_a_time_axis():
    """`dla_tree` ships eight named branches and `pbmc3k_processed` ships cell types; both
    are an unordered IDENTITY per row. Before this they reached no plot at all — `labels_of`
    knew two kinds and neither fit — so every embedding was drawn grey and nothing could say
    which cells moved."""
    import anndata
    import numpy as np
    import pandas as pd

    from manyruns.pipeline import loading

    a = anndata.AnnData(np.zeros((4, 3)),
                        obs=pd.DataFrame({"branch": ["b1", "b1", "b2", "b2"]}))
    labels, kind = loading.labels_of(a)
    assert kind == "group"
    assert list(labels) == ["b1", "b1", "b2", "b2"]


def test_time_and_condition_still_win_over_a_group_column():
    """Order is the whole safety property: a file carrying BOTH a timepoint and a cell type
    is a trajectory dataset, and demoting it to a colouring would silently stop MIOFlow from
    seeing its time axis."""
    import anndata
    import numpy as np
    import pandas as pd

    from manyruns.pipeline import loading

    both = anndata.AnnData(np.zeros((2, 3)), obs=pd.DataFrame(
        {"timepoint": ["d0", "d3"], "cell_type": ["T", "B"]}))
    assert loading.labels_of(both)[1] == "time"

    cond = anndata.AnnData(np.zeros((2, 3)), obs=pd.DataFrame(
        {"disease": ["treated", "healthy"], "cell_type": ["T", "B"]}))
    assert loading.labels_of(cond)[1] == "condition"


def test_group_is_not_a_synonym_for_the_column_named_group():
    """`"group"` is already in `CONDITION_KEYS` — an unordered case/control axis. The KIND
    named `group` is a different thing (an identity per row), and listing the column name in
    both tables would make one shadow the other depending on which is checked first."""
    from manyruns import vocab

    assert "group" in vocab.CONDITION_KEYS
    assert "group" not in vocab.GROUP_KEYS
