"""Loading: a path, a name, or a generator script → an array (+ labels).

Every reader manyruns owns lives here. This is the module a future `sources.resolve()`
absorbs wholesale — it is deliberately the seam, not a permanent home.
"""
from __future__ import annotations

from manyruns.installation import REINSTALL

from pathlib import Path
from typing import Any, Iterable

from manyruns.aligned import check_aligned
from manyruns.pipeline.io import quiet  # noqa: F401 - re-exported for callers

# extensions we know how to load directly (the `.py` case is handled separately)
_TABLE_SUFFIXES = {".csv", ".tsv", ".txt"}
_GENERATOR_ENTRYPOINTS = ("load", "generate", "make", "get_data", "sample", "data", "main")
_GENERATOR_ATTRS = ("X", "data", "DATA", "dataset")


# ── the synthetic time course ────────────────────────────────────────────────
#
# The one bundled dataset that carries a real time axis. Measured 2026-07-28: the other 12
# declare `shape: manifold` or `clusters`, and `vocab.SHAPE_PROVIDES` maps `time-course →
# ("time",)`, so before this existed MIOFlow could only ever run on the pseudotime fallback
# in `steps._step_mioflow` — an ordering derived FROM the embedding, which returns a full
# [0, 1] range on data with no time in it.
#
# TIME IS A GENERATIVE PARAMETER. Each snapshot is `randn(n, d) + drift * t` carrying the
# label `t` — the construction manylatents' own MIOFlow tests use — so the timepoint is an
# ARGUMENT to the generator and never a quantity read back out of the geometry. That is the
# categorical difference from pseudotime, and it is what this module exists to keep.
#
# It demonstrates PLUMBING, NOT BIOLOGY: three Gaussian clouds translated along a line.

#: Per-timepoint translation. Chosen so the snapshots OVERLAP rather than separate — a drift
#: large enough to make three clusters would be testing clustering, not a time axis.
TIME_COURSE_DRIFT = 3.0
TIME_COURSE_TIMEPOINTS = (0.0, 0.5, 1.0)
TIME_COURSE_N_PER_TIME = 100
TIME_COURSE_DIM = 5
#: Fixed, so a g-vector measured on this dataset is comparable across runs and machines.
TIME_COURSE_SEED = 42

#: What `handle.ref` resolves to. A generated ref is not a path, and is checked BEFORE
#: `p.exists()` so a typo reports the refs that exist rather than sending you to look on disk
#: for something that was never going to be there.
TIME_COURSE_REF = "synthetic:time-course"


def time_course_arrays():
    """``(X, t)`` — the matrix and its per-row timepoint, both generated from `t`.

    Deterministic: a fresh generator seeded with :data:`TIME_COURSE_SEED` every call, so two
    runs are byte-identical and a measurement on this dataset means the same thing twice."""
    import numpy as np

    rng = np.random.default_rng(TIME_COURSE_SEED)
    blocks = [
        rng.standard_normal((TIME_COURSE_N_PER_TIME, TIME_COURSE_DIM)) + TIME_COURSE_DRIFT * t
        for t in TIME_COURSE_TIMEPOINTS
    ]
    X = np.vstack(blocks)
    t = np.repeat(np.asarray(TIME_COURSE_TIMEPOINTS, dtype=float), TIME_COURSE_N_PER_TIME)
    return X, t


def time_course_anndata():
    """The same data as an AnnData whose `.obs` carries a `timepoint` column.

    AnnData rather than an `(X, labels)` tuple on purpose: `load_labeled`'s non-AnnData branch
    returns `(as_matrix(obj), None, None)`, so a tuple would drop the time axis WITHOUT an
    error and MIOFlow would fall back to pseudotime on a dataset declaring `shape:
    time-course`. The column is named from the SHARED obs vocabulary (`vocab.TIME_KEYS`), not
    by a rule private to this generator — a second private time vocabulary is the exact
    mistake `vocab`'s docstring records."""
    import anndata
    import pandas as pd

    X, t = time_course_arrays()
    obs = pd.DataFrame({"timepoint": t}, index=[str(i) for i in range(X.shape[0])])
    return anndata.AnnData(X=X, obs=obs)


#: Generated refs, by name. A mapping rather than an if-chain so the refusal below can list
#: what actually exists.
_GENERATED = {TIME_COURSE_REF: time_course_anndata}


def _generated(p: Path):
    """The builder for a `synthetic:` ref, or None when `p` is an ordinary path.

    Raises on an unknown `synthetic:` ref rather than falling through to the filesystem, so
    `synthetic:tiem-course` says which refs exist instead of "data path not found"."""
    ref = str(p)
    if not ref.startswith("synthetic:"):
        return None
    builder = _GENERATED.get(ref)
    if builder is None:
        raise ValueError(
            f"unknown generated dataset {ref!r}. Known: {', '.join(sorted(_GENERATED))}"
        )
    return builder


# ── loading ──────────────────────────────────────────────────────────────────
def load_array(path: Path, *, allow_scripts: bool = True) -> Any:
    """Load raw data from a file path. Returns whatever the source yields (ndarray,
    DataFrame, AnnData, tuple, list); use :func:`as_matrix` to get a 2-D float array.

    Display metadata reads set ``allow_scripts=False`` to refuse fresh Python generator
    executions, including scripts selected within folders. Analysis keeps the default.
    """
    p = Path(path)
    # Checked before `exists()`: a generated ref is not a path, and a typo in one must not be
    # reported as a missing file.
    builder = _generated(p)
    if builder is not None:
        return builder()
    if not p.exists():
        raise ValueError(f"data path not found: {p}")
    if p.is_dir():
        # 10x Cell Ranger output: a filtered_feature_bc_matrix/ dir (matrix.mtx[.gz] +
        # barcodes + features) — read as one AnnData rather than a bare .mtx.
        if any(p.glob("matrix.mtx*")):
            import scanpy as sc

            return sc.read_10x_mtx(p)
        # a directory of per-sample sub-folders (10x timepoints) → concat into one AnnData
        samples = [
            d for d in sorted(p.iterdir())
            if d.is_dir() and (any(d.glob("matrix.mtx*")) or list(d.glob("*.h5ad")) or list(d.glob("*.h5")))
        ]
        if samples:
            import anndata

            adata = anndata.concat([load_array(d, allow_scripts=allow_scripts) for d in samples],
                                   join="outer")
            adata.obs_names_make_unique()
            return adata
        loadable = _TABLE_SUFFIXES | {".py", ".npy", ".h5ad", ".mtx"}
        cands = [f for f in sorted(p.iterdir()) if f.suffix.lower() in loadable]
        if not cands:
            raise ValueError(f"no loadable data file found in {p} (looked for {sorted(loadable)})")
        p = cands[0]
    suffix = p.suffix.lower()
    if suffix == ".py":
        if not allow_scripts:
            raise ValueError("Python generator metadata cannot be read after the run "
                             "because the original metadata was not saved.")
        return _load_from_script(p)
    if suffix in _TABLE_SUFFIXES:
        return _load_table(p, sep="\t" if suffix in (".tsv", ".txt") else ",")
    if suffix == ".npy":
        import numpy as np

        # THE RULE ALREADY EXISTS: artifacts.load_array says "an artifact is data, and a
        # `.npy` that can execute on load is a code path from the filesystem into the
        # process"; figspec.load disables pickle too. The path a person actually supplies
        # must obey the same rule. Nothing is shipped yet, so there are no existing users'
        # object arrays to preserve: refuse them, without conversion or a pickle fallback.
        try:
            return np.load(p, allow_pickle=False)
        except ValueError as exc:
            # NumPy uses ValueError for malformed headers too. Only its object-array
            # refusal establishes that pickle is required; preserve other failures.
            if str(exc) != "Object arrays cannot be loaded when allow_pickle=False":
                raise
            raise ValueError(
                f"refusing {p}: this .npy contains an object array that requires pickle, "
                "which can execute code when loaded. Export a numeric array from the "
                "original data source and save it with np.save(..., allow_pickle=False)."
            ) from exc
    if suffix == ".h5ad":
        import anndata  # noqa: PLC0415 - optional, lazy

        return _read_hdf5(p, lambda: anndata.read_h5ad(p))
    if suffix == ".h5":  # 10x Cell Ranger filtered_feature_bc_matrix.h5
        import scanpy as sc

        return _read_hdf5(p, lambda: sc.read_10x_h5(str(p)))
    if suffix == ".mtx":
        import numpy as np
        import scipy.io

        return np.asarray(scipy.io.mmread(p).todense())
    raise ValueError(
        f"don't know how to load {p.name} (suffix {suffix!r}). Supported: .py generator, "
        f"{sorted(_TABLE_SUFFIXES)}, .npy, .h5ad, .h5 (10x), .mtx, or a 10x matrix dir"
    )


def _read_hdf5(path: Path, read: Any) -> Any:
    """Run an HDF5-backed reader, and give a container-level failure this module's error shape.

    h5py refuses a bad file with ``OSError: Unable to synchronously open file (file signature
    not found)`` — no path, no cause, and a traceback through three libraries. The
    unsupported-suffix branch of `load_array` already declines in one readable sentence; a
    file with the right suffix and the wrong bytes deserves the same, and is the likelier
    mistake: a placeholder, a truncated download, or an unfetched git-LFS pointer. The byte
    count is included because it usually settles which one it is on sight.

    Reachable from the front door — `manyruns init ./data` on such a file raised the bare
    h5py error until this existed. Only the container is guarded: a genuine schema problem
    inside a valid HDF5 file is anndata's to report, and wrapping it would hide the detail."""
    try:
        return read()
    except OSError as exc:
        size = path.stat().st_size if path.exists() else 0
        raise ValueError(
            f"{path.name} has a {path.suffix} suffix but is not readable as HDF5: {exc}. "
            f"The file is {size} bytes — check for a placeholder, a truncated download, or "
            f"a git-LFS pointer that was never fetched."
        ) from exc


def _load_from_script(path: Path) -> Any:
    """Import a generator ``.py`` by path and pull the data out of it.

    Convention (first match wins): a no-arg callable named one of
    ``load/generate/make/get_data/sample/data/main``, else a module attribute
    ``X/data/DATA/dataset``. Keeps generators like ``swissroll.py`` first-class."""
    import importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not import generator script: {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        # The file uses package-relative imports (e.g. `from .synthetic_dataset import …`),
        # so it must be imported through its package. Reconstruct the dotted name by
        # walking up the `__init__.py` chain and import it properly.
        mod = _import_as_package_member(path)

    for name in _GENERATOR_ENTRYPOINTS:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn()
    for name in _GENERATOR_ATTRS:
        if hasattr(mod, name):
            val = getattr(mod, name)
            if not callable(val):
                return val
    # manylatents style: a LightningDataModule / torch Dataset class defined in the
    # module (e.g. SwissRollDataModule). Instantiate with defaults; as_matrix extracts.
    cls = _find_data_class(mod)
    if cls is not None:
        try:
            return cls()
        except Exception as e:  # noqa: BLE001 - report which class + why
            raise ValueError(
                f"found data class {cls.__name__} in {path.name} but couldn't instantiate "
                f"it with defaults: {type(e).__name__}: {e}"
            ) from e
    public = [n for n in dir(mod) if not n.startswith("_")]
    raise ValueError(
        f"{path.name} has no recognized data entrypoint. Expose a no-arg function named "
        f"one of {_GENERATOR_ENTRYPOINTS} that returns the data, a module-level array "
        f"named one of {_GENERATOR_ATTRS}, or a DataModule/Dataset class. Module defines: {public}"
    )


def _find_data_class(mod):
    """Find a DataModule-/Dataset-like class DEFINED in this module (duck-typed, so no
    hard torch/lightning dependency). Prefer a DataModule (it wires the dataset)."""
    import inspect

    own = [
        c for _, c in inspect.getmembers(mod, inspect.isclass)
        if getattr(c, "__module__", None) == mod.__name__
    ]
    for c in own:  # DataModule-like: builds a dataset via setup()
        if hasattr(c, "setup") and (hasattr(c, "train_dataloader") or hasattr(c, "train_dataset")):
            return c
    for c in own:  # plain Dataset-like
        if hasattr(c, "__getitem__") and hasattr(c, "__len__"):
            return c
    return None


def _import_as_package_member(path: Path):
    """Import a `.py` that lives inside a package (has package-relative imports).

    Walks up the `__init__.py` chain to reconstruct the dotted module name, puts the
    package's parent on sys.path *for the duration of the import only*, and imports it so
    relative imports resolve.

    The path entry is REMOVED afterwards. It used to be inserted and left there, so a
    worker loading several generator-script datasets grew `sys.path` without bound and
    could shadow modules for every later run in that process — which the planned
    cross-product harness (many datasets per pooled worker) would hit immediately."""
    import importlib
    import sys

    parts = [path.stem]
    d = path.parent
    while (d / "__init__.py").exists():
        parts.insert(0, d.name)
        d = d.parent
    if len(parts) == 1:
        raise ValueError(
            f"{path.name} uses package-relative imports but isn't inside a package "
            f"(no __init__.py alongside it). Point at the file in its installed package "
            f"(e.g. the copy under site-packages), or give a self-contained data file."
        )
    entry = str(d)
    added = entry not in sys.path
    if added:
        sys.path.insert(0, entry)
    try:
        return importlib.import_module(".".join(parts))
    finally:
        if added:
            try:
                sys.path.remove(entry)
            except ValueError:  # something else already removed it
                pass


def _load_table(path: Path, sep: str) -> Any:
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            return _read_table(source, sep)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {path} as a numeric matrix: {exc}") from exc


def _read_table(source: Any, sep: str) -> Any:
    """Read CSV or tab-delimited TSV/TXT, including bounded inspection previews.

    Skip leading blank lines as pandas does: spaces and (for CSV) tabs are whitespace;
    a TSV/TXT tab is a field delimiter. Quoted whitespace is a record, not a blank line.
    An entirely numeric first record is data (headerless); otherwise it is the header.
    Non-numeric columns are metadata and omitted. Numeric columns, including an index-like
    first column, are retained: guessing which numbers are identifiers would lose features.
    Numeric column names are therefore ambiguous and treated as a data row.
    """
    import csv
    import warnings
    import pandas as pd

    blank_chars = " \r\n" + ("\t" if sep != "\t" else "")
    while True:
        position = source.tell()
        line = source.readline()
        if not line or line.strip(blank_chars):
            source.seek(position)
            break
    try:
        first = next(csv.reader(source, delimiter=sep, strict=True), [])
    except csv.Error as exc:
        raise ValueError(f"invalid table: {exc}") from exc
    source.seek(0)
    try:
        for value in first:
            float(value)
        header = None
    except ValueError:
        header = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            table = pd.read_csv(source, sep=sep, header=header, index_col=False)
    except pd.errors.ParserWarning as exc:
        raise ValueError(f"invalid table: {exc}") from exc
    # Validate here too: inspection and execution must refuse the same unusable tables.
    as_matrix(table)
    return table


def load_named_dataset(name: str) -> Any:
    """Load a built-in dataset by NAME from manylatents (e.g. ``swissroll``).

    Delegates to manylatents' own registry. This used to re-implement the lookup by walking
    packages for a matching MODULE name, while manylatents resolves CLASS-name variants —
    two resolvers, two answers: `gaussian_blob` and `hf_text` loaded under the engine and
    raised here, so `--dataset gaussian_blob` failed on the in-process loader while the
    engine's own resolver found it. The engine owns its catalogue; we ask it."""
    try:
        from manylatents.data import get_datamodule
    except ImportError as e:
        raise RuntimeError(
            f"--dataset {name!r} needs manylatents, which is a BASE dependency — so this "
            "install is incomplete, not missing an extra. Repair it with `uv sync` in a "
            f"checkout, or `{REINSTALL}` for an installed tool. "
            f"(`[datasets]` carries nothing since it was emptied; adding it fixes nothing.) ({e})"
        ) from e

    try:
        return get_datamodule(name)
    except Exception as e:  # noqa: BLE001 - surface the engine's own vocabulary in the error
        raise RuntimeError(f"manylatents has no dataset named {name!r}: {e}") from e



def as_matrix(obj: Any) -> Any:
    """Coerce a loaded object to a 2-D float numpy array (rows = samples).

    Handles ndarray / DataFrame / AnnData / torch Tensor, (X, labels) tuples, dicts with
    a ``data`` key, and — for manylatents — LightningDataModule and torch Dataset objects
    (runs ``setup()`` and stacks each sample's ``data`` field)."""
    import numpy as np

    obj = _resolve_data_object(obj)
    if hasattr(obj, "select_dtypes") and hasattr(obj, "to_numpy"):
        from pandas.api.types import infer_dtype

        # Numeric/boolean values are features even in object columns; text (including
        # numeric strings or mixed text/numbers) stays metadata. Preserve column order.
        numeric_dtypes = set(obj.select_dtypes(["number", "bool"]).dtypes)
        numeric = [i for i, (_, column) in enumerate(obj.items())
                   if column.dtype in numeric_dtypes or (
                       column.dtype == object and infer_dtype(column, skipna=True) in {
                           "integer", "floating", "mixed-integer-float", "boolean",
                           "decimal", "complex"})]
        obj = obj.iloc[:, numeric]
        if not obj.shape[0] or not obj.shape[1]:
            raise ValueError("table has no rows of data or no numeric features")
        # Refuse missing values explicitly, including pd.NA in nullable/object columns.
        # Convert to floats before isfinite: NumPy cannot check pandas' object arrays.
        if obj.isna().to_numpy().any():
            raise ValueError("numeric table columns contain missing or non-finite values")
        try:
            obj = obj.to_numpy(dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"numeric table columns cannot be converted to floats: {exc}") from exc
        if not np.isfinite(obj).all():
            raise ValueError("numeric table columns contain missing or non-finite values")

    # `dense` FIRST: since the cutover `_resolve_data_object` can hand back an AnnData's `.X` as
    # stored, and `np.asarray` on a CSR wraps rather than converts. This function's contract is
    # "a 2-D float numpy array", so the conversion belongs here — see `dense`.
    arr = np.asarray(dense(obj), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D data (samples × features), got shape {arr.shape}")
    return arr


def _resolve_data_object(obj: Any) -> Any:
    """Unwrap a loaded object down to an array-like (numpy/DataFrame/list)."""
    tname, tmod = type(obj).__name__, (type(obj).__module__ or "")
    # torch Tensor (has detach+numpy; ndarrays don't have detach)
    if hasattr(obj, "detach") and hasattr(obj, "numpy"):
        return obj.detach().cpu().numpy()
    # AnnData → preprocessed matrix (scRNA: normalize → log1p → PCA), else raw .X
    if tname == "AnnData" or "anndata" in tmod:
        return _anndata_matrix(obj)
    # LightningDataModule-like → setup(), then stack the train dataset
    if hasattr(obj, "setup") and (hasattr(obj, "train_dataloader") or hasattr(obj, "train_dataset")):
        try:
            obj.setup()
        except TypeError:
            obj.setup(stage=None)  # some setups require the stage arg
        ds = getattr(obj, "train_dataset", None) or getattr(obj, "dataset", None)
        return _stack_dataset(ds)
    # torch Dataset-like → stack samples (exclude plain containers/strings/arrays)
    #
    # A SPARSE MATRIX IS EXCLUDED BY `toarray`, and it has to be: scipy's classes carry both
    # `__getitem__` and `__len__`, so before the cutover made sparse actually flow through here
    # this branch claimed one and `_stack_dataset` walked it row by row — `len(csr)` itself
    # raises `TypeError: sparse array length is ambiguous`. Duck-typed on `toarray` rather than
    # importing scipy, which this module keeps optional.
    if (
        hasattr(obj, "__getitem__")
        and hasattr(obj, "__len__")
        and not isinstance(obj, (list, tuple, dict, str, bytes))
        and tname != "ndarray"
        and not hasattr(obj, "toarray")
        # DataFrame, Series, Index and ndarray subclasses implement the array protocol.
        # Their integer keys need not be row positions; they are not torch Datasets.
        and not hasattr(obj, "__array__")
    ):
        return _stack_dataset(obj)
    # (X, labels) tuple/list → feature matrix is the first element
    if isinstance(obj, (tuple, list)) and obj and hasattr(obj[0], "__len__"):
        return obj[0]
    # a single sample dict {'data': ...}
    if isinstance(obj, dict) and "data" in obj:
        return _resolve_data_object(obj["data"])
    return obj


def dense(x: Any) -> Any:
    """A matrix as a dense ndarray — the ONE safe way to leave sparse behind.

    **`np.asarray` DOES NOT CONVERT A SPARSE MATRIX, IT WRAPS ONE**, which is why this is public
    and named rather than inlined. Measured in this checkout: `np.ascontiguousarray(csr)` returns
    `shape=(1,), dtype=object`. Nothing raises, so the caller hands an engine a one-element object
    array and finds out several layers down, or never.

    That is not hypothetical — it shipped twice. Once upstream (manylatents-omics#61, where
    `normalize` returned a 0-d object array on sparse input), and once here: the cutover made
    `_anndata_matrix` return `.X` as stored, and the densification that used to happen inside the
    removed preamble went with it.

    WHERE TO CALL IT: at the boundary where something genuinely needs dense — a torch tensor, an
    engine's `input_data` — and NOT at load. Densifying pbmc3k costs 353.6 MB from an 18.3 MB CSR;
    every prep filter accepts sparse, and `filter_genes` drops 19,024 of 32,738 columns, so a
    densification deferred past prep is a far cheaper one.
    """
    import numpy as np

    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


#: The private name this had before the cutover gave it callers outside this module.
_dense = dense


def looks_like_counts(X: Any) -> bool:
    """Does this matrix hold integer counts? — ONE discriminator, now with two readers.

    It was inline in :func:`_anndata_matrix`, where it decides whether to run the scRNA
    preamble (normalize_total → log1p → PCA). :func:`gene_axis` hands a downstream gene-level
    step that same `.X` UNTRANSFORMED, so that step has to ask the identical question before
    it normalizes — and a second copy of the test is exactly what `vocab.py`'s docstring
    records as drifting in silence. Extracted rather than re-written: the body below is the
    original expression, so `_anndata_matrix`'s branch is unchanged.

    Reads the first 100 rows only, and that is not laziness: `_dense` on all of pbmc3k
    materialises 353.6 MB from an 18.3 MB CSR (measured, `data/pbmc3k_raw.h5ad`,
    2700 × 32738 float32) to answer a yes/no question.

    Answers False for anything it cannot inspect — the conservative direction, because a
    False leaves `.X` alone rather than transforming something it did not understand."""
    import numpy as np

    try:
        sample = _dense(X[:100])
        return bool(np.all(sample >= 0) and np.allclose(sample, np.round(sample)))
    except Exception:  # noqa: BLE001 - detection is best-effort
        return False


def gene_axis(obj: Any) -> tuple:
    """``(counts, genes)`` — the untransformed expression matrix and the name of each of its
    COLUMNS — or ``(None, None)`` for anything that has no gene axis.

    THE GAP THIS CLOSES, measured on `data/pbmc3k_raw.h5ad` in this checkout: the AnnData is
    `(2700, 32738)` with `var_names[:3] == ['MIR1302-10', 'FAM138A', 'OR4F5']`, and every
    route out of this module returns `(2700, 50)` — `_anndata_matrix`, `as_matrix` and
    `load_labeled` alike, because the counts branch of `_anndata_matrix` returns
    `obsm["X_pca"]`. The 32,738-column gene axis is gone, and the names were never returned
    by anything here even on the branch that keeps the columns (`_dense(X)`), so a step
    asking "which genes distinguish these groups" has neither half of the answer.
    `steps._step_composition`'s docstring has said so for as long as it has existed: *"it
    needs the original expression matrix + gene names threaded through, which this pipeline
    doesn't yet carry"*.

    **Both or neither, from ONE object.** They are returned together because they are two
    halves of one fact and fetching them separately is how they come to disagree — the same
    argument :func:`labels_of` makes for rows, one axis over. The column check below is that
    argument enforced; `anndata` guarantees `len(var_names) == n_vars`, so it can only fire
    for a duck-typed stand-in, which is precisely the case no other check covers.

    **Untransformed, and never densified.** `counts` is `.X` as stored — the same object,
    not a copy (`state["counts"] is adata.X`). On pbmc3k that is an 18.3 MB CSR; `_dense`
    would make it 353.6 MB, a 19× copy of the scientist's data made silently on their behalf.
    Whether it needs normalizing is the caller's question, and :func:`looks_like_counts` is
    the answer — this function does not decide it, because deciding it here is how the
    preamble became invisible in the first place.

    `var_names` rather than `var['gene_ids']`: it is the index, so anndata guarantees exactly
    one per column, while `gene_ids` is a per-dataset column that may not exist. `np.asarray`
    rather than a `str()` comprehension — anndata already coerces the index to str, and the
    comprehension measured 2.5 MB / 4.6 ms against 262 KB / ~0 ms for the same 32,738 names."""
    import numpy as np

    if not _is_anndata(obj):
        return None, None
    counts = getattr(obj, "X", None)
    if counts is None or getattr(counts, "ndim", 0) != 2:
        return None, None
    genes = np.asarray(obj.var_names)
    if genes.shape[0] != counts.shape[1]:
        raise ValueError(
            f"gene axis misalignment: the matrix has {counts.shape[1]} columns but "
            f"{genes.shape[0]} gene names were supplied for them. Off by even one and every "
            f"gene a downstream readout reports is the wrong gene, with nothing to notice it."
        )
    return counts, genes


def layers_of(obj: Any, want: "Iterable[str] | None" = None) -> dict:
    """`{name: matrix}` for the layers ASKED FOR, or `{}`. Never "all of them".

    DECLARED-ONLY, and the alternative was considered and refused. An `.h5ad` may carry a dozen
    layers; loading every one makes memory a property of the file rather than of the analysis,
    and there is no threshold anybody can pick well. So the caller says what it needs — the same
    shape `metrics` already has, where manyruns declares which it wants and the loader supplies
    exactly that.

    A name that is asked for and absent is simply not returned. It is not an error HERE because
    this function cannot know whether the step that wanted it is optional; `vocab.unmet` is
    where a missing `splicing` becomes a refusal, at plan time, with a sentence.

    Never densified and never copied — the same rule `gene_axis` follows for `.X`, and it
    matters more here because there are N of these rather than one.
    """
    if not want or not _is_anndata(obj):
        return {}
    layers = getattr(obj, "layers", None)
    if layers is None:
        return {}
    return {name: layers[name] for name in want if name in layers}


def _anndata_matrix(adata: Any) -> Any:
    """AnnData → the matrix as stored. **No transform, no reduction, no PCA.**

    This function used to apply the standard scRNA preamble — `normalize_total(1e4)` →
    `log1p`/`sqrt` → `PCA(50)` — and return `obsm["X_pca"]`. Its own docstring recorded the
    problem it was causing: the preamble was UNDECLARED, "not a recipe step, so it appears in no
    trace, no g-vector and no caveat, and `n_comps` is not settable from a recipe". On
    `data/pbmc3k_raw.h5ad` it turned 2700 × 32738 into 2700 × 50 before the runner saw anything,
    which is why `rank_genes` and `qc` could not declare the gene axis they need — the axis was
    gone by the time a step could ask for it.

    Those three operations are declared steps now (`normalize` and `transform` in the `prep`
    group, `pca` in `latent`), and every bundled recipe carries them explicitly. This returns the
    frame; the recipe says what happens to it.

    **Sparse is returned as sparse.** The old body densified on every path, and `_dense` on
    pbmc3k materialises 353.6 MB from an 18.3 MB CSR. The prep steps and their upstream
    operations accept sparse, so the densification now happens where a step needs it — if it
    happens at all.
    """
    return adata.X


def _detect_time_key(adata: Any) -> "str | None":
    """The per-cell timepoint column in AnnData.obs, per the SHARED obs vocabulary.

    Deliberately does not match `sample`/`batch` (ambiguous — guessing turns batch
    structure into a fake trajectory) or `condition` (that is the contrast label, not a
    time axis). Use `--time-key` to name one explicitly."""
    from manyruns.vocab import find_time_key

    return find_time_key(adata.obs.columns)


def _detect_condition_key(adata: Any) -> "str | None":
    """The per-cell CONDITION column (disease/group/…) — the case/control label, used when
    there is no time axis (see manyruns.app.select_analysis). SHARED vocabulary."""
    from manyruns.vocab import find_condition_key

    return find_condition_key(adata.obs.columns)


def _detect_group_key(adata: Any) -> "str | None":
    from manyruns.vocab import find_group_key

    return find_group_key(adata.obs.columns)


def load_labeled(path: Path, time_key: "str | None" = None):
    """Return ``(matrix, labels, kind)``: a 2-D array, optional per-row labels, and whether
    those labels are a ``"time"`` axis or a ``"condition"`` axis (or ``None`` if unlabelled).

    Handles a single AnnData (.h5ad/.h5) carrying a timepoint column in .obs, or a
    DIRECTORY of per-timepoint sub-samples (each a 10x/.h5ad) — concatenated and labeled
    by sub-folder name (the raw Embryoid-Body layout: one 10x sample per collection day).
    Falls back to (matrix, None) for everything else — MIOFlow then computes pseudotime.

    **Returns the frame, untransformed.** The `transform=` parameter is gone with the preamble
    `_anndata_matrix` used to run — see that function. It was a load-time flag for a step-level
    fact, and keeping it would leave two places saying what the recipe's `transform` step says.
    """
    import numpy as np

    p = Path(path)
    if p.is_dir():
        samples = [
            d for d in sorted(p.iterdir())
            if d.is_dir() and (any(d.glob("matrix.mtx*")) or list(d.glob("*.h5ad")) or list(d.glob("*.h5")))
        ]
        if samples:
            import anndata

            ads, labels = [], []
            for d in samples:
                a = load_array(d)  # 10x dir / h5ad → AnnData
                ads.append(a)
                labels.extend([d.name] * a.n_obs)
            adata = anndata.concat(ads, join="outer")
            adata.obs_names_make_unique()  # barcodes repeat across timepoints
            matrix = _anndata_matrix(adata)
            # The one place this function can silently lie: `labels` is accumulated per
            # sub-sample while `matrix` comes back from a separate concat. If a
            # step ever drops or duplicates rows, the timepoints belong to the wrong cells
            # and MIOFlow trains on it without complaint. Guard the seam, don't trust it.
            check_aligned(matrix, labels, what=f"load_labeled({p.name})")
            return matrix, np.asarray(labels), "time"  # per-timepoint sub-samples

    obj = load_array(p)
    if _is_anndata(obj):
        labels, kind = labels_of(obj, time_key=time_key)
        return _anndata_matrix(obj), labels, kind
    return as_matrix(obj), None, None


def _is_anndata(obj: Any) -> bool:
    return type(obj).__name__ == "AnnData" or "anndata" in (type(obj).__module__ or "")


def labels_of(obj: Any, time_key: "str | None" = None) -> tuple:
    """Per-row ``(labels, kind)`` from a loaded object — WITHOUT touching the matrix.

    Split out of :func:`load_labeled` so the stackless engine can have labels without also
    inheriting that function's preprocessing. THAT SPLIT IS NOW MOSTLY HISTORICAL: since the
    cutover, `load_labeled` returns the same untransformed matrix `load_array` does, so the two
    paths no longer disagree about what a matrix is. It is kept because the reason underneath
    still holds — labels and matrix are fetched separately from the SAME object, so row order is
    shared by construction rather than by two loads landing in the same order.

    A TIME axis (for a trajectory) and a CONDITION axis (case/control) are different things
    and must not be confused: MIOFlow reads its label as time, so handing it a condition
    column trains a fabricated trajectory (treated → healthy "over time"). The kind travels
    with the labels so the trajectory step can refuse a condition.

    A GROUP axis is the third, and it is checked LAST. It is an unordered identity per row —
    a cell type, a cluster, a branch — and it exists so an embedding can be COLOURED, which is
    the whole content of "what changed when I turned that knob". It is refused by the
    trajectory step for the same reason a condition is: `runner._ml_lightning` passes labels to
    MIOFlow only when the kind is `time` (an allowlist, deliberately — a denylist of one let
    exactly this kind through)."""
    import numpy as np

    key, kind = label_key_of(obj, time_key)
    return (np.asarray(obj.obs[key].values), kind) if key is not None else (None, None)


def label_key_of(obj: Any, time_key: "str | None" = None) -> tuple:
    """Return ``(key, analysis_kind)`` with exactly :func:`labels_of`'s precedence.

    Display classification is separate: selecting a colour never invents a time axis.
    """
    if not _is_anndata(obj):
        return None, None
    for key, kind in ((time_key or _detect_time_key(obj), "time"),
                      (_detect_condition_key(obj), "condition"),
                      (_detect_group_key(obj), "group")):
        if key and key in obj.obs.columns:
            return key, kind
    return None, None


MISSING_COLOR = "(missing)"


def color_kind(series: Any) -> str:
    """Display kind: unordered dtypes or <= LEGEND_MAX finite numeric levels are categorical.

    Nullable pandas numeric dtypes count only finite, nonmissing values. This rule does
    not classify analysis labels: an automatically detected time axis stays continuous.
    """
    import numpy as np
    import pandas as pd
    from manyruns.figspec import LEGEND_MAX

    series = pd.Series(series)
    if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
        return "categorical"
    values = series.dropna()
    return "categorical" if values[np.isfinite(values)].nunique() <= LEGEND_MAX else "continuous"


def color_column_info(series: Any) -> dict:
    """JSON-plain picker facts and the shared CLI/UI availability policy; no level names.

    Unique barcode/ID columns cannot explain groups; other high-cardinality categories
    remain selectable and disclose legend suppression. Nonfinite counts include missing.
    """
    import numpy as np
    import pandas as pd
    from manyruns.figspec import LEGEND_MAX

    series = pd.Series(series)
    kind = color_kind(series)
    missing = int(series.isna().sum())
    numeric = pd.api.types.is_numeric_dtype(series.dtype)
    if numeric:
        values = series.dropna()
        finite = values[np.isfinite(values)]
        distinct = int(finite.nunique())
        nonfinite = len(series) - len(finite)
    else:
        distinct = int(series.nunique(dropna=True))
        nonfinite = missing
    reason = ""
    if len(series) == nonfinite:
        reason = "all values are missing or nonfinite"
    elif (str(series.name).lower() in {"barcode", "barcodes", "cell_id", "cell_ids", "obs_names"}
          and distinct == len(series) and len(series) > 1):
        reason = "unique value for every row — uninformative for colouring"
    levels = distinct + int(nonfinite > 0)
    warning = (f"legend suppressed above {LEGEND_MAX} categories"
               if kind == "categorical" and levels > LEGEND_MAX else "")
    return {"kind": kind, "n_distinct": int(distinct), "n_missing": missing,
            "n_nonfinite": nonfinite, "available": not reason, "reason": reason,
            "warning": warning}


def color_columns_of(obj: Any) -> list[dict]:
    """Detailed metadata-only inspection for an explicit colour action, in source order."""
    obs = getattr(obj, "obs", None)
    if obs is None:
        return []
    return [{"key": str(key), **color_column_info(obs[key])} for key in obs.columns]


def color_selection_reason(values: Any) -> str | None:
    """Explain an unavailable plotted selection after source rows have been narrowed.

    Source availability does not guarantee usable surviving values. Apply only the
    all-missing rule here; keep the source's colour kind and identifier policy unchanged.
    """
    if color_column_info(values)["n_nonfinite"] == len(values):
        return "all plotted values are missing or nonfinite"
    return None


def color_values_of(series: Any, kind: str) -> Any:
    """Normalize display values without changing row order or the source Series.

    Continuous missing values become NaN (nonfinite values are masked by the renderer).
    Categorical missing/nonfinite numeric values share the labelled missing category.
    Append stars to that label when necessary to keep literal categories distinct.
    """
    import numpy as np
    import pandas as pd

    series = pd.Series(series)
    if kind == "continuous":
        return series.to_numpy(dtype=float, na_value=np.nan)
    if kind != "categorical":
        raise ValueError(f"unknown display colour kind: {kind}")
    missing = series.isna().to_numpy()
    if pd.api.types.is_numeric_dtype(series.dtype):
        # Pandas copy-on-write can expose even this temporary mask as read-only.
        missing = missing | ~np.isfinite(series).fillna(False).to_numpy(dtype=bool)
    levels = {str(value) for value, absent in zip(series, missing) if not absent}
    missing_label = MISSING_COLOR
    while missing_label in levels:
        missing_label += "*"
    return np.asarray([missing_label if absent else str(value)
                       for value, absent in zip(series, missing)], dtype=str)


def color_channel_of(obj: Any, key: str) -> dict:
    """Return ``{key, values, kind, ...counts}`` on the original row axis.

    Raises an actionable ValueError for unknown or unavailable metadata. Both the CLI
    and picker use this policy; display selection does not affect :func:`labels_of`.
    Categorical values retain their dtype and raw missing/nonfinite values in a copied
    pandas array through positional row selection. In particular, nullable integers never
    round through float. The renderer counts the selected rows before applying
    :func:`color_values_of`. Continuous values are floats with NaN for missing.
    Counts here describe the full source only.
    """
    obs = getattr(obj, "obs", None)
    if obs is None or not len(obs.columns):
        raise ValueError("no metadata columns to colour by; available columns: none")
    if key not in obs.columns:
        raise ValueError(f"unknown colour column {key!r}; available columns: "
                         + ", ".join(map(str, obs.columns)))
    info = color_column_info(obs[key])
    if not info["available"]:
        raise ValueError(f"{key}: {info['reason']}")
    values = (color_values_of(obs[key], "continuous") if info["kind"] == "continuous"
              else obs[key].array.copy())
    return {"key": key, "values": values, **info}


def _stack_dataset(ds: Any) -> Any:
    """Stack a torch-style Dataset into a 2-D array: each item is a tensor, a
    ``{'data': ...}`` dict, or an ``(x, y)`` tuple; take the features and flatten."""
    import numpy as np

    if ds is None:
        raise ValueError("data module produced no dataset (train_dataset/dataset is None)")
    rows = []
    for i in range(len(ds)):
        item = ds[i]
        if isinstance(item, dict):
            v = item.get("data", next(iter(item.values())))
        elif isinstance(item, (tuple, list)):
            v = item[0]
        else:
            v = item
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        rows.append(np.asarray(v, dtype=float).ravel())
    return np.stack(rows)
