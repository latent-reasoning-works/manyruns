"""What the screens read. **No Textual import in this file** — that is the point of it.

This is the seam that keeps the app testable without a terminal, and it is the seam that keeps
the three surfaces honest. Everything here is derivable from `catalog`, `vocab`, `narrate` and
a `Session`; nothing here computes a fact those do not already expose. Where a fact was genuinely
missing it was added to `narrate` (`blocked`, `split_question`, `contents`, `step_mark`) rather
than to this module, because `narrate` is what the other two surfaces read and a fact computed
here would be a fourth account of one run.

Three things the screens need, one section each below:

    roster()   the drop folder + the bundled datasets
    ledger()   every recipe, and why a hollow one is hollow
    RunFeed    the live step record as `on_step` delivers it

DATACLASSES, NOT DICTS, and the reason is specific to what these are. `catalog`'s header argues
the opposite for recipes and is right there: a recipe goes YAML → dict → runner, so a type in
the middle is a second form of one thing that exists only to be unwrapped. Nothing here
round-trips. These are built once from four sources, read once by a renderer, never persisted
and never fed back into the engine — so there is no second form, and what a type buys is that
`row.recipe` misspelled is an AttributeError in a test rather than a silent `None` in a rendered
column. Every one carries `to_dict()`, which is pure JSON (str / int / float / bool / list /
None), so a test can assert on the whole structure without a terminal.

CONFIG LOCATION is the product's existing seam, not a parameter here: `$MANYRUNS_RECIPE_DIR`
and `$MANYRUNS_DATASET_DIR` override the bundled sets (see `catalog`'s header), and
`$MANYRUNS_DATA_DIR` overrides the drop folder (`shell.data_dir`). Threading a `config_dir`
argument through would have forked `narrate.offer`, which resolves the catalog itself and takes
no such argument.

Settle invariant: every filesystem source inspected or selected by the live UI — a
drop-folder file or directory, catalog alias inside or outside watched roots, fetched
sample, or symlink and its target — must match the signature captured on two consecutive
polls. The signature must change when the bytes a load would read change. Inspection
and selection recheck that same evidence before and after reading; they never promote
a fresh stat to a settled observation. Missing samples remain selectable for Fetch/Back.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Optional

from manyruns import narrate
from manyruns.narrate import Observation
from manyruns.tui.dropwatch import read_errors, row_guard

# `shell`, `catalog` and `app` are imported inside the functions that need them. `shell` is the
# largest module in the product and pulls `figures`; this module is imported by every screen and
# by tests that only want the ledger, so it stays cheap to import. Measured: `import
# manyruns.tui.state` loads neither `rich` nor `textual`.


# ══ §3.1 · the roster ════════════════════════════════════════════════════════
def _human_bytes(n: int) -> str:
    """A file size a person reads. Binary units, one decimal, because 5.6 MB is the fact and
    5,855,727 bytes is a number you have to count the digits of."""
    step = 1024.0
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < step or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= step
    return f"{size:,.1f} GB"


@dataclass(frozen=True)
class DataEntry:
    """One row of the roster: something you could point the product at.

    **No capability column, deliberately.** A recipe count alone says little about what
    a dataset can support. The analysis menu evaluates each recipe's declared needs against
    the observed data and explains missing requirements where the user chooses an analysis.
    """

    name: str
    kind: str                                   # "file" | "folder" | "bundled"
    obs: Observation                            # what `narrate.read_data` (or the YAML) saw
    path: Optional[Path] = None                 # a droppable file/folder, or a path-handle set
    dataset: Optional[str] = None               # a bundled dataset's name, for the engine
    topology: tuple = ()                        # the dataset's declared `topology:`, if any
    nbytes: Optional[int] = None                # on-disk size, when there is a file
    #: `qc.facts` for a dropped file, `None` when there is nothing to count.
    #:
    #: CACHED MEASUREMENTS, CARRIED WITH THE ROW. A property calling `qc.for_path` here would
    #: load the matrix on a cache miss. Listing only reads available cached measurements;
    #: the screen rebuilds its cells from `filtered()` on each filter change and cursor move.
    #:
    #: `compare=False` keeps it out of `__eq__` and `__hash__`, so carrying a dict does not make
    #: a frozen row unhashable — the same cost `declared` avoids by being a tuple.
    qc: Optional[dict] = field(default=None, compare=False, repr=False)

    missing: bool = False
    refusal: tuple[str, ...] = ()
    downloadable: bool = False
    pending: bool = False
    download_bytes: Optional[int] = None
    # Carry the inspected watcher signature through ledger/run-again selections. An alias
    # may depend on its containing directory, not just the file it eventually loads.
    observation: Optional[tuple[Path, tuple]] = field(default=None, compare=False, repr=False)

    # Live rows supply the canonical target captured during observation. Rendering and
    # dataclass replacements must retain it even after the source becomes a broken link.
    identity: tuple[str, str] | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.identity is not None:
            return
        from manyruns.tui.dropwatch import absolute, canonical

        key = self.name
        if self.kind != "bundled" and self.path is not None:
            try:
                key = str(canonical(self.path))
            except (OSError, ValueError):
                # Injected/unavailable rows have no observed target; keep their spelling.
                key = str(absolute(self.path))
        object.__setattr__(self, 'identity', (self.kind, key))

    @property
    def shape(self) -> str:
        """The `vocab.SHAPES` word. This is what `ledger()` keys legality on."""
        return self.obs.shape

    @property
    def modality(self) -> str:
        return self.obs.modality

    @cached_property
    def size(self) -> str:
        """The "size" column: rows × columns when we could read them, else bytes, else the
        generator's honest answer.

        Rows × columns first because that is the size a scientist means. `synthetic` is not a
        missing value — a bundled handle names a GENERATOR, so there is no file and no matrix
        until the engine makes one, and printing a byte count for it would be an invention.

        The test is "bundled and no file on disk", not "handle kind is manylatents", because
        both handle kinds can name a generator: `synthetic_timecourse.yaml` is
        `{kind: path, ref: "synthetic:time-course"}` and its own comment says "`synthetic:` is a
        generated ref, not a file". Keying on the kind printed `—` for it.
        """
        if self.pending:
            return "copying…"
        if self.missing:
            return (f"download · {_human_bytes(self.download_bytes)}"
                    if self.downloadable and self.download_bytes else "not on this machine")
        if self.refusal:
            return "unavailable"
        if self.obs.n_obs and self.obs.n_vars:
            return f"{self.obs.n_obs:,} × {self.obs.n_vars:,}"
        if self.obs.n_obs:
            return f"{self.obs.n_obs:,} rows"
        if self.kind == "bundled" and self.nbytes is None:
            return "synthetic"
        if self.obs.n_timepoints:
            return f"{self.obs.n_timepoints} timepoints"
        if self.nbytes is not None:
            return _human_bytes(self.nbytes)
        return "folder" if self.kind == "folder" else "—"

    @cached_property
    def contents(self) -> str:
        """The "what's in it" column — `narrate.contents`, never phrased here.

        `obs.groups` is passed so a dropped file that carries a label column says so instead of
        saying "one population" over eight named cell types. Only the drop-folder rows can move:
        every bundled row either declares a `topology:` (which still wins) or has `groups is
        None`, `pbmc3k_raw.h5ad` — the row `test_the_rows_are_the_roster_plus_a_find_data_row_at_the_bottom`
        pins to "one population" — among them, since it carries no group column at all.
        """
        return narrate.contents(self.obs.shape, self.topology, self.obs.groups)

    @cached_property
    def glyphs(self) -> str:
        """The declared topology drawn as marks — what `contents` says, said in the ledger's
        own vocabulary, so one dataset reads the same on both screens.

        Empty for every dropped file: a file has no declaration to draw. That is why the group
        holding them shows QC numbers instead of a geometry column of blanks."""
        return narrate.glyphs(self.topology)

    @property
    def qc_cells(self) -> dict:
        """`self.qc` rendered for a table, phrased by `narrate` and not by a screen."""
        return narrate.qc_cells(self.qc)

    def as_source(self) -> tuple:
        """`(data_folder, dataset, modality, obs)` — exactly the tuple `shell._pick_source`
        returns and `shell._run` takes, so a screen hands a selected row straight to the
        existing run path instead of unpacking it into a second calling convention."""
        return (self.path, self.dataset, self.obs.modality, self.obs)

    @staticmethod
    def _obs_json(obs) -> dict:
        """An `Observation` as JSON that ROUND-TRIPS, which `asdict` alone does not.

        `Observation.declared` is a TUPLE (it is a fact set, and a tuple is hashable and cannot be
        mutated by a reader), and `json.dumps` writes a tuple as an array which `json.loads` reads
        back as a LIST — so `loads(dumps(payload)) != payload` and the whole view stops being
        comparable to itself. Caught by `test_every_view_serialises_to_plain_json`, in CI and not
        locally, the day `declared` was added.

        Converted rather than the field being changed to a list: the in-process type is right, and
        a JSON surface is where the conversion belongs. Same reason `path` is `str()`-ed and
        `topology` is `list()`-ed two lines up.
        """
        out = asdict(obs)
        if out.get("declared") is not None:
            out["declared"] = list(out["declared"])
        return out

    def to_dict(self) -> dict:
        return {
            "name": self.name, "kind": self.kind, "size": self.size,
            "contents": self.contents, "shape": self.shape, "modality": self.modality,
            "path": str(self.path) if self.path is not None else None,
            "dataset": self.dataset, "topology": list(self.topology),
            "nbytes": self.nbytes, "glyphs": self.glyphs, "qc": self.qc,
            "obs": self._obs_json(self.obs),
            "missing": self.missing, "refusal": list(self.refusal),
            "downloadable": self.downloadable, "pending": self.pending,
            "download_bytes": self.download_bytes,
        }


#: The kinds that are the practitioner's OWN data, as opposed to something this app shipped.
#: Stated once beside the `kind` field it reads, so a screen does not invent its own membership
#: test and a folder of per-sample subfolders is not quietly treated as a different kind of
#: thing from a single file.
OWN_KINDS = frozenset({"file", "folder"})


def _qc(p: Path) -> Optional[dict]:
    """Carry already measured QC; a roster cache miss must never load a matrix."""
    from manyruns import inspected
    from manyruns.vocab import _GENE_AXIS_SUFFIXES

    try:
        if not (p.is_file() and p.suffix.lower() in _GENE_AXIS_SUFFIXES):
            return None
        cached = inspected.get(p)
        measured = cached.get('qc') if cached else None
        return measured if isinstance(measured, dict) else None
    except (*read_errors(), TypeError):
        return None


def _held(path: Path, skip: set[Path]) -> bool:
    from manyruns.tui.dropwatch import canonical
    key = canonical(path)
    return any(key == held or held in key.parents for held in skip)


@dataclass
class _SourceObservation:
    source: Path
    key: Path
    watched: Path
    signature: tuple | None
    on_changed: Any = None

    def unchanged(self, resolved: Path | None = None) -> bool:
        from manyruns.tui.dropwatch import canonical, fingerprint

        if (self.signature is not None and canonical(self.source) == self.key
                and (resolved is None or canonical(resolved) == self.key)
                and fingerprint(self.watched) == self.signature):
            return True
        if self.on_changed:
            self.on_changed(self.watched)
        return False

    @property
    def evidence(self) -> tuple[Path, tuple]:
        return self.watched, self.signature


def _source_observation(path: Path, *, skip=(), snapshot=None, observation=None,
                        expected=None, require_observation=False,
                        on_changed=None) -> _SourceObservation | None:
    """Apply the settle invariant to every filesystem source, before any reader runs.

    Live callers supply a DropWatch snapshot plus its held set, or retain an inspected
    row's evidence for selection. A missing snapshot signature is never replaced by a
    fresh stat. Synchronous callers without watcher context retain their inspection API.
    """
    from manyruns.tui.dropwatch import absolute, canonical, fingerprint

    source = absolute(path)
    key = canonical(source)
    watched = key
    if snapshot is not None:
        watched = snapshot.references.get(source, next(
            (p for p in (key, *key.parents) if p in snapshot.entries), key))
        expected = snapshot.entries.get(watched)
    if observation is not None:
        watched, expected = observation
    if _held(watched, skip) or _held(source, skip):
        return None
    if not _held(source, {watched}) or (require_observation and observation is None):
        expected = None
    elif snapshot is None and observation is None and not require_observation and expected is None:
        expected = fingerprint(watched)
    result = _SourceObservation(source, key, watched, expected, on_changed)
    return result if result.unchanged() else None


def _check_readable(path: Path, signature: tuple) -> tuple[int | None, int | None]:
    """Probe headers, never matrix payloads. Deeper validation belongs to run start.

    Follow the execution loader's directory precedence without executing scripts or
    concatenating samples. A table preview is bounded; its total row count is unknown.
    The caller rechecks the same settle evidence after all metadata reads.
    """
    suffix = path.suffix.lower()
    if path.is_dir():
        children = sorted(path.iterdir())
        matrices = [p for p in children if p.name in {'matrix.mtx', 'matrix.mtx.gz'}]
        if matrices:
            for names in ({'barcodes.tsv', 'barcodes.tsv.gz'},
                          {'features.tsv', 'features.tsv.gz', 'genes.tsv', 'genes.tsv.gz'}):
                if not any(p.name in names and p.is_file() for p in children):
                    raise ValueError(f'10x matrix folder is missing {sorted(names)}')
            genes, cells = _check_readable(matrices[0], signature)
            return cells, genes  # 10x stores features × barcodes
        from manyruns import app

        samples = app._sample_subdirs(path)
        if samples:
            shapes = [_check_readable(p, signature) for p in samples]
            # An outer gene join can change the width; headers alone cannot know it.
            rows = [s[0] for s in shapes]
            return sum(rows) if all(n is not None for n in rows) else None, None
        loadable = {'.csv', '.tsv', '.txt', '.npy', '.h5ad', '.h5', '.mtx', '.py'}
        candidates = [p for p in children if p.is_file() and p.suffix.lower() in loadable]
        if not candidates:
            raise ValueError(f'no loadable data file found in {path}')
        return _check_readable(candidates[0], signature)
    if suffix == '.py':
        return None, None
    if suffix == '.mtx' or path.name == 'matrix.mtx.gz':
        from scipy.io import mminfo

        rows, cols, *_ = mminfo(path)
        shape = rows, cols
    elif suffix == '.h5ad':
        import h5py

        with h5py.File(path, 'r') as handle:
            shape = _h5ad_dimensions(handle['X'])
    elif suffix == '.h5':
        import h5py

        with h5py.File(path, 'r') as handle:
            matrices = [handle['matrix']] if 'matrix' in handle else list(handle.values())
            if not matrices:
                raise ValueError('no 10x matrix header')
            shape = None
            for node in matrices:
                # Only the two-integer shape dataset is read, never counts or indices.
                if node['shape'].shape != (2,):
                    raise ValueError('invalid 10x matrix shape header')
                genes, cells = map(int, node['shape'][...])
                for key in ('data', 'indices', 'indptr', 'barcodes'):
                    if key not in node:
                        raise ValueError(f'incomplete 10x matrix header: missing {key}')
                shape = cells, genes
            if len(matrices) > 1:
                raise ValueError('multiple 10x genomes; provide a single-genome source')
    elif suffix == '.npy':
        import numpy as np

        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            shape = matrix.shape
        finally:
            matrix._mmap.close()
        if len(shape) == 1:
            shape = (shape[0], 1)  # the execution adapter treats vectors as one feature
    elif suffix in {'.csv', '.tsv', '.txt'}:
        import csv
        from io import StringIO
        from itertools import islice
        from manyruns.pipeline.loading import _read_table

        sep = ',' if suffix == '.csv' else '\t'
        try:
            with path.open(newline='', encoding='utf-8-sig') as handle:
                # Sample complete records: wide headers and quoted newlines are legal.
                reader = csv.reader(handle, delimiter=sep, strict=True)
                records = list(islice((row for row in reader if row), 6))
        except csv.Error as error:
            raise ValueError(f'unreadable table header: {error}') from error
        text = StringIO()
        csv.writer(text, delimiter=sep).writerows(records)
        text.seek(0)
        _read_table(text, sep)
        return None, None  # the preview cannot establish the full matrix shape
    else:
        raise ValueError(f'unsupported data suffix {suffix!r}')
    if len(shape) != 2:
        raise ValueError(f'expected a two-dimensional matrix, got shape {shape}')
    if shape[0] <= 0:
        raise ValueError('no rows of data')
    if shape[1] <= 0:
        raise ValueError('no numeric features')
    return int(shape[0]), int(shape[1])


def _h5ad_dimensions(node) -> tuple:
    """Modern and pre-0.7 sparse headers, without reading X or its sparse arrays."""
    import h5py
    from anndata._io.specs.registry import Reader, _REGISTRY, get_spec

    _REGISTRY.get_read(type(node), get_spec(node), reader=Reader(_REGISTRY))
    if isinstance(node, h5py.Dataset):
        return node.shape
    attribute = 'shape' if 'shape' in node.attrs else 'h5sparse_shape'
    return tuple(node.attrs[attribute])


def _h5ad_observation(path: Path, modality: str, source: str) -> Observation:
    """Read only observation metadata, including its existing narration semantics.

    Even anndata.read_h5ad(backed='r') eagerly reads layers, obsm and other arrays.
    Read the obs element alone so none of those matrices can enter a roster inspection.
    """
    import h5py
    from anndata import AnnData
    from anndata._io.h5ad import read_dataframe
    from anndata.compat import _clean_uns
    from anndata.io import read_elem
    from manyruns import vocab
    from manyruns.pipeline.loading import color_columns_of

    with h5py.File(path, 'r') as handle:
        # AnnData's compatibility reader decodes both compound datasets and modern
        # dataframe groups; reading this element alone leaves all matrices untouched.
        obs = read_dataframe(handle['obs'])
        dimensions = _h5ad_dimensions(handle['X'])
        if len(obs) != dimensions[0]:
            raise ValueError('observation rows do not match matrix header')
        metadata = AnnData(obs=obs)  # no X, layers, or other matrix payloads
        if isinstance(handle['obs'], h5py.Dataset) and 'uns' in handle:
            # Pre-0.7 categorical codes keep their levels in uns. Read only the
            # matching obs categories, then use AnnData's own compatibility decoder.
            metadata.uns = {
                key: read_elem(handle['uns'][key]) for key in handle['uns']
                if key.endswith('_categories') and key.replace('_categories', '') in obs
            }
            _clean_uns(metadata)
        obs = metadata.obs
    columns = [[c['key'], c['kind'], c['n_distinct']] for c in color_columns_of(metadata)]
    cols = {c.lower(): c for c in obs.columns}
    has_time = any(k in cols for k in vocab.TIME_KEYS)
    conditions = None
    for key in vocab.CONDITION_KEYS:
        if key in cols:
            values = [str(v) for v in obs[cols[key]].unique()]
            if len(values) >= 2:
                conditions = values
                break
    group_key = None
    groups = None
    if not has_time and conditions is None:
        key = vocab.find_group_key(obs.columns)
        if key is not None:
            counts: dict[str, int] = {}
            for value in obs[key]:
                label = str(value)
                counts[label] = counts.get(label, 0) + 1
            if len(counts) >= 2:
                group_key = key
                groups = sorted(counts, key=lambda s: (-counts[s], s))
    return Observation(shape=narrate.shape_of(has_time, conditions, modality), modality=modality,
                       source=source, conditions=conditions, group_key=group_key, groups=groups,
                       n_obs=int(dimensions[0]), n_vars=int(dimensions[1]),
                       signals={'has_time': has_time, 'conditions': conditions, 'group': group_key},
                       columns=columns)


def _resolve_path_source(path: Path, console):
    """Use the shared metadata resolver except where its backed read loads matrices."""
    from manyruns import app, shell

    if path.is_file() and path.suffix.lower() == '.h5ad':
        modality = app.detect_modality(path)
        return path, None, modality, _h5ad_observation(path, modality, path.name)
    if path.is_file() and path.suffix.lower() == '.h5':
        return path, None, 'scrna', Observation(shape='single', modality='scrna', source=path.name)
    return shell._resolve_path_source(path, console)


def _resolve_dataset(name: str, ds: dict, found: Path | None):
    from manyruns import shell
    if found is not None and found.is_file() and found.suffix.lower() in {'.h5ad', '.h5'}:
        modality = ds.get('modality', 'unknown')
        obs = (_h5ad_observation(found, modality, name) if found.suffix.lower() == '.h5ad'
               else Observation(shape='single', modality=modality, source=name))
        return found, None, modality, replace(obs, shape=ds.get('shape') or 'unknown',
                                            declared=tuple(narrate.declared_facts(ds)))
    return shell._resolve_dataset(name)


def _with_dimensions(obs: Observation, dimensions: tuple) -> Observation:
    """Show header dimensions without making bare arrays claim a named gene axis."""
    rows, cols = dimensions
    from manyruns.vocab import _GENE_AXIS_SUFFIXES

    declared = obs.declared
    if cols is not None and obs.n_vars is None and declared is None:
        declared = ('genes',) if Path(obs.source).suffix.lower() in _GENE_AXIS_SUFFIXES else ()
    return replace(obs, n_obs=rows if rows is not None else obs.n_obs,
                   n_vars=cols if cols is not None else obs.n_vars, declared=declared)


def _finish_entry(entry: DataEntry) -> DataEntry:
    """Build data-derived display cells before leaving the per-source guard."""
    for label in (entry.size, entry.contents, entry.glyphs, *entry.qc_cells.values()):
        if not isinstance(label, str):
            raise TypeError('row labels must be strings')
    return entry


@row_guard
def catalog_entry(name: str, *, skip=(), snapshot=None, on_changed=None,
                  observation=None, require_observation=False) -> DataEntry:
    """Resolve one declaration, without sniffing a held source or inventing availability."""
    from manyruns import catalog, datasetfetch, shell
    from manyruns.tui.dropwatch import absolute

    ds = catalog.load_dataset(name)
    handle = ds.get("handle") or {}
    ref = handle['ref']
    if not isinstance(ref, str):
        raise ValueError(f"invalid dataset {name!r}: handle.ref must be a string")
    found = shell.dataset_ref_path(ref) if handle.get("kind") == "path" else None
    # Use that same resolution for availability: a second lookup could find a file
    # that appeared after the first and send it to inspection without any evidence.
    missing = bool(handle.get("kind") == "path" and ref
                   and not ref.startswith(shell._GENERATED_REF) and found is None)
    source = (_source_observation(found, skip=skip, snapshot=snapshot,
                                 observation=observation, require_observation=require_observation,
                                 on_changed=on_changed) if found is not None else None)
    pending = found is not None and source is None
    declared = Observation(shape=ds.get("shape", "unknown"),
                           modality=ds.get("modality", "unknown"), source=name,
                           declared=tuple(sorted(narrate.declared_facts(ds))))
    folder = found if found is not None else Path(ref) if handle.get("kind") == "path" else None
    dataset = None if handle.get("kind") == "path" else handle.get("ref", name)
    obs = declared
    inspected_observation = None
    refusal = tuple(shell.where_the_file_should_be(name, ds)) if missing else ()
    if not missing and not pending:
        try:
            dimensions = (None, None)
            if source is not None:
                dimensions = _check_readable(source.source, source.signature)
            folder, dataset, _, obs = _resolve_dataset(name, ds, found)
            obs = _with_dimensions(obs, dimensions)
        except read_errors() as error:  # one unreadable file must not remove its declaration
            refusal = (f"Could not read {folder}: {type(error).__name__}: {error}",)
        if source is not None and not source.unchanged(folder):
            pending = True
            obs = declared
        elif source is not None and not refusal:
            inspected_observation = source.evidence
    if obs.declared is None:
        obs = replace(obs, declared=declared.declared)
    if folder is not None and found is not None:
        folder = absolute(folder)
    entry = DataEntry(
        name=name, kind="bundled", obs=obs, path=folder, dataset=dataset,
        topology=tuple(ds.get("topology") or ()),
        nbytes=found.stat().st_size if found and found.is_file() and not pending else None,
        missing=missing, pending=pending, refusal=refusal,
        downloadable=datasetfetch.can_fetch(ds), download_bytes=handle.get("bytes"),
        observation=inspected_observation,
    )
    return _finish_entry(entry)


@row_guard
def local_entry(path: Path, *, expected=None, skip=(), snapshot=None, observation=None,
                require_observation=False, on_changed=None, on_failed=None,
                previous: DataEntry | None = None) -> Optional[DataEntry]:
    """Inspect once and publish only if the source still matches the observed bytes.

    Directory cache keys omit descendant changes, so directory observations bypass that
    cache. File rows also carry the signature inspected, because the cache writer stats
    independently and may publish after the file has changed.
    """
    from manyruns import inspected
    from manyruns.tui.dropwatch import absolute

    p = absolute(path)
    source = _source_observation(p, expected=expected, skip=skip, snapshot=snapshot,
                                 observation=observation, require_observation=require_observation,
                                 on_changed=on_changed)
    if source is None:
        return None
    # A live row already contains the inspection and QC for these exact bytes. Reuse
    # it before consulting the bounded disk cache, which may have evicted this source.
    if (previous is not None and previous.path == p and not previous.pending
            and not previous.missing
            and previous.identity == (previous.kind, str(source.key))
            and previous.observation == source.evidence):
        return previous
    before = source.signature
    is_folder = p.is_dir()
    cached = None if is_folder else inspected.get(p)
    if cached and (cached.get('observed_signature') != list(before)
                   or cached.get('observed_source') != str(p)
                   or not cached.get('source_readable')):
        cached = None
    obs = inspected.to_observation(cached, p.name) if cached else None
    if obs is not None and p.suffix.lower() == '.h5ad' and obs.columns is None:
        # The earlier header reader cached unknown columns even after a successful
        # read. Refresh that metadata while preserving separately measured QC.
        obs = None
    fresh = obs is None
    if fresh:
        dimensions = _check_readable(p, source.signature)
        console = _RecordingConsole()
        resolved = _resolve_path_source(p, console)
        if resolved is None:
            raise ValueError('\n'.join(console.lines) or 'source resolver returned no data')
        _, _, modality, obs = resolved
        obs = _with_dimensions(obs, dimensions)
    if not source.unchanged():
        return None
    qc = _qc(p)
    if fresh and not is_folder:
        # Preserve QC already measured elsewhere for these bytes while refreshing metadata.
        row = dict(cached or {})
        row.update(inspected.as_row(modality, obs))
        if qc is not None:
            row['qc'] = qc
        row['observed_signature'] = list(before)
        row['observed_source'] = str(p)
        row['source_readable'] = True
        inspected.put(p, row)
    if not source.unchanged():
        return None
    kind = "folder" if is_folder else "file"
    return _finish_entry(DataEntry(name=p.name, kind=kind, obs=obs,
                     path=p, nbytes=None if is_folder else before[0], qc=qc,
                     observation=source.evidence, identity=(kind, str(source.key))))


def roster(ddir: Optional[Path] = None, *, skip=(), snapshot=None,
           on_changed=None, on_failed=None, on_skipped=None, previous=(),
           failures=None) -> list[DataEntry]:
    """Local data and declarations, using the caller's single stat snapshot when supplied.

    Held drops are omitted before cache/QC work; their catalog aliases remain declared-only
    and marked copying. Reuse unchanged local rows supplied by a live caller; catalog
    declarations still reload so an explicit rescan sees config repairs. No listing
    path downloads or verifies sample bytes. Report skipped declaration names separately
    from transient local-file failures, so the screen can name them without retrying I/O.
    """
    from manyruns import shell
    from manyruns.catalog import discover_datasets
    from manyruns.tui.dropwatch import canonical

    held = {canonical(p) for p in skip}
    paths = shell.drop_folder_entries(ddir) if snapshot is None else snapshot.entries
    if snapshot is not None and snapshot.drops is not None:
        paths = snapshot.drops
    entries = []
    names = set()
    local_rows = {e.path: e for e in previous if e.kind in OWN_KINDS and not e.refusal}
    failures = {} if failures is None else failures
    for path in list(failures):
        if path not in paths:
            del failures[path]

    def changed(path):
        # _SourceObservation supplies the captured watcher key, even if it now loops.
        held.add(path)
        if on_changed:
            on_changed(path)

    for p in paths:
        # Ignore bytecode caches created by executing a declared Python source.
        if p.name == '__pycache__' or p.name in names:
            continue
        names.add(p.name)

        def failed(reason):
            # Only memoize the same settled bytes, never a fresh post-error stat.
            try:
                source = _source_observation(p, snapshot=snapshot, skip=held,
                                             on_changed=changed)
            except read_errors():
                source = None
            if source is not None:
                if snapshot is not None:
                    failures[p] = (source.evidence, reason)
                # Directory fingerprints carry descendants; file fingerprints do not.
                kind = 'folder' if len(source.signature) > 2 else 'file'
                entries.append(DataEntry(
                    p.name, kind, Observation(shape='unknown', modality='unknown', source=p.name),
                    path=source.source, refusal=(reason,), observation=source.evidence,
                    identity=(kind, str(source.key))))
            if on_failed:
                on_failed(reason)

        try:
            if _held(p, held):
                failures.pop(p, None)
                continue
            if p in failures:
                evidence, reason = failures[p]
                source = _source_observation(p, snapshot=snapshot, skip=held, on_changed=changed)
                if source is not None and source.evidence == evidence:
                    failed(reason)
                    continue
                failures.pop(p, None)

            entry = local_entry(p, snapshot=snapshot, skip=held,
                                on_changed=changed, on_failed=failed,
                                previous=local_rows.get(p))
            if entry is not None:
                entries.append(entry)
        except read_errors() as error:  # retain this source's diagnosis, not just its path
            failed(str(error))
    for name in discover_datasets():
        try:
            entry = catalog_entry(name, skip=held, snapshot=snapshot, on_changed=changed)
        except read_errors() as error:
            if on_skipped:
                on_skipped(f'{name}: {error}')
            continue
        entries.append(entry)
    return entries


class _RecordingConsole:
    """Retain the shared resolver's refusal without writing underneath the screen."""

    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        parts = []
        for arg in args:
            text = str(arg)
            # The resolver colors either its entire error or the "not found:" label.
            # Remove that known wrapper without importing a renderer into this seam.
            if text.startswith('[red]'):
                text = text[len('[red]'):]
                text = (text[:-len('[/red]')] if text.endswith('[/red]') else
                        text.replace('[/red]', '', 1))
            parts.append(text)
        self.lines.append(' '.join(parts))


# ══ §3.2 · the ledger ════════════════════════════════════════════════════════
@dataclass(frozen=True)
class LedgerRow:
    """One analysis, filled or hollow. A hollow row is on screen precisely because its absence
    is the informative part — §0 of the spec names invisible refusal as one of the three defects
    the rewrite exists to fix: today a recipe that cannot run is simply missing from the list,
    so the scientist never learns that a disease column would unlock `contrast`."""

    recipe: str
    ask: str                      # "Trace the path"        — the first column
    gloss: str                    # "where are these cells heading…" — the second, when runnable
    can_run: bool
    blocked: tuple = ()           # why not, in the scientist's words; empty iff `can_run`
    recommended: bool = False     # `suits:` covers this shape — advice, never legality
    #: THE PRIMITIVES THE RECIPE ALREADY DECLARED, and which no surface used to read. `ask`/
    #: `gloss` are prose — two clauses a reader has to parse to learn what a row runs — and the
    #: recipes carry the facts underneath them: `claims` is the geometry the analysis ASSUMES,
    #: `id` names the method, `source.cite` names the paper. They come from `narrate.offer`
    #: rather than from a second catalog read here, which is the rule this package is built
    #: under: no screen may compute a fact `narrate` does not expose.
    #:
    #: `assumption`/`glyph` are None for the conditioning recipes, and that is not missing data:
    #: they declare `claims: None` because they CONDITION the data rather than assume a shape of
    #: it, and giving them an invented assumption to fill the column is exactly the sort of tidy
    #: lie this pane is being rebuilt to stop telling. (Named by their declaration and not here —
    #: `test_the_ledger_is_catalog_derived_and_names_no_recipe` reads this comment too, and it is
    #: right to: a recipe named in the ledger's source is one the catalog no longer controls.)
    assumption: "str | None" = None   # a topology word — the grouping key
    glyph: "str | None" = None        # "∴" — one column wide, `narrate.ASSUMPTION_GLYPH`
    method: str = ""                  # "leiden", off the recipe's `id`
    cite: str = ""                    # "Traag 2019"
    #: What the numbers say about this analysis's RESULT. Advice and never legality — a row
    #: with a caution is exactly as selectable as one without, which is the freestyle rule:
    #: the marker informs, it does not disable. Sentences phrased by `narrate`, never here.
    caution: tuple = ()
    #: Whether a measurement was taken at all, so "nothing to say" and "nothing was asked" are
    #: not the same blank.
    checked: bool = False

    @property
    def measured(self) -> str:
        """One word for the status column: `refused` outranks everything (a row this data cannot
        run already carries its reason), then `caution`, then `clear`, else `unmeasured`."""
        if not self.can_run:
            return "refused"
        if self.caution:
            return "caution"
        return "clear" if self.checked else "unmeasured"

    @property
    def detail(self) -> str:
        """The second column, whichever kind of row this is.

        §3.2 draws the reason where the gloss goes, so the two are one column and this is where
        that choice is made once. Left to each screen it would be made twice and drift; it is
        not left to the screen for the same reason the phrasing is not.
        """
        return self.gloss if self.can_run else "; ".join(self.blocked)

    def as_offer(self) -> dict:
        """This row as one entry of a decision record's `offered` list.

        THE ONE CONVERSION, and it lives here rather than in `decisions.py` so the corpus records
        the rows the screen actually drew instead of a recomputed ledger. Re-deriving would be a
        second account of what was offered — the same failure "no screen computes a fact
        `narrate` does not expose" prevents, one direction further out.

        `ask`/`gloss` are omitted: they are the phrasing, and a corpus that carried them would
        re-label itself every time a `question:` was reworded.
        """
        # `measured` is the one WORD, never the sentence: the corpus must record what the
        # screen showed — a later reader cannot otherwise tell "picked the row the numbers
        # warned about" from "picked a row nothing was known about" — but the sentence is
        # phrasing, and a corpus that carried it would re-label itself on every rewording.
        return {"recipe": self.recipe, "can_run": self.can_run,
                "blocked": list(self.blocked), "recommended": self.recommended,
                "measured": self.measured}

    def to_dict(self) -> dict:
        return {"recipe": self.recipe, "ask": self.ask, "gloss": self.gloss,
                "detail": self.detail, "can_run": self.can_run,
                "blocked": list(self.blocked), "recommended": self.recommended,
                "measured": self.measured, "caution": list(self.caution),
                "checked": self.checked}


def measurement(entry: "DataEntry") -> Optional[dict]:
    """The measurement the roster took for this dataset, if it took one.

    A ONE-LINE ACCESSOR, ON PURPOSE. The attribute it reads is spelled the same as a recipe,
    and two guards forbid that spelling in `ledger`'s source and in the ledger screen's — so
    the read lives here, in neither, and both call this.
    """
    return entry.qc


def ledger(obs: Observation,
           provided: "frozenset | set | tuple | None" = None,
           measured: "dict | None" = None) -> list[LedgerRow]:
    """Every recipe against this data: can it run, and if not WHICH FACT is missing.

    This is `vocab.unmet` rendered directly, which is what §3.2 asks for. Legality is
    `narrate.blocked` (one call that wraps `unmet`, so a verdict and its reason cannot be
    computed separately and disagree); the label and `recommended` come from `narrate.offer`,
    which is already catalog-derived — adding a recipe YAML adds a ledger row with no code
    change here.

    `provided` is what the caller already holds — a precomputed embedding, a graph another tool
    built — and is passed straight through to `unmet`, whose docstring explains why that
    matters: the test is structural (does the object exist), never canonical (did a blessed step
    make it).

    ORDER: runnable first, then recommended-first, then alphabetical. Recipes you can select
    come before ones you cannot, because a hollow row is not selectable and a list that
    interleaves them makes the keyboard skip. Within each group `offer`'s own order is kept, so
    this does not become a second ranking — the head of the runnable block is `offer`'s
    recommendation, and `shell._pick_recipe`'s comment on why that head is meaningful applies
    unchanged.

    `contrast` needs a condition column; a sample without one gets a hollow row with an
    explanation. Recipe availability follows the data's declared and observed capabilities.
    """
    from manyruns.catalog import discover_recipes, load_recipe

    rows: list[LedgerRow] = []
    for offered in narrate.offer(obs, available=discover_recipes()):
        try:
            recipe = load_recipe(offered["recipe"])
        except Exception:  # noqa: BLE001 - a broken recipe must not empty the ledger
            continue
        # Two questions, kept apart at the source and joined only here: does THIS DATA carry the
        # facts these steps need (`blocked`), and can these steps run at all right now
        # (`unavailable` — a step broken upstream is missing no data fact). Both make a row
        # hollow and both put a reason on it, which is all this screen needs them to have in
        # common; see `narrate.unavailable` for why folding them into one call would cost the
        # invariant that keeps a verdict and its reason from disagreeing.
        why = narrate.blocked(recipe, obs.shape, provided) + narrate.unavailable(recipe)
        ask, gloss = narrate.split_question(offered["label"])
        rows.append(LedgerRow(
            recipe=offered["recipe"], ask=ask, gloss=gloss, can_run=not why,
            blocked=tuple(why), recommended=bool(offered["recommended"]),
            assumption=offered.get("assumption"), glyph=offered.get("glyph"),
            method=offered.get("method", ""), cite=offered.get("cite", ""),
            # The sentence is narrate's; this only carries it. `measured is not None` is the
            # checked flag so a synthetic dataset — never measured — is not read as clean.
            caution=tuple(narrate.concerns(recipe, measured)),
            checked=measured is not None,
        ))
    rows.sort(key=lambda r: (not r.can_run,))   # stable: `offer`'s order survives inside groups
    return rows


# ══ §3.3 · the run ═══════════════════════════════════════════════════════════
@dataclass(frozen=True)
class StepView:
    """One step of a run, as the run screen draws it — declared-and-waiting, running, or done.

    `detail` is the RAW message, not clipped. `narrate.run_panel` clips at 58 characters because
    a plain-text line has no layout engine and a 300-character traceback would destroy it; a
    Textual screen does have one, and §2 of the spec says making layout a constraint rather than
    a magic number is the whole reason for Textual. Clipping here would hand the screen a
    pre-truncated string it cannot un-truncate at a wider width.
    """

    index: int
    name: str
    state: str                       # "queued" | "running" | a `watch.OUTCOMES` word
    glyph: str
    word: str
    seconds: Optional[float] = None  # a DURATION, so None until the step has settled
    started: Optional[float] = None  # the monotonic stamp a live clock ticks from, else None
    detail: str = ""
    deltas: tuple = ()               # `narrate.GeometryRow` — what this step moved
    plots: tuple = ()                # figures this step drew, from `rec["plots"]`

    @property
    def running(self) -> bool:
        return self.state == "running"

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "state": self.state,
                "glyph": self.glyph, "word": self.word, "seconds": self.seconds,
                "started": self.started, "detail": self.detail,
                "deltas": [{"label": d.label, "value": d.value, "measured": d.measured}
                           for d in self.deltas],
                "plots": list(self.plots)}


class RunFeed:
    """The live step record, as `on_step` already delivers it.

    Hand `feed.on_step` to `Session(on_step=…)` (or to `runner.run_*`) and read `feed.rows()`.
    It is the same observer contract `shell.live_dashboard` uses, and it is deliberately a
    plain accumulator: `runner.apply_step` appends each record to the lineage list BEFORE the
    executor runs and reports the same list on every change, so there is nothing to reconstruct
    here — the fold is the only work.

    An observer must never take the run down with it. `runner._report` already swallows
    everything this raises, which means a screen bug would vanish silently; the listener is
    therefore called inside its own guard and the exception is KEPT on `observer_errors` rather
    than discarded, so it can be asserted on in a test and shown in a debug pane.
    """

    def __init__(self, recipe: Optional[dict] = None,
                 listener: Optional[Callable[["RunFeed"], None]] = None) -> None:
        self.declared: list = list((recipe or {}).get("steps") or [])
        self.records: list = []
        self.listener = listener
        self.observer_errors: list = []

    def on_step(self, rec: dict, records: list) -> None:
        """The observer. `records` is the lineage list the runner owns; it is COPIED, because
        the runner keeps mutating the live record in place and a screen that renders the same
        list on the next frame would show a settled step it never received a report for."""
        self.records = list(records)
        if self.listener is None:
            return
        try:
            self.listener(self)
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed; see the class docstring
            self.observer_errors.append(f"{type(e).__name__}: {e}")

    def rows(self) -> list[StepView]:
        """Declared steps overlaid with what has actually happened — the live view.

        The overlay is what makes a two-minute embedding watchable: every declared step is on
        screen as `queued` before anything runs, and fills in as its record lands.

        UNLIKE `shell._progress_panel`, a record whose index is past the declared list is still
        shown, appended after them. That panel drops it (`by_index` over `enumerate(declared)`),
        which is right for a fixed recipe and wrong here: `Session.step` issues ad-hoc actions
        with no declaration behind them, and `Session.run_recipe` deliberately does NOT
        de-duplicate, so re-running a step is a second record at its own index. A step that ran
        and is not on screen is the one failure a run view must not have.
        """
        by_index = {r.get("index"): r for r in self.records}
        views = [self._view(i, str(step.get("name") or "?"), by_index.get(i))
                 for i, step in enumerate(self.declared)]
        for rec in self.records:
            i = rec.get("index")
            if isinstance(i, int) and 0 <= i < len(self.declared):
                continue
            views.append(self._view(i if isinstance(i, int) else len(views),
                                    str(rec.get("name") or "?"), rec))
        return views

    @staticmethod
    def _view(index: int, name: str, rec: Optional[dict]) -> StepView:
        glyph, word = narrate.step_mark(rec)
        if rec is None:
            return StepView(index=index, name=name, state="queued", glyph=glyph, word=word)
        live = rec.get("state")
        return StepView(
            index=index, name=name,
            state=str(live or rec.get("outcome") or "queued"), glyph=glyph, word=word,
            # a duration only once there is one: `_settle` drops `state`/`started` when the
            # step ends, and `seconds` is 0.0 on a record that has not run yet, which would
            # render as a finished step that took no time.
            seconds=None if live else _as_float(rec.get("seconds")),
            started=_as_float(rec.get("started")) if live == "running" else None,
            detail=str(rec.get("detail") or ""),
            deltas=tuple(narrate.geometry_delta_rows(rec)),
            plots=tuple(rec.get("plots") or ()),
        )

    @property
    def total(self) -> int:
        return len(self.rows())

    @property
    def done(self) -> int:
        """Steps that have settled. Mirrors `shell._progress_panel`'s footer count — a running
        step is not done, and a step with no record has not started."""
        return sum(1 for v in self.rows() if v.state not in ("queued", "running"))

    def to_dict(self) -> dict:
        return {"declared": [str(s.get("name") or "?") for s in self.declared],
                "done": self.done, "total": self.total,
                "rows": [v.to_dict() for v in self.rows()],
                "observer_errors": list(self.observer_errors)}


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
