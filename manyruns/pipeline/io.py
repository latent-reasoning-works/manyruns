"""Side-effecting edges of the compute path: console suppression and figure writing.

Split out because everything here is a *side effect on global state* — logger levels,
the warnings filter, matplotlib's backend — which is exactly the state a pooled worker
running many recipes concurrently can corrupt. Keeping it in one small module makes that
surface auditable instead of scattered through 1000 lines.
"""
from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path

from manyruns.figspec import serialized_render

# noisy libraries the CFlows compute pulls in (Lightning banners/tables, PHATE's SGD-MDS
# notes, scanpy/sklearn matmul RuntimeWarnings). The REPL prints its own clean line, so
# swallow their console output during a step.
_NOISY_LOGGERS = (
    "lightning", "lightning.pytorch", "pytorch_lightning",
    "phate", "tasklogger", "graphtools", "scprep", "anndata",
)


@contextlib.contextmanager
def quiet():
    """Silence library stdout/stderr, warnings, and numpy FP errors during heavy compute.
    Yields a buffer holding whatever was captured (useful to surface on failure)."""
    prev = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    buf = io.StringIO()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            warnings.simplefilter("ignore")
            # ...but NEVER swallow a row-alignment complaint. The only such check anywhere
            # downstream is this UserWarning in manylatents' PrecomputedDataModule, and a
            # blanket ignore makes a misaligned matrix — cells attributed to the wrong
            # donor — completely silent. Misalignment is a bug, not noise: promote to raise.
            warnings.filterwarnings(
                "error", category=UserWarning, message=r".*does not match embeddings length.*"
            )
            try:
                import numpy as np

                with np.errstate(all="ignore"):
                    yield buf
            except ImportError:
                yield buf
    finally:
        for name, lvl in prev.items():
            logging.getLogger(name).setLevel(lvl)


def _labels_to_numeric(labels):
    """Per-cell labels (e.g. timepoints day0…day9) → a numeric color array in [0, 1], or None.

    Same magnitude-preserving mapping the trajectory model uses (`vocab.numeric_time`), so plots
    are coloured by the real time spacing, not by even ranks."""
    if labels is None:
        return None
    from manyruns import vocab

    return vocab.numeric_time(labels)


#: Okabe & Ito's colour-blind-safe qualitative set, in their published order. Eight colours, which
#: is exactly what both demo fixtures carry — 8 cell types on act 2, 8 branches on act 1 — so a
#: category never wraps onto a repeated swatch. Used up to eight; see `_categorical` for why it
#: does not stretch further and for the `tab10` collision it replaces.
OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000")


def _categorical(labels):
    """Per-cell identity labels → `(values, {name: hex}, palette)` — the strings KEPT, not ranked.

    The sibling of `_labels_to_numeric`, and the whole difference is `vocab.CATEGORICAL_KINDS`:
    that one folds an ORDERED axis onto a magnitude, this one refuses to invent an order and
    hands back a key a reader can look the colour up in. Neither replaces the other — a `time`
    kind still goes through `_labels_to_numeric`, which is what keeps day0/day3/day9 at 0, ⅓, 1
    instead of even ranks.

    `.astype(str)` IS LOAD-BEARING. `loading.labels_of` returns `np.asarray(obs[col].values)`,
    dtype OBJECT on both bundled fixtures; `np.savez_compressed` pickles an object array and
    `figspec.load` opens with `allow_pickle=False`, so the spec raises ValueError on read, comes
    back None, and `figspec.export` returns [] forever — with the PNG still on disk and nothing
    saying why. Measured. A `<U` array round-trips.

    `category_colors` is manylatents' own key, not ours: `PlotEmbeddings._export_plot_data`
    writes `{str(label): mcolors.to_hex(...)}` on its categorical branch, which is the format
    this module's header promises to mirror rather than to improve on. Hex, not RGBA tuples, for
    the same reason — and because `figspec.save` catches `(OSError, ValueError, IndexError)` and
    NOT the `TypeError` a numpy scalar in `meta` would raise, which `_save_scatter`'s blanket
    `except Exception` would then swallow into a silently missing spec.

    `matplotlib.colormaps`, never `pyplot`: this runs BEFORE `figspec.figure` calls
    `matplotlib.use("Agg")` (its output feeds the spec that `figure` is handed), and importing
    pyplot ahead of that is the documented hang in `_save_scatter` below. `colormaps` and
    `colors.to_hex` claim no backend at all — verified: neither imports pyplot.

    NOT STABLE UNDER A NARROWING THAT REMOVES A WHOLE CATEGORY, deliberately. Names come from
    `sorted(set(...))` over the labels PRESENT, so a filter that dropped all 15 Megakaryocytes
    would shift every later name one colour. The alternative — freezing the list at run open
    while the values are rebuilt on each narrowing — is HAZARD 7 exactly: two halves of one
    colour channel computed at different moments. Recomputing both here, from one array, at one
    moment, is self-consistent by construction. It does not bite the demo (`filter_cells
    (min_genes=200)` drops 0 of 2,700 cells on pbmc3k).

    OKABE-ITO UP TO EIGHT, AND THAT IS THE FOLLOW-UP AN EARLIER REVISION OF THIS DOCSTRING
    PROMISED. It used `tab10`, which is qualitative but not colour-blind safe, and the specific
    collision was unlucky: on the act-2 fixture `tab10` puts CD4 T cells on green #2ca02c and
    CD8 T cells on red #d62728 — the classic deuteranope confusion pair — and those are also the
    two most interleaved populations in that figure, so a red-green colour-blind viewer saw the
    largest and fourth-largest populations merge into one mass.

    Okabe & Ito's set is eight colours chosen to stay distinct under the common dichromacies, and
    eight is exactly what both demo fixtures carry (8 cell types, 8 branches), so nothing wraps.
    Past eight this falls back to `tab10`/`tab20` — a safe palette that has run out of safe
    colours is not safer than an unsafe one, it just hides the wrap. Above 20 the swatches
    repeat outright; see `figspec.LEGEND_MAX`, which is why the DRAWING backs off there.

    Black is included as Okabe-Ito specifies. The figure's own canvas is white (matplotlib's
    default; the terminal's dark theme paints around the PNG, not inside it), so it reads as a
    colour rather than as absence.
    """
    import numpy as np
    from matplotlib import colormaps
    from matplotlib import colors as mcolors

    values = np.asarray(labels).astype(str)
    names = sorted(set(values.tolist()))
    if len(names) <= len(OKABE_ITO):
        return values, {n: OKABE_ITO[i] for i, n in enumerate(names)}, "okabe-ito"
    palette = "tab10" if len(names) <= 10 else "tab20"
    base = colormaps[palette]
    return values, {n: mcolors.to_hex(base(i % base.N)) for i, n in enumerate(names)}, palette


@contextlib.contextmanager
def _numeric_quiet():
    """Silence chatty numeric libs for a compute loop — but never a row-misalignment
    complaint (see `quiet`). Does NOT redirect stdout/stderr; engines print progress."""
    import warnings as _w

    import numpy as np

    with _w.catch_warnings(), np.errstate(all="ignore"):
        _w.simplefilter("ignore")
        _w.filterwarnings(
            "error", category=UserWarning, message=r".*does not match embeddings length.*"
        )
        yield



_UNSET_COLOR_BY = object()


@serialized_render
def _save_scatter(emb, color, out_dir: Path, fname: str, plots: list, title: str, *,
                  cmap: str = "viridis", colorbar: bool = True, trajectories=None,
                  categories=None, color_by=_UNSET_COLOR_BY, color_kind=None, obs_names=None,
                  skipped_channels=None) -> str | None:
    """Save a 2-D scatter of the embedding locally, AND the spec that can remake it.

    The scatter itself is no longer built here: `figspec.figure` is the one place in this product
    a scatter is constructed, and `figspec.export` calls the same function to emit the PDF a
    person submits. That is what makes "what you saw in the console is what you publish" a
    property of the code rather than of someone keeping two plotting calls in step.

    THE PNG IS STILL WRITTEN AT dpi=120, unchanged, because it is what a terminal looks at and
    every existing reader (`rec["plots"]`, `figures.FigurePane`, `shell._render_plots`) expects
    it at that path. What is new is the spec beside it — a run that used to leave only a picture
    now leaves the numbers the picture was made of, so a publication-quality version does not
    require re-running the analysis.

    `cmap`/`colorbar` are per-caller because they mean different things for different colour
    channels: `dpt`/`mioflow` colour by a CONTINUOUS pseudotime, where a viridis colorbar reads
    a magnitude; `discretize_time` colours by a small integer BIN, where a colorbar invites
    reading a magnitude that group index is not one (`vocab.discretize_pseudotime`'s point) —
    a discrete/qualitative cmap (e.g. `tab20`) with no colorbar is the honest picture there.

    `categories` IS THAT THIRD CASE, and it goes one step further than the `discretize_time`
    one: it carries the label STRINGS for an unordered identity (`vocab.CATEGORICAL_KINDS` —
    a cell type, a cluster, a condition arm), so the figure gets a qualitative colour AND the
    names to look it up in. `_categorical` maps them, the spec carries `is_categorical` +
    `category_colors`, and `figspec.figure` draws a legend instead of a colorbar. Without it,
    measured on `data/pbmc3k_annotated.h5ad`: eight named immune types went out on a continuous
    viridis colorbar reading 0.0–1.0, alphabetically ranked, and the eight names appeared in
    ZERO files under `outputs/` — so neither the picture nor the record said what a colour was.

    The names ride in the spec, not only in the drawing, which is what lets `figspec.export` and
    any later reader rebuild the key months after the run closed — the same reason `labeled`
    travels rather than being re-derived.

    What the key costs, measured on the REAL act-2 run (`manyruns run data/pbmc3k_annotated.h5ad
    --recipe embed`, 2,638 cells, 8 types) on the 600×480 PNG this writes at `dpi=120`:

        viridis + colorbar (before)   axes 372.5×366.8   47.4% of canvas
        legend (after)                axes 324.0×366.8   41.3%   legend 137.0×112.2
        no key at all                 axes 465.7×366.8   59.3%

    So the picture narrows 48.5 px, 13.0% — and the honest comparison is the first row, not the
    third: this SWAPS one key for a truthful one rather than spending a picture on a legend. The
    legend's right edge lands at x=580.6 of 600, nothing clipped, from `bbox_to_anchor` outside
    the axes plus the `tight_layout` that was already there.

    `trajectories`, when given, is `(n_bins, n_trajectories, d)` — MIOFlow's own sample paths
    (`mioflow._run_mioflow_experiment`, sourced from `MIOFlow.trajectories`, which
    `on_train_end` computes unconditionally on every fit) — drawn as arrows over the scatter by
    `figspec.figure`. Every other caller passes nothing and gets the plain scatter, unchanged.

    ``skipped_channels``, when supplied, collects ``(color_by, reason)`` alongside each
    logged all-missing/nonfinite skip. The display wrapper returns these to its callers.
    """
    import numpy as np
    from manyruns import figspec as _figspec
    from manyruns.pipeline.loading import color_column_info, color_selection_reason, color_values_of

    # Positional legacy calls are best effort and retain the subsampling fallback. Explicit
    # channels/identities are an alignment contract: they must fail visibly, never turn grey.
    explicit = color_by is not _UNSET_COLOR_BY or color_kind is not None or obs_names is not None
    if color_by is _UNSET_COLOR_BY:
        color_by = None
    emb = np.asarray(emb)
    if explicit:
        for values in (color, categories):
            if values is not None and len(values) != len(emb):
                raise ValueError(f"colour length {len(values)} does not match {len(emb)} coordinates")
        if obs_names is not None:
            obs_names = _figspec._names(obs_names, len(emb))
        if color is not None:
            reason = color_selection_reason(color)
            if reason is not None:
                logging.getLogger(__name__).warning("skipping colour channel %r: %s", color_by, reason)
                if skipped_channels is not None:
                    skipped_channels.append((color_by, reason))
                return None
    fig = None
    try:
        plots_dir = Path(out_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        if color is not None and len(color) != len(emb):
            color = None
        cats = None
        info = color_column_info(color) if color_kind is not None and color is not None else {}
        if color_kind is not None and color is not None:
            color = color_values_of(color, color_kind)
            if color_kind == "categorical":
                categories = color
        if categories is not None and len(categories) == len(emb):
            color, cats, cmap = _categorical(categories)
        spec = {"coords": emb[:, :2], "color_values": color,
                "title": title, "xlabel": "dim 1", "ylabel": "dim 2",
                "cmap": cmap, "point_size": 8,
                "labeled": color is not None and colorbar and cats is None}
        if cats is not None:
            spec.update(is_categorical=True, category_colors=cats)
        if explicit:
            spec.update(color_by=color_by, color_kind=color_kind)
            spec.update({k: info[k] for k in ("n_missing", "n_nonfinite") if k in info})
        traj = None if trajectories is None else np.asarray(trajectories)
        if traj is not None and traj.ndim == 3 and traj.shape[-1] >= 2:
            spec["trajectories"] = traj[:, :, :2]
        path = plots_dir / fname
        fig = _figspec.figure(spec, figsize=(5, 4), dpi=120, rasterized=False)
        fig.savefig(path, dpi=120)
        saved = _figspec.save(path, spec["coords"], color, spec,
                              trajectories=spec.get("trajectories"), obs_names=obs_names)
        if explicit and (not saved or not all(p.is_file() and p.stat().st_size
                                              for p in (path, *_figspec._paths(path)))):
            raise OSError("could not persist the complete display figure spec")
        plots.append(str(path))
        return str(path)
    except Exception:  # legacy plotting remains best effort
        if explicit:
            raise
        return None
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)


class DisplayScatterResult(list[str]):
    """Persisted PNG paths with caller-readable reasons for skipped colour channels."""

    def __init__(self):
        super().__init__()
        self.skipped_channels: list[tuple[str | None, str]] = []


def save_display_scatter(emb, channels, out_dir, fname, plots, title, *, obs_names=None,
                         trajectories=None) -> DisplayScatterResult:
    """Draw named display channels on the supplied, already selected row axis.

    ``save_display_scatter(emb, channels, out_dir, fname, plots, title, *, obs_names=None,
    trajectories=None) -> DisplayScatterResult`` accepts channels with ``key``, ``values``, ``kind``.
    Supply raw categorical values, including missing/nonfinite entries: counts are computed
    on this selected row axis before normalization to the labelled missing category.
    It returns a list subclass and appends every successfully persisted PNG path to ``plots``.
    ``result.skipped_channels`` contains ``(key, reason)`` for every skipped channel, in
    input order; ``key`` may be None for default labels. This per-call list is empty when
    nothing was skipped. Read it even when the path list is empty. Callers can display or
    retain ``f"skipping colour channel {key!r}: {reason}"``; logs alone may be hidden by the UI.
    Existing iteration, indexing and list equality still operate on paths alone. The first
    channel uses the base ``fname``; additional paths use a safe slug/digest and ordinal.
    All-missing/nonfinite selections are skipped with a logged notice before writing;
    existing default figures are preserved and other channels can still be persisted.
    Empty channels keep the unlabelled rendering. ``key=None`` is legal for default labels:
    a bare vector cannot supply a column name. Explicit lengths/identities raise on mismatch;
    legacy positional :func:`_save_scatter` calls retain their best-effort grey fallback.
    """
    from manyruns import figspec

    written = DisplayScatterResult()
    for i, channel in enumerate(channels or [None]):
        name = fname if i == 0 else (f"{Path(fname).stem}-by-"
                                    f"{figspec.safe_column(channel['key'])}-{i}.png")
        persisted = []
        path = _save_scatter(emb, None if channel is None else channel["values"],
                             out_dir, name, persisted, title, obs_names=obs_names,
                             trajectories=trajectories,
                             color_by=None if channel is None else channel["key"],
                             color_kind=None if channel is None else channel["kind"],
                             skipped_channels=written.skipped_channels)
        if path is not None and figspec.load(path) is not None:
            written.append(path)
            plots.append(path)
    return written
