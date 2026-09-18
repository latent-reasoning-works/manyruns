"""The figure, kept as something you can REMAKE rather than only as a picture you kept.

A run used to leave a 600×480 PNG at `dpi=120`. That is the right thing to look at in a terminal
and the wrong thing to put in a paper, and there was no third option: the artist tree that drew
it was closed at the end of `_save_scatter`, so the only route to a publication figure was to
re-run the analysis — refit PHATE, refit the flow — and hope it landed the same.

**The object matplotlib gives you before any raster is `Figure`, and it re-renders.** Measured on
2,700 points from ONE artist tree: 640×480 PNG in 25.6 ms, 1920×1440 at 300 dpi in 57.9 ms, SVG
in 47 ms, PDF in 623 ms, and an RGBA buffer with no file at all in 16.9 ms. `sc.get_offsets()`
hands the coordinates back exactly. One tree answers every medium.

**But a `Figure` is not a format.** `pickle.dumps(fig)` works — 280 KB — and is locked to the
matplotlib that wrote it, which makes it a cache, not an archive. The durable object is the
INPUT: coordinates, a colour channel, and how to map it. That is what this module persists, and
it is deliberately the shape `manylatents`' own `PlotEmbeddings` callback already writes
(`embedding_data_*.npz` + `.json`: `coords`, `color_values`, and `{color_by, is_categorical,
category_colors}`) — so when `api.run` grows a callback passthrough, the engine's spec drops into
this reader instead of becoming a second convention.

TWO OF THOSE THREE ARE NOW WRITTEN HERE, not merely mirrored: `_save_scatter` sets
`is_categorical` and `category_colors` (`{name: hex}`, the label strings as keys — manylatents'
`PlotEmbeddings._export_plot_data` shape exactly) whenever the label kind is an unordered
identity, and `figure()` draws them as a legend. That is what puts the eight pbmc3k cell-type
names into the record instead of onto an anonymous 0.0–1.0 viridis ramp.

Display helpers also record `color_by`/`color_kind` and original sample identities selected
onto the plotted rows. Legacy specs remain readable; recolouring one discloses its positional
alignment. These are figure views, with no effect on analysis labels or decision rows.

**One builder, every medium.** `figure()` is the only place a scatter is constructed in this
product. The run calls it to save the PNG a person looks at; `export()` calls it to emit the PDF
they submit. "What you saw is what you publish" holds by construction rather than by two
functions being kept in step.

**Addressed by `run_id`, like every other artifact.** The spec sits beside the PNG in the run's
`plots/` folder, so `index.jsonl` already reaches it and a figure from six months ago is
re-emittable without re-running anything.

**Never the input matrix.** `coords` is the 2-D projection being drawn, not `state["X"]` — the
line `artifacts.PERSISTED` draws, for the same reason: geometry leaves, the person's data does
not.
"""
from __future__ import annotations

import json
import functools
import threading
from pathlib import Path
from typing import Any, Optional

#: Suffixes of the two files one spec is made of. Arrays in `.npz`, everything else in `.json` —
#: manylatents' split, kept rather than improved on, because the point is that the engine's own
#: callback output can be read by the same loader.
ARRAYS, META = ".spec.npz", ".spec.json"

#: **Vector is never traded away silently, at any point count.** Rasterising the points inside an
#: SVG makes the file smaller and makes it stop being the thing that was asked for — an editable
#: scatter where every cell is its own path. So `rasterized` defaults to False everywhere and is
#: only ever an explicit request.
#:
#: The size it costs is real and is recorded so the choice can be made with a number rather than
#: a surprise. Measured at 600 dpi with `bbox_inches="tight"`:
#:
#:     points    SVG vector   PDF vector  |  SVG rasterised   PDF rasterised
#:      2,700        432 KB        75 KB  |         886 KB           503 KB
#:     50,000      7,715 KB     1,129 KB  |         943 KB           602 KB
#:
#: Note the crossover: below ~10k points rasterising is BIGGER, so it buys nothing at the sizes
#: this product usually sees. Above it a vector scatter grows without bound, which is where a
#: journal's upload limit and a vector editor both give out — `large()` says so in the toast, and
#: the person decides.
RASTER_ABOVE = 10_000

#: Categories past which a legend stops being a key and becomes the figure. Measured at
#: `figsize=(5,4), dpi=120` — what `_save_scatter` writes the run's PNG at:
#:
#:     categories    legend px    of canvas height
#:              8      137×112                0.23
#:             20       80×273                0.57
#:             35       80×474    0.99 — TALLER THAN THE AXES IT EXPLAINS
#:
#: 20 is also `tab20`'s size, past which the colours themselves repeat, so this is a correctness
#: boundary and not only a legibility one. It matters because `vocab.GROUP_KEYS` admits `leiden`,
#: `louvain`, `cluster` and `clusters`, which routinely carry 20–50 levels — colour by a
#: clustering and you cross it immediately.
#:
#: Above it the spec STILL CARRIES EVERY NAME in `category_colors`; only the DRAWING backs off.
#: A later reader keeps the key, and the picture stays a picture.
LEGEND_MAX = 20

#: What a quickdrop writes, in the order it is reported. Three, because they answer three
#: questions and someone who has just found the embedding they wanted should not have to know
#: which one they need yet: SVG is the editable figure that goes into Illustrator or Inkscape,
#: PDF is what a journal takes, PNG at 300 dpi is what goes in the slide deck an hour later.
#: All three come from ONE artist tree, so they cannot disagree.
QUICKDROP = ("svg", "pdf", "png")

#: The dpi a raster export uses. 300 is the floor most journals state for a figure, and it is
#: 2.5x the 120 the run's own PNG is written at — which is the whole reason this module exists.
EXPORT_DPI = 300


# Matplotlib has process-global rendering state. Every product rendering transaction uses
# this re-entrant lock, including exports and post-hoc views from daemon UI workers.
RENDER_LOCK = threading.RLock()


def serialized_render(fn):
    @functools.wraps(fn)
    def locked(*args, **kwargs):
        with RENDER_LOCK:
            return fn(*args, **kwargs)
    return locked


def _names(obs_names: Any, n: int) -> Any:
    import numpy as np

    names = np.asarray(obs_names, dtype=str)
    if names.ndim != 1 or len(names) != n:
        raise ValueError(f"obs_names must have one name per coordinate ({n})")
    if len(np.unique(names)) != n:
        raise ValueError("obs_names must be unique")
    return names


def stem_for(png_path: Path | str) -> Path:
    """The spec files' stem for a saved figure. `plots/phate.png` → `plots/phate`."""
    return Path(png_path).with_suffix("")


def _paths(png_or_stem: Path | str) -> tuple[Path, Path]:
    stem = stem_for(png_or_stem)
    return stem.with_name(stem.name + ARRAYS), stem.with_name(stem.name + META)


def save(png_path: Path | str, coords: Any, color_values: Any, meta: dict,
        trajectories: Any = None, *, obs_names: Any = None) -> Optional[str]:
    """Write one figure's spec beside its PNG. Returns the `.json` path, or None.

    ``obs_names`` supplies one unique name per coordinate, stored as a Unicode array
    in npz, never in JSON or as object pickle. Invalid identities raise ValueError.

    Best effort, the same way `artifacts.save_array` is: a run that produced a valid figure must
    not become a failed run because a disk was full. The caller records the absence.

    `trajectories` (MIOFlow's sample paths, `(n_bins, n_trajectories, 2)`) rides in the SAME
    `.npz` as `coords`/`color_values` — one more array key, not a second file — so `load()`
    hands it back with no format the reader has to know about ahead of time.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    names = None if obs_names is None else _names(obs_names, len(coords))
    arrays_path, meta_path = _paths(png_path)
    try:
        arrays = {"coords": np.asarray(coords)[:, :2]}
        if color_values is not None:
            arrays["color_values"] = np.asarray(color_values)
            if arrays["color_values"].dtype.kind == "O":
                arrays["color_values"] = arrays["color_values"].astype(str)
        if names is not None:
            arrays["obs_names"] = names
        if trajectories is not None:
            arrays["trajectories"] = np.asarray(trajectories)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_path, **arrays)
        metadata = {k: v for k, v in meta.items()
                    if k not in ("coords", "color_values", "trajectories", "obs_names")}
        meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        return str(meta_path)
    except (OSError, ValueError, IndexError, TypeError):
        return None


def load(png_or_stem: Path | str) -> Optional[dict]:
    """Read a spec back: `{coords, color_values, **meta}`. None if it is not there.

    Takes the PNG's path as readily as the stem, because the step record carries the PNG and no
    caller should have to know the naming convention to get from one to the other.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    arrays_path, meta_path = _paths(png_or_stem)
    if not (arrays_path.exists() and meta_path.exists()):
        return None
    try:
        with np.load(arrays_path, allow_pickle=False) as z:
            spec: dict = {k: z[k] for k in z.files}
        spec.update(json.loads(meta_path.read_text()))
        return spec
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def large(spec: dict) -> bool:
    """Is this figure past the size where a fully-vector file starts to hurt? Advisory only —
    nothing in this module acts on it. It exists so a caller can put the number in front of the
    person instead of deciding for them. See `RASTER_ABOVE`."""
    coords = spec.get("coords")
    return coords is not None and len(coords) > RASTER_ABOVE


@serialized_render
def figure(spec: dict, *, figsize: tuple = (5, 4), dpi: int = 120,
           rasterized: bool = False) -> Any:
    """Build the artist tree. **The only place a scatter is constructed in this product.**

    `rasterized=False` ALWAYS, unless a caller explicitly asks otherwise, and at any point count.
    An SVG of a scatter is wanted because every point is its own path — that is the difference
    between a figure you can edit and a picture of one — and an export that quietly swapped in a
    bitmap above some threshold would return something that opens, looks right, and is not what
    was asked for. It stays a parameter because a journal upload limit is a real constraint; it
    is not a default, because the size is the caller's trade to make.

    When it IS set, only the POINTS become a bitmap: axes, ticks, labels and the title stay
    vector, which is what journals ask for and what keeps the file openable in a vector editor.

    **Trajectories** (`spec["trajectories"]`, `(n_bins, n_trajectories, 2)` — MIOFlow's sample
    paths) draw as thin lines with an arrowhead at the endpoint, ON TOP of the scatter, one per
    trajectory. They are never rasterized regardless of `rasterized`: unlike the points, there
    is no size problem that motivates it (a few dozen–hundred paths, not thousands of points),
    and an SVG/PDF export is exactly where "which way is it flowing" needs to stay legible at
    zoom — a bitmap arrow blurs into a smear a rasterized point never does.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyArrowPatch

    coords = spec["coords"]
    color = spec.get("color_values")

    # Unregistered artist tree: failed construction cannot leak a pyplot-managed figure.
    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.subplots()
    # ONE `ax.scatter`, and it stays above the trajectory block so `_ml_lightning`'s arrows still
    # draw OVER a categorical scatter exactly as they do over a continuous one. Categorical
    # colour is a different `c` (explicit per-point hex, no cmap), not a different code path:
    # a second scatter call is how "what you saw is what you publish" stops being structural.
    cats = spec.get("category_colors") if spec.get("is_categorical") else None
    kw = ({"c": [cats.get(str(v), "#808080") for v in color]} if cats else
          {"c": color if color is not None else coords[:, 0],
           "cmap": spec.get("cmap", "viridis")})
    if not cats and color is not None:
        import numpy as np

        kw["c"] = np.ma.masked_invalid(np.asarray(color, dtype=float))
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=spec.get("point_size", 8),
                    rasterized=rasterized, zorder=2, **kw)
    traj = spec.get("trajectories")
    if traj is not None and len(traj) >= 2:
        # `(n_bins, n_trajectories, d)`, so a path is a COLUMN — `traj[:, i, :]` below, not a
        # row. Only the count is needed here; the bin count is `len(traj)`, already guarded above.
        n_traj = traj.shape[1]
        # More paths crowd the axes faster than more points do (each is a multi-segment line,
        # not a dot), so alpha backs off as the count grows rather than staying fixed and
        # turning "arrows" into a grey wash once `n_trajectories` climbs toward its 100 default.
        alpha = max(0.15, min(0.6, 8.0 / max(n_traj, 1)))
        for i in range(n_traj):
            path = traj[:, i, :]
            ax.plot(path[:, 0], path[:, 1], color="0.25", linewidth=0.7, alpha=alpha,
                    zorder=1, solid_capstyle="round")
            ax.add_patch(FancyArrowPatch(
                tuple(path[-2]), tuple(path[-1]), arrowstyle="-|>", mutation_scale=8,
                color="0.25", alpha=min(alpha * 2, 0.9), linewidth=0, zorder=3,
            ))
    ax.set(title=spec.get("title") or "", xlabel=spec.get("xlabel", "dim 1"),
           ylabel=spec.get("ylabel", "dim 2"))
    # The key is drawn only when the colour means something a reader can look up, and WHICH key
    # follows from what the colour IS. An unordered identity (`vocab.CATEGORICAL_KINDS` — a cell
    # type, a cluster) gets a LEGEND: the names, against their swatches. A magnitude gets the
    # colorbar. Without labels the shading is the embedding's own first axis — decoration, so
    # the plot is not a flat wall of one colour — and a scale bar on it would invite reading a
    # quantity that is not one. `_save_scatter` made that last distinction; it is kept here.
    #
    # `bbox_to_anchor` puts the legend OUTSIDE the axes, and `tight_layout` below is what then
    # makes room for it. Measured on the real act-2 run (2,638 pbmc3k cells, 8 types) at
    # `figsize=(5,4), dpi=120`: the axes narrow 372.5 → 324.0 px (13.0%) against the
    # viridis+colorbar this replaces, the legend is 137.0×112.2, and its right edge lands at
    # x=580.6 of 600 — on canvas, nothing clipped. Past `LEGEND_MAX` the names are still in the
    # spec but the legend is not drawn; see that constant for the measurements.
    #
    # `save` writes the meta with `sort_keys=True`, so `cats.items()` comes back alphabetical
    # and legend ORDER is deterministic across a reload — the export cannot disagree with the
    # PNG about which name sat where.
    if cats and len(cats) <= LEGEND_MAX:
        from matplotlib.lines import Line2D

        ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=4,
                                  color=c, label=n) for n, c in cats.items()],
                  loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6, frameon=False,
                  handletextpad=0.3, borderaxespad=0.0, labelspacing=0.3)
    elif spec.get("labeled"):
        fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    return fig


@serialized_render
def export(png_or_stem: Path | str, out_dir: Path | str, stem: str,
           formats: "tuple[str, ...]" = QUICKDROP, dpi: int = EXPORT_DPI,
           rasterized: bool = False, *, spec: Optional[dict] = None) -> list[str]:
    """Re-emit one figure at publication quality, in every format asked for. Paths written.

    ONE artist tree, N files — built once and saved once per format, so the SVG, the PDF and the
    PNG cannot disagree with each other or with the PNG the terminal showed. That is the whole
    property: `figure()` is also what the run drew with, over the same numbers.

    Rebuilt from the SPEC rather than from a live `Figure`, which is what makes this work months
    later and from another process — the run that drew it is long closed. `[]` when there is no
    spec (a figure from before this existed, or a failed write), which the caller reports rather
    than presenting an empty folder as success.

    `rasterized` is forwarded and defaults to False at any point count: the SVG is a real vector
    scatter or this function did not do its job.

    A captured ``spec`` preserves a live attempt across deferred rendering even if tuning
    renames or replaces its sidecars. When supplied, the source path is never read again.
    """
    if spec is None:
        spec = load(png_or_stem)
    if spec is None:
        return []
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    written: list[str] = []
    try:
        out.mkdir(parents=True, exist_ok=True)
        fig = figure(spec, figsize=(6.4, 4.8), dpi=dpi, rasterized=rasterized)
    except (OSError, ValueError, KeyError):
        return []
    try:
        for fmt in formats:
            path = out / f"{stem}.{fmt}"
            # `bbox_inches="tight"` because an export goes into a document, where the figure's
            # own whitespace becomes someone else's layout problem.
            fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
            written.append(str(path))
    except (OSError, ValueError):
        pass
    finally:
        plt.close(fig)
    return written



def join(spec: dict, obs: Any, col: str) -> tuple[Any, bool]:
    """Return ``(aligned_series, positional)`` for the plotted survivors.

    Named joins reject duplicate source/spec identities and count missing survivors. Extra
    source rows are allowed. Legacy specs accept only equal-length positional alignment,
    including sources with duplicate names since no identity lookup is performed.
    """
    import pandas as pd

    if col not in obs.columns:
        raise ValueError(f"unknown colour column {col!r}; available columns: "
                         + ", ".join(map(str, obs.columns)))
    if spec.get("obs_names") is None:
        if len(obs) != len(spec["coords"]):
            raise ValueError(f"positional alignment length mismatch: {len(obs)} source rows, "
                             f"{len(spec['coords'])} coordinates")
        return obs[col].copy(), True
    index = pd.Index(obs.index.astype(str))
    duplicate = int(index.duplicated(keep=False).sum())
    if duplicate:
        raise ValueError(f"duplicate source names: {duplicate} rows")
    names = _names(spec["obs_names"], len(spec["coords"]))
    positions = index.get_indexer(names)
    missing = int((positions < 0).sum())
    if missing:
        raise ValueError(f"missing source names: {missing} plotted rows")
    return obs[col].iloc[positions].copy(), False


def recolor(spec: dict, values: Any, key: Optional[str], kind: str) -> dict:
    """Pure display view: preserve coordinates/IDs/paths and replace the colour encoding.

    ``key=None`` is legal for default labels whose provenance was not supplied. Arrays are
    copied so even a later consumer mutation cannot change the original spec.
    """
    import copy
    from manyruns.pipeline.io import _categorical
    from manyruns.pipeline.loading import color_column_info, color_values_of

    if len(values) != len(spec["coords"]):
        raise ValueError(f"colour length {len(values)} does not match {len(spec['coords'])} coordinates")
    result = copy.deepcopy(spec)
    for stale in ("category_colors", "is_categorical", "labeled", "cmap"):
        result.pop(stale, None)
    result.update(color_by=key, color_kind=kind, cmap="viridis", labeled=kind == "continuous")
    info = color_column_info(values)
    result.update({k: info[k] for k in ("n_missing", "n_nonfinite")})
    normalized = color_values_of(values, kind)
    result["color_values"] = normalized
    if kind == "categorical":
        color, cats, cmap = _categorical(normalized)
        result.update(color_values=color, category_colors=cats, cmap=cmap, is_categorical=True)
    base = result.setdefault("base_title", result.get("title", ""))
    result["title"] = f"{base} — colour by {key}" if key is not None else base
    return result


def safe_column(key: Optional[str]) -> str:
    """Bounded filename component: readable ASCII slug plus a stable Unicode-key digest."""
    import hashlib
    import re
    import unicodedata

    raw = "default" if key is None else str(key)
    ascii_key = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", ascii_key).strip("-_")[:40] or "column"
    return slug + "-" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def remove_view(path: str | Path) -> None:
    """Remove only this newly allocated view and its sidecars (also used on cancellation)."""
    for part in (Path(path), *_paths(path)):
        part.unlink(missing_ok=True)


@serialized_render
def write_view(png_path: str | Path, spec: dict) -> str:
    """Persist a unique post-hoc PNG + spec; never overwrite an original or an earlier view.

    Render in memory, then stage all three files under a hidden directory. Publish the PNG
    last so an app exiting during a daemon render cannot leave a PNG without its spec.
    On failure, including interruption, remove only this view's files and close the figure.
    """
    from io import BytesIO
    import tempfile

    source = Path(png_path)
    prefix = f"{source.stem[:40]}-by-{safe_column(spec.get('color_by'))}-view-"
    fig, path = None, None
    try:
        fig = figure(spec)
        with BytesIO() as rendered:
            fig.savefig(rendered, format="png", dpi=120)
            with tempfile.TemporaryDirectory(prefix=f".{prefix}", dir=source.parent) as directory:
                staging = Path(directory)
                candidate = source.parent / f"{staging.name[1:]}.png"
                if any(part.exists() for part in (candidate, *_paths(candidate))):
                    raise FileExistsError(f"colour view already exists: {candidate}")
                path = candidate
                staged = staging / path.name
                staged.write_bytes(rendered.getvalue())
                saved = save(staged, spec["coords"], spec.get("color_values"), spec,
                             trajectories=spec.get("trajectories"), obs_names=spec.get("obs_names"))
                if not saved or not all(p.is_file() and p.stat().st_size
                                        for p in (staged, *_paths(staged))):
                    raise OSError("could not persist the complete figure spec")
                for temporary, final in zip((*_paths(staged), staged), (*_paths(path), path)):
                    temporary.replace(final)
                if not all(p.is_file() and p.stat().st_size for p in (path, *_paths(path))):
                    raise OSError("could not persist the complete figure spec")
                return str(path)
    except BaseException:
        if path is not None:
            remove_view(path)
        raise
    finally:
        if fig is not None:
            import matplotlib.pyplot as plt

            plt.close(fig)
