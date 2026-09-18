"""Eight named cell types must come out as a KEY, not as an anonymous 0.0–1.0 ramp.

The defect these fence off was measured on `data/pbmc3k_annotated.h5ad`: eight CATEGORICAL immune
types were drawn with a CONTINUOUS viridis colorbar, ranked ALPHABETICALLY onto an even [0,1]
ramp — so NK cells (154 of 2,638 cells) got the brightest yellow and 15 Megakaryocytes the second
brightest, while CD4 T (1,144) and CD8 T (316) landed on adjacent teals. Worse than the picture:
the cell-type STRINGS appeared in ZERO files under `outputs/`, so the record could not say what a
colour had meant either.

Three properties are pinned here, and the third is the one that bites silently:

  * the NAMES reach the persisted spec and the exported figure — the finding's own check, turned
    into a fence;
  * a `time` kind is UNTOUCHED — this is a guard against over-reach, and it is the same
    assertion `test_apply_step.py::test_the_colour_the_plot_is_drawn_with_is_rebuilt_after_a_
    narrowing` makes, for the same HAZARD 7 reason;
  * `color_values` is a `<U` array and not `object` — an object array pickles into the `.npz`,
    `figspec.load` opens with `allow_pickle=False` and refuses it, `load` returns None, and
    `export` returns `[]` FOREVER while the PNG sits there looking fine. Nothing raises.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from manyruns import figspec, pipeline  # noqa: E402
from manyruns.pipeline import io as _io  # noqa: E402

#: The real act-2 fixture's eight `cell_type` values, alphabetical — the order `sorted(set(...))`
#: produces and `sort_keys=True` reads back, so the legend order here is the demo's legend order.
PBMC_TYPES = ("B cells", "CD14+ Monocytes", "CD4 T cells", "CD8 T cells",
              "Dendritic cells", "FCGR3A+ Monocytes", "Megakaryocytes", "NK cells")


def _run(tmp_path, monkeypatch, labels, label_kind):
    """One latent step through the real `run_manylatents`, with the engine faked.

    The engine is stubbed and everything downstream of it is REAL — `_categories_of`,
    `_save_scatter`, `_categorical`, `figspec.save`. The colour path is what is under test, not
    PHATE; a real embedding would only make the test slow and the assertions no stronger.
    """
    api = types.ModuleType("manylatents.api")
    api.run = lambda **kw: {"embeddings": np.asarray(kw["input_data"])[:, :2], "scores": {}}
    monkeypatch.setitem(sys.modules, "manylatents", types.ModuleType("manylatents"))
    monkeypatch.setitem(sys.modules, "manylatents.api", api)

    out = pipeline.run_manylatents(
        {"name": "r", "steps": [{"name": "phate", "group": "latent", "params": {}}]},
        array=np.asarray(np.random.default_rng(0).normal(size=(len(labels), 4)), dtype=np.float32),
        labels=np.asarray(labels, dtype=object), label_kind=label_kind, out_dir=tmp_path)
    return out["plots"][0]


# ── the names reach the record ───────────────────────────────────────────────
def test_the_eight_cell_type_names_reach_a_file_under_the_output_tree(tmp_path, monkeypatch):
    """The adversarial finding was "the strings appear in zero files under `outputs/`". This is
    that grep, run against the spec the run actually persists and the SVG a person submits."""
    labels = [PBMC_TYPES[i % 8] for i in range(80)]
    png = _run(tmp_path, monkeypatch, labels, "group")

    meta = Path(str(figspec.stem_for(png)) + figspec.META).read_text()
    for name in PBMC_TYPES:
        assert name in meta, f"{name!r} never reached the spec on disk"

    spec = figspec.load(png)
    assert spec["is_categorical"] is True
    assert set(spec["category_colors"]) == set(PBMC_TYPES)
    assert len(set(spec["category_colors"].values())) == 8, "eight types, eight distinct colours"
    # ...and the colorbar is GONE, not merely joined by a legend. `labeled` is the flag
    # `figspec.figure` branches on; leaving it True would put a 0.0–1.0 scale bar back on a
    # string array, which is the whole defect.
    assert spec["labeled"] is False

    svg = Path(figspec.export(png, tmp_path, "act2", formats=("svg",))[0]).read_text()
    for name in PBMC_TYPES:
        assert name in svg, f"{name!r} did not survive into the exported figure"


def test_the_legend_is_drawn_and_the_colorbar_is_not(tmp_path, monkeypatch):
    """One `ax.scatter`, one key, and the key is the legend. A second axes would mean a colorbar
    came along too — `fig.colorbar` is what adds one, and it must not fire on an identity."""
    png = _run(tmp_path, monkeypatch, [PBMC_TYPES[i % 8] for i in range(40)], "group")
    fig = figspec.figure(figspec.load(png))

    ax = fig.axes[0]
    assert len(fig.axes) == 1, "a second axes here is a colorbar on a categorical figure"
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == list(PBMC_TYPES), "alphabetical, stable"


def test_a_condition_kind_is_categorical_too_because_the_allowlist_says_so(tmp_path, monkeypatch):
    """`vocab.CATEGORICAL_KINDS` is an ALLOWLIST of two. A `condition` (case/control) is an
    unordered identity exactly as a cell type is, and it must not be a colorbar either."""
    png = _run(tmp_path, monkeypatch, ["treated", "healthy"] * 10, "condition")
    spec = figspec.load(png)

    assert spec["is_categorical"] is True
    assert set(spec["category_colors"]) == {"treated", "healthy"}


# ── and nothing else changes ─────────────────────────────────────────────────
def test_a_time_kind_keeps_its_colorbar_and_its_real_spacing(tmp_path, monkeypatch):
    """THE GUARD AGAINST OVER-REACH. `time` is deliberately absent from `CATEGORICAL_KINDS`:
    `numeric_time` preserves day0/day3/day9 at 0, ⅓, 1 — real spacing, not even ranks — and a
    scale bar is what reads that. A guard that keyed on "the labels look like strings" instead
    of on the KIND would turn this figure categorical and lose the magnitude."""
    png = _run(tmp_path, monkeypatch, ["day0", "day3", "day9"] * 4, "time")
    spec = figspec.load(png)

    assert "is_categorical" not in spec and "category_colors" not in spec
    assert spec["cmap"] == "viridis"
    assert spec["labeled"] is True
    assert np.allclose(spec["color_values"][:3], [0.0, 1 / 3, 1.0]), spec["color_values"][:3]


def test_no_labels_at_all_still_shades_by_the_first_axis(tmp_path, monkeypatch):
    """`label_kind=None` is the unlabelled path — decoration, no key. Pinned because
    `_categories_of` returns None there and must not invent a category out of nothing."""
    png = _run(tmp_path, monkeypatch, ["a", "b"] * 10, None)
    spec = figspec.load(png)

    assert "is_categorical" not in spec
    assert spec["cmap"] == "viridis"


# ── the two silent-loss traps ────────────────────────────────────────────────
def test_the_label_array_round_trips_as_unicode_not_as_object(tmp_path, monkeypatch):
    """THE FAILURE THIS CATCHES RAISES NOTHING. `loading.labels_of` hands back dtype OBJECT on
    both bundled fixtures; `np.savez_compressed` pickles that, `figspec.load` opens with
    `allow_pickle=False` and refuses it, so `load` returns None and `export` returns `[]` while
    the PNG stays on disk looking correct. Assert the DTYPE, not merely that it loaded."""
    png = _run(tmp_path, monkeypatch, [PBMC_TYPES[i % 8] for i in range(24)], "group")
    spec = figspec.load(png)

    assert spec is not None, "the spec vanished — the object-dtype pickle trap"
    assert spec["color_values"].dtype.kind == "U", spec["color_values"].dtype


def test_the_spec_meta_is_plain_json_types(tmp_path, monkeypatch):
    """`figspec.save` catches `(OSError, ValueError, IndexError)` and NOT `TypeError`, and
    `_save_scatter`'s blanket `except Exception` swallows whatever escapes — so one numpy scalar
    in `meta` means the spec silently never exists. Hex strings and `str` keys only."""
    import json

    png = _run(tmp_path, monkeypatch, [PBMC_TYPES[i % 8] for i in range(24)], "group")
    meta = json.loads(Path(str(figspec.stem_for(png)) + figspec.META).read_text())

    for name, hexcolor in meta["category_colors"].items():
        assert type(name) is str and type(hexcolor) is str
        assert hexcolor.startswith("#") and len(hexcolor) == 7, hexcolor


# ── the cap ──────────────────────────────────────────────────────────────────
def test_past_the_cap_the_legend_backs_off_but_the_names_are_still_kept():
    """`vocab.GROUP_KEYS` admits `leiden`/`louvain`/`cluster`, routinely 20–50 levels. At 35
    categories the legend measures 80×474 px on a 480 px canvas — TALLER THAN THE AXES. Drawing
    it would make the legend the figure, so the drawing backs off; the spec keeps every name, so
    a later reader still has the key. Both halves are the assertion."""
    n = 35
    labels = np.array([f"cluster_{i:02d}" for i in range(n)])
    values, colors, palette = _io._categorical(labels)
    assert palette == "tab20" and len(colors) == n > figspec.LEGEND_MAX

    fig = figspec.figure({"coords": np.random.default_rng(0).normal(size=(n, 2)),
                          "color_values": values, "is_categorical": True,
                          "category_colors": colors, "title": "leiden"})
    ax = fig.axes[0]

    assert ax.get_legend() is None, "35 entries is a legend taller than the axes it explains"
    assert len(fig.axes) == 1, "and it must not fall back to a colorbar either"
    # The COLOURS are still drawn — backing off the key must not back off the picture.
    drawn = {tuple(c) for c in ax.collections[0].get_facecolors()}
    assert len(drawn) == 20, drawn
    # ...and that 20, against 35 names, IS THE SECOND HALF OF WHY THE CAP EXISTS. `_categorical`
    # wraps with `i % base.N`, so past `tab20`'s twenty swatches two clusters share a colour. A
    # legend here would not merely be too tall, it would be WRONG — the same swatch against two
    # different names. The cap is a correctness boundary before it is a legibility one.
    assert len(colors) > len(set(colors.values())), "the wrap is real, and it is why 20"


def test_the_palette_switches_at_ten_because_tab10_runs_out():
    """`tab10` has ten swatches and `tab20` has twenty; past twenty they repeat, which is why
    `LEGEND_MAX` is a correctness boundary and not only a legibility one."""
    assert _io._categorical(np.array([f"c{i}" for i in range(10)]))[2] == "tab10"
    assert _io._categorical(np.array([f"c{i}" for i in range(11)]))[2] == "tab20"
    assert figspec.LEGEND_MAX == 20


def test_the_names_and_the_values_are_computed_from_one_array_at_one_moment():
    """HAZARD 7, stated as a property of `_categorical` itself: every value it returns is a key
    of the mapping it returns. That is what makes a narrowing safe — names built at run open and
    values rebuilt on a filter could disagree, and this cannot."""
    labels = np.array(["B cells"] * 3 + ["NK cells"] * 2, dtype=object)
    values, colors, _ = _io._categorical(labels)

    assert set(values.tolist()) <= set(colors)
    assert list(values) == ["B cells"] * 3 + ["NK cells"] * 2, "row order is untouched"
