"""The figure you can remake — `manyruns/figspec.py` and the quickdrop that uses it.

A run left a 600×480 PNG at `dpi=120`: right for a terminal, wrong for a paper, and there was no
third option — the artist tree was closed at the end of `_save_scatter`, so the only route to a
publication figure was to re-run the analysis and hope it landed the same.

Two properties are pinned here and they are the whole design:

  * **one artist tree, every medium** — `figspec.figure` is the only place a scatter is built, so
    the PNG on screen and the SVG in the submission are the same function over the same numbers;
  * **vector at ANY point count** — an SVG of a scatter is wanted because every point is its own
    path. An export that quietly swapped in a bitmap above some threshold would return a file
    that opens, looks right, and is not what was asked for.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from manyruns import figspec  # noqa: E402
from manyruns.pipeline import io as _io  # noqa: E402


def _run_a_figure(tmp_path, n=300, color=None):
    """What a real step does: `_save_scatter` draws the PNG and writes the spec beside it."""
    plots: list = []
    emb = np.asarray(np.random.default_rng(0).normal(size=(n, 3)))
    _io._save_scatter(emb, color, tmp_path, "phate.png", plots, title="phate")
    return plots[0], emb


# ── the spec ─────────────────────────────────────────────────────────────────
def test_a_run_leaves_the_numbers_the_picture_was_made_of(tmp_path):
    """Not just the picture. The PNG is 120 dpi because that is what a terminal wants; without
    the spec, the only way to a print-quality version was to re-run the analysis."""
    png, emb = _run_a_figure(tmp_path)
    spec = figspec.load(png)

    assert spec is not None
    assert spec["coords"].shape == (300, 2)
    assert np.allclose(spec["coords"], emb[:, :2])
    assert spec["title"] == "phate" and spec["cmap"] == "viridis"


def test_the_spec_is_addressed_by_the_png_not_by_a_convention_you_have_to_know(tmp_path):
    """The step record carries the PNG path, so that is what `load` takes. A caller should not
    have to know the naming to get from the figure it has to the spec beside it."""
    png, _ = _run_a_figure(tmp_path)

    assert figspec.load(png) is not None
    assert figspec.load(figspec.stem_for(png)) is not None
    assert figspec.load(tmp_path / "plots" / "never-drawn.png") is None


def test_the_colorbar_decision_travels_in_the_spec_rather_than_being_re_derived(tmp_path):
    """`labeled` is the difference between "these are timepoints, here is the scale" and "this
    shading is the embedding's own first axis, do not read a quantity off it". By export time
    `color` has been folded into `color_values` and the distinction would be gone."""
    unlabelled, _ = _run_a_figure(tmp_path / "a")
    labelled, _ = _run_a_figure(tmp_path / "b", color=np.arange(300) % 5)

    assert figspec.load(unlabelled)["labeled"] is False
    assert figspec.load(labelled)["labeled"] is True
    assert "color_values" in figspec.load(labelled)


def test_a_failed_spec_write_never_costs_the_figure(tmp_path, monkeypatch):
    """The PNG is written first and the spec second, so a full disk loses the ability to
    re-render and not the figure a reader already has."""
    monkeypatch.setattr(figspec, "save", lambda *a, **k: None)
    png, _ = _run_a_figure(tmp_path)

    assert Path(png).exists()
    assert figspec.load(png) is None


# ── one artist tree, every medium ────────────────────────────────────────────
def test_the_export_is_the_same_builder_the_run_drew_with(tmp_path):
    """"What you saw in the console is what you publish" is a property of the code — one
    function over one set of numbers — not of two plotting calls being kept in step. If
    `_save_scatter` ever grows its own `ax.scatter` again, this is what should fail."""
    import inspect

    source = inspect.getsource(_io._save_scatter)

    assert "figspec.figure" in source or "_figspec.figure" in source
    assert "ax.scatter" not in source, "the run must not build its own scatter"


def test_a_quickdrop_writes_all_three_and_they_come_from_one_figure(tmp_path):
    """SVG, PDF and PNG: the editable figure, the one a journal takes, the one for the slide an
    hour later. Someone who has just found the embedding they wanted should not have to know
    which they need yet."""
    png, _ = _run_a_figure(tmp_path)

    written = figspec.export(png, tmp_path / "figures", "phate-run1")

    assert [Path(p).suffix for p in written] == [".svg", ".pdf", ".png"]
    assert all(Path(p).exists() and Path(p).stat().st_size > 0 for p in written)
    assert all(Path(p).stem == "phate-run1" for p in written)


def test_the_export_is_300_dpi_and_the_terminal_png_is_not(tmp_path):
    """The whole reason this module exists. Enlarging the 120-dpi PNG would hand someone a blurry
    figure that looks like it was meant to be that size."""
    from PIL import Image

    png, _ = _run_a_figure(tmp_path)
    exported = [p for p in figspec.export(png, tmp_path / "figures", "f") if p.endswith(".png")]

    assert Image.open(png).size == (600, 480)                 # 5x4in at dpi=120
    assert Image.open(exported[0]).size[0] > 1500             # 6.4in at dpi=300, tight bbox


def test_there_is_nothing_to_export_before_specs_existed(tmp_path):
    """A figure from an older run has no spec. `[]`, so the caller can say so — reporting
    "saved" when nothing was written is worse than not offering the key."""
    stray = tmp_path / "plots" / "old.png"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"\x89PNG\r\n\x1a\n")

    assert figspec.export(stray, tmp_path / "figures", "old") == []


# ── vector at any point count ────────────────────────────────────────────────
@pytest.mark.parametrize("n", [300, 12_000])
def test_every_point_is_its_own_path_however_many_there_are(tmp_path, n):
    """THE requirement: the capacity to get a real SVG regardless of the point count.

    12,000 is deliberately past `RASTER_ABOVE`, which is the count at which a fully-vector file
    starts to cost real size (measured at 50,000: 7.7 MB vector against 943 KB rasterised). That
    is a size to be TOLD about, not a reason to silently return something else — so `rasterized`
    defaults to False at every count and the SVG carries no `<image>` tag at all.
    """
    png, _ = _run_a_figure(tmp_path, n=n)
    svg = next(p for p in figspec.export(png, tmp_path / "figures", "f") if p.endswith(".svg"))
    text = Path(svg).read_text()

    assert "<image " not in text, "the points were rasterised behind the caller's back"
    assert text.count("<use ") + text.count("<path ") >= n, "one element per point, at least"


def test_rasterising_stays_available_as_an_explicit_request(tmp_path):
    """A journal upload limit is a real constraint. It is a choice, not a default — and when it
    IS made, only the points become a bitmap: axes, ticks and text stay vector, which is what
    keeps the file openable in a vector editor."""
    png, _ = _run_a_figure(tmp_path, n=2000)

    svg = next(p for p in figspec.export(png, tmp_path / "figures", "f", formats=("svg",),
                                         rasterized=True) if p.endswith(".svg"))
    text = Path(svg).read_text()

    assert "<image " in text                       # the points, as one bitmap
    assert "phate" in text                         # and the title still as text


def test_large_reports_and_never_acts(tmp_path):
    """Advisory only. Nothing in the module reads it — it exists so a caller can put the number
    in front of a person rather than deciding for them."""
    assert figspec.large({"coords": np.zeros((figspec.RASTER_ABOVE + 1, 2))})
    assert not figspec.large({"coords": np.zeros((10, 2))})

    png, _ = _run_a_figure(tmp_path, n=200)
    written = figspec.export(png, tmp_path / "figures", "f", formats=("svg",))
    assert "<image " not in Path(written[0]).read_text()      # unchanged by the advice


# WS6: explicit display channels and durable alignment.
def test_display_scatter_api_records_each_channel_and_unicode_ids(tmp_path):
    emb = np.arange(12).reshape(4, 3)
    ids = ["α", "β", "γ", "δ"]
    channels = [{"key": "group/name", "kind": "categorical", "values": ["a", "b"] * 2},
                {"key": "score", "kind": "continuous", "values": [1., np.nan, 3., np.inf]}]
    plots = []
    paths = _io.save_display_scatter(emb, channels, tmp_path, "phate.png", plots, "PHATE",
                                     obs_names=ids)
    assert paths == plots and len(paths) == 2
    assert paths.skipped_channels == []
    assert Path(paths[0]).name == "phate.png"
    for path, channel in zip(paths, channels):
        spec = figspec.load(path)
        assert spec["color_by"] == channel["key"] and spec["color_kind"] == channel["kind"]
        np.testing.assert_array_equal(spec["obs_names"], ids)
        assert spec["obs_names"].dtype.kind == "U"
        import json
        meta = json.loads(figspec._paths(path)[1].read_text())
        assert "obs_names" not in meta and "coords" not in meta
    fig = figspec.figure(figspec.load(paths[1]))
    assert fig.axes[0].collections[0].get_offsets().mask[:, 0].tolist() == [False, True, False, True]
    import matplotlib.pyplot as plt
    plt.close(fig)


@pytest.mark.parametrize("existing_default", [False, True])
@pytest.mark.parametrize("kind", ["continuous", "categorical"])
def test_display_scatter_skips_all_missing_selected_rows(
        tmp_path, monkeypatch, caplog, existing_default, kind):
    import pandas as pd
    from types import SimpleNamespace
    from manyruns.pipeline import loading

    values = ([np.nan, np.inf, -np.inf, *range(27)] if kind == "continuous"
              else [None] * 3 + ["a", "b", "c"] * 9)
    obs = pd.DataFrame({"score": values}, index=[f"cell-{i}" for i in range(30)])
    channel = loading.color_channel_of(SimpleNamespace(obs=obs), "score")
    assert channel["available"] and channel["kind"] == kind
    rows = [2, 0, 1]
    selected = {**channel, "values": channel["values"][rows]}
    emb = np.arange(6).reshape(3, 2)
    ids = obs.index[rows]
    plots = []
    if existing_default:
        _io.save_display_scatter(emb, [], tmp_path, "p.png", plots, "P", obs_names=ids)
    assert tmp_path.is_dir()
    original = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    original_plots = plots.copy()
    monkeypatch.setattr(figspec, "figure", lambda *a, **k: pytest.fail("must skip rendering"))

    paths = _io.save_display_scatter(emb, [selected], tmp_path, "p.png", plots, "P",
                                     obs_names=ids)

    assert paths == [] and plots == original_plots
    assert paths.skipped_channels == [("score", "all plotted values are missing or nonfinite")]
    assert "skipping colour channel 'score': all plotted values are missing or nonfinite" in caplog.text
    assert tmp_path.is_dir()
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == original


@pytest.mark.parametrize("invalid_first", [False, True])
def test_display_scatter_continues_after_skipping_an_all_nonfinite_channel(
        tmp_path, caplog, invalid_first):
    channels = [{"key": "score", "kind": "continuous", "values": [np.nan, np.inf, -np.inf]},
                {"key": "group", "kind": "categorical", "values": ["a", "b", "a"]}]
    if not invalid_first:
        channels.reverse()
    plots = []

    paths = _io.save_display_scatter(np.arange(6).reshape(3, 2), channels, tmp_path,
                                     "p.png", plots, "P")

    assert paths == plots and len(paths) == 1
    assert paths.skipped_channels == [("score", "all plotted values are missing or nonfinite")]
    assert figspec.load(paths[0])["color_by"] == "group"
    assert "skipping colour channel 'score': all plotted values are missing or nonfinite" in caplog.text
    root = tmp_path / "plots"
    assert root.is_dir()
    assert set(root.iterdir()) == {Path(paths[0]), *figspec._paths(paths[0])}


def test_display_scatter_returns_every_skipped_channel_reason(tmp_path):
    channels = [
        {"key": None, "kind": "continuous", "values": [np.nan, np.inf]},
        {"key": "group/name", "kind": "categorical", "values": [None, None]},
        {"key": "score", "kind": "continuous", "values": [-np.inf, np.nan]},
    ]
    plots = []
    paths = _io.save_display_scatter(np.zeros((2, 2)), channels, tmp_path,
                                     "p.png", plots, "P")

    # Caller-visible diagnostics remain available even when no PNG survives, without
    # changing list truthiness, iteration, equality or the paths-only plots argument.
    assert isinstance(paths, list) and not paths and list(paths) == plots == []
    reason = "all plotted values are missing or nonfinite"
    assert paths.skipped_channels == [(None, reason), ("group/name", reason), ("score", reason)]
    next_paths = _io.save_display_scatter(np.zeros((2, 2)), [], tmp_path,
                                          "bare.png", [], "Bare")
    assert next_paths and next_paths.skipped_channels == []
    assert len(paths.skipped_channels) == 3


@pytest.mark.parametrize("names", [["a"], ["a", "a"]])
def test_explicit_ids_refuse_length_and_duplicates(tmp_path, names):
    with pytest.raises(ValueError, match="obs_names"):
        figspec.save(tmp_path / "x.png", np.zeros((2, 2)), None, {}, obs_names=names)


def test_explicit_channel_mismatch_is_visible_but_legacy_falls_back(tmp_path):
    emb = np.zeros((3, 2))
    with pytest.raises(ValueError, match="colour.*3"):
        _io.save_display_scatter(emb, [{"key": "day", "values": [0, 1], "kind": "continuous"}],
                                 tmp_path, "x.png", [], "x")
    plots = []
    _io._save_scatter(emb, [0, 1], tmp_path, "legacy.png", plots, "x")
    assert figspec.load(plots[0]).get("color_values") is None  # preservation


def test_join_shuffles_survivors_allows_extras_and_discloses_legacy():
    import pandas as pd
    obs = pd.DataFrame({"score": [10, 20, 30, 40]}, index=["δ", "β", "α", "extra"])
    spec = {"coords": np.zeros((2, 2)), "obs_names": np.array(["α", "β"])}
    values, positional = figspec.join(spec, obs, "score")
    assert list(values) == [30, 20] and positional is False
    legacy = {"coords": np.zeros((4, 2))}
    values, positional = figspec.join(legacy, obs, "score")
    assert list(values) == [10, 20, 30, 40] and positional is True
    with pytest.raises(ValueError, match="positional.*length"):
        figspec.join({"coords": np.zeros((2, 2))}, obs, "score")
    with pytest.raises(ValueError, match="missing.*1"):
        figspec.join(spec, obs.drop("α"), "score")
    obs.index = ["α", "α", "β", "extra"]
    with pytest.raises(ValueError, match="duplicate.*2"):
        figspec.join(spec, obs, "score")


def test_positional_join_accepts_duplicate_source_names_but_requires_equal_lengths():
    import pandas as pd

    obs = pd.DataFrame({"condition": ["treated", "control", "treated"]},
                       index=["barcode", "barcode", "other"])
    values, positional = figspec.join({"coords": np.zeros((3, 2))}, obs, "condition")
    pd.testing.assert_series_equal(values, obs["condition"])
    assert positional is True
    with pytest.raises(ValueError, match="positional alignment length mismatch"):
        figspec.join({"coords": np.zeros((2, 2))}, obs, "condition")


def test_recolor_is_pure_clears_encodings_and_does_not_compound_titles():
    import copy
    spec = {"coords": np.arange(6).reshape(3, 2), "obs_names": np.array(["a", "b", "c"]),
            "trajectories": np.zeros((2, 1, 2)), "title": "PHATE", "cmap": "plasma"}
    original = copy.deepcopy(spec)
    first = figspec.recolor(spec, ["a", "b", "a"], "group", "categorical")
    second = figspec.recolor(first, [1, 2, 3], "score", "continuous")
    third = figspec.recolor(second, [True, False, True], "flag", "categorical")
    assert "category_colors" not in second and not second.get("is_categorical")
    assert second["labeled"] and second["cmap"] == "viridis"
    assert third["is_categorical"] and not third["labeled"]
    assert third["title"] == "PHATE — colour by flag"
    for field in ("coords", "obs_names", "trajectories"):
        np.testing.assert_equal(third[field], original[field])
        np.testing.assert_equal(spec[field], original[field])
    assert spec["title"] == "PHATE" and "color_values" not in spec


def test_safe_channel_names_survive_collisions_and_duplicates(tmp_path):
    keys = ["base", "../a/b", "..\\a\\b", "a b", "a/b", "α" * 100, "a/b"]
    channels = [{"key": k, "kind": "continuous", "values": [1, 2]} for k in keys]
    paths = _io.save_display_scatter(np.zeros((2, 2)), channels, tmp_path, "p.png", [], "p")
    assert len(set(paths)) == len(keys)
    assert all(Path(p).parent == tmp_path / "plots" and len(Path(p).name) < 110 for p in paths)
    assert [figspec.load(p)["color_by"] for p in paths] == keys


def test_repeated_views_are_unique_and_failed_writes_preserve_originals(tmp_path, monkeypatch):
    png, _ = _run_a_figure(tmp_path, n=3)
    files = [Path(png), *figspec._paths(png)]
    before = {p: p.read_bytes() for p in files}
    changed = figspec.recolor(figspec.load(png), [1, 2, 3], "../../score", "continuous")
    first = figspec.write_view(png, changed)
    second = figspec.write_view(png, changed)
    assert first != second and first != png
    assert all(Path(p).is_file() for view in (first, second) for p in [view, *figspec._paths(view)])
    assert (tmp_path / "plots").is_dir()
    persisted = set((tmp_path / "plots").iterdir())
    monkeypatch.setattr(figspec, "save", lambda *a, **k: None)
    with pytest.raises(OSError, match="spec"):
        figspec.write_view(png, changed)
    assert (tmp_path / "plots").is_dir()
    assert set((tmp_path / "plots").iterdir()) == persisted
    assert {p: p.read_bytes() for p in files} == before


def test_explicit_unknown_provenance_still_enforces_alignment(tmp_path):
    with pytest.raises(ValueError, match="colour length"):
        _io._save_scatter(np.zeros((3, 2)), [1], tmp_path, "x.png", [], "X", color_by=None)


def test_empty_display_channels_record_unknown_provenance_and_only_complete_paths(tmp_path, monkeypatch):
    plots = []
    paths = _io.save_display_scatter(np.zeros((2, 2)), [], tmp_path, "bare.png", plots, "Bare")
    spec = figspec.load(paths[0])
    assert spec["color_by"] is None and spec["color_kind"] is None
    assert not spec["labeled"] and plots == paths
    monkeypatch.setattr(figspec, "save", lambda *a, **k: None)
    unsuccessful = []
    with pytest.raises(OSError, match="spec"):
        _io.save_display_scatter(np.zeros((2, 2)), [], tmp_path, "failed.png", unsuccessful, "Bare")
    assert unsuccessful == []


def test_view_failure_after_npz_write_removes_only_new_partials(tmp_path, monkeypatch):
    png, _ = _run_a_figure(tmp_path, n=3)
    changed = figspec.recolor(figspec.load(png), [1, 2, np.nan], "score", "continuous")
    root = Path(png).parent
    assert root.is_dir()
    original = {p: p.read_bytes() for p in root.iterdir()}
    real_write = Path.write_text

    def fail_meta(path, text, *args, **kwargs):
        if "-view-" in path.name and path.name.endswith(figspec.META):
            raise OSError("full disk")
        return real_write(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_meta)
    with pytest.raises(OSError, match="spec"):
        figspec.write_view(png, changed)
    assert root.is_dir()
    assert {p: p.read_bytes() for p in root.iterdir()} == original
    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []


def test_recoloured_views_export_all_formats_and_keep_original_coordinates(tmp_path):
    # Preservation: every output format still uses the one existing builder.
    png, emb = _run_a_figure(tmp_path, n=3)
    spec = figspec.recolor(figspec.load(png), ["a", None, "b"], "group", "categorical")
    view = figspec.write_view(png, spec)
    outputs = figspec.export(view, tmp_path / "exported", "selected")
    assert [Path(p).suffix for p in outputs] == [".svg", ".pdf", ".png"]
    assert all(Path(p).stat().st_size for p in outputs)
    np.testing.assert_array_equal(figspec.load(view)["coords"], emb[:, :2])
    assert figspec.load(png).get("color_by") is None


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["render", "spec"])
def test_interrupted_view_write_cleans_partials_and_preserves_existing_views(
        tmp_path, monkeypatch, interrupt, stage):
    from matplotlib.figure import Figure

    png, _ = _run_a_figure(tmp_path, n=3)
    changed = figspec.recolor(figspec.load(png), [1, 2, 3], "score", "continuous")
    figspec.write_view(png, changed)
    root = Path(png).parent
    assert root.is_dir()
    original = {p: p.read_bytes() for p in root.iterdir()}
    real_save = figspec.save

    def interrupted(*args, **kwargs):
        if stage == "spec":
            real_save(*args, **kwargs)
        raise interrupt("interrupted view")

    if stage == "render":
        monkeypatch.setattr(Figure, "savefig", interrupted)
    else:
        monkeypatch.setattr(figspec, "save", interrupted)
    with pytest.raises(interrupt, match="interrupted view"):
        figspec.write_view(png, changed)
    assert root.is_dir()
    assert {p: p.read_bytes() for p in root.iterdir()} == original
    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []


@pytest.mark.parametrize("stage", ["spec", "png", "skipped_png"])
def test_view_publication_failure_cleans_only_new_files(tmp_path, monkeypatch, stage):
    png, _ = _run_a_figure(tmp_path, n=3)
    root = Path(png).parent
    assert root.is_dir()
    original = {p: p.read_bytes() for p in root.iterdir()}
    changed = figspec.recolor(figspec.load(png), [1, 2, 3], "score", "continuous")
    real_replace = Path.replace

    def fail_publication(path, target):
        if target.suffix == ".png":
            assert all(part.is_file() and part.stat().st_size for part in figspec._paths(target))
            if stage == "skipped_png":
                return target
        if ((stage == "png" and target.suffix == ".png")
                or (stage == "spec" and target.suffix != ".png")):
            raise OSError("interrupted publication")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication)
    with pytest.raises(OSError, match="publication|complete figure spec"):
        figspec.write_view(png, changed)
    assert root.is_dir()
    assert {p: p.read_bytes() for p in root.iterdir()} == original


def test_process_exit_during_daemon_render_leaves_no_view_png(tmp_path):
    import os
    import subprocess
    import sys
    import textwrap

    png, _ = _run_a_figure(tmp_path, n=3)
    root = Path(png).parent
    assert root.is_dir()
    original = {p: p.read_bytes() for p in root.iterdir()}
    script = textwrap.dedent("""
        import sys
        import threading
        from matplotlib.figure import Figure
        from manyruns import figspec

        entered = threading.Event()
        def blocked_render(*args, **kwargs):
            entered.set()
            threading.Event().wait(30)
        Figure.savefig = blocked_render
        png = sys.argv[1]
        spec = figspec.recolor(figspec.load(png), [1, 2, 3], "score", "continuous")
        threading.Thread(target=figspec.write_view, args=(png, spec), daemon=True).start()
        if not entered.wait(15):
            raise RuntimeError("render did not start")
        print("exiting during render", flush=True)
    """)
    env = dict(os.environ, PYTHONPATH=str(Path(figspec.__file__).resolve().parent.parent),
               MPLCONFIGDIR=str(tmp_path / "matplotlib-cache"))
    result = subprocess.run([sys.executable, "-c", script, png], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr
    assert "exiting during render" in result.stdout
    assert root.is_dir()
    assert {p: p.read_bytes() for p in root.iterdir()} == original


def test_spec_rejects_duplicate_named_survivors_even_with_unique_source():
    import pandas as pd
    with pytest.raises(ValueError, match="obs_names must be unique"):
        figspec.join({"coords": np.zeros((2, 2)), "obs_names": ["a", "a"]},
                     pd.DataFrame({"score": [1, 2]}, index=["a", "b"]), "score")
