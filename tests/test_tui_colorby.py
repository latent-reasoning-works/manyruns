"""Reusable colour picker and standalone FigureScreen; no RunScreen integration here."""
import asyncio
import functools
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from textual.app import App
from textual.widgets import OptionList

from manyruns import figspec
from manyruns.narrate import Observation
from manyruns.pipeline import io, loading
from manyruns.tui import colorby
from manyruns.tui.figure import FigureScreen
from manyruns.tui.state import DataEntry


def drives(body):
    @functools.wraps(body)
    def run(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))
    return run


class Harness(App):
    def __init__(self, screen):
        super().__init__()
        self.initial = screen
        self.result = "pending"

    def on_mount(self):
        self.push_screen(self.initial, lambda result: setattr(self, "result", result))


def entry_for(path, name="sample"):
    return DataEntry(name, "file", Observation("single"), path=path)


def source(tmp_path, *, empty=False):
    path = tmp_path / "data.h5ad"
    obs = pd.DataFrame(index=[f"cell-{i}" for i in range(6)])
    if not empty:
        obs["group"] = ["a", "b"] * 3
        obs["day"] = [0, 1, 2] * 2
        obs["empty"] = [np.nan] * 6
        obs["barcode"] = [f"id-{i}" for i in range(6)]
    AnnData(np.zeros((6, 3)), obs=obs).write_h5ad(path)
    return entry_for(path)


def figure(tmp_path, *, named=True):
    return io.save_display_scatter(np.arange(12).reshape(6, 2), [], tmp_path, "p.png", [], "P",
                                   obs_names=[f"cell-{i}" for i in range(6)] if named else None)[0]


async def until(pilot, predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "UI operation did not finish"
        await pilot.pause(0.02)


def notices(app):
    return "\n".join(str(n.message) for n in app._notifications)


def test_metadata_loader_reads_generated_timepoint_and_refreshes_stale_cache(tmp_path):
    entry = source(tmp_path)
    assert entry.obs.columns is None
    metadata = colorby.load_metadata(entry)
    assert [c["key"] for c in metadata.columns] == ["group", "day", "empty", "barcode"]
    generated = entry_for(Path(loading.TIME_COURSE_REF), "synthetic_timecourse")
    data = colorby.load_metadata(generated)
    assert data.columns[0]["key"] == "timepoint" and len(data.obs) == 300
    with pytest.raises(ValueError, match="no metadata source"):
        colorby.load_metadata(entry_for(None, "generator"))
    entry.path.unlink()
    with pytest.raises(ValueError, match="could not read.*data.h5ad"):
        colorby.load_metadata(entry)


@pytest.mark.parametrize("layout", ["flat", "nested"])
@drives
async def test_folder_picker_matches_analysis_source_and_aligns_surviving_rows(tmp_path, layout):
    folder = tmp_path / "source"
    folder.mkdir()
    obs = pd.DataFrame({"condition": ["treated", "control", "control", "treated"]},
                       index=["shared", "first", "shared-1", "last"])
    if layout == "flat":
        AnnData(np.zeros((4, 2)), obs=obs).write_h5ad(folder / "annotated.h5ad")
        # The analysis loader chooses the first supported file, not every annotation.
        decoy = obs.rename(columns={"condition": "ignored"})
        AnnData(np.zeros((4, 2)), obs=decoy).write_h5ad(folder / "z-ignored.h5ad")
    else:
        # Samples take precedence over files at the root and concatenate in sorted order.
        for name, rows in (("b", [2, 3]), ("a", [0, 1])):
            sample = folder / name
            sample.mkdir()
            sample_obs = obs.iloc[rows].copy()
            sample_obs.index = ["shared", sample_obs.index[1]]
            AnnData(np.zeros((2, 2)), obs=sample_obs).write_h5ad(sample / "annotated.h5ad")
        AnnData(np.zeros((1, 2))).write_h5ad(folder / "ignored.h5ad")
    loaded = loading.load_array(folder)
    pd.testing.assert_frame_equal(loaded.obs.astype(object), obs.astype(object))
    channel = loading.color_channel_of(loaded, "condition")
    np.testing.assert_array_equal(channel["values"], obs["condition"])
    ids = ["shared-1", "shared", "last"]
    coords = np.arange(6).reshape(3, 2)
    png = io.save_display_scatter(coords, [], tmp_path, "p.png", [], "P", obs_names=ids)[0]
    views = []
    screen = FigureScreen([("p", png)], entry=entry_for(folder),
                          on_view=lambda *view: views.append(view))
    app = Harness(screen)
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("c")
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen)
                    or "no metadata columns" in notices(app))
        assert isinstance(app.screen, colorby.ColorByScreen), notices(app)
        pd.testing.assert_frame_equal(app.screen.metadata.obs, loaded.obs)
        assert app.screen.metadata.columns[0]["key"] == "condition"
        await pilot.press("enter")
        await until(pilot, lambda: bool(views))
        assert "positional" not in str(screen.query_one("#figure-note").render())
    spec = figspec.load(views[0][1])
    assert spec["color_by"] == "condition" and spec["positional"] is False
    np.testing.assert_array_equal(spec["obs_names"], ids)
    np.testing.assert_array_equal(spec["coords"], coords)
    np.testing.assert_array_equal(spec["color_values"], ["control", "treated", "treated"])


def test_folder_with_unreadable_source_reports_the_read_failure(tmp_path):
    (tmp_path / "annotated.h5ad").write_bytes(b"not an h5ad")
    with pytest.raises(ValueError, match="could not read metadata.*annotated.h5ad"):
        colorby.load_metadata(entry_for(tmp_path))


def test_folder_with_a_bare_matrix_has_no_metadata_columns(tmp_path):
    # Preservation: supported sources without obs still get the empty-column guidance.
    np.save(tmp_path / "matrix.npy", np.zeros((3, 2)), allow_pickle=False)
    metadata = colorby.load_metadata(entry_for(tmp_path))
    assert metadata.columns == [] and metadata.obs.empty


@pytest.fixture
def changing_generator(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    script = folder / "annotated.py"
    counter = tmp_path / "calls.txt"
    script.write_text(
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from anndata import AnnData\n\n"
        "def load():\n"
        f"    counter = Path({str(counter)!r})\n"
        "    calls = int(counter.read_text()) if counter.exists() else 0\n"
        "    counter.write_text(str(calls + 1))\n"
        "    groups = ['a', 'b', 'a'] if calls % 2 == 0 else ['b', 'a', 'b']\n"
        "    obs = pd.DataFrame({'cluster': groups, 'time': [0., 1., 2.]},\n"
        "                       index=['cell-0', 'cell-1', 'cell-2'])\n"
        "    return AnnData(np.arange(6).reshape(3, 2), obs=obs)\n"
    )
    loaded = loading.load_array(script)
    labels, kind = loading.labels_of(loaded)
    assert kind == "time"
    np.testing.assert_array_equal(labels, [0., 1., 2.])
    regenerated = loading.load_array(script)
    assert list(regenerated.obs_names) == list(loaded.obs_names)
    assert (regenerated.obs["cluster"] != loaded.obs["cluster"]).all()
    counter.write_text("0")  # Count only display-triggered executions below.
    return script, counter, loaded


@pytest.mark.parametrize("layout", ["file", "folder", "nested"])
@drives
async def test_python_generator_picker_refuses_without_executing(
        tmp_path, changing_generator, layout):
    script, counter, loaded = changing_generator
    path = script if layout == "file" else script.parent
    if layout == "nested":
        # Per-sample dispatch must propagate the refusal to its selected .py file.
        loaded.write_h5ad(script.parent / "z-annotated.h5ad")
        path = tmp_path
    png = io.save_display_scatter(loaded.X[[2, 0]], [], tmp_path, "p.png", [], "P",
                                  obs_names=loaded.obs_names[[2, 0]])[0]
    plot_dir = Path(png).parent
    assert plot_dir.is_dir()
    originals = {p: p.read_bytes() for p in plot_dir.iterdir()}
    views = []
    screen = FigureScreen([("p", png)], entry=entry_for(path),
                          on_view=lambda *view: views.append(view))
    app = Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press("c")
        await until(pilot, lambda: not screen._color_busy
                    or isinstance(app.screen, colorby.ColorByScreen))
        assert counter.read_text() == "0", "display must not execute the generator"
        assert app.screen is screen
        assert "Python generator metadata cannot be read after the run" in notices(app)
        assert not views
    assert {p: p.read_bytes() for p in plot_dir.iterdir()} == originals


def test_python_generator_metadata_helper_refuses_without_executing(changing_generator):
    script, counter, _ = changing_generator
    with pytest.raises(ValueError, match="Python generator metadata cannot be read after the run"):
        colorby.load_metadata(entry_for(script))
    assert counter.read_text() == "0"


@drives
async def test_10x_h5_without_obs_columns_shows_empty_metadata_guidance(tmp_path):
    import h5py
    from scipy.sparse import csc_matrix

    path = tmp_path / "filtered_feature_bc_matrix.h5"
    matrix = csc_matrix(np.arange(18).reshape(3, 6))
    with h5py.File(path, "w") as handle:
        group = handle.create_group("matrix")
        for key in ("data", "indices", "indptr", "shape"):
            group.create_dataset(key, data=getattr(matrix, key))
        group.create_dataset("barcodes", data=np.array([f"cell-{i}" for i in range(6)], dtype="S"))
        features = group.create_group("features")
        for key, values in {
            "id": ["g0", "g1", "g2"], "name": ["g0", "g1", "g2"],
            "feature_type": ["Gene Expression"] * 3, "genome": ["reference"] * 3,
        }.items():
            features.create_dataset(key, data=np.array(values, dtype="S"))
    loaded = loading.load_array(path)
    assert loaded.shape == (6, 3) and list(loaded.obs.columns) == []
    screen = FigureScreen([("p", figure(tmp_path))], entry=entry_for(path, "10x sample"))
    app = Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press("c")
        await until(pilot, lambda: "metadata" in notices(app))
        assert app.screen is screen
        assert ("10x sample has no metadata columns to colour by — "
                "drop an annotated .h5ad to pick one") in notices(app)
        assert "could not read metadata" not in notices(app)


@pytest.mark.parametrize("gesture", ["enter", "click", "escape"])
@drives
async def test_picker_enter_click_and_escape(tmp_path, gesture):
    metadata = colorby.load_metadata(source(tmp_path))
    picker = colorby.ColorByScreen(metadata, current="day")
    app = Harness(picker)
    async with app.run_test(size=(100, 28)) as pilot:
        await pilot.pause()
        assert picker.query_one(OptionList).highlighted == 1
        if gesture == "click":
            await pilot.click(OptionList, offset=(4, 0))
        else:
            await pilot.press(gesture)
        await until(pilot, lambda: app.result != "pending")
    assert app.result == {"enter": "day", "click": "group", "escape": None}[gesture]


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("tier", ["halfcell", "unicode"])
@drives
async def test_figure_recolour_inserts_selects_and_calls_persistence_callback(tmp_path, named,
                                                                          tier, monkeypatch):
    from manyruns.tui import figure as figure_module, images

    monkeypatch.setattr(images, "_Renderable", images._BY_NAME[tier][0])
    assert images.tier() == tier and not images.is_pixel_perfect()
    opened = []
    monkeypatch.setattr(figure_module, "open_externally", lambda path: opened.append(path))
    entry, png = source(tmp_path), figure(tmp_path, named=named)
    views = []
    screen = FigureScreen([("p", png)], entry=entry, on_view=lambda step, path: views.append((step, path)))
    app = Harness(screen)
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("c")
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        await pilot.press("enter")
        await until(pilot, lambda: len(views) == 1)
        assert screen.figures[screen.index] == views[0]
        assert ("positional" in str(screen.query_one("#figure-note").render())) is (not named)
        assert "o open" in str(screen.query_one("#figure-hint").render())
        assert "press o to open it" in str(screen.query_one("#figure-note").render())
        assert screen.query_one("#figure").image is None
        assert figspec.load(views[0][1])["color_by"] == "group"
        await pilot.press("o")
        assert opened == [views[0][1]]
    assert views[0][1] != png and Path(views[0][1]).is_file()
    assert figspec.load(views[0][1])["positional"] is (not named)

    # Reopening uses the persisted view, with no source or in-memory note from the first screen.
    reopened = FigureScreen([("p", png), views[0]], index=1)
    async with Harness(reopened).run_test(size=(110, 30)) as pilot:
        if not named:
            await until(pilot, lambda: "positional" in str(reopened.query_one("#figure-note").render()))
        else:
            await pilot.pause()
            assert "positional" not in str(reopened.query_one("#figure-note").render())
        await pilot.press("left")
        assert "positional" not in str(reopened.query_one("#figure-note").render())
        await pilot.press("right")
        assert ("positional" in str(reopened.query_one("#figure-note").render())) is (not named)


@pytest.mark.parametrize("steps", [1, 2])
@drives
async def test_recolour_after_navigation_inserts_beside_source_and_preserves_selection(
        tmp_path, monkeypatch, steps):
    entry = source(tmp_path)
    originals = [(step, io.save_display_scatter(np.arange(12).reshape(6, 2), [], tmp_path,
                  f"{step}.png", [], step, obs_names=[f"cell-{i}" for i in range(6)])[0])
                 for step in ("phate", "umap", "pca")]
    started, release = threading.Event(), threading.Event()
    real = colorby.recolor_figure

    def delayed(*args, **kwargs):
        started.set()
        if not release.wait(60):
            raise TimeoutError("test did not release colour render")
        return real(*args, **kwargs)

    monkeypatch.setattr(colorby, "recolor_figure", delayed)
    views = []
    screen = FigureScreen(originals, entry=entry, on_view=lambda *view: views.append(view))
    app = Harness(screen)
    try:
        async with app.run_test() as pilot:
            await pilot.press("c")
            await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
            await pilot.press("enter")
            await until(pilot, started.is_set)
            await pilot.press(*(["right"] * steps))
            selected = screen.figures[screen.index]
            assert selected == originals[steps]
            release.set()
            await until(pilot, lambda: bool(views))
            assert screen.figures == [originals[0], views[0], *originals[1:]]
            assert screen.figures[screen.index] == selected
            assert "press o to open" not in notices(app)
            await pilot.press(*(["left"] * steps))
            assert screen.figures[screen.index] == views[0]
            await pilot.press("left")
            assert screen.figures[screen.index] == originals[0]
    finally:
        release.set()


@drives
async def test_duplicate_source_names_recolour_positionally_with_a_visible_warning(tmp_path):
    obs = pd.DataFrame({"condition": ["treated", "control", "treated"]},
                       index=["barcode", "barcode", "other"])
    path = tmp_path / "duplicates.h5ad"
    with pytest.warns(UserWarning, match="Observation names are not unique"):
        AnnData(np.zeros((3, 2)), obs=obs).write_h5ad(path)
    png = io.save_display_scatter(np.arange(6).reshape(3, 2), [], tmp_path,
                                  "p.png", [], "P")[0]
    original = {p: p.read_bytes() for p in (Path(png), *figspec._paths(png))}
    views = []
    screen = FigureScreen([("p", png)], entry=entry_for(path),
                          on_view=lambda *view: views.append(view))
    app = Harness(screen)
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("c")
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        assert app.screen.metadata.columns[0]["key"] == "condition"
        await pilot.press("enter")
        await until(pilot, lambda: bool(views) or "duplicate source names" in notices(app))
        assert views, notices(app)
        assert "positional" in str(screen.query_one("#figure-note").render())
    spec = figspec.load(views[0][1])
    assert spec["positional"] is True and spec["color_by"] == "condition"
    assert spec.get("obs_names") is None
    np.testing.assert_array_equal(spec["color_values"], obs["condition"])
    assert {p: p.read_bytes() for p in original} == original


@drives
async def test_viewer_exports_distinct_colour_views_without_replacing_earlier_files(tmp_path,
                                                                                  monkeypatch):
    monkeypatch.chdir(tmp_path)
    entry, png = source(tmp_path), figure(tmp_path)
    original = {p: p.read_bytes() for p in (Path(png), *figspec._paths(png))}
    screen = FigureScreen([("p", png)], entry=entry, run_id="abc12345deadbeef")
    app = Harness(screen)
    out = tmp_path / "figures"
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.press("s")
        await until(pilot, lambda: "saved to" in notices(app))
        assert out.is_dir()
        earlier = {p: p.read_bytes() for p in out.iterdir()}
        assert {p.suffix for p in earlier} == {".svg", ".pdf", ".png"}
        for i, key in enumerate(("group", "day", "group"), start=1):
            await pilot.press("c")
            await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
            picker = app.screen
            picker.query_one(OptionList).highlighted = next(
                j for j, column in enumerate(picker.metadata.columns) if column["key"] == key)
            await pilot.press("enter")
            await until(pilot, lambda: len(screen.figures) == i + 1)
            assert figspec.load(screen.figures[screen.index][1])["color_by"] == key
            await pilot.press("s")
            await until(pilot, lambda: notices(app).count("saved to") == i + 1)
            assert out.is_dir()
            current = {p: p.read_bytes() for p in out.iterdir()}
            added = current.keys() - earlier.keys()
            assert len(added) == 3
            assert {p.suffix for p in added} == {".svg", ".pdf", ".png"}
            assert {p: current[p] for p in earlier} == earlier
            assert all(p.name.startswith("p-abc12345-") for p in added)
            earlier = current
    assert {p: p.read_bytes() for p in original} == original


@pytest.mark.parametrize("dismiss", [False, True])
@drives
async def test_save_waiting_for_render_lock_keeps_ui_responsive(tmp_path, monkeypatch, dismiss):
    monkeypatch.chdir(tmp_path)
    png = figure(tmp_path)
    screen = FigureScreen([("p", png)])
    app = Harness(screen)
    held, release, expired, ended = (threading.Event() for _ in range(4))

    def hold_renderer():
        with figspec.RENDER_LOCK:
            held.set()
            if not release.wait(3):
                expired.set()  # bounded watchdog: the broken UI cannot release the lock
        ended.set()

    holder = threading.Thread(target=hold_renderer, daemon=True)
    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            holder.start()
            await until(pilot, held.is_set)
            await pilot.press("s")
            ticks = []
            app.call_later(lambda: ticks.append(True))
            await pilot.pause(0.02)
            assert not expired.is_set(), "save blocked the event loop on the renderer"
            assert ticks
            if dismiss:
                await pilot.press("escape")
                assert app.screen is not screen
            release.set()
            await until(pilot, ended.is_set)
            if dismiss:
                await pilot.pause(0.1)
                assert not (tmp_path / "figures").exists()
                assert "saved to" not in notices(app)
            else:
                await until(pilot, lambda: "saved to" in notices(app))
                out = tmp_path / "figures"
                assert out.is_dir()
                assert {p.suffix for p in out.iterdir()} == {".png", ".pdf", ".svg"}
    finally:
        release.set()
        holder.join(timeout=4)


@drives
async def test_save_runs_on_daemon_serializes_duplicates_and_suppresses_late_notice(tmp_path,
                                                                                 monkeypatch):
    from manyruns.tui import figure as figure_module

    started, release, ended = (threading.Event() for _ in range(3))
    calls = []
    ui_thread = threading.get_ident()

    def delayed(*args, **kwargs):
        calls.append(args)
        assert threading.get_ident() != ui_thread
        assert threading.current_thread().daemon
        started.set()
        try:
            assert release.wait(60)
            raise OSError("late export failure")
        finally:
            ended.set()

    monkeypatch.setattr(figure_module, "quickdrop", delayed)
    screen = FigureScreen([("p", figure(tmp_path))], run_id="run-123")
    app = Harness(screen)
    try:
        async with app.run_test() as pilot:
            await pilot.press("s")
            await until(pilot, started.is_set)
            await pilot.press("s")
            assert len(calls) == 1
            assert calls[0] == (screen.figures[0][1], "p", "run-123")
            await pilot.press("escape")
            release.set()
            await until(pilot, ended.is_set)
            await pilot.pause()
            assert "late export failure" not in notices(app)
            assert "saved to" not in notices(app)
    finally:
        release.set()


@drives
async def test_export_errors_are_reported_on_ui_thread_and_allow_retry(tmp_path, monkeypatch):
    from manyruns.tui import figure as figure_module

    monkeypatch.chdir(tmp_path)
    screen = FigureScreen([("p", figure(tmp_path))])
    app = Harness(screen)
    real = figure_module.quickdrop
    notify = screen.notify
    ui_thread = threading.get_ident()
    def checked_notify(*args, **kwargs):
        assert threading.get_ident() == ui_thread
        return notify(*args, **kwargs)
    monkeypatch.setattr(screen, "notify", checked_notify)
    def failed(*args, **kwargs):
        raise OSError("could not export [literal]")
    monkeypatch.setattr(figure_module, "quickdrop", failed)
    async with app.run_test() as pilot:
        await pilot.press("s")
        await until(pilot, lambda: "could not export" in notices(app))
        monkeypatch.setattr(figure_module, "quickdrop", real)
        await pilot.press("s")
        await until(pilot, lambda: "saved to" in notices(app))


@drives
async def test_live_guard_refuses_before_reading_or_writing(tmp_path, monkeypatch):
    figures = [("p", figure(tmp_path)), ("other", figure(tmp_path / "other"))]
    screen = FigureScreen(figures, entry=source(tmp_path), can_recolor=lambda: False)
    calls, jobs = [], []

    def record(name, function):
        def called(*args, **kwargs):
            calls.append(name)
            return function(*args, **kwargs)
        return called

    # Observe forbidden work without raising in a daemon thread, where pytest.fail only
    # produces a warning. Assertions belong on this test's thread after jobs have run.
    monkeypatch.setattr(colorby, "load_metadata", record("metadata", colorby.load_metadata))
    monkeypatch.setattr(figspec, "load", record("spec", figspec.load))
    monkeypatch.setattr(figspec, "_paths", record("sidecars", figspec._paths))
    monkeypatch.setattr(figspec, "write_view", record("write", figspec.write_view))
    real_work = screen._color_work

    def observed_work(job, done, cleanup=lambda result: None):
        finished = threading.Event()
        jobs.append(finished)

        def observed_job():
            try:
                return job()
            finally:
                finished.set()

        real_work(observed_job, done, cleanup)

    monkeypatch.setattr(screen, "_color_work", observed_work)
    app = Harness(screen)
    async with app.run_test() as pilot:
        await pilot.pause()  # initial draw also tries to restore an archived view note
        screen.draw()
        await pilot.press("right")
        assert screen.index == 1
        await pilot.press("c", "left", "c")
        assert screen.index == 0
        await until(pilot, lambda: all(job.is_set() for job in jobs))
        assert "finish this run before recolouring" in notices(app)
        assert calls == [], f"live run accessed metadata/specs: {calls}"


@drives
async def test_empty_metadata_does_not_open_an_empty_modal(tmp_path):
    screen = FigureScreen([("p", figure(tmp_path))], entry=source(tmp_path, empty=True))
    app = Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press("c")
        await until(pilot, lambda: "no metadata columns" in notices(app))
        assert app.screen is screen
        assert "drop an annotated .h5ad" in notices(app)


@pytest.mark.parametrize("kind", ["continuous", "categorical"])
@drives
async def test_recolour_refuses_all_missing_plotted_rows_without_saving_a_view(tmp_path, kind):
    values = ([np.nan, np.inf, -np.inf, *range(27)] if kind == "continuous"
              else [None] * 3 + ["a", "b", "c"] * 9)
    obs = pd.DataFrame({"score": values}, index=[f"cell-{i}" for i in range(30)])
    path = tmp_path / "data.h5ad"
    AnnData(np.zeros((30, 2)), obs=obs).write_h5ad(path)
    entry = entry_for(path)
    metadata = colorby.load_metadata(entry)
    assert metadata.columns[0]["available"] and metadata.columns[0]["kind"] == kind
    png = io.save_display_scatter(np.arange(6).reshape(3, 2), [], tmp_path, "p.png", [], "P",
                                  obs_names=["cell-2", "cell-0", "cell-1"])[0]
    root = Path(png).parent
    assert root.is_dir()
    original = {p: p.read_bytes() for p in root.iterdir()}
    views = []
    screen = FigureScreen([("p", png)], entry=entry, on_view=lambda *view: views.append(view))
    app = Harness(screen)
    async with app.run_test() as pilot:
        await pilot.press("c")
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        await pilot.press("enter")
        await until(pilot, lambda: not screen._color_busy)
        assert "score: all plotted values are missing or nonfinite" in notices(app)
        assert "saved colour view" not in notices(app)
        assert views == [] and screen.figures == [("p", png)]
    assert root.is_dir()
    assert {p: p.read_bytes() for p in root.iterdir()} == original


@drives
async def test_dismissal_during_render_cleans_late_view_and_never_calls_callback(tmp_path, monkeypatch):
    entry, png = source(tmp_path), figure(tmp_path)
    started, release, ended = threading.Event(), threading.Event(), threading.Event()
    views, written = [], []
    real = colorby.recolor_figure

    def delayed(*args, **kwargs):
        started.set()
        assert release.wait(60)
        try:
            result = real(*args, **kwargs)
            written.append(result[0])
            return result
        finally:
            ended.set()

    monkeypatch.setattr(colorby, "recolor_figure", delayed)
    screen = FigureScreen([("p", png)], entry=entry, on_view=lambda *v: views.append(v))
    app = Harness(screen)
    try:
        async with app.run_test() as pilot:
            await pilot.press("c")
            await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
            await pilot.press("enter")
            await until(pilot, started.is_set)
            await pilot.press("c")  # duplicate action is serialized
            await pilot.press("escape")
            release.set()
            await until(pilot, ended.is_set)
            await until(pilot, lambda: bool(written) and not Path(written[0]).exists())
            assert views == [] and "saved colour view" not in notices(app)
    finally:
        release.set()


@pytest.mark.parametrize("case,expected", [
    ("no_entry", "no dataset source"), ("no_source", "no metadata source"),
    ("no_figure", "no figure to recolour"), ("no_spec", "no figure spec"),
    ("removed", "could not read metadata"), ("unreadable", "could not read metadata"),
])
@drives
async def test_unavailable_sources_give_specific_feedback(tmp_path, case, expected):
    entry, png = source(tmp_path), figure(tmp_path)
    figures = [("p", png)]
    if case == "no_entry":
        entry = None
    elif case == "no_source":
        entry = entry_for(None)
    elif case == "no_figure":
        figures = []
    elif case == "no_spec":
        for p in figspec._paths(png):
            p.unlink()
    elif case == "removed":
        entry.path.unlink()
    elif case == "unreadable":
        entry.path.write_bytes(b"not an h5ad")
    app = Harness(FigureScreen(figures, entry=entry))
    async with app.run_test() as pilot:
        await pilot.press("c")
        await until(pilot, lambda: expected in notices(app))
        assert not isinstance(app.screen, colorby.ColorByScreen)


@drives
async def test_disabled_reasons_are_literal_and_match_channel_refusals(tmp_path):
    entry = source(tmp_path)
    metadata = colorby.load_metadata(entry)
    metadata.name = "[red]literal dataset[/red]"
    app = Harness(colorby.ColorByScreen(metadata))
    async with app.run_test(size=(110, 35)) as pilot:
        await pilot.pause()
        rows = app.screen.query_one(OptionList)
        text = "\n".join(s.text for s in app.screen._compositor.render_strips())
        assert "[red]literal dataset[/red]" in text
        assert "all values are missing" in text and "unique value for every row" in text
        for i in (2, 3):
            assert rows.get_option_at_index(i).disabled
            with pytest.raises(ValueError, match=metadata.columns[i]["reason"]):
                colorby.recolor_figure(figure(tmp_path), metadata, metadata.columns[i]["key"])
        await pilot.press("escape")
        assert app.result is None


@drives
async def test_generated_timepoint_recolours_without_a_data_file(tmp_path):
    entry = entry_for(Path(loading.TIME_COURSE_REF), "synthetic_timecourse")
    obj = loading.load_array(entry.path)
    png = io.save_display_scatter(obj.X[:, :2], [], tmp_path, "generated.png", [], "Generated",
                                  obs_names=obj.obs_names)[0]
    views = []
    screen = FigureScreen([("generated", png)], entry=entry, on_view=lambda *v: views.append(v))
    app = Harness(screen)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("c")
        await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
        assert app.screen.metadata.columns[0]["key"] == "timepoint"
        await pilot.press("enter")
        await until(pilot, lambda: bool(views))
    spec = figspec.load(views[0][1])
    assert spec["color_by"] == "timepoint" and spec["color_kind"] == "categorical"
    np.testing.assert_array_equal(spec["obs_names"], obj.obs_names)
    assert loading.labels_of(obj)[1] == "time"


@drives
async def test_picker_escape_writes_nothing_and_allows_another_action(tmp_path):
    screen = FigureScreen([("p", figure(tmp_path))], entry=source(tmp_path))
    root = tmp_path / "plots"
    assert root.is_dir()
    before = {p: p.read_bytes() for p in root.iterdir()}
    app = Harness(screen)
    async with app.run_test() as pilot:
        for _ in range(2):
            await pilot.press("c")
            await until(pilot, lambda: isinstance(app.screen, colorby.ColorByScreen))
            await pilot.press("escape")
            assert app.screen is screen
    assert root.is_dir()
    assert {p: p.read_bytes() for p in root.iterdir()} == before


@drives
async def test_late_metadata_after_dismissal_cannot_open_a_picker(tmp_path, monkeypatch):
    started, release, ended = threading.Event(), threading.Event(), threading.Event()
    real = colorby.load_metadata
    main_thread = threading.get_ident()

    def delayed(entry):
        assert threading.get_ident() != main_thread
        started.set()
        assert release.wait(60)
        try:
            return real(entry)
        finally:
            ended.set()

    monkeypatch.setattr(colorby, "load_metadata", delayed)
    screen = FigureScreen([("p", figure(tmp_path))], entry=source(tmp_path))
    app = Harness(screen)
    try:
        async with app.run_test() as pilot:
            await pilot.press("c")
            await until(pilot, started.is_set)
            await pilot.press("escape")
            release.set()
            await until(pilot, ended.is_set)
            await pilot.pause()
            assert app.screen is not screen
            assert not isinstance(app.screen, colorby.ColorByScreen)
    finally:
        release.set()
