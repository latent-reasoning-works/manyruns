"""Offline ownership and lifecycle checks; no claim about terminal-emulator rendering."""
import asyncio
import functools
import importlib.util
import io
import sys

import pytest
from PIL import Image

from manyruns.tui import images


@pytest.fixture
def tgp_output(monkeypatch):
    from textual_image.renderable import tgp

    # THE BASE-INSTALL JOB RESOLVES textual-image FRESH, so the private seam this fixture
    # patches is whatever that day's release exposes: measured on CI, 3.11 resolved a build
    # with no `prepare_terminal_sequence` while 3.12 had one, and `monkeypatch.setattr` on an
    # absent attribute is an error rather than a skip. What is under test is OUR cleanup, so a
    # library that does not offer the seam has nothing here to check.
    for seam in ("prepare_terminal_sequence", "_send_tgp_message"):
        if not hasattr(tgp, seam):
            pytest.skip(f"installed textual-image exposes no {seam!r} to observe")

    output = io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", output)
    monkeypatch.setattr(tgp, "prepare_terminal_sequence", lambda sequence: sequence)
    monkeypatch.setattr(images.AutoImage, "_Renderable", images._OwnedTGP)
    return output


def test_replacement_rerender_and_unmount_delete_only_the_owned_upload(tgp_output):
    pixels = Image.new("RGB", (4, 4))
    first = images.AutoImage(pixels)
    other = images.AutoImage(pixels)
    other.render()
    other._renderable.terminal_image_id = 99
    first.render()
    old = first._renderable
    old.terminal_image_id = 11

    first.image = pixels.copy()  # replacement uses the renderable's corrected cleanup
    assert old.terminal_image_id is None
    first.render()
    first._renderable.terminal_image_id = 12
    first.render()               # resize/rerender uses that same cleanup
    first._renderable.terminal_image_id = 13
    first.on_unmount()
    first.on_unmount()
    assert first._renderable is None
    assert other._renderable.terminal_image_id == 99
    assert tgp_output.getvalue() == "".join(
        f"\x1b_Ga=d,d=I,i={number},q=2\x1b\\" for number in (11, 12, 13))
    other.on_unmount()


def test_an_unuploaded_image_and_repeated_cleanup_emit_nothing(tgp_output):
    rendered = images._OwnedTGP(Image.new("RGB", (4, 4)))
    rendered.cleanup()
    rendered.cleanup()
    assert tgp_output.getvalue() == ""


def drives(body):
    @functools.wraps(body)
    def run(*args, **kwargs):
        return asyncio.run(body(*args, **kwargs))
    return run


@drives
async def test_textual_removal_dispatches_owned_cleanup(tgp_output):
    from textual.app import App

    widget = images.AutoImage(Image.new("RGB", (4, 4)))
    class Harness(App):
        def compose(self):
            yield widget

    async with Harness().run_test() as pilot:
        await pilot.pause()
        # Headless Textual does not upload pixels; attach an owned ID to its live renderable.
        widget._renderable.terminal_image_id = 42
        await widget.remove()
        await pilot.pause()
        assert widget._renderable is None
    assert "\x1b_Ga=d,d=I,i=42,q=2\x1b\\" in tgp_output.getvalue()


@pytest.fixture(params=["halfcell", "sixel", "tgp"])
def forced_images(request, monkeypatch):
    monkeypatch.setenv("MANYRUNS_IMAGE", request.param)
    # Load tier selection independently without changing the screens' imported class.
    spec = importlib.util.spec_from_file_location("forced_images", images.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.tier() == request.param
    return module


@pytest.mark.parametrize("change", ["rename", "delete", "truncate"])
@drives
async def test_repaint_keeps_pixels_after_source_changes(tmp_path, monkeypatch, forced_images, change):
    from textual.app import App

    path = tmp_path / "figure.png"
    Image.new("RGB", (120, 90), "red").save(path)
    widget = forced_images.AutoImage(path)

    class Harness(App):
        def compose(self):
            yield widget

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        previous = widget._renderable
        if change == "rename":
            path.rename(path.with_name("figure@1.png"))
        elif change == "delete":
            path.unlink()
        else:
            path.write_bytes(b"\x89PNG\r\n\x1a\n")
        opens = []
        real_open = Image.open

        def track_open(*args, **kwargs):
            opens.append(args[0])
            return real_open(*args, **kwargs)

        monkeypatch.setattr(Image, "open", track_open)
        widget.refresh(layout=True)
        await pilot.resize_terminal(90, 30)
        await pilot.pause()
        assert app.is_running
        assert widget._renderable is not previous
        assert opens == [], "repainting must not reopen the source"
        assert widget.image == path
        assert widget._image.getpixel((0, 0)) == (255, 0, 0)


@drives
async def test_tiny_image_draw_keeps_app_running(forced_images):
    from textual.app import App

    widget = forced_images.AutoImage(Image.new("RGB", (4, 4), "red"))

    class Harness(App):
        def compose(self):
            yield widget

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        widget.refresh(layout=True)
        await pilot.resize_terminal(20, 5)
        await pilot.pause()
        assert app.is_running


@drives
async def test_draw_failure_keeps_app_running(monkeypatch, forced_images):
    from rich.segment import Segment
    from textual.app import App
    from textual.geometry import Region

    widget = forced_images.AutoImage(Image.new("RGB", (120, 90), "red"))
    draws = []

    if forced_images.tier() == "sixel":
        # Discover the implementation through composition, without importing a private name.
        child_type = type(next(widget.compose()))

        def fail(self, crop):
            draws.append(crop)
            raise ValueError("height and width must be > 0")

        monkeypatch.setattr(child_type, "render_lines", fail)
    else:
        def fail(self, console, options):
            draws.append(options)
            yield Segment("partial image")
            raise ValueError("height and width must be > 0")

        monkeypatch.setattr(widget._Renderable, "__rich_console__", fail)

    class Harness(App):
        def compose(self):
            yield widget

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert draws, "exercise drawing, not just constructing a renderable"
        assert app.is_running
        if forced_images.tier() == "sixel":
            strips = widget.children[0].render_lines(Region(0, 0, 7, 3))
            assert len(strips) == 3
            assert all(strip.cell_length == 7 and strip.text == " " * 7 for strip in strips)
        else:
            assert list(app.console.render(widget.render())) == []


@pytest.mark.parametrize("fails", [False, True])
def test_draw_proxy_forwards_measurement_and_buffers_nested_output(monkeypatch, fails):
    from rich.console import Console
    from rich.measure import Measurement
    from rich.segment import Segment

    class Nested:
        def __rich_console__(self, console, options):
            yield Segment("picture")
            if fails:
                raise RuntimeError("late nested draw failure")

    class Renderable:
        marker = object()

        def __init__(self, *args):
            pass

        def __rich_measure__(self, console, options):
            return Measurement(7, 9)

        def __rich_console__(self, console, options):
            yield Nested()

    widget = images.AutoImage(Image.new("RGB", (120, 90)))
    monkeypatch.setattr(widget, "_Renderable", Renderable)
    rendered = widget.render()
    console = Console(width=80)
    assert isinstance(widget._renderable, Renderable)
    assert rendered.marker is widget._renderable.marker
    assert Measurement.get(console, console.options, rendered) == Measurement(7, 9)
    assert list(console.render(rendered)) == ([] if fails else [Segment("picture")])


@pytest.mark.parametrize("size", [(0, 5), (5, 0)])
def test_zero_content_size_skips_drawing(monkeypatch, forced_images, size):
    from textual.geometry import Region, Size

    widget = forced_images.AutoImage(Image.new("RGB", (120, 90)))
    monkeypatch.setattr(type(widget), "is_mounted", property(lambda self: True))
    monkeypatch.setattr(type(widget), "content_size", property(lambda self: Size(*size)))

    def fail(*args):
        pytest.fail("zero-size content must skip image drawing")

    monkeypatch.setattr(widget, "_Renderable", fail)
    assert widget.render() == ""
    if forced_images.tier() == "sixel":
        child_type = type(next(widget.compose()))
        monkeypatch.setattr(child_type, "content_size", property(lambda self: Size(*size)))
        monkeypatch.setattr(child_type, "render_lines", fail)
        child = next(widget.compose())
        strips = child.render_lines(Region(0, 0, 7, 3))
        assert len(strips) == 3
        assert all(strip.cell_length == 7 and strip.text == " " * 7 for strip in strips)


@pytest.mark.parametrize("error", [OSError, ValueError, RuntimeError])
def test_renderable_failure_degrades_to_no_picture(monkeypatch, error):
    widget = images.AutoImage(Image.new("RGB", (4, 4)))

    def fail(*args, **kwargs):
        raise error("renderable failed")

    monkeypatch.setattr(widget, "_Renderable", fail)
    assert widget.render() == ""
    assert widget._renderable is None


def test_assignment_decodes_before_accepting_a_file(tmp_path):
    path = tmp_path / "partial.png"
    Image.new("RGB", (4, 4)).save(path)
    path.write_bytes(path.read_bytes()[:41])  # Header and dimensions, but no pixel data.
    widget = images.AutoImage(path)
    assert widget.image is None
    assert widget.render() == ""
