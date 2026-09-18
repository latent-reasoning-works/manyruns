"""Offline ownership and lifecycle checks; no claim about terminal-emulator rendering."""
import asyncio
import functools
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
    other.render().terminal_image_id = 99
    old = first.render()
    old.terminal_image_id = 11

    first.image = pixels.copy()  # replacement uses the renderable's corrected cleanup
    assert old.terminal_image_id is None
    first.render().terminal_image_id = 12
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
