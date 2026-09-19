"""The one place this product decides how a picture reaches a terminal.

`textual-image` does the drawing and picks its own tier — sixel → TGP → half-cell → unicode.
It probes sixel first because only kitty supports its TGP placeholder implementation;
Konsole and WezTerm report TGP but need sixel with this library. Kitty does not support sixel,
so it reaches TGP. The filming cost: sixel cleanup is a no-op, so our owned-image unmount
cleanup applies on kitty, not on terminals that land on sixel. This module is a seam, not a
renderer. It exists for three reasons, each otherwise repeated at every call site:

**1. A partial install has no images and must not raise.** `textual-image` is a hard dependency
now that the compute is fixed in, so its absence means a broken or hand-assembled environment
rather than a choice. The import is guarded here once anyway — a front door that raises on a
missing package is worse than one that prints the path, and `AutoImage` degrades to a widget
that does exactly that, which is what the pane did before any of this.

**2. The tier has to be askable.** A cell-based tier is a THUMBNAIL: 60×20 cells is 60×40 pixels,
and a scatter of an embedding is pointwise — the branch tip with three cells on it is the finding.
Screens phrase that honestly (`figure.FigureScreen`) rather than presenting a smear as the
picture, and `is_pixel_perfect()` is how they know which case they are in.

**3. Detection happens at IMPORT and needs a live tty.** `textual_image.renderable` queries the
terminal at module scope — "querying the terminal isn't possible anymore once Textual is started"
is its own note — so the import must be eager and must happen before `App.run()`. Screens import
this module at module scope, which puts it inside `manyruns.tui.app`'s import, which
`app.cmd_app` performs before launching. Moving it into a handler would silently downgrade every
terminal to the fallback tier.

WHAT THIS REPLACES. `tui/pixels.py` was a hand-rolled half-block renderer — `▀` with a foreground
and a background, two pixels per cell, sampled from the PNG. It was written on the belief that an
image escape could not survive Textual's compositor. It can (measured in a pty, 4 kitty/TGP
sequences out of a two-pane scrolling layout); the earlier finding was about a naive escape
inside a `Static`. Its one property no image renderer has — binning the coordinates, which
preserved a 20x density ratio where the PNG route reported 4.64x — turned out to optimise the
wrong quantity: density is not what a PHATE or trajectory plot is read for. Pointwise is.
"""
from __future__ import annotations

from typing import Any

from manyruns import env

#: Force a tier when detection is wrong. `MANYRUNS_IMAGE=tgp|sixel|halfcell|unicode`.
#:
#: Detection asks the terminal and believes the answer, which is right until a terminal lies or
#: cannot be asked: tmux without passthrough swallows the query, a multiplexer answers for the
#: wrong terminal. The CLI's VS Code `enableImages` advice is not a TUI tier switch:
#: textual-image has no iTerm renderer. An override is the difference between "images do not work
#: here" and one environment variable, and it is also what makes the tier drivable from a pty.
_OVERRIDE = "IMAGE"

try:  # a hard dependency; guarded so a broken install degrades instead of raising
    from textual_image.renderable import Image as _Detected
    from textual_image.renderable import HalfcellImage as _Half
    from textual_image.renderable import SixelImage as _Sixel
    from textual_image.renderable import TGPImage as _TGP
    from textual_image.renderable import UnicodeImage as _Unicode
    from textual_image.widget import HalfcellImage as _WHalf
    from textual_image.widget import Image as _Detected_Widget
    from textual_image.widget import SixelImage as _WSixel
    from textual_image.widget import TGPImage as _WTGP
    from textual_image.widget import UnicodeImage as _WUnicode

    _BY_NAME = {"tgp": (_TGP, _WTGP), "sixel": (_Sixel, _WSixel),
                "halfcell": (_Half, _WHalf), "unicode": (_Unicode, _WUnicode)}
    # `widget.Image`, not `widget.AutoImage`: on a sixel terminal the library swaps in its
    # `SixelImage` widget, which carries sixel-specific options its generic one does not. Taking
    # whichever it chose means this seam does not quietly downgrade the best tier.
    _Renderable, _Widget = _BY_NAME.get(env.get(_OVERRIDE, "").strip().lower(),
                                        (_Detected, _Detected_Widget))
    HAVE_IMAGES = True
except ImportError:  # pragma: no cover - exercised by the base-install run, not by this suite
    _Renderable = _Sixel = _TGP = _Widget = None
    HAVE_IMAGES = False


def is_pixel_perfect() -> bool:
    """Is the chosen tier a real graphics protocol, or a grid of coloured cells?

    True only for sixel and TGP. Half-cell and unicode are honest fallbacks and dishonest
    scatters — at two pixels per cell they cannot show where an individual point sits, which is
    the only thing a manifold embedding is looked at for.
    """
    return HAVE_IMAGES and _Renderable in (_Sixel, _TGP)


def tier() -> str:
    """Which renderer is in use, for a footnote or a bug report."""
    if not HAVE_IMAGES:
        return "none"
    return {_TGP: "tgp", _Sixel: "sixel"}.get(_Renderable, _Renderable.__module__.rsplit(".", 1)[-1])


def forced() -> bool:
    """Was the tier chosen by `$MANYRUNS_IMAGE` rather than by asking the terminal?

    **A forced tier can render nothing at all, silently.** Sixel escapes go out whether or not
    the terminal can decode them — forcing sixel on an unsupported terminal can leave the
    pane simply empty, which on a dark theme reads as a
    black rectangle with no explanation. Detection cannot produce that, because it only picks a
    graphics tier the terminal claimed. So a forced tier is worth SAYING, and an auto-detected
    one is not.
    """
    return HAVE_IMAGES and env.get(_OVERRIDE, "").strip().lower() in _BY_NAME


def hint() -> str:
    """Tier guidance for Textual; the CLI's iTerm protocol is not available here."""
    return ("TUI images need kitty (TGP) or a sixel-capable terminal; restart manyruns there "
            "for detection. If detection is wrong, set MANYRUNS_IMAGE=tgp (kitty) or "
            "MANYRUNS_IMAGE=sixel (with sixel support).")


if HAVE_IMAGES:
    from PIL import Image as PILImage
    from textual.strip import Strip

    class _SafeImageRenderable:
        """Keep lazy drawing inside the guard, including any nested renderables."""

        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            # Rich measurement and Textual's protocol probes still reach the original.
            return getattr(self._inner, name)

        def __rich_console__(self, console, options):
            try:
                segments = list(console.render(self._inner, options))
            except Exception:  # noqa: BLE001 - a failed draw must not fail the screen
                return
            yield from segments

    def _guard_sixel_draw(child):
        # Wrap the composed instance's public method: no private upstream class import,
        # constructor, or CSS identity to keep in sync across textual-image releases.
        render_lines = child.render_lines

        def safe_render_lines(crop):
            try:
                if child.content_size.width > 0 and child.content_size.height > 0:
                    return render_lines(crop)
            except Exception:  # noqa: BLE001 - sixel draws outside Rich's render protocol
                pass
            return [Strip.blank(crop.width) for _ in range(crop.height)]

        child.render_lines = safe_render_lines

    class _OwnedTGP(_TGP):
        """Delete only this renderable's upload, on replacement as well as disposal.

        The installed textual-image cleanup sends `a=d,I=<id>` while its upload uses `i`.
        Supply the ID selector ourselves; retaining upstream cleanup would lose the ID after
        emitting that different command. Keep its sender for terminal/multiplexer framing.
        This is an offline protocol correction, not evidence of an emulator's behaviour.
        """

        def cleanup(self) -> None:
            # `_send_tgp_message` IS PRIVATE UPSTREAM, so its presence is a fact about the
            # installed release rather than a contract. Measured on CI: a freshly resolved
            # textual-image on 3.11 exposed a different set of names from 3.12's. This runs
            # from `on_unmount`, and an exception out of a Textual handler takes the app down
            # — so a library that has moved on costs the deletion and nothing else. The
            # placement then outlives us, which is the state before this class existed.
            try:
                from textual_image.renderable.tgp import _send_tgp_message
            except ImportError:
                self.terminal_image_id = None
                return

            if self.terminal_image_id is not None:
                try:
                    _send_tgp_message(a="d", d="I", i=self.terminal_image_id, q=2)
                except Exception:  # noqa: BLE001 - cleanup must never fail the screen
                    pass
                self.terminal_image_id = None

    class AutoImage(_Widget, Renderable=_OwnedTGP if _Renderable is _TGP else _Renderable):  # type: ignore[misc,valid-type,call-arg]
        """Snapshot pixels on assignment; repainting never reopens the source file.

        A decoded copy outlives a rename, replacement, or deletion during tuning. The `image`
        getter retains the assigned source for callers, while the renderer uses only pixels.
        Decode, construction, and draw-time failures degrade to no picture, including zero-size
        scaling in textual-image 0.12, which is the version that resolves on Python 3.11.

        `Renderable=` is required by `BaseImage.__init_subclass__` and is the same one the
        library picked, with owned TGP cleanup — subclassing without it is a TypeError.
        """

        def on_unmount(self) -> None:
            # Assignment and rerender already clean up; the installed widget omits unmount.
            # Clear the reference so repeated disposal cannot delete twice. At app quit,
            # Textual stops application mode BEFORE unmount dispatch: this hook alone cannot
            # establish alternate-screen cleanup timing. Sixel's cleanup remains a no-op.
            if self._renderable is not None:
                self._renderable.cleanup()
                self._renderable = None

        @property
        def image(self):  # noqa: D102 - see the base class
            return self._source

        @image.setter
        def image(self, value) -> None:
            self._source = None
            try:
                if value is None:
                    pixels = None
                elif isinstance(value, PILImage.Image):
                    pixels = value.copy()
                else:
                    with PILImage.open(value) as opened:
                        pixels = opened.copy()
                _Widget.image.fset(self, pixels)
                self._source = value
            except Exception:  # noqa: BLE001 - every decode failure degrades to no picture
                _Widget.image.fset(self, None)

        def render(self):  # noqa: D102 - see the base class
            # Before mounting, content_size is unknown rather than a squeezed pane.
            if self.is_mounted and (self.content_size.width == 0 or self.content_size.height == 0):
                return ""
            try:
                rendered = super().render()
                # The base retains the INNER renderable for replacement/unmount cleanup.
                return _SafeImageRenderable(rendered) if rendered != "" else ""
            except Exception:  # noqa: BLE001 - constructing a picture must not fail a paint
                self._renderable = None
                return ""

        if _Widget is _WSixel:
            def compose(self):  # noqa: D102 - see the base class
                # Sixel composes a child from the public getter; give it the snapshot too.
                for child in super().compose():
                    child.image = self._image
                    _guard_sixel_draw(child)
                    yield child

else:
    from textual.widgets import Static

    class AutoImage(Static):  # type: ignore[no-redef]
        """The no-images stand-in, with the API the screens already use.

        Assigning `.image` sets a line of text instead of pixels — the same thing the figure
        pane did before any of this, and the same thing it must keep doing on an install where
        there is nothing to draw. It names the product rather than the library, because
        the product is the thing a user installs and `textual-image` is our business.
        """

        _image: Any = None

        @property
        def image(self) -> Any:
            return self._image

        @image.setter
        def image(self, value: Any) -> None:
            self._image = value
            from pathlib import Path

            from rich.markup import escape

            self.update(f"[dim]{escape(Path(str(value)).name)} — no image support in "
                        f"this install[/dim]" if value else "")
