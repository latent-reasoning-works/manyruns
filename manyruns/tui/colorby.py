"""Shared metadata picker and file-producing colour views, independent of run screens.

``load_metadata(entry)`` makes a detached metadata snapshot. ``ColorByScreen`` returns a
column key (or None); ``recolor_figure(png, metadata, key, *, spec=None)`` returns a new
``(path, positional)`` view. Call the file/metadata helpers off the UI thread. Nothing here
runs analysis, changes decision rows, or modifies the source or its original figures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from manyruns import figspec
from manyruns.pipeline import loading


@dataclass
class Metadata:
    """Detached obs only; no matrix, open file or cached availability decisions."""

    name: str
    obs: Any
    columns: list[dict]

    @property
    def empty_message(self) -> str:
        return (f"{self.name} has no metadata columns to colour by — "
                "drop an annotated .h5ad to pick one")


def load_metadata(entry: Any) -> Metadata:
    """Read fresh metadata on explicit request, including the local generated time course.

    Ignore stale ``entry.obs.columns``; a cached None is not evidence of no columns.
    Ordinary .h5ad files are opened backed and closed even on classification errors.
    Other sources use the analysis loader's dispatch, file selection, sample concatenation
    and row identities, then retain only detached obs. Python scripts are refused without
    execution: repeated row names cannot establish identity across generator calls, and
    no original metadata snapshot is persisted. The built-in time course is deterministic.
    Engine-only generators (path=None) expose no metadata through this product.
    """
    name = str(getattr(entry, "name", "this dataset"))
    path = getattr(entry, "path", None)
    if path is None:
        raise ValueError(f"{name} has no metadata source exposed by this product")
    obj = None
    try:
        if Path(path).suffix.lower() == ".h5ad" and not Path(path).is_dir():
            import anndata

            obj = anndata.read_h5ad(path, backed="r")
        else:
            obj = loading.load_array(path, allow_scripts=False)
        obs = getattr(obj, "obs", None)
        if obs is None:
            import pandas as pd

            return Metadata(name, pd.DataFrame(), [])
        obs = obs.copy(deep=True)
        columns = loading.color_columns_of(SimpleNamespace(obs=obs))
        return Metadata(name, obs, columns)
    except Exception as error:
        raise ValueError(f"could not read metadata for {name} from {path}: {error}") from error
    finally:
        if obj is not None and getattr(obj, "isbacked", False) and obj.file is not None:
            obj.file.close()


def recolor_figure(png: str | Path, metadata: Metadata, key: str, *,
                   spec: Optional[dict] = None) -> tuple[str, bool]:
    """Produce a unique view from a captured spec/source; return ``(path, positional)``.

    Shared CLI/UI availability is checked on the full source; alignment selects surviving
    rows once and refuses an all-missing selection. A legacy equal-length positional join
    is persisted and disclosed by callers.
    Raises on unreadable specs, unavailable columns, alignment errors or incomplete writes.
    """
    original = figspec.load(png) if spec is None else spec
    if original is None:
        raise ValueError("no figure spec beside this PNG — re-run to get one")
    channel = loading.color_channel_of(SimpleNamespace(obs=metadata.obs), key)
    values, positional = figspec.join(original, metadata.obs, key)
    reason = loading.color_selection_reason(values)
    if reason is not None:
        raise ValueError(f"{key}: {reason}")
    changed = figspec.recolor(original, values, key, channel["kind"])
    changed["positional"] = positional
    return figspec.write_view(png, changed), positional


class ColorByScreen(ModalScreen[Optional[str]]):
    """Pick one available column; Enter/click returns its key, Escape returns None.

    Callers handle ``metadata.columns == []`` by showing ``metadata.empty_message`` instead
    of pushing an empty modal. Labels use literal Rich Text so external markup stays text.
    """

    BINDINGS = [Binding("escape", "cancel", "back", priority=True)]
    DEFAULT_CSS = """
    ColorByScreen { align: center middle; background: $background 70%; }
    ColorByScreen #color-dialog { width: 85%; max-width: 100; height: auto;
        max-height: 85%; border: round $accent; padding: 1 2; background: $surface; }
    ColorByScreen #color-title { height: auto; margin-bottom: 1; }
    ColorByScreen #color-options { height: auto; max-height: 18; }
    ColorByScreen #color-help { height: auto; margin-top: 1; color: $text-muted; }
    """

    def __init__(self, metadata: Metadata, current: Optional[str] = None) -> None:
        super().__init__()
        self.metadata = metadata
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="color-dialog"):
            yield Static(Text(f"Colour {self.metadata.name} by a metadata column"), id="color-title")
            options = []
            for i, col in enumerate(self.metadata.columns):
                current = " · current" if col["key"] == self.current else ""
                line = (f"{col['key']} · {col['kind']} · {col['n_distinct']} distinct{current}")
                details = []
                if col["n_missing"]:
                    details.append(f"{col['n_missing']} missing")
                if col["n_nonfinite"] > col["n_missing"]:
                    details.append(f"{col['n_nonfinite']} missing/nonfinite")
                details.extend(filter(None, (col["reason"], col["warning"])))
                if details:
                    line += "\n  " + " · ".join(details)
                options.append(Option(Text(line), id=f"column-{i}", disabled=not col["available"]))
            yield OptionList(*options, id="color-options", compact=True)
            yield Static("Enter or click to choose · esc back", id="color-help")

    def on_mount(self) -> None:
        rows = self.query_one(OptionList)
        available = [i for i, c in enumerate(self.metadata.columns) if c["available"]]
        current = next((i for i in available if self.metadata.columns[i]["key"] == self.current), None)
        rows.highlighted = current if current is not None else next(iter(available), None)
        rows.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        col = self.metadata.columns[event.option_index]
        if col["available"]:
            self.dismiss(col["key"])

    def action_cancel(self) -> None:
        self.dismiss(None)
