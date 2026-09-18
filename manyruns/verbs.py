"""The harness's tool surface — the top-level things the interactive shell can *do*.

One declarative registry, two consumers:
  - the shell's home menu renders from it (`manyruns/shell.py`);
  - the optional LLM intent tier routes a free-typed request to one of these names
    (`manyruns/intent.py`).

This is the "tool access" layer of the harness: the verbs
are the moves; the shell owns the loop; the model, when present, only *chooses* the next move.
Dependency-free on purpose — importing this must never pull rich, questionary, or an SDK.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Verb:
    name: str
    summary: str        # shown in the menu and handed to the LLM router as the choice's meaning
    needs_data: bool = False


# The home menu, in display order. `explore` is the front door; the rest are catalog/utility.
VERBS: tuple[Verb, ...] = (
    Verb("explore", "Explore a dataset — see its shape, run a recipe, get a plain-English finding",
         needs_data=True),
    Verb("check", "Check the catalog — validate every recipe and dataset, and which the data can run"),
    Verb("recipes", "List the available recipes (the workflows you can run)"),
    Verb("datasets", "List the available datasets"),
    Verb("quit", "Leave"),
)

_BY_NAME = {v.name: v for v in VERBS}


def get(name: str) -> Verb | None:
    return _BY_NAME.get(name)


def names() -> list[str]:
    return [v.name for v in VERBS]
