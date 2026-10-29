"""`manyruns actions` — the legal moves for one dataset, as one JSON document.

THE LEDGER, SERIALIZED, for any outside program that drives the CLI. It answers the question a
person answers by looking at the ledger screen: which analyses can run on this data, which
cannot and why, and which one is recommended. It is not a second account of any of that:
legality, the recommendation, membership and order all come from `tui.state.ledger`, and each
`offered` entry is exactly `LedgerRow.as_offer()`, the conversion a decision row records.

**Declared, never sniffed.** The Observation is `narrate.declared_observation`, built from the
YAML alone, so the document is the same on a machine that holds the data file and on one that
does not. Nothing here opens, downloads or writes anything, builds a `Session` or runs a step.

**No phrasing and no outcomes.** A recipe's `question:` text is presentation and a run's result
is the run record's business (`outputs/index.jsonl`); neither belongs in a statement of what is
legal.

Only the `ledger` surface exists. A caller learns the build it is talking to from
`tool_version`, and which task space it chose in from `legal_id`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

from manyruns import catalog, narrate, vocab

#: The surfaces `actions` can describe. One today.
SURFACES = ("ledger",)

SCHEMA_VERSION = 1


def legal_id(legal: Sequence[str], tool_version: str) -> str:
    """Which task space a choice was made in: the sorted legal set plus the build.

    Order-free by construction, so two callers that list the same legal recipes differently
    get one id. Two rows are comparable only when this is equal (environment contract §2.3).
    """
    raw = json.dumps(sorted(legal)) + "|" + tool_version
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _observation(name: str, config_dir: Optional[Path | str] = None):
    ds = catalog.load_dataset(name, config_dir)
    return ds, narrate.declared_observation(ds, name)


def _ledger(obs):
    from manyruns.tui import state

    return state.ledger(obs, obs.provides(), measured=None)


def ledger_offer(name: str, config_dir: Optional[Path | str] = None) -> list[dict]:
    """The offer set for a dataset, from its declaration: `LedgerRow.as_offer()` per row, in
    ledger order. The one source of it for `actions` and `decide`."""
    _, obs = _observation(name, config_dir)
    return [r.as_offer() for r in _ledger(obs)]


def document(name: str, tool_version: str,
             config_dir: Optional[Path | str] = None) -> dict:
    """The whole `actions` document for one dataset. Raises what `catalog.load_dataset` raises."""
    ds, obs = _observation(name, config_dir)
    provided = obs.provides()
    rows = _ledger(obs)
    offered = [r.as_offer() for r in rows]
    rank = {o["recipe"]: i for i, o in
            enumerate(narrate.offer(obs, available=catalog.discover_recipes()))}
    detail = {}
    for row in rows:
        recipe = catalog.load_recipe(row.recipe)
        detail[row.recipe] = {
            "id": recipe.get("id"),
            "steps": [f"{s.get('group')}:{s.get('name')}" for s in recipe.get("steps") or ()],
            "claims": recipe.get("claims"),
            "suits": list(recipe.get("suits") or ()),
            # The same call `narrate.blocked` makes for the row, so `unmet == []` is exactly
            # the data half of `can_run`.
            "unmet": sorted(vocab.unmet(recipe, obs.shape, provided)),
            "invalidated": sorted(vocab.invalidated(recipe, obs.shape, provided)),
            "unavailable": narrate.unavailable(recipe),
            "offer_rank": rank.get(row.recipe),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": tool_version,
        "dataset": {
            "name": name,
            "shape": obs.shape,
            "topology": list(ds.get("topology") or ()),
            "modality": obs.modality,
            "handle_kind": (ds.get("handle") or {}).get("kind"),
            "provides": sorted(provided),
            "dims": ds.get("dims") or None,
        },
        "offered": offered,
        "detail": detail,
        "legal_id": legal_id([o["recipe"] for o in offered if o["can_run"]], tool_version),
    }
