"""The verdict writer — one structure-class prediction about one stored run.

A **verdict** is a probability for each of the six `vocab.TOPOLOGIES` names, stated about the
run that produced the graph. `manyruns claim` writes it, and `score.class_auprc` and
`score.brier` score it against label annotations.

A verdict is a new row of `outputs/index.jsonl`, not an edit of the run row and not an
annotation. It copies the identity fields of its run (so it shares that run's `run_id`), leaves
`ok` and `complete` null and `steps` and `g_vector` empty, and carries the prediction in the
nested ``extra.verdict`` namespace that `store.py` reserves for downstream fields. A reader
tells the two row types apart with `score.verdict_of(row)`.

The probabilities are independent, one per class, with no sum constraint: a dataset can be
`clusters` and `multi-branching` at once. A class the claim does not name is `0.0`.

`chooser` and `bound` follow the environment contract (`docs/environment-contract.md`, §2.1 and
§2.2). A bare `human` is refused here: a verdict names a pseudonym, a model, a learner or a
prior. A `bound` is never assumed for a non-human chooser, because only the code path that
constrained a model's output knows how it was constrained.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from manyruns import annotations, decisions, score, store, vocab

#: The one chooser grammar manyruns exports. The `human:` part is the annotation pseudonym
#: pattern, imported rather than copied, so one annotator id is valid in a verdict row exactly
#: when it is valid in a label row. A bare `human` does not match.
CHOOSER_PATTERN = re.compile(
    rf"(?:human:{annotations.PSEUDONYM_PATTERN}"
    rf"|llm:{annotations._MODEL_PATTERN}"
    r"|learner:[A-Za-z0-9][A-Za-z0-9._@-]{0,79}"
    r"|prior:[A-Za-z0-9][A-Za-z0-9._-]{0,79})"
)

#: The fields a verdict row copies from its run row.
_COPIED = ("run_id", "spec_id", "parent", "dataset", "dataset_name", "data_kwargs", "recipe",
           "engine", "seed", "tool_version")


def parse_topology(text: str) -> dict[str, float]:
    """`NAME=P[,NAME=P...]` into six probabilities in `TOPOLOGIES` order; unnamed names are 0.0.

    `ValueError`, naming the bad token, for an unknown name, a value that is not a float, a
    value outside [0, 1] or NaN, a repeated name, or a text with no pair. Not normalized.
    """
    out = {t: 0.0 for t in vocab.TOPOLOGIES}
    seen: set[str] = set()
    tokens = [tok.strip() for tok in (text or "").split(",") if tok.strip()]
    if not tokens:
        raise ValueError(f"--topology has no NAME=P pair: {text!r}")
    for tok in tokens:
        name, sep, value = tok.partition("=")
        name, value = name.strip(), value.strip()
        if not sep:
            raise ValueError(f"--topology token {tok!r} is not NAME=P")
        if name not in vocab.TOPOLOGIES:
            raise ValueError(f"unknown topology {name!r} in {tok!r}: must be from "
                             f"{', '.join(vocab.TOPOLOGIES)}")
        try:
            p = float(value)
        except ValueError:
            raise ValueError(f"probability {value!r} for {name} is not a number") from None
        if math.isnan(p) or not 0.0 <= p <= 1.0:
            raise ValueError(f"probability {value!r} for {name} must be in [0, 1]")
        if name in seen:
            raise ValueError(f"topology {name!r} is named twice in --topology")
        seen.add(name)
        out[name] = p
    return out


def legal_id(legal: Sequence[str]) -> str:
    """The decision-row `legal_id` formula: a hash of the sorted legal set plus `vocab.VERSION`.

    `VERSION` is read at call time, so a vocabulary bump moves every id."""
    return hashlib.sha256(
        (json.dumps(sorted(set(legal))) + vocab.VERSION).encode()).hexdigest()[:12]


def _resolve(chooser: Any, bound: Optional[str]) -> str:
    """Check `chooser` and return the bound to store. `ValueError` on any failure."""
    if not isinstance(chooser, str) or not CHOOSER_PATTERN.fullmatch(chooser):
        raise ValueError(
            f"chooser {chooser!r} is not one of human:<pseudonym>, llm:<model>, learner:<id> or "
            "prior:<id> (a pseudonym is lowercase letters, digits, _ and -, at most 32 "
            "characters; a bare 'human' is not accepted)")
    if bound is None:
        if chooser.startswith("human:"):
            return "none"
        raise ValueError(f"chooser {chooser!r} is not a person: say how its output was "
                         f"constrained with --bound, one of {', '.join(decisions.BOUNDS)}")
    if bound not in decisions.BOUNDS:
        raise ValueError(f"unknown bound {bound!r}: must be one of {', '.join(decisions.BOUNDS)}")
    return bound


def _check_probs(topology_probs: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(topology_probs, Mapping) or set(topology_probs) != set(vocab.TOPOLOGIES):
        raise ValueError("topology_probs must name exactly the six vocab.TOPOLOGIES (use "
                         "parse_topology)")
    out = {}
    for t in vocab.TOPOLOGIES:
        p = topology_probs[t]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or math.isnan(p) \
                or not 0.0 <= p <= 1.0:
            raise ValueError(f"probability {p!r} for {t} must be a number in [0, 1]")
        out[t] = float(p)
    return out


def write_verdict(run_id: str, topology_probs: Mapping[str, float], *, chooser: str,
                  bound: Optional[str] = None, out_dir: Path | str = "outputs") -> dict:
    """Append one verdict row for `run_id` and return its verdict payload.

    The chooser and bound are checked first (`ValueError`, nothing read or written). The target
    is the LAST row with that `run_id` that is not itself a verdict, so a claim never copies
    another claim; none is `LookupError`, and nothing is written.
    """
    resolved = _resolve(chooser, bound)
    probs = _check_probs(topology_probs)
    found = None
    for row in store.read(out_dir):
        if row.get("run_id") == run_id and score.verdict_of(row) is None:
            found = row
    if found is None:
        raise LookupError(f"no run {run_id} in {store.index_path(out_dir)}")
    payload = {
        "topology_probs": probs,
        "chooser": chooser,
        "bound": resolved,
        "legal_id": legal_id(vocab.TOPOLOGIES),
        "model": chooser[len("llm:"):] if chooser.startswith("llm:") else None,
    }
    results = {key: found.get(key) for key in _COPIED}
    results["at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    store.append(results, out_dir=out_dir, extra={"extra": {"verdict": payload}})
    return payload
