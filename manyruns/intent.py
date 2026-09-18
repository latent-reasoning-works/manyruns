"""The intent layer — turn a free-typed sentence into one move.

Three tiers, and only the top one touches an LLM:
  A. rule-based (default, in the base install): keyword matching + `narrate`'s refusal/interpret.
  B. one Anthropic call (optional, behind the `[agents]` extra): a single forced-tool pick from a
     fixed set of legal choices. NOT an agent loop — the shell owns the loop; the model only
     chooses. Import-guarded and exception-guarded, so a missing SDK / key / network just falls
     back to Tier A and the shell keeps working.

The Anthropic import lives *inside* `_llm_choose`, so importing this module — and therefore the
whole base app / binary — never imports the SDK. That is the answer to "do we need a full SDK?":
no. The interactive harness runs entirely on Tier A; the SDK is one optional, swappable picker.
"""
from __future__ import annotations

from typing import Optional

from manyruns import narrate
from manyruns.narrate import Observation


def route_recipe(
    text: str, obs: Observation, legal: list[str], *, allow_llm: bool = False
) -> tuple[Optional[str], Optional[str]]:
    """Map a free-typed request to one recipe name from `legal`.

    Returns (recipe, refusal). If the request asks for something the data can't honestly
    support (a trajectory with no time axis), `refusal` is a plain-English reframe and recipe
    is None — the app declines rather than inventing a story. Otherwise recipe is the routed
    name (or None if nothing matched, so the caller falls back to the recommended default).
    """
    refuse = narrate.refusal(text, obs)
    if refuse:
        return None, refuse
    rec = narrate.interpret(text, obs)                      # Tier A: rule-based
    if rec is not None and rec in legal:
        return rec, None
    if allow_llm:                                           # Tier B: one LLM call, optional
        picked = _llm_choose(
            system=(
                "You route a user's plain-English data-analysis request to exactly one geometric "
                "recipe. Choose only from the given options; if none clearly fits, choose the first."
            ),
            prompt=(f"Data shape: {obs.shape}. Groups: {obs.conditions}. "
                    f"Time points: {obs.n_timepoints}.\n\nRequest: {text}"),
            choices=legal,
            key="recipe",
        )
        if picked in legal:
            return picked, None
    return None, None


def route_verb(text: str, *, allow_llm: bool = False) -> Optional[str]:
    """Map free text at the home prompt to a verb name (`manyruns.verbs`), or None."""
    from manyruns import verbs

    t = (text or "").lower().strip()
    for kw, verb in (
        ("check", "check"), ("valid", "check"),
        ("recipe", "recipes"), ("workflow", "recipes"),
        ("dataset", "datasets"), ("data", "datasets"),
        ("quit", "quit"), ("exit", "quit"),
        ("explore", "explore"), ("run", "explore"), ("look", "explore"),
    ):
        if kw in t:
            return verb
    if allow_llm:
        picked = _llm_choose(
            system="Route the user's request to exactly one action. Choose only from the options.",
            prompt=text,
            choices=[v.name for v in verbs.VERBS],
            key="action",
        )
        if picked in verbs.names():
            return picked
    return None


def llm_available() -> bool:
    """Is the optional model tier usable? — `agents.available`, re-exported for this surface.

    ONE PREDICATE, because there were two and they disagreed. This gated on the raw SDK while the
    seam prefers manyAgents, so with a key and nothing installed the pair answered True and False
    to the same question — and this is the one the product actually reads (`shell.py:560`,
    `tui/search.py:316`, `tui/find.py:233`), so the seam's own predicate had no callers at all.

    Kept as a name rather than deleted: three call sites read it, it is the vocabulary those
    surfaces already speak, and moving them is a rename with no behaviour in it.
    """
    from manyruns import agents

    return agents.available()


def _llm_choose(*, system: str, prompt: str, choices: list[str], key: str) -> Optional[str]:
    """One forced tool call returning exactly one of `choices`.

    CONSOLIDATED INTO `manyruns.agents`, which is now the one place this product asks a model
    anything. This held manyruns's only LLM call until `tune` needed a second one, and two direct
    SDK calls would have meant two sets of guards, two model names, and two places to add a
    provider. `agents.choose_one` is this call with manyAgents' `ClaudeAdapter` preferred over the
    raw SDK, and the raw SDK kept as the fallback so an install that has only the SDK — which
    worked before — still does.

    Everything remains optional and guarded: a missing package, key or network, or a reply outside
    the choice set, all resolve to None and the caller falls back to the rule-based tier.
    """
    from manyruns import agents

    return agents.choose_one(system=system, prompt=prompt, key=key, choices=choices)
    return None
