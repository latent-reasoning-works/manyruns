"""The stage vocabulary, and the handoff that is no longer a call — `manyruns/modes.py`.

THIS FILE USED TO TEST A SEAM THAT NO LONGER EXISTS. It pinned the mapping between
manyruns's STAGE vocabulary (infer / eval / train) and the learner's ENGINE mode vocabulary
(rl / learn / sequence / trace / test), because collapsing the two is how `_train` came to
send `mode="train"` to an engine that has no such mode.

There is now no mapping to get wrong, because there is no track. Per the environment
contract (§3.5), manyruns holds NO reference to a learner: the learner observes manyruns's
artifacts and drives it through the CLI, never the reverse. `manyruns/train.py` and
`harness/trainer.py` are deleted, and `_train` plans rather than delegates.

So what is under test here is the property that replaced the mapping: the train STAGE
survives — the workflow would lie about its own shape without it — and it reaches the store
without importing, naming or invoking a learner.
"""
from __future__ import annotations

import sys

import pytest

from manyruns import modes
from public_safety import forbidden_count


def test_the_stage_vocabulary_is_unchanged_by_the_removal():
    """The three stages are manyruns's own, and were never the learner's. Losing one when the
    track was cut would mean the track had been load-bearing for the product's vocabulary,
    which is the coupling §3.5 says must not exist."""
    assert set(modes.MODES) == {"infer", "eval", "train"}


def test_train_plans_and_stops_at_the_store():
    """`run -> label -> store` is the whole harness workflow. Training is the learner's, and
    the handoff is a PATH rather than a call — which is the point: a path can be read by a
    process manyruns knows nothing about."""
    plan = modes.run("train", {"store": "outputs/run7"})

    assert plan["mode"] == "train"
    assert plan["store"] == "outputs/run7"
    assert "planned" in plan["status"]
    assert "not manyruns's to run" in plan["delegate"]


def test_the_plan_names_no_engine_mode_because_there_is_no_engine_to_name():
    """The old plan carried `engine_mode` (`rl` / `learn`) — manyruns choosing an
    intervention inside a system it is not supposed to know about. A key reappearing here is
    the track growing back."""
    plan = modes.run("train", {})

    assert "engine_mode" not in plan
    forbidden = forbidden_count(str(plan))
    assert forbidden == 0


def test_planning_a_train_imports_nothing_from_the_learner():
    """The acceptance criterion, executed rather than grepped. `_train` used to
    `import manyruns.train`, which imported the learner's `api` — so asking for a plan on a
    machine WITH the engine installed reached into it. Nothing may appear in `sys.modules`
    that was not there before."""
    before = sum(forbidden_count(m) for m in sys.modules)

    modes.run("train", {})

    assert sum(forbidden_count(m) for m in sys.modules) == before


def test_the_deleted_module_is_actually_gone():
    """`manyruns/train.py` was a passthrough to the learner's `api.run`. It is deleted rather
    than emptied, so that an import of it fails loudly instead of a caller finding a stub
    that silently does nothing."""
    with pytest.raises(ImportError):
        from manyruns import train  # noqa: F401
