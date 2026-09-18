"""The inspection cache — `manyruns/inspected.py` and the roster loop that reads it.

REPORTED FROM A REAL LAUNCH: "stuck after the very first screen … they didn't show up so it was
likely scanning the data folder slowly." The roster opens a file per dropped row
(`narrate.read_data`) and did it on EVERY launch, because nothing was written down. One file is a
blink; a real drop folder is not one file, and the screen a scientist meets first is the one that
pays for all of them before it can draw a row.

Measured on this checkout, one dropped file, separate processes: **435 ms → 46 ms.** The
multiplier grows with the folder.

WHAT IS ACTUALLY UNDER TEST is not the speed — it is that the cache may only ever change a
DURATION. A cache that can change an answer is worse than no cache, so most of this file is
about the ways a stale or broken one must fail: to a read, never to a wrong row.
"""
from __future__ import annotations

import json

import pytest

from manyruns import inspected  # noqa: E402
from manyruns.narrate import Observation  # noqa: E402


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    """Never touch the developer's real `~/.manyruns` from a test."""
    monkeypatch.setattr(inspected, "HOME", tmp_path / ".manyruns")


def _file(tmp_path, name="d.h5ad", content=b"x"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


OBS = Observation(shape="case-control", modality="scrna", conditions=["healthy", "disease"],
                  n_obs=2700, n_vars=32738, source="d.h5ad")


# ── the round trip ───────────────────────────────────────────────────────────
def test_an_inspection_comes_back_exactly_as_it_went_in(tmp_path):
    """Field for field. `conditions` is the one that matters most — `vocab.unmet` keys
    `contrast`'s legality on it, so a cache that lost it would make the ledger refuse a recipe
    that should run."""
    p = _file(tmp_path)
    inspected.put(p, inspected.as_row("scrna", OBS))

    back = inspected.to_observation(inspected.get(p), "d.h5ad")

    assert back == OBS


def test_every_field_is_carried_not_a_hand_picked_few():
    """Read off `dataclasses.fields`, so a field added to `Observation` tomorrow is stored
    without anyone remembering to come here. A hand-list drops what gets added next, silently,
    and what it drops is something some caller keys a decision on."""
    import dataclasses

    row = inspected.as_row("scrna", OBS)

    assert set(row["obs"]) == {f.name for f in dataclasses.fields(Observation)}


# ── the ways it must fail: to a read, never to a wrong row ───────────────────
def test_a_file_replaced_under_the_same_name_is_a_miss(tmp_path):
    """THE failure a path-keyed cache has, and it is silent: the roster would describe the old
    contents of a name that now holds new data. The key is `(path, size, mtime_ns)`."""
    p = _file(tmp_path, content=b"one")
    inspected.put(p, inspected.as_row("scrna", OBS))
    assert inspected.get(p) is not None

    p.write_bytes(b"two-different-bytes")

    assert inspected.get(p) is None


def test_a_corrupt_cache_reads_as_empty_rather_than_raising(tmp_path):
    """A half-written file from a killed process must cost one read, not the landing screen."""
    p = _file(tmp_path)
    inspected.put(p, inspected.as_row("scrna", OBS))
    inspected.path().write_text("{not json")

    assert inspected.get(p) is None


def test_an_older_schema_is_ignored_wholesale(tmp_path):
    """Not migrated. The cost of being wrong is a wrong answer on the landing screen; the cost
    of ignoring an old file is one read."""
    p = _file(tmp_path)
    inspected.put(p, inspected.as_row("scrna", OBS))
    raw = json.loads(inspected.path().read_text())
    raw["version"] = inspected.VERSION - 1
    inspected.path().write_text(json.dumps(raw))

    assert inspected.get(p) is None


def test_a_row_missing_a_field_rebuilds_as_nothing(tmp_path):
    """A half-restored Observation is a wrong answer wearing the shape of a right one. The
    caller's fallback — read the file — is cheap and correct."""
    row = inspected.as_row("scrna", OBS)
    del row["obs"]["conditions"]

    assert inspected.to_observation(row, "d.h5ad") is None
    assert inspected.to_observation({}, "d.h5ad") is None


def test_a_file_that_cannot_be_stat_ed_is_a_miss(tmp_path):
    assert inspected.key(tmp_path / "never-existed.h5ad") is None
    assert inspected.get(tmp_path / "never-existed.h5ad") is None


def test_an_unwritable_home_costs_speed_and_not_the_screen(tmp_path, monkeypatch):
    """A read-only home directory must not be why the front door fails to open."""
    blocked = tmp_path / "ro"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setattr(inspected, "HOME", blocked / "nested")
    try:
        inspected.put(_file(tmp_path), inspected.as_row("scrna", OBS))   # must not raise
    finally:
        blocked.chmod(0o700)


def test_the_cache_does_not_grow_without_bound(tmp_path, monkeypatch):
    """A folder someone points at once must not live in here forever. Oldest out."""
    monkeypatch.setattr(inspected, "MAX_ROWS", 3)
    for i in range(6):
        inspected.put(_file(tmp_path, f"f{i}.h5ad", content=str(i).encode()),
                      inspected.as_row("scrna", OBS))

    rows = json.loads(inspected.path().read_text())["rows"]
    assert len(rows) == 3
    assert all("f5.h5ad" in k or "f4.h5ad" in k or "f3.h5ad" in k for k in rows)


def test_forget_is_the_escape_hatch(tmp_path):
    """Why there is no `--no-cache` flag: a reader who suspects it can drop it."""
    p = _file(tmp_path)
    inspected.put(p, inspected.as_row("scrna", OBS))

    inspected.forget()

    assert inspected.get(p) is None
    inspected.forget()                       # and twice is not an error


# ── the roster, which is the only caller ─────────────────────────────────────
def test_the_roster_says_the_same_thing_cached_and_uncached(tmp_path, monkeypatch):
    """THE load-bearing property: the cache may change a duration and nothing else.

    Asserted against the real drop folder rather than a fixture, because the thing that would
    break it is a field of a REAL Observation that `as_row` failed to carry."""
    pytest.importorskip("anndata")
    from manyruns.tui import state

    inspected.forget()
    fresh = {e.name: e.obs for e in state.roster()}      # populates the cache
    cached = {e.name: e.obs for e in state.roster()}     # reads it

    assert set(cached) == set(fresh)
    assert all(cached[k] == fresh[k] for k in fresh)


def test_a_dropped_file_is_read_once_and_then_not_again(tmp_path, monkeypatch):
    """The point, counted rather than timed: a second roster must not re-open the file.

    THE DROP FOLDER IS BUILT HERE, and it did not used to be. This test read the AMBIENT folder
    — whatever `shell.data_dir()` resolved to on the machine running it — so it passed on a
    developer's checkout with files in `./data` and asserted `first > 0` against an empty one
    anywhere else. It never showed up because `importorskip("anndata")` skipped it in CI, and it
    surfaced the moment anndata joined the base job. A test whose fixture is "whatever happens to
    be lying around" proves nothing on the machine that has nothing, which is its own assertion
    message saying so.

    `$MANYRUNS_DATA_DIR` is the documented override (`shell.data_dir`), so the folder is ours
    and the count is deterministic. A `.csv` rather than an `.h5ad` because `_DATA_SUFFIXES`
    takes both and this needs no reader to be installed to be a droppable FILE.
    """
    pytest.importorskip("anndata")
    from manyruns import shell
    from manyruns.tui import state

    drop = tmp_path / "drop"
    drop.mkdir()
    (drop / "dropped.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(drop))

    reads: list = []
    real = shell._resolve_path_source
    monkeypatch.setattr(shell, "_resolve_path_source",
                        lambda p, c: reads.append(p) or real(p, c))

    inspected.forget()
    state.roster()
    first = len(reads)
    state.roster()

    assert first > 0, "the fixture drop folder has nothing in it; this test proves nothing"
    assert len(reads) == first, "the second launch re-opened a file it had already inspected"


def test_a_bundled_dataset_costs_no_read_at_all():
    """The declaration IS the alias. A dataset that declares its shape needs no file opened, and
    `synthetic_timecourse` — whose `path` ref is the generator `synthetic:time-course`, not a
    file — is the case that proves it: it used to infer `unknown` against a declared
    `time-course`, which cost it the `cflows` recommendation it exists for."""
    from manyruns import shell

    _, _, _, obs = shell._resolve_dataset("synthetic_timecourse")

    assert obs.shape == "time-course"
