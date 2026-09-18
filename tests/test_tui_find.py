"""`find data…` and the fetch confirm — component 5 of the TUI rewrite.

TWO HALVES, TESTED THE TWO WAYS THEY ARE BUILT. `manyruns.tui.search` is dependency-free (no
Textual, no rich, no network unless you hand it a URL), so the ranking, the size probe, the
progress throttle and the fetch plan are exercised as plain functions. `manyruns.tui.find` is
driven through `App.run_test()` with a `Pilot`, which §5 of the spec picks over the injected
prompter for the app surface.

THE FOUR PROPERTIES WORTH THE MOST HERE, in the order they can hurt someone:

    the confirm states BOTH sizes, and neither is ever estimated from the other
    enter on arrival at the confirm does NOT start an 100 MB download
    a cancelled fetch says where its partial bytes are and never claims a rollback
    a synthetic is a peer in the ranking, choosing it touches no network at all

NOTHING HERE DOWNLOADS ANYTHING. The two network calls the product makes (the HEAD probe and
`urlretrieve`) are injected or monkeypatched in every test. All catalogue rows and
size measurements are synthetic fixtures. The screen half requires Textual.
"""
from __future__ import annotations

import ast
import asyncio
import functools
import inspect
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from manyruns.harness import data_acquire as da
from manyruns.narrate import Observation
from manyruns.tui import search
from manyruns.tui.search import Candidate, DownloadTicks, FetchPlan, Probe
from manyruns.tui.state import DataEntry

# A generic query against invented metadata, including a common abbreviation.
EXAMPLE_QUERY = "sample healthy controls"


@pytest.fixture(autouse=True)
def configured_manifest(monkeypatch, synthetic_manifest):
    monkeypatch.setenv("MANYRUNS_ACQUIRE_MANIFEST", str(synthetic_manifest))


def _entry(name: str = "tree_wide", shape: str = "manifold", **kw) -> DataEntry:
    obs = Observation(shape=shape, modality=kw.pop("modality", "synthetic"), source=name)
    return DataEntry(name=name, kind=kw.pop("kind", "bundled"), obs=obs, **kw)


def _cohort(name: str = "GSE999999", **kw) -> Candidate:
    return Candidate(kind="cohort", name=name, accession=name,
                     summary=kw.pop("summary", "material · control · 4 samples"),
                     url=kw.pop("url", f"https://example.invalid/{name}"), **kw)


def test_the_search_half_pulls_no_terminal_and_no_sdk():
    """`manyruns.tui.search` is the dependency-free half, and this is what makes that a fact
    rather than an intention.

    In a SUBPROCESS, because this file imports `textual` forty lines below and an in-process
    `sys.modules` check would pass vacuously depending on test order. `anthropic` is in the list
    for the same reason `intent`'s import guard exists: tier 3 must cost nothing when it is not
    used, and an SDK imported at module scope is a cost every launch pays.
    """
    import subprocess

    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; import manyruns.tui.search; "
         "print('textual' in sys.modules, 'rich' in sys.modules, 'anthropic' in sys.modules)"],
        capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False", "False"], out.stdout


# ══ tier 1 · the manifest search ═════════════════════════════════════════════
def test_the_spec_example_query_puts_the_right_cohort_first():
    """The generic control aliases and a typed stem rank the synthetic row first."""
    found = search.search(EXAMPLE_QUERY, datasets=[])
    assert found[0].name == "GSE000001"
    assert set(found[0].matched) == {"sample", "healthy", "controls"}
    assert found[1].score < found[0].score


def test_every_alias_target_is_a_word_that_is_actually_in_the_manifest():
    """The synthetic fixture exercises every supported control abbreviation."""
    text = " ".join(" ".join([e.accession, e.source, e.modality, e.fmt, *e.meta.values()])
                    for e in da.load_manifest())
    for typed, targets in search.ALIASES.items():
        for target in targets:
            assert search._word_in(target, text), f"{typed} → {target!r} is not in the manifest"


def test_an_abbreviation_matches_as_a_word_and_not_inside_one():
    """Aliases match whole words; typed stems can match longer descriptions."""
    assert search.hits(("healthy",), "benchcraft") == ()
    assert search.hits(("healthy",), "HC/control sample") == ("healthy",)
    assert search.hits(("sample",), "samples") == ("sample",)


def test_a_query_that_matches_nothing_still_returns_the_whole_catalogue():
    """A finder that empties itself is a dead end at the moment you most need it.

    Same argument as the ledger's hollow rows: what is missing from a list teaches nothing, and
    the note on a row that missed your words is often what tells you which word to type.
    """
    found = search.search("xenon", datasets=[_entry()])
    assert [c.score for c in found] == [0] * len(found)
    assert len(found) == len(search.cohorts()) + 1


def test_an_empty_query_shows_the_cohorts_first_because_the_synthetics_are_already_home():
    """The tie-break, and it only ever applies within one score.

    Every synthetic in this list is already a row on the roster screen the user pressed `?` to
    leave; a cohort is what this screen exists to reach. `test_a_synthetic_outranks_…` pins the
    other half: the tie-break never overrules a score.
    """
    found = search.search("", datasets=[_entry()])
    kinds = [c.kind for c in found]
    assert kinds == sorted(kinds, key=lambda k: {"cohort": 0, "synthetic": 1}[k])
    assert [c.name for c in found if c.kind == "cohort"] == \
           [e.accession for e in da.load_manifest()]        # manifest order survives


# ══ tier 2 · make one instead ════════════════════════════════════════════════
def test_a_synthetic_outranks_a_cohort_that_answers_fewer_words():
    """PEER, NOT FALLBACK — the brief's word. A generated dataset that answers two of your words
    is above a public cohort that answers one, in one list, with no section headings."""
    found = search.search("a branching tree",
                          datasets=[_entry("tree_wide", topology=("multi-branching",))])
    assert found[0].name == "tree_wide"
    assert found[0].kind == "synthetic"
    assert found[0].score > found[1].score


def test_a_synthetic_carries_the_rosters_own_row_and_not_a_copy_of_it():
    """What the finder hands back must be indistinguishable from what the roster hands back.

    `entry.as_source()` is `shell._run`'s `(data_folder, dataset, modality, obs)` tuple, so a
    second convention here would mean the app has two ways to start a run depending on which
    screen chose the data — which is exactly what `as_source()` was shaped to prevent.
    """
    row = _entry("swissroll", dataset="swissroll")
    candidate = search.synthetics([row])[0]
    assert candidate.entry is row
    assert candidate.entry.as_source() == row.as_source()


def test_the_bundled_generators_really_do_reach_the_pool():
    """The one test that reads the real catalogue, so the injected rows above cannot all be
    lying together. Uses an empty drop folder so it does not depend on this checkout's `data/`."""
    import os

    previous = os.environ.get("MANYRUNS_DATA_DIR")
    os.environ["MANYRUNS_DATA_DIR"] = "/nonexistent-drop-folder-for-this-test"
    try:
        pool = search.pool()
    finally:
        if previous is None:
            del os.environ["MANYRUNS_DATA_DIR"]
        else:
            os.environ["MANYRUNS_DATA_DIR"] = previous
    from manyruns.catalog import discover_datasets

    synthetic = {c.name for c in pool if c.kind == "synthetic"}
    assert synthetic == set(discover_datasets())
    assert {c.name for c in pool if c.kind == "cohort"} == {e.accession for e in
                                                            da.load_manifest()}


# ══ what cannot be fetched, and why it is still on screen ════════════════════
def test_the_unfetchable_sources_are_the_ones_data_acquire_itself_refuses():
    """`UNFETCHABLE`'s keys are READ from `data_acquire.fetch`, not decided here.

    That function raises `NotImplementedError` for exactly two `source` values; a third one
    added there without a phrase here would put a row on screen that looks fetchable and dies
    the moment it is chosen. This parses its source so that cannot happen quietly.
    """
    tree = ast.parse(inspect.getsource(da.fetch).lstrip())
    refused = {
        node.test.comparators[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "source"
        and any(isinstance(b, ast.Raise) for b in node.body)
    }
    assert refused == set(search.UNFETCHABLE), (
        f"data_acquire.fetch refuses {refused}, search.UNFETCHABLE phrases "
        f"{set(search.UNFETCHABLE)}")


def test_a_cohort_that_cannot_be_fetched_is_on_screen_and_unavailable():
    """The ledger's hollow row, on this screen. The fixture includes unsupported source types."""
    hollow = [c for c in search.cohorts() if not c.available]
    assert {c.name for c in hollow} == {"CELLxGENE-TEST", "E-MTAB-0001", "E-MTAB-0002"}
    assert all(c.cannot_fetch for c in hollow)
    assert all(c.available for c in search.cohorts() if c.name.startswith("GSE"))


# ══ the size probe ═══════════════════════════════════════════════════════════
class _Response:
    def __init__(self, headers: dict, status: int = 200) -> None:
        self.headers, self.status = headers, status

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_the_probe_reports_the_servers_number_and_nothing_else():
    """Display the supplied byte count using binary units."""
    probe = search.probe("https://example.invalid/x",
                         opener=lambda *a, **k: _Response({"content-length": "104857600"}))
    assert probe.nbytes == 104857600
    assert probe.size == "100.0 MB"          # binary units, `state._human_bytes`, one home


def test_a_server_with_no_content_length_says_unknown_and_never_estimates():
    """THE RULE, and the reason it has its own test: where the size is unknown, say so. An
    estimate scaled from another cohort would be the one number the confirm exists to state,
    invented."""
    probe = search.probe("https://example.invalid/x",
                         opener=lambda *a, **k: _Response({}))
    assert probe.nbytes is None
    assert probe.size == "size unknown"
    assert "GB" not in probe.size and "MB" not in probe.size


def test_a_missing_archive_is_a_404_and_the_screen_can_tell():
    """Measured against GEO: a nonexistent accession answers 404 in 0.27 s. It is the one probe
    result that should stop a fetch rather than merely fail to describe one."""
    def boom(*a, **k):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    probe = search.probe("https://example.invalid/x", opener=boom)
    assert probe.missing and probe.status == 404
    assert "404" in (probe.error or "")


def test_the_probe_is_never_the_reason_a_screen_dies():
    def boom(*a, **k):
        raise OSError("network is down")

    probe = search.probe("https://example.invalid/x", opener=boom)
    assert probe.nbytes is None and probe.size == "size unknown"
    assert "network is down" in (probe.error or "")


# ══ the plan: both sizes, and where the bytes land ═══════════════════════════
def test_the_plan_states_both_sizes_and_neither_is_derived_from_the_other(tmp_path):
    """Probe bytes and recorded disk usage are independent synthetic measurements."""
    recorded = [c for c in search.cohorts() if c.name == "GSE000001"][0]
    plan = search.plan(recorded, out_dir=tmp_path)
    assert plan.disk == "~2 GB"
    assert plan.download == "size unknown"          # not probed yet — and not guessed from disk

    from dataclasses import replace

    probed = replace(plan, probe=Probe(nbytes=104857600, status=200))
    assert probed.download == "100.0 MB" and probed.disk == "~2 GB"


def test_an_unmeasured_cohort_admits_the_disk_size_is_unknown(tmp_path):
    """Without a recorded disk measurement the plan must say it is unknown."""
    plan = search.plan(_cohort(), out_dir=tmp_path)
    assert "unknown until it runs" in plan.disk
    assert not any(unit in plan.disk for unit in (" GB", " MB"))


def test_the_fetch_lands_where_the_roster_looks(tmp_path, monkeypatch):
    """THE INTEGRATION, in one line: `out_dir` defaults to the drop folder, so a fetched `.h5ad`
    is a roster row the next time the app opens — no registration, no second catalogue."""
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path))
    plan = search.plan(_cohort("GSE000001"))
    from manyruns import shell

    assert plan.out_path == shell.data_dir() / "GSE000001.h5ad"


def test_the_work_dir_is_hidden_so_a_partial_download_is_never_a_roster_row(tmp_path):
    """100 MB of tar and an extracted tree land beside the `.h5ad`, and neither is data.

    `shell._scan_drop_folder` skips any entry whose name starts with a dot — read, not assumed,
    and re-asserted here against the real scanner rather than against the claim.
    """
    from manyruns import shell

    plan = search.plan(_cohort("GSE000001"), out_dir=tmp_path)
    plan.work_dir.mkdir(parents=True)
    plan.partial.write_bytes(b"half a tar")
    (tmp_path / "real.h5ad").write_bytes(b"")

    assert plan.partial.exists()
    assert [p.name for p in shell.drop_folder_entries(tmp_path)] == ["real.h5ad"]


def test_a_file_that_is_already_here_is_not_downloaded_again(tmp_path):
    """`data_acquire.convert` returns the existing path untouched unless `force=True`, so this
    is not advice — it is what the fetch would do."""
    (tmp_path / "GSE000001.h5ad").write_bytes(b"")
    assert search.plan(_cohort("GSE000001"), out_dir=tmp_path).already_here
    assert not search.plan(_cohort("GSE000018"), out_dir=tmp_path).already_here


# ══ the progress throttle ════════════════════════════════════════════════════
def test_a_large_download_repaints_once_per_percent():
    """The claim in `DownloadTicks`' docstring, measured rather than asserted from arithmetic.

    `urlretrieve` reads in 8 KB blocks (`bs = 1024*8` in CPython's `urllib/request.py`) and
    calls its hook once before the first read and once per block after it. Driving it exactly
    that way over a synthetic 104,857,600-byte length: 12,801 calls in, 101 reports out.
    """
    total, block = 104857600, 8192
    ticks = DownloadTicks()
    reports, calls, read, blocknum = [], 1, 0, 0
    first = ticks(0, block, total)
    if first:
        reports.append(first)
    while read < total:
        read, blocknum, calls = min(read + block, total), blocknum + 1, calls + 1
        tick = ticks(blocknum, block, total)
        if tick:
            reports.append(tick)

    assert calls == 12801 and len(reports) == 101
    assert reports[0] == ("download", 0, total)
    # The LAST tick hands over to extract rather than reading 100%: everything after the final
    # block is `tar.extractall`, which reports nothing, and a full silent bar is the exact
    # failure the brief names as the worst version of this screen.
    assert reports[-1] == ("extract", total, total)
    assert all(r[0] == "download" for r in reports[:-1])


def test_a_server_that_reports_no_length_still_moves_the_number():
    """`urlretrieve` passes `total = -1` when there is no `content-length`; there is no percent
    to change, so the throttle falls back to one report per ~4 MB."""
    ticks = DownloadTicks()
    reports = [ticks(i, 8192, -1) for i in range(2048)]
    kept = [r for r in reports if r is not None]
    assert len(kept) == 4 and all(r[0] == "download" for r in kept)
    assert kept[1][1] == 512 * 8192


# ══ tier 3 · ask a model ═════════════════════════════════════════════════════
def test_tier_three_is_silent_with_no_key(monkeypatch):
    """No SDK or no key means tiers 1 and 2 are the whole product, unchanged. The guard is
    `intent.llm_available()`, which checks both without importing the SDK."""
    from manyruns import intent

    monkeypatch.setattr(intent, "llm_available", lambda: False)
    monkeypatch.setattr(intent, "_llm_choose",
                        lambda **kw: pytest.fail("called with no key available"))
    assert search.ask_a_model("anything", search.cohorts()) is None


def test_tier_three_can_only_pick_a_cohort_that_is_already_on_screen(monkeypatch):
    """It CHOOSES; it does not name. A model free-typing an accession would hand
    `data_acquire.fetch` a GEO URL for a dataset that does not exist — the manifest miss falls
    back to `geo_supplementary_url` — and the failure would arrive as a 404 after a confirm that
    had already quoted "size unknown" as though it were a fact about something real.
    """
    from manyruns import intent

    monkeypatch.setattr(intent, "llm_available", lambda: True)
    monkeypatch.setattr(intent, "_llm_choose", lambda **kw: "GSE9999-invented")
    assert search.ask_a_model("measurements from a sample", search.cohorts()) is None

    monkeypatch.setattr(intent, "_llm_choose", lambda **kw: "GSE000003")
    picked = search.ask_a_model("measurements from a sample", search.cohorts())
    assert picked is not None and picked.name == "GSE000003"
    assert picked.by_model, "provenance has to travel with the row"


def test_tier_three_is_never_offered_an_unfetchable_row(monkeypatch):
    """Choosing one would be a suggestion the product cannot act on."""
    from manyruns import intent

    seen = {}
    monkeypatch.setattr(intent, "llm_available", lambda: True)
    monkeypatch.setattr(intent, "_llm_choose",
                        lambda **kw: seen.setdefault("choices", kw["choices"]) and None)
    search.ask_a_model("sample", search.cohorts())
    assert "CELLxGENE-TEST" not in seen["choices"]
    assert "GSE000001" in seen["choices"]


# ══ serialisation ════════════════════════════════════════════════════════════
def test_every_view_serialises_to_plain_json(tmp_path):
    """Same contract component 1 set: `to_dict()` is pure JSON, so a test can assert on the
    whole structure and a debug dump costs nothing."""
    plan = search.plan(_cohort("GSE000001"), out_dir=tmp_path)
    for view in (search.cohorts()[0], search.synthetics([_entry()])[0],
                 Probe(nbytes=1, status=200), plan):
        assert json.loads(json.dumps(view.to_dict())) == view.to_dict()


# ══════════════════════════════════════════════════════════════════════════════
# The screens
# ══════════════════════════════════════════════════════════════════════════════
pytest.importorskip("textual")

from textual.app import App  # noqa: E402
from textual.widgets import Button, Input, OptionList  # noqa: E402

from manyruns.tui.app import ManyrunsApp  # noqa: E402
from manyruns.tui.find import BLOCKED_MARK, FETCH_MARK, LOCAL_MARK  # noqa: E402
from manyruns.tui.find import FetchScreen, FindScreen  # noqa: E402


@pytest.fixture(autouse=True)
def _rich_is_whole():
    """Repair a `rich` that an earlier test dismantled. NOT this component's bug — component 2
    and component 3 both carry this same fixture and both name the cause:
    `tests/test_shell.py::test_base_import_chain_pulls_no_sdk_or_tui` pops `sys.modules["rich"]`
    and leaves the ~40 `rich.*` submodules behind, so the next `import rich` builds a fresh
    package object with none of them attached and `rich.repr.auto` — which Textual uses
    pervasively — raises `AttributeError`. The real fix is three lines in `test_shell.py`, which
    is not this component's file.
    """
    import rich

    package = sys.modules["rich"]
    for name, module in list(sys.modules.items()):
        head, _, tail = name.partition(".")
        if head == "rich" and tail and "." not in tail:
            if getattr(package, tail, None) is not module:
                setattr(package, tail, module)
    assert rich is package


def astest(fn):
    """Run one `async def` test to completion. There is no `pytest-asyncio` in this repo and
    this component does not add one — components 2 and 3 both carry this same four-line
    decorator. `functools.wraps` keeps pytest reporting the real name AND injecting fixtures;
    without the decorator an `async def test_` is collected and SKIPPED, which is a green file
    that ran nothing."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


def _screen_text(app: App) -> str:
    """What is composited — what a scientist would actually see.

    `_compositor.render_strips()` is private to Textual and used deliberately, for component
    3's reason: a widget's `renderable` is what it was HANDED, and the question these tests ask
    is what came out the other end of the layout engine.
    """
    return "\n".join(strip.text for strip in app.screen._compositor.render_strips())


def _flat(app: App) -> str:
    """The composited screen with its line breaks collapsed.

    For asserting on a SENTENCE. Textual wraps on spaces, so a phrase that fits at one width is
    split at another — "…it does not / resume." at 100 columns — and a test that asserts the
    unwrapped phrase is measuring the terminal, not the screen. The panel's vertical border sits
    between the two halves of every wrapped line, so it comes out too: it is chrome, and the
    question here is what the screen SAYS. What is still being asserted is that the words were
    composited — a widget that never rendered contributes nothing to this.
    """
    return " ".join(_screen_text(app).replace("│", " ").split())


CANDIDATES = [
    _cohort("GSE000001", summary="sample material · control · 2 samples",
            note="Synthetic HC sample fixture.", disk_after="~2 GB"),
    _cohort("E-MTAB-0002", summary="material · treated · 3 samples",
            cannot_fetch="the processed matrix has to be picked by hand at EBI first"),
    Candidate(kind="synthetic", name="tree_wide", summary="synthetic · a branching tree",
              note="generated here, with the structure known: multi-branching",
              entry=_entry("tree_wide", dataset="tree_wide")),
]


class FinderApp(ManyrunsApp):
    """The real app, landing on the finder with its rows injected.

    A `ManyrunsApp` subclass rather than a bare `App`, so the CSS the app actually ships is the
    CSS these screens are measured under — `stylesheet()` outranks a screen's `DEFAULT_CSS`, so
    a screen checked without it is checked under rules that will not be in force.
    """

    def __init__(self, candidates=None, out_dir=None) -> None:
        super().__init__()
        self._candidates = CANDIDATES if candidates is None else candidates
        self._out_dir = out_dir
        self.chosen: list = []

    def get_default_screen(self):
        return FindScreen(candidates=self._candidates, out_dir=self._out_dir)

class HostApp(ManyrunsApp):
    """A base screen to push a `FetchScreen` onto. `dismiss` refuses to pop the last screen off
    the stack, which is why every dismissing screen in this app has to be pushed."""

    def __init__(self, plan: FetchPlan) -> None:
        super().__init__()
        self.plan = plan
        self.result: list = []

    def get_default_screen(self):
        return FindScreen(candidates=[])

    def on_mount(self) -> None:
        self.push_screen(FetchScreen(self.plan, probe=False), callback=self.result.append)


# ── the finder ───────────────────────────────────────────────────────────────
@astest
async def test_the_finder_lists_cohorts_and_synthetics_in_one_list():
    """§"find data": a synthetic is listed WITH the search results, not underneath them. The
    marks say which is which so that "listed together" does not mean "indistinguishable"."""
    app = FinderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        text = _screen_text(app)
        assert "GSE000001" in text and "tree_wide" in text
        assert FETCH_MARK in text and LOCAL_MARK in text and BLOCKED_MARK in text
        assert "a branching tree" in text


@astest
async def test_typing_reranks_and_the_best_match_leads():
    app = FinderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.query_one(Input).value = "branching"
        await pilot.pause()
        assert app.screen.results[0].name == "tree_wide"
        assert app.screen.results[0].score == 1


@astest
async def test_enter_in_the_query_box_moves_to_the_list_and_does_not_choose():
    """One keystroke of distance between typing and a confirm screen whose next enter would
    start an 100 MB download. Pinned because deleting it looks like a convenience."""
    app = FinderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, FindScreen)      # nothing was pushed
        assert app.screen.focused is app.screen.query_one(OptionList)


@astest
async def test_choosing_a_synthetic_dismisses_with_the_rosters_row_and_touches_no_network(
        monkeypatch):
    """Tier 2 finishes in one keystroke: no confirm, no probe, no download.

    The probe is replaced with a landmine rather than merely not asserted on — "did not use the
    network" is only worth testing if using it would fail.
    """
    monkeypatch.setattr(search, "probe",
                        lambda *a, **k: pytest.fail("a synthetic must not touch the network"))
    app = FinderApp()
    got: list = []
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.dismiss = got.append          # the screen is the base screen here
        app.screen.query_one(Input).value = "branching"
        await pilot.pause()
        app.screen._chosen(app.screen.results[0])
        await pilot.pause()
    assert len(got) == 1 and isinstance(got[0], DataEntry)
    assert got[0].name == "tree_wide"
    assert got[0].as_source() == CANDIDATES[2].entry.as_source()


@astest
async def test_tab_gets_you_back_to_the_query_after_enter_moved_you_off_it():
    """`enter` in the query box moves to the list on purpose, so the way back has to exist and
    has to be advertised. It is Textual's own focus cycling rather than a rebind — this asserts
    the round trip, and `HINT` is what tells the user about it."""
    from manyruns.tui.find import HINT

    app = FinderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.focused is app.screen.query_one(OptionList)
        await pilot.press("tab")
        await pilot.pause()
        assert app.screen.focused is app.screen.query_one(Input)
        assert "tab" in HINT and HINT in _flat(app)


@astest
async def test_a_cohort_that_cannot_be_fetched_is_visible_and_unselectable():
    """The ledger's rule, on this screen: on screen to be READ. `OptionList` enforces it —
    `action_select` returns without posting when the option is disabled, and the highlight
    cannot land on it — so this asserts the disabled flag AND the reason being legible."""
    app = FinderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        options = app.screen.query_one(OptionList)
        blocked = [o for o in options._options if o.disabled]
        assert len(blocked) == 1
        assert "picked by hand at EBI" in _flat(app)


@astest
async def test_choosing_a_cohort_pushes_the_confirm(tmp_path):
    app = FinderApp(out_dir=tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        finder = app.screen
        finder.query_one(Input).value = "GSE000001"
        await pilot.pause()
        finder._chosen(finder.results[0])
        await pilot.pause()
        assert isinstance(app.screen, FetchScreen)
        assert app.screen.plan.accession == "GSE000001"


# ── the confirm ──────────────────────────────────────────────────────────────
def _plan(tmp_path, **kw) -> FetchPlan:
    plan = search.plan(CANDIDATES[0], out_dir=tmp_path)
    if kw:
        from dataclasses import replace

        plan = replace(plan, **kw)
    return plan


@astest
async def test_the_confirm_states_both_sizes(tmp_path):
    """Show both independent size measurements on the confirm screen."""
    app = HostApp(_plan(tmp_path, probe=Probe(nbytes=104857600, status=200)))
    async with app.run_test() as pilot:
        await pilot.pause()
        text = _screen_text(app)
        assert "download" in text and "100.0 MB" in text
        assert "on disk" in text and "2 GB" in text
        assert "after extract + convert" in text


@astest
async def test_an_unknown_download_size_says_unknown_on_screen(tmp_path):
    app = HostApp(_plan(tmp_path, probe=Probe(status=200)))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "size unknown" in _screen_text(app)


@astest
async def test_the_confirm_names_every_phase_that_follows_the_bar(tmp_path):
    """"Progress must name what follows the bar" — before the bar exists, so that a yes is
    informed rather than explained afterwards."""
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        text = _screen_text(app)
        for phase in search.PHASES:
            assert phase in text, f"{phase} is not on the confirm"


@astest
async def test_enter_on_arrival_does_not_start_a_download(tmp_path, monkeypatch):
    """THE ONE THAT MATTERS. This screen is reached by pressing enter on a list; the enter still
    under the finger must not land on "fetch it".

    Mutation-checked: swapping the two buttons in `compose` so that `fetch-go` takes focus first
    turns this red. The fetch worker is replaced with a landmine, so "did not start" is a
    failure and not an absence of evidence.
    """
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        confirm = app.screen
        monkeypatch.setattr(type(confirm), "start",
                            lambda self: pytest.fail("enter started a download"))
        assert confirm.focused is confirm.query_one("#fetch-cancel", Button)
        await pilot.press("enter")
        await pilot.pause()
    assert app.result == [None]           # it dismissed, having downloaded nothing


@astest
async def test_y_is_the_deliberate_yes(tmp_path, monkeypatch):
    """The other half of `test_enter_on_arrival_…`: there IS a one-key start, it is just not the
    key that was already being pressed."""
    started: list = []
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(type(app.screen), "start", lambda self: started.append(True))
        await pilot.press("y")
        await pilot.pause()
    assert started == [True]


@astest
async def test_the_finder_reads_the_real_catalogue_when_nothing_is_injected(tmp_path,
                                                                           monkeypatch):
    """The one screen test with no injected rows, so the injected ones above cannot all be
    agreeing with each other about a pool that never loads.

    `$MANYRUNS_DATA_DIR` points at an empty tmp folder: what is sitting in this checkout's
    `data/` is not this test's business, and reading a real `.h5ad` costs 361 ms.
    """
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path))

    class RealApp(ManyrunsApp):
        def get_default_screen(self):
            return FindScreen()

    app = RealApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        assert len(app.screen.pool) == len(search.cohorts()) + len(search.synthetics())
        # Type a query to exercise filtering in the real catalogue pool.
        app.screen.query_one(Input).value = EXAMPLE_QUERY
        await pilot.pause()
        assert "GSE000001" in _flat(app)
        assert app.screen.results[0].name == "GSE000001"


@astest
async def test_a_file_you_already_have_is_handed_back_without_a_download(tmp_path):
    """`convert` would return the existing path untouched; running four phases that each decide
    to do nothing would be a progress bar for a no-op."""
    (tmp_path / "GSE000001.h5ad").write_bytes(b"")
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert "you already have this" in _flat(app)
        app.screen.start()
        await pilot.pause()
    assert app.result == [tmp_path / "GSE000001.h5ad"]


# ── the fetch ────────────────────────────────────────────────────────────────
def _fake_acquire(monkeypatch, tmp_path, *, hook_calls=0, fail=None):
    """Stand in for `data_acquire`'s two impure calls. The REAL ones are wired in `find.py`;
    what is faked here is the network and scanpy, not the wiring."""
    def fetch(accession, work_dir, *, url=None, reporthook=None):
        for i in range(hook_calls):
            reporthook(i, 8192, 8192 * hook_calls)
        if fail:
            raise fail
        root = Path(work_dir) / accession
        root.mkdir(parents=True, exist_ok=True)
        return root

    def convert(src, accession, *, out_dir=None, **kw):
        out = Path(out_dir) / f"{accession}.h5ad"
        out.write_bytes(b"not really an h5ad")
        return out

    monkeypatch.setattr(da, "fetch", fetch)
    monkeypatch.setattr(da, "convert", convert)


@astest
async def test_a_fetch_that_lands_returns_the_path_and_walks_every_phase(tmp_path, monkeypatch):
    _fake_acquire(monkeypatch, tmp_path, hook_calls=300)
    monkeypatch.setattr(search, "roster_entry", lambda path, ddir=None: _entry("GSE000001.h5ad"))
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.start()
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()
    assert app.result == [tmp_path / "GSE000001.h5ad"]


@astest
async def test_a_download_that_does_not_resolve_as_data_is_reported_not_hidden(
        tmp_path, monkeypatch):
    """`state.roster` SKIPS a file it cannot resolve, silently — component 1 reports that as a
    known gap. After a 2 GB download is the one moment where a silent skip is unaffordable, so
    the verify step is the roster's own read and its failure is on screen."""
    _fake_acquire(monkeypatch, tmp_path)
    monkeypatch.setattr(search, "roster_entry", lambda path, ddir=None: None)
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.start()
        await app.workers.wait_for_complete()
        await pilot.pause()
        text = _flat(app)
        assert "does not resolve as data" in text
        assert app.screen.state == "failed"
    assert app.result == []              # still on screen: nothing was dismissed


@astest
async def test_a_failed_fetch_shows_the_reason_and_does_not_pretend_it_finished(
        tmp_path, monkeypatch):
    _fake_acquire(monkeypatch, tmp_path, fail=NotImplementedError("E-MTAB is ArrayExpress"))
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        app.screen.start()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "NotImplementedError" in _screen_text(app)
        assert app.screen.state == "failed"


@astest
async def test_a_cancelled_fetch_says_where_the_partial_is_and_promises_no_rollback(tmp_path):
    """THE UI MUST NOT PROMISE WHAT THE CODE DOES NOT DO. `urlretrieve` opens the destination
    `'wb'` and streams into it, so an interrupted download leaves a short `<acc>_RAW.tar` on
    disk and a re-run truncates rather than resuming (read in CPython's `urllib/request.py`).
    The screen names the path instead of saying "cleaned up".

    A SHORT out_dir on purpose, and nothing is written to it: pytest's `tmp_path` is 90-odd
    characters, and at an 80-column terminal the sentence wraps past the bottom of the frame, so
    the test would be measuring the width of a temp directory rather than what the screen says.
    """
    plan = _plan(Path("/tmp/gm-cancel"))
    app = HostApp(plan)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        confirm = app.screen
        confirm.state = "running"
        confirm.post_message(FetchScreen.Ended(None, cancelled=True))
        await pilot.pause()
        text = _flat(app)
        assert confirm.state == "cancelled"
        assert "a re-run starts it over, it does not resume" in text
        assert str(plan.partial) in text
        for word in ("cleaned up", "removed", "deleted"):
            assert word not in text, f"the screen promises a rollback it does not perform: {word}"


@astest
async def test_the_bar_belongs_to_the_download_and_the_phase_line_says_what_follows(tmp_path):
    """A bar that reaches 100% and then sits there through the extract is the worst version of
    this screen. The hand-over is a phase change, not a full bar."""
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.state = "running"
        screen.post_message(FetchScreen.Progress("download", 4096, 8192))
        await pilot.pause()
        assert screen.query_one("#fetch-bar").display is True

        screen.post_message(FetchScreen.Progress("extract", 8192, 8192))
        await pilot.pause()
        assert screen.query_one("#fetch-bar").display is False
        assert search.PHASE_GLOSS["extract"] in _flat(app)


@astest
async def test_cancelling_after_the_download_says_it_cannot_be_interrupted(tmp_path):
    """Neither `tarfile.extractall` nor scanpy's readers offer a callback to check a cancel
    flag from, so there is no checkpoint to stop at. Said plainly, rather than shown as a cancel
    that appears to work and does not."""
    app = HostApp(_plan(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = app.screen
        screen.state, screen.phase = "running", "convert"
        screen.action_back()
        await pilot.pause()
        assert "cannot be interrupted" in _flat(app)
        assert screen.state == "running"      # it did not lie about having stopped


# ── the seam back to the app ─────────────────────────────────────────────────
def test_the_finder_hands_the_app_the_same_type_the_roster_does():
    """`FindScreen` is `Screen[Optional[DataEntry]]` and `RosterScreen.DataChosen` carries a
    `DataEntry`; the app's `open_ledger(entry)` takes one. One type across both doors, which is
    what makes the two-line wiring in `find.py`'s docstring the whole integration."""
    import typing

    from manyruns.tui.roster import RosterScreen

    assert typing.get_type_hints(ManyrunsApp.open_ledger)["entry"] is DataEntry
    assert typing.get_type_hints(RosterScreen.DataChosen.__init__)["entry"] is DataEntry
    assert FindScreen.__orig_bases__[0].__args__[0] == typing.Optional[DataEntry]


def test_finder_without_a_manifest_keeps_synthetics(monkeypatch):
    monkeypatch.delenv("MANYRUNS_ACQUIRE_MANIFEST", raising=False)
    found = search.search("branching", datasets=[_entry(topology=("branching",))])
    assert len(found) == 1
    assert found[0].kind == "synthetic"
