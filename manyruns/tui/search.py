"""What `find data…` searches — the half with **no Textual and no rich in it**.

`manyruns/tui/find.py` is the screen; this is everything it would otherwise compute inside a
widget. Same seam and same reason as `tui/state.py`: the ranking, the size probe and the fetch
plan are testable with no terminal, and the screen is left owning layout and key handling.

THREE TIERS, CHEAPEST FIRST, AND THE ORDERING IS THE FEATURE — the common path must be instant
and must work with no key and no network:

    1. `pool()` + `rank()`   a user-supplied manifest + the bundled generators (13),
                             substring-matched locally. No network, no key, no SDK.
    2. …the same call        a bundled synthetic is a PEER in that ranking, not a fallback
                             printed underneath it — see `synthetics()`.
    3. `ask_a_model()`       one optional Anthropic call, only when 1 and 2 matched nothing,
                             only when `intent.llm_available()`. Degrades to None.

WHAT IS NOT HERE, and why: no downloading. `harness/data_acquire.py` already does
fetch → convert and it works; `plan()` computes what to SAY about a fetch (both sizes, the
phases, where the bytes land) and `find.py` calls `data_acquire` to do it. Reimplementing the
download to get a progress bar was the alternative and it is the wrong trade — the one thing
that was missing is a `reporthook`, which `urllib.request.urlretrieve` has always taken, so it
was added as a keyword to `data_acquire.fetch` rather than forked here. (Not to `acquire`:
this screen calls `fetch` and `convert` separately, because it has to name the boundary
between them, so a forwarded keyword there would be an unused parameter.)
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from manyruns.tui.state import DataEntry, _human_bytes

# `_human_bytes` is `state`'s, private, and imported anyway: it is the byte phrasing the roster
# already prints, and a second one here would give the same file two sizes on two screens. That
# is the same argument `state.roster` makes for calling `shell._resolve_path_source`.
# Sizes use binary units (1024) consistently with the roster.


# ══ the query ════════════════════════════════════════════════════════════════
#: Short substrings match unrelated words too easily; keep typed tokens at least
#: three letters long. Common abbreviations take the whole-word path below.
_MIN_TOKEN = 3

#: Generic control abbreviations can occur in user-supplied metadata. Match aliases
#: as whole words so a short abbreviation cannot match inside an unrelated word;
#: typed words remain substrings so stems can match longer descriptions.
ALIASES: dict[str, tuple[str, ...]] = {
    "healthy": ("HC",),
    "control": ("HC",),
    "controls": ("HC",),
}


def query_tokens(text: str) -> tuple[str, ...]:
    """A typed sentence → the words worth matching on. Deduped, order kept."""
    words = [w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) >= _MIN_TOKEN]
    return tuple(dict.fromkeys(words))


def _word_in(word: str, text: str) -> bool:
    """`word` as a whole token in `text`, case-sensitively-spelled but matched case-insensitively.

    Letter-only lookarounds recognize slash-separated abbreviations while preventing
    matches inside longer words.
    """
    return re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.I) is not None


def hits(tokens: "tuple[str, ...]", text: str) -> tuple[str, ...]:
    """Which of `tokens` this text answers — directly, or through one of its aliases."""
    low = text.lower()
    return tuple(t for t in tokens
                 if t in low or any(_word_in(a, text) for a in ALIASES.get(t, ())))


# ══ what a search returns ════════════════════════════════════════════════════
@dataclass(frozen=True)
class Candidate:
    """One thing you could get: a public cohort to fetch, or a generator to run here.

    Both kinds in ONE list and one ranking, because §"find data" of the component brief is
    explicit that a synthetic is a *peer* of the search results and not a consolation prize
    below them — and because for a hypothesis it is often the better answer: the generator made
    the structure, so the answer is known, where a public cohort's structure is what you are
    trying to find out.

    A `synthetic` carries the roster's own `DataEntry`, so choosing one hands the app exactly
    what the roster hands it (`entry.as_source()` → `shell._run`'s tuple) and nothing downstream
    learns that this screen exists.
    """

    kind: str                                  # "cohort" | "synthetic"
    name: str                                  # accession, or the bundled dataset's name
    summary: str                               # the one-line "what this is"
    note: str = ""                             # the curator's note, verbatim
    score: int = 0                             # how many query tokens it answered
    matched: tuple = ()
    accession: Optional[str] = None            # cohorts only — what `data_acquire` takes
    url: str = ""
    disk_after: str = ""                       # recorded, only where someone has fetched it
    entry: Optional[DataEntry] = None          # synthetics only — the roster row itself
    by_model: bool = False                     # tier 3 picked it, not the substring match
    cannot_fetch: str = ""                     # why this one cannot be downloaded, if it cannot

    @property
    def needs_fetch(self) -> bool:
        """True when choosing this means downloading. The one branch the screen takes on."""
        return self.kind == "cohort"

    @property
    def available(self) -> bool:
        """False for a row that is on screen to be READ, not chosen — the finder's hollow row.

        Straight from the ledger (`LedgerRow.can_run`) and for the same reason §3.2 argues:
        some sources cannot be fetched by this product, and dropping them would
        teach the scientist nothing where showing them with `cannot_fetch` teaches exactly what
        to do instead.
        """
        return not self.cannot_fetch

    @property
    def text(self) -> str:
        """Everything this candidate is matched against — one string, built once."""
        return " ".join(x for x in (self.name, self.summary, self.note) if x)

    def scored(self, tokens: "tuple[str, ...]") -> "Candidate":
        from dataclasses import replace

        got = hits(tokens, self.text)
        return replace(self, score=len(got), matched=got)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "summary": self.summary,
                "note": self.note, "score": self.score, "matched": list(self.matched),
                "accession": self.accession, "url": self.url, "disk_after": self.disk_after,
                "by_model": self.by_model, "cannot_fetch": self.cannot_fetch,
                "available": self.available,
                "entry": self.entry.to_dict() if self.entry is not None else None}


#: Why a manifest row cannot be downloaded here, keyed by its `source:` column. These are
#: `data_acquire.fetch`'s OWN two `NotImplementedError` branches, read from that function and
#: restated for a scientist — not a second policy about what is fetchable.
#: `test_the_unfetchable_sources_are_the_ones_data_acquire_itself_refuses` reads the source and
#: fails if a third branch appears there without a phrase here.
UNFETCHABLE: dict[str, str] = {
    "cellxgene":    "an atlas, not a download — it loads through the census, with no file",
    "arrayexpress": "the processed matrix has to be picked by hand at EBI first",
}


# ══ tier 1 · the curated manifest ════════════════════════════════════════════
def cohorts(entries: "list | None" = None) -> list[Candidate]:
    """The manifest's rows as candidates. Local, instant, and the answer to most questions.

    `harness/data_acquire.load_manifest` is the reader — the user supplies a TSV or CSV
    with a parser and a `meta` dict that carries every non-required column verbatim, which is
    what lets `disk_after` be a data column rather than a table in this file.

    Returns `[]` when the manifest is missing (`load_manifest` raises `FileNotFoundError` by
    design, and a finder with no cohorts is still a working finder — tier 2 does not need it).
    """
    from manyruns.harness import data_acquire as da

    try:
        rows = list(entries) if entries is not None else da.load_manifest()
    except Exception:  # noqa: BLE001 - see the docstring: tier 2 still works without tier 1
        return []
    out = []
    for e in rows:
        meta = dict(getattr(e, "meta", {}) or {})
        bits = [meta.get("tissue", ""), meta.get("disease", ""),
                f"{meta.get('n_samples', '')} samples" if meta.get("n_samples") else "",
                e.modality]
        out.append(Candidate(
            kind="cohort", name=e.accession,
            summary=" · ".join(b for b in bits if b and b != "NA samples"),
            note=meta.get("note", ""), accession=e.accession, url=e.url,
            disk_after=meta.get("disk_after", ""),
            cannot_fetch=UNFETCHABLE.get(e.source, ""),
        ))
    return out


# ══ tier 2 · make one instead ════════════════════════════════════════════════
def synthetics(datasets: "list[DataEntry] | None" = None) -> list[Candidate]:
    """The bundled generators, as peers of the cohorts.

    WHAT "MAKE ONE" MEANS HERE, precisely, because the honest scope is narrower than the phrase:
    it is *choose the generator whose structure you described*, not *synthesise a new one to
    order*. The 13 bundled datasets are handles onto manylatents generators (`swissroll`,
    `dla_tree`, …) and the structure is ground truth because the generator produced it —
    `tree_wide.yaml` says exactly that about its own `topology:`. Generating a NEW one with new
    parameters is compute, and CLAUDE.md puts compute in manylatents; a new *combination* is a
    dataset YAML, which is config manyruns owns but which no screen should be writing while the
    user waits.

    Reuses `state.roster()` rather than re-reading the dataset YAMLs, so a synthetic found here
    is the same `DataEntry` object the roster would have handed over — same size, same contents
    phrase, same `as_source()` tuple, and the same known defect (`synthetic_timecourse` reports
    shape `unknown`, pinned by component 1) rather than a second, differently-wrong account.
    """
    from manyruns.tui import state

    try:
        rows = datasets if datasets is not None else [
            e for e in state.roster() if e.kind == "bundled"
        ]
    except Exception:  # noqa: BLE001 - a broken dataset dir must not empty the finder
        return []
    return [Candidate(
        kind="synthetic", name=e.name,
        summary=f"{e.size} · {e.contents}",
        # The note is what makes it a peer: it says what a public cohort cannot say about
        # itself. `topology` is the generator's own label, not an annotation.
        note="generated here, with the structure known: "
             + (", ".join(e.topology) if e.topology else e.shape),
        entry=e,
    ) for e in rows]


# ══ the ranking ══════════════════════════════════════════════════════════════
#: Tie-break between kinds when the query separates them by nothing. Cohorts first, and the
#: reason is where the user came FROM: every synthetic in this list is already a row on the
#: roster screen they pressed `?` to leave. On an empty query that keeps the finder showing what
#: is new. It never overrides a score — a synthetic that answers two words outranks a cohort
#: that answers one, which is what "peer, not a fallback" means.
_KIND_ORDER = {"cohort": 0, "synthetic": 1}


def pool(entries: "list | None" = None,
         datasets: "list[DataEntry] | None" = None) -> list[Candidate]:
    """Everything findable, unscored. Built ONCE — `rank` is what runs per keystroke.

    Split from `rank` because the screen filters as you type and this half opens files:
    `state.roster()` is 361 ms cold (component 1 measured it on `data/pbmc3k_raw.h5ad`) and the
    manifest is a TSV parse. Ranking a built pool touches no disk at all.
    """
    return cohorts(entries) + synthetics(datasets)


def rank(text: str, candidates: list[Candidate]) -> list[Candidate]:
    """Score every candidate against the query and sort. Pure — no disk, no network.

    An empty query scores everything 0 and therefore returns EVERYTHING, in pool order. A finder
    that shows nothing until you type is a dead end at the moment you arrive, and the 25 rows are
    the whole catalogue anyone can reach from here.

    Non-matching rows are kept below the matching ones for the same reason the ledger keeps its
    hollow rows: what is absent from a list teaches nothing, and the curator's note on a row that
    missed your words is often the thing that tells you which word to type instead.
    """
    tokens = query_tokens(text)
    scored = list(enumerate(c.scored(tokens) for c in candidates))
    # The pool index is the last key, so the manifest's own order and `discover_datasets`'
    # sorted order both survive inside a score band — this is a filter, not a second catalogue
    # ordering (the same argument `state.ledger` makes for keeping `narrate.offer`'s order).
    scored.sort(key=lambda pair: (-pair[1].score, _KIND_ORDER.get(pair[1].kind, 9), pair[0]))
    return [c for _, c in scored]


def search(text: str, *, entries: "list | None" = None,
           datasets: "list[DataEntry] | None" = None) -> list[Candidate]:
    """`pool` + `rank` in one call — for a test, a REPL, or a one-shot caller."""
    return rank(text, pool(entries, datasets))


# ══ tier 3 · ask a model ═════════════════════════════════════════════════════
def ask_a_model(text: str, candidates: list[Candidate]) -> Optional[Candidate]:
    """One Anthropic call that picks ONE of the cohorts already on screen. Optional, guarded.

    WHAT IT DELIBERATELY IS NOT: a model naming a public accession we do not have. That was the
    ambitious reading of tier 3 and it was declined twice over — `data_acquire.fetch` would
    happily build a GEO URL for a hallucinated accession (it falls back to
    `geo_supplementary_url` when the manifest has no such row), and the failure would arrive as
    a 404 after a confirm screen that had already quoted "size unknown" as if it were a fact
    about a real dataset. Choosing from the supplied manifest cannot invent one.

    A model can relate a query to metadata whose wording differs from the query,
    while remaining restricted to the available candidates.

    `intent._llm_choose` is called rather than a second SDK call site, so there is exactly one
    place in the product that knows how to talk to Anthropic, one import guard and one exception
    guard. It is private to `intent`; calling it anyway is the same trade `state.roster` makes
    with `shell._resolve_path_source` — a second copy is how two callers come to disagree.

    Returns None whenever anything is missing (no SDK, no key, no network, an unexpected reply)
    or when the pick is not one of the cohorts offered.
    """
    from manyruns import intent

    fetchable = [c for c in candidates if c.needs_fetch and c.available]
    if not fetchable or not (text or "").strip() or not intent.llm_available():
        return None
    picked = intent._llm_choose(
        system=("You match a scientist's description of the data they need to exactly one "
                "dataset from a curated list. Choose the closest; if none is close, choose the "
                "first."),
        prompt="\n".join([f"They need: {text}", "", "The datasets:"]
                         + [f"{c.name}: {c.summary}. {c.note}" for c in fetchable]),
        choices=[c.name for c in fetchable],
        key="accession",
    )
    from dataclasses import replace

    for c in fetchable:
        if c.name == picked:
            return replace(c, by_model=True)
    return None


# ══ the fetch: what to say before starting one ═══════════════════════════════
#: What follows the bar. The download is the FAST part on the one cohort anyone here has
#: fetched, so a bar that reaches 100% and then appears to hang is the failure this tuple
#: exists to prevent: every phase is on screen from the start, and the bar belongs to the first.
PHASES: tuple[str, ...] = ("download", "extract", "convert", "verify")

#: What each phase is doing, for the line under the bar. `extract`/`convert` are
#: `data_acquire.fetch`'s `tar.extractall` and `data_acquire.convert`'s scanpy read + `.h5ad`
#: write; `verify` is this product reading the result back with the same reader the roster uses.
#: Kept short enough to sit on ONE line inside the fetch frame, measured at an 80-column
#: terminal (the frame is 70 wide there, and the longest of these is 53): a gloss that wraps
#: pushes the buttons off a 24-row screen.
PHASE_GLOSS: dict[str, str] = {
    "download": "pulling the archive",
    "extract":  "unpacking the tar — nothing to report until it is done",
    "convert":  "reading the matrices, writing one .h5ad",
    "verify":   "opening the result the way the roster will",
}

#: Bound the server wait; the screen runs this probe off the event loop.
PROBE_TIMEOUT = 20.0


class DownloadTicks:
    """Which of `urlretrieve`'s reporthook calls are worth repainting for.

    `urllib.request.urlretrieve` reads in 8 KB blocks. Repainting every block can
    overwhelm the event loop for large archives; one update per whole percent
    bounds repaint work independently of the number of downloaded bytes.

    `__call__` returns the report to post, or None to skip. The LAST tick is `("extract", …)`,
    not `("download", 100%)` — everything after the final block is `tar.extractall`, which
    reports nothing, so a bar that stays full and silent through it is the failure mode this
    hand-over exists to prevent.
    """

    #: When the server sent no `content-length`, `urlretrieve` passes `total = -1` and there is
    #: no percentage to change. Report every this-many blocks instead — 512 × 8 KB ≈ 4 MB, which
    #: is often enough that the number visibly moves and rare enough to stay off the loop.
    BLIND_EVERY = 512

    def __init__(self, every_percent: int = 1) -> None:
        self.every_percent = every_percent
        self.last = -1

    def __call__(self, blocknum: int, blocksize: int,
                 total: int) -> "Optional[tuple[str, int, int]]":
        read = blocknum * blocksize
        if total > 0 and read >= total:
            return ("extract", total, total)
        if total <= 0:
            return ("download", read, total) if blocknum % self.BLIND_EVERY == 0 else None
        percent = int(read * 100 // total) // self.every_percent
        if percent == self.last:
            return None
        self.last = percent
        return ("download", read, total)


@dataclass(frozen=True)
class Probe:
    """What the server said about the archive, before anyone downloads it."""

    nbytes: Optional[int] = None
    status: Optional[int] = None
    error: Optional[str] = None

    @property
    def size(self) -> str:
        """The download size, or the honest absence of one — **never an estimate.**"""
        return _human_bytes(self.nbytes) if self.nbytes is not None else "size unknown"

    @property
    def missing(self) -> bool:
        """The server says there is nothing there. 404 is the answer GEO gives for an accession
        that does not exist (measured on GSE000000), and it is the one probe result that should
        stop a fetch rather than merely fail to describe it."""
        return self.status == 404

    def to_dict(self) -> dict:
        return {"nbytes": self.nbytes, "status": self.status, "error": self.error,
                "size": self.size, "missing": self.missing}


def probe(url: str, timeout: float = PROBE_TIMEOUT,
          opener: Optional[Callable[..., Any]] = None) -> Probe:
    """HEAD the archive and read `content-length`. Never raises; never guesses.

    A server that reports no `content-length` gives `Probe(nbytes=None)`, which prints "size
    unknown". Estimating from anything — the sample count, another cohort's ratio — would be
    inventing the one number the confirm exists to state.

    `opener` is injectable so the tests exercise this with no network.
    """
    if not url:
        return Probe(error="no url")
    try:
        request = urllib.request.Request(url, method="HEAD")
        with (opener or urllib.request.urlopen)(request, timeout=timeout) as response:
            length = response.headers.get("content-length")
            return Probe(nbytes=int(length) if length and length.isdigit() else None,
                         status=getattr(response, "status", None))
    except urllib.error.HTTPError as e:
        return Probe(status=e.code, error=f"HTTP {e.code} {e.reason}")
    except Exception as e:  # noqa: BLE001 - offline is a fine answer, a crashed screen is not
        return Probe(error=f"{type(e).__name__}: {e}")


@dataclass(frozen=True)
class FetchPlan:
    """Everything the confirm screen states, and where the bytes will land.

    Compressed archives can expand substantially during extraction and conversion.
    Show both independently measured sizes. Where expansion has not been measured,
    `disk` says so; it is never scaled from another dataset's ratio.
    """

    accession: str
    url: str
    work_dir: Path
    out_path: Path
    probe: Probe = Probe()
    disk_after: str = ""              # the manifest's recorded figure, where there is one
    already_here: bool = False

    @property
    def download(self) -> str:
        return self.probe.size

    @property
    def disk(self) -> str:
        """The on-disk size after extract + convert — recorded, or honestly absent.

        There is no formula. The expansion is a property of how many empty droplets a study
        deposited, which nothing in the archive's headers reports.
        """
        return self.disk_after or "unknown until it runs — the archive expands"

    @property
    def partial(self) -> Path:
        """Where a cancelled download leaves its bytes.

        NOT a promise that anything is cleaned up, because nothing is: `urlretrieve` opens the
        destination `'wb'` and streams into it (read in CPython's `urllib/request.py`), so an
        interrupted fetch leaves a short `<accession>_RAW.tar` exactly here, and a re-run
        TRUNCATES and starts over rather than resuming. The screen says that instead of showing
        a "cancelled" that implies a rollback.
        """
        return self.work_dir / f"{self.accession}_RAW.tar"

    def to_dict(self) -> dict:
        return {"accession": self.accession, "url": self.url, "work_dir": str(self.work_dir),
                "out_path": str(self.out_path), "download": self.download, "disk": self.disk,
                "already_here": self.already_here, "partial": str(self.partial),
                "probe": self.probe.to_dict()}


def plan(candidate: Candidate, *, out_dir: "Path | None" = None,
         probe_it: bool = False) -> FetchPlan:
    """What a fetch of this cohort would cost and where it would put things.

    OUT_DIR IS THE DROP FOLDER, and that is the whole integration: `shell.data_dir()` is where
    `state.roster()` looks, so a fetched `.h5ad` is a roster row the next time the app opens —
    no registration step, no second catalog, no name to remember.

    The work dir is `.fetch/<accession>` INSIDE it, hidden on purpose: `shell._scan_drop_folder`
    skips any entry whose name starts with a dot (read, not assumed), so the downloaded tar and the
    extracted tree never appear as droppable rows beside the `.h5ad` they produced.

    `probe_it` is off by default because the probe is network and the caller decides when to
    spend 8 seconds on it — `find.py` does it in a worker after the screen is already up.
    """
    from manyruns import shell

    base = Path(out_dir) if out_dir is not None else shell.data_dir()
    accession = candidate.accession or candidate.name
    return FetchPlan(
        accession=accession, url=candidate.url,
        work_dir=base / ".fetch" / accession,
        out_path=base / f"{accession}.h5ad",
        probe=probe(candidate.url) if probe_it else Probe(),
        disk_after=candidate.disk_after,
        # `data_acquire.convert` returns early (and `acquire` never downloads) when the
        # canonical path already exists, so this is not advice — it is what the fetch will do.
        already_here=(base / f"{accession}.h5ad").exists(),
    )


def roster_entry(path: Path, ddir: "Path | None" = None) -> Optional[DataEntry]:
    """The fetched file as a roster row — the read-back that ends a fetch.

    THIS IS THE VERIFY STEP, and it is deliberately the product's own reader rather than
    `data_acquire.assert_loadable`. That function asks "can manylatents build an AnnDataModule
    from this", which is the harness's question; the question a scientist just downloaded
    is "is it on my roster", and the two can differ — `state.roster` SKIPS a file
    `shell._resolve_path_source` cannot resolve, silently (component 1 reports that gap). Doing
    the roster's own read here is what turns that silent skip into a visible failure at the one
    moment someone can act on it.

    Returns None when the file is not a roster row, which the screen reports as a failed verify.
    """
    from manyruns.tui import state

    resolved = Path(path).resolve()
    return next((e for e in state.roster(ddir)
                 if e.path is not None and Path(e.path).resolve() == resolved), None)
