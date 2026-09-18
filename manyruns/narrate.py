"""The human-facing layer — read the data, speak what it sees, offer plain-English questions,
narrate the result, and refuse an analysis that would mislead.

This is the "killer demo" spine (see the brainstorm): the app knows the science and spends it
on the human's behalf. It reshapes geometry into *findings in the user's vocabulary* and never
utters an algorithm name. Pure/testable — `read_data` reuses `app`'s detection (lazy import);
everything else is string logic over an `Observation` or a run's g-vector.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

# words that mean "a process over time" — a trajectory only makes sense with a time axis.
_TRAJECTORY_WORDS = (
    "trajector", "progress", "over time", "develop", "pseudotime", "time course",
    "lineage", "differentiat", "unfold", "evolv", "advance",
)
# words that mean "compare groups".
_CONTRAST_WORDS = ("compare", "differ", "versus", " vs", "separate", "distinguish", "contrast",
                   "which cells", "which genes", "shift", "enrich")


#: The one line under the name, on both front doors. Lives here because this is the module
#: both already read for human phrasing, and because it used to be written out twice —
#: `shell.run_shell` and the roster's title — with a comment on each saying that if it ever
#: became a constant both should read it. It is the loop the product is: read the data, name
#: what you are assuming, run it, look, go again.
TAGLINE = "load. assume. run. repeat."


@dataclass
class Observation:
    """What the app sees in the data, before it decides anything."""

    shape: str  # "time-course" | "case-control" | "single" | "unknown"
    modality: str = "unknown"
    conditions: Optional[list] = None
    n_timepoints: Optional[int] = None
    n_obs: Optional[int] = None
    n_vars: Optional[int] = None
    source: str = "your data"
    signals: dict = field(default_factory=dict)  # raw detection facts for tests/inspection
    #: What a dataset DECLARATION states this data carries beyond its shape — `declared_facts`
    #: of the YAML this Observation was built from. `None` means "no declaration in hand", which
    #: is every dropped file: a file someone dragged into the drop folder has no YAML at all, so
    #: there is nothing to prefer and `provides` falls back to the sniff.
    declared: Optional[tuple] = None
    #: The unordered IDENTITY axis read off `.obs` — the column that COLOURS the plot, and the
    #: names in it (`vocab.GROUP_KEYS`; `cell_type`, `branch`, `leiden`, …). `None` when there is
    #: none, or when a time or condition axis outranks it — `app._inspect_obs` gates on
    #: `pipeline.loading.labels_of`'s precedence, so what is reported here is what the run will
    #: actually colour by.
    #:
    #: A NARRATION FACT AND NOTHING ELSE. It is not passed to `shape_of`, it is not in
    #: `provides()`, and it is not a `vocab.DATASET_FACTS` word — so it moves no legality. It
    #: exists because the app was drawing eight named cell types and saying, on the same screen,
    #: "no groups or time labels I recognize".
    #:
    #: `list`, never `tuple`, unlike `declared`: `tui.state.DataEntry.to_dict` is asserted to
    #: round-trip through JSON (`test_every_view_serialises_to_plain_json`), and a tuple comes
    #: back a list. `declared` pays for its tuple with a `_obs_json` special case; this field
    #: does not need one, so it does not have one.
    group_key: Optional[str] = None
    groups: Optional[list] = None
    #: Small JSON-plain metadata inventory: [name, display kind, distinct count]. No levels
    #: or expression data. None means uninspected/unreadable; [] means inspected and empty.
    #: Detailed availability/missing counts are read only on an explicit colour action.
    columns: Optional[list] = None

    def provides(self) -> frozenset:
        """The calculus facts this observation carries beyond its `shape`.

        The shell's counterpart to a dataset declaration's `handle`. `experiment.plan` and
        `app`'s coverage table hold a DECLARATION and read `vocab.dataset_provides(shape,
        handle)`; the shell has already loaded the data and holds one of these instead, so the
        same fact has to be reachable from here or dropping real scRNA into the app would
        refuse the very steps that need a gene axis.

        **A DECLARATION OUTRANKS THE SNIFF**, and the sniff is what remains when there is none.
        Both surfaces reach the calculus through this method, so a fact stated in a YAML that
        could not be read from here was a fact neither front door could see. Measured
        2026-08-17, with the declared file NOT REACHABLE from where the app was launched — an
        installed `uv tool install` manyruns, a worktree, a `cd` out of the checkout with an
        empty drop folder: `configs/dataset/pbmc3k.yaml` declares `handle.provides: [genes]`,
        `shell._resolve_dataset` cannot open the bytes and returns an Observation with
        `n_vars=None`, so `provides()` was empty and `tui.state.ledger` drew `qc`, `markers` and
        `preprocess` hollow — "needs data with gene names — a point cloud has no genes to read",
        about the one declaration in the folder that states it has them. 6 analyses offered
        where the declaration offers 9. Where the file IS reachable the two agreed, which is why
        it was invisible: the sniff is right exactly there and silent everywhere else.

        The declaration REPLACES the sniff rather than being unioned with it, matching
        `vocab.dataset_provides`, where an explicit `provides:` also wins over the suffix guess.
        That is what makes it an authority: a declaration is the reviewable statement about what
        the bytes carry, and one that says a `.h5ad` has no usable gene axis has to be able to
        say so. The two cannot disagree by accident on today's catalog — `read_data` sets
        `n_vars` only on `.h5ad`/`.h5`, which is exactly `vocab._GENE_AXIS_SUFFIXES` — so a
        divergence is always something a declaration went out of its way to state.

        `n_vars` is the discriminator for the fallback and is not a proxy: `read_data` sets it
        only on the `.h5ad`/`.h5` branch, via `_h5ad_shape`, which is precisely the branch on
        which a gene axis exists. A folder of 10x samples or a synthetic point cloud leaves it
        None.
        """
        if self.declared is not None:
            return frozenset(self.declared)
        return frozenset({"genes"}) if self.n_vars is not None else frozenset()


def declared_facts(ds: "dict | None") -> frozenset:
    """A dataset DECLARATION → the calculus facts its `handle:` states, beyond its shape.

    THE ONE READER OF A DECLARATION'S FACTS on the human-facing side, so that a surface holding
    a loaded `dataset/*.yaml` never re-implements the lookup. It is a thin projection of
    `vocab.dataset_provides`, not a second answer: that function owns both the declared list and
    the suffix guess behind it, and this only subtracts the SHAPE half so the result is exactly
    what `Observation.declared` means ("beyond its shape"). The shape half needs no carrying —
    `vocab._walk` already unions `dataset_provides(shape, …)` in from the shape it is given, so
    passing it through `provided` as well would be the same fact travelling twice.

    Measured on the fourteen bundled declarations, 2026-08-17: `pbmc3k` returns `{"genes"}` and
    the other thirteen return `frozenset()` — twelve `kind: manylatents` handles the engine
    loads and manyruns never sees the columns of, and one generated `synthetic:` ref.

    An empty result is a STATEMENT ("this declaration says nothing beyond its shape"), which is
    why every dataset row gets one and only a dropped file gets `None`. A swissroll genuinely
    has no genes, and saying so is not the same as having nothing to say.
    """
    from manyruns.vocab import dataset_provides

    ds = ds or {}
    shape = str(ds.get("shape") or "unknown")
    return dataset_provides(shape, ds.get("handle")) - dataset_provides(shape)


def read_data(data_path: Optional[Path], modality: str, *, source: Optional[str] = None) -> Observation:
    """Inspect the data's shape (reuses app's detection) → an Observation. Never raises."""
    from manyruns import app

    p = Path(data_path) if data_path is not None else None
    src = source or (p.name if p is not None else "your data")

    has_time = app._has_timepoint_subdirs(p)
    conditions = None
    group = None
    n_obs = n_vars = n_timepoints = None
    if has_time:
        n_timepoints = sum(1 for _ in app._sample_subdirs(p))
    elif p is not None and p.is_dir():
        conditions = app._dir_conditions(p)
    elif p is not None and p.suffix.lower() in {".h5ad", ".h5"}:
        has_time, conditions, group = app._inspect_obs(p)
        n_obs, n_vars = _h5ad_shape(p)

    # The group axis reaches the Observation and NOT `shape_of` — the shape is derived from the
    # same two facts it always was, so nothing downstream of `Observation.shape` moves.
    group_key = group[0] if group else None
    groups = list(group[1]) if group else None
    shape = shape_of(has_time, conditions, modality)
    return Observation(
        shape=shape, modality=modality, conditions=conditions, n_timepoints=n_timepoints,
        n_obs=n_obs, n_vars=n_vars, source=src, group_key=group_key, groups=groups,
        signals={"has_time": has_time, "conditions": conditions, "group": group_key},
        columns=metadata_columns(p),
    )



def metadata_columns(path: Optional[Path]) -> Optional[list]:
    """Metadata-only backed inspection: ``[[name, kind, n_distinct], ...]`` or None.

    Keep ``app._inspect_obs``'s three-value analysis API intact. This extra open never
    reads X or generates a dataset during roster inspection. An explicit picker reads
    afresh, gives detailed availability reasons and reports source read failures.
    """
    if path is None or Path(path).suffix.lower() != ".h5ad":
        return None
    try:
        import anndata
        from manyruns.pipeline.loading import color_columns_of

        obj = anndata.read_h5ad(path, backed="r")
        try:
            return [[c["key"], c["kind"], c["n_distinct"]] for c in color_columns_of(obj)]
        finally:
            if getattr(obj, "isbacked", False) and obj.file is not None:
                obj.file.close()
    except Exception:  # metadata must not change the existing shape/legality inspection
        return None


def _h5ad_shape(path: Path) -> tuple[Optional[int], Optional[int]]:
    try:
        import anndata

        a = anndata.read_h5ad(path, backed="r")
        try:
            return int(a.n_obs), int(a.n_vars)
        finally:
            if getattr(a, "isbacked", False) and a.file is not None:
                a.file.close()
    except Exception:  # noqa: BLE001 - anndata absent / not a real .h5ad
        return None, None


def shape_of(has_time_axis: bool, conditions: Optional[list], modality: str = "unknown") -> str:
    """The two observable facts → a data shape. The ONE derivation.

    Extracted so `app._choose_recipe` and `read_data` cannot disagree about what a given
    (time axis, conditions) pair means. They did: the interactive path routed an unlabelled
    array to `embed` while the non-interactive path routed the same array to `cflows`.
    """
    if has_time_axis:
        return "time-course"
    if conditions and len(conditions) >= 2:
        return "case-control"
    if modality in ("scrna", "bulk"):
        return "single"
    return "unknown"


#: How many group names `describe` will spell out before it elides. Six fits the demo fixture
#: (`data/pbmc3k_annotated.h5ad`, eight immune types) with an honest "…" on the end, and keeps a
#: `leiden` column with 40 categories from spilling its whole vocabulary into one sentence. The
#: COUNT is stated either way, so the elision never hides how many there are.
_MAX_NAMED_GROUPS = 6


def _group_phrase(obs: Observation) -> str:
    """"labelled by `cell_type` into 8 groups: …" — or "" when there is no group axis.

    Names the COLUMN, because that is the fact the user can go and check for themselves, and
    because `shell._loaded_stats` already spends the bare word "groups" on `obs.conditions`,
    which is a different axis entirely. Order is `app._inspect_obs`'s: biggest population first,
    so a truncated list shows the groups that hold the most cells rather than the alphabet's.
    """
    if not obs.groups:
        return ""
    names = [str(g) for g in obs.groups]
    shown = ", ".join(names[:_MAX_NAMED_GROUPS]) + (", …" if len(names) > _MAX_NAMED_GROUPS else "")
    key = f" by `{obs.group_key}`" if obs.group_key else ""
    return f"labelled{key} into {len(names)} groups: {shown}"


def describe(obs: Observation) -> str:
    """Plain-English 'here's what I see' — stated before the app decides anything.

    THE GROUP CLAUSE IS NOT COSMETIC. Without it, this function said "one population, no groups
    or time labels I recognize" over `data/pbmc3k_annotated.h5ad` — 2,638 cells whose one `obs`
    column names eight immune types, which the very next screen draws the plot coloured by. The
    app denied, in words, the labels it was about to spend. Same sentence on `data/tree8.h5ad`
    and its eight branches.

    The clause hangs off `obs.groups`, NOT off `obs.shape`, which is still `single` for both
    files — see `app._inspect_obs` for why the shape deliberately did not move with it. The
    "no time axis and no case/control arms" half is not filler: it is the reason a `single`
    with groups is still a `single`, and it is guaranteed true here because `_inspect_obs` only
    reports a group axis when neither of the other two is present.
    """
    n = f"{obs.n_obs:,} cells" if obs.n_obs else "your rows"
    if obs.shape == "time-course":
        tp = f" at {obs.n_timepoints} time points" if obs.n_timepoints else ""
        return (f"I see {n} measured{tp} — a time course. "
                "Cells caught in the middle of changing.")
    if obs.shape == "case-control":
        groups = ", ".join(obs.conditions or [])
        return (f"I see {n} in {len(obs.conditions)} groups: {groups}. "
                "No time axis — a snapshot, not a time course.")
    grouped = _group_phrase(obs)
    if obs.shape == "single":
        if grouped:
            return (f"I see {n} {grouped} — no time axis and no case/control arms, so these say "
                    "who each cell is, not what was done to it.")
        return f"I see {n}, one population, no groups or time labels I recognize."
    dims = f" × {obs.n_vars:,} columns" if obs.n_vars else ""
    if grouped:
        return (f"I see {n}{dims}, {grouped} — no time axis and no case/control arms, so these "
                "say who each row is, not what was done to it.")
    return (f"I see {n}{dims}"
            ". No labels I recognize — no groups, no time, no tags. Just the numbers.")


def _short_cite(cite: str) -> str:
    """`"Cutler & Breiman, 'Archetypal Analysis', Technometrics 36(4):338-347 (1994)"` ->
    `"Cutler 1994"`.

    The recipes carry a full citation with a DOI and no surface ever showed it. A ledger row has
    one line, so this is the form that fits: lead author (or the source's name, for the two
    tutorial-derived recipes) plus the year. The full `source` block stays in the YAML for
    anyone who wants the DOI — this is a label, not a replacement.

    The year pattern is deliberately narrow (`19xx`/`20xx`/`21xx` on word boundaries) so a page
    range like `29705-29718` cannot be read as one.
    """
    if not cite:
        return ""
    lead = re.split(r",| & | et al", cite, maxsplit=1)[0].strip().strip("'\"")
    years = re.findall(r"\b(1[89]\d{2}|20\d{2}|21\d{2})\b", cite)
    return f"{lead} {years[-1]}" if years else lead


def offer(obs: Observation, available: "list | None" = None) -> list[dict]:
    """The question menu — derived from the CATALOG, not from a table in this file.

    Each recipe declares what it answers (`question:`) and the shapes it suits (`suits:`), so
    adding a recipe to `configs/recipe/` adds its own menu entry. The previous version was
    three hardcoded shape branches naming `cflows` / `contrast` / `embed` as literals, and it
    disagreed with the calculus that decides legality: measured across the six shapes, it
    offered fewer recipes than the data could legally run on FIVE of them, hiding `cflows`
    everywhere but `time-course`. A recipe it had no phrase for reached the menu as the bare
    label "Run the <name> recipe", beside options written as questions.

    `recommended` marks suitability, never legality. `available` is the legal set from
    `vocab.unmet`; a recipe that is legal but unsuited is still returned, unrecommended, so the
    caller can offer it one level down. Legality is structural, suitability is advice, and
    collapsing them is how a menu comes to hide a runnable analysis.

    **A recipe that cannot run anywhere is never RECOMMENDED, and it is still returned.** Both
    halves matter, and the split cost a real inconsistency between surfaces: `unavailable()` had
    exactly one caller — `tui.state.ledger` — so the app drew `archetypes` as a hollow row
    reading "blocked upstream", while the rich shell, reading this function, offered the same
    recipe as its TOP recommendation for `swissroll`. Measured by driving both. One recipe, two
    answers, which is the failure `vocab.py`'s header describes and the reason the suppression
    belongs HERE rather than in a second call each surface has to remember to make.

    Still returned, because a caller that wants to SHOW the refusal needs the row: the ledger's
    hollow entry is informative precisely because it is on screen with its reason. Dropping it
    from the menu would take a stated blockage back to an invisible one — §0's third defect.

    **A recipe this DATA cannot run is not RECOMMENDED either**, and that is the second half of
    the same invariant. `recommended` was `suits` minus `unavailable`, which was sound while a
    recipe on its own declared suit could only be blocked by a missing DATA fact — something
    `suits` rules out by construction. A narrowing blocks for a reason internal to step ORDER,
    which is shape-independent and about which `suits` says nothing, so `[phate, filter_genes,
    mioflow] suits: [manifold]` sorted FIRST and starred here while `tui.state.ledger` drew it
    as a hollow row. Through `blocked` rather than `unmet` directly, so both surfaces ask the
    one question in the one way.
    """
    from manyruns.catalog import discover_recipes, load_recipe

    names = list(available) if available is not None else discover_recipes()
    out: list[dict] = []
    for name in names:
        try:
            recipe = load_recipe(name)
        except Exception:  # noqa: BLE001 - a broken recipe must not empty the menu
            continue
        # A DEV FIXTURE IS NOT A PRODUCT MENU ENTRY, and it says so itself: it declares a `dev.`
        # id, and the one in the catalog today has a header calling it a recipe of steps that do
        # not work yet, so the app can be driven against them. It reached the shipped menu as an
        # offer to "drive the app", advertising that none of its steps is built.
        #
        # Filtered on the PREFIX rather than on a name — deliberately, and this function is
        # guarded on it (`test_a_recipe_carries_its_own_question_so_the_menu_needs_no_table`
        # reads this comment too): a second fixture is then excluded by existing, and the
        # catalog keeps deciding what the menu holds.
        rid = str(recipe.get("id") or "")
        if rid.startswith("dev."):
            continue
        question = recipe.get("question") or f"Run the {name} recipe"
        suits = tuple(recipe.get("suits") or ())
        # `claims` is the geometry the analysis ASSUMES, already written in the topology
        # vocabulary — so a caller can group the menu by it without asking the catalog a
        # second question of its own. `None` for the conditioning recipes, which prepare the
        # data rather than assume a shape of it and so have no assumption to state.
        claims = recipe.get("claims")
        out.append({"label": question, "recipe": name, "id": rid,
                    "method": rid.rsplit(".", 1)[-1] if rid else name,
                    "assumption": claims,
                    "glyph": ASSUMPTION_GLYPH.get(claims) if claims else None,
                    "cite": _short_cite((recipe.get("source") or {}).get("cite") or ""),
                    "recommended": (obs.shape in suits and not unavailable(recipe)
                                    and not blocked(recipe, obs.shape))})
    # recommended first, then alphabetical — a stable order, so the menu does not reshuffle
    # between runs on the same data.
    out.sort(key=lambda o: (not o["recommended"], o["recipe"]))
    return out


#: A missing fact, in the scientist's words — the reason a hollow row on the ledger is hollow.
#:
#: A PRESENTATION table over `vocab.facts()`, not a second declaration of the vocabulary. It is
#: the same relationship `_OUTCOME_MARKS` has to `watch.OUTCOMES`: the keys must be exactly the
#: fact names the calculus already declares, and the sentence is the only thing added here.
#: `test_tui_state.py::test_every_fact_the_calculus_names_has_a_phrase_and_no_others` fails in
#: BOTH directions, which is what stops it from becoming a private word list that drifts.
#:
#: WHY IT LIVES IN `narrate` AND NOT IN `vocab`. `vocab`'s job is the closed word list and the
#: subtraction over it; `unmet` returns fact NAMES and deliberately says nothing about how to
#: say them. This module is the one both other surfaces already read for human phrasing
#: (`shell.render_result` and `run_panel` both come here), so a fifth surface asking "why can't
#: I run this?" gets the same sentence rather than its own. Putting the prose in `vocab` would
#: have worked too, and the deciding argument is the rule the TUI is built under: no screen may
#: compute a fact `narrate` does not expose. The screens need this sentence, so it is here.
#:
#: `time` is in the table and is currently UNREACHABLE through `unmet` — measured across all 8
#: bundled recipes × all 6 shapes, `unmet` returns only `conditions` (and only for `contrast`),
#: because no step declares a `time` need. It is phrased anyway because it is one of `vocab`'s
#: four fact words and the exhaustiveness check covers the vocabulary, not today's reachable
#: subset. Stated rather than discovered later: a ledger row reading "needs timepoints" cannot
#: appear until some step declares `time` in `vocab.STEP_NEEDS`.
MISSING_FACT: dict[str, str] = {
    "conditions": "needs a healthy/disease column",
    "time":       "needs timepoints",
    "embedding":  "needs coordinates first — nothing here lays the cells out",
    "clusters":   "needs groups first — nothing here partitions the cells",
    # A FRAME fact, unlike the two above, and the phrasing has to say so. No step can produce a
    # gene axis and no reordering can conjure one, so the fix is never "add a step" — it is
    # different data. A sentence borrowed from the step-list vocabulary would send someone
    # looking for an algorithm that cannot exist.
    "genes":      "needs data with gene names — a point cloud has no genes to read",
    # A probe asks a question of a fitted object. With none anywhere in the recipe the fix is to
    # add the step that trains one, which is what this sentence has to say — unlike the CLEARED
    # phrasing below, where one was trained and a filter then invalidated it.
    "model":      "needs a trained model first — nothing here fits one to ask",
    # A FRAME fact like `genes`: no step can produce splicing and none ever will — it comes off
    # the sequencer. So the fix is never "add a step", it is different data, and the sentence
    # has to point there rather than at an algorithm that cannot exist.
    "splicing":   "needs spliced/unspliced counts in the file — quantify with velocyto or "
                  "kallisto|bustools first; a plain count matrix cannot be made to have them",
    # STEP-produced, so this one IS "add a step" — and the step is a tool that runs in its own
    # interpreter, which is why the sentence names where to look rather than an algorithm.
    "velocity":   "needs a velocity step first — nothing here computes a field "
                  "(`manyruns tools list`), or supply one with the data",
}

#: The SAME fact, when a narrowing took it away rather than nothing having produced it.
#:
#: Two reasons, because there are two ways to arrive at one missing fact and they call for
#: opposite fixes. `[mioflow]` never lays the cells out and wants a latent step;
#: `[phate, filter_genes, mioflow]` lays them out and then removes the cells the coordinates
#: were fitted over, and wants the filter moved. Told the first sentence, a user of the second
#: recipe adds a second embedder and is refused again — the reason above is a claim about the
#: STEP LIST, and it is false about a recipe whose first step is phate.
#:
#: A PRESENTATION table like `MISSING_FACT`, and keyed by a DERIVED set: exactly
#: `vocab.step_facts()`, the facts a step produces, which is exactly what `have &= keep` can
#: take away. A frame fact survives every narrowing, so `time` and `conditions` are absent by
#: derivation rather than by hand — checked in both directions by
#: `test_tui_state.py::test_every_fact_a_narrowing_can_clear_has_its_own_phrase_and_no_others`.
CLEARED_FACT: dict[str, str] = {
    "embedding": "needs coordinates again — a filter after them dropped the cells they were "
                 "fitted over",
    "clusters":  "needs groups again — a filter after them dropped the cells they were "
                 "grouped from",
    # The costliest of the three to re-earn, and the most dangerous to leave stale: a stale
    # embedding is coordinates you might mistrust, a stale model ANSWERS QUESTIONS — confidently,
    # in the right shape, about cells that are gone.
    "model":     "needs fitting again — a filter after it dropped the cells it was trained on",
    # A field is fitted over a NEIGHBOURHOOD, so a filter invalidates the rows that remain —
    # the same argument as the embedding above, and the reason `velocity` is step-produced
    # rather than a frame fact like the LAYERS it was computed from.
    "velocity":  "needs computing again — a filter after it dropped the cells the field was "
                 "fitted over",
}


def blocked(recipe: dict, shape: str,
            provided: "frozenset | set | tuple | None" = None) -> list[str]:
    """Why this recipe cannot run on this data, in the scientist's words. `[]` means it can.

    One call, one answer. `vocab.unmet` decides legality and this says it out loud, so a caller
    never holds a legality verdict and a reason that were computed separately — which is how the
    two come to disagree. The invariant is exact and is pinned by test: `blocked(...) == []` if
    and only if `unmet(...) == frozenset()`.

    WHICH sentence comes from `vocab.invalidated`, the second projection of the same fold, so
    the reason is computed from the same walk as the verdict rather than guessed at from the
    step list here. That is the whole reason it lives in the calculus: re-deriving "did a filter
    clear this" in this module would be a second implementation of the fold, which is the drift
    the split exists to prevent.

    An unphrased fact degrades to `needs <fact>` rather than dropping out of the list. A silent
    drop would turn an illegal recipe into a runnable-looking one with no reason attached, which
    is worse than showing a machine word; the two tables' exhaustiveness tests are what keep the
    fallbacks unreached. The cleared fallback says `again`, because falling back to
    `MISSING_FACT` there would print the one sentence known to be false about that recipe.

    Sorted, so the ledger does not reshuffle between renders of the same data.
    """
    from manyruns.vocab import invalidated, unmet

    cleared = invalidated(recipe, shape, provided)
    return [CLEARED_FACT.get(f, f"needs {f} again") if f in cleared
            else MISSING_FACT.get(f, f"needs {f}")
            for f in sorted(unmet(recipe, shape, provided))]


def unavailable(recipe: dict) -> list[str]:
    """Why this recipe cannot run ANYWHERE right now, whatever data you point it at. `[]` if it
    can.

    A SECOND question from `blocked`, deliberately not folded into it. `blocked` answers "does
    this data carry the facts these steps need", and its invariant — `blocked(...) == []` if and
    only if `unmet(...) == frozenset()` — is what stops a legality verdict and its reason from
    being computed separately and disagreeing. A step whose implementation is broken upstream is
    missing no data fact at all, so answering it through `unmet` would either break that
    invariant or make `unmet` mean two things.

    WHAT IT EXISTS FOR, measured: `manylatents`' `aa` raises `TypeError: AA.__init__() got an
    unexpected keyword argument 'method'` — a version skew against the `archetypes` package,
    one line, and theirs not ours. `archetypes.yaml` has recorded that in prose since it was
    written, and the ledger offered the recipe anyway: on `swissroll` it was the FIRST analysis
    the front door suggested, and taking it ran `pca`, failed `aa`, and returned a g-vector with
    no archetypes in it. The recipe still ships, because its steps are right and the fix is one
    line in someone else's repo — it just stops being offered as though it worked.

    The reason is the recipe's own `blocked:` field, so the file that documents the breakage is
    the file that declares it. A `blocked:` with no reason is not a block: an unexplained
    hollow row is the interface defect §0 of the TUI spec names.
    """
    reason = recipe.get("blocked")
    if isinstance(reason, str):
        reason = [reason]
    return [str(r) for r in (reason or []) if str(r).strip()]


def split_question(question: str) -> tuple[str, str]:
    """A recipe's `question:` → `(ask, gloss)` — the two columns the ledger draws.

    DERIVED from the catalog, not a second table of short labels. Every bundled recipe already
    writes its question as `<the ask> — <what it will actually do>`; measured, all 8 split on
    the em-dash ("Trace the path — where are these cells heading, and what drives the change?"
    → "Trace the path" / "where are these cells heading, and what drives the change?"). A
    hand-kept table of short verbs beside the questions would be the two-homes failure, and it
    would drift the moment someone reworded a `question:` and not its twin.

    A question with no em-dash is all ask and no gloss — the honest degradation, since inventing
    a gloss would put words in a recipe's mouth. `?` is kept on the ask (it is part of the
    sentence); the em-dash and its spaces are not.
    """
    text = " ".join(str(question or "").split())
    head, sep, tail = text.partition("—")
    if not sep:
        return text, ""
    return head.strip(), tail.strip()


#: What a dataset holds, in three or four words — the roster's "what's in it" column.
#:
#: A presentation table over `vocab.TOPOLOGIES` + `vocab.SHAPES`, same discipline as
#: `MISSING_FACT`: keys exactly those two closed lists, checked in both directions. The phrases
#: restate the comments those tuples already carry (`multi-branching` — "a tree: one arm splits
#: into two or more"), so this adds a rendering, not a claim.
#:
#: TOPOLOGY WINS over shape when a dataset declares one, and `tree_wide.yaml` is the measured
#: reason: it is `shape: manifold, topology: [multi-branching]`, and its own comment says the
#: shape vocabulary "CANNOT express this dataset (a 20-branch tree is not a `manifold`, but no
#: closer word exists)". Reading `shape` first would print "a continuum" over a branching tree.
_TOPOLOGY_PHRASE: dict[str, str] = {
    "clusters":          "separate groups",
    "single-trajectory": "one continuous path",
    "multi-branching":   "a branching tree",
    "archetypal":        "extremes, and mixtures of them",
    "cycle":             "a closed loop",
    "surface":           "a curved sheet",
}

#: The assumed geometry, DRAWN. A presentation table over the topology vocabulary, exactly as
#: `_TOPOLOGY_PHRASE` above is its English twin — same keys, checked in both directions by
#: `test_every_claimed_geometry_has_a_glyph_and_no_others`, so this cannot quietly become a
#: private word list.
#:
#: THE GLYPH IS THE GEOMETRY, not an icon chosen for flavour: `∴` is three separated points,
#: `△` is the simplex archetypal analysis literally fits, `∿` is a curved sheet. A reader who
#: knows the method recognises the picture, and a reader who does not can still see that two
#: rows sharing a glyph assume the same thing — which is the whole point of grouping by it.
#:
#: ONE COLUMN WIDE, EACH. The ledger pane wraps badly at 100 columns (measured), so the glyph
#: column is the only alignment the layout can rely on; a two-cell character silently breaks
#: every row below it. `test_every_glyph_is_exactly_one_column_wide` asks `rich.cells.cell_len`,
#: not `len`, because an emoji is one character and two cells — and rich rather than `wcwidth`
#: because rich is what draws this pane, so its answer is the one the layout is computed with.
#: The family is the one the README banner already draws with (`△ ∿ ◇ · ∴ ○ □`).
ASSUMPTION_GLYPH: dict[str, str] = {
    "clusters":          "\u2234",   # ∴  three separated points
    "single-trajectory": "\u219d",   # ↝  a directed path
    "multi-branching":   "\u2442",   # ⑂  a tree
    "archetypal":        "\u25b3",   # △  a simplex — what AA fits
    "cycle":             "\u25cb",   # ○  a closed loop
    "surface":           "\u223f",   # ∿  a curved sheet
}

_SHAPE_PHRASE: dict[str, str] = {
    "manifold":     "a continuum",
    "clusters":     "separate groups",
    "time-course":  "measured over time",
    "case-control": "labelled conditions",
    "single":       "one population",
    "unknown":      "unlabelled — just the numbers",
}


def contents(shape: str, topology: "Iterable | None" = None,
             groups: "list | None" = None) -> str:
    """"What's in it" for one dataset row — its declared topology, else its groups, else shape.

    Multi-label topology is joined, because `vocab.TOPOLOGIES`'s own docstring is explicit that
    the benchmark aggregates by union and a dataset "frequently exhibits more than one
    organizational regime": collapsing to the first would drop half of what was declared.

    OBSERVED GROUPS SIT BETWEEN THE TWO, and that ordering is the point. A declaration still
    outranks them — a `topology:` is a reviewed statement, `groups` is a sniff. But a sniff that
    counted eight named cell types outranks the shape word, because the shape word was the thing
    lying: this is the string the TUI actually renders (`tui.state.DataEntry.contents` → the
    roster's "what's in it" column, re-rendered as the ledger's border subtitle), and it read
    "one population" for `pbmc3k_annotated.h5ad` and `tree8.h5ad` while the app plotted their
    eight groups. `describe` — the sentence the rich shell prints — was making the same denial,
    but the TUI is the demo's front door and never calls it, so fixing only that fixed a string
    nobody saw.

    Kept SHORT ("8 labelled groups", not the names): `tui.roster.columns_that_fit` measures
    content width and drops the rightmost column when it does not fit, so a long phrase here
    costs the column at narrow terminal widths. The names are `describe`'s job.

    Returns the shape phrase when there is neither (every bundled dataset that predates the
    topology field, and every dropped file with no group column — a dropped file has no
    declaration at all, only the shape `read_data` inferred).
    """
    words = [_TOPOLOGY_PHRASE[t] for t in (topology or ()) if t in _TOPOLOGY_PHRASE]
    if words:
        return ", ".join(dict.fromkeys(words))   # dedupe, keep declaration order
    if groups:
        return f"{len(groups)} labelled groups"
    return _SHAPE_PHRASE.get(shape, _SHAPE_PHRASE["unknown"])



#: What a dataset with no gene axis shows where a QC number would go. A DASH AND NEVER A ZERO:
#: `qc.facts` already draws that distinction one level down — "not the same as it being zero"
#: — and a `0.0 %` in a mitochondrial column is a claim about biology where the truth is a
#: claim about the file.
QC_UNKNOWN = "\u2014"


def glyphs(topology: "Iterable | None" = None) -> str:
    """The declared topology, drawn — the same fact `contents` says in words.

    A dataset that declares two regimes gets two marks (`torus` is a closed loop AND a curved
    sheet), deduped and in declaration order, exactly as `contents` joins its phrases. A word
    the glyph table does not know is SKIPPED rather than drawn as a placeholder, for the reason
    `contents` skips it too: a mark that stands for "we have no mark" is worse than a gap.

    Empty for every dropped file, and that is the informative case rather than a missing one —
    a file someone dragged in declares no topology at all, so its shape is precisely what has
    not been established yet. The roster draws no geometry column for those rows instead of a
    column of blanks.
    """
    marks = [ASSUMPTION_GLYPH[t] for t in (topology or ()) if t in ASSUMPTION_GLYPH]
    return " ".join(dict.fromkeys(marks))


def qc_cells(facts: "dict | None") -> dict:
    """`qc.facts` rendered for a table — a presentation table over QC's fact NAMES.

    The same relationship `MISSING_FACT` has to `vocab.facts()`: the keys here are the keys QC
    already emits, and the only thing added is how to say them. A surface that formatted these
    itself would be a second account of numbers `qc` computes once.

    Every value is `QC_UNKNOWN` when the fact is absent, because absence here has a meaning —
    no gene axis to count — and printing `0` for it would be a measurement nobody made.
    """
    f = facts or {}
    per_cell = f.get("qc_median_genes_per_cell")
    mito = f.get("qc_pct_mito_median")
    return {
        "per_cell": QC_UNKNOWN if per_cell is None else f"{per_cell:,.0f}",
        "mito": QC_UNKNOWN if mito is None else f"{mito:.1f} %",
        "drops": qc_drops(f),
    }


def qc_drops(facts: "dict | None") -> str:
    """What a standard filter would remove, naming only the halves that are not zero.

    THE COLUMN IS TOO NARROW TO SPEND ON A ZERO. Measured on `pbmc3k_raw.h5ad`: 0 cells fall
    below 200 genes and 19,024 of 32,738 genes appear in fewer than 3 cells, so a row that
    printed both would spend most of its width saying nothing happened.

    The two CELL cuts are never summed. `qc_would_drop_cells` (too few genes) and
    `qc_would_drop_cells_mito` (too much mitochondrial signal) are two thresholds over one
    axis and a cell can fail both. Nor is their maximum a total: two disjoint one-cell cuts
    produce two counts of one. The facts carry counts, not masks, so label each cut separately.
    """
    f = facts or {}
    if f.get("qc_n_cells") is None:
        return QC_UNKNOWN
    # THE CUTS ARE DEFINED OVER COUNTS, so they say nothing about a matrix that does not hold
    # any. `qc.facts` sets `qc_input` exactly when the values are already normalised or
    # transformed, and there "fewer than 200 genes" is not a shallow cell — it is a different
    # quantity being compared to a threshold meant for library sizes. Measured on this
    # checkout: it reports 800 of 800 cells for `tree8.h5ad` and 2,100 of 2,638 for
    # `pbmc3k_processed.h5ad`, both of which are fine files. A number that alarming, that
    # wrong, is worse than no column.
    if f.get("qc_input"):
        return "not counts"
    parts = []
    genes = f.get("qc_would_drop_genes") or 0
    cells = int(f.get("qc_would_drop_cells") or 0)
    mito = int(f.get("qc_would_drop_cells_mito") or 0)
    if genes:
        parts.append(f"{genes:,} genes")
    if cells:
        parts.append(f"{cells:,} cells (low genes)")
    if mito:
        parts.append(f"{mito:,} cells (high mito)")
    return " · ".join(parts) if parts else "nothing"


class QcConcern(NamedTuple):
    """One thing a standard filter would remove, and the analyses it would spoil.

    `count` over `total` is the share; above `share` it is worth a scientist's attention.
    `facts` are the calculus words the concern is computed INTO — an analysis is implicated
    when its steps need or produce one of them. That is the whole join: no recipe is named,
    so a recipe added tomorrow is covered by what its steps declare.
    """
    count: str
    total: str
    cut: str
    share: float
    facts: frozenset
    sentence: str


#: THE ONE THAT EARNS THE COLUMN. `pbmc3k_raw.h5ad` carries 19,024 of 32,738 genes in fewer
#: than 3 cells; a rank over that axis ranks noise, and a partition judged over it is judged
#: over noise. That is exactly the call an expert makes and a retrieval engine does not — so
#: it is the caution this pane exists to draw.
#:
#: SHALLOW CELLS ARE DELIBERATELY NOT HERE YET. `qc_would_drop_cells` reports 2,356 of 2,638
#: for `pbmc3k_annotated.h5ad`, which is subset to 2,000 highly-variable genes: the 200-gene
#: cut assumes the full gene space and a cell in a 2,000-gene matrix cannot clear it. A
#: concern that fired on the demo's own dataset for a threshold artefact would teach the reader
#: to ignore the column. It wants a cut that adapts to the gene space (Heumos et al. 2023,
#: MAD-based), and that is its own change.
QC_CONCERNS: tuple = (
    QcConcern("qc_would_drop_genes", "qc_n_genes", "qc.min_cells", 0.25,
              frozenset({"genes", "clusters"}),
              "{share:.0%} of genes sit in <{cut} cells — ranks and partitions run over noise"),
    QcConcern("qc_would_drop_cells_mito", "qc_n_cells", "qc.max_pct_mito", 0.10,
              frozenset({"embedding"}),
              "{share:.0%} of cells >{cut}% mito — dying cells pull the layout"),
)


def concerns(recipe: dict, measured: "dict | None") -> list[str]:
    """What the measurement says about THIS recipe's result. Advice, never legality.

    Derived from the facts the recipe's steps declare (`vocab.recipe_needs` and
    `vocab.step_provides`), joined to `QC_CONCERNS` — never from a list of names, which the
    catalog-derivation guards forbid and which would silently exclude the next recipe added.

    A recipe whose steps PRODUCE nothing is exempt: it has no derived result to spoil, and the
    conditioning recipes are what a reader would run to address a concern, not victims of it.

    `None` means no measurement was taken and returns nothing; a measurement over a matrix that
    is not counts (`qc_input`) also returns nothing, because every cut here is defined over
    counts and firing them over normalised values reports a threshold artefact as a finding.
    """
    if not measured or measured.get("qc_input") or measured.get("qc_n_cells") is None:
        return []
    from manyruns import vocab

    steps = recipe.get("steps") or []
    produces: set = set()
    for step in steps:
        produces |= set(vocab.step_provides(step))
    if not produces:
        return []
    implicated = set(vocab.recipe_needs(recipe)) | produces

    out = []
    for c in QC_CONCERNS:
        if not (implicated & c.facts):
            continue
        count, total = measured.get(c.count), measured.get(c.total)
        if not total or count is None:
            continue
        share = float(count) / float(total)
        if share > c.share:
            out.append(c.sentence.format(share=share, cut=measured.get(c.cut)))
    return out


def qc_thresholds(facts: "dict | None") -> str:
    """The cuts the drop counts were measured against, in the reader's words.

    IT MUST BE POSSIBLE TO SEE THE THRESHOLD. A drop count is meaningless without it, and the
    failure is not hypothetical: `pbmc3k_annotated.h5ad` holds integer counts but is subset to
    2,000 highly-variable genes, so a cell carries ~132 of them and the 200-gene cut — which
    assumes the full ~32,000-gene space — reports 2,356 of 2,638 cells. The number is correct
    under the declared threshold and useless without it, so the threshold travels with it.

    `qc.facts` carries these three keys for exactly this reason; qc.py calls them declared
    rather than settled, and Heumos et al. 2023 prefers MAD-based outliers to fixed cuts.
    """
    f = facts or {}
    if f.get("qc_n_cells") is None:
        return ""
    return (f"filter: <{f.get('qc.min_genes')} genes/cell, "
            f"gene in <{f.get('qc.min_cells')} cells, "
            f">{f.get('qc.max_pct_mito')}% mito")


# ── the record, for a reader who was not in the room ─────────────────────────
#: `qc_cells`' three keys with the roster's own column headings, in the roster's own order
#: (`tui/roster.COLUMNS`). Written out here rather than imported because the dependency runs
#: one way — every screen reads this module and this module reads no screen — and because a
#: prompt is lines rather than a table, so what it borrows is the wording, not the layout.
_QC_PROMPT_LABELS: tuple[tuple[str, str], ...] = (
    ("per_cell", "genes/cell"), ("mito", "% mito"), ("drops", "filter drops"),
)


def _claims_line(claims: Any) -> str:
    """A recipe's `claims:` said the way the ledger says it — the word AND the commitment.

    Both halves, because they have different readers. `clusters` is the vocabulary word a
    record can be keyed on and the one `ASSUMPTION_GLYPH` is a table over; "assume separate
    groups" (`tui/ledger._heading`, verbatim down to the verb) is what tells a reader what
    running the row would commit them to. A word the topology vocabulary does not know is
    printed alone rather than dressed in an invented phrase, for `glyphs`' reason: a stand-in
    for "we have no phrase for this" is worse than the bare word.
    """
    if not claims:
        return "none — condition the data first"
    phrase = _TOPOLOGY_PHRASE.get(str(claims))
    return f"{claims} — assume {phrase}" if phrase else str(claims)


def record_for_prompt(*, entry: "dict | None" = None, recipe: "dict | None" = None,
                      steps: "Iterable | None" = (), tunables: "Iterable | None" = (),
                      constraints: "dict | None" = None,
                      said: "Iterable | None" = (), concerns: "Iterable | None" = ()) -> str:
    """One run as plain lines — the whole record, for a model that is handed it in a single call.

    THE MODEL READS WHAT THE PERSON READ. Every value here is one a screen already draws, and
    each arrives from the function that draws it: `qc_cells` and `qc_thresholds` (the roster's
    QC columns), the `StepView`s the steps pane renders, the concerns the ledger drew. Nothing
    is re-measured on the way past. A second reading that disagreed with the pane — a QC pass
    over a file that has since been filtered, say — would answer about a run nobody watched,
    which is the failure the "no screen computes a fact narrate does not expose" rule exists to
    prevent, arriving from the other direction.

    **A section the screen collapses is absent here, not empty.** `dataset`, `recipe` and
    `steps` are always written, because their panes are always on screen: a fact nobody
    measured is `QC_UNKNOWN` (a dash and never a zero) and a run with no rows gets the steps
    pane's own sentence. `tunable`, `said` and `caution` appear only when they hold something,
    exactly as the widgets holding them collapse to nothing — and `caveats_view`'s argument for
    that, "a placeholder here would be a claim", is stronger for this reader than for a person:
    a person skims an empty heading, a model answers under it.

    **Deterministic, and that is the whole point of the bytes.** No clock, no path, no `id()`,
    no iteration over anything the caller did not put in order; the only dict read is by key.
    So the same inputs render the same bytes and `context_sha256` is a name for a specific
    prompt. That is what lets the ask event carry the hash INSTEAD of the text (spec §2): a
    corpus that stored every prompt would be the same few hundred tokens thousands of times,
    and a later reader rebuilds them from the record and checks the hash.

    Plain values, keyword-only, no Textual import — and none in `tui.state` either, where the
    `StepView`s come from — so a prompt can be rendered and asserted on with no terminal.
    Keyword-only because four of the six arguments are sequences of strings or tuples: a caller
    that swapped two positionally would render a plausible record of the wrong run in silence.

    **Two departures from spec §3's list, and both are that same rule enforced.** The list is
    "name, state, seconds, and every delta, every number". First, `detail` is rendered as well:
    measured on this tree, three of the clustering recipe's six steps settle as `skipped` under
    `run_inproc`, and the only thing on screen saying WHY is the detail line under each of them
    ("no in-process implementation"). Without it a model is asked whether to trust a step it
    cannot see did not run, so the list gained an item rather than the record losing a
    sentence. Second, the state column carries `StepView.word` and not the raw state, because
    `step_mark` draws the outcome `reported` as "ran" — see the line itself.

    Measured on this tree: `runner.run_inproc` over `pbmc3k_raw.h5ad` (2,700 × 32,738), the
    four-step embedding recipe with its real QC facts, one tunable (`knn=40`) and one said
    line (`kept attempt 1 as phate@1.png`) — 29 lines and 749 bytes, four `StepView`s of which
    one carried five deltas. The two strings are quoted because the byte count is theirs: PHATE
    is stochastic, so its digits move the total by a character or two between runs while the
    line count does not move at all. A whole run is the size of a short email, which is what
    spec §1's "one call, not a loop" rests on: there is nothing in here for a query tool to
    select among.
    """
    lines: list[str] = []

    e = entry or {}
    lines.append(f"dataset: {e.get('name') or QC_UNKNOWN}")
    lines.append(f"  size: {e.get('size') or QC_UNKNOWN}")
    lines.append(f"  what's in it: {e.get('contents') or QC_UNKNOWN}")
    cells = qc_cells(e.get("qc"))
    lines += [f"  {label}: {cells[key]}" for key, label in _QC_PROMPT_LABELS]
    # The thresholds line is DROPPED rather than dashed when there is no gene axis to count,
    # because `qc_cells` has already said so three times over — and `qc_thresholds`' own
    # argument is that a drop count without its cut is meaningless, not that a cut without a
    # drop count is worth printing.
    cuts = qc_thresholds(e.get("qc"))
    if cuts:
        lines.append(f"  {cuts}")

    r = recipe or {}
    lines.append("")
    if not r:
        lines.append(f"recipe: {QC_UNKNOWN}")
    else:
        lines.append(f"recipe: {r.get('name') or QC_UNKNOWN}")
        lines.append(f"  claims: {_claims_line(r.get('claims'))}")
        chain = [str(s.get("name") or "?") for s in (r.get("steps") or [])]
        lines.append(f"  steps: {' → '.join(chain) if chain else QC_UNKNOWN}")

    lines.append("")
    lines.append("steps:")
    rows = list(steps or ())
    if not rows:
        # `steps_view`'s sentence, verbatim: an engine that declared nothing and reported
        # nothing must not read as a clean run here either.
        lines.append("  no steps declared, and none reported")
    else:
        # The columns the steps pane lays out, in its order and its widths — name, word,
        # duration (`tui/run.steps_view`, the same `{:<9}` state column) — so someone reading a
        # traced prompt beside the screen it came from is reading one thing twice, not two
        # accounts. `word` AND NOT `state`, which is the whole reason `StepView` carries both:
        # `step_mark` draws the outcome `reported` as "ran" and the live `running` as
        # "running…", so the raw state would tell a model a step was "reported" where the
        # person watching was told it ran — a second vocabulary for how a step ended, invented
        # in the one place whose entire claim is that the two read the same record.
        # `seconds` is None until a step settles and prints as nothing rather than `0.00s`,
        # which would report a finished step that took no time; the word beside it is what says
        # why the column is empty.
        width = max(len(str(row.name)) for row in rows)
        delta_width = max((len(d.label) for row in rows for d in row.deltas), default=0)
        for row in rows:
            secs = "" if row.seconds is None else f"{row.seconds:.2f}s"
            lines.append(f"  {str(row.name):<{width}}  {str(row.word):<9}{secs:>7}".rstrip())
            # `detail` IS NOT ON §3's LIST, and it is here anyway — see the deviation note in
            # this function's docstring. In full, `steps_view`'s call: `run_panel` clips at 58
            # because a plain line cannot re-flow, and a prompt has no width to re-flow into,
            # so clipping here would drop the second half of the one sentence that says why a
            # step did not run.
            if getattr(row, "detail", ""):
                lines.append(f"      {row.detail}")
            for delta in row.deltas:
                # An absence is LABELLED before its reason is given, `geometry_view`'s call: in
                # a column of `3.394`s a lone "returned nan" reads as a value.
                value = delta.value if delta.measured else f"not measured — {delta.value}"
                lines.append(f"      {delta.label:<{delta_width}}  {value}")

    tuned = list(tunables or ())
    if tuned:
        # Use the prompt's key=value spelling, so a displayed value can be typed back.
        lines.append("")
        lines.append("tunable:")
        lines += [f"  {name}={value}" for name, value in tuned]

    if constraints:
        # Separate from `said`: its three-line window can evict a bounds note before an ask.
        # These caps were resolved against the tune loop's retained INPUT, not its output.
        import json

        lines += ["", "constraints (partial recipe caps; unknown means manyruns asserts nothing):",
                  "  " + json.dumps(constraints, sort_keys=True)]

    # EVERY LINE HANDED IN IS RENDERED, and the cap is the CALLER's. The run screen keeps
    # `self.said` unbounded and slices to `SAID_LINES` only when it paints, so a caller that
    # passes the whole list serializes the whole list. Capping here would make this function
    # decide what the record contains, which is the screen's decision — it knows what it showed.
    spoken = [str(s) for s in (said or ())]
    if spoken:
        lines.append("")
        lines.append("said:")
        lines += [f"  {s}" for s in spoken]

    warned = [str(c) for c in (concerns or ())]
    if warned:
        # The ledger's word for the same sentences (`LedgerRow.caution`), and they are passed
        # IN rather than recomputed from `recipe` + the QC facts, which are both to hand: what
        # belongs in the record is the caution the screen actually drew.
        lines.append("")
        lines.append("caution:")
        lines += [f"  {c}" for c in warned]

    return "\n".join(lines) + "\n"


def context_sha256(context: str) -> str:
    """The hash the ask event carries in place of the prompt itself.

    Over UTF-8 bytes, full 64 hex digits, no truncation — a short prefix would save 48
    characters on a line that is already carrying an answer, and cost the one property the
    field has: that two events with one hash were asked over one record.

    `hashlib` is imported here rather than at module top on this file's usual grounds. Measured
    with `-X importtime`, and as a RANGE like the one `geometry_delta_rows` quotes below,
    because a single figure from a single run does not reproduce: 0.7-0.9 ms to import
    (`_hashlib` pulls OpenSSL) against `narrate`'s own 9.4-13.3 ms — 6-9% on every launch,
    every `manyruns recipes` and every test that reads a phrase, paid for a function only the
    ask path calls.
    """
    import hashlib

    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def refusal(user_text: str, obs: Observation) -> Optional[str]:
    """If the user asks for a trajectory on data with no time axis, decline + reframe. Else None."""
    t = (user_text or "").lower()
    wants_trajectory = any(w in t for w in _TRAJECTORY_WORDS)
    if wants_trajectory and obs.shape in ("case-control", "single", "unknown"):
        return (
            "There's no progression to show here — these cells were captured at one moment"
            + (", from different people" if obs.shape == "case-control" else "")
            + ". There's no clock in this data. I could force a trajectory, but it would draw a "
            "confident story of change that the data can't support — a line invented from noise. "
            "I'd rather not hand you that.\n  What I think you want: "
            + ("how the groups differ from each other." if obs.shape == "case-control"
               else "the structure of the data, mapped honestly.")
        )
    return None


def interpret(user_text: str, obs: Observation) -> Optional[str]:
    """Map a free-typed question to a recipe (rule-based; the LLM router is the later upgrade)."""
    t = (user_text or "").lower()
    if any(w in t for w in _CONTRAST_WORDS) and obs.conditions:
        return "contrast"
    if any(w in t for w in _TRAJECTORY_WORDS) and obs.shape == "time-course":
        return "cflows"
    if "map" in t or "embed" in t or "explore" in t or "structure" in t:
        return "embed"
    return None


#: Outcome → the glyph and the word this panel shows. A PRESENTATION table over
#: `watch.OUTCOMES`, not a second declaration of the vocabulary: the keys must be exactly
#: those four, and the glyph/word pair is the only fact added here.
#:
#: Checked rather than derived — the pair cannot be computed from the name, so a comprehension
#: over `OUTCOMES` would still need this literal per outcome. What keeps it honest is
#: `test_every_renderer_covers_the_whole_outcome_vocabulary`, which fails in both directions
#: (a missing key and an extra one) for this table AND for `shell._OUTCOME_STYLE`. Before it,
#: the two tables and two dead tuple declarations stated the same four names four times with
#: nothing comparing any pair, so a new outcome would have rendered here and as `?` in rich —
#: `run_panel` and `shell.render_result` are two renderings of ONE record and must not
#: disagree about what an outcome is called.
_OUTCOME_MARKS = {
    "ok": ("✓", "ok"),
    "skipped": ("⊘", "skipped"),
    "error": ("✗", "error"),
    "reported": ("•", "ran"),
}


#: The two LIVE states, which are not outcomes and must not be added to `_OUTCOME_MARKS`.
#: `runner.apply_step` says it in the code: "`running` is a live state, not an outcome … the
#: outcome vocabulary must keep meaning 'how it ended'", and `_settle` strips both fields off
#: every finished record. `test_watch.test_every_renderer_covers_the_whole_outcome_vocabulary`
#: asserts `set(_OUTCOME_MARKS) == set(watch.OUTCOMES)` exactly, so merging the two tables
#: would fail it — correctly.
#:
#: `queued` is what a step that has NO record yet is, and it exists only where the declared
#: recipe is known: the live overlay. `shell._PENDING` is the rich surface's copy of this pair;
#: the two are not unified here because `shell` is a separate surface with its own colours, and
#: `_progress_panel` is where that unification belongs.
_LIVE_MARKS = {
    "queued":  ("·", "queued"),
    "running": ("◐", "running…"),
}


def step_mark(rec: Optional[dict]) -> tuple[str, str]:
    """One step record → `(glyph, word)`, covering queued and running as well as the outcomes.

    `rec is None` means the step is declared and has not started; `rec["state"] == "running"`
    means `apply_step` has reported it and not yet settled. Everything else reads the outcome
    through `_OUTCOME_MARKS`, so there is one table for "how it ended" and `run_panel` and a
    live view cannot name an outcome differently.
    """
    if rec is None:
        return _LIVE_MARKS["queued"]
    if rec.get("state") == "running":
        return _LIVE_MARKS["running"]
    outcome = rec.get("outcome")
    return _OUTCOME_MARKS.get(outcome, ("?", str(outcome)))


def _clip(text: Any, limit: int = 58) -> str:
    """One-line detail. Long messages are truncated, never wrapped — the full text is in
    `summary.md`, and a panel that reflows is harder to scan than one that elides."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ── the g-vector, sorted by what kind of fact each key is ────────────────────
def _size_keys() -> frozenset:
    """The size descriptors — shown once as the before/after arrows, never again as bare
    values. Read from the ONE place they are declared, `runner.GVECTOR_SIZE`.

    They used to be declared twice: an ordered tuple in the producer and an independent
    frozenset here, identical as sets with nothing keeping them equal. The producer owns the
    vocabulary, so this is a view of the producer's list.

    It reads `GVECTOR_SIZE`, NOT `GVECTOR_CORE`. The two are the same four keys today and
    still must not be collapsed: `GVECTOR_CORE` means "the keys every g-vector carries" and
    `GVECTOR_SIZE` means "the keys that describe the run's size rather than its geometry".
    This function suppresses its keys from the measurement list, which is right for a size
    descriptor and wrong for a geometry number. Reading the core set here made extending it
    with an always-computed geometry key delete that key from the panel with the full suite green, because a test
    deriving both sides of its assertion from one tuple cannot see the difference.
    `test_a_core_key_that_is_not_a_size_key_still_reaches_the_panel` is what now fails first.

    Imported inside the function rather than at module scope to keep a renderer free of a
    module-scope dependency on the pipeline. The saving is real but small: measured with
    `-X importtime`, `manyruns.narrate` costs 9.5-14.2 ms and `manyruns.pipeline.runner`
    adds 2.8-4.4 ms on top of it. (An earlier version of this note claimed 15 ms and
    attributed the cost to the front door; both were wrong — 15 ms is runner's standalone
    cumulative time, which double-counts what narrate already pulled in, and `shell` imports
    narrate function-locally, so no launch pays either figure.) The import is safe from
    anywhere, including the stackless path: verified by importing `manyruns.pipeline.runner`
    with numpy, scipy, sklearn, matplotlib, torch and rich all blocked, because the pipeline
    package keeps every heavy import function-local."""
    from manyruns.pipeline.runner import GVECTOR_SIZE

    return frozenset(GVECTOR_SIZE)


class GeometryRow(NamedTuple):
    """One line of the geometry panel.

    `measured=False` means the metric was DECLARED and could not be computed: `value` is then
    the reason, in prose, not a number. Renderers must keep the two visually apart — a
    right-aligned column of numbers with one sentence in it reads as a number until you look
    twice."""

    label: str
    value: str
    measured: bool = True


class GeometrySection(NamedTuple):
    """Rows that are the same KIND of fact, plus the caption saying which kind."""

    title: str
    note: str
    rows: list


def geometry_sections(g: dict) -> list:
    """The g-vector split into the different kinds of thing it holds.

    Producers annotate settings and measurements in `_provenance`, including their evaluation
    stage. A dotted key alone says neither: `pca.trustworthiness` is an engine measurement,
    while `filter_genes.min_cells` is a threshold. Old dotted keys without provenance are
    displayed with an explicit unrecorded stage, never under a settings claim.

    **Absences are rows, not gaps.** `suite.measure` declares every metric whether or not it
    could be computed — that design is what tells "no engine" apart from "no embedding" apart
    from "it raised" — and a renderer that skips `None` throws all of it away: the same run
    showed `suite_measured 10 / suite_declared 12` with no way to learn which two were missing.
    They are listed with the reason from `<name>_note`. On that run the two are
    `geodesic_distance_correlation` (manylatents returns NaN unless the dataset exposes
    `get_gt_dists`, and `suite._Ambient` deliberately does not) and `kernel_sparsity` (needs a
    fitted `LatentModule` to read a kernel matrix off; the suite passes only an array). Both
    are structural — verified by calling the registry directly — not flaky.

    **Still no invented baseline.** Only `cells` and `dimensions` are shown as changes, because
    only they have a real before and a real after in the g-vector. Everything else is a value
    this run produced and is listed as one.

    **Nothing here is a finding**: the measured section carries `suite_null` verbatim as its
    caption, which is the one place `vocab.NULL_KIND`'s rule is stated — an embedding-derived
    statistic needs a `data` null, and none is implemented.
    """
    g = g or {}
    moved: list = []
    if g.get("n_samples") and g.get("n_embedded"):
        arrow = "→" if g["n_samples"] == g["n_embedded"] else "⇢"
        moved.append(GeometryRow(
            "cells", f"{int(g['n_samples']):,} {arrow} {int(g['n_embedded']):,}"))
    if g.get("n_features") and g.get("final_dim"):
        moved.append(GeometryRow(
            "dimensions", f"{int(g['n_features']):,} → {int(g['final_dim']):,}"))

    measured: list = []
    settings: list = []
    staged: dict = {}
    setting_stages: set = set()
    bookkeeping: list = []
    size_keys = _size_keys()   # resolved once per panel, not once per key
    for key, value in g.items():
        # `_note` is the REASON carried by its base key, and `suite_null` is the caption below.
        # Neither is a row of its own; a `_note` whose base key is absent says nothing at all.
        if key in size_keys or key.endswith("_note") or key == "suite_null":
            continue
        provenance = g.get("_provenance", {}).get(key, {})
        row = _geometry_row(key, value, g)
        if provenance.get("kind") == "setting" and isinstance(value, (str, bool)):
            row = GeometryRow(key, str(value))
        if row is None:
            continue
        if key.startswith("suite_"):
            bookkeeping.append(row)
        else:
            stage = provenance.get("stage") or "stage unrecorded"
            if provenance.get("kind") == "setting":
                settings.append(row)
                setting_stages.add(stage)
            elif stage not in ("final embedding", "stage unrecorded"):
                staged.setdefault(stage, []).append(row)
            elif "." in key and stage == "stage unrecorded":
                staged.setdefault("stage unrecorded (legacy)", []).append(row)
            else:
                measured.append(row)

    sections = [
        GeometrySection("input → output", "the only real before/after here", moved),
        GeometrySection("final measurements", str(g.get("suite_null") or ""), measured),
        *(GeometrySection("step measurements", stage, rows) for stage, rows in staged.items()),
        GeometrySection("settings", "declared or effective configuration · "
                        + ", ".join(sorted(setting_stages)), settings),
        GeometrySection("bookkeeping", "about the measuring, not about the data", bookkeeping),
    ]
    return [s for s in sections if s.rows]


def _geometry_row(key: str, value: Any, g: dict):
    """One key → a row, or None if it is not renderable as a measurement."""
    if value is None:
        note = g.get(f"{key}_note")
        return GeometryRow(key, _clip(note, 44) if note else "no reason recorded", measured=False)
    # bools are flags (`composition_absent_in_one`), not quantities; strings and lists
    # (`composition_between`, `pseudotime_range`) are not scalars a column can align.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return GeometryRow(key, f"{value:.4g}")


def geometry_delta_rows(rec: dict) -> list:
    """What ONE step did to the geometry — the runner's `rec["geometry"]`, `{metric: [from, to]}`.

    This is the only delta in the product that is not invented. `geometry_sections` refuses to
    subtract a baseline because it has none; here both numbers were measured, on the same
    metric, either side of the same step.

    `from` is None when there was no prior embedding. That is an APPEARANCE, not a change, and
    it renders `— → 3.394`: treating the absent side as zero would report a 3.394-unit move
    that nothing measured. The key is absent on steps that changed no embedding, so a step with
    no rows here is a step that moved no geometry, not a step nobody looked at.
    """
    geometry = rec.get("geometry") if isinstance(rec, dict) else None
    if not isinstance(geometry, dict):
        return []
    rows: list = []
    for name, pair in geometry.items():
        # the contract is a two-element [from, to]; anything else is not renderable, and
        # guessing at it would put a number on screen that no one measured
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        before, after = _as_float(pair[0]), _as_float(pair[1])
        if before is None and after is None:
            continue
        rows.append(GeometryRow(str(name), f"{_side(before)} → {_side(after)}"))
    return rows


def _side(v: Optional[float]) -> str:
    """One end of a `[from, to]` pair. `—` for an absent side, never `0`."""
    return "—" if v is None else f"{v:.4g}"


def run_panel(results: dict) -> str:
    """What the RUN did — one line per step, from the step record. A reader, not a judge.

    This deliberately makes no claim about the user's data. Every line is a fact about
    execution (did it run, how long, why not), which is why it cannot be wrong — the four
    readouts that tried to adjudicate whether structure was real all failed against a
    matched control, and this is what is left that is honest.

    Two things it says out loud that the product used to hide:
      * ATTEMPTED vs EXECUTED. `trace` has one entry per declared step whether or not any
        of them ran, so `manyruns run` could not distinguish three steps running from
        three steps failing. The record can.
      * whether the run is reproducible at all. the in-process loop threads no seed (issue #29),
        so those runs cannot be repeated — an absent seed is reported, never invented.

    Unlike :func:`narrate`, this names steps. That is not a violation of the no-algorithm
    rule but the other side of it: `narrate` speaks about the user's FINDINGS and must not
    make them learn an algorithm; this speaks about the RUN, and when a step fails the name
    is the identifier they need to find it in `summary.md`.

    Degrades rather than lies: engines that report no step record (the mock, and the learner's engine,
    which synthesises its trace from the recipe's declared steps) say so instead of
    presenting a declaration as an execution.

    It also carries the geometry — per-step deltas under their step, then the sectioned
    g-vector — because `shell.render_result` delegates here whenever rich is absent or output
    is piped, and a fallback that showed strictly fewer facts would make the two renderings
    two different accounts of one run. Same source (`geometry_sections`,
    `geometry_delta_rows`), same clipping, different ink."""
    steps = results.get("steps") or []
    recipe = results.get("recipe") or "(adaptive)"
    engine = results.get("engine") or "?"
    head = f"── {recipe} · engine={engine} " + "─" * 24
    lines = [head]

    if steps:
        width = max(len(str(s.get("name") or "?")) for s in steps)
        for s in steps:
            # through `step_mark`, not the table directly, so the plain surface and the TUI's
            # live overlay cannot name one outcome two ways. Every record here has settled
            # (`_settle` ran), so this is the `_OUTCOME_MARKS` branch and the rendering is
            # unchanged — which `tests/test_run_panel.py` and
            # `test_watch.test_every_renderer_covers_the_whole_outcome_vocabulary` prove.
            glyph, word = step_mark(s)
            # one line per step: a 300-character traceback message would destroy the layout
            # and tell the user nothing they can act on. `summary.md` keeps it in full.
            detail = f"  {_clip(s['detail'])}" if s.get("detail") else ""
            secs = f"{float(s.get('seconds') or 0.0):6.2f}s"
            lines.append(f"  {glyph} {str(s.get('name') or '?'):<{width}}  {word:<8}{secs}{detail}")
            for row in geometry_delta_rows(s):
                lines.append(f"      {row.label:<24} {row.value}")
        ran = sum(1 for s in steps if s.get("outcome") == "ok")
        total = sum(float(s.get("seconds") or 0.0) for s in steps)
        lines.append("")
        lines.append(f"  {ran} of {len(steps)} steps produced a result · {total:.2f}s")
    elif results.get("trace"):
        for entry in results["trace"]:
            lines.append(f"  · {entry}")
        lines.append("")
        lines.append(f"  {len(results['trace'])} steps declared. This engine reports no "
                     "per-step outcome,")
        lines.append("  so none of the above is evidence that anything ran.")
    else:
        lines.append("  (no steps recorded)")

    lines.extend(_geometry_lines(results.get("g_vector") or {}))

    # Through `caveats()` rather than inline, so the app draws the same two sentences from the
    # same place. The rendering is byte-identical to what this function built itself, which
    # `tests/test_run_panel.py` proves unchanged.
    for caveat in caveats(results):
        lines.append(f"  {caveat.mark} {caveat.text}")
    return "\n".join(lines)


class Caveat(NamedTuple):
    """One thing a reader has to know about a run that its numbers do not say."""

    mark: str      # `ⓘ` a note about the route taken · `⚠` a warning about the run itself
    text: str


def caveats(results: dict) -> list[Caveat]:
    """What a run's numbers do not say — the two sentences every surface has to be able to draw.

    LIFTED OUT OF `run_panel` AT INTEGRATION, and the reason is the rule the TUI is built under:
    no screen may compute a fact `narrate` does not expose, so a screen that needed these had to
    either re-derive them from the results dict — a fourth account of one run — or find them
    here. `manyruns/tui/run.py`'s header flagged exactly this as a gap and named this fix.

    WHY IT MATTERS MORE THAN IT LOOKS. The first entry is normally
    `"engine=mock — every number here is invented; no data was read"`, and it is the only thing
    on screen that distinguishes a fabricated g-vector from a measured one — `app.DEV_ENGINES`'
    own comment says the mock's panel is "indistinguishable at a glance from a measured one".
    A surface that drops this list shows invented numbers with nothing saying so.

    Two kinds, deliberately marked apart:

      * `ⓘ` — a legal-but-unusual route to a fact. NOT a warning that something is wrong: the run
        is valid and it finished. It is reported because a reader who does not know the
        composition was off the beaten path cannot judge the result or repeat it, and because the
        alternative this replaced was refusing the composition outright — which prunes exactly
        the unusual moves that turn into method.
      * `⚠` — the run cannot be reproduced. the in-process loop threads no seed (issue #29), so those
        runs cannot be repeated; an absent seed is reported, never invented. Conditioned on
        `steps`, because a run that recorded no steps has nothing to reproduce.
    """
    out = [Caveat("ⓘ", str(c)) for c in (results.get("caveats") or [])]
    if results.get("seed") is None and (results.get("steps") or []):
        out.append(Caveat("⚠", "no seed recorded — this run cannot be reproduced exactly."))
    return out


def _geometry_lines(g: dict) -> list[str]:
    """The sectioned g-vector as plain text — `render_result`'s panel without the box drawing."""
    sections = geometry_sections(g)
    if not sections:
        return []
    width = max(len(r.label) for s in sections for r in s.rows)
    lines: list[str] = []
    for sec in sections:
        lines.append("")
        lines.append(f"  {sec.title}" + (f"  ({sec.note})" if sec.note else ""))
        for r in sec.rows:
            # an absence is labelled as one. Printing the bare reason in the value column
            # would let "returned nan" sit where a number goes and read like a result.
            shown = r.value if r.measured else f"not measured — {r.value}"
            lines.append(f"    {r.label:<{width}}  {shown}")
    return lines


def narrate(results: dict) -> str:
    """Turn a run's g-vector into a plain-English finding — the backmap, in the user's words.

    Narrates ONLY what was computed (no invented findings). Handles the real keys
    (`separation_silhouette`, `composition_*`) and the mock stand-ins."""
    g = results.get("g_vector") or {}
    lines: list[str] = []

    sep = g.get("separation_silhouette")
    if sep is None and "separation" in g:  # mock stand-in
        sep = _as_float(g.get("separation"))
    if sep is not None:
        sep_p = g.get("separation_silhouette__null_p")
        if sep_p is not None and sep_p > _ALPHA:
            lines.append(f"The groups do not separate beyond chance — {_score(sep)} out of 1, "
                         f"but shuffling the group labels does at least as well {_pct(sep_p)} "
                         f"(p={sep_p:.2f}). Read this as no difference.")
        else:
            strength = "a strong, real difference" if sep > 0.5 else (
                "a moderate difference" if sep > 0.2 else "only a weak difference")
            lines.append(f"The groups separate — {_score(sep)} out of 1 "
                         "(0 = groups sit on top of each other, 1 = fully apart). "
                         f"That's {strength}{_ranked(g.get('separation_silhouette__null_p'))}.")

    if g.get("composition_max_log2_shift") is not None:
        lines.append(_composition_sentence(g))

    # NO trajectory/direction readout. `granger` used to narrate "some signals lead others"
    # here off `granger_min_p`; the step is deleted because that
    # sentence fired on pure noise in 12/12 seeds at median p 2.9e-29. What replaces it is a
    # refusal, below — direction is not identifiable from a static snapshot at all, so on
    # every dataset this repo currently ships the honest answer is that it cannot be assessed.
    if _is_trajectory(results):
        lines.append(_no_direction(results))

    if not lines:  # embed / structure-only: mapped, nothing was compared
        # A CLAIM ABOUT THE RECIPE, NOT ABOUT THE DATA, and that is a correction rather than a
        # rewording. This used to end "no groups or time labels to test against" — a statement
        # about what the FILE carries, made from a dict that does not carry it. Measured on
        # `runner.run_inproc(X, embed, labels=<8 cell types>, label_kind="group")`: `labels` and
        # `label_kind` reach the run STATE and stop there — `runner._finalize` puts neither into
        # `results`, and `open_gvector` records only `n_samples`/`n_features` — so the results
        # dict this function reads holds zero information about labels, and the sentence fired
        # verbatim over `data/pbmc3k_annotated.h5ad`'s eight named immune types and
        # `data/tree8.h5ad`'s eight branches. What IS derivable here is that no readout was
        # produced, which is a fact about the steps that ran; so that is what it now says, and
        # it is true on every dataset the repo ships, labelled or not.
        dim = results.get("final_dim")
        lines.append("Mapped the structure" + (f" into {dim} dimensions" if dim else "") +
                     " — this recipe ran no comparison, so this is the shape itself, "
                     "laid out for you to read.")
    return "\n  ".join(lines)


#: Significance threshold for the empirical nulls. A readout above this is narrated as a
#: NON-finding, in words, rather than being silently dropped — "we looked and found nothing"
#: and "we never looked" are different sentences and the user needs to be able to tell them
#: apart.
_ALPHA = 0.05


def _pct(p: float) -> str:
    """An empirical p as a frequency, because that is what a permutation p literally is:
    the share of label shufflings that did at least as well as the real labels."""
    return f"~{p * 100:.0f}% of the time"


def _score(v: float) -> str:
    """Format a score without a signed zero — "-0.00 out of 1" reads as a real negative."""
    return f"{0.0 if abs(v) < 0.005 else v:.2f}"


def _ranked(p: Optional[float]) -> str:
    """The clause that turns a magnitude into a ranked magnitude."""
    if p is None:
        return " (unranked — no null was run, so treat it as a description, not a finding)"
    return f", and shuffled labels beat it in only {p * 100:.1f}% of draws"


def _composition_sentence(g: dict) -> str:
    """The product's most confident sentence, with the two things it used to omit.

    It used to read *"one population is ~434× more abundant"* with no null and no mention
    that the winning cluster was empty in one condition. Both are now said out loud:

    * the fold is a MAXIMUM over k clusters × condition pairs, so it is large on noise too
      — measured, it reached ~68× on sixty random points — which is why the p gates it;
    * when the cluster is empty in one condition the ratio is a lower bound set by how many
      cells were looked at, not a measured ratio, so it is stated as "none at all".
    """
    shift = float(g["composition_max_log2_shift"])
    p = g.get("composition_max_log2_shift__null_p")
    between = g.get("composition_between") or "the groups"
    first = between.split(" vs ")[0]

    if p is not None and p > _ALPHA:
        return ("No population shifts more than chance would produce — reshuffling which "
                f"samples are {first} does at least as well {_pct(p)} (p={p:.2f}). "
                "This is the shape of the data, not a difference between the groups.")

    if g.get("composition_absent_in_one"):
        body = (f"one population is present in {first} and essentially absent in the other")
    else:
        body = f"one population is ~{2 ** shift:.0f}× more abundant in {first} than the other"
    return (f"The biggest driver: {body}. A specific population expands while another "
            f"shrinks — not every cell changing a little{_ranked(p)}.")


#: Steps that lay cells out along a path. Membership here is what makes the direction
#: refusal fire, so it is a list of TRAJECTORY steps, not of every step a recipe may hold.
_TRAJECTORY_STEPS = ("mioflow", "cflows")


def _is_trajectory(results: dict) -> bool:
    """Did this run actually lay cells out along a path?

    Read from what RAN — the executed trace, plus the pseudotime the real engine records —
    never from the recipe NAME: a recipe whose trajectory step was skipped or errored has no
    path to talk about, and would otherwise be told it cannot have a direction for a path it
    never built. The trace is the authority because the mock writes the g-vector only for
    `analysis` steps, so a mock trajectory run has a trace but an empty g.
    """
    g = results.get("g_vector") or {}
    if "pseudotime_range" in g:
        return True
    trace = results.get("trace") or []
    return any(any(s in str(entry) for s in _TRAJECTORY_STEPS) for entry in trace)


def _no_direction(results: dict) -> str:
    """The sentence that replaced a false one.

    `granger` reported "some signals lead others — there's a directional order" whenever
    p < 0.05, which was every run: 12/12 rejections on pure noise, and the no-lead-lag case
    scoring MORE significant (median p 9.9e-38) than a real 20-step lead-lag. Deleting it
    leaves a gap, and the gap must be spoken rather than left silent — a user who asked
    "where are these cells heading" and gets no answer will assume the answer was boring,
    not that the question was unanswerable.

    It is unanswerable, and not for want of a better method: from a static snapshot,
    "different assumptions about the rates and location of cell entry and exit lead to
    fundamentally different inferences of the direction of cell progression" (Weinreb et al.,
    PNAS 115:E2467, 2018). Real time, splicing, metabolic labelling or lineage barcodes are
    what resolve it — so the refusal names them, making it a next step instead of a dead end.
    """
    return ("Which way along the path, I can't tell you — and neither can any method, from "
            "data caught at one moment. The layout shows cells are spread along a path; it "
            "cannot show which end is the start. Running it forwards and backwards fits this "
            "data equally well.\n  To get direction you need a clock in the data itself: "
            "samples collected at known times, RNA velocity (spliced/unspliced counts), "
            "metabolic labelling, or lineage barcodes. With any of those I can answer it.")


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
