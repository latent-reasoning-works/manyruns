"""One state lineage, driven one step at a time — the interactive half of the ONE step loop.

`serving.py` frames a manyruns "model" as a DRIVER LOOP — a `while` that repeatedly asks
for the next open-ended step. `Session` makes that loop *steppable*: it holds one running
state and applies one step per action. The human REPL (`manyruns init` → `manyruns>`) and,
later, the world-model policy both drive it the same way.

**This module implements no compute and no step loop.** It is a driver over
`pipeline.runner.apply_step`, which is also what the recipe loop (`runner._run_steps`) drives.
That is the whole point of the rewrite: there was a second implementation here, and it was
the weaker one. Measured on this checkout, `embed` (one `phate` step declaring
`n_components: 3`) over the same 120x12 array with the same seed, `engine=manylatents` —

    former session loop   10 top-level keys,  1 g-vector key, final_dim 2, no state/ folder
    runner.run_manylatents   18 top-level keys, 23 g-vector keys, final_dim 3, state/ + COMPLETE

— and the `final_dim` gap is not a reporting difference. `session.step` took a bare NAME, so
the recipe's declared `params` had no channel to travel down and the fit really did run at the
engine default of 2. An interaction surface whose unit is a string cannot carry params; the
seam takes a step DICT, so it can. After the rewrite both sides of that table are the runner's
column (re-measured in the same way — see `tests/test_session.py`).

Engines: whatever `runner.dispatch_for` has a table for — `real`, `manylatents`, `mock`.
A non-steppable engine says so rather than pretending, because it executes a whole recipe in one
call and reports no per-step outcome.

**One session is not necessarily one `run_id`.** A `run_id` names one STATE LINEAGE — one
chain of steps applied to one starting state — so a person who chains phate → mioflow →
another mioflow has ONE run of three steps, and a person who rewinds to phate's embedding and
takes a different next step has TWO, the second naming the first in `parent` (`branch`). An
atomic `manyruns run` is the degenerate lineage: issued all at once, no parent. That is the
only reading under which the artifact folder's `NN` stays a POSITION rather than becoming a
graph node, and under which `COMPLETE` keeps meaning "this folder is finished".
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from manyruns.pipeline import runner as _runner


def catalog_steps(config_dir: Optional[Path | str] = None) -> dict[str, dict]:
    """Every step any recipe declares — `catalog.known_steps`, re-exported for this surface."""
    from manyruns import catalog

    return catalog.known_steps(config_dir)


#: Groups a person is not offered as a MOVE, though every one of them still runs as part of a
#: recipe and is still typeable by name through `resolve_step`.
#:
#: They were absent from the menu until the cutover for an accidental reason — `catalog_steps`
#: only knows what some recipe declares, and no recipe declared one. Giving seven recipes a prep
#: block would have silently added five entries beside `phate` and `mioflow`, reversing spec
#: §13.3's decision by omission. Hence a declared exclusion rather than a lucky absence.
#:
#: The distinction is what the step is FOR. `phate` is a question a person asks of their data and
#: looks at the answer to. `normalize` is data conditioning — it produces no fact, its result is
#: only ever an input to the next step, and choosing it à la carte mid-session is how you get a
#: matrix normalized twice. A `probe` is the other direction: it interrogates a fitted model, so
#: offering it before anything is fitted lists a move that can only decline.
UNOFFERED_GROUPS = ("prep", "probe")


def offered(step: dict) -> bool:
    """Is this catalog step offered as a MOVE? — the one place the rule is written.

    Two surfaces read it: `available_actions` (the interactive menu, and through it the
    session banner and the `unknown action … try …` line) and `commands.step_commands`
    (the `offered` flag `render_help` groups `--help` and the REPL's `help` by). A second
    copy of `group not in UNOFFERED_GROUPS` is exactly how those surfaces came apart:
    measured on the bundled catalog before this existed, `--help` listed 23 steps under one
    undifferentiated `steps` heading while the banner offered 16, so `normalize` read as the
    same kind of thing as `phate` and nothing on the page said otherwise.
    """
    return step.get("group") not in UNOFFERED_GROUPS


def available_actions(engine: Optional[str] = None) -> list[str]:
    """The steps that can actually be dispatched, in catalog order.

    Was a hardcoded list of four names, which is the shape that does not survive a growing
    vocabulary: a step added to `configs/recipe/` was runnable by `manyruns run` and invisible
    in the session. Now it is the catalog (what steps exist) intersected with
    `runner.dispatchable` (what this engine can run), so adding a recipe adds its steps to the
    menu and no engine offers a step it would only decline.

    `engine=None` asks what steps EXIST rather than what one engine can run — that is the
    question `--help` and the command registry ask, and the answer must not change with the
    install. A named engine narrows it: measured on the bundled catalog 2026-08-17, `mock` and
    `manylatents` both answer with the same 16 as `None`, and the `_inproc` test substrate with
    11 (it resolves latent/analysis steps by name out of its own tables). A fourth row read
    "the learner's engine with none — it executes a whole recipe downstream"; that engine
    does not exist (§3.5), so every engine here is steppable and the zero case is now reached
    only by a name that is not an engine. The stale "the same four" this docstring used to
    claim predates the cutover's 23-step catalog.
    """
    steps = {n: s for n, s in catalog_steps().items() if offered(s)}
    if engine is None:
        return list(steps)
    return [name for name, step in steps.items()
            if _runner.dispatchable(engine, name, step.get("group"))]


def resolve_step(action: Any, engine: Optional[str] = None,
                 recipe: Optional[dict] = None) -> Optional[dict]:
    """A typed action (any casing/spacing/alias) → the step dict to apply, or None.

    Returns the CATALOG's step — group and declared params included — rather than a
    `(name, group)` pair, because the params are the half that used to be dropped.

    **THIS RECIPE'S OWN DECLARATION WINS OVER THE CATALOG'S.** A step NAME is global across the
    catalog, but a step's PARAMS belong to the recipe that declared them, and until the cutover
    nothing made those two facts disagree. Now they do: `catalog.known_steps` is
    first-declaration-wins over `discover_recipes()`, which is alphabetical, so with
    `archetypes` declaring `pca n_components: 10` and `cflows` declaring `50`, `archetypes` won
    and EVERY typed `pca` resolved to 10 — including one typed while walking `cflows`, whose own
    file says 50. Measured; the run then also stopped claiming the recipe's name, because the
    step it issued was not the step the recipe declares.

    Reading the session's recipe first answers the question a user is actually asking. "The next
    step" means the next step OF THIS ANALYSIS, and the catalog is the fallback for a name this
    recipe does not declare — which is what makes mix-and-match still work (`commands.py`'s
    model: any step is typeable in any session).

    `recipe=None` keeps the catalog fallback for callers without a recipe context.
    Typed REPL steps and the tune parser both pass the active recipe when they have one.
    """
    from manyruns import commands

    cmd = commands.resolve(action)
    name = cmd.name if cmd is not None and cmd.kind == "step" else None
    if name is None:
        return None
    declared = next((s for s in ((recipe or {}).get("steps") or [])
                     if s.get("name") == name), None)
    step = declared or catalog_steps().get(name)
    if step is None or (engine is not None
                        and not _runner.dispatchable(engine, name, step.get("group"))):
        return None
    return dict(step)


def resumable(out_dir: Path | str) -> Optional[dict]:
    """What a fresh `Session` over `out_dir` could pick up, or `None` if there is nothing to
    resume — a brand-new project, or a history that is mock-only and so never wrote a
    `state/` folder at all (`runner._mock_transform`'s docstring explains why).

    Returns `{"run_id", "arrays": {slot: array}, "index": {slot: int}, "origin": {slot:
    run_id}, "row_identity": {slot: identity_or_none},
    "steps": [(step, slot, index), ...]}`.

    **Folded over EVERY completed run, not just the newest one.** `open` mints a fresh random
    `run_id` per process (`runner.identity`), so "the last completed run" is a different
    folder every session, and each one's manifest only lists what THAT session's own steps
    touched. A session that resumes an embedding and re-runs `phate` (but not `dpt`) closes
    with a manifest holding `emb` alone; if it happens to close AFTER an earlier session that
    computed `pseudotime`, reading only the latest folder would make that pseudotime
    invisible — not because it was lost, but because a later, unrelated completion shadowed
    it. Measured: `phate → dpt` in session A, `phate` (embedding resumed, not re-dpt'd) in
    session B, both close clean — no crash — and session C's `open` used to come back with
    `emb` only, `discretize_time` refusing for want of a pseudotime that is sitting on disk
    two folders over. Folding oldest→newest and letting a later run's write win PER SLOT
    (rather than per run) is what makes every fact the project has ever completed resumable,
    the same "latest write wins" rule §`_persist`/`finish_carry` already apply within one
    run's own manifest, just no longer stopping at the run boundary.

    `origin` is why `run_id` alone stopped being enough to address a resumed value: two
    slots can now legitimately come from two different runs, and `app._build_session`'s
    `parent` (which lineage step this embedding's ancestry points at) has to name the run
    that ACTUALLY produced the slot being resumed, not just "the most recent completion."
    `run_id` itself is the latest run that actually CONTRIBUTED a resumed fact (not
    necessarily the latest completed folder — see below) — the session announcement's
    header — while `origin[slot]` is the per-fact answer.

    `arrays`/`index`/`origin` hold only the slots SOME run actually wrote — a lineage that
    only ever called `dpt` on top of an already-resumed embedding writes `pseudotime` but no
    second `emb` (`runner._persist` skips a slot a step didn't replace), so a resume can
    legitimately come back with one slot and not the other.

    Never raises. One run's `COMPLETE` marker existing but being unreadable (`finish()`
    writes it with a bare `write_text`, not `save_array`'s atomic tmp-then-rename, so a
    process killed mid-write can leave one truncated) drops only THAT run from the fold
    rather than failing `open` — the other completed runs still resume. A slot whose array
    file went missing after its run completed is dropped the same way, rather than resuming
    a stale in-memory value with no file behind it.
    """
    from manyruns import artifacts as _artifacts

    run_ids = _artifacts.completed_runs(out_dir)  # oldest → newest
    if not run_ids:
        return None
    by_slot: dict[str, dict] = {}
    for run_id in run_ids:  # a LATER run's write always wins per slot, whatever its own index
        try:
            manifest = _artifacts.read_manifest(out_dir, run_id)
        except (OSError, ValueError, KeyError):
            continue
        per_run: dict[str, dict] = {}
        for path, entry in manifest.items():
            slot = entry["slot"]
            if slot not in per_run or entry["index"] > per_run[slot]["index"]:
                per_run[slot] = {**entry, "path": path}
        for slot, entry in per_run.items():
            by_slot[slot] = {**entry, "run_id": run_id}
    arrays: dict[str, Any] = {}
    for slot, entry in list(by_slot.items()):
        try:
            arrays[slot] = _artifacts.load_array(entry["path"])
        except (OSError, ValueError):
            del by_slot[slot]
    if not arrays:
        return None
    # The latest run that actually CONTRIBUTED a fact — not necessarily `run_ids[-1]`: the
    # single most-recent completed folder can be exactly the one whose manifest just failed
    # to parse, and naming it here would point the session announcement at a run that gave
    # this resume nothing.
    order = {run_id: i for i, run_id in enumerate(run_ids)}
    top_run_id = max((e["run_id"] for e in by_slot.values()), key=lambda rid: order[rid])
    return {
        "run_id": top_run_id,
        "arrays": arrays,
        "index": {slot: e["index"] for slot, e in by_slot.items()},
        "origin": {slot: e["run_id"] for slot, e in by_slot.items()},
        "row_identity": {slot: e.get("row_identity") for slot, e in by_slot.items()},
        "steps": [(e["step"], slot, e["index"]) for slot, e in by_slot.items()],
    }


class Session:
    """One state lineage, applied one step at a time.

    Holds exactly what `runner._run_steps` holds in locals — `state`, `g`, the `carry`, and a
    monotonic index — plus the run-scoped facts `_finalize` needs. Nothing here is a second
    copy of the loop: `apply` is one call to `apply_step`, and `results` is one call to
    `_finalize`, the same assembler `run_inproc` and `run_manylatents` end with.
    """

    def __init__(
        self,
        project: str,
        engine: str = "mock",
        out_dir: Path = Path("."),
        seed: int = 42,
        fast_dev_run: bool = True,
        modality: str = "unknown",
        recipe: Optional[dict] = None,
        array: Any = None,
        labels: Any = None,
        label_kind: Any = None,
        dataset: Optional[str] = None,
        data_kwargs: Optional[dict] = None,
        *,
        dataset_name: Optional[str] = None,
        target_dim: int = 3,
        device: str = "cpu",
        metrics: Any = None,
        embedding: Any = None,
        on_step: Any = None,
        shape: str = "unknown",
        parent: Any = None,
        sample_ids: Any = None,
        declared_shape: Any = None,
        color: Any = None,
        label_key: Optional[str] = None,
        counts: Any = None,
        genes: Any = None,
        layers: Optional[dict] = None,
    ):
        from manyruns.pipeline import bounds

        declared_shape = bounds.declared_shape(declared_shape)
        self.project = project
        self.engine = engine
        self.out_dir = Path(out_dir)
        self.seed = int(seed)
        self.fast_dev_run = bool(fast_dev_run)
        self.modality = modality
        self.recipe = recipe or {}
        self.labels = labels
        # "time" | "condition" | "group" — three kinds in one channel, and every consumer
        # ALLOWS the one it wants rather than denying the ones it does not: only "time"
        # reaches a trajectory step, only "condition" reaches `separation`/`composition`.
        self.label_kind = label_kind
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.data_kwargs = dict(data_kwargs or {})
        self.shape = shape
        self.on_step = on_step
        self.compute = None  # Optional execution placement; never an engine selection.
        # A caller-supplied embedding is a fact the CALLER provided, which is what
        # `vocab.noncanonical` needs to say "legal and unusual" rather than nothing at all.
        self.provided = {"embedding"} if getattr(embedding, "ndim", 0) == 2 else set()
        # Which lineage this one started from, or None. Normalised (and rejected if it is
        # half an address) at OPEN, next to `run_id`, because that is when the answer is
        # known and because a session's close can be an hour of prompting away.
        self.parent = _runner.as_parent(parent)

        self.dispatch = _runner.dispatch_for(engine)   # raises for a non-steppable engine
        self.state = _runner._new_state(
            X=array if getattr(array, "ndim", 0) == 2 else None,
            labels=labels, seed=self.seed, emb=embedding,
            counts=counts, genes=genes, layers=layers,
        )
        if counts is not None and self.state["X"] is not None:
            from manyruns.aligned import check_aligned

            check_aligned(self.state["X"], counts, what="Session(counts= vs array)")
        self.state["label_kind"] = label_kind
        display = _runner.display_context(self.state, color, sample_ids, label_key, declared_shape)
        # The size of the data this lineage OPENED on — see `runner.open_gvector`. A session's
        # data can shrink mid-lineage (a `prep` filter narrows `state["X"]`), and `n_samples`
        # means the count that entered, so it is read here rather than at `close()`.
        self.g: dict[str, Any] = _runner.open_gvector(self.state)
        self.plots: list[str] = []
        self.steps: list[dict] = []
        # DECLARED STEPS THAT RAN AND WERE THEN THROWN AWAY — a driver's channel, empty on
        # every path that has no way to throw one away. `self.steps` cannot carry this: a
        # cancelled tune attempt IS in there with an `ok` record, because it did run; what it
        # did not do is survive (`tune.run_tune_loop`'s cancel path `_reset()`s `state` and `g`
        # back to the baseline). Appending a name here is how a driver says "the recipe did not
        # get through", and `results()` below turns it into the two things a reader checks:
        # `complete: false` and a caveat that names the step.
        self.discarded: list[str] = []
        self.carry = _runner.new_carry()
        # `run_id` is minted at OPEN, not at close: `_persist` reads `ctx["run_id"]` to address
        # the artifact step 0 writes, so a lineage with no address at step 0 writes nothing.
        # `spec_id` is the half that cannot be known yet — it hashes the step list, and a
        # session's step list is not known until the session ends. See `results`.
        self.run_id = _runner.identity(self.recipe, engine=engine, seed=self.seed,
                                       dataset=dataset, dataset_name=dataset_name,
                                       data_kwargs=self.data_kwargs)[0]
        self.ctx: dict[str, Any] = {
            "out_dir": self.out_dir, "plots": self.plots, "target_dim": target_dim,
            "seed": self.seed, "fast_dev_run": self.fast_dev_run, "device": device,
            "array": array, "data_ref": dataset, "data_kwargs": self.data_kwargs,
            "dataset_name": dataset_name,
            "declared_shape": declared_shape,
            # The declared suite reaches the ENGINE here, the same way `_explore_project`
            # sends it — `configs/metrics/default.yaml` was inert on this path entirely.
            "metrics": _default_metrics() if metrics is None else metrics,
            "color": _color(labels),
            "run_id": self.run_id,
            "caveats": [],
            "sample_ids": sample_ids,
            # mock-only: the string its stand-in vector is derived from. Identical to what the
            # deleted `_step_mock` used, so a mock session's numbers are unchanged.
            "mock_seed": f"{modality}:{project}",
        }

        self.ctx.update(display)

    # ── the seam ─────────────────────────────────────────────────────────────
    def apply(self, step: dict) -> dict:
        """Apply one step to this lineage and return its record.

        `index=len(self.steps)` is the monotonic counter, evaluated BEFORE the call — the
        record is appended inside `apply_step` (that is what gives a live observer the list it
        renders), so appending it again here would double-count and break the index.
        """
        if self.compute is not None:
            index = len(self.steps)

            def reported(rec, steps):
                if rec.get('state') == 'running':
                    self.steps[:] = steps
                    _runner._report(self.on_step, rec, self.steps)

            try:
                state, g, ctx, carry, steps = self.compute.call(
                    _runner.compute_step, step, self.state, self.g,
                    dispatch=self.dispatch, ctx=self.ctx, index=index, carry=self.carry,
                    steps=self.steps, on_step=reported)
            except _runner.ComputeCancelled:
                # A progress message is provisional: no returned snapshot means this
                # attempt never reached the retained state. The driver records its
                # declared occurrence in not_run together with the remaining recipe.
                del self.steps[index:]
                raise
            except Exception as exc:
                # No returned snapshot means no committed transition. Retain the last
                # authoritative state and mark the issued occurrence as interrupted/error.
                if len(self.steps) == index:
                    self.steps.append({'index': index, 'name': step.get('name'),
                                       'group': _runner.step_group(step),
                                       'params': dict(step.get('params') or {})})
                rec = self.steps[index]
                rec.update(outcome='error', detail=f'{type(exc).__name__}: {exc}', seconds=0.0)
                _runner._settle(rec)
                _runner._report(self.on_step, rec, self.steps)
                return rec
            for target, source in ((self.state, state), (self.g, g),
                                   (self.ctx, ctx), (self.carry, carry)):
                target.clear()
                target.update(source)
            self.plots[:] = ctx['plots']
            self.ctx['plots'] = self.plots
            self.steps[:] = steps
            _runner._report(self.on_step, self.steps[index], self.steps)
            return self.steps[index]
        return _runner.apply_step(
            step, self.state, self.g, dispatch=self.dispatch, ctx=self.ctx,
            index=len(self.steps), carry=self.carry, on_step=self.on_step, steps=self.steps,
        )

    def step(self, action: Any) -> dict:
        """Apply one step and return it in the shape the REPL renders.

        `action` is either a NAME — what a person types, resolved against the catalog so the
        step still arrives carrying its declared `group` and `params` — or a step DICT, which
        is what a loaded recipe holds and what a caller with its own params passes. The dict
        form is not a convenience: re-resolving a recipe's step through its name discards the
        params that recipe declared, which is the defect this whole seam exists to close.

        The full record is under `record`; everything else is a view of it.
        """
        # `recipe=self.recipe` is what makes a typed name mean the step THIS analysis
        # declares. See `resolve_step`: without it a typed `pca` resolved to whichever
        # recipe sorts first alphabetically, so walking `cflows` by hand gave a 10-dim PCA
        # where its own file declares 50.
        step = (action if isinstance(action, dict)
                else resolve_step(action, engine=self.engine, recipe=self.recipe))
        if step is None:
            offered = ", ".join(available_actions(self.engine)) or "(none on this engine)"
            return {"ok": False, "name": str(action),
                    "error": f"unknown action {action!r}; try {offered}"}
        return self._view(self.apply(step))

    def run_recipe(self) -> list[dict]:
        """Issue the loaded recipe's steps, in order, as step DICTS.

        No de-duplication against what already ran. The old loop skipped any step whose name
        was already in `status`, which silently refused to re-run a step you deliberately
        wanted to re-run with different params — the exact gesture "mix and match" is made of.
        Re-running a step is now a second entry in the lineage at its own index, which is what
        keeps both embeddings on disk rather than the second overwriting the first.
        """
        return [self.step(dict(s)) for s in (self.recipe.get("steps") or [])]

    # ── rewind: the second lineage ───────────────────────────────────────────
    def branch(self, index: int, *, recipe: Optional[dict] = None) -> "Session":
        """Rewind to the embedding step `index` produced and start a NEW lineage from it.

        "phate, then mioflow, then a different mioflow" is two different things, and only the
        gesture tells them apart. Chained, it is ONE lineage of three steps — `state["emb"]`
        is mutated in place, so the third step consumes the second's output. Rewound, it is
        TWO lineages sharing a prefix, and this is the rewind: a fresh `run_id`, a fresh
        `state/<run_id>/` folder whose `NN` starts at 00 again, and `parent` naming the
        `(run_id, index)` whose output it starts from.

        Measured on this checkout (in-process loop, 150x12): parent runs `phate` then `mioflow`
        and holds `00-phate_emb.npy 01-mioflow_pseudotime.npy`; `branch(0)` then runs
        `mioflow` and holds `00-mioflow_pseudotime.npy` in its own folder, with the parent's
        two files unchanged and its embedding still loadable. Extending the parent's folder
        instead would need `NN` to become a graph node rather than a position, and would make
        the parent's `COMPLETE` marker mean "finished, and also still growing".

        **The coordinates come off disk, not out of memory.** `state["emb"]` holds the LAST
        embedding, so by the time a person wants to rewind, the one they want to rewind to is
        gone from the process. `_persist` wrote it at the moment the step ran, and this is the
        first product caller of the read half (`artifacts.load_array`) — the round trip that
        `tests/test_artifacts.py` pins for the atomic path, made a gesture.

        A step whose embedding was never written is refused rather than branched from: mock
        steps write no arrays at all, a step that only read the embedding replaced no slot,
        and an over-size array is recorded as `"not written: …"` rather than a path. Falling
        back to the current `state["emb"]` in any of those cases would silently branch from
        the wrong coordinates — the branch would look identical and be a different experiment.

        The child inherits the parent's data, engine, seed and metrics; only the state chain
        restarts. `array` in particular is passed on, because `n_samples`/`n_features` come
        from it and a branch whose g-vector lost them would not be comparable with the lineage
        it came from — which is the entire point of recording the edge.

        Two things this does NOT do, both deliberate. It does not close the parent:
        `finish_carry` is the only writer of `COMPLETE` and the parent may still be stepped,
        so whoever opened it still closes it (measured: after `branch`, the parent's folder has
        no `COMPLETE`; after `parent.close()` it does, with a manifest listing both files).
        And it does not rename plots — those are `plots/<step>.png`, one per step name and not
        per lineage, so a branch that re-runs `mioflow` overwrites the parent's PNG while both
        arrays survive. The picture was never the artifact; the embedding is.
        """
        if not 0 <= index < len(self.steps):
            raise IndexError(f"no step {index} in this lineage (it has {len(self.steps)})")
        rec = self.steps[index]
        path = (rec.get("artifacts") or {}).get("emb")
        if not path or not Path(str(path)).is_file():
            raise ValueError(
                f"step {index} ({rec.get('name')}) left no embedding on disk to rewind to"
                + (f" — {path}" if path else "")
            )
        alignment = rec.get("metadata_aligned")
        if alignment is False:
            raise ValueError("cannot branch: metadata alignment unavailable for this engine output")
        # A NARROWING SINCE THAT STEP is the one thing the rewind cannot carry. The child
        # inherits this lineage's DATA (`array` below) as it now stands — filtered — while the
        # embedding on disk was fitted over the cells that were there BEFORE the filter, so the
        # two disagree by exactly the cells `dropped` names, and nothing downstream would say
        # so: `separation` reads labels against clusters and mis-aligns, `_save_scatter` drops
        # the colouring on a length mismatch (`io.py:106`). Refused for the same reason a step
        # that wrote no embedding is refused — a branch that looks identical and is a different
        # experiment is worse than no branch.
        #
        # KEYED ON THE COUNT, not on the presence of the key. A filter records `dropped` even
        # when it removed nothing — that is a measurement, and on pbmc3k `filter_cells(
        # min_genes=200)` drops exactly 0 of 2700 while `detect_doublets(remove: false)`
        # returns an all-True mask by design (spec §5). Measured before this changed: a
        # zero-drop filter made `branch(0)` raise "step 0 (phate)'s embedding predates a
        # filter: cut dropped 0 rows since" — a refusal that states its own falsity, and one
        # that stood for the rest of the lineage.
        dropped = [s for s in self.steps[index + 1:] if (s.get("dropped") or {}).get("n")]
        if dropped:
            n = sum(int(s["dropped"].get("n") or 0) for s in dropped)
            raise ValueError(
                f"step {index} ({rec.get('name')})'s embedding predates a filter: "
                f"{', '.join(str(s.get('name')) for s in dropped)} dropped {n} "
                f"{dropped[0]['dropped'].get('axis')} since. Rewind to a step after the "
                f"filter, or start a new lineage from the data."
            )
        from manyruns import artifacts as _artifacts

        child = Session(
            project=self.project, engine=self.engine, out_dir=self.out_dir, seed=self.seed,
            fast_dev_run=self.fast_dev_run, modality=self.modality,
            # NOT the parent's recipe: this lineage did not run that recipe's earlier steps,
            # it inherited their output. Claiming the name would make `recipe` and `spec_id`
            # describe different things — `None` renders as "(adaptive)" at both readers.
            recipe=recipe or {},
            # `array` and `labels` travel as a PAIR, and the labels are the ones state holds
            # rather than the ones this session opened with: a prep step narrows both together
            # (`runner._apply_transition`), and handing the child a filtered matrix with the
            # unfiltered label vector is the desync `aligned` exists to catch — one row off and
            # every cell is attributed to the wrong donor. Identical objects on every lineage
            # that ran no filter, which is all of them until a recipe declares one.
            array=self.ctx["array"], labels=self.state.get("labels"),
            sample_ids=(self.ctx["sample_ids"][self.state["rows"]]
                        if self.ctx.get("sample_ids") is not None else None),
            label_kind=self.state.get("label_kind"),
            label_key=self.state.get("label_key"),
            color=([{**c, "values": c["values"][self.state["rows"]]}
                    for c in self.ctx.get("display_channels") or ()]),
            counts=self.state.get("counts"), genes=self.state.get("genes"),
            layers=self.state.get("layers"),
            dataset=self.dataset, data_kwargs=self.data_kwargs,
            dataset_name=self.dataset_name,
            declared_shape=self.ctx.get("declared_shape"),
            target_dim=self.ctx["target_dim"], device=self.ctx["device"],
            metrics=self.ctx["metrics"], on_step=self.on_step, shape=self.shape,
            embedding=_artifacts.load_array(path),
            parent={"run_id": self.run_id, "index": index},
        )
        if alignment is None:
            # A positional branch inherits the uncertainty and must not turn reloaded
            # source metadata into verified identities on its newly written artifacts.
            child.ctx["metadata_aligned"] = None
            child.ctx["caveats"].append(
                "branched from positional alignment: sample identity remains unverified; "
                "source metadata will not be attached to derived coordinates")
        return child

    # ── the record ───────────────────────────────────────────────────────────
    def performed(self) -> dict:
        """What this session actually issued, in the shape the catalog loads.

        A recipe is an output as well as an input: this is `{"name", "steps"}` with the same
        `{name, group, params}` step dicts `check_recipe` validates, so a session can be saved
        as a recipe without a second serializer.
        """
        return {
            "name": self._recipe_name(),
            "steps": [{"name": s["name"], "group": s["group"], "params": dict(s["params"])}
                      for s in self.steps],
        }

    def results(self, *, abandoned: bool = False, not_run=()) -> dict:
        """The record as it stands — `runner._finalize`, the ONE g-vector assembler.

        ``abandoned=True`` skips final geometry itself. ``not_run`` is a sequence of
        {index, name, reason} mappings for declared occurrences never issued by the driver;
        indices are zero-based, and this is separate from reverted tune attempts.

        Safe to call mid-session, and that is why it does not close the lineage: the REPL's
        `summary` command asks for it while more steps may still be coming, and `COMPLETE`
        written then would mean "this state folder is finished" about a folder that is not.
        `close` is the one that ends the lineage.

        `_finalize` is handed a COPY of `g`. It merges the full metric suite, and
        `suite.measure` lets a value already in `g` win — so folding the suite back into the
        session's own `g` would pin the FIRST call's metrics and every later `results()` would
        report geometry measured on an embedding two steps stale.
        """
        performed = self.performed()
        # `spec_id` over the sequence AS ISSUED, including steps that skipped or errored, so it
        # stays the same function of the same kind of input as the recipe path — where the
        # declared list also contains steps that may skip. Hashing only what succeeded would
        # make a session's `spec_id` incomparable with a recipe's the moment one step skipped.
        spec_id = _runner.identity(performed, engine=self.engine, seed=self.seed,
                                   dataset=self.dataset, dataset_name=self.dataset_name,
                                   data_kwargs=self.data_kwargs)[1]
        from manyruns import vocab as _vocab

        # `engine=` also picks up a STAND-IN: a step whose executor is a different algorithm
        # from the one its name and `via:` designate (`vocab.standins`). Same list, because it
        # is the same class of event — legal, unusual, running anyway — and a second channel
        # for it is how two answers to one question drift apart.
        #
        # Over `executed`, NOT over `performed`. `spec_id` above wants the sequence as issued;
        # the caveats are an account of what happened, and `noncanonical` clears the run's
        # step-produced facts on every DECLARED narrowing — so a `prep` step that skipped or
        # errored would rewrite where the record says the embedding came from. See
        # `runner.executed` for the measured case.
        caveats = _vocab.noncanonical(_runner.executed(performed, self.steps), self.shape,
                                      provided=self.provided, engine=self.engine)
        # THE SAME LIST, because it is the same class of event this channel already carries:
        # something about HOW this run reached its numbers that a reader has to know before
        # trusting them ("legal and unusual, running anyway" / "every number here is
        # invented"). A discarded step is the strongest member of that class — the geometry
        # below is measured on the state as it stood BEFORE the step, and nothing else in the
        # record says so. A second channel for it is how two accounts of one run drift apart
        # (`noncanonical`'s own docstring makes this argument against reporting one event in
        # two vocabularies), and `store.append` already writes `caveats` into `index.jsonl`,
        # so this reaches the learner's corpus with no schema change.
        caveats.extend(self.ctx.get("caveats") or [])
        caveats.extend(
            f"{name}: discarded at the tune prompt — this run does not contain it, and the "
            + ("retained state is as it stood before it" if abandoned else
               "geometry here is measured on the state as it stood before it")
            for name in self.discarded
        )
        finalize = _runner._finalize
        if self.compute is not None and not abandoned:
            def finalize(*args, **kwargs):
                try:
                    return self.compute.call(_runner._finalize, *args, **kwargs)
                except _runner.ComputeCancelled:
                    # The interrupted metric response was never committed. Existing step
                    # scores survive; the same abandoned assembler discloses missing geometry.
                    kwargs['abandoned'] = True
                    return _runner._finalize(*args, **kwargs)
                except (RuntimeError, OSError) as exc:
                    # A dead metric worker or failed startup must still produce one honest
                    # record. Never retry expensive geometry on the UI process.
                    kwargs['compute_error'] = str(exc)
                    return _runner._finalize(*args, **kwargs)
        return finalize(
            self.state, dict(self.g), engine=self.engine, recipe=performed, steps=self.steps,
            plots=self.plots, seed=self.seed, dataset=self.dataset, caveats=caveats,
            dataset_name=self.dataset_name, data_kwargs=self.data_kwargs,
            ident=(self.run_id, spec_id), parent=self.parent, discarded=self.discarded,
            abandoned=abandoned, not_run=not_run,
        )

    def close(self, *, abandoned: bool = False, not_run=()) -> dict:
        """Assemble the record and mark retained artifacts complete once compute stops.

        `finish_carry` is the only writer of `COMPLETE` and it is called here, once — never per
        step. A marker written per step is True while the session is still open, which destroys
        the single guarantee it exists to give.
        """
        try:
            return self.results(abandoned=abandoned, not_run=not_run)
        finally:
            if self.compute is not None:
                self.compute.close()
            # The child must be stopped before marking the parent's retained artifacts
            # complete; a cancelled child may otherwise still be writing them.
            _runner.finish_carry(self.carry, self.ctx)

    # ── views over the record (never a second accumulator) ───────────────────
    @property
    def trace(self) -> list[str]:
        """`["latent:phate", …]` — derived from the step record, as in `_finalize`."""
        return [f"{s['group']}:{s['name']}" for s in self.steps]

    @property
    def status(self) -> dict[str, str]:
        """`{name: status line}` — derived, so it cannot drift from `steps`. A repeated step
        name collapses here (last write wins) exactly as it does in `_finalize`; `steps` keeps
        both entries and is where to look when the two disagree."""
        return {s["name"]: _runner._status_line(s) for s in self.steps}

    def _recipe_name(self) -> Optional[str]:
        """The loaded recipe's name while the issued sequence is still a prefix of it, else None.

        A session that follows `embed` exactly IS a run of `embed` and should say so; one that
        deviates is adaptive and must not claim the name, because `spec_id` and the name would
        then describe different things. `None` already degrades correctly at both readers —
        `app` and `narrate` print "(adaptive)"."""
        declared = self.recipe.get("steps") or []
        if len(self.steps) > len(declared):
            return None
        for issued, want in zip(self.steps, declared):
            if (issued["name"], issued["group"], issued["params"]) != (
                    want.get("name"), want.get("group"), dict(want.get("params") or {})):
                return None
        return self.recipe.get("name")

    def _view(self, rec: dict) -> dict:
        """The record as the REPL renders it. `ok` uses `_finalize`'s definition — an outcome
        of exactly "ok" — so a step the panel calls failed is a step the row calls failed."""
        ok = rec["outcome"] == "ok"
        return {
            "ok": ok, "name": rec["name"], "outcome": rec["outcome"],
            "detail": rec["detail"] or _produced(rec),
            "error": rec["detail"] or rec["outcome"],
            "record": rec, "g_vector": dict(self.g),
        }


def _produced(rec: dict) -> str:
    """What a successful step has to show for itself, from the record's own output half."""
    bits = [str(rec["produced"])] if rec.get("produced") else []
    if rec.get("emitted"):
        bits.append(", ".join(rec["emitted"]))
    return " · ".join(bits) or "ok"


def _default_metrics() -> list:
    """The declared suite, or `[]`. A missing suite file is engine defaults, not a failure."""
    from manyruns import catalog

    try:
        return catalog.load_suite()
    except Exception:  # noqa: BLE001 - the suite is an input to a run, never its gate
        return []


def _color(labels: Any) -> Any:
    """Numeric colours for the scatter plots, when the labels can carry them."""
    from manyruns.pipeline import io as _io

    return _io._labels_to_numeric(labels)
