"""A NAMED dataset must not silently lose the data fact its own declaration asserts.

The DATA-level instance of the fabrication class, one level up from the step-ORDER fix
(commit df478cc). `app._load_inputs` routed through `pipeline.load_labeled` only when there
was no named dataset, so for `--dataset <name>` labels and label_kind came back None
unconditionally. Measured on this tree before the fix:

    _load_inputs("_inproc",        None, "swissroll", …) -> (SwissRollDataModule, None, None)
    _load_inputs("manylatents", None, "swissroll", …) -> (None,                None, None)

identical for `torus` and `dla_tree`. A dataset declaring `shape: time-course` therefore had
its time axis erased between the declaration the legality check reads and the run that
consumes it: `vocab.unmet(cflows, "time-course")` is `frozenset()` — LEGAL — and
`runner._ml_lightning` then computes `time_labels = state["labels"]` = None, so
`mioflow._run_mioflow_experiment` falls back to a diffusion pseudotime over the embedding.

End to end, `--dataset swissroll --engine real --recipe cflows`, seed 42, 5000x3
(as measured then; `--engine real` has since been removed — the loop it named is `run_inproc`):
trace `latent:phate -> lightning:mioflow`, status `phate: ok, mioflow: ok`, run `ok=True`,
`g_vector["pseudotime_range"] == [0.0, 1.0]` — a full-range trajectory whose clock came from
the geometry it is meant to be independent of, indistinguishable in the record from a run
driven by real collection times.

Carrying the labels instead of refusing was ruled out by measurement, not taste: manylatents'
dataset interface (`manylatents.data.capabilities.get_capabilities`) declares four
capabilities — gt_dists, graph, labels, centers — and no notion of time, and `get_labels()`
returns an untyped integer per row whose meaning differs per dataset (swissroll: 100
distribution indices, Spearman -0.224 against the roll's own arc coordinate `ts`, because the
per-distribution means are drawn `rng.random`; dla_tree: 20 branch ids; gaussian_blob: 3
cluster ids; archetypal: 4 archetype ids; torus and saddlesurface: no labels at all).

These tests are dependency-free on purpose: the refusal happens before any scientific import,
which is itself part of the contract. They declare their own datasets through
`MANYRUNS_DATASET_DIR` (catalog.py's documented override) rather than leaning on the shipped
catalog, so a dataset landing or leaving does not silently change what they assert; the one
test that DOES read the real catalog pins a biconditional, not a count.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from manyruns import app

#: A dataset that CLAIMS a time axis and handles to a manylatents ref — i.e. reaches a run by
#: the `--dataset NAME` route, the one that cannot deliver it. The 12 original bundled
#: datasets declare `manifold` or `clusters`, both of which `vocab.dataset_provides` maps to
#: `frozenset()`, so the hole was invisible; the 13th (`synthetic_timecourse`, added
#: concurrently) declares `time-course` but handles to a PATH, so it takes the route that does
#: deliver. Neither exercises this combination, so the fixture declares it.
_TIME_COURSE_YAML = """
name: fake_days
handle: {kind: manylatents, ref: swissroll}
modality: synthetic
shape: time-course
"""

_CASE_CONTROL_YAML = """
name: fake_arms
handle: {kind: manylatents, ref: gaussian_blob}
modality: synthetic
shape: case-control
"""

#: Declares nothing that has to reach the run — the control for every assertion below.
_MANIFOLD_YAML = """
name: fake_roll
handle: {kind: manylatents, ref: swissroll}
modality: synthetic
shape: manifold
"""


@pytest.fixture()
def dataset_dir(tmp_path, monkeypatch):
    """A catalog of three declarations, injected via the documented env override."""
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "fake_days.yaml").write_text(_TIME_COURSE_YAML)
    (d / "fake_arms.yaml").write_text(_CASE_CONTROL_YAML)
    (d / "fake_roll.yaml").write_text(_MANIFOLD_YAML)
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(d))
    return d


# ── the premise, re-derived here rather than taken on trust ──────────────────
def test_a_named_dataset_yields_no_labels_from_the_reader(dataset_dir):
    """The blackhole itself: the READING half returns no labels for a named dataset.

    `_read_inputs` is `_load_inputs` minus the refusal, so this pins the fact the refusal
    exists to cover — and pins it for the engine (`manylatents`) whose branch requires no
    scientific import, so it holds with and without the private stack."""
    loaded = app._read_inputs("manylatents", None, "fake_days", None)
    array, labels, kind = loaded["array"], loaded["labels"], loaded["kind"]

    assert (array, labels, kind) == (None, None, None), (
        "a named dataset still reaches the engine with labels; the refusal below may be moot"
    )


def test_unmet_calls_a_trajectory_recipe_legal_on_a_time_course_declaration():
    """Why the loader is the only line of defence: the legality function reads the
    DECLARATION and cannot see that the fact will not arrive. Recorded, not fixed here —
    `vocab.py` is out of scope for this change."""
    from manyruns.vocab import unmet

    cflows = {"name": "cflows", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        {"name": "mioflow", "group": "lightning", "params": {}},
    ]}

    assert unmet(cflows, "time-course") == frozenset(), (
        "unmet now prunes this — the refusal in app.py may be redundant, re-check it"
    )


# ── the refusal ──────────────────────────────────────────────────────────────
def test_load_inputs_refuses_a_named_time_course_dataset(dataset_dir):
    """THE regression. Before the fix this returned (None, None, None) and the run continued."""
    with pytest.raises(ValueError) as e:
        app._load_inputs("manylatents", None, "fake_days", None)

    msg = str(e.value)
    assert "fake_days" in msg
    assert "time-course" in msg, "the refusal must name the declaration it is refusing"
    assert "`time`" in msg, "the refusal must name the MISSING fact"
    assert "pseudotime" in msg, "the refusal must say what would otherwise be fabricated"


def test_run_explorations_refuses_it_too(dataset_dir):
    """The funnel every path reaches (`explore_once`, `modes._infer`, the harness) skips
    `_load_inputs` for a named dataset — `data_folder is None` — so the rule has to hold here
    independently or those three run unchecked. Same reason `_require_embedding` is called
    from both engines."""
    recipe = {"name": "cflows", "steps": [{"name": "phate", "group": "latent", "params": {}}]}

    with pytest.raises(ValueError, match="fake_days"):
        app.run_explorations(None, "synthetic", recipe=recipe, engine="manylatents",
                             dataset="fake_days")


def test_the_refusal_precedes_any_engine_work(dataset_dir):
    """It must refuse BEFORE the server is built or any array is touched. A server that was
    already asked to predict has, for the real engines, already started reading data."""
    called = []

    class Boom:
        def predict(self, _inputs):
            called.append(_inputs)
            raise AssertionError("the engine was reached despite an undeliverable declaration")

    recipe = {"name": "cflows", "steps": [{"name": "phate", "group": "latent", "params": {}}]}
    with pytest.raises(ValueError):
        app.run_explorations(None, "synthetic", server=Boom(), recipe=recipe,
                             engine="manylatents", dataset="fake_days")

    assert called == []


def test_a_case_control_declaration_is_refused_for_its_own_fact(dataset_dir):
    """The same hole swallows `conditions`, and the reason given must be that fact's, not
    time's — a copy-pasted message would report the wrong consequence.

    Uses the `manylatents` engine branch, which returns without importing anything: on the
    `real` branch the READ happens first (the refusal is a postcondition over it), so a
    stackless run there is stopped earlier, by `load_named_dataset`'s own "needs manylatents".
    Later, but still before any compute."""
    with pytest.raises(ValueError) as e:
        app._load_inputs("manylatents", None, "fake_arms", None)

    msg = str(e.value)
    assert "`conditions`" in msg
    assert "separation/composition" in msg
    assert "pseudotime" not in msg, "the time consequence must not be reported for conditions"


# ── what it must NOT refuse ──────────────────────────────────────────────────
def test_a_dataset_that_declares_no_data_fact_is_untouched(dataset_dir):
    """`manifold` provides nothing, so nothing is promised and nothing is broken. This is the
    over-refusal guard: it is what keeps the rule off all twelve bundled datasets."""
    assert app._load_inputs("manylatents", None, "fake_roll", None) == app._EMPTY_INPUTS


def test_the_guard_fires_over_the_shipped_catalog_exactly_where_a_fact_is_declared(monkeypatch):
    """Precision over the REAL catalog, with the env override removed: refuse a name whose
    declared shape provides something, leave every other name alone. Not "refuses nothing" —
    that was true of the twelve `manifold`/`clusters` datasets and stopped being true the
    moment `synthetic_timecourse` (`shape: time-course`) landed. The invariant that survives
    a growing catalog is the biconditional, so that is what is pinned."""
    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)
    from manyruns import catalog
    from manyruns.vocab import dataset_provides

    names = catalog.discover_datasets()
    assert len(names) >= 12, f"catalog shrank unexpectedly: {names}"
    for name in names:
        declares = bool(dataset_provides(catalog.load_dataset(name).get("shape")))
        try:
            app._require_declared_facts("manylatents", name, None, None)
            refused = False
        except ValueError:
            refused = True
        assert refused is declares, (
            f"{name}: declares a data fact={declares} but refused={refused}"
        )


def test_a_path_handle_dataset_is_not_on_this_route_at_all(dataset_dir):
    """The guard is keyed on `--dataset NAME`, and must stay off the route that DOES deliver.

    `synthetic_timecourse` declares `shape: time-course` and carries a real one, but it
    handles to `{kind: path, ref: synthetic:time-course}` — `experiment._request` sends every
    non-manylatents handle as `data=`, leaving `dataset=None`, so it loads through
    `pipeline.load_labeled`. Measured on that route: a (300, 5) matrix with `label_kind="time"`
    and timepoints {0.0, 0.5, 1.0}. Nothing is missing, and the guard must not invent a
    problem: with no `--dataset` name there is no named-dataset claim to check."""
    app._require_declared_facts("manylatents", None, None, None)
    app._require_declared_facts("_inproc", "", None, None)


def test_an_undeclared_dataset_name_asserts_nothing(dataset_dir):
    """`--dataset dla_tree` is a manylatents name with no manyruns declaration. It promises
    nothing, so refusing it would be inventing a contract that was never written."""
    app._require_declared_facts("manylatents", "dla_tree", None, None)
    app._require_declared_facts("_inproc", "not_a_dataset_anywhere", None, None)


def test_engines_that_do_not_load_here_are_left_alone(dataset_dir):
    """`mock` reads no data at all ("invents numbers and never reads your data"), so a missing
    label is not manyruns's to refuse for it.

    This used to check the learner's engine alongside it, for a DIFFERENT reason — the whole recipe went
    downstream in one call, so the engine loaded the named dataset itself and might well have
    had the axis. That engine is gone (§3.5), and `mock` is the only member left. The guard is
    unchanged: it keys on `_MANYRUNS_LOADS_FOR`, so a backend added later is silently exempt
    until someone puts it in that set — which is the right default, since the set names the
    engines whose loading manyruns OWNS."""
    app._require_declared_facts("mock", "fake_days", None, None)
    # and the contrast, so this test cannot pass by the guard having stopped working at all:
    # `manylatents` IS in the set, and the same call refuses.
    with pytest.raises(ValueError, match="declares shape: time-course"):
        app._require_declared_facts("manylatents", "fake_days", None, None)


def test_delivering_the_declared_fact_clears_the_refusal(dataset_dir):
    """A POSTCONDITION, not a hardcoded 'named datasets never have time'. If a future loader
    does return per-row time for a named dataset, the refusal stops firing on its own instead
    of having to be remembered and deleted."""
    app._require_declared_facts("manylatents", "fake_days", ["d0", "d3", "d9"], "time")
    app._require_declared_facts("_inproc", "fake_arms", ["treated", "healthy"], "condition")


def test_the_wrong_kind_of_label_does_not_satisfy_a_time_declaration(dataset_dir):
    """A condition axis is not a clock. Handing MIOFlow one trains a fabricated trajectory
    (treated -> healthy 'over time'), which is why `label_kind` travels with the labels at all;
    delivering conditions must not discharge a `time` declaration."""
    with pytest.raises(ValueError, match="`time`"):
        app._require_declared_facts("manylatents", "fake_days", ["treated", "healthy"], "condition")


def test_a_kind_with_no_labels_does_not_count_as_delivery(dataset_dir):
    """`kind` without rows is a claim with nothing behind it."""
    with pytest.raises(ValueError, match="`time`"):
        app._require_declared_facts("manylatents", "fake_days", None, "time")


# ══ the same class of fabrication, one level down: the FILE the declaration names ═══════════
#
# The tests above pin "a named dataset must not lose the data FACT it declares". These pin the
# fact underneath it: a named dataset must not lose the FILE it declares, and must not be
# offered as loadable when that file is not on this machine.
#
# Measured on this checkout, 2026-08-17, before `shell.dataset_ref_path` existed —
# `pbmc3k.yaml` declares `ref: data/pbmc3k_raw.h5ad`, which `shell._resolve_dataset` resolved
# with a bare `Path(ref)` while `catalog.discover_datasets()` reads that YAML out of the WHEEL:
#
#     cwd=<repo root>   shell._resolve_dataset("pbmc3k") -> Observation(n_obs=2700, n_vars=32738)
#     cwd=/private/tmp  shell._resolve_dataset("pbmc3k") -> Observation(n_obs=None, n_vars=None)
#
# The second is the whole defect in one line: the row is on the menu of every install, the file
# is in one directory of one checkout, and the mismatch was reported nowhere — the picker
# printed "✓ data loaded" off the declared shape, offered the recipes that shape allows, and
# the run then failed on a path that was never there. `Observation.provides()` also reads empty
# there, so a `.h5ad` that DECLARES a gene axis was refused the steps that need one.
#
# These tests declare their own catalog (`$MANYRUNS_DATASET_DIR`) and their own drop folder
# (`$MANYRUNS_DATA_DIR`) and `chdir` into a tmp dir, for the reason the defect exists: an
# assertion about where a ref resolves that is itself run from the repo root would pass on the
# one directory where the bug was invisible.

_ABSENT_YAML = """
name: absent_cohort
handle: {kind: path, ref: elsewhere/absent_cohort.csv}
modality: scrna
shape: clusters
source: {kind: tutorial, url: "https://example.invalid/absent_cohort.csv"}
"""

_DROPPED_YAML = """
name: dropped_cohort
handle: {kind: path, ref: elsewhere/dropped_cohort.csv}
modality: scrna
shape: clusters
"""

_GENERATED_YAML = """
name: made_up
handle: {kind: path, ref: "synthetic:time-course"}
modality: synthetic
shape: time-course
"""


@pytest.fixture()
def ref_catalog(tmp_path, monkeypatch):
    """Four declarations, an empty drop folder, and a CWD that is not the checkout root.

    Returns `(dataset_dir, drop_folder)`; the caller materialises whatever it wants present.
    """
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "absent_cohort.yaml").write_text(_ABSENT_YAML)
    (d / "dropped_cohort.yaml").write_text(_DROPPED_YAML)
    (d / "made_up.yaml").write_text(_GENERATED_YAML)
    (d / "fake_roll.yaml").write_text(_MANIFOLD_YAML)
    drop = tmp_path / "drop"
    drop.mkdir()
    monkeypatch.setenv("MANYRUNS_DATASET_DIR", str(d))
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(drop))
    monkeypatch.chdir(tmp_path)
    return d, drop


class _Console:
    """Captures instead of printing. The refusals below are only useful if they SAY something."""

    is_terminal = False

    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class _PickerBot:
    """Answers `select` by matching a substring against the OFFERED LABELS, and records them.

    Matching on the label rather than replaying a fixed index is the point: these tests assert
    what the menu SAYS, so they have to read the same strings a person would.
    """

    def __init__(self, *wants):
        self.wants = list(wants)
        self.offered: list[list[str]] = []

    def select(self, message, choices, default=None):
        self.offered.append([label for label, _ in choices])
        want = self.wants.pop(0)
        for label, value in choices:
            if want in label:
                return value
        raise AssertionError(f"no row matching {want!r} in {[c[0] for c in choices]}")

    def text(self, message, default=""):
        return ""


# ── the anchoring ────────────────────────────────────────────────────────────
def test_a_declared_ref_is_found_in_the_drop_folder_from_any_directory(ref_catalog):
    """THE regression for the CWD dependence. The ref is relative and the CWD is not its
    anchor, so the file is looked for by name where the product tells you to put data."""
    from manyruns import shell

    _, drop = ref_catalog
    (drop / "dropped_cohort.csv").write_text("a,b\n1,2\n")

    found = shell.dataset_ref_path("elsewhere/dropped_cohort.csv")
    assert found is not None and found.resolve() == (drop / "dropped_cohort.csv").resolve()


def test_the_ref_as_written_outranks_a_drop_folder_namesake(ref_catalog):
    """Anchor ORDER, not just anchor set. A ref that resolves where it says it does must win,
    or a stray file of the same name in the drop folder silently replaces the declared one."""
    from manyruns import shell

    _, drop = ref_catalog
    (drop / "dropped_cohort.csv").write_text("drop\n")
    here = Path("elsewhere")
    here.mkdir()
    (here / "dropped_cohort.csv").write_text("cwd\n")

    found = shell.dataset_ref_path("elsewhere/dropped_cohort.csv")
    assert found is not None and found.read_text() == "cwd\n"


def test_an_absolute_ref_is_never_answered_with_a_different_file(ref_catalog):
    """An absolute ref states a location exactly. Falling back to a same-named file in the drop
    folder would answer a question that was not asked — the one way basename matching could
    hand back the wrong data."""
    from manyruns import shell

    _, drop = ref_catalog
    (drop / "dropped_cohort.csv").write_text("a,b\n1,2\n")

    assert shell.dataset_ref_path("/nowhere/at/all/dropped_cohort.csv") is None


def test_a_generated_ref_is_never_looked_for_on_disk(ref_catalog):
    """`synthetic:time-course` is a generator (`pipeline.loading._GENERATED`), not a file, so
    "not found" is the wrong report and a drop-folder search for it is nonsense."""
    from manyruns import shell
    from manyruns.catalog import load_dataset

    assert shell.dataset_ref_path("synthetic:time-course") is None
    assert shell.missing_dataset_file(load_dataset("made_up")) is None


def test_a_handle_the_engine_loads_is_not_a_missing_file(ref_catalog):
    """A `kind: manylatents` ref is a NAME the engine resolves; there is no file here to miss,
    and reporting one would strike out the twelve bundled synthetics."""
    from manyruns import shell
    from manyruns.catalog import load_dataset

    assert shell.missing_dataset_file(load_dataset("fake_roll")) is None


# ── the listing and the run agreeing ─────────────────────────────────────────
def test_a_declaration_whose_file_is_nowhere_reports_it(ref_catalog):
    """The fact the silent degrade withheld, now available to every surface that lists."""
    from manyruns import shell
    from manyruns.catalog import load_dataset

    assert shell.missing_dataset_file(load_dataset("absent_cohort")) == \
        "elsewhere/absent_cohort.csv"


def test_the_menu_row_says_the_data_is_not_here(ref_catalog):
    """The listing half. Before this the row read `absent_cohort  (sample · clusters)` — the
    same text as a sample whose bytes are present."""
    from manyruns import shell

    bot, console = _PickerBot("← back"), _Console()
    shell._pick_source(bot, console)

    rows = bot.offered[0]
    absent = next(r for r in rows if r.startswith("absent_cohort"))
    assert "not on this machine" in absent, absent
    # …and the two that ARE runnable must not be tarred with it.
    assert not any("not on this machine" in r for r in rows
                   if r.startswith(("made_up", "fake_roll")))


def test_picking_it_refuses_with_an_address_instead_of_degrading(ref_catalog):
    """The run half. `_pick_source` must not hand back a source tuple for data that is not
    here: before this it returned `(Path("elsewhere/absent_cohort.csv"), None, "scrna",
    Observation(shape="clusters"))` — a pick that looks loaded and fails at the run.

    The menu is re-shown (the pick is not fatal), and the message has to be actionable: the
    file's name, where it was looked for, the declared `source.url`, and the drop folder."""
    from manyruns import shell

    _, drop = ref_catalog
    bot, console = _PickerBot("absent_cohort", "← back"), _Console()

    assert shell._pick_source(bot, console) is None       # backed out, nothing loaded
    assert len(bot.offered) == 2, "the menu was not re-shown after the refusal"
    out = console.text
    assert "absent_cohort.csv" in out
    assert "elsewhere/absent_cohort.csv" in out
    assert "https://example.invalid/absent_cohort.csv" in out, "the fetch address is the fix"
    assert str(drop) in out


def test_a_dataset_whose_file_is_present_is_picked_and_read(ref_catalog):
    """The over-refusal guard. Drop the file in and the same row loads — the refusal is a
    statement about this machine, and it has to stop the moment the machine changes."""
    from manyruns import shell

    _, drop = ref_catalog
    (drop / "dropped_cohort.csv").write_text("a,b\n1,2\n")

    bot, console = _PickerBot("dropped_cohort  (sample"), _Console()
    picked = shell._pick_source(bot, console)

    assert picked is not None
    folder, dataset, modality, obs = picked
    assert folder is not None and folder.resolve() == (drop / "dropped_cohort.csv").resolve()
    assert (dataset, modality, obs.source) == (None, "scrna", "dropped_cohort")
    assert not any("not on this machine" in r for r in bot.offered[0]
                   if r.startswith("dropped_cohort  (sample"))


def test_resolve_dataset_reads_the_file_it_found_not_the_ref_it_was_given(ref_catalog):
    """`_resolve_dataset` is what `tui.state.roster` calls too, so the anchoring has to happen
    THERE rather than in one picker — two surfaces resolving one declaration differently is the
    two-accounts-of-one-dataset failure its own comment already records."""
    from manyruns import shell

    _, drop = ref_catalog
    (drop / "dropped_cohort.csv").write_text("a,b\n1,2\n")

    folder, _, _, obs = shell._resolve_dataset("dropped_cohort")
    assert folder.resolve() == (drop / "dropped_cohort.csv").resolve()
    assert obs.source == "dropped_cohort"


def test_an_unresolvable_ref_still_reports_where_the_declaration_pointed(ref_catalog):
    """It degrades to the declared shape — that part is unchanged and is what keeps the roster
    row on screen — but the path it returns must remain the ref AS DECLARED, so the surface
    that prints it says where to look rather than inventing a location."""
    from manyruns import shell

    folder, dataset, _, obs = shell._resolve_dataset("absent_cohort")
    assert (str(folder), dataset) == ("elsewhere/absent_cohort.csv", None)
    assert (obs.shape, obs.n_obs, obs.n_vars) == ("clusters", None, None)


def test_the_datasets_listing_marks_the_same_row_the_picker_does(ref_catalog):
    """`datasets` and the picker are two listings of one catalog; the finding is that a listing
    must not disagree with the run, and two listings disagreeing with each other is the same
    defect twice."""
    from manyruns import shell

    console = _Console()
    shell._list_datasets(console)
    absent = next(ln for ln in console.lines if "absent_cohort" in ln)
    assert "not on this machine" in absent
    assert not any("not on this machine" in ln for ln in console.lines if "made_up" in ln)


def test_the_listing_shows_where_a_ref_was_actually_found(ref_catalog):
    """When the anchor that answered was not the ref as written, the listing says so — the ref
    alone would now be the misleading half."""
    from manyruns import shell

    _, drop = ref_catalog
    (drop / "dropped_cohort.csv").write_text("a,b\n1,2\n")

    console = _Console()
    shell._list_datasets(console)
    line = next(ln for ln in console.lines if "dropped_cohort" in ln)
    assert "elsewhere/dropped_cohort.csv" in line, "the declaration is still identified"
    assert str(drop / "dropped_cohort.csv") in line, "so is the file that answered it"


# ── the shipped catalog, over the real thing ─────────────────────────────────
def test_the_one_bundled_file_ref_is_not_reported_missing_when_it_is_present(monkeypatch,
                                                                            tmp_path):
    """Over the REAL catalog, from a CWD that is not the checkout root — the directory where
    the defect was measured. A biconditional, not a hardcoded name: whatever the shipped
    catalog declares as a file, `missing_dataset_file` is None exactly when the anchors find
    it. Measured 2026-08-17: `pbmc3k` is the only `kind: path` non-generated ref of fourteen."""
    from manyruns import shell
    from manyruns.catalog import discover_datasets, load_dataset

    monkeypatch.delenv("MANYRUNS_DATASET_DIR", raising=False)
    drop = tmp_path / "drop"
    drop.mkdir()
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(drop))
    monkeypatch.chdir(tmp_path)

    file_refs = {n: (load_dataset(n).get("handle") or {}).get("ref")
                 for n in discover_datasets()
                 if (load_dataset(n).get("handle") or {}).get("kind") == "path"
                 and not str((load_dataset(n).get("handle") or {}).get("ref")).startswith(
                     "synthetic:")}
    assert file_refs, "the catalog no longer declares any file ref; this test is now vacuous"

    for name, ref in file_refs.items():
        ds = load_dataset(name)
        assert shell.missing_dataset_file(ds) == ref, (
            f"{name}: nothing is on this machine yet, so it must report {ref!r} as absent"
        )
        (drop / Path(ref).name).write_text("stand-in for the declared bytes\n")
        assert shell.missing_dataset_file(ds) is None, (
            f"{name}: the declared file is in the drop folder and it is still called missing"
        )
