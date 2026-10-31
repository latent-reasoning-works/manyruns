"""`run_id` names one STATE LINEAGE — the record's answer to "one run or many".

"phate, then mioflow, then a different mioflow" is two different experiments and the record
has to be able to tell them apart:

  chained  — the third step consumes the second's output (`state["emb"]` is mutated in
             place). ONE lineage, three steps, indices 0/1/2.
  rewound  — the person goes back to phate's embedding and takes a different next step.
             TWO lineages sharing a prefix, the second carrying `parent={run_id, index}`.

The alternatives were rejected on what they break, not on taste. One `run_id` per SESSION
cannot express the rewind at all: `NN` in `state/<run_id>/NN-step_slot.npy` is a POSITION, so
a branch written into the same folder needs it to become a graph node, and the folder's
`COMPLETE` marker would have to mean "finished, and also still growing". One `run_id` per
STEP destroys the person's unit of meaning and pays the full metric suite per step.

Everything below was measured with `engine=real` (no private stack) before that engine was
removed; the loop it named is now `run_inproc`, and the measurements stand as taken. And
each assertion was checked by breaking the thing it names. Measured, the rewind case:

    parent  run_id 0a6d07fdad7c  steps [0, 1]  state/0a6d07fdad7c/
                                               00-phate_emb.npy 01-mioflow_pseudotime.npy
    branch  run_id 9bd7bb526c1f  parent {'run_id': '0a6d07fdad7c', 'index': 0}
                                               state/9bd7bb526c1f/00-mioflow_pseudotime.npy

with the parent's two files unchanged and its step-0 embedding still loadable afterwards.
"""
from __future__ import annotations

import json

import pytest

from manyruns import artifacts, store
from manyruns.pipeline import runner
from manyruns.session import Session

np = pytest.importorskip("numpy")


CHAIN = {"name": "cflows", "steps": [
    {"name": "phate", "group": "latent", "params": {"n_components": 3}},
    {"name": "mioflow", "group": "lightning", "params": {}},
]}
#: two LATENT steps, so the lineage's current embedding is not the one step 0 wrote — which
#: is what makes "the branch starts from the step it names" a testable claim at all.
REEMBED = {"name": "reembed", "steps": [
    {"name": "phate", "group": "latent", "params": {"n_components": 3}},
    {"name": "phate", "group": "latent", "params": {"n_components": 2}},
]}
EMBED_ONLY = {"name": "embed", "steps": [
    {"name": "phate", "group": "latent", "params": {"n_components": 3}},
]}


def _data(n=150, d=12, seed=0):
    return np.asarray(np.random.default_rng(seed).normal(size=(n, d)))


def _real(tmp_path, recipe=CHAIN, **kw):
    pytest.importorskip("sklearn")
    pytest.importorskip("phate")
    return Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                   recipe=recipe, array=_data(), seed=0, **kw)


def _names(root):
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


# ── the chain: the degenerate case, unchanged ────────────────────────────────────────────
def test_a_chain_is_one_lineage_with_one_id_and_a_monotonic_index(tmp_path):
    """Two steps issued one after another are ONE run, not two, and the artifact folder stays
    positional. This is the case that must not move: an atomic `manyruns run` is exactly this
    lineage issued all at once, so if it changed shape every row already in `index.jsonl`
    would stop meaning what it meant."""
    s = _real(tmp_path)
    s.run_recipe()
    res = s.close()

    assert [r["index"] for r in res["steps"]] == [0, 1]
    assert res["parent"] is None
    assert _names(artifacts.root(tmp_path, s.run_id)) == [
        "00-phate_emb.npy", "01-mioflow_pseudotime.npy", "COMPLETE"]
    # one lineage, one folder — no second `state/` directory was minted along the way
    assert [p.name for p in (tmp_path / "state").iterdir()] == [s.run_id]


def test_an_atomic_run_is_a_lineage_with_no_parent(tmp_path):
    """`parent` is the ONE schema addition, and it is None for every row written today. A run
    that starts from data has no earlier state to name; the field being present-and-None is
    what keeps a reader from having to know which writer produced the row."""
    res = runner.run_inproc(_data(), CHAIN, tmp_path, seed=0)

    assert "parent" in res and res["parent"] is None
    store.append(res, out_dir=tmp_path)
    assert next(iter(store.read(tmp_path)))["parent"] is None


# ── the rewind: the second lineage ───────────────────────────────────────────────────────
def test_a_rewind_is_a_second_lineage_that_names_the_state_it_started_from(tmp_path):
    """THE payload. A branch is a new `run_id` with its own folder and its own `NN` sequence,
    and the edge back to where it came from is recorded rather than lost.

    The parent's artifacts are asserted intact because the alternative — writing the branch
    into the parent's folder — silently destroys them: same index, same step name, same
    filename, second write wins. That is measured in `tests/test_apply_step.py` for the
    within-lineage case and it is the same mechanism here."""
    s = _real(tmp_path)
    s.run_recipe()
    before = _names(artifacts.root(tmp_path, s.run_id))

    b = s.branch(0)
    b.step("mioflow")
    res = b.close()

    assert b.run_id != s.run_id
    assert res["parent"] == {"run_id": s.run_id, "index": 0}
    # its own folder, its own positions, starting at 00 again
    assert _names(artifacts.root(tmp_path, b.run_id)) == [
        "00-mioflow_pseudotime.npy", "COMPLETE"]
    assert _names(artifacts.root(tmp_path, s.run_id)) == before
    assert artifacts.load_array(s.steps[0]["artifacts"]["emb"]).shape == (150, 3)


def test_a_branch_starts_from_the_step_it_names_not_from_the_latest_state(tmp_path):
    """Rewinding means rewinding. The lineage's live `state["emb"]` is the LAST embedding —
    after a second `phate` at `n_components: 2` the 3-D one is gone from the process — so the
    coordinates have to come off disk, from the file `_persist` wrote when that step ran.

    Measured: after the two-`phate` lineage the session holds a 150x2 embedding, and
    `branch(0)` opens on the 150x3 array `00-phate_emb.npy` holds, element for element."""
    s = _real(tmp_path, recipe=REEMBED)
    s.run_recipe()
    step0 = artifacts.load_array(s.steps[0]["artifacts"]["emb"])

    assert s.state["emb"].shape == (150, 2)          # the live state moved on

    b = s.branch(0)
    assert b.state["emb"].shape == (150, 3)
    assert np.array_equal(b.state["emb"], step0)


def test_a_branch_inherits_the_data_so_the_two_lineages_stay_comparable(tmp_path):
    """`n_samples`/`n_features` come from `state["X"]`, and a branch that dropped the input
    would answer a different question with a shorter g-vector — which is precisely what
    recording the edge exists to prevent. The supplied embedding is also declared, so the
    unusual route is reported rather than silently taken."""
    s = _real(tmp_path)
    s.run_recipe()
    b = s.branch(0)
    b.step("mioflow")
    res = b.close()

    assert res["g_vector"]["n_samples"] == 150 and res["g_vector"]["n_features"] == 12
    assert b.provided == {"embedding"}
    assert any("not from a latent step" in c for c in res["caveats"])


def test_a_branch_does_not_claim_the_recipe_its_parent_ran(tmp_path):
    """The branch did not run `cflows`; it inherited one step's output and went its own way.
    Claiming the name would make `recipe` and `spec_id` describe different analyses, and
    `None` already renders as "(adaptive)" at both readers."""
    s = _real(tmp_path)
    s.run_recipe()
    b = s.branch(0)
    b.step("mioflow")

    assert s.close()["recipe"] == "cflows"
    assert b.close()["recipe"] is None


def test_branching_does_not_seal_the_parent(tmp_path):
    """`finish_carry` stays the ONLY writer of `COMPLETE`, and a branch is not a close: the
    parent may still be stepped. Measured — after `branch()` the parent's folder has no
    marker; after `parent.close()` it does, and the manifest describes both of its files."""
    s = _real(tmp_path)
    s.run_recipe()
    root = artifacts.root(tmp_path, s.run_id)

    b = s.branch(0)
    b.step("mioflow")
    b.close()
    assert artifacts.complete(root) is False       # the branch closed; the parent did not

    s.close()
    assert artifacts.complete(root) is True
    assert len(json.loads((root / artifacts.DONE).read_text())) == 2


def test_a_step_that_wrote_no_embedding_cannot_be_rewound_to(tmp_path):
    """A refusal, not a fallback. `mioflow` on the in-process loop sets `pseudotime` and leaves the
    embedding object alone, so `_persist` writes no `emb` slot for it — there is nothing at
    that address to start from. Falling back to the live `state["emb"]` would branch from
    coordinates the person did not name, and the two branches would be indistinguishable in
    every record either one produced."""
    s = _real(tmp_path)
    s.run_recipe()

    assert "emb" not in (s.steps[1].get("artifacts") or {})
    with pytest.raises(ValueError, match="no embedding on disk"):
        s.branch(1)
    with pytest.raises(IndexError):
        s.branch(2)


def test_the_mock_engine_has_nothing_to_rewind_to_and_says_so(tmp_path):
    """The mock transforms a stand-in vector and writes no arrays at all, so every step in a
    mock lineage is unbranchable. It refuses rather than handing back a lineage seeded from
    `None`, which would run and record `parent` as though state had been inherited."""
    s = Session(project="p", engine="mock", out_dir=tmp_path, modality="scrna", recipe=CHAIN)
    s.run_recipe()

    with pytest.raises(ValueError, match="no embedding on disk"):
        s.branch(0)


# ── the two questions a comparison reader has to ask ─────────────────────────────────────
def test_same_spec_and_same_starting_state_are_two_separate_questions(tmp_path):
    """`spec_id` is "was this the same analysis"; `parent` is "was it the same input". Neither
    implies the other, which is why `parent` is NOT hashed into `spec_id`.

    Two branches of two different lineages, each issuing `mioflow`, are the same analysis on
    different coordinates: equal `spec_id`, different `parent`. A reader that could only see
    `spec_id` would read them as re-runs of one another."""
    a, b = _real(tmp_path / "a"), _real(tmp_path / "b")
    a.run_recipe()
    b.run_recipe()
    assert a.close()["spec_id"] == b.close()["spec_id"]        # same recipe, seed, dataset

    ba, bb = a.branch(0), b.branch(0)
    ba.step("mioflow")
    bb.step("mioflow")
    ra, rb = ba.close(), bb.close()

    assert ra["spec_id"] == rb["spec_id"]                      # the same question asked
    assert ra["parent"] != rb["parent"]                        # of two different states
    assert ra["run_id"] != rb["run_id"]


def test_the_branch_is_not_a_rerun_of_the_lineage_it_came_from(tmp_path):
    """A branch issues a shorter sequence over inherited state, so it must not hash to the
    spec of the lineage that produced that state — otherwise "has this analysis been run
    before" answers yes for an analysis that has not been run."""
    s = _real(tmp_path)
    s.run_recipe()
    b = s.branch(0)
    b.step("mioflow")

    assert b.close()["spec_id"] != s.close()["spec_id"]


def test_the_store_row_carries_the_edge(tmp_path):
    """The row is what a comparison reader gets. `store.read` has only test callers today and
    `compare` does not exist — so the point of the column is that when that reader is built,
    the edge is already in the history rather than needing a migration."""
    s = _real(tmp_path)
    s.run_recipe()
    b = s.branch(0)
    b.step("mioflow")
    store.append(b.close(), out_dir=tmp_path)
    store.append(s.close(), out_dir=tmp_path)

    branch_row, parent_row = list(store.read(tmp_path))
    assert parent_row["parent"] is None
    assert branch_row["parent"] == {"run_id": parent_row["run_id"], "index": 0}
    # survives the json round trip as a dict, not as a positional pair
    assert set(branch_row["parent"]) == {"run_id", "index"}


# ── the address, as a value ──────────────────────────────────────────────────────────────
def test_one_form_reaches_the_record_whichever_form_the_caller_used(tmp_path):
    """`(run_id, index)` reads naturally at a call site and `{"run_id", "index"}` is what the
    row has to hold — a tuple would come back from JSON as a list, giving two readers for one
    field. The coercion happens once, in `as_parent`, so the record has one form."""
    assert runner.as_parent(("abc", 0)) == {"run_id": "abc", "index": 0}
    assert runner.as_parent({"run_id": "abc", "index": 0}) == {"run_id": "abc", "index": 0}
    assert runner.as_parent(None) is None

    res = runner.run_inproc(_data(), {"name": "m", "steps": [
        {"name": "mioflow", "group": "lightning", "params": {}}]}, tmp_path, seed=0,
        embedding=_data(d=3), parent=("abc", 0))
    assert res["parent"] == {"run_id": "abc", "index": 0}
    assert json.loads(json.dumps(res["parent"])) == res["parent"]


@pytest.mark.parametrize("ref", [("abc",), {"run_id": "abc"}, {"index": 0}, ("abc", None)])
def test_half_an_address_is_refused_at_open(ref, tmp_path):
    """A lineage that claims a parent it cannot name is worse than one that claims none: the
    edge reads as recorded and resolves to nothing. Refused before any compute runs, so the
    caller's bug does not cost a fit."""
    with pytest.raises(ValueError, match="run_id and an index"):
        runner.run_inproc(_data(), CHAIN, tmp_path, seed=0, parent=ref)


# ── the front door writes the history it produces (§4.3 P1) ──────────────────────────────
class _Prompter:
    def __init__(self, selects):
        self._selects = list(selects)

    def select(self, message, choices, default=None):
        assert self._selects, f"unexpected select: {message}"
        return self._selects.pop(0)

    def text(self, message, default=""):
        return ""


class _Console:
    is_terminal = False

    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args):
        self.lines.append(" ".join(str(a) for a in args))

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def test_the_front_door_records_the_run_it_just_did(tmp_path, monkeypatch):
    """`store.append` had exactly one product caller — `app._explore_project`, the `run` /
    `init --batch` path — and `shell.py` referenced `store` nowhere. Measured before this:
    a full mock run through `run_shell` produced a record with a `run_id` and left no
    `outputs/index.jsonl` at all, so the state folder it wrote was addressable only by a run
    id nobody ever saw. That is the prerequisite a resume menu starts from."""
    pytest.importorskip("numpy")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MANYRUNS_DATA_DIR", str(tmp_path / "data"))
    from manyruns import app, shell

    con = _Console()
    pr = _Prompter(["explore", ("dataset", "swissroll"), "cflows", "quit"])
    rc = shell.run_shell(app._build_parser().parse_args(["shell", "--engine", "mock"]),
                         prompter=pr, console=con)

    assert rc == 0
    rows = list(store.read(tmp_path / "outputs"))
    assert len(rows) == 1, f"the front door recorded {len(rows)} runs"
    assert rows[0]["run_id"] and rows[0]["parent"] is None
    assert rows[0]["run_id"] in con.text      # and told the person where it went


def test_the_repl_records_the_lineage_it_stepped(tmp_path, monkeypatch):
    """The other interactive door, and the same gap: `manyruns init` on a steppable engine
    opens the REPL, which now closes its lineage with a `COMPLETE` marker — and, until this,
    with no row naming the folder that marker guards.

    A lineage that ran no step writes no row: `_finalize` calls a zero-step run `ok` because
    every step succeeded vacuously, and a history full of those describes work that never
    happened. Both halves are asserted here."""
    monkeypatch.chdir(tmp_path)
    from manyruns import app
    from manyruns.session import Session

    def _repl(script):
        s = Session(project="p", engine="mock", out_dir=tmp_path, modality="scrna",
                    recipe=app.load_recipe("cflows"))
        out: list = []
        app.interactive_session(s, read=lambda _="": script.pop(0), write=out.append)
        return s, "\n".join(out)

    stepped, said = _repl(["phate", "accept", "quit"])
    rows = list(store.read(tmp_path / "outputs"))
    assert [r["run_id"] for r in rows] == [stepped.run_id]
    assert stepped.run_id in said

    _repl(["quit"])                       # opened, closed, ran nothing
    assert len(list(store.read(tmp_path / "outputs"))) == 1


def test_the_repl_closes_the_lineage_it_stepped(tmp_path, monkeypatch):
    """The REPL is the only thing that can end a hand-driven lineage, and it must CLOSE it.

    The test above is on the mock, and the mock cannot see this: it transforms a stand-in
    vector and writes no arrays, so `carry["written"]` is empty and `finish_carry` is a no-op —
    `close()` and `results()` return byte-identical records there. Measured: replacing
    `session.close()` with `session.results()` in `app.interactive_session` and running the
    whole suite gave 516 passed, 1 skipped, unchanged. On an engine that actually writes
    arrays the difference is the `COMPLETE` marker, which is the gate a resume menu reads
    (`artifacts.complete` → `store.read` → `load_array`) — so without it every REPL lineage is
    permanently indistinguishable from one that was interrupted mid-step.
    """
    pytest.importorskip("sklearn")
    pytest.importorskip("phate")
    monkeypatch.chdir(tmp_path)
    from manyruns import app

    s = Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                recipe=CHAIN, array=_data(), seed=0)
    script = ["phate", "accept", "quit"]
    app.interactive_session(s, read=lambda _="": script.pop(0), write=lambda *_: None)

    root = artifacts.root(tmp_path, s.run_id)
    assert _names(root) == ["00-phate_emb.npy", "COMPLETE"]
    assert artifacts.complete(root) is True
    # and the manifest describes the folder it marks, rather than being an empty stamp
    assert list(json.loads((root / artifacts.DONE).read_text())) == [
        str(s.steps[0]["artifacts"]["emb"])]


def test_a_mock_lineage_is_identifiable_in_the_store(tmp_path, monkeypatch):
    """A row whose numbers were invented must say so where the history is read.

    `store.append`'s schema had no `caveats` column, so a mock run landed in `index.jsonl`
    looking exactly like a measured one. That was survivable while only `_explore_project`
    appended; the interactive doors now append too, so mock lineages reach the history."""
    from manyruns import store
    from manyruns.serving import LocalServer

    monkeypatch.chdir(tmp_path)
    recipe = {"name": "r", "steps": [{"name": "phate", "group": "latent", "params": {}}]}
    store.append(LocalServer(engine="mock").predict({"recipe": recipe}))

    row = list(store.read())[0]
    assert row["engine"] == "mock"
    assert any("invented" in c for c in row["caveats"]), row["caveats"]


def test_a_measured_run_carries_no_caveat(tmp_path):
    """Quiet unless something is genuinely worth saying — a column that is always populated
    is a column nobody reads."""
    import numpy as np

    from manyruns import catalog, store
    from manyruns.pipeline import runner

    # 8 columns → 60. `embed` gained a declared `normalize → transform → pca(50)` prep block in
    # the cutover (#54), and `pca` declares `limits: {n_components: [n_samples, n_features]}`,
    # so on an 8-column array `bounds.fit` clamps 50 → 8 and writes a caveat saying so. That
    # caveat is worth reading — the run did not do what the recipe declared — so the fix is
    # data wide enough for the declared 50 to stand, not a looser assertion. (The Gaussian
    # point cloud makes `normalize`/`transform` decline; a decline is recorded as
    # `outcome="skipped"` on the step, not as a caveat, which is why this column stays empty.)
    res = runner.run_inproc(np.random.default_rng(0).normal(size=(120, 60)),
                              catalog.load_recipe("embed"), tmp_path, seed=0)
    store.append(res, out_dir=tmp_path)

    assert list(store.read(out_dir=tmp_path))[0]["caveats"] == []


# ── a rewind past a filter ───────────────────────────────────────────────────────────────
#
# Reachable only since the `prep` group started running (`runner._apply_transition`): a
# session's data can now SHRINK mid-lineage, and `branch` hands the child that data as it now
# stands. The two tests below are the pair that keeps the child coherent — the labels travel
# with the matrix they index, and a rewind to an embedding that predates the filter is refused.


def _filtering_session(tmp_path, mask):
    """A real-engine session that embeds, then drops cells, using a prep step registered for
    the duration. Task 6 lands the declared filters; the step loop is what is under test."""
    from manyruns.pipeline import prep as _prep

    s = _real(tmp_path, recipe={"name": "f", "steps": [
        {"name": "phate", "group": "latent", "params": {"n_components": 3}},
        {"name": "cut", "group": "prep", "params": {}}]},
        labels=np.array(["d0"] * 75 + ["d9"] * 75))
    _prep._PREP_STEPS["cut"] = lambda view, X, params: {"mask": mask, "axis": "rows"}
    try:
        s.run_recipe()
    finally:
        _prep._PREP_STEPS.pop("cut", None)
    return s


def test_a_filter_moves_the_labels_with_the_matrix_they_index(tmp_path):
    """One selection, so the two cannot disagree — asserted at the SESSION, because a session
    is where the two are held apart longest (`self.labels` is what it opened with, `ctx["array"]`
    is what steps have done since). A branch takes both, and taking one of each would attribute
    every cell to the wrong donor with nothing downstream raising."""
    keep = np.zeros(150, dtype=bool)
    keep[:40] = True
    s = _filtering_session(tmp_path, keep)

    assert s.steps[1]["dropped"] == {"axis": "rows", "n": 110, "remaining": 40}
    assert s.state["labels"].tolist() == ["d0"] * 40
    assert s.ctx["array"].shape[0] == len(s.state["labels"]) == 40
    assert len(s.ctx["color"]) == 40, "the colouring was rebuilt, not left at 150"


def test_a_rewind_to_an_embedding_that_predates_a_filter_is_refused(tmp_path):
    """A refusal, not a silent mispairing. `branch` hands the child THIS lineage's data, which
    is now 40 cells; step 0's embedding on disk is over 150. The child would run every step
    with a 150-row embedding beside a 40-row matrix and a 40-row label vector — `separation`
    mis-aligns, `_save_scatter` silently drops the colouring (`io.py:106`), and both branches
    look identical in the record."""
    keep = np.zeros(150, dtype=bool)
    keep[:40] = True
    s = _filtering_session(tmp_path, keep)

    assert artifacts.load_array(s.steps[0]["artifacts"]["emb"]).shape == (150, 3)
    with pytest.raises(ValueError, match="predates a filter"):
        s.branch(0)


def test_a_branch_after_a_filter_inherits_the_labels_that_match_its_data(tmp_path):
    """The legal half of the rewind, and the line the refusal above does not cover: filter
    FIRST, embed second, branch from the embedding. That child is coherent — its embedding was
    fitted over the cells that remain — so it must receive the 40 labels state holds and not
    the 150 the session opened with. Mutation-checked: `labels=self.labels` in `branch` gives
    the child a 150-row label vector beside a 40-row matrix."""
    from manyruns.pipeline import prep as _prep

    keep = np.zeros(150, dtype=bool)
    keep[:40] = True
    s = _real(tmp_path, recipe={"name": "f", "steps": [
        {"name": "cut", "group": "prep", "params": {}},
        {"name": "phate", "group": "latent", "params": {"n_components": 3}}]},
        labels=np.array(["d0"] * 75 + ["d9"] * 75))
    _prep._PREP_STEPS["cut"] = lambda view, X, params: {"mask": keep, "axis": "rows"}
    try:
        s.run_recipe()
    finally:
        _prep._PREP_STEPS.pop("cut", None)

    b = s.branch(1)

    assert b.state["X"].shape[0] == len(b.state["labels"]) == 40
    assert b.state["emb"].shape == (40, 3)
    assert b.state["rows"].tolist() == list(range(40)), "the child's own selection is complete"


def test_a_rewind_survives_a_filter_that_dropped_nothing(tmp_path):
    """The refusal above is keyed on the presence of `dropped`, and a filter that removed no
    cell writes one too — `detect_doublets` with `remove: false` returns an all-True mask by
    design (spec §5, "flags without narrowing"), and on pbmc3k `filter_cells(min_genes=200)`
    drops exactly 0 of 2700.

    Measured before the fix: `branch(0)` raised `step 0 (phate)'s embedding predates a filter:
    cut dropped 0 rows since` — a message that states its own falsity, and a lineage refused
    for the rest of its life over a filter that changed nothing. The record keeps saying
    `dropped: {n: 0}`, because that is a measurement; what moved is the refusal, which now
    asks how many."""
    s = _filtering_session(tmp_path, np.ones(150, dtype=bool))

    assert s.steps[1]["dropped"] == {"axis": "rows", "n": 0, "remaining": 150}
    b = s.branch(0)

    assert b.state["emb"].shape == (150, 3)
    assert b.parent == {"run_id": s.run_id, "index": 0}

# ── resume: the prerequisite named above, closing the loop ───────────────────────────────
#
# `open` builds a brand-new `Session` per process — a fresh, empty `state` — so a step run in
# an earlier `manyruns open` and then quit out of was, until this, simply gone: `dpt` (needs
# `state["emb"]`) refused with "none was supplied and no step produced one" even though the
# embedding it wanted was sitting on disk one directory down. `session.resumable` is the read
# half `artifacts.complete` → `load_array` promised; `_build_session` is the writer that
# threads it into the next process's state before step 0.
def test_resumable_is_none_for_a_project_with_no_completed_run(tmp_path):
    from manyruns.session import resumable

    assert resumable(tmp_path) is None


def test_resumable_finds_the_last_run_s_embedding(tmp_path):
    from manyruns.session import resumable

    s = _real(tmp_path, recipe=EMBED_ONLY)
    s.step("phate")
    s.close()

    found = resumable(tmp_path)
    assert found["run_id"] == s.run_id
    assert found["steps"] == [("phate", "emb", 0)]
    np.testing.assert_array_equal(found["arrays"]["emb"], artifacts.load_array(
        s.steps[0]["artifacts"]["emb"]))


def test_resumable_picks_the_most_recently_closed_lineage(tmp_path):
    """Two completed runs in the same project (two `open`/`quit` cycles) — the newer one
    wins, same as reopening a process picks up where the LAST session left off, not the
    first one that happened to exist."""
    import time

    from manyruns.session import resumable

    older = _real(tmp_path, recipe=EMBED_ONLY)
    older.step("phate")
    older.close()
    time.sleep(0.02)
    newer = _real(tmp_path, recipe=EMBED_ONLY)
    newer.step("phate")
    newer.close()

    assert resumable(tmp_path)["run_id"] == newer.run_id


def test_resumable_does_not_let_a_later_completion_shadow_an_older_fact(tmp_path):
    """THE bug report this closes, reproduced exactly: session A computes an embedding AND a
    pseudotime and closes clean; session B resumes the embedding, re-runs `phate` (never
    touches `dpt`), and ALSO closes clean — no crash anywhere. Reading only the single latest
    completed run (the old behaviour) would return `emb` from B and silently drop the
    `pseudotime` A computed, even though nothing failed and it is still sitting on disk.
    `discretize_time` then refuses for want of a pseudotime that objectively exists."""
    import time

    pytest.importorskip("scanpy")
    from manyruns.session import resumable

    older = _real(tmp_path, recipe={"name": "pseudotime", "steps": [
        {"name": "phate", "group": "latent", "params": {"n_components": 3}},
        {"name": "dpt", "group": "analysis", "params": {}},
    ]})
    older.step("phate")
    older.step("dpt")
    older.close()
    time.sleep(0.02)
    newer = _real(tmp_path, recipe=EMBED_ONLY, embedding=older.state["emb"])
    newer.step("phate")   # a fresh embedding — no dpt in THIS lineage
    newer.close()

    found = resumable(tmp_path)
    assert found["run_id"] == newer.run_id, "the header still names the latest completion"
    assert set(found["arrays"]) == {"emb", "pseudotime"}, (
        "the older run's pseudotime must survive a newer, unrelated completion"
    )
    assert found["origin"]["emb"] == newer.run_id
    assert found["origin"]["pseudotime"] == older.run_id
    np.testing.assert_array_equal(
        found["arrays"]["pseudotime"],
        artifacts.load_array(older.steps[1]["artifacts"]["pseudotime"]),
    )
    np.testing.assert_array_equal(
        found["arrays"]["emb"], artifacts.load_array(newer.steps[0]["artifacts"]["emb"])
    )


def test_resumable_skips_one_corrupted_runs_marker_but_still_resumes_the_others(tmp_path):
    """One bad `COMPLETE` file (`finish()`'s non-atomic write, per this function's docstring)
    must cost the fold only ITS OWN run, not every fact any other completed run holds."""
    from manyruns.session import resumable

    good = _real(tmp_path, recipe=EMBED_ONLY)
    good.step("phate")
    good.close()

    bad = _real(tmp_path, recipe=EMBED_ONLY)
    bad.step("phate")
    bad.close()
    (tmp_path / "state" / bad.run_id / "COMPLETE").write_text("{not json")

    found = resumable(tmp_path)
    assert found is not None
    assert found["run_id"] == good.run_id
    np.testing.assert_array_equal(
        found["arrays"]["emb"], artifacts.load_array(good.steps[0]["artifacts"]["emb"])
    )


def test_a_resumed_state_satisfies_the_embedding_a_step_needs(tmp_path):
    """The regression this feature exists to close, at the exact seam that raised it:
    `_require_embedding` refusing `dpt` for "none was supplied and no step produced one" when
    the embedding was computed by a PRIOR process. A resumed `Session` must satisfy it with no
    `phate` (or any latent step) run in its OWN lineage."""
    from manyruns.pipeline.steps import _require_embedding
    from manyruns.session import Session, resumable

    first = _real(tmp_path, recipe=EMBED_ONLY)
    first.step("phate")
    first.close()

    resumed = resumable(tmp_path)
    second = Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                     recipe={"name": "pseudotime", "steps": []}, array=_data(), seed=0,
                     embedding=resumed["arrays"]["emb"],
                     parent={"run_id": resumed["run_id"], "index": resumed["index"]["emb"]})

    _require_embedding(second.state, "dpt")           # must not raise
    assert second.steps == []                          # nothing ran in THIS lineage
    assert second.parent == {"run_id": first.run_id, "index": 0}


def _resume_args():
    """The namespace `_build_session` reads, and only that: `seed`, `time_key` and
    `data_kwargs` are read BARE (`app.py:441/471/479`), the rest through `getattr`."""
    import argparse

    return argparse.Namespace(seed=0, time_key=None, data_kwargs=None, device=None,
                              smoke=True, full=False)


def _resume_project(recipe):
    return {"project": "p", "data_folder": None, "dataset": None, "engine": "_inproc",
            "modality": "scrna", "recipe": recipe}


def test_build_session_resumes_by_default_because_open_means_continue(tmp_path, monkeypatch):
    """The default must not move. `manyruns open` reopening the SAME project is "continue
    where I left off", and this is the seam that makes it true — `cmd_open` names no keyword."""
    from manyruns import app

    monkeypatch.chdir(tmp_path)
    first = _real(tmp_path / "outputs" / "p", recipe=EMBED_ONLY)
    first.step("phate")
    first.close()

    resumed = app._build_session(_resume_project(EMBED_ONLY), _resume_args())

    assert resumed.state.get("emb") is not None, "a resumed session started empty"
    assert resumed.parent == {"run_id": first.run_id, "index": 0}
    assert resumed.resumed is not None


def test_build_session_can_be_told_not_to_resume_and_then_starts_clean(tmp_path, monkeypatch):
    """`resume=False` — for a caller whose verb is not "continue". There is exactly one, the
    app's stepped front door (`tui/app.py:_stepped_run`), whose gesture is "pick data, pick a
    recipe, run".

    TWO THINGS BREAK WITHOUT IT and the second is the serious one. The two front doors would
    record different lineage for one gesture (`explore_once` never resumes and writes
    `parent: None`); and `run_tune_loop` snapshots `baseline_state` at entry and `_reset()`
    restores it on CANCEL, so cancelling the embedding step would leave the RESUMED embedding
    sitting in `state` — and `session.close()` → `_finalize` would then report `n_embedded`,
    `final_dim` and the whole metric suite over an embedding this run never computed.

    Asserted as "indistinguishable from a project with no completed run", which is the whole
    property: all four consequences of the resume — the embedding, the `parent`, the pseudotime
    seed and `session.resumed` — have to go together or the opt-out is half an opt-out.
    """
    from manyruns import app
    from manyruns.session import resumable

    monkeypatch.chdir(tmp_path)
    first = _real(tmp_path / "outputs" / "p", recipe=EMBED_ONLY)
    first.step("phate")
    first.close()
    # the resume is genuinely THERE to be declined — otherwise this passes vacuously
    assert resumable(tmp_path / "outputs" / "p") is not None

    fresh = app._build_session(_resume_project(EMBED_ONLY), _resume_args(), resume=False)

    assert fresh.state.get("emb") is None, "the front door inherited a previous run's embedding"
    assert fresh.parent is None, "a fresh analysis was stamped as a branch of an earlier run"
    assert fresh.resumed is None
    # `.get(...) is None`, not `not in`: `_new_state` pre-creates every slot with a `None`, so
    # the question is whether anything was SEEDED, not whether the key exists.
    assert fresh.state.get("pseudotime") is None


def test_the_repl_announces_a_resumed_state(tmp_path):
    """`interactive_session` prints the resume, so a person typing `dpt` cold sees WHY it
    works rather than being left to infer it — the whole gap this feature closes was that
    silence, just pointed the other way (a refusal with no visible cause)."""
    from manyruns import app
    from manyruns.session import Session

    s = Session(project="p", engine="_inproc", out_dir=tmp_path, modality="scrna",
                recipe={"name": "pseudotime", "steps": []}, array=_data(), seed=0,
                embedding=_data(d=3))
    s.resumed = {"run_id": "abcdef123456", "steps": [("phate", "emb", 0)]}

    out: list = []
    app.interactive_session(s, read=lambda _="": "quit", write=out.append)

    assert any("resumed from run abcdef12" in line and "phate→emb" in line for line in out)
