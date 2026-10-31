"""The trained model is KEPT, and can be asked questions — `probe` steps over `state["model"]`.

The original failure was inside our own code:

    emb, scores = _mioflow._run_mioflow_experiment(...)
    state["emb"] = emb

> **The trained flow — the simulatable object, the model — is discarded at the return
> statement, and only its coordinates are kept.** The one step in the product capable of
> producing a world model throws it away and stores the coordinate system.

It is discarded by MANYRUNS's own return line, not by anything upstream: `run_experiment`
takes the algorithm as an argument and the caller owns it throughout. So retaining it is a
return value, and what it buys is the difference between a transcript and a model — a
transcript answers "what did the embedding look like", a model answers "what happens to THESE
cells under the flow".

WHY A PROBE IS A STEP AND NOT A NEW AXIS. A probe reads a fact a `lightning` step produced and
writes to the g-vector; it transforms no data and advances no selection. That is a step group
plus an executor, which is exactly the extension path `CLAUDE.md` prescribes for a new
capability — not a fourth axis beside capability / substrate / fidelity.
"""
from __future__ import annotations

import pytest

from manyruns import vocab
from manyruns.pipeline import runner as _runner

np = pytest.importorskip("numpy")


class _FakeFlow:
    """A trained model, as far as this seam is concerned: an object with methods.

    Deliberately not a MIOFlow — the point under test is that manyruns keeps and calls
    whatever the engine trained, never that it understands the class. A test that imported
    MIOFlow would be testing torch.
    """

    def __init__(self, n=6, dim=2):
        self.n, self.dim = n, dim
        self.calls = []

    def encode(self, x):
        self.calls.append(("encode", len(x)))
        return np.zeros((len(x), self.dim), dtype=np.float32)

    @property
    def trajectories(self):
        return np.zeros((4, self.n, self.dim), dtype=np.float32)


def _framed(n=6, d=3):
    X = np.arange(n * d, dtype=np.float32).reshape(n, d)
    return _runner._new_state(X=X, labels=np.array(list("aabbcc"))[:n], seed=0, counts=X.copy(),
                              genes=np.array(list("xyz"))[:d])


def _ctx(tmp_path):
    return {"out_dir": tmp_path, "run_id": "RUN", "plots": [], "target_dim": 2,
            "caveats": [], "seed": 0, "fast_dev_run": True}


def _apply(state, step, dispatch, tmp_path):
    return _runner.apply_step(step, state, {}, dispatch=dispatch, ctx=_ctx(tmp_path),
                              index=0, carry=_runner.new_carry())


# ── the fact ─────────────────────────────────────────────────────────────────
def test_model_is_a_fact_a_named_step_produces():
    """Keyed by NAME, not by group, and the distinction is the same one `STEP_PRODUCES`'
    commentary draws for `clusters`. `GROUP_PROVIDES` says a fact is true of every executor in
    the group; the in-process `lightning` executor (`steps._step_mioflow`) computes a diffusion
    pseudotime and trains nothing, so "a lightning step produces a model" is true of one engine
    and a lie about another — the exact test that keeps `lightning` out of `GROUP_PROVIDES`
    today."""
    assert vocab.STEP_PRODUCES["mioflow"] == ("model",)
    assert "model" not in vocab.GROUP_PROVIDES.get("lightning", ())


def test_a_probe_before_a_fit_is_refused_at_plan_time():
    """The ordering the fact exists to enforce. A probe with no trained model anywhere is the
    composition that cannot compose, and plan time is where a search over recipes has to get
    its answer — it cannot afford to run each one to find out."""
    recipe = {"name": "t", "steps": [
        {"name": "sample_trajectories", "group": "probe", "params": {}}]}

    assert vocab.unmet(recipe, "time-course") == frozenset({"model"})


def test_a_probe_after_a_fit_is_legal():
    """The other half, and the one that makes the refusal mean something."""
    recipe = {"name": "t", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        {"name": "mioflow", "group": "lightning", "params": {}},
        {"name": "sample_trajectories", "group": "probe", "params": {}}]}

    assert vocab.unmet(recipe, "time-course") == frozenset()


def test_a_filter_between_the_fit_and_the_probe_is_refused():
    """A flow fitted over cells a filter has since dropped is not a model of the ones that
    remain — the same argument that clears `embedding`, and it comes for free because `model`
    is step-produced and `frame_facts` therefore excludes it from `keep`."""
    recipe = {"name": "t", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        {"name": "mioflow", "group": "lightning", "params": {}},
        {"name": "filter_cells", "group": "prep", "params": {"min_genes": 200}},
        {"name": "sample_trajectories", "group": "probe", "params": {}}]}

    assert vocab.unmet(recipe, "time-course") == frozenset({"model"})
    assert vocab.invalidated(recipe, "time-course") == frozenset({"model"}), (
        "and it is reported as CLEARED, not as never-produced — different fixes")


def test_model_is_never_a_frame_fact():
    """The disjointness the invalidation rests on. If `model` were ever in `frame_facts` a
    narrowing would silently stop clearing it, and a probe would run against a flow fitted over
    cells that are gone — a plausible-looking answer, which is the failure mode this repo
    refuses everywhere else."""
    assert "model" not in vocab.frame_facts()
    assert "model" in vocab.step_facts()


# ── the state slot ───────────────────────────────────────────────────────────
def test_the_trained_model_reaches_the_state(tmp_path):
    """§3.1's line, inverted. The executor keeps what it trained."""
    flow = _FakeFlow()
    state = _framed()
    state["emb"] = np.zeros((6, 2), dtype=np.float32)
    dispatch = {"lightning": lambda name, params, st, g, ctx: st.__setitem__("model", flow)}

    _apply(state, {"name": "mioflow", "group": "lightning", "params": {}}, dispatch, tmp_path)

    assert state["model"] is flow


def test_a_narrowing_clears_the_model_and_says_so(tmp_path):
    """The run-time half of the plan-time refusal above. Both have to hold: `unmet` refusing a
    recipe whose model is still in `state` would be a plan-time/run-time disagreement, which is
    what `STEP_NEEDS`' mioflow note exists to prevent."""
    state = _framed()
    state["emb"] = np.zeros((6, 2), dtype=np.float32)
    state["model"] = _FakeFlow()
    dispatch = {"prep": lambda name, params, st, g, ctx: {
        "mask": np.array([True, True, True, False, False, False]), "axis": "rows"}}

    rec = _apply(state, {"name": "filter_cells", "group": "prep", "params": {}},
                 dispatch, tmp_path)

    assert state["model"] is None
    assert "model" in rec["invalidated"]


def test_a_step_that_drops_nothing_keeps_the_model(tmp_path):
    """`detect_doublets(remove: false)` returns an all-True mask by design. Destroying a trained
    flow because a step reported that it removed nothing would be the zero-drop defect that bit
    `emb` and `pseudotime` already (fix f0f5933), repeated on a costlier object."""
    state = _framed()
    flow = _FakeFlow()
    state["model"] = flow
    dispatch = {"prep": lambda name, params, st, g, ctx: {
        "mask": np.ones(6, dtype=bool), "axis": "rows"}}

    rec = _apply(state, {"name": "detect_doublets", "group": "prep", "params": {}},
                 dispatch, tmp_path)

    assert state["model"] is flow
    assert "invalidated" not in rec


def test_the_model_never_reaches_disk(tmp_path):
    """`artifacts.PERSISTED` is `("emb", "pseudotime")` and the model is not in it. A
    LightningModule is not a geometry artifact — it carries weights, and `artifacts.py`'s rule
    is that geometry leaves and the person's data does not. Pickling one into an outputs folder
    is a data-handling decision nobody has taken."""
    from manyruns import artifacts

    assert "model" not in artifacts.PERSISTED


# ── the probe ────────────────────────────────────────────────────────────────
def test_probe_is_a_declared_step_group():
    """A new capability is a group plus an executor — not a new `--engine` value and not a
    fourth axis. `CLAUDE.md` prescribes exactly this shape."""
    assert "probe" in vocab.STEP_GROUPS


def test_sample_trajectories_reads_the_model_and_writes_to_the_g_vector(tmp_path):
    """What a probe IS: it asks the model a question and records the answer. It transforms no
    data, moves no selection, and produces no embedding."""
    state = _framed()
    state["model"] = _FakeFlow()
    g: dict = {}
    ctx = _ctx(tmp_path)

    out = _runner._run_probe_step("sample_trajectories", {}, state, g, ctx)

    assert out is None
    # (T, n, d) = (4, 6, 2): four time bins over six cells. Asserted in this direction because
    # torchdiffeq returns time FIRST, so reading shape[0] as a population size is the natural
    # mistake — and it reads perfectly in a g-vector.
    assert g["sample_trajectories.n_steps"] == 4
    assert g["sample_trajectories.n_trajectories"] == 6
    assert g["sample_trajectories.dims"] == 2
    assert state["emb"] is None, "a probe answers a question; it does not re-embed"


def test_a_probe_with_no_model_declines_with_a_reason(tmp_path):
    """The engine-specific half the calculus cannot express. `unmet` takes no engine argument
    on purpose — its answer has to hold for every engine — so "this engine's lightning step is
    a stand-in that trains nothing" is a RUN-time fact. `_StepSkipped` is how a step declines,
    and the record carries the reason."""
    from manyruns.pipeline.steps import _StepSkipped

    with pytest.raises(_StepSkipped, match="no trained model"):
        _runner._run_probe_step("sample_trajectories", {}, _framed(), {}, _ctx(tmp_path))


def test_an_unknown_probe_declines_rather_than_inventing_an_answer(tmp_path):
    """A probe resolves by NAME from a table manyruns owns, so `dispatchable` can answer
    "could this engine run this step" without running it."""
    from manyruns.pipeline.steps import _StepSkipped

    state = _framed()
    state["model"] = _FakeFlow()
    with pytest.raises(_StepSkipped, match="no probe named"):
        _runner._run_probe_step("read_its_mind", {}, state, {}, _ctx(tmp_path))


def test_a_probe_is_dispatchable_only_where_a_model_can_exist():
    """`mock` invents numbers and trains nothing; the in-process loop computes a pseudotime and
    trains nothing. Offering a probe on either would put a step in the menu that can only ever
    decline — which is the honesty `dispatchable` exists for."""
    assert _runner.dispatchable("manylatents", "sample_trajectories", "probe") is True
    assert _runner.dispatchable(vocab.INPROC, "sample_trajectories", "probe") is False
    assert _runner.dispatchable("mock", "sample_trajectories", "probe") is False
