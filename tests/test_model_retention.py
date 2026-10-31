"""What a model step leaves behind is KEPT, and can be asked questions — `probe` steps.

The original failure was inside our own code: the in-process MIOFlow route returned the trained
flow and manyruns kept only its coordinates. That route is removed. MIOFlow now runs only as a
tool step, through the model seam, and the fitted model stays in the tool's interpreter; what
crosses the seam is `trajectories`, which fills `state["trajectories"]`. So the probe that reads
paths, `sample_trajectories`, reads that slot (`probes.PROBE_READS`). The machinery that keeps a
`state["model"]` (a narrowing clears it; it never reaches disk) stays for any executor that
leaves one.

WHY A PROBE IS A STEP AND NOT A NEW AXIS. A probe reads a fact another step produced and writes
to the g-vector; it transforms no data and advances no selection. That is a step group plus an
executor, which is exactly the extension path `CLAUDE.md` prescribes for a new capability — not a
fourth axis beside capability / substrate / fidelity.
"""
from __future__ import annotations

from pathlib import Path

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
STUB = Path(__file__).resolve().parent / "fixtures" / "mioflow_tool"
TOOL_MIOFLOW = {"name": "mioflow", "group": "lightning", "params": {}, "tool": "mioflow"}


@pytest.fixture()
def mioflow_declared(monkeypatch):
    """The stub MIOFlow folder on the tool search path, so the manifest's facts are read."""
    assert STUB.is_dir()
    monkeypatch.delenv("MANYRUNS_TOOL_DIR", raising=False)
    monkeypatch.setenv("MANYRUNS_TOOL_PATH", str(STUB))


def test_the_mioflow_step_produces_trajectories_not_a_model(mioflow_declared):
    """A tool step's facts are its manifest's. The MIOFlow manifest produces `trajectories`, and
    the fitted model stays in the tool's interpreter, so `model` would be a false claim. The name
    table no longer says `mioflow` produces `model` (requirement "MIOFlow readouts read the
    trajectories slot")."""
    assert "mioflow" not in vocab.STEP_PRODUCES
    assert vocab.step_provides(TOOL_MIOFLOW) == frozenset({"trajectories"})
    assert "model" not in vocab.GROUP_PROVIDES.get("lightning", ())


def test_a_probe_before_a_fit_is_refused_at_plan_time():
    """The ordering the fact exists to enforce. A probe with no trajectories anywhere is the
    composition that cannot compose, and plan time is where a search over recipes has to get
    its answer — it cannot afford to run each one to find out."""
    recipe = {"name": "t", "steps": [
        {"name": "sample_trajectories", "group": "probe", "params": {}}]}

    assert vocab.unmet(recipe, "time-course") == frozenset({"trajectories"})


def test_a_probe_after_a_fit_is_legal(mioflow_declared):
    """The other half, and the one that makes the refusal mean something."""
    recipe = {"name": "t", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        TOOL_MIOFLOW,
        {"name": "sample_trajectories", "group": "probe", "params": {}}]}

    assert vocab.unmet(recipe, "time-course") == frozenset()


def test_a_filter_between_the_fit_and_the_probe_is_refused(mioflow_declared):
    """Trajectories fitted over cells a filter has since dropped do not describe the ones that
    remain — the same argument that clears `embedding`, and it comes for free because
    `trajectories` is step-produced and `frame_facts` therefore excludes it from `keep`."""
    recipe = {"name": "t", "steps": [
        {"name": "phate", "group": "latent", "params": {}},
        TOOL_MIOFLOW,
        {"name": "filter_cells", "group": "prep", "params": {"min_genes": 200}},
        {"name": "sample_trajectories", "group": "probe", "params": {}}]}

    assert vocab.unmet(recipe, "time-course") == frozenset({"trajectories"})
    assert vocab.invalidated(recipe, "time-course") == frozenset({"trajectories"}), (
        "and it is reported as CLEARED, not as never-produced — different fixes")


def test_trajectories_and_model_are_never_frame_facts():
    """The disjointness the invalidation rests on. If `trajectories` were ever in `frame_facts`
    a narrowing would silently stop clearing it, and a probe would answer about cells that are
    gone. `model` is no longer step-produced by any table (no bundled step leaves one), and it
    is still not a frame fact."""
    assert "trajectories" not in vocab.frame_facts()
    assert "trajectories" in vocab.step_facts()
    assert "model" not in vocab.frame_facts()


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
    what `STEP_NEEDS`' embedding note exists to prevent."""
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


def test_sample_trajectories_reads_the_slot_and_writes_to_the_g_vector(tmp_path):
    """What a probe IS: it asks a question of what a model step returned and records the answer.
    It transforms no data, moves no selection, and produces no embedding. `sample_trajectories`
    reads `state["trajectories"]` (`probes.PROBE_READS`), never a model object."""
    state = _framed()
    state["trajectories"] = np.zeros((4, 6, 2), dtype=np.float32)
    state["model"] = object()        # not read: a model without `trajectories` would decline
    g: dict = {}
    ctx = _ctx(tmp_path)

    out = _runner._run_probe_step("sample_trajectories", {}, state, g, ctx)

    assert out is None
    # (T, n, d) = (4, 6, 2): four time bins over six paths. Asserted in this direction because
    # the seam's array is time FIRST, so reading shape[0] as a population size is the natural
    # mistake — and it reads perfectly in a g-vector.
    assert g["sample_trajectories.n_steps"] == 4
    assert g["sample_trajectories.n_trajectories"] == 6
    assert g["sample_trajectories.dims"] == 2
    assert state["emb"] is None, "a probe answers a question; it does not re-embed"


def test_a_probe_with_no_trajectories_declines_with_a_reason(tmp_path):
    """A run-time decline with a reason, for a state where no step returned the slot. A model
    in `state["model"]` does not stand in for it."""
    from manyruns.pipeline.steps import _StepSkipped

    state = _framed()
    state["model"] = _FakeFlow()
    with pytest.raises(_StepSkipped, match="no trajectories"):
        _runner._run_probe_step("sample_trajectories", {}, state, {}, _ctx(tmp_path))


def test_an_unknown_probe_declines_rather_than_inventing_an_answer(tmp_path):
    """A probe resolves by NAME from a table manyruns owns, so `dispatchable` can answer
    "could this engine run this step" without running it."""
    from manyruns.pipeline.steps import _StepSkipped

    state = _framed()
    state["model"] = _FakeFlow()
    with pytest.raises(_StepSkipped, match="no probe named"):
        _runner._run_probe_step("read_its_mind", {}, state, {}, _ctx(tmp_path))


def test_a_probe_is_dispatchable_where_a_tool_step_can_fill_its_slot():
    """A probe reads what a model step returned, and a tool step runs the same way on
    manylatents and on the in-process loop, so both offer the probe. `mock` never launches a
    tool, so nothing fills the slot there; offering a probe would put a step in the menu that
    can only ever decline — which is the honesty `dispatchable` exists for."""
    assert _runner.dispatchable("manylatents", "sample_trajectories", "probe") is True
    assert _runner.dispatchable(vocab.INPROC, "sample_trajectories", "probe") is True
    assert _runner.dispatchable("mock", "sample_trajectories", "probe") is False
