"""Labels reach a run even when the caller passes only a data PATH.

The experiment harness calls `modes.run("infer", {data: <path>})` directly, without the
CLI's up-front load. Before this wiring, its runs reached separation/composition with no
condition labels — so a `contrast` recipe on case/control data silently produced `None`,
which the results table then read as a missing metric. The precondition calculus flags the
cell as legal; this makes it actually runnable.
"""
from __future__ import annotations

from manyruns import app


class _CapturingServer:
    """A serving backend that records the inputs it was handed."""

    def __init__(self):
        self.seen = None

    def predict(self, inputs):
        self.seen = inputs
        return {"served_by": "test", "engine": "x", "g_vector": {}, "trace": [], "status": {}}


def test_run_explorations_loads_labels_from_a_path_when_none_were_passed(monkeypatch, tmp_path):
    # stub the loader: a path + manylatents → (array, condition labels), as load_labeled does.
    # A DICT since the loader stopped being a positional tuple. A fake that returns the wrong
    # KEYS fails loudly; a fake that returned the wrong ARITY used to keep passing while the
    # real seam had moved, which is why the shape changed (see `app._EMPTY_INPUTS`).
    monkeypatch.setattr(app, "_load_inputs",
                        lambda *a, **k: {"array": "ARRAY", "kind": "condition",
                                         "labels": ["treated", "healthy", "treated"],
                                         "counts": None, "genes": None, "layers": {}})
    srv = _CapturingServer()

    app.run_explorations(
        data_folder=tmp_path, modality="scrna", server=srv,
        recipe={"name": "contrast", "steps": [{"name": "separation", "group": "analysis"}]},
        engine="manylatents",
    )
    assert srv.seen["labels"] == ["treated", "healthy", "treated"]   # threaded into the run
    assert srv.seen["array"] == "ARRAY"


def test_a_pre_loaded_caller_is_not_double_loaded(monkeypatch, tmp_path):
    """The CLI loads up front and passes labels in — the auto-load must NOT fire and clobber."""
    called = []
    monkeypatch.setattr(app, "_load_inputs", lambda *a, **k: called.append(1) or dict(app._EMPTY_INPUTS))
    srv = _CapturingServer()

    app.run_explorations(
        data_folder=tmp_path, modality="scrna", server=srv,
        recipe={"name": "contrast", "steps": [{"name": "separation", "group": "analysis"}]},
        engine="manylatents", array="PRELOADED", labels=["a", "b"],
    )
    assert not called, "auto-load fired even though the caller pre-loaded"
    assert srv.seen["labels"] == ["a", "b"]


def test_mock_and_named_dataset_load_nothing_here(tmp_path):
    # a named engine dataset is loaded by the engine, not by us
    assert app._load_inputs("manylatents", tmp_path, "swissroll", None) == app._EMPTY_INPUTS
    assert app._load_inputs("mock", tmp_path, None, None) == app._EMPTY_INPUTS
    # A third line named the learner's engine here — it loaded its own data downstream. It
    # is not an engine now (§3.5), and an unknown name must still load nothing rather than
    # reaching for the disk on a caller's behalf.
    assert app._load_inputs("not_an_engine", tmp_path, None, None) == app._EMPTY_INPUTS


# ── the full chain, with real bytes (gated on the stack) ─────────────────────
def test_a_case_control_h5ad_yields_condition_labels_end_to_end(tmp_path):
    """path → _load_inputs → load_labeled → the `disease` column, no time axis present.

    This is what makes `contrast` runnable in a sweep: the calculus says the cell is legal
    (case-control provides `conditions`), and this proves the labels actually arrive."""
    import pytest

    ad = pytest.importorskip("anndata")
    pytest.importorskip("scanpy")
    import numpy as np

    a = ad.AnnData(np.random.default_rng(0).random((6, 4), dtype="float32"))
    a.obs["disease"] = ["treated", "treated", "treated", "healthy", "healthy", "healthy"]
    path = tmp_path / "cohort.h5ad"
    a.write_h5ad(path)

    loaded = app._load_inputs("manylatents", path, None, None)
    array, labels, kind = loaded["array"], loaded["labels"], loaded["kind"]
    assert array is not None and array.shape[0] == 6
    assert list(labels) == ["treated", "treated", "treated", "healthy", "healthy", "healthy"]
    assert kind == "condition"   # the disease column is a condition axis, NOT time
