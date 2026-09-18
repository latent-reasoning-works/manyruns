"""The gene axis is read off the file AND arrives at the engine — the second half, which was
missing.

`app._read_inputs` has derived `(counts, genes)` since the cutover, and its own comment says why:
without them the calculus calls `preprocess`/`qc`/`markers` legal at PLAN time and
`prep._genes_or_refuse` refuses them at RUN time, "which is the worst of both". That producer
worked. The CONSUMER did not.

`run_explorations` re-reads the path only when it was handed no array (`array is None and
labels is None`), and `cmd_run` always hands it one — it loads up front to print progress. So
`counts`/`genes` were locals initialised to None on every command-line run, and the axis died
one frame after being read.

MEASURED on `manyruns run data/pbmc3k_raw.h5ad --recipe markers --engine manylatents`, the one
real dataset this repo ships:

    before   rank_genes_top: not measured — no gene axis in this state
    after    rank_genes_n_groups 8 · rank_genes_n_tested 32740

These tests pin the WIRING rather than that output, because the output needs manylatents and
this must fail in CI too.
"""
from __future__ import annotations

import inspect

from manyruns import app, modes


class _Capture:
    """A server that records the payload instead of running it."""

    def __init__(self):
        self.seen = None

    def predict(self, inputs):
        self.seen = inputs
        return {"served_by": "test", "engine": "x", "g_vector": {}, "trace": [], "status": {}}


def test_run_explorations_accepts_a_preloaded_gene_axis():
    """The signature is the fix. As locals these could only ever come from the branch that
    re-reads the file, which the CLI path never enters."""
    params = inspect.signature(app.run_explorations).parameters
    assert "counts" in params and "genes" in params


def test_a_preloaded_axis_is_forwarded_rather_than_dropped(tmp_path):
    """The caller loaded it; the engine must receive it. Passing an `array` is what makes the
    re-read branch skip — the exact condition under which the axis used to vanish."""
    srv = _Capture()
    app.run_explorations(
        data_folder=tmp_path, modality="scrna", server=srv,
        recipe={"name": "markers", "steps": [{"name": "rank_genes", "group": "analysis"}]},
        engine="manylatents", array="ARRAY", labels=None,
        counts="COUNTS", genes=["g1", "g2"],
    )
    assert srv.seen["counts"] == "COUNTS"
    assert srv.seen["genes"] == ["g1", "g2"]


def test_the_mode_router_carries_the_axis_between_them():
    """`modes._infer` is the hop `cmd_run` reaches `run_explorations` through. A key the router
    does not forward is a key the payload never has, however carefully both ends are wired."""
    source = inspect.getsource(modes._infer)
    assert 'counts=req.get("counts")' in source
    assert 'genes=req.get("genes")' in source


def test_a_caller_that_loaded_nothing_still_gets_the_re_read(tmp_path, monkeypatch):
    """The fallback the parameters replace must survive them: the experiment harness passes only
    a path, and its runs would otherwise reach separation/composition with no labels either."""
    called = []

    def _fake(engine, folder, dataset, time_key, layers=None):
        called.append(1)
        return {"array": "ARR", "labels": ["a", "b"], "kind": "condition",
                "counts": "COUNTS", "genes": ["g1"], "layers": {}}

    monkeypatch.setattr(app, "_load_inputs", _fake)
    srv = _Capture()
    app.run_explorations(
        data_folder=tmp_path, modality="scrna", server=srv,
        recipe={"name": "markers", "steps": [{"name": "rank_genes", "group": "analysis"}]},
        engine="manylatents",
    )
    assert called, "a caller with no array must still trigger the re-read"
    assert srv.seen["counts"] == "COUNTS"
