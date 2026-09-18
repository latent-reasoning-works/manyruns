"""The act-1 fixture — `docs/sidecars/make_tree_fixture.py`.

Why a FILE and not `--dataset dla_tree`: a named manylatents dataset is loaded inside the
engine and this process never sees the array (`runner.py:1087`), so `_read_inputs` returns
`_EMPTY_INPUTS` and `labels_of` is never called. The ground truth would be generated, named,
coloured, and never read. As a file it goes down the same path pbmc3k does.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.source_checkout

pytest.importorskip("anndata")
pytest.importorskip("manylatents")


def test_the_fixture_carries_eight_branches_the_loader_reads_as_a_group_axis(tmp_path):
    import anndata

    from manyruns.pipeline import loading

    out = tmp_path / "tree8.h5ad"
    subprocess.run([sys.executable, "docs/sidecars/make_tree_fixture.py", str(out)],
                   check=True, capture_output=True)
    a = anndata.read_h5ad(out)
    assert a.shape == (800, 60)
    assert len(set(a.obs["branch"])) == 8

    labels, kind = loading.labels_of(a)
    assert kind == "group", "the fixture's own column must be the one `labels_of` finds"
    assert len(set(labels)) == 8
