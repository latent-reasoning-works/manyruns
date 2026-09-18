"""An installed omics runtime must ship the preprocessing module the product delegates to."""
from importlib import import_module, metadata

import pytest


def test_installed_singlecell_runtime_ships_preprocessing():
    for distribution in ('manylatents', 'manylatents-omics'):
        try:
            metadata.version(distribution)
        except metadata.PackageNotFoundError:
            pytest.skip('single-cell runtime is absent in the base test environment')
    # Do not importorskip this module: a present but incomplete wheel is the release blocker.
    module = import_module('manylatents.singlecell.preprocessing')
    assert callable(module.looks_like_counts)
    assert callable(module.normalize)
