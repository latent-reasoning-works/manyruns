import numpy as np
import pytest
from manyruns.pipeline import shape_profile


def test_real_helper_matches_stub_contract():
    module = pytest.importorskip('manylatents.metrics.shape_profile')
    x = np.random.default_rng(12).normal(size=(600,5))
    shape_profile._check_rows(module.shape_profile(x), selection=False)
    shape_profile._check_rows(module.shape_profile(x, selection=np.arange(600)<30), selection=True)
