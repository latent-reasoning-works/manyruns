"""Opt-in start-to-tables timing with installed real helpers."""
import json
import platform
import subprocess
import sys
import time

import pytest


@pytest.mark.profile_timing
def test_5000_by_50_start_to_two_tables_under_budget(tmp_path):
    pytest.importorskip('manylatents.metrics.shape_profile')
    pytest.importorskip('manylatents.singlecell.dataset_features')
    code = """
import sys
import numpy as np
from manyruns import catalog
from manyruns.pipeline.runner import run_inproc
from pathlib import Path
x = np.random.default_rng(5).normal(size=(5000,50))
out = run_inproc(x, catalog.load_recipe('profile'), Path(sys.argv[1]))
assert out['complete'], out['steps']
assert Path(out['steps'][0]['artifacts']['dataset_features']).is_file()
assert Path(out['steps'][2]['artifacts']['shape_profile']).is_file()
"""
    start = time.perf_counter()
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    print(json.dumps({'seconds':elapsed, 'platform':platform.platform(), 'machine':platform.machine(), 'processor':platform.processor(), 'python':sys.version}))
    assert result.returncode == 0, result.stderr + result.stdout
    assert elapsed < 15
