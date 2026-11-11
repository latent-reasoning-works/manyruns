from pathlib import Path

import pytest


@pytest.mark.source_checkout
def test_the_base_job_installs_every_direct_import_of_the_tool_seam_tests():
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text()
    base = text.split("name: base install (no compute stack)")[1].split("\n  # ──")[0]
    assert '--with "manykinds>=0.1.2"' in base
