"""The one-way boundary covers source, strings, configs, dependencies and prose.

The tracked-tree gate lives in test_public_safety; these regressions exercise its name
matcher with a synthetic token so neither fixtures nor failure messages reveal a name.
"""
import hashlib

import pytest

from public_safety import forbidden_count


@pytest.mark.parametrize('template', [
    'import {}', 'from {}.api import run', 'importlib.import_module("{}")',
    'engine: {}', '"{}>=1.0"', 'name = "{}"', '# A note about {}.',
    'A paragraph names {}.', 'configs/{}/default.yaml', 'prefix_{}_suffix',
    '{}Backend', '{}-plugin',
])
def test_reintroduction_is_rejected_in_code_config_dependencies_and_prose(template):
    synthetic = 'Sentinel'
    hashes = {hashlib.sha256(synthetic.lower().encode()).hexdigest()}
    assert forbidden_count(template.format(synthetic), hashes) > 0
    assert forbidden_count(template.format(synthetic.upper()), hashes) > 0


def test_digest_list_contains_only_sha256_hashes():
    import re
    from learners import LEARNER_HASHES

    assert LEARNER_HASHES
    assert all(re.fullmatch('[0-9a-f]{64}', value) for value in LEARNER_HASHES)


def test_unrelated_prose_is_allowed():
    assert forbidden_count('The learner observes the environment.') == 0


def test_the_train_stage_survives_the_removal():
    from manyruns import modes

    plan = modes.run('train', {'store': 'outputs/x'})
    assert 'train' in modes.MODES
    assert plan['store'] == 'outputs/x'
    assert 'planned' in plan['status']


THEME_COMMENT = "#: Sourced from the brand deck's `:root` colour tokens."


def test_theme_colour_comment_names_no_file():
    from pathlib import Path

    import manyruns

    theme = Path(manyruns.__file__).resolve().parent / 'tui' / 'theme.py'
    assert theme.is_file(), 'theme module moved'
    # Compare first so a failure never echoes the line under test.
    matches = theme.read_text(encoding='utf-8').splitlines()[30] == THEME_COMMENT
    assert matches, 'theme.py line 31 is not the spec comment'
