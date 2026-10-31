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


EXPECTED_HASHES = frozenset({
    "98b1c9e8888c2fe13853a8ef691b7b2a9cc3257b6923f79eaec10cb762fdb074",
    "9dfe5c985dbba48aeef7b1af1cf2f5f3564b304d1aac0b91159efe060adcc808",
    "b32cbd7a08dbe442fe0c07ce5d82f609b2e5f94dcf8acdd1917e8f9038b10824",
})


def test_learner_hashes_hold_three_digests():
    from learners import LEARNER_HASHES

    assert LEARNER_HASHES == EXPECTED_HASHES


def test_no_stray_digest_is_stored():
    import re
    from pathlib import Path

    from learners import LEARNER_HASHES

    tests_dir = Path(__file__).resolve().parent
    assert tests_dir.is_dir(), 'tests folder moved'
    text = (tests_dir / 'learners.py').read_text(encoding='utf-8')
    assert set(re.findall(r"[0-9a-f]{64}", text)) == LEARNER_HASHES
