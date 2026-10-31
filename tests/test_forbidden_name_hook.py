"""The local name hook blocks a forbidden name before a commit exists, and never prints it.

Every test runs in a throwaway repository under tmp_path whose copied digest list holds only
the digest of a synthetic token, so no real name appears in fixtures or failure messages.
The hook is never installed into this checkout or any other real repository.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / '.githooks'
TOKEN = 'Sentinel'
TOKEN_DIGEST = hashlib.sha256(TOKEN.lower().encode()).hexdigest()


def _env():
    # Drop GIT_* so a suite run from inside a git hook cannot redirect git at this checkout.
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env['PYTHONPATH'] = str(ROOT)
    return env


def _git(repo, *args, check=True):
    return subprocess.run(['git', *args], cwd=repo, env=_env(), capture_output=True,
                          text=True, check=check)


def _hook(repo, *args):
    return subprocess.run(['python3', '.githooks/forbidden_names.py', *args], cwd=repo,
                          env=_env(), capture_output=True, text=True)


def _stage(repo, name, text):
    (repo / name).write_text(text, encoding='utf-8')
    _git(repo, 'add', '--', name)


@pytest.fixture
def hook_repo(tmp_path):
    assert HOOKS.is_dir(), 'hook folder moved'
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q')
    shutil.copytree(HOOKS, repo / '.githooks')
    (repo / 'tests').mkdir()
    shutil.copy2(ROOT / 'tests' / 'public_safety.py', repo / 'tests' / 'public_safety.py')
    learners = (ROOT / 'tests' / 'learners.py').read_text(encoding='utf-8')
    learners, replaced = re.subn(r'LEARNER_HASHES = frozenset\(\{.*?\}\)',
                                 f'LEARNER_HASHES = frozenset({{"{TOKEN_DIGEST}"}})',
                                 learners, flags=re.DOTALL)
    assert replaced == 1, 'learners.py no longer defines LEARNER_HASHES as a frozenset literal'
    (repo / 'tests' / 'learners.py').write_text(learners, encoding='utf-8')
    _git(repo, 'config', 'user.name', 'Hook Test')
    _git(repo, 'config', 'user.email', 'hook-test@example.invalid')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    return repo


def test_staged_name_blocks_and_stays_hidden(hook_repo):
    _stage(hook_repo, 'notes.md', f'A title.\nA paragraph about {TOKEN.upper()}.\n')
    result = _hook(hook_repo, 'pre-commit')
    assert result.returncode == 1
    assert 'notes.md:2' in result.stderr
    assert TOKEN.lower() not in result.stderr.lower()
    assert TOKEN.lower() not in result.stdout.lower()


def test_name_in_path_reported_by_position(hook_repo):
    _stage(hook_repo, f'{TOKEN}.md', 'clean text\n')
    result = _hook(hook_repo, 'pre-commit')
    assert result.returncode == 1
    assert 'staged path 1 of 1' in result.stderr
    assert '.md' not in result.stderr
    assert TOKEN.lower() not in result.stderr.lower()


def test_blocked_commit_does_not_exist(hook_repo):
    _stage(hook_repo, 'notes.md', f'A title.\nA paragraph about {TOKEN.upper()}.\n')
    commit = _git(hook_repo, '-c', 'core.hooksPath=.githooks', 'commit', '-q', '-m', 'add',
                  check=False)
    assert commit.returncode != 0
    assert TOKEN.lower() not in commit.stderr.lower()
    assert _git(hook_repo, 'rev-parse', '--verify', '-q', 'HEAD', check=False).returncode != 0


def test_commit_message_is_checked(hook_repo):
    message = hook_repo / 'message.txt'
    message.write_text(f'Subject line\n\nMention {TOKEN} here.\n', encoding='utf-8')
    result = _hook(hook_repo, 'commit-msg', str(message))
    assert result.returncode == 1
    assert 'commit message line 3' in result.stderr
    assert TOKEN.lower() not in result.stderr.lower()


def test_clean_commit_passes(hook_repo):
    _stage(hook_repo, 'notes.md', 'clean text\n')
    result = _hook(hook_repo, 'pre-commit')
    assert result.returncode == 0
    assert result.stderr == ''


def test_hook_files_hold_no_digest():
    assert HOOKS.is_dir(), 'hook folder moved'
    files = [path for path in HOOKS.rglob('*') if path.is_file()]
    assert files, 'the hook folder holds no files'
    holding = [path.name for path in files
               if re.search(r'[0-9a-f]{64}', path.read_text(encoding='utf-8'))]
    assert not holding, f'{len(holding)} hook files hold a 64-hex string'
    staged = subprocess.check_output(
        ['git', 'ls-files', '-s', '.githooks/pre-commit', '.githooks/commit-msg'],
        cwd=ROOT, env=_env(), text=True).splitlines()
    assert len(staged) == 2, 'both wrappers must be tracked'
    assert all(line.split()[0] == '100755' for line in staged)
