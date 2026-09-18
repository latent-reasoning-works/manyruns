"""Release archives explicitly include only the application and its testable source."""
from pathlib import Path
from email.parser import BytesParser
import tarfile
import tomllib
import zipfile

from packaging.requirements import Requirement
import pytest

ROOT = Path(__file__).resolve().parents[1]
SDIST_ROOTS = {'manyruns', 'tests', 'README.md', 'LICENSE', 'CHANGELOG.md', 'pyproject.toml'}
# The sdist must contain the application, tests, README, license, pyproject and generated
# PKG-INFO, plus CHANGELOG.md when present in the source tree. Hatch may force-include
# .gitignore and .hgignore; allow only those optional roots. Checkout tooling, local data,
# credentials, caches and any other unlisted root must never enter the archive.
SDIST_VCS_IGNORE_FILES = {'.gitignore', '.hgignore'}


def test_build_targets_use_explicit_include_lists():
    assert ROOT.is_dir()
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    targets = config['tool']['hatch']['build']['targets']
    assert {p.strip('/') for p in targets['sdist']['include']} == SDIST_ROOTS
    assert targets['wheel']['include'] == ['/manyruns/']
    assert targets['wheel']['packages'] == ['manyruns']
    assert 'sources' not in config['tool']['uv']
    assert 'extra-build-dependencies' not in config['tool']['uv']
    assert 'manyagents>=0.0.1' in config['project']['optional-dependencies']['agents']


def assert_agents_metadata(archive):
    metadata_path = next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
    metadata = BytesParser().parsebytes(archive.read(metadata_path))
    requirements = [Requirement(value) for value in metadata.get_all('Requires-Dist', [])]
    agents = [requirement for requirement in requirements if requirement.name == 'manyagents']
    assert len(agents) == 1
    assert str(agents[0].specifier) == '>=0.0.1'
    assert str(agents[0].marker) == 'extra == "agents"'
    assert all(requirement.url is None for requirement in requirements)


def test_built_archives_and_wheel_rebuilt_from_sdist(tmp_path, monkeypatch):
    build = pytest.importorskip('hatchling.build', reason='requires the build backend')
    assert ROOT.is_dir()
    monkeypatch.chdir(ROOT)
    dist = tmp_path / 'dist'
    dist.mkdir()
    sdist = dist / build.build_sdist(str(dist))
    wheel = dist / build.build_wheel(str(dist))
    export = tmp_path / 'export'
    export.mkdir()
    with tarfile.open(sdist) as archive:
        members = archive.getnames()
        assert not any("/harness/manifests/" in name for name in members)
        assert any(name.endswith("/tests/fixtures/acquire_manifest.tsv") for name in members)
        roots = {Path(name).parts[1] for name in members if len(Path(name).parts) > 1}
        expected = SDIST_ROOTS | {'PKG-INFO'}
        if not (ROOT / 'CHANGELOG.md').exists():
            expected -= {'CHANGELOG.md'}
        assert roots - SDIST_VCS_IGNORE_FILES == expected
        archive.extractall(export, filter='data')
    with zipfile.ZipFile(wheel) as archive:
        assert_agents_metadata(archive)
        names = set(archive.namelist())
        assert not any("/harness/manifests/" in name for name in names)
        assert all(name.startswith('manyruns/') or '.dist-info/' in name for name in names)
        assert 'manyruns/configs/recipe/embed.yaml' in names
        assert any(name.endswith('.tcss') for name in names)
    unpacked = export / members[0].split('/')[0]
    assert unpacked.is_dir()
    monkeypatch.chdir(unpacked)
    rebuilt_dir = tmp_path / 'rebuilt'
    rebuilt_dir.mkdir()
    rebuilt = rebuilt_dir / build.build_wheel(str(rebuilt_dir))
    with zipfile.ZipFile(rebuilt) as archive:
        assert_agents_metadata(archive)
        assert set(archive.namelist()) == names
