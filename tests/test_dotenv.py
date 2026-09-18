"""Startup reads only the working directory's environment file."""
import pytest

from manyruns import app

pytest.importorskip('dotenv')


def test_startup_does_not_load_an_ancestor_environment_file(tmp_path, monkeypatch):
    (tmp_path / '.env').write_text('MANYRUNS_DOTENV_TEST=ancestor\n')
    child = tmp_path / 'child'
    child.mkdir()
    monkeypatch.chdir(child)
    monkeypatch.delenv('MANYRUNS_DOTENV_TEST', raising=False)

    app._load_dotenv()

    from manyruns import env
    assert env.get('DOTENV_TEST') is None


@pytest.mark.parametrize('existing, expected', [(None, 'local'), ('exported', 'exported')])
def test_startup_loads_local_file_without_overriding_exports(tmp_path, monkeypatch,
                                                          existing, expected):
    (tmp_path / '.env').write_text('MANYRUNS_DOTENV_TEST=local\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('MANYRUNS_DOTENV_TEST', raising=False)
    if existing is not None:
        monkeypatch.setenv('MANYRUNS_DOTENV_TEST', existing)

    app._load_dotenv()

    from manyruns import env
    assert env.get('DOTENV_TEST') == expected
