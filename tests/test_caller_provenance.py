"""Wrapping-program provenance is recorded and printed, never computed with."""
from copy import deepcopy
from importlib import metadata
from unittest.mock import Mock

import pytest

from manyruns import app, decisions, store
from manyruns.pipeline import runner


@pytest.fixture(autouse=True)
def clean_caller(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_CALLER", raising=False)
    monkeypatch.delenv("MANYRUNS_CALLER_VERSION", raising=False)


@pytest.mark.parametrize("caller,version", [
    (None, None), ("", "0.1.0"), ("labtool", "0.1.0"),
    ("labtool", None), ("labtool", ""), ("labtool%", "0.1%"),
])
@pytest.mark.parametrize("available", [True, False])
def test_version(caller, version, available, monkeypatch, capsys):
    if available:
        dist = Mock(version="1.2.3")
        dist.read_text.return_value = '{"vcs_info":{"commit_id":"abc123"}}'
        monkeypatch.setattr(metadata, "distribution", lambda _: dist)
        build = "1.2.3 · revision abc123"
    else:
        def missing(_):
            raise metadata.PackageNotFoundError
        monkeypatch.setattr(metadata, "distribution", missing)
        build = "version unavailable · revision unavailable"
    monkeypatch.setattr("sys.argv", ["wrapper-script"])
    # Set after app import: wrappers configure provenance before calling main.
    if caller is not None:
        monkeypatch.setenv("MANYRUNS_CALLER", caller)
    if version is not None:
        monkeypatch.setenv("MANYRUNS_CALLER_VERSION", version)
    with pytest.raises(SystemExit) as exc:
        app.main(["--version"])
    assert exc.value.code == 0
    prefix = f"{caller}{' ' + version if version else ''} · built on manyruns" if caller else "wrapper-script"
    assert capsys.readouterr().out == f"{prefix} {build}\n"


@pytest.mark.parametrize("caller,version", [(None, None), ("", ""), ("labtool", None), ("labtool", ""), ("labtool", "0.1.0"), ("x" * 100, "v" * 100)])
def test_mock_run_record(caller, version, monkeypatch):
    if caller is not None:
        monkeypatch.setenv("MANYRUNS_CALLER", caller)
    if version is not None:
        monkeypatch.setenv("MANYRUNS_CALLER_VERSION", version)
    assert app.main(["run", "swissroll", "--engine", "mock", "--project", "sample"]) == 0
    row, = store.read()
    if caller:
        assert row["caller"] == {"name": caller, "version": version or None}
    else:
        assert "caller" not in row


@pytest.mark.parametrize("caller", [None, "labtool"])
def test_extra_cannot_override_caller(caller, monkeypatch):
    if caller:
        monkeypatch.setenv("MANYRUNS_CALLER", caller)
    with pytest.raises(ValueError, match="extra cannot override caller"):
        store.append({}, extra={"caller": {"name": "other"}})
    assert list(store.read()) == []


@pytest.mark.parametrize("caller,version", [("x" * 101, None), ("lab\ntool", None), ("lab\ttool", None), ("lab\x7ftool", None), ("lab\x85tool", None), ("labtool", "v" * 101), (None, "v" * 101)])
def test_invalid_caller_refused_before_run(caller, version, monkeypatch, capsys, tmp_path):
    if caller is not None:
        monkeypatch.setenv("MANYRUNS_CALLER", caller)
    if version is not None:
        monkeypatch.setenv("MANYRUNS_CALLER_VERSION", version)
    run = Mock(side_effect=AssertionError("run must not start"))
    monkeypatch.setattr(app, "cmd_run", run)
    with pytest.raises(SystemExit) as exc:
        app.main(["run", "swissroll", "--engine", "mock"])
    assert exc.value.code == 2
    assert "MANYRUNS_CALLER" in capsys.readouterr().err
    run.assert_not_called()
    assert not (tmp_path / "outputs").exists()


def test_actions_identical(monkeypatch, capsys):
    assert app.main(["actions", "swissroll"]) == 0
    before = capsys.readouterr().out
    monkeypatch.setenv("MANYRUNS_CALLER", "labtool")
    monkeypatch.setenv("MANYRUNS_CALLER_VERSION", "0.1.0")
    assert app.main(["actions", "swissroll"]) == 0
    assert capsys.readouterr().out == before


def test_mock_identity_inputs_identical(monkeypatch):
    inputs = []
    original = runner.identity

    def capture(*args, **kwargs):
        inputs.append(deepcopy((args, kwargs)))
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "identity", capture)
    argv = ["run", "swissroll", "--engine", "mock", "--project", "sample"]
    assert app.main(argv) == 0
    before = deepcopy(inputs)
    inputs.clear()
    monkeypatch.setenv("MANYRUNS_CALLER", "labtool")
    monkeypatch.setenv("MANYRUNS_CALLER_VERSION", "0.1.0")
    assert app.main(argv) == 0
    assert inputs == before and inputs
    rows = list(store.read())
    assert rows[0]["spec_id"] == rows[1]["spec_id"]
    # Elapsed measurement time can differ between otherwise identical executions.
    values = [{key: value for key, value in row["g_vector"].items()
               if key != "suite_seconds"} for row in rows]
    assert values[0] == values[1]


def test_decisions_have_no_caller(monkeypatch):
    monkeypatch.setenv("MANYRUNS_CALLER", "labtool")
    assert decisions.append(offered=[{"name": "embed"}], chosen="embed")
    row, = decisions.read()
    assert "caller" not in row
