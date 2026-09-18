"""Path ingestion works without the optional engine, including pandas 3's string dtype."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from manyruns import app
from manyruns.pipeline import loading
from manyruns.tui import state


@pytest.mark.parametrize("suffix,sep", [(".csv", ","), (".tsv", "\t"), (".txt", "\t")])
@pytest.mark.parametrize("header", [True, False])
def test_numeric_tables_keep_every_row(tmp_path, suffix, sep, header):
    matrix = np.arange(3000).reshape(150, 20)
    path = tmp_path / ("tiny" + suffix)
    pd.DataFrame(matrix).to_csv(path, sep=sep, header=[f"f{i}" for i in range(20)] if header else False,
                                index=False)
    np.testing.assert_array_equal(loading.load_labeled(path)[0], matrix)
    entry = state.local_entry(path)
    assert entry is not None and not entry.refusal


@pytest.mark.parametrize("text,expected", [
    ("label,x,y\na,1,2\nb,3,4\n", [[1, 2], [3, 4]]),
    (",x,y\n0,1,2\n1,3,4\n", [[0, 1, 2], [1, 3, 4]]),
    ("index,x,y\n10,1,2\n11,3,4\n", [[10, 1, 2], [11, 3, 4]]),
    ("1,2\n", [[1, 2]]),
])
def test_table_column_policy(tmp_path, text, expected):
    path = tmp_path / "table.csv"
    path.write_text(text)
    np.testing.assert_array_equal(loading.load_labeled(path)[0], expected)
    assert state.local_entry(path) is not None


@pytest.mark.parametrize("dtype,values,expected", [
    ("Int64", [1, 2], [1., 2.]),
    ("Float64", [1.5, 2.5], [1.5, 2.5]),
    ("boolean", [True, False], [1., 0.]),
    ("object", [1, 2], [1., 2.]),
    ("object", [1.5, 2.5], [1.5, 2.5]),
    ("object", [True, False], [1., 0.]),
])
@pytest.mark.parametrize("labeled", [False, True])
def test_nullable_and_object_numeric_columns_become_floats_and_text_stays_metadata(
        dtype, values, expected, labeled):
    frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype),
                          "y": pd.Series(values[::-1], dtype=dtype),
                          "label": ["a", "b"]})
    obj = (frame, np.array([0, 1])) if labeled else frame
    matrix = loading.as_matrix(obj)
    assert matrix.dtype == np.dtype(float)
    np.testing.assert_array_equal(matrix, np.column_stack([expected, expected[::-1]]))


@pytest.mark.parametrize("dtype,value", [
    ("Int64", 1), ("Float64", 1.5), ("boolean", True),
    ("object", 1), ("object", 1.5), ("object", True),
])
def test_nullable_and_object_numeric_missing_values_are_refused_not_dropped(dtype, value):
    frame = pd.DataFrame({"x": pd.Series([value, pd.NA], dtype=dtype), "y": [3, 4]})
    with pytest.raises(ValueError, match="numeric table columns contain missing or non-finite"):
        loading.as_matrix((frame, np.array([0, 1])))


@pytest.mark.parametrize("dtype", ["Float64", "object"])
@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nullable_and_object_numeric_infinities_are_refused_not_dropped(dtype, value):
    frame = pd.DataFrame({"x": pd.Series([1., value], dtype=dtype), "y": [3, 4]})
    with pytest.raises(ValueError, match="numeric table columns contain missing or non-finite"):
        loading.as_matrix(frame)


@pytest.mark.parametrize("values", [["1", "2"], [1, "2"]])
def test_numeric_strings_and_mixed_text_numeric_columns_stay_metadata(values):
    frame = pd.DataFrame({"label": pd.Series(values, dtype=object), "x": [3, 4]})
    np.testing.assert_array_equal(loading.as_matrix(frame), [[3], [4]])


@pytest.mark.parametrize("value", [1 + 2j, 10 ** 400])
def test_object_numeric_values_that_cannot_become_floats_are_refused(value):
    frame = pd.DataFrame({"x": pd.Series([value, value], dtype=object), "y": [3, 4]})
    with pytest.raises(ValueError, match="numeric table columns cannot be converted to floats"):
        loading.as_matrix(frame)


@pytest.mark.parametrize("suffix,sep", [(".csv", ","), (".tsv", "\t"), (".txt", "\t")])
@pytest.mark.parametrize("prefix", ["\n", " \n", "\n  \n"])
def test_headerless_tables_skip_leading_blank_lines_and_keep_every_numeric_row(
        tmp_path, suffix, sep, prefix):
    path = tmp_path / ("blank-lines" + suffix)
    path.write_text(prefix + f"1{sep}2\n3{sep}4\n", encoding="utf-8")
    np.testing.assert_array_equal(loading.load_labeled(path)[0], [[1, 2], [3, 4]])
    entry = state.local_entry(path)
    assert entry is not None and not entry.refusal


def test_headerless_csv_skips_tab_only_lines_as_whitespace(tmp_path):
    path = tmp_path / "blank-tabs.csv"
    path.write_text(" \t \n1,2\n3,4\n", encoding="utf-8")
    np.testing.assert_array_equal(loading.load_labeled(path)[0], [[1, 2], [3, 4]])


@pytest.mark.parametrize("suffix", [".tsv", ".txt"])
def test_tab_only_line_is_an_empty_header_in_tab_delimited_tables(tmp_path, suffix):
    path = tmp_path / ("empty-fields" + suffix)
    path.write_text("\t\n1\t2\n3\t4\n", encoding="utf-8")
    np.testing.assert_array_equal(loading.load_labeled(path)[0], [[1, 2], [3, 4]])


def test_quoted_whitespace_is_a_header_not_a_skipped_blank_line(tmp_path):
    path = tmp_path / "quoted-header.csv"
    path.write_text('" "\n1\n2\n', encoding="utf-8")
    np.testing.assert_array_equal(loading.load_labeled(path)[0], [[1], [2]])


@pytest.mark.parametrize("suffix,sep", [(".csv", ","), (".tsv", "\t"), (".txt", "\t")])
def test_bom_headerless_table_is_accepted_by_loader_and_roster(tmp_path, suffix, sep):
    path = tmp_path / ("bom" + suffix)
    path.write_text(f"\ufeff1{sep}2\n", encoding="utf-8")
    np.testing.assert_array_equal(loading.load_labeled(path)[0], [[1, 2]])
    entry = state.local_entry(path)
    assert entry is not None and not entry.pending and not entry.refusal


@pytest.mark.parametrize("make", [
    lambda: pd.DataFrame({"x": [1, 2]}, index=[10, 11]),
    lambda: pd.Series([1, 2], index=[10, 11]),
    lambda: pd.Index([1, 2]),
    lambda: pd.array([1, 2]),
])
def test_pandas_array_objects_are_not_torch_datasets(make):
    np.testing.assert_array_equal(loading.as_matrix(make()), [[1], [2]])


@pytest.mark.parametrize("command", ["run", "init", "explore"])
def test_cli_paths_deliver_numeric_table_without_an_engine(tmp_path, monkeypatch, command):
    from manyruns.serving import LocalServer

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "tiny.csv"
    path.write_text("x,y\n1,2\n3,4\n")
    arrays = []

    class MockCompute:
        def predict(self, payload):
            arrays.append(payload["array"])
            return LocalServer(engine="mock").predict(payload)

    monkeypatch.setattr(app, "_server_for", lambda *a, **k: MockCompute())
    def inspect_session(session, *args, **kwargs):
        arrays.append(session.state["X"])
        return 0
    monkeypatch.setattr(app, "interactive_session", inspect_session)
    if command == "explore":
        app.write_project("table", path, "unknown", app.load_recipe("embed"), engine="manylatents")
        argv = ["explore", "--project", "table"]
    else:
        argv = [command, str(path), "--project", "table", "--recipe", "embed"]
    assert app.main([*argv, "--engine", "manylatents"]) == 0
    assert len(arrays) == 1
    np.testing.assert_array_equal(arrays[0], [[1, 2], [3, 4]])


@pytest.mark.parametrize("content,suffix", [
    ("", ".csv"), ("x,y\n", ".csv"), ("name,label\na,b\n", ".csv"),
    ("x,y\n1,2,3\n", ".csv"), ('x,y\n"1,2\n', ".csv"),
    ("x,y\n1,2\n3,\n", ".csv"), ("x,y\n1,inf\n", ".csv"),
    ("x y\n1 2\n", ".txt"),
])
@pytest.mark.parametrize("command", ["run", "init", "explore"])
def test_unusable_table_is_a_cli_refusal(tmp_path, monkeypatch, capsys, content, suffix, command):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MANYRUNS_TRACEBACK", raising=False)
    path = tmp_path / ("bad" + suffix)
    path.write_text(content)
    if command == "explore":
        app.write_project("table", path, "unknown", app.load_recipe("embed"), engine="manylatents")
        argv = ["explore", "--project", "table"]
    else:
        argv = [command, str(path), "--project", "table", "--recipe", "embed"]
    assert app.main([*argv, "--engine", "manylatents"]) == 1
    error = capsys.readouterr().err
    assert "numeric matrix" in error and path.name in error
    assert "Traceback" not in error
    with pytest.raises(ValueError, match="Could not read"):
        state.local_entry(path)


def test_csv_reaches_manylatents(tmp_path, monkeypatch):
    pytest.importorskip("manylatents")
    monkeypatch.chdir(tmp_path)
    path = Path("tiny.csv")
    pd.DataFrame(np.random.default_rng(7).normal(size=(150, 20)),
                 columns=[f"f{i}" for i in range(20)]).to_csv(path, index=False)
    assert app.main(["run", str(path), "--project", "table", "--recipe", "embed",
                     "--engine", "manylatents"]) == 0
