"""Display metadata is independent of analysis labels and has one availability policy."""
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from manyruns.pipeline import loading


@pytest.fixture
def annotated():
    n = 30
    obs = pd.DataFrame({
        "day": np.arange(n) % 3,
        "disease": ["healthy", "ill"] * 15,
        "category": pd.Categorical(["a", "b", None] * 10),
        "score": pd.array([*range(27), None, np.inf, -np.inf], dtype="Float64"),
        "flag": pd.array([True, False, None] * 10, dtype="boolean"),
        "nullable": pd.array([1, 2, None] * 10, dtype="Int64"),
        "empty": pd.array([None] * n, dtype="Float64"),
        "barcode": [f"cell-{i}" for i in range(n)],
        "many_groups": [f"group-{i % 21}" for i in range(n)],
    }, index=[f"cell-α-{i}" for i in range(n)])
    return AnnData(np.zeros((n, 3)), obs=obs)


def test_label_key_uses_analysis_precedence_without_changing_labels(annotated):
    before, kind = loading.labels_of(annotated)
    assert loading.label_key_of(annotated) == ("day", "time")
    assert loading.label_key_of(annotated, "score") == ("score", "time")
    loading.color_channel_of(annotated, "disease")
    after, after_kind = loading.labels_of(annotated)
    np.testing.assert_array_equal(before, after)
    assert after_kind == kind == "time"
    del annotated.obs["day"]
    assert loading.label_key_of(annotated) == ("disease", "condition")
    assert loading.labels_of(object()) == (None, None)  # preservation
    assert loading.label_key_of(object()) == (None, None)


def test_channels_preserve_row_order_and_nullable_missing_values(annotated):
    channel = loading.color_channel_of(annotated, "score")
    assert channel["kind"] == "continuous"
    np.testing.assert_equal(channel["values"], np.array([*range(27), np.nan, np.inf, -np.inf]))
    assert channel["n_missing"] == 1 and channel["n_nonfinite"] == 3
    for key in ("category", "flag", "nullable"):
        channel = loading.color_channel_of(annotated, key)
        assert channel["kind"] == "categorical"
        assert pd.isna(channel["values"][2])
        assert loading.color_values_of(channel["values"], channel["kind"])[2] == "(missing)"
        assert channel["n_missing"] == 10
    np.testing.assert_array_equal(loading.color_channel_of(annotated, "disease")["values"],
                                  annotated.obs["disease"].to_numpy())


@pytest.mark.parametrize(("dtype", "values", "expected"), [
    ("float64", [1., np.nan, np.inf, -np.inf], ["1.0", "(missing)", "(missing)", "(missing)"]),
    ("Float64", [1., None, np.inf, -np.inf], ["1.0", "(missing)", "(missing)", "(missing)"]),
    ("Int64", [1, None, 2], ["1", "(missing)", "2"]),
    ("boolean", [True, None, False], ["True", "(missing)", "False"]),
])
def test_categorical_normalization_accepts_readonly_pandas_arrays(monkeypatch, dtype, values, expected):
    """Pandas can expose immutable arrays, including the freshly computed missing mask."""
    series = pd.Series(values, dtype=dtype)
    before = series.copy(deep=True)
    to_numpy = pd.Series.to_numpy

    def readonly(self, *args, **kwargs):
        result = to_numpy(self, *args, **kwargs)
        result.flags.writeable = False
        return result

    monkeypatch.setattr(pd.Series, "to_numpy", readonly)
    np.testing.assert_array_equal(loading.color_values_of(series, "categorical"), expected)
    pd.testing.assert_series_equal(series, before)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "Int64", "UInt64"])
@pytest.mark.parametrize("count", [2, 21])
def test_large_integer_distinct_counts_and_classification_are_exact(dtype, count):
    values = pd.Series(range(2**53, 2**53 + count), dtype=dtype)
    if dtype[0].isupper():
        values.loc[count] = pd.NA
    info = loading.color_column_info(values)
    assert info["n_distinct"] == count
    assert loading.color_kind(values) == ("categorical" if count == 2 else "continuous")
    assert info["n_missing"] == info["n_nonfinite"] == int(dtype[0].isupper())


@pytest.mark.parametrize("dtype", ["Int64", "UInt64"])
def test_large_integer_categories_match_transport_and_recolour_after_selection(tmp_path, dtype):
    from manyruns import figspec
    from manyruns.pipeline import io

    obj = AnnData(np.zeros((5, 2)), obs=pd.DataFrame({
        "group": pd.array([2**53, 2**53 + 1, None, 2**53 + 1, 2**53], dtype=dtype),
    }, index=[f"cell-{i}" for i in range(5)]))
    channel = loading.color_channel_of(obj, "group")
    rows = np.array([3, 2, 0])
    values = channel["values"][rows]
    assert values.dtype == obj.obs["group"].dtype
    assert int(values[0]) == 2**53 + 1 and int(values[2]) == 2**53
    assert pd.isna(values[1])
    png = io.save_display_scatter(obj.X[rows], [{**channel, "values": values}], tmp_path,
                                  "p.png", [], "Selected", obs_names=obj.obs_names[rows])[0]
    transported = figspec.load(png)
    aligned, positional = figspec.join(transported, obj.obs.iloc[::-1], "group")
    recoloured = figspec.recolor(transported, aligned, "group", "categorical")
    assert not positional
    assert set(transported["category_colors"]) == {str(2**53), str(2**53 + 1), "(missing)"}
    assert transported["category_colors"] == recoloured["category_colors"]
    np.testing.assert_array_equal(transported["color_values"], recoloured["color_values"])
    assert transported["n_missing"] == transported["n_nonfinite"] == 1


@pytest.mark.parametrize("key", ["category", "flag", "nullable", "score", "small_score",
                                 "literal_missing"])
@pytest.mark.parametrize("rows", [None, [29, 2, 0, 27, 1, 28], [0, 1]])
def test_metadata_channel_persists_selected_missing_counts(tmp_path, annotated, key, rows):
    from manyruns import figspec
    from manyruns.pipeline import io

    annotated.obs["small_score"] = pd.array([1., 2., np.inf, -np.inf, None] * 6,
                                             dtype="Float64")
    annotated.obs["literal_missing"] = ["(missing)", "present", None] * 10
    channel = loading.color_channel_of(annotated, key)
    rows = np.arange(len(annotated)) if rows is None else np.array(rows)
    selected = annotated.obs[key].iloc[rows]
    # Plot transport selects the original-axis values once. Full-source aggregate counts
    # deliberately remain in the channel and cannot describe these surviving rows.
    channel = {**channel, "values": channel["values"][rows]}
    path = io.save_display_scatter(np.zeros((len(rows), 2)), [channel], tmp_path, "p.png", [],
                                   "Selected", obs_names=annotated.obs_names[rows])[0]
    spec = figspec.load(path)
    expected_missing = int(selected.isna().sum())
    expected_nonfinite = expected_missing
    if pd.api.types.is_numeric_dtype(selected.dtype):
        expected_nonfinite = int((~np.isfinite(selected.to_numpy(dtype=float,
                                                                na_value=np.nan))).sum())
    assert spec["n_missing"] == expected_missing
    assert spec["n_nonfinite"] == expected_nonfinite
    if channel["kind"] == "categorical" and expected_nonfinite:
        assert "(missing)" in spec["category_colors"]
    np.testing.assert_array_equal(spec["obs_names"], annotated.obs_names[rows])


@pytest.mark.parametrize("dtype", [object, "category", "string"])
def test_literal_missing_categories_stay_distinct_from_missing_rows(tmp_path, dtype):
    from manyruns import figspec
    from manyruns.pipeline import io

    values = pd.Series(["(missing)", None, "(missing)*", "(missing)**", "a", "a", "b", "b"],
                       dtype=dtype, name="group")
    before = values.copy(deep=True)
    info = loading.color_column_info(values)
    normalized = loading.color_values_of(values, "categorical")
    literal_levels = set(values.dropna().astype(str))
    missing_label = normalized[1]
    assert missing_label not in literal_levels
    assert missing_label.startswith("(missing)")
    assert normalized[0] == "(missing)" and normalized[2] == "(missing)*"
    assert len(set(normalized)) == info["n_distinct"] + 1
    coords = np.arange(16).reshape(8, 2)
    channel = {"key": "group", "kind": "categorical", "values": values.array.copy()}
    png = io.save_display_scatter(coords, [channel], tmp_path, "p.png", [], "P")[0]
    spec = figspec.load(png)
    recoloured = figspec.recolor({"coords": coords}, values, "group", "categorical")
    for result in (spec, recoloured):
        assert result["n_missing"] == result["n_nonfinite"] == 1
        np.testing.assert_array_equal(result["color_values"], normalized)
        colors = result["category_colors"]
        assert set(colors) == literal_levels | {missing_label}
        assert colors[missing_label] != colors["(missing)"]
    pd.testing.assert_series_equal(values, before)


def test_shared_availability_explains_empty_identifiers_and_legend(annotated):
    columns = {c["key"]: c for c in loading.color_columns_of(annotated)}
    assert not columns["empty"]["available"]
    assert "missing" in columns["empty"]["reason"]
    assert not columns["barcode"]["available"]
    assert "unique" in columns["barcode"]["reason"]
    assert columns["many_groups"]["available"]
    assert "legend" in columns["many_groups"]["warning"]
    for key in ("empty", "barcode"):
        with pytest.raises(ValueError, match=columns[key]["reason"]):
            loading.color_channel_of(annotated, key)
    assert loading.color_channel_of(annotated, "many_groups")["kind"] == "categorical"
    with pytest.raises(ValueError, match="available columns:.*day.*score"):
        loading.color_channel_of(annotated, "absent")
    with pytest.raises(ValueError, match="no metadata"):
        loading.color_channel_of(object(), "day")


@pytest.mark.parametrize("series,expected", [
    (pd.Series(range(20)), "categorical"),
    (pd.Series(range(21)), "continuous"),
    (pd.Series([True, False]), "categorical"),
    (pd.Series(["one", None], dtype="string"), "categorical"),
    (pd.Series([np.inf, -np.inf, np.nan]), "categorical"),
])
def test_color_kind_counts_finite_values(series, expected):
    assert loading.color_kind(series) == expected


def test_generated_timepoint_is_display_metadata_and_still_analysis_time():
    obj = loading.load_array(loading.TIME_COURSE_REF)
    labels, kind = loading.labels_of(obj)
    channel = loading.color_channel_of(obj, "timepoint")
    assert channel["kind"] == "categorical"
    assert loading.label_key_of(obj) == ("timepoint", "time")
    assert obj.shape == (300, 5) and kind == "time"
    np.testing.assert_array_equal(channel["values"], labels)


def test_observation_columns_round_trip_without_changing_analysis_shape(tmp_path):
    import json
    from manyruns import app, inspected, narrate

    obj = AnnData(np.zeros((6, 3)), obs=pd.DataFrame({
        "day": [0, 1, 2] * 2, "disease": ["a", "b"] * 3,
        "hidden_levels": ["do-not-cache-this-level"] * 6,
    }, index=[f"cell-{i}" for i in range(6)]))
    path = tmp_path / "annotated.h5ad"
    obj.write_h5ad(path)
    assert len(app._inspect_obs(path)) == 3  # preservation: existing callers unpack three
    observation = narrate.read_data(path, "scrna")
    assert observation.shape == "time-course" and observation.n_vars == 3
    assert observation.columns == [["day", "categorical", 3], ["disease", "categorical", 2],
                                   ["hidden_levels", "categorical", 1]]
    row = json.loads(json.dumps(inspected.as_row("scrna", observation)))
    assert "do-not-cache-this-level" not in json.dumps(row)
    assert inspected.to_observation(row, observation.source) == observation
    del row["obs"]["columns"]
    assert inspected.to_observation(row, "old") is None


def test_metadata_inspection_is_backed_closes_and_distinguishes_unknown(tmp_path, monkeypatch):
    import anndata
    from manyruns import narrate

    path = tmp_path / "bare.h5ad"
    AnnData(np.zeros((2, 3))).write_h5ad(path)
    opened = []
    real = anndata.read_h5ad

    def read(*args, **kwargs):
        assert kwargs["backed"] == "r"
        obj = real(*args, **kwargs)
        opened.append(obj)
        return obj

    monkeypatch.setattr(anndata, "read_h5ad", read)
    assert narrate.metadata_columns(path) == []
    assert opened and not opened[0].file.is_open
    assert narrate.metadata_columns(tmp_path / "missing.h5ad") is None
    monkeypatch.setattr(loading, "load_array", lambda *a: pytest.fail("roster must not generate"))
    assert narrate.metadata_columns(loading.TIME_COURSE_REF) is None
    monkeypatch.setattr(loading, "color_columns_of", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    assert narrate.metadata_columns(path) is None
    assert not opened[-1].file.is_open


def test_label_key_group_fallback_and_explicit_display_do_not_create_time():
    obj = AnnData(np.zeros((4, 2)), obs=pd.DataFrame({
        "cell_type": ["a", "b"] * 2, "score": [10, 20, 30, 40],
    }, index=[f"cell-{i}" for i in range(4)]))
    assert loading.label_key_of(obj) == ("cell_type", "group")
    original, _ = loading.labels_of(obj)
    loading.color_channel_of(obj, "score")
    np.testing.assert_array_equal(loading.labels_of(obj)[0], original)
    del obj.obs["cell_type"]
    assert loading.label_key_of(obj) == (None, None)
    assert loading.labels_of(obj) == (None, None)


def test_all_nonfinite_values_are_unavailable_and_numeric_categories_keep_dtype_policy():
    series = pd.Series(pd.Categorical(range(25)))
    assert loading.color_kind(series) == "categorical"
    obj = AnnData(np.zeros((3, 2)), obs=pd.DataFrame({"score": [np.nan, np.inf, -np.inf]},
                                                   index=["a", "b", "c"]))
    col = loading.color_columns_of(obj)[0]
    assert col["n_missing"] == 1 and col["n_nonfinite"] == 3
    assert not col["available"]
    with pytest.raises(ValueError, match="all values are missing or nonfinite"):
        loading.color_channel_of(obj, "score")


def test_unique_high_cardinality_categories_stay_selectable_when_not_identifiers():
    obj = AnnData(np.zeros((25, 2)), obs=pd.DataFrame({
        "cluster": pd.Categorical([f"group-{i}" for i in range(25)]),
        "barcode": [f"cell-{i}" for i in range(25)],
    }, index=[f"cell-{i}" for i in range(25)]))
    columns = {c["key"]: c for c in loading.color_columns_of(obj)}
    assert columns["cluster"]["available"]
    assert "legend" in columns["cluster"]["warning"]
    assert loading.color_channel_of(obj, "cluster")["kind"] == "categorical"
    assert not columns["barcode"]["available"]
