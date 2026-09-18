"""The row-alignment guard (manyruns.aligned). Dep-free.

The guard's whole value is that it does NOT guess. Most of these tests are about the
shapes it must refuse — a permissive row count is how the guard grows holes and starts
certifying misaligned data as fine.
"""
from __future__ import annotations

import pytest

from manyruns.aligned import AlignmentError, check_aligned, n_rows


def _mat(n):
    return [[float(i), float(i)] for i in range(n)]


def _rows(n):
    return [{"stream_id": "s1", "sample": f"donor{i}"} for i in range(n)]


# ── the happy path ───────────────────────────────────────────────────────────
def test_matching_rows_pass_and_return_the_count():
    assert check_aligned(_mat(3), _rows(3)) == 3


def test_shape_wins_over_len_for_array_likes():
    class FakeArray:
        shape = (5, 2)

        def __len__(self):
            return 999  # a DataFrame's len() is rows, but an ndarray's is rows too — use .shape

    assert check_aligned(FakeArray(), _rows(5)) == 5


# ── the guard's reason to exist ──────────────────────────────────────────────
def test_row_misalignment_raises():
    with pytest.raises(AlignmentError, match="row misalignment"):
        check_aligned(_mat(4), _rows(3))


def test_off_by_one_raises():
    with pytest.raises(AlignmentError, match="matrix has 100 rows .* metadata has 99"):
        check_aligned(_mat(100), _rows(99))


def test_the_error_names_the_source():
    with pytest.raises(AlignmentError, match=r"load_labeled\(eb\)"):
        check_aligned(_mat(2), _rows(1), what="load_labeled(eb)")


# ── the holes it must NOT have (the previous draft passed all of these) ──────
def test_a_columnar_dict_is_refused_not_counted_by_columns():
    """A dict of columns has len() == n_COLUMNS. Counting it validates the wrong axis.

    This is the exact shape you build before pd.DataFrame(...) — 3 rows, 2 columns. The
    earlier duck-typed version returned 2 here and ACCEPTED a 2-row matrix against it.
    """
    meta = {"sample": ["a", "b", "c"], "stream_id": ["s", "s", "s"]}  # 3 rows, 2 cols
    assert n_rows(meta) is None
    with pytest.raises(AlignmentError, match="counts COLUMNS"):
        check_aligned(_mat(2), meta)


def test_a_string_is_refused_not_counted_by_characters():
    assert n_rows("abc") is None
    with pytest.raises(AlignmentError, match="no honest row count"):
        check_aligned(_mat(3), "abc")  # len("abc") == 3 would have "aligned"


def test_a_set_is_refused():
    with pytest.raises(AlignmentError, match="no honest row count"):
        check_aligned(_mat(2), {"a", "b"})


def test_a_matrix_with_no_row_count_is_refused():
    with pytest.raises(AlignmentError, match="matrix has no row count"):
        check_aligned(object(), _rows(3))


def test_a_scalar_shape_does_not_crash():
    class Odd:
        shape = ()

    assert n_rows(Odd()) is None


# ── n_rows directly ──────────────────────────────────────────────────────────
def test_n_rows_on_the_shapes_with_an_honest_answer():
    assert n_rows(_mat(3)) == 3
    assert n_rows((1, 2)) == 2
    assert n_rows([]) == 0


def test_n_rows_with_numpy():
    np = pytest.importorskip("numpy")
    assert n_rows(np.zeros((7, 3))) == 7


def test_n_rows_with_a_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"sample": ["a", "b", "c"]})  # 3 rows, 1 column
    assert n_rows(df) == 3, "a DataFrame must count ROWS (.shape[0]), not columns"
    assert check_aligned(_mat(3), df) == 3


def test_zero_rows_is_aligned_not_an_error():
    assert check_aligned([], []) == 0
