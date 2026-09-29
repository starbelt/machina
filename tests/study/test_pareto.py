"""
The Pareto layer, ``machina.study.pareto``: the non-dominated set of a table of
configurations, ideal/nadir normalization, the weighted-sum switch points and
the epsilon-constraint pick. numpy only; the DataFrame adapter test skips
without pandas.

Test classes
------------
TestObjectivesAreValidated                  -- sense, names, plain strings, duplicates
TestFrontOfMinimizedObjectives              -- min/min front, duplicates, weak dominance
TestMaximizedObjectivesAreFlipped           -- max sense
TestInvalidRowsAreListedNotScored           -- NaN, inf, log of non-positive, infeasible
TestEveryTableFormReadsTheSame              -- mapping, rows, as_columns(), DataFrame, errors
TestRowSelectionsArePositions               -- pandas Index / Series, bool-int mixtures refused
TestLogDoesNotChangeDominance               -- same front, non-positive rows excluded
TestFrontMatchesBruteForce                  -- seeded random tables vs an O(n²) reference
TestNormalizationMapsIdealToZeroNadirToOne  -- both senses, log, over, nadir, degenerate
TestZeroSpans                               -- front-nadir fallback, ±inf off a degenerate
                                               ideal, overflowing spans
TestMaximizedObjectiveEntersAsItsNormalizedValue -- the best η gets η̂ = 0
TestTwoObjectiveSwitchPoints                -- exact hull, switch points, support classes
TestSimplexGridForMoreObjectives            -- m = 3 grid count, order, determinism, ties
TestEpsilonConstraint                       -- bounds in both senses, ties, no qualifier
"""

import math

import numpy as np
import pytest

from machina.study import (
    Normalized,
    Objective,
    Segment,
    WeightedSweep,
    epsilon_constraint,
    excluded,
    nondominated,
    normalize,
    pareto,
    weighted_sweep,
)

LATENCY_ETA = [Objective("L", "min", log=True), Objective("eta", "max")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def front_reference(rows, senses):
    """Pure-Python O(n²) non-dominated mask; ``rows`` is a list of value tuples."""
    signs = [1.0 if s == "min" else -1.0 for s in senses]
    out = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if j == i:
                continue
            no_worse = all(g * o <= g * r for g, o, r in zip(signs, other, row))
            better = any(g * o < g * r for g, o, r in zip(signs, other, row))
            if no_worse and better:
                dominated = True
                break
        out.append(not dominated)
    return out


def sweep_points(points, **kwargs):
    """``weighted_sweep`` over 2-D points (columns ``a``, ``b``, both minimized)."""
    a, b = zip(*points)
    return weighted_sweep({"a": list(a), "b": list(b)}, ["a", "b"], **kwargs)


def argmin_rows(values, w, tol=1e-12):
    """Rows minimizing ``w·a + (1 − w)·b`` within ``tol`` over the finite rows of ``values``."""
    rows = np.flatnonzero(np.isfinite(values).all(axis=1))
    j = w * values[rows, 0] + (1.0 - w) * values[rows, 1]
    return tuple(int(r) for r in rows[j <= j.min() + tol])


def segment_at(sweep, w):
    """The segment whose open interval holds ``w``."""
    for segment in sweep.segments:
        if segment.w_lo < w < segment.w_hi:
            return segment
    raise AssertionError(f"no segment holds w = {w}")


def random_points(seed, n):
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(0.0, 1.0, size=(n, 2))]


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

class TestObjectivesAreValidated:

    def test_sense_must_be_min_or_max(self):
        with pytest.raises(ValueError, match="sense must be 'min' or 'max'"):
            Objective("L", "minimize")

    def test_name_must_be_a_non_empty_string(self):
        with pytest.raises(ValueError, match="non-empty column name"):
            Objective("")

    def test_log_must_be_a_bool(self):
        with pytest.raises(TypeError, match="log must be True or False"):
            Objective("L", log=1)

    def test_a_plain_string_is_a_minimized_linear_objective(self):
        table = {"a": [1.0, 2.0, 3.0], "b": [3.0, 1.0, 4.0]}
        np.testing.assert_array_equal(
            nondominated(table, ["a", "b"]),
            nondominated(table, [Objective("a"), Objective("b", "min", log=False)]))

    def test_an_objective_listed_twice_is_an_error(self):
        with pytest.raises(ValueError, match="listed twice"):
            nondominated({"a": [1.0]}, ["a", Objective("a", "max")])

    def test_a_single_objective_not_in_a_list_is_an_error(self):
        with pytest.raises(TypeError, match="wrap it in a list"):
            nondominated({"a": [1.0]}, "a")


# ---------------------------------------------------------------------------
# Dominance
# ---------------------------------------------------------------------------

class TestFrontOfMinimizedObjectives:

    def test_the_front_of_a_small_table(self):
        table = {"a": [1.0, 2.0, 3.0, 2.5, 4.0],
                 "b": [5.0, 3.0, 1.0, 3.5, 0.5]}
        # (2.5, 3.5) is dominated by (2.0, 3.0); the others trade off.
        np.testing.assert_array_equal(nondominated(table, ["a", "b"]),
                                      [True, True, True, False, True])

    def test_exact_duplicates_are_all_kept(self):
        table = {"a": [1.0, 2.0, 1.0, 2.0], "b": [2.0, 1.0, 2.0, 1.0]}
        np.testing.assert_array_equal(nondominated(table, ["a", "b"]), [True] * 4)

    def test_a_weakly_dominated_row_is_removed(self):
        """Equal on one objective and worse on the other is dominated."""
        table = {"a": [1.0, 1.0, 2.0], "b": [2.0, 3.0, 2.0]}
        np.testing.assert_array_equal(nondominated(table, ["a", "b"]), [True, False, False])

    def test_the_block_size_does_not_change_the_answer(self, monkeypatch):
        points = random_points(7, 150)
        table = {"a": [p[0] for p in points], "b": [p[1] for p in points]}
        expected = nondominated(table, ["a", "b"])
        monkeypatch.setattr(pareto, "_BLOCK", 16)
        np.testing.assert_array_equal(nondominated(table, ["a", "b"]), expected)
        np.testing.assert_array_equal(expected, front_reference(points, ["min", "min"]))


class TestMaximizedObjectivesAreFlipped:

    def test_max_sense_keeps_the_largest_values(self):
        table = {"L": [1.0, 2.0, 3.0, 2.0], "eta": [0.2, 0.6, 0.9, 0.5]}
        # (2.0, 0.5) loses to (2.0, 0.6) on eta.
        np.testing.assert_array_equal(
            nondominated(table, ["L", Objective("eta", "max")]), [True, True, True, False])

    def test_flipping_both_senses_reverses_a_chain(self):
        table = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}
        both_max = [Objective("a", "max"), Objective("b", "max")]
        np.testing.assert_array_equal(nondominated(table, ["a", "b"]), [True, False, False])
        np.testing.assert_array_equal(nondominated(table, both_max), [False, False, True])


class TestInvalidRowsAreListedNotScored:

    def test_a_nan_row_is_off_the_front_and_listed(self):
        table = {"L": [2.0, float("nan"), 3.0], "eta": [0.5, 0.99, 0.7]}
        np.testing.assert_array_equal(nondominated(table, LATENCY_ETA), [True, False, True])
        assert excluded(table, LATENCY_ETA) == [(1, "nan in 'L'")]

    def test_an_infinite_value_is_listed(self):
        table = {"L": [2.0, float("inf")], "eta": [0.5, 0.9]}
        assert excluded(table, LATENCY_ETA) == [(1, "inf in 'L'")]

    def test_an_infeasible_row_does_not_dominate(self):
        """Row 1 would dominate everything; infeasible, it knocks nothing off."""
        table = {"L": [2.0, 1.0, 3.0], "eta": [0.5, 0.99, 0.7], "ok": [True, False, True]}
        np.testing.assert_array_equal(nondominated(table, LATENCY_ETA, feasible="ok"),
                                      [True, False, True])
        np.testing.assert_array_equal(nondominated(table, LATENCY_ETA),
                                      [False, True, False])
        assert excluded(table, LATENCY_ETA, feasible="ok") == [(1, "infeasible")]

    def test_feasible_as_a_mask_matches_feasible_as_a_column(self):
        table = {"L": [2.0, 1.0, 3.0], "eta": [0.5, 0.99, 0.7], "ok": [1, 0, 1]}
        by_column = nondominated(table, LATENCY_ETA, feasible="ok")
        np.testing.assert_array_equal(
            nondominated(table, LATENCY_ETA, feasible=[True, False, True]), by_column)
        np.testing.assert_array_equal(
            nondominated(table, LATENCY_ETA, feasible=np.array([True, False, True])),
            by_column)

    def test_log_of_a_non_positive_value_is_listed(self):
        table = {"L": [0.0, -1.0, 2.0], "eta": [0.9, 0.9, 0.1]}
        assert excluded(table, LATENCY_ETA) == [
            (0, "log of non-positive 'L'"), (1, "log of non-positive 'L'")]
        np.testing.assert_array_equal(nondominated(table, LATENCY_ETA), [False, False, True])

    def test_several_reasons_are_joined_in_a_fixed_order(self):
        table = {"L": [float("nan"), 1.0], "eta": [float("nan"), 0.5], "ok": [False, True]}
        assert excluded(table, LATENCY_ETA, feasible="ok") == [
            (0, "nan in 'L'; nan in 'eta'; infeasible")]

    def test_a_nan_in_the_feasible_column_is_an_error(self):
        table = {"L": [1.0, 2.0], "eta": [0.5, 0.6], "ok": [1.0, float("nan")]}
        with pytest.raises(ValueError, match="feasible column 'ok' is NaN at row 1"):
            nondominated(table, LATENCY_ETA, feasible="ok")

    def test_a_feasible_mask_of_the_wrong_length_is_an_error(self):
        with pytest.raises(ValueError, match="feasible has 1 entries but the table has 2"):
            nondominated({"L": [1.0, 2.0], "eta": [0.5, 0.6]}, LATENCY_ETA, feasible=[True])

    def test_a_float32_nan_in_an_object_feasible_column_is_an_error(self):
        """Any real NaN, not only a Python float; bool(nan) would read it as feasible."""
        table = {"L": [1.0, 2.0], "eta": [0.2, 0.9],
                 "ok": np.array([True, np.float32("nan")], dtype=object)}
        with pytest.raises(ValueError, match="feasible column 'ok' holds .*nan.* at row 1; "
                                             "give True or False for every row"):
            excluded(table, LATENCY_ETA, feasible="ok")

    def test_a_pandas_na_in_a_boolean_feasible_column_is_an_error(self):
        pd = pytest.importorskip("pandas")
        table = pd.DataFrame({"L": [1.0, 2.0, 3.0], "eta": [0.2, 0.5, 0.9],
                              "ok": pd.array([True, None, True], dtype="boolean")})
        with pytest.raises(ValueError, match="feasible column 'ok' holds <NA> at row 1; "
                                             "give True or False for every row"):
            excluded(table, LATENCY_ETA, feasible="ok")


# ---------------------------------------------------------------------------
# Table adapters
# ---------------------------------------------------------------------------

class _Columns:
    """Stands in for a table type that exposes ``as_columns()``."""

    def __init__(self, columns):
        self._columns = columns

    def as_columns(self):
        return self._columns


COLUMNS = {"L": [3.0, 1.0, 2.0, 4.0], "eta": [0.9, 0.2, 0.5, 0.8], "name": list("wxyz")}
FRONT = [True, True, True, False]


class TestEveryTableFormReadsTheSame:

    def test_a_mapping_of_lists_and_of_arrays(self):
        arrays = {k: np.asarray(v) for k, v in COLUMNS.items()}
        np.testing.assert_array_equal(nondominated(COLUMNS, LATENCY_ETA), FRONT)
        np.testing.assert_array_equal(nondominated(arrays, LATENCY_ETA), FRONT)

    def test_a_list_of_rows(self):
        rows = [dict(zip(COLUMNS, values)) for values in zip(*COLUMNS.values())]
        np.testing.assert_array_equal(nondominated(rows, LATENCY_ETA), FRONT)

    def test_a_missing_key_in_a_row_reads_as_nan(self):
        rows = [{"L": 1.0, "eta": 0.2}, {"L": 2.0}, {"eta": 0.9, "L": 3.0, "extra": "x"}]
        assert excluded(rows, LATENCY_ETA) == [(1, "nan in 'eta'")]
        np.testing.assert_array_equal(nondominated(rows, LATENCY_ETA), [True, False, True])

    def test_an_object_with_as_columns(self):
        np.testing.assert_array_equal(nondominated(_Columns(COLUMNS), LATENCY_ETA), FRONT)

    def test_a_dataframe(self):
        pd = pytest.importorskip("pandas")
        np.testing.assert_array_equal(nondominated(pd.DataFrame(COLUMNS), LATENCY_ETA), FRONT)

    def test_an_unknown_objective_lists_the_columns(self):
        with pytest.raises(KeyError, match=r"'latency' is not a column.*\['L', 'eta', 'name'\]"):
            nondominated(COLUMNS, ["latency"])

    def test_a_non_numeric_objective_column_is_an_error(self):
        with pytest.raises(TypeError, match="objective column 'name' is not numeric"):
            nondominated(COLUMNS, ["L", "name"])

    def test_a_string_in_an_object_column_names_the_row(self):
        table = {"a": np.array([1.0, "2", 3.0], dtype=object), "b": [1.0, 2.0, 3.0]}
        with pytest.raises(TypeError, match="non-numeric value '2' at row 1"):
            nondominated(table, ["a", "b"])

    def test_columns_of_different_lengths_are_an_error(self):
        with pytest.raises(ValueError, match="different lengths"):
            nondominated({"a": [1.0, 2.0], "b": [1.0]}, ["a", "b"])

    def test_a_row_that_is_not_a_mapping_is_an_error(self):
        with pytest.raises(TypeError, match="row 1 of the table is a tuple"):
            nondominated([{"a": 1.0}, (2.0,)], ["a"])

    def test_an_unreadable_table_is_an_error(self):
        with pytest.raises(TypeError, match="cannot read a table from a int"):
            nondominated(42, ["a"])


class TestRowSelectionsArePositions:
    """``over`` and ``among`` select row positions; a label must never pass for one."""

    @staticmethod
    def sorted_frame(pd):
        """Sorted by L, so the index labels are no longer the positions: [5, 1, 3, 2, 4, 0]."""
        frame = pd.DataFrame({"L": [5.0, 1.0, 3.0, 2.0, 4.0, 0.5],
                              "eta": [0.9, 0.2, 0.6, 0.4, 0.8, 0.95],
                              "kind": ["d", "d", "d", "d", "d", "oracle"]})
        return frame.sort_values("L")

    def test_a_pandas_index_is_refused(self):
        pd = pytest.importorskip("pandas")
        frame = self.sorted_frame(pd)
        labels = frame.index[frame.kind == "d"]
        with pytest.raises(TypeError, match="pandas Index.*labels are not positions"):
            normalize(frame, LATENCY_ETA, over=labels)
        with pytest.raises(TypeError, match="`among` is a pandas Index"):
            weighted_sweep(frame, LATENCY_ETA, among=labels)

    def test_a_non_boolean_series_is_refused(self):
        pd = pytest.importorskip("pandas")
        frame = self.sorted_frame(pd)
        with pytest.raises(TypeError, match="Series of dtype int64.*labels are not positions"):
            normalize(frame, LATENCY_ETA, over=pd.Series([1, 2, 3]))

    def test_a_boolean_series_is_a_positional_mask(self):
        pd = pytest.importorskip("pandas")
        frame = self.sorted_frame(pd)
        mask = frame.kind == "d"
        by_series = normalize(frame, LATENCY_ETA, over=mask)
        by_position = normalize(frame, LATENCY_ETA, over=np.flatnonzero(mask.to_numpy()))
        np.testing.assert_array_equal(by_series.values, by_position.values)
        assert by_series.values[0, 1] < 0.0  # the oracle, first after sorting, beats the ideal

    def test_a_list_mixing_bools_and_ints_is_refused(self):
        """np.asarray([True, False, 1]) is [1, 0, 1]: rows 0 and 1, not a mask."""
        with pytest.raises(TypeError, match="`over` mixes booleans and ints"):
            normalize({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]}, ["a", "b"],
                      over=[True, False, 1])


class TestLogDoesNotChangeDominance:

    def test_the_front_is_the_same_with_and_without_log(self):
        rng = np.random.default_rng(3)
        table = {"L": rng.lognormal(0.0, 3.0, 200), "eta": rng.uniform(0.0, 1.0, 200)}
        linear = [Objective("L"), Objective("eta", "max")]
        np.testing.assert_array_equal(nondominated(table, LATENCY_ETA),
                                      nondominated(table, linear))

    def test_non_positive_rows_are_excluded_only_on_a_log_axis(self):
        table = {"L": [0.0, 1.0, 2.0], "eta": [0.1, 0.5, 0.9]}
        linear = [Objective("L"), Objective("eta", "max")]
        np.testing.assert_array_equal(nondominated(table, linear), [True, True, True])
        np.testing.assert_array_equal(nondominated(table, LATENCY_ETA), [False, True, True])


class TestFrontMatchesBruteForce:

    @pytest.mark.parametrize("seed", range(8))
    def test_random_tables_with_ties(self, seed):
        rng = np.random.default_rng(seed)
        m = 2 + seed % 3
        n = 60
        # Small integer values force ties and exact duplicates; one column stays continuous.
        values = rng.integers(0, 5, size=(n, m)).astype(float)
        values[:, 0] += rng.choice([0.0, 0.5], size=n)
        senses = [("min", "max")[k] for k in rng.integers(0, 2, size=m)]
        table = {f"x{i}": values[:, i] for i in range(m)}
        objectives = [Objective(f"x{i}", senses[i]) for i in range(m)]
        expected = front_reference([tuple(r) for r in values], senses)
        np.testing.assert_array_equal(nondominated(table, objectives), expected)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Front: (1, .2), (2, .5), (4, .9); row 3 (3, .4) is dominated by row 1,
# row 4 (6, .1) by everything.
DESIGNS = {"L": [1.0, 2.0, 4.0, 3.0, 6.0], "eta": [0.2, 0.5, 0.9, 0.4, 0.1]}
LINEAR = [Objective("L"), Objective("eta", "max")]


class TestNormalizationMapsIdealToZeroNadirToOne:

    def test_both_senses_against_the_front_nadir(self):
        norm = normalize(DESIGNS, LINEAR)
        np.testing.assert_allclose(norm.ideal, [1.0, 0.9], rtol=0, atol=1e-15)
        np.testing.assert_allclose(norm.nadir, [4.0, 0.2], rtol=0, atol=1e-15)
        L, eta = np.array(DESIGNS["L"]), np.array(DESIGNS["eta"])
        np.testing.assert_allclose(norm.values[:, 0], (L - 1.0) / 3.0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(norm.values[:, 1], (eta - 0.9) / (0.2 - 0.9),
                                   rtol=0, atol=1e-15)
        # Ideal rows at 0, front-nadir rows at 1, whatever the sense.
        assert norm.values[0, 0] == 0.0 and norm.values[2, 0] == 1.0
        assert norm.values[2, 1] == 0.0 and norm.values[0, 1] == 1.0

    def test_a_dominated_row_can_lie_beyond_the_front_nadir(self):
        norm = normalize(DESIGNS, LINEAR)
        assert norm.values[4, 0] > 1.0 and norm.values[4, 1] > 1.0

    def test_nadir_all_takes_the_worst_valid_row(self):
        norm = normalize(DESIGNS, LINEAR, nadir="all")
        np.testing.assert_allclose(norm.nadir, [6.0, 0.1], rtol=0, atol=1e-15)
        assert np.nanmax(norm.values) == 1.0 and np.nanmin(norm.values) == 0.0

    def test_log_is_applied_before_normalizing(self):
        table = {"L": [1.0, 10.0, 100.0], "eta": [0.1, 0.5, 0.9]}
        logged = normalize(table, LATENCY_ETA)
        np.testing.assert_allclose(logged.values[:, 0], [0.0, 0.5, 1.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(logged.ideal[0], 0.0, rtol=0, atol=1e-15)
        np.testing.assert_allclose(logged.nadir[0], math.log(100.0), rtol=1e-15)
        linear = normalize(table, LINEAR)
        np.testing.assert_allclose(linear.values[:, 0], [0.0, 9.0 / 99.0, 1.0],
                                   rtol=0, atol=1e-15)

    def test_a_row_outside_over_maps_outside_the_unit_interval(self):
        """An oracle upper bound is scored against the designs, not part of their range."""
        table = {"L": DESIGNS["L"] + [0.5], "eta": DESIGNS["eta"] + [1.0]}
        designs = [0, 1, 2, 3, 4]
        by_index = normalize(table, LINEAR, over=designs)
        by_mask = normalize(table, LINEAR, over=[True] * 5 + [False])
        np.testing.assert_array_equal(by_index.values, by_mask.values)
        np.testing.assert_allclose(by_index.ideal, [1.0, 0.9], rtol=0, atol=1e-15)
        np.testing.assert_allclose(by_index.values[5], [(0.5 - 1.0) / 3.0, (1.0 - 0.9) / -0.7],
                                   rtol=1e-14)
        assert (by_index.values[5] < 0.0).all()
        np.testing.assert_allclose(by_index.values[:5], normalize(DESIGNS, LINEAR).values,
                                   rtol=0, atol=1e-15)

    def test_a_constant_objective_is_degenerate_and_zero(self):
        table = {"a": [1.0, 2.0, 3.0, float("nan")], "c": [5.0, 5.0, 5.0, 5.0]}
        norm = normalize(table, ["a", "c"], nadir="all")
        assert norm.degenerate == ("c",)
        np.testing.assert_array_equal(norm.values[:3, 1], [0.0, 0.0, 0.0])
        assert np.isnan(norm.values[3]).all()

    def test_excluded_rows_are_nan_and_the_record_is_read_only(self):
        table = {"L": [1.0, float("nan"), 3.0], "eta": [0.2, 0.5, 0.9]}
        norm = normalize(table, LINEAR)
        assert isinstance(norm, Normalized)
        assert np.isnan(norm.values[1]).all() and np.isfinite(norm.values[[0, 2]]).all()
        assert norm.objectives == tuple(LINEAR)
        with pytest.raises(ValueError):
            norm.values[0, 0] = 1.0

    def test_an_unknown_nadir_rule_is_an_error(self):
        with pytest.raises(ValueError, match="nadir must be 'front' or 'all'"):
            normalize(DESIGNS, LINEAR, nadir="worst")

    def test_no_valid_row_in_over_is_an_error(self):
        table = {"L": [1.0, float("nan")], "eta": [0.2, 0.5]}
        with pytest.raises(ValueError, match="no valid row in `over`"):
            normalize(table, LINEAR, over=[1])


class TestZeroSpans:

    def test_a_dominating_row_takes_the_nadir_from_all_rows(self):
        """Row 0 dominates, so the front is one point and its span is 0 on both objectives."""
        table = {"L": [100.0, 1000.0, 5000.0], "eta": [0.9, 0.5, 0.1]}
        norm = normalize(table, LATENCY_ETA)
        assert norm.nadir_fallback == ("L", "eta")
        assert norm.degenerate == ()
        np.testing.assert_allclose(norm.nadir, [math.log(5000.0), 0.1], rtol=1e-15, atol=0)
        span = math.log(5000.0) - math.log(100.0)
        np.testing.assert_allclose(
            norm.values[:, 0], [0.0, (math.log(1000.0) - math.log(100.0)) / span, 1.0],
            rtol=1e-14, atol=0)
        np.testing.assert_allclose(norm.values[:, 1], [0.0, 0.5, 1.0], rtol=1e-14, atol=0)

    def test_the_fallback_is_per_objective(self):
        """Row 0 dominates; 'a' falls back to all rows, the constant 'c' stays degenerate."""
        norm = normalize({"a": [1.0, 2.0, 3.0], "c": [5.0, 5.0, 5.0]}, ["a", "c"])
        assert norm.nadir_fallback == ("a",)
        assert norm.degenerate == ("c",)
        np.testing.assert_array_equal(norm.values, [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])

    def test_no_fallback_when_the_front_has_a_span(self):
        assert normalize(DESIGNS, LINEAR).nadir_fallback == ()
        assert normalize(DESIGNS, LINEAR, nadir="all").nadir_fallback == ()

    @pytest.mark.parametrize("nadir", ["front", "all"])
    def test_rows_off_a_degenerate_ideal_are_infinite(self, nadir):
        """Over rows 0-2 'eta' (max) and 'P' (min) are constant; rows 3 and 4 are not."""
        table = {"L": [10.0, 20.0, 40.0, 5.0, 50.0], "eta": [0.5, 0.5, 0.5, 0.1, 0.9],
                 "P": [2.0, 2.0, 2.0, 3.0, 1.0]}
        objectives = [Objective("L", log=True), Objective("eta", "max"), Objective("P")]
        norm = normalize(table, objectives, over=[0, 1, 2], nadir=nadir)
        assert norm.degenerate == ("eta", "P")
        inf = math.inf
        np.testing.assert_array_equal(norm.values[:, 1], [0.0, 0.0, 0.0, inf, -inf])
        np.testing.assert_array_equal(norm.values[:, 2], [0.0, 0.0, 0.0, inf, -inf])
        assert np.isfinite(norm.values[:, 0]).all()

    def test_the_sweep_refuses_candidates_that_differ_on_a_degenerate_objective(self):
        """Row 3 has the worse eta; with eta flattened to 0 it used to win at w = 0."""
        table = {"L": [10.0, 20.0, 40.0, 5.0], "eta": [0.5, 0.5, 0.5, 0.1]}
        with pytest.raises(ValueError, match=r"candidates differ on objective 'eta' \(rows \[3\]"
                                             r".*zero span over `over`.*widen `over`"):
            weighted_sweep(table, LATENCY_ETA, over=[0, 1, 2], among=[0, 1, 2, 3], nadir="all")

    @pytest.mark.filterwarnings("error")
    def test_an_overflowing_span_is_an_error(self):
        """-1e308 to 1e308 is a span of inf: x̂ was NaN in a valid row, with no error."""
        table = {"a": [-1e308, 1e308, 0.0], "b": [1.0, 0.0, 0.5]}
        with pytest.raises(ValueError, match="normalize: objective 'a' runs from .* a span "
                                             "wider than a float holds; rescale the column 'a'"):
            normalize(table, ["a", "b"])
        with pytest.raises(ValueError, match="objective 'a' .* wider than a float holds"):
            weighted_sweep(table, ["a", "b"])


class TestMaximizedObjectiveEntersAsItsNormalizedValue:
    """
    With x̂ = (x − ideal)/(nadir − ideal), the best η gets η̂ = 0, so J_w uses η̂
    directly. ``(1 − η̂)`` would score the best η worst.
    """

    TABLE = {"L": [10.0, 1000.0, 100.0], "eta": [0.3, 0.95, 0.7]}

    def test_the_best_eta_normalizes_to_zero(self):
        norm = normalize(self.TABLE, LATENCY_ETA)
        assert norm.values[1, 1] == 0.0 and norm.values[0, 1] == 1.0

    def test_all_weight_on_eta_picks_the_best_eta(self):
        eta_hat = normalize(self.TABLE, LATENCY_ETA).values[:, 1]
        assert int(np.argmin(eta_hat)) == 1
        assert int(np.argmin(1.0 - eta_hat)) == 0  # the wrong form picks the worst eta
        sweep = weighted_sweep(self.TABLE, LATENCY_ETA)
        assert sweep.segments[0].rows == (1,)   # w = 0: all weight on eta
        assert sweep.segments[-1].rows == (0,)  # w = 1: all weight on L


# ---------------------------------------------------------------------------
# Weighted sweep
# ---------------------------------------------------------------------------

class TestTwoObjectiveSwitchPoints:

    def test_two_points_switch_at_the_closed_form_weight(self):
        table = {"L": [2.0, 8.0, 1.0, 10.0], "eta": [0.3, 0.9, 0.1, 0.95]}
        sweep = weighted_sweep(table, LATENCY_ETA, among=[0, 1], nadir="all")
        a = np.log(np.array([2.0, 8.0])) / np.log(10.0)
        b = (np.array([0.3, 0.9]) - 0.95) / (0.1 - 0.95)
        expected = (b[0] - b[1]) / ((b[0] - b[1]) + (a[1] - a[0]))
        np.testing.assert_allclose(sweep.switch_points, [expected], rtol=1e-14)
        assert sweep.segments == (Segment(0.0, sweep.switch_points[0], (1,)),
                                  Segment(sweep.switch_points[0], 1.0, (0,)))
        w = sweep.switch_points[0]
        np.testing.assert_allclose(w * a[0] + (1 - w) * b[0], w * a[1] + (1 - w) * b[1],
                                   rtol=1e-14)

    @pytest.mark.parametrize("seed", range(4))
    def test_the_envelope_matches_a_dense_weight_grid(self, seed):
        sweep = sweep_points(random_points(seed, 40))
        values = sweep.normalized.values
        breakpoints = np.array((0.0,) + sweep.switch_points + (1.0,))
        checked = 0
        for w in np.linspace(0.0, 1.0, 10_001):
            if np.min(np.abs(breakpoints - w)) < 1e-6:
                continue
            assert segment_at(sweep, w).rows == argmin_rows(values, w), f"w = {w}"
            checked += 1
        assert checked > 9_900

    @pytest.mark.parametrize("seed", range(4))
    def test_the_support_classes_partition_the_front(self, seed):
        points = random_points(seed, 40)
        sweep = sweep_points(points)
        front = tuple(int(r) for r in np.flatnonzero(front_reference(points, ["min", "min"])))
        together = sweep.supported + sweep.weakly_supported + sweep.unsupported
        assert tuple(sorted(together)) == front
        assert len(set(together)) == len(together)

    def test_segments_tile_the_unit_interval(self):
        sweep = sweep_points(random_points(11, 60))
        assert isinstance(sweep, WeightedSweep)
        assert sweep.segments[0].w_lo == 0.0 and sweep.segments[-1].w_hi == 1.0
        for left, right in zip(sweep.segments, sweep.segments[1:]):
            assert left.w_hi == right.w_lo
            assert left.w_lo < left.w_hi
        assert sweep.switch_points == tuple(s.w_hi for s in sweep.segments[:-1])
        assert list(sweep.switch_points) == sorted(sweep.switch_points)
        assert sweep.grid == ()

    def test_a_concave_point_never_wins(self):
        sweep = sweep_points([(0.0, 1.0), (1.0, 0.0), (0.6, 0.6)], nadir="all")
        assert sweep.supported == (0, 1)
        assert sweep.unsupported == (2,)
        assert sweep.weakly_supported == ()
        assert sweep.switch_points == (0.5,)
        for w in np.linspace(0.0, 1.0, 101):
            assert 2 not in argmin_rows(sweep.normalized.values, w)

    def test_a_collinear_point_is_weakly_supported(self):
        sweep = sweep_points([(0.0, 1.0), (1.0, 0.0), (0.25, 0.75)], nadir="all")
        assert sweep.supported == (0, 1)
        assert sweep.weakly_supported == (2,)
        assert sweep.unsupported == ()
        assert [s.rows for s in sweep.segments] == [(1,), (0,)]

    def test_duplicate_vertices_share_a_segment(self):
        sweep = sweep_points([(0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.3, 0.3)], nadir="all")
        assert [s.rows for s in sweep.segments] == [(1,), (3,), (0, 2)]
        np.testing.assert_allclose(sweep.switch_points, [0.3, 0.7], rtol=1e-14)
        assert sweep.supported == (0, 1, 2, 3)

    def test_endpoint_ties_go_to_the_better_other_objective(self):
        """At w = 1 rows 0 and 1 tie on a; at w = 0 rows 2 and 3 tie on b."""
        sweep = sweep_points([(0.0, 0.4), (0.0, 1.0), (1.0, 0.0), (0.95, 0.0)], nadir="all")
        assert sweep.segments[-1].rows == (0,)
        assert sweep.segments[0].rows == (3,)
        listed = sweep.supported + sweep.weakly_supported + sweep.unsupported
        assert 1 not in listed and 2 not in listed

    def test_a_single_candidate_wins_everywhere(self):
        sweep = sweep_points(random_points(5, 10), among=[4])
        assert sweep.segments == (Segment(0.0, 1.0, (4,)),)
        assert sweep.switch_points == ()
        assert sweep.supported == (4,)

    def test_identical_rows_form_one_segment(self):
        sweep = sweep_points([(2.0, 3.0)] * 3)
        assert sweep.segments == (Segment(0.0, 1.0, (0, 1, 2)),)
        assert sweep.normalized.degenerate == ("a", "b")

    def test_a_degenerate_objective_leaves_the_other_to_decide(self):
        sweep = sweep_points([(0.4, 1.0), (0.2, 1.0), (0.9, 1.0)], nadir="all")
        assert sweep.normalized.degenerate == ("b",)
        assert sweep.segments == (Segment(0.0, 1.0, (1,)),)

    def test_candidates_default_to_over(self):
        """The oracle row 3 is outside ``over``, so it is no candidate unless ``among`` says so."""
        table = {"a": [0.0, 1.0, 0.5, -1.0], "b": [1.0, 0.0, 0.5, -1.0]}
        designs = weighted_sweep(table, ["a", "b"], over=[0, 1, 2])
        assert 3 not in designs.supported + designs.weakly_supported + designs.unsupported
        with_oracle = weighted_sweep(table, ["a", "b"], over=[0, 1, 2], among=[0, 1, 2, 3])
        assert with_oracle.segments == (Segment(0.0, 1.0, (3,)),)

    def test_one_objective_is_an_error(self):
        with pytest.raises(ValueError, match="at least two objectives"):
            weighted_sweep({"a": [1.0, 2.0]}, ["a"])


class TestSimplexGridForMoreObjectives:

    @staticmethod
    def table(seed=0, n=30):
        rng = np.random.default_rng(seed)
        return {f"x{i}": rng.uniform(0.0, 1.0, n) for i in range(3)}

    @pytest.mark.parametrize("resolution", [1, 4, 20])
    def test_the_grid_has_one_point_per_composition(self, resolution):
        sweep = weighted_sweep(self.table(), ["x0", "x1", "x2"], resolution=resolution)
        assert len(sweep.grid) == math.comb(resolution + 2, 2)
        weights = [w for w, _ in sweep.grid]
        assert weights == sorted(weights)
        assert weights[0] == (0.0, 0.0, 1.0) and weights[-1] == (1.0, 0.0, 0.0)
        for w in weights:
            np.testing.assert_allclose(sum(w), 1.0, rtol=0, atol=1e-15)
        assert sweep.segments == () and sweep.switch_points == ()

    def test_each_grid_point_records_the_argmin(self):
        sweep = weighted_sweep(self.table(1), ["x0", "x1", "x2"], resolution=10)
        values = sweep.normalized.values
        for weights, rows in sweep.grid:
            j = values @ np.array(weights)
            assert rows == tuple(int(r) for r in np.flatnonzero(j <= j.min() + 1e-12))

    def test_the_sweep_is_deterministic(self):
        first = weighted_sweep(self.table(2), ["x0", "x1", "x2"])
        second = weighted_sweep(self.table(2), ["x0", "x1", "x2"])
        assert first.grid == second.grid
        assert first.supported == second.supported
        assert first.unsupported == second.unsupported

    def test_the_support_classes_partition_the_front(self):
        table = self.table(3)
        sweep = weighted_sweep(table, ["x0", "x1", "x2"])
        front = tuple(int(r) for r in np.flatnonzero(nondominated(table, ["x0", "x1", "x2"])))
        together = sweep.supported + sweep.weakly_supported + sweep.unsupported
        assert tuple(sorted(together)) == front
        assert len(set(together)) == len(together)
        values = sweep.normalized.values
        winners = sorted({r for _, rows in sweep.grid for r in rows
                          if (values[list(rows)] == values[rows[0]]).all()})
        assert list(sweep.supported) == winners

    def test_a_row_that_only_ties_at_a_vertex_is_weakly_supported(self):
        """
        Row 0 reaches the minimum only at w = (1, 0, 0), tied with rows 1 and 2 on x = 0;
        at every other weight min(w_y, w_z) < 0.6 (w_y + w_z) and it loses.
        """
        table = {"x": [0.0, 0.0, 0.0, 1.0], "y": [0.6, 0.0, 1.0, 0.0],
                 "z": [0.6, 1.0, 0.0, 0.0]}
        sweep = weighted_sweep(table, ["x", "y", "z"], resolution=20)
        assert sweep.supported == (1, 2, 3)
        assert sweep.weakly_supported == (0,)
        assert sweep.unsupported == ()
        assert [rows for _, rows in sweep.grid if 0 in rows] == [(0, 1, 2)]

    def test_exact_duplicates_win_together(self):
        """Rows 0 and 1 are one point: the minimum of x alone at w = (1, 0, 0) is a win."""
        table = {"x": [0.0, 0.0, 1.0, 1.0], "y": [1.0, 1.0, 0.0, 1.0],
                 "z": [1.0, 1.0, 1.0, 0.0]}
        sweep = weighted_sweep(table, ["x", "y", "z"], resolution=4)
        assert sweep.grid[-1] == ((1.0, 0.0, 0.0), (0, 1))
        assert sweep.supported == (0, 1, 2, 3)
        assert sweep.weakly_supported == ()


class TestEpsilonConstraint:

    TABLE = {"L": [5.0, 1.0, 3.0, 2.0, 3.0, float("nan")],
             "eta": [0.9, 0.2, 0.6, 0.4, 0.6, 1.0]}

    def test_minimize_subject_to_a_floor_on_a_maximized_objective(self):
        assert epsilon_constraint(self.TABLE, LATENCY_ETA, primary="L",
                                  bounds={"eta": 0.5}) == 2
        assert epsilon_constraint(self.TABLE, LATENCY_ETA, primary="L",
                                  bounds={"eta": 0.4}) == 3

    def test_maximize_subject_to_a_ceiling_on_a_minimized_objective(self):
        """Row 5 has the best eta but a NaN L, so it is not scored."""
        assert epsilon_constraint(self.TABLE, LATENCY_ETA, primary="eta",
                                  bounds={"L": 3.0}) == 2
        assert epsilon_constraint(self.TABLE, LATENCY_ETA, primary="eta", bounds={}) == 0

    def test_ties_go_to_the_lowest_row(self):
        """Rows 2 and 4 are identical."""
        assert epsilon_constraint(self.TABLE, LATENCY_ETA, primary="L",
                                  bounds={"eta": 0.6}) == 2

    def test_no_qualifying_row_is_none(self):
        assert epsilon_constraint(self.TABLE, LATENCY_ETA, primary="L",
                                  bounds={"eta": 0.95}) is None

    def test_unknown_names_are_errors(self):
        with pytest.raises(ValueError, match="primary 'V' is not one of the objectives"):
            epsilon_constraint(self.TABLE, LATENCY_ETA, primary="V", bounds={})
        with pytest.raises(ValueError, match="bound on 'V'"):
            epsilon_constraint(self.TABLE, LATENCY_ETA, primary="L", bounds={"V": 1.0})
