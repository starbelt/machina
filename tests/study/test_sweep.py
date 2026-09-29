"""
``machina.study.sweep``: one solve per grid point, routed by role, one row per point.

The fleet is the two-satellite power budget of ``tests/compiler/test_problem.py``,
re-created so this file stands alone:

    maximise  50 (d_a + d_b) - 0.1 (d_a^2 + d_b^2)  s.t.  100 (d_a + d_b) <= limit,  0 <= d <= 1

For ``limit`` in [0, 200] symmetry gives ``d_a = d_b = limit / 200``; with ``d_a``
pinned at ``a`` the other satellite takes ``min(1, limit / 100 - a)``. A negative
``limit`` is infeasible (``d >= 0``), which is how a point is made to fail.

``Tracker`` pulls a quantity of any shape towards ``scale * target``, so a
collected vector or matrix has a closed form element by element. ``Chain`` is
the 30-element parametric NLP of ``tests/solver/test_warm_start.py``, where a
warm start pays off clearly. ``LogWall`` has ``-log(x - wall)`` in its cost:
the previous point's optimum lies outside the next point's domain, so the
warm start fails at its first evaluation and a cold start from the stored
guess succeeds.
"""

import math

import casadi as ca
import numpy as np
import pytest

from machina.astro import SingleSatCoverage
from machina.compiler import Problem
from machina.model import (
    Aggregation,
    Component,
    Constraint,
    Cost,
    Declaration,
    Quantity,
    Role,
    Scope,
    SignalRegistry,
)
from machina.study.sweep import STATUS_COLUMNS, SweepTable, sweep

pytestmark = pytest.mark.requires_casadi

RTOL = 1e-6
ATOL = 1e-6


# --- the fleet ---------------------------------------------------------------------------------


def fleet_registry() -> SignalRegistry:
    reg = SignalRegistry()
    reg.declare("power", 1, "W", aggregation=Aggregation.SUM, doc="Electrical load")
    reg.declare("rate", 1, "1/s", aggregation=Aggregation.SUM, doc="Useful data rate")
    return reg


class Payload(Component):
    """``power = 100 duty``, ``rate = 50 duty``, with a small effort penalty."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("duty", unit="1", lb=0.0, ub=1.0, default=0.1,
                                 doc="Duty cycle", provenance="A", source="test fixture"),),
            produces=("power", "rate"),
            costs=(Cost("effort", weight=0.1, doc="Squared duty cycle"),),
        )

    def build(self, helpers):
        duty = ca.SX.sym("duty")
        return {
            "g": ca.Function("payload_g", [duty], [100.0 * duty, 50.0 * duty],
                             ["duty"], ["power", "rate"]),
            "J": ca.Function("payload_J", [duty], [duty ** 2], ["duty"], ["effort"]),
        }


class FleetBudget(Component):
    """Caps the fleet's total power and scores its total rate."""

    def declare(self):
        return Declaration(
            algebraic=(("power_a", "/a/power"), ("power_b", "/b/power"),
                       ("rate_a", "/a/rate"), ("rate_b", "/b/rate")),
            quantities=(Quantity("limit", unit="W", role=Role.PARAMETER, default=150.0,
                                 doc="Fleet power cap", provenance="D", source="bus spec"),),
            constraints=(Constraint("margin", lb=0.0, doc="limit - total power >= 0"),),
            costs=(Cost("neg_rate", weight=1.0, doc="Negated fleet data rate"),),
        )

    def build(self, helpers):
        power_a, power_b = ca.SX.sym("power_a"), ca.SX.sym("power_b")
        rate_a, rate_b = ca.SX.sym("rate_a"), ca.SX.sym("rate_b")
        limit = ca.SX.sym("limit")
        return {
            "h": ca.Function("budget_h", [power_a, power_b, limit],
                             [limit - (power_a + power_b)],
                             ["power_a", "power_b", "limit"], ["margin"]),
            "J": ca.Function("budget_J", [rate_a, rate_b], [-(rate_a + rate_b)],
                             ["rate_a", "rate_b"], ["neg_rate"]),
        }


def fleet(**compile_kwargs) -> Problem:
    """The fleet, compiled and built."""
    problem = Problem([Scope("a", [Payload()]), Scope("b", [Payload()]), FleetBudget()],
                      registry=fleet_registry(), verbose=False)
    return problem.compile(**compile_kwargs).build()


# --- small models ------------------------------------------------------------------------------


class Tracker(Component):
    """``x`` of ``target``'s shape, pulled towards ``scale * target``; ``scale`` is a parameter."""

    def __init__(self, target, *, name=None):
        super().__init__(name)
        self._target = np.asarray(target, dtype=float)

    def declare(self):
        return Declaration(
            quantities=(Quantity("x", shape=self._target.shape, default=0.0, provenance="A",
                                 source="fixture"),
                        Quantity("scale", role=Role.PARAMETER, default=1.0, provenance="A",
                                 source="fixture")),
            costs=(Cost("miss", weight=2.0, doc="Squared distance to scale * target"),),
        )

    def build(self, helpers):
        x = ca.SX.sym("x", *self._target.shape)
        scale = ca.SX.sym("scale")
        miss = ca.sumsqr(x - scale * ca.DM(self._target))
        return {"J": ca.Function("tracker_J", [x, scale], [miss], ["x", "scale"], ["miss"])}


MATRIX = [[1.0, 2.0], [3.0, 4.0]]
VECTOR = [[1.0], [2.0], [3.0]]


def trackers() -> Problem:
    """A 2x2 tracker in scope ``m`` and a 3x1 one in scope ``v``, built."""
    problem = Problem([Scope("m", [Tracker(MATRIX)]), Scope("v", [Tracker(VECTOR)])],
                      registry=SignalRegistry(), verbose=False)
    return problem.compile().build()


def tracker(target) -> Problem:
    """One tracker, compiled only: a callable target's return value."""
    return Problem([Tracker(target)], registry=SignalRegistry(), verbose=False).compile()


class Pull(Component):
    """``(x - target)^2`` with ``target`` baked in: a structural parameter."""

    def __init__(self, target, *, name=None):
        super().__init__(name)
        self._target = float(target)

    def declare(self):
        return Declaration(
            quantities=(Quantity("x", default=0.0, provenance="A", source="fixture"),),
            costs=(Cost("pull"),),
        )

    def build(self, helpers):
        x = ca.SX.sym("x")
        return {"J": ca.Function("pull_J", [x], [(x - self._target) ** 2], ["x"], ["pull"])}


def pulled(target) -> Problem:
    return Problem([Pull(target)], registry=SignalRegistry(), verbose=False).compile()


N = 30


class Chain(Component):
    """``tests/solver/test_warm_start.py::chain`` as a component."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("x", shape=(N, 1), lb=-2.0, ub=0.9, default=0.0,
                                 provenance="A", source="fixture"),
                        Quantity("p", role=Role.PARAMETER, default=1.0, provenance="A",
                                 source="fixture")),
            constraints=(Constraint("sum", lb=-5.0, ub=5.0),),
            costs=(Cost("chain"),),
        )

    def build(self, helpers):
        x, p = ca.SX.sym("x", N), ca.SX.sym("p")
        cost = (ca.sumsqr(x - p) + ca.sum1(ca.exp(0.1 * x))
                + 10 * ca.sumsqr(x[1:] - x[:-1] ** 2))
        return {"J": ca.Function("chain_J", [x, p], [cost], ["x", "p"], ["chain"]),
                "h": ca.Function("chain_h", [x], [ca.sum1(x)], ["x"], ["sum"])}


class LogWall(Component):
    """``(x - 1)^2 - log(x - wall)``: defined only right of the wall."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("x", default=10.0, provenance="A", source="fixture"),
                        Quantity("wall", role=Role.PARAMETER, default=0.0, provenance="A",
                                 source="fixture")),
            costs=(Cost("barrier"),),
        )

    def build(self, helpers):
        x, wall = ca.SX.sym("x"), ca.SX.sym("wall")
        return {"J": ca.Function("wall_J", [x, wall], [(x - 1) ** 2 - ca.log(x - wall)],
                                 ["x", "wall"], ["barrier"])}


def single(component) -> Problem:
    return Problem([component], registry=SignalRegistry(), verbose=False).compile().build()


def column(table: SweepTable, name: str) -> list:
    return [row[name] for row in table.rows]


# --- tests -------------------------------------------------------------------------------------


class TestTheGrid:

    def test_the_grid_is_cartesian_with_the_last_axis_fastest(self):
        table = sweep(trackers(), {"m/scale": [1.0, 2.0], "v/scale": [10.0, 20.0, 30.0]})
        points = [(row["m/scale"], row["v/scale"]) for row in table.rows]
        assert points == [(1.0, 10.0), (1.0, 20.0), (1.0, 30.0),
                          (2.0, 10.0), (2.0, 20.0), (2.0, 30.0)]
        assert table.axes == ("m/scale", "v/scale")

    def test_a_list_of_pairs_keeps_its_order(self):
        table = sweep(trackers(), [("v/scale", [1.0, 2.0]), ("m/scale", [3.0])])
        assert table.axes == ("v/scale", "m/scale")
        assert [(row["v/scale"], row["m/scale"]) for row in table.rows] == [(1.0, 3.0),
                                                                            (2.0, 3.0)]

    def test_columns_are_axes_then_status_then_collected(self):
        table = sweep(fleet(), {"limit": [150.0]}, collect=["a/duty", "cost:neg_rate"])
        assert table.columns == ("limit",) + STATUS_COLUMNS + ("a/duty", "cost:neg_rate")
        assert "t_wall" not in table.columns
        assert list(table.rows[0]) == list(table.columns) + ["t_wall"]

    def test_numpy_axis_values_become_python_scalars(self):
        table = sweep(fleet(), {"limit": np.array([140.0, 150.0])})
        assert [type(value) for value in column(table, "limit")] == [float, float]
        assert column(table, "limit") == [140.0, 150.0]

    def test_a_range_axis_keeps_its_ints(self):
        table = sweep(fleet(), {"limit": range(140, 160, 10)})
        assert column(table, "limit") == [140, 150]
        assert all(type(value) is int for value in column(table, "limit"))


class TestAxesAreChecked:

    def test_a_set_axis_is_refused_by_name(self):
        with pytest.raises(TypeError, match="'limit'") as err:
            sweep(fleet(), {"limit": {140.0, 150.0}})
        assert "no order" in str(err.value)

    def test_a_set_of_axes_is_refused(self):
        with pytest.raises(TypeError, match="no order"):
            sweep(fleet(), {("limit", (150.0,))})

    def test_an_empty_axis_is_refused_by_name(self):
        with pytest.raises(ValueError, match="'limit' has no values"):
            sweep(fleet(), {"limit": []})

    def test_no_axes_at_all_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            sweep(fleet(), {})

    def test_a_bare_number_is_not_an_axis(self):
        with pytest.raises(TypeError, match="ordered sequence"):
            sweep(fleet(), {"limit": 150.0})

    def test_an_axis_given_twice_is_refused(self):
        with pytest.raises(ValueError, match="twice"):
            sweep(fleet(), [("limit", [140.0]), ("limit", [150.0])])

    def test_a_value_that_is_not_a_scalar_is_refused(self):
        with pytest.raises(TypeError, match="not a scalar"):
            sweep(fleet(), {"limit": [[140.0, 150.0]]})

    def test_a_string_value_is_refused_on_a_problem_target(self):
        with pytest.raises(TypeError, match="not a number"):
            sweep(fleet(), {"limit": ["high"]})


class TestRoutingOnAProblem:

    def test_a_parameter_axis_reaches_each_solve(self):
        table = sweep(fleet(), {"limit": [100.0, 150.0]}, collect=["a/duty"])
        np.testing.assert_allclose(column(table, "a/duty"), [0.5, 0.75], rtol=RTOL)

    def test_a_parameter_axis_is_not_persisted(self):
        problem = fleet()
        sweep(problem, {"limit": [100.0, 120.0]})
        np.testing.assert_allclose(problem.backend.parameter_value("limit"), [150.0], rtol=0.0)
        np.testing.assert_allclose(problem.solve()["a/duty"], [0.75], rtol=RTOL)

    def test_a_variable_axis_pins_the_variable(self):
        table = sweep(fleet(), {"a/duty": [0.25, 0.5]}, collect=["b/duty"], keep_results=True)
        assert all(column(table, "success"))
        assert column(table, "warm_started") == [False, True]
        np.testing.assert_allclose([r["a/duty"][0] for r in table.results], [0.25, 0.5],
                                   rtol=1e-12)
        np.testing.assert_allclose(column(table, "b/duty"), [1.0, 1.0], rtol=RTOL)

    def test_a_variable_axis_leaves_bounds_and_initial_guess_as_found(self):
        problem = fleet()
        problem.set_bounds("a/duty", ub=0.9)                 # not the declared bounds
        problem.backend.set_initial_guess("a/duty", 0.3)     # nor the declared guess
        sweep(problem, {"a/duty": [0.25, 0.5]})
        lb, ub = problem.backend.bounds("a/duty")
        np.testing.assert_allclose(lb, [0.0], rtol=0.0)
        np.testing.assert_allclose(ub, [0.9], rtol=0.0)
        np.testing.assert_allclose(problem.backend.initial_guess("a/duty"), [0.3], rtol=0.0)

    def test_a_variable_axis_is_restored_after_an_exception_mid_sweep(self):
        problem = fleet()
        with pytest.raises(ValueError, match="NaN"):
            sweep(problem, {"a/duty": [0.25], "limit": [150.0, math.nan]}, raise_on_error=True)
        lb, ub = problem.backend.bounds("a/duty")
        np.testing.assert_allclose(lb, [0.0], rtol=0.0)
        np.testing.assert_allclose(ub, [1.0], rtol=0.0)
        np.testing.assert_allclose(problem.backend.initial_guess("a/duty"), [0.1], rtol=0.0)

    def test_a_discrete_axis_is_pinned_like_a_variable(self):
        problem = Problem([Pull(2.6)], registry=SignalRegistry(), verbose=False)
        problem.compile(roles={"x": Role.DISCRETE}, discrete_mode="relax").build()
        table = sweep(problem, {"x": [2.0, 3.0]}, collect=["cost:pull"])
        np.testing.assert_allclose(column(table, "cost:pull"), [0.36, 0.16], rtol=RTOL)
        np.testing.assert_allclose(problem.backend.bounds("x")[1], [math.inf])

    def test_a_fixed_axis_is_refused_with_the_way_to_sweep_it(self):
        problem = fleet(roles={"limit": Role.FIXED})
        with pytest.raises(ValueError, match="'fixed'") as err:
            sweep(problem, {"limit": [100.0]})
        assert "compile(roles={'limit': 'parameter'})" in str(err.value)
        assert "callable target" in str(err.value)

    def test_an_unknown_path_lists_the_free_paths(self):
        with pytest.raises(ValueError, match="'a/powr'") as err:
            sweep(fleet(), {"a/powr": [1.0]})
        for path in ("'a/duty'", "'b/duty'", "'limit'"):
            assert path in str(err.value)

    def test_a_signal_is_not_an_axis(self):
        with pytest.raises(ValueError, match="not a quantity"):
            sweep(fleet(), {"a/power": [50.0]})

    def test_an_unbuilt_problem_is_refused_naming_build(self):
        problem = Problem([Pull(1.0)], registry=SignalRegistry(), verbose=False).compile()
        with pytest.raises(RuntimeError, match=r"problem\.build\(\)"):
            sweep(problem, {"x": [1.0]})

    def test_an_uncompiled_problem_is_refused_naming_compile(self):
        problem = Problem([Pull(1.0)], registry=SignalRegistry(), verbose=False)
        with pytest.raises(RuntimeError, match=r"compile\("):
            sweep(problem, {"x": [1.0]})

    def test_a_target_that_is_neither_is_refused(self):
        with pytest.raises(TypeError, match="built Problem or a callable"):
            sweep(42, {"x": [1.0]})


class TestWarmStarts:

    def test_a_point_after_a_success_is_warm_and_after_a_failure_is_cold(self):
        table = sweep(fleet(), {"limit": [150.0, -10.0, 150.0, 152.0]}, retry_cold=False)
        assert column(table, "success") == [True, False, True, True]
        assert column(table, "warm_started") == [False, True, False, True]
        assert column(table, "retried") == [False, False, False, False]
        assert table.rows[1]["status"] == "Infeasible_Problem_Detected"

    def test_warm_start_false_starts_every_point_cold(self):
        table = sweep(fleet(), {"limit": [150.0, 152.0]}, warm_start=False)
        assert column(table, "warm_started") == [False, False]

    def test_a_failed_warm_start_is_retried_cold(self):
        table = sweep(single(LogWall()), {"wall": [0.0, 5.0]}, collect=["x"])
        row = table.rows[1]
        assert (row["success"], row["warm_started"], row["retried"]) == (True, False, True)
        # (x - 1)^2 - log(x - 5): 2 (x - 1)(x - 5) = 1, x = 3 + sqrt(4.5)
        np.testing.assert_allclose(row["x"], 3.0 + math.sqrt(4.5), rtol=RTOL)

    def test_without_retry_the_failed_warm_start_is_recorded(self):
        table = sweep(single(LogWall()), {"wall": [0.0, 5.0]}, retry_cold=False)
        row = table.rows[1]
        assert (row["success"], row["warm_started"], row["retried"]) == (False, True, False)
        assert row["status"] == "Invalid_Number_Detected"
        assert math.isnan(row["objective"])

    def test_a_retry_that_fails_too_keeps_the_retrys_outcome(self):
        table = sweep(fleet(), {"limit": [150.0, -10.0]})
        row = table.rows[1]
        assert (row["success"], row["warm_started"], row["retried"]) == (False, False, True)
        assert row["status"] == "Infeasible_Problem_Detected"

    def test_warm_starts_cut_the_iterations_of_a_smooth_parameter_sweep(self):
        axes = {"p": [1.0, 1.05, 1.10, 1.20, 1.50]}
        warm = sweep(single(Chain()), axes)
        cold = sweep(single(Chain()), axes, warm_start=False)
        assert all(column(warm, "success")) and all(column(cold, "success"))
        assert column(warm, "warm_started") == [False, True, True, True, True]
        assert sum(column(warm, "iterations")) < sum(column(cold, "iterations"))
        np.testing.assert_allclose(column(warm, "objective"), column(cold, "objective"),
                                   rtol=1e-8)


class TestFailures:

    def test_an_exception_is_recorded_and_the_sweep_goes_on(self):
        table = sweep(fleet(), {"limit": [150.0, math.nan, 140.0]}, collect=["a/duty"])
        failed = table.rows[1]
        assert (failed["success"], failed["status"], failed["iterations"]) == (False, "error",
                                                                               None)
        assert math.isnan(failed["objective"]) and math.isnan(failed["a/duty"])
        assert failed["error"].startswith("ValueError: ") and "NaN" in failed["error"]
        assert failed["t_wall"] is None
        assert table.rows[2]["success"] and not table.rows[2]["warm_started"]
        assert [row["error"] for row in table.rows][::2] == [None, None]

    def test_raise_on_error_re_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            sweep(fleet(), {"limit": [150.0, math.nan]}, raise_on_error=True)

    def test_an_unconverged_point_has_nan_objective_and_collected_values(self):
        table = sweep(fleet(), {"limit": [-10.0]}, collect=["a/duty", "cost:neg_rate"])
        row = table.rows[0]
        assert not row["success"] and row["error"] is None
        assert row["iterations"] > 0
        assert all(math.isnan(row[name]) for name in ("objective", "a/duty", "cost:neg_rate"))

    def test_only_the_first_line_of_a_message_is_kept(self):
        def target(x):
            if x < 0:
                raise ValueError("negative target\rsecond part\nthird part")
            return pulled(x)
        table = sweep(target, {"x": [-1.0, 1.0]})
        assert table.rows[0]["error"] == "ValueError: negative target"
        assert table.rows[1]["success"]


class TestCollect:

    def test_a_scalar_keeps_its_name(self):
        table = sweep(fleet(), {"limit": [150.0]}, collect=["a/power"])
        np.testing.assert_allclose(table.rows[0]["a/power"], 75.0, rtol=RTOL)

    def test_a_vector_and_a_matrix_expand_column_major(self):
        table = sweep(trackers(), {"m/scale": [2.0], "v/scale": [10.0]},
                      collect=["m/x", "v/x"])
        names = [f"m/x[{i}]" for i in range(4)] + [f"v/x[{i}]" for i in range(3)]
        assert table.columns[-7:] == tuple(names)
        row = table.rows[0]
        np.testing.assert_allclose([row[name] for name in names[:4]], [2.0, 6.0, 4.0, 8.0],
                                   rtol=RTOL)
        np.testing.assert_allclose([row[name] for name in names[4:]], [10.0, 20.0, 30.0],
                                   rtol=RTOL)

    def test_a_cost_term_is_collected_weighted(self):
        table = sweep(fleet(), {"limit": [150.0]}, collect=["cost:a/effort", "cost:neg_rate"])
        row = table.rows[0]
        np.testing.assert_allclose(row["cost:a/effort"], 0.1 * 0.75 ** 2, rtol=RTOL)
        np.testing.assert_allclose(row["cost:neg_rate"], -75.0, rtol=RTOL)
        np.testing.assert_allclose(row["cost:a/effort"] * 2 + row["cost:neg_rate"],
                                   row["objective"], rtol=1e-9)

    def test_a_bad_path_is_refused_before_any_solve(self):
        problem = fleet()
        with pytest.raises(ValueError, match="'a/powr'") as err:
            sweep(problem, {"limit": [150.0]}, collect=["a/powr"])
        assert "'a/power'" in str(err.value)
        assert not problem.backend.is_solved

    def test_an_unknown_cost_term_lists_the_cost_terms(self):
        problem = fleet()
        with pytest.raises(ValueError, match="'cost:rate'") as err:
            sweep(problem, {"limit": [150.0]}, collect=["cost:rate"])
        assert "'cost:neg_rate'" in str(err.value)
        assert not problem.backend.is_solved

    def test_a_bad_entry_on_a_callable_target_is_refused_before_any_solve(self):
        made = []

        def target(x):
            made.append(pulled(x))
            return made[-1]
        with pytest.raises(ValueError, match="'nope'"):
            sweep(target, {"x": [1.0, 2.0]}, collect=["nope"])
        assert len(made) == 1 and not made[0].backend.is_solved

    def test_a_string_is_not_a_collect_sequence(self):
        with pytest.raises(TypeError, match=r"collect=\('a/power',\)"):
            sweep(fleet(), {"limit": [150.0]}, collect="a/power")

    def test_an_entry_named_like_an_axis_is_refused(self):
        with pytest.raises(ValueError, match="axis"):
            sweep(fleet(), {"limit": [150.0]}, collect=["limit"])


class TestCallableTargets:

    def test_the_target_is_rebuilt_per_point(self):
        made = []

        def target(x):
            made.append(pulled(x))
            return made[-1]
        table = sweep(target, {"x": [1.0, 3.0]}, collect=["cost:pull"])
        assert len(made) == 2 and made[0] is not made[1]
        assert all(problem.is_built for problem in made)
        # The axis is a keyword of the target; the Problem's own x is the optimum.
        np.testing.assert_allclose(column(table, "cost:pull"), [0.0, 0.0], atol=1e-12)
        assert column(table, "warm_started") == [False, True]

    def test_a_structural_axis_changes_the_optimum(self):
        table = sweep(lambda n: tracker(np.arange(1.0, n + 1.0).reshape(n, 1)),
                      {"n": [2, 3]}, keep_results=True)
        assert all(column(table, "success"))
        np.testing.assert_allclose(table.results[0]["x"], [1.0, 2.0], rtol=RTOL)
        np.testing.assert_allclose(table.results[1]["x"], [1.0, 2.0, 3.0], rtol=RTOL)
        assert column(table, "warm_started") == [False, True]   # seeded; nothing matched

    def test_a_variable_that_changed_shape_starts_cold(self):
        table = sweep(lambda rows: tracker(np.ones((rows, 6 // rows))), {"rows": [2, 3]})
        assert column(table, "success") == [True, True]
        assert column(table, "warm_started") == [False, False]

    def test_collected_shapes_are_fixed_by_the_first_problem(self):
        table = sweep(lambda n: tracker(np.ones((n, 1))), {"n": [2, 3]}, collect=["x"])
        assert table.columns[-2:] == ("x[0]", "x[1]")
        assert table.rows[1]["status"] == "error"
        assert "fixed the columns" in table.rows[1]["error"]

    def test_an_axis_name_must_be_an_identifier(self):
        with pytest.raises(ValueError, match="identifier"):
            sweep(pulled, {"sat/p": [1.0]})

    def test_a_target_that_returns_something_else_is_refused(self):
        with pytest.raises(TypeError, match="not a Problem"):
            sweep(lambda x: 42, {"x": [1.0]})

    def test_a_target_that_returns_an_uncompiled_problem_is_refused(self):
        with pytest.raises(RuntimeError, match="not compiled"):
            sweep(lambda x: Problem([Pull(x)], registry=SignalRegistry(), verbose=False),
                  {"x": [1.0]})

    def test_string_values_reach_the_target(self):
        targets = {"near": 1.0, "far": 5.0}
        table = sweep(lambda where: pulled(targets[where]), {"where": ["near", "far"]},
                      keep_results=True)
        assert column(table, "where") == ["near", "far"]
        np.testing.assert_allclose([r["x"][0] for r in table.results], [1.0, 5.0], rtol=RTOL)


class TestResults:

    def test_results_are_kept_on_request(self):
        table = sweep(fleet(), {"limit": [150.0, math.nan, 140.0]}, keep_results=True)
        assert len(table.results) == len(table.rows) == 3
        assert table.results[1] is None
        for row, result in zip(table.rows[::2], table.results[::2]):
            assert row["objective"] == result.f_opt
            assert row["iterations"] == result.iterations
            assert row["t_wall"] == result.t_wall

    def test_results_are_dropped_by_default(self):
        assert sweep(fleet(), {"limit": [150.0]}).results is None


class TestARealisticModel:

    R_EARTH = 6378.137

    def coverage(self) -> Problem:
        component = SingleSatCoverage(target_lat_deg=38.9, target_lon_deg=-77.0,
                                      n_sample_points=24, min_elevation_deg=10.0,
                                      sigmoid_k=20.0, perigee_min_km=200.0, apogee_max_km=1600.0,
                                      mu=398600.4418, R_earth=self.R_EARTH)
        tan_half_i = math.tan(math.radians(51.6) / 2.0)
        raan = math.radians(240.0)
        start = {"p": self.R_EARTH + 500.0, "f": 0.01, "g": 0.0,
                 "h": tan_half_i * math.cos(raan), "k": tan_half_i * math.sin(raan)}
        box = (("p", self.R_EARTH + 200.0, self.R_EARTH + 1600.0), ("f", -0.30, 0.30),
               ("g", -0.30, 0.30), ("h", -1.5, 1.5), ("k", -1.5, 1.5))
        overrides = {f"sat/{name}": {"x0": start[name], "lb": lb, "ub": ub}
                     for name, lb, ub in box}
        problem = Problem([Scope("sat", [component])], verbose=False).compile(
            overrides=overrides)
        problem.add_cost(-problem.expr("sat/coverage_total").symbol, name="neg_coverage")
        return problem.build()

    def test_pinning_the_scaled_semi_latus_rectum_sweeps_the_coverage(self):
        problem = self.coverage()
        altitudes = [self.R_EARTH + 1000.0, self.R_EARTH + 1500.0]
        table = sweep(problem, {"sat/p": altitudes},
                      collect=["sat/coverage_total", "cost:neg_coverage"], keep_results=True)
        assert all(column(table, "success"))
        assert column(table, "warm_started") == [False, True]
        np.testing.assert_allclose([r["sat/p"][0] for r in table.results], altitudes, rtol=1e-12)
        coverage = column(table, "sat/coverage_total")
        assert all(0.0 < value < 1.0 for value in coverage)
        np.testing.assert_allclose(column(table, "cost:neg_coverage"), [-c for c in coverage],
                                   rtol=1e-12)
        lb, ub = problem.backend.bounds("sat/p")
        np.testing.assert_allclose([lb[0], ub[0]],
                                   [self.R_EARTH + 200.0, self.R_EARTH + 1600.0], rtol=0.0)
