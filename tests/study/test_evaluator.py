"""
``machina.study.Evaluator``: a model's signal graph at given leaf values, without a solve.

The fleet is the two-satellite power budget of ``tests/compiler/test_problem.py``,
re-created so this file stands alone:

    power = 100 duty,  rate = 50 duty   (per satellite)
    maximise  50 (d_a + d_b) - 0.1 (d_a^2 + d_b^2)  s.t.  100 (d_a + d_b) <= limit

At ``limit = 150`` the optimum is ``d_a = d_b = 0.75``. The oracle for the
``from_problem`` path is ``Problem.evaluate`` itself: fed a result's variables
and parameters, the Evaluator must return the same paths, in the same order,
with the same shapes and the same numbers. The builder path is checked against
closed forms on paper, and against ``Problem.evaluate`` too, through
``problem.builder``.
"""

import math

import casadi as ca
import numpy as np
import pytest

from machina.astro import SingleSatCoverage
from machina.compiler import Problem
from machina.model import (
    Aggregation,
    Builder,
    Component,
    Constraint,
    Cost,
    Declaration,
    Quantity,
    Role,
    Scope,
    SignalRegistry,
)
from machina.study.evaluate import Evaluator

pytestmark = pytest.mark.requires_casadi

RTOL = 1e-12
ATOL = 1e-12


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


def fleet_components() -> list:
    return [Scope("a", [Payload()]), Scope("b", [Payload()]), FleetBudget()]


def fleet_builder() -> Builder:
    return Builder(fleet_components(), registry=fleet_registry()).declare()


def fleet(**compile_kwargs) -> Problem:
    return Problem(fleet_components(), registry=fleet_registry(),
                   verbose=False).compile(**compile_kwargs)


def solved_fleet(**compile_kwargs):
    problem = fleet(**compile_kwargs).build()
    return problem, problem.solve()


# --- small static, matrix and dynamic models ---------------------------------------------------


class Mixer(Component):
    """``total = a + 2 b``; ``a`` has a declared default, ``b`` has none."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("a", default=1.0, provenance="A", source="fixture"),
                        Quantity("b")),
            produces=("total",),
        )

    def build(self, helpers):
        a, b = ca.SX.sym("a"), ca.SX.sym("b")
        return {"g": ca.Function("mixer_g", [a, b], [a + 2.0 * b], ["a", "b"], ["total"])}


def mixer_builder() -> Builder:
    reg = SignalRegistry()
    reg.declare("total", 1, "1", doc="a + 2 b")
    return Builder([Mixer()], registry=reg).declare()


TARGET = np.array([[1.0, 2.0], [3.0, 4.0]])


class Grid(Component):
    """A (2, 2) quantity ``m``, ``twice = 2 m``, ``trace = tr m``, pulled towards TARGET."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("m", shape=(2, 2), default=1.0, provenance="A",
                                 source="fixture"),),
            produces=("twice", "trace"),
            costs=(Cost("miss", doc="Squared distance to TARGET"),),
        )

    def build(self, helpers):
        m = ca.SX.sym("m", 2, 2)
        return {
            "g": ca.Function("grid_g", [m], [2.0 * m, ca.trace(m)], ["m"], ["twice", "trace"]),
            "J": ca.Function("grid_J", [m], [ca.sumsqr(m - ca.DM(TARGET))], ["m"], ["miss"]),
        }


def grid_registry() -> SignalRegistry:
    reg = SignalRegistry()
    reg.declare("twice", (2, 2), "1", doc="2 m")
    reg.declare("trace", 1, "1", doc="Trace of m")
    return reg


def grid_builder() -> Builder:
    return Builder([Grid()], registry=grid_registry()).declare()


class Plant(Component):
    """``x_dot = -gain x + u`` and ``y = gain x + u``: a state, an input and a quantity."""

    def declare(self):
        return Declaration(
            states=("x",), inputs=("u",),
            quantities=(Quantity("gain", unit="1/s", default=2.0, provenance="A",
                                 source="fixture"),),
            derivatives=("x",), produces=("y",),
        )

    def build(self, helpers):
        x, u, gain = ca.SX.sym("x"), ca.SX.sym("u"), ca.SX.sym("gain")
        return {
            "f": ca.Function("plant_f", [x, u, gain], [-gain * x + u],
                             ["x", "u", "gain"], ["x_dot"]),
            "g": ca.Function("plant_g", [x, u, gain], [gain * x + u],
                             ["x", "u", "gain"], ["y"]),
        }


def plant_builder() -> Builder:
    reg = SignalRegistry()
    reg.declare("x", 1, "m", doc="Plant state")
    reg.declare("u", 1, "m/s", doc="Plant input")
    reg.declare("y", 1, "m/s", doc="Plant output")
    return Builder([Plant()], registry=reg).declare()


# --- the coverage problem (vector leaves), as tests/astro/test_single_sat_coverage.py ---------

R_EARTH = 6378.137


def coverage_problem() -> Problem:
    """Maximise coverage of Washington DC from an ISS-like start; converges in ~21 iterations."""
    component = SingleSatCoverage(target_lat_deg=38.9, target_lon_deg=-77.0, n_sample_points=24,
                                  min_elevation_deg=10.0, sigmoid_k=20.0, perigee_min_km=200.0,
                                  apogee_max_km=1600.0, mu=398600.4418, R_earth=R_EARTH)
    tan_half_i = math.tan(math.radians(51.6) / 2.0)
    raan = math.radians(240.0)
    start = {"p": R_EARTH + 500.0, "f": 0.01, "g": 0.0,
             "h": tan_half_i * math.cos(raan), "k": tan_half_i * math.sin(raan)}
    box = (("p", R_EARTH + 200.0, R_EARTH + 1600.0), ("f", -0.30, 0.30), ("g", -0.30, 0.30),
           ("h", -1.5, 1.5), ("k", -1.5, 1.5))
    overrides = {f"sat/{name}": {"x0": start[name], "lb": lb, "ub": ub} for name, lb, ub in box}
    problem = Problem([Scope("sat", [component])], verbose=False).compile(overrides=overrides)
    problem.add_cost(-problem.expr("sat/coverage_total").symbol, name="neg_coverage")
    return problem.build()


# --- helpers -----------------------------------------------------------------------------------


def at_result(result) -> dict:
    """Every leaf of a ``from_problem`` Evaluator, at a solve result."""
    return {**result.x_opt, **result.p_opt}


def assert_same_values(got: dict, expected: dict) -> None:
    """Same paths in the same order, same shapes, same numbers."""
    assert list(got) == list(expected)
    for path, value in expected.items():
        assert got[path].shape == value.shape, path
        np.testing.assert_allclose(got[path], value, rtol=RTOL, atol=ATOL, err_msg=path)


# --- tests -------------------------------------------------------------------------------------


class TestFromProblemReproducesEvaluate:

    def test_every_fleet_output_matches_evaluate_at_the_optimum(self):
        problem, result = solved_fleet()
        assert_same_values(Evaluator.from_problem(problem)(at_result(result)),
                           problem.evaluate(result))

    def test_every_coverage_output_matches_evaluate_at_the_optimum(self):
        problem = coverage_problem()
        result = problem.solve()
        assert result.success
        values = Evaluator.from_problem(problem)(at_result(result))
        assert_same_values(values, problem.evaluate(result))
        assert values["sat/L"].shape == (24,) and values["sat/r_target"].shape == (3,)

    def test_the_outputs_are_quantities_then_algebraic_signals(self):
        problem, result = solved_fleet()
        evaluator = Evaluator.from_problem(problem)
        assert list(evaluator.outputs) == list(problem.evaluate(result))
        assert list(evaluator.outputs) == \
            problem.builder.quantity_order + problem.builder.algebraic_order

    def test_the_leaves_are_the_variables_then_the_parameters(self):
        evaluator = Evaluator.from_problem(fleet())
        assert evaluator.leaves == ("a/duty", "b/duty", "limit")
        function = evaluator.function
        assert [function.name_in(i) for i in range(function.n_in())] == list(evaluator.leaves)
        assert function.n_out() == len(evaluator.outputs)

    def test_a_compiled_problem_needs_no_build_or_solve(self):
        problem = fleet()
        values = Evaluator.from_problem(problem)(**{"a/duty": 0.5})
        np.testing.assert_allclose(values["a/power"], [50.0], rtol=RTOL)
        assert not problem.is_built

    def test_an_uncompiled_problem_says_to_compile(self):
        problem = Problem(fleet_components(), registry=fleet_registry(), verbose=False)
        with pytest.raises(RuntimeError, match=r"problem\.compile\("):
            Evaluator.from_problem(problem)

    def test_a_builder_given_to_from_problem_points_to_the_constructor(self):
        with pytest.raises(TypeError, match=r"Evaluator\(builder\)"):
            Evaluator.from_problem(fleet_builder())


class TestFromProblemDefaults:

    def test_defaults_are_the_initial_guesses_and_the_parameter_values(self):
        values = Evaluator.from_problem(fleet())()
        np.testing.assert_allclose(values["a/duty"], [0.1], rtol=RTOL)
        np.testing.assert_allclose(values["limit"], [150.0], rtol=RTOL)
        np.testing.assert_allclose(values["b/power"], [10.0], rtol=RTOL)

    def test_compile_overrides_reach_the_defaults(self):
        problem = fleet(values={"limit": 120.0}, overrides={"a/duty": {"x0": 0.3}})
        values = Evaluator.from_problem(problem)()
        np.testing.assert_allclose(values["a/duty"], [0.3], rtol=RTOL)
        np.testing.assert_allclose(values["b/duty"], [0.1], rtol=RTOL)
        np.testing.assert_allclose(values["limit"], [120.0], rtol=RTOL)

    def test_defaults_are_read_when_the_evaluator_is_made(self):
        problem = fleet()
        before = Evaluator.from_problem(problem)
        problem.set_value("limit", 200.0)
        np.testing.assert_allclose(before()["limit"], [150.0], rtol=RTOL)
        np.testing.assert_allclose(Evaluator.from_problem(problem)()["limit"], [200.0],
                                   rtol=RTOL)


class TestFixedQuantities:

    def test_a_fixed_problem_quantity_is_an_output_and_not_a_leaf(self):
        problem = fleet(roles={"a/duty": Role.FIXED}, overrides={"a/duty": {"value": 0.5}})
        evaluator = Evaluator.from_problem(problem)
        assert evaluator.leaves == ("b/duty", "limit")
        values = evaluator()
        np.testing.assert_allclose(values["a/duty"], [0.5], rtol=RTOL)
        np.testing.assert_allclose(values["a/power"], [50.0], rtol=RTOL)
        np.testing.assert_allclose(evaluator.fixed["a/duty"], [0.5], rtol=RTOL)

    def test_a_fixed_problem_quantity_matches_evaluate_at_the_optimum(self):
        problem = fleet(roles={"a/duty": Role.FIXED},
                        overrides={"a/duty": {"value": 0.5}}).build()
        result = problem.solve()
        assert_same_values(Evaluator.from_problem(problem)(at_result(result)),
                           problem.evaluate(result))

    def test_varying_a_fixed_problem_quantity_points_to_the_builder_route(self):
        problem = fleet(roles={"a/duty": Role.FIXED}, overrides={"a/duty": {"value": 0.5}})
        with pytest.raises(ValueError, match="FIXED in this Problem") as err:
            Evaluator.from_problem(problem)(**{"a/duty": 0.25})
        assert "Evaluator(problem.builder, fixed=" in str(err.value)

    def test_a_builder_fixed_quantity_is_folded_in(self):
        evaluator = Evaluator(fleet_builder(), fixed={"limit": 100.0})
        assert evaluator.leaves == ("a/duty", "b/duty")
        assert "limit" in evaluator.outputs
        values = evaluator(**{"a/duty": 0.5})
        np.testing.assert_allclose(values["limit"], [100.0], rtol=RTOL)
        np.testing.assert_allclose(values["a/power"], [50.0], rtol=RTOL)

    def test_varying_a_builder_fixed_quantity_is_refused(self):
        evaluator = Evaluator(fleet_builder(), fixed={"limit": 100.0})
        with pytest.raises(ValueError, match="in fixed= of this Evaluator"):
            evaluator(limit=120.0)

    def test_an_unknown_fixed_path_lists_the_quantities(self):
        with pytest.raises(ValueError, match="limt") as err:
            Evaluator(fleet_builder(), fixed={"limt": 100.0})
        assert "'limit'" in str(err.value) and "'a/duty'" in str(err.value)

    def test_a_state_cannot_be_fixed(self):
        with pytest.raises(ValueError, match="only quantities can be fixed"):
            Evaluator(plant_builder(), fixed={"x": 1.0})

    def test_a_scalar_is_not_broadcast_into_a_fixed_matrix(self):
        with pytest.raises(ValueError, match="nothing is broadcast"):
            Evaluator(grid_builder(), fixed={"m": 1.0})

    def test_a_fixed_value_must_be_finite(self):
        with pytest.raises(ValueError, match="not finite"):
            Evaluator(fleet_builder(), fixed={"limit": math.inf})

    def test_a_flat_fixed_matrix_is_read_column_major(self):
        values = Evaluator(grid_builder(), fixed={"m": [1.0, 2.0, 3.0, 4.0]})()
        np.testing.assert_allclose(values["m"], [[1.0, 3.0], [2.0, 4.0]], rtol=RTOL)
        np.testing.assert_allclose(values["trace"], [5.0], rtol=RTOL)

    def test_a_path_in_both_fixed_and_values_is_refused(self):
        with pytest.raises(ValueError, match="also in fixed="):
            Evaluator(fleet_builder(), fixed={"limit": 100.0}, values={"limit": 120.0})


class TestTheBuilderPath:

    def test_it_matches_the_closed_form_without_building_anything(self):
        values = Evaluator(mixer_builder())(a=1.0, b=2.0)
        np.testing.assert_allclose(values["total"], [5.0], rtol=RTOL)
        assert list(values) == ["a", "b", "total"]

    def test_it_matches_problem_evaluate_through_the_problem_builder(self):
        problem, result = solved_fleet()
        evaluator = Evaluator(problem.builder)
        assert_same_values(evaluator(at_result(result)), problem.evaluate(result))

    def test_values_are_defaults_and_the_declared_default_is_the_fallback(self):
        evaluator = Evaluator(mixer_builder(), values={"b": 3.0})
        np.testing.assert_allclose(evaluator()["total"], [7.0], rtol=RTOL)       # a = 1 declared
        np.testing.assert_allclose(evaluator(b=0.0)["total"], [1.0], rtol=RTOL)

    def test_a_none_falls_through_to_the_default(self):
        evaluator = Evaluator(mixer_builder(), values={"a": None, "b": 3.0})
        np.testing.assert_allclose(evaluator(b=None)["total"], [7.0], rtol=RTOL)

    def test_a_values_key_that_is_not_a_leaf_lists_the_leaves(self):
        with pytest.raises(ValueError, match="'c'") as err:
            Evaluator(mixer_builder(), values={"c": 1.0})
        assert "['a', 'b']" in str(err.value)

    def test_an_undeclared_builder_says_to_declare(self):
        reg = SignalRegistry()
        reg.declare("total", 1, "1", doc="a + 2 b")
        with pytest.raises(RuntimeError, match=r"builder\.declare\(\)"):
            Evaluator(Builder([Mixer()], registry=reg))

    def test_a_problem_given_as_a_builder_points_to_from_problem(self):
        with pytest.raises(TypeError, match=r"Evaluator\.from_problem"):
            Evaluator(fleet())


class TestLeavesByPath:

    def test_a_slash_path_in_a_mapping(self):
        values = Evaluator(fleet_builder())({"a/duty": 0.5})
        np.testing.assert_allclose(values["a/power"], [50.0], rtol=RTOL)

    def test_a_slash_path_through_keyword_unpacking(self):
        values = Evaluator(fleet_builder())(**{"a/duty": 0.5, "b/duty": 0.25})
        np.testing.assert_allclose(values["b/rate"], [12.5], rtol=RTOL)

    def test_a_mapping_and_keywords_merge(self):
        values = Evaluator(fleet_builder())({"a/duty": 0.5}, **{"b/duty": 0.25}, limit=90.0)
        np.testing.assert_allclose([values["a/power"][0], values["b/power"][0],
                                    values["limit"][0]], [50.0, 25.0, 90.0], rtol=RTOL)

    def test_a_path_given_twice_is_refused(self):
        with pytest.raises(ValueError, match="given twice"):
            Evaluator(fleet_builder())({"a/duty": 0.5}, **{"a/duty": 0.25})

    def test_an_unknown_leaf_lists_the_free_leaves(self):
        with pytest.raises(ValueError, match="a/dutyy") as err:
            Evaluator(fleet_builder())(**{"a/dutyy": 0.5})
        assert "['a/duty', 'b/duty', 'limit']" in str(err.value)

    def test_an_algebraic_signal_is_an_output_not_an_input(self):
        with pytest.raises(ValueError, match="algebraic signals"):
            Evaluator(fleet_builder())(**{"a/power": 50.0})

    def test_a_missing_required_leaf_is_named(self):
        with pytest.raises(ValueError, match=r"no value for \['b'\]"):
            Evaluator(mixer_builder())(a=1.0)

    def test_a_scalar_is_not_broadcast_into_a_matrix_leaf(self):
        with pytest.raises(ValueError, match="nothing is broadcast"):
            Evaluator(grid_builder())(m=1.0)

    def test_a_wrong_shape_names_the_leaf(self):
        with pytest.raises(ValueError, match="'m'"):
            Evaluator(grid_builder())(m=[1.0, 2.0, 3.0])

    def test_a_positional_argument_must_be_a_mapping(self):
        with pytest.raises(TypeError, match="mapping"):
            Evaluator(mixer_builder())([1.0, 2.0])


class TestMatrixOutputs:

    def test_matrix_leaves_and_signals_come_back_in_their_declared_shape(self):
        values = Evaluator(grid_builder())(m=TARGET)
        np.testing.assert_allclose(values["m"], TARGET, rtol=RTOL)
        np.testing.assert_allclose(values["twice"], 2.0 * TARGET, rtol=RTOL)
        assert values["m"].shape == values["twice"].shape == (2, 2)
        assert values["trace"].shape == (1,)

    def test_a_flat_matrix_leaf_is_read_column_major(self):
        values = Evaluator(grid_builder())(m=[1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(values["m"], [[1.0, 3.0], [2.0, 4.0]], rtol=RTOL)

    def test_the_shapes_match_problem_evaluate_on_both_paths(self):
        problem = Problem([Grid()], registry=grid_registry(), verbose=False).compile().build()
        result = problem.solve()
        expected = problem.evaluate(result)
        np.testing.assert_allclose(expected["m"], TARGET, atol=1e-8)
        assert_same_values(Evaluator.from_problem(problem)(at_result(result)), expected)
        assert_same_values(Evaluator(problem.builder)(at_result(result)), expected)


class TestTable:

    def test_rows_come_back_in_order_with_the_row_keys_first(self):
        rows = [{"b/duty": 0.25, "a/duty": 0.5}, {"a/duty": 0.1}, {"limit": 90.0}]
        table = Evaluator(fleet_builder()).table(rows, outputs=["a/power", "limit"])
        assert [list(entry) for entry in table] == [
            ["b/duty", "a/duty", "a/power", "limit"],
            ["a/duty", "a/power", "limit"],
            ["limit", "a/power"],
        ]
        np.testing.assert_allclose([entry["a/power"] for entry in table], [50.0, 10.0, 10.0],
                                   rtol=RTOL)
        assert table[2]["limit"] == 90.0

    def test_size_one_outputs_are_floats_and_larger_ones_are_arrays(self):
        (entry,) = Evaluator(grid_builder()).table([{"m": TARGET}])
        assert type(entry["trace"]) is float
        np.testing.assert_allclose(entry["trace"], 5.0, rtol=RTOL)
        assert isinstance(entry["twice"], np.ndarray) and entry["twice"].shape == (2, 2)

    def test_the_default_is_every_output(self):
        (entry,) = Evaluator(mixer_builder()).table([{"b": 2.0}])
        assert list(entry) == ["b", "a", "total"]
        np.testing.assert_allclose(entry["total"], 5.0, rtol=RTOL)

    def test_an_unknown_output_lists_the_outputs(self):
        with pytest.raises(ValueError, match="a/powerr") as err:
            Evaluator(fleet_builder()).table([{}], outputs=["a/powerr"])
        assert "'a/power'" in str(err.value)

    def test_a_bare_string_for_outputs_is_the_missing_comma(self):
        with pytest.raises(TypeError, match=r"\('a/power',\)"):
            Evaluator(fleet_builder()).table([{}], outputs="a/power")

    def test_a_label_key_is_refused_with_the_way_to_carry_it(self):
        with pytest.raises(ValueError, match=r"row 0 has \['label'\]") as err:
            Evaluator(fleet_builder()).table([{"a/duty": 0.5, "label": "half"}])
        assert "zip(labels, evaluator.table(rows))" in str(err.value)
        assert "'a/duty'" in str(err.value)

    def test_a_bad_row_is_named_by_its_index(self):
        with pytest.raises(ValueError, match="row 1"):
            Evaluator(fleet_builder()).table([{"a/duty": 0.5}, {"a/dutyy": 0.5}])

    def test_a_single_mapping_is_not_a_table(self):
        with pytest.raises(TypeError, match="single mapping"):
            Evaluator(fleet_builder()).table({"a/duty": 0.5})


class TestDynamicModels:

    def test_states_and_inputs_are_leaves_and_outputs(self):
        evaluator = Evaluator(plant_builder())
        assert evaluator.leaves == ("x", "u", "gain")
        assert evaluator.outputs == ("x", "u", "gain", "y")      # no state derivative
        values = evaluator(x=1.0, u=0.5)
        np.testing.assert_allclose(values["y"], [2.5], rtol=RTOL)
        np.testing.assert_allclose(values["x"], [1.0], rtol=RTOL)

    def test_a_state_has_no_default_unless_values_gives_one(self):
        with pytest.raises(ValueError, match=r"no value for \['x'\]"):
            Evaluator(plant_builder())(u=0.5)
        values = Evaluator(plant_builder(), values={"x": 3.0, "u": 0.0})(gain=1.0)
        np.testing.assert_allclose(values["y"], [3.0], rtol=RTOL)
