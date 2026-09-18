"""SolutionResult additions: named duals, sensitivities, cost breakdown, pickling."""

import pickle

import casadi as ca
import numpy as np
import pytest

from machina.solver import ConstraintValue, SolutionResult, SolverBackend

pytestmark = pytest.mark.requires_casadi

ATOL = 1e-6


def budget_problem(budget=10.0):
    """
    maximize 3 a + 2 b  subject to  a + b <= budget,  0 <= a <= 4,  b >= 0.

    Optimum a = 4, b = budget - 4; one more unit of budget is worth 2.
    """
    s = SolverBackend(verbose=False)
    a = s.add_variable("a", 1, lb=0.0, ub=4.0, initial_guess=1.0)
    b = s.add_variable("b", 1, lb=0.0, initial_guess=1.0)
    limit = s.add_parameter("budget", 1, value=budget)
    s.add_cost(-3.0 * a, name="value_a")
    s.add_cost(-b, name="value_b", weight=2.0)
    s.add_constraint(a + b - limit, ub=0.0, name="budget")
    s.add_constraint(a - b, lb=-100.0, ub=100.0, name="slack_band")
    s.build()
    return s


class TestNamedAccess:

    def test_constraint_value_multiplier_and_activity(self):
        res = budget_problem().solve()
        assert res.success
        con = res.constraint("budget")
        assert isinstance(con, ConstraintValue)
        np.testing.assert_allclose(con.value, [0.0], atol=1e-5)
        np.testing.assert_allclose(con.multiplier, [2.0], atol=1e-5)     # shadow price
        assert con.active.tolist() == [True]
        band = res.constraint("slack_band")
        assert band.active.tolist() == [False]
        np.testing.assert_allclose(band.multiplier, [0.0], atol=1e-5)
        np.testing.assert_allclose(band.violation, [0.0])

    def test_shadow_price_matches_a_finite_difference(self):
        """The multiplier is d f*/d budget with the sign flipped: a thesis-style number."""
        f0 = budget_problem(10.0).solve().f_opt
        f1 = budget_problem(10.5).solve().f_opt
        res = budget_problem(10.0).solve()
        np.testing.assert_allclose((f1 - f0) / 0.5, -res.constraint("budget").multiplier[0],
                                   atol=1e-4)
        np.testing.assert_allclose(res.sensitivity("budget"), [(f1 - f0) / 0.5], atol=1e-4)

    def test_bound_multiplier(self):
        res = budget_problem().solve()
        np.testing.assert_allclose(res.bound_multiplier("a"), [1.0], atol=1e-5)   # a at ub, worth 3-2
        np.testing.assert_allclose(res.bound_multiplier("b"), [0.0], atol=1e-5)

    def test_unknown_names_list_what_exists(self):
        res = budget_problem().solve()
        with pytest.raises(KeyError, match="budget"):
            res.constraint("nope")
        with pytest.raises(KeyError, match="variable"):
            res.bound_multiplier("nope")

    def test_cost_breakdown_sums_to_the_objective(self):
        res = budget_problem().solve()
        assert list(res.cost_terms) == ["value_a", "value_b"]
        np.testing.assert_allclose(res.cost_terms["value_a"], -12.0, atol=1e-5)
        np.testing.assert_allclose(res.cost_terms["value_b"], -12.0, atol=1e-5)   # weight applied
        np.testing.assert_allclose(sum(res.cost_terms.values()), res.f_opt, atol=1e-9)

    def test_self_describing_fields(self):
        res = budget_problem().solve()
        assert res.plugin == "ipopt"
        assert res.status == "Solve_Succeeded" and res.unified_status == "SOLVER_RET_SUCCESS"
        assert [e.name for e in res.x_layout] == ["a", "b"]
        assert [e.name for e in res.g_layout] == ["budget", "slack_band"]
        np.testing.assert_array_equal(res.p_opt["budget"], [10.0])
        np.testing.assert_array_equal(res.ubg, [0.0, 100.0])
        assert res.max_constraint_violation < 1e-6


class TestCompatibilityAndPickling:

    def test_positional_construction_still_works(self):
        """Code written against the April dataclass keeps working."""
        res = SolutionResult(True, {"x": np.zeros(1)}, 0.0, np.zeros(0), np.zeros(1),
                             np.zeros(0), {}, {})
        assert res.lam_p is None and res.status == "" and res.relaxed_discrete == ()
        assert res.max_constraint_violation == 0.0

    def test_result_pickles(self):
        res = budget_problem().solve()
        clone = pickle.loads(pickle.dumps(res))
        np.testing.assert_array_equal(clone["a"], res["a"])
        np.testing.assert_array_equal(clone.lam_g, res.lam_g)
        assert clone.x_layout == res.x_layout
        np.testing.assert_allclose(clone.constraint("budget").multiplier,
                                   res.constraint("budget").multiplier)
        np.testing.assert_allclose(clone.raw_sol["x"].full(), res.raw_sol["x"].full())

    def test_pickled_result_warm_starts_a_fresh_backend(self):
        """What a cached or parallel sweep does: carry results across processes."""
        first = budget_problem(10.0)
        stored = pickle.loads(pickle.dumps(first.solve()))
        second = budget_problem(10.2)
        cold = second.solve()
        warm = second.solve(warm_start=stored)
        assert warm.warm_started and warm.iterations < cold.iterations
        np.testing.assert_allclose(warm["b"], [6.2], atol=1e-5)

    def test_matrix_variable_round_trip_in_result(self):
        s = SolverBackend(verbose=False)
        X = s.add_variable("X", (2, 3), initial_guess=0.0)
        target = ca.DM(np.arange(6.0).reshape(2, 3))
        s.add_cost(ca.sumsqr(X - target))
        s.build()
        res = s.solve()
        np.testing.assert_allclose(res["X"], np.arange(6.0).reshape(2, 3), atol=ATOL)
        assert res.x_layout[0].shape == (2, 3) and res.x_layout[0].numel == 6
