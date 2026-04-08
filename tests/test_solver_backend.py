"""
Tests for machina.solver.SolverBackend and SolutionResult.

Sections
--------
TestRegistration      — add_variable / add_parameter / add_constraint / add_cost
TestBuild             — build() behavior and error cases
TestSolveValidation   — solve() pre-flight error cases (no solver call)
TestSolveConvergence  — known problems with analytically known solutions
TestResultInterface   — SolutionResult fields, __getitem__, lam_p
TestExtractAndStats   — extract() and stats() methods

All tests are marked `requires_casadi` at the module level and are
auto-skipped by conftest when CasADi is not installed.
"""

import pytest
import numpy as np
import casadi as ca

from machina.solver.SolverBackend import SolverBackend
from machina.solver.SolutionResult import SolutionResult

pytestmark = pytest.mark.requires_casadi

# ---------------------------------------------------------------------------
# Tolerance for numerical comparisons. IPOPT default tol is 1e-8, so 1e-5
# gives comfortable headroom without being loose.
# ---------------------------------------------------------------------------
ATOL = 1e-5


def make_backend(solver_opts=None):
    """Return a SolverBackend with IPOPT console output suppressed."""
    opts = {"ipopt.print_level": 0, "print_time": False}
    if solver_opts:
        opts.update(solver_opts)
    return SolverBackend(solver_opts=opts)


# ===========================================================================
# Registration
# ===========================================================================

class TestRegistration:
    """Tests for add_variable, add_parameter, add_constraint, add_cost."""

    # --- add_variable -------------------------------------------------------

    def test_add_variable_returns_mx_symbol(self):
        b = make_backend()
        x = b.add_variable("x", 3)
        assert isinstance(x, ca.MX)
        assert x.shape == (3, 1)

    def test_add_variable_scalar_broadcast(self):
        b = make_backend()
        b.add_variable("x", 3, lb=-1.0, ub=2.0, initial_guess=0.5)
        assert b._lbw == [-1.0, -1.0, -1.0]
        assert b._ubw == [2.0, 2.0, 2.0]
        assert b._w0 == [0.5, 0.5, 0.5]

    def test_add_variable_array_bounds(self):
        b = make_backend()
        b.add_variable("x", 3, lb=[-1.0, -2.0, -3.0], ub=[1.0, 2.0, 3.0])
        assert b._lbw == [-1.0, -2.0, -3.0]
        assert b._ubw == [1.0, 2.0, 3.0]

    def test_add_variable_duplicate_name_raises(self):
        b = make_backend()
        b.add_variable("x", 1)
        with pytest.raises(ValueError, match="'x'"):
            b.add_variable("x", 1)

    def test_add_variable_offset_tracking_single(self):
        b = make_backend()
        b.add_variable("x", 4)
        assert b._var_map["x"] == (0, 4, (4, 1))
        assert b._offset == 4

    def test_add_variable_offset_tracking_multiple(self):
        """Each variable occupies a contiguous, non-overlapping slice."""
        b = make_backend()
        b.add_variable("a", 2)
        b.add_variable("b", 3)
        b.add_variable("c", 1)
        assert b._var_map["a"] == (0, 2, (2, 1))
        assert b._var_map["b"] == (2, 5, (3, 1))
        assert b._var_map["c"] == (5, 6, (1, 1))
        assert b._offset == 6

    def test_add_variable_added_to_names_set(self):
        b = make_backend()
        b.add_variable("x", 1)
        assert "x" in b._names

    # --- add_parameter ------------------------------------------------------

    def test_add_parameter_returns_mx_symbol(self):
        b = make_backend()
        p = b.add_parameter("p", 2)
        assert isinstance(p, ca.MX)
        assert p.shape == (2, 1)

    def test_add_parameter_duplicate_name_raises(self):
        b = make_backend()
        b.add_parameter("p", 1)
        with pytest.raises(ValueError, match="'p'"):
            b.add_parameter("p", 1)

    def test_add_parameter_name_conflicts_with_variable(self):
        """Variables and parameters share a single namespace."""
        b = make_backend()
        b.add_variable("shared", 1)
        with pytest.raises(ValueError, match="'shared'"):
            b.add_parameter("shared", 1)

    def test_add_parameter_offset_tracking(self):
        b = make_backend()
        b.add_parameter("p1", 2)
        b.add_parameter("p2", 3)
        assert b._param_map["p1"] == (0, 2, (2, 1))
        assert b._param_map["p2"] == (2, 5, (3, 1))
        assert b._p_offset == 5

    # --- add_constraint -----------------------------------------------------

    def test_add_constraint_appends_expression(self):
        b = make_backend()
        x = b.add_variable("x", 3)
        b.add_constraint(x, lb=0.0, ub=1.0)
        assert len(b._g) == 1

    def test_add_constraint_scalar_bound_broadcast(self):
        b = make_backend()
        x = b.add_variable("x", 3)
        b.add_constraint(x, lb=0.0, ub=1.0)
        assert b._lbg == [0.0, 0.0, 0.0]
        assert b._ubg == [1.0, 1.0, 1.0]

    def test_add_constraint_array_bounds(self):
        b = make_backend()
        x = b.add_variable("x", 3)
        b.add_constraint(x, lb=[-1.0, -2.0, -3.0], ub=[1.0, 2.0, 3.0])
        assert b._lbg == [-1.0, -2.0, -3.0]
        assert b._ubg == [1.0, 2.0, 3.0]

    def test_add_constraint_records_name_and_size(self):
        b = make_backend()
        x = b.add_variable("x", 2)
        b.add_constraint(x, lb=0.0, name="my_con")
        assert b._constraint_names[-1] == ("my_con", 2)

    def test_add_constraint_name_optional(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_constraint(x)   # no name — should not raise
        assert b._constraint_names[-1][0] is None

    def test_add_constraint_does_not_pollute_names_set(self):
        """Constraint names must NOT enter the variable/parameter namespace."""
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_constraint(x, lb=0.0, name="con")
        # A variable named 'con' should still be registerable
        b.add_variable("con", 1)

    def test_add_equality_sets_zero_bounds(self):
        b = make_backend()
        x = b.add_variable("x", 2)
        b.add_equality(x - 1.0)
        assert b._lbg == [0.0, 0.0]
        assert b._ubg == [0.0, 0.0]

    def test_add_equality_delegates_to_add_constraint(self):
        """add_equality(expr) is equivalent to add_constraint(expr, lb=0, ub=0)."""
        b1 = make_backend()
        x1 = b1.add_variable("x", 1)
        b1.add_equality(x1 - 5.0)

        b2 = make_backend()
        x2 = b2.add_variable("x", 1)
        b2.add_constraint(x2 - 5.0, lb=0.0, ub=0.0)

        assert b1._lbg == b2._lbg
        assert b1._ubg == b2._ubg

    # --- add_cost -----------------------------------------------------------

    def test_add_cost_non_scalar_raises(self):
        b = make_backend()
        x = b.add_variable("x", 3)
        with pytest.raises(ValueError):
            b.add_cost(x)   # shape (3, 1), not (1, 1)

    def test_add_cost_accumulates_terms(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2, name="term1")
        b.add_cost(x**2, name="term2")
        assert len(b._cost_terms) == 2

    def test_add_cost_does_not_pollute_names_set(self):
        """Cost names must NOT enter the variable/parameter namespace."""
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2, name="cost")
        # Should be able to register a variable with the same name
        b.add_variable("cost", 1)


# ===========================================================================
# Build
# ===========================================================================

class TestBuild:

    def test_build_no_variables_raises(self):
        b = make_backend()
        with pytest.raises(RuntimeError, match="No decision variables"):
            b.build()

    def test_build_returns_casadi_function(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        result = b.build()
        assert isinstance(result, ca.Function)

    def test_build_sets_built_flag(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        assert not b._built
        b.build()
        assert b._built

    def test_build_twice_raises(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        b.build()
        with pytest.raises(RuntimeError, match="already been built"):
            b.build()

    def test_build_opts_override_defaults(self):
        """build(opts) layer can tighten tolerances without error."""
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        b.build(opts={"ipopt.tol": 1e-10, "ipopt.max_iter": 500})
        assert b._built

    def test_build_no_parameters_omits_p_key(self):
        """
        When no parameters are registered, solve() should work with p_val=None.
        Implicitly verifies that 'p' was not added to the NLP dict
        (CasADi would require p= at every call if it were).
        """
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost((x - 1)**2)
        b.build()
        res = b.solve()   # must not raise even though p_val=None
        assert res.success

    def test_build_zero_cost_warns(self):
        """build() emits a UserWarning when no cost terms have been added."""
        b = make_backend()
        b.add_variable("x", 1)
        with pytest.warns(UserWarning, match="objective will be zero"):
            b.build()


# ===========================================================================
# Solve — validation (pre-flight checks, no solver execution)
# ===========================================================================

class TestSolveValidation:

    def test_solve_before_build_raises(self):
        b = make_backend()
        b.add_variable("x", 1)
        with pytest.raises(RuntimeError, match="build()"):
            b.solve()

    def test_solve_params_registered_pval_none_raises(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        p = b.add_parameter("p", 1)
        b.add_cost((x - p)**2)
        b.build()
        with pytest.raises(ValueError, match="p_val is None"):
            b.solve(p_val=None)

    def test_solve_params_wrong_length_raises(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        p = b.add_parameter("p", 3)
        b.add_cost((x - p[0])**2)
        b.build()
        with pytest.raises(ValueError, match="length"):
            b.solve(p_val=[1.0, 2.0])   # needs 3, got 2

    def test_solve_p_val_2d_array_normalized(self):
        """p_val is normalized to a flat array before being passed to CasADi.
        A column vector [[3.0]] should behave identically to [3.0]."""
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        p = b.add_parameter("p", 1)
        b.add_cost((x - p)**2)
        b.build()
        res_flat   = b.solve(p_val=[3.0])
        res_2d     = b.solve(p_val=[[3.0]])          # 2-D nested list
        res_column = b.solve(p_val=np.array([[3.0]])) # (1,1) numpy array
        np.testing.assert_allclose(res_flat["x"],   [3.0], atol=ATOL)
        np.testing.assert_allclose(res_2d["x"],     [3.0], atol=ATOL)
        np.testing.assert_allclose(res_column["x"], [3.0], atol=ATOL)

    def test_stats_before_solve_raises(self):
        """stats() must raise before any solve() call has been made."""
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        b.build()
        with pytest.raises(RuntimeError, match="solve()"):
            b.stats()


# ===========================================================================
# Solve — convergence on known problems
# ===========================================================================

class TestSolveConvergence:
    """
    Each test states the problem, the analytic solution, and the tolerance.
    All problems are simple enough that IPOPT converges in < 50 iterations.
    """

    def test_unconstrained_quadratic(self):
        """
        minimize  (x - 3)^2
        Solution: x* = 3,  f* = 0
        """
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        b.add_cost((x - 3)**2)
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res["x"], [3.0], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 0.0, atol=ATOL)

    def test_constrained_equality_quadratic(self):
        """
        minimize  x^2 + y^2
        s.t.      x + y = 1
        Solution: x* = y* = 0.5,  f* = 0.5
        """
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        y = b.add_variable("y", 1, initial_guess=0.0)
        b.add_cost(x**2 + y**2)
        b.add_equality(x + y - 1, name="sum_to_one")
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res["x"], [0.5], atol=ATOL)
        np.testing.assert_allclose(res["y"], [0.5], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 0.5, atol=ATOL)

    def test_lower_variable_bound_active(self):
        """
        minimize  x^2
        s.t.      x >= 2  (encoded as variable lower bound)
        Solution: x* = 2,  f* = 4
        """
        b = make_backend()
        x = b.add_variable("x", 1, lb=2.0, initial_guess=5.0)
        b.add_cost(x**2)
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res["x"], [2.0], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 4.0, atol=ATOL)

    def test_upper_inequality_constraint_active(self):
        """
        minimize  (x - 5)^2
        s.t.      x <= 3  (encoded as constraint upper bound)
        Solution: x* = 3,  f* = 4
        """
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        b.add_cost((x - 5)**2)
        b.add_constraint(x, ub=3.0, name="upper_cap")
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res["x"], [3.0], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 4.0, atol=ATOL)

    def test_double_sided_inequality_constraint(self):
        """
        minimize  (x - 10)^2
        s.t.      1 <= x <= 4
        Solution: x* = 4,  f* = 36
        """
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        b.add_cost((x - 10)**2)
        b.add_constraint(x, lb=1.0, ub=4.0, name="band")
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res["x"], [4.0], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 36.0, atol=ATOL)

    def test_rosenbrock(self):
        """
        minimize  (1 - x)^2 + 100*(y - x^2)^2
        Solution: x* = 1,  y* = 1,  f* = 0
        Classic nonlinear benchmark; requires more iterations than quadratics.
        """
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        y = b.add_variable("y", 1, initial_guess=0.0)
        b.add_cost((1 - x)**2 + 100*(y - x**2)**2)
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res["x"], [1.0], atol=ATOL)
        np.testing.assert_allclose(res["y"], [1.0], atol=ATOL)
        np.testing.assert_allclose(res.f_opt, 0.0, atol=ATOL)

    def test_parametric_nlp_two_solve_calls(self):
        """
        minimize  (x - p)^2  where p is a parameter
        Solution: x* = p  for any p.
        Verifies that the same built solver can be re-called with different
        parameter values without rebuilding.
        """
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        p = b.add_parameter("p", 1)
        b.add_cost((x - p)**2)
        b.build()

        for p_val in [5.0, -2.0, 0.0]:
            res = b.solve(p_val=[p_val])
            assert res.success, f"Failed for p={p_val}"
            np.testing.assert_allclose(res["x"], [p_val], atol=ATOL)
            np.testing.assert_allclose(res.f_opt, 0.0, atol=ATOL)

    def test_multi_variable_extraction(self):
        """
        Three separately registered variables of different sizes.
        Verifies that _var_map slicing is correct for each.

        minimize  (x - 1)^2 + ||y - [2, 3]||^2 + ||z - [4, 5, 6]||^2
        Solution: x* = 1,  y* = [2, 3],  z* = [4, 5, 6],  f* = 0
        """
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        y = b.add_variable("y", 2, initial_guess=0.0)
        z = b.add_variable("z", 3, initial_guess=0.0)
        b.add_cost(
            (x - 1)**2
            + (y[0] - 2)**2 + (y[1] - 3)**2
            + (z[0] - 4)**2 + (z[1] - 5)**2 + (z[2] - 6)**2
        )
        b.build()
        res = b.solve()

        assert res.success
        np.testing.assert_allclose(res["x"], [1.0], atol=ATOL)
        np.testing.assert_allclose(res["y"], [2.0, 3.0], atol=ATOL)
        np.testing.assert_allclose(res["z"], [4.0, 5.0, 6.0], atol=ATOL)

    def test_success_flag_false_on_infeasible(self):
        """
        Contradictory constraints (x >= 2 AND x <= 1) should cause IPOPT to
        report failure.  error_on_fail=False prevents a Python exception so we
        can inspect the SolutionResult directly.
        """
        b = SolverBackend(solver_opts={
            "ipopt.print_level": 0,
            "print_time": False,
            "error_on_fail": False,
        })
        x = b.add_variable("x", 1, initial_guess=0.0)
        b.add_cost(x**2)
        b.add_constraint(x, lb=2.0, name="lb_con")   # x >= 2
        b.add_constraint(x, ub=1.0, name="ub_con")   # x <= 1  (contradicts lb)
        b.build()
        res = b.solve()

        assert not res.success


# ===========================================================================
# SolutionResult interface
# ===========================================================================

class TestResultInterface:

    @pytest.fixture
    def simple_result(self):
        """Solve a trivial 2-variable problem; return the SolutionResult."""
        b = make_backend()
        x = b.add_variable("x", 2, initial_guess=0.0)
        b.add_cost((x[0] - 1)**2 + (x[1] - 2)**2)
        b.build()
        return b.solve()

    def test_result_is_dataclass_instance(self, simple_result):
        assert isinstance(simple_result, SolutionResult)

    def test_getitem_delegates_to_x_opt(self, simple_result):
        np.testing.assert_array_equal(simple_result["x"], simple_result.x_opt["x"])

    def test_lam_p_is_none_without_parameters(self, simple_result):
        assert simple_result.lam_p is None

    def test_lam_p_is_array_with_parameters(self):
        """When parameters are registered, lam_p must be a numpy array (not None)."""
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=0.0)
        p = b.add_parameter("p", 1)
        b.add_cost((x - p)**2)
        b.build()
        res = b.solve(p_val=[3.0])
        assert res.lam_p is not None
        assert isinstance(res.lam_p, np.ndarray)
        assert res.lam_p.shape == (1,)

    def test_g_opt_shape_matches_total_constraint_rows(self):
        """g_opt length must equal the total number of constraint rows."""
        b = make_backend()
        x = b.add_variable("x", 3, initial_guess=1.0)
        b.add_cost(ca.sumsqr(x))
        b.add_equality(x - 1.0, name="fix")  # 3 rows
        b.build()
        res = b.solve()
        assert res.g_opt.shape == (3,)

    def test_lam_x_shape_matches_total_variable_size(self):
        """lam_x length must equal the total number of decision variable elements."""
        b = make_backend()
        b.add_variable("a", 2, initial_guess=0.0)
        b.add_variable("b", 3, initial_guess=0.0)
        x_a = b._w[0]
        x_b = b._w[1]
        b.add_cost(ca.sumsqr(x_a) + ca.sumsqr(x_b))
        b.build()
        res = b.solve()
        assert res.lam_x.shape == (5,)

    def test_f_opt_is_float(self, simple_result):
        assert isinstance(simple_result.f_opt, float)

    def test_raw_sol_contains_standard_keys(self, simple_result):
        for key in ("x", "f", "g", "lam_x", "lam_g"):
            assert key in simple_result.raw_sol

    def test_stats_dict_in_result(self, simple_result):
        assert isinstance(simple_result.stats, dict)
        assert "return_status" in simple_result.stats

    def test_x_opt_keys_match_registered_names(self):
        b = make_backend()
        b.add_variable("alpha", 2, initial_guess=0.0)
        b.add_variable("beta", 1, initial_guess=0.0)
        w0, w1 = b._w
        b.add_cost(ca.sumsqr(w0) + ca.sumsqr(w1))
        b.build()
        res = b.solve()
        assert set(res.x_opt.keys()) == {"alpha", "beta"}

    def test_x_opt_values_are_numpy_arrays(self):
        b = make_backend()
        x = b.add_variable("x", 3, initial_guess=0.0)
        b.add_cost(ca.sumsqr(x))
        b.build()
        res = b.solve()
        assert isinstance(res.x_opt["x"], np.ndarray)


# ===========================================================================
# extract() and stats()
# ===========================================================================

class TestExtractAndStats:

    @pytest.fixture
    def backend_and_result(self):
        b = make_backend()
        x = b.add_variable("x", 2, initial_guess=0.0)
        y = b.add_variable("y", 1, initial_guess=0.0)
        b.add_cost((x[0] - 1)**2 + (x[1] - 2)**2 + (y - 3)**2)
        b.build()
        res = b.solve()
        return b, res

    def test_extract_first_variable_matches_x_opt(self, backend_and_result):
        b, res = backend_and_result
        extracted = b.extract(res.raw_sol, "x")
        np.testing.assert_array_equal(extracted, res.x_opt["x"])

    def test_extract_second_variable_matches_x_opt(self, backend_and_result):
        b, res = backend_and_result
        extracted = b.extract(res.raw_sol, "y")
        np.testing.assert_array_equal(extracted, res.x_opt["y"])

    def test_extract_returns_numpy_array(self, backend_and_result):
        b, res = backend_and_result
        result = b.extract(res.raw_sol, "x")
        assert isinstance(result, np.ndarray)

    def test_stats_returns_dict(self, backend_and_result):
        b, _ = backend_and_result
        assert isinstance(b.stats(), dict)

    def test_stats_has_return_status(self, backend_and_result):
        b, _ = backend_and_result
        assert "return_status" in b.stats()

    def test_stats_has_iter_count(self, backend_and_result):
        b, _ = backend_and_result
        assert "iter_count" in b.stats()

    def test_stats_return_status_matches_result(self, backend_and_result):
        """stats() after solve should reflect the last solve call."""
        b, res = backend_and_result
        assert b.stats()["return_status"] == res.stats["return_status"]
