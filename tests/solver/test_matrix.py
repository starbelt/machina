"""
Matrix-shaped variables and parameters (Decision Log #16).

The decision vector stores a matrix column-major (``ca.vec``). Bounds, guesses
and parameter values given in the natural ``(rows, cols)`` shape must land on
the matching elements. The April backend flattened them C-order, which
scrambled every non-symmetric array; these tests pin the fix.
"""

import casadi as ca
import numpy as np
import pytest

from machina.solver import SolverBackend

pytestmark = pytest.mark.requires_casadi

ATOL = 1e-6


def make_backend():
    return SolverBackend(verbose=False)


class TestMatrixVariables:

    def test_matrix_variable_bounds_are_column_major(self):
        """Pin every element with lb == ub; the solution must equal the matrix."""
        target = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]])
        b = make_backend()
        X = b.add_variable("X", (2, 3), lb=target, ub=target, initial_guess=target)
        b.add_cost(ca.sumsqr(X))
        b.build()
        res = b.solve()
        assert res.success
        np.testing.assert_allclose(res["X"], target, atol=ATOL)

    def test_flat_bounds_are_taken_as_column_major(self):
        target = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]])
        b = make_backend()
        b.add_variable("X", (2, 3), lb=target.ravel(order='F'), ub=target.ravel(order='F'))
        lb, ub = b.bounds("X")
        np.testing.assert_array_equal(lb, target)
        np.testing.assert_array_equal(ub, target)

    def test_matrix_initial_guess_is_column_major(self):
        """With zero iterations the solver returns the (projected) initial guess."""
        guess = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        b = SolverBackend(verbose=False, solver_opts={"ipopt.max_iter": 0})
        X = b.add_variable("X", (3, 2), initial_guess=guess)
        b.add_cost(ca.sumsqr(X - 10.0))
        b.build()
        res = b.solve()
        np.testing.assert_allclose(res["X"], guess, atol=1e-8)
        np.testing.assert_array_equal(b.initial_guess("X"), guess)

    def test_matrix_variable_active_elementwise_bound(self):
        """Only element (0, 2) is capped; the cap must bind there and nowhere else."""
        ub = np.full((2, 3), np.inf)
        ub[0, 2] = 1.0
        b = make_backend()
        X = b.add_variable("X", (2, 3), ub=ub, initial_guess=0.0)
        b.add_cost(ca.sumsqr(X - 5.0))
        b.build()
        res = b.solve()
        expected = np.full((2, 3), 5.0)
        expected[0, 2] = 1.0
        np.testing.assert_allclose(res["X"], expected, atol=1e-5)
        assert res.bound_multiplier("X")[0, 2] > 1.0
        np.testing.assert_allclose(np.delete(res.bound_multiplier("X").ravel(order='F'), 4),
                                   0.0, atol=1e-5)

    def test_wrong_shape_bounds_raise_and_name_the_variable(self):
        b = make_backend()
        with pytest.raises(ValueError, match="'X'"):
            b.add_variable("X", (2, 3), lb=np.zeros((3, 2)))

    def test_compiler_style_column_shapes(self):
        """The compiler always passes tuples, including (n, 1) and (1, 1)."""
        b = make_backend()
        v = b.add_variable("v", (3, 1), lb=[0.0, 1.0, 2.0], initial_guess=[[5.0], [5.0], [5.0]])
        s = b.add_variable("s", (1, 1), initial_guess=1.0)
        b.add_cost(ca.sumsqr(v + 1.0) + s**2)   # pulls v onto its lower bounds
        b.build()
        res = b.solve()
        np.testing.assert_allclose(res["v"], [0.0, 1.0, 2.0], atol=1e-5)
        assert res["v"].shape == (3,)
        assert res["s"].shape == (1,)


class TestMatrixParameters:

    def _least_squares(self):
        b = make_backend()
        x = b.add_variable("x", 2, initial_guess=0.0)
        A = b.add_parameter("A", (4, 2))
        rhs = b.add_parameter("b", 4)
        r = A @ x - rhs
        b.add_cost(0.5 * ca.sumsqr(r))
        b.build()
        A_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 9.0]])
        b_data = np.array([1.0, 2.0, 3.0, 5.0])
        x_ref = np.linalg.lstsq(A_data, b_data, rcond=None)[0]
        return b, A_data, b_data, x_ref

    def test_matrix_parameter_values_are_column_major(self):
        b, A_data, b_data, x_ref = self._least_squares()
        res = b.solve(p_val={"A": A_data, "b": b_data})
        assert res.success
        np.testing.assert_allclose(res["x"], x_ref, atol=ATOL)
        np.testing.assert_array_equal(res.p_opt["A"], A_data)

    def test_p_val_flat_vector_is_registration_order_column_major(self):
        b, A_data, b_data, x_ref = self._least_squares()
        flat = np.concatenate([A_data.flatten(order='F'), b_data])
        res = b.solve(p_val=flat)
        np.testing.assert_allclose(res["x"], x_ref, atol=ATOL)

    def test_stored_matrix_parameter_value(self):
        b, A_data, b_data, x_ref = self._least_squares()
        b.set_parameter("A", A_data)
        b.set_parameter("b", b_data)
        res = b.solve()
        np.testing.assert_allclose(res["x"], x_ref, atol=ATOL)
        np.testing.assert_array_equal(b.parameter_value("A"), A_data)

    def test_sensitivity_has_the_parameter_shape(self):
        b, A_data, b_data, _ = self._least_squares()
        res = b.solve(p_val={"A": A_data, "b": b_data})
        assert res.sensitivity("A").shape == (4, 2)
        assert res.sensitivity("b").shape == (4,)
