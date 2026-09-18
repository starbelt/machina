"""Input validation, the registration lock, and naming rules of SolverBackend."""

import casadi as ca
import numpy as np
import pytest

from machina.solver import SolverBackend

pytestmark = pytest.mark.requires_casadi


def make_backend():
    return SolverBackend(verbose=False)


class TestValueValidation:

    def test_wrong_length_bound_raises(self):
        b = make_backend()
        with pytest.raises(ValueError, match="lb of 'x'"):
            b.add_variable("x", 3, lb=[0.0, 1.0])

    def test_nan_bound_raises(self):
        b = make_backend()
        with pytest.raises(ValueError, match="NaN"):
            b.add_variable("x", 2, ub=[1.0, np.nan])

    def test_lb_above_ub_raises(self):
        b = make_backend()
        with pytest.raises(ValueError, match="lb > ub"):
            b.add_variable("x", 1, lb=2.0, ub=1.0)

    def test_zero_d_array_bounds_are_scalars(self):
        """np.float64 and 0-d arrays broke the April ``list(lb)`` path."""
        b = make_backend()
        x = b.add_variable("x", 2)
        b.add_constraint(x, lb=np.array(0.0), ub=np.float64(1.0))
        con = b.constraints()[0]
        np.testing.assert_array_equal(con.lb, [0.0, 0.0])
        np.testing.assert_array_equal(con.ub, [1.0, 1.0])

    def test_constraint_bound_wrong_length_raises(self):
        b = make_backend()
        x = b.add_variable("x", 3)
        with pytest.raises(ValueError, match="'rows'"):
            b.add_constraint(x, lb=[0.0, 0.0], name="rows")

    def test_bad_shape_argument_raises(self):
        b = make_backend()
        with pytest.raises(ValueError, match="positive"):
            b.add_variable("x", 0)
        with pytest.raises(TypeError, match="shape"):
            b.add_variable("y", 2.5)

    def test_non_positive_scale_raises(self):
        b = make_backend()
        with pytest.raises(ValueError, match="scale"):
            b.add_variable("x", 1, scale=0.0)


class TestExpressionTypes:

    def test_matrix_constraint_expression_raises(self):
        b = make_backend()
        X = b.add_variable("X", (2, 2))
        with pytest.raises(ValueError, match="ca.vec"):
            b.add_constraint(X, lb=0.0)

    def test_sx_expression_is_refused_with_the_fix(self):
        b = make_backend()
        b.add_variable("x", 1)
        with pytest.raises(TypeError, match="ca.Function"):
            b.add_cost(ca.SX.sym("s") ** 2)
        with pytest.raises(TypeError, match="ca.Function"):
            b.add_constraint(ca.SX.sym("s"), lb=0.0)

    def test_sx_function_called_with_mx_is_the_supported_pattern(self):
        s = ca.SX.sym("s", 2)
        resid = ca.Function("resid", [s], [ca.vertcat(s[0] - 1.0, s[1] + 2.0)])
        b = make_backend()
        x = b.add_variable("x", 2)
        b.add_cost(ca.sumsqr(resid(x)))
        b.build()
        np.testing.assert_allclose(b.solve()["x"], [1.0, -2.0], atol=1e-6)

    def test_numeric_constants_are_accepted(self):
        """A FIXED-role substitution can turn a constraint or cost into a constant."""
        b = make_backend()
        x = b.add_variable("x", 1, initial_guess=1.0)
        b.add_cost((x - 2.0) ** 2)
        b.add_cost(3.0, name="offset")
        b.add_constraint(ca.DM([1.0, 2.0]), lb=0.0, name="constant")
        b.build()
        res = b.solve()
        assert res.success
        np.testing.assert_allclose(res.f_opt, 3.0, atol=1e-8)
        np.testing.assert_allclose(res.constraint("constant").value, [1.0, 2.0])


class TestNames:

    def test_duplicate_constraint_name_raises(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_constraint(x, lb=0.0, name="cap")
        with pytest.raises(ValueError, match="'cap'"):
            b.add_constraint(x, ub=1.0, name="cap")

    def test_duplicate_cost_name_raises(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2, name="term")
        with pytest.raises(ValueError, match="'term'"):
            b.add_cost(x**4, name="term")

    def test_unnamed_terms_are_auto_named_by_index(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_constraint(x, lb=0.0)
        b.add_constraint(x, ub=5.0, name="cap")
        b.add_constraint(x, ub=9.0)
        b.add_cost(x**2)
        assert [c.name for c in b.constraints()] == ["g0", "cap", "g2"]
        assert [c.auto_named for c in b.constraints()] == [True, False, True]
        assert [c.name for c in b.cost_terms()] == ["J0"]

    def test_empty_name_raises(self):
        b = make_backend()
        with pytest.raises(ValueError, match="non-empty"):
            b.add_variable("", 1)


class TestRegistrationLock:

    @pytest.fixture
    def built(self):
        b = make_backend()
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        b.build()
        return b, x

    def test_add_variable_after_build_raises(self, built):
        b, _ = built
        with pytest.raises(RuntimeError, match="locked"):
            b.add_variable("y", 1)

    def test_add_parameter_constraint_cost_after_build_raise(self, built):
        b, x = built
        with pytest.raises(RuntimeError, match="locked"):
            b.add_parameter("p", 1)
        with pytest.raises(RuntimeError, match="locked"):
            b.add_constraint(x, lb=0.0)
        with pytest.raises(RuntimeError, match="locked"):
            b.add_cost(x**2)

    def test_unknown_names_raise_keyerror_listing_what_exists(self, built):
        b, _ = built
        with pytest.raises(KeyError, match="Registered"):
            b.set_bounds("nope", lb=0.0)
        with pytest.raises(KeyError, match="nope"):
            b.slice_of("nope")

    def test_p_val_without_parameters_raises(self, built):
        b, _ = built
        with pytest.raises(ValueError, match="no parameters"):
            b.solve(p_val=[1.0])
