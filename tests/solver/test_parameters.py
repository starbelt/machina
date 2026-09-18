"""Stored parameter values, per-call overrides, and editable bounds and guesses."""

import numpy as np
import pytest

from machina.solver import SolverBackend

pytestmark = pytest.mark.requires_casadi

ATOL = 1e-6


def tracking_problem():
    """minimize (x - a)^2 + (y - b)^2 with parameters a, b."""
    s = SolverBackend(verbose=False)
    x = s.add_variable("x", 1, lb=-10, ub=10)
    y = s.add_variable("y", 1, lb=-10, ub=10)
    a = s.add_parameter("a", 1, value=1.0)
    b = s.add_parameter("b", 1)
    s.add_cost((x - a) ** 2 + (y - b) ** 2)
    return s


class TestStoredParameters:

    def test_missing_value_error_names_the_parameter(self):
        s = tracking_problem()
        s.build()
        with pytest.raises(ValueError, match=r"\['b'\]"):
            s.solve()

    def test_stored_values_are_used_without_p_val(self):
        s = tracking_problem()
        s.set_parameter("b", 2.0)
        s.build()
        res = s.solve()
        np.testing.assert_allclose([res["x"][0], res["y"][0]], [1.0, 2.0], atol=ATOL)
        assert res.p_opt == {"a": np.array([1.0]), "b": np.array([2.0])}

    def test_p_val_dict(self):
        """A dict overrides just the named parameters, for this call only."""
        s = tracking_problem()
        s.set_parameter("b", 2.0)
        s.build()
        res = s.solve(p_val={"a": -3.0})
        np.testing.assert_allclose([res["x"][0], res["y"][0]], [-3.0, 2.0], atol=ATOL)
        np.testing.assert_allclose(s.parameter_value("a"), [1.0])     # store untouched
        again = s.solve()
        np.testing.assert_allclose(again["x"], [1.0], atol=ATOL)

    def test_flat_p_val_still_means_the_whole_vector(self):
        s = tracking_problem()
        s.build()
        res = s.solve(p_val=[4.0, 5.0])
        np.testing.assert_allclose([res["x"][0], res["y"][0]], [4.0, 5.0], atol=ATOL)

    def test_unknown_parameter_in_dict_raises(self):
        s = tracking_problem()
        s.set_parameter("b", 0.0)
        s.build()
        with pytest.raises(KeyError, match="'c'"):
            s.solve(p_val={"c": 1.0})

    def test_set_parameter_after_build_needs_no_rebuild(self):
        s = tracking_problem()
        s.set_parameter("b", 0.0)
        s.build()
        fn = s.nlp_function()
        for value in (2.0, -1.5, 7.0):
            s.set_parameter("a", value)
            np.testing.assert_allclose(s.solve()["x"], [value], atol=ATOL)
        assert s.nlp_function() is fn

    def test_non_finite_parameter_raises(self):
        s = tracking_problem()
        s.build()
        with pytest.raises(ValueError, match="'b'"):
            s.solve(p_val={"b": np.inf})

    def test_parameter_weight_on_a_cost_term(self):
        """Weights given as parameters can be swept without a rebuild."""
        s = SolverBackend(verbose=False)
        x = s.add_variable("x", 1)
        w = s.add_parameter("w", 1, value=1.0)
        s.add_cost((x - 0.0) ** 2, name="stay")
        s.add_cost((x - 10.0) ** 2, name="go", weight=w)
        s.build()
        np.testing.assert_allclose(s.solve()["x"], [5.0], atol=ATOL)
        res = s.solve(p_val={"w": 3.0})
        np.testing.assert_allclose(res["x"], [7.5], atol=ATOL)
        np.testing.assert_allclose(res.cost_terms["go"], 3.0 * 2.5**2, atol=1e-5)


class TestEditableBoundsAndGuesses:

    def make(self):
        s = SolverBackend(verbose=False)
        x = s.add_variable("x", 2, lb=-1.0, ub=1.0, initial_guess=0.5)
        s.add_cost((x[0] - 5.0) ** 2 + (x[1] + 5.0) ** 2)
        return s

    def test_set_bounds_after_build(self):
        s = self.make()
        s.build()
        np.testing.assert_allclose(s.solve()["x"], [1.0, -1.0], atol=1e-5)
        s.set_bounds("x", lb=[-1.0, -3.0], ub=4.0)
        np.testing.assert_allclose(s.solve()["x"], [4.0, -3.0], atol=1e-5)

    def test_set_bounds_before_build(self):
        s = self.make()
        s.set_bounds("x", ub=2.0)
        s.build()
        np.testing.assert_allclose(s.solve()["x"], [2.0, -1.0], atol=1e-5)

    def test_set_bounds_rejects_crossed_bounds(self):
        s = self.make()
        with pytest.raises(ValueError, match="lb > ub"):
            s.set_bounds("x", lb=2.0)

    def test_fix_and_unfix(self):
        s = self.make()
        s.build()
        s.fix("x", [0.25, -0.75])
        res = s.solve()
        np.testing.assert_allclose(res["x"], [0.25, -0.75], atol=1e-8)
        assert s.variables()[0].is_fixed
        s.unfix("x")
        np.testing.assert_allclose(s.solve()["x"], [1.0, -1.0], atol=1e-5)

    def test_reset_restores_registered_values(self):
        s = self.make()
        s.set_bounds("x", lb=-9.0, ub=9.0)
        s.set_initial_guess("x", [3.0, 3.0])
        s.reset()
        lb, ub = s.bounds("x")
        np.testing.assert_array_equal(lb, [-1.0, -1.0])
        np.testing.assert_array_equal(ub, [1.0, 1.0])
        np.testing.assert_array_equal(s.initial_guess("x"), [0.5, 0.5])

    def test_x0_override_per_solve(self):
        """A non-convex cost: the start decides which minimum is found."""
        s = SolverBackend(verbose=False)
        x = s.add_variable("x", 1, lb=-3, ub=3, initial_guess=1.5)
        s.add_cost((x**2 - 1.0) ** 2)
        s.build()
        np.testing.assert_allclose(s.solve()["x"], [1.0], atol=1e-5)
        np.testing.assert_allclose(s.solve(x0={"x": -1.5})["x"], [-1.0], atol=1e-5)
        np.testing.assert_allclose(s.solve(x0=[-2.0])["x"], [-1.0], atol=1e-5)
        np.testing.assert_array_equal(s.initial_guess("x"), [1.5])    # stored guess untouched

    def test_wrong_length_x0_raises(self):
        s = self.make()
        s.build()
        with pytest.raises(ValueError, match="x0"):
            s.solve(x0=[1.0, 2.0, 3.0])
