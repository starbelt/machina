"""
Variable and constraint scaling.

Callers work in physical units throughout; the solver sees ``x / scale`` and
``g / scale``. Measured motivation (2026-09-18): the single-satellite coverage
problem mixes p ~ 7000 km with elements of order 1 and altitude constraints of
order 7000^2. Unscaled and strict it ends in ``Invalid_Number_Detected``; the
April workaround (``acceptable_tol=1e-2``) stops early at f = -0.126. Scaled,
the same defaults converge strictly to f = -0.164 with the apogee bound active.
"""

import math
import warnings

import casadi as ca
import numpy as np
import pytest

from machina.agents import SingleSatCoverage
from machina.compiler.compiler_stub import CompilerStub
from machina.solver import SolverBackend

pytestmark = pytest.mark.requires_casadi


def badly_scaled(scale_x=1.0, scale_g=1.0):
    """Active upper bound on a, an active inequality, and one parameter."""
    b = SolverBackend(verbose=False, solver_opts={"ipopt.tol": 1e-10})
    a = b.add_variable("a", 1, lb=0.0, ub=3000.0, initial_guess=1000.0, scale=scale_x)
    v = b.add_variable("v", 2, lb=-5.0, ub=5.0, initial_guess=1.0)
    p = b.add_parameter("p", 1, value=1.0)
    b.add_cost((a - 5000.0 * p) ** 2 / 1e6 + (v[0] - 2) ** 2 + (v[1] + 1) ** 2 + v[0] * v[1],
               name="cost")
    b.add_constraint(1e4 * (v[0] + v[1]), lb=1e4, name="sum", scale=scale_g)
    b.add_constraint(a * v[0], ub=4000.0, name="product")
    return b


class TestScalingIsTransparent:

    def test_unit_scale_builds_the_unscaled_nlp(self):
        """scale == 1 everywhere must hand nlpsol the registered symbols themselves."""
        b = badly_scaled()
        b.build()
        nlp = b.nlp_expressions()
        assert b.nlp_function().size_in("x0") == nlp["x"].shape
        assert not b.is_scaled

    def test_results_and_multipliers_are_in_physical_units(self):
        ref = badly_scaled()
        ref.build()
        r0 = ref.solve()

        scaled = badly_scaled(scale_x=1000.0, scale_g=1e4)
        scaled.build()
        r1 = scaled.solve()

        assert r0.success and r1.success
        np.testing.assert_allclose(r1["a"], r0["a"], rtol=1e-7)
        np.testing.assert_allclose(r1["v"], r0["v"], atol=1e-7)
        np.testing.assert_allclose(r1.f_opt, r0.f_opt, rtol=1e-9)
        np.testing.assert_allclose(r1.g_opt, r0.g_opt, rtol=1e-7)
        np.testing.assert_allclose(r1.lam_x, r0.lam_x, atol=1e-8)
        np.testing.assert_allclose(r1.lam_g, r0.lam_g, atol=1e-8)
        np.testing.assert_allclose(r1.lam_p, r0.lam_p, rtol=1e-6)
        # The upper bound on a really is active, so the comparison means something.
        np.testing.assert_allclose(r1["a"], [3000.0], rtol=1e-6)
        assert r1.bound_multiplier("a")[0] > 1e-4

    def test_bounds_guess_and_extract_stay_physical(self):
        b = badly_scaled(scale_x=1000.0, scale_g=1e4)
        b.build()
        res = b.solve()
        lb, ub = b.bounds("a")
        assert (lb[0], ub[0]) == (0.0, 3000.0)
        assert b.initial_guess("a")[0] == 1000.0
        np.testing.assert_allclose(b.extract(res.raw_sol, "a"), res["a"])
        np.testing.assert_array_equal(b.variables()[0].scale, [1000.0])

    def test_fix_and_set_bounds_are_converted(self):
        b = badly_scaled(scale_x=1000.0)
        b.build()
        b.fix("a", 1234.5)
        np.testing.assert_allclose(b.solve()["a"], [1234.5], rtol=1e-10)
        b.unfix("a")
        b.set_bounds("a", ub=2000.0)
        np.testing.assert_allclose(b.solve()["a"], [2000.0], rtol=1e-6)

    def test_scaled_problem_with_a_rootfinder_and_expand(self):
        """The Function wrap must survive a rootfinder call node, with and without expand."""
        y, u = ca.MX.sym("y"), ca.MX.sym("u")
        rf = ca.rootfinder("rf", "newton", {"x": y, "p": u, "g": y**3 + y - u})
        for expand in (False, True):
            b = SolverBackend(verbose=False, solver_opts={"expand": expand})
            v = b.add_variable("v", 1, initial_guess=0.5)
            w = b.add_variable("w", 1, initial_guess=100.0, scale=100.0)
            root = rf(x0=1.0, p=v)["x"]
            b.add_cost((root - 1.0) ** 2 + (w - 300.0) ** 2 / 1e4)
            b.build()
            res = b.solve()
            assert res.success
            np.testing.assert_allclose([res["v"][0], res["w"][0]], [2.0, 300.0], rtol=1e-5)


class TestCoverageProblem:
    """The real motivating case, through the compiler stub's ``scale`` overrides."""

    MU, R_EARTH = 398600.4418, 6378.137

    def compile(self, solver_opts, scaled):
        i, raan = math.radians(51.6), math.radians(240.0)
        h0, k0 = math.tan(i / 2) * math.cos(raan), math.tan(i / 2) * math.sin(raan)
        agent = SingleSatCoverage("sat", {
            "n_sample_points": 24,
            "ground_target": {"lat_deg": 38.9, "lon_deg": -77.0},
            "coverage": {"min_elevation_deg": 10.0, "sigmoid_k": 20.0},
            "altitude_bounds": {"perigee_min_km": 200.0, "apogee_max_km": 1600.0},
            "constants": {"mu": self.MU, "R_earth": self.R_EARTH},
        })
        compiler = CompilerStub(solver=SolverBackend(verbose=False, solver_opts=solver_opts))
        compiler.add_agent(agent)
        R = self.R_EARTH
        overrides = {
            "sat/orbital/p": {"value": R + 500.0, "lb": R + 200.0, "ub": R + 1600.0},
            "sat/orbital/f": {"value": 0.01, "lb": -0.30, "ub": 0.30},
            "sat/orbital/g": {"value": 0.0, "lb": -0.30, "ub": 0.30},
            "sat/orbital/h": {"value": h0, "lb": -1.5, "ub": 1.5},
            "sat/orbital/k": {"value": k0, "lb": -1.5, "ub": 1.5},
        }
        if scaled:
            overrides["sat/orbital/p"]["scale"] = 7000.0
            overrides["sat/perigee_altitude"] = {"scale": 7000.0**2}
            overrides["sat/apogee_altitude"] = {"scale": 7000.0**2}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            compiler.compile(overrides=overrides)
        compiler.add_cost(-compiler.resolve("sat", "coverage/total").symbol, name="neg_coverage")
        compiler.build_solver()
        return compiler

    def test_april_recipe_is_unchanged_at_unit_scale(self):
        """Regression oracle for the unscaled path: same stop point as April."""
        compiler = self.compile({"ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-2,
                                 "ipopt.acceptable_iter": 3}, scaled=False)
        res = compiler.solve()
        assert res.status == "Solved_To_Acceptable_Level"
        assert res.iterations == 13
        np.testing.assert_allclose(res.f_opt, -0.12583632, atol=1e-7)

    def test_unscaled_strict_solve_fails(self):
        res = self.compile({}, scaled=False).solve()
        assert not res.success
        assert res.status == "Invalid_Number_Detected"

    def test_scaled_strict_solve_converges_to_the_apogee_bound(self):
        res = self.compile({}, scaled=True).solve()
        assert res.status == "Solve_Succeeded"
        np.testing.assert_allclose(res.f_opt, -0.16394, atol=1e-4)
        np.testing.assert_allclose(res["sat/orbital/p"], [self.R_EARTH + 1600.0], rtol=1e-6)
        assert res.bound_multiplier("sat/orbital/p")[0] > 0.0
        # Reported in physical units (km^2). Feasible to the solver tolerance,
        # which applies to the scaled row, hence the division.
        apogee = res.constraint("sat/apogee_altitude")
        assert apogee.value[0] / 7000.0**2 >= -1e-7
