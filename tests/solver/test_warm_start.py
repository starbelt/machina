"""
Warm starting.

Measured 2026-09-18: duals only help when IPOPT's warm-start options are set,
and those are fixed when the nlpsol object is constructed. The backend
therefore keeps a second solver, created on the first call that supplies
duals. Cold solves never touch it.
"""

import casadi as ca
import numpy as np
import pytest

from machina.solver import SolverBackend

pytestmark = pytest.mark.requires_casadi

N = 30


def chain(n=N, **kwargs):
    """A smooth parametric NLP with active bounds and one constraint."""
    b = SolverBackend(verbose=False, **kwargs)
    x = b.add_variable("x", n, lb=-2.0, ub=0.9, initial_guess=0.0)
    p = b.add_parameter("p", 1, value=1.0)
    b.add_cost(ca.sumsqr(x - p) + ca.sum1(ca.exp(0.1 * x))
               + 10 * ca.sumsqr(x[1:] - x[:-1] ** 2), name="cost")
    b.add_constraint(ca.sum1(x), lb=-5.0, ub=5.0, name="sum")
    return b


class TestWarmStart:

    def test_warm_start_seeds(self):
        b = chain()
        b.build()
        base = b.solve()
        cold = b.solve(p_val={"p": 1.05})
        warm = b.solve(p_val={"p": 1.05}, warm_start=base)
        assert cold.success and warm.success
        assert not cold.warm_started and warm.warm_started
        assert warm.iterations < cold.iterations
        assert warm.iterations <= 4
        np.testing.assert_allclose(warm["x"], cold["x"], atol=1e-6)
        np.testing.assert_allclose(warm.f_opt, cold.f_opt, rtol=1e-8)

    def test_cold_solves_never_build_the_warm_solver(self):
        b = chain()
        b.build()
        b.solve()
        b.solve(x0={"x": np.full(N, 0.1)})       # a primal seed alone stays on the cold solver
        assert not b.has_warm_solver

    def test_stats_follow_the_solver_that_ran_last(self):
        b = chain()
        b.build()
        base = b.solve()
        warm = b.solve(p_val={"p": 1.1}, warm_start=base)
        assert b.stats()["iter_count"] == warm.iterations
        cold = b.solve(p_val={"p": 1.1})
        assert b.stats()["iter_count"] == cold.iterations

    def test_sweep_stays_warm(self):
        b = chain()
        b.build()
        prev = b.solve()
        cold_iters = prev.iterations
        for value in (1.05, 1.10, 1.20, 1.50):
            prev = b.solve(p_val={"p": value}, warm_start=prev)
            assert prev.success
            assert prev.iterations < cold_iters

    def test_explicit_duals_by_name(self):
        b = chain()
        b.build()
        base = b.solve()
        res = b.solve(p_val={"p": 1.05}, x0={"x": base["x"]},
                      lam_x0={"x": base.bound_multiplier("x")},
                      lam_g0={"sum": base.constraint("sum").multiplier})
        assert res.warm_started and res.iterations <= 4

    def test_result_from_another_layout_seeds_primal_by_name(self):
        """A rebuilt, differently shaped problem can still reuse matching variables."""
        small = chain()
        small.build()
        prev = small.solve()

        bigger = SolverBackend(verbose=False)
        x = bigger.add_variable("x", N, lb=-2.0, ub=0.9)
        y = bigger.add_variable("y", 1)
        p = bigger.add_parameter("p", 1, value=1.0)
        bigger.add_cost(ca.sumsqr(x - p) + ca.sum1(ca.exp(0.1 * x))
                              + 10 * ca.sumsqr(x[1:] - x[:-1] ** 2) + (y - 3.0) ** 2)
        bigger.add_constraint(ca.sum1(x), lb=-5.0, ub=5.0, name="sum")
        bigger.build()
        cold = bigger.solve()
        seeded = bigger.solve(warm_start=prev)
        assert seeded.success
        assert not seeded.warm_started          # layouts differ: duals are dropped
        assert seeded.iterations <= cold.iterations

    def test_scaled_problem_warm_starts_in_physical_units(self):
        b = SolverBackend(verbose=False)
        a = b.add_variable("a", 1, lb=0.0, ub=3000.0, initial_guess=1000.0, scale=1000.0)
        v = b.add_variable("v", 1, lb=-5.0, ub=5.0)
        p = b.add_parameter("p", 1, value=1.0)
        b.add_cost((a - 5000.0 * p) ** 2 / 1e6 + (v - 2) ** 2)
        b.add_constraint(a * v, ub=4000.0, name="product", scale=1e3)
        b.build()
        base = b.solve()
        cold = b.solve(p_val={"p": 1.02})
        warm = b.solve(p_val={"p": 1.02}, warm_start=base)
        assert warm.warm_started and warm.iterations < cold.iterations
        np.testing.assert_allclose(warm["a"], cold["a"], rtol=1e-6)
        np.testing.assert_allclose(warm.lam_g, cold.lam_g, atol=1e-6)

    def test_plugin_without_warm_options_reuses_the_cold_solver(self):
        if not ca.has_nlpsol("sqpmethod"):
            pytest.skip("sqpmethod plugin not available")
        b = chain(n=6, solver="sqpmethod")
        b.build()
        base = b.solve()
        res = b.solve(p_val={"p": 1.05}, warm_start=base)
        assert res.success
        assert not b.has_warm_solver
