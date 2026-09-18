"""Plugin-agnostic behaviour: defaults, success flag, status fields, quiet mode."""

import casadi as ca
import numpy as np
import pytest

from machina.solver import SolverBackend, SolverError, plugins

pytestmark = pytest.mark.requires_casadi


def rosenbrock(solver, **kwargs):
    b = SolverBackend(solver=solver, verbose=False, **kwargs)
    x = b.add_variable("x", 1, lb=-5, ub=5, initial_guess=0.0)
    y = b.add_variable("y", 1, lb=-5, ub=5, initial_guess=0.0)
    b.add_cost((1 - x) ** 2 + 100 * (y - x**2) ** 2)
    b.add_constraint(x + y, lb=-10, ub=10, name="box")
    return b


def needs(plugin):
    return pytest.mark.skipif(not ca.has_nlpsol(plugin), reason=f"{plugin} plugin not available")


class TestOtherPlugins:
    """The April backend could not build these at all: it always sent ``ipopt.*`` options."""

    @needs("sqpmethod")
    def test_success_from_stats_for_sqpmethod(self):
        b = rosenbrock("sqpmethod")
        b.build()
        res = b.solve()
        assert res.success
        assert res.plugin == "sqpmethod"
        assert res.unified_status == "SOLVER_RET_SUCCESS"
        np.testing.assert_allclose([res["x"][0], res["y"][0]], [1.0, 1.0], atol=1e-5)

    @needs("fatrop")
    def test_fatrop_builds_and_reports_a_string_status(self):
        b = rosenbrock("fatrop")
        b.build()
        res = b.solve()
        assert res.success
        assert isinstance(res.status, str)       # fatrop reports an int return_status

    @needs("sqpmethod")
    def test_foreign_prefix_option_raises_solver_error(self):
        b = rosenbrock("sqpmethod", solver_opts={"ipopt.tol": 1e-8})
        with pytest.raises(SolverError, match="ipopt.tol"):
            b.build()

    def test_unavailable_plugin_raises_solver_error(self, capfd):
        b = rosenbrock("not_a_plugin")
        with pytest.raises(SolverError, match="not_a_plugin"):
            b.build()
        capfd.readouterr()      # CasADi prints its own plugin-search warning


class TestIpoptDefaultsAndStatus:

    def test_ipopt_defaults_are_the_april_defaults(self):
        merged = plugins.merge_options(plugins.spec_for("ipopt"), verbose=True,
                                       solver_opts=None, build_opts=None)
        assert merged == {
            "print_time": True, "ipopt.tol": 1e-8, "ipopt.max_iter": 2000,
            "ipopt.linear_solver": "mumps", "ipopt.mu_strategy": "adaptive",
            "ipopt.print_level": 5,
        }

    def test_option_precedence(self):
        spec = plugins.spec_for("ipopt")
        merged = plugins.merge_options(
            spec, verbose=False,
            solver_opts={"ipopt.tol": 1e-6, "ipopt": {"max_iter": 50}},
            build_opts={"ipopt.tol": 1e-4},
        )
        assert merged["ipopt.tol"] == 1e-4          # build opts beat ctor opts
        assert merged["ipopt.max_iter"] == 50       # nested spelling is flattened
        assert merged["ipopt.print_level"] == 0     # quiet layer
        assert merged["ipopt.linear_solver"] == "mumps"

    def test_explicit_print_level_beats_quiet(self):
        merged = plugins.merge_options(plugins.spec_for("ipopt"), verbose=False,
                                       solver_opts={"ipopt.print_level": 3}, build_opts=None)
        assert merged["ipopt.print_level"] == 3

    def test_unknown_plugin_gets_an_empty_spec(self):
        spec = plugins.spec_for("some_future_plugin")
        assert spec.defaults == {} and not spec.supports_discrete

    def test_acceptable_level_counts_as_success(self):
        """The MEE coverage recipe relies on 'Solved_To_Acceptable_Level'."""
        b = SolverBackend(verbose=False, solver_opts={
            "ipopt.tol": 1e-30, "ipopt.dual_inf_tol": 1e-30, "ipopt.compl_inf_tol": 1e-30,
            "ipopt.constr_viol_tol": 1e-30, "ipopt.acceptable_tol": 1e-4,
            "ipopt.acceptable_iter": 3, "ipopt.max_iter": 200,
        })
        x = b.add_variable("x", 2, initial_guess=[-1.2, 1.0])
        b.add_cost((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2 + 1e-3 * ca.exp(x[0]))
        b.build()
        res = b.solve()
        assert res.status == "Solved_To_Acceptable_Level"
        assert res.success

    def test_infeasible_reports_failure_with_a_status(self):
        b = SolverBackend(verbose=False)
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        b.add_constraint(x, lb=2.0, name="lo")
        b.add_constraint(x, ub=1.0, name="hi")
        b.build()
        res = b.solve()
        assert not res.success
        assert res.status == "Infeasible_Problem_Detected"
        assert res.max_constraint_violation > 0.1

    def test_iteration_limit_is_reported_as_limited(self):
        b = rosenbrock("ipopt", solver_opts={"ipopt.max_iter": 1})
        b.build()
        res = b.solve()
        assert not res.success
        assert res.unified_status == "SOLVER_RET_LIMITED"
        assert res.iterations == 1

    def test_result_carries_timing_and_iterations(self):
        b = rosenbrock("ipopt")
        b.build()
        res = b.solve()
        assert res.plugin == "ipopt"
        assert res.iterations > 0
        assert res.t_wall >= 0.0

    def test_quiet_mode_prints_nothing(self, capfd):
        b = rosenbrock("ipopt")
        b.build()
        b.solve()
        out, err = capfd.readouterr()
        assert out == "" and err == ""
