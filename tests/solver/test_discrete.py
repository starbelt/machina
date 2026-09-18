"""
Discrete variables: native mixed-integer solve (bonmin), relaxation, and rebuild().

Decision Log #38: the thesis has unavoidable discrete choices (operating
point, number of satellites). ``native`` needs a plugin with integer support;
``relax`` always works and records which variables were relaxed.
"""

import casadi as ca
import numpy as np
import pytest

from machina.solver import SolverBackend, SolverError

pytestmark = pytest.mark.requires_casadi


def integer_toy(solver="ipopt", **kwargs):
    """
    minimize (x - 1.3)^2 + (y - 2.6)^2,  x and n integer-valued where flagged.

    Continuous optimum (1.3, 2.6); with x, y integer the optimum is (1, 3).
    """
    b = SolverBackend(solver=solver, verbose=False, **kwargs)
    x = b.add_variable("x", 1, lb=-5, ub=5, initial_guess=0.0, discrete=True)
    y = b.add_variable("y", 1, lb=-5, ub=5, initial_guess=0.0, discrete=True)
    z = b.add_variable("z", 1, lb=-5, ub=5, initial_guess=0.0)
    b.add_cost((x - 1.3) ** 2 + (y - 2.6) ** 2 + (z - 0.25) ** 2)
    b.add_constraint(x + y, lb=-10, ub=10, name="box")
    return b


class TestRelaxMode:

    def test_native_mode_on_a_continuous_plugin_raises(self):
        b = integer_toy("ipopt")
        with pytest.raises(SolverError, match="bonmin") as err:
            b.build()
        assert "['x', 'y']" in str(err.value)
        assert "relax" in str(err.value)

    def test_discrete_relax_mode(self):
        b = integer_toy("ipopt")
        b.build(discrete_mode="relax")
        res = b.solve()
        assert res.success
        assert res.relaxed_discrete == ("x", "y")
        np.testing.assert_allclose([res["x"][0], res["y"][0]], [1.3, 2.6], atol=1e-5)

    def test_continuous_problem_reports_nothing_relaxed(self):
        b = SolverBackend(verbose=False)
        x = b.add_variable("x", 1)
        b.add_cost((x - 1.0) ** 2)
        b.build(discrete_mode="relax")
        assert b.solve().relaxed_discrete == ()

    def test_unknown_discrete_mode_raises(self):
        b = integer_toy("ipopt")
        with pytest.raises(ValueError, match="discrete_mode"):
            b.build(discrete_mode="round")

    def test_scaled_discrete_variable_is_rejected(self):
        b = SolverBackend(verbose=False)
        with pytest.raises(ValueError, match="scale 1"):
            b.add_variable("n", 1, discrete=True, scale=10.0)

    def test_records_expose_the_flags_per_element(self):
        b = SolverBackend(verbose=False)
        b.add_variable("n", 3, discrete=[True, False, True])
        rec = b.variables()[0]
        np.testing.assert_array_equal(rec.discrete, [True, False, True])
        assert rec.is_discrete


@pytest.mark.requires_bonmin
class TestNativeBonmin:

    def test_discrete_requires_bonmin(self, capfd):
        b = integer_toy("bonmin")
        b.build()
        res = b.solve()
        capfd.readouterr()      # bonmin's NLP log lines cannot be silenced
        assert res.success
        assert res.plugin == "bonmin"
        assert res.relaxed_discrete == ()
        np.testing.assert_allclose([res["x"][0], res["y"][0]], [1.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(res["z"], [0.25], atol=1e-5)

    def test_relax_then_rebuild_native(self, capfd):
        """One registered problem, two targets: relaxed IPOPT, then bonmin."""
        b = integer_toy("ipopt", solver_opts={"ipopt.tol": 1e-9})
        b.build(discrete_mode="relax")
        relaxed = b.solve()
        assert relaxed.relaxed_discrete == ("x", "y")

        # The ipopt.* option is dropped on the plugin switch instead of
        # being handed to bonmin.
        b.rebuild(solver="bonmin", discrete_mode="native")
        res = b.solve(warm_start=relaxed)
        capfd.readouterr()
        assert res.success and res.plugin == "bonmin"
        np.testing.assert_allclose([res["x"][0], res["y"][0]], [1.0, 3.0], atol=1e-6)
        assert res.f_opt >= relaxed.f_opt - 1e-9

    def test_one_hot_selection_of_an_operating_point(self, capfd):
        """
        The thesis pattern: pick one row of a measured table. Latency and power
        are table lookups ``t . y`` and ``w . y`` with ``sum(y) == 1``.
        """
        latency = np.array([9.0, 4.0, 2.0, 1.0])     # s
        power = np.array([5.0, 10.0, 20.0, 45.0])    # W
        b = SolverBackend(solver="bonmin", verbose=False)
        y = b.add_variable("pick", 4, lb=0, ub=1, initial_guess=0.25, discrete=True)
        b.add_cost(ca.dot(ca.DM(latency), y), name="latency")
        b.add_equality(y[0] + y[1] + y[2] + y[3] - 1.0, name="one_hot")
        b.add_constraint(ca.dot(ca.DM(power), y), ub=25.0, name="power_budget")
        b.build()
        res = b.solve()
        capfd.readouterr()
        assert res.success
        np.testing.assert_allclose(res["pick"], [0, 0, 1, 0], atol=1e-6)
        np.testing.assert_allclose(res.constraint("power_budget").value, [20.0], atol=1e-6)


class TestRebuild:

    def test_rebuild_before_build_raises(self):
        b = integer_toy("ipopt")
        with pytest.raises(RuntimeError, match="build"):
            b.rebuild()

    def test_rebuild_changes_options_without_reregistering(self):
        b = SolverBackend(verbose=False, solver_opts={"ipopt.max_iter": 1})
        x = b.add_variable("x", 2, initial_guess=[-1.2, 1.0])
        b.add_cost((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)
        b.build()
        assert not b.solve().success
        b.rebuild(solver_opts={"ipopt.max_iter": 500}, opts={"expand": True})
        res = b.solve()
        assert res.success
        np.testing.assert_allclose(res["x"], [1.0, 1.0], atol=1e-5)

    def test_stats_are_reset_by_rebuild(self):
        b = SolverBackend(verbose=False)
        x = b.add_variable("x", 1)
        b.add_cost(x**2)
        b.build()
        b.solve()
        b.rebuild()
        assert not b.is_solved
        with pytest.raises(RuntimeError, match="solve"):
            b.stats()
