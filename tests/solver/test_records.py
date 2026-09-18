"""
Public records and accessors: the surface the viz tool, the compiler and the
future report and study packages read instead of backend internals.
"""

import casadi as ca
import numpy as np
import pytest

from machina.solver import (
    ConstraintRecord,
    CostRecord,
    ParameterRecord,
    SolverBackend,
    VariableRecord,
)

pytestmark = pytest.mark.requires_casadi


def documented_problem():
    b = SolverBackend(verbose=False)
    alt = b.add_variable("sat/alt", 1, lb=200e3, ub=1600e3, initial_guess=500e3, scale=1e6,
                         unit="m", doc="orbit altitude", component="sat", frame="eci")
    duty = b.add_variable("sat/duty", (2, 2), lb=0.0, ub=1.0, initial_guess=0.5,
                          unit="1", component="sat")
    budget = b.add_parameter("sat/power_budget", 1, value=45.0, unit="W",
                             provenance="D", source="System Constraints!C9", component="sat")
    gain = b.add_parameter("gain", (2, 2))
    b.add_cost(-ca.sum1(ca.vec(gain * duty)), name="goodput", doc="aggregate goodput",
               component="sat")
    b.add_cost(alt / 1e6, name="altitude_penalty", weight=0.1)
    b.add_constraint(40.0 * ca.sum1(ca.vec(duty)) - budget, ub=0.0, name="power",
                     doc="power budget", component="sat", scale=10.0)
    b.add_equality(duty[0, 1] - duty[1, 0], name="symmetry")
    return b


class TestRecords:

    def test_public_records(self):
        b = documented_problem()
        v_alt, v_duty = b.variables()
        assert isinstance(v_alt, VariableRecord)
        assert (v_alt.name, v_alt.shape, v_alt.slice) == ("sat/alt", (1, 1), slice(0, 1))
        assert (v_alt.unit, v_alt.doc, v_alt.component, v_alt.frame) == (
            "m", "orbit altitude", "sat", "eci")
        np.testing.assert_array_equal(v_alt.scale, [1e6])
        assert (v_duty.shape, v_duty.slice, v_duty.numel) == ((2, 2), slice(1, 5), 4)
        assert v_duty.lb.shape == (2, 2) and v_duty.x0.shape == (2, 2)
        assert not v_duty.is_discrete and not v_duty.is_fixed

        p_budget, p_gain = b.parameters()
        assert isinstance(p_budget, ParameterRecord)
        assert (p_budget.provenance, p_budget.source, p_budget.unit) == (
            "D", "System Constraints!C9", "W")
        np.testing.assert_array_equal(p_budget.value, [45.0])
        assert p_gain.value is None and p_gain.slice == slice(1, 5)

        power, symmetry = b.constraints()
        assert isinstance(power, ConstraintRecord)
        assert (power.name, power.index, power.slice, power.n_rows) == ("power", 0, slice(0, 1), 1)
        assert (power.doc, power.component) == ("power budget", "sat")
        assert not power.is_equality and symmetry.is_equality
        np.testing.assert_array_equal(power.scale, [10.0])

        goodput, penalty = b.cost_terms()
        assert isinstance(goodput, CostRecord)
        assert (goodput.name, goodput.weight, penalty.weight) == ("goodput", 1.0, 0.1)
        assert goodput.weighted is goodput.expr

    def test_orders_sizes_and_lookups(self):
        b = documented_problem()
        assert b.variable_order() == ["sat/alt", "sat/duty"]
        assert b.parameter_order() == ["sat/power_budget", "gain"]
        assert (b.n_x, b.n_p, b.n_g) == (5, 5, 2)
        assert b.slice_of("gain") == slice(1, 5)
        assert b.shape_of("sat/duty") == (2, 2)
        assert b.symbol_of("sat/duty").shape == (2, 2)
        assert b.has("gain") and not b.has("power")       # constraints are a separate namespace

    def test_records_are_snapshots_of_current_values(self):
        b = documented_problem()
        before = b.variables()[1]
        b.set_bounds("sat/duty", ub=0.25)
        after = b.variables()[1]
        np.testing.assert_array_equal(before.ub, np.ones((2, 2)))
        np.testing.assert_array_equal(after.ub, np.full((2, 2), 0.25))
        with pytest.raises(AttributeError):
            after.name = "renamed"      # frozen

    def test_dependencies_resolve_to_registered_names(self):
        """What the report needs: ca.symvar over a record, mapped back to names."""
        b = documented_problem()
        power = b.constraints()[0]
        names = [s.name() for s in ca.symvar(power.expr)]
        assert names == ["sat/duty", "sat/power_budget"]
        assert all(b.has(n) for n in names)

    def test_nlp_expressions_are_physical_and_available_before_build(self):
        b = documented_problem()
        nlp = b.nlp_expressions()
        assert nlp["x"].shape == (5, 1) and nlp["p"].shape == (5, 1)
        assert nlp["g"].shape == (2, 1) and nlp["f"].shape == (1, 1)
        f = ca.Function("f", [nlp["x"], nlp["p"]], [nlp["g"]])
        x = np.array([500e3, 0.5, 0.5, 0.5, 0.5])
        g = np.asarray(f(x, np.array([45.0, 1, 1, 1, 1]))).ravel()
        np.testing.assert_allclose(g, [40.0 * 2.0 - 45.0, 0.0])     # unscaled

    def test_nlp_function_requires_build(self):
        b = documented_problem()
        with pytest.raises(RuntimeError, match="build"):
            b.nlp_function()

    def test_registration_order_is_the_layout(self):
        """Deterministic ordering: layout follows registration, never a sort or a set."""
        b = SolverBackend(verbose=False)
        for name in ("zeta", "alpha", "mid"):
            b.add_variable(name, 2)
        assert b.variable_order() == ["zeta", "alpha", "mid"]
        assert [r.slice.start for r in b.variables()] == [0, 2, 4]
