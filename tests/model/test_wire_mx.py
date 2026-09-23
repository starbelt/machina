"""
Wiring the graph with MX leaves from the solver backend, and solving it.

``wire()`` is the one seam between the model layer and Layer 1, and this file
is the model layer's own consumer of it. :class:`machina.compiler.Problem`
now does this for real, with the role and value precedence rules on top;
``compile_problem`` below is the same walk without them -- create a backend
leaf per quantity, wire, register every ``h`` as a constraint and every ``J``
as a cost term, build -- so the seam stays testable on its own.

The model is a two-satellite power budget with a known closed-form optimum,
so the assertions pin numbers rather than shapes:

    maximise  50 (d_a + d_b) - 0.1 (d_a^2 + d_b^2)
    subject to  100 (d_a + d_b) <= limit,  0 <= d <= 1

At ``limit = 150`` the constraint binds, symmetry gives ``d_a = d_b = 0.75``,
the objective is ``-74.8875`` and ``d f*/d limit = -0.4985``.
"""

import casadi as ca
import numpy as np
import pytest

from machina.model import (
    Aggregation,
    Builder,
    Component,
    Constraint,
    Cost,
    Declaration,
    ModelError,
    Quantity,
    Role,
    Scope,
    SignalRegistry,
)
from machina.solver import SolverBackend

pytestmark = pytest.mark.requires_casadi


def make_registry() -> SignalRegistry:
    reg = SignalRegistry()
    reg.declare("power", 1, "W", aggregation=Aggregation.SUM, doc="Electrical load")
    reg.declare("rate", 1, "1/s", aggregation=Aggregation.SUM, doc="Useful data rate")
    return reg


class Payload(Component):
    """``power = 100 duty``, ``rate = 50 duty``, with a small effort penalty."""

    def declare(self):
        return Declaration(
            quantities=(Quantity("duty", unit="1", lb=0.0, ub=1.0, default=0.1,
                                 doc="Duty cycle", provenance="A", source="test fixture"),),
            produces=("power", "rate"),
            costs=(Cost("effort", weight=0.1, doc="Squared duty cycle"),),
        )

    def build(self, helpers):
        duty = ca.SX.sym("duty")
        return {
            "g": ca.Function("payload_g", [duty], [100.0 * duty, 50.0 * duty],
                             ["duty"], ["power", "rate"]),
            "J": ca.Function("payload_J", [duty], [duty ** 2], ["duty"], ["effort"]),
        }


class FleetBudget(Component):
    """Caps the fleet's total power and scores its total rate.

    It reads both satellites by absolute path, which is the ordinary way one
    component couples two instances.
    """

    def declare(self):
        return Declaration(
            algebraic=(("power_a", "/a/power"), ("power_b", "/b/power"),
                       ("rate_a", "/a/rate"), ("rate_b", "/b/rate")),
            quantities=(Quantity("limit", unit="W", role=Role.PARAMETER, default=150.0,
                                 doc="Fleet power cap", provenance="D", source="bus spec"),),
            constraints=(Constraint("margin", lb=0.0, doc="limit - total power >= 0"),),
            costs=(Cost("neg_rate", weight=1.0, doc="Negated fleet data rate"),),
        )

    def build(self, helpers):
        power_a, power_b = ca.SX.sym("power_a"), ca.SX.sym("power_b")
        rate_a, rate_b = ca.SX.sym("rate_a"), ca.SX.sym("rate_b")
        limit = ca.SX.sym("limit")
        return {
            "h": ca.Function("budget_h", [power_a, power_b, limit],
                             [limit - (power_a + power_b)],
                             ["power_a", "power_b", "limit"], ["margin"]),
            "J": ca.Function("budget_J", [rate_a, rate_b], [-(rate_a + rate_b)],
                             ["rate_a", "rate_b"], ["neg_rate"]),
        }


def fleet(registry=None):
    return Builder([Scope("a", [Payload()]), Scope("b", [Payload()]), FleetBudget()],
                   registry=registry or make_registry()).declare()


def compile_problem(builder, *, values=None, solver="ipopt"):
    """What ``compiler.Problem.compile`` does, minus the value precedence.

    The model-layer-only check: the compiler is the only thing that calls
    ``backend.add_*``; components and the builder never do.
    """
    values = dict(values or {})
    backend = SolverBackend(solver, verbose=False)
    leaves = {}
    for path, quantity, owner in builder.quantities():
        role = quantity.default_role if quantity.role is Role.FLEXIBLE else quantity.role
        value = values.get(path, quantity.default)
        if role is Role.PARAMETER:
            leaves[path] = backend.add_parameter(
                path, quantity.shape, value=value, unit=quantity.unit, doc=quantity.doc,
                provenance=quantity.provenance, source=quantity.source, component=owner)
        else:
            leaves[path] = backend.add_variable(
                path, quantity.shape, lb=quantity.lb, ub=quantity.ub, initial_guess=value,
                discrete=role is Role.DISCRETE, unit=quantity.unit, doc=quantity.doc,
                provenance=quantity.provenance, source=quantity.source, component=owner)

    wired = builder.wire(leaves)

    for constraint in wired.constraints:
        backend.add_constraint(constraint.expr, lb=constraint.lb, ub=constraint.ub,
                               name=constraint.path, doc=constraint.doc,
                               component=constraint.component)
    for cost in wired.costs:
        backend.add_cost(cost.expr, name=cost.path, weight=cost.weight, doc=cost.doc,
                         component=cost.component)
    backend.build()
    return backend, wired


class TestWireIsSymbolTypeAgnostic:

    def test_wiring_with_mx_leaves_yields_mx_expressions(self):
        builder = fleet()
        leaves = {path: ca.MX.sym(path.replace("/", "_"))
                  for path in builder.quantity_order}
        wired = builder.wire(leaves)
        assert isinstance(wired.values["a/power"], ca.MX)
        assert isinstance(wired.constraints[0].expr, ca.MX)
        assert isinstance(wired.costs[0].expr, ca.MX)

    def test_the_same_components_wire_with_sx_leaves(self):
        builder = fleet()
        leaves = {path: ca.SX.sym(path.replace("/", "_")) for path in builder.quantity_order}
        assert isinstance(builder.wire(leaves).values["a/power"], ca.SX)

    def test_mx_and_sx_wiring_agree_numerically(self):
        builder = fleet()
        args_mx = [ca.MX.sym(p.replace("/", "_")) for p in builder.quantity_order]
        args_sx = [ca.SX.sym(p.replace("/", "_")) for p in builder.quantity_order]
        mx = builder.wire(dict(zip(builder.quantity_order, args_mx)))
        sx = builder.wire(dict(zip(builder.quantity_order, args_sx)))
        f_mx = ca.Function("f_mx", args_mx, [mx.constraints[0].expr])
        f_sx = ca.Function("f_sx", args_sx, [sx.constraints[0].expr])
        np.testing.assert_allclose(float(f_mx(0.3, 0.4, 150.0)), float(f_sx(0.3, 0.4, 150.0)))

    def test_a_missing_leaf_names_what_is_missing(self):
        builder = fleet()
        with pytest.raises(ModelError, match=r"a/duty"):
            builder.wire({"b/duty": ca.MX.sym("b"), "limit": ca.MX.sym("l")})

    def test_the_missing_leaf_error_says_algebraic_signals_are_computed(self):
        builder = fleet()
        with pytest.raises(ModelError, match="algebraic signals are computed, not supplied"):
            builder.wire({})

    def test_an_extra_leaf_is_a_typo_and_is_refused(self):
        builder = fleet()
        leaves = {path: ca.MX.sym(path.replace("/", "_")) for path in builder.quantity_order}
        leaves["a/dutyy"] = ca.MX.sym("typo")
        with pytest.raises(ModelError, match="Check for a typo in the instance path"):
            builder.wire(leaves)


class TestTheCompiledProblemSolves:

    def test_the_optimum_is_the_closed_form_one(self):
        backend, _ = compile_problem(fleet())
        result = backend.solve()
        assert result.success
        np.testing.assert_allclose(result["a/duty"], [0.75], rtol=1e-6)
        np.testing.assert_allclose(result["b/duty"], [0.75], rtol=1e-6)
        np.testing.assert_allclose(result.f_opt, -74.8875, rtol=1e-8)

    def test_the_power_cap_is_active_and_priced(self):
        backend, _ = compile_problem(fleet())
        result = backend.solve()
        margin = result.constraint("margin")
        np.testing.assert_allclose(margin.value, [0.0], atol=1e-6)
        assert margin.active
        assert abs(margin.multiplier[0]) > 1e-6

    def test_the_parameter_sensitivity_matches_the_derivative_on_paper(self):
        """d f*/d limit = -1/2 + limit / 100000, which is -0.4985 at 150 W."""
        backend, _ = compile_problem(fleet())
        result = backend.solve()
        np.testing.assert_allclose(result.sensitivity("limit"), [-0.4985], rtol=1e-4)

    def test_the_sensitivity_agrees_with_a_finite_difference(self):
        backend, _ = compile_problem(fleet())
        base = backend.solve()
        backend.set_parameter("limit", 151.0)
        bumped = backend.solve()
        np.testing.assert_allclose(base.sensitivity("limit"),
                                   [bumped.f_opt - base.f_opt], rtol=1e-3)

    def test_the_cost_breakdown_is_per_declared_term_and_sums_to_the_objective(self):
        backend, _ = compile_problem(fleet())
        result = backend.solve()
        assert sorted(result.cost_terms) == ["a/effort", "b/effort", "neg_rate"]
        np.testing.assert_allclose(result.cost_terms["neg_rate"], -75.0, rtol=1e-6)
        np.testing.assert_allclose(result.cost_terms["a/effort"], 0.05625, rtol=1e-5)
        np.testing.assert_allclose(sum(result.cost_terms.values()), result.f_opt, rtol=1e-9)

    def test_a_tighter_cap_moves_the_optimum_without_a_rebuild(self):
        backend, _ = compile_problem(fleet())
        tight = backend.solve()
        backend.set_parameter("limit", 100.0)
        loose = backend.solve()
        np.testing.assert_allclose(loose["a/duty"], [0.5], rtol=1e-6)
        assert loose.f_opt > tight.f_opt

    def test_a_cap_that_does_not_bind_leaves_the_bound_active_instead(self):
        backend, _ = compile_problem(fleet(), values={"limit": 400.0})
        result = backend.solve()
        np.testing.assert_allclose(result["a/duty"], [1.0], rtol=1e-5)
        assert not result.constraint("margin").active


class TestMetadataSurvivesTheSeam:

    def test_quantity_metadata_reaches_the_backend_records(self):
        backend, _ = compile_problem(fleet())
        duty = next(v for v in backend.variables() if v.name == "a/duty")
        assert (duty.unit, duty.provenance, duty.source) == ("1", "A", "test fixture")
        assert duty.component == "a/payload"
        assert duty.doc == "Duty cycle"

    def test_a_parameter_role_becomes_a_backend_parameter(self):
        backend, _ = compile_problem(fleet())
        assert [p.name for p in backend.parameters()] == ["limit"]
        assert [v.name for v in backend.variables()] == ["a/duty", "b/duty"]
        assert backend.parameters()[0].provenance == "D"

    def test_constraint_and_cost_names_are_instance_paths(self):
        backend, _ = compile_problem(fleet())
        assert [c.name for c in backend.constraints()] == ["margin"]
        assert [c.name for c in backend.cost_terms()] == ["a/effort", "b/effort", "neg_rate"]

    def test_the_constraint_records_its_owning_component_and_doc(self):
        backend, _ = compile_problem(fleet())
        margin = backend.constraints()[0]
        assert margin.component == "fleet_budget"
        assert margin.doc == "limit - total power >= 0"

    def test_the_declared_cost_weight_reaches_the_backend(self):
        backend, _ = compile_problem(fleet())
        effort = next(c for c in backend.cost_terms() if c.name == "a/effort")
        assert effort.weight == 0.1

    def test_nothing_in_the_model_layer_touches_the_solver(self):
        """The builder gets leaves; it never sees the backend."""
        builder = fleet()
        assert not hasattr(builder, "solver") and not hasattr(builder, "_solver")
        leaves = {p: ca.MX.sym(p.replace("/", "_")) for p in builder.quantity_order}
        assert builder.wire(leaves).constraints[0].path == "margin"
