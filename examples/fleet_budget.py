"""
examples/fleet_budget.py -- write your own component: copy this file to start one.

A component DECLARES what it owns (quantities), reads (signals, by name or by
path) and contributes (signals, constraints, cost terms), then BUILDS its
physics as SX ca.Functions whose arguments carry the declared names. It never
touches a solver: the Problem picks each quantity's role and value, wires the
graph in MX and owns the objective. Two payloads share a power cap here:

    minimise  -50 (d_a + d_b) + 0.1 (d_a^2 + d_b^2)   s.t.  100 (d_a + d_b) <= limit

At limit = 150 W: d_a = d_b = 0.75, f* = -74.8875, d f*/d limit = -0.4985 per W.
Run from the project root:  python examples/fleet_budget.py
"""

import casadi as ca

from machina.compiler import Problem
from machina.model import Component, Constraint, Cost, Declaration, Quantity, Role, Scope
from machina.model.signals import Aggregation, SignalRegistry

# A signal crosses a component boundary: declared once, with a shape and an SI unit. SUM is
# for budgets (producers add; an unproduced one is zero). This registry is the example's own.
reg = SignalRegistry()
reg.declare("power", 1, "W", aggregation=Aggregation.SUM, doc="Electrical load")
reg.declare("rate", 1, "1/s", aggregation=Aggregation.SUM, doc="Useful data rate")


class Payload(Component):
    """power = peak_power * duty, rate = 50 duty, plus a small effort penalty."""

    def declare(self):
        # No symbols here: the Problem reads this interface before anything is built.
        return Declaration(
            quantities=(
                # provenance: D documented, P physics, E estimate, A assumption, M measured.
                # FLEXIBLE: no opinion -- a variable by default; compile(roles=...) can make
                # it a swept parameter or a fixed constant without editing this file.
                Quantity("duty", unit="1", role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                         default=0.1, lb=0.0, ub=1.0, doc="Duty cycle",
                         provenance="A", source="initial guess"),
                # FIXED: substituted as a constant; the solver never sees it.
                Quantity("peak_power", unit="W", role=Role.FIXED, default=100.0,
                         doc="Power drawn at full duty", provenance="D",
                         source="payload datasheet"),
            ),
            produces=("power", "rate"),
            costs=(Cost("effort", weight=0.1, doc="Squared duty cycle"),),
        )

    def build(self, helpers):
        # Inputs are bound by the ca.Function input names (the list after the outputs); they
        # must match the declared names, and their order is free.
        duty, peak_power = ca.SX.sym("duty"), ca.SX.sym("peak_power")
        # The contract: keys "g" = produced signals, "h" = constraints, "J" = costs ("f" = state
        # derivatives, unused here). Each Function's outputs are matched by position to that
        # kind's declaration order (produces=("power", "rate") -> [power, rate]).
        return {
            "g": ca.Function("payload_g", [duty, peak_power], [peak_power * duty, 50.0 * duty],
                             ["duty", "peak_power"], ["power", "rate"]),
            "J": ca.Function("payload_J", [duty], [duty ** 2], ["duty"], ["effort"]),
        }


class FleetBudget(Component):
    """Caps the fleet's total power, reading both payloads by absolute path."""

    def declare(self):
        return Declaration(
            # (local name, path): "/a/power" reaches into scope a -- cross-instance coupling.
            algebraic=(("power_a", "/a/power"), ("power_b", "/b/power")),
            # PARAMETER: held fixed in the solve but symbolic, so the result can report
            # d f*/d limit (sensitivity).
            quantities=(Quantity("limit", unit="W", role=Role.PARAMETER, default=150.0,
                                 doc="Fleet power cap", provenance="D", source="bus spec"),),
            constraints=(Constraint("margin", lb=0.0, doc="limit - total power >= 0"),),
        )

    def build(self, helpers):
        power_a, power_b, limit = ca.SX.sym("power_a"), ca.SX.sym("power_b"), ca.SX.sym("limit")
        return {"h": ca.Function("budget_h", [power_a, power_b, limit],
                                 [limit - (power_a + power_b)],
                                 ["power_a", "power_b", "limit"], ["margin"])}


def main():
    # Scopes prefix instance paths (a/duty, b/power); FleetBudget sits at the root (limit).
    problem = Problem([Scope("a", [Payload()]), Scope("b", [Payload()]), FleetBudget()],
                      registry=reg, verbose=False)
    problem.compile()   # roles=, values=, overrides= change the problem, not the components
    print("roles:", {path: role.value for path, role in problem.roles.items()})

    # The problem owns the objective: no component declared the fleet's rate as a cost.
    problem.add_cost(-(problem.expr("a/rate").symbol + problem.expr("b/rate").symbol),
                     name="neg_rate")
    result = problem.build().solve()
    values = problem.evaluate(result)

    print(f"{result.status}: f* = {result.f_opt:.4f}   (closed form -74.8875)")
    for scope in ("a", "b"):
        print(f"  {scope}/power = {values[f'{scope}/power'].item():.2f} W "
              f"at duty {values[f'{scope}/duty'].item():.4f}")
    cap = result.constraint("margin")
    print(f"cap 'margin': active = {bool(cap.active.item())}, "
          f"shadow price (multiplier) = {cap.multiplier.item():+.4f} per W")
    sens = result.sensitivity("limit").item()     # the parameter that carries the cap
    print(f"d f*/d limit = {sens:+.4f} per W (closed form -0.4985): one more watt of cap "
          f"lowers f* by {-sens:.4f}, read off this solve")


if __name__ == "__main__":
    main()
