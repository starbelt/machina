"""
examples/pareto_frontier.py -- a two-objective trade study: sweep, frontier, switch points.

A processor runs one job per cycle. Its clock fraction sets both how long the
job takes and how much power it draws, and the job's result loses value as it
ages on a log-logistic curve (``cost.loglogistic_goodput`` from the swapc pack):

    latency = t_fixed + work / clock          power = p_idle + p_dyn clock
    value   = 1 / (1 + (latency / tau)^q)

The study asks for the whole trade, not one answer. ``sweep`` re-solves
"maximise value subject to power <= cap" over a grid of caps (the epsilon-
constraint trace of the front; warm-started point to point), and the Pareto
layer reads the resulting table:

    nondominated      which caps give a configuration nobody beats on both axes
    excluded          which rows are listed but not scored (the infeasible cap)
    normalize         0 = ideal, 1 = nadir on each axis, whatever its sense
    weighted_sweep    for J_w = w power^ + (1 - w) value^, the exact weights w at
                      which the best configuration switches, and the points no
                      weight ever picks (the front is S-shaped: value saturates at
                      both ends, so its low-power stretch is not convex)

Run from the project root:  python examples/pareto_frontier.py
"""

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt

import machina.swapc  # noqa: F401  (registers cost.loglogistic_goodput)
from machina.compiler import Problem
from machina.library import registry
from machina.model import Component, Constraint, Declaration, Quantity, Role
from machina.model.signals import SignalRegistry
from machina.study import Objective, excluded, nondominated, sweep, weighted_sweep

reg = SignalRegistry()
reg.declare("latency", 1, "s", doc="Time from the start of a cycle to a delivered result")
reg.declare("power", 1, "W", doc="Processor power draw")
reg.declare("value", 1, "1", doc="Timeliness value of the delivered result")

CAPS = (2.2, 2.5, 2.75, 3.0, 3.25, 3.5, 3.8, 4.2, 4.7, 5.5, 6.5, 8.0, 10.0, 13.0, 16.0, 20.0)


class Processor(Component):
    """latency = t_fixed + work / clock; power = p_idle + p_dyn clock (clock at fixed voltage)."""

    def declare(self):
        return Declaration(
            quantities=(
                Quantity("clock", unit="1", role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                         default=0.5, lb=0.02, ub=1.0, doc="Clock as a fraction of the maximum",
                         provenance="A", source="initial guess"),
                Quantity("work", unit="s", role=Role.FIXED, default=40.0,
                         doc="Job duration at full clock", provenance="E", source="example"),
                Quantity("t_fixed", unit="s", role=Role.FIXED, default=240.0,
                         doc="Latency terms the clock does not touch", provenance="E",
                         source="example"),
                Quantity("p_idle", unit="W", role=Role.FIXED, default=2.0,
                         doc="Power at zero clock", provenance="E", source="example"),
                Quantity("p_dyn", unit="W", role=Role.FIXED, default=18.0,
                         doc="Dynamic power at full clock", provenance="E", source="example"),
            ),
            produces=("latency", "power"),
        )

    def build(self, helpers):
        clock, work, t_fixed = ca.SX.sym("clock"), ca.SX.sym("work"), ca.SX.sym("t_fixed")
        p_idle, p_dyn = ca.SX.sym("p_idle"), ca.SX.sym("p_dyn")
        return {"g": ca.Function("processor_g", [clock, work, t_fixed, p_idle, p_dyn],
                                 [t_fixed + work / clock, p_idle + p_dyn * clock],
                                 ["clock", "work", "t_fixed", "p_idle", "p_dyn"],
                                 ["latency", "power"])}


class Timeliness(Component):
    """value = G(latency), the log-logistic goodput with q = 4, tau = 500 s."""

    def declare(self):
        return Declaration(algebraic=("latency",), produces=("value",))

    def build(self, helpers):
        goodput = registry.get("cost.loglogistic_goodput")(q=4.0, tau=500.0)
        latency = ca.SX.sym("latency")
        return {"g": ca.Function("timeliness_g", [latency], [goodput.function(latency)],
                                 ["latency"], ["value"])}


class PowerCap(Component):
    """power <= cap, with the cap a PARAMETER so a sweep re-solves without a rebuild."""

    def declare(self):
        return Declaration(
            algebraic=("power",),
            quantities=(Quantity("cap", unit="W", role=Role.PARAMETER, default=10.0,
                                 doc="Power cap", provenance="A", source="swept"),),
            constraints=(Constraint("headroom", lb=0.0, doc="cap - power >= 0"),),
        )

    def build(self, helpers):
        power, cap = ca.SX.sym("power"), ca.SX.sym("cap")
        return {"h": ca.Function("cap_h", [power, cap], [cap - power], ["power", "cap"],
                                 ["headroom"])}


def make_problem() -> Problem:
    problem = Problem([Processor(), Timeliness(), PowerCap()], registry=reg, verbose=False)
    problem.compile()
    problem.add_cost(-problem.expr("value").symbol, name="neg_value")
    return problem.build()


def main():
    table = sweep(make_problem(), {"cap": CAPS}, collect=("clock", "latency", "power", "value"))
    objectives = [Objective("power"), Objective("value", "max")]

    print(f"{len(table)} caps swept; {sum(table['warm_started'])} solves warm-started")
    for row, reason in excluded(table, objectives, feasible="success"):
        print(f"  listed, not scored: cap = {table['cap'][row]} W ({reason}; "
              f"status {table['status'][row]})")

    front = nondominated(table, objectives, feasible="success")
    print(f"non-dominated: {int(front.sum())} of {len(table)} rows")

    trade = weighted_sweep(table, objectives, feasible="success")
    print("J_w = w * power^ + (1 - w) * value^  (0 = ideal, 1 = nadir on each axis):")
    for segment in trade.segments:
        row = segment.rows[0]
        print(f"  w in [{segment.w_lo:.3f}, {segment.w_hi:.3f}]: cap {table['cap'][row]:5.2f} W"
              f" -> power {table['power'][row]:5.2f} W, value {table['value'][row]:.3f}")
    print("switch points:", ", ".join(f"{w:.3f}" for w in trade.switch_points))
    never = ", ".join(f"{table['cap'][row]}" for row in trade.unsupported)
    print(f"non-dominated but never picked by any weight (the non-convex stretch): caps {never}")

    power, value = table["power"], table["value"]
    fig, ax = plt.subplots(figsize=(6, 4))
    scored = [i for i in range(len(table)) if table["success"][i]]
    ax.plot([power[i] for i in scored], [value[i] for i in scored], "o-", color="0.6",
            label="swept caps")
    ax.plot([power[i] for i in trade.supported], [value[i] for i in trade.supported], "o",
            color="tab:blue", label="picked by some weight")
    ax.plot([power[i] for i in trade.unsupported], [value[i] for i in trade.unsupported], "o",
            color="tab:orange", label="never picked (non-convex)")
    ax.set_xlabel("power [W]")
    ax.set_ylabel("value")
    ax.set_title("Power vs timeliness value")
    ax.legend()
    fig.tight_layout()
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
