"""
examples/power_budget.py -- size the solar array and battery of a flyby compute spacecraft.

Two pack components (machina.swapc): ComputeLoad produces the processor's orbit-average
load power_load = E_scene / T_refresh, and PowerBudget reads it, adds its own front-end,
bus and communications loads, and requires the array to carry the total through the
longest GEO eclipse. Neither declares a cost; the problem minimises the array area, so the
margin is active and the array is the smallest that closes the balance:

    minimise  A   s.t.  P_gen(A) - P_tot >= 0,   P_gen(A) = S eta L_d A T_d / (T_e/X_e + T_d/X_d)

    A* = P_tot (T_e/X_e + T_d/X_d) / (T_d S eta L_d),   E_batt = P_tot T_e / (DoD eta_b)

The compute load here is 10 J per 300 s scene = 0.033 W, against 35 W of front-end, bus and
communications: the processor is about 0.1 % of the power budget, and the array is sized by
the bus. Compute moves the array only when E_scene grows by orders of magnitude or the
cadence shortens; d A*/d E_scene below says by how much per joule.
Run from the project root:  python examples/power_budget.py
"""

from machina.compiler import Problem
from machina.swapc import ComputeLoad, PowerBudget

# PowerBudget's documented defaults, for the closed form.
S, ETA, L_D, X_E, X_D = 1361.0, 0.30, 0.85, 0.65, 0.85
T_E, T_ORBIT, DOD, ETA_B = 4320.0, 86164.0, 0.8, 0.9


def main():
    # One processor and one budget, both at the root: the budget reads power_load by its bare
    # name. A second processor would go in its own Scope and be listed in PowerBudget(loads=...).
    problem = Problem([ComputeLoad(), PowerBudget()], verbose=False)
    problem.compile(values={"E_scene": 10.0})    # J per scene; T_refresh stays 300 s
    print("roles:", {path: role.value for path, role in problem.roles.items()
                     if role.value != "fixed"})

    # The problem owns the objective: size the array, nothing else.
    problem.add_cost(problem.expr("array_area").symbol, name="area")
    result = problem.build().solve()
    values = problem.evaluate(result)

    compute = values["power_load"].item()
    P_tot = compute + 5.0 + 20.0 + 10.0
    T_d = T_ORBIT - T_E
    per_watt = (T_E / X_E + T_d / X_D) / (T_d * S * ETA * L_D)
    area = values["array_area"].item()
    print(f"{result.status}: compute load {compute:.4f} W of {P_tot:.4f} W total "
          f"({100.0 * compute / P_tot:.2f} %)")
    print(f"  array area A* = {area:.6f} m^2   (closed form {P_tot * per_watt:.6f})")

    margin = result.constraint("power_margin")
    print(f"  'power_margin': active = {bool(margin.active.item())}, shadow price "
          f"(multiplier) = {margin.multiplier.item():+.6f} m^2 per W")

    energy = values["battery_energy"].item()
    print(f"  battery energy = {energy:.1f} J = {energy / 3600.0:.2f} Wh (Wh for display only; "
          f"closed form {P_tot * T_E / (DOD * ETA_B):.1f} J)")

    # d A* / d p for every parameter: -lam_p, read off this one solve.
    for path, unit in (("bus_power", "W"), ("comm_power", "W"), ("front_power", "W"),
                       ("E_scene", "J"), ("eclipse_duration", "s")):
        print(f"  d A*/d {path:<16} = {result.sensitivity(path).item():+.4e} m^2 per {unit}")
    print(f"  (closed form for any load in W: {per_watt:+.4e} m^2 per W)")


if __name__ == "__main__":
    main()
