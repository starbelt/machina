"""
examples/flyby_goodput.py
--------------------------
Simplified flyby goodput optimization — wired by hand on a SolverBackend, without Problem.

Problem
-------
A flyby satellite observes 3 weather data products during a single pass.
Each product has a fixed observation delay (determined by orbital geometry
and sensor parameters) and a processing delay that we can choose. All three
products share a compute resource: the total processing time available
during the pass is limited to 240 seconds.

The satellite's mission is to maximize the aggregate goodput — a
timeliness-weighted measure of how much value is delivered to users.
Earlier delivery (lower TTP) means higher goodput per product.

Decision variables
    proc_time[3] : processing time allocated to each product (seconds)
    Bounds: 0 <= proc_time[i] <= 300

Fixed parameters (baked into the problem, not optimized)
    obs_delay[3]  : observation delays [60, 90, 120] seconds
    weights[3]    : product importance [0.50, 0.30, 0.20]
    sigmoid k     : 0.08  (controls sharpness of the goodput deadline)
    sigmoid t50   : 200   (TTP at which goodput = 0.5, i.e. the soft deadline)
    compute_budget: 240 seconds

TTP model
    TTP_i = obs_delay_i + proc_time_i     (via util.ttp_computation)

Resource constraint
    proc_time[0] + proc_time[1] + proc_time[2] = 240  (equality)
    The full compute window must be allocated — the satellite is processing
    data throughout the contact pass and cannot leave time unused.

Objective
    maximize  sum_i( weights[i] * G(TTP_i) )
    where G(ttp) = 1 / (1 + exp(k * (ttp - t50)))

This example calls four registry factories (the swapc pack's and the generic ones):
    util.ttp_computation   — assembles TTP from delay components
    cost.sigmoid_goodput   — per-product timeliness value
    cost.aggregate_goodput — weighted sum across all products
    constraint.linear      — enforces the compute budget constraint

`Problem` with a component does this wiring; see fleet_budget.py.

Run from the project root:
    python examples/flyby_goodput.py
"""

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import machina.swapc  # noqa: F401
from machina.library import registry
from machina.solver.backend import SolverBackend
from machina.viz import build_nlp_graph, draw_nlp_graph

# ---------------------------------------------------------------------------
# Problem data
# (In a real problem these come from mission design; here they are illustrative)
# ---------------------------------------------------------------------------

OBS_DELAYS     = [60.0, 90.0, 120.0]  # seconds — fixed by orbital geometry
WEIGHTS        = [0.50, 0.30, 0.20]   # importance — higher = more critical product
N_PRODUCTS     = 3
COMPUTE_BUDGET = 240.0                 # seconds of processing time available
SIGMOID_K      = 0.08                  # 1/s — goodput decay rate
SIGMOID_T50    = 200.0                 # s   — TTP at which goodput = 0.5

# ---------------------------------------------------------------------------
# 1. Configure the factories
# ---------------------------------------------------------------------------

# TTP assembly function: ttp_i = obs_delay_i + proc_time_i (2 components)
ttp_func = registry.get('util.ttp_computation')(n_components=2)

# Per-product sigmoid goodput: G(ttp) = 1 / (1 + exp(k * (ttp - t50)))
# k and t50 are baked in here — they are properties of the product set,
# not decision variables.
sigmoid = registry.get('cost.sigmoid_goodput')(k=SIGMOID_K, t50=SIGMOID_T50)

# Aggregate goodput: weighted sum over all products.
# The sigmoid descriptor is passed in so aggregate_goodput can embed it
# symbolically — the same goodput model applies to all three products.
agg_goodput = registry.get('cost.aggregate_goodput')(
    per_product_func=sigmoid,
    n_products=N_PRODUCTS,
    weights=WEIGHTS,
)

# Linear constraint function for the compute budget.
# coefficients=[1,1,1] means it computes proc_time[0]+proc_time[1]+proc_time[2].
budget_constraint = registry.get('constraint.linear')(
    n=N_PRODUCTS,
    coefficients=[1.0, 1.0, 1.0],
)

# ---------------------------------------------------------------------------
# 2. Set up the solver backend and add the decision variables
# ---------------------------------------------------------------------------
solver = SolverBackend(solver_opts={
    'ipopt.print_level': 0,   # silence IPOPT — we print our own summary below
    'ipopt.sb': 'yes',        # and its startup banner
    'print_time': False,
})

# proc_time[i]: seconds of processing allocated to product i.
# All products start with an equal share as the initial guess.
proc_time = solver.add_variable(
    'proc_time',
    N_PRODUCTS,
    lb=0.0,
    ub=300.0,
    initial_guess=COMPUTE_BUDGET / N_PRODUCTS,
)

# ---------------------------------------------------------------------------
# 3. Build TTP expressions and assemble the objective
# ---------------------------------------------------------------------------

# For each product, TTP = obs_delay (fixed) + proc_time (decision variable).
# obs_delay is a numeric constant wrapped as an MX literal so it can be
# concatenated with the MX decision variable using ca.vertcat.
ttp_exprs = []
for i in range(N_PRODUCTS):
    delays_i = ca.vertcat(ca.MX(OBS_DELAYS[i]), proc_time[i])
    ttp_i = ttp_func(delays=delays_i)
    ttp_exprs.append(ttp_i)

# Stack individual TTP scalars into the (N_PRODUCTS, 1) vector that
# aggregate_goodput expects.
ttp_vector = ca.vertcat(*ttp_exprs)

# Add the objective: minimize the NEGATIVE aggregate goodput.
# (IPOPT minimizes; we want to maximize goodput, so we negate.)
solver.add_cost(-agg_goodput(ttp_vector=ttp_vector), name='neg_aggregate_goodput')

# ---------------------------------------------------------------------------
# 4. Apply the compute budget constraint
# ---------------------------------------------------------------------------
# Must allocate exactly COMPUTE_BUDGET seconds total.
# We write this as: sum(proc_time) - COMPUTE_BUDGET == 0 (equality).
solver.add_equality(
    budget_constraint(x=proc_time) - COMPUTE_BUDGET,
    name='compute_budget',
)

# ---------------------------------------------------------------------------
# 5. Build and solve
# ---------------------------------------------------------------------------
solver.build()
result = solver.solve()

# ---------------------------------------------------------------------------
# 6. Print results
# ---------------------------------------------------------------------------
proc_opt = result['proc_time']

print("=" * 55)
print("Flyby goodput optimization - results")
print("=" * 55)
print(f"Converged: {result.success}")
print(f"Objective (-goodput): {result.f_opt:.6f}")
print(f"Aggregate goodput:    {-result.f_opt:.6f}")
print()

# Per-product breakdown
print(f"{'Product':<10} {'Weight':>7} {'Obs (s)':>9} {'Proc (s)':>10} "
      f"{'TTP (s)':>9} {'Goodput':>9}")
print("-" * 55)
for i in range(N_PRODUCTS):
    ttp_val     = OBS_DELAYS[i] + proc_opt[i]
    goodput_val = 1.0 / (1.0 + np.exp(SIGMOID_K * (ttp_val - SIGMOID_T50)))
    print(f"  {i:<8} {WEIGHTS[i]:>7.2f} {OBS_DELAYS[i]:>9.1f} "
          f"{proc_opt[i]:>10.2f} {ttp_val:>9.2f} {goodput_val:>9.4f}")

print("-" * 55)
total_proc = sum(proc_opt)
weighted_sum = sum(WEIGHTS[i] * (1.0 / (1.0 + np.exp(SIGMOID_K * (
    OBS_DELAYS[i] + proc_opt[i] - SIGMOID_T50)))) for i in range(N_PRODUCTS))
print(f"  {'Total':<8} {sum(WEIGHTS):>7.2f} {'':>9} {total_proc:>10.2f} "
      f"{'':>9} {weighted_sum:>9.4f}")
print()
print(f"Compute budget:    {COMPUTE_BUDGET:.1f} s  "
      f"(allocated: {total_proc:.4f} s)")
print(f"Solver iterations: {result.stats['iter_count']}")

# ---------------------------------------------------------------------------
# 7. Visualize the NLP wiring
# ---------------------------------------------------------------------------
G = build_nlp_graph(solver)
draw_nlp_graph(G, title='Flyby goodput NLP')
plt.tight_layout()
if matplotlib.get_backend().lower() != "agg":
    plt.show()
