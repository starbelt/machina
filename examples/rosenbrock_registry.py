"""
examples/rosenbrock_registry.py
--------------------------------
The Rosenbrock problem solved through the factory registry.

    minimize  (a - xy[0])^2 + b*(xy[1] - xy[0]^2)^2

Global minimum is at xy* = [a, a^2] = [1, 1] with f* = 0.

Compare with examples/rosenbrock.py, which writes the objective expression
by hand directly into the solver backend (Layer 1 only). This example instead:

  1. Looks up the factory by name from the function registry.
  2. Calls the factory with configuration parameters to get a FunctionDescriptor.
  3. Calls the descriptor with MX variables to get an MX cost expression.
  4. Registers that expression with the solver backend as usual.

The mathematical result is identical — the registry is purely additive.
What it changes is the assembly pattern: functions are named, reusable, and
discoverable. It is the pattern a component's build() uses to compose the
physics it declares (examples/fleet_budget.py writes a component).

Run from the project root:
    python examples/rosenbrock_registry.py
"""

import matplotlib.pyplot as plt

from machina.library import registry
from machina.solver.backend import SolverBackend
from machina.viz import build_nlp_graph, draw_nlp_graph

# ---------------------------------------------------------------------------
# 1. Look up and configure the function
# ---------------------------------------------------------------------------
# registry.get() returns the factory callable registered under this name.
# Calling the factory with keyword arguments bakes those parameters into the
# ca.Function — they become numeric constants, not optimized variables.

rosenbrock = registry.get('cost.rosenbrock')(a=1.0, b=100.0)

# The descriptor carries metadata extracted directly from the ca.Function.
# Useful for debugging wiring before building the NLP.
print(f"Function: {rosenbrock}")
print(f"  Description: {rosenbrock.description}")
print()

# ---------------------------------------------------------------------------
# 2. Set up the solver backend and rent a decision variable
# ---------------------------------------------------------------------------
solver = SolverBackend(solver_opts={
    'ipopt.print_level': 5,
    'print_time': True,
})

# The variable is registered as a 2-vector because cost.rosenbrock takes
# a single (2, 1) input named 'xy'. Its two elements are the x and y
# from the original Rosenbrock problem — packed together as a convention
# of this specific factory.
xy = solver.add_variable('xy', 2, lb=-2.0, ub=2.0, initial_guess=0.0)

# ---------------------------------------------------------------------------
# 3. Call the descriptor with the MX variable
# ---------------------------------------------------------------------------
# The descriptor validates that 'xy' has the expected shape (2, 1) before
# delegating to the underlying ca.Function. The return value is an MX
# expression that CasADi can differentiate automatically.
cost_expr = rosenbrock(xy=xy)

# Alternatively, positional calling works too:
#   cost_expr = rosenbrock(xy)

solver.add_cost(cost_expr, name='rosenbrock')

# ---------------------------------------------------------------------------
# 4. Build and solve — identical to the Layer 1 approach from here on
# ---------------------------------------------------------------------------
solver.build()
result = solver.solve()

# ---------------------------------------------------------------------------
# 5. Inspect results
# ---------------------------------------------------------------------------
print("\n--- Rosenbrock result (via the factory registry) ---")
print(f"Converged:  {result.success}")
print(f"xy*[0]    = {result['xy'][0]:.8f}  (expected 1.0)")
print(f"xy*[1]    = {result['xy'][1]:.8f}  (expected 1.0)")
print(f"f*        = {result.f_opt:.2e}   (expected 0.0)")
print(f"Iterations: {result.stats['iter_count']}")

# ---------------------------------------------------------------------------
# 6. Visualize the NLP wiring
# ---------------------------------------------------------------------------
G = build_nlp_graph(solver)
draw_nlp_graph(G, title='Rosenbrock NLP (via the factory registry)')
plt.tight_layout()
plt.show()
