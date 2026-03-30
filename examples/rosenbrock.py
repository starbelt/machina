"""
examples/rosenbrock.py
----------------------
Minimal example of using SolverBackend directly to solve the Rosenbrock problem.

    minimize  (1 - x)^2 + 100*(y - x^2)^2

The global minimum is at (x, y) = (1, 1) with f* = 0.
This is a classic nonlinear benchmark — the valley is easy to find but the
bottom is a long, narrow, banana-shaped ridge that challenges gradient methods.

Run from the project root:

    python examples/rosenbrock.py
"""

import casadi as ca
from machina.solver.SolverBackend import SolverBackend

# ---------------------------------------------------------------------------
# 1. Create the backend
# ---------------------------------------------------------------------------
backend = SolverBackend(solver_opts={
    'ipopt.print_level': 5,   # set to 0 to silence IPOPT output
    'print_time': True,
})

# ---------------------------------------------------------------------------
# 2. Register decision variables
# ---------------------------------------------------------------------------
x = backend.add_variable('x', 1, lb=-2.0, ub=2.0, initial_guess=0.0)
y = backend.add_variable('y', 1, lb=-2.0, ub=2.0, initial_guess=0.0)

# ---------------------------------------------------------------------------
# 3. Define and register the objective
# ---------------------------------------------------------------------------
cost = (1 - x)**2 + 100 * (y - x**2)**2
backend.add_cost(cost, name='rosenbrock')

# ---------------------------------------------------------------------------
# 4. Build the solver (assembles the NLP, creates the ca.nlpsol object)
# ---------------------------------------------------------------------------
backend.build()

# ---------------------------------------------------------------------------
# 5. Solve
# ---------------------------------------------------------------------------
result = backend.solve()

# ---------------------------------------------------------------------------
# 6. Inspect results
# ---------------------------------------------------------------------------
print(f"\n--- Rosenbrock result ---")
print(f"Converged:  {result.success}")
print(f"x*        = {result['x'][0]:.8f}  (expected 1.0)")
print(f"y*        = {result['y'][0]:.8f}  (expected 1.0)")
print(f"f*        = {result.f_opt:.2e}   (expected 0.0)")
print(f"Iterations: {result.stats['iter_count']}")
