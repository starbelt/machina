"""
examples/least_squares.py
--------------------------
A simple least-squares problem using the Layer 2 block library.

    minimize  0.5 * ||Ax - b||^2

A (4×2) and b (4-vector) are fixed parameters; x (2-vector) is the decision
variable. A is registered as a matrix-valued parameter using the (rows, cols)
tuple form of add_parameter introduced in Layer 1 v2.

Compare with examples/rosenbrock_blocks.py for the basic block-library pattern.

Run from the project root:
    python examples/least_squares.py
"""

import matplotlib.pyplot as plt
import numpy as np

from machina.blocks import registry
from machina.solver.backend import SolverBackend
from machina.viz import build_nlp_graph, draw_nlp_graph

# ---------------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------------
A_data = np.array([[1.0, 1.0],
                   [1.0, 2.0],
                   [2.0, 1.0],
                   [2.0, 2.0]])
b_data = np.array([1.0, 2.0, 3.0, 4.0])

# ---------------------------------------------------------------------------
# 1. Look up and configure the block
# ---------------------------------------------------------------------------
least_squares = registry.get('cost.least_squares')(m=4, n=2)
print(f"Function: {least_squares}")
print(f"  Description: {least_squares.description}")
print()

# ---------------------------------------------------------------------------
# 2. Set up the solver and register symbols
# ---------------------------------------------------------------------------
solver = SolverBackend(solver_opts={
    'ipopt.print_level': 0,
    'print_time': False,
})

# A is a (4, 2) matrix parameter — values are supplied at solve() time.
# b is a length-4 vector parameter.
A = solver.add_parameter('A', (4, 2))
b = solver.add_parameter('b', 4)
x = solver.add_variable('x', 2, initial_guess=0.0)

# ---------------------------------------------------------------------------
# 3. Wire the block and build
# ---------------------------------------------------------------------------
cost_expr = least_squares(A=A, b=b, x=x)
solver.add_cost(cost_expr, name='least_squares_cost')
solver.build()

# ---------------------------------------------------------------------------
# 4. Solve — supply parameter values column-major (A) then flat (b)
# Matrix parameters are packed column-major by ca.vec, so A_data must be
# flattened with order='F' to match the layout in the parameter vector.
# ---------------------------------------------------------------------------
p_val = np.concatenate([A_data.flatten(order='F'), b_data])
result = solver.solve(p_val=p_val)

# ---------------------------------------------------------------------------
# 5. Inspect results
# ---------------------------------------------------------------------------
print("--- Least Squares result ---")
print(f"Converged:  {result.success}")
print(f"x*[0]     = {result['x'][0]:.8f}")
print(f"x*[1]     = {result['x'][1]:.8f}")
print(f"f*        = {result.f_opt:.2e}")
print(f"Iterations: {result.stats['iter_count']}")

# ---------------------------------------------------------------------------
# 6. Visualize the NLP wiring
# ---------------------------------------------------------------------------
G = build_nlp_graph(solver)
draw_nlp_graph(G, title='Least Squares NLP (via block library)')
plt.tight_layout()
plt.show()
