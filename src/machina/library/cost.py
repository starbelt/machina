"""
Generic cost factories: ``cost.quadratic``, ``cost.rosenbrock`` and
``cost.least_squares``.

Importing this module registers the three factories; ``import machina``
imports it through :mod:`machina.library`.
"""

import casadi as ca

from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor


@register('cost.quadratic')
def make_quadratic(*, n: int) -> FunctionDescriptor:
    """
    Sum-of-squares cost: f(x) = x[0]^2 + x[1]^2 + ... + x[n-1]^2.

    A general-purpose regularization or test cost. Produces a smooth,
    strongly convex bowl — the global minimum is at x* = 0 with f* = 0.
    Useful as a regularization term (penalizing large variable values)
    or as a baseline test problem to verify NLP wiring.

    Factory parameters
    ------------------
    n : int
        Dimension of the input vector. Must be >= 1.

    Function interface
    ------------------
    Input
        x  :  (n, 1)  — the vector to penalize
    Output
        cost : (1, 1)  — scalar sum of squared elements (dimensionless)

    Sign convention
    ---------------
    Returns a non-negative scalar. To minimize, pass directly to
    solver.add_cost(). To use as a regularizer alongside other terms,
    scale before adding: solver.add_cost(lambda_reg * quadratic(x=x)).

    Usage example
    -------------
        quad = registry.get('cost.quadratic')(n=3)
        x = solver.add_variable('x', 3)
        solver.add_cost(quad(x=x))
    """
    x = ca.SX.sym('x', n)
    cost = ca.dot(x, x)
    f = ca.Function('quadratic', [x], [cost], ['x'], ['cost'])
    return FunctionDescriptor(f, description=f'Sum-of-squares cost (n={n})')


@register('cost.rosenbrock')
def make_rosenbrock(*, a: float = 1.0, b: float = 100.0) -> FunctionDescriptor:
    """
    Rosenbrock banana function: f(xy) = (a - xy[0])^2 + b*(xy[1] - xy[0]^2)^2.

    A classic nonlinear benchmark. The global minimum is at xy* = [a, a^2]
    with f* = 0. The function has a narrow curved valley that is easy to
    find but hard to follow, making it a stringent test for AD correctness
    and solver convergence.

    This function is primarily a **development and testing tool**, not a
    mission-relevant cost term. Its main use is verifying that automatic
    differentiation flows correctly through FunctionDescriptor.__call__ —
    if IPOPT converges to the known minimum, the AD graph is intact.

    Factory parameters
    ------------------
    a : float, default 1.0
        Shifts the minimum along xy[0]. With a=1, minimum is at [1, 1].
    b : float, default 100.0
        Controls the curvature of the banana valley. Higher b makes the
        valley narrower and the problem harder. Standard value is 100.

    Function interface
    ------------------
    Input
        xy : (2, 1)  — two-element vector; xy[0] and xy[1] are the two
                        free variables (no physical units — test problem)
    Output
        cost : (1, 1)  — scalar objective value (dimensionless, >= 0)

    Sign convention
    ---------------
    Non-negative; minimum is 0. Pass directly to solver.add_cost() for
    a minimization test. The problem has no maximum — f → ∞ as ||xy|| → ∞.

    Usage example
    -------------
        rosenbrock = registry.get('cost.rosenbrock')(a=1.0, b=100.0)
        xy = solver.add_variable('xy', 2, initial_guess=0.0)
        solver.add_cost(rosenbrock(xy=xy))
        # Optimal solution: xy* = [1.0, 1.0], f* = 0.0
    """
    xy = ca.SX.sym('xy', 2)
    cost = (a - xy[0])**2 + b * (xy[1] - xy[0]**2)**2
    f = ca.Function('rosenbrock', [xy], [cost], ['xy'], ['cost'])
    return FunctionDescriptor(f, description=f'Rosenbrock (a={a}, b={b})')


@register('cost.least_squares')
def make_least_squares(*, m: int, n: int) -> FunctionDescriptor:
    """
    Least-squares cost: f(x) = 0.5 * ||Ax - b||^2.

    A standard regression cost. The returned function accepts A, b, and x as inputs and computes the least-squares cost. 
    The factor of 0.5 is a common convention that simplifies the gradient expression (the 2 from the square cancels with the 0.5). 
    This paradigm fixed m and n at construction time, so the returned function expect A to be (m, n), b to be (m, 1), and x to be (n, 1). 
    The cost is a scalar (1, 1). This decision is made so that the function will automatically validate the shapes of A, b, and x when called,
    catching any mismatches early in the process. I don't expect this to hurt performance in terms of reusability for AD of the produced ca.Function,
    since a typical least squares problem has only one least squares in the entire problem. 

    Factory parameters
    ------------------
    m: int - number of rows in matrix A
    n: int - number of columns in matrix A (and size of decision variable x)

    Function interface
    ------------------
    Input
        A: (m, n) - matrix of coefficients. Generally a parameter of the problem, not a decision variable.
        b: (m, 1) - vector of observations. Generally a parameter of the problem, not a decision variable.
        x: (n, 1) - vector of decision variables.

    Output
        cost: (1, 1) - scalar least-squares cost value.

    Sign convention
    ---------------
    Returns a positive value to MINIMIZE. Generally passed to solver as-is.

        solver.add_cost(least_squares(A, b, x))

    Usage example
    -------------
        least_squares = registry.get('cost.least_squares')(m=m, n=n) # Obtain the function descriptor

        A = ... # Define or load your (m, n) coefficient matrix
        b = ... # Define or load your (m, 1) observation vector
        
        x_vec = solver.add_variable('x', n, initial_guess=0.0) # Decision variable vector

        solver.add_cost(least_squares(A=A, b=b, x=x_vec)) # Add the least-squares cost to the solver

        # Solve the problem and analyze results as usual. The optimal x will minimize the least-squares cost given A and b.
    """

    A = ca.SX.sym('A', m, n)
    b = ca.SX.sym('b', m, 1)
    x = ca.SX.sym('x', n, 1)
    cost = 0.5 * ca.sumsqr(A @ x - b)
    f = ca.Function('least_squares', [A, b, x], [cost], ['A', 'b', 'x'], ['cost'])
    return FunctionDescriptor(f, description=f'Least-squares cost ({m}x{n})')
