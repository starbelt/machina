"""
Generic constraint factories: ``constraint.linear``.

Importing this module registers it; ``import machina`` imports it through
:mod:`machina.library`.
"""

import casadi as ca

from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor


@register('constraint.linear')
def make_linear(*, n: int, coefficients: list) -> FunctionDescriptor:
    """
    Linear form: f(x) = c[0]*x[0] + c[1]*x[1] + ... + c[n-1]*x[n-1].

    Computes the scalar dot product c^T x. The constraint direction and
    bound values are NOT part of this function — they are applied by the
    agent type when registering with the solver. This separation keeps the
    function library pure math and lets the same linear form express
    equality constraints, upper bounds, or lower bounds depending on how
    the agent wires it.

    Common uses in the flyby problem
    ---------------------------------
    - Resource budget constraints: total power draw, compute load, or
      downlink bandwidth as a linear combination of per-product demands.
      e.g. sum(power_per_product[i] * x[i]) <= total_power_budget

    - Allocation constraints: total time or capacity allocated across
      products must equal a fixed budget.
      e.g. sum(proc_time[i]) == total_compute_window  (equality)

    - Simple weighted sums that appear in constraint expressions.

    Note: For resource constraints of the form "sum of resource demands
    <= budget", the coefficients are the per-unit resource demands and x
    is the allocation vector. The budget (right-hand side) is applied as
    the `ub` argument to solver.add_constraint(), not baked into this
    function.

    Factory parameters
    ------------------
    n : int
        Dimension of the input vector. Must be >= 1.
    coefficients : list[float]
        Length-n list of scalar multipliers, one per element of x.
        Must have exactly n elements; raises ValueError otherwise.
        Units should be consistent with x so that c^T x has the
        physical units of the constraint (e.g., watts, seconds, Mbps).

    Function interface
    ------------------
    Input
        x : (n, 1)  — the quantity vector (decision variables, expressions,
                       or a mix of both). Units depend on context.
    Output
        value : (1, 1)  — scalar c^T x. Units are [coefficients] * [x].

    Applying constraint semantics
    ------------------------------
    This function returns an expression; the agent type decides what to do
    with it. All three patterns below are valid:

        expr = linear(x=x_mx)

        # Inequality: c^T x <= upper_limit
        solver.add_constraint(expr, ub=upper_limit, name='power_budget')

        # Inequality: c^T x >= lower_limit
        solver.add_constraint(expr, lb=lower_limit, name='min_downlink')

        # Equality: c^T x == target
        solver.add_equality(expr - target, name='compute_budget')

    Usage example (resource budget)
    ---------------------------------
        # 3 products, power demands 10W / 15W / 8W, budget 25W
        power_draw = registry.get('constraint.linear')(
            n=3,
            coefficients=[10.0, 15.0, 8.0],   # watts per unit allocation
        )
        alloc = solver.add_variable('alloc', 3, lb=0.0, ub=1.0)
        solver.add_constraint(power_draw(x=alloc), ub=25.0, name='power_budget')

    Usage example (allocation equality)
    --------------------------------------
        # Processing time must sum to exactly 150 seconds
        total_time = registry.get('constraint.linear')(
            n=2,
            coefficients=[1.0, 1.0],
        )
        proc_time = solver.add_variable('proc_time', 2, lb=0.0)
        solver.add_equality(total_time(x=proc_time) - 150.0, name='time_budget')
    """
    if len(coefficients) != n:
        raise ValueError(
            f"constraint.linear: expected {n} coefficients, "
            f"got {len(coefficients)}."
        )
    x = ca.SX.sym('x', n)
    value = sum(float(coefficients[i]) * x[i] for i in range(n))
    f = ca.Function('linear_constraint', [x], [value], ['x'], ['value'])
    return FunctionDescriptor(
        f, description=f'Linear constraint c^T x (n={n})'
    )
