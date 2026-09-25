"""
Latency factory: ``util.ttp_computation`` sums the delay stages of a data
product's time to product (TTP).

The registered name keeps its ``util.`` prefix. Importing this module registers
the factory; ``import machina.swapc`` imports it.
"""

import casadi as ca

from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor


@register('util.ttp_computation')
def make_ttp_computation(*, n_components: int) -> FunctionDescriptor:
    """
    Assemble TTP (Time To Product) from additive delay components.

    TTP is the total end-to-end latency from the moment a data product's
    acquisition begins to the moment it is delivered to the end user. It
    is modelled as a sum of independent, sequential delay stages. This
    function takes all of those delay values as a single vector and returns
    their scalar sum.

    Typical TTP breakdown for a flyby satellite product:
        t_observe    — time for the satellite to collect the raw observation
                       (e.g. integration time, sensor dwell). Often fixed
                       by geometry and sensor parameters.
        t_process    — onboard processing time (compression, formatting).
                       May be a decision variable if processing rate is
                       being optimized.
        t_downlink   — time to transmit the processed data to a ground
                       station or relay. Depends on contact window geometry
                       and link budget.
        t_distribute — time from ground receipt to final user delivery
                       (ground processing, routing). Often a fixed constant.

    Not all problems require all four stages. Construct the descriptor
    with n_components equal to the number of delay stages that are
    relevant to your problem.

    The input is a single stacked vector of all delays. The agent type is
    responsible for assembling this vector from its individual components,
    which may be decision variables, fixed constants, or MX expressions
    computed from other functions. Use ca.vertcat() to assemble the vector
    before calling this function.

    Factory parameters
    ------------------
    n_components : int
        Number of delay stages to sum. Must be >= 1. Raises ValueError
        if 0 or negative.

    Function interface
    ------------------
    Input
        delays : (n_components, 1)  — vector of individual delay durations,
                                       all in consistent time units (seconds
                                       recommended). The order of elements
                                       does not affect the result (addition
                                       is commutative), but should be
                                       consistent with how the agent builds
                                       the vector.
    Output
        ttp : (1, 1)  — scalar total TTP in the same units as the inputs.

    Usage example (2-stage: fixed observation + optimizable processing)
    -------------------------------------------------------------------
        ttp_func = registry.get('util.ttp_computation')(n_components=2)

        t_obs = ca.MX(50.0)                           # fixed, seconds
        t_proc = solver.add_variable('proc', 1,       # decision variable
                                     lb=0.0, ub=300.0)

        ttp_expr = ttp_func(delays=ca.vertcat(t_obs, t_proc))
        # ttp_expr is an MX expression: 50.0 + t_proc

    Usage example (4-stage: full pipeline)
    ----------------------------------------
        ttp_func = registry.get('util.ttp_computation')(n_components=4)

        delays = ca.vertcat(
            t_observe,     # MX expression or constant
            t_process,     # decision variable
            t_downlink,    # computed from contact geometry
            ca.MX(30.0),   # fixed distribution delay
        )
        ttp_expr = ttp_func(delays=delays)

    Usage in the goodput pipeline
    ------------------------------
    The output ttp_expr is typically passed directly to cost.sigmoid_goodput
    or assembled into a ttp_vector for cost.aggregate_goodput:

        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=150.0)
        goodput_expr = sigmoid(ttp=ttp_expr)
        solver.add_cost(-goodput_expr)

    Notes
    -----
    This function is intentionally trivial (it is just ca.sum1). Its value
    is semantic clarity and consistent naming in the agent assembly code,
    not computational complexity. An agent could write ca.sum1(delays)
    directly, but using util.ttp_computation makes the intent explicit and
    the computation appear in the function registry for introspection.
    """
    if n_components < 1:
        raise ValueError(
            f"util.ttp_computation: n_components must be >= 1, got {n_components}."
        )
    delays = ca.SX.sym('delays', n_components)
    ttp = ca.sum1(delays)
    f = ca.Function('ttp_computation', [delays], [ttp], ['delays'], ['ttp'])
    return FunctionDescriptor(
        f, description=f'TTP = sum of {n_components} delay components'
    )
