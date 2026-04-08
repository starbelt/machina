import casadi as ca
from machina.blocks.registry import register
from machina.blocks.descriptor import FunctionDescriptor


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


@register("util.sum")
def make_sum(*, n_terms: int) -> FunctionDescriptor:
    """
    Sum a vector of terms.

    Factory parameters
    ------------------
    n_terms : int
        Number of terms to sum. Must be >= 1. Raises ValueError if 0 or
        negative.

    Function interface
    ------------------
    Input
        terms : (n_terms, 1)  — vector of terms to sum.
    Output
        total : (1, 1)  — scalar sum of the input terms.

    Notes
    -----
    This function is a simple wrapper around ca.sum1 for semantic clarity.
    An agent could write ca.sum1(terms) directly, but using util.sum makes
    the intent explicit and the computation appear in the function registry
    for introspection.
    """
    if n_terms < 1:
        raise ValueError(
            f"util.sum: n_terms must be >= 1, got {n_terms}."
        )
    terms = ca.SX.sym('terms', n_terms)
    total = ca.sum1(terms)
    f = ca.Function('sum', [terms], [total], ['terms'], ['total'])
    return FunctionDescriptor(
        f, description=f'Σ ({n_terms} terms)'
    )

@register("util.rotate_x")
def make_rotate_x() -> FunctionDescriptor:
    """
    Rotate a 3D vector about the x-axis by a given angle.

    Factory parameters
    ------------------
    None

    Function interface
    ------------------
    Input
        v : (3, 1)  — 3D vector to rotate.
        theta : (1, 1)  — rotation angle in radians.
    Output
        v_rot : (3, 1)  — rotated vector.

    Notes
    -----
    This function is provided as a utility for rotating vectors in 3D space.
    It uses the standard rotation matrix for rotations about the x-axis.
    """
    v = ca.SX.sym('v', 3)
    theta = ca.SX.sym('theta')
    c = ca.cos(theta)
    s = ca.sin(theta)
    R_x = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, c, -s),
        ca.horzcat(0, s, c)
    )
    v_rot = R_x @ v
    f = ca.Function('rotate_x', [v, theta], [v_rot], ['v', 'theta'], ['v_rot'])
    return FunctionDescriptor(
        f, description='Rx(theta) * v'
    )

@register("util.rotate_y")
def make_rotate_y() -> FunctionDescriptor:
    """
    Rotate a 3D vector about the y-axis by a given angle.

    Factory parameters
    ------------------
    None

    Function interface
    ------------------
    Input
        v : (3, 1)  — 3D vector to rotate.
        theta : (1, 1)  — rotation angle in radians.
    Output
        v_rot : (3, 1)  — rotated vector.

    Notes
    -----
    This function is provided as a utility for rotating vectors in 3D space.
    It uses the standard rotation matrix for rotations about the y-axis.
    """
    v = ca.SX.sym('v', 3)
    theta = ca.SX.sym('theta')
    c = ca.cos(theta)
    s = ca.sin(theta)
    R_y = ca.vertcat(
        ca.horzcat(c, 0, s),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-s, 0, c)
    )
    v_rot = R_y @ v
    f = ca.Function('rotate_y', [v, theta], [v_rot], ['v', 'theta'], ['v_rot'])
    return FunctionDescriptor(
        f, description='Ry(theta) * v'
    )

@register("util.rotate_z")
def make_rotate_z() -> FunctionDescriptor:
    """
    Rotate a 3D vector about the z-axis by a given angle.

    Factory parameters
    ------------------
    None

    Function interface
    ------------------
    Input
        v : (3, 1)  — 3D vector to rotate.
        theta : (1, 1)  — rotation angle in radians.
    Output
        v_rot : (3, 1)  — rotated vector.

    Notes
    -----
    This function is provided as a utility for rotating vectors in 3D space.
    It uses the standard rotation matrix for rotations about the z-axis.
    """
    v = ca.SX.sym('v', 3)
    theta = ca.SX.sym('theta')
    c = ca.cos(theta)
    s = ca.sin(theta)
    R_z = ca.vertcat(
        ca.horzcat(c, -s, 0),
        ca.horzcat(s, c, 0),
        ca.horzcat(0, 0, 1)
    )
    v_rot = R_z @ v
    f = ca.Function('rotate_z', [v, theta], [v_rot], ['v', 'theta'], ['v_rot'])
    return FunctionDescriptor(
        f, description='Rz(theta) * v'
    )