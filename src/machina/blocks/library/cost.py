import casadi as ca

from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor


@register('cost.sigmoid_goodput')
def make_sigmoid_goodput(*, k: float, t50: float) -> FunctionDescriptor:
    """
    Per-product sigmoid goodput: G(ttp) = 1 / (1 + exp(k * (ttp - t50))).

    Goodput measures the timeliness value of a delivered data product.
    It is modelled as a sigmoid that decays from ~1 (very timely) toward
    ~0 (very late). The key design insight is that value is not binary —
    a product delivered slightly late still has substantial value, but
    one delivered far too late has nearly none.

    The sigmoid has two physical interpretations of its parameters:
      - t50 is the "deadline" in soft terms: the TTP at which half the
        potential value has been lost.
      - k controls how sharp that deadline is. A small k (e.g. 0.01)
        gives a gradual decay over a wide time window. A large k (e.g. 1.0)
        creates a hard cliff near t50.

    This function operates on a **single product**. To aggregate across
    multiple products, use cost.aggregate_goodput, which calls this
    function once per product and computes the weighted sum.

    Factory parameters
    ------------------
    k : float
        Steepness of the sigmoid (units: 1/seconds if TTP is in seconds).
        Must be positive. Higher values create a sharper drop at t50.
        Typical range for space mission products: 0.01 – 0.5.
    t50 : float
        TTP at which goodput = 0.5 (same units as TTP — typically seconds).
        This is the soft deadline: the inflection point of the decay curve.
        Typical range: 60 – 3600 seconds depending on product urgency.

    Function interface
    ------------------
    Input
        ttp : (1, 1)  — Time To Product in seconds (scalar). Must be the
                        total end-to-end latency for this product: from
                        observation start to delivery to the end user.
                        Typically assembled via util.ttp_computation.
    Output
        goodput : (1, 1)  — dimensionless value in (0, 1).
                            Values near 1 mean timely delivery (ttp << t50).
                            Values near 0 mean very late delivery (ttp >> t50).
                            Exactly 0.5 when ttp == t50.

    Sign convention
    ---------------
    Goodput is a value to MAXIMIZE. To minimize the negative goodput,
    negate the output when adding to the solver:

        solver.add_cost(-sigmoid(ttp=ttp_expr))

    For the aggregate case this negation happens outside cost.aggregate_goodput
    (which returns the weighted sum as a positive value).

    Usage example (single product)
    --------------------------------
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=120.0)

        ttp = solver.add_variable('ttp', 1, lb=0.0, ub=600.0, initial_guess=60.0)
        solver.add_cost(-sigmoid(ttp=ttp))   # maximize goodput

    Usage example (as input to aggregate_goodput)
    -----------------------------------------------
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=120.0)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid,
            n_products=3,
            weights=[0.5, 0.3, 0.2],
        )
        ttp_vec = solver.add_variable('ttp', 3, lb=0.0, ub=600.0)
        solver.add_cost(-agg(ttp_vector=ttp_vec))

    Notes
    -----
    k and t50 are baked into the ca.Function at construction time as
    numeric constants. They are not decision variables or parameters —
    they are part of the problem definition. If you need to sweep k or
    t50 (e.g., for sensitivity analysis), construct a new descriptor for
    each configuration.
    """
    ttp = ca.SX.sym('ttp')
    goodput = 1.0 / (1.0 + ca.exp(k * (ttp - t50)))
    f = ca.Function('sigmoid_goodput', [ttp], [goodput], ['ttp'], ['goodput'])
    return FunctionDescriptor(f, description=f'Sigmoid goodput (k={k}, t50={t50})')


@register('cost.aggregate_goodput')
def make_aggregate_goodput(
    *,
    per_product_func: FunctionDescriptor,
    n_products: int,
    weights: list,
) -> FunctionDescriptor:
    """
    Weighted sum of per-product goodput: A = sum_i( w_i * G(ttp_i) ).

    This is the primary objective function for the flyby goodput problem.
    It takes a vector of per-product TTPs and a pre-built per-product
    goodput function, and returns the importance-weighted sum of goodput
    values. A higher aggregate goodput means the flyby system is delivering
    more timely value to more important products.

    The per-product goodput function is embedded at construction time via
    symbolic composition (SX-level). Each product's TTP is extracted as
    a scalar from the input vector and fed through the goodput function
    independently. This means all products share the same goodput model
    (same k and t50) — if products have different deadlines, construct
    per-product descriptors and wire them manually at the agent level.

    Factory parameters
    ------------------
    per_product_func : FunctionDescriptor
        A scalar goodput function taking a single (1,1) TTP input and
        returning a (1,1) goodput value. Typically cost.sigmoid_goodput.
        Must already be constructed (via its own factory call) before
        passing here. Its ca.Function is embedded symbolically during
        construction of the aggregate function.
    n_products : int
        Number of products. Determines the size of the TTP input vector.
        Must match the length of weights.
    weights : list[float]
        Importance weight for each product, in the same order as TTP
        elements. Weights need not sum to 1 — they are multiplicative
        scale factors. Higher weight = that product contributes more to
        the aggregate objective.

    Function interface
    ------------------
    Input
        ttp_vector : (n_products, 1)  — TTP for each product in seconds.
                                        Element [i] is the total end-to-end
                                        latency for product i (observation
                                        + processing + downlink + distribute).
    Output
        aggregate_goodput : (1, 1)  — dimensionless weighted sum.
                                       Upper bound is sum(weights) if all
                                       products have ttp << t50. Lower bound
                                       approaches 0 if all ttp >> t50.

    Sign convention
    ---------------
    Returns a positive value to MAXIMIZE. Negate when passing to the solver:

        solver.add_cost(-agg(ttp_vector=ttp_mx))

    Usage example
    -------------
        sigmoid = registry.get('cost.sigmoid_goodput')(k=0.1, t50=150.0)
        agg = registry.get('cost.aggregate_goodput')(
            per_product_func=sigmoid,
            n_products=4,
            weights=[0.4, 0.3, 0.2, 0.1],  # product importance weights
        )

        ttp_vec = solver.add_variable('ttp', 4, lb=0.0, ub=600.0)
        solver.add_cost(-agg(ttp_vector=ttp_vec))

    Notes
    -----
    Composition pattern: per_product_func.function is called directly
    (not per_product_func()) during construction. This bypasses the
    MX-oriented shape validation in FunctionDescriptor.__call__, which
    is correct behaviour — we are operating at the SX level during
    factory construction, not the MX level. See block_library_interface_v1.md
    for a full explanation of this pattern.

    All products must share the same per-product goodput model. If
    products have heterogeneous deadline profiles, the agent type should
    wire per-product sigmoid descriptors manually rather than using this
    factory.
    """
    ttp_vec = ca.SX.sym('ttp_vector', n_products)
    total = ca.SX(0)
    for i in range(n_products):
        gi = per_product_func.function(ttp_vec[i])
        total = total + weights[i] * gi
    f = ca.Function(
        'aggregate_goodput',
        [ttp_vec], [total],
        ['ttp_vector'], ['aggregate_goodput'],
    )
    return FunctionDescriptor(
        f, description=f'Aggregate goodput ({n_products} products)'
    )


@register('cost.smooth_coverage')
def make_smooth_coverage() -> FunctionDescriptor:
    """
    Smooth sigmoid coverage indicator: c(e) = 1 / (1 + exp(-k*(e - e_min))).

    Returns ~0 when elevation << min_elevation, ~1 when >> min_elevation.
    Exactly 0.5 when elevation == min_elevation (the soft threshold).

    Unlike cost.sigmoid_goodput (where k and t50 are factory parameters baked
    into the function), all three inputs here are function arguments.  This
    allows the agent to pass MX parameter symbols for min_elevation and k,
    which is needed in the SingleSatCoverage agent where these quantities
    are runtime parameters, not compile-time constants.

    Function interface
    ------------------
    Inputs
        elevation     : (1,1)  -- elevation angle [rad]
        min_elevation : (1,1)  -- threshold elevation [rad]; coverage = 0.5 here
        k             : (1,1)  -- sigmoid steepness [1/rad]; larger = sharper
                                   transition near min_elevation.
                                   Typical: 20-50 [1/rad] (~2-5 deg transition width)
    Output
        coverage : (1,1)  -- smooth coverage value in (0, 1)

    Sign convention
    ---------------
    Coverage is a value to MAXIMIZE.  Negate when adding to the solver:

        solver.add_cost(-coverage_expr)

    Usage example
    -------------
        smooth_cov = registry.get('cost.smooth_coverage')()

        # With numeric inputs
        e_rad = ca.DM(math.radians(20.0))
        e_min_rad = ca.DM(math.radians(10.0))
        k_val = ca.DM(20.0)   # [1/rad]
        cov = smooth_cov(elevation=e_rad, min_elevation=e_min_rad, k=k_val)

        # With MX symbols (typical agent usage)
        elev_sym = ca.MX.sym('elev', 1, 1)
        e_min_param = solver.add_parameter('e_min', 1)
        k_param     = solver.add_parameter('k', 1)
        cov_expr = smooth_cov(elevation=elev_sym,
                              min_elevation=e_min_param,
                              k=k_param)
    """
    elevation     = ca.SX.sym('elevation')
    min_elevation = ca.SX.sym('min_elevation')
    k             = ca.SX.sym('k')

    coverage = 1.0 / (1.0 + ca.exp(-k * (elevation - min_elevation)))

    fn = ca.Function(
        'smooth_coverage',
        [elevation, min_elevation, k], [coverage],
        ['elevation', 'min_elevation', 'k'], ['coverage'],
    )
    return FunctionDescriptor(
        fn,
        description='Sigmoid coverage indicator c(elevation, min_elevation, k)',
    )
