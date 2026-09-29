"""
Goodput factories: ``cost.sigmoid_goodput`` scores the timeliness of one
delivered product, ``cost.aggregate_goodput`` the importance-weighted sum over
products, and ``cost.loglogistic_goodput`` scores a latency against an event
timescale on a log axis. ``loglogistic_shape`` turns two latencies (the 90 %
and 10 % value points) into that factory's ``(tau, q)``.

The registered names keep their ``cost.`` prefix: registered names are a
stable ABI. Importing this module registers the three factories;
``import machina.swapc`` imports it.
"""

import math
import numbers

import casadi as ca

from machina.library.numerics import TINY
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
    per-product descriptors and wire them manually in the caller.

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
    factory construction, not the MX level.

    All products must share the same per-product goodput model. If
    products have heterogeneous deadline profiles, the caller should
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


@register('cost.loglogistic_goodput')
def make_loglogistic_goodput(*, q: float, tau: float) -> FunctionDescriptor:
    """
    Log-logistic goodput: G(L) = 1 / (1 + (L / tau)**q).

    The value of an observation that arrives a latency L after the event it
    reports, for an event that evolves on a timescale tau. It is the logistic
    of ``cost.sigmoid_goodput`` applied to log(L / tau) instead of L, so the
    decay is set by the ratio L / tau: halving the latency buys the same
    value at every scale. G(0) = 1, G(tau) = 0.5, G -> 0 as L -> inf.

    Factory parameters
    ------------------
    q : float
        Steepness, > 0. G falls from 0.9 to 0.1 while L grows by a factor
        of 81**(1/q): q = 2 spans 9x, q = 10 about 1.55x (close to a step at
        tau). ``loglogistic_shape`` computes q from those two latencies.
    tau : float
        Event timescale, > 0, in the unit of the latency input (seconds for
        the thesis): the latency at which half the value is gone.

    Function interface
    ------------------
    Input
        latency : (1, 1)  -- latency L >= 0, same unit as tau.
    Output
        goodput : (1, 1)  -- dimensionless value in (0, 1].

    Numerics
    --------
    Evaluated as 0.5 - 0.5 * tanh(z / 2) with z = q * log(max(L / tau, TINY)),
    which equals 1 / (1 + exp(z)) but never forms exp(z): at a huge latency
    exp(z) overflows to inf and the gradient becomes inf / inf = NaN. The
    floor at TINY keeps log finite at L = 0, where the gradient is then 0. For
    q < 1 the true gradient is unbounded as L -> 0+; the floor caps it.

    Sign convention
    ---------------
    A value to MAXIMIZE, like ``cost.sigmoid_goodput``: pass ``-G`` as a cost.

    Usage example
    -------------
        tau, q = loglogistic_shape(latency_at_90=600.0, latency_at_10=5400.0)
        G = registry.get('cost.loglogistic_goodput')(q=q, tau=tau)
        value = G(latency=latency_expr)
    """
    q = _positive(q, 'q', 'cost.loglogistic_goodput')
    tau = _positive(tau, 'tau', 'cost.loglogistic_goodput')
    latency = ca.SX.sym('latency')
    z = q * ca.log(ca.fmax(latency / tau, TINY))
    goodput = 0.5 - 0.5 * ca.tanh(0.5 * z)
    f = ca.Function('loglogistic_goodput', [latency], [goodput], ['latency'], ['goodput'])
    return FunctionDescriptor(f, description=f'Log-logistic goodput (q={q}, tau={tau})')


def loglogistic_shape(*, latency_at_90: float, latency_at_10: float) -> tuple:
    """
    ``(tau, q)`` of the log-logistic goodput with G = 0.9 at *latency_at_90*
    and G = 0.1 at *latency_at_10*.

    From (a / tau)**q = 1/9 and (b / tau)**q = 9: tau = sqrt(a * b) and
    q = ln(81) / ln(b / a). Both latencies must be positive, in the same
    unit, with latency_at_90 < latency_at_10.
    """
    a = _positive(latency_at_90, 'latency_at_90', 'loglogistic_shape')
    b = _positive(latency_at_10, 'latency_at_10', 'loglogistic_shape')
    if not a < b:
        raise ValueError(
            f"loglogistic_shape: latency_at_90 ({a}) must be smaller than "
            f"latency_at_10 ({b}); value falls as latency grows.")
    return math.sqrt(a * b), math.log(81.0) / math.log(b / a)


def _positive(value, name: str, where: str) -> float:
    """*value* as a finite positive float, or a ValueError naming *where*."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{where}: {name} must be a real number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{where}: {name} must be a finite number > 0, got {value!r}")
    return number
