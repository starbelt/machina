"""
Antenna-beam geometry: the off-axis angle of a direction from a boresight, a
smooth in-cone indicator, and the time-mean fraction of a relative orbit spent
inside a cone.

Registered functions
--------------------
geometry.off_axis_angle -- angle between a boresight and a direction
cost.smooth_in_cone     -- sigmoid indicator of "inside a cone of half-angle alpha"
cost.mean_beam_duty     -- mean in-cone indicator over one relative orbit

The ``cost.`` names follow ``cost.smooth_coverage``: they are values to
maximise or minimise, and the problem owns the sign. Importing this module
registers the three factories; ``import machina.astro`` imports it.
"""

import math

import casadi as ca

from machina.astro.relative import _require_positive, make_roe_to_rtn
from machina.library.numerics import safe_norm
from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor

__all__ = ["make_off_axis_angle", "make_smooth_in_cone", "make_mean_beam_duty"]

# Cap on z = k (angle - half_angle) in cost.smooth_in_cone. exp(z) overflows past
# z ~ 709.8 and the sigmoid's derivative then evaluates inf/inf = NaN; exp(700) is
# finite, so capped value and first/second derivatives stay finite. The cap moves the
# value by less than 1e-304 and zeroes the (already underflowed) gradient beyond it.
_Z_MAX = 700.0


# ---------------------------------------------------------------------------
# geometry.off_axis_angle
# ---------------------------------------------------------------------------

@register('geometry.off_axis_angle')
def make_off_axis_angle() -> FunctionDescriptor:
    """
    Factory for the angle between a boresight and a direction.

        angle = atan2(|b x r|, b . r)      in [0, pi]

    Function interface
    ------------------
    Inputs
        boresight : (3,1)  -- boresight vector, any frame, need not be unit
        direction : (3,1)  -- direction vector, same frame, need not be unit
    Output
        angle : (1,1)  -- off-axis angle [rad]

    Numerics
    --------
    ``atan2`` of the cross-product norm, not ``acos`` of the normalised dot
    product: ``acos`` has an unbounded derivative at 0 and pi and loses all
    precision for small angles. ``|b x r|`` is a ``safe_norm``, so on-axis the
    value is ~1e-16 / (|b| |r|) and the gradient is finite (zero), a valid
    subgradient of the cone-shaped function there. Antiparallel vectors give
    pi. Either vector at the origin gives a meaningless (but finite) angle.
    """
    boresight = ca.SX.sym('boresight', 3)
    direction = ca.SX.sym('direction', 3)

    angle = ca.atan2(safe_norm(ca.cross(boresight, direction)), ca.dot(boresight, direction))

    fn = ca.Function('off_axis_angle', [boresight, direction], [angle],
                     ['boresight', 'direction'], ['angle'])
    return FunctionDescriptor(fn, description='Off-axis angle atan2(|b x r|, b . r) [rad]')


# ---------------------------------------------------------------------------
# cost.smooth_in_cone
# ---------------------------------------------------------------------------

@register('cost.smooth_in_cone')
def make_smooth_in_cone() -> FunctionDescriptor:
    """
    Smooth indicator of "inside a cone": c = 1 / (1 + exp(k (angle - half_angle))).

    ~1 well inside the cone, ~0 well outside, exactly 0.5 on the edge. The
    mirror image of ``cost.smooth_coverage`` (which rises with elevation; this
    falls with off-axis angle), with the same interface convention: all three
    inputs are function arguments so MX parameters can flow through.

    Function interface
    ------------------
    Inputs
        angle      : (1,1)  -- off-axis angle [rad]
        half_angle : (1,1)  -- cone half-angle [rad]; indicator = 0.5 here
        k          : (1,1)  -- steepness [1/rad]; larger is sharper
    Output
        in_cone : (1,1)  -- smooth indicator in (0, 1]

    Numerics
    --------
    ``cost.smooth_coverage`` evaluates the bare sigmoid, whose derivative is NaN
    once ``exp`` overflows (argument > ~709.8, reachable with the large ``k``
    a narrow beam needs). Here the exponent is capped at 700, the approach of
    ``cost.loglogistic_goodput``: identical below the cap, value < 1e-304 and a
    zero gradient above it.
    """
    angle = ca.SX.sym('angle')
    half_angle = ca.SX.sym('half_angle')
    k = ca.SX.sym('k')

    z = ca.fmin(k * (angle - half_angle), _Z_MAX)
    in_cone = 1.0 / (1.0 + ca.exp(z))

    fn = ca.Function('smooth_in_cone', [angle, half_angle, k], [in_cone],
                     ['angle', 'half_angle', 'k'], ['in_cone'])
    return FunctionDescriptor(
        fn, description='Sigmoid in-cone indicator c(angle, half_angle, k)')


# ---------------------------------------------------------------------------
# cost.mean_beam_duty
# ---------------------------------------------------------------------------

@register('cost.mean_beam_duty')
def make_mean_beam_duty(*, a: float, n_samples: int) -> FunctionDescriptor:
    """
    Time-mean fraction of a period-matched relative orbit that a deputy spends
    inside a cone fixed in the chief's RTN frame.

        duty = (1/n) sum_j smooth_in_cone(off_axis_angle(b, r_rtn(lam_j)), alpha, k)
        lam_j = 2 pi j / n,  j = 0 .. n-1

    ``r_rtn`` is ``transform.roe_to_rtn`` (so the delta a = 0, near-circular,
    near-equatorial chief assumptions apply); for a circular chief, uniform
    longitude samples are uniform time samples, so the mean is a time mean.

    The direction is chief -> deputy (``r_rtn``): the question is whether the
    deputy sits inside the chief's transmit cone, apex at the chief. For a
    geostationary chief the RTN frame is Earth-fixed (R toward the
    sub-satellite point, N along the spin axis, T east), so a boresight on a
    ground station is a constant vector in RTN.

    Composed at SX level from ``transform.roe_to_rtn``,
    ``geometry.off_axis_angle`` and ``cost.smooth_in_cone`` through
    ``.function(...)``.

    Factory parameters
    ------------------
    a : float
        Chief semi-major axis [km]. Baked in.
    n_samples : int
        Number of longitude samples over one orbit (>= 1). The hard-indicator
        limit needs enough samples to resolve the in-cone arc: a cone of
        half-angle alpha seen from a relative orbit of size ~a|delta| is crossed
        in a longitude arc of order alpha, so n_samples >> 2 pi / alpha.

    Function interface
    ------------------
    Inputs
        roe           : (5,1)  -- [dlam, df, dg, dix, diy]
        boresight_rtn : (3,1)  -- cone axis in the chief's RTN frame, need not be unit
        half_angle    : (1,1)  -- cone half-angle [rad]
        k             : (1,1)  -- sigmoid steepness [1/rad]
    Output
        duty : (1,1)  -- mean smooth in-cone indicator, in [0, 1]
    """
    if isinstance(n_samples, bool) or int(n_samples) != n_samples or n_samples < 1:
        raise ValueError(
            f"cost.mean_beam_duty: n_samples must be a positive integer, got {n_samples!r}.")
    n_samples = int(n_samples)
    a = _require_positive('a', a, 'cost.mean_beam_duty')
    roe_to_rtn = make_roe_to_rtn(a=a)
    off_axis_angle = make_off_axis_angle()
    smooth_in_cone = make_smooth_in_cone()

    roe = ca.SX.sym('roe', 5)
    boresight = ca.SX.sym('boresight_rtn', 3)
    half_angle = ca.SX.sym('half_angle')
    k = ca.SX.sym('k')

    terms = []
    for j in range(n_samples):
        lam_j = 2.0 * math.pi * j / n_samples
        r_rtn = roe_to_rtn.function(roe, lam_j)
        angle = off_axis_angle.function(boresight, r_rtn)
        terms.append(smooth_in_cone.function(angle, half_angle, k))
    duty = ca.sum1(ca.vertcat(*terms)) / n_samples

    fn = ca.Function('mean_beam_duty', [roe, boresight, half_angle, k], [duty],
                     ['roe', 'boresight_rtn', 'half_angle', 'k'], ['duty'])
    return FunctionDescriptor(
        fn, description=f'Mean in-cone duty over one relative orbit '
                        f'(a={a} km, n_samples={n_samples})')
