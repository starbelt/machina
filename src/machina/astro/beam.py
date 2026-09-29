"""
Antenna-beam geometry: the off-axis angle of a direction from a boresight, a
smooth in-cone indicator, and the time-mean fraction of a relative orbit spent
inside a cone.

Registered functions
--------------------
geometry.off_axis_angle -- angle between a boresight and a direction
cost.smooth_in_cone     -- sigmoid indicator of "inside a cone of half-angle alpha"
cost.mean_beam_duty     -- mean in-cone indicator over one relative orbit

Plain helper (not registered): :func:`beam_duty_sharpness`, the sigmoid
steepness ``k`` that matches a ``cost.mean_beam_duty`` sample count.

The ``cost.`` names follow ``cost.smooth_coverage``: they are values to
maximise or minimise, and the problem owns the sign. Importing this module
registers the three factories; ``import machina.astro`` imports it.
"""

import math

import casadi as ca

from machina.astro.relative import _require_count, _require_positive, make_roe_to_rtn
from machina.library.numerics import safe_norm
from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor

__all__ = [
    "make_off_axis_angle",
    "make_smooth_in_cone",
    "make_mean_beam_duty",
    "beam_duty_sharpness",
]

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
        Number of longitude samples over one orbit (>= 1). See the sampling
        rule below.

    Function interface
    ------------------
    Inputs
        roe           : (5,1)  -- [dlam, df, dg, dix, diy]
        boresight_rtn : (3,1)  -- cone axis in the chief's RTN frame, need not be unit
        half_angle    : (1,1)  -- cone half-angle [rad]
        k             : (1,1)  -- sigmoid steepness [1/rad]
    Output
        duty : (1,1)  -- mean smooth in-cone indicator, in [0, 1]

    Sampling rule (choose k and n_samples together)
    -----------------------------------------------
    The duty is a sum of ``n_samples`` sigmoids of edge width ~1/k in angle. It
    is smooth in the ROE only while each edge spans about one sample. For
    ``dlam = 0`` the relative orbit ``r = P cos lam + Q sin lam`` is an ellipse
    centred on the chief, and the chief->member direction turns at
    ``|P x Q| / |r|^2`` rad per rad of lam: at most the ellipse's axis ratio,
    which is ``<= sqrt(4 + |di|^2/|de|^2)`` (equal for parallel e/i vectors), so
    at most sqrt(5) for ``|di| <= |de|`` (the along-track motion alone gives 2
    at a nadir crossing, for any ``dlam``). One sample then sweeps up to
    ``2 pi sqrt(5) / n_samples`` rad of off-axis angle, so

    * ``k = n_samples / (2 pi sqrt(5))`` puts one sample across the edge; this
      is :func:`beam_duty_sharpness`;
    * ``n_samples >= 20 pi sqrt(5) / alpha`` makes the in-cone arc of lam
      (``2 alpha / sqrt(5)`` at that rate) span >= ~20 samples. Then
      ``k alpha >= 10``, and the sigmoid's bias on the duty is below 1e-4
      relative. For alpha = 1.2 deg: n_samples >= 6709 (7200 is 12 s at GEO).

    Off the rule the duty is wrong without any error. A sharper k (edge
    narrower than a sample) makes the duty a staircase in the ROE, with steps of
    1/n_samples and a gradient that is ~0 between steps, which IPOPT cannot use:
    k = 1e4 at n_samples = 3600 moves a 0.30 % duty by 9 % relative under a
    half-sample phase shift, and n_samples = 100 returns 0 or 1 %. A softer k
    biases the duty upward: at n_samples = 3600, alpha = 1.2 deg, the parallel
    |de| = |di| formation below reads 0.298 % at k = 300, 0.315 % at k = 100 and
    0.50 % at k = 30. Formations that turn faster (|di| > |de|, or ``dlam != 0``
    passing close to the chief) need k and n_samples scaled by their rate.

    Reference values (hard indicator, boresight -R)
    -----------------------------------------------
    * Parallel e/i, ``|de| = |di|``, ``dlam = 0``: the direction crosses -R on
      the ellipse's minor axis at rate sqrt(5), and
      ``duty = atan(tan(alpha) / sqrt(5)) / pi`` (0.298 % at alpha = 1.2 deg).
    * Bound for any ROE: ``duty <= atan(tan(alpha) / 2) / pi``
      (0.333 % at 1.2 deg). On the nadir side the off-axis angle is
      ``atan(sqrt(T^2 + N^2) / -R)``; R and T do not depend on di and N only
      adds to it, so the in-plane formation (di = 0) with the same dlam, de is
      the worst case. There ``T / -R = (x + 2 sin psi) / cos psi`` with
      ``x = dlam/|de|`` and ``psi = lam - arg(de)``, whose in-cone set has
      length exactly ``2 atan(tan(alpha) / 2)`` for every ``|x| < 2`` and less
      for ``|x| >= 2``. Equality: di = 0 and ``|dlam| < 2 |de|``.

    The smooth duty follows these to within the sigmoid bias above.
    """
    n_samples = _require_count('n_samples', n_samples, 'cost.mean_beam_duty')
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


# ---------------------------------------------------------------------------
# Plain helper
# ---------------------------------------------------------------------------

def beam_duty_sharpness(n_samples: int) -> float:
    """Sigmoid steepness k [1/rad] matched to a ``cost.mean_beam_duty`` sample
    count: ``n_samples / (2 pi sqrt(5))``.

    The chief->member direction of a ``dlam = 0``, ``|di| <= |de|`` formation
    turns at most sqrt(5) rad per rad of longitude, so one sample sweeps up to
    ``2 pi sqrt(5) / n_samples`` rad of off-axis angle; this k makes the
    sigmoid edge (~1/k) that wide. Sharper k makes the duty a staircase in the
    ROE; softer k biases it upward. Pair it with ``n_samples >= 20 pi sqrt(5) /
    half_angle`` (the sampling rule in :func:`make_mean_beam_duty`).

    Raises ValueError unless ``n_samples`` is a positive integer.
    """
    n_samples = _require_count('n_samples', n_samples, 'beam_duty_sharpness')
    return n_samples / (2.0 * math.pi * math.sqrt(5.0))
