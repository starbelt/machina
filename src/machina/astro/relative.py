"""
Relative motion of a period-matched (co-located) formation about a
near-circular, near-equatorial chief, and the synodic period of two circular
orbits.

Registered functions
--------------------
transform.roe_to_rtn       -- relative orbital elements -> chief-RTN position
geometry.rn_min_separation -- minimum radial/normal separation of an e/i formation
util.synodic_period        -- synodic period of two circular orbits

Plain helpers (not registered): :func:`ring_pass_interval`,
:func:`ring_size_for_interval`.

Importing this module registers the three factories; ``import machina.astro``
imports it.

Relative orbital elements (ROE)
-------------------------------
The relative vectors are longitude-referenced (equinoctial), so they stay
non-singular at i = 0, where the classical ``delta i_y = delta Omega sin i``
is not. With chief (c) and deputy (d) MEE ``(p, f, g, h, k, L)``::

    roe = [dlam, df, dg, dix, diy]
    dlam       = lam_d - lam_c                  (mean longitudes)
    (df, dg)   = (f_d - f_c, g_d - g_c)         (relative eccentricity vector)
    (dix, diy) = 2 (h_d - h_c, k_d - k_c)       (relative inclination vector)

The last line is the small-inclination form of ``i (cos Omega, sin Omega)``:
``2 tan(i/2) = i + O(i^3)``.

Linear relative motion
----------------------
For a chief of semi-major axis ``a`` on a circular equatorial orbit at
longitude ``lam`` (mean = true longitude) and a deputy with the **same**
semi-major axis (``delta a = 0``, period-matched, the co-located case), the
deputy position in the chief's RTN frame (R radial out, T along-track,
N orbit normal) is, to first order in the deltas::

    R = -a (df cos lam + dg sin lam)
    T =  a dlam + 2 a (df sin lam - dg cos lam)
    N =  a (dix sin lam - diy cos lam)

Derivation. Deputy radius ``r = a (1 - e cos M) + O(e^2)`` with
``e cos M = f cos lam_d + g sin lam_d`` (``M = lam_d - varpi``); true longitude
``L = lam_d + 2 e sin M + O(e^2)`` with ``e sin M = f sin lam_d - g cos lam_d``;
out of plane ``z = r sin i sin(L - Omega) = a (2h sin lam - 2k cos lam)``
to first order. R is ``r - a``, T is ``a (L_d - lam)``, N is ``z``. The
residual against exact two-body positions is ``O(a |delta|^2)``
(``tests/astro/test_relative_motion.py`` measures it through
``transform.mee_to_eci``). A chief inclination ``i_c`` adds ``O(i_c |delta|)``
terms: the frame assumes a near-equatorial chief.

``delta a != 0`` is out of scope: it adds a secular along-track drift
``-(3/2) n delta a t`` and a radial offset ``delta a`` that this map omits.
"""

import math

import casadi as ca

from machina.library.numerics import TINY, safe_divide, safe_sqrt
from machina.library.registry import register
from machina.model.descriptor import FunctionDescriptor

__all__ = [
    "make_roe_to_rtn",
    "make_rn_min_separation",
    "make_synodic_period",
    "ring_pass_interval",
    "ring_size_for_interval",
]


def _require_positive(name: str, value: float, where: str) -> float:
    """``float(value)``, or a ValueError naming the factory when it is not > 0."""
    value = float(value)
    if not (value > 0.0 and math.isfinite(value)):
        raise ValueError(f"{where}: {name} must be a positive finite number, got {value!r}.")
    return value


# ---------------------------------------------------------------------------
# transform.roe_to_rtn
# ---------------------------------------------------------------------------

@register('transform.roe_to_rtn')
def make_roe_to_rtn(*, a: float) -> FunctionDescriptor:
    """
    Factory for the linear map from relative orbital elements to the deputy's
    position in the chief's RTN frame, period-matched formation (delta a = 0).

    Factory parameters
    ------------------
    a : float
        Chief semi-major axis [km]. Baked in.

    Function interface
    ------------------
    Inputs
        roe : (5,1)  -- [dlam, df, dg, dix, diy] (see the module docstring) [rad, 1]
        lam : (1,1)  -- chief longitude (mean = true, circular chief) [rad]
    Output
        r_rtn : (3,1)  -- deputy position relative to the chief, RTN [km]

    Formulas
    --------
        R = -a (df cos lam + dg sin lam)
        T =  a dlam + 2 a (df sin lam - dg cos lam)
        N =  a (dix sin lam - diy cos lam)

    Valid to first order in the deltas for a near-circular, near-equatorial
    chief and a deputy of the same semi-major axis; the error is
    O(a |delta|^2). Linear, so no guard is needed.
    """
    a = _require_positive('a', a, 'transform.roe_to_rtn')

    roe = ca.SX.sym('roe', 5)
    lam = ca.SX.sym('lam')
    dlam, df, dg, dix, diy = roe[0], roe[1], roe[2], roe[3], roe[4]
    c = ca.cos(lam)
    s = ca.sin(lam)

    r_rtn = ca.vertcat(
        -a * (df * c + dg * s),
        a * dlam + 2 * a * (df * s - dg * c),
        a * (dix * s - diy * c),
    )
    fn = ca.Function('roe_to_rtn', [roe, lam], [r_rtn], ['roe', 'lam'], ['r_rtn'])
    return FunctionDescriptor(
        fn, description=f'ROE -> chief RTN position, linear, delta a = 0 (a={a} km)')


# ---------------------------------------------------------------------------
# geometry.rn_min_separation
# ---------------------------------------------------------------------------

@register('geometry.rn_min_separation')
def make_rn_min_separation(*, a: float) -> FunctionDescriptor:
    """
    Factory for the minimum radial/normal separation of a period-matched e/i
    formation over one orbit.

    Along-track position is not constrained by e/i separation (it depends on
    dlam, which drifts under perturbations), so the safety figure is the
    closest approach in the R-N plane. From :func:`make_roe_to_rtn`, with
    ``v = (cos lam, sin lam)``, ``de = (df, dg)``, ``di = (dix, diy)`` and
    ``w = (-diy, dix)``::

        R^2 + N^2 = a^2 v^T M v,        M = de de^T + w w^T
        tr M  = |de|^2 + |di|^2
        det M = (de x w)^2 = (de . di)^2
        min over lam = a^2 lambda_min(M)

    The eigenvalue is written without cancellation::

        disc       = tr^2 - 4 det = (|de|^2 - |di|^2)^2 + 4 (de x di)^2   (>= 0)
        lambda_min = (tr - sqrt(disc)) / 2 = 2 det / (tr + sqrt(disc))
        d_min      = a sqrt(lambda_min)

    Parallel (or antiparallel) vectors give ``d_min = a min(|de|, |di|)``;
    perpendicular ones give 0 (the relative orbit crosses the along-track axis).

    Factory parameters
    ------------------
    a : float
        Chief semi-major axis [km]. Baked in.

    Function interface
    ------------------
    Inputs
        de : (2,1)  -- relative eccentricity vector (df, dg) [1]
        di : (2,1)  -- relative inclination vector (dix, diy) [rad]
    Output
        d_min : (1,1)  -- minimum sqrt(R^2 + N^2) over one orbit [km]

    Numerics
    --------
    The eigenvalue is evaluated on ``a de`` and ``a di`` (km), so ``lambda_min``
    is in km^2 and the guards' floors are negligible for any physical
    formation. Both square roots are ``safe_sqrt``: ``disc`` is 0 when |de| = |di| and the
    vectors are parallel (the two eigenvalues cross), and ``lambda_min`` is 0
    for perpendicular vectors or de = 0. The true function has a kink at both;
    the guards return a finite subgradient there. The division is guarded with
    ``safe_divide`` (its denominator ``tr + sqrt(disc)`` is 0 only when both
    vectors vanish). Valid under the assumptions of ``transform.roe_to_rtn``.
    """
    a = _require_positive('a', a, 'geometry.rn_min_separation')

    de = ca.SX.sym('de', 2)
    di = ca.SX.sym('di', 2)

    # Scaled to km first, so the TINY floors (km^2, km^4) sit far below any
    # physical formation; in the dimensionless deltas (|delta|^2 ~ 1e-14 for a
    # 4 m GEO formation) sqrt(TINY) = 1e-16 would be a visible bias.
    u = a * de
    w = a * di
    u2 = ca.dot(u, u)
    w2 = ca.dot(w, w)
    dot = u[0] * w[0] + u[1] * w[1]
    cross = u[0] * w[1] - u[1] * w[0]

    trace = u2 + w2
    det = dot ** 2
    disc = (u2 - w2) ** 2 + 4 * cross ** 2
    lambda_min = safe_divide(2 * det, trace + safe_sqrt(disc))
    d_min = safe_sqrt(lambda_min)

    fn = ca.Function('rn_min_separation', [de, di], [d_min], ['de', 'di'], ['d_min'])
    return FunctionDescriptor(
        fn, description=f'Minimum R-N separation of an e/i formation (a={a} km)')


# ---------------------------------------------------------------------------
# util.synodic_period
# ---------------------------------------------------------------------------

@register('util.synodic_period')
def make_synodic_period(*, mu: float) -> FunctionDescriptor:
    """
    Factory for the synodic period of two coplanar circular orbits.

        n_i   = sqrt(mu / a_i^3)
        T_syn = 2 pi / |n_1 - n_2|

    For a small radius difference ``dr = |a_1 - a_2|`` about ``a``,
    ``T_syn ~ (2/3) T a / dr`` with ``T = 2 pi / n``; the relative error of that
    approximation is O(dr / a).

    Factory parameters
    ------------------
    mu : float
        Gravitational parameter [km^3/s^2]. Baked in.

    Function interface
    ------------------
    Inputs
        a1 : (1,1)  -- radius of the first orbit [km]
        a2 : (1,1)  -- radius of the second orbit [km]
    Output
        T_syn : (1,1)  -- synodic period [s]

    Numerics
    --------
    ``|n_1 - n_2|`` is floored at ``TINY``, so equal radii return
    ``2 pi / TINY ~ 6e32 s``: "never", not a division by zero. ``mu / a^3`` is a
    ``safe_divide`` (radii are positive by construction).
    """
    mu = _require_positive('mu', mu, 'util.synodic_period')

    a1 = ca.SX.sym('a1')
    a2 = ca.SX.sym('a2')
    n1 = safe_sqrt(safe_divide(mu, a1 ** 3))
    n2 = safe_sqrt(safe_divide(mu, a2 ** 3))
    T_syn = 2 * math.pi / ca.fmax(ca.fabs(n1 - n2), TINY)

    fn = ca.Function('synodic_period', [a1, a2], [T_syn], ['a1', 'a2'], ['T_syn'])
    return FunctionDescriptor(
        fn, description=f'Synodic period of two circular orbits (mu={mu} km^3/s^2)')


# ---------------------------------------------------------------------------
# Plain helpers
# ---------------------------------------------------------------------------

def ring_pass_interval(T_syn: float, n: int) -> float:
    """Time between passes [s] of ``n`` equally phased ring members past one
    spacecraft on a neighbouring orbit: ``T_syn / n``.

    Raises ValueError unless ``T_syn`` is positive and finite and ``n`` is a
    positive integer.
    """
    T_syn = _require_positive('T_syn', T_syn, 'ring_pass_interval')
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError(f"ring_pass_interval: n must be a positive integer, got {n!r}.")
    return T_syn / int(n)


def ring_size_for_interval(T_syn: float, T_target: float) -> int:
    """Smallest ring size whose pass interval is at most ``T_target``:
    ``ceil(T_syn / T_target)``.

    Raises ValueError unless both arguments are positive and finite.
    """
    T_syn = _require_positive('T_syn', T_syn, 'ring_size_for_interval')
    T_target = _require_positive('T_target', T_target, 'ring_size_for_interval')
    return math.ceil(T_syn / T_target)
