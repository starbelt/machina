"""
Geometry factories for orbital mechanics.

Registered functions
--------------------
geometry.ground_target_eci  -- Geodetic lat/lon -> ECI position (spherical Earth)
geometry.elevation_angle    -- Elevation angle of satellite above ground target horizon

Both functions use the spherical Earth approximation and snapshot geometry
(Earth rotation is not modelled).  The target is treated as stationary in ECI.
"""

import casadi as ca

from machina.blocks.descriptor import FunctionDescriptor
from machina.blocks.registry import register

_TINY = 1e-32   # Guard for division by near-zero norms (same pattern as transforms.py)


# ---------------------------------------------------------------------------
# geometry.ground_target_eci
# ---------------------------------------------------------------------------

@register('geometry.ground_target_eci')
def make_ground_target_eci(*, R_earth: float) -> FunctionDescriptor:
    """
    Factory for geodetic lat/lon -> ECI position on a spherical Earth.

    R_earth is baked in at factory time (consistent with the mee_to_eci(mu=...)
    pattern).  This keeps the function interface minimal and matches the
    project convention that physical constants are factory parameters.

    Factory parameters
    ------------------
    R_earth : float
        Mean Earth radius [km] (or any consistent length unit).
        Typical value: 6378.137 km (WGS-84 equatorial radius).

    Function interface
    ------------------
    Inputs
        lat : (1,1)  -- geodetic latitude  [rad]
        lon : (1,1)  -- geodetic longitude [rad]
    Output
        r_target : (3,1)  -- ECI position [km] (same units as R_earth)

    Notes
    -----
    Uses the spherical Earth approximation:
        r_target = R_earth * [cos(lat)*cos(lon),
                               cos(lat)*sin(lon),
                               sin(lat)]
    Earth rotation is not modelled — the target is stationary in ECI.
    For Phase 3c coverage geometry this is an adequate approximation.
    """
    lat = ca.SX.sym('lat')
    lon = ca.SX.sym('lon')

    r_target = ca.vertcat(
        R_earth * ca.cos(lat) * ca.cos(lon),
        R_earth * ca.cos(lat) * ca.sin(lon),
        R_earth * ca.sin(lat),
    )

    fn = ca.Function(
        'ground_target_eci',
        [lat, lon], [r_target],
        ['lat', 'lon'], ['r_target'],
    )
    return FunctionDescriptor(
        fn,
        description=(
            f'Geodetic lat/lon [rad] -> ECI [km] '
            f'(R_earth={R_earth} km, spherical Earth)'
        ),
    )


# ---------------------------------------------------------------------------
# geometry.elevation_angle
# ---------------------------------------------------------------------------

@register('geometry.elevation_angle')
def make_elevation_angle() -> FunctionDescriptor:
    """
    Factory for satellite elevation angle above a ground target's horizon.

    Computes the angle between the range vector (target -> satellite) and
    the local horizontal plane at the target.  Positive elevation means the
    satellite is above the horizon; zero means on the horizon; negative means
    below the horizon.

    Uses the spherical Earth approximation: the local vertical at the target
    is the geocentric radial direction r_target / ||r_target||.

    Function interface
    ------------------
    Inputs
        r_sat    : (3,1)  -- satellite ECI position [km]
        r_target : (3,1)  -- ground target ECI position [km]
    Output
        elevation : (1,1)  -- elevation angle [rad]
                              Range: [-pi/2, pi/2]
                              Positive = above horizon, negative = below.

    Math
    ----
        rho     = r_sat - r_target              (range vector)
        rho_hat = rho / ||rho||                 (unit range vector)
        n_hat   = r_target / ||r_target||       (local vertical at target)
        epsilon = arcsin(clamp(rho_hat . n_hat, -1, 1))

    Notes
    -----
    The _TINY guard on ||rho|| prevents NaN when r_sat == r_target
    (degenerate case, not physical).  The dot-product is clamped to
    [-1, 1] before arcsin to avoid domain errors from floating-point noise.
    """
    r_sat    = ca.SX.sym('r_sat',    3)
    r_target = ca.SX.sym('r_target', 3)

    # Range vector
    rho      = r_sat - r_target
    rho_norm = ca.norm_2(rho)
    rho_hat  = rho / ca.fmax(rho_norm, _TINY)

    # Local vertical at target (geocentric radial direction)
    r_target_norm = ca.norm_2(r_target)
    n_hat         = r_target / ca.fmax(r_target_norm, _TINY)

    # Elevation angle via arcsin; clamp argument to avoid domain errors
    dot_val     = ca.dot(rho_hat, n_hat)
    dot_clamped = ca.fmin(ca.fmax(dot_val, -1.0), 1.0)
    elevation   = ca.asin(dot_clamped)

    fn = ca.Function(
        'elevation_angle',
        [r_sat, r_target], [elevation],
        ['r_sat', 'r_target'], ['elevation'],
    )
    return FunctionDescriptor(
        fn,
        description='Elevation angle [rad] of satellite above ground target horizon',
    )
