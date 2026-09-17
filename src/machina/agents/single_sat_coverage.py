"""
Layer 3 -- SingleSatCoverage agent type.

Optimizes Modified Equinoctial Elements (p, f, g, h, k) for a single satellite
to maximize time-average coverage over a fixed ground target.

Coverage is computed using static orbital geometry: the true longitude L is
sampled at N equally-spaced points across [0, 2*pi), the MEE state is
converted to ECI via transform.mee_to_eci at each sample, and the elevation
angle above the ground target is computed via geometry.elevation_angle.  A
smooth sigmoid function (cost.smooth_coverage) maps elevation to a [0, 1]
indicator value; the mean over all N sample points is the coverage metric.

No propagation (Phase 3b) is needed -- this is geometric optimization only.

Config structure
----------------
{
    'n_sample_points': 72,               # int; number of L samples
    'ground_target': {
        'lat_deg': 38.9,                 # float; geodetic latitude [deg]
        'lon_deg': -77.0,                # float; geodetic longitude [deg]
    },
    'coverage': {
        'min_elevation_deg': 10.0,       # float; minimum elevation threshold [deg]
        'sigmoid_k': 20.0,               # float; sigmoid steepness [1/rad]
    },
    'altitude_bounds': {
        'perigee_min_km': 200.0,         # float; minimum perigee altitude [km]
        'apogee_max_km': 40000.0,        # float; maximum apogee altitude [km]
    },
    'constants': {
        'mu':      398600.4418,          # float; Earth GM [km^3/s^2]
        'R_earth': 6378.137,             # float; Earth mean radius [km]
    },
}

All config keys have the values above as defaults.

Declared quantities (declare())
--------------------------------
Path                     Shape   Role             Default
orbital/p                (1,1)   flexible/var     R_earth + 500 [km]
orbital/f                (1,1)   flexible/var     0.0 (circular)
orbital/g                (1,1)   flexible/var     0.0 (circular)
orbital/h                (1,1)   flexible/var     tan(51.6deg/2) ~ 0.484
orbital/k                (1,1)   flexible/var     0.0 (RAAN = 0)
sample_points/L          (N,1)   always_parameter linspace(0, 2*pi, N+1)[:-1]
target/position          (3,1)   always_parameter computed from lat/lon
coverage/min_elevation   (1,1)   always_parameter min_elevation_deg -> rad
coverage/sigmoid_k       (1,1)   always_parameter sigmoid_k [1/rad]

Constraints returned by build()
---------------------------------
perigee_altitude : p/(1 + sqrt(f^2+g^2)) - R_earth >= perigee_min_km
apogee_altitude  : p/(1 - sqrt(f^2+g^2)) - R_earth <= apogee_max_km

Namespace (post-build)
-----------------------
orbital/p, orbital/f, orbital/g, orbital/h, orbital/k
orbital/sma          -- computed: p / (1 - f^2 - g^2)
orbital/ecc          -- computed: sqrt(f^2 + g^2)
orbital/inc          -- computed: 2*arctan(sqrt(h^2 + k^2)) [rad]
orbital/period       -- computed: 2*pi*sqrt(a^3 / mu) [s]
target/position      -- parameter (3,1) ECI [km]
coverage/min_elevation  -- parameter (1,1) [rad]
coverage/sigmoid_k   -- parameter (1,1) [1/rad]
coverage/total       -- computed: mean coverage in [0,1]
coverage/per_point/{i}  -- computed: per-sample-point coverage (i = 0..N-1)
"""

import math

import casadi as ca
import numpy as np

from machina.agents.agent_type import (
    AgentType,
    ConstraintDeclaration,
    QuantityDeclaration,
)
from machina.blocks import registry
from machina.blocks.descriptor import SymbolDescriptor

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    'n_sample_points': 72,
    'ground_target': {
        'lat_deg': 38.9,
        'lon_deg': -77.0,
    },
    'coverage': {
        'min_elevation_deg': 10.0,
        'sigmoid_k': 20.0,
    },
    'altitude_bounds': {
        'perigee_min_km': 200.0,
        'apogee_max_km': 40000.0,
    },
    'constants': {
        'mu': 398600.4418,
        'R_earth': 6378.137,
    },
}


def _merge_config(user_cfg: dict) -> dict:
    """Deep-merge user config over defaults (one level deep for nested dicts)."""
    merged = {}
    for key, default_val in _DEFAULT_CONFIG.items():
        if isinstance(default_val, dict):
            user_sub = user_cfg.get(key, {})
            merged[key] = {**default_val, **user_sub}
        else:
            merged[key] = user_cfg.get(key, default_val)
    return merged


# ---------------------------------------------------------------------------
# SingleSatCoverage
# ---------------------------------------------------------------------------

class SingleSatCoverage(AgentType):
    """
    Single-satellite coverage optimization agent.

    Optimizes MEE orbital elements to maximize time-average coverage
    fraction over a fixed ground target using static orbit geometry.

    Parameters
    ----------
    name : str
        Unique agent name (used as path prefix in the compiler).
    config : dict
        Configuration dict.  All keys are optional; see module docstring
        for defaults and expected structure.
    """

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        cfg = _merge_config(config)

        # Physical constants (Python floats -- baked into Layer 2 factories)
        self._mu      = float(cfg['constants']['mu'])
        self._R_earth = float(cfg['constants']['R_earth'])

        # Coverage parameters
        self._min_elevation_deg = float(cfg['coverage']['min_elevation_deg'])
        self._min_elevation_rad = math.radians(self._min_elevation_deg)
        self._sigmoid_k         = float(cfg['coverage']['sigmoid_k'])

        # Ground target
        self._lat_deg = float(cfg['ground_target']['lat_deg'])
        self._lon_deg = float(cfg['ground_target']['lon_deg'])
        self._lat_rad = math.radians(self._lat_deg)
        self._lon_rad = math.radians(self._lon_deg)

        # Altitude bounds
        self._perigee_min_km = float(cfg['altitude_bounds']['perigee_min_km'])
        self._apogee_max_km  = float(cfg['altitude_bounds']['apogee_max_km'])

        # Sampling
        self._n_sample_points = int(cfg['n_sample_points'])

        # Precompute target ECI position (spherical Earth, Python numpy)
        self._r_target_np = self._R_earth * np.array([
            math.cos(self._lat_rad) * math.cos(self._lon_rad),
            math.cos(self._lat_rad) * math.sin(self._lon_rad),
            math.sin(self._lat_rad),
        ])

        # Default initial guess: ISS-like orbit at 500 km, i=51.6 deg.
        # f=0.01 (small eccentricity) avoids the degenerate Jacobian of
        # sqrt(f^2+g^2) at f=g=0 (circular orbit singularity).
        _p0 = self._R_earth + 500.0
        _h0 = math.tan(math.radians(51.6) / 2.0)  # ~0.484
        self._orbital_defaults = {
            'p': _p0,
            'f': 0.01,   # small offset prevents zero-gradient at circular orbit
            'g': 0.0,
            'h': _h0,
            'k': 0.0,
        }

    # ------------------------------------------------------------------
    # declare()
    # ------------------------------------------------------------------

    def declare(self) -> list[QuantityDeclaration]:
        """
        Return 9 QuantityDeclarations: 5 orbital elements (flexible/variable)
        and 4 parameters (always_parameter).
        """
        N  = self._n_sample_points
        Re = self._R_earth

        # True longitude samples: N equally-spaced values in [0, 2*pi)
        L_samples = np.linspace(0.0, 2.0 * math.pi, N + 1)[:-1]

        return [
            # ---- MEE orbital elements (decision variables by default) ----
            QuantityDeclaration(
                path='orbital/p',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=self._orbital_defaults['p'],
                lb=100.0,
                ub=Re + self._apogee_max_km,
                description='Semi-latus rectum [km]',
                units='km',
                role='flexible',
                default_role='variable',
            ),
            QuantityDeclaration(
                path='orbital/f',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=self._orbital_defaults['f'],
                lb=-1.0,
                ub=1.0,
                description='MEE eccentricity vector x-component (= e*cos(omega+RAAN))',
                role='flexible',
                default_role='variable',
            ),
            QuantityDeclaration(
                path='orbital/g',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=self._orbital_defaults['g'],
                lb=-1.0,
                ub=1.0,
                description='MEE eccentricity vector y-component (= e*sin(omega+RAAN))',
                role='flexible',
                default_role='variable',
            ),
            QuantityDeclaration(
                path='orbital/h',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=self._orbital_defaults['h'],
                lb=-1.5,
                ub=1.5,
                description=(
                    'MEE inclination vector x-component (= tan(i/2)*cos(RAAN)). '
                    'Bounds [-1.5, 1.5] cover inclinations 0-123 deg. '
                    'Do not extend beyond +-2 — IPOPT Hessian may become ill-conditioned '
                    'for near-retrograde orbits (i > ~140 deg).'
                ),
                role='flexible',
                default_role='variable',
            ),
            QuantityDeclaration(
                path='orbital/k',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=self._orbital_defaults['k'],
                lb=-1.5,
                ub=1.5,
                description=(
                    'MEE inclination vector y-component (= tan(i/2)*sin(RAAN)). '
                    'Bounds [-1.5, 1.5] cover inclinations 0-123 deg.'
                ),
                role='flexible',
                default_role='variable',
            ),
            # ---- Parameters ----
            QuantityDeclaration(
                path='sample_points/L',
                shape=(N, 1),
                semantic_type='vector',
                default_value=L_samples,
                lb=-np.inf,
                ub=np.inf,
                description=f'True longitude sample points [rad] (N={N})',
                units='rad',
                role='always_parameter',
            ),
            QuantityDeclaration(
                path='target/position',
                shape=(3, 1),
                semantic_type='vector',
                default_value=self._r_target_np,
                lb=-np.inf,
                ub=np.inf,
                description=(
                    f'Ground target ECI position [km] '
                    f'(lat={self._lat_deg:.2f} deg, lon={self._lon_deg:.2f} deg)'
                ),
                units='km',
                frame='ECI',
                role='always_parameter',
            ),
            QuantityDeclaration(
                path='coverage/min_elevation',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=self._min_elevation_rad,
                lb=-np.inf,
                ub=np.inf,
                description=f'Minimum elevation threshold [rad] ({self._min_elevation_deg:.1f} deg)',
                units='rad',
                role='always_parameter',
            ),
            QuantityDeclaration(
                path='coverage/sigmoid_k',
                shape=(1, 1),
                semantic_type='scalar',
                default_value=self._sigmoid_k,
                lb=-np.inf,
                ub=np.inf,
                description='Sigmoid steepness parameter [1/rad]',
                units='1/rad',
                role='always_parameter',
            ),
        ]

    # ------------------------------------------------------------------
    # build()
    # ------------------------------------------------------------------

    def build(self, symbols: dict[str, ca.MX]) -> list[ConstraintDeclaration]:
        """
        Build coverage expressions for N sample points, populate namespace,
        and return altitude constraint declarations.

        Parameters
        ----------
        symbols : dict[str, ca.MX]
            Maps every declared path to an MX symbol created by the compiler.

        Returns
        -------
        list[ConstraintDeclaration]
            Two constraints: perigee altitude lower bound and apogee altitude
            upper bound.
        """
        # Unpack symbols
        p        = symbols['orbital/p']
        f        = symbols['orbital/f']
        g        = symbols['orbital/g']
        h        = symbols['orbital/h']
        k        = symbols['orbital/k']
        L_vec    = symbols['sample_points/L']       # MX (N,1) parameter
        r_target = symbols['target/position']       # MX (3,1) parameter
        min_elev = symbols['coverage/min_elevation']# MX (1,1) parameter
        sig_k    = symbols['coverage/sigmoid_k']    # MX (1,1) parameter

        # Fetch Layer 2 functions (mu baked into mee_to_eci at Python level)
        mee_to_eci_fd  = registry.get('transform.mee_to_eci')(mu=self._mu)
        elev_angle_fd  = registry.get('geometry.elevation_angle')()
        smooth_cov_fd  = registry.get('cost.smooth_coverage')()

        # ------------------------------------------------------------------
        # Per-sample-point coverage loop
        # ------------------------------------------------------------------
        N = self._n_sample_points
        coverage_terms = []

        for i in range(N):
            L_i   = L_vec[i]                           # MX scalar
            mee_i = ca.vertcat(p, f, g, h, k, L_i)    # MX (6,1)

            # MEE -> ECI position (ignore velocity)
            r_sat_i = mee_to_eci_fd.function(mee_i)[0]  # MX (3,1)

            # Elevation angle above ground target
            elev_i = elev_angle_fd.function(r_sat_i, r_target)  # MX (1,1)

            # Smooth coverage value
            cov_i = smooth_cov_fd.function(elev_i, min_elev, sig_k)  # MX (1,1)

            coverage_terms.append(cov_i)
            self._namespace[f'coverage/per_point/{i}'] = SymbolDescriptor(
                symbol=cov_i,
                name=f'cov_{i}',
                shape=(1, 1),
                semantic_type='scalar',
            )

        # ------------------------------------------------------------------
        # Aggregate coverage: mean over N sample points (dimensionless, [0,1])
        # ------------------------------------------------------------------
        total_cov = ca.sum1(ca.vertcat(*coverage_terms)) / N   # MX scalar (1,1)

        # ------------------------------------------------------------------
        # Derived orbital quantities
        # ------------------------------------------------------------------
        # Guard sqrt(f^2+g^2) against 0/0 in its Jacobian at f=g=0.
        # TINY = 1e-32 follows the same pattern as the Stumpff functions.
        _TINY = 1e-32
        e2_expr  = f ** 2 + g ** 2
        e_expr   = ca.sqrt(ca.fmax(e2_expr, _TINY))
        sma_expr = p / (1 - e2_expr)
        inc_expr = 2 * ca.atan(ca.sqrt(ca.fmax(h ** 2 + k ** 2, _TINY)))
        T_expr   = 2 * math.pi * ca.sqrt(sma_expr ** 3 / self._mu)

        # ------------------------------------------------------------------
        # Populate namespace
        # ------------------------------------------------------------------
        self._namespace['orbital/p'] = SymbolDescriptor(
            symbol=p, name='p', shape=(1, 1), semantic_type='scalar', units='km')
        self._namespace['orbital/f'] = SymbolDescriptor(
            symbol=f, name='f', shape=(1, 1), semantic_type='scalar')
        self._namespace['orbital/g'] = SymbolDescriptor(
            symbol=g, name='g', shape=(1, 1), semantic_type='scalar')
        self._namespace['orbital/h'] = SymbolDescriptor(
            symbol=h, name='h', shape=(1, 1), semantic_type='scalar')
        self._namespace['orbital/k'] = SymbolDescriptor(
            symbol=k, name='k', shape=(1, 1), semantic_type='scalar')
        self._namespace['orbital/sma'] = SymbolDescriptor(
            symbol=sma_expr, name='sma', shape=(1, 1), semantic_type='scalar', units='km')
        self._namespace['orbital/ecc'] = SymbolDescriptor(
            symbol=e_expr, name='ecc', shape=(1, 1), semantic_type='scalar')
        self._namespace['orbital/inc'] = SymbolDescriptor(
            symbol=inc_expr, name='inc', shape=(1, 1), semantic_type='scalar', units='rad')
        self._namespace['orbital/period'] = SymbolDescriptor(
            symbol=T_expr, name='period', shape=(1, 1), semantic_type='scalar', units='s')
        self._namespace['target/position'] = SymbolDescriptor(
            symbol=r_target, name='target_position', shape=(3, 1),
            semantic_type='vector', frame='ECI', units='km')
        self._namespace['coverage/min_elevation'] = SymbolDescriptor(
            symbol=min_elev, name='min_elevation', shape=(1, 1),
            semantic_type='scalar', units='rad')
        self._namespace['coverage/sigmoid_k'] = SymbolDescriptor(
            symbol=sig_k, name='sigmoid_k', shape=(1, 1), semantic_type='scalar')
        self._namespace['coverage/total'] = SymbolDescriptor(
            symbol=total_cov, name='coverage_total', shape=(1, 1),
            semantic_type='scalar',
            units='fraction')

        self._built = True

        # ------------------------------------------------------------------
        # Structural constraints: perigee and apogee altitude bounds.
        #
        # Formulation avoids sqrt(f^2+g^2) in the constraint expression,
        # because d/df[sqrt(f^2+g^2)] = f/sqrt(f^2+g^2) = 0/0 at f=g=0
        # (circular orbit), making the Jacobian degenerate at a common
        # initial guess.
        #
        # Instead use the squared equivalent (valid since p > 0):
        #   r_p >= R_min  <=>  p/(1+e) >= R_min
        #                <=>  (p - R_min)^2 >= R_min^2 * e^2
        #                <=>  (p - R_min)^2 - R_min^2 * (f^2+g^2) >= 0
        #
        #   r_a <= R_max  <=>  p/(1-e) <= R_max
        #                <=>  (R_max - p)^2 >= R_max^2 * e^2
        #                <=>  (R_max - p)^2 - R_max^2 * (f^2+g^2) >= 0
        #
        # Both forms are differentiable everywhere, including at f=g=0.
        # The sign ambiguity introduced by squaring is resolved by the
        # variable bounds: p >= R_earth+perigee_min and p <= R_earth+apogee_max.
        # ------------------------------------------------------------------
        R_min = self._R_earth + self._perigee_min_km   # minimum periapsis radius [km]
        R_max = self._R_earth + self._apogee_max_km    # maximum apoapsis radius [km]

        perigee_expr = (p - R_min) ** 2 - R_min ** 2 * e2_expr
        apogee_expr  = (R_max - p) ** 2 - R_max ** 2 * e2_expr

        return [
            ConstraintDeclaration(
                expr=perigee_expr,
                lb=0.0,
                ub=np.inf,
                name='perigee_altitude',
                description=(
                    f'Perigee altitude >= {self._perigee_min_km:.0f} km '
                    f'[(p - R_min)^2 >= R_min^2*(f^2+g^2)]'
                ),
            ),
            ConstraintDeclaration(
                expr=apogee_expr,
                lb=0.0,
                ub=np.inf,
                name='apogee_altitude',
                description=(
                    f'Apogee altitude <= {self._apogee_max_km:.0f} km '
                    f'[(R_max - p)^2 >= R_max^2*(f^2+g^2)]'
                ),
            ),
        ]
