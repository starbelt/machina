"""
Single-satellite coverage of one fixed ground target, in modified equinoctial
elements.

The April 2026 prototype's ``SingleSatCoverage`` (deleted in Phase 3b), ported
to the ``Component`` API. The physics is unchanged: the true longitude ``L`` is
sampled at ``N`` equally-spaced points across ``[0, 2*pi)``, each sample is
converted to ECI, the elevation angle above the target is taken, and a
sigmoid maps elevation to a ``[0, 1]`` indicator. The mean over the samples
is ``coverage_total``. There is no propagation -- this is static geometry.

Three things changed in the port, all of them rules the vault records:

* The component **produces no derived elements**. ``sma``, ``ecc``, ``inc``
  and ``period`` are display-only (Decision Log #33) and are computed from a
  solution, not wired into a cost or a constraint. Per-sample coverage is
  gone for a different reason: its shape depends on ``N``, and a registry
  signal has one fixed shape.
* ``n_sample_points`` defaults to **24**, not April's 72. 24 is the
  documented convergence finding; the 72 in the April code contradicted it.
* The component declares **no costs**. Coverage is a value to maximise, so
  the problem adds ``-coverage_total`` itself and owns the sign and the
  weight (Decision Log #20, #34).

``mu`` and ``R_earth`` are factory parameters, baked into the astro factories
at construction, never runtime inputs. Distances are kilometres throughout, a
recorded decision -- the MEE factories are written in km and the constraint
scaling below is tuned for them.
"""

import math

import casadi as ca
import numpy as np

from machina.astro.coverage import make_smooth_coverage
from machina.astro.geometry import make_elevation_angle
from machina.astro.transforms import make_mee_to_eci
from machina.model import Component, Constraint, Declaration, Quantity, Role

__all__ = ["SingleSatCoverage"]


class SingleSatCoverage(Component):
    """Five MEE elements, four parameters, one coverage number, two altitude rows.

    Parameters
    ----------
    name:
        Instance suffix. ``None`` gives the component its default name,
        ``single_sat_coverage``.
    target_lat_deg, target_lon_deg:
        Ground target, geodetic degrees. Converted once, here.
    n_sample_points:
        Number of true-longitude samples, ``N``. It sets the declared shape
        of ``L`` and the number of geometry evaluations in ``build()``.
    min_elevation_deg, sigmoid_k:
        The coverage mask: elevation at which the smooth indicator reads 0.5,
        and the steepness of the transition in ``1/rad``.
    perigee_min_km, apogee_max_km:
        Altitude box, as altitudes above ``R_earth``.
    mu, R_earth:
        Earth gravitational parameter ``[km^3/s^2]`` and mean radius ``[km]``.
        Baked into the factories at construction.
    """

    def __init__(self, name: str = None, *, target_lat_deg: float = 38.9,
                 target_lon_deg: float = -77.0, n_sample_points: int = 24,
                 min_elevation_deg: float = 10.0, sigmoid_k: float = 20.0,
                 perigee_min_km: float = 200.0, apogee_max_km: float = 40000.0,
                 mu: float = 398600.4418, R_earth: float = 6378.137):
        super().__init__(name)
        self.target_lat_deg = float(target_lat_deg)
        self.target_lon_deg = float(target_lon_deg)
        self.n_sample_points = int(n_sample_points)
        self.min_elevation_deg = float(min_elevation_deg)
        self.sigmoid_k = float(sigmoid_k)
        self.perigee_min_km = float(perigee_min_km)
        self.apogee_max_km = float(apogee_max_km)
        self.mu = float(mu)
        self.R_earth = float(R_earth)

    # --- derived numbers, all of them plain Python ------------------------------------------

    @property
    def R_min(self) -> float:
        """Smallest admissible perigee radius ``[km]``."""
        return self.R_earth + self.perigee_min_km

    @property
    def R_max(self) -> float:
        """Largest admissible apogee radius ``[km]``."""
        return self.R_earth + self.apogee_max_km

    @property
    def p_scale(self) -> float:
        """Nominal magnitude of ``p``: a radius, so of the order of ``R_earth``.

        This is the MEE Numerics scaling recipe (measured as 7000 km for Earth)
        derived from the body instead of hard-coded, so an instance built for
        another central body keeps the solver's view of ``p`` O(1).
        """
        return self.R_earth

    @property
    def altitude_scale(self) -> float:
        """Nominal magnitude of the squared altitude rows, ``p_scale ** 2``."""
        return self.p_scale ** 2

    def sample_longitudes(self) -> np.ndarray:
        """The ``(N, 1)`` true-longitude grid, ``N`` points across ``[0, 2 pi)``."""
        n = self.n_sample_points
        return np.linspace(0.0, 2.0 * math.pi, n + 1)[:-1].reshape(n, 1)

    def target_position(self) -> np.ndarray:
        """The target's ``(3, 1)`` ECI position ``[km]``, spherical Earth."""
        lat, lon = math.radians(self.target_lat_deg), math.radians(self.target_lon_deg)
        return self.R_earth * np.array([[math.cos(lat) * math.cos(lon)],
                                        [math.cos(lat) * math.sin(lon)],
                                        [math.sin(lat)]])

    # --- declare ----------------------------------------------------------------------------

    def declare(self) -> Declaration:
        n = self.n_sample_points
        return Declaration(
            # Order is April's, so the decision vector is laid out identically.
            quantities=(
                Quantity(
                    "p", unit="km", doc="Semi-latus rectum",
                    role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                    default=self.R_earth + 500.0,
                    lb=100.0, ub=self.R_earth + self.apogee_max_km,
                    scale=self.p_scale,
                    provenance="A", source="ISS-like starting orbit; MEE Numerics Rules 3-4"),
                Quantity(
                    "f", unit="1",
                    doc="MEE eccentricity vector x-component, e cos(omega + RAAN)",
                    role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                    default=0.01, lb=-1.0, ub=1.0,
                    provenance="A",
                    source="non-zero f avoids the circular-orbit singularity (Rule 3)"),
                Quantity(
                    "g", unit="1",
                    doc="MEE eccentricity vector y-component, e sin(omega + RAAN)",
                    role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                    default=0.0, lb=-1.0, ub=1.0,
                    provenance="A",
                    source="g = 0 at the start; f carries the non-zero eccentricity (Rule 3)"),
                Quantity(
                    "h", unit="1",
                    doc="MEE inclination vector x-component, tan(i/2) cos(RAAN). The bounds "
                        "[-1.5, 1.5] on h and k cover inclinations up to 112.6 deg at any RAAN "
                        "(129.5 deg at the box corners); do not extend them past +-2, where "
                        "the Hessian goes ill-conditioned",
                    role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                    default=math.tan(math.radians(51.6) / 2.0), lb=-1.5, ub=1.5,
                    provenance="A", source="i = 51.6 deg, RAAN = 0; bounds per Rule 5"),
                Quantity(
                    "k", unit="1",
                    doc="MEE inclination vector y-component, tan(i/2) sin(RAAN). The bounds "
                        "[-1.5, 1.5] on h and k cover inclinations up to 112.6 deg at any RAAN "
                        "(129.5 deg at the box corners)",
                    role=Role.FLEXIBLE, default_role=Role.VARIABLE,
                    default=0.0, lb=-1.5, ub=1.5,
                    provenance="A", source="i = 51.6 deg, RAAN = 0; bounds per Rule 5"),
                Quantity(
                    "L", shape=(n, 1), unit="rad", role=Role.PARAMETER,
                    doc=f"True longitude sample points (N = {n})",
                    default=self.sample_longitudes(),
                    provenance="A", source=f"uniform true-longitude grid, N = {n} (Rule 6)"),
                Quantity(
                    "r_target", shape=(3, 1), unit="km", frame="eci", role=Role.PARAMETER,
                    doc=f"Ground target position (lat = {self.target_lat_deg:.2f} deg, "
                        f"lon = {self.target_lon_deg:.2f} deg)",
                    default=self.target_position(),
                    provenance="A",
                    source="a chosen site, placed on a spherical Earth of radius R_earth; "
                           "snapshot ECI geometry"),
                Quantity(
                    "min_elevation", unit="rad", role=Role.PARAMETER,
                    doc=f"Elevation at which the coverage indicator reads 0.5 "
                        f"({self.min_elevation_deg:.1f} deg)",
                    default=math.radians(self.min_elevation_deg),
                    provenance="A", source="elevation mask"),
                Quantity(
                    "sigmoid_k", unit="1/rad", role=Role.PARAMETER,
                    doc="Steepness of the smooth coverage indicator; larger is a sharper "
                        "transition about min_elevation",
                    default=self.sigmoid_k,
                    provenance="A", source="April tuning of the smooth coverage indicator"),
            ),
            produces=("coverage_total",),
            # Squared altitude forms, per MEE Numerics Rules 1-2: the naive
            # p / (1 +- sqrt(f^2 + g^2)) has a degenerate Jacobian at e = 0, which is a
            # common initial guess, and IPOPT declares the problem infeasible there.
            constraints=(
                Constraint(
                    "perigee_altitude", lb=0.0, scale=self.altitude_scale,
                    doc=f"Perigee radius >= R_earth + {self.perigee_min_km:g} km, written "
                        f"(p - R_min)^2 - R_min^2 (f^2 + g^2) >= 0"),
                Constraint(
                    "apogee_altitude", lb=0.0, scale=self.altitude_scale,
                    doc=f"Apogee radius <= R_earth + {self.apogee_max_km:g} km, written "
                        f"(R_max - p)^2 - R_max^2 (f^2 + g^2) >= 0"),
            ),
        )

    # --- build ------------------------------------------------------------------------------

    def build(self, helpers: dict) -> dict:
        n = self.n_sample_points
        mee_to_eci = make_mee_to_eci(mu=self.mu)
        elevation_angle = make_elevation_angle()
        smooth_coverage = make_smooth_coverage()

        # Dense symbols of the declared shapes, named as the quantities are: the builder
        # binds a Function's inputs by name and refuses a sparse or mis-shaped argument.
        p = ca.SX.sym("p")
        f = ca.SX.sym("f")
        g = ca.SX.sym("g")
        h = ca.SX.sym("h")
        k = ca.SX.sym("k")
        longitudes = ca.SX.sym("L", n, 1)
        r_target = ca.SX.sym("r_target", 3, 1)
        min_elevation = ca.SX.sym("min_elevation")
        sigmoid_k = ca.SX.sym("sigmoid_k")

        # Composed at SX level through .function(...), not the validating __call__.
        terms = []
        for i in range(n):
            mee_i = ca.vertcat(p, f, g, h, k, longitudes[i])
            r_sat = mee_to_eci.function(mee_i)[0]
            elevation = elevation_angle.function(r_sat, r_target)
            terms.append(smooth_coverage.function(elevation, min_elevation, sigmoid_k))
        coverage_total = ca.sum1(ca.vertcat(*terms)) / n

        coverage = ca.Function(
            f"{self.name}_g",
            [p, f, g, h, k, longitudes, r_target, min_elevation, sigmoid_k],
            [coverage_total],
            ["p", "f", "g", "h", "k", "L", "r_target", "min_elevation", "sigmoid_k"],
            ["coverage_total"],
        )

        e_squared = f ** 2 + g ** 2
        altitudes = ca.Function(
            f"{self.name}_h",
            [p, f, g],
            [(p - self.R_min) ** 2 - self.R_min ** 2 * e_squared,
             (self.R_max - p) ** 2 - self.R_max ** 2 * e_squared],
            ["p", "f", "g"],
            ["perigee_altitude", "apogee_altitude"],
        )

        return {"g": coverage, "h": altitudes}
