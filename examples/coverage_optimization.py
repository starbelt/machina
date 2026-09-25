"""
examples/coverage_optimization.py
---------------------------------
Single-satellite coverage of Washington DC, on ``Problem`` and the astro pack.

Three demonstrations:
  1. Elevation profile over Washington DC:
     The astro factories evaluated straight from the registry
     (``transform.koe_to_mee``, ``transform.mee_to_eci``,
     ``geometry.ground_target_eci``, ``geometry.elevation_angle``,
     ``cost.smooth_coverage``) for an ISS-like orbit with RAAN chosen so the
     orbit passes over DC (in snapshot ECI geometry, RAAN = 240 deg gives
     ~65 deg max elevation). Shows how the sigmoid coverage indicator
     transitions sharply near the 10 deg elevation mask.

  2. Coverage optimization:
     The ``SingleSatCoverage`` component in ``Scope("sat", ...)``, compiled by
     ``Problem``: maximize the mean coverage fraction over DC. Starting from an
     ISS-like orbit (500 km, f = 0.01, 51.6 deg, RAAN = 240 deg), IPOPT adjusts
     p, f, g, h, k inside a 200-1600 km altitude box, |f|, |g| <= 0.3 and
     |h|, |k| <= 1.5 (tan(i/2) = sqrt(h^2 + k^2): inclinations up to 112.6 deg
     at any RAAN, 129.5 deg at the corners of the box). The component declares
     its own scaling (p / R_earth, altitude rows / R_earth^2), so the strict
     IPOPT defaults converge. The April 2026 prototype's recipe (unit scale,
     acceptable_tol = 1e-2) runs beside it for comparison, together with the
     scaled problem under the same April tolerances.

  3. Coverage vs. inclination sweep:
     Evaluates the coverage fraction for a fixed circular orbit at 700 km
     as inclination varies from 0 to 110 deg, keeping the best of six RAAN
     values at each step. Shows the coverage landscape and the trade-off
     between inclination and ground track geometry.

Note on snapshot geometry
--------------------------
The coverage model uses "snapshot" ECI geometry -- the ground target is fixed
in ECI (Earth rotation is not modelled). This is a valid approximation for
instantaneous geometry calculations in the NLP, but the quantitative coverage
fractions are not directly comparable to time-averaged values that account for
Earth rotation. The optimizer is internally consistent and finds the orbit
that maximizes coverage in this static geometry, which captures the essential
inclination and altitude trade-offs.

Distances are km and angles radians in every signal (the MEE factories are
written in km); degrees appear only in printed text and plot labels.

Run from the project root:
    python examples/coverage_optimization.py
"""

import math

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Importing the pack declares its frames and signals and registers its factories
# (transform.*, geometry.*, cost.smooth_coverage) -- the registry.get() calls below need it.
from machina.astro import SingleSatCoverage
from machina.compiler import Problem
from machina.library import registry
from machina.model import Scope

# ---------------------------------------------------------------------------
# Constants and target
# ---------------------------------------------------------------------------

MU      = 398600.4418   # km^3/s^2, baked into the factories (a factory parameter)
R_EARTH = 6378.137      # km

LAT_DC    = 38.9        # deg  (Washington DC)
LON_DC    = -77.0       # deg
MIN_ELEV  = 10.0        # deg  elevation mask
SIGMOID_K = 20.0        # 1/rad, steepness of the smooth coverage indicator

# Demo 2's start, box and April options: the recipe that
# tests/astro/test_single_sat_coverage.py::TestOracles pins.
SCOPE          = "sat"
START_INC_DEG  = 51.6
START_RAAN_DEG = 240.0
ELEMENT_BOUNDS = (("p", R_EARTH + 200.0, R_EARTH + 1600.0), ("f", -0.30, 0.30),
                  ("g", -0.30, 0.30), ("h", -1.5, 1.5), ("k", -1.5, 1.5))
APRIL_OPTS     = {"ipopt.tol": 1e-4, "ipopt.acceptable_tol": 1e-2, "ipopt.acceptable_iter": 3}

# ---------------------------------------------------------------------------
# Factories from the registry
# ---------------------------------------------------------------------------

koe_to_mee_fd = registry.get("transform.koe_to_mee")()
mee_to_koe_fd = registry.get("transform.mee_to_koe")()
mee_to_eci_fd = registry.get("transform.mee_to_eci")(mu=MU)
gt_eci_fd     = registry.get("geometry.ground_target_eci")(R_earth=R_EARTH)
elev_angle_fd = registry.get("geometry.elevation_angle")()
smooth_cov_fd = registry.get("cost.smooth_coverage")()

# Ground target ECI position (DC, snapshot geometry)
r_dc_dm = gt_eci_fd.function(ca.DM(math.radians(LAT_DC)), ca.DM(math.radians(LON_DC)))
r_dc    = np.array(r_dc_dm).flatten()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def koe_to_mee_vals(a, e, i_deg, raan_deg=0.0, aop_deg=0.0, nu_deg=0.0):
    """Return dict of MEE values from KOE."""
    koe = ca.DM([a, e, math.radians(i_deg),
                 math.radians(raan_deg), math.radians(aop_deg), math.radians(nu_deg)])
    mee = koe_to_mee_fd.function(koe)
    return {c: float(mee[i]) for i, c in enumerate("pfghkL")}


def elevation_profile(p, f, g, h, k, n_pts=360):
    """Evaluate elevation over DC at n_pts equally-spaced true longitudes."""
    L_arr = np.linspace(0, 2 * math.pi, n_pts + 1)[:-1]
    elevs = []
    for L in L_arr:
        mee_i = ca.DM([p, f, g, h, k, L])
        r_sat, _ = mee_to_eci_fd.function(mee_i)
        elev_rad = float(elev_angle_fd.function(r_sat, r_dc_dm))
        elevs.append(math.degrees(elev_rad))
    return L_arr, np.array(elevs)


def coverage_mean(p, f, g, h, k, n_pts=72, k_sig=SIGMOID_K, emin_deg=MIN_ELEV):
    """Evaluate mean smooth coverage over DC."""
    emin_rad = math.radians(emin_deg)
    L_arr = np.linspace(0, 2 * math.pi, n_pts + 1)[:-1]
    total = 0.0
    for L in L_arr:
        mee_i = ca.DM([p, f, g, h, k, L])
        r_sat, _ = mee_to_eci_fd.function(mee_i)
        elev = elev_angle_fd.function(r_sat, r_dc_dm)
        cov = smooth_cov_fd.function(elev, ca.DM(emin_rad), ca.DM(k_sig))
        total += float(cov)
    return total / n_pts


def keplerian_elements(values: dict) -> dict:
    """a, e, i, RAAN and the perigee/apogee altitudes of the ``sat`` orbit.

    ``values`` maps ``sat/p``, ``sat/f``, ``sat/g``, ``sat/h`` and ``sat/k``
    to numbers or one-element arrays: a ``SolutionResult.x_opt``, the dict
    ``Problem.evaluate(result)`` returns, or a plain dict. The elements go
    through ``transform.mee_to_koe`` with L = 0 (true longitude enters none of
    a, e, i, RAAN). Returns km for ``a`` and the altitudes, radians for ``i``
    and ``raan`` (RAAN wrapped to [0, 2 pi)).

    Display only. Derived elements are never wired into a cost or a
    constraint -- the NLP couples components in raw p, f, g, h, k, and the
    altitude limits are the component's squared MEE rows.
    """
    mee = [np.asarray(values[f"{SCOPE}/{name}"], dtype=float).item() for name in "pfghk"]
    koe = np.asarray(mee_to_koe_fd.function(ca.DM(mee + [0.0])).full(), dtype=float).ravel()
    a, e, inc, raan = (float(x) for x in koe[:4])
    return {"a": a, "e": e, "i": inc, "raan": raan % (2.0 * math.pi),
            "perigee_altitude": a * (1.0 - e) - R_EARTH,
            "apogee_altitude": a * (1.0 + e) - R_EARTH}


def start_elements() -> dict:
    """Demo 2's initial guess: ISS-like, 500 km, f = 0.01 (never exactly 0), RAAN 240 deg."""
    tan_half_i = math.tan(math.radians(START_INC_DEG) / 2.0)
    raan = math.radians(START_RAAN_DEG)
    return {"p": R_EARTH + 500.0, "f": 0.01, "g": 0.0,
            "h": tan_half_i * math.cos(raan), "k": tan_half_i * math.sin(raan)}


def coverage_problem(solver_opts=None, *, unit_scale=False) -> Problem:
    """The coverage NLP, compiled and built.

    The component owns the physics, the altitude rows and their scaling; the
    problem owns the start, the box, the solver options and the objective.
    ``unit_scale=True`` overrides the declared scales back to 1 (April's
    unscaled problem).
    """
    sat = SingleSatCoverage(target_lat_deg=LAT_DC, target_lon_deg=LON_DC, n_sample_points=24,
                            min_elevation_deg=MIN_ELEV, sigmoid_k=SIGMOID_K,
                            perigee_min_km=200.0, apogee_max_km=1600.0, mu=MU, R_earth=R_EARTH)
    problem = Problem([Scope(SCOPE, [sat])], solver_opts=solver_opts, verbose=False)

    start = start_elements()
    overrides = {f"{SCOPE}/{name}": {"x0": start[name], "lb": lb, "ub": ub}
                 for name, lb, ub in ELEMENT_BOUNDS}
    if unit_scale:
        overrides[f"{SCOPE}/p"]["scale"] = 1.0
        overrides[f"{SCOPE}/perigee_altitude"] = {"scale": 1.0}
        overrides[f"{SCOPE}/apogee_altitude"] = {"scale": 1.0}
    problem.compile(overrides=overrides)

    # The component declares no cost: coverage is a value, and the problem owns the sign.
    problem.add_cost(-problem.expr(f"{SCOPE}/coverage_total").symbol, name="neg_coverage")
    return problem.build()


# ---------------------------------------------------------------------------
# Demo 1 - Elevation and coverage profile over Washington DC
# ---------------------------------------------------------------------------

def demo_elevation_profile():
    print("=" * 60)
    print("Demo 1 - Elevation profile over Washington DC")
    print("=" * 60)

    # ISS-like orbit, RAAN=240 deg chosen so orbit passes over DC
    # (in ECI snapshot geometry, RAAN=240 deg gives max elevation ~65 deg for DC)
    a_demo = 6778.0     # km (ISS semi-major axis)
    e_demo = 0.001
    i_demo = 51.6       # deg
    raan_demo = 240.0   # deg -- this RAAN puts DC in the orbit's ground track

    mee_demo = koe_to_mee_vals(a_demo, e_demo, i_demo, raan_demo)
    L_arr, elevs = elevation_profile(*(mee_demo[c] for c in "pfghk"), n_pts=360)
    covs = np.array([float(smooth_cov_fd.function(
        ca.DM(math.radians(e_deg)), ca.DM(math.radians(MIN_ELEV)), ca.DM(SIGMOID_K)))
        for e_deg in elevs])

    vis_frac = np.mean(elevs > MIN_ELEV)
    cov_mean = covs.mean()

    print(f"  ISS-like orbit (a={a_demo:.0f} km, e={e_demo}, i={i_demo} deg, "
          f"RAAN={raan_demo} deg)")
    print(f"  DC ECI: ({r_dc[0]:.1f}, {r_dc[1]:.1f}, {r_dc[2]:.1f}) km, "
          f"|r| = {np.linalg.norm(r_dc):.1f} km")
    print(f"  Max elevation:   {elevs.max():.1f} deg")
    print(f"  Visible fraction (e > {MIN_ELEV} deg): {vis_frac*100:.1f}%")
    print(f"  Mean smooth coverage: {cov_mean:.4f}  ({cov_mean*100:.1f}%)")
    print()

    fig1, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig1.suptitle(
        f"ISS-like orbit over Washington DC  "
        f"(a={a_demo:.0f} km, e={e_demo}, i={i_demo} deg, RAAN={raan_demo} deg)",
        fontsize=11, fontweight="bold",
    )

    ax_e.plot(np.degrees(L_arr), elevs, "royalblue", linewidth=1.5)
    ax_e.axhline(MIN_ELEV, color="tomato", linewidth=1.0, linestyle="--",
                 label=f"{MIN_ELEV} deg mask")
    ax_e.fill_between(np.degrees(L_arr), elevs, MIN_ELEV,
                      where=(elevs > MIN_ELEV), alpha=0.25, color="royalblue",
                      label=f"Visible arc ({vis_frac*100:.0f}%)")
    ax_e.set_ylabel("Elevation [deg]")
    ax_e.set_ylim(-90, 90)
    ax_e.legend(fontsize=9)
    ax_e.grid(True, linewidth=0.4, alpha=0.6)
    ax_e.set_yticks(range(-90, 91, 30))

    ax_c.plot(np.degrees(L_arr), covs, "seagreen", linewidth=1.5)
    ax_c.axhline(0.5, color="gray", linewidth=0.7, linestyle=":", label="c=0.5 (threshold)")
    ax_c.axhline(cov_mean, color="seagreen", linewidth=0.8, linestyle="--", alpha=0.7,
                 label=f"Mean = {cov_mean:.3f}")
    ax_c.set_xlabel("True longitude L [deg]")
    ax_c.set_ylabel("Smooth coverage c(L)")
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.legend(fontsize=9)
    ax_c.grid(True, linewidth=0.4, alpha=0.6)

    fig1.tight_layout()


# ---------------------------------------------------------------------------
# Demo 2 - Coverage optimization with the SingleSatCoverage component
# ---------------------------------------------------------------------------

def demo_optimization():
    print("=" * 60)
    print("Demo 2 - Coverage optimization")
    print("=" * 60)

    start = start_elements()
    cov_start = coverage_mean(*(start[c] for c in "pfghk"), n_pts=24)   # the NLP's own grid
    print(f"  Start: 500 km, f = 0.01, i = {START_INC_DEG} deg, RAAN = {START_RAAN_DEG:.0f} deg"
          f"  ->  coverage {cov_start:.4f}")
    print("  (Demo 1's mean coverage differs: 400 km, e = 0.001 and 360 sample points there;")
    print("  500 km, e = 0.01 and the NLP's 24 points here.)")
    print(f"  Declared scaling: p / {R_EARTH} km, altitude rows / {R_EARTH}^2 km^2; "
          f"IPOPT default tolerances")
    print()

    # Strict: the declared scales and IPOPT's default tolerances.
    problem = coverage_problem()
    result = problem.solve()
    orbit = keplerian_elements(result.x_opt)

    print(f"  Solver status: {result.status}")
    print(f"  Iterations:    {result.iterations}")
    print(f"  Coverage:      {-result.f_opt:.4f}  "
          f"(+{(-result.f_opt - cov_start)*100:.1f} pp vs start)")
    print("  Optimal orbit (display only, via transform.mee_to_koe):")
    print(f"    a    = {orbit['a']:.1f} km")
    print(f"    e    = {orbit['e']:.4f}")
    print(f"    i    = {math.degrees(orbit['i']):.1f} deg")
    print(f"    RAAN = {math.degrees(orbit['raan']):.1f} deg")
    print(f"    perigee / apogee altitude = {orbit['perigee_altitude']:.1f} / "
          f"{orbit['apogee_altitude']:.1f} km  (box 200-1600 km)")
    print(f"    bound multiplier on sat/p: {result.bound_multiplier('sat/p').item():.3e}  "
          f"(p sits on its upper bound)")
    # At p = R_max the apogee row is -R_max^2 e^2, so a row met to IPOPT's scaled tolerance
    # still leaves sqrt(|row|) km of apogee overshoot -- why the apogee prints just above 1600.
    apogee_row = result.constraint("sat/apogee_altitude").value.item()
    print(f"    apogee row = {apogee_row:.3f} km^2 ({apogee_row / R_EARTH**2:.1e} as the solver "
          f"sees it), i.e. {math.sqrt(abs(apogee_row)):.1f} km of apogee overshoot")
    print()

    # April: unit scale and a loose acceptable tolerance. (Unit scale with the strict
    # defaults does not converge at all: IPOPT stops on Invalid_Number_Detected.)
    april = coverage_problem(APRIL_OPTS, unit_scale=True).solve()
    # The same April tolerances on the scaled problem separate scaling from tolerance.
    april_scaled = coverage_problem(APRIL_OPTS).solve()
    shortfall = 1.0 - april.f_opt / result.f_opt
    print("  April recipe (unit scale, acceptable_tol = 1e-2):")
    print(f"    Solver status: {april.status}")
    print(f"    Iterations:    {april.iterations}")
    print(f"    Coverage:      {-april.f_opt:.4f}")
    print()
    print(f"  {'':<26}{'f*':>10}{'iterations':>12}  status")
    for label, res in (("strict, scaled", result), ("scaled, April tolerances", april_scaled),
                       ("April, unit scale", april)):
        print(f"  {label:<26}{res.f_opt:>10.4f}{res.iterations:>12d}  {res.status}")
    print("  Why they differ: scaling, not the tolerance and not physics -- the scaled problem")
    print("  reaches the same optimum under the April tolerances. At unit scale p ~ 7e3 km and")
    print("  the altitude rows ~ R_earth^2 ~ 4e7 km^2 swamp IPOPT's error measures, so")
    print(f"  acceptable_tol = 1e-2 stopped early, {shortfall*100:.0f} % short of the optimum")
    print("  the scaled solves reach (scale, do not loosen tolerances).")
    print()

    # Plot elevation profiles: start vs optimal
    L_fine, elevs_init = elevation_profile(*(start[c] for c in "pfghk"), n_pts=360)
    _, elevs_opt = elevation_profile(*(result[f"{SCOPE}/{c}"].item() for c in "pfghk"),
                                     n_pts=360)
    vis_init = np.mean(elevs_init > MIN_ELEV)
    vis_opt = np.mean(elevs_opt > MIN_ELEV)

    fig2, ax = plt.subplots(figsize=(10, 4))
    fig2.suptitle(
        "Coverage optimization over Washington DC -- elevation profiles",
        fontsize=11, fontweight="bold",
    )
    ax.plot(np.degrees(L_fine), elevs_init, "royalblue", linewidth=1.5,
            label=f"Start: ISS-like (500 km, i={START_INC_DEG:.0f} deg, "
                  f"RAAN={START_RAAN_DEG:.0f} deg)  -->  {vis_init*100:.0f}% visible")
    ax.plot(np.degrees(L_fine), elevs_opt, "seagreen", linewidth=1.8, linestyle="--",
            label=f"Optimal (a={orbit['a']:.0f} km, i={math.degrees(orbit['i']):.0f} deg, "
                  f"RAAN={math.degrees(orbit['raan']):.0f} deg)  -->  {vis_opt*100:.0f}% visible")
    ax.axhline(MIN_ELEV, color="tomato", linewidth=1.0, linestyle=":",
               label=f"{MIN_ELEV} deg mask")
    ax.fill_between(np.degrees(L_fine), elevs_opt, MIN_ELEV,
                    where=(elevs_opt > MIN_ELEV), alpha=0.12, color="seagreen")
    ax.set_xlabel("True longitude L [deg]")
    ax.set_ylabel("Elevation angle [deg]")
    ax.set_ylim(-90, 90)
    ax.set_yticks(range(-90, 91, 30))
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig2.tight_layout()


# ---------------------------------------------------------------------------
# Demo 3 - Coverage vs. inclination (best RAAN at each inclination)
# ---------------------------------------------------------------------------

def demo_inclination_sweep():
    print("=" * 60)
    print("Demo 3 - Coverage vs. inclination at 700 km altitude")
    print("=" * 60)

    a_scan = R_EARTH + 700.0
    N_scan = 36
    # For each inclination, try six RAAN values and keep the best
    raans_to_try = [0, 60, 120, 180, 240, 300]

    inclinations = np.arange(0, 111, 5)   # 0 to 110 deg in 5 deg steps
    cov_best_raan = []

    for inc_deg in inclinations:
        best = 0.0
        for raan_deg in raans_to_try:
            koe = ca.DM([a_scan, 0.001, math.radians(inc_deg),
                         math.radians(raan_deg), 0.0, 0.0])
            mee = koe_to_mee_fd.function(koe)
            p_s, f_s, g_s, h_s, k_s = [float(mee[i]) for i in range(5)]
            cov = coverage_mean(p_s, f_s, g_s, h_s, k_s, n_pts=N_scan)
            if cov > best:
                best = cov
        cov_best_raan.append(best)

    cov_best_raan = np.array(cov_best_raan)
    best_idx = np.argmax(cov_best_raan)
    best_inc = inclinations[best_idx]
    best_cov = cov_best_raan[best_idx]

    print(f"  Altitude: {a_scan - R_EARTH:.0f} km  |  min elevation: {MIN_ELEV} deg")
    print(f"  Best inclination: {best_inc:.0f} deg  ->  "
          f"coverage = {best_cov*100:.1f}%")
    print(f"  (Coverage fraction over DC, snapshot ECI geometry, "
          f"best RAAN among {raans_to_try})")
    print()

    fig3, ax = plt.subplots(figsize=(9, 4))
    fig3.suptitle(
        f"Coverage over DC vs. inclination  "
        f"(circular orbit, alt={a_scan-R_EARTH:.0f} km, best RAAN tested)",
        fontsize=11, fontweight="bold",
    )
    ax.plot(inclinations, cov_best_raan * 100, "royalblue", linewidth=1.8)
    ax.axvline(LAT_DC, color="tomato", linewidth=1.0, linestyle="--",
               label=f"Target latitude ({LAT_DC} deg)")
    ax.axvline(best_inc, color="seagreen", linewidth=1.2, linestyle=":",
               label=f"Best inclination ({best_inc:.0f} deg, {best_cov*100:.1f}%)")
    ax.set_xlabel("Inclination [deg]")
    ax.set_ylabel("Coverage fraction [%]")
    ax.set_xlim(0, 110)
    ax.legend(fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig3.tight_layout()


def main():
    demo_elevation_profile()
    demo_optimization()
    demo_inclination_sweep()
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
