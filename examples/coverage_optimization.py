"""
examples/coverage_optimization.py
-----------------------------------
Demo of the Phase 3c SingleSatCoverage agent.

Three demonstrations:
  1. Elevation profile over Washington DC:
     Plots the elevation angle of an ISS-like orbit with RAAN chosen so the
     orbit passes over DC (in snapshot ECI geometry, RAAN=240 deg gives
     ~65 deg max elevation).  Shows how the sigmoid coverage indicator
     transitions sharply near the 10 deg elevation mask.

  2. Coverage optimization:
     Maximize the time-average coverage fraction over Washington DC.
     Starting from the same ISS-like orbit (500 km, 51.6 deg, RAAN=240 deg),
     the optimizer is free to adjust: altitude (200-1600 km), eccentricity
     (up to 0.25), and inclination/RAAN (encoded via h, k with bounds giving
     inclinations up to ~123 deg).

  3. Coverage vs. inclination sweep:
     Evaluates the coverage fraction for a fixed circular orbit at 700 km
     as inclination varies from 0 to 110 deg, with RAAN chosen at each step
     to give maximum coverage at that inclination.  Shows the coverage
     landscape and the trade-off between inclination and ground track geometry.

Note on snapshot geometry
--------------------------
The coverage model uses "snapshot" ECI geometry — the ground target is fixed
in ECI (Earth rotation is not modelled).  This is a valid approximation for
instantaneous geometry calculations in the NLP, but the quantitative coverage
fractions are not directly comparable to time-averaged values that account for
Earth rotation.  The optimizer is internally consistent and finds the orbit
that maximizes coverage in this static geometry, which captures the essential
inclination and altitude trade-offs.

Run from the project root:
    python examples/coverage_optimization.py
"""

import math
import warnings

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from machina.agents import SingleSatCoverage
from machina.blocks import registry
from machina.compiler.compiler_stub import CompilerStub

# ---------------------------------------------------------------------------
# Constants and target
# ---------------------------------------------------------------------------

MU      = 398600.4418   # km^3/s^2
R_EARTH = 6378.137      # km

LAT_DC     = 38.9       # deg  (Washington DC)
LON_DC     = -77.0      # deg
MIN_ELEV   = 10.0       # deg  elevation mask

# ---------------------------------------------------------------------------
# Fetch Layer 2 functions
# ---------------------------------------------------------------------------

koe_to_mee_fd = registry.get('transform.koe_to_mee')()
mee_to_eci_fd = registry.get('transform.mee_to_eci')(mu=MU)
gt_eci_fd     = registry.get('geometry.ground_target_eci')(R_earth=R_EARTH)
elev_angle_fd = registry.get('geometry.elevation_angle')()
smooth_cov_fd = registry.get('cost.smooth_coverage')()

# Ground target ECI position (DC, snapshot geometry)
r_dc_dm = gt_eci_fd.function(ca.DM(math.radians(LAT_DC)),
                               ca.DM(math.radians(LON_DC)))
r_dc    = np.array(r_dc_dm).flatten()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def koe_to_mee_vals(a, e, i_deg, raan_deg=0.0, aop_deg=0.0, nu_deg=0.0):
    """Return dict of MEE values from KOE."""
    koe = ca.DM([a, e, math.radians(i_deg),
                 math.radians(raan_deg), math.radians(aop_deg), math.radians(nu_deg)])
    mee = koe_to_mee_fd.function(koe)
    return {c: float(mee[i]) for i, c in enumerate('pfghkL')}


def elevation_profile(p, f, g, h, k, n_pts=360):
    """Evaluate elevation over DC at n_pts equally-spaced true longitudes."""
    L_arr = np.linspace(0, 2 * math.pi, n_pts + 1)[:-1]
    elevs = []
    for L in L_arr:
        mee_i = ca.DM([p, f, g, h, k, L])
        r_sat, _ = mee_to_eci_fd.function(mee_i)
        elev_rad  = float(elev_angle_fd.function(r_sat, r_dc_dm))
        elevs.append(math.degrees(elev_rad))
    return L_arr, np.array(elevs)


def coverage_mean(p, f, g, h, k, n_pts=72, k_sig=20.0, emin_deg=10.0):
    """Evaluate mean smooth coverage over DC."""
    emin_rad = math.radians(emin_deg)
    L_arr = np.linspace(0, 2 * math.pi, n_pts + 1)[:-1]
    total = 0.0
    for L in L_arr:
        mee_i = ca.DM([p, f, g, h, k, L])
        r_sat, _ = mee_to_eci_fd.function(mee_i)
        elev = elev_angle_fd.function(r_sat, r_dc_dm)
        cov  = smooth_cov_fd.function(elev, ca.DM(emin_rad), ca.DM(k_sig))
        total += float(cov)
    return total / n_pts


# ---------------------------------------------------------------------------
# Demo 1 - Elevation and coverage profile over Washington DC
# ---------------------------------------------------------------------------

print("=" * 60)
print("Demo 1 - Elevation profile over Washington DC")
print("=" * 60)

# ISS-like orbit, RAAN=240 deg chosen so orbit passes over DC
# (in ECI snapshot geometry, RAAN=240 deg gives max elevation ~65 deg for DC)
a_demo = 6778.0     # km (ISS semi-major axis)
e_demo = 0.001
i_demo = 51.6       # deg
raan_demo = 240.0   # deg — this RAAN puts DC in the orbit's ground track

mee_demo = koe_to_mee_vals(a_demo, e_demo, i_demo, raan_demo)
p_d, f_d, g_d, h_d, k_d = (mee_demo['p'], mee_demo['f'], mee_demo['g'],
                              mee_demo['h'], mee_demo['k'])

L_arr, elevs = elevation_profile(p_d, f_d, g_d, h_d, k_d, n_pts=360)
covs = np.array([float(smooth_cov_fd.function(
    ca.DM(math.radians(e_deg)), ca.DM(math.radians(MIN_ELEV)), ca.DM(20.0)))
    for e_deg in elevs])

vis_frac = np.mean(elevs > MIN_ELEV)
cov_mean = covs.mean()

print(f"  ISS-like orbit (a={a_demo:.0f} km, e={e_demo}, i={i_demo} deg, RAAN={raan_demo} deg)")
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
    fontsize=11, fontweight='bold',
)

ax_e.plot(np.degrees(L_arr), elevs, 'royalblue', linewidth=1.5)
ax_e.axhline(MIN_ELEV, color='tomato', linewidth=1.0, linestyle='--',
             label=f'{MIN_ELEV} deg mask')
ax_e.fill_between(np.degrees(L_arr), elevs, MIN_ELEV,
                  where=(elevs > MIN_ELEV), alpha=0.25, color='royalblue',
                  label=f'Visible arc ({vis_frac*100:.0f}%)')
ax_e.set_ylabel('Elevation [deg]')
ax_e.set_ylim(-90, 90)
ax_e.legend(fontsize=9)
ax_e.grid(True, linewidth=0.4, alpha=0.6)
ax_e.set_yticks(range(-90, 91, 30))

ax_c.plot(np.degrees(L_arr), covs, 'seagreen', linewidth=1.5)
ax_c.axhline(0.5, color='gray', linewidth=0.7, linestyle=':', label='c=0.5 (threshold)')
ax_c.axhline(cov_mean, color='seagreen', linewidth=0.8, linestyle='--', alpha=0.7,
             label=f'Mean = {cov_mean:.3f}')
ax_c.set_xlabel('True longitude L [deg]')
ax_c.set_ylabel('Smooth coverage c(L)')
ax_c.set_ylim(-0.05, 1.05)
ax_c.legend(fontsize=9)
ax_c.grid(True, linewidth=0.4, alpha=0.6)

fig1.tight_layout()


# ---------------------------------------------------------------------------
# Demo 2 - Coverage optimization with SingleSatCoverage agent
# ---------------------------------------------------------------------------

print("=" * 60)
print("Demo 2 - Coverage optimization")
print("=" * 60)

N_OPT = 24    # sample points (every 15 deg true longitude)

config = {
    'n_sample_points': N_OPT,
    'ground_target': {'lat_deg': LAT_DC, 'lon_deg': LON_DC},
    'coverage': {'min_elevation_deg': MIN_ELEV, 'sigmoid_k': 20.0},
    'altitude_bounds': {'perigee_min_km': 200.0, 'apogee_max_km': 1600.0},
    'constants': {'mu': MU, 'R_earth': R_EARTH},
}

agent = SingleSatCoverage('sat', config)
compiler = CompilerStub(solver_opts={
    'ipopt.print_level': 0,
    'ipopt.sb': 'yes',
    'ipopt.tol': 1e-4,
    'ipopt.acceptable_tol': 1e-2,
    'ipopt.acceptable_iter': 3,
})
compiler.add_agent(agent)

# Initial guess: RAAN=240, i=51.6 deg, 500 km circular
# h = tan(51.6/2)*cos(240) = 0.484*(-0.5) = -0.242
# k = tan(51.6/2)*sin(240) = 0.484*(-0.866) = -0.419
overrides = {
    'sat/orbital/p': {'value': R_EARTH + 500.0,
                       'lb': R_EARTH + 200.0, 'ub': R_EARTH + 1600.0},
    'sat/orbital/f': {'value': 0.01, 'lb': -0.30, 'ub': 0.30},
    'sat/orbital/g': {'value': 0.0,  'lb': -0.30, 'ub': 0.30},
    'sat/orbital/h': {'value': h_d,  'lb': -1.5,  'ub': 1.5},
    'sat/orbital/k': {'value': k_d,  'lb': -1.5,  'ub': 1.5},
}

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    compiler.compile(overrides=overrides)

cov_sd = compiler.resolve('sat', 'coverage/total')
compiler.add_cost(-cov_sd.symbol, name='neg_coverage')
compiler.build_solver()

print(f"  Starting from: a={a_demo:.0f} km, i={i_demo} deg, RAAN={raan_demo} deg")
print(f"  Initial coverage: {cov_mean*100:.1f}%")
print()
print("  Running IPOPT...")
result = compiler.solve()
print(f"  Solver status: {result.stats.get('return_status')}")
print(f"  Iterations:    {result.stats.get('iter_count')}")
print()

if result.success:
    p_opt = float(result['sat/orbital/p'][0])
    f_opt = float(result['sat/orbital/f'][0])
    g_opt = float(result['sat/orbital/g'][0])
    h_opt = float(result['sat/orbital/h'][0])
    k_opt = float(result['sat/orbital/k'][0])

    e_opt   = math.sqrt(f_opt**2 + g_opt**2)
    sma_opt = p_opt / (1 - e_opt**2) if e_opt < 1.0 else p_opt
    inc_opt = math.degrees(2 * math.atan(math.sqrt(h_opt**2 + k_opt**2)))
    # RAAN from h, k:  h = tan(i/2)*cos(RAAN) → RAAN = atan2(k, h)
    raan_opt = math.degrees(math.atan2(k_opt, h_opt)) % 360

    cov_opt = coverage_mean(p_opt, f_opt, g_opt, h_opt, k_opt,
                             n_pts=360, k_sig=20.0, emin_deg=MIN_ELEV)

    print(f"  Optimal orbit:")
    print(f"    a    = {sma_opt:.1f} km  (alt ~ {sma_opt - R_EARTH:.0f} km)")
    print(f"    e    = {e_opt:.4f}")
    print(f"    i    = {inc_opt:.1f} deg")
    print(f"    RAAN = {raan_opt:.1f} deg")
    print(f"    Coverage: {cov_opt*100:.1f}%  "
          f"(+{(cov_opt - cov_mean)*100:.1f} pp vs initial)")
    print()

    # Plot elevation profiles: initial vs optimal
    L_fine = np.linspace(0, 2 * math.pi, 360)
    _, elevs_init = elevation_profile(p_d, f_d, g_d, h_d, k_d, n_pts=360)
    _, elevs_opt  = elevation_profile(p_opt, f_opt, g_opt, h_opt, k_opt, n_pts=360)

    vis_init = np.mean(elevs_init > MIN_ELEV)
    vis_opt  = np.mean(elevs_opt  > MIN_ELEV)

    fig2, ax = plt.subplots(figsize=(10, 4))
    fig2.suptitle(
        f"Coverage optimization over Washington DC -- elevation profiles",
        fontsize=11, fontweight='bold',
    )
    ax.plot(np.degrees(L_fine), elevs_init, 'royalblue', linewidth=1.5,
            label=f'Initial: ISS-like (a={a_demo:.0f} km, i={i_demo:.0f} deg, '
                  f'RAAN={raan_demo:.0f} deg)  -->  {vis_init*100:.0f}% visible')
    ax.plot(np.degrees(L_fine), elevs_opt, 'seagreen', linewidth=1.8, linestyle='--',
            label=f'Optimal (a={sma_opt:.0f} km, i={inc_opt:.0f} deg, '
                  f'RAAN={raan_opt:.0f} deg)  -->  {vis_opt*100:.0f}% visible')
    ax.axhline(MIN_ELEV, color='tomato', linewidth=1.0, linestyle=':',
               label=f'{MIN_ELEV} deg mask')
    ax.fill_between(np.degrees(L_fine), elevs_opt, MIN_ELEV,
                    where=(elevs_opt > MIN_ELEV), alpha=0.12, color='seagreen')
    ax.set_xlabel('True longitude L [deg]')
    ax.set_ylabel('Elevation angle [deg]')
    ax.set_ylim(-90, 90)
    ax.set_yticks(range(-90, 91, 30))
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig2.tight_layout()
else:
    print(f"  Note: solver returned {result.stats.get('return_status')}. "
          f"Coverage at solver limit: {-result.f_opt*100:.1f}%")


# ---------------------------------------------------------------------------
# Demo 3 - Coverage vs. inclination (best RAAN at each inclination)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Demo 3 - Coverage vs. inclination at 700 km altitude")
print("=" * 60)

a_scan  = R_EARTH + 700.0
N_scan  = 36
# For each inclination, try 4 RAAN values and pick the best
raans_to_try = [0, 60, 120, 180, 240, 300]

inclinations  = np.arange(0, 111, 5)   # 0 to 110 deg in 5 deg steps
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
    fontsize=11, fontweight='bold',
)
ax.plot(inclinations, cov_best_raan * 100, 'royalblue', linewidth=1.8)
ax.axvline(LAT_DC, color='tomato', linewidth=1.0, linestyle='--',
           label=f'Target latitude ({LAT_DC} deg)')
ax.axvline(best_inc, color='seagreen', linewidth=1.2, linestyle=':',
           label=f'Best inclination ({best_inc:.0f} deg, {best_cov*100:.1f}%)')
ax.set_xlabel('Inclination [deg]')
ax.set_ylabel('Coverage fraction [%]')
ax.set_xlim(0, 110)
ax.legend(fontsize=9)
ax.grid(True, linewidth=0.4, alpha=0.6)
fig3.tight_layout()

# ---------------------------------------------------------------------------
# Show all figures
# ---------------------------------------------------------------------------

plt.show()
