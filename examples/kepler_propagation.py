"""
examples/kepler_propagation.py
-------------------------------
Demo of the astro pack's universal-variable Kepler propagation factories.

Three demonstrations:
  1. Stumpff functions C(ψ) and S(ψ) — the mathematical engine underneath
     the universal Kepler solver.

  2. Orbit propagation — four orbit classes in 3D:
       • ISS-like  LEO   (a = 6 778 km,  e = 0.001, i = 51.6°)
       • GEO             (a = 42 164 km, e = 0.000, i =  0.0°)
       • Molniya   HEO   (a = 26 560 km, e = 0.740, i = 63.4°)
       • Hyperbolic flyby (a = -7 000 km, e = 2.000, i = 30.0°)
     Each orbit is traced by calling propagate_universal at many time steps,
     then plotted in ECI frame with Earth drawn to scale.

  3. Long-arc accuracy -- propagate ISS by 1x through 20x the orbital period
     and report the position and velocity roundtrip error.  Demonstrates that
     the ca.rootfinder-based solver stays accurate over many orbits.

All functions come from the factory registry (importing machina.astro
registers them).  No solver backend, no Problem — this is pure orbital
mechanics evaluation: each factory's ca.Function called on numbers through
``.function(...)``, the same functions a component composes at SX level.

Run from the project root:
    python examples/kepler_propagation.py
"""

import math

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

import machina.astro  # noqa: F401
from machina.library import registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU      = 398600.4418   # Earth GM [km³/s²]
R_EARTH = 6378.137      # Earth equatorial radius [km], as in the rest of machina

# ---------------------------------------------------------------------------
# Grab four astro factories from the registry
# ---------------------------------------------------------------------------

stumpff_fd  = registry.get('transform.stumpff_cs')()
prop_fd     = registry.get('transform.propagate_universal')(mu=MU)
koe_to_mee  = registry.get('transform.koe_to_mee')()
mee_to_eci  = registry.get('transform.mee_to_eci')(mu=MU)

print("Factories from machina.astro:")
for fd in (stumpff_fd, prop_fd, koe_to_mee, mee_to_eci):
    print(f"  {fd}")
print()


# ---------------------------------------------------------------------------
# Helper: KOE → ECI initial state (the MEE transforms feed the propagator)
# ---------------------------------------------------------------------------

def koe_to_initial_state(a, e, i_deg, raan_deg=0.0, aop_deg=0.0, nu_deg=0.0):
    """Return (r0_np, v0_np) in ECI from classical orbital elements."""
    koe = ca.DM([
        a,
        e,
        math.radians(i_deg),
        math.radians(raan_deg),
        math.radians(aop_deg),
        math.radians(nu_deg),
    ])
    mee   = koe_to_mee.function(koe)
    r, v  = mee_to_eci.function(mee)
    return np.array(r).flatten(), np.array(v).flatten()


def orbit_period(a):
    return 2.0 * math.pi * math.sqrt(a**3 / MU)


# ---------------------------------------------------------------------------
# Demo 1 - Stumpff functions C(ψ) and S(ψ)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Demo 1 - Stumpff functions")
print("=" * 60)

psi_range = np.linspace(-50.0, 60.0, 2000)
C_vals, S_vals = [], []
for psi in psi_range:
    C_dm, S_dm = stumpff_fd.function(ca.DM(float(psi)))
    C_vals.append(float(C_dm))
    S_vals.append(float(S_dm))
C_vals = np.array(C_vals)
S_vals = np.array(S_vals)

# Spot-check key values
print(f"  C(0) = {float(stumpff_fd.function(ca.DM(0.0))[0]):.6f}   (exact: 0.5)")
print(f"  S(0) = {float(stumpff_fd.function(ca.DM(0.0))[1]):.6f}   (exact: 0.166667)")
C_pi2, S_pi2 = [float(x) for x in stumpff_fd.function(ca.DM(math.pi**2))]
print(f"  C(pi^2) = {C_pi2:.6f}   (exact: {(1-math.cos(math.pi))/math.pi**2:.6f})")
print()

fig1, axes = plt.subplots(1, 2, figsize=(11, 4))
fig1.suptitle("Stumpff Functions  C(ψ) and S(ψ)", fontsize=13, fontweight='bold')

for ax, vals, name, color in zip(axes, [C_vals, S_vals], ['C(ψ)', 'S(ψ)'], ['royalblue', 'tomato']):
    # Shade regimes
    ax.axvspan(-50, 0,  alpha=0.07, color='darkorange', label='hyperbolic  (ψ < 0)')
    ax.axvspan(0,  60,  alpha=0.07, color='steelblue',  label='elliptic  (ψ > 0)')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.plot(psi_range, vals, color=color, linewidth=1.8, label=name)
    # Mark ψ = π², 4π², 9π² (where √ψ = π, 2π, 3π — singularities of the
    # exact formula denominator if not for the Stumpff formulation)
    for k in range(1, 4):
        psi_k = (k * math.pi)**2
        if psi_k <= 60:
            ax.axvline(psi_k, color='gray', linewidth=0.6, linestyle=':')
            ax.text(psi_k + 0.5, ax.get_ylim()[1] if k == 1 else 0,
                    f'({k}π)²', fontsize=7, color='gray', va='top')
    ax.set_xlabel('ψ = α χ²')
    ax.set_ylabel(name)
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.set_xlim(-50, 60)
    ax.grid(True, linewidth=0.4, alpha=0.6)

fig1.tight_layout()

# ---------------------------------------------------------------------------
# Demo 2 - Orbit propagation in 3D
# ---------------------------------------------------------------------------

print("=" * 60)
print("Demo 2 - Orbit propagation")
print("=" * 60)

orbits = [
    # (label, a km, e, i deg, raan, aop, nu, color, n_pts, arc_s)
    # arc_s = None  → one full orbital period (elliptic only)
    # arc_s = float → propagate ±arc_s/2 seconds centred on start point (hyperbolic)
    ("ISS  (LEO)",     6778.0,  0.001, 51.6, 0.0,   0.0, 0.0, 'royalblue',  300, None),
    ("GEO",           42164.0,  0.000,  0.0, 0.0,   0.0, 0.0, 'darkorange', 200, None),
    ("Molniya (HEO)", 26560.0,  0.740, 63.4, 0.0, 270.0, 0.0, 'seagreen',   400, None),
    # Hyperbolic flyby: a < 0, e > 1.  rp = a*(1-e) = 7000 km (~622 km alt).
    # Asymptote half-angle = arccos(-1/e) = 120 deg; arc covers ±100 deg (r < 35000 km).
    ("Hyperbolic flyby", -7000.0, 2.0, 30.0, 0.0, 0.0, 0.0, 'orchid', 300, 7200),
]

fig2 = plt.figure(figsize=(10, 8))
ax3d = fig2.add_subplot(111, projection='3d')

# Draw Earth sphere
u_e = np.linspace(0, 2*np.pi, 60)
v_e = np.linspace(0, np.pi, 30)
xe = R_EARTH * np.outer(np.cos(u_e), np.sin(v_e))
ye = R_EARTH * np.outer(np.sin(u_e), np.sin(v_e))
ze = R_EARTH * np.outer(np.ones_like(u_e), np.cos(v_e))
ax3d.plot_surface(xe, ye, ze, color='deepskyblue', alpha=0.25, linewidth=0, zorder=0)

# Equatorial circle outline
theta = np.linspace(0, 2*np.pi, 300)
ax3d.plot(R_EARTH*np.cos(theta), R_EARTH*np.sin(theta), np.zeros_like(theta),
          color='deepskyblue', linewidth=0.7, alpha=0.5)

for label, a, e, i_deg, raan_deg, aop_deg, nu_deg, color, n_pts, arc_s in orbits:
    r0, v0 = koe_to_initial_state(a, e, i_deg, raan_deg, aop_deg, nu_deg)

    if arc_s is None:
        # Elliptic: propagate one full period
        times = np.linspace(0.0, orbit_period(a), n_pts)
    else:
        # Hyperbolic: propagate symmetrically around periapsis
        half = arc_s / 2.0
        times = np.linspace(-half, half, n_pts)

    xs, ys, zs = [], [], []
    for dt in times:
        r_dm, _ = prop_fd.function(ca.DM(r0), ca.DM(v0), ca.DM(float(dt)))
        r = np.array(r_dm).flatten()
        xs.append(r[0]); ys.append(r[1]); zs.append(r[2])

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    lw = 1.6 if e < 1 else 2.0
    ax3d.plot(xs, ys, zs, color=color, linewidth=lw, label=label)
    # Mark periapsis (closest point = midpoint of time array for hyperbolic, t=0 for elliptic)
    mid = n_pts // 2 if arc_s is not None else 0
    ax3d.scatter([xs[mid]], [ys[mid]], [zs[mid]], color=color, s=40, zorder=5)

    # Summary
    rp = a * (1 - e)   # periapsis radius (works for both elliptic and hyperbolic)
    if e < 1:
        T = orbit_period(a)
        print(f"  {label:<24} a={a:>7.0f} km  e={e:.3f}  "
              f"i={i_deg:>5.1f} deg  T={T/60:>7.2f} min   alt_p~{rp-R_EARTH:.0f} km")
    else:
        v_inf = math.sqrt(-MU / a)   # hyperbolic excess speed
        print(f"  {label:<24} a={a:>7.0f} km  e={e:.1f}    "
              f"i={i_deg:>5.1f} deg  v_inf={v_inf:.2f} km/s  alt_p~{rp-R_EARTH:.0f} km")

print()

# Axis labels and formatting
max_range = 50000.0
ax3d.set_xlim(-max_range, max_range); ax3d.set_ylim(-max_range, max_range)
ax3d.set_zlim(-max_range, max_range)
ax3d.set_xlabel('X ECI [km]'); ax3d.set_ylabel('Y ECI [km]'); ax3d.set_zlabel('Z ECI [km]')
ax3d.set_title('Universal Kepler Propagation -- four orbit classes', fontsize=12, fontweight='bold')
ax3d.legend(loc='upper left', fontsize=9)
ax3d.set_box_aspect([1, 1, 1])
ax3d.xaxis.pane.fill = False; ax3d.yaxis.pane.fill = False; ax3d.zaxis.pane.fill = False
fig2.tight_layout()

# ---------------------------------------------------------------------------
# Demo 3 - Long-arc accuracy (multi-period roundtrip error)
# ---------------------------------------------------------------------------

print("=" * 60)
print("Demo 3 - Long-arc propagation accuracy")
print("=" * 60)

a_iss  = 6778.0
T_iss  = orbit_period(a_iss)
r0, v0 = koe_to_initial_state(a_iss, 0.001, 51.6)

n_periods = [0.25, 0.5, 1, 2, 5, 10, 20]
r_errors, v_errors = [], []

print(f"  {'Periods':>8}  {'dt (min)':>10}  {'|dr| (m)':>12}  {'|dv| (mm/s)':>13}")
print("  " + "-" * 50)

for n in n_periods:
    dt_fwd = n * T_iss

    # Forward n periods
    r1, v1 = [np.array(x).flatten()
               for x in prop_fd.function(ca.DM(r0), ca.DM(v0), ca.DM(dt_fwd))]

    # Back n periods
    r2, v2 = [np.array(x).flatten()
               for x in prop_fd.function(ca.DM(r1), ca.DM(v1), ca.DM(-dt_fwd))]

    dr = np.linalg.norm(r2 - r0) * 1e3   # m
    dv = np.linalg.norm(v2 - v0) * 1e6   # mm/s
    r_errors.append(dr); v_errors.append(dv)
    print(f"  {n:>8.2f}  {dt_fwd/60:>10.1f}  {dr:>12.2e}  {dv:>13.2e}")

print()

from matplotlib.ticker import ScalarFormatter

fig3, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(10, 4))
fig3.suptitle("ISS Roundtrip Error vs. Propagation Duration", fontsize=12, fontweight='bold')

periods_arr = np.array(n_periods)

for ax, vals, ylabel, title, color, marker in [
    (ax_r, r_errors, 'Position error |dr| [m]',    'Position roundtrip error', 'royalblue', 'o'),
    (ax_v, v_errors, 'Velocity error |dv| [mm/s]', 'Velocity roundtrip error', 'tomato',    's'),
]:
    ax.loglog(periods_arr, vals, f'{marker}-', color=color, linewidth=1.8, markersize=6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.15, 25)
    ax.set_xticks(periods_arr)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Number of orbital periods')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which='both', linewidth=0.4, alpha=0.6)

fig3.tight_layout()

# ---------------------------------------------------------------------------
# Show all figures (skipped under a non-interactive backend such as MPLBACKEND=Agg)
# ---------------------------------------------------------------------------

if matplotlib.get_backend().lower() != "agg":
    plt.show()
