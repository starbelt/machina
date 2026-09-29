"""
examples/formation_placement.py
--------------------------------
Placement of a small spacecraft beside a geostationary chief, with the astro
pack's formation and beam factories.

Three results:
  1. Co-located formation. A period-matched (delta a = 0) formation with
     parallel relative eccentricity and inclination vectors, |de| = |di|,
     a |de| = 5 km. Its minimum radial/normal separation is
     geometry.rn_min_separation; the relative orbit comes from
     transform.roe_to_rtn.

  2. Beam duty. The fraction of the day the member sits inside the chief's
     downlink cone (half-angle 1.2 deg), cost.mean_beam_duty. The chief's RTN
     frame is Earth-fixed for a geostationary chief, so a boresight on a ground
     station is a constant RTN vector: -R (nadir) tilted a few degrees. Numbers
     are illustrative; the answer depends on the tilt direction and on the
     formation's shape.

  3. Ring. A ring of spacecraft on a radius dr above the chief passes it once
     per synodic period per member (util.synodic_period, ring_size_for_interval).

No solver: each factory's ca.Function is evaluated on numbers through
``.function(...)``, the same functions a component composes at SX level.

Run from the project root:
    python examples/formation_placement.py
"""

import math

import casadi as ca
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import machina.astro  # noqa: F401
from machina.astro.relative import ring_pass_interval, ring_size_for_interval
from machina.library import registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU = 398600.4418                   # Earth GM [km^3/s^2]
A = 42164.0                        # geostationary radius [km]
SEPARATION_KM = 5.0                # a |de| = a |di| [km]
PHASE = math.radians(30.0)         # common phase of the e and i vectors
HALF_ANGLE = math.radians(1.2)     # downlink cone half-angle [rad]
TILT = math.radians(3.0)           # boresight tilt off nadir [rad]
K_CONE = 1e4                       # in-cone sigmoid steepness [1/rad]
N_SAMPLES = 3600                   # longitude samples per orbit (24 s each at GEO)

roe_to_rtn = registry.get('transform.roe_to_rtn')(a=A)
rn_min = registry.get('geometry.rn_min_separation')(a=A)
beam_duty = registry.get('cost.mean_beam_duty')(a=A, n_samples=N_SAMPLES)
synodic = registry.get('util.synodic_period')(mu=MU)

print("Factories from machina.astro:")
for fd in (roe_to_rtn, rn_min, beam_duty, synodic):
    print(f"  {fd}")
print()

# ---------------------------------------------------------------------------
# 1. Co-located e/i formation
# ---------------------------------------------------------------------------

delta = SEPARATION_KM / A
de = delta * np.array([math.cos(PHASE), math.sin(PHASE)])
di = delta * np.array([math.cos(PHASE), math.sin(PHASE)])
roe = np.r_[0.0, de, di]           # [dlam, df, dg, dix, diy]

d_min = float(rn_min.function(ca.DM(de), ca.DM(di)))

lam = np.linspace(0.0, 2.0 * math.pi, 721)
rtn = np.array(roe_to_rtn.function.map(lam.size)(ca.DM(roe), ca.DM(lam).T))
radius = np.linalg.norm(rtn, axis=0)

print("1. Co-located e/i formation (delta a = 0, parallel e/i vectors)")
print(f"   a |de| = a |di| = {SEPARATION_KM:.1f} km, phase {math.degrees(PHASE):.0f} deg")
print(f"   min R-N separation   d_min = {d_min:.3f} km")
print(f"   relative orbit: R in [{rtn[0].min():+.2f}, {rtn[0].max():+.2f}] km, "
      f"T in [{rtn[1].min():+.2f}, {rtn[1].max():+.2f}] km, "
      f"N in [{rtn[2].min():+.2f}, {rtn[2].max():+.2f}] km")
print(f"   range to chief: {radius.min():.2f} .. {radius.max():.2f} km")
print()

# ---------------------------------------------------------------------------
# 2. Time-mean duty inside the chief's downlink cone
# ---------------------------------------------------------------------------


def boresight(tilt: float, azimuth: float) -> np.ndarray:
    """-R tilted by ``tilt`` toward the direction ``azimuth`` in the T-N plane
    (0 = +T east, pi/2 = +N north)."""
    return np.array([-math.cos(tilt),
                     math.sin(tilt) * math.cos(azimuth),
                     math.sin(tilt) * math.sin(azimuth)])


def duty(b: np.ndarray) -> float:
    return float(beam_duty.function(ca.DM(roe), ca.DM(b), HALF_ANGLE, K_CONE))


# With |de| = |di| parallel the relative orbit is a plane ellipse through the chief
# containing R and the direction (T, N) = (2, 1), so the member's direction from the
# chief sweeps one great circle: a tilted boresight either lies on it or misses it.
in_plane = math.atan2(1.0, 2.0)
azimuths = np.radians(np.arange(0.0, 360.0, 5.0))
tilted = [duty(boresight(TILT, az)) for az in azimuths]
rows = (
    ("boresight -R (nadir)", duty(boresight(0.0, 0.0))),
    (f"-R tilted {math.degrees(TILT):.0f} deg toward the relative-orbit plane "
     f"(azimuth {math.degrees(in_plane):.1f} deg)", duty(boresight(TILT, in_plane))),
    (f"-R tilted {math.degrees(TILT):.0f} deg toward +N", duty(boresight(TILT, math.pi / 2))),
    (f"-R tilted {math.degrees(TILT):.0f} deg, mean over {azimuths.size} azimuths",
     float(np.mean(tilted))),
)

print(f"2. Duty inside a {math.degrees(HALF_ANGLE):.1f} deg half-angle cone "
      f"(k = {K_CONE:g} 1/rad, {N_SAMPLES} samples)")
for label, value in rows:
    print(f"   {label + ':':<68} {100.0 * value:6.3f} %")
print()

# ---------------------------------------------------------------------------
# 3. Ring of spacecraft on a neighbouring radius
# ---------------------------------------------------------------------------

PASS_TARGET = 600.0   # one pass every 10 minutes [s]

print(f"3. Ring on a radius a + dr: synodic period and the ring size for one pass "
      f"every {PASS_TARGET / 60.0:.0f} min")
print(f"   {'dr [km]':>8} {'T_syn [days]':>13} {'ring size':>10} {'pass interval [min]':>20}")
for dr in (10.0, 100.0, 1000.0):
    T_syn = float(synodic.function(A, A + dr))
    n_ring = ring_size_for_interval(T_syn, PASS_TARGET)
    print(f"   {dr:>8.0f} {T_syn / 86400.0:>13.1f} {n_ring:>10d} "
          f"{ring_pass_interval(T_syn, n_ring) / 60.0:>20.2f}")
print()

# ---------------------------------------------------------------------------
# Plot: the relative orbit in the R-N plane (the e/i ellipse) and R-T plane
# ---------------------------------------------------------------------------

fig, (ax_rn, ax_rt) = plt.subplots(1, 2, figsize=(10, 4.5))

ax_rn.plot(rtn[2], rtn[0], color='tab:blue')
ax_rn.add_patch(plt.Circle((0.0, 0.0), d_min, fill=False, linestyle='--', color='tab:red',
                           label=f'd_min = {d_min:.2f} km'))
ax_rn.plot(0.0, 0.0, 'k+', markersize=10, label='chief')
ax_rn.set_xlabel('N (orbit normal) [km]')
ax_rn.set_ylabel('R (radial) [km]')
ax_rn.set_title('R-N plane: e/i separation')
ax_rn.set_aspect('equal')
ax_rn.legend(loc='upper right', fontsize=8)
ax_rn.grid(True, alpha=0.3)

ax_rt.plot(rtn[1], rtn[0], color='tab:blue')
ax_rt.plot(0.0, 0.0, 'k+', markersize=10, label='chief')
ax_rt.set_xlabel('T (along-track) [km]')
ax_rt.set_ylabel('R (radial) [km]')
ax_rt.set_title('R-T plane: 2:1 relative ellipse')
ax_rt.set_aspect('equal')
ax_rt.legend(loc='upper right', fontsize=8)
ax_rt.grid(True, alpha=0.3)

fig.suptitle('Period-matched formation beside a geostationary chief')
plt.tight_layout()
if matplotlib.get_backend().lower() != "agg":
    plt.show()
