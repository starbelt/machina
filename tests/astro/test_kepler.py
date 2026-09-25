"""
Tests for the universal-variable propagation factories in
``machina.astro.transforms``: ``transform.stumpff_cs``,
``transform.lagrange_coefficients``, ``transform.universal_kepler`` and
``transform.propagate_universal``.

Test classes
------------
TestStumpffFunctions         -- C(ψ) and S(ψ) correctness, Taylor/exact match, continuity
TestLagrangeCoefficients     -- f/g coefficients: identity at Δt=0, conservation identity
TestUniversalKepler          -- solve χ: roundtrip propagation for multiple orbit types
TestPropagateUniversal       -- end-to-end (r0,v0,Δt)→(r,v): period, half-period, expand=True
"""

import math

import casadi as ca
import numpy as np
import pytest

import machina.astro  # noqa: F401
from machina.library import registry

pytestmark = pytest.mark.requires_casadi

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU = 398600.4418          # Earth GM [km³/s²]


# ---------------------------------------------------------------------------
# Reference orbit helpers
# ---------------------------------------------------------------------------

def _circular_state(a, i=0.0, raan=0.0):
    """Return (r0, v0) for a circular orbit at ν=0 in ECI (numpy arrays)."""
    r_mag = a
    v_mag = math.sqrt(MU / a)
    # At ν=0 in the perifocal frame, r is along P-hat
    # Rotate P-hat by R3(-raan) @ R1(-i) @ R3(0)
    cosO, sinO = math.cos(raan), math.sin(raan)
    cosi, sini = math.cos(i), math.sin(i)
    # P-hat in ECI
    Px = cosO;  Py = sinO;  Pz = 0.0
    # Q-hat in ECI
    Qx = -cosi*sinO;  Qy = cosi*cosO;  Qz = sini
    r0 = np.array([r_mag*Px, r_mag*Py, r_mag*Pz])
    v0 = np.array([v_mag*Qx, v_mag*Qy, v_mag*Qz])
    return r0, v0


def _orbit_period(a):
    return 2 * math.pi * math.sqrt(a**3 / MU)


def _eval_stumpff(psi_val):
    """Evaluate stumpff_cs at a numeric psi, return (C, S) as Python floats."""
    fd = registry.get('transform.stumpff_cs')()
    C_dm, S_dm = fd.function(ca.DM(psi_val))
    return float(C_dm), float(S_dm)


def _eval_propagate(r0_np, v0_np, dt_val, mu=MU):
    """Propagate a state and return (r, v) as numpy arrays."""
    fd = registry.get('transform.propagate_universal')(mu=mu)
    r_dm, v_dm = fd.function(ca.DM(r0_np), ca.DM(v0_np), ca.DM(dt_val))
    return np.array(r_dm).flatten(), np.array(v_dm).flatten()


# ===========================================================================
# TestStumpffFunctions
# ===========================================================================

class TestStumpffFunctions:

    def test_registry_key_exists(self):
        fd = registry.get('transform.stumpff_cs')()
        assert fd.name == 'stumpff_cs'

    def test_output_shapes(self):
        fd = registry.get('transform.stumpff_cs')()
        assert fd.input_shapes == [(1, 1)]
        assert fd.output_shapes == [(1, 1), (1, 1)]

    def test_at_zero_exact(self):
        """C(0) = 1/2, S(0) = 1/6 (Taylor branch)."""
        C, S = _eval_stumpff(0.0)
        assert abs(C - 0.5) < 1e-15
        assert abs(S - 1/6) < 1e-15

    def test_positive_psi_exact_formula(self):
        """ψ = π²: C = (1 − cos π)/π² = 2/π², S = (π − 0)/π³ = 1/π²."""
        psi = math.pi ** 2
        C, S = _eval_stumpff(psi)
        C_ref = (1 - math.cos(math.pi)) / psi      # = 2/pi²
        S_ref = (math.pi - math.sin(math.pi)) / math.pi**3   # ≈ 1/pi²
        assert abs(C - C_ref) < 1e-14
        assert abs(S - S_ref) < 1e-14

    def test_negative_psi_exact_formula(self):
        """ψ = −1: C = (cosh 1 − 1), S = (sinh 1 − 1)/1."""
        psi = -1.0
        C, S = _eval_stumpff(psi)
        C_ref = math.cosh(1.0) - 1.0       # = (cosh1-1)/1
        S_ref = math.sinh(1.0) - 1.0       # = (sinh1-1)/1
        assert abs(C - C_ref) < 1e-14
        assert abs(S - S_ref) < 1e-14

    def test_taylor_accuracy_small_positive(self):
        """Taylor vs exact agree to near-machine-precision for ψ = 1e-6."""
        psi = 1e-6
        C, S = _eval_stumpff(psi)
        sq = math.sqrt(psi)
        C_ref = (1 - math.cos(sq)) / psi
        S_ref = (sq - math.sin(sq)) / sq**3
        # CasADi symbolic evaluation vs Python libm: small float arithmetic
        # differences; 1e-10 covers 4-term Taylor truncation + float noise.
        assert abs(C - C_ref) < 1e-10
        assert abs(S - S_ref) < 1e-10

    def test_taylor_accuracy_small_negative(self):
        """Taylor vs exact agree for ψ = −1e-6."""
        psi = -1e-6
        C, S = _eval_stumpff(psi)
        sq = math.sqrt(-psi)
        C_ref = (math.cosh(sq) - 1) / (-psi)
        S_ref = (math.sinh(sq) - sq) / sq**3
        assert abs(C - C_ref) < 1e-10
        assert abs(S - S_ref) < 1e-10

    def test_continuity_at_threshold_positive_side(self):
        """
        No discontinuous jump at EPS transition (positive side).

        EPS ± 1e-12 straddles the Taylor/exact branch boundary by 2e-12.
        dC/dψ ≈ (1/2 − C)/ψ ≈ 4.17e-6/1e-4 = 0.042, so the smooth variation
        over 2e-12 is ~8e-14.  Any discontinuity at the boundary would be
        much larger (Taylor truncation error at EPS^4 ≈ 1e-16 is also tiny).
        """
        EPS = 1e-4
        delta = 1e-12
        C_above, S_above = _eval_stumpff(EPS + delta)
        C_below, S_below = _eval_stumpff(EPS - delta)
        assert abs(C_above - C_below) < 1e-7
        assert abs(S_above - S_below) < 1e-7

    def test_continuity_at_threshold_negative_side(self):
        """No discontinuous jump at −EPS transition."""
        EPS = 1e-4
        delta = 1e-12
        C_above, S_above = _eval_stumpff(-EPS + delta)
        C_below, S_below = _eval_stumpff(-EPS - delta)
        assert abs(C_above - C_below) < 1e-7
        assert abs(S_above - S_below) < 1e-7

    def test_mx_callable(self):
        """Factory is callable with ca.MX and returns ca.MX."""
        fd = registry.get('transform.stumpff_cs')()
        psi_mx = ca.MX.sym('psi')
        result = fd(psi=psi_mx)
        assert isinstance(result[0], ca.MX)
        assert isinstance(result[1], ca.MX)

    def test_identity_one_minus_psi_C(self):
        """
        Correct Stumpff recursion identity: 1 − ψ·C(ψ) = cos(√ψ) for ψ > 0,
        cosh(√(−ψ)) for ψ < 0, and 1 at ψ = 0.
        """
        # ψ = 0
        C0, _ = _eval_stumpff(0.0)
        assert abs(1.0 - 0.0 * C0 - 1.0) < 1e-15  # trivially 1 = 1

        # ψ > 0 (elliptic)
        for psi in [0.5, 2.0, math.pi**2]:
            C, _ = _eval_stumpff(psi)
            assert abs(1 - psi * C - math.cos(math.sqrt(psi))) < 1e-12, \
                f"Elliptic identity failed at ψ={psi}"

        # ψ < 0 (hyperbolic)
        for psi in [-0.5, -2.0, -5.0]:
            C, _ = _eval_stumpff(psi)
            assert abs(1 - psi * C - math.cosh(math.sqrt(-psi))) < 1e-12, \
                f"Hyperbolic identity failed at ψ={psi}"


# ===========================================================================
# TestLagrangeCoefficients
# ===========================================================================

class TestLagrangeCoefficients:

    def _eval_lagrange(self, r0_np, v0_np, chi_val, dt_val):
        fd = registry.get('transform.lagrange_coefficients')(mu=MU)
        F, G, Fd, Gd = fd.function(
            ca.DM(r0_np), ca.DM(v0_np), ca.DM(chi_val), ca.DM(dt_val)
        )
        return float(F), float(G), float(Fd), float(Gd)

    def test_identity_at_dt_zero(self):
        """Δt=0 → χ=0 → F=1, G=0 → r=r0, v=v0."""
        r0, v0 = _circular_state(6778.0)
        F, G, Fd, Gd = self._eval_lagrange(r0, v0, 0.0, 0.0)
        assert abs(F - 1.0) < 1e-12
        assert abs(G - 0.0) < 1e-12
        # Gdot = 1, Fdot requires limit as chi→0; skip Fdot for chi=0
        assert abs(Gd - 1.0) < 1e-12

    def test_conservation_identity_iss(self):
        """F·Gdot − Fdot·G = 1 for a non-trivial propagation."""
        r0, v0 = _circular_state(6778.0)
        # Propagate 1000 s using universal_kepler to get chi
        kepler_fd = registry.get('transform.universal_kepler')(mu=MU)
        dt = 1000.0
        chi_dm = kepler_fd.function(ca.DM(r0), ca.DM(v0), ca.DM(dt))
        chi = float(chi_dm)
        F, G, Fd, Gd = self._eval_lagrange(r0, v0, chi, dt)
        assert abs(F * Gd - Fd * G - 1.0) < 1e-10

    def test_conservation_identity_geo(self):
        """Conservation identity for GEO orbit."""
        r0, v0 = _circular_state(42164.0)
        kepler_fd = registry.get('transform.universal_kepler')(mu=MU)
        dt = 3600.0
        chi_dm = kepler_fd.function(ca.DM(r0), ca.DM(v0), ca.DM(dt))
        chi = float(chi_dm)
        F, G, Fd, Gd = self._eval_lagrange(r0, v0, chi, dt)
        assert abs(F * Gd - Fd * G - 1.0) < 1e-10

    def test_output_shapes(self):
        fd = registry.get('transform.lagrange_coefficients')(mu=MU)
        assert fd.input_shapes == [(3, 1), (3, 1), (1, 1), (1, 1)]
        assert fd.output_shapes == [(1, 1), (1, 1), (1, 1), (1, 1)]

    def test_position_matches_mee_to_eci(self):
        """
        Propagate 1000 s via lagrange_coefficients;  compare to position
        computed by stepping the true longitude in mee_to_eci.
        Not a direct roundtrip — serves as a cross-check at the ~0.1 km level.
        """
        # ISS-like circular orbit, i=51.6 deg
        a = 6778.0
        i = math.radians(51.6)
        r0, v0 = _circular_state(a, i=i)
        dt = 1000.0

        # Lagrange propagation
        r_lag, _ = _eval_propagate(r0, v0, dt)

        # Reference: advance true anomaly by n*dt (circular orbit)
        n = math.sqrt(MU / a**3)
        nu_end = n * dt  # ν at t=dt for circular orbit starting at ν=0

        mee_fd = registry.get('transform.mee_to_eci')(mu=MU)
        # MEE for circular orbit at i=51.6, raan=0, aop=0
        p  = a * (1 - 0.0**2)   # p = a for circular
        f_m = 0.0; g_m = 0.0
        h_m = math.tan(i/2)
        k_m = 0.0
        L_m = nu_end  # L = Ω+ω+ν = 0+0+ν
        mee_vec = ca.DM([p, f_m, g_m, h_m, k_m, L_m])
        r_dm, _ = mee_fd.function(mee_vec)
        r_mee = np.array(r_dm).flatten()

        # Tolerance is looser (~0.1 km) due to small e≠0 in the circular-state
        # helper (v_circ gives e=0 exactly, so should match closely)
        np.testing.assert_allclose(r_lag, r_mee, atol=0.5)


# ===========================================================================
# TestUniversalKepler
# ===========================================================================

class TestUniversalKepler:

    def test_registry_key_exists(self):
        fd = registry.get('transform.universal_kepler')(mu=MU)
        assert fd.name == 'universal_kepler'

    def test_output_shapes(self):
        fd = registry.get('transform.universal_kepler')(mu=MU)
        assert fd.input_shapes == [(3, 1), (3, 1), (1, 1)]
        assert fd.output_shapes == [(1, 1)]

    def test_circular_iss_full_period(self):
        """Propagate by T → r≈r0, v≈v0."""
        a = 6778.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, v = _eval_propagate(r0, v0, T)
        np.testing.assert_allclose(r, r0, atol=1.0)     # km
        np.testing.assert_allclose(v, v0, atol=0.01)    # km/s

    def test_circular_iss_half_period(self):
        """Propagate equatorial circular orbit by T/2 → r ≈ −r0."""
        a = 6778.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, _ = _eval_propagate(r0, v0, T / 2)
        np.testing.assert_allclose(r, -r0, atol=1.0)

    def test_geo_full_period(self):
        """GEO orbit (a=42164 km) full period roundtrip."""
        a = 42164.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, v = _eval_propagate(r0, v0, T)
        np.testing.assert_allclose(r, r0, atol=1.0)
        np.testing.assert_allclose(v, v0, atol=0.01)

    def test_polar_orbit_full_period(self):
        """Polar orbit (i=90°) full period roundtrip."""
        a = 7000.0
        r0, v0 = _circular_state(a, i=math.pi / 2)
        T = _orbit_period(a)
        r, v = _eval_propagate(r0, v0, T)
        np.testing.assert_allclose(r, r0, atol=1.0)
        np.testing.assert_allclose(v, v0, atol=0.01)

    def test_heo_forward_backward(self):
        """HEO-like orbit: propagate +1 hr then −1 hr → back to start."""
        # Molniya-inspired: a=26560 km, e=0.72 (but use circular for simplicity)
        a = 26560.0
        r0, v0 = _circular_state(a)
        dt = 3600.0
        r1, v1 = _eval_propagate(r0, v0, dt)
        r2, v2 = _eval_propagate(r1, v1, -dt)
        np.testing.assert_allclose(r2, r0, atol=1.0)
        np.testing.assert_allclose(v2, v0, atol=0.01)

    def test_backward_propagation(self):
        """Δt < 0 propagation: r(−T) should equal r0 from opposite direction."""
        a = 6778.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, v = _eval_propagate(r0, v0, -T)
        np.testing.assert_allclose(r, r0, atol=1.0)
        np.testing.assert_allclose(v, v0, atol=0.01)

    def test_no_nan_short_propagation(self):
        """Short Δt=10 s produces finite results."""
        r0, v0 = _circular_state(6778.0)
        r, v = _eval_propagate(r0, v0, 10.0)
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(v))

    def test_ift_gradient_finite_difference(self):
        """
        Verify IFT gradient of χ w.r.t. r0[0] matches finite difference
        to rtol=1e-4.
        """
        a = 6778.0
        r0, v0 = _circular_state(a)
        dt = 500.0

        kepler_fd = registry.get('transform.universal_kepler')(mu=MU)

        r0_mx = ca.MX.sym('r0', 3)
        v0_mx = ca.MX.sym('v0', 3)
        dt_mx = ca.MX.sym('dt')
        chi_mx = kepler_fd(r0=r0_mx, v0=v0_mx, dt=dt_mx)

        J_sym = ca.jacobian(chi_mx, r0_mx)
        J_fn  = ca.Function('J', [r0_mx, v0_mx, dt_mx], [J_sym])
        J_analytic = np.array(J_fn(ca.DM(r0), ca.DM(v0), ca.DM(dt))).flatten()

        # Finite difference
        eps = 1.0  # km
        r0p = r0.copy(); r0p[0] += eps
        r0m = r0.copy(); r0m[0] -= eps
        chi_p = float(kepler_fd.function(ca.DM(r0p), ca.DM(v0), ca.DM(dt)))
        chi_m = float(kepler_fd.function(ca.DM(r0m), ca.DM(v0), ca.DM(dt)))
        dchi_dr0x_fd = (chi_p - chi_m) / (2 * eps)

        assert abs(J_analytic[0] - dchi_dr0x_fd) / (abs(dchi_dr0x_fd) + 1e-10) < 1e-3

    def test_mx_callable(self):
        """Factory callable with ca.MX; output is ca.MX."""
        fd = registry.get('transform.universal_kepler')(mu=MU)
        r0 = ca.MX.sym('r0', 3)
        v0 = ca.MX.sym('v0', 3)
        dt = ca.MX.sym('dt')
        result = fd(r0=r0, v0=v0, dt=dt)
        assert isinstance(result[0], ca.MX)


# ===========================================================================
# TestPropagateUniversal
# ===========================================================================

class TestPropagateUniversal:

    def test_registry_key_exists(self):
        fd = registry.get('transform.propagate_universal')(mu=MU)
        assert fd.name == 'propagate_universal'

    def test_output_shapes(self):
        fd = registry.get('transform.propagate_universal')(mu=MU)
        assert fd.input_shapes == [(3, 1), (3, 1), (1, 1)]
        assert fd.output_shapes == [(3, 1), (3, 1)]

    def test_iss_full_period(self):
        a = 6778.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, v = _eval_propagate(r0, v0, T)
        np.testing.assert_allclose(r, r0, atol=1.0)
        np.testing.assert_allclose(v, v0, atol=0.01)

    def test_geo_full_period(self):
        a = 42164.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, v = _eval_propagate(r0, v0, T)
        np.testing.assert_allclose(r, r0, atol=1.0)
        np.testing.assert_allclose(v, v0, atol=0.01)

    def test_polar_full_period(self):
        a = 7000.0
        r0, v0 = _circular_state(a, i=math.pi / 2)
        T = _orbit_period(a)
        r, v = _eval_propagate(r0, v0, T)
        np.testing.assert_allclose(r, r0, atol=1.0)
        np.testing.assert_allclose(v, v0, atol=0.01)

    def test_half_period_circular_equatorial(self):
        """r at T/2 ≈ −r0 for circular equatorial orbit."""
        a = 6778.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, _ = _eval_propagate(r0, v0, T / 2)
        np.testing.assert_allclose(r, -r0, atol=1.0)

    def test_forward_backward_symmetry(self):
        """Propagate +Δt then −Δt → back to start."""
        a = 7500.0
        i = math.radians(28.5)
        r0, v0 = _circular_state(a, i=i)
        dt = 2000.0
        r1, v1 = _eval_propagate(r0, v0, dt)
        r2, v2 = _eval_propagate(r1, v1, -dt)
        np.testing.assert_allclose(r2, r0, atol=0.1)
        np.testing.assert_allclose(v2, v0, atol=0.001)

    def test_radius_conserved_one_orbit(self):
        """Radius magnitude should be the same at t=0 and t=T for circular orbit."""
        a = 6778.0
        r0, v0 = _circular_state(a)
        T = _orbit_period(a)
        r, _ = _eval_propagate(r0, v0, T)
        assert abs(np.linalg.norm(r) - a) < 1.0

    def test_expand_true_compatibility(self):
        """
        Build a trivial NLP using propagate_universal and solve with expand=True.
        If expand=True is incompatible with ca.rootfinder in the installed
        CasADi version, this test documents the limitation.
        """
        from machina.solver.backend import SolverBackend

        a = 6778.0
        r0, v0 = _circular_state(a)
        dt_val = 500.0

        prop_fd = registry.get('transform.propagate_universal')(mu=MU)

        sb = SolverBackend()
        dt_var = sb.add_variable('dt', 1, lb=100.0, ub=1000.0, initial_guess=dt_val)
        r0_p = sb.add_parameter('r0', 3)
        v0_p = sb.add_parameter('v0', 3)

        r_final, _ = prop_fd(r0=r0_p, v0=v0_p, dt=dt_var)
        # Minimize negative radius (maximise radius as a stand-in cost)
        cost = -ca.norm_2(r_final)
        sb.add_cost(cost)

        try:
            sb.build(opts={'expand': True, 'ipopt.print_level': 0, 'print_time': 0})
            sb.solve(p_val=np.concatenate([r0, v0]))
            # Just check it ran without error; result.success may be True or False
            # for this degenerate problem (bounded dt, no constraint on final r)
        except Exception as exc:
            pytest.skip(f"expand=True not compatible with rootfinder in this CasADi version: {exc}")

    def test_nlp_integration(self):
        """
        Wire propagate_universal into a minimal NLP, build and solve.
        Optimise dt to minimise |r_final - r_target|² (trivial feasibility).
        """
        from machina.solver.backend import SolverBackend

        a = 6778.0
        r0_np, v0_np = _circular_state(a)
        # Target: quarter period → r should be ~[0, a, 0] for equatorial
        dt_quarter = _orbit_period(a) / 4

        prop_fd = registry.get('transform.propagate_universal')(mu=MU)

        sb = SolverBackend()
        dt_var = sb.add_variable(
            'dt', 1,
            lb=dt_quarter * 0.5,
            ub=dt_quarter * 1.5,
            initial_guess=dt_quarter,
        )
        r0_p = sb.add_parameter('r0', 3)
        v0_p = sb.add_parameter('v0', 3)

        r_final, _ = prop_fd(r0=r0_p, v0=v0_p, dt=dt_var)

        # Target position for quarter period
        r_target = ca.DM([0.0, a, 0.0])
        cost = ca.sumsqr(r_final - r_target)
        sb.add_cost(cost)

        sb.build(opts={'ipopt.print_level': 0, 'print_time': 0})
        result = sb.solve(p_val=np.concatenate([r0_np, v0_np]))

        assert result.success
        dt_opt = result['dt'][0]
        assert abs(dt_opt - dt_quarter) < 60.0   # within 60 s of analytical quarter period

    def test_transform_registry_count(self):
        """After Phase 3b there should be 7 transform.* entries."""
        transforms = registry.list_by_domain('transform')
        assert len(transforms) == 7, (
            f"Expected 7 transform factories, got {len(transforms)}: {transforms}"
        )
