"""
Phase 3a tests — coordinate transforms and their registration.

Sections
--------
TestCoordinateTransforms   -- KOE↔MEE↔ECI, roundtrip and against reference
TestTransformRegistry      -- registry registration and FunctionDescriptor metadata

Reference implementation
------------------------
_koe_to_eci_ref() in this file provides an independent numpy-based PQW→ECI
computation used to validate mee_to_eci without relying on the same formula.
"""

import casadi as ca
import numpy as np
import pytest

import machina.astro  # noqa: F401
from machina.library import registry

pytestmark = pytest.mark.requires_casadi

MU_EARTH = 398600.4418  # km³/s²


# ===========================================================================
# Helpers
# ===========================================================================

def _koe_to_eci_ref(a, e, inc, raan, aop, nu):
    """
    Reference ECI position via perifocal-frame rotation (pure numpy).

    This is the independent implementation used to validate mee_to_eci.
    It does NOT use the MEE formulas; it uses the classical rotation
    R3(-raan) @ R1(-inc) @ R3(-aop).

    Reference: Curtis, Orbital Mechanics for Engineering Students (2019), Ch. 4.
    """
    p = a * (1 - e ** 2)
    r_mag = p / (1 + e * np.cos(nu))
    r_pqw = r_mag * np.array([np.cos(nu), np.sin(nu), 0.0])

    def R1(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def R3(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    # PQW→ECI: columns are [P̂, Q̂, Ŵ] expressed in ECI.
    # Equivalent to R3(raan) @ R1(inc) @ R3(aop) with the active-rotation
    # convention R3(θ) = [[cosθ,-sinθ,0],[sinθ,cosθ,0],[0,0,1]].
    Q = R3(raan) @ R1(inc) @ R3(aop)
    return Q @ r_pqw


def _call_koe_to_mee(koe_np):
    factory = registry.get('transform.koe_to_mee')()
    koe_dm = ca.DM(koe_np)
    result = factory.function(koe_dm)
    return np.array(result).flatten()


def _call_mee_to_koe(mee_np):
    factory = registry.get('transform.mee_to_koe')()
    mee_dm = ca.DM(mee_np)
    result = factory.function(mee_dm)
    return np.array(result).flatten()


def _call_mee_to_eci(mee_np, mu=MU_EARTH):
    factory = registry.get('transform.mee_to_eci')(mu=mu)
    mee_dm = ca.DM(mee_np)
    r, v = factory.function(mee_dm)
    return np.array(r).flatten(), np.array(v).flatten()


# ===========================================================================
# Reference orbit fixtures
# ===========================================================================

# koe = [a (km), e, i (rad), Ω (rad), ω (rad), ν (rad)]

KOE_ISS = [6778.0, 0.001, np.radians(51.6), np.radians(23.0),
           np.radians(45.0), np.radians(120.0)]

KOE_NEAR_CIRCULAR_EQUATORIAL = [7000.0, 0.001, np.radians(0.5),
                                  np.radians(10.0), np.radians(20.0),
                                  np.radians(90.0)]

KOE_POLAR = [7000.0, 0.02, np.radians(90.0), np.radians(60.0),
             np.radians(30.0), np.radians(200.0)]

KOE_HEO = [26560.0, 0.72, np.radians(63.4), np.radians(150.0),
           np.radians(270.0), np.radians(10.0)]

# Circular equatorial at ν=0: r_eci should be exactly [a, 0, 0].
KOE_CIRC_EQ_NU0 = [7000.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Circular equatorial at ν=π/2: r_eci should be exactly [0, a, 0].
KOE_CIRC_EQ_NU90 = [7000.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]


# ===========================================================================
# Section 1 — Coordinate Transform Tests
# ===========================================================================

class TestCoordinateTransforms:
    """KOE ↔ MEE roundtrip and MEE → ECI against independent reference."""

    @pytest.mark.parametrize('koe', [
        KOE_ISS,
        KOE_NEAR_CIRCULAR_EQUATORIAL,
        KOE_POLAR,
        KOE_HEO,
    ], ids=['iss', 'near_circular_equatorial', 'polar', 'heo'])
    def test_koe_mee_koe_roundtrip(self, koe):
        """KOE → MEE → KOE recovers original elements within floating-point.

        Angles are compared modulo 2π so that equivalent representations
        (e.g. ω=270° vs ω=-90°) pass.  atan2-based inversion can return
        angles in (-π, π] rather than [0, 2π), which is mathematically
        identical.
        """
        mee = _call_koe_to_mee(koe)
        koe_back = _call_mee_to_koe(mee)

        # Compare scalar quantities (a, e) directly.
        np.testing.assert_allclose(koe_back[:2], koe[:2], atol=1e-8,
                                   err_msg=f"Scalar elements failed for KOE={koe}")

        # Compare angular quantities modulo 2π via unit-vector comparison.
        for idx in range(2, 6):
            expected = float(koe[idx])
            actual = float(koe_back[idx])
            # cos/sin comparison handles wrapping cleanly (e.g. 270° vs -90°).
            np.testing.assert_allclose(np.cos(actual), np.cos(expected), atol=1e-8,
                                       err_msg=f"angle[{idx}] cos mismatch: {np.degrees(actual):.3f}° vs {np.degrees(expected):.3f}°")
            np.testing.assert_allclose(np.sin(actual), np.sin(expected), atol=1e-8,
                                       err_msg=f"angle[{idx}] sin mismatch: {np.degrees(actual):.3f}° vs {np.degrees(expected):.3f}°")

    def test_koe_to_mee_iss_spot_check(self):
        """Manually verify one MEE element for the ISS orbit."""
        a, e, inc, raan, aop, nu = KOE_ISS
        mee = _call_koe_to_mee(KOE_ISS)
        p_expected = a * (1 - e ** 2)
        np.testing.assert_allclose(mee[0], p_expected, rtol=1e-10,
                                   err_msg="p = a*(1-e²) failed")

    def test_mee_to_eci_circular_equatorial_nu0(self):
        """Circular equatorial at ν=0: r_eci = [a, 0, 0] exactly."""
        a = KOE_CIRC_EQ_NU0[0]
        mee = _call_koe_to_mee(KOE_CIRC_EQ_NU0)
        r_eci, _ = _call_mee_to_eci(mee)
        np.testing.assert_allclose(r_eci, [a, 0.0, 0.0], atol=1e-6,
                                   err_msg="r_eci ≠ [a, 0, 0] at ν=0")

    def test_mee_to_eci_circular_equatorial_nu90(self):
        """Circular equatorial at ν=π/2: r_eci = [0, a, 0] exactly."""
        a = KOE_CIRC_EQ_NU90[0]
        mee = _call_koe_to_mee(KOE_CIRC_EQ_NU90)
        r_eci, _ = _call_mee_to_eci(mee)
        np.testing.assert_allclose(r_eci, [0.0, a, 0.0], atol=1e-6,
                                   err_msg="r_eci ≠ [0, a, 0] at ν=π/2")

    @pytest.mark.parametrize('koe', [
        KOE_ISS,
        KOE_NEAR_CIRCULAR_EQUATORIAL,
        KOE_POLAR,
    ], ids=['iss', 'near_circular_equatorial', 'polar'])
    def test_mee_to_eci_position_vs_reference(self, koe):
        """MEE→ECI position matches independent PQW reference (atol=0.01 km)."""
        a, e, inc, raan, aop, nu = koe
        mee = _call_koe_to_mee(koe)
        r_eci_mee, _ = _call_mee_to_eci(mee)
        r_eci_ref = _koe_to_eci_ref(a, e, inc, raan, aop, nu)
        np.testing.assert_allclose(
            r_eci_mee, r_eci_ref, atol=0.01,
            err_msg=f"ECI position mismatch for KOE={koe}",
        )

    def test_mee_to_eci_radius_magnitude(self):
        """Radius magnitude |r_eci| matches r = a*(1-e²)/(1+e*cosν)."""
        a, e, inc, raan, aop, nu = KOE_ISS
        mee = _call_koe_to_mee(KOE_ISS)
        r_eci, _ = _call_mee_to_eci(mee)
        r_expected = a * (1 - e ** 2) / (1 + e * np.cos(nu))
        np.testing.assert_allclose(
            np.linalg.norm(r_eci), r_expected, rtol=1e-8,
        )

    def test_mee_to_eci_velocity_magnitude_vis_viva(self):
        """Velocity magnitude |v_eci| matches vis-viva: v² = μ*(2/r - 1/a)."""
        a, e, inc, raan, aop, nu = KOE_ISS
        mee = _call_koe_to_mee(KOE_ISS)
        r_eci, v_eci = _call_mee_to_eci(mee)
        r = np.linalg.norm(r_eci)
        v_expected = np.sqrt(MU_EARTH * (2 / r - 1 / a))
        np.testing.assert_allclose(
            np.linalg.norm(v_eci), v_expected, rtol=1e-6,
        )

    def test_mee_to_eci_r_v_orthogonal_circular(self):
        """For a circular orbit, r and v are orthogonal (r·v = 0)."""
        mee = _call_koe_to_mee(KOE_CIRC_EQ_NU0)
        r_eci, v_eci = _call_mee_to_eci(mee)
        np.testing.assert_allclose(np.dot(r_eci, v_eci), 0.0, atol=1e-6)

    def test_singularity_near_retrograde_equatorial(self):
        """i = π - 1e-4 rad: output should be finite (not NaN)."""
        koe_near_retro = [7000.0, 0.01, np.pi - 1e-4,
                          np.radians(45.0), np.radians(30.0), np.radians(60.0)]
        # NOTE: this is near the MEE singularity at i=π (h,k→∞).
        # The transform still runs without raising, but h/k are large.
        mee = _call_koe_to_mee(koe_near_retro)
        assert np.all(np.isfinite(mee)), (
            "koe_to_mee produced non-finite values near i=π singularity"
        )

    def test_koe_to_mee_produces_casadi_mx_when_called_with_mx(self):
        """koe_to_mee called with MX input returns MX output."""
        factory = registry.get('transform.koe_to_mee')()
        koe_mx = ca.MX.sym('koe', 6)
        result = factory(koe=koe_mx)
        assert isinstance(result, ca.MX)


# ===========================================================================
# Section 2 — Transform Registry Tests
# ===========================================================================

class TestTransformRegistry:
    """Registry registration and FunctionDescriptor metadata for transforms."""

    def test_koe_to_mee_is_registered(self):
        factory = registry.get('transform.koe_to_mee')
        assert callable(factory)

    def test_mee_to_koe_is_registered(self):
        factory = registry.get('transform.mee_to_koe')
        assert callable(factory)

    def test_mee_to_eci_is_registered(self):
        factory = registry.get('transform.mee_to_eci')
        assert callable(factory)

    def test_transform_domain_lists_phase3a_transforms(self):
        # Phase 3b adds 4 more transform.* entries; check the 3 from Phase 3a
        # are present rather than asserting the exact total count.
        transforms = registry.list_by_domain('transform')
        assert len(transforms) >= 3
        assert {'transform.koe_to_mee', 'transform.mee_to_koe', 'transform.mee_to_eci'} \
               .issubset(set(transforms))

    def test_koe_to_mee_returns_function_descriptor(self):
        from machina.model import FunctionDescriptor
        fd = registry.get('transform.koe_to_mee')()
        assert isinstance(fd, FunctionDescriptor)

    def test_mee_to_koe_returns_function_descriptor(self):
        from machina.model import FunctionDescriptor
        fd = registry.get('transform.mee_to_koe')()
        assert isinstance(fd, FunctionDescriptor)

    def test_mee_to_eci_returns_function_descriptor(self):
        from machina.model import FunctionDescriptor
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert isinstance(fd, FunctionDescriptor)

    def test_koe_to_mee_input_shape(self):
        fd = registry.get('transform.koe_to_mee')()
        assert fd.input_shapes == [(6, 1)]

    def test_koe_to_mee_output_shape(self):
        fd = registry.get('transform.koe_to_mee')()
        assert fd.output_shapes == [(6, 1)]

    def test_mee_to_eci_input_shape(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert fd.input_shapes == [(6, 1)]

    def test_mee_to_eci_output_shapes_two_3x1(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert fd.output_shapes == [(3, 1), (3, 1)]

    def test_mee_to_eci_output_names(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert fd.output_names == ['r_eci', 'v_eci']

    def test_koe_to_mee_callable_with_mx_returns_mx(self):
        fd = registry.get('transform.koe_to_mee')()
        koe_mx = ca.MX.sym('koe', 6)
        result = fd(koe=koe_mx)
        assert isinstance(result, ca.MX)

    def test_mee_to_eci_callable_with_mx_returns_list(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        mee_mx = ca.MX.sym('mee', 6)
        result = fd(mee=mee_mx)
        # Multi-output CasADi function returns a list-like object
        assert len(result) == 2

    def test_koe_to_mee_description_nonempty(self):
        fd = registry.get('transform.koe_to_mee')()
        assert len(fd.description) > 0

    def test_mee_to_eci_description_contains_mu(self):
        fd = registry.get('transform.mee_to_eci')(mu=MU_EARTH)
        assert str(MU_EARTH) in fd.description
