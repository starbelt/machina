"""
Tests for the antenna-beam factories in ``machina.astro.beam``:
``geometry.off_axis_angle``, ``cost.smooth_in_cone`` and ``cost.mean_beam_duty``.

Test classes
------------
TestOffAxisAngle     -- against numpy, on-axis / antiparallel limits, gradients
TestSmoothInCone     -- edge value, limits, mirror of cost.smooth_coverage, overflow
TestMeanBeamDuty     -- against a dense hard-indicator numpy computation, off-beam,
                        on the boresight line, gradients

The relative-motion map it composes (``transform.roe_to_rtn``) is tested in
``test_relative_motion.py`` beside this file.
"""

import math

import casadi as ca
import numpy as np
import pytest

import machina.astro  # noqa: F401
from machina.library import registry

pytestmark = pytest.mark.requires_casadi

A_GEO = 42164.0  # km


# ===========================================================================
# Helpers
# ===========================================================================

def _angle(boresight, direction):
    fd = registry.get('geometry.off_axis_angle')()
    return float(fd.function(ca.DM(boresight), ca.DM(direction)))


def _in_cone(angle, half_angle, k):
    fd = registry.get('cost.smooth_in_cone')()
    return float(fd.function(ca.DM(angle), ca.DM(half_angle), ca.DM(k)))


def _rtn_numpy(roe, lam, a=A_GEO):
    """Closed-form R, T, N (3, n) for an array of longitudes."""
    dlam, df, dg, dix, diy = roe
    c, s = np.cos(lam), np.sin(lam)
    return a * np.array([-(df * c + dg * s),
                         dlam + 2.0 * (df * s - dg * c),
                         dix * s - diy * c])


def _hard_duty(roe, boresight, half_angle, n=200_000):
    """Fraction of n uniform longitudes with the chief->deputy direction inside the cone."""
    lam = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    r = _rtn_numpy(roe, lam)
    b = np.asarray(boresight, dtype=float).reshape(3, 1)
    cross = np.cross(b, r, axis=0)
    angle = np.arctan2(np.linalg.norm(cross, axis=0), (b * r).sum(axis=0))
    return float(np.mean(angle < half_angle))


def _duty(roe, boresight, half_angle, k, n_samples=3600):
    fd = registry.get('cost.mean_beam_duty')(a=A_GEO, n_samples=n_samples)
    return float(fd.function(ca.DM(roe), ca.DM(boresight), ca.DM(half_angle), ca.DM(k)))


# A period-matched formation with parallel e/i vectors (phase 0) and no along-track
# offset: at lam = 0 the deputy sits exactly on the chief's -R axis.
PARALLEL_EI = [0.0, 1e-4, 0.0, 1e-4, 0.0]


# ===========================================================================
# geometry.off_axis_angle
# ===========================================================================

class TestOffAxisAngle:

    def test_descriptor_shapes(self):
        fd = registry.get('geometry.off_axis_angle')()
        assert fd.input_names == ['boresight', 'direction']
        assert fd.input_shapes == [(3, 1), (3, 1)]
        assert fd.output_shapes == [(1, 1)]

    def test_matches_numpy_arccos(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            b = rng.normal(size=3)
            r = rng.normal(size=3) * 5.0
            ref = math.acos(np.dot(b, r) / (np.linalg.norm(b) * np.linalg.norm(r)))
            np.testing.assert_allclose(_angle(b, r), ref, rtol=1e-12, atol=1e-12)

    def test_scale_invariant(self):
        b = np.array([1.0, 2.0, -0.5])
        r = np.array([-3.0, 0.4, 2.0])
        np.testing.assert_allclose(_angle(1e-3 * b, 4e3 * r), _angle(b, r), rtol=1e-14)

    def test_on_axis_is_zero(self):
        assert _angle([0.0, 0.0, 2.0], [0.0, 0.0, 7.0]) < 1e-15

    def test_antiparallel_is_pi(self):
        np.testing.assert_allclose(_angle([1.0, 1.0, 0.0], [-3.0, -3.0, 0.0]), math.pi,
                                   rtol=0.0, atol=1e-14)

    def test_perpendicular_is_half_pi(self):
        np.testing.assert_allclose(_angle([1.0, 0.0, 0.0], [0.0, 5.0, 0.0]), math.pi / 2,
                                   rtol=0.0, atol=1e-15)

    def test_small_angle_is_resolved(self):
        """1e-9 rad off axis: acos of a normalised dot product would return 0."""
        np.testing.assert_allclose(_angle([1.0, 0.0, 0.0], [1.0, 1e-9, 0.0]), 1e-9, rtol=1e-9)

    @pytest.mark.parametrize('b, r', [
        ([0.0, 0.0, 1.0], [0.0, 0.0, 3.0]),     # on axis
        ([1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]),    # antiparallel
        ([1.0, 0.2, 0.0], [0.3, 1.0, -0.4]),    # generic
    ])
    def test_gradient_is_finite(self, b, r):
        fd = registry.get('geometry.off_axis_angle')()
        xb = ca.SX.sym('b', 3)
        xr = ca.SX.sym('r', 3)
        angle = fd.function(xb, xr)
        grad = ca.Function('angle_gradient', [xb, xr], [ca.gradient(angle, ca.vertcat(xb, xr))])
        values = np.array(grad(ca.DM(b), ca.DM(r)))
        assert np.all(np.isfinite(values)), f"non-finite gradient {values.ravel()}"


# ===========================================================================
# cost.smooth_in_cone
# ===========================================================================

class TestSmoothInCone:

    def test_descriptor_shapes(self):
        fd = registry.get('cost.smooth_in_cone')()
        assert fd.input_names == ['angle', 'half_angle', 'k']
        assert fd.input_shapes == [(1, 1), (1, 1), (1, 1)]
        assert fd.output_names == ['in_cone']

    @pytest.mark.parametrize('k', [10.0, 1e3, 1e6])
    def test_half_on_the_edge(self, k):
        alpha = math.radians(1.2)
        np.testing.assert_allclose(_in_cone(alpha, alpha, k), 0.5, atol=1e-15)

    def test_inside_and_outside(self):
        alpha = math.radians(1.2)
        k = 1e4
        assert _in_cone(0.0, alpha, k) > 1.0 - 1e-12
        assert _in_cone(math.radians(2.0), alpha, k) < 1e-12

    def test_monotonically_decreasing(self):
        alpha = math.radians(5.0)
        values = [_in_cone(x, alpha, 50.0) for x in np.linspace(0.0, 0.3, 25)]
        assert all(v1 > v2 for v1, v2 in zip(values[:-1], values[1:]))

    def test_mirror_of_smooth_coverage(self):
        """in_cone(angle, alpha, k) = smooth_coverage(elevation=alpha, min_elevation=angle, k)."""
        coverage = registry.get('cost.smooth_coverage')()
        for angle in np.linspace(0.0, 0.5, 7):
            ref = float(coverage.function(ca.DM(0.2), ca.DM(angle), ca.DM(30.0)))
            np.testing.assert_allclose(_in_cone(angle, 0.2, 30.0), ref, rtol=1e-14, atol=0.0)

    @pytest.mark.parametrize('angle', [math.pi, 1.0, 0.0])
    def test_value_and_gradient_finite_when_exp_would_overflow(self, angle):
        """k (angle - half_angle) up to ~3e6: exp overflows without the cap."""
        fd = registry.get('cost.smooth_in_cone')()
        x = ca.SX.sym('x', 3)
        c = fd.function(x[0], x[1], x[2])
        f = ca.Function('in_cone_gradient', [x], [c, ca.gradient(c, x)])
        value, grad = f(ca.DM([angle, 0.02, 1e6]))
        assert np.all(np.isfinite(np.array(grad)))
        assert 0.0 <= float(value) <= 1.0


# ===========================================================================
# cost.mean_beam_duty
# ===========================================================================

class TestMeanBeamDuty:

    def test_descriptor_shapes(self):
        fd = registry.get('cost.mean_beam_duty')(a=A_GEO, n_samples=16)
        assert fd.input_names == ['roe', 'boresight_rtn', 'half_angle', 'k']
        assert fd.input_shapes == [(5, 1), (3, 1), (1, 1), (1, 1)]
        assert fd.output_names == ['duty']
        assert fd.output_shapes == [(1, 1)]

    @pytest.mark.parametrize('roe, boresight, half_angle_deg', [
        # nadir boresight, parallel e/i formation passing through it: duty ~0.3 %
        (PARALLEL_EI, [-1.0, 0.0, 0.0], 1.2),
        # boresight 2 deg off -R toward +T, wider cone: duty ~1.2 %
        (PARALLEL_EI, [-math.cos(0.035), math.sin(0.035), 0.0], 5.0),
    ])
    def test_matches_dense_hard_indicator(self, roe, boresight, half_angle_deg):
        """k = 1e4 1/rad (edge width ~1e-4 rad) and 3600 samples vs a 2e5-sample hard
        indicator. Sampling error is ~2/3600 per in-cone arc."""
        alpha = math.radians(half_angle_deg)
        hard = _hard_duty(roe, boresight, alpha)
        assert hard > 0.0
        np.testing.assert_allclose(_duty(roe, boresight, alpha, 1e4), hard, rtol=0.0, atol=5e-4)

    def test_far_off_beam_is_zero(self):
        """42 km ahead along-track with a ~0.4 km e/i ellipse: always ~90 deg from nadir."""
        roe = [1e-3, 1e-5, 0.0, 1e-5, 0.0]
        assert _duty(roe, [-1.0, 0.0, 0.0], math.radians(5.0), 1e3) < 1e-12

    def test_on_the_boresight_line_is_nonzero(self):
        """Boresight chosen as r_rtn at lam0 = 1 rad for a generic formation."""
        roe = np.array([2e-5, 1e-4, 3e-5, -4e-5, 1e-4])
        boresight = _rtn_numpy(roe, np.array([1.0])).ravel()
        alpha = math.radians(1.0)
        hard = _hard_duty(roe, boresight, alpha)
        duty = _duty(roe, boresight, alpha, 1e4)
        assert duty > 1e-3
        np.testing.assert_allclose(duty, hard, rtol=0.0, atol=5e-4)

    def test_sample_on_axis_has_finite_gradient(self):
        """lam = 0 is a sample and the PARALLEL_EI deputy sits exactly on -R there."""
        fd = registry.get('cost.mean_beam_duty')(a=A_GEO, n_samples=360)
        roe = ca.SX.sym('roe', 5)
        duty = fd.function(roe, ca.DM([-1.0, 0.0, 0.0]), math.radians(1.2), 1e4)
        grad = ca.Function('duty_gradient', [roe], [duty, ca.gradient(duty, roe)])
        value, gradient = grad(ca.DM(PARALLEL_EI))
        assert float(value) > 0.0
        assert np.all(np.isfinite(np.array(gradient)))

    def test_gradient_finite_far_off_beam_with_large_k(self):
        """Every sample overflows exp without the cap in cost.smooth_in_cone."""
        fd = registry.get('cost.mean_beam_duty')(a=A_GEO, n_samples=64)
        roe = ca.SX.sym('roe', 5)
        duty = fd.function(roe, ca.DM([1.0, 0.0, 0.0]), math.radians(1.0), 1e5)
        grad = ca.Function('duty_gradient', [roe], [ca.gradient(duty, roe)])
        assert np.all(np.isfinite(np.array(grad(ca.DM([1e-3, 1e-5, 0.0, 1e-5, 0.0])))))

    @pytest.mark.parametrize('n_samples', [0, -3, 2.5, True])
    def test_bad_sample_count_refused(self, n_samples):
        with pytest.raises(ValueError, match="cost.mean_beam_duty"):
            registry.get('cost.mean_beam_duty')(a=A_GEO, n_samples=n_samples)
