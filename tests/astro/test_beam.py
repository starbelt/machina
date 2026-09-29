"""
Tests for the antenna-beam factories in ``machina.astro.beam``:
``geometry.off_axis_angle``, ``cost.smooth_in_cone`` and ``cost.mean_beam_duty``.

Test classes
------------
TestOffAxisAngle     -- against numpy, on-axis / antiparallel limits, gradients
TestSmoothInCone     -- edge value, limits, mirror of cost.smooth_coverage, overflow
TestMeanBeamDuty     -- against a dense hard-indicator numpy computation, off-beam,
                        on the boresight line, gradients
TestBeamDutySampling -- beam_duty_sharpness; sub-sample phase shifts (smooth at the
                        recommended k, a staircase at k = 1e4); closed-form duty
TestNadirBound       -- duty <= atan(tan alpha / 2) / pi for a nadir boresight,
                        attained by in-plane formations
TestBeamConvention   -- chief->member direction, boresight as given in RTN
                        (-R toward Earth, T east, N north), pinned with dlam != 0

The relative-motion map it composes (``transform.roe_to_rtn``) is tested in
``test_relative_motion.py`` beside this file.
"""

import functools
import math

import casadi as ca
import numpy as np
import pytest

import machina.astro  # noqa: F401
from machina.astro.beam import beam_duty_sharpness
from machina.library import registry

pytestmark = pytest.mark.requires_casadi

A_GEO = 42164.0  # km
ALPHA = math.radians(1.2)  # downlink cone half-angle of the example [rad]
N_RULE = 7200  # >= 20 pi sqrt(5) / ALPHA = 6709, the sampling rule's minimum
NADIR = [-1.0, 0.0, 0.0]


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


@functools.cache
def _duty_function(n_samples):
    return registry.get('cost.mean_beam_duty')(a=A_GEO, n_samples=n_samples).function


def _duty(roe, boresight, half_angle, k, n_samples=3600):
    fn = _duty_function(n_samples)
    return float(fn(ca.DM(roe), ca.DM(boresight), ca.DM(half_angle), ca.DM(k)))


def _formation(phase, dlam=0.0, di_over_de=1.0, separation_km=5.0):
    """ROE with parallel e/i vectors at ``phase``, a |de| = separation_km."""
    delta = separation_km / A_GEO
    u = np.array([math.cos(phase), math.sin(phase)])
    return np.r_[dlam, delta * u, di_over_de * delta * u]


def _nadir_bound(alpha):
    """Largest hard-indicator duty of any ROE for a -R boresight."""
    return math.atan(math.tan(alpha) / 2.0) / math.pi


def _in_plane_duty(x, tilt, alpha):
    """Closed-form hard duty of an in-plane formation (di = 0, x = dlam / |de|,
    |x| < 2) for a boresight -R tilted by ``tilt`` toward +T.

    The direction's angle phi from -R toward +T has tan phi = (x + 2 sin psi) /
    cos psi, monotonic on the nadir half; tan phi = t at
    psi(t) = acos(x / sqrt(4 + t^2)) - atan2(2, t). The duty is the psi-arc
    between tan(tilt - alpha) and tan(tilt + alpha), over 2 pi.
    """
    def psi(t):
        return math.acos(x / math.sqrt(4.0 + t * t)) - math.atan2(2.0, t)
    return (psi(math.tan(tilt + alpha)) - psi(math.tan(tilt - alpha))) / (2.0 * math.pi)


def _tilted(tilt, azimuth):
    """-R tilted by ``tilt`` toward ``azimuth`` in the T-N plane (0 = +T, pi/2 = +N)."""
    return np.array([-math.cos(tilt), math.sin(tilt) * math.cos(azimuth),
                     math.sin(tilt) * math.sin(azimuth)])


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

    @pytest.mark.parametrize('n_samples', [0, -3, 2.5, True, math.nan, math.inf, '12'])
    def test_bad_sample_count_refused(self, n_samples):
        with pytest.raises(ValueError, match="cost.mean_beam_duty"):
            registry.get('cost.mean_beam_duty')(a=A_GEO, n_samples=n_samples)


# ===========================================================================
# Sampling rule: beam_duty_sharpness and the staircase it avoids
# ===========================================================================

class TestBeamDutySampling:
    """k = n / (2 pi sqrt 5) puts one sample across the sigmoid edge. The
    reference formation is the example's: parallel |de| = |di|, a |de| = 5 km,
    phase 30 deg, dlam = 0, nadir boresight, alpha = 1.2 deg, whose exact duty is
    atan(tan alpha / sqrt 5) / pi = 0.2982 %."""

    PHASE = math.radians(30.0)

    def test_sharpness_value(self):
        np.testing.assert_allclose(beam_duty_sharpness(3600),
                                   3600 / (2.0 * math.pi * math.sqrt(5.0)), rtol=1e-15)
        np.testing.assert_allclose(beam_duty_sharpness(7200), 512.47, rtol=1e-4)

    @pytest.mark.parametrize('n_samples', [0, -1, 2.5, True, math.nan, math.inf, None])
    def test_sharpness_refuses_bad_sample_count(self, n_samples):
        with pytest.raises(ValueError, match="beam_duty_sharpness"):
            beam_duty_sharpness(n_samples)

    def _shifted(self, n_samples, k, fraction):
        """Duty with the formation rotated by ``fraction`` of one sample step (a
        pure time shift for dlam = 0: the continuous duty does not change)."""
        step = 2.0 * math.pi / n_samples
        return _duty(_formation(self.PHASE + fraction * step), NADIR, ALPHA, k, n_samples)

    @pytest.mark.parametrize('n_samples', [3600, N_RULE])
    def test_sub_sample_shift_is_smooth_at_recommended_k(self, n_samples):
        """Measured: 1.1e-4 relative at n = 3600, 2.4e-7 at n = 7200."""
        k = beam_duty_sharpness(n_samples)
        base = self._shifted(n_samples, k, 0.0)
        for fraction in (0.25, 0.5):
            change = abs(self._shifted(n_samples, k, fraction) / base - 1.0)
            assert change < 1e-2, f"{fraction} sample shift moved the duty by {change:.2%}"

    @pytest.mark.parametrize('n_samples', [3600, N_RULE])
    def test_sub_sample_shift_is_a_staircase_at_large_k(self, n_samples):
        """The failure mode the rule avoids: at k = 1e4 the edge (1e-4 rad) is 20-40x
        narrower than a sample, so the duty moves in steps of 1/n_samples (9.0 % at
        n = 3600, 4.6 % at 7200 under a half-sample shift)."""
        base = self._shifted(n_samples, 1e4, 0.0)
        changes = [abs(self._shifted(n_samples, 1e4, f) / base - 1.0) for f in (0.25, 0.5)]
        assert max(changes) > 3e-2, f"k = 1e4 changes {changes} look smooth"

    @pytest.mark.parametrize('n_samples', [3600, N_RULE])
    def test_parallel_formation_matches_closed_form(self, n_samples):
        """Measured: +9.8e-4 relative at n = 3600 (k alpha = 5.4), +1.2e-5 at 7200."""
        exact = math.atan(math.tan(ALPHA) / math.sqrt(5.0)) / math.pi
        duty = _duty(_formation(self.PHASE), NADIR, ALPHA, beam_duty_sharpness(n_samples),
                     n_samples)
        np.testing.assert_allclose(duty, exact, rtol=1e-2)
        np.testing.assert_allclose(exact, 0.002982, rtol=1e-3)


# ===========================================================================
# Nadir bound: duty <= atan(tan alpha / 2) / pi
# ===========================================================================

class TestNadirBound:
    """For a -R boresight no ROE beats the in-plane formation (di = 0), whose
    duty is atan(tan alpha / 2) / pi for every |dlam| < 2 |de|: N only adds to
    the off-axis angle, and the in-plane in-cone arc has the same length for
    every along-track offset (derivation in make_mean_beam_duty)."""

    def test_bound_at_one_point_two_degrees(self):
        np.testing.assert_allclose(_nadir_bound(ALPHA), 0.0033337, rtol=1e-4)

    @pytest.mark.parametrize('x', [0.0, 0.8, -1.5, 1.9])
    def test_in_plane_formation_attains_the_bound(self, x):
        """x = dlam / |de|; measured within 1.2e-5 relative at n = 7200."""
        roe = _formation(0.4, di_over_de=0.0)
        roe[0] = x * np.linalg.norm(roe[1:3])
        duty = _duty(roe, NADIR, ALPHA, beam_duty_sharpness(N_RULE), N_RULE)
        np.testing.assert_allclose(duty, _nadir_bound(ALPHA), rtol=1e-2)

    def test_random_formations_stay_below_the_bound(self):
        """300 random ROE (di from 0 to 3 |de|, |dlam| up to 2.5 |de|). The smooth
        duty exceeds the hard bound only by the sigmoid bias (measured max
        1.00001 x bound at n = 7200); the planar draws come within a few percent."""
        rng = np.random.default_rng(0)
        k = beam_duty_sharpness(N_RULE)
        delta = 5.0 / A_GEO
        ratios = []
        for _ in range(300):
            de = rng.normal(size=2)
            di = rng.normal(size=2) * rng.choice([0.0, 0.05, 0.3, 1.0, 3.0])
            dlam = rng.uniform(-2.5, 2.5) * np.linalg.norm(de)
            roe = np.r_[dlam, de, di] * delta
            ratios.append(_duty(roe, NADIR, ALPHA, k, N_RULE) / _nadir_bound(ALPHA))
        assert max(ratios) <= 1.0 + 1e-3, f"duty / bound reached {max(ratios):.6f}"
        assert max(ratios) > 0.99, "no draw came near the bound; the test lost its teeth"


# ===========================================================================
# Beam convention: chief->member direction, boresight in RTN as given
# ===========================================================================

class TestBeamConvention:
    """For dlam = 0 the relative orbit is centred on the chief, r(lam + pi) =
    -r(lam), so a flipped boresight (+R instead of -R) or a flipped direction
    (member -> chief) gives the same duty. An along-track offset breaks the
    symmetry. Convention: the direction is chief -> member (``transform.roe_to_rtn``),
    the boresight is used as given in RTN: -R toward the Earth, +T east, +N north
    for a prograde chief."""

    def test_centred_formation_cannot_tell_the_conventions_apart(self):
        """Why the tests below need dlam != 0: with the boresight on the path at
        lam = 1 rad, b and -b give the same (non-zero) duty."""
        roe = _formation(0.7, di_over_de=0.4)
        b = _rtn_numpy(roe, np.array([1.0])).ravel()
        k = beam_duty_sharpness(N_RULE)
        duty = _duty(roe, b, ALPHA, k, N_RULE)
        assert duty > 1e-3
        np.testing.assert_allclose(_duty(roe, -b, ALPHA, k, N_RULE), duty, rtol=1e-9)

    def test_east_tilt_with_along_track_offset_matches_closed_form(self):
        """In-plane formation, member on average 1.5 |de| east (dlam > 0), boresight
        -R tilted 20 deg east: 0.438 %. The flipped convention (boresight -b, the
        same as the member -> chief direction) sees the mirror case x -> -x:
        0.293 %."""
        x, tilt = 1.5, math.radians(20.0)
        roe = _formation(0.7, di_over_de=0.0)
        roe[0] = x * np.linalg.norm(roe[1:3])
        b = _tilted(tilt, 0.0)
        k = beam_duty_sharpness(N_RULE)
        correct = _in_plane_duty(x, tilt, ALPHA)
        flipped = _in_plane_duty(-x, tilt, ALPHA)
        assert correct > 1.4 * flipped
        np.testing.assert_allclose(_duty(roe, b, ALPHA, k, N_RULE), correct, rtol=1e-2)
        np.testing.assert_allclose(_duty(roe, -b, ALPHA, k, N_RULE), flipped, rtol=1e-2)

    def test_every_flip_loses_a_member_below_east_and_north(self):
        """a * roe = [2, 5, 0.5, 2, -1] km puts the member at (R, T, N) = (-5, 1, 1) km
        at lam = 0: below, east and north of the chief. A boresight aimed there
        (15.8 deg off nadir) holds it 0.353 % of the orbit; flipping the boresight,
        or only its R, T or N component, never sees it."""
        roe = np.array([2.0, 5.0, 0.5, 2.0, -1.0]) / A_GEO
        r0 = _rtn_numpy(roe, np.array([0.0])).ravel()
        np.testing.assert_allclose(r0, [-5.0, 1.0, 1.0], rtol=1e-12)
        b = r0 / np.linalg.norm(r0)
        k = beam_duty_sharpness(N_RULE)
        correct = _duty(roe, b, ALPHA, k, N_RULE)
        np.testing.assert_allclose(correct, _hard_duty(roe, b, ALPHA, n=400_000), rtol=1e-2)
        assert correct > 3e-3
        for label, flipped in (('-b (member -> chief)', -b),
                               ('+R', b * [-1.0, 1.0, 1.0]),
                               ('west', b * [1.0, -1.0, 1.0]),
                               ('south', b * [1.0, 1.0, -1.0])):
            assert _duty(roe, flipped, ALPHA, k, N_RULE) < 1e-6, f"{label} still sees it"
